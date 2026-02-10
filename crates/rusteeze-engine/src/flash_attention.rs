//! # Flash Attention v2 — Radical Rewrite
//!
//! ## Breakthrough Innovations (vs. original)
//!
//! 1. **Rayon Parallelism**: All (batch, head) pairs processed in parallel
//! 2. **Zero-Allocation Hot Path**: Thread-local scratch buffers, no Vec in inner loops
//! 3. **Compile-Time SIMD Dispatch**: Function pointers resolved once at startup
//! 4. **Cache-Aware Tiling**: Block sizes auto-tuned to L1/L2 cache hierarchy
//! 5. **Software Prefetching**: Prefetch next KV blocks while processing current
//! 6. **Online Softmax**: Numerically stable single-pass with running max/sum
//! 7. **Grouped-Query Attention (GQA)**: Fused with flash attention, not a separate path

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::simd_dispatch::{self, dot_product, fused_mul_add, prefetch_read, vec_max, vec_scale, with_scratch_a};

/// Configuration for Flash Attention.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Number of attention heads for queries
    pub num_heads: usize,
    /// Number of KV heads (for GQA: num_heads / num_kv_heads = group_size)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Whether to apply causal masking
    pub causal: bool,
    /// Dropout probability (0.0 during inference)
    pub dropout_prob: f32,
    /// Override Q block size (0 = auto-tune from cache hierarchy)
    pub q_block_size: usize,
    /// Override KV block size (0 = auto-tune from cache hierarchy)
    pub kv_block_size: usize,
    /// Softmax scale (0.0 = use 1/sqrt(head_dim))
    pub softmax_scale: f32,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            causal: true,
            dropout_prob: 0.0,
            q_block_size: 0,
            kv_block_size: 0,
            softmax_scale: 0.0,
        }
    }
}

/// Statistics for attention computation.
#[derive(Debug, Default)]
pub struct FlashAttentionStats {
    /// Total forward calls
    pub total_calls: AtomicU64,
    /// Total tokens processed
    pub total_tokens_processed: AtomicU64,
    /// Total FLOPs
    pub total_flops: AtomicU64,
}

impl Clone for FlashAttentionStats {
    fn clone(&self) -> Self {
        Self {
            total_calls: AtomicU64::new(self.total_calls.load(Ordering::Relaxed)),
            total_tokens_processed: AtomicU64::new(self.total_tokens_processed.load(Ordering::Relaxed)),
            total_flops: AtomicU64::new(self.total_flops.load(Ordering::Relaxed)),
        }
    }
}

impl FlashAttentionStats {
    fn record(&self, seq_len: usize, head_dim: usize, total_heads: usize) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.total_tokens_processed.fetch_add(seq_len as u64, Ordering::Relaxed);
        let flops = 2 * seq_len * seq_len * head_dim * total_heads;
        self.total_flops.fetch_add(flops as u64, Ordering::Relaxed);
    }
}

/// Flash Attention v2 engine — the heart of inference latency.
pub struct FlashAttention {
    config: FlashAttentionConfig,
    scale: f32,
    q_blk: usize,
    kv_blk: usize,
    gqa_group: usize,
    stats: FlashAttentionStats,
}

impl FlashAttention {
    /// Create a new FlashAttention engine with auto-tuned parameters.
    pub fn new(config: FlashAttentionConfig) -> Self {
        simd_dispatch::init();
        let caps = simd_dispatch::simd();

        let scale = if config.softmax_scale > 0.0 {
            config.softmax_scale
        } else {
            1.0 / (config.head_dim as f32).sqrt()
        };

        let float_size = std::mem::size_of::<f32>();
        let hd = config.head_dim;

        let q_blk = if config.q_block_size > 0 {
            config.q_block_size
        } else {
            let l1_q_budget = caps.cache.l1d_size * 2 / 5 / float_size;
            (l1_q_budget / hd.max(1)).max(1).min(128)
        };

        let kv_blk = if config.kv_block_size > 0 {
            config.kv_block_size
        } else {
            let l2_kv_budget = caps.cache.l2_size * 3 / 5 / float_size;
            (l2_kv_budget / (2 * hd).max(1)).max(1).min(256)
        };

        let gqa_group = config.num_heads / config.num_kv_heads.max(1);

        tracing::info!(
            q_block = q_blk, kv_block = kv_blk, head_dim = hd,
            num_heads = config.num_heads, num_kv_heads = config.num_kv_heads,
            gqa_group, scale, "FlashAttention v2 initialized"
        );

        Self { config, scale, q_blk, kv_blk, gqa_group, stats: FlashAttentionStats::default() }
    }

    /// Compute attention for a batch of sequences.
    ///
    /// Layout: `[batch_size, num_heads, seq_len, head_dim]` flattened.
    /// K/V layout: `[batch_size, num_kv_heads, seq_len, head_dim]` flattened.
    /// Fully parallel across (batch, head) pairs via rayon.
    pub fn forward(
        &self, q: &[f32], k: &[f32], v: &[f32],
        batch_size: usize, seq_len: usize, output: &mut [f32],
    ) {
        let hd = self.config.head_dim;
        let nh = self.config.num_heads;
        let nkv = self.config.num_kv_heads;

        let q_head_stride = seq_len * hd;
        let q_batch_stride = nh * q_head_stride;
        let kv_head_stride = seq_len * hd;
        let kv_batch_stride = nkv * kv_head_stride;

        let work_items: Vec<(usize, usize)> = (0..batch_size)
            .flat_map(|b| (0..nh).map(move |h| (b, h)))
            .collect();

        self.stats.record(seq_len, hd, nh * batch_size);

        // Parallel over all (batch, head) pairs
        let o_ptr = output.as_ptr() as usize; // Send as usize for Send
        let o_len = output.len();

        work_items.par_iter().for_each(|&(b, h)| {
            let kv_h = h / self.gqa_group;
            let q_offset = b * q_batch_stride + h * q_head_stride;
            let k_offset = b * kv_batch_stride + kv_h * kv_head_stride;
            let v_offset = k_offset;
            let o_offset = b * q_batch_stride + h * q_head_stride;

            let q_slice = &q[q_offset..q_offset + q_head_stride];
            let k_slice = &k[k_offset..k_offset + kv_head_stride];
            let v_slice = &v[v_offset..v_offset + kv_head_stride];

            // SAFETY: Each (b, h) writes to a disjoint region of output
            let o_slice = unsafe {
                let ptr = o_ptr as *mut f32;
                debug_assert!(o_offset + q_head_stride <= o_len);
                std::slice::from_raw_parts_mut(ptr.add(o_offset), q_head_stride)
            };

            self.compute_attention_tiled(q_slice, k_slice, v_slice, o_slice, seq_len);
        });
    }

    /// Tiled attention for a single (batch, head) pair.
    /// Uses online softmax, zero allocations via thread-local scratch.
    fn compute_attention_tiled(
        &self, q: &[f32], k: &[f32], v: &[f32],
        output: &mut [f32], seq_len: usize,
    ) {
        let hd = self.config.head_dim;
        let q_blk = self.q_blk.min(seq_len);
        let kv_blk = self.kv_blk.min(seq_len);

        let scores_size = q_blk * kv_blk;
        let total_scratch = scores_size + 2 * q_blk; // scores + row_max + row_sum

        for x in output.iter_mut() { *x = 0.0; }

        with_scratch_a(total_scratch, |scratch| {
            let (scores, rest) = scratch.split_at_mut(scores_size);
            let (row_max, row_sum) = rest.split_at_mut(q_blk);

            for q_start in (0..seq_len).step_by(q_blk) {
                let q_end = (q_start + q_blk).min(seq_len);
                let q_rows = q_end - q_start;

                let rm = &mut row_max[..q_rows];
                let rs = &mut row_sum[..q_rows];
                for x in rm.iter_mut() { *x = f32::NEG_INFINITY; }
                for x in rs.iter_mut() { *x = 0.0; }
                for x in output[q_start * hd..q_end * hd].iter_mut() { *x = 0.0; }

                let kv_limit = if self.config.causal { q_end } else { seq_len };
                for kv_start in (0..kv_limit).step_by(kv_blk) {
                    let kv_end = (kv_start + kv_blk).min(kv_limit);
                    let kv_rows = kv_end - kv_start;

                    if kv_end < kv_limit {
                        let next = kv_end * hd;
                        if next < k.len() { prefetch_read(k[next..].as_ptr()); }
                    }

                    self.compute_scores(
                        &q[q_start * hd..q_end * hd],
                        &k[kv_start * hd..kv_end * hd],
                        &mut scores[..q_rows * kv_rows],
                        q_rows, kv_rows, hd,
                    );

                    if self.config.causal {
                        Self::apply_causal_mask(
                            &mut scores[..q_rows * kv_rows],
                            q_rows, kv_rows, q_start, kv_start,
                        );
                    }

                    self.online_softmax_update(
                        &scores[..q_rows * kv_rows],
                        &v[kv_start * hd..kv_end * hd],
                        &mut output[q_start * hd..q_end * hd],
                        rm, rs, q_rows, kv_rows, hd,
                    );
                }

                for qi in 0..q_rows {
                    let inv = if rs[qi] > 0.0 { 1.0 / rs[qi] } else { 0.0 };
                    let row = &mut output[(q_start + qi) * hd..(q_start + qi + 1) * hd];
                    vec_scale(row, inv);
                }
            }
        });
    }

    #[inline]
    fn compute_scores(
        &self, q: &[f32], k: &[f32], scores: &mut [f32],
        q_rows: usize, kv_rows: usize, hd: usize,
    ) {
        let scale = self.scale;
        for qi in 0..q_rows {
            let q_row = &q[qi * hd..(qi + 1) * hd];
            if qi + 1 < q_rows { prefetch_read(q[(qi + 1) * hd..].as_ptr()); }
            for ki in 0..kv_rows {
                let k_row = &k[ki * hd..(ki + 1) * hd];
                scores[qi * kv_rows + ki] = dot_product(q_row, k_row) * scale;
            }
        }
    }

    #[inline]
    fn apply_causal_mask(
        scores: &mut [f32], q_rows: usize, kv_rows: usize,
        q_start: usize, kv_start: usize,
    ) {
        for qi in 0..q_rows {
            let gq = q_start + qi;
            for ki in 0..kv_rows {
                if kv_start + ki > gq {
                    scores[qi * kv_rows + ki] = f32::NEG_INFINITY;
                }
            }
        }
    }

    #[inline]
    fn online_softmax_update(
        &self, scores: &[f32], v: &[f32], output: &mut [f32],
        row_max: &mut [f32], row_sum: &mut [f32],
        q_rows: usize, kv_rows: usize, hd: usize,
    ) {
        for qi in 0..q_rows {
            let score_row = &scores[qi * kv_rows..(qi + 1) * kv_rows];
            let new_max = vec_max(score_row).max(row_max[qi]);
            let correction = if row_max[qi] > f32::NEG_INFINITY {
                (row_max[qi] - new_max).exp()
            } else { 0.0 };

            let out_row = &mut output[qi * hd..(qi + 1) * hd];
            if correction > 0.0 && correction < 1.0 {
                vec_scale(out_row, correction);
                row_sum[qi] *= correction;
            } else if correction == 0.0 {
                for x in out_row.iter_mut() { *x = 0.0; }
                row_sum[qi] = 0.0;
            }

            let mut local_sum = 0.0f32;
            for ki in 0..kv_rows {
                let w = (score_row[ki] - new_max).exp();
                if w > 0.0 {
                    local_sum += w;
                    fused_mul_add(out_row, &v[ki * hd..(ki + 1) * hd], w);
                }
            }
            row_sum[qi] += local_sum;
            row_max[qi] = new_max;
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &FlashAttentionStats { &self.stats }
    /// Get configuration.
    pub fn config(&self) -> &FlashAttentionConfig { &self.config }
}

/// Grouped-Query Attention — thin wrapper around FlashAttention.
pub struct GroupedQueryAttention {
    flash: FlashAttention,
}

impl GroupedQueryAttention {
    /// Create GQA with specified group size.
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, causal: bool) -> Self {
        Self { flash: FlashAttention::new(FlashAttentionConfig {
            num_heads, num_kv_heads, head_dim, causal, ..Default::default()
        }) }
    }

    /// Forward pass — delegates to FlashAttention which handles GQA natively.
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], batch_size: usize, seq_len: usize, output: &mut [f32]) {
        self.flash.forward(q, k, v, batch_size, seq_len, output);
    }
}

/// Multi-Query Attention — GQA with num_kv_heads = 1.
pub struct MultiQueryAttention {
    flash: FlashAttention,
}

impl MultiQueryAttention {
    /// Create MQA.
    pub fn new(num_heads: usize, head_dim: usize, causal: bool) -> Self {
        Self { flash: FlashAttention::new(FlashAttentionConfig {
            num_heads, num_kv_heads: 1, head_dim, causal, ..Default::default()
        }) }
    }
    /// Forward pass.
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32], batch_size: usize, seq_len: usize, output: &mut [f32]) {
        self.flash.forward(q, k, v, batch_size, seq_len, output);
    }
}

/// Linear projection: output = input × weight^T + bias (parallel over batch).
pub fn linear_projection(
    input: &[f32], weight: &[f32], bias: Option<&[f32]>,
    batch_size: usize, in_features: usize, out_features: usize,
    output: &mut [f32],
) {
    output.par_chunks_mut(out_features).enumerate().for_each(|(b, out_row)| {
        let inp = &input[b * in_features..(b + 1) * in_features];
        for o in 0..out_features {
            out_row[o] = dot_product(inp, &weight[o * in_features..(o + 1) * in_features]);
        }
        if let Some(bias) = bias {
            simd_dispatch::vec_add(out_row, &bias[..out_features]);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attention_basic() {
        let config = FlashAttentionConfig {
            num_heads: 2, num_kv_heads: 2, head_dim: 4,
            causal: false, ..Default::default()
        };
        let attn = FlashAttention::new(config);
        let (batch, seq, hd, nh) = (1, 3, 4, 2);
        let total = batch * nh * seq * hd;
        let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..total).map(|i| ((total - i) as f32) * 0.1).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.05).collect();
        let mut output = vec![0.0f32; total];
        attn.forward(&q, &k, &v, batch, seq, &mut output);
        assert!(output.iter().all(|x| x.is_finite()));
        assert!(output.iter().any(|x| *x != 0.0));
    }

    #[test]
    fn test_flash_attention_causal() {
        let attn = FlashAttention::new(FlashAttentionConfig {
            num_heads: 1, num_kv_heads: 1, head_dim: 4,
            causal: true, ..Default::default()
        });
        let (batch, seq, hd) = (1, 4, 4);
        let total = batch * seq * hd;
        let q = vec![1.0f32; total];
        let k = vec![1.0f32; total];
        let v: Vec<f32> = (0..total).map(|i| (i / hd) as f32).collect();
        let mut output = vec![0.0f32; total];
        attn.forward(&q, &k, &v, batch, seq, &mut output);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_gqa() {
        let gqa = GroupedQueryAttention::new(8, 2, 64, true);
        let (batch, seq, hd, nh, nkv) = (2, 16, 64, 8, 2);
        let q = vec![0.1f32; batch * nh * seq * hd];
        let k = vec![0.1f32; batch * nkv * seq * hd];
        let v = vec![0.1f32; batch * nkv * seq * hd];
        let mut output = vec![0.0f32; batch * nh * seq * hd];
        gqa.forward(&q, &k, &v, batch, seq, &mut output);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_linear_projection() {
        let (batch, inf, outf) = (2, 4, 3);
        let input = vec![1.0f32; batch * inf];
        let weight = vec![1.0f32; outf * inf];
        let bias = vec![0.5f32; outf];
        let mut output = vec![0.0f32; batch * outf];
        linear_projection(&input, &weight, Some(&bias), batch, inf, outf, &mut output);
        for &v in &output { assert!((v - 4.5).abs() < 1e-5); }
    }
}
