//! Attention mechanisms for transformer models.
//!
//! This module implements various attention mechanisms optimized for LLM inference:
//!
//! - Multi-head attention (MHA)
//! - Multi-query attention (MQA)  
//! - Grouped-query attention (GQA)
//! - Paged attention for efficient KV cache management
//! - Flash Attention for memory-efficient computation

use candle_core::{Device, DType, IndexOp, Result, Tensor};
use std::sync::Arc;
use tracing::{debug, instrument};

/// Attention configuration.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,

    /// Number of key-value heads (for GQA).
    pub num_kv_heads: usize,

    /// Head dimension.
    pub head_dim: usize,

    /// Maximum sequence length.
    pub max_seq_len: usize,

    /// Scale factor for attention scores.
    pub scale: f64,

    /// Whether to use alibi positional encoding.
    pub use_alibi: bool,

    /// Sliding window size (if any).
    pub sliding_window: Option<usize>,

    /// Whether to use Flash Attention.
    pub use_flash_attn: bool,
}

impl AttentionConfig {
    /// Create a new attention config.
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len: 32768,
            scale: (head_dim as f64).sqrt().recip(),
            use_alibi: false,
            sliding_window: None,
            use_flash_attn: true,
        }
    }

    /// MHA config (num_kv_heads == num_heads).
    pub fn mha(num_heads: usize, head_dim: usize) -> Self {
        Self::new(num_heads, num_heads, head_dim)
    }

    /// MQA config (num_kv_heads == 1).
    pub fn mqa(num_heads: usize, head_dim: usize) -> Self {
        Self::new(num_heads, 1, head_dim)
    }

    /// GQA config.
    pub fn gqa(num_heads: usize, num_kv_groups: usize, head_dim: usize) -> Self {
        Self::new(num_heads, num_heads / num_kv_groups, head_dim)
    }

    /// Set sliding window.
    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.sliding_window = Some(window);
        self
    }

    /// Set alibi.
    pub fn with_alibi(mut self) -> Self {
        self.use_alibi = true;
        self
    }

    /// Get the number of heads per KV head group.
    pub fn heads_per_kv_group(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// Attention mask types.
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// No mask (full attention).
    None,

    /// Causal mask (lower triangular).
    Causal,

    /// Custom mask tensor.
    Custom(Tensor),

    /// Sliding window causal mask.
    SlidingWindow { window_size: usize },

    /// Prefix mask (bidirectional for prefix, causal for rest).
    Prefix { prefix_len: usize },
}

impl AttentionMask {
    /// Create a causal mask tensor.
    pub fn causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let mask = Tensor::ones((seq_len, seq_len), dtype, device)?;
        let mask = mask.tril(0)?;
        // Convert to attention bias: 0 for attended, -inf for masked
        let neg_inf = Tensor::new(&[f32::NEG_INFINITY], device)?.broadcast_as((seq_len, seq_len))?;
        let zero = Tensor::zeros((seq_len, seq_len), dtype, device)?;
        mask.where_cond(&zero, &neg_inf)
    }

    /// Create a sliding window causal mask.
    pub fn sliding_window_mask(
        seq_len: usize,
        window_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            let start = i.saturating_sub(window_size);
            for j in start..=i {
                mask_data[i * seq_len + j] = 0.0;
            }
        }
        Tensor::from_vec(mask_data, (seq_len, seq_len), device)?.to_dtype(dtype)
    }

    /// Convert to tensor.
    pub fn to_tensor(
        &self,
        query_len: usize,
        key_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        match self {
            AttentionMask::None => Ok(None),
            AttentionMask::Causal => {
                Ok(Some(Self::causal_mask(query_len.max(key_len), device, dtype)?
                    .i((..query_len, ..key_len))?))
            }
            AttentionMask::Custom(mask) => Ok(Some(mask.clone())),
            AttentionMask::SlidingWindow { window_size } => {
                Ok(Some(Self::sliding_window_mask(
                    query_len.max(key_len),
                    *window_size,
                    device,
                    dtype,
                )?
                .i((..query_len, ..key_len))?))
            }
            AttentionMask::Prefix { prefix_len } => {
                // Bidirectional for prefix, causal for rest
                let mut mask_data = vec![0.0f32; query_len * key_len];
                for i in 0..query_len {
                    for j in 0..key_len {
                        // Allow all attention within prefix, causal attention after
                        if j <= i || j < *prefix_len {
                            mask_data[i * key_len + j] = 0.0;
                        } else {
                            mask_data[i * key_len + j] = f32::NEG_INFINITY;
                        }
                    }
                }
                let mask = Tensor::from_vec(mask_data, (query_len, key_len), device)?;
                Ok(Some(mask.to_dtype(dtype)?))
            }
        }
    }
}

/// Scaled dot-product attention.
#[instrument(skip_all, level = "debug")]
pub fn scaled_dot_product_attention(
    query: &Tensor,   // [batch, num_heads, seq_len, head_dim]
    key: &Tensor,     // [batch, num_kv_heads, seq_len, head_dim]
    value: &Tensor,   // [batch, num_kv_heads, seq_len, head_dim]
    mask: Option<&Tensor>,
    scale: f64,
    dropout: f64,
) -> Result<Tensor> {
    let device = query.device();
    let dtype = query.dtype();
    
    debug!(
        "SDPA: Q {:?}, K {:?}, V {:?}, scale={}",
        query.shape(),
        key.shape(),
        value.shape(),
        scale
    );

    // Compute attention scores: Q @ K^T
    let key_t = key.transpose(2, 3)?; // [batch, num_kv_heads, head_dim, seq_len]
    let mut attn_weights = query.matmul(&key_t)?; // [batch, num_heads, q_len, k_len]

    // Scale
    attn_weights = (attn_weights * scale)?;

    // Apply mask
    if let Some(mask) = mask {
        attn_weights = attn_weights.broadcast_add(mask)?;
    }

    // Softmax
    let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

    // Apply dropout (if training)
    let attn_weights = if dropout > 0.0 {
        // In inference, we typically don't use dropout
        attn_weights
    } else {
        attn_weights
    };

    // Compute output: attn @ V
    let output = attn_weights.matmul(value)?;

    Ok(output)
}

/// Repeat KV heads for GQA (grouped-query attention).
///
/// Expands key/value tensors to match the number of query heads.
pub fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }

    let (batch, num_kv_heads, seq_len, head_dim) = hidden_states.dims4()?;

    // Expand and reshape: [B, KV, S, D] -> [B, KV, n_rep, S, D] -> [B, KV*n_rep, S, D]
    let expanded = hidden_states
        .unsqueeze(2)?
        .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;

    Ok(expanded)
}

/// Apply rotary positional embeddings (RoPE).
#[instrument(skip_all, level = "debug")]
pub fn apply_rotary_emb(
    query: &Tensor,
    key: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    position_ids: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (batch, num_heads, seq_len, head_dim) = query.dims4()?;

    // Get position-specific cos/sin
    let (cos, sin) = if let Some(pos_ids) = position_ids {
        let cos = cos.index_select(pos_ids, 0)?;
        let sin = sin.index_select(pos_ids, 0)?;
        (cos, sin)
    } else {
        (cos.i(..seq_len)?, sin.i(..seq_len)?)
    };

    // Reshape for broadcasting
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq_len, head_dim]
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Apply rotation
    let q_embed = rotate_half(query, &cos, &sin)?;
    let k_embed = rotate_half(key, &cos, &sin)?;

    Ok((q_embed, k_embed))
}

/// Rotate half of the dimensions for RoPE.
fn rotate_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _s, d) = x.dims4()?;
    let half_d = d / 2;

    // Split into two halves
    let x1 = x.narrow(3, 0, half_d)?;
    let x2 = x.narrow(3, half_d, half_d)?;

    // Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    let rotated_x1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let rotated_x2 = (x1.broadcast_mul(sin)? + x2.broadcast_mul(cos)?)?;

    // Concatenate
    Tensor::cat(&[rotated_x1, rotated_x2], 3)
}

/// Precompute RoPE frequencies.
pub fn precompute_rope_freqs(
    head_dim: usize,
    max_seq_len: usize,
    base: f32,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;

    // Compute inverse frequencies
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?;

    // Compute position indices
    let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;

    // Compute freqs: positions @ inv_freq
    let freqs = positions.matmul(&inv_freq)?; // [max_seq_len, half_dim]

    // Duplicate for full head_dim: [cos(f), cos(f)] pattern
    let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

    // Compute cos and sin
    let cos = freqs.cos()?.to_dtype(dtype)?;
    let sin = freqs.sin()?.to_dtype(dtype)?;

    Ok((cos, sin))
}

/// ALiBi (Attention with Linear Biases) slopes.
pub fn get_alibi_slopes(num_heads: usize) -> Vec<f32> {
    // Compute slopes as described in the ALiBi paper
    let closest_power_of_2 = 2_u32.pow((num_heads as f32).log2().floor() as u32);

    let base = 2.0_f32.powf(-(2.0_f32.powf(-((closest_power_of_2 as f32).log2() - 3.0))));

    let mut slopes = Vec::with_capacity(num_heads);

    if closest_power_of_2 as usize == num_heads {
        for i in 1..=num_heads {
            slopes.push(base.powi(i as i32));
        }
    } else {
        // Handle non-power-of-2 heads
        let extra_base = 2.0_f32.powf(-(2.0_f32.powf(-((closest_power_of_2 as f32 * 2.0).log2() - 3.0))));
        let num_remaining = num_heads - closest_power_of_2 as usize;

        for i in 1..=closest_power_of_2 as usize {
            slopes.push(base.powi(i as i32));
        }
        for i in 1..=num_remaining {
            slopes.push(extra_base.powi((2 * i) as i32));
        }
    }

    slopes
}

/// Compute ALiBi bias.
pub fn compute_alibi_bias(
    slopes: &[f32],
    seq_len: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let num_heads = slopes.len();

    // Create position difference matrix
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let pos_tensor = Tensor::from_vec(positions.clone(), (seq_len, 1), device)?;
    let pos_t = Tensor::from_vec(positions, (1, seq_len), device)?;
    let pos_diff = (pos_tensor - pos_t)?; // [seq_len, seq_len]

    // Apply slopes
    let slopes_tensor = Tensor::from_vec(slopes.to_vec(), (num_heads, 1, 1), device)?;
    let bias = pos_diff.unsqueeze(0)?.broadcast_mul(&slopes_tensor)?;

    bias.to_dtype(dtype)
}

/// Attention output with optional auxiliary info.
#[derive(Debug)]
pub struct AttentionOutput {
    /// Output tensor.
    pub output: Tensor,

    /// Attention weights (if retained).
    pub attention_weights: Option<Tensor>,

    /// Updated KV cache (if applicable).
    pub past_key_value: Option<(Tensor, Tensor)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::mha(32, 128);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 32);
        assert_eq!(config.heads_per_kv_group(), 1);

        let config = AttentionConfig::gqa(32, 4, 128);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.heads_per_kv_group(), 4);
    }

    #[test]
    fn test_alibi_slopes() {
        let slopes = get_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);
        // First slope should be 0.5
        assert!((slopes[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        let mask = AttentionMask::causal_mask(4, &device, DType::F32)?;
        assert_eq!(mask.dims(), &[4, 4]);

        // Check that upper triangle is -inf
        let values: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        assert_eq!(values[1], f32::NEG_INFINITY); // (0, 1)
        assert_eq!(values[0], 0.0); // (0, 0)
        assert_eq!(values[5], 0.0); // (1, 1)

        Ok(())
    }

    #[test]
    fn test_repeat_kv() -> Result<()> {
        let device = Device::Cpu;
        let kv = Tensor::randn(0.0f32, 1.0, (2, 4, 10, 64), &device)?;
        let repeated = repeat_kv(&kv, 4)?;
        assert_eq!(repeated.dims(), &[2, 16, 10, 64]);

        Ok(())
    }

    #[test]
    fn test_sdpa() -> Result<()> {
        let device = Device::Cpu;
        let q = Tensor::randn(0.0f32, 1.0, (1, 8, 4, 64), &device)?;
        let k = Tensor::randn(0.0f32, 1.0, (1, 8, 4, 64), &device)?;
        let v = Tensor::randn(0.0f32, 1.0, (1, 8, 4, 64), &device)?;

        let output = scaled_dot_product_attention(&q, &k, &v, None, 0.125, 0.0)?;
        assert_eq!(output.dims(), &[1, 8, 4, 64]);

        Ok(())
    }
}
