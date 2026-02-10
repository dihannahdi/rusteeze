//! # Kernel Fusion — Radical Rewrite
//!
//! Fused operations that eliminate intermediate memory traffic:
//! 1. **LayerNorm + Linear**: Single pass over data
//! 2. **GELU + MLP**: Fused activation with linear
//! 3. **RMSNorm + Residual**: Single read-write pass
//!
//! All operations: rayon-parallel over batch, zero-allocation, pre-resolved SIMD.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::simd_dispatch::{self, dot_product, fused_mul_add, vec_scale, with_scratch_a, with_scratch_b};

/// Configuration for kernel fusion.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Epsilon for layer normalization
    pub layer_norm_eps: f32,
    /// Enable GELU approximation (faster, ~0.01% less accurate)
    pub use_fast_gelu: bool,
    /// Minimum batch size to trigger rayon parallelism
    pub parallel_threshold: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            layer_norm_eps: 1e-5,
            use_fast_gelu: true,
            parallel_threshold: 4,
        }
    }
}

/// Types of fused operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOpType {
    /// LayerNorm + Linear
    LayerNormLinear,
    /// GELU + MLP (up + down projection)
    GeluMlp,
    /// RMSNorm + Residual add
    RmsNormResidual,
    /// RMSNorm standalone
    RmsNorm,
}

/// Pattern for a fused operation (used by the fusion optimizer).
#[derive(Debug, Clone)]
pub struct FusionPattern {
    /// Type of fused operation
    pub op_type: FusedOpType,
    /// Name for logging/tracing
    pub name: String,
}

/// Statistics for kernel fusion operations.
#[derive(Debug, Default)]
pub struct FusionStats {
    /// Total fused operations executed
    pub total_fused_ops: AtomicU64,
    /// Total bytes of memory traffic saved by fusion
    pub bytes_saved: AtomicU64,
}

impl Clone for FusionStats {
    fn clone(&self) -> Self {
        Self {
            total_fused_ops: AtomicU64::new(self.total_fused_ops.load(Ordering::Relaxed)),
            bytes_saved: AtomicU64::new(self.bytes_saved.load(Ordering::Relaxed)),
        }
    }
}

/// Kernel Fusion engine — fuses adjacent operations to eliminate memory traffic.
pub struct KernelFusion {
    config: FusionConfig,
    stats: FusionStats,
}

impl KernelFusion {
    /// Create a new KernelFusion engine.
    pub fn new(config: FusionConfig) -> Self {
        simd_dispatch::init();
        Self { config, stats: FusionStats::default() }
    }

    /// Fused LayerNorm + Linear: normalize then multiply by weight in one pass.
    ///
    /// `input`: [batch_size, hidden_size]
    /// `gamma`, `beta`: [hidden_size] (LN params)
    /// `weight`: [out_features, hidden_size]
    /// `bias`: [out_features] (optional)
    /// `output`: [batch_size, out_features] (pre-allocated)
    pub fn fused_layernorm_linear(
        &self,
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        batch_size: usize,
        hidden_size: usize,
        out_features: usize,
        output: &mut [f32],
    ) {
        let eps = self.config.layer_norm_eps;

        // Parallel over batch dimension
        let process_batch = |b: usize, out_row: &mut [f32]| {
            let inp = &input[b * hidden_size..(b + 1) * hidden_size];

            // Compute mean + variance in single pass (Welford-like)
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for &x in inp.iter() {
                let xd = x as f64;
                sum += xd;
                sum_sq += xd * xd;
            }
            let n = hidden_size as f64;
            let mean = sum / n;
            let var = (sum_sq / n) - mean * mean;
            let inv_std = (1.0 / (var + eps as f64).sqrt()) as f32;
            let mean_f = mean as f32;

            // Fused: normalize + linear in one pass using scratch for normalized values
            with_scratch_a(hidden_size, |normalized| {
                for i in 0..hidden_size {
                    normalized[i] = (inp[i] - mean_f) * inv_std * gamma[i] + beta[i];
                }
                // Linear: out = normalized × weight^T + bias
                for o in 0..out_features {
                    let w_row = &weight[o * hidden_size..(o + 1) * hidden_size];
                    out_row[o] = dot_product(normalized, w_row);
                }
                if let Some(bias) = bias {
                    simd_dispatch::vec_add(out_row, &bias[..out_features]);
                }
            });
        };

        if batch_size >= self.config.parallel_threshold {
            output.par_chunks_mut(out_features).enumerate().for_each(|(b, out_row)| {
                process_batch(b, out_row);
            });
        } else {
            for b in 0..batch_size {
                let out_row = &mut output[b * out_features..(b + 1) * out_features];
                process_batch(b, out_row);
            }
        }

        self.stats.total_fused_ops.fetch_add(1, Ordering::Relaxed);
        let saved = batch_size * hidden_size * std::mem::size_of::<f32>();
        self.stats.bytes_saved.fetch_add(saved as u64, Ordering::Relaxed);
    }

    /// Fused GELU MLP: up_project → GELU → down_project in minimal passes.
    ///
    /// `input`: [batch_size, hidden_size]
    /// `up_weight`: [intermediate_size, hidden_size]
    /// `down_weight`: [hidden_size, intermediate_size]
    /// `output`: [batch_size, hidden_size] (pre-allocated)
    pub fn fused_gelu_mlp(
        &self,
        input: &[f32],
        up_weight: &[f32],
        up_bias: Option<&[f32]>,
        down_weight: &[f32],
        down_bias: Option<&[f32]>,
        batch_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        output: &mut [f32],
    ) {
        let use_fast = self.config.use_fast_gelu;

        let process_batch = |b: usize, out_row: &mut [f32]| {
            let inp = &input[b * hidden_size..(b + 1) * hidden_size];

            // Use thread-local scratch for intermediate activations
            with_scratch_a(intermediate_size, |intermediate| {
                // Up projection
                for o in 0..intermediate_size {
                    let w_row = &up_weight[o * hidden_size..(o + 1) * hidden_size];
                    intermediate[o] = dot_product(inp, w_row);
                }
                if let Some(bias) = up_bias {
                    simd_dispatch::vec_add(intermediate, &bias[..intermediate_size]);
                }

                // GELU activation in-place
                if use_fast {
                    fast_gelu_inplace(intermediate);
                } else {
                    for x in intermediate.iter_mut() {
                        *x = *x * 0.5 * (1.0 + (*x * 0.7978845608 * (1.0 + 0.044715 * *x * *x)).tanh());
                    }
                }

                // Down projection
                for o in 0..hidden_size {
                    let w_row = &down_weight[o * intermediate_size..(o + 1) * intermediate_size];
                    out_row[o] = dot_product(intermediate, w_row);
                }
                if let Some(bias) = down_bias {
                    simd_dispatch::vec_add(out_row, &bias[..hidden_size]);
                }
            });
        };

        if batch_size >= self.config.parallel_threshold {
            output.par_chunks_mut(hidden_size).enumerate().for_each(|(b, out_row)| {
                process_batch(b, out_row);
            });
        } else {
            for b in 0..batch_size {
                let out_row = &mut output[b * hidden_size..(b + 1) * hidden_size];
                process_batch(b, out_row);
            }
        }

        self.stats.total_fused_ops.fetch_add(1, Ordering::Relaxed);
        let saved = batch_size * intermediate_size * std::mem::size_of::<f32>() * 2;
        self.stats.bytes_saved.fetch_add(saved as u64, Ordering::Relaxed);
    }

    /// Fused RMSNorm + Residual: output = RMSNorm(input + residual)
    ///
    /// Single read-write pass. No intermediate buffer needed.
    pub fn fused_rmsnorm_residual(
        &self,
        input: &[f32],
        residual: &[f32],
        gamma: &[f32],
        batch_size: usize,
        hidden_size: usize,
        output: &mut [f32],
    ) {
        let eps = self.config.layer_norm_eps;

        let process_batch = |b: usize, out_row: &mut [f32]| {
            let inp = &input[b * hidden_size..(b + 1) * hidden_size];
            let res = &residual[b * hidden_size..(b + 1) * hidden_size];

            // Compute RMS of (input + residual) in single pass
            let mut sum_sq = 0.0f64;
            for i in 0..hidden_size {
                let val = inp[i] + res[i];
                sum_sq += (val as f64) * (val as f64);
                out_row[i] = val; // Store sum temporarily
            }
            let rms = ((sum_sq / hidden_size as f64) + eps as f64).sqrt() as f32;
            let inv_rms = 1.0 / rms;

            // Normalize and apply gamma
            for i in 0..hidden_size {
                out_row[i] *= inv_rms * gamma[i];
            }
        };

        if batch_size >= self.config.parallel_threshold {
            output.par_chunks_mut(hidden_size).enumerate().for_each(|(b, out_row)| {
                process_batch(b, out_row);
            });
        } else {
            for b in 0..batch_size {
                let out_row = &mut output[b * hidden_size..(b + 1) * hidden_size];
                process_batch(b, out_row);
            }
        }

        self.stats.total_fused_ops.fetch_add(1, Ordering::Relaxed);
    }

    /// Standalone RMSNorm (no residual).
    pub fn rmsnorm(
        &self,
        input: &[f32],
        gamma: &[f32],
        batch_size: usize,
        hidden_size: usize,
        output: &mut [f32],
    ) {
        let eps = self.config.layer_norm_eps;

        let process = |b: usize, out: &mut [f32]| {
            let inp = &input[b * hidden_size..(b + 1) * hidden_size];
            let mut sum_sq = 0.0f64;
            for &x in inp.iter() {
                sum_sq += (x as f64) * (x as f64);
            }
            let inv_rms = (1.0 / ((sum_sq / hidden_size as f64) + eps as f64).sqrt()) as f32;
            for i in 0..hidden_size {
                out[i] = inp[i] * inv_rms * gamma[i];
            }
        };

        if batch_size >= self.config.parallel_threshold {
            output.par_chunks_mut(hidden_size).enumerate().for_each(|(b, out)| process(b, out));
        } else {
            for b in 0..batch_size {
                process(b, &mut output[b * hidden_size..(b + 1) * hidden_size]);
            }
        }
    }

    /// Get fusion statistics.
    pub fn stats(&self) -> &FusionStats { &self.stats }
}

/// Fast GELU approximation using tanh polynomial.
/// x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn fast_gelu_inplace(data: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const COEFF: f32 = 0.044715;

    for x in data.iter_mut() {
        let x3 = *x * *x * *x;
        let inner = SQRT_2_OVER_PI * (*x + COEFF * x3);
        *x = *x * 0.5 * (1.0 + fast_tanh(inner));
    }
}

/// Fast tanh using rational approximation. Max error ~3e-5.
#[inline(always)]
fn fast_tanh(x: f32) -> f32 {
    let x = x.clamp(-9.0, 9.0);
    let x2 = x * x;
    // Padé [3/2] approximation
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + 28.0 * x2));
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_layernorm_linear() {
        let config = FusionConfig::default();
        let fusion = KernelFusion::new(config);

        let (batch, hidden, out) = (2, 8, 4);
        let input = vec![1.0f32; batch * hidden];
        let gamma = vec![1.0f32; hidden];
        let beta = vec![0.0f32; hidden];
        let weight = vec![1.0f32; out * hidden];
        let bias = vec![0.0f32; out];
        let mut output = vec![0.0f32; batch * out];

        fusion.fused_layernorm_linear(
            &input, &gamma, &beta, &weight, Some(&bias),
            batch, hidden, out, &mut output,
        );

        // LN of constant is 0+beta=0, so output should be near 0
        for &v in &output { assert!(v.abs() < 1e-3, "Expected ~0, got {}", v); }
    }

    #[test]
    fn test_fused_gelu_mlp() {
        let fusion = KernelFusion::new(FusionConfig::default());
        let (batch, hidden, inter) = (2, 4, 8);
        let input = vec![1.0f32; batch * hidden];
        let up_w = vec![0.1f32; inter * hidden];
        let down_w = vec![0.1f32; hidden * inter];
        let mut output = vec![0.0f32; batch * hidden];

        fusion.fused_gelu_mlp(
            &input, &up_w, None, &down_w, None,
            batch, hidden, inter, &mut output,
        );
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_fused_rmsnorm_residual() {
        let fusion = KernelFusion::new(FusionConfig::default());
        let (batch, hidden) = (2, 8);
        let input = vec![1.0f32; batch * hidden];
        let residual = vec![1.0f32; batch * hidden];
        let gamma = vec![1.0f32; hidden];
        let mut output = vec![0.0f32; batch * hidden];

        fusion.fused_rmsnorm_residual(
            &input, &residual, &gamma, batch, hidden, &mut output,
        );

        // (1+1)=2, RMS=2, normalized = 2/2 * 1 = 1
        for &v in &output { assert!((v - 1.0).abs() < 1e-3, "Expected ~1.0, got {}", v); }
    }

    #[test]
    fn test_fast_gelu() {
        let mut data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let expected: Vec<f32> = data.iter().map(|&x| {
            let x = x as f32;
            x * 0.5 * (1.0 + (0.7978845608_f32 * (x + 0.044715 * x * x * x)).tanh())
        }).collect();
        fast_gelu_inplace(&mut data);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 0.01, "GELU mismatch: {} vs {}", a, b);
        }
    }
}
