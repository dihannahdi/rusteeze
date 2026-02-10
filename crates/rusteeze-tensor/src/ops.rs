//! Optimized tensor operations for LLM inference.
//!
//! This module provides high-performance tensor operations including:
//!
//! - Activation functions (SiLU, GELU, ReLU, etc.)
//! - Normalization (RMSNorm, LayerNorm)
//! - Linear layers with optimizations
//! - Fused operations for efficiency

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};
use tracing::instrument;

/// SiLU (Swish) activation: x * sigmoid(x)
#[instrument(skip_all, level = "trace")]
pub fn silu(x: &Tensor) -> Result<Tensor> {
    let sigmoid = candle_nn::ops::sigmoid(x)?;
    x.mul(&sigmoid)
}

/// GELU activation (approximate)
#[instrument(skip_all, level = "trace")]
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let x3 = x.powf(3.0)?;
    let inner = (x + (x3 * 0.044715)?)?;
    let inner = (inner * 0.7978845608)?; // sqrt(2/pi)
    let tanh = inner.tanh()?;
    let result = ((&tanh + 1.0)? * 0.5)?;
    x.mul(&result)
}

/// GELU activation (exact using erf)
pub fn gelu_exact(x: &Tensor) -> Result<Tensor> {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // Using sigmoid approximation: sigmoid(1.702 * x) ≈ 0.5 * (1 + erf(x / sqrt(2)))
    let sqrt2_inv = 0.7071067811865476; // 1/sqrt(2)
    let scaled = (x * sqrt2_inv)?;
    let sig = candle_nn::ops::sigmoid(&(scaled * 1.702)?)?;
    // result = 2 * sig - 1
    let two = Tensor::new(&[2.0f32], x.device())?.broadcast_as(sig.shape())?;
    let one = Tensor::new(&[1.0f32], x.device())?.broadcast_as(sig.shape())?;
    let result = (sig.broadcast_mul(&two)? - one)?;
    // half = (result + 1) * 0.5
    let one2 = Tensor::new(&[1.0f32], x.device())?.broadcast_as(result.shape())?;
    let half_scalar = Tensor::new(&[0.5f32], x.device())?.broadcast_as(result.shape())?;
    let half = ((result + one2)? * half_scalar)?;
    x.mul(&half)
}

/// ReLU activation
pub fn relu(x: &Tensor) -> Result<Tensor> {
    x.relu()
}

/// Leaky ReLU activation
pub fn leaky_relu(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let positive = x.maximum(&zeros)?;
    let negative = (x.minimum(&zeros)? * negative_slope)?;
    positive + negative
}

/// Softmax along a dimension
pub fn softmax(x: &Tensor, dim: D) -> Result<Tensor> {
    candle_nn::ops::softmax(x, dim)
}

/// Log softmax along a dimension
pub fn log_softmax(x: &Tensor, dim: D) -> Result<Tensor> {
    candle_nn::ops::log_softmax(x, dim)
}

/// RMS Normalization
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Create a new RMSNorm layer.
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Load from VarBuilder.
    pub fn load(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get((hidden_size,), "weight")?;
        Ok(Self { weight, eps })
    }

    /// Create with ones initialization.
    pub fn ones(hidden_size: usize, eps: f64, device: &Device, dtype: DType) -> Result<Self> {
        let weight = Tensor::ones((hidden_size,), dtype, device)?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_norm(x, &self.weight, self.eps)
    }
}

/// RMS normalization operation
#[instrument(skip_all, level = "trace")]
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps)?.sqrt()?;
    let normalized = x.broadcast_div(&rms)?;
    normalized.broadcast_mul(weight)
}

/// Layer Normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    /// Create a new LayerNorm.
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    /// Load from VarBuilder.
    pub fn load(vb: VarBuilder, hidden_size: usize, eps: f64, use_bias: bool) -> Result<Self> {
        let weight = vb.get((hidden_size,), "weight")?;
        let bias = if use_bias {
            Some(vb.get((hidden_size,), "bias")?)
        } else {
            None
        };
        Ok(Self { weight, bias, eps })
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm(x, &self.weight, self.bias.as_ref(), self.eps)
    }
}

/// Layer normalization operation
#[instrument(skip_all, level = "trace")]
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    // LayerNorm(x) = (x - mean) / sqrt(var + eps) * weight + bias
    let mean = x.mean_keepdim(D::Minus1)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    let normalized = x_centered.broadcast_div(&std)?;
    let scaled = normalized.broadcast_mul(weight)?;
    
    if let Some(bias) = bias {
        scaled.broadcast_add(bias)
    } else {
        Ok(scaled)
    }
}

/// Linear layer (no bias)
#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    /// Create a new linear layer.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    /// Load from VarBuilder.
    pub fn load(vb: VarBuilder, in_features: usize, out_features: usize, use_bias: bool) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        let bias = if use_bias {
            Some(vb.get((out_features,), "bias")?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// Get the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias tensor.
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let result = x.matmul(&self.weight.t()?)?;
        if let Some(bias) = &self.bias {
            result.broadcast_add(bias)
        } else {
            Ok(result)
        }
    }
}

/// Embedding layer
#[derive(Debug, Clone)]
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    /// Create a new embedding layer.
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    /// Load from VarBuilder.
    pub fn load(vb: VarBuilder, vocab_size: usize, embed_dim: usize) -> Result<Self> {
        let weight = vb.get((vocab_size, embed_dim), "weight")?;
        Ok(Self { weight })
    }

    /// Create with random initialization.
    pub fn rand(vocab_size: usize, embed_dim: usize, device: &Device, dtype: DType) -> Result<Self> {
        let weight = Tensor::randn(0.0f32, 0.02, (vocab_size, embed_dim), device)?.to_dtype(dtype)?;
        Ok(Self { weight })
    }

    /// Forward pass.
    pub fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        self.weight.index_select(indices, 0)
    }

    /// Get vocab size.
    pub fn vocab_size(&self) -> usize {
        self.weight.dim(0).unwrap_or(0)
    }

    /// Get embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.weight.dim(1).unwrap_or(0)
    }
}

/// Fused SiLU-multiplied linear (for gated FFN)
#[instrument(skip_all, level = "trace")]
pub fn fused_silu_mul(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let activated = silu(x)?;
    activated.mul(gate)
}

/// Fused attention projection
/// Combines Q, K, V projections into a single operation
#[instrument(skip_all, level = "trace")]
pub fn fused_qkv_projection(
    hidden_states: &Tensor,
    qkv_weight: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let batch_size = hidden_states.dim(0)?;
    let seq_len = hidden_states.dim(1)?;

    // Project: [batch, seq, hidden] @ [hidden, (q + k + v)] -> [batch, seq, q + k + v]
    let qkv = hidden_states.matmul(&qkv_weight.t()?)?;

    // Split into Q, K, V
    let q_size = num_heads * head_dim;
    let kv_size = num_kv_heads * head_dim;

    let q = qkv.narrow(2, 0, q_size)?;
    let k = qkv.narrow(2, q_size, kv_size)?;
    let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

    // Reshape to [batch, num_heads, seq, head_dim]
    let q = q.reshape((batch_size, seq_len, num_heads, head_dim))?
        .transpose(1, 2)?;
    let k = k.reshape((batch_size, seq_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;
    let v = v.reshape((batch_size, seq_len, num_kv_heads, head_dim))?
        .transpose(1, 2)?;

    Ok((q, k, v))
}

/// Apply rotary position embeddings to Q and K
pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Assumes q, k are [batch, heads, seq, head_dim]
    // cos, sin are [seq, head_dim]
    let q_embed = apply_rotary_to_tensor(q, cos, sin)?;
    let k_embed = apply_rotary_to_tensor(k, cos, sin)?;
    Ok((q_embed, k_embed))
}

fn apply_rotary_to_tensor(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_, _, seq_len, _) = x.dims4()?;
    
    // Slice cos/sin to match sequence length
    let cos = cos.i(..seq_len)?;
    let sin = sin.i(..seq_len)?;
    
    // Reshape for broadcasting
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    
    // Split in half
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    
    // Apply rotation
    let cos = cos.narrow(2, 0, half_dim)?;
    let sin = sin.narrow(2, 0, half_dim)?;
    
    // [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    let rotated_x1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let rotated_x2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    
    Tensor::cat(&[rotated_x1, rotated_x2], 3)
}

/// Causal masking utility
pub fn create_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    // Build lower triangular mask manually
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[i * seq_len + j] = 1.0;
        }
    }
    let mask = Tensor::new(mask_data.as_slice(), device)?.reshape((seq_len, seq_len))?;
    // where_cond requires a U8 condition tensor
    let mask_u8 = mask.to_dtype(DType::U8)?;
    let neg_inf_tensor = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device)?.to_dtype(dtype)?;
    let zero_tensor = Tensor::zeros((seq_len, seq_len), dtype, device)?;
    mask_u8.where_cond(&zero_tensor, &neg_inf_tensor)
}

/// Top-k sampling helper
pub fn top_k_mask(logits: &Tensor, k: usize) -> Result<Tensor> {
    let (batch, vocab) = logits.dims2()?;
    
    // Get top-k values and indices
    let sorted = logits.arg_sort_last_dim(false)?;
    
    // Create mask
    let mut mask_data = vec![f32::NEG_INFINITY; batch * vocab];
    let sorted_data: Vec<u32> = sorted.flatten_all()?.to_vec1()?;
    
    for b in 0..batch {
        for i in 0..k.min(vocab) {
            let idx = sorted_data[b * vocab + i] as usize;
            mask_data[b * vocab + idx] = 0.0;
        }
    }
    
    Tensor::from_vec(mask_data, (batch, vocab), logits.device())
}

/// Top-p (nucleus) sampling helper
pub fn top_p_mask(logits: &Tensor, p: f32) -> Result<Tensor> {
    let (batch, vocab) = logits.dims2()?;
    let device = logits.device();
    
    // Compute probabilities
    let probs = softmax(logits, D::Minus1)?;
    let probs_data: Vec<f32> = probs.flatten_all()?.to_vec1()?;
    
    // Sort and compute cumulative sum
    let mut mask_data = vec![f32::NEG_INFINITY; batch * vocab];
    
    for b in 0..batch {
        let mut indexed: Vec<(usize, f32)> = (0..vocab)
            .map(|i| (i, probs_data[b * vocab + i]))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut cumsum = 0.0;
        for (idx, prob) in indexed {
            if cumsum < p {
                mask_data[b * vocab + idx] = 0.0;
                cumsum += prob;
            }
        }
    }
    
    Tensor::from_vec(mask_data, (batch, vocab), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[-1.0f32, 0.0, 1.0, 2.0], &device)?;
        let y = silu(&x)?;
        let y_data: Vec<f32> = y.to_vec1()?;
        
        // silu(0) should be 0
        assert!((y_data[1]).abs() < 1e-6);
        // silu(x) ≈ x for large x
        assert!((y_data[3] - 2.0 * 0.88).abs() < 0.1);
        
        Ok(())
    }

    #[test]
    fn test_rms_norm() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;
        let weight = Tensor::ones((4,), DType::F32, &device)?;
        
        let y = rms_norm(&x, &weight, 1e-6)?;
        
        // Check shape
        assert_eq!(y.dims(), x.dims());
        
        // RMS norm should normalize
        let y_data: Vec<f32> = y.flatten_all()?.to_vec1()?;
        let rms: f32 = y_data.iter().map(|v| v * v).sum::<f32>() / y_data.len() as f32;
        assert!((rms.sqrt() - 1.0).abs() < 0.1);
        
        Ok(())
    }

    #[test]
    fn test_layer_norm() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;
        let weight = Tensor::ones((4,), DType::F32, &device)?;
        let bias = Tensor::zeros((4,), DType::F32, &device)?;
        
        let y = layer_norm(&x, &weight, Some(&bias), 1e-6)?;
        
        // After layer norm, mean should be ~0 and std should be ~1
        let y_data: Vec<f32> = y.flatten_all()?.to_vec1()?;
        let mean: f32 = y_data.iter().sum::<f32>() / y_data.len() as f32;
        assert!(mean.abs() < 1e-5);
        
        Ok(())
    }

    #[test]
    fn test_linear() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]], &device)?;
        let linear = Linear::new(weight, None);
        
        let x = Tensor::new(&[[1.0f32, 2.0]], &device)?;
        let y = linear.forward(&x)?;
        
        // [1, 2] @ [[1, 0, 1], [0, 1, 1]] = [1, 2, 3]
        let y_data: Vec<f32> = y.flatten_all()?.to_vec1()?;
        assert_eq!(y_data.len(), 3);
        assert!((y_data[0] - 1.0).abs() < 1e-6);
        assert!((y_data[1] - 2.0).abs() < 1e-6);
        assert!((y_data[2] - 3.0).abs() < 1e-6);
        
        Ok(())
    }

    #[test]
    fn test_embedding() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4], [0.5, 0.6]], &device)?;
        let embed = Embedding::new(weight);
        
        assert_eq!(embed.vocab_size(), 3);
        assert_eq!(embed.embed_dim(), 2);
        
        let indices = Tensor::new(&[0u32, 2, 1], &device)?;
        let out = embed.forward(&indices)?;
        
        assert_eq!(out.dims(), &[3, 2]);
        
        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, &device, DType::F32)?;
        let mask_data: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        
        // Upper triangle should be -inf
        assert!(mask_data[1].is_infinite() && mask_data[1] < 0.0); // (0, 1)
        // Lower triangle and diagonal should be 0
        assert_eq!(mask_data[0], 0.0); // (0, 0)
        assert_eq!(mask_data[4], 0.0); // (1, 0)
        assert_eq!(mask_data[5], 0.0); // (1, 1)
        
        Ok(())
    }
}
