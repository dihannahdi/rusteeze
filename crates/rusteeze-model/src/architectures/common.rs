//! Common model components.
//!
//! Shared building blocks for transformer architectures.

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use super::ModelError;

/// RMS normalization layer.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Create new RMS norm.
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self, ModelError> {
        let weight = vb
            .get(size, "weight")
            .map_err(|e| ModelError::MissingWeight(e.to_string()))?;

        Ok(Self { weight, eps })
    }

    /// Create from weight tensor.
    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ModelError> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;

        // Compute variance
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        // Apply weight
        let x = x.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;

        Ok(x.to_dtype(dtype)?)
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward(x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

/// Rotary positional embedding.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Cosine cache.
    cos_cache: Tensor,

    /// Sine cache.
    sin_cache: Tensor,

    /// Head dimension.
    head_dim: usize,
}

impl RotaryEmbedding {
    /// Create new rotary embedding.
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let inv_freq = Self::compute_inv_freq(head_dim, base, device)?;
        let (cos_cache, sin_cache) =
            Self::compute_cache(&inv_freq, max_seq_len, dtype, device)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    /// Compute inverse frequencies.
    fn compute_inv_freq(
        head_dim: usize,
        base: f32,
        device: &Device,
    ) -> Result<Tensor, ModelError> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(i as f32 * 2.0 / head_dim as f32))
            .collect();

        Tensor::new(inv_freq, device).map_err(|e| ModelError::TensorError(e.to_string()))
    }

    /// Compute cos/sin cache.
    fn compute_cache(
        inv_freq: &Tensor,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor), ModelError> {
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::new(positions, device)
            .map_err(|e| ModelError::TensorError(e.to_string()))?
            .unsqueeze(1)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        let inv_freq = inv_freq
            .unsqueeze(0)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        let freqs = positions
            .matmul(&inv_freq)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        // [seq_len, head_dim]
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        let cos = freqs
            .cos()
            .map_err(|e| ModelError::TensorError(e.to_string()))?
            .to_dtype(dtype)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        let sin = freqs
            .sin()
            .map_err(|e| ModelError::TensorError(e.to_string()))?
            .to_dtype(dtype)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        Ok((cos, sin))
    }

    /// Apply rotary embedding.
    pub fn apply(&self, q: &Tensor, k: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor), ModelError> {
        let cos = self.cos_cache.index_select(position_ids, 0)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;
        let sin = self.sin_cache.index_select(position_ids, 0)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        let q_embed = self.rotate_half(q, &cos, &sin)?;
        let k_embed = self.rotate_half(k, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }

    /// Rotate half.
    fn rotate_half(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, ModelError> {
        let half = self.head_dim / 2;
        let x1 = x.narrow(candle_core::D::Minus1, 0, half)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;
        let x2 = x.narrow(candle_core::D::Minus1, half, half)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        let rotated = Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        let cos = cos.unsqueeze(1).map_err(|e| ModelError::TensorError(e.to_string()))?;
        let sin = sin.unsqueeze(1).map_err(|e| ModelError::TensorError(e.to_string()))?;

        let result = x.broadcast_mul(&cos)?.broadcast_add(&rotated.broadcast_mul(&sin)?)?;
        Ok(result)
    }
}

/// Multi-head attention.
#[derive(Debug)]
pub struct Attention {
    /// Query projection.
    q_proj: Linear,

    /// Key projection.
    k_proj: Linear,

    /// Value projection.
    v_proj: Linear,

    /// Output projection.
    o_proj: Linear,

    /// Number of attention heads.
    num_heads: usize,

    /// Number of key-value heads.
    num_kv_heads: usize,

    /// Head dimension.
    head_dim: usize,

    /// Softmax scale.
    scale: f64,
}

impl Attention {
    /// Create new attention layer.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self, ModelError> {
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))
            .map_err(|e| ModelError::MissingWeight(e.to_string()))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))
            .map_err(|e| ModelError::MissingWeight(e.to_string()))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))
            .map_err(|e| ModelError::MissingWeight(e.to_string()))?;
        let o_proj = linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))
            .map_err(|e| ModelError::MissingWeight(e.to_string()))?;

        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rotary_emb: &RotaryEmbedding,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut super::LayerKVCache>,
    ) -> Result<Tensor, ModelError> {
        let (batch_size, seq_len, _) = hidden_states.dims3()
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // [batch, num_heads, seq, head_dim]
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embedding
        let (q, k) = rotary_emb.apply(&q, &k, position_ids)?;

        // Update KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?;
            (cache.get_key()?, cache.get_value()?)
        } else {
            (k, v)
        };

        // Repeat KV for GQA
        let k = Self::repeat_kv(&k, self.num_heads / self.num_kv_heads)?;
        let v = Self::repeat_kv(&v, self.num_heads / self.num_kv_heads)?;

        // Attention
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * self.scale)?;

        // Apply attention mask
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

        // Apply to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        let output = self.o_proj.forward(&attn_output)?;

        Ok(output)
    }

    /// Repeat KV for grouped query attention.
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor, ModelError> {
        if n_rep == 1 {
            return Ok(x.clone());
        }

        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        let x = x
            .unsqueeze(2)?
            .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;

        Ok(x)
    }
}

/// MLP (feed-forward) layer with SwiGLU activation.
#[derive(Debug)]
pub struct MLP {
    /// Gate projection.
    gate_proj: Linear,

    /// Up projection.
    up_proj: Linear,

    /// Down projection.
    down_proj: Linear,
}

impl MLP {
    /// Create new MLP layer.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self, ModelError> {
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| ModelError::MissingWeight(e.to_string()))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("up_proj"))
            .map_err(|e| ModelError::MissingWeight(e.to_string()))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("down_proj"))
            .map_err(|e| ModelError::MissingWeight(e.to_string()))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass with SwiGLU activation.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ModelError> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let x = (gate * up)?;
        let x = self.down_proj.forward(&x)?;
        Ok(x)
    }
}

/// Create causal attention mask.
pub fn create_causal_mask(
    seq_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, ModelError> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if j <= i {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();

    Tensor::from_vec(mask, (seq_len, seq_len), device)
        .map_err(|e| ModelError::TensorError(e.to_string()))?
        .to_dtype(dtype)
        .map_err(|e| ModelError::TensorError(e.to_string()))
}

/// Create attention mask from sequence lengths.
pub fn create_attention_mask_from_lengths(
    lengths: &[usize],
    max_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, ModelError> {
    let batch_size = lengths.len();
    let mask: Vec<f32> = lengths
        .iter()
        .flat_map(|&len| {
            (0..max_len).map(move |i| {
                if i < len {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();

    Tensor::from_vec(mask, (batch_size, 1, 1, max_len), device)
        .map_err(|e| ModelError::TensorError(e.to_string()))?
        .to_dtype(dtype)
        .map_err(|e| ModelError::TensorError(e.to_string()))
}
