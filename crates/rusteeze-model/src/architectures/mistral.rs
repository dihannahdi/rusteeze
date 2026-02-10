//! Mistral model implementation.
//!
//! This module provides a complete implementation of the Mistral architecture
//! including Mistral 7B and its variants with sliding window attention.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

use super::common::{Attention, MLP, RmsNorm, RotaryEmbedding, create_causal_mask};
use super::{KVCache, LayerKVCache, Model, ModelError};
use crate::config::ModelConfig;

/// Mistral-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralConfig {
    /// Vocabulary size.
    pub vocab_size: usize,

    /// Hidden size.
    pub hidden_size: usize,

    /// Intermediate (FFN) size.
    pub intermediate_size: usize,

    /// Number of hidden layers.
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA).
    pub num_key_value_heads: usize,

    /// Maximum sequence length.
    pub max_position_embeddings: usize,

    /// RMS norm epsilon.
    pub rms_norm_eps: f64,

    /// RoPE theta (base frequency).
    pub rope_theta: f32,

    /// Sliding window size.
    pub sliding_window: Option<usize>,

    /// Tie word embeddings.
    pub tie_word_embeddings: bool,
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            sliding_window: Some(4096),
            tie_word_embeddings: false,
        }
    }
}

impl From<&ModelConfig> for MistralConfig {
    fn from(config: &ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads.unwrap_or(config.num_attention_heads / 4),
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f32,
            sliding_window: config.sliding_window,
            tie_word_embeddings: config.tie_word_embeddings,
        }
    }
}

/// Mistral decoder layer with sliding window attention.
#[derive(Debug)]
pub struct MistralDecoderLayer {
    /// Self attention.
    self_attn: MistralAttention,

    /// MLP.
    mlp: MLP,

    /// Input layer norm.
    input_layernorm: RmsNorm,

    /// Post attention layer norm.
    post_attention_layernorm: RmsNorm,
}

impl MistralDecoderLayer {
    /// Create new decoder layer.
    pub fn new(config: &MistralConfig, vb: VarBuilder) -> Result<Self, ModelError> {
        let head_dim = config.hidden_size / config.num_attention_heads;

        let self_attn = MistralAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
            config.sliding_window,
            vb.pp("self_attn"),
        )?;

        let mlp = MLP::new(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("mlp"),
        )?;

        let input_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;

        let post_attention_layernorm = RmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rotary_emb: &RotaryEmbedding,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut LayerKVCache>,
    ) -> Result<Tensor, ModelError> {
        // Self attention with residual
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            rotary_emb,
            position_ids,
            attention_mask,
            kv_cache,
        )?;
        let hidden_states = (residual + hidden_states)?;

        // MLP with residual
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        Ok(hidden_states)
    }
}

/// Mistral attention with sliding window.
#[derive(Debug)]
pub struct MistralAttention {
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

    /// Sliding window size.
    sliding_window: Option<usize>,
}

impl MistralAttention {
    /// Create new attention layer.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
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
            sliding_window,
        })
    }

    /// Forward pass with sliding window attention.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rotary_emb: &RotaryEmbedding,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut LayerKVCache>,
    ) -> Result<Tensor, ModelError> {
        let (batch_size, seq_len, _) = hidden_states.dims3()
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
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

        // Attention scores
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * self.scale)?;

        // Apply sliding window mask
        let attn_weights = if let Some(window) = self.sliding_window {
            self.apply_sliding_window_mask(&attn_weights, window)?
        } else {
            attn_weights
        };

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

    /// Apply sliding window attention mask.
    fn apply_sliding_window_mask(
        &self,
        attn_weights: &Tensor,
        window_size: usize,
    ) -> Result<Tensor, ModelError> {
        let (_, _, q_len, kv_len) = attn_weights.dims4()
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        // Create sliding window mask
        let mask: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    let pos = kv_len - q_len + i;
                    if j <= pos && pos - j < window_size {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        let mask = Tensor::from_vec(mask, (q_len, kv_len), attn_weights.device())
            .map_err(|e| ModelError::TensorError(e.to_string()))?
            .to_dtype(attn_weights.dtype())
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        attn_weights.broadcast_add(&mask)
            .map_err(|e| ModelError::TensorError(e.to_string()))
    }
}

/// Mistral model.
pub struct MistralModel {
    /// Model configuration.
    config: ModelConfig,

    /// Mistral-specific configuration.
    mistral_config: MistralConfig,

    /// Token embeddings.
    embed_tokens: Embedding,

    /// Decoder layers.
    layers: Vec<MistralDecoderLayer>,

    /// Final layer norm.
    norm: RmsNorm,

    /// LM head (output projection).
    lm_head: Linear,

    /// Rotary embeddings.
    rotary_emb: RotaryEmbedding,

    /// Device.
    device: Device,

    /// Data type.
    dtype: DType,
}

impl MistralModel {
    /// Create new Mistral model.
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self, ModelError> {
        let mistral_config = MistralConfig::from(config);
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Build model components
        let embed_tokens = embedding(
            mistral_config.vocab_size,
            mistral_config.hidden_size,
            vb.pp("model.embed_tokens"),
        ).map_err(|e| ModelError::MissingWeight(e.to_string()))?;

        // Build decoder layers
        let mut layers = Vec::with_capacity(mistral_config.num_hidden_layers);
        for i in 0..mistral_config.num_hidden_layers {
            let layer = MistralDecoderLayer::new(
                &mistral_config,
                vb.pp(format!("model.layers.{}", i)),
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::new(
            mistral_config.hidden_size,
            mistral_config.rms_norm_eps,
            vb.pp("model.norm"),
        )?;

        // LM head (may be tied with embeddings)
        let lm_head = if mistral_config.tie_word_embeddings {
            let weight = embed_tokens.embeddings().clone();
            Linear::new(weight, None)
        } else {
            linear(
                mistral_config.hidden_size,
                mistral_config.vocab_size,
                vb.pp("lm_head"),
            ).map_err(|e| ModelError::MissingWeight(e.to_string()))?
        };

        // Rotary embeddings
        let head_dim = mistral_config.hidden_size / mistral_config.num_attention_heads;
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            mistral_config.max_position_embeddings,
            mistral_config.rope_theta,
            dtype,
            &device,
        )?;

        Ok(Self {
            config: config.clone(),
            mistral_config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            device,
            dtype,
        })
    }

    /// Get logits from hidden states.
    pub fn get_logits(&self, hidden_states: &Tensor) -> Result<Tensor, ModelError> {
        self.lm_head.forward(hidden_states)
            .map_err(|e| ModelError::TensorError(e.to_string()))
    }
}

impl Model for MistralModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor, ModelError> {
        self.forward_with_mask(input_ids, position_ids, None, kv_cache)
    }

    fn forward_with_mask(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor, ModelError> {
        let (batch_size, seq_len) = input_ids.dims2()
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        // Get embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Create causal mask if not provided
        let causal_mask = if attention_mask.is_none() && seq_len > 1 {
            Some(create_causal_mask(seq_len, self.dtype, &self.device)?)
        } else {
            None
        };
        let attention_mask = attention_mask.or(causal_mask.as_ref());

        // Process through layers
        if let Some(cache) = kv_cache {
            for (i, layer) in self.layers.iter().enumerate() {
                let layer_cache = cache.layer_mut(i);
                hidden_states = layer.forward(
                    &hidden_states,
                    &self.rotary_emb,
                    position_ids,
                    attention_mask,
                    layer_cache,
                )?;
            }
        } else {
            for layer in &self.layers {
                hidden_states = layer.forward(
                    &hidden_states,
                    &self.rotary_emb,
                    position_ids,
                    attention_mask,
                    None,
                )?;
            }
        }

        // Final normalization
        let hidden_states = self.norm.forward(&hidden_states)?;

        // LM head
        let logits = self.get_logits(&hidden_states)?;

        Ok(logits)
    }

    fn hidden_states(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor, ModelError> {
        let (batch_size, seq_len) = input_ids.dims2()
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let attention_mask = if seq_len > 1 {
            Some(create_causal_mask(seq_len, self.dtype, &self.device)?)
        } else {
            None
        };

        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                &self.rotary_emb,
                position_ids,
                attention_mask.as_ref(),
                None,
            )?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;

        Ok(hidden_states)
    }

    fn num_layers(&self) -> usize {
        self.mistral_config.num_hidden_layers
    }

    fn hidden_size(&self) -> usize {
        self.mistral_config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.mistral_config.vocab_size
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mistral_config_default() {
        let config = MistralConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.sliding_window, Some(4096));
    }
}
