//! Llama model implementation.
//!
//! This module provides a complete implementation of the Llama architecture
//! including Llama, Llama 2, Llama 3, and Code Llama variants.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

use super::common::{Attention, MLP, RmsNorm, RotaryEmbedding, create_causal_mask};
use super::{KVCache, LayerKVCache, Model, ModelError};
use crate::config::ModelConfig;

/// Llama-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
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

    /// Tie word embeddings.
    pub tie_word_embeddings: bool,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
        }
    }
}

impl From<&ModelConfig> for LlamaConfig {
    fn from(config: &ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size.unwrap_or(config.hidden_size * 4),
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads.unwrap_or(config.num_attention_heads),
            max_position_embeddings: config.max_position_embeddings.unwrap_or(4096),
            rms_norm_eps: config.rms_norm_eps.unwrap_or(1e-5),
            rope_theta: config.rope_theta.unwrap_or(10000.0),
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
        }
    }
}

/// Llama decoder layer.
#[derive(Debug)]
pub struct LlamaDecoderLayer {
    /// Self attention.
    self_attn: Attention,

    /// MLP.
    mlp: MLP,

    /// Input layer norm.
    input_layernorm: RmsNorm,

    /// Post attention layer norm.
    post_attention_layernorm: RmsNorm,
}

impl LlamaDecoderLayer {
    /// Create new decoder layer.
    pub fn new(config: &LlamaConfig, vb: VarBuilder) -> Result<Self, ModelError> {
        let head_dim = config.hidden_size / config.num_attention_heads;

        let self_attn = Attention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
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

/// Llama model.
pub struct LlamaModel {
    /// Model configuration.
    config: ModelConfig,

    /// Llama-specific configuration.
    llama_config: LlamaConfig,

    /// Token embeddings.
    embed_tokens: Embedding,

    /// Decoder layers.
    layers: Vec<LlamaDecoderLayer>,

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

impl LlamaModel {
    /// Create new Llama model.
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self, ModelError> {
        let llama_config = LlamaConfig::from(config);
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Build model components
        let embed_tokens = embedding(
            llama_config.vocab_size,
            llama_config.hidden_size,
            vb.pp("model.embed_tokens"),
        ).map_err(|e| ModelError::MissingWeight(e.to_string()))?;

        // Build decoder layers
        let mut layers = Vec::with_capacity(llama_config.num_hidden_layers);
        for i in 0..llama_config.num_hidden_layers {
            let layer = LlamaDecoderLayer::new(
                &llama_config,
                vb.pp(format!("model.layers.{}", i)),
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::new(
            llama_config.hidden_size,
            llama_config.rms_norm_eps,
            vb.pp("model.norm"),
        )?;

        // LM head (may be tied with embeddings)
        let lm_head = if llama_config.tie_word_embeddings {
            // Create linear from embedding weights
            let weight = embed_tokens.embeddings().clone();
            Linear::new(weight, None)
        } else {
            linear(
                llama_config.hidden_size,
                llama_config.vocab_size,
                vb.pp("lm_head"),
            ).map_err(|e| ModelError::MissingWeight(e.to_string()))?
        };

        // Rotary embeddings
        let head_dim = llama_config.hidden_size / llama_config.num_attention_heads;
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            llama_config.max_position_embeddings,
            llama_config.rope_theta,
            dtype,
            &device,
        )?;

        Ok(Self {
            config: config.clone(),
            llama_config,
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

impl Model for LlamaModel {
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

        // Get embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Create causal mask
        let attention_mask = if seq_len > 1 {
            Some(create_causal_mask(seq_len, self.dtype, &self.device)?)
        } else {
            None
        };

        // Process through layers
        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                &self.rotary_emb,
                position_ids,
                attention_mask.as_ref(),
                None,
            )?;
        }

        // Final normalization
        let hidden_states = self.norm.forward(&hidden_states)?;

        Ok(hidden_states)
    }

    fn num_layers(&self) -> usize {
        self.llama_config.num_hidden_layers
    }

    fn hidden_size(&self) -> usize {
        self.llama_config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.llama_config.vocab_size
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
    fn test_llama_config_default() {
        let config = LlamaConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
    }

    #[test]
    fn test_llama_config_from_model_config() {
        let model_config = ModelConfig {
            vocab_size: 128256,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            intermediate_size: Some(14336),
            max_position_embeddings: Some(8192),
            ..Default::default()
        };

        let llama_config = LlamaConfig::from(&model_config);
        assert_eq!(llama_config.vocab_size, 128256);
        assert_eq!(llama_config.num_key_value_heads, 8);
        assert_eq!(llama_config.intermediate_size, 14336);
    }
}
