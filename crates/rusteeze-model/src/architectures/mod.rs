//! Model architecture implementations.
//!
//! This module provides implementations for various transformer architectures
//! including Llama, Mistral, Qwen, Phi, and others.

pub mod llama;
pub mod mistral;
pub mod common;

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::config::{ModelArchitecture, ModelConfig};
use crate::loader::LoaderError;

pub use llama::{LlamaModel, LlamaConfig};
pub use mistral::{MistralModel, MistralConfig};
pub use common::*;

/// Trait for all model architectures.
pub trait Model: Send + Sync {
    /// Get model configuration.
    fn config(&self) -> &ModelConfig;

    /// Forward pass.
    fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor, ModelError>;

    /// Forward pass with attention mask.
    fn forward_with_mask(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor, ModelError>;

    /// Get hidden states (for embedding models).
    fn hidden_states(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor, ModelError>;

    /// Number of layers.
    fn num_layers(&self) -> usize;

    /// Hidden size.
    fn hidden_size(&self) -> usize;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Device.
    fn device(&self) -> &Device;

    /// Data type.
    fn dtype(&self) -> DType;
}

/// KV cache for efficient inference.
#[derive(Debug)]
pub struct KVCache {
    /// Layer caches.
    layers: Vec<LayerKVCache>,

    /// Maximum sequence length.
    max_seq_len: usize,

    /// Current sequence length.
    seq_len: usize,
}

impl KVCache {
    /// Create new KV cache.
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            layers.push(LayerKVCache::new(
                batch_size,
                max_seq_len,
                num_kv_heads,
                head_dim,
                dtype,
                device,
            )?);
        }

        Ok(Self {
            layers,
            max_seq_len,
            seq_len: 0,
        })
    }

    /// Get layer cache.
    pub fn layer(&self, index: usize) -> Option<&LayerKVCache> {
        self.layers.get(index)
    }

    /// Get mutable layer cache.
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut LayerKVCache> {
        self.layers.get_mut(index)
    }

    /// Current sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Update sequence length.
    pub fn set_seq_len(&mut self, seq_len: usize) {
        self.seq_len = seq_len;
    }

    /// Reset cache.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

/// Per-layer KV cache.
#[derive(Debug)]
pub struct LayerKVCache {
    /// Key cache: [batch, num_kv_heads, seq_len, head_dim]
    key: Tensor,

    /// Value cache: [batch, num_kv_heads, seq_len, head_dim]
    value: Tensor,

    /// Current position.
    position: usize,
}

impl LayerKVCache {
    /// Create new layer cache.
    pub fn new(
        batch_size: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, ModelError> {
        let shape = (batch_size, num_kv_heads, max_seq_len, head_dim);

        let key = Tensor::zeros(shape, dtype, device)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;
        let value = Tensor::zeros(shape, dtype, device)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        Ok(Self {
            key,
            value,
            position: 0,
        })
    }

    /// Get key tensor.
    pub fn key(&self) -> &Tensor {
        &self.key
    }

    /// Get value tensor.
    pub fn value(&self) -> &Tensor {
        &self.value
    }

    /// Update cache with new keys and values.
    pub fn update(&mut self, new_key: &Tensor, new_value: &Tensor) -> Result<(), ModelError> {
        let seq_len = new_key.dim(2).map_err(|e| ModelError::TensorError(e.to_string()))?;

        // Update key cache
        self.key = self.key
            .slice_set(new_key, 2, self.position)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        // Update value cache
        self.value = self.value
            .slice_set(new_value, 2, self.position)
            .map_err(|e| ModelError::TensorError(e.to_string()))?;

        self.position += seq_len;
        Ok(())
    }

    /// Get cached keys up to current position.
    pub fn get_key(&self) -> Result<Tensor, ModelError> {
        self.key
            .narrow(2, 0, self.position)
            .map_err(|e| ModelError::TensorError(e.to_string()))
    }

    /// Get cached values up to current position.
    pub fn get_value(&self) -> Result<Tensor, ModelError> {
        self.value
            .narrow(2, 0, self.position)
            .map_err(|e| ModelError::TensorError(e.to_string()))
    }

    /// Reset cache.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Current position.
    pub fn position(&self) -> usize {
        self.position
    }
}

/// Model errors.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Tensor error: {0}")]
    TensorError(String),

    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Missing weight: {0}")]
    MissingWeight(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Unsupported architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("Unsupported dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Cache error: {0}")]
    CacheError(String),
}

impl From<candle_core::Error> for ModelError {
    fn from(e: candle_core::Error) -> Self {
        ModelError::TensorError(e.to_string())
    }
}

impl From<LoaderError> for ModelError {
    fn from(e: LoaderError) -> Self {
        ModelError::TensorError(e.to_string())
    }
}

/// Load model from directory.
pub fn load_model(
    model_dir: &str,
    device: &Device,
    dtype: DType,
) -> Result<Arc<dyn Model>, ModelError> {
    use crate::loader::{LoaderConfig, ModelLoader};

    let loader_config = LoaderConfig::new(device.clone(), dtype);
    let loader = ModelLoader::new(model_dir, loader_config)?;

    let model_config = loader
        .model_config()
        .ok_or_else(|| ModelError::ConfigError("No model config found".to_string()))?
        .clone();

    let vb = loader.load_weights()?;

    match model_config.architecture() {
        Some(ModelArchitecture::Llama) | Some(ModelArchitecture::Llama2) | Some(ModelArchitecture::Llama3) => {
            let model = LlamaModel::new(&model_config, vb)?;
            Ok(Arc::new(model))
        }
        Some(ModelArchitecture::Mistral) => {
            let model = MistralModel::new(&model_config, vb)?;
            Ok(Arc::new(model))
        }
        Some(arch) => Err(ModelError::UnsupportedArchitecture(format!("{:?}", arch))),
        None => Err(ModelError::UnsupportedArchitecture("Unknown".to_string())),
    }
}
