//! Weight management and tensor utilities.
//!
//! This module provides utilities for managing model weights,
//! including weight conversion, quantization, and optimization.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tracing::{debug, info, warn};

use crate::loader::LoaderError;

/// Weight naming conventions for different model formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightNamingConvention {
    /// HuggingFace Transformers naming.
    HuggingFace,
    /// vLLM naming.
    Vllm,
    /// GPT-NeoX naming.
    GptNeox,
    /// Megatron naming.
    Megatron,
    /// GGML/GGUF naming.
    Ggml,
}

/// Weight conversion utilities.
pub struct WeightConverter {
    /// Source naming convention.
    source: WeightNamingConvention,

    /// Target naming convention.
    target: WeightNamingConvention,

    /// Custom mappings.
    mappings: HashMap<String, String>,
}

impl WeightConverter {
    /// Create a new converter.
    pub fn new(source: WeightNamingConvention, target: WeightNamingConvention) -> Self {
        Self {
            source,
            target,
            mappings: HashMap::new(),
        }
    }

    /// Add custom mapping.
    pub fn add_mapping(&mut self, source_name: &str, target_name: &str) {
        self.mappings
            .insert(source_name.to_string(), target_name.to_string());
    }

    /// Convert weight name.
    pub fn convert_name(&self, name: &str) -> String {
        // Check custom mappings first
        if let Some(mapped) = self.mappings.get(name) {
            return mapped.clone();
        }

        // Standard conversions
        match (self.source, self.target) {
            (WeightNamingConvention::HuggingFace, WeightNamingConvention::Vllm) => {
                Self::hf_to_vllm(name)
            }
            (WeightNamingConvention::Vllm, WeightNamingConvention::HuggingFace) => {
                Self::vllm_to_hf(name)
            }
            _ => name.to_string(),
        }
    }

    /// Convert HuggingFace naming to vLLM naming.
    fn hf_to_vllm(name: &str) -> String {
        let mut result = name.replace("model.", "");
        result = result.replace("self_attn.", "attn.");

        // Handle QKV projection fusion: q_proj, k_proj, v_proj -> qkv_proj
        // Must check BEFORE replacement to avoid cascading matches
        if result.contains("q_proj") || result.contains("k_proj") || result.contains("v_proj") {
            result = result
                .replace("q_proj", "qkv_proj")
                .replace("k_proj", "qkv_proj");
            // v_proj must be replaced carefully since "qkv_proj" contains "v_proj"
            // Only replace standalone v_proj (not the one inside qkv_proj)
            if !result.contains("qkv_proj") {
                result = result.replace("v_proj", "qkv_proj");
            }
        }

        // Handle gate/up projection fusion
        if result.contains("mlp.gate_proj") || result.contains("mlp.up_proj") {
            if result.contains("mlp.gate_proj") {
                result = result.replace("mlp.gate_proj", "mlp.gate_up_proj");
            } else {
                result = result.replace("mlp.up_proj", "mlp.gate_up_proj");
            }
        }

        result
    }

    /// Convert vLLM naming to HuggingFace naming.
    fn vllm_to_hf(name: &str) -> String {
        // Inverse of hf_to_vllm
        name.replace("attn.", "self_attn.")
    }
}

/// Weight statistics.
#[derive(Debug, Clone)]
pub struct WeightStats {
    /// Weight name.
    pub name: String,

    /// Shape.
    pub shape: Vec<usize>,

    /// Data type.
    pub dtype: DType,

    /// Number of parameters.
    pub num_params: usize,

    /// Size in bytes.
    pub size_bytes: usize,

    /// Min value (if computed).
    pub min: Option<f64>,

    /// Max value (if computed).
    pub max: Option<f64>,

    /// Mean value (if computed).
    pub mean: Option<f64>,

    /// Standard deviation (if computed).
    pub std: Option<f64>,
}

impl WeightStats {
    /// Compute stats from tensor.
    pub fn from_tensor(name: &str, tensor: &Tensor) -> Result<Self, candle_core::Error> {
        let shape = tensor.dims().to_vec();
        let dtype = tensor.dtype();
        let num_params: usize = shape.iter().product();
        let size_bytes = num_params * dtype.size_in_bytes();

        // Compute statistics (expensive, only do when needed)
        let (min, max, mean, std) = if num_params < 1_000_000 {
            let tensor_f32 = tensor.to_dtype(DType::F32)?;
            let data = tensor_f32.flatten_all()?.to_vec1::<f32>()?;

            let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = data.iter().sum();
            let mean = sum / data.len() as f32;
            let variance: f32 =
                data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
            let std = variance.sqrt();

            (
                Some(min as f64),
                Some(max as f64),
                Some(mean as f64),
                Some(std as f64),
            )
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            name: name.to_string(),
            shape,
            dtype,
            num_params,
            size_bytes,
            min,
            max,
            mean,
            std,
        })
    }
}

/// Weight manager for model weights.
pub struct WeightManager {
    /// Weights by name.
    weights: HashMap<String, Tensor>,

    /// Device.
    device: Device,

    /// Default dtype.
    dtype: DType,
}

impl WeightManager {
    /// Create a new weight manager.
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            weights: HashMap::new(),
            device,
            dtype,
        }
    }

    /// Create from VarBuilder.
    pub fn from_var_builder(vb: &VarBuilder) -> Result<Self, LoaderError> {
        // VarBuilder doesn't expose weights directly, so we create empty manager
        Ok(Self {
            weights: HashMap::new(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Add weight.
    pub fn add(&mut self, name: &str, tensor: Tensor) {
        self.weights.insert(name.to_string(), tensor);
    }

    /// Get weight.
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    /// Get weight or error.
    pub fn get_required(&self, name: &str) -> Result<&Tensor, LoaderError> {
        self.get(name)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))
    }

    /// Check if weight exists.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Get all weight names.
    pub fn names(&self) -> Vec<&str> {
        self.weights.keys().map(|s| s.as_str()).collect()
    }

    /// Get total parameter count.
    pub fn total_params(&self) -> usize {
        self.weights
            .values()
            .map(|t| t.dims().iter().product::<usize>())
            .sum()
    }

    /// Get total size in bytes.
    pub fn total_size(&self) -> usize {
        self.weights
            .values()
            .map(|t| t.dims().iter().product::<usize>() * t.dtype().size_in_bytes())
            .sum()
    }

    /// Convert all weights to dtype.
    pub fn convert_dtype(&mut self, dtype: DType) -> Result<(), LoaderError> {
        for (name, tensor) in self.weights.iter_mut() {
            *tensor = tensor
                .to_dtype(dtype)
                .map_err(|e| LoaderError::TensorError(e.to_string()))?;
        }
        self.dtype = dtype;
        Ok(())
    }

    /// Move all weights to device.
    pub fn to_device(&mut self, device: &Device) -> Result<(), LoaderError> {
        for (name, tensor) in self.weights.iter_mut() {
            *tensor = tensor
                .to_device(device)
                .map_err(|e| LoaderError::TensorError(e.to_string()))?;
        }
        self.device = device.clone();
        Ok(())
    }

    /// Get weight statistics.
    pub fn get_stats(&self) -> Result<Vec<WeightStats>, LoaderError> {
        let mut stats = Vec::new();
        for (name, tensor) in &self.weights {
            let stat = WeightStats::from_tensor(name, tensor)
                .map_err(|e| LoaderError::TensorError(e.to_string()))?;
            stats.push(stat);
        }
        Ok(stats)
    }
}

/// Fuse QKV projections for efficient attention.
pub fn fuse_qkv_weights(
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
) -> Result<Tensor, LoaderError> {
    // Concatenate along first dimension
    Tensor::cat(&[q_weight, k_weight, v_weight], 0)
        .map_err(|e| LoaderError::TensorError(e.to_string()))
}

/// Fuse gate and up projections for efficient MLP.
pub fn fuse_gate_up_weights(
    gate_weight: &Tensor,
    up_weight: &Tensor,
) -> Result<Tensor, LoaderError> {
    // Concatenate along first dimension
    Tensor::cat(&[gate_weight, up_weight], 0)
        .map_err(|e| LoaderError::TensorError(e.to_string()))
}

/// Split fused QKV weights.
pub fn split_qkv_weights(
    fused: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor), LoaderError> {
    let q_size = num_heads * head_dim;
    let k_size = num_kv_heads * head_dim;
    let v_size = num_kv_heads * head_dim;

    let q = fused
        .narrow(0, 0, q_size)
        .map_err(|e| LoaderError::TensorError(e.to_string()))?;
    let k = fused
        .narrow(0, q_size, k_size)
        .map_err(|e| LoaderError::TensorError(e.to_string()))?;
    let v = fused
        .narrow(0, q_size + k_size, v_size)
        .map_err(|e| LoaderError::TensorError(e.to_string()))?;

    Ok((q, k, v))
}

/// Weight shard for tensor parallelism.
#[derive(Debug, Clone)]
pub struct WeightShard {
    /// Shard index.
    pub index: usize,

    /// Total shards.
    pub total: usize,

    /// Shard start offset.
    pub offset: usize,

    /// Shard size.
    pub size: usize,
}

impl WeightShard {
    /// Create shard info.
    pub fn new(index: usize, total: usize, tensor_dim: usize) -> Self {
        let base_size = tensor_dim / total;
        let remainder = tensor_dim % total;

        let offset = index * base_size + index.min(remainder);
        let size = base_size + if index < remainder { 1 } else { 0 };

        Self {
            index,
            total,
            offset,
            size,
        }
    }

    /// Get slice range.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.size
    }
}

/// Shard tensor along dimension.
pub fn shard_tensor(
    tensor: &Tensor,
    dim: usize,
    shard: &WeightShard,
) -> Result<Tensor, LoaderError> {
    tensor
        .narrow(dim, shard.offset, shard.size)
        .map_err(|e| LoaderError::TensorError(e.to_string()))
}

/// Gather sharded tensors.
pub fn gather_shards(
    shards: &[Tensor],
    dim: usize,
) -> Result<Tensor, LoaderError> {
    Tensor::cat(shards, dim).map_err(|e| LoaderError::TensorError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_converter() {
        let converter =
            WeightConverter::new(WeightNamingConvention::HuggingFace, WeightNamingConvention::Vllm);

        assert_eq!(
            converter.convert_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.attn.qkv_proj.weight"
        );
    }

    #[test]
    fn test_weight_shard() {
        let shard = WeightShard::new(0, 4, 16);
        assert_eq!(shard.offset, 0);
        assert_eq!(shard.size, 4);

        let shard = WeightShard::new(1, 4, 16);
        assert_eq!(shard.offset, 4);
        assert_eq!(shard.size, 4);
    }

    #[test]
    fn test_weight_shard_uneven() {
        // 10 elements, 3 shards -> 4, 3, 3
        let shard0 = WeightShard::new(0, 3, 10);
        let shard1 = WeightShard::new(1, 3, 10);
        let shard2 = WeightShard::new(2, 3, 10);

        assert_eq!(shard0.size, 4);
        assert_eq!(shard1.size, 3);
        assert_eq!(shard2.size, 3);
    }
}
