//! Worker for model execution.
//!
//! The worker executes model inference for scheduled batches.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use rusteeze_model::architectures::{KVCache, Model};

use crate::batch::BatchInput;
use crate::sampler::{SampleResult, Sampler, SamplerError};
use crate::scheduler::SchedulerOutput;
use crate::sequence::SequenceId;

/// Worker configuration.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Device to run on.
    pub device: Device,

    /// Data type for computation.
    pub dtype: DType,

    /// Random seed for sampling.
    pub seed: Option<u64>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F16,
            seed: None,
        }
    }
}

/// Execution result for a batch.
#[derive(Debug)]
pub struct ExecuteResult {
    /// Sampled tokens per sequence.
    pub outputs: Vec<(SequenceId, SampleResult)>,
}

/// Model worker for executing inference.
pub struct Worker {
    /// Configuration.
    config: WorkerConfig,

    /// Model.
    model: Arc<dyn Model>,

    /// KV cache per sequence.
    kv_caches: Vec<Option<KVCache>>,

    /// Token sampler.
    sampler: Sampler,
}

impl Worker {
    /// Create new worker.
    pub fn new(config: WorkerConfig, model: Arc<dyn Model>) -> Self {
        let sampler = Sampler::new(config.seed);

        // Initialize KV caches placeholder
        let kv_caches = Vec::new();

        info!(
            "Initialized worker: device={:?}, dtype={:?}",
            config.device, config.dtype
        );

        Self {
            config,
            model,
            kv_caches,
            sampler,
        }
    }

    /// Execute batch.
    pub fn execute(&mut self, input: &BatchInput) -> Result<ExecuteResult, WorkerError> {
        // Build input tensors
        let input_ids = self.build_input_tensor(&input.token_ids)?;
        let position_ids = self.build_position_tensor(&input.position_ids)?;

        // Forward pass
        let logits = self.model
            .forward(&input_ids, &position_ids, None)
            .map_err(|e| WorkerError::ModelError(e.to_string()))?;

        // Sample tokens
        let mut outputs = Vec::new();

        for (idx, (seq_id, params)) in input.seq_info.iter().enumerate() {
            // Get logits for this sequence
            let seq_logits = if input.seq_info.len() > 1 {
                // Batch dimension
                logits
                    .narrow(0, idx, 1)
                    .map_err(|e| WorkerError::TensorError(e.to_string()))?
            } else {
                logits.clone()
            };

            // Sample
            let result = self.sampler
                .sample(&seq_logits, *seq_id, params)
                .map_err(|e| WorkerError::SamplerError(e))?;

            outputs.push((*seq_id, result));
        }

        Ok(ExecuteResult { outputs })
    }

    /// Build input tensor from token IDs.
    fn build_input_tensor(&self, token_ids: &[Vec<u32>]) -> Result<Tensor, WorkerError> {
        if token_ids.len() == 1 {
            // Single sequence
            Tensor::new(token_ids[0].as_slice(), &self.config.device)
                .map_err(|e| WorkerError::TensorError(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| WorkerError::TensorError(e.to_string()))
        } else {
            // Batch - need to pad
            let max_len = token_ids.iter().map(|t| t.len()).max().unwrap_or(0);
            let batch_size = token_ids.len();

            let mut padded = vec![0u32; batch_size * max_len];
            for (i, tokens) in token_ids.iter().enumerate() {
                for (j, &t) in tokens.iter().enumerate() {
                    padded[i * max_len + j] = t;
                }
            }

            Tensor::from_vec(padded, (batch_size, max_len), &self.config.device)
                .map_err(|e| WorkerError::TensorError(e.to_string()))
        }
    }

    /// Build position tensor.
    fn build_position_tensor(&self, position_ids: &[Vec<u32>]) -> Result<Tensor, WorkerError> {
        if position_ids.len() == 1 {
            Tensor::new(position_ids[0].as_slice(), &self.config.device)
                .map_err(|e| WorkerError::TensorError(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| WorkerError::TensorError(e.to_string()))
        } else {
            let max_len = position_ids.iter().map(|p| p.len()).max().unwrap_or(0);
            let batch_size = position_ids.len();

            let mut padded = vec![0u32; batch_size * max_len];
            for (i, positions) in position_ids.iter().enumerate() {
                for (j, &p) in positions.iter().enumerate() {
                    padded[i * max_len + j] = p;
                }
            }

            Tensor::from_vec(padded, (batch_size, max_len), &self.config.device)
                .map_err(|e| WorkerError::TensorError(e.to_string()))
        }
    }

    /// Get device.
    pub fn device(&self) -> &Device {
        &self.config.device
    }

    /// Get dtype.
    pub fn dtype(&self) -> DType {
        self.config.dtype
    }

    /// Reset sampler state for sequence.
    pub fn reset_sampler(&mut self, seq_id: &SequenceId) {
        self.sampler.reset_mirostat(seq_id);
    }
}

/// Worker errors.
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Tensor error: {0}")]
    TensorError(String),

    #[error("Sampler error: {0}")]
    SamplerError(#[from] SamplerError),

    #[error("Cache error: {0}")]
    CacheError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.dtype, DType::F16);
    }
}
