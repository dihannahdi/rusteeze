//! # Worker — Radical Rewrite
//!
//! Worker thread for model execution with batched sampling,
//! pre-allocated buffers, and pipeline-parallel support.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

use crate::parallel_sampler::{ParallelSampler, ParallelSamplerConfig};
use crate::simd_dispatch;
use rusteeze_core::sampling::SamplingParams;

/// Worker configuration.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Worker ID
    pub worker_id: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            worker_id: 0,
            vocab_size: 32000,
            max_batch_size: 64,
            max_seq_len: 4096,
        }
    }
}

/// Worker execution stats (atomic).
#[derive(Debug, Default)]
pub struct WorkerStats {
    pub iterations: AtomicU64,
    pub tokens_generated: AtomicU64,
    pub prefill_tokens: AtomicU64,
}

impl Clone for WorkerStats {
    fn clone(&self) -> Self {
        Self {
            iterations: AtomicU64::new(self.iterations.load(Ordering::Relaxed)),
            tokens_generated: AtomicU64::new(self.tokens_generated.load(Ordering::Relaxed)),
            prefill_tokens: AtomicU64::new(self.prefill_tokens.load(Ordering::Relaxed)),
        }
    }
}

/// A batch of work for the worker.
#[derive(Debug, Clone)]
pub struct WorkBatch {
    /// Sequence IDs
    pub seq_ids: Vec<u64>,
    /// Input tokens (flattened)
    pub input_tokens: Vec<u32>,
    /// Sampling parameters per sequence
    pub sampling_params: Vec<SamplingParams>,
    /// Previous tokens per sequence (for repetition penalty)
    pub prev_tokens: Vec<Vec<u32>>,
    /// Whether each sequence is prefill
    pub is_prefill: Vec<bool>,
}

/// Result of processing a batch.
#[derive(Debug, Clone)]
pub struct WorkResult {
    /// Generated tokens per sequence
    pub tokens: Vec<(u64, u32)>,
    /// Log probabilities per sequence
    pub logprobs: Vec<(u64, f32)>,
}

/// Model execution worker.
pub struct Worker {
    config: WorkerConfig,
    /// Batched sampler (uses rayon internally)
    sampler: ParallelSampler,
    /// Pre-allocated logits buffer [max_batch_size * vocab_size]
    logits_buffer: Vec<f32>,
    /// Pre-allocated result buffer
    result_buffer: WorkResult,
    /// Stats
    stats: WorkerStats,
    /// Active flag
    active: AtomicBool,
}

impl Worker {
    /// Create a new worker.
    pub fn new(config: WorkerConfig) -> Self {
        simd_dispatch::init();
        let vocab = config.vocab_size;
        let max_batch = config.max_batch_size;

        let sampler = ParallelSampler::new(ParallelSamplerConfig {
            vocab_size: vocab,
            parallel_threshold: 2,
            ..Default::default()
        });

        Self {
            config: config.clone(),
            sampler,
            logits_buffer: vec![0.0f32; max_batch * vocab],
            result_buffer: WorkResult {
                tokens: Vec::with_capacity(max_batch),
                logprobs: Vec::with_capacity(max_batch),
            },
            stats: WorkerStats::default(),
            active: AtomicBool::new(true),
        }
    }

    /// Execute one step: run model forward + sample.
    ///
    /// `model_forward`: closure that fills `logits_buffer` from model
    pub fn step<F>(&mut self, batch: &WorkBatch, model_forward: F) -> &WorkResult
    where
        F: FnOnce(&[u32], &mut [f32]),
    {
        let batch_size = batch.seq_ids.len();
        let vocab = self.config.vocab_size;

        // Ensure logits buffer is large enough
        let needed = batch_size * vocab;
        if self.logits_buffer.len() < needed {
            self.logits_buffer.resize(needed, 0.0);
        }

        // Model forward pass — fills logits_buffer
        model_forward(&batch.input_tokens, &mut self.logits_buffer[..needed]);

        // Batched sampling
        let prev_refs: Vec<&[u32]> = batch.prev_tokens.iter()
            .map(|v| v.as_slice())
            .collect();

        let results = self.sampler.sample_batch(
            &self.logits_buffer[..needed],
            &batch.sampling_params,
            &prev_refs,
        );

        // Fill result buffer (pre-allocated)
        self.result_buffer.tokens.clear();
        self.result_buffer.logprobs.clear();

        for (i, result) in results.into_iter().enumerate() {
            let seq_id = batch.seq_ids[i];
            self.result_buffer.tokens.push((seq_id, result.token_id));
            self.result_buffer.logprobs.push((seq_id, result.logprob));
        }

        // Update stats
        self.stats.iterations.fetch_add(1, Ordering::Relaxed);
        self.stats.tokens_generated.fetch_add(batch_size as u64, Ordering::Relaxed);

        &self.result_buffer
    }

    /// Get stats.
    pub fn stats(&self) -> &WorkerStats { &self.stats }

    /// Check if active.
    pub fn is_active(&self) -> bool { self.active.load(Ordering::Relaxed) }

    /// Shutdown.
    pub fn shutdown(&self) { self.active.store(false, Ordering::Release); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_step() {
        let config = WorkerConfig { vocab_size: 10, max_batch_size: 4, ..Default::default() };
        let mut worker = Worker::new(config);

        let batch = WorkBatch {
            seq_ids: vec![1, 2],
            input_tokens: vec![5, 10],
            sampling_params: vec![
                SamplingParams { temperature: 0.0, ..Default::default() },
                SamplingParams { temperature: 0.0, ..Default::default() },
            ],
            prev_tokens: vec![vec![], vec![]],
            is_prefill: vec![false, false],
        };

        let result = worker.step(&batch, |_input, logits| {
            // Fake model: token 3 gets highest logit for all sequences
            for chunk in logits.chunks_mut(10) {
                for (i, v) in chunk.iter_mut().enumerate() {
                    *v = if i == 3 { 10.0 } else { 0.1 };
                }
            }
        });

        assert_eq!(result.tokens.len(), 2);
        assert_eq!(result.tokens[0].1, 3);
        assert_eq!(result.tokens[1].1, 3);
    }
}
