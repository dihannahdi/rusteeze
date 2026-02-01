//! Parallel sampling using rayon for multi-sequence batches.
//!
//! This module provides parallel token sampling for batched inference,
//! leveraging rayon for CPU-bound sampling operations.
//!
//! # Design
//!
//! - Parallel processing of independent sequences in a batch
//! - Work stealing for load balancing across sequences
//! - SIMD operations integrated with parallel processing
//! - Adaptive parallelism based on batch size
//!
//! # Performance Benefits
//!
//! - Linear scaling with CPU cores for large batches
//! - Reduced latency through parallel execution
//! - Efficient work distribution via rayon's work stealing

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::sampler::{SampleResult, Sampler, SamplingState};
use crate::simd_ops;
use rusteeze_core::SamplingParams;

/// Parallel sampler configuration.
#[derive(Debug, Clone)]
pub struct ParallelSamplerConfig {
    /// Minimum batch size for parallel processing.
    /// Below this threshold, sequential processing is used.
    pub parallel_threshold: usize,

    /// Number of threads for sampling.
    /// 0 means use rayon's default (usually number of CPU cores).
    pub num_threads: usize,

    /// Enable SIMD operations in parallel sampling.
    pub enable_simd: bool,

    /// Chunk size for work distribution.
    pub chunk_size: usize,
}

impl Default for ParallelSamplerConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: 4,
            num_threads: 0, // Use rayon default
            enable_simd: true,
            chunk_size: 8,
        }
    }
}

/// Parallel batch sampler.
pub struct ParallelSampler {
    /// Configuration.
    config: ParallelSamplerConfig,

    /// Base sampler for single-sequence operations.
    base_sampler: Sampler,

    /// Statistics.
    stats: ParallelSamplerStats,
}

/// Sampling statistics.
#[derive(Debug, Default)]
pub struct ParallelSamplerStats {
    /// Total samples processed.
    total_samples: AtomicU64,
    /// Samples processed in parallel.
    parallel_samples: AtomicU64,
    /// Samples processed sequentially.
    sequential_samples: AtomicU64,
}

impl ParallelSamplerStats {
    /// Get total samples.
    pub fn total_samples(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// Get parallel samples.
    pub fn parallel_samples(&self) -> u64 {
        self.parallel_samples.load(Ordering::Relaxed)
    }

    /// Get sequential samples.
    pub fn sequential_samples(&self) -> u64 {
        self.sequential_samples.load(Ordering::Relaxed)
    }

    /// Get parallel ratio.
    pub fn parallel_ratio(&self) -> f64 {
        let total = self.total_samples() as f64;
        if total == 0.0 {
            return 0.0;
        }
        self.parallel_samples() as f64 / total
    }
}

/// Input for parallel sampling.
#[derive(Debug)]
pub struct ParallelSampleInput {
    /// Sequence index in batch.
    pub seq_idx: usize,
    /// Logits for this sequence (vocab_size).
    pub logits: Vec<f32>,
    /// Sampling parameters.
    pub params: SamplingParams,
    /// Sampling state (for stateful samplers).
    pub state: Option<SamplingState>,
    /// Previously generated tokens (for penalties).
    pub prev_tokens: Vec<u32>,
}

/// Output from parallel sampling.
#[derive(Debug)]
pub struct ParallelSampleOutput {
    /// Sequence index in batch.
    pub seq_idx: usize,
    /// Sample result.
    pub result: SampleResult,
    /// Updated sampling state.
    pub new_state: Option<SamplingState>,
}

impl ParallelSampler {
    /// Create a new parallel sampler.
    pub fn new(config: ParallelSamplerConfig) -> Self {
        // Configure rayon thread pool if specified
        if config.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.num_threads)
                .build_global()
                .ok(); // Ignore if already initialized
        }

        Self {
            config,
            base_sampler: Sampler::new(),
            stats: ParallelSamplerStats::default(),
        }
    }

    /// Sample from a batch of logits in parallel.
    pub fn sample_batch(&self, inputs: Vec<ParallelSampleInput>) -> Vec<ParallelSampleOutput> {
        let batch_size = inputs.len();
        
        self.stats.total_samples.fetch_add(batch_size as u64, Ordering::Relaxed);

        if batch_size < self.config.parallel_threshold {
            // Sequential processing for small batches
            self.stats.sequential_samples.fetch_add(batch_size as u64, Ordering::Relaxed);
            return self.sample_sequential(inputs);
        }

        // Parallel processing
        self.stats.parallel_samples.fetch_add(batch_size as u64, Ordering::Relaxed);
        self.sample_parallel(inputs)
    }

    /// Sample sequentially (for small batches).
    fn sample_sequential(&self, inputs: Vec<ParallelSampleInput>) -> Vec<ParallelSampleOutput> {
        inputs
            .into_iter()
            .map(|input| self.sample_single(input))
            .collect()
    }

    /// Sample in parallel (for large batches).
    fn sample_parallel(&self, inputs: Vec<ParallelSampleInput>) -> Vec<ParallelSampleOutput> {
        inputs
            .into_par_iter()
            .map(|input| self.sample_single(input))
            .collect()
    }

    /// Sample a single sequence.
    fn sample_single(&self, input: ParallelSampleInput) -> ParallelSampleOutput {
        let mut logits = input.logits;
        let params = &input.params;
        let vocab_size = logits.len();

        // Apply repetition penalty
        if params.repetition_penalty != 1.0 && !input.prev_tokens.is_empty() {
            if self.config.enable_simd {
                simd_ops::apply_repetition_penalty(
                    &mut logits,
                    &input.prev_tokens,
                    params.repetition_penalty,
                );
            } else {
                for &token in &input.prev_tokens {
                    if (token as usize) < vocab_size {
                        let logit = &mut logits[token as usize];
                        if *logit > 0.0 {
                            *logit /= params.repetition_penalty;
                        } else {
                            *logit *= params.repetition_penalty;
                        }
                    }
                }
            }
        }

        // Apply temperature
        if params.temperature > 0.0 && params.temperature != 1.0 {
            if self.config.enable_simd {
                simd_ops::scale_logits_inplace(&mut logits, 1.0 / params.temperature);
            } else {
                let inv_temp = 1.0 / params.temperature;
                for logit in &mut logits {
                    *logit *= inv_temp;
                }
            }
        }

        // Sample based on strategy
        let (token_id, logprob) = if params.temperature == 0.0 {
            // Greedy decoding
            self.sample_greedy(&logits)
        } else if params.top_p < 1.0 {
            // Nucleus (top-p) sampling
            self.sample_nucleus(&mut logits, params.top_p)
        } else if params.top_k > 0 {
            // Top-k sampling
            self.sample_top_k(&mut logits, params.top_k)
        } else {
            // Full distribution sampling
            self.sample_full(&mut logits)
        };

        ParallelSampleOutput {
            seq_idx: input.seq_idx,
            result: SampleResult {
                token_id,
                logprob,
                top_logprobs: None,
            },
            new_state: input.state,
        }
    }

    /// Greedy decoding (argmax).
    fn sample_greedy(&self, logits: &[f32]) -> (u32, f32) {
        if self.config.enable_simd {
            let (max_val, max_idx) = simd_ops::simd_argmax(logits);
            (max_idx as u32, max_val)
        } else {
            let (max_idx, max_val) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, &v)| (i, v))
                .unwrap_or((0, f32::NEG_INFINITY));
            (max_idx as u32, max_val)
        }
    }

    /// Top-k sampling.
    fn sample_top_k(&self, logits: &mut [f32], k: usize) -> (u32, f32) {
        let k = k.min(logits.len());
        
        // Get top-k indices and values
        let top_k = if self.config.enable_simd {
            simd_ops::fast_topk(logits, k)
        } else {
            let mut indexed: Vec<_> = logits.iter().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            indexed.into_iter()
                .take(k)
                .map(|(i, &v)| (i, v))
                .collect()
        };

        // Apply softmax to top-k
        let max_val = top_k.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = top_k.iter().map(|(_, v)| (*v - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Sample
        let r: f32 = fastrand::f32();
        let mut cumsum = 0.0;
        for (i, &(idx, val)) in top_k.iter().enumerate() {
            cumsum += probs[i];
            if r <= cumsum {
                return (idx as u32, val.ln() - sum.ln());
            }
        }

        // Fallback to last
        let (idx, val) = top_k.last().unwrap();
        (*idx as u32, val.ln() - sum.ln())
    }

    /// Nucleus (top-p) sampling.
    fn sample_nucleus(&self, logits: &mut [f32], p: f32) -> (u32, f32) {
        let vocab_size = logits.len();

        // Convert to probabilities
        if self.config.enable_simd {
            simd_ops::simd_softmax_inplace(logits);
        } else {
            let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for logit in logits.iter_mut() {
                *logit = (*logit - max_val).exp();
            }
            let sum: f32 = logits.iter().sum();
            for logit in logits.iter_mut() {
                *logit /= sum;
            }
        }

        // Prepare nucleus sampling
        let (sorted_indices, sorted_probs, cumsum) = if self.config.enable_simd {
            simd_ops::prepare_nucleus_sampling(logits)
        } else {
            let mut indexed: Vec<_> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            
            let sorted_indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
            let sorted_probs: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
            
            let mut cumsum = Vec::with_capacity(sorted_probs.len());
            let mut sum = 0.0;
            for &prob in &sorted_probs {
                sum += prob;
                cumsum.push(sum);
            }
            
            (sorted_indices, sorted_probs, cumsum)
        };

        // Find cutoff
        let cutoff_idx = cumsum.iter().position(|&c| c >= p).unwrap_or(vocab_size - 1);
        let nucleus_size = cutoff_idx + 1;

        // Renormalize nucleus
        let nucleus_sum = cumsum[cutoff_idx];
        
        // Sample from nucleus
        let r: f32 = fastrand::f32() * nucleus_sum;
        let mut cumsum = 0.0;
        for i in 0..nucleus_size {
            cumsum += sorted_probs[i];
            if r <= cumsum {
                let idx = sorted_indices[i];
                let logprob = (sorted_probs[i] / nucleus_sum).ln();
                return (idx as u32, logprob);
            }
        }

        // Fallback
        let idx = sorted_indices[0];
        let logprob = (sorted_probs[0] / nucleus_sum).ln();
        (idx as u32, logprob)
    }

    /// Full distribution sampling.
    fn sample_full(&self, logits: &mut [f32]) -> (u32, f32) {
        // Convert to probabilities
        if self.config.enable_simd {
            simd_ops::simd_softmax_inplace(logits);
        } else {
            let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            for logit in logits.iter_mut() {
                *logit = (*logit - max_val).exp();
            }
            let sum: f32 = logits.iter().sum();
            for logit in logits.iter_mut() {
                *logit /= sum;
            }
        }

        // Sample
        let r: f32 = fastrand::f32();
        let mut cumsum = 0.0;
        for (i, &prob) in logits.iter().enumerate() {
            cumsum += prob;
            if r <= cumsum {
                return (i as u32, prob.ln());
            }
        }

        // Fallback to last
        ((logits.len() - 1) as u32, logits.last().unwrap().ln())
    }

    /// Get statistics.
    pub fn stats(&self) -> &ParallelSamplerStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.stats.total_samples.store(0, Ordering::Relaxed);
        self.stats.parallel_samples.store(0, Ordering::Relaxed);
        self.stats.sequential_samples.store(0, Ordering::Relaxed);
    }
}

/// Batch sampling utilities.
pub mod batch_utils {
    use super::*;

    /// Prepare batch inputs from raw logits tensor.
    pub fn prepare_batch_inputs(
        batch_logits: &[f32],  // [batch_size, vocab_size]
        batch_size: usize,
        vocab_size: usize,
        params: &[SamplingParams],
        prev_tokens: &[Vec<u32>],
    ) -> Vec<ParallelSampleInput> {
        (0..batch_size)
            .map(|i| {
                let start = i * vocab_size;
                let end = start + vocab_size;
                let logits = batch_logits[start..end].to_vec();
                
                ParallelSampleInput {
                    seq_idx: i,
                    logits,
                    params: params.get(i).cloned().unwrap_or_default(),
                    state: None,
                    prev_tokens: prev_tokens.get(i).cloned().unwrap_or_default(),
                }
            })
            .collect()
    }

    /// Extract token IDs from batch outputs.
    pub fn extract_token_ids(outputs: &[ParallelSampleOutput]) -> Vec<u32> {
        let mut tokens = vec![0u32; outputs.len()];
        for output in outputs {
            tokens[output.seq_idx] = output.result.token_id;
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_random_logits(vocab_size: usize) -> Vec<f32> {
        (0..vocab_size).map(|_| fastrand::f32() * 10.0 - 5.0).collect()
    }

    #[test]
    fn test_parallel_sampler_greedy() {
        let config = ParallelSamplerConfig::default();
        let sampler = ParallelSampler::new(config);

        let mut logits = vec![0.1, 0.2, 0.5, 0.1, 0.1];
        logits[2] = 10.0; // Make index 2 the maximum

        let input = ParallelSampleInput {
            seq_idx: 0,
            logits,
            params: SamplingParams {
                temperature: 0.0, // Greedy
                ..Default::default()
            },
            state: None,
            prev_tokens: vec![],
        };

        let outputs = sampler.sample_batch(vec![input]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].result.token_id, 2);
    }

    #[test]
    fn test_parallel_sampler_batch() {
        let config = ParallelSamplerConfig {
            parallel_threshold: 2,
            ..Default::default()
        };
        let sampler = ParallelSampler::new(config);

        let inputs: Vec<_> = (0..10)
            .map(|i| {
                let mut logits = create_random_logits(100);
                logits[i] = 100.0; // Make each sequence select a different token
                
                ParallelSampleInput {
                    seq_idx: i,
                    logits,
                    params: SamplingParams {
                        temperature: 0.0,
                        ..Default::default()
                    },
                    state: None,
                    prev_tokens: vec![],
                }
            })
            .collect();

        let outputs = sampler.sample_batch(inputs);
        
        assert_eq!(outputs.len(), 10);
        for output in &outputs {
            // Each output should select its corresponding index
            assert_eq!(output.result.token_id as usize, output.seq_idx);
        }

        // Check statistics
        assert_eq!(sampler.stats().total_samples(), 10);
        assert_eq!(sampler.stats().parallel_samples(), 10);
    }

    #[test]
    fn test_sequential_for_small_batch() {
        let config = ParallelSamplerConfig {
            parallel_threshold: 10,
            ..Default::default()
        };
        let sampler = ParallelSampler::new(config);

        let inputs: Vec<_> = (0..5)
            .map(|i| ParallelSampleInput {
                seq_idx: i,
                logits: create_random_logits(100),
                params: SamplingParams::default(),
                state: None,
                prev_tokens: vec![],
            })
            .collect();

        sampler.sample_batch(inputs);
        
        // Should be sequential since batch_size < threshold
        assert_eq!(sampler.stats().sequential_samples(), 5);
        assert_eq!(sampler.stats().parallel_samples(), 0);
    }

    #[test]
    fn test_top_k_sampling() {
        let config = ParallelSamplerConfig::default();
        let sampler = ParallelSampler::new(config);

        let mut logits = vec![-100.0; 1000];
        // Make top 5 have positive values
        for i in 0..5 {
            logits[i] = 10.0 - i as f32;
        }

        let input = ParallelSampleInput {
            seq_idx: 0,
            logits,
            params: SamplingParams {
                temperature: 1.0,
                top_k: 5,
                ..Default::default()
            },
            state: None,
            prev_tokens: vec![],
        };

        let outputs = sampler.sample_batch(vec![input]);
        // Result should be in top 5
        assert!(outputs[0].result.token_id < 5);
    }

    #[test]
    fn test_nucleus_sampling() {
        let config = ParallelSamplerConfig::default();
        let sampler = ParallelSampler::new(config);

        let mut logits = vec![-100.0; 1000];
        // Make top tokens have most probability mass
        logits[0] = 10.0;
        logits[1] = 9.0;
        logits[2] = 8.0;

        let input = ParallelSampleInput {
            seq_idx: 0,
            logits,
            params: SamplingParams {
                temperature: 1.0,
                top_p: 0.9,
                ..Default::default()
            },
            state: None,
            prev_tokens: vec![],
        };

        let outputs = sampler.sample_batch(vec![input]);
        // Result should likely be in top 3 (where most probability mass is)
        assert!(outputs[0].result.token_id < 10);
    }

    #[test]
    fn test_batch_utils() {
        let batch_size = 4;
        let vocab_size = 100;
        let batch_logits: Vec<f32> = (0..batch_size * vocab_size)
            .map(|_| fastrand::f32())
            .collect();
        let params = vec![SamplingParams::default(); batch_size];
        let prev_tokens = vec![vec![]; batch_size];

        let inputs = batch_utils::prepare_batch_inputs(
            &batch_logits,
            batch_size,
            vocab_size,
            &params,
            &prev_tokens,
        );

        assert_eq!(inputs.len(), batch_size);
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(input.seq_idx, i);
            assert_eq!(input.logits.len(), vocab_size);
        }
    }
}
