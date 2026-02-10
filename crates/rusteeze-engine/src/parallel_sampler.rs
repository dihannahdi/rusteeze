//! # Parallel Sampler — Radical Rewrite
//!
//! Batch sampling with rayon parallelism over sequences, zero-copy logit
//! slicing (no `.to_vec()` per sequence), and pre-allocated scratch buffers.

use rayon::prelude::*;
use rand::SeedableRng;

use rusteeze_core::sampling::SamplingParams;
use crate::sampler::{SampleResult, Sampler};
use crate::simd_dispatch;

/// Configuration for the parallel sampler.
#[derive(Debug, Clone)]
pub struct ParallelSamplerConfig {
    /// Number of worker threads (0 = rayon default)
    pub num_threads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Minimum batch size to trigger parallel sampling
    pub parallel_threshold: usize,
}

impl Default for ParallelSamplerConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            vocab_size: 32000,
            parallel_threshold: 2,
        }
    }
}

/// Parallel sampler for batch token generation.
pub struct ParallelSampler {
    config: ParallelSamplerConfig,
}

impl ParallelSampler {
    /// Create a new parallel sampler.
    pub fn new(config: ParallelSamplerConfig) -> Self {
        simd_dispatch::init();
        Self { config }
    }

    /// Sample tokens for a batch of sequences.
    ///
    /// `batch_logits`: [batch_size * vocab_size] flattened
    /// `params`: per-sequence sampling parameters
    /// `prev_tokens`: per-sequence previous tokens
    ///
    /// Returns one SampleResult per sequence.
    pub fn sample_batch(
        &self,
        batch_logits: &[f32],
        params: &[SamplingParams],
        prev_tokens: &[&[u32]],
    ) -> Vec<SampleResult> {
        let vocab_size = self.config.vocab_size;
        let batch_size = params.len();

        if batch_size == 0 { return Vec::new(); }

        assert!(
            batch_logits.len() >= batch_size * vocab_size,
            "batch_logits too small: {} < {} * {}",
            batch_logits.len(), batch_size, vocab_size
        );

        if batch_size < self.config.parallel_threshold {
            // Sequential for small batches
            let mut results = Vec::with_capacity(batch_size);
            let mut sampler = Sampler::new(None, vocab_size);
            for i in 0..batch_size {
                let logits = &batch_logits[i * vocab_size..(i + 1) * vocab_size];
                let prev = if i < prev_tokens.len() { prev_tokens[i] } else { &[] };
                results.push(sampler.sample(logits, &params[i], prev));
            }
            results
        } else {
            // Parallel: each thread gets its own Sampler (with own RNG + buffer)
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    // Thread-local sampler with unique seed
                    let mut sampler = Sampler::new(
                        Some(i as u64 ^ 0xDEADBEEF),
                        vocab_size,
                    );
                    // Zero-copy: slice directly into batch_logits
                    let logits = &batch_logits[i * vocab_size..(i + 1) * vocab_size];
                    let prev = if i < prev_tokens.len() { prev_tokens[i] } else { &[] };
                    sampler.sample(logits, &params[i], prev)
                })
                .collect()
        }
    }

    /// Get vocab size.
    pub fn vocab_size(&self) -> usize { self.config.vocab_size }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_sampling() {
        let config = ParallelSamplerConfig {
            vocab_size: 10,
            parallel_threshold: 1,
            ..Default::default()
        };
        let sampler = ParallelSampler::new(config);

        let batch_size = 4;
        let vocab = 10;
        let logits: Vec<f32> = (0..batch_size * vocab)
            .map(|i| if i % vocab == 3 { 10.0 } else { 0.1 })
            .collect();

        let params = vec![SamplingParams { temperature: 0.0, ..Default::default() }; batch_size];
        let prev: Vec<&[u32]> = vec![&[]; batch_size];

        let results = sampler.sample_batch(&logits, &params, &prev);
        assert_eq!(results.len(), batch_size);
        for r in &results {
            assert_eq!(r.token_id, 3, "Should pick token 3 (highest logit)");
        }
    }
}
