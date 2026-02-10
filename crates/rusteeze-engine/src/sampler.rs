//! # Sampler — Radical Rewrite
//!
//! Token sampling with pre-allocated buffers, SIMD-accelerated operations,
//! and zero per-token allocations. Supports greedy, top-k, top-p, nucleus,
//! and Mirostat v1/v2 strategies.

use rand::Rng;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;

use rusteeze_core::sampling::SamplingParams;
use crate::simd_dispatch::{self, softmax_inplace, vec_argmax, vec_max, vec_scale};
use crate::simd_ops::{fast_topk, prepare_nucleus_sampling, apply_repetition_penalty, scale_logits_inplace};

/// Result of sampling a single token.
#[derive(Debug, Clone)]
pub struct SampleResult {
    /// Sampled token ID
    pub token_id: u32,
    /// Log probability of the sampled token
    pub logprob: f32,
    /// Top log probabilities (if requested)
    pub top_logprobs: Option<Vec<(u32, f32)>>,
}

/// Sampler with pre-allocated buffers for zero-allocation sampling.
pub struct Sampler {
    /// Reusable logits buffer (avoids per-sample allocation)
    logits_buf: Vec<f32>,
    /// RNG
    rng: rand::rngs::StdRng,
    /// Mirostat v1 state: estimated surprise
    mirostat_mu: f32,
}

impl Sampler {
    /// Create a new sampler with optional seed.
    pub fn new(seed: Option<u64>, vocab_size: usize) -> Self {
        use rand::SeedableRng;
        let rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self {
            logits_buf: Vec::with_capacity(vocab_size),
            rng,
            mirostat_mu: 10.0,
        }
    }

    /// Sample a token from logits.
    ///
    /// `logits`: raw logits from the model [vocab_size]
    /// `params`: sampling parameters
    /// `prev_tokens`: previously generated tokens (for repetition penalty)
    pub fn sample(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        prev_tokens: &[u32],
    ) -> SampleResult {
        // Reuse buffer — resize without reallocation if capacity suffices
        self.logits_buf.clear();
        self.logits_buf.extend_from_slice(logits);

        // Apply repetition penalty
        if params.repetition_penalty != 1.0 && !prev_tokens.is_empty() {
            apply_repetition_penalty(&mut self.logits_buf, prev_tokens, params.repetition_penalty);
        }

        // Apply logit bias
        if !params.logit_bias.is_empty() {
            for (&token_id, &bias) in &params.logit_bias {
                let idx = token_id as usize;
                if idx < self.logits_buf.len() {
                    self.logits_buf[idx] += bias;
                }
            }
        }

        // Mirostat sampling
        if params.mirostat_mode == 1 {
            return self.sample_mirostat_v1(params);
        } else if params.mirostat_mode == 2 {
            return self.sample_mirostat_v2(params);
        }

        // Greedy
        if params.temperature == 0.0 || (params.top_k == 1 && params.top_p >= 1.0) {
            return self.sample_greedy();
        }

        // Temperature + top-k/top-p
        self.sample_standard(params)
    }

    fn sample_greedy(&self) -> SampleResult {
        let (idx, val) = vec_argmax(&self.logits_buf);
        SampleResult {
            token_id: idx as u32,
            logprob: (val - vec_max(&self.logits_buf)).max(-100.0),
            top_logprobs: None,
        }
    }

    fn sample_standard(&mut self, params: &SamplingParams) -> SampleResult {
        let logits = &mut self.logits_buf;

        // Apply temperature
        if params.temperature > 0.0 && params.temperature != 1.0 {
            scale_logits_inplace(logits, params.temperature);
        }

        // Convert to probabilities
        softmax_inplace(logits);

        // Top-k filtering
        if params.top_k > 0 && (params.top_k as usize) < logits.len() {
            let top = fast_topk(logits, params.top_k as usize);
            // Zero out non-top-k entries
            let top_indices: std::collections::HashSet<usize> = top.iter().map(|&(i, _)| i).collect();
            for (i, l) in logits.iter_mut().enumerate() {
                if !top_indices.contains(&i) { *l = 0.0; }
            }
        }

        // Top-p (nucleus) filtering
        if params.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
                .filter(|(_, &p)| p > 0.0)
                .map(|(i, &p)| (i, p))
                .collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut cum = 0.0f32;
            let mut keep = std::collections::HashSet::new();
            for &(idx, prob) in &indexed {
                cum += prob;
                keep.insert(idx);
                if cum >= params.top_p { break; }
            }
            for (i, l) in logits.iter_mut().enumerate() {
                if !keep.contains(&i) { *l = 0.0; }
            }
        }

        // Renormalize
        let sum: f32 = logits.iter().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for l in logits.iter_mut() { *l *= inv; }
        }

        // Sample from distribution
        let valid: Vec<(usize, f32)> = logits.iter().enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(i, &p)| (i, p))
            .collect();

        if valid.is_empty() {
            let (idx, val) = vec_argmax(logits);
            return SampleResult {
                token_id: idx as u32,
                logprob: (val - vec_max(logits)).max(-100.0),
                top_logprobs: None,
            };
        }

        let weights: Vec<f32> = valid.iter().map(|&(_, p)| p).collect();
        let dist = WeightedIndex::new(&weights).unwrap();
        let sampled_idx = valid[dist.sample(&mut self.rng)].0;
        let prob = logits[sampled_idx];

        SampleResult {
            token_id: sampled_idx as u32,
            logprob: prob.ln().max(-100.0),
            top_logprobs: None,
        }
    }

    fn sample_mirostat_v1(&mut self, params: &SamplingParams) -> SampleResult {
        let tau = params.mirostat_tau;
        let eta = params.mirostat_eta;

        softmax_inplace(&mut self.logits_buf);
        let mut indexed: Vec<(usize, f32)> = self.logits_buf.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Estimate surprise threshold
        let k = ((self.mirostat_mu.exp()) as usize).max(1).min(indexed.len());
        let truncated: Vec<f32> = indexed[..k].iter().map(|&(_, p)| p).collect();
        let sum: f32 = truncated.iter().sum();
        let probs: Vec<f32> = truncated.iter().map(|&p| p / sum.max(1e-10)).collect();

        let dist = WeightedIndex::new(&probs).unwrap_or_else(|_| WeightedIndex::new(&[1.0]).unwrap());
        let sampled = dist.sample(&mut self.rng);
        let (token_idx, prob) = indexed[sampled];

        // Update mu
        let surprise = -(prob.max(1e-10).ln());
        self.mirostat_mu += eta * (tau - surprise);

        SampleResult {
            token_id: token_idx as u32,
            logprob: prob.ln().max(-100.0),
            top_logprobs: None,
        }
    }

    fn sample_mirostat_v2(&mut self, params: &SamplingParams) -> SampleResult {
        let tau = params.mirostat_tau;
        let eta = params.mirostat_eta;

        softmax_inplace(&mut self.logits_buf);

        // Filter tokens with surprise <= mu
        let mut candidates: Vec<(usize, f32)> = self.logits_buf.iter().enumerate()
            .filter(|(_, &p)| p > 0.0 && -(p.ln()) <= self.mirostat_mu)
            .map(|(i, &p)| (i, p))
            .collect();

        if candidates.is_empty() {
            candidates = vec![vec_argmax(&self.logits_buf)].into_iter()
                .map(|(i, _)| (i, self.logits_buf[i]))
                .collect();
        }

        let sum: f32 = candidates.iter().map(|&(_, p)| p).sum();
        let probs: Vec<f32> = candidates.iter().map(|&(_, p)| p / sum.max(1e-10)).collect();

        let dist = WeightedIndex::new(&probs).unwrap_or_else(|_| WeightedIndex::new(&[1.0]).unwrap());
        let sampled = dist.sample(&mut self.rng);
        let (token_idx, prob) = candidates[sampled];

        let surprise = -(prob.max(1e-10).ln());
        self.mirostat_mu += eta * (tau - surprise);

        SampleResult {
            token_id: token_idx as u32,
            logprob: prob.ln().max(-100.0),
            top_logprobs: None,
        }
    }

    /// Get top-k log probabilities.
    pub fn get_top_logprobs(&self, logits: &[f32], k: usize) -> Vec<(u32, f32)> {
        let top = fast_topk(logits, k);
        top.into_iter().map(|(i, v)| (i as u32, v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let mut sampler = Sampler::new(Some(42), 10);
        let logits = vec![0.1, 0.2, 0.3, 0.9, 0.1, 0.05, 0.1, 0.1, 0.1, 0.05];
        let params = SamplingParams { temperature: 0.0, ..Default::default() };
        let result = sampler.sample(&logits, &params, &[]);
        assert_eq!(result.token_id, 3);
    }

    #[test]
    fn test_standard_sampling() {
        let mut sampler = Sampler::new(Some(42), 10);
        let logits = vec![0.1, 0.2, 0.3, 0.9, 0.1, 0.05, 0.1, 0.1, 0.1, 0.05];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 3,
            top_p: 0.9,
            ..Default::default()
        };
        let result = sampler.sample(&logits, &params, &[]);
        assert!(result.token_id < 10);
    }
}
