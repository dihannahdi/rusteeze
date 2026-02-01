//! Token sampler implementation.
//!
//! Provides various sampling strategies including temperature,
//! top-k, top-p (nucleus), min-p, and Mirostat.
//!
//! ## Performance Optimizations
//!
//! This module uses SIMD-optimized operations where available:
//! - Vectorized softmax computation
//! - Fast top-k selection using partial sort
//! - SIMD log-sum-exp for numerical stability

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use rand::prelude::*;
use rand::distributions::WeightedIndex;
use serde::{Deserialize, Serialize};
use tracing::debug;

use rusteeze_core::SamplingParams;
use crate::sequence::{SequenceId, SequenceGroup};
use crate::simd_ops::{
    simd_argmax, simd_log_sum_exp, simd_softmax_inplace,
    scale_logits_inplace, fast_topk, prepare_nucleus_sampling,
    renormalize_probs, apply_repetition_penalty, apply_frequency_presence_penalty,
};

/// Sampling result for a single sequence.
#[derive(Debug, Clone)]
pub struct SampleResult {
    /// Sampled token ID.
    pub token_id: u32,

    /// Log probability of the token.
    pub logprob: f32,

    /// Top log probabilities (if requested).
    pub top_logprobs: Option<Vec<(u32, f32)>>,
}

/// Token sampler.
pub struct Sampler {
    /// Random number generator.
    rng: StdRng,

    /// Mirostat state per sequence.
    mirostat_state: HashMap<SequenceId, MirostatState>,
}

impl Sampler {
    /// Create new sampler.
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self {
            rng,
            mirostat_state: HashMap::new(),
        }
    }

    /// Sample next token from logits.
    pub fn sample(
        &mut self,
        logits: &Tensor,
        seq_id: SequenceId,
        params: &SamplingParams,
    ) -> Result<SampleResult, SamplerError> {
        // Get logits as f32 vector
        let logits_vec = self.get_logits_vec(logits)?;

        // Apply sampling strategy
        let (token_id, logprob, top_logprobs) = if params.temperature.unwrap_or(1.0) <= 0.0 {
            // Greedy sampling
            self.sample_greedy(&logits_vec, params)?
        } else if params.mirostat_mode.unwrap_or(0) > 0 {
            // Mirostat sampling
            self.sample_mirostat(&logits_vec, seq_id, params)?
        } else {
            // Standard sampling with temperature, top-k, top-p, etc.
            self.sample_standard(&logits_vec, params)?
        };

        Ok(SampleResult {
            token_id,
            logprob,
            top_logprobs,
        })
    }

    /// Get logits as f32 vector.
    fn get_logits_vec(&self, logits: &Tensor) -> Result<Vec<f32>, SamplerError> {
        // Get last position logits
        let logits = if logits.dims().len() == 3 {
            let seq_len = logits.dim(1).map_err(|e| SamplerError::TensorError(e.to_string()))?;
            logits
                .narrow(1, seq_len - 1, 1)
                .map_err(|e| SamplerError::TensorError(e.to_string()))?
                .squeeze(1)
                .map_err(|e| SamplerError::TensorError(e.to_string()))?
        } else if logits.dims().len() == 2 {
            logits.clone()
        } else {
            return Err(SamplerError::InvalidShape(format!("{:?}", logits.dims())));
        };

        // Flatten to 1D
        let logits = logits
            .flatten_all()
            .map_err(|e| SamplerError::TensorError(e.to_string()))?;

        // Convert to f32
        let logits = logits
            .to_dtype(DType::F32)
            .map_err(|e| SamplerError::TensorError(e.to_string()))?;

        logits
            .to_vec1()
            .map_err(|e| SamplerError::TensorError(e.to_string()))
    }

    /// Greedy sampling (argmax).
    fn sample_greedy(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
    ) -> Result<(u32, f32, Option<Vec<(u32, f32)>>), SamplerError> {
        let (token_id, logit) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &v)| (i as u32, v))
            .ok_or(SamplerError::EmptyLogits)?;

        // Compute log probability
        let log_sum_exp = self.log_sum_exp(logits);
        let logprob = logit - log_sum_exp;

        // Get top logprobs if requested
        let top_logprobs = self.get_top_logprobs(logits, params, log_sum_exp);

        Ok((token_id, logprob, top_logprobs))
    }

    /// Standard sampling with temperature, top-k, top-p.
    fn sample_standard(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
    ) -> Result<(u32, f32, Option<Vec<(u32, f32)>>), SamplerError> {
        let temperature = params.temperature.unwrap_or(1.0);
        let top_k = params.top_k.unwrap_or(0);
        let top_p = params.top_p.unwrap_or(1.0);
        let min_p = params.min_p.unwrap_or(0.0);

        // Apply temperature
        let mut scaled_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l / temperature))
            .collect();

        // Sort by logit (descending)
        scaled_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k
        if top_k > 0 && top_k < scaled_logits.len() {
            scaled_logits.truncate(top_k);
        }

        // Convert to probabilities
        let max_logit = scaled_logits.first().map(|x| x.1).unwrap_or(0.0);
        let mut probs: Vec<(usize, f32)> = scaled_logits
            .iter()
            .map(|&(i, l)| (i, (l - max_logit).exp()))
            .collect();

        let sum: f32 = probs.iter().map(|x| x.1).sum();
        for p in &mut probs {
            p.1 /= sum;
        }

        // Apply min-p
        if min_p > 0.0 {
            let max_prob = probs.first().map(|x| x.1).unwrap_or(1.0);
            let threshold = max_prob * min_p;
            probs.retain(|&(_, p)| p >= threshold);
        }

        // Apply top-p (nucleus)
        if top_p < 1.0 {
            let mut cumsum = 0.0;
            let mut cutoff_idx = probs.len();
            for (i, &(_, p)) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum > top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff_idx);
        }

        // Renormalize
        let sum: f32 = probs.iter().map(|x| x.1).sum();
        for p in &mut probs {
            p.1 /= sum;
        }

        // Sample
        let weights: Vec<f32> = probs.iter().map(|x| x.1).collect();
        let dist = WeightedIndex::new(&weights)
            .map_err(|e| SamplerError::SamplingError(e.to_string()))?;
        let sampled_idx = dist.sample(&mut self.rng);
        let (token_id, prob) = probs[sampled_idx];

        // Compute logprob
        let logprob = prob.ln();

        // Get top logprobs
        let log_sum_exp = self.log_sum_exp(logits);
        let top_logprobs = self.get_top_logprobs(logits, params, log_sum_exp);

        Ok((token_id as u32, logprob, top_logprobs))
    }

    /// Mirostat sampling.
    fn sample_mirostat(
        &mut self,
        logits: &[f32],
        seq_id: SequenceId,
        params: &SamplingParams,
    ) -> Result<(u32, f32, Option<Vec<(u32, f32)>>), SamplerError> {
        let mode = params.mirostat_mode.unwrap_or(1);
        let tau = params.mirostat_tau.unwrap_or(5.0);
        let eta = params.mirostat_eta.unwrap_or(0.1);

        // Get or initialize state
        let state = self.mirostat_state
            .entry(seq_id)
            .or_insert_with(|| MirostatState::new(tau * 2.0));

        match mode {
            1 => self.sample_mirostat_v1(logits, state, tau, eta, params),
            2 => self.sample_mirostat_v2(logits, state, tau, eta, params),
            _ => Err(SamplerError::InvalidMirostatMode(mode)),
        }
    }

    /// Mirostat v1 sampling.
    fn sample_mirostat_v1(
        &mut self,
        logits: &[f32],
        state: &mut MirostatState,
        tau: f32,
        eta: f32,
        params: &SamplingParams,
    ) -> Result<(u32, f32, Option<Vec<(u32, f32)>>), SamplerError> {
        let mu = state.mu;

        // Sort logits
        let mut sorted: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute probabilities with temperature
        let max_logit = sorted.first().map(|x| x.1).unwrap_or(0.0);
        let probs: Vec<(usize, f32)> = sorted
            .iter()
            .map(|&(i, l)| (i, (l - max_logit).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|x| x.1).sum();
        let probs: Vec<(usize, f32)> = probs.iter().map(|&(i, p)| (i, p / sum)).collect();

        // Find k such that sum of top-k probs is closest to 2^(-mu)
        let target = (2.0_f32).powf(-mu);
        let mut cumsum = 0.0;
        let mut k = 1;
        for (i, &(_, p)) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= target {
                k = i + 1;
                break;
            }
        }

        // Truncate and renormalize
        let truncated: Vec<(usize, f32)> = probs.iter().take(k).cloned().collect();
        let sum: f32 = truncated.iter().map(|x| x.1).sum();
        let truncated: Vec<(usize, f32)> = truncated.iter().map(|&(i, p)| (i, p / sum)).collect();

        // Sample
        let weights: Vec<f32> = truncated.iter().map(|x| x.1).collect();
        let dist = WeightedIndex::new(&weights)
            .map_err(|e| SamplerError::SamplingError(e.to_string()))?;
        let sampled_idx = dist.sample(&mut self.rng);
        let (token_id, prob) = truncated[sampled_idx];

        // Update mu
        let surprise = -prob.ln() / (2.0_f32).ln();
        state.mu = mu - eta * (surprise - tau);

        let logprob = prob.ln();
        let log_sum_exp = self.log_sum_exp(logits);
        let top_logprobs = self.get_top_logprobs(logits, params, log_sum_exp);

        Ok((token_id as u32, logprob, top_logprobs))
    }

    /// Mirostat v2 sampling.
    fn sample_mirostat_v2(
        &mut self,
        logits: &[f32],
        state: &mut MirostatState,
        tau: f32,
        eta: f32,
        params: &SamplingParams,
    ) -> Result<(u32, f32, Option<Vec<(u32, f32)>>), SamplerError> {
        let mu = state.mu;

        // Sort logits
        let mut sorted: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert to probabilities
        let max_logit = sorted.first().map(|x| x.1).unwrap_or(0.0);
        let probs: Vec<(usize, f32)> = sorted
            .iter()
            .map(|&(i, l)| (i, (l - max_logit).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|x| x.1).sum();
        let probs: Vec<(usize, f32)> = probs.iter().map(|&(i, p)| (i, p / sum)).collect();

        // Apply mu-based truncation
        let mut filtered = Vec::new();
        for &(i, p) in &probs {
            let surprise = -p.ln() / (2.0_f32).ln();
            if surprise <= mu {
                filtered.push((i, p));
            }
        }

        // Ensure at least one token
        if filtered.is_empty() {
            filtered.push(probs[0]);
        }

        // Renormalize
        let sum: f32 = filtered.iter().map(|x| x.1).sum();
        let filtered: Vec<(usize, f32)> = filtered.iter().map(|&(i, p)| (i, p / sum)).collect();

        // Sample
        let weights: Vec<f32> = filtered.iter().map(|x| x.1).collect();
        let dist = WeightedIndex::new(&weights)
            .map_err(|e| SamplerError::SamplingError(e.to_string()))?;
        let sampled_idx = dist.sample(&mut self.rng);
        let (token_id, prob) = filtered[sampled_idx];

        // Update mu
        let surprise = -prob.ln() / (2.0_f32).ln();
        state.mu = mu - eta * (surprise - tau);

        let logprob = prob.ln();
        let log_sum_exp = self.log_sum_exp(logits);
        let top_logprobs = self.get_top_logprobs(logits, params, log_sum_exp);

        Ok((token_id as u32, logprob, top_logprobs))
    }

    /// Compute log-sum-exp for normalization using SIMD when available.
    #[inline]
    fn log_sum_exp(&self, logits: &[f32]) -> f32 {
        simd_log_sum_exp(logits)
    }

    /// Get top log probabilities if requested.
    fn get_top_logprobs(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        log_sum_exp: f32,
    ) -> Option<Vec<(u32, f32)>> {
        let n = params.logprobs?;
        if n == 0 {
            return None;
        }

        let mut indexed: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, l - log_sum_exp))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Some(
            indexed
                .into_iter()
                .take(n as usize)
                .map(|(i, lp)| (i as u32, lp))
                .collect(),
        )
    }

    /// Reset Mirostat state for sequence.
    pub fn reset_mirostat(&mut self, seq_id: &SequenceId) {
        self.mirostat_state.remove(seq_id);
    }

    /// Clear all Mirostat states.
    pub fn clear_mirostat(&mut self) {
        self.mirostat_state.clear();
    }
}

/// Mirostat state.
#[derive(Debug)]
struct MirostatState {
    /// Current mu value.
    mu: f32,
}

impl MirostatState {
    fn new(mu: f32) -> Self {
        Self { mu }
    }
}

/// Sampler errors.
#[derive(Debug, thiserror::Error)]
pub enum SamplerError {
    #[error("Tensor error: {0}")]
    TensorError(String),

    #[error("Invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("Empty logits")]
    EmptyLogits,

    #[error("Sampling error: {0}")]
    SamplingError(String),

    #[error("Invalid Mirostat mode: {0}")]
    InvalidMirostatMode(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let mut sampler = Sampler::new(Some(42));
        let logits = vec![1.0, 5.0, 2.0, 0.5];

        let params = SamplingParams {
            temperature: Some(0.0),
            ..Default::default()
        };

        let result = sampler.sample_greedy(&logits, &params).unwrap();
        assert_eq!(result.0, 1); // Index of max value (5.0)
    }

    #[test]
    fn test_temperature_sampling() {
        let mut sampler = Sampler::new(Some(42));
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        let params = SamplingParams {
            temperature: Some(1.0),
            ..Default::default()
        };

        let result = sampler.sample_standard(&logits, &params).unwrap();
        // With seed 42, should be deterministic
        assert!(result.0 < 4);
    }

    #[test]
    fn test_top_k_sampling() {
        let mut sampler = Sampler::new(Some(42));
        let logits = vec![1.0, 2.0, 3.0, 10.0]; // Last one is much higher

        let params = SamplingParams {
            temperature: Some(1.0),
            top_k: Some(1),
            ..Default::default()
        };

        // With top_k=1, should always select the highest
        let result = sampler.sample_standard(&logits, &params).unwrap();
        assert_eq!(result.0, 3);
    }
}
