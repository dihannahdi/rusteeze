//! Sampling parameters for text generation.
//!
//! This module provides comprehensive sampling configuration used to control
//! the behavior of text generation, including temperature, top-p, top-k,
//! and advanced techniques like min-p, typical-p, and repetition penalties.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default temperature value.
pub const DEFAULT_TEMPERATURE: f32 = 1.0;
/// Default top-p value.
pub const DEFAULT_TOP_P: f32 = 1.0;
/// Default top-k value (disabled).
pub const DEFAULT_TOP_K: i32 = -1;
/// Default frequency penalty.
pub const DEFAULT_FREQUENCY_PENALTY: f32 = 0.0;
/// Default presence penalty.
pub const DEFAULT_PRESENCE_PENALTY: f32 = 0.0;
/// Default repetition penalty.
pub const DEFAULT_REPETITION_PENALTY: f32 = 1.0;

/// Comprehensive sampling parameters for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Number of sequences to generate.
    #[serde(default = "default_n")]
    pub n: u32,

    /// Maximum number of tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// Temperature for sampling (0.0 = deterministic, higher = more random).
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold.
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling (number of highest probability tokens to consider).
    #[serde(default = "default_top_k")]
    pub top_k: i32,

    /// Min-p sampling threshold (alternative to top-p).
    #[serde(default)]
    pub min_p: f32,

    /// Typical-p sampling threshold.
    #[serde(default)]
    pub typical_p: f32,

    /// Frequency penalty (penalize tokens based on frequency in generated text).
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Presence penalty (penalize tokens that appear in generated text).
    #[serde(default)]
    pub presence_penalty: f32,

    /// Repetition penalty (multiplicative penalty for repeated tokens).
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Length penalty for beam search.
    #[serde(default = "default_length_penalty")]
    pub length_penalty: f32,

    /// Number of beams for beam search (1 = no beam search).
    #[serde(default = "default_best_of")]
    pub best_of: u32,

    /// Use beam search instead of sampling.
    #[serde(default)]
    pub use_beam_search: bool,

    /// Early stopping for beam search.
    #[serde(default)]
    pub early_stopping: EarlyStoppingMode,

    /// Sequences to stop generation on.
    #[serde(default)]
    pub stop: Vec<String>,

    /// Token IDs to stop on.
    #[serde(default)]
    pub stop_token_ids: Vec<u32>,

    /// Include stop string in output.
    #[serde(default)]
    pub include_stop_str_in_output: bool,

    /// Skip special tokens in output.
    #[serde(default = "default_skip_special_tokens")]
    pub skip_special_tokens: bool,

    /// Add spaces between special tokens.
    #[serde(default)]
    pub spaces_between_special_tokens: bool,

    /// Random seed for reproducibility.
    #[serde(default)]
    pub seed: Option<u64>,

    /// Logit bias (token_id -> bias).
    #[serde(default)]
    pub logit_bias: HashMap<u32, f32>,

    /// Number of log probabilities to return.
    #[serde(default)]
    pub logprobs: Option<u32>,

    /// Return log probabilities for prompt tokens.
    #[serde(default)]
    pub prompt_logprobs: Option<u32>,

    /// Whether to ignore EOS token.
    #[serde(default)]
    pub ignore_eos: bool,

    /// Minimum number of tokens to generate before considering stop conditions.
    #[serde(default)]
    pub min_tokens: u32,

    /// Eta cutoff for eta sampling.
    #[serde(default)]
    pub eta_cutoff: f32,

    /// Epsilon cutoff for epsilon sampling.
    #[serde(default)]
    pub epsilon_cutoff: f32,

    /// Mirostat mode (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    #[serde(default)]
    pub mirostat_mode: u8,

    /// Mirostat target entropy.
    #[serde(default = "default_mirostat_tau")]
    pub mirostat_tau: f32,

    /// Mirostat learning rate.
    #[serde(default = "default_mirostat_eta")]
    pub mirostat_eta: f32,

    /// Guided decoding configuration.
    #[serde(default)]
    pub guided_decoding: Option<GuidedDecodingParams>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            n: 1,
            max_tokens: None,
            temperature: DEFAULT_TEMPERATURE,
            top_p: DEFAULT_TOP_P,
            top_k: DEFAULT_TOP_K,
            min_p: 0.0,
            typical_p: 0.0,
            frequency_penalty: DEFAULT_FREQUENCY_PENALTY,
            presence_penalty: DEFAULT_PRESENCE_PENALTY,
            repetition_penalty: DEFAULT_REPETITION_PENALTY,
            length_penalty: 1.0,
            best_of: 1,
            use_beam_search: false,
            early_stopping: EarlyStoppingMode::False,
            stop: vec![],
            stop_token_ids: vec![],
            include_stop_str_in_output: false,
            skip_special_tokens: true,
            spaces_between_special_tokens: false,
            seed: None,
            logit_bias: HashMap::new(),
            logprobs: None,
            prompt_logprobs: None,
            ignore_eos: false,
            min_tokens: 0,
            eta_cutoff: 0.0,
            epsilon_cutoff: 0.0,
            mirostat_mode: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            guided_decoding: None,
        }
    }
}

impl SamplingParams {
    /// Create new sampling parameters with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create greedy sampling parameters (temperature = 0).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: -1,
            ..Default::default()
        }
    }

    /// Create beam search parameters.
    pub fn beam_search(num_beams: u32) -> Self {
        Self {
            use_beam_search: true,
            best_of: num_beams,
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Create sampling parameters with a specific temperature.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Default::default()
        }
    }

    /// Builder: set temperature.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Builder: set top-p.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Builder: set top-k.
    pub fn top_k(mut self, top_k: i32) -> Self {
        self.top_k = top_k;
        self
    }

    /// Builder: set max tokens.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Builder: set stop sequences.
    pub fn stop<S: Into<String>>(mut self, stop: Vec<S>) -> Self {
        self.stop = stop.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Builder: add a stop sequence.
    pub fn add_stop(mut self, stop: impl Into<String>) -> Self {
        self.stop.push(stop.into());
        self
    }

    /// Builder: set seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builder: set number of sequences.
    pub fn n(mut self, n: u32) -> Self {
        self.n = n;
        self
    }

    /// Builder: set frequency penalty.
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Builder: set presence penalty.
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Builder: set repetition penalty.
    pub fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Builder: set logprobs.
    pub fn logprobs(mut self, n: u32) -> Self {
        self.logprobs = Some(n);
        self
    }

    /// Builder: set min-p.
    pub fn min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    /// Builder: set typical-p.
    pub fn typical_p(mut self, typical_p: f32) -> Self {
        self.typical_p = typical_p;
        self
    }

    /// Builder: add logit bias.
    pub fn add_logit_bias(mut self, token_id: u32, bias: f32) -> Self {
        self.logit_bias.insert(token_id, bias);
        self
    }

    /// Builder: set guided decoding.
    pub fn guided_decoding(mut self, params: GuidedDecodingParams) -> Self {
        self.guided_decoding = Some(params);
        self
    }

    /// Validate the sampling parameters.
    pub fn validate(&self) -> Result<(), SamplingParamsError> {
        // Validate temperature
        if self.temperature < 0.0 {
            return Err(SamplingParamsError::InvalidTemperature(self.temperature));
        }

        // Validate top-p
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(SamplingParamsError::InvalidTopP(self.top_p));
        }

        // Validate min-p
        if self.min_p < 0.0 || self.min_p > 1.0 {
            return Err(SamplingParamsError::InvalidMinP(self.min_p));
        }

        // Validate typical-p
        if self.typical_p < 0.0 || self.typical_p > 1.0 {
            return Err(SamplingParamsError::InvalidTypicalP(self.typical_p));
        }

        // Validate frequency penalty
        if self.frequency_penalty < -2.0 || self.frequency_penalty > 2.0 {
            return Err(SamplingParamsError::InvalidFrequencyPenalty(self.frequency_penalty));
        }

        // Validate presence penalty
        if self.presence_penalty < -2.0 || self.presence_penalty > 2.0 {
            return Err(SamplingParamsError::InvalidPresencePenalty(self.presence_penalty));
        }

        // Validate repetition penalty
        if self.repetition_penalty < 0.0 {
            return Err(SamplingParamsError::InvalidRepetitionPenalty(self.repetition_penalty));
        }

        // Validate n
        if self.n == 0 {
            return Err(SamplingParamsError::InvalidN(self.n));
        }

        // Validate best_of
        if self.best_of == 0 || self.best_of < self.n {
            return Err(SamplingParamsError::InvalidBestOf {
                best_of: self.best_of,
                n: self.n,
            });
        }

        // Validate beam search constraints
        if self.use_beam_search {
            if self.temperature != 0.0 {
                return Err(SamplingParamsError::BeamSearchWithTemperature);
            }
            if self.top_p < 1.0 {
                return Err(SamplingParamsError::BeamSearchWithTopP);
            }
            if self.top_k > 0 {
                return Err(SamplingParamsError::BeamSearchWithTopK);
            }
        }

        // Validate logit bias values
        for (&token_id, &bias) in &self.logit_bias {
            if bias < -100.0 || bias > 100.0 {
                return Err(SamplingParamsError::InvalidLogitBias { token_id, bias });
            }
        }

        Ok(())
    }

    /// Check if sampling is deterministic.
    pub fn is_deterministic(&self) -> bool {
        self.temperature == 0.0 || self.use_beam_search
    }

    /// Check if logprobs are requested.
    pub fn wants_logprobs(&self) -> bool {
        self.logprobs.is_some() || self.prompt_logprobs.is_some()
    }

    /// Get effective temperature (accounting for min-p, typical-p).
    pub fn effective_temperature(&self) -> f32 {
        if self.temperature == 0.0 {
            0.0
        } else {
            self.temperature
        }
    }

    /// Check if any repetition penalty is active.
    pub fn has_repetition_penalty(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }

    /// Check if guided decoding is enabled.
    pub fn has_guided_decoding(&self) -> bool {
        self.guided_decoding.is_some()
    }

    /// Get the sampling strategy being used.
    pub fn strategy(&self) -> SamplingStrategy {
        if self.use_beam_search {
            SamplingStrategy::BeamSearch
        } else if self.temperature == 0.0 {
            SamplingStrategy::Greedy
        } else if self.mirostat_mode > 0 {
            SamplingStrategy::Mirostat
        } else if self.min_p > 0.0 {
            SamplingStrategy::MinP
        } else if self.typical_p > 0.0 {
            SamplingStrategy::TypicalP
        } else if self.top_k > 0 && self.top_p < 1.0 {
            SamplingStrategy::TopKTopP
        } else if self.top_k > 0 {
            SamplingStrategy::TopK
        } else if self.top_p < 1.0 {
            SamplingStrategy::TopP
        } else {
            SamplingStrategy::Random
        }
    }
}

/// Sampling strategy being used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Greedy decoding (always pick highest probability).
    Greedy,
    /// Random sampling.
    Random,
    /// Top-k sampling.
    TopK,
    /// Top-p (nucleus) sampling.
    TopP,
    /// Combined top-k and top-p.
    TopKTopP,
    /// Min-p sampling.
    MinP,
    /// Typical-p sampling.
    TypicalP,
    /// Mirostat sampling.
    Mirostat,
    /// Beam search.
    BeamSearch,
}

/// Early stopping mode for beam search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EarlyStoppingMode {
    /// Never stop early.
    #[default]
    False,
    /// Stop when best_of beams are complete.
    True,
    /// Stop when it's very unlikely a better result will be found.
    Never,
}

/// Parameters for guided decoding (structured output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidedDecodingParams {
    /// JSON schema to follow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,

    /// Regular expression to match.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,

    /// Context-free grammar.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,

    /// Choice of strings to output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice: Option<Vec<String>>,

    /// Backend to use for guided decoding.
    #[serde(default)]
    pub backend: GuidedDecodingBackend,
}

/// Backend for guided decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GuidedDecodingBackend {
    /// Automatic selection.
    #[default]
    Auto,
    /// Outlines library.
    Outlines,
    /// LMFE library.
    Lmfe,
    /// XGrammar library.
    XGrammar,
}

/// Errors that can occur when validating sampling parameters.
#[derive(Debug, thiserror::Error)]
pub enum SamplingParamsError {
    #[error("Invalid temperature: {0} (must be >= 0)")]
    InvalidTemperature(f32),

    #[error("Invalid top_p: {0} (must be in [0, 1])")]
    InvalidTopP(f32),

    #[error("Invalid min_p: {0} (must be in [0, 1])")]
    InvalidMinP(f32),

    #[error("Invalid typical_p: {0} (must be in [0, 1])")]
    InvalidTypicalP(f32),

    #[error("Invalid frequency_penalty: {0} (must be in [-2, 2])")]
    InvalidFrequencyPenalty(f32),

    #[error("Invalid presence_penalty: {0} (must be in [-2, 2])")]
    InvalidPresencePenalty(f32),

    #[error("Invalid repetition_penalty: {0} (must be >= 0)")]
    InvalidRepetitionPenalty(f32),

    #[error("Invalid n: {0} (must be > 0)")]
    InvalidN(u32),

    #[error("Invalid best_of: {best_of} (must be >= n={n})")]
    InvalidBestOf { best_of: u32, n: u32 },

    #[error("Beam search requires temperature=0")]
    BeamSearchWithTemperature,

    #[error("Beam search requires top_p=1")]
    BeamSearchWithTopP,

    #[error("Beam search requires top_k=-1")]
    BeamSearchWithTopK,

    #[error("Invalid logit bias for token {token_id}: {bias} (must be in [-100, 100])")]
    InvalidLogitBias { token_id: u32, bias: f32 },
}

// Default value functions for serde
fn default_n() -> u32 { 1 }
fn default_temperature() -> f32 { DEFAULT_TEMPERATURE }
fn default_top_p() -> f32 { DEFAULT_TOP_P }
fn default_top_k() -> i32 { DEFAULT_TOP_K }
fn default_repetition_penalty() -> f32 { DEFAULT_REPETITION_PENALTY }
fn default_length_penalty() -> f32 { 1.0 }
fn default_best_of() -> u32 { 1 }
fn default_skip_special_tokens() -> bool { true }
fn default_mirostat_tau() -> f32 { 5.0 }
fn default_mirostat_eta() -> f32 { 0.1 }

/// Sampler state for stateful sampling algorithms like Mirostat.
#[derive(Debug, Clone)]
pub struct SamplerState {
    /// Current Mirostat mu value.
    pub mirostat_mu: f32,

    /// RNG state (for reproducibility).
    pub rng_state: Option<u64>,

    /// Token counts for repetition penalty.
    pub token_counts: HashMap<u32, u32>,
}

impl Default for SamplerState {
    fn default() -> Self {
        Self {
            mirostat_mu: 10.0, // Default Mirostat starting value
            rng_state: None,
            token_counts: HashMap::new(),
        }
    }
}

impl SamplerState {
    /// Create a new sampler state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng_state: Some(seed),
            ..Default::default()
        }
    }

    /// Record a token being generated.
    pub fn record_token(&mut self, token_id: u32) {
        *self.token_counts.entry(token_id).or_insert(0) += 1;
    }

    /// Get the count for a token.
    pub fn token_count(&self, token_id: u32) -> u32 {
        self.token_counts.get(&token_id).copied().unwrap_or(0)
    }

    /// Reset token counts.
    pub fn reset_counts(&mut self) {
        self.token_counts.clear();
    }

    /// Update Mirostat mu value.
    pub fn update_mirostat_mu(&mut self, mu: f32) {
        self.mirostat_mu = mu;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, DEFAULT_TEMPERATURE);
        assert_eq!(params.top_p, DEFAULT_TOP_P);
        assert_eq!(params.top_k, DEFAULT_TOP_K);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_greedy_params() {
        let params = SamplingParams::greedy();
        assert_eq!(params.temperature, 0.0);
        assert!(params.is_deterministic());
        assert_eq!(params.strategy(), SamplingStrategy::Greedy);
    }

    #[test]
    fn test_beam_search_params() {
        let params = SamplingParams::beam_search(4);
        assert!(params.use_beam_search);
        assert_eq!(params.best_of, 4);
        assert!(params.validate().is_ok());
        assert_eq!(params.strategy(), SamplingStrategy::BeamSearch);
    }

    #[test]
    fn test_builder_pattern() {
        let params = SamplingParams::new()
            .temperature(0.7)
            .top_p(0.9)
            .top_k(50)
            .max_tokens(100)
            .seed(42)
            .add_stop("<|end|>");

        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 50);
        assert_eq!(params.max_tokens, Some(100));
        assert_eq!(params.seed, Some(42));
        assert_eq!(params.stop, vec!["<|end|>"]);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_validation_errors() {
        let params = SamplingParams::new().temperature(-1.0);
        assert!(matches!(
            params.validate(),
            Err(SamplingParamsError::InvalidTemperature(_))
        ));

        let params = SamplingParams::new().top_p(1.5);
        assert!(matches!(
            params.validate(),
            Err(SamplingParamsError::InvalidTopP(_))
        ));
    }

    #[test]
    fn test_strategy_detection() {
        let params = SamplingParams::new().top_k(50);
        assert_eq!(params.strategy(), SamplingStrategy::TopK);

        let params = SamplingParams::new().top_p(0.9);
        assert_eq!(params.strategy(), SamplingStrategy::TopP);

        let params = SamplingParams::new().top_k(50).top_p(0.9);
        assert_eq!(params.strategy(), SamplingStrategy::TopKTopP);

        let params = SamplingParams::new().min_p(0.1);
        assert_eq!(params.strategy(), SamplingStrategy::MinP);
    }

    #[test]
    fn test_sampler_state() {
        let mut state = SamplerState::new();
        state.record_token(100);
        state.record_token(100);
        state.record_token(200);

        assert_eq!(state.token_count(100), 2);
        assert_eq!(state.token_count(200), 1);
        assert_eq!(state.token_count(300), 0);

        state.reset_counts();
        assert_eq!(state.token_count(100), 0);
    }

    #[test]
    fn test_serialization() {
        let params = SamplingParams::new()
            .temperature(0.8)
            .max_tokens(100);

        let json = serde_json::to_string(&params).unwrap();
        let parsed: SamplingParams = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.temperature, 0.8);
        assert_eq!(parsed.max_tokens, Some(100));
    }
}
