//! Engine configuration.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct EngineConfig {
    /// Maximum number of sequences.
    #[serde(default = "default_max_seqs")]
    pub max_num_seqs: usize,

    /// Maximum number of batched tokens.
    #[serde(default = "default_max_tokens")]
    pub max_num_batched_tokens: usize,

    /// Block size for paged attention.
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    /// GPU memory utilization (0.0-1.0).
    #[serde(default = "default_gpu_utilization")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub gpu_memory_utilization: f32,

    /// Swap space in GB.
    #[serde(default = "default_swap_space")]
    pub swap_space_gb: f32,

    /// Scheduling policy.
    #[serde(default)]
    pub scheduling: SchedulingConfig,

    /// Sampling defaults.
    #[serde(default)]
    pub sampling: SamplingDefaults,

    /// Speculative decoding configuration.
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,

    /// Enable prefix caching.
    #[serde(default)]
    pub enable_prefix_caching: bool,

    /// Enable chunked prefill.
    #[serde(default)]
    pub enable_chunked_prefill: bool,

    /// Max prefill tokens per iteration.
    #[serde(default)]
    pub max_prefill_tokens: Option<usize>,

    /// Number of GPU blocks override.
    #[serde(default)]
    pub num_gpu_blocks: Option<usize>,

    /// Number of CPU blocks override.
    #[serde(default)]
    pub num_cpu_blocks: Option<usize>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            swap_space_gb: 4.0,
            scheduling: SchedulingConfig::default(),
            sampling: SamplingDefaults::default(),
            speculative: None,
            enable_prefix_caching: false,
            enable_chunked_prefill: false,
            max_prefill_tokens: None,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
        }
    }
}

fn default_max_seqs() -> usize {
    256
}

fn default_max_tokens() -> usize {
    8192
}

fn default_block_size() -> usize {
    16
}

fn default_gpu_utilization() -> f32 {
    0.9
}

fn default_swap_space() -> f32 {
    4.0
}

/// Scheduling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConfig {
    /// Scheduling policy.
    #[serde(default)]
    pub policy: SchedulingPolicy,

    /// Maximum waiting time in seconds.
    #[serde(default = "default_max_waiting")]
    pub max_waiting_time_seconds: f32,

    /// Delay factor for preemption.
    #[serde(default = "default_delay_factor")]
    pub delay_factor: f32,

    /// Enable preemption.
    #[serde(default = "default_true")]
    pub enable_preemption: bool,

    /// Preemption mode.
    #[serde(default)]
    pub preemption_mode: PreemptionMode,
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Fcfs,
            max_waiting_time_seconds: 60.0,
            delay_factor: 0.5,
            enable_preemption: true,
            preemption_mode: PreemptionMode::Recompute,
        }
    }
}

fn default_max_waiting() -> f32 {
    60.0
}

fn default_delay_factor() -> f32 {
    0.5
}

fn default_true() -> bool {
    true
}

/// Scheduling policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SchedulingPolicy {
    /// First-come, first-served.
    #[default]
    Fcfs,

    /// Shortest job first.
    Sjf,

    /// Priority-based.
    Priority,
}

/// Preemption mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PreemptionMode {
    /// Recompute KV cache after preemption.
    #[default]
    Recompute,

    /// Swap KV cache to CPU.
    Swap,
}

/// Default sampling parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingDefaults {
    /// Default temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Default top-p.
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Default top-k.
    #[serde(default)]
    pub top_k: Option<usize>,

    /// Default max tokens.
    #[serde(default = "default_max_gen_tokens")]
    pub max_tokens: usize,

    /// Default presence penalty.
    #[serde(default)]
    pub presence_penalty: f32,

    /// Default frequency penalty.
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Default repetition penalty.
    #[serde(default = "default_rep_penalty")]
    pub repetition_penalty: f32,

    /// Default stop sequences.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            max_tokens: 256,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
        }
    }
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_max_gen_tokens() -> usize {
    256
}

fn default_rep_penalty() -> f32 {
    1.0
}

/// Speculative decoding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Draft model path.
    pub draft_model: String,

    /// Number of speculative tokens.
    #[serde(default = "default_spec_tokens")]
    pub num_speculative_tokens: usize,

    /// Acceptance threshold.
    #[serde(default = "default_acceptance")]
    pub acceptance_threshold: f32,

    /// Use same tokenizer.
    #[serde(default = "default_true")]
    pub use_same_tokenizer: bool,
}

fn default_spec_tokens() -> usize {
    5
}

fn default_acceptance() -> f32 {
    0.9
}

impl EngineConfig {
    /// Create a new engine config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max number of sequences.
    pub fn with_max_seqs(mut self, max: usize) -> Self {
        self.max_num_seqs = max;
        self
    }

    /// Set max batched tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_num_batched_tokens = max;
        self
    }

    /// Set block size.
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }

    /// Set GPU memory utilization.
    pub fn with_gpu_utilization(mut self, util: f32) -> Self {
        self.gpu_memory_utilization = util.clamp(0.0, 1.0);
        self
    }

    /// Enable prefix caching.
    pub fn with_prefix_caching(mut self, enabled: bool) -> Self {
        self.enable_prefix_caching = enabled;
        self
    }

    /// Enable chunked prefill.
    pub fn with_chunked_prefill(mut self, enabled: bool) -> Self {
        self.enable_chunked_prefill = enabled;
        self
    }

    /// Set speculative decoding.
    pub fn with_speculative(mut self, config: SpeculativeConfig) -> Self {
        self.speculative = Some(config);
        self
    }

    /// Calculate number of GPU blocks.
    pub fn calculate_gpu_blocks(&self, available_memory: usize, kv_cache_size_per_block: usize) -> usize {
        if let Some(blocks) = self.num_gpu_blocks {
            return blocks;
        }

        let usable_memory = (available_memory as f32 * self.gpu_memory_utilization) as usize;
        usable_memory / kv_cache_size_per_block
    }

    /// Calculate number of CPU blocks.
    pub fn calculate_cpu_blocks(&self, kv_cache_size_per_block: usize) -> usize {
        if let Some(blocks) = self.num_cpu_blocks {
            return blocks;
        }

        let swap_bytes = (self.swap_space_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        swap_bytes / kv_cache_size_per_block
    }
}
