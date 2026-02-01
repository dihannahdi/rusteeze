//! Configuration types for the Rusteeze inference engine.
//!
//! This module provides comprehensive configuration structures for all
//! aspects of the inference engine including model loading, scheduling,
//! caching, and API server settings.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::device::{DeviceType, DType};

/// Default values
pub const DEFAULT_GPU_MEMORY_UTILIZATION: f64 = 0.90;
pub const DEFAULT_BLOCK_SIZE: usize = 16;
pub const DEFAULT_MAX_NUM_BATCHED_TOKENS: usize = 8192;
pub const DEFAULT_MAX_NUM_SEQS: usize = 256;
pub const DEFAULT_MAX_MODEL_LEN: usize = 32768;
pub const DEFAULT_SWAP_SPACE_GB: f64 = 4.0;

/// Main configuration for the Rusteeze engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Model configuration.
    pub model: ModelConfig,

    /// Cache configuration.
    pub cache: CacheConfig,

    /// Scheduler configuration.
    pub scheduler: SchedulerConfig,

    /// Parallel configuration.
    pub parallel: ParallelConfig,

    /// LoRA configuration.
    #[serde(default)]
    pub lora: Option<LoraConfig>,

    /// Speculative decoding configuration.
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,

    /// Observability configuration.
    #[serde(default)]
    pub observability: ObservabilityConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            cache: CacheConfig::default(),
            scheduler: SchedulerConfig::default(),
            parallel: ParallelConfig::default(),
            lora: None,
            speculative: None,
            observability: ObservabilityConfig::default(),
        }
    }
}

impl EngineConfig {
    /// Create a new engine config with the given model path.
    pub fn with_model(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model: ModelConfig::with_path(model_path),
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.model.validate()?;
        self.cache.validate()?;
        self.scheduler.validate()?;
        self.parallel.validate()?;
        if let Some(ref lora) = self.lora {
            lora.validate()?;
        }
        if let Some(ref spec) = self.speculative {
            spec.validate()?;
        }
        Ok(())
    }
}

/// Model loading and configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to model weights.
    pub model_path: PathBuf,

    /// Model name/identifier.
    #[serde(default)]
    pub model_name: Option<String>,

    /// Revision/branch (for HuggingFace).
    #[serde(default)]
    pub revision: Option<String>,

    /// Data type for model weights.
    #[serde(default)]
    pub dtype: DType,

    /// Device to load model on.
    #[serde(default)]
    pub device: DeviceType,

    /// Trust remote code (for HuggingFace models).
    #[serde(default)]
    pub trust_remote_code: bool,

    /// Maximum model length (context window).
    #[serde(default = "default_max_model_len")]
    pub max_model_len: usize,

    /// Tokenizer path (if different from model).
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,

    /// Tokenizer mode.
    #[serde(default)]
    pub tokenizer_mode: TokenizerMode,

    /// Quantization method.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    /// Use Flash Attention.
    #[serde(default = "default_true")]
    pub use_flash_attention: bool,

    /// Enforce eager mode (no CUDA graphs).
    #[serde(default)]
    pub enforce_eager: bool,

    /// Custom model config overrides.
    #[serde(default)]
    pub config_overrides: HashMap<String, serde_json::Value>,

    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    /// Sliding window attention size.
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            model_name: None,
            revision: None,
            dtype: DType::Auto,
            device: DeviceType::Cuda,
            trust_remote_code: false,
            max_model_len: DEFAULT_MAX_MODEL_LEN,
            tokenizer_path: None,
            tokenizer_mode: TokenizerMode::Auto,
            quantization: None,
            use_flash_attention: true,
            enforce_eager: false,
            config_overrides: HashMap::new(),
            rope_scaling: None,
            sliding_window: None,
        }
    }
}

impl ModelConfig {
    /// Create config with model path.
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: path.into(),
            ..Default::default()
        }
    }

    /// Validate model config.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_path.as_os_str().is_empty() {
            return Err(ConfigError::MissingField("model.model_path".to_string()));
        }
        if self.max_model_len == 0 {
            return Err(ConfigError::InvalidValue {
                field: "model.max_model_len".to_string(),
                message: "must be > 0".to_string(),
            });
        }
        Ok(())
    }

    /// Get effective model name.
    pub fn effective_model_name(&self) -> String {
        self.model_name.clone().unwrap_or_else(|| {
            self.model_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        })
    }
}

/// Tokenizer mode.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TokenizerMode {
    /// Automatic detection.
    #[default]
    Auto,
    /// Slow tokenizer (Python).
    Slow,
    /// Fast tokenizer (Rust).
    Fast,
    /// Mistral tokenizer.
    Mistral,
}

/// Quantization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method.
    pub method: QuantizationMethod,

    /// Bits for quantization.
    #[serde(default)]
    pub bits: Option<u8>,

    /// Group size for grouped quantization.
    #[serde(default)]
    pub group_size: Option<usize>,

    /// Use double quantization.
    #[serde(default)]
    pub double_quant: bool,
}

/// Quantization method.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMethod {
    /// No quantization.
    None,
    /// GPTQ quantization.
    Gptq,
    /// AWQ quantization.
    Awq,
    /// SqueezeLLM.
    Squeezellm,
    /// GGUF/GGML format.
    Gguf,
    /// FP8 quantization.
    Fp8,
    /// Marlin (optimized GPTQ).
    Marlin,
    /// bitsandbytes (int8/fp4).
    Bitsandbytes,
}

/// RoPE scaling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    /// Scaling type.
    pub scaling_type: RopeScalingType,

    /// Scaling factor.
    pub factor: f32,

    /// Original max position (for dynamic scaling).
    #[serde(default)]
    pub original_max_position: Option<usize>,

    /// Low frequency factor (for YaRN).
    #[serde(default)]
    pub low_freq_factor: Option<f32>,

    /// High frequency factor (for YaRN).
    #[serde(default)]
    pub high_freq_factor: Option<f32>,
}

/// RoPE scaling type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RopeScalingType {
    /// Linear interpolation.
    Linear,
    /// Dynamic NTK-aware scaling.
    Dynamic,
    /// YaRN (Yet another RoPE extension).
    Yarn,
    /// Llama3 scaling.
    Llama3,
}

/// Cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Block size for paged attention.
    #[serde(default = "default_block_size")]
    pub block_size: usize,

    /// GPU memory utilization target (0.0-1.0).
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f64,

    /// Swap space in GB for CPU offloading.
    #[serde(default = "default_swap_space_gb")]
    pub swap_space_gb: f64,

    /// KV cache data type.
    #[serde(default)]
    pub cache_dtype: CacheDType,

    /// Number of GPU blocks (auto if None).
    #[serde(default)]
    pub num_gpu_blocks: Option<usize>,

    /// Number of CPU blocks (auto if None).
    #[serde(default)]
    pub num_cpu_blocks: Option<usize>,

    /// Enable prefix caching.
    #[serde(default)]
    pub enable_prefix_caching: bool,

    /// Enable chunked prefill.
    #[serde(default)]
    pub enable_chunked_prefill: bool,

    /// Sliding window size for cache.
    #[serde(default)]
    pub sliding_window: Option<usize>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            gpu_memory_utilization: DEFAULT_GPU_MEMORY_UTILIZATION,
            swap_space_gb: DEFAULT_SWAP_SPACE_GB,
            cache_dtype: CacheDType::Auto,
            num_gpu_blocks: None,
            num_cpu_blocks: None,
            enable_prefix_caching: false,
            enable_chunked_prefill: false,
            sliding_window: None,
        }
    }
}

impl CacheConfig {
    /// Validate cache config.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.block_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "cache.block_size".to_string(),
                message: "must be > 0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.gpu_memory_utilization) {
            return Err(ConfigError::InvalidValue {
                field: "cache.gpu_memory_utilization".to_string(),
                message: "must be in [0.0, 1.0]".to_string(),
            });
        }
        if self.swap_space_gb < 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "cache.swap_space_gb".to_string(),
                message: "must be >= 0".to_string(),
            });
        }
        Ok(())
    }
}

/// KV cache data type.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheDType {
    /// Auto-detect from model.
    #[default]
    Auto,
    /// FP16.
    Fp16,
    /// BF16.
    Bf16,
    /// FP8 E5M2.
    Fp8E5m2,
    /// FP8 E4M3.
    Fp8E4m3,
}

/// Scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of batched tokens.
    #[serde(default = "default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,

    /// Maximum number of sequences.
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,

    /// Maximum padding in a batch.
    #[serde(default = "default_max_paddings")]
    pub max_paddings: usize,

    /// Scheduling policy.
    #[serde(default)]
    pub policy: SchedulingPolicy,

    /// Preemption mode.
    #[serde(default)]
    pub preemption_mode: PreemptionMode,

    /// Delay factor for scheduling.
    #[serde(default)]
    pub delay_factor: f64,

    /// Enable chunked prefill.
    #[serde(default)]
    pub enable_chunked_prefill: bool,

    /// Chunk size for prefill.
    #[serde(default)]
    pub max_prefill_chunk_size: Option<usize>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            max_paddings: 512,
            policy: SchedulingPolicy::Fcfs,
            preemption_mode: PreemptionMode::Recompute,
            delay_factor: 0.0,
            enable_chunked_prefill: false,
            max_prefill_chunk_size: None,
        }
    }
}

impl SchedulerConfig {
    /// Validate scheduler config.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_num_batched_tokens == 0 {
            return Err(ConfigError::InvalidValue {
                field: "scheduler.max_num_batched_tokens".to_string(),
                message: "must be > 0".to_string(),
            });
        }
        if self.max_num_seqs == 0 {
            return Err(ConfigError::InvalidValue {
                field: "scheduler.max_num_seqs".to_string(),
                message: "must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

/// Scheduling policy.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchedulingPolicy {
    /// First-come-first-served.
    #[default]
    Fcfs,
    /// Priority-based.
    Priority,
}

/// Preemption mode.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PreemptionMode {
    /// Recompute preempted sequences.
    #[default]
    Recompute,
    /// Swap to CPU memory.
    Swap,
}

/// Parallel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Tensor parallel size.
    #[serde(default = "default_one")]
    pub tensor_parallel_size: usize,

    /// Pipeline parallel size.
    #[serde(default = "default_one")]
    pub pipeline_parallel_size: usize,

    /// Worker use ray (for distributed).
    #[serde(default)]
    pub worker_use_ray: bool,

    /// Maximum parallel loading workers.
    #[serde(default = "default_one")]
    pub max_parallel_loading_workers: usize,

    /// Disable custom all-reduce.
    #[serde(default)]
    pub disable_custom_all_reduce: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            worker_use_ray: false,
            max_parallel_loading_workers: 1,
            disable_custom_all_reduce: false,
        }
    }
}

impl ParallelConfig {
    /// Validate parallel config.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.tensor_parallel_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "parallel.tensor_parallel_size".to_string(),
                message: "must be > 0".to_string(),
            });
        }
        if self.pipeline_parallel_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "parallel.pipeline_parallel_size".to_string(),
                message: "must be > 0".to_string(),
            });
        }
        Ok(())
    }

    /// Get world size.
    pub fn world_size(&self) -> usize {
        self.tensor_parallel_size * self.pipeline_parallel_size
    }
}

/// LoRA configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Maximum number of LoRA adapters.
    #[serde(default = "default_max_loras")]
    pub max_loras: usize,

    /// Maximum LoRA rank.
    #[serde(default = "default_max_lora_rank")]
    pub max_lora_rank: usize,

    /// LoRA extra vocabulary size.
    #[serde(default)]
    pub lora_extra_vocab_size: usize,

    /// LoRA dtype.
    #[serde(default)]
    pub lora_dtype: Option<DType>,

    /// Maximum CPU LoRAs.
    #[serde(default)]
    pub max_cpu_loras: Option<usize>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            max_loras: 1,
            max_lora_rank: 16,
            lora_extra_vocab_size: 256,
            lora_dtype: None,
            max_cpu_loras: None,
        }
    }
}

impl LoraConfig {
    /// Validate LoRA config.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_loras == 0 {
            return Err(ConfigError::InvalidValue {
                field: "lora.max_loras".to_string(),
                message: "must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

/// Speculative decoding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Draft model path.
    pub draft_model: PathBuf,

    /// Number of speculative tokens.
    #[serde(default = "default_num_speculative_tokens")]
    pub num_speculative_tokens: usize,

    /// Speculative decoding method.
    #[serde(default)]
    pub method: SpeculativeMethod,

    /// Draft model tensor parallel size.
    #[serde(default = "default_one")]
    pub draft_tensor_parallel_size: usize,
}

impl SpeculativeConfig {
    /// Validate speculative config.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.num_speculative_tokens == 0 {
            return Err(ConfigError::InvalidValue {
                field: "speculative.num_speculative_tokens".to_string(),
                message: "must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

/// Speculative decoding method.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeculativeMethod {
    /// Standard draft model.
    #[default]
    Draft,
    /// Medusa heads.
    Medusa,
    /// MLPSpeculator.
    Mlp,
    /// N-gram speculation.
    Ngram,
}

/// Observability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Enable metrics collection.
    #[serde(default = "default_true")]
    pub enable_metrics: bool,

    /// Prometheus metrics port.
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    /// Enable tracing.
    #[serde(default)]
    pub enable_tracing: bool,

    /// Log level.
    #[serde(default)]
    pub log_level: LogLevel,

    /// Disable logging.
    #[serde(default)]
    pub disable_log_stats: bool,

    /// Log stats interval (seconds).
    #[serde(default = "default_log_stats_interval")]
    pub log_stats_interval_secs: u64,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_port: 9090,
            enable_tracing: false,
            log_level: LogLevel::Info,
            disable_log_stats: false,
            log_stats_interval_secs: 10,
        }
    }
}

/// Log level.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Trace level.
    Trace,
    /// Debug level.
    Debug,
    /// Info level.
    #[default]
    Info,
    /// Warn level.
    Warn,
    /// Error level.
    Error,
}

impl LogLevel {
    /// Convert to tracing level.
    pub fn to_tracing_level(&self) -> tracing::Level {
        match self {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

/// API server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Host to bind to.
    #[serde(default = "default_host")]
    pub host: String,

    /// Port to bind to.
    #[serde(default = "default_port")]
    pub port: u16,

    /// API key for authentication.
    #[serde(default)]
    pub api_key: Option<String>,

    /// Maximum request size (bytes).
    #[serde(default = "default_max_request_size")]
    pub max_request_size: usize,

    /// Request timeout (seconds).
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Enable CORS.
    #[serde(default = "default_true")]
    pub enable_cors: bool,

    /// Allowed origins for CORS.
    #[serde(default)]
    pub cors_origins: Vec<String>,

    /// Enable SSL/TLS.
    #[serde(default)]
    pub enable_ssl: bool,

    /// SSL certificate path.
    #[serde(default)]
    pub ssl_cert_path: Option<PathBuf>,

    /// SSL key path.
    #[serde(default)]
    pub ssl_key_path: Option<PathBuf>,

    /// Response token limit.
    #[serde(default = "default_response_token_limit")]
    pub response_token_limit: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            api_key: None,
            max_request_size: 10 * 1024 * 1024, // 10 MB
            request_timeout_secs: 600,          // 10 minutes
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
            enable_ssl: false,
            ssl_cert_path: None,
            ssl_key_path: None,
            response_token_limit: 4096,
        }
    }
}

impl ApiConfig {
    /// Get socket address.
    pub fn socket_addr(&self) -> std::net::SocketAddr {
        use std::net::ToSocketAddrs;
        format!("{}:{}", self.host, self.port)
            .to_socket_addrs()
            .expect("Invalid host/port")
            .next()
            .expect("No addresses found")
    }
}

/// Configuration error.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid value for {field}: {message}")]
    InvalidValue { field: String, message: String },

    #[error("Failed to parse config: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("YAML error: {0}")]
    YamlError(String),

    #[error("TOML error: {0}")]
    TomlError(String),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

// Default value functions
fn default_true() -> bool { true }
fn default_one() -> usize { 1 }
fn default_max_model_len() -> usize { DEFAULT_MAX_MODEL_LEN }
fn default_block_size() -> usize { DEFAULT_BLOCK_SIZE }
fn default_gpu_memory_utilization() -> f64 { DEFAULT_GPU_MEMORY_UTILIZATION }
fn default_swap_space_gb() -> f64 { DEFAULT_SWAP_SPACE_GB }
fn default_max_num_batched_tokens() -> usize { DEFAULT_MAX_NUM_BATCHED_TOKENS }
fn default_max_num_seqs() -> usize { DEFAULT_MAX_NUM_SEQS }
fn default_max_paddings() -> usize { 512 }
fn default_max_loras() -> usize { 1 }
fn default_max_lora_rank() -> usize { 16 }
fn default_num_speculative_tokens() -> usize { 5 }
fn default_metrics_port() -> u16 { 9090 }
fn default_log_stats_interval() -> u64 { 10 }
fn default_host() -> String { "0.0.0.0".to_string() }
fn default_port() -> u16 { 8000 }
fn default_max_request_size() -> usize { 10 * 1024 * 1024 }
fn default_request_timeout() -> u64 { 600 }
fn default_response_token_limit() -> usize { 4096 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EngineConfig::default();
        assert_eq!(config.cache.block_size, DEFAULT_BLOCK_SIZE);
        assert_eq!(config.scheduler.max_num_seqs, DEFAULT_MAX_NUM_SEQS);
    }

    #[test]
    fn test_model_config() {
        let config = ModelConfig::with_path("/path/to/model");
        assert_eq!(config.model_path, PathBuf::from("/path/to/model"));
        assert!(config.use_flash_attention);
    }

    #[test]
    fn test_validation() {
        let mut config = EngineConfig::default();
        config.model.model_path = PathBuf::from("/model");
        assert!(config.validate().is_ok());

        config.cache.gpu_memory_utilization = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_parallel_world_size() {
        let config = ParallelConfig {
            tensor_parallel_size: 2,
            pipeline_parallel_size: 4,
            ..Default::default()
        };
        assert_eq!(config.world_size(), 8);
    }

    #[test]
    fn test_api_config() {
        let config = ApiConfig::default();
        assert_eq!(config.port, 8000);
        assert!(config.enable_cors);
    }

    #[test]
    fn test_serialization() {
        let config = EngineConfig::with_model("/model");
        let json = serde_json::to_string(&config).unwrap();
        let parsed: EngineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model.model_path, config.model.model_path);
    }
}
