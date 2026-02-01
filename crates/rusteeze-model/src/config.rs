//! Model configuration parsing and validation.
//!
//! This module handles loading and parsing model configurations from
//! various sources (JSON, YAML, HuggingFace Hub).

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::info;

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    /// LLaMA architecture (1, 2, 3).
    Llama,
    /// Mistral architecture.
    Mistral,
    /// Mixtral MoE architecture.
    Mixtral,
    /// Qwen architecture.
    Qwen,
    /// Qwen2 architecture.
    Qwen2,
    /// Phi architecture.
    Phi,
    /// Phi-3 architecture.
    Phi3,
    /// Gemma architecture.
    Gemma,
    /// Gemma2 architecture.
    Gemma2,
    /// DeepSeek architecture.
    DeepSeek,
    /// DeepSeekV2 MoE architecture.
    DeepSeekV2,
    /// Falcon architecture.
    Falcon,
    /// Starcoder architecture.
    Starcoder,
    /// GPT-NeoX architecture.
    GptNeox,
    /// GPT-J architecture.
    Gptj,
    /// BLOOM architecture.
    Bloom,
    /// OPT architecture.
    Opt,
    /// InternLM architecture.
    InternLM,
    /// InternLM2 architecture.
    InternLM2,
    /// Yi architecture.
    Yi,
    /// Command-R architecture.
    CommandR,
    /// DBRX architecture.
    Dbrx,
    /// Custom/unknown architecture.
    Custom,
}

impl ModelArchitecture {
    /// Parse from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" | "llamaforcausallm" => Self::Llama,
            "mistral" | "mistralforcausallm" => Self::Mistral,
            "mixtral" | "mixtralforcausallm" => Self::Mixtral,
            "qwen" | "qwenforcausallm" => Self::Qwen,
            "qwen2" | "qwen2forcausallm" => Self::Qwen2,
            "phi" | "phiforcausallm" => Self::Phi,
            "phi3" | "phi3forcausallm" => Self::Phi3,
            "gemma" | "gemmaforcausallm" => Self::Gemma,
            "gemma2" | "gemma2forcausallm" => Self::Gemma2,
            "deepseek" | "deepseekforcausallm" => Self::DeepSeek,
            "deepseekv2" | "deepseekv2forcausallm" => Self::DeepSeekV2,
            "falcon" | "falconforcausallm" => Self::Falcon,
            "starcoder" | "starcoderforcausallm" => Self::Starcoder,
            "gptneox" | "gptneoxforcausallm" => Self::GptNeox,
            "gptj" | "gptjforcausallm" => Self::Gptj,
            "bloom" | "bloomforcausallm" => Self::Bloom,
            "opt" | "optforcausallm" => Self::Opt,
            "internlm" | "internlmforcausallm" => Self::InternLM,
            "internlm2" | "internlm2forcausallm" => Self::InternLM2,
            "yi" | "yiforcausallm" => Self::Yi,
            "commandr" | "commandrforcausallm" | "cohere" => Self::CommandR,
            "dbrx" | "dbrxforcausallm" => Self::Dbrx,
            _ => Self::Custom,
        }
    }

    /// Get default attention implementation.
    pub fn default_attention(&self) -> AttentionImplementation {
        match self {
            Self::Llama | Self::Mistral | Self::Qwen | Self::Qwen2 
            | Self::Phi | Self::Phi3 | Self::Gemma | Self::Gemma2
            | Self::Yi | Self::InternLM | Self::InternLM2 => AttentionImplementation::FlashAttention,
            Self::Falcon | Self::GptNeox | Self::Bloom => AttentionImplementation::Eager,
            _ => AttentionImplementation::Sdpa,
        }
    }
}

/// Attention implementation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AttentionImplementation {
    /// Eager (basic) attention.
    Eager,
    /// Flash Attention.
    FlashAttention,
    /// Scaled dot-product attention.
    #[default]
    Sdpa,
    /// Paged attention.
    Paged,
}

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture.
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Hidden size.
    pub hidden_size: usize,

    /// Intermediate size (FFN).
    pub intermediate_size: usize,

    /// Number of hidden layers.
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA).
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Maximum position embeddings.
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,

    /// RMS norm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Layer norm epsilon (alternative).
    #[serde(default)]
    pub layer_norm_eps: Option<f64>,

    /// RoPE theta.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// RoPE scaling configuration.
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,

    /// Sliding window attention size.
    #[serde(default)]
    pub sliding_window: Option<usize>,

    /// Attention dropout.
    #[serde(default)]
    pub attention_dropout: f64,

    /// Hidden activation function.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// Model type/name.
    #[serde(default)]
    pub model_type: Option<String>,

    /// Torch dtype.
    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Tie word embeddings.
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,

    /// BOS token ID.
    #[serde(default)]
    pub bos_token_id: Option<u32>,

    /// EOS token ID.
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,

    /// PAD token ID.
    #[serde(default)]
    pub pad_token_id: Option<u32>,

    /// Use cache.
    #[serde(default = "default_true")]
    pub use_cache: bool,

    /// MLP bias.
    #[serde(default)]
    pub mlp_bias: bool,

    /// Attention bias.
    #[serde(default)]
    pub attention_bias: bool,

    /// Head dimension (if different from hidden_size / num_heads).
    #[serde(default)]
    pub head_dim: Option<usize>,

    /// Number of experts (for MoE).
    #[serde(default)]
    pub num_experts: Option<usize>,

    /// Number of experts per token (for MoE).
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,

    /// Additional fields.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// EOS token ID (can be single or multiple).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EosTokenId {
    /// Single EOS token.
    Single(u32),
    /// Multiple EOS tokens.
    Multiple(Vec<u32>),
}

impl EosTokenId {
    /// Get as vector.
    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            EosTokenId::Single(id) => vec![*id],
            EosTokenId::Multiple(ids) => ids.clone(),
        }
    }

    /// Check if contains a token.
    pub fn contains(&self, token_id: u32) -> bool {
        match self {
            EosTokenId::Single(id) => *id == token_id,
            EosTokenId::Multiple(ids) => ids.contains(&token_id),
        }
    }
}

/// RoPE scaling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    /// Scaling type.
    #[serde(rename = "type")]
    pub scaling_type: String,

    /// Scaling factor.
    pub factor: f32,

    /// Original max position (for some scaling types).
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,

    /// Low frequency factor (for YaRN).
    #[serde(default)]
    pub low_freq_factor: Option<f32>,

    /// High frequency factor (for YaRN).
    #[serde(default)]
    pub high_freq_factor: Option<f32>,

    /// Attention factor (for LongRoPE).
    #[serde(default)]
    pub attention_factor: Option<f32>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec![],
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            layer_norm_eps: None,
            rope_theta: 10000.0,
            rope_scaling: None,
            sliding_window: None,
            attention_dropout: 0.0,
            hidden_act: "silu".to_string(),
            model_type: None,
            torch_dtype: None,
            tie_word_embeddings: true,
            bos_token_id: Some(1),
            eos_token_id: Some(EosTokenId::Single(2)),
            pad_token_id: None,
            use_cache: true,
            mlp_bias: false,
            attention_bias: false,
            head_dim: None,
            num_experts: None,
            num_experts_per_tok: None,
            extra: HashMap::new(),
        }
    }
}

impl ModelConfig {
    /// Load from JSON file.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        Self::from_json_str(&content)
    }

    /// Load from JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, ConfigError> {
        serde_json::from_str(json)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// Get the architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        self.architectures
            .first()
            .map(|s| ModelArchitecture::from_str(s))
            .unwrap_or_else(|| {
                self.model_type
                    .as_ref()
                    .map(|s| ModelArchitecture::from_str(s))
                    .unwrap_or(ModelArchitecture::Custom)
            })
    }

    /// Get number of KV heads (defaults to num_attention_heads).
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Get epsilon for normalization.
    pub fn norm_eps(&self) -> f64 {
        self.layer_norm_eps.unwrap_or(self.rms_norm_eps)
    }

    /// Check if using GQA.
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads() < self.num_attention_heads
    }

    /// Check if using MQA.
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads() == 1
    }

    /// Check if MoE model.
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some() && self.num_experts.unwrap() > 1
    }

    /// Get EOS token IDs.
    pub fn eos_token_ids(&self) -> Vec<u32> {
        self.eos_token_id
            .as_ref()
            .map(|e| e.to_vec())
            .unwrap_or_default()
    }

    /// Get memory estimate for model weights (bytes).
    pub fn estimate_weight_memory(&self, dtype_bytes: usize) -> usize {
        let vocab_embed = self.vocab_size * self.hidden_size;
        let lm_head = if self.tie_word_embeddings { 0 } else { self.vocab_size * self.hidden_size };

        let per_layer = {
            let qkv = self.hidden_size * (self.num_attention_heads + 2 * self.num_kv_heads()) * self.head_dim();
            let o_proj = self.num_attention_heads * self.head_dim() * self.hidden_size;
            let gate = self.hidden_size * self.intermediate_size;
            let up = self.hidden_size * self.intermediate_size;
            let down = self.intermediate_size * self.hidden_size;
            let norms = 2 * self.hidden_size;
            qkv + o_proj + gate + up + down + norms
        };

        let total_params = vocab_embed + lm_head + self.num_hidden_layers * per_layer + self.hidden_size;
        total_params * dtype_bytes
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.hidden_size == 0 {
            return Err(ConfigError::InvalidValue("hidden_size must be > 0".to_string()));
        }
        if self.num_hidden_layers == 0 {
            return Err(ConfigError::InvalidValue("num_hidden_layers must be > 0".to_string()));
        }
        if self.num_attention_heads == 0 {
            return Err(ConfigError::InvalidValue("num_attention_heads must be > 0".to_string()));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ConfigError::InvalidValue(
                "hidden_size must be divisible by num_attention_heads".to_string()
            ));
        }
        if self.num_attention_heads % self.num_kv_heads() != 0 {
            return Err(ConfigError::InvalidValue(
                "num_attention_heads must be divisible by num_key_value_heads".to_string()
            ));
        }
        Ok(())
    }
}

/// Configuration error.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    IoError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Missing field: {0}")]
    MissingField(String),
}

// Default value functions
fn default_max_position() -> usize { 4096 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_rope_theta() -> f64 { 10000.0 }
fn default_hidden_act() -> String { "silu".to_string() }
fn default_true() -> bool { true }

/// Tokenizer configuration (for chat templates).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenizerConfig {
    /// Chat template (Jinja2).
    #[serde(default)]
    pub chat_template: Option<String>,

    /// BOS token.
    #[serde(default)]
    pub bos_token: Option<String>,

    /// EOS token.
    #[serde(default)]
    pub eos_token: Option<String>,

    /// PAD token.
    #[serde(default)]
    pub pad_token: Option<String>,

    /// UNK token.
    #[serde(default)]
    pub unk_token: Option<String>,

    /// Add BOS token.
    #[serde(default)]
    pub add_bos_token: Option<bool>,

    /// Add EOS token.
    #[serde(default)]
    pub add_eos_token: Option<bool>,
}

impl TokenizerConfig {
    /// Load from JSON file.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        serde_json::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }
}

/// Generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum length.
    #[serde(default)]
    pub max_length: Option<usize>,

    /// Maximum new tokens.
    #[serde(default)]
    pub max_new_tokens: Option<usize>,

    /// Minimum length.
    #[serde(default)]
    pub min_length: Option<usize>,

    /// Temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p.
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k.
    #[serde(default)]
    pub top_k: Option<usize>,

    /// Repetition penalty.
    #[serde(default = "default_one_f32")]
    pub repetition_penalty: f32,

    /// Do sample.
    #[serde(default = "default_true")]
    pub do_sample: bool,

    /// Num beams.
    #[serde(default = "default_one_usize")]
    pub num_beams: usize,

    /// EOS token ID.
    #[serde(default)]
    pub eos_token_id: Option<EosTokenId>,

    /// PAD token ID.
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: Some(2048),
            max_new_tokens: Some(256),
            min_length: None,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            do_sample: true,
            num_beams: 1,
            eos_token_id: None,
            pad_token_id: None,
        }
    }
}

impl GenerationConfig {
    /// Load from JSON file.
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        serde_json::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }
}

fn default_temperature() -> f32 { 1.0 }
fn default_top_p() -> f32 { 1.0 }
fn default_one_f32() -> f32 { 1.0 }
fn default_one_usize() -> usize { 1 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_architecture_parsing() {
        assert_eq!(ModelArchitecture::from_str("LlamaForCausalLM"), ModelArchitecture::Llama);
        assert_eq!(ModelArchitecture::from_str("mistral"), ModelArchitecture::Mistral);
        assert_eq!(ModelArchitecture::from_str("Qwen2ForCausalLM"), ModelArchitecture::Qwen2);
    }

    #[test]
    fn test_gqa_detection() {
        let mut config = ModelConfig::default();
        config.num_key_value_heads = Some(8);
        assert!(config.is_gqa());
        assert!(!config.is_mqa());

        config.num_key_value_heads = Some(1);
        assert!(config.is_mqa());
    }

    #[test]
    fn test_eos_token_id() {
        let single = EosTokenId::Single(2);
        assert!(single.contains(2));
        assert!(!single.contains(3));
        assert_eq!(single.to_vec(), vec![2]);

        let multiple = EosTokenId::Multiple(vec![2, 32000]);
        assert!(multiple.contains(2));
        assert!(multiple.contains(32000));
        assert!(!multiple.contains(3));
    }

    #[test]
    fn test_memory_estimate() {
        let config = ModelConfig {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            ..Default::default()
        };

        // FP16: 2 bytes per param
        let mem = config.estimate_weight_memory(2);
        // Should be around 13-14 GB for a 7B model
        assert!(mem > 10_000_000_000);
        assert!(mem < 20_000_000_000);
    }

    #[test]
    fn test_config_validation() {
        let mut config = ModelConfig::default();

        // Valid
        assert!(config.validate().is_ok());

        // Invalid: hidden_size not divisible by num_heads
        config.hidden_size = 4097;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_json_parsing() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "model_type": "llama"
        }"#;

        let config = ModelConfig::from_json_str(json).unwrap();
        assert_eq!(config.architecture(), ModelArchitecture::Llama);
        assert_eq!(config.num_kv_heads(), 8);
        assert!(config.is_gqa());
    }
}
