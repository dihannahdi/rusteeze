//! Model configuration.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ModelConfig {
    /// Path to the model (local path or HuggingFace repo ID).
    #[validate(length(min = 1, message = "Model path cannot be empty"))]
    pub path: String,

    /// Model name/ID for API responses.
    #[serde(default)]
    pub name: Option<String>,

    /// Revision/branch for HuggingFace models.
    #[serde(default)]
    pub revision: Option<String>,

    /// Data type for model weights.
    #[serde(default = "default_dtype")]
    pub dtype: String,

    /// Quantization configuration.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,

    /// Device to run on.
    #[serde(default = "default_device")]
    pub device: String,

    /// Maximum model length.
    #[serde(default)]
    pub max_model_len: Option<usize>,

    /// Trust remote code (for custom models).
    #[serde(default)]
    pub trust_remote_code: bool,

    /// Download timeout in seconds.
    #[serde(default = "default_download_timeout")]
    pub download_timeout: u64,

    /// Use memory mapping for weights.
    #[serde(default = "default_true")]
    pub use_mmap: bool,

    /// Tokenizer path (if different from model).
    #[serde(default)]
    pub tokenizer_path: Option<String>,

    /// Chat template override.
    #[serde(default)]
    pub chat_template: Option<String>,

    /// Tensor parallelism degree.
    #[serde(default = "default_tp")]
    pub tensor_parallel_size: usize,

    /// Pipeline parallelism degree.
    #[serde(default = "default_pp")]
    pub pipeline_parallel_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            name: None,
            revision: None,
            dtype: "auto".to_string(),
            quantization: None,
            device: "auto".to_string(),
            max_model_len: None,
            trust_remote_code: false,
            download_timeout: 300,
            use_mmap: true,
            tokenizer_path: None,
            chat_template: None,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
        }
    }
}

fn default_dtype() -> String {
    "auto".to_string()
}

fn default_device() -> String {
    "auto".to_string()
}

fn default_download_timeout() -> u64 {
    300
}

fn default_true() -> bool {
    true
}

fn default_tp() -> usize {
    1
}

fn default_pp() -> usize {
    1
}

/// Quantization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method.
    pub method: QuantizationMethod,

    /// Bits for quantization.
    #[serde(default = "default_bits")]
    pub bits: u8,

    /// Group size for grouped quantization.
    #[serde(default = "default_group_size")]
    pub group_size: usize,

    /// Symmetric quantization.
    #[serde(default)]
    pub symmetric: bool,

    /// Use double quantization (QLoRA).
    #[serde(default)]
    pub double_quant: bool,

    /// Quantization dtype.
    #[serde(default)]
    pub quant_dtype: Option<String>,
}

fn default_bits() -> u8 {
    4
}

fn default_group_size() -> usize {
    128
}

/// Quantization method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMethod {
    /// No quantization.
    None,

    /// GPTQ quantization.
    Gptq,

    /// AWQ quantization.
    Awq,

    /// BitsAndBytes 4-bit.
    Bnb4,

    /// BitsAndBytes 8-bit.
    Bnb8,

    /// GGUF format.
    Gguf,

    /// Marlin (optimized GPTQ).
    Marlin,

    /// SqueezeLLM.
    SqueezeLlm,

    /// FP8 quantization.
    Fp8,
}

impl Default for QuantizationMethod {
    fn default() -> Self {
        Self::None
    }
}

impl ModelConfig {
    /// Create a new model config with path.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }

    /// Set the model name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the revision.
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Set the dtype.
    pub fn with_dtype(mut self, dtype: impl Into<String>) -> Self {
        self.dtype = dtype.into();
        self
    }

    /// Set quantization.
    pub fn with_quantization(mut self, config: QuantizationConfig) -> Self {
        self.quantization = Some(config);
        self
    }

    /// Set the device.
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    /// Set max model length.
    pub fn with_max_model_len(mut self, len: usize) -> Self {
        self.max_model_len = Some(len);
        self
    }

    /// Set tensor parallelism.
    pub fn with_tensor_parallel(mut self, size: usize) -> Self {
        self.tensor_parallel_size = size;
        self
    }

    /// Get the effective model name.
    pub fn effective_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.path)
    }

    /// Check if using distributed inference.
    pub fn is_distributed(&self) -> bool {
        self.tensor_parallel_size > 1 || self.pipeline_parallel_size > 1
    }

    /// Get total parallel degree.
    pub fn world_size(&self) -> usize {
        self.tensor_parallel_size * self.pipeline_parallel_size
    }
}
