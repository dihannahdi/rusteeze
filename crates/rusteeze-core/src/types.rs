//! Core types used throughout Rusteeze.
//!
//! This module contains fundamental type definitions that are shared
//! across all components of the inference engine.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A unique identifier for a request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub String);

impl RequestId {
    /// Generate a new unique request ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::now_v7().to_string())
    }

    /// Create a request ID from a string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// A unique identifier for a sequence (generation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SequenceId(pub u64);

impl SequenceId {
    /// Create a new sequence ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the numeric value.
    pub const fn value(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for SequenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seq_{}", self.0)
    }
}

impl From<u64> for SequenceId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

/// Token ID type (vocabulary index).
pub type TokenId = u32;

/// A position in a sequence.
pub type Position = usize;

/// A layer index in the model.
pub type LayerIdx = usize;

/// A head index in multi-head attention.
pub type HeadIdx = usize;

/// Timestamp in milliseconds since epoch.
pub type TimestampMs = u64;

/// Status of a generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    /// Request is waiting in the queue
    Pending,
    /// Request is currently being processed
    Running,
    /// Request completed successfully
    Completed,
    /// Request failed
    Failed,
    /// Request was cancelled
    Cancelled,
    /// Request timed out
    TimedOut,
}

impl RequestStatus {
    /// Check if this status represents a terminal state.
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            RequestStatus::Completed
                | RequestStatus::Failed
                | RequestStatus::Cancelled
                | RequestStatus::TimedOut
        )
    }

    /// Check if this status represents an active state.
    pub const fn is_active(&self) -> bool {
        matches!(self, RequestStatus::Pending | RequestStatus::Running)
    }
}

impl fmt::Display for RequestStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestStatus::Pending => write!(f, "pending"),
            RequestStatus::Running => write!(f, "running"),
            RequestStatus::Completed => write!(f, "completed"),
            RequestStatus::Failed => write!(f, "failed"),
            RequestStatus::Cancelled => write!(f, "cancelled"),
            RequestStatus::TimedOut => write!(f, "timed_out"),
        }
    }
}

/// Reason for finishing generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Reached the end-of-sequence token
    Stop,
    /// Reached the maximum length
    Length,
    /// Matched a stop sequence
    StopSequence,
    /// Generation was cancelled
    Cancelled,
    /// Generation was aborted
    Abort,
    /// Generation errored
    Error,
    /// Tool/function call was generated
    ToolCalls,
    /// Content was filtered
    ContentFilter,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::StopSequence => write!(f, "stop_sequence"),
            FinishReason::Cancelled => write!(f, "cancelled"),
            FinishReason::Abort => write!(f, "abort"),
            FinishReason::Error => write!(f, "error"),
            FinishReason::ToolCalls => write!(f, "tool_calls"),
            FinishReason::ContentFilter => write!(f, "content_filter"),
        }
    }
}

/// Role in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt
    System,
    /// User message
    User,
    /// Assistant response
    Assistant,
    /// Tool/function result
    Tool,
    /// Function (legacy)
    Function,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
            Role::Function => write!(f, "function"),
        }
    }
}

impl std::str::FromStr for Role {
    type Err = crate::error::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "tool" => Ok(Role::Tool),
            "function" => Ok(Role::Function),
            _ => Err(crate::error::Error::validation(format!(
                "Invalid role: {s}"
            ))),
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of tokens in the prompt
    pub prompt_tokens: u32,
    /// Number of tokens generated
    pub completion_tokens: u32,
    /// Total tokens (prompt + completion)
    pub total_tokens: u32,
    /// Number of tokens from cache (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

impl TokenUsage {
    /// Create a new token usage instance.
    pub const fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cached_tokens: None,
        }
    }

    /// Create with cached tokens.
    pub const fn with_cache(
        prompt_tokens: u32,
        completion_tokens: u32,
        cached_tokens: u32,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cached_tokens: Some(cached_tokens),
        }
    }

    /// Add completion tokens.
    pub fn add_completion(&mut self, tokens: u32) {
        self.completion_tokens += tokens;
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
    }
}

/// Performance metrics for a generation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: f64,
    /// Total generation time in milliseconds
    pub total_time_ms: f64,
    /// Tokens per second (generation rate)
    pub tokens_per_second: f64,
    /// Prefill time in milliseconds
    pub prefill_time_ms: f64,
    /// Decode time in milliseconds
    pub decode_time_ms: f64,
    /// Queue wait time in milliseconds
    pub queue_wait_ms: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from timing data.
    pub fn calculate(
        tokens_generated: u32,
        time_to_first_token_ms: f64,
        total_time_ms: f64,
        prefill_time_ms: f64,
        queue_wait_ms: f64,
    ) -> Self {
        let decode_time_ms = total_time_ms - prefill_time_ms;
        let tokens_per_second = if total_time_ms > 0.0 {
            (tokens_generated as f64) / (total_time_ms / 1000.0)
        } else {
            0.0
        };

        Self {
            time_to_first_token_ms,
            total_time_ms,
            tokens_per_second,
            prefill_time_ms,
            decode_time_ms,
            queue_wait_ms,
        }
    }
}

/// Log probability information for a token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProb {
    /// The token string
    pub token: String,
    /// The token ID
    pub token_id: TokenId,
    /// Log probability of the token
    pub logprob: f32,
    /// UTF-8 byte representation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Top log probabilities for a position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogProbs {
    /// The chosen token's log prob
    pub token_logprob: LogProb,
    /// Top-k alternatives
    pub top_logprobs: Vec<LogProb>,
}

/// Model architecture type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    /// Llama architecture (Llama, Llama 2, Code Llama, etc.)
    Llama,
    /// Mistral architecture
    Mistral,
    /// Mixtral (MoE)
    Mixtral,
    /// Phi architecture
    Phi,
    /// Qwen architecture
    Qwen,
    /// Gemma architecture
    Gemma,
    /// GPT-NeoX architecture
    GptNeox,
    /// Falcon architecture
    Falcon,
    /// MPT architecture
    Mpt,
    /// Bloom architecture
    Bloom,
    /// StarCoder architecture
    Starcoder,
    /// Custom/unknown architecture
    Custom,
}

impl fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelArchitecture::Llama => write!(f, "llama"),
            ModelArchitecture::Mistral => write!(f, "mistral"),
            ModelArchitecture::Mixtral => write!(f, "mixtral"),
            ModelArchitecture::Phi => write!(f, "phi"),
            ModelArchitecture::Qwen => write!(f, "qwen"),
            ModelArchitecture::Gemma => write!(f, "gemma"),
            ModelArchitecture::GptNeox => write!(f, "gpt_neox"),
            ModelArchitecture::Falcon => write!(f, "falcon"),
            ModelArchitecture::Mpt => write!(f, "mpt"),
            ModelArchitecture::Bloom => write!(f, "bloom"),
            ModelArchitecture::Starcoder => write!(f, "starcoder"),
            ModelArchitecture::Custom => write!(f, "custom"),
        }
    }
}

/// Quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationType {
    /// No quantization (full precision)
    None,
    /// 8-bit quantization
    Int8,
    /// 4-bit quantization
    Int4,
    /// GPTQ quantization
    Gptq,
    /// AWQ quantization
    Awq,
    /// GGUF/GGML quantization
    Gguf,
    /// FP8 quantization
    Fp8,
}

impl Default for QuantizationType {
    fn default() -> Self {
        Self::None
    }
}

impl fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantizationType::None => write!(f, "none"),
            QuantizationType::Int8 => write!(f, "int8"),
            QuantizationType::Int4 => write!(f, "int4"),
            QuantizationType::Gptq => write!(f, "gptq"),
            QuantizationType::Awq => write!(f, "awq"),
            QuantizationType::Gguf => write!(f, "gguf"),
            QuantizationType::Fp8 => write!(f, "fp8"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_generation() {
        let id1 = RequestId::new();
        let id2 = RequestId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.total_tokens, 150);

        let mut usage = TokenUsage::new(100, 0);
        usage.add_completion(25);
        assert_eq!(usage.completion_tokens, 25);
        assert_eq!(usage.total_tokens, 125);
    }

    #[test]
    fn test_request_status() {
        assert!(!RequestStatus::Pending.is_terminal());
        assert!(!RequestStatus::Running.is_terminal());
        assert!(RequestStatus::Completed.is_terminal());
        assert!(RequestStatus::Failed.is_terminal());
    }

    #[test]
    fn test_role_parsing() {
        assert_eq!("system".parse::<Role>().unwrap(), Role::System);
        assert_eq!("USER".parse::<Role>().unwrap(), Role::User);
        assert_eq!("Assistant".parse::<Role>().unwrap(), Role::Assistant);
    }
}
