//! API request/response types.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model to use.
    pub model: String,

    /// Messages in conversation.
    pub messages: Vec<ChatMessage>,

    /// Maximum tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// Temperature (0.0-2.0).
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Top-p (nucleus sampling).
    #[serde(default)]
    pub top_p: Option<f32>,

    /// Number of completions to generate.
    #[serde(default = "default_n")]
    pub n: u32,

    /// Enable streaming.
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences.
    #[serde(default)]
    pub stop: Option<StopSequence>,

    /// Presence penalty (-2.0 to 2.0).
    #[serde(default)]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty (-2.0 to 2.0).
    #[serde(default)]
    pub frequency_penalty: Option<f32>,

    /// Log probabilities.
    #[serde(default)]
    pub logprobs: Option<bool>,

    /// Top log probabilities to return.
    #[serde(default)]
    pub top_logprobs: Option<u32>,

    /// User identifier.
    #[serde(default)]
    pub user: Option<String>,

    /// Seed for reproducibility.
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_n() -> u32 {
    1
}

/// Chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role.
    pub role: String,

    /// Message content.
    pub content: String,

    /// Optional name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Stop sequence (single or multiple).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    /// Single stop string.
    Single(String),
    /// Multiple stop strings.
    Multiple(Vec<String>),
}

impl StopSequence {
    /// Get as vector.
    pub fn as_vec(&self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s.clone()],
            Self::Multiple(v) => v.clone(),
        }
    }
}

/// Chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Response ID.
    pub id: String,

    /// Object type (always "chat.completion").
    pub object: String,

    /// Creation timestamp.
    pub created: i64,

    /// Model used.
    pub model: String,

    /// Choices.
    pub choices: Vec<ChatChoice>,

    /// Token usage.
    pub usage: Usage,

    /// System fingerprint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

impl ChatCompletionResponse {
    /// Create new response.
    pub fn new(
        model: String,
        choices: Vec<ChatChoice>,
        usage: Usage,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices,
            usage,
            system_fingerprint: Some("rusteeze-v1".to_string()),
        }
    }
}

/// Chat completion choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// Choice index.
    pub index: u32,

    /// Generated message.
    pub message: ChatMessage,

    /// Finish reason.
    pub finish_reason: Option<String>,

    /// Log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

/// Streaming chat completion chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Response ID.
    pub id: String,

    /// Object type (always "chat.completion.chunk").
    pub object: String,

    /// Creation timestamp.
    pub created: i64,

    /// Model used.
    pub model: String,

    /// Choices.
    pub choices: Vec<ChatChunkChoice>,

    /// System fingerprint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

impl ChatCompletionChunk {
    /// Create new chunk.
    pub fn new(id: String, model: String, choices: Vec<ChatChunkChoice>) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices,
            system_fingerprint: Some("rusteeze-v1".to_string()),
        }
    }
}

/// Streaming choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    /// Choice index.
    pub index: u32,

    /// Delta (incremental update).
    pub delta: ChatDelta,

    /// Finish reason.
    pub finish_reason: Option<String>,

    /// Log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
}

/// Delta for streaming.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatDelta {
    /// Role (first chunk only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,

    /// Content (incremental).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Log probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    /// Content log probs.
    pub content: Option<Vec<TokenLogProb>>,
}

/// Token log probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    /// Token string.
    pub token: String,

    /// Log probability.
    pub logprob: f32,

    /// Bytes (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,

    /// Top log probs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopLogProb>>,
}

/// Top log probability entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogProb {
    /// Token string.
    pub token: String,

    /// Log probability.
    pub logprob: f32,

    /// Bytes (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Token usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: u32,

    /// Completion tokens.
    pub completion_tokens: u32,

    /// Total tokens.
    pub total_tokens: u32,
}

/// Models list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    /// Object type.
    pub object: String,

    /// Model list.
    pub data: Vec<ModelInfo>,
}

/// Model info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID.
    pub id: String,

    /// Object type.
    pub object: String,

    /// Creation timestamp.
    pub created: i64,

    /// Owner.
    pub owned_by: String,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Status.
    pub status: String,

    /// Version.
    pub version: String,

    /// Uptime in seconds.
    pub uptime_seconds: u64,

    /// Requests processed.
    pub requests_processed: u64,

    /// Waiting requests.
    pub waiting_requests: usize,

    /// Running requests.
    pub running_requests: usize,

    /// GPU memory usage.
    pub gpu_memory_usage: f32,
}
