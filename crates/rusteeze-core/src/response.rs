//! Response types for the Rusteeze API.
//!
//! These types are designed to be compatible with the OpenAI API specification
//! while providing additional Rusteeze-specific extensions.

use serde::{Deserialize, Serialize};

use crate::types::{FinishReason, PerformanceMetrics, Role, TokenUsage, TopLogProbs};

/// A chat completion response (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// Unique identifier for the completion
    pub id: String,

    /// Object type (always "chat.completion")
    pub object: String,

    /// Unix timestamp of creation
    pub created: u64,

    /// Model used for completion
    pub model: String,

    /// Unique system fingerprint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,

    /// List of completion choices
    pub choices: Vec<ChatCompletionChoice>,

    /// Token usage statistics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,

    /// Rusteeze-specific metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rusteeze_metrics: Option<PerformanceMetrics>,
}

impl ChatCompletionResponse {
    /// Create a new chat completion response.
    pub fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        choices: Vec<ChatCompletionChoice>,
        usage: Option<TokenUsage>,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: model.into(),
            system_fingerprint: None,
            choices,
            usage,
            rusteeze_metrics: None,
        }
    }

    /// Add performance metrics.
    pub fn with_metrics(mut self, metrics: PerformanceMetrics) -> Self {
        self.rusteeze_metrics = Some(metrics);
        self
    }

    /// Get the first choice's content if available.
    pub fn first_content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.message.content.as_deref())
    }
}

/// A choice in a chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    /// Index of this choice
    pub index: u32,

    /// The generated message
    pub message: AssistantMessage,

    /// Reason for stopping
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Log probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
}

/// An assistant message in a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// The role (always "assistant")
    pub role: Role,

    /// The content of the message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::request::ToolCall>>,

    /// Function call (legacy)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<crate::request::FunctionCall>,

    /// Audio content (for multimodal)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<serde_json::Value>,

    /// Refusal message (for content filtering)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl AssistantMessage {
    /// Create a new assistant message with content.
    pub fn with_content(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: None,
            function_call: None,
            audio: None,
            refusal: None,
        }
    }

    /// Create an empty assistant message.
    pub fn empty() -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            tool_calls: None,
            function_call: None,
            audio: None,
            refusal: None,
        }
    }

    /// Create a message with tool calls.
    pub fn with_tool_calls(tool_calls: Vec<crate::request::ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(tool_calls),
            function_call: None,
            audio: None,
            refusal: None,
        }
    }
}

/// Log probabilities in a chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatLogProbs {
    /// Log probabilities for each content token
    pub content: Vec<TokenLogProb>,
}

/// Log probability for a token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    /// The token string
    pub token: String,

    /// Log probability
    pub logprob: f32,

    /// UTF-8 byte representation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,

    /// Top alternative tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopTokenLogProb>>,
}

/// Top token log probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopTokenLogProb {
    /// The token string
    pub token: String,

    /// Log probability
    pub logprob: f32,

    /// UTF-8 byte representation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// A streaming chat completion chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique identifier
    pub id: String,

    /// Object type (always "chat.completion.chunk")
    pub object: String,

    /// Unix timestamp of creation
    pub created: u64,

    /// Model used
    pub model: String,

    /// System fingerprint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,

    /// Delta choices
    pub choices: Vec<ChatCompletionChunkChoice>,

    /// Usage (only in final chunk if requested)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl ChatCompletionChunk {
    /// Create a new streaming chunk.
    pub fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        choices: Vec<ChatCompletionChunkChoice>,
    ) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: model.into(),
            system_fingerprint: None,
            choices,
            usage: None,
        }
    }

    /// Create a final chunk with usage.
    pub fn final_with_usage(id: impl Into<String>, model: impl Into<String>, usage: TokenUsage) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: model.into(),
            system_fingerprint: None,
            choices: vec![],
            usage: Some(usage),
        }
    }

    /// Create a content delta chunk.
    pub fn content_delta(id: impl Into<String>, model: impl Into<String>, content: impl Into<String>, index: u32) -> Self {
        Self::new(
            id,
            model,
            vec![ChatCompletionChunkChoice {
                index,
                delta: ChatDelta {
                    role: None,
                    content: Some(content.into()),
                    tool_calls: None,
                    function_call: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
        )
    }

    /// Create a role delta chunk (typically the first chunk).
    pub fn role_delta(id: impl Into<String>, model: impl Into<String>, index: u32) -> Self {
        Self::new(
            id,
            model,
            vec![ChatCompletionChunkChoice {
                index,
                delta: ChatDelta {
                    role: Some(Role::Assistant),
                    content: None,
                    tool_calls: None,
                    function_call: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
        )
    }

    /// Create a finish chunk.
    pub fn finish(id: impl Into<String>, model: impl Into<String>, index: u32, reason: FinishReason) -> Self {
        Self::new(
            id,
            model,
            vec![ChatCompletionChunkChoice {
                index,
                delta: ChatDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    function_call: None,
                },
                finish_reason: Some(reason),
                logprobs: None,
            }],
        )
    }
}

/// A choice in a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkChoice {
    /// Index of this choice
    pub index: u32,

    /// Delta update
    pub delta: ChatDelta,

    /// Finish reason (only in final chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Log probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
}

/// Delta update in a streaming chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatDelta {
    /// Role (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,

    /// Content delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    /// Tool calls delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,

    /// Function call delta (legacy)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCallDelta>,
}

/// Tool call delta in streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Index of this tool call
    pub index: u32,

    /// Tool call ID (only in first chunk for this tool call)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Tool type (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub tool_type: Option<String>,

    /// Function delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// Function call delta in streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name (only in first chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Arguments delta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Legacy completion response (non-chat).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Unique identifier
    pub id: String,

    /// Object type (always "text_completion")
    pub object: String,

    /// Unix timestamp
    pub created: u64,

    /// Model used
    pub model: String,

    /// System fingerprint
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,

    /// Completion choices
    pub choices: Vec<CompletionChoice>,

    /// Token usage
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

/// A choice in a completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// The generated text
    pub text: String,

    /// Index of this choice
    pub index: u32,

    /// Log probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<CompletionLogProbs>,

    /// Finish reason
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

/// Log probabilities for a completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionLogProbs {
    /// Tokens
    pub tokens: Vec<String>,

    /// Token log probabilities
    pub token_logprobs: Vec<f32>,

    /// Top log probabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopLogProbs>>,

    /// Text offsets
    pub text_offset: Vec<u32>,
}

/// Embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Object type (always "list")
    pub object: String,

    /// Embedding data
    pub data: Vec<EmbeddingData>,

    /// Model used
    pub model: String,

    /// Token usage
    pub usage: EmbeddingUsage,
}

/// Individual embedding data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    /// Object type (always "embedding")
    pub object: String,

    /// Embedding vector
    pub embedding: EmbeddingVector,

    /// Index in the input
    pub index: u32,
}

/// Embedding vector (float or base64).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    /// Float array
    Float(Vec<f32>),

    /// Base64-encoded
    Base64(String),
}

/// Token usage for embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    /// Number of prompt tokens
    pub prompt_tokens: u32,

    /// Total tokens
    pub total_tokens: u32,
}

/// Model information response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,

    /// Object type (always "model")
    pub object: String,

    /// Creation timestamp
    pub created: u64,

    /// Owner/organization
    pub owned_by: String,

    /// Permission information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permission: Option<Vec<serde_json::Value>>,

    /// Root model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,

    /// Parent model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

/// Model list response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    /// Object type (always "list")
    pub object: String,

    /// List of models
    pub data: Vec<ModelInfo>,
}

/// Error response (OpenAI format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error details
    pub error: ErrorDetail,
}

impl ErrorResponse {
    /// Create a new error response.
    pub fn new(
        message: impl Into<String>,
        error_type: impl Into<String>,
        code: Option<String>,
    ) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                param: None,
                code,
            },
        }
    }

    /// Create from a Rusteeze error.
    pub fn from_error(err: &crate::error::Error) -> Self {
        Self::new(
            err.to_string(),
            err.error_code(),
            Some(err.error_code().to_string()),
        )
    }
}

/// Error detail in an error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Error message
    pub message: String,

    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,

    /// Parameter that caused the error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,

    /// Error code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Health status
    pub status: HealthStatus,

    /// Version information
    pub version: String,

    /// Uptime in seconds
    pub uptime_seconds: u64,

    /// Additional details
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<HealthDetails>,
}

/// Health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Service is healthy
    Healthy,
    /// Service is degraded but operational
    Degraded,
    /// Service is unhealthy
    Unhealthy,
}

/// Health check details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthDetails {
    /// Model loaded status
    pub model_loaded: bool,

    /// Number of active requests
    pub active_requests: u32,

    /// Pending requests in queue
    pub pending_requests: u32,

    /// GPU memory usage (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_used_bytes: Option<u64>,

    /// GPU memory total (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_total_bytes: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_response_creation() {
        let choice = ChatCompletionChoice {
            index: 0,
            message: AssistantMessage::with_content("Hello!"),
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        };

        let response = ChatCompletionResponse::new(
            "cmpl-123",
            "gpt-4",
            vec![choice],
            Some(TokenUsage::new(10, 5)),
        );

        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.first_content(), Some("Hello!"));
    }

    #[test]
    fn test_streaming_chunk_creation() {
        let chunk = ChatCompletionChunk::content_delta("cmpl-123", "gpt-4", "Hello", 0);
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
    }

    #[test]
    fn test_error_response() {
        let err = crate::error::Error::validation("Invalid parameter");
        let response = ErrorResponse::from_error(&err);
        assert_eq!(response.error.error_type, "VALIDATION_ERROR");
    }

    #[test]
    fn test_serialization() {
        let response = ChatCompletionResponse::new(
            "test",
            "model",
            vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage::with_content("Hello"),
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            None,
        );

        let json = serde_json::to_string(&response).unwrap();
        let parsed: ChatCompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, response.id);
    }
}
