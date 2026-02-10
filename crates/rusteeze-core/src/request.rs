//! Request types for the Rusteeze API.
//!
//! These types are designed to be compatible with the OpenAI API specification
//! while providing additional Rusteeze-specific extensions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::sampling::SamplingParams;
use crate::types::{Role, TokenId};

/// A chat completion request (OpenAI-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// ID of the model to use
    pub model: String,

    /// Messages comprising the conversation
    pub messages: Vec<ChatMessage>,

    /// Sampling temperature (0.0 to 2.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter (0.0 to 1.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling parameter
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Number of completions to generate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Maximum tokens to generate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Maximum completion tokens (alias for max_tokens)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Stream options (for streaming responses)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Stop sequences
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Repetition penalty (for some models)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Token bias modifications
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Whether to return log probabilities
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// Number of top log probabilities to return
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// Random seed for reproducibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// User identifier for rate limiting/abuse monitoring
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Tools available to the model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// How to handle tool calls
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Response format specification
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Rusteeze-specific extensions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rusteeze: Option<RusteezeExtensions>,
}

impl ChatCompletionRequest {
    /// Create a new chat completion request with the minimum required fields.
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            top_p: None,
            top_k: None,
            n: None,
            max_tokens: None,
            max_completion_tokens: None,
            stream: false,
            stream_options: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            seed: None,
            user: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            rusteeze: None,
        }
    }

    /// Get the effective max tokens.
    pub fn effective_max_tokens(&self) -> Option<u32> {
        self.max_tokens.or(self.max_completion_tokens)
    }

    /// Convert to sampling parameters.
    pub fn to_sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature.unwrap_or(1.0),
            top_p: self.top_p.unwrap_or(1.0),
            top_k: self.top_k.map(|k| k as i32).unwrap_or(-1),
            max_tokens: self.effective_max_tokens(),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            repetition_penalty: self.repetition_penalty.unwrap_or(1.0),
            stop: self.stop.clone().map(|s| s.into_vec()).unwrap_or_default(),
            seed: self.seed,
            logit_bias: self.logit_bias.clone()
                .map(|map| map.into_iter()
                    .filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, v)))
                    .collect())
                .unwrap_or_default(),
            n: self.n.unwrap_or(1),
            logprobs: self.top_logprobs,
            prompt_logprobs: if self.logprobs.unwrap_or(false) { Some(5) } else { None },
            ..Default::default()
        }
    }

    /// Validate the request.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.messages.is_empty() {
            return Err(crate::error::Error::validation_field(
                "Messages array cannot be empty",
                "messages",
            ));
        }

        if let Some(temp) = self.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(crate::error::Error::validation_field(
                    "Temperature must be between 0.0 and 2.0",
                    "temperature",
                ));
            }
        }

        if let Some(top_p) = self.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(crate::error::Error::validation_field(
                    "Top-p must be between 0.0 and 1.0",
                    "top_p",
                ));
            }
        }

        if let Some(n) = self.n {
            if n == 0 || n > 128 {
                return Err(crate::error::Error::validation_field(
                    "n must be between 1 and 128",
                    "n",
                ));
            }
        }

        if let Some(top_logprobs) = self.top_logprobs {
            if top_logprobs > 20 {
                return Err(crate::error::Error::validation_field(
                    "top_logprobs must be between 0 and 20",
                    "top_logprobs",
                ));
            }
        }

        Ok(())
    }
}

/// A message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message author
    pub role: Role,

    /// The content of the message
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,

    /// Name of the author (for function results)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool calls made by the assistant
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Tool call ID (for tool results)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Create a new system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a new user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a new assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(MessageContent::Text(content.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Get the text content if this is a simple text message.
    pub fn text_content(&self) -> Option<&str> {
        match &self.content {
            Some(MessageContent::Text(s)) => Some(s),
            _ => None,
        }
    }
}

/// Content of a message (can be text or multi-modal).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),
    /// Multi-modal content (text + images)
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Convert to a string representation.
    pub fn to_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// A part of multi-modal message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Text content
    Text {
        /// The text content
        text: String,
    },
    /// Image content
    ImageUrl {
        /// Image URL specification
        image_url: ImageUrl,
    },
}

/// Image URL specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// The URL or base64-encoded image
    pub url: String,
    /// Detail level for image processing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// Image processing detail level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Low resolution
    Low,
    /// High resolution
    High,
    /// Automatic selection
    Auto,
}

impl Default for ImageDetail {
    fn default() -> Self {
        Self::Auto
    }
}

/// Stop sequence specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    /// Single stop sequence
    Single(String),
    /// Multiple stop sequences
    Multiple(Vec<String>),
}

impl StopSequence {
    /// Convert to a vector of strings.
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequence::Single(s) => vec![s],
            StopSequence::Multiple(v) => v,
        }
    }
}

/// Stream options for chunked responses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamOptions {
    /// Include usage statistics in stream
    #[serde(default)]
    pub include_usage: bool,
}

/// Tool/function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool type (currently only "function")
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function definition
    pub function: FunctionDefinition,
}

/// Function definition for tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name
    pub name: String,
    /// Function description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Parameters schema (JSON Schema)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    /// Whether the function is strict
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Tool choice specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// String option ("none", "auto", "required")
    Mode(String),
    /// Specific tool to use
    Specific {
        /// Tool type
        #[serde(rename = "type")]
        tool_type: String,
        /// Function to call
        function: ToolChoiceFunction,
    },
}

/// Function specification for tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    /// Function name
    pub name: String,
}

/// Tool call made by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for the tool call
    pub id: String,
    /// Tool type
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Function call details
    pub function: FunctionCall,
}

/// Function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name
    pub name: String,
    /// Function arguments (JSON string)
    pub arguments: String,
}

/// Response format specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    /// Response type
    #[serde(rename = "type")]
    pub format_type: ResponseFormatType,
    /// JSON schema for structured output
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchema>,
}

/// Response format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatType {
    /// Plain text output
    Text,
    /// JSON output
    JsonObject,
    /// JSON with schema
    JsonSchema,
}

/// JSON schema specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSchema {
    /// Schema name
    pub name: String,
    /// Schema description
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON schema
    pub schema: serde_json::Value,
    /// Whether to enforce strict schema adherence
    #[serde(default)]
    pub strict: bool,
}

/// Rusteeze-specific request extensions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RusteezeExtensions {
    /// Priority level (higher = processed sooner)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,

    /// Request timeout in milliseconds
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,

    /// Whether to include timing metrics in response
    #[serde(default)]
    pub include_metrics: bool,

    /// Minimum number of tokens to generate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,

    /// Skip special tokens in output
    #[serde(default)]
    pub skip_special_tokens: bool,

    /// Token IDs for early stopping
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<TokenId>>,

    /// Use prefix caching
    #[serde(default)]
    pub use_prefix_cache: bool,

    /// Echo the prompt in the output
    #[serde(default)]
    pub echo: bool,
}

/// Legacy completion request (non-chat).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model to use
    pub model: String,

    /// Prompt to complete
    pub prompt: PromptInput,

    /// Suffix to append (for fill-in-the-middle)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    /// Maximum tokens to generate
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p sampling
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Number of completions
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Whether to stream
    #[serde(default)]
    pub stream: bool,

    /// Stop sequences
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,

    /// Presence penalty
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Best of (for non-streaming)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,

    /// Logit bias
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Return log probs
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,

    /// Echo the prompt
    #[serde(default)]
    pub echo: bool,

    /// Seed
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// User ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Prompt input (string or array).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    /// Single string prompt
    Single(String),
    /// Multiple prompts
    Multiple(Vec<String>),
    /// Token IDs
    TokenIds(Vec<TokenId>),
    /// Multiple token ID sequences
    MultipleTokenIds(Vec<Vec<TokenId>>),
}

impl PromptInput {
    /// Get the prompts as strings (if possible).
    pub fn as_strings(&self) -> Option<Vec<String>> {
        match self {
            PromptInput::Single(s) => Some(vec![s.clone()]),
            PromptInput::Multiple(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Get the prompts as token IDs (if available).
    pub fn as_token_ids(&self) -> Option<Vec<Vec<TokenId>>> {
        match self {
            PromptInput::TokenIds(ids) => Some(vec![ids.clone()]),
            PromptInput::MultipleTokenIds(v) => Some(v.clone()),
            _ => None,
        }
    }
}

/// Embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Model to use
    pub model: String,

    /// Input text(s)
    pub input: EmbeddingInput,

    /// Encoding format
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<EncodingFormat>,

    /// Number of dimensions (for truncation)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// User ID
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Embedding input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// Single string
    Single(String),
    /// Multiple strings
    Multiple(Vec<String>),
    /// Token IDs
    TokenIds(Vec<TokenId>),
    /// Multiple token ID sequences
    MultipleTokenIds(Vec<Vec<TokenId>>),
}

/// Encoding format for embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    /// Float array
    Float,
    /// Base64-encoded
    Base64,
}

impl Default for EncodingFormat {
    fn default() -> Self {
        Self::Float
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_creation() {
        let system = ChatMessage::system("You are a helpful assistant.");
        assert_eq!(system.role, Role::System);
        assert!(system.text_content().is_some());

        let user = ChatMessage::user("Hello!");
        assert_eq!(user.role, Role::User);
    }

    #[test]
    fn test_chat_request_validation() {
        let req = ChatCompletionRequest::new("gpt-4", vec![]);
        assert!(req.validate().is_err());

        let req = ChatCompletionRequest::new(
            "gpt-4",
            vec![ChatMessage::user("Hello")],
        );
        assert!(req.validate().is_ok());

        let mut req = ChatCompletionRequest::new(
            "gpt-4",
            vec![ChatMessage::user("Hello")],
        );
        req.temperature = Some(3.0);
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_stop_sequence() {
        let single: StopSequence = serde_json::from_str(r#""stop""#).unwrap();
        assert_eq!(single.into_vec(), vec!["stop"]);

        let multiple: StopSequence = serde_json::from_str(r#"["a", "b"]"#).unwrap();
        assert_eq!(multiple.into_vec(), vec!["a", "b"]);
    }

    #[test]
    fn test_message_content_deserialization() {
        let text: MessageContent = serde_json::from_str(r#""Hello""#).unwrap();
        assert!(matches!(text, MessageContent::Text(_)));

        let parts: MessageContent = serde_json::from_str(
            r#"[{"type": "text", "text": "Hello"}]"#,
        ).unwrap();
        assert!(matches!(parts, MessageContent::Parts(_)));
    }
}
