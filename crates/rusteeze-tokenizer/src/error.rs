//! Tokenizer error types.

use thiserror::Error;

/// Tokenizer error.
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// Failed to load tokenizer.
    #[error("Failed to load tokenizer: {0}")]
    LoadError(String),

    /// Failed to encode text.
    #[error("Failed to encode text: {0}")]
    EncodeError(String),

    /// Failed to decode tokens.
    #[error("Failed to decode tokens: {0}")]
    DecodeError(String),

    /// Invalid token ID.
    #[error("Invalid token ID: {0}")]
    InvalidTokenId(u32),

    /// Missing special token.
    #[error("Missing special token: {0}")]
    MissingSpecialToken(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Chat template error.
    #[error("Chat template error: {0}")]
    ChatTemplateError(String),
}

impl From<tokenizers::Error> for TokenizerError {
    fn from(err: tokenizers::Error) -> Self {
        Self::LoadError(err.to_string())
    }
}
