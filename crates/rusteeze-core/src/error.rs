//! Error types for the Rusteeze inference engine.
//!
//! This module provides a comprehensive error hierarchy that captures all possible
//! failure modes in the inference pipeline, from configuration issues to runtime errors.

use std::fmt;
use thiserror::Error;

/// Specialized Result type for Rusteeze operations.
pub type Result<T> = std::result::Result<T, Error>;

/// The main error type for Rusteeze operations.
///
/// This error type provides detailed context about failures and is designed
/// to be both machine-readable (via error codes) and human-readable (via messages).
#[derive(Error, Debug)]
pub enum Error {
    /// Configuration-related errors
    #[error("Configuration error: {message}")]
    Config {
        /// Detailed error message
        message: String,
        /// Optional source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Model loading and initialization errors
    #[error("Model error: {message}")]
    Model {
        /// Detailed error message
        message: String,
        /// Model identifier if available
        model_id: Option<String>,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Tokenizer-related errors
    #[error("Tokenizer error: {message}")]
    Tokenizer {
        /// Detailed error message
        message: String,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Tensor operation errors
    #[error("Tensor error: {message}")]
    Tensor {
        /// Detailed error message
        message: String,
        /// Expected shape if applicable
        expected_shape: Option<Vec<usize>>,
        /// Actual shape if applicable
        actual_shape: Option<Vec<usize>>,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Device-related errors (GPU, CPU)
    #[error("Device error: {message}")]
    Device {
        /// Detailed error message
        message: String,
        /// Device identifier
        device_id: Option<usize>,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Memory allocation and management errors
    #[error("Memory error: {message}")]
    Memory {
        /// Detailed error message
        message: String,
        /// Requested bytes
        requested_bytes: Option<usize>,
        /// Available bytes
        available_bytes: Option<usize>,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Inference execution errors
    #[error("Inference error: {message}")]
    Inference {
        /// Detailed error message
        message: String,
        /// Request ID if available
        request_id: Option<String>,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Request validation errors
    #[error("Validation error: {message}")]
    Validation {
        /// Detailed error message
        message: String,
        /// Field that failed validation
        field: Option<String>,
        /// Validation rule that was violated
        rule: Option<String>,
    },

    /// Scheduling and batching errors
    #[error("Scheduler error: {message}")]
    Scheduler {
        /// Detailed error message
        message: String,
        /// Queue length at time of error
        queue_length: Option<usize>,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Cache-related errors (KV cache, etc.)
    #[error("Cache error: {message}")]
    Cache {
        /// Detailed error message
        message: String,
        /// Cache key if applicable
        cache_key: Option<String>,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Network and HTTP errors
    #[error("Network error: {message}")]
    Network {
        /// Detailed error message
        message: String,
        /// HTTP status code if applicable
        status_code: Option<u16>,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Timeout errors
    #[error("Timeout error: operation timed out after {duration_ms}ms")]
    Timeout {
        /// Operation that timed out
        operation: String,
        /// Duration in milliseconds
        duration_ms: u64,
    },

    /// Resource exhaustion errors
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted {
        /// Resource type that was exhausted
        resource: String,
        /// Optional limit
        limit: Option<usize>,
    },

    /// Internal errors (bugs, invariant violations)
    #[error("Internal error: {message}")]
    Internal {
        /// Detailed error message
        message: String,
        /// Optional backtrace location
        location: Option<String>,
        /// Source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// IO errors
    #[error("IO error: {message}")]
    Io {
        /// Detailed error message
        message: String,
        /// Path involved if applicable
        path: Option<String>,
        /// Source error
        #[source]
        source: Option<std::io::Error>,
    },

    /// Cancellation
    #[error("Operation cancelled: {reason}")]
    Cancelled {
        /// Reason for cancellation
        reason: String,
    },

    /// Rate limiting
    #[error("Rate limit exceeded")]
    RateLimitExceeded {
        /// Requests per second limit
        limit: f64,
        /// Retry after (seconds)
        retry_after_secs: Option<f64>,
    },

    /// Unsupported operation or feature
    #[error("Unsupported: {feature}")]
    Unsupported {
        /// Feature that is not supported
        feature: String,
        /// Suggested alternative if any
        alternative: Option<String>,
    },
}

impl Error {
    /// Create a configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            source: None,
        }
    }

    /// Create a configuration error with source
    pub fn config_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Config {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create a model error
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model {
            message: message.into(),
            model_id: None,
            source: None,
        }
    }

    /// Create a model error with model ID
    pub fn model_with_id(message: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self::Model {
            message: message.into(),
            model_id: Some(model_id.into()),
            source: None,
        }
    }

    /// Create a tokenizer error
    pub fn tokenizer(message: impl Into<String>) -> Self {
        Self::Tokenizer {
            message: message.into(),
            source: None,
        }
    }

    /// Create a tensor error
    pub fn tensor(message: impl Into<String>) -> Self {
        Self::Tensor {
            message: message.into(),
            expected_shape: None,
            actual_shape: None,
            source: None,
        }
    }

    /// Create a tensor shape mismatch error
    pub fn tensor_shape_mismatch(
        message: impl Into<String>,
        expected: Vec<usize>,
        actual: Vec<usize>,
    ) -> Self {
        Self::Tensor {
            message: message.into(),
            expected_shape: Some(expected),
            actual_shape: Some(actual),
            source: None,
        }
    }

    /// Create a device error
    pub fn device(message: impl Into<String>) -> Self {
        Self::Device {
            message: message.into(),
            device_id: None,
            source: None,
        }
    }

    /// Create a device error with ID
    pub fn device_with_id(message: impl Into<String>, device_id: usize) -> Self {
        Self::Device {
            message: message.into(),
            device_id: Some(device_id),
            source: None,
        }
    }

    /// Create a memory error
    pub fn memory(message: impl Into<String>) -> Self {
        Self::Memory {
            message: message.into(),
            requested_bytes: None,
            available_bytes: None,
            source: None,
        }
    }

    /// Create a memory allocation error with details
    pub fn memory_allocation(
        message: impl Into<String>,
        requested: usize,
        available: usize,
    ) -> Self {
        Self::Memory {
            message: message.into(),
            requested_bytes: Some(requested),
            available_bytes: Some(available),
            source: None,
        }
    }

    /// Create an inference error
    pub fn inference(message: impl Into<String>) -> Self {
        Self::Inference {
            message: message.into(),
            request_id: None,
            source: None,
        }
    }

    /// Create an inference error with request ID
    pub fn inference_with_request(message: impl Into<String>, request_id: impl Into<String>) -> Self {
        Self::Inference {
            message: message.into(),
            request_id: Some(request_id.into()),
            source: None,
        }
    }

    /// Create a validation error
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            field: None,
            rule: None,
        }
    }

    /// Create a validation error with field
    pub fn validation_field(message: impl Into<String>, field: impl Into<String>) -> Self {
        Self::Validation {
            message: message.into(),
            field: Some(field.into()),
            rule: None,
        }
    }

    /// Create a scheduler error
    pub fn scheduler(message: impl Into<String>) -> Self {
        Self::Scheduler {
            message: message.into(),
            queue_length: None,
            source: None,
        }
    }

    /// Create a cache error
    pub fn cache(message: impl Into<String>) -> Self {
        Self::Cache {
            message: message.into(),
            cache_key: None,
            source: None,
        }
    }

    /// Create a network error
    pub fn network(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
            status_code: None,
            source: None,
        }
    }

    /// Create a timeout error
    pub fn timeout(operation: impl Into<String>, duration_ms: u64) -> Self {
        Self::Timeout {
            operation: operation.into(),
            duration_ms,
        }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted(resource: impl Into<String>) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
            limit: None,
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
            location: None,
            source: None,
        }
    }

    /// Create an IO error
    pub fn io(message: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            message: message.into(),
            path: None,
            source: Some(source),
        }
    }

    /// Create an IO error with path
    pub fn io_with_path(
        message: impl Into<String>,
        path: impl Into<String>,
        source: std::io::Error,
    ) -> Self {
        Self::Io {
            message: message.into(),
            path: Some(path.into()),
            source: Some(source),
        }
    }

    /// Create a cancellation error
    pub fn cancelled(reason: impl Into<String>) -> Self {
        Self::Cancelled {
            reason: reason.into(),
        }
    }

    /// Create a rate limit error
    pub fn rate_limited(limit: f64, retry_after_secs: Option<f64>) -> Self {
        Self::RateLimitExceeded {
            limit,
            retry_after_secs,
        }
    }

    /// Create an unsupported feature error
    pub fn unsupported(feature: impl Into<String>) -> Self {
        Self::Unsupported {
            feature: feature.into(),
            alternative: None,
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::Network { .. }
                | Error::Timeout { .. }
                | Error::RateLimitExceeded { .. }
                | Error::Scheduler { .. }
        )
    }

    /// Get the error code for API responses
    pub fn error_code(&self) -> &'static str {
        match self {
            Error::Config { .. } => "CONFIG_ERROR",
            Error::Model { .. } => "MODEL_ERROR",
            Error::Tokenizer { .. } => "TOKENIZER_ERROR",
            Error::Tensor { .. } => "TENSOR_ERROR",
            Error::Device { .. } => "DEVICE_ERROR",
            Error::Memory { .. } => "MEMORY_ERROR",
            Error::Inference { .. } => "INFERENCE_ERROR",
            Error::Validation { .. } => "VALIDATION_ERROR",
            Error::Scheduler { .. } => "SCHEDULER_ERROR",
            Error::Cache { .. } => "CACHE_ERROR",
            Error::Network { .. } => "NETWORK_ERROR",
            Error::Timeout { .. } => "TIMEOUT_ERROR",
            Error::ResourceExhausted { .. } => "RESOURCE_EXHAUSTED",
            Error::Internal { .. } => "INTERNAL_ERROR",
            Error::Io { .. } => "IO_ERROR",
            Error::Cancelled { .. } => "CANCELLED",
            Error::RateLimitExceeded { .. } => "RATE_LIMIT_EXCEEDED",
            Error::Unsupported { .. } => "UNSUPPORTED",
        }
    }

    /// Get suggested HTTP status code for this error
    pub fn suggested_http_status(&self) -> u16 {
        match self {
            Error::Validation { .. } => 400,
            Error::RateLimitExceeded { .. } => 429,
            Error::Cancelled { .. } => 499,
            Error::Timeout { .. } => 504,
            Error::ResourceExhausted { .. } => 503,
            Error::Network { status_code: Some(code), .. } => *code,
            Error::Network { .. } => 502,
            Error::Internal { .. } => 500,
            Error::Unsupported { .. } => 501,
            _ => 500,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            message: err.to_string(),
            path: None,
            source: Some(err),
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Self::Validation {
            message: format!("JSON parsing error: {err}"),
            field: None,
            rule: Some("valid_json".to_string()),
        }
    }
}

impl From<uuid::Error> for Error {
    fn from(err: uuid::Error) -> Self {
        Self::Validation {
            message: format!("Invalid UUID: {err}"),
            field: None,
            rule: Some("valid_uuid".to_string()),
        }
    }
}

/// Extension trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn context(self, message: impl Into<String>) -> Result<T>;

    /// Add context to an error with a closure
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T, E: std::error::Error + Send + Sync + 'static> ErrorContext<T>
    for std::result::Result<T, E>
{
    fn context(self, message: impl Into<String>) -> Result<T> {
        self.map_err(|e| Error::Internal {
            message: message.into(),
            location: None,
            source: Some(Box::new(e)),
        })
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| Error::Internal {
            message: f(),
            location: None,
            source: Some(Box::new(e)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = Error::config("test error");
        assert_eq!(err.error_code(), "CONFIG_ERROR");
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_retryable_errors() {
        assert!(Error::timeout("test", 1000).is_retryable());
        assert!(Error::rate_limited(100.0, Some(1.0)).is_retryable());
        assert!(!Error::validation("invalid").is_retryable());
    }

    #[test]
    fn test_http_status_codes() {
        assert_eq!(Error::validation("invalid").suggested_http_status(), 400);
        assert_eq!(Error::rate_limited(100.0, None).suggested_http_status(), 429);
        assert_eq!(Error::internal("bug").suggested_http_status(), 500);
    }
}
