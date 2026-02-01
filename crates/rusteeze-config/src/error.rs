//! Configuration error types.

use thiserror::Error;
use validator::ValidationErrors;

/// Configuration error.
#[derive(Error, Debug)]
pub enum ConfigError {
    /// File not found.
    #[error("Configuration file not found: {0}")]
    FileNotFound(String),

    /// Parse error.
    #[error("Failed to parse configuration: {0}")]
    ParseError(String),

    /// Validation error.
    #[error("Configuration validation failed: {0}")]
    ValidationError(String),

    /// Environment variable error.
    #[error("Environment variable error: {0}")]
    EnvError(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// YAML parsing error.
    #[error("YAML parsing error: {0}")]
    YamlError(#[from] serde_yaml::Error),

    /// TOML parsing error.
    #[error("TOML parsing error: {0}")]
    TomlError(#[from] toml::de::Error),

    /// JSON parsing error.
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Missing required field.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid value.
    #[error("Invalid value for {field}: {message}")]
    InvalidValue { field: String, message: String },

    /// Unsupported format.
    #[error("Unsupported configuration format: {0}")]
    UnsupportedFormat(String),
}

impl From<ValidationErrors> for ConfigError {
    fn from(errors: ValidationErrors) -> Self {
        let messages: Vec<String> = errors
            .field_errors()
            .into_iter()
            .map(|(field, errors)| {
                let error_msgs: Vec<String> = errors
                    .iter()
                    .map(|e| {
                        e.message
                            .as_ref()
                            .map(|m| m.to_string())
                            .unwrap_or_else(|| format!("validation failed for {}", e.code))
                    })
                    .collect();
                format!("{}: {}", field, error_msgs.join(", "))
            })
            .collect();

        ConfigError::ValidationError(messages.join("; "))
    }
}

impl ConfigError {
    /// Create a missing field error.
    pub fn missing_field(field: impl Into<String>) -> Self {
        Self::MissingField(field.into())
    }

    /// Create an invalid value error.
    pub fn invalid_value(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidValue {
            field: field.into(),
            message: message.into(),
        }
    }
}
