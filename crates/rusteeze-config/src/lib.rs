//! Rusteeze Configuration Management.
//!
//! This crate provides configuration loading and validation for the Rusteeze
//! LLM inference engine. It supports YAML, TOML, and JSON configuration files,
//! as well as environment variable overrides.
//!
//! # Example
//!
//! ```rust,ignore
//! use rusteeze_config::{Config, ConfigLoader};
//!
//! // Load from file
//! let config = ConfigLoader::new()
//!     .with_file("config.yaml")
//!     .with_env_prefix("RUSTEEZE")
//!     .load()?;
//!
//! // Access configuration
//! println!("Model: {}", config.model.path);
//! println!("Port: {}", config.server.port);
//! ```

pub mod error;
pub mod model;
pub mod server;
pub mod engine;
pub mod loader;
pub mod validation;

pub use error::ConfigError;
pub use model::ModelConfig;
pub use server::ServerConfig;
pub use engine::EngineConfig;
pub use loader::ConfigLoader;

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Main configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct Config {
    /// Model configuration.
    #[validate(nested)]
    pub model: ModelConfig,

    /// Server configuration.
    #[validate(nested)]
    pub server: ServerConfig,

    /// Engine configuration.
    #[validate(nested)]
    pub engine: EngineConfig,

    /// Logging configuration.
    #[validate(nested)]
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Metrics configuration.
    #[serde(default)]
    pub metrics: MetricsConfig,
}

impl Config {
    /// Create a new configuration from file.
    pub fn from_file(path: &str) -> Result<Self, ConfigError> {
        ConfigLoader::new().with_file(path).load()
    }

    /// Create from environment variables.
    pub fn from_env() -> Result<Self, ConfigError> {
        ConfigLoader::new().with_env_prefix("RUSTEEZE").load()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        <Self as Validate>::validate(self).map_err(ConfigError::from)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            server: ServerConfig::default(),
            engine: EngineConfig::default(),
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LoggingConfig {
    /// Log level.
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Log format (json, pretty, compact).
    #[serde(default = "default_log_format")]
    pub format: String,

    /// Log file path (optional).
    pub file: Option<String>,

    /// Enable request logging.
    #[serde(default = "default_true")]
    pub request_logging: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
            file: None,
            request_logging: true,
        }
    }
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> String {
    "pretty".to_string()
}

fn default_true() -> bool {
    true
}

/// Metrics configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Metrics endpoint path.
    #[serde(default = "default_metrics_path")]
    pub path: String,

    /// Enable Prometheus export.
    #[serde(default = "default_true")]
    pub prometheus: bool,

    /// Histogram buckets for latency.
    #[serde(default = "default_buckets")]
    pub latency_buckets: Vec<f64>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/metrics".to_string(),
            prometheus: true,
            latency_buckets: vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        }
    }
}

fn default_metrics_path() -> String {
    "/metrics".to_string()
}

fn default_buckets() -> Vec<f64> {
    vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
}
