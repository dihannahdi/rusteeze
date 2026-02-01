//! Configuration loader.

use std::path::Path;

use tracing::{debug, info};

use crate::error::ConfigError;
use crate::{Config, EngineConfig, ModelConfig, ServerConfig};

/// Configuration file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    Yaml,
    Toml,
    Json,
}

impl ConfigFormat {
    /// Detect format from file extension.
    pub fn from_extension(path: &str) -> Option<Self> {
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());

        match ext.as_deref() {
            Some("yaml") | Some("yml") => Some(Self::Yaml),
            Some("toml") => Some(Self::Toml),
            Some("json") => Some(Self::Json),
            _ => None,
        }
    }

    /// Parse content in this format.
    pub fn parse<T: serde::de::DeserializeOwned>(&self, content: &str) -> Result<T, ConfigError> {
        match self {
            Self::Yaml => serde_yaml::from_str(content).map_err(ConfigError::from),
            Self::Toml => toml::from_str(content).map_err(ConfigError::from),
            Self::Json => serde_json::from_str(content).map_err(ConfigError::from),
        }
    }
}

/// Configuration loader.
pub struct ConfigLoader {
    /// Config file path.
    file_path: Option<String>,

    /// Environment variable prefix.
    env_prefix: Option<String>,

    /// Default values.
    defaults: Config,
}

impl ConfigLoader {
    /// Create a new config loader.
    pub fn new() -> Self {
        Self {
            file_path: None,
            env_prefix: None,
            defaults: Config::default(),
        }
    }

    /// Set the config file path.
    pub fn with_file(mut self, path: impl Into<String>) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Set the environment variable prefix.
    pub fn with_env_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.env_prefix = Some(prefix.into());
        self
    }

    /// Set default values.
    pub fn with_defaults(mut self, defaults: Config) -> Self {
        self.defaults = defaults;
        self
    }

    /// Load the configuration.
    pub fn load(self) -> Result<Config, ConfigError> {
        // Start with defaults
        let mut config = self.defaults;

        // Load from file if specified
        if let Some(ref path) = self.file_path {
            config = self.load_from_file(path)?;
        }

        // Override with environment variables
        if let Some(ref prefix) = self.env_prefix {
            self.apply_env_overrides(&mut config, prefix)?;
        }

        // Validate
        config.validate()?;

        Ok(config)
    }

    /// Load from file.
    fn load_from_file(&self, path: &str) -> Result<Config, ConfigError> {
        debug!("Loading configuration from {}", path);

        // Check if file exists
        if !Path::new(path).exists() {
            return Err(ConfigError::FileNotFound(path.to_string()));
        }

        // Read content
        let content = std::fs::read_to_string(path)?;

        // Detect format
        let format = ConfigFormat::from_extension(path)
            .ok_or_else(|| ConfigError::UnsupportedFormat(path.to_string()))?;

        // Parse
        let config: Config = format.parse(&content)?;

        info!("Loaded configuration from {}", path);

        Ok(config)
    }

    /// Apply environment variable overrides.
    fn apply_env_overrides(&self, config: &mut Config, prefix: &str) -> Result<(), ConfigError> {
        // Model overrides
        if let Ok(val) = std::env::var(format!("{}_MODEL_PATH", prefix)) {
            config.model.path = val;
        }
        if let Ok(val) = std::env::var(format!("{}_MODEL_NAME", prefix)) {
            config.model.name = Some(val);
        }
        if let Ok(val) = std::env::var(format!("{}_MODEL_DTYPE", prefix)) {
            config.model.dtype = val;
        }
        if let Ok(val) = std::env::var(format!("{}_MODEL_DEVICE", prefix)) {
            config.model.device = val;
        }
        if let Ok(val) = std::env::var(format!("{}_MODEL_MAX_LEN", prefix)) {
            config.model.max_model_len = val.parse().ok();
        }
        if let Ok(val) = std::env::var(format!("{}_TENSOR_PARALLEL", prefix)) {
            if let Ok(tp) = val.parse() {
                config.model.tensor_parallel_size = tp;
            }
        }

        // Server overrides
        if let Ok(val) = std::env::var(format!("{}_HOST", prefix)) {
            config.server.host = val;
        }
        if let Ok(val) = std::env::var(format!("{}_PORT", prefix)) {
            if let Ok(port) = val.parse() {
                config.server.port = port;
            }
        }
        if let Ok(val) = std::env::var(format!("{}_API_KEY", prefix)) {
            config.server.api_key = Some(val);
        }

        // Engine overrides
        if let Ok(val) = std::env::var(format!("{}_MAX_SEQS", prefix)) {
            if let Ok(n) = val.parse() {
                config.engine.max_num_seqs = n;
            }
        }
        if let Ok(val) = std::env::var(format!("{}_MAX_TOKENS", prefix)) {
            if let Ok(n) = val.parse() {
                config.engine.max_num_batched_tokens = n;
            }
        }
        if let Ok(val) = std::env::var(format!("{}_GPU_MEMORY_UTIL", prefix)) {
            if let Ok(n) = val.parse() {
                config.engine.gpu_memory_utilization = n;
            }
        }

        // Logging overrides
        if let Ok(val) = std::env::var(format!("{}_LOG_LEVEL", prefix)) {
            config.logging.level = val;
        }

        Ok(())
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Load configuration from default locations.
pub fn load_default_config() -> Result<Config, ConfigError> {
    // Try common config file locations
    let paths = [
        "rusteeze.yaml",
        "rusteeze.yml",
        "rusteeze.toml",
        "config.yaml",
        "config.yml",
        "config.toml",
        "/etc/rusteeze/config.yaml",
    ];

    for path in &paths {
        if Path::new(path).exists() {
            return ConfigLoader::new()
                .with_file(*path)
                .with_env_prefix("RUSTEEZE")
                .load();
        }
    }

    // No config file found, use defaults with env overrides
    ConfigLoader::new().with_env_prefix("RUSTEEZE").load()
}

/// Builder for programmatic configuration.
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Create a new config builder.
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// Set model configuration.
    pub fn model(mut self, config: ModelConfig) -> Self {
        self.config.model = config;
        self
    }

    /// Set server configuration.
    pub fn server(mut self, config: ServerConfig) -> Self {
        self.config.server = config;
        self
    }

    /// Set engine configuration.
    pub fn engine(mut self, config: EngineConfig) -> Self {
        self.config.engine = config;
        self
    }

    /// Set model path.
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.config.model.path = path.into();
        self
    }

    /// Set server port.
    pub fn port(mut self, port: u16) -> Self {
        self.config.server.port = port;
        self
    }

    /// Set API key.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.server.api_key = Some(key.into());
        self
    }

    /// Build the configuration.
    pub fn build(self) -> Result<Config, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
