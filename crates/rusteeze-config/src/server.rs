//! Server configuration.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use validator::Validate;

/// Server configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServerConfig {
    /// Host to bind to.
    #[serde(default = "default_host")]
    pub host: String,

    /// Port to listen on.
    #[serde(default = "default_port")]
    #[validate(range(min = 1, max = 65535))]
    pub port: u16,

    /// API key for authentication.
    #[serde(default)]
    pub api_key: Option<String>,

    /// CORS configuration.
    #[serde(default)]
    pub cors: CorsConfig,

    /// TLS configuration.
    #[serde(default)]
    pub tls: Option<TlsConfig>,

    /// Request timeout in seconds.
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// Maximum concurrent connections.
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Keep-alive timeout in seconds.
    #[serde(default = "default_keepalive")]
    pub keepalive_seconds: u64,

    /// Rate limiting configuration.
    #[serde(default)]
    pub rate_limit: Option<RateLimitConfig>,

    /// Enable compression.
    #[serde(default = "default_true")]
    pub compression: bool,

    /// Maximum request body size in bytes.
    #[serde(default = "default_max_body_size")]
    pub max_body_size: usize,

    /// Worker threads (0 = auto).
    #[serde(default)]
    pub worker_threads: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            api_key: None,
            cors: CorsConfig::default(),
            tls: None,
            timeout_seconds: 300,
            max_connections: 10000,
            keepalive_seconds: 75,
            rate_limit: None,
            compression: true,
            max_body_size: 16 * 1024 * 1024, // 16MB
            worker_threads: 0,
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8000
}

fn default_timeout() -> u64 {
    300
}

fn default_max_connections() -> usize {
    10000
}

fn default_keepalive() -> u64 {
    75
}

fn default_true() -> bool {
    true
}

fn default_max_body_size() -> usize {
    16 * 1024 * 1024
}

/// CORS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    /// Enable CORS.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Allowed origins.
    #[serde(default = "default_origins")]
    pub allowed_origins: Vec<String>,

    /// Allowed methods.
    #[serde(default = "default_methods")]
    pub allowed_methods: Vec<String>,

    /// Allowed headers.
    #[serde(default = "default_headers")]
    pub allowed_headers: Vec<String>,

    /// Max age in seconds.
    #[serde(default = "default_max_age")]
    pub max_age: u64,

    /// Allow credentials.
    #[serde(default)]
    pub allow_credentials: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-Request-ID".to_string(),
            ],
            max_age: 86400,
            allow_credentials: false,
        }
    }
}

fn default_origins() -> Vec<String> {
    vec!["*".to_string()]
}

fn default_methods() -> Vec<String> {
    vec![
        "GET".to_string(),
        "POST".to_string(),
        "PUT".to_string(),
        "DELETE".to_string(),
        "OPTIONS".to_string(),
    ]
}

fn default_headers() -> Vec<String> {
    vec![
        "Content-Type".to_string(),
        "Authorization".to_string(),
        "X-Request-ID".to_string(),
    ]
}

fn default_max_age() -> u64 {
    86400
}

/// TLS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Certificate file path.
    pub cert_path: String,

    /// Key file path.
    pub key_path: String,

    /// CA certificate path for client auth.
    #[serde(default)]
    pub ca_path: Option<String>,

    /// Require client certificate.
    #[serde(default)]
    pub require_client_cert: bool,
}

/// Rate limiting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Requests per minute.
    #[serde(default = "default_rpm")]
    pub requests_per_minute: u32,

    /// Requests per day.
    #[serde(default)]
    pub requests_per_day: Option<u32>,

    /// Tokens per minute.
    #[serde(default)]
    pub tokens_per_minute: Option<u32>,

    /// Tokens per day.
    #[serde(default)]
    pub tokens_per_day: Option<u32>,

    /// Burst size.
    #[serde(default = "default_burst")]
    pub burst: u32,

    /// Rate limit by.
    #[serde(default)]
    pub by: RateLimitBy,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_minute: 60,
            requests_per_day: None,
            tokens_per_minute: None,
            tokens_per_day: None,
            burst: 10,
            by: RateLimitBy::ApiKey,
        }
    }
}

fn default_rpm() -> u32 {
    60
}

fn default_burst() -> u32 {
    10
}

/// Rate limit identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RateLimitBy {
    /// Rate limit by API key.
    #[default]
    ApiKey,

    /// Rate limit by IP address.
    Ip,

    /// Rate limit globally.
    Global,
}

impl ServerConfig {
    /// Create a new server config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the host.
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set the port.
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set the API key.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Set TLS configuration.
    pub fn with_tls(mut self, cert_path: impl Into<String>, key_path: impl Into<String>) -> Self {
        self.tls = Some(TlsConfig {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
            ca_path: None,
            require_client_cert: false,
        });
        self
    }

    /// Get the bind address.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Get the timeout as Duration.
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_seconds)
    }

    /// Get keepalive as Duration.
    pub fn keepalive(&self) -> Duration {
        Duration::from_secs(self.keepalive_seconds)
    }

    /// Check if TLS is enabled.
    pub fn is_tls_enabled(&self) -> bool {
        self.tls.is_some()
    }

    /// Check if auth is required.
    pub fn is_auth_required(&self) -> bool {
        self.api_key.is_some()
    }
}
