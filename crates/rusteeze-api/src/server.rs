//! API server.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio::signal;
use tracing::{error, info, warn};

use rusteeze_engine::Engine;

use crate::error::ApiError;
use crate::handlers::AppState;
use crate::routes::{create_router, RouterBuilder};

/// API server configuration.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Host to bind to.
    pub host: String,

    /// Port to listen on.
    pub port: u16,

    /// API key for authentication (optional).
    pub api_key: Option<String>,

    /// Enable CORS.
    pub cors_enabled: bool,

    /// Enable compression.
    pub compression_enabled: bool,

    /// Request timeout.
    pub request_timeout: Duration,

    /// Rate limit (requests per minute).
    pub rate_limit: Option<u32>,

    /// Max concurrent connections.
    pub max_connections: usize,

    /// Keep-alive timeout.
    pub keep_alive_timeout: Duration,

    /// Enable request logging.
    pub request_logging: bool,

    /// TLS configuration.
    pub tls: Option<TlsConfig>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            api_key: None,
            cors_enabled: true,
            compression_enabled: true,
            request_timeout: Duration::from_secs(300),
            rate_limit: None,
            max_connections: 10000,
            keep_alive_timeout: Duration::from_secs(75),
            request_logging: true,
            tls: None,
        }
    }
}

/// TLS configuration.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Certificate path.
    pub cert_path: String,

    /// Key path.
    pub key_path: String,
}

/// API server.
pub struct ApiServer {
    /// Configuration.
    config: ApiConfig,

    /// Engine.
    engine: Arc<Engine>,

    /// Model ID.
    model_id: String,
}

impl ApiServer {
    /// Create a new API server.
    pub fn new(engine: Arc<Engine>, model_id: String, config: ApiConfig) -> Self {
        Self {
            config,
            engine,
            model_id,
        }
    }

    /// Create with default config.
    pub fn with_defaults(engine: Arc<Engine>, model_id: String) -> Self {
        Self::new(engine, model_id, ApiConfig::default())
    }

    /// Get the bind address.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.config.host, self.config.port)
    }

    /// Start the server.
    pub async fn serve(self) -> Result<(), ApiError> {
        let addr = self.bind_addr();

        info!("Starting Rusteeze API server on {}", addr);

        // Create app state
        let state = Arc::new(AppState {
            engine: self.engine,
            model_id: self.model_id,
            start_time: std::time::Instant::now(),
            request_counter: std::sync::atomic::AtomicU64::new(0),
        });

        // Build router
        let router = RouterBuilder::new(state)
            .with_cors(self.config.cors_enabled)
            .with_compression(self.config.compression_enabled)
            .with_tracing(self.config.request_logging)
            .build();

        // Bind listener
        let listener = TcpListener::bind(&addr).await.map_err(|e| {
            ApiError::Internal(format!("Failed to bind to {}: {}", addr, e))
        })?;

        info!("Rusteeze API server listening on {}", addr);
        info!("  OpenAI-compatible endpoint: http://{}/v1/chat/completions", addr);
        info!("  Health check: http://{}/health", addr);
        info!("  Metrics: http://{}/metrics", addr);

        // Run server with graceful shutdown
        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| ApiError::Internal(format!("Server error: {}", e)))?;

        info!("Rusteeze API server shut down");

        Ok(())
    }

    /// Start server with TLS.
    pub async fn serve_tls(self, tls_config: TlsConfig) -> Result<(), ApiError> {
        let addr = self.bind_addr();

        info!("Starting Rusteeze API server with TLS on {}", addr);

        // Create app state
        let state = Arc::new(AppState {
            engine: self.engine,
            model_id: self.model_id,
            start_time: std::time::Instant::now(),
            request_counter: std::sync::atomic::AtomicU64::new(0),
        });

        // Build router
        let router = RouterBuilder::new(state)
            .with_cors(self.config.cors_enabled)
            .with_compression(self.config.compression_enabled)
            .with_tracing(self.config.request_logging)
            .build();

        // Load TLS config
        let cert = std::fs::read(&tls_config.cert_path).map_err(|e| {
            ApiError::Internal(format!("Failed to read certificate: {}", e))
        })?;
        let key = std::fs::read(&tls_config.key_path).map_err(|e| {
            ApiError::Internal(format!("Failed to read key: {}", e))
        })?;

        // Note: For production TLS, you'd use rustls or openssl
        // This is a placeholder for the TLS configuration
        warn!("TLS support requires additional configuration");

        // For now, fall back to non-TLS
        let listener = TcpListener::bind(&addr).await.map_err(|e| {
            ApiError::Internal(format!("Failed to bind to {}: {}", addr, e))
        })?;

        axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| ApiError::Internal(format!("Server error: {}", e)))?;

        Ok(())
    }
}

/// Server builder for fluent configuration.
pub struct ServerBuilder {
    engine: Option<Arc<Engine>>,
    model_id: Option<String>,
    config: ApiConfig,
}

impl ServerBuilder {
    /// Create a new server builder.
    pub fn new() -> Self {
        Self {
            engine: None,
            model_id: None,
            config: ApiConfig::default(),
        }
    }

    /// Set the engine.
    pub fn engine(mut self, engine: Arc<Engine>) -> Self {
        self.engine = Some(engine);
        self
    }

    /// Set the model ID.
    pub fn model_id(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Set the host.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.config.host = host.into();
        self
    }

    /// Set the port.
    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    /// Set the API key.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    /// Enable/disable CORS.
    pub fn cors(mut self, enabled: bool) -> Self {
        self.config.cors_enabled = enabled;
        self
    }

    /// Enable/disable compression.
    pub fn compression(mut self, enabled: bool) -> Self {
        self.config.compression_enabled = enabled;
        self
    }

    /// Set request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.request_timeout = timeout;
        self
    }

    /// Set rate limit.
    pub fn rate_limit(mut self, limit: u32) -> Self {
        self.config.rate_limit = Some(limit);
        self
    }

    /// Set TLS configuration.
    pub fn tls(mut self, cert_path: impl Into<String>, key_path: impl Into<String>) -> Self {
        self.config.tls = Some(TlsConfig {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
        });
        self
    }

    /// Build the server.
    pub fn build(self) -> Result<ApiServer, ApiError> {
        let engine = self
            .engine
            .ok_or_else(|| ApiError::Internal("Engine not configured".to_string()))?;

        let model_id = self
            .model_id
            .ok_or_else(|| ApiError::Internal("Model ID not configured".to_string()))?;

        Ok(ApiServer::new(engine, model_id, self.config))
    }
}

impl Default for ServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Shutdown signal handler.
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, initiating graceful shutdown");
        }
        _ = terminate => {
            info!("Received terminate signal, initiating graceful shutdown");
        }
    }
}

/// Quick start function for simple deployments.
pub async fn serve(
    engine: Arc<Engine>,
    model_id: impl Into<String>,
    host: impl Into<String>,
    port: u16,
) -> Result<(), ApiError> {
    let server = ServerBuilder::new()
        .engine(engine)
        .model_id(model_id)
        .host(host)
        .port(port)
        .build()?;

    server.serve().await
}

/// Start server with environment configuration.
pub async fn serve_from_env(engine: Arc<Engine>, model_id: impl Into<String>) -> Result<(), ApiError> {
    let host = std::env::var("RUSTEEZE_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("RUSTEEZE_PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse()
        .unwrap_or(8000);
    let api_key = std::env::var("RUSTEEZE_API_KEY").ok();

    let mut builder = ServerBuilder::new()
        .engine(engine)
        .model_id(model_id)
        .host(host)
        .port(port);

    if let Some(key) = api_key {
        builder = builder.api_key(key);
    }

    let server = builder.build()?;
    server.serve().await
}
