//! Prometheus exporter.

use prometheus::{Encoder, TextEncoder};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tracing::{debug, error, info};

use crate::MetricsError;

/// Prometheus exporter configuration.
#[derive(Debug, Clone)]
pub struct PrometheusConfig {
    /// Host to bind to.
    pub host: String,

    /// Port to listen on.
    pub port: u16,

    /// Metrics path.
    pub path: String,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 9090,
            path: "/metrics".to_string(),
        }
    }
}

/// Prometheus metrics exporter.
pub struct PrometheusExporter {
    config: PrometheusConfig,
}

impl PrometheusExporter {
    /// Create a new exporter.
    pub fn new(config: PrometheusConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(PrometheusConfig::default())
    }

    /// Get metrics in Prometheus format.
    pub fn gather(&self) -> Result<String, MetricsError> {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).into_owned())
    }

    /// Start the exporter server.
    pub async fn serve(self) -> Result<(), MetricsError> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| MetricsError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid address: {}", e),
            )))?;

        let listener = TcpListener::bind(addr).await?;

        info!(
            "Prometheus metrics exporter listening on http://{}{}",
            addr, self.config.path
        );

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let path = self.config.path.clone();
                    tokio::spawn(async move {
                        if let Err(e) = handle_connection(stream, &path).await {
                            error!("Error handling metrics request: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Error accepting connection: {}", e);
                }
            }
        }
    }
}

/// Handle a single connection.
async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    metrics_path: &str,
) -> Result<(), std::io::Error> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut buffer = [0; 1024];
    let n = stream.read(&mut buffer).await?;

    let request = String::from_utf8_lossy(&buffer[..n]);
    let request_line = request.lines().next().unwrap_or("");

    debug!("Metrics request: {}", request_line);

    // Parse request
    let response = if request_line.starts_with("GET") {
        let path = request_line
            .split_whitespace()
            .nth(1)
            .unwrap_or("/");

        if path == metrics_path || path == "/" {
            // Serve metrics
            let encoder = TextEncoder::new();
            let metric_families = prometheus::gather();
            let mut buffer = Vec::new();
            encoder.encode(&metric_families, &mut buffer).ok();
            let body = String::from_utf8_lossy(&buffer);

            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )
        } else if path == "/health" {
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nOK".to_string()
        } else {
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_string()
        }
    } else {
        "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n".to_string()
    };

    stream.write_all(response.as_bytes()).await?;
    stream.flush().await?;

    Ok(())
}

/// Start exporter on default port.
pub async fn start_exporter() -> Result<(), MetricsError> {
    PrometheusExporter::with_defaults().serve().await
}

/// Start exporter with custom port.
pub async fn start_exporter_on_port(port: u16) -> Result<(), MetricsError> {
    let config = PrometheusConfig {
        port,
        ..Default::default()
    };
    PrometheusExporter::new(config).serve().await
}

/// Push gateway configuration.
#[derive(Debug, Clone)]
pub struct PushGatewayConfig {
    /// Gateway URL.
    pub url: String,

    /// Job name.
    pub job: String,

    /// Instance name.
    pub instance: Option<String>,

    /// Push interval in seconds.
    pub interval_seconds: u64,
}

/// Push metrics to Prometheus push gateway.
pub async fn push_to_gateway(config: &PushGatewayConfig) -> Result<(), MetricsError> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;

    let url = if let Some(ref instance) = config.instance {
        format!(
            "{}/metrics/job/{}/instance/{}",
            config.url, config.job, instance
        )
    } else {
        format!("{}/metrics/job/{}", config.url, config.job)
    };

    // Note: In production, use reqwest or similar HTTP client
    debug!("Would push metrics to: {}", url);

    Ok(())
}
