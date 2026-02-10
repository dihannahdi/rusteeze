//! Tracing setup and configuration.

use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::TracerProvider;
use tracing::Level;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

use crate::MetricsError;

/// Tracing configuration.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Log level.
    pub level: Level,

    /// Log format.
    pub format: LogFormat,

    /// Enable OpenTelemetry.
    pub otlp_enabled: bool,

    /// OTLP endpoint.
    pub otlp_endpoint: Option<String>,

    /// Service name.
    pub service_name: String,

    /// Enable span events.
    pub span_events: bool,

    /// Log file path.
    pub file_path: Option<String>,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            format: LogFormat::Pretty,
            otlp_enabled: false,
            otlp_endpoint: None,
            service_name: "rusteeze".to_string(),
            span_events: false,
            file_path: None,
        }
    }
}

/// Log format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable format.
    Pretty,

    /// Compact format.
    Compact,

    /// JSON format.
    Json,
}

impl LogFormat {
    /// Parse from string.
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => Self::Json,
            "compact" => Self::Compact,
            _ => Self::Pretty,
        }
    }
}

/// Initialize tracing.
pub fn init_tracing(config: TracingConfig) -> Result<(), MetricsError> {
    // Build filter
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.level.to_string()));

    // Build subscriber based on format
    let span_events = if config.span_events {
        FmtSpan::NEW | FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };

    match config.format {
        LogFormat::Json => {
            let subscriber = tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .json()
                        .with_span_events(span_events)
                        .with_target(true)
                        .with_thread_ids(true)
                        .with_file(true)
                        .with_line_number(true),
                );

            subscriber
                .try_init()
                .map_err(|e| MetricsError::TracingError(e.to_string()))?;
        }
        LogFormat::Compact => {
            let subscriber = tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .compact()
                        .with_span_events(span_events)
                        .with_target(true),
                );

            subscriber
                .try_init()
                .map_err(|e| MetricsError::TracingError(e.to_string()))?;
        }
        LogFormat::Pretty => {
            let subscriber = tracing_subscriber::registry()
                .with(filter)
                .with(
                    fmt::layer()
                        .pretty()
                        .with_span_events(span_events)
                        .with_target(true)
                        .with_thread_names(true),
                );

            subscriber
                .try_init()
                .map_err(|e| MetricsError::TracingError(e.to_string()))?;
        }
    }

    Ok(())
}

/// Initialize tracing with OpenTelemetry.
pub fn init_tracing_with_otlp(config: TracingConfig) -> Result<(), MetricsError> {
    use opentelemetry_otlp::WithExportConfig;

    let endpoint = config
        .otlp_endpoint
        .clone()
        .unwrap_or_else(|| "http://localhost:4317".to_string());

    // Create OTLP exporter using new_exporter interface
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(&endpoint);

    // Create tracer provider
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .map_err(|e| MetricsError::TracingError(e.to_string()))?;

    // Create OpenTelemetry layer
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

    // Build filter
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(config.level.to_string()));

    // Build subscriber
    let subscriber = tracing_subscriber::registry()
        .with(filter)
        .with(telemetry)
        .with(fmt::layer().with_target(true));

    subscriber
        .try_init()
        .map_err(|e| MetricsError::TracingError(e.to_string()))?;

    Ok(())
}

/// Initialize simple console tracing.
pub fn init_console_tracing() -> Result<(), MetricsError> {
    init_tracing(TracingConfig::default())
}

/// Initialize tracing from environment.
pub fn init_tracing_from_env() -> Result<(), MetricsError> {
    let level = std::env::var("LOG_LEVEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(Level::INFO);

    let format = std::env::var("LOG_FORMAT")
        .map(|s| LogFormat::from_str(&s))
        .unwrap_or(LogFormat::Pretty);

    let otlp_enabled = std::env::var("OTLP_ENABLED")
        .map(|s| s == "true" || s == "1")
        .unwrap_or(false);

    let otlp_endpoint = std::env::var("OTLP_ENDPOINT").ok();

    let service_name = std::env::var("SERVICE_NAME")
        .unwrap_or_else(|_| "rusteeze".to_string());

    let config = TracingConfig {
        level,
        format,
        otlp_enabled,
        otlp_endpoint,
        service_name,
        span_events: false,
        file_path: None,
    };

    if config.otlp_enabled {
        init_tracing_with_otlp(config)
    } else {
        init_tracing(config)
    }
}

/// Tracing guard that flushes on drop.
pub struct TracingGuard {
    _private: (),
}

impl TracingGuard {
    /// Create a new guard.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        // Flush OpenTelemetry
        opentelemetry::global::shutdown_tracer_provider();
    }
}

impl Default for TracingGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a request span.
#[macro_export]
macro_rules! request_span {
    ($request_id:expr) => {
        tracing::info_span!(
            "request",
            request_id = %$request_id,
            otel.kind = "server"
        )
    };
    ($request_id:expr, $model:expr) => {
        tracing::info_span!(
            "request",
            request_id = %$request_id,
            model = %$model,
            otel.kind = "server"
        )
    };
}

/// Create an inference span.
#[macro_export]
macro_rules! inference_span {
    ($batch_size:expr, $tokens:expr) => {
        tracing::info_span!(
            "inference",
            batch_size = $batch_size,
            tokens = $tokens,
            otel.kind = "internal"
        )
    };
}
