//! Rusteeze Metrics and Observability.
//!
//! This crate provides comprehensive metrics collection, tracing, and
//! observability features for the Rusteeze LLM inference engine.
//!
//! # Features
//!
//! - Prometheus metrics export
//! - OpenTelemetry tracing
//! - Request latency histograms
//! - Token throughput counters
//! - GPU utilization gauges
//! - Custom metric registration
//!
//! # Example
//!
//! ```rust,ignore
//! use rusteeze_metrics::{MetricsCollector, init_metrics};
//!
//! // Initialize metrics
//! init_metrics()?;
//!
//! // Record metrics
//! MetricsCollector::record_request_latency(0.5);
//! MetricsCollector::record_tokens_generated(100);
//! ```

pub mod collector;
pub mod counters;
pub mod gauges;
pub mod histograms;
pub mod prometheus;
pub mod tracing_setup;

pub use collector::{MetricsCollector, MetricsConfig};
pub use counters::*;
pub use gauges::*;
pub use histograms::*;
pub use prometheus::PrometheusExporter;
pub use tracing_setup::{init_tracing, TracingConfig};

use ::prometheus::{Encoder, TextEncoder};
use std::sync::OnceLock;
use thiserror::Error;

/// Global metrics collector.
static METRICS: OnceLock<MetricsCollector> = OnceLock::new();

/// Metrics error.
#[derive(Error, Debug)]
pub enum MetricsError {
    #[error("Metrics already initialized")]
    AlreadyInitialized,

    #[error("Metrics not initialized")]
    NotInitialized,

    #[error("Prometheus error: {0}")]
    PrometheusError(#[from] ::prometheus::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Tracing error: {0}")]
    TracingError(String),
}

/// Initialize global metrics collector.
pub fn init_metrics(config: MetricsConfig) -> Result<(), MetricsError> {
    let collector = MetricsCollector::new(config)?;
    METRICS
        .set(collector)
        .map_err(|_| MetricsError::AlreadyInitialized)?;
    Ok(())
}

/// Get global metrics collector.
pub fn metrics() -> Result<&'static MetricsCollector, MetricsError> {
    METRICS.get().ok_or(MetricsError::NotInitialized)
}

/// Export metrics in Prometheus format.
pub fn export_prometheus() -> Result<String, MetricsError> {
    let encoder = TextEncoder::new();
    let metric_families = ::prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

/// Common metric labels.
#[derive(Debug, Clone)]
pub struct MetricLabels {
    pub model: String,
    pub instance: String,
    pub environment: String,
}

impl Default for MetricLabels {
    fn default() -> Self {
        Self {
            model: "unknown".to_string(),
            instance: hostname(),
            environment: std::env::var("ENVIRONMENT").unwrap_or_else(|_| "production".to_string()),
        }
    }
}

/// Get hostname.
fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "localhost".to_string())
}

/// Macro for timing code blocks.
#[macro_export]
macro_rules! time_block {
    ($name:expr, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let duration = start.elapsed();
        tracing::debug!("{} took {:?}", $name, duration);
        if let Ok(m) = $crate::metrics() {
            m.record_operation_latency($name, duration.as_secs_f64());
        }
        result
    }};
}
