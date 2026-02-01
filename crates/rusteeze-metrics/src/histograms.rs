//! Histogram metrics.

use prometheus::{Histogram, HistogramOpts, HistogramVec};
use std::sync::OnceLock;

use crate::MetricsError;

/// Request latency histogram.
static REQUEST_LATENCY: OnceLock<HistogramVec> = OnceLock::new();

/// Token generation latency.
static TOKEN_LATENCY: OnceLock<Histogram> = OnceLock::new();

/// Default latency buckets (in seconds).
pub const DEFAULT_LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Fine-grained latency buckets for token generation.
pub const TOKEN_LATENCY_BUCKETS: &[f64] = &[
    0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1,
];

/// Initialize histograms.
pub fn init_histograms() -> Result<(), MetricsError> {
    let request_latency = HistogramVec::new(
        HistogramOpts::new("rusteeze_request_latency_seconds", "Request latency")
            .buckets(DEFAULT_LATENCY_BUCKETS.to_vec()),
        &["model", "endpoint"],
    )?;
    REQUEST_LATENCY
        .set(request_latency)
        .map_err(|_| MetricsError::AlreadyInitialized)?;

    let token_latency = Histogram::with_opts(
        HistogramOpts::new("rusteeze_token_latency_seconds", "Per-token latency")
            .buckets(TOKEN_LATENCY_BUCKETS.to_vec()),
    )?;
    TOKEN_LATENCY
        .set(token_latency)
        .map_err(|_| MetricsError::AlreadyInitialized)?;

    Ok(())
}

/// Record request latency.
pub fn observe_request_latency(model: &str, endpoint: &str, latency: f64) {
    if let Some(hist) = REQUEST_LATENCY.get() {
        hist.with_label_values(&[model, endpoint]).observe(latency);
    }
}

/// Record token latency.
pub fn observe_token_latency(latency: f64) {
    if let Some(hist) = TOKEN_LATENCY.get() {
        hist.observe(latency);
    }
}

/// Simple histogram wrapper.
pub struct SimpleHistogram {
    histogram: Histogram,
}

impl SimpleHistogram {
    /// Create a new histogram.
    pub fn new(name: &str, help: &str, buckets: Vec<f64>) -> Result<Self, MetricsError> {
        let histogram = Histogram::with_opts(
            HistogramOpts::new(name, help).buckets(buckets),
        )?;
        Ok(Self { histogram })
    }

    /// Create with default buckets.
    pub fn with_default_buckets(name: &str, help: &str) -> Result<Self, MetricsError> {
        Self::new(name, help, DEFAULT_LATENCY_BUCKETS.to_vec())
    }

    /// Observe a value.
    pub fn observe(&self, v: f64) {
        self.histogram.observe(v);
    }

    /// Get count of observations.
    pub fn get_sample_count(&self) -> u64 {
        self.histogram.get_sample_count()
    }

    /// Get sum of observations.
    pub fn get_sample_sum(&self) -> f64 {
        self.histogram.get_sample_sum()
    }
}

/// Labeled histogram wrapper.
pub struct LabeledHistogram {
    histogram: HistogramVec,
}

impl LabeledHistogram {
    /// Create a new labeled histogram.
    pub fn new(
        name: &str,
        help: &str,
        labels: &[&str],
        buckets: Vec<f64>,
    ) -> Result<Self, MetricsError> {
        let histogram = HistogramVec::new(
            HistogramOpts::new(name, help).buckets(buckets),
            labels,
        )?;
        Ok(Self { histogram })
    }

    /// Create with default buckets.
    pub fn with_default_buckets(
        name: &str,
        help: &str,
        labels: &[&str],
    ) -> Result<Self, MetricsError> {
        Self::new(name, help, labels, DEFAULT_LATENCY_BUCKETS.to_vec())
    }

    /// Observe a value with labels.
    pub fn observe(&self, labels: &[&str], v: f64) {
        self.histogram.with_label_values(labels).observe(v);
    }

    /// Get count with labels.
    pub fn get_sample_count(&self, labels: &[&str]) -> u64 {
        self.histogram.with_label_values(labels).get_sample_count()
    }

    /// Get sum with labels.
    pub fn get_sample_sum(&self, labels: &[&str]) -> f64 {
        self.histogram.with_label_values(labels).get_sample_sum()
    }
}

/// Timer for measuring durations.
pub struct Timer {
    start: std::time::Instant,
}

impl Timer {
    /// Start a new timer.
    pub fn start() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }

    /// Get elapsed time in seconds.
    pub fn elapsed_seconds(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Stop and observe on a histogram.
    pub fn observe_on(self, histogram: &SimpleHistogram) {
        histogram.observe(self.elapsed_seconds());
    }
}

/// RAII timer that records on drop.
pub struct ScopedTimer<'a> {
    start: std::time::Instant,
    histogram: &'a SimpleHistogram,
}

impl<'a> ScopedTimer<'a> {
    /// Create a new scoped timer.
    pub fn new(histogram: &'a SimpleHistogram) -> Self {
        Self {
            start: std::time::Instant::now(),
            histogram,
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        self.histogram.observe(self.start.elapsed().as_secs_f64());
    }
}
