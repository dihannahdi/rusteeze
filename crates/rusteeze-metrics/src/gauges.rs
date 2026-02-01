//! Gauge metrics.

use prometheus::{Gauge, GaugeVec, Opts};
use std::sync::OnceLock;

use crate::MetricsError;

/// Active requests gauge.
static ACTIVE_REQUESTS: OnceLock<Gauge> = OnceLock::new();

/// Queue size gauge.
static QUEUE_SIZE: OnceLock<GaugeVec> = OnceLock::new();

/// Memory usage gauge.
static MEMORY_USAGE: OnceLock<GaugeVec> = OnceLock::new();

/// Initialize gauges.
pub fn init_gauges() -> Result<(), MetricsError> {
    let active_requests = Gauge::with_opts(
        Opts::new("rusteeze_active_requests", "Number of active requests"),
    )?;
    ACTIVE_REQUESTS
        .set(active_requests)
        .map_err(|_| MetricsError::AlreadyInitialized)?;

    let queue_size = GaugeVec::new(
        Opts::new("rusteeze_queue_size", "Queue size"),
        &["queue"],
    )?;
    QUEUE_SIZE
        .set(queue_size)
        .map_err(|_| MetricsError::AlreadyInitialized)?;

    let memory_usage = GaugeVec::new(
        Opts::new("rusteeze_memory_bytes", "Memory usage in bytes"),
        &["type", "device"],
    )?;
    MEMORY_USAGE
        .set(memory_usage)
        .map_err(|_| MetricsError::AlreadyInitialized)?;

    Ok(())
}

/// Set active requests.
pub fn set_active_requests(count: i64) {
    if let Some(gauge) = ACTIVE_REQUESTS.get() {
        gauge.set(count as f64);
    }
}

/// Increment active requests.
pub fn inc_active_requests() {
    if let Some(gauge) = ACTIVE_REQUESTS.get() {
        gauge.inc();
    }
}

/// Decrement active requests.
pub fn dec_active_requests() {
    if let Some(gauge) = ACTIVE_REQUESTS.get() {
        gauge.dec();
    }
}

/// Set queue size.
pub fn set_queue_size(queue: &str, size: usize) {
    if let Some(gauge) = QUEUE_SIZE.get() {
        gauge.with_label_values(&[queue]).set(size as f64);
    }
}

/// Set memory usage.
pub fn set_memory_usage(memory_type: &str, device: &str, bytes: u64) {
    if let Some(gauge) = MEMORY_USAGE.get() {
        gauge
            .with_label_values(&[memory_type, device])
            .set(bytes as f64);
    }
}

/// Get active requests.
pub fn get_active_requests() -> f64 {
    ACTIVE_REQUESTS.get().map(|g| g.get()).unwrap_or(0.0)
}

/// Get queue size.
pub fn get_queue_size(queue: &str) -> f64 {
    QUEUE_SIZE
        .get()
        .map(|g| g.with_label_values(&[queue]).get())
        .unwrap_or(0.0)
}

/// Simple gauge wrapper.
pub struct SimpleGauge {
    gauge: Gauge,
}

impl SimpleGauge {
    /// Create a new gauge.
    pub fn new(name: &str, help: &str) -> Result<Self, MetricsError> {
        let gauge = Gauge::with_opts(Opts::new(name, help))?;
        Ok(Self { gauge })
    }

    /// Set value.
    pub fn set(&self, v: f64) {
        self.gauge.set(v);
    }

    /// Increment by 1.
    pub fn inc(&self) {
        self.gauge.inc();
    }

    /// Decrement by 1.
    pub fn dec(&self) {
        self.gauge.dec();
    }

    /// Add value.
    pub fn add(&self, v: f64) {
        self.gauge.add(v);
    }

    /// Subtract value.
    pub fn sub(&self, v: f64) {
        self.gauge.sub(v);
    }

    /// Get current value.
    pub fn get(&self) -> f64 {
        self.gauge.get()
    }
}

/// Labeled gauge wrapper.
pub struct LabeledGauge {
    gauge: GaugeVec,
}

impl LabeledGauge {
    /// Create a new labeled gauge.
    pub fn new(name: &str, help: &str, labels: &[&str]) -> Result<Self, MetricsError> {
        let gauge = GaugeVec::new(Opts::new(name, help), labels)?;
        Ok(Self { gauge })
    }

    /// Set value with labels.
    pub fn set(&self, labels: &[&str], v: f64) {
        self.gauge.with_label_values(labels).set(v);
    }

    /// Increment with labels.
    pub fn inc(&self, labels: &[&str]) {
        self.gauge.with_label_values(labels).inc();
    }

    /// Decrement with labels.
    pub fn dec(&self, labels: &[&str]) {
        self.gauge.with_label_values(labels).dec();
    }

    /// Get value with labels.
    pub fn get(&self, labels: &[&str]) -> f64 {
        self.gauge.with_label_values(labels).get()
    }
}
