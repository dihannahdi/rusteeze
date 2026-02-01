//! Counter metrics.

use prometheus::{Counter, CounterVec, Opts};
use std::sync::OnceLock;

use crate::MetricsError;

/// Request counter.
static REQUEST_COUNTER: OnceLock<CounterVec> = OnceLock::new();

/// Token counter.
static TOKEN_COUNTER: OnceLock<CounterVec> = OnceLock::new();

/// Initialize counters.
pub fn init_counters() -> Result<(), MetricsError> {
    let request_counter = CounterVec::new(
        Opts::new("rusteeze_requests_total", "Total requests"),
        &["status", "model", "endpoint"],
    )?;
    REQUEST_COUNTER
        .set(request_counter)
        .map_err(|_| MetricsError::AlreadyInitialized)?;

    let token_counter = CounterVec::new(
        Opts::new("rusteeze_tokens_total", "Total tokens"),
        &["type", "model"],
    )?;
    TOKEN_COUNTER
        .set(token_counter)
        .map_err(|_| MetricsError::AlreadyInitialized)?;

    Ok(())
}

/// Increment request counter.
pub fn inc_requests(status: &str, model: &str, endpoint: &str) {
    if let Some(counter) = REQUEST_COUNTER.get() {
        counter.with_label_values(&[status, model, endpoint]).inc();
    }
}

/// Add to request counter.
pub fn add_requests(status: &str, model: &str, endpoint: &str, count: u64) {
    if let Some(counter) = REQUEST_COUNTER.get() {
        counter
            .with_label_values(&[status, model, endpoint])
            .inc_by(count as f64);
    }
}

/// Increment token counter.
pub fn inc_tokens(token_type: &str, model: &str, count: u64) {
    if let Some(counter) = TOKEN_COUNTER.get() {
        counter
            .with_label_values(&[token_type, model])
            .inc_by(count as f64);
    }
}

/// Get current request count.
pub fn get_request_count(status: &str, model: &str, endpoint: &str) -> f64 {
    REQUEST_COUNTER
        .get()
        .map(|c| c.with_label_values(&[status, model, endpoint]).get())
        .unwrap_or(0.0)
}

/// Get current token count.
pub fn get_token_count(token_type: &str, model: &str) -> f64 {
    TOKEN_COUNTER
        .get()
        .map(|c| c.with_label_values(&[token_type, model]).get())
        .unwrap_or(0.0)
}

/// Simple counter wrapper.
pub struct SimpleCounter {
    counter: Counter,
}

impl SimpleCounter {
    /// Create a new counter.
    pub fn new(name: &str, help: &str) -> Result<Self, MetricsError> {
        let counter = Counter::with_opts(Opts::new(name, help))?;
        Ok(Self { counter })
    }

    /// Increment by 1.
    pub fn inc(&self) {
        self.counter.inc();
    }

    /// Increment by value.
    pub fn inc_by(&self, v: f64) {
        self.counter.inc_by(v);
    }

    /// Get current value.
    pub fn get(&self) -> f64 {
        self.counter.get()
    }
}
