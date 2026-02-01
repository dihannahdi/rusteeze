//! Metrics collector.

use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec, Opts, Registry,
};
use std::sync::Arc;
use tracing::debug;

use crate::MetricsError;

/// Metrics configuration.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// Namespace for metrics.
    pub namespace: String,

    /// Subsystem for metrics.
    pub subsystem: String,

    /// Latency histogram buckets.
    pub latency_buckets: Vec<f64>,

    /// Token histogram buckets.
    pub token_buckets: Vec<f64>,

    /// Enable detailed metrics.
    pub detailed: bool,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            namespace: "rusteeze".to_string(),
            subsystem: "inference".to_string(),
            latency_buckets: vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            token_buckets: vec![1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0],
            detailed: true,
        }
    }
}

/// Metrics collector.
pub struct MetricsCollector {
    registry: Registry,

    // Request metrics
    requests_total: CounterVec,
    requests_active: Gauge,
    request_latency: HistogramVec,

    // Token metrics
    prompt_tokens_total: Counter,
    generation_tokens_total: Counter,
    tokens_per_second: Gauge,
    time_to_first_token: Histogram,
    inter_token_latency: Histogram,

    // Batch metrics
    batch_size: Histogram,
    batch_tokens: Histogram,
    batches_total: Counter,

    // Queue metrics
    queue_waiting: Gauge,
    queue_running: Gauge,
    queue_swapped: Gauge,
    queue_wait_time: Histogram,

    // Memory metrics
    gpu_memory_used: GaugeVec,
    gpu_memory_total: GaugeVec,
    cpu_memory_used: Gauge,
    kv_cache_usage: Gauge,
    block_usage: GaugeVec,

    // Error metrics
    errors_total: CounterVec,

    // Sampling metrics
    sampling_temperature: Histogram,
    sampling_top_p: Histogram,

    // Cache metrics
    prefix_cache_hits: Counter,
    prefix_cache_misses: Counter,

    // Operation latency
    operation_latency: HistogramVec,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new(config: MetricsConfig) -> Result<Self, MetricsError> {
        let registry = Registry::new();

        // Request metrics
        let requests_total = CounterVec::new(
            Opts::new("requests_total", "Total number of requests")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
            &["status", "model"],
        )?;

        let requests_active = Gauge::with_opts(
            Opts::new("requests_active", "Number of active requests")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let request_latency = HistogramVec::new(
            HistogramOpts::new("request_latency_seconds", "Request latency in seconds")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem)
                .buckets(config.latency_buckets.clone()),
            &["model", "stream"],
        )?;

        // Token metrics
        let prompt_tokens_total = Counter::with_opts(
            Opts::new("prompt_tokens_total", "Total prompt tokens processed")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let generation_tokens_total = Counter::with_opts(
            Opts::new("generation_tokens_total", "Total tokens generated")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let tokens_per_second = Gauge::with_opts(
            Opts::new("tokens_per_second", "Token generation rate")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let time_to_first_token = Histogram::with_opts(
            HistogramOpts::new("time_to_first_token_seconds", "Time to first token")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem)
                .buckets(vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
        )?;

        let inter_token_latency = Histogram::with_opts(
            HistogramOpts::new("inter_token_latency_seconds", "Inter-token latency")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem)
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]),
        )?;

        // Batch metrics
        let batch_size = Histogram::with_opts(
            HistogramOpts::new("batch_size", "Batch size")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem)
                .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]),
        )?;

        let batch_tokens = Histogram::with_opts(
            HistogramOpts::new("batch_tokens", "Tokens in batch")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem)
                .buckets(config.token_buckets.clone()),
        )?;

        let batches_total = Counter::with_opts(
            Opts::new("batches_total", "Total batches processed")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        // Queue metrics
        let queue_waiting = Gauge::with_opts(
            Opts::new("queue_waiting", "Requests waiting in queue")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let queue_running = Gauge::with_opts(
            Opts::new("queue_running", "Requests currently running")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let queue_swapped = Gauge::with_opts(
            Opts::new("queue_swapped", "Requests swapped to CPU")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let queue_wait_time = Histogram::with_opts(
            HistogramOpts::new("queue_wait_time_seconds", "Time spent waiting in queue")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem)
                .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]),
        )?;

        // Memory metrics
        let gpu_memory_used = GaugeVec::new(
            Opts::new("gpu_memory_used_bytes", "GPU memory used")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
            &["device"],
        )?;

        let gpu_memory_total = GaugeVec::new(
            Opts::new("gpu_memory_total_bytes", "GPU memory total")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
            &["device"],
        )?;

        let cpu_memory_used = Gauge::with_opts(
            Opts::new("cpu_memory_used_bytes", "CPU memory used")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let kv_cache_usage = Gauge::with_opts(
            Opts::new("kv_cache_usage_ratio", "KV cache usage ratio")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let block_usage = GaugeVec::new(
            Opts::new("block_usage", "Block usage")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
            &["type"],
        )?;

        // Error metrics
        let errors_total = CounterVec::new(
            Opts::new("errors_total", "Total errors")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
            &["type"],
        )?;

        // Sampling metrics
        let sampling_temperature = Histogram::with_opts(
            HistogramOpts::new("sampling_temperature", "Sampling temperature")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem)
                .buckets(vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]),
        )?;

        let sampling_top_p = Histogram::with_opts(
            HistogramOpts::new("sampling_top_p", "Sampling top-p")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem)
                .buckets(vec![0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]),
        )?;

        // Cache metrics
        let prefix_cache_hits = Counter::with_opts(
            Opts::new("prefix_cache_hits_total", "Prefix cache hits")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        let prefix_cache_misses = Counter::with_opts(
            Opts::new("prefix_cache_misses_total", "Prefix cache misses")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem),
        )?;

        // Operation latency
        let operation_latency = HistogramVec::new(
            HistogramOpts::new("operation_latency_seconds", "Operation latency")
                .namespace(&config.namespace)
                .subsystem(&config.subsystem)
                .buckets(config.latency_buckets),
            &["operation"],
        )?;

        // Register all metrics
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(requests_active.clone()))?;
        registry.register(Box::new(request_latency.clone()))?;
        registry.register(Box::new(prompt_tokens_total.clone()))?;
        registry.register(Box::new(generation_tokens_total.clone()))?;
        registry.register(Box::new(tokens_per_second.clone()))?;
        registry.register(Box::new(time_to_first_token.clone()))?;
        registry.register(Box::new(inter_token_latency.clone()))?;
        registry.register(Box::new(batch_size.clone()))?;
        registry.register(Box::new(batch_tokens.clone()))?;
        registry.register(Box::new(batches_total.clone()))?;
        registry.register(Box::new(queue_waiting.clone()))?;
        registry.register(Box::new(queue_running.clone()))?;
        registry.register(Box::new(queue_swapped.clone()))?;
        registry.register(Box::new(queue_wait_time.clone()))?;
        registry.register(Box::new(gpu_memory_used.clone()))?;
        registry.register(Box::new(gpu_memory_total.clone()))?;
        registry.register(Box::new(cpu_memory_used.clone()))?;
        registry.register(Box::new(kv_cache_usage.clone()))?;
        registry.register(Box::new(block_usage.clone()))?;
        registry.register(Box::new(errors_total.clone()))?;
        registry.register(Box::new(sampling_temperature.clone()))?;
        registry.register(Box::new(sampling_top_p.clone()))?;
        registry.register(Box::new(prefix_cache_hits.clone()))?;
        registry.register(Box::new(prefix_cache_misses.clone()))?;
        registry.register(Box::new(operation_latency.clone()))?;

        debug!("Initialized metrics collector");

        Ok(Self {
            registry,
            requests_total,
            requests_active,
            request_latency,
            prompt_tokens_total,
            generation_tokens_total,
            tokens_per_second,
            time_to_first_token,
            inter_token_latency,
            batch_size,
            batch_tokens,
            batches_total,
            queue_waiting,
            queue_running,
            queue_swapped,
            queue_wait_time,
            gpu_memory_used,
            gpu_memory_total,
            cpu_memory_used,
            kv_cache_usage,
            block_usage,
            errors_total,
            sampling_temperature,
            sampling_top_p,
            prefix_cache_hits,
            prefix_cache_misses,
            operation_latency,
        })
    }

    // Request metrics
    pub fn record_request_started(&self) {
        self.requests_active.inc();
    }

    pub fn record_request_completed(&self, status: &str, model: &str, latency: f64, stream: bool) {
        self.requests_active.dec();
        self.requests_total.with_label_values(&[status, model]).inc();
        self.request_latency
            .with_label_values(&[model, if stream { "true" } else { "false" }])
            .observe(latency);
    }

    // Token metrics
    pub fn record_prompt_tokens(&self, count: u64) {
        self.prompt_tokens_total.inc_by(count as f64);
    }

    pub fn record_generation_tokens(&self, count: u64) {
        self.generation_tokens_total.inc_by(count as f64);
    }

    pub fn set_tokens_per_second(&self, tps: f64) {
        self.tokens_per_second.set(tps);
    }

    pub fn record_time_to_first_token(&self, seconds: f64) {
        self.time_to_first_token.observe(seconds);
    }

    pub fn record_inter_token_latency(&self, seconds: f64) {
        self.inter_token_latency.observe(seconds);
    }

    // Batch metrics
    pub fn record_batch(&self, size: usize, tokens: usize) {
        self.batch_size.observe(size as f64);
        self.batch_tokens.observe(tokens as f64);
        self.batches_total.inc();
    }

    // Queue metrics
    pub fn set_queue_sizes(&self, waiting: usize, running: usize, swapped: usize) {
        self.queue_waiting.set(waiting as f64);
        self.queue_running.set(running as f64);
        self.queue_swapped.set(swapped as f64);
    }

    pub fn record_queue_wait_time(&self, seconds: f64) {
        self.queue_wait_time.observe(seconds);
    }

    // Memory metrics
    pub fn set_gpu_memory(&self, device: &str, used: u64, total: u64) {
        self.gpu_memory_used.with_label_values(&[device]).set(used as f64);
        self.gpu_memory_total.with_label_values(&[device]).set(total as f64);
    }

    pub fn set_cpu_memory(&self, used: u64) {
        self.cpu_memory_used.set(used as f64);
    }

    pub fn set_kv_cache_usage(&self, ratio: f64) {
        self.kv_cache_usage.set(ratio);
    }

    pub fn set_block_usage(&self, gpu_blocks: usize, cpu_blocks: usize) {
        self.block_usage.with_label_values(&["gpu"]).set(gpu_blocks as f64);
        self.block_usage.with_label_values(&["cpu"]).set(cpu_blocks as f64);
    }

    // Error metrics
    pub fn record_error(&self, error_type: &str) {
        self.errors_total.with_label_values(&[error_type]).inc();
    }

    // Sampling metrics
    pub fn record_sampling_params(&self, temperature: f32, top_p: f32) {
        self.sampling_temperature.observe(temperature as f64);
        self.sampling_top_p.observe(top_p as f64);
    }

    // Cache metrics
    pub fn record_prefix_cache_hit(&self) {
        self.prefix_cache_hits.inc();
    }

    pub fn record_prefix_cache_miss(&self) {
        self.prefix_cache_misses.inc();
    }

    // Operation latency
    pub fn record_operation_latency(&self, operation: &str, seconds: f64) {
        self.operation_latency
            .with_label_values(&[operation])
            .observe(seconds);
    }

    /// Get registry for custom metrics.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}
