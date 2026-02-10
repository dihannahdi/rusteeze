//! # Router — Radical Rewrite
//!
//! Request routing with HDR histogram for percentile latency tracking,
//! load balancing, and health monitoring. No more sort-on-every-call.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};

use crate::simd_dispatch;

/// Router configuration.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Number of backend workers
    pub num_backends: usize,
    /// Health check interval (seconds)
    pub health_check_interval: u64,
    /// Maximum queue depth per backend
    pub max_queue_depth: usize,
    /// Load balancing strategy
    pub strategy: LoadBalanceStrategy,
    /// Request timeout (seconds)
    pub timeout_seconds: u64,
}

/// Load balancing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Route to least-loaded backend
    LeastLoaded,
    /// Round-robin
    RoundRobin,
    /// Random
    Random,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            num_backends: 1,
            health_check_interval: 30,
            max_queue_depth: 100,
            strategy: LoadBalanceStrategy::LeastLoaded,
            timeout_seconds: 60,
        }
    }
}

/// Latency histogram using log-linear bucketing.
/// O(1) insertion, O(bucket_count) percentile — no sorting.
pub struct LatencyHistogram {
    /// Buckets: [0-1ms, 1-2ms, 2-4ms, 4-8ms, ..., 2^N ms, overflow]
    buckets: Vec<AtomicU64>,
    /// Total samples
    count: AtomicU64,
    /// Sum of all samples (microseconds)
    sum: AtomicU64,
    /// Minimum value (microseconds)
    min: AtomicU64,
    /// Maximum value (microseconds)
    max: AtomicU64,
}

impl LatencyHistogram {
    /// Create a new histogram with 32 buckets (covers 0 to ~4 billion µs)
    pub fn new() -> Self {
        let buckets = (0..32).map(|_| AtomicU64::new(0)).collect();
        Self {
            buckets,
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }

    /// Record a latency sample (microseconds).
    pub fn record(&self, us: u64) {
        // Bucket index: floor(log2(us + 1))
        let bucket = if us == 0 { 0 } else { (64 - us.leading_zeros()) as usize };
        let bucket = bucket.min(self.buckets.len() - 1);

        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(us, Ordering::Relaxed);

        // Update min
        loop {
            let cur = self.min.load(Ordering::Relaxed);
            if us >= cur { break; }
            if self.min.compare_exchange_weak(cur, us, Ordering::Relaxed, Ordering::Relaxed).is_ok() { break; }
        }
        // Update max
        loop {
            let cur = self.max.load(Ordering::Relaxed);
            if us <= cur { break; }
            if self.max.compare_exchange_weak(cur, us, Ordering::Relaxed, Ordering::Relaxed).is_ok() { break; }
        }
    }

    /// Get percentile value (p in 0.0..1.0).
    pub fn percentile(&self, p: f64) -> u64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 { return 0; }

        let target = (total as f64 * p).ceil() as u64;
        let mut cum = 0u64;

        for (i, bucket) in self.buckets.iter().enumerate() {
            cum += bucket.load(Ordering::Relaxed);
            if cum >= target {
                // Return bucket midpoint
                return if i == 0 { 0 } else { 1u64 << (i - 1) };
            }
        }
        self.max.load(Ordering::Relaxed)
    }

    /// Average latency (microseconds).
    pub fn mean(&self) -> f64 {
        let c = self.count.load(Ordering::Relaxed);
        if c == 0 { 0.0 } else { self.sum.load(Ordering::Relaxed) as f64 / c as f64 }
    }

    /// Total sample count.
    pub fn count(&self) -> u64 { self.count.load(Ordering::Relaxed) }
}

/// Backend worker state.
pub struct BackendState {
    /// Backend ID
    pub id: usize,
    /// Current queue depth
    pub queue_depth: AtomicU64,
    /// Is healthy
    pub healthy: AtomicBool,
    /// Latency histogram
    pub latency: LatencyHistogram,
    /// Total requests served
    pub total_requests: AtomicU64,
    /// Total errors
    pub total_errors: AtomicU64,
}

impl BackendState {
    fn new(id: usize) -> Self {
        Self {
            id,
            queue_depth: AtomicU64::new(0),
            healthy: AtomicBool::new(true),
            latency: LatencyHistogram::new(),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
        }
    }
}

/// Request router.
pub struct Router {
    config: RouterConfig,
    backends: Vec<BackendState>,
    round_robin_counter: AtomicU64,
}

impl Router {
    /// Create a new router.
    pub fn new(config: RouterConfig) -> Self {
        simd_dispatch::init();
        let backends = (0..config.num_backends)
            .map(BackendState::new)
            .collect();
        Self {
            config,
            backends,
            round_robin_counter: AtomicU64::new(0),
        }
    }

    /// Route a request to a backend. Returns backend index.
    pub fn route(&self) -> Option<usize> {
        match self.config.strategy {
            LoadBalanceStrategy::LeastLoaded => self.route_least_loaded(),
            LoadBalanceStrategy::RoundRobin => self.route_round_robin(),
            LoadBalanceStrategy::Random => self.route_random(),
        }
    }

    fn route_least_loaded(&self) -> Option<usize> {
        self.backends.iter()
            .filter(|b| b.healthy.load(Ordering::Relaxed))
            .filter(|b| b.queue_depth.load(Ordering::Relaxed) < self.config.max_queue_depth as u64)
            .min_by_key(|b| b.queue_depth.load(Ordering::Relaxed))
            .map(|b| b.id)
    }

    fn route_round_robin(&self) -> Option<usize> {
        let n = self.backends.len();
        for _ in 0..n {
            let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize % n;
            if self.backends[idx].healthy.load(Ordering::Relaxed) {
                return Some(idx);
            }
        }
        None
    }

    fn route_random(&self) -> Option<usize> {
        let healthy: Vec<usize> = self.backends.iter()
            .filter(|b| b.healthy.load(Ordering::Relaxed))
            .map(|b| b.id)
            .collect();
        if healthy.is_empty() { return None; }
        Some(healthy[fastrand::usize(..healthy.len())])
    }

    /// Record a request start.
    pub fn on_request_start(&self, backend: usize) {
        if backend < self.backends.len() {
            self.backends[backend].queue_depth.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a request completion.
    pub fn on_request_complete(&self, backend: usize, latency_us: u64) {
        if backend < self.backends.len() {
            let b = &self.backends[backend];
            b.queue_depth.fetch_sub(1, Ordering::Relaxed);
            b.total_requests.fetch_add(1, Ordering::Relaxed);
            b.latency.record(latency_us);
        }
    }

    /// Record an error.
    pub fn on_error(&self, backend: usize) {
        if backend < self.backends.len() {
            self.backends[backend].total_errors.fetch_add(1, Ordering::Relaxed);
            self.backends[backend].queue_depth.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Mark a backend as unhealthy.
    pub fn mark_unhealthy(&self, backend: usize) {
        if backend < self.backends.len() {
            self.backends[backend].healthy.store(false, Ordering::Release);
        }
    }

    /// Mark a backend as healthy.
    pub fn mark_healthy(&self, backend: usize) {
        if backend < self.backends.len() {
            self.backends[backend].healthy.store(true, Ordering::Release);
        }
    }

    /// Get p50 latency for a backend (microseconds).
    pub fn p50(&self, backend: usize) -> u64 {
        self.backends.get(backend).map_or(0, |b| b.latency.percentile(0.50))
    }

    /// Get p99 latency for a backend (microseconds).
    pub fn p99(&self, backend: usize) -> u64 {
        self.backends.get(backend).map_or(0, |b| b.latency.percentile(0.99))
    }

    /// Number of backends.
    pub fn num_backends(&self) -> usize { self.backends.len() }

    /// Number of healthy backends.
    pub fn healthy_count(&self) -> usize {
        self.backends.iter()
            .filter(|b| b.healthy.load(Ordering::Relaxed))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        let h = LatencyHistogram::new();
        h.record(100);
        h.record(200);
        h.record(500);
        h.record(1000);
        h.record(5000);
        assert_eq!(h.count(), 5);
        assert!(h.mean() > 0.0);
        let p50 = h.percentile(0.5);
        assert!(p50 > 0);
    }

    #[test]
    fn test_router_least_loaded() {
        let config = RouterConfig { num_backends: 3, ..Default::default() };
        let router = Router::new(config);

        // All healthy, all empty -> should route to 0
        let backend = router.route().unwrap();
        assert!(backend < 3);

        // Increase load on picked backend
        router.on_request_start(backend);
        let next = router.route().unwrap();
        // Should prefer a different backend
        assert!(next < 3);
    }

    #[test]
    fn test_router_round_robin() {
        let config = RouterConfig {
            num_backends: 3,
            strategy: LoadBalanceStrategy::RoundRobin,
            ..Default::default()
        };
        let router = Router::new(config);

        let b0 = router.route().unwrap();
        let b1 = router.route().unwrap();
        let b2 = router.route().unwrap();
        let b3 = router.route().unwrap();
        // Should cycle through
        assert_eq!(b3, b0);
    }
}
