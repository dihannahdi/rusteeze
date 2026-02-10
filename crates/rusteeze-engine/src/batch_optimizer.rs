//! # Batch Optimizer — Radical Rewrite
//!
//! Dynamic batch size optimization with Arc<Request> (no double clone),
//! token budget management, and adaptive batching.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, Duration};

use crate::simd_dispatch;

/// Batch optimizer configuration.
#[derive(Debug, Clone)]
pub struct BatchOptimizerConfig {
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum total tokens per batch
    pub max_tokens_per_batch: usize,
    /// Target latency (microseconds) for adaptive batching
    pub target_latency_us: u64,
    /// Enable adaptive batch sizing
    pub adaptive: bool,
    /// Smoothing factor for latency EMA
    pub ema_alpha: f64,
}

impl Default for BatchOptimizerConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 1,
            max_batch_size: 64,
            max_tokens_per_batch: 4096,
            target_latency_us: 50_000, // 50ms
            adaptive: true,
            ema_alpha: 0.1,
        }
    }
}

/// A request for batching (Arc-wrapped to avoid cloning).
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub request_id: u64,
    pub tokens: Vec<u32>,
    pub priority: f32,
    pub arrival: Instant,
}

/// An optimized batch.
#[derive(Debug)]
pub struct OptimizedBatch {
    /// Requests in this batch (Arc — no data clone)
    pub requests: Vec<Arc<BatchRequest>>,
    /// Total tokens
    pub total_tokens: usize,
    /// Batch creation time
    pub created_at: Instant,
}

/// Batch optimizer stats.
#[derive(Debug, Default)]
pub struct BatchOptimizerStats {
    pub batches_created: AtomicU64,
    pub requests_processed: AtomicU64,
    pub total_tokens: AtomicU64,
    pub avg_batch_size: AtomicU64, // Stored as size * 100 for fixed-point
}

impl Clone for BatchOptimizerStats {
    fn clone(&self) -> Self {
        Self {
            batches_created: AtomicU64::new(self.batches_created.load(Ordering::Relaxed)),
            requests_processed: AtomicU64::new(self.requests_processed.load(Ordering::Relaxed)),
            total_tokens: AtomicU64::new(self.total_tokens.load(Ordering::Relaxed)),
            avg_batch_size: AtomicU64::new(self.avg_batch_size.load(Ordering::Relaxed)),
        }
    }
}

/// Batch optimizer.
pub struct BatchOptimizer {
    config: BatchOptimizerConfig,
    /// Request queue (Arc-wrapped)
    queue: VecDeque<Arc<BatchRequest>>,
    /// Current adaptive batch size
    current_batch_size: usize,
    /// EMA of observed latency (microseconds)
    ema_latency: f64,
    /// Stats
    stats: BatchOptimizerStats,
}

impl BatchOptimizer {
    /// Create a new batch optimizer.
    pub fn new(config: BatchOptimizerConfig) -> Self {
        simd_dispatch::init();
        let batch_size = config.max_batch_size;
        Self {
            config: config.clone(),
            queue: VecDeque::with_capacity(config.max_batch_size * 2),
            current_batch_size: batch_size,
            ema_latency: 0.0,
            stats: BatchOptimizerStats::default(),
        }
    }

    /// Add a request (Arc — no data clone).
    pub fn add_request(&mut self, request: Arc<BatchRequest>) {
        self.queue.push_back(request);
    }

    /// Build the next optimized batch.
    pub fn build_batch(&mut self) -> Option<OptimizedBatch> {
        if self.queue.is_empty() { return None; }

        let max_requests = self.current_batch_size.min(self.queue.len());
        let token_budget = self.config.max_tokens_per_batch;
        let mut batch = Vec::with_capacity(max_requests);
        let mut total_tokens = 0;

        while !self.queue.is_empty() && batch.len() < max_requests {
            let req = self.queue.front().unwrap();
            let req_tokens = req.tokens.len();

            if total_tokens + req_tokens > token_budget && !batch.is_empty() {
                break;
            }

            let req = self.queue.pop_front().unwrap();
            total_tokens += req.tokens.len();
            batch.push(req);
        }

        if batch.is_empty() { return None; }

        let n = batch.len();
        self.stats.batches_created.fetch_add(1, Ordering::Relaxed);
        self.stats.requests_processed.fetch_add(n as u64, Ordering::Relaxed);
        self.stats.total_tokens.fetch_add(total_tokens as u64, Ordering::Relaxed);

        Some(OptimizedBatch {
            requests: batch,
            total_tokens,
            created_at: Instant::now(),
        })
    }

    /// Report latency for adaptive batch sizing.
    pub fn report_latency(&mut self, latency_us: u64) {
        if !self.config.adaptive { return; }

        let alpha = self.config.ema_alpha;
        self.ema_latency = alpha * latency_us as f64 + (1.0 - alpha) * self.ema_latency;

        let target = self.config.target_latency_us as f64;

        if self.ema_latency > target * 1.2 {
            // Too slow — reduce batch size
            self.current_batch_size = (self.current_batch_size * 3 / 4)
                .max(self.config.min_batch_size);
        } else if self.ema_latency < target * 0.6 {
            // Fast enough — increase batch size
            self.current_batch_size = (self.current_batch_size * 5 / 4)
                .min(self.config.max_batch_size);
        }
    }

    /// Current adaptive batch size.
    pub fn current_batch_size(&self) -> usize { self.current_batch_size }

    /// Queue depth.
    pub fn queue_depth(&self) -> usize { self.queue.len() }

    /// Stats.
    pub fn stats(&self) -> &BatchOptimizerStats { &self.stats }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_batch() {
        let config = BatchOptimizerConfig { max_batch_size: 4, ..Default::default() };
        let mut optimizer = BatchOptimizer::new(config);

        for i in 0..3 {
            optimizer.add_request(Arc::new(BatchRequest {
                request_id: i,
                tokens: vec![1, 2, 3],
                priority: 1.0,
                arrival: Instant::now(),
            }));
        }

        let batch = optimizer.build_batch().unwrap();
        assert_eq!(batch.requests.len(), 3);
        assert_eq!(batch.total_tokens, 9);
    }

    #[test]
    fn test_adaptive_sizing() {
        let config = BatchOptimizerConfig {
            max_batch_size: 64,
            min_batch_size: 1,
            target_latency_us: 50_000,
            adaptive: true,
            ..Default::default()
        };
        let mut optimizer = BatchOptimizer::new(config);

        // Report high latency → should decrease batch size
        for _ in 0..20 {
            optimizer.report_latency(100_000);
        }
        assert!(optimizer.current_batch_size() < 64);

        // Report low latency → should increase
        for _ in 0..20 {
            optimizer.report_latency(10_000);
        }
        // May still be less than 64 but should have increased
    }
}
