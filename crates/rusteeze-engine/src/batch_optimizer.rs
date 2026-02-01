//! Batch optimization strategies for maximum throughput.
//!
//! This module provides intelligent batching algorithms that maximize
//! GPU utilization while minimizing latency.
//!
//! # Strategies
//!
//! - **Dynamic Batching**: Adjusts batch size based on sequence lengths
//! - **Bucketed Batching**: Groups sequences by length for efficient padding
//! - **Prefill-Decode Splitting**: Separates prefill and decode phases
//! - **Adaptive Scheduling**: Balances latency and throughput dynamically
//!
//! # Key Optimizations
//!
//! - Minimizes padding waste through smart grouping
//! - Maximizes GPU memory utilization
//! - Reduces inter-batch latency
//! - Supports heterogeneous sequence lengths

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use ahash::AHashMap;
use smallvec::SmallVec;

/// Batch optimization configuration.
#[derive(Debug, Clone)]
pub struct BatchOptConfig {
    /// Maximum batch size.
    pub max_batch_size: usize,

    /// Maximum total tokens in batch.
    pub max_batch_tokens: usize,

    /// Enable bucketed batching.
    pub enable_bucketing: bool,

    /// Bucket boundaries (sequence lengths).
    pub bucket_boundaries: Vec<usize>,

    /// Target padding efficiency (0.0-1.0).
    pub target_efficiency: f32,

    /// Maximum wait time for batching (microseconds).
    pub max_wait_us: u64,

    /// Enable prefill-decode separation.
    pub separate_prefill_decode: bool,

    /// Prefill priority boost factor.
    pub prefill_priority_boost: f32,
}

impl Default for BatchOptConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_batch_tokens: 8192,
            enable_bucketing: true,
            bucket_boundaries: vec![128, 256, 512, 1024, 2048, 4096],
            target_efficiency: 0.85,
            max_wait_us: 1000,
            separate_prefill_decode: true,
            prefill_priority_boost: 1.5,
        }
    }
}

/// Request metadata for batching decisions.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Unique request ID.
    pub id: u64,
    /// Sequence length (prompt + generated).
    pub seq_len: usize,
    /// Prompt length.
    pub prompt_len: usize,
    /// Is this in prefill phase.
    pub is_prefill: bool,
    /// Priority (lower = higher priority).
    pub priority: u32,
    /// Arrival timestamp (microseconds).
    pub arrival_time_us: u64,
    /// Estimated compute cost.
    pub compute_cost: f32,
}

impl BatchRequest {
    /// Calculate effective priority considering wait time.
    pub fn effective_priority(&self, current_time_us: u64, aging_factor: f32) -> f32 {
        let wait_time = (current_time_us - self.arrival_time_us) as f32;
        let age_bonus = wait_time * aging_factor;
        self.priority as f32 - age_bonus
    }
}

/// Result of batch optimization.
#[derive(Debug)]
pub struct OptimizedBatch {
    /// Request IDs in this batch.
    pub request_ids: SmallVec<[u64; 64]>,
    /// Total tokens in batch.
    pub total_tokens: usize,
    /// Maximum sequence length in batch.
    pub max_seq_len: usize,
    /// Padding efficiency (0.0-1.0).
    pub efficiency: f32,
    /// Is this a prefill batch.
    pub is_prefill: bool,
    /// Estimated compute cost.
    pub estimated_cost: f32,
}

impl OptimizedBatch {
    /// Create an empty batch.
    pub fn empty(is_prefill: bool) -> Self {
        Self {
            request_ids: SmallVec::new(),
            total_tokens: 0,
            max_seq_len: 0,
            efficiency: 1.0,
            is_prefill,
            estimated_cost: 0.0,
        }
    }

    /// Calculate padding efficiency.
    fn calculate_efficiency(&self) -> f32 {
        if self.request_ids.is_empty() || self.max_seq_len == 0 {
            return 1.0;
        }
        let padded_total = self.request_ids.len() * self.max_seq_len;
        self.total_tokens as f32 / padded_total as f32
    }
}

/// Wrapper for priority queue ordering.
#[derive(Debug)]
struct PrioritizedRequest {
    request: BatchRequest,
    effective_priority: f32,
}

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.effective_priority == other.effective_priority
    }
}

impl Eq for PrioritizedRequest {}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower priority value = higher priority (reverse ordering)
        other.effective_priority
            .partial_cmp(&self.effective_priority)
            .unwrap_or(Ordering::Equal)
    }
}

/// Length bucket for efficient batching.
#[derive(Debug)]
struct LengthBucket {
    /// Bucket boundary (max length).
    boundary: usize,
    /// Requests in this bucket.
    requests: Vec<BatchRequest>,
}

impl LengthBucket {
    fn new(boundary: usize) -> Self {
        Self {
            boundary,
            requests: Vec::new(),
        }
    }

    fn add(&mut self, request: BatchRequest) {
        self.requests.push(request);
    }

    fn total_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.seq_len).sum()
    }
}

/// Batch optimizer for maximum throughput.
pub struct BatchOptimizer {
    /// Configuration.
    config: BatchOptConfig,

    /// Prefill queue (priority queue).
    prefill_queue: BinaryHeap<PrioritizedRequest>,

    /// Decode queue (priority queue).
    decode_queue: BinaryHeap<PrioritizedRequest>,

    /// Length buckets for prefill.
    prefill_buckets: Vec<LengthBucket>,

    /// Length buckets for decode.
    decode_buckets: Vec<LengthBucket>,

    /// Current timestamp for aging calculations.
    current_time_us: u64,

    /// Statistics.
    stats: BatchStats,
}

/// Batching statistics.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total batches created.
    pub total_batches: u64,
    /// Total requests processed.
    pub total_requests: u64,
    /// Average batch size.
    pub avg_batch_size: f32,
    /// Average padding efficiency.
    pub avg_efficiency: f32,
    /// Prefill batches.
    pub prefill_batches: u64,
    /// Decode batches.
    pub decode_batches: u64,
}

impl BatchOptimizer {
    /// Create a new batch optimizer.
    pub fn new(config: BatchOptConfig) -> Self {
        let mut prefill_buckets = Vec::with_capacity(config.bucket_boundaries.len() + 1);
        let mut decode_buckets = Vec::with_capacity(config.bucket_boundaries.len() + 1);

        for &boundary in &config.bucket_boundaries {
            prefill_buckets.push(LengthBucket::new(boundary));
            decode_buckets.push(LengthBucket::new(boundary));
        }
        // Add overflow bucket
        prefill_buckets.push(LengthBucket::new(usize::MAX));
        decode_buckets.push(LengthBucket::new(usize::MAX));

        Self {
            config,
            prefill_queue: BinaryHeap::new(),
            decode_queue: BinaryHeap::new(),
            prefill_buckets,
            decode_buckets,
            current_time_us: 0,
            stats: BatchStats::default(),
        }
    }

    /// Add a request to the optimizer.
    pub fn add_request(&mut self, request: BatchRequest) {
        let effective_priority = if request.is_prefill {
            // Boost prefill priority
            request.priority as f32 / self.config.prefill_priority_boost
        } else {
            request.priority as f32
        };

        let prioritized = PrioritizedRequest {
            request: request.clone(),
            effective_priority,
        };

        if request.is_prefill {
            self.prefill_queue.push(prioritized);
            
            // Also add to bucket if bucketing enabled
            if self.config.enable_bucketing {
                let bucket_idx = self.find_bucket_idx(request.seq_len);
                self.prefill_buckets[bucket_idx].add(request);
            }
        } else {
            self.decode_queue.push(prioritized);
            
            if self.config.enable_bucketing {
                let bucket_idx = self.find_bucket_idx(request.seq_len);
                self.decode_buckets[bucket_idx].add(request);
            }
        }
    }

    /// Find the appropriate bucket index for a sequence length.
    fn find_bucket_idx(&self, seq_len: usize) -> usize {
        self.config
            .bucket_boundaries
            .iter()
            .position(|&b| seq_len <= b)
            .unwrap_or(self.config.bucket_boundaries.len())
    }

    /// Get the next optimized batch.
    pub fn get_next_batch(&mut self) -> Option<OptimizedBatch> {
        // Prioritize prefill if separate mode enabled
        if self.config.separate_prefill_decode {
            if !self.prefill_queue.is_empty() {
                return Some(self.build_prefill_batch());
            }
            if !self.decode_queue.is_empty() {
                return Some(self.build_decode_batch());
            }
        } else {
            // Mixed mode
            return self.build_mixed_batch();
        }

        None
    }

    /// Build an optimized prefill batch.
    fn build_prefill_batch(&mut self) -> OptimizedBatch {
        let mut batch = OptimizedBatch::empty(true);
        let mut remaining_tokens = self.config.max_batch_tokens;

        // Use bucketed batching if enabled
        if self.config.enable_bucketing {
            // Find the bucket with best efficiency potential
            let best_bucket = self.find_best_bucket(&self.prefill_buckets, remaining_tokens);
            
            if let Some(bucket_idx) = best_bucket {
                let bucket = &mut self.prefill_buckets[bucket_idx];
                
                while !bucket.requests.is_empty() 
                    && batch.request_ids.len() < self.config.max_batch_size
                    && remaining_tokens >= bucket.boundary
                {
                    if let Some(req) = bucket.requests.pop() {
                        batch.request_ids.push(req.id);
                        batch.total_tokens += req.seq_len;
                        batch.max_seq_len = batch.max_seq_len.max(req.seq_len);
                        batch.estimated_cost += req.compute_cost;
                        remaining_tokens = remaining_tokens.saturating_sub(req.seq_len);
                    }
                }
            }
        } else {
            // Simple priority-based batching
            while let Some(prioritized) = self.prefill_queue.pop() {
                let req = prioritized.request;
                
                if batch.total_tokens + req.seq_len > self.config.max_batch_tokens {
                    // Put it back and stop
                    self.prefill_queue.push(PrioritizedRequest {
                        request: req,
                        effective_priority: prioritized.effective_priority,
                    });
                    break;
                }
                
                if batch.request_ids.len() >= self.config.max_batch_size {
                    self.prefill_queue.push(PrioritizedRequest {
                        request: req,
                        effective_priority: prioritized.effective_priority,
                    });
                    break;
                }
                
                batch.request_ids.push(req.id);
                batch.total_tokens += req.seq_len;
                batch.max_seq_len = batch.max_seq_len.max(req.seq_len);
                batch.estimated_cost += req.compute_cost;
            }
        }

        batch.efficiency = batch.calculate_efficiency();
        self.update_stats(&batch);
        batch
    }

    /// Build an optimized decode batch.
    fn build_decode_batch(&mut self) -> OptimizedBatch {
        let mut batch = OptimizedBatch::empty(false);
        
        // Decode is simpler - each sequence generates 1 token
        while let Some(prioritized) = self.decode_queue.pop() {
            let req = prioritized.request;
            
            // For decode, each request contributes 1 token
            if batch.request_ids.len() >= self.config.max_batch_size {
                self.decode_queue.push(PrioritizedRequest {
                    request: req,
                    effective_priority: prioritized.effective_priority,
                });
                break;
            }
            
            // Check token budget (seq_len for context, 1 for new token)
            let tokens_needed = req.seq_len + 1;
            if batch.total_tokens + tokens_needed > self.config.max_batch_tokens {
                self.decode_queue.push(PrioritizedRequest {
                    request: req,
                    effective_priority: prioritized.effective_priority,
                });
                break;
            }
            
            batch.request_ids.push(req.id);
            batch.total_tokens += tokens_needed;
            batch.max_seq_len = batch.max_seq_len.max(req.seq_len + 1);
            batch.estimated_cost += req.compute_cost;
        }

        batch.efficiency = batch.calculate_efficiency();
        self.update_stats(&batch);
        batch
    }

    /// Build a mixed prefill/decode batch.
    fn build_mixed_batch(&mut self) -> Option<OptimizedBatch> {
        // Combine requests with best efficiency
        let mut batch = OptimizedBatch::empty(false);
        let mut remaining_tokens = self.config.max_batch_tokens;

        // Interleave prefill and decode based on priority
        while batch.request_ids.len() < self.config.max_batch_size && remaining_tokens > 0 {
            let prefill_priority = self.prefill_queue.peek()
                .map(|p| p.effective_priority)
                .unwrap_or(f32::MAX);
            let decode_priority = self.decode_queue.peek()
                .map(|p| p.effective_priority)
                .unwrap_or(f32::MAX);

            if prefill_priority == f32::MAX && decode_priority == f32::MAX {
                break;
            }

            if prefill_priority <= decode_priority {
                if let Some(prioritized) = self.prefill_queue.pop() {
                    let req = prioritized.request;
                    if req.seq_len <= remaining_tokens {
                        batch.request_ids.push(req.id);
                        batch.total_tokens += req.seq_len;
                        batch.max_seq_len = batch.max_seq_len.max(req.seq_len);
                        batch.estimated_cost += req.compute_cost;
                        remaining_tokens -= req.seq_len;
                        batch.is_prefill = true; // Mixed batch with prefill
                    } else {
                        self.prefill_queue.push(PrioritizedRequest {
                            request: req,
                            effective_priority: prioritized.effective_priority,
                        });
                        break;
                    }
                }
            } else {
                if let Some(prioritized) = self.decode_queue.pop() {
                    let req = prioritized.request;
                    let tokens_needed = 1; // Decode adds 1 token
                    if tokens_needed <= remaining_tokens {
                        batch.request_ids.push(req.id);
                        batch.total_tokens += tokens_needed;
                        batch.max_seq_len = batch.max_seq_len.max(req.seq_len + 1);
                        batch.estimated_cost += req.compute_cost;
                        remaining_tokens -= tokens_needed;
                    } else {
                        self.decode_queue.push(PrioritizedRequest {
                            request: req,
                            effective_priority: prioritized.effective_priority,
                        });
                        break;
                    }
                }
            }
        }

        if batch.request_ids.is_empty() {
            return None;
        }

        batch.efficiency = batch.calculate_efficiency();
        self.update_stats(&batch);
        Some(batch)
    }

    /// Find the best bucket for batching.
    fn find_best_bucket(&self, buckets: &[LengthBucket], max_tokens: usize) -> Option<usize> {
        buckets
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.requests.is_empty() && b.boundary <= max_tokens)
            .max_by(|(_, a), (_, b)| {
                // Prefer buckets with more requests and better fill rate
                let a_score = a.requests.len() as f32 * (a.total_tokens() as f32 / a.boundary as f32);
                let b_score = b.requests.len() as f32 * (b.total_tokens() as f32 / b.boundary as f32);
                a_score.partial_cmp(&b_score).unwrap_or(Ordering::Equal)
            })
            .map(|(idx, _)| idx)
    }

    /// Update statistics.
    fn update_stats(&mut self, batch: &OptimizedBatch) {
        self.stats.total_batches += 1;
        self.stats.total_requests += batch.request_ids.len() as u64;
        
        // Running average
        let n = self.stats.total_batches as f32;
        self.stats.avg_batch_size = 
            self.stats.avg_batch_size * ((n - 1.0) / n) + 
            batch.request_ids.len() as f32 / n;
        self.stats.avg_efficiency = 
            self.stats.avg_efficiency * ((n - 1.0) / n) + 
            batch.efficiency / n;

        if batch.is_prefill {
            self.stats.prefill_batches += 1;
        } else {
            self.stats.decode_batches += 1;
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Update current time for priority aging.
    pub fn update_time(&mut self, time_us: u64) {
        self.current_time_us = time_us;
    }

    /// Get pending request count.
    pub fn pending_count(&self) -> usize {
        self.prefill_queue.len() + self.decode_queue.len()
    }

    /// Check if there are pending requests.
    pub fn has_pending(&self) -> bool {
        !self.prefill_queue.is_empty() || !self.decode_queue.is_empty()
    }
}

/// Compute cost estimator for batch optimization.
pub struct ComputeCostEstimator {
    /// Base cost per token.
    base_cost_per_token: f32,
    /// Attention cost factor (O(n^2) attention).
    attention_factor: f32,
    /// Model hidden size.
    hidden_size: usize,
    /// Number of layers.
    num_layers: usize,
}

impl ComputeCostEstimator {
    /// Create a new cost estimator.
    pub fn new(hidden_size: usize, num_layers: usize) -> Self {
        Self {
            base_cost_per_token: 1.0,
            attention_factor: 0.01,
            hidden_size,
            num_layers,
        }
    }

    /// Estimate compute cost for a request.
    pub fn estimate(&self, seq_len: usize, is_prefill: bool) -> f32 {
        let base = seq_len as f32 * self.base_cost_per_token;
        
        // Attention is O(n^2) for prefill, O(n) for decode
        let attention = if is_prefill {
            (seq_len * seq_len) as f32 * self.attention_factor
        } else {
            seq_len as f32 * self.attention_factor
        };
        
        (base + attention) * self.num_layers as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_optimizer_basic() {
        let config = BatchOptConfig::default();
        let mut optimizer = BatchOptimizer::new(config);

        // Add some prefill requests
        for i in 0..5 {
            optimizer.add_request(BatchRequest {
                id: i,
                seq_len: 100 + i as usize * 10,
                prompt_len: 100 + i as usize * 10,
                is_prefill: true,
                priority: 0,
                arrival_time_us: i,
                compute_cost: 1.0,
            });
        }

        // Get batch
        let batch = optimizer.get_next_batch().unwrap();
        assert!(batch.is_prefill);
        assert!(!batch.request_ids.is_empty());
        assert!(batch.efficiency > 0.0);
    }

    #[test]
    fn test_batch_optimizer_decode() {
        let config = BatchOptConfig::default();
        let mut optimizer = BatchOptimizer::new(config);

        // Add decode requests
        for i in 0..10 {
            optimizer.add_request(BatchRequest {
                id: i,
                seq_len: 50 + i as usize,
                prompt_len: 50,
                is_prefill: false,
                priority: 0,
                arrival_time_us: i,
                compute_cost: 0.5,
            });
        }

        let batch = optimizer.get_next_batch().unwrap();
        assert!(!batch.is_prefill);
        assert_eq!(batch.request_ids.len(), 10);
    }

    #[test]
    fn test_batch_size_limit() {
        let mut config = BatchOptConfig::default();
        config.max_batch_size = 3;
        let mut optimizer = BatchOptimizer::new(config);

        // Add more requests than batch size
        for i in 0..10 {
            optimizer.add_request(BatchRequest {
                id: i,
                seq_len: 10,
                prompt_len: 10,
                is_prefill: true,
                priority: 0,
                arrival_time_us: i,
                compute_cost: 1.0,
            });
        }

        let batch = optimizer.get_next_batch().unwrap();
        assert!(batch.request_ids.len() <= 3);
    }

    #[test]
    fn test_token_limit() {
        let mut config = BatchOptConfig::default();
        config.max_batch_tokens = 100;
        let mut optimizer = BatchOptimizer::new(config);

        // Add requests that exceed token budget
        for i in 0..5 {
            optimizer.add_request(BatchRequest {
                id: i,
                seq_len: 50,
                prompt_len: 50,
                is_prefill: true,
                priority: 0,
                arrival_time_us: i,
                compute_cost: 1.0,
            });
        }

        let batch = optimizer.get_next_batch().unwrap();
        assert!(batch.total_tokens <= 100);
    }

    #[test]
    fn test_compute_cost_estimator() {
        let estimator = ComputeCostEstimator::new(4096, 32);
        
        let prefill_cost = estimator.estimate(100, true);
        let decode_cost = estimator.estimate(100, false);
        
        // Prefill should be more expensive due to O(n^2) attention
        assert!(prefill_cost > decode_cost);
    }

    #[test]
    fn test_priority_ordering() {
        let config = BatchOptConfig::default();
        let mut optimizer = BatchOptimizer::new(config);

        // Add requests with different priorities
        optimizer.add_request(BatchRequest {
            id: 1,
            seq_len: 100,
            prompt_len: 100,
            is_prefill: true,
            priority: 10, // Low priority
            arrival_time_us: 0,
            compute_cost: 1.0,
        });
        optimizer.add_request(BatchRequest {
            id: 2,
            seq_len: 100,
            prompt_len: 100,
            is_prefill: true,
            priority: 1, // High priority
            arrival_time_us: 1,
            compute_cost: 1.0,
        });

        let batch = optimizer.get_next_batch().unwrap();
        // Higher priority request (id=2) should come first
        assert!(batch.request_ids.contains(&2));
    }
}
