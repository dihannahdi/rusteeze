//! # Batching V2 — Radical Rewrite
//!
//! Continuous batching with prefix cache, O(1) LRU eviction via intrusive
//! doubly-linked list, and pre-allocated scheduling buffers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::simd_dispatch;

/// Batching V2 configuration.
#[derive(Debug, Clone)]
pub struct BatchingV2Config {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Enable prefix caching
    pub enable_prefix_cache: bool,
    /// Maximum prefix cache entries
    pub max_prefix_entries: usize,
    /// KV block size (tokens per block)
    pub kv_block_size: usize,
    /// Padding alignment
    pub padding_alignment: usize,
}

impl Default for BatchingV2Config {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_seq_len: 4096,
            enable_prefix_cache: true,
            max_prefix_entries: 1024,
            kv_block_size: 16,
            padding_alignment: 8,
        }
    }
}

/// A request in the iteration scheduler.
#[derive(Debug, Clone)]
pub struct IterRequest {
    pub request_id: u64,
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub generated_len: usize,
    pub max_tokens: usize,
    pub priority: f32,
    pub is_prefill: bool,
}

/// Batch result from the scheduler.
#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub request_ids: Vec<u64>,
    pub tokens: Vec<u32>,
    pub positions: Vec<usize>,
    pub is_prefill: Vec<bool>,
    pub total_tokens: usize,
}

/// LRU node for prefix cache (intrusive doubly-linked list).
struct LruNode {
    key: u64,
    blocks: Vec<u32>,
    prev: Option<usize>,
    next: Option<usize>,
}

/// O(1) LRU cache for prefix blocks.
pub struct PrefixCache {
    nodes: Vec<LruNode>,
    map: HashMap<u64, usize>, // key -> node index
    head: Option<usize>,      // Most recently used
    tail: Option<usize>,      // Least recently used
    free_list: Vec<usize>,    // Recycled indices
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl PrefixCache {
    /// Create a new prefix cache.
    pub fn new(max_entries: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(max_entries),
            map: HashMap::with_capacity(max_entries),
            head: None,
            tail: None,
            free_list: Vec::new(),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Hash a token prefix.
    pub fn hash_prefix(tokens: &[u32]) -> u64 {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for &t in tokens {
            hash ^= t as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Look up prefix blocks.
    pub fn get(&mut self, tokens: &[u32]) -> Option<&[u32]> {
        let key = Self::hash_prefix(tokens);
        if let Some(&idx) = self.map.get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.move_to_head(idx);
            Some(&self.nodes[idx].blocks)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert prefix blocks into cache.
    pub fn insert(&mut self, tokens: &[u32], blocks: Vec<u32>) {
        let key = Self::hash_prefix(tokens);

        // If already exists, update
        if let Some(&idx) = self.map.get(&key) {
            self.nodes[idx].blocks = blocks;
            self.move_to_head(idx);
            return;
        }

        // Evict if full
        if self.map.len() >= self.max_entries {
            self.evict_lru();
        }

        // Allocate node
        let idx = if let Some(free_idx) = self.free_list.pop() {
            self.nodes[free_idx] = LruNode { key, blocks, prev: None, next: self.head };
            free_idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(LruNode { key, blocks, prev: None, next: self.head });
            idx
        };

        // Link to head
        if let Some(old_head) = self.head {
            self.nodes[old_head].prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }

        self.map.insert(key, idx);
    }

    fn move_to_head(&mut self, idx: usize) {
        if self.head == Some(idx) { return; }

        // Unlink from current position
        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;

        if let Some(p) = prev { self.nodes[p].next = next; }
        if let Some(n) = next { self.nodes[n].prev = prev; }
        if self.tail == Some(idx) { self.tail = prev; }

        // Link to head
        self.nodes[idx].prev = None;
        self.nodes[idx].next = self.head;
        if let Some(old_head) = self.head {
            self.nodes[old_head].prev = Some(idx);
        }
        self.head = Some(idx);
    }

    fn evict_lru(&mut self) {
        if let Some(tail_idx) = self.tail {
            let new_tail = self.nodes[tail_idx].prev;
            if let Some(nt) = new_tail {
                self.nodes[nt].next = None;
            }
            self.tail = new_tail;
            if self.head == Some(tail_idx) { self.head = None; }

            self.map.remove(&self.nodes[tail_idx].key);
            self.free_list.push(tail_idx);
        }
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        if h + m == 0.0 { 0.0 } else { h / (h + m) }
    }
}

/// Iteration-level scheduler for continuous batching.
pub struct IterationScheduler {
    config: BatchingV2Config,
    /// Queued requests (waiting)
    waiting: Vec<IterRequest>,
    /// Running requests (generating)
    running: Vec<IterRequest>,
    /// Prefix cache
    prefix_cache: PrefixCache,
    /// Pre-allocated batch buffer
    batch_buf: ScheduledBatch,
    /// Generation counter
    next_id: AtomicU64,
}

impl IterationScheduler {
    /// Create a new scheduler.
    pub fn new(config: BatchingV2Config) -> Self {
        simd_dispatch::init();
        let max = config.max_batch_size;
        let prefix_cache = PrefixCache::new(config.max_prefix_entries);
        Self {
            config: config.clone(),
            waiting: Vec::with_capacity(max),
            running: Vec::with_capacity(max),
            prefix_cache,
            batch_buf: ScheduledBatch {
                request_ids: Vec::with_capacity(max),
                tokens: Vec::with_capacity(max * 4),
                positions: Vec::with_capacity(max),
                is_prefill: Vec::with_capacity(max),
                total_tokens: 0,
            },
            next_id: AtomicU64::new(0),
        }
    }

    /// Add a request to the scheduler.
    pub fn add_request(&mut self, mut req: IterRequest) {
        req.request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        req.is_prefill = true;
        self.waiting.push(req);
    }

    /// Schedule the next iteration batch.
    pub fn schedule(&mut self) -> &ScheduledBatch {
        // Reuse pre-allocated buffers
        self.batch_buf.request_ids.clear();
        self.batch_buf.tokens.clear();
        self.batch_buf.positions.clear();
        self.batch_buf.is_prefill.clear();
        self.batch_buf.total_tokens = 0;

        let max_batch = self.config.max_batch_size;
        let mut budget = max_batch;

        // First: schedule running (decode) requests — they need only 1 token each
        let mut completed = Vec::new();
        for (i, req) in self.running.iter().enumerate() {
            if budget == 0 { break; }
            if req.generated_len >= req.max_tokens {
                completed.push(i);
                continue;
            }
            self.batch_buf.request_ids.push(req.request_id);
            if let Some(&last_token) = req.tokens.last() {
                self.batch_buf.tokens.push(last_token);
            }
            self.batch_buf.positions.push(req.prompt_len + req.generated_len);
            self.batch_buf.is_prefill.push(false);
            self.batch_buf.total_tokens += 1;
            budget -= 1;
        }

        // Remove completed (reverse order to preserve indices)
        for &i in completed.iter().rev() {
            self.running.remove(i);
        }

        // Second: schedule waiting (prefill) requests
        while budget > 0 && !self.waiting.is_empty() {
            let req = self.waiting.remove(0);
            let prompt_tokens = req.prompt_len.min(budget);

            self.batch_buf.request_ids.push(req.request_id);
            self.batch_buf.tokens.extend_from_slice(&req.tokens[..prompt_tokens]);
            self.batch_buf.positions.push(0);
            self.batch_buf.is_prefill.push(true);
            self.batch_buf.total_tokens += prompt_tokens;
            budget = budget.saturating_sub(prompt_tokens);

            self.running.push(req);
        }

        &self.batch_buf
    }

    /// Notify that a token was generated for a request.
    pub fn on_token_generated(&mut self, request_id: u64, token: u32) {
        if let Some(req) = self.running.iter_mut().find(|r| r.request_id == request_id) {
            req.tokens.push(token);
            req.generated_len += 1;
            req.is_prefill = false;
        }
    }

    /// Remove a completed request.
    pub fn complete_request(&mut self, request_id: u64) {
        self.running.retain(|r| r.request_id != request_id);
    }

    /// Number of waiting requests.
    pub fn waiting_count(&self) -> usize { self.waiting.len() }
    /// Number of running requests.
    pub fn running_count(&self) -> usize { self.running.len() }
    /// Prefix cache reference.
    pub fn prefix_cache(&self) -> &PrefixCache { &self.prefix_cache }
    /// Mutable prefix cache reference.
    pub fn prefix_cache_mut(&mut self) -> &mut PrefixCache { &mut self.prefix_cache }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefix_cache_lru() {
        let mut cache = PrefixCache::new(3);
        cache.insert(&[1, 2, 3], vec![10, 20]);
        cache.insert(&[4, 5, 6], vec![30, 40]);
        cache.insert(&[7, 8, 9], vec![50, 60]);
        assert!(cache.get(&[1, 2, 3]).is_some());

        // Insert one more — should evict LRU = [4,5,6]
        cache.insert(&[10, 11, 12], vec![70, 80]);
        assert!(cache.get(&[4, 5, 6]).is_none());
        assert!(cache.get(&[1, 2, 3]).is_some()); // Was moved to head by earlier get
    }

    #[test]
    fn test_scheduler_basic() {
        let config = BatchingV2Config { max_batch_size: 4, ..Default::default() };
        let mut sched = IterationScheduler::new(config);

        sched.add_request(IterRequest {
            request_id: 0, tokens: vec![1, 2, 3], prompt_len: 3,
            generated_len: 0, max_tokens: 10, priority: 1.0, is_prefill: true,
        });

        let batch = sched.schedule();
        assert_eq!(batch.request_ids.len(), 1);
        assert!(batch.is_prefill[0]);
    }
}
