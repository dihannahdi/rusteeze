//! # Speculative Decoding — Radical Rewrite
//!
//! Tree-based speculative decoding with Medusa heads, vectorized
//! draft generation, and O(k) partial sort for top-k selection.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::simd_dispatch::{self, dot_product, partial_sort_top_k, softmax_inplace, vec_argmax, with_scratch_a};

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Maximum speculation depth (number of tokens to draft ahead)
    pub max_speculation_length: usize,
    /// Number of candidates per position
    pub num_candidates: usize,
    /// Number of Medusa heads
    pub num_medusa_heads: usize,
    /// Acceptance threshold
    pub acceptance_threshold: f32,
    /// Minimum acceptance rate to keep speculating
    pub min_acceptance_rate: f32,
    /// Enable adaptive speculation length
    pub adaptive: bool,
    /// Top-k for draft sampling
    pub draft_top_k: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            max_speculation_length: 5,
            num_candidates: 3,
            num_medusa_heads: 4,
            acceptance_threshold: 0.5,
            min_acceptance_rate: 0.3,
            adaptive: true,
            draft_top_k: 10,
            vocab_size: 32000,
        }
    }
}

/// A node in the speculation tree.
#[derive(Debug, Clone)]
pub struct SpeculationTreeNode {
    /// Token ID
    pub token_id: u32,
    /// Log probability from draft model
    pub logprob: f32,
    /// Depth in tree (0 = root's children)
    pub depth: usize,
    /// Parent node index (usize::MAX for root children)
    pub parent: usize,
    /// Children indices
    pub children: Vec<usize>,
}

/// Tree of speculated tokens.
#[derive(Debug, Clone)]
pub struct SpeculationTree {
    /// Tree nodes (index 0..N)
    pub nodes: Vec<SpeculationTreeNode>,
    /// Maximum depth
    pub max_depth: usize,
}

impl SpeculationTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        Self { nodes: Vec::new(), max_depth: 0 }
    }

    /// Add a node and return its index.
    pub fn add_node(&mut self, token_id: u32, logprob: f32, depth: usize, parent: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(SpeculationTreeNode {
            token_id, logprob, depth, parent, children: Vec::new(),
        });
        if parent < self.nodes.len() - 1 {
            self.nodes[parent].children.push(idx);
        }
        self.max_depth = self.max_depth.max(depth);
        idx
    }

    /// Get all leaf paths (each path = sequence of token IDs from root to leaf).
    pub fn get_leaf_paths(&self) -> Vec<Vec<u32>> {
        let mut paths = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.children.is_empty() {
                // Walk up to build path
                let mut path = Vec::new();
                let mut cur = i;
                while cur < self.nodes.len() {
                    path.push(self.nodes[cur].token_id);
                    if self.nodes[cur].parent == usize::MAX { break; }
                    cur = self.nodes[cur].parent;
                }
                path.reverse();
                paths.push(path);
            }
        }
        paths
    }

    /// Number of nodes.
    pub fn len(&self) -> usize { self.nodes.len() }

    /// Is the tree empty?
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }
}

/// Statistics for speculative decoding.
#[derive(Debug, Default)]
pub struct SpeculativeStats {
    /// Total speculation attempts
    pub total_attempts: AtomicU64,
    /// Total tokens accepted
    pub tokens_accepted: AtomicU64,
    /// Total tokens drafted
    pub tokens_drafted: AtomicU64,
    /// Current adaptive speculation length
    pub current_spec_length: AtomicU64,
}

impl Clone for SpeculativeStats {
    fn clone(&self) -> Self {
        Self {
            total_attempts: AtomicU64::new(self.total_attempts.load(Ordering::Relaxed)),
            tokens_accepted: AtomicU64::new(self.tokens_accepted.load(Ordering::Relaxed)),
            tokens_drafted: AtomicU64::new(self.tokens_drafted.load(Ordering::Relaxed)),
            current_spec_length: AtomicU64::new(self.current_spec_length.load(Ordering::Relaxed)),
        }
    }
}

impl SpeculativeStats {
    /// Acceptance rate.
    pub fn acceptance_rate(&self) -> f64 {
        let drafted = self.tokens_drafted.load(Ordering::Relaxed);
        if drafted == 0 { return 0.0; }
        self.tokens_accepted.load(Ordering::Relaxed) as f64 / drafted as f64
    }
}

/// Medusa heads for multi-token drafting.
/// Uses vectorized logit computation with pre-resolved SIMD.
pub struct MedusaHeads {
    /// Weight matrices: [num_heads, vocab_size, hidden_size]
    weights: Vec<Vec<f32>>,
    /// Biases: [num_heads, vocab_size]
    biases: Vec<Vec<f32>>,
    /// Number of heads
    num_heads: usize,
    /// Vocabulary size
    vocab_size: usize,
    /// Hidden size
    hidden_size: usize,
}

impl MedusaHeads {
    /// Create Medusa heads.
    pub fn new(num_heads: usize, vocab_size: usize, hidden_size: usize) -> Self {
        // Initialize with small random weights
        let mut rng = fastrand::Rng::new();
        let scale = (2.0 / (hidden_size + vocab_size) as f32).sqrt();

        let weights: Vec<Vec<f32>> = (0..num_heads)
            .map(|_| (0..vocab_size * hidden_size)
                .map(|_| (rng.f32() * 2.0 - 1.0) * scale)
                .collect())
            .collect();

        let biases: Vec<Vec<f32>> = (0..num_heads)
            .map(|_| vec![0.0; vocab_size])
            .collect();

        Self { weights, biases, num_heads, vocab_size, hidden_size }
    }

    /// Compute logits for all heads — parallel across heads.
    /// `hidden_state`: [hidden_size]
    /// Returns: Vec of logit vectors, one per head.
    pub fn compute_logits(&self, hidden_state: &[f32]) -> Vec<Vec<f32>> {
        (0..self.num_heads)
            .into_par_iter()
            .map(|h| {
                let weight = &self.weights[h];
                let bias = &self.biases[h];
                let mut logits = vec![0.0f32; self.vocab_size];

                for v in 0..self.vocab_size {
                    let w_row = &weight[v * self.hidden_size..(v + 1) * self.hidden_size];
                    logits[v] = dot_product(hidden_state, w_row) + bias[v];
                }
                logits
            })
            .collect()
    }

    /// Sample top-k from logits using O(n) partial sort.
    pub fn sample_top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
        let k = k.min(logits.len());
        let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        let top = partial_sort_top_k(&mut indexed, k);

        // Sort the top-k by probability (descending)
        let mut result: Vec<(u32, f32)> = top.iter().map(|&(i, v)| (i as u32, v)).collect();
        result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }
}

/// Speculative decoding engine.
pub struct SpeculativeEngine {
    config: SpeculativeConfig,
    /// Medusa heads for multi-token drafting
    medusa: Option<MedusaHeads>,
    /// Adaptive speculation length history (ring buffer of acceptance rates)
    acceptance_history: parking_lot::Mutex<Vec<f32>>,
    /// Current adaptive speculation length
    current_length: std::sync::atomic::AtomicUsize,
    /// Statistics
    stats: SpeculativeStats,
}

impl SpeculativeEngine {
    /// Create a new speculative engine.
    pub fn new(config: SpeculativeConfig) -> Self {
        simd_dispatch::init();
        let medusa = if config.num_medusa_heads > 0 {
            Some(MedusaHeads::new(config.num_medusa_heads, config.vocab_size, 128))
        } else { None };

        Self {
            current_length: std::sync::atomic::AtomicUsize::new(config.max_speculation_length),
            acceptance_history: parking_lot::Mutex::new(Vec::with_capacity(64)),
            config,
            medusa,
            stats: SpeculativeStats::default(),
        }
    }

    /// Build a speculation tree from draft model logits.
    ///
    /// `draft_logits`: [spec_length, vocab_size] — logits from draft model
    /// Returns a speculation tree with candidate sequences.
    pub fn build_tree(&self, draft_logits: &[Vec<f32>]) -> SpeculationTree {
        let mut tree = SpeculationTree::new();
        let spec_len = self.current_length.load(Ordering::Relaxed).min(draft_logits.len());
        let top_k = self.config.num_candidates;

        if spec_len == 0 || draft_logits.is_empty() {
            return tree;
        }

        // Level 0: top-k from first position
        let top_tokens = MedusaHeads::sample_top_k(&draft_logits[0], top_k);
        let mut current_level: Vec<usize> = Vec::new();

        for (token_id, logprob) in &top_tokens {
            let idx = tree.add_node(*token_id, *logprob, 0, usize::MAX);
            current_level.push(idx);
        }

        // Deeper levels: extend each leaf
        for depth in 1..spec_len {
            if depth >= draft_logits.len() { break; }
            let mut next_level = Vec::new();

            for &parent_idx in &current_level {
                let top = MedusaHeads::sample_top_k(&draft_logits[depth], top_k.min(2));
                for (token_id, logprob) in &top {
                    let idx = tree.add_node(*token_id, *logprob, depth, parent_idx);
                    next_level.push(idx);
                }
            }
            current_level = next_level;
        }

        self.stats.tokens_drafted.fetch_add(tree.len() as u64, Ordering::Relaxed);
        tree
    }

    /// Verify drafted tokens against the target model.
    /// Returns the number of accepted tokens and accepted token IDs.
    pub fn verify_drafts(
        &self,
        tree: &SpeculationTree,
        target_logits: &[Vec<f32>],
    ) -> (usize, Vec<u32>) {
        self.stats.total_attempts.fetch_add(1, Ordering::Relaxed);

        if tree.is_empty() || target_logits.is_empty() {
            return (0, Vec::new());
        }

        let paths = tree.get_leaf_paths();
        let mut best_accepted = 0;
        let mut best_tokens = Vec::new();

        for path in &paths {
            let mut accepted = Vec::new();
            for (i, &token_id) in path.iter().enumerate() {
                if i >= target_logits.len() { break; }
                let logits = &target_logits[i];

                // Get target model's top token
                let (target_top_idx, target_top_val) = vec_argmax(logits);

                // Accept if drafted token matches target top or has high enough probability
                if token_id == target_top_idx as u32 {
                    accepted.push(token_id);
                } else {
                    // Check probability ratio
                    let draft_prob = if (token_id as usize) < logits.len() {
                        logits[token_id as usize]
                    } else { f32::NEG_INFINITY };

                    if draft_prob > target_top_val * self.config.acceptance_threshold {
                        accepted.push(token_id);
                    } else {
                        // Reject — add target's top token instead
                        accepted.push(target_top_idx as u32);
                        break;
                    }
                }
            }

            if accepted.len() > best_accepted {
                best_accepted = accepted.len();
                best_tokens = accepted;
            }
        }

        self.stats.tokens_accepted.fetch_add(best_accepted as u64, Ordering::Relaxed);

        // Update adaptive length
        if self.config.adaptive {
            self.update_adaptive_length(best_accepted);
        }

        (best_accepted, best_tokens)
    }

    /// Update speculation length based on recent acceptance rates.
    fn update_adaptive_length(&self, accepted: usize) {
        let current = self.current_length.load(Ordering::Relaxed);
        let rate = accepted as f32 / current.max(1) as f32;

        let mut history = self.acceptance_history.lock();
        history.push(rate);
        if history.len() > 32 { history.remove(0); }

        let avg_rate: f32 = history.iter().sum::<f32>() / history.len() as f32;

        let new_len = if avg_rate > 0.8 {
            (current + 1).min(self.config.max_speculation_length * 2)
        } else if avg_rate < self.config.min_acceptance_rate {
            (current.saturating_sub(1)).max(1)
        } else {
            current
        };

        self.current_length.store(new_len, Ordering::Relaxed);
        self.stats.current_spec_length.store(new_len as u64, Ordering::Relaxed);
    }

    /// Get current adaptive speculation length.
    pub fn current_speculation_length(&self) -> usize {
        self.current_length.load(Ordering::Relaxed)
    }

    /// Get statistics.
    pub fn stats(&self) -> &SpeculativeStats { &self.stats }
    /// Get config.
    pub fn config(&self) -> &SpeculativeConfig { &self.config }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculation_tree() {
        let mut tree = SpeculationTree::new();
        let idx0 = tree.add_node(42, -0.5, 0, usize::MAX);
        let idx1 = tree.add_node(43, -0.3, 1, idx0);
        let _idx2 = tree.add_node(44, -0.2, 2, idx1);

        assert_eq!(tree.len(), 3);
        let paths = tree.get_leaf_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![42, 43, 44]);
    }

    #[test]
    fn test_medusa_sample_top_k() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6, 0.7, 0.0];
        let top3 = MedusaHeads::sample_top_k(&logits, 3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].0, 3); // 0.9
        assert_eq!(top3[1].0, 5); // 0.8
    }

    #[test]
    fn test_speculative_engine() {
        let config = SpeculativeConfig {
            max_speculation_length: 3,
            num_candidates: 2,
            vocab_size: 10,
            ..Default::default()
        };
        let engine = SpeculativeEngine::new(config);

        let draft_logits: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..10).map(|i| if i == 5 { 10.0 } else { 0.1 }).collect())
            .collect();

        let tree = engine.build_tree(&draft_logits);
        assert!(!tree.is_empty());

        let target_logits: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..10).map(|i| if i == 5 { 10.0 } else { 0.1 }).collect())
            .collect();

        let (accepted, tokens) = engine.verify_drafts(&tree, &target_logits);
        assert!(accepted > 0);
        assert!(!tokens.is_empty());
    }
}
