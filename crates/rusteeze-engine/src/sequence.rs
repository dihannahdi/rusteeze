//! Sequence management for inference.
//!
//! This module manages sequences (individual generation requests)
//! and sequence groups (batched requests).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use rusteeze_core::{FinishReason, RequestStatus, SamplingParams};

/// Unique sequence ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SequenceId(pub u64);

impl SequenceId {
    /// Generate new sequence ID.
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for SequenceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SequenceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seq_{}", self.0)
    }
}

/// Sequence status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Waiting in queue.
    Waiting,
    /// Currently running.
    Running,
    /// Swapped to CPU.
    Swapped,
    /// Finished successfully.
    Finished(FinishReason),
    /// Failed with error.
    Failed,
}

impl SequenceStatus {
    /// Check if sequence is finished.
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Finished(_) | Self::Failed)
    }

    /// Check if sequence is running.
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }
}

/// A single sequence (one beam in beam search, or single generation).
#[derive(Debug)]
pub struct Sequence {
    /// Unique sequence ID.
    pub id: SequenceId,

    /// Prompt tokens.
    prompt_tokens: Vec<u32>,

    /// Generated output tokens.
    output_tokens: Vec<u32>,

    /// Current status.
    status: SequenceStatus,

    /// Logical token blocks assigned.
    logical_blocks: Vec<usize>,

    /// Cumulative log probability.
    cumulative_logprob: f32,

    /// Token log probabilities.
    token_logprobs: Vec<f32>,

    /// Number of tokens computed (for prefix caching).
    num_computed_tokens: usize,

    /// Creation timestamp.
    created_at: Instant,
}

impl Sequence {
    /// Create new sequence.
    pub fn new(id: SequenceId, prompt_tokens: Vec<u32>) -> Self {
        Self {
            id,
            prompt_tokens,
            output_tokens: Vec::new(),
            status: SequenceStatus::Waiting,
            logical_blocks: Vec::new(),
            cumulative_logprob: 0.0,
            token_logprobs: Vec::new(),
            num_computed_tokens: 0,
            created_at: Instant::now(),
        }
    }

    /// Get prompt tokens.
    pub fn prompt_tokens(&self) -> &[u32] {
        &self.prompt_tokens
    }

    /// Get output tokens.
    pub fn output_tokens(&self) -> &[u32] {
        &self.output_tokens
    }

    /// Get all tokens (prompt + output).
    pub fn all_tokens(&self) -> Vec<u32> {
        let mut tokens = self.prompt_tokens.clone();
        tokens.extend(&self.output_tokens);
        tokens
    }

    /// Get prompt length.
    pub fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    /// Get output length.
    pub fn output_len(&self) -> usize {
        self.output_tokens.len()
    }

    /// Get total length.
    pub fn total_len(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// Get status.
    pub fn status(&self) -> SequenceStatus {
        self.status
    }

    /// Set status.
    pub fn set_status(&mut self, status: SequenceStatus) {
        self.status = status;
    }

    /// Add output token.
    pub fn append_token(&mut self, token: u32, logprob: f32) {
        self.output_tokens.push(token);
        self.token_logprobs.push(logprob);
        self.cumulative_logprob += logprob;
    }

    /// Get last token.
    pub fn last_token(&self) -> Option<u32> {
        self.output_tokens.last().copied()
            .or_else(|| self.prompt_tokens.last().copied())
    }

    /// Get logical blocks.
    pub fn logical_blocks(&self) -> &[usize] {
        &self.logical_blocks
    }

    /// Add logical block.
    pub fn add_logical_block(&mut self, block_id: usize) {
        self.logical_blocks.push(block_id);
    }

    /// Get cumulative log probability.
    pub fn cumulative_logprob(&self) -> f32 {
        self.cumulative_logprob
    }

    /// Get average log probability.
    pub fn avg_logprob(&self) -> f32 {
        if self.output_tokens.is_empty() {
            0.0
        } else {
            self.cumulative_logprob / self.output_tokens.len() as f32
        }
    }

    /// Check if finished.
    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    /// Mark as finished.
    pub fn finish(&mut self, reason: FinishReason) {
        self.status = SequenceStatus::Finished(reason);
    }

    /// Get number of new tokens to compute.
    pub fn get_num_new_tokens(&self) -> usize {
        self.total_len().saturating_sub(self.num_computed_tokens)
    }

    /// Update computed tokens count.
    pub fn set_num_computed_tokens(&mut self, num: usize) {
        self.num_computed_tokens = num;
    }

    /// Get elapsed time since creation.
    pub fn elapsed(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

/// Group ID for sequence groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GroupId(pub Uuid);

impl GroupId {
    /// Generate new group ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for GroupId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for GroupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "grp_{}", self.0)
    }
}

/// A group of sequences for a single request.
/// Multiple sequences for beam search or parallel sampling.
#[derive(Debug)]
pub struct SequenceGroup {
    /// Group ID.
    pub id: GroupId,

    /// Request ID (from API).
    pub request_id: String,

    /// Sequences in this group.
    sequences: HashMap<SequenceId, Sequence>,

    /// Sampling parameters.
    pub sampling_params: SamplingParams,

    /// Arrival time.
    arrival_time: Instant,

    /// Is prefill phase complete.
    is_prefill_done: bool,

    /// Number of sequences to return (best_of).
    pub best_of: usize,

    /// Stop sequences.
    pub stop_sequences: Vec<Vec<u32>>,

    /// Maximum tokens to generate.
    pub max_tokens: usize,
}

impl SequenceGroup {
    /// Create new sequence group.
    pub fn new(
        request_id: String,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
    ) -> Self {
        let id = GroupId::new();
        let seq_id = SequenceId::new();
        let sequence = Sequence::new(seq_id, prompt_tokens);

        let mut sequences = HashMap::new();
        sequences.insert(seq_id, sequence);

        let best_of = sampling_params.best_of.unwrap_or(1).max(1);

        Self {
            id,
            request_id,
            sequences,
            sampling_params,
            arrival_time: Instant::now(),
            is_prefill_done: false,
            best_of,
            stop_sequences: Vec::new(),
            max_tokens,
        }
    }

    /// Add sequence to group.
    pub fn add_sequence(&mut self, sequence: Sequence) {
        self.sequences.insert(sequence.id, sequence);
    }

    /// Get sequence by ID.
    pub fn get_sequence(&self, id: &SequenceId) -> Option<&Sequence> {
        self.sequences.get(id)
    }

    /// Get mutable sequence by ID.
    pub fn get_sequence_mut(&mut self, id: &SequenceId) -> Option<&mut Sequence> {
        self.sequences.get_mut(id)
    }

    /// Get all sequences.
    pub fn sequences(&self) -> impl Iterator<Item = &Sequence> {
        self.sequences.values()
    }

    /// Get all sequences mutably.
    pub fn sequences_mut(&mut self) -> impl Iterator<Item = &mut Sequence> {
        self.sequences.values_mut()
    }

    /// Get sequence IDs.
    pub fn sequence_ids(&self) -> Vec<SequenceId> {
        self.sequences.keys().copied().collect()
    }

    /// Get number of sequences.
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Get prompt length (from first sequence).
    pub fn prompt_len(&self) -> usize {
        self.sequences
            .values()
            .next()
            .map(|s| s.prompt_len())
            .unwrap_or(0)
    }

    /// Check if prefill is done.
    pub fn is_prefill_done(&self) -> bool {
        self.is_prefill_done
    }

    /// Mark prefill as done.
    pub fn set_prefill_done(&mut self) {
        self.is_prefill_done = true;
    }

    /// Check if all sequences are finished.
    pub fn is_finished(&self) -> bool {
        self.sequences.values().all(|s| s.is_finished())
    }

    /// Get unfinished sequences.
    pub fn get_unfinished_sequences(&self) -> Vec<&Sequence> {
        self.sequences
            .values()
            .filter(|s| !s.is_finished())
            .collect()
    }

    /// Get finished sequences.
    pub fn get_finished_sequences(&self) -> Vec<&Sequence> {
        self.sequences
            .values()
            .filter(|s| s.is_finished())
            .collect()
    }

    /// Get best sequence by log probability.
    pub fn get_best_sequence(&self) -> Option<&Sequence> {
        self.sequences
            .values()
            .filter(|s| s.is_finished())
            .max_by(|a, b| {
                a.cumulative_logprob()
                    .partial_cmp(&b.cumulative_logprob())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Check if should stop on token.
    pub fn should_stop(&self, sequence: &Sequence, eos_token_id: u32) -> Option<FinishReason> {
        // Check max tokens
        if sequence.output_len() >= self.max_tokens {
            return Some(FinishReason::Length);
        }

        // Check EOS token
        if let Some(last_token) = sequence.last_token() {
            if last_token == eos_token_id {
                return Some(FinishReason::Stop);
            }
        }

        // Check stop sequences
        let output = sequence.output_tokens();
        for stop_seq in &self.stop_sequences {
            if output.len() >= stop_seq.len() {
                let suffix = &output[output.len() - stop_seq.len()..];
                if suffix == stop_seq.as_slice() {
                    return Some(FinishReason::Stop);
                }
            }
        }

        None
    }

    /// Get waiting time.
    pub fn waiting_time(&self) -> std::time::Duration {
        self.arrival_time.elapsed()
    }

    /// Get total generated tokens.
    pub fn total_generated_tokens(&self) -> usize {
        self.sequences.values().map(|s| s.output_len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_id() {
        let id1 = SequenceId::new();
        let id2 = SequenceId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_sequence_tokens() {
        let mut seq = Sequence::new(SequenceId::new(), vec![1, 2, 3]);
        assert_eq!(seq.prompt_len(), 3);
        assert_eq!(seq.output_len(), 0);

        seq.append_token(4, -0.5);
        seq.append_token(5, -0.3);
        assert_eq!(seq.output_len(), 2);
        assert_eq!(seq.total_len(), 5);
        assert_eq!(seq.all_tokens(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_sequence_group() {
        let sampling = SamplingParams::default();
        let mut group = SequenceGroup::new(
            "test".to_string(),
            vec![1, 2, 3],
            sampling,
            100,
        );

        assert_eq!(group.num_sequences(), 1);
        assert_eq!(group.prompt_len(), 3);
        assert!(!group.is_finished());
    }
}
