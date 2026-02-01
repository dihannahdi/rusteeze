//! Sequence management for inference requests.
//!
//! This module handles the lifecycle of token sequences during inference,
//! including prompt processing, generation, and completion tracking.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::sampling::SamplingParams;
use crate::types::{FinishReason, RequestId, SequenceId, TokenId};

/// Global sequence counter for unique IDs.
static SEQUENCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a new unique sequence ID.
pub fn generate_sequence_id() -> SequenceId {
    SequenceId(SEQUENCE_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Status of a sequence in the scheduling pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SequenceStatus {
    /// Waiting to be scheduled.
    Waiting,
    /// Currently running (prompt or generation phase).
    Running,
    /// Swapped out to CPU memory.
    Swapped,
    /// Successfully finished.
    Finished(FinishReason),
    /// Finished due to an error.
    FinishedError,
    /// Finished due to being ignored (e.g., exceeded limits).
    FinishedIgnored,
    /// Aborted by user request.
    FinishedAborted,
}

impl SequenceStatus {
    /// Check if the sequence is finished.
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            SequenceStatus::Finished(_)
                | SequenceStatus::FinishedError
                | SequenceStatus::FinishedIgnored
                | SequenceStatus::FinishedAborted
        )
    }

    /// Check if the sequence is currently running.
    pub fn is_running(&self) -> bool {
        matches!(self, SequenceStatus::Running)
    }

    /// Check if the sequence is waiting.
    pub fn is_waiting(&self) -> bool {
        matches!(self, SequenceStatus::Waiting)
    }

    /// Check if the sequence is swapped.
    pub fn is_swapped(&self) -> bool {
        matches!(self, SequenceStatus::Swapped)
    }

    /// Get the finish reason if finished.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        match self {
            SequenceStatus::Finished(reason) => Some(*reason),
            SequenceStatus::FinishedError => Some(FinishReason::Error),
            SequenceStatus::FinishedAborted => Some(FinishReason::Abort),
            _ => None,
        }
    }
}

/// A sequence of tokens being processed.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique sequence identifier.
    pub seq_id: SequenceId,

    /// Parent request ID.
    pub request_id: RequestId,

    /// Input token IDs (prompt).
    pub prompt_token_ids: Vec<TokenId>,

    /// Generated output token IDs.
    pub output_token_ids: Vec<TokenId>,

    /// Current status.
    pub status: SequenceStatus,

    /// Sampling parameters.
    pub sampling_params: Arc<SamplingParams>,

    /// Cumulative log probability.
    pub cumulative_logprob: f32,

    /// Number of tokens from prompt processed.
    pub num_prompt_tokens_processed: usize,

    /// Block table for KV cache.
    pub block_table: Vec<u32>,

    /// Logical token position.
    pub logical_token_blocks: usize,

    /// Creation timestamp.
    pub created_at: Instant,

    /// First token timestamp.
    pub first_token_at: Option<Instant>,

    /// Finish timestamp.
    pub finished_at: Option<Instant>,

    /// Number of computed tokens in current step.
    pub num_computed_tokens: usize,

    /// Whether this is in prefill phase.
    pub is_prefill: bool,

    /// Prefix cache hit length.
    pub prefix_cache_hit_len: usize,

    /// Parent sequence ID (for beam search).
    pub parent_seq_id: Option<SequenceId>,

    /// Beam search score.
    pub beam_score: f32,
}

impl Sequence {
    /// Create a new sequence.
    pub fn new(
        request_id: RequestId,
        prompt_token_ids: Vec<TokenId>,
        sampling_params: Arc<SamplingParams>,
    ) -> Self {
        Self {
            seq_id: generate_sequence_id(),
            request_id,
            prompt_token_ids,
            output_token_ids: Vec::new(),
            status: SequenceStatus::Waiting,
            sampling_params,
            cumulative_logprob: 0.0,
            num_prompt_tokens_processed: 0,
            block_table: Vec::new(),
            logical_token_blocks: 0,
            created_at: Instant::now(),
            first_token_at: None,
            finished_at: None,
            num_computed_tokens: 0,
            is_prefill: true,
            prefix_cache_hit_len: 0,
            parent_seq_id: None,
            beam_score: 0.0,
        }
    }

    /// Get total length (prompt + output).
    pub fn len(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    /// Check if sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.prompt_token_ids.is_empty() && self.output_token_ids.is_empty()
    }

    /// Get prompt length.
    pub fn prompt_len(&self) -> usize {
        self.prompt_token_ids.len()
    }

    /// Get output length.
    pub fn output_len(&self) -> usize {
        self.output_token_ids.len()
    }

    /// Get the last token ID.
    pub fn last_token_id(&self) -> Option<TokenId> {
        self.output_token_ids
            .last()
            .or_else(|| self.prompt_token_ids.last())
            .copied()
    }

    /// Get all token IDs.
    pub fn all_token_ids(&self) -> impl Iterator<Item = &TokenId> {
        self.prompt_token_ids
            .iter()
            .chain(self.output_token_ids.iter())
    }

    /// Append a generated token.
    pub fn append_token(&mut self, token_id: TokenId, logprob: f32) {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        self.output_token_ids.push(token_id);
        self.cumulative_logprob += logprob;
    }

    /// Mark sequence as finished.
    pub fn finish(&mut self, reason: FinishReason) {
        self.status = SequenceStatus::Finished(reason);
        self.finished_at = Some(Instant::now());
    }

    /// Check if max tokens reached.
    pub fn is_max_tokens_reached(&self) -> bool {
        if let Some(max_tokens) = self.sampling_params.max_tokens {
            self.output_token_ids.len() >= max_tokens as usize
        } else {
            false
        }
    }

    /// Check if min tokens reached.
    pub fn is_min_tokens_reached(&self) -> bool {
        self.output_token_ids.len() >= self.sampling_params.min_tokens as usize
    }

    /// Get time to first token (TTFT).
    pub fn time_to_first_token(&self) -> Option<std::time::Duration> {
        self.first_token_at.map(|t| t.duration_since(self.created_at))
    }

    /// Get total generation time.
    pub fn total_time(&self) -> Option<std::time::Duration> {
        self.finished_at.map(|t| t.duration_since(self.created_at))
    }

    /// Get inter-token latency (ITL).
    pub fn inter_token_latency(&self) -> Option<std::time::Duration> {
        if self.output_token_ids.is_empty() {
            return None;
        }
        if let (Some(first), Some(finished)) = (self.first_token_at, self.finished_at) {
            let decode_time = finished.duration_since(first);
            let num_decode_tokens = self.output_token_ids.len().saturating_sub(1);
            if num_decode_tokens > 0 {
                return Some(decode_time / num_decode_tokens as u32);
            }
        }
        None
    }

    /// Get tokens per second.
    pub fn tokens_per_second(&self) -> Option<f64> {
        if let Some(total) = self.total_time() {
            let secs = total.as_secs_f64();
            if secs > 0.0 {
                return Some(self.output_token_ids.len() as f64 / secs);
            }
        }
        None
    }

    /// Fork this sequence (for beam search).
    pub fn fork(&self, new_sampling_params: Option<Arc<SamplingParams>>) -> Self {
        let mut forked = self.clone();
        forked.seq_id = generate_sequence_id();
        forked.parent_seq_id = Some(self.seq_id);
        if let Some(params) = new_sampling_params {
            forked.sampling_params = params;
        }
        forked
    }

    /// Get number of blocks needed.
    pub fn num_blocks_needed(&self, block_size: usize) -> usize {
        (self.len() + block_size - 1) / block_size
    }

    /// Update block table.
    pub fn set_block_table(&mut self, block_table: Vec<u32>) {
        self.block_table = block_table;
    }

    /// Append a block to the table.
    pub fn append_block(&mut self, block_id: u32) {
        self.block_table.push(block_id);
    }

    /// Check if sequence should stop on token.
    pub fn should_stop_on_token(&self, token_id: TokenId) -> bool {
        self.sampling_params.stop_token_ids.contains(&token_id.0)
    }

    /// Transition to running state.
    pub fn set_running(&mut self) {
        self.status = SequenceStatus::Running;
    }

    /// Transition to waiting state.
    pub fn set_waiting(&mut self) {
        self.status = SequenceStatus::Waiting;
    }

    /// Transition to swapped state.
    pub fn set_swapped(&mut self) {
        self.status = SequenceStatus::Swapped;
    }

    /// Complete prefill phase.
    pub fn complete_prefill(&mut self) {
        self.is_prefill = false;
        self.num_prompt_tokens_processed = self.prompt_token_ids.len();
    }
}

/// A group of sequences from the same request.
#[derive(Debug)]
pub struct SequenceGroup {
    /// Request ID.
    pub request_id: RequestId,

    /// Sequences in this group.
    pub sequences: Vec<Sequence>,

    /// Sampling parameters (shared).
    pub sampling_params: Arc<SamplingParams>,

    /// Arrival time.
    pub arrival_time: Instant,

    /// Priority (lower = higher priority).
    pub priority: u32,

    /// Whether streaming is enabled.
    pub stream: bool,

    /// Prompt text (for stop string matching).
    pub prompt: Option<String>,

    /// Whether this group is preempted.
    pub is_preempted: bool,

    /// Number of times preempted.
    pub num_preemptions: u32,

    /// State for lora (if any).
    pub lora_request: Option<LoraRequest>,

    /// Multi-modal data (if any).
    pub multi_modal_data: Option<serde_json::Value>,
}

impl SequenceGroup {
    /// Create a new sequence group.
    pub fn new(
        request_id: RequestId,
        sequences: Vec<Sequence>,
        sampling_params: Arc<SamplingParams>,
        arrival_time: Instant,
    ) -> Self {
        Self {
            request_id,
            sequences,
            sampling_params,
            arrival_time,
            priority: 0,
            stream: false,
            prompt: None,
            is_preempted: false,
            num_preemptions: 0,
            lora_request: None,
            multi_modal_data: None,
        }
    }

    /// Create from a single sequence.
    pub fn from_sequence(sequence: Sequence) -> Self {
        let request_id = sequence.request_id.clone();
        let sampling_params = sequence.sampling_params.clone();
        Self::new(
            request_id,
            vec![sequence],
            sampling_params,
            Instant::now(),
        )
    }

    /// Get the number of sequences.
    pub fn num_seqs(&self) -> usize {
        self.sequences.len()
    }

    /// Get running sequences.
    pub fn running_seqs(&self) -> impl Iterator<Item = &Sequence> {
        self.sequences.iter().filter(|s| s.status.is_running())
    }

    /// Get mutable running sequences.
    pub fn running_seqs_mut(&mut self) -> impl Iterator<Item = &mut Sequence> {
        self.sequences.iter_mut().filter(|s| s.status.is_running())
    }

    /// Get waiting sequences.
    pub fn waiting_seqs(&self) -> impl Iterator<Item = &Sequence> {
        self.sequences.iter().filter(|s| s.status.is_waiting())
    }

    /// Get finished sequences.
    pub fn finished_seqs(&self) -> impl Iterator<Item = &Sequence> {
        self.sequences.iter().filter(|s| s.status.is_finished())
    }

    /// Check if all sequences are finished.
    pub fn is_finished(&self) -> bool {
        self.sequences.iter().all(|s| s.status.is_finished())
    }

    /// Check if any sequence is running.
    pub fn has_running(&self) -> bool {
        self.sequences.iter().any(|s| s.status.is_running())
    }

    /// Check if any sequence is in prefill.
    pub fn is_prefill(&self) -> bool {
        self.sequences.iter().any(|s| s.is_prefill)
    }

    /// Get the number of unfinished sequences.
    pub fn num_unfinished(&self) -> usize {
        self.sequences.iter().filter(|s| !s.status.is_finished()).count()
    }

    /// Get the first sequence.
    pub fn first(&self) -> Option<&Sequence> {
        self.sequences.first()
    }

    /// Get the first mutable sequence.
    pub fn first_mut(&mut self) -> Option<&mut Sequence> {
        self.sequences.first_mut()
    }

    /// Get sequence by ID.
    pub fn get_seq(&self, seq_id: SequenceId) -> Option<&Sequence> {
        self.sequences.iter().find(|s| s.seq_id == seq_id)
    }

    /// Get mutable sequence by ID.
    pub fn get_seq_mut(&mut self, seq_id: SequenceId) -> Option<&mut Sequence> {
        self.sequences.iter_mut().find(|s| s.seq_id == seq_id)
    }

    /// Add a sequence.
    pub fn add_seq(&mut self, seq: Sequence) {
        self.sequences.push(seq);
    }

    /// Remove finished sequences.
    pub fn remove_finished(&mut self) {
        self.sequences.retain(|s| !s.status.is_finished());
    }

    /// Get prompt length (from first sequence).
    pub fn prompt_len(&self) -> usize {
        self.sequences.first().map(|s| s.prompt_len()).unwrap_or(0)
    }

    /// Get maximum output length across sequences.
    pub fn max_output_len(&self) -> usize {
        self.sequences.iter().map(|s| s.output_len()).max().unwrap_or(0)
    }

    /// Get total tokens across all sequences.
    pub fn total_tokens(&self) -> usize {
        self.sequences.iter().map(|s| s.len()).sum()
    }

    /// Get wait time.
    pub fn wait_time(&self) -> std::time::Duration {
        self.arrival_time.elapsed()
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set streaming.
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set prompt.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Mark as preempted.
    pub fn mark_preempted(&mut self) {
        self.is_preempted = true;
        self.num_preemptions += 1;
    }

    /// Get the best sequence (by cumulative logprob).
    pub fn best_sequence(&self) -> Option<&Sequence> {
        self.sequences
            .iter()
            .filter(|s| s.status.is_finished())
            .max_by(|a, b| a.cumulative_logprob.partial_cmp(&b.cumulative_logprob).unwrap())
    }
}

/// LoRA request information.
#[derive(Debug, Clone)]
pub struct LoraRequest {
    /// LoRA adapter ID.
    pub lora_id: String,

    /// LoRA adapter name.
    pub lora_name: String,

    /// Path to LoRA weights.
    pub lora_path: String,

    /// LoRA rank.
    pub lora_rank: u32,
}

impl LoraRequest {
    /// Create a new LoRA request.
    pub fn new(
        lora_id: impl Into<String>,
        lora_name: impl Into<String>,
        lora_path: impl Into<String>,
        lora_rank: u32,
    ) -> Self {
        Self {
            lora_id: lora_id.into(),
            lora_name: lora_name.into(),
            lora_path: lora_path.into(),
            lora_rank,
        }
    }
}

/// Output from a completed sequence.
#[derive(Debug, Clone)]
pub struct SequenceOutput {
    /// Sequence ID.
    pub seq_id: SequenceId,

    /// Parent sequence ID (for beam search).
    pub parent_seq_id: Option<SequenceId>,

    /// Output token ID.
    pub output_token: TokenId,

    /// Log probability of the token.
    pub logprob: f32,

    /// Top log probabilities.
    pub top_logprobs: Option<Vec<(TokenId, f32)>>,
}

/// Output from a sequence group.
#[derive(Debug, Clone)]
pub struct SequenceGroupOutput {
    /// Outputs for each sequence.
    pub outputs: Vec<SequenceOutput>,

    /// Prompt log probabilities.
    pub prompt_logprobs: Option<Vec<f32>>,
}

/// Metadata about a scheduled sequence group.
#[derive(Debug, Clone)]
pub struct SequenceGroupMetadata {
    /// Request ID.
    pub request_id: RequestId,

    /// Whether in prefill phase.
    pub is_prefill: bool,

    /// Sequence data (seq_id -> token positions).
    pub seq_data: std::collections::HashMap<SequenceId, SequenceData>,

    /// Sampling parameters.
    pub sampling_params: Arc<SamplingParams>,

    /// Block tables (seq_id -> block IDs).
    pub block_tables: std::collections::HashMap<SequenceId, Vec<u32>>,

    /// LoRA request.
    pub lora_request: Option<LoraRequest>,

    /// Computed block numbers.
    pub computed_block_nums: Option<Vec<u32>>,

    /// Token chunk size for chunked prefill.
    pub token_chunk_size: Option<usize>,

    /// Multi-modal data.
    pub multi_modal_data: Option<serde_json::Value>,
}

/// Data about a specific sequence for scheduling.
#[derive(Debug, Clone)]
pub struct SequenceData {
    /// Prompt token IDs.
    pub prompt_token_ids: Vec<TokenId>,

    /// Output token IDs.
    pub output_token_ids: Vec<TokenId>,

    /// Cumulative log probability.
    pub cumulative_logprob: f32,

    /// Number of computed tokens.
    pub num_computed_tokens: usize,
}

impl SequenceData {
    /// Create from a sequence.
    pub fn from_sequence(seq: &Sequence) -> Self {
        Self {
            prompt_token_ids: seq.prompt_token_ids.clone(),
            output_token_ids: seq.output_token_ids.clone(),
            cumulative_logprob: seq.cumulative_logprob,
            num_computed_tokens: seq.num_computed_tokens,
        }
    }

    /// Get total length.
    pub fn len(&self) -> usize {
        self.prompt_token_ids.len() + self.output_token_ids.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.prompt_token_ids.is_empty() && self.output_token_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_creation() {
        let params = Arc::new(SamplingParams::default());
        let seq = Sequence::new(
            RequestId::new(),
            vec![TokenId(1), TokenId(2), TokenId(3)],
            params,
        );

        assert_eq!(seq.prompt_len(), 3);
        assert_eq!(seq.output_len(), 0);
        assert_eq!(seq.len(), 3);
        assert!(seq.status.is_waiting());
    }

    #[test]
    fn test_sequence_token_generation() {
        let params = Arc::new(SamplingParams::default().max_tokens(10));
        let mut seq = Sequence::new(
            RequestId::new(),
            vec![TokenId(1)],
            params,
        );

        seq.append_token(TokenId(100), -0.5);
        seq.append_token(TokenId(101), -0.3);

        assert_eq!(seq.output_len(), 2);
        assert_eq!(seq.len(), 3);
        assert_eq!(seq.last_token_id(), Some(TokenId(101)));
        assert!(seq.first_token_at.is_some());
    }

    #[test]
    fn test_sequence_status() {
        let params = Arc::new(SamplingParams::default());
        let mut seq = Sequence::new(RequestId::new(), vec![TokenId(1)], params);

        assert!(seq.status.is_waiting());
        seq.set_running();
        assert!(seq.status.is_running());
        seq.finish(FinishReason::Stop);
        assert!(seq.status.is_finished());
        assert_eq!(seq.status.finish_reason(), Some(FinishReason::Stop));
    }

    #[test]
    fn test_sequence_group() {
        let params = Arc::new(SamplingParams::default());
        let seq = Sequence::new(RequestId::new(), vec![TokenId(1)], params.clone());
        let mut group = SequenceGroup::from_sequence(seq);

        assert_eq!(group.num_seqs(), 1);
        assert!(!group.is_finished());
        assert_eq!(group.prompt_len(), 1);

        // Finish the sequence
        group.first_mut().unwrap().finish(FinishReason::Stop);
        assert!(group.is_finished());
    }

    #[test]
    fn test_sequence_fork() {
        let params = Arc::new(SamplingParams::default());
        let mut seq = Sequence::new(RequestId::new(), vec![TokenId(1)], params);
        seq.append_token(TokenId(100), -0.5);

        let forked = seq.fork(None);
        assert_ne!(forked.seq_id, seq.seq_id);
        assert_eq!(forked.parent_seq_id, Some(seq.seq_id));
        assert_eq!(forked.output_token_ids, seq.output_token_ids);
    }

    #[test]
    fn test_unique_sequence_ids() {
        let params = Arc::new(SamplingParams::default());
        let seq1 = Sequence::new(RequestId::new(), vec![], params.clone());
        let seq2 = Sequence::new(RequestId::new(), vec![], params);
        assert_ne!(seq1.seq_id, seq2.seq_id);
    }
}
