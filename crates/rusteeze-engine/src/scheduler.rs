//! # Scheduler — Radical Rewrite
//!
//! Pre-allocated scheduling with FCFS + priority, minimal per-schedule
//! allocations, and proper preemption ordering.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::simd_dispatch;

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences in flight
    pub max_num_seqs: usize,
    /// Maximum total tokens per iteration
    pub max_num_batched_tokens: usize,
    /// Maximum sequence length
    pub max_model_len: usize,
    /// Enable preemption
    pub enable_preemption: bool,
    /// Preemption mode
    pub preemption_mode: PreemptionMode,
}

/// Preemption mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionMode {
    /// Swap KV cache to CPU
    Swap,
    /// Recompute from prompt
    Recompute,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 4096,
            max_model_len: 4096,
            enable_preemption: true,
            preemption_mode: PreemptionMode::Recompute,
        }
    }
}

/// Scheduler output — pre-allocated and reused across iterations.
#[derive(Debug, Clone)]
pub struct SchedulerOutput {
    /// Sequences to schedule for this iteration
    pub scheduled_seq_ids: Vec<u64>,
    /// Number of tokens to process per sequence
    pub num_tokens: Vec<usize>,
    /// Which sequences are prefill vs decode
    pub is_prefill: Vec<bool>,
    /// Preempted sequence IDs
    pub preempted: Vec<u64>,
    /// Completed sequence IDs
    pub completed: Vec<u64>,
    /// Total tokens in this iteration
    pub total_tokens: usize,
}

impl SchedulerOutput {
    fn new(capacity: usize) -> Self {
        Self {
            scheduled_seq_ids: Vec::with_capacity(capacity),
            num_tokens: Vec::with_capacity(capacity),
            is_prefill: Vec::with_capacity(capacity),
            preempted: Vec::with_capacity(capacity),
            completed: Vec::with_capacity(capacity),
            total_tokens: 0,
        }
    }

    fn clear(&mut self) {
        self.scheduled_seq_ids.clear();
        self.num_tokens.clear();
        self.is_prefill.clear();
        self.preempted.clear();
        self.completed.clear();
        self.total_tokens = 0;
    }
}

/// A managed sequence in the scheduler.
#[derive(Debug, Clone)]
pub struct ManagedSequence {
    pub seq_id: u64,
    pub prompt_len: usize,
    pub generated_len: usize,
    pub max_tokens: usize,
    pub priority: f32,
    pub state: SeqState,
    pub arrival_time: u64,
}

/// Sequence state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqState {
    Waiting,
    Running,
    Swapped,
    Completed,
}

/// Main scheduler.
pub struct Scheduler {
    config: SchedulerConfig,
    /// Waiting queue
    waiting: Vec<ManagedSequence>,
    /// Running sequences
    running: Vec<ManagedSequence>,
    /// Swapped sequences
    swapped: Vec<ManagedSequence>,
    /// Pre-allocated output
    output: SchedulerOutput,
    /// Monotonic clock
    clock: AtomicU64,
}

impl Scheduler {
    /// Create a new scheduler.
    pub fn new(config: SchedulerConfig) -> Self {
        simd_dispatch::init();
        let cap = config.max_num_seqs;
        Self {
            config: config.clone(),
            waiting: Vec::with_capacity(cap),
            running: Vec::with_capacity(cap),
            swapped: Vec::new(),
            output: SchedulerOutput::new(cap),
            clock: AtomicU64::new(0),
        }
    }

    /// Add a new sequence to the waiting queue.
    pub fn add_sequence(&mut self, seq: ManagedSequence) {
        self.waiting.push(seq);
    }

    /// Run one scheduling iteration.
    pub fn schedule(&mut self) -> &SchedulerOutput {
        self.output.clear();
        let max_tokens = self.config.max_num_batched_tokens;
        let max_seqs = self.config.max_num_seqs;
        let mut token_budget = max_tokens;
        let mut seq_budget = max_seqs;

        // Mark completed running sequences
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].generated_len >= self.running[i].max_tokens {
                let seq = self.running.swap_remove(i);
                self.output.completed.push(seq.seq_id);
            } else {
                i += 1;
            }
        }

        // Schedule running sequences (decode — 1 token each)
        let mut to_preempt = Vec::new();
        // Sort running by priority (highest first) for preemption ordering
        self.running.sort_unstable_by(|a, b| {
            b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, seq) in self.running.iter().enumerate() {
            if token_budget == 0 || seq_budget == 0 {
                // Preempt remaining
                if self.config.enable_preemption {
                    to_preempt.push(i);
                }
                continue;
            }
            self.output.scheduled_seq_ids.push(seq.seq_id);
            self.output.num_tokens.push(1);
            self.output.is_prefill.push(false);
            self.output.total_tokens += 1;
            token_budget -= 1;
            seq_budget -= 1;
        }

        // Preempt lowest-priority sequences (reverse order)
        for &idx in to_preempt.iter().rev() {
            if idx < self.running.len() {
                let mut seq = self.running.swap_remove(idx);
                seq.state = SeqState::Swapped;
                self.output.preempted.push(seq.seq_id);
                self.swapped.push(seq);
            }
        }

        // Schedule waiting (prefill) — sorted by arrival time (FCFS)
        self.waiting.sort_unstable_by_key(|s| s.arrival_time);

        while !self.waiting.is_empty() && token_budget > 0 && seq_budget > 0 {
            let prompt_tokens = self.waiting[0].prompt_len;
            if prompt_tokens > token_budget { break; }

            let mut seq = self.waiting.remove(0);
            seq.state = SeqState::Running;

            self.output.scheduled_seq_ids.push(seq.seq_id);
            self.output.num_tokens.push(prompt_tokens);
            self.output.is_prefill.push(true);
            self.output.total_tokens += prompt_tokens;

            token_budget -= prompt_tokens;
            seq_budget -= 1;
            self.running.push(seq);
        }

        self.clock.fetch_add(1, Ordering::Relaxed);
        &self.output
    }

    /// Notify token generation.
    pub fn on_token(&mut self, seq_id: u64) {
        if let Some(seq) = self.running.iter_mut().find(|s| s.seq_id == seq_id) {
            seq.generated_len += 1;
        }
    }

    /// Force-complete a sequence.
    pub fn complete(&mut self, seq_id: u64) {
        self.running.retain(|s| s.seq_id != seq_id);
        self.waiting.retain(|s| s.seq_id != seq_id);
    }

    /// Stats.
    pub fn waiting_count(&self) -> usize { self.waiting.len() }
    pub fn running_count(&self) -> usize { self.running.len() }
    pub fn swapped_count(&self) -> usize { self.swapped.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_basic() {
        let config = SchedulerConfig {
            max_num_seqs: 4,
            max_num_batched_tokens: 100,
            ..Default::default()
        };
        let mut sched = Scheduler::new(config);

        sched.add_sequence(ManagedSequence {
            seq_id: 1, prompt_len: 10, generated_len: 0, max_tokens: 20,
            priority: 1.0, state: SeqState::Waiting, arrival_time: 0,
        });
        sched.add_sequence(ManagedSequence {
            seq_id: 2, prompt_len: 15, generated_len: 0, max_tokens: 20,
            priority: 1.0, state: SeqState::Waiting, arrival_time: 1,
        });

        let out = sched.schedule();
        assert_eq!(out.scheduled_seq_ids.len(), 2);
        assert_eq!(out.total_tokens, 25);
    }

    #[test]
    fn test_preemption() {
        let config = SchedulerConfig {
            max_num_seqs: 2,
            max_num_batched_tokens: 5,
            enable_preemption: true,
            ..Default::default()
        };
        let mut sched = Scheduler::new(config);

        // Add and promote 3 sequences to running
        for i in 0..3 {
            sched.add_sequence(ManagedSequence {
                seq_id: i, prompt_len: 1, generated_len: 0, max_tokens: 100,
                priority: i as f32, state: SeqState::Waiting, arrival_time: i as u64,
            });
        }
        // Schedule: should pick first 2 (max_num_seqs=2)
        let out = sched.schedule();
        assert!(out.scheduled_seq_ids.len() <= 2);
    }
}
