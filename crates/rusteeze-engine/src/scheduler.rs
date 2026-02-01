//! Scheduler for continuous batching.
//!
//! The scheduler manages the lifecycle of requests, from queuing
//! through execution to completion. It implements continuous batching
//! with preemption support.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use tracing::{debug, info, warn};

use rusteeze_core::SamplingParams;

use crate::block_manager::{AllocationResult, BlockManager, BlockManagerConfig};
use crate::sequence::{GroupId, SequenceGroup, SequenceId, SequenceStatus};

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences per batch.
    pub max_num_seqs: usize,

    /// Maximum number of tokens per batch.
    pub max_num_batched_tokens: usize,

    /// Maximum model length.
    pub max_model_len: usize,

    /// Delay factor for prioritization.
    pub delay_factor: f32,

    /// Enable chunked prefill.
    pub enable_chunked_prefill: bool,

    /// Chunk size for prefill.
    pub prefill_chunk_size: usize,

    /// Policy for sequence selection.
    pub policy: SchedulingPolicy,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            max_model_len: 8192,
            delay_factor: 0.0,
            enable_chunked_prefill: false,
            prefill_chunk_size: 512,
            policy: SchedulingPolicy::Fcfs,
        }
    }
}

/// Scheduling policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First come, first served.
    Fcfs,
    /// Priority-based (lower priority value = higher priority).
    Priority,
    /// Shortest job first (by prompt length).
    Sjf,
}

/// Scheduler output for a single step.
#[derive(Debug)]
pub struct SchedulerOutput {
    /// Scheduled sequence groups.
    pub scheduled_groups: Vec<ScheduledGroup>,

    /// Number of prefill tokens.
    pub num_prefill_tokens: usize,

    /// Number of decode tokens.
    pub num_decode_tokens: usize,

    /// Blocks to swap in.
    pub blocks_to_swap_in: Vec<(usize, usize)>,

    /// Blocks to swap out.
    pub blocks_to_swap_out: Vec<(usize, usize)>,

    /// Blocks to copy.
    pub blocks_to_copy: Vec<(usize, usize)>,

    /// Preempted group IDs.
    pub preempted: Vec<GroupId>,

    /// Ignored group IDs (no capacity).
    pub ignored: Vec<GroupId>,
}

impl SchedulerOutput {
    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.scheduled_groups.is_empty()
    }

    /// Get total tokens in batch.
    pub fn num_tokens(&self) -> usize {
        self.num_prefill_tokens + self.num_decode_tokens
    }
}

/// Scheduled sequence group info.
#[derive(Debug)]
pub struct ScheduledGroup {
    /// Group ID.
    pub group_id: GroupId,

    /// Sequence IDs in batch.
    pub seq_ids: Vec<SequenceId>,

    /// Is this prefill phase.
    pub is_prefill: bool,

    /// Token budget for this group.
    pub token_budget: usize,
}

/// Scheduler for managing inference batches.
pub struct Scheduler {
    /// Configuration.
    config: SchedulerConfig,

    /// Block manager.
    block_manager: BlockManager,

    /// Waiting queue.
    waiting: VecDeque<SequenceGroup>,

    /// Running groups.
    running: HashMap<GroupId, SequenceGroup>,

    /// Swapped groups.
    swapped: HashMap<GroupId, SequenceGroup>,

    /// Finished groups (for collection).
    finished: Vec<SequenceGroup>,
}

impl Scheduler {
    /// Create new scheduler.
    pub fn new(config: SchedulerConfig, block_config: BlockManagerConfig) -> Self {
        let block_manager = BlockManager::new(block_config);

        info!(
            "Initialized scheduler: max_seqs={}, max_tokens={}",
            config.max_num_seqs, config.max_num_batched_tokens
        );

        Self {
            config,
            block_manager,
            waiting: VecDeque::new(),
            running: HashMap::new(),
            swapped: HashMap::new(),
            finished: Vec::new(),
        }
    }

    /// Add new request to scheduler.
    pub fn add_request(
        &mut self,
        request_id: String,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        max_tokens: usize,
    ) -> GroupId {
        let seq_group = SequenceGroup::new(
            request_id,
            prompt_tokens,
            sampling_params,
            max_tokens,
        );

        let group_id = seq_group.id;
        self.waiting.push_back(seq_group);

        debug!("Added request {} to waiting queue", group_id);
        group_id
    }

    /// Schedule next batch.
    pub fn schedule(&mut self) -> SchedulerOutput {
        let mut output = SchedulerOutput {
            scheduled_groups: Vec::new(),
            num_prefill_tokens: 0,
            num_decode_tokens: 0,
            blocks_to_swap_in: Vec::new(),
            blocks_to_swap_out: Vec::new(),
            blocks_to_copy: Vec::new(),
            preempted: Vec::new(),
            ignored: Vec::new(),
        };

        // First, try to schedule running sequences (decode phase)
        self.schedule_running(&mut output);

        // Then, try to schedule waiting sequences (prefill phase)
        self.schedule_waiting(&mut output);

        // Try to swap in if we have capacity
        self.schedule_swapped(&mut output);

        output
    }

    /// Schedule running sequences for decode.
    fn schedule_running(&mut self, output: &mut SchedulerOutput) {
        let mut running_to_remove = Vec::new();
        let mut preempt_candidates = Vec::new();

        // Check each running sequence
        for (group_id, group) in &self.running {
            let unfinished = group.get_unfinished_sequences();
            if unfinished.is_empty() {
                running_to_remove.push(*group_id);
                continue;
            }

            // Check if we can append slots for new tokens
            let mut can_continue = true;
            for seq in &unfinished {
                if !self.block_manager.can_append_slot(&seq.id, seq.total_len()) {
                    can_continue = false;
                    break;
                }
            }

            if can_continue {
                let seq_ids: Vec<_> = unfinished.iter().map(|s| s.id).collect();
                let num_tokens = seq_ids.len(); // 1 token per sequence in decode

                output.scheduled_groups.push(ScheduledGroup {
                    group_id: *group_id,
                    seq_ids,
                    is_prefill: false,
                    token_budget: num_tokens,
                });
                output.num_decode_tokens += num_tokens;
            } else {
                preempt_candidates.push(*group_id);
            }
        }

        // Remove finished groups
        for group_id in running_to_remove {
            if let Some(group) = self.running.remove(&group_id) {
                self.finished.push(group);
            }
        }

        // Handle preemption
        for group_id in preempt_candidates {
            if let Some(group) = self.running.remove(&group_id) {
                // Try to swap out
                let seq_ids = group.sequence_ids();
                let mut swapped = true;
                for seq_id in &seq_ids {
                    if !self.block_manager.swap_out(seq_id) {
                        swapped = false;
                        break;
                    }
                }

                if swapped {
                    self.swapped.insert(group_id, group);
                    debug!("Swapped out group {}", group_id);
                } else {
                    // Requeue
                    self.waiting.push_front(group);
                    output.preempted.push(group_id);
                    debug!("Preempted group {}", group_id);
                }
            }
        }
    }

    /// Schedule waiting sequences for prefill.
    fn schedule_waiting(&mut self, output: &mut SchedulerOutput) {
        let mut budget = self.config.max_num_batched_tokens.saturating_sub(output.num_tokens());
        let mut num_seqs = output.scheduled_groups.len();

        while !self.waiting.is_empty() && num_seqs < self.config.max_num_seqs && budget > 0 {
            let group = match self.waiting.front() {
                Some(g) => g,
                None => break,
            };

            // Check prompt length
            let prompt_len = group.prompt_len();
            if prompt_len > self.config.max_model_len {
                let group = self.waiting.pop_front().unwrap();
                output.ignored.push(group.id);
                warn!("Ignored request {} - prompt too long", group.request_id);
                continue;
            }

            // Check allocation
            match self.block_manager.can_allocate(group) {
                AllocationResult::Ok => {}
                AllocationResult::NeedPreemption => {
                    // Need to preempt running sequences
                    if !self.preempt_for_waiting(output) {
                        break;
                    }
                }
                AllocationResult::NoBlocks => {
                    break;
                }
            }

            // Calculate token budget
            let tokens_needed = if self.config.enable_chunked_prefill {
                prompt_len.min(self.config.prefill_chunk_size).min(budget)
            } else {
                prompt_len
            };

            if tokens_needed > budget {
                break;
            }

            // Pop from waiting and allocate
            let mut group = self.waiting.pop_front().unwrap();
            if !self.block_manager.allocate(&group) {
                output.ignored.push(group.id);
                continue;
            }

            // Update sequences to running
            for seq in group.sequences_mut() {
                seq.set_status(SequenceStatus::Running);
            }

            let group_id = group.id;
            let seq_ids = group.sequence_ids();

            output.scheduled_groups.push(ScheduledGroup {
                group_id,
                seq_ids,
                is_prefill: true,
                token_budget: tokens_needed,
            });
            output.num_prefill_tokens += tokens_needed;

            self.running.insert(group_id, group);
            budget -= tokens_needed;
            num_seqs += 1;
        }
    }

    /// Schedule swapped sequences.
    fn schedule_swapped(&mut self, output: &mut SchedulerOutput) {
        let mut budget = self.config.max_num_batched_tokens.saturating_sub(output.num_tokens());
        let mut to_swap_in = Vec::new();

        for (group_id, group) in &self.swapped {
            if budget == 0 {
                break;
            }

            // Check if we can swap in
            let seq_ids = group.sequence_ids();
            let mut can_swap = true;
            for seq_id in &seq_ids {
                if self.block_manager.num_free_gpu_blocks() == 0 {
                    can_swap = false;
                    break;
                }
            }

            if can_swap {
                to_swap_in.push(*group_id);
                budget -= group.num_sequences();
            }
        }

        // Perform swap-in
        for group_id in to_swap_in {
            if let Some(mut group) = self.swapped.remove(&group_id) {
                let seq_ids = group.sequence_ids();
                let mut swapped_in = true;

                for seq_id in &seq_ids {
                    if !self.block_manager.swap_in(seq_id) {
                        swapped_in = false;
                        break;
                    }
                }

                if swapped_in {
                    for seq in group.sequences_mut() {
                        seq.set_status(SequenceStatus::Running);
                    }

                    output.scheduled_groups.push(ScheduledGroup {
                        group_id,
                        seq_ids: group.sequence_ids(),
                        is_prefill: false,
                        token_budget: group.num_sequences(),
                    });
                    output.num_decode_tokens += group.num_sequences();

                    self.running.insert(group_id, group);
                    debug!("Swapped in group {}", group_id);
                } else {
                    self.swapped.insert(group_id, group);
                }
            }
        }
    }

    /// Preempt running sequences to make room for waiting.
    fn preempt_for_waiting(&mut self, output: &mut SchedulerOutput) -> bool {
        // Find lowest priority running sequence to preempt
        let victim = self.running.keys().next().copied();

        if let Some(group_id) = victim {
            if let Some(group) = self.running.remove(&group_id) {
                let seq_ids = group.sequence_ids();

                // Free blocks
                for seq_id in &seq_ids {
                    self.block_manager.free_sequence(seq_id);
                }

                // Requeue
                self.waiting.push_front(group);
                output.preempted.push(group_id);

                return true;
            }
        }

        false
    }

    /// Update scheduler with step results.
    pub fn update(&mut self, group_id: GroupId, finished_seqs: Vec<SequenceId>) {
        if let Some(group) = self.running.get_mut(&group_id) {
            for seq_id in finished_seqs {
                self.block_manager.free_sequence(&seq_id);
            }

            if group.is_finished() {
                if let Some(group) = self.running.remove(&group_id) {
                    self.finished.push(group);
                }
            }
        }
    }

    /// Get and clear finished groups.
    pub fn get_finished(&mut self) -> Vec<SequenceGroup> {
        std::mem::take(&mut self.finished)
    }

    /// Get running group.
    pub fn get_running(&self, group_id: &GroupId) -> Option<&SequenceGroup> {
        self.running.get(group_id)
    }

    /// Get mutable running group.
    pub fn get_running_mut(&mut self, group_id: &GroupId) -> Option<&mut SequenceGroup> {
        self.running.get_mut(group_id)
    }

    /// Abort request.
    pub fn abort(&mut self, request_id: &str) -> bool {
        // Check waiting queue
        if let Some(pos) = self.waiting.iter().position(|g| g.request_id == request_id) {
            self.waiting.remove(pos);
            return true;
        }

        // Check running
        let group_id = self.running
            .iter()
            .find(|(_, g)| g.request_id == request_id)
            .map(|(id, _)| *id);

        if let Some(group_id) = group_id {
            if let Some(group) = self.running.remove(&group_id) {
                for seq_id in group.sequence_ids() {
                    self.block_manager.free_sequence(&seq_id);
                }
                return true;
            }
        }

        // Check swapped
        let group_id = self.swapped
            .iter()
            .find(|(_, g)| g.request_id == request_id)
            .map(|(id, _)| *id);

        if let Some(group_id) = group_id {
            self.swapped.remove(&group_id);
            return true;
        }

        false
    }

    /// Get number of waiting requests.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Get number of running requests.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Get number of swapped requests.
    pub fn num_swapped(&self) -> usize {
        self.swapped.len()
    }

    /// Check if scheduler has pending work.
    pub fn has_pending(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty() || !self.swapped.is_empty()
    }

    /// Get block manager reference.
    pub fn block_manager(&self) -> &BlockManager {
        &self.block_manager
    }

    /// Get mutable block manager reference.
    pub fn block_manager_mut(&mut self) -> &mut BlockManager {
        &mut self.block_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_add_request() {
        let config = SchedulerConfig::default();
        let block_config = BlockManagerConfig::default();
        let mut scheduler = Scheduler::new(config, block_config);

        let group_id = scheduler.add_request(
            "test".to_string(),
            vec![1, 2, 3],
            SamplingParams::default(),
            100,
        );

        assert_eq!(scheduler.num_waiting(), 1);
        assert_eq!(scheduler.num_running(), 0);
    }

    #[test]
    fn test_scheduler_schedule() {
        let config = SchedulerConfig::default();
        let block_config = BlockManagerConfig::default();
        let mut scheduler = Scheduler::new(config, block_config);

        scheduler.add_request(
            "test".to_string(),
            vec![1, 2, 3],
            SamplingParams::default(),
            100,
        );

        let output = scheduler.schedule();
        assert_eq!(output.scheduled_groups.len(), 1);
        assert!(output.scheduled_groups[0].is_prefill);
        assert_eq!(scheduler.num_waiting(), 0);
        assert_eq!(scheduler.num_running(), 1);
    }
}
