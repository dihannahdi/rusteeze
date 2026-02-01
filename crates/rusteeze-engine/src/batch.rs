//! Batch construction for inference.
//!
//! This module handles building input batches from scheduled sequences.

use std::collections::HashMap;

use rusteeze_core::SamplingParams;

use crate::scheduler::{ScheduledGroup, SchedulerOutput};
use crate::sequence::{SequenceGroup, SequenceId};

/// Input batch for model execution.
#[derive(Debug)]
pub struct BatchInput {
    /// Token IDs per sequence.
    pub token_ids: Vec<Vec<u32>>,

    /// Position IDs per sequence.
    pub position_ids: Vec<Vec<u32>>,

    /// Sequence info (ID, sampling params).
    pub seq_info: Vec<(SequenceId, SamplingParams)>,

    /// Block tables per sequence.
    pub block_tables: Vec<Vec<usize>>,

    /// Context lengths per sequence.
    pub context_lens: Vec<usize>,

    /// Is prefill batch.
    pub is_prefill: bool,
}

impl BatchInput {
    /// Create new empty batch.
    pub fn new(is_prefill: bool) -> Self {
        Self {
            token_ids: Vec::new(),
            position_ids: Vec::new(),
            seq_info: Vec::new(),
            block_tables: Vec::new(),
            context_lens: Vec::new(),
            is_prefill,
        }
    }

    /// Add sequence to batch.
    pub fn add_sequence(
        &mut self,
        seq_id: SequenceId,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        sampling_params: SamplingParams,
        block_table: Vec<usize>,
        context_len: usize,
    ) {
        self.token_ids.push(tokens);
        self.position_ids.push(positions);
        self.seq_info.push((seq_id, sampling_params));
        self.block_tables.push(block_table);
        self.context_lens.push(context_len);
    }

    /// Get batch size.
    pub fn batch_size(&self) -> usize {
        self.token_ids.len()
    }

    /// Get total tokens.
    pub fn total_tokens(&self) -> usize {
        self.token_ids.iter().map(|t| t.len()).sum()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }
}

/// Batch builder for constructing batches from scheduled groups.
pub struct BatchBuilder {
    /// Maximum batch size.
    max_batch_size: usize,

    /// Maximum tokens per batch.
    max_tokens: usize,
}

impl BatchBuilder {
    /// Create new batch builder.
    pub fn new(max_batch_size: usize, max_tokens: usize) -> Self {
        Self {
            max_batch_size,
            max_tokens,
        }
    }

    /// Build batch from scheduler output.
    pub fn build(
        &self,
        output: &SchedulerOutput,
        groups: &HashMap<crate::sequence::GroupId, &SequenceGroup>,
        block_tables: &HashMap<SequenceId, Vec<usize>>,
    ) -> (Option<BatchInput>, Option<BatchInput>) {
        let mut prefill_batch = BatchInput::new(true);
        let mut decode_batch = BatchInput::new(false);

        for scheduled in &output.scheduled_groups {
            let group = match groups.get(&scheduled.group_id) {
                Some(g) => *g,
                None => continue,
            };

            for seq_id in &scheduled.seq_ids {
                let seq = match group.get_sequence(seq_id) {
                    Some(s) => s,
                    None => continue,
                };

                let block_table = block_tables
                    .get(seq_id)
                    .cloned()
                    .unwrap_or_default();

                if scheduled.is_prefill {
                    // Prefill: use all prompt tokens
                    let tokens = seq.prompt_tokens().to_vec();
                    let positions: Vec<u32> = (0..tokens.len() as u32).collect();

                    prefill_batch.add_sequence(
                        *seq_id,
                        tokens,
                        positions,
                        group.sampling_params.clone(),
                        block_table,
                        0, // No context yet
                    );
                } else {
                    // Decode: use last token
                    let all_tokens = seq.all_tokens();
                    let last_token = all_tokens.last().copied().unwrap_or(0);
                    let position = (all_tokens.len() - 1) as u32;

                    decode_batch.add_sequence(
                        *seq_id,
                        vec![last_token],
                        vec![position],
                        group.sampling_params.clone(),
                        block_table,
                        all_tokens.len() - 1,
                    );
                }
            }
        }

        let prefill = if prefill_batch.is_empty() {
            None
        } else {
            Some(prefill_batch)
        };

        let decode = if decode_batch.is_empty() {
            None
        } else {
            Some(decode_batch)
        };

        (prefill, decode)
    }
}

/// Metadata for a batch execution.
#[derive(Debug)]
pub struct BatchMetadata {
    /// Number of prefill tokens.
    pub num_prefill_tokens: usize,

    /// Number of decode tokens.
    pub num_decode_tokens: usize,

    /// Number of sequences.
    pub num_seqs: usize,

    /// Maximum sequence length in batch.
    pub max_seq_len: usize,

    /// Slot mappings for PagedAttention.
    pub slot_mapping: Vec<usize>,
}

impl BatchMetadata {
    /// Create from batch input.
    pub fn from_batch(batch: &BatchInput) -> Self {
        let num_tokens = batch.total_tokens();
        let max_seq_len = batch.token_ids.iter().map(|t| t.len()).max().unwrap_or(0);

        Self {
            num_prefill_tokens: if batch.is_prefill { num_tokens } else { 0 },
            num_decode_tokens: if batch.is_prefill { 0 } else { num_tokens },
            num_seqs: batch.batch_size(),
            max_seq_len,
            slot_mapping: Vec::new(), // Populated by block manager
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_input() {
        let mut batch = BatchInput::new(true);
        assert!(batch.is_empty());

        batch.add_sequence(
            crate::sequence::SequenceId::new(),
            vec![1, 2, 3],
            vec![0, 1, 2],
            SamplingParams::default(),
            vec![0],
            0,
        );

        assert_eq!(batch.batch_size(), 1);
        assert_eq!(batch.total_tokens(), 3);
    }
}
