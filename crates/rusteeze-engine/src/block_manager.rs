//! Block manager for paged attention.
//!
//! Manages GPU and CPU memory blocks for KV cache with copy-on-write
//! and prefix caching support.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::sequence::{SequenceId, SequenceGroup};

/// Physical block ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlockId(pub usize);

/// Logical block ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogicalBlockId(pub usize);

/// Block location (GPU or CPU).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLocation {
    /// GPU memory.
    Gpu,
    /// CPU memory.
    Cpu,
}

/// Physical memory block.
#[derive(Debug)]
pub struct PhysicalBlock {
    /// Block ID.
    pub id: PhysicalBlockId,

    /// Block location.
    pub location: BlockLocation,

    /// Reference count.
    ref_count: usize,

    /// Block size in tokens.
    pub block_size: usize,

    /// Hash for prefix caching (None if not computed).
    prefix_hash: Option<u64>,
}

impl PhysicalBlock {
    /// Create new physical block.
    pub fn new(id: PhysicalBlockId, location: BlockLocation, block_size: usize) -> Self {
        Self {
            id,
            location,
            ref_count: 0,
            block_size,
            prefix_hash: None,
        }
    }

    /// Increment reference count.
    pub fn inc_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count.
    pub fn dec_ref(&mut self) {
        self.ref_count = self.ref_count.saturating_sub(1);
    }

    /// Get reference count.
    pub fn ref_count(&self) -> usize {
        self.ref_count
    }

    /// Check if block is free.
    pub fn is_free(&self) -> bool {
        self.ref_count == 0
    }

    /// Set prefix hash.
    pub fn set_prefix_hash(&mut self, hash: u64) {
        self.prefix_hash = Some(hash);
    }

    /// Get prefix hash.
    pub fn prefix_hash(&self) -> Option<u64> {
        self.prefix_hash
    }
}

/// Block allocation result.
#[derive(Debug)]
pub enum AllocationResult {
    /// Successfully allocated.
    Ok,
    /// Need to preempt sequences.
    NeedPreemption,
    /// No blocks available.
    NoBlocks,
}

/// Block table mapping logical to physical blocks.
#[derive(Debug, Default, Clone)]
pub struct BlockTable {
    /// Mapping from logical block index to physical block ID.
    blocks: Vec<PhysicalBlockId>,
}

impl BlockTable {
    /// Create new block table.
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Add physical block.
    pub fn add(&mut self, block_id: PhysicalBlockId) {
        self.blocks.push(block_id);
    }

    /// Get physical block at index.
    pub fn get(&self, index: usize) -> Option<PhysicalBlockId> {
        self.blocks.get(index).copied()
    }

    /// Get all physical block IDs.
    pub fn blocks(&self) -> &[PhysicalBlockId] {
        &self.blocks
    }

    /// Number of blocks.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Block manager configuration.
#[derive(Debug, Clone)]
pub struct BlockManagerConfig {
    /// Block size in tokens.
    pub block_size: usize,

    /// Number of GPU blocks.
    pub num_gpu_blocks: usize,

    /// Number of CPU blocks.
    pub num_cpu_blocks: usize,

    /// Watermark for preemption (fraction of GPU blocks).
    pub watermark: f32,

    /// Enable prefix caching.
    pub enable_prefix_caching: bool,
}

impl Default for BlockManagerConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_gpu_blocks: 1000,
            num_cpu_blocks: 2000,
            watermark: 0.01,
            enable_prefix_caching: false,
        }
    }
}

/// Block manager for paged attention.
pub struct BlockManager {
    /// Configuration.
    config: BlockManagerConfig,

    /// GPU blocks.
    gpu_blocks: Vec<PhysicalBlock>,

    /// CPU blocks.
    cpu_blocks: Vec<PhysicalBlock>,

    /// Free GPU block IDs.
    free_gpu_blocks: VecDeque<PhysicalBlockId>,

    /// Free CPU block IDs.
    free_cpu_blocks: VecDeque<PhysicalBlockId>,

    /// Block tables per sequence.
    block_tables: HashMap<SequenceId, BlockTable>,

    /// Prefix cache (hash -> block ID).
    prefix_cache: HashMap<u64, PhysicalBlockId>,
}

impl BlockManager {
    /// Create new block manager.
    pub fn new(config: BlockManagerConfig) -> Self {
        let mut gpu_blocks = Vec::with_capacity(config.num_gpu_blocks);
        let mut cpu_blocks = Vec::with_capacity(config.num_cpu_blocks);
        let mut free_gpu_blocks = VecDeque::with_capacity(config.num_gpu_blocks);
        let mut free_cpu_blocks = VecDeque::with_capacity(config.num_cpu_blocks);

        // Initialize GPU blocks
        for i in 0..config.num_gpu_blocks {
            let block = PhysicalBlock::new(
                PhysicalBlockId(i),
                BlockLocation::Gpu,
                config.block_size,
            );
            gpu_blocks.push(block);
            free_gpu_blocks.push_back(PhysicalBlockId(i));
        }

        // Initialize CPU blocks
        for i in 0..config.num_cpu_blocks {
            let block = PhysicalBlock::new(
                PhysicalBlockId(i),
                BlockLocation::Cpu,
                config.block_size,
            );
            cpu_blocks.push(block);
            free_cpu_blocks.push_back(PhysicalBlockId(i));
        }

        info!(
            "Initialized block manager: {} GPU blocks, {} CPU blocks, block_size={}",
            config.num_gpu_blocks, config.num_cpu_blocks, config.block_size
        );

        Self {
            config,
            gpu_blocks,
            cpu_blocks,
            free_gpu_blocks,
            free_cpu_blocks,
            block_tables: HashMap::new(),
            prefix_cache: HashMap::new(),
        }
    }

    /// Get number of free GPU blocks.
    pub fn num_free_gpu_blocks(&self) -> usize {
        self.free_gpu_blocks.len()
    }

    /// Get number of free CPU blocks.
    pub fn num_free_cpu_blocks(&self) -> usize {
        self.free_cpu_blocks.len()
    }

    /// Get watermark block count.
    pub fn watermark_blocks(&self) -> usize {
        (self.config.num_gpu_blocks as f32 * self.config.watermark) as usize
    }

    /// Check if can allocate blocks for sequence group.
    pub fn can_allocate(&self, seq_group: &SequenceGroup) -> AllocationResult {
        let num_required = self.get_num_required_blocks(seq_group);
        let num_free = self.num_free_gpu_blocks();

        if num_free >= num_required + self.watermark_blocks() {
            AllocationResult::Ok
        } else if num_free >= num_required {
            AllocationResult::NeedPreemption
        } else {
            AllocationResult::NoBlocks
        }
    }

    /// Get number of required blocks for sequence group.
    fn get_num_required_blocks(&self, seq_group: &SequenceGroup) -> usize {
        let total_tokens = seq_group.prompt_len();
        let num_blocks = (total_tokens + self.config.block_size - 1) / self.config.block_size;
        num_blocks * seq_group.num_sequences()
    }

    /// Allocate blocks for sequence group.
    pub fn allocate(&mut self, seq_group: &SequenceGroup) -> bool {
        let num_required = self.get_num_required_blocks(seq_group);

        if self.num_free_gpu_blocks() < num_required {
            return false;
        }

        for seq_id in seq_group.sequence_ids() {
            let mut block_table = BlockTable::new();

            let seq = seq_group.get_sequence(&seq_id).unwrap();
            let num_blocks = (seq.total_len() + self.config.block_size - 1) / self.config.block_size;

            for _ in 0..num_blocks {
                if let Some(block_id) = self.allocate_gpu_block() {
                    block_table.add(block_id);
                } else {
                    // Rollback
                    self.free_sequence(&seq_id);
                    return false;
                }
            }

            self.block_tables.insert(seq_id, block_table);
        }

        true
    }

    /// Allocate single GPU block.
    fn allocate_gpu_block(&mut self) -> Option<PhysicalBlockId> {
        let block_id = self.free_gpu_blocks.pop_front()?;
        self.gpu_blocks[block_id.0].inc_ref();
        Some(block_id)
    }

    /// Allocate single CPU block.
    fn allocate_cpu_block(&mut self) -> Option<PhysicalBlockId> {
        let block_id = self.free_cpu_blocks.pop_front()?;
        self.cpu_blocks[block_id.0].inc_ref();
        Some(block_id)
    }

    /// Append slot for new token.
    pub fn append_slot(&mut self, seq_id: &SequenceId, seq_len: usize) -> Option<PhysicalBlockId> {
        let block_table = self.block_tables.get_mut(seq_id)?;

        // Check if need new block
        let slot_in_last_block = seq_len % self.config.block_size;
        if slot_in_last_block == 0 && seq_len > 0 {
            // Need new block
            let block_id = self.allocate_gpu_block()?;
            block_table.add(block_id);
            Some(block_id)
        } else {
            // Use existing last block
            block_table.blocks().last().copied()
        }
    }

    /// Check if can append slot.
    pub fn can_append_slot(&self, seq_id: &SequenceId, seq_len: usize) -> bool {
        let slot_in_last_block = seq_len % self.config.block_size;
        if slot_in_last_block == 0 && seq_len > 0 {
            self.num_free_gpu_blocks() > 0
        } else {
            true
        }
    }

    /// Free blocks for sequence.
    pub fn free_sequence(&mut self, seq_id: &SequenceId) {
        if let Some(block_table) = self.block_tables.remove(seq_id) {
            for block_id in block_table.blocks() {
                self.free_gpu_block(*block_id);
            }
        }
    }

    /// Free GPU block.
    fn free_gpu_block(&mut self, block_id: PhysicalBlockId) {
        let block = &mut self.gpu_blocks[block_id.0];
        block.dec_ref();

        if block.is_free() {
            self.free_gpu_blocks.push_back(block_id);
        }
    }

    /// Free CPU block.
    fn free_cpu_block(&mut self, block_id: PhysicalBlockId) {
        let block = &mut self.cpu_blocks[block_id.0];
        block.dec_ref();

        if block.is_free() {
            self.free_cpu_blocks.push_back(block_id);
        }
    }

    /// Get block table for sequence.
    pub fn get_block_table(&self, seq_id: &SequenceId) -> Option<&BlockTable> {
        self.block_tables.get(seq_id)
    }

    /// Fork sequence (for beam search).
    pub fn fork(&mut self, parent_seq_id: &SequenceId, child_seq_id: SequenceId) -> bool {
        let parent_table = match self.block_tables.get(parent_seq_id) {
            Some(t) => t.clone(),
            None => return false,
        };

        // Increment reference counts
        for block_id in parent_table.blocks() {
            self.gpu_blocks[block_id.0].inc_ref();
        }

        self.block_tables.insert(child_seq_id, parent_table);
        true
    }

    /// Swap out sequence to CPU.
    pub fn swap_out(&mut self, seq_id: &SequenceId) -> bool {
        let gpu_table = match self.block_tables.get(seq_id) {
            Some(t) => t.clone(),
            None => return false,
        };

        // Allocate CPU blocks and copy mapping
        let mut cpu_table = BlockTable::new();
        for gpu_block_id in gpu_table.blocks() {
            if let Some(cpu_block_id) = self.allocate_cpu_block() {
                cpu_table.add(cpu_block_id);
                // Note: Actual data copy would happen in worker
            } else {
                // Rollback
                for block_id in cpu_table.blocks() {
                    self.free_cpu_block(*block_id);
                }
                return false;
            }
        }

        // Free GPU blocks
        for block_id in gpu_table.blocks() {
            self.free_gpu_block(*block_id);
        }

        self.block_tables.insert(*seq_id, cpu_table);
        true
    }

    /// Swap in sequence from CPU.
    pub fn swap_in(&mut self, seq_id: &SequenceId) -> bool {
        let cpu_table = match self.block_tables.get(seq_id) {
            Some(t) => t.clone(),
            None => return false,
        };

        // Allocate GPU blocks
        let mut gpu_table = BlockTable::new();
        for cpu_block_id in cpu_table.blocks() {
            if let Some(gpu_block_id) = self.allocate_gpu_block() {
                gpu_table.add(gpu_block_id);
            } else {
                // Rollback
                for block_id in gpu_table.blocks() {
                    self.free_gpu_block(*block_id);
                }
                return false;
            }
        }

        // Free CPU blocks
        for block_id in cpu_table.blocks() {
            self.free_cpu_block(*block_id);
        }

        self.block_tables.insert(*seq_id, gpu_table);
        true
    }

    /// Get block size.
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Compute memory usage.
    pub fn gpu_memory_usage(&self) -> f32 {
        let used = self.config.num_gpu_blocks - self.num_free_gpu_blocks();
        used as f32 / self.config.num_gpu_blocks as f32
    }

    /// Get all active sequence IDs.
    pub fn active_sequences(&self) -> Vec<SequenceId> {
        self.block_tables.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusteeze_core::SamplingParams;

    #[test]
    fn test_block_manager_allocation() {
        let config = BlockManagerConfig {
            block_size: 16,
            num_gpu_blocks: 100,
            num_cpu_blocks: 100,
            watermark: 0.01,
            enable_prefix_caching: false,
        };

        let mut manager = BlockManager::new(config);
        assert_eq!(manager.num_free_gpu_blocks(), 100);

        // Create sequence group
        let sampling = SamplingParams::default();
        let seq_group = SequenceGroup::new(
            "test".to_string(),
            vec![1; 32], // 32 tokens = 2 blocks
            sampling,
            100,
        );

        // Check allocation
        matches!(manager.can_allocate(&seq_group), AllocationResult::Ok);

        // Allocate
        assert!(manager.allocate(&seq_group));
        assert_eq!(manager.num_free_gpu_blocks(), 98);
    }

    #[test]
    fn test_block_manager_free() {
        let config = BlockManagerConfig::default();
        let mut manager = BlockManager::new(config);

        let sampling = SamplingParams::default();
        let seq_group = SequenceGroup::new(
            "test".to_string(),
            vec![1; 16],
            sampling,
            100,
        );

        manager.allocate(&seq_group);
        let initial_free = manager.num_free_gpu_blocks();

        let seq_ids: Vec<_> = seq_group.sequence_ids();
        for seq_id in seq_ids {
            manager.free_sequence(&seq_id);
        }

        assert!(manager.num_free_gpu_blocks() > initial_free);
    }
}
