//! # Block Manager — Radical Rewrite
//!
//! KV cache block management with SmallVec for block tables,
//! iterator-based operations (no collect), and efficient swap/fork.

use std::collections::HashMap;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU64, Ordering};

/// Physical block ID.
pub type PhysicalBlockId = u32;

/// Block table: maps sequence positions to physical blocks.
/// SmallVec avoids heap allocation for sequences with ≤8 blocks.
pub type BlockTable = SmallVec<[PhysicalBlockId; 8]>;

/// Block manager configuration.
#[derive(Debug, Clone)]
pub struct BlockManagerConfig {
    /// Block size (number of tokens per block)
    pub block_size: usize,
    /// Number of GPU blocks (or CPU SIMD blocks)
    pub num_gpu_blocks: usize,
    /// Number of CPU swap blocks
    pub num_cpu_blocks: usize,
}

impl Default for BlockManagerConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            num_gpu_blocks: 1024,
            num_cpu_blocks: 512,
        }
    }
}

/// Block allocation stats.
#[derive(Debug, Default)]
pub struct BlockStats {
    pub allocations: AtomicU64,
    pub deallocations: AtomicU64,
    pub swaps: AtomicU64,
    pub forks: AtomicU64,
}

impl Clone for BlockStats {
    fn clone(&self) -> Self {
        Self {
            allocations: AtomicU64::new(self.allocations.load(Ordering::Relaxed)),
            deallocations: AtomicU64::new(self.deallocations.load(Ordering::Relaxed)),
            swaps: AtomicU64::new(self.swaps.load(Ordering::Relaxed)),
            forks: AtomicU64::new(self.forks.load(Ordering::Relaxed)),
        }
    }
}

/// Free block pool (LIFO stack for locality).
struct FreePool {
    blocks: Vec<PhysicalBlockId>,
}

impl FreePool {
    fn new(num_blocks: usize) -> Self {
        Self {
            blocks: (0..num_blocks as u32).rev().collect(),
        }
    }

    fn allocate(&mut self) -> Option<PhysicalBlockId> {
        self.blocks.pop()
    }

    fn free(&mut self, block: PhysicalBlockId) {
        self.blocks.push(block);
    }

    fn available(&self) -> usize {
        self.blocks.len()
    }
}

/// KV cache block manager.
pub struct BlockManager {
    config: BlockManagerConfig,
    /// GPU block pool
    gpu_pool: FreePool,
    /// CPU block pool
    cpu_pool: FreePool,
    /// Sequence ID → GPU block table
    gpu_tables: HashMap<u64, BlockTable>,
    /// Sequence ID → CPU block table (swapped)
    cpu_tables: HashMap<u64, BlockTable>,
    /// Reference counts for copy-on-write
    ref_counts: HashMap<PhysicalBlockId, u32>,
    /// Stats
    stats: BlockStats,
}

impl BlockManager {
    /// Create a new block manager.
    pub fn new(config: BlockManagerConfig) -> Self {
        let gpu_pool = FreePool::new(config.num_gpu_blocks);
        let cpu_pool = FreePool::new(config.num_cpu_blocks);
        Self {
            config,
            gpu_pool,
            cpu_pool,
            gpu_tables: HashMap::new(),
            cpu_tables: HashMap::new(),
            ref_counts: HashMap::new(),
            stats: BlockStats::default(),
        }
    }

    /// Allocate blocks for a new sequence.
    pub fn allocate(&mut self, seq_id: u64, num_blocks: usize) -> bool {
        if self.gpu_pool.available() < num_blocks { return false; }

        let mut table = BlockTable::new();
        for _ in 0..num_blocks {
            let block = self.gpu_pool.allocate().unwrap();
            self.ref_counts.insert(block, 1);
            table.push(block);
        }

        self.gpu_tables.insert(seq_id, table);
        self.stats.allocations.fetch_add(num_blocks as u64, Ordering::Relaxed);
        true
    }

    /// Append a block to a sequence's block table.
    pub fn append_block(&mut self, seq_id: u64) -> Option<PhysicalBlockId> {
        let block = self.gpu_pool.allocate()?;
        self.ref_counts.insert(block, 1);
        self.gpu_tables.entry(seq_id).or_insert_with(BlockTable::new).push(block);
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);
        Some(block)
    }

    /// Free all blocks for a sequence.
    pub fn free(&mut self, seq_id: u64) {
        if let Some(table) = self.gpu_tables.remove(&seq_id) {
            for &block in &table {
                let rc = self.ref_counts.get_mut(&block).unwrap();
                *rc -= 1;
                if *rc == 0 {
                    self.ref_counts.remove(&block);
                    self.gpu_pool.free(block);
                }
            }
            self.stats.deallocations.fetch_add(table.len() as u64, Ordering::Relaxed);
        }
    }

    /// Swap sequence blocks from GPU to CPU.
    pub fn swap_out(&mut self, seq_id: u64) -> bool {
        let gpu_table = match self.gpu_tables.get(&seq_id) {
            Some(t) => t,
            None => return false,
        };
        let needed = gpu_table.len();
        if self.cpu_pool.available() < needed { return false; }

        let mut cpu_table = BlockTable::new();
        for _ in 0..needed {
            cpu_table.push(self.cpu_pool.allocate().unwrap());
        }

        // Move block table to CPU (actual data copy would happen elsewhere)
        self.cpu_tables.insert(seq_id, cpu_table);

        // Free GPU blocks
        let gpu_table = self.gpu_tables.remove(&seq_id).unwrap();
        for &block in &gpu_table {
            let rc = self.ref_counts.get_mut(&block).unwrap();
            *rc -= 1;
            if *rc == 0 {
                self.ref_counts.remove(&block);
                self.gpu_pool.free(block);
            }
        }

        self.stats.swaps.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Swap sequence blocks from CPU to GPU.
    pub fn swap_in(&mut self, seq_id: u64) -> bool {
        let cpu_table = match self.cpu_tables.get(&seq_id) {
            Some(t) => t,
            None => return false,
        };
        let needed = cpu_table.len();
        if self.gpu_pool.available() < needed { return false; }

        let mut gpu_table = BlockTable::new();
        for _ in 0..needed {
            let block = self.gpu_pool.allocate().unwrap();
            self.ref_counts.insert(block, 1);
            gpu_table.push(block);
        }

        self.gpu_tables.insert(seq_id, gpu_table);

        // Free CPU blocks
        let cpu_table = self.cpu_tables.remove(&seq_id).unwrap();
        for &block in &cpu_table {
            self.cpu_pool.free(block);
        }

        self.stats.swaps.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Fork a sequence (copy-on-write: share blocks, increment refcounts).
    pub fn fork(&mut self, src_seq_id: u64, dst_seq_id: u64) -> bool {
        let table = match self.gpu_tables.get(&src_seq_id) {
            Some(t) => t.clone(), // SmallVec clone is stack-allocated for ≤8 blocks
            None => return false,
        };

        // Increment ref counts (no block copy needed — CoW)
        for &block in &table {
            *self.ref_counts.entry(block).or_insert(0) += 1;
        }

        self.gpu_tables.insert(dst_seq_id, table);
        self.stats.forks.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Get block table for a sequence.
    pub fn get_block_table(&self, seq_id: u64) -> Option<&BlockTable> {
        self.gpu_tables.get(&seq_id)
    }

    /// Number of free GPU blocks.
    pub fn free_gpu_blocks(&self) -> usize { self.gpu_pool.available() }
    /// Number of free CPU blocks.
    pub fn free_cpu_blocks(&self) -> usize { self.cpu_pool.available() }
    /// Stats.
    pub fn stats(&self) -> &BlockStats { &self.stats }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_free() {
        let config = BlockManagerConfig { num_gpu_blocks: 10, num_cpu_blocks: 5, block_size: 16 };
        let mut mgr = BlockManager::new(config);

        assert!(mgr.allocate(1, 3));
        assert_eq!(mgr.free_gpu_blocks(), 7);

        mgr.free(1);
        assert_eq!(mgr.free_gpu_blocks(), 10);
    }

    #[test]
    fn test_swap() {
        let config = BlockManagerConfig { num_gpu_blocks: 10, num_cpu_blocks: 5, block_size: 16 };
        let mut mgr = BlockManager::new(config);

        mgr.allocate(1, 3);
        assert!(mgr.swap_out(1));
        assert_eq!(mgr.free_gpu_blocks(), 10);
        assert_eq!(mgr.free_cpu_blocks(), 2);

        assert!(mgr.swap_in(1));
        assert_eq!(mgr.free_gpu_blocks(), 7);
        assert_eq!(mgr.free_cpu_blocks(), 5);
    }

    #[test]
    fn test_fork_cow() {
        let config = BlockManagerConfig { num_gpu_blocks: 10, num_cpu_blocks: 5, block_size: 16 };
        let mut mgr = BlockManager::new(config);

        mgr.allocate(1, 3);
        assert!(mgr.fork(1, 2));
        // No new GPU blocks allocated (CoW)
        assert_eq!(mgr.free_gpu_blocks(), 7);

        // Free original — blocks still held by fork
        mgr.free(1);
        assert_eq!(mgr.free_gpu_blocks(), 7); // Still 7 because refcount > 0

        // Free fork — blocks fully freed
        mgr.free(2);
        assert_eq!(mgr.free_gpu_blocks(), 10);
    }
}
