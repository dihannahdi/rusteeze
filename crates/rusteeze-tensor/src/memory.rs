//! Memory management utilities for efficient GPU memory usage.
//!
//! This module provides:
//!
//! - Memory pools for tensor allocation
//! - Memory tracking and profiling
//! - Automatic memory defragmentation
//! - OOM handling strategies

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};
use tracing::{debug, info, warn};

/// Memory pool configuration.
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial pool size in bytes.
    pub initial_size: usize,

    /// Maximum pool size in bytes.
    pub max_size: usize,

    /// Growth factor when expanding.
    pub growth_factor: f64,

    /// Enable memory defragmentation.
    pub enable_defrag: bool,

    /// Minimum allocation size (for alignment).
    pub min_allocation_size: usize,

    /// Memory alignment requirement.
    pub alignment: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024 * 1024, // 1 GB
            max_size: 32 * 1024 * 1024 * 1024, // 32 GB
            growth_factor: 1.5,
            enable_defrag: true,
            min_allocation_size: 256,
            alignment: 256,
        }
    }
}

/// A memory block in the pool.
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Block ID.
    pub id: u64,

    /// Offset in the pool.
    pub offset: usize,

    /// Size in bytes.
    pub size: usize,

    /// Whether the block is in use.
    pub in_use: bool,

    /// Allocation tag (for debugging).
    pub tag: Option<String>,
}

impl MemoryBlock {
    /// Create a new memory block.
    pub fn new(id: u64, offset: usize, size: usize) -> Self {
        Self {
            id,
            offset,
            size,
            in_use: false,
            tag: None,
        }
    }

    /// End offset of this block.
    pub fn end(&self) -> usize {
        self.offset + self.size
    }

    /// Check if this block can fit the requested size.
    pub fn can_fit(&self, size: usize) -> bool {
        !self.in_use && self.size >= size
    }
}

/// Memory allocator using a free list approach.
#[derive(Debug)]
pub struct MemoryAllocator {
    /// Configuration.
    config: MemoryPoolConfig,

    /// All blocks.
    blocks: RwLock<Vec<MemoryBlock>>,

    /// Free blocks sorted by size (size -> block_id).
    free_by_size: Mutex<HashMap<usize, Vec<u64>>>,

    /// Next block ID.
    next_id: AtomicU64,

    /// Total allocated bytes.
    allocated_bytes: AtomicUsize,

    /// Total pool size.
    pool_size: AtomicUsize,

    /// Peak usage.
    peak_usage: AtomicUsize,

    /// Number of allocations.
    num_allocations: AtomicU64,

    /// Number of frees.
    num_frees: AtomicU64,
}

impl MemoryAllocator {
    /// Create a new memory allocator.
    pub fn new(config: MemoryPoolConfig) -> Self {
        let initial_size = config.initial_size;

        // Create initial free block
        let initial_block = MemoryBlock::new(0, 0, initial_size);

        let mut free_by_size = HashMap::new();
        free_by_size.insert(initial_size, vec![0]);

        info!(
            "MemoryAllocator initialized: {} MB initial, {} MB max",
            initial_size / (1024 * 1024),
            config.max_size / (1024 * 1024)
        );

        Self {
            config,
            blocks: RwLock::new(vec![initial_block]),
            free_by_size: Mutex::new(free_by_size),
            next_id: AtomicU64::new(1),
            allocated_bytes: AtomicUsize::new(0),
            pool_size: AtomicUsize::new(initial_size),
            peak_usage: AtomicUsize::new(0),
            num_allocations: AtomicU64::new(0),
            num_frees: AtomicU64::new(0),
        }
    }

    /// Allocate a block of memory.
    pub fn allocate(&self, size: usize, tag: Option<&str>) -> Option<u64> {
        // Align size
        let aligned_size = self.align_size(size);

        // Find a suitable free block
        let block_id = self.find_free_block(aligned_size)?;

        // Split block if necessary
        let actual_id = {
            let mut blocks = self.blocks.write();
            let block_idx = blocks.iter().position(|b| b.id == block_id)?;
            let block = &mut blocks[block_idx];

            if block.size > aligned_size + self.config.min_allocation_size {
                // Split the block
                let remaining_size = block.size - aligned_size;
                let remaining_offset = block.offset + aligned_size;
                let remaining_id = self.next_id.fetch_add(1, Ordering::SeqCst);

                block.size = aligned_size;
                block.in_use = true;
                block.tag = tag.map(|s| s.to_string());
                let alloc_id = block.id;

                // Create new free block from remainder
                let remaining_block = MemoryBlock::new(remaining_id, remaining_offset, remaining_size);
                blocks.push(remaining_block);

                // Add remainder to free list
                let mut free = self.free_by_size.lock();
                free.entry(remaining_size).or_default().push(remaining_id);

                alloc_id
            } else {
                block.in_use = true;
                block.tag = tag.map(|s| s.to_string());
                block.id
            }
        };

        // Update stats
        self.allocated_bytes.fetch_add(aligned_size, Ordering::SeqCst);
        let current = self.allocated_bytes.load(Ordering::SeqCst);
        self.peak_usage.fetch_max(current, Ordering::SeqCst);
        self.num_allocations.fetch_add(1, Ordering::SeqCst);

        debug!(
            "Allocated {} bytes (block {}), tag: {:?}",
            aligned_size, actual_id, tag
        );

        Some(actual_id)
    }

    /// Free a memory block.
    pub fn free(&self, block_id: u64) -> bool {
        let freed_size = {
            let mut blocks = self.blocks.write();
            let block_idx = match blocks.iter().position(|b| b.id == block_id) {
                Some(idx) => idx,
                None => return false,
            };

            let block = &mut blocks[block_idx];
            if !block.in_use {
                warn!("Double free detected for block {}", block_id);
                return false;
            }

            let size = block.size;
            block.in_use = false;
            block.tag = None;

            // Add to free list
            let mut free = self.free_by_size.lock();
            free.entry(size).or_default().push(block_id);

            size
        };

        self.allocated_bytes.fetch_sub(freed_size, Ordering::SeqCst);
        self.num_frees.fetch_add(1, Ordering::SeqCst);

        debug!("Freed {} bytes (block {})", freed_size, block_id);

        // Try to coalesce free blocks
        if self.config.enable_defrag {
            self.coalesce_free_blocks();
        }

        true
    }

    /// Get block info.
    pub fn get_block(&self, block_id: u64) -> Option<MemoryBlock> {
        let blocks = self.blocks.read();
        blocks.iter().find(|b| b.id == block_id).cloned()
    }

    /// Get allocation statistics.
    pub fn stats(&self) -> MemoryStats {
        let blocks = self.blocks.read();
        let free_blocks = blocks.iter().filter(|b| !b.in_use).count();
        let used_blocks = blocks.iter().filter(|b| b.in_use).count();
        let fragmentation = self.calculate_fragmentation(&blocks);

        MemoryStats {
            total_size: self.pool_size.load(Ordering::SeqCst),
            allocated_bytes: self.allocated_bytes.load(Ordering::SeqCst),
            peak_usage: self.peak_usage.load(Ordering::SeqCst),
            free_bytes: self.pool_size.load(Ordering::SeqCst) - self.allocated_bytes.load(Ordering::SeqCst),
            num_allocations: self.num_allocations.load(Ordering::SeqCst),
            num_frees: self.num_frees.load(Ordering::SeqCst),
            free_blocks,
            used_blocks,
            fragmentation,
        }
    }

    /// Align size to minimum allocation size.
    fn align_size(&self, size: usize) -> usize {
        let alignment = self.config.alignment;
        (size + alignment - 1) / alignment * alignment
    }

    /// Find a free block that can fit the requested size.
    fn find_free_block(&self, size: usize) -> Option<u64> {
        let mut free = self.free_by_size.lock();

        // Try to find exact match first
        if let Some(ids) = free.get_mut(&size) {
            if let Some(id) = ids.pop() {
                if ids.is_empty() {
                    free.remove(&size);
                }
                return Some(id);
            }
        }

        // Find smallest block that fits (best-fit)
        let mut best_size = None;
        for &available_size in free.keys() {
            if available_size >= size {
                if best_size.is_none() || available_size < best_size.unwrap() {
                    best_size = Some(available_size);
                }
            }
        }

        if let Some(block_size) = best_size {
            if let Some(ids) = free.get_mut(&block_size) {
                if let Some(id) = ids.pop() {
                    if ids.is_empty() {
                        free.remove(&block_size);
                    }
                    return Some(id);
                }
            }
        }

        None
    }

    /// Coalesce adjacent free blocks.
    fn coalesce_free_blocks(&self) {
        let mut blocks = self.blocks.write();
        let mut free = self.free_by_size.lock();

        // Sort blocks by offset
        blocks.sort_by_key(|b| b.offset);

        let mut i = 0;
        while i < blocks.len() - 1 {
            if !blocks[i].in_use && !blocks[i + 1].in_use {
                // Coalesce
                let merged_size = blocks[i].size + blocks[i + 1].size;
                let removed_id = blocks[i + 1].id;
                let old_size1 = blocks[i].size;
                let old_size2 = blocks[i + 1].size;

                blocks[i].size = merged_size;

                // Remove from free lists
                if let Some(ids) = free.get_mut(&old_size1) {
                    ids.retain(|&id| id != blocks[i].id);
                    if ids.is_empty() {
                        free.remove(&old_size1);
                    }
                }
                if let Some(ids) = free.get_mut(&old_size2) {
                    ids.retain(|&id| id != removed_id);
                    if ids.is_empty() {
                        free.remove(&old_size2);
                    }
                }

                // Add merged block to free list
                free.entry(merged_size).or_default().push(blocks[i].id);

                // Remove the coalesced block
                blocks.remove(i + 1);

                debug!("Coalesced blocks: {} bytes", merged_size);
            } else {
                i += 1;
            }
        }
    }

    /// Calculate fragmentation ratio.
    fn calculate_fragmentation(&self, blocks: &[MemoryBlock]) -> f64 {
        let free_blocks: Vec<_> = blocks.iter().filter(|b| !b.in_use).collect();
        if free_blocks.is_empty() {
            return 0.0;
        }

        let total_free: usize = free_blocks.iter().map(|b| b.size).sum();
        let largest_free = free_blocks.iter().map(|b| b.size).max().unwrap_or(0);

        if total_free == 0 {
            0.0
        } else {
            1.0 - (largest_free as f64 / total_free as f64)
        }
    }

    /// Defragment the memory pool.
    pub fn defragment(&self) {
        // In a real implementation, this would move allocations to reduce fragmentation
        // For now, we just coalesce free blocks
        self.coalesce_free_blocks();
        info!("Memory defragmentation completed");
    }

    /// Check if allocation would succeed.
    pub fn can_allocate(&self, size: usize) -> bool {
        let aligned_size = self.align_size(size);
        let free = self.free_by_size.lock();

        for &available_size in free.keys() {
            if available_size >= aligned_size {
                return true;
            }
        }
        false
    }

    /// Get utilization ratio.
    pub fn utilization(&self) -> f64 {
        let allocated = self.allocated_bytes.load(Ordering::SeqCst);
        let total = self.pool_size.load(Ordering::SeqCst);
        if total == 0 {
            0.0
        } else {
            allocated as f64 / total as f64
        }
    }
}

/// Memory statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total pool size in bytes.
    pub total_size: usize,

    /// Currently allocated bytes.
    pub allocated_bytes: usize,

    /// Peak memory usage.
    pub peak_usage: usize,

    /// Free bytes.
    pub free_bytes: usize,

    /// Number of allocations.
    pub num_allocations: u64,

    /// Number of frees.
    pub num_frees: u64,

    /// Number of free blocks.
    pub free_blocks: usize,

    /// Number of used blocks.
    pub used_blocks: usize,

    /// Fragmentation ratio (0-1).
    pub fragmentation: f64,
}

impl MemoryStats {
    /// Get utilization percentage.
    pub fn utilization_percent(&self) -> f64 {
        if self.total_size == 0 {
            0.0
        } else {
            self.allocated_bytes as f64 / self.total_size as f64 * 100.0
        }
    }

    /// Format as human-readable string.
    pub fn to_string_human(&self) -> String {
        format!(
            "Memory: {:.2} GB / {:.2} GB ({:.1}% used), {} allocs, {:.1}% fragmented",
            self.allocated_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.total_size as f64 / (1024.0 * 1024.0 * 1024.0),
            self.utilization_percent(),
            self.num_allocations,
            self.fragmentation * 100.0
        )
    }
}

/// GPU memory tracker.
#[derive(Debug)]
pub struct GpuMemoryTracker {
    /// Device ID.
    pub device_id: usize,

    /// Total memory in bytes.
    pub total_memory: usize,

    /// Reserved memory in bytes.
    pub reserved_memory: AtomicUsize,

    /// Used memory in bytes.
    pub used_memory: AtomicUsize,

    /// Peak memory usage.
    pub peak_memory: AtomicUsize,
}

impl GpuMemoryTracker {
    /// Create a new GPU memory tracker.
    pub fn new(device_id: usize, total_memory: usize) -> Self {
        Self {
            device_id,
            total_memory,
            reserved_memory: AtomicUsize::new(0),
            used_memory: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
        }
    }

    /// Record allocation.
    pub fn record_alloc(&self, size: usize) {
        self.used_memory.fetch_add(size, Ordering::SeqCst);
        let current = self.used_memory.load(Ordering::SeqCst);
        self.peak_memory.fetch_max(current, Ordering::SeqCst);
    }

    /// Record free.
    pub fn record_free(&self, size: usize) {
        self.used_memory.fetch_sub(size, Ordering::SeqCst);
    }

    /// Get available memory.
    pub fn available_memory(&self) -> usize {
        let reserved = self.reserved_memory.load(Ordering::SeqCst);
        let used = self.used_memory.load(Ordering::SeqCst);
        self.total_memory.saturating_sub(reserved + used)
    }

    /// Get utilization.
    pub fn utilization(&self) -> f64 {
        let used = self.used_memory.load(Ordering::SeqCst);
        used as f64 / self.total_memory as f64
    }

    /// Reserve memory for other uses.
    pub fn reserve(&self, size: usize) {
        self.reserved_memory.fetch_add(size, Ordering::SeqCst);
    }
}

/// Memory profiler for detailed tracking.
#[derive(Debug, Default)]
pub struct MemoryProfiler {
    /// Allocation records.
    records: Mutex<Vec<AllocationRecord>>,

    /// Enable profiling.
    enabled: std::sync::atomic::AtomicBool,
}

/// A single allocation record.
#[derive(Debug, Clone)]
pub struct AllocationRecord {
    /// Size in bytes.
    pub size: usize,

    /// Allocation tag.
    pub tag: String,

    /// Timestamp (milliseconds since start).
    pub timestamp_ms: u64,

    /// Whether this is an allocation (true) or free (false).
    pub is_alloc: bool,
}

impl MemoryProfiler {
    /// Create a new memory profiler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable profiling.
    pub fn enable(&self) {
        self.enabled.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Disable profiling.
    pub fn disable(&self) {
        self.enabled.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Record an allocation.
    pub fn record_alloc(&self, size: usize, tag: &str) {
        if self.enabled.load(std::sync::atomic::Ordering::SeqCst) {
            let mut records = self.records.lock();
            records.push(AllocationRecord {
                size,
                tag: tag.to_string(),
                timestamp_ms: 0, // Would use actual timestamp in production
                is_alloc: true,
            });
        }
    }

    /// Record a free.
    pub fn record_free(&self, size: usize, tag: &str) {
        if self.enabled.load(std::sync::atomic::Ordering::SeqCst) {
            let mut records = self.records.lock();
            records.push(AllocationRecord {
                size,
                tag: tag.to_string(),
                timestamp_ms: 0,
                is_alloc: false,
            });
        }
    }

    /// Get all records.
    pub fn get_records(&self) -> Vec<AllocationRecord> {
        self.records.lock().clone()
    }

    /// Clear records.
    pub fn clear(&self) {
        self.records.lock().clear();
    }

    /// Get summary by tag.
    pub fn summary_by_tag(&self) -> HashMap<String, (usize, usize)> {
        let records = self.records.lock();
        let mut summary: HashMap<String, (usize, usize)> = HashMap::new();

        for record in records.iter() {
            let entry = summary.entry(record.tag.clone()).or_default();
            if record.is_alloc {
                entry.0 += record.size;
            } else {
                entry.1 += record.size;
            }
        }

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_allocator() {
        let config = MemoryPoolConfig {
            initial_size: 1024 * 1024, // 1 MB
            ..Default::default()
        };
        let allocator = MemoryAllocator::new(config);

        // Allocate some blocks
        let block1 = allocator.allocate(1000, Some("test1")).unwrap();
        let block2 = allocator.allocate(2000, Some("test2")).unwrap();
        let block3 = allocator.allocate(3000, Some("test3")).unwrap();

        let stats = allocator.stats();
        assert!(stats.allocated_bytes >= 6000);
        assert_eq!(stats.used_blocks, 3);

        // Free middle block
        allocator.free(block2);
        let stats = allocator.stats();
        assert_eq!(stats.used_blocks, 2);

        // Allocate new block in freed space
        let block4 = allocator.allocate(1500, Some("test4")).unwrap();
        assert!(block4 > 0);
    }

    #[test]
    fn test_coalescing() {
        let config = MemoryPoolConfig {
            initial_size: 1024 * 1024,
            enable_defrag: true,
            ..Default::default()
        };
        let allocator = MemoryAllocator::new(config);

        // Allocate three blocks
        let b1 = allocator.allocate(1000, None).unwrap();
        let b2 = allocator.allocate(1000, None).unwrap();
        let b3 = allocator.allocate(1000, None).unwrap();

        // Free adjacent blocks
        allocator.free(b1);
        allocator.free(b2);

        // They should coalesce
        let stats = allocator.stats();
        assert!(stats.free_blocks <= 2); // Might be coalesced
    }

    #[test]
    fn test_gpu_memory_tracker() {
        let tracker = GpuMemoryTracker::new(0, 16 * 1024 * 1024 * 1024); // 16 GB

        tracker.record_alloc(1024 * 1024 * 1024); // 1 GB
        assert!((tracker.utilization() - 0.0625).abs() < 0.01);

        tracker.record_alloc(1024 * 1024 * 1024); // Another 1 GB
        assert!((tracker.utilization() - 0.125).abs() < 0.01);

        tracker.record_free(1024 * 1024 * 1024);
        assert!((tracker.utilization() - 0.0625).abs() < 0.01);
    }

    #[test]
    fn test_memory_profiler() {
        let profiler = MemoryProfiler::new();
        profiler.enable();

        profiler.record_alloc(1000, "layer1");
        profiler.record_alloc(2000, "layer1");
        profiler.record_alloc(500, "layer2");
        profiler.record_free(1000, "layer1");

        let records = profiler.get_records();
        assert_eq!(records.len(), 4);

        let summary = profiler.summary_by_tag();
        assert_eq!(summary.get("layer1"), Some(&(3000, 1000)));
        assert_eq!(summary.get("layer2"), Some(&(500, 0)));
    }
}
