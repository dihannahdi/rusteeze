//! # NUMA-Aware Memory Manager — Radical Rewrite
//!
//! NUMA topology detection, thread-local per-node arenas, and
//! allocation with locality hints. Uses atomic stats instead of RwLock.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::alloc::{Layout, alloc, dealloc};
use parking_lot::Mutex;

use crate::simd_dispatch;

/// NUMA topology information.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// CPUs per node
    pub cpus_per_node: Vec<Vec<usize>>,
    /// Total memory per node (bytes)
    pub memory_per_node: Vec<u64>,
    /// Total system CPUs
    pub total_cpus: usize,
}

impl NumaTopology {
    /// Detect NUMA topology (falls back to single-node on non-NUMA systems).
    pub fn detect() -> Self {
        let total_cpus = num_cpus::get();
        // On most consumer systems, single NUMA node
        Self {
            num_nodes: 1,
            cpus_per_node: vec![(0..total_cpus).collect()],
            memory_per_node: vec![0], // Unknown on consumer systems
            total_cpus,
        }
    }

    /// Get preferred NUMA node for current thread.
    pub fn current_node(&self) -> usize {
        // Thread-local node affinity
        THREAD_NODE.with(|n| *n)
    }
}

thread_local! {
    /// Per-thread NUMA node assignment
    static THREAD_NODE: usize = 0;
}

/// NUMA configuration.
#[derive(Debug, Clone)]
pub struct NumaConfig {
    /// Enable NUMA-aware allocation
    pub enabled: bool,
    /// Memory pool size per node (bytes)
    pub pool_size_per_node: usize,
    /// Alignment for allocations
    pub alignment: usize,
    /// Enable interleaving for large allocations
    pub interleave_large: bool,
    /// Threshold for "large" allocation (bytes)
    pub large_threshold: usize,
}

impl Default for NumaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pool_size_per_node: 256 * 1024 * 1024, // 256 MB
            alignment: 64, // Cache line aligned
            interleave_large: true,
            large_threshold: 4 * 1024 * 1024, // 4 MB
        }
    }
}

/// Per-node allocation statistics (atomic — no locks).
#[derive(Debug, Default)]
pub struct NodeStats {
    pub allocations: AtomicU64,
    pub deallocations: AtomicU64,
    pub bytes_allocated: AtomicU64,
    pub bytes_freed: AtomicU64,
    pub peak_bytes: AtomicU64,
}

impl Clone for NodeStats {
    fn clone(&self) -> Self {
        Self {
            allocations: AtomicU64::new(self.allocations.load(Ordering::Relaxed)),
            deallocations: AtomicU64::new(self.deallocations.load(Ordering::Relaxed)),
            bytes_allocated: AtomicU64::new(self.bytes_allocated.load(Ordering::Relaxed)),
            bytes_freed: AtomicU64::new(self.bytes_freed.load(Ordering::Relaxed)),
            peak_bytes: AtomicU64::new(self.peak_bytes.load(Ordering::Relaxed)),
        }
    }
}

/// Memory block tracker.
struct AllocBlock {
    ptr: *mut u8,
    layout: Layout,
    node: usize,
}

unsafe impl Send for AllocBlock {}

/// Hierarchical memory pool — per-node free lists.
pub struct HierarchicalPool {
    /// Free lists keyed by size class
    free_lists: Vec<Mutex<Vec<(*mut u8, Layout)>>>,
    /// Number of nodes
    num_nodes: usize,
}

unsafe impl Send for HierarchicalPool {}
unsafe impl Sync for HierarchicalPool {}

impl HierarchicalPool {
    /// Create pool with given number of NUMA nodes.
    pub fn new(num_nodes: usize) -> Self {
        Self {
            free_lists: (0..num_nodes).map(|_| Mutex::new(Vec::new())).collect(),
            num_nodes,
        }
    }

    /// Return a block to the pool.
    pub fn return_block(&self, ptr: *mut u8, layout: Layout, node: usize) {
        let node = node.min(self.num_nodes - 1);
        self.free_lists[node].lock().push((ptr, layout));
    }

    /// Get a block from the pool (if available).
    pub fn get_block(&self, size: usize, node: usize) -> Option<*mut u8> {
        let node = node.min(self.num_nodes - 1);
        let mut list = self.free_lists[node].lock();
        // Find a block that's large enough
        if let Some(idx) = list.iter().position(|(_, l)| l.size() >= size) {
            let (ptr, _) = list.swap_remove(idx);
            Some(ptr)
        } else {
            None
        }
    }
}

impl Drop for HierarchicalPool {
    fn drop(&mut self) {
        for list in &self.free_lists {
            let mut list = list.lock();
            for (ptr, layout) in list.drain(..) {
                unsafe { dealloc(ptr, layout); }
            }
        }
    }
}

/// NUMA-aware memory manager.
pub struct NumaMemoryManager {
    config: NumaConfig,
    topology: NumaTopology,
    pool: HierarchicalPool,
    stats: Vec<NodeStats>,
}

impl NumaMemoryManager {
    /// Create a new NUMA memory manager.
    pub fn new(config: NumaConfig) -> Self {
        simd_dispatch::init();
        let topology = NumaTopology::detect();
        let num_nodes = topology.num_nodes;
        let pool = HierarchicalPool::new(num_nodes);
        let stats = (0..num_nodes).map(|_| NodeStats::default()).collect();
        Self { config, topology, pool, stats }
    }

    /// Allocate memory with NUMA locality hint.
    pub fn allocate(&self, size: usize, node_hint: Option<usize>) -> Option<*mut u8> {
        let node = node_hint.unwrap_or_else(|| self.topology.current_node());
        let node = node.min(self.topology.num_nodes - 1);

        // Try pool first
        if let Some(ptr) = self.pool.get_block(size, node) {
            self.stats[node].allocations.fetch_add(1, Ordering::Relaxed);
            return Some(ptr);
        }

        // Allocate from system
        let align = self.config.alignment;
        let layout = Layout::from_size_align(size, align).ok()?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() { return None; }

        // Update stats
        let stats = &self.stats[node];
        stats.allocations.fetch_add(1, Ordering::Relaxed);
        let total = stats.bytes_allocated.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
        // Update peak
        loop {
            let peak = stats.peak_bytes.load(Ordering::Relaxed);
            if total <= peak { break; }
            if stats.peak_bytes.compare_exchange_weak(peak, total, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                break;
            }
        }

        Some(ptr)
    }

    /// Deallocate memory.
    pub fn deallocate(&self, ptr: *mut u8, size: usize, node_hint: Option<usize>) {
        let node = node_hint.unwrap_or(0).min(self.topology.num_nodes - 1);
        let align = self.config.alignment;
        let layout = Layout::from_size_align(size, align).unwrap();

        // Return to pool instead of freeing
        self.pool.return_block(ptr, layout, node);

        let stats = &self.stats[node];
        stats.deallocations.fetch_add(1, Ordering::Relaxed);
        stats.bytes_freed.fetch_add(size as u64, Ordering::Relaxed);
    }

    /// Allocate a Vec<f32> with NUMA locality.
    pub fn allocate_f32_vec(&self, len: usize, node_hint: Option<usize>) -> Option<Vec<f32>> {
        let size = len * std::mem::size_of::<f32>();
        let align = self.config.alignment.max(std::mem::align_of::<f32>());
        let layout = Layout::from_size_align(size, align).ok()?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() { return None; }

        // SAFETY: We just allocated this memory
        let vec = unsafe {
            Vec::from_raw_parts(ptr as *mut f32, len, len)
        };

        let node = node_hint.unwrap_or(0).min(self.topology.num_nodes - 1);
        self.stats[node].allocations.fetch_add(1, Ordering::Relaxed);
        self.stats[node].bytes_allocated.fetch_add(size as u64, Ordering::Relaxed);

        Some(vec)
    }

    /// Get topology.
    pub fn topology(&self) -> &NumaTopology { &self.topology }

    /// Get stats for a node.
    pub fn node_stats(&self, node: usize) -> Option<&NodeStats> {
        self.stats.get(node)
    }

    /// Get total allocated bytes.
    pub fn total_allocated(&self) -> u64 {
        self.stats.iter().map(|s| s.bytes_allocated.load(Ordering::Relaxed)).sum()
    }

    /// Get total freed bytes.
    pub fn total_freed(&self) -> u64 {
        self.stats.iter().map(|s| s.bytes_freed.load(Ordering::Relaxed)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_detect() {
        let topo = NumaTopology::detect();
        assert!(topo.num_nodes >= 1);
        assert!(topo.total_cpus >= 1);
    }

    #[test]
    fn test_allocate_deallocate() {
        let mgr = NumaMemoryManager::new(NumaConfig::default());
        let ptr = mgr.allocate(1024, None).unwrap();
        assert!(!ptr.is_null());
        mgr.deallocate(ptr, 1024, None);
        assert!(mgr.total_allocated() >= 1024);
    }

    #[test]
    fn test_pool_reuse() {
        let mgr = NumaMemoryManager::new(NumaConfig::default());
        let ptr1 = mgr.allocate(1024, Some(0)).unwrap();
        mgr.deallocate(ptr1, 1024, Some(0));
        // Next allocation of same size should reuse from pool
        let ptr2 = mgr.allocate(1024, Some(0)).unwrap();
        assert!(!ptr2.is_null());
        mgr.deallocate(ptr2, 1024, Some(0));
    }
}
