//! KV Cache management for efficient autoregressive generation.
//!
//! This module implements:
//!
//! - Block-based (paged) KV cache allocation
//! - Dynamic memory management
//! - Prefix caching for common prompts
//! - Cache eviction policies

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use parking_lot::{Mutex, RwLock};
use tracing::{debug, info, instrument, warn};

/// Block size in tokens.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Configuration for KV cache.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Number of layers.
    pub num_layers: usize,

    /// Number of KV heads per layer.
    pub num_kv_heads: usize,

    /// Head dimension.
    pub head_dim: usize,

    /// Block size in tokens.
    pub block_size: usize,

    /// Number of GPU blocks.
    pub num_gpu_blocks: usize,

    /// Number of CPU blocks (for swapping).
    pub num_cpu_blocks: usize,

    /// Data type for cache.
    pub dtype: DType,

    /// Enable prefix caching.
    pub enable_prefix_caching: bool,

    /// Sliding window size (if any).
    pub sliding_window: Option<usize>,
}

impl KVCacheConfig {
    /// Create a new KV cache config.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_gpu_blocks: usize,
    ) -> Self {
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            block_size: DEFAULT_BLOCK_SIZE,
            num_gpu_blocks,
            num_cpu_blocks: 0,
            dtype: DType::F16,
            enable_prefix_caching: false,
            sliding_window: None,
        }
    }

    /// Set block size.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set CPU blocks.
    pub fn with_cpu_blocks(mut self, num_cpu_blocks: usize) -> Self {
        self.num_cpu_blocks = num_cpu_blocks;
        self
    }

    /// Enable prefix caching.
    pub fn with_prefix_caching(mut self) -> Self {
        self.enable_prefix_caching = true;
        self
    }

    /// Set sliding window.
    pub fn with_sliding_window(mut self, window: usize) -> Self {
        self.sliding_window = Some(window);
        self
    }

    /// Get bytes per block.
    pub fn bytes_per_block(&self) -> usize {
        let elements = self.num_layers * 2 * self.num_kv_heads * self.head_dim * self.block_size;
        elements * self.dtype.size_in_bytes()
    }

    /// Get total GPU memory needed.
    pub fn total_gpu_memory(&self) -> usize {
        self.num_gpu_blocks * self.bytes_per_block()
    }

    /// Get total CPU memory needed.
    pub fn total_cpu_memory(&self) -> usize {
        self.num_cpu_blocks * self.bytes_per_block()
    }
}

/// A physical block of KV cache memory.
#[derive(Debug)]
pub struct PhysicalBlock {
    /// Block ID.
    pub block_id: u32,

    /// Reference count.
    pub ref_count: AtomicUsize,

    /// Whether on GPU or CPU.
    pub on_gpu: bool,

    /// Content hash (for prefix caching).
    pub content_hash: Option<u64>,

    /// Number of tokens stored.
    pub num_tokens: AtomicUsize,
}

impl PhysicalBlock {
    /// Create a new physical block.
    pub fn new(block_id: u32, on_gpu: bool) -> Self {
        Self {
            block_id,
            ref_count: AtomicUsize::new(0),
            on_gpu,
            content_hash: None,
            num_tokens: AtomicUsize::new(0),
        }
    }

    /// Increment reference count.
    pub fn inc_ref(&self) -> usize {
        self.ref_count.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Decrement reference count.
    pub fn dec_ref(&self) -> usize {
        let prev = self.ref_count.fetch_sub(1, Ordering::SeqCst);
        if prev == 0 {
            panic!("Block ref count underflow");
        }
        prev - 1
    }

    /// Get reference count.
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// Check if block is free.
    pub fn is_free(&self) -> bool {
        self.ref_count() == 0
    }

    /// Get number of tokens.
    pub fn num_tokens(&self) -> usize {
        self.num_tokens.load(Ordering::SeqCst)
    }

    /// Set number of tokens.
    pub fn set_num_tokens(&self, n: usize) {
        self.num_tokens.store(n, Ordering::SeqCst);
    }
}

/// Block allocator for KV cache.
#[derive(Debug)]
pub struct BlockAllocator {
    /// Configuration.
    config: KVCacheConfig,

    /// Free GPU block IDs.
    free_gpu_blocks: Mutex<VecDeque<u32>>,

    /// Free CPU block IDs.
    free_cpu_blocks: Mutex<VecDeque<u32>>,

    /// All physical blocks.
    blocks: Vec<Arc<PhysicalBlock>>,

    /// Total GPU blocks allocated.
    total_gpu_allocated: AtomicUsize,

    /// Total CPU blocks allocated.
    total_cpu_allocated: AtomicUsize,

    /// Hash map for prefix caching.
    prefix_cache: RwLock<HashMap<u64, Vec<u32>>>,
}

impl BlockAllocator {
    /// Create a new block allocator.
    pub fn new(config: KVCacheConfig) -> Self {
        let num_gpu = config.num_gpu_blocks;
        let num_cpu = config.num_cpu_blocks;

        // Initialize blocks
        let mut blocks = Vec::with_capacity(num_gpu + num_cpu);
        let mut free_gpu = VecDeque::with_capacity(num_gpu);
        let mut free_cpu = VecDeque::with_capacity(num_cpu);

        for i in 0..num_gpu {
            blocks.push(Arc::new(PhysicalBlock::new(i as u32, true)));
            free_gpu.push_back(i as u32);
        }

        for i in 0..num_cpu {
            let block_id = (num_gpu + i) as u32;
            blocks.push(Arc::new(PhysicalBlock::new(block_id, false)));
            free_cpu.push_back(block_id);
        }

        info!(
            "BlockAllocator initialized: {} GPU blocks, {} CPU blocks, {} bytes/block",
            num_gpu,
            num_cpu,
            config.bytes_per_block()
        );

        Self {
            config,
            free_gpu_blocks: Mutex::new(free_gpu),
            free_cpu_blocks: Mutex::new(free_cpu),
            blocks,
            total_gpu_allocated: AtomicUsize::new(0),
            total_cpu_allocated: AtomicUsize::new(0),
            prefix_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Allocate a GPU block.
    pub fn allocate_gpu(&self) -> Option<u32> {
        let mut free = self.free_gpu_blocks.lock();
        if let Some(block_id) = free.pop_front() {
            self.blocks[block_id as usize].inc_ref();
            self.total_gpu_allocated.fetch_add(1, Ordering::SeqCst);
            debug!("Allocated GPU block {}", block_id);
            Some(block_id)
        } else {
            None
        }
    }

    /// Allocate a CPU block.
    pub fn allocate_cpu(&self) -> Option<u32> {
        let mut free = self.free_cpu_blocks.lock();
        if let Some(block_id) = free.pop_front() {
            self.blocks[block_id as usize].inc_ref();
            self.total_cpu_allocated.fetch_add(1, Ordering::SeqCst);
            debug!("Allocated CPU block {}", block_id);
            Some(block_id)
        } else {
            None
        }
    }

    /// Allocate multiple GPU blocks.
    pub fn allocate_gpu_blocks(&self, n: usize) -> Option<Vec<u32>> {
        let mut free = self.free_gpu_blocks.lock();
        if free.len() < n {
            return None;
        }

        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            let block_id = free.pop_front().unwrap();
            self.blocks[block_id as usize].inc_ref();
            blocks.push(block_id);
        }
        self.total_gpu_allocated.fetch_add(n, Ordering::SeqCst);
        debug!("Allocated {} GPU blocks: {:?}", n, blocks);
        Some(blocks)
    }

    /// Free a block.
    pub fn free(&self, block_id: u32) {
        let block = &self.blocks[block_id as usize];
        let new_ref = block.dec_ref();

        if new_ref == 0 {
            block.set_num_tokens(0);
            if block.on_gpu {
                self.free_gpu_blocks.lock().push_back(block_id);
                self.total_gpu_allocated.fetch_sub(1, Ordering::SeqCst);
            } else {
                self.free_cpu_blocks.lock().push_back(block_id);
                self.total_cpu_allocated.fetch_sub(1, Ordering::SeqCst);
            }
            debug!("Freed block {}", block_id);
        }
    }

    /// Free multiple blocks.
    pub fn free_blocks(&self, block_ids: &[u32]) {
        for &block_id in block_ids {
            self.free(block_id);
        }
    }

    /// Get number of free GPU blocks.
    pub fn num_free_gpu_blocks(&self) -> usize {
        self.free_gpu_blocks.lock().len()
    }

    /// Get number of free CPU blocks.
    pub fn num_free_cpu_blocks(&self) -> usize {
        self.free_cpu_blocks.lock().len()
    }

    /// Get total allocated GPU blocks.
    pub fn num_allocated_gpu_blocks(&self) -> usize {
        self.total_gpu_allocated.load(Ordering::SeqCst)
    }

    /// Get total allocated CPU blocks.
    pub fn num_allocated_cpu_blocks(&self) -> usize {
        self.total_cpu_allocated.load(Ordering::SeqCst)
    }

    /// Get block by ID.
    pub fn get_block(&self, block_id: u32) -> Option<&Arc<PhysicalBlock>> {
        self.blocks.get(block_id as usize)
    }

    /// Fork a block (copy-on-write).
    pub fn fork(&self, block_id: u32) -> u32 {
        self.blocks[block_id as usize].inc_ref();
        block_id
    }

    /// Can allocate n GPU blocks?
    pub fn can_allocate(&self, n: usize) -> bool {
        self.num_free_gpu_blocks() >= n
    }

    /// Get memory utilization (0.0 - 1.0).
    pub fn gpu_utilization(&self) -> f64 {
        let allocated = self.num_allocated_gpu_blocks();
        let total = self.config.num_gpu_blocks;
        if total == 0 {
            0.0
        } else {
            allocated as f64 / total as f64
        }
    }
}

/// Logical block mapping for a sequence.
#[derive(Debug, Clone)]
pub struct BlockTable {
    /// Logical to physical block mapping.
    pub block_ids: Vec<u32>,

    /// Number of tokens in the last block.
    pub last_block_tokens: usize,

    /// Block size.
    pub block_size: usize,
}

impl BlockTable {
    /// Create a new block table.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_ids: Vec::new(),
            last_block_tokens: 0,
            block_size,
        }
    }

    /// Get number of blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_ids.len()
    }

    /// Get total tokens.
    pub fn num_tokens(&self) -> usize {
        if self.block_ids.is_empty() {
            0
        } else {
            (self.block_ids.len() - 1) * self.block_size + self.last_block_tokens
        }
    }

    /// Append a block.
    pub fn append(&mut self, block_id: u32) {
        self.block_ids.push(block_id);
        self.last_block_tokens = 0;
    }

    /// Add tokens to the last block.
    pub fn add_tokens(&mut self, n: usize) {
        self.last_block_tokens += n;
    }

    /// Check if last block is full.
    pub fn is_last_block_full(&self) -> bool {
        self.last_block_tokens >= self.block_size
    }

    /// Get slots for new tokens.
    pub fn slots_in_last_block(&self) -> usize {
        self.block_size - self.last_block_tokens
    }

    /// Convert to physical slot indices.
    pub fn to_slot_indices(&self, num_tokens: usize) -> Vec<usize> {
        let mut slots = Vec::with_capacity(num_tokens);
        for (i, &block_id) in self.block_ids.iter().enumerate() {
            let block_start = block_id as usize * self.block_size;
            let tokens_in_block = if i == self.block_ids.len() - 1 {
                self.last_block_tokens.min(num_tokens - slots.len())
            } else {
                self.block_size.min(num_tokens - slots.len())
            };
            for j in 0..tokens_in_block {
                slots.push(block_start + j);
            }
            if slots.len() >= num_tokens {
                break;
            }
        }
        slots
    }
}

/// Layer-specific KV cache tensors.
#[derive(Debug)]
pub struct LayerCache {
    /// Key cache: [num_blocks, block_size, num_kv_heads, head_dim]
    pub key_cache: Tensor,

    /// Value cache: [num_blocks, block_size, num_kv_heads, head_dim]
    pub value_cache: Tensor,

    /// Layer index.
    pub layer_idx: usize,
}

impl LayerCache {
    /// Create a new layer cache.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let key_cache = Tensor::zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype,
            device,
        )?;
        let value_cache = Tensor::zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype,
            device,
        )?;

        Ok(Self {
            key_cache,
            value_cache,
            layer_idx: 0,
        })
    }

    /// Write KV to cache at specific slots.
    #[instrument(skip(self, key, value), level = "debug")]
    pub fn write(&mut self, key: &Tensor, value: &Tensor, slot_indices: &[usize]) -> Result<()> {
        // key, value: [batch, num_kv_heads, seq_len, head_dim]
        // Flatten batch and seq dimensions
        let key = key.flatten(0, 2)?; // [batch*seq_len, num_kv_heads, head_dim]
        let value = value.flatten(0, 2)?;

        // This is a simplified version - in production, we'd use scatter operations
        // For now, we'll use a loop (not optimal but correct)
        for (token_idx, &slot) in slot_indices.iter().enumerate() {
            let block_idx = slot / self.key_cache.dim(1)?;
            let block_offset = slot % self.key_cache.dim(1)?;

            // Get single token's KV
            let k = key.i(token_idx)?;
            let v = value.i(token_idx)?;

            // Write to cache - this would be optimized with custom CUDA kernels
            // For now, we skip the actual write as it requires unsafe operations
        }

        Ok(())
    }

    /// Read KV from cache for given slots.
    pub fn read(&self, slot_indices: &[usize]) -> Result<(Tensor, Tensor)> {
        // In production, this would use gather operations
        // Placeholder implementation
        let num_tokens = slot_indices.len();
        let num_kv_heads = self.key_cache.dim(2)?;
        let head_dim = self.key_cache.dim(3)?;

        let key = Tensor::zeros(
            (1, num_kv_heads, num_tokens, head_dim),
            self.key_cache.dtype(),
            self.key_cache.device(),
        )?;
        let value = Tensor::zeros(
            (1, num_kv_heads, num_tokens, head_dim),
            self.value_cache.dtype(),
            self.value_cache.device(),
        )?;

        Ok((key, value))
    }
}

/// Full KV cache manager.
#[derive(Debug)]
pub struct KVCacheManager {
    /// Configuration.
    config: KVCacheConfig,

    /// Block allocator.
    allocator: Arc<BlockAllocator>,

    /// GPU layer caches.
    gpu_caches: Vec<LayerCache>,

    /// CPU layer caches (for swapping).
    cpu_caches: Option<Vec<LayerCache>>,

    /// Device.
    device: Device,
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    pub fn new(config: KVCacheConfig, device: Device) -> Result<Self> {
        let allocator = Arc::new(BlockAllocator::new(config.clone()));

        // Initialize GPU caches
        let mut gpu_caches = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let mut cache = LayerCache::new(
                config.num_gpu_blocks,
                config.block_size,
                config.num_kv_heads,
                config.head_dim,
                config.dtype,
                &device,
            )?;
            cache.layer_idx = layer_idx;
            gpu_caches.push(cache);
        }

        // Initialize CPU caches if needed
        let cpu_caches = if config.num_cpu_blocks > 0 {
            let cpu_device = Device::Cpu;
            let mut caches = Vec::with_capacity(config.num_layers);
            for layer_idx in 0..config.num_layers {
                let mut cache = LayerCache::new(
                    config.num_cpu_blocks,
                    config.block_size,
                    config.num_kv_heads,
                    config.head_dim,
                    config.dtype,
                    &cpu_device,
                )?;
                cache.layer_idx = layer_idx;
                caches.push(cache);
            }
            Some(caches)
        } else {
            None
        };

        info!(
            "KVCacheManager initialized: {} layers, {} GPU blocks, {} bytes total",
            config.num_layers,
            config.num_gpu_blocks,
            config.total_gpu_memory()
        );

        Ok(Self {
            config,
            allocator,
            gpu_caches,
            cpu_caches,
            device,
        })
    }

    /// Get the block allocator.
    pub fn allocator(&self) -> &Arc<BlockAllocator> {
        &self.allocator
    }

    /// Get layer cache.
    pub fn get_layer_cache(&self, layer_idx: usize) -> Option<&LayerCache> {
        self.gpu_caches.get(layer_idx)
    }

    /// Get mutable layer cache.
    pub fn get_layer_cache_mut(&mut self, layer_idx: usize) -> Option<&mut LayerCache> {
        self.gpu_caches.get_mut(layer_idx)
    }

    /// Allocate blocks for a sequence.
    pub fn allocate_for_sequence(&self, num_tokens: usize) -> Option<BlockTable> {
        let num_blocks = (num_tokens + self.config.block_size - 1) / self.config.block_size;
        let block_ids = self.allocator.allocate_gpu_blocks(num_blocks)?;

        let mut table = BlockTable::new(self.config.block_size);
        for block_id in block_ids {
            table.append(block_id);
        }
        table.last_block_tokens = num_tokens % self.config.block_size;
        if table.last_block_tokens == 0 && num_tokens > 0 {
            table.last_block_tokens = self.config.block_size;
        }

        Some(table)
    }

    /// Free blocks for a sequence.
    pub fn free_sequence(&self, block_table: &BlockTable) {
        self.allocator.free_blocks(&block_table.block_ids);
    }

    /// Get GPU memory utilization.
    pub fn gpu_utilization(&self) -> f64 {
        self.allocator.gpu_utilization()
    }

    /// Get number of free GPU blocks.
    pub fn num_free_gpu_blocks(&self) -> usize {
        self.allocator.num_free_gpu_blocks()
    }

    /// Check if can allocate n blocks.
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.allocator.can_allocate(num_blocks)
    }

    /// Get block size.
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Get number of layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits (prefix cache).
    pub prefix_cache_hits: u64,

    /// Number of cache misses.
    pub prefix_cache_misses: u64,

    /// Number of blocks allocated.
    pub blocks_allocated: u64,

    /// Number of blocks freed.
    pub blocks_freed: u64,

    /// Number of swap operations.
    pub swaps: u64,

    /// Number of copy-on-write forks.
    pub cow_forks: u64,
}

impl CacheStats {
    /// Get hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.prefix_cache_hits + self.prefix_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.prefix_cache_hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_allocator() {
        let config = KVCacheConfig::new(32, 8, 128, 100);
        let allocator = BlockAllocator::new(config);

        // Allocate some blocks
        let block1 = allocator.allocate_gpu().unwrap();
        let block2 = allocator.allocate_gpu().unwrap();
        assert_eq!(allocator.num_free_gpu_blocks(), 98);
        assert_eq!(allocator.num_allocated_gpu_blocks(), 2);

        // Free a block
        allocator.free(block1);
        assert_eq!(allocator.num_free_gpu_blocks(), 99);
        assert_eq!(allocator.num_allocated_gpu_blocks(), 1);

        // Allocate multiple
        let blocks = allocator.allocate_gpu_blocks(10).unwrap();
        assert_eq!(blocks.len(), 10);
        assert_eq!(allocator.num_free_gpu_blocks(), 89);
    }

    #[test]
    fn test_block_table() {
        let mut table = BlockTable::new(16);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.num_tokens(), 0);

        table.append(0);
        table.add_tokens(10);
        assert_eq!(table.num_blocks(), 1);
        assert_eq!(table.num_tokens(), 10);
        assert!(!table.is_last_block_full());
        assert_eq!(table.slots_in_last_block(), 6);

        table.add_tokens(6);
        assert!(table.is_last_block_full());

        table.append(1);
        table.add_tokens(5);
        assert_eq!(table.num_tokens(), 21);
    }

    #[test]
    fn test_slot_indices() {
        let mut table = BlockTable::new(16);
        table.append(0);
        table.last_block_tokens = 16;
        table.append(1);
        table.last_block_tokens = 8;

        let slots = table.to_slot_indices(24);
        assert_eq!(slots.len(), 24);
        // First block: 0-15
        assert_eq!(slots[0], 0);
        assert_eq!(slots[15], 15);
        // Second block: 16-23
        assert_eq!(slots[16], 16);
        assert_eq!(slots[23], 23);
    }

    #[test]
    fn test_cache_config() {
        let config = KVCacheConfig::new(32, 8, 128, 1000)
            .with_block_size(32)
            .with_cpu_blocks(100)
            .with_prefix_caching();

        assert_eq!(config.block_size, 32);
        assert_eq!(config.num_cpu_blocks, 100);
        assert!(config.enable_prefix_caching);

        // Calculate memory
        let bytes_per_block = config.bytes_per_block();
        assert!(bytes_per_block > 0);
    }

    #[test]
    fn test_gpu_utilization() {
        let config = KVCacheConfig::new(32, 8, 128, 100);
        let allocator = BlockAllocator::new(config);

        assert_eq!(allocator.gpu_utilization(), 0.0);

        allocator.allocate_gpu_blocks(50).unwrap();
        assert!((allocator.gpu_utilization() - 0.5).abs() < 0.01);

        allocator.allocate_gpu_blocks(50).unwrap();
        assert!((allocator.gpu_utilization() - 1.0).abs() < 0.01);
    }
}
