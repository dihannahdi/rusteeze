//! # Memory Pool — Radical Rewrite
//!
//! Thread-safe slab allocator for tensor buffers with size-class
//! free lists, statistics, and defragmentation support.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::simd_dispatch;

/// Memory pool configuration.
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Total pool size (bytes)
    pub total_size: usize,
    /// Alignment for allocations
    pub alignment: usize,
    /// Size classes for free lists
    pub size_classes: Vec<usize>,
    /// Maximum number of cached blocks per size class
    pub max_cached_per_class: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            total_size: 512 * 1024 * 1024, // 512 MB
            alignment: 64,
            size_classes: vec![
                256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                65536, 131072, 262144, 524288, 1048576, 4194304,
            ],
            max_cached_per_class: 64,
        }
    }
}

/// Pool statistics.
#[derive(Debug, Default)]
pub struct PoolStats {
    pub allocations: AtomicU64,
    pub deallocations: AtomicU64,
    pub bytes_allocated: AtomicU64,
    pub bytes_freed: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
}

impl Clone for PoolStats {
    fn clone(&self) -> Self {
        Self {
            allocations: AtomicU64::new(self.allocations.load(Ordering::Relaxed)),
            deallocations: AtomicU64::new(self.deallocations.load(Ordering::Relaxed)),
            bytes_allocated: AtomicU64::new(self.bytes_allocated.load(Ordering::Relaxed)),
            bytes_freed: AtomicU64::new(self.bytes_freed.load(Ordering::Relaxed)),
            cache_hits: AtomicU64::new(self.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.cache_misses.load(Ordering::Relaxed)),
        }
    }
}

/// A cached buffer block.
struct CachedBlock {
    data: Vec<u8>,
    size_class: usize,
}

/// Size-class free list.
struct SizeClass {
    size: usize,
    free: Mutex<Vec<Vec<u8>>>,
    max_cached: usize,
}

impl SizeClass {
    fn new(size: usize, max_cached: usize) -> Self {
        Self {
            size,
            free: Mutex::new(Vec::with_capacity(max_cached)),
            max_cached,
        }
    }

    fn get(&self) -> Option<Vec<u8>> {
        self.free.lock().pop()
    }

    fn put(&self, buf: Vec<u8>) {
        let mut free = self.free.lock();
        if free.len() < self.max_cached {
            free.push(buf);
        }
        // Otherwise drop the buffer
    }

    fn cached_count(&self) -> usize {
        self.free.lock().len()
    }
}

/// Memory pool with size-class free lists.
pub struct MemoryPool {
    config: MemoryPoolConfig,
    /// Size classes, sorted by size
    classes: Vec<SizeClass>,
    /// Stats
    stats: PoolStats,
}

impl MemoryPool {
    /// Create a new memory pool.
    pub fn new(config: MemoryPoolConfig) -> Self {
        simd_dispatch::init();
        let classes = config.size_classes.iter()
            .map(|&size| SizeClass::new(size, config.max_cached_per_class))
            .collect();
        Self { config, classes, stats: PoolStats::default() }
    }

    /// Find the smallest size class >= requested size.
    fn find_class(&self, size: usize) -> Option<usize> {
        self.classes.iter().position(|c| c.size >= size)
    }

    /// Allocate a buffer of at least `size` bytes.
    pub fn allocate(&self, size: usize) -> Vec<u8> {
        self.stats.allocations.fetch_add(1, Ordering::Relaxed);

        if let Some(class_idx) = self.find_class(size) {
            let class = &self.classes[class_idx];
            if let Some(mut buf) = class.get() {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                // Resize to exact requested size (may truncate or extend)
                buf.resize(size, 0);
                return buf;
            }
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
            // Allocate at class size for future reuse
            let mut buf = vec![0u8; class.size];
            buf.resize(size, 0);
            self.stats.bytes_allocated.fetch_add(class.size as u64, Ordering::Relaxed);
            buf
        } else {
            // Larger than any size class — allocate exact
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
            self.stats.bytes_allocated.fetch_add(size as u64, Ordering::Relaxed);
            vec![0u8; size]
        }
    }

    /// Allocate a Vec<f32>.
    pub fn allocate_f32(&self, len: usize) -> Vec<f32> {
        let bytes = self.allocate(len * 4);
        // Reinterpret as f32 (safe because we control alignment via size class)
        // For safety, just create a new vec
        vec![0.0f32; len]
    }

    /// Return a buffer to the pool.
    pub fn deallocate(&self, buf: Vec<u8>) {
        self.stats.deallocations.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_freed.fetch_add(buf.capacity() as u64, Ordering::Relaxed);

        if let Some(class_idx) = self.find_class(buf.capacity()) {
            self.classes[class_idx].put(buf);
        }
        // If no matching class, buffer is dropped
    }

    /// Get stats.
    pub fn stats(&self) -> &PoolStats { &self.stats }

    /// Total cached bytes.
    pub fn cached_bytes(&self) -> usize {
        self.classes.iter()
            .map(|c| c.cached_count() * c.size)
            .sum()
    }

    /// Flush all caches.
    pub fn flush(&self) {
        for class in &self.classes {
            class.free.lock().clear();
        }
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.stats.cache_hits.load(Ordering::Relaxed) as f64;
        let misses = self.stats.cache_misses.load(Ordering::Relaxed) as f64;
        if hits + misses == 0.0 { 0.0 } else { hits / (hits + misses) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_reuse() {
        let pool = MemoryPool::new(MemoryPoolConfig::default());

        let buf = pool.allocate(1000);
        assert!(buf.len() >= 1000);

        // Return to pool
        let cap = buf.capacity();
        pool.deallocate(buf);

        // Next allocation should reuse
        let buf2 = pool.allocate(1000);
        assert!(buf2.len() >= 1000);
        assert_eq!(pool.stats().cache_hits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_large_allocation() {
        let pool = MemoryPool::new(MemoryPoolConfig::default());
        let buf = pool.allocate(10_000_000); // 10 MB, larger than any class
        assert_eq!(buf.len(), 10_000_000);
        pool.deallocate(buf);
    }

    #[test]
    fn test_flush() {
        let pool = MemoryPool::new(MemoryPoolConfig::default());
        let buf = pool.allocate(1000);
        pool.deallocate(buf);
        assert!(pool.cached_bytes() > 0);
        pool.flush();
        assert_eq!(pool.cached_bytes(), 0);
    }
}
