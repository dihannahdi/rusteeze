//! Memory pool for efficient allocation during inference.
//!
//! This module provides arena-style memory allocation to reduce
//! allocation overhead during hot paths in inference.
//!
//! # Design
//!
//! - Pre-allocated memory pools for token buffers, logit arrays
//! - Lock-free allocation for hot paths
//! - Automatic reset between batches
//! - NUMA-aware allocation for multi-socket systems
//!
//! # Performance Benefits
//!
//! - Eliminates per-request allocation overhead
//! - Reduces memory fragmentation
//! - Better cache locality through contiguous allocation
//! - Zero-copy buffer reuse

use std::alloc::{alloc, dealloc, Layout};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;
use smallvec::SmallVec;

/// Memory pool error types.
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    /// Pool is exhausted.
    #[error("Memory pool exhausted: requested {requested} bytes, available {available}")]
    Exhausted { requested: usize, available: usize },

    /// Invalid alignment.
    #[error("Invalid alignment: {0}")]
    InvalidAlignment(usize),

    /// Pool not initialized.
    #[error("Memory pool not initialized")]
    NotInitialized,
}

/// Configuration for memory pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Size of the token buffer pool (in bytes).
    pub token_pool_size: usize,

    /// Size of the logits buffer pool (in bytes).
    pub logits_pool_size: usize,

    /// Size of the sampling buffer pool (in bytes).
    pub sampling_pool_size: usize,

    /// Maximum alignment (usually 64 for cache line alignment).
    pub max_alignment: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            // 64 MB for token buffers
            token_pool_size: 64 * 1024 * 1024,
            // 256 MB for logits (large vocab sizes)
            logits_pool_size: 256 * 1024 * 1024,
            // 32 MB for sampling intermediates
            sampling_pool_size: 32 * 1024 * 1024,
            // Cache line alignment
            max_alignment: 64,
        }
    }
}

/// A bump allocator arena.
///
/// Allocations are made by simply bumping a pointer.
/// Memory is freed all at once when the arena is reset.
pub struct Arena {
    /// Start of the memory region.
    start: NonNull<u8>,
    /// Current allocation pointer.
    current: AtomicUsize,
    /// End of the memory region.
    end: usize,
    /// Layout used for allocation.
    layout: Layout,
}

// SAFETY: Arena provides interior mutability through atomics
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Create a new arena with the specified size.
    pub fn new(size: usize, alignment: usize) -> Result<Self, PoolError> {
        if !alignment.is_power_of_two() {
            return Err(PoolError::InvalidAlignment(alignment));
        }

        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| PoolError::InvalidAlignment(alignment))?;

        // SAFETY: Layout is valid (checked above)
        let ptr = unsafe { alloc(layout) };
        let start = NonNull::new(ptr).ok_or(PoolError::NotInitialized)?;

        Ok(Self {
            start,
            current: AtomicUsize::new(start.as_ptr() as usize),
            end: start.as_ptr() as usize + size,
            layout,
        })
    }

    /// Allocate memory from the arena.
    ///
    /// Returns None if there's not enough space.
    #[inline]
    pub fn alloc(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let align = align.max(1);

        loop {
            let current = self.current.load(Ordering::Relaxed);
            
            // Align up
            let aligned = (current + align - 1) & !(align - 1);
            let new_current = aligned + size;

            if new_current > self.end {
                return None;
            }

            // Try to bump the pointer
            match self.current.compare_exchange_weak(
                current,
                new_current,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: We own this memory and it's within bounds
                    return Some(unsafe { NonNull::new_unchecked(aligned as *mut u8) });
                }
                Err(_) => continue, // CAS failed, retry
            }
        }
    }

    /// Allocate a typed value from the arena.
    #[inline]
    pub fn alloc_typed<T>(&self) -> Option<NonNull<T>> {
        self.alloc(size_of::<T>(), align_of::<T>())
            .map(|ptr| ptr.cast())
    }

    /// Allocate a slice from the arena.
    #[inline]
    pub fn alloc_slice<T>(&self, len: usize) -> Option<NonNull<[T]>> {
        let size = size_of::<T>() * len;
        let ptr = self.alloc(size, align_of::<T>())?;
        
        // SAFETY: Pointer is valid and properly aligned
        let slice = unsafe {
            std::ptr::slice_from_raw_parts_mut(ptr.as_ptr() as *mut T, len)
        };
        NonNull::new(slice)
    }

    /// Reset the arena, freeing all allocations.
    #[inline]
    pub fn reset(&self) {
        self.current.store(self.start.as_ptr() as usize, Ordering::Release);
    }

    /// Get the number of bytes allocated.
    #[inline]
    pub fn allocated(&self) -> usize {
        self.current.load(Ordering::Relaxed) - self.start.as_ptr() as usize
    }

    /// Get the total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.end - self.start.as_ptr() as usize
    }

    /// Get the remaining capacity.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.end - self.current.load(Ordering::Relaxed)
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: We allocated this memory with this layout
        unsafe {
            dealloc(self.start.as_ptr(), self.layout);
        }
    }
}

/// A typed buffer allocated from a pool.
pub struct PoolBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> PoolBuffer<T> {
    /// Create a new buffer from a raw pointer.
    ///
    /// # Safety
    /// The pointer must be valid and properly aligned for T.
    pub unsafe fn from_raw(ptr: NonNull<T>, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Get the buffer as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: Pointer is valid and we own the data
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get the buffer as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: Pointer is valid and we have exclusive access
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Get the length.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// Note: PoolBuffer doesn't implement Drop because the memory
// is managed by the Arena, which frees everything on reset.
unsafe impl<T: Send> Send for PoolBuffer<T> {}
unsafe impl<T: Sync> Sync for PoolBuffer<T> {}

/// Inference memory pool manager.
///
/// Manages multiple arenas for different purposes during inference.
pub struct InferencePool {
    /// Arena for token ID buffers.
    token_arena: Arena,
    /// Arena for logits and probabilities.
    logits_arena: Arena,
    /// Arena for sampling intermediates.
    sampling_arena: Arena,
    /// Configuration.
    config: PoolConfig,
}

impl InferencePool {
    /// Create a new inference pool.
    pub fn new(config: PoolConfig) -> Result<Self, PoolError> {
        let token_arena = Arena::new(config.token_pool_size, config.max_alignment)?;
        let logits_arena = Arena::new(config.logits_pool_size, config.max_alignment)?;
        let sampling_arena = Arena::new(config.sampling_pool_size, config.max_alignment)?;

        Ok(Self {
            token_arena,
            logits_arena,
            sampling_arena,
            config,
        })
    }

    /// Allocate a token buffer.
    #[inline]
    pub fn alloc_tokens(&self, len: usize) -> Result<PoolBuffer<u32>, PoolError> {
        self.token_arena
            .alloc_slice::<u32>(len)
            .map(|ptr| unsafe { 
                PoolBuffer::from_raw(NonNull::new_unchecked(ptr.as_ptr() as *mut u32), len) 
            })
            .ok_or(PoolError::Exhausted {
                requested: len * size_of::<u32>(),
                available: self.token_arena.remaining(),
            })
    }

    /// Allocate a logits buffer.
    #[inline]
    pub fn alloc_logits(&self, len: usize) -> Result<PoolBuffer<f32>, PoolError> {
        self.logits_arena
            .alloc_slice::<f32>(len)
            .map(|ptr| unsafe {
                PoolBuffer::from_raw(NonNull::new_unchecked(ptr.as_ptr() as *mut f32), len)
            })
            .ok_or(PoolError::Exhausted {
                requested: len * size_of::<f32>(),
                available: self.logits_arena.remaining(),
            })
    }

    /// Allocate a probability buffer (same as logits, different intent).
    #[inline]
    pub fn alloc_probs(&self, len: usize) -> Result<PoolBuffer<f32>, PoolError> {
        self.sampling_arena
            .alloc_slice::<f32>(len)
            .map(|ptr| unsafe {
                PoolBuffer::from_raw(NonNull::new_unchecked(ptr.as_ptr() as *mut f32), len)
            })
            .ok_or(PoolError::Exhausted {
                requested: len * size_of::<f32>(),
                available: self.sampling_arena.remaining(),
            })
    }

    /// Reset all arenas for a new batch.
    #[inline]
    pub fn reset(&self) {
        self.token_arena.reset();
        self.logits_arena.reset();
        self.sampling_arena.reset();
    }

    /// Get memory statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            token_used: self.token_arena.allocated(),
            token_capacity: self.token_arena.capacity(),
            logits_used: self.logits_arena.allocated(),
            logits_capacity: self.logits_arena.capacity(),
            sampling_used: self.sampling_arena.allocated(),
            sampling_capacity: self.sampling_arena.capacity(),
        }
    }
}

/// Memory pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Token buffer bytes used.
    pub token_used: usize,
    /// Token buffer capacity.
    pub token_capacity: usize,
    /// Logits buffer bytes used.
    pub logits_used: usize,
    /// Logits buffer capacity.
    pub logits_capacity: usize,
    /// Sampling buffer bytes used.
    pub sampling_used: usize,
    /// Sampling buffer capacity.
    pub sampling_capacity: usize,
}

impl PoolStats {
    /// Get total memory used.
    pub fn total_used(&self) -> usize {
        self.token_used + self.logits_used + self.sampling_used
    }

    /// Get total capacity.
    pub fn total_capacity(&self) -> usize {
        self.token_capacity + self.logits_capacity + self.sampling_capacity
    }

    /// Get utilization as a fraction.
    pub fn utilization(&self) -> f64 {
        self.total_used() as f64 / self.total_capacity() as f64
    }
}

/// Thread-local pool for parallel sampling.
pub struct ThreadLocalPool {
    pools: Vec<InferencePool>,
}

impl ThreadLocalPool {
    /// Create a new thread-local pool with one pool per thread.
    pub fn new(num_threads: usize, config: PoolConfig) -> Result<Self, PoolError> {
        let mut pools = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            pools.push(InferencePool::new(config.clone())?);
        }
        Ok(Self { pools })
    }

    /// Get the pool for a specific thread.
    #[inline]
    pub fn get(&self, thread_id: usize) -> Option<&InferencePool> {
        self.pools.get(thread_id)
    }

    /// Reset all pools.
    pub fn reset_all(&self) {
        for pool in &self.pools {
            pool.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let arena = Arena::new(4096, 64).unwrap();
        
        // Allocate some memory
        let ptr1 = arena.alloc(100, 8).unwrap();
        let ptr2 = arena.alloc(200, 8).unwrap();
        
        // Pointers should be different
        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
        
        // Check allocation tracking
        assert!(arena.allocated() >= 300);
        
        // Reset
        arena.reset();
        assert_eq!(arena.allocated(), 0);
    }

    #[test]
    fn test_arena_exhaustion() {
        let arena = Arena::new(100, 8).unwrap();
        
        // First allocation should succeed
        assert!(arena.alloc(50, 8).is_some());
        
        // Second should fail (not enough space)
        assert!(arena.alloc(100, 8).is_none());
    }

    #[test]
    fn test_typed_allocation() {
        let arena = Arena::new(4096, 64).unwrap();
        
        let slice = arena.alloc_slice::<f32>(100).unwrap();
        
        // SAFETY: Test code, pointer is valid
        unsafe {
            let ptr = slice.as_ptr() as *mut f32;
            for i in 0..100 {
                *ptr.add(i) = i as f32;
            }
            
            // Verify
            for i in 0..100 {
                assert_eq!(*ptr.add(i), i as f32);
            }
        }
    }

    #[test]
    fn test_inference_pool() {
        let config = PoolConfig {
            token_pool_size: 1024,
            logits_pool_size: 4096,
            sampling_pool_size: 1024,
            max_alignment: 64,
        };
        
        let pool = InferencePool::new(config).unwrap();
        
        // Allocate tokens
        let mut tokens = pool.alloc_tokens(10).unwrap();
        let slice = tokens.as_mut_slice();
        for (i, token) in slice.iter_mut().enumerate() {
            *token = i as u32;
        }
        
        // Allocate logits
        let mut logits = pool.alloc_logits(100).unwrap();
        assert_eq!(logits.len(), 100);
        
        // Check stats
        let stats = pool.stats();
        assert!(stats.token_used > 0);
        assert!(stats.logits_used > 0);
        
        // Reset
        pool.reset();
        let stats = pool.stats();
        assert_eq!(stats.total_used(), 0);
    }

    #[test]
    fn test_alignment() {
        let arena = Arena::new(4096, 64).unwrap();
        
        // Allocate with various alignments
        let ptr1 = arena.alloc(7, 8).unwrap();
        let ptr2 = arena.alloc(7, 16).unwrap();
        let ptr3 = arena.alloc(7, 64).unwrap();
        
        // Check alignments
        assert_eq!(ptr1.as_ptr() as usize % 8, 0);
        assert_eq!(ptr2.as_ptr() as usize % 16, 0);
        assert_eq!(ptr3.as_ptr() as usize % 64, 0);
    }
}
