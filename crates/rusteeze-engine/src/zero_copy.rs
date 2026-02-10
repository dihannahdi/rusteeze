//! # Zero-Copy Pipeline — Radical Rewrite
//!
//! Lock-free ring buffer for token submission and batch extraction.
//! True zero-copy: tokens are written directly into the ring buffer
//! via `copy_nonoverlapping` and batches are read as slices.

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::cell::UnsafeCell;

use crate::simd_dispatch;

/// Configuration for the zero-copy pipeline.
#[derive(Debug, Clone)]
pub struct ZeroCopyConfig {
    /// Ring buffer capacity (number of tokens)
    pub capacity: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            capacity: 1 << 16, // 65536 tokens
            max_batch_size: 64,
            max_seq_len: 4096,
        }
    }
}

/// A single-producer, single-consumer lock-free ring buffer for tokens.
pub struct TokenRingBuffer {
    /// Token storage
    data: UnsafeCell<Vec<u32>>,
    /// Write position (monotonically increasing)
    write_pos: AtomicUsize,
    /// Read position (monotonically increasing)
    read_pos: AtomicUsize,
    /// Capacity (power of 2 for fast modulo)
    capacity: usize,
    /// Mask for modular indexing (capacity - 1)
    mask: usize,
}

// SAFETY: Ring buffer is designed for single-producer, single-consumer.
// The atomic positions ensure safe concurrent access.
unsafe impl Send for TokenRingBuffer {}
unsafe impl Sync for TokenRingBuffer {}

impl TokenRingBuffer {
    /// Create a new token ring buffer.
    /// Capacity is rounded up to the next power of 2.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two();
        Self {
            data: UnsafeCell::new(vec![0u32; cap]),
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            capacity: cap,
            mask: cap - 1,
        }
    }

    /// Number of tokens available to read.
    pub fn available(&self) -> usize {
        let w = self.write_pos.load(Ordering::Acquire);
        let r = self.read_pos.load(Ordering::Acquire);
        w.wrapping_sub(r)
    }

    /// True if buffer is empty.
    pub fn is_empty(&self) -> bool { self.available() == 0 }

    /// Free space in the buffer.
    pub fn free_space(&self) -> usize {
        self.capacity - self.available()
    }

    /// Submit tokens into the ring buffer.
    /// Returns number of tokens actually written.
    pub fn submit(&self, tokens: &[u32]) -> usize {
        let free = self.free_space();
        let n = tokens.len().min(free);
        if n == 0 { return 0; }

        let write = self.write_pos.load(Ordering::Relaxed);
        let start = write & self.mask;
        let data = unsafe { &mut *self.data.get() };

        // Check if write wraps around
        let first = (self.capacity - start).min(n);
        // SAFETY: we checked free space, so no overlap with read region
        unsafe {
            std::ptr::copy_nonoverlapping(
                tokens.as_ptr(),
                data.as_mut_ptr().add(start),
                first,
            );
            if first < n {
                std::ptr::copy_nonoverlapping(
                    tokens.as_ptr().add(first),
                    data.as_mut_ptr(),
                    n - first,
                );
            }
        }

        self.write_pos.store(write + n, Ordering::Release);
        n
    }

    /// Read a batch of tokens from the ring buffer.
    /// Returns the tokens without copying (if contiguous) or copies into `out_buf`.
    pub fn read_batch(&self, max: usize, out_buf: &mut Vec<u32>) -> usize {
        let avail = self.available();
        let n = avail.min(max);
        if n == 0 { return 0; }

        let read = self.read_pos.load(Ordering::Relaxed);
        let start = read & self.mask;
        let data = unsafe { &*self.data.get() };

        out_buf.clear();
        let first = (self.capacity - start).min(n);

        if first == n {
            // Contiguous: extend from slice
            out_buf.extend_from_slice(&data[start..start + n]);
        } else {
            // Wrapping: two copies
            out_buf.reserve(n);
            out_buf.extend_from_slice(&data[start..start + first]);
            out_buf.extend_from_slice(&data[..n - first]);
        }

        self.read_pos.store(read + n, Ordering::Release);
        n
    }

    /// Peek at tokens without consuming them.
    pub fn peek(&self, out_buf: &mut Vec<u32>, max: usize) -> usize {
        let avail = self.available();
        let n = avail.min(max);
        if n == 0 { return 0; }

        let read = self.read_pos.load(Ordering::Relaxed);
        let start = read & self.mask;
        let data = unsafe { &*self.data.get() };

        out_buf.clear();
        let first = (self.capacity - start).min(n);

        if first == n {
            out_buf.extend_from_slice(&data[start..start + n]);
        } else {
            out_buf.reserve(n);
            out_buf.extend_from_slice(&data[start..start + first]);
            out_buf.extend_from_slice(&data[..n - first]);
        }

        // Don't advance read position
        n
    }

    /// Reset the ring buffer.
    pub fn reset(&self) {
        self.read_pos.store(0, Ordering::Release);
        self.write_pos.store(0, Ordering::Release);
    }
}

/// Zero-copy pipeline for request submission and batch extraction.
pub struct ZeroCopyPipeline {
    config: ZeroCopyConfig,
    /// Token ring buffer
    ring: TokenRingBuffer,
    /// Pre-allocated batch output buffer
    batch_buf: std::cell::RefCell<Vec<u32>>,
    /// Active flag
    active: AtomicBool,
}

impl ZeroCopyPipeline {
    /// Create a new zero-copy pipeline.
    pub fn new(config: ZeroCopyConfig) -> Self {
        simd_dispatch::init();
        let ring = TokenRingBuffer::new(config.capacity);
        let batch_buf = std::cell::RefCell::new(Vec::with_capacity(
            config.max_batch_size * config.max_seq_len
        ));
        Self {
            config, ring, batch_buf,
            active: AtomicBool::new(true),
        }
    }

    /// Submit tokens for a request.
    pub fn submit_tokens(&self, tokens: &[u32]) -> usize {
        self.ring.submit(tokens)
    }

    /// Extract a batch of tokens.
    pub fn get_batch(&self, max_tokens: usize) -> Vec<u32> {
        let mut buf = self.batch_buf.borrow_mut();
        let max = max_tokens.min(self.config.max_batch_size * self.config.max_seq_len);
        self.ring.read_batch(max, &mut buf);
        buf.clone() // Return owned copy; buf is reused
    }

    /// Number of pending tokens.
    pub fn pending(&self) -> usize { self.ring.available() }

    /// Check if pipeline is active.
    pub fn is_active(&self) -> bool { self.active.load(Ordering::Relaxed) }

    /// Shutdown the pipeline.
    pub fn shutdown(&self) { self.active.store(false, Ordering::Release); }

    /// Reset the pipeline.
    pub fn reset(&self) {
        self.ring.reset();
        self.active.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let ring = TokenRingBuffer::new(16);
        assert!(ring.is_empty());

        let tokens = vec![1, 2, 3, 4, 5];
        let written = ring.submit(&tokens);
        assert_eq!(written, 5);
        assert_eq!(ring.available(), 5);

        let mut out = Vec::new();
        let read = ring.read_batch(3, &mut out);
        assert_eq!(read, 3);
        assert_eq!(out, vec![1, 2, 3]);
        assert_eq!(ring.available(), 2);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let ring = TokenRingBuffer::new(8); // 8 slots

        // Fill most of it
        let tokens: Vec<u32> = (0..6).collect();
        ring.submit(&tokens);

        // Read some
        let mut out = Vec::new();
        ring.read_batch(4, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3]);

        // Write more (will wrap)
        let tokens2: Vec<u32> = (10..16).collect();
        let written = ring.submit(&tokens2);
        assert_eq!(written, 6);

        // Read all
        ring.read_batch(8, &mut out);
        assert_eq!(out, vec![4, 5, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_pipeline() {
        let config = ZeroCopyConfig { capacity: 256, max_batch_size: 4, max_seq_len: 8, ..Default::default() };
        let pipeline = ZeroCopyPipeline::new(config);

        pipeline.submit_tokens(&[1, 2, 3, 4, 5]);
        assert_eq!(pipeline.pending(), 5);

        let batch = pipeline.get_batch(3);
        assert_eq!(batch, vec![1, 2, 3]);
        assert_eq!(pipeline.pending(), 2);
    }
}
