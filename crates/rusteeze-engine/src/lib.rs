//! # Rusteeze Engine
//!
//! High-performance LLM inference engine with continuous batching,
//! PagedAttention, and speculative decoding.
//!
//! ## Architecture
//!
//! The engine consists of several key components:
//!
//! - **Scheduler**: Manages request queues and continuous batching
//! - **Worker**: Executes model inference
//! - **Sampler**: Handles token sampling strategies
//! - **Block Manager**: Manages KV cache memory with paged attention
//! - **SIMD Ops**: Vectorized operations for CPU-bound sampling
//! - **Memory Pool**: Arena-style allocation for reduced overhead
//! - **Batch Optimizer**: Dynamic batching for maximum throughput
//! - **Parallel Sampler**: Multi-threaded sampling with rayon
//!
//! ## Example
//!
//! ```rust,ignore
//! use rusteeze_engine::{Engine, EngineConfig};
//!
//! let config = EngineConfig::default();
//! let engine = Engine::new(config).await?;
//!
//! let response = engine.generate("Hello, world!").await?;
//! ```

#![warn(missing_docs)]
// Note: unsafe_code is allowed only in simd_ops and memory_pool modules

pub mod scheduler;
pub mod sampler;
pub mod worker;
pub mod block_manager;
pub mod sequence;
pub mod engine;
pub mod batch;

#[allow(unsafe_code)]
pub mod simd_ops;

#[allow(unsafe_code)]
pub mod memory_pool;

pub mod batch_optimizer;
pub mod parallel_sampler;

pub use scheduler::*;
pub use sampler::*;
pub use worker::*;
pub use block_manager::*;
pub use sequence::*;
pub use engine::*;
pub use batch::*;
pub use memory_pool::*;
pub use batch_optimizer::*;
pub use parallel_sampler::*;

/// Prelude for common imports
pub mod prelude {
    pub use super::engine::*;
    pub use super::scheduler::*;
    pub use super::sampler::*;
}
