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
#![deny(unsafe_code)]

pub mod scheduler;
pub mod sampler;
pub mod worker;
pub mod block_manager;
pub mod sequence;
pub mod engine;
pub mod batch;

pub use scheduler::*;
pub use sampler::*;
pub use worker::*;
pub use block_manager::*;
pub use sequence::*;
pub use engine::*;
pub use batch::*;

/// Prelude for common imports
pub mod prelude {
    pub use super::engine::*;
    pub use super::scheduler::*;
    pub use super::sampler::*;
}
