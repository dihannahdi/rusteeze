//! # Rusteeze Tensor
//!
//! High-performance tensor operations for the Rusteeze LLM inference engine.
//!
//! This crate provides:
//!
//! - **Tensor operations**: Optimized matrix operations using Candle
//! - **Paged Attention**: Memory-efficient attention mechanisms
//! - **KV Cache**: Key-value cache management for autoregressive generation
//! - **Quantization**: Support for various quantization schemes
//! - **Memory management**: Efficient GPU memory allocation and pooling
//!
//! ## Features
//!
//! - `cuda` - Enable CUDA GPU support
//! - `metal` - Enable Apple Metal GPU support
//! - `accelerate` - Enable Apple Accelerate framework
//! - `mkl` - Enable Intel MKL support

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod attention;
pub mod cache;
pub mod ops;
pub mod quantize;
pub mod memory;

pub use attention::*;
pub use cache::*;
pub use ops::*;
pub use quantize::*;
pub use memory::*;

/// Re-export candle types for convenience
pub mod candle {
    pub use candle_core::{Device, DType, Tensor, Shape, Error as CandleError};
    pub use candle_nn::{Module, VarBuilder, VarMap};
}

/// Prelude for common imports
pub mod prelude {
    pub use super::candle::*;
    pub use super::attention::*;
    pub use super::cache::*;
    pub use super::ops::*;
    pub use super::quantize::*;
    pub use super::memory::*;
}
