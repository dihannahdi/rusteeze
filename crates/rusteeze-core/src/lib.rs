//! # Rusteeze Core
//!
//! Core types, traits, and utilities for the Rusteeze LLM inference engine.
//!
//! This crate provides the foundational abstractions used throughout Rusteeze:
//!
//! - **Error handling**: Comprehensive error types with context
//! - **Configuration**: Type-safe configuration structures
//! - **Request/Response types**: OpenAI-compatible API types
//! - **Sequence management**: Token sequences and attention masks
//! - **Device abstraction**: CPU/GPU device handling
//!
//! ## Example
//!
//! ```rust
//! use rusteeze_core::{
//!     error::Result,
//!     request::CompletionRequest,
//!     response::CompletionResponse,
//! };
//!
//! fn process_request(req: CompletionRequest) -> Result<CompletionResponse> {
//!     // Process the request
//!     todo!()
//! }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]
#![deny(unsafe_code)]

pub mod config;
pub mod device;
pub mod error;
pub mod request;
pub mod response;
pub mod sampling;
pub mod sequence;
pub mod types;

pub use config::*;
pub use device::*;
pub use error::{Error, Result};
pub use request::*;
pub use response::*;
pub use sampling::*;
pub use sequence::*;
pub use types::*;

/// Re-export commonly used external types
pub mod prelude {
    pub use crate::config::*;
    pub use crate::device::*;
    pub use crate::error::{Error, Result};
    pub use crate::request::*;
    pub use crate::response::*;
    pub use crate::sampling::*;
    pub use crate::sequence::*;
    pub use crate::types::*;

    pub use half::{bf16, f16};
    pub use smallvec::SmallVec;
    pub use uuid::Uuid;
}
