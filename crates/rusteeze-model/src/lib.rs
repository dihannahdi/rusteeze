//! # Rusteeze Model
//!
//! Model loading and architecture implementations for the Rusteeze LLM inference engine.
//!
//! This crate provides:
//!
//! - **Model loading**: SafeTensors, GGUF, and other formats
//! - **Architectures**: Llama, Mistral, Qwen, Phi, and more
//! - **Weight management**: Memory-mapped loading, quantization support
//! - **Configuration**: Model configuration parsing and validation
//!
//! ## Supported Architectures
//!
//! - LLaMA (1, 2, 3)
//! - Mistral / Mixtral
//! - Qwen (1.5, 2)
//! - Phi (1, 2, 3)
//! - Gemma
//! - DeepSeek
//!
//! ## Example
//!
//! ```rust,ignore
//! use rusteeze_model::{ModelLoader, ModelConfig};
//!
//! let config = ModelConfig::from_path("model/config.json")?;
//! let model = ModelLoader::load("model/", config)?;
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod config;
pub mod loader;
pub mod architectures;
pub mod weights;

pub use config::*;
pub use loader::*;
pub use architectures::*;
pub use weights::*;

/// Prelude for common imports
pub mod prelude {
    pub use super::config::*;
    pub use super::loader::*;
    pub use super::architectures::{Model, KVCache, LayerKVCache, ModelError};
    pub use super::weights::*;
}
