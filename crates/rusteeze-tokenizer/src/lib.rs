//! # Rusteeze Tokenizer
//!
//! High-performance tokenization for the Rusteeze LLM inference engine.
//!
//! This crate provides a unified interface to HuggingFace tokenizers with
//! additional features for LLM inference:
//!
//! - **Batch encoding**: Efficient parallel tokenization
//! - **Special tokens**: Automatic chat template handling
//! - **Streaming**: Incremental decoding for streaming inference
//!
//! ## Example
//!
//! ```rust,ignore
//! use rusteeze_tokenizer::Tokenizer;
//!
//! let tokenizer = Tokenizer::from_file("tokenizer.json")?;
//! let encoding = tokenizer.encode("Hello, world!")?;
//! println!("Tokens: {:?}", encoding.ids());
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

mod tokenizer;
mod chat_template;
mod error;

pub use tokenizer::*;
pub use chat_template::*;
pub use error::*;

/// Prelude for common imports
pub mod prelude {
    pub use super::tokenizer::*;
    pub use super::chat_template::*;
    pub use super::error::*;
}
