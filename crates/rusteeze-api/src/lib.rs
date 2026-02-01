//! # Rusteeze API
//!
//! OpenAI-compatible REST API for the Rusteeze LLM inference engine.
//!
//! ## Endpoints
//!
//! - `POST /v1/chat/completions` - Chat completions
//! - `POST /v1/completions` - Text completions  
//! - `GET /v1/models` - List models
//! - `GET /health` - Health check
//! - `GET /metrics` - Prometheus metrics
//!
//! ## Example
//!
//! ```rust,ignore
//! use rusteeze_api::{ApiServer, ApiConfig};
//!
//! let config = ApiConfig::default();
//! let server = ApiServer::new(config, engine).await?;
//! server.serve().await?;
//! ```

#![warn(missing_docs)]

pub mod server;
pub mod routes;
pub mod handlers;
pub mod types;
pub mod middleware;
pub mod error;

pub use server::*;
pub use types::*;
pub use error::*;

/// Prelude for common imports
pub mod prelude {
    pub use super::server::*;
    pub use super::types::*;
}
