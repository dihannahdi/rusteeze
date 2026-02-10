//! # Rusteeze Engine
//!
//! High-performance LLM inference engine with continuous batching,
//! PagedAttention, speculative decoding, and Recursive Language Model (RLM) inference.
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
//! ## Recursive Language Model (RLM) Engine
//!
//! Based on ["Recursive Language Models" (Zhang, Kraska, Khattab 2026)](https://arxiv.org/abs/2512.24601):
//!
//! - **VariableStore**: Persistent REPL-like variable management
//! - **PromptEnvironment**: REPL state where prompts are variables, not context
//! - **RecursiveScheduler**: Tree-structured recursive call management
//! - **RecursiveInferenceEngine**: Main RLM loop (Algorithm 1 from the paper)
//!
//! ## Advanced Features (50x Performance Optimizations)
//!
//! - **Kernel Fusion**: Fused LayerNorm + Linear + GELU operations
//! - **Flash Attention v2**: O(N) memory attention with tiling
//! - **Zero-Copy Pipeline**: Lock-free token processing
//! - **Speculative Decoding**: 3-5x throughput with draft models
//! - **Continuous Batching v2**: Prefix caching and KV sharing
//! - **Advanced Quantization**: INT4/INT8/FP8 with GPTQ/AWQ
//!
//! ## Example
//!
//! ```rust,ignore
//! use rusteeze_engine::{Engine, EngineConfig};
//! use rusteeze_engine::{RecursiveInferenceEngine, RecursiveEngineConfig, RecursiveRequest};
//!
//! // Standard inference
//! let config = EngineConfig::default();
//! let engine = Engine::new(config).await?;
//! let response = engine.generate("Hello, world!").await?;
//!
//! // Recursive inference (handles 10M+ token prompts)
//! let rlm = RecursiveInferenceEngine::new(RecursiveEngineConfig::default());
//! let req = RecursiveRequest { /* ... */ };
//! let result = rlm.process(&req, &model)?;
//! ```

// Note: unsafe_code is allowed in performance-critical modules
#![allow(unused_imports, dead_code, missing_docs)]

// ─── Foundation: SIMD dispatch (resolved once at startup) ───
#[allow(unsafe_code)]
pub mod simd_dispatch;

// ─── Core modules ───
pub mod scheduler;
pub mod sampler;
pub mod worker;
pub mod block_manager;
pub mod sequence;
pub mod engine;
pub mod batch;

// ─── Performance-critical modules ───
#[allow(unsafe_code)]
pub mod simd_ops;

#[allow(unsafe_code)]
pub mod memory_pool;

#[allow(unsafe_code)]
pub mod kernel_fusion;

#[allow(unsafe_code)]
pub mod flash_attention;

#[allow(unsafe_code)]
pub mod zero_copy;

#[allow(unsafe_code)]
pub mod quantization;

// ─── Advanced features ───
pub mod batch_optimizer;
pub mod parallel_sampler;
pub mod speculative;
pub mod batching_v2;

// ─── Infrastructure ───
#[allow(unsafe_code)]
pub mod numa;

pub mod tensor_parallel;
pub mod router;

// ─── Recursive Language Model (RLM) ───
pub mod variable_store;
pub mod prompt_env;
pub mod recursive_scheduler;
pub mod recursive_engine;

// ─── Re-exports ───
pub use engine::{Engine, EngineConfig, EngineError, EngineStats, GenerationRequest, GenerationOutput, StreamChunk};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerOutput};
pub use sampler::{Sampler, SampleResult};
pub use worker::{Worker, WorkerConfig, WorkBatch, WorkResult};
pub use block_manager::{BlockManager, BlockManagerConfig, BlockTable, PhysicalBlockId};
pub use memory_pool::{MemoryPool, MemoryPoolConfig, PoolStats};
pub use batch_optimizer::{BatchOptimizer, BatchOptimizerConfig, BatchRequest, OptimizedBatch};
pub use parallel_sampler::{ParallelSampler, ParallelSamplerConfig};

// Advanced feature re-exports
pub use kernel_fusion::{KernelFusion, FusionConfig, FusionStats};
pub use flash_attention::{FlashAttention, FlashAttentionConfig, GroupedQueryAttention};
pub use zero_copy::{ZeroCopyPipeline, ZeroCopyConfig, TokenRingBuffer};
pub use speculative::{SpeculativeEngine, SpeculativeConfig, SpeculationTree};
pub use batching_v2::{IterationScheduler, BatchingV2Config, PrefixCache};
pub use quantization::{QuantEngine, AdvancedQuantConfig, QuantMethod, QuantizedTensor};
pub use numa::{NumaMemoryManager, NumaConfig, NumaTopology, HierarchicalPool};
pub use tensor_parallel::{TensorParallelEngine, TensorParallelConfig, Collective, ProcessGroup};
pub use router::{Router, RouterConfig, LoadBalanceStrategy};

// RLM re-exports
pub use variable_store::{VariableStore, VarId, VarValue, VarStoreError};
pub use prompt_env::{PromptEnvironment, PromptEnvConfig, ReplOperation, OpResult, PendingCall, EnvStats};
pub use recursive_scheduler::{RecursiveScheduler, RecursiveSchedulerConfig, CallId, CallStatus, CallNode};
pub use recursive_engine::{
    RecursiveInferenceEngine, RecursiveEngineConfig, RecursiveRequest, RecursiveResponse,
    RecursiveStats, InferenceModel, InferenceResult, InferenceError, RecursiveInferenceError,
};

/// Prelude for common imports
pub mod prelude {
    pub use super::engine::*;
    pub use super::scheduler::{Scheduler, SchedulerConfig};
    pub use super::sampler::{Sampler, SampleResult};
    
    // Advanced features
    pub use super::kernel_fusion::KernelFusion;
    pub use super::flash_attention::FlashAttention;
    pub use super::speculative::SpeculativeEngine;
    pub use super::batching_v2::IterationScheduler;
    pub use super::quantization::QuantEngine;
    pub use super::numa::NumaMemoryManager;
    pub use super::tensor_parallel::TensorParallelEngine;
    pub use super::router::Router;

    // Recursive Language Model
    pub use super::recursive_engine::{RecursiveInferenceEngine, RecursiveEngineConfig, RecursiveRequest};
    pub use super::prompt_env::PromptEnvironment;
    pub use super::variable_store::VariableStore;
}
