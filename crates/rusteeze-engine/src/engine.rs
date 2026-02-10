//! Main engine implementation.
//!
//! The engine orchestrates all components for inference:
//! scheduler, worker, tokenizer, and block manager.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use rusteeze_core::{FinishReason, SamplingParams, TokenUsage};
use rusteeze_model::architectures::Model;
use rusteeze_tokenizer::Tokenizer;

use crate::block_manager::{BlockManager, BlockManagerConfig};
use crate::scheduler::{Scheduler, SchedulerConfig, SchedulerOutput, ManagedSequence, SeqState};
use crate::sequence::{GroupId, SequenceGroup, SequenceId, SequenceStatus};
use crate::worker::{Worker, WorkerConfig, WorkBatch};

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Scheduler configuration.
    pub scheduler: SchedulerConfig,

    /// Block manager configuration.
    pub block_manager: BlockManagerConfig,

    /// Worker configuration.
    pub worker: WorkerConfig,

    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,

    /// Request timeout.
    pub request_timeout: Duration,

    /// Enable streaming.
    pub enable_streaming: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            block_manager: BlockManagerConfig::default(),
            worker: WorkerConfig::default(),
            max_concurrent_requests: 1000,
            request_timeout: Duration::from_secs(600),
            enable_streaming: true,
        }
    }
}

/// Generation request.
#[derive(Debug)]
pub struct GenerationRequest {
    /// Request ID.
    pub request_id: String,

    /// Input prompt.
    pub prompt: String,

    /// Sampling parameters.
    pub sampling_params: SamplingParams,

    /// Maximum tokens to generate.
    pub max_tokens: usize,

    /// Stream responses.
    pub stream: bool,
}

/// Generation output.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    /// Request ID.
    pub request_id: String,

    /// Generated text.
    pub text: String,

    /// Generated token IDs.
    pub token_ids: Vec<u32>,

    /// Finish reason.
    pub finish_reason: Option<FinishReason>,

    /// Token usage.
    pub usage: TokenUsage,

    /// Log probabilities (if requested).
    pub logprobs: Option<Vec<f32>>,
}

/// Streaming output chunk.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Request ID.
    pub request_id: String,

    /// New text.
    pub text: String,

    /// New token ID.
    pub token_id: u32,

    /// Is finished.
    pub is_finished: bool,

    /// Finish reason (if finished).
    pub finish_reason: Option<FinishReason>,
}

/// Engine error types.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Initialization error: {0}")]
    InitError(String),

    #[error("Request error: {0}")]
    RequestError(String),

    #[error("Worker error: {0}")]
    WorkerError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Timeout")]
    Timeout,

    #[error("Cancelled")]
    Cancelled,

    #[error("Engine shutdown")]
    Shutdown,
}

/// Request state tracking.
struct RequestState {
    /// Scheduler sequence ID.
    seq_id: u64,

    /// Prompt tokens.
    prompt_tokens: Vec<u32>,

    /// Response sender.
    response_tx: Option<oneshot::Sender<Result<GenerationOutput, EngineError>>>,

    /// Stream sender.
    stream_tx: Option<mpsc::Sender<StreamChunk>>,

    /// Start time.
    start_time: Instant,

    /// Generated tokens so far.
    generated_tokens: Vec<u32>,

    /// Sampling parameters.
    sampling_params: SamplingParams,

    /// Max tokens.
    max_tokens: usize,
}

/// LLM Inference Engine.
pub struct Engine {
    /// Configuration.
    config: EngineConfig,

    /// Model.
    model: Arc<dyn Model>,

    /// Tokenizer.
    tokenizer: Arc<Tokenizer>,

    /// Scheduler.
    scheduler: Mutex<Scheduler>,

    /// Block manager.
    block_manager: Mutex<BlockManager>,

    /// Worker.
    worker: Mutex<Worker>,

    /// Request states: request_id -> state.
    request_states: RwLock<HashMap<String, RequestState>>,

    /// Seq ID -> request ID mapping.
    seq_to_request: RwLock<HashMap<u64, String>>,

    /// Next sequence ID counter.
    next_seq_id: std::sync::atomic::AtomicU64,

    /// Shutdown flag.
    shutdown: RwLock<bool>,
}

impl Engine {
    /// Create new engine.
    pub fn new(
        config: EngineConfig,
        model: Arc<dyn Model>,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Self, EngineError> {
        let scheduler = Scheduler::new(config.scheduler.clone());
        let block_manager = BlockManager::new(config.block_manager.clone());
        let worker = Worker::new(config.worker.clone());

        info!(
            "Initialized engine: max_seqs={}, max_tokens={}",
            config.scheduler.max_num_seqs,
            config.scheduler.max_num_batched_tokens
        );

        Ok(Self {
            config,
            model,
            tokenizer,
            scheduler: Mutex::new(scheduler),
            block_manager: Mutex::new(block_manager),
            worker: Mutex::new(worker),
            request_states: RwLock::new(HashMap::new()),
            seq_to_request: RwLock::new(HashMap::new()),
            next_seq_id: std::sync::atomic::AtomicU64::new(1),
            shutdown: RwLock::new(false),
        })
    }

    fn alloc_seq_id(&self) -> u64 {
        self.next_seq_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Submit generation request.
    pub async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationOutput, EngineError> {
        {
            if *self.shutdown.read() {
                return Err(EngineError::Shutdown);
            }
        }

        let encoding = self.tokenizer
            .encode(&request.prompt, true)
            .map_err(|e| EngineError::TokenizerError(e.to_string()))?;
        let prompt_tokens = encoding.ids().to_vec();
        let prompt_len = prompt_tokens.len();

        let (tx, rx) = oneshot::channel();
        let seq_id = self.alloc_seq_id();

        // Add to scheduler
        {
            let mut scheduler = self.scheduler.lock();
            scheduler.add_sequence(ManagedSequence {
                seq_id,
                prompt_len,
                generated_len: 0,
                max_tokens: request.max_tokens,
                priority: 1.0,
                state: SeqState::Waiting,
                arrival_time: seq_id,
            });
        }

        // Track request
        {
            let mut states = self.request_states.write();
            states.insert(request.request_id.clone(), RequestState {
                seq_id,
                prompt_tokens,
                response_tx: Some(tx),
                stream_tx: None,
                start_time: Instant::now(),
                generated_tokens: Vec::new(),
                sampling_params: request.sampling_params.clone(),
                max_tokens: request.max_tokens,
            });
        }
        {
            let mut map = self.seq_to_request.write();
            map.insert(seq_id, request.request_id.clone());
        }

        // Run engine step
        self.step().await?;

        // Wait for response
        match tokio::time::timeout(self.config.request_timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(EngineError::Cancelled),
            Err(_) => Err(EngineError::Timeout),
        }
    }

    /// Submit streaming generation request.
    pub async fn generate_stream(
        &self,
        request: GenerationRequest,
    ) -> Result<mpsc::Receiver<StreamChunk>, EngineError> {
        {
            if *self.shutdown.read() {
                return Err(EngineError::Shutdown);
            }
        }

        let encoding = self.tokenizer
            .encode(&request.prompt, true)
            .map_err(|e| EngineError::TokenizerError(e.to_string()))?;
        let prompt_tokens = encoding.ids().to_vec();
        let prompt_len = prompt_tokens.len();

        let (tx, rx) = mpsc::channel(100);
        let seq_id = self.alloc_seq_id();

        {
            let mut scheduler = self.scheduler.lock();
            scheduler.add_sequence(ManagedSequence {
                seq_id,
                prompt_len,
                generated_len: 0,
                max_tokens: request.max_tokens,
                priority: 1.0,
                state: SeqState::Waiting,
                arrival_time: seq_id,
            });
        }

        {
            let mut states = self.request_states.write();
            states.insert(request.request_id.clone(), RequestState {
                seq_id,
                prompt_tokens,
                response_tx: None,
                stream_tx: Some(tx),
                start_time: Instant::now(),
                generated_tokens: Vec::new(),
                sampling_params: request.sampling_params.clone(),
                max_tokens: request.max_tokens,
            });
        }
        {
            let mut map = self.seq_to_request.write();
            map.insert(seq_id, request.request_id.clone());
        }

        Ok(rx)
    }

    /// Run single engine step.
    pub async fn step(&self) -> Result<(), EngineError> {
        let scheduler_output = {
            let mut scheduler = self.scheduler.lock();
            scheduler.schedule().clone()
        };

        if scheduler_output.scheduled_seq_ids.is_empty() {
            return Ok(());
        }

        // Process completed sequences
        for &completed_id in &scheduler_output.completed {
            self.finish_sequence(completed_id).await;
        }

        // Build work batch from scheduler output
        let batch = {
            let states = self.request_states.read();
            let seq_map = self.seq_to_request.read();

            let mut seq_ids = Vec::new();
            let mut input_tokens = Vec::new();
            let mut sampling_params = Vec::new();
            let mut prev_tokens = Vec::new();
            let mut is_prefill = Vec::new();

            for (i, &sid) in scheduler_output.scheduled_seq_ids.iter().enumerate() {
                if let Some(request_id) = seq_map.get(&sid) {
                    if let Some(state) = states.get(request_id) {
                        seq_ids.push(sid);

                        let prefill = i < scheduler_output.is_prefill.len() && scheduler_output.is_prefill[i];
                        is_prefill.push(prefill);

                        if prefill {
                            input_tokens.extend_from_slice(&state.prompt_tokens);
                        } else if let Some(&last) = state.generated_tokens.last() {
                            input_tokens.push(last);
                        } else if let Some(&last) = state.prompt_tokens.last() {
                            input_tokens.push(last);
                        }

                        sampling_params.push(state.sampling_params.clone());
                        prev_tokens.push(state.generated_tokens.clone());
                    }
                }
            }

            WorkBatch {
                seq_ids,
                input_tokens,
                sampling_params,
                prev_tokens,
                is_prefill,
            }
        };

        if batch.seq_ids.is_empty() {
            return Ok(());
        }

        // Execute worker step
        let result = {
            let mut worker = self.worker.lock();
            let model_ref = Arc::clone(&self.model);
            worker.step(&batch, |_input_tokens, logits| {
                // In production, this closure calls model.forward()
                // For now, fill with dummy logits
                for l in logits.iter_mut() { *l = 0.0; }
            }).clone()
        };

        // Process generated tokens
        for &(seq_id, token_id) in &result.tokens {
            self.on_token_generated(seq_id, token_id).await;
        }

        Ok(())
    }

    async fn on_token_generated(&self, seq_id: u64, token_id: u32) {
        let eos_token_id = self.tokenizer.eos_token_id().unwrap_or(2);

        // Update request state
        let request_id = {
            let map = self.seq_to_request.read();
            map.get(&seq_id).cloned()
        };

        if let Some(ref req_id) = request_id {
            {
                let mut states = self.request_states.write();
                if let Some(state) = states.get_mut(req_id) {
                    state.generated_tokens.push(token_id);
                }
            }
        }

        // Update scheduler
        {
            let mut scheduler = self.scheduler.lock();
            scheduler.on_token(seq_id);
        }

        // Check stop condition
        let is_finished = token_id == eos_token_id;

        if is_finished {
            {
                let mut scheduler = self.scheduler.lock();
                scheduler.complete(seq_id);
            }
            self.finish_sequence(seq_id).await;
        }

        // Send streaming update
        if let Some(ref req_id) = request_id {
            let stream_tx = {
                let states = self.request_states.read();
                states.get(req_id).and_then(|state| state.stream_tx.clone())
            };

            if let Some(tx) = stream_tx {
                let text = self.tokenizer
                    .decode(&[token_id], true)
                    .unwrap_or_default();

                let finish_reason = if is_finished { Some(FinishReason::Stop) } else { None };
                let chunk = StreamChunk {
                    request_id: req_id.clone(),
                    text,
                    token_id,
                    is_finished,
                    finish_reason,
                };
                let _ = tx.send(chunk).await;
            }
        }
    }

    async fn finish_sequence(&self, seq_id: u64) {
        let request_id = {
            let map = self.seq_to_request.read();
            map.get(&seq_id).cloned()
        };

        let Some(request_id) = request_id else { return };

        let state = {
            let mut states = self.request_states.write();
            states.remove(&request_id)
        };
        let Some(mut state) = state else { return };

        // Free blocks
        {
            let mut bm = self.block_manager.lock();
            bm.free(seq_id);
        }

        // Remove mapping
        {
            let mut map = self.seq_to_request.write();
            map.remove(&seq_id);
        }

        let output_tokens = state.generated_tokens.clone();
        let text = self.tokenizer
            .decode(&output_tokens, true)
            .unwrap_or_default();

        let output = GenerationOutput {
            request_id: request_id.clone(),
            text,
            token_ids: output_tokens.clone(),
            finish_reason: Some(FinishReason::Stop),
            usage: TokenUsage {
                prompt_tokens: state.prompt_tokens.len() as u32,
                completion_tokens: output_tokens.len() as u32,
                total_tokens: (state.prompt_tokens.len() + output_tokens.len()) as u32,
                cached_tokens: None,
            },
            logprobs: None,
        };

        if let Some(tx) = state.response_tx.take() {
            let _ = tx.send(Ok(output));
        }
        if let Some(tx) = state.stream_tx.take() {
            let chunk = StreamChunk {
                request_id: request_id.clone(),
                text: String::new(),
                token_id: 0,
                is_finished: true,
                finish_reason: Some(FinishReason::Stop),
            };
            let _ = tx.send(chunk).await;
        }
    }

    /// Run engine loop.
    pub async fn run(&self) -> Result<(), EngineError> {
        info!("Starting engine loop");

        loop {
            let is_shutdown = { *self.shutdown.read() };
            if is_shutdown {
                break;
            }
            self.step().await?;
            tokio::time::sleep(Duration::from_micros(100)).await;
        }

        info!("Engine loop stopped");
        Ok(())
    }

    /// Abort request.
    pub fn abort(&self, request_id: &str) -> bool {
        let states = self.request_states.read();
        let seq_id = match states.get(request_id) {
            Some(state) => state.seq_id,
            None => return false,
        };
        drop(states);

        let mut scheduler = self.scheduler.lock();
        scheduler.complete(seq_id);

        let mut states = self.request_states.write();
        if let Some(state) = states.remove(request_id) {
            if let Some(tx) = state.response_tx {
                let _ = tx.send(Err(EngineError::Cancelled));
            }
        }

        let mut map = self.seq_to_request.write();
        map.remove(&seq_id);

        true
    }

    /// Shutdown engine.
    pub fn shutdown(&self) {
        *self.shutdown.write() = true;
        info!("Engine shutdown requested");
    }

    /// Get engine statistics.
    pub fn stats(&self) -> EngineStats {
        let scheduler = self.scheduler.lock();
        let block_manager = self.block_manager.lock();
        EngineStats {
            num_waiting: scheduler.waiting_count(),
            num_running: scheduler.running_count(),
            num_swapped: scheduler.swapped_count(),
            gpu_memory_usage: 1.0 - (block_manager.free_gpu_blocks() as f32
                / (block_manager.free_gpu_blocks() as f32 + 1.0)),
        }
    }
}

/// Engine statistics.
#[derive(Debug, Clone)]
pub struct EngineStats {
    /// Number of waiting requests.
    pub num_waiting: usize,

    /// Number of running requests.
    pub num_running: usize,

    /// Number of swapped requests.
    pub num_swapped: usize,

    /// GPU memory usage (0.0-1.0).
    pub gpu_memory_usage: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config_default() {
        let config = EngineConfig::default();
        assert_eq!(config.max_concurrent_requests, 1000);
        assert!(config.enable_streaming);
    }
}
