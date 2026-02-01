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

use crate::batch::{BatchBuilder, BatchInput};
use crate::block_manager::BlockManagerConfig;
use crate::scheduler::{Scheduler, SchedulerConfig, SchedulerOutput};
use crate::sequence::{GroupId, SequenceGroup, SequenceId, SequenceStatus};
use crate::worker::{Worker, WorkerConfig, WorkerError};

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
    WorkerError(#[from] WorkerError),

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
    /// Group ID.
    group_id: GroupId,

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

    /// Worker.
    worker: Mutex<Worker>,

    /// Batch builder.
    batch_builder: BatchBuilder,

    /// Request states.
    request_states: RwLock<HashMap<String, RequestState>>,

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
        let scheduler = Scheduler::new(
            config.scheduler.clone(),
            config.block_manager.clone(),
        );

        let worker = Worker::new(config.worker.clone(), Arc::clone(&model));

        let batch_builder = BatchBuilder::new(
            config.scheduler.max_num_seqs,
            config.scheduler.max_num_batched_tokens,
        );

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
            worker: Mutex::new(worker),
            batch_builder,
            request_states: RwLock::new(HashMap::new()),
            shutdown: RwLock::new(false),
        })
    }

    /// Submit generation request.
    pub async fn generate(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationOutput, EngineError> {
        // Check shutdown
        if *self.shutdown.read() {
            return Err(EngineError::Shutdown);
        }

        // Tokenize prompt
        let encoding = self.tokenizer
            .encode(&request.prompt, true)
            .map_err(|e| EngineError::TokenizerError(e.to_string()))?;
        let prompt_tokens = encoding.ids().to_vec();

        // Create response channel
        let (tx, rx) = oneshot::channel();

        // Add to scheduler
        let group_id = {
            let mut scheduler = self.scheduler.lock();
            scheduler.add_request(
                request.request_id.clone(),
                prompt_tokens.clone(),
                request.sampling_params.clone(),
                request.max_tokens,
            )
        };

        // Track request state
        {
            let mut states = self.request_states.write();
            states.insert(request.request_id.clone(), RequestState {
                group_id,
                prompt_tokens,
                response_tx: Some(tx),
                stream_tx: None,
                start_time: Instant::now(),
                generated_tokens: Vec::new(),
            });
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
        // Check shutdown
        if *self.shutdown.read() {
            return Err(EngineError::Shutdown);
        }

        // Tokenize prompt
        let encoding = self.tokenizer
            .encode(&request.prompt, true)
            .map_err(|e| EngineError::TokenizerError(e.to_string()))?;
        let prompt_tokens = encoding.ids().to_vec();

        // Create stream channel
        let (tx, rx) = mpsc::channel(100);

        // Add to scheduler
        let group_id = {
            let mut scheduler = self.scheduler.lock();
            scheduler.add_request(
                request.request_id.clone(),
                prompt_tokens.clone(),
                request.sampling_params.clone(),
                request.max_tokens,
            )
        };

        // Track request state
        {
            let mut states = self.request_states.write();
            states.insert(request.request_id.clone(), RequestState {
                group_id,
                prompt_tokens,
                response_tx: None,
                stream_tx: Some(tx),
                start_time: Instant::now(),
                generated_tokens: Vec::new(),
            });
        }

        Ok(rx)
    }

    /// Run single engine step.
    pub async fn step(&self) -> Result<(), EngineError> {
        // Schedule batch
        let scheduler_output = {
            let mut scheduler = self.scheduler.lock();
            scheduler.schedule()
        };

        if scheduler_output.is_empty() {
            return Ok(());
        }

        // Build batch inputs
        let groups: HashMap<_, _> = {
            let scheduler = self.scheduler.lock();
            scheduler_output.scheduled_groups.iter()
                .filter_map(|sg| {
                    scheduler.get_running(&sg.group_id)
                        .map(|g| (sg.group_id, g))
                })
                .collect()
        };

        // Get block tables
        let block_tables: HashMap<SequenceId, Vec<usize>> = {
            let scheduler = self.scheduler.lock();
            let block_manager = scheduler.block_manager();

            groups.values()
                .flat_map(|g| g.sequence_ids())
                .filter_map(|seq_id| {
                    block_manager.get_block_table(&seq_id)
                        .map(|bt| (seq_id, bt.blocks().iter().map(|b| b.0).collect()))
                })
                .collect()
        };

        // This is a simplified version - in production would handle prefill/decode separately
        let (prefill_batch, decode_batch) = self.batch_builder.build(
            &scheduler_output,
            &groups.iter().map(|(k, v)| (*k, *v)).collect(),
            &block_tables,
        );

        // Execute batches
        if let Some(batch) = prefill_batch {
            self.execute_batch(batch).await?;
        }
        if let Some(batch) = decode_batch {
            self.execute_batch(batch).await?;
        }

        // Process finished requests
        self.process_finished().await?;

        Ok(())
    }

    /// Execute batch.
    async fn execute_batch(&self, batch: BatchInput) -> Result<(), EngineError> {
        let result = {
            let mut worker = self.worker.lock();
            worker.execute(&batch)?
        };

        // Process outputs
        for (seq_id, sample_result) in result.outputs {
            self.process_sample(seq_id, sample_result).await?;
        }

        Ok(())
    }

    /// Process sample result.
    async fn process_sample(
        &self,
        seq_id: SequenceId,
        sample_result: crate::sampler::SampleResult,
    ) -> Result<(), EngineError> {
        let token_id = sample_result.token_id;
        let eos_token_id = self.tokenizer.eos_token_id().unwrap_or(2);

        // Find request for this sequence
        let request_id = {
            let scheduler = self.scheduler.lock();
            // Find group containing this sequence
            // This is simplified - production would have better lookup
            None::<String>
        };

        // Update sequence with new token
        {
            let mut scheduler = self.scheduler.lock();
            // Would update sequence here
        }

        // Check for stop condition
        let finish_reason = if token_id == eos_token_id {
            Some(FinishReason::Stop)
        } else {
            None
        };

        // Send streaming update if applicable
        if let Some(req_id) = request_id {
            let states = self.request_states.read();
            if let Some(state) = states.get(&req_id) {
                if let Some(ref tx) = state.stream_tx {
                    let text = self.tokenizer
                        .decode(&[token_id], true)
                        .unwrap_or_default();

                    let chunk = StreamChunk {
                        request_id: req_id.clone(),
                        text,
                        token_id,
                        is_finished: finish_reason.is_some(),
                        finish_reason,
                    };

                    let _ = tx.send(chunk).await;
                }
            }
        }

        Ok(())
    }

    /// Process finished requests.
    async fn process_finished(&self) -> Result<(), EngineError> {
        let finished = {
            let mut scheduler = self.scheduler.lock();
            scheduler.get_finished()
        };

        for group in finished {
            let request_id = group.request_id.clone();

            // Get best sequence
            let best_seq = match group.get_best_sequence() {
                Some(s) => s,
                None => continue,
            };

            // Build output
            let output_tokens = best_seq.output_tokens().to_vec();
            let text = self.tokenizer
                .decode(&output_tokens, true)
                .unwrap_or_default();

            let finish_reason = match best_seq.status() {
                SequenceStatus::Finished(reason) => Some(reason),
                _ => None,
            };

            let output = GenerationOutput {
                request_id: request_id.clone(),
                text,
                token_ids: output_tokens.clone(),
                finish_reason,
                usage: TokenUsage {
                    prompt_tokens: group.prompt_len() as u32,
                    completion_tokens: output_tokens.len() as u32,
                    total_tokens: (group.prompt_len() + output_tokens.len()) as u32,
                },
                logprobs: None,
            };

            // Send response
            let mut states = self.request_states.write();
            if let Some(mut state) = states.remove(&request_id) {
                if let Some(tx) = state.response_tx.take() {
                    let _ = tx.send(Ok(output));
                }
                if let Some(tx) = state.stream_tx.take() {
                    // Send final chunk
                    let chunk = StreamChunk {
                        request_id: request_id.clone(),
                        text: String::new(),
                        token_id: 0,
                        is_finished: true,
                        finish_reason,
                    };
                    let _ = tx.send(chunk).await;
                }
            }
        }

        Ok(())
    }

    /// Run engine loop.
    pub async fn run(&self) -> Result<(), EngineError> {
        info!("Starting engine loop");

        while !*self.shutdown.read() {
            self.step().await?;

            // Small sleep to prevent busy loop
            tokio::time::sleep(Duration::from_micros(100)).await;
        }

        info!("Engine loop stopped");
        Ok(())
    }

    /// Abort request.
    pub fn abort(&self, request_id: &str) -> bool {
        let mut scheduler = self.scheduler.lock();
        let aborted = scheduler.abort(request_id);

        if aborted {
            let mut states = self.request_states.write();
            if let Some(state) = states.remove(request_id) {
                if let Some(tx) = state.response_tx {
                    let _ = tx.send(Err(EngineError::Cancelled));
                }
            }
        }

        aborted
    }

    /// Shutdown engine.
    pub fn shutdown(&self) {
        *self.shutdown.write() = true;
        info!("Engine shutdown requested");
    }

    /// Get engine statistics.
    pub fn stats(&self) -> EngineStats {
        let scheduler = self.scheduler.lock();
        EngineStats {
            num_waiting: scheduler.num_waiting(),
            num_running: scheduler.num_running(),
            num_swapped: scheduler.num_swapped(),
            gpu_memory_usage: scheduler.block_manager().gpu_memory_usage(),
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
