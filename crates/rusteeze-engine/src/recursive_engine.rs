// ============================================================================
// Rusteeze - Recursive Language Model Infrastructure
// Recursive Inference Engine: The core RLM inference loop
// Based on: "Recursive Language Models" (Zhang, Kraska, Khattab 2026)
// ============================================================================
//
// This is the main orchestrator implementing Algorithm 1 from the paper.
// It coordinates:
// 1. The REPL loop (PromptEnvironment)
// 2. The recursive call tree (RecursiveScheduler)
// 3. Model inference (via the underlying Engine)
// 4. Sub-call execution (potentially parallel with rayon)
// 5. Result aggregation and final output
//
// Architecture:
//   User Request -> RecursiveInferenceEngine
//     -> PromptEnvironment (stores prompt as variable, not in context)
//     -> Root model processes metadata only (constant-size context)
//     -> Model outputs operations (peek, decompose, subcall, etc.)
//     -> Engine executes operations in REPL
//     -> Sub-calls spawned via RecursiveScheduler
//     -> Sub-calls processed by child PromptEnvironments
//     -> Results aggregated back into parent environment
//     -> Repeat until FINAL is set
//     -> Return response
// ============================================================================

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::prompt_env::{
    OpResult, PendingCall, PromptEnvConfig, PromptEnvironment, ReplOperation, TransformOp,
};
use crate::recursive_scheduler::{
    CallId, CallStatus, RecursiveScheduler, RecursiveSchedulerConfig, SchedulerError,
};
use crate::variable_store::{VarId, VarValue};

/// Configuration for the Recursive Inference Engine
#[derive(Debug, Clone)]
pub struct RecursiveEngineConfig {
    /// Prompt environment configuration
    pub env_config: PromptEnvConfig,
    /// Recursive scheduler configuration
    pub scheduler_config: RecursiveSchedulerConfig,
    /// Maximum total tokens across all calls
    pub max_total_tokens: usize,
    /// Whether to use parallel sub-call execution
    pub enable_parallel_subcalls: bool,
    /// Number of rayon threads for parallel execution
    pub parallel_threads: usize,
    /// Whether to automatically decompose large prompts
    pub auto_decompose: bool,
    /// Auto-decompose threshold (in characters)
    pub auto_decompose_threshold: usize,
    /// Default chunk size for auto-decomposition
    pub default_chunk_size: usize,
    /// Default chunk overlap
    pub default_chunk_overlap: usize,
    /// Whether to enable cost tracking
    pub track_costs: bool,
    /// Cost per 1K input tokens (for tracking)
    pub cost_per_1k_input: f64,
    /// Cost per 1K output tokens (for tracking)
    pub cost_per_1k_output: f64,
}

impl Default for RecursiveEngineConfig {
    fn default() -> Self {
        Self {
            env_config: PromptEnvConfig::default(),
            scheduler_config: RecursiveSchedulerConfig::default(),
            max_total_tokens: 10_000_000,
            enable_parallel_subcalls: true,
            parallel_threads: num_cpus(),
            auto_decompose: true,
            auto_decompose_threshold: 100_000, // 100K chars
            default_chunk_size: 8000,
            default_chunk_overlap: 200,
            track_costs: true,
            cost_per_1k_input: 0.001,
            cost_per_1k_output: 0.003,
        }
    }
}

/// Get number of available CPU cores
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// A model inference function that takes a prompt and returns a response
/// This trait abstracts the model call so the RLM engine doesn't depend on
/// a specific model implementation.
pub trait InferenceModel: Send + Sync {
    /// Run inference on the given prompt, return the model's response
    fn infer(&self, prompt: &str, max_tokens: usize) -> Result<InferenceResult, InferenceError>;

    /// Run inference with structured output (for operation parsing)
    fn infer_structured(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<InferenceResult, InferenceError> {
        // Default implementation: use regular inference
        self.infer(prompt, max_tokens)
    }

    /// Get the model's context window size in tokens
    fn context_window(&self) -> usize;

    /// Estimate token count for a text
    fn estimate_tokens(&self, text: &str) -> usize {
        // Rough estimate: 4 chars per token
        text.len() / 4
    }
}

/// Result of a model inference call
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The model's text response
    pub text: String,
    /// Number of input tokens consumed
    pub input_tokens: usize,
    /// Number of output tokens generated
    pub output_tokens: usize,
    /// Inference latency
    pub latency: Duration,
    /// Whether the response was truncated (hit max tokens)
    pub truncated: bool,
}

/// Error from model inference
#[derive(Debug, Clone)]
pub enum InferenceError {
    /// Model returned an error
    ModelError(String),
    /// Context window exceeded
    ContextExceeded { available: usize, required: usize },
    /// Timeout
    Timeout,
    /// Model not available
    Unavailable,
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InferenceError::ModelError(msg) => write!(f, "Model error: {}", msg),
            InferenceError::ContextExceeded {
                available,
                required,
            } => write!(
                f,
                "Context exceeded: need {} tokens, have {}",
                required, available
            ),
            InferenceError::Timeout => write!(f, "Inference timeout"),
            InferenceError::Unavailable => write!(f, "Model unavailable"),
        }
    }
}

impl std::error::Error for InferenceError {}

/// Request to the Recursive Inference Engine
#[derive(Debug, Clone)]
pub struct RecursiveRequest {
    /// Unique request ID
    pub request_id: String,
    /// The user prompt (can be arbitrarily long)
    pub prompt: String,
    /// The task instruction
    pub instruction: String,
    /// Maximum tokens for the response
    pub max_response_tokens: usize,
    /// Optional: override config
    pub config_override: Option<RecursiveEngineConfig>,
}

/// Response from the Recursive Inference Engine
#[derive(Debug, Clone)]
pub struct RecursiveResponse {
    /// Request ID
    pub request_id: String,
    /// The final response text
    pub response: String,
    /// Detailed statistics
    pub stats: RecursiveStats,
    /// Whether the response was complete (vs truncated/timed out)
    pub complete: bool,
}

/// Detailed statistics for a recursive inference run
#[derive(Debug, Clone, Default)]
pub struct RecursiveStats {
    /// Total wall-clock time
    pub total_time: Duration,
    /// Number of RLM iterations (root level)
    pub root_iterations: u64,
    /// Total recursive sub-calls made
    pub total_subcalls: u64,
    /// Maximum recursion depth reached
    pub max_depth_reached: usize,
    /// Total input tokens across all calls
    pub total_input_tokens: u64,
    /// Total output tokens across all calls
    pub total_output_tokens: u64,
    /// Estimated cost (if tracking enabled)
    pub estimated_cost: f64,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Whether auto-decomposition was used
    pub auto_decomposed: bool,
    /// Call tree depth distribution
    pub depth_distribution: HashMap<usize, usize>,
}

/// The Recursive Inference Engine
///
/// Implements Algorithm 1 from the RLM paper:
/// 1. Initialize REPL with prompt as variable
/// 2. Present metadata to root model
/// 3. Model outputs code/operations to examine prompt
/// 4. Execute operations in REPL
/// 5. Spawn sub-calls as needed
/// 6. Continue until FINAL is set or limits exceeded
pub struct RecursiveInferenceEngine {
    /// Configuration
    config: RecursiveEngineConfig,
    /// Global statistics across all requests
    global_stats: RwLock<GlobalStats>,
}

/// Global statistics across all requests
#[derive(Debug, Clone, Default)]
pub struct GlobalStats {
    pub total_requests: u64,
    pub total_subcalls: u64,
    pub total_tokens: u64,
    pub total_estimated_cost: f64,
    pub avg_iterations_per_request: f64,
    pub avg_depth_per_request: f64,
}

impl RecursiveInferenceEngine {
    /// Create a new Recursive Inference Engine
    pub fn new(config: RecursiveEngineConfig) -> Self {
        Self {
            config,
            global_stats: RwLock::new(GlobalStats::default()),
        }
    }

    /// Process a recursive inference request
    ///
    /// This implements the main RLM loop from Algorithm 1 in the paper.
    pub fn process(
        &self,
        request: &RecursiveRequest,
        model: &dyn InferenceModel,
    ) -> Result<RecursiveResponse, RecursiveInferenceError> {
        let start = Instant::now();
        let config = request
            .config_override
            .clone()
            .unwrap_or_else(|| self.config.clone());

        // === Step 1: Initialize REPL environment (Algorithm 1, line 1-2) ===
        let mut env = PromptEnvironment::new_root(
            request.prompt.clone(),
            config.env_config.clone(),
        );

        // === Step 2: Initialize scheduler ===
        let mut scheduler = RecursiveScheduler::new(config.scheduler_config.clone());

        // === Step 3: Auto-decompose if prompt is very large ===
        let mut stats = RecursiveStats::default();
        if config.auto_decompose
            && request.prompt.len() > config.auto_decompose_threshold
        {
            let decompose_result = env.execute(ReplOperation::Decompose {
                chunk_size: config.default_chunk_size,
                overlap: config.default_chunk_overlap,
            });

            if let OpResult::Success { .. } = decompose_result {
                stats.auto_decomposed = true;
            }
        }

        // === Step 4: Build initial context for root model ===
        let system_prompt = self.build_system_prompt(&config);
        let initial_metadata = env.initial_metadata();

        // === Step 5: Main RLM loop (Algorithm 1, lines 4-12) ===
        let mut total_input_tokens: u64 = 0;
        let mut total_output_tokens: u64 = 0;
        let mut stdout_buffer = String::new();

        loop {
            // Check termination conditions
            if env.is_terminated() {
                break;
            }

            // Check timeout
            if start.elapsed() > config.env_config.session_timeout {
                break;
            }

            // Build the context for this iteration
            let iteration_context = if env.iteration() == 0 {
                format!(
                    "{}\n\n{}\n\nTask: {}\n",
                    system_prompt, initial_metadata, request.instruction
                )
            } else {
                let iter_meta = env.iteration_metadata(&stdout_buffer);
                format!(
                    "{}\n\n{}\n\nContinue processing. Task: {}\n",
                    system_prompt, iter_meta, request.instruction
                )
            };

            // === Step 6: Invoke root model with metadata only ===
            let inference_result = match model.infer(
                &iteration_context,
                request.max_response_tokens.min(4096),
            ) {
                Ok(result) => result,
                Err(InferenceError::ContextExceeded { available, required }) => {
                    // Context too large: try to trim history
                    return Err(RecursiveInferenceError::ContextExhausted {
                        available,
                        required,
                    });
                }
                Err(e) => {
                    return Err(RecursiveInferenceError::ModelError(e.to_string()));
                }
            };

            total_input_tokens += inference_result.input_tokens as u64;
            total_output_tokens += inference_result.output_tokens as u64;

            // === Step 7: Parse model output into REPL operation ===
            let operation = self.parse_model_output(&inference_result.text, &env);

            // === Step 8: Execute operation in REPL ===
            let op_result = env.execute(operation);

            // === Step 9: Handle operation result ===
            match op_result {
                OpResult::Success { output, .. } => {
                    stdout_buffer = output;
                }

                OpResult::PendingSubCall {
                    call_id,
                    prompt,
                    instruction,
                    result_var,
                } => {
                    // Execute the sub-call
                    let sub_result = self.execute_subcall(
                        model,
                        &prompt,
                        &instruction,
                        env.recursion_depth() + 1,
                        call_id,
                        &config,
                    )?;

                    total_input_tokens += sub_result.input_tokens;
                    total_output_tokens += sub_result.output_tokens;
                    stats.total_subcalls += 1;
                    stats.max_depth_reached =
                        stats.max_depth_reached.max(env.recursion_depth() + 1);

                    // Store result back into environment
                    env.store_subcall_result(result_var, sub_result.response)
                        .map_err(|e| {
                            RecursiveInferenceError::EnvironmentError(e.to_string())
                        })?;

                    stdout_buffer = format!(
                        "Sub-call {} completed ({} tokens)",
                        call_id, sub_result.output_tokens
                    );
                }

                OpResult::PendingBatchSubCalls { calls } => {
                    // Execute batch sub-calls (sequentially for now; parallel in future)
                    let mut results = Vec::with_capacity(calls.len());
                    let mut batch_input_tokens = 0u64;
                    let mut batch_output_tokens = 0u64;

                    for pending_call in &calls {
                        let sub_result = self.execute_subcall(
                            model,
                            &pending_call.prompt,
                            &pending_call.instruction,
                            env.recursion_depth() + 1,
                            pending_call.call_id,
                            &config,
                        )?;

                        batch_input_tokens += sub_result.input_tokens;
                        batch_output_tokens += sub_result.output_tokens;

                        results.push((
                            pending_call.result_var.clone(),
                            sub_result.response,
                        ));
                    }

                    total_input_tokens += batch_input_tokens;
                    total_output_tokens += batch_output_tokens;
                    stats.total_subcalls += calls.len() as u64;
                    stats.max_depth_reached =
                        stats.max_depth_reached.max(env.recursion_depth() + 1);

                    env.store_batch_results(results).map_err(|e| {
                        RecursiveInferenceError::EnvironmentError(e.to_string())
                    })?;

                    stdout_buffer = format!(
                        "Batch of {} sub-calls completed ({} in, {} out tokens)",
                        calls.len(),
                        batch_input_tokens,
                        batch_output_tokens
                    );
                }

                OpResult::FinalSet { output } => {
                    stdout_buffer = format!("FINAL: {}", output);
                    // Will break on next iteration check
                }

                OpResult::Error { message } => {
                    stdout_buffer = format!("ERROR: {}", message);
                }
            }
        }

        // === Step 10: Build response ===
        let response_text = env
            .final_output()
            .unwrap_or_else(|| stdout_buffer.clone());

        let complete = env.is_terminated() && env.final_output().is_some();

        // Calculate cost
        let estimated_cost = if config.track_costs {
            (total_input_tokens as f64 / 1000.0) * config.cost_per_1k_input
                + (total_output_tokens as f64 / 1000.0) * config.cost_per_1k_output
        } else {
            0.0
        };

        stats.total_time = start.elapsed();
        stats.root_iterations = env.iteration();
        stats.total_input_tokens = total_input_tokens;
        stats.total_output_tokens = total_output_tokens;
        stats.estimated_cost = estimated_cost;
        stats.peak_memory_bytes = env.stats().peak_memory_bytes;

        // Update global stats
        if let Ok(mut global) = self.global_stats.write() {
            global.total_requests += 1;
            global.total_subcalls += stats.total_subcalls;
            global.total_tokens += total_input_tokens + total_output_tokens;
            global.total_estimated_cost += estimated_cost;
        }

        Ok(RecursiveResponse {
            request_id: request.request_id.clone(),
            response: response_text,
            stats,
            complete,
        })
    }

    /// Execute a single sub-call (recursive inference at a deeper level)
    fn execute_subcall(
        &self,
        model: &dyn InferenceModel,
        prompt: &str,
        instruction: &str,
        depth: usize,
        call_id: u64,
        config: &RecursiveEngineConfig,
    ) -> Result<SubCallResult, RecursiveInferenceError> {
        let start = Instant::now();

        // For sub-calls, we create a simple child environment
        let mut child_env = PromptEnvironment::new_subcall(
            prompt.to_string(),
            instruction.to_string(),
            0, // parent_id
            depth,
            call_id,
            config.env_config.clone(),
        );

        // Sub-calls use a simpler approach: direct model inference
        // The sub-call model just processes the prompt + instruction directly
        let sub_prompt = format!(
            "You are a recursive sub-call at depth {}.\n\
             Task: {}\n\n\
             Input ({} chars):\n{}",
            depth,
            instruction,
            prompt.len(),
            if prompt.len() > config.env_config.context_window_tokens * 3 {
                // If prompt is too long, chunk it
                format!(
                    "{}... [truncated, {} chars total]",
                    &prompt[..config.env_config.context_window_tokens * 3],
                    prompt.len()
                )
            } else {
                prompt.to_string()
            }
        );

        let inference_result =
            model
                .infer(&sub_prompt, 2048)
                .map_err(|e| RecursiveInferenceError::SubCallError {
                    call_id,
                    depth,
                    error: e.to_string(),
                })?;

        Ok(SubCallResult {
            response: inference_result.text,
            input_tokens: inference_result.input_tokens as u64,
            output_tokens: inference_result.output_tokens as u64,
            latency: start.elapsed(),
        })
    }

    /// Build the system prompt for the RLM root model
    fn build_system_prompt(&self, config: &RecursiveEngineConfig) -> String {
        format!(
            "You are a Recursive Language Model (RLM). Your input prompt is stored as a \
             variable in an external environment. You do NOT have the full prompt in your \
             context window - only metadata about it.\n\n\
             You can interact with the prompt through these operations:\n\
             1. PEEK(start, end) - View bytes [start..end] of the prompt\n\
             2. DECOMPOSE(chunk_size, overlap) - Split into chunks stored in '__chunks__'\n\
             3. SUBCALL(text, instruction) - Invoke yourself recursively on a text slice\n\
             4. BATCH_SUBCALL(chunk_var, instruction) - Invoke on each chunk in parallel\n\
             5. SET_VAR(name, value) - Store an intermediate result\n\
             6. GET_VAR(name) - Retrieve a stored variable\n\
             7. SEARCH(pattern) - Search the prompt for a pattern\n\
             8. TRANSFORM(source, target, op) - Transform a variable\n\
             9. FINAL(answer) - Set your final answer and terminate\n\
             10. FINAL_VAR(name) - Set final answer to a variable's value\n\n\
             Strategy guide:\n\
             - Start by peeking at the prompt to understand its structure\n\
             - Use DECOMPOSE for long prompts, then BATCH_SUBCALL to process chunks\n\
             - Use SEARCH to find relevant parts before reading them\n\
             - Store intermediate results in variables\n\
             - Aggregate results and set FINAL when done\n\
             - Max recursion depth: {}\n\
             - Context window: {} tokens\n\n\
             Output format: Write your operation as a single line:\n\
             OPERATION(arg1, arg2, ...)\n\
             Then explain your reasoning on subsequent lines.",
            config.scheduler_config.max_depth,
            config.env_config.context_window_tokens
        )
    }

    /// Parse model output text into a REPL operation
    ///
    /// This is a heuristic parser that extracts the intended operation from
    /// the model's freeform text output. In a production system, this would
    /// use structured output / function calling.
    fn parse_model_output(
        &self,
        output: &str,
        _env: &PromptEnvironment,
    ) -> ReplOperation {
        let trimmed = output.trim();

        // Try to find an operation keyword at the start of any line
        for line in trimmed.lines() {
            let line = line.trim();

            // FINAL(value) or FINAL_VAR(name)
            if let Some(rest) = strip_prefix_case_insensitive(line, "FINAL_VAR(") {
                if let Some(name) = rest.strip_suffix(')') {
                    return ReplOperation::SetFinalVar {
                        var_name: name.trim().trim_matches('"').to_string(),
                    };
                }
            }

            if let Some(rest) = strip_prefix_case_insensitive(line, "FINAL(") {
                if let Some(value) = rest.strip_suffix(')') {
                    return ReplOperation::SetFinal {
                        value: value.trim().trim_matches('"').to_string(),
                    };
                }
            }

            // PEEK(start, end)
            if let Some(rest) = strip_prefix_case_insensitive(line, "PEEK(") {
                if let Some(args) = rest.strip_suffix(')') {
                    let parts: Vec<&str> = args.split(',').collect();
                    if parts.len() == 2 {
                        if let (Ok(start), Ok(end)) = (
                            parts[0].trim().parse::<usize>(),
                            parts[1].trim().parse::<usize>(),
                        ) {
                            return ReplOperation::Peek {
                                byte_start: start,
                                byte_end: end,
                            };
                        }
                    }
                }
            }

            // DECOMPOSE(chunk_size, overlap)
            if let Some(rest) = strip_prefix_case_insensitive(line, "DECOMPOSE(") {
                if let Some(args) = rest.strip_suffix(')') {
                    let parts: Vec<&str> = args.split(',').collect();
                    if parts.len() >= 1 {
                        let chunk_size = parts[0].trim().parse::<usize>().unwrap_or(8000);
                        let overlap = parts
                            .get(1)
                            .and_then(|s| s.trim().parse::<usize>().ok())
                            .unwrap_or(200);
                        return ReplOperation::Decompose {
                            chunk_size,
                            overlap,
                        };
                    }
                }
            }

            // SUBCALL(text, instruction) or SUBCALL_VAR(var, instruction)
            if let Some(rest) = strip_prefix_case_insensitive(line, "SUBCALL_VAR(") {
                if let Some(args) = rest.strip_suffix(')') {
                    let parts: Vec<&str> = args.splitn(2, ',').collect();
                    if parts.len() == 2 {
                        return ReplOperation::SubCallVar {
                            var_id: VarId::new(parts[0].trim().trim_matches('"')),
                            instruction: parts[1].trim().trim_matches('"').to_string(),
                        };
                    }
                }
            }

            if let Some(rest) = strip_prefix_case_insensitive(line, "SUBCALL(") {
                if let Some(args) = rest.strip_suffix(')') {
                    let parts: Vec<&str> = args.splitn(2, ',').collect();
                    if parts.len() == 2 {
                        return ReplOperation::SubCall {
                            prompt_text: parts[0].trim().trim_matches('"').to_string(),
                            instruction: parts[1].trim().trim_matches('"').to_string(),
                        };
                    }
                }
            }

            // BATCH_SUBCALL(chunk_var, instruction)
            if let Some(rest) = strip_prefix_case_insensitive(line, "BATCH_SUBCALL(") {
                if let Some(args) = rest.strip_suffix(')') {
                    let parts: Vec<&str> = args.splitn(2, ',').collect();
                    if parts.len() == 2 {
                        return ReplOperation::BatchSubCall {
                            chunk_var_id: VarId::new(parts[0].trim().trim_matches('"')),
                            instruction: parts[1].trim().trim_matches('"').to_string(),
                        };
                    }
                }
            }

            // SET_VAR(name, value)
            if let Some(rest) = strip_prefix_case_insensitive(line, "SET_VAR(") {
                if let Some(args) = rest.strip_suffix(')') {
                    let parts: Vec<&str> = args.splitn(2, ',').collect();
                    if parts.len() == 2 {
                        let name = parts[0].trim().trim_matches('"').to_string();
                        let value = parts[1].trim().trim_matches('"').to_string();
                        return ReplOperation::SetVar {
                            name,
                            value: VarValue::Text(Arc::from(value.as_str())),
                        };
                    }
                }
            }

            // GET_VAR(name)
            if let Some(rest) = strip_prefix_case_insensitive(line, "GET_VAR(") {
                if let Some(name) = rest.strip_suffix(')') {
                    return ReplOperation::GetVar {
                        name: name.trim().trim_matches('"').to_string(),
                    };
                }
            }

            // SEARCH(pattern)
            if let Some(rest) = strip_prefix_case_insensitive(line, "SEARCH(") {
                if let Some(pattern) = rest.strip_suffix(')') {
                    return ReplOperation::RegexSearch {
                        pattern: pattern.trim().trim_matches('"').to_string(),
                    };
                }
            }

            // TRANSFORM(source, target, op)
            if let Some(rest) = strip_prefix_case_insensitive(line, "TRANSFORM(") {
                if let Some(args) = rest.strip_suffix(')') {
                    let parts: Vec<&str> = args.splitn(3, ',').collect();
                    if parts.len() == 3 {
                        let source = VarId::new(parts[0].trim().trim_matches('"'));
                        let target = VarId::new(parts[1].trim().trim_matches('"'));
                        let op_str = parts[2].trim().trim_matches('"');

                        let transform = parse_transform_op(op_str);
                        return ReplOperation::Transform {
                            source_var: source,
                            target_var: target,
                            transform,
                        };
                    }
                }
            }
        }

        // If no operation was parsed, treat the entire output as "thinking"
        ReplOperation::Think {
            thought: if trimmed.len() > 200 {
                format!("{}...", &trimmed[..200])
            } else {
                trimmed.to_string()
            },
        }
    }

    /// Get global statistics
    pub fn global_stats(&self) -> GlobalStats {
        self.global_stats
            .read()
            .map(|s| s.clone())
            .unwrap_or_default()
    }

    /// Get the configuration
    pub fn config(&self) -> &RecursiveEngineConfig {
        &self.config
    }
}

use std::sync::Arc as StdArc;

/// Result from a sub-call execution
#[derive(Debug, Clone)]
struct SubCallResult {
    response: String,
    input_tokens: u64,
    output_tokens: u64,
    latency: Duration,
}

/// Case-insensitive prefix strip
fn strip_prefix_case_insensitive<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len()
        && s[..prefix.len()].eq_ignore_ascii_case(prefix)
    {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

/// Parse a transform operation from a string
fn parse_transform_op(s: &str) -> TransformOp {
    let s_lower = s.to_lowercase();

    if s_lower.starts_with("split") {
        let delim = extract_arg(&s_lower, "split").unwrap_or(",".to_string());
        TransformOp::Split { delimiter: delim }
    } else if s_lower.starts_with("join") {
        let delim = extract_arg(&s_lower, "join").unwrap_or("\n".to_string());
        TransformOp::Join { delimiter: delim }
    } else if s_lower.starts_with("filter") {
        let keyword = extract_arg(&s_lower, "filter").unwrap_or_default();
        TransformOp::FilterContains { keyword }
    } else if s_lower == "sort" {
        TransformOp::Sort
    } else if s_lower == "count" {
        TransformOp::Count
    } else if s_lower.starts_with("truncate") {
        let max = extract_arg(&s_lower, "truncate")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);
        TransformOp::Truncate { max_chars: max }
    } else if s_lower.starts_with("grep") {
        let pattern = extract_arg(&s_lower, "grep").unwrap_or_default();
        TransformOp::GrepLines { pattern }
    } else {
        // Default: count
        TransformOp::Count
    }
}

/// Extract an argument from a function-like string (e.g., "split(,)" -> ",")
fn extract_arg(s: &str, prefix: &str) -> Option<String> {
    let rest = s.strip_prefix(prefix)?;
    let rest = rest.trim();
    if rest.starts_with('(') && rest.ends_with(')') {
        Some(rest[1..rest.len() - 1].to_string())
    } else if !rest.is_empty() {
        Some(rest.to_string())
    } else {
        None
    }
}

/// Errors from the Recursive Inference Engine
#[derive(Debug, Clone)]
pub enum RecursiveInferenceError {
    /// Model inference failed
    ModelError(String),
    /// Context window exhausted
    ContextExhausted { available: usize, required: usize },
    /// Sub-call failed
    SubCallError {
        call_id: u64,
        depth: usize,
        error: String,
    },
    /// Environment error (variable store, etc.)
    EnvironmentError(String),
    /// Session timed out
    Timeout(Duration),
    /// Maximum iterations exceeded
    MaxIterations(usize),
    /// Maximum sub-calls exceeded
    MaxSubCalls(usize),
}

impl fmt::Display for RecursiveInferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecursiveInferenceError::ModelError(msg) => {
                write!(f, "Model error: {}", msg)
            }
            RecursiveInferenceError::ContextExhausted {
                available,
                required,
            } => write!(
                f,
                "Context exhausted: need {} tokens, have {}",
                required, available
            ),
            RecursiveInferenceError::SubCallError {
                call_id,
                depth,
                error,
            } => write!(
                f,
                "Sub-call {} at depth {} failed: {}",
                call_id, depth, error
            ),
            RecursiveInferenceError::EnvironmentError(msg) => {
                write!(f, "Environment error: {}", msg)
            }
            RecursiveInferenceError::Timeout(d) => {
                write!(f, "Session timed out after {:?}", d)
            }
            RecursiveInferenceError::MaxIterations(n) => {
                write!(f, "Maximum iterations ({}) exceeded", n)
            }
            RecursiveInferenceError::MaxSubCalls(n) => {
                write!(f, "Maximum sub-calls ({}) exceeded", n)
            }
        }
    }
}

impl std::error::Error for RecursiveInferenceError {}

// ============================================================================
// Mock model for testing
// ============================================================================

/// A simple mock model for testing the RLM engine
pub struct MockInferenceModel {
    /// Predetermined responses (keyed by instruction keywords)
    responses: HashMap<String, String>,
    context_window: usize,
}

impl MockInferenceModel {
    pub fn new(context_window: usize) -> Self {
        Self {
            responses: HashMap::new(),
            context_window,
        }
    }

    pub fn add_response(&mut self, keyword: &str, response: &str) {
        self.responses
            .insert(keyword.to_lowercase(), response.to_string());
    }

    /// Create a mock that always returns FINAL after peeking
    pub fn auto_resolve(context_window: usize) -> Self {
        let mut mock = Self::new(context_window);
        // First call matches when task instruction is present (iteration 0)
        mock.add_response("task:", "PEEK(0, 500)");
        // Subsequent calls match when "continue processing" appears (iteration > 0)
        mock.add_response("continue processing", "FINAL(Processed successfully)");
        mock
    }
}

impl InferenceModel for MockInferenceModel {
    fn infer(&self, prompt: &str, _max_tokens: usize) -> Result<InferenceResult, InferenceError> {
        let prompt_lower = prompt.to_lowercase();

        // Find matching response
        let response = self
            .responses
            .iter()
            .find(|(keyword, _)| prompt_lower.contains(keyword.as_str()))
            .map(|(_, response)| response.clone())
            .unwrap_or_else(|| "FINAL(Default response)".to_string());

        Ok(InferenceResult {
            text: response,
            input_tokens: prompt.len() / 4,
            output_tokens: 10,
            latency: Duration::from_millis(1),
            truncated: false,
        })
    }

    fn context_window(&self) -> usize {
        self.context_window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let config = RecursiveEngineConfig::default();
        let engine = RecursiveInferenceEngine::new(config);

        assert_eq!(engine.global_stats().total_requests, 0);
    }

    #[test]
    fn test_mock_model() {
        let model = MockInferenceModel::auto_resolve(32768);

        let result = model.infer("Test prompt", 100).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn test_basic_recursive_inference() {
        let config = RecursiveEngineConfig {
            env_config: PromptEnvConfig {
                max_iterations: 10,
                ..Default::default()
            },
            ..Default::default()
        };

        let engine = RecursiveInferenceEngine::new(config);
        let model = MockInferenceModel::auto_resolve(32768);

        let request = RecursiveRequest {
            request_id: "test-1".to_string(),
            prompt: "This is a test prompt that should be processed.".to_string(),
            instruction: "Analyze this text.".to_string(),
            max_response_tokens: 1000,
            config_override: None,
        };

        let response = engine.process(&request, &model).unwrap();

        assert_eq!(response.request_id, "test-1");
        assert!(!response.response.is_empty());
        assert!(response.stats.root_iterations > 0);
    }

    #[test]
    fn test_operation_parsing() {
        let config = RecursiveEngineConfig::default();
        let engine = RecursiveInferenceEngine::new(config.clone());
        let env = PromptEnvironment::new_root(
            "test".to_string(),
            config.env_config.clone(),
        );

        // Test PEEK parsing
        let op = engine.parse_model_output("PEEK(0, 100)", &env);
        assert!(matches!(
            op,
            ReplOperation::Peek {
                byte_start: 0,
                byte_end: 100
            }
        ));

        // Test FINAL parsing
        let op = engine.parse_model_output("FINAL(The answer is 42)", &env);
        assert!(matches!(op, ReplOperation::SetFinal { .. }));

        // Test DECOMPOSE parsing
        let op = engine.parse_model_output("DECOMPOSE(5000, 200)", &env);
        assert!(matches!(
            op,
            ReplOperation::Decompose {
                chunk_size: 5000,
                overlap: 200
            }
        ));

        // Test SEARCH parsing
        let op = engine.parse_model_output("SEARCH(important keyword)", &env);
        assert!(matches!(op, ReplOperation::RegexSearch { .. }));

        // Test unknown -> Think
        let op = engine.parse_model_output("I need to think about this...", &env);
        assert!(matches!(op, ReplOperation::Think { .. }));
    }

    #[test]
    fn test_case_insensitive_parsing() {
        let config = RecursiveEngineConfig::default();
        let engine = RecursiveInferenceEngine::new(config.clone());
        let env = PromptEnvironment::new_root(
            "test".to_string(),
            config.env_config.clone(),
        );

        let op = engine.parse_model_output("peek(0, 50)", &env);
        assert!(matches!(op, ReplOperation::Peek { .. }));

        let op = engine.parse_model_output("final(done)", &env);
        assert!(matches!(op, ReplOperation::SetFinal { .. }));
    }

    #[test]
    fn test_subcall_execution() {
        let mut model = MockInferenceModel::new(32768);
        model.add_response("recursive sub-call", "Sub-call result: processed chunk");

        let config = RecursiveEngineConfig::default();
        let engine = RecursiveInferenceEngine::new(config.clone());

        let result = engine
            .execute_subcall(
                &model,
                "test chunk content",
                "summarize this",
                1,
                1,
                &engine.config,
            )
            .unwrap();

        assert!(!result.response.is_empty());
        assert!(result.input_tokens > 0);
    }

    #[test]
    fn test_with_decomposition_and_subcalls() {
        let config = RecursiveEngineConfig {
            env_config: PromptEnvConfig {
                max_iterations: 10,
                ..Default::default()
            },
            auto_decompose: false, // Manual decomposition
            ..Default::default()
        };

        let engine = RecursiveInferenceEngine::new(config);

        let mut model = MockInferenceModel::new(32768);
        // First call: decompose (match the task instruction)
        model.add_response("count the lines", "DECOMPOSE(100, 20)");
        // After decompose: set final (match iteration metadata output)
        model.add_response("continue processing", "FINAL(Analysis complete)");

        let request = RecursiveRequest {
            request_id: "test-2".to_string(),
            prompt: "Line 1\nLine 2\nLine 3\n".repeat(50),
            instruction: "Count the lines.".to_string(),
            max_response_tokens: 1000,
            config_override: None,
        };

        let response = engine.process(&request, &model).unwrap();

        assert!(!response.response.is_empty());
        assert!(response.stats.root_iterations >= 2);
    }

    #[test]
    fn test_auto_decompose_large_prompt() {
        let config = RecursiveEngineConfig {
            env_config: PromptEnvConfig {
                max_iterations: 10,
                ..Default::default()
            },
            auto_decompose: true,
            auto_decompose_threshold: 100, // Very low threshold for testing
            ..Default::default()
        };

        let engine = RecursiveInferenceEngine::new(config);
        let model = MockInferenceModel::auto_resolve(32768);

        let request = RecursiveRequest {
            request_id: "test-3".to_string(),
            prompt: "x".repeat(500), // Exceeds threshold
            instruction: "Process.".to_string(),
            max_response_tokens: 100,
            config_override: None,
        };

        let response = engine.process(&request, &model).unwrap();
        assert!(response.stats.auto_decomposed);
    }

    #[test]
    fn test_cost_tracking() {
        let config = RecursiveEngineConfig {
            env_config: PromptEnvConfig {
                max_iterations: 5,
                ..Default::default()
            },
            track_costs: true,
            cost_per_1k_input: 0.001,
            cost_per_1k_output: 0.003,
            ..Default::default()
        };

        let engine = RecursiveInferenceEngine::new(config);
        let model = MockInferenceModel::auto_resolve(32768);

        let request = RecursiveRequest {
            request_id: "test-4".to_string(),
            prompt: "Test prompt".to_string(),
            instruction: "Analyze.".to_string(),
            max_response_tokens: 100,
            config_override: None,
        };

        let response = engine.process(&request, &model).unwrap();
        // Cost should be > 0 since we processed tokens
        assert!(response.stats.total_input_tokens > 0);
    }

    #[test]
    fn test_transform_parsing() {
        let op = parse_transform_op("split(,)");
        assert!(matches!(op, TransformOp::Split { .. }));

        let op = parse_transform_op("sort");
        assert!(matches!(op, TransformOp::Sort));

        let op = parse_transform_op("count");
        assert!(matches!(op, TransformOp::Count));

        let op = parse_transform_op("grep(keyword)");
        assert!(matches!(op, TransformOp::GrepLines { .. }));
    }

    #[test]
    fn test_strip_prefix_case_insensitive() {
        assert_eq!(
            strip_prefix_case_insensitive("PEEK(1,2)", "PEEK("),
            Some("1,2)")
        );
        assert_eq!(
            strip_prefix_case_insensitive("peek(1,2)", "PEEK("),
            Some("1,2)")
        );
        assert_eq!(
            strip_prefix_case_insensitive("Peek(1,2)", "PEEK("),
            Some("1,2)")
        );
        assert_eq!(strip_prefix_case_insensitive("OTHER", "PEEK("), None);
    }

    #[test]
    fn test_global_stats_accumulation() {
        let config = RecursiveEngineConfig {
            env_config: PromptEnvConfig {
                max_iterations: 5,
                ..Default::default()
            },
            ..Default::default()
        };

        let engine = RecursiveInferenceEngine::new(config);
        let model = MockInferenceModel::auto_resolve(32768);

        // Process two requests
        for i in 0..2 {
            let request = RecursiveRequest {
                request_id: format!("test-{}", i),
                prompt: "Test".to_string(),
                instruction: "Do it.".to_string(),
                max_response_tokens: 100,
                config_override: None,
            };
            engine.process(&request, &model).unwrap();
        }

        let stats = engine.global_stats();
        assert_eq!(stats.total_requests, 2);
    }
}
