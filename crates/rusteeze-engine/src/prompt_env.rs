// ============================================================================
// Rusteeze - Recursive Language Model Infrastructure
// Prompt Environment: REPL state management for recursive inference
// Based on: "Recursive Language Models" (Zhang, Kraska, Khattab 2026)
// ============================================================================
//
// The PromptEnvironment manages the REPL state for the RLM inference loop.
// It encapsulates:
// 1. The user prompt stored as an external variable (NOT in context window)
// 2. A VariableStore for intermediate results
// 3. Operations that the model can perform on the prompt (peek, slice, decompose)
// 4. Metadata generation for constant-size context window injection
// 5. Sub-RLM call tracking and management
//
// This is the Rust equivalent of the Python REPL in the paper.
// ============================================================================

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::variable_store::{VarId, VarStoreError, VarValue, VariableStore};

/// Configuration for the prompt environment
#[derive(Debug, Clone)]
pub struct PromptEnvConfig {
    /// Maximum number of RLM iterations (root level)
    pub max_iterations: usize,
    /// Maximum recursion depth for sub-calls
    pub max_recursion_depth: usize,
    /// Maximum total sub-calls across all levels
    pub max_total_subcalls: usize,
    /// Context window size of the underlying model (tokens)
    pub context_window_tokens: usize,
    /// Maximum characters in metadata prefix
    pub metadata_prefix_len: usize,
    /// Maximum bytes for variable store
    pub max_memory_bytes: usize,
    /// Timeout for the entire RLM session
    pub session_timeout: Duration,
    /// Whether to enable asynchronous sub-calls (rayon parallelism)
    pub enable_async_subcalls: bool,
    /// Maximum number of concurrent sub-calls
    pub max_concurrent_subcalls: usize,
}

impl Default for PromptEnvConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_recursion_depth: 3,
            max_total_subcalls: 1000,
            context_window_tokens: 32768,
            metadata_prefix_len: 128,
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            session_timeout: Duration::from_secs(600),
            enable_async_subcalls: true,
            max_concurrent_subcalls: 16,
        }
    }
}

/// The operation requested by the model at each iteration
#[derive(Debug, Clone)]
pub enum ReplOperation {
    /// View a slice of the prompt: peek(start, end)
    Peek {
        byte_start: usize,
        byte_end: usize,
    },

    /// Decompose the prompt into chunks: decompose(chunk_size, overlap)
    Decompose {
        chunk_size: usize,
        overlap: usize,
    },

    /// Invoke a sub-RLM on a text slice: subcall(prompt_slice, instruction)
    SubCall {
        prompt_text: String,
        instruction: String,
    },

    /// Invoke sub-RLM on a variable: subcall_var(var_id, instruction)
    SubCallVar {
        var_id: VarId,
        instruction: String,
    },

    /// Batch sub-calls on chunks: batch_subcall(chunks, instruction)
    BatchSubCall {
        chunk_var_id: VarId,
        instruction: String,
    },

    /// Set a variable: set_var(name, value)
    SetVar {
        name: String,
        value: VarValue,
    },

    /// Get a variable: get_var(name)
    GetVar {
        name: String,
    },

    /// Regex search on prompt: regex_search(pattern)
    RegexSearch {
        pattern: String,
    },

    /// Set the final output (triggers termination)
    SetFinal {
        value: String,
    },

    /// Set final to a variable reference (triggers termination)
    SetFinalVar {
        var_name: String,
    },

    /// Execute a transform: apply function to variable
    Transform {
        source_var: VarId,
        target_var: VarId,
        transform: TransformOp,
    },

    /// No-op (model is still thinking)
    Think {
        thought: String,
    },
}

/// Transformation operations on variables
#[derive(Debug, Clone)]
pub enum TransformOp {
    /// Split text by delimiter
    Split { delimiter: String },
    /// Join text list with delimiter
    Join { delimiter: String },
    /// Filter text list by predicate (keyword contains)
    FilterContains { keyword: String },
    /// Sort text list alphabetically
    Sort,
    /// Count items
    Count,
    /// Truncate text to N chars
    Truncate { max_chars: usize },
    /// Extract lines matching pattern
    GrepLines { pattern: String },
    /// Map: apply sub-RLM to each item in list
    MapSubCall { instruction: String },
    /// Reduce: aggregate list items with sub-RLM
    ReduceSubCall { instruction: String },
}

/// Result of executing an operation
#[derive(Debug, Clone)]
pub enum OpResult {
    /// Operation completed successfully with output
    Success {
        output: String,
        variables_modified: Vec<VarId>,
    },

    /// Sub-call request that needs to be executed
    PendingSubCall {
        call_id: u64,
        prompt: String,
        instruction: String,
        result_var: VarId,
    },

    /// Batch of sub-calls to execute (potentially in parallel)
    PendingBatchSubCalls {
        calls: Vec<PendingCall>,
    },

    /// Final output has been set, terminate
    FinalSet {
        output: String,
    },

    /// Error during operation
    Error {
        message: String,
    },
}

/// A pending sub-call to be executed
#[derive(Debug, Clone)]
pub struct PendingCall {
    pub call_id: u64,
    pub prompt: String,
    pub instruction: String,
    pub result_var: VarId,
}

/// Statistics for the prompt environment
#[derive(Debug, Clone, Default)]
pub struct EnvStats {
    pub iterations: u64,
    pub total_subcalls: u64,
    pub total_tokens_in_context: u64,
    pub total_tokens_generated: u64,
    pub max_recursion_depth_reached: usize,
    pub peak_memory_bytes: usize,
    pub total_transform_ops: u64,
    pub elapsed: Duration,
}

/// The Prompt Environment: REPL state for recursive inference
///
/// This is the core runtime environment for the RLM loop. It:
/// 1. Stores the prompt as a variable, NOT in the context window
/// 2. Provides operations for the model to examine/decompose the prompt  
/// 3. Manages intermediate variables and sub-call results
/// 4. Generates constant-size metadata for context window injection
/// 5. Tracks recursion depth and resource limits
pub struct PromptEnvironment {
    /// Configuration
    config: PromptEnvConfig,
    /// Variable store (the "REPL state")
    store: VariableStore,
    /// Current iteration number
    iteration: u64,
    /// Current recursion depth (0 = root)
    recursion_depth: usize,
    /// Parent environment ID (for sub-calls)
    parent_id: Option<u64>,
    /// Unique ID for this environment
    env_id: u64,
    /// Next sub-call ID
    next_subcall_id: u64,
    /// Statistics
    stats: EnvStats,
    /// History of model outputs per iteration (trimmed for context)
    iteration_history: Vec<IterationRecord>,
    /// Session start time
    start_time: Instant,
    /// Whether the environment has terminated
    terminated: bool,
}

/// Record of a single iteration in the RLM loop
#[derive(Debug, Clone)]
pub struct IterationRecord {
    pub iteration: u64,
    pub operation: String,
    pub output_preview: String,
    pub tokens_used: usize,
    pub subcalls_launched: usize,
    pub duration: Duration,
}

impl PromptEnvironment {
    /// Create a new root-level prompt environment
    pub fn new_root(prompt: String, config: PromptEnvConfig) -> Self {
        let mut store = VariableStore::with_max_memory(config.max_memory_bytes);

        // Store the prompt as an external variable (Algorithm 1, line 1)
        store
            .set(
                VarId::prompt(),
                VarValue::Text(Arc::from(prompt.as_str())),
            )
            .expect("Failed to store initial prompt");

        Self {
            config,
            store,
            iteration: 0,
            recursion_depth: 0,
            parent_id: None,
            env_id: 0,
            next_subcall_id: 1,
            stats: EnvStats::default(),
            iteration_history: Vec::new(),
            start_time: Instant::now(),
            terminated: false,
        }
    }

    /// Create a child environment for a sub-RLM call
    pub fn new_subcall(
        prompt: String,
        instruction: String,
        parent_id: u64,
        recursion_depth: usize,
        env_id: u64,
        config: PromptEnvConfig,
    ) -> Self {
        let mut store = VariableStore::with_max_memory(config.max_memory_bytes);

        store
            .set(
                VarId::prompt(),
                VarValue::Text(Arc::from(prompt.as_str())),
            )
            .expect("Failed to store sub-call prompt");

        // Store the instruction as a variable too
        store
            .set(
                VarId::new("__instruction__"),
                VarValue::Text(Arc::from(instruction.as_str())),
            )
            .expect("Failed to store instruction");

        Self {
            config,
            store,
            iteration: 0,
            recursion_depth,
            parent_id: Some(parent_id),
            env_id,
            next_subcall_id: 1,
            stats: EnvStats::default(),
            iteration_history: Vec::new(),
            start_time: Instant::now(),
            terminated: false,
        }
    }

    /// Generate the initial metadata for the first iteration
    /// This is what goes into the context window instead of the raw prompt
    pub fn initial_metadata(&self) -> String {
        let prompt_metadata = self
            .store
            .get_metadata(&VarId::prompt())
            .expect("Prompt must exist");

        let mut meta = format!(
            "=== RLM Environment (depth={}, env_id={}) ===\n",
            self.recursion_depth, self.env_id
        );

        meta.push_str(&format!(
            "Prompt: {} chars, type={}\n",
            prompt_metadata.length, prompt_metadata.var_type
        ));

        meta.push_str(&format!(
            "Prefix: \"{}\"\n",
            if prompt_metadata.prefix.len() > self.config.metadata_prefix_len {
                format!(
                    "{}...",
                    &prompt_metadata.prefix[..self.config.metadata_prefix_len]
                )
            } else {
                prompt_metadata.prefix.clone()
            }
        ));

        meta.push_str(&format!(
            "Context window: {} tokens\n",
            self.config.context_window_tokens
        ));

        meta.push_str(&format!(
            "Max iterations: {}, Max recursion depth: {}\n",
            self.config.max_iterations, self.config.max_recursion_depth
        ));

        if let Some(parent) = self.parent_id {
            meta.push_str(&format!("Parent env: {}\n", parent));
        }

        if let Some(instr) = self.store.peek(&VarId::new("__instruction__")) {
            meta.push_str(&format!("Instruction: {}\n", instr));
        }

        meta.push_str("\nAvailable operations:\n");
        meta.push_str("  peek(start, end) - View a byte range of the prompt\n");
        meta.push_str("  decompose(chunk_size, overlap) - Split prompt into chunks\n");
        meta.push_str("  subcall(text, instruction) - Invoke sub-RLM on text\n");
        meta.push_str("  batch_subcall(chunk_var, instruction) - Sub-RLM on each chunk\n");
        meta.push_str("  set_var(name, value) - Store intermediate result\n");
        meta.push_str("  get_var(name) - Retrieve stored variable\n");
        meta.push_str("  regex_search(pattern) - Search prompt with regex\n");
        meta.push_str("  transform(source, target, op) - Transform a variable\n");
        meta.push_str("  FINAL(value) - Set final output and terminate\n");
        meta.push_str("  FINAL_VAR(name) - Set output to variable value and terminate\n");

        meta
    }

    /// Generate metadata for a subsequent iteration
    /// Includes the history of previous iterations (trimmed) and current state
    pub fn iteration_metadata(&self, stdout: &str) -> String {
        let mut meta = format!(
            "=== Iteration {} (depth={}) ===\n",
            self.iteration, self.recursion_depth
        );

        // Include trimmed stdout from last operation
        if !stdout.is_empty() {
            let trimmed = if stdout.len() > 500 {
                format!(
                    "{}... [truncated, {} chars total]",
                    &stdout[..500],
                    stdout.len()
                )
            } else {
                stdout.to_string()
            };
            meta.push_str(&format!("Output: {}\n", trimmed));
        }

        // Include current variable state
        meta.push_str(&self.store.all_metadata_string());
        meta.push_str("\n");

        // Include summary of recent iterations
        let recent_count = 5.min(self.iteration_history.len());
        if recent_count > 0 {
            meta.push_str("\nRecent history:\n");
            for record in self
                .iteration_history
                .iter()
                .rev()
                .take(recent_count)
                .rev()
            {
                meta.push_str(&format!(
                    "  [{}] {} -> {} ({:?})\n",
                    record.iteration,
                    record.operation,
                    if record.output_preview.len() > 80 {
                        format!("{}...", &record.output_preview[..80])
                    } else {
                        record.output_preview.clone()
                    },
                    record.duration
                ));
            }
        }

        // Resource usage
        meta.push_str(&format!(
            "\nResources: {} subcalls used, {:.1}s elapsed\n",
            self.stats.total_subcalls,
            self.start_time.elapsed().as_secs_f64()
        ));

        meta
    }

    /// Execute an operation in the REPL environment
    pub fn execute(&mut self, op: ReplOperation) -> OpResult {
        if self.terminated {
            return OpResult::Error {
                message: "Environment has been terminated".to_string(),
            };
        }

        // Check timeout
        if self.start_time.elapsed() > self.config.session_timeout {
            self.terminated = true;
            return OpResult::Error {
                message: "Session timeout exceeded".to_string(),
            };
        }

        // Check iteration limit
        if self.iteration as usize >= self.config.max_iterations {
            self.terminated = true;
            return OpResult::Error {
                message: format!(
                    "Maximum iterations ({}) exceeded",
                    self.config.max_iterations
                ),
            };
        }

        let iter_start = Instant::now();
        self.iteration += 1;
        self.stats.iterations = self.iteration;

        let (result, op_name) = match op {
            ReplOperation::Peek {
                byte_start,
                byte_end,
            } => (self.execute_peek(byte_start, byte_end), "peek"),

            ReplOperation::Decompose {
                chunk_size,
                overlap,
            } => (self.execute_decompose(chunk_size, overlap), "decompose"),

            ReplOperation::SubCall {
                prompt_text,
                instruction,
            } => (
                self.execute_subcall(prompt_text, instruction),
                "subcall",
            ),

            ReplOperation::SubCallVar {
                var_id,
                instruction,
            } => (
                self.execute_subcall_var(&var_id, &instruction),
                "subcall_var",
            ),

            ReplOperation::BatchSubCall {
                chunk_var_id,
                instruction,
            } => (
                self.execute_batch_subcall(&chunk_var_id, &instruction),
                "batch_subcall",
            ),

            ReplOperation::SetVar { name, value } => {
                (self.execute_set_var(name, value), "set_var")
            }

            ReplOperation::GetVar { name } => (self.execute_get_var(&name), "get_var"),

            ReplOperation::RegexSearch { pattern } => {
                (self.execute_regex_search(&pattern), "regex_search")
            }

            ReplOperation::SetFinal { value } => {
                (self.execute_set_final(value), "FINAL")
            }

            ReplOperation::SetFinalVar { var_name } => {
                (self.execute_set_final_var(&var_name), "FINAL_VAR")
            }

            ReplOperation::Transform {
                source_var,
                target_var,
                transform,
            } => (
                self.execute_transform(&source_var, &target_var, transform),
                "transform",
            ),

            ReplOperation::Think { thought } => (
                OpResult::Success {
                    output: format!("[Thinking: {}]", thought),
                    variables_modified: vec![],
                },
                "think",
            ),
        };

        // Record iteration
        let output_preview = match &result {
            OpResult::Success { output, .. } => output.clone(),
            OpResult::PendingSubCall { instruction, .. } => {
                format!("subcall: {}", instruction)
            }
            OpResult::PendingBatchSubCalls { calls } => {
                format!("batch_subcalls: {} calls", calls.len())
            }
            OpResult::FinalSet { output } => format!("FINAL: {}", output),
            OpResult::Error { message } => format!("ERROR: {}", message),
        };

        let subcalls_in_iter = match &result {
            OpResult::PendingSubCall { .. } => 1,
            OpResult::PendingBatchSubCalls { calls } => calls.len(),
            _ => 0,
        };

        self.iteration_history.push(IterationRecord {
            iteration: self.iteration,
            operation: op_name.to_string(),
            output_preview,
            tokens_used: 0, // Will be filled by the engine
            subcalls_launched: subcalls_in_iter,
            duration: iter_start.elapsed(),
        });

        // Update stats
        self.stats.elapsed = self.start_time.elapsed();
        self.stats.peak_memory_bytes = self
            .stats
            .peak_memory_bytes
            .max(self.store.total_memory());

        result
    }

    // ========================================================================
    // Operation implementations
    // ========================================================================

    fn execute_peek(&mut self, byte_start: usize, byte_end: usize) -> OpResult {
        match self.store.peek(&VarId::prompt()) {
            Some(VarValue::Text(prompt)) => {
                let start = byte_start.min(prompt.len());
                let end = byte_end.min(prompt.len());

                if start >= end {
                    return OpResult::Error {
                        message: format!(
                            "Invalid byte range: [{}..{}] for prompt of {} bytes",
                            byte_start,
                            byte_end,
                            prompt.len()
                        ),
                    };
                }

                // Find valid UTF-8 boundaries
                let safe_start = find_char_boundary(prompt.as_ref(), start);
                let safe_end = find_char_boundary(prompt.as_ref(), end);

                let slice = &prompt[safe_start..safe_end];

                OpResult::Success {
                    output: format!(
                        "Bytes[{}..{}] ({} chars):\n{}",
                        safe_start,
                        safe_end,
                        slice.chars().count(),
                        slice
                    ),
                    variables_modified: vec![],
                }
            }
            _ => OpResult::Error {
                message: "Prompt not found or not text".to_string(),
            },
        }
    }

    fn execute_decompose(&mut self, chunk_size: usize, overlap: usize) -> OpResult {
        let chunks = match self.store.peek(&VarId::prompt()) {
            Some(VarValue::Text(prompt)) => {
                adaptive_chunk(prompt.as_ref(), chunk_size, overlap)
            }
            _ => {
                return OpResult::Error {
                    message: "Prompt not found or not text".to_string(),
                }
            }
        };

        let num_chunks = chunks.len();
        let chunk_arcs: Vec<Arc<str>> =
            chunks.into_iter().map(|c| Arc::from(c.as_str())).collect();

        let var_id = VarId::new("__chunks__");
        match self.store.set(var_id.clone(), VarValue::TextList(chunk_arcs)) {
            Ok(()) => OpResult::Success {
                output: format!(
                    "Decomposed prompt into {} chunks (size={}, overlap={}). Stored in '__chunks__'.",
                    num_chunks, chunk_size, overlap
                ),
                variables_modified: vec![var_id],
            },
            Err(e) => OpResult::Error {
                message: format!("Failed to store chunks: {}", e),
            },
        }
    }

    fn execute_subcall(
        &mut self,
        prompt_text: String,
        instruction: String,
    ) -> OpResult {
        // Check recursion depth
        if self.recursion_depth >= self.config.max_recursion_depth {
            return OpResult::Error {
                message: format!(
                    "Maximum recursion depth ({}) exceeded",
                    self.config.max_recursion_depth
                ),
            };
        }

        // Check total sub-calls
        if self.stats.total_subcalls >= self.config.max_total_subcalls as u64 {
            return OpResult::Error {
                message: format!(
                    "Maximum total sub-calls ({}) exceeded",
                    self.config.max_total_subcalls
                ),
            };
        }

        let call_id = self.next_subcall_id;
        self.next_subcall_id += 1;
        self.stats.total_subcalls += 1;

        let result_var = VarId::new(format!("subcall_result_{}", call_id));

        OpResult::PendingSubCall {
            call_id,
            prompt: prompt_text,
            instruction,
            result_var,
        }
    }

    fn execute_subcall_var(
        &mut self,
        var_id: &VarId,
        instruction: &str,
    ) -> OpResult {
        // Resolve the variable to text
        let text = match self.store.peek(var_id) {
            Some(VarValue::Text(s)) => s.to_string(),
            Some(VarValue::TextList(list)) => list
                .iter()
                .map(|s| s.as_ref())
                .collect::<Vec<_>>()
                .join("\n"),
            Some(other) => other.to_text_repr(),
            None => {
                return OpResult::Error {
                    message: format!("Variable '{}' not found", var_id),
                }
            }
        };

        self.execute_subcall(text, instruction.to_string())
    }

    fn execute_batch_subcall(
        &mut self,
        chunk_var_id: &VarId,
        instruction: &str,
    ) -> OpResult {
        // Resolve the chunks list
        let chunks = match self.store.peek(chunk_var_id) {
            Some(VarValue::TextList(list)) => list.clone(),
            _ => {
                return OpResult::Error {
                    message: format!(
                        "Variable '{}' not found or not a text list",
                        chunk_var_id
                    ),
                }
            }
        };

        // Check limits
        let num_calls = chunks.len();
        if self.stats.total_subcalls + num_calls as u64 > self.config.max_total_subcalls as u64 {
            return OpResult::Error {
                message: format!(
                    "Batch of {} sub-calls would exceed limit ({})",
                    num_calls, self.config.max_total_subcalls
                ),
            };
        }

        if self.recursion_depth >= self.config.max_recursion_depth {
            return OpResult::Error {
                message: format!(
                    "Maximum recursion depth ({}) exceeded",
                    self.config.max_recursion_depth
                ),
            };
        }

        let calls: Vec<PendingCall> = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let call_id = self.next_subcall_id;
                self.next_subcall_id += 1;
                self.stats.total_subcalls += 1;

                PendingCall {
                    call_id,
                    prompt: chunk.to_string(),
                    instruction: instruction.to_string(),
                    result_var: VarId::new(format!("batch_result_{}", call_id)),
                }
            })
            .collect();

        OpResult::PendingBatchSubCalls { calls }
    }

    fn execute_set_var(&mut self, name: String, value: VarValue) -> OpResult {
        let var_id = VarId::new(&name);
        match self.store.set(var_id.clone(), value) {
            Ok(()) => OpResult::Success {
                output: format!("Variable '{}' set successfully", name),
                variables_modified: vec![var_id],
            },
            Err(e) => OpResult::Error {
                message: format!("Failed to set variable '{}': {}", name, e),
            },
        }
    }

    fn execute_get_var(&mut self, name: &str) -> OpResult {
        let var_id = VarId::new(name);
        match self.store.get(&var_id) {
            Some(value) => OpResult::Success {
                output: format!("{} = {}", name, value.to_text_repr()),
                variables_modified: vec![],
            },
            None => OpResult::Error {
                message: format!("Variable '{}' not found", name),
            },
        }
    }

    fn execute_regex_search(&mut self, pattern: &str) -> OpResult {
        let prompt = match self.store.peek(&VarId::prompt()) {
            Some(VarValue::Text(s)) => s.clone(),
            _ => {
                return OpResult::Error {
                    message: "Prompt not found or not text".to_string(),
                }
            }
        };

        // Simple keyword search (avoiding regex crate dependency for now)
        // In production, use the regex crate
        let pattern_lower = pattern.to_lowercase();
        let prompt_lower = prompt.to_lowercase();

        let mut matches: Vec<(usize, String)> = Vec::new();
        let mut search_start = 0;

        while let Some(pos) = prompt_lower[search_start..].find(&pattern_lower) {
            let abs_pos = search_start + pos;
            // Extract context around the match
            let context_start = abs_pos.saturating_sub(50);
            let context_end = (abs_pos + pattern.len() + 50).min(prompt.len());

            // Find UTF-8 boundaries
            let safe_start = find_char_boundary(prompt.as_ref(), context_start);
            let safe_end = find_char_boundary(prompt.as_ref(), context_end);

            matches.push((abs_pos, prompt[safe_start..safe_end].to_string()));

            search_start = abs_pos + pattern.len();
            if search_start >= prompt_lower.len() {
                break;
            }

            // Limit matches to prevent explosion
            if matches.len() >= 50 {
                break;
            }
        }

        let output = if matches.is_empty() {
            format!("No matches found for pattern '{}'", pattern)
        } else {
            let mut result = format!(
                "Found {} matches for '{}'\n",
                matches.len(),
                pattern
            );
            for (i, (pos, context)) in matches.iter().enumerate() {
                result.push_str(&format!(
                    "  [{}] byte {}: ...{}...\n",
                    i, pos, context
                ));
            }
            result
        };

        OpResult::Success {
            output,
            variables_modified: vec![],
        }
    }

    fn execute_set_final(&mut self, value: String) -> OpResult {
        match self.store.set(
            VarId::final_output(),
            VarValue::Text(Arc::from(value.as_str())),
        ) {
            Ok(()) => {
                self.terminated = true;
                OpResult::FinalSet {
                    output: value,
                }
            }
            Err(e) => OpResult::Error {
                message: format!("Failed to set final output: {}", e),
            },
        }
    }

    fn execute_set_final_var(&mut self, var_name: &str) -> OpResult {
        // Store the variable name reference
        match self.store.set(
            VarId::final_var(),
            VarValue::Text(Arc::from(var_name)),
        ) {
            Ok(()) => {
                self.terminated = true;
                let resolved = self.store.resolve_final().unwrap_or_default();
                OpResult::FinalSet { output: resolved }
            }
            Err(e) => OpResult::Error {
                message: format!("Failed to set final var: {}", e),
            },
        }
    }

    fn execute_transform(
        &mut self,
        source_var: &VarId,
        target_var: &VarId,
        transform: TransformOp,
    ) -> OpResult {
        self.stats.total_transform_ops += 1;

        let source_value = match self.store.peek(source_var) {
            Some(v) => v.clone(),
            None => {
                return OpResult::Error {
                    message: format!("Source variable '{}' not found", source_var),
                }
            }
        };

        let result_value = match transform {
            TransformOp::Split { ref delimiter } => match &source_value {
                VarValue::Text(s) => {
                    let parts: Vec<Arc<str>> = s
                        .split(delimiter.as_str())
                        .map(|p| Arc::from(p))
                        .collect();
                    VarValue::TextList(parts)
                }
                _ => {
                    return OpResult::Error {
                        message: format!(
                            "Split requires text, got {}",
                            source_value.type_name()
                        ),
                    }
                }
            },

            TransformOp::Join { ref delimiter } => match &source_value {
                VarValue::TextList(list) => {
                    let joined: String = list
                        .iter()
                        .map(|s| s.as_ref())
                        .collect::<Vec<_>>()
                        .join(delimiter.as_str());
                    VarValue::Text(Arc::from(joined.as_str()))
                }
                _ => {
                    return OpResult::Error {
                        message: format!(
                            "Join requires text_list, got {}",
                            source_value.type_name()
                        ),
                    }
                }
            },

            TransformOp::FilterContains { ref keyword } => match &source_value {
                VarValue::TextList(list) => {
                    let keyword_lower = keyword.to_lowercase();
                    let filtered: Vec<Arc<str>> = list
                        .iter()
                        .filter(|s| s.to_lowercase().contains(&keyword_lower))
                        .cloned()
                        .collect();
                    VarValue::TextList(filtered)
                }
                _ => {
                    return OpResult::Error {
                        message: format!(
                            "Filter requires text_list, got {}",
                            source_value.type_name()
                        ),
                    }
                }
            },

            TransformOp::Sort => match &source_value {
                VarValue::TextList(list) => {
                    let mut sorted = list.clone();
                    sorted.sort();
                    VarValue::TextList(sorted)
                }
                _ => {
                    return OpResult::Error {
                        message: format!(
                            "Sort requires text_list, got {}",
                            source_value.type_name()
                        ),
                    }
                }
            },

            TransformOp::Count => match &source_value {
                VarValue::TextList(list) => VarValue::Number(list.len() as f64),
                VarValue::NumberList(list) => VarValue::Number(list.len() as f64),
                VarValue::Text(s) => VarValue::Number(s.len() as f64),
                VarValue::Tokens(t) => VarValue::Number(t.len() as f64),
                _ => VarValue::Number(0.0),
            },

            TransformOp::Truncate { max_chars } => match &source_value {
                VarValue::Text(s) => {
                    let truncated: String = s.chars().take(max_chars).collect();
                    VarValue::Text(Arc::from(truncated.as_str()))
                }
                _ => {
                    return OpResult::Error {
                        message: format!(
                            "Truncate requires text, got {}",
                            source_value.type_name()
                        ),
                    }
                }
            },

            TransformOp::GrepLines { ref pattern } => match &source_value {
                VarValue::Text(s) => {
                    let pattern_lower = pattern.to_lowercase();
                    let matched: Vec<Arc<str>> = s
                        .lines()
                        .filter(|line| line.to_lowercase().contains(&pattern_lower))
                        .map(|line| Arc::from(line))
                        .collect();
                    VarValue::TextList(matched)
                }
                _ => {
                    return OpResult::Error {
                        message: format!(
                            "GrepLines requires text, got {}",
                            source_value.type_name()
                        ),
                    }
                }
            },

            TransformOp::MapSubCall { instruction } => {
                // This produces pending sub-calls, not an immediate result
                match &source_value {
                    VarValue::TextList(list) => {
                        let calls: Vec<PendingCall> = list
                            .iter()
                            .enumerate()
                            .map(|(i, item)| {
                                let call_id = self.next_subcall_id;
                                self.next_subcall_id += 1;
                                self.stats.total_subcalls += 1;
                                PendingCall {
                                    call_id,
                                    prompt: item.to_string(),
                                    instruction: instruction.clone(),
                                    result_var: VarId::new(format!(
                                        "{}_map_{}",
                                        target_var, i
                                    )),
                                }
                            })
                            .collect();

                        return OpResult::PendingBatchSubCalls { calls };
                    }
                    _ => {
                        return OpResult::Error {
                            message: format!(
                                "MapSubCall requires text_list, got {}",
                                source_value.type_name()
                            ),
                        }
                    }
                }
            }

            TransformOp::ReduceSubCall { instruction } => {
                // Reduce requires sequential sub-calls
                match &source_value {
                    VarValue::TextList(list) => {
                        let combined: String = list
                            .iter()
                            .map(|s| s.as_ref())
                            .collect::<Vec<_>>()
                            .join("\n---\n");

                        let call_id = self.next_subcall_id;
                        self.next_subcall_id += 1;
                        self.stats.total_subcalls += 1;

                        return OpResult::PendingSubCall {
                            call_id,
                            prompt: combined,
                            instruction,
                            result_var: target_var.clone(),
                        };
                    }
                    _ => {
                        return OpResult::Error {
                            message: format!(
                                "ReduceSubCall requires text_list, got {}",
                                source_value.type_name()
                            ),
                        }
                    }
                }
            }
        };

        match self.store.set(target_var.clone(), result_value) {
            Ok(()) => OpResult::Success {
                output: format!(
                    "Transform applied: {} -> {}",
                    source_var, target_var
                ),
                variables_modified: vec![target_var.clone()],
            },
            Err(e) => OpResult::Error {
                message: format!("Failed to store transform result: {}", e),
            },
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the current iteration number
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Get the recursion depth
    pub fn recursion_depth(&self) -> usize {
        self.recursion_depth
    }

    /// Get the environment ID
    pub fn env_id(&self) -> u64 {
        self.env_id
    }

    /// Check if the environment has terminated
    pub fn is_terminated(&self) -> bool {
        self.terminated || self.store.has_final()
    }

    /// Get resolved final output
    pub fn final_output(&self) -> Option<String> {
        self.store.resolve_final()
    }

    /// Get the variable store (immutable)
    pub fn store(&self) -> &VariableStore {
        &self.store
    }

    /// Get the variable store (mutable)
    pub fn store_mut(&mut self) -> &mut VariableStore {
        &mut self.store
    }

    /// Get environment statistics
    pub fn stats(&self) -> &EnvStats {
        &self.stats
    }

    /// Get iteration history
    pub fn history(&self) -> &[IterationRecord] {
        &self.iteration_history
    }

    /// Get the config
    pub fn config(&self) -> &PromptEnvConfig {
        &self.config
    }

    /// Store a sub-call result back into the environment
    pub fn store_subcall_result(
        &mut self,
        result_var: VarId,
        result: String,
    ) -> Result<(), VarStoreError> {
        self.store
            .set(result_var, VarValue::Text(Arc::from(result.as_str())))
    }

    /// Store batch sub-call results
    pub fn store_batch_results(
        &mut self,
        results: Vec<(VarId, String)>,
    ) -> Result<(), VarStoreError> {
        for (var_id, result) in results {
            self.store
                .set(var_id, VarValue::Text(Arc::from(result.as_str())))?;
        }
        Ok(())
    }
}

impl fmt::Display for PromptEnvironment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PromptEnv(id={}, depth={}, iter={}, vars={}, terminated={})",
            self.env_id,
            self.recursion_depth,
            self.iteration,
            self.store.len(),
            self.terminated
        )
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Find the nearest valid UTF-8 character boundary at or before `index`
fn find_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Adaptively chunk text respecting sentence/paragraph boundaries
///
/// This implements the chunking strategy described in the paper's Section 4.1:
/// RLMs chunk by newlines or sentences to create manageable sub-call units.
pub fn adaptive_chunk(text: &str, target_chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![];
    }

    if text.len() <= target_chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut position = 0;

    while position < text.len() {
        let chunk_end = (position + target_chunk_size).min(text.len());

        // Try to find a good break point (newline or sentence end)
        let break_point = if chunk_end < text.len() {
            find_break_point(text, position, chunk_end)
        } else {
            chunk_end
        };

        // Find safe UTF-8 boundaries
        let safe_start = find_char_boundary(text, position);
        let safe_end = find_char_boundary(text, break_point);

        if safe_start < safe_end {
            chunks.push(text[safe_start..safe_end].to_string());
        }

        // Advance position with overlap
        let advance = if break_point > position + overlap {
            break_point - overlap
        } else {
            break_point
        };

        if advance <= position {
            // Avoid infinite loop
            position = chunk_end;
        } else {
            position = advance;
        }
    }

    chunks
}

/// Find the best breaking point near `target` between `start` and `target`
fn find_break_point(text: &str, start: usize, target: usize) -> usize {
    let search_start = if target > 200 { target - 200 } else { start };

    // Priority: paragraph break > sentence end > line break > word break
    let search_region = &text[search_start..target];

    // Try paragraph break (double newline)
    if let Some(pos) = search_region.rfind("\n\n") {
        return search_start + pos + 2;
    }

    // Try sentence end
    for end_marker in &[". ", "! ", "? ", ".\n", "!\n", "?\n"] {
        if let Some(pos) = search_region.rfind(end_marker) {
            return search_start + pos + end_marker.len();
        }
    }

    // Try line break
    if let Some(pos) = search_region.rfind('\n') {
        return search_start + pos + 1;
    }

    // Try word break
    if let Some(pos) = search_region.rfind(' ') {
        return search_start + pos + 1;
    }

    // Fall back to target position
    target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_env_creation() {
        let prompt = "This is a test prompt with some content.".to_string();
        let config = PromptEnvConfig::default();
        let env = PromptEnvironment::new_root(prompt.clone(), config);

        assert_eq!(env.iteration(), 0);
        assert_eq!(env.recursion_depth(), 0);
        assert!(!env.is_terminated());
        assert!(env.store().peek(&VarId::prompt()).is_some());
    }

    #[test]
    fn test_initial_metadata() {
        let prompt = "Hello world ".repeat(100);
        let config = PromptEnvConfig::default();
        let env = PromptEnvironment::new_root(prompt, config);

        let metadata = env.initial_metadata();
        assert!(metadata.contains("RLM Environment"));
        assert!(metadata.contains("depth=0"));
        assert!(metadata.contains("Available operations"));
    }

    #[test]
    fn test_peek_operation() {
        let prompt = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_string();
        let config = PromptEnvConfig::default();
        let mut env = PromptEnvironment::new_root(prompt, config);

        let result = env.execute(ReplOperation::Peek {
            byte_start: 0,
            byte_end: 5,
        });

        match result {
            OpResult::Success { output, .. } => {
                assert!(output.contains("ABCDE"));
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_decompose_operation() {
        let prompt = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n".to_string();
        let config = PromptEnvConfig::default();
        let mut env = PromptEnvironment::new_root(prompt, config);

        let result = env.execute(ReplOperation::Decompose {
            chunk_size: 15,
            overlap: 0,
        });

        match result {
            OpResult::Success { output, .. } => {
                assert!(output.contains("chunks"));
                // Verify chunks were stored
                assert!(env.store().peek(&VarId::new("__chunks__")).is_some());
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_subcall_operation() {
        let prompt = "Test prompt".to_string();
        let config = PromptEnvConfig::default();
        let mut env = PromptEnvironment::new_root(prompt, config);

        let result = env.execute(ReplOperation::SubCall {
            prompt_text: "sub prompt".to_string(),
            instruction: "Summarize this".to_string(),
        });

        match result {
            OpResult::PendingSubCall {
                call_id,
                prompt,
                instruction,
                ..
            } => {
                assert_eq!(call_id, 1);
                assert_eq!(prompt, "sub prompt");
                assert_eq!(instruction, "Summarize this");
            }
            _ => panic!("Expected PendingSubCall"),
        }
    }

    #[test]
    fn test_set_final() {
        let prompt = "Test".to_string();
        let config = PromptEnvConfig::default();
        let mut env = PromptEnvironment::new_root(prompt, config);

        let result = env.execute(ReplOperation::SetFinal {
            value: "The answer".to_string(),
        });

        match result {
            OpResult::FinalSet { output } => {
                assert_eq!(output, "The answer");
            }
            _ => panic!("Expected FinalSet"),
        }

        assert!(env.is_terminated());
        assert_eq!(env.final_output().unwrap(), "The answer");
    }

    #[test]
    fn test_regex_search() {
        let prompt =
            "The quick brown fox jumps over the lazy dog. The fox is clever.".to_string();
        let config = PromptEnvConfig::default();
        let mut env = PromptEnvironment::new_root(prompt, config);

        let result = env.execute(ReplOperation::RegexSearch {
            pattern: "fox".to_string(),
        });

        match result {
            OpResult::Success { output, .. } => {
                assert!(output.contains("Found 2 matches"));
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_transform_split_join() {
        let prompt = "a,b,c,d,e".to_string();
        let config = PromptEnvConfig::default();
        let mut env = PromptEnvironment::new_root(prompt, config);

        // Set a variable with CSV data
        env.execute(ReplOperation::SetVar {
            name: "data".to_string(),
            value: VarValue::Text(Arc::from("a,b,c,d,e")),
        });

        // Split it
        let result = env.execute(ReplOperation::Transform {
            source_var: VarId::new("data"),
            target_var: VarId::new("parts"),
            transform: TransformOp::Split {
                delimiter: ",".to_string(),
            },
        });
        assert!(matches!(result, OpResult::Success { .. }));

        // Verify
        let parts = env.store().peek(&VarId::new("parts")).unwrap();
        assert_eq!(parts.as_text_list().unwrap().len(), 5);

        // Join it back
        env.execute(ReplOperation::Transform {
            source_var: VarId::new("parts"),
            target_var: VarId::new("joined"),
            transform: TransformOp::Join {
                delimiter: " | ".to_string(),
            },
        });

        let joined = env.store().peek(&VarId::new("joined")).unwrap();
        assert_eq!(joined.as_text().unwrap(), "a | b | c | d | e");
    }

    #[test]
    fn test_iteration_limit() {
        let prompt = "Test".to_string();
        let config = PromptEnvConfig {
            max_iterations: 3,
            ..Default::default()
        };
        let mut env = PromptEnvironment::new_root(prompt, config);

        // Execute 3 operations (at limit)
        for _ in 0..3 {
            let r = env.execute(ReplOperation::Think {
                thought: "thinking".to_string(),
            });
            match r {
                OpResult::Success { .. } => {}
                OpResult::Error { message } => {
                    assert!(message.contains("Maximum iterations"));
                    return;
                }
                _ => panic!("Unexpected result"),
            }
        }

        // 4th should fail
        let r = env.execute(ReplOperation::Think {
            thought: "one more".to_string(),
        });
        assert!(matches!(r, OpResult::Error { .. }));
    }

    #[test]
    fn test_recursion_depth_limit() {
        let prompt = "Test".to_string();
        let config = PromptEnvConfig {
            max_recursion_depth: 0, // No sub-calls allowed
            ..Default::default()
        };
        let mut env = PromptEnvironment::new_root(prompt, config);

        let result = env.execute(ReplOperation::SubCall {
            prompt_text: "sub".to_string(),
            instruction: "do something".to_string(),
        });

        assert!(matches!(result, OpResult::Error { .. }));
    }

    #[test]
    fn test_adaptive_chunking() {
        let text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.";
        let chunks = adaptive_chunk(text, 30, 0);
        assert!(!chunks.is_empty());

        // All chunks should be non-empty
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }

        // Chunks with overlap should have more chunks
        let chunks_overlap = adaptive_chunk(text, 30, 10);
        assert!(chunks_overlap.len() >= chunks.len());
    }

    #[test]
    fn test_subcall_result_storage() {
        let prompt = "Test".to_string();
        let config = PromptEnvConfig::default();
        let mut env = PromptEnvironment::new_root(prompt, config);

        // Simulate storing a sub-call result
        env.store_subcall_result(
            VarId::new("subcall_result_1"),
            "The fox is red".to_string(),
        )
        .unwrap();

        let result = env
            .store()
            .peek(&VarId::new("subcall_result_1"))
            .unwrap();
        assert_eq!(result.as_text().unwrap(), "The fox is red");
    }

    #[test]
    fn test_batch_subcall() {
        let prompt = "Test".to_string();
        let config = PromptEnvConfig::default();
        let mut env = PromptEnvironment::new_root(prompt, config);

        // Set up chunks
        env.execute(ReplOperation::SetVar {
            name: "chunks".to_string(),
            value: VarValue::TextList(vec![
                Arc::from("chunk 1"),
                Arc::from("chunk 2"),
                Arc::from("chunk 3"),
            ]),
        });

        let result = env.execute(ReplOperation::BatchSubCall {
            chunk_var_id: VarId::new("chunks"),
            instruction: "Summarize this chunk".to_string(),
        });

        match result {
            OpResult::PendingBatchSubCalls { calls } => {
                assert_eq!(calls.len(), 3);
                assert_eq!(calls[0].instruction, "Summarize this chunk");
            }
            _ => panic!("Expected PendingBatchSubCalls"),
        }
    }

    #[test]
    fn test_child_environment() {
        let config = PromptEnvConfig::default();
        let child = PromptEnvironment::new_subcall(
            "child prompt".to_string(),
            "analyze this".to_string(),
            0,   // parent_id
            1,   // recursion_depth
            42,  // env_id
            config,
        );

        assert_eq!(child.recursion_depth(), 1);
        assert_eq!(child.env_id(), 42);

        let metadata = child.initial_metadata();
        assert!(metadata.contains("depth=1"));
        assert!(metadata.contains("Parent env: 0"));
        assert!(metadata.contains("analyze this"));
    }
}
