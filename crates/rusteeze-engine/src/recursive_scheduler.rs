// ============================================================================
// Rusteeze - Recursive Language Model Infrastructure
// Recursive Scheduler: Call-tree management for recursive inference
// Based on: "Recursive Language Models" (Zhang, Kraska, Khattab 2026)
// ============================================================================
//
// The RecursiveScheduler manages the tree of recursive inference calls.
// Unlike the flat scheduler in scheduler.rs, this one handles:
// 1. Tree-structured call hierarchies (root -> sub-calls -> sub-sub-calls)
// 2. Resource allocation across recursion levels
// 3. Parallel execution of independent sub-calls (rayon)
// 4. Backpressure when too many sub-calls are pending
// 5. Priority-based scheduling (root calls get priority)
// ============================================================================

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the recursive scheduler
#[derive(Debug, Clone)]
pub struct RecursiveSchedulerConfig {
    /// Maximum recursion depth
    pub max_depth: usize,
    /// Maximum total pending calls across all levels
    pub max_pending_calls: usize,
    /// Maximum concurrent calls at each depth level
    pub max_concurrent_per_level: usize,
    /// Timeout for individual sub-calls
    pub subcall_timeout: Duration,
    /// Whether to enable parallel execution of sub-calls
    pub enable_parallelism: bool,
    /// Maximum batch size for parallel sub-calls
    pub parallel_batch_size: usize,
    /// Priority decay factor per depth level (deeper = lower priority)
    pub priority_decay: f32,
}

impl Default for RecursiveSchedulerConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_pending_calls: 1000,
            max_concurrent_per_level: 16,
            subcall_timeout: Duration::from_secs(120),
            enable_parallelism: true,
            parallel_batch_size: 32,
            priority_decay: 0.7,
        }
    }
}

/// Unique identifier for a call in the recursion tree
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct CallId(pub u64);

impl CallId {
    pub fn root() -> Self {
        Self(0)
    }
}

impl fmt::Display for CallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Call({})", self.0)
    }
}

/// Status of a call in the recursion tree
#[derive(Debug, Clone, PartialEq)]
pub enum CallStatus {
    /// Waiting to be executed
    Pending,
    /// Currently being executed
    Running,
    /// Completed with a result
    Completed { result: String },
    /// Failed with an error
    Failed { error: String },
    /// Timed out
    TimedOut,
    /// Cancelled (e.g., parent was cancelled)
    Cancelled,
}

impl CallStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            CallStatus::Completed { .. }
                | CallStatus::Failed { .. }
                | CallStatus::TimedOut
                | CallStatus::Cancelled
        )
    }

    pub fn is_success(&self) -> bool {
        matches!(self, CallStatus::Completed { .. })
    }
}

/// A node in the recursive call tree
#[derive(Debug, Clone)]
pub struct CallNode {
    /// Unique ID
    pub id: CallId,
    /// Parent call (None for root)
    pub parent: Option<CallId>,
    /// Children calls spawned by this call
    pub children: Vec<CallId>,
    /// Depth in the recursion tree (0 = root)
    pub depth: usize,
    /// The prompt/input for this call
    pub prompt: String,
    /// The instruction/task for this call
    pub instruction: String,
    /// Variable name where result should be stored
    pub result_var: String,
    /// Current status
    pub status: CallStatus,
    /// Priority (higher = more important, decays with depth)
    pub priority: f32,
    /// Creation timestamp
    pub created_at: Instant,
    /// Start execution timestamp
    pub started_at: Option<Instant>,
    /// Completion timestamp
    pub completed_at: Option<Instant>,
    /// Token count for prompt
    pub prompt_tokens: usize,
    /// Token count for response
    pub response_tokens: usize,
}

impl CallNode {
    /// Create a new root call
    pub fn root(prompt: String, instruction: String) -> Self {
        Self {
            id: CallId::root(),
            parent: None,
            children: Vec::new(),
            depth: 0,
            prompt,
            instruction,
            result_var: String::new(),
            status: CallStatus::Pending,
            priority: 1.0,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            prompt_tokens: 0,
            response_tokens: 0,
        }
    }

    /// Create a new child call
    pub fn child(
        id: CallId,
        parent: CallId,
        depth: usize,
        prompt: String,
        instruction: String,
        result_var: String,
        priority_decay: f32,
    ) -> Self {
        let priority = priority_decay.powi(depth as i32);
        Self {
            id,
            parent: Some(parent),
            children: Vec::new(),
            depth,
            prompt,
            instruction,
            result_var,
            status: CallStatus::Pending,
            priority,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            prompt_tokens: 0,
            response_tokens: 0,
        }
    }

    /// Get execution duration
    pub fn duration(&self) -> Option<Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            (Some(start), None) => Some(start.elapsed()),
            _ => None,
        }
    }
}

/// Statistics for the recursive scheduler
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub total_calls_created: u64,
    pub total_calls_completed: u64,
    pub total_calls_failed: u64,
    pub total_calls_timed_out: u64,
    pub total_calls_cancelled: u64,
    pub max_depth_reached: usize,
    pub max_concurrent_calls: usize,
    pub total_prompt_tokens: u64,
    pub total_response_tokens: u64,
    pub avg_call_duration_ms: f64,
}

/// Output of a scheduling step
#[derive(Debug)]
pub struct RecursiveSchedulerOutput {
    /// Calls ready to be executed (ordered by priority)
    pub ready_calls: Vec<CallId>,
    /// Calls that have completed since last step
    pub completed_calls: Vec<(CallId, String)>,
    /// Whether the root call has completed
    pub root_completed: bool,
    /// Root result if completed
    pub root_result: Option<String>,
}

/// The Recursive Scheduler: manages the call tree for RLM inference
///
/// This scheduler extends the basic scheduler with tree-structured call management.
/// It handles:
/// - Spawning and tracking recursive sub-calls
/// - Resource allocation with depth-based priority decay
/// - Parallel batch execution of independent sub-calls
/// - Backpressure when limits are exceeded
/// - Timeout and cancellation propagation
pub struct RecursiveScheduler {
    /// Configuration
    config: RecursiveSchedulerConfig,
    /// All calls in the recursion tree
    calls: HashMap<CallId, CallNode>,
    /// Queue of pending calls (priority-ordered)
    pending_queue: VecDeque<CallId>,
    /// Currently running calls
    running: Vec<CallId>,
    /// Next call ID to assign
    next_call_id: u64,
    /// Statistics
    stats: SchedulerStats,
    /// Duration accumulator for averaging
    total_duration_ms: f64,
    duration_count: u64,
}

impl RecursiveScheduler {
    /// Create a new recursive scheduler
    pub fn new(config: RecursiveSchedulerConfig) -> Self {
        Self {
            config,
            calls: HashMap::new(),
            pending_queue: VecDeque::new(),
            running: Vec::new(),
            next_call_id: 0,
            stats: SchedulerStats::default(),
            total_duration_ms: 0.0,
            duration_count: 0,
        }
    }

    /// Initialize with a root call
    pub fn init_root(&mut self, prompt: String, instruction: String) -> CallId {
        let root = CallNode::root(prompt, instruction);
        let root_id = root.id;

        self.calls.insert(root_id, root);
        self.pending_queue.push_back(root_id);
        self.stats.total_calls_created += 1;

        root_id
    }

    /// Spawn a sub-call from a parent
    pub fn spawn_subcall(
        &mut self,
        parent_id: CallId,
        prompt: String,
        instruction: String,
        result_var: String,
    ) -> Result<CallId, SchedulerError> {
        // Validate parent exists
        let parent_depth = match self.calls.get(&parent_id) {
            Some(parent) => parent.depth,
            None => return Err(SchedulerError::CallNotFound(parent_id)),
        };

        // Check depth limit
        let child_depth = parent_depth + 1;
        if child_depth > self.config.max_depth {
            return Err(SchedulerError::MaxDepthExceeded {
                current: child_depth,
                max: self.config.max_depth,
            });
        }

        // Check pending limit
        let pending_count = self
            .calls
            .values()
            .filter(|c| matches!(c.status, CallStatus::Pending))
            .count();
        if pending_count >= self.config.max_pending_calls {
            return Err(SchedulerError::MaxPendingExceeded {
                current: pending_count,
                max: self.config.max_pending_calls,
            });
        }

        // Create child ID
        self.next_call_id += 1;
        let child_id = CallId(self.next_call_id);

        // Create child node
        let child = CallNode::child(
            child_id,
            parent_id,
            child_depth,
            prompt,
            instruction,
            result_var,
            self.config.priority_decay,
        );

        // Register child with parent
        if let Some(parent) = self.calls.get_mut(&parent_id) {
            parent.children.push(child_id);
        }

        // Track max depth
        self.stats.max_depth_reached = self.stats.max_depth_reached.max(child_depth);
        self.stats.total_calls_created += 1;

        // Add to calls and pending queue
        self.calls.insert(child_id, child);
        self.pending_queue.push_back(child_id);

        Ok(child_id)
    }

    /// Spawn a batch of sub-calls from a parent
    pub fn spawn_batch(
        &mut self,
        parent_id: CallId,
        prompts: Vec<String>,
        instruction: String,
        result_var_prefix: String,
    ) -> Result<Vec<CallId>, SchedulerError> {
        let mut call_ids = Vec::with_capacity(prompts.len());

        for (i, prompt) in prompts.into_iter().enumerate() {
            let result_var = format!("{}_{}", result_var_prefix, i);
            let call_id = self.spawn_subcall(
                parent_id,
                prompt,
                instruction.clone(),
                result_var,
            )?;
            call_ids.push(call_id);
        }

        Ok(call_ids)
    }

    /// Schedule next batch of calls to execute
    pub fn schedule(&mut self) -> RecursiveSchedulerOutput {
        // Check for timed-out running calls
        let now = Instant::now();
        let timed_out: Vec<CallId> = self
            .running
            .iter()
            .filter(|id| {
                if let Some(call) = self.calls.get(id) {
                    if let Some(started) = call.started_at {
                        return now.duration_since(started) > self.config.subcall_timeout;
                    }
                }
                false
            })
            .copied()
            .collect();

        for id in &timed_out {
            self.timeout_call(*id);
        }

        // Collect completed calls
        let completed_calls: Vec<(CallId, String)> = self
            .calls
            .iter()
            .filter(|(_, call)| {
                matches!(&call.status, CallStatus::Completed { .. })
                    && call.completed_at.map(|t| t.elapsed().as_millis() < 100).unwrap_or(false)
            })
            .filter_map(|(id, call)| {
                if let CallStatus::Completed { result } = &call.status {
                    Some((*id, result.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Sort pending by priority and select ready calls
        let mut ready: Vec<(CallId, f32)> = self
            .pending_queue
            .iter()
            .filter_map(|id| {
                self.calls
                    .get(id)
                    .filter(|c| matches!(c.status, CallStatus::Pending))
                    .map(|c| (*id, c.priority))
            })
            .collect();
        ready.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit concurrent calls per level
        let max_ready = self.config.max_concurrent_per_level
            - self.running.len().min(self.config.max_concurrent_per_level);
        let ready_calls: Vec<CallId> = ready
            .into_iter()
            .take(max_ready)
            .map(|(id, _)| id)
            .collect();

        // Mark ready calls as running
        for id in &ready_calls {
            if let Some(call) = self.calls.get_mut(id) {
                call.status = CallStatus::Running;
                call.started_at = Some(Instant::now());
            }
            self.running.push(*id);
            self.pending_queue.retain(|queued_id| queued_id != id);
        }

        self.stats.max_concurrent_calls = self
            .stats
            .max_concurrent_calls
            .max(self.running.len());

        // Check if root is completed
        let root_completed = self
            .calls
            .get(&CallId::root())
            .map(|c| c.status.is_terminal())
            .unwrap_or(false);

        let root_result = if root_completed {
            self.calls.get(&CallId::root()).and_then(|c| {
                if let CallStatus::Completed { result } = &c.status {
                    Some(result.clone())
                } else {
                    None
                }
            })
        } else {
            None
        };

        RecursiveSchedulerOutput {
            ready_calls,
            completed_calls,
            root_completed,
            root_result,
        }
    }

    /// Mark a call as completed with a result
    pub fn complete_call(&mut self, id: CallId, result: String) {
        if let Some(call) = self.calls.get_mut(&id) {
            call.status = CallStatus::Completed {
                result: result.clone(),
            };
            call.completed_at = Some(Instant::now());

            // Update stats
            if let Some(duration) = call.duration() {
                self.total_duration_ms += duration.as_secs_f64() * 1000.0;
                self.duration_count += 1;
                self.stats.avg_call_duration_ms =
                    self.total_duration_ms / self.duration_count as f64;
            }

            self.stats.total_calls_completed += 1;
            self.stats.total_response_tokens += call.response_tokens as u64;
        }

        // Remove from running
        self.running.retain(|running_id| *running_id != id);
    }

    /// Mark a call as failed
    pub fn fail_call(&mut self, id: CallId, error: String) {
        if let Some(call) = self.calls.get_mut(&id) {
            call.status = CallStatus::Failed {
                error: error.clone(),
            };
            call.completed_at = Some(Instant::now());
            self.stats.total_calls_failed += 1;
        }
        self.running.retain(|running_id| *running_id != id);

        // Cancel children
        let children: Vec<CallId> = self
            .calls
            .get(&id)
            .map(|c| c.children.clone())
            .unwrap_or_default();
        for child_id in children {
            self.cancel_call(child_id);
        }
    }

    /// Timeout a call
    fn timeout_call(&mut self, id: CallId) {
        if let Some(call) = self.calls.get_mut(&id) {
            call.status = CallStatus::TimedOut;
            call.completed_at = Some(Instant::now());
            self.stats.total_calls_timed_out += 1;
        }
        self.running.retain(|running_id| *running_id != id);
    }

    /// Cancel a call and all its children
    pub fn cancel_call(&mut self, id: CallId) {
        let children: Vec<CallId> = self
            .calls
            .get(&id)
            .map(|c| c.children.clone())
            .unwrap_or_default();

        if let Some(call) = self.calls.get_mut(&id) {
            if !call.status.is_terminal() {
                call.status = CallStatus::Cancelled;
                call.completed_at = Some(Instant::now());
                self.stats.total_calls_cancelled += 1;
            }
        }

        self.running.retain(|running_id| *running_id != id);
        self.pending_queue.retain(|queued_id| *queued_id != id);

        // Cascade to children
        for child_id in children {
            self.cancel_call(child_id);
        }
    }

    /// Get a call node by ID
    pub fn get_call(&self, id: &CallId) -> Option<&CallNode> {
        self.calls.get(id)
    }

    /// Get all children of a call
    pub fn get_children(&self, parent_id: &CallId) -> Vec<&CallNode> {
        self.calls
            .get(parent_id)
            .map(|parent| {
                parent
                    .children
                    .iter()
                    .filter_map(|child_id| self.calls.get(child_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if all children of a call are completed
    pub fn all_children_completed(&self, parent_id: &CallId) -> bool {
        self.calls
            .get(parent_id)
            .map(|parent| {
                parent.children.iter().all(|child_id| {
                    self.calls
                        .get(child_id)
                        .map(|c| c.status.is_terminal())
                        .unwrap_or(true)
                })
            })
            .unwrap_or(true)
    }

    /// Get results of all children of a call
    pub fn collect_children_results(
        &self,
        parent_id: &CallId,
    ) -> Vec<(String, String)> {
        self.calls
            .get(parent_id)
            .map(|parent| {
                parent
                    .children
                    .iter()
                    .filter_map(|child_id| {
                        let child = self.calls.get(child_id)?;
                        if let CallStatus::Completed { result } = &child.status {
                            Some((child.result_var.clone(), result.clone()))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the number of pending calls
    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }

    /// Get the number of running calls
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Get the total number of calls
    pub fn total_calls(&self) -> usize {
        self.calls.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Get depth distribution of calls
    pub fn depth_distribution(&self) -> HashMap<usize, usize> {
        let mut dist = HashMap::new();
        for call in self.calls.values() {
            *dist.entry(call.depth).or_insert(0) += 1;
        }
        dist
    }

    /// Get the call tree as a visual representation
    pub fn tree_to_string(&self) -> String {
        let mut result = String::new();
        self.format_subtree(&CallId::root(), 0, &mut result);
        result
    }

    fn format_subtree(&self, id: &CallId, indent: usize, result: &mut String) {
        if let Some(call) = self.calls.get(id) {
            let prefix = " ".repeat(indent * 2);
            let status = match &call.status {
                CallStatus::Pending => "PENDING",
                CallStatus::Running => "RUNNING",
                CallStatus::Completed { .. } => "DONE",
                CallStatus::Failed { .. } => "FAILED",
                CallStatus::TimedOut => "TIMEOUT",
                CallStatus::Cancelled => "CANCELLED",
            };

            result.push_str(&format!(
                "{}{} [{}] depth={} pri={:.2} instr=\"{}\"\n",
                prefix,
                id,
                status,
                call.depth,
                call.priority,
                if call.instruction.len() > 40 {
                    format!("{}...", &call.instruction[..40])
                } else {
                    call.instruction.clone()
                }
            ));

            for child_id in &call.children {
                self.format_subtree(child_id, indent + 1, result);
            }
        }
    }

    /// Check if the entire tree is resolved (all calls terminal)
    pub fn is_fully_resolved(&self) -> bool {
        self.calls.values().all(|c| c.status.is_terminal())
    }

    /// Get the root call result
    pub fn root_result(&self) -> Option<&str> {
        self.calls.get(&CallId::root()).and_then(|c| {
            if let CallStatus::Completed { result } = &c.status {
                Some(result.as_str())
            } else {
                None
            }
        })
    }
}

impl fmt::Display for RecursiveScheduler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RecursiveScheduler(total={}, pending={}, running={}, max_depth={})",
            self.calls.len(),
            self.pending_count(),
            self.running_count(),
            self.stats.max_depth_reached
        )
    }
}

/// Errors from the recursive scheduler
#[derive(Debug, Clone)]
pub enum SchedulerError {
    /// Call not found in the tree
    CallNotFound(CallId),
    /// Maximum recursion depth exceeded
    MaxDepthExceeded { current: usize, max: usize },
    /// Maximum pending calls exceeded
    MaxPendingExceeded { current: usize, max: usize },
    /// Call is in an invalid state for the requested operation
    InvalidState { call: CallId, expected: String, actual: String },
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedulerError::CallNotFound(id) => write!(f, "Call {} not found", id),
            SchedulerError::MaxDepthExceeded { current, max } => {
                write!(f, "Recursion depth {} exceeds maximum {}", current, max)
            }
            SchedulerError::MaxPendingExceeded { current, max } => {
                write!(
                    f,
                    "Pending calls {} exceeds maximum {}",
                    current, max
                )
            }
            SchedulerError::InvalidState {
                call,
                expected,
                actual,
            } => write!(
                f,
                "Call {} in invalid state: expected {}, got {}",
                call, expected, actual
            ),
        }
    }
}

impl std::error::Error for SchedulerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let config = RecursiveSchedulerConfig::default();
        let scheduler = RecursiveScheduler::new(config);

        assert_eq!(scheduler.total_calls(), 0);
        assert_eq!(scheduler.pending_count(), 0);
        assert_eq!(scheduler.running_count(), 0);
    }

    #[test]
    fn test_root_call() {
        let config = RecursiveSchedulerConfig::default();
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root(
            "Hello world".to_string(),
            "Process this text".to_string(),
        );

        assert_eq!(root_id, CallId::root());
        assert_eq!(scheduler.total_calls(), 1);
        assert_eq!(scheduler.pending_count(), 1);

        // Schedule should return the root as ready
        let output = scheduler.schedule();
        assert_eq!(output.ready_calls.len(), 1);
        assert_eq!(output.ready_calls[0], root_id);
        assert!(!output.root_completed);
    }

    #[test]
    fn test_subcall_spawning() {
        let config = RecursiveSchedulerConfig::default();
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root(
            "Big prompt".to_string(),
            "Analyze".to_string(),
        );

        // Schedule root to run
        let _ = scheduler.schedule();

        // Spawn sub-calls from root
        let sub1 = scheduler
            .spawn_subcall(
                root_id,
                "chunk 1".to_string(),
                "summarize".to_string(),
                "result_1".to_string(),
            )
            .unwrap();

        let sub2 = scheduler
            .spawn_subcall(
                root_id,
                "chunk 2".to_string(),
                "summarize".to_string(),
                "result_2".to_string(),
            )
            .unwrap();

        assert_eq!(scheduler.total_calls(), 3); // root + 2 children
        assert!(scheduler.pending_count() >= 2);

        // Children should be at depth 1
        assert_eq!(scheduler.get_call(&sub1).unwrap().depth, 1);
        assert_eq!(scheduler.get_call(&sub2).unwrap().depth, 1);
    }

    #[test]
    fn test_depth_limit() {
        let config = RecursiveSchedulerConfig {
            max_depth: 1,
            ..Default::default()
        };
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root("prompt".to_string(), "task".to_string());

        // Depth 1 should succeed
        let sub_id = scheduler
            .spawn_subcall(
                root_id,
                "sub prompt".to_string(),
                "sub task".to_string(),
                "result".to_string(),
            )
            .unwrap();

        // Depth 2 should fail
        let result = scheduler.spawn_subcall(
            sub_id,
            "sub-sub prompt".to_string(),
            "sub-sub task".to_string(),
            "result".to_string(),
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            SchedulerError::MaxDepthExceeded { .. } => {}
            _ => panic!("Expected MaxDepthExceeded"),
        }
    }

    #[test]
    fn test_call_completion() {
        let config = RecursiveSchedulerConfig::default();
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root("prompt".to_string(), "task".to_string());

        // Schedule and run root
        let _ = scheduler.schedule();

        // Complete root
        scheduler.complete_call(root_id, "The answer is 42".to_string());

        let call = scheduler.get_call(&root_id).unwrap();
        assert!(call.status.is_success());

        // Schedule should show root completed
        let output = scheduler.schedule();
        assert!(output.root_completed);
        assert_eq!(output.root_result.unwrap(), "The answer is 42");
    }

    #[test]
    fn test_failure_cascading() {
        let config = RecursiveSchedulerConfig::default();
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root("prompt".to_string(), "task".to_string());
        let _ = scheduler.schedule();

        let sub1 = scheduler
            .spawn_subcall(
                root_id,
                "chunk 1".to_string(),
                "task".to_string(),
                "r1".to_string(),
            )
            .unwrap();

        let sub2 = scheduler
            .spawn_subcall(
                sub1,
                "chunk 1a".to_string(),
                "task".to_string(),
                "r1a".to_string(),
            )
            .unwrap();

        // Fail sub1 -> should cascade to sub2
        scheduler.fail_call(sub1, "some error".to_string());

        assert!(matches!(
            scheduler.get_call(&sub1).unwrap().status,
            CallStatus::Failed { .. }
        ));
        assert!(matches!(
            scheduler.get_call(&sub2).unwrap().status,
            CallStatus::Cancelled
        ));
    }

    #[test]
    fn test_children_collection() {
        let config = RecursiveSchedulerConfig::default();
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root("prompt".to_string(), "task".to_string());
        let _ = scheduler.schedule();

        let sub1 = scheduler
            .spawn_subcall(root_id, "c1".to_string(), "t".to_string(), "r1".to_string())
            .unwrap();
        let sub2 = scheduler
            .spawn_subcall(root_id, "c2".to_string(), "t".to_string(), "r2".to_string())
            .unwrap();

        let _ = scheduler.schedule();

        scheduler.complete_call(sub1, "result 1".to_string());
        scheduler.complete_call(sub2, "result 2".to_string());

        assert!(scheduler.all_children_completed(&root_id));

        let results = scheduler.collect_children_results(&root_id);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_batch_spawn() {
        let config = RecursiveSchedulerConfig::default();
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root("prompt".to_string(), "task".to_string());
        let _ = scheduler.schedule();

        let prompts = vec![
            "chunk 1".to_string(),
            "chunk 2".to_string(),
            "chunk 3".to_string(),
        ];

        let call_ids = scheduler
            .spawn_batch(
                root_id,
                prompts,
                "summarize".to_string(),
                "batch_result".to_string(),
            )
            .unwrap();

        assert_eq!(call_ids.len(), 3);
        assert_eq!(scheduler.total_calls(), 4); // root + 3 children
    }

    #[test]
    fn test_tree_visualization() {
        let config = RecursiveSchedulerConfig::default();
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root("prompt".to_string(), "analyze".to_string());
        let _ = scheduler.schedule();

        scheduler
            .spawn_subcall(
                root_id,
                "c1".to_string(),
                "summarize chunk 1".to_string(),
                "r1".to_string(),
            )
            .unwrap();
        scheduler
            .spawn_subcall(
                root_id,
                "c2".to_string(),
                "summarize chunk 2".to_string(),
                "r2".to_string(),
            )
            .unwrap();

        let tree = scheduler.tree_to_string();
        assert!(tree.contains("Call(0)"));
        assert!(tree.contains("summarize chunk 1"));
        assert!(tree.contains("summarize chunk 2"));
    }

    #[test]
    fn test_depth_distribution() {
        let config = RecursiveSchedulerConfig::default();
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root("p".to_string(), "t".to_string());
        let _ = scheduler.schedule();

        let sub1 = scheduler
            .spawn_subcall(root_id, "c1".to_string(), "t".to_string(), "r1".to_string())
            .unwrap();
        scheduler
            .spawn_subcall(root_id, "c2".to_string(), "t".to_string(), "r2".to_string())
            .unwrap();
        scheduler
            .spawn_subcall(sub1, "c1a".to_string(), "t".to_string(), "r1a".to_string())
            .unwrap();

        let dist = scheduler.depth_distribution();
        assert_eq!(dist[&0], 1); // root
        assert_eq!(dist[&1], 2); // sub1, sub2
        assert_eq!(dist[&2], 1); // sub1a
    }

    #[test]
    fn test_priority_decay() {
        let config = RecursiveSchedulerConfig {
            priority_decay: 0.5,
            ..Default::default()
        };
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root("p".to_string(), "t".to_string());
        let _ = scheduler.schedule();

        let sub_id = scheduler
            .spawn_subcall(root_id, "c".to_string(), "t".to_string(), "r".to_string())
            .unwrap();

        let root_priority = scheduler.get_call(&root_id).unwrap().priority;
        let sub_priority = scheduler.get_call(&sub_id).unwrap().priority;

        assert!(root_priority > sub_priority);
        assert!((sub_priority - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_stats() {
        let config = RecursiveSchedulerConfig::default();
        let mut scheduler = RecursiveScheduler::new(config);

        let root_id = scheduler.init_root("p".to_string(), "t".to_string());
        let _ = scheduler.schedule();

        scheduler
            .spawn_subcall(root_id, "c".to_string(), "t".to_string(), "r".to_string())
            .unwrap();

        let stats = scheduler.stats();
        assert_eq!(stats.total_calls_created, 2);
        assert_eq!(stats.max_depth_reached, 1);
    }
}
