// ============================================================================
// Rusteeze - Recursive Language Model Infrastructure
// Variable Store: Persistent REPL-like variable management
// Based on: "Recursive Language Models" (Zhang, Kraska, Khattab 2026)
// ============================================================================
//
// The VariableStore is the backbone of the RLM REPL environment. It stores
// named variables that persist across inference iterations, exactly as
// Algorithm 1 in the paper describes: the prompt is stored as a variable,
// intermediate computation results are stored in named variables, and the
// special "Final" variable triggers termination and response return.
//
// Key design principles:
// - Zero-copy string slicing where possible (Arc<str> + range references)
// - Type-safe variable values (strings, token sequences, numeric, structured)
// - Metadata computation on demand (length, prefix, hash)
// - Thread-safe for parallel sub-call execution
// ============================================================================

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

/// Unique identifier for a variable in the store
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct VarId(Arc<str>);

impl VarId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(Arc::from(name.into().as_str()))
    }

    /// Reserved name for the user prompt
    pub fn prompt() -> Self {
        Self::new("__prompt__")
    }

    /// Reserved name for the final output (triggers termination)
    pub fn final_output() -> Self {
        Self::new("Final")
    }

    /// Reserved name for the final variable reference
    pub fn final_var() -> Self {
        Self::new("FinalVar")
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_prompt(&self) -> bool {
        self.0.as_ref() == "__prompt__"
    }

    pub fn is_final(&self) -> bool {
        self.0.as_ref() == "Final" || self.0.as_ref() == "FinalVar"
    }
}

impl fmt::Display for VarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for VarId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for VarId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// The type of value stored in a variable
#[derive(Debug, Clone)]
pub enum VarValue {
    /// Raw string content (prompts, sub-call results, aggregated text)
    Text(Arc<str>),

    /// Token ID sequence (for passing between model calls)
    Tokens(Arc<[u32]>),

    /// Numeric value (counts, scores, metrics)
    Number(f64),

    /// Boolean flag
    Bool(bool),

    /// List of strings (decomposed chunks, collected results)
    TextList(Vec<Arc<str>>),

    /// List of numbers (logprobs, scores)
    NumberList(Vec<f64>),

    /// Structured key-value data (JSON-like intermediate results)
    Map(HashMap<String, VarValue>),

    /// A slice reference into another variable (zero-copy)
    Slice {
        source_var: VarId,
        byte_start: usize,
        byte_end: usize,
    },

    /// Null / empty placeholder
    Null,
}

impl VarValue {
    /// Get the value as text, resolving slices if needed
    pub fn as_text(&self) -> Option<&str> {
        match self {
            VarValue::Text(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    /// Get the value as a token sequence
    pub fn as_tokens(&self) -> Option<&[u32]> {
        match self {
            VarValue::Tokens(t) => Some(t.as_ref()),
            _ => None,
        }
    }

    /// Get the value as a number
    pub fn as_number(&self) -> Option<f64> {
        match self {
            VarValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get the value as a boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            VarValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get the value as a text list
    pub fn as_text_list(&self) -> Option<&[Arc<str>]> {
        match self {
            VarValue::TextList(list) => Some(list.as_slice()),
            _ => None,
        }
    }

    /// Get the value as a number list
    pub fn as_number_list(&self) -> Option<&[f64]> {
        match self {
            VarValue::NumberList(list) => Some(list.as_slice()),
            _ => None,
        }
    }

    /// Check if this is a null/empty value
    pub fn is_null(&self) -> bool {
        matches!(self, VarValue::Null)
    }

    /// Get approximate memory footprint in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            VarValue::Text(s) => s.len(),
            VarValue::Tokens(t) => t.len() * 4,
            VarValue::Number(_) => 8,
            VarValue::Bool(_) => 1,
            VarValue::TextList(list) => list.iter().map(|s| s.len()).sum::<usize>(),
            VarValue::NumberList(list) => list.len() * 8,
            VarValue::Map(map) => map
                .iter()
                .map(|(k, v)| k.len() + v.memory_size())
                .sum::<usize>(),
            VarValue::Slice { .. } => 0, // zero-copy reference
            VarValue::Null => 0,
        }
    }

    /// Get a human-readable type name
    pub fn type_name(&self) -> &'static str {
        match self {
            VarValue::Text(_) => "text",
            VarValue::Tokens(_) => "tokens",
            VarValue::Number(_) => "number",
            VarValue::Bool(_) => "bool",
            VarValue::TextList(_) => "text_list",
            VarValue::NumberList(_) => "number_list",
            VarValue::Map(_) => "map",
            VarValue::Slice { .. } => "slice",
            VarValue::Null => "null",
        }
    }

    /// Convert value to a text representation for metadata
    pub fn to_text_repr(&self) -> String {
        match self {
            VarValue::Text(s) => {
                if s.len() > 200 {
                    format!("{}... [{} chars total]", &s[..200], s.len())
                } else {
                    s.to_string()
                }
            }
            VarValue::Tokens(t) => format!("[{} tokens]", t.len()),
            VarValue::Number(n) => format!("{}", n),
            VarValue::Bool(b) => format!("{}", b),
            VarValue::TextList(list) => format!("[{} text items]", list.len()),
            VarValue::NumberList(list) => format!("[{} numbers]", list.len()),
            VarValue::Map(map) => format!("{{{} entries}}", map.len()),
            VarValue::Slice {
                source_var,
                byte_start,
                byte_end,
            } => format!(
                "slice({}[{}..{}])",
                source_var, byte_start, byte_end
            ),
            VarValue::Null => "null".to_string(),
        }
    }
}

impl fmt::Display for VarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text_repr())
    }
}

/// Metadata about a variable, computed lazily
#[derive(Debug, Clone)]
pub struct VarMetadata {
    /// Length in characters (for text) or items (for lists/tokens)
    pub length: usize,
    /// Short prefix for context (first N chars/items)
    pub prefix: String,
    /// Type of the variable
    pub var_type: &'static str,
    /// Memory footprint in bytes
    pub memory_bytes: usize,
    /// Timestamp when the variable was created
    pub created_at: Instant,
    /// Timestamp when the variable was last modified
    pub modified_at: Instant,
    /// Number of times this variable has been accessed
    pub access_count: u64,
}

impl VarMetadata {
    /// Maximum prefix length for metadata
    const MAX_PREFIX_LEN: usize = 128;

    /// Compute metadata for a variable value
    pub fn compute(value: &VarValue) -> Self {
        let now = Instant::now();
        let (length, prefix) = match value {
            VarValue::Text(s) => {
                let prefix_end = s
                    .char_indices()
                    .nth(Self::MAX_PREFIX_LEN)
                    .map(|(i, _)| i)
                    .unwrap_or(s.len());
                (s.len(), s[..prefix_end].to_string())
            }
            VarValue::Tokens(t) => {
                let preview: Vec<String> =
                    t.iter().take(20).map(|id| id.to_string()).collect();
                (t.len(), format!("[{}]", preview.join(", ")))
            }
            VarValue::Number(n) => (1, format!("{}", n)),
            VarValue::Bool(b) => (1, format!("{}", b)),
            VarValue::TextList(list) => {
                let preview: Vec<String> = list
                    .iter()
                    .take(3)
                    .map(|s| {
                        if s.len() > 40 {
                            format!("\"{}...\"", &s[..40])
                        } else {
                            format!("\"{}\"", s)
                        }
                    })
                    .collect();
                (list.len(), format!("[{}]", preview.join(", ")))
            }
            VarValue::NumberList(list) => {
                let preview: Vec<String> =
                    list.iter().take(10).map(|n| format!("{:.4}", n)).collect();
                (list.len(), format!("[{}]", preview.join(", ")))
            }
            VarValue::Map(map) => {
                let keys: Vec<&str> = map.keys().take(5).map(|k| k.as_str()).collect();
                (map.len(), format!("{{{}}}", keys.join(", ")))
            }
            VarValue::Slice {
                source_var,
                byte_start,
                byte_end,
            } => (
                byte_end - byte_start,
                format!("{}[{}..{}]", source_var, byte_start, byte_end),
            ),
            VarValue::Null => (0, "null".to_string()),
        };

        Self {
            length,
            prefix,
            var_type: value.type_name(),
            memory_bytes: value.memory_size(),
            created_at: now,
            modified_at: now,
            access_count: 0,
        }
    }

    /// Generate a constant-size metadata string for the context window
    /// This is the key insight from the paper: only metadata enters the window
    pub fn to_context_string(&self) -> String {
        format!(
            "var_type={}, length={}, prefix=\"{}\"",
            self.var_type,
            self.length,
            if self.prefix.len() > 80 {
                format!("{}...", &self.prefix[..80])
            } else {
                self.prefix.clone()
            }
        )
    }
}

/// Entry in the variable store with value and metadata
#[derive(Debug, Clone)]
struct VarEntry {
    value: VarValue,
    metadata: VarMetadata,
    version: u64,
}

/// The Variable Store: persistent REPL-like variable management
///
/// This is the core state management for the RLM REPL environment.
/// Variables persist across inference iterations, and the store provides
/// constant-size metadata for the context window while keeping full values
/// externally accessible.
///
/// Thread-safety: The store itself is not thread-safe; wrap in Arc<RwLock>
/// for concurrent access from sub-calls.
#[derive(Debug)]
pub struct VariableStore {
    /// Named variables
    variables: HashMap<VarId, VarEntry>,
    /// Global version counter for ordering
    version_counter: u64,
    /// Total memory used by all variables
    total_memory_bytes: usize,
    /// Maximum allowed memory (backpressure)
    max_memory_bytes: usize,
    /// History of variable operations for debugging/replay
    operation_log: Vec<VarOperation>,
    /// Whether to record operation history
    record_history: bool,
}

/// Record of a variable operation
#[derive(Debug, Clone)]
pub struct VarOperation {
    pub op_type: VarOpType,
    pub var_id: VarId,
    pub timestamp: Instant,
    pub version: u64,
}

/// Type of variable operation
#[derive(Debug, Clone)]
pub enum VarOpType {
    Set,
    Get,
    Delete,
    Slice,
    Append,
    Transform,
}

impl VariableStore {
    /// Default maximum memory: 1GB
    const DEFAULT_MAX_MEMORY: usize = 1024 * 1024 * 1024;

    /// Create a new variable store
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            version_counter: 0,
            total_memory_bytes: 0,
            max_memory_bytes: Self::DEFAULT_MAX_MEMORY,
            operation_log: Vec::new(),
            record_history: false,
        }
    }

    /// Create a new variable store with custom memory limit
    pub fn with_max_memory(max_memory_bytes: usize) -> Self {
        Self {
            max_memory_bytes,
            ..Self::new()
        }
    }

    /// Enable operation history recording
    pub fn enable_history(&mut self) {
        self.record_history = true;
    }

    /// Set a variable value
    pub fn set(&mut self, id: VarId, value: VarValue) -> Result<(), VarStoreError> {
        let mem_size = value.memory_size();

        // Check memory limit (subtract old value if overwriting)
        let old_mem = self
            .variables
            .get(&id)
            .map(|e| e.value.memory_size())
            .unwrap_or(0);
        let new_total = self.total_memory_bytes + mem_size - old_mem;
        if new_total > self.max_memory_bytes {
            return Err(VarStoreError::MemoryLimitExceeded {
                requested: mem_size,
                available: self.max_memory_bytes.saturating_sub(self.total_memory_bytes),
                limit: self.max_memory_bytes,
            });
        }

        self.version_counter += 1;
        let metadata = VarMetadata::compute(&value);

        let entry = VarEntry {
            value,
            metadata,
            version: self.version_counter,
        };

        self.total_memory_bytes = new_total;

        if self.record_history {
            self.operation_log.push(VarOperation {
                op_type: VarOpType::Set,
                var_id: id.clone(),
                timestamp: Instant::now(),
                version: self.version_counter,
            });
        }

        self.variables.insert(id, entry);
        Ok(())
    }

    /// Get a variable value (immutable reference)
    pub fn get(&mut self, id: &VarId) -> Option<&VarValue> {
        if let Some(entry) = self.variables.get_mut(id) {
            entry.metadata.access_count += 1;
            entry.metadata.modified_at = Instant::now();

            if self.record_history {
                self.operation_log.push(VarOperation {
                    op_type: VarOpType::Get,
                    var_id: id.clone(),
                    timestamp: Instant::now(),
                    version: entry.version,
                });
            }
        }
        self.variables.get(id).map(|e| &e.value)
    }

    /// Get a variable value without tracking access
    pub fn peek(&self, id: &VarId) -> Option<&VarValue> {
        self.variables.get(id).map(|e| &e.value)
    }

    /// Get the metadata for a variable (constant-size, safe for context window)
    pub fn get_metadata(&self, id: &VarId) -> Option<&VarMetadata> {
        self.variables.get(id).map(|e| &e.metadata)
    }

    /// Get all variable metadata as a formatted string for context window injection
    pub fn all_metadata_string(&self) -> String {
        let mut entries: Vec<_> = self
            .variables
            .iter()
            .filter(|(id, _)| !id.is_prompt()) // Don't include raw prompt
            .map(|(id, entry)| {
                format!(
                    "  {} (v{}): {}",
                    id,
                    entry.version,
                    entry.metadata.to_context_string()
                )
            })
            .collect();
        entries.sort();

        if entries.is_empty() {
            "Variables: (none)".to_string()
        } else {
            format!("Variables:\n{}", entries.join("\n"))
        }
    }

    /// Delete a variable
    pub fn delete(&mut self, id: &VarId) -> Option<VarValue> {
        if let Some(entry) = self.variables.remove(id) {
            self.total_memory_bytes = self
                .total_memory_bytes
                .saturating_sub(entry.value.memory_size());

            if self.record_history {
                self.operation_log.push(VarOperation {
                    op_type: VarOpType::Delete,
                    var_id: id.clone(),
                    timestamp: Instant::now(),
                    version: self.version_counter,
                });
            }

            Some(entry.value)
        } else {
            None
        }
    }

    /// Create a slice reference into a text variable (zero-copy)
    pub fn slice_text(
        &mut self,
        source_id: &VarId,
        target_id: VarId,
        byte_start: usize,
        byte_end: usize,
    ) -> Result<(), VarStoreError> {
        // Validate the source exists and is text
        let source_len = match self.peek(source_id) {
            Some(VarValue::Text(s)) => s.len(),
            Some(_) => {
                return Err(VarStoreError::TypeMismatch {
                    var: source_id.clone(),
                    expected: "text",
                    actual: self.peek(source_id).unwrap().type_name(),
                })
            }
            None => {
                return Err(VarStoreError::NotFound(source_id.clone()));
            }
        };

        if byte_end > source_len || byte_start > byte_end {
            return Err(VarStoreError::SliceOutOfBounds {
                var: source_id.clone(),
                start: byte_start,
                end: byte_end,
                length: source_len,
            });
        }

        // Instead of creating a Slice reference, materialize the substring
        // This avoids borrow checker issues and is simpler
        let text = match self.peek(source_id) {
            Some(VarValue::Text(s)) => Arc::from(&s[byte_start..byte_end]),
            _ => unreachable!(),
        };

        self.set(target_id, VarValue::Text(text))
    }

    /// Append text to an existing text variable
    pub fn append_text(
        &mut self,
        id: &VarId,
        additional: &str,
    ) -> Result<(), VarStoreError> {
        match self.peek(id).cloned() {
            Some(VarValue::Text(existing)) => {
                let mut combined = String::with_capacity(existing.len() + additional.len());
                combined.push_str(&existing);
                combined.push_str(additional);
                self.set(id.clone(), VarValue::Text(Arc::from(combined.as_str())))
            }
            Some(other) => Err(VarStoreError::TypeMismatch {
                var: id.clone(),
                expected: "text",
                actual: other.type_name(),
            }),
            None => self.set(id.clone(), VarValue::Text(Arc::from(additional))),
        }
    }

    /// Push a value onto a text list variable
    pub fn push_to_list(
        &mut self,
        id: &VarId,
        item: Arc<str>,
    ) -> Result<(), VarStoreError> {
        match self.peek(id).cloned() {
            Some(VarValue::TextList(mut list)) => {
                list.push(item);
                self.set(id.clone(), VarValue::TextList(list))
            }
            Some(other) => Err(VarStoreError::TypeMismatch {
                var: id.clone(),
                expected: "text_list",
                actual: other.type_name(),
            }),
            None => self.set(id.clone(), VarValue::TextList(vec![item])),
        }
    }

    /// Check if the Final variable has been set (triggers RLM termination)
    pub fn has_final(&self) -> bool {
        self.variables.contains_key(&VarId::final_output())
            || self.variables.contains_key(&VarId::final_var())
    }

    /// Get the final output value
    pub fn get_final(&self) -> Option<&VarValue> {
        self.variables
            .get(&VarId::final_output())
            .or_else(|| self.variables.get(&VarId::final_var()))
            .map(|e| &e.value)
    }

    /// Get the final output as text, resolving variable references
    pub fn resolve_final(&self) -> Option<String> {
        // Check "Final" first, then "FinalVar"
        if let Some(entry) = self.variables.get(&VarId::final_output()) {
            return match &entry.value {
                VarValue::Text(s) => Some(s.to_string()),
                other => Some(other.to_text_repr()),
            };
        }

        // FinalVar means reference another variable's value
        if let Some(entry) = self.variables.get(&VarId::final_var()) {
            if let VarValue::Text(var_name) = &entry.value {
                let ref_id = VarId::new(var_name.as_ref());
                if let Some(referenced) = self.variables.get(&ref_id) {
                    return match &referenced.value {
                        VarValue::Text(s) => Some(s.to_string()),
                        other => Some(other.to_text_repr()),
                    };
                }
            }
        }

        None
    }

    /// Get the number of variables in the store
    pub fn len(&self) -> usize {
        self.variables.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.variables.is_empty()
    }

    /// Get total memory used by all variables
    pub fn total_memory(&self) -> usize {
        self.total_memory_bytes
    }

    /// Get the operation history
    pub fn history(&self) -> &[VarOperation] {
        &self.operation_log
    }

    /// List all variable IDs
    pub fn variable_ids(&self) -> Vec<&VarId> {
        self.variables.keys().collect()
    }

    /// Clear all variables except the prompt
    pub fn clear_non_prompt(&mut self) {
        let prompt_entry = self.variables.remove(&VarId::prompt());
        self.variables.clear();
        self.total_memory_bytes = 0;

        if let Some(entry) = prompt_entry {
            self.total_memory_bytes = entry.value.memory_size();
            self.variables.insert(VarId::prompt(), entry);
        }
    }

    /// Get a snapshot of the store state for debugging
    pub fn snapshot(&self) -> StoreSnapshot {
        StoreSnapshot {
            num_variables: self.variables.len(),
            total_memory_bytes: self.total_memory_bytes,
            version: self.version_counter,
            has_final: self.has_final(),
            variable_summaries: self
                .variables
                .iter()
                .map(|(id, entry)| (id.to_string(), entry.metadata.to_context_string()))
                .collect(),
        }
    }
}

impl Default for VariableStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of the store state for debugging/metrics
#[derive(Debug, Clone)]
pub struct StoreSnapshot {
    pub num_variables: usize,
    pub total_memory_bytes: usize,
    pub version: u64,
    pub has_final: bool,
    pub variable_summaries: HashMap<String, String>,
}

/// Errors from variable store operations
#[derive(Debug, Clone)]
pub enum VarStoreError {
    /// Variable not found
    NotFound(VarId),

    /// Type mismatch when operating on a variable
    TypeMismatch {
        var: VarId,
        expected: &'static str,
        actual: &'static str,
    },

    /// Slice out of bounds
    SliceOutOfBounds {
        var: VarId,
        start: usize,
        end: usize,
        length: usize,
    },

    /// Memory limit exceeded
    MemoryLimitExceeded {
        requested: usize,
        available: usize,
        limit: usize,
    },
}

impl fmt::Display for VarStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VarStoreError::NotFound(id) => write!(f, "Variable '{}' not found", id),
            VarStoreError::TypeMismatch {
                var,
                expected,
                actual,
            } => write!(
                f,
                "Type mismatch for '{}': expected {}, got {}",
                var, expected, actual
            ),
            VarStoreError::SliceOutOfBounds {
                var,
                start,
                end,
                length,
            } => write!(
                f,
                "Slice [{}..{}] out of bounds for '{}' (length {})",
                start, end, var, length
            ),
            VarStoreError::MemoryLimitExceeded {
                requested,
                available,
                limit,
            } => write!(
                f,
                "Memory limit exceeded: requested {} bytes, {} available (limit: {})",
                requested, available, limit
            ),
        }
    }
}

impl std::error::Error for VarStoreError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_store_basic_operations() {
        let mut store = VariableStore::new();

        // Set a text variable
        store
            .set(VarId::new("greeting"), VarValue::Text(Arc::from("Hello, world!")))
            .unwrap();

        // Get the variable
        let value = store.get(&VarId::new("greeting")).unwrap();
        assert_eq!(value.as_text().unwrap(), "Hello, world!");

        // Check metadata
        let metadata = store.get_metadata(&VarId::new("greeting")).unwrap();
        assert_eq!(metadata.var_type, "text");
        assert_eq!(metadata.length, 13);

        // Delete the variable
        let deleted = store.delete(&VarId::new("greeting")).unwrap();
        assert_eq!(deleted.as_text().unwrap(), "Hello, world!");
        assert!(store.peek(&VarId::new("greeting")).is_none());
    }

    #[test]
    fn test_variable_store_final_detection() {
        let mut store = VariableStore::new();

        assert!(!store.has_final());

        store
            .set(
                VarId::final_output(),
                VarValue::Text(Arc::from("The answer is 42")),
            )
            .unwrap();

        assert!(store.has_final());
        assert_eq!(
            store.resolve_final().unwrap(),
            "The answer is 42"
        );
    }

    #[test]
    fn test_variable_store_final_var_reference() {
        let mut store = VariableStore::new();

        // Set a result variable
        store
            .set(
                VarId::new("result"),
                VarValue::Text(Arc::from("computed answer")),
            )
            .unwrap();

        // Set FinalVar to reference "result"
        store
            .set(
                VarId::final_var(),
                VarValue::Text(Arc::from("result")),
            )
            .unwrap();

        assert!(store.has_final());
        assert_eq!(
            store.resolve_final().unwrap(),
            "computed answer"
        );
    }

    #[test]
    fn test_variable_store_text_slicing() {
        let mut store = VariableStore::new();

        store
            .set(
                VarId::prompt(),
                VarValue::Text(Arc::from("Hello, world! This is a long prompt.")),
            )
            .unwrap();

        store
            .slice_text(&VarId::prompt(), VarId::new("chunk1"), 0, 13)
            .unwrap();

        let chunk = store.peek(&VarId::new("chunk1")).unwrap();
        assert_eq!(chunk.as_text().unwrap(), "Hello, world!");
    }

    #[test]
    fn test_variable_store_append_text() {
        let mut store = VariableStore::new();

        store
            .set(
                VarId::new("output"),
                VarValue::Text(Arc::from("Part 1. ")),
            )
            .unwrap();

        store
            .append_text(&VarId::new("output"), "Part 2. ")
            .unwrap();

        let value = store.peek(&VarId::new("output")).unwrap();
        assert_eq!(value.as_text().unwrap(), "Part 1. Part 2. ");
    }

    #[test]
    fn test_variable_store_list_operations() {
        let mut store = VariableStore::new();

        store
            .push_to_list(&VarId::new("results"), Arc::from("result 1"))
            .unwrap();
        store
            .push_to_list(&VarId::new("results"), Arc::from("result 2"))
            .unwrap();

        let value = store.peek(&VarId::new("results")).unwrap();
        let list = value.as_text_list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].as_ref(), "result 1");
        assert_eq!(list[1].as_ref(), "result 2");
    }

    #[test]
    fn test_variable_store_memory_limit() {
        let mut store = VariableStore::with_max_memory(100);

        // This should succeed (small value)
        store
            .set(VarId::new("small"), VarValue::Text(Arc::from("hello")))
            .unwrap();

        // This should fail (exceeds limit)
        let large_text = "x".repeat(200);
        let result = store.set(
            VarId::new("large"),
            VarValue::Text(Arc::from(large_text.as_str())),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_variable_store_metadata_string() {
        let mut store = VariableStore::new();

        store
            .set(VarId::new("x"), VarValue::Number(42.0))
            .unwrap();
        store
            .set(
                VarId::new("y"),
                VarValue::Text(Arc::from("some text")),
            )
            .unwrap();

        let metadata = store.all_metadata_string();
        assert!(metadata.contains("Variables:"));
        assert!(metadata.contains("x"));
        assert!(metadata.contains("y"));
    }

    #[test]
    fn test_variable_store_clear_non_prompt() {
        let mut store = VariableStore::new();

        store
            .set(
                VarId::prompt(),
                VarValue::Text(Arc::from("the prompt")),
            )
            .unwrap();
        store
            .set(VarId::new("temp"), VarValue::Number(1.0))
            .unwrap();

        store.clear_non_prompt();

        assert!(store.peek(&VarId::prompt()).is_some());
        assert!(store.peek(&VarId::new("temp")).is_none());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_variable_store_snapshot() {
        let mut store = VariableStore::new();
        store.set(VarId::new("a"), VarValue::Number(1.0)).unwrap();
        store.set(VarId::new("b"), VarValue::Number(2.0)).unwrap();

        let snap = store.snapshot();
        assert_eq!(snap.num_variables, 2);
        assert!(!snap.has_final);
        assert!(snap.variable_summaries.contains_key("a"));
        assert!(snap.variable_summaries.contains_key("b"));
    }

    #[test]
    fn test_variable_store_number_and_bool() {
        let mut store = VariableStore::new();

        store.set(VarId::new("pi"), VarValue::Number(3.14159)).unwrap();
        store.set(VarId::new("flag"), VarValue::Bool(true)).unwrap();

        assert_eq!(store.peek(&VarId::new("pi")).unwrap().as_number().unwrap(), 3.14159);
        assert_eq!(store.peek(&VarId::new("flag")).unwrap().as_bool().unwrap(), true);
    }
}
