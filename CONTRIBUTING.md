# Contributing to Rusteeze

Thank you for your interest in contributing to Rusteeze! This document provides guidelines and information about contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building something amazing together.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- CUDA 12.0+ (for GPU support)
- cuDNN 8.9+ (for GPU support)
- CMake 3.20+ (for building native extensions)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/dihannahdi/rusteeze.git
cd rusteeze

# Install development tools
cargo install cargo-watch cargo-nextest cargo-audit cargo-deny

# Build in debug mode
cargo build

# Run tests
cargo nextest run

# Run specific crate tests
cargo nextest run -p rusteeze-core
```

## Project Structure

```
rusteeze/
├── crates/
│   ├── rusteeze-core/       # Core types and utilities
│   ├── rusteeze-tensor/     # Tensor operations, attention, quantization
│   ├── rusteeze-model/      # Model architectures (Llama, Mistral, etc.)
│   ├── rusteeze-tokenizer/  # Tokenization and chat templates
│   ├── rusteeze-engine/     # Inference engine (scheduler, batching)
│   ├── rusteeze-api/        # REST API server (OpenAI-compatible)
│   ├── rusteeze-config/     # Configuration loading and validation
│   ├── rusteeze-metrics/    # Prometheus metrics and tracing
│   └── rusteeze-cli/        # Command-line interface
├── docs/                    # Documentation
├── examples/                # Example code
├── benches/                 # Benchmarks
└── tests/                   # Integration tests
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write clean, idiomatic Rust code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
cargo nextest run

# Run tests with coverage
cargo llvm-cov nextest

# Run lints
cargo clippy --all-targets --all-features -- -D warnings

# Check formatting
cargo fmt --check
```

### 4. Commit Your Changes

We use conventional commits:

```
feat: Add support for Qwen2 models
fix: Resolve memory leak in KV cache
docs: Update API documentation
perf: Optimize attention computation
refactor: Simplify scheduler logic
test: Add unit tests for tokenizer
```

### 5. Submit a Pull Request

- Fill out the PR template
- Link related issues
- Request review from maintainers

## Coding Standards

### Rust Guidelines

- Use `rustfmt` for formatting
- Use `clippy` for linting
- Prefer safe Rust over unsafe when possible
- Document all public APIs
- Write comprehensive error messages

### Error Handling

```rust
// Good: Specific error types
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Failed to load model weights: {0}")]
    WeightLoadError(String),
    
    #[error("Invalid tensor shape: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

// Good: Use Result with specific errors
pub fn load_model(path: &str) -> Result<Model, ModelError> {
    // ...
}
```

### Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature() {
        // Arrange
        let input = create_test_input();
        
        // Act
        let result = process(input);
        
        // Assert
        assert_eq!(result, expected);
    }
    
    #[tokio::test]
    async fn test_async_feature() {
        // Async test
    }
}
```

### Documentation

```rust
/// Processes input tokens through the model.
///
/// # Arguments
///
/// * `tokens` - Input token IDs
/// * `config` - Generation configuration
///
/// # Returns
///
/// Generated token IDs with sampling metadata.
///
/// # Errors
///
/// Returns `ModelError` if:
/// - Input exceeds maximum context length
/// - Model weights are not loaded
///
/// # Examples
///
/// ```
/// let tokens = vec![1, 2, 3];
/// let result = model.generate(&tokens, config)?;
/// ```
pub fn generate(&self, tokens: &[u32], config: &Config) -> Result<Output, Error> {
    // ...
}
```

## Performance Guidelines

### Memory Efficiency

- Minimize allocations in hot paths
- Use arena allocators for temporary data
- Profile memory usage with `heaptrack` or `valgrind`

### GPU Optimization

- Batch operations to maximize GPU utilization
- Minimize CPU-GPU transfers
- Use asynchronous CUDA streams

### Profiling

```bash
# CPU profiling with flamegraph
cargo flamegraph --bin rusteeze -- serve --model test

# GPU profiling with nsight
nsys profile target/release/rusteeze serve --model test
```

## Adding New Model Architectures

1. Create a new file in `rusteeze-model/src/architectures/`
2. Implement the `ModelArchitecture` trait
3. Add the model to the architecture registry
4. Add tests and documentation
5. Update the README model support list

```rust
// Example: Adding a new model
pub struct NewModel {
    layers: Vec<TransformerLayer>,
    config: ModelConfig,
}

impl ModelArchitecture for NewModel {
    fn forward(&self, input: &Tensor, cache: &mut KVCache) -> Result<Tensor> {
        // Implementation
    }
}
```

## Release Process

1. Update version in `Cargo.toml`
2. Update CHANGELOG.md
3. Create a release tag
4. GitHub Actions will build and publish

## Getting Help

- Open an issue for bugs or feature requests
- Join our Discord for discussions
- Check existing issues and PRs

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
