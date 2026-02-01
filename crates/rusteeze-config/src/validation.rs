//! Configuration validation.

use crate::error::ConfigError;
use crate::Config;

/// Validate a configuration.
pub fn validate_config(config: &Config) -> Result<(), ConfigError> {
    // Validate model
    validate_model_config(config)?;

    // Validate server
    validate_server_config(config)?;

    // Validate engine
    validate_engine_config(config)?;

    Ok(())
}

/// Validate model configuration.
fn validate_model_config(config: &Config) -> Result<(), ConfigError> {
    // Model path is required
    if config.model.path.is_empty() {
        return Err(ConfigError::missing_field("model.path"));
    }

    // Validate dtype
    let valid_dtypes = ["auto", "float32", "float16", "bfloat16", "int8", "int4"];
    if !valid_dtypes.contains(&config.model.dtype.as_str()) {
        return Err(ConfigError::invalid_value(
            "model.dtype",
            format!(
                "must be one of: {}",
                valid_dtypes.join(", ")
            ),
        ));
    }

    // Validate device
    let valid_devices = ["auto", "cpu", "cuda", "metal", "mps"];
    let device_lower = config.model.device.to_lowercase();
    let is_valid_device = valid_devices.contains(&device_lower.as_str())
        || device_lower.starts_with("cuda:");

    if !is_valid_device {
        return Err(ConfigError::invalid_value(
            "model.device",
            format!(
                "must be one of: {} or cuda:<device_id>",
                valid_devices.join(", ")
            ),
        ));
    }

    // Validate parallelism
    if config.model.tensor_parallel_size == 0 {
        return Err(ConfigError::invalid_value(
            "model.tensor_parallel_size",
            "must be at least 1",
        ));
    }

    if config.model.pipeline_parallel_size == 0 {
        return Err(ConfigError::invalid_value(
            "model.pipeline_parallel_size",
            "must be at least 1",
        ));
    }

    Ok(())
}

/// Validate server configuration.
fn validate_server_config(config: &Config) -> Result<(), ConfigError> {
    // Port range
    if config.server.port == 0 {
        return Err(ConfigError::invalid_value(
            "server.port",
            "must be between 1 and 65535",
        ));
    }

    // Timeout
    if config.server.timeout_seconds == 0 {
        return Err(ConfigError::invalid_value(
            "server.timeout_seconds",
            "must be greater than 0",
        ));
    }

    // Max connections
    if config.server.max_connections == 0 {
        return Err(ConfigError::invalid_value(
            "server.max_connections",
            "must be greater than 0",
        ));
    }

    // TLS config
    if let Some(ref tls) = config.server.tls {
        if tls.cert_path.is_empty() {
            return Err(ConfigError::missing_field("server.tls.cert_path"));
        }
        if tls.key_path.is_empty() {
            return Err(ConfigError::missing_field("server.tls.key_path"));
        }
    }

    Ok(())
}

/// Validate engine configuration.
fn validate_engine_config(config: &Config) -> Result<(), ConfigError> {
    // Max sequences
    if config.engine.max_num_seqs == 0 {
        return Err(ConfigError::invalid_value(
            "engine.max_num_seqs",
            "must be greater than 0",
        ));
    }

    // Max tokens
    if config.engine.max_num_batched_tokens == 0 {
        return Err(ConfigError::invalid_value(
            "engine.max_num_batched_tokens",
            "must be greater than 0",
        ));
    }

    // Block size must be power of 2
    if !config.engine.block_size.is_power_of_two() {
        return Err(ConfigError::invalid_value(
            "engine.block_size",
            "must be a power of 2",
        ));
    }

    // GPU memory utilization
    if config.engine.gpu_memory_utilization <= 0.0 || config.engine.gpu_memory_utilization > 1.0 {
        return Err(ConfigError::invalid_value(
            "engine.gpu_memory_utilization",
            "must be between 0.0 and 1.0",
        ));
    }

    // Swap space
    if config.engine.swap_space_gb < 0.0 {
        return Err(ConfigError::invalid_value(
            "engine.swap_space_gb",
            "cannot be negative",
        ));
    }

    // Speculative decoding
    if let Some(ref spec) = config.engine.speculative {
        if spec.draft_model.is_empty() {
            return Err(ConfigError::missing_field("engine.speculative.draft_model"));
        }
        if spec.num_speculative_tokens == 0 {
            return Err(ConfigError::invalid_value(
                "engine.speculative.num_speculative_tokens",
                "must be greater than 0",
            ));
        }
        if spec.acceptance_threshold <= 0.0 || spec.acceptance_threshold > 1.0 {
            return Err(ConfigError::invalid_value(
                "engine.speculative.acceptance_threshold",
                "must be between 0.0 and 1.0",
            ));
        }
    }

    Ok(())
}

/// Check if a model path looks like a HuggingFace repo.
pub fn is_huggingface_repo(path: &str) -> bool {
    // HuggingFace repos have format: org/model or model
    !path.contains('/') 
        || (path.matches('/').count() == 1 && !path.starts_with('/') && !path.ends_with('/'))
}

/// Check if a model path is a local path.
pub fn is_local_path(path: &str) -> bool {
    std::path::Path::new(path).exists()
        || path.starts_with('/')
        || path.starts_with("./")
        || path.starts_with("../")
        || (cfg!(windows) && path.chars().nth(1) == Some(':'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_repo_detection() {
        assert!(is_huggingface_repo("meta-llama/Llama-2-7b"));
        assert!(is_huggingface_repo("mistralai/Mistral-7B-v0.1"));
        assert!(!is_huggingface_repo("/models/llama"));
        assert!(!is_huggingface_repo("./models/llama"));
    }
}
