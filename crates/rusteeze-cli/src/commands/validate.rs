//! Validate command - validate configuration files.

use anyhow::Result;
use clap::Args;
use colored::Colorize;

use rusteeze_config::{Config, ConfigLoader};

/// Validate command arguments.
#[derive(Args, Debug)]
pub struct ValidateArgs {
    /// Configuration file to validate.
    pub config_file: String,

    /// Also validate model path exists.
    #[arg(long)]
    pub check_model: bool,

    /// Verbose output.
    #[arg(short, long)]
    pub verbose: bool,
}

/// Execute the validate command.
pub fn execute(args: ValidateArgs, json: bool) -> Result<()> {
    if !json {
        println!(
            "\n{} {}",
            "Validating".bright_green().bold(),
            args.config_file.bright_cyan()
        );
        println!();
    }

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Try to load the configuration
    let config = match ConfigLoader::new()
        .with_file(&args.config_file)
        .load()
    {
        Ok(c) => {
            if !json && args.verbose {
                println!("  {} Configuration loaded successfully", "✓".bright_green());
            }
            Some(c)
        }
        Err(e) => {
            errors.push(format!("Failed to load configuration: {}", e));
            None
        }
    };

    if let Some(ref config) = config {
        // Validate model configuration
        if config.model.path.is_empty() {
            errors.push("model.path is required".to_string());
        }

        // Check model exists if requested
        if args.check_model && !config.model.path.is_empty() {
            let path = std::path::Path::new(&config.model.path);
            if !path.exists() && !config.model.path.contains('/') {
                // Could be a HuggingFace model ID
                warnings.push(format!(
                    "Model path '{}' does not exist locally (may be HuggingFace ID)",
                    config.model.path
                ));
            } else if !path.exists() {
                errors.push(format!("Model path '{}' does not exist", config.model.path));
            }
        }

        // Validate server configuration
        if config.server.port == 0 {
            errors.push("server.port must be between 1 and 65535".to_string());
        }

        // Validate engine configuration
        if config.engine.gpu_memory_utilization <= 0.0 || config.engine.gpu_memory_utilization > 1.0
        {
            errors.push("engine.gpu_memory_utilization must be between 0.0 and 1.0".to_string());
        }

        if config.engine.max_num_seqs == 0 {
            errors.push("engine.max_num_seqs must be greater than 0".to_string());
        }

        if !config.engine.block_size.is_power_of_two() {
            warnings.push("engine.block_size should be a power of 2 for optimal performance".to_string());
        }

        // Validate TLS if enabled
        if let Some(ref tls) = config.server.tls {
            if tls.cert_path.is_empty() {
                errors.push("server.tls.cert_path is required when TLS is enabled".to_string());
            }
            if tls.key_path.is_empty() {
                errors.push("server.tls.key_path is required when TLS is enabled".to_string());
            }
        }

        // Verbose output
        if !json && args.verbose {
            println!();
            println!("  {}", "Configuration Summary".bright_cyan().underline());
            println!("    Model: {}", config.model.path);
            println!("    Server: {}:{}", config.server.host, config.server.port);
            println!("    Max Sequences: {}", config.engine.max_num_seqs);
            println!(
                "    GPU Memory: {}%",
                (config.engine.gpu_memory_utilization * 100.0).round()
            );
            println!("    Block Size: {}", config.engine.block_size);
            println!();
        }
    }

    // Output results
    if json {
        let result = serde_json::json!({
            "file": args.config_file,
            "valid": errors.is_empty(),
            "errors": errors,
            "warnings": warnings,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        if !errors.is_empty() {
            println!("  {}", "Errors:".bright_red().bold());
            for error in &errors {
                println!("    {} {}", "✗".bright_red(), error);
            }
            println!();
        }

        if !warnings.is_empty() {
            println!("  {}", "Warnings:".bright_yellow().bold());
            for warning in &warnings {
                println!("    {} {}", "⚠".bright_yellow(), warning);
            }
            println!();
        }

        if errors.is_empty() {
            println!(
                "  {} Configuration is {}",
                "✓".bright_green(),
                "valid".bright_green().bold()
            );
        } else {
            println!(
                "  {} Configuration is {}",
                "✗".bright_red(),
                "invalid".bright_red().bold()
            );
        }
    }

    if !errors.is_empty() {
        anyhow::bail!("Configuration validation failed");
    }

    Ok(())
}
