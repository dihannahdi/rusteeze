//! Download command - download models from HuggingFace.

use anyhow::Result;
use clap::Args;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

/// Download command arguments.
#[derive(Args, Debug)]
pub struct DownloadArgs {
    /// Model repository ID (e.g., meta-llama/Llama-2-7b-hf).
    pub model_id: String,

    /// Revision/branch to download.
    #[arg(long, default_value = "main")]
    pub revision: String,

    /// Output directory.
    #[arg(short, long)]
    pub output: Option<String>,

    /// HuggingFace token for private models.
    #[arg(long, env = "HF_TOKEN")]
    pub token: Option<String>,

    /// Only download model weights (no tokenizer).
    #[arg(long)]
    pub weights_only: bool,

    /// Download specific files (comma-separated).
    #[arg(long)]
    pub files: Option<String>,

    /// Resume incomplete download.
    #[arg(long)]
    pub resume: bool,
}

/// Execute the download command.
pub async fn execute(args: DownloadArgs, json: bool) -> Result<()> {
    let output_dir = args
        .output
        .unwrap_or_else(|| format!("models/{}", args.model_id.replace('/', "_")));

    if !json {
        println!(
            "\n{} {}",
            "Downloading".bright_green().bold(),
            args.model_id.bright_cyan()
        );
        println!("  Revision: {}", args.revision.bright_yellow());
        println!("  Output: {}", output_dir.bright_yellow());
        if args.weights_only {
            println!("  Mode: weights only");
        }
        println!();
    }

    // Create output directory
    std::fs::create_dir_all(&output_dir)?;

    // Simulate file list
    let files = if args.weights_only {
        vec![
            ("model.safetensors", 14_000_000_000u64),
            ("config.json", 1_500),
        ]
    } else {
        vec![
            ("model.safetensors", 14_000_000_000),
            ("config.json", 1_500),
            ("tokenizer.json", 2_500_000),
            ("tokenizer_config.json", 500),
            ("special_tokens_map.json", 300),
            ("generation_config.json", 200),
        ]
    };

    // Filter files if specified
    let files: Vec<_> = if let Some(ref filter) = args.files {
        let patterns: Vec<&str> = filter.split(',').map(|s| s.trim()).collect();
        files
            .into_iter()
            .filter(|(name, _)| patterns.iter().any(|p| name.contains(p)))
            .collect()
    } else {
        files
    };

    if !json {
        println!("{} Downloading {} files...", "→".bright_blue(), files.len());
        println!();
    }

    let mut downloaded_files = Vec::new();

    for (filename, size) in &files {
        if !json {
            let pb = ProgressBar::new(*size);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_message(filename.to_string());

            // Simulate download with progress
            let chunk_size = size / 100;
            for _ in 0..100 {
                tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
                pb.inc(chunk_size);
            }
            pb.finish_with_message(format!("✓ {}", filename));
        } else {
            // Simulate download
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        downloaded_files.push(serde_json::json!({
            "name": filename,
            "size": size,
            "path": format!("{}/{}", output_dir, filename),
        }));
    }

    if json {
        let result = serde_json::json!({
            "model_id": args.model_id,
            "revision": args.revision,
            "output_dir": output_dir,
            "files": downloaded_files,
        });
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!();
        println!(
            "{} Downloaded {} to {}",
            "✓".bright_green(),
            args.model_id.bright_cyan(),
            output_dir.bright_yellow()
        );

        // Print size summary
        let total_size: u64 = files.iter().map(|(_, s)| s).sum();
        println!(
            "  Total size: {}",
            format_size(total_size).bright_yellow()
        );
    }

    Ok(())
}

/// Format file size.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
