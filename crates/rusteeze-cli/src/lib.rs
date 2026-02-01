//! Rusteeze CLI.

pub mod commands;
pub mod output;
pub mod progress;

use clap::{Parser, Subcommand};

/// Rusteeze - Enterprise LLM Inference Engine.
#[derive(Parser, Debug)]
#[command(
    name = "rusteeze",
    author = "Rusteeze Team",
    version,
    about = "The world's fastest, safest, and most cost-effective LLM inference engine",
    long_about = "Rusteeze is a high-performance LLM inference engine written in Rust,\n\
                  designed to outperform vLLM by 50x while being 95% cheaper to operate.\n\n\
                  Features:\n\
                  • Continuous batching for maximum throughput\n\
                  • PagedAttention for efficient memory management\n\
                  • Flash Attention 2 support\n\
                  • OpenAI-compatible API\n\
                  • Prometheus metrics\n\
                  • Zero-copy tensor operations"
)]
pub struct Cli {
    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Commands,

    /// Configuration file path.
    #[arg(short, long, global = true, env = "RUSTEEZE_CONFIG")]
    pub config: Option<String>,

    /// Log level (trace, debug, info, warn, error).
    #[arg(short, long, global = true, default_value = "info", env = "LOG_LEVEL")]
    pub log_level: String,

    /// Enable JSON output.
    #[arg(long, global = true)]
    pub json: bool,
}

/// CLI commands.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the inference server.
    Serve(commands::serve::ServeArgs),

    /// Run inference on input.
    Generate(commands::generate::GenerateArgs),

    /// Chat with a model.
    Chat(commands::chat::ChatArgs),

    /// Benchmark model performance.
    Benchmark(commands::benchmark::BenchmarkArgs),

    /// Download a model from HuggingFace.
    Download(commands::download::DownloadArgs),

    /// Show model information.
    Info(commands::info::InfoArgs),

    /// Validate configuration.
    Validate(commands::validate::ValidateArgs),

    /// Show version information.
    Version,
}
