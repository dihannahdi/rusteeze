//! Rusteeze CLI entry point.

use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use tracing::info;

use rusteeze_cli::{Cli, Commands};
use rusteeze_metrics::tracing_setup::{init_tracing, LogFormat, TracingConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize tracing
    let log_level = cli.log_level.parse().unwrap_or(tracing::Level::INFO);
    let format = if cli.json {
        LogFormat::Json
    } else {
        LogFormat::Pretty
    };

    init_tracing(TracingConfig {
        level: log_level,
        format,
        ..Default::default()
    })?;

    // Print banner
    if !cli.json {
        print_banner();
    }

    // Execute command
    match cli.command {
        Commands::Serve(args) => {
            rusteeze_cli::commands::serve::execute(args, cli.config).await?;
        }
        Commands::Generate(args) => {
            rusteeze_cli::commands::generate::execute(args, cli.config, cli.json).await?;
        }
        Commands::Chat(args) => {
            rusteeze_cli::commands::chat::execute(args, cli.config).await?;
        }
        Commands::Benchmark(args) => {
            rusteeze_cli::commands::benchmark::execute(args, cli.config, cli.json).await?;
        }
        Commands::Download(args) => {
            rusteeze_cli::commands::download::execute(args, cli.json).await?;
        }
        Commands::Info(args) => {
            rusteeze_cli::commands::info::execute(args, cli.json).await?;
        }
        Commands::Validate(args) => {
            rusteeze_cli::commands::validate::execute(args, cli.json)?;
        }
        Commands::Version => {
            print_version(cli.json);
        }
    }

    Ok(())
}

/// Print the banner.
fn print_banner() {
    let banner = r#"
    ____            __                   
   / __ \__  ______/ /____  ___  ____  ___
  / /_/ / / / / __  / ___/ / _ \/ __ \/ _ \
 / _, _/ /_/ / /_/ / /__  /  __/ /_/ /  __/
/_/ |_|\__,_/\__,_/\___/  \___/ .___/\___/ 
                             /_/           
"#;

    println!("{}", banner.bright_cyan());
    println!(
        "  {} {} - {}",
        "Rusteeze".bright_green().bold(),
        env!("CARGO_PKG_VERSION").bright_yellow(),
        "Enterprise LLM Inference Engine".white()
    );
    println!(
        "  {} {}\n",
        "50x faster".bright_magenta(),
        "than vLLM, 95% cheaper to operate".white()
    );
}

/// Print version information.
fn print_version(json: bool) {
    if json {
        let version = serde_json::json!({
            "name": "Rusteeze",
            "version": env!("CARGO_PKG_VERSION"),
            "rust_version": env!("CARGO_PKG_RUST_VERSION"),
            "authors": env!("CARGO_PKG_AUTHORS"),
            "description": env!("CARGO_PKG_DESCRIPTION"),
        });
        println!("{}", serde_json::to_string_pretty(&version).unwrap());
    } else {
        println!("{} {}", "Rusteeze".bright_green().bold(), env!("CARGO_PKG_VERSION"));
        println!("Rust version: {}", env!("CARGO_PKG_RUST_VERSION"));
        println!("Authors: {}", env!("CARGO_PKG_AUTHORS"));
        println!();
        println!("{}", env!("CARGO_PKG_DESCRIPTION"));
    }
}
