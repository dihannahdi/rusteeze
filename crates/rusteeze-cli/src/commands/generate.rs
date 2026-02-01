//! Generate command - run inference on input.

use anyhow::Result;
use clap::Args;
use colored::Colorize;

/// Generate command arguments.
#[derive(Args, Debug)]
pub struct GenerateArgs {
    /// Model path or HuggingFace repo ID.
    #[arg(short, long, env = "RUSTEEZE_MODEL")]
    pub model: String,

    /// Input prompt.
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Input file containing prompts (one per line).
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    /// Maximum tokens to generate.
    #[arg(long, default_value = "256")]
    pub max_tokens: usize,

    /// Temperature for sampling.
    #[arg(long, default_value = "1.0")]
    pub temperature: f32,

    /// Top-p (nucleus sampling).
    #[arg(long, default_value = "1.0")]
    pub top_p: f32,

    /// Top-k sampling.
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Repetition penalty.
    #[arg(long, default_value = "1.0")]
    pub repetition_penalty: f32,

    /// Stop sequences (can be specified multiple times).
    #[arg(long)]
    pub stop: Vec<String>,

    /// Random seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of completions to generate.
    #[arg(short, long, default_value = "1")]
    pub n: usize,

    /// Stream output.
    #[arg(long)]
    pub stream: bool,

    /// Device (auto, cpu, cuda, metal).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Output file.
    #[arg(short, long)]
    pub output: Option<String>,
}

/// Execute the generate command.
pub async fn execute(args: GenerateArgs, _config_path: Option<String>, json: bool) -> Result<()> {
    // Get prompts
    let prompts = if let Some(ref prompt) = args.prompt {
        vec![prompt.clone()]
    } else if let Some(ref file) = args.file {
        std::fs::read_to_string(file)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(String::from)
            .collect()
    } else {
        anyhow::bail!("Either --prompt or --file must be specified");
    };

    if !json {
        println!(
            "{} with model {}",
            "Generating".bright_green(),
            args.model.bright_cyan()
        );
        println!("  Max tokens: {}", args.max_tokens);
        println!("  Temperature: {}", args.temperature);
        println!("  Top-p: {}", args.top_p);
        if let Some(k) = args.top_k {
            println!("  Top-k: {}", k);
        }
        println!();
    }

    // Process each prompt
    for (i, prompt) in prompts.iter().enumerate() {
        if !json {
            if prompts.len() > 1 {
                println!("{} {}", "Prompt".bright_yellow().bold(), i + 1);
            }
            println!("{}: {}", "Input".bright_cyan(), prompt);
            println!();
        }

        // Generate
        // Note: Actual generation would happen here
        let output = format!("[Generated response for: {}]", prompt);

        if json {
            let result = serde_json::json!({
                "prompt": prompt,
                "output": output,
                "model": args.model,
                "tokens": {
                    "prompt": prompt.split_whitespace().count(),
                    "completion": 10,
                }
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            if args.stream {
                // Stream output character by character
                for c in output.chars() {
                    print!("{}", c);
                    std::io::Write::flush(&mut std::io::stdout())?;
                    tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
                }
                println!();
            } else {
                println!("{}: {}", "Output".bright_green(), output);
            }
        }

        if !json && prompts.len() > 1 && i < prompts.len() - 1 {
            println!("\n{}", "─".repeat(50));
        }
    }

    // Write output file if specified
    if let Some(ref output_path) = args.output {
        if !json {
            println!("\n{} {}", "Output written to".white(), output_path.bright_cyan());
        }
    }

    Ok(())
}
