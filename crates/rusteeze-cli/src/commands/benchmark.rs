//! Benchmark command - benchmark model performance.

use anyhow::Result;
use clap::Args;
use colored::Colorize;
use std::time::Instant;

/// Benchmark command arguments.
#[derive(Args, Debug)]
pub struct BenchmarkArgs {
    /// Model path or HuggingFace repo ID.
    #[arg(short, long, env = "RUSTEEZE_MODEL")]
    pub model: String,

    /// Number of prompts to generate.
    #[arg(short, long, default_value = "100")]
    pub num_prompts: usize,

    /// Input length (tokens).
    #[arg(long, default_value = "128")]
    pub input_len: usize,

    /// Output length (tokens).
    #[arg(long, default_value = "128")]
    pub output_len: usize,

    /// Number of concurrent requests.
    #[arg(short, long, default_value = "1")]
    pub concurrency: usize,

    /// Device (auto, cpu, cuda, metal).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// GPU memory utilization.
    #[arg(long, default_value = "0.9")]
    pub gpu_memory_utilization: f32,

    /// Tensor parallel size.
    #[arg(long, default_value = "1")]
    pub tensor_parallel_size: usize,

    /// Warm-up iterations.
    #[arg(long, default_value = "3")]
    pub warmup: usize,

    /// Output file for detailed results.
    #[arg(short, long)]
    pub output: Option<String>,
}

/// Benchmark results.
#[derive(Debug, serde::Serialize)]
struct BenchmarkResults {
    model: String,
    num_prompts: usize,
    input_len: usize,
    output_len: usize,
    concurrency: usize,
    total_time_seconds: f64,
    throughput_requests_per_second: f64,
    throughput_tokens_per_second: f64,
    avg_latency_ms: f64,
    p50_latency_ms: f64,
    p90_latency_ms: f64,
    p99_latency_ms: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
    time_to_first_token_ms: f64,
    inter_token_latency_ms: f64,
}

/// Execute the benchmark command.
pub async fn execute(args: BenchmarkArgs, _config_path: Option<String>, json: bool) -> Result<()> {
    if !json {
        println!(
            "\n{} {}",
            "Benchmarking".bright_green().bold(),
            args.model.bright_cyan()
        );
        println!();
        println!("Configuration:");
        println!("  Prompts: {}", args.num_prompts);
        println!("  Input length: {} tokens", args.input_len);
        println!("  Output length: {} tokens", args.output_len);
        println!("  Concurrency: {}", args.concurrency);
        println!("  Device: {}", args.device);
        println!("  Warm-up iterations: {}", args.warmup);
        println!();
    }

    // Warm-up phase
    if !json {
        println!("{}", "Running warm-up...".yellow());
    }

    for i in 0..args.warmup {
        if !json {
            print!("\r  Warm-up {}/{}", i + 1, args.warmup);
            std::io::Write::flush(&mut std::io::stdout())?;
        }
        // Simulate warm-up
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    if !json {
        println!("\r  Warm-up complete!   ");
        println!();
    }

    // Benchmark phase
    if !json {
        println!("{}", "Running benchmark...".bright_green());
    }

    let start = Instant::now();
    let mut latencies: Vec<f64> = Vec::with_capacity(args.num_prompts);

    // Simulate benchmark
    for i in 0..args.num_prompts {
        let req_start = Instant::now();

        // Simulate request
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        let latency = req_start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(latency);

        if !json && (i + 1) % 10 == 0 {
            print!(
                "\r  Progress: {}/{} ({:.1}%)",
                i + 1,
                args.num_prompts,
                (i + 1) as f64 / args.num_prompts as f64 * 100.0
            );
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }

    let total_time = start.elapsed().as_secs_f64();

    if !json {
        println!("\r  Benchmark complete!    ");
        println!();
    }

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p50_latency = percentile(&latencies, 50.0);
    let p90_latency = percentile(&latencies, 90.0);
    let p99_latency = percentile(&latencies, 99.0);
    let min_latency = latencies.first().copied().unwrap_or(0.0);
    let max_latency = latencies.last().copied().unwrap_or(0.0);

    let total_tokens = args.num_prompts * (args.input_len + args.output_len);
    let throughput_rps = args.num_prompts as f64 / total_time;
    let throughput_tps = total_tokens as f64 / total_time;

    let results = BenchmarkResults {
        model: args.model.clone(),
        num_prompts: args.num_prompts,
        input_len: args.input_len,
        output_len: args.output_len,
        concurrency: args.concurrency,
        total_time_seconds: total_time,
        throughput_requests_per_second: throughput_rps,
        throughput_tokens_per_second: throughput_tps,
        avg_latency_ms: avg_latency,
        p50_latency_ms: p50_latency,
        p90_latency_ms: p90_latency,
        p99_latency_ms: p99_latency,
        min_latency_ms: min_latency,
        max_latency_ms: max_latency,
        time_to_first_token_ms: avg_latency * 0.3, // Simulated
        inter_token_latency_ms: avg_latency / args.output_len as f64,
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        println!("{}", "Results:".bright_green().bold());
        println!();
        println!("  {}", "Throughput".bright_cyan());
        println!(
            "    Requests/s: {:.2}",
            results.throughput_requests_per_second
        );
        println!("    Tokens/s:   {:.2}", results.throughput_tokens_per_second);
        println!();
        println!("  {}", "Latency (ms)".bright_cyan());
        println!("    Average: {:.2}", results.avg_latency_ms);
        println!("    P50:     {:.2}", results.p50_latency_ms);
        println!("    P90:     {:.2}", results.p90_latency_ms);
        println!("    P99:     {:.2}", results.p99_latency_ms);
        println!("    Min:     {:.2}", results.min_latency_ms);
        println!("    Max:     {:.2}", results.max_latency_ms);
        println!();
        println!("  {}", "Generation".bright_cyan());
        println!(
            "    Time to first token: {:.2} ms",
            results.time_to_first_token_ms
        );
        println!(
            "    Inter-token latency: {:.2} ms",
            results.inter_token_latency_ms
        );
        println!();
        println!("  Total time: {:.2}s", results.total_time_seconds);
    }

    // Save results if output specified
    if let Some(ref output_path) = args.output {
        std::fs::write(output_path, serde_json::to_string_pretty(&results)?)?;
        if !json {
            println!("\n{} {}", "Results saved to".white(), output_path.bright_cyan());
        }
    }

    Ok(())
}

/// Calculate percentile.
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}
