//! Serve command - start the inference server.

use anyhow::Result;
use clap::Args;
use colored::Colorize;
use tracing::info;

/// Serve command arguments.
#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Model path or HuggingFace repo ID.
    #[arg(short, long, env = "RUSTEEZE_MODEL")]
    pub model: String,

    /// Host to bind to.
    #[arg(long, default_value = "0.0.0.0", env = "RUSTEEZE_HOST")]
    pub host: String,

    /// Port to listen on.
    #[arg(short, long, default_value = "8000", env = "RUSTEEZE_PORT")]
    pub port: u16,

    /// API key for authentication.
    #[arg(long, env = "RUSTEEZE_API_KEY")]
    pub api_key: Option<String>,

    /// GPU memory utilization (0.0-1.0).
    #[arg(long, default_value = "0.9")]
    pub gpu_memory_utilization: f32,

    /// Maximum number of sequences.
    #[arg(long, default_value = "256")]
    pub max_num_seqs: usize,

    /// Maximum model length.
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Tensor parallel size.
    #[arg(long, default_value = "1")]
    pub tensor_parallel_size: usize,

    /// Data type (auto, float16, bfloat16, float32).
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Quantization method.
    #[arg(long)]
    pub quantization: Option<String>,

    /// Device (auto, cpu, cuda, metal).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Trust remote code.
    #[arg(long)]
    pub trust_remote_code: bool,

    /// Enable prefix caching.
    #[arg(long)]
    pub enable_prefix_caching: bool,

    /// Enable chunked prefill.
    #[arg(long)]
    pub enable_chunked_prefill: bool,

    /// Block size for paged attention.
    #[arg(long, default_value = "16")]
    pub block_size: usize,

    /// Swap space in GB.
    #[arg(long, default_value = "4")]
    pub swap_space: f32,

    /// Enable CORS.
    #[arg(long, default_value = "true")]
    pub cors: bool,

    /// Metrics port.
    #[arg(long, default_value = "9090")]
    pub metrics_port: u16,

    /// Chat template override.
    #[arg(long)]
    pub chat_template: Option<String>,
}

/// Execute the serve command.
pub async fn execute(args: ServeArgs, config_path: Option<String>) -> Result<()> {
    println!(
        "{} {} on {}:{}",
        "Starting server for".white(),
        args.model.bright_cyan(),
        args.host.bright_yellow(),
        args.port.to_string().bright_yellow()
    );

    // Load configuration
    let config = if let Some(path) = config_path {
        info!("Loading configuration from {}", path);
        rusteeze_config::Config::from_file(&path)?
    } else {
        build_config_from_args(&args)?
    };

    // Print configuration
    println!("\n{}", "Configuration:".bright_green().bold());
    println!("  Model: {}", config.model.path.bright_cyan());
    println!("  Device: {}", config.model.device.bright_yellow());
    println!("  Dtype: {}", config.model.dtype.bright_yellow());
    println!(
        "  Tensor Parallel: {}",
        config.model.tensor_parallel_size.to_string().bright_yellow()
    );
    println!(
        "  Max Sequences: {}",
        config.engine.max_num_seqs.to_string().bright_yellow()
    );
    println!(
        "  GPU Memory: {}%",
        (config.engine.gpu_memory_utilization * 100.0)
            .round()
            .to_string()
            .bright_yellow()
    );

    // Initialize metrics
    println!("\n{}", "Initializing metrics...".white());
    rusteeze_metrics::init_metrics(rusteeze_metrics::MetricsConfig::default())?;

    // Start metrics server
    let metrics_port = args.metrics_port;
    tokio::spawn(async move {
        if let Err(e) = rusteeze_metrics::prometheus::start_exporter_on_port(metrics_port).await {
            tracing::error!("Metrics server error: {}", e);
        }
    });

    // Load model
    println!("\n{}", "Loading model...".white());
    // Note: Model loading would happen here
    // let model = rusteeze_model::load_model(&config.model).await?;

    // Create engine
    println!("{}", "Creating inference engine...".white());
    // Note: Engine creation would happen here
    // let engine = rusteeze_engine::Engine::new(model, config.engine)?;

    // Start server
    println!("\n{}", "Starting API server...".bright_green().bold());
    println!(
        "  OpenAI API: {}",
        format!("http://{}:{}/v1/chat/completions", args.host, args.port).bright_cyan()
    );
    println!(
        "  Health: {}",
        format!("http://{}:{}/health", args.host, args.port).bright_cyan()
    );
    println!(
        "  Metrics: {}",
        format!("http://{}:{}/metrics", args.host, metrics_port).bright_cyan()
    );

    // Note: Server would start here
    // rusteeze_api::serve(engine, config.model.path, args.host, args.port).await?;

    // For now, just wait
    println!("\n{}", "Server ready! Press Ctrl+C to stop.".bright_green());

    tokio::signal::ctrl_c().await?;

    println!("\n{}", "Shutting down...".yellow());

    Ok(())
}

/// Build configuration from CLI arguments.
fn build_config_from_args(args: &ServeArgs) -> Result<rusteeze_config::Config> {
    use rusteeze_config::{Config, EngineConfig, ModelConfig, ServerConfig};

    let mut model_config = ModelConfig::new(&args.model)
        .with_dtype(&args.dtype)
        .with_device(&args.device)
        .with_tensor_parallel(args.tensor_parallel_size);

    if let Some(len) = args.max_model_len {
        model_config = model_config.with_max_model_len(len);
    }

    model_config.trust_remote_code = args.trust_remote_code;
    model_config.chat_template = args.chat_template.clone();

    let mut server_config = ServerConfig::new()
        .with_host(&args.host)
        .with_port(args.port);

    if let Some(ref key) = args.api_key {
        server_config = server_config.with_api_key(key);
    }

    let engine_config = EngineConfig::new()
        .with_max_seqs(args.max_num_seqs)
        .with_gpu_utilization(args.gpu_memory_utilization)
        .with_block_size(args.block_size)
        .with_prefix_caching(args.enable_prefix_caching)
        .with_chunked_prefill(args.enable_chunked_prefill);

    Ok(Config {
        model: model_config,
        server: server_config,
        engine: engine_config,
        ..Default::default()
    })
}
