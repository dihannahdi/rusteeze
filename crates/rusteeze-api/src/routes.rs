//! API routes configuration.

use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    trace::TraceLayer,
};

use crate::handlers::{self, AppState};
use crate::middleware;

/// Create the API router.
pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions_handler))
        .route("/v1/models", get(handlers::list_models))

        // Health endpoints
        .route("/health", get(handlers::health_check))
        .route("/health/live", get(liveness_probe))
        .route("/health/ready", get(readiness_probe))

        // Metrics endpoint
        .route("/metrics", get(metrics_handler))

        // Version
        .route("/version", get(version_handler))

        // State
        .with_state(state)

        // Middleware layers
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
}

/// Chat completions route handler - delegates based on stream flag.
async fn chat_completions_handler(
    state: axum::extract::State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Result<axum::response::Response, crate::error::ApiError> {
    use axum::response::IntoResponse;

    // Parse the request to check stream flag
    let request: crate::types::ChatCompletionRequest =
        serde_json::from_slice(&body).map_err(|e| crate::error::ApiError::BadRequest(e.to_string()))?;

    if request.stream.unwrap_or(false) {
        let response = handlers::chat_completions_stream(state, axum::Json(request)).await?;
        Ok(response.into_response())
    } else {
        let response = handlers::chat_completions(state, axum::Json(request)).await?;
        Ok(response.into_response())
    }
}

/// Liveness probe.
async fn liveness_probe() -> &'static str {
    "OK"
}

/// Readiness probe.
async fn readiness_probe(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Result<&'static str, crate::error::ApiError> {
    // Check if engine is ready
    let stats = state.engine.stats();

    // Ready if engine is accepting requests
    if stats.is_running {
        Ok("OK")
    } else {
        Err(crate::error::ApiError::ServiceUnavailable("Engine not ready".to_string()))
    }
}

/// Metrics handler (Prometheus format).
async fn metrics_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> String {
    let stats = state.engine.stats();
    let uptime = state.start_time.elapsed().as_secs();
    let requests = state.request_counter.load(std::sync::atomic::Ordering::Relaxed);

    format!(
        r#"# HELP rusteeze_uptime_seconds Time since server started
# TYPE rusteeze_uptime_seconds gauge
rusteeze_uptime_seconds {}

# HELP rusteeze_requests_total Total number of requests processed
# TYPE rusteeze_requests_total counter
rusteeze_requests_total {}

# HELP rusteeze_waiting_requests Number of waiting requests
# TYPE rusteeze_waiting_requests gauge
rusteeze_waiting_requests {}

# HELP rusteeze_running_requests Number of running requests
# TYPE rusteeze_running_requests gauge
rusteeze_running_requests {}

# HELP rusteeze_swapped_requests Number of swapped requests
# TYPE rusteeze_swapped_requests gauge
rusteeze_swapped_requests {}

# HELP rusteeze_gpu_memory_usage_bytes GPU memory usage
# TYPE rusteeze_gpu_memory_usage_bytes gauge
rusteeze_gpu_memory_usage_bytes {}

# HELP rusteeze_tokens_generated_total Total tokens generated
# TYPE rusteeze_tokens_generated_total counter
rusteeze_tokens_generated_total {}

# HELP rusteeze_avg_generation_time_ms Average generation time
# TYPE rusteeze_avg_generation_time_ms gauge
rusteeze_avg_generation_time_ms {}
"#,
        uptime,
        requests,
        stats.num_waiting,
        stats.num_running,
        stats.num_swapped,
        stats.gpu_memory_usage,
        stats.total_tokens_generated,
        stats.avg_generation_time_ms,
    )
}

/// Version handler.
async fn version_handler() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "name": "Rusteeze",
        "version": env!("CARGO_PKG_VERSION"),
        "rust_version": env!("CARGO_PKG_RUST_VERSION"),
        "git_commit": option_env!("GIT_COMMIT").unwrap_or("unknown"),
        "build_time": option_env!("BUILD_TIME").unwrap_or("unknown"),
    }))
}

/// Create router with custom middleware.
pub fn create_router_with_auth(
    state: Arc<AppState>,
    api_key: Option<String>,
) -> Router {
    let base_router = create_router(state);

    if let Some(key) = api_key {
        base_router.layer(axum::middleware::from_fn(move |req, next| {
            middleware::auth_middleware(req, next, key.clone())
        }))
    } else {
        base_router
    }
}

/// Router builder for more complex configurations.
pub struct RouterBuilder {
    state: Arc<AppState>,
    api_key: Option<String>,
    cors_enabled: bool,
    compression_enabled: bool,
    tracing_enabled: bool,
    rate_limit: Option<u32>,
}

impl RouterBuilder {
    /// Create a new router builder.
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            state,
            api_key: None,
            cors_enabled: true,
            compression_enabled: true,
            tracing_enabled: true,
            rate_limit: None,
        }
    }

    /// Set API key for authentication.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Enable/disable CORS.
    pub fn with_cors(mut self, enabled: bool) -> Self {
        self.cors_enabled = enabled;
        self
    }

    /// Enable/disable compression.
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    /// Enable/disable tracing.
    pub fn with_tracing(mut self, enabled: bool) -> Self {
        self.tracing_enabled = enabled;
        self
    }

    /// Set rate limit (requests per second).
    pub fn with_rate_limit(mut self, limit: u32) -> Self {
        self.rate_limit = Some(limit);
        self
    }

    /// Build the router.
    pub fn build(self) -> Router {
        let mut router = Router::new()
            .route("/v1/chat/completions", post(chat_completions_handler))
            .route("/v1/models", get(handlers::list_models))
            .route("/health", get(handlers::health_check))
            .route("/health/live", get(liveness_probe))
            .route("/health/ready", get(readiness_probe))
            .route("/metrics", get(metrics_handler))
            .route("/version", get(version_handler))
            .with_state(self.state);

        if self.tracing_enabled {
            router = router.layer(TraceLayer::new_for_http());
        }

        if self.compression_enabled {
            router = router.layer(CompressionLayer::new());
        }

        if self.cors_enabled {
            router = router.layer(CorsLayer::permissive());
        }

        // Add auth if configured
        if let Some(key) = self.api_key {
            router = router.layer(axum::middleware::from_fn(move |req, next| {
                middleware::auth_middleware(req, next, key.clone())
            }));
        }

        router
    }
}
