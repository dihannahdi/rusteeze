//! API request handlers.

use std::sync::Arc;

use axum::{
    extract::State,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::Stream;
use tokio_stream::StreamExt;
use tracing::{debug, info, warn};

use rusteeze_core::{FinishReason, SamplingParams};
use rusteeze_engine::{Engine, GenerationRequest};

use crate::error::ApiError;
use crate::types::*;

/// Application state.
pub struct AppState {
    /// Engine.
    pub engine: Arc<Engine>,

    /// Model ID.
    pub model_id: String,

    /// Start time.
    pub start_time: std::time::Instant,

    /// Request counter.
    pub request_counter: std::sync::atomic::AtomicU64,
}

/// Chat completions handler.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    debug!("Chat completion request for model: {}", request.model);

    // Increment counter
    state.request_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Build prompt from messages
    let prompt = build_prompt(&request.messages);

    // Build sampling params
    let sampling_params = build_sampling_params(&request);

    // Create generation request
    let gen_request = GenerationRequest {
        request_id: uuid::Uuid::new_v4().to_string(),
        prompt,
        sampling_params,
        max_tokens: request.max_tokens.unwrap_or(256) as usize,
        stream: false,
    };

    // Generate
    let output = state.engine.generate(gen_request).await?;

    // Build response
    let choice = ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: output.text,
            name: None,
        },
        finish_reason: output.finish_reason.map(|r| match r {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ContentFilter => "content_filter",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::StopSequence => "stop",
            _ => "stop",
        }.to_string()),
        logprobs: None,
    };

    let usage = Usage {
        prompt_tokens: output.usage.prompt_tokens,
        completion_tokens: output.usage.completion_tokens,
        total_tokens: output.usage.total_tokens,
    };

    let response = ChatCompletionResponse::new(
        state.model_id.clone(),
        vec![choice],
        usage,
    );

    Ok(Json(response))
}

/// Streaming chat completions handler.
pub async fn chat_completions_stream(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>> + Send>, ApiError> {
    debug!("Streaming chat completion request for model: {}", request.model);

    // Increment counter
    state.request_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Build prompt
    let prompt = build_prompt(&request.messages);

    // Build sampling params
    let sampling_params = build_sampling_params(&request);

    // Create generation request
    let request_id = uuid::Uuid::new_v4().to_string();
    let gen_request = GenerationRequest {
        request_id: request_id.clone(),
        prompt,
        sampling_params,
        max_tokens: request.max_tokens.unwrap_or(256) as usize,
        stream: true,
    };

    // Get stream
    let mut rx = state.engine.generate_stream(gen_request).await?;
    let model_id = state.model_id.clone();
    let chunk_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    // First chunk with role
    let first_chunk = ChatCompletionChunk::new(
        chunk_id.clone(),
        model_id.clone(),
        vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
    );

    let stream = async_stream::stream! {
        // Send first chunk
        let data = serde_json::to_string(&first_chunk).unwrap_or_default();
        yield Ok(Event::default().data(data));

        // Stream content
        while let Some(chunk) = rx.recv().await {
            let delta = ChatDelta {
                role: None,
                content: if chunk.text.is_empty() { None } else { Some(chunk.text) },
            };

            let finish_reason = chunk.finish_reason.map(|r| match r {
                FinishReason::Stop => "stop",
                FinishReason::Length => "length",
                _ => "stop",
            }.to_string());

            let chunk_response = ChatCompletionChunk::new(
                chunk_id.clone(),
                model_id.clone(),
                vec![ChatChunkChoice {
                    index: 0,
                    delta,
                    finish_reason: finish_reason.clone(),
                    logprobs: None,
                }],
            );

            let data = serde_json::to_string(&chunk_response).unwrap_or_default();
            yield Ok(Event::default().data(data));

            if chunk.is_finished {
                break;
            }
        }

        // Final [DONE] marker
        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(Sse::new(stream))
}

/// List models handler.
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<ModelsResponse> {
    let model = ModelInfo {
        id: state.model_id.clone(),
        object: "model".to_string(),
        created: chrono::Utc::now().timestamp(),
        owned_by: "rusteeze".to_string(),
    };

    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![model],
    })
}

/// Health check handler.
pub async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    let stats = state.engine.stats();

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        requests_processed: state.request_counter.load(std::sync::atomic::Ordering::Relaxed),
        waiting_requests: stats.num_waiting,
        running_requests: stats.num_running,
        gpu_memory_usage: stats.gpu_memory_usage,
    })
}

/// Build prompt from messages.
fn build_prompt(messages: &[ChatMessage]) -> String {
    // Simple concatenation - would use chat template in production
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build sampling params from request.
fn build_sampling_params(request: &ChatCompletionRequest) -> SamplingParams {
    SamplingParams {
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(1.0),
        top_k: -1,
        min_p: 0.0,
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        repetition_penalty: 1.0,
        seed: request.seed,
        logprobs: request.top_logprobs,
        ..Default::default()
    }
}
