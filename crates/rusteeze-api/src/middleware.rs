//! API middleware.

use axum::{
    body::Body,
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tracing::{debug, warn};

use crate::error::ApiError;

/// Authentication middleware.
pub async fn auth_middleware(
    request: Request,
    next: Next,
    api_key: String,
) -> Result<Response, ApiError> {
    // Get authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header[7..];
            if token == api_key {
                Ok(next.run(request).await)
            } else {
                Err(ApiError::Unauthorized("Invalid API key".to_string()))
            }
        }
        Some(_) => Err(ApiError::Unauthorized(
            "Invalid authorization header format. Use 'Bearer <key>'".to_string(),
        )),
        None => Err(ApiError::Unauthorized("Missing authorization header".to_string())),
    }
}

/// Rate limiter state.
pub struct RateLimiter {
    /// Requests per client.
    requests: RwLock<HashMap<String, Vec<Instant>>>,

    /// Max requests per window.
    max_requests: u32,

    /// Window duration.
    window: Duration,
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new(max_requests: u32, window: Duration) -> Self {
        Self {
            requests: RwLock::new(HashMap::new()),
            max_requests,
            window,
        }
    }

    /// Check if a client is rate limited.
    pub fn is_rate_limited(&self, client_id: &str) -> bool {
        let now = Instant::now();
        let mut requests = self.requests.write().unwrap();

        let client_requests = requests.entry(client_id.to_string()).or_default();

        // Remove old requests
        client_requests.retain(|&t| now.duration_since(t) < self.window);

        // Check limit
        if client_requests.len() >= self.max_requests as usize {
            true
        } else {
            client_requests.push(now);
            false
        }
    }

    /// Get remaining requests for a client.
    pub fn remaining(&self, client_id: &str) -> u32 {
        let now = Instant::now();
        let requests = self.requests.read().unwrap();

        if let Some(client_requests) = requests.get(client_id) {
            let valid_requests = client_requests
                .iter()
                .filter(|&&t| now.duration_since(t) < self.window)
                .count();
            self.max_requests.saturating_sub(valid_requests as u32)
        } else {
            self.max_requests
        }
    }

    /// Get reset time for a client.
    pub fn reset_time(&self, client_id: &str) -> Option<Instant> {
        let requests = self.requests.read().unwrap();

        requests.get(client_id).and_then(|r| r.first().copied())
    }
}

/// Rate limiting middleware.
pub async fn rate_limit_middleware(
    request: Request,
    next: Next,
    limiter: Arc<RateLimiter>,
) -> Result<Response, ApiError> {
    // Get client ID from header or IP
    let client_id = request
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .or_else(|| {
            request
                .headers()
                .get("X-Forwarded-For")
                .and_then(|v| v.to_str().ok())
                .map(String::from)
        })
        .unwrap_or_else(|| "anonymous".to_string());

    if limiter.is_rate_limited(&client_id) {
        warn!("Rate limited client: {}", client_id);
        return Err(ApiError::RateLimited(
            "Rate limit exceeded. Please retry later.".to_string(),
        ));
    }

    let remaining = limiter.remaining(&client_id);
    let reset_time = limiter
        .reset_time(&client_id)
        .map(|t| t + limiter.window)
        .unwrap_or_else(Instant::now);

    let mut response = next.run(request).await;

    // Add rate limit headers
    let headers = response.headers_mut();
    headers.insert(
        "X-RateLimit-Limit",
        limiter.max_requests.to_string().parse().unwrap(),
    );
    headers.insert("X-RateLimit-Remaining", remaining.to_string().parse().unwrap());
    headers.insert(
        "X-RateLimit-Reset",
        format!("{}", reset_time.elapsed().as_secs()).parse().unwrap(),
    );

    Ok(response)
}

/// Request logging middleware.
pub async fn logging_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = Instant::now();

    debug!("--> {} {}", method, uri);

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    if status.is_success() {
        debug!("<-- {} {} {:?} {}", method, uri, duration, status);
    } else {
        warn!("<-- {} {} {:?} {}", method, uri, duration, status);
    }

    response
}

/// Request ID middleware.
pub async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    // Generate or extract request ID
    let request_id = request
        .headers()
        .get("X-Request-ID")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Add to request extensions
    request.extensions_mut().insert(request_id.clone());

    // Run handler
    let mut response = next.run(request).await;

    // Add to response
    response
        .headers_mut()
        .insert("X-Request-ID", request_id.parse().unwrap());

    response
}

/// Timeout middleware configuration.
pub struct TimeoutConfig {
    pub timeout: Duration,
}

/// Timeout middleware.
pub async fn timeout_middleware(
    request: Request,
    next: Next,
    config: TimeoutConfig,
) -> Result<Response, ApiError> {
    match tokio::time::timeout(config.timeout, next.run(request)).await {
        Ok(response) => Ok(response),
        Err(_) => Err(ApiError::Timeout("Request timed out".to_string())),
    }
}

/// CORS configuration.
pub struct CorsConfig {
    pub allowed_origins: Vec<String>,
    pub allowed_methods: Vec<String>,
    pub allowed_headers: Vec<String>,
    pub max_age: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-Request-ID".to_string(),
            ],
            max_age: 86400,
        }
    }
}

/// Build CORS layer from config.
pub fn cors_layer(config: CorsConfig) -> tower_http::cors::CorsLayer {
    use tower_http::cors::{Any, CorsLayer};

    let mut layer = CorsLayer::new();

    // Origins
    if config.allowed_origins.contains(&"*".to_string()) {
        layer = layer.allow_origin(Any);
    } else {
        let origins: Vec<_> = config
            .allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        layer = layer.allow_origin(origins);
    }

    // Methods
    let methods: Vec<_> = config
        .allowed_methods
        .iter()
        .filter_map(|m| m.parse().ok())
        .collect();
    layer = layer.allow_methods(methods);

    // Headers
    let headers: Vec<_> = config
        .allowed_headers
        .iter()
        .filter_map(|h| h.parse().ok())
        .collect();
    layer = layer.allow_headers(headers);

    // Max age
    layer = layer.max_age(Duration::from_secs(config.max_age));

    layer
}

/// Compression configuration.
pub struct CompressionConfig {
    pub gzip: bool,
    pub deflate: bool,
    pub br: bool,
    pub zstd: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            gzip: true,
            deflate: true,
            br: true,
            zstd: true,
        }
    }
}

/// Build compression layer from config.
pub fn compression_layer(config: CompressionConfig) -> tower_http::compression::CompressionLayer {
    use tower_http::compression::CompressionLayer;

    let mut layer = CompressionLayer::new();

    if config.gzip {
        layer = layer.gzip(true);
    }
    if config.br {
        layer = layer.br(true);
    }
    if config.deflate {
        layer = layer.deflate(true);
    }
    if config.zstd {
        layer = layer.zstd(true);
    }

    layer
}
