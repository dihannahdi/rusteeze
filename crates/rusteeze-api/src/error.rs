//! API error types.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

/// API error.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Rate limited: {0}")]
    RateLimited(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Timeout: {0}")]
    Timeout(String),
}

impl ApiError {
    /// Get HTTP status code.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::RateLimited(_) => StatusCode::TOO_MANY_REQUESTS,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
        }
    }

    /// Get error type string.
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "invalid_request_error",
            Self::Unauthorized(_) => "authentication_error",
            Self::NotFound(_) => "not_found_error",
            Self::RateLimited(_) => "rate_limit_error",
            Self::Internal(_) => "server_error",
            Self::ServiceUnavailable(_) => "service_unavailable",
            Self::Timeout(_) => "timeout_error",
        }
    }
}

/// OpenAI-compatible error response.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// Error details.
    pub error: ErrorDetail,
}

/// Error detail.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetail {
    /// Error message.
    pub message: String,

    /// Error type.
    #[serde(rename = "type")]
    pub error_type: String,

    /// Parameter that caused error (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,

    /// Error code (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_type = self.error_type().to_string();
        let message = self.to_string();

        let body = ErrorResponse {
            error: ErrorDetail {
                message,
                error_type,
                param: None,
                code: None,
            },
        };

        (status, Json(body)).into_response()
    }
}

impl From<rusteeze_engine::EngineError> for ApiError {
    fn from(err: rusteeze_engine::EngineError) -> Self {
        match err {
            rusteeze_engine::EngineError::Timeout => Self::Timeout("Request timed out".to_string()),
            rusteeze_engine::EngineError::Cancelled => Self::BadRequest("Request cancelled".to_string()),
            rusteeze_engine::EngineError::Shutdown => Self::ServiceUnavailable("Engine shutdown".to_string()),
            _ => Self::Internal(err.to_string()),
        }
    }
}
