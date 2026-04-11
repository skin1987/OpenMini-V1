use thiserror::Error;
use axum::response::{IntoResponse, Response};
use axum::http::StatusCode;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("未授权")]
    Unauthorized,
    #[error("权限不足")]
    #[allow(dead_code)]
    Forbidden,
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    BadRequest(String),
    #[error("内部错误: {0}")]
    Internal(#[from] anyhow::Error),
    #[error("数据库错误: {0}")]
    Database(#[from] sqlx::Error),
    #[error("JWT 错误: {0}")]
    Jwt(#[from] jsonwebtoken::errors::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self {
            AppError::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized"),
            AppError::Forbidden => (StatusCode::FORBIDDEN, "forbidden"),
            AppError::NotFound(_) => (StatusCode::NOT_FOUND, "not_found"),
            AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "bad_request"),
            AppError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
            AppError::Database(_) => (StatusCode::INTERNAL_SERVER_ERROR, "database_error"),
            AppError::Jwt(_) => (StatusCode::UNAUTHORIZED, "invalid_token"),
        };

        (
            status,
            axum::Json(serde_json::json!({
                "error": error_type,
                "message": self.to_string()
            })),
        )
            .into_response()
    }
}
