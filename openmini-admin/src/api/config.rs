use axum::{extract::{Query, State}, Json};
use serde::Deserialize;
use sqlx::Row;

use crate::error::AppError;
use crate::AppState;
use crate::db::models::ConfigHistory;

#[derive(Deserialize)]
pub struct PageQuery { page: Option<u64>, page_size: Option<u64> }

fn row_to_config_history(row: sqlx::sqlite::SqliteRow) -> ConfigHistory {
    ConfigHistory {
        id: row.get("id"),
        changed_by: row.get("changed_by"),
        section: row.get("section"),
        old_value: row.get("old_value"),
        new_value: row.get("new_value"),
        change_reason: row.get("change_reason"),
        created_at: row.get("created_at"),
    }
}

pub async fn get_config(State(_state): State<AppState>) -> Result<Json<serde_json::Value>, AppError> {
    Ok(Json(serde_json::json!({
        "server": {"host": "0.0.0.0", "port": 7070, "max_connections": 1000},
        "thread_pool": {"size": 4},
        "memory": {"max_memory_gb": 32, "model_memory_gb": 24, "cache_memory_gb": 8},
        "model": {"path": "/models/openmini-v1", "quantization": "Q4_K_M", "context_length": 4096},
        "worker": {"count": 3, "restart_on_failure": true},
        "grpc": {"max_message_size_mb": 16}
    })))
}

pub async fn update_config(State(_state): State<AppState>, Json(req): Json<serde_json::Value>) -> Result<Json<serde_json::Value>, AppError> {
    tracing::info!(config = %req, "更新配置");
    Ok(Json(serde_json::json!({"message": "配置已更新"})))
}

pub async fn reload_config(State(_state): State<AppState>) -> Result<axum::http::StatusCode, AppError> {
    tracing::info!("重载配置");
    Ok(axum::http::StatusCode::OK)
}

pub async fn get_history(State(state): State<AppState>, Query(q): Query<PageQuery>) -> Result<Json<serde_json::Value>, AppError> {
    let page = q.page.unwrap_or(1) as i64;
    let size = q.page_size.unwrap_or(20) as i64;

    let rows = sqlx::query(
        "SELECT * FROM config_history ORDER BY created_at DESC LIMIT ? OFFSET ?"
    )
    .bind(size).bind((page - 1) * size)
    .fetch_all(&*state.pool)
    .await?;

    let history: Vec<ConfigHistory> = rows.into_iter().map(row_to_config_history).collect();

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM config_history").fetch_one(&*state.pool).await?;

    Ok(Json(serde_json::json!({ "items": history, "total": count.0 as u64, "page": page as u64, "page_size": size as u64 })))
}
