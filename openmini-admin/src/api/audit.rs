use axum::{
    extract::{Query, State},
    Json,
};
use serde::Deserialize;
use sqlx::Row;

use crate::db::models::AuditLog;
use crate::error::AppError;
use crate::AppState;

#[derive(Deserialize)]
pub struct PageQuery {
    pub page: Option<u64>,
    pub page_size: Option<u64>,
    pub action: Option<String>,
}

fn row_to_audit_log(row: sqlx::sqlite::SqliteRow) -> AuditLog {
    AuditLog {
        id: row.get("id"),
        user_id: row.get("user_id"),
        action: row.get("action"),
        resource_type: row.get("resource_type"),
        resource_id: row.get("resource_id"),
        detail: row.get("detail"),
        ip_address: row.get("ip_address"),
        user_agent: row.get("user_agent"),
        created_at: row.get("created_at"),
    }
}

pub async fn list_logs(
    State(state): State<AppState>,
    Query(q): Query<PageQuery>,
) -> Result<Json<serde_json::Value>, AppError> {
    let page = q.page.unwrap_or(1) as i64;
    let size = q.page_size.unwrap_or(20) as i64;

    let logs: Vec<AuditLog> = if let Some(action) = &q.action {
        let rows = sqlx::query(
            "SELECT * FROM audit_logs WHERE action LIKE ? ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        .bind(format!("%{}%", action))
        .bind(size).bind((page - 1) * size)
        .fetch_all(&*state.pool).await?;
        rows.into_iter().map(row_to_audit_log).collect()
    } else {
        let rows =
            sqlx::query("SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ? OFFSET ?")
                .bind(size)
                .bind((page - 1) * size)
                .fetch_all(&*state.pool)
                .await?;
        rows.into_iter().map(row_to_audit_log).collect()
    };

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM audit_logs")
        .fetch_one(&*state.pool)
        .await?;

    Ok(Json(
        serde_json::json!({ "items": logs, "total": count.0 as u64, "page": page as u64, "page_size": size as u64 }),
    ))
}

pub async fn get_stats(State(state): State<AppState>) -> Result<Json<serde_json::Value>, AppError> {
    let stats =
        sqlx::query_as::<_, (i64, i64)>("SELECT COUNT(*), COUNT(DISTINCT user_id) FROM audit_logs")
            .fetch_one(&*state.pool)
            .await?;

    Ok(Json(serde_json::json!({
        "total_operations": stats.0,
        "active_users": stats.1
    })))
}
