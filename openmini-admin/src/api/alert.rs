use axum::{extract::{Path, Query, State}, Json};
use serde::Deserialize;
use sqlx::Row;

use crate::error::AppError;
use crate::AppState;
use crate::db::models::{AlertRule, AlertRecord};

#[derive(Deserialize)]
pub struct PageQuery { page: Option<u64>, page_size: Option<u64> }

fn row_to_alert_rule(row: sqlx::sqlite::SqliteRow) -> AlertRule {
    AlertRule {
        id: row.get("id"),
        name: row.get("name"),
        metric_name: row.get("metric_name"),
        condition: row.get("condition"),
        threshold: row.get("threshold"),
        duration_seconds: row.get("duration_seconds"),
        severity: row.get("severity"),
        channels: row.get("channels"),
        webhook_url: row.get("webhook_url"),
        is_enabled: row.get("is_enabled"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }
}

fn row_to_alert_record(row: sqlx::sqlite::SqliteRow) -> AlertRecord {
    AlertRecord {
        id: row.get("id"),
        rule_id: row.get("rule_id"),
        status: row.get("status"),
        severity: row.get("severity"),
        message: row.get("message"),
        value: row.get("value"),
        fired_at: row.get("fired_at"),
        acknowledged_at: row.get("acknowledged_at"),
        acknowledged_by: row.get("acknowledged_by"),
        resolved_at: row.get("resolved_at"),
        resolved_by: row.get("resolved_by"),
    }
}

pub async fn list_rules(State(state): State<AppState>) -> Result<Json<Vec<AlertRule>>, AppError> {
    let rows = sqlx::query(
        "SELECT * FROM alert_rules ORDER BY created_at DESC"
    )
    .fetch_all(&*state.pool)
    .await?;
    
    let rules: Vec<AlertRule> = rows.into_iter().map(row_to_alert_rule).collect();
    Ok(Json(rules))
}

pub async fn create_rule(State(state): State<AppState>, Json(req): Json<serde_json::Value>) -> Result<Json<AlertRule>, AppError> {
    let name = req.get("name").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let metric = req.get("metric_name").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();

    if name.is_empty() {
        return Err(AppError::BadRequest("规则名称不能为空".to_string()));
    }
    if metric.is_empty() {
        return Err(AppError::BadRequest("指标名称不能为空".to_string()));
    }

    let condition = req.get("condition").and_then(|v| v.as_str()).unwrap_or("gt");
    let threshold = req.get("threshold").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let duration = req.get("duration_seconds").and_then(|v| v.as_i64()).unwrap_or(300);
    let severity_str = req.get("severity").and_then(|v| v.as_str()).unwrap_or("warning");
    let channels = req.get("channels").and_then(|v| v.as_str()).unwrap_or("[]");

    let severity = match severity_str {
        "critical" => 0i32,
        "info" => 2i32,
        _ => 1i32,
    };

    let row = sqlx::query(
        "INSERT INTO alert_rules (name, metric_name, condition, threshold, duration_seconds, severity, channels) VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING *"
    )
    .bind(name)
    .bind(metric)
    .bind(condition)
    .bind(threshold)
    .bind(duration)
    .bind(severity)
    .bind(channels)
    .fetch_one(&*state.pool)
    .await?;
    Ok(Json(row_to_alert_rule(row)))
}

pub async fn update_rule(Path(id): Path<i64>, State(state): State<AppState>, Json(req): Json<serde_json::Value>) -> Result<Json<AlertRule>, AppError> {
    if let Some(name) = req.get("name").and_then(|v| v.as_str()) {
        sqlx::query("UPDATE alert_rules SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?")
            .bind(name).bind(id).execute(&*state.pool).await?;
    }
    let row = sqlx::query("SELECT * FROM alert_rules WHERE id = ?")
        .bind(id).fetch_one(&*state.pool).await?;
    Ok(Json(row_to_alert_rule(row)))
}

pub async fn delete_rule(Path(id): Path<i64>, State(state): State<AppState>) -> Result<axum::http::StatusCode, AppError> {
    sqlx::query("DELETE FROM alert_rules WHERE id = ?").bind(id).execute(&*state.pool).await?;
    Ok(axum::http::StatusCode::OK)
}

pub async fn toggle_rule(Path(id): Path<i64>, State(state): State<AppState>) -> Result<Json<AlertRule>, AppError> {
    let row = sqlx::query(
        "UPDATE alert_rules SET is_enabled = CASE WHEN is_enabled = 1 THEN 0 ELSE 1 END, updated_at = CURRENT_TIMESTAMP WHERE id = ? RETURNING *"
    ).bind(id).fetch_one(&*state.pool).await?;
    Ok(Json(row_to_alert_rule(row)))
}

pub async fn list_records(State(state): State<AppState>, Query(q): Query<PageQuery>) -> Result<Json<serde_json::Value>, AppError> {
    let page = q.page.unwrap_or(1) as i64;
    let size = q.page_size.unwrap_or(20) as i64;

    let rows = sqlx::query(
        "SELECT ar.* FROM alert_records ar ORDER BY ar.fired_at DESC LIMIT ? OFFSET ?"
    )
    .bind(size).bind((page - 1) * size)
    .fetch_all(&*state.pool)
    .await?;

    let records: Vec<AlertRecord> = rows.into_iter().map(row_to_alert_record).collect();

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM alert_records").fetch_one(&*state.pool).await?;

    Ok(Json(serde_json::json!({ "items": records, "total": count.0 as u64, "page": page as u64, "page_size": size as u64 })))
}

pub async fn acknowledge_alert(Path(id): Path<i64>, State(state): State<AppState>) -> Result<Json<AlertRecord>, AppError> {
    let row = sqlx::query(
        "UPDATE alert_records SET status = 1, acknowledged_at = CURRENT_TIMESTAMP WHERE id = ? AND status = 0 RETURNING *"
    ).bind(id).fetch_one(&*state.pool).await?;
    Ok(Json(row_to_alert_record(row)))
}

pub async fn resolve_alert(Path(id): Path<i64>, State(state): State<AppState>) -> Result<Json<AlertRecord>, AppError> {
    let row = sqlx::query(
        "UPDATE alert_records SET status = 2, resolved_at = CURRENT_TIMESTAMP WHERE id = ? AND status != 2 RETURNING *"
    ).bind(id).fetch_one(&*state.pool).await?;
    Ok(Json(row_to_alert_record(row)))
}

pub async fn get_summary(State(state): State<AppState>) -> Result<Json<serde_json::Value>, AppError> {
    let firing: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM alert_records WHERE status = 0").fetch_one(&*state.pool).await?;
    let acked: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM alert_records WHERE status = 1").fetch_one(&*state.pool).await?;
    let resolved: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM alert_records WHERE status = 2").fetch_one(&*state.pool).await?;

    Ok(Json(serde_json::json!({
        "firing": firing.0,
        "acknowledged": acked.0,
        "resolved": resolved.0,
        "total": firing.0 + acked.0 + resolved.0
    })))
}
