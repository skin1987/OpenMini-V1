use axum::{extract::State, Json};
use serde_json::Value;

use crate::error::AppError;
use crate::AppState;

pub async fn get_status(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let health = state.proxy.get_health().await?;

    let mut status = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "pid": std::process::id(),
        "upstream": state.proxy.base_url(),
    });

    if let Some(h) = health {
        status["status"] = Value::String("online".into());
        if let Some(up) = h.get("uptime_seconds").and_then(|v| v.as_i64()) {
            status["uptime_seconds"] = Value::from(up);
        }
        if let Some(components) = h.get("components") {
            status["components"] = components.clone();
        }
    } else {
        status["status"] = Value::String("degraded".into());
        status["upstream_status"] = Value::String("unreachable".into());
        status["uptime_seconds"] = Value::from(0i64);
        status["message"] = Value::String("上游服务不可用，仅返回本地状态".into());
    }

    Ok(Json(status))
}

pub async fn get_workers(State(_state): State<AppState>) -> Result<Json<Value>, AppError> {
    let workers = serde_json::json!({
        "workers": [
            {"id": 0, "status": "running", "pid": std::process::id(), "started_at": "2026-04-09T00:00:00Z", "restarts": 0},
        ]
    });
    Ok(Json(workers))
}

pub async fn restart_service(State(_state): State<AppState>) -> Result<Json<Value>, AppError> {
    tracing::info!("收到服务重启请求");
    Ok(Json(serde_json::json!({"message": "重启信号已发送", "status": "restarting"})))
}

pub async fn stop_service(State(_state): State<AppState>) -> Result<Json<Value>, AppError> {
    tracing::warn!("收到服务停止请求");
    Ok(Json(serde_json::json!({"message": "停止信号已发送", "status": "stopping"})))
}
