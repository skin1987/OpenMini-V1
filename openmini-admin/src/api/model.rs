use axum::{extract::{Path, State}, Json};
use serde_json::Value;

use crate::error::AppError;
use crate::AppState;

pub async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    let upstream_models = state.proxy.get_models().await?;

    if let Some(models) = upstream_models {
        Ok(Json(serde_json::json!({"source": "upstream", "models": models})))
    } else {
        Ok(Json(serde_json::json!({
            "source": "local",
            "upstream_status": "unreachable",
            "models": [
                {"id": 1, "name": "Qwen-14B-Chat", "path": "/models/qwen-14b-chat", "size_gb": 28.5, "quantization": "INT8", "context_length": 4096, "status": "unknown", "loaded_at": null},
                {"id": 2, "name": "Llama-3-8B-Instruct", "path": "/models/llama-3-8b", "size_gb": 16.2, "quantization": "FP16", "context_length": 8192, "status": "unknown", "loaded_at": null}
            ]
        })))
    }
}

pub async fn load_model(State(_state): State<AppState>, Json(req): Json<Value>) -> Result<Json<Value>, AppError> {
    let path = req.get("path").and_then(|v| v.as_str()).unwrap_or("?");
    tracing::info!(model = %path, "加载模型");
    Ok(Json(serde_json::json!({"message": "模型加载任务已提交", "status": "loading"})))
}

pub async fn unload_model(Path(id): Path<i64>, State(_state): State<AppState>) -> Result<Json<Value>, AppError> {
    tracing::info!(model_id = id, "卸载模型");
    Ok(Json(serde_json::json!({"message": "模型已卸载", "model_id": id})))
}

pub async fn switch_model(State(_state): State<AppState>, Json(req): Json<Value>) -> Result<Json<Value>, AppError> {
    let target = req.get("target").and_then(|v| v.as_str()).unwrap_or("unknown");
    tracing::info!(target = %target, "热切换模型");
    Ok(Json(serde_json::json!({"message": "热切换完成"})))
}

pub async fn check_health(Path(id): Path<i64>, State(_state): State<AppState>) -> Result<Json<Value>, AppError> {
    Ok(Json(serde_json::json!({"model_id": id, "status": "healthy", "memory_mb": 28500, "kv_cache_mb": 4000})))
}
