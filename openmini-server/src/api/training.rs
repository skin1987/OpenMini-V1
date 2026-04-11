//! Training RESTful API

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::training::trainer::{Trainer, TrainingConfig, TrainingState, StepMetrics, TrainingMode, SFTConfig};

#[derive(Clone)]
pub struct TrainingAppState {
    pub trainer: Arc<Mutex<Option<Trainer>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StartTrainingRequest {
    pub config: TrainingConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StartTrainingResponse {
    pub status: String,
    pub message: String,
    pub training_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResponse {
    pub status: String,
    pub state: Option<TrainingState>,
    pub latest_metrics: Option<Vec<StepMetrics>>,
}

pub fn training_routes() -> Router<TrainingAppState> {
    Router::new()
        .route("/api/v1/training/start", post(start_training))
        .route("/api/v1/training/pause", post(pause_training))
        .route("/api/v1/training/resume", post(resume_training))
        .route("/api/v1/training/stop", post(stop_training))
        .route("/api/v1/training/status", get(get_status))
        .route("/api/v1/training/metrics", get(get_metrics))
}

async fn start_training(
    State(state): State<TrainingAppState>,
    Json(req): Json<StartTrainingRequest>,
) -> Result<Json<StartTrainingResponse>, StatusCode> {
    let mut trainer_guard = state.trainer.lock().await;

    if trainer_guard.is_some() {
        return Err(StatusCode::CONFLICT);
    }

    match Trainer::new(req.config) {
        Ok(trainer) => {
            *trainer_guard = Some(trainer);
            Ok(Json(StartTrainingResponse {
                status: "started".to_string(),
                message: "Training started successfully".to_string(),
                training_id: Some(uuid::Uuid::new_v4().to_string()),
            }))
        }
        Err(e) => {
            eprintln!("Failed to create trainer: {}", e);
            Err(StatusCode::BAD_REQUEST)
        }
    }
}

async fn pause_training(
    State(state): State<TrainingAppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut trainer_guard = state.trainer.lock().await;

    match trainer_guard.as_mut() {
        Some(trainer) => {
            trainer.pause();
            Ok(Json(serde_json::json!({"status": "paused"})))
        }
        None => Err(StatusCode::CONFLICT),
    }
}

async fn resume_training(
    State(state): State<TrainingAppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut trainer_guard = state.trainer.lock().await;

    match trainer_guard.as_mut() {
        Some(trainer) => {
            trainer.resume();
            Ok(Json(serde_json::json!({"status": "resumed"})))
        }
        None => Err(StatusCode::CONFLICT),
    }
}

async fn stop_training(
    State(state): State<TrainingAppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut trainer_guard = state.trainer.lock().await;

    match trainer_guard.as_mut() {
        Some(trainer) => {
            trainer.stop();
            let summary = serde_json::json!({
                "status": "stopped",
                "message": "Training stopped gracefully"
            });
            *trainer_guard = None;
            Ok(Json(summary))
        }
        None => Err(StatusCode::CONFLICT),
    }
}

async fn get_status(
    State(state): State<TrainingAppState>,
) -> Json<StatusResponse> {
    let trainer_guard = state.trainer.lock().await;

    match trainer_guard.as_ref() {
        Some(trainer) => {
            Json(StatusResponse {
                status: "running".to_string(),
                state: Some(trainer.state().clone()),
                latest_metrics: Some(trainer.recent_metrics(10).to_vec()),
            })
        }
        None => {
            Json(StatusResponse {
                status: "not_started".to_string(),
                state: None,
                latest_metrics: None,
            })
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct MetricsParams {
    #[serde(default = "default_last_n")]
    last_n: usize,
}

fn default_last_n() -> usize { 100 }

async fn get_metrics(
    State(state): State<TrainingAppState>,
    Query(params): Query<MetricsParams>,
) -> Json<serde_json::Value> {
    let trainer_guard = state.trainer.lock().await;

    match trainer_guard.as_ref() {
        Some(trainer) => {
            let metrics = trainer.recent_metrics(params.last_n);
            Json(serde_json::json!({
                "metrics": metrics,
                "count": metrics.len()
            }))
        }
        None => {
            Json(serde_json::json!({"metrics": [], "count": 0}))
        }
    }
}
