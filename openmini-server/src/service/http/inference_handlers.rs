//! 高性能推理 API 处理器
//!
//! 集成 HighPerformancePipeline 到 HTTP REST API。

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    response::{Json, IntoResponse},
};
use ndarray::Array2;
use serde::Serialize;
use tracing::{debug, error, info};
use uuid::Uuid;

use super::inference_types::*;
use super::handlers::AppState;
use super::types::ApiError;
use crate::model::inference::high_performance_pipeline::{
    HighPerformancePipeline, HighPerfPipelineConfig,
};

// ============================================================================
// 推理端点处理器
// ============================================================================

/// 执行单次推理计算
///
/// POST /api/v1/inference/compute
///
/// 直接调用 HighPerformancePipeline.forward()
pub async fn inference_compute(
    State(state): State<AppState>,
    Json(req): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, ApiError> {
    let start = Instant::now();
    
    // 获取推理状态
    let inference_state = state.inference.as_ref().ok_or(ApiError {
        error_type: "service_unavailable".to_string(),
        message: "Inference service not available".to_string(),
        status_code: 503,
    })?;
    
    debug!(
        query_rows = req.query.len(),
        key_rows = req.key.len(),
        "收到推理计算请求"
    );
    
    // 验证输入
    if req.query.is_empty() || req.key.is_empty() || req.value.is_empty() {
        return Err(ApiError {
            error_type: "invalid_request".to_string(),
            message: "Query, Key, and Value tensors cannot be empty".to_string(),
            status_code: 400,
        });
    }
    
    // 转换为 ndarray Array2
    let q = match vec_to_array2(&req.query) {
        Ok(arr) => arr,
        Err(e) => {
            return Err(ApiError {
                error_type: "invalid_request".to_string(),
                message: format!("Invalid query tensor format: {}", e),
                status_code: 400,
            });
        }
    };
    
    let k = match vec_to_array2(&req.key) {
        Ok(arr) => arr,
        Err(e) => {
            return Err(ApiError {
                error_type: "invalid_request".to_string(),
                message: format!("Invalid key tensor format: {}", e),
                status_code: 400,
            });
        }
    };
    
    let v = match vec_to_array2(&req.value) {
        Ok(arr) => arr,
        Err(e) => {
            return Err(ApiError {
                error_type: "invalid_request".to_string(),
                message: format!("Invalid value tensor format: {}", e),
                status_code: 400,
            });
        }
    };
    
    // 执行推理
    let mut pipeline = inference_state.pipeline.write().await;
    let output = match pipeline.forward(&q, &k, &v) {
        Ok(result) => result,
        Err(e) => {
            error!(error = %e, "推理计算失败");
            return Err(ApiError {
                error_type: "inference_error".to_string(),
                message: format!("Inference computation failed: {}", e),
                status_code: 500,
            });
        }
    };
    
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let stats = pipeline.stats();
    
    // 转换输出为 Vec<Vec<f32>>
    let output_vec = array2_to_vec(&output);
    let output_shape = vec![output.nrows(), output.ncols()];
    
    Ok(Json(InferenceResponse {
        id: Uuid::new_v4().to_string(),
        output: output_vec,
        output_shape,
        strategy: stats.strategy.to_string(),
        inference_time_ms: elapsed,
        stats: Some(InferenceStats {
            total_tokens: stats.generated_tokens,
            tokens_per_second: stats.tokens_per_second,
            kv_cache_utilization: stats.kv_cache_utilization,
            blocks_used: stats.blocks_used,
        }),
    }))
}

/// 批量推理计算
///
/// POST /api/v1/inference/batch
pub async fn inference_batch(
    State(state): State<AppState>,
    Json(req): Json<BatchInferenceRequest>,
) -> Result<Json<BatchInferenceResponse>, ApiError> {
    let start = Instant::now();
    
    let inference_state = state.inference.as_ref().ok_or(ApiError {
        error_type: "service_unavailable".to_string(),
        message: "Inference service not available".to_string(),
        status_code: 503,
    })?;
    
    debug!(
        batch_size = req.requests.len(),
        "收到批量推理请求"
    );
    
    if req.requests.is_empty() {
        return Err(ApiError {
            error_type: "invalid_request".to_string(),
            message: "Requests list cannot be empty".to_string(),
            status_code: 400,
        });
    }
    
    let mut outputs = Vec::with_capacity(req.requests.len());
    
    for (idx, single_req) in req.requests.iter().enumerate() {
        let q = match vec_to_array2(&single_req.query) {
            Ok(arr) => arr,
            Err(e) => {
                return Err(ApiError {
                    error_type: "invalid_request".to_string(),
                    message: format!("Invalid query tensor in request {}: {}", idx, e),
                    status_code: 400,
                });
            }
        };
        
        let k = match vec_to_array2(&single_req.key) {
            Ok(arr) => arr,
            Err(e) => {
                return Err(ApiError {
                    error_type: "invalid_request".to_string(),
                    message: format!("Invalid key tensor in request {}: {}", idx, e),
                    status_code: 400,
                });
            }
        };
        
        let v = match vec_to_array2(&single_req.value) {
            Ok(arr) => arr,
            Err(e) => {
                return Err(ApiError {
                    error_type: "invalid_request".to_string(),
                    message: format!("Invalid value tensor in request {}: {}", idx, e),
                    status_code: 400,
                });
            }
        };
        
        let mut pipeline = inference_state.pipeline.write().await;
        let output = match pipeline.forward(&q, &k, &v) {
            Ok(result) => result,
            Err(e) => {
                return Err(ApiError {
                    error_type: "inference_error".to_string(),
                    message: format!("Inference failed for request {}: {}", idx, e),
                    status_code: 500,
                });
            }
        };
        
        let stats = pipeline.stats();
        
        outputs.push(InferenceResponse {
            id: Uuid::new_v4().to_string(),
            output: array2_to_vec(&output),
            output_shape: vec![output.nrows(), output.ncols()],
            strategy: stats.strategy.to_string(),
            inference_time_ms: 0.0,
            stats: Some(InferenceStats {
                total_tokens: stats.generated_tokens,
                tokens_per_second: stats.tokens_per_second,
                kv_cache_utilization: stats.kv_cache_utilization,
                blocks_used: stats.blocks_used,
            }),
        });
    }
    
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let total_tokens: usize = outputs.iter()
        .filter_map(|o| o.stats.as_ref())
        .map(|s| s.total_tokens)
        .sum();
    
    let avg_tps = if elapsed > 0.0 {
        (total_tokens as f32) / (elapsed as f32 / 1000.0)
    } else {
        0.0
    };
    
    Ok(Json(BatchInferenceResponse {
        id: Uuid::new_v4().to_string(),
        outputs,
        total_time_ms: elapsed,
        avg_tokens_per_second: avg_tps,
    }))
}

/// 获取 Pipeline 性能统计
///
/// GET /api/v1/inference/stats
pub async fn inference_stats(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    match state.inference {
        Some(ref inf) => {
            let pipeline = inf.pipeline.read().await;
            let stats = pipeline.stats();
            let config = pipeline.config();
            let kv_info = pipeline.kv_cache_info();
            
            Ok(Json(serde_json::json!({
                "current_strategy": stats.strategy.to_string(),
                "total_processed_tokens": pipeline.total_processed_tokens,
                "attention_time_ms": stats.attention_time_ms,
                "generated_tokens": stats.generated_tokens,
                "tokens_per_second": stats.tokens_per_second,
                "kv_cache": {
                    "available_blocks": kv_info.map_or(0, |(a, _, _)| a),
                    "allocated_blocks": kv_info.map_or(0, |(_, b, _)| b),
                    "utilization": kv_info.map_or(0.0, |(_, _, u)| u),
                    "block_size": config.kv_block_size,
                    "max_blocks": config.max_kv_blocks
                }
            })))
        }
        None => Ok(Json(serde_json::json!({
            "error": "Inference service not available"
        }))),
    }
}

/// 获取 Pipeline 配置信息
///
/// GET /api/v1/inference/config
pub async fn inference_config(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    match state.inference {
        Some(ref inf) => {
            let pipeline = inf.pipeline.read().await;
            let config = pipeline.config();
            let kv_info = pipeline.kv_cache_info();
            
            Ok(Json(serde_json::json!({
                "max_seq_len": config.max_seq_len,
                "num_heads": config.num_heads,
                "head_dim": config.head_dim,
                "num_kv_heads": config.num_kv_heads,
                "num_layers": config.num_layers,
                "enable_fa3": config.enable_fa3,
                "enable_mla": config.enable_mla,
                "enable_streaming": config.enable_streaming,
                "streaming_threshold": config.streaming_threshold,
                "kv_cache": {
                    "available_blocks": kv_info.map_or(0, |(a, _, _)| a),
                    "allocated_blocks": kv_info.map_or(0, |(_, b, _)| b),
                    "utilization": kv_info.map_or(0.0, |(_, _, u)| u),
                    "block_size": config.kv_block_size,
                    "max_blocks": config.max_kv_blocks
                }
            })))
        }
        None => Err(ApiError {
            error_type: "service_unavailable".to_string(),
            message: "Inference service not available".to_string(),
            status_code: 503,
        }),
    }
}

/// 重置 Pipeline 状态
///
/// POST /api/v1/inference/reset
pub async fn inference_reset(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    match state.inference {
        Some(ref inf) => {
            let mut pipeline = inf.pipeline.write().await;
            pipeline.reset();
            info!("Pipeline 状态已重置");
        }
        None => {
            info!("无可用 Pipeline 实例");
        }
    }
    
    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Pipeline state reset successfully"
    })))
}

// ============================================================================
// 辅助类型和函数
// ============================================================================

/// 推理服务状态（包含Pipeline实例）
#[derive(Clone)]
pub struct InferenceState {
    /// 高性能推理 Pipeline
    pub pipeline: Arc<tokio::sync::RwLock<HighPerformancePipeline>>,
}

impl InferenceState {
    pub fn new(config: HighPerfPipelineConfig) -> Self {
        let pipeline = HighPerformancePipeline::new(config.clone())
            .expect("Failed to create HighPerformancePipeline");
        
        info!(
            num_heads = config.num_heads,
            head_dim = config.head_dim,
            enable_fa3 = config.enable_fa3,
            "高性能推理 Pipeline 初始化成功"
        );
        
        Self {
            pipeline: Arc::new(tokio::sync::RwLock::new(pipeline)),
        }
    }
}

impl Default for InferenceState {
    fn default() -> Self {
        Self::new(HighPerfPipelineConfig::for_7b_model())
    }
}

/// Pipeline 统计响应
#[derive(Debug, Clone, Serialize)]
pub struct PipelineStatsResponse {
    pub current_strategy: String,
    pub total_processed_tokens: usize,
    pub attention_time_ms: f64,
    pub generated_tokens: usize,
    pub tokens_per_second: f32,
    pub kv_cache: super::inference_types::KvCacheInfo,
}

/// 重置响应
#[derive(Debug, Clone, Serialize)]
pub struct ResetResponse {
    pub success: bool,
    pub message: String,
}

/// 将 Vec<Vec<f32>> 转换为 Array2<f32>
fn vec_to_array2(data: &[Vec<f32>]) -> Result<Array2<f32>, String> {
    if data.is_empty() {
        return Err("Empty data".to_string());
    }
    
    let rows = data.len();
    let cols = data[0].len();
    
    if cols == 0 {
        return Err("Zero columns".to_string());
    }
    
    for (i, row) in data.iter().enumerate() {
        if row.len() != cols {
            return Err(format!(
                "Row {} has length {}, expected {}",
                i,
                row.len(),
                cols
            ));
        }
    }
    
    let mut flat_data = Vec::with_capacity(rows * cols);
    for row in data {
        flat_data.extend_from_slice(row);
    }
    
    Array2::from_shape_vec((rows, cols), flat_data)
        .map_err(|e| format!("Failed to create array: {}", e))
}

/// 将 Array2<f32> 转换为 Vec<Vec<f32>>
fn array2_to_vec(arr: &Array2<f32>) -> Vec<Vec<f32>> {
    (0..arr.nrows())
        .map(|i| arr.row(i).to_vec())
        .collect()
}
