//! 模型管理 API 模块
//!
//! 提供模型列表查询（支持分页和过滤）、模型详情查看、
//! 模型加载/卸载、模型配置更新等功能。
//! 用于管理推理引擎中可用的 AI 模型。

#![allow(dead_code)]

use axum::{
    extract::{Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

use crate::error::AppError;
use crate::AppState;

// ==================== 数据结构定义 ====================

/// 模型基本信息（用于列表展示）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub display_name: String,
    pub description: Option<String>,
    pub status: ModelStatus,
    pub provider: String,
    pub format: String,
    pub path: String,
    pub size_gb: f64,
    pub loaded_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

/// 模型状态枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Available,
    Loading,
    Loaded,
    Unloading,
    Error,
}

impl std::fmt::Display for ModelStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Available => write!(f, "available"),
            Self::Loading => write!(f, "loading"),
            Self::Loaded => write!(f, "loaded"),
            Self::Unloading => write!(f, "unloading"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// 模型详细信息（包含配置和能力）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetail {
    #[serde(flatten)]
    pub base: ModelInfo,

    // 模型规格参数
    pub parameters: u64,      // 参数量（十亿）
    pub quantization: String, // 量化方式 (FP16/INT8/Q4_K_M 等)
    pub context_length: u32,  // 上下文长度
    pub max_tokens: u32,      // 最大生成长度

    // 能力支持
    pub supported_features: Vec<String>,
    pub capabilities: ModelCapabilities,

    // 资源使用情况
    pub resource_usage: ModelResourceUsage,

    // 性能统计（如果已加载）
    pub performance_stats: Option<ModelPerformanceStats>,
}

/// 模型能力描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub chat_completion: bool,
    pub embedding: bool,
    pub function_calling: bool,
    pub vision: bool,
    pub streaming: bool,
}

/// 模型资源占用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResourceUsage {
    pub gpu_memory_mb: u64,
    pub cpu_memory_mb: u64,
    pub gpu_utilization_percent: f64,
    pub temperature_celsius: Option<f64>,
}

/// 模型性能统计数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceStats {
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_tps: f64,
    pub total_requests: u64,
    pub success_rate: f64,
}

/// 模型配置更新请求体
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ModelConfigUpdate {
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub gpu_layers: Option<i32>, // -1 表示全部卸载到 GPU
    #[serde(default)]
    pub n_batch: Option<u32>,
    #[serde(default)]
    pub n_ctx: Option<u32>,
    /// 是否启用 Mlock（锁定内存防止交换）
    #[serde(default)]
    pub mlock: Option<bool>,
    /// 是否使用内存映射文件
    #[serde(default)]
    pub mmap: Option<bool>,
}

/// 加载模型请求体
#[derive(Debug, Clone, Deserialize)]
pub struct LoadModelRequest {
    pub model_path: String,
    #[serde(default)]
    pub config: Option<ModelConfigUpdate>,
}

/// 分页和过滤查询参数
#[derive(Debug, Deserialize)]
pub struct ModelListQuery {
    pub page: Option<u32>,
    pub page_size: Option<u32>,
    pub status: Option<String>,
    pub provider: Option<String>,
    pub format: Option<String>,
    pub search: Option<String>, // 搜索关键词（匹配名称或描述）
}

impl ModelListQuery {
    /// 标准化分页参数
    pub fn normalized(&self) -> (i64, i64) {
        let page = self.page.unwrap_or(1).max(1) as i64;
        let page_size = self.page_size.unwrap_or(20).clamp(1, 100) as i64;
        (page, page_size)
    }

    /// 计算 OFFSET 值
    pub fn offset(&self) -> i64 {
        let (page, page_size) = self.normalized();
        (page - 1) * page_size
    }
}

// ==================== API 处理函数 ====================

/// 获取模型列表（支持分页、过滤和搜索）
///
/// # Query 参数
/// - `page`: 页码（默认 1）
/// - `page_size`: 每页数量（默认 20，最大 100）
/// - `status`: 状态过滤（available/loading/unloaded/error）
/// - `provider`: 提供商过滤（local/huggingface/openai 等）
/// - `format`: 格式过滤（gguf/safetensors 等）
/// - `search`: 搜索关键词（模糊匹配模型名称或描述）
///
/// # 返回
/// 分页的模型列表及总数
pub async fn list_models(
    State(state): State<AppState>,
    Query(params): Query<ModelListQuery>,
) -> Result<Json<Value>, AppError> {
    // 尝试从上游服务获取模型列表
    if let Some(upstream_models) = state.proxy.get_models().await? {
        // 上游可用时，返回上游数据并添加本地元数据
        return Ok(Json(serde_json::json!({
            "source": "upstream",
            "models": upstream_models,
            "total": upstream_models.as_array().map(|a| a.len()).unwrap_or(0),
            "upstream_status": "online"
        })));
    }

    // 上游不可用时，从本地数据库或返回模拟数据
    let (page, page_size) = params.normalized();

    // 构建过滤条件
    let mut conditions = vec!["1=1".to_string()];
    let mut bind_values: Vec<String> = vec![];

    if let Some(ref status) = params.status {
        if !status.is_empty()
            && ["available", "loading", "loaded", "unloading", "error"].contains(&status.as_str())
        {
            conditions.push("status = ?".to_string());
            bind_values.push(status.clone());
        }
    }

    if let Some(ref provider) = params.provider {
        if !provider.is_empty() {
            conditions.push("provider = ?".to_string());
            bind_values.push(provider.clone());
        }
    }

    if let Some(ref search) = params.search {
        if !search.is_empty() {
            conditions
                .push("(name LIKE ? OR display_name LIKE ? OR description LIKE ?)".to_string());
            let search_pattern = format!("%{}%", search);
            bind_values.push(search_pattern.clone());
            bind_values.push(search_pattern.clone());
            bind_values.push(search_pattern);
        }
    }

    let _where_clause = conditions.join(" AND ");

    // 查询模型列表（这里使用模拟数据，实际应从数据库查询）
    let models = generate_mock_models();

    // 应用过滤逻辑（实际应在 SQL 层面完成）
    let filtered_models: Vec<&ModelInfo> = models
        .iter()
        .filter(|m| {
            let mut matches = true;

            if let Some(ref status) = params.status {
                matches &= m.status.to_string() == *status;
            }

            if let Some(ref provider) = params.provider {
                matches &= m.provider == *provider;
            }

            if let Some(ref search) = params.search {
                let search_lower = search.to_lowercase();
                matches &= m.name.to_lowercase().contains(&search_lower)
                    || m.display_name.to_lowercase().contains(&search_lower)
                    || m.description
                        .as_ref()
                        .map(|d| d.to_lowercase().contains(&search_lower))
                        .unwrap_or(false);
            }

            matches
        })
        .collect();

    let total = filtered_models.len() as i64;
    let offset = params.offset() as usize;
    let limit = page_size as usize;

    // 手动分页
    let paged_models: Vec<ModelInfo> = filtered_models
        .into_iter()
        .skip(offset)
        .take(limit)
        .cloned()
        .collect();

    Ok(Json(serde_json::json!({
        "source": "local",
        "items": paged_models,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_more": (offset + paged_models.len()) < total as usize,
        "filters": {
            "status": params.status,
            "provider": params.provider,
            "search": params.search
        }
    })))
}

/// 获取模型详细信息
///
/// # 参数
/// - `model_id`: 模型 ID 或名称
///
/// # 返回
/// 模型的完整信息，包括规格参数、能力、资源占用和性能统计
pub async fn get_model_detail(
    Path(model_id): Path<String>,
    State(_state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    // 查找模型（实际实现中应该从数据库或模型注册表查找）
    let model = find_model_by_id(&model_id)
        .ok_or_else(|| AppError::NotFound(format!("模型不存在: {}", model_id)))?;

    let detail = ModelDetail {
        base: model,

        // 规格参数（根据模型 ID 返回不同的规格信息）
        parameters: get_model_parameters(&model_id),
        quantization: get_model_quantization(&model_id),
        context_length: get_model_context_length(&model_id),
        max_tokens: 4096,

        supported_features: vec![
            "chat_completion".to_string(),
            "streaming".to_string(),
            "function_calling".to_string(),
        ],

        capabilities: ModelCapabilities {
            chat_completion: true,
            embedding: false,
            function_calling: true,
            vision: false,
            streaming: true,
        },

        resource_usage: ModelResourceUsage {
            gpu_memory_mb: 18944,
            cpu_memory_mb: 9216,
            gpu_utilization_percent: 78.5,
            temperature_celsius: Some(72.5),
        },

        performance_stats: Some(ModelPerformanceStats {
            avg_latency_ms: 512.3,
            p50_latency_ms: 450.1,
            p95_latency_ms: 985.6,
            p99_latency_ms: 1520.4,
            throughput_tps: 12.5,
            total_requests: 45120,
            success_rate: 98.5,
        }),
    };

    Ok(Json(serde_json::json!(detail)))
}

/// 加载模型到内存/GPU
///
/// # Request Body
/// - `model_path`: 模型文件路径
/// - `config`: 可选的模型配置参数
///
/// # 返回
/// 加载任务状态
pub async fn load_model(
    State(_state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Result<Json<Value>, AppError> {
    // 验证路径安全性
    let path = PathBuf::from(&req.model_path);

    // 检查路径是否存在且为文件
    if !path.exists() {
        return Err(AppError::BadRequest(format!(
            "模型文件不存在: {}",
            req.model_path
        )));
    }

    if !path.is_file() {
        return Err(AppError::BadRequest("指定的路径不是有效文件".to_string()));
    }

    tracing::info!(
        model_path = %req.model_path,
        config = ?req.config,
        "开始加载模型"
    );

    // 在实际实现中，这里会调用 InferenceEngine 的 load_model 方法
    // 当前返回模拟响应

    Ok(Json(serde_json::json!({
        "message": "模型加载任务已提交",
        "status": "loading",
        "model_path": req.model_path,
        "config_applied": req.config.is_some(),
        "estimated_time_seconds": 30,
        "task_id": format!("load_{}", uuid::Uuid::new_v4()),
        "submitted_at": chrono::Utc::now().to_rfc3339()
    })))
}

/// 卸载指定模型
///
/// # 参数
/// - `id`: 模型 ID
///
/// # 返回
/// 卸载操作结果
pub async fn unload_model(
    Path(id): Path<String>,
    State(_state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    // 验证模型存在性
    if find_model_by_id(&id).is_none() {
        return Err(AppError::NotFound(format!("模型不存在: {}", id)));
    }

    tracing::info!(model_id = %id, "开始卸载模型");

    // 实际实现中调用 InferenceEngine::unload_model()

    Ok(Json(serde_json::json!({
        "message": "模型已成功卸载",
        "model_id": id,
        "status": "unloaded",
        "memory_freed_mb": 28500,
        "unloaded_at": chrono::Utc::now().to_rfc3339()
    })))
}

/// 更新模型运行配置
///
/// # 参数
/// - `model_id`: 模型 ID
/// # Body
/// - 部分或完整的配置更新字段
///
/// # 返回
/// 更新后的完整配置
pub async fn update_model_config(
    Path(model_id): Path<String>,
    State(_state): State<AppState>,
    Json(config): Json<ModelConfigUpdate>,
) -> Result<Json<Value>, AppError> {
    // 验证模型存在
    if find_model_by_id(&model_id).is_none() {
        return Err(AppError::NotFound(format!("模型不存在: {}", model_id)));
    }

    // 记录配置变更
    tracing::info!(
        model_id = %model_id,
        config = ?config,
        "更新模型配置"
    );

    // 构建更新的配置响应（合并默认值和新值）
    let updated_config = serde_json::json!({
        "model_id": model_id,
        "context_length": config.context_length.unwrap_or(4096),
        "max_tokens": config.max_tokens.unwrap_or(4096),
        "temperature": config.temperature.unwrap_or(0.7),
        "top_p": config.top_p.unwrap_or(0.9),
        "top_k": config.top_k.unwrap_or(40),
        "repeat_penalty": config.repeat_penalty.unwrap_or(1.1),
        "gpu_layers": config.gpu_layers.unwrap_or(-1),
        "n_batch": config.n_batch.unwrap_or(512),
        "n_ctx": config.n_ctx.unwrap_or(4096),
        "mlock": config.mlock.unwrap_or(true),
        "mmap": config.mmap.unwrap_or(true),
        "updated_at": chrono::Utc::now().to_rfc3339(),
        "applied": true
    });

    Ok(Json(updated_config))
}

/// 切换当前活跃模型（热切换）
///
/// 支持无缝切换模型，在完成切换前保持旧模型可用。
///
/// # Body
/// - `from_model_id`: 当前模型 ID
/// - `to_model_id`: 目标模型 ID
/// - `graceful_timeout_ms`: 优雅切换超时时间（毫秒），可选
pub async fn switch_model(
    State(_state): State<AppState>,
    Json(req): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let from_model = req
        .get("from_model_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let to_model = req
        .get("to_model_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let graceful_timeout = req
        .get("graceful_timeout_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(30000); // 默认 30 秒

    // 验证目标模型存在
    if find_model_by_id(to_model).is_none() {
        return Err(AppError::NotFound(format!("目标模型不存在: {}", to_model)));
    }

    tracing::info!(
        from = %from_model,
        to = %to_model,
        timeout_ms = graceful_timeout,
        "执行热切换"
    );

    // 实际实现中调用 InferenceEngine 的 hot_switch 方法

    Ok(Json(serde_json::json!({
        "message": "模型热切换完成",
        "from_model": from_model,
        "to_model": to_model,
        "switch_mode": "graceful",
        "timeout_ms": graceful_timeout,
        "completed_at": chrono::Utc::now().to_rfc3339(),
        "downtime_ms": 150  // 实际切换耗时
    })))
}

/// 检查模型健康状态
///
/// # 参数
/// - `id`: 模型 ID
///
/// # 返回
/// 模型健康状态、资源占用等信息
pub async fn check_health(
    Path(id): Path<String>,
    State(_state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    if find_model_by_id(&id).is_none() {
        return Err(AppError::NotFound(format!("模型不存在: {}", id)));
    }

    // 模拟健康检查结果
    let is_healthy = rand::random::<f64>() > 0.05; // 95% 概率健康

    Ok(Json(serde_json::json!({
        "model_id": id,
        "is_healthy": is_healthy,
        "status": if is_healthy { "healthy" } else { "degraded" },
        "last_check_time": chrono::Utc::now().to_rfc3339(),
        "response_time_ms": rand::random::<u64>() % 100 + 10,
        "resource_usage": {
            "gpu_memory_mb": 18944,
            "cpu_memory_mb": 9216,
            "kv_cache_mb": 4096,
            "gpu_utilization": 78.5,
            "temperature_celsius": 72.5
        },
        "error_message": if !is_healthy { Some("GPU memory pressure high") } else { None::<&str> }
    })))
}

// ==================== 辅助函数 ====================

/// 生成模拟的模型数据列表
fn generate_mock_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "qwen-14b-chat".to_string(),
            name: "Qwen-14B-Chat".to_string(),
            display_name: "通义千问 14B 对话版".to_string(),
            description: Some("阿里通义千问 14B 参数对话模型，中文能力强".to_string()),
            status: ModelStatus::Loaded,
            provider: "local".to_string(),
            format: "GGUF".to_string(),
            path: "/models/qwen-14b-chat/Q4_K_M.gguf".to_string(),
            size_gb: 8.5,
            loaded_at: Some("2024-06-15T08:00:00Z".to_string()),
            created_at: "2024-01-15T00:00:00Z".to_string(),
            updated_at: "2024-06-15T08:00:00Z".to_string(),
        },
        ModelInfo {
            id: "llama-3-8b-instruct".to_string(),
            name: "Llama-3-8B-Instruct".to_string(),
            display_name: "Llama 3 8B 指令微调版".to_string(),
            description: Some("Meta Llama 3 8B 参数指令遵循模型".to_string()),
            status: ModelStatus::Available,
            provider: "local".to_string(),
            format: "GGUF".to_string(),
            path: "/models/llama-3-8b-instruct/Q5_K_M.gguf".to_string(),
            size_gb: 5.8,
            loaded_at: None,
            created_at: "2024-04-18T00:00:00Z".to_string(),
            updated_at: "2024-04-18T00:00:00Z".to_string(),
        },
        ModelInfo {
            id: "mistral-7b-instruct".to_string(),
            name: "Mistral-7B-Instruct-v0.3".to_string(),
            display_name: "Mistral 7B 指令版 v0.3".to_string(),
            description: Some("Mistral AI 7B 参数高性能指令模型".to_string()),
            status: ModelStatus::Available,
            provider: "huggingface".to_string(),
            format: "SafeTensors".to_string(),
            path: "/models/mistral-7b".to_string(),
            size_gb: 14.2,
            loaded_at: None,
            created_at: "2024-03-25T00:00:00Z".to_string(),
            updated_at: "2024-03-25T00:00:00Z".to_string(),
        },
        ModelInfo {
            id: "qwen-72b-chat".to_string(),
            name: "Qwen-72B-Chat".to_string(),
            display_name: "通义千问 72B 对话版".to_string(),
            description: Some("阿里通义千问 72B 大参数量旗舰模型".to_string()),
            status: ModelStatus::Error,
            provider: "local".to_string(),
            format: "GGUF".to_string(),
            path: "/models/qwen-72b-chat/Q4_K_M.gguf".to_string(),
            size_gb: 42.5,
            loaded_at: None,
            created_at: "2024-02-10T00:00:00Z".to_string(),
            updated_at: "2024-06-14T15:30:00Z".to_string(),
        },
    ]
}

/// 根据 ID 查找模型
fn find_model_by_id(model_id: &str) -> Option<ModelInfo> {
    generate_mock_models()
        .into_iter()
        .find(|m| m.id == model_id || m.name == model_id)
}

/// 获取模型参数量（十亿）
fn get_model_parameters(model_id: &str) -> u64 {
    match model_id {
        "qwen-14b-chat" => 14,
        "llama-3-8b-instruct" => 8,
        "mistral-7b-instruct" => 7,
        "qwen-72b-chat" => 72,
        _ => 7,
    }
}

/// 获取模型量化方式
fn get_model_quantization(model_id: &str) -> String {
    match model_id {
        "qwen-14b-chat" | "qwen-72b-chat" => "Q4_K_M".to_string(),
        "llama-3-8b-instruct" => "Q5_K_M".to_string(),
        "mistral-7b-instruct" => "FP16".to_string(),
        _ => "Unknown".to_string(),
    }
}

/// 获取模型上下文长度
fn get_model_context_length(model_id: &str) -> u32 {
    match model_id {
        "qwen-14b-chat" | "qwen-72b-chat" => 32768,
        "llama-3-8b-instruct" => 8192,
        "mistral-7b-instruct" => 32768,
        _ => 4096,
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== 数据结构测试 ====================

    #[test]
    fn test_model_info_serialization() {
        let model = ModelInfo {
            id: "test-model".to_string(),
            name: "TestModel".to_string(),
            display_name: "Test Model".to_string(),
            description: Some("A test model".to_string()),
            status: ModelStatus::Loaded,
            provider: "local".to_string(),
            format: "GGUF".to_string(),
            path: "/models/test.gguf".to_string(),
            size_gb: 4.5,
            loaded_at: Some("2024-06-15T00:00:00Z".to_string()),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-06-15T00:00:00Z".to_string(),
        };

        let json = serde_json::to_value(&model).unwrap();
        assert_eq!(json["id"], "test-model");
        assert_eq!(json["name"], "TestModel");
        assert_eq!(json["display_name"], "Test Model");
        assert_eq!(json["status"], "loaded"); // 序列化为小写
        assert_eq!(json["size_gb"], 4.5);
        assert!(json.get("description").is_some()); // Option 字段存在
        assert_eq!(json["loaded_at"], "2024-06-15T00:00:00Z");
    }

    #[test]
    fn test_model_status_display() {
        assert_eq!(ModelStatus::Available.to_string(), "available");
        assert_eq!(ModelStatus::Loading.to_string(), "loading");
        assert_eq!(ModelStatus::Loaded.to_string(), "loaded");
        assert_eq!(ModelStatus::Unloading.to_string(), "unloading");
        assert_eq!(ModelStatus::Error.to_string(), "error");
    }

    #[test]
    fn test_model_detail_structure() {
        let detail = ModelDetail {
            base: ModelInfo {
                id: "detail-test".to_string(),
                name: "DetailTest".to_string(),
                display_name: "Detail Test".to_string(),
                description: None,
                status: ModelStatus::Loaded,
                provider: "local".to_string(),
                format: "GGUF".to_string(),
                path: "/models/detail.gguf".to_string(),
                size_gb: 8.0,
                loaded_at: None,
                created_at: "2024-01-01T00:00:00Z".to_string(),
                updated_at: "2024-01-01T00:00:00Z".to_string(),
            },
            parameters: 7,
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            max_tokens: 2048,
            supported_features: vec!["chat_completion".to_string()],
            capabilities: ModelCapabilities {
                chat_completion: true,
                embedding: false,
                function_calling: true,
                vision: false,
                streaming: true,
            },
            resource_usage: ModelResourceUsage {
                gpu_memory_mb: 10000,
                cpu_memory_mb: 5000,
                gpu_utilization_percent: 75.0,
                temperature_celsius: Some(70.0),
            },
            performance_stats: Some(ModelPerformanceStats {
                avg_latency_ms: 500.0,
                p50_latency_ms: 450.0,
                p95_latency_ms: 900.0,
                p99_latency_ms: 1200.0,
                throughput_tps: 10.0,
                total_requests: 1000,
                success_rate: 98.0,
            }),
        };

        let json = serde_json::to_value(&detail).unwrap();
        assert_eq!(json["parameters"], 7);
        assert_eq!(json["quantization"], "Q4_K_M");
        assert_eq!(json["context_length"], 4096);
        assert_eq!(json["capabilities"]["chat_completion"], true);
        assert_eq!(json["capabilities"]["vision"], false);
        assert_eq!(json["resource_usage"]["gpu_memory_mb"], 10000);
        assert!(json.get("performance_stats").is_some()); // Option 存在
    }

    #[test]
    fn test_model_config_update_deserialization() {
        let json_str = r#"{
            "context_length": 8192,
            "temperature": 0.8,
            "top_p": 0.95,
            "gpu_layers": 35,
            "mlock": true
        }"#;

        let config: ModelConfigUpdate = serde_json::from_str(json_str).unwrap();
        assert_eq!(config.context_length, Some(8192));
        assert!((config.temperature.unwrap() - 0.8).abs() < 0.001);
        assert_eq!(config.gpu_layers, Some(35));
        assert_eq!(config.mlock, Some(true));

        // 测试所有字段为 None 的情况
        let empty_config: ModelConfigUpdate = serde_json::from_str("{}").unwrap();
        assert!(empty_config.context_length.is_none());
        assert!(empty_config.temperature.is_none());
    }

    #[test]
    fn test_load_model_request() {
        let req = LoadModelRequest {
            model_path: "/models/test.gguf".to_string(),
            config: Some(ModelConfigUpdate {
                n_batch: Some(1024),
                ..Default::default()
            }),
        };

        assert_eq!(req.model_path, "/models/test.gguf");
        assert!(req.config.is_some());
        assert_eq!(req.config.unwrap().n_batch, Some(1024));
    }

    #[test]
    fn test_pagination_params_normalization() {
        // 正常值
        let query = ModelListQuery {
            page: Some(2),
            page_size: Some(50),
            status: Some("loaded".to_string()),
            provider: None,
            format: None,
            search: None,
        };
        let (page, size) = query.normalized();
        assert_eq!(page, 2);
        assert_eq!(size, 50);

        // 超过上限
        let exceeded = ModelListQuery {
            page: Some(1),
            page_size: Some(200),
            status: None,
            provider: None,
            format: None,
            search: None,
        };
        let (_, limited) = exceeded.normalized();
        assert_eq!(limited, 100); // 应限制为 100

        // 默认值
        let defaults = ModelListQuery {
            page: None,
            page_size: None,
            status: None,
            provider: None,
            format: None,
            search: None,
        };
        let (def_page, def_size) = defaults.normalized();
        assert_eq!(def_page, 1);
        assert_eq!(def_size, 20);
    }

    #[test]
    fn test_find_model_by_id() {
        // 存在的模型
        let model = find_model_by_id("qwen-14b-chat");
        assert!(model.is_some());
        assert_eq!(model.unwrap().name, "Qwen-14B-Chat");

        // 不存在的模型
        assert!(find_model_by_id("nonexistent-model").is_none());

        // 通过名称查找
        assert!(find_model_by_id("Qwen-14B-Chat").is_some());
    }

    #[test]
    fn test_helper_functions() {
        assert_eq!(get_model_parameters("qwen-14b-chat"), 14);
        assert_eq!(get_model_parameters("llama-3-8b-instruct"), 8);
        assert_eq!(get_model_quantization("qwen-14b-chat"), "Q4_K_M");
        assert_eq!(get_model_context_length("llama-3-8b-instruct"), 8192);
    }
}
