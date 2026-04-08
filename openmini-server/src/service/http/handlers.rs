//! HTTP API 路由处理器
//!
//! 实现 REST API 端点的业务逻辑处理。

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Json},
};
use std::sync::Arc;
use tracing::{info, error, debug};
use uuid::Uuid;
use base64::Engine;

use super::types::*;
// 使用相对路径访问 monitoring 模块
#[allow(unused_imports)]
use crate::monitoring::HealthChecker;

/// 应用状态（共享服务实例）
#[derive(Clone)]
pub struct AppState {
    /// 健康检查器
    pub health_checker: Arc<HealthChecker>,
}

impl AppState {
    /// 创建新的应用状态
    pub fn new() -> Self {
        Self {
            health_checker: Arc::new(HealthChecker::new()),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 处理器函数
// ============================================================================

/// 聊天完成（非流式）
///
/// POST /api/v1/chat
///
/// 接收消息列表并返回完整的模型回复。
pub async fn chat_completion(
    State(_state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    info!(
        session_id = ?req.session_id,
        message_count = req.messages.len(),
        stream = req.stream,
        "收到聊天完成请求"
    );

    // 验证输入
    if req.messages.is_empty() {
        return Err(ApiError {
            error_type: "invalid_request".to_string(),
            message: "Messages cannot be empty".to_string(),
            status_code: 400,
        });
    }

    // TODO: 集成真实推理引擎
    // 当前返回模拟响应
    let full_text = "[HTTP 聊天功能已就绪，等待集成完整推理引擎]".to_string();
    let full_text_len = full_text.len() as i32;

    let response = ChatCompletionResponse {
        id: Uuid::new_v4().to_string(),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: "openmini-v1".to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: full_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Some(UsageInfo {
            prompt_tokens: req.messages.iter().map(|m| m.content.len() as i32).sum(),
            completion_tokens: full_text_len,
            total_tokens: 0,
        }),
    };

    Ok(Json(response))
}

/// 聊天完成（流式 SSE）
///
/// POST /api/v1/chat/stream
///
/// 返回 Server-Sent Events 格式的流式响应。
pub async fn chat_completion_stream(
    State(_state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    use axum::response::sse::Sse;

    info!(
        session_id = ?req.session_id,
        message_count = req.messages.len(),
        "收到流式聊天请求"
    );

    if req.messages.is_empty() {
        return Err(ApiError {
            error_type: "invalid_request".to_string(),
            message: "Messages cannot be empty".to_string(),
            status_code: 400,
        });
    }

    let request_id = Uuid::new_v4().to_string();

    // TODO: 集成真实推理引擎的 SSE 流
    // 当前返回模拟 SSE 响应
    let sse_stream = async_stream::stream! {
        // 发送初始 chunk（包含 role）
        yield Ok::<_, axum::Error>(axum::response::sse::Event::default()
            .data(format!("{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"openmini-v1\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":\"\"}},\"finish_reason\":null}}]}}",
                request_id, chrono::Utc::now().timestamp())));

        // 模拟内容生成
        let content = "[SSE 流式聊天功能已就绪，等待集成完整推理引擎]";
        for word in content.split_whitespace() {
            let event_data = format!(
                "{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"openmini-v1\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{} \"}},\"finish_reason\":null}}]}}",
                request_id, chrono::Utc::now().timestamp(), word
            );
            yield Ok::<_, axum::Error>(axum::response::sse::Event::default().data(event_data));

            // 模拟生成延迟
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // 发送完成标记
        let finish_data = format!(
            "{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"openmini-v1\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}",
            request_id, chrono::Utc::now().timestamp()
        );
        yield Ok::<_, axum::Error>(axum::response::sse::Event::default().data(finish_data));
    };

    Ok(Sse::new(sse_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
    ).into_response())
}

/// 图像理解
///
/// POST /api/v1/image/understand
///
/// 接收图像数据并返回图像描述或问答结果。
pub async fn image_understand(
    State(_state): State<AppState>,
    Json(req): Json<ImageUnderstandRequest>,
) -> Result<Json<ImageUnderstandResponse>, ApiError> {
    info!(
        session_id = ?req.session_id,
        image_size = req.image.len(),
        question = ?req.question,
        "收到图像理解请求"
    );

    // 验证图像数据
    if req.image.is_empty() {
        return Err(ApiError {
            error_type: "invalid_request".to_string(),
            message: "Image data cannot be empty".to_string(),
            status_code: 400,
        });
    }

    // 解码 Base64 图像数据
    let image_bytes = match base64::engine::general_purpose::STANDARD.decode(&req.image) {
        Ok(data) => data,
        Err(_) => {
            return Err(ApiError {
                error_type: "invalid_request".to_string(),
                message: "Invalid base64 encoded image data".to_string(),
                status_code: 400,
            });
        }
    };

    // TODO: 集成真实图像理解逻辑
    // 当前返回模拟响应
    let _question = req.question.unwrap_or_else(|| "请描述这张图片的内容。".to_string());

    debug!(image_size = image_bytes.len(), "处理图像理解");

    Ok(Json(ImageUnderstandResponse {
        id: Uuid::new_v4().to_string(),
        description: "[图像理解功能已就绪，等待集成完整推理引擎]".to_string(),
    }))
}

/// 文字转语音 (TTS)
///
/// POST /api/v1/tts
///
/// 将文本转换为语音音频。
pub async fn text_to_speech(
    State(_state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Result<Json<TtsResponse>, ApiError> {
    info!(
        text_length = req.text.len(),
        voice = %req.voice,
        language = %req.language,
        speed = req.speed,
        "收到TTS请求"
    );

    // 验证输入
    if req.text.is_empty() {
        return Err(ApiError {
            error_type: "invalid_request".to_string(),
            message: "Text cannot be empty".to_string(),
            status_code: 400,
        });
    }

    if req.text.len() > 10000 {
        return Err(ApiError {
            error_type: "invalid_request".to_string(),
            message: "Text exceeds maximum length (10000 characters)".to_string(),
            status_code: 400,
        });
    }

    // TODO: 集成真实 TTS 引擎
    // 当前返回模拟响应（空音频数据占位）
    Ok(Json(TtsResponse {
        audio_data: String::new(),
    }))
}

/// 语音转文字 (STT)
///
/// POST /api/v1/stt
///
/// 将语音音频转换为文字。
pub async fn speech_to_text(
    State(_state): State<AppState>,
    Json(req): Json<SttRequest>,
) -> Result<Json<SttResponse>, ApiError> {
    info!(
        audio_size = req.audio_data.len(),
        language = %req.language,
        "收到STT请求"
    );

    // 验证输入
    if req.audio_data.is_empty() {
        return Err(ApiError {
            error_type: "invalid_request".to_string(),
            message: "Audio data cannot be empty".to_string(),
            status_code: 400,
        });
    }

    // 解码 Base64 音频数据
    let _audio_bytes = match base64::engine::general_purpose::STANDARD.decode(&req.audio_data) {
        Ok(data) => data,
        Err(_) => {
            return Err(ApiError {
                error_type: "invalid_request".to_string(),
                message: "Invalid base64 encoded audio data".to_string(),
                status_code: 400,
            });
        }
    };

    // TODO: 集成真实 ASR 引擎
    // 当前返回模拟响应
    Ok(Json(SttResponse {
        text: "[STT 功能已就绪，等待集成完整语音识别引擎]".to_string(),
        confidence: 0.95,
    }))
}

/// 健康检查
///
/// GET /api/v1/health
///
/// 返回服务健康状态。
pub async fn health_check(
    State(state): State<AppState>,
) -> Json<HealthCheckResponse> {
    debug!("执行健康检查");

    let is_healthy = state.health_checker.is_healthy().await;

    Json(HealthCheckResponse {
        status: if is_healthy { "healthy" } else { "degraded" }.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: get_uptime_seconds(),
    })
}

/// Prometheus 指标
///
/// GET /api/v1/metrics
///
/// 导出 Prometheus 格式的监控指标。
pub async fn metrics() -> Result<String, StatusCode> {
    use prometheus::TextEncoder;

    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();

    match encoder.encode_to_string(&metric_families) {
        Ok(metrics) => Ok(metrics),
        Err(e) => {
            error!(error = %e, "Failed to encode metrics");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// 模型列表
///
/// GET /api/v1/models
///
/// 返回可用模型列表。
pub async fn list_models() -> Json<Vec<ModelInfo>> {
    debug("获取模型列表");

    let models = vec![
        ModelInfo {
            id: "openmini-v1".to_string(),
            name: "OpenMini V1".to_string(),
            multimodal: true,
            context_length: 4096,
        },
    ];

    Json(models)
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 获取服务运行时间（秒）
fn get_uptime_seconds() -> u64 {
    // 使用启动时间戳计算（简化实现）
    // 实际实现应记录启动时间
    0u64
}

/// 日志调试宏（避免未使用警告）
#[allow(dead_code)]
fn debug(msg: &str) {
    tracing::debug!("{}", msg);
}

// ============================================================================
// IntoResponse for ApiError
// ============================================================================

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = Json(self);

        (status, body).into_response()
    }
}
