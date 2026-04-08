//! HTTP REST API 类型定义
//!
//! 定义 HTTP API 的请求和响应结构体，与 gRPC 类型保持一致的风格。

use serde::{Deserialize, Serialize};

/// 聊天完成请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// 会话ID（可选，用于上下文管理）
    #[serde(default)]
    pub session_id: Option<String>,
    /// 消息列表
    pub messages: Vec<ChatMessage>,
    /// 是否流式输出
    #[serde(default)]
    pub stream: bool,
    /// 最大生成 token 数
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,
    /// 采样温度 (0.0-2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

/// 聊天消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// 角色: system/user/assistant
    pub role: String,
    /// 消息内容
    pub content: String,
}

/// 聊天完成响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// 唯一请求ID
    pub id: String,
    /// 对象类型
    pub object: String,
    /// 创建时间戳
    pub created: i64,
    /// 模型名称
    pub model: String,
    /// 选择列表
    pub choices: Vec<ChatChoice>,
    /// 使用统计
    pub usage: Option<UsageInfo>,
}

/// 聊天选择项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    /// 索引
    pub index: i32,
    /// 消息内容
    pub message: ChatMessage,
    /// 完成原因
    pub finish_reason: String,
}

/// 流式聊天 chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// 唯一请求ID
    pub id: String,
    /// 对象类型
    pub object: String,
    /// 创建时间戳
    pub created: i64,
    /// 模型名称
    pub model: String,
    /// 选择列表
    pub choices: Vec<DeltaChoice>,
}

/// 流式增量选择
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaChoice {
    /// 索引
    pub index: i32,
    /// 增量内容
    pub delta: DeltaContent,
    /// 完成原因
    pub finish_reason: Option<String>,
}

/// 增量内容
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaContent {
    /// 角色流（仅第一个chunk）
    pub role: Option<String>,
    /// 内容增量
    pub content: Option<String>,
}

/// 图像理解请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUnderstandRequest {
    /// 会话ID
    #[serde(default)]
    pub session_id: Option<String>,
    /// Base64 编码的图像数据
    pub image: String,
    /// 问题文本
    #[serde(default)]
    pub question: Option<String>,
}

/// 图像理解响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUnderstandResponse {
    /// 唯一请求ID
    pub id: String,
    /// 图像描述/回答
    pub description: String,
}

/// TTS 请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRequest {
    /// 要转换的文本
    pub text: String,
    /// 语音ID
    #[serde(default = "default_voice")]
    pub voice: String,
    /// 语言代码
    #[serde(default = "default_language")]
    pub language: String,
    /// 语速 (0.25-4.0)
    #[serde(default = "default_speed")]
    pub speed: f32,
    /// 音调 (-20.0 到 20.0)
    #[serde(default)]
    pub pitch: f32,
}

/// TTS 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsResponse {
    /// 音频数据（Base64编码）
    pub audio_data: String,
}

/// STT 请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttRequest {
    /// 音频数据（Base64编码）
    pub audio_data: String,
    /// 语言代码
    #[serde(default = "default_language")]
    pub language: String,
}

/// STT 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttResponse {
    /// 识别的文本
    pub text: String,
    /// 置信度 (0.0-1.0)
    pub confidence: f32,
}

/// 使用统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    /// 提示 token 数
    pub prompt_tokens: i32,
    /// 完成 token 数
    pub completion_tokens: i32,
    /// 总 token 数
    pub total_tokens: i32,
}

/// 模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// 模型ID
    pub id: String,
    /// 模型名称
    pub name: String,
    /// 是否支持多模态
    pub multimodal: bool,
    /// 上下文长度
    pub context_length: usize,
}

/// API 错误响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// 错误类型
    pub error_type: String,
    /// 错误消息
    pub message: String,
    /// HTTP 状态码
    pub status_code: u16,
}

/// 健康检查响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    /// 服务状态
    pub status: String,
    /// 版本号
    pub version: String,
    /// 启动时间
    pub uptime_seconds: u64,
}

// ============================================================================
// 默认值函数
// ============================================================================

fn default_max_tokens() -> i32 { 1024 }
fn default_temperature() -> f32 { 0.7 }
fn default_voice() -> String { "alloy".to_string() }
fn default_language() -> String { "zh".to_string() }
fn default_speed() -> f32 { 1.0 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_request_defaults() {
        let json = r#"{"messages": [{"role": "user", "content": "Hello"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 1024);
        assert!((req.temperature - 0.7).abs() < 1e-6);
        assert!(!req.stream);
        assert!(req.session_id.is_none());
    }

    #[test]
    fn test_chat_completion_request_full() {
        let req = ChatCompletionRequest {
            session_id: Some("sess-123".to_string()),
            messages: vec![ChatMessage { role: "user".to_string(), content: "Hi".to_string() }],
            stream: true,
            max_tokens: 2048,
            temperature: 0.9,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("sess-123"));
        assert!(json.contains("2048"));
    }

    #[test]
    fn test_image_understand_request() {
        let req = ImageUnderstandRequest {
            session_id: None,
            image: "base64data...".to_string(),
            question: Some("What's this?".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("base64data"));
        assert!(json.contains("What's this?"));
    }

    #[test]
    fn test_tts_request_defaults() {
        let json = r#"{"text": "Hello world"}"#;
        let req: TtsRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.voice, "alloy");
        assert_eq!(req.language, "zh");
        assert!((req.speed - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_api_error_serialization() {
        let err = ApiError {
            error_type: "invalid_request".to_string(),
            message: "Messages cannot be empty".to_string(),
            status_code: 400,
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("invalid_request"));
        assert!(json.contains("400"));
    }

    #[test]
    fn test_model_info() {
        let model = ModelInfo {
            id: "openmini-v1".to_string(),
            name: "OpenMini V1".to_string(),
            multimodal: true,
            context_length: 4096,
        };
        assert!(model.multimodal);
        assert_eq!(model.context_length, 4096);
    }
}
