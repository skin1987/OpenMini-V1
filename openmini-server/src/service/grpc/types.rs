//! gRPC 类型定义
//!
//! 定义 gRPC 服务中使用的消息类型

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub session_id: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    pub max_tokens: i32,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub session_id: String,
    pub token: String,
    pub finished: bool,
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageRequest {
    pub session_id: String,
    pub image_data: Vec<u8>,
    pub question: String,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResponse {
    pub session_id: String,
    pub token: String,
    pub finished: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub healthy: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniChatRequest {
    pub session_id: String,
    pub input: Option<OmniInput>,
    pub stream: bool,
    pub max_tokens: i32,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OmniInput {
    Text(String),
    AudioData(Vec<u8>),
    VideoData(Vec<u8>),
    ImageData(Vec<u8>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OmniChatResponse {
    pub session_id: String,
    pub output: Option<OmniOutput>,
    pub finished: bool,
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OmniOutput {
    Text(String),
    AudioData(Vec<u8>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechToTextRequest {
    pub session_id: String,
    pub audio_data: Vec<u8>,
    pub language: String,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechToTextResponse {
    pub session_id: String,
    pub text: String,
    pub finished: bool,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextToSpeechRequest {
    pub session_id: String,
    pub text: String,
    pub voice: String,
    pub language: String,
    pub speed: f32,
    pub pitch: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextToSpeechResponse {
    pub session_id: String,
    pub audio_data: Vec<u8>,
    pub finished: bool,
}
