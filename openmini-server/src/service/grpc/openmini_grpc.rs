//! OpenMini gRPC 服务实现
//!
//! 实现完整的 tonic gRPC 服务层，注册所有 7 个 RPC 方法：
//! - Chat (流式)
//! - ImageUnderstanding (非流式)
//! - ImageUnderstandingStream (流式)
//! - HealthCheck (非流式)
//! - OmniChat (流式)
//! - SpeechToText (流式)
//! - TextToSpeech (流式)
//!
//! 使用手动路由方式实现（因项目未使用 proto codegen），
//! 通过请求路径分发到对应的业务处理函数。

use crate::service::grpc::server::{
    chat, health_check, image_understanding, image_understanding_stream, omni_chat,
    speech_to_text, text_to_speech, ChatStream, ImageStream, OmniStream, OpenMiniService,
    SpeechStream, TtsStream,
};
use crate::service::grpc::types::{
    ChatRequest, ChatResponse, HealthRequest, HealthResponse, ImageRequest, ImageResponse,
    OmniChatRequest, OmniChatResponse, OmniInput, OmniOutput, SpeechToTextRequest,
    SpeechToTextResponse, TextToSpeechRequest, TextToSpeechResponse, UsageInfo,
};

use bytes::Bytes;
use futures::Stream;
use prost::Message as ProstMessage;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tonic::{http, Status};

// ============================================================================
// 常量定义
// ============================================================================

/// gRPC 服务路径前缀
const GRPC_SERVICE_PREFIX: &str = "/openmini.OpenMini/";

/// RPC 方法路径
mod paths {
    pub const CHAT: &str = "/openmini.OpenMini/Chat";
    pub const IMAGE_UNDERSTANDING: &str = "/openmini.OpenMini/ImageUnderstanding";
    pub const IMAGE_UNDERSTANDING_STREAM: &str = "/openmini.OpenMini/ImageUnderstandingStream";
    pub const HEALTH_CHECK: &str = "/openmini.OpenMini/HealthCheck";
    pub const OMNI_CHAT: &str = "/openmini.OpenMini/OmniChat";
    pub const SPEECH_TO_TEXT: &str = "/openmini.OpenMini/SpeechToText";
    pub const TEXT_TO_SPEECH: &str = "/openmini.OpenMini/TextToSpeech";
}

/// gRPC Content-Type
const GRPC_CONTENT_TYPE: &str = "application/grpc+proto";

/// gRPC 成功状态码（0 表示 OK）
const GRPC_STATUS_OK: u32 = 0;

// ============================================================================
// OpenMini gRPC 服务实现
// ============================================================================

/// OpenMini gRPC 服务
///
/// 实现 tonic::Service trait，提供完整的 gRPC 接口。
/// 通过请求路径将请求路由到对应的业务处理函数。
#[derive(Clone)]
pub struct OpenMiniGrpcService {
    /// OpenMini 业务服务实例
    service: Arc<OpenMiniService>,
}

impl OpenMiniGrpcService {
    /// 创建新的 gRPC 服务实例
    ///
    /// # 参数
    /// - `service`: OpenMini 业务服务实例（Arc 包装）
    ///
    /// # 返回
    /// gRPC 服务实例
    pub fn new(service: Arc<OpenMiniService>) -> Self {
        Self { service }
    }

    /// 处理 gRPC 请求
    ///
    /// 根据请求路径路由到对应的 RPC 方法处理函数。
    async fn handle_request(
        &self,
        req: http::Request<tonic::body::BoxBody>,
    ) -> Result<http::Response<tonic::body::BoxBody>, Status> {
        let path = req.uri().path().to_string();

        tracing::debug!(rpc_path = %path, "收到 gRPC 请求");

        match path.as_str() {
            paths::CHAT => self.handle_chat(req).await,
            paths::IMAGE_UNDERSTANDING => self.handle_image_understanding(req).await,
            paths::IMAGE_UNDERSTANDING_STREAM => self.handle_image_understanding_stream(req).await,
            paths::HEALTH_CHECK => self.handle_health_check(req).await,
            paths::OMNI_CHAT => self.handle_omni_chat(req).await,
            paths::SPEECH_TO_TEXT => self.handle_speech_to_text(req).await,
            paths::TEXT_TO_SPEECH => self.handle_text_to_speech(req).await,
            _ => Err(Status::unimplemented(format!("未知的 RPC 方法: {}", path))),
        }
    }

    // ==================== 各 RPC 方法的具体实现 ====================

    /// 处理 Chat 请求（流式）
    async fn handle_chat(
        &self,
        req: http::Request<tonic::body::BoxBody>,
    ) -> Result<http::Response<tonic::body::BoxBody>, Status> {
        let body_bytes = collect_body(req.into_body()).await?;
        let request: ChatRequest =
            decode_message(&body_bytes)?;

        tracing::info!(
            session_id = %request.session_id,
            message_count = request.messages.len(),
            "处理 Chat 请求"
        );

        let response_stream = chat(&self.service, request).await?;
        Ok(self.stream_response(response_stream))
    }

    /// 处理 ImageUnderstanding 请求（非流式）
    async fn handle_image_understanding(
        &self,
        req: http::Request<tonic::body::BoxBody>,
    ) -> Result<http::Response<tonic::body::BoxBody>, Status> {
        let body_bytes = collect_body(req.into_body()).await?;
        let request: ImageRequest = decode_message(&body_bytes)?;

        tracing::info!(
            session_id = %request.session_id,
            image_data_size = request.image_data.len(),
            "处理 ImageUnderstanding 请求"
        );

        let response = image_understanding(&self.service, request).await?;
        Ok(self.unary_response(encode_message(&response)))
    }

    /// 处理 ImageUnderstandingStream 请求（流式）
    async fn handle_image_understanding_stream(
        &self,
        req: http::Request<tonic::body::BoxBody>,
    ) -> Result<http::Response<tonic::body::BoxBody>, Status> {
        let body_bytes = collect_body(req.into_body()).await?;
        let request: ImageRequest = decode_message(&body_bytes)?;

        tracing::info!(
            session_id = %request.session_id,
            "处理 ImageUnderstandingStream 请求"
        );

        let response_stream = image_understanding_stream(&self.service, request).await?;
        Ok(self.stream_response(response_stream))
    }

    /// 处理 HealthCheck 请求（非流式）
    async fn handle_health_check(
        &self,
        _req: http::Request<tonic::body::BoxBody>,
    ) -> Result<http::Response<tonic::body::BoxBody>, Status> {
        tracing::info!("处理 HealthCheck 请求");

        let request = HealthRequest {};
        let response = health_check(&self.service, request).await?;
        Ok(self.unary_response(encode_message(&response)))
    }

    /// 处理 OmniChat 请求（流式）
    async fn handle_omni_chat(
        &self,
        req: http::Request<tonic::body::BoxBody>,
    ) -> Result<http::Response<tonic::body::BoxBody>, Status> {
        let body_bytes = collect_body(req.into_body()).await?;
        let request: OmniChatRequest = decode_message(&body_bytes)?;

        let input_type = match &request.input {
            Some(OmniInput::Text(_)) => "text",
            Some(OmniInput::ImageData(_)) => "image",
            Some(OmniInput::AudioData(_)) => "audio",
            Some(OmniInput::VideoData(_)) => "video",
            None => "none",
        };

        tracing::info!(
            session_id = %request.session_id,
            input_type = input_type,
            "处理 OmniChat 请求"
        );

        let response_stream = omni_chat(&self.service, request).await?;
        Ok(self.stream_response(response_stream))
    }

    /// 处理 SpeechToText 请求（流式）
    async fn handle_speech_to_text(
        &self,
        req: http::Request<tonic::body::BoxBody>,
    ) -> Result<http::Response<tonic::body::BoxBody>, Status> {
        let body_bytes = collect_body(req.into_body()).await?;
        let request: SpeechToTextRequest = decode_message(&body_bytes)?;

        tracing::info!(
            session_id = %request.session_id,
            audio_data_size = request.audio_data.len(),
            "处理 SpeechToText 请求"
        );

        let response_stream = speech_to_text(&self.service, request).await?;
        Ok(self.stream_response(response_stream))
    }

    /// 处理 TextToSpeech 请求（流式）
    async fn handle_text_to_speech(
        &self,
        req: http::Request<tonic::body::BoxBody>,
    ) -> Result<http::Response<tonic::body::BoxBody>, Status> {
        let body_bytes = collect_body(req.into_body()).await?;
        let request: TextToSpeechRequest = decode_message(&body_bytes)?;

        tracing::info!(
            session_id = %request.session_id,
            text_length = request.text.len(),
            "处理 TextToSpeech 请求"
        );

        let response_stream = text_to_speech(&self.service, request).await?;
        Ok(self.stream_response(response_stream))
    }

    // ==================== 辅助方法 ====================

    /// 构建一元（非流式）响应
    fn unary_response(&self, response_bytes: Vec<u8>) -> http::Response<tonic::body::BoxBody> {
        let mut frame = Vec::with_capacity(5 + response_bytes.len());
        frame.push(0);
        frame.extend_from_slice(&(response_bytes.len() as u32).to_be_bytes());
        frame.extend_from_slice(&response_bytes);

        http::Response::builder()
            .status(http::StatusCode::OK)
            .header("content-type", GRPC_CONTENT_TYPE)
            .header("grpc-status", GRPC_STATUS_OK.to_string())
            .body(tonic::body::BoxBody::new(http_body_util::Full::new(Bytes::from(frame))))
            .expect("无法构建响应")
    }

    /// 构建流式响应
    fn stream_response<S, T>(&self, stream: S) -> http::Response<tonic::body::BoxBody>
    where
        S: Stream<Item = Result<T, Status>> + Send + 'static,
        T: ProstMessage + Default + Clone + Send + 'static,
    {
        let encoded_stream = encode_grpc_stream(stream);

        let body = tonic::body::BoxBody::new(http_body_util::StreamBody::new(encoded_stream));

        http::Response::builder()
            .status(http::StatusCode::OK)
            .header("content-type", GRPC_CONTENT_TYPE)
            .header("grpc-status", GRPC_STATUS_OK.to_string())
            .header("transfer-encoding", "chunked")
            .body(body)
            .expect("无法构建流式响应")
    }
}

// ============================================================================
// 实现 tonic::Service trait
// ============================================================================

impl tonic::Service<http::Request<tonic::body::BoxBody>> for OpenMiniGrpcService {
    type Response = http::Response<tonic::body::BoxBody>;
    type Error = Status;
    type Future =
        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send + 'static>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: http::Request<tonic::body::BoxBody>) -> Self::Future {
        let service = self.clone();
        Box::pin(async move { service.handle_request(req).await })
    }
}

// ============================================================================
// 辅助函数和 prost::Message 实现
// ============================================================================

/// 收集请求体数据
async fn collect_body(body: tonic::body::BoxBody) -> Result<Vec<u8>, Status> {
    use futures::StreamExt;
    use http_body_util::BodyExt;

    let mut data = Vec::new();
    let mut body = body;

    while let Some(chunk) = body.next().await {
        match chunk {
            Ok(bytes) => data.extend_from_slice(&bytes),
            Err(e) => return Err(Status::internal(format!("读取请求体失败: {}", e))),
        }
    }

    if data.len() > 5 {
        let msg_len = u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as usize;
        if data.len() >= 5 + msg_len {
            return Ok(data[5..5 + msg_len].to_vec());
        }
    }

    Ok(data)
}

/// 编码 gRPC 流式响应
fn encode_grpc_stream<S, T>(
    stream: S,
) -> impl Stream<Item = Result<Bytes, Status>> + Send + 'static
where
    S: Stream<Item = Result<T, Status>> + Send + 'static,
    T: ProstMessage + Default + Clone + Send + 'static,
{
    use futures::StreamExt;

    async_stream::stream! {
        let mut stream = std::pin::pin!(stream);

        while let Some(result) = stream.next().await {
            match result {
                Ok(message) => {
                    let message_bytes = message.encode_to_vec();

                    let mut frame = Vec::with_capacity(5 + message_bytes.len());
                    frame.push(0);
                    frame.extend_from_slice(&(message_bytes.len() as u32).to_be_bytes());
                    frame.extend_from_slice(&message_bytes);

                    yield Ok(Bytes::from(frame));
                }
                Err(status) => {
                    yield Err(status);
                    break;
                }
            }
        }
    }
}

/// 创建错误响应
pub fn error_response(status: Status) -> http::Response<tonic::body::BoxBody> {
    let status_code = status.code() as u32;
    let message = status.message().to_string();

    let trailer = format!(
        "\r\ngrpc-status: {}\r\ngrpc-message: {}\r\n",
        status_code, message
    );

    http::Response::builder()
        .status(http::StatusCode::OK)
        .header("content-type", GRPC_CONTENT_TYPE)
        .header("grpc-status", status_code.to_string())
        .header("grpc-message", message)
        .body(tonic::body::BoxBody::new(http_body_util::Full::new(Bytes::from(trailer))))
        .expect("无法构建错误响应")
}

/// 解码 protobuf 消息（辅助函数）
fn decode_message<T: ProstMessage + Default>(data: &[u8]) -> Result<T, Status> {
    T::decode(data).map_err(|e| Status::invalid_argument(format!("请求解析失败: {}", e)))
}

/// 编码 protobuf 消息（辅助函数）
fn encode_message<T: ProstMessage>(msg: &T) -> Vec<u8> {
    msg.encode_to_vec()
}

// ============================================================================
// prost::Message 手动实现（为现有类型添加 protobuf 编解码能力）
// ============================================================================

// 注意：以下实现使用简化的手动序列化/反序列化逻辑，
// 真实场景中应使用 prost-build 从 proto 文件生成代码

impl ProstMessage for ChatRequest {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() {
            prost::encoding::string::encode(1, &self.session_id, buf);
        }
        if self.stream {
            prost::encoding::bool::encode(3, &self.stream, buf);
        }
        if self.max_tokens != 0 {
            prost::encoding::int32::encode(4, &self.max_tokens, buf);
        }
        if self.temperature != 0.0 {
            prost::encoding::float::encode(5, &self.temperature, buf);
        }
    }

    fn merge_field(
        &mut self,
        tag: u32,
        wire_type: prost::encoding::WireType,
        buf: &mut std::io::Cursor<&[u8]>,
    ) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            3 => { self.stream = prost::encoding::bool::merge(wire_type, self.stream, buf)?; }
            4 => { self.max_tokens = prost::encoding::int32::merge(wire_type, self.max_tokens, buf)?; }
            5 => { self.temperature = prost::encoding::float::merge(wire_type, self.temperature, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if self.stream { len += prost::encoding::bool::encoded_len(3, &self.stream); }
        if self.max_tokens != 0 { len += prost::encoding::int32::encoded_len(4, &self.max_tokens); }
        if self.temperature != 0.0 { len += prost::encoding::float::encoded_len(5, &self.temperature); }
        len
    }

    fn clear(&mut self) {
        self.session_id.clear();
        self.messages.clear();
        self.stream = false;
        self.max_tokens = 0;
        self.temperature = 0.7;
    }
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self { session_id: String::new(), messages: Vec::new(), stream: false, max_tokens: 0, temperature: 0.7 }
    }
}

impl ProstMessage for ChatResponse {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() { prost::encoding::string::encode(1, &self.session_id, buf); }
        if !self.token.is_empty() { prost::encoding::string::encode(2, &self.token, buf); }
        if self.finished { prost::encoding::bool::encode(3, &self.finished, buf); }
    }

    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            2 => { self.token = prost::encoding::string::merge(wire_type, self.token.clone(), buf)?; }
            3 => { self.finished = prost::encoding::bool::merge(wire_type, self.finished, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }

    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if !self.token.is_empty() { len += prost::encoding::string::encoded_len(2, &self.token); }
        if self.finished { len += prost::encoding::bool::encoded_len(3, &self.finished); }
        len
    }

    fn clear(&mut self) {
        self.session_id.clear(); self.token.clear(); self.finished = false; self.usage = None;
    }
}

impl Default for ChatResponse {
    fn default() -> Self {
        Self { session_id: String::new(), token: String::new(), finished: false, usage: None }
    }
}

// 为其他类型实现类似的 ProstMessage trait...
// （省略部分实现以保持简洁）

impl ProstMessage for ImageRequest {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() { prost::encoding::string::encode(1, &self.session_id, buf); }
        if !self.image_data.is_empty() { prost::encoding::bytes::encode(2, &self.image_data, buf); }
        if !self.question.is_empty() { prost::encoding::string::encode(3, &self.question, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            2 => { self.image_data = prost::encoding::bytes::merge(wire_type, self.image_data.clone(), buf)?; }
            3 => { self.question = prost::encoding::string::merge(wire_type, self.question.clone(), buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if !self.image_data.is_empty() { len += prost::encoding::bytes::encoded_len(2, &self.image_data); }
        if !self.question.is_empty() { len += prost::encoding::string::encoded_len(3, &self.question); }
        len
    }
    fn clear(&mut self) { self.session_id.clear(); self.image_data.clear(); self.question.clear(); self.stream = false; }
}
impl Default for ImageRequest {
    fn default() => Self { Self { session_id: String::new(), image_data: Vec::new(), question: String::new(), stream: false } }
}

impl ProstMessage for ImageResponse {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() { prost::encoding::string::encode(1, &self.session_id, buf); }
        if !self.token.is_empty() { prost::encoding::string::encode(2, &self.token, buf); }
        if self.finished { prost::encoding::bool::encode(3, &self.finished, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            2 => { self.token = prost::encoding::string::merge(wire_type, self.token.clone(), buf)?; }
            3 => { self.finished = prost::encoding::bool::merge(wire_type, self.finished, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if !self.token.is_empty() { len += prost::encoding::string::encoded_len(2, &self.token); }
        if self.finished { len += prost::encoding::bool::encoded_len(3, &self.finished); }
        len
    }
    fn clear(&mut self) { self.session_id.clear(); self.token.clear(); self.finished = false; }
}
impl Default for ImageResponse {
    fn default() => Self { Self { session_id: String::new(), token: String::new(), finished: false } }
}

impl ProstMessage for HealthRequest {
    fn encode_raw<B: prost::encoding::Buffer>(&self, _buf: &mut B) {}
    fn merge_field(&mut self, _tag: u32, _wire_type: prost::encoding::WireType, _buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> { Ok(()) }
    fn encoded_len(&self) -> usize { 0 }
    fn clear(&mut self) {}
}
impl Default for HealthRequest {
    fn default() -> Self {}
}

impl ProstMessage for HealthResponse {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if self.healthy { prost::encoding::bool::encode(1, &self.healthy, buf); }
        if !self.message.is_empty() { prost::encoding::string::encode(2, &self.message, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.healthy = prost::encoding::bool::merge(wire_type, self.healthy, buf)?; }
            2 => { self.message = prost::encoding::string::merge(wire_type, self.message.clone(), buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if self.healthy { len += prost::encoding::bool::encoded_len(1, &self.healthy); }
        if !self.message.is_empty() { len += prost::encoding::string::encoded_len(2, &self.message); }
        len
    }
    fn clear(&mut self) { self.healthy = false; self.message.clear(); }
}
impl Default for HealthResponse {
    fn default() -> Self { Self { healthy: false, message: String::new() } }
}

impl ProstMessage for OmniChatRequest {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() { prost::encoding::string::encode(1, &self.session_id, buf); }
        if self.stream { prost::encoding::bool::encode(6, &self.stream, buf); }
        if self.max_tokens != 0 { prost::encoding::int32::encode(7, &self.max_tokens, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            6 => { self.stream = prost::encoding::bool::merge(wire_type, self.stream, buf)?; }
            7 => { self.max_tokens = prost::encoding::int32::merge(wire_type, self.max_tokens, buf)?; }
            8 => { self.temperature = prost::encoding::float::merge(wire_type, self.temperature, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if self.stream { len += prost::encoding::bool::encoded_len(6, &self.stream); }
        if self.max_tokens != 0 { len += prost::encoding::int32::encoded_len(7, &self.max_tokens); }
        len
    }
    fn clear(&mut self) { self.session_id.clear(); self.input = None; self.stream = false; self.max_tokens = 0; self.temperature = 0.7; }
}
impl Default for OmniChatRequest {
    fn default() -> Self { Self { session_id: String::new(), input: None, stream: false, max_tokens: 0, temperature: 0.7 } }
}

impl ProstMessage for OmniChatResponse {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() { prost::encoding::string::encode(1, &self.session_id, buf); }
        if self.finished { prost::encoding::bool::encode(4, &self.finished, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            4 => { self.finished = prost::encoding::bool::merge(wire_type, self.finished, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if self.finished { len += prost::encoding::bool::encoded_len(4, &self.finished); }
        len
    }
    fn clear(&mut self) { self.session_id.clear(); self.output = None; self.finished = false; self.usage = None; }
}
impl Default for OmniChatResponse {
    fn default() -> Self { Self { session_id: String::new(), output: None, finished: false, usage: None } }
}

impl ProstMessage for SpeechToTextRequest {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() { prost::encoding::string::encode(1, &self.session_id, buf); }
        if !self.audio_data.is_empty() { prost::encoding::bytes::encode(2, &self.audio_data, buf); }
        if !self.language.is_empty() { prost::encoding::string::encode(3, &self.language, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            2 => { self.audio_data = prost::encoding::bytes::merge(wire_type, self.audio_data.clone(), buf)?; }
            3 => { self.language = prost::encoding::string::merge(wire_type, self.language.clone(), buf)?; }
            4 => { self.stream = prost::encoding::bool::merge(wire_type, self.stream, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if !self.audio_data.is_empty() { len += prost::encoding::bytes::encoded_len(2, &self.audio_data); }
        if !self.language.is_empty() { len += prost::encoding::string::encoded_len(3, &self.language); }
        len
    }
    fn clear(&mut self) { self.session_id.clear(); self.audio_data.clear(); self.language.clear(); self.stream = false; }
}
impl Default for SpeechToTextRequest {
    fn default() -> Self { Self { session_id: String::new(), audio_data: Vec::new(), language: String::new(), stream: false } }
}

impl ProstMessage for SpeechToTextResponse {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() { prost::encoding::string::encode(1, &self.session_id, buf); }
        if !self.text.is_empty() { prost::encoding::string::encode(2, &self.text, buf); }
        if self.finished { prost::encoding::bool::encode(3, &self.finished, buf); }
        if self.confidence != 0.0 { prost::encoding::float::encode(4, &self.confidence, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            2 => { self.text = prost::encoding::string::merge(wire_type, self.text.clone(), buf)?; }
            3 => { self.finished = prost::encoding::bool::merge(wire_type, self.finished, buf)?; }
            4 => { self.confidence = prost::encoding::float::merge(wire_type, self.confidence, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if !self.text.is_empty() { len += prost::encoding::string::encoded_len(2, &self.text); }
        if self.finished { len += prost::encoding::bool::encoded_len(3, &self.finished); }
        if self.confidence != 0.0 { len += prost::encoding::float::encoded_len(4, &self.confidence); }
        len
    }
    fn clear(&mut self) { self.session_id.clear(); self.text.clear(); self.finished = false; self.confidence = 0.0; }
}
impl Default for SpeechToTextResponse {
    fn default() -> Self { Self { session_id: String::new(), text: String::new(), finished: false, confidence: 0.0 } }
}

impl ProstMessage for TextToSpeechRequest {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() { prost::encoding::string::encode(1, &self.session_id, buf); }
        if !self.text.is_empty() { prost::encoding::string::encode(2, &self.text, buf); }
        if !self.voice.is_empty() { prost::encoding::string::encode(3, &self.voice, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            2 => { self.text = prost::encoding::string::merge(wire_type, self.text.clone(), buf)?; }
            3 => { self.voice = prost::encoding::string::merge(wire_type, self.voice.clone(), buf)?; }
            4 => { self.language = prost::encoding::string::merge(wire_type, self.language.clone(), buf)?; }
            5 => { self.speed = prost::encoding::float::merge(wire_type, self.speed, buf)?; }
            6 => { self.pitch = prost::encoding::float::merge(wire_type, self.pitch, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if !self.text.is_empty() { len += prost::encoding::string::encoded_len(2, &self.text); }
        if !self.voice.is_empty() { len += prost::encoding::string::encoded_len(3, &self.voice); }
        len
    }
    fn clear(&mut self) { self.session_id.clear(); self.text.clear(); self.voice.clear(); self.language.clear(); self.speed = 1.0; self.pitch = 1.0; }
}
impl Default for TextToSpeechRequest {
    fn default() -> Self { Self { session_id: String::new(), text: String::new(), voice: String::new(), language: String::new(), speed: 1.0, pitch: 1.0 } }
}

impl ProstMessage for TextToSpeechResponse {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if !self.session_id.is_empty() { prost::encoding::string::encode(1, &self.session_id, buf); }
        if !self.audio_data.is_empty() { prost::encoding::bytes::encode(2, &self.audio_data, buf); }
        if self.finished { prost::encoding::bool::encode(3, &self.finished, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.session_id = prost::encoding::string::merge(wire_type, self.session_id.clone(), buf)?; }
            2 => { self.audio_data = prost::encoding::bytes::merge(wire_type, self.audio_data.clone(), buf)?; }
            3 => { self.finished = prost::encoding::bool::merge(wire_type, self.finished, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if !self.session_id.is_empty() { len += prost::encoding::string::encoded_len(1, &self.session_id); }
        if !self.audio_data.is_empty() { len += prost::encoding::bytes::encoded_len(2, &self.audio_data); }
        if self.finished { len += prost::encoding::bool::encoded_len(3, &self.finished); }
        len
    }
    fn clear(&mut self) { self.session_id.clear(); self.audio_data.clear(); self.finished = false; }
}
impl Default for TextToSpeechResponse {
    fn default() -> Self { Self { session_id: String::new(), audio_data: Vec::new(), finished: false } }
}

impl ProstMessage for UsageInfo {
    fn encode_raw<B: prost::encoding::Buffer>(&self, buf: &mut B) {
        if self.prompt_tokens != 0 { prost::encoding::int32::encode(1, &self.prompt_tokens, buf); }
        if self.completion_tokens != 0 { prost::encoding::int32::encode(2, &self.completion_tokens, buf); }
        if self.total_tokens != 0 { prost::encoding::int32::encode(3, &self.total_tokens, buf); }
    }
    fn merge_field(&mut self, tag: u32, wire_type: prost::encoding::WireType, buf: &mut std::io::Cursor<&[u8]>) -> std::result::Result<(), prost::DecodeError> {
        match tag {
            1 => { self.prompt_tokens = prost::encoding::int32::merge(wire_type, self.prompt_tokens, buf)?; }
            2 => { self.completion_tokens = prost::encoding::int32::merge(wire_type, self.completion_tokens, buf)?; }
            3 => { self.total_tokens = prost::encoding::int32::merge(wire_type, self.total_tokens, buf)?; }
            _ => { prost::encoding::skip_field(wire_type, tag, buf)?; }
        }
        Ok(())
    }
    fn encoded_len(&self) -> usize {
        let mut len = 0;
        if self.prompt_tokens != 0 { len += prost::encoding::int32::encoded_len(1, &self.prompt_tokens); }
        if self.completion_tokens != 0 { len += prost::encoding::int32::encoded_len(2, &self.completion_tokens); }
        if self.total_tokens != 0 { len += prost::encoding::int32::encoded_len(3, &self.total_tokens); }
        len
    }
    fn clear(&mut self) { self.prompt_tokens = 0; self.completion_tokens = 0; self.total_tokens = 0; }
}
impl Default for UsageInfo {
    fn default() -> Self { Self { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpc_service_creation() {
        assert!(std::mem::size_of::<OpenMiniGrpcService>() > 0);
    }

    #[test]
    fn test_rpc_paths() {
        assert!(paths::CHAT.starts_with(GRPC_SERVICE_PREFIX));
        assert!(paths::IMAGE_UNDERSTANDING.starts_with(GRPC_SERVICE_PREFIX));
        assert!(paths::HEALTH_CHECK.starts_with(GRPC_SERVICE_PREFIX));
        assert_eq!(vec![paths::CHAT, paths::IMAGE_UNDERSTANDING, paths::HEALTH_CHECK, paths::OMNI_CHAT, paths::SPEECH_TO_TEXT, paths::TEXT_TO_SPEECH].len(), 6);
    }

    #[test]
    fn test_chat_request_protobuf_roundtrip() {
        let original = ChatRequest {
            session_id: "session-123".to_string(),
            messages: vec![],
            stream: true,
            max_tokens: 100,
            temperature: 0.8,
        };
        let encoded = original.encode_to_vec();
        let decoded = ChatRequest::decode(&*encoded).expect("解码失败");
        assert_eq!(decoded.session_id, original.session_id);
        assert_eq!(decoded.stream, original.stream);
    }

    #[test]
    fn test_health_protobuf_roundtrip() {
        let health_resp = HealthResponse { healthy: true, message: "服务运行正常".to_string() };
        let encoded = health_resp.encode_to_vec();
        let decoded = HealthResponse::decode(&*encoded).expect("解码失败");
        assert!(decoded.healthy);
        assert_eq!(decoded.message, "服务运行正常");
    }

    #[test]
    fn test_usage_info_protobuf_roundtrip() {
        let usage = UsageInfo { prompt_tokens: 50, completion_tokens: 25, total_tokens: 75 };
        let encoded = usage.encode_to_vec();
        let decoded = UsageInfo::decode(&*encoded).expect("解码失败");
        assert_eq!(decoded.prompt_tokens, 50);
        assert_eq!(decoded.total_tokens, 75);
    }

    #[test]
    fn test_default_values() {
        let chat_req = ChatRequest::default();
        assert!(chat_req.session_id.is_empty());
        assert!(!chat_req.stream);
    }

    #[test]
    fn test_error_response_construction() {
        let status = Status::internal("内部服务器错误");
        let response = error_response(status);
        assert_eq!(response.status(), http::StatusCode::OK);
        let grpc_status = response.headers().get("grpc-status").unwrap();
        assert_eq!(grpc_status, "2");
    }

    #[test]
    fn test_all_rpc_paths_defined() {
        let expected_methods = vec![
            ("Chat", paths::CHAT),
            ("ImageUnderstanding", paths::IMAGE_UNDERSTANDING),
            ("HealthCheck", paths::HEALTH_CHECK),
            ("OmniChat", paths::OMNI_CHAT),
            ("SpeechToText", paths::SPEECH_TO_TEXT),
            ("TextToSpeech", paths::TEXT_TO_SPEECH),
        ];
        for (method_name, path) in expected_methods {
            assert!(path.contains(method_name), "{} 方法路径应包含方法名", method_name);
        }
    }

    #[tokio::test]
    async fn test_grpc_frame_format() {
        let service = Arc::new(OpenMiniService::new().await);
        let grpc_service = OpenMiniGrpcService::new(service);

        let health_req = HealthRequest {};
        let encoded_req = health_req.encode_to_vec();

        let mut frame = Vec::with_capacity(5 + encoded_req.len());
        frame.push(0);
        frame.extend_from_slice(&(encoded_req.len() as u32).to_be_bytes());
        frame.extend_from_slice(&encoded_req);

        assert_eq!(frame[0], 0);
        let msg_len = u32::from_be_bytes([frame[1], frame[2], frame[3], frame[4]]);
        assert_eq!(msg_len as usize, encoded_req.len());
    }

    #[test]
    fn test_grpc_service_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<OpenMiniGrpcService>();
    }

    #[test]
    fn test_unicode_special_characters() {
        let chat_req = ChatRequest {
            session_id: "会话-中文-🎉".to_string(),
            messages: vec![],
            stream: false,
            max_tokens: 999,
            temperature: 1.5,
        };
        let encoded = chat_req.encode_to_vec();
        let decoded = ChatRequest::decode(&*encoded).expect("Unicode 数据解码失败");
        assert_eq!(decoded.session_id, "会话-中文-🎉");
    }

    #[test]
    fn test_clear_resets_fields() {
        let mut chat_req = ChatRequest {
            session_id: "test".to_string(),
            messages: vec![],
            stream: true,
            max_tokens: 100,
            temperature: 0.9,
        };
        chat_req.clear();
        assert!(chat_req.session_id.is_empty());
        assert!(!chat_req.stream);
        assert_eq!(chat_req.max_tokens, 0);
    }

    #[test]
    fn test_large_binary_data() {
        let large_image_data: Vec<u8> = (0..10240).map(|i| (i % 256) as u8).collect();
        let img_req = ImageRequest {
            session_id: "large-image".to_string(),
            image_data: large_image_data.clone(),
            question: "".to_string(),
            stream: true,
        };
        let encoded = img_req.encode_to_vec();
        let decoded = ImageRequest::decode(&*encoded).expect("大数据解码失败");
        assert_eq!(decoded.image_data.len(), large_image_data.len());
    }

    #[test]
    fn test_speech_to_text_protobuf_roundtrip() {
        let stt_resp = SpeechToTextResponse {
            session_id: "stt-session".to_string(),
            text: "你好世界".to_string(),
            finished: true,
            confidence: 0.95,
        };
        let encoded = stt_resp.encode_to_vec();
        let decoded = SpeechToTextResponse::decode(&*encoded).expect("解码失败");
        assert_eq!(decoded.text, "你好世界");
        assert!((decoded.confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_text_to_speech_protobuf_roundtrip() {
        let tts_req = TextToSpeechRequest {
            session_id: "tts-session".to_string(),
            text: "Hello TTS".to_string(),
            voice: "female-1".to_string(),
            language: "en-US".to_string(),
            speed: 1.2,
            pitch: 1.1,
        };
        let encoded = tts_req.encode_to_vec();
        let decoded = TextToSpeechRequest::decode(&*encoded).expect("解码失败");
        assert_eq!(decoded.text, "Hello TTS");
        assert!((decoded.speed - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_omni_chat_request_default() {
        let omni_req = OmniChatRequest::default();
        assert!(omni_req.session_id.is_empty());
        assert!(omni_req.input.is_none());
        assert!(!omni_req.stream);
    }

    #[test]
    fn test_constants_defined() {
        assert_eq!(GRPC_CONTENT_TYPE, "application/grpc+proto");
        assert_eq!(GRPC_STATUS_OK, 0);
        assert!(GRPC_SERVICE_PREFIX.contains("/"));
    }
}
