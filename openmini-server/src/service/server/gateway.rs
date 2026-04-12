//! 网关服务 - TCP 服务端实现
//!
//! 提供 TCP 服务端网关，支持:
//! - 多连接管理
//! - 请求解析和路由
//! - 会话管理
//! - 流式响应
//!
//! ## 架构说明 (TaskScheduler 模式)
//!
//! 本网关已迁移至 TaskScheduler 单进程架构：
//! - 不再依赖 ThreadPool（已移除）
//! - 使用 AsyncInferencePool 进行请求排队和批处理
//! - 实际任务调度由外层 TaskScheduler 负责

#![allow(dead_code)]

use crate::service::worker::{AsyncInferencePool, InferenceTask};
use bytes::{Bytes, BytesMut};
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::Sender;
use tokio::time::timeout;

use super::connection::ConnectionPool;

/// 最大消息大小 (16MB)
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;
/// 默认缓冲区大小 (64KB)
const DEFAULT_BUFFER_SIZE: usize = 64 * 1024;
/// 请求超时时间
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
/// 读取超时时间
const READ_TIMEOUT: Duration = Duration::from_secs(30);
/// 写入超时时间
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);

/// 网关结果类型
pub type Result<T> = std::result::Result<T, GatewayError>;

/// 网关错误类型
#[derive(Debug)]
pub enum GatewayError {
    /// IO 错误
    Io(tokio::io::Error),
    /// 解析错误
    Parse(String),
    /// 超时错误
    Timeout(String),
    /// 连接池耗尽
    ConnectionPoolExhausted,
    /// 无效请求
    InvalidRequest(String),
    /// 内部错误
    Internal(String),
    /// 网关关闭
    Shutdown,
}

impl std::fmt::Display for GatewayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GatewayError::Io(e) => write!(f, "IO error: {}", e),
            GatewayError::Parse(s) => write!(f, "Parse error: {}", s),
            GatewayError::Timeout(s) => write!(f, "Timeout: {}", s),
            GatewayError::ConnectionPoolExhausted => write!(f, "Connection pool exhausted"),
            GatewayError::InvalidRequest(s) => write!(f, "Invalid request: {}", s),
            GatewayError::Internal(s) => write!(f, "Internal error: {}", s),
            GatewayError::Shutdown => write!(f, "Gateway shutdown"),
        }
    }
}

impl std::error::Error for GatewayError {}

impl From<tokio::io::Error> for GatewayError {
    fn from(e: tokio::io::Error) -> Self {
        GatewayError::Io(e)
    }
}

/// 请求类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestType {
    /// 聊天请求
    Chat,
    /// 图像理解请求
    ImageUnderstanding,
    /// 图像理解流式请求
    ImageUnderstandingStream,
    /// 健康检查
    HealthCheck,
    /// 未知类型
    Unknown,
}

/// 请求结构
#[derive(Debug)]
pub struct Request {
    /// 请求类型
    pub request_type: RequestType,
    /// 会话 ID
    pub session_id: String,
    /// 请求负载
    pub payload: Bytes,
    /// 是否流式响应
    pub stream: bool,
    /// 创建时间
    pub created_at: std::time::Instant,
}

impl Request {
    /// 创建新请求
    pub fn new(
        request_type: RequestType,
        session_id: String,
        payload: Bytes,
        stream: bool,
    ) -> Self {
        Self {
            request_type,
            session_id,
            payload,
            stream,
            created_at: std::time::Instant::now(),
        }
    }
}

/// 响应结构
#[derive(Debug)]
pub struct Response {
    /// 会话 ID
    pub session_id: String,
    /// 响应负载
    pub payload: Bytes,
    /// 是否完成
    pub finished: bool,
    /// 错误信息
    pub error: Option<String>,
}

impl Response {
    /// 创建新响应
    pub fn new(session_id: String, payload: Bytes, finished: bool) -> Self {
        Self {
            session_id,
            payload,
            finished,
            error: None,
        }
    }

    /// 创建错误响应
    pub fn with_error(session_id: String, error: String) -> Self {
        Self {
            session_id,
            payload: Bytes::new(),
            finished: true,
            error: Some(error),
        }
    }
}

/// 基于原子计数器的连接限制器（替代 Semaphore）
struct ConnectionLimiter {
    current: AtomicU64,
    max_connections: usize,
}

impl ConnectionLimiter {
    fn new(max: usize) -> Self {
        Self {
            current: AtomicU64::new(0),
            max_connections: max,
        }
    }

    fn try_acquire(&self) -> std::result::Result<(), ()> {
        let current = self.current.fetch_add(1, Ordering::Relaxed);
        if current >= self.max_connections as u64 {
            self.current.fetch_sub(1, Ordering::Relaxed);
            Err(())
        } else {
            Ok(())
        }
    }

    fn release(&self) {
        self.current.fetch_sub(1, Ordering::Relaxed);
    }

    fn count(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }
}

/// 网关统计信息
pub struct GatewayStats {
    /// 总连接数
    pub total_connections: AtomicU64,
    /// 活跃连接数
    pub active_connections: AtomicU64,
    /// 总请求数
    pub total_requests: AtomicU64,
    /// 成功请求数
    pub successful_requests: AtomicU64,
    /// 失败请求数
    pub failed_requests: AtomicU64,
    /// 接收字节数
    pub bytes_received: AtomicU64,
    /// 发送字节数
    pub bytes_sent: AtomicU64,
}

impl GatewayStats {
    /// 创建新的统计实例
    pub fn new() -> Self {
        Self {
            total_connections: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
        }
    }

    /// 获取统计快照
    pub fn snapshot(&self) -> GatewayStatsSnapshot {
        GatewayStatsSnapshot {
            total_connections: self.total_connections.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_requests: self.successful_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
        }
    }
}

impl Default for GatewayStats {
    fn default() -> Self {
        Self::new()
    }
}

/// 网关统计快照
#[derive(Debug, Clone)]
pub struct GatewayStatsSnapshot {
    /// 总连接数
    pub total_connections: u64,
    /// 活跃连接数
    pub active_connections: u64,
    /// 总请求数
    pub total_requests: u64,
    /// 成功请求数
    pub successful_requests: u64,
    /// 失败请求数
    pub failed_requests: u64,
    /// 接收字节数
    pub bytes_received: u64,
    /// 发送字节数
    pub bytes_sent: u64,
}

/// TCP 网关服务
///
/// 监听 TCP 连接，解析请求并通过 AsyncInferencePool 分发处理。
/// (TaskScheduler 模式: 不再使用 ThreadPool)
pub struct Gateway {
    /// 监听地址
    addr: SocketAddr,
    /// 连接池
    connection_pool: ConnectionPool,
    /// 统计信息
    stats: Arc<GatewayStats>,
    /// 关闭标志
    shutdown_flag: Arc<AtomicBool>,
    /// 最大并发连接数
    max_concurrent_connections: usize,
    /// 连接限制器
    connection_limiter: Arc<ConnectionLimiter>,
    /// 会话映射
    sessions: Arc<DashMap<String, Sender<Response>>>,
    /// 缓冲区池
    buffer_pool: Arc<BufferPool>,
    /// 异步推理任务池
    inference_pool: Arc<AsyncInferencePool>,
}

impl Gateway {
    /// 创建新的网关 (TaskScheduler 模式)
    pub fn new(addr: SocketAddr, inference_pool: Arc<AsyncInferencePool>) -> Self {
        Self::with_options(addr, inference_pool, 100, 100)
    }

    /// 使用自定义选项创建网关
    pub fn with_options(
        addr: SocketAddr,
        inference_pool: Arc<AsyncInferencePool>,
        max_connections: usize,
        max_concurrent: usize,
    ) -> Self {
        Self {
            addr,
            connection_pool: ConnectionPool::new(max_connections),
            stats: Arc::new(GatewayStats::new()),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            max_concurrent_connections: max_concurrent,
            connection_limiter: Arc::new(ConnectionLimiter::new(max_concurrent)),
            sessions: Arc::new(DashMap::new()),
            buffer_pool: Arc::new(BufferPool::new(DEFAULT_BUFFER_SIZE)),
            inference_pool,
        }
    }

    /// 运行网关主循环
    pub async fn run(&self) -> Result<()> {
        let listener = TcpListener::bind(self.addr).await?;

        tracing::info!("Gateway listening on {}", self.addr);

        loop {
            if self.shutdown_flag.load(Ordering::Relaxed) {
                tracing::info!("Gateway shutting down");
                break;
            }

            let accept_result = timeout(Duration::from_millis(100), listener.accept()).await;

            match accept_result {
                Ok(Ok((stream, peer_addr))) => {
                    if self.connection_limiter.try_acquire().is_err() {
                        tracing::warn!(
                            "Connection limit reached ({}/{}), rejecting from {}",
                            self.connection_limiter.count(),
                            self.max_concurrent_connections,
                            peer_addr
                        );
                        continue;
                    }

                    self.stats.total_connections.fetch_add(1, Ordering::Relaxed);
                    self.stats
                        .active_connections
                        .fetch_add(1, Ordering::Relaxed);

                    let stats = Arc::clone(&self.stats);
                    let shutdown_flag = Arc::clone(&self.shutdown_flag);
                    let sessions = self.sessions.clone();
                    
                    let connection_limiter = Arc::clone(&self.connection_limiter);
                    let buffer_pool = Arc::clone(&self.buffer_pool);
                    let inference_pool = Arc::clone(&self.inference_pool);

                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(
                            stream,
                            peer_addr,
                            stats,
                            shutdown_flag,
                            sessions,
                            connection_limiter,
                            buffer_pool,
                            inference_pool,
                        )
                        .await
                        {
                            tracing::error!("Connection error from {}: {}", peer_addr, e);
                        }
                    });
                }
                Ok(Err(e)) => {
                    tracing::error!("Accept error: {}", e);
                }
                Err(_) => {
                    continue;
                }
            }
        }

        Ok(())
    }

    /// 处理单个连接
    async fn handle_connection(
        mut stream: TcpStream,
        peer_addr: SocketAddr,
        stats: Arc<GatewayStats>,
        shutdown_flag: Arc<AtomicBool>,
        sessions: Arc<DashMap<String, Sender<Response>>>,
        connection_limiter: Arc<ConnectionLimiter>,
        buffer_pool: Arc<BufferPool>,
        inference_pool: Arc<AsyncInferencePool>,
    ) -> Result<()> {
        tracing::debug!("New connection from {}", peer_addr);

        let mut buffer = buffer_pool.acquire();
        let mut session_id: Option<String> = None;

        loop {
            if shutdown_flag.load(Ordering::Relaxed) {
                break;
            }

            let read_result = timeout(READ_TIMEOUT, async {
                let mut read_buf = vec![0u8; DEFAULT_BUFFER_SIZE];
                let n = stream.read(&mut read_buf).await?;
                Ok::<_, tokio::io::Error>((n, read_buf))
            })
            .await;

            match read_result {
                Ok(Ok((0, _))) => {
                    tracing::debug!("Connection closed by peer {}", peer_addr);
                    break;
                }
                Ok(Ok((n, read_buf))) => {
                    stats.bytes_received.fetch_add(n as u64, Ordering::Relaxed);
                    buffer.extend_from_slice(&read_buf[..n]);

                    while let Some(request) = Self::parse_request(&mut buffer)? {
                        stats.total_requests.fetch_add(1, Ordering::Relaxed);

                        if session_id.is_none() {
                            session_id = Some(request.session_id.clone());
                        }

                        match Self::dispatch_request(request, &inference_pool).await {
                            Ok(response) => {
                                let write_result = timeout(WRITE_TIMEOUT, async {
                                    stream.write_all(&response.payload).await
                                })
                                .await;

                                match write_result {
                                    Ok(Ok(_)) => {
                                        stats.bytes_sent.fetch_add(
                                            response.payload.len() as u64,
                                            Ordering::Relaxed,
                                        );
                                        stats.successful_requests.fetch_add(1, Ordering::Relaxed);
                                        stream.flush().await?;
                                    }
                                    Ok(Err(e)) => {
                                        stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                                        return Err(GatewayError::Io(e));
                                    }
                                    Err(_) => {
                                        stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                                        return Err(GatewayError::Timeout(
                                            "Write timeout".to_string(),
                                        ));
                                    }
                                }
                            }
                            Err(e) => {
                                stats.failed_requests.fetch_add(1, Ordering::Relaxed);
                                tracing::error!("Request dispatch error: {}", e);
                            }
                        }
                    }
                }
                Ok(Err(e)) => {
                    tracing::error!("Read error from {}: {}", peer_addr, e);
                    break;
                }
                Err(_) => {
                    continue;
                }
            }
        }

        if let Some(sid) = session_id {
            sessions.remove(&sid);
        }

        connection_limiter.release();
        stats.active_connections.fetch_sub(1, Ordering::Relaxed);
        buffer_pool.release(buffer);
        tracing::debug!("Connection from {} closed", peer_addr);

        Ok(())
    }

    /// 解析请求
    fn parse_request(buffer: &mut BytesMut) -> Result<Option<Request>> {
        if buffer.len() < 4 {
            return Ok(None);
        }

        let len_bytes = [buffer[0], buffer[1], buffer[2], buffer[3]];
        let payload_len = u32::from_be_bytes(len_bytes) as usize;

        if payload_len > MAX_MESSAGE_SIZE {
            buffer.clear();
            return Err(GatewayError::Parse(format!(
                "Message too large: {} bytes",
                payload_len
            )));
        }

        if buffer.len() < 4 + payload_len {
            return Ok(None);
        }

        let payload = buffer.split_to(4 + payload_len);
        let payload_bytes = Bytes::copy_from_slice(&payload[4..]);

        let request_type_byte = if !payload_bytes.is_empty() {
            payload_bytes[0]
        } else {
            0
        };

        let request_type = match request_type_byte {
            1 => RequestType::Chat,
            2 => RequestType::ImageUnderstanding,
            3 => RequestType::ImageUnderstandingStream,
            4 => RequestType::HealthCheck,
            _ => RequestType::Unknown,
        };

        let session_id = if payload_bytes.len() > 37 {
            String::from_utf8_lossy(&payload_bytes[1..37]).to_string()
        } else {
            format!("session-{}", uuid::Uuid::new_v4())
        };

        let stream = payload_bytes.len() > 38 && payload_bytes[38] == 1;

        let actual_payload = if payload_bytes.len() > 39 {
            Bytes::copy_from_slice(&payload_bytes[39..])
        } else {
            Bytes::new()
        };

        Ok(Some(Request::new(
            request_type,
            session_id,
            actual_payload,
            stream,
        )))
    }

    /// 分发请求到处理器
    async fn dispatch_request(request: Request, inference_pool: &Arc<AsyncInferencePool>) -> Result<Response> {
        match request.request_type {
            RequestType::Chat => Self::handle_chat_request(request, inference_pool).await,
            RequestType::ImageUnderstanding => {
                Self::handle_image_request(request, inference_pool, false).await
            }
            RequestType::ImageUnderstandingStream => {
                Self::handle_image_request(request, inference_pool, true).await
            }
            RequestType::HealthCheck => Ok(Response::new(
                request.session_id,
                Bytes::from_static(b"OK"),
                true,
            )),
            RequestType::Unknown => Err(GatewayError::InvalidRequest(
                "Unknown request type".to_string(),
            )),
        }
    }

    /// 处理聊天请求
    async fn handle_chat_request(
        request: Request,
        inference_pool: &Arc<AsyncInferencePool>,
    ) -> Result<Response> {
        let session_id = request.session_id.clone();
        let prompt = String::from_utf8_lossy(&request.payload).to_string();

        let task = InferenceTask {
            prompt,
            session_id: session_id.clone(),
            max_tokens: Some(2048),
            temperature: None,
        };

        match inference_pool.submit(task).await {
            Ok(result) => Ok(Response::new(session_id, Bytes::from(result.text), result.finished)),
            Err(e) => Ok(Response::with_error(session_id, e)),
        }
    }

    /// 处理图像理解请求
    async fn handle_image_request(
        request: Request,
        inference_pool: &Arc<AsyncInferencePool>,
        _stream: bool,
    ) -> Result<Response> {
        let session_id = request.session_id.clone();
        let prompt = String::from_utf8_lossy(&request.payload).to_string();

        let task = InferenceTask {
            prompt,
            session_id: session_id.clone(),
            max_tokens: Some(2048),
            temperature: None,
        };

        match inference_pool.submit(task).await {
            Ok(result) => Ok(Response::new(session_id, Bytes::from(result.text), result.finished)),
            Err(e) => Ok(Response::with_error(session_id, e)),
        }
    }

    /// 关闭网关
    pub fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
    }

    /// 获取统计信息
    pub fn stats(&self) -> Arc<GatewayStats> {
        Arc::clone(&self.stats)
    }

    /// 获取连接池
    pub fn connection_pool(&self) -> &ConnectionPool {
        &self.connection_pool
    }

    /// 检查是否运行中
    pub fn is_running(&self) -> bool {
        !self.shutdown_flag.load(Ordering::Relaxed)
    }

    /// 获取最大并发连接数
    pub fn max_concurrent_connections(&self) -> usize {
        self.max_concurrent_connections
    }
}

/// 网关构建器
pub struct GatewayBuilder {
    addr: SocketAddr,
    max_connections: usize,
    max_concurrent: usize,
    idle_timeout: Duration,
    connection_timeout: Duration,
}

impl GatewayBuilder {
    /// 创建新的构建器
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            max_connections: 100,
            max_concurrent: 100,
            idle_timeout: Duration::from_secs(300),
            connection_timeout: Duration::from_secs(30),
        }
    }

    /// 设置最大连接数
    pub fn max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    /// 设置最大并发数
    pub fn max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// 设置空闲超时
    pub fn idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// 设置连接超时
    pub fn connection_timeout(mut self, timeout: Duration) -> Self {
        self.connection_timeout = timeout;
        self
    }

    /// 构建网关 (TaskScheduler 模式)
    pub fn build(self, inference_pool: Arc<AsyncInferencePool>) -> Gateway {
        Gateway::with_options(
            self.addr,
            inference_pool,
            self.max_connections,
            self.max_concurrent,
        )
    }
}

/// 零拷贝缓冲区
pub struct ZeroCopyBuffer {
    data: Bytes,
    position: usize,
}

impl ZeroCopyBuffer {
    /// 创建新缓冲区
    pub fn new(data: Bytes) -> Self {
        Self { data, position: 0 }
    }

    /// 获取剩余数据长度
    pub fn remaining(&self) -> usize {
        self.data.len() - self.position
    }

    /// 获取剩余数据切片
    pub fn slice(&self) -> Bytes {
        self.data.slice(self.position..)
    }

    /// 前进指定位置
    pub fn advance(&mut self, n: usize) {
        self.position = (self.position + n).min(self.data.len());
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.position >= self.data.len()
    }
}

/// 全局复用缓冲区池，减少每连接内存分配开销
struct BufferPool {
    free_list: std::sync::Mutex<Vec<BytesMut>>,
    buffer_size: usize,
}

impl BufferPool {
    fn new(buffer_size: usize) -> Self {
        Self {
            free_list: std::sync::Mutex::new(Vec::new()),
            buffer_size,
        }
    }

    fn acquire(&self) -> BytesMut {
        self.free_list.lock().unwrap_or_else(|e| e.into_inner())
            .pop()
            .unwrap_or_else(|| BytesMut::with_capacity(self.buffer_size))
    }

    fn release(&self, mut buf: BytesMut) {
        if buf.capacity() <= self.buffer_size * 2 {
            buf.clear();
            if let Ok(mut list) = self.free_list.lock() {
                if list.len() < 1024 {
                    list.push(buf);
                }
            }
        }
    }
}

/// 响应流
pub struct ResponseStream {
    session_id: String,
    tx: Sender<Response>,
    finished: bool,
}

impl ResponseStream {
    /// 创建新响应流
    pub fn new(session_id: String, tx: Sender<Response>) -> Self {
        Self {
            session_id,
            tx,
            finished: false,
        }
    }

    /// 发送响应
    pub async fn send(&mut self, payload: Bytes, finished: bool) -> Result<()> {
        if self.finished {
            return Err(GatewayError::Internal(
                "Stream already finished".to_string(),
            ));
        }

        let response = Response::new(self.session_id.clone(), payload, finished);
        self.tx
            .send(response)
            .await
            .map_err(|_| GatewayError::Internal("Failed to send response".to_string()))?;

        if finished {
            self.finished = true;
        }

        Ok(())
    }

    /// 检查是否已完成
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gateway_error_display() {
        let err = GatewayError::Parse("test error".to_string());
        assert_eq!(format!("{}", err), "Parse error: test error");
    }

    #[test]
    fn test_request_new() {
        let request = Request::new(
            RequestType::Chat,
            "session-123".to_string(),
            Bytes::from_static(b"test"),
            true,
        );
        assert_eq!(request.request_type, RequestType::Chat);
        assert_eq!(request.session_id, "session-123");
        assert!(request.stream);
    }

    #[test]
    fn test_response_new() {
        let response = Response::new("session-123".to_string(), Bytes::from_static(b"test"), true);
        assert_eq!(response.session_id, "session-123");
        assert!(response.finished);
        assert!(response.error.is_none());
    }

    #[test]
    fn test_response_with_error() {
        let response = Response::with_error("session-123".to_string(), "error message".to_string());
        assert!(response.error.is_some());
        assert!(response.finished);
    }

    #[test]
    fn test_gateway_stats() {
        let stats = GatewayStats::new();
        stats.total_requests.fetch_add(10, Ordering::Relaxed);
        stats.successful_requests.fetch_add(8, Ordering::Relaxed);
        stats.failed_requests.fetch_add(2, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 10);
        assert_eq!(snapshot.successful_requests, 8);
        assert_eq!(snapshot.failed_requests, 2);
    }

    #[test]
    fn test_zero_copy_buffer() {
        let data = Bytes::from_static(b"hello world");
        let mut buffer = ZeroCopyBuffer::new(data);

        assert_eq!(buffer.remaining(), 11);
        assert!(!buffer.is_empty());

        buffer.advance(5);
        assert_eq!(buffer.remaining(), 6);

        let slice = buffer.slice();
        assert_eq!(&slice[..], b" world");
    }

    #[test]
    fn test_gateway_builder() {
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let builder = GatewayBuilder::new(addr)
            .max_connections(50)
            .max_concurrent(50);

        assert_eq!(builder.max_connections, 50);
        assert_eq!(builder.max_concurrent, 50);
    }

    #[test]
    fn test_gateway_error_all_variants() {
        // 测试所有 GatewayError 变体的 Display 和 Debug
        let errors: Vec<GatewayError> = vec![
            GatewayError::Io(tokio::io::Error::new(
                tokio::io::ErrorKind::NotFound,
                "file not found",
            )),
            GatewayError::Parse("invalid format".to_string()),
            GatewayError::Timeout("operation timed out".to_string()),
            GatewayError::ConnectionPoolExhausted,
            GatewayError::InvalidRequest("bad request".to_string()),
            GatewayError::Internal("internal failure".to_string()),
            GatewayError::Shutdown,
        ];

        for error in &errors {
            let display = format!("{}", error);
            let debug = format!("{:?}", error);
            assert!(!display.is_empty(), "Display should not be empty");
            assert!(!debug.is_empty(), "Debug should not be empty");
            // 验证 Debug 有足够的内容
            assert!(
                debug.len() > 5,
                "Debug output should have meaningful content, got: {}",
                debug
            );
        }
    }

    #[test]
    fn test_request_type_all_variants() {
        // 测试所有请求类型
        let types = vec![
            RequestType::Chat,
            RequestType::ImageUnderstanding,
            RequestType::ImageUnderstandingStream,
            RequestType::HealthCheck,
            RequestType::Unknown,
        ];

        for req_type in &types {
            let request = Request::new(
                *req_type,
                "test-session".to_string(),
                Bytes::from_static(b"data"),
                false,
            );
            assert_eq!(request.request_type, *req_type);
            assert!(!request.stream); // 非流式请求
        }

        // 流式请求
        let stream_request = Request::new(
            RequestType::ImageUnderstandingStream,
            "stream-session".to_string(),
            Bytes::from_static(b"stream data"),
            true,
        );
        assert!(stream_request.stream);
    }

    #[test]
    fn test_gateway_stats_default_and_full_snapshot() {
        // 测试默认统计信息
        let stats = GatewayStats::default();
        let snapshot = stats.snapshot();

        assert_eq!(snapshot.total_connections, 0);
        assert_eq!(snapshot.active_connections, 0);
        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.successful_requests, 0);
        assert_eq!(snapshot.failed_requests, 0);
        assert_eq!(snapshot.bytes_received, 0);
        assert_eq!(snapshot.bytes_sent, 0);

        // 更新所有统计字段
        stats.total_connections.fetch_add(100, Ordering::Relaxed);
        stats.active_connections.fetch_add(50, Ordering::Relaxed);
        stats.total_requests.fetch_add(1000, Ordering::Relaxed);
        stats.successful_requests.fetch_add(950, Ordering::Relaxed);
        stats.failed_requests.fetch_add(50, Ordering::Relaxed);
        stats
            .bytes_received
            .fetch_add(1024 * 1024, Ordering::Relaxed); // 1MB
        stats.bytes_sent.fetch_add(512 * 1024, Ordering::Relaxed); // 512KB

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_connections, 100);
        assert_eq!(snapshot.active_connections, 50);
        assert_eq!(snapshot.total_requests, 1000);
        assert_eq!(snapshot.successful_requests, 950);
        assert_eq!(snapshot.failed_requests, 50);
        assert_eq!(snapshot.bytes_received, 1024 * 1024);
        assert_eq!(snapshot.bytes_sent, 512 * 1024);
    }

    #[test]
    fn test_zero_copy_buffer_edge_cases() {
        // 空数据缓冲区
        let empty_data = Bytes::new();
        let buffer = ZeroCopyBuffer::new(empty_data);
        assert_eq!(buffer.remaining(), 0);
        assert!(buffer.is_empty());

        // advance 超过数据长度
        let data = Bytes::from_static(b"short");
        let mut buffer = ZeroCopyBuffer::new(data);
        buffer.advance(100); // 超过长度
        assert!(buffer.is_empty()); // 应该被限制在数据末尾

        // 完全消耗缓冲区
        let data2 = Bytes::from_static(b"exact");
        let mut buffer2 = ZeroCopyBuffer::new(data2);
        buffer2.advance(5); // 正好消耗完
        assert!(buffer2.is_empty());
    }

    #[test]
    fn test_response_stream_finished_state() {
        // 注意：ResponseStream 需要 Sender，这里只测试基本逻辑概念
        // 完整的流测试需要异步运行时
        let response = Response::new(
            "session-test".to_string(),
            Bytes::from_static(b"partial data"),
            false,
        );
        assert!(!response.finished);

        let finished_response = Response::new(
            "session-test".to_string(),
            Bytes::from_static(b"complete data"),
            true,
        );
        assert!(finished_response.finished);
    }

    #[test]
    fn test_gateway_stats_snapshot_clone() {
        // 测试快照的 Clone 特性
        let stats = GatewayStats::new();
        stats.total_requests.fetch_add(42, Ordering::Relaxed);

        let snapshot1 = stats.snapshot();
        let snapshot2 = snapshot1.clone();

        assert_eq!(snapshot1.total_requests, snapshot2.total_requests);
        assert_eq!(snapshot1.successful_requests, snapshot2.successful_requests);
    }

    #[test]
    fn test_request_with_empty_payload() {
        // 测试空负载请求
        let request = Request::new(
            RequestType::Chat,
            "empty-session".to_string(),
            Bytes::new(),
            false,
        );
        assert!(request.payload.is_empty());
        assert_eq!(request.session_id, "empty-session");
    }

    #[test]
    fn test_response_error_message_preserved() {
        // 测试错误消息保留
        let error_msg = "Detailed error information for debugging";
        let response = Response::with_error("error-session".to_string(), error_msg.to_string());

        assert_eq!(response.error.as_deref(), Some(error_msg));
        assert!(response.finished);
        assert!(response.payload.is_empty());
    }

    #[test]
    fn test_gateway_builder_full_options() {
        // 测试构建器的所有选项
        let addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        let builder = GatewayBuilder::new(addr)
            .max_connections(200)
            .max_concurrent(150)
            .idle_timeout(Duration::from_secs(600))
            .connection_timeout(Duration::from_secs(60));

        assert_eq!(builder.addr, addr);
        assert_eq!(builder.max_connections, 200);
        assert_eq!(builder.max_concurrent, 150);
        assert_eq!(builder.idle_timeout, Duration::from_secs(600));
        assert_eq!(builder.connection_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_gateway_stats_atomic_operations() {
        // 测试原子操作的并发安全性（基本验证）
        let stats = Arc::new(GatewayStats::new());

        // 模拟多次原子操作
        for _ in 0..100 {
            stats.total_requests.fetch_add(1, Ordering::Relaxed);
        }

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_requests, 100);
    }

    // ==================== 新增分支覆盖测试 (8个) ====================

    #[test]
    fn test_gateway_error_io_conversion() {
        // 覆盖分支: 从 tokio::io::Error 转换为 GatewayError
        use std::io;

        let io_err = io::Error::new(io::ErrorKind::ConnectionRefused, "connection refused");
        let gateway_err: GatewayError = io_err.into();

        match gateway_err {
            GatewayError::Io(e) => {
                assert_eq!(e.kind(), io::ErrorKind::ConnectionRefused);
                assert!(format!("{}", e).contains("connection refused"));
            }
            _ => panic!("Expected Io variant"),
        }
    }

    #[test]
    fn test_gateway_parse_request_minimal_data() {
        // 覆盖分支: 解析最小有效请求（只有长度头和1字节类型）
        // 格式: [4字节长度] [1字节类型]
        let mut buffer = BytesMut::new();

        // 最小 payload: 1 字节类型 + 36 字节 session + 1 字节 stream flag = 38 字节
        // 但我们测试更小的情况
        let minimal_payload: Vec<u8> = vec![0x01]; // RequestType::Chat
        let len = minimal_payload.len() as u32;
        buffer.extend_from_slice(&len.to_be_bytes());
        buffer.extend_from_slice(&minimal_payload);

        let result = Gateway::parse_request(&mut buffer).unwrap();
        assert!(result.is_some());
        let request = result.unwrap();
        assert_eq!(request.request_type, RequestType::Chat);
        // session_id 应该是自动生成的 UUID（因为数据不足 37 字节）
        assert!(request.session_id.starts_with("session-"));
    }

    #[test]
    fn test_gateway_parse_request_oversized_message() {
        // 覆盖分支: 超大消息拒绝
        let mut buffer = BytesMut::new();

        // 设置一个超过 MAX_MESSAGE_SIZE (16MB) 的长度
        let oversized_len = (MAX_MESSAGE_SIZE + 1) as u32;
        buffer.extend_from_slice(&oversized_len.to_be_bytes());

        // 填充一些数据使长度有效
        buffer.extend_from_slice(&[0u8; 4]);

        let result = Gateway::parse_request(&mut buffer);
        assert!(result.is_err());

        match result.unwrap_err() {
            GatewayError::Parse(msg) => {
                assert!(msg.contains("too large"));
            }
            _ => panic!("Expected Parse error"),
        }

        // buffer 应该被清空
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_gateway_parse_request_incomplete_data() {
        // 覆盖分支: 不完整的数据返回 None（等待更多数据）
        let mut buffer = BytesMut::new();

        // 只有 2 字节，不足以读取长度头 (需要 4 字节)
        buffer.extend_from_slice(&[0x00, 0x01]);

        let result = Gateway::parse_request(&mut buffer);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // 需要更多数据

        // 现在有 3 字节
        buffer.extend_from_slice(&[0x02]);
        let result2 = Gateway::parse_request(&mut buffer);
        assert!(result2.is_ok());
        assert!(result2.unwrap().is_none()); // 仍然不够

        // 添加第 4 字节，但声明一个很大的长度
        buffer.extend_from_slice(&[0x00, 0x01]); // 长度 = 65536 (太大)
        let result3 = Gateway::parse_request(&mut buffer);
        // 数据不完整，应该返回 None
        assert!(result3.is_ok());
        assert!(result3.unwrap().is_none());
    }

    #[test]
    fn test_gateway_request_type_parsing_all_types() {
        // 覆盖分支: 所有请求类型的字节解析
        let type_bytes: Vec<(u8, RequestType)> = vec![
            (0x01, RequestType::Chat),
            (0x02, RequestType::ImageUnderstanding),
            (0x03, RequestType::ImageUnderstandingStream),
            (0x04, RequestType::HealthCheck),
            (0xFF, RequestType::Unknown), // 无效值
            (0x00, RequestType::Unknown), // 零值
        ];

        for (byte, expected_type) in type_bytes {
            let mut buffer = BytesMut::new();
            // 构造足够长的 payload 以包含 session_id 和 stream flag
            let mut payload = vec![byte]; // 类型字节
            payload.extend_from_slice(&[b'a'; 40]); // session_id (36 字节) + stream flag + 一些额外数据
            let len = payload.len() as u32;
            buffer.extend_from_slice(&len.to_be_bytes());
            buffer.extend_from_slice(&payload);

            let result = Gateway::parse_request(&mut buffer).unwrap().unwrap();
            assert_eq!(
                result.request_type, expected_type,
                "Byte 0x{:02X} should parse to {:?}",
                byte, expected_type
            );
        }
    }

    #[test]
    fn test_gateway_stream_flag_detection() {
        let mut buffer = BytesMut::new();

        let mut payload = vec![0x01];
        let session = b"session-test-session-id-test-123";
        let padding_len = 38 - 1 - session.len(); // pad so stream flag lands at index 38
        payload.extend_from_slice(session);
        payload.extend_from_slice(&vec![0u8; padding_len]);
        payload.push(0x01); // stream at index 38
        payload.extend_from_slice(b"data");

        let len = payload.len() as u32;
        buffer.extend_from_slice(&len.to_be_bytes());
        buffer.extend_from_slice(&payload);

        let request = Gateway::parse_request(&mut buffer).unwrap().unwrap();
        assert!(request.stream);

        let mut buffer2 = BytesMut::new();
        let mut payload2 = vec![0x01];
        payload2.extend_from_slice(session);
        payload2.extend_from_slice(&vec![0u8; padding_len]);
        payload2.push(0x00);
        payload2.extend_from_slice(b"data");
        let len2 = payload2.len() as u32;
        buffer2.extend_from_slice(&len2.to_be_bytes());
        buffer2.extend_from_slice(&payload2);

        let request2 = Gateway::parse_request(&mut buffer2).unwrap().unwrap();
        assert!(!request2.stream);
    }

    #[test]
    fn test_gateway_response_with_various_errors() {
        // 覆盖分支: 各种错误响应创建
        let long_error_msg = "x".repeat(500);
        let error_cases: Vec<(&str, &str)> = vec![
            ("timeout error", "request timeout exceeded"),
            ("memory error", "out of memory allocating buffer"),
            ("", ""),                                           // 空错误消息
            (long_error_msg.as_str(), long_error_msg.as_str()), // 长错误消息 (<20KB)
        ];

        for (session_id, error_msg) in error_cases {
            let response = Response::with_error(
                format!("session-{}", session_id.len()),
                error_msg.to_string(),
            );

            assert!(response.finished);
            assert!(response.payload.is_empty()); // 错误响应无 payload
            assert_eq!(response.error.as_deref(), Some(error_msg));
        }
    }

    #[test]
    fn test_zero_copy_buffer_boundary_operations() {
        // 覆盖边界: ZeroCopyBuffer 的各种边界操作

        // 单字节数据
        let mut single_byte = ZeroCopyBuffer::new(Bytes::from_static(b"A"));
        assert_eq!(single_byte.remaining(), 1);
        assert!(!single_byte.is_empty());

        single_byte.advance(1); // 消耗完
        assert!(single_byte.is_empty());
        assert_eq!(single_byte.remaining(), 0);

        // advance 超过边界应被限制
        let data = Bytes::from_static(b"AB");
        let mut buf = ZeroCopyBuffer::new(data);
        buf.advance(100); // 尝试前进超过长度
        assert!(buf.is_empty()); // 应该被限制在末尾

        // slice 在空缓冲区上
        let empty_buf = ZeroCopyBuffer::new(Bytes::new());
        let slice = empty_buf.slice();
        assert!(slice.is_empty());
    }
}
