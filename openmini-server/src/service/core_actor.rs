//! Per-Core Actor — 每个 CPU 核心的独立推理服务实例
//!
//! 每个 Actor 绑定到一个物理 CPU 核心，拥有：
//! - 独立的 Tokio 单线程运行时（事件循环）
//! - 独立的 InferenceEngine 实例
//! - 独立的 KV Cache 内存
//! - 独立的连接管理能力

use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::mpsc;
use tracing::{debug, error, info};
#[cfg(not(target_os = "linux"))]
use tracing::warn;

/// 推理请求（从 Router 转发过来）
pub struct CoreRequest {
    pub prompt: String,
    pub session_id: String,
    pub response_tx: mpsc::Sender<CoreResponse>,
}

/// 推理响应
pub struct CoreResponse {
    pub session_id: String,
    pub text: String,
    pub finished: bool,
    pub error: Option<String>,
}

impl CoreResponse {
    pub fn ok(session_id: String, text: String) -> Self {
        Self {
            session_id,
            text,
            finished: true,
            error: None,
        }
    }

    pub fn err(session_id: String, error: String) -> Self {
        Self {
            session_id,
            text: String::new(),
            finished: true,
            error: Some(error),
        }
    }
}

/// Per-Core Actor — 单核推理服务
pub struct PerCoreActor {
    /// 核心 ID（0, 1, 2, ... N-1）
    pub core_id: usize,
    /// 当前活跃连接数
    active_connections: AtomicU64,
    /// 请求接收端
    request_rx: mpsc::Receiver<CoreRequest>,
}

impl PerCoreActor {
    /// 创建新的 PerCoreActor
    ///
    /// # 参数
    /// - `core_id`: CPU 核心编号
    /// - `request_rx`: 请求通道接收端
    pub fn new(core_id: usize, request_rx: mpsc::Receiver<CoreRequest>) -> Self {
        Self {
            core_id,
            active_connections: AtomicU64::new(0),
            request_rx,
        }
    }

    /// 运行 Actor 主循环
    ///
    /// 此方法会绑定到指定 CPU 核心，然后进入事件循环处理请求。
    /// 在 current_thread Tokio 运行时中运行。
    pub async fn run(mut self) {
        info!("Core-{} Actor started", self.core_id);

        if let Err(e) = bind_to_core(self.core_id) {
            error!("Failed to bind to core {}: {}", self.core_id, e);
        } else {
            info!("Core-{} bound to CPU core {}", self.core_id, self.core_id);
        }

        while let Some(request) = self.request_rx.recv().await {
            self.active_connections.fetch_add(1, Ordering::Relaxed);

            let conn_count = self.active_connections.load(Ordering::Relaxed);
            debug!("Core-{} active connections: {}", self.core_id, conn_count);

            let response = Self::handle_request(&request).await;

            let _ = request.response_tx.send(response).await;

            self.active_connections.fetch_sub(1, Ordering::Relaxed);
        }

        info!("Core-{} Actor stopped", self.core_id);
    }

    /// 处理单个推理请求
    async fn handle_request(request: &CoreRequest) -> CoreResponse {
        let session_id = request.session_id.clone();

        debug!(
            "Core-{} processing request for session {}",
            /* 需要获取 core_id 但 self 不可用 */ 0, session_id
        );

        // TODO: Phase 3 完成后对接 AsyncInferencePool
        // 目前返回模拟响应
        CoreResponse::ok(session_id, "Core response".to_string())
    }

    /// 获取当前活跃连接数
    pub fn active_connections(&self) -> u64 {
        self.active_connections.load(Ordering::Relaxed)
    }
}

/// 将当前线程绑定到指定的 CPU 核心（Linux）
#[cfg(target_os = "linux")]
fn bind_to_core(core_id: usize) -> Result<(), String> {
    use std::thread;

    let core = core_id as libc::c_int;
    let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(core, &mut cpuset);
    }

    let result = unsafe {
        libc::pthread_setaffinity_np(
            std::thread::current().native_handle().as_raw_handle() as *mut _,
            std::mem::size_of::<libc::cpu_set_t>(),
            &cpuset as *const _ as *const _,
        )
    };

    if result != 0 {
        return Err(format!("pthread_setaffinity_np failed: {}", result));
    }

    Ok(())
}

/// 将当前线程绑定到指定的 CPU 核心（macOS/BSD）
///
/// 注意：macOS 不支持线程级 CPU 亲和性绑定
/// 此处返回成功但无实际效果（与项目 hyperthreading 模块保持一致）
#[cfg(target_os = "macos")]
fn bind_to_core(core_id: usize) -> Result<(), String> {
    let _ = core_id;
    warn!("CPU affinity binding not supported on macOS (no effect)");
    Ok(())
}

/// 其他平台暂不支持 CPU 绑定
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn bind_to_core(_core_id: usize) -> Result<(), String> {
    warn!("CPU affinity binding not supported on this platform");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_response_ok() {
        let resp = CoreResponse::ok("sess-1".to_string(), "hello".to_string());
        assert_eq!(resp.session_id, "sess-1");
        assert_eq!(resp.text, "hello");
        assert!(resp.finished);
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_core_response_err() {
        let resp = CoreResponse::err("sess-2".to_string(), "boom".to_string());
        assert_eq!(resp.session_id, "sess-2");
        assert!(resp.error.as_deref() == Some("boom"));
        assert!(resp.finished);
    }

    #[test]
    fn test_actor_creation() {
        let (_, rx) = mpsc::channel::<CoreRequest>(100);
        let actor = PerCoreActor::new(0, rx);
        assert_eq!(actor.core_id, 0);
        assert_eq!(actor.active_connections(), 0);
    }
}
