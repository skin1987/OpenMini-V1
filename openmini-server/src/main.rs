//! OpenMini 服务器主入口模块
//!
//! 本模块是 OpenMini 推理服务器的启动入口，负责：
//! - 初始化日志系统
//! - 加载服务器配置
//! - 创建统一任务调度器 (TaskScheduler)
//! - 启动网关服务
//! - 处理优雅关闭信号
//!
//! # 架构概述
//!
//! OpenMini 服务器采用 **TaskScheduler 单进程架构**：
//! - 主进程：运行 TaskScheduler 统一调度所有推理任务
//! - 基于 Tokio Runtime 的异步任务调度
//! - 使用 spawn_blocking 处理 CPU 密集型推理任务
//!
//! # 启动流程
//!
//! 1. 初始化日志系统（支持 RUST_LOG 环境变量）
//! 2. 加载服务器配置文件
//! 3. 初始化内存监控器
//! 4. 创建 TaskScheduler 统一任务调度器
//! 5. 创建 AsyncInferencePool 用于请求排队
//! 6. 启动网关服务
//! 7. 等待关闭信号并优雅退出

// Clippy 配置：允许预期的 cfg 条件值和函数参数过多
#![allow(unexpected_cfgs)]
#![allow(clippy::too_many_arguments)]

// ============================================================================
// 模块声明
// ============================================================================

mod config;
mod error;
mod hardware;
mod kernel;
mod model;
mod monitoring;
mod service;

// ============================================================================
// 外部依赖导入
// ============================================================================

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use config::ServerConfig;
use service::scheduler::{SchedulerConfig, TaskScheduler};
use service::worker::AsyncInferencePool;

// ============================================================================
// 主函数
// ============================================================================

/// 服务器主入口函数
///
/// 使用 Tokio 异步运行时，执行以下流程：
/// 1. 初始化日志订阅器
/// 2. 初始化主进程的各项组件
/// 3. 启动 TaskScheduler 和网关服务
/// 4. 等待关闭信号并优雅退出
///
/// # 返回值
///
/// 成功返回 `Ok(())`，失败返回错误信息
///
/// # 环境变量
///
/// - `RUST_LOG`: 控制日志级别，默认为 "info"
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ========================================================================
    // 日志系统初始化
    // ========================================================================

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // ========================================================================
    // 主进程初始化
    // ========================================================================

    info!("OpenMini Server starting (TaskScheduler mode)...");

    let _hardware_profile = hardware::detect_hardware();
    let cpu_backend = hardware::CpuBackend::create();
    let cpu_info = hardware::CpuBackend::cpu_info();

    info!(
        "CPU Backend: {} ({})",
        cpu_backend.backend_name(),
        cpu_backend.backend_type()
    );
    info!("CPU Info:\n{}", cpu_info);

    // ========================================================================
    // 配置加载
    // ========================================================================

    let config = load_config();
    info!(
        "Configuration loaded: scheduler(max_concurrent={}, queue_capacity={}), {} max connections",
        config.scheduler.max_concurrent,
        config.scheduler.queue_capacity,
        config.server.max_connections
    );

    // ========================================================================
    // 内存监控器初始化
    // ========================================================================

    info!(
        "Initializing memory monitor (max {} GB)...",
        config.memory.max_memory_gb
    );
    let _memory_monitor = Arc::new(hardware::memory::MemoryMonitor::new(
        config.memory.max_memory_gb * 1024 * 1024 * 1024,
    ));

    // ========================================================================
    // TaskScheduler 统一任务调度器创建 (替代旧的 ThreadPool + WorkerPool + CoreRouter)
    // ========================================================================

    let scheduler_config = SchedulerConfig::new(
        config.scheduler.max_concurrent,
        config.scheduler.queue_capacity,
    )
    .with_batching(config.scheduler.batch_size, config.scheduler.batch_timeout_ms);

    info!(
        max_concurrent = scheduler_config.max_concurrent,
        queue_capacity = scheduler_config.queue_capacity,
        batch_size = scheduler_config.batch_size,
        batch_timeout_ms = scheduler_config.batch_timeout_ms,
        "Creating TaskScheduler..."
    );
    let task_scheduler = Arc::new(TaskScheduler::new(&scheduler_config));
    info!("TaskScheduler initialized successfully");

    // ========================================================================
    // 异步推理池创建 (用于请求排队和批处理)
    // ========================================================================

    info!("Creating async inference pool...");
    let (inference_pool, _inference_shutdown) = AsyncInferencePool::create(
        config.scheduler.queue_capacity,
        config.scheduler.batch_size,
        std::time::Duration::from_millis(config.scheduler.batch_timeout_ms),
        |tasks| {
            tasks
                .iter()
                .map(|t| service::worker::async_pool::InferenceResult {
                    session_id: t.session_id.clone(),
                    text: format!("Processed: {}", t.prompt),
                    finished: true,
                })
                .collect()
        },
    );
    let inference_pool = Arc::new(inference_pool);
    info!("AsyncInferencePool created successfully");

    // ========================================================================
    // 网关启动
    // ========================================================================

    let addr: SocketAddr = config.server.host.parse()?;
    info!("Creating gateway on {}...", addr);

    let gateway = service::server::Gateway::new(addr, inference_pool);

    let shutdown_signal = setup_shutdown_signal();

    info!(
        "Server listening on {}:{}",
        config.server.host, config.server.port
    );
    info!("Ready to accept connections!");
    info!("Architecture: TaskScheduler (single-process,Tokio-based)");

    // ========================================================================
    // 主事件循环
    // ========================================================================

    tokio::select! {
        result = gateway.run() => {
            if let Err(e) = result {
                error!("Gateway error: {}", e);
            }
        }
        _ = shutdown_signal => {
            info!("Shutdown signal received");
        }
    }

    // ========================================================================
    // 优雅关闭
    // ========================================================================

    info!("Shutting down gracefully...");

    info!("Shutting down TaskScheduler...");
    if let Err(e) = task_scheduler.shutdown().await {
        warn!("TaskScheduler shutdown error: {}", e);
    }
    info!("TaskScheduler stopped");

    info!(
        completed = task_scheduler.completed_count(),
        failed = task_scheduler.failed_count(),
        "Final statistics"
    );

    info!("Server stopped");

    Ok(())
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 加载服务器配置
fn load_config() -> ServerConfig {
    let config_path = std::path::Path::new("config/server.toml");

    if config_path.exists() {
        match ServerConfig::from_file(config_path) {
            Ok(config) => {
                info!("Loaded config from {}", config_path.display());
                return config;
            }
            Err(e) => {
                warn!("Failed to load config file: {}, using defaults", e);
            }
        }
    }

    ServerConfig::default()
}

/// 设置关闭信号监听器
async fn setup_shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = signal::ctrl_c().await {
            error!("Failed to install Ctrl+C handler: {}", e);
            std::process::exit(1);
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                error!("Failed to install signal handler: {}", e);
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
