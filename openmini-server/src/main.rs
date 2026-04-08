//! OpenMini 服务器主入口模块
//!
//! 本模块是 OpenMini 推理服务器的启动入口，负责：
//! - 初始化日志系统
//! - 加载服务器配置
//! - 创建并管理 Worker 进程池
//! - 启动 gRPC 网关服务
//! - 处理优雅关闭信号

// 允许文档缺失和未使用代码（内部实现）
#![allow(missing_docs)]
#![allow(dead_code)]
#![allow(unused_imports)]
//!
//! # 架构概述
//!
//! OpenMini 服务器采用多进程架构：
//! - 主进程：负责启动和管理 Worker 进程，运行 gRPC 网关
//! - Worker 进程：执行实际的模型推理任务
//!
//! # 启动流程
//!
//! 1. 初始化日志系统（支持 RUST_LOG 环境变量）
//! 2. 检测是否为 Worker 进程，如果是则直接运行 Worker 逻辑
//! 3. 加载服务器配置文件
//! 4. 初始化内存监控器
//! 5. 创建线程池和 Worker 池
//! 6. 启动 gRPC 网关服务
//! 7. 等待关闭信号并优雅退出

// ============================================================================
// 模块声明
// ============================================================================

mod config;
mod hardware;
mod model;
mod monitoring;
mod service;

// ============================================================================
// 外部依赖导入
// ============================================================================

use std::sync::Arc;
use std::net::SocketAddr;
use tokio::signal;
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use config::ServerConfig;
use service::thread::ThreadPool;

// ============================================================================
// 主函数
// ============================================================================

/// 服务器主入口函数
///
/// 使用 Tokio 异步运行时，执行以下流程：
/// 1. 初始化日志订阅器
/// 2. 检测并启动 Worker 进程（如果是子进程模式）
/// 3. 初始化主进程的各项组件
/// 4. 启动 gRPC 服务并等待关闭信号
///
/// # 返回值
///
/// 成功返回 `Ok(())`，失败返回错误信息
///
/// # 环境变量
///
/// - `RUST_LOG`: 控制日志级别，默认为 "info"
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ========================================================================
    // 日志系统初始化
    // ========================================================================

    // 配置日志订阅器，支持通过 RUST_LOG 环境变量控制日志级别
    // 默认日志级别为 info，输出格式为标准格式
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // ========================================================================
    // Worker 进程检测与启动
    // ========================================================================

    // 检查当前进程是否为 Worker 进程
    // Worker 进程由主进程通过 fork 创建，用于执行推理任务
    if service::worker::is_worker_process() {
        if let Some(id) = service::worker::get_worker_id() {
            info!("Starting worker {} with pid {}", id, std::process::id());
            let worker = service::worker::Worker::new(id);
            worker.run();
            info!("Worker {} exited", id);
        }
        // Worker 进程执行完毕后直接退出
        return Ok(());
    }

    // ========================================================================
    // 主进程初始化
    // ========================================================================

    info!("OpenMini Server starting...");

    let _hardware_profile = hardware::detect_hardware();
    let cpu_backend = hardware::CpuBackend::create();
    let cpu_info = hardware::CpuBackend::cpu_info();
    
    info!("CPU Backend: {} ({})", cpu_backend.backend_name(), cpu_backend.backend_type());
    info!("CPU Info:\n{}", cpu_info);
    
    info!("Hardware: {} CPU cores, {} GB max memory",
          num_cpus::get(), config::ServerConfig::default().memory.max_memory_gb);

    // ========================================================================
    // 配置加载
    // ========================================================================

    let config = load_config();
    info!("Configuration loaded: {} workers, {} max connections",
          config.worker.count, config.server.max_connections);

    // ========================================================================
    // 内存监控器初始化
    // ========================================================================

    // 创建内存监控器，用于跟踪内存使用情况
    // 当内存使用接近限制时，会触发警告或拒绝新请求
    info!("Initializing memory monitor (max {} GB)...", config.memory.max_memory_gb);
    let _memory_monitor = Arc::new(hardware::memory::MemoryMonitor::new(config.memory.max_memory_gb * 1024 * 1024 * 1024));

    // ========================================================================
    // 线程池创建
    // ========================================================================

    // 创建线程池用于并行处理请求
    // 线程池大小由配置文件决定，通常设置为 CPU 核心数
    info!("Creating thread pool ({} threads)...", config.thread_pool.size);
    let thread_pool = Arc::new(ThreadPool::new(config.thread_pool.size));

    // ========================================================================
    // Worker 进程池创建
    // ========================================================================

    // 创建 Worker 进程池，每个 Worker 独立运行模型推理
    // Worker 数量由配置决定，通常根据 GPU 数量和内存限制设置
    info!("Initializing worker pool ({} workers)...", config.worker.count);
    let worker_pool = Arc::new(service::worker::WorkerPool::new(config.worker.clone().into())?);

    // ========================================================================
    // gRPC 网关启动
    // ========================================================================

    // 解析服务器监听地址
    let addr: SocketAddr = config.server.host.parse()?;
    info!("Creating gRPC gateway on {}...", addr);

    // 创建 gRPC 网关，将请求分发到线程池和 Worker 池
    let gateway = service::server::Gateway::new(addr, thread_pool);

    // 设置关闭信号监听器
    let shutdown_signal = setup_shutdown_signal();

    info!("Server listening on {}:{}", config.server.host, config.server.port);
    info!("Ready to accept connections!");

    // ========================================================================
    // 主事件循环
    // ========================================================================

    // 使用 tokio::select! 同时等待网关运行和关闭信号
    // 任一事件触发都会继续执行
    tokio::select! {
        result = gateway.run() => {
            // 网关运行出错时记录错误
            if let Err(e) = result {
                error!("Gateway error: {}", e);
            }
        }
        _ = shutdown_signal => {
            // 收到关闭信号（Ctrl+C 或 SIGTERM）
            info!("Shutdown signal received");
        }
    }

    // ========================================================================
    // 优雅关闭
    // ========================================================================

    info!("Shutting down gracefully...");

    // 关闭 Worker 进程池，等待所有 Worker 完成当前任务
    worker_pool.shutdown();

    info!("Server stopped");

    Ok(())
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 加载服务器配置
///
/// 尝试从 `config/server.toml` 文件加载配置。
/// 如果文件不存在或加载失败，则使用默认配置。
///
/// # 返回值
///
/// 返回服务器配置对象
///
/// # 配置文件格式
///
/// 配置文件使用 TOML 格式，包含以下部分：
/// - `[server]`: 服务器配置（主机、端口、最大连接数等）
/// - `[worker]`: Worker 进程配置（数量、内存限制等）
/// - `[thread_pool]`: 线程池配置
/// - `[memory]`: 内存限制配置
fn load_config() -> ServerConfig {
    let config_path = std::path::Path::new("config/server.toml");

    if config_path.exists() {
        match ServerConfig::from_file(config_path) {
            Ok(config) => {
                info!("Loaded config from {}", config_path.display());
                return config;
            }
            Err(e) => {
                // 配置文件解析失败时使用默认配置
                warn!("Failed to load config file: {}, using defaults", e);
            }
        }
    }

    // 使用默认配置
    ServerConfig::default()
}

/// 设置关闭信号监听器
///
/// 监听以下信号：
/// - `Ctrl+C` (SIGINT): 用户中断
/// - `SIGTERM`: 终止信号（如 systemd stop）
///
/// 在 Unix 系统上同时监听两种信号，
/// 在非 Unix 系统上只监听 Ctrl+C。
///
/// # 返回值
///
/// 返回一个 Future，当收到关闭信号时完成
async fn setup_shutdown_signal() {
    // Ctrl+C 信号处理
    let ctrl_c = async {
        if let Err(e) = signal::ctrl_c().await {
            error!("Failed to install Ctrl+C handler: {}", e);
            std::process::exit(1);
        }
    };

    // Unix 系统特有的 SIGTERM 信号处理
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

    // 非 Unix 系统使用 pending future（永不完成）
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    // 等待任一信号
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
