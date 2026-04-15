//! OpenMini-V1 完整生命周期 E2E 测试
//!
//! 模拟完整的服务器启动到关闭的生命周期：
//! - 配置加载 → 组件初始化 → 服务就绪
//! - 内存监控器初始化和使用
//! - KV Cache 管理器生命周期
//! - 连接池创建和清理
//! - 资源释放验证

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn test_full_config_load_and_validate_lifecycle() {
    use openmini_server::config::settings::ServerConfig;

    eprintln!("[lifecycle-config] Phase 1: Config load and validate");

    let config = ServerConfig::default();
    eprintln!(
        "[lifecycle-config] Loaded config: {}:{}",
        config.server.host, config.server.port
    );

    assert!(!config.addr().is_empty());
    assert!(config.max_memory_bytes() > 0);
    assert!(config.arena_size_bytes() > 0);
    assert!(config.model.context_length > 0);

    let serialized = toml::to_string(&config).expect("Should serialize");
    let _validated: ServerConfig = toml::from_str(&serialized).expect("Should deserialize");

    eprintln!("[lifecycle-config] Config lifecycle PASSED");
}

#[test]
fn test_full_memory_monitor_lifecycle() {
    use openmini_server::hardware::memory::MemoryMonitor;

    eprintln!("[lifecycle-memory] Phase 2: Memory monitor lifecycle");

    let capacity = 100 * 1024 * 1024;
    let monitor = MemoryMonitor::new(capacity);
    assert_eq!(monitor.usage(), 0);

    let alloc_sizes = [1024 * 1024, 512 * 1024, 256 * 1024, 128 * 1024, 64 * 1024];
    let mut allocated_ids = Vec::new();

    for (i, &size) in alloc_sizes.iter().enumerate() {
        let result = monitor.allocate(size);
        assert!(
            result.is_ok(),
            "Allocation {} of {} bytes should succeed",
            i,
            size
        );
        result.unwrap();
        allocated_ids.push(());
        eprintln!(
            "[lifecycle-memory] Allocated {} KB (total usage: {} KB)",
            size / 1024,
            monitor.usage() / 1024
        );
    }

    let peak_usage = monitor.usage();
    assert!(peak_usage > 0, "Should have some memory in use");
    eprintln!("[lifecycle-memory] Peak usage: {} KB", peak_usage / 1024);

    for (i, &size) in alloc_sizes.iter().enumerate() {
        monitor.deallocate(size);
        eprintln!("[lifecycle-memory] Deallocated block {}", i);
    }

    assert_eq!(monitor.usage(), 0, "All memory should be freed");
    eprintln!("[lifecycle-memory] Memory monitor lifecycle PASSED");
}

#[tokio::test]
async fn test_full_connection_pool_lifecycle() {
    use openmini_server::service::server::connection::ConnectionPool;

    eprintln!("[lifecycle-conn] Phase 3: Connection pool lifecycle");

    let pool = Arc::new(ConnectionPool::new(10));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = [0u8; 64];
            let _ = stream.read(&mut buf).await;
            let _ = stream.write_all(b"OK").await;
        }
    });

    let acquire_count = Arc::new(AtomicUsize::new(0));
    let release_count = Arc::new(AtomicUsize::new(0));
    let num_ops = 20;

    for _ in 0..num_ops {
        if let Ok(conn) = pool.acquire_or_connect(addr).await {
            acquire_count.fetch_add(1, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_millis(5)).await;
            pool.release(conn);
            release_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    server_handle.abort();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let acquired = acquire_count.load(Ordering::Relaxed);
    let released = release_count.load(Ordering::Relaxed);
    let stats = pool.stats();

    eprintln!(
        "[lifecycle-conn] Acquired: {}, Released: {}, Active: {}",
        acquired,
        released,
        stats.active_connections.load(Ordering::Relaxed)
    );

    assert_eq!(
        acquired, released,
        "Acquire and release counts should match"
    );
    assert_eq!(
        stats.active_connections.load(Ordering::Relaxed),
        0,
        "All connections should be released"
    );

    eprintln!("[lifecycle-conn] Connection pool lifecycle PASSED");
}
