//! OpenMini-V1 配置验证集成测试
//!
//! 全面验证配置系统的正确性和健壮性：
//! - 默认配置完整性检查
//! - TOML 文件加载和序列化
//! - 配置字段范围和约束验证
//! - 边界条件和异常输入处理

use std::path::PathBuf;

#[test]
fn test_default_config_completeness() {
    use openmini_server::config::settings::ServerConfig;

    let config = ServerConfig::default();

    assert_eq!(config.server.host, "0.0.0.0");
    assert_eq!(config.server.port, 50051);
    assert_eq!(config.server.max_connections, 10000);
    assert_eq!(config.server.request_timeout_ms, 60000);

    assert!(config.thread_pool.size > 0);
    assert_eq!(config.thread_pool.stack_size_kb, 8192);

    assert_eq!(config.memory.max_memory_gb, 12);
    assert_eq!(config.memory.model_memory_gb, 6);
    assert_eq!(config.memory.cache_memory_gb, 4);
    assert_eq!(config.memory.arena_size_mb, 256);

    assert!(!config.model.path.as_os_str().is_empty());
    assert_eq!(config.model.quantization, "Q4_K_M");
    assert_eq!(config.model.context_length, 4096);

    assert_eq!(config.worker.count, 1);
    assert!(config.worker.restart_on_failure);
    assert_eq!(config.worker.health_check_interval_ms, 5000);

    assert_eq!(config.grpc.max_message_size_mb, 100);
    assert_eq!(config.grpc.keepalive_time_ms, 30000);
    assert_eq!(config.grpc.keepalive_timeout_ms, 10000);
}

#[test]
fn test_toml_roundtrip_consistency() {
    use openmini_server::config::settings::ServerConfig;

    let original = ServerConfig::default();
    let toml_str = toml::to_string(&original).expect("Serialization should succeed");
    assert!(!toml_str.is_empty(), "TOML output should not be empty");

    let loaded: ServerConfig = toml::from_str(&toml_str).expect("Deserialization should succeed");

    assert_eq!(loaded.server.host, original.server.host);
    assert_eq!(loaded.server.port, original.server.port);
    assert_eq!(
        loaded.server.max_connections,
        original.server.max_connections
    );
    assert_eq!(loaded.model.quantization, original.model.quantization);
    assert_eq!(loaded.model.context_length, original.model.context_length);
    assert_eq!(loaded.thread_pool.size, original.thread_pool.size);
    assert_eq!(loaded.memory.max_memory_gb, original.memory.max_memory_gb);
}

#[test]
fn test_custom_toml_file_loading() {
    use openmini_server::config::settings::ServerConfig;

    let toml_content = r#"
[server]
host = "192.168.1.100"
port = 9090
max_connections = 500
request_timeout_ms = 120000

[thread_pool]
size = 8
stack_size_kb = 16384

[memory]
max_memory_gb = 32
model_memory_gb = 16
cache_memory_gb = 12
arena_size_mb = 512

[model]
path = "/data/models/custom.gguf"
quantization = "Q8_0"
context_length = 8192

[worker]
count = 4
restart_on_failure = false
health_check_interval_ms = 10000

[grpc]
max_message_size_mb = 256
keepalive_time_ms = 60000
keepalive_timeout_ms = 30000

[database]
path = "/data/db/custom.db"
pool_size = 20
busy_timeout_ms = 10000
"#;

    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("openmini_custom_config_test.toml");
    std::fs::write(&temp_file, toml_content).expect("Failed to write temp config file");

    let config = ServerConfig::from_file(&temp_file).expect("Should load custom TOML");

    assert_eq!(config.server.host, "192.168.1.100");
    assert_eq!(config.server.port, 9090);
    assert_eq!(config.server.max_connections, 500);
    assert_eq!(config.server.request_timeout_ms, 120000);

    assert_eq!(config.thread_pool.size, 8);
    assert_eq!(config.thread_pool.stack_size_kb, 16384);

    assert_eq!(config.memory.max_memory_gb, 32);
    assert_eq!(config.memory.model_memory_gb, 16);
    assert_eq!(config.memory.cache_memory_gb, 12);
    assert_eq!(config.memory.arena_size_mb, 512);

    assert_eq!(config.model.path, PathBuf::from("/data/models/custom.gguf"));
    assert_eq!(config.model.quantization, "Q8_0");
    assert_eq!(config.model.context_length, 8192);

    assert_eq!(config.worker.count, 4);
    assert!(!config.worker.restart_on_failure);
    assert_eq!(config.worker.health_check_interval_ms, 10000);

    assert_eq!(config.grpc.max_message_size_mb, 256);
    assert_eq!(config.grpc.keepalive_time_ms, 60000);
    assert_eq!(config.grpc.keepalive_timeout_ms, 30000);

    assert_eq!(config.database.pool_size, 20);
    assert_eq!(config.database.busy_timeout_ms, 10000);

    std::fs::remove_file(temp_file).ok();
}

#[test]
fn test_config_validation_edge_cases() {
    use openmini_server::config::settings::ServerConfig;

    let config = ServerConfig::default();

    assert!(config.server.port > 0); // port is u16, always <= 65535
    assert!(config.server.max_connections >= 1 && config.server.max_connections <= 100000);
    assert!(config.server.request_timeout_ms >= 1000);

    assert!(config.memory.max_memory_gb >= config.memory.model_memory_gb);
    assert!(
        config.memory.model_memory_gb + config.memory.cache_memory_gb
            <= config.memory.max_memory_gb + 4
    );
    assert!(config.memory.arena_size_mb >= 1);

    assert!(config.model.context_length >= 512 && config.model.context_length <= 131072);

    let valid_quantizations = ["Q4_K_M", "Q4_0", "Q8_0", "F16", "F32", "Q5_K_M", "Q6_K"];
    assert!(
        valid_quantizations.contains(&config.model.quantization.as_str()),
        "Default quantization should be valid"
    );

    assert!(config.worker.count >= 1 && config.worker.count <= 64);
    assert!(config.worker.health_check_interval_ms >= 1000);

    assert!(config.grpc.max_message_size_mb >= 1 && config.grpc.max_message_size_mb <= 1024);
    assert!(config.grpc.keepalive_time_ms >= 1000);
    assert!(config.grpc.keepalive_timeout_ms >= 1000);
    assert!(
        config.grpc.keepalive_timeout_ms < config.grpc.keepalive_time_ms,
        "Timeout should be less than keepalive time"
    );

    let addr = config.addr();
    assert!(addr.contains(':'), "Address should contain ':' separator");
    assert_eq!(
        addr,
        format!("{}:{}", config.server.host, config.server.port)
    );

    let max_bytes = config.max_memory_bytes();
    assert_eq!(max_bytes, config.memory.max_memory_gb * 1024 * 1024 * 1024);

    let arena_bytes = config.arena_size_bytes();
    assert_eq!(arena_bytes, config.memory.arena_size_mb * 1024 * 1024);
    assert!(
        arena_bytes < max_bytes,
        "Arena should fit within max memory"
    );
}
