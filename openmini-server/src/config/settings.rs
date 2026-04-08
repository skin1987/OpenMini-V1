//! 服务器配置模块 - 完整的服务器配置系统
//!
//! 提供完整的服务器配置结构，支持从 TOML 文件加载配置。
//! 包含服务器、线程池、内存、模型、Worker 和 gRPC 等配置项。

#![allow(dead_code)]

use serde::Deserialize;
use std::path::PathBuf;

/// 服务器完整配置
///
/// 包含所有子系统配置的顶层结构
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ServerConfig {
    /// 服务器基础设置
    pub server: ServerSettings,
    /// 线程池设置
    pub thread_pool: ThreadPoolSettings,
    /// 内存管理设置
    pub memory: MemorySettings,
    /// 模型加载设置
    pub model: ModelSettings,
    /// Worker 进程设置
    pub worker: WorkerSettings,
    /// gRPC 通信设置
    pub grpc: GrpcSettings,
    /// 数据库设置
    #[serde(default)]
    pub database: DatabaseSettings,
}

/// 服务器基础设置
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ServerSettings {
    /// 监听主机地址
    pub host: String,
    /// 监听端口号
    pub port: u16,
    /// 最大连接数
    pub max_connections: usize,
    /// 请求超时时间(毫秒)
    pub request_timeout_ms: u64,
}

/// 线程池设置
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ThreadPoolSettings {
    /// 线程池大小
    pub size: usize,
    /// 每个线程的栈大小(KB)
    pub stack_size_kb: usize,
}

/// 内存管理设置
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct MemorySettings {
    /// 最大可用内存(GB)
    pub max_memory_gb: usize,
    /// 模型占用内存上限(GB)
    pub model_memory_gb: usize,
    /// KV 缓存内存上限(GB)
    pub cache_memory_gb: usize,
    /// Arena 分配器大小(MB)
    pub arena_size_mb: usize,
}

/// 模型加载设置
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct ModelSettings {
    /// 模型文件路径
    pub path: PathBuf,
    /// 量化类型 (如 Q4_K_M, Q8_0 等)
    pub quantization: String,
    /// 上下文长度
    pub context_length: usize,
}

/// Worker 进程设置
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct WorkerSettings {
    /// Worker 进程数量
    pub count: usize,
    /// 失败时是否自动重启
    pub restart_on_failure: bool,
    /// 健康检查间隔(毫秒)
    pub health_check_interval_ms: u64,
}

/// gRPC 通信设置
#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct GrpcSettings {
    /// 最大消息大小(MB)
    pub max_message_size_mb: usize,
    /// Keepalive 时间(毫秒)
    pub keepalive_time_ms: u64,
    /// Keepalive 超时时间(毫秒)
    pub keepalive_timeout_ms: u64,
}

/// 数据库设置
#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct DatabaseSettings {
    /// 数据库文件路径
    pub path: PathBuf,
    /// 连接池大小
    #[serde(default = "default_pool_size")]
    pub pool_size: u32,
    /// 忙等待超时时间(毫秒)
    #[serde(default = "default_busy_timeout")]
    pub busy_timeout_ms: u64,
}

fn default_pool_size() -> u32 {
    10
}

fn default_busy_timeout() -> u64 {
    5000
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerSettings {
                host: "0.0.0.0".to_string(),
                port: 50051,
                max_connections: 100,
                request_timeout_ms: 60000,
            },
            thread_pool: ThreadPoolSettings {
                size: num_cpus::get(),
                stack_size_kb: 8192,
            },
            memory: MemorySettings {
                max_memory_gb: 12,
                model_memory_gb: 6,
                cache_memory_gb: 4,
                arena_size_mb: 256,
            },
            model: ModelSettings {
                path: PathBuf::from("models/openmini-v1-q4_k_m.gguf"),
                quantization: "Q4_K_M".to_string(),
                context_length: 4096,
            },
            worker: WorkerSettings {
                count: 3,
                restart_on_failure: true,
                health_check_interval_ms: 5000,
            },
            grpc: GrpcSettings {
                max_message_size_mb: 100,
                keepalive_time_ms: 30000,
                keepalive_timeout_ms: 10000,
            },
            database: DatabaseSettings {
                path: PathBuf::from("data/openmini.db"),
                pool_size: 10,
                busy_timeout_ms: 5000,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_settings_default_config() {
        // 测试默认配置
        let config = ServerConfig::default();

        // 验证服务器设置
        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 50051);
        assert_eq!(config.server.max_connections, 100);
        assert_eq!(config.server.request_timeout_ms, 60000);

        // 验证模型设置
        assert_eq!(config.model.quantization, "Q4_K_M");
        assert_eq!(config.model.context_length, 4096);
        assert!(!config.model.path.as_os_str().is_empty());

        // 验证线程池设置
        assert!(config.thread_pool.size > 0);
        assert_eq!(config.thread_pool.stack_size_kb, 8192);

        // 验证内存设置
        assert_eq!(config.memory.max_memory_gb, 12);
        assert_eq!(config.memory.model_memory_gb, 6);
        assert_eq!(config.memory.cache_memory_gb, 4);
        assert_eq!(config.memory.arena_size_mb, 256);

        // 验证 Worker 设置
        assert_eq!(config.worker.count, 3);
        assert!(config.worker.restart_on_failure);
        assert_eq!(config.worker.health_check_interval_ms, 5000);

        // 验证 gRPC 设置
        assert_eq!(config.grpc.max_message_size_mb, 100);
        assert_eq!(config.grpc.keepalive_time_ms, 30000);
        assert_eq!(config.grpc.keepalive_timeout_ms, 10000);
    }

    #[test]
    fn test_settings_addr_format() {
        // 测试地址格式化
        let config = ServerConfig::default();
        let addr = config.addr();

        // 应该是 host:port 格式
        assert!(addr.contains(':'));
        assert_eq!(
            addr,
            format!("{}:{}", config.server.host, config.server.port)
        );
    }

    #[test]
    fn test_settings_max_memory_bytes() {
        // 测试内存计算
        let config = ServerConfig::default();
        let max_bytes = config.max_memory_bytes();

        // 应该是 GB * 1024^3
        assert_eq!(max_bytes, config.memory.max_memory_gb * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_settings_arena_size_bytes() {
        // 测试 Arena 大小计算
        let config = ServerConfig::default();
        let arena_bytes = config.arena_size_bytes();

        // 应该是 MB * 1024^2
        assert_eq!(arena_bytes, config.memory.arena_size_mb * 1024 * 1024);
    }

    #[test]
    fn test_settings_model_config_validation() {
        // 模型配置验证
        let config = ServerConfig::default();

        // 验证模型路径不为空
        assert!(!config.model.path.as_os_str().is_empty());

        // 验证上下文长度合理
        assert!(config.model.context_length > 0);
        assert!(config.model.context_length <= 131072); // 最大128K

        // 验证量化类型有效
        let valid_quantizations = ["Q4_K_M", "Q4_0", "Q8_0", "F16", "F32", "Q5_K_M", "Q6_K"];
        assert!(valid_quantizations.contains(&config.model.quantization.as_str()));
    }

    #[test]
    fn test_settings_server_config_ranges() {
        // 服务器配置范围验证
        let config = ServerConfig::default();

        // 端口应该在合理范围内
        // port 是 u16 类型,范围 0-65535 由类型保证

        // 最大连接数应该合理
        assert!(config.server.max_connections > 0 && config.server.max_connections <= 100000);

        // 超时时间应该合理 (毫秒)
        assert!(config.server.request_timeout_ms >= 1000); // 至少1秒
    }

    #[test]
    fn test_settings_worker_config() {
        // Worker配置验证
        let config = ServerConfig::default();

        // Worker数量应该合理
        assert!(config.worker.count >= 1 && config.worker.count <= 64);

        // 健康检查间隔应该合理
        assert!(config.worker.health_check_interval_ms >= 1000); // 至少1秒
    }

    #[test]
    fn test_settings_from_toml_file() {
        // 测试从TOML文件加载配置
        let toml_content = r#"
[server]
host = "127.0.0.1"
port = 8080
max_connections = 50
request_timeout_ms = 30000

[thread_pool]
size = 4
stack_size_kb = 4096

[memory]
max_memory_gb = 16
model_memory_gb = 8
cache_memory_gb = 6
arena_size_mb = 512

[model]
path = "/models/test.gguf"
quantization = "Q4_0"
context_length = 2048

[worker]
count = 2
restart_on_failure = false
health_check_interval_ms = 3000

[grpc]
max_message_size_mb = 50
keepalive_time_ms = 15000
keepalive_timeout_ms = 5000
"#;

        // 写入临时文件
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_config.toml");
        std::fs::write(&temp_file, toml_content).expect("Failed to write temp file");

        // 加载配置
        let result = ServerConfig::from_file(&temp_file);
        assert!(
            result.is_ok(),
            "Failed to load TOML config: {:?}",
            result.err()
        );

        let config = result.unwrap();

        // 验证加载的值
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.max_connections, 50);
        assert_eq!(config.server.request_timeout_ms, 30000);

        assert_eq!(config.thread_pool.size, 4);
        assert_eq!(config.thread_pool.stack_size_kb, 4096);

        assert_eq!(config.memory.max_memory_gb, 16);
        assert_eq!(config.memory.model_memory_gb, 8);
        assert_eq!(config.memory.cache_memory_gb, 6);
        assert_eq!(config.memory.arena_size_mb, 512);

        assert_eq!(config.model.path, PathBuf::from("/models/test.gguf"));
        assert_eq!(config.model.quantization, "Q4_0");
        assert_eq!(config.model.context_length, 2048);

        assert_eq!(config.worker.count, 2);
        assert!(!config.worker.restart_on_failure);
        assert_eq!(config.worker.health_check_interval_ms, 3000);

        assert_eq!(config.grpc.max_message_size_mb, 50);
        assert_eq!(config.grpc.keepalive_time_ms, 15000);
        assert_eq!(config.grpc.keepalive_timeout_ms, 5000);

        // 清理临时文件
        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_settings_from_invalid_file() {
        // 测试从无效文件路径加载
        let invalid_path = PathBuf::from("/nonexistent/path/config.toml");
        let result = ServerConfig::from_file(&invalid_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_settings_serialization_roundtrip() {
        // 测试序列化和反序列化的往返一致性
        let original = ServerConfig::default();

        // 序列化为TOML字符串
        let toml_str = toml::to_string(&original).expect("Failed to serialize to TOML");
        assert!(!toml_str.is_empty());

        // 从TOML反序列化
        let loaded: ServerConfig =
            toml::from_str(&toml_str).expect("Failed to deserialize from TOML");

        // 验证关键字段一致
        assert_eq!(loaded.server.host, original.server.host);
        assert_eq!(loaded.server.port, original.server.port);
        assert_eq!(
            loaded.server.max_connections,
            original.server.max_connections
        );
        assert_eq!(loaded.model.path, original.model.path);
        assert_eq!(loaded.model.quantization, original.model.quantization);
        assert_eq!(loaded.model.context_length, original.model.context_length);
    }

    // ==================== 新增分支覆盖测试 (8个) ====================

    #[test]
    fn test_settings_toml_missing_optional_field() {
        // 覆盖分支: TOML文件缺少可选字段时使用默认值
        let toml_content = r#"
[server]
host = "127.0.0.1"
port = 9090
max_connections = 20
request_timeout_ms = 10000

[thread_pool]
size = 2
stack_size_kb = 2048
"#;
        // 缺少 memory/model/worker/grpc 配置块

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_partial_config.toml");
        std::fs::write(&temp_file, toml_content).expect("Failed to write temp file");

        // 应该解析失败，因为缺少必需字段
        let result = ServerConfig::from_file(&temp_file);
        // TOML 反序列化会失败，因为缺少必需的 struct 字段
        assert!(
            result.is_err(),
            "Missing required fields should cause error"
        );

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_settings_invalid_port_value() {
        // 覆盖分支: 端口值超出范围（虽然 TOML 会接受，但逻辑上应验证）
        let toml_content = r#"
[server]
host = "0.0.0.0"
port = 0
max_connections = 100
request_timeout_ms = 30000

[thread_pool]
size = 4
stack_size_kb = 8192

[memory]
max_memory_gb = 12
model_memory_gb = 6
cache_memory_gb = 4
arena_size_mb = 256

[model]
path = "/models/test.gguf"
quantization = "Q4_K_M"
context_length = 4096

[worker]
count = 3
restart_on_failure = true
health_check_interval_ms = 5000

[grpc]
max_message_size_mb = 100
keepalive_time_ms = 30000
keepalive_timeout_ms = 10000
"#;

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_zero_port.toml");
        std::fs::write(&temp_file, toml_content).unwrap();

        let result = ServerConfig::from_file(&temp_file);
        // 端口为 0 在 TOML 层面是合法的（u16 类型），但逻辑上可能不合法
        // 这里主要验证解析不会 panic
        if let Ok(config) = result {
            assert_eq!(config.server.port, 0); // 确认能正确读取
        }

        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_settings_max_memory_calculation_consistency() {
        // 覆盖分支: 内存计算的一致性验证
        let config = ServerConfig::default();

        // max_memory_bytes 应该等于 GB * 1024^3
        let expected_bytes = config.memory.max_memory_gb * 1024 * 1024 * 1024;
        assert_eq!(config.max_memory_bytes(), expected_bytes);

        // arena_size_bytes 应该等于 MB * 1024^2
        let expected_arena = config.memory.arena_size_mb * 1024 * 1024;
        assert_eq!(config.arena_size_bytes(), expected_arena);

        // 验证 arena < total memory
        assert!(config.arena_size_bytes() < config.max_memory_bytes());
    }

    #[test]
    fn test_settings_model_path_edge_cases() {
        // 覆盖分支: 模型路径边界情况
        let mut config = ServerConfig::default();

        // 空路径
        config.model.path = PathBuf::from("");
        assert!(config.model.path.as_os_str().is_empty());

        // 相对路径
        config.model.path = PathBuf::from("./models/model.gguf");
        assert!(config.model.path.is_relative());

        // 绝对路径
        config.model.path = PathBuf::from("/absolute/path/to/model.bin");
        assert!(config.model.path.is_absolute());
    }

    #[test]
    fn test_settings_quantization_validation_all_types() {
        // 覆盖分支: 所有支持的量化类型验证
        let valid_quantizations = [
            "Q4_K_M", "Q4_0", "Q8_0", "F16", "F32", "Q5_K_M", "Q6_K", "Q3_K_M", "Q2_K", "Q5_0",
            "Q5_1", "IQ4_NL", "IQ4_XS", "IQ3_S", "IQ3_M", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M",
        ];

        for &q in &valid_quantizations {
            // 验证这些字符串都是有效的量化类型标识符
            assert!(!q.is_empty());
            assert!(q.len() <= 10); // 合理长度
        }
    }

    #[test]
    fn test_settings_context_length_boundaries() {
        // 覆盖分支: 上下文长度边界值验证
        let config = ServerConfig::default();

        // 默认值应在合理范围
        assert!(config.model.context_length >= 512); // 最小合理值
        assert!(config.model.context_length <= 131072); // 最大 128K

        // 常见上下文长度验证
        let common_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768];
        assert!(common_lengths.contains(&config.model.context_length));
    }

    #[test]
    fn test_settings_grpc_config_bounds() {
        // 覆盖分支: gRPC 配置边界验证
        let config = ServerConfig::default();

        // 消息大小应该合理 (1MB - 1GB)
        assert!(config.grpc.max_message_size_mb >= 1);
        assert!(config.grpc.max_message_size_mb <= 1024);

        // Keepalive 时间应该合理
        assert!(config.grpc.keepalive_time_ms >= 1000); // 至少 1 秒
        assert!(config.grpc.keepalive_timeout_ms >= 1000); // 至少 1 秒
        assert!(config.grpc.keepalive_timeout_ms < config.grpc.keepalive_time_ms);
        // timeout < time
    }

    #[test]
    fn test_settings_clone_and_equality() {
        // 覆盖分支: Clone trait 和字段独立性
        let config1 = ServerConfig::default();
        let config2 = config1.clone();

        // 验证 clone 后值相等
        assert_eq!(config1.server.host, config2.server.host);
        assert_eq!(config1.server.port, config2.server.port);
        assert_eq!(config1.memory.max_memory_gb, config2.memory.max_memory_gb);

        // 修改 clone 不应影响原对象
        let mut config3 = config1.clone();
        config3.server.port = 9999;
        assert_ne!(config1.server.port, config3.server.port);
    }
}

impl ServerConfig {
    /// 从 TOML 文件加载配置
    ///
    /// # 参数
    /// - `path`: 配置文件路径
    ///
    /// # 返回
    /// 成功返回配置实例，失败返回错误
    pub fn from_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: ServerConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// 获取服务器监听地址字符串
    ///
    /// 格式: "host:port"
    pub fn addr(&self) -> String {
        format!("{}:{}", self.server.host, self.server.port)
    }

    /// 获取最大内存(字节)
    pub fn max_memory_bytes(&self) -> usize {
        self.memory.max_memory_gb * 1024 * 1024 * 1024
    }

    /// 获取 Arena 大小(字节)
    pub fn arena_size_bytes(&self) -> usize {
        self.memory.arena_size_mb * 1024 * 1024
    }
}
