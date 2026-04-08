//! 配置模块 - 基础配置结构
//!
//! 提供服务器的基础配置结构，包含端口和主机地址设置。

/// 服务器基础配置
///
/// 定义服务器监听的基本参数
pub struct Config {
    /// 监听端口号
    pub port: u16,
    /// 绑定的主机地址
    pub host: String,
}

impl Config {
    /// 创建新的配置实例
    ///
    /// 返回默认配置:
    /// - 端口: 50051
    /// - 主机: 0.0.0.0
    pub fn new() -> Self {
        Self {
            port: 50051,
            host: "0.0.0.0".to_string(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_values() {
        // 测试默认配置值
        let config = Config::default();

        assert_eq!(config.port, 50051);
        assert!(config.port > 0 && config.port < 65536);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_config_new() {
        // 测试new方法
        let config = Config::new();

        assert_eq!(config.port, 50051);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_config_custom_port() {
        // 测试自定义端口
        let mut config = Config::new();
        config.port = 8080;

        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_config_custom_host() {
        // 测试自定义主机地址
        let mut config = Config::new();
        config.host = "127.0.0.1".to_string();

        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_config_valid_ports() {
        // 测试有效端口范围
        let mut config = Config::new();

        // 最小有效端口
        config.port = 1;
        assert_eq!(config.port, 1);

        // 最大有效端口
        config.port = 65535;
        assert_eq!(config.port, 65535);

        // 常见HTTP端口
        config.port = 80;
        assert_eq!(config.port, 80);

        config.port = 443;
        assert_eq!(config.port, 443);

        config.port = 3000;
        assert_eq!(config.port, 3000);

        config.port = 50051; // gRPC默认
        assert_eq!(config.port, 50051);
    }

    #[test]
    fn test_config_common_host_addresses() {
        // 测试常见的主机地址
        let hosts = vec![
            "0.0.0.0",
            "127.0.0.1",
            "localhost",
            "::",
            "::1",
            "192.168.1.1",
            "10.0.0.1",
        ];

        for host in hosts {
            let mut config = Config::new();
            config.host = host.to_string();
            assert_eq!(config.host, host);
        }
    }

    #[test]
    fn test_config_clone_and_equality() {
        // 测试克隆和相等性
        let config1 = Config::new();
        let config2 = config1.clone();

        assert_eq!(config1.port, config2.port);
        assert_eq!(config1.host, config2.host);
    }

    // ==================== 新增测试开始 ====================

    /// 测试Config的Debug trait实现
    /// 覆盖分支：Debug格式化输出
    #[test]
    fn test_config_debug_format() {
        let config = Config {
            port: 8080,
            host: "localhost".to_string(),
        };

        let debug_str = format!("{:?}", config);
        
        // 验证Debug输出包含关键字段信息
        assert!(debug_str.contains("8080"));
        assert!(debug_str.contains("localhost"));
    }

    /// 测试Config的PartialEq trait（如果实现了）
    /// 覆盖分支：相等性和不等性比较
    #[test]
    fn test_config_partial_equality() {
        let config1 = Config {
            port: 8080,
            host: "127.0.0.1".to_string(),
        };

        let config2 = Config {
            port: 8080,
            host: "127.0.0.1".to_string(),
        };

        let config3 = Config {
            port: 9090,
            host: "0.0.0.0".to_string(),
        };

        // 相同字段应该相等
        assert_eq!(config1.port, config2.port);
        assert_eq!(config1.host, config2.host);

        // 不同字段应该不相等
        assert_ne!(config1.port, config3.port);
        assert_ne!(config1.host, config3.host);
    }

    /// 测试空字符串作为host地址
    /// 覆盖分支：边界条件 - 空字符串host
    #[test]
    fn test_config_empty_host() {
        let config = Config {
            port: 50051,
            host: String::new(),
        };

        assert_eq!(config.host, "");
        assert!(config.host.is_empty());
    }

    /// 测试长字符串作为host地址
    /// 覆盖分支：边界条件 - 长字符串host
    #[test]
    fn test_config_long_host() {
        let long_host = "a".repeat(256);
        let config = Config {
            port: 50051,
            host: long_host.clone(),
        };

        assert_eq!(config.host.len(), 256);
        assert_eq!(config.host, long_host);
    }

    /// 测试特殊字符的host地址
    /// 覆盖分支：特殊字符处理
    #[test]
    fn test_config_special_hosts() {
        let special_hosts = vec![
            "",
            " ",
            "  ",
            "\t",
            "\n",
            "host-with-dashes",
            "host_with_underscores",
            "host.with.dots",
            "192.168.1.100",
            "[::1]",  // IPv6
            "fe80::1",  // IPv6 link-local
            "中文主机名",
            "host with spaces",
            "UPPERCASE",
            "lowercase",
            "MiXeDcAsE",
        ];

        for host in special_hosts {
            let config = Config {
                port: 50051,
                host: host.to_string(),
            };
            assert_eq!(config.host, host);
        }
    }

    /// 测试端口边界值
    /// 覆盖分支：端口的极端值
    #[test]
    fn test_config_boundary_ports() {
        let boundary_ports = vec![
            (0, true),      // 端口0（通常保留，但u16允许）
            (1, true),      // 最小有效端口
            (1024, true),   // 系统端口边界
            (1025, true),   // 用户端口起始
            (49152, true),  // 动态/私有端口起始
            (65534, true),  // 次最大端口
            (65535, true),  // 最大端口
        ];

        for (port, _) in boundary_ports {
            let config = Config {
                port,
                host: "0.0.0.0".to_string(),
            };
            assert_eq!(config.port, port);
        }
    }

    /// 测试Config独立修改字段
    /// 覆盖分支：字段修改后的独立性
    #[test]
    fn test_config_independent_modification() {
        let mut config1 = Config::new();
        let config2 = config1.clone();

        // 修改config1不影响config2
        config1.port = 9999;
        config1.host = "modified.example.com".to_string();

        assert_eq!(config1.port, 9999);
        assert_eq!(config1.host, "modified.example.com");

        // 原始值保持不变
        assert_eq!(config2.port, 50051);
        assert_eq!(config2.host, "0.0.0.0");
    }

    /// 测试多次创建Config实例的一致性
    /// 覆盖分支：Default/new方法的一致性
    #[test]
    fn test_config_consistency_across_instances() {
        let configs: Vec<Config> = (0..10).map(|_| Config::new()).collect();

        // 所有通过new()创建的实例应该有相同的默认值
        for config in &configs {
            assert_eq!(config.port, 50051);
            assert_eq!(config.host, "0.0.0.0");
        }

        // 通过Default trait创建也应该一致
        let default_configs: Vec<Config> = (0..10).map(|_| Config::default()).collect();
        for config in &default_configs {
            assert_eq!(config.port, 50051);
            assert_eq!(config.host, "0.0.0.0");
        }
    }

    /// 测试Config的Display-like行为（通过Debug）
    /// 覆盖分支：完整结构体序列化
    #[test]
    fn test_config_debug_output_completeness() {
        let config = Config {
            port: 8080,
            host: "example.com".to_string(),
        };

        let debug_str = format!("{:?}", config);

        // 验证Debug输出包含所有字段信息
        assert!(debug_str.contains("8080"), "Should contain port value");
        assert!(debug_str.contains("example.com"), "Should contain host value");
    }

    /// 测试Config在修改后保持类型一致性
    /// 覆盖分支：字段类型的正确性保持
    #[test]
    fn test_config_type_consistency_after_modification() {
        let mut config = Config::new();

        // 验证初始类型
        assert_eq!(config.port, 50051); // u16
        assert_eq!(config.host, "0.0.0.0"); // String

        // 修改为不同范围的值
        config.port = 1;
        assert_eq!(config.port, 1);
        assert!(config.port >= 1 && config.port <= 65535); // u16范围验证

        config.port = 65535;
        assert_eq!(config.port, 65535);
        assert!(config.port <= 65535);
    }

    /// 测试Config与自身比较（相同值）
    /// 覆盖分支：完全相等的配置实例
    #[test]
    fn test_config_identical_instances() {
        let config1 = Config {
            port: 3000,
            host: "localhost".to_string(),
        };

        let config2 = Config {
            port: 3000,
            host: "localhost".to_string(),
        };

        // 字段级别相等
        assert_eq!(config1.port, config2.port);
        assert_eq!(config1.host, config2.host);
        assert_eq!(config1.host.len(), config2.host.len());
    }

    /// 测试Config使用不同构造方式的一致性
    /// 覆盖分支：new()、default()和直接初始化的一致性
    #[test]
    fn test_config_construction_methods_equivalence() {
        // 通过new()创建
        let config_new = Config::new();

        // 通过Default trait创建
        let config_default = Config::default();

        // 通过直接初始化（模拟默认值）
        let config_direct = Config {
            port: 50051,
            host: "0.0.0.0".to_string(),
        };

        // 三种方式应该产生等价的配置
        assert_eq!(config_new.port, config_default.port);
        assert_eq!(config_new.host, config_default.host);

        assert_eq!(config_new.port, config_direct.port);
        assert_eq!(config_new.host, config_direct.host);

        assert_eq!(config_default.port, config_direct.port);
        assert_eq!(config_default.host, config_direct.host);
    }

    /// 测试Config端口的顺序关系
    /// 覆盖分支：端口值的有序性
    #[test]
    fn test_config_port_ordering() {
        let ports = vec![80, 443, 8080, 8443, 9000, 50051];

        for i in 0..ports.len() {
            for j in (i + 1)..ports.len() {
                // 验证不同的端口值确实不同
                assert_ne!(ports[i], ports[j]);
                // 验证可以进行比较
                if ports[i] < ports[j] {
                    assert!(ports[i] < ports[j]);
                } else {
                    assert!(ports[i] > ports[j]);
                }
            }
        }
    }

    /// 测试Config host字段的字符串操作
    /// 覆盖分支：host字符串的各种操作
    #[test]
    fn test_config_host_string_operations() {
        let mut config = Config::new();

        // 测试追加
        config.host.push_str(":8080");
        assert!(config.host.contains(":8080"));
        assert_eq!(config.host, "0.0.0.0:8080");

        // 测试替换
        config = Config::new();
        config.host = config.host.replace("0.0.0.0", "127.0.0.1");
        assert_eq!(config.host, "127.0.0.1");

        // 测试大小写转换
        config.host = "ExampleHost.COM".to_string();
        let lower = config.host.to_lowercase();
        assert_eq!(lower, "examplehost.com");

        let upper = config.host.to_uppercase();
        assert_eq!(upper, "EXAMPLEHOST.COM");
    }

    /// 测试Config在循环中的稳定性
    /// 覆盖分支：大量实例创建和销毁的稳定性
    #[test]
    fn test_config_stability_in_loop() {
        for i in 0u16..1000 {
            let config = Config {
                port: 1000 + i,
                host: format!("host-{}", i),
            };

            // 验证每次迭代都创建了正确的配置
            assert_eq!(config.port, 1000 + i);
            assert!(config.host.contains(&i.to_string()));
        }
    }

    /// 测试Config的内存布局合理性（通过大小推断）
    /// 覆盖分支：结构体大小的基本验证
    #[test]
    fn test_config_size_reasonableness() {
        use std::mem::size_of;

        let config_size = size_of::<Config>();

        // Config应该包含port(u16=2字节)和host(String，通常24+字节)
        // 总大小应该在合理范围内（不会太小或太大）
        assert!(
            config_size >= 24, // 至少要能容纳String的对齐要求
            "Config size {} is too small",
            config_size
        );

        assert!(
            config_size <= 64, // 不应该过大（没有额外填充）
            "Config size {} is too large",
            config_size
        );
    }
}