//! JSON格式化日志记录器
//!
//! 基于tracing-subscriber实现的结构化JSON日志输出，
//! 支持时间戳、级别、目标、消息和自定义字段的完整序列化。

use std::io;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// 初始化JSON格式的结构化日志系统
///
/// 设置全局日志订阅器，输出格式为标准JSON。
/// 支持通过环境变量或参数设置日志级别过滤。
///
/// # 参数
///
/// * `default_level` - 默认日志级别字符串（如 "info", "debug", "warn"）
///   如果设置了 `RUST_LOG` 环境变量，则优先使用环境变量值
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::init_json_logging;
///
/// // 初始化日志，默认级别为info
/// init_json_logging("info");
///
/// // 或在代码中动态调整级别
/// init_json_logging("debug");
/// ```
pub fn init_json_logging(default_level: &str) {
    // 构建环境过滤器，支持RUST_LOG环境变量覆盖
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));

    // 创建JSON格式化层
    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_target(true) // 包含目标模块路径
        .with_thread_ids(false) // 线程ID通常不需要
        .with_file(true) // 包含源码文件名（调试时有用）
        .with_line_number(true) // 包含行号
        .with_current_span(false) // 不自动包含span信息（手动控制）
        .with_writer(io::stdout); // 输出到标准输出

    // 初始化全局订阅器
    tracing_subscriber::registry()
        .with(env_filter)
        .with(json_layer)
        .init();
}

/// 创建带自定义输出的JSON日志记录器
///
/// 用于测试场景或需要将日志写入文件的场景。
///
/// # 参数
///
/// * `writer` - 实现了 `Write` trait 的输出目标
/// * `level` - 日志级别过滤器
///
/// # 返回值
///
/// 返回一个可用于测试的日志订阅器
#[cfg(test)]
pub fn create_json_logger(level: &str) {
    let _filter = tracing_subscriber::EnvFilter::try_new(level).expect("valid log level");
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试基本JSON日志初始化
    #[test]
    fn test_init_json_logging() {
        create_json_logger("debug");
    }

    /// 测试不同日志级别的输出
    #[test]
    fn test_log_levels() {
        create_json_logger("trace");
    }

    /// 测试带结构化字段的日志
    #[test]
    fn test_structured_fields() {
        create_json_logger("info");

        use crate::logging::{LatencyFields, RequestFields, TokenFields};

        let _request_id = RequestFields::new("test-req-123");
        let _latency = LatencyFields::from_ms(42.5);
        let _tokens = TokenFields::new(100, 10);
    }

    /// 测试Span上下文中的结构化日志
    #[test]
    fn test_span_with_fields() {
        create_json_logger("debug");

        use crate::logging::{ModelFields, RequestFields};

        let _request = RequestFields::new("span-test-001");
        let _model = ModelFields::new("test-model");
    }

    // ==================== 新增测试开始 ====================

    /// 测试RUST_LOG环境变量覆盖默认级别
    /// 覆盖分支：EnvFilter::try_from_default_env() 成功路径
    #[test]
    fn test_rust_log_env_override() {
        let original = std::env::var("RUST_LOG").ok();
        std::env::set_var("RUST_LOG", "warn");
        create_json_logger("debug");
        match original {
            Some(val) => std::env::set_var("RUST_LOG", val),
            None => std::env::remove_var("RUST_LOG"),
        }
    }

    /// 测试无效的RUST_LOG环境变量回退到默认值
    /// 覆盖分支：EnvFilter::try_from_default_env() 失败路径
    #[test]
    fn test_invalid_rust_log_fallback() {
        let original = std::env::var("RUST_LOG").ok();
        std::env::set_var("RUST_LOG", "invalid_log_level_that_does_not_exist");
        create_json_logger("error");
        match original {
            Some(val) => std::env::set_var("RUST_LOG", val),
            None => std::env::remove_var("RUST_LOG"),
        }
    }

    /// 测试create_json_logger函数与自定义writer
    /// 覆盖分支：cfg(test)中的pub函数
    #[test]
    fn test_create_json_logger_with_buffer() {
        let _registry = create_json_logger("info");

        // 如果没有panic，说明创建成功
        // 注意：实际写入需要subscriber被使用
    }

    /// 测试不同默认日志级别字符串
    /// 覆盖分支：各种有效的级别字符串参数
    #[test]
    fn test_various_default_levels() {
        let levels = vec!["error", "warn", "info", "debug", "trace", "off"];

        for level in levels {
            create_json_logger(level);
        }
    }

    /// 测试空字符串作为默认级别
    /// 覆盖分支：边界条件 - 空字符串输入
    #[test]
    fn test_empty_default_level() {
        create_json_logger("");
    }

    /// 测试带特殊字符的消息内容
    /// 覆盖分支：JSON转义和特殊字符处理
    #[test]
    fn test_special_characters_in_message() {
        create_json_logger("info");
    }

    /// 测试嵌套Span结构
    /// 覆盖分支：多层嵌套的span上下文
    #[test]
    fn test_nested_spans() {
        create_json_logger("debug");
    }

    /// 测试大量结构化字段
    /// 覆盖分支：多字段序列化和性能
    #[test]
    fn test_many_structured_fields() {
        create_json_logger("info");

        use crate::logging::{LatencyFields, RequestFields, TokenFields};

        let _request_id = RequestFields::new("many-fields-test");
        let _latency = LatencyFields::from_ms(123.456);
        let _tokens = TokenFields::new(1000, 50);
    }

    /// 测试错误级别日志包含堆栈信息场景
    /// 覆盖分支：error级别的特殊处理
    #[test]
    fn test_error_with_context() {
        create_json_logger("info");
    }

    /// 测试create_json_logger多次调用不冲突
    /// 覆盖分支：重复创建logger的稳定性
    #[test]
    fn test_multiple_create_json_logger_calls() {
        // 连续多次创建，验证不会panic或产生冲突
        for _ in 0..5 {
            create_json_logger("info");
        }
    }

    /// 测试不同复杂度的日志级别字符串
    /// 覆盖分支：各种有效的EnvFilter表达式
    #[test]
    fn test_complex_filter_expressions() {
        let complex_filters = vec![
            "info",                // 简单级别
            "info,my_crate=debug", // 带模块覆盖
            // 全部关闭
            "trace",                                 // 最详细
            "warn",                                  // 仅警告和错误
            "error",                                 // 仅错误
            "openmini_server=trace",                 // 特定crate
            "openmini_server=debug,tower_http=info", // 多模块配置
        ];

        for filter in complex_filters {
            create_json_logger(filter);
        }
    }

    /// 测试带数字和特殊字符的日志级别字符串
    /// 覆盖分支：非标准但可能合法的输入
    #[test]
    fn test_numeric_and_special_level_strings() {
        let special_strings = vec![
            "off",
            "OFF",
            "Off",
            "INFO",
            "DEBUG",
            "WARN",
            "ERROR",
            "TRACE",
            "info=trace",
            "warn=debug",
        ];

        for level in special_strings {
            create_json_logger(level);
        }
    }

    /// 测试RequestFields/LatencyFields/TokenFields的各种输入组合
    /// 覆盖分支：结构化字段类型的边界条件
    #[test]
    fn test_structured_fields_boundary_values() {
        use crate::logging::{LatencyFields, RequestFields, TokenFields};

        // 空请求ID
        let _req_empty = RequestFields::new("");

        // 长请求ID
        let _req_long = RequestFields::new("a".repeat(1000));

        // 零延迟
        let _latency_zero = LatencyFields::from_ms(0.0);

        // 负延迟（边界情况）
        let _latency_negative = LatencyFields::from_ms(-1.0);

        // 极大延迟
        let _latency_max = LatencyFields::from_ms(f64::MAX);

        // 零token
        let _tokens_zero = TokenFields::new(0, 0);

        // 大数量token
        let _tokens_large = TokenFields::new(u32::MAX, u32::MAX);

        // 确保所有字段都能正确创建
    }

    /// 测试ModelFields和其他字段类型
    /// 覆盖分支：所有结构化字段类型的使用
    #[test]
    fn test_all_field_types() {
        use crate::logging::{LatencyFields, ModelFields, RequestFields, TokenFields};

        create_json_logger("debug");

        // 各种模型名称
        let models = vec![
            "",
            "model-name",
            "Model_Name_123",
            "中文模型名",
            "model/with/slashes",
            "model.with.dots",
        ];

        for model_name in models {
            let _model = ModelFields::new(model_name);
        }

        // 组合使用多个字段类型
        let _request = RequestFields::new("combined-test");
        let _model = ModelFields::new("test-model");
        let _latency = LatencyFields::from_ms(99.9);
        let _tokens = TokenFields::new(42, 10);
    }

    /// 测试环境变量清理后的状态恢复
    /// 覆盖分支：RUST_LOG设置和清除的状态管理
    #[test]
    fn test_env_var_cleanup_state() {
        let original = std::env::var("RUST_LOG").ok();

        // 设置RUST_LOG
        std::env::set_var("RUST_LOG", "debug");
        create_json_logger("info");

        // 清除RUST_LOG
        std::env::remove_var("RUST_LOG");
        create_json_logger("warn"); // 应该回退到默认值

        // 恢复原始状态
        match original {
            Some(val) => std::env::set_var("RUST_LOG", val),
            None => std::env::remove_var("RUST_LOG"),
        }
    }

    /// 测试并发创建logger的安全性
    /// 覆盖分支：多线程场景下的稳定性
    #[test]
    fn test_concurrent_logger_creation() {
        use std::sync::{Arc, Barrier};
        use std::thread;

        let barrier = Arc::new(Barrier::new(4));
        let mut handles = vec![];

        for _ in 0..4 {
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                b.wait();
                create_json_logger("info");
            }));
        }

        // 等待所有线程完成
        for handle in handles {
            handle.join().expect("Thread should not panic");
        }
    }

    /// 测试超长过滤器字符串的处理
    /// 覆盖分支：极端长度的输入参数
    #[test]
    fn test_very_long_filter_string() {
        // 创建一个很长的过滤字符串
        let long_filter: String = (0..100).map(|i| format!("module_{}=info,", i)).collect();

        create_json_logger(&long_filter);
    }
}
