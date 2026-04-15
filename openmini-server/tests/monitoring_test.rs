/// 监控模块集成测试
///
/// 验证 Prometheus 指标导出、Health Check 端点、敏感数据脱敏等功能。

// ============================================================================
// Prometheus 业务指标测试
// ============================================================================

#[cfg(test)]
mod prometheus_metrics_tests {
    use openmini_server::monitoring::business_metrics::*;
    use std::time::Duration;

    #[test]
    fn test_inference_tokens_total_counter_exists() {
        // 验证计数器存在（通过调用不会 panic 来验证）
        let _counter = inference_tokens_total();
        // IntCounterVec 没有 get() 方法，只能通过 with_label_values 获取具体标签的计数器
    }

    #[test]
    fn test_record_inference_completion_success() {
        // 记录成功推理
        record_inference_completion("test-model", 100, true);

        // 验证计数器存在且可以获取（通过 remove_label_values 和 get 来验证）
        let counter = inference_tokens_total();
        let labeled_counter = counter.with_label_values(&["test-model", "success"]);
        let value = labeled_counter.get();
        assert_eq!(
            value, 100,
            "Should record 100 tokens for successful inference"
        );
    }

    #[test]
    fn test_record_inference_completion_error() {
        // 记录失败推理
        record_inference_completion("test-model", 50, false);

        let counter = inference_tokens_total();
        let labeled_counter = counter.with_label_values(&["test-model", "error"]);
        let value = labeled_counter.get();
        assert_eq!(value, 50, "Should record 50 tokens for failed inference");
    }

    #[test]
    fn test_request_duration_histogram() {
        // 记录请求耗时
        observe_request_duration("/v1/completions", "http", Duration::from_millis(150));

        // 验证直方图记录了数据（无法直接读取值，但应无 panic）
        // 实际生产中可通过 /metrics endpoint 验证
    }

    #[test]
    fn test_kv_cache_usage_gauge() {
        update_kv_cache_usage(0, 1024 * 1024); // 1 MB

        // 验证 gauge 设置成功（不 panic 即可）
    }

    #[test]
    fn test_worker_queue_length_gauge() {
        update_worker_queue_length(10);
        update_worker_queue_length(0); // 重置
    }
}

// ============================================================================
// Health Check 端点测试
// ============================================================================

#[cfg(test)]
mod health_check_tests {
    use openmini_server::monitoring::health_check::HealthChecker;

    #[tokio::test]
    async fn test_health_check_creation() {
        let checker = HealthChecker::new();
        assert!(checker.check().await.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_status() {
        let checker = HealthChecker::new();
        let status = checker.check().await.unwrap();

        assert!(status.is_healthy());
        // HealthStatus 没有 version() 方法，只检查状态
        assert!(!status.status.is_empty());
    }
}

// ============================================================================
// 敏感数据过滤测试
// ============================================================================

#[cfg(test)]
mod sensitive_data_filter_tests {
    use openmini_server::service::http::sensitive_filter::mask_sensitive_data;

    #[test]
    fn test_mask_api_key_in_header() {
        let input = "api_key=sk-proj-abcdefghijklmnopqrstuvwxyz123456";
        let output = mask_sensitive_data(input);

        assert!(!output.contains("sk-proj"), "API key should be masked");
        assert!(
            output.contains("***MASKED***"),
            "Should contain masked placeholder"
        );
    }

    #[test]
    fn test_mask_bearer_token() {
        let input = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        let output = mask_sensitive_data(input);

        assert!(!output.contains("eyJhbGci"), "JWT should be masked");
    }

    #[test]
    fn test_mask_password() {
        let input = "{\"password\": \"my_secret_password\"}";
        let output = mask_sensitive_data(input);

        assert!(
            !output.contains("my_secret_password"),
            "Password should be masked"
        );
    }

    #[test]
    fn test_mask_credit_card() {
        let input = "card_number=4111111111111111";
        let output = mask_sensitive_data(input);

        assert!(
            !output.contains("4111111111111111"),
            "Credit card should be masked"
        );
    }

    #[test]
    fn test_preserve_normal_data() {
        let input = "Hello, this is normal text without sensitive data";
        let output = mask_sensitive_data(input);

        assert_eq!(input, output, "Normal data should be preserved");
    }

    #[test]
    fn test_mask_multiple_patterns() {
        let input = "api_key=sk-abc password=secret123 token=eyJWT";
        let output = mask_sensitive_data(input);

        assert!(!output.contains("sk-abc"), "API key should be masked");
        assert!(!output.contains("secret123"), "Password should be masked");
        assert!(!output.contains("eyJWT"), "Token should be masked");
    }
}
