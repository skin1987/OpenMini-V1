// 监控模块集成测试
//
// 验证 Prometheus 指标导出、Health Check 端点、敏感数据脱敏等功能。

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
// 敏感数据过滤测试（待实现：SensitiveDataFilter 模块尚未创建）
// ============================================================================

// #[cfg(test)]
// mod sensitive_data_filter_tests {
//     use openmini_server::monitoring::sensitive_data_filter::SensitiveDataFilter;
//
//     #[test]
//     fn test_filter_api_key() {
//         let filter = SensitiveDataFilter::new();
//         let input = "API key: sk-abc123def456";
//         let filtered = filter.filter(input);
//         assert!(!filtered.contains("sk-abc123def456"), "API key should be masked");
//         assert!(filtered.contains("sk-***"), "Should show masked pattern");
//     }
//
//     #[test]
//     fn test_filter_password() {
//         let filter = SensitiveDataFilter::new();
//         let input = "password: mySecret123";
//         let filtered = filter.filter(input);
//         assert!(
//             !filtered.contains("mySecret123"),
//             "Password should be masked"
//         );
//     }
//
//     #[test]
//     fn test_preserve_normal_content() {
//         let filter = SensitiveDataFilter::new();
//         let input = "Hello, this is a normal message";
//         let filtered = filter.filter(input);
//         assert_eq!(filtered, input, "Normal content should be preserved");
//     }
// }

// ============================================================================
// 端到端集成测试
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use openmini_server::monitoring::business_metrics::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_full_monitoring_pipeline() {
        // 模拟完整的监控流程
        record_inference_completion("integration-test-model", 42, true);
        observe_request_duration("/v1/chat/completions", "http", Duration::from_millis(85));
        update_kv_cache_usage(1, 512 * 1024); // 512 KB
        update_worker_queue_length(5);

        // 验证所有指标都可以正常记录（不 panic）
        // 注意：在实际应用中，这些指标会通过 /metrics 端点暴露给 Prometheus
    }

    #[test]
    fn test_concurrent_metric_updates() {
        use std::sync::Arc;
        use std::thread;

        let counter = Arc::new(inference_tokens_total());
        let mut handles = vec![];

        for i in 0..10 {
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                let labeled = c.with_label_values(&["concurrent-test", "success"]);
                labeled.inc_by(i * 10);
            }));
        }

        for h in handles {
            h.join().expect("Thread should not panic");
        }

        // 验证计数器的值
        let final_counter = counter.with_label_values(&["concurrent-test", "success"]);
        let total: u64 = (0..10).map(|i| i * 10).sum();
        assert_eq!(
            final_counter.get(),
            total,
            "Concurrent updates should sum correctly"
        );
    }
}
