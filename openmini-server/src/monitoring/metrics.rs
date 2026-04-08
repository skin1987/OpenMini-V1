//! Prometheus 指标定义和收集
//!
//! 定义所有监控指标，包括推理指标、系统资源指标、DSA/量化指标和 Continuous Batching 指标。
//!
//! ## 指标分类
//!
//! ### 推理指标 (Inference Metrics)
//! - 推理请求总数、延迟、吞吐量、活跃请求数、批处理大小
//!
//! ### 系统资源指标 (System Resource Metrics)
//! - GPU利用率、显存使用、CPU使用率、内存使用量、KV Cache大小
//!
//! ### DSA/量化指标 (DSA/Quantization Metrics)
//! - DSA Top-K选择时间、反量化操作计数、反量化吞吐量
//!
//! ### Continuous Batching指标 (Continuous Batching Metrics)
//! - 调度队列长度、抢占次数、平均等待时间

use once_cell::sync::Lazy;
use prometheus::{
    opts, register_int_counter_vec, register_histogram_vec, register_gauge,
    register_int_counter, register_histogram, IntCounterVec, HistogramVec,
    IntCounter, Histogram, Gauge,
};

// ============================================================================
// 推理指标 (Inference Metrics)
// ============================================================================

/// 推理请求总数
///
/// 记录所有推理请求，按模型名称和状态（success/error）分类。
/// Label: model_name, status (success/error)
pub static INFERENCE_REQUESTS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!(
        "openmini_inference_requests_total",
        "Total number of inference requests",
        &["model_name", "status"]
    ).expect("Cannot register inference_requests_total counter")
});

/// 推理延迟直方图 (ms)
///
/// 记录推理请求的端到端延迟分布。
/// Label: model_name, quantization, backend
/// Buckets: [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000] ms
pub static INFERENCE_LATENCY_HISTOGRAM: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "openmini_inference_latency_ms",
        "Inference latency in milliseconds",
        &["model_name", "quantization", "backend"],
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0]
    ).expect("Cannot register inference_latency histogram")
});

/// Token生成吞吐量 (tokens/second)
///
/// 实时记录当前token生成速率。
pub static TOKEN_THROUGHPUT: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_token_throughput",
            "Token generation throughput in tokens per second"
        )
    ).expect("Cannot register token_throughput gauge")
});

/// 活跃请求数
///
/// 当前正在处理的推理请求数量。
pub static ACTIVE_REQUESTS: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_active_requests",
            "Number of currently active inference requests"
        )
    ).expect("Cannot register active_requests gauge")
});

/// 批处理大小分布
///
/// 当前批处理中的请求数量。
pub static BATCH_SIZE: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_batch_size",
            "Current batch size for inference"
        )
    ).expect("Cannot register batch_size gauge")
});

// ============================================================================
// 系统资源指标 (System Resource Metrics)
// ============================================================================

/// GPU利用率 (%)
///
/// 实时GPU计算单元利用率，范围 0-100。
pub static GPU_UTILIZATION: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_gpu_utilization_percent",
            "GPU utilization percentage (0-100)"
        )
    ).expect("Cannot register gpu_utilization gauge")
});

/// GPU显存使用 (bytes)
///
/// 当前GPU显存使用量（字节）。
pub static GPU_MEMORY_USED: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_gpu_memory_used_bytes",
            "GPU memory used in bytes"
        )
    ).expect("Cannot register gpu_memory_used gauge")
});

/// CPU使用率 (%)
///
/// CPU整体使用率，范围 0-100。
pub static CPU_USAGE: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_cpu_usage_percent",
            "CPU usage percentage (0-100)"
        )
    ).expect("Cannot register cpu_usage gauge")
});

/// 内存使用量 (bytes)
///
/// 进程当前内存使用量（字节）。
pub static MEMORY_USED: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_memory_used_bytes",
            "Memory used by the process in bytes"
        )
    ).expect("Cannot register memory_used gauge")
});

/// KV Cache大小 (bytes)
///
/// KV Cache占用的总内存（字节）。
pub static KV_CACHE_SIZE: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_kv_cache_size_bytes",
            "Total KV cache size in bytes"
        )
    ).expect("Cannot register kv_cache_size gauge")
});

// ============================================================================
// DSA/量化指标 (DSA/Quantization Metrics)
// ============================================================================

/// DSA Top-K选择时间 (μs)
///
/// 记录动态稀疏注意力中Top-K选择的耗时分布。
pub static DSA_TOP_K_LATENCY: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "openmini_dsa_top_k_latency_us",
        "DSA Top-K selection latency in microseconds",
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    ).expect("Cannot register dsa_top_k_latency histogram")
});

/// 反量化操作计数
///
/// 累计执行的反量化操作次数。
pub static DEQUANTIZE_OPS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(
        "openmini_dequantize_ops_total",
        "Total number of dequantize operations"
    ).expect("Cannot register dequantize_ops_total counter")
});

/// 反量化吞吐量 (elements/second)
///
/// 反量化操作的处理速率。
pub static DEQUANTIZE_THROUGHPUT: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_dequantize_throughput",
            "Dequantize throughput in elements per second"
        )
    ).expect("Cannot register dequantize_throughput gauge")
});

// ============================================================================
// Continuous Batching指标 (Continuous Batching Metrics)
// ============================================================================

/// 调度队列长度
///
/// 等待调度的请求数量。
pub static SCHEDULER_QUEUE_LENGTH: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_scheduler_queue_length",
            "Number of requests waiting in scheduler queue"
        )
    ).expect("Cannot register scheduler_queue_length gauge")
});

/// 抢占次数
///
/// 由于资源限制被抢占的请求累计次数。
pub static PREEMPTIONS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    register_int_counter!(
        "openmini_preemptions_total",
        "Total number of request preemptions"
    ).expect("Cannot register preemptions_total counter")
});

/// 平均等待时间 (ms)
///
/// 请求在调度队列中的平均等待时间。
pub static AVG_WAIT_TIME: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "openmini_avg_wait_time_ms",
            "Average wait time in scheduler queue (milliseconds)"
        )
    ).expect("Cannot register avg_wait_time gauge")
});

// ============================================================================
// 辅助函数：便捷的指标记录方法
// ============================================================================

/// 记录推理请求开始
///
/// 增加活跃请求数计数器。
#[inline]
pub fn record_inference_start() {
    ACTIVE_REQUESTS.inc();
}

/// 记录推理请求完成
///
/// 减少活跃请求数，并记录请求结果。
///
/// # 参数
///
/// - `model_name`: 模型名称
/// - `success`: 是否成功
/// - `latency_ms`: 延迟（毫秒）
/// - `quantization`: 量化类型
/// - `backend`: 后端类型
#[inline]
pub fn record_inference_complete(
    model_name: &str,
    success: bool,
    latency_ms: f64,
    quantization: &str,
    backend: &str,
) {
    ACTIVE_REQUESTS.dec();
    let status = if success { "success" } else { "error" };
    INFERENCE_REQUESTS_TOTAL
        .with_label_values(&[model_name, status])
        .inc();
    INFERENCE_LATENCY_HISTOGRAM
        .with_label_values(&[model_name, quantization, backend])
        .observe(latency_ms);
}

/// 更新Token吞吐量
///
/// # 参数
///
/// - `throughput`: tokens/second
#[inline]
pub fn update_token_throughput(throughput: f64) {
    TOKEN_THROUGHPUT.set(throughput);
}

/// 更新批处理大小
///
/// # 参数
///
/// - `size`: 当前批次大小
#[inline]
pub fn update_batch_size(size: f64) {
    BATCH_SIZE.set(size);
}

/// 更新GPU指标
///
/// # 参数
///
/// - `utilization`: GPU利用率 (0-100)
/// - `memory_used_bytes`: 显存使用量（字节）
#[inline]
pub fn update_gpu_metrics(utilization: f64, memory_used_bytes: u64) {
    GPU_UTILIZATION.set(utilization);
    GPU_MEMORY_USED.set(memory_used_bytes as f64);
}

/// 更新CPU和内存指标
///
/// # 参数
///
/// - `cpu_percent`: CPU使用率 (0-100)
/// - `memory_bytes`: 内存使用量（字节）
#[inline]
pub fn update_system_metrics(cpu_percent: f64, memory_bytes: u64) {
    CPU_USAGE.set(cpu_percent);
    MEMORY_USED.set(memory_bytes as f64);
}

/// 更新KV Cache大小
///
/// # 参数
///
/// - `size_bytes`: KV Cache大小（字节）
#[inline]
pub fn update_kv_cache_size(size_bytes: u64) {
    KV_CACHE_SIZE.set(size_bytes as f64);
}

/// 记录DSA Top-K操作延迟
///
/// # 参数
///
/// - `latency_us`: 延迟（微秒）
#[inline]
pub fn record_dsa_top_k_latency(latency_us: f64) {
    DSA_TOP_K_LATENCY.observe(latency_us);
}

/// 记录反量化操作
///
/// # 参数
///
/// - `element_count`: 处理的元素数量
#[inline]
pub fn record_dequantize_op(_element_count: usize) {
    DEQUANTIZE_OPS_TOTAL.inc();
}

/// 更新反量化吞吐量
///
/// # 参数
///
/// - `throughput`: elements/second
#[inline]
pub fn update_dequantize_throughput(throughput: f64) {
    DEQUANTIZE_THROUGHPUT.set(throughput);
}

/// 更新调度器指标
///
/// # 参数
///
/// - `queue_length`: 队列长度
/// - `avg_wait_ms`: 平均等待时间（毫秒）
#[inline]
pub fn update_scheduler_metrics(queue_length: f64, avg_wait_ms: f64) {
    SCHEDULER_QUEUE_LENGTH.set(queue_length);
    AVG_WAIT_TIME.set(avg_wait_ms);
}

/// 记录抢占事件
#[inline]
pub fn record_preemption() {
    PREEMPTIONS_TOTAL.inc();
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试推理指标初始化
    #[test]
    fn test_inference_metrics_initialization() {
        // 验证所有推理指标都能正常访问和操作（不假设初始值为0）
        let _active = ACTIVE_REQUESTS.get();
        let _throughput = TOKEN_THROUGHPUT.get();
        let _batch = BATCH_SIZE.get();

        // 验证指标可以被正常操作
        INFERENCE_REQUESTS_TOTAL
            .with_label_values(&["init-test", "success"])
            .inc();
        let val = INFERENCE_REQUESTS_TOTAL
            .with_label_values(&["init-test", "success"])
            .get();
        assert!(val >= 1, "Counter should be at least 1 after inc()");
    }

    /// 测试系统资源指标可访问性
    #[test]
    fn test_system_metrics_accessible() {
        // 验证所有系统指标都能正常访问（不假设初始值）
        let _gpu_util = GPU_UTILIZATION.get();
        let _gpu_mem = GPU_MEMORY_USED.get();
        let _cpu = CPU_USAGE.get();
        let _mem = MEMORY_USED.get();
        let _kv_cache = KV_CACHE_SIZE.get();
    }

    /// 测试DSA/量化指标初始化
    #[test]
    fn test_dsa_quantization_metrics_initialization() {
        // 验证DSA/量化指标能正常访问和操作（不假设初始值为0）
        let _ops_before = DEQUANTIZE_OPS_TOTAL.get();
        let _throughput_before = DEQUANTIZE_THROUGHPUT.get();

        // 验证指标可以被正常操作
        DEQUANTIZE_OPS_TOTAL.inc();
        let ops_after = DEQUANTIZE_OPS_TOTAL.get();
        assert!(ops_after > _ops_before, "Counter should increase after inc()");

        DEQUANTIZE_THROUGHPUT.set(999.0);
        assert!((DEQUANTIZE_THROUGHPUT.get() - 999.0).abs() < 0.01);
    }

    /// 测试Continuous Batching指标可访问性
    #[test]
    fn test_continuous_batching_metrics_accessible() {
        // 验证所有调度指标都能正常访问
        let _queue_len = SCHEDULER_QUEUE_LENGTH.get();
        let _preemptions = PREEMPTIONS_TOTAL.get();
        let _avg_wait = AVG_WAIT_TIME.get();
    }

    /// 测试推理请求记录流程
    #[test]
    fn test_record_inference_flow() {
        // 记录当前值作为基准
        let active_before = ACTIVE_REQUESTS.get();

        // 开始推理
        record_inference_start(); // active += 1
        assert!((ACTIVE_REQUESTS.get() - (active_before + 1.0)).abs() < 0.001);

        // 完成推理
        record_inference_complete(
            "test-model",
            true,
            50.5,
            "q4_0",
            "metal",
        );
        // 验证活跃请求数回到基准值（start和complete配对）
        assert!((ACTIVE_REQUESTS.get() - active_before).abs() < 0.001);

        // 验证计数器增加
        let count = INFERENCE_REQUESTS_TOTAL
            .with_label_values(&["test-model", "success"])
            .get();
        assert!(count >= 1, "Success counter should be at least 1");
    }

    /// 测试GPU指标更新
    #[test]
    fn test_update_gpu_metrics() {
        let test_value = 85.5;
        let test_memory = 1024.0 * 1024.0 * 1024.0; // 1GB
        update_gpu_metrics(test_value, test_memory as u64);
        // 验证值被设置（由于全局状态共享，使用较宽松的检查）
        let gpu_val = GPU_UTILIZATION.get();
        assert!(gpu_val > 0.0, "GPU utilization should be positive, got {}", gpu_val);
        assert!((gpu_val - test_value).abs() < 1.0, "GPU utilization should be close to {}, got {}", test_value, gpu_val);

        let mem_val = GPU_MEMORY_USED.get();
        assert!(mem_val > 0.0, "GPU memory should be positive, got {}", mem_val);
    }

    /// 测试系统指标更新
    #[test]
    fn test_update_system_metrics() {
        update_system_metrics(45.2, 512 * 1024 * 512); // 45.2%, 256MB
        assert!((CPU_USAGE.get() - 45.2).abs() < 0.01);
        assert!((MEMORY_USED.get() - 268435456.0).abs() < 1.0);
    }

    /// 测试调度器指标更新
    #[test]
    fn test_update_scheduler_metrics() {
        update_scheduler_metrics(10.0, 25.5);
        assert!((SCHEDULER_QUEUE_LENGTH.get() - 10.0).abs() < 0.01);
        assert!((AVG_WAIT_TIME.get() - 25.5).abs() < 0.01);
    }

    /// 测试反量化操作记录
    #[test]
    fn test_record_dequantize_ops() {
        // 记录当前值作为基准
        let ops_before = DEQUANTIZE_OPS_TOTAL.get();

        record_dequantize_op(1024);
        let ops_after_1 = DEQUANTIZE_OPS_TOTAL.get();
        assert_eq!(ops_after_1, ops_before + 1);

        update_dequantize_throughput(1000000.0); // 1M elements/s
        assert!((DEQUANTIZE_THROUGHPUT.get() - 1000000.0).abs() < 0.01);
    }

    /// 测试抢占事件记录
    #[test]
    fn test_record_preemption() {
        record_preemption();
        assert_eq!(PREEMPTIONS_TOTAL.get(), 1);

        record_preemption();
        assert_eq!(PREEMPTIONS_TOTAL.get(), 2);
    }

    /// 测试KV Cache大小更新
    #[test]
    fn test_update_kv_cache_size() {
        update_kv_cache_size(256 * 1024 * 1024); // 256MB
        assert!((KV_CACHE_SIZE.get() - 268435456.0).abs() < 1.0);
    }

    /// 测试Token吞吐量和批处理大小更新
    #[test]
    fn test_update_throughput_and_batch() {
        update_token_throughput(150.5);
        assert!((TOKEN_THROUGHPUT.get() - 150.5).abs() < 0.01);

        update_batch_size(8.0);
        assert!((BATCH_SIZE.get() - 8.0).abs() < 0.01);
    }

    // ===== 边界条件和分支覆盖率测试 =====

    #[test]
    fn test_metrics_increment_various_labels() {
        // 使用不同label组合增加计数器，验证它们是独立的
        INFERENCE_REQUESTS_TOTAL
            .with_label_values(&["model-a", "success"])
            .inc();
        INFERENCE_REQUESTS_TOTAL
            .with_label_values(&["model-b", "error"])
            .inc_by(5);

        // 验证不同label是独立的计数器
        let model_a_success = INFERENCE_REQUESTS_TOTAL
            .with_label_values(&["model-a", "success"])
            .get();
        let model_b_error = INFERENCE_REQUESTS_TOTAL
            .with_label_values(&["model-b", "error"])
            .get();

        assert_eq!(model_a_success, 1, "model-a success count should be 1");
        assert_eq!(model_b_error, 5, "model-b error count should be 5");
    }

    #[test]
    fn test_histogram_observe_various_values() {
        // 观察不同范围的延迟值
        INFERENCE_LATENCY_HISTOGRAM
            .with_label_values(&["model-x", "q4_0", "cpu"])
            .observe(0.5);   // < 1ms bucket
        INFERENCE_LATENCY_HISTOGRAM
            .with_label_values(&["model-x", "q4_0", "cpu"])
            .observe(15.0);  // 10-25ms bucket
        INFERENCE_LATENCY_HISTOGRAM
            .with_label_values(&["model-x", "q4_0", "cpu"])
            .observe(3000.0); // > 2500ms bucket

        // 验证直方图接受这些值（不会panic）
        // 实际的bucket计数需要通过gather_metrics验证
    }

    #[test]
    fn test_gauge_set_and_dec_operations() {
        // 测试Gauge的set、inc、dec操作
        ACTIVE_REQUESTS.set(10.0);
        assert!((ACTIVE_REQUESTS.get() - 10.0).abs() < 0.001);

        ACTIVE_REQUESTS.inc();
        assert!((ACTIVE_REQUESTS.get() - 11.0).abs() < 0.001);

        ACTIVE_REQUESTS.dec();
        assert!((ACTIVE_REQUESTS.get() - 10.0).abs() < 0.001);

        // 使用set来模拟减少
        ACTIVE_REQUESTS.set(5.0);
        assert!((ACTIVE_REQUESTS.get() - 5.0).abs() < 0.001);

        // 使用set来模拟增加
        ACTIVE_REQUESTS.set(8.0);
        assert!((ACTIVE_REQUESTS.get() - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_gauge_set_zero_and_negative_handling() {
        // 测试设置零和接近零的值
        GPU_UTILIZATION.set(0.0);
        assert!((GPU_UTILIZATION.get() - 0.0).abs() < 0.001);

        TOKEN_THROUGHPUT.set(0.001); // 很小的正数
        assert!(TOKEN_THROUGHPUT.get() > 0.0);
    }

    #[test]
    fn test_counter_inc_by_multiple() {
        // 测试批量增加计数器
        let initial = DEQUANTIZE_OPS_TOTAL.get();
        
        // 使用多次inc来模拟批量增加
        for _ in 0..10 {
            DEQUANTIZE_OPS_TOTAL.inc();
        }
        assert_eq!(DEQUANTIZE_OPS_TOTAL.get(), initial + 10);

        for _ in 0..100 {
            DEQUANTIZE_OPS_TOTAL.inc();
        }
        assert_eq!(DEQUANTIZE_OPS_TOTAL.get(), initial + 110);
    }

    #[test]
    fn test_inference_complete_with_different_statuses() {
        // 测试成功和失败状态的记录
        // 注意：由于全局状态共享且测试可能并行执行，
        // 我们只验证操作本身的有效性（不panic）和计数器记录

        // 执行 start/complete 操作（验证不会panic）
        record_inference_start();
        record_inference_complete("test-model-1", true, 50.0, "q4_0", "metal");

        record_inference_start();
        record_inference_complete("test-model-2", false, 5000.0, "q4_1", "cpu");

        // 验证不同状态被分别记录（使用 >= 因为可能有其他测试的残留）
        let success_count = INFERENCE_REQUESTS_TOTAL
            .with_label_values(&["test-model-1", "success"])
            .get();
        let error_count = INFERENCE_REQUESTS_TOTAL
            .with_label_values(&["test-model-2", "error"])
            .get();

        assert!(success_count >= 1, "Success count should be at least 1, got {}", success_count);
        assert!(error_count >= 1, "Error count should be at least 1, got {}", error_count);
    }

    #[test]
    fn test_dsa_latency_histogram_observation() {
        // 测试DSA延迟直方图观察各种微秒级延迟
        DSA_TOP_K_LATENCY.observe(0.5);   // < 1μs
        DSA_TOP_K_LATENCY.observe(5.0);   // 5μs
        DSA_TOP_K_LATENCY.observe(50.0);  // 50μs
        DSA_TOP_K_LATENCY.observe(500.0); // 500μs
        
        // 验证没有panic即可（实际值通过prometheus格式验证）
    }

    #[test]
    fn test_multiple_model_metrics_isolation() {
        // 测试多个模型的指标相互隔离
        update_token_throughput(100.0);
        update_batch_size(4.0);

        // 更新同一指标用于不同目的（模拟多模型场景）
        let throughput_before = TOKEN_THROUGHPUT.get();
        let batch_before = BATCH_SIZE.get();

        update_token_throughput(200.0);
        update_batch_size(8.0);

        // Gauge应该覆盖之前的值
        assert!((TOKEN_THROUGHPUT.get() - 200.0).abs() < 0.01);
        assert!((BATCH_SIZE.get() - 8.0).abs() < 0.01);
        
        // 确认之前的不同
        assert!((throughput_before - 100.0).abs() < 0.01);
        assert!((batch_before - 4.0).abs() < 0.01);
    }
}
