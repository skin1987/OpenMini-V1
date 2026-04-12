//! 指标仪表盘 API 模块
//!
//! 提供系统资源监控、推理性能指标和历史趋势数据查询功能。
//! 用于实时监控服务运行状态和性能表现。

use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::AppError;
use crate::AppState;

// ==================== 数据结构定义 ====================

/// 系统资源指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    // CPU 相关
    pub cpu_usage_percent: f64,
    pub cpu_cores: u32,
    pub load_average_1m: f64,
    pub load_average_5m: f64,
    pub load_average_15m: f64,

    // 内存相关
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_free_mb: u64,
    pub memory_usage_percent: f64,

    // GPU 相关（如果可用）
    pub gpu_metrics: Option<GpuMetrics>,

    // 磁盘相关
    pub disk_total_gb: u64,
    pub disk_used_gb: u64,
    pub disk_usage_percent: f64,

    // 时间戳
    pub collected_at: String,
}

/// GPU 详细指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub gpu_name: String,
    pub gpu_utilization_percent: f64,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_free_mb: u64,
    pub temperature_celsius: f64,
    pub power_draw_watts: f64,
    pub fan_speed_percent: f64,
}

/// 推理性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    // 吞吐量相关
    pub requests_per_second: f64,
    pub tokens_per_second: f64,
    pub output_tokens_per_second: f64,

    // 延迟相关
    pub avg_ttft_ms: f64,        // Time To First Token
    pub p50_ttft_ms: f64,
    pub p95_ttft_ms: f64,
    pub p99_ttft_ms: f64,

    pub avg_tpot_ms: f64,        // Time Per Output Token
    pub p50_tpot_ms: f64,
    pub p95_tpot_ms: f64,

    // 总体延迟
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,

    // 队列和并发
    pub active_requests: u64,
    pub queued_requests: u64,
    pub max_concurrent_requests: u64,

    // 成功率
    pub success_rate: f64,
    pub error_rate: f64,
    pub total_requests: u64,

    // 时间戳
    pub calculated_at: String,
}

/// 指标类型枚举
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    CpuUsage,
    MemoryUsage,
    GpuUsage,
    GpuMemory,
    GpuTemperature,
    RequestsPerSecond,
    TokensPerSecond,
    LatencyAvg,
    LatencyP95,
    QueueLength,
}

impl MetricType {
    /// 获取指标类型的显示名称
    pub fn display_name(&self) -> &'static str {
        match self {
            MetricType::CpuUsage => "CPU 使用率",
            MetricType::MemoryUsage => "内存使用率",
            MetricType::GpuUsage => "GPU 使用率",
            MetricType::GpuMemory => "GPU 显存使用",
            MetricType::GpuTemperature => "GPU 温度",
            MetricType::RequestsPerSecond => "每秒请求数 (QPS)",
            MetricType::TokensPerSecond => "每秒 Token 数",
            MetricType::LatencyAvg => "平均延迟",
            MetricType::LatencyP95 => "P95 延迟",
            MetricType::QueueLength => "队列长度",
        }
    }

    /// 获取指标单位
    pub fn unit(&self) -> &'static str {
        match self {
            MetricType::CpuUsage | MetricType::MemoryUsage | MetricType::GpuUsage |
            MetricType::GpuMemory | MetricType::GpuTemperature | MetricType::QueueLength => "%",
            MetricType::RequestsPerSecond => "req/s",
            MetricType::TokensPerSecond => "tok/s",
            MetricType::LatencyAvg | MetricType::LatencyP95 => "ms",
        }
    }
}

/// 时间范围参数
#[derive(Debug, Deserialize)]
pub struct MetricsTimeRange {
    #[serde(default = "default_time_range")]
    pub time_range: String,
    #[serde(default)]
    pub interval_seconds: Option<u64>,
}

fn default_time_range() -> String {
    "1h".to_string()
}

/// 历史数据点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: i64,      // Unix 时间戳（秒）
    pub value: f64,          // 指标值
    pub metric_type: String, // 指标类型标识
}

/// 历史趋势响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsHistoryResponse {
    pub metric_type: String,
    pub points: Vec<MetricPoint>,
    pub time_range: String,
    pub interval_seconds: u64,
    pub summary: MetricSummary,
}

/// 指标摘要统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    pub min: f64,
    pub max: f64,
    pub avg: f64,
    pub current: f64,
    pub data_points_count: usize,
}

// ==================== 辅助函数 ====================

/// 获取当前 Unix 时间戳（秒）
fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// 解析时间范围为秒数
fn parse_time_range_to_seconds(range: &str) -> u64 {
    match range {
        "5m" => 300,
        "15m" => 900,
        "1h" => 3600,
        "6h" => 21600,
        "24h" => 86400,
        "7d" => 604800,
        "30d" => 2592000,
        _ => 3600, // 默认 1 小时
    }
}

/// 根据时间范围计算合适的采样间隔
fn calculate_interval(time_range_secs: u64, requested_interval: Option<u64>) -> u64 {
    if let Some(interval) = requested_interval {
        return interval.max(10); // 最小 10 秒间隔
    }

    // 自动计算：根据时间范围选择合适的间隔，确保数据点数量合理（50-300 个点）
    match time_range_secs {
        t if t <= 300 => 10,       // 5 分钟内：10 秒间隔
        t if t <= 900 => 30,       // 15 分钟内：30 秒间隔
        t if t <= 3600 => 60,      // 1 小时内：1 分钟间隔
        t if t <= 21600 => 300,    // 6 小时内：5 分钟间隔
        t if t <= 86400 => 600,    // 24 小时内：10 分钟间隔
        _ => 3600,                 // 更长时间：1 小时间隔
    }
}

// ==================== API 处理函数 ====================

/// 获取系统资源指标（CPU/GPU/内存/磁盘）
///
/// 返回当前系统的资源使用情况，包括：
/// - CPU 使用率和负载
/// - 内存使用情况
/// - GPU 状态（如果有 GPU）
/// - 磁盘使用情况
pub async fn system_metrics(
    State(_state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    // 在实际实现中，这里应该调用系统 API 获取真实的硬件信息
    // 当前返回模拟数据用于演示

    let metrics = SystemMetrics {
        cpu_usage_percent: 45.2,
        cpu_cores: 8,
        load_average_1m: 2.35,
        load_average_5m: 2.12,
        load_average_15m: 1.98,

        memory_total_mb: 32768,
        memory_used_mb: 18432,
        memory_free_mb: 14336,
        memory_usage_percent: 56.3,

        gpu_metrics: Some(GpuMetrics {
            gpu_name: "NVIDIA RTX 4090".to_string(),
            gpu_utilization_percent: 78.5,
            memory_total_mb: 24576,
            memory_used_mb: 18944,
            memory_free_mb: 5632,
            temperature_celsius: 72.5,
            power_draw_watts: 350.2,
            fan_speed_percent: 65.0,
        }),

        disk_total_gb: 1000,
        disk_used_gb: 650,
        disk_usage_percent: 65.0,

        collected_at: chrono::Utc::now().to_rfc3339(),
    };

    Ok(Json(serde_json::json!(metrics)))
}

/// 获取推理性能指标（TTFT/TPOT/QPS/throughput）
///
/// 返回当前推理引擎的性能统计数据，包括：
/// - 吞吐量（QPS、TPS）
/// - 延迟分布（TTFT、TPOT、总体延迟的 P50/P95/P99）
/// - 队列状态
/// - 成功率统计
pub async fn inference_metrics(
    State(_state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    // 当前返回模拟数据
    // 实际实现中可以从上游服务的 /metrics 端点获取 Prometheus 格式的指标数据并解析

    let metrics = InferenceMetrics {
        requests_per_second: 12.5,
        tokens_per_second: 1250.8,
        output_tokens_per_second: 85.3,
        avg_ttft_ms: 245.6,
        p50_ttft_ms: 210.3,
        p95_ttft_ms: 520.7,
        p99_ttft_ms: 850.2,
        avg_tpot_ms: 35.2,
        p50_tpot_ms: 28.5,
        p95_tpot_ms: 68.9,
        avg_latency_ms: 512.3,
        p50_latency_ms: 450.1,
        p95_latency_ms: 985.6,
        p99_latency_ms: 1520.4,
        active_requests: 8,
        queued_requests: 3,
        max_concurrent_requests: 16,
        success_rate: 98.5,
        error_rate: 1.5,
        total_requests: 45120,
        calculated_at: chrono::Utc::now().to_rfc3339(),
    };

    Ok(Json(serde_json::json!(metrics)))
}

/// 获取历史趋势数据
///
/// # Query 参数
/// - `metric_type`: 指标类型（cpu_usage/memory_usage/gpu_usage/requests_per_second 等）
/// - `time_range`: 时间范围（5m/15m/1h/6h/24h/7d/30d）
/// - `interval_seconds`: 自定义采样间隔（可选）
///
/// 返回指定时间范围内的历史数据点及摘要统计
pub async fn metrics_history(
    State(_state): State<AppState>,
    Query(params): Query<MetricsTimeRange>,
) -> Result<Json<Value>, AppError> {
    // 解析指标类型（从路径或查询参数获取，这里简化处理）
    let time_range_secs = parse_time_range_to_seconds(&params.time_range);
    let interval = calculate_interval(time_range_secs, params.interval_seconds);

    // 计算数据点数量
    let num_points = (time_range_secs / interval).min(500).max(10) as usize;

    // 生成模拟的历史数据
    // 实际实现中应该从时序数据库或缓存中读取真实数据
    let now = current_timestamp();
    let mut points = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let timestamp = now - ((num_points - i) as i64 * interval as i64);

        // 生成带有一定随机波动的模拟值
        let base_value = match params.time_range.as_str() {
            "cpu_usage" => 45.0,
            "memory_usage" => 56.0,
            "gpu_usage" => 78.0,
            "gpu_memory" => 77.0,
            "gpu_temperature" => 72.0,
            "requests_per_second" => 12.0,
            "tokens_per_second" => 1200.0,
            "latency_avg" => 500.0,
            "latency_p95" => 980.0,
            _ => 50.0,
        };

        // 添加基于时间的波动和噪声
        let wave = ((i as f64 * 0.1).sin() * 10.0); // 正弦波动
        let noise = (rand::random::<f64>() - 0.5) * 5.0; // 随机噪声
        let value = (base_value + wave + noise).max(0.0);

        points.push(MetricPoint {
            timestamp,
            value,
            metric_type: params.time_range.clone(),
        });
    }

    // 计算摘要统计
    let values: Vec<f64> = points.iter().map(|p| p.value).collect();
    let summary = MetricSummary {
        min: values.iter().cloned().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        avg: values.iter().sum::<f64>() / values.len() as f64,
        current: points.last().map(|p| p.value).unwrap_or(0.0),
        data_points_count: points.len(),
    };

    let response = MetricsHistoryResponse {
        metric_type: params.time_range.clone(),
        points,
        time_range: params.time_range,
        interval_seconds: interval,
        summary,
    };

    Ok(Json(serde_json::json!(response)))
}

/// 获取实时监控概览（聚合多个指标）
///
/// 用于 Dashboard 首页展示关键指标的快照视图。
/// 一次性返回最重要的核心指标，减少前端请求次数。
pub async fn dashboard_overview(State(state): State<AppState>) -> Result<Json<Value>, AppError> {
    // 并行获取多个指标以提高响应速度
    let (system_result, inference_result, session_stats) =
        tokio::try_join!(system_metrics(State(state.clone())), inference_metrics(State(state.clone())), async {
            // 快速获取会话统计（最近 1 小时）
            let active: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM sessions WHERE status = 'active'")
                .fetch_one(&*state.pool)
                .await
                .unwrap_or((0,));

            let total_today: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM sessions WHERE created_at >= datetime('now', '-1 day')"
            )
            .fetch_one(&*state.pool)
            .await
            .unwrap_or((0,));

            Ok::<_, AppError>(serde_json::json!({
                "active_sessions": active.0,
                "sessions_today": total_today.0
            }))
        })?;

    let system_data = system_result.0;
    let inference_data = inference_result.0;

    Ok(Json(serde_json::json!({
        "system": system_data,
        "inference": inference_data,
        "sessions": session_stats,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "status": "healthy"
    })))
}

/// 获取告警阈值配置
///
/// 返回当前配置的各项指标告警阈值，
/// 用于前端展示和比较是否触发告警。
pub async fn alert_thresholds(
    State(_state): State<AppState>,
) -> Result<Json<Value>, AppError> {
    let thresholds = serde_json::json!({
        "cpu_usage": {
            "warning": 80.0,
            "critical": 95.0,
            "unit": "%"
        },
        "memory_usage": {
            "warning": 80.0,
            "critical": 95.0,
            "unit": "%"
        },
        "gpu_usage": {
            "warning": 85.0,
            "critical": 98.0,
            "unit": "%"
        },
        "gpu_memory": {
            "warning": 85.0,
            "critical": 98.0,
            "unit": "%"
        },
        "gpu_temperature": {
            "warning": 80.0,
            "critical": 90.0,
            "unit": "°C"
        },
        "disk_usage": {
            "warning": 80.0,
            "critical": 95.0,
            "unit": "%"
        },
        "latency_p95": {
            "warning": 1000.0,
            "critical": 2000.0,
            "unit": "ms"
        },
        "error_rate": {
            "warning": 5.0,
            "critical": 10.0,
            "unit": "%"
        }
    });

    Ok(Json(thresholds))
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== 数据结构测试 ====================

    #[test]
    fn test_system_metrics_serialization() {
        let metrics = SystemMetrics {
            cpu_usage_percent: 45.2,
            cpu_cores: 8,
            load_average_1m: 2.35,
            load_average_5m: 2.12,
            load_average_15m: 1.98,
            memory_total_mb: 32768,
            memory_used_mb: 18432,
            memory_free_mb: 14336,
            memory_usage_percent: 56.3,
            gpu_metrics: None,
            disk_total_gb: 1000,
            disk_used_gb: 650,
            disk_usage_percent: 65.0,
            collected_at: "2024-06-15T12:00:00Z".to_string(),
        };

        let json = serde_json::to_value(&metrics).unwrap();
        assert_eq!(json["cpu_usage_percent"], 45.2);
        assert_eq!(json["cpu_cores"], 8);
        assert_eq!(json["memory_total_mb"], 32768);
        assert_eq!(json["disk_usage_percent"], 65.0);
        assert!(json.get("gpu_metrics").is_some()); // Option 字段始终存在
    }

    #[test]
    fn test_gpu_metrics_with_values() {
        let gpu = GpuMetrics {
            gpu_name: "NVIDIA RTX 4090".to_string(),
            gpu_utilization_percent: 78.5,
            memory_total_mb: 24576,
            memory_used_mb: 18944,
            memory_free_mb: 5632,
            temperature_celsius: 72.5,
            power_draw_watts: 350.2,
            fan_speed_percent: 65.0,
        };

        let json = serde_json::to_value(&gpu).unwrap();
        assert_eq!(json["gpu_name"], "NVIDIA RTX 4090");
        assert_eq!(json["gpu_utilization_percent"], 78.5);
        assert_eq!(json["temperature_celsius"], 72.5);
        assert_eq!(json["power_draw_watts"], 350.2);
    }

    #[test]
    fn test_inference_metrics_structure() {
        let metrics = InferenceMetrics {
            requests_per_second: 12.5,
            tokens_per_second: 1250.8,
            output_tokens_per_second: 85.3,
            avg_ttft_ms: 245.6,
            p50_ttft_ms: 210.3,
            p95_ttft_ms: 520.7,
            p99_ttft_ms: 850.2,
            avg_tpot_ms: 35.2,
            p50_tpot_ms: 28.5,
            p95_tpot_ms: 68.9,
            avg_latency_ms: 512.3,
            p50_latency_ms: 450.1,
            p95_latency_ms: 985.6,
            p99_latency_ms: 1520.4,
            active_requests: 8,
            queued_requests: 3,
            max_concurrent_requests: 16,
            success_rate: 98.5,
            error_rate: 1.5,
            total_requests: 45120,
            calculated_at: "2024-06-15T12:00:00Z".to_string(),
        };

        let json = serde_json::to_value(&metrics).unwrap();
        assert!((json["requests_per_second"].as_f64().unwrap() - 12.5).abs() < 0.001);
        assert!((json["success_rate"].as_f64().unwrap() - 98.5).abs() < 0.001);
        assert_eq!(json["active_requests"], 8);
        assert_eq!(json["queued_requests"], 3);
    }

    #[test]
    fn test_metric_point_structure() {
        let point = MetricPoint {
            timestamp: 1718400000,
            value: 45.2,
            metric_type: "cpu_usage".to_string(),
        };

        let json = serde_json::to_value(&point).unwrap();
        assert_eq!(json["timestamp"], 1718400000);
        assert!((json["value"].as_f64().unwrap() - 45.2).abs() < 0.001);
        assert_eq!(json["metric_type"], "cpu_usage");
    }

    // ==================== 辅助函数测试 ====================

    #[test]
    fn test_parse_time_range() {
        assert_eq!(parse_time_range_to_seconds("5m"), 300);
        assert_eq!(parse_time_range_to_seconds("15m"), 900);
        assert_eq!(parse_time_range_to_seconds("1h"), 3600);
        assert_eq!(parse_time_range_to_seconds("6h"), 21600);
        assert_eq!(parse_time_range_to_seconds("24h"), 86400);
        assert_eq!(parse_time_range_to_seconds("7d"), 604800);
        assert_eq!(parse_time_range_to_seconds("30d"), 2592000);
        assert_eq!(parse_time_range_to_seconds("unknown"), 3600); // 默认值
    }

    #[test]
    fn test_calculate_interval_auto() {
        // 测试自动计算的采样间隔
        assert_eq!(calculate_interval(300, None), 10);   // 5分钟
        assert_eq!(calculate_interval(900, None), 30);   // 15分钟
        assert_eq!(calculate_interval(3600, None), 60);  // 1小时
        assert_eq!(calculate_interval(21600, None), 300); // 6小时
        assert_eq!(calculate_interval(86400, None), 600); // 24小时
    }

    #[test]
    fn test_calculate_interval_custom() {
        // 测试自定义间隔
        assert_eq!(calculate_interval(3600, Some(120)), 120);
        assert_eq!(calculate_interval(3600, Some(5)), 10); // 最小 10 秒
    }

    #[test]
    fn test_metric_type_display_and_unit() {
        assert_eq!(MetricType::CpuUsage.display_name(), "CPU 使用率");
        assert_eq!(MetricType::CpuUsage.unit(), "%");

        assert_eq!(MetricType::RequestsPerSecond.display_name(), "每秒请求数 (QPS)");
        assert_eq!(MetricType::RequestsPerSecond.unit(), "req/s");

        assert_eq!(MetricType::LatencyP95.display_name(), "P95 延迟");
        assert_eq!(MetricType::LatencyP95.unit(), "ms");
    }

    #[test]
    fn test_metric_summary_calculation() {
        let points = vec![
            MetricPoint { timestamp: 1000, value: 10.0, metric_type: "test".into() },
            MetricPoint { timestamp: 2000, value: 20.0, metric_type: "test".into() },
            MetricPoint { timestamp: 3000, value: 30.0, metric_type: "test".into() },
        ];

        let values: Vec<f64> = points.iter().map(|p| p.value).collect();
        let summary = MetricSummary {
            min: values.iter().cloned().fold(f64::INFINITY, f64::min),
            max: values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            avg: values.iter().sum::<f64>() / values.len() as f64,
            current: points.last().map(|p| p.value).unwrap_or(0.0),
            data_points_count: points.len(),
        };

        assert!((summary.min - 10.0).abs() < 0.001);
        assert!((summary.max - 30.0).abs() < 0.001);
        assert!((summary.avg - 20.0).abs() < 0.001);
        assert!((summary.current - 30.0).abs() < 0.001);
        assert_eq!(summary.data_points_count, 3);
    }
}
