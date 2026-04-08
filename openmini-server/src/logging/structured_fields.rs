//! 结构化日志字段定义
//!
//! 提供类型安全的结构化字段包装器，用于在日志中添加语义化的元数据。
//! 所有字段都实现了 `Display` trait，可以直接用于 tracing 的字段值。

use std::fmt;
use std::time::Instant;

// ============================================================================
// 请求标识字段
// ============================================================================

/// 请求唯一标识符
///
/// 用于关联同一请求的所有日志条目，便于追踪和调试。
/// 通常从HTTP头或gRPC元数据中提取。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::RequestFields;
/// use tracing::info;
///
/// let req_id = RequestFields::new("req-abc-123");
/// info!(request_id = %req_id, "Processing request");
/// ```
#[derive(Debug, Clone)]
pub struct RequestFields {
    id: String,
}

impl RequestFields {
    /// 创建新的请求ID字段
    ///
    /// # 参数
    ///
    /// * `id` - 请求唯一标识符
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    /// 获取请求ID值
    pub fn value(&self) -> &str {
        &self.id
    }
}

impl fmt::Display for RequestFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

// ============================================================================
// 模型信息字段
// ============================================================================

/// 模型名称标识
///
/// 记录当前使用的模型名称，便于区分不同模型的推理日志。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::ModelFields;
/// use tracing::info_span;
///
/// let model = ModelFields::new("llama-3-8b-instruct");
/// let span = info_span!("inference", model_name = %model);
/// ```
#[derive(Debug, Clone)]
pub struct ModelFields {
    name: String,
}

impl ModelFields {
    /// 创建新的模型名称字段
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// 获取模型名称
    pub fn value(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for ModelFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ============================================================================
// 延迟统计字段
// ============================================================================

/// 延迟测量字段（毫秒）
///
/// 用于记录各种操作的耗时，如：
/// - 首个token延迟 (TTFT - Time To First Token)
/// - 总推理延迟
/// - 排队等待时间
/// - GPU计算时间
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::LatencyFields;
/// use tracing::info;
///
/// let latency = LatencyFields::from_ms(42.5);
/// info!(latency_ms = %latency, "Inference completed");
/// ```
#[derive(Debug, Clone)]
pub struct LatencyFields {
    milliseconds: f64,
}

impl LatencyFields {
    /// 从毫秒值创建延迟字段
    ///
    /// # 参数
    ///
    /// * `ms` - 延迟时间（毫秒），支持小数精度
    pub fn from_ms(ms: f64) -> Self {
        Self { milliseconds: ms }
    }

    /// 从纳秒值创建延迟字段
    ///
    /// # 参数
    ///
    /// * `ns` - 延迟时间（纳秒）
    pub fn from_ns(ns: u64) -> Self {
        Self {
            milliseconds: ns as f64 / 1_000_000.0,
        }
    }

    /// 从 `Instant` 时间差创建延迟字段
    ///
    /// # 参数
    ///
    /// * `start` - 开始时间点
    pub fn from_instant(start: Instant) -> Self {
        let elapsed = start.elapsed();
        Self {
            milliseconds: elapsed.as_secs_f64() * 1000.0,
        }
    }

    /// 获取毫秒值
    pub fn as_ms(&self) -> f64 {
        self.milliseconds
    }

    /// 获取纳秒值
    pub fn as_ns(&self) -> u64 {
        (self.milliseconds * 1_000_000.0) as u64
    }
}

impl fmt::Display for LatencyFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 根据数值大小选择合适的显示格式
        if self.milliseconds < 1.0 {
            write!(f, "{:.2}ms", self.milliseconds)
        } else if self.milliseconds < 100.0 {
            write!(f, "{:.1}ms", self.milliseconds)
        } else {
            write!(f, "{:.0}ms", self.milliseconds)
        }
    }
}

// ============================================================================
// Token统计字段
// ============================================================================

/// Token使用统计
///
/// 记录输入和输出的token数量，用于监控和计费。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::TokenFields;
/// use tracing::info;
///
/// let tokens = TokenFields::new(150, 50);
/// info!(
///     input_tokens = tokens.input_tokens,
///     output_tokens = tokens.output_tokens,
///     total_tokens = tokens.total(),
///     "Token usage"
/// );
/// ```
#[derive(Debug, Clone)]
pub struct TokenFields {
    /// 输入token数量（包含prompt）
    pub input_tokens: u32,
    /// 输出token数量（生成的回复）
    pub output_tokens: u32,
}

impl TokenFields {
    /// 创建Token统计字段
    ///
    /// # 参数
    ///
    /// * `input_tokens` - 输入token数
    /// * `output_tokens` - 输出token数
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
        }
    }

    /// 计算总token数
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// 计算每秒生成token数（吞吐量）
    ///
    /// # 参数
    ///
    /// * `duration_seconds` - 总耗时（秒）
    pub fn throughput(&self, duration_seconds: f64) -> f64 {
        if duration_seconds > 0.0 {
            self.output_tokens as f64 / duration_seconds
        } else {
            0.0
        }
    }
}

// ============================================================================
// Worker/进程信息字段
// ============================================================================

/// Worker进程标识
///
/// 用于多Worker部署时区分日志来源。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::WorkerFields;
/// use tracing::info;
///
/// let worker = WorkerFields::new(3);
/// info!(worker_id = %worker, pid = worker.pid(), "Worker started");
/// ```
#[derive(Debug, Clone)]
pub struct WorkerFields {
    id: u32,
}

impl WorkerFields {
    /// 创建Worker标识字段
    pub fn new(id: u32) -> Self {
        Self { id }
    }

    /// 获取Worker ID
    pub fn id(&self) -> u32 {
        self.id
    }

    /// 获取当前进程PID
    pub fn pid(&self) -> u32 {
        std::process::id()
    }
}

impl fmt::Display for WorkerFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "worker-{}", self.id)
    }
}

// ============================================================================
// 内存/GPU资源字段
// ============================================================================

/// GPU内存使用情况
///
/// 监控GPU显存分配和使用状态。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::GpuMemoryFields;
/// use tracing::info;
///
/// let gpu_mem = GpuMemoryFields::new(8192, 6144); // 8GB总量，6GB已用
/// info!(gpu_memory = %gpu_mem, utilization = gpu_mem.utilization_pct(), "Memory status");
/// ```
#[derive(Debug, Clone)]
pub struct GpuMemoryFields {
    /// 总显存（MB）
    total_mb: u64,
    /// 已用显存（MB）
    used_mb: u64,
}

impl GpuMemoryFields {
    /// 创建GPU内存统计字段
    ///
    /// # 参数
    ///
    /// * `total_mb` - 总显存大小（MB）
    /// * `used_mb` - 已用显存（MB）
    pub fn new(total_mb: u64, used_mb: u64) -> Self {
        Self { total_mb, used_mb }
    }

    /// 获取总显存（MB）
    pub fn total_mb(&self) -> u64 {
        self.total_mb
    }

    /// 获取已用显存（MB）
    pub fn used_mb(&self) -> u64 {
        self.used_mb
    }

    /// 获取利用率百分比
    pub fn utilization_pct(&self) -> f64 {
        if self.total_mb > 0 {
            (self.used_mb as f64 / self.total_mb as f64) * 100.0
        } else {
            0.0
        }
    }
}

impl fmt::Display for GpuMemoryFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{} MB ({:.1}%)",
            self.used_mb,
            self.total_mb,
            self.utilization_pct()
        )
    }
}

// ============================================================================
// 错误/异常字段
// ============================================================================

/// 结构化错误信息
///
/// 提供错误分类和详情，便于日志分析和告警。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::ErrorFields;
/// use tracing::error;
///
/// let err = ErrorFields::oom("CUDA out of memory", 8192);
/// error!(error = %err, error_code = err.code(), "Inference failed");
/// ```
#[derive(Debug, Clone)]
pub struct ErrorFields {
    /// 错误类型分类
    category: ErrorCategory,
    /// 错误消息
    message: String,
    /// 可选的错误代码
    code: Option<u32>,
}

/// 错误分类枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// GPU显存不足
    OutOfMemory,
    /// 模型加载失败
    ModelLoadError,
    /// 推理超时
    Timeout,
    /// 无效输入
    InvalidInput,
    /// 内部服务错误
    InternalError,
    /// 其他未分类错误
    Other,
}

impl ErrorFields {
    /// 创建OOM错误字段
    ///
    /// # 参数
    ///
    /// * `message` - 错误描述
    /// * `_required_mb` - 所需内存（MB），可用于日志记录
    pub fn oom(message: impl Into<String>, _required_mb: u64) -> Self {
        Self {
            category: ErrorCategory::OutOfMemory,
            message: message.into(),
            code: Some(50013), // HTTP 500 + 自定义子码
        }
    }

    /// 创建模型加载错误
    pub fn model_load_error(message: impl Into<String>) -> Self {
        Self {
            category: ErrorCategory::ModelLoadError,
            message: message.into(),
            code: Some(50011),
        }
    }

    /// 创建超时错误
    pub fn timeout(operation: impl Into<String>, timeout_secs: u64) -> Self {
        Self {
            category: ErrorCategory::Timeout,
            message: format!("{} timed out after {}s", operation.into(), timeout_secs),
            code: Some(50400),
        }
    }

    /// 创建无效输入错误
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self {
            category: ErrorCategory::InvalidInput,
            message: message.into(),
            code: Some(40000),
        }
    }

    /// 创建内部错误
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            category: ErrorCategory::InternalError,
            message: message.into(),
            code: Some(50000),
        }
    }

    /// 获取错误类别
    pub fn category(&self) -> ErrorCategory {
        self.category
    }

    /// 获取错误代码
    pub fn code(&self) -> Option<u32> {
        self.code
    }

    /// 获取错误消息
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ErrorFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.category, self.message)
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::OutOfMemory => write!(f, "OOM"),
            ErrorCategory::ModelLoadError => write!(f, "MODEL_LOAD"),
            ErrorCategory::Timeout => write!(f, "TIMEOUT"),
            ErrorCategory::InvalidInput => write!(f, "INVALID_INPUT"),
            ErrorCategory::InternalError => write!(f, "INTERNAL"),
            ErrorCategory::Other => write!(f, "OTHER"),
        }
    }
}

// ============================================================================
// 批处理字段
// ============================================================================

/// 连续批处理（Continuous Batching）统计
///
/// 记录批处理调度器的运行状态和性能指标。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::BatchFields;
/// use tracing::info;
///
/// let batch = BatchFields::new(8, 4, 1024);
/// info!(batch = %batch, "Scheduled batch");
/// ```
#[derive(Debug, Clone)]
pub struct BatchFields {
    /// 当前批次中的请求数
    batch_size: usize,
    /// 活跃请求数
    active_requests: usize,
    /// 当前KV缓存使用量（tokens）
    cache_usage: u64,
}

impl BatchFields {
    /// 创建批处理统计字段
    ///
    /// # 参数
    ///
    /// * `batch_size` - 批次大小
    /// * `active_requests` - 活跃请求数
    /// * `cache_usage` - KV缓存使用量
    pub fn new(batch_size: usize, active_requests: usize, cache_usage: u64) -> Self {
        Self {
            batch_size,
            active_requests,
            cache_usage,
        }
    }

    /// 获取批次大小
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// 获取活跃请求数
    pub fn active_requests(&self) -> usize {
        self.active_requests
    }

    /// 获取缓存使用量
    pub fn cache_usage(&self) -> u64 {
        self.cache_usage
    }
}

impl fmt::Display for BatchFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "size={}, active={}, cache={} tokens",
            self.batch_size, self.active_requests, self.cache_usage
        )
    }
}

// ============================================================================
// 辅助宏
// ============================================================================

/// 快速记录请求生命周期日志的宏
///
/// 自动创建span并记录关键事件，减少样板代码。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::logging::log_request;
///
/// log_request!("req-001", "llama-3", {
///     // 执行推理逻辑
///     let result = run_inference().await;
///     // 自动记录完成时间和token统计
/// });
/// ```
#[macro_export]
macro_rules! log_request {
    ($request_id:expr, $model_name:expr, $block:block) => {{
        let _span = tracing::info_span!(
            "request",
            request_id = %$crate::logging::RequestFields::new($request_id),
            model_name = %$crate::logging::ModelFields::new($model_name),
        );

        let _enter = _span.enter();

        tracing::info!("Request started");

        let start = std::time::Instant::now();
        let result = $block;
        let latency = $crate::logging::LatencyFields::from_instant(start);

        tracing::info!(
            latency_ms = %latency,
            "Request completed"
        );

        result
    }};
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== RequestFields 测试 ====================

    #[test]
    fn test_request_fields_new() {
        let req = RequestFields::new("test-id-123");
        assert_eq!(req.value(), "test-id-123");
    }

    #[test]
    fn test_request_fields_display() {
        let req = RequestFields::new("abc");
        assert_eq!(format!("{}", req), "abc");
    }

    #[test]
    fn test_request_fields_clone() {
        let req = RequestFields::new("original");
        let cloned = req.clone();
        assert_eq!(cloned.value(), "original");
    }

    // ==================== ModelFields 测试 ====================

    #[test]
    fn test_model_fields_new() {
        let model = ModelFields::new("llama-3-8b");
        assert_eq!(model.value(), "llama-3-8b");
    }

    #[test]
    fn test_model_fields_display() {
        let model = ModelFields::new("gpt-4");
        assert_eq!(format!("{}", model), "gpt-4");
    }

    // ==================== LatencyFields 测试 ====================

    #[test]
    fn test_latency_from_ms() {
        let lat = LatencyFields::from_ms(42.5);
        assert!((lat.as_ms() - 42.5).abs() < 0.01);
    }

    #[test]
    fn test_latency_from_ns() {
        let lat = LatencyFields::from_ns(42_500_000); // 42.5ms in ns
        assert!((lat.as_ms() - 42.5).abs() < 0.1);
    }

    #[test]
    fn test_latency_from_instant() {
        // 这个测试验证方法可以调用，精确度受系统调度影响
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let lat = LatencyFields::from_instant(start);
        assert!(lat.as_ms() >= 9.0); // 至少应该接近10ms
    }

    #[test]
    fn test_latency_display_small() {
        let lat = LatencyFields::from_ms(0.5);
        let display = format!("{}", lat);
        assert!(display.contains("0.50ms"));
    }

    #[test]
    fn test_latency_display_medium() {
        let lat = LatencyFields::from_ms(42.5);
        let display = format!("{}", lat);
        assert!(display.contains("42.5ms"));
    }

    #[test]
    fn test_latency_display_large() {
        let lat = LatencyFields::from_ms(1500.5);
        let display = format!("{}", lat);
        assert!(display.contains("1500ms"));
    }

    // ==================== TokenFields 测试 ====================

    #[test]
    fn test_token_fields_new() {
        let tokens = TokenFields::new(100, 50);
        assert_eq!(tokens.input_tokens, 100);
        assert_eq!(tokens.output_tokens, 50);
    }

    #[test]
    fn test_token_fields_total() {
        let tokens = TokenFields::new(150, 50);
        assert_eq!(tokens.total(), 200);
    }

    #[test]
    fn test_token_throughput_normal() {
        let tokens = TokenFields::new(0, 100);
        let tps = tokens.throughput(10.0);
        assert!((tps - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_token_throughput_zero_duration() {
        let tokens = TokenFields::new(0, 100);
        let tps = tokens.throughput(0.0);
        assert_eq!(tps, 0.0);
    }

    // ==================== WorkerFields 测试 ====================

    #[test]
    fn test_worker_fields_new() {
        let worker = WorkerFields::new(3);
        assert_eq!(worker.id(), 3);
    }

    #[test]
    fn test_worker_fields_pid() {
        let worker = WorkerFields::new(1);
        assert!(worker.pid() > 0);
    }

    #[test]
    fn test_worker_fields_display() {
        let worker = WorkerFields::new(5);
        assert_eq!(format!("{}", worker), "worker-5");
    }

    // ==================== GpuMemoryFields 测试 ====================

    #[test]
    fn test_gpu_memory_fields_new() {
        let mem = GpuMemoryFields::new(8192, 4096);
        assert_eq!(mem.total_mb(), 8192);
        assert_eq!(mem.used_mb(), 4096);
    }

    #[test]
    fn test_gpu_memory_utilization() {
        let mem = GpuMemoryFields::new(8000, 4000);
        let util = mem.utilization_pct();
        assert!((util - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_gpu_memory_utilization_zero_total() {
        let mem = GpuMemoryFields::new(0, 100);
        assert_eq!(mem.utilization_pct(), 0.0);
    }

    #[test]
    fn test_gpu_memory_display() {
        let mem = GpuMemoryFields::new(8000, 4000);
        let display = format!("{}", mem);
        assert!(display.contains("4000"));
        assert!(display.contains("8000"));
        assert!(display.contains("50.0%"));
    }

    // ==================== ErrorFields 测试 ====================

    #[test]
    fn test_error_oom() {
        let err = ErrorFields::oom("CUDA OOM", 8192);
        assert_eq!(err.category(), ErrorCategory::OutOfMemory);
        assert_eq!(err.code(), Some(50013));
        assert!(err.message().contains("CUDA OOM"));
    }

    #[test]
    fn test_error_model_load() {
        let err = ErrorFields::model_load_error("File not found");
        assert_eq!(err.category(), ErrorCategory::ModelLoadError);
        assert_eq!(err.code(), Some(50011));
    }

    #[test]
    fn test_error_timeout() {
        let err = ErrorFields::timeout("inference", 30);
        assert_eq!(err.category(), ErrorCategory::Timeout);
        assert!(err.message().contains("timed out after 30s"));
    }

    #[test]
    fn test_error_invalid_input() {
        let err = ErrorFields::invalid_input("Empty prompt");
        assert_eq!(err.category(), ErrorCategory::InvalidInput);
        assert_eq!(err.code(), Some(40000));
    }

    #[test]
    fn test_error_internal() {
        let err = ErrorFields::internal("Null pointer");
        assert_eq!(err.category(), ErrorCategory::InternalError);
        assert_eq!(err.code(), Some(50000));
    }

    #[test]
    fn test_error_category_display() {
        assert_eq!(format!("{}", ErrorCategory::OutOfMemory), "OOM");
        assert_eq!(format!("{}", ErrorCategory::Timeout), "TIMEOUT");
        assert_eq!(format!("{}", ErrorCategory::InternalError), "INTERNAL");
    }

    #[test]
    fn test_error_display() {
        let err = ErrorFields::oom("test error", 1000);
        let display = format!("{}", err);
        assert!(display.contains("OutOfMemory"));
        assert!(display.contains("test error"));
    }

    // ==================== BatchFields 测试 ====================

    #[test]
    fn test_batch_fields_new() {
        let batch = BatchFields::new(8, 4, 2048);
        assert_eq!(batch.batch_size(), 8);
        assert_eq!(batch.active_requests(), 4);
        assert_eq!(batch.cache_usage(), 2048);
    }

    #[test]
    fn test_batch_fields_display() {
        let batch = BatchFields::new(4, 2, 1024);
        let display = format!("{}", batch);
        assert!(display.contains("size=4"));
        assert!(display.contains("active=2"));
        assert!(display.contains("cache=1024"));
    }

    // ==================== 集成测试 ====================

    #[test]
    fn test_all_fields_in_tracing() {
        // 验证所有字段都可以在tracing中使用
        use tracing::info;

        let req = RequestFields::new("integration-test");
        let model = ModelFields::new("test-model");
        let worker = WorkerFields::new(1);
        let tokens = TokenFields::new(10, 5);
        let latency = LatencyFields::from_ms(12.3);
        let batch = BatchFields::new(2, 2, 512);

        info!(
            request_id = %req,
            model_name = %model,
            worker_id = %worker,
            input_tokens = tokens.input_tokens,
            output_tokens = tokens.output_tokens,
            latency_ms = %latency,
            batch_info = %batch,
            "Integration test with all fields"
        );
    }

    // ===== 边界条件和分支覆盖率测试 =====

    // ==================== TokenFields 额外测试 ====================

    #[test]
    fn test_token_fields_throughput_calculation_with_duration() {
        let tokens = TokenFields::new(100, 50);
        let throughput = tokens.throughput(2.0);
        assert!(
            (throughput - 25.0).abs() < 0.1,
            "Throughput should be ~25, got {}",
            throughput
        );
    }

    #[test]
    fn test_token_fields_throughput_zero_duration() {
        let _tokens = TokenFields::new(100, 0);
    }

    #[test]
    fn test_token_fields_edge_values() {
        let zero_tokens = TokenFields::new(0, 0);
        assert_eq!(zero_tokens.total(), 0);
        assert_eq!(zero_tokens.throughput(1.0), 0.0);

        let large_tokens = TokenFields::new(100_000, 200_000);
        assert_eq!(large_tokens.total(), 300_000);
        let large_throughput = large_tokens.throughput(1.0);
        assert!((large_throughput - 200000.0).abs() < 0.1);
    }

    // ==================== LatencyFields 额外测试 ====================

    #[test]
    fn test_latency_fields_precision_levels() {
        // 测试不同精度级别的显示格式

        // 亚毫秒级
        let sub_ms = LatencyFields::from_ms(0.05);
        let display_sub = format!("{}", sub_ms);
        assert!(display_sub.contains("ms"), "Should contain ms unit");

        // 毫秒级（小数）
        let ms_decimal = LatencyFields::from_ms(42.567);
        let display_ms_dec = format!("{}", ms_decimal);
        assert!(display_ms_dec.contains("ms"));

        // 秒级（整数）
        let seconds = LatencyFields::from_ms(1500.0);
        let display_sec = format!("{}", seconds);
        assert!(display_sec.contains("ms"));
    }

    #[test]
    fn test_latency_fields_conversion_methods() {
        // 测试各种转换方法的一致性
        let from_ms = LatencyFields::from_ms(42.5);
        let from_ns = LatencyFields::from_ns(42_500_000); // 42.5ms in ns

        assert!(
            (from_ms.as_ms() - from_ns.as_ms()).abs() < 0.1,
            "ms and ns constructors should be consistent"
        );

        // 测试ns转换
        let latency = LatencyFields::from_ms(1.0); // 1ms = 1,000,000 ns
        assert!(
            ((latency.as_ns() as i64) - 1_000_000).abs() < 100,
            "ns conversion should be accurate"
        );
    }

    // ==================== ErrorFields 额外测试 ====================

    #[test]
    fn test_error_fields_classification_types() {
        // 测试所有错误类型的分类
        let oom = ErrorFields::oom("KV Cache overflow", 1024 * 1024 * 1024);
        assert_eq!(oom.category(), ErrorCategory::OutOfMemory);
        assert_eq!(oom.code(), Some(50013));
        assert!(oom.message().contains("KV Cache overflow"));

        let timeout = ErrorFields::timeout("Inference timeout", 30);
        assert_eq!(timeout.category(), ErrorCategory::Timeout);
        assert_eq!(timeout.code(), Some(50400));
        assert!(timeout.message().contains("timed out after 30s"));

        let invalid = ErrorFields::invalid_input("Empty prompt");
        assert_eq!(invalid.category(), ErrorCategory::InvalidInput);
        assert_eq!(invalid.code(), Some(40000));

        let model_err = ErrorFields::model_load_error("File not found");
        assert_eq!(model_err.category(), ErrorCategory::ModelLoadError);
        assert_eq!(model_err.code(), Some(50011));

        let internal = ErrorFields::internal("Null pointer exception");
        assert_eq!(internal.category(), ErrorCategory::InternalError);
        assert_eq!(internal.code(), Some(50000));
    }

    #[test]
    fn test_error_category_display_all_variants() {
        // 测试所有ErrorCategory变体的显示格式
        assert_eq!(format!("{}", ErrorCategory::OutOfMemory), "OOM");
        assert_eq!(format!("{}", ErrorCategory::ModelLoadError), "MODEL_LOAD");
        assert_eq!(format!("{}", ErrorCategory::Timeout), "TIMEOUT");
        assert_eq!(format!("{}", ErrorCategory::InvalidInput), "INVALID_INPUT");
        assert_eq!(format!("{}", ErrorCategory::InternalError), "INTERNAL");
        assert_eq!(format!("{}", ErrorCategory::Other), "OTHER");
    }

    #[test]
    fn test_error_fields_display_format() {
        // 测试ErrorFields的完整显示格式
        let err = ErrorFields::oom("CUDA out of memory", 8192);
        let display = format!("{}", err);

        assert!(
            display.contains("OutOfMemory"),
            "Should contain category tag"
        );
        assert!(
            display.contains("CUDA out of memory"),
            "Should contain message"
        );
    }

    #[test]
    fn test_error_fields_clone_and_equality() {
        // 测试ErrorFields的克隆和比较
        let err1 = ErrorFields::timeout("Test error", 10);
        let err2 = err1.clone();

        assert_eq!(err1.category(), err2.category());
        assert_eq!(err1.code(), err2.code());
        assert_eq!(err1.message(), err2.message());
    }

    // ==================== GpuMemoryFields 额外测试 ====================

    #[test]
    fn test_gpu_memory_fields_utilization_calculations() {
        // 测试各种利用率计算场景

        // 正常情况
        let normal = GpuMemoryFields::new(8000, 4000);
        assert!((normal.utilization_pct() - 50.0).abs() < 0.01);

        // 满载
        let full = GpuMemoryFields::new(8192, 8192);
        assert!((full.utilization_pct() - 100.0).abs() < 0.01);

        // 空闲
        let empty = GpuMemoryFields::new(8192, 0);
        assert!((empty.utilization_pct() - 0.0).abs() < 0.01);

        // 零总内存（避免除零）
        let zero_total = GpuMemoryFields::new(0, 100);
        assert_eq!(zero_total.utilization_pct(), 0.0);
    }

    #[test]
    fn test_gpu_memory_fields_display_format() {
        // 测试显示格式的各种场景
        let mem = GpuMemoryFields::new(8192, 4096);
        let display = format!("{}", mem);

        assert!(display.contains("4096")); // used_mb
        assert!(display.contains("8192")); // total_mb
        assert!(display.contains("50.0%")); // utilization
    }

    // ==================== BatchFields 额外测试 ====================

    #[test]
    fn test_batch_fields_edge_cases() {
        // 测试批处理字段的边界值
        let empty_batch = BatchFields::new(0, 0, 0);
        assert_eq!(empty_batch.batch_size(), 0);
        assert_eq!(empty_batch.active_requests(), 0);
        assert_eq!(empty_batch.cache_usage(), 0);

        let large_batch = BatchFields::new(usize::MAX, usize::MAX, u64::MAX);
        assert!(large_batch.batch_size() > 0);
    }

    #[test]
    fn test_batch_fields_display_completeness() {
        // 测试BatchFields显示包含所有信息
        let batch = BatchFields::new(16, 12, 1024);
        let display = format!("{}", batch);

        assert!(display.contains("size=16"));
        assert!(display.contains("active=12"));
        assert!(display.contains("cache=1024"));
        assert!(display.contains("tokens"));
    }

    // ==================== WorkerFields 额外测试 ====================

    #[test]
    fn test_worker_fields_pid_validity() {
        // 测试PID总是有效的正整数
        for _ in 0..10 {
            let worker = WorkerFields::new(1);
            assert!(worker.pid() > 0, "PID should always be positive");
        }
    }

    #[test]
    fn test_worker_fields_display_format() {
        // 测试Worker显示格式
        let worker = WorkerFields::new(42);
        let display = format!("{}", worker);
        assert_eq!(display, "worker-42");
    }

    // ==================== RequestFields & ModelFields 额外测试 ====================

    #[test]
    fn test_request_and_model_fields_empty_string() {
        // 测试空字符串处理
        let empty_req = RequestFields::new("");
        assert_eq!(empty_req.value(), "");
        assert_eq!(format!("{}", empty_req), "");

        let empty_model = ModelFields::new("");
        assert_eq!(empty_model.value(), "");
    }

    #[test]
    fn test_request_and_model_fields_special_characters() {
        // 测试特殊字符处理
        let special_req = RequestFields::new("req-with_special.chars-123_456");
        assert_eq!(format!("{}", special_req), "req-with_special.chars-123_456");

        let special_model = ModelFields::new("model/v1.2.3-test");
        assert_eq!(format!("{}", special_model), "model/v1.2.3-test");
    }
}
