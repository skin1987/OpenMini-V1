//! # OpenMini 业务监控指标模块
//!
//! 定义和暴露 **6 个核心业务指标**，用于监控推理服务的运行状态和性能。
//! 所有指标遵循 Prometheus 命名规范，支持 Grafana 可视化和自动告警。
//!
//! ## 指标总览
//!
//! | 指标名称 | Prometheus 类型 | 用途 | 标签 (Labels) |
//! |---------|---------------|------|--------------|
//! | `openmini_inference_tokens_total` | **Counter** | Token 吞吐量统计 | model, status |
//! | `openmini_request_duration_seconds` | **Histogram** | 请求延迟分布 | endpoint, method |
//! | `openmini_kv_cache_usage_bytes` | **Gauge** | KV Cache 内存占用 | layer |
//! | `openmini_worker_queue_length` | **Gauge** | Worker 队列深度 | (无) |
//! | `openmini_active_connections` | **Gauge** | 活跃连接数 | (无) |
//! | `openmini_model_loaded` | **Gauge** | 模型加载状态 | model_name |
//!
//! ## Prometheus 数据类型说明
//!
//! ### Counter (计数器)
//!
//! - **单调递增**: 只能增加，不能减少（除非服务重启）
//! - **用途**: 累计统计（总请求数、总 token 数）
//! - **查询技巧**: 使用 `rate()` 计算速率
//!
//! ### Histogram (直方图)
//!
//! - **分桶统计**: 将值分配到预定义的桶中
//! - **内置分位数**: 自动计算 _bucket, _sum, _count
//! - **用途**: 延迟分布分析（P50/P95/P99）
//!
//! ### Gauge (仪表盘)
//!
//! - **可增可减**: 表示当前瞬时值
//! - **用途**: 资源使用率、队列深度、状态标记
//!
//! ## Grafana Dashboard 配置示例
//!
//! ### Panel 1: Token 吞吐量 (tokens/sec)
//!
//! ```promql
//! // 过去 5 分钟的 token 生成速率
//! sum(rate(openmini_inference_tokens_total{status="success"}[5m])) by (model)
//! ```
//!
//! ### Panel 2: 请求延迟 P99 (秒)
//!
//! ```promql
//! // 请求延迟的 99 分位数
//! histogram_quantile(0.99,
//!   sum(rate(openmini_request_duration_seconds_bucket[5m])) by (le, endpoint)
//! )
//! ```
//!
//! ### Panel 3: KV Cache 内存使用 (GB)
//!
//! ```promql
//! // 各层 KV Cache 总内存占用
//! sum(openmini_kv_cache_usage_bytes) / 1024 / 1024 / 1024
//! ```
//!
//! ### Panel 4: Worker 队列积压
//!
//! ```promql
//! // 当前等待处理的请求数
//! openmini_worker_queue_length
//! ```
//!
//! ## 推荐告警规则
//!
//! ### 🔴 P0 - 紧急告警 (立即通知)
//!
//! | 规则名称 | PromQL 表达式 | 阈值 | 通知渠道 |
//! |---------|-------------|------|---------|
//! | 高错误率 | `rate(openmini_inference_tokens_total{status="error"}[5m]) / rate(openmini_inference_tokens_total[5m]) > 0.05` | 错误率 > 5% | PagerDuty + Slack |
//! | 请求超时 | `histogram_quantile(0.99, rate(openmini_request_duration_seconds_bucket[5m])) > 10` | P99 > 10s | PagerDuty + Slack |
//! | 模型未加载 | `openmini_model_loaded != 1` | 模型离线 | PagerDuty |
//!
//! ### 🟡 P1 - 重要告警 (30分钟内处理)
//!
//! | 规则名称 | PromQL 表达式 | 阈值 | 通知渠道 |
//! |---------|-------------|------|---------|
//! | 队列积压 | `openmini_worker_queue_length > 100` | 队列 > 100 | Slack + Email |
//! | KV Cache 高占用 | `sum(openmini_kv_cache_usage_bytes) / 1024^3 > 20` | 内存 > 20GB | Slack |
//! | 连接数异常 | `openmini_active_connections > 500` | 连接 > 500 | Slack |
//!
//! ### 🟢 P2 - 一般告警 (每日巡检)
//!
//! | 规则名称 | PromQL 表达式 | 阈值 | 通知渠道 |
//! |---------|-------------|------|---------|
//! | 吞吐量下降 | `sum(rate(openmini_inference_tokens_total[1h])) < 1000` | < 1K tokens/s | Email |
//! | 平均延迟升高 | `histogram_quantile(0.50, rate(...)) > 2` | P50 > 2s | Email |
//!
//! ## 使用示例
//!
//! ```rust,ignore
//! use openmini_server::monitoring::business_metrics::*;
//! use std::time::Duration;
//!
//! // 记录一次成功的推理完成（生成 128 个 token）
//! record_inference_completion("llama-3-8b", 128, true);
//!
//! // 记录请求耗时
//! let start = std::time::Instant::now();
//! // ... 执行推理 ...
//! observe_request_duration("/v1/completions", "POST", start.elapsed());
//!
//! // 更新 KV Cache 使用量（第 0 层使用了 1GB）
//! update_kv_cache_usage(0, 1024 * 1024 * 1024);
//!
//! // 更新 Worker 队列深度
//! update_worker_queue_length(42);
//!
//! // 更新活跃连接数
//! update_active_connections(128);
//!
//! // 标记模型已加载
//! update_model_loaded("llama-3-8b", true);
//! ```
//!
//! ## Prometheus 抓取配置
//!
//! 在 `prometheus.yml` 中添加：
//!
//! ```yaml
//! scrape_configs:
//!   - job_name: 'openmini'
//!     scrape_interval: 15s
//!     static_configs:
//!       - targets: ['localhost:9090']  # metrics 端口
//!         labels:
//!           env: production
//!           service: openmini-inference
//! ```
//!
//! ## 指标生命周期管理
//!
//! - **初始化**: 使用 `OnceLock` 实现懒加载，首次访问时注册到 Prometheus
//! - **更新**: 通过提供的函数安全地更新指标值
//! - **清理**: 服务关闭时自动释放（无需手动注销）
//! - **线程安全**: 所有操作都是并发安全的（内部使用 Mutex 保护）

use prometheus::{Gauge, IntCounterVec, HistogramVec, GaugeVec, opts, histogram_opts, register_int_counter_vec, register_histogram_vec, register_gauge_vec};
use std::sync::OnceLock;

/// Token 吞吐量计数器 (Counter)
///
/// 记录推理服务生成的 **token 总数**，按模型和状态分类。
///
/// # 指标定义
///
/// - **名称**: `openmini_inference_tokens_total`
/// - **类型**: Counter (单调递增)
/// - **标签**:
///   - `model`: 模型名称（如 "llama-3-8b", "qwen-7b"）
///   - `status`: 推理结果状态 ("success" | "error")
///
/// # 使用场景
///
/// 1. **计费统计**: 基于 token 用量计算 API 调用费用
/// 2. **吞吐量监控**: 使用 `rate()` 计算 tokens/sec
/// 3. **错误追踪**: 监控 `status="error"` 的 token 占比
///
/// # PromQL 查询示例
///
/// ```promql
/// // 各模型的 token 生成速率 (tokens/sec)
/// sum(rate(openmini_inference_tokens_total[5m])) by (model)
///
/// // 错误率 (%)
/// rate(openmini_inference_tokens_total{status="error"}[5m]) /
/// rate(openmini_inference_tokens_total[5m]) * 100
/// ```
pub fn inference_tokens_total() -> &'static IntCounterVec {
    static INSTANCE: OnceLock<IntCounterVec> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let opts = opts!("openmini_inference_tokens_total", "Total number of inference tokens generated");
        register_int_counter_vec!(opts, &["model", "status"]).unwrap()
    })
}

/// 请求延迟直方图 (Histogram)
///
/// 记录 **推理请求的处理耗时**，支持分位数分析（P50/P95/P99）。
///
/// # 指标定义
///
/// - **名称**: `openmini_request_duration_seconds`
/// - **类型**: Histogram (分桶统计)
/// - **单位**: 秒 (seconds)
/// - **标签**:
///   - `endpoint`: API 端点路径（如 "/v1/completions", "/v1/chat/completions"）
///   - `method`: HTTP 方法（"GET", "POST"）
///
/// # 预定义分桶边界
///
/// `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]` 秒
///
/// 覆盖范围：5ms ~ 10s，适合大多数推理场景。
///
/// # 使用场景
///
/// 1. **SLA 监控**: 检查 P99 延迟是否满足 < 200ms 的 SLA
/// 2. **性能回归检测**: 版本升级后延迟是否异常升高
/// 3. **容量规划**: 根据延迟趋势决定是否扩容
///
/// # PromQL 查询示例
///
/// ```promql
/// // 各端点的 P99 延迟 (秒)
/// histogram_quantile(0.99,
///   sum(rate(openmini_request_duration_seconds_bucket[5m])) by (le, endpoint)
/// )
///
/// // 平均延迟 (秒)
/// sum(rate(openmini_request_duration_seconds_sum[5m])) /
/// sum(rate(openmini_request_duration_seconds_count[5m]))
/// ```
pub fn request_duration_seconds() -> &'static HistogramVec {
    static INSTANCE: OnceLock<HistogramVec> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let opts = histogram_opts!("openmini_request_duration_seconds", "Inference request duration in seconds")
            .buckets(vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]);
        register_histogram_vec!(opts, &["endpoint", "method"]).unwrap()
    })
}

/// KV Cache 内存使用量 (Gauge)
///
/// 实时监控 **KV Cache 各层的内存占用**，单位为字节。
///
/// # 指标定义
///
/// - **名称**: `openmini_kv_cache_usage_bytes`
/// - **类型**: Gauge (可增可减)
/// - **单位**: 字节 (bytes)
/// - **标签**:
///   - `layer`: Transformer 层索引（0, 1, 2, ..., n-1）
///
/// # 使用场景
///
/// 1. **内存规划**: 监控总 KV Cache 占用，防止 OOM
/// 2. **分页策略**: 当内存超限时触发 CPU Offload
/// 3. **性能调优**: 分析各层缓存分布是否均匀
///
/// # 告警建议
///
/// - **警告阈值**: 总占用 > GPU 显存的 80%
/// - **紧急阈值**: 总占用 > GPU 显存的 95%（即将 OOM）
///
/// # PromQL 查询示例
///
/// ```promql
/// // KV Cache 总占用 (GB)
/// sum(openmini_kv_cache_usage_bytes) / 1024 / 1024 / 1024
///
/// // 单层最大占用 (MB)
/// max(openmini_kv_cache_usage_bytes) / 1024 / 1024
///
/// // 内存增长率 (MB/min)
/// deriv(sum(openmini_kv_cache_usage_bytes)[5m]) / 60
/// ```
pub fn kv_cache_usage_bytes() -> &'static GaugeVec {
    static INSTANCE: OnceLock<GaugeVec> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let opts = opts!("openmini_kv_cache_usage_bytes", "Current KV Cache memory usage in bytes");
        register_gauge_vec!(opts, &["layer"]).unwrap()
    })
}

/// Worker 队列深度 (Gauge)
///
/// 监控 **任务队列中等待处理的请求数量**。
///
/// # 指标定义
///
/// - **名称**: `openmini_worker_queue_length`
/// - **类型**: Gauge (瞬时值)
/// - **单位**: 个数 (count)
/// - **标签**: 无
///
/// # 使用场景
///
/// 1. **背压检测**: 队列积压表明系统过载
/// 2. **自动扩容**: 基于队列深度触发水平扩展
/// 3. **负载均衡**: 用于流量调度决策
///
/// # 正常范围
///
/// - **空闲**: < 10
/// - **正常**: 10-50
/// - **繁忙**: 50-100
/// - **过载**: > 100 (需要扩容或限流)
pub fn worker_queue_length() -> &'static Gauge {
    static INSTANCE: OnceLock<Gauge> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let opts = opts!("openmini_worker_queue_length", "Number of pending requests in worker queue");
        prometheus::register_gauge!(opts).unwrap()
    })
}

/// 活跃连接数 (Gauge)
///
/// 记录 **当前活跃的客户端连接数**（HTTP/gRPC/WebSocket）。
///
/// # 指标定义
///
/// - **名称**: `openmini_active_connections`
/// - **类型**: Gauge (瞬时值)
/// - **单位**: 个数 (count)
/// - **标签**: 无
///
/// # 使用场景
///
/// 1. **容量规划**: 监控并发连接数趋势
/// 2. **连接泄露检测**: 异常增长可能表示连接未正确关闭
/// 3. **限流决策**: 基于连接数实施动态限流
pub fn active_connections() -> &'static Gauge {
    static INSTANCE: OnceLock<Gauge> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let opts = opts!("openmini_active_connections", "Number of active client connections");
        prometheus::register_gauge!(opts).unwrap()
    })
}

/// 模型加载状态 (Gauge)
///
/// 标记 **模型是否已加载到内存/GPU**，用于健康检查和告警。
///
/// # 指标定义
///
/// - **名称**: `openmini_model_loaded`
/// - **类型**: Gauge (二值状态)
/// - **取值范围**: 0 (未加载) | 1 (已加载)
/// - **标签**:
///   - `model_name`: 模型标识符（如 "llama-3-8b-q4"）
///
/// # 使用场景
///
/// 1. **健康检查**: 快速判断服务是否可用
/// 2. **故障告警**: 模型意外卸载时立即通知
/// 3. **多模型管理**: 监控多个模型实例的加载状态
pub fn model_loaded() -> &'static GaugeVec {
    static INSTANCE: OnceLock<GaugeVec> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let opts = opts!("openmini_model_loaded", "Whether the model is currently loaded (1=yes, 0=no)");
        register_gauge_vec!(opts, &["model_name"]).unwrap()
    })
}

/// 记录推理请求完成
///
/// 在每次推理完成后调用，更新 token 计数器。
///
/// # 参数
///
/// * `model` - 模型名称（如 "llama-3-8b"），用于标签分类
/// * `token_count` - 本次推理生成的 token 数量
/// * `success` - 推理是否成功 (true=success, false=error)
///
/// # 示例
///
/// ```rust,ignore
/// // 成功完成：生成了 128 个 token
/// record_inference_completion("llama-3-8b", 128, true);
///
/// // 失败：生成了 0 个 token（或部分生成后失败）
/// record_inference_completion("llama-3-8b", 0, false);
/// ```
///
/// # 线程安全
///
/// 此函数是并发安全的，可从多个线程/协程同时调用。
pub fn record_inference_completion(model: &str, token_count: u64, success: bool) {
    let status = if success { "success" } else { "error" };
    inference_tokens_total().with_label_values(&[model, status]).inc_by(token_count);
}

/// 记录请求耗时
///
/// 在 HTTP 请求处理完成后调用，记录请求的总处理时间。
///
/// # 参数
///
/// * `endpoint` - API 端点路径（如 "/v1/completions", "/v1/chat/completions"）
/// * `method` - HTTP 方法（"GET", "POST", "DELETE" 等）
/// * `duration` - 请求总耗时（从接收到响应完成）
///
/// # 最佳实践
///
/// 1. **在中间件中统一记录**: 避免在每个 handler 中重复调用
/// 2. **包含排队时间**: duration 应包含在队列中的等待时间
/// 3. **使用 Instant::now()**: 高精度计时，避免系统时间跳变影响
///
/// # 示例
///
/// ```rust,ignore
/// use std::time::Instant;
///
/// // 在请求开始时
/// let start = Instant::now();
///
/// // ... 处理请求 ...
///
/// // 在请求完成后
/// observe_request_duration("/v1/chat/completions", "POST", start.elapsed());
/// ```
pub fn observe_request_duration(endpoint: &str, method: &str, duration: std::time::Duration) {
    request_duration_seconds()
        .with_label_values(&[endpoint, method])
        .observe(duration.as_secs_f64());
}

/// 更新 KV Cache 使用量
///
/// 在每次请求处理完成后调用，更新指定层的 KV Cache 内存占用。
///
/// # 参数
///
/// * `layer` - Transformer 层索引（从 0 开始）
/// * `bytes` - 该层当前占用的内存字节数
///
/// # 调用时机
///
/// - **请求完成后**: 更新为当前实际占用
/// - **缓存淘汰后**: 减少对应大小
/// - **分页换出时**: 标记为 0 或减少
///
/// # 示例
///
/// ```rust,ignore
/// // 第 0 层使用了 1GB KV Cache
/// update_kv_cache_usage(0, 1024 * 1024 * 1024);
///
/// // 第 1 层使用了 512MB
/// update_kv_cache_usage(1, 512 * 1024 * 1024);
/// ```
pub fn update_kv_cache_usage(layer: usize, bytes: u64) {
    kv_cache_usage_bytes().with_label_values(&[&layer.to_string()]).set(bytes as f64);
}

/// 更新 Worker 队列深度
///
/// 定期或在任务提交/完成时调用，反映当前队列积压情况。
///
/// # 参数
///
/// * `length` - 当前队列中等待处理的任务数量
///
/// # 调用频率建议
///
/// - **高频更新**: 每次任务入队/出队时（推荐）
/// - **低频轮询**: 每 100ms 轮询一次（备选）
///
/// # 示例
///
/// ```rust,ignore
/// // 任务入队后
/// update_worker_queue_length(queue.len() as u64);
///
/// // 任务完成后
/// update_worker_queue_length(queue.len() as u64);
/// ```
pub fn update_worker_queue_length(length: u64) {
    worker_queue_length().set(length as f64);
}

/// 更新活跃连接数
///
/// 在连接建立/断开时调用，跟踪当前活跃的客户端连接数量。
///
/// # 参数
///
/// * `count` - 当前活跃连接总数（HTTP + gRPC + WebSocket）
///
/// # 调用时机
///
/// - **新连接建立时**: `count += 1`
/// - **连接断开/超时时**: `count -= 1`
///
/// # 注意事项
///
/// 确保计数准确：使用 Arc<AtomicI64> 或 Mutex 保护内部计数器，
/// 避免并发修改导致的数据不一致。
pub fn update_active_connections(count: i64) {
    active_connections().set(count as f64);
}

/// 更新模型加载状态
///
/// 在模型加载完成或卸载时调用，标记模型的可用状态。
///
/// # 参数
///
/// * `model_name` - 模型标识符（应与配置中的 model.name 一致）
/// * `loaded` - 是否已加载到内存/GPU (true=1, false=0)
///
/// # 调用时机
///
/// - **启动时**: 模型加载成功后调用 `update_model_loaded("xxx", true)`
/// - **热更新**: 新模型就绪后先标记旧模型为 false，再标记新模型为 true
/// - **错误恢复**: 模型崩溃卸载时调用 `update_model_loaded("xxx", false)`
///
/// # 告警集成
///
/// 此指标通常与 Prometheus AlertManager 集成：
/// ```yaml
/// alert: ModelNotLoaded
/// expr: openmini_model_loaded != 1
/// for: 1m
/// labels:
///   severity: critical
/// ```
pub fn update_model_loaded(model_name: &str, loaded: bool) {
    model_loaded().with_label_values(&[model_name]).set(if loaded { 1.0 } else { 0.0 });
}
