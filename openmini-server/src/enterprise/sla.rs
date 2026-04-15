//! # SLA (服务级别协议) 保障模块
//!
//! 提供服务级别监控与告警功能，支持：
//! - **可用性监控** - 服务 uptime 跟踪
//! - **延迟监控** - P95/P99 延迟统计
//! - **错误率监控** - HTTP 错误率追踪
//! - **吞吐量监控** - TPS/QPS 指标
//!
//! ## SLA 指标类型
//!
//! | 指标 | 描述 | 典型阈值 |
//! |------|------|----------|
//! | 可用性 | 服务正常运行时间百分比 | 99.9% |
//! | P95 延迟 | 95% 请求的响应时间 | <1000ms |
//! | 错误率 | HTTP 5xx 错误占比 | <1% |
//! | 吞吐量 | 每秒处理请求数 | >1000 TPS |
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::enterprise::sla::{SlaMonitor, SlaConfig, SlaMetric};
//!
//! let mut sla = SlaMonitor::new(&SlaConfig::default())?;
//!
//! // 记录指标
//! sla.record_metric(SlaMetric::LatencyP95 { max_ms: 1000.0 }, 850.5);
//! sla.record_metric(SlaMetric::Availability { target: 99.9 }, 99.95);
//!
//! // 检查违规
//! let violations = sla.check_violations();
//! if !violations.is_empty() {
//!     // 触发告警
//! }
//!
//! // 生成报告
//! let report = sla.generate_report(Duration::hours(24));
//! println!("SLA Compliance: {:.2}%", report.compliance_rate);
//! ```

use crate::enterprise::SlaConfig;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

/// SLA 错误类型
#[derive(Debug)]
pub enum SlaError {
    /// 目标不存在
    TargetNotFound(String),
    /// 无效的配置
    InvalidConfig(String),
}

impl std::fmt::Display for SlaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlaError::TargetNotFound(s) => write!(f, "SLA target not found: {}", s),
            SlaError::InvalidConfig(s) => write!(f, "Invalid SLA config: {}", s),
        }
    }
}

impl std::error::Error for SlaError {}

/// 将 f32 值转换为定点数（精度0.01）存储为 u32
fn f32_to_fixed(value: f32) -> u32 {
    (value * 100.0) as u32
}

/// 将定点数（u32）转换回 f32 值
fn fixed_to_f32(value: u32) -> f32 {
    value as f32 / 100.0
}

/// SLA 监控与告警管理器
///
/// 核心监控组件，负责：
/// - 收集和聚合服务指标
/// - 检测 SLA 违规并触发告警
/// - 生成合规性报告
///
/// # 线程安全
///
/// 使用 `AtomicU32` 进行无锁指标更新（存储为定点数，精度0.01），支持高并发写入。
/// 内部状态使用 `Mutex` 保护，确保一致性。
///
/// # 示例
///
/// ```rust,ignore
/// let mut sla = SlaMonitor::new(&SlaConfig {
///     availability_threshold: 99.9,
///     latency_p95_ms: 1000.0,
///     ..Default::default()
/// })?;
///
/// // 添加监控目标
/// sla.add_target("api-availability", SlaMetric::Availability { target: 99.9 });
/// sla.add_target("response-latency", SlaMetric::LatencyP95 { max_ms: 500.0 });
///
/// // 记录实时指标
/// for latency in sample_latencies {
///     sla.record_metric(SlaMetric::LatencyP95 { max_ms: 500.0 }, latency);
/// }
/// ```
#[derive(Debug)]
pub struct SlaMonitor {
    /// 监控目标列表（基础版）
    targets: Vec<SlaTarget>,
    /// 监控窗口期（秒）
    window: Duration,
    /// 当前活跃的告警
    alerts: Mutex<Vec<SlaAlert>>,
    /// 配置
    config: SlaConfig,

    // ========== 增强功能字段 ==========
    /// 请求总数计数器
    request_counter: Option<AtomicU64>,
    /// 成功请求计数器
    success_counter: Option<AtomicU64>,
    /// 错误请求计数器
    error_counter: Option<AtomicU64>,
    /// 延迟滑动窗口
    latency_window: Option<Mutex<Vec<f32>>>,
    /// 历史快照记录
    snapshot_history: Option<Mutex<Vec<SlaSnapshot>>>,
    /// 增强型违规记录
    enhanced_violations: Option<Mutex<Vec<EnhancedSlaViolation>>>,
    /// 增强型目标列表
    enhanced_targets: Vec<EnhancedSlaTarget>,
}

impl SlaMonitor {
    /// 创建新的 SLA 监控器
    ///
    /// 根据提供的配置初始化监控系统，并添加默认监控目标。
    ///
    /// # 参数
    ///
    /// * `config` - SLA 配置，包含阈值和窗口期设置
    ///
    /// # 默认监控目标
    ///
    /// 初始化时会自动添加以下目标（如果启用）：
    /// - 可用性监控 (availability)
    /// - P95 延迟监控 (latency-p95)
    /// - 错误率监控 (error-rate)
    /// - 吞吐量监控 (throughput)
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let sla = SlaMonitor::new(&SlaConfig::default())?;
    /// ```
    pub fn new(config: &SlaConfig) -> Result<Self, SlaError> {
        let mut monitor = Self {
            targets: Vec::new(),
            window: Duration::seconds(config.window_secs as i64),
            alerts: Mutex::new(Vec::new()),
            config: config.clone(),
            // 初始化增强功能字段为 None（需要显式调用 init_enhanced_features() 启用）
            request_counter: None,
            success_counter: None,
            error_counter: None,
            latency_window: None,
            snapshot_history: None,
            enhanced_violations: None,
            enhanced_targets: Vec::new(),
        };

        if config.enabled {
            // 添加默认监控目标
            monitor.add_target(
                "availability".to_string(),
                SlaMetric::Availability {
                    target: config.availability_threshold,
                },
            );

            monitor.add_target(
                "latency-p95".to_string(),
                SlaMetric::LatencyP95 {
                    max_ms: config.latency_p95_ms,
                },
            );

            monitor.add_target(
                "error-rate".to_string(),
                SlaMetric::ErrorRate { max_pct: 1.0 },
            );

            monitor.add_target(
                "throughput".to_string(),
                SlaMetric::Throughput { min_tps: 100.0 },
            );
        }

        Ok(monitor)
    }

    /// 添加监控目标
    ///
    /// 注册一个新的 SLA 监控目标。
    ///
    /// # 参数
    ///
    /// * `name` - 目标名称（唯一标识符）
    /// * `metric` - 指标类型和阈值配置
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// sla.add_target(
    ///     "custom-metric",
    ///     SlaMetric::LatencyP95 { max_ms: 200.0 },
    /// );
    /// ```
    pub fn add_target(&mut self, name: String, metric: SlaMetric) {
        // 先计算 threshold，避免 move 后借用问题
        let threshold = match &metric {
            SlaMetric::Availability { target } => *target,
            SlaMetric::LatencyP95 { max_ms } => *max_ms,
            SlaMetric::ErrorRate { max_pct } => *max_pct,
            SlaMetric::Throughput { min_tps } => *min_tps,
        };

        self.targets.push(SlaTarget {
            name,
            metric,
            threshold,
            current_value: AtomicU32::new(0),
            samples: Mutex::new(Vec::new()),
        });
    }

    /// 记录指标值
    ///
    /// 更新指定指标的当前值。此方法是线程安全的，
    /// 可以从多个并发任务中调用。
    ///
    /// # 参数
    ///
    /// * `metric` - 要更新的指标类型
    /// * `value` - 新的测量值
    ///
    /// # 性能说明
    ///
    /// 使用原子操作更新当前值，O(1) 时间复杂度。
    /// 同时将样本添加到滑动窗口用于统计分析。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// // 在请求处理完成后记录延迟
    /// let latency_ms = response_time.as_millis() as f32;
    /// sla.record_metric(SlaMetric::LatencyP95 { max_ms: 1000.0 }, latency_ms);
    /// ```
    pub fn record_metric(&self, metric: &SlaMetric, value: f32) {
        for target in &self.targets {
            if &target.metric == metric {
                // 原子更新当前值（转换为定点数存储）
                target
                    .current_value
                    .store(f32_to_fixed(value), Ordering::Relaxed);

                // 添加样本到窗口
                if let Ok(mut samples) = target.samples.lock() {
                    samples.push(Sample {
                        timestamp: Utc::now(),
                        value,
                    });

                    // 清理过期样本
                    let cutoff = Utc::now() - self.window;
                    samples.retain(|s| s.timestamp > cutoff);
                }
                break;
            }
        }
    }

    /// 检查 SLA 是否违反
    ///
    /// 扫描所有监控目标，检测是否有任何指标超出阈值。
    ///
    /// # 返回值
    ///
    /// 返回所有当前违反 SLA 的目标列表。空向量表示所有指标正常。
    ///
    /// # 违规判定逻辑
    ///
    /// 根据指标类型不同，违规条件也不同：
    /// - **可用性**: current < threshold (越低越差)
    /// - **延迟**: current > threshold (越高越差)
    /// - **错误率**: current > threshold (越高越差)
    /// - **吞吐量**: current < threshold (越低越差)
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let violations = sla.check_violations();
    /// for v in violations {
    ///     trigger_alert(&v);
    /// }
    /// ```
    pub fn check_violations(&self) -> Vec<SlaViolation> {
        let violations: Vec<SlaViolation> = self
            .targets
            .iter()
            .filter_map(|target| {
                let current = fixed_to_f32(target.current_value.load(Ordering::Relaxed));
                let violated = match target.metric {
                    SlaMetric::Availability { .. } => current < target.threshold,
                    SlaMetric::LatencyP95 { .. } => current > target.threshold,
                    SlaMetric::ErrorRate { .. } => current > target.threshold,
                    SlaMetric::Throughput { .. } => current < target.threshold,
                };

                if violated {
                    Some(SlaViolation {
                        target_name: target.name.clone(),
                        threshold: target.threshold,
                        actual_value: current,
                        violation_time: Utc::now(),
                        severity: self.calculate_severity(
                            &target.metric,
                            current,
                            target.threshold,
                        ),
                    })
                } else {
                    None
                }
            })
            .collect();

        // 记录告警
        if !violations.is_empty() {
            if let Ok(mut alerts) = self.alerts.lock() {
                for violation in &violations {
                    alerts.push(SlaAlert {
                        violation_id: uuid::Uuid::new_v4(),
                        target_name: violation.target_name.clone(),
                        triggered_at: violation.violation_time,
                        severity: violation.severity,
                        acknowledged: false,
                    });
                }
            }
        }

        violations
    }

    /// 获取 SLA 报告
    ///
    /// 生成指定时间段内的 SLA 合规性报告。
    ///
    /// # 参数
    ///
    /// * `period` - 报告覆盖的时间段
    ///
    /// # 返回值
    ///
    /// 返回包含以下信息的报告：
    /// - 各指标的平均值、最大值、最小值
    /// - 合规率（满足 SLA 的目标比例）
    /// - 违规次数统计
    /// - 整体健康评分
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// // 生成最近 24 小时的报告
    /// let report = sla.generate_report(Duration::hours(24));
    /// println!("Overall Health Score: {:.1}%", report.health_score);
    /// ```
    pub fn generate_report(&self, period: Duration) -> SlaReport {
        let now = Utc::now();
        let start = now - period;

        let mut metrics_summary: Vec<MetricSummary> = Vec::new();
        let mut total_targets = 0;
        let mut compliant_targets = 0;

        for target in &self.targets {
            total_targets += 1;

            if let Ok(samples) = target.samples.lock() {
                let period_samples: Vec<&Sample> = samples
                    .iter()
                    .filter(|s| s.timestamp >= start && s.timestamp <= now)
                    .collect();

                if !period_samples.is_empty() {
                    let values: Vec<f32> = period_samples.iter().map(|s| s.value).collect();
                    let avg = values.iter().sum::<f32>() / values.len() as f32;
                    let max = *values
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();
                    let min = *values
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap();

                    // 检查是否合规（使用平均值）
                    let is_compliant = match target.metric {
                        SlaMetric::Availability { .. } => avg >= target.threshold,
                        SlaMetric::LatencyP95 { .. } => avg <= target.threshold,
                        SlaMetric::ErrorRate { .. } => avg <= target.threshold,
                        SlaMetric::Throughput { .. } => avg >= target.threshold,
                    };

                    if is_compliant {
                        compliant_targets += 1;
                    }

                    metrics_summary.push(MetricSummary {
                        name: target.name.clone(),
                        metric_type: format!("{:?}", target.metric),
                        average: avg,
                        maximum: max,
                        minimum: min,
                        threshold: target.threshold,
                        is_compliant,
                        sample_count: values.len(),
                    });
                }
            }
        }

        let compliance_rate = if total_targets > 0 {
            (compliant_targets as f32 / total_targets as f32) * 100.0
        } else {
            100.0
        };

        // 计算健康评分 (0-100)
        let health_score = compliance_rate;

        // 获取活跃告警数量
        let active_alerts = self.alerts.lock().map_or(0, |a| a.len());

        SlaReport {
            generated_at: now,
            period_start: start,
            period_end: now,
            metrics: metrics_summary,
            compliance_rate,
            health_score,
            total_violations: active_alerts,
            targets_total: total_targets,
            targets_compliant: compliant_targets,
        }
    }

    /// 获取当前所有目标的指标快照
    pub fn get_current_metrics(&self) -> Vec<MetricSnapshot> {
        self.targets
            .iter()
            .map(|target| {
                let current = fixed_to_f32(target.current_value.load(Ordering::Relaxed));
                MetricSnapshot {
                    name: target.name.clone(),
                    metric: target.metric.clone(),
                    current_value: current,
                    threshold: target.threshold,
                    status: {
                        match target.metric {
                            SlaMetric::Availability { .. } => {
                                if current >= target.threshold {
                                    MetricStatus::Healthy
                                } else {
                                    MetricStatus::Violated
                                }
                            }
                            _ => {
                                if current <= target.threshold {
                                    MetricStatus::Healthy
                                } else {
                                    MetricStatus::Violated
                                }
                            }
                        }
                    },
                }
            })
            .collect()
    }

    /// 获取活跃告警列表
    pub fn get_alerts(&self) -> Vec<SlaAlert> {
        self.alerts.lock().unwrap().clone()
    }

    /// 确认告警
    pub fn acknowledge_alert(&self, alert_id: &uuid::Uuid) -> bool {
        if let Ok(mut alerts) = self.alerts.lock() {
            if let Some(alert) = alerts.iter_mut().find(|a| &a.violation_id == alert_id) {
                alert.acknowledged = true;
                return true;
            }
        }
        false
    }

    /// 清除已确认的告警
    pub fn clear_acknowledged_alerts(&self) -> usize {
        if let Ok(mut alerts) = self.alerts.lock() {
            let before = alerts.len();
            alerts.retain(|a| !a.acknowledged);
            return before - alerts.len();
        }
        0
    }

    // ========== 内部方法 ==========

    /// 计算违规严重程度
    fn calculate_severity(&self, metric: &SlaMetric, actual: f32, threshold: f32) -> Severity {
        let deviation = match metric {
            SlaMetric::Availability { .. } => threshold - actual, // 可用性：低于阈值是坏事
            SlaMetric::LatencyP95 { .. } => actual - threshold,   // 延迟：高于阈值是坏事
            SlaMetric::ErrorRate { .. } => actual - threshold,    // 错误率：高于阈值是坏事
            SlaMetric::Throughput { .. } => threshold - actual,   // 吞吐量：低于阈值是坏事
        };

        // 根据偏离程度判断严重级别
        let deviation_pct = (deviation / threshold.abs().max(0.01)) * 100.0;

        if deviation_pct > 50.0 {
            Severity::Critical
        } else if deviation_pct > 20.0 {
            Severity::Warning
        } else {
            Severity::Info
        }
    }
}

/// SLA 监控目标
struct SlaTarget {
    /// 目标名称
    name: String,
    /// 指标类型和配置
    metric: SlaMetric,
    /// 阈值
    threshold: f32,
    /// 当前值（原子操作，存储为定点数，精度0.01）
    current_value: AtomicU32,
    /// 历史样本（用于报告生成）
    samples: Mutex<Vec<Sample>>,
}

impl std::fmt::Debug for SlaTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlaTarget")
            .field("name", &self.name)
            .field("metric", &self.metric)
            .field("threshold", &self.threshold)
            .field(
                "current_value",
                &fixed_to_f32(self.current_value.load(Ordering::Relaxed)),
            )
            .finish()
    }
}

/// 指标样本
struct Sample {
    timestamp: DateTime<Utc>,
    value: f32,
}

/// SLA 指标类型枚举
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SlaMetric {
    /// 可用性指标（目标百分比）
    Availability {
        /// 目标可用性（如 99.9 表示 99.9%）
        target: f32,
    },
    /// P95 延迟指标（毫秒）
    LatencyP95 {
        /// 最大允许延迟（毫秒）
        max_ms: f32,
    },
    /// 错误率指标（百分比）
    ErrorRate {
        /// 最大允许错误率（如 1.0 表示 1%）
        max_pct: f32,
    },
    /// 吞吐量指标（TPS）
    Throughput {
        /// 最小吞吐量要求
        min_tps: f32,
    },
}

/// SLA 违规记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaViolation {
    /// 违规的目标名称
    pub target_name: String,
    /// 阈值
    pub threshold: f32,
    /// 实际值
    pub actual_value: f32,
    /// 违规发生时间
    pub violation_time: DateTime<Utc>,
    /// 严重程度
    pub severity: Severity,
}

/// 严重程度枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// 信息级别（轻微偏差）
    Info,
    /// 警告级别（需要关注）
    Warning,
    /// 严重级别（立即处理）
    Critical,
}

impl Severity {
    /// 转换为字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Critical => "critical",
        }
    }
}

/// SLA 告警
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaAlert {
    /// 告警唯一标识
    pub violation_id: uuid::Uuid,
    /// 关联的目标名称
    pub target_name: String,
    /// 触发时间
    pub triggered_at: DateTime<Utc>,
    /// 严重程度
    pub severity: Severity,
    /// 是否已被确认
    pub acknowledged: bool,
}

/// SLA 报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaReport {
    /// 报告生成时间
    pub generated_at: DateTime<Utc>,
    /// 报告周期开始
    pub period_start: DateTime<Utc>,
    /// 报告周期结束
    pub period_end: DateTime<Utc>,
    /// 各指标汇总
    pub metrics: Vec<MetricSummary>,
    /// 合规率（百分比）
    pub compliance_rate: f32,
    /// 健康评分（0-100）
    pub health_score: f32,
    /// 总违规次数
    pub total_violations: usize,
    /// 总目标数
    pub targets_total: usize,
    /// 合规目标数
    pub targets_compliant: usize,
}

/// 指标汇总信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    /// 指标名称
    pub name: String,
    /// 指标类型描述
    pub metric_type: String,
    /// 平均值
    pub average: f32,
    /// 最大值
    pub maximum: f32,
    /// 最小值
    pub minimum: f32,
    /// 阈值
    pub threshold: f32,
    /// 是否合规
    pub is_compliant: bool,
    /// 样本数量
    pub sample_count: usize,
}

/// 指标快照（实时状态）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    /// 指标名称
    pub name: String,
    /// 指标类型
    pub metric: SlaMetric,
    /// 当前值
    pub current_value: f32,
    /// 阈值
    pub threshold: f32,
    /// 状态
    pub status: MetricStatus,
}

/// 指标状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetricStatus {
    /// 健康（在阈值范围内）
    Healthy,
    /// 违规（超出阈值）
    Violated,
}

// ==================== 增强的 SLA 监控系统 ====================

/// SLA 级别枚举
///
/// 定义四个级别的服务级别协议标准，每个级别有不同的可用性目标和阈值配置。
/// 用于对服务进行分级管理和差异化监控。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SlaLevel {
    /// P0 - 关键业务（99.99%可用性）
    ///
    /// 适用于核心支付、用户认证等关键路径服务。
    P0 {
        /// 目标可用性百分比
        target_availability: f32,
    },
    /// P1 - 重要业务（99.9%可用性）
    ///
    /// 适用于订单管理、消息推送等重要业务。
    P1 {
        /// 目标可用性百分比
        target_availability: f32,
    },
    /// P2 - 一般业务（99%可用性）
    ///
    /// 适用于报表生成、数据分析等一般业务。
    P2 {
        /// 目标可用性百分比
        target_availability: f32,
    },
    /// P3 - 低优先级（95%可用性）
    ///
    /// 适用于日志收集、后台任务等低优先级服务。
    P3 {
        /// 目标可用性百分比
        target_availability: f32,
    },
}

impl SlaLevel {
    /// 获取目标可用性
    pub fn target_availability(&self) -> f32 {
        match self {
            SlaLevel::P0 {
                target_availability,
            } => *target_availability,
            SlaLevel::P1 {
                target_availability,
            } => *target_availability,
            SlaLevel::P2 {
                target_availability,
            } => *target_availability,
            SlaLevel::P3 {
                target_availability,
            } => *target_availability,
        }
    }

    /// 获取级别名称
    pub fn level_name(&self) -> &'static str {
        match self {
            SlaLevel::P0 { .. } => "P0",
            SlaLevel::P1 { .. } => "P1",
            SlaLevel::P2 { .. } => "P2",
            SlaLevel::P3 { .. } => "P3",
        }
    }

    /// 获取级别的典型阈值配置
    ///
    /// 返回元组：(max_latency_p95_ms, max_error_rate_pct, min_throughput_tps)
    pub fn typical_thresholds(&self) -> (f32, f32, u32) {
        match self {
            SlaLevel::P0 { .. } => (100.0, 0.1, 5000),
            SlaLevel::P1 { .. } => (500.0, 0.5, 2000),
            SlaLevel::P2 { .. } => (1000.0, 1.0, 1000),
            SlaLevel::P3 { .. } => (2000.0, 5.0, 100),
        }
    }

    /// 根据目标可用性创建对应级别
    pub fn from_target(target: f32) -> Self {
        if target >= 99.99 {
            SlaLevel::P0 {
                target_availability: target,
            }
        } else if target >= 99.9 {
            SlaLevel::P1 {
                target_availability: target,
            }
        } else if target >= 99.0 {
            SlaLevel::P2 {
                target_availability: target,
            }
        } else {
            SlaLevel::P3 {
                target_availability: target,
            }
        }
    }
}

/// SLA 目标定义（增强版）
///
/// 包含完整的服务级别协议定义，支持多维度指标监控。
/// 可用于注册到监控系统并进行合规性检查。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSlaTarget {
    /// 目标唯一标识符
    pub id: String,
    /// 目标名称
    pub name: String,
    /// SLA 级别
    pub level: SlaLevel,
    /// 最大 P95 延迟（毫秒）
    pub max_latency_p95_ms: f32,
    /// 最大错误率（百分比）
    pub max_error_rate_pct: f32,
    /// 最小吞吐量（TPS）
    pub min_throughput_tps: u32,
    /// 创建时间
    pub created_at: DateTime<Utc>,
}

impl EnhancedSlaTarget {
    /// 使用级别典型阈值创建目标
    ///
    /// 自动根据 SLA 级别设置推荐的阈值配置。
    pub fn from_level(id: &str, name: &str, level: SlaLevel) -> Self {
        let (latency, error_rate, throughput) = level.typical_thresholds();
        Self {
            id: id.to_string(),
            name: name.to_string(),
            level,
            max_latency_p95_ms: latency,
            max_error_rate_pct: error_rate,
            min_throughput_tps: throughput,
            created_at: Utc::now(),
        }
    }

    /// 创建自定义阈值的 SLA 目标
    pub fn new(
        id: &str,
        name: &str,
        level: SlaLevel,
        max_latency_p95_ms: f32,
        max_error_rate_pct: f32,
        min_throughput_tps: u32,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            level,
            max_latency_p95_ms,
            max_error_rate_pct,
            min_throughput_tps,
            created_at: Utc::now(),
        }
    }

    /// 检查给定指标是否符合目标
    ///
    /// 所有指标都必须满足要求才算合规。
    pub fn is_compliant(
        &self,
        availability: f32,
        latency_p95: f32,
        error_rate: f32,
        throughput: u32,
    ) -> bool {
        availability >= self.level.target_availability()
            && latency_p95 <= self.max_latency_p95_ms
            && error_rate <= self.max_error_rate_pct
            && throughput >= self.min_throughput_tps
    }
}

/// SLA 快照（用于趋势分析）
///
/// 记录某一时刻的系统状态快照，可用于历史趋势分析和性能回溯。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaSnapshot {
    /// 快照时间戳
    pub timestamp: DateTime<Utc>,
    /// 当前可用性百分比
    pub availability_pct: f32,
    /// P95 延迟（毫秒）
    pub latency_p95_ms: f32,
    /// P99 延迟（毫秒）
    pub latency_p99_ms: f32,
    /// 错误率百分比
    pub error_rate_pct: f32,
    /// 吞吐量（TPS）
    pub throughput_tps: f32,
    /// 总请求数
    pub total_requests: u64,
    /// 违规次数
    pub violations_count: u32,
}

impl Default for SlaSnapshot {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            availability_pct: 100.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
            error_rate_pct: 0.0,
            throughput_tps: 0.0,
            total_requests: 0,
            violations_count: 0,
        }
    }
}

/// SLA 违规类型（增强版）
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlaViolationType {
    /// 可用性低于目标
    AvailabilityBelowTarget,
    /// P95 延迟超限
    LatencyExceededP95,
    /// 错误率超标
    ErrorRateExceeded,
    /// 吞吐量不足
    ThroughputBelowMinimum,
}

impl std::fmt::Display for SlaViolationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlaViolationType::AvailabilityBelowTarget => write!(f, "可用性低于目标"),
            SlaViolationType::LatencyExceededP95 => write!(f, "P95延迟超限"),
            SlaViolationType::ErrorRateExceeded => write!(f, "错误率超标"),
            SlaViolationType::ThroughputBelowMinimum => write!(f, "吞吐量不足"),
        }
    }
}

/// 违规严重程度（增强版）
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// 严重（P0 违规）
    Critical,
    /// 主要（P1 违规）
    Major,
    /// 次要（P2 违规）
    Minor,
    /// 警告（P3 违规）
    Warning,
}

impl std::fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViolationSeverity::Critical => write!(f, "严重"),
            ViolationSeverity::Major => write!(f, "主要"),
            ViolationSeverity::Minor => write!(f, "次要"),
            ViolationSeverity::Warning => write!(f, "警告"),
        }
    }
}

impl ViolationSeverity {
    /// 从 SLA 级别推断严重程度
    pub fn from_level(level: &SlaLevel) -> Self {
        match level {
            SlaLevel::P0 { .. } => ViolationSeverity::Critical,
            SlaLevel::P1 { .. } => ViolationSeverity::Major,
            SlaLevel::P2 { .. } => ViolationSeverity::Minor,
            SlaLevel::P3 { .. } => ViolationSeverity::Warning,
        }
    }
}

/// SLA 违规记录（增强版）
///
/// 记录详细的违规信息，包括类型、严重程度、实际值与阈值对比等。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSlaViolation {
    /// 违规唯一标识符
    pub id: String,
    /// 关联的目标 ID
    pub target_id: String,
    /// 违规类型
    pub violation_type: SlaViolationType,
    /// 违规发生时间
    pub timestamp: DateTime<Utc>,
    /// 实际测量值
    pub actual_value: f32,
    /// 阈值限制
    pub threshold_value: f32,
    /// 严重程度
    pub severity: ViolationSeverity,
    /// 是否已解决
    pub resolved: bool,
    /// 解决时间
    pub resolved_at: Option<DateTime<Utc>>,
}

impl EnhancedSlaViolation {
    /// 创建新的违规记录
    pub fn new(
        target_id: &str,
        violation_type: SlaViolationType,
        actual_value: f32,
        threshold_value: f32,
        severity: ViolationSeverity,
    ) -> Self {
        Self {
            id: format!("vio-{}", uuid::Uuid::new_v4()),
            target_id: target_id.to_string(),
            violation_type,
            timestamp: Utc::now(),
            actual_value,
            threshold_value,
            severity,
            resolved: false,
            resolved_at: None,
        }
    }

    /// 标记为已解决
    pub fn resolve(&mut self) {
        self.resolved = true;
        self.resolved_at = Some(Utc::now());
    }
}

// ==================== 增强 SlaMonitor 实现 ====================

impl SlaMonitor {
    /// 记录单个请求的指标
    ///
    /// 高性能 O(1) 操作，使用原子计数器更新统计信息。
    /// 适用于高并发场景下的实时指标采集。
    ///
    /// # 参数
    ///
    /// * `latency_ms` - 请求延迟（毫秒）
    /// * `success` - 请求是否成功
    pub fn record_request(&self, latency_ms: f32, success: bool) {
        // 更新总请求数
        if let Some(counter) = &self.request_counter {
            counter.fetch_add(1, Ordering::Relaxed);
        }

        // 更新成功/失败计数
        if success {
            if let Some(counter) = &self.success_counter {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        } else if let Some(counter) = &self.error_counter {
            counter.fetch_add(1, Ordering::Relaxed);
        }

        // 更新延迟统计（使用滑动窗口）
        if let Some(latencies) = &self.latency_window {
            if let Ok(mut window) = latencies.lock() {
                window.push(latency_ms);
                // 保持窗口大小限制
                if window.len() > 10000 {
                    let excess = window.len() - 10000;
                    window.drain(0..excess);
                }
            }
        }
    }

    /// 获取当前状态快照
    ///
    /// 基于当前累积的指标数据生成系统状态快照，
    /// 包含可用性、延迟、错误率和吞吐量等核心指标。
    pub fn current_status(&self) -> SlaSnapshot {
        let now = Utc::now();

        // 计算各项指标
        let total_requests = self
            .request_counter
            .as_ref()
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0);

        let success_count = self
            .success_counter
            .as_ref()
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0);

        let error_count = self
            .error_counter
            .as_ref()
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0);

        // 计算可用性
        let availability_pct = if total_requests > 0 {
            (success_count as f32 / total_requests as f32) * 100.0
        } else {
            100.0
        };

        // 计算错误率
        let error_rate_pct = if total_requests > 0 {
            (error_count as f32 / total_requests as f32) * 100.0
        } else {
            0.0
        };

        // 计算 P95/P99 延迟
        let (latency_p95_ms, latency_p99_ms) = self.calculate_percentile_latencies();

        // 计算吞吐量（简化实现：基于最近请求数）
        let throughput_tps = if total_requests > 0 {
            total_requests as f32 / 60.0 // 假设窗口为 60 秒
        } else {
            0.0
        };

        // 获取违规次数
        let violations_count = self
            .enhanced_violations
            .as_ref()
            .and_then(|v| v.lock().ok())
            .map(|v| v.len() as u32)
            .unwrap_or(0);

        SlaSnapshot {
            timestamp: now,
            availability_pct,
            latency_p95_ms,
            latency_p99_ms,
            error_rate_pct,
            throughput_tps,
            total_requests,
            violations_count,
        }
    }

    /// 检查是否有违规（针对增强型目标）
    ///
    /// 对比当前状态与指定目标的阈值要求，
    /// 如果有任何指标不达标，返回第一个检测到的违规记录。
    pub fn check_enhanced_violation(
        &self,
        target: &EnhancedSlaTarget,
    ) -> Option<EnhancedSlaViolation> {
        let status = self.current_status();
        let severity = ViolationSeverity::from_level(&target.level);

        // 检查可用性
        if status.availability_pct < target.level.target_availability() {
            return Some(EnhancedSlaViolation::new(
                &target.id,
                SlaViolationType::AvailabilityBelowTarget,
                status.availability_pct,
                target.level.target_availability(),
                severity,
            ));
        }

        // 检查 P95 延迟
        if status.latency_p95_ms > target.max_latency_p95_ms {
            return Some(EnhancedSlaViolation::new(
                &target.id,
                SlaViolationType::LatencyExceededP95,
                status.latency_p95_ms,
                target.max_latency_p95_ms,
                severity,
            ));
        }

        // 检查错误率
        if status.error_rate_pct > target.max_error_rate_pct {
            return Some(EnhancedSlaViolation::new(
                &target.id,
                SlaViolationType::ErrorRateExceeded,
                status.error_rate_pct,
                target.max_error_rate_pct,
                severity,
            ));
        }

        // 检查吞吐量
        if status.throughput_tps < target.min_throughput_tps as f32 {
            return Some(EnhancedSlaViolation::new(
                &target.id,
                SlaViolationType::ThroughputBelowMinimum,
                status.throughput_tps,
                target.min_throughput_tps as f32,
                severity,
            ));
        }

        None
    }

    /// 获取历史趋势数据
    ///
    /// 从历史快照记录中提取指定时间窗口内的数据点，
    /// 用于绘制趋势图和分析性能变化模式。
    ///
    /// # 参数
    ///
    /// * `window` - 时间窗口长度
    ///
    /// # 返回值
    ///
    /// 返回按时间排序的快照列表，可用于可视化展示。
    pub fn trend(&self, window: Duration) -> Vec<SlaSnapshot> {
        let cutoff = Utc::now() - window;

        self.snapshot_history
            .as_ref()
            .and_then(|h| h.lock().ok())
            .map(|snapshots| {
                snapshots
                    .iter()
                    .filter(|s| s.timestamp >= cutoff)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// 注册 SLA 目标（增强版）
    ///
    /// 将增强型 SLA 目标注册到监控系统中进行持续监控。
    ///
    /// # 参数
    ///
    /// * `target` - 要注册的 SLA 目标
    ///
    /// # 错误
    ///
    /// 如果目标 ID 已存在，返回错误。
    pub fn register_target(&mut self, target: EnhancedSlaTarget) -> Result<(), SlaError> {
        // 检查是否已存在相同 ID 的目标
        if self.enhanced_targets.iter().any(|t| t.id == target.id) {
            return Err(SlaError::InvalidConfig(format!(
                "Target with id '{}' already exists",
                target.id
            )));
        }

        self.enhanced_targets.push(target);
        Ok(())
    }

    /// 注销 SLA 目标
    ///
    /// 从监控系统中移除指定的 SLA 目标。
    ///
    /// # 参数
    ///
    /// * `target_id` - 要注销的目标 ID
    ///
    /// # 错误
    ///
    /// 如果目标不存在，返回错误。
    pub fn unregister_target(&mut self, target_id: &str) -> Result<(), SlaError> {
        let original_len = self.enhanced_targets.len();
        self.enhanced_targets.retain(|t| t.id != target_id);

        if self.enhanced_targets.len() == original_len {
            Err(SlaError::TargetNotFound(target_id.to_string()))
        } else {
            Ok(())
        }
    }

    /// 获取所有活跃目标的合规状态
    ///
    /// 对每个注册的增强型目标执行合规性检查，
    /// 返回包含目标 ID、合规状态和当前快照的报告列表。
    pub fn compliance_report(&self) -> Vec<(String, bool, SlaSnapshot)> {
        let current_status = self.current_status();

        self.enhanced_targets
            .iter()
            .map(|target| {
                let is_compliant = target.is_compliant(
                    current_status.availability_pct,
                    current_status.latency_p95_ms,
                    current_status.error_rate_pct,
                    current_status.throughput_tps as u32,
                );
                (target.id.clone(), is_compliant, current_status.clone())
            })
            .collect()
    }

    /// 初始化增强功能的内部状态
    pub fn init_enhanced_features(&mut self) {
        self.request_counter = Some(AtomicU64::new(0));
        self.success_counter = Some(AtomicU64::new(0));
        self.error_counter = Some(AtomicU64::new(0));
        self.latency_window = Some(Mutex::new(Vec::new()));
        self.snapshot_history = Some(Mutex::new(Vec::new()));
        self.enhanced_violations = Some(Mutex::new(Vec::new()));
        self.enhanced_targets = Vec::new();
    }

    /// 保存当前快照到历史记录
    pub fn save_snapshot(&self) {
        let snapshot = self.current_status();
        if let Some(history) = &self.snapshot_history {
            if let Ok(mut hist) = history.lock() {
                hist.push(snapshot);
                // 限制历史记录数量
                if hist.len() > 10000 {
                    let excess = hist.len() - 10000;
                    hist.drain(0..excess);
                }
            }
        }
    }

    // ========== 内部辅助方法 ==========

    /// 计算百分位延迟
    fn calculate_percentile_latencies(&self) -> (f32, f32) {
        if let Some(latencies) = &self.latency_window {
            if let Ok(window) = latencies.lock() {
                if window.is_empty() {
                    return (0.0, 0.0);
                }

                let mut sorted: Vec<f32> = window.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let p95_idx = ((sorted.len() as f32 * 0.95) as usize).min(sorted.len() - 1);
                let p99_idx = ((sorted.len() as f32 * 0.99) as usize).min(sorted.len() - 1);

                return (sorted[p95_idx], sorted[p99_idx]);
            }
        }
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sla_monitor() -> SlaMonitor {
        let config = SlaConfig {
            enabled: true,
            window_secs: 300,
            availability_threshold: 99.9,
            latency_p95_ms: 1000.0,
        };
        SlaMonitor::new(&config).expect("Failed to create SlaMonitor")
    }

    #[test]
    fn test_sla_monitor_creation() {
        let sla = create_test_sla_monitor();
        assert!(!sla.get_current_metrics().is_empty());
        // 应该有 4 个默认目标
        assert_eq!(sla.get_current_metrics().len(), 4);
    }

    #[test]
    fn test_record_and_check_availability() {
        let sla = create_test_sla_monitor();

        // 记录正常可用性（阈值是 99.9，记录 100.0 应该不违规）
        sla.record_metric(&SlaMetric::Availability { target: 99.9 }, 100.0);

        let violations = sla.check_violations();
        // 检查是否有 availability 相关的违规
        let avail_violations: Vec<_> = violations
            .iter()
            .filter(|v| v.target_name == "availability")
            .collect();
        assert!(
            avail_violations.is_empty(),
            "Expected no availability violations, but got: {:?}",
            avail_violations
        );

        // 记录明显低可用性
        sla.record_metric(&SlaMetric::Availability { target: 99.9 }, 50.0);

        let violations = sla.check_violations();
        let avail_violations: Vec<_> = violations
            .iter()
            .filter(|v| v.target_name == "availability")
            .collect();
        assert_eq!(
            avail_violations.len(),
            1,
            "Expected 1 availability violation"
        ); // 50.0 < 99.9，违规
        assert_eq!(violations[0].target_name, "availability");
    }

    #[test]
    fn test_record_and_check_latency() {
        let sla = create_test_sla_monitor();

        // 正常延迟
        sla.record_metric(&SlaMetric::LatencyP95 { max_ms: 1000.0 }, 800.0);
        let violations = sla.check_violations();
        let latency_violations: Vec<_> = violations
            .iter()
            .filter(|v| v.target_name == "latency-p95")
            .collect();
        assert!(latency_violations.is_empty());

        // 高延迟
        sla.record_metric(&SlaMetric::LatencyP95 { max_ms: 1000.0 }, 1500.0);
        let violations = sla.check_violations();
        let latency_violations: Vec<_> = violations
            .iter()
            .filter(|v| v.target_name == "latency-p95")
            .collect();
        assert_eq!(latency_violations.len(), 1); // 1500 > 1000，违规
    }

    #[test]
    fn test_error_rate_monitoring() {
        let sla = create_test_sla_monitor();

        // 低错误率
        sla.record_metric(&SlaMetric::ErrorRate { max_pct: 1.0 }, 0.5);
        let violations = sla.check_violations();
        assert!(violations.iter().all(|v| v.target_name != "error-rate"));

        // 高错误率
        sla.record_metric(&SlaMetric::ErrorRate { max_pct: 1.0 }, 5.0);
        let violations = sla.check_violations();
        assert!(violations.iter().any(|v| v.target_name == "error-rate"));
    }

    #[test]
    fn test_throughput_monitoring() {
        let sla = create_test_sla_monitor();

        // 高吞吐量
        sla.record_metric(&SlaMetric::Throughput { min_tps: 100.0 }, 150.0);
        let violations = sla.check_violations();
        assert!(violations.iter().all(|v| v.target_name != "throughput"));

        // 低吞吐量
        sla.record_metric(&SlaMetric::Throughput { min_tps: 100.0 }, 50.0);
        let violations = sla.check_violations();
        assert!(violations.iter().any(|v| v.target_name == "throughput"));
    }

    #[test]
    fn test_add_custom_target() {
        let mut sla = create_test_sla_monitor();

        sla.add_target(
            "custom-latency".to_string(),
            SlaMetric::LatencyP95 { max_ms: 200.0 },
        );

        let metrics = sla.get_current_metrics();
        assert!(metrics.iter().any(|m| m.name == "custom-latency"));

        // 测试自定义目标
        sla.record_metric(&SlaMetric::LatencyP95 { max_ms: 200.0 }, 250.0);
        let violations = sla.check_violations();
        assert!(violations.iter().any(|v| v.target_name == "custom-latency"));
    }

    #[test]
    fn test_generate_report() {
        let sla = create_test_sla_monitor();

        // 记录一些数据
        for i in 0..10 {
            sla.record_metric(
                &SlaMetric::Availability { target: 99.9 },
                99.95 + (i as f32 * 0.01),
            );
        }

        let report = sla.generate_report(Duration::hours(1));
        assert!(report.compliance_rate > 0.0);
        assert!(report.health_score > 0.0);
        assert_eq!(report.targets_total, 4); // 默认 4 个目标
    }

    #[test]
    fn test_alert_lifecycle() {
        let sla = create_test_sla_monitor();

        // 触发违规
        sla.record_metric(&SlaMetric::Availability { target: 99.9 }, 95.0);
        let violations = sla.check_violations();
        assert!(!violations.is_empty());

        // 检查告警
        let alerts = sla.get_alerts();
        assert!(!alerts.is_empty());

        // 确认告警
        let alert_id = alerts[0].violation_id;
        assert!(sla.acknowledge_alert(&alert_id));

        // 清除已确认的告警
        let cleared = sla.clear_acknowledged_alerts();
        assert!(cleared > 0);
    }

    #[test]
    fn test_current_metrics_snapshot() {
        let sla = create_test_sla_monitor();

        sla.record_metric(&SlaMetric::Availability { target: 99.9 }, 99.95);
        sla.record_metric(&SlaMetric::LatencyP95 { max_ms: 1000.0 }, 500.0);

        let snapshot = sla.get_current_metrics();

        let avail = snapshot.iter().find(|m| m.name == "availability").unwrap();
        assert!((avail.current_value - 99.95).abs() < f32::EPSILON);
        assert_eq!(avail.status, MetricStatus::Healthy);

        let latency = snapshot.iter().find(|m| m.name == "latency-p95").unwrap();
        assert!((latency.current_value - 500.0).abs() < f32::EPSILON);
        assert_eq!(latency.status, MetricStatus::Healthy);
    }

    #[test]
    fn test_severity_calculation() {
        let sla = create_test_sla_monitor();

        // 轻微违规（阈值 99.9，记录 99.0，偏离约 1%）
        sla.record_metric(&SlaMetric::Availability { target: 99.9 }, 99.0);
        let violations = sla.check_violations();
        let avail_violation = violations.iter().find(|v| v.target_name == "availability");
        // 偏离 = (99.9 - 99.0) / 99.9 * 100 ≈ 0.9%，应该是 Info
        assert_eq!(avail_violation.unwrap().severity, Severity::Info);

        // 严重违规（阈值 99.9，记录 10.0，偏离约 90%）
        sla.record_metric(&SlaMetric::Availability { target: 99.9 }, 10.0);
        let violations = sla.check_violations();
        let avail_violation = violations.iter().find(|v| v.target_name == "availability");
        // 偏离 = (99.9 - 10.0) / 99.9 * 100 ≈ 90%，应该是 Critical
        assert_eq!(avail_violation.unwrap().severity, Severity::Critical);
    }

    #[test]
    fn test_multiple_simultaneous_violations() {
        let sla = create_test_sla_monitor();

        // 同时触发多个违规
        sla.record_metric(&SlaMetric::Availability { target: 99.9 }, 95.0);
        sla.record_metric(&SlaMetric::LatencyP95 { max_ms: 1000.0 }, 2000.0);
        sla.record_metric(&SlaMetric::ErrorRate { max_pct: 1.0 }, 10.0);

        let violations = sla.check_violations();
        assert!(violations.len() >= 3); // 至少 3 个违规
    }

    #[test]
    fn test_disabled_monitoring() {
        let config = SlaConfig {
            enabled: false,
            ..SlaConfig::default()
        };
        let sla = SlaMonitor::new(&config).unwrap();

        // 禁用时不应该有默认目标
        assert!(sla.get_current_metrics().is_empty());
    }

    // ==================== 新增增强功能测试 ====================

    // SlaLevel 测试
    #[test]
    fn test_sla_level_p0_target_availability() {
        let level = SlaLevel::P0 {
            target_availability: 99.99,
        };
        assert!((level.target_availability() - 99.99).abs() < 0.001);
        assert_eq!(level.level_name(), "P0");
    }

    #[test]
    fn test_sla_level_typical_thresholds() {
        // 测试每个级别的典型阈值是否合理
        let levels = vec![
            SlaLevel::P0 {
                target_availability: 99.99,
            },
            SlaLevel::P1 {
                target_availability: 99.9,
            },
            SlaLevel::P2 {
                target_availability: 99.0,
            },
            SlaLevel::P3 {
                target_availability: 95.0,
            },
        ];
        for level in &levels {
            let (lat, err, tps) = level.typical_thresholds();
            assert!(lat > 0.0);
            assert!(err >= 0.0);
            assert!(tps > 0);
        }
    }

    // EnhancedSlaTarget 测试
    #[test]
    fn test_enhanced_target_from_level() {
        let target = EnhancedSlaTarget::from_level(
            "test-p0",
            "Critical API",
            SlaLevel::P0 {
                target_availability: 99.99,
            },
        );
        assert_eq!(target.id, "test-p0");
        assert!(target.is_compliant(99.99, 50.0, 0.05, 6000)); // 应该合规
        assert!(!target.is_compliant(99.0, 50.0, 0.05, 6000)); // 可用性不足
    }

    // SlaMonitor record_request 测试
    #[test]
    fn test_record_request_statistics() {
        let config = SlaConfig::default();
        let mut monitor = SlaMonitor::new(&config).unwrap();
        monitor.init_enhanced_features();

        // 记录10次成功请求
        for _ in 0..10 {
            monitor.record_request(100.0, true);
        }
        // 记录2次失败请求
        monitor.record_request(500.0, false);

        let status = monitor.current_status();
        assert_eq!(status.total_requests, 12);
        // 错误率应该约为 2/12 ≈ 16.67%
        assert!((status.error_rate_pct - 16.67).abs() < 0.1);
    }

    // check_enhanced_violation 测试
    #[test]
    fn test_check_violation_latency_exceeded() {
        let config = SlaConfig::default();
        let mut monitor = SlaMonitor::new(&config).unwrap();
        monitor.init_enhanced_features();

        let target = EnhancedSlaTarget::from_level(
            "lat-test",
            "Latency Test",
            SlaLevel::P1 {
                target_availability: 99.9,
            },
        );
        monitor.register_target(target).unwrap();

        // 模拟高延迟请求
        for _ in 0..100 {
            monitor.record_request(2000.0, true); // P1阈值是500ms，这会超限
        }

        let targets = monitor.enhanced_targets.clone(); // 获取目标列表
        if let Some(violation) = monitor.check_enhanced_violation(&targets[0]) {
            assert_eq!(
                violation.violation_type,
                SlaViolationType::LatencyExceededP95
            );
        }
    }

    // trend 测试
    #[test]
    fn test_trend_data_extraction() {
        let config = SlaConfig::default();
        let mut monitor = SlaMonitor::new(&config).unwrap();
        monitor.init_enhanced_features();

        // 记录一些数据
        for i in 0..20 {
            monitor.record_request(100.0 + i as f32, true);
            monitor.save_snapshot(); // 手动保存快照以生成趋势数据
        }

        let trends = monitor.trend(Duration::minutes(5));
        assert!(!trends.is_empty());
    }

    // register_target / unregister_target 测试
    #[test]
    fn test_target_lifecycle() {
        let config = SlaConfig::default();
        let mut monitor = SlaMonitor::new(&config).unwrap();
        monitor.init_enhanced_features();

        let target = EnhancedSlaTarget::from_level(
            "lc-test",
            "Lifecycle Test",
            SlaLevel::P2 {
                target_availability: 99.0,
            },
        );

        // 注册
        monitor.register_target(target).unwrap();
        assert_eq!(monitor.enhanced_targets.len(), 1);

        // 注销
        monitor.unregister_target("lc-test").unwrap();
        assert_eq!(monitor.enhanced_targets.len(), 0);

        // 重复注销应报错
        assert!(monitor.unregister_target("lc-test").is_err());
    }

    // compliance_report 测试
    #[test]
    fn test_compliance_report_generation() {
        let config = SlaConfig::default();
        let mut monitor = SlaMonitor::new(&config).unwrap();
        monitor.init_enhanced_features();

        let target = EnhancedSlaTarget::from_level(
            "comp-test",
            "Compliance Test",
            SlaLevel::P3 {
                target_availability: 95.0,
            },
        );
        monitor.register_target(target).unwrap();

        // 正常操作（应该合规）
        for _ in 0..50 {
            monitor.record_request(100.0, true);
        }

        let report = monitor.compliance_report();
        assert_eq!(report.len(), 1); // 应该有1个目标的报告
        assert!(report[0].1); // 应该是合规的(true)
    }

    // 边界情况测试
    #[test]
    fn test_edge_cases_zero_requests() {
        let config = SlaConfig::default();
        let mut monitor = SlaMonitor::new(&config).unwrap();
        monitor.init_enhanced_features();

        let status = monitor.current_status();
        assert_eq!(status.total_requests, 0);
        assert_eq!(status.error_rate_pct, 0.0); // 无请求时错误率应为0
    }

    #[test]
    fn test_edge_cases_extreme_latency() {
        let config = SlaConfig::default();
        let mut monitor = SlaMonitor::new(&config).unwrap();
        monitor.init_enhanced_features();

        // 极端延迟值
        monitor.record_request(99999.0, true);
        let status = monitor.current_status();
        assert!(status.latency_p95_ms > 10000.0);
    }

    #[test]
    fn test_edge_cases_100_percent_error() {
        let config = SlaConfig::default();
        let mut monitor = SlaMonitor::new(&config).unwrap();
        monitor.init_enhanced_features();

        for _ in 0..20 {
            monitor.record_request(100.0, false);
        }

        let status = monitor.current_status();
        assert!((status.error_rate_pct - 100.0).abs() < 0.1);
    }

    // SlaSnapshot Default 测试
    #[test]
    fn test_snapshot_default_values() {
        let snapshot = SlaSnapshot::default();
        assert_eq!(snapshot.total_requests, 0);
        assert_eq!(snapshot.violations_count, 0);
    }

    // ViolationSeverity from_level 测试
    #[test]
    fn test_violation_severity_from_level() {
        assert_eq!(
            ViolationSeverity::from_level(&SlaLevel::P0 {
                target_availability: 99.99
            }),
            ViolationSeverity::Critical
        );
        assert_eq!(
            ViolationSeverity::from_level(&SlaLevel::P1 {
                target_availability: 99.9
            }),
            ViolationSeverity::Major
        );
        assert_eq!(
            ViolationSeverity::from_level(&SlaLevel::P2 {
                target_availability: 99.0
            }),
            ViolationSeverity::Minor
        );
        assert_eq!(
            ViolationSeverity::from_level(&SlaLevel::P3 {
                target_availability: 95.0
            }),
            ViolationSeverity::Warning
        );
    }

    // EnhancedSlaViolation resolve 测试
    #[test]
    fn test_violation_resolve() {
        let mut violation = EnhancedSlaViolation::new(
            "v1",
            SlaViolationType::AvailabilityBelowTarget,
            99.0,
            99.99,
            ViolationSeverity::Critical,
        );
        assert!(!violation.resolved);

        violation.resolve();
        assert!(violation.resolved);
        assert!(violation.resolved_at.is_some());
    }
}
