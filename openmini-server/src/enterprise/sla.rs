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
use std::sync::atomic::{AtomicU32, Ordering};
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
    /// 监控目标列表
    targets: Vec<SlaTarget>,
    /// 监控窗口期（秒）
    window: Duration,
    /// 当前活跃的告警
    alerts: Mutex<Vec<SlaAlert>>,
    /// 配置
    config: SlaConfig,
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
                target.current_value.store(f32_to_fixed(value), Ordering::Relaxed);

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
                        severity: self.calculate_severity(&target.metric, current, target.threshold),
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
                    let max = *values.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                    let min = *values.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

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
            SlaMetric::LatencyP95 { .. } => actual - threshold, // 延迟：高于阈值是坏事
            SlaMetric::ErrorRate { .. } => actual - threshold, // 错误率：高于阈值是坏事
            SlaMetric::Throughput { .. } => threshold - actual, // 吞吐量：低于阈值是坏事
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
            .field("current_value", &fixed_to_f32(self.current_value.load(Ordering::Relaxed)))
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
        let avail_violations: Vec<_> = violations.iter()
            .filter(|v| v.target_name == "availability")
            .collect();
        assert!(avail_violations.is_empty(), "Expected no availability violations, but got: {:?}", avail_violations);

        // 记录明显低可用性
        sla.record_metric(&SlaMetric::Availability { target: 99.9 }, 50.0);

        let violations = sla.check_violations();
        let avail_violations: Vec<_> = violations.iter()
            .filter(|v| v.target_name == "availability")
            .collect();
        assert_eq!(avail_violations.len(), 1, "Expected 1 availability violation"); // 50.0 < 99.9，违规
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
            sla.record_metric(&SlaMetric::Availability { target: 99.9 }, 99.95 + (i as f32 * 0.01));
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
}
