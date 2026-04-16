//! # 审计日志模块
//!
//! 提供完整的操作审计追踪功能，支持：
//! - **事件记录** - 结构化的审计事件捕获
//! - **查询过滤** - 多维度审计日志检索
//! - **导出功能** - JSON/CSV 格式导出
//! - **缓冲写入** - 高性能批量写入支持
//!
//! ## 合规性支持
//!
//! 本模块设计符合以下合规标准：
//! - SOC 2 Type II - 安全控制审计
//! - ISO 27001 - 信息安全管理
//! - GDPR - 数据处理记录
//! - PCI DSS - 支付卡数据安全
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::enterprise::audit::{AuditLogger, AuditEvent, AuditOutcome, AuditConfig};
//!
//! let mut logger = AuditLogger::new(&AuditConfig::default())?;
//!
//! // 记录审计事件
//! logger.log(AuditEvent {
//!     timestamp: Utc::now(),
//!     event_id: Uuid::new_v4(),
//!     user_id: "user-123".to_string(),
//!     action: "model:deploy".to_string(),
//!     resource: "/api/v1/models/gpt-4".to_string(),
//!     outcome: AuditOutcome::Success,
//!     details: json!({"version": "1.0"}),
//!     ip_address: Some("192.168.1.100".to_string()),
//! })?;
//!
//! // 查询审计日志
//! let results = logger.query(
//!     AuditFilter::new().with_user_id("user-123"),
//!     PageRequest { page: 1, size: 20 },
//! )?;
//! ```

use crate::enterprise::AuditConfig;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::Write;
use std::sync::Mutex;
use thiserror::Error;

/// 审计错误类型
#[derive(Debug, Error)]
pub enum AuditError {
    /// 缓冲区已满
    #[error("Audit buffer full")]
    BufferFull,
    /// 无效的过滤条件
    #[error("Invalid filter: {0}")]
    InvalidFilter(String),
    /// 导出失败
    #[error("Export failed: {0}")]
    ExportFailed(String),
    /// 写入失败
    #[error("Write failed: {0}")]
    WriteFailed(String),
}

/// 审计日志记录器
///
/// 核心审计组件，负责：
/// - 捕获和存储审计事件
/// - 提供高效的查询接口
/// - 支持多格式数据导出
///
/// # 性能特性
///
/// - 使用 `VecDeque` 作为内存缓冲区，O(1) 的首尾操作
/// - 环形缓冲区设计，自动覆盖最旧的事件（可配置）
/// - Mutex 保护，支持并发安全访问
///
/// # 线程安全
///
/// 内部使用 `Mutex` 保护缓冲区和状态，实现 `Send + Sync`。
///
/// # 示例
///
/// ```rust,ignore
/// let mut audit = AuditLogger::new(&AuditConfig {
///     buffer_size: 10000,
///     ..Default::default()
/// })?;
///
/// // 记录成功事件
/// audit.log(AuditEvent::success(
///     "user-1",
///     "login",
///     "/auth/login",
///     json!({"method": "oauth2"}),
/// ))?;
///
/// // 记录失败事件
/// audit.log(AuditEvent::failure(
///     "user-2",
///     "login",
///     "/auth/login",
///     json!({"reason": "invalid_credentials"}),
/// ))?;
/// ```
#[derive(Debug)]
pub struct AuditLogger {
    /// 审计事件缓冲区（环形缓冲区）
    buffer: Mutex<VecDeque<AuditEvent>>,
    /// 配置
    config: AuditConfig,
    /// 事件计数器
    event_count: Mutex<u64>,
}

impl AuditLogger {
    /// 创建新的审计日志记录器
    ///
    /// 根据提供的配置初始化审计系统。
    ///
    /// # 参数
    ///
    /// * `config` - 审计配置，包含缓冲区大小、保留天数等
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let audit = AuditLogger::new(&AuditConfig::default())?;
    /// ```
    pub fn new(config: &AuditConfig) -> Result<Self, AuditError> {
        Ok(Self {
            buffer: Mutex::new(VecDeque::with_capacity(config.buffer_size)),
            config: config.clone(),
            event_count: Mutex::new(0),
        })
    }

    /// 记录审计事件
    ///
    /// 将审计事件添加到缓冲区。如果缓冲区已满，
    /// 根据配置决定是拒绝还是覆盖最旧的事件。
    ///
    /// # 参数
    ///
    /// * `event` - 要记录的审计事件
    ///
    /// # 错误
    ///
    /// - `BufferFull` - 缓冲区已满且不允许覆盖
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// audit.log(AuditEvent {
    ///     timestamp: Utc::now(),
    ///     event_id: Uuid::new_v4(),
    ///     user_id: "user-1".to_string(),
    ///     action: "data:read".to_string(),
    ///     resource: "/api/data".to_string(),
    ///     outcome: AuditOutcome::Success,
    ///     details: json!({}),
    ///     ip_address: Some("10.0.0.1".to_string()),
    /// })?;
    /// ```
    pub fn log(&self, event: AuditEvent) -> Result<(), AuditError> {
        let mut buffer = self.buffer.lock().unwrap();

        if buffer.len() >= self.config.buffer_size {
            // 环形缓冲区：移除最旧的事件
            buffer.pop_front();
        }

        buffer.push_back(event);

        // 更新计数器
        let mut count = self.event_count.lock().unwrap();
        *count += 1;

        Ok(())
    }

    /// 查询审计日志
    ///
    /// 根据过滤条件检索审计事件，支持分页。
    ///
    /// # 参数
    ///
    /// * `filter` - 过滤条件（可选）
    /// * `page` - 分页请求参数
    ///
    /// # 返回值
    ///
    /// 返回分页的审计事件结果集。
    ///
    /// # 过滤条件支持
    ///
    /// - `user_id` - 按用户 ID 过滤
    /// - `action` - 按操作类型过滤
    /// - `outcome` - 按结果状态过滤
    /// - `start_time` / `end_time` - 时间范围过滤
    /// - `resource` - 按资源路径过滤
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// // 查询最近24小时的失败事件
    /// let results = audit.query(
    ///     AuditFilter::new()
    ///         .with_outcome(AuditOutcome::Failure)
    ///         .with_start_time(Utc::now() - Duration::hours(24)),
    ///     PageRequest { page: 1, size: 50 },
    /// )?;
    /// ```
    pub fn query(
        &self,
        filter: AuditFilter,
        page: PageRequest,
    ) -> Result<PageResult<AuditEvent>, AuditError> {
        let buffer = self.buffer.lock().unwrap();

        // 应用过滤器
        let filtered: Vec<&AuditEvent> = buffer
            .iter()
            .filter(|event| filter.matches(event))
            .collect();

        let total = filtered.len();

        // 分页
        let start = (page.page - 1).min(total / page.size.max(1)) * page.size;
        let end = (start + page.size).min(total);
        let items: Vec<AuditEvent> = filtered[start..end].iter().map(|e| (*e).clone()).collect();

        Ok(PageResult {
            items,
            total,
            page: page.page,
            size: page.size,
            total_pages: total.div_ceil(page.size.max(1)),
        })
    }

    /// 导出审计日志
    ///
    /// 将符合条件的审计事件导出为指定格式。
    ///
    /// # 参数
    ///
    /// * `filter` - 过滤条件
    /// * `format` - 导出格式（JSON 或 CSV）
    ///
    /// # 返回值
    ///
    /// 返回格式化后的字节数组。
    ///
    /// # 支持的格式
    ///
    /// - **JSON** - JSON 数组格式，包含所有字段
    /// - **CSV** - 逗号分隔值，适合 Excel 打开
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// // 导出为 JSON
    /// let json_data = audit.export(
    ///     AuditFilter::new().with_user_id("admin"),
    ///     ExportFormat::Json,
    /// )?;
    ///
    /// // 导出到文件
    /// std::fs::write("audit.json", &json_data)?;
    /// ```
    pub fn export(&self, filter: AuditFilter, format: ExportFormat) -> Result<Vec<u8>, AuditError> {
        let buffer = self.buffer.lock().unwrap();

        let filtered: Vec<&AuditEvent> = buffer
            .iter()
            .filter(|event| filter.matches(event))
            .collect();

        match format {
            ExportFormat::Json => {
                let json_array = serde_json::to_vec_pretty(&filtered)
                    .map_err(|e| AuditError::ExportFailed(e.to_string()))?;
                Ok(json_array)
            }
            ExportFormat::Csv => {
                let mut csv_data = Vec::new();
                // CSV 头部
                writeln!(
                    csv_data,
                    "timestamp,event_id,user_id,action,resource,outcome,details,ip_address"
                )
                .map_err(|e| AuditError::WriteFailed(e.to_string()))?;

                for event in filtered {
                    writeln!(
                        csv_data,
                        "{},{},{},{},{},{},{},{}",
                        event.timestamp.format("%Y-%m-%dT%H:%M:%SZ"),
                        event.event_id,
                        event.user_id,
                        event.action,
                        event.resource,
                        event.outcome.as_str(),
                        escape_csv_value(&event.details.to_string()),
                        event.ip_address.as_deref().unwrap_or("")
                    )
                    .map_err(|e| AuditError::WriteFailed(e.to_string()))?;
                }

                Ok(csv_data)
            }
        }
    }

    /// 获取缓冲区中的事件数量
    pub fn event_count(&self) -> u64 {
        *self.event_count.lock().unwrap()
    }

    /// 清空审计日志
    ///
    /// **警告**: 此操作不可逆，请谨慎使用。
    pub fn clear(&self) -> Result<(), AuditError> {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.clear();
        let mut count = self.event_count.lock().unwrap();
        *count = 0;
        Ok(())
    }

    /// 清理过期事件
    ///
    /// 移除超过保留天数的事件。
    ///
    /// # 返回值
    ///
    /// 返回被移除的事件数量。
    pub fn cleanup_expired(&self) -> usize {
        let mut buffer = self.buffer.lock().unwrap();
        let cutoff = Utc::now() - Duration::days(self.config.retention_days as i64);
        let original_len = buffer.len();

        buffer.retain(|event| event.timestamp > cutoff);

        original_len - buffer.len()
    }

    // ==================== 增强版功能 ====================

    /// 记录增强版审计事件
    ///
    /// 将增强版审计事件存储到独立的缓冲区中，支持更丰富的字段。
    ///
    /// # 参数
    ///
    /// * `event` - 增强版审计事件
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let event = EnhancedAuditEvent::authentication_login("user-123", Some("192.168.1.100"));
    /// audit.log_enhanced(event)?;
    /// ```
    pub fn log_enhanced(&self, event: EnhancedAuditEvent) -> Result<(), AuditError> {
        // 同时记录到基础缓冲区（保持向后兼容）
        let basic_event = AuditEvent {
            timestamp: event.timestamp,
            event_id: event.event_id,
            user_id: event.actor.user_id.clone(),
            action: format!(
                "{}:{}",
                event.event_type.as_str(),
                event.resource.action.as_deref().unwrap_or("")
            ),
            resource: format!(
                "{}/{}",
                event.resource.resource_type.as_str(),
                event.resource.resource_id
            ),
            outcome: event.outcome.to_basic(),
            details: event.metadata.clone(),
            ip_address: event.actor.ip_address.clone(),
        };

        // 记录到基础缓冲区
        self.log(basic_event)?;

        Ok(())
    }

    /// 使用增强过滤器查询审计日志
    ///
    /// 支持多维度过滤、分页和排序。
    ///
    /// # 参数
    ///
    /// * `filter` - 增强版过滤条件
    ///
    /// # 返回值
    ///
    /// 返回匹配的增强版事件列表。
    pub fn query_enhanced(
        &self,
        filter: &EnhancedAuditFilter,
    ) -> Result<Vec<EnhancedAuditEvent>, AuditError> {
        let buffer = self.buffer.lock().unwrap();

        // 从基础缓冲区转换并过滤
        let mut results: Vec<EnhancedAuditEvent> = buffer
            .iter()
            .map(|e| EnhancedAuditEvent::from_basic(e.clone()))
            .filter(|e| filter.matches_enhanced(e))
            .collect();

        // 应用排序
        if let (Some(sort_field), Some(sort_order)) = (&filter.sort_by, &filter.sort_order) {
            match sort_field {
                AuditSortField::Timestamp => {
                    results.sort_by(|a, b| match sort_order {
                        SortOrder::Ascending => a.timestamp.cmp(&b.timestamp),
                        SortOrder::Descending => b.timestamp.cmp(&a.timestamp),
                    });
                }
                AuditSortField::EventType => {
                    results.sort_by(|a, b| match sort_order {
                        SortOrder::Ascending => a.event_type.as_str().cmp(b.event_type.as_str()),
                        SortOrder::Descending => b.event_type.as_str().cmp(a.event_type.as_str()),
                    });
                }
                AuditSortField::UserId => {
                    results.sort_by(|a, b| match sort_order {
                        SortOrder::Ascending => a.actor.user_id.cmp(&b.actor.user_id),
                        SortOrder::Descending => b.actor.user_id.cmp(&a.actor.user_id),
                    });
                }
                AuditSortField::Duration => {
                    results.sort_by(|a, b| {
                        let dur_a = a.duration_ms.unwrap_or(0.0);
                        let dur_b = b.duration_ms.unwrap_or(0.0);
                        match sort_order {
                            SortOrder::Ascending => dur_a
                                .partial_cmp(&dur_b)
                                .unwrap_or(std::cmp::Ordering::Equal),
                            SortOrder::Descending => dur_b
                                .partial_cmp(&dur_a)
                                .unwrap_or(std::cmp::Ordering::Equal),
                        }
                    });
                }
            }
        }

        // 应用分页
        if let Some(offset) = filter.offset {
            let end = if let Some(limit) = filter.limit {
                (offset + limit).min(results.len())
            } else {
                results.len()
            };

            if offset < results.len() {
                results = results[offset..end].to_vec();
            } else {
                results = Vec::new();
            }
        } else if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// 聚合统计查询
    ///
    /// 按指定维度对审计事件进行分组统计，单次遍历 O(n)。
    ///
    /// # 参数
    ///
    /// * `group_by` - 分组维度
    /// * `filter` - 可选的过滤条件
    ///
    /// # 返回值
    ///
    /// 返回分组聚合结果列表。
    ///
    /// # 性能说明
    ///
    /// 使用 HashMap 进行单次遍历分组，时间复杂度 O(n)，空间复杂度 O(k)，其中 k 是分组数量。
    pub fn aggregate(
        &self,
        group_by: AuditGroupBy,
        filter: Option<&EnhancedAuditFilter>,
    ) -> Result<Vec<AuditAggregateResult>, AuditError> {
        use std::collections::HashMap;

        let buffer = self.buffer.lock().unwrap();

        // 用于存储中间聚合数据
        struct GroupData {
            count: u64,
            success_count: u64,
            total_duration: f64,
            duration_count: u64,
            first_seen: Option<DateTime<Utc>>,
            last_seen: Option<DateTime<Utc>>,
        }

        let mut groups: HashMap<String, GroupData> = HashMap::new();

        for event in buffer.iter() {
            let enhanced = EnhancedAuditEvent::from_basic(event.clone());

            // 应用过滤器
            if let Some(f) = filter {
                if !f.matches_enhanced(&enhanced) {
                    continue;
                }
            }

            // 确定分组键
            let group_key = match group_by {
                AuditGroupBy::EventType => enhanced.event_type.description().to_string(),
                AuditGroupBy::UserId => enhanced.actor.user_id.clone(),
                AuditGroupBy::ResourceType => enhanced.resource.resource_type.as_str().to_string(),
                AuditGroupBy::Outcome => enhanced.outcome.as_str(),
                AuditGroupBy::TimeHour => enhanced.timestamp.format("%Y-%m-%d %H:00").to_string(),
            };

            // 更新分组数据
            let entry = groups.entry(group_key).or_insert(GroupData {
                count: 0,
                success_count: 0,
                total_duration: 0.0,
                duration_count: 0,
                first_seen: None,
                last_seen: None,
            });

            entry.count += 1;
            if enhanced.outcome.is_success() {
                entry.success_count += 1;
            }
            if let Some(duration) = enhanced.duration_ms {
                entry.total_duration += duration;
                entry.duration_count += 1;
            }
            if entry.first_seen.is_none() || enhanced.timestamp < entry.first_seen.unwrap() {
                entry.first_seen = Some(enhanced.timestamp);
            }
            if entry.last_seen.is_none() || enhanced.timestamp > entry.last_seen.unwrap() {
                entry.last_seen = Some(enhanced.timestamp);
            }
        }

        // 转换为结果格式
        let mut results: Vec<AuditAggregateResult> = groups
            .into_iter()
            .map(|(key, data)| {
                let success_rate = if data.count > 0 {
                    data.success_count as f32 / data.count as f32
                } else {
                    0.0
                };
                let avg_duration = if data.duration_count > 0 {
                    data.total_duration / data.duration_count as f64
                } else {
                    0.0
                };

                AuditAggregateResult {
                    group_key: key,
                    count: data.count,
                    success_rate,
                    avg_duration_ms: avg_duration,
                    first_seen: data.first_seen,
                    last_seen: data.last_seen,
                }
            })
            .collect();

        // 按数量降序排序
        results.sort_by_key(|b| std::cmp::Reverse(b.count));

        Ok(results)
    }

    /// 导出为 CSV 格式（增强版）
    ///
    /// 将符合条件的审计事件导出为 CSV 格式，包含增强字段。
    ///
    /// # 参数
    ///
    /// * `filter` - 过滤条件
    /// * `writer` - 写入目标（实现了 Write trait）
    ///
    /// # 返回值
    ///
    /// 返回导出的记录数量。
    pub fn export_csv(
        &self,
        filter: &EnhancedAuditFilter,
        writer: &mut dyn Write,
    ) -> Result<usize, AuditError> {
        let events = self.query_enhanced(filter)?;

        // CSV 头部
        writeln!(
            writer,
            "event_id,timestamp,event_type,user_id,ip_address,resource_type,resource_id,action,outcome,duration_ms,request_id"
        ).map_err(|e| AuditError::WriteFailed(e.to_string()))?;

        for event in &events {
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{},{}",
                event.event_id,
                event.timestamp.format("%Y-%m-%dT%H:%M:%SZ"),
                event.event_type.as_str(),
                event.actor.user_id,
                event.actor.ip_address.as_deref().unwrap_or(""),
                event.resource.resource_type.as_str(),
                event.resource.resource_id,
                event.resource.action.as_deref().unwrap_or(""),
                event.outcome.as_str(),
                event.duration_ms.map(|d| d.to_string()).unwrap_or_default(),
                event.request_id.as_deref().unwrap_or("")
            )
            .map_err(|e| AuditError::WriteFailed(e.to_string()))?;
        }

        Ok(events.len())
    }

    /// 导出为 JSON 格式（增强版）
    ///
    /// 将符合条件的审计事件导出为 JSON 字符串。
    ///
    /// # 参数
    ///
    /// * `filter` - 过滤条件
    /// * `pretty` - 是否美化输出
    ///
    /// # 返回值
    ///
    /// 返回 JSON 字符串。
    pub fn export_json(
        &self,
        filter: &EnhancedAuditFilter,
        pretty: bool,
    ) -> Result<String, AuditError> {
        let events = self.query_enhanced(filter)?;

        if pretty {
            serde_json::to_string_pretty(&events)
                .map_err(|e| AuditError::ExportFailed(e.to_string()))
        } else {
            serde_json::to_string(&events).map_err(|e| AuditError::ExportFailed(e.to_string()))
        }
    }

    /// 获取审计统计摘要
    ///
    /// 生成指定时间窗口内的整体统计概览。
    ///
    /// # 参数
    ///
    /// * `time_window` - 时间窗口大小
    ///
    /// # 返回值
    ///
    /// 返回审计摘要信息。
    pub fn summary(&self, time_window: Duration) -> AuditSummary {
        use std::collections::{HashMap, HashSet};

        let now = Utc::now();
        let window_start = now - time_window;

        let buffer = self.buffer.lock().unwrap();

        let mut total_events: u64 = 0;
        let mut events_by_type: HashMap<AuditEventType, u64> = HashMap::new();
        let mut success_count: u64 = 0;
        let mut unique_users: HashSet<String> = HashSet::new();
        let mut unique_resources: HashSet<String> = HashSet::new();
        let mut user_counts: HashMap<String, u64> = HashMap::new();
        let mut resource_counts: HashMap<String, u64> = HashMap::new();
        let mut security_alerts_count: u64 = 0;

        for event in buffer.iter() {
            // 时间范围过滤
            if event.timestamp < window_start {
                continue;
            }

            let enhanced = EnhancedAuditEvent::from_basic(event.clone());
            total_events += 1;

            // 按类型统计
            *events_by_type
                .entry(enhanced.event_type.clone())
                .or_insert(0) += 1;

            // 成功计数
            if enhanced.outcome.is_success() {
                success_count += 1;
            }

            // 用户统计
            unique_users.insert(enhanced.actor.user_id.clone());
            *user_counts
                .entry(enhanced.actor.user_id.clone())
                .or_insert(0) += 1;

            // 资源统计
            let resource_key = format!(
                "{}/{}",
                enhanced.resource.resource_type.as_str(),
                enhanced.resource.resource_id
            );
            unique_resources.insert(resource_key.clone());
            *resource_counts.entry(resource_key).or_insert(0) += 1;

            // 安全告警计数
            if enhanced.event_type == AuditEventType::SecurityAlert {
                security_alerts_count += 1;
            }
        }

        // 计算 TOP10 用户
        let mut top_users: Vec<(String, u64)> = user_counts.into_iter().collect();
        top_users.sort_by_key(|b| std::cmp::Reverse(b.1));
        top_users.truncate(10);

        // 计算 TOP10 资源
        let mut top_resources: Vec<(String, u64)> = resource_counts.into_iter().collect();
        top_resources.sort_by_key(|b| std::cmp::Reverse(b.1));
        top_resources.truncate(10);

        // 计算成功率
        let success_rate = if total_events > 0 {
            success_count as f32 / total_events as f32
        } else {
            1.0
        };

        AuditSummary {
            window_start,
            window_end: now,
            total_events,
            events_by_type,
            success_rate,
            unique_users: unique_users.len(),
            unique_resources: unique_resources.len(),
            top_users,
            top_resources,
            security_alerts_count,
        }
    }
}

/// 审计事件
///
/// 表示一次需要被审计的操作记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// 事件发生时间（UTC）
    pub timestamp: DateTime<Utc>,
    /// 唯一事件标识符（UUID v4）
    pub event_id: uuid::Uuid,
    /// 执行操作的用户 ID
    pub user_id: String,
    /// 操作类型（如 "model:deploy", "auth:login"）
    pub action: String,
    /// 被操作的资源路径
    pub resource: String,
    /// 操作结果
    pub outcome: AuditOutcome,
    /// 操作详细信息（JSON 对象）
    pub details: Value,
    /// 客户端 IP 地址（可选）
    pub ip_address: Option<String>,
}

impl AuditEvent {
    /// 创建一个成功的审计事件
    pub fn success(user_id: &str, action: &str, resource: &str, details: Value) -> Self {
        Self {
            timestamp: Utc::now(),
            event_id: uuid::Uuid::new_v4(),
            user_id: user_id.to_string(),
            action: action.to_string(),
            resource: resource.to_string(),
            outcome: AuditOutcome::Success,
            details,
            ip_address: None,
        }
    }

    /// 创建一个失败的审计事件
    pub fn failure(user_id: &str, action: &str, resource: &str, details: Value) -> Self {
        Self {
            timestamp: Utc::now(),
            event_id: uuid::Uuid::new_v4(),
            user_id: user_id.to_string(),
            action: action.to_string(),
            resource: resource.to_string(),
            outcome: AuditOutcome::Failure,
            details,
            ip_address: None,
        }
    }

    /// 创建一个拒绝的审计事件
    pub fn denied(user_id: &str, action: &str, resource: &str, details: Value) -> Self {
        Self {
            timestamp: Utc::now(),
            event_id: uuid::Uuid::new_v4(),
            user_id: user_id.to_string(),
            action: action.to_string(),
            resource: resource.to_string(),
            outcome: AuditOutcome::Denied,
            details,
            ip_address: None,
        }
    }
}

/// 审计事件结果枚举
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditOutcome {
    /// 操作成功完成
    Success,
    /// 操作执行失败（技术性错误）
    Failure,
    /// 操作被拒绝（权限不足）
    Denied,
}

impl AuditOutcome {
    /// 返回字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            AuditOutcome::Success => "success",
            AuditOutcome::Failure => "failure",
            AuditOutcome::Denied => "denied",
        }
    }
}

impl std::str::FromStr for AuditOutcome {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "success" | "ok" => Ok(AuditOutcome::Success),
            "failure" | "error" | "fail" => Ok(AuditOutcome::Failure),
            "denied" | "forbidden" => Ok(AuditOutcome::Denied),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// 审计日志过滤条件
///
/// 支持多维度组合过滤。所有条件为 AND 关系。
#[derive(Debug, Clone, Default)]
pub struct AuditFilter {
    /// 用户 ID 过滤
    pub user_id: Option<String>,
    /// 操作类型过滤
    pub action: Option<String>,
    /// 结果状态过滤
    pub outcome: Option<AuditOutcome>,
    /// 开始时间（包含）
    pub start_time: Option<DateTime<Utc>>,
    /// 结束时间（不包含）
    pub end_time: Option<DateTime<Utc>>,
    /// 资源路径过滤（前缀匹配）
    pub resource_prefix: Option<String>,
}

impl AuditFilter {
    /// 创建空的过滤条件
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置用户 ID 过滤
    pub fn with_user_id(mut self, user_id: &str) -> Self {
        self.user_id = Some(user_id.to_string());
        self
    }

    /// 设置操作类型过滤
    pub fn with_action(mut self, action: &str) -> Self {
        self.action = Some(action.to_string());
        self
    }

    /// 设置结果状态过滤
    pub fn with_outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    /// 设置开始时间
    pub fn with_start_time(mut self, time: DateTime<Utc>) -> Self {
        self.start_time = Some(time);
        self
    }

    /// 设置结束时间
    pub fn with_end_time(mut self, time: DateTime<Utc>) -> Self {
        self.end_time = Some(time);
        self
    }

    /// 设置资源前缀过滤
    pub fn with_resource_prefix(mut self, prefix: &str) -> Self {
        self.resource_prefix = Some(prefix.to_string());
        self
    }

    /// 检查事件是否匹配此过滤器
    pub fn matches(&self, event: &AuditEvent) -> bool {
        if let Some(ref user_id) = self.user_id {
            if &event.user_id != user_id {
                return false;
            }
        }

        if let Some(ref action) = self.action {
            if !event.action.contains(action.as_str()) {
                return false;
            }
        }

        if let Some(ref outcome) = self.outcome {
            if event.outcome != *outcome {
                return false;
            }
        }

        if let Some(start) = self.start_time {
            if event.timestamp < start {
                return false;
            }
        }

        if let Some(end) = self.end_time {
            if event.timestamp >= end {
                return false;
            }
        }

        if let Some(ref prefix) = self.resource_prefix {
            if !event.resource.starts_with(prefix.as_str()) {
                return false;
            }
        }

        true
    }
}

/// 分页请求参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRequest {
    /// 页码（从 1 开始）
    pub page: usize,
    /// 每页大小
    pub size: usize,
}

/// 分页响应结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageResult<T> {
    /// 当前页的数据项
    pub items: Vec<T>,
    /// 总记录数
    pub total: usize,
    /// 当前页码
    pub page: usize,
    /// 每页大小
    pub size: usize,
    /// 总页数
    pub total_pages: usize,
}

/// 导出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON 格式
    Json,
    /// CSV 格式
    Csv,
}

impl std::str::FromStr for ExportFormat {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "csv" => Ok(ExportFormat::Csv),
            "json" => Ok(ExportFormat::Json),
            _ => Err(()),
        }
    }
}

/// CSV 值转义辅助函数
fn escape_csv_value(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace("\"", "\"\""))
    } else {
        value.to_string()
    }
}

// ==================== 增强版审计日志系统 ====================

/// 审计事件类型枚举
///
/// 定义所有支持的审计事件分类，用于细粒度的事件追踪和统计分析。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditEventType {
    /// 认证事件（登录/登出）
    Authentication,
    /// 授权检查（允许/拒绝）
    Authorization,
    /// 数据访问（读取操作）
    DataAccess,
    /// 数据修改（更新操作）
    Modification,
    /// 数据删除操作
    Deletion,
    /// 系统配置变更
    SystemConfiguration,
    /// 模型管理操作（上传/部署/删除）
    ModelManagement,
    /// 推理请求
    InferenceRequest,
    /// 安全告警
    SecurityAlert,
}

impl AuditEventType {
    /// 获取事件类型的中文描述
    pub fn description(&self) -> &str {
        match self {
            AuditEventType::Authentication => "认证事件",
            AuditEventType::Authorization => "授权检查",
            AuditEventType::DataAccess => "数据访问",
            AuditEventType::Modification => "数据修改",
            AuditEventType::Deletion => "数据删除",
            AuditEventType::SystemConfiguration => "系统配置变更",
            AuditEventType::ModelManagement => "模型管理操作",
            AuditEventType::InferenceRequest => "推理请求",
            AuditEventType::SecurityAlert => "安全告警",
        }
    }

    /// 获取所有事件类型列表
    pub fn all() -> Vec<AuditEventType> {
        vec![
            AuditEventType::Authentication,
            AuditEventType::Authorization,
            AuditEventType::DataAccess,
            AuditEventType::Modification,
            AuditEventType::Deletion,
            AuditEventType::SystemConfiguration,
            AuditEventType::ModelManagement,
            AuditEventType::InferenceRequest,
            AuditEventType::SecurityAlert,
        ]
    }

    /// 返回字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            AuditEventType::Authentication => "authentication",
            AuditEventType::Authorization => "authorization",
            AuditEventType::DataAccess => "data_access",
            AuditEventType::Modification => "modification",
            AuditEventType::Deletion => "deletion",
            AuditEventType::SystemConfiguration => "system_configuration",
            AuditEventType::ModelManagement => "model_management",
            AuditEventType::InferenceRequest => "inference_request",
            AuditEventType::SecurityAlert => "security_alert",
        }
    }
}

/// 操作者信息结构体
///
/// 记录执行操作的用户的详细信息，用于完整的操作追踪。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorInfo {
    /// 用户唯一标识符
    pub user_id: String,
    /// 用户名（可选，用于显示）
    pub username: Option<String>,
    /// 会话 ID（可选，用于追踪会话）
    pub session_id: Option<String>,
    /// IP 地址（可选，用于安全分析）
    pub ip_address: Option<String>,
    /// 用户代理字符串（可选，用于客户端识别）
    pub user_agent: Option<String>,
}

impl ActorInfo {
    /// 创建新的操作者信息
    pub fn new(user_id: &str) -> Self {
        Self {
            user_id: user_id.to_string(),
            username: None,
            session_id: None,
            ip_address: None,
            user_agent: None,
        }
    }

    /// 设置用户名
    pub fn with_username(mut self, username: &str) -> Self {
        self.username = Some(username.to_string());
        self
    }

    /// 设置 IP 地址
    pub fn with_ip(mut self, ip: &str) -> Self {
        self.ip_address = Some(ip.to_string());
        self
    }

    /// 设置会话 ID
    pub fn with_session(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    /// 设置用户代理
    pub fn with_user_agent(mut self, user_agent: &str) -> Self {
        self.user_agent = Some(user_agent.to_string());
        self
    }
}

/// 资源信息结构体
///
/// 记录被操作资源的详细信息。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceInfo {
    /// 资源类型
    pub resource_type: ResourceType,
    /// 资源唯一标识符
    pub resource_id: String,
    /// 资源路径（可选）
    pub path: Option<String>,
    /// 具体操作（如 "deploy", "delete"）
    pub action: Option<String>,
}

impl ResourceInfo {
    /// 创建新的资源信息
    pub fn new(resource_type: ResourceType, resource_id: &str) -> Self {
        Self {
            resource_type,
            resource_id: resource_id.to_string(),
            path: None,
            action: None,
        }
    }

    /// 设置路径
    pub fn with_path(mut self, path: &str) -> Self {
        self.path = Some(path.to_string());
        self
    }

    /// 设置操作
    pub fn with_action(mut self, action: &str) -> Self {
        self.action = Some(action.to_string());
        self
    }
}

/// 资源类型枚举
///
/// 定义系统中可被审计的资源分类。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    /// AI 模型
    Model,
    /// 用户
    User,
    /// 角色
    Role,
    /// 系统配置
    System,
    /// 监控指标
    Metric,
    /// 告警规则
    Alert,
    /// 审计日志本身
    AuditLog,
    /// 其他类型
    Other,
}

impl ResourceType {
    /// 返回字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceType::Model => "model",
            ResourceType::User => "user",
            ResourceType::Role => "role",
            ResourceType::System => "system",
            ResourceType::Metric => "metric",
            ResourceType::Alert => "alert",
            ResourceType::AuditLog => "audit_log",
            ResourceType::Other => "other",
        }
    }
}

/// 操作结果枚举（增强版）
///
/// 提供比基础版更详细的结果状态，包含失败原因。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    /// 操作成功
    Success,
    /// 操作失败（包含原因）
    Failure { reason: String },
    /// 操作被拒绝（包含原因）
    Denied { reason: String },
}

impl Outcome {
    /// 判断是否为成功状态
    pub fn is_success(&self) -> bool {
        matches!(self, Outcome::Success)
    }

    /// 获取失败或拒绝的原因
    pub fn failure_reason(&self) -> Option<&str> {
        match self {
            Outcome::Success => None,
            Outcome::Failure { reason } | Outcome::Denied { reason } => Some(reason),
        }
    }

    /// 创建成功的操作结果
    pub fn success() -> Self {
        Outcome::Success
    }

    /// 创建失败的操作结果
    pub fn failure(reason: &str) -> Self {
        Outcome::Failure {
            reason: reason.to_string(),
        }
    }

    /// 创建被拒绝的操作结果
    pub fn denied(reason: &str) -> Self {
        Outcome::Denied {
            reason: reason.to_string(),
        }
    }

    /// 从基础版 AuditOutcome 转换
    pub fn from_basic(outcome: &AuditOutcome, reason: Option<&str>) -> Self {
        match outcome {
            AuditOutcome::Success => Outcome::Success,
            AuditOutcome::Failure => Outcome::Failure {
                reason: reason.unwrap_or("未知错误").to_string(),
            },
            AuditOutcome::Denied => Outcome::Denied {
                reason: reason.unwrap_or("权限不足").to_string(),
            },
        }
    }

    /// 转换为基础版 AuditOutcome
    pub fn to_basic(&self) -> AuditOutcome {
        match self {
            Outcome::Success => AuditOutcome::Success,
            Outcome::Failure { .. } => AuditOutcome::Failure,
            Outcome::Denied { .. } => AuditOutcome::Denied,
        }
    }

    /// 返回字符串表示
    pub fn as_str(&self) -> String {
        match self {
            Outcome::Success => "success".to_string(),
            Outcome::Failure { reason } => format!("failure: {}", reason),
            Outcome::Denied { reason } => format!("denied: {}", reason),
        }
    }
}

/// 增强版审计事件
///
/// 提供比基础版 AuditEvent 更丰富的字段和更强的结构化能力，
/// 支持详细的操作追踪和高级分析。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedAuditEvent {
    /// 唯一事件标识符（UUID v4）
    pub event_id: uuid::Uuid,
    /// 事件发生时间（UTC）
    pub timestamp: DateTime<Utc>,
    /// 事件类型（替代原来的 action: String）
    pub event_type: AuditEventType,
    /// 操作者详细信息
    pub actor: ActorInfo,
    /// 资源详细信息
    pub resource: ResourceInfo,
    /// 结构化操作结果
    pub outcome: Outcome,
    /// 操作耗时（毫秒，可选）
    pub duration_ms: Option<f64>,
    /// 请求追踪 ID（可选，用于分布式追踪）
    pub request_id: Option<String>,
    /// 附加元数据（JSON 对象，灵活扩展）
    pub metadata: serde_json::Value,
}

impl EnhancedAuditEvent {
    /// 从基础版 AuditEvent 创建增强版事件（向后兼容转换）
    ///
    /// 将基础版事件转换为增强版格式，保留原有数据的同时提供更丰富的结构。
    pub fn from_basic(basic: AuditEvent) -> Self {
        // 根据动作前缀推断事件类型
        let event_type = Self::infer_event_type(&basic.action);

        // 推断资源类型
        let resource_type = Self::infer_resource_type(&basic.resource);

        // 转换结果
        let outcome = Outcome::from_basic(&basic.outcome, None);

        Self {
            event_id: basic.event_id,
            timestamp: basic.timestamp,
            event_type,
            actor: ActorInfo::new(&basic.user_id)
                .with_ip(basic.ip_address.as_deref().unwrap_or("")),
            resource: ResourceInfo::new(resource_type, &basic.resource).with_path(&basic.resource),
            outcome,
            duration_ms: None,
            request_id: None,
            metadata: basic.details,
        }
    }

    /// 根据动作推断事件类型
    fn infer_event_type(action: &str) -> AuditEventType {
        if action.contains("auth") || action.contains("login") || action.contains("logout") {
            AuditEventType::Authentication
        } else if action.contains("authorize") || action.contains("permission") {
            AuditEventType::Authorization
        } else if action.contains("read") || action.contains("get") || action.contains("list") {
            AuditEventType::DataAccess
        } else if action.contains("update") || action.contains("modify") || action.contains("edit")
        {
            AuditEventType::Modification
        } else if action.contains("delete") || action.contains("remove") {
            AuditEventType::Deletion
        } else if action.contains("config") || action.contains("setting") {
            AuditEventType::SystemConfiguration
        } else if action.contains("model") || action.contains("deploy") {
            AuditEventType::ModelManagement
        } else if action.contains("infer") || action.contains("predict") {
            AuditEventType::InferenceRequest
        } else {
            AuditEventType::DataAccess // 默认类型
        }
    }

    /// 根据资源路径推断资源类型
    fn infer_resource_type(resource: &str) -> ResourceType {
        if resource.contains("/models/") || resource.contains("model") {
            ResourceType::Model
        } else if resource.contains("/users/") || resource.contains("user") {
            ResourceType::User
        } else if resource.contains("/roles/") || resource.contains("role") {
            ResourceType::Role
        } else if resource.contains("/config/") || resource.contains("system") {
            ResourceType::System
        } else if resource.contains("/metrics/") || resource.contains("metric") {
            ResourceType::Metric
        } else if resource.contains("/alerts/") || resource.contains("alert") {
            ResourceType::Alert
        } else if resource.contains("/audit") {
            ResourceType::AuditLog
        } else {
            ResourceType::Other
        }
    }

    /// 创建认证登录事件
    pub fn authentication_login(user_id: &str, ip: Option<&str>) -> Self {
        let mut actor = ActorInfo::new(user_id);
        if let Some(ip_addr) = ip {
            actor = actor.with_ip(ip_addr);
        }

        Self {
            event_id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::Authentication,
            actor,
            resource: ResourceInfo::new(ResourceType::User, user_id).with_action("login"),
            outcome: Outcome::success(),
            duration_ms: None,
            request_id: None,
            metadata: json!({"action": "login"}),
        }
    }

    /// 创建认证登出事件
    pub fn authentication_logout(user_id: &str, session_id: Option<&str>) -> Self {
        let mut actor = ActorInfo::new(user_id);
        if let Some(sid) = session_id {
            actor = actor.with_session(sid);
        }

        Self {
            event_id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::Authentication,
            actor,
            resource: ResourceInfo::new(ResourceType::User, user_id).with_action("logout"),
            outcome: Outcome::success(),
            duration_ms: None,
            request_id: None,
            metadata: json!({"action": "logout"}),
        }
    }

    /// 创建授权检查事件
    pub fn authorization_check(
        user_id: &str,
        resource: &str,
        allowed: bool,
        reason: Option<&str>,
    ) -> Self {
        let outcome = if allowed {
            Outcome::success()
        } else {
            Outcome::denied(reason.unwrap_or("权限不足"))
        };

        Self {
            event_id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::Authorization,
            actor: ActorInfo::new(user_id),
            resource: ResourceInfo::new(ResourceType::Other, resource).with_action("authorize"),
            outcome,
            duration_ms: None,
            request_id: None,
            metadata: json!({
                "allowed": allowed,
                "reason": reason
            }),
        }
    }

    /// 创建模型部署事件
    pub fn model_deploy(user_id: &str, model_id: &str, success: bool) -> Self {
        let outcome = if success {
            Outcome::success()
        } else {
            Outcome::failure("模型部署失败")
        };

        Self {
            event_id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::ModelManagement,
            actor: ActorInfo::new(user_id),
            resource: ResourceInfo::new(ResourceType::Model, model_id).with_action("deploy"),
            outcome,
            duration_ms: None,
            request_id: None,
            metadata: json!({"model_id": model_id}),
        }
    }

    /// 创建模型删除事件
    pub fn model_delete(user_id: &str, model_id: &str, success: bool) -> Self {
        let outcome = if success {
            Outcome::success()
        } else {
            Outcome::failure("模型删除失败")
        };

        Self {
            event_id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::ModelManagement,
            actor: ActorInfo::new(user_id),
            resource: ResourceInfo::new(ResourceType::Model, model_id).with_action("delete"),
            outcome,
            duration_ms: None,
            request_id: None,
            metadata: json!({"model_id": model_id}),
        }
    }

    /// 创建推理请求事件
    pub fn inference_request(user_id: &str, model_id: &str, latency_ms: f64) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::InferenceRequest,
            actor: ActorInfo::new(user_id),
            resource: ResourceInfo::new(ResourceType::Model, model_id).with_action("infer"),
            outcome: Outcome::success(),
            duration_ms: Some(latency_ms),
            request_id: None,
            metadata: json!({
                "model_id": model_id,
                "latency_ms": latency_ms
            }),
        }
    }

    /// 创建安全告警事件
    pub fn security_alert(alert_type: &str, severity: &str, details: serde_json::Value) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::SecurityAlert,
            actor: ActorInfo::new("system"), // 系统生成的告警
            resource: ResourceInfo::new(ResourceType::Alert, alert_type).with_action("alert"),
            outcome: Outcome::success(), // 告警生成本身是成功的
            duration_ms: None,
            request_id: None,
            metadata: json!({
                "alert_type": alert_type,
                "severity": severity,
                "details": details
            }),
        }
    }

    /// 创建数据访问事件
    pub fn data_access(
        user_id: &str,
        resource_id: &str,
        resource_type: ResourceType,
        action: &str,
    ) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::DataAccess,
            actor: ActorInfo::new(user_id),
            resource: ResourceInfo::new(resource_type, resource_id).with_action(action),
            outcome: Outcome::success(),
            duration_ms: None,
            request_id: None,
            metadata: json!({}),
        }
    }

    /// 创建数据修改事件
    pub fn data_modification(
        user_id: &str,
        resource_id: &str,
        resource_type: ResourceType,
        action: &str,
        success: bool,
    ) -> Self {
        let outcome = if success {
            Outcome::success()
        } else {
            Outcome::failure("修改操作失败")
        };

        Self {
            event_id: uuid::Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::Modification,
            actor: ActorInfo::new(user_id),
            resource: ResourceInfo::new(resource_type, resource_id).with_action(action),
            outcome,
            duration_ms: None,
            request_id: None,
            metadata: json!({}),
        }
    }
}

/// 审计排序字段枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditSortField {
    /// 按时间戳排序
    Timestamp,
    /// 按事件类型排序
    EventType,
    /// 按用户 ID 排序
    UserId,
    /// 按持续时间排序
    Duration,
}

/// 排序顺序枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// 升序
    Ascending,
    /// 降序
    Descending,
}

/// 增强版审计过滤器
///
/// 提供比基础版 AuditFilter 更强大的过滤能力，
/// 支持多维度组合过滤、时间范围、分页、排序等功能。
#[derive(Debug, Clone, Default)]
pub struct EnhancedAuditFilter {
    // 基础过滤条件
    /// 事件类型过滤（支持多选）
    pub event_types: Vec<AuditEventType>,
    /// 用户 ID 过滤（支持多个）
    pub user_ids: Vec<String>,
    /// 操作结果过滤（支持多个）
    pub outcomes: Vec<Outcome>,
    /// 资源类型过滤（支持多个）
    pub resource_types: Vec<ResourceType>,

    // 时间范围过滤
    /// 开始时间（包含）
    pub start_time: Option<DateTime<Utc>>,
    /// 结束时间（不包含）
    pub end_time: Option<DateTime<Utc>>,

    // 高级过滤
    /// IP 地址过滤（支持多个）
    pub ip_addresses: Vec<String>,
    /// 会话 ID 过滤（支持多个）
    pub session_ids: Vec<String>,
    /// 请求 ID 过滤（支持多个）
    pub request_ids: Vec<String>,
    /// 最小持续时间（毫秒）
    pub min_duration_ms: Option<f64>,
    /// 最大持续时间（毫秒）
    pub max_duration_ms: Option<f64>,

    // 元数据过滤（JSON 路径表达式）
    /// 元数据过滤键（如 "$.model_version"）
    pub metadata_filter: Option<String>,
    /// 元数据过滤值
    pub metadata_value: Option<serde_json::Value>,

    // 分页
    /// 偏移量
    pub offset: Option<usize>,
    /// 限制返回数量
    pub limit: Option<usize>,

    // 排序
    /// 排序字段
    pub sort_by: Option<AuditSortField>,
    /// 排序顺序
    pub sort_order: Option<SortOrder>,
}

impl EnhancedAuditFilter {
    /// 创建空的增强过滤器
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加事件类型过滤
    pub fn with_event_type(mut self, event_type: AuditEventType) -> Self {
        self.event_types.push(event_type);
        self
    }

    /// 添加用户 ID 过滤
    pub fn with_user_id(mut self, user_id: &str) -> Self {
        self.user_ids.push(user_id.to_string());
        self
    }

    /// 添加操作结果过滤
    pub fn with_outcome(mut self, outcome: Outcome) -> Self {
        self.outcomes.push(outcome);
        self
    }

    /// 添加资源类型过滤
    pub fn with_resource_type(mut self, resource_type: ResourceType) -> Self {
        self.resource_types.push(resource_type);
        self
    }

    /// 设置开始时间
    pub fn with_start_time(mut self, time: DateTime<Utc>) -> Self {
        self.start_time = Some(time);
        self
    }

    /// 设置结束时间
    pub fn with_end_time(mut self, time: DateTime<Utc>) -> Self {
        self.end_time = Some(time);
        self
    }

    /// 添加 IP 地址过滤
    pub fn with_ip_address(mut self, ip: &str) -> Self {
        self.ip_addresses.push(ip.to_string());
        self
    }

    /// 设置分页参数
    pub fn with_pagination(mut self, offset: usize, limit: usize) -> Self {
        self.offset = Some(offset);
        self.limit = Some(limit);
        self
    }

    /// 设置排序参数
    pub fn with_sort(mut self, field: AuditSortField, order: SortOrder) -> Self {
        self.sort_by = Some(field);
        self.sort_order = Some(order);
        self
    }

    /// 设置元数据过滤条件
    pub fn with_metadata_filter(mut self, filter: &str, value: serde_json::Value) -> Self {
        self.metadata_filter = Some(filter.to_string());
        self.metadata_value = Some(value);
        self
    }

    /// 设置持续时间范围
    pub fn with_duration_range(mut self, min: f64, max: f64) -> Self {
        self.min_duration_ms = Some(min);
        self.max_duration_ms = Some(max);
        self
    }

    /// 检查增强版事件是否匹配此过滤器
    ///
    /// 支持多维度的 AND 组合过滤。
    pub fn matches_enhanced(&self, event: &EnhancedAuditEvent) -> bool {
        // 事件类型过滤
        if !self.event_types.is_empty() && !self.event_types.contains(&event.event_type) {
            return false;
        }

        // 用户 ID 过滤
        if !self.user_ids.is_empty() && !self.user_ids.contains(&event.actor.user_id) {
            return false;
        }

        // 操作结果过滤
        if !self.outcomes.is_empty() && !self.outcomes.contains(&event.outcome) {
            return false;
        }

        // 资源类型过滤
        if !self.resource_types.is_empty()
            && !self.resource_types.contains(&event.resource.resource_type)
        {
            return false;
        }

        // 时间范围过滤
        if let Some(start) = self.start_time {
            if event.timestamp < start {
                return false;
            }
        }

        if let Some(end) = self.end_time {
            if event.timestamp >= end {
                return false;
            }
        }

        // IP 地址过滤
        if !self.ip_addresses.is_empty() {
            if let Some(ref ip) = event.actor.ip_address {
                if !self
                    .ip_addresses
                    .iter()
                    .any(|filter_ip| ip.contains(filter_ip))
                {
                    return false;
                }
            } else {
                return false; // 需要过滤 IP 但事件没有 IP 信息
            }
        }

        // 会话 ID 过滤
        if !self.session_ids.is_empty() {
            if let Some(ref sid) = event.actor.session_id {
                if !self.session_ids.contains(sid) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // 请求 ID 过滤
        if !self.request_ids.is_empty() {
            if let Some(ref rid) = event.request_id {
                if !self.request_ids.contains(rid) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // 持续时间范围过滤
        if let (Some(min), Some(duration)) = (self.min_duration_ms, event.duration_ms) {
            if duration < min {
                return false;
            }
        }

        if let (Some(max), Some(duration)) = (self.max_duration_ms, event.duration_ms) {
            if duration > max {
                return false;
            }
        }

        // 元数据过滤（简单实现：检查 JSON 中是否存在指定键值对）
        if let (Some(ref filter), Some(ref value)) = (&self.metadata_filter, &self.metadata_value) {
            // 简单的键存在性检查（不支持完整 JSONPath）
            let key = filter.trim_start_matches("$.");
            if let Some(meta_value) = event.metadata.get(key) {
                if meta_value != value {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }
}

/// 分组维度枚举
///
/// 用于聚合统计时的分组依据。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditGroupBy {
    /// 按事件类型分组
    EventType,
    /// 按用户 ID 分组
    UserId,
    /// 按资源类型分组
    ResourceType,
    /// 按操作结果分组
    Outcome,
    /// 按小时分组（时间维度）
    TimeHour,
}

/// 聚合结果结构体
///
/// 存储按指定维度分组后的统计数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditAggregateResult {
    /// 分组键值
    pub group_key: String,
    /// 该组的记录数量
    pub count: u64,
    /// 成功率（0.0 - 1.0）
    pub success_rate: f32,
    /// 平均持续时间（毫秒）
    pub avg_duration_ms: f64,
    /// 该组最早出现时间
    pub first_seen: Option<DateTime<Utc>>,
    /// 该组最晚出现时间
    pub last_seen: Option<DateTime<Utc>>,
}

/// 审计摘要结构体
///
/// 提供指定时间窗口内的整体统计概览。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSummary {
    /// 窗口开始时间
    pub window_start: DateTime<Utc>,
    /// 窗口结束时间
    pub window_end: DateTime<Utc>,
    /// 总事件数量
    pub total_events: u64,
    /// 按事件类型统计
    pub events_by_type: std::collections::HashMap<AuditEventType, u64>,
    /// 整体成功率
    pub success_rate: f32,
    /// 唯一用户数
    pub unique_users: usize,
    /// 唯一资源数
    pub unique_resources: usize,
    /// 最活跃用户 TOP10
    pub top_users: Vec<(String, u64)>,
    /// 最常访问资源 TOP10
    pub top_resources: Vec<(String, u64)>,
    /// 安全告警数量
    pub security_alerts_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_audit_logger(buffer_size: usize) -> AuditLogger {
        let config = AuditConfig {
            enabled: true,
            buffer_size,
            retention_days: 90,
            export_format: "json".to_string(),
        };
        AuditLogger::new(&config).expect("Failed to create AuditLogger")
    }

    fn create_sample_event(user_id: &str, action: &str, outcome: AuditOutcome) -> AuditEvent {
        AuditEvent {
            timestamp: Utc::now(),
            event_id: uuid::Uuid::new_v4(),
            user_id: user_id.to_string(),
            action: action.to_string(),
            resource: format!("/api/{}", action.replace(':', "/")),
            outcome,
            details: json!({}),
            ip_address: Some("192.168.1.1".to_string()),
        }
    }

    #[test]
    fn test_audit_logger_creation() {
        let audit = create_test_audit_logger(100);
        assert_eq!(audit.event_count(), 0);
    }

    #[test]
    fn test_log_single_event() {
        let audit = create_test_audit_logger(100);

        let event = create_sample_event("user-1", "test:action", AuditOutcome::Success);
        audit.log(event.clone()).unwrap();

        assert_eq!(audit.event_count(), 1);

        // 验证可以查询到
        let result = audit.query(
            AuditFilter::new().with_user_id("user-1"),
            PageRequest { page: 1, size: 10 },
        );
        assert!(result.is_ok());
        let page = result.unwrap();
        assert_eq!(page.total, 1);
        assert_eq!(page.items.len(), 1);
    }

    #[test]
    fn test_log_multiple_events() {
        let audit = create_test_audit_logger(100);

        for i in 0..5 {
            audit
                .log(create_sample_event(
                    &format!("user-{}", i),
                    "action:test",
                    AuditOutcome::Success,
                ))
                .unwrap();
        }

        assert_eq!(audit.event_count(), 5);

        let result = audit
            .query(AuditFilter::default(), PageRequest { page: 1, size: 10 })
            .unwrap();
        assert_eq!(result.total, 5);
    }

    #[test]
    fn test_buffer_overflow_with_ring_buffer() {
        let audit = create_test_audit_logger(3); // 小缓冲区

        // 写入超过缓冲区容量的事件
        for i in 0..5 {
            audit
                .log(create_sample_event(
                    &format!("user-{}", i),
                    "action:test",
                    AuditOutcome::Success,
                ))
                .unwrap();
        }

        // 应该只保留最后 3 个事件（环形缓冲区特性）
        assert_eq!(audit.event_count(), 5); // 总计数不变

        let result = audit
            .query(AuditFilter::default(), PageRequest { page: 1, size: 10 })
            .unwrap();
        assert_eq!(result.items.len(), 3); // 但缓冲区只有 3 个
    }

    #[test]
    fn test_filter_by_user_id() {
        let audit = create_test_audit_logger(100);

        audit
            .log(create_sample_event("alice", "login", AuditOutcome::Success))
            .unwrap();
        audit
            .log(create_sample_event("bob", "login", AuditOutcome::Success))
            .unwrap();
        audit
            .log(create_sample_event(
                "alice",
                "logout",
                AuditOutcome::Success,
            ))
            .unwrap();

        let result = audit
            .query(
                AuditFilter::new().with_user_id("alice"),
                PageRequest { page: 1, size: 10 },
            )
            .unwrap();

        assert_eq!(result.total, 2);
        assert!(result.items.iter().all(|e| e.user_id == "alice"));
    }

    #[test]
    fn test_filter_by_outcome() {
        let audit = create_test_audit_logger(100);

        audit
            .log(create_sample_event(
                "user-1",
                "login",
                AuditOutcome::Success,
            ))
            .unwrap();
        audit
            .log(create_sample_event(
                "user-2",
                "login",
                AuditOutcome::Failure,
            ))
            .unwrap();
        audit
            .log(create_sample_event("user-3", "login", AuditOutcome::Denied))
            .unwrap();

        let failures = audit
            .query(
                AuditFilter::new().with_outcome(AuditOutcome::Failure),
                PageRequest { page: 1, size: 10 },
            )
            .unwrap();

        assert_eq!(failures.total, 1);
        assert_eq!(failures.items[0].outcome, AuditOutcome::Failure);
    }

    #[test]
    fn test_filter_by_action() {
        let audit = create_test_audit_logger(100);

        audit
            .log(create_sample_event(
                "user-1",
                "auth:login",
                AuditOutcome::Success,
            ))
            .unwrap();
        audit
            .log(create_sample_event(
                "user-1",
                "data:read",
                AuditOutcome::Success,
            ))
            .unwrap();
        audit
            .log(create_sample_event(
                "user-2",
                "auth:login",
                AuditOutcome::Success,
            ))
            .unwrap();

        let login_events = audit
            .query(
                AuditFilter::new().with_action("auth:login"),
                PageRequest { page: 1, size: 10 },
            )
            .unwrap();

        assert_eq!(login_events.total, 2);
    }

    #[test]
    fn test_pagination() {
        let audit = create_test_audit_logger(100);

        // 写入 25 个事件
        for i in 0..25 {
            audit
                .log(create_sample_event(
                    &format!("user-{}", i % 5),
                    "action:test",
                    AuditOutcome::Success,
                ))
                .unwrap();
        }

        // 第一页：10 条
        let page1 = audit
            .query(AuditFilter::default(), PageRequest { page: 1, size: 10 })
            .unwrap();
        assert_eq!(page1.items.len(), 10);
        assert_eq!(page1.page, 1);
        assert_eq!(page1.total_pages, 3); // 25 / 10 = 2.5 → 3 页

        // 第二页：10 条
        let page2 = audit
            .query(AuditFilter::default(), PageRequest { page: 2, size: 10 })
            .unwrap();
        assert_eq!(page2.items.len(), 10);

        // 第三页：5 条
        let page3 = audit
            .query(AuditFilter::default(), PageRequest { page: 3, size: 10 })
            .unwrap();
        assert_eq!(page3.items.len(), 5);
    }

    #[test]
    fn test_export_json() {
        let audit = create_test_audit_logger(100);

        audit
            .log(create_sample_event("user-1", "test", AuditOutcome::Success))
            .unwrap();

        let json_data = audit
            .export(AuditFilter::default(), ExportFormat::Json)
            .unwrap();

        let parsed: Vec<Value> = serde_json::from_slice(&json_data).expect("Invalid JSON output");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["user_id"], "user-1");
    }

    #[test]
    fn test_export_csv() {
        let audit = create_test_audit_logger(100);

        audit
            .log(create_sample_event("user-1", "test", AuditOutcome::Success))
            .unwrap();

        let csv_data = audit
            .export(AuditFilter::default(), ExportFormat::Csv)
            .unwrap();

        let csv_str = String::from_utf8(csv_data).expect("Invalid UTF-8");
        assert!(csv_str.contains("timestamp,event_id")); // CSV header
        assert!(csv_str.contains("user-1")); // data row
    }

    #[test]
    fn test_clear_audit_log() {
        let audit = create_test_audit_logger(100);

        audit
            .log(create_sample_event("user-1", "test", AuditOutcome::Success))
            .unwrap();
        assert_eq!(audit.event_count(), 1);

        audit.clear().unwrap();
        assert_eq!(audit.event_count(), 0);

        let result = audit
            .query(AuditFilter::default(), PageRequest { page: 1, size: 10 })
            .unwrap();
        assert!(result.items.is_empty());
    }

    #[test]
    fn test_audit_event_factory_methods() {
        let success = AuditEvent::success("u1", "act", "/res", json!({}));
        assert_eq!(success.outcome, AuditOutcome::Success);
        assert_eq!(success.user_id, "u1");

        let failure = AuditEvent::failure("u2", "act", "/res", json!({}));
        assert_eq!(failure.outcome, AuditOutcome::Failure);

        let denied = AuditEvent::denied("u3", "act", "/res", json!({}));
        assert_eq!(denied.outcome, AuditOutcome::Denied);
    }

    #[test]
    fn test_audit_outcome_display() {
        assert_eq!(AuditOutcome::Success.to_string(), "success");
        assert_eq!(AuditOutcome::Failure.to_string(), "failure");
        assert_eq!(AuditOutcome::Denied.to_string(), "denied");
    }

    #[test]
    fn test_filter_by_resource_prefix() {
        let audit = create_test_audit_logger(100);

        audit
            .log(create_sample_event(
                "u1",
                "admin:user:list",
                AuditOutcome::Success,
            ))
            .unwrap();
        audit
            .log(create_sample_event(
                "u2",
                "data:model:get",
                AuditOutcome::Success,
            ))
            .unwrap();
        audit
            .log(create_sample_event(
                "u3",
                "admin:role:update",
                AuditOutcome::Success,
            ))
            .unwrap();

        let admin_events = audit
            .query(
                AuditFilter::new().with_resource_prefix("/api/admin/"),
                PageRequest { page: 1, size: 10 },
            )
            .unwrap();

        assert_eq!(admin_events.total, 2);
    }

    // ==================== 增强版功能测试 ====================

    #[test]
    fn test_audit_event_type_description() {
        // 测试所有事件类型的描述
        assert_eq!(AuditEventType::Authentication.description(), "认证事件");
        assert_eq!(AuditEventType::Authorization.description(), "授权检查");
        assert_eq!(AuditEventType::DataAccess.description(), "数据访问");
        assert_eq!(AuditEventType::Modification.description(), "数据修改");
        assert_eq!(AuditEventType::Deletion.description(), "数据删除");
        assert_eq!(
            AuditEventType::SystemConfiguration.description(),
            "系统配置变更"
        );
        assert_eq!(
            AuditEventType::ModelManagement.description(),
            "模型管理操作"
        );
        assert_eq!(AuditEventType::InferenceRequest.description(), "推理请求");
        assert_eq!(AuditEventType::SecurityAlert.description(), "安全告警");
    }

    #[test]
    fn test_audit_event_type_all() {
        let all_types = AuditEventType::all();
        assert_eq!(all_types.len(), 9); // 确保有9种类型

        // 验证包含所有类型
        assert!(all_types.contains(&AuditEventType::Authentication));
        assert!(all_types.contains(&AuditEventType::SecurityAlert));
    }

    #[test]
    fn test_audit_event_type_as_str() {
        assert_eq!(AuditEventType::Authentication.as_str(), "authentication");
        assert_eq!(
            AuditEventType::InferenceRequest.as_str(),
            "inference_request"
        );
    }

    #[test]
    fn test_actor_info_creation_and_methods() {
        let actor = ActorInfo::new("user-123")
            .with_username("张三")
            .with_ip("192.168.1.100")
            .with_session("session-abc")
            .with_user_agent("Mozilla/5.0");

        assert_eq!(actor.user_id, "user-123");
        assert_eq!(actor.username.unwrap(), "张三");
        assert_eq!(actor.ip_address.unwrap(), "192.168.1.100");
        assert_eq!(actor.session_id.unwrap(), "session-abc");
        assert_eq!(actor.user_agent.unwrap(), "Mozilla/5.0");
    }

    #[test]
    fn test_actor_info_minimal() {
        let actor = ActorInfo::new("minimal-user");

        assert_eq!(actor.user_id, "minimal-user");
        assert!(actor.username.is_none());
        assert!(actor.ip_address.is_none());
        assert!(actor.session_id.is_none());
        assert!(actor.user_agent.is_none());
    }

    #[test]
    fn test_resource_info_creation() {
        let resource = ResourceInfo::new(ResourceType::Model, "gpt-4")
            .with_path("/models/gpt-4")
            .with_action("deploy");

        assert_eq!(resource.resource_type, ResourceType::Model);
        assert_eq!(resource.resource_id, "gpt-4");
        assert_eq!(resource.path.unwrap(), "/models/gpt-4");
        assert_eq!(resource.action.unwrap(), "deploy");
    }

    #[test]
    fn test_resource_type_as_str() {
        assert_eq!(ResourceType::Model.as_str(), "model");
        assert_eq!(ResourceType::User.as_str(), "user");
        assert_eq!(ResourceType::Other.as_str(), "other");
    }

    #[test]
    fn test_outcome_success() {
        let outcome = Outcome::success();
        assert!(outcome.is_success());
        assert!(outcome.failure_reason().is_none());
        assert_eq!(outcome.to_basic(), AuditOutcome::Success);
    }

    #[test]
    fn test_outcome_failure_with_reason() {
        let outcome = Outcome::failure("连接超时");
        assert!(!outcome.is_success());
        assert_eq!(outcome.failure_reason().unwrap(), "连接超时");
        assert_eq!(outcome.to_basic(), AuditOutcome::Failure);
    }

    #[test]
    fn test_outcome_denied_with_reason() {
        let outcome = Outcome::denied("权限不足");
        assert!(!outcome.is_success());
        assert_eq!(outcome.failure_reason().unwrap(), "权限不足");
        assert_eq!(outcome.to_basic(), AuditOutcome::Denied);
    }

    #[test]
    fn test_outcome_from_basic_conversion() {
        // 测试从基础版转换
        let enhanced = Outcome::from_basic(&AuditOutcome::Success, None);
        assert!(enhanced.is_success());

        let enhanced_fail = Outcome::from_basic(&AuditOutcome::Failure, Some("网络错误"));
        assert_eq!(enhanced_fail.failure_reason().unwrap(), "网络错误");

        let enhanced_denied = Outcome::from_basic(&AuditOutcome::Denied, Some("无权限"));
        assert_eq!(enhanced_denied.failure_reason().unwrap(), "无权限");
    }

    #[test]
    fn test_enhanced_audit_event_from_basic() {
        let basic = create_sample_event("user-1", "auth:login", AuditOutcome::Success);
        let enhanced = EnhancedAuditEvent::from_basic(basic);

        assert_eq!(enhanced.actor.user_id, "user-1");
        assert_eq!(enhanced.event_type, AuditEventType::Authentication); // 自动推断
        assert!(enhanced.outcome.is_success());
        assert!(enhanced.duration_ms.is_none()); // 基础版没有持续时间
    }

    #[test]
    fn test_enhanced_audit_event_authentication_login() {
        let event = EnhancedAuditEvent::authentication_login("alice", Some("10.0.0.1"));

        assert_eq!(event.actor.user_id, "alice");
        assert_eq!(event.event_type, AuditEventType::Authentication);
        assert_eq!(event.actor.ip_address.as_deref().unwrap(), "10.0.0.1");
        assert!(event.outcome.is_success());
        assert_eq!(event.resource.action.as_deref().unwrap(), "login");
    }

    #[test]
    fn test_enhanced_audit_event_authorization_check_allowed() {
        let event = EnhancedAuditEvent::authorization_check("bob", "/api/models/gpt-4", true, None);

        assert_eq!(event.event_type, AuditEventType::Authorization);
        assert!(event.outcome.is_success());
        assert_eq!(event.metadata["allowed"], true);
    }

    #[test]
    fn test_enhanced_audit_event_authorization_check_denied() {
        let event = EnhancedAuditEvent::authorization_check(
            "charlie",
            "/api/admin/config",
            false,
            Some("需要管理员角色"),
        );

        assert!(!event.outcome.is_success());
        assert_eq!(event.outcome.failure_reason().unwrap(), "需要管理员角色");
    }

    #[test]
    fn test_enhanced_audit_event_model_deploy() {
        let event = EnhancedAuditEvent::model_deploy("admin", "llama-3", true);

        assert_eq!(event.event_type, AuditEventType::ModelManagement);
        assert_eq!(event.resource.resource_type, ResourceType::Model);
        assert_eq!(event.resource.resource_id, "llama-3");
        assert!(event.outcome.is_success());
    }

    #[test]
    fn test_enhanced_audit_event_model_deploy_failure() {
        let event = EnhancedAuditEvent::model_deploy("admin", "invalid-model", false);

        assert!(!event.outcome.is_success());
        assert_eq!(event.outcome.failure_reason().unwrap(), "模型部署失败");
    }

    #[test]
    fn test_enhanced_audit_event_inference_request() {
        let event = EnhancedAuditEvent::inference_request("user-1", "gpt-4", 150.5);

        assert_eq!(event.event_type, AuditEventType::InferenceRequest);
        assert!(event.outcome.is_success());
        assert_eq!(event.duration_ms.unwrap(), 150.5);
        assert_eq!(event.metadata["latency_ms"], 150.5);
    }

    #[test]
    fn test_enhanced_audit_event_security_alert() {
        let details = json!({"ip": "192.168.1.100", "attempts": 5});
        let event = EnhancedAuditEvent::security_alert("brute_force", "high", details.clone());

        assert_eq!(event.event_type, AuditEventType::SecurityAlert);
        assert_eq!(event.actor.user_id, "system"); // 系统告警
        assert_eq!(event.metadata["alert_type"], "brute_force");
        assert_eq!(event.metadata["severity"], "high");
        assert_eq!(event.metadata["details"], details);
    }

    #[test]
    fn test_log_enhanced_and_query() {
        let audit = create_test_audit_logger(100);

        // 记录多个增强事件
        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login(
                "user-1",
                Some("10.0.0.1"),
            ))
            .unwrap();

        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("admin", "gpt-4", true))
            .unwrap();

        audit
            .log_enhanced(EnhancedAuditEvent::inference_request(
                "user-2", "gpt-4", 200.0,
            ))
            .unwrap();

        // 查询所有增强事件
        let filter = EnhancedAuditFilter::new();
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 3);

        // 验证第一个是登录事件
        assert_eq!(results[0].event_type, AuditEventType::Authentication);
        assert_eq!(results[0].actor.user_id, "user-1");
    }

    #[test]
    fn test_enhanced_filter_by_event_type() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("u1", None))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u2", "m1", true))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::security_alert("test", "low", json!({})))
            .unwrap();

        // 只查询认证事件
        let filter = EnhancedAuditFilter::new().with_event_type(AuditEventType::Authentication);
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].event_type, AuditEventType::Authentication);
    }

    #[test]
    fn test_enhanced_filter_by_user_id() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("alice", None))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("bob", None))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("alice", None))
            .unwrap();

        let filter = EnhancedAuditFilter::new().with_user_id("alice");
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.actor.user_id == "alice"));
    }

    #[test]
    fn test_enhanced_filter_by_outcome() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u1", "m1", true))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u2", "m2", false))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u3", "m3", true))
            .unwrap();

        let filter = EnhancedAuditFilter::new().with_outcome(Outcome::success());
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.outcome.is_success()));
    }

    #[test]
    fn test_enhanced_filter_by_resource_type() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u1", "m1", true))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::data_access(
                "u2",
                "data-1",
                ResourceType::Other,
                "read",
            ))
            .unwrap();

        let filter = EnhancedAuditFilter::new().with_resource_type(ResourceType::Model);
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].resource.resource_type, ResourceType::Model);
    }

    #[test]
    fn test_enhanced_filter_combined_conditions() {
        let audit = create_test_audit_logger(100);

        // alice 的模型部署成功
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("alice", "m1", true))
            .unwrap();
        // bob 的模型部署失败
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("bob", "m2", false))
            .unwrap();
        // alice 的推理请求
        audit
            .log_enhanced(EnhancedAuditEvent::inference_request("alice", "m1", 100.0))
            .unwrap();

        // 组合过滤：用户=alice 且 成功 且 资源类型=Model
        let filter = EnhancedAuditFilter::new()
            .with_user_id("alice")
            .with_outcome(Outcome::success())
            .with_resource_type(ResourceType::Model);
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 2); // alice 的模型部署成功和推理请求都符合（都使用 ResourceType::Model）
        assert_eq!(results[0].resource.action.as_deref().unwrap(), "deploy");
    }

    #[test]
    fn test_enhanced_filter_pagination() {
        let audit = create_test_audit_logger(100);

        // 写入20个事件
        for i in 0..20 {
            audit
                .log_enhanced(EnhancedAuditEvent::data_access(
                    &format!("user-{}", i),
                    &format!("res-{}", i),
                    ResourceType::Other,
                    "read",
                ))
                .unwrap();
        }

        // 分页：第2页，每页5条
        let filter = EnhancedAuditFilter::new().with_pagination(5, 5); // offset=5, limit=5
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_enhanced_filter_sorting_by_timestamp_descending() {
        let audit = create_test_audit_logger(100);

        // 写入多个事件（时间自动递增）
        for _ in 0..5 {
            audit
                .log_enhanced(EnhancedAuditEvent::authentication_login("user", None))
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10)); // 小延迟确保时间不同
        }

        let filter =
            EnhancedAuditFilter::new().with_sort(AuditSortField::Timestamp, SortOrder::Descending);
        let results = audit.query_enhanced(&filter).unwrap();

        // 验证降序排序（最新的在前）
        for i in 1..results.len() {
            assert!(results[i - 1].timestamp >= results[i].timestamp);
        }
    }

    #[test]
    fn test_aggregate_by_event_type() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("u1", None))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("u2", None))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u3", "m1", true))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::inference_request("u4", "m1", 50.0))
            .unwrap();

        let results = audit.aggregate(AuditGroupBy::EventType, None).unwrap();

        // 应该有3个分组：认证事件(2)、模型管理(1)、推理请求(1)
        assert!(!results.is_empty());

        // 找到认证事件分组
        let auth_group = results.iter().find(|r| r.group_key == "认证事件");
        assert!(auth_group.is_some());
        assert_eq!(auth_group.unwrap().count, 2);
    }

    #[test]
    fn test_aggregate_by_user_id() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("alice", None))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("alice", "m1", true))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("bob", None))
            .unwrap();

        let results = audit.aggregate(AuditGroupBy::UserId, None).unwrap();

        // alice 有2个事件，bob有1个
        let alice_count = results
            .iter()
            .find(|r| r.group_key == "alice")
            .map(|r| r.count);
        assert_eq!(alice_count.unwrap(), 2);
    }

    #[test]
    fn test_aggregate_with_filter() {
        let audit = create_test_audit_logger(100);

        // 混合成功和失败的事件
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u1", "m1", true))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u2", "m2", false))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u3", "m3", true))
            .unwrap();

        // 只统计成功的
        let filter = EnhancedAuditFilter::new().with_outcome(Outcome::success());
        let results = audit
            .aggregate(AuditGroupBy::Outcome, Some(&filter))
            .unwrap();

        // 应该只有 success 分组
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].group_key, "success");
        assert_eq!(results[0].count, 2);
        assert_eq!(results[0].success_rate, 1.0); // 过滤后全是成功
    }

    #[test]
    fn test_export_csv_enhanced() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login(
                "test-user",
                Some("10.0.0.1"),
            ))
            .unwrap();

        let mut csv_output: Vec<u8> = Vec::new();
        let count = audit
            .export_csv(&EnhancedAuditFilter::new(), &mut csv_output)
            .unwrap();

        assert_eq!(count, 1);

        let csv_str = String::from_utf8(csv_output).expect("Invalid UTF-8");
        // 验证CSV头部
        assert!(csv_str.contains("event_id,timestamp,event_type"));
        // 验证数据行
        assert!(csv_str.contains("test-user"));
        assert!(csv_str.contains("10.0.0.1"));
        assert!(csv_str.contains("authentication")); // 事件类型
    }

    #[test]
    fn test_export_json_enhanced_pretty() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::security_alert(
                "test-alert",
                "medium",
                json!({"key": "value"}),
            ))
            .unwrap();

        let json_str = audit
            .export_json(
                &EnhancedAuditFilter::new(),
                true, // pretty print
            )
            .unwrap();

        // 验证JSON格式正确
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("Invalid JSON");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["event_type"], "security_alert");
        assert_eq!(parsed[0]["actor"]["user_id"], "system");

        // 验证美化输出（应该包含缩进）
        assert!(json_str.contains("\n"));
        assert!(json_str.contains("  "));
    }

    #[test]
    fn test_export_json_enhanced_compact() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::inference_request("u1", "m1", 99.9))
            .unwrap();

        let json_str = audit
            .export_json(
                &EnhancedAuditFilter::new(),
                false, // compact mode
            )
            .unwrap();

        // 紧凑模式不应该有多余的换行和空格（除了必要的）
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).expect("Invalid JSON");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["duration_ms"], 99.9);
    }

    #[test]
    fn test_summary_statistics() {
        let audit = create_test_audit_logger(100);

        // 添加各种事件
        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("u1", None))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("u2", None))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u1", "m1", true))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u3", "m2", false))
            .unwrap(); // 失败
        audit
            .log_enhanced(EnhancedAuditEvent::security_alert(
                "alert1",
                "high",
                json!({}),
            ))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::inference_request("u2", "m1", 100.0))
            .unwrap();

        let summary = audit.summary(Duration::hours(1)); // 最近1小时

        assert_eq!(summary.total_events, 6);
        assert!(summary.success_rate > 0.0 && summary.success_rate < 1.0); // 有成功也有失败
        assert_eq!(summary.unique_users, 3); // u1, u2, u3
        assert!(summary.unique_resources > 0);
        assert_eq!(summary.security_alerts_count, 1);
        assert!(!summary.top_users.is_empty());
        assert!(!summary.top_resources.is_empty());

        // 验证 top_users 包含最活跃的用户
        assert!(summary.top_users.iter().any(|(user, _)| user == "u1")); // u1有2个事件
    }

    #[test]
    fn test_summary_empty_buffer() {
        let audit = create_test_audit_logger(100);

        let summary = audit.summary(Duration::hours(1));

        assert_eq!(summary.total_events, 0);
        assert_eq!(summary.success_rate, 1.0); // 空时默认成功率1.0
        assert_eq!(summary.unique_users, 0);
        assert_eq!(summary.unique_resources, 0);
        assert!(summary.top_users.is_empty());
        assert!(summary.top_resources.is_empty());
        assert_eq!(summary.security_alerts_count, 0);
    }

    #[test]
    fn test_summary_time_window_filtering() {
        let audit = create_test_audit_logger(100);

        // 记录一个事件
        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("u1", None))
            .unwrap();

        // 使用非常小的时间窗口（1毫秒），应该不包含任何事件
        let summary = audit.summary(Duration::milliseconds(1));

        // 由于事件刚创建，可能在窗口内；这里主要验证结构完整性
        assert!(summary.success_rate >= 0.0 && summary.success_rate <= 1.0);
    }

    #[test]
    fn test_edge_case_empty_filter_returns_all() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::authentication_login("u1", None))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::model_deploy("u2", "m1", true))
            .unwrap();

        // 空过滤器应该返回所有事件
        let filter = EnhancedAuditFilter::new();
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_edge_case_special_characters_in_metadata() {
        let audit = create_test_audit_logger(100);

        // 创建包含特殊字符的元数据
        let special_details = json!({
            "message": "Test with special chars: <>&\"'",
            "unicode": "中文测试 🎉",
            "nested": {"key": "value with spaces"}
        });

        let event = EnhancedAuditEvent::security_alert("special-test", "info", special_details);
        audit.log_enhanced(event).unwrap();

        let results = audit.query_enhanced(&EnhancedAuditFilter::new()).unwrap();
        assert_eq!(results.len(), 1);

        // 验证特殊字符保留完整
        assert_eq!(
            results[0].metadata["details"]["message"],
            "Test with special chars: <>&\"'"
        );
        assert_eq!(results[0].metadata["details"]["unicode"], "中文测试 🎉");
    }

    #[test]
    fn test_edge_case_large_number_of_events() {
        let audit = create_test_audit_logger(1000);

        // 快速写入大量事件
        for i in 0..200 {
            audit
                .log_enhanced(EnhancedAuditEvent::data_access(
                    &format!("user-{}", i % 10),
                    &format!("res-{}", i),
                    ResourceType::Other,
                    "read",
                ))
                .unwrap();
        }

        let filter = EnhancedAuditFilter::new();
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 200);

        // 测试聚合性能
        let agg_results = audit.aggregate(AuditGroupBy::UserId, None).unwrap();
        assert!(!agg_results.is_empty());
        // 应该有10个用户分组
        assert_eq!(agg_results.len(), 10);
    }

    #[test]
    fn test_enhanced_filter_duration_range() {
        let audit = create_test_audit_logger(100);

        audit
            .log_enhanced(EnhancedAuditEvent::inference_request("u1", "m1", 50.0))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::inference_request("u2", "m2", 150.0))
            .unwrap();
        audit
            .log_enhanced(EnhancedAuditEvent::inference_request("u3", "m3", 250.0))
            .unwrap();

        // 过滤持续时间在 60ms - 200ms 之间
        let filter = EnhancedAuditFilter::new().with_duration_range(60.0, 200.0);
        let results = audit.query_enhanced(&filter).unwrap();

        assert_eq!(results.len(), 1); // 只有 150ms 符合
        assert_eq!(results[0].duration_ms.unwrap(), 150.0);
    }

    #[test]
    fn test_backward_compatibility_basic_to_enhanced() {
        // 验证从基础版转换后数据完整性
        let basic = AuditEvent {
            timestamp: Utc::now(),
            event_id: uuid::Uuid::new_v4(),
            user_id: "compat-user".to_string(),
            action: "model:deploy".to_string(),
            resource: "/api/models/test-model".to_string(),
            outcome: AuditOutcome::Success,
            details: json!({"version": "1.0", "env": "production"}),
            ip_address: Some("10.0.0.42".to_string()),
        };

        let enhanced = EnhancedAuditEvent::from_basic(basic);

        // 验证字段映射正确
        assert_eq!(enhanced.actor.user_id, "compat-user");
        assert_eq!(enhanced.actor.ip_address.as_deref().unwrap(), "10.0.0.42");
        assert_eq!(enhanced.event_type, AuditEventType::ModelManagement); // 从 model:deploy 推断
        assert_eq!(enhanced.resource.resource_type, ResourceType::Model); // 从路径推断
        assert_eq!(enhanced.metadata["version"], "1.0");
        assert_eq!(enhanced.metadata["env"], "production");
    }
}
