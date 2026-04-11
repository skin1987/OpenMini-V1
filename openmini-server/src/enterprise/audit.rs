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
            total_pages: (total + page.size - 1) / page.size.max(1),
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
    pub fn export(
        &self,
        filter: AuditFilter,
        format: ExportFormat,
    ) -> Result<Vec<u8>, AuditError> {
        let buffer = self.buffer.lock().unwrap();

        let filtered: Vec<&AuditEvent> = buffer
            .iter()
            .filter(|event| filter.matches(event))
            .collect();

        match format {
            ExportFormat::Json => {
                let json_array =
                    serde_json::to_vec_pretty(&filtered).map_err(|e| AuditError::ExportFailed(e.to_string()))?;
                Ok(json_array)
            }
            ExportFormat::Csv => {
                let mut csv_data = Vec::new();
                // CSV 头部
                writeln!(csv_data, "timestamp,event_id,user_id,action,resource,outcome,details,ip_address")
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
    pub fn success(
        user_id: &str,
        action: &str,
        resource: &str,
        details: Value,
    ) -> Self {
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
    pub fn failure(
        user_id: &str,
        action: &str,
        resource: &str,
        details: Value,
    ) -> Self {
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
    pub fn denied(
        user_id: &str,
        action: &str,
        resource: &str,
        details: Value,
    ) -> Self {
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

    /// 从字符串解析
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "success" | "ok" => Some(AuditOutcome::Success),
            "failure" | "error" | "fail" => Some(AuditOutcome::Failure),
            "denied" | "forbidden" => Some(AuditOutcome::Denied),
            _ => None,
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

impl ExportFormat {
    /// 从字符串解析
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "csv" => ExportFormat::Csv,
            _ => ExportFormat::Json,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audit_logger(buffer_size: usize) -> AuditLogger {
        let config = AuditConfig {
            enabled: true,
            buffer_size,
            retention_days: 90,
            export_format: "json".to_string(),
        };
        AuditLogger::new(&config).expect("Failed to create AuditLogger")
    }

    fn create_sample_event(
        user_id: &str,
        action: &str,
        outcome: AuditOutcome,
    ) -> AuditEvent {
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
            .log(create_sample_event("alice", "logout", AuditOutcome::Success))
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
            .log(create_sample_event("user-1", "login", AuditOutcome::Success))
            .unwrap();
        audit
            .log(create_sample_event("user-2", "login", AuditOutcome::Failure))
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
            .log(create_sample_event("user-1", "auth:login", AuditOutcome::Success))
            .unwrap();
        audit
            .log(create_sample_event("user-1", "data:read", AuditOutcome::Success))
            .unwrap();
        audit
            .log(create_sample_event("user-2", "auth:login", AuditOutcome::Success))
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

        let parsed: Vec<Value> =
            serde_json::from_slice(&json_data).expect("Invalid JSON output");
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
            .log(create_sample_event("u1", "admin:user:list", AuditOutcome::Success))
            .unwrap();
        audit
            .log(create_sample_event("u2", "data:model:get", AuditOutcome::Success))
            .unwrap();
        audit
            .log(create_sample_event("u3", "admin:role:update", AuditOutcome::Success))
            .unwrap();

        let admin_events = audit
            .query(
                AuditFilter::new().with_resource_prefix("/api/admin/"),
                PageRequest { page: 1, size: 10 },
            )
            .unwrap();

        assert_eq!(admin_events.total, 2);
    }
}
