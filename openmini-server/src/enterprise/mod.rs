//! # Enterprise 企业版功能模块
//!
//! 本模块提供企业级功能支持，包括：
//! - **OAuth2/OIDC 认证** - 支持多种身份提供商集成
//! - **RBAC 权限控制** - 基于角色的细粒度访问控制
//! - **审计日志** - 完整的操作审计追踪
//! - **SLA 保障** - 服务级别协议监控与告警
//!
//! ## 架构设计
//!
//! 采用模块化设计，各子模块可独立使用或通过 `EnterpriseSuite` 聚合器统一管理：
//!
//! ```rust,ignore
//! use openmini_server::enterprise::{EnterpriseSuite, EnterpriseConfig};
//!
//! let config = EnterpriseConfig::default();
//! let suite = EnterpriseSuite::new(&config)?;
//!
//! // 使用中间件进行认证+授权+审计一体化处理
//! let response = suite.middleware(&request)?;
//! ```
//!
//! ## 线程安全保证
//!
//! 所有核心结构体均实现 `Send + Sync`，可在多线程环境中安全使用。
//! 内部使用 `Arc<RwLock<T>` 或 `Mutex<T>` 保证并发安全。

pub mod audit;
pub mod auth;
pub mod rbac;
pub mod sla;

use audit::AuditLogger;
use auth::AuthManager;
use rbac::RbacManager;
use serde::{Deserialize, Serialize};
use sla::SlaMonitor;
use std::sync::Arc;
use thiserror::Error;

/// 企业版功能聚合器
///
/// 将认证、授权、审计、SLA监控整合为统一的入口点，
/// 提供开箱即用的企业级功能套件。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::enterprise::{EnterpriseSuite, EnterpriseConfig};
///
/// // 创建默认配置
/// let config = EnterpriseConfig::default();
///
/// // 初始化企业版功能套件
/// let suite = EnterpriseSuite::new(&config)?;
///
/// // 获取各子模块的引用
/// let _auth = &suite.auth;
/// let _rbac = &suite.rbac;
/// ```
#[derive(Debug)]
pub struct EnterpriseSuite {
    /// OAuth2/OIDC 认证管理器
    pub auth: Arc<AuthManager>,
    /// RBAC 权限控制管理器
    pub rbac: Arc<RbacManager>,
    /// 审计日志记录器
    pub audit: Arc<AuditLogger>,
    /// SLA 监控与告警
    pub sla: Arc<SlaMonitor>,
}

/// 企业版配置
///
/// 包含所有企业级功能的配置项，支持从 TOML/JSON 反序列化。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnterpriseConfig {
    /// 认证配置
    #[serde(default)]
    pub auth: AuthConfig,
    /// RBAC 配置
    #[serde(default)]
    pub rbac: RbacConfig,
    /// 审计日志配置
    #[serde(default)]
    pub audit: AuditConfig,
    /// SLA 配置
    #[serde(default)]
    pub sla: SlaConfig,
}

/// 认证配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// OAuth2 客户端 ID
    #[serde(default = "default_client_id")]
    pub client_id: String,
    /// OAuth2 客户端密钥
    #[serde(default)]
    pub client_secret: Option<String>,
    /// OIDC 发现端点 URL
    #[serde(default)]
    pub oidc_discovery_url: Option<String>,
    /// Token 过期时间（秒）
    #[serde(default = "default_token_expiry")]
    pub token_expiry_secs: u64,
    /// 会话最大数量
    #[serde(default = "default_max_sessions")]
    pub max_sessions: usize,
}

fn default_client_id() -> String {
    "openmini-enterprise".to_string()
}

fn default_token_expiry() -> u64 {
    3600
}

fn default_max_sessions() -> usize {
    10000
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            client_id: default_client_id(),
            client_secret: None,
            oidc_discovery_url: None,
            token_expiry_secs: default_token_expiry(),
            max_sessions: default_max_sessions(),
        }
    }
}

/// RBAC 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacConfig {
    /// 是否启用 RBAC
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// 默认策略文件路径
    #[serde(default)]
    pub policy_file: Option<String>,
    /// 缓存策略评估结果（秒）
    #[serde(default = "default_cache_ttl")]
    pub cache_ttl_secs: u64,
}

fn default_enabled() -> bool {
    true
}

fn default_cache_ttl() -> u64 {
    300
}

impl Default for RbacConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            policy_file: None,
            cache_ttl_secs: default_cache_ttl(),
        }
    }
}

/// 审计日志配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// 是否启用审计日志
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// 日志缓冲区大小
    #[serde(default = "default_buffer_size")]
    pub buffer_size: usize,
    /// 日志保留天数
    #[serde(default = "default_retention_days")]
    pub retention_days: u32,
    /// 导出格式 (json/csv)
    #[serde(default = "default_export_format")]
    pub export_format: String,
}

fn default_buffer_size() -> usize {
    10000
}

fn default_retention_days() -> u32 {
    90
}

fn default_export_format() -> String {
    "json".to_string()
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            buffer_size: default_buffer_size(),
            retention_days: default_retention_days(),
            export_format: default_export_format(),
        }
    }
}

/// SLA 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaConfig {
    /// 是否启用 SLA 监控
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// 监控窗口期（秒）
    #[serde(default = "default_window_secs")]
    pub window_secs: u64,
    /// 告警阈值（可用性百分比）
    #[serde(default = "default_availability_threshold")]
    pub availability_threshold: f32,
    /// P95 延迟阈值（毫秒）
    #[serde(default = "default_latency_p95")]
    pub latency_p95_ms: f32,
}

fn default_window_secs() -> u64 {
    300
}

fn default_availability_threshold() -> f32 {
    99.9
}

fn default_latency_p95() -> f32 {
    1000.0
}

impl Default for SlaConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            window_secs: default_window_secs(),
            availability_threshold: default_availability_threshold(),
            latency_p95_ms: default_latency_p95(),
        }
    }
}

/// 企业版错误类型
#[derive(Debug, Error)]
pub enum EnterpriseError {
    /// 认证错误
    #[error("Authentication error: {0}")]
    Authentication(String),
    /// 授权错误
    #[error("Authorization error: {0}")]
    Authorization(String),
    /// 审计日志错误
    #[error("Audit error: {0}")]
    Audit(String),
    /// SLA 错误
    #[error("SLA error: {0}")]
    Sla(String),
    /// 配置错误
    #[error("Configuration error: {0}")]
    Configuration(String),
    /// 内部错误
    #[error("Internal error: {0}")]
    Internal(String),
}

impl EnterpriseSuite {
    /// 创建新的企业版功能套件
    ///
    /// 根据提供的配置初始化所有子模块：
    /// - 认证管理器 (AuthManager)
    /// - RBAC 管理器 (RbacManager)
    /// - 审计日志记录器 (AuditLogger)
    /// - SLA 监控器 (SlaMonitor)
    ///
    /// # 参数
    ///
    /// * `config` - 企业版配置对象
    ///
    /// # 错误
    ///
    /// 返回 `EnterpriseError` 如果任何子模块初始化失败
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use openmini_server::enterprise::{EnterpriseSuite, EnterpriseConfig};
    ///
    /// let config = EnterpriseConfig::default();
    /// let suite = EnterpriseSuite::new(&config)?;
    /// ```
    pub fn new(config: &EnterpriseConfig) -> Result<Self, EnterpriseError> {
        let auth = AuthManager::new(&config.auth)
            .map_err(|e| EnterpriseError::Authentication(e.to_string()))?;

        let rbac = RbacManager::new(&config.rbac)
            .map_err(|e| EnterpriseError::Authorization(e.to_string()))?;

        let audit =
            AuditLogger::new(&config.audit).map_err(|e| EnterpriseError::Audit(e.to_string()))?;

        let sla = SlaMonitor::new(&config.sla).map_err(|e| EnterpriseError::Sla(e.to_string()))?;

        Ok(Self {
            auth: Arc::new(auth),
            rbac: Arc::new(rbac),
            audit: Arc::new(audit),
            sla: Arc::new(sla),
        })
    }

    /// 中间件：认证+授权+审计一体化处理
    ///
    /// 对传入请求执行完整的企业级处理流程：
    /// 1. **认证验证** - 验证 Token 并提取用户上下文
    /// 2. **权限检查** - 根据 RBAC 策略检查访问权限
    /// 3. **审计记录** - 记录操作结果到审计日志
    ///
    /// # 参数
    ///
    /// * `request` - HTTP 请求信息（简化表示）
    ///
    /// # 返回值
    ///
    /// 返回处理结果，包含用户上下文和授权状态
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// use openmini_server::enterprise::EnterpriseSuite;
    ///
    /// let suite = EnterpriseSuite::new(&config)?;
    /// let result = suite.middleware(&request)?;
    /// ```
    pub fn middleware(
        &self,
        token: &str,
        resource: &str,
        action: &str,
        ip_address: Option<&str>,
    ) -> Result<MiddlewareResult, EnterpriseError> {
        // Step 1: 认证
        let ctx = self
            .auth
            .authenticate(token)
            .map_err(|e| EnterpriseError::Authentication(e.to_string()))?;

        // Step 2: 授权 (RBAC)
        let action_enum = rbac::Action::from_str_or_read(action);
        let authorized = self.rbac.check_permission(&ctx, resource, action_enum);

        if !authorized {
            // 记录拒绝事件
            self.audit
                .log(audit::AuditEvent {
                    timestamp: chrono::Utc::now(),
                    event_id: uuid::Uuid::new_v4(),
                    user_id: ctx.user_id.clone(),
                    action: format!("{}:{}", action, resource),
                    resource: resource.to_string(),
                    outcome: audit::AuditOutcome::Denied,
                    details: serde_json::json!({"reason": "permission_denied"}),
                    ip_address: ip_address.map(|s| s.to_string()),
                })
                .map_err(|e| EnterpriseError::Audit(e.to_string()))?;

            return Err(EnterpriseError::Authorization(
                "Permission denied".to_string(),
            ));
        }

        // Step 3: 记录成功审计
        self.audit
            .log(audit::AuditEvent {
                timestamp: chrono::Utc::now(),
                event_id: uuid::Uuid::new_v4(),
                user_id: ctx.user_id.clone(),
                action: format!("{}:{}", action, resource),
                resource: resource.to_string(),
                outcome: audit::AuditOutcome::Success,
                details: serde_json::json!({"roles": ctx.roles}),
                ip_address: ip_address.map(|s| s.to_string()),
            })
            .map_err(|e| EnterpriseError::Audit(e.to_string()))?;

        Ok(MiddlewareResult {
            context: ctx,
            authorized: true,
        })
    }
}

/// 中间件处理结果
#[derive(Debug, Clone, Serialize)]
pub struct MiddlewareResult {
    /// 认证后的用户上下文
    pub context: auth::AuthContext,
    /// 是否授权成功
    pub authorized: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enterprise_config_default() {
        let config = EnterpriseConfig::default();
        assert_eq!(config.auth.client_id, "openmini-enterprise");
        assert_eq!(config.auth.token_expiry_secs, 3600);
        assert!(config.rbac.enabled);
        assert_eq!(config.audit.buffer_size, 10000);
        assert!(config.sla.enabled);
    }

    #[test]
    fn test_auth_config_default() {
        let config = AuthConfig::default();
        assert_eq!(config.client_id, "openmini-enterprise");
        assert_eq!(config.max_sessions, 10000);
    }

    #[test]
    fn test_rbac_config_default() {
        let config = RbacConfig::default();
        assert!(config.enabled);
        assert_eq!(config.cache_ttl_secs, 300);
    }

    #[test]
    fn test_audit_config_default() {
        let config = AuditConfig::default();
        assert!(config.enabled);
        assert_eq!(config.buffer_size, 10000);
        assert_eq!(config.retention_days, 90);
    }

    #[test]
    fn test_sla_config_default() {
        let config = SlaConfig::default();
        assert!(config.enabled);
        assert_eq!(config.window_secs, 300);
        assert!((config.availability_threshold - 99.9).abs() < f32::EPSILON);
    }
}
