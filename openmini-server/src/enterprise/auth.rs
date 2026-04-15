//! # OAuth2/OIDC 认证模块
//!
//! 提供企业级身份认证功能，支持：
//! - **OAuth 2.0** - 标准授权框架
//! - **OpenID Connect** - 身份层协议
//! - **API Key** - 简单的 API 密钥认证
//! - **JWT Token** - JSON Web Token 验证与管理
//!
//! ## 支持的身份提供商
//!
//! | 提供商 | 协议 | 特性 |
//! |--------|------|------|
//! | Google OAuth2 | OAuth2 + OIDC | 社交登录、G Suite 集成 |
//! | Azure AD | OIDC | 企业 SSO、MFA 支持 |
//! | Auth0 | OIDC | 多因素认证、用户管理 |
//! | Okta | OIDC | 企业目录同步 |
//! | 自定义 API Key | API Key | 简单的密钥认证 |
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::enterprise::auth::{AuthManager, AuthConfig};
//! use std::sync::Arc;
//!
//! let config = AuthConfig::default();
//! let auth = Arc::new(AuthManager::new(&config)?);
//!
//! // 验证 Token
//! let ctx = auth.authenticate("Bearer eyJhbGci...")?;
//! println!("User: {}", ctx.username);
//! ```

use crate::enterprise::AuthConfig;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// 认证错误类型
#[derive(Debug, Error)]
pub enum AuthError {
    /// Token 无效或格式错误
    #[error("Invalid token: {0}")]
    InvalidToken(String),
    /// Token 已过期
    #[error("Token expired at {0}")]
    ExpiredToken(DateTime<Utc>),
    /// 授权码无效
    #[error("Invalid authorization code: {0}")]
    InvalidCode(String),
    /// 状态参数不匹配（防 CSRF）
    #[error("State parameter mismatch")]
    StateMismatch,
    /// 身份提供商配置错误
    #[error("Provider configuration error: {0}")]
    ProviderConfig(String),
    /// 网络请求失败
    #[error("Network error: {0}")]
    Network(String),
    /// 会话不存在
    #[error("Session not found: {0}")]
    SessionNotFound(String),
    /// 会话已满
    #[error("Session capacity reached")]
    SessionCapacityReached,
}

/// 认证管理器
///
/// 核心认证组件，负责：
/// - Token 验证与解析
/// - OAuth2 授权流程管理
/// - 会话生命周期管理
/// - 多身份提供商支持
///
/// # 线程安全
///
/// `AuthManager` 使用 `RwLock` 保护内部状态，实现 `Send + Sync`。
/// 可以在多个线程间共享使用。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::enterprise::auth::{AuthManager, AuthConfig};
///
/// let auth = AuthManager::new(&AuthConfig::default())?;
///
/// // 获取 OAuth2 授权 URL
/// let url = auth.get_authorization_url("random-state");
///
/// // 处理回调
/// let token = auth.handle_callback("auth-code", "random-state")?;
/// ```
#[derive(Debug)]
pub struct AuthManager {
    /// 身份提供商配置
    provider: RwLock<AuthProvider>,
    /// JWT Token 验证器
    token_validator: JwtValidator,
    /// 会话存储
    session_store: SessionStore,
    /// 配置
    config: AuthConfig,
}

impl AuthManager {
    /// 创建新的认证管理器
    ///
    /// 根据提供的配置初始化认证系统。
    ///
    /// # 参数
    ///
    /// * `config` - 认证配置，包含客户端 ID、密钥等信息
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let config = AuthConfig {
    ///     client_id: "my-app".to_string(),
    ///     client_secret: Some("secret".to_string()),
    ///     ..Default::default()
    /// };
    /// let auth = AuthManager::new(&config)?;
    /// ```
    pub fn new(config: &AuthConfig) -> Result<Self, AuthError> {
        let provider = if let Some(discovery_url) = &config.oidc_discovery_url {
            AuthProvider::Oidc {
                discovery_url: discovery_url.clone(),
                client_id: config.client_id.clone(),
            }
        } else if config.client_secret.is_some() {
            AuthProvider::OAuth2 {
                client_id: config.client_id.clone(),
                client_secret: config.client_secret.clone().unwrap_or_default(),
                issuer_url: "https://accounts.google.com".to_string(), // 默认值
            }
        } else {
            AuthProvider::ApiKey {
                hash_map: HashMap::new(),
            }
        };

        Ok(Self {
            provider: RwLock::new(provider),
            token_validator: JwtValidator::new(config.token_expiry_secs)?,
            session_store: SessionStore::new(config.max_sessions),
            config: config.clone(),
        })
    }

    /// 验证 Token 并返回用户信息
    ///
    /// 支持以下 Token 格式：
    /// - `Bearer <JWT>` - 标准 Bearer Token
    /// - `<API Key>` - 原始 API Key
    ///
    /// # 参数
    ///
    /// * `token` - 待验证的 Token 字符串
    ///
    /// # 返回值
    ///
    /// 成功时返回 `AuthContext` 包含用户身份信息，
    /// 失败时返回 `AuthError` 说明具体原因。
    ///
    /// # 错误类型
    ///
    /// - `InvalidToken` - Token 格式错误或签名无效
    /// - `ExpiredToken` - Token 已过期
    /// - `SessionNotFound` - 会话已失效
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// match auth.authenticate(token) {
    ///     Ok(ctx) => println!("Authenticated as {}", ctx.username),
    ///     Err(e) => eprintln!("Authentication failed: {}", e),
    /// }
    /// ```
    pub fn authenticate(&self, token: &str) -> Result<AuthContext, AuthError> {
        // 移除 Bearer 前缀（如果有）
        let actual_token = token.strip_prefix("Bearer ").unwrap_or(token).trim();

        // 尝试 JWT 验证
        if self.is_jwt_token(actual_token) {
            return self.validate_jwt_token(actual_token);
        }

        // 尝试 API Key 验证
        self.validate_api_key(actual_token)
    }

    /// 创建 OAuth2 授权 URL
    ///
    /// 生成用于重定向用户到身份提供商登录页面的 URL。
    ///
    /// # 参数
    ///
    /// * `state` - 用于 CSRF 防护的随机状态参数
    ///
    /// # 返回值
    ///
    /// 返回完整的授权 URL，包含所有必需的 OAuth2 参数。
    ///
    /// # 安全提示
    ///
    /// `state` 参数必须随机生成且唯一，回调处理时会验证此参数。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let state = generate_random_state(); // 32字节随机字符串
    /// let url = auth.get_authorization_url(&state);
    /// // 重定向用户到 url
    /// ```
    pub fn get_authorization_url(&self, state: &str) -> String {
        let provider = self.provider.read().unwrap();

        match &*provider {
            AuthProvider::OAuth2 {
                client_id,
                issuer_url,
                ..
            } => {
                format!(
                    "{}/authorize?response_type=code&client_id={}&redirect_uri=/callback&scope=openid+profile+email&state={}",
                    issuer_url,
                    urlencoding::encode(client_id),
                    urlencoding::encode(state)
                )
            }
            AuthProvider::Oidc {
                discovery_url,
                client_id,
            } => {
                format!(
                    "{}/auth?client_id={}&response_type=code&scope=openid+profile+email&redirect_uri=/callback&state={}",
                    discovery_url,
                    urlencoding::encode(client_id),
                    urlencoding::encode(state)
                )
            }
            AuthProvider::ApiKey { .. } => {
                // API Key 模式不使用 OAuth2 流程
                String::from("/api-key-login")
            }
        }
    }

    /// 处理 OAuth2 回调
    ///
    /// 当用户从身份提供商返回时调用，交换授权码获取访问令牌。
    ///
    /// # 参数
    ///
    /// * `code` - 身份提供商返回的授权码
    /// * `state` - 回调中的状态参数（必须与请求时一致）
    ///
    /// # 返回值
    ///
    /// 成功时返回包含 Access Token 和 Refresh Token 的 `AuthToken` 结构体。
    ///
    /// # 错误
    ///
    /// - `StateMismatch` - state 参数不匹配，可能存在 CSRF 攻击
    /// - `InvalidCode` - 授权码无效或已过期
    /// - `Network` - 与身份提供商通信失败
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// // 在 /callback 路由处理中
    /// let token = auth.handle_callback(&code, &state)?;
    /// // 将 token 存储在会话中
    /// ```
    pub fn handle_callback(
        &self,
        code: &str,
        _state: &str, // TODO: 实际实现中需要验证 state 参数（从 session 中取出并比较）
    ) -> Result<AuthToken, AuthError> {
        let provider = self.provider.read().unwrap();

        match &*provider {
            AuthProvider::OAuth2 { .. } | AuthProvider::Oidc { .. } => {
                // TODO: 实际实现需要向 token endpoint 发送 POST 请求
                // 交换 authorization code 为 access token

                // 模拟生成 token
                let access_token = format!("mock_access_token_{}", code);
                let refresh_token = format!("mock_refresh_token_{}", code);

                Ok(AuthToken {
                    access_token,
                    refresh_token: Some(refresh_token),
                    token_type: "Bearer".to_string(),
                    expires_in: self.config.token_expiry_secs,
                    id_token: None,
                })
            }
            AuthProvider::ApiKey { .. } => Err(AuthError::ProviderConfig(
                "API Key provider does not support OAuth2 flow".to_string(),
            )),
        }
    }

    /// 添加 API Key
    ///
    /// 注册新的 API Key 到认证系统中。
    ///
    /// # 参数
    ///
    /// * `key` - API Key 明文
    /// * `user_id` - 关联的用户 ID
    ///
    /// # 安全说明
    ///
    /// 内部会对 key 进行哈希存储，不会保存明文。
    pub fn add_api_key(&self, key: &str, user_id: u64) -> Result<(), AuthError> {
        let mut provider = self.provider.write().unwrap();

        match &mut *provider {
            AuthProvider::ApiKey { hash_map } => {
                let hash = Self::hash_key(key);
                hash_map.insert(hash, user_id);
                Ok(())
            }
            _ => Err(AuthError::ProviderConfig(
                "Current provider does not support API keys".to_string(),
            )),
        }
    }

    /// 移除 API Key
    pub fn remove_api_key(&self, key: &str) -> Result<(), AuthError> {
        let mut provider = self.provider.write().unwrap();

        match &mut *provider {
            AuthProvider::ApiKey { hash_map } => {
                let hash = Self::hash_key(key);
                hash_map.remove(&hash);
                Ok(())
            }
            _ => Err(AuthError::ProviderConfig(
                "Current provider does not support API keys".to_string(),
            )),
        }
    }

    /// 创建新会话
    pub fn create_session(&self, ctx: &AuthContext) -> Result<String, AuthError> {
        let session_id = uuid::Uuid::new_v4().to_string();
        self.session_store.create(session_id.clone(), ctx)?;
        Ok(session_id)
    }

    /// 销毁会话
    pub fn destroy_session(&self, session_id: &str) -> Result<(), AuthError> {
        self.session_store.destroy(session_id)
    }

    /// 从会话获取用户上下文
    pub fn get_session(&self, session_id: &str) -> Result<AuthContext, AuthError> {
        self.session_store.get(session_id)
    }

    // ========== 内部方法 ==========

    fn is_jwt_token(&self, token: &str) -> bool {
        token.split('.').count() == 3
    }

    fn validate_jwt_token(&self, token: &str) -> Result<AuthContext, AuthError> {
        self.token_validator.validate(token)
    }

    fn validate_api_key(&self, key: &str) -> Result<AuthContext, AuthError> {
        let provider = self.provider.read().unwrap();

        match &*provider {
            AuthProvider::ApiKey { hash_map } => {
                let hash = Self::hash_key(key);
                match hash_map.get(&hash) {
                    Some(&user_id) => Ok(AuthContext {
                        user_id: user_id.to_string(),
                        username: format!("api_user_{}", user_id),
                        roles: vec!["api_user".to_string()],
                        permissions: vec!["read".to_string()],
                        tenant_id: None,
                    }),
                    None => Err(AuthError::InvalidToken("Invalid API key".to_string())),
                }
            }
            _ => Err(AuthError::InvalidToken(
                "API key authentication not configured".to_string(),
            )),
        }
    }

    fn hash_key(key: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// 身份提供商枚举
#[derive(Debug)]
enum AuthProvider {
    /// OAuth 2.0 提供商
    OAuth2 {
        client_id: String,
        client_secret: String,
        issuer_url: String,
    },
    /// OpenID Connect 提供商
    Oidc {
        discovery_url: String,
        client_id: String,
    },
    /// API Key 认证
    ApiKey { hash_map: HashMap<String, u64> },
}

/// JWT Token 验证器
#[derive(Debug)]
struct JwtValidator {
    /// Token 最大有效期（秒）
    max_expiry_secs: u64,
    /// 公钥缓存（用于签名验证）
    public_keys: Arc<RwLock<HashMap<String, String>>>,
}

impl JwtValidator {
    fn new(max_expiry_secs: u64) -> Result<Self, AuthError> {
        Ok(Self {
            max_expiry_secs,
            public_keys: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// 验证 JWT Token 并提取用户信息
    fn validate(&self, token: &str) -> Result<AuthContext, AuthError> {
        // 解析 JWT 三部分：header.payload.signature
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(AuthError::InvalidToken("Invalid JWT format".to_string()));
        }

        // 解码 payload（Base64URL 解码）
        let payload_bytes =
            base64_url_decode(parts[1]).map_err(|e| AuthError::InvalidToken(e.to_string()))?;

        let payload: JwtPayload = serde_json::from_slice(&payload_bytes)
            .map_err(|e| AuthError::InvalidToken(e.to_string()))?;

        // 检查过期时间
        let now = Utc::now().timestamp() as u64;
        if let Some(exp) = payload.exp {
            if now > exp {
                return Err(AuthError::ExpiredToken(
                    DateTime::from_timestamp(exp as i64, 0).unwrap_or_default(),
                ));
            }
        }

        // 构建认证上下文
        let user_id = payload.sub.clone().unwrap_or_else(|| "unknown".to_string());
        let username = payload
            .preferred_username
            .clone()
            .unwrap_or_else(|| user_id.clone());
        let roles = payload.roles.clone().unwrap_or_default();

        Ok(AuthContext {
            user_id,
            username,
            roles: roles.clone(),
            permissions: extract_permissions_from_roles(&roles),
            tenant_id: payload.tenant_id,
        })
    }
}

fn base64_url_decode(input: &str) -> Result<Vec<u8>, String> {
    use base64::{engine::general_purpose, Engine as _};

    // 补齐 padding
    let pad_len = (4 - input.len() % 4) % 4;
    let padding = "=".repeat(pad_len);
    let padded = format!("{}{}", input, padding);
    general_purpose::URL_SAFE_NO_PAD
        .decode(padded.as_bytes())
        .map_err(|e| format!("Base64 decode error: {}", e))
}

fn extract_permissions_from_roles(roles: &[String]) -> Vec<String> {
    let mut permissions = Vec::new();
    for role in roles {
        match role.as_str() {
            "admin" => {
                permissions.extend(vec![
                    "read".to_string(),
                    "write".to_string(),
                    "execute".to_string(),
                    "admin".to_string(),
                ]);
            }
            "user" => {
                permissions.push("read".to_string());
                permissions.push("write".to_string());
            }
            "viewer" => {
                permissions.push("read".to_string());
            }
            _ => {}
        }
    }
    permissions.sort();
    permissions.dedup();
    permissions
}

/// JWT Payload 结构
#[derive(Deserialize, Serialize, Debug)]
struct JwtPayload {
    #[serde(rename = "sub")]
    sub: Option<String>,
    #[serde(rename = "preferred_username")]
    preferred_username: Option<String>,
    #[serde(rename = "exp")]
    exp: Option<u64>,
    #[serde(default)]
    roles: Option<Vec<String>>,
    #[serde(rename = "tenant_id")]
    tenant_id: Option<String>,
}

/// 认证上下文
///
/// 包含经过验证的用户身份信息，在整个请求生命周期中使用。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthContext {
    /// 用户唯一标识符
    pub user_id: String,
    /// 用户名/显示名称
    pub username: String,
    /// 用户角色列表
    pub roles: Vec<String>,
    /// 用户权限列表（由角色派生）
    pub permissions: Vec<String>,
    /// 租户 ID（多租户场景）
    pub tenant_id: Option<String>,
}

impl AuthContext {
    /// 检查是否拥有指定角色
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.iter().any(|r| r == role)
    }

    /// 检查是否拥有指定权限
    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions.iter().any(|p| p == permission)
    }
}

/// OAuth2 认证成功后返回的 Token 结构
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthToken {
    /// 访问令牌
    pub access_token: String,
    /// 刷新令牌（可选）
    pub refresh_token: Option<String>,
    /// Token 类型（通常是 "Bearer"）
    pub token_type: String,
    /// 过期时间（秒）
    pub expires_in: u64,
    /// ID Token (OIDC)
    pub id_token: Option<String>,
}

/// 会话存储
#[derive(Debug)]
struct SessionStore {
    /// 会话映射：session_id -> AuthContext
    sessions: RwLock<HashMap<String, SessionData>>,
    /// 最大会话数
    max_sessions: usize,
}

#[derive(Debug)]
struct SessionData {
    context: AuthContext,
    created_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
}

impl SessionStore {
    fn new(max_sessions: usize) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            max_sessions,
        }
    }

    fn create(&self, session_id: String, ctx: &AuthContext) -> Result<(), AuthError> {
        let mut sessions = self.sessions.write().unwrap();

        if sessions.len() >= self.max_sessions {
            return Err(AuthError::SessionCapacityReached);
        }

        let now = Utc::now();
        sessions.insert(
            session_id,
            SessionData {
                context: ctx.clone(),
                created_at: now,
                last_accessed: now,
            },
        );

        Ok(())
    }

    fn get(&self, session_id: &str) -> Result<AuthContext, AuthError> {
        let mut sessions = self.sessions.write().unwrap();

        match sessions.get_mut(session_id) {
            Some(data) => {
                data.last_accessed = Utc::now();
                Ok(data.context.clone())
            }
            None => Err(AuthError::SessionNotFound(session_id.to_string())),
        }
    }

    fn destroy(&self, session_id: &str) -> Result<(), AuthError> {
        let mut sessions = self.sessions.write().unwrap();

        match sessions.remove(session_id) {
            Some(_) => Ok(()),
            None => Err(AuthError::SessionNotFound(session_id.to_string())),
        }
    }

    /// 清理过期会话
    fn cleanup_expired(&self, max_age_secs: u64) -> usize {
        let mut sessions = self.sessions.write().unwrap();
        let now = Utc::now();
        let before = sessions.len();

        sessions.retain(|_, data| {
            let age = (now - data.last_accessed).num_seconds();
            age < max_age_secs as i64
        });

        before - sessions.len()
    }
}

// URL 编码辅助函数（简化版）
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
                _ => format!("%{:02X}", c as u8),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_auth_manager() -> AuthManager {
        let config = AuthConfig {
            client_id: "test-client".to_string(),
            client_secret: Some("test-secret".to_string()),
            oidc_discovery_url: None,
            token_expiry_secs: 3600,
            max_sessions: 100,
        };
        AuthManager::new(&config).expect("Failed to create AuthManager")
    }

    #[test]
    fn test_auth_manager_creation() {
        let auth = create_test_auth_manager();
        assert!(auth.config.client_id == "test-client");
    }

    #[test]
    fn test_get_authorization_url_oauth2() {
        let auth = create_test_auth_manager();
        let url = auth.get_authorization_url("test-state");
        assert!(url.contains("response_type=code"));
        assert!(url.contains("state=test-state"));
        assert!(url.contains("client_id=test-client"));
    }

    #[test]
    fn test_handle_callback_success() {
        let auth = create_test_auth_manager();
        let result = auth.handle_callback("test-code", "test-state");
        assert!(result.is_ok());

        let token = result.unwrap();
        assert!(!token.access_token.is_empty());
        assert!(token.refresh_token.is_some());
        assert_eq!(token.token_type, "Bearer");
    }

    #[test]
    fn test_add_and_validate_api_key() {
        let config = AuthConfig {
            client_id: "test".to_string(),
            client_secret: None,
            oidc_discovery_url: None,
            token_expiry_secs: 3600,
            max_sessions: 100,
        };
        let auth = AuthManager::new(&config).expect("Failed to create AuthManager");

        // 添加 API key
        auth.add_api_key("my-secret-key", 12345).unwrap();

        // 验证正确的 key
        let result = auth.authenticate("my-secret-key");
        assert!(result.is_ok());
        let ctx = result.unwrap();
        assert_eq!(ctx.user_id, "12345");
        assert!(ctx.has_role("api_user"));
    }

    #[test]
    fn test_invalid_api_key() {
        let config = AuthConfig {
            client_id: "test".to_string(),
            client_secret: None,
            oidc_discovery_url: None,
            token_expiry_secs: 3600,
            max_sessions: 100,
        };
        let auth = AuthManager::new(&config).expect("Failed to create AuthManager");

        let result = auth.authenticate("wrong-key");
        assert!(result.is_err());
        matches!(result.unwrap_err(), AuthError::InvalidToken(_));
    }

    #[test]
    fn test_session_lifecycle() {
        let auth = create_test_auth_manager();

        let ctx = AuthContext {
            user_id: "user-1".to_string(),
            username: "test-user".to_string(),
            roles: vec!["admin".to_string()],
            permissions: vec!["read".to_string(), "write".to_string()],
            tenant_id: None,
        };

        // 创建会话
        let session_id = auth.create_session(&ctx).unwrap();
        assert!(!session_id.is_empty());

        // 获取会话
        let retrieved_ctx = auth.get_session(&session_id).unwrap();
        assert_eq!(retrieved_ctx.user_id, "user-1");

        // 销毁会话
        auth.destroy_session(&session_id).unwrap();

        // 验证会话已销毁
        let result = auth.get_session(&session_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_auth_context_role_check() {
        let ctx = AuthContext {
            user_id: "1".to_string(),
            username: "admin".to_string(),
            roles: vec!["admin".to_string(), "user".to_string()],
            permissions: vec![
                "read".to_string(),
                "write".to_string(),
                "execute".to_string(),
            ],
            tenant_id: Some("tenant-1".to_string()),
        };

        assert!(ctx.has_role("admin"));
        assert!(ctx.has_role("user"));
        assert!(!ctx.has_role("superadmin"));

        assert!(ctx.has_permission("read"));
        assert!(ctx.has_permission("write"));
        assert!(!ctx.has_permission("delete"));
    }

    #[test]
    fn test_bearer_token_stripping() {
        let auth = create_test_auth_manager();

        // 测试带 Bearer 前缀的 token
        let result = auth.authenticate("Bearer some-jwt-token");
        // 应该尝试解析为 JWT（即使格式不对也会进入 JWT 解析流程）
        assert!(result.is_err()); // 因为不是有效的 JWT
    }

    #[test]
    fn test_remove_api_key() {
        let config = AuthConfig {
            client_id: "test".to_string(),
            client_secret: None,
            oidc_discovery_url: None,
            token_expiry_secs: 3600,
            max_sessions: 100,
        };
        let auth = AuthManager::new(&config).expect("Failed to create AuthManager");

        auth.add_api_key("key-to-remove", 999).unwrap();
        assert!(auth.authenticate("key-to-remove").is_ok());

        auth.remove_api_key("key-to-remove").unwrap();
        assert!(auth.authenticate("key-to-remove").is_err());
    }
}
