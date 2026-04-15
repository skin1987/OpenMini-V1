use serde::{Deserialize, Serialize};

// User 结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: i64,
    pub username: String,
    pub email: String,
    pub password_hash: String,
    pub role: i32,
    pub is_active: bool,
    pub last_login_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

// UserRole 枚举（用于业务逻辑）
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(i32)]
pub enum UserRole {
    Admin = 0,
    Operator = 1,
    Viewer = 2,
}

impl Default for UserRole {
    fn default() -> Self {
        Self::Admin
    }
}

impl From<i32> for UserRole {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Admin,
            1 => Self::Operator,
            2 => Self::Viewer,
            _ => Self::Admin,
        }
    }
}

impl From<UserRole> for i32 {
    fn from(role: UserRole) -> i32 {
        role as i32
    }
}

impl std::fmt::Display for UserRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admin => write!(f, "admin"),
            Self::Operator => write!(f, "operator"),
            Self::Viewer => write!(f, "viewer"),
        }
    }
}

// ApiKey 结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: i64,
    pub key_prefix: String,
    pub key_hash: String,
    pub name: String,
    pub owner_id: i64,
    pub quota_daily_requests: Option<i64>,
    pub quota_monthly_tokens: Option<i64>,
    pub used_today_requests: i64,
    pub used_month_tokens: i64,
    pub is_active: bool,
    pub expires_at: Option<String>,
    pub last_used_at: Option<String>,
    pub created_at: String,
    pub revoked_at: Option<String>,
}

// AlertRule 结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: i64,
    pub name: String,
    pub metric_name: String,
    pub condition: String,
    pub threshold: f64,
    pub duration_seconds: i64,
    pub severity: i32,
    pub channels: String,
    pub webhook_url: Option<String>,
    pub is_enabled: bool,
    pub created_at: String,
    pub updated_at: String,
}

// AlertSeverity 枚举（用于业务逻辑）
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(i32)]
pub enum AlertSeverity {
    Critical = 0,
    Warning = 1,
    Info = 2,
}

impl Default for AlertSeverity {
    fn default() -> Self {
        Self::Warning
    }
}

impl From<i32> for AlertSeverity {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Critical,
            1 => Self::Warning,
            2 => Self::Info,
            _ => Self::Warning,
        }
    }
}

impl From<AlertSeverity> for i32 {
    fn from(severity: AlertSeverity) -> i32 {
        severity as i32
    }
}

// AlertRecord 结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRecord {
    pub id: i64,
    pub rule_id: i64,
    pub status: i32,
    pub severity: i32,
    pub message: String,
    pub value: f64,
    pub fired_at: String,
    pub acknowledged_at: Option<String>,
    pub acknowledged_by: Option<i64>,
    pub resolved_at: Option<String>,
    pub resolved_by: Option<i64>,
}

// AlertStatus 枚举（用于业务逻辑）
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(i32)]
pub enum AlertStatus {
    Firing = 0,
    Acknowledged = 1,
    Resolved = 2,
}

impl Default for AlertStatus {
    fn default() -> Self {
        Self::Firing
    }
}

impl From<i32> for AlertStatus {
    fn from(value: i32) -> Self {
        match value {
            0 => Self::Firing,
            1 => Self::Acknowledged,
            2 => Self::Resolved,
            _ => Self::Firing,
        }
    }
}

impl From<AlertStatus> for i32 {
    fn from(status: AlertStatus) -> i32 {
        status as i32
    }
}

// AuditLog 结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub id: i64,
    pub user_id: Option<i64>,
    pub action: String,
    pub resource_type: Option<String>,
    pub resource_id: Option<String>,
    pub detail: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub created_at: String,
}

// ConfigHistory 结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigHistory {
    pub id: i64,
    pub changed_by: Option<i64>,
    pub section: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub change_reason: Option<String>,
    pub created_at: String,
}

// ============ 单元测试 ============

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    // ==================== User 模型测试 ====================

    #[test]
    fn test_user_serialization() {
        let user = User {
            id: 1,
            username: "testuser".to_string(),
            email: "test@example.com".to_string(),
            password_hash: "hashed_password".to_string(),
            role: 0,
            is_active: true,
            last_login_at: Some("2024-01-01T00:00:00Z".to_string()),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_value(&user).unwrap();
        assert_eq!(json["id"], 1);
        assert_eq!(json["username"], "testuser");
        assert_eq!(json["email"], "test@example.com");
        assert!(json.get("password_hash").is_some());
        assert_eq!(json["role"], 0);
        assert_eq!(json["is_active"], true);
        assert!(json.get("last_login_at").is_some());
        assert_eq!(json["created_at"], "2024-01-01T00:00:00Z");
        assert_eq!(json["updated_at"], "2024-01-01T00:00:00Z");
    }

    #[test]
    fn test_user_deserialization() {
        let json_str = r#"{
            "id": 42,
            "username": "deserialized_user",
            "email": "deser@test.com",
            "password_hash": "some_hash",
            "role": 1,
            "is_active": false,
            "last_login_at": null,
            "created_at": "2024-06-15T12:30:00Z",
            "updated_at": "2024-06-15T12:30:00Z"
        }"#;

        let user: User = serde_json::from_str(json_str).unwrap();
        assert_eq!(user.id, 42);
        assert_eq!(user.username, "deserialized_user");
        assert_eq!(user.email, "deser@test.com");
        assert_eq!(user.role, 1);
        assert!(!user.is_active);
        assert!(user.last_login_at.is_none());
    }

    // ==================== UserRole 枚举测试 ====================

    #[test]
    fn test_user_role_from_i32() {
        assert_eq!(UserRole::from(0), UserRole::Admin);
        assert_eq!(UserRole::from(1), UserRole::Operator);
        assert_eq!(UserRole::from(2), UserRole::Viewer);

        // 无效值应该默认为 Admin
        assert_eq!(UserRole::from(99), UserRole::Admin);
        assert_eq!(UserRole::from(-1), UserRole::Admin);
    }

    #[test]
    fn test_user_role_to_i32() {
        assert_eq!(i32::from(UserRole::Admin), 0);
        assert_eq!(i32::from(UserRole::Operator), 1);
        assert_eq!(i32::from(UserRole::Viewer), 2);
    }

    #[test]
    fn test_user_role_display() {
        assert_eq!(format!("{}", UserRole::Admin), "admin");
        assert_eq!(format!("{}", UserRole::Operator), "operator");
        assert_eq!(format!("{}", UserRole::Viewer), "viewer");
    }

    #[test]
    fn test_user_role_default() {
        assert_eq!(UserRole::default(), UserRole::Admin);
    }

    #[test]
    fn test_user_role_equality() {
        assert_eq!(UserRole::Admin, UserRole::Admin);
        assert_ne!(UserRole::Admin, UserRole::Operator);
        assert_ne!(UserRole::Operator, UserRole::Viewer);
    }

    // ==================== ApiKey 模型测试 ====================

    #[test]
    fn test_apikey_serialization() {
        let apikey = ApiKey {
            id: 1,
            key_prefix: "om-sk_abc12345".to_string(),
            key_hash: "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
                .to_string(),
            name: "Test API Key".to_string(),
            owner_id: 10,
            quota_daily_requests: Some(1000),
            quota_monthly_tokens: Some(100000),
            used_today_requests: 50,
            used_month_tokens: 5000,
            is_active: true,
            expires_at: Some("2025-12-31T23:59:59Z".to_string()),
            last_used_at: Some("2024-06-15T10:00:00Z".to_string()),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            revoked_at: None,
        };

        let json = serde_json::to_value(&apikey).unwrap();
        assert_eq!(json["id"], 1);
        assert_eq!(json["key_prefix"], "om-sk_abc12345");
        assert_eq!(json["name"], "Test API Key");
        assert_eq!(json["owner_id"], 10);
        assert_eq!(json["quota_daily_requests"], 1000);
        assert_eq!(json["quota_monthly_tokens"], 100000);
        assert_eq!(json["used_today_requests"], 50);
        assert_eq!(json["used_month_tokens"], 5000);
        assert_eq!(json["is_active"], true);
        assert!(json.get("expires_at").is_some());
        assert!(json.get("last_used_at").is_some());
        // Option 在 JSON 中序列化为 null，而不是缺失
        assert_eq!(json.get("revoked_at").and_then(|v| v.as_null()), Some(())); // None 应该是 null
    }

    #[test]
    fn test_apikey_deserialization_with_optionals() {
        let json_str = r#"{
            "id": 5,
            "key_prefix": "om-sk_test",
            "key_hash": "hash123",
            "name": "Minimal Key",
            "owner_id": 1,
            "quota_daily_requests": null,
            "quota_monthly_tokens": null,
            "used_today_requests": 0,
            "used_month_tokens": 0,
            "is_active": true,
            "expires_at": null,
            "last_used_at": null,
            "created_at": "2024-01-01T00:00:00Z",
            "revoked_at": null
        }"#;

        let apikey: ApiKey = serde_json::from_str(json_str).unwrap();
        assert_eq!(apikey.id, 5);
        assert!(apikey.quota_daily_requests.is_none());
        assert!(apikey.quota_monthly_tokens.is_none());
        assert!(apikey.expires_at.is_none());
        assert!(apikey.last_used_at.is_none());
        assert!(apikey.revoked_at.is_none());
    }

    // ==================== AlertRule 模型测试 ====================

    #[test]
    fn test_alert_rule_serialization() {
        let rule = AlertRule {
            id: 1,
            name: "High CPU Usage".to_string(),
            metric_name: "cpu_usage".to_string(),
            condition: ">".to_string(),
            threshold: 90.5,
            duration_seconds: 300,
            severity: 0, // Critical
            channels: "[\"email\", \"slack\"]".to_string(),
            webhook_url: Some("https://example.com/webhook".to_string()),
            is_enabled: true,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_value(&rule).unwrap();
        assert_eq!(json["name"], "High CPU Usage");
        assert_eq!(json["metric_name"], "cpu_usage");
        assert_eq!(json["threshold"], 90.5);
        assert_eq!(json["severity"], 0); // Critical
        assert_eq!(json["is_enabled"], true);
    }

    // ==================== AlertSeverity 枚举测试 ====================

    #[test]
    fn test_alert_severity_from_i32() {
        assert_eq!(AlertSeverity::from(0), AlertSeverity::Critical);
        assert_eq!(AlertSeverity::from(1), AlertSeverity::Warning);
        assert_eq!(AlertSeverity::from(2), AlertSeverity::Info);

        // 无效值默认为 Warning
        assert_eq!(AlertSeverity::from(99), AlertSeverity::Warning);
    }

    #[test]
    fn test_alert_severity_default() {
        assert_eq!(AlertSeverity::default(), AlertSeverity::Warning);
    }

    // ==================== AlertStatus 枚举测试 ====================

    #[test]
    fn test_alert_status_from_i32() {
        assert_eq!(AlertStatus::from(0), AlertStatus::Firing);
        assert_eq!(AlertStatus::from(1), AlertStatus::Acknowledged);
        assert_eq!(AlertStatus::from(2), AlertStatus::Resolved);

        // 无效值默认为 Firing
        assert_eq!(AlertStatus::from(99), AlertStatus::Firing);
    }

    #[test]
    fn test_alert_status_default() {
        assert_eq!(AlertStatus::default(), AlertStatus::Firing);
    }

    // ==================== AuditLog 模型测试 ====================

    #[test]
    fn test_audit_log_serialization() {
        let log = AuditLog {
            id: 1,
            user_id: Some(10),
            action: "USER_CREATED".to_string(),
            resource_type: Some("users".to_string()),
            resource_id: Some("5".to_string()),
            detail: Some("Created new user admin".to_string()),
            ip_address: Some("192.168.1.1".to_string()),
            user_agent: Some("Mozilla/5.0".to_string()),
            created_at: "2024-06-15T10:30:00Z".to_string(),
        };

        let json = serde_json::to_value(&log).unwrap();
        assert_eq!(json["action"], "USER_CREATED");
        assert_eq!(json["resource_type"], "users");
        assert_eq!(json["ip_address"], "192.168.1.1");
    }

    #[test]
    fn test_audit_log_minimal() {
        let log = AuditLog {
            id: 2,
            user_id: None,
            action: "SYSTEM_STARTUP".to_string(),
            resource_type: None,
            resource_id: None,
            detail: None,
            ip_address: None,
            user_agent: None,
            created_at: "2024-06-16T08:00:00Z".to_string(),
        };

        let json = serde_json::to_value(&log).unwrap();
        assert_eq!(json["action"], "SYSTEM_STARTUP");
        // Option 在 JSON 中序列化为 null
        assert_eq!(json.get("user_id").and_then(|v| v.as_null()), Some(()));
        assert_eq!(
            json.get("resource_type").and_then(|v| v.as_null()),
            Some(())
        );
    }

    // ==================== ConfigHistory 模型测试 ====================

    #[test]
    fn test_config_history_serialization() {
        let history = ConfigHistory {
            id: 1,
            changed_by: Some(1),
            section: "server".to_string(),
            old_value: Some("8080".to_string()),
            new_value: Some("3000".to_string()),
            change_reason: Some("Port conflict resolution".to_string()),
            created_at: "2024-06-15T14:00:00Z".to_string(),
        };

        let json = serde_json::to_value(&history).unwrap();
        assert_eq!(json["section"], "server");
        assert_eq!(json["old_value"], "8080");
        assert_eq!(json["new_value"], "3000");
        assert_eq!(json["change_reason"], "Port conflict resolution");
    }
}
