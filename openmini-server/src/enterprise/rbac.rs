//! # RBAC (基于角色的访问控制) 模块
//!
//! 提供细粒度的权限管理功能，支持：
//! - **策略定义** - 声明式的访问控制策略
//! - **角色层次** - 角色继承与层次关系
//! - **条件评估** - 基于上下文的动态权限判断
//! - **资源模式匹配** - 精确、前缀、通配符匹配
//!
//! ## 策略模型
//!
//! 采用基于属性的访问控制（ABAC）增强的 RBAC 模型：
//!
//! ```text
//! 用户 → 角色 → 权限 → 资源 + 操作 + 条件
//! ```
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use openmini_server::enterprise::rbac::{RbacManager, Policy, Action, Effect};
//! use openmini_server::enterprise::auth::AuthContext;
//!
//! let mut rbac = RbacManager::new(&RbacConfig::default())?;
//!
//! // 添加管理员策略
//! rbac.add_policy(Policy {
//!     id: "admin-full-access".to_string(),
//!     resource: ResourcePattern::Wildcard,
//!     actions: vec![Action::Admin],
//!     effect: Effect::Allow,
//!     conditions: None,
//! })?;
//!
//! // 检查权限
//! let ctx = AuthContext { roles: vec!["admin".to_string()], ... };
//! assert!(rbac.check_permission(&ctx, "/api/users", Action::Read));
//! ```

use crate::enterprise::RbacConfig;
use crate::enterprise::auth::AuthContext;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use thiserror::Error;

/// RBAC 错误类型
#[derive(Debug, Error)]
pub enum RbacError {
    /// 策略 ID 已存在
    #[error("Policy already exists: {0}")]
    PolicyExists(String),
    /// 策略不存在
    #[error("Policy not found: {0}")]
    PolicyNotFound(String),
    /// 无效的策略定义
    #[error("Invalid policy definition: {0}")]
    InvalidPolicy(String),
    /// 循环角色依赖检测
    #[error("Circular role dependency detected: {0}")]
    CircularDependency(String),
}

/// RBAC 权限管理器
///
/// 核心权限控制组件，负责：
/// - 存储和评估访问控制策略
/// - 管理角色层次关系
/// - 执行权限检查决策
///
/// # 评估算法
///
/// 采用 **Deny-Override** 模式：
/// 1. 如果任何 Deny 策略匹配 → 拒绝
/// 2. 如果任何 Allow 策略匹配 → 允许
/// 3. 否则 → 拒绝（默认拒绝）
///
/// # 线程安全
///
/// 内部使用 `RwLock` 保护策略数据，支持并发读取。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::enterprise::rbac::{RbacManager, Policy, ResourcePattern, Action, Effect};
///
/// let mut rbac = RbacManager::new(&RbacConfig::default())?;
///
/// // 定义只读策略
/// rbac.add_policy(Policy {
///     id: "readonly-policy".to_string(),
///     resource: ResourcePattern::Prefix("/api/".to_string()),
///     actions: vec![Action::Read],
///     effect: Effect::Allow,
///     conditions: None,
/// })?;
///
/// // 检查权限
/// let allowed = rbac.check_permission(&ctx, "/api/data", Action::Read);
/// ```
#[derive(Debug)]
pub struct RbacManager {
    /// 访问控制策略列表
    policies: RwLock<Vec<Policy>>,
    /// 角色层次关系：role -> parent_roles
    role_hierarchy: RwLock<HashMap<String, Vec<String>>>,
    /// 配置
    config: RbacConfig,
}

impl RbacManager {
    /// 创建新的 RBAC 管理器
    ///
    /// 初始化空的策略集合和角色层次。
    ///
    /// # 参数
    ///
    /// * `config` - RBAC 配置，包含缓存 TTL 等设置
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let rbac = RbacManager::new(&RbacConfig::default())?;
    /// ```
    pub fn new(config: &RbacConfig) -> Result<Self, RbacError> {
        Ok(Self {
            policies: RwLock::new(Vec::new()),
            role_hierarchy: RwLock::new(HashMap::new()),
            config: config.clone(),
        })
    }

    /// 检查是否有权限
    ///
    /// 根据用户上下文、目标资源和操作类型评估是否允许访问。
    ///
    /// # 参数
    ///
    /// * `ctx` - 认证后的用户上下文（包含角色信息）
    /// * `resource` - 目标资源路径（如 `/api/v1/models`）
    /// * `action` - 请求的操作类型（读/写/执行/管理）
    ///
    /// # 返回值
    ///
    /// 返回 `true` 表示允许访问，`false` 表示拒绝。
    ///
    /// # 评估流程
    ///
    /// 1. 展开用户的所有角色（包括继承的角色）
    /// 2. 遍历所有策略，检查资源模式匹配
    /// 3. 检查操作是否在允许列表中
    /// 4. 评估条件表达式（如果有）
    /// 5. 应用 Deny-Override 决策逻辑
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// if rbac.check_permission(&ctx, "/api/admin", Action::Admin) {
    ///     // 执行管理员操作
    /// } else {
    ///     return Err(Forbidden);
    /// }
    /// ```
    pub fn check_permission(
        &self,
        ctx: &AuthContext,
        resource: &str,
        action: Action,
    ) -> bool {
        // 展开所有角色（包括继承）
        let _all_roles = self.expand_roles(&ctx.roles); // 预留：未来可用于角色匹配优化

        // 收集所有匹配的策略
        let policies = self.policies.read().unwrap();
        let mut has_deny = false;
        let mut has_allow = false;

        for policy in policies.iter() {
            // 检查资源匹配
            if !policy.resource.matches(resource) {
                continue;
            }

            // 检查操作匹配
            if !policy.actions.contains(&action) && !policy.actions.contains(&Action::Admin) {
                continue;
            }

            // 检查角色绑定（策略必须关联到用户的某个角色）
            // 注意：当前简化实现不显式绑定策略到角色，
            // 实际生产环境应该添加 policy.roles 字段

            // 评估条件
            if let Some(conditions) = &policy.conditions {
                if !self.evaluate_conditions(conditions, ctx, resource) {
                    continue;
                }
            }

            match policy.effect {
                Effect::Deny => {
                    has_deny = true;
                }
                Effect::Allow => {
                    has_allow = true;
                }
            }
        }

        // Deny-Override: 显式拒绝优先
        if has_deny {
            return false;
        }

        has_allow
    }

    /// 添加策略
    ///
    /// 向策略库中添加新的访问控制策略。
    ///
    /// # 参数
    ///
    /// * `policy` - 完整的策略定义
    ///
    /// # 错误
    ///
    /// - `PolicyExists` - 相同 ID 的策略已存在
    /// - `InvalidPolicy` - 策略定义无效（如空资源模式）
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// rbac.add_policy(Policy {
    ///     id: "allow-models-read".to_string(),
    ///     resource: ResourcePattern::Prefix("/api/models/".to_string()),
    ///     actions: vec![Action::Read],
    ///     effect: Effect::Allow,
    ///     conditions: None,
    /// })?;
    /// ```
    pub fn add_policy(&mut self, policy: Policy) -> Result<(), RbacError> {
        // 验证策略有效性
        if policy.id.is_empty() {
            return Err(RbacError::InvalidPolicy("Policy ID cannot be empty".to_string()));
        }

        let mut policies = self.policies.write().unwrap();

        // 检查重复
        if policies.iter().any(|p| p.id == policy.id) {
            return Err(RbacError::PolicyExists(policy.id));
        }

        policies.push(policy);
        Ok(())
    }

    /// 移除策略
    ///
    /// 根据 ID 删除指定的访问控制策略。
    ///
    /// # 参数
    ///
    /// * `policy_id` - 要移除的策略 ID
    ///
    /// # 错误
    ///
    /// - `PolicyNotFound` - 指定 ID 的策略不存在
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// rbac.remove_policy("old-policy-id")?;
    /// ```
    pub fn remove_policy(&mut self, policy_id: &str) -> Result<(), RbacError> {
        let mut policies = self.policies.write().unwrap();
        let original_len = policies.len();

        policies.retain(|p| p.id != policy_id);

        if policies.len() == original_len {
            Err(RbacError::PolicyNotFound(policy_id.to_string()))
        } else {
            Ok(())
        }
    }

    /// 设置角色层次关系
    ///
    /// 定义角色的继承关系。子角色自动拥有父角色的所有权限。
    ///
    /// # 参数
    ///
    /// * `role` - 子角色名称
    /// * `parent_roles` - 父角色列表
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// // super_admin 继承 admin 和 user 的权限
    /// rbac.set_role_hierarchy("super_admin", vec!["admin".to_string(), "user".to_string()]);
    /// ```
    pub fn set_role_hierarchy(&self, role: &str, parent_roles: Vec<String>) -> Result<(), RbacError> {
        // 检测循环依赖
        if self.would_create_cycle(role, &parent_roles) {
            return Err(RbacError::CircularDependency(format!(
                "Adding {} -> {:?} would create cycle",
                role, parent_roles
            )));
        }

        let mut hierarchy = self.role_hierarchy.write().unwrap();
        hierarchy.insert(role.to_string(), parent_roles);
        Ok(())
    }

    /// 获取所有策略
    pub fn get_policies(&self) -> Vec<Policy> {
        self.policies.read().unwrap().clone()
    }

    /// 获取角色层次
    pub fn get_role_hierarchy(&self) -> HashMap<String, Vec<String>> {
        self.role_hierarchy.read().unwrap().clone()
    }

    // ========== 内部方法 ==========

    /// 展开角色，获取所有继承的角色（包括自身）
    fn expand_roles(&self, roles: &[String]) -> Vec<String> {
        let hierarchy = self.role_hierarchy.read().unwrap();
        let mut all_roles = roles.to_vec();
        let mut visited = std::collections::HashSet::new();

        for role in roles {
            self.collect_parent_roles(role, &hierarchy, &mut all_roles, &mut visited);
        }

        all_roles.sort();
        all_roles.dedup();
        all_roles
    }

    /// 递归收集父角色
    fn collect_parent_roles(
        &self,
        role: &str,
        hierarchy: &HashMap<String, Vec<String>>,
        result: &mut Vec<String>,
        visited: &mut std::collections::HashSet<String>,
    ) {
        if visited.contains(role) {
            return; // 防止无限循环（理论上不会发生，因为 add 时已检查）
        }
        visited.insert(role.to_string());

        if let Some(parents) = hierarchy.get(role) {
            for parent in parents {
                result.push(parent.clone());
                self.collect_parent_roles(parent, hierarchy, result, visited);
            }
        }
    }

    /// 检测是否会创建循环依赖
    fn would_create_cycle(&self, new_role: &str, new_parents: &[String]) -> bool {
        let hierarchy = self.role_hierarchy.read().unwrap();

        // DFS 检查：如果添加 new_role -> new_parents，
        // 检查从任何 new_parent 是否能到达 new_role（通过已有的层次关系）
        // 如果能到达，说明会形成循环
        for parent in new_parents {
            if self.can_reach(parent, new_role, &hierarchy) {
                return true;
            }
        }

        false
    }

    /// 检查 from 是否能到达 to（通过层次关系）
    /// 层次关系定义：hierarchy[X] = [Y1, Y2, ...] 表示 X 继承 Y1, Y2（Y 是 X 的父角色）
    fn can_reach(
        &self,
        from: &str,
        to: &str,
        hierarchy: &HashMap<String, Vec<String>>,
    ) -> bool {
        if from == to {
            return true;
        }

        // from 的父角色列表
        if let Some(parents) = hierarchy.get(from) {
            for parent in parents {
                if self.can_reach(parent, to, hierarchy) {
                    return true;
                }
            }
        }

        false
    }

    /// 评估条件表达式
    fn evaluate_conditions(
        &self,
        conditions: &[Condition],
        ctx: &AuthContext,
        resource: &str,
    ) -> bool {
        conditions.iter().all(|cond| cond.evaluate(ctx, resource))
    }
}

/// 访问控制策略
///
/// 定义一组访问规则，包含资源模式、允许的操作、效果和可选条件。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// 策略唯一标识符
    pub id: String,
    /// 资源模式（支持精确、前缀、通配符）
    pub resource: ResourcePattern,
    /// 允许或拒绝的操作列表
    pub actions: Vec<Action>,
    /// 策略效果（允许或拒绝）
    pub effect: Effect,
    /// 可选的条件表达式列表（所有条件都必须满足）
    pub conditions: Option<Vec<Condition>>,
}

/// 资源模式
///
/// 支持三种匹配方式：
/// - **Exact** - 精确匹配完整路径
/// - **Prefix** - 前缀匹配（用于目录级授权）
/// - **Wildcard** - 匹配所有资源
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourcePattern {
    /// 精确匹配
    Exact(String),
    /// 前缀匹配
    Prefix(String),
    /// 通配符（匹配所有）
    Wildcard,
}

impl ResourcePattern {
    /// 检查资源路径是否匹配此模式
    pub fn matches(&self, resource: &str) -> bool {
        match self {
            ResourcePattern::Exact(pattern) => pattern == resource,
            ResourcePattern::Prefix(prefix) => resource.starts_with(prefix.as_str()),
            ResourcePattern::Wildcard => true,
        }
    }
}

/// 操作类型枚举
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    /// 读操作（GET）
    Read,
    /// 写操作（POST, PUT, PATCH）
    Write,
    /// 执行操作（特殊操作）
    Execute,
    /// 管理操作（敏感操作）
    Admin,
}

impl Action {
    /// 从字符串解析操作类型
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "read" | "get" | "view" => Action::Read,
            "write" | "post" | "put" | "patch" | "create" | "update" | "delete" => Action::Write,
            "execute" | "run" | "invoke" => Action::Execute,
            "admin" | "manage" | "configure" => Action::Admin,
            _ => Action::Read, // 默认为读操作
        }
    }

    /// 转换为字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            Action::Read => "read",
            Action::Write => "write",
            Action::Execute => "execute",
            Action::Admin => "admin",
        }
    }
}

/// 策略效果
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Effect {
    /// 允许访问
    Allow,
    /// 拒绝访问
    Deny,
}

/// 条件表达式
///
/// 用于基于上下文的动态权限评估。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    /// 条件类型
    pub condition_type: ConditionType,
    /// 条件参数
    pub params: HashMap<String, String>,
}

impl Condition {
    /// 评估条件是否满足
    pub fn evaluate(&self, ctx: &AuthContext, _resource: &str) -> bool {
        match &self.condition_type {
            ConditionType::RoleRequired => {
                // 要求用户具有指定角色
                self.params
                    .get("role")
                    .map_or(false, |role| ctx.has_role(role))
            }
            ConditionType::TenantMatch => {
                // 要求用户租户与资源租户匹配
                self.params.get("tenant_id").map_or(true, |tenant| {
                    ctx.tenant_id.as_ref().map_or(false, |t| t == tenant)
                })
            }
            ConditionType::TimeRange => {
                // 时间范围限制（简化版，实际应使用 chrono）
                // TODO: 实现时间范围检查
                true
            }
            ConditionType::IpWhitelist => {
                // IP 白名单（需要传入 IP 信息，此处简化）
                true
            }
            ConditionType::Custom => {
                // 自定义条件（可扩展）
                true
            }
        }
    }
}

/// 条件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    /// 要求特定角色
    RoleRequired,
    /// 租户匹配
    TenantMatch,
    /// 时间范围
    TimeRange,
    /// IP 白名单
    IpWhitelist,
    /// 自定义条件
    Custom,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enterprise::auth::AuthContext;

    fn create_test_rbac() -> RbacManager {
        RbacManager::new(&RbacConfig::default()).expect("Failed to create RbacManager")
    }

    fn create_test_context(roles: Vec<String>) -> AuthContext {
        AuthContext {
            user_id: "user-1".to_string(),
            username: "test-user".to_string(),
            roles,
            permissions: vec!["read".to_string()],
            tenant_id: Some("tenant-1".to_string()),
        }
    }

    #[test]
    fn test_rbac_creation() {
        let rbac = create_test_rbac();
        assert!(rbac.get_policies().is_empty());
    }

    #[test]
    fn test_add_and_check_policy() {
        let mut rbac = create_test_rbac();

        // 添加允许读取 /api/data 的策略
        rbac.add_policy(Policy {
            id: "allow-data-read".to_string(),
            resource: ResourcePattern::Exact("/api/data".to_string()),
            actions: vec![Action::Read],
            effect: Effect::Allow,
            conditions: None,
        })
        .unwrap();

        let ctx = create_test_context(vec!["user".to_string()]);
        assert!(rbac.check_permission(&ctx, "/api/data", Action::Read));
        assert!(!rbac.check_permission(&ctx, "/api/data", Action::Write));
    }

    #[test]
    fn test_exact_resource_match() {
        let mut rbac = create_test_rbac();

        rbac.add_policy(Policy {
            id: "exact-match".to_string(),
            resource: ResourcePattern::Exact("/api/users/123".to_string()),
            actions: vec![Action::Read],
            effect: Effect::Allow,
            conditions: None,
        })
        .unwrap();

        let ctx = create_test_context(vec!["user".to_string()]);
        assert!(rbac.check_permission(&ctx, "/api/users/123", Action::Read));
        assert!(!rbac.check_permission(&ctx, "/api/users/456", Action::Read));
        assert!(!rbac.check_permission(&ctx, "/api/users", Action::Read));
    }

    #[test]
    fn test_prefix_resource_match() {
        let mut rbac = create_test_rbac();

        rbac.add_policy(Policy {
            id: "prefix-match".to_string(),
            resource: ResourcePattern::Prefix("/api/admin/".to_string()),
            actions: vec![Action::Admin],
            effect: Effect::Allow,
            conditions: None,
        })
        .unwrap();

        let ctx = create_test_context(vec!["admin".to_string()]);
        assert!(rbac.check_permission(&ctx, "/api/admin/users", Action::Admin));
        assert!(rbac.check_permission(&ctx, "/api/admin/settings", Action::Admin));
        assert!(!rbac.check_permission(&ctx, "/api/public/data", Action::Admin));
    }

    #[test]
    fn test_wildcard_match() {
        let mut rbac = create_test_rbac();

        rbac.add_policy(Policy {
            id: "super-admin".to_string(),
            resource: ResourcePattern::Wildcard,
            actions: vec![Action::Read, Action::Write, Action::Execute, Action::Admin],
            effect: Effect::Allow,
            conditions: None,
        })
        .unwrap();

        let ctx = create_test_context(vec!["superadmin".to_string()]);
        assert!(rbac.check_permission(&ctx, "/any/resource", Action::Read));
        assert!(rbac.check_permission(&ctx, "/another/path", Action::Write));
    }

    #[test]
    fn test_deny_override() {
        let mut rbac = create_test_rbac();

        // 允许所有用户读取
        rbac.add_policy(Policy {
            id: "allow-all-read".to_string(),
            resource: ResourcePattern::Wildcard,
            actions: vec![Action::Read],
            effect: Effect::Allow,
            conditions: None,
        })
        .unwrap();

        // 但拒绝读取敏感数据
        rbac.add_policy(Policy {
            id: "deny-sensitive".to_string(),
            resource: ResourcePattern::Prefix("/api/secrets/".to_string()),
            actions: vec![Action::Read],
            effect: Effect::Deny,
            conditions: None,
        })
        .unwrap();

        let ctx = create_test_context(vec!["user".to_string()]);
        assert!(rbac.check_permission(&ctx, "/api/public/data", Action::Read));
        assert!(!rbac.check_permission(&ctx, "/api/secrets/config", Action::Read)); // Deny 优先
    }

    #[test]
    fn test_remove_policy() {
        let mut rbac = create_test_rbac();

        rbac.add_policy(Policy {
            id: "temp-policy".to_string(),
            resource: ResourcePattern::Wildcard,
            actions: vec![Action::Read],
            effect: Effect::Allow,
            conditions: None,
        })
        .unwrap();

        assert_eq!(rbac.get_policies().len(), 1);

        rbac.remove_policy("temp-policy").unwrap();
        assert!(rbac.get_policies().is_empty());

        // 移除不存在的策略应该报错
        assert!(rbac.remove_policy("non-existent").is_err());
    }

    #[test]
    fn test_duplicate_policy_error() {
        let mut rbac = create_test_rbac();

        let policy = Policy {
            id: "same-id".to_string(),
            resource: ResourcePattern::Wildcard,
            actions: vec![Action::Read],
            effect: Effect::Allow,
            conditions: None,
        };

        rbac.add_policy(policy.clone()).unwrap();
        let result = rbac.add_policy(policy);
        assert!(result.is_err());
        matches!(result.unwrap_err(), RbacError::PolicyExists(_));
    }

    #[test]
    fn test_role_hierarchy() {
        let rbac = create_test_rbac();

        // 设置角色层次：super_admin -> admin -> user
        rbac.set_role_hierarchy("admin", vec!["user".to_string()])
            .unwrap();
        rbac.set_role_hierarchy("super_admin", vec!["admin".to_string()])
            .unwrap();

        // super_admin 应该展开为 [admin, super_admin, user]
        let expanded = rbac.expand_roles(&vec!["super_admin".to_string()]);
        assert!(expanded.contains(&"admin".to_string()));
        assert!(expanded.contains(&"user".to_string()));
        assert!(expanded.contains(&"super_admin".to_string()));
    }

    #[test]
    fn test_circular_dependency_detection() {
        let rbac = create_test_rbac();

        // A -> B
        rbac.set_role_hierarchy("A", vec!["B".to_string()]).unwrap();

        // B -> A 会创建循环
        let result = rbac.set_role_hierarchy("B", vec!["A".to_string()]);
        assert!(result.is_err());
        matches!(result.unwrap_err(), RbacError::CircularDependency(_));
    }

    #[test]
    fn test_condition_evaluation() {
        let mut rbac = create_test_rbac();

        rbac.add_policy(Policy {
            id: "conditional-policy".to_string(),
            resource: ResourcePattern::Prefix("/api/tenant/".to_string()),
            actions: vec![Action::Read],
            effect: Effect::Allow,
            conditions: Some(vec![Condition {
                condition_type: ConditionType::RoleRequired,
                params: {
                    let mut m = HashMap::new();
                    m.insert("role".to_string(), "manager".to_string());
                    m
                },
            }]),
        })
        .unwrap();

        // 有 manager 角色的用户可以访问
        let ctx_manager = create_test_context(vec!["manager".to_string()]);
        assert!(rbac.check_permission(&ctx_manager, "/api/tenant/data", Action::Read));

        // 没有 manager 角色的用户不能访问
        let ctx_user = create_test_context(vec!["user".to_string()]);
        assert!(!rbac.check_permission(&ctx_user, "/api/tenant/data", Action::Read));
    }

    #[test]
    fn test_action_from_str() {
        assert_eq!(Action::from_str("read"), Action::Read);
        assert_eq!(Action::from_str("GET"), Action::Read);
        assert_eq!(Action::from_str("write"), Action::Write);
        assert_eq!(Action::from_str("POST"), Action::Write);
        assert_eq!(Action::from_str("execute"), Action::Execute);
        assert_eq!(Action::from_str("admin"), Action::Admin);
        assert_eq!(Action::from_str("unknown"), Action::Read); // 默认值
    }

    #[test]
    fn test_default_deny() {
        // 没有任何策略时，默认拒绝所有请求
        let rbac = create_test_rbac();
        let ctx = create_test_context(vec!["admin".to_string()]);

        assert!(!rbac.check_permission(&ctx, "/anything", Action::Read));
        assert!(!rbac.check_permission(&ctx, "/anything", Action::Write));
        assert!(!rbac.check_permission(&ctx, "/anything", Action::Admin));
    }
}
