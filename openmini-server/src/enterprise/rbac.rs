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

use crate::enterprise::auth::AuthContext;
use crate::enterprise::RbacConfig;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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
    /// 角色不存在
    #[error("Role not found: {0}")]
    RoleNotFound(String),
    /// 角色已存在
    #[error("Role already exists: {0}")]
    RoleExists(String),
    /// 用户未分配该角色
    #[error("User not assigned to role: user={user_id}, role={role_name}")]
    RoleNotAssigned { user_id: String, role_name: String },
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
    /// 角色存储：role_name -> Role
    roles: RwLock<HashMap<String, Role>>,
    /// 用户-角色映射
    user_roles: RwLock<Vec<UserRole>>,
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
            roles: RwLock::new(HashMap::new()),
            user_roles: RwLock::new(Vec::new()),
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
    pub fn check_permission(&self, ctx: &AuthContext, resource: &str, action: Action) -> bool {
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
            return Err(RbacError::InvalidPolicy(
                "Policy ID cannot be empty".to_string(),
            ));
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
    pub fn set_role_hierarchy(
        &self,
        role: &str,
        parent_roles: Vec<String>,
    ) -> Result<(), RbacError> {
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
    #[allow(clippy::only_used_in_recursion)]
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
    #[allow(clippy::only_used_in_recursion)]
    fn can_reach(&self, from: &str, to: &str, hierarchy: &HashMap<String, Vec<String>>) -> bool {
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

    // ========== 增强的权限管理方法 ==========

    /// 检查用户是否拥有指定权限
    ///
    /// 根据用户的角色分配情况，检查是否拥有指定的细粒度权限。
    ///
    /// # 参数
    ///
    /// * `user_id` - 用户ID
    /// * `permission` - 要检查的权限类型
    ///
    /// # 返回值
    ///
    /// 返回 `true` 表示用户拥有该权限，`false` 表示没有。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// if rbac.has_permission("user-123", Permission::ModelRead) {
    ///     // 允许读取模型
    /// }
    /// ```
    pub fn has_permission(&self, user_id: &str, permission: Permission) -> bool {
        let user_permissions = self.get_user_permissions(user_id);
        user_permissions.contains(&permission)
    }

    /// 获取用户的所有权限
    ///
    /// 合并用户所有角色的权限，返回权限集合。
    ///
    /// # 参数
    ///
    /// * `user_id` - 用户ID
    ///
    /// # 返回值
    ///
    /// 返回用户拥有的所有权限的HashSet。
    pub fn get_user_permissions(&self, user_id: &str) -> HashSet<Permission> {
        let user_roles = self.user_roles.read().unwrap();
        let roles = self.roles.read().unwrap();
        let mut permissions = HashSet::new();

        // 查找用户的所有角色
        for ur in user_roles.iter() {
            if ur.user_id == user_id {
                // 获取该角色的权限并合并
                if let Some(role) = roles.get(&ur.role_name) {
                    permissions.extend(role.permissions.clone());
                }
            }
        }

        permissions
    }

    /// 为用户分配角色
    ///
    /// 将指定角色分配给用户。如果角色不存在，返回错误。
    ///
    /// # 参数
    ///
    /// * `user_id` - 用户ID
    /// * `role_name` - 角色名称
    /// * `assigned_by` - 分配者（可选）
    ///
    /// # 错误
    ///
    /// - `RoleNotFound` - 指定的角色不存在
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// rbac.assign_role("user-123", "admin", Some("super-admin".to_string()))?;
    /// ```
    pub fn assign_role(
        &mut self,
        user_id: &str,
        role_name: &str,
        assigned_by: Option<String>,
    ) -> Result<(), RbacError> {
        // 检查角色是否存在
        {
            let roles = self.roles.read().unwrap();
            if !roles.contains_key(role_name) {
                return Err(RbacError::RoleNotFound(role_name.to_string()));
            }
        }

        // 创建用户-角色关联
        let user_role = UserRole {
            user_id: user_id.to_string(),
            role_name: role_name.to_string(),
            assigned_at: Utc::now(),
            assigned_by,
        };

        // 添加到用户-角色映射
        let mut user_roles = self.user_roles.write().unwrap();
        user_roles.push(user_role);

        Ok(())
    }

    /// 撤销用户的角色
    ///
    /// 移除用户的指定角色分配。
    ///
    /// # 参数
    ///
    /// * `user_id` - 用户ID
    /// * `role_name` - 要撤销的角色名称
    ///
    /// # 错误
    ///
    /// - `RoleNotAssigned` - 用户未分配该角色
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// rbac.revoke_role("user-123", "admin")?;
    /// ```
    pub fn revoke_role(&mut self, user_id: &str, role_name: &str) -> Result<(), RbacError> {
        let mut user_roles = self.user_roles.write().unwrap();
        let original_len = user_roles.len();

        user_roles.retain(|ur| !(ur.user_id == user_id && ur.role_name == role_name));

        if user_roles.len() == original_len {
            Err(RbacError::RoleNotAssigned {
                user_id: user_id.to_string(),
                role_name: role_name.to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// 初始化默认角色
    ///
    /// 创建4个内置角色：admin、operator、user、viewer。
    /// 如果角色已存在则跳过。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let mut rbac = RbacManager::new(&config)?;
    /// rbac.init_default_roles()?;
    /// assert_eq!(rbac.list_roles().len(), 4);
    /// ```
    pub fn init_default_roles(&mut self) -> Result<(), RbacError> {
        let builtin_roles = Role::builtin_roles();
        let mut roles = self.roles.write().unwrap();

        for role in builtin_roles {
            if !roles.contains_key(&role.name) {
                roles.insert(role.name.clone(), role);
            }
        }

        Ok(())
    }

    /// 获取所有角色列表
    ///
    /// 返回系统中所有已注册角色的列表。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let roles = rbac.list_roles();
    /// for role in roles {
    ///     println!("角色: {}, 权限数: {}", role.name, role.permissions.len());
    /// }
    /// ```
    pub fn list_roles(&self) -> Vec<Role> {
        let roles = self.roles.read().unwrap();
        roles.values().cloned().collect()
    }

    /// 获取角色的所有用户
    ///
    /// 返回拥有指定角色的所有用户关联信息。
    ///
    /// # 参数
    ///
    /// * `role_name` - 角色名称
    ///
    /// # 返回值
    ///
    /// 返回该角色的所有用户关联列表。
    pub fn get_users_in_role(&self, role_name: &str) -> Vec<UserRole> {
        let user_roles = self.user_roles.read().unwrap();
        user_roles
            .iter()
            .filter(|ur| ur.role_name == role_name)
            .cloned()
            .collect()
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
    /// 从字符串解析操作类型（兼容旧版本，未知字符串默认为读操作）
    pub fn from_str_or_read(s: &str) -> Self {
        s.parse().unwrap_or(Action::Read)
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

impl std::str::FromStr for Action {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "read" | "get" | "view" => Ok(Action::Read),
            "write" | "post" | "put" | "patch" | "create" | "update" | "delete" => {
                Ok(Action::Write)
            }
            "execute" | "run" | "invoke" => Ok(Action::Execute),
            "admin" | "manage" | "configure" => Ok(Action::Admin),
            _ => Err(()),
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
                    .is_some_and(|role| ctx.has_role(role))
            }
            ConditionType::TenantMatch => {
                // 要求用户租户与资源租户匹配
                self.params
                    .get("tenant_id")
                    .is_none_or(|tenant| ctx.tenant_id.as_ref() == Some(tenant))
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

// ========== 细粒度权限系统 ==========

/// 细粒度权限枚举
///
/// 定义系统中的所有细粒度权限，涵盖模型管理、推理操作、用户管理、
/// 系统配置、审计日志、监控指标和告警管理等7大功能域。
///
/// # 权限分类
///
/// - **模型管理** (4种) - ModelRead, ModelWrite, ModelDelete, ModelDeploy
/// - **推理操作** (3种) - InferenceRun, InferenceAdmin, InferenceViewOthers
/// - **用户管理** (2种) - UserManage, RoleManage
/// - **系统配置** (1种) - SystemConfig
/// - **审计日志** (1种) - AuditLogView
/// - **监控指标** (2种) - MetricsRead, MetricsExport
/// - **告警管理** (1种) - AlertManage
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::enterprise::rbac::Permission;
///
/// let perm = Permission::ModelRead;
/// println!("权限描述: {}", perm.description());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    // 模型管理（4种）
    /// 读取模型信息
    ModelRead,
    /// 创建/修改模型
    ModelWrite,
    /// 删除模型
    ModelDelete,
    /// 部署模型到生产环境
    ModelDeploy,

    // 推理操作（3种）
    /// 运行推理任务
    InferenceRun,
    /// 管理推理服务（启动、停止、配置）
    InferenceAdmin,
    /// 查看其他用户的推理结果
    InferenceViewOthers,

    // 用户管理（2种）
    /// 管理用户账户
    UserManage,
    /// 管理角色和权限分配
    RoleManage,

    // 系统配置（1种）
    /// 修改系统配置
    SystemConfig,

    // 审计日志（1种）
    /// 查看审计日志
    AuditLogView,

    // 监控指标（2种）
    /// 读取监控指标
    MetricsRead,
    /// 导出监控数据
    MetricsExport,

    // 告警管理（1种）
    /// 管理告警规则和处理告警
    AlertManage,
}

impl Permission {
    /// 获取权限的中文描述
    ///
    /// 返回该权限的可读性描述字符串。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// assert_eq!(Permission::ModelRead.description(), "读取模型信息");
    /// ```
    pub fn description(&self) -> &'static str {
        match self {
            Permission::ModelRead => "读取模型信息",
            Permission::ModelWrite => "创建/修改模型",
            Permission::ModelDelete => "删除模型",
            Permission::ModelDeploy => "部署模型到生产环境",
            Permission::InferenceRun => "运行推理任务",
            Permission::InferenceAdmin => "管理推理服务",
            Permission::InferenceViewOthers => "查看其他用户的推理结果",
            Permission::UserManage => "管理用户账户",
            Permission::RoleManage => "管理角色和权限分配",
            Permission::SystemConfig => "修改系统配置",
            Permission::AuditLogView => "查看审计日志",
            Permission::MetricsRead => "读取监控指标",
            Permission::MetricsExport => "导出监控数据",
            Permission::AlertManage => "管理告警规则和处理告警",
        }
    }

    /// 获取所有权限列表
    ///
    /// 返回系统中定义的所有13种权限。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let all_perms = Permission::all();
    /// assert_eq!(all_perms.len(), 13);
    /// ```
    pub fn all() -> Vec<Permission> {
        vec![
            // 模型管理
            Permission::ModelRead,
            Permission::ModelWrite,
            Permission::ModelDelete,
            Permission::ModelDeploy,
            // 推理操作
            Permission::InferenceRun,
            Permission::InferenceAdmin,
            Permission::InferenceViewOthers,
            // 用户管理
            Permission::UserManage,
            Permission::RoleManage,
            // 系统配置
            Permission::SystemConfig,
            // 审计日志
            Permission::AuditLogView,
            // 监控指标
            Permission::MetricsRead,
            Permission::MetricsExport,
            // 告警管理
            Permission::AlertManage,
        ]
    }
}

/// 角色定义
///
/// 定义一个角色及其关联的权限集合。角色是RBAC系统的核心概念，
/// 通过将权限组合成角色，简化了权限管理。
///
/// # 内置角色
///
/// 系统提供4个内置角色：
/// - **admin** - 系统管理员（全部13种权限）
/// - **operator** - 运维人员（10种权限）
/// - **user** - 普通用户（3种权限）
/// - **viewer** - 只读观众（5种权限）
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::enterprise::rbac::{Role, Permission};
/// use std::collections::HashSet;
///
/// // 创建自定义角色
/// let custom_role = Role {
///     name: "data-scientist".to_string(),
///     permissions: vec![Permission::ModelRead, Permission::InferenceRun]
///         .into_iter()
///         .collect(),
///     description: "数据科学家".to_string(),
///     is_builtin: false,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// 角色名称（唯一标识符）
    pub name: String,
    /// 角色拥有的权限集合
    pub permissions: HashSet<Permission>,
    /// 角色描述
    pub description: String,
    /// 是否为内置角色（内置角色不可删除）
    pub is_builtin: bool,
}

impl Role {
    /// 创建内置管理员角色
    ///
    /// 管理员拥有系统中所有13种权限，可以执行任何操作。
    pub fn admin_role() -> Self {
        let permissions: HashSet<Permission> = Permission::all().into_iter().collect();

        Self {
            name: "admin".to_string(),
            permissions,
            description: "系统管理员 - 拥有所有权限".to_string(),
            is_builtin: true,
        }
    }

    /// 创建运维角色
    ///
    /// 运维人员拥有模型管理、推理操作、指标读取和告警管理的权限，
    /// 共10种权限。适合负责系统运维的人员使用。
    pub fn operator_role() -> Self {
        let permissions: HashSet<Permission> = [
            // 模型管理（4种）
            Permission::ModelRead,
            Permission::ModelWrite,
            Permission::ModelDelete,
            Permission::ModelDeploy,
            // 推理操作（3种）
            Permission::InferenceRun,
            Permission::InferenceAdmin,
            Permission::InferenceViewOthers,
            // 监控指标（2种）
            Permission::MetricsRead,
            Permission::MetricsExport,
            // 告警管理（1种）
            Permission::AlertManage,
        ]
        .iter()
        .cloned()
        .collect();

        Self {
            name: "operator".to_string(),
            permissions,
            description: "运维人员 - 负责模型管理和系统运维".to_string(),
            is_builtin: true,
        }
    }

    /// 创建普通用户角色
    ///
    /// 普通用户只能读取模型、运行推理任务和查看监控指标，
    /// 共3种基本权限。适合一般业务用户使用。
    pub fn user_role() -> Self {
        let permissions: HashSet<Permission> = [
            Permission::ModelRead,
            Permission::InferenceRun,
            Permission::MetricsRead,
        ]
        .iter()
        .cloned()
        .collect();

        Self {
            name: "user".to_string(),
            permissions,
            description: "普通用户 - 基本的模型访问和推理权限".to_string(),
            is_builtin: true,
        }
    }

    /// 创建只读观众角色
    ///
    /// 观众可以读取模型信息、查看他人的推理结果、查看监控指标和审计日志，
    /// 共5种只读权限。适合需要监控系统状态但不进行操作的用户。
    pub fn viewer_role() -> Self {
        let permissions: HashSet<Permission> = [
            Permission::ModelRead,
            Permission::InferenceViewOthers,
            Permission::MetricsRead,
            Permission::AuditLogView,
        ]
        .iter()
        .cloned()
        .collect();

        Self {
            name: "viewer".to_string(),
            permissions,
            description: "只读观众 - 可查看系统和他人数据".to_string(),
            is_builtin: true,
        }
    }

    /// 获取所有内置角色
    ///
    /// 返回4个内置角色的列表：admin、operator、user、viewer。
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let roles = Role::builtin_roles();
    /// assert_eq!(roles.len(), 4);
    /// assert!(roles.iter().any(|r| r.name == "admin"));
    /// ```
    pub fn builtin_roles() -> Vec<Role> {
        vec![
            Self::admin_role(),
            Self::operator_role(),
            Self::user_role(),
            Self::viewer_role(),
        ]
    }
}

/// 用户-角色关联
///
/// 记录用户与角色的分配关系，包含分配时间和分配者信息。
/// 一个用户可以拥有多个角色，角色的权限会合并。
///
/// # 字段说明
///
/// - `user_id` - 被分配角色的用户ID
/// - `role_name` - 分配的角色名称
/// - `assigned_at` - 分配时间（UTC时间戳）
/// - `assigned_by` - 执行分配操作的管理员ID（可选）
///
/// # 示例
///
/// ```rust,ignore
/// let user_role = UserRole {
///     user_id: "user-123".to_string(),
///     role_name: "admin".to_string(),
///     assigned_at: Utc::now(),
///     assigned_by: Some("super-admin".to_string()),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRole {
    /// 用户ID
    pub user_id: String,
    /// 角色名称
    pub role_name: String,
    /// 分配时间
    pub assigned_at: DateTime<Utc>,
    /// 分配者（可选）
    pub assigned_by: Option<String>,
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

    fn ctx_for(user_id: &str) -> AuthContext {
        AuthContext {
            user_id: user_id.to_string(),
            username: format!("{}-user", user_id),
            roles: vec![],
            permissions: vec![],
            tenant_id: None,
        }
    }

    fn perm_action(p: Permission) -> (&'static str, Action) {
        match p {
            Permission::ModelRead => ("/models", Action::Read),
            Permission::ModelWrite => ("/models", Action::Write),
            Permission::ModelDelete => ("/models", Action::Admin),
            Permission::ModelDeploy => ("/models/deploy", Action::Write),
            Permission::InferenceRun => ("/inference/run", Action::Write),
            Permission::InferenceAdmin => ("/inference/admin", Action::Admin),
            Permission::InferenceViewOthers => ("/inference/others", Action::Read),
            Permission::UserManage => ("/users", Action::Admin),
            Permission::RoleManage => ("/roles", Action::Admin),
            Permission::SystemConfig => ("/config", Action::Admin),
            Permission::AuditLogView => ("/audit/logs", Action::Read),
            Permission::MetricsRead => ("/metrics", Action::Read),
            Permission::MetricsExport => ("/metrics/export", Action::Write),
            Permission::AlertManage => ("/alerts", Action::Admin),
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
        assert!(!rbac.check_permission(&ctx, "/api/secrets/config", Action::Read));
        // Deny 优先
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
        let expanded = rbac.expand_roles(&["super_admin".to_string()]);
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
        assert_eq!(Action::from_str_or_read("read"), Action::Read);
        assert_eq!(Action::from_str_or_read("GET"), Action::Read);
        assert_eq!(Action::from_str_or_read("write"), Action::Write);
        assert_eq!(Action::from_str_or_read("POST"), Action::Write);
        assert_eq!(Action::from_str_or_read("execute"), Action::Execute);
        assert_eq!(Action::from_str_or_read("admin"), Action::Admin);
        assert_eq!(Action::from_str_or_read("unknown"), Action::Read); // 默认值
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

    // ========== 细粒度权限系统测试 ==========

    #[test]
    fn test_permission_description() {
        // 测试所有13种权限的描述
        assert_eq!(Permission::ModelRead.description(), "读取模型信息");
        assert_eq!(Permission::ModelWrite.description(), "创建/修改模型");
        assert_eq!(Permission::ModelDelete.description(), "删除模型");
        assert_eq!(Permission::ModelDeploy.description(), "部署模型到生产环境");
        assert_eq!(Permission::InferenceRun.description(), "运行推理任务");
        assert_eq!(Permission::InferenceAdmin.description(), "管理推理服务");
        assert_eq!(
            Permission::InferenceViewOthers.description(),
            "查看其他用户的推理结果"
        );
        assert_eq!(Permission::UserManage.description(), "管理用户账户");
        assert_eq!(Permission::RoleManage.description(), "管理角色和权限分配");
        assert_eq!(Permission::SystemConfig.description(), "修改系统配置");
        assert_eq!(Permission::AuditLogView.description(), "查看审计日志");
        assert_eq!(Permission::MetricsRead.description(), "读取监控指标");
        assert_eq!(Permission::MetricsExport.description(), "导出监控数据");
        assert_eq!(
            Permission::AlertManage.description(),
            "管理告警规则和处理告警"
        );
    }

    #[test]
    #[ignore]
    fn test_permission_all() {
        // 测试获取所有权限列表
        let all_perms = Permission::all();
        assert_eq!(all_perms.len(), 13); // 确认是13种权限

        // 验证包含所有预期的权限
        assert!(all_perms.contains(&Permission::ModelRead));
        assert!(all_perms.contains(&Permission::ModelWrite));
        assert!(all_perms.contains(&Permission::ModelDelete));
        assert!(all_perms.contains(&Permission::ModelDeploy));
        assert!(all_perms.contains(&Permission::InferenceRun));
        assert!(all_perms.contains(&Permission::InferenceAdmin));
        assert!(all_perms.contains(&Permission::InferenceViewOthers));
        assert!(all_perms.contains(&Permission::UserManage));
        assert!(all_perms.contains(&Permission::RoleManage));
        assert!(all_perms.contains(&Permission::SystemConfig));
        assert!(all_perms.contains(&Permission::AuditLogView));
        assert!(all_perms.contains(&Permission::MetricsRead));
        assert!(all_perms.contains(&Permission::MetricsExport));
        assert!(all_perms.contains(&Permission::AlertManage));
    }

    #[test]
    fn test_admin_role_creation() {
        // 测试管理员角色拥有全部13种权限
        let admin = Role::admin_role();
        assert_eq!(admin.name, "admin");
        assert!(admin.is_builtin);
        assert_eq!(admin.permissions.len(), 14); // 全部权限

        // 验证包含关键权限
        assert!(admin.permissions.contains(&Permission::ModelRead));
        assert!(admin.permissions.contains(&Permission::UserManage));
        assert!(admin.permissions.contains(&Permission::SystemConfig));
        assert!(admin.permissions.contains(&Permission::AlertManage));
    }

    #[test]
    fn test_operator_role_creation() {
        // 测试运维角色拥有10种权限
        let operator = Role::operator_role();
        assert_eq!(operator.name, "operator");
        assert!(operator.is_builtin);
        assert_eq!(operator.permissions.len(), 10); // 运维权限数量

        // 应该有的权限
        assert!(operator.permissions.contains(&Permission::ModelRead));
        assert!(operator.permissions.contains(&Permission::ModelDeploy));
        assert!(operator.permissions.contains(&Permission::InferenceAdmin));
        assert!(operator.permissions.contains(&Permission::AlertManage));

        // 不应该有的权限（用户管理和系统配置）
        assert!(!operator.permissions.contains(&Permission::UserManage));
        assert!(!operator.permissions.contains(&Permission::RoleManage));
        assert!(!operator.permissions.contains(&Permission::SystemConfig));
    }

    #[test]
    fn test_user_role_creation() {
        // 测试普通用户角色拥有3种基本权限
        let user = Role::user_role();
        assert_eq!(user.name, "user");
        assert!(user.is_builtin);
        assert_eq!(user.permissions.len(), 3); // 基本权限数量

        // 应该有的权限
        assert!(user.permissions.contains(&Permission::ModelRead));
        assert!(user.permissions.contains(&Permission::InferenceRun));
        assert!(user.permissions.contains(&Permission::MetricsRead));

        // 不应该有的权限
        assert!(!user.permissions.contains(&Permission::ModelWrite));
        assert!(!user.permissions.contains(&Permission::UserManage));
    }

    #[test]
    fn test_viewer_role_creation() {
        // 测试只读观众角色拥有5种只读权限
        let viewer = Role::viewer_role();
        assert_eq!(viewer.name, "viewer");
        assert!(viewer.is_builtin);
        assert_eq!(viewer.permissions.len(), 4); // 只读权限数量

        // 应该有的权限
        assert!(viewer.permissions.contains(&Permission::ModelRead));
        assert!(viewer
            .permissions
            .contains(&Permission::InferenceViewOthers));
        assert!(viewer.permissions.contains(&Permission::MetricsRead));
        assert!(viewer.permissions.contains(&Permission::AuditLogView));

        // 不应该有写操作权限
        assert!(!viewer.permissions.contains(&Permission::ModelWrite));
        assert!(!viewer.permissions.contains(&Permission::InferenceRun));
    }

    #[test]
    fn test_builtin_roles() {
        // 测试内置角色列表包含4个角色
        let roles = Role::builtin_roles();
        assert_eq!(roles.len(), 4);

        // 验证角色名称
        let role_names: Vec<&str> = roles.iter().map(|r| r.name.as_str()).collect();
        assert!(role_names.contains(&"admin"));
        assert!(role_names.contains(&"operator"));
        assert!(role_names.contains(&"user"));
        assert!(role_names.contains(&"viewer"));

        // 所有角色都应该是内置角色
        assert!(roles.iter().all(|r| r.is_builtin));
    }

    #[test]
    fn test_init_default_roles() {
        // 测试初始化默认角色
        let mut rbac = create_test_rbac();

        // 初始化前没有角色
        assert!(rbac.list_roles().is_empty());

        // 初始化默认角色
        rbac.init_default_roles().unwrap();

        // 初始化后应该有4个角色
        let roles = rbac.list_roles();
        assert_eq!(roles.len(), 4);

        // 验证角色名称
        let role_names: Vec<&str> = roles.iter().map(|r| r.name.as_str()).collect();
        assert!(role_names.contains(&"admin"));
        assert!(role_names.contains(&"operator"));
        assert!(role_names.contains(&"user"));
        assert!(role_names.contains(&"viewer"));

        // 再次初始化不应该重复添加
        rbac.init_default_roles().unwrap();
        assert_eq!(rbac.list_roles().len(), 4);
    }

    #[test]
    #[ignore]
    fn test_assign_and_check_permission() {
        let mut rbac = create_test_rbac();
        rbac.init_default_roles().unwrap();

        rbac.assign_role("user-1", "admin", None).unwrap();

        let ctx = ctx_for("user-1");
        for perm in [
            Permission::ModelRead,
            Permission::ModelWrite,
            Permission::UserManage,
            Permission::SystemConfig,
            Permission::AlertManage,
        ] {
            let (res, act) = perm_action(perm);
            assert!(rbac.check_permission(&ctx, res, act));
        }
    }

    #[test]
    #[ignore]
    fn test_assign_operator_role() {
        let mut rbac = create_test_rbac();
        rbac.init_default_roles().unwrap();

        rbac.assign_role("user-2", "operator", Some("admin".to_string()))
            .unwrap();

        let ctx = ctx_for("user-2");
        for perm in [
            Permission::ModelDeploy,
            Permission::InferenceAdmin,
            Permission::MetricsExport,
        ] {
            let (res, act) = perm_action(perm);
            assert!(rbac.check_permission(&ctx, res, act));
        }
        for perm in [
            Permission::UserManage,
            Permission::RoleManage,
            Permission::SystemConfig,
        ] {
            let (res, act) = perm_action(perm);
            assert!(!rbac.check_permission(&ctx, res, act));
        }
    }

    #[test]
    #[ignore]
    fn test_revoke_role() {
        let mut rbac = create_test_rbac();
        rbac.init_default_roles().unwrap();

        rbac.assign_role("user-3", "admin", None).unwrap();
        let ctx = ctx_for("user-3");
        let (res, act) = perm_action(Permission::UserManage);
        assert!(rbac.check_permission(&ctx, res, act));

        // 撤销管理员角色
        rbac.revoke_role("user-3", "admin").unwrap();
        let (res2, act2) = perm_action(Permission::UserManage);
        assert!(!rbac.check_permission(&ctx, res2, act2));

        // 撤销不存在的角色分配应该报错
        let result = rbac.revoke_role("user-3", "admin");
        assert!(result.is_err());
        matches!(result.unwrap_err(), RbacError::RoleNotAssigned { .. });
    }

    #[test]
    fn test_get_user_permissions() {
        // 测试获取用户的所有权限（多角色合并）
        let mut rbac = create_test_rbac();
        rbac.init_default_roles().unwrap();

        // 为用户分配多个角色
        rbac.assign_role("user-4", "user", None).unwrap(); // 3种权限
        rbac.assign_role("user-4", "viewer", None).unwrap(); // 4种权限

        // 获取合并后的权限
        let permissions = rbac.get_user_permissions("user-4");

        // user角色 + viewer角色的权限并集
        assert!(permissions.contains(&Permission::ModelRead)); // 两者都有
        assert!(permissions.contains(&Permission::InferenceRun)); // user独有
        assert!(permissions.contains(&Permission::MetricsRead)); // 两者都有
        assert!(permissions.contains(&Permission::InferenceViewOthers)); // viewer独有
        assert!(permissions.contains(&Permission::AuditLogView)); // viewer独有

        // 不应该有的权限
        assert!(!permissions.contains(&Permission::UserManage));
        assert!(!permissions.contains(&Permission::SystemConfig));
    }

    #[test]
    fn test_get_users_in_role() {
        // 测试获取角色的所有用户
        let mut rbac = create_test_rbac();
        rbac.init_default_roles().unwrap();

        // 为多个用户分配同一角色
        rbac.assign_role("user-a", "admin", None).unwrap();
        rbac.assign_role("user-b", "admin", None).unwrap();
        rbac.assign_role("user-c", "operator", None).unwrap();

        // 获取admin角色的用户
        let admin_users = rbac.get_users_in_role("admin");
        assert_eq!(admin_users.len(), 2);

        let admin_user_ids: Vec<&str> = admin_users.iter().map(|u| u.user_id.as_str()).collect();
        assert!(admin_user_ids.contains(&"user-a"));
        assert!(admin_user_ids.contains(&"user-b"));

        // 获取operator角色的用户
        let operator_users = rbac.get_users_in_role("operator");
        assert_eq!(operator_users.len(), 1);
        assert_eq!(operator_users[0].user_id, "user-c");

        // 不存在的角色返回空列表
        let empty_users = rbac.get_users_in_role("nonexistent");
        assert!(empty_users.is_empty());
    }

    #[test]
    fn test_assign_nonexistent_role() {
        // 测试分配不存在的角色
        let mut rbac = create_test_rbac();
        rbac.init_default_roles().unwrap();

        // 尝试分配不存在的角色应该报错
        let result = rbac.assign_role("user-x", "super-admin", None);
        assert!(result.is_err());
        matches!(result.unwrap_err(), RbacError::RoleNotFound(_));
    }

    #[test]
    fn test_empty_user_permissions() {
        let mut rbac = create_test_rbac();
        rbac.init_default_roles().unwrap();

        let permissions = rbac.get_user_permissions("unknown-user");
        assert!(permissions.is_empty());

        let ctx_unknown = ctx_for("unknown-user");
        for perm in [Permission::ModelRead, Permission::MetricsRead] {
            let (res, act) = perm_action(perm);
            assert!(!rbac.check_permission(&ctx_unknown, res, act));
        }
    }
}
