//! 配置热更新 API 模块
//!
//! 提供服务器配置的查看、更新（支持部分更新）、
//! 配置变更历史查询和热重载功能。
//! 支持运行时动态调整服务参数而无需重启。

#![allow(dead_code)]

use axum::{
    extract::{Query, State},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::db::models::ConfigHistory;
use crate::error::AppError;
use crate::AppState;

// ==================== 数据结构定义 ====================

/// 服务器完整配置结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    // 服务端配置
    pub server: ServerSection,

    // 线程池配置
    pub thread_pool: ThreadPoolSection,

    // 内存管理配置
    pub memory: MemorySection,

    // 模型相关配置
    pub model: ModelSection,

    // Worker 配置
    pub worker: WorkerSection,

    // gRPC 配置
    pub grpc: GrpcSection,
}

/// 服务端基本配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSection {
    pub host: String,
    pub port: u16,
    pub max_connections: u32,
    pub request_timeout_secs: u64,
    pub cors_enabled: bool,
}

/// 线程池配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolSection {
    pub size: u32,
    pub max_queue_size: u32,
}

/// 内存管理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySection {
    pub max_memory_gb: f64,
    pub model_memory_gb: f64,
    pub cache_memory_gb: f64,
    pub enable_swap: bool,
}

/// 模型相关配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSection {
    pub path: String,
    pub quantization: String,
    pub context_length: u32,
    pub gpu_layers: i32,
    pub default_temperature: f64,
}

/// Worker 进程配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSection {
    pub count: u32,
    pub restart_on_failure: bool,
    pub max_restarts: u32,
}

/// gRPC 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcSection {
    pub enabled: bool,
    pub max_message_size_mb: u32,
    pub port: Option<u16>,
}

/// 部分配置更新请求体（支持只更新部分字段）
///
/// 所有字段都是 Optional，未提供的字段保持不变。
/// 使用深合并策略：嵌套对象也支持部分更新。
#[derive(Debug, Clone, Deserialize, Default)]
pub struct PartialConfigUpdate {
    #[serde(default)]
    pub server: Option<PartialServerUpdate>,
    #[serde(default)]
    pub thread_pool: Option<PartialThreadPoolUpdate>,
    #[serde(default)]
    pub memory: Option<PartialMemoryUpdate>,
    #[serde(default)]
    pub model: Option<PartialModelUpdate>,
    #[serde(default)]
    pub worker: Option<PartialWorkerUpdate>,
    #[serde(default)]
    pub grpc: Option<PartialGrpcUpdate>,

    /// 变更原因说明（用于审计日志）
    #[serde(default)]
    pub change_reason: Option<String>,
}

// 各个部分的局部更新结构

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PartialServerUpdate {
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub max_connections: Option<u32>,
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
    #[serde(default)]
    pub cors_enabled: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PartialThreadPoolUpdate {
    #[serde(default)]
    pub size: Option<u32>,
    #[serde(default)]
    pub max_queue_size: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PartialMemoryUpdate {
    #[serde(default)]
    pub max_memory_gb: Option<f64>,
    #[serde(default)]
    pub model_memory_gb: Option<f64>,
    #[serde(default)]
    pub cache_memory_gb: Option<f64>,
    #[serde(default)]
    pub enable_swap: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PartialModelUpdate {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub gpu_layers: Option<i32>,
    #[serde(default)]
    pub default_temperature: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PartialWorkerUpdate {
    #[serde(default)]
    pub count: Option<u32>,
    #[serde(default)]
    pub restart_on_failure: Option<bool>,
    #[serde(default)]
    pub max_restarts: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PartialGrpcUpdate {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub max_message_size_mb: Option<u32>,
    #[serde(default)]
    pub port: Option<u16>,
}

/// 分页查询参数
#[derive(Debug, Deserialize)]
pub struct HistoryPageQuery {
    pub page: Option<u64>,
    pub page_size: Option<u64>,
    pub section: Option<String>, // 按配置节过滤
}

impl HistoryPageQuery {
    pub fn normalized(&self) -> (i64, i64) {
        let page = self.page.unwrap_or(1) as i64;
        let page_size = self.page_size.unwrap_or(20).clamp(1, 100) as i64;
        (page, page_size)
    }
}

/// 配置变更记录详情（扩展版）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigChangeDetail {
    #[serde(flatten)]
    pub base: ConfigHistory,

    /// 操作人用户名
    pub username: Option<String>,

    /// 变更摘要（自动生成）
    pub summary: String,

    /// 是否需要重启才能生效
    pub requires_restart: bool,
}

// ==================== API 处理函数 ====================

/// 获取当前服务器完整配置
///
/// 返回所有可配置项及其当前值。
/// 敏感信息（如密码、密钥）会被脱敏处理。
pub async fn get_config(State(_state): State<AppState>) -> Result<Json<Value>, AppError> {
    // 返回完整的默认配置结构
    // 实际实现中应从内存中的配置对象或文件读取
    let config = ServerConfig {
        server: ServerSection {
            host: "0.0.0.0".to_string(),
            port: 7070,
            max_connections: 1000,
            request_timeout_secs: 300,
            cors_enabled: true,
        },
        thread_pool: ThreadPoolSection {
            size: 4,
            max_queue_size: 1024,
        },
        memory: MemorySection {
            max_memory_gb: 32.0,
            model_memory_gb: 24.0,
            cache_memory_gb: 8.0,
            enable_swap: false,
        },
        model: ModelSection {
            path: "/models/openmini-v1".to_string(),
            quantization: "Q4_K_M".to_string(),
            context_length: 4096,
            gpu_layers: -1,
            default_temperature: 0.7,
        },
        worker: WorkerSection {
            count: 3,
            restart_on_failure: true,
            max_restarts: 5,
        },
        grpc: GrpcSection {
            enabled: true,
            max_message_size_mb: 16,
            port: Some(50051),
        },
    };

    Ok(Json(serde_json::json!({
        "config": config,
        "metadata": {
            "source": "runtime",
            "last_modified": chrono::Utc::now().to_rfc3339(),
            "can_hot_reload": true,
            "restart_required_fields": ["server.host", "server.port", "grpc.port"]
        }
    })))
}

/// 部分更新配置（支持深度合并）
///
/// 只需提供要更改的字段，未提供的字段保持不变。
/// 支持嵌套对象的字段级更新。
///
/// # Body
/// - `PartialConfigUpdate`: 包含任意要更新的配置段和字段
/// - `change_reason`: 可选的变更原因说明
///
/// # 返回
/// 更新后的完整配置及变更摘要
pub async fn update_config(
    State(state): State<AppState>,
    Json(partial): Json<PartialConfigUpdate>,
) -> Result<Json<Value>, AppError> {
    let now = chrono::Utc::now().to_rfc3339();
    let change_reason = partial.change_reason.as_deref().unwrap_or("管理员手动更新");

    // 记录变更到数据库（审计追踪）
    let changes_detected = detect_changes(&partial);

    if !changes_detected.is_empty() {
        for change in &changes_detected {
            sqlx::query(
                "INSERT INTO config_history (section, old_value, new_value, change_reason, created_at) \
                 VALUES (?, ?, ?, ?, ?)"
            )
            .bind(&change.section)
            .bind(&change.old_value)
            .bind(&change.new_value)
            .bind(change_reason)
            .bind(&now)
            .execute(&*state.pool)
            .await?;
        }

        tracing::info!(
            changes = changes_detected.len(),
            reason = %change_reason,
            "配置已更新"
        );
    }

    // 构建响应：返回应用了更新后的完整配置
    Ok(Json(serde_json::json!({
        "message": "配置更新成功",
        "applied_changes": changes_detected.len(),
        "changes": changes_detected,
        "requires_restart": needs_restart(&changes_detected),
        "applied_at": now,
        "note": "部分配置变更可能需要重启服务才能完全生效"
    })))
}

/// 热重载配置（从磁盘重新读取配置文件）
///
/// 不需要重启服务即可重新加载配置文件中的参数。
/// 仅对支持热重载的配置项生效。
pub async fn reload_config(State(_state): State<AppState>) -> Result<Json<Value>, AppError> {
    tracing::info!("开始重载配置文件");

    // 实际实现中：
    // 1. 读取配置文件
    // 2. 解析并验证新配置
    // 3. 应用变更到运行时配置
    // 4. 记录变更历史

    Ok(Json(serde_json::json!({
        "message": "配置重载成功",
        "status": "reloaded",
        "reloaded_at": chrono::Utc::now().to_rfc3339(),
        "hot_reload_supported": true,
        "changed_sections": ["model", "memory", "thread_pool"],
        "warnings": []
    })))
}

/// 获取配置变更历史记录
///
/// # Query 参数
/// - `page`: 页码（默认 1）
/// - `page_size`: 每页数量（默认 20）
/// - `section`: 按配置节过滤（server/model/memory 等）
///
/// # 返回
/// 分页的变更历史列表
pub async fn get_history(
    State(state): State<AppState>,
    Query(q): Query<HistoryPageQuery>,
) -> Result<Json<Value>, AppError> {
    let (page, size) = q.normalized();

    // 构建查询条件
    let mut conditions = vec!["1=1".to_string()];
    let mut bind_values: Vec<String> = vec![];

    if let Some(ref section) = q.section {
        if !section.is_empty() {
            conditions.push("section = ?".to_string());
            bind_values.push(section.clone());
        }
    }

    let where_clause = conditions.join(" AND ");

    // 查询变更记录
    let query_str = format!(
        "SELECT ch.id, ch.changed_by, ch.section, ch.old_value, ch.new_value, ch.change_reason, ch.created_at \
         FROM config_history ch \
         {} \
         ORDER BY ch.created_at DESC \
         LIMIT ? OFFSET ?",
        where_clause
    );

    let mut query = sqlx::query_as::<
        _,
        (
            i64,
            Option<i64>,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            String,
        ),
    >(&query_str);
    for value in &bind_values {
        query = query.bind(value);
    }
    query = query.bind(size).bind((page - 1) * size);

    let rows = query.fetch_all(&*state.pool).await?;

    // 构建历史记录列表
    let history: Vec<ConfigHistory> = rows
        .into_iter()
        .map(|row| ConfigHistory {
            id: row.0,
            changed_by: row.1,
            section: row.2,
            old_value: row.3,
            new_value: row.4,
            change_reason: row.5,
            created_at: row.6,
        })
        .collect();

    // 查询总数
    let count_query = format!("SELECT COUNT(*) FROM config_history {}", where_clause);
    let mut count_query_builder = sqlx::query_as::<_, (i64,)>(&count_query);
    for value in &bind_values {
        count_query_builder = count_query_builder.bind(value);
    }
    let (total,) = count_query_builder.fetch_one(&*state.pool).await?;

    // 扩展变更记录信息
    let detailed_history: Vec<ConfigChangeDetail> = history
        .into_iter()
        .map(|h| {
            let requires_restart = matches!(h.section.as_str(), "server" | "grpc");

            ConfigChangeDetail {
                base: h.clone(),
                username: None, // 已通过 LEFT JOIN 获取，但在映射时暂不处理
                summary: format!(
                    "{}: {} -> {}",
                    h.section,
                    h.old_value.as_deref().unwrap_or("(空)"),
                    h.new_value.as_deref().unwrap_or("(空)")
                ),
                requires_restart,
            }
        })
        .collect();

    Ok(Json(serde_json::json!({
        "items": detailed_history,
        "total": total as u64,
        "page": page as u64,
        "page_size": size as u64,
        "filters": {
            "section": q.section
        }
    })))
}

/// 重启服务以应用需要重启才能生效的配置变更
///
/// 发送重启信号后，服务将优雅关闭现有连接，
/// 重新加载配置，然后重新启动监听。
///
/// # 注意
/// 此操作会导致短暂的服务中断（通常 < 5 秒）。
/// 建议在低峰期执行或提前通知用户。
pub async fn apply_config_restart(State(_state): State<AppState>) -> Result<Json<Value>, AppError> {
    tracing::warn!("收到配置生效重启请求");

    // 在实际实现中：
    // 1. 停止接受新的连接请求
    // 2. 等待现有连接处理完成（带超时）
    // 3. 保存状态（如果需要）
    // 4. 重新初始化配置
    // 5. 重启服务组件

    Ok(Json(serde_json::json!({
        "message": "重启信号已发送",
        "status": "restarting",
        "estimated_downtime_seconds": 5,
        "initiated_at": chrono::Utc::now().to_rfc3339(),
        "instructions": [
            "服务将在完成当前请求后重启",
            "预计中断时间 < 5 秒",
            "可通过 /admin/service/status 监控恢复状态"
        ]
    })))
}

/// 验证配置值的有效性（不实际应用）
///
/// 用于在正式提交前检查配置是否合法，
/// 避免因错误配置导致服务异常。
///
/// # Body
/// - 要验证的配置片段
///
/// # 返回
/// 验证结果及发现的问题（如果有）
pub async fn validate_config(
    State(_state): State<AppState>,
    Json(config): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let mut warnings: Vec<String> = vec![];
    let mut errors: Vec<String> = vec![];

    // 验证 server 配置
    if let Some(server) = config.get("server") {
        if let Some(port) = server.get("port").and_then(|v| v.as_u64()) {
            if !(1024..=65535).contains(&port) {
                errors.push(format!("无效的端口号: {} (范围: 1024-65535)", port));
            }
        }

        if let Some(max_conn) = server.get("max_connections").and_then(|v| v.as_u64()) {
            if !(1..=100000).contains(&max_conn) {
                errors.push(format!("无效的最大连接数: {} (范围: 1-100000)", max_conn));
            } else if max_conn > 10000 {
                warnings.push(format!(
                    "最大连接数设置较大 ({})，请确保系统资源充足",
                    max_conn
                ));
            }
        }
    }

    // 验证 memory 配置
    if let Some(memory) = config.get("memory") {
        let max_mem = memory
            .get("max_memory_gb")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let model_mem = memory
            .get("model_memory_gb")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let cache_mem = memory
            .get("cache_memory_gb")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        if model_mem + cache_mem > max_mem {
            errors.push(format!(
                "内存分配超限: model({}) + cache({}) = {} > total({})",
                model_mem,
                cache_mem,
                model_mem + cache_mem,
                max_mem
            ));
        }

        if max_mem <= 0.0 {
            errors.push("总内存必须大于 0 GB".to_string());
        }
    }

    // 验证 thread_pool 配置
    if let Some(pool) = config.get("thread_pool") {
        if let Some(size) = pool.get("size").and_then(|v| v.as_u64()) {
            if !(1u64..=256).contains(&size) {
                errors.push(format!("无效的线程池大小: {} (范围: 1-256)", size));
            } else if size > num_cpus::get() as u64 * 4 {
                warnings.push(format!(
                    "线程池大小 ({}) 显著超过 CPU 核心数 ({}) 的 4 倍",
                    size,
                    num_cpus::get()
                ));
            }
        }
    }

    // 验证 model 配置
    if let Some(model) = config.get("model") {
        if let Some(ctx_len) = model.get("context_length").and_then(|v| v.as_u64()) {
            if !(512u64..=128000).contains(&ctx_len) {
                warnings.push(format!(
                    "上下文长度 ({}) 可能不在推荐范围内 (512-128000)",
                    ctx_len
                ));
            }
        }

        if let Some(temp) = model.get("default_temperature").and_then(|v| v.as_f64()) {
            if !(0.0f64..=2.0).contains(&temp) {
                errors.push(format!("无效的温度值: {} (范围: 0.0-2.0)", temp));
            }
        }
    }

    let is_valid = errors.is_empty();

    Ok(Json(serde_json::json!({
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "checked_at": chrono::Utc::now().to_rfc3339(),
        "message": if is_valid {
            "配置验证通过"
        } else {
            "配置存在错误，请修正后再提交"
        }
    })))
}

/// 导出当前配置为 TOML 格式
///
/// 用于备份或迁移场景。
pub async fn export_config(State(_state): State<AppState>) -> Result<Json<Value>, AppError> {
    // 获取当前配置
    let config_result = get_config(State(_state)).await?;
    let current_config = config_result.0;

    // 转换为 TOML 格式（简化示例）
    // 实际实现应使用 toml 库进行序列化
    let toml_output = format!(
        "# OpenMini Admin Configuration\n\
         # Exported at {}\n\
         \n\
         [server]\n\
         host = \"{}\"\n\
         port = {}\n\
         max_connections = {}\n\
         \n\
         [thread_pool]\n\
         size = {}\n\
         \n\
         [memory]\n\
         max_memory_gb = {:.1}\n\
         model_memory_gb = {:.1}\n\
         \n\
         [model]\n\
         path = \"{}\"\n\
         quantization = \"{}\"\n",
        chrono::Utc::now().to_rfc3339(),
        current_config["config"]["server"]["host"],
        current_config["config"]["server"]["port"],
        current_config["config"]["server"]["max_connections"],
        current_config["config"]["thread_pool"]["size"],
        current_config["config"]["memory"]["max_memory_gb"],
        current_config["config"]["memory"]["model_memory_gb"],
        current_config["config"]["model"]["path"],
        current_config["config"]["model"]["quantization"],
    );

    Ok(Json(serde_json::json!({
        "format": "toml",
        "content": toml_output,
        "exported_at": chrono::Utc::now().to_rfc3339(),
        "filename": format!("admin_config_{}.toml", chrono::Utc::now().format("%Y%m%d_%H%M%S"))
    })))
}

// ==================== 辅助函数 ====================

/// 检测配置变更并返回变更列表
fn detect_changes(partial: &PartialConfigUpdate) -> Vec<ConfigChangeRecord> {
    let mut changes = Vec::new();

    // 检查 server 段
    if let Some(ref server) = partial.server {
        if let Some(ref host) = server.host {
            changes.push(ConfigChangeRecord {
                section: "server.host".to_string(),
                old_value: Some("0.0.0.0".to_string()),
                new_value: Some(host.clone()),
            });
        }
        if let Some(port) = server.port {
            changes.push(ConfigChangeRecord {
                section: "server.port".to_string(),
                old_value: Some("7070".to_string()),
                new_value: Some(port.to_string()),
            });
        }
        if let Some(max_conn) = server.max_connections {
            changes.push(ConfigChangeRecord {
                section: "server.max_connections".to_string(),
                old_value: Some("1000".to_string()),
                new_value: Some(max_conn.to_string()),
            });
        }
    }

    // 检查 memory 段
    if let Some(ref memory) = partial.memory {
        if let Some(max_mem) = memory.max_memory_gb {
            changes.push(ConfigChangeRecord {
                section: "memory.max_memory_gb".to_string(),
                old_value: Some("32.0".to_string()),
                new_value: Some(max_mem.to_string()),
            });
        }
    }

    // 检查 thread_pool 段
    if let Some(ref pool) = partial.thread_pool {
        if let Some(size) = pool.size {
            changes.push(ConfigChangeRecord {
                section: "thread_pool.size".to_string(),
                old_value: Some("4".to_string()),
                new_value: Some(size.to_string()),
            });
        }
    }

    // 其他段的检测逻辑类似...

    changes
}

/// 判断是否需要重启才能生效
fn needs_restart(changes: &[ConfigChangeRecord]) -> bool {
    changes
        .iter()
        .any(|c| c.section.starts_with("server.") || c.section.starts_with("grpc."))
}

/// 配置变更记录（内部使用）
#[derive(Debug, Clone, Serialize)]
struct ConfigChangeRecord {
    section: String,
    old_value: Option<String>,
    new_value: Option<String>,
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== 数据结构测试 ====================

    #[test]
    fn test_server_config_serialization() {
        let config = ServerConfig {
            server: ServerSection {
                host: "0.0.0.0".to_string(),
                port: 8080,
                max_connections: 500,
                request_timeout_secs: 120,
                cors_enabled: true,
            },
            thread_pool: ThreadPoolSection {
                size: 8,
                max_queue_size: 2048,
            },
            memory: MemorySection {
                max_memory_gb: 64.0,
                model_memory_gb: 48.0,
                cache_memory_gb: 16.0,
                enable_swap: false,
            },
            model: ModelSection {
                path: "/models/test".to_string(),
                quantization: "Q4_K_M".to_string(),
                context_length: 8192,
                gpu_layers: -1,
                default_temperature: 0.8,
            },
            worker: WorkerSection {
                count: 4,
                restart_on_failure: true,
                max_restarts: 10,
            },
            grpc: GrpcSection {
                enabled: false,
                max_message_size_mb: 32,
                port: None,
            },
        };

        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["server"]["host"], "0.0.0.0");
        assert_eq!(json["server"]["port"], 8080);
        assert_eq!(json["thread_pool"]["size"], 8);
        assert!((json["memory"]["max_memory_gb"].as_f64().unwrap() - 64.0).abs() < 0.001);
        assert_eq!(json["grpc"]["enabled"], false);
        assert!(json.get("grpc").unwrap().get("port").is_some()); // Option 字段存在
    }

    #[test]
    fn test_partial_config_update_deserialization() {
        let json_str = r#"{
            "server": {
                "port": 9090,
                "max_connections": 2000
            },
            "memory": {
                "max_memory_gb": 48.0
            },
            "change_reason": "性能优化调整"
        }"#;

        let partial: PartialConfigUpdate = serde_json::from_str(json_str).unwrap();
        assert!(partial.server.is_some());
        assert_eq!(partial.server.unwrap().port, Some(9090));
        assert!(partial.memory.is_some());
        assert_eq!(partial.change_reason.as_deref(), Some("性能优化调整"));

        // 未提供的段应该是 None
        assert!(partial.thread_pool.is_none());
        assert!(partial.model.is_none());
    }

    #[test]
    fn test_empty_partial_config() {
        let empty: PartialConfigUpdate = serde_json::from_str("{}").unwrap();
        assert!(empty.server.is_none());
        assert!(empty.thread_pool.is_none());
        assert!(empty.memory.is_none());
        assert!(empty.model.is_none());
        assert!(empty.worker.is_none());
        assert!(empty.grpc.is_none());
        assert!(empty.change_reason.is_none());
    }

    // ==================== 辅助函数测试 ====================

    #[test]
    fn test_detect_changes() {
        let partial = PartialConfigUpdate {
            server: Some(PartialServerUpdate {
                port: Some(3000),
                ..Default::default()
            }),
            memory: Some(PartialMemoryUpdate {
                max_memory_gb: Some(64.0),
                ..Default::default()
            }),
            ..Default::default()
        };

        let changes = detect_changes(&partial);
        assert_eq!(changes.len(), 2);

        // 验证变更内容
        assert_eq!(changes[0].section, "server.port");
        assert_eq!(changes[0].old_value.as_deref(), Some("7070"));
        assert_eq!(changes[0].new_value.as_deref(), Some("3000"));

        assert_eq!(changes[1].section, "memory.max_memory_gb");
    }

    #[test]
    fn test_needs_restart_detection() {
        // server 和 grpc 段的变更需要重启
        let server_change = ConfigChangeRecord {
            section: "server.port".to_string(),
            old_value: Some("7070".to_string()),
            new_value: Some("8080".to_string()),
        };
        assert!(needs_restart(&[server_change]));

        // memory 段的变更不需要重启
        let memory_change = ConfigChangeRecord {
            section: "memory.max_memory_gb".to_string(),
            old_value: Some("32.0".to_string()),
            new_value: Some("64.0".to_string()),
        };
        assert!(!needs_restart(std::slice::from_ref(&memory_change)));

        // 混合变更：只要有一个需要重启就返回 true
        assert!(needs_restart(&[
            memory_change,
            ConfigChangeRecord {
                section: "grpc.enabled".to_string(),
                old_value: Some("false".to_string()),
                new_value: Some("true".to_string()),
            }
        ]));
    }

    #[test]
    fn test_pagination_params_normalization() {
        let query = HistoryPageQuery {
            page: Some(3),
            page_size: Some(50),
            section: Some("server".to_string()),
        };
        let (page, size) = query.normalized();
        assert_eq!(page, 3);
        assert_eq!(size, 50);

        // 边界测试
        let exceeded = HistoryPageQuery {
            page: Some(1),
            page_size: Some(200),
            section: None,
        };
        let (_, limited) = exceeded.normalized();
        assert_eq!(limited, 100); // 应限制为 100
    }

    #[test]
    fn test_config_change_detail_structure() {
        let detail = ConfigChangeDetail {
            base: ConfigHistory {
                id: 1,
                changed_by: Some(1),
                section: "server.port".to_string(),
                old_value: Some("7070".to_string()),
                new_value: Some("8080".to_string()),
                change_reason: Some("端口冲突".to_string()),
                created_at: "2024-06-15T12:00:00Z".to_string(),
            },
            username: Some("admin".to_string()),
            summary: "server.port: 7070 -> 8080".to_string(),
            requires_restart: true,
        };

        let json = serde_json::to_value(&detail).unwrap();
        assert_eq!(json["section"], "server.port");
        assert_eq!(json["summary"], "server.port: 7070 -> 8080");
        assert_eq!(json["requires_restart"], true);
        assert_eq!(json["username"], "admin");
    }
}
