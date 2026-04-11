use sqlx::SqlitePool;
use anyhow::Result;

const MIGRATIONS: &[&str] = &[
    r#"
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role INTEGER NOT NULL DEFAULT 0,
        is_active BOOLEAN NOT NULL DEFAULT 1,
        last_login_at DATETIME,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key_prefix TEXT NOT NULL,
        key_hash TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        owner_id INTEGER NOT NULL REFERENCES users(id),
        quota_daily_requests INTEGER,
        quota_monthly_tokens INTEGER,
        used_today_requests INTEGER NOT NULL DEFAULT 0,
        used_month_tokens INTEGER NOT NULL DEFAULT 0,
        is_active BOOLEAN NOT NULL DEFAULT 1,
        expires_at DATETIME,
        last_used_at DATETIME,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        revoked_at DATETIME
    );
    CREATE INDEX IF NOT EXISTS idx_api_keys_owner ON api_keys(owner_id);
    CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS alert_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        condition TEXT NOT NULL,
        threshold REAL NOT NULL,
        duration_seconds INTEGER NOT NULL DEFAULT 300,
        severity INTEGER NOT NULL DEFAULT 1,
        channels TEXT NOT NULL DEFAULT '[]',
        webhook_url TEXT,
        is_enabled BOOLEAN NOT NULL DEFAULT 1,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS alert_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_id INTEGER NOT NULL REFERENCES alert_rules(id),
        status INTEGER NOT NULL DEFAULT 0,
        severity INTEGER NOT NULL DEFAULT 1,
        message TEXT NOT NULL,
        value REAL NOT NULL,
        fired_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        acknowledged_at DATETIME,
        acknowledged_by INTEGER REFERENCES users(id),
        resolved_at DATETIME,
        resolved_by INTEGER REFERENCES users(id)
    );
    CREATE INDEX IF NOT EXISTS idx_alert_records_rule ON alert_records(rule_id);
    CREATE INDEX IF NOT EXISTS idx_alert_records_status ON alert_records(status);
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER REFERENCES users(id),
        action TEXT NOT NULL,
        resource_type TEXT,
        resource_id TEXT,
        detail TEXT,
        ip_address TEXT,
        user_agent TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id);
    CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
    CREATE INDEX IF NOT EXISTS idx_audit_logs_time ON audit_logs(created_at);
    "#,
    r#"
    CREATE TABLE IF NOT EXISTS config_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        changed_by INTEGER REFERENCES users(id),
        section TEXT NOT NULL,
        old_value TEXT,
        new_value TEXT,
        change_reason TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_config_history_section ON config_history(section);
    "#,
];

pub async fn init_pool(database_url: &str) -> Result<SqlitePool> {
    let pool = SqlitePool::connect(database_url).await?;

    sqlx::query("PRAGMA journal_mode=WAL")
        .execute(&pool)
        .await?;

    sqlx::query("PRAGMA foreign_keys=ON")
        .execute(&pool)
        .await?;

    for migration in MIGRATIONS {
        sqlx::query(migration).execute(&pool).await?;
    }

    Ok(pool)
}

// ============ 单元测试 ============

#[cfg(test)]
mod tests {
    use super::*;

    /// 创建内存数据库用于测试
    async fn create_memory_pool() -> SqlitePool {
        SqlitePool::connect(":memory:").await.unwrap()
    }

    #[tokio::test]
    async fn test_init_pool_creates_tables() {
        let pool = create_memory_pool().await;
        
        // 初始化数据库
        let result = init_pool_with_pool(&pool).await;
        assert!(result.is_ok());

        // 验证 users 表存在
        let table_exists: bool = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='users'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        
        assert!(table_exists, "users table should exist after migration");
    }

    #[tokio::test]
    async fn test_init_pool_all_tables_created() {
        let pool = create_memory_pool().await;
        init_pool_with_pool(&pool).await.unwrap();

        // 检查所有预期的表是否存在
        let expected_tables = vec![
            "users",
            "api_keys",
            "alert_rules",
            "alert_records",
            "audit_logs",
            "config_history"
        ];

        for table_name in expected_tables {
            let exists: bool = sqlx::query_scalar(
                &format!(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{}'",
                    table_name
                )
            )
            .fetch_one(&pool)
            .await
            .unwrap();
            
            assert!(exists, "{} table should exist", table_name);
        }
    }

    #[tokio::test]
    async fn test_migration_idempotent() {
        let pool = create_memory_pool().await;

        // 第一次执行迁移
        let result1 = init_pool_with_pool(&pool).await;
        assert!(result1.is_ok(), "First migration should succeed");

        // 第二次执行迁移（应该成功，因为使用了 IF NOT EXISTS）
        let result2 = init_pool_with_pool(&pool).await;
        assert!(result2.is_ok(), "Second migration should succeed (idempotent)");
    }

    #[tokio::test]
    async fn test_users_table_schema() {
        let pool = create_memory_pool().await;
        init_pool_with_pool(&pool).await.unwrap();

        // 验证 users 表的列
        let columns: Vec<(String, String)> = sqlx::query_as(
            "SELECT name, type FROM pragma_table_info('users') ORDER BY cid"
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        let column_names: Vec<&str> = columns.iter().map(|(name, _)| name.as_str()).collect();
        
        // 验证关键列存在
        assert!(column_names.contains(&"id"), "Should have id column");
        assert!(column_names.contains(&"username"), "Should have username column");
        assert!(column_names.contains(&"email"), "Should have email column");
        assert!(column_names.contains(&"password_hash"), "Should have password_hash column");
        assert!(column_names.contains(&"role"), "Should have role column");
        assert!(column_names.contains(&"is_active"), "Should have is_active column");
        assert!(column_names.contains(&"created_at"), "Should have created_at column");
        assert!(column_names.contains(&"updated_at"), "Should have updated_at column");
    }

    #[tokio::test]
    async fn test_api_keys_table_schema() {
        let pool = create_memory_pool().await;
        init_pool_with_pool(&pool).await.unwrap();

        // 验证 api_keys 表的列
        let columns: Vec<String> = sqlx::query_as::<_, (String,)>(
            "SELECT name FROM pragma_table_info('api_keys') ORDER BY cid"
        )
        .fetch_all(&pool)
        .await
        .unwrap()
        .into_iter()
        .map(|(name,)| name)
        .collect();

        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"key_prefix".to_string()));
        assert!(columns.contains(&"key_hash".to_string()));
        assert!(columns.contains(&"name".to_string()));
        assert!(columns.contains(&"owner_id".to_string()));
        assert!(columns.contains(&"quota_daily_requests".to_string()));
        assert!(columns.contains(&"is_active".to_string()));
    }

    #[tokio::test]
    async fn test_indexes_created() {
        let pool = create_memory_pool().await;
        init_pool_with_pool(&pool).await.unwrap();

        // 验证索引是否创建
        let indexes: Vec<String> = sqlx::query_as::<_, (String,)>(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        .fetch_all(&pool)
        .await
        .unwrap()
        .into_iter()
        .map(|(name,)| name)
        .collect();

        // 应该有这些索引（根据迁移脚本）
        assert!(indexes.iter().any(|i| i.contains("api_keys_owner")), "Should have index on api_keys.owner_id");
        assert!(indexes.iter().any(|i| i.contains("api_keys_prefix")), "Should have index on api_keys.key_prefix");
        assert!(indexes.iter().any(|i| i.contains("alert_records_rule")), "Should have index on alert_records.rule_id");
        assert!(indexes.iter().any(|i| i.contains("alert_records_status")), "Should have index on alert_records.status");
        assert!(indexes.iter().any(|i| i.contains("audit_logs_user")), "Should have index on audit_logs.user_id");
        assert!(indexes.iter().any(|i| i.contains("audit_logs_action")), "Should have index on audit_logs.action");
        assert!(indexes.iter().any(|i| i.contains("audit_logs_time")), "Should have index on audit_logs.created_at");
        assert!(indexes.iter().any(|i| i.contains("config_history_section")), "Should have index on config_history.section");
    }

    #[tokio::test]
    async fn test_foreign_key_constraints() {
        let pool = create_memory_pool().await;
        init_pool_with_pool(&pool).await.unwrap();

        // 验证外键约束是否启用
        let fk_enabled: i64 = sqlx::query_scalar("PRAGMA foreign_keys")
            .fetch_one(&pool)
            .await
            .unwrap();
        
        assert_eq!(fk_enabled, 1, "Foreign keys should be enabled");
    }

    #[tokio::test]
    async fn test_wal_mode_enabled() {
        // 注意：内存数据库不支持 WAL 模式，所以这个测试只验证文件数据库
        // 对于内存数据库，journal_mode 会返回 "memory"
        // 这个测试在实际的文件数据库上运行时才会返回 "wal"
        // 所以我们跳过这个测试或改为测试文件数据库
        
        // 创建临时文件数据库来测试 WAL 模式
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("test_wal_{}.db", std::process::id()));
        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        
        let pool = SqlitePool::connect(&db_url).await.unwrap();
        init_pool_with_pool(&pool).await.unwrap();

        // 验证 WAL 模式是否启用
        let journal_mode: String = sqlx::query_scalar("PRAGMA journal_mode")
            .fetch_one(&pool)
            .await
            .unwrap();
        
        assert_eq!(journal_mode.to_lowercase(), "wal", "Journal mode should be WAL");
        
        // 清理临时文件
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
        let _ = std::fs::remove_file(db_path.with_extension("db-shm"));
    }

    #[tokio::test]
    async fn test_can_insert_and_query_user() {
        let pool = create_memory_pool().await;
        init_pool_with_pool(&pool).await.unwrap();

        // 插入用户
        let result = sqlx::query(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)"
        )
        .bind("testuser")
        .bind("test@test.com")
        .bind("hashed_password")
        .bind(0i32)
        .execute(&pool)
        .await
        .unwrap();

        let user_id = result.last_insert_rowid();
        assert!(user_id > 0, "User ID should be positive");

        // 查询用户
        let (username, email): (String, String) = sqlx::query_as(
            "SELECT username, email FROM users WHERE id = ?"
        )
        .bind(user_id)
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(username, "testuser");
        assert_eq!(email, "test@test.com");
    }

    #[tokio::test]
    async fn test_can_insert_and_query_api_key() {
        let pool = create_memory_pool().await;
        init_pool_with_pool(&pool).await.unwrap();

        // 先插入一个 owner 用户
        let user_result = sqlx::query(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)"
        )
        .bind("owner")
        .bind("owner@test.com")
        .bind("hash")
        .bind(0i32)
        .execute(&pool)
        .await
        .unwrap();
        let owner_id = user_result.last_insert_rowid();

        // 插入 API key
        let key_result = sqlx::query(
            "INSERT INTO api_keys (key_prefix, key_hash, name, owner_id) VALUES (?, ?, ?, ?)"
        )
        .bind("om-sk_test12345")
        .bind("hash_value_1234567890123456789012345678901234567890123456789012345678")
        .bind("Test Key")
        .bind(owner_id)
        .execute(&pool)
        .await
        .unwrap();

        let key_id = key_result.last_insert_rowid();
        assert!(key_id > 0);

        // 查询 API key
        let (name, key_owner_id): (String, i64) = sqlx::query_as(
            "SELECT name, owner_id FROM api_keys WHERE id = ?"
        )
        .bind(key_id)
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(name, "Test Key");
        assert_eq!(key_owner_id, owner_id);
    }

    /// 辅助函数：对已有的 pool 执行迁移（不创建新连接）
    async fn init_pool_with_pool(pool: &SqlitePool) -> Result<()> {
        sqlx::query("PRAGMA journal_mode=WAL")
            .execute(pool)
            .await?;

        sqlx::query("PRAGMA foreign_keys=ON")
            .execute(pool)
            .await?;

        for migration in MIGRATIONS {
            sqlx::query(migration).execute(pool).await?;
        }

        Ok(())
    }
}
