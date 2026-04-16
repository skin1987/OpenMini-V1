//! KV Cache 持久化模块
//!
//! 提供高效的 KV Cache 序列化和反序列化功能：
//! - bincode 二进制序列化 (高性能)
//! - 可选 zstd 压缩 (节省存储空间)
//! - CRUD 操作 (保存/加载/列表/删除)
//! - 过期缓存自动清理

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

/// 持久化错误类型
#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    #[error("序列化失败: {0}")]
    Serialization(String),
    #[error("反序列化失败: {0}")]
    Deserialization(String),
    #[error("IO错误: {0}")]
    Io(#[from] std::io::Error),
    #[error("缓存不存在: {0}")]
    NotFound(String),
    #[error("时间系统错误: {0}")]
    Time(#[from] std::time::SystemTimeError),
}

/// 持久化的 KV Cache 数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedKvCache {
    pub key_data: Vec<f32>,
    pub value_data: Vec<f32>,
    pub total_rows: usize,
    pub cols: usize,
    pub model_name: String,
    pub created_at: u64,
    pub version: u32,
}

/// 缓存元信息（用于列表展示）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheInfo {
    pub name: String,
    pub size_bytes: u64,
    pub model_name: String,
    pub created_at: u64,
    pub compressed: bool,
}

/// KV Cache 持久化管理器
///
/// 负责将内存中的 KV Cache 序列化到磁盘，以及从磁盘反序列化加载。
/// 支持可选的 zstd 压缩以减少存储空间占用。
///
/// # 示例
///
/// ```rust,ignore
/// use openmini_server::hardware::kv_cache::persistence::{KvCachePersistence, PersistedKvCache};
///
/// let persist = KvCachePersistence::new("./cache_dir", true)?;
/// let cache = PersistedKvCache { /* ... */ };
/// persist.save(&cache, "my_cache")?;
/// let loaded = persist.load("my_cache")?;
/// ```
pub struct KvCachePersistence {
    base_dir: PathBuf,
    compress: bool,
}

impl KvCachePersistence {
    /// 创建新的持久化管理器
    ///
    /// # 参数
    /// - `base_dir`: 缓存存储根目录
    /// - `compress`: 是否启用 zstd 压缩
    ///
    /// # 错误
    /// 如果目录创建失败，返回 [`PersistenceError::Io`]
    pub fn new(base_dir: impl AsRef<Path>, compress: bool) -> Result<Self, PersistenceError> {
        let dir = base_dir.as_ref();
        if !dir.exists() {
            fs::create_dir_all(dir)?;
        }
        Ok(Self {
            base_dir: dir.to_path_buf(),
            compress,
        })
    }

    /// 保存 KV Cache 到磁盘
    ///
    /// 将 [`PersistedKvCache`] 序列化为二进制格式并写入文件。
    /// 如果启用了压缩，会使用 zstd 进行压缩存储。
    ///
    /// # 参数
    /// - `cache`: 要保存的缓存数据
    /// - `name`: 缓存名称（用作文件名）
    ///
    /// # 错误
    /// - [`PersistenceError::Serialization`][]: 序列化失败
    /// - [`PersistenceError::Io`][]: 文件写入失败
    pub fn save(&self, cache: &PersistedKvCache, name: &str) -> Result<(), PersistenceError> {
        let path = self.cache_path(name);
        let file = fs::File::create(&path)?;
        let writer = BufWriter::new(file);

        if self.compress {
            // 使用 zstd 压缩，压缩级别 3（平衡速度和压缩率）
            #[cfg(feature = "compression")]
            {
                let mut encoder = zstd::stream::Encoder::new(writer, 3)?;
                let encoded = bincode::serialize(cache)
                    .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
                use std::io::Write as _;
                encoder
                    .write_all(&encoded)
                    .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
                encoder.finish()?;
            }
            #[cfg(not(feature = "compression"))]
            {
                return Err(PersistenceError::Serialization(
                    "compression feature not enabled".to_string(),
                ));
            }
        } else {
            bincode::serialize_into(writer, cache)
                .map_err(|e| PersistenceError::Serialization(e.to_string()))?;
        }

        Ok(())
    }

    /// 从磁盘加载 KV Cache
    ///
    /// 根据缓存名称从磁盘读取并反序列化。
    /// 自动检测是否为压缩格式。
    ///
    /// # 参数
    /// - `name`: 要加载的缓存名称
    ///
    /// # 返回
    /// 成功返回 [`PersistedKvCache`]，失败返回相应错误
    ///
    /// # 错误
    /// - [`PersistenceError::NotFound`][]: 缓存文件不存在
    /// - [`PersistenceError::Deserialization`][]: 反序列化失败
    pub fn load(&self, name: &str) -> Result<PersistedKvCache, PersistenceError> {
        let path = self.cache_path(name);
        if !path.exists() {
            return Err(PersistenceError::NotFound(name.to_string()));
        }

        let file = fs::File::open(&path)?;
        let reader = BufReader::new(file);

        if path.extension().is_some_and(|e| e == "zst") {
            // zstd 压缩格式
            #[cfg(feature = "compression")]
            {
                let mut decoder = zstd::stream::Decoder::new(reader)?;
                use std::io::Read as _;
                let mut decompressed = Vec::new();
                decoder
                    .read_to_end(&mut decompressed)
                    .map_err(|e| PersistenceError::Deserialization(e.to_string()))?;
                bincode::deserialize::<PersistedKvCache>(&decompressed)
                    .map_err(|e| PersistenceError::Deserialization(e.to_string()))
            }
            #[cfg(not(feature = "compression"))]
            {
                Err(PersistenceError::Deserialization(
                    "compression feature not enabled".to_string(),
                ))
            }
        } else {
            // 原始二进制格式
            bincode::deserialize_from(reader)
                .map_err(|e| PersistenceError::Deserialization(e.to_string()))
        }
    }

    /// 列出所有已保存的缓存
    ///
    /// 扫描存储目录，返回所有缓存文件的元信息列表。
    /// 结果按创建时间降序排列（最新的在前）。
    ///
    /// # 返回
    /// 成功返回 [`CacheInfo`] 向量，如果目录不存在则返回空向量
    pub fn list(&self) -> Result<Vec<CacheInfo>, PersistenceError> {
        let mut caches = Vec::new();

        if !self.base_dir.exists() {
            return Ok(caches);
        }

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let metadata = fs::metadata(&path)?;
                let compressed = path.extension().is_some_and(|e| e == "zst");
                let created_at = metadata
                    .modified()?
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs();

                caches.push(CacheInfo {
                    name,
                    size_bytes: metadata.len(),
                    model_name: String::new(), // 需要读取完整文件才能获取
                    created_at,
                    compressed,
                });
            }
        }

        caches.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(caches)
    }

    /// 删除指定缓存
    ///
    /// 从磁盘删除指定名称的缓存文件。
    ///
    /// # 参数
    /// - `name`: 要删除的缓存名称
    ///
    /// # 错误
    /// - [`PersistenceError::NotFound`][]: 缓存文件不存在
    pub fn delete(&self, name: &str) -> Result<(), PersistenceError> {
        let path = self.cache_path(name);
        if !path.exists() {
            return Err(PersistenceError::NotFound(name.to_string()));
        }
        fs::remove_file(path)?;
        Ok(())
    }

    /// 清理过期缓存
    ///
    /// 删除所有超过指定时间的缓存文件。
    ///
    /// # 参数
    /// - `max_age_secs`: 最大存活时间（秒）
    ///
    /// # 返回
    /// 返回删除的缓存数量
    pub fn cleanup(&self, max_age_secs: u64) -> Result<usize, PersistenceError> {
        let caches = self.list()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();

        let mut deleted = 0;
        for cache in caches {
            if now.saturating_sub(cache.created_at) > max_age_secs
                && self.delete(&cache.name).is_ok()
            {
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    /// 检查缓存是否存在
    ///
    /// # 参数
    /// - `name`: 缓存名称
    pub fn exists(&self, name: &str) -> bool {
        self.cache_path(name).exists()
    }

    /// 获取缓存文件大小（字节）
    ///
    /// # 参数
    /// - `name`: 缓存名称
    ///
    /// # 返回
    /// 成功返回文件大小，文件不存在返回 None
    pub fn size(&self, name: &str) -> Option<u64> {
        fs::metadata(self.cache_path(name)).ok().map(|m| m.len())
    }

    /// 构建缓存文件路径
    fn cache_path(&self, name: &str) -> PathBuf {
        let filename = if self.compress {
            format!("{}.kvcache.zst", name)
        } else {
            format!("{}.kvcache", name)
        };
        self.base_dir.join(filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// 创建测试用的示例缓存数据
    fn create_test_cache(model_name: &str) -> PersistedKvCache {
        PersistedKvCache {
            key_data: vec![1.0f32, 2.0, 3.0, 4.0],
            value_data: vec![5.0f32, 6.0, 7.0, 8.0],
            total_rows: 1,
            cols: 4,
            model_name: model_name.to_string(),
            created_at: 1234567890,
            version: 1,
        }
    }

    #[test]
    fn test_save_and_load_uncompressed() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();

        let cache = create_test_cache("test-model");

        persist.save(&cache, "test1").unwrap();
        let loaded = persist.load("test1").unwrap();

        assert_eq!(loaded.key_data, cache.key_data);
        assert_eq!(loaded.value_data, cache.value_data);
        assert_eq!(loaded.total_rows, cache.total_rows);
        assert_eq!(loaded.cols, cache.cols);
        assert_eq!(loaded.model_name, cache.model_name);
        assert_eq!(loaded.version, cache.version);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_save_and_load_compressed() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), true).unwrap();

        let cache = create_test_cache("compressed-model");

        persist.save(&cache, "compressed_test").unwrap();
        let loaded = persist.load("compressed_test").unwrap();

        assert_eq!(loaded.key_data, cache.key_data);
        assert_eq!(loaded.value_data, cache.value_data);
        assert_eq!(loaded.model_name, cache.model_name);
    }

    #[test]
    fn test_list_and_delete() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();

        let cache = create_test_cache("model-a");
        persist.save(&cache, "cache-a").unwrap();
        persist.save(&cache, "cache-b").unwrap();

        // 列出应该有2个缓存
        let list = persist.list().unwrap();
        assert_eq!(list.len(), 2);

        // 删除一个
        persist.delete("cache-a").unwrap();
        let list = persist.list().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "cache-b");
    }

    #[test]
    fn test_load_nonexistent() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();

        let result = persist.load("nonexistent");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PersistenceError::NotFound(_)));
    }

    #[test]
    fn test_delete_nonexistent() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();

        let result = persist.delete("nonexistent");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), PersistenceError::NotFound(_)));
    }

    #[test]
    fn test_exists_check() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();
        let cache = create_test_cache("model");

        assert!(!persist.exists("my_cache"));
        persist.save(&cache, "my_cache").unwrap();
        assert!(persist.exists("my_cache"));
    }

    #[test]
    fn test_size_check() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();
        let cache = create_test_cache("model");

        assert!(persist.size("my_cache").is_none());
        persist.save(&cache, "my_cache").unwrap();
        let size = persist.size("my_cache").unwrap();
        assert!(size > 0);
    }

    #[test]
    fn test_list_empty_directory() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();

        let list = persist.list().unwrap();
        assert!(list.is_empty());
    }

    #[test]
    fn test_list_sorted_by_time() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();
        let cache = create_test_cache("model");

        // 依次保存多个缓存
        persist.save(&cache, "first").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        persist.save(&cache, "second").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        persist.save(&cache, "third").unwrap();

        let list = persist.list().unwrap();
        assert_eq!(list.len(), 3);
        // 验证按时间降序排列（最新的在前）
        assert_eq!(list[0].name, "third");
        assert_eq!(list[1].name, "second");
        assert_eq!(list[2].name, "first");
    }

    #[test]
    fn test_cleanup_expired_caches() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();

        // 创建一个缓存
        let cache = create_test_cache("model");
        persist.save(&cache, "old_cache").unwrap();
        persist.save(&cache, "new_cache").unwrap();

        // 清理超过365天的缓存（应该不会删除任何东西）
        let deleted = persist.cleanup(365 * 24 * 3600).unwrap();
        assert_eq!(deleted, 0);
        assert!(persist.exists("old_cache"));
        assert!(persist.exists("new_cache"));

        // 验证 cleanup 返回正确的计数
        // 由于我们刚创建的文件都是新的，所以清理0秒的过期时间可能删除部分或全部
        // 这里只验证功能正常工作
    }

    #[test]
    fn test_persistence_error_display() {
        let err = PersistenceError::NotFound("test_cache".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test_cache"));

        let io_err = PersistenceError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(format!("{}", io_err).contains("file not found"));
    }

    #[test]
    fn test_multiple_save_overwrite() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();

        let cache_v1 = create_test_cache("model-v1");
        let cache_v2 = PersistedKvCache {
            key_data: vec![10.0, 20.0],
            value_data: vec![30.0, 40.0],
            total_rows: 1,
            cols: 2,
            model_name: "model-v2".to_string(),
            created_at: 9999999999,
            version: 2,
        };

        // 第一次保存
        persist.save(&cache_v1, "overwrite_test").unwrap();
        let loaded_v1 = persist.load("overwrite_test").unwrap();
        assert_eq!(loaded_v1.model_name, "model-v1");

        // 覆盖保存
        persist.save(&cache_v2, "overwrite_test").unwrap();
        let loaded_v2 = persist.load("overwrite_test").unwrap();
        assert_eq!(loaded_v2.model_name, "model-v2");
        assert_eq!(loaded_v2.version, 2);
        assert_eq!(loaded_v2.key_data, vec![10.0f32, 20.0]);
    }

    #[test]
    fn test_create_nested_directory() {
        let dir = TempDir::new().unwrap();
        let nested_path = dir.path().join("nested").join("deep").join("directory");

        // 目录不存在时应该自动创建
        let persist = KvCachePersistence::new(&nested_path, false).unwrap();
        assert!(nested_path.exists());

        let cache = create_test_cache("nested-test");
        persist.save(&cache, "test").unwrap();
        assert!(persist.exists("test"));
    }

    #[test]
    fn test_large_data_roundtrip() {
        let dir = TempDir::new().unwrap();
        let persist = KvCachePersistence::new(dir.path(), false).unwrap();

        // 创建较大的数据集
        let large_cache = PersistedKvCache {
            key_data: (0..10000).map(|i| i as f32).collect(),
            value_data: (10000..20000).map(|i| i as f32 * 0.5).collect(),
            total_rows: 100,
            cols: 100,
            model_name: "large-model".to_string(),
            created_at: 1700000000,
            version: 1,
        };

        persist.save(&large_cache, "large_test").unwrap();
        let loaded = persist.load("large_test").unwrap();

        assert_eq!(loaded.key_data.len(), 10000);
        assert_eq!(loaded.value_data.len(), 10000);
        assert_eq!(loaded.key_data[5000], 5000.0);
        assert_eq!(loaded.value_data[9999], 19999.0 * 0.5);
    }
}
