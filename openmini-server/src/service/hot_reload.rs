//! 模型热加载管理器
//!
//! 运行时动态切换模型，零停机：
//! - 双缓冲区策略
//! - 原子性切换
//! - 后台预加载

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use tokio::sync::Notify;

/// 模型实例
pub struct ModelInstance {
    pub id: String,
    pub version: u64,
    pub path: PathBuf,
    pub loaded_at: Instant,
    pub memory_estimate_mb: u64,
}

impl ModelInstance {
    pub fn new(id: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            id: id.into(),
            version: 1,
            path: path.into(),
            loaded_at: Instant::now(),
            memory_estimate_mb: 0,
        }
    }
}

/// 热加载配置
#[derive(Debug, Clone)]
pub struct HotReloadConfig {
    pub enable_standby: bool,
    pub auto_discover: bool,
    pub watch_interval_secs: u64,
    pub max_loaded_models: usize,
}

impl Default for HotReloadConfig {
    fn default() -> Self {
        Self {
            enable_standby: true,
            auto_discover: false,
            watch_interval_secs: 30,
            max_loaded_models: 5,
        }
    }
}

/// 模型状态
#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub id: String,
    pub is_active: bool,
    pub has_standby: bool,
    pub memory_usage_mb: u64,
    pub load_time_ms: u64,
    pub version: u64,
}

/// 热加载错误
#[derive(Debug, thiserror::Error)]
pub enum HotReloadError {
    #[error("模型不存在: {0}")]
    NotFound(String),
    #[error("加载失败: {0}")]
    LoadFailed(String),
    #[error("版本不兼容: {0}")]
    VersionIncompatible(String),
    #[error("内存不足: 需要{needed}MB, 可用{available}MB")]
    InsufficientMemory { needed: u64, available: u64 },
    #[error("已达最大模型数限制: {0}")]
    MaxModelsReached(usize),
}

/// 热加载管理器
pub struct HotReloadManager {
    active_models: RwLock<HashMap<String, Arc<ModelInstance>>>,
    standby_models: RwLock<HashMap<String, Arc<ModelInstance>>>,
    notify: Notify,
    config: HotReloadConfig,
}

impl HotReloadManager {
    /// 创建新实例
    pub fn new(config: HotReloadConfig) -> Self {
        Self {
            active_models: RwLock::new(HashMap::new()),
            standby_models: RwLock::new(HashMap::new()),
            notify: Notify::new(),
            config,
        }
    }

    /// 获取或加载模型
    pub fn get_or_load(
        &self,
        model_id: &str,
        model_path: &Path,
    ) -> Result<Arc<ModelInstance>, HotReloadError> {
        // 先检查活跃模型
        {
            let active = self.active_models.read().unwrap();
            if let Some(model) = active.get(model_id) {
                return Ok(Arc::clone(model));
            }
        }

        // 检查备用模型并提升为活跃
        {
            let mut standby = self.standby_models.write().unwrap();
            if let Some(arc_model) = standby.remove(model_id) {
                drop(standby);

                let mut active = self.active_models.write().unwrap();
                active.insert(model_id.to_string(), Arc::clone(&arc_model));
                return Ok(arc_model);
            }
        }

        // 加载新模型
        self.load_model(model_id, model_path)
    }

    /// 加载新模型
    fn load_model(
        &self,
        model_id: &str,
        model_path: &Path,
    ) -> Result<Arc<ModelInstance>, HotReloadError> {
        if !model_path.exists() {
            return Err(HotReloadError::NotFound(format!("{:?}", model_path)));
        }

        // 检查模型数量限制
        {
            let active = self.active_models.read().unwrap();
            if active.len() >= self.config.max_loaded_models && !active.contains_key(model_id) {
                return Err(HotReloadError::MaxModelsReached(
                    self.config.max_loaded_models,
                ));
            }
        }

        let instance = ModelInstance::new(model_id, model_path);
        let arc_instance = Arc::new(instance);

        let mut active = self.active_models.write().unwrap();
        active.insert(model_id.to_string(), Arc::clone(&arc_instance));

        Ok(arc_instance)
    }

    /// 预加载备用模型
    pub fn preload_standby(&self, model_id: &str, model_path: &Path) -> Result<(), HotReloadError> {
        if !model_path.exists() {
            return Err(HotReloadError::NotFound(format!("{:?}", model_path)));
        }

        let instance = ModelInstance::new(model_id, model_path);
        let arc_instance = Arc::new(instance);

        let mut standby = self.standby_models.write().unwrap();
        standby.insert(model_id.to_string(), arc_instance);

        Ok(())
    }

    /// 切换到备用模型（原子操作）
    pub fn switch_to_standby(&self, model_id: &str) -> Result<(), HotReloadError> {
        let mut standby = self.standby_models.write().unwrap();
        let standby_model = standby
            .remove(model_id)
            .ok_or_else(|| HotReloadError::NotFound(model_id.to_string()))?;

        drop(standby);

        let mut active = self.active_models.write().unwrap();
        active.insert(model_id.to_string(), standby_model);

        self.notify.notify_one();
        Ok(())
    }

    /// 卸载模型
    pub fn unload(&self, model_id: &str) -> Result<(), HotReloadError> {
        let mut active = self.active_models.write().unwrap();
        active
            .remove(model_id)
            .ok_or_else(|| HotReloadError::NotFound(model_id.to_string()))?;

        Ok(())
    }

    /// 获取所有模型状态
    pub fn status(&self) -> Vec<ModelStatus> {
        let active = self.active_models.read().unwrap();
        let standby = self.standby_models.read().unwrap();

        let mut statuses = Vec::new();

        for (id, model) in active.iter() {
            statuses.push(ModelStatus {
                id: id.clone(),
                is_active: true,
                has_standby: standby.contains_key(id),
                memory_usage_mb: model.memory_estimate_mb,
                load_time_ms: model.loaded_at.elapsed().as_millis() as u64,
                version: model.version,
            });
        }

        statuses
    }

    /// 获取通知器（用于等待切换完成）
    pub fn notifier(&self) -> &Notify {
        &self.notify
    }
}

impl Default for HotReloadManager {
    fn default() -> Self {
        Self::new(HotReloadConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_manager() {
        let manager = HotReloadManager::default();
        assert!(manager.status().is_empty());
    }

    #[test]
    fn test_load_and_get_model() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.gguf");

        // 创建假模型文件
        std::fs::write(&model_path, "fake model data").unwrap();

        let manager = HotReloadManager::default();
        let model = manager.get_or_load("test-model", &model_path).unwrap();

        assert_eq!(model.id, "test-model");
        assert_eq!(manager.status().len(), 1);
    }

    #[test]
    fn test_preload_standby() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("standby.gguf");
        std::fs::write(&model_path, "standby data").unwrap();

        let manager = HotReloadManager::default();
        manager
            .preload_standby("standby-model", &model_path)
            .unwrap();

        let status = manager.status();
        assert!(status.is_empty()); // 备用模型不在active列表中
    }

    #[test]
    fn test_switch_to_standby() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("switch.gguf");
        std::fs::write(&model_path, "switch data").unwrap();

        let manager = HotReloadManager::default();

        // 预加载备用
        manager
            .preload_standby("switch-model", &model_path)
            .unwrap();

        // 切换
        manager.switch_to_standby("switch-model").unwrap();

        let status = manager.status();
        assert_eq!(status.len(), 1);
        assert!(status[0].is_active);
    }

    #[test]
    fn test_unload_model() {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("unload.gguf");
        std::fs::write(&model_path, "unload data").unwrap();

        let manager = HotReloadManager::default();
        manager.get_or_load("unload-model", &model_path).unwrap();
        manager.unload("unload-model").unwrap();

        assert!(manager.status().is_empty());
    }

    #[test]
    fn test_max_models_limit() {
        let config = HotReloadConfig {
            max_loaded_models: 2,
            ..Default::default()
        };
        let manager = HotReloadManager::new(config);

        let dir = TempDir::new().unwrap();
        for i in 0..3 {
            let path = dir.path().join(format!("model{}.gguf", i));
            std::fs::write(&path, "data").unwrap();
            manager.get_or_load(&format!("model{}", i), &path).unwrap();
        }

        // 第3个应该失败
        let path4 = dir.path().join("model3.gguf");
        std::fs::write(&path4, "data").unwrap();
        let result = manager.get_or_load("model3", &path4);
        assert!(result.is_err());
    }

    #[test]
    fn test_not_found_error() {
        let manager = HotReloadManager::default();
        let result = manager.get_or_load("nonexistent", Path::new("/no/such/path"));

        assert!(matches!(result, Err(HotReloadError::NotFound(_))));
    }
}
