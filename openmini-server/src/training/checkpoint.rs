//! CheckpointManager - 训练检查点管理

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// 检查点操作错误类型
#[derive(Debug)]
pub enum CheckpointError {
    /// IO 错误
    Io(std::io::Error),
    /// 序列化错误
    Serialization(String),
    /// 检查点未找到
    NotFound(PathBuf),
    /// 检查点损坏
    Corrupted(PathBuf),
    /// 其他错误
    Other(String),
}

impl From<std::io::Error> for CheckpointError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Self::NotFound(path) => write!(f, "Checkpoint not found: {:?}", path),
            Self::Corrupted(path) => write!(f, "Checkpoint corrupted: {:?}", path),
            Self::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for CheckpointError {}

/// 保存策略枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SaveStrategy {
    /// 每隔 N 步保存一次
    Steps(u64),
    /// 每个 epoch 结束时保存
    Epoch(usize),
    /// 仅在最佳验证损失时保存
    Best,
}

/// 检查点元信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    /// 检查点路径
    pub path: PathBuf,
    /// 全局步数
    pub step: u64,
    /// 当前 epoch
    pub epoch: usize,
    /// 验证损失值
    pub val_loss: f64,
    /// 创建时间 (RFC3339 格式)
    pub created_at: String,
}

/// 检查点数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    /// 当前 epoch
    pub epoch: usize,
    /// 全局步数
    pub global_step: u64,
    /// 最佳验证损失
    pub best_val_loss: f64,
    /// 优化器状态字节序列
    pub optimizer_state_bytes: Vec<u8>,
}

/// 检查点管理器
pub struct CheckpointManager {
    output_dir: PathBuf,
    save_strategy: SaveStrategy,
    save_total_limit: usize,
    checkpoints: Vec<CheckpointInfo>,
}

impl CheckpointManager {
    /// 创建新的检查点管理器
    ///
    /// # Arguments
    ///
    /// * `output_dir` - 检查点输出目录
    /// * `strategy` - 保存策略
    /// * `limit` - 最大保留检查点数量
    ///
    /// # Returns
    ///
    /// 返回新的 CheckpointManager 实例或错误
    pub fn new(
        output_dir: PathBuf,
        strategy: SaveStrategy,
        limit: usize,
    ) -> Result<Self, CheckpointError> {
        fs::create_dir_all(&output_dir)?;
        Ok(Self {
            output_dir,
            save_strategy: strategy,
            save_total_limit: limit,
            checkpoints: Vec::new(),
        })
    }

    /// 保存检查点
    ///
    /// # Arguments
    ///
    /// * `state_data` - 训练状态数据
    /// * `val_loss` - 当前验证损失值
    ///
    /// # Returns
    ///
    /// 返回保存的检查点路径或错误
    pub fn save(
        &mut self,
        state_data: &CheckpointData,
        val_loss: f64,
    ) -> Result<PathBuf, CheckpointError> {
        let step = state_data.global_step;
        let epoch = state_data.epoch;

        let dir_name = format!("checkpoint_{:08}", step);
        let dir_path = self.output_dir.join(&dir_name);
        fs::create_dir_all(&dir_path)?;

        // 保存训练状态为 JSON
        let state_json = serde_json::to_string_pretty(state_data)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;
        fs::write(dir_path.join("training_state.json"), state_json)?;

        // 保存优化器状态为二进制
        fs::write(
            dir_path.join("optimizer_state.bin"),
            &state_data.optimizer_state_bytes,
        )?;

        // 元数据
        let info = CheckpointInfo {
            path: dir_path.clone(),
            step,
            epoch,
            val_loss,
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        self.checkpoints.push(info.clone());

        // 清理过期检查点
        if self.checkpoints.len() > self.save_total_limit {
            self.cleanup()?;
        }

        Ok(dir_path)
    }

    /// 加载最新的检查点
    ///
    /// # Returns
    ///
    /// 返回最新检查点的数据或错误
    pub fn load_latest(&self) -> Result<CheckpointData, CheckpointError> {
        if let Some(latest) = self.checkpoints.last() {
            self.load(&latest.path)
        } else {
            Err(CheckpointError::NotFound(self.output_dir.clone()))
        }
    }

    /// 加载指定路径的检查点
    ///
    /// # Arguments
    ///
    /// * `path` - 检查点路径
    ///
    /// # Returns
    ///
    /// 返回检查点数据或错误
    pub fn load(&self, path: &Path) -> Result<CheckpointData, CheckpointError> {
        let state_file = path.join("training_state.json");
        if !state_file.exists() {
            return Err(CheckpointError::NotFound(path.to_path_buf()));
        }

        let content = fs::read_to_string(&state_file).map_err(CheckpointError::Io)?;

        let data: CheckpointData = serde_json::from_str(&content)
            .map_err(|e| CheckpointError::Serialization(e.to_string()))?;

        Ok(data)
    }

    /// 清理过期检查点
    ///
    /// # Returns
    ///
    /// 返回清理后剩余的检查点数量或错误
    pub fn cleanup(&mut self) -> Result<usize, CheckpointError> {
        while self.checkpoints.len() > self.save_total_limit {
            if let Some(oldest) = self.checkpoints.first() {
                if oldest.path.exists() {
                    fs::remove_dir_all(&oldest.path)?;
                }
                self.checkpoints.remove(0);
            } else {
                break;
            }
        }
        Ok(self.checkpoints.len())
    }

    /// 获取最佳检查点（验证损失最低）
    ///
    /// # Returns
    ///
    /// 返回最佳检查点的信息引用，如果没有则返回 None
    pub fn best_checkpoint(&self) -> Option<&CheckpointInfo> {
        self.checkpoints
            .iter()
            .min_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).unwrap())
    }

    /// 列出所有检查点
    ///
    /// # Returns
    ///
    /// 返回所有检查点信息的切片
    pub fn list_checkpoints(&self) -> &[CheckpointInfo] {
        &self.checkpoints
    }
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_manager() {
        let tmp = TempDir::new().unwrap();
        let mgr = CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Steps(100), 5);
        assert!(mgr.is_ok());
    }

    #[test]
    fn test_save_and_load() {
        let tmp = TempDir::new().unwrap();
        let mut mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Steps(100), 5).unwrap();

        let data = CheckpointData {
            epoch: 1,
            global_step: 100,
            best_val_loss: 2.3456,
            optimizer_state_bytes: vec![1, 2, 3, 4],
        };

        let path = mgr.save(&data, 2.3456).unwrap();
        assert!(path.exists());

        let loaded = mgr.load(&path).unwrap();
        assert_eq!(loaded.epoch, 1);
        assert_eq!(loaded.global_step, 100);
    }

    #[test]
    fn test_cleanup_old_checkpoints() {
        let tmp = TempDir::new().unwrap();
        let mut mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Steps(100), 2).unwrap();

        for i in 0..4 {
            let data = CheckpointData {
                epoch: i,
                global_step: i as u64 * 100,
                best_val_loss: 10.0 - i as f64,
                optimizer_state_bytes: vec![],
            };
            mgr.save(&data, 10.0 - i as f64).unwrap();
        }

        assert_eq!(mgr.list_checkpoints().len(), 2);
    }

    #[test]
    fn test_load_nonexistent_checkpoint() {
        let tmp = TempDir::new().unwrap();
        let mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Steps(100), 5).unwrap();

        // 尝试加载不存在的检查点
        let result = mgr.load_latest();
        assert!(result.is_err());
        match result.unwrap_err() {
            CheckpointError::NotFound(_) => {}
            _ => panic!("期望 NotFound 错误"),
        }
    }

    #[test]
    fn test_save_load_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let mut mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Steps(100), 5).unwrap();

        let original_data = CheckpointData {
            epoch: 42,
            global_step: 12345,
            best_val_loss: 0.123456789,
            optimizer_state_bytes: vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        };

        let path = mgr
            .save(&original_data, original_data.best_val_loss)
            .unwrap();

        // 加载并验证数据完全一致
        let loaded_data = mgr.load(&path).unwrap();
        assert_eq!(loaded_data.epoch, original_data.epoch);
        assert_eq!(loaded_data.global_step, original_data.global_step);
        assert!((loaded_data.best_val_loss - original_data.best_val_loss).abs() < 1e-9);
        assert_eq!(
            loaded_data.optimizer_state_bytes,
            original_data.optimizer_state_bytes
        );
    }

    #[test]
    fn test_best_checkpoint_selection() {
        let tmp = TempDir::new().unwrap();
        let mut mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Steps(100), 10).unwrap();

        // 保存多个具有不同损失的检查点
        let losses = [1.5, 0.8, 2.3, 0.5, 1.2];
        for (i, &loss) in losses.iter().enumerate() {
            let data = CheckpointData {
                epoch: i,
                global_step: (i + 1) as u64 * 100,
                best_val_loss: loss,
                optimizer_state_bytes: vec![],
            };
            mgr.save(&data, loss).unwrap();
        }

        // 最佳检查点应该是 loss=0.5 的那个
        let best = mgr.best_checkpoint().unwrap();
        assert!((best.val_loss - 0.5).abs() < 1e-6);
        assert_eq!(best.step, 400); // 第4个检查点
    }

    #[test]
    fn test_multiple_saves_increments() {
        let tmp = TempDir::new().unwrap();
        let mut mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Steps(100), 10).unwrap();

        for step in [100, 200, 300, 400, 500].iter() {
            let data = CheckpointData {
                epoch: (*step / 100) as usize,
                global_step: *step,
                best_val_loss: *step as f64 / 100.0,
                optimizer_state_bytes: vec![*step as u8],
            };
            mgr.save(&data, data.best_val_loss).unwrap();
        }

        assert_eq!(mgr.checkpoints.len(), 5);

        // 验证每个检查点的步数正确
        for (i, cp) in mgr.list_checkpoints().iter().enumerate() {
            assert_eq!(cp.step, (i as u64 + 1) * 100);
        }
    }

    #[test]
    fn test_checkpoint_info_metadata() {
        let tmp = TempDir::new().unwrap();
        let mut mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Epoch(1), 5).unwrap();

        let data = CheckpointData {
            epoch: 5,
            global_step: 500,
            best_val_loss: 1.23456,
            optimizer_state_bytes: vec![255; 1024], // 较大的数据
        };

        let _path = mgr.save(&data, data.best_val_loss).unwrap();

        // 验证元信息
        let checkpoints = mgr.list_checkpoints();
        assert_eq!(checkpoints.len(), 1);
        let info = &checkpoints[0];

        assert_eq!(info.epoch, 5);
        assert_eq!(info.step, 500);
        assert!((info.val_loss - 1.23456).abs() < 1e-6);
        assert!(info.path.exists());
        assert!(!info.created_at.is_empty());
    }

    #[test]
    fn test_empty_optimizer_state() {
        let tmp = TempDir::new().unwrap();
        let mut mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Steps(100), 5).unwrap();

        let data = CheckpointData {
            epoch: 0,
            global_step: 0,
            best_val_loss: 999.999,
            optimizer_state_bytes: vec![],
        };

        let path = mgr.save(&data, data.best_val_loss).unwrap();
        let loaded = mgr.load(&path).unwrap();

        assert!(loaded.optimizer_state_bytes.is_empty());
        assert_eq!(loaded.epoch, 0);
    }

    #[test]
    fn test_large_checkpoint_data() {
        let tmp = TempDir::new().unwrap();
        let mut mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Best, 3).unwrap();

        // 模拟大型模型状态（1MB 数据）
        let large_data = vec![42u8; 1024 * 1024];
        let data = CheckpointData {
            epoch: 100,
            global_step: 50000,
            best_val_loss: 0.001,
            optimizer_state_bytes: large_data.clone(),
        };

        let path = mgr.save(&data, data.best_val_loss).unwrap();
        let loaded = mgr.load(&path).unwrap();

        assert_eq!(loaded.optimizer_state_bytes.len(), large_data.len());
        assert_eq!(loaded.optimizer_state_bytes, large_data);
    }

    #[test]
    fn test_cleanup_preserves_newest() {
        let tmp = TempDir::new().unwrap();
        let mut mgr =
            CheckpointManager::new(tmp.path().to_path_buf(), SaveStrategy::Steps(100), 2).unwrap();

        // 保存3个检查点，限制为2个
        for i in 0..3u64 {
            let data = CheckpointData {
                epoch: i as usize,
                global_step: (i + 1) * 100,
                best_val_loss: 10.0 - i as f64,
                optimizer_state_bytes: vec![i as u8; 10],
            };
            mgr.save(&data, 10.0 - i as f64).unwrap();
        }

        // 应该保留最后2个（最新的）
        assert_eq!(mgr.list_checkpoints().len(), 2);
        let checkpoints = mgr.list_checkpoints();

        // 验证保留的是最新的两个（step 200 和 300）
        assert_eq!(checkpoints[0].step, 200);
        assert_eq!(checkpoints[1].step, 300);
    }

    #[test]
    fn test_directory_creation() {
        let tmp = TempDir::new().unwrap();
        let subdir = tmp
            .path()
            .join("nested")
            .join("directory")
            .join("structure");

        // 目录应该被自动创建
        let mgr = CheckpointManager::new(subdir.clone(), SaveStrategy::Steps(100), 5);
        assert!(mgr.is_ok());
        assert!(subdir.exists());

        // 可以正常保存和加载
        let mut mgr = mgr.unwrap();
        let data = CheckpointData {
            epoch: 1,
            global_step: 100,
            best_val_loss: 1.0,
            optimizer_state_bytes: vec![1, 2, 3],
        };

        let path = mgr.save(&data, 1.0).unwrap();
        assert!(path.exists());
        let loaded = mgr.load(&path).unwrap();
        assert_eq!(loaded.global_step, 100);
    }
}
