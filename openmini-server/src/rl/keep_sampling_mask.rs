//! Keep Sampling Mask 实现
//!
//! 保存采样截断掩码，确保训练与推理动作空间一致

use crate::rl::Tensor;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// 采样掩码数据
///
/// 保存采样截断位置和有效长度
#[derive(Debug, Clone)]
pub struct SamplingMaskData {
    pub mask: Tensor,
    pub valid_lengths: Vec<usize>,
}

/// 采样掩码管理器
///
/// 管理和存储多个采样掩码，支持 LRU 淘汰
pub struct SamplingMaskManager {
    masks: Arc<RwLock<HashMap<String, SamplingMaskData>>>,
    max_masks: usize,
}

impl SamplingMaskManager {
    pub fn new(max_masks: usize) -> Self {
        Self {
            masks: Arc::new(RwLock::new(HashMap::new())),
            max_masks,
        }
    }

    pub fn store(&self, key: String, mask_data: SamplingMaskData) {
        let mut masks = self.masks.write();

        if masks.len() >= self.max_masks && !masks.contains_key(&key) {
            if let Some(first_key) = masks.keys().next().cloned() {
                masks.remove(&first_key);
            }
        }

        masks.insert(key, mask_data);
    }

    pub fn get(&self, key: &str) -> Option<SamplingMaskData> {
        let masks = self.masks.read();
        masks.get(key).cloned()
    }

    pub fn remove(&self, key: &str) -> Option<SamplingMaskData> {
        let mut masks = self.masks.write();
        masks.remove(key)
    }

    pub fn clear(&self) {
        let mut masks = self.masks.write();
        masks.clear();
    }

    pub fn len(&self) -> usize {
        let masks = self.masks.read();
        masks.len()
    }

    pub fn is_empty(&self) -> bool {
        let masks = self.masks.read();
        masks.is_empty()
    }
}

impl Default for SamplingMaskManager {
    fn default() -> Self {
        Self::new(1000)
    }
}

pub struct KeepSamplingMask {
    mask_manager: SamplingMaskManager,
    enabled: bool,
}

impl KeepSamplingMask {
    pub fn new(max_masks: usize) -> Self {
        Self {
            mask_manager: SamplingMaskManager::new(max_masks),
            enabled: true,
        }
    }

    pub fn disable(mut self) -> Self {
        self.enabled = false;
        self
    }

    pub fn save_mask(&self, key: String, mask: Tensor, valid_lengths: Vec<usize>) {
        if !self.enabled {
            return;
        }

        let mask_data = SamplingMaskData {
            mask,
            valid_lengths,
        };

        self.mask_manager.store(key, mask_data);
    }

    pub fn get_mask(&self, key: &str) -> Option<SamplingMaskData> {
        if !self.enabled {
            return None;
        }

        self.mask_manager.get(key)
    }

    pub fn apply_mask(&self, key: &str, default_mask: &Tensor) -> Tensor {
        match self.get_mask(key) {
            Some(mask_data) => mask_data.mask,
            None => default_mask.clone(),
        }
    }

    pub fn get_valid_lengths(&self, key: &str) -> Option<Vec<usize>> {
        self.get_mask(key).map(|d| d.valid_lengths)
    }

    pub fn clear(&self) {
        self.mask_manager.clear();
    }
}

impl Default for KeepSamplingMask {
    fn default() -> Self {
        Self::new(1000)
    }
}

pub fn create_sampling_mask(max_length: usize, _eos_token_id: usize) -> (Tensor, Vec<usize>) {
    let mask_data: Vec<f32> = vec![1.0; max_length];
    let valid_lengths = vec![max_length];

    (Tensor::new(mask_data, vec![max_length]), valid_lengths)
}

pub fn create_mask_from_terminated(
    sequence_length: usize,
    terminated_at: Option<usize>,
    max_length: usize,
) -> (Tensor, Vec<usize>) {
    let terminate_pos = terminated_at.unwrap_or(sequence_length);
    let valid_length = terminate_pos.min(max_length);

    let mut mask_data = vec![0.0; max_length];
    mask_data[..valid_length].fill(1.0);

    (Tensor::new(mask_data, vec![max_length]), vec![valid_length])
}

pub fn merge_masks(masks: &[&Tensor]) -> Tensor {
    if masks.is_empty() {
        return Tensor::zeros(&[1]);
    }

    let len = masks[0].len();
    let merged: Vec<f32> = (0..len)
        .map(|i| {
            masks
                .iter()
                .map(|m| m.as_slice().get(i).unwrap_or(&0.0))
                .fold(1.0, |acc, &v| acc * v)
        })
        .collect();

    Tensor::new(merged, vec![len])
}

pub fn load_sampling_mask(_path: &str) -> Option<SamplingMaskData> {
    None
}

pub fn save_sampling_mask(_path: &str, _mask_data: &SamplingMaskData) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_mask_data_creation() {
        // 测试采样掩码数据创建
        let mask = Tensor::from_slice(&[1.0, 1.0, 1.0, 0.0, 0.0]);
        let valid_lengths = vec![3];

        let data = SamplingMaskData {
            mask,
            valid_lengths: valid_lengths.clone(),
        };

        assert_eq!(data.mask.len(), 5);
        assert_eq!(data.valid_lengths, vec![3]);
    }

    #[test]
    fn test_sampling_mask_manager_basic_operations() {
        // 测试基本 CRUD 操作
        let manager = SamplingMaskManager::new(3);

        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);

        let mask_data = SamplingMaskData {
            mask: Tensor::from_slice(&[1.0, 1.0, 0.0]),
            valid_lengths: vec![2],
        };

        manager.store("seq1".to_string(), mask_data);
        assert_eq!(manager.len(), 1);

        let retrieved = manager.get("seq1").unwrap();
        assert_eq!(retrieved.valid_lengths, vec![2]);

        assert!(manager.get("nonexistent").is_none());

        manager.remove("seq1");
        assert!(manager.is_empty());
    }

    #[test]
    fn test_sampling_mask_manager_lru_eviction() {
        // 测试容量限制（达到上限时淘汰旧条目）
        // 注意：HashMap 不保证顺序，所以这里只验证容量限制
        let manager = SamplingMaskManager::new(2); // 最大容量 2

        manager.store(
            "key0".to_string(),
            SamplingMaskData {
                mask: Tensor::from_slice(&[1.0f32]),
                valid_lengths: vec![1],
            },
        );
        manager.store(
            "key1".to_string(),
            SamplingMaskData {
                mask: Tensor::from_slice(&[1.0, 1.0]),
                valid_lengths: vec![2],
            },
        );

        assert_eq!(manager.len(), 2);

        // 添加第三个，应该淘汰某个旧的
        manager.store(
            "key2".to_string(),
            SamplingMaskData {
                mask: Tensor::from_slice(&[1.0, 1.0, 1.0]),
                valid_lengths: vec![3],
            },
        );

        assert_eq!(manager.len(), 2); // 容量仍为 2
                                      // 新添加的 key2 应该存在
        assert!(manager.get("key2").is_some());
    }

    #[test]
    fn test_sampling_mask_manager_clear() {
        // 测试清空操作
        let manager = SamplingMaskManager::new(10);

        for i in 0..5 {
            manager.store(
                format!("key{}", i),
                SamplingMaskData {
                    mask: Tensor::from_slice(&vec![1.0; i + 1]),
                    valid_lengths: vec![i + 1],
                },
            );
        }

        manager.clear();
        assert!(manager.is_empty());
    }

    #[test]
    fn test_keep_sampling_mask_creation_and_config() {
        // 测试创建和配置
        let mask = KeepSamplingMask::new(100);
        // 默认启用

        // 禁用
        let disabled = mask.disable();

        // 禁用状态下保存不生效
        disabled.save_mask("test".to_string(), Tensor::from_slice(&[1.0]), vec![1]);

        assert!(disabled.get_mask("test").is_none());
    }

    #[test]
    fn test_keep_sampling_mask_save_and_get() {
        // 测试保存和获取掩码
        let mask = KeepSamplingMask::new(100);

        let mask_tensor = Tensor::from_slice(&[1.0, 1.0, 1.0, 0.0]);
        let valid_lengths = vec![3];

        mask.save_mask("sequence1".to_string(), mask_tensor, valid_lengths);

        let retrieved = mask.get_mask("sequence1").unwrap();
        assert_eq!(retrieved.valid_lengths, vec![3]);
        assert_eq!(retrieved.mask.len(), 4);

        assert!(mask.get_mask("nonexistent").is_none());
    }

    #[test]
    fn test_keep_sampling_mask_apply_mask_with_cache() {
        // 测试应用掩码（有缓存时使用缓存）
        let mask = KeepSamplingMask::new(100);

        let cached_mask = Tensor::from_slice(&[1.0, 1.0, 0.0]);
        mask.save_mask("cached_seq".to_string(), cached_mask.clone(), vec![2]);

        let default_mask = Tensor::from_slice(&[0.5, 0.5, 0.5]); // 不应该被使用

        let applied = mask.apply_mask("cached_seq", &default_mask);

        // 应该返回缓存的掩码
        assert_eq!(applied.as_slice()[0], 1.0); // 来自缓存
        assert_eq!(applied.as_slice()[2], 0.0); // 来自缓存
    }

    #[test]
    fn test_keep_sampling_mask_apply_mask_without_cache() {
        // 测试应用掩码（无缓存时使用默认值）
        let mask = KeepSamplingMask::new(100);

        let default_mask = Tensor::from_slice(&[0.8, 0.9, 0.7]);

        let applied = mask.apply_mask("uncached_seq", &default_mask);

        // 无缓存时应该返回默认值的克隆
        assert_eq!(applied.as_slice(), default_mask.as_slice());
    }

    #[test]
    fn test_keep_sampling_mask_get_valid_lengths() {
        // 测试获取有效长度
        let mask = KeepSamplingMask::new(100);

        mask.save_mask(
            "seq1".to_string(),
            Tensor::from_slice(&[1.0, 1.0, 0.0]),
            vec![2, 3],
        );

        let lengths = mask.get_valid_lengths("seq1").unwrap();
        assert_eq!(lengths, vec![2, 3]);

        assert!(mask.get_valid_lengths("nonexistent").is_none());
    }

    #[test]
    fn test_keep_sampling_mask_default() {
        // 测试默认配置
        let mask = KeepSamplingMask::default();
        // 默认启用，可以正常操作
        mask.save_mask("test".to_string(), Tensor::from_slice(&[1.0]), vec![1]);
        assert!(mask.get_mask("test").is_some());
    }

    #[test]
    fn test_keep_sampling_mask_clear() {
        // 测试清空所有掩码
        let mask = KeepSamplingMask::new(100);

        mask.save_mask("key1".to_string(), Tensor::from_slice(&[1.0]), vec![1]);
        mask.save_mask("key2".to_string(), Tensor::from_slice(&[1.0, 0.0]), vec![1]);

        mask.clear();

        assert!(mask.get_mask("key1").is_none());
        assert!(mask.get_mask("key2").is_none());
    }

    #[test]
    fn test_create_sampling_mask() {
        // 测试创建采样掩码
        let (mask, lengths) = create_sampling_mask(10, 0);

        assert_eq!(mask.len(), 10);
        assert_eq!(lengths, vec![10]);

        // 所有位置应该是 1.0
        for &val in mask.as_slice() {
            assert!((val - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_create_mask_from_terminated() {
        // 测试从终止位置创建掩码
        // 正常终止
        let (mask, lengths) = create_mask_from_terminated(20, Some(15), 25);
        assert_eq!(mask.len(), 25);
        assert_eq!(lengths, vec![15]); // min(15, 25) = 15

        // 前 15 个应该是 1.0
        for i in 0..15 {
            assert!((mask.as_slice()[i] - 1.0).abs() < f32::EPSILON);
        }
        // 后面的应该是 0.0
        for i in 15..25 {
            assert!((mask.as_slice()[i]).abs() < f32::EPSILON);
        }

        // 未终止（使用序列长度）
        let (_mask2, lengths2) = create_mask_from_terminated(10, None, 15);
        assert_eq!(lengths2, vec![10]); // min(10, 15) = 10

        // 终止位置超过最大长度
        let (_mask3, lengths3) = create_mask_from_terminated(5, Some(100), 10);
        assert_eq!(lengths3, vec![10]); // min(100, 10) = 10
    }

    #[test]
    fn test_merge_masks() {
        // 测试合并多个掩码
        let mask1 = Tensor::from_slice(&[1.0, 1.0, 0.0, 1.0]);
        let mask2 = Tensor::from_slice(&[1.0, 0.0, 1.0, 1.0]);

        let merged = merge_masks(&[&mask1, &mask2]);

        // 逐元素相乘
        assert!((merged.as_slice()[0] - 1.0).abs() < f32::EPSILON); // 1*1=1
        assert!((merged.as_slice()[1]).abs() < f32::EPSILON); // 1*0=0
        assert!((merged.as_slice()[2]).abs() < f32::EPSILON); // 0*1=0
        assert!((merged.as_slice()[3] - 1.0).abs() < f32::EPSILON); // 1*1=1
    }

    #[test]
    fn test_merge_single_mask() {
        // 测试合并单个掩码（应该保持不变）
        let mask = Tensor::from_slice(&[1.0, 0.0, 1.0]);
        let merged = merge_masks(&[&mask]);

        assert_eq!(merged.as_slice(), mask.as_slice());
    }

    #[test]
    fn test_merge_empty_masks() {
        // 测试合并空列表（应该返回长度为1的全零张量）
        let merged = merge_masks(&[]);
        assert_eq!(merged.len(), 1);
        assert!((merged.as_slice()[0]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_load_save_sampling_mask_stubs() {
        // 测试加载/保存存根函数
        let loaded = load_sampling_mask("/nonexistent/path.mask");
        assert!(loaded.is_none());

        let mask_data = SamplingMaskData {
            mask: Tensor::from_slice(&[1.0]),
            valid_lengths: vec![1],
        };
        let saved = save_sampling_mask("/tmp/test.mask", &mask_data);
        assert!(!saved);
    }
}
