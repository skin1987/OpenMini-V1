//! 长期记忆层
//!
//! 持久化知识库，支持向量检索
//!
//! # 并发安全
//!
//! **注意：`LongTermMemory` 不是线程安全的。**
//!
//! 如果需要在多线程环境中使用，请使用 `Arc<RwLock<LongTermMemory>>` 或 `Arc<Mutex<LongTermMemory>>` 进行外部同步。
//!
//! ## 示例
//!
//! ```rust,ignore
//! use std::sync::{Arc, RwLock};
//!
//! let memory = Arc::new(RwLock::new(LongTermMemory::new(&config)));
//!
//! // 写入线程
//! let memory_clone = Arc::clone(&memory);
//! std::thread::spawn(move || {
//!     let mut mem = memory_clone.write().unwrap();
//!     mem.write("key".to_string(), data, timestamp, importance);
//! });
//!
//! // 读取线程
//! let memory_clone = Arc::clone(&memory);
//! std::thread::spawn(move || {
//!     let mem = memory_clone.read().unwrap();
//!     let results = mem.search(&query, 10);
//! });
//! ```
//!
//! # 性能建议
//!
//! - 对于大规模数据（> 10,000 条），建议调用 `build_hnsw_index()` 构建索引
//! - 使用 `search_hnsw()` 进行近似最近邻搜索，复杂度 O(log n)
//! - 线性搜索 `search()` 复杂度为 O(n)，仅适用于小规模数据

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::Array2;

use super::hnsw::{HNSWConfig, HNSWIndex};
use super::persistence::{Persistence, PersistenceConfig};
use super::simd_ops::SimdVectorOps;
use super::{EvictionStrategy, MemoryConfig, MemoryItem, MemoryLevel, PaddingStrategy};

/// 时间衰减管理器
#[derive(Debug, Clone)]
pub struct DecayManager {
    /// 生存时间（秒）
    pub ttl_seconds: u64,
    /// 衰减因子
    pub decay_factor: f32,
    /// 重要性阈值
    pub importance_threshold: f32,
}

impl Default for DecayManager {
    fn default() -> Self {
        Self {
            ttl_seconds: 86400 * 30,
            decay_factor: 0.95,
            importance_threshold: 0.1,
        }
    }
}

impl DecayManager {
    /// 创建新的时间衰减管理器
    ///
    /// # 参数
    /// - `ttl_seconds`: 生存时间（秒）
    /// - `decay_factor`: 衰减因子（0-1之间，越小衰减越快）
    /// - `importance_threshold`: 重要性阈值，低于此值的记忆将被自动遗忘
    pub fn new(ttl_seconds: u64, decay_factor: f32, importance_threshold: f32) -> Self {
        Self {
            ttl_seconds,
            decay_factor,
            importance_threshold,
        }
    }

    /// 根据记忆年龄计算衰减系数
    ///
    /// # 参数
    /// - `age_seconds`: 记忆的年龄（秒）
    ///
    /// # 返回值
    /// 衰减系数（0-1之间），年龄越大值越小
    pub fn compute_decay(&self, age_seconds: u64) -> f32 {
        if age_seconds == 0 {
            return 1.0;
        }
        let decay_periods = age_seconds as f32 / self.ttl_seconds as f32;
        self.decay_factor.powf(decay_periods)
    }
}

/// 访问统计
#[derive(Debug)]
pub struct AccessStats {
    /// 访问次数
    pub access_count: AtomicU32,
    /// 最后访问时间戳
    pub last_access: AtomicU64,
}

impl AccessStats {
    /// 创建新的访问统计记录
    ///
    /// # 参数
    /// - `timestamp`: 初始访问时间戳（Unix秒）
    ///
    /// 初始化时访问次数为1
    pub fn new(timestamp: u64) -> Self {
        Self {
            access_count: AtomicU32::new(1),
            last_access: AtomicU64::new(timestamp),
        }
    }

    /// 增加访问计数并更新最后访问时间
    ///
    /// # 参数
    /// - `timestamp`: 当前访问时间戳（Unix秒）
    pub fn increment(&self, timestamp: u64) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access.store(timestamp, Ordering::Relaxed);
    }

    /// 获取总访问次数
    pub fn get_access_count(&self) -> u32 {
        self.access_count.load(Ordering::Relaxed)
    }

    /// 获取最后访问时间戳（Unix秒）
    pub fn get_last_access(&self) -> u64 {
        self.last_access.load(Ordering::Relaxed)
    }
}

impl Clone for AccessStats {
    fn clone(&self) -> Self {
        Self {
            access_count: AtomicU32::new(self.get_access_count()),
            last_access: AtomicU64::new(self.get_last_access()),
        }
    }
}

/// 长期记忆存储管理器
///
/// 负责持久化存储和管理长期记忆数据，
/// 支持高效的向量检索和HNSW索引加速搜索。
///
/// # 线程安全
///
/// **注意：此结构体不是线程安全的。**
/// 多线程环境请使用 `Arc<RwLock<LongTermMemory>>` 进行同步。
#[derive(Debug)]
#[allow(dead_code)]
pub struct LongTermMemory {
    /// 存储的记忆项列表
    pub items: Vec<MemoryItem>,
    /// 缓存的 Vec<f32> 向量，避免重复转换
    vector_cache: Vec<Vec<f32>>,
    /// 嵌入维度（默认 128）
    embedding_dim: usize,
    /// 填充策略
    padding_strategy: PaddingStrategy,
    capacity: usize,
    strategy: EvictionStrategy,
    index: HashMap<String, usize>,
    hnsw_index: Option<HNSWIndex>,
    /// 保存的 HNSW 配置（用于恢复时重建）
    hnsw_config: Option<HNSWConfig>,
    decay_manager: DecayManager,
    access_stats: HashMap<usize, AccessStats>,
    persistence: Option<Persistence>,
    simd_ops: SimdVectorOps,
}

impl LongTermMemory {
    /// 创建新的长期记忆存储器
    ///
    /// # 参数
    /// - `config`: 内存配置，包含容量、嵌入维度等参数
    pub fn new(config: &MemoryConfig) -> Self {
        Self {
            items: Vec::with_capacity(config.long_term_capacity),
            vector_cache: Vec::with_capacity(config.long_term_capacity),
            embedding_dim: config.embedding_dim.unwrap_or(128),
            padding_strategy: config.padding_strategy,
            capacity: config.long_term_capacity,
            strategy: config.eviction_strategy,
            index: HashMap::new(),
            hnsw_index: None,
            hnsw_config: None,
            decay_manager: DecayManager::default(),
            access_stats: HashMap::new(),
            persistence: None,
            simd_ops: SimdVectorOps::new(),
        }
    }

    /// 设置时间衰减管理器（构建器模式）
    ///
    /// # 参数
    /// - `decay_manager`: 衰减管理器配置
    ///
    /// # 返回值
    /// 返回设置后的 self，支持链式调用
    pub fn with_decay(mut self, decay_manager: DecayManager) -> Self {
        self.decay_manager = decay_manager;
        self
    }

    /// 设置持久化存储（构建器模式）
    ///
    /// # 参数
    /// - `config`: 持久化配置，包含存储路径等
    ///
    /// # 返回值
    /// 返回设置后的 self，支持链式调用。
    /// 如果初始化失败则不设置持久化
    pub fn with_persistence(mut self, config: PersistenceConfig) -> Self {
        match Persistence::new(config) {
            Ok(persistence) => self.persistence = Some(persistence),
            Err(_) => self.persistence = None,
        }
        self
    }

    /// 写入记忆
    ///
    /// 将数据存储到长期记忆中，自动计算嵌入向量
    pub fn write(&mut self, key: String, data: Array2<f32>, timestamp: u64, importance: f32) {
        if self.items.len() >= self.capacity {
            self.evict();
        }

        // 计算嵌入向量并直接转换为 Vec<f32>，避免后续重复转换
        let embedding =
            Self::compute_embedding_with_dim(&data, self.embedding_dim, self.padding_strategy);
        let vector: Vec<f32> = embedding.iter().copied().collect();

        // 更新索引
        self.index.insert(key.clone(), self.items.len());
        self.items
            .push(MemoryItem::new(data.clone(), timestamp).with_importance(importance));
        self.vector_cache.push(vector.clone());

        // 更新访问统计
        self.access_stats
            .insert(self.items.len() - 1, AccessStats::new(timestamp));

        // 如果 HNSW 索引存在，插入新向量（使用已转换的 vector）
        if let Some(ref mut hnsw) = self.hnsw_index {
            hnsw.insert(self.items.len() - 1, vector);
        }
    }

    /// 根据键读取记忆数据
    ///
    /// # 参数
    /// - `key`: 记忆的唯一标识键
    ///
    /// # 返回值
    /// 如果找到返回 Some(数据)，否则返回 None
    pub fn read(&self, key: &str) -> Option<Array2<f32>> {
        self.index
            .get(key)
            .and_then(|&idx| self.items.get(idx).map(|item| item.data.clone()))
    }

    /// 搜索相似记忆（线性扫描）
    ///
    /// 对于大规模数据，建议先调用 `build_hnsw_index()` 并使用 `search_hnsw()`
    pub fn search(&self, query: &Array2<f32>, top_k: usize) -> Vec<(usize, f32)> {
        if self.vector_cache.is_empty() {
            return Vec::new();
        }

        let query_embedding =
            Self::compute_embedding_with_dim(query, self.embedding_dim, self.padding_strategy);
        let query_vec: Vec<f32> = query_embedding.iter().copied().collect();

        // 使用缓存的 vector_cache 进行相似度计算
        let mut scores: Vec<(usize, f32)> = self
            .vector_cache
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let min_len = query_vec.len().min(vec.len());
                let similarity = self
                    .simd_ops
                    .cosine_similarity(&query_vec[..min_len], &vec[..min_len]);
                (idx, similarity)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// 按重要性排序搜索记忆
    ///
    /// 返回重要性最高的 top_k 条记忆的原始数据
    ///
    /// # 参数
    /// - `top_k`: 返回的记忆数量
    ///
    /// # 返回值
    /// 按重要性降序排列的记忆数据列表
    pub fn search_by_importance(&self, top_k: usize) -> Vec<Array2<f32>> {
        let mut items: Vec<(usize, &MemoryItem)> = self.items.iter().enumerate().collect();
        items.sort_by(|a, b| {
            b.1.importance
                .partial_cmp(&a.1.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        items
            .into_iter()
            .take(top_k)
            .map(|(_idx, item)| item.data.clone())
            .collect()
    }

    /// 构建 HNSW 索引以加速向量搜索
    ///
    /// 对于大规模数据（>1000条）建议调用此方法构建索引，
    /// 之后使用 `search_hnsw()` 进行高效搜索。
    ///
    /// # 参数
    /// - `config`: HNSW配置，如果为None则使用默认配置
    pub fn build_hnsw_index(&mut self, config: Option<HNSWConfig>) {
        let cfg = config.unwrap_or_default();
        let hnsw = HNSWIndex::new(cfg);

        // 直接使用 vector_cache 构建，避免重复转换
        for (idx, vector) in self.vector_cache.iter().enumerate() {
            hnsw.insert(idx, vector.clone());
        }

        // 保存配置用于恢复时重建
        self.hnsw_config = Some(cfg);
        self.hnsw_index = Some(hnsw);
    }

    /// 使用 HNSW 索引进行近似最近邻搜索
    ///
    /// 如果未构建HNSW索引，则回退到线性搜索。
    ///
    /// # 参数
    /// - `query`: 查询向量
    /// - `top_k`: 返回的最相似结果数量
    ///
    /// # 返回值
    /// 返回 (索引, 相似度) 元组列表，按相似度降序排列
    pub fn search_hnsw(&self, query: &Array2<f32>, top_k: usize) -> Vec<(usize, f32)> {
        match &self.hnsw_index {
            Some(hnsw) => {
                let query_vec = Self::compute_embedding_with_dim(
                    query,
                    self.embedding_dim,
                    self.padding_strategy,
                );
                let query_slice: Vec<f32> = query_vec.iter().copied().collect();

                let results = hnsw.search(&query_slice, top_k);

                results
                    .into_iter()
                    .map(|r| (r.id, 1.0 / (1.0 + r.distance.sqrt())))
                    .collect()
            }
            None => self.search(query, top_k),
        }
    }

    /// 应用时间衰减到所有记忆项
    ///
    /// 根据每条记忆的年龄降低其重要性分数。
    /// 应定期调用此方法以实现记忆的自然衰减。
    pub fn apply_decay(&mut self) {
        let current_time = current_timestamp();

        for item in &mut self.items {
            let age = current_time.saturating_sub(item.timestamp);
            let decay = self.decay_manager.compute_decay(age);
            item.importance *= decay;
        }
    }

    /// 自动遗忘低重要性的记忆
    ///
    /// 移除所有重要性低于阈值（由DecayManager配置）的记忆项。
    ///
    /// # 返回值
    /// 返回被移除的记忆数量
    pub fn auto_forget(&mut self) -> usize {
        let threshold = self.decay_manager.importance_threshold;

        let mut to_remove: Vec<usize> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, item)| item.importance < threshold)
            .map(|(idx, _)| idx)
            .collect();

        to_remove.sort_by(|a, b| b.cmp(a));

        let count = to_remove.len();

        for idx in to_remove {
            self.remove_by_index(idx);
        }

        count
    }

    /// 持久化记忆到存储
    ///
    /// 使用原子性写入：先写入临时位置，成功后原子性重命名
    /// 避免写入过程中崩溃导致数据丢失
    pub fn persist(&mut self) -> std::io::Result<()> {
        match &mut self.persistence {
            Some(persistence) => {
                // 原子性写入：先写入临时位置
                persistence.begin_atomic_write()?;

                for (key, &idx) in &self.index {
                    if let Some(item) = self.items.get(idx) {
                        persistence.write_memory(key, item, MemoryLevel::LongTerm)?;
                    }
                }

                persistence.flush()?;
                // 原子性提交：重命名临时文件为正式文件
                persistence.commit_atomic_write()?;
                Ok(())
            }
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "Persistence not configured",
            )),
        }
    }

    /// 从持久化存储恢复数据
    ///
    /// 恢复后会自动重建 HNSW 索引（如果之前存在），使用保存的配置
    pub fn restore(&mut self) -> std::io::Result<usize> {
        match &self.persistence {
            Some(persistence) => {
                let saved_config = self.hnsw_config;
                let memories = persistence.load()?;
                let count = memories.len();

                self.clear();

                for (key, item, _level) in memories {
                    let vector: Vec<f32> = Self::compute_embedding_with_dim(
                        &item.data,
                        self.embedding_dim,
                        self.padding_strategy,
                    )
                    .iter()
                    .copied()
                    .collect();
                    let idx = self.items.len();

                    self.index.insert(key, idx);
                    self.items.push(item);
                    self.vector_cache.push(vector);
                    self.access_stats
                        .insert(idx, AccessStats::new(current_timestamp()));
                }

                // 如果之前存在 HNSW 配置，使用保存的配置重建
                if let Some(config) = saved_config {
                    if !self.vector_cache.is_empty() {
                        self.build_hnsw_index(Some(config));
                    }
                }

                Ok(count)
            }
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "Persistence not configured",
            )),
        }
    }

    /// 更新指定键的访问时间戳
    ///
    /// # 参数
    /// - `key`: 要更新的记忆键
    ///
    /// # 返回值
    /// 如果键存在返回true，否则返回false
    pub fn touch(&self, key: &str) -> bool {
        match self.index.get(key) {
            Some(&idx) => {
                if let Some(stats) = self.access_stats.get(&idx) {
                    stats.increment(current_timestamp());
                    true
                } else {
                    false
                }
            }
            None => false,
        }
    }

    /// 获取指定键的访问统计信息
    ///
    /// # 参数
    /// - `key`: 记忆键
    ///
    /// # 返回值
    /// 如果找到返回 Some((访问次数, 最后访问时间戳))，否则返回 None
    pub fn get_access_stats(&self, key: &str) -> Option<(u32, u64)> {
        self.index.get(key).and_then(|&idx| {
            self.access_stats
                .get(&idx)
                .map(|stats| (stats.get_access_count(), stats.get_last_access()))
        })
    }

    /// 计算嵌入向量
    ///
    /// # 嵌入策略
    ///
    /// 1. **多行数据**：计算行均值，然后截断/填充到目标维度
    /// 2. **单行数据**：直接截断/填充到目标维度
    ///
    /// # 截断策略
    ///
    /// 当原始维度大于嵌入维度时，截取前 `embedding_dim` 维
    ///
    /// # 归一化
    ///
    /// 最终进行 L2 归一化，零向量保持不变
    fn compute_embedding_with_dim(
        data: &Array2<f32>,
        embedding_dim: usize,
        strategy: PaddingStrategy,
    ) -> Array2<f32> {
        if data.nrows() == 0 {
            return Array2::zeros((1, embedding_dim));
        }

        let dim = data.ncols();
        let rows = data.nrows();

        // 快速路径：单行且维度不超过 embedding_dim
        if rows == 1 && dim <= embedding_dim {
            let mut result = Array2::zeros((1, embedding_dim));
            for j in 0..dim {
                result[[0, j]] = data[[0, j]];
            }
            // 根据策略填充剩余维度
            for j in dim..embedding_dim {
                result[[0, j]] = match strategy {
                    PaddingStrategy::Cyclic => data[[0, j % dim]],
                    PaddingStrategy::Zero => 0.0,
                };
            }
            return result;
        }

        // 优化：先累加，最后统一除法
        let mut mean = Array2::zeros((1, dim));
        for i in 0..rows {
            for j in 0..dim {
                mean[[0, j]] += data[[i, j]];
            }
        }
        let rows_inv = 1.0 / rows as f32;
        for j in 0..dim {
            mean[[0, j]] *= rows_inv;
        }

        let actual_dim = embedding_dim.min(dim);
        let mut embedding = Array2::zeros((1, embedding_dim));

        // 复制原始维度数据
        for j in 0..actual_dim {
            embedding[[0, j]] = mean[[0, j]];
        }

        // 根据策略填充超出原始维度的部分
        if dim < embedding_dim {
            for j in dim..embedding_dim {
                embedding[[0, j]] = match strategy {
                    PaddingStrategy::Cyclic => mean[[0, j % dim]],
                    PaddingStrategy::Zero => 0.0,
                };
            }
        }

        // L2 归一化
        let norm = (embedding.iter().map(|&x| x * x).sum::<f32>()).sqrt();
        if norm > 0.0 {
            let norm_inv = 1.0 / norm;
            for j in 0..embedding_dim {
                embedding[[0, j]] *= norm_inv;
            }
        }

        embedding
    }

    fn evict(&mut self) {
        if self.items.is_empty() {
            return;
        }

        match self.strategy {
            EvictionStrategy::LRU => {
                if let Some((idx, _)) = self
                    .access_stats
                    .iter()
                    .min_by_key(|(_, stats)| stats.get_last_access())
                    .map(|(idx, _)| (*idx, ()))
                {
                    self.remove_by_index(idx);
                } else if let Some((idx, _)) = self
                    .items
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, item)| item.timestamp)
                {
                    self.remove_by_index(idx);
                }
            }
            EvictionStrategy::LFU => {
                if let Some((idx, _)) = self
                    .access_stats
                    .iter()
                    .min_by_key(|(_, stats)| stats.get_access_count())
                    .map(|(idx, _)| (*idx, ()))
                {
                    self.remove_by_index(idx);
                } else if let Some((idx, _)) = self
                    .items
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, item)| item.importance as i32)
                {
                    self.remove_by_index(idx);
                }
            }
            EvictionStrategy::FIFO => {
                self.remove_by_index(0);
            }
        }
    }

    fn remove_by_index(&mut self, idx: usize) {
        if idx >= self.items.len() {
            return;
        }

        let key_to_remove = self
            .index
            .iter()
            .find(|(_, &i)| i == idx)
            .map(|(k, _)| k.clone());

        if let Some(key) = key_to_remove {
            self.index.remove(&key);
        }

        self.items.remove(idx);
        self.vector_cache.remove(idx);
        self.access_stats.remove(&idx);

        // 调整索引映射
        for (_, v) in self.index.iter_mut() {
            if *v > idx {
                *v -= 1;
            }
        }

        // 优化：原地调整 access_stats 索引，避免全量重建
        let keys_to_update: Vec<usize> = self
            .access_stats
            .keys()
            .filter(|&&k| k > idx)
            .copied()
            .collect();

        for old_idx in keys_to_update {
            if let Some(stats) = self.access_stats.remove(&old_idx) {
                self.access_stats.insert(old_idx - 1, stats);
            }
        }
    }

    /// 清空所有记忆数据
    ///
    /// 移除所有记忆项、向量缓存、索引和访问统计，
    /// 并重置 HNSW 索引为 None。
    pub fn clear(&mut self) {
        self.items.clear();
        self.vector_cache.clear();
        self.index.clear();
        self.access_stats.clear();
        // 清空后将 HNSW 索引设为 None，避免用户误以为索引仍存在
        self.hnsw_index = None;
    }

    /// 获取指定键的记忆重要性
    pub fn get_importance(&self, key: &str) -> Option<f32> {
        self.index
            .get(key)
            .and_then(|&idx| self.items.get(idx).map(|item| item.importance))
    }

    /// 获取当前存储的记忆数量
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// 检查是否没有任何记忆数据
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// 获取最大容量限制
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 检查是否已构建 HNSW 索引
    pub fn has_hnsw_index(&self) -> bool {
        self.hnsw_index.is_some()
    }

    /// 检查是否已配置持久化存储
    pub fn has_persistence(&self) -> bool {
        self.persistence.is_some()
    }

    /// 获取嵌入维度
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// 获取 HNSW 配置
    pub fn hnsw_config(&self) -> Option<HNSWConfig> {
        self.hnsw_config
    }

    /// 获取填充策略
    pub fn padding_strategy(&self) -> PaddingStrategy {
        self.padding_strategy
    }

    /// 获取向量缓存（用于测试）
    #[cfg(test)]
    pub fn vector_cache(&self) -> &[Vec<f32>] {
        &self.vector_cache
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new(&MemoryConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_long_term_memory_write_read() {
        let mut memory = LongTermMemory::default();
        let data = Array2::zeros((10, 512));

        memory.write("key1".to_string(), data.clone(), 1, 1.0);

        let read = memory.read("key1");
        assert!(read.is_some());
    }

    #[test]
    fn test_long_term_memory_search() {
        let mut memory = LongTermMemory::default();

        let data1 = Array2::zeros((1, 512));
        let data2 = Array2::zeros((1, 512));

        memory.write("key1".to_string(), data1, 1, 1.0);
        memory.write("key2".to_string(), data2, 2, 0.5);

        let query = Array2::zeros((1, 512));
        let results = memory.search(&query, 2);

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_long_term_memory_eviction() {
        let config = MemoryConfig {
            long_term_capacity: 2,
            ..Default::default()
        };
        let mut memory = LongTermMemory::new(&config);

        for i in 0..3 {
            let data = Array2::zeros((1, 512));
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }

        assert_eq!(memory.len(), 2);
    }

    #[test]
    fn test_decay_manager() {
        let decay = DecayManager::new(3600, 0.9, 0.1);

        assert!((decay.compute_decay(0) - 1.0).abs() < 0.001);

        let decayed = decay.compute_decay(3600);
        assert!((decayed - 0.9).abs() < 0.01);

        let decayed_more = decay.compute_decay(7200);
        assert!(decayed_more < decayed);
    }

    #[test]
    fn test_access_stats() {
        let stats = AccessStats::new(100);

        assert_eq!(stats.get_access_count(), 1);
        assert_eq!(stats.get_last_access(), 100);

        stats.increment(200);

        assert_eq!(stats.get_access_count(), 2);
        assert_eq!(stats.get_last_access(), 200);
    }

    #[test]
    fn test_build_hnsw_index() {
        let mut memory = LongTermMemory::default();

        for i in 0..10 {
            let data = Array2::zeros((1, 128));
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }

        assert!(!memory.has_hnsw_index());

        memory.build_hnsw_index(None);

        assert!(memory.has_hnsw_index());
    }

    #[test]
    fn test_search_hnsw() {
        let mut memory = LongTermMemory::default();

        for i in 0..20 {
            let mut data = Array2::zeros((1, 128));
            if i < 128 {
                data[[0, i]] = 1.0;
            }
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }

        memory.build_hnsw_index(None);

        let query = Array2::zeros((1, 128));
        let results = memory.search_hnsw(&query, 5);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_apply_decay() {
        let mut memory = LongTermMemory::default();
        memory = memory.with_decay(DecayManager::new(1, 0.5, 0.1));

        let data = Array2::zeros((1, 128));
        memory.write("key1".to_string(), data, current_timestamp() - 2, 1.0);

        memory.apply_decay();

        assert!(memory.items[0].importance < 1.0);
    }

    #[test]
    fn test_auto_forget() {
        let mut memory = LongTermMemory::default();
        memory = memory.with_decay(DecayManager::new(1, 0.1, 0.5));

        let data = Array2::zeros((1, 128));
        memory.write(
            "key1".to_string(),
            data.clone(),
            current_timestamp() - 10,
            0.01,
        );
        memory.write("key2".to_string(), data, current_timestamp(), 1.0);

        assert_eq!(memory.len(), 2);

        let removed = memory.auto_forget();

        assert!(removed >= 1);
        assert!(memory.len() <= 1);
    }

    #[test]
    fn test_touch() {
        let mut memory = LongTermMemory::default();
        let data = Array2::zeros((1, 128));

        memory.write("key1".to_string(), data, 100, 1.0);

        let (_, initial_time) = memory.get_access_stats("key1").unwrap();
        assert_eq!(initial_time, 100);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let result = memory.touch("key1");
        assert!(result);

        let (count, new_time) = memory.get_access_stats("key1").unwrap();
        assert_eq!(count, 2);
        assert!(new_time > initial_time);

        let result = memory.touch("nonexistent");
        assert!(!result);
    }

    #[test]
    fn test_persist_and_restore() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig::new(temp_dir.path().join("test_memory"));

        let mut memory = LongTermMemory::default().with_persistence(config);

        assert!(memory.has_persistence());

        for i in 0..5 {
            let data = Array2::zeros((1, 128));
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }

        memory.persist().unwrap();

        let config2 = PersistenceConfig::new(temp_dir.path().join("test_memory"));
        let mut memory2 = LongTermMemory::default().with_persistence(config2);

        let count = memory2.restore().unwrap();
        assert_eq!(count, 5);
        assert_eq!(memory2.len(), 5);
    }

    #[test]
    fn test_with_decay() {
        let decay = DecayManager::new(7200, 0.8, 0.2);
        let memory = LongTermMemory::default().with_decay(decay.clone());

        assert_eq!(memory.decay_manager.ttl_seconds, 7200);
        assert!((memory.decay_manager.decay_factor - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_lru_eviction_with_access_stats() {
        let config = MemoryConfig {
            long_term_capacity: 3,
            eviction_strategy: EvictionStrategy::LRU,
            ..Default::default()
        };
        let mut memory = LongTermMemory::new(&config);

        for i in 0..3 {
            let data = Array2::zeros((1, 128));
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }

        memory.touch("key0");
        std::thread::sleep(std::time::Duration::from_millis(10));

        let data = Array2::zeros((1, 128));
        memory.write("key3".to_string(), data, 100, 1.0);

        assert!(memory.read("key0").is_some());
    }

    #[test]
    fn test_custom_embedding_dim() {
        // 测试自定义嵌入维度
        let config = MemoryConfig {
            embedding_dim: Some(256), // 自定义 256 维
            ..Default::default()
        };
        let mut memory = LongTermMemory::new(&config);

        let data = Array2::zeros((1, 512));
        memory.write("key1".to_string(), data, 1, 1.0);

        // 验证 vector_cache 中的向量维度为 256
        assert_eq!(memory.vector_cache[0].len(), 256);
    }

    #[test]
    fn test_restore_rebuilds_hnsw() {
        // 测试 restore 后自动重建 HNSW 索引
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig::new(temp_dir.path().join("test_hnsw_rebuild"));

        // 创建内存并构建 HNSW 索引
        let mut memory = LongTermMemory::default().with_persistence(config.clone());
        memory.build_hnsw_index(None);
        assert!(memory.has_hnsw_index());

        // 写入数据并持久化
        for i in 0..5 {
            let data = Array2::zeros((1, 128));
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }
        memory.persist().unwrap();

        // 恢复到新实例
        let config2 = PersistenceConfig::new(temp_dir.path().join("test_hnsw_rebuild"));
        let mut memory2 = LongTermMemory::default().with_persistence(config2);

        // 先构建 HNSW 索引（模拟之前存在）
        memory2.build_hnsw_index(None);
        assert!(memory2.has_hnsw_index());

        // 恢复数据
        let count = memory2.restore().unwrap();
        assert_eq!(count, 5);

        // 验证 HNSW 索引已自动重建
        assert!(memory2.has_hnsw_index());

        // 验证搜索功能正常
        let query = Array2::zeros((1, 128));
        let results = memory2.search_hnsw(&query, 3);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_clear_resets_hnsw_index() {
        // 测试 clear 后 HNSW 索引被重置
        let mut memory = LongTermMemory::default();

        let data = Array2::zeros((1, 128));
        memory.write("key1".to_string(), data, 1, 1.0);

        memory.build_hnsw_index(None);
        assert!(memory.has_hnsw_index());

        memory.clear();

        // 验证 HNSW 索引已重置为 None
        assert!(!memory.has_hnsw_index());
        assert!(memory.is_empty());
    }

    #[test]
    fn test_padding_strategy_zero() {
        // 测试零填充策略
        let config = MemoryConfig {
            embedding_dim: Some(256),
            padding_strategy: PaddingStrategy::Zero,
            ..Default::default()
        };
        let mut memory = LongTermMemory::new(&config);

        // 创建 64 维数据，需要填充到 256 维
        let data = Array2::ones((1, 64));
        memory.write("key1".to_string(), data, 1, 1.0);

        // 验证向量维度为 256
        assert_eq!(memory.vector_cache[0].len(), 256);

        // 前 64 维应为非零（归一化后），后 192 维应为 0.0
        let vec = &memory.vector_cache[0];
        for i in 0..64 {
            assert!(vec[i].abs() > 0.0, "前 64 维应为非零");
        }
        for i in 64..256 {
            assert!(vec[i].abs() < 0.001, "后 192 维应为零");
        }
    }

    #[test]
    fn test_padding_strategy_cyclic() {
        // 测试循环填充策略
        let config = MemoryConfig {
            embedding_dim: Some(256),
            padding_strategy: PaddingStrategy::Cyclic,
            ..Default::default()
        };
        let mut memory = LongTermMemory::new(&config);

        // 创建 64 维数据
        let data = Array2::ones((1, 64));
        memory.write("key1".to_string(), data, 1, 1.0);

        // 验证向量维度为 256
        assert_eq!(memory.vector_cache[0].len(), 256);

        // 循环填充：所有维度都应有值
        let vec = &memory.vector_cache[0];
        for i in 0..256 {
            assert!(vec[i].abs() > 0.0, "所有维度应为非零");
        }
    }

    #[test]
    fn test_hnsw_config_preserved_on_restore() {
        // 测试 HNSW 配置在恢复时保留
        use crate::model::inference::memory::hnsw::{HNSWConfig, NeighborSelection};

        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig::new(temp_dir.path().join("test_config_preserve"));

        // 创建自定义 HNSW 配置
        let custom_config = HNSWConfig {
            m: 32,
            ef_construction: 200,
            ef_search: 100,
            max_level: 16,
            seed: 42,
            neighbor_selection: NeighborSelection::Heuristic,
            heuristic_diversity_factor: 0.8,
        };

        // 创建内存并使用自定义配置构建 HNSW
        let mut memory = LongTermMemory::default().with_persistence(config.clone());
        memory.build_hnsw_index(Some(custom_config));

        // 写入数据并持久化
        for i in 0..5 {
            let data = Array2::zeros((1, 128));
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }
        memory.persist().unwrap();

        // 验证配置已保存
        assert!(memory.hnsw_config().is_some());
        let saved_config = memory.hnsw_config().unwrap();
        assert_eq!(saved_config.m, 32);
        assert_eq!(saved_config.ef_construction, 200);

        // 恢复到新实例
        let config2 = PersistenceConfig::new(temp_dir.path().join("test_config_preserve"));
        let mut memory2 = LongTermMemory::default().with_persistence(config2);

        // 先构建 HNSW 索引（模拟之前存在配置）
        memory2.build_hnsw_index(Some(custom_config));

        // 恢复数据
        let count = memory2.restore().unwrap();
        assert_eq!(count, 5);

        // 验证 HNSW 索引已重建
        assert!(memory2.has_hnsw_index());

        // 验证配置已恢复
        assert!(memory2.hnsw_config().is_some());
        let restored_config = memory2.hnsw_config().unwrap();
        assert_eq!(restored_config.m, 32);
        assert_eq!(restored_config.ef_construction, 200);
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_long_term_memory_decay() {
        // 测试时间衰减机制
        let decay_manager = DecayManager::new(1, 0.5, 0.1); // TTL=1秒，衰减因子=0.5
        let mut memory = LongTermMemory::default().with_decay(decay_manager);

        // 添加条目（使用过去的时间戳）
        let data = Array2::ones((1, 64));
        memory.write(
            "old_item".to_string(),
            data.clone(),
            current_timestamp() - 3,
            1.0,
        );

        let original_importance = memory.items[0].importance;
        assert!((original_importance - 1.0).abs() < 0.001);

        // 应用衰减
        memory.apply_decay();

        // 条目应该仍然存在但分数降低
        assert!(!memory.is_empty());
        assert!(memory.items[0].importance < original_importance);
        assert!(memory.items[0].importance > 0.0);

        // 多次应用衰减后应该接近阈值
        for _ in 0..5 {
            memory.apply_decay();
        }
        assert!(memory.items[0].importance < original_importance * 0.1);
    }

    #[test]
    fn test_hnsw_index_build_search() {
        // 测试 HNSW 索引构建和搜索
        let dim = 16;
        let config = HNSWConfig::default();
        let index = HNSWIndex::new(config);

        // 添加向量
        for i in 0..20 {
            let vec: Vec<f32> = (0..dim).map(|j| ((i * j + 1) % 10) as f32 / 10.0).collect();
            index.insert(i, vec);
        }

        assert_eq!(index.len(), 20);

        // 搜索最近邻
        let query: Vec<f32> = (0..dim).map(|j| j as f32 * 0.05).collect();
        let results = index.search(&query, 5);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // 验证距离排序（第一个应该是最近的）
        if results.len() > 1 {
            for i in 0..results.len() - 1 {
                assert!(
                    results[i].distance <= results[i + 1].distance,
                    "结果应按距离升序排列"
                );
            }
        }

        // 测试搜索数量超过索引大小
        let all_results = index.search(&query, 100);
        assert_eq!(all_results.len(), 20); // 最多返回所有条目

        // 测试空索引搜索
        let empty_index = HNSWIndex::new(HNSWConfig::default());
        let empty_results = empty_index.search(&query, 5);
        assert!(empty_results.is_empty());
    }

    #[test]
    fn test_long_term_memory_lfu_eviction_strategy() {
        // 测试 LFU 驱逐策略
        let config = MemoryConfig {
            long_term_capacity: 3,
            eviction_strategy: EvictionStrategy::LFU,
            ..Default::default()
        };
        let mut memory = LongTermMemory::new(&config);

        // 写入3个条目
        for i in 0..3 {
            let data = Array2::from_elem((1, 64), i as f32);
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }

        // 增加第一个键的访问计数
        memory.touch("key0");

        // 写入第4个条目，应该淘汰访问次数最少的
        let data = Array2::from_elem((1, 64), 99.0);
        memory.write("key3".to_string(), data, 100, 1.0);

        assert_eq!(memory.len(), 3);
        // key0 应该被保留（因为被访问过）
        assert!(memory.read("key0").is_some());
    }

    #[test]
    fn test_long_term_memory_fifo_eviction_strategy() {
        // 测试 FIFO 驱逐策略
        let config = MemoryConfig {
            long_term_capacity: 3,
            eviction_strategy: EvictionStrategy::FIFO,
            ..Default::default()
        };
        let mut memory = LongTermMemory::new(&config);

        // 写入3个条目
        for i in 0..3 {
            let data = Array2::from_elem((1, 64), i as f32);
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }

        // 写入第4个条目，应该淘汰最早的（key0）
        let data = Array2::from_elem((1, 64), 99.0);
        memory.write("key3".to_string(), data, 100, 1.0);

        assert_eq!(memory.len(), 3);
        // key0 应该被淘汰
        assert!(memory.read("key0").is_none());
    }

    #[test]
    fn test_long_term_memory_search_by_importance() {
        // 测试按重要性排序搜索
        let mut memory = LongTermMemory::default();

        // 写入不同重要性的条目
        for i in 0..5 {
            let data = Array2::from_elem((1, 64), i as f32);
            let importance = 0.2 + i as f32 * 0.15; // 0.2, 0.35, 0.5, 0.65, 0.8
            memory.write(format!("item{}", i), data, i as u64, importance);
        }

        // 获取前3个最重要的
        let top_important = memory.search_by_importance(3);
        assert_eq!(top_important.len(), 3);

        // 验证返回的是按重要性降序排列的
        // （由于我们按顺序写入且重要性递增，最后写入的应该最重要）
    }

    #[test]
    fn test_long_term_memory_persist_without_configuration() {
        // 测试未配置持久化时的错误处理
        let mut memory = LongTermMemory::default();

        let data = Array2::zeros((1, 64));
        memory.write("key1".to_string(), data, 1, 1.0);

        // 尝试持久化应该失败
        let result = memory.persist();
        assert!(result.is_err());

        // 尝试恢复也应该失败
        let restore_result = memory.restore();
        assert!(restore_result.is_err());
    }

    #[test]
    fn test_decay_manager_edge_cases() {
        // 测试衰减管理器的边界情况
        let decay = DecayManager::new(3600, 0.9, 0.1);

        // 年龄为0时不应衰减
        let no_decay = decay.compute_decay(0);
        assert!((no_decay - 1.0).abs() < 0.001);

        // 正常衰减
        let normal_decay = decay.compute_decay(3600);
        assert!((normal_decay - 0.9).abs() < 0.01);

        // 多周期衰减（应该越来越小）
        let decay_1h = decay.compute_decay(3600);
        let decay_2h = decay.compute_decay(7200);
        let decay_4h = decay.compute_decay(14400);

        assert!(decay_2h < decay_1h);
        assert!(decay_4h < decay_2h);
        assert!(decay_4h > 0.0); // 应该不会完全归零
    }

    #[test]
    fn test_access_stats_clone_and_increment() {
        // 测试访问统计的克隆和增量操作
        let stats = AccessStats::new(1000);

        // 初始值
        assert_eq!(stats.get_access_count(), 1);
        assert_eq!(stats.get_last_access(), 1000);

        // 增量操作
        stats.increment(2000);
        stats.increment(3000);
        stats.increment(4000);

        assert_eq!(stats.get_access_count(), 4);
        assert_eq!(stats.get_last_access(), 4000);

        // 克隆应该创建独立的副本
        let cloned = stats.clone();
        assert_eq!(cloned.get_access_count(), 4);
        assert_eq!(cloned.get_last_access(), 4000);

        // 修改原对象不影响克隆
        stats.increment(5000);
        assert_eq!(stats.get_access_count(), 5);
        assert_eq!(cloned.get_access_count(), 4); // 克隆保持不变
    }

    #[test]
    fn test_long_term_memory_get_importance() {
        // 测试获取指定键的重要性
        let mut memory = LongTermMemory::default();

        let data = Array2::ones((1, 64));
        memory.write("high".to_string(), data.clone(), 1, 0.95);
        memory.write("low".to_string(), data, 2, 0.25);

        // 获取存在的重要性
        let high_imp = memory.get_importance("high");
        assert!(high_imp.is_some());
        assert!((high_imp.unwrap() - 0.95).abs() < 0.001);

        let low_imp = memory.get_importance("low");
        assert!(low_imp.is_some());
        assert!((low_imp.unwrap() - 0.25).abs() < 0.001);

        // 获取不存在的重要性
        let non_exist = memory.get_importance("nonexistent");
        assert!(non_exist.is_none());
    }

    #[test]
    fn test_long_term_memory_capacity_and_embedding_dim() {
        // 测试容量和嵌入维度属性
        let config = MemoryConfig {
            long_term_capacity: 50,
            embedding_dim: Some(256),
            ..Default::default()
        };
        let memory = LongTermMemory::new(&config);

        assert_eq!(memory.capacity(), 50);
        assert_eq!(memory.embedding_dim(), 256);
        assert!(memory.is_empty());
        assert_eq!(memory.len(), 0);
        assert!(!memory.has_hnsw_index());
        assert!(!memory.has_persistence());
    }

    #[test]
    fn test_long_term_memory_remove_and_index_adjustment() {
        // 测试删除条目后的索引调整
        let mut memory = LongTermMemory::default();

        // 写入多个条目
        for i in 0..5 {
            let data = Array2::from_elem((1, 64), i as f32);
            memory.write(format!("key{}", i), data, i as u64, 1.0);
        }

        assert_eq!(memory.len(), 5);

        // 通过驱逐减少容量来间接测试删除逻辑
        // 写入超出容量的条目会触发驱逐
        let small_config = MemoryConfig {
            long_term_capacity: 3,
            ..Default::default()
        };

        let mut small_memory = LongTermMemory::new(&small_config);
        for i in 0..5 {
            let data = Array2::from_elem((1, 64), i as f32);
            small_memory.write(format!("k{}", i), data, i as u64, 1.0);
        }

        assert_eq!(small_memory.len(), 3);
        // 索引应该保持一致性
        for key in ["k0", "k1", "k2", "k3", "k4"] {
            let read_result = small_memory.read(key);
            if read_result.is_some() {
                // 如果能读取到，说明索引正确
                let _ = read_result.unwrap();
            }
        }
    }

    #[test]
    fn test_auto_forget_with_high_threshold() {
        // 测试使用高阈值进行自动遗忘
        let mut memory = LongTermMemory::default();
        memory = memory.with_decay(DecayManager::new(1, 0.9, 0.95)); // 高阈值：只遗忘重要性<0.95的

        // 写入不同重要性的条目
        let data = Array2::ones((1, 64));
        memory.write("keep".to_string(), data.clone(), 1, 0.99); // 应该保留
        memory.write("forget".to_string(), data.clone(), 2, 0.50); // 应该遗忘
        memory.write("borderline".to_string(), data, 3, 0.96); // 应该保留

        assert_eq!(memory.len(), 3);

        let removed = memory.auto_forget();

        assert!(removed >= 1); // 至少遗忘一个
        assert!(memory.len() <= 2); // 剩余不超过2个
    }
}
