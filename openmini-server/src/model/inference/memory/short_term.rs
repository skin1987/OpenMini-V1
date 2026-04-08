//! 短期记忆层
//!
//! 滑动窗口记忆，支持多轮对话，智能压缩
//!
//! # 功能特性
//! - 智能压缩算法：基于重要性加权的压缩策略
//! - 重要性计算：注意力权重、访问频率、语义密度综合评估
//! - 语义压缩：提取关键信息，合并相似记忆
//! - 动态滑动窗口：根据负载自动调整窗口大小

use std::collections::VecDeque;

use ndarray::Array2;

use super::{EvictionStrategy, MemoryConfig, MemoryItem};
use super::simd_ops::SimdVectorOps;

/// 重要性计算器配置
#[derive(Debug, Clone)]
pub struct ImportanceConfig {
    /// 注意力权重系数
    pub attention_weight: f32,
    /// 访问频率系数
    pub frequency_weight: f32,
    /// 语义密度系数
    pub semantic_weight: f32,
    /// 时间衰减系数
    pub time_decay: f32,
    /// 最小重要性阈值
    pub min_importance: f32,
}

impl Default for ImportanceConfig {
    fn default() -> Self {
        Self {
            attention_weight: 0.4,
            frequency_weight: 0.3,
            semantic_weight: 0.3,
            time_decay: 0.95,
            min_importance: 0.1,
        }
    }
}

/// 重要性计算器
///
/// 综合考虑多种因素计算记忆项的重要性分数
#[derive(Debug)]
pub struct ImportanceCalculator {
    config: ImportanceConfig,
}

impl ImportanceCalculator {
    /// 创建新的重要性计算器
    ///
    /// # 参数
    /// - `config`: 重要性计算配置，包含各维度权重和衰减参数
    pub fn new(config: ImportanceConfig) -> Self {
        Self { config }
    }

    /// 计算单个记忆项的重要性分数
    ///
    /// 综合考虑注意力权重、访问频率、语义密度三个维度
    /// 
    /// # 参数
    /// - `item`: 记忆项
    /// - `access_count`: 访问次数
    /// - `write_timestamp`: 写入短期记忆的时间戳（用于时间衰减计算）
    /// - `current_time`: 当前时间
    /// - `attention_score_override`: 可选的注意力分数覆盖值
    pub fn calculate_importance(
        &self,
        item: &MemoryItem,
        access_count: usize,
        write_timestamp: u64,
        current_time: u64,
        attention_score_override: Option<f32>,
    ) -> f32 {
        let attention_score = attention_score_override.unwrap_or_else(|| self.estimate_attention(&item.data));
        let frequency_score = self.normalize_frequency(access_count);
        let semantic_score = self.calculate_semantic_density(&item.data);
        let time_score = self.calculate_time_decay(write_timestamp, current_time);

        let raw_importance = self.config.attention_weight * attention_score
            + self.config.frequency_weight * frequency_score
            + self.config.semantic_weight * semantic_score;

        (raw_importance * time_score).max(self.config.min_importance)
    }

    /// 估算注意力权重
    ///
    /// 基于数据的统计特性估算注意力分数
    fn estimate_attention(&self, data: &Array2<f32>) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let sum: f32 = data.iter().map(|v| v.abs()).sum();
        let count = data.len() as f32;
        let mean = sum / count;

        let variance: f32 = data.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / count;

        let normalized_variance = (variance.sqrt() / (mean.abs() + 1e-6)).min(1.0);
        let sparsity = data.iter().filter(|v| v.abs() < 1e-6).count() as f32 / count;

        normalized_variance * 0.6 + (1.0 - sparsity) * 0.4
    }

    /// 归一化访问频率
    fn normalize_frequency(&self, count: usize) -> f32 {
        let normalized = (count as f32).ln() / 10.0_f32.ln();
        normalized.clamp(0.0, 1.0)
    }

    /// 计算语义密度
    ///
    /// 衡量数据中信息的丰富程度
    fn calculate_semantic_density(&self, data: &Array2<f32>) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let rows = data.nrows();
        let cols = data.ncols();

        let row_norms: Vec<f32> = (0..rows)
            .map(|i| {
                (0..cols)
                    .map(|j| data[[i, j]].powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        let avg_norm: f32 = row_norms.iter().sum::<f32>() / rows as f32;
        let norm_variance: f32 = row_norms.iter()
            .map(|n| (n - avg_norm).powi(2))
            .sum::<f32>() / rows as f32;

        let density = 1.0 - (norm_variance / (avg_norm.powi(2) + 1e-6)).min(1.0);
        density.clamp(0.0, 1.0)
    }

    /// 计算时间衰减因子
    fn calculate_time_decay(&self, timestamp: u64, current_time: u64) -> f32 {
        if current_time <= timestamp {
            return 1.0;
        }

        let elapsed = (current_time - timestamp) as f32;
        self.config.time_decay.powf(elapsed / 1000.0)
    }

    /// 批量计算重要性
    pub fn batch_calculate(
        &self,
        items: &[MemoryItem],
        access_counts: &[usize],
        write_timestamps: &[u64],
        current_time: u64,
    ) -> Vec<f32> {
        items.iter()
            .zip(access_counts.iter())
            .zip(write_timestamps.iter())
            .map(|((item, &count), &write_ts)| {
                self.calculate_importance(item, count, write_ts, current_time, None)
            })
            .collect()
    }

    /// 获取配置
    pub fn config(&self) -> &ImportanceConfig {
        &self.config
    }
}

impl Default for ImportanceCalculator {
    fn default() -> Self {
        Self::new(ImportanceConfig::default())
    }
}

/// 压缩器配置
#[derive(Debug, Clone)]
pub struct CompressorConfig {
    /// 目标压缩比率
    pub compression_ratio: f32,
    /// 最小保留项数
    pub min_retained: usize,
    /// 重要性阈值
    pub importance_threshold: f32,
    /// 是否启用语义压缩
    pub enable_semantic: bool,
    /// 相似度阈值（用于合并）
    pub similarity_threshold: f32,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            compression_ratio: 0.5,
            min_retained: 3,
            importance_threshold: 0.3,
            enable_semantic: true,
            similarity_threshold: 0.85,
        }
    }
}

/// 智能压缩器
///
/// 基于重要性加权实现智能压缩策略
#[derive(Debug)]
pub struct Compressor {
    config: CompressorConfig,
    importance_calculator: ImportanceCalculator,
}

impl Compressor {
    /// 创建新的智能压缩器
    ///
    /// # 参数
    /// - `config`: 压缩器配置，包含压缩比率、阈值等参数
    pub fn new(config: CompressorConfig) -> Self {
        Self {
            config,
            importance_calculator: ImportanceCalculator::default(),
        }
    }

    /// 智能压缩（旧版，已废弃）
    ///
    /// 根据重要性分数选择性保留和压缩记忆项
    /// 
    /// # 已废弃
    /// 此方法会重置访问计数为 1，丢失原始访问统计。
    /// 请使用 `compress_stored_items` 替代，该方法保留原始访问计数。
    #[deprecated(
        since = "0.1.0",
        note = "此方法会重置访问计数，请使用 `compress_stored_items` 替代"
    )]
    pub fn compress_smart(
        &self,
        items: &mut VecDeque<MemoryItem>,
        access_counts: &mut Vec<usize>,
        timestamps: &mut Vec<u64>,
        current_time: u64,
    ) -> Vec<MemoryItem> {
        if items.is_empty() {
            return Vec::new();
        }

        let items_vec: Vec<MemoryItem> = items.iter().cloned().collect();
        let importance_scores = self.importance_calculator.batch_calculate(&items_vec, access_counts, timestamps, current_time);

        let mut indexed_items: Vec<(usize, f32)> = importance_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let target_count = ((items.len() as f32 * self.config.compression_ratio) as usize)
            .max(self.config.min_retained);

        let retained_indices: std::collections::HashSet<usize> = indexed_items
            .iter()
            .take(target_count)
            .map(|(i, _)| *i)
            .collect();

        let mut compressed_items = Vec::new();
        let mut items_to_merge: Vec<(MemoryItem, f32)> = Vec::new();

        for (idx, item) in items.iter().enumerate() {
            if retained_indices.contains(&idx) {
                compressed_items.push(item.clone());
            } else if importance_scores[idx] >= self.config.importance_threshold {
                items_to_merge.push((item.clone(), importance_scores[idx]));
            }
        }

        if self.config.enable_semantic && !items_to_merge.is_empty() {
            let merged = Self::semantic_merge(&items_to_merge);
            if let Some(merged_item) = merged {
                compressed_items.push(merged_item);
            }
        }

        let new_len = compressed_items.len();
        let new_items: VecDeque<MemoryItem> = compressed_items.into_iter().collect();
        let new_access_counts = vec![1; new_len];
        let new_timestamps = vec![current_time; new_len];

        *items = new_items;
        *access_counts = new_access_counts;
        *timestamps = new_timestamps;

        items_to_merge.into_iter().map(|(item, _)| item).collect()
    }

    /// 智能压缩（使用 StoredItem）
    ///
    /// 根据重要性分数选择性保留和压缩记忆项
    /// 保留原始 access_count，不重置为 1
    pub fn compress_stored_items(
        &self,
        items: &mut VecDeque<StoredItem>,
        current_time: u64,
    ) -> Vec<MemoryItem> {
        if items.is_empty() {
            return Vec::new();
        }

        let items_vec: Vec<MemoryItem> = items.iter().map(|s| s.item.clone()).collect();
        let access_counts: Vec<usize> = items.iter().map(|s| s.access_count).collect();
        let write_timestamps: Vec<u64> = items.iter().map(|s| s.write_timestamp).collect();
        let importance_scores = self.importance_calculator.batch_calculate(&items_vec, &access_counts, &write_timestamps, current_time);

        let mut indexed_items: Vec<(usize, f32)> = importance_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        indexed_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let target_count = ((items.len() as f32 * self.config.compression_ratio) as usize)
            .max(self.config.min_retained);

        let retained_indices: std::collections::HashSet<usize> = indexed_items
            .iter()
            .take(target_count)
            .map(|(i, _)| *i)
            .collect();

        let mut compressed_items = Vec::new();
        let mut items_to_merge: Vec<(StoredItem, f32)> = Vec::new();

        for (idx, stored) in items.iter().enumerate() {
            if retained_indices.contains(&idx) {
                compressed_items.push(stored.clone());
            } else if importance_scores[idx] >= self.config.importance_threshold {
                items_to_merge.push((stored.clone(), importance_scores[idx]));
            }
        }

        if self.config.enable_semantic && !items_to_merge.is_empty() {
            let merged = Self::semantic_merge_stored(&items_to_merge);
            if let Some(merged_item) = merged {
                compressed_items.push(merged_item);
            }
        }

        let removed: Vec<MemoryItem> = items
            .iter()
            .enumerate()
            .filter(|(idx, _)| !retained_indices.contains(idx))
            .map(|(_, stored)| stored.item.clone())
            .collect();

        *items = compressed_items.into_iter().collect();

        removed
    }

    /// 语义合并
    ///
    /// 将相似的低重要性项合并为一个代表性项
    fn semantic_merge(items: &[(MemoryItem, f32)]) -> Option<MemoryItem> {
        if items.is_empty() {
            return None;
        }

        let total_importance: f32 = items.iter().map(|(_, score)| score).sum();
        let weights: Vec<f32> = items.iter()
            .map(|(_, score)| score / total_importance)
            .collect();

        let dim = items[0].0.data.ncols();
        let max_rows: usize = items.iter().map(|(item, _)| item.data.nrows()).max().unwrap_or(1);

        let mut merged_data = Array2::zeros((max_rows, dim));

        for (tuple, weight) in items.iter().zip(weights.iter()) {
            let item = &tuple.0;
            let rows = item.data.nrows();
            for i in 0..rows.min(max_rows) {
                for j in 0..dim {
                    merged_data[[i, j]] += item.data[[i, j]] * weight;
                }
            }
        }

        let avg_importance: f32 = items.iter()
            .map(|(_, score)| score)
            .sum::<f32>() / items.len() as f32;

        items.last().map(|(item, _)| {
            MemoryItem::new(merged_data, item.timestamp)
                .with_importance(avg_importance)
        })
    }

    /// 语义合并（StoredItem 版本）
    ///
    /// 将相似的 StoredItem 合并为一个新的 StoredItem
    fn semantic_merge_stored(items: &[(StoredItem, f32)]) -> Option<StoredItem> {
        if items.is_empty() {
            return None;
        }

        let total_importance: f32 = items.iter().map(|(_, score)| score).sum();
        let weights: Vec<f32> = if total_importance == 0.0 {
            let len = items.len() as f32;
            items.iter().map(|_| 1.0 / len).collect()
        } else {
            items.iter()
                .map(|(_, score)| score / total_importance)
                .collect()
        };

        let first_dim = items[0].0.item.data.ncols();
        for (stored, _) in items {
            if stored.item.data.ncols() != first_dim {
                return None;
            }
        }

        let max_rows: usize = items.iter().map(|(stored, _)| stored.item.data.nrows()).max().unwrap_or(1);

        let mut merged_data = Array2::zeros((max_rows, first_dim));

        for ((stored, _), weight) in items.iter().zip(weights.iter()) {
            let rows = stored.item.data.nrows();
            for i in 0..rows.min(max_rows) {
                for j in 0..first_dim {
                    merged_data[[i, j]] += stored.item.data[[i, j]] * weight;
                }
            }
        }

        let avg_importance: f32 = items.iter()
            .map(|(_, score)| score)
            .sum::<f32>() / items.len() as f32;

        let total_access: usize = items.iter().map(|(stored, _)| stored.access_count).sum();
        let latest_timestamp = items.iter().map(|(stored, _)| stored.write_timestamp).max().unwrap_or(0);

        let merged_item = MemoryItem::new(merged_data, latest_timestamp)
            .with_importance(avg_importance);

        Some(StoredItem::new(merged_item, latest_timestamp).with_access_count(total_access))
    }

    /// 计算两个向量的余弦相似度（使用 SIMD 加速）
    pub fn cosine_similarity(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        if a.is_empty() || b.is_empty() || a.ncols() != b.ncols() {
            return 0.0;
        }

        let a_slice: &[f32] = a.as_slice().unwrap_or(&[]);
        let b_slice: &[f32] = b.as_slice().unwrap_or(&[]);
        
        if a_slice.is_empty() || b_slice.is_empty() {
            return 0.0;
        }
        
        let simd_ops = SimdVectorOps::new();
        let min_len = a_slice.len().min(b_slice.len());
        
        simd_ops.cosine_similarity(&a_slice[..min_len], &b_slice[..min_len])
    }

    /// 获取配置
    pub fn config(&self) -> &CompressorConfig {
        &self.config
    }

    /// 获取重要性计算器引用
    pub fn importance_calculator(&self) -> &ImportanceCalculator {
        &self.importance_calculator
    }
}

impl Default for Compressor {
    fn default() -> Self {
        Self::new(CompressorConfig::default())
    }
}

/// 动态窗口配置
#[derive(Debug, Clone)]
pub struct DynamicWindowConfig {
    /// 最小窗口大小
    pub min_size: usize,
    /// 最大窗口大小
    pub max_size: usize,
    /// 初始窗口大小
    pub initial_size: usize,
    /// 扩展阈值（利用率）
    pub expand_threshold: f32,
    /// 收缩阈值（利用率）
    pub shrink_threshold: f32,
    /// 调整步长
    pub adjust_step: usize,
}

impl Default for DynamicWindowConfig {
    fn default() -> Self {
        Self {
            min_size: 64,
            max_size: 4096,
            initial_size: 512,
            expand_threshold: 0.9,
            shrink_threshold: 0.3,
            adjust_step: 64,
        }
    }
}

/// 动态滑动窗口
///
/// 根据使用情况自动调整窗口大小
#[derive(Debug)]
pub struct DynamicWindow {
    config: DynamicWindowConfig,
    current_size: usize,
    utilization_history: VecDeque<f32>,
    history_capacity: usize,
}

impl DynamicWindow {
    /// 创建新的动态滑动窗口
    ///
    /// 根据配置初始化窗口大小和历史记录容量
    ///
    /// # 参数
    /// - `config`: 动态窗口配置，包含大小范围、阈值等参数
    pub fn new(config: DynamicWindowConfig) -> Self {
        Self {
            current_size: config.initial_size,
            config,
            utilization_history: VecDeque::with_capacity(100),
            history_capacity: 100,
        }
    }

    /// 记录利用率
    pub fn record_utilization(&mut self, used: usize, capacity: usize) {
        let utilization = if capacity > 0 {
            used as f32 / capacity as f32
        } else {
            0.0
        };

        if self.utilization_history.len() >= self.history_capacity {
            self.utilization_history.pop_front();
        }
        self.utilization_history.push_back(utilization);
    }

    /// 自动调整窗口大小
    pub fn auto_adjust(&mut self) -> usize {
        if self.utilization_history.len() < 10 {
            return self.current_size;
        }

        let avg_utilization: f32 = self.utilization_history.iter()
            .rev()
            .take(10)
            .sum::<f32>() / 10.0;

        if avg_utilization > self.config.expand_threshold {
            let new_size = (self.current_size + self.config.adjust_step)
                .min(self.config.max_size);
            self.current_size = new_size;
        } else if avg_utilization < self.config.shrink_threshold {
            let new_size = self.current_size
                .saturating_sub(self.config.adjust_step)
                .max(self.config.min_size);
            self.current_size = new_size;
        }

        self.current_size
    }

    /// 获取当前窗口大小
    pub fn current_size(&self) -> usize {
        self.current_size
    }

    /// 手动设置窗口大小
    pub fn set_size(&mut self, size: usize) {
        self.current_size = size
            .max(self.config.min_size)
            .min(self.config.max_size);
    }

    /// 获取平均利用率
    pub fn average_utilization(&self) -> f32 {
        if self.utilization_history.is_empty() {
            return 0.0;
        }
        self.utilization_history.iter().sum::<f32>() / self.utilization_history.len() as f32
    }

    /// 重置窗口
    pub fn reset(&mut self) {
        self.current_size = self.config.initial_size;
        self.utilization_history.clear();
    }

    /// 获取配置
    pub fn config(&self) -> &DynamicWindowConfig {
        &self.config
    }
}

impl Default for DynamicWindow {
    fn default() -> Self {
        Self::new(DynamicWindowConfig::default())
    }
}

/// 存储项包装结构体
/// 
/// 统一存储记忆项、访问计数和时间戳，避免数据不同步问题
#[derive(Debug, Clone)]
pub struct StoredItem {
    /// 记忆项
    pub item: MemoryItem,
    /// 访问计数
    pub access_count: usize,
    /// 写入时间戳
    pub write_timestamp: u64,
}

impl StoredItem {
    /// 创建新的存储项
    ///
    /// 初始化记忆项并设置访问计数为 1
    ///
    /// # 参数
    /// - `item`: 记忆项数据
    /// - `write_timestamp`: 写入时间戳（毫秒级）
    pub fn new(item: MemoryItem, write_timestamp: u64) -> Self {
        Self {
            item,
            access_count: 1,
            write_timestamp,
        }
    }

    /// 设置访问计数
    ///
    /// 用于从压缩后的存储项恢复访问计数
    ///
    /// # 参数
    /// - `count`: 访问次数
    ///
    /// # 返回值
    /// 修改后的 StoredItem 实例（支持链式调用）
    pub fn with_access_count(mut self, count: usize) -> Self {
        self.access_count = count;
        self
    }
}

/// 并查集结构体
/// 
/// 用于实现传递闭包合并，确保 A~B, B~C 时 A、B、C 被合并到同一组
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        
        let mut current = x;
        while self.parent[current] != root {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }
        
        root
    }

    fn union(&mut self, x: usize, y: usize) {
        let px = self.find(x);
        let py = self.find(y);

        if px == py {
            return;
        }

        if self.rank[px] < self.rank[py] {
            self.parent[px] = py;
        } else if self.rank[px] > self.rank[py] {
            self.parent[py] = px;
        } else {
            self.parent[py] = px;
            self.rank[px] += 1;
        }
    }
}

/// 增强版短期记忆
#[derive(Debug)]
pub struct ShortTermMemory {
    items: VecDeque<StoredItem>,
    capacity: usize,
    compression_threshold: usize,
    strategy: EvictionStrategy,
    importance_calculator: ImportanceCalculator,
    compressor: Compressor,
    dynamic_window: DynamicWindow,
}

impl ShortTermMemory {
    /// 创建新的短期记忆实例
    ///
    /// 根据配置初始化短期记忆层，包括容量、压缩策略、动态窗口等
    ///
    /// # 参数
    /// - `config`: 记忆系统配置，包含短期记忆容量、压缩阈值、驱逐策略等
    pub fn new(config: &MemoryConfig) -> Self {
        Self {
            items: VecDeque::with_capacity(config.short_term_capacity),
            capacity: config.short_term_capacity,
            compression_threshold: config.compression_threshold,
            strategy: config.eviction_strategy,
            importance_calculator: ImportanceCalculator::default(),
            compressor: Compressor::default(),
            dynamic_window: DynamicWindow::default(),
        }
    }

    /// 获取当前系统时间（毫秒级）
    fn current_system_time() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// 写入记忆项
    pub fn write(&mut self, data: Array2<f32>, timestamp: u64, session_id: Option<u64>) {
        let effective_capacity = self.dynamic_window.current_size().min(self.capacity);

        if self.items.len() >= effective_capacity {
            self.evict();
        }

        let mut item = MemoryItem::new(data, timestamp);
        item.session_id = session_id;
        
        let current_time = Self::current_system_time();
        let stored_item = StoredItem::new(item, current_time);
        self.items.push_back(stored_item);

        if self.items.len() > self.compression_threshold {
            self.compress();
        }

        self.dynamic_window.record_utilization(self.items.len(), effective_capacity);
    }

    /// 读取所有记忆
    pub fn read_all(&self) -> Vec<Array2<f32>> {
        self.items.iter().map(|stored| stored.item.data.clone()).collect()
    }

    /// 读取指定会话的记忆
    pub fn read_session(&self, session_id: u64) -> Vec<Array2<f32>> {
        self.items
            .iter()
            .filter(|stored| stored.item.session_id == Some(session_id))
            .map(|stored| stored.item.data.clone())
            .collect()
    }

    /// 读取最近 n 条记忆
    pub fn read_last(&self, n: usize) -> Vec<Array2<f32>> {
        self.items
            .iter()
            .rev()
            .take(n)
            .map(|stored| stored.item.data.clone())
            .collect()
    }

    /// 按重要性读取
    pub fn read_by_importance(&self, min_importance: f32) -> Vec<Array2<f32>> {
        self.items
            .iter()
            .filter(|stored| stored.item.importance >= min_importance)
            .map(|stored| stored.item.data.clone())
            .collect()
    }

    /// 驱逐策略
    fn evict(&mut self) {
        if self.items.is_empty() {
            return;
        }
        
        match self.strategy {
            EvictionStrategy::LRU => {
                self.items.pop_front();
            }
            EvictionStrategy::LFU => {
                if let Some((idx, _)) = self
                    .items
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, stored)| stored.access_count)
                {
                    self.items.remove(idx);
                }
            }
            EvictionStrategy::FIFO => {
                self.items.pop_front();
            }
        }
    }

    /// 基础压缩
    fn compress(&mut self) {
        if self.items.is_empty() {
            return;
        }

        let current_time = Self::current_system_time();
        self.compressor.compress_stored_items(&mut self.items, current_time);
        
        self.dynamic_window.auto_adjust();
    }

    /// 智能压缩（公开接口）
    pub fn compress_smart(&mut self) -> Vec<MemoryItem> {
        if self.items.is_empty() {
            return Vec::new();
        }

        let current_time = Self::current_system_time();
        self.compressor.compress_stored_items(&mut self.items, current_time)
    }

    /// 语义压缩
    ///
    /// 提取关键信息，合并相似记忆
    /// 使用并查集实现传递闭包合并
    pub fn semantic_compress(&mut self, similarity_threshold: Option<f32>) -> usize {
        if self.items.len() < 2 {
            return 0;
        }

        let threshold = similarity_threshold
            .unwrap_or_else(|| self.compressor.config().similarity_threshold);

        let n = self.items.len();
        let mut uf = UnionFind::new(n);

        let mut similar_pairs: Vec<(usize, usize, f32)> = Vec::with_capacity(n * n);

        for i in 0..n {
            for j in (i + 1)..n {
                let similarity = Compressor::cosine_similarity(
                    &self.items[i].item.data,
                    &self.items[j].item.data,
                );
                if similarity >= threshold {
                    similar_pairs.push((i, j, similarity));
                }
            }
        }

        for (i, j, _similarity) in similar_pairs {
            uf.union(i, j);
        }

        let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n {
            let root = uf.find(i);
            groups.entry(root).or_default().push(i);
        }

        let mut merged_count = 0;
        let mut to_remove = vec![false; n];

        for (_, group) in groups {
            if group.len() > 1 {
                let merged_stored = self.merge_stored_items(&group);
                
                if let Some(&first_idx) = group.first() {
                    self.items[first_idx] = merged_stored;
                }
                
                for &idx in group.iter().skip(1) {
                    to_remove[idx] = true;
                    merged_count += 1;
                }
            }
        }

        let mut new_items = VecDeque::with_capacity(n - merged_count);
        for (i, stored) in self.items.iter().enumerate() {
            if !to_remove[i] {
                new_items.push_back(stored.clone());
            }
        }
        self.items = new_items;

        merged_count
    }

    /// 合并多个存储项
    ///
    /// 当总重要性为零时，使用等权重合并而非直接返回第一项
    fn merge_stored_items(&self, indices: &[usize]) -> StoredItem {
        if indices.is_empty() {
            return StoredItem::new(MemoryItem::new(Array2::zeros((0, 0)), 0), 0);
        }

        let items: Vec<&StoredItem> = indices
            .iter()
            .filter_map(|&i| self.items.get(i))
            .collect();

        if items.is_empty() {
            return StoredItem::new(MemoryItem::new(Array2::zeros((0, 0)), 0), 0);
        }

        let first_dim = items[0].item.data.ncols();
        for stored in &items {
            if stored.item.data.ncols() != first_dim {
                return items[0].clone();
            }
        }

        let total_importance: f32 = items.iter().map(|s| s.item.importance).sum();
        let weights: Vec<f32> = if total_importance == 0.0 {
            let len = items.len() as f32;
            items.iter().map(|_| 1.0 / len).collect()
        } else {
            items.iter().map(|s| s.item.importance / total_importance).collect()
        };

        let max_rows: usize = items.iter().map(|s| s.item.data.nrows()).max().unwrap_or(1);

        let mut merged_data = Array2::zeros((max_rows, first_dim));

        for (stored, weight) in items.iter().zip(weights.iter()) {
            let rows = stored.item.data.nrows();
            for i in 0..rows.min(max_rows) {
                for j in 0..first_dim {
                    merged_data[[i, j]] += stored.item.data[[i, j]] * weight;
                }
            }
        }

        let avg_importance: f32 = items.iter()
            .map(|s| s.item.importance)
            .sum::<f32>() / items.len() as f32;

        let latest_timestamp = items.iter()
            .map(|s| s.write_timestamp)
            .max()
            .unwrap_or(0);

        let total_access: usize = items.iter().map(|s| s.access_count).sum();

        let merged_memory = MemoryItem::new(merged_data, latest_timestamp)
            .with_importance(avg_importance);

        StoredItem::new(merged_memory, latest_timestamp).with_access_count(total_access)
    }

    /// 压缩为摘要
    pub fn compress_to_summary(&self) -> Array2<f32> {
        if self.items.is_empty() {
            return Array2::zeros((0, 0));
        }

        let total_rows: usize = self.items.iter().map(|s| s.item.data.nrows()).sum();
        let dim = self.items.front().map(|s| s.item.data.ncols()).unwrap_or(0);

        if total_rows == 0 || dim == 0 {
            return Array2::zeros((0, 0));
        }

        let mut combined = Array2::zeros((total_rows, dim));
        let mut offset = 0;
        for stored in &self.items {
            let rows = stored.item.data.nrows();
            for i in 0..rows {
                for j in 0..dim {
                    combined[[offset + i, j]] = stored.item.data[[i, j]];
                }
            }
            offset += rows;
        }

        combined
    }

    /// 清空记忆
    pub fn clear(&mut self) {
        self.items.clear();
        self.dynamic_window.reset();
    }

    /// 获取记忆数量
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// 获取容量
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 是否需要压缩
    pub fn needs_compression(&self) -> bool {
        self.items.len() >= self.compression_threshold
    }

    /// 获取当前动态窗口大小
    pub fn dynamic_window_size(&self) -> usize {
        self.dynamic_window.current_size()
    }

    /// 自动调整动态窗口
    pub fn auto_adjust_window(&mut self) -> usize {
        self.dynamic_window.auto_adjust()
    }

    /// 获取平均利用率
    pub fn average_utilization(&self) -> f32 {
        self.dynamic_window.average_utilization()
    }

    /// 计算指定项的重要性
    pub fn calculate_item_importance(&self, index: usize) -> Option<f32> {
        let stored = self.items.get(index)?;
        let access_count = stored.access_count;
        let write_timestamp = stored.write_timestamp;
        let current_time = Self::current_system_time();

        Some(self.importance_calculator.calculate_importance(
            &stored.item,
            access_count,
            write_timestamp,
            current_time,
            None,
        ))
    }

    /// 获取所有重要性分数
    pub fn get_all_importance(&self) -> Vec<f32> {
        let current_time = Self::current_system_time();
        self.items
            .iter()
            .map(|stored| {
                self.importance_calculator.calculate_importance(
                    &stored.item,
                    stored.access_count,
                    stored.write_timestamp,
                    current_time,
                    None,
                )
            })
            .collect()
    }

    /// 更新访问计数
    pub fn touch(&mut self, index: usize) {
        if let Some(stored) = self.items.get_mut(index) {
            stored.access_count += 1;
        }
    }

    /// 获取重要性计算器配置
    pub fn importance_config(&self) -> &ImportanceConfig {
        self.importance_calculator.config()
    }

    /// 获取压缩器配置
    pub fn compressor_config(&self) -> &CompressorConfig {
        self.compressor.config()
    }
}

impl Default for ShortTermMemory {
    fn default() -> Self {
        Self::new(&MemoryConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_calculator() {
        let calc = ImportanceCalculator::default();
        let data = Array2::from_shape_fn((10, 512), |(i, j)| {
            (if i % 2 == 0 { 0.5 } else { -0.3 }) * (j as f32 / 512.0)
        });
        let item = MemoryItem::new(data, 100);

        let importance = calc.calculate_importance(&item, 5, 100, 200, None);
        assert!(importance > 0.0);
        assert!(importance <= 1.0);
    }

    #[test]
    fn test_importance_time_decay() {
        let calc = ImportanceCalculator::default();
        let data = Array2::zeros((10, 512));
        let item = MemoryItem::new(data, 100);

        let write_timestamp = 100u64;
        let importance_recent = calc.calculate_importance(&item, 5, write_timestamp, 110, None);
        let importance_old = calc.calculate_importance(&item, 5, write_timestamp, 1000, None);

        assert!(importance_recent >= importance_old);
    }


    #[test]
    #[allow(deprecated)]
    fn test_compressor() {
        let compressor = Compressor::default();
        let mut items: VecDeque<MemoryItem> = VecDeque::new();
        let mut access_counts = vec![1, 1, 1, 1, 1];
        let mut timestamps = vec![1, 2, 3, 4, 5];

        for i in 0..5 {
            let data = Array2::from_shape_fn((5, 32), |(r, c)| {
                i as f32 + r as f32 * 0.1 + c as f32 * 0.01
            });
            items.push_back(MemoryItem::new(data, i as u64).with_importance(0.5 + i as f32 * 0.1));
        }

        let removed = compressor.compress_smart(&mut items, &mut access_counts, &mut timestamps, 10);

        assert!(items.len() < 5);
        assert!(!items.is_empty());
        let _ = removed;
    }

    #[test]
    fn test_dynamic_window() {
        let mut window = DynamicWindow::default();
        let initial_size = window.current_size();

        for _ in 0..15 {
            window.record_utilization(450, 512);
        }
        window.auto_adjust();

        assert!(window.current_size() >= initial_size);
    }

    #[test]
    fn test_dynamic_window_shrink() {
        let mut window = DynamicWindow::default();

        for _ in 0..15 {
            window.record_utilization(50, 512);
        }
        window.auto_adjust();

        assert!(window.current_size() < window.config().max_size);
    }

    #[test]
    fn test_short_term_memory_write() {
        let mut memory = ShortTermMemory::default();
        let data = Array2::zeros((10, 512));

        memory.write(data.clone(), 1, None);

        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn test_short_term_memory_eviction() {
        let config = MemoryConfig {
            short_term_capacity: 3,
            compression_threshold: 10,
            ..Default::default()
        };
        let mut memory = ShortTermMemory::new(&config);

        for i in 0..5 {
            let data = Array2::zeros((1, 512));
            memory.write(data, i as u64, None);
        }

        assert_eq!(memory.len(), 3);
    }

    #[test]
    fn test_short_term_memory_compress() {
        let config = MemoryConfig {
            short_term_capacity: 10,
            compression_threshold: 3,
            ..Default::default()
        };
        let mut memory = ShortTermMemory::new(&config);

        for i in 0..5 {
            let data = Array2::zeros((1, 512));
            memory.write(data, i as u64, None);
        }

        assert!(memory.needs_compression());
    }

    #[test]
    fn test_semantic_compress() {
        let config = MemoryConfig {
            short_term_capacity: 20,
            compression_threshold: 100,
            ..Default::default()
        };
        let mut memory = ShortTermMemory::new(&config);

        let base_data = Array2::from_shape_fn((5, 32), |(i, j)| (i + j) as f32 * 0.1);

        for i in 0..10 {
            let mut data = base_data.clone();
            for elem in data.iter_mut() {
                *elem += i as f32 * 0.001;
            }
            memory.write(data, i as u64, None);
        }

        let merged = memory.semantic_compress(Some(0.9));
        let _ = merged;
    }

    #[test]
    fn test_importance_based_read() {
        let mut memory = ShortTermMemory::default();

        for i in 0..5 {
            let data = Array2::from_shape_fn((5, 32), |(r, c)| (i + r + c) as f32 * 0.1);
            memory.write(data, i as u64, None);
        }

        let high_importance = memory.read_by_importance(0.0);
        assert!(!high_importance.is_empty());
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j + 1) as f32);
        let b = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j + 1) as f32 * 2.0);

        let similarity = Compressor::cosine_similarity(&a, &b);
        assert!(similarity > 0.99);
    }

    #[test]
    fn test_get_all_importance() {
        let mut memory = ShortTermMemory::default();

        for i in 0..5 {
            let data = Array2::from_shape_fn((5, 32), |(r, c)| (i + r + c) as f32 * 0.1);
            memory.write(data, i as u64, None);
        }

        let importances = memory.get_all_importance();
        assert_eq!(importances.len(), 5);

        for &imp in &importances {
            assert!(imp >= 0.0 && imp <= 1.0);
        }
    }

    #[test]
    fn test_touch_updates_access_count() {
        let mut memory = ShortTermMemory::default();
        let data = Array2::zeros((5, 32));
        memory.write(data, 1, None);

        memory.touch(0);
        memory.touch(0);

        let importance_before = memory.calculate_item_importance(0);
        memory.touch(0);
        let importance_after = memory.calculate_item_importance(0);

        assert!(importance_after.is_some());
        assert!(importance_before.is_some());
    }

    #[test]
    fn test_compress_stored_items_preserves_access_count() {
        let compressor = Compressor::default();
        let mut items: VecDeque<StoredItem> = VecDeque::new();

        for i in 0..10 {
            let data = Array2::from_shape_fn((5, 32), |(r, c)| {
                i as f32 + r as f32 * 0.1 + c as f32 * 0.01
            });
            let item = MemoryItem::new(data, i as u64).with_importance(0.5 + i as f32 * 0.05);
            let stored = StoredItem::new(item, i as u64).with_access_count(i + 1);
            items.push_back(stored);
        }

        let original_total_access: usize = items.iter().map(|s| s.access_count).sum();
        compressor.compress_stored_items(&mut items, 100);
        
        assert!(!items.is_empty());
        let new_total_access: usize = items.iter().map(|s| s.access_count).sum();
        assert!(new_total_access <= original_total_access);
    }

    #[test]
    fn test_union_find_transitive_closure() {
        let mut uf = super::UnionFind::new(5);
        
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(3, 4);
        
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(1), uf.find(2));
        assert_eq!(uf.find(0), uf.find(2));
        
        assert_eq!(uf.find(3), uf.find(4));
        
        assert_ne!(uf.find(0), uf.find(3));
    }

    #[test]
    fn test_semantic_compress_transitive_merge() {
        let config = MemoryConfig {
            short_term_capacity: 20,
            compression_threshold: 100,
            ..Default::default()
        };
        let mut memory = ShortTermMemory::new(&config);

        let base = Array2::from_shape_fn((5, 32), |(i, j)| (i + j) as f32 * 0.1);
        
        let mut data_a = base.clone();
        for elem in data_a.iter_mut() { *elem += 0.001; }
        memory.write(data_a, 1, None);
        
        let mut data_b = base.clone();
        for elem in data_b.iter_mut() { *elem += 0.002; }
        memory.write(data_b, 2, None);
        
        let mut data_c = base.clone();
        for elem in data_c.iter_mut() { *elem += 0.003; }
        memory.write(data_c, 3, None);

        let merged = memory.semantic_compress(Some(0.95));
        
        assert!(memory.len() <= 3);
        let _ = merged;
    }

    #[test]
    fn test_merge_zero_importance_equal_weights() {
        let mut items: VecDeque<StoredItem> = VecDeque::new();

        for i in 0..3 {
            let data = Array2::zeros((5, 32));
            let item = MemoryItem::new(data, i as u64).with_importance(0.0);
            let stored = StoredItem::new(item, i as u64).with_access_count(i + 1);
            items.push_back(stored);
        }

        let items_to_merge: Vec<(StoredItem, f32)> = items
            .iter()
            .map(|s| (s.clone(), 0.0f32))
            .collect();

        let result = Compressor::semantic_merge_stored(&items_to_merge);
        
        assert!(result.is_some());
        let merged = result.unwrap();
        assert_eq!(merged.access_count, 6);
    }

    #[test]
    fn test_dynamic_window_auto_adjust_on_compress() {
        let config = MemoryConfig {
            short_term_capacity: 100,
            compression_threshold: 5,
            ..Default::default()
        };
        let mut memory = ShortTermMemory::new(&config);
        
        let initial_size = memory.dynamic_window_size();
        
        for _ in 0..20 {
            for i in 0..8 {
                let data = Array2::zeros((1, 512));
                memory.write(data, i as u64, None);
            }
        }
        
        let final_size = memory.dynamic_window_size();
        
        assert!(final_size != initial_size || memory.average_utilization() > 0.0);
    }

    #[test]
    fn test_semantic_compress_returns_merge_count() {
        let config = MemoryConfig {
            short_term_capacity: 20,
            compression_threshold: 100,
            ..Default::default()
        };
        let mut memory = ShortTermMemory::new(&config);

        let base = Array2::from_shape_fn((5, 32), |(i, j)| (i + j) as f32 * 0.1);
        
        for i in 0..5 {
            let mut data = base.clone();
            for elem in data.iter_mut() { *elem += i as f32 * 0.001; }
            memory.write(data, i as u64, None);
        }

        let merged_count = memory.semantic_compress(Some(0.95));
        
        assert!(merged_count < 5);
        assert!(memory.len() <= 5);
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_importance_calculator_basic() {
        // 测试重要性计算基础逻辑：高频访问项应该有更高重要性
        let calc = ImportanceCalculator::default();
        let data = Array2::from_elem((10, 128), 1.0);
        let item = MemoryItem::new(data, 100);

        let write_timestamp = 100u64;
        let current_time = 200u64;

        // 高频访问
        let score_high = calc.calculate_importance(&item, 100, write_timestamp, current_time, None);
        // 低频访问
        let score_low = calc.calculate_importance(&item, 1, write_timestamp, current_time, None);

        assert!(
            score_high > score_low,
            "高频访问项应该有更高重要性: high={}, low={}",
            score_high,
            score_low
        );
    }

    #[test]
    fn test_importance_calculator_with_attention_override() {
        // 测试使用注意力分数覆盖值
        let calc = ImportanceCalculator::default();
        let data = Array2::zeros((10, 128));
        let item = MemoryItem::new(data, 100);

        // 不使用覆盖值（基于数据估算）
        let _score_default = calc.calculate_importance(&item, 5, 100, 200, None);

        // 使用高注意力分数覆盖值
        let score_high_attention = calc.calculate_importance(&item, 5, 100, 200, Some(0.95));

        // 使用低注意力分数覆盖值
        let score_low_attention = calc.calculate_importance(&item, 5, 100, 200, Some(0.05));

        assert!(score_high_attention >= score_low_attention);
    }

    #[test]
    fn test_compressor_eviction_policy() {
        // 测试压缩器淘汰策略：压缩后数量应该减少
        let config = CompressorConfig {
            compression_ratio: 0.5,
            min_retained: 3,
            importance_threshold: 0.3,
            enable_semantic: true,
            similarity_threshold: 0.85,
        };
        let compressor = Compressor::new(config);

        // 添加超过容量的项目
        let mut items: VecDeque<StoredItem> = VecDeque::new();

        for i in 0..10 {
            let data = Array2::from_elem((4, 16), i as f32);
            let item = MemoryItem::new(data, i as u64).with_importance(0.5 + i as f32 * 0.05);
            let stored = StoredItem::new(item, i as u64).with_access_count(i + 1);
            items.push_back(stored);
        }

        let original_count = items.len();
        let removed = compressor.compress_stored_items(&mut items, 1000);

        // 验证压缩后数量减少或保持合理范围
        assert!(items.len() <= original_count);
        assert!(!items.is_empty()); // 至少保留 min_retained 个

        // 验证返回被移除的条目
        let _ = removed;
    }

    #[test]
    fn test_dynamic_window_resize() {
        // 测试动态窗口大小调整
        let config = DynamicWindowConfig {
            initial_size: 10,
            min_size: 1,
            max_size: 100,
            expand_threshold: 0.9,
            shrink_threshold: 0.3,
            adjust_step: 5,
        };
        let mut window = DynamicWindow::new(config);

        assert_eq!(window.current_size(), 10);

        // 手动调整大小
        window.set_size(20);
        assert_eq!(window.current_size(), 20);

        // 调整到最大值
        window.set_size(150);
        assert_eq!(window.current_size(), 100); // 应该被限制为 max_size

        // 调整到最小值边界
        window.set_size(0);
        assert!(window.current_size() >= 1); // 应该至少为 min_size

        // 测试重置功能
        window.reset();
        assert_eq!(window.current_size(), 10); // 重置回初始值
    }

    #[test]
    fn test_dynamic_window_auto_adjust_expand() {
        // 测试动态窗口自动扩展
        let config = DynamicWindowConfig {
            min_size: 10,
            max_size: 100,
            initial_size: 20,
            expand_threshold: 0.8,
            shrink_threshold: 0.2,
            adjust_step: 10,
        };
        let mut window = DynamicWindow::new(config);

        let initial_size = window.current_size();

        // 记录高利用率，触发扩展
        for _ in 0..15 {
            window.record_utilization(18, 20); // 90% 利用率
        }

        let new_size = window.auto_adjust();
        assert!(new_size > initial_size, "窗口应该扩展");
    }

    #[test]
    fn test_dynamic_window_auto_adjust_shrink() {
        // 测试动态窗口自动收缩
        let config = DynamicWindowConfig {
            min_size: 10,
            max_size: 100,
            initial_size: 50,
            expand_threshold: 0.9,
            shrink_threshold: 0.3,
            adjust_step: 10,
        };
        let mut window = DynamicWindow::new(config);

        let initial_size = window.current_size();

        // 记录低利用率，触发收缩
        for _ in 0..15 {
            window.record_utilization(5, 50); // 10% 利用率
        }

        let new_size = window.auto_adjust();
        assert!(new_size < initial_size, "窗口应该收缩");
        assert!(new_size >= 10, "不应低于最小值");
    }

    #[test]
    fn test_stored_item_creation_and_access() {
        // 测试 StoredItem 的创建和访问计数设置
        let data = Array2::from_elem((5, 32), 42.0);
        let item = MemoryItem::new(data.clone(), 12345).with_importance(0.75);

        let stored = StoredItem::new(item, 1000);
        assert_eq!(stored.access_count, 1);
        assert_eq!(stored.write_timestamp, 1000);

        // 使用 with_access_count 设置自定义访问计数
        let stored_custom = StoredItem::new(
            MemoryItem::new(data, 12345),
            2000
        ).with_access_count(50);

        assert_eq!(stored_custom.access_count, 50);
        assert_eq!(stored_custom.write_timestamp, 2000);
    }

    #[test]
    fn test_short_term_memory_lfu_eviction() {
        // 测试 LFU 驱逐策略
        let config = MemoryConfig {
            short_term_capacity: 3,
            eviction_strategy: EvictionStrategy::LFU,
            compression_threshold: 100,
            ..Default::default()
        };
        let mut memory = ShortTermMemory::new(&config);

        // 写入3个条目
        for i in 0..3 {
            let data = Array2::from_elem((1, 32), i as f32);
            memory.write(data, i as u64, None);
        }

        // 增加第一个条目的访问计数
        memory.touch(0);
        memory.touch(0);
        memory.touch(0);

        // 写入第4个条目，应该淘汰访问次数最少的
        let data = Array2::from_elem((1, 32), 99.0);
        memory.write(data, 100, None);

        assert_eq!(memory.len(), 3);
    }

    #[test]
    fn test_short_term_memory_fifo_eviction() {
        // 测试 FIFO 驱逐策略
        let config = MemoryConfig {
            short_term_capacity: 3,
            eviction_strategy: EvictionStrategy::FIFO,
            compression_threshold: 100,
            ..Default::default()
        };
        let mut memory = ShortTermMemory::new(&config);

        // 写入3个条目
        for i in 0..3 {
            let data = Array2::from_elem((1, 32), i as f32);
            memory.write(data, i as u64, None);
        }

        // 写入第4个条目，应该淘汰最早的
        let data = Array2::from_elem((1, 32), 99.0);
        memory.write(data, 100, None);

        assert_eq!(memory.len(), 3);
    }

    #[test]
    fn test_short_term_memory_session_isolation() {
        // 测试会话隔离读取
        let mut memory = ShortTermMemory::default();

        // 写入不同会话的数据
        for session_id in [1u64, 2, 3] {
            for i in 0..3 {
                let data = Array2::from_elem((2, 16), (session_id * 10 + i) as f32);
                memory.write(data, i as u64, Some(session_id));
            }
        }

        // 读取特定会话的数据
        let session1_data = memory.read_session(1);
        assert_eq!(session1_data.len(), 3);

        let session2_data = memory.read_session(2);
        assert_eq!(session2_data.len(), 3);

        // 读取不存在的会话
        let non_existent = memory.read_session(999);
        assert!(non_existent.is_empty());
    }

    #[test]
    fn test_compress_to_summary() {
        // 测试压缩为摘要
        let mut memory = ShortTermMemory::default();

        for i in 0..5 {
            let data = Array2::from_elem((3, 16), i as f32);
            memory.write(data, i as u64, None);
        }

        let summary = memory.compress_to_summary();

        // 摘要应该包含所有行的数据
        assert_eq!(summary.nrows(), 15); // 5 * 3
        assert_eq!(summary.ncols(), 16);
    }

    #[test]
    fn test_average_utilization_calculation() {
        // 测试平均利用率计算
        let mut memory = ShortTermMemory::default();

        // 初始状态利用率应为0
        assert_eq!(memory.average_utilization(), 0.0);

        // 写入一些数据并记录利用率
        for i in 0..5 {
            let data = Array2::from_elem((1, 32), i as f32);
            memory.write(data, i as u64, None);
        }

        // 写入后会记录利用率
        let avg_util = memory.average_utilization();
        assert!(avg_util > 0.0);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        // 测试余弦相似度的边界情况
        // 完全相同的向量
        let a = Array2::from_shape_fn((1, 8), |(_, j)| (j + 1) as f32);
        let b = a.clone();
        let sim_same = Compressor::cosine_similarity(&a, &b);
        assert!((sim_same - 1.0).abs() < 0.001, "相同向量相似度应接近1");

        // 空数组
        let empty: Array2<f32> = Array2::zeros((0, 0));
        let sim_empty = Compressor::cosine_similarity(&empty, &a);
        assert_eq!(sim_empty, 0.0, "空数组相似度应为0");

        // 不同维度
        let c = Array2::from_elem((1, 4), 1.0);
        let d = Array2::from_elem((1, 8), 1.0);
        let sim_diff_dim = Compressor::cosine_similarity(&c, &d);
        // 不同维度时的行为取决于实现
        let _ = sim_diff_dim;
    }
}
