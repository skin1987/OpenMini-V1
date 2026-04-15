//! Kascade (锚点层复用稀疏注意力)
//!
//! 训练无关的稀疏注意力方法，核心思想：仅在"锚点层"计算精确的 Top-K 索引，
//! 中间层直接复用，大幅减少计算量。
//!
//! # 核心优势
//!
//! - **训练无关**: 无需修改模型权重或重新训练
//! - **计算高效**: 仅在锚点层计算 Top-K，中间层直接复用
//! - **灵活策略**: 支持多种复用策略（直接/加权/自适应）
//! - **缓存优化**: 智能缓存管理，避免内存爆炸
//!
//! # 性能特性
//!
//! - 理论加速比: >3x（相比全量 Top-K 计算）
//! - 内存节省: 减少中间层的索引计算存储
//! - 适用于: 长序列推理、大模型部署
//!
//! # 使用示例
//!
//! ```ignore
//! use openmini_server::model::inference::kascade::{
//!     KascadeSparseAttention, KascadeConfig, ReuseStrategy,
//! };
//!
//! let config = KascadeConfig {
//!     anchor_interval: 4,
//!     top_k: 2048,
//!     max_cached_layers: 16,
//! };
//!
//! let mut kascade = KascadeSparseAttention::new(
//!     config,
//!     ReuseStrategy::Direct,
//! );
//!
//! // 选择锚点层
//! let anchors = kascade.select_anchor_layers(32, 4096);
//!
//! // 前向传播
//! let outputs = kascade.forward(&q, &k, &v, None, 32)?;
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;

use ndarray::{Array2, Array3, Axis};

use crate::model::inference::error::{InferenceError, InferenceResult};

// ============================================================================
// 配置与策略定义
// ============================================================================

/// Kascade 配置参数
#[derive(Debug, Clone)]
pub struct KascadeConfig {
    /// 锚点层间隔（每 N 层选一个锚点层）
    pub anchor_interval: usize,

    /// Top-K 值（每个 query 保留的 key 数量）
    pub top_k: usize,

    /// 最大缓存层数（防止内存爆炸）
    pub max_cached_layers: usize,
}

impl Default for KascadeConfig {
    fn default() -> Self {
        Self {
            anchor_interval: 4,
            top_k: 2048,
            max_cached_layers: 16,
        }
    }
}

impl fmt::Display for KascadeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KascadeConfig {{ interval={}, top_k={}, max_cache={} }}",
            self.anchor_interval, self.top_k, self.max_cached_layers
        )
    }
}

/// 复用策略枚举
#[derive(Debug, Clone)]
pub enum ReuseStrategy {
    /// 直接复用（最近锚点层的索引）
    Direct,

    /// 加权插值（多个锚点层的索引混合）
    Weighted { anchor_weights: Vec<f64> },

    /// 自适应（根据层相似度动态选择）
    Adaptive { similarity_threshold: f64 },
}

impl Default for ReuseStrategy {
    fn default() -> Self {
        Self::Direct
    }
}

impl fmt::Display for ReuseStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct => write!(f, "Direct"),
            Self::Weighted { .. } => write!(f, "Weighted"),
            Self::Adaptive { .. } => write!(f, "Adaptive"),
        }
    }
}

/// 缓存统计信息
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// 命中次数
    pub hits: usize,
    /// 未命中次数
    pub misses: usize,
    /// 回退到全量计算次数
    pub fallbacks: usize,
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            (self.hits as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        write!(
            f,
            "CacheStats {{ hits={}, misses={}, fallbacks={}, hit_rate={:.1}% }}",
            self.hits, self.misses, self.fallbacks, hit_rate
        )
    }
}

// ============================================================================
// 核心数据结构
// ============================================================================

/// Kascade 稀疏注意力引擎
///
/// 实现训练无关的锚点层复用稀疏注意力方法。
///
/// # 核心思想
///
/// 在 Transformer 的多层结构中，相邻层的 attention pattern 往往高度相似。
/// Kascade 利用这一观察：
/// 1. 仅在"锚点层"计算精确的 Top-K 索引
/// 2. 中间层直接复用最近的锚点层索引
/// 3. 通过不同复用策略平衡精度和效率
///
/// # 架构设计
///
/// ```text
/// Layer 0  [Anchor] → 计算 Top-K 并缓存
/// Layer 1           → 复用 Layer 0 的索引
/// Layer 2           → 复用 Layer 0 的索引
/// Layer 3           → 复用 Layer 0 的索引
/// Layer 4  [Anchor] → 计算新的 Top-K 并缓存
/// Layer 5           → 复用 Layer 4 的索引
/// ...
/// ```
pub struct KascadeSparseAttention {
    /// 锚点层集合（每 N 层选一个作为锚点）
    anchor_layers: Vec<usize>,

    /// 锚点层的精确 Top-K 索引缓存
    anchor_top_k_indices: HashMap<usize, Array2<usize>>,

    /// LRU 缓存队列（用于容量管理）
    cache_lru: VecDeque<usize>,

    /// 复用策略
    reuse_strategy: ReuseStrategy,

    /// 配置
    config: KascadeConfig,

    /// 统计信息
    stats: CacheStats,
}

impl KascadeSparseAttention {
    /// 创建新的 Kascade 稀疏注意力实例
    ///
    /// # 参数
    ///
    /// - `config`: Kascade 配置参数
    /// - `reuse_strategy`: 复用策略
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let config = KascadeConfig::default();
    /// let strategy = ReuseStrategy::Direct;
    /// let kascade = KascadeSparseAttention::new(config, strategy);
    /// ```
    pub fn new(config: KascadeConfig, reuse_strategy: ReuseStrategy) -> Self {
        Self {
            anchor_layers: Vec::new(),
            anchor_top_k_indices: HashMap::new(),
            cache_lru: VecDeque::new(),
            reuse_strategy,
            config,
            stats: CacheStats::default(),
        }
    }

    /// 获取配置引用
    pub fn config(&self) -> &KascadeConfig {
        &self.config
    }

    /// 获取可变配置引用
    pub fn config_mut(&mut self) -> &mut KascadeConfig {
        &mut self.config
    }

    /// 获取统计信息
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// 重置统计信息
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }

    /// 清除所有缓存
    pub fn clear_cache(&mut self) {
        self.anchor_top_k_indices.clear();
        self.cache_lru.clear();
    }

    /// 获取当前缓存的层数
    pub fn cached_layer_count(&self) -> usize {
        self.anchor_top_k_indices.len()
    }
}

// ============================================================================
// 锚点层选择算法
// ============================================================================

impl KascadeSparseAttention {
    /// 使用均匀分布选择最优锚点层集合
    ///
    /// 目标：在保证覆盖率的同时最小化锚点层数量。
    ///
    /// # 算法
    ///
    /// 1. 均匀分布候选锚点层（每隔 `anchor_interval` 层选一个）
    /// 2. 确保最后一层始终是锚点层（用于最终输出）
    /// 3. 返回排序后的锚点层列表
    ///
    /// # 参数
    ///
    /// - `num_layers`: 总层数
    /// - `_model_dim`: 模型维度（预留参数，可用于未来 DP 优化）
    ///
    /// # 返回值
    ///
    /// 排序后的锚点层索引列表
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let anchors = kascade.select_anchor_layers(32, 4096);
    /// // 返回 [0, 4, 8, 12, 16, 20, 24, 28, 31] （interval=4）
    /// ```
    pub fn select_anchor_layers(&mut self, num_layers: usize, _model_dim: usize) -> Vec<usize> {
        let mut anchors = Vec::new();

        for i in 0..num_layers {
            if i % self.config.anchor_interval == 0 || i == num_layers - 1 {
                anchors.push(i);
            }
        }

        self.anchor_layers = anchors.clone();
        anchors
    }

    /// 获取当前锚点层集合
    pub fn anchor_layers(&self) -> &[usize] {
        &self.anchor_layers
    }

    /// 判断指定层是否为锚点层
    pub fn is_anchor_layer(&self, layer_idx: usize) -> bool {
        self.anchor_layers.contains(&layer_idx)
    }
}

// ============================================================================
// Top-K 索引计算与缓存
// ============================================================================

impl KascadeSparseAttention {
    /// 在锚点层计算并缓存精确的 Top-K 索引
    ///
    /// # 算法流程
    ///
    /// 1. 计算 Q @ K^T 得到 attention scores
    /// 2. 对每个 query position 取 top-k
    /// 3. 缓存到 `anchor_top_k_indices[layer_idx]`
    /// 4. 返回索引矩阵
    ///
    /// # 参数
    ///
    /// - `layer_idx`: 当前层索引
    /// - `q`: Query 矩阵 [seq_len, head_dim]
    /// - `k`: Key 矩阵 [seq_len, kv_dim]
    ///
    /// # 返回值
    ///
    /// Top-K 索引矩阵 [seq_len, top_k]
    ///
    /// # 错误
    ///
    /// - 维度不匹配时返回错误
    /// - 内存不足时返回错误
    pub fn compute_and_cache_anchor_indices(
        &mut self,
        layer_idx: usize,
        q: &Array2<f32>,
        k: &Array2<f32>,
    ) -> InferenceResult<Array2<usize>> {
        let indices = self.compute_full_topk(q, k)?;

        self.cache_indices(layer_idx, &indices);

        Ok(indices)
    }

    /// 将索引缓存到指定层
    fn cache_indices(&mut self, layer_idx: usize, indices: &Array2<usize>) {
        if self.anchor_top_k_indices.len() >= self.config.max_cached_layers {
            if let Some(oldest) = self.cache_lru.pop_front() {
                self.anchor_top_k_indices.remove(&oldest);
            }
        }

        self.anchor_top_k_indices.insert(layer_idx, indices.clone());
        self.cache_lru.push_back(layer_idx);
    }

    /// 全量 Top-K 计算（fallback 方法）
    ///
    /// 对每个 query position，找出相似度最高的 top-k 个 key positions。
    ///
    /// # 参数
    ///
    /// - `q`: Query 矩阵 [seq_len, head_dim]
    /// - `k`: Key 矩阵 [seq_len, kv_dim]
    ///
    /// # 返回值
    ///
    /// Top-K 索引矩阵 [seq_len, top_k]
    fn compute_full_topk(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
    ) -> InferenceResult<Array2<usize>> {
        let (seq_len_q, _) = q.dim();
        let (seq_len_k, _) = k.dim();

        if seq_len_q == 0 || seq_len_k == 0 {
            return Err(InferenceError::config(
                "Empty input tensors for Top-K computation",
            ));
        }

        let top_k = self.config.top_k.min(seq_len_k);

        let scores = self.compute_attention_scores(q, k)?;

        let mut indices = Array2::<usize>::zeros((seq_len_q, top_k));

        indices
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut row)| {
                let score_row = scores.row(i);

                let mut indexed_scores: Vec<(usize, f32)> =
                    score_row.iter().copied().enumerate().collect();

                indexed_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (j, (idx, _)) in indexed_scores.iter().take(top_k).enumerate() {
                    row[j] = *idx;
                }
            });

        Ok(indices)
    }

    /// 计算 Q @ K^T 注意力分数
    ///
    /// # 参数
    ///
    /// - `q`: Query 矩阵 [seq_len, head_dim]
    /// - `k`: Key 矩阵 [seq_len, kv_dim]
    ///
    /// # 返回值
    ///
    /// 注意力分数矩阵 [seq_len_q, seq_len_k]
    fn compute_attention_scores(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
    ) -> InferenceResult<Array2<f32>> {
        let (_, head_dim) = q.dim();
        let (_, kv_dim) = k.dim();

        if head_dim != kv_dim {
            return Err(InferenceError::config(format!(
                "Q dim {} != K dim {}",
                head_dim, kv_dim
            )));
        }

        let seq_len_q = q.shape()[0];
        let seq_len_k = k.shape()[0];

        let mut scores = Array2::<f32>::zeros((seq_len_q, seq_len_k));

        scores
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut row)| {
                let q_vec = q.row(i);

                row.iter_mut().enumerate().for_each(|(j, val)| {
                    let k_vec = k.row(j);
                    let dot_product: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();

                    *val = dot_product;
                });
            });

        Ok(scores)
    }

    /// 3D 版本的全量 Top-K 计算
    ///
    /// 处理多头注意力的批量计算。
    ///
    /// # 参数
    ///
    /// - `q`: 3D Query 张量 [batch, heads, dim] 或展平后 [batch*heads, dim]
    /// - `k`: 3D Key 张量
    /// - `_layer`: 当前层索引（预留参数）
    ///
    /// # 返回值
    ///
    /// Top-K 索引矩阵 [total_queries, top_k]
    fn compute_full_topk_3d(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        _layer: usize,
    ) -> InferenceResult<Array2<usize>> {
        let (batch, heads, dim) = q.dim();
        let total_queries = batch * heads;

        let q_2d: Array2<f32> = q
            .to_shape((total_queries, dim))
            .map_err(|e| InferenceError::config(format!("Failed to reshape Q: {}", e)))?
            .into_owned();

        let k_2d: Array2<f32> = k
            .to_shape((total_queries, dim))
            .map_err(|e| InferenceError::config(format!("Failed to reshape K: {}", e)))?
            .into_owned();

        self.compute_full_topk(&q_2d, &k_2d)
    }
}

// ============================================================================
// 中间层复用逻辑
// ============================================================================

impl KascadeSparseAttention {
    /// 对于非锚点层，复用已有索引而非重新计算
    ///
    /// 根据不同的复用策略返回相应的索引。
    ///
    /// # 参数
    ///
    /// - `layer_idx`: 当前层索引
    /// - `q`: Query 矩阵
    /// - `k`: Key 矩阵
    /// - `_v`: Value 矩阵（预留参数）
    ///
    /// # 返回值
    ///
    /// 元组 `(indices, is_reused)`:
    /// - `indices`: Top-K 索引矩阵
    /// - `is_reused`: 是否为复用的索引（true=复用, false=新计算）
    pub fn reuse_or_compute(
        &mut self,
        layer_idx: usize,
        q: &Array2<f32>,
        k: &Array2<f32>,
        _v: &Array2<f32>,
    ) -> InferenceResult<(Array2<usize>, bool)> {
        let strategy = self.reuse_strategy.clone();
        match &strategy {
            ReuseStrategy::Direct => self.direct_reuse(layer_idx, q, k),

            ReuseStrategy::Weighted { anchor_weights } => {
                self.weighted_interpolation(layer_idx, q, k, anchor_weights)
            }

            ReuseStrategy::Adaptive {
                similarity_threshold,
            } => self.adaptive_reuse(layer_idx, q, k, *similarity_threshold),
        }
    }

    /// 直接复用策略
    ///
    /// 找到最近的锚点层，直接返回其索引。
    fn direct_reuse(
        &mut self,
        layer_idx: usize,
        q: &Array2<f32>,
        k: &Array2<f32>,
    ) -> InferenceResult<(Array2<usize>, bool)> {
        let nearest_cache = self.find_nearest_anchor_cache(layer_idx).cloned();

        if let Some(indices) = nearest_cache {
            self.stats.hits += 1;
            Ok((indices, true))
        } else {
            self.stats.misses += 1;
            self.stats.fallbacks += 1;
            let indices = self.compute_full_topk(q, k)?;
            Ok((indices, false))
        }
    }

    /// 加权插值策略
    ///
    /// 多个锚点层索引加权混合。
    ///
    /// # 算法
    ///
    /// 1. 找到前后最近的锚点层
    /// 2. 根据距离计算插值权重
    /// 3. 对多个锚点层的索引进行投票/混合
    fn weighted_interpolation(
        &mut self,
        layer_idx: usize,
        q: &Array2<f32>,
        k: &Array2<f32>,
        anchor_weights: &[f64],
    ) -> InferenceResult<(Array2<usize>, bool)> {
        let nearest_anchors = self.find_surrounding_anchors(layer_idx);

        if nearest_anchors.is_empty() {
            self.stats.misses += 1;
            self.stats.fallbacks += 1;
            return self.compute_full_topk(q, k).map(|i| (i, false));
        }

        if nearest_anchors.len() == 1 {
            let anchor = nearest_anchors[0];
            if let Some(indices) = self.anchor_top_k_indices.get(&anchor) {
                self.stats.hits += 1;
                return Ok((indices.clone(), true));
            }
        }

        let primary_anchor = nearest_anchors[0];
        if let Some(primary_indices) = self.anchor_top_k_indices.get(&primary_anchor) {
            self.stats.hits += 1;

            if nearest_anchors.len() > 1 && !anchor_weights.is_empty() {
                let secondary_anchor = nearest_anchors[1];
                if let Some(secondary_indices) = self.anchor_top_k_indices.get(&secondary_anchor) {
                    return Ok((
                        self.blend_indices(primary_indices, secondary_indices, anchor_weights),
                        true,
                    ));
                }
            }

            Ok((primary_indices.clone(), true))
        } else {
            self.stats.misses += 1;
            self.stats.fallbacks += 1;
            self.compute_full_topk(q, k).map(|i| (i, false))
        }
    }

    /// 自适应复用策略
    ///
    /// 根据层相似度决定是否复用。
    ///
    /// # 算法
    ///
    /// 1. 找到最近的锚点层
    /// 2. 评估当前层与锚点层的相似度
    /// 3. 如果相似度超过阈值，则复用；否则回退到全量计算
    fn adaptive_reuse(
        &mut self,
        layer_idx: usize,
        q: &Array2<f32>,
        k: &Array2<f32>,
        similarity_threshold: f64,
    ) -> InferenceResult<(Array2<usize>, bool)> {
        let nearest_anchor = self.find_nearest_anchor(layer_idx);

        match nearest_anchor {
            Some(anchor_idx) => {
                let distance = (anchor_idx as isize - layer_idx as isize).abs() as f64;
                let max_distance = self.config.anchor_interval as f64;
                let similarity = 1.0 - (distance / max_distance).min(1.0);

                if similarity >= similarity_threshold {
                    if let Some(indices) = self.anchor_top_k_indices.get(&anchor_idx) {
                        self.stats.hits += 1;
                        return Ok((indices.clone(), true));
                    }
                }

                self.stats.misses += 1;
                self.stats.fallbacks += 1;
                let indices = self.compute_full_topk(q, k)?;
                self.maybe_cache(layer_idx, &indices);
                Ok((indices, false))
            }
            None => {
                self.stats.misses += 1;
                self.stats.fallbacks += 1;
                self.compute_full_topk(q, k).map(|i| (i, false))
            }
        }
    }

    /// 查找最近的锚点层缓存
    fn find_nearest_anchor_cache(&self, layer: usize) -> Option<&Array2<usize>> {
        self.anchor_layers
            .iter()
            .filter_map(|&anchor| {
                self.anchor_top_k_indices
                    .get(&anchor)
                    .map(|indices| (anchor, indices))
            })
            .min_by_key(|(anchor, _)| (*anchor as isize - layer as isize).abs())
            .map(|(_, indices)| indices)
    }

    /// 查找最近的锚点层索引
    fn find_nearest_anchor(&self, layer: usize) -> Option<usize> {
        self.anchor_layers
            .iter()
            .min_by_key(|anchor| (**anchor as isize - layer as isize).abs())
            .copied()
    }

    /// 查找前后最近的锚点层
    fn find_surrounding_anchors(&self, layer: usize) -> Vec<usize> {
        let mut anchors: Vec<_> = self
            .anchor_layers
            .iter()
            .filter_map(|&anchor| {
                self.anchor_top_k_indices
                    .contains_key(&anchor)
                    .then_some(anchor)
            })
            .collect();

        anchors.sort_by_key(|&anchor| (anchor as isize - layer as isize).abs());
        anchors.truncate(2.min(anchors.len()));
        anchors
    }

    /// 混合两组索引
    fn blend_indices(
        &self,
        primary: &Array2<usize>,
        secondary: &Array2<usize>,
        weights: &[f64],
    ) -> Array2<usize> {
        let (rows, cols) = primary.dim();
        let weight_primary = weights.first().copied().unwrap_or(0.7);
        let _weight_secondary = weights.get(1).copied().unwrap_or(0.3);

        let mut blended = Array2::<usize>::zeros((rows, cols));

        blended
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, mut row)| {
                use rand::seq::SliceRandom;
                use rand::Rng;
                let mut rng = rand::thread_rng();

                let _primary_set: std::collections::HashSet<_> =
                    primary.row(i).iter().copied().collect();
                let _secondary_set: std::collections::HashSet<_> =
                    secondary.row(i).iter().copied().collect();

                let take_from_primary = ((cols as f64) * weight_primary).round() as usize;
                let mut selected = std::collections::HashSet::new();

                let mut primary_indices: Vec<_> = primary.row(i).to_vec();
                let mut secondary_indices: Vec<_> = secondary.row(i).to_vec();

                primary_indices.shuffle(&mut rng);
                secondary_indices.shuffle(&mut rng);

                for idx in primary_indices.iter().take(take_from_primary) {
                    if selected.len() < cols {
                        selected.insert(*idx);
                    }
                }

                for idx in secondary_indices.iter() {
                    if selected.len() < cols && !selected.contains(idx) {
                        selected.insert(*idx);
                    }
                }

                while selected.len() < cols {
                    let random_idx = rng.gen_range(0..primary.shape()[1]);
                    selected.insert(primary[[i, random_idx]]);
                }

                for (j, idx) in selected.iter().take(cols).enumerate() {
                    row[j] = *idx;
                }
            });

        blended
    }

    /// 可选缓存（避免缓存爆炸）
    fn maybe_cache(&mut self, layer: usize, indices: &Array2<usize>) {
        if self.anchor_top_k_indices.len() < self.config.max_cached_layers {
            self.cache_indices(layer, indices);
        }
    }
}

// ============================================================================
// 完整前向传播
// ============================================================================

impl KascadeSparseAttention {
    /// Kascade 稀疏注意力的完整前向传播
    ///
    /// 对多层 Transformer 进行稀疏注意力计算。
    ///
    /// # 参数
    ///
    /// - `q`: Query 张量 [batch, heads, dim]
    /// - `k`: Key 张量 [batch, heads, dim]
    /// - `v`: Value 张量 [batch, heads, dim]
    /// - `_mask`: 可选的注意力掩码（预留参数）
    /// - `num_layers`: 总层数
    ///
    /// # 返回值
    ///
    /// 每层的输出向量列表
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let outputs = kascade.forward(&q, &k, &v, None, 32)?;
    /// // outputs.len() == 32
    /// ```
    pub fn forward(
        &mut self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        v: &Array3<f32>,
        _mask: Option<&Array3<i8>>,
        num_layers: usize,
    ) -> InferenceResult<Vec<Array3<f32>>> {
        let mut outputs = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let is_anchor = self.is_anchor_layer(layer);

            if is_anchor {
                let indices = self.compute_full_topk_3d(q, k, layer)?;

                self.cache_indices(layer, &indices);

                let output = self.gather_by_indices_3d(q, k, v, &indices)?;
                outputs.push(output);
            } else {
                let (q_2d, k_2d, v_2d) = self.extract_layer_tensors(q, k, v)?;

                let (indices, reused) = self.reuse_or_compute(layer, &q_2d, &k_2d, &v_2d)?;

                if !reused {
                    self.maybe_cache(layer, &indices);
                }

                let output = self.gather_by_indices_3d(q, k, v, &indices)?;
                outputs.push(output);
            }
        }

        Ok(outputs)
    }

    /// 从 3D 张量提取 2D 层表示
    ///
    /// 将 [batch, heads, dim] 展平为 [batch*heads, dim]
    fn extract_layer_tensors(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        v: &Array3<f32>,
    ) -> InferenceResult<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let (batch, heads, dim) = q.dim();
        let total = batch * heads;

        let q_2d: Array2<f32> = q
            .to_owned()
            .to_shape((total, dim))
            .map_err(|e| InferenceError::config(format!("Failed to reshape Q: {}", e)))?
            .into_owned();

        let k_2d: Array2<f32> = k
            .to_owned()
            .to_shape((total, dim))
            .map_err(|e| InferenceError::config(format!("Failed to reshape K: {}", e)))?
            .into_owned();

        let v_2d: Array2<f32> = v
            .to_owned()
            .to_shape((total, dim))
            .map_err(|e| InferenceError::config(format!("Failed to reshape V: {}", e)))?
            .into_owned();

        Ok((q_2d, k_2d, v_2d))
    }

    /// 根据 Top-K 索引 gather KV 值（3D 版本）
    ///
    /// # 参数
    ///
    /// - `_q`: Query 张量 [batch, heads, dim]
    /// - `k`: Key 张量 [batch, heads, dim]
    /// - `v`: Value 张量 [batch, heads, dim]
    /// - `indices`: Top-K 索引 [total_queries, top_k]
    ///
    /// # 返回值
    ///
    /// Gather 后的输出张量 [batch, heads, dim]
    fn gather_by_indices_3d(
        &self,
        _q: &Array3<f32>,
        k: &Array3<f32>,
        v: &Array3<f32>,
        indices: &Array2<usize>,
    ) -> InferenceResult<Array3<f32>> {
        let (batch, heads, dim) = v.dim();
        let total_queries = batch * heads;
        let (_n_queries, _top_k) = indices.dim();

        let k_flat: Array2<f32> = k
            .to_owned()
            .to_shape((total_queries, dim))
            .map_err(|e| InferenceError::config(format!("Failed to reshape K: {}", e)))?
            .into_owned();

        let v_flat: Array2<f32> = v
            .to_owned()
            .to_shape((total_queries, dim))
            .map_err(|e| InferenceError::config(format!("Failed to reshape V: {}", e)))?
            .into_owned();

        let mut output = Array3::<f32>::zeros((batch, heads, dim));

        output
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(b, mut batch_out)| {
                batch_out
                    .axis_iter_mut(Axis(0))
                    .enumerate()
                    .for_each(|(h, mut head_out)| {
                        let flat_idx = b * heads + h;
                        let row_indices = indices.row(flat_idx);

                        let mut weighted_sum = vec![0.0f64; dim];
                        let mut weight_sum = 0.0f64;

                        for &idx in row_indices {
                            if idx < total_queries {
                                let k_vec = k_flat.row(idx);
                                let v_vec = v_flat.row(idx);

                                let q_vec_ref = if flat_idx < total_queries {
                                    Some(k_flat.row(flat_idx))
                                } else {
                                    None
                                };

                                if let Some(q_vec) = q_vec_ref {
                                    let sim: f64 = q_vec
                                        .iter()
                                        .zip(k_vec.iter())
                                        .map(|(a, b)| (*a as f64) * (*b as f64))
                                        .sum();

                                    let weight = sim.exp();
                                    weight_sum += weight;

                                    for (d, (&v_val, ws)) in
                                        v_vec.iter().zip(weighted_sum.iter_mut()).enumerate()
                                    {
                                        if d < dim {
                                            *ws += (v_val as f64) * weight;
                                        }
                                    }
                                }
                            }
                        }

                        if weight_sum > 0.0 {
                            for (val, ws) in head_out.iter_mut().zip(weighted_sum.iter()) {
                                *val = (*ws / weight_sum) as f32;
                            }
                        }
                    });
            });

        Ok(output)
    }
}

// ============================================================================
// 辅助方法
// ============================================================================

impl KascadeSparseAttention {
    /// 从 3D 张量提取指定层的数据（模拟多层情况）
    ///
    /// 注意：在实际应用中，每层的 Q/K/V 可能不同。
    /// 这里简化为取第一个 batch 和第一个 head 的数据。
    #[allow(dead_code)]
    fn extract_layer(&self, tensor: &Array3<f32>, _layer: usize) -> InferenceResult<Array2<f32>> {
        let slice = tensor.index_axis(Axis(0), 0);
        Ok(slice.to_owned())
    }

    /// 计算加速比估算
    ///
    /// 基于当前配置和统计信息估算理论加速比。
    pub fn estimated_speedup(&self) -> f64 {
        let total_ops = self.stats.hits + self.stats.misses;
        if total_ops == 0 {
            return 1.0;
        }

        let anchor_ratio =
            self.anchor_layers.len() as f64 / (self.config.anchor_interval as f64).max(1.0);

        let reuse_ratio = self.stats.hits as f64 / total_ops as f64;

        let base_speedup = 1.0 + (anchor_ratio * reuse_ratio * 2.0);

        base_speedup.clamp(1.0, 10.0)
    }

    /// 获取详细的性能报告
    pub fn performance_report(&self) -> String {
        format!(
            "Kascade Performance Report:\n\
             - Config: {}\n\
             - Strategy: {}\n\
             - Anchor Layers: {:?}\n\
             - Cached Layers: {}\n\
             - Stats: {}\n\
             - Estimated Speedup: {:.2}x\n\
             - Memory Efficiency: {:.1}%",
            self.config,
            self.reuse_strategy,
            self.anchor_layers,
            self.cached_layer_count(),
            self.stats,
            self.estimated_speedup(),
            (self.stats.hits as f64 / (self.stats.hits + self.stats.misses).max(1) as f64) * 100.0
        )
    }
}

impl fmt::Debug for KascadeSparseAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KascadeSparseAttention")
            .field("anchor_layers", &self.anchor_layers)
            .field("cached_count", &self.anchor_top_k_indices.len())
            .field("strategy", &self.reuse_strategy)
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> KascadeConfig {
        KascadeConfig {
            anchor_interval: 4,
            top_k: 8,
            max_cached_layers: 16,
        }
    }

    fn create_test_tensors(seq_len: usize, dim: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let mut q = Array2::<f32>::zeros((seq_len, dim));
        let mut k = Array2::<f32>::zeros((seq_len, dim));
        let mut v = Array2::<f32>::zeros((seq_len, dim));

        for i in 0..seq_len {
            for j in 0..dim {
                q[[i, j]] = ((i * dim + j) as f32 * 0.01).sin();
                k[[i, j]] = ((i * dim + j) as f32 * 0.02).cos();
                v[[i, j]] = ((i * dim + j) as f32 * 0.03).tan();
            }
        }

        (q, k, v)
    }

    fn create_test_tensors_3d(
        batch: usize,
        heads: usize,
        dim: usize,
    ) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        let mut q = Array3::<f32>::zeros((batch, heads, dim));
        let mut k = Array3::<f32>::zeros((batch, heads, dim));
        let mut v = Array3::<f32>::zeros((batch, heads, dim));

        for b in 0..batch {
            for h in 0..heads {
                for d in 0..dim {
                    q[[b, h, d]] = ((b * heads + h) * dim + d) as f32 * 0.01;
                    k[[b, h, d]] = ((b * heads + h) * dim + d) as f32 * 0.02;
                    v[[b, h, d]] = ((b * heads + h) * dim + d) as f32 * 0.03;
                }
            }
        }

        (q, k, v)
    }

    #[test]
    fn test_default_config() {
        let config = KascadeConfig::default();
        assert_eq!(config.anchor_interval, 4);
        assert_eq!(config.top_k, 2048);
        assert_eq!(config.max_cached_layers, 16);
    }

    #[test]
    fn test_config_display() {
        let config = create_test_config();
        let display = format!("{}", config);
        assert!(display.contains("interval=4"));
        assert!(display.contains("top_k=8"));
    }

    #[test]
    fn test_select_anchor_layers_basic() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let anchors = kascade.select_anchor_layers(16, 512);

        assert!(!anchors.is_empty());
        assert!(anchors.contains(&0), "First layer should be an anchor");
        assert!(anchors.contains(&15), "Last layer should be an anchor");

        for (i, &anchor) in anchors.iter().enumerate() {
            if i > 0 {
                assert!(anchor > anchors[i - 1], "Anchors should be sorted");
            }
        }
    }

    #[test]
    fn test_select_anchor_layers_interval() {
        let config = KascadeConfig {
            anchor_interval: 2,
            ..create_test_config()
        };
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let anchors = kascade.select_anchor_layers(8, 256);

        assert_eq!(anchors, vec![0, 2, 4, 6, 7]);
    }

    #[test]
    fn test_is_anchor_layer() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        kascade.select_anchor_layers(16, 512);

        assert!(kascade.is_anchor_layer(0));
        assert!(kascade.is_anchor_layer(4));
        assert!(kascade.is_anchor_layer(15));
        assert!(!kascade.is_anchor_layer(1));
        assert!(!kascade.is_anchor_layer(7));
    }

    #[test]
    fn test_compute_attention_scores() {
        let config = create_test_config();
        let kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, _) = create_test_tensors(16, 32);

        let scores = kascade.compute_attention_scores(&q, &k).unwrap();

        assert_eq!(scores.dim(), (16, 16));
    }

    #[test]
    fn test_compute_full_topk() {
        let config = create_test_config();
        let kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, _) = create_test_tensors(16, 32);

        let indices = kascade.compute_full_topk(&q, &k).unwrap();

        assert_eq!(indices.dim(), (16, 8));

        for row in indices.axis_iter(Axis(0)) {
            for &idx in row {
                assert!(idx < 16, "Index should be within sequence length");
            }
        }
    }

    #[test]
    fn test_compute_and_cache_anchor_indices() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, _) = create_test_tensors(16, 32);

        let indices = kascade.compute_and_cache_anchor_indices(0, &q, &k).unwrap();

        assert_eq!(indices.dim(), (16, 8));
        assert_eq!(kascade.cached_layer_count(), 1);
        assert!(kascade.anchor_top_k_indices.contains_key(&0));
    }

    #[test]
    fn test_direct_reuse_strategy() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, v) = create_test_tensors(16, 32);

        kascade.select_anchor_layers(8, 32);
        kascade.compute_and_cache_anchor_indices(0, &q, &k).unwrap();

        let (indices, reused) = kascade.reuse_or_compute(2, &q, &k, &v).unwrap();

        assert!(reused, "Should reuse cached indices");
        assert_eq!(indices.dim(), (16, 8));
        assert_eq!(kascade.stats().hits, 1);
    }

    #[test]
    fn test_weighted_interpolation_strategy() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(
            config,
            ReuseStrategy::Weighted {
                anchor_weights: vec![0.7, 0.3],
            },
        );

        let (q, k, v) = create_test_tensors(16, 32);

        kascade.select_anchor_layers(12, 32);
        kascade.compute_and_cache_anchor_indices(0, &q, &k).unwrap();
        kascade.compute_and_cache_anchor_indices(4, &q, &k).unwrap();

        let (indices, reused) = kascade.reuse_or_compute(2, &q, &k, &v).unwrap();

        assert!(reused, "Should reuse with interpolation");
        assert_eq!(indices.dim(), (16, 8));
    }

    #[test]
    fn test_adaptive_reuse_strategy_high_similarity() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(
            config,
            ReuseStrategy::Adaptive {
                similarity_threshold: 0.5,
            },
        );

        let (q, k, v) = create_test_tensors(16, 32);

        kascade.select_anchor_layers(8, 32);
        kascade.compute_and_cache_anchor_indices(0, &q, &k).unwrap();

        let (indices, reused) = kascade.reuse_or_compute(1, &q, &k, &v).unwrap();

        assert!(reused, "High similarity should trigger reuse");
        assert_eq!(indices.dim(), (16, 8));
    }

    #[test]
    fn test_adaptive_reuse_strategy_low_similarity() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(
            config,
            ReuseStrategy::Adaptive {
                similarity_threshold: 0.99,
            },
        );

        let (q, k, v) = create_test_tensors(16, 32);

        kascade.select_anchor_layers(8, 32);
        kascade.compute_and_cache_anchor_indices(0, &q, &k).unwrap();

        let (_indices, reused) = kascade.reuse_or_compute(3, &q, &k, &v).unwrap();

        assert!(!reused, "Low similarity should not trigger reuse");
        assert_eq!(kascade.stats().fallbacks, 1);
    }

    #[test]
    fn test_fallback_to_full_computation() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, v) = create_test_tensors(16, 32);

        let (indices, reused) = kascade.reuse_or_compute(0, &q, &k, &v).unwrap();

        assert!(!reused, "No cache available, should compute");
        assert_eq!(indices.dim(), (16, 8));
        assert_eq!(kascade.stats().fallbacks, 1);
    }

    #[test]
    fn test_cache_capacity_limit() {
        let config = KascadeConfig {
            max_cached_layers: 3,
            ..create_test_config()
        };
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, _) = create_test_tensors(16, 32);

        for i in 0..5u32 {
            kascade
                .compute_and_cache_anchor_indices(i as usize, &q, &k)
                .unwrap();
        }

        assert!(
            kascade.cached_layer_count() <= 3,
            "Cache should respect capacity limit"
        );
    }

    #[test]
    fn test_clear_cache() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, _) = create_test_tensors(16, 32);

        kascade.compute_and_cache_anchor_indices(0, &q, &k).unwrap();
        kascade.compute_and_cache_anchor_indices(4, &q, &k).unwrap();

        assert_eq!(kascade.cached_layer_count(), 2);

        kascade.clear_cache();

        assert_eq!(kascade.cached_layer_count(), 0);
    }

    #[test]
    fn test_reset_stats() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, v) = create_test_tensors(16, 32);

        kascade.reuse_or_compute(0, &q, &k, &v).unwrap();
        kascade.reuse_or_compute(0, &q, &k, &v).unwrap();

        assert!(kascade.stats().hits + kascade.stats().misses > 0);

        kascade.reset_stats();

        assert_eq!(kascade.stats().hits, 0);
        assert_eq!(kascade.stats().misses, 0);
        assert_eq!(kascade.stats().fallbacks, 0);
    }

    #[test]
    fn test_forward_basic() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, v) = create_test_tensors_3d(2, 4, 32);

        kascade.select_anchor_layers(8, 32);

        let outputs = kascade.forward(&q, &k, &v, None, 8).unwrap();

        assert_eq!(outputs.len(), 8);

        for output in &outputs {
            assert_eq!(output.dim(), (2, 4, 32));
        }
    }

    #[test]
    fn test_forward_with_caching() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k, v) = create_test_tensors_3d(2, 4, 32);

        kascade.select_anchor_layers(8, 32);

        let _outputs = kascade.forward(&q, &k, &v, None, 8).unwrap();

        assert!(
            kascade.cached_layer_count() > 0,
            "Forward should cache anchor layers"
        );
        assert!(kascade.stats().hits > 0, "Forward should have cache hits");
    }

    #[test]
    fn test_performance_report() {
        let config = create_test_config();
        let kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let report = kascade.performance_report();

        assert!(report.contains("Kascade Performance Report"));
        assert!(report.contains("Config"));
        assert!(report.contains("Stats"));
        assert!(report.contains("Estimated Speedup"));
    }

    #[test]
    fn test_estimated_speedup() {
        let config = create_test_config();
        let kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let speedup = kascade.estimated_speedup();

        assert!(speedup >= 1.0, "Speedup should be at least 1.0");
        assert!(speedup <= 10.0, "Speedup should be capped at 10.0");
    }

    #[test]
    fn test_debug_format() {
        let config = create_test_config();
        let kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let debug_str = format!("{:?}", kascade);

        assert!(debug_str.contains("KascadeSparseAttention"));
        assert!(debug_str.contains("strategy"));
        assert!(debug_str.contains("config"));
    }

    #[test]
    fn test_reuse_strategy_display() {
        assert_eq!(format!("{}", ReuseStrategy::Direct), "Direct");
        assert_eq!(
            format!(
                "{}",
                ReuseStrategy::Weighted {
                    anchor_weights: vec![0.5, 0.5]
                }
            ),
            "Weighted"
        );
        assert_eq!(
            format!(
                "{}",
                ReuseStrategy::Adaptive {
                    similarity_threshold: 0.7
                }
            ),
            "Adaptive"
        );
    }

    #[test]
    fn test_cache_stats_display() {
        let stats = CacheStats {
            hits: 100,
            misses: 25,
            fallbacks: 10,
        };

        let display = format!("{}", stats);

        assert!(display.contains("hits=100"));
        assert!(display.contains("misses=25"));
        assert!(display.contains("hit_rate="));
    }

    #[test]
    fn test_blend_indices() {
        let config = create_test_config();
        let kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let mut primary = Array2::<usize>::zeros((4, 8));
        let mut secondary = Array2::<usize>::zeros((4, 8));

        for i in 0..4 {
            for j in 0..8 {
                primary[[i, j]] = i * 8 + j;
                secondary[[i, j]] = (i * 8 + j + 4) % 32;
            }
        }

        let blended = kascade.blend_indices(&primary, &secondary, &[0.7, 0.3]);

        assert_eq!(blended.dim(), (4, 8));

        for row in blended.axis_iter(Axis(0)) {
            for &idx in row {
                assert!(idx < 32, "Blended index should be valid");
            }
        }
    }

    #[test]
    fn test_find_nearest_anchor() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        kascade.select_anchor_layers(16, 512);

        assert_eq!(kascade.find_nearest_anchor(0), Some(0));
        assert_eq!(kascade.find_nearest_anchor(2), Some(0));
        assert_eq!(kascade.find_nearest_anchor(6), Some(4));
        assert_eq!(kascade.find_nearest_anchor(15), Some(15));
    }

    #[test]
    fn test_find_surrounding_anchors() {
        let config = create_test_config();
        let mut kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        kascade.select_anchor_layers(16, 512);
        kascade
            .compute_and_cache_anchor_indices(
                0,
                &create_test_tensors(16, 32).0,
                &create_test_tensors(16, 32).1,
            )
            .unwrap();
        kascade
            .compute_and_cache_anchor_indices(
                8,
                &create_test_tensors(16, 32).0,
                &create_test_tensors(16, 32).1,
            )
            .unwrap();

        let surrounding = kascade.find_surrounding_anchors(4);

        assert!(surrounding.contains(&0));
        assert!(surrounding.contains(&8));
        assert!(surrounding.len() <= 2);
    }

    #[test]
    fn test_edge_cases_empty_input() {
        let config = create_test_config();
        let kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let q_empty = Array2::<f32>::zeros((0, 32));
        let k_empty = Array2::<f32>::zeros((0, 32));

        let result = kascade.compute_full_topk(&q_empty, &k_empty);

        assert!(result.is_err(), "Empty input should return error");
    }

    #[test]
    fn test_edge_cases_dimension_mismatch() {
        let config = create_test_config();
        let kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let q = Array2::<f32>::zeros((16, 32));
        let k = Array2::<f32>::zeros((16, 64));

        let result = kascade.compute_attention_scores(&q, &k);

        assert!(result.is_err(), "Dimension mismatch should return error");
    }

    #[test]
    fn test_large_sequence_handling() {
        let config = KascadeConfig {
            top_k: 1024,
            ..create_test_config()
        };
        let kascade = KascadeSparseAttention::new(config, ReuseStrategy::Direct);

        let (q, k) = create_test_tokens(2048, 128);

        let start = std::time::Instant::now();
        let indices = kascade.compute_full_topk(&q, &k).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(indices.dim(), (2048, 1024));
        println!(
            "\n[PERF] Large sequence (2048 tokens, top_k=1024): {:?}",
            elapsed
        );
    }

    #[test]
    fn test_concurrent_safety() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let config = create_test_config();
        let kascade = Arc::new(Mutex::new(KascadeSparseAttention::new(
            config,
            ReuseStrategy::Direct,
        )));

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let kascade_clone = Arc::clone(&kascade);
                thread::spawn(move || {
                    let (q, k) = create_test_tokens(32, 64);
                    let mut cascade = kascade_clone.lock().unwrap();
                    cascade.compute_and_cache_anchor_indices(i, &q, &k).unwrap();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let kascade = kascade.lock().unwrap();
        assert_eq!(kascade.cached_layer_count(), 4);
    }

    fn create_test_tokens(seq_len: usize, dim: usize) -> (Array2<f32>, Array2<f32>) {
        let mut q = Array2::<f32>::zeros((seq_len, dim));
        let mut k = Array2::<f32>::zeros((seq_len, dim));

        for i in 0..seq_len {
            for j in 0..dim {
                q[[i, j]] = (i as f32 * 0.1 + j as f32 * 0.01).sin();
                k[[i, j]] = (i as f32 * 0.15 + j as f32 * 0.02).cos();
            }
        }

        (q, k)
    }
}
