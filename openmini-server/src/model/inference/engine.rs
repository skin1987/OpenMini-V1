//! 推理引擎核心模块
//!
//! 提供底层推理计算功能，包括：
//! - KV Cache 管理
//! - 注意力计算
//! - DSA 稀疏注意力优化（见 dsa.rs）

#![allow(dead_code)]

use ndarray::{s, Array2};
use std::borrow::Cow;

use crate::error::{AppError, EngineError};
use crate::kernel::cpu::simd_softmax::simd_softmax_rows;

use super::dsa::DSATopKConfig;
use super::model::ModelConfig;
use crate::hardware::kv_cache::streaming::{StreamingAttention, StreamingAttentionConfig};

/// KV Cache 返回类型别名
type KvCacheTuple<'a> = Option<(Cow<'a, Array2<f32>>, Cow<'a, Array2<f32>>)>;

// ============================================================================
// KV Cache
// ============================================================================

/// 默认块大小（tokens）
const DEFAULT_CHUNK_SIZE: usize = 512;

/// KV Cache 层（分块存储优化）
///
/// 使用分块存储避免每次追加时的完整克隆，将 O(n²) 复杂度降为 O(n)。
/// 适合长序列场景（>2k tokens）。
#[derive(Clone)]
#[allow(dead_code)]
pub struct KvCacheLayer {
    /// Key 缓存块
    chunks_k: Vec<Array2<f32>>,
    /// Value 缓存块
    #[allow(dead_code)]
    chunks_v: Vec<Array2<f32>>,
    /// 总行数
    total_rows: usize,
    /// 列维度（head_dim）
    cols: Option<usize>,
    /// 块大小
    chunk_size: usize,
}

impl std::fmt::Debug for KvCacheLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvCacheLayer")
            .field("chunks", &self.chunks_k.len())
            .field("total_rows", &self.total_rows)
            .field("cols", &self.cols)
            .field("chunk_size", &self.chunk_size)
            .finish()
    }
}

impl Default for KvCacheLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl KvCacheLayer {
    /// 创建新的 KV Cache 层
    pub fn new() -> Self {
        Self {
            chunks_k: Vec::new(),
            chunks_v: Vec::new(),
            total_rows: 0,
            cols: None,
            chunk_size: DEFAULT_CHUNK_SIZE,
        }
    }

    /// 创建指定块大小的 KV Cache 层
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunks_k: Vec::new(),
            chunks_v: Vec::new(),
            total_rows: 0,
            cols: None,
            chunk_size,
        }
    }

    /// 更新缓存（自动分块累积）
    ///
    /// 将新的 K、V 追加到缓存中。使用分块存储，避免完整克隆。
    /// 当最后一个块的行数小于 `chunk_size` 时，会尝试将新数据合并到最后一个块，
    /// 减少小块数量，提高 `get()` 拼接效率。
    pub fn update(&mut self, k: Array2<f32>, v: Array2<f32>) -> Result<(), AppError> {
        let rows = k.nrows();
        let cols = k.ncols();

        // 验证维度一致性
        if let Some(existing_cols) = self.cols {
            if cols != existing_cols {
                return Err(AppError::Engine(EngineError::KvCacheDimensionMismatch {
                    expected: existing_cols,
                    actual: cols,
                }));
            }
            if v.ncols() != existing_cols {
                return Err(AppError::Engine(EngineError::KvCacheDimensionMismatch {
                    expected: existing_cols,
                    actual: v.ncols(),
                }));
            }
        } else {
            self.cols = Some(cols);
        }
        if v.nrows() != rows {
            return Err(AppError::Engine(EngineError::KvCacheRowMismatch {
                expected: rows,
                actual: v.nrows(),
            }));
        }

        // 尝试合并到最后一个块（合并后不超过 chunk_size）
        // 注：rows > 0，所以 last_rows + rows <= chunk_size 已隐含 last_rows < chunk_size
        let should_merge = self
            .chunks_k
            .last()
            .is_some_and(|last_k| last_k.nrows() + rows <= self.chunk_size);

        if should_merge {
            // 使用 pop 获取最后一个块的所有权，合并后 push 回去
            if let (Some(last_k), Some(last_v)) = (self.chunks_k.pop(), self.chunks_v.pop()) {
                let last_rows = last_k.nrows();
                let total_new_rows = last_rows + rows;

                // 创建合并后的块
                let mut merged_k = Array2::zeros((total_new_rows, cols));
                let mut merged_v = Array2::zeros((total_new_rows, cols));

                // 复制原有数据和新数据
                merged_k.slice_mut(s![..last_rows, ..]).assign(&last_k);
                merged_k.slice_mut(s![last_rows.., ..]).assign(&k);
                merged_v.slice_mut(s![..last_rows, ..]).assign(&last_v);
                merged_v.slice_mut(s![last_rows.., ..]).assign(&v);

                // 放回合并后的块
                self.chunks_k.push(merged_k);
                self.chunks_v.push(merged_v);
            } else {
                // 如果pop失败（不应该发生），创建新块
                self.chunks_k.push(k);
                self.chunks_v.push(v);
            }
        } else {
            // 创建新块
            self.chunks_k.push(k);
            self.chunks_v.push(v);
        }

        self.total_rows += rows;
        Ok(())
    }

    /// 获取缓存（零拷贝优化）
    ///
    /// 单块场景：返回 `Cow::Borrowed`，零拷贝！
    /// 多块场景：返回 `Cow::Owned`（需要拼接所有块）。
    ///
    /// 如果只需要遍历数据，建议使用 `iter_chunks()`。
    pub fn get(&self) -> KvCacheTuple<'_> {
        if self.chunks_k.is_empty() {
            return None;
        }

        // 单块场景：零拷贝返回借用
        if self.chunks_k.len() == 1 {
            return Some((
                Cow::Borrowed(&self.chunks_k[0]),
                Cow::Borrowed(&self.chunks_v[0]),
            ));
        }

        // 多块场景：需要合并
        let cols = self.cols?;

        // 预分配内存
        let mut combined_k = Array2::zeros((self.total_rows, cols));
        let mut combined_v = Array2::zeros((self.total_rows, cols));

        // 拼接所有块
        let mut offset = 0;
        for (k_chunk, v_chunk) in self.chunks_k.iter().zip(self.chunks_v.iter()) {
            let rows = k_chunk.nrows();
            combined_k
                .slice_mut(s![offset..offset + rows, ..])
                .assign(k_chunk);
            combined_v
                .slice_mut(s![offset..offset + rows, ..])
                .assign(v_chunk);
            offset += rows;
        }

        Some((Cow::Owned(combined_k), Cow::Owned(combined_v)))
    }

    /// 获取缓存引用（兼容旧接口）
    ///
    /// 返回第一个块的引用。如果只有一个块，这很高效。
    /// 如果有多个块，建议使用 `get()` 获取完整数据。
    pub fn get_first_chunk(&self) -> Option<(&Array2<f32>, &Array2<f32>)> {
        self.chunks_k.first().zip(self.chunks_v.first())
    }

    /// 迭代所有块
    pub fn iter_chunks(&self) -> impl Iterator<Item = (&Array2<f32>, &Array2<f32>)> {
        self.chunks_k.iter().zip(self.chunks_v.iter())
    }

    /// 获取指定索引的块
    ///
    /// # 参数
    /// - `idx`: 块索引（从 0 开始）
    ///
    /// # 返回
    /// 返回指定块的 K、V 引用，如果索引越界则返回 None。
    ///
    /// # 用途
    /// 便于实现分块注意力计算，避免拼接所有块的开销。
    pub fn get_chunk(&self, idx: usize) -> Option<(&Array2<f32>, &Array2<f32>)> {
        self.chunks_k.get(idx).zip(self.chunks_v.get(idx))
    }

    /// 获取指定索引的块（可变引用）
    ///
    /// # Warning
    /// 修改块内容时必须保持维度不变（行数、列数）。
    /// 改变维度会导致后续 `get()` 或 `iter_chunks()` 出现 panic 或数据错误。
    ///
    /// # Safety
    /// 此方法本身是安全的，但调用者需确保不破坏数据不变性。
    pub fn get_chunk_mut(&mut self, idx: usize) -> Option<(&mut Array2<f32>, &mut Array2<f32>)> {
        self.chunks_k.get_mut(idx).zip(self.chunks_v.get_mut(idx))
    }

    /// 获取块数量
    pub fn chunk_count(&self) -> usize {
        self.chunks_k.len()
    }

    /// 获取总行数
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// 清空缓存
    pub fn clear(&mut self) {
        self.chunks_k.clear();
        self.chunks_v.clear();
        self.total_rows = 0;
        self.cols = None;
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.chunks_k.is_empty()
    }

    /// 整理碎片（合并所有小块）
    ///
    /// 将所有块合并为一个大块，减少内存碎片。
    /// 适用于空闲时优化，或在需要频繁调用 `get()` 前整理。
    ///
    /// # 注意
    /// 此方法会分配新内存并复制所有数据，仅在块数量较多时使用。
    pub fn defrag(&mut self) {
        if self.chunks_k.len() <= 1 {
            return;
        }

        // 手动拼接所有块（避免借用冲突）
        let cols = self.cols.unwrap_or(0);
        if cols == 0 || self.total_rows == 0 {
            return;
        }

        // 预分配内存并拼接
        let mut combined_k = Array2::zeros((self.total_rows, cols));
        let mut combined_v = Array2::zeros((self.total_rows, cols));

        let mut offset = 0;
        for (k_chunk, v_chunk) in self.chunks_k.iter().zip(self.chunks_v.iter()) {
            let rows = k_chunk.nrows();
            combined_k
                .slice_mut(s![offset..offset + rows, ..])
                .assign(k_chunk);
            combined_v
                .slice_mut(s![offset..offset + rows, ..])
                .assign(v_chunk);
            offset += rows;
        }

        // 替换为单个合并块
        self.chunks_k = vec![combined_k];
        self.chunks_v = vec![combined_v];
    }

    /// 获取碎片化程度
    ///
    /// 返回块数量与理想块数量的比率。比率越低越高效。
    /// 理想块数量 = ceil(total_rows / chunk_size)
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.total_rows == 0 {
            return 0.0;
        }
        let ideal_chunks = self.total_rows.div_ceil(self.chunk_size);
        if ideal_chunks == 0 {
            return 0.0;
        }
        self.chunks_k.len() as f64 / ideal_chunks as f64
    }
}

/// 推理上下文
#[derive(Debug)]
pub struct InferenceContext {
    /// 每层的 KV Cache
    pub kv_caches: Vec<KvCacheLayer>,
    /// StreamingAttention (每层独立)
    pub streaming_attentions: Vec<Option<StreamingAttention>>,
    /// Streaming 配置
    pub streaming_config: StreamingAttentionConfig,
    /// 当前序列长度
    pub seq_len: usize,
    /// DSA 配置
    pub dsa_config: DSATopKConfig,
    /// 是否启用 DSA
    pub use_dsa: bool,
}

impl InferenceContext {
    /// 创建新的推理上下文
    pub fn new(config: &ModelConfig) -> Self {
        let streaming_config = StreamingAttentionConfig {
            block_size: 512,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            max_blocks: 1024,
        };
        Self {
            kv_caches: (0..config.num_hidden_layers)
                .map(|_| KvCacheLayer::new())
                .collect(),
            streaming_attentions: (0..config.num_hidden_layers).map(|_| None).collect(),
            streaming_config,
            seq_len: 0,
            dsa_config: DSATopKConfig::default(),
            use_dsa: true,
        }
    }

    /// 启用 StreamingAttention (为每层创建独立实例)
    ///
    /// # Warning
    /// **必须在任何 `update_kv` 调用之前调用此方法！**
    ///
    /// 如果在已有 KV 缓存后调用，已有数据将被丢弃，`seq_len` 将重置为 0。
    /// 这是因为 StreamingAttention 和传统 KV Cache 使用不同的存储结构，
    /// 当前实现不支持数据迁移。
    ///
    /// # Side Effects
    /// - 清空所有传统 KV 缓存
    /// - 重置 `seq_len` 为 0
    /// - 为每层创建新的 `StreamingAttention` 实例
    ///
    /// # Example
    /// ```ignore
    /// let mut ctx = InferenceContext::new(&config);
    /// ctx.enable_streaming_attention(); // 在第一次 update_kv 之前调用
    /// ctx.update_kv(0, k, v);           // 现在数据会写入 StreamingAttention
    /// ```
    pub fn enable_streaming_attention(&mut self) {
        for layer in &mut self.kv_caches {
            layer.clear();
        }
        for sa in &mut self.streaming_attentions {
            *sa = Some(StreamingAttention::new(self.streaming_config.clone()));
        }
        self.seq_len = 0;
    }

    /// 更新指定层的 KV Cache (Zero-Copy 优化)
    ///
    /// # 参数
    /// - `layer_idx`: 层索引（从 0 开始）
    /// - `k`: Key 矩阵，形状为 `(seq_len, head_dim)`，**必须是 C-order 连续内存布局**
    /// - `v`: Value 矩阵，形状为 `(seq_len, head_dim)`，**必须是 C-order 连续内存布局**
    ///
    /// # 内存布局要求
    /// 传入的 `Array2<f32>` 必须是 C-order（行优先）连续布局。
    /// 可以通过 `array.is_contiguous()` 检查，或使用 `array.to_owned()` 确保连续。
    ///
    /// # Errors
    /// 当启用 StreamingAttention 且传入的数组不是连续内存布局时返回错误。
    /// 这通常发生在使用 `slice`、`select` 等操作得到的视图上。
    pub fn update_kv(
        &mut self,
        layer_idx: usize,
        k: Array2<f32>,
        v: Array2<f32>,
    ) -> Result<(), AppError> {
        let seq_len = k.nrows();

        if let Some(ref mut sa) = self.streaming_attentions[layer_idx] {
            let k_flat = k
                .as_slice()
                .ok_or(AppError::Engine(EngineError::ArrayNotContiguous))?;
            let v_flat = v
                .as_slice()
                .ok_or(AppError::Engine(EngineError::ArrayNotContiguous))?;
            sa.write(self.seq_len, k_flat, v_flat).map_err(|e| {
                AppError::Engine(EngineError::StreamingAttentionWriteFailed(e.to_string()))
            })?;
        } else {
            self.kv_caches[layer_idx].update(k, v)?;
        }

        self.seq_len += seq_len;
        Ok(())
    }

    /// 获取指定层的 KV Cache（零拷贝优化）
    ///
    /// # 注意
    /// 当该层启用 StreamingAttention 时，此方法返回 None。
    /// 请使用 `streaming_attention_query` 进行增量注意力计算。
    ///
    /// # 性能说明
    /// 单块时零拷贝返回，多块时需要拼接。如果只需要遍历数据，
    /// 建议使用 `kv_caches[layer_idx].iter_chunks()`。
    pub fn get_kv(&self, layer_idx: usize) -> KvCacheTuple<'_> {
        if self.streaming_attentions[layer_idx].is_some() {
            None
        } else {
            self.kv_caches[layer_idx].get()
        }
    }

    /// 使用 StreamingAttention 进行增量注意力计算
    ///
    /// # 参数
    /// - `layer_idx`: 层索引
    /// - `query`: 查询向量
    /// - `pos`: 当前位置
    /// - `scale`: 缩放因子
    pub fn streaming_attention_query(
        &self,
        layer_idx: usize,
        query: &[f32],
        pos: usize,
        scale: f32,
    ) -> Option<Vec<f32>> {
        self.streaming_attentions[layer_idx]
            .as_ref()
            .map(|sa| sa.incremental_attention(query, pos, scale))
    }

    /// 清空所有缓存
    pub fn clear(&mut self) {
        for layer in &mut self.kv_caches {
            layer.clear();
        }
        for s in self.streaming_attentions.iter_mut().flatten() {
            s.clear();
        }
        self.seq_len = 0;
    }

    /// 获取 StreamingAttention 统计信息（汇总所有层）
    pub fn get_streaming_stats(
        &self,
    ) -> Vec<Option<crate::hardware::kv_cache::streaming::StreamingAttentionStats>> {
        self.streaming_attentions
            .iter()
            .map(|sa| sa.as_ref().map(|s| s.stats()))
            .collect()
    }

    /// 检查是否应该使用 DSA
    pub fn should_use_dsa(&self, threshold: usize) -> bool {
        self.use_dsa && self.seq_len > threshold
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// Softmax 按行计算
///
/// 对输入矩阵的每一行独立计算 softmax，使每行元素和为 1。
/// 使用数值稳定的算法：减去最大值后再计算 exp。
/// 优化：使用乘法逆代替除法，减少重复计算。
///
/// # 泛型参数
/// - `S`: 数组存储类型，支持 `Array2<f32>` 和 `ArrayView2<'_, f32>`
///
/// # 性能优化
///
/// 当前版本: SIMD 优化版 (自动检测 CPU 特性)
/// - AVX2: ~4x 加速 (x86_64)
/// - NEON: ~2x 加速 (aarch64)
/// - Scalar: 回退实现 (无 SIMD 支持)
pub fn softmax_rows<S>(x: &ndarray::ArrayBase<S, ndarray::Ix2>) -> Array2<f32>
where
    S: ndarray::Data<Elem = f32>,
{
    // 使用 SIMD 优化版本（自动检测并选择最优路径）
    simd_softmax_rows(x)
}

/// 构建多模态提示
///
/// # 参数
/// - `text_tokens`: 预编码的文本 token ID 序列（需使用 tokenizer 编码）
/// - `num_image_tokens`: 图像 token 数量
///
/// # 注意
/// 此函数不执行 tokenization，调用者需使用 tokenizer 将文本编码为 token ID。
/// 直接使用 Unicode 码点值作为 token ID 是无效的。
pub fn build_multimodal_prompt(text_tokens: &[usize], num_image_tokens: usize) -> Vec<usize> {
    let mut tokens = Vec::with_capacity(text_tokens.len() + num_image_tokens + 2);

    tokens.push(super::model::IM_START_TOKEN_ID);
    tokens.extend(std::iter::repeat_n(
        super::model::IM_PATCH_TOKEN_ID,
        num_image_tokens,
    ));
    tokens.push(super::model::IM_END_TOKEN_ID);

    // 使用预编码的 token ID
    tokens.extend_from_slice(text_tokens);

    tokens
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::inference::model::{IM_END_TOKEN_ID, IM_PATCH_TOKEN_ID, IM_START_TOKEN_ID};

    #[test]
    fn test_kv_cache_layer() {
        let mut layer = KvCacheLayer::new();
        assert!(layer.get().is_none());
        assert!(layer.is_empty());

        let k = Array2::zeros((10, 64));
        let v = Array2::zeros((10, 64));
        let _ = layer.update(k, v);

        assert!(layer.get().is_some());
        assert_eq!(layer.chunk_count(), 1);
        assert_eq!(layer.total_rows(), 10);
    }

    #[test]
    fn test_inference_context() {
        let config = ModelConfig::default();
        let ctx = InferenceContext::new(&config);

        assert_eq!(ctx.kv_caches.len(), config.num_hidden_layers);
        assert_eq!(ctx.seq_len, 0);
    }

    #[test]
    fn test_softmax_rows() {
        let x = ndarray::arr2(&[[1.0f32, 2.0, 3.0]]);
        let result = softmax_rows(&x);

        let sum: f32 = result.row(0).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_update() {
        let mut cache = KvCacheLayer::new();

        // 第一次更新
        let k1 = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let v1 = ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let _ = cache.update(k1, v1);

        assert_eq!(cache.chunk_count(), 1);
        let (k, v) = cache.get().unwrap();
        assert_eq!(k.dim(), (2, 2));
        assert_eq!(v.dim(), (2, 2));

        // 第二次更新（自动合并到第一个块，因为 2 < 512）
        let k2 = ndarray::arr2(&[[9.0, 10.0], [11.0, 12.0]]);
        let v2 = ndarray::arr2(&[[13.0, 14.0], [15.0, 16.0]]);
        let _ = cache.update(k2, v2);

        // 验证自动合并：仍然只有 1 个块
        assert_eq!(cache.chunk_count(), 1);
        let (k, v) = cache.get().unwrap();
        // 验证行数正确追加
        assert_eq!(k.dim(), (4, 2)); // 2 + 2 = 4
        assert_eq!(v.dim(), (4, 2));

        // 验证数据正确
        assert!((k[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((k[[1, 0]] - 3.0).abs() < 1e-6);
        assert!((k[[2, 0]] - 9.0).abs() < 1e-6);
        assert!((k[[3, 0]] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_multiple_updates() {
        let mut cache = KvCacheLayer::new();

        // 多次更新
        for i in 0..5 {
            let k = ndarray::arr2(&[[i as f32, (i + 1) as f32]]);
            let v = ndarray::arr2(&[[i as f32, (i + 2) as f32]]);
            let _ = cache.update(k, v);
        }

        // 验证自动合并：5 次更新，每次 1 行，默认 chunk_size=512
        // 所有数据应该合并到第一个块
        assert_eq!(cache.chunk_count(), 1);
        assert_eq!(cache.total_rows(), 5);

        let (k, v) = cache.get().unwrap();
        // 验证最终行数
        assert_eq!(k.dim(), (5, 2));
        assert_eq!(v.dim(), (5, 2));
    }

    #[test]
    fn test_kv_cache_auto_merge() {
        // 测试自动分块累积逻辑
        let mut cache = KvCacheLayer::with_chunk_size(10);

        // 第一次更新：创建第一个块
        let k1 = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let v1 = ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let _ = cache.update(k1, v1);
        assert_eq!(cache.chunk_count(), 1);

        // 第二次更新：合并到第一个块（因为 2 < 10）
        let k2 = ndarray::arr2(&[[9.0, 10.0]]);
        let v2 = ndarray::arr2(&[[11.0, 12.0]]);
        let _ = cache.update(k2, v2);
        assert_eq!(cache.chunk_count(), 1); // 仍然只有一个块
        assert_eq!(cache.total_rows(), 3);

        // 验证数据正确
        let (k, _v) = cache.get().unwrap();
        assert_eq!(k.dim(), (3, 2));
        assert!((k[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((k[[2, 0]] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_chunk_boundary() {
        // 测试块边界行为
        let mut cache = KvCacheLayer::with_chunk_size(5);

        // 第一次：5 行，刚好填满一个块
        let k1 = Array2::zeros((5, 4));
        let v1 = Array2::zeros((5, 4));
        let _ = cache.update(k1, v1);
        assert_eq!(cache.chunk_count(), 1);

        // 第二次：3 行，应该创建新块（因为第一个块已满）
        let k2 = Array2::zeros((3, 4));
        let v2 = Array2::zeros((3, 4));
        let _ = cache.update(k2, v2);
        assert_eq!(cache.chunk_count(), 2);

        // 第三次：2 行，应该合并到第二个块
        let k3 = Array2::zeros((2, 4));
        let v3 = Array2::zeros((2, 4));
        let _ = cache.update(k3, v3);
        assert_eq!(cache.chunk_count(), 2); // 仍然是 2 个块
        assert_eq!(cache.total_rows(), 10);
    }

    #[test]
    fn test_kv_cache_defrag() {
        // 测试 defrag 功能
        // 使用较小的 chunk_size 强制创建多个块
        let mut cache = KvCacheLayer::with_chunk_size(10);

        // 模拟多个小块（每次刚好填满一个块，不会触发自动合并）
        for i in 0..5 {
            let k = Array2::from_shape_fn((10, 4), |(r, c)| (i * 10 + r) as f32 + c as f32);
            let v = Array2::zeros((10, 4));
            let _ = cache.update(k, v);
        }

        // 验证有 5 个块
        assert_eq!(cache.chunk_count(), 5);

        // 执行 defrag
        cache.defrag();

        // 验证合并为一个块
        assert_eq!(cache.chunk_count(), 1);
        assert_eq!(cache.total_rows(), 50);

        // 验证数据完整性
        let (k, _v) = cache.get().unwrap();
        assert!((k[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((k[[49, 0]] - 49.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_fragmentation_ratio() {
        let mut cache = KvCacheLayer::with_chunk_size(10);

        // 理想情况：数据刚好填满块
        let k1 = Array2::zeros((10, 4));
        let v1 = Array2::zeros((10, 4));
        let _ = cache.update(k1, v1);
        assert!((cache.fragmentation_ratio() - 1.0).abs() < 0.01);

        // 碎片化：添加刚好填满的块（不会触发自动合并）
        for _ in 0..5 {
            let k = Array2::zeros((10, 4));
            let v = Array2::zeros((10, 4));
            let _ = cache.update(k, v);
        }

        // 6 个块，理想也是 6 个块，比率 = 1
        assert_eq!(cache.chunk_count(), 6);
        assert!((cache.fragmentation_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kv_cache_chunked_performance() {
        // 测试分块存储的正确性和自动合并
        let mut cache = KvCacheLayer::new();

        // 模拟长序列：100 次更新，每次 100 tokens
        for batch in 0..100 {
            let k = Array2::from_shape_fn((100, 64), |(i, j)| {
                (batch * 100 + i) as f32 + j as f32 * 0.01
            });
            let v = Array2::from_shape_fn((100, 64), |(i, j)| {
                (batch * 100 + i) as f32 * 2.0 + j as f32 * 0.01
            });
            let _ = cache.update(k, v);
        }

        // 验证总行数
        assert_eq!(cache.total_rows(), 10000);

        // 验证块数量在合理范围内（自动合并后应该远小于 100）
        // 默认 chunk_size=512，每次更新 100 行，预期块数量约 17
        assert!(
            cache.chunk_count() < 50,
            "Chunk count should be reduced by auto-merge"
        );

        // 验证拼接后的数据正确
        let (k, _v) = cache.get().unwrap();
        assert_eq!(k.dim(), (10000, 64));

        // 验证第一个和最后一个元素
        assert!((k[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((k[[9999, 0]] - 9999.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_get_chunk() {
        // 使用较小的 chunk_size 以便测试多个块
        let mut cache = KvCacheLayer::with_chunk_size(2);

        // 添加 3 个块（每个 2 行）
        for i in 0..3 {
            let k = ndarray::arr2(&[[i as f32, (i + 1) as f32], [(i + 2) as f32, (i + 3) as f32]]);
            let v = ndarray::arr2(&[
                [i as f32 * 2.0, (i + 1) as f32 * 2.0],
                [(i + 2) as f32 * 2.0, (i + 3) as f32 * 2.0],
            ]);
            let _ = cache.update(k, v);
        }

        // 验证：由于每次更新 2 行，刚好等于 chunk_size，所以创建了 3 个块
        assert_eq!(cache.chunk_count(), 3);

        // 验证 get_chunk
        assert!(cache.get_chunk(0).is_some());
        assert!(cache.get_chunk(1).is_some());
        assert!(cache.get_chunk(2).is_some());
        assert!(cache.get_chunk(3).is_none()); // 越界

        // 验证第一个块的数据
        let (k0, _v0) = cache.get_chunk(0).unwrap();
        assert_eq!(k0.dim(), (2, 2));
        assert!((k0[[0, 0]] - 0.0).abs() < 1e-6);

        // 验证最后一个块的数据
        let (k2, _v2) = cache.get_chunk(2).unwrap();
        assert!((k2[[0, 0]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_rows_with_view() {
        // 测试 softmax_rows 接受 ArrayView2
        let x = ndarray::arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        // 使用 owned 数组
        let result1 = softmax_rows(&x);

        // 使用 view
        let view = x.view();
        let result2 = softmax_rows(&view);

        // 验证结果一致
        for i in 0..2 {
            let sum1: f32 = result1.row(i).sum();
            let sum2: f32 = result2.row(i).sum();
            assert!((sum1 - 1.0).abs() < 1e-6);
            assert!((sum2 - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_get_kv_with_streaming_attention() {
        let config = ModelConfig::default();
        let mut ctx = InferenceContext::new(&config);

        // 初始状态：get_kv 应该返回 None（kv_caches 为空）
        assert!(ctx.get_kv(0).is_none());

        // 禁用 StreamingAttention 时更新 kv_caches
        let k = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let v = ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let _ = ctx.update_kv(0, k, v);

        // get_kv 应该返回有效数据
        let (kv_k, _kv_v) = ctx.get_kv(0).unwrap();
        assert_eq!(kv_k.dim(), (2, 2));

        // 验证 seq_len 已更新
        assert_eq!(ctx.seq_len, 2);

        // 启用 StreamingAttention
        ctx.enable_streaming_attention();

        // 启用后 get_kv 应该返回 None（因为该层现在使用 StreamingAttention）
        assert!(ctx.get_kv(0).is_none());

        // streaming_attentions[0] 应该有数据
        assert!(ctx.streaming_attentions[0].is_some());

        // 验证 seq_len 已重置为 0
        assert_eq!(ctx.seq_len, 0);
    }

    #[test]
    fn test_enable_streaming_resets_seq_len() {
        let config = ModelConfig::default();
        let mut ctx = InferenceContext::new(&config);

        // 写入多层数据
        for layer in 0..3 {
            let k = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
            let v = ndarray::arr2(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
            let _ = ctx.update_kv(layer, k, v);
        }

        // 验证 seq_len 累积（每次 3 行，共 3 次 = 9）
        assert_eq!(ctx.seq_len, 9);

        // 启用 StreamingAttention
        ctx.enable_streaming_attention();

        // 验证 seq_len 重置为 0
        assert_eq!(ctx.seq_len, 0);

        // 验证所有层都启用了 StreamingAttention
        for (i, sa) in ctx.streaming_attentions.iter().enumerate() {
            assert!(sa.is_some(), "Layer {} should have StreamingAttention", i);
        }
    }

    #[test]
    fn test_streaming_stats_returns_all_layers() {
        let config = ModelConfig::default();
        let mut ctx = InferenceContext::new(&config);

        // 启用 StreamingAttention
        ctx.enable_streaming_attention();

        // 获取统计信息
        let stats = ctx.get_streaming_stats();

        // 验证返回的 Vec 长度与层数一致
        assert_eq!(stats.len(), config.num_hidden_layers);

        // 所有层都应该有统计信息（因为都启用了）
        for (i, stat) in stats.iter().enumerate() {
            assert!(stat.is_some(), "Layer {} should have stats", i);
        }
    }

    #[test]
    fn test_clear_resets_all_state() {
        let config = ModelConfig::default();
        let mut ctx = InferenceContext::new(&config);

        // 写入一些数据
        let k = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let v = ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let _ = ctx.update_kv(0, k.clone(), v.clone());
        let _ = ctx.update_kv(1, k, v);

        assert_eq!(ctx.seq_len, 4);

        // 清空
        ctx.clear();

        // 验证状态已重置
        assert_eq!(ctx.seq_len, 0);
        assert!(ctx.get_kv(0).is_none());
        assert!(ctx.get_kv(1).is_none());
    }

    #[test]
    fn test_traditional_vs_streaming_mode() {
        let config = ModelConfig::default();
        let head_dim = config.hidden_size / config.num_attention_heads;

        // 传统模式
        let mut ctx_traditional = InferenceContext::new(&config);
        let k = Array2::zeros((2, head_dim));
        let v = Array2::zeros((2, head_dim));
        let _ = ctx_traditional.update_kv(0, k.clone(), v.clone());

        // 验证传统模式数据存储正确
        let (kv_k, kv_v) = ctx_traditional.get_kv(0).unwrap();
        assert_eq!(kv_k.dim(), (2, head_dim));
        assert_eq!(kv_v.dim(), (2, head_dim));

        // 流式模式 - 验证启用后状态正确
        let mut ctx_streaming = InferenceContext::new(&config);
        ctx_streaming.enable_streaming_attention();

        // 验证 streaming_attentions 已启用（每层都有实例）
        assert!(ctx_streaming.streaming_attentions[0].is_some());

        // 验证启用后 get_kv 返回 None（因为该层使用流式）
        assert!(ctx_streaming.get_kv(0).is_none());

        // 验证 seq_len 初始为 0
        assert_eq!(ctx_streaming.seq_len, 0);
    }

    #[test]
    fn test_multiple_update_kv_accumulates_seq_len() {
        let config = ModelConfig::default();
        let mut ctx = InferenceContext::new(&config);

        // 多次更新
        for _ in 0..3 {
            let k = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
            let v = ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);
            let _ = ctx.update_kv(0, k, v);
        }

        // 验证 seq_len 累积正确（每次 2 行，共 3 次 = 6）
        assert_eq!(ctx.seq_len, 6);

        // 验证 kv_caches 数据正确累积
        let (k, _v) = ctx.get_kv(0).unwrap();
        assert_eq!(k.dim(), (6, 2));
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_kv_cache_layer_default_and_new() {
        // 测试默认构造和自定义chunk_size
        let layer_default = KvCacheLayer::default();
        assert!(layer_default.is_empty());
        assert_eq!(layer_default.total_rows(), 0);
        assert_eq!(layer_default.chunk_count(), 0);

        let layer_custom = KvCacheLayer::with_chunk_size(256);
        assert!(layer_custom.is_empty());
        assert_eq!(layer_custom.total_rows(), 0);
    }

    #[test]
    fn test_kv_cache_layer_clear_resets_state() {
        // 测试clear后状态完全重置
        let mut layer = KvCacheLayer::with_chunk_size(10);

        // 添加数据
        let k = Array2::from_shape_fn((5, 64), |(i, j)| i as f32 + j as f32);
        let v = Array2::from_shape_fn((5, 64), |(i, j)| (i as f32 + j as f32) * 2.0);
        let _ = layer.update(k, v);

        assert!(!layer.is_empty());
        assert_eq!(layer.total_rows(), 5);

        // 清空
        layer.clear();

        // 验证状态重置
        assert!(layer.is_empty());
        assert_eq!(layer.total_rows(), 0);
        assert_eq!(layer.chunk_count(), 0);
        assert!(layer.get().is_none());
    }

    #[test]
    fn test_kv_cache_layer_get_first_chunk() {
        // 测试get_first_chunk方法
        let mut layer = KvCacheLayer::new();

        // 空时返回None
        assert!(layer.get_first_chunk().is_none());

        // 添加数据后返回第一个块
        let k = Array2::zeros((10, 64));
        let v = Array2::zeros((10, 64));
        let _ = layer.update(k, v);

        let (first_k, first_v) = layer.get_first_chunk().unwrap();
        assert_eq!(first_k.dim(), (10, 64));
        assert_eq!(first_v.dim(), (10, 64));
    }

    #[test]
    fn test_kv_cache_layer_iter_chunks() {
        // 测试iter_chunks迭代器
        let mut layer = KvCacheLayer::with_chunk_size(5);

        // 添加多个块
        for i in 0..3 {
            let k = Array2::from_shape_fn((5, 4), |(r, c)| (i * 5 + r) as f32 + c as f32);
            let v = Array2::zeros((5, 4));
            let _ = layer.update(k, v);
        }

        // 验证迭代器遍历所有块
        let chunk_count = layer.iter_chunks().count();
        assert_eq!(chunk_count, 3);

        // 验证每个块的行数
        for (_idx, (k_chunk, _v_chunk)) in layer.iter_chunks().enumerate() {
            assert_eq!(k_chunk.nrows(), 5);
            assert_eq!(k_chunk.ncols(), 4);
        }
    }

    #[test]
    fn test_kv_cache_layer_get_chunk_mut() {
        // 测试可变引用访问块
        let mut layer = KvCacheLayer::with_chunk_size(10);

        let k = Array2::zeros((5, 64));
        let v = Array2::zeros((5, 64));
        let _ = layer.update(k, v);

        // 获取可变引用并修改数据
        if let Some((k_mut, v_mut)) = layer.get_chunk_mut(0) {
            k_mut[[0, 0]] = 42.0;
            v_mut[[0, 0]] = 99.0;
        }

        // 验证修改生效
        let (k_read, v_read) = layer.get().unwrap();
        assert!((k_read[[0, 0]] - 42.0).abs() < 1e-6);
        assert!((v_read[[0, 0]] - 99.0).abs() < 1e-6);
    }

    #[test]
    fn test_kv_cache_layer_fragmentation_edge_cases() {
        // 测试碎片化率的边界条件
        let mut layer = KvCacheLayer::with_chunk_size(10);

        // 空缓存碎片化率应为0
        assert!((layer.fragmentation_ratio() - 0.0).abs() < 0.01);

        // 单个块且未满
        let k = Array2::zeros((5, 4));
        let v = Array2::zeros((5, 4));
        let _ = layer.update(k, v);

        // 5行，理想块数=1，实际块数=1，比率=1.0
        assert!((layer.fragmentation_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kv_cache_layer_defrag_single_block() {
        // 测试单块时的defrag（应该立即返回）
        let mut layer = KvCacheLayer::with_chunk_size(100);

        let k = Array2::zeros((50, 64));
        let v = Array2::zeros((50, 64));
        let _ = layer.update(k, v);

        assert_eq!(layer.chunk_count(), 1);
        layer.defrag(); // 应该立即返回，不做任何事
        assert_eq!(layer.chunk_count(), 1);
        assert_eq!(layer.total_rows(), 50);
    }

    #[test]
    fn test_inference_context_should_use_dsa() {
        // 测试DSA使用判断逻辑
        let config = ModelConfig::default();
        let mut ctx = InferenceContext::new(&config);

        // 初始状态：seq_len=0，不应该使用DSA
        assert!(!ctx.should_use_dsa(100));

        // 写入一些数据使seq_len超过阈值
        let k = Array2::zeros((150, config.hidden_size / config.num_attention_heads));
        let v = Array2::zeros((150, config.hidden_size / config.num_attention_heads));
        let _ = ctx.update_kv(0, k, v);

        // seq_len > threshold，应该使用DSA
        assert!(ctx.should_use_dsa(100));

        // 禁用DSA后不应该使用
        ctx.use_dsa = false;
        assert!(!ctx.should_use_dsa(100));
    }

    #[test]
    fn test_inference_context_streaming_attention_query() {
        // 测试流式注意力查询接口
        let config = ModelConfig::default();
        let head_dim = config.hidden_size / config.num_attention_heads;
        let mut ctx = InferenceContext::new(&config);

        // 未启用流式注意力时返回None
        let query = vec![0.0f32; head_dim];
        let result = ctx.streaming_attention_query(0, &query, 0, 1.0 / (head_dim as f32).sqrt());
        assert!(result.is_none());

        // 启用流式注意力
        ctx.enable_streaming_attention();

        // 启用后应该返回Some（即使数据可能不完整）
        let result = ctx.streaming_attention_query(0, &query, 0, 1.0 / (head_dim as f32).sqrt());
        assert!(result.is_some());
    }

    #[test]
    fn test_softmax_rows_numerical_stability() {
        // 测试softmax数值稳定性（大值和小值）
        // 大值测试
        let x_large = ndarray::arr2(&[[1000.0f32, 1001.0, 1002.0]]);
        let result_large = softmax_rows(&x_large);
        let sum_large: f32 = result_large.row(0).sum();
        assert!((sum_large - 1.0).abs() < 1e-6);

        // 小值测试
        let x_small = ndarray::arr2(&[[-1000.0f32, -999.0, -998.0]]);
        let result_small = softmax_rows(&x_small);
        let sum_small: f32 = result_small.row(0).sum();
        assert!((sum_small - 1.0).abs() < 1e-6);

        // 全零测试
        let x_zero = ndarray::arr2(&[[0.0f32, 0.0, 0.0]]);
        let result_zero = softmax_rows(&x_zero);
        let sum_zero: f32 = result_zero.row(0).sum();
        assert!((sum_zero - 1.0).abs() < 1e-6);
        // 全零时应该是均匀分布
        for val in result_zero.row(0) {
            assert!((*val - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_build_multimodal_prompt_basic() {
        // 测试多模态提示构建
        let text_tokens: Vec<usize> = vec![100, 200, 300];
        let num_image_tokens = 4;

        let prompt = build_multimodal_prompt(&text_tokens, num_image_tokens);

        // 验证结构: [IM_START] + [IM_PATCH]*4 + [IM_END] + text_tokens
        assert_eq!(prompt.len(), 1 + 4 + 1 + 3); // 9 tokens
        assert_eq!(prompt[0], IM_START_TOKEN_ID);
        for i in 1..=4 {
            assert_eq!(prompt[i], IM_PATCH_TOKEN_ID);
        }
        assert_eq!(prompt[5], IM_END_TOKEN_ID);
        // 验证文本token部分
        for (i, &token) in prompt[6..].iter().enumerate() {
            assert_eq!(token, text_tokens[i]);
        }
    }

    #[test]
    fn test_build_multimodal_prompt_no_images() {
        // 测试无图像的多模态提示
        let text_tokens: Vec<usize> = vec![100, 200];
        let num_image_tokens = 0;

        let prompt = build_multimodal_prompt(&text_tokens, num_image_tokens);

        // 结构: [IM_START] + [] + [IM_END] + text_tokens
        assert_eq!(prompt.len(), 1 + 0 + 1 + 2);
        assert_eq!(prompt[0], IM_START_TOKEN_ID);
        assert_eq!(prompt[1], IM_END_TOKEN_ID);
    }

    // ==================== 零拷贝优化专项测试 ====================

    #[test]
    fn test_get_single_chunk_zero_copy() {
        // 测试单块场景：应该返回 Cow::Borrowed（零拷贝）
        let mut layer = KvCacheLayer::with_chunk_size(512);

        // 添加一个小于 chunk_size 的数据块
        let k = Array2::from_shape_fn((100, 64), |(i, j)| i as f32 + j as f32 * 0.01);
        let v = Array2::from_shape_fn((100, 64), |(i, j)| (i as f32 + j as f32) * 2.0);
        layer.update(k, v).expect("Update should succeed");

        // 验证只有1个块
        assert_eq!(layer.chunk_count(), 1);

        // 获取数据并验证是 Borrowed
        let (k_result, v_result) = layer.get().expect("Should return Some");

        // 验证返回的是 Borrowed（零拷贝）
        match (&k_result, &v_result) {
            (Cow::Borrowed(_), Cow::Borrowed(_)) => {
                // ✅ 零拷贝成功！
            }
            _ => panic!("Single chunk should return Cow::Borrowed for zero-copy optimization"),
        }

        // 验证数据正确性
        assert_eq!(k_result.dim(), (100, 64));
        assert_eq!(v_result.dim(), (100, 64));
        assert!((k_result[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((v_result[[0, 0]] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_multi_chunks_owned() {
        // 测试多块场景：应该返回 Cow::Owned
        let mut layer = KvCacheLayer::with_chunk_size(5); // 小的 chunk_size 强制创建多个块

        // 添加多个刚好填满的块（不会触发自动合并）
        for i in 0..3 {
            let k = Array2::from_shape_fn((5, 4), |(r, c)| (i * 5 + r) as f32 + c as f32);
            let v = Array2::zeros((5, 4));
            layer.update(k, v).expect("Update should succeed");
        }

        // 验证有多个块
        assert_eq!(layer.chunk_count(), 3);

        // 获取数据并验证是 Owned
        let (k_result, v_result) = layer.get().expect("Should return Some");

        // 验证返回的是 Owned（需要拼接）
        match (&k_result, &v_result) {
            (Cow::Owned(_), Cow::Owned(_)) => {
                // ✅ 多块时正确返回 Owned
            }
            _ => panic!("Multiple chunks should return Cow::Owned"),
        }

        // 验证数据正确性和完整性
        assert_eq!(k_result.dim(), (15, 4)); // 3 块 × 5 行
        assert_eq!(v_result.dim(), (15, 4));

        // 验证第一个元素和最后一个元素
        assert!((k_result[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((k_result[[14, 0]] - 14.0).abs() < 1e-6);
    }

    #[test]
    fn test_cow_transparent_usage() {
        // 测试 Cow 对调用方的透明性：可以像引用一样使用
        let mut layer = KvCacheLayer::new();

        let k = Array2::from_shape_fn((10, 8), |(i, j)| i as f32 * 10.0 + j as f32);
        let v = Array2::from_shape_fn((10, 8), |(i, j)| i as f32 * 20.0 + j as f32);
        layer.update(k, v).expect("Update should succeed");

        // 单块场景：获取 Cow
        let (k_cow, v_cow) = layer.get().unwrap();

        // Cow 可以像引用一样使用（Deref trait）
        assert_eq!(k_cow.nrows(), 10);
        assert_eq!(k_cow.ncols(), 8);
        assert_eq!(v_cow.nrows(), 10);

        // 可以访问元素
        assert!((k_cow[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((k_cow[[9, 7]] - 97.0).abs() < 1e-6); // 9*10 + 7 = 97

        // 可以使用 .dim() 等方法
        assert_eq!(k_cow.dim(), (10, 8));

        println!("✅ Cow 对调用方完全透明，无需修改现有代码！");
    }

    #[test]
    fn test_defrag_with_cow() {
        // 测试 defrag 方法与 Cow 的兼容性
        let mut layer = KvCacheLayer::with_chunk_size(10);

        // 创建多个块
        for i in 0..4 {
            let k = Array2::from_shape_fn((10, 4), |(r, c)| (i * 10 + r) as f32 + c as f32);
            let v = Array2::zeros((10, 4));
            layer.update(k, v).expect("Update should succeed");
        }

        assert_eq!(layer.chunk_count(), 4);

        // 执行 defrag
        layer.defrag();

        // 验证合并为一个块
        assert_eq!(layer.chunk_count(), 1);

        // 验证 get() 现在返回 Borrowed（因为只有1个块了）
        let (k_result, _v_result) = layer.get().unwrap();
        match &k_result {
            Cow::Borrowed(_) => {
                // ✅ defrag 后单块，再次调用 get() 返回 Borrowed
            }
            _ => panic!("After defrag, single chunk should return Borrowed"),
        }

        // 验证数据完整性
        assert_eq!(k_result.dim(), (40, 4)); // 4 块 × 10 行
        assert!((k_result[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((k_result[[39, 0]] - 39.0).abs() < 1e-6);
    }
}
