//! MLA 低秩 KV 缓存模块
//!
//! 管理 MLA 中的压缩 KV 缓存：
//! - 存储压缩向量 c_KV（而非完整的 K、V）
//! - 支持增量更新
//! - 支持批量读取和解压

#![allow(dead_code)]

use ndarray::Array2;
use std::sync::RwLock;
use crate::hardware::kv_cache::mla::config::MLAConfig;

#[derive(Debug)]
pub struct MLALatentCache {
    config: MLAConfig,
    c_kv_cache: RwLock<Array2<f32>>,
    q_rope_cache: RwLock<Option<Array2<f32>>>,
    k_rope_cache: RwLock<Option<Array2<f32>>>,
    seq_len: RwLock<usize>,
    max_seq_len: usize,
    q_rope_dim: RwLock<Option<usize>>,
    k_rope_dim: RwLock<Option<usize>>,
}

impl MLALatentCache {
    pub fn new(config: MLAConfig) -> Self {
        let latent_dim = config.latent_dim;
        let max_seq_len = config.max_seq_len;

        Self {
            config: config.clone(),
            c_kv_cache: RwLock::new(Array2::zeros((max_seq_len, latent_dim))),
            q_rope_cache: RwLock::new(None),
            k_rope_cache: RwLock::new(None),
            seq_len: RwLock::new(0),
            max_seq_len,
            q_rope_dim: RwLock::new(None),
            k_rope_dim: RwLock::new(None),
        }
    }

    pub fn with_capacity(max_seq_len: usize, config: &MLAConfig) -> Self {
        let mut adjusted_config = config.clone();
        adjusted_config.max_seq_len = max_seq_len;

        Self {
            config: adjusted_config.clone(),
            c_kv_cache: RwLock::new(Array2::zeros((max_seq_len, config.latent_dim))),
            q_rope_cache: RwLock::new(None),
            k_rope_cache: RwLock::new(None),
            seq_len: RwLock::new(0),
            max_seq_len,
            q_rope_dim: RwLock::new(None),
            k_rope_dim: RwLock::new(None),
        }
    }

    pub fn append(&self, c_kv: &Array2<f32>, q_rope: Option<&Array2<f32>>, k_rope: Option<&Array2<f32>>) -> Result<(), String> {
        if c_kv.nrows() != 1 {
            return Err("Only single token append supported".to_string());
        }

        if c_kv.ncols() != self.config.latent_dim {
            return Err(format!(
                "c_kv dimension mismatch: expected {}, got {}",
                self.config.latent_dim,
                c_kv.ncols()
            ));
        }

        let mut seq_len_guard = self.seq_len.write().map_err(|e| e.to_string())?;
        let current_len = *seq_len_guard;

        if current_len >= self.max_seq_len {
            return Err("Cache full".to_string());
        }

        if let Some(q) = q_rope {
            let q_dim_guard = self.q_rope_dim.read().map_err(|e| e.to_string())?;
            if let Some(expected_dim) = *q_dim_guard {
                if q.ncols() != expected_dim {
                    return Err(format!(
                        "q_rope dimension mismatch: expected {}, got {}",
                        expected_dim,
                        q.ncols()
                    ));
                }
            }
        }

        if let Some(k) = k_rope {
            let k_dim_guard = self.k_rope_dim.read().map_err(|e| e.to_string())?;
            if let Some(expected_dim) = *k_dim_guard {
                if k.ncols() != expected_dim {
                    return Err(format!(
                        "k_rope dimension mismatch: expected {}, got {}",
                        expected_dim,
                        k.ncols()
                    ));
                }
            }
        }

        {
            let mut cache = self.c_kv_cache.write().map_err(|e| e.to_string())?;
            cache.row_mut(current_len).assign(&c_kv.row(0));
        }

        if let Some(q) = q_rope {
            let mut q_cache = self.q_rope_cache.write().map_err(|e| e.to_string())?;
            let mut q_dim_guard = self.q_rope_dim.write().map_err(|e| e.to_string())?;

            if q_cache.is_none() {
                *q_cache = Some(Array2::zeros((self.max_seq_len, q.ncols())));
                *q_dim_guard = Some(q.ncols());
            }
            if let Some(ref mut q_cache) = *q_cache {
                q_cache.row_mut(current_len).assign(&q.row(0));
            }
        }

        if let Some(k) = k_rope {
            let mut k_cache = self.k_rope_cache.write().map_err(|e| e.to_string())?;
            let mut k_dim_guard = self.k_rope_dim.write().map_err(|e| e.to_string())?;

            if k_cache.is_none() {
                *k_cache = Some(Array2::zeros((self.max_seq_len, k.ncols())));
                *k_dim_guard = Some(k.ncols());
            }
            if let Some(ref mut k_cache) = *k_cache {
                k_cache.row_mut(current_len).assign(&k.row(0));
            }
        }

        *seq_len_guard = current_len + 1;

        Ok(())
    }

    pub fn get_c_kv(&self, start: usize, len: usize) -> Result<Array2<f32>, String> {
        let seq_len_guard = self.seq_len.read().map_err(|e| e.to_string())?;
        let total_len = *seq_len_guard;

        if start + len > total_len {
            return Err(format!(
                "Out of bounds: start={}, len={}, total={}",
                start, len, total_len
            ));
        }

        let cache = self.c_kv_cache.read().map_err(|e| e.to_string())?;
        let latent_dim = self.config.latent_dim;
        let mut result = Array2::zeros((len, latent_dim));

        for i in 0..len {
            result.row_mut(i).assign(&cache.row(start + i));
        }

        Ok(result)
    }

    pub fn get_all_c_kv(&self) -> Result<Array2<f32>, String> {
        let seq_len_guard = self.seq_len.read().map_err(|e| e.to_string())?;
        let total_len = *seq_len_guard;

        if total_len == 0 {
            return Ok(Array2::zeros((0, self.config.latent_dim)));
        }

        self.get_c_kv(0, total_len)
    }

    pub fn get_q_rope(&self, start: usize, len: usize) -> Result<Option<Array2<f32>>, String> {
        let q_cache = self.q_rope_cache.read().map_err(|e| e.to_string())?;

        if let Some(ref cache) = *q_cache {
            let seq_len_guard = self.seq_len.read().map_err(|e| e.to_string())?;
            let total_len = *seq_len_guard;

            if start + len > total_len {
                return Err(format!(
                    "Out of bounds: start={}, len={}, total={}",
                    start, len, total_len
                ));
            }

            let mut result = Array2::zeros((len, cache.ncols()));
            for i in 0..len {
                result.row_mut(i).assign(&cache.row(start + i));
            }

            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    pub fn get_k_rope(&self, start: usize, len: usize) -> Result<Option<Array2<f32>>, String> {
        let k_cache = self.k_rope_cache.read().map_err(|e| e.to_string())?;

        if let Some(ref cache) = *k_cache {
            let seq_len_guard = self.seq_len.read().map_err(|e| e.to_string())?;
            let total_len = *seq_len_guard;

            if start + len > total_len {
                return Err(format!(
                    "Out of bounds: start={}, len={}, total={}",
                    start, len, total_len
                ));
            }

            let mut result = Array2::zeros((len, cache.ncols()));
            for i in 0..len {
                result.row_mut(i).assign(&cache.row(start + i));
            }

            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    pub fn seq_len(&self) -> Result<usize, String> {
        let guard = self.seq_len.read().map_err(|e| e.to_string())?;
        Ok(*guard)
    }

    pub fn clear(&self) -> Result<(), String> {
        let mut seq_len_guard = self.seq_len.write().map_err(|e| e.to_string())?;
        *seq_len_guard = 0;

        let mut q_cache = self.q_rope_cache.write().map_err(|e| e.to_string())?;
        *q_cache = None;

        let mut k_cache = self.k_rope_cache.write().map_err(|e| e.to_string())?;
        *k_cache = None;

        let mut q_dim = self.q_rope_dim.write().map_err(|e| e.to_string())?;
        *q_dim = None;

        let mut k_dim = self.k_rope_dim.write().map_err(|e| e.to_string())?;
        *k_dim = None;

        Ok(())
    }

    pub fn memory_usage(&self) -> usize {
        let seq_len_guard = self.seq_len.read().unwrap_or_else(|e| e.into_inner());
        let actual_len = *seq_len_guard;

        let c_kv_mem = actual_len * self.config.latent_dim * 4;

        let q_rope_mem = self.q_rope_cache
            .read()
            .map(|g| {
                g.as_ref().map_or(0, |c| actual_len * c.ncols() * 4)
            })
            .unwrap_or(0);

        let k_rope_mem = self.k_rope_cache
            .read()
            .map(|g| {
                g.as_ref().map_or(0, |c| actual_len * c.ncols() * 4)
            })
            .unwrap_or(0);

        c_kv_mem + q_rope_mem + k_rope_mem
    }

    pub fn max_memory_usage(&self) -> usize {
        let c_kv_mem = self.max_seq_len * self.config.latent_dim * 4;

        let q_rope_mem = self.q_rope_cache
            .read()
            .map(|g| {
                g.as_ref().map_or(0, |c| self.max_seq_len * c.ncols() * 4)
            })
            .unwrap_or(0);

        let k_rope_mem = self.k_rope_cache
            .read()
            .map(|g| {
                g.as_ref().map_or(0, |c| self.max_seq_len * c.ncols() * 4)
            })
            .unwrap_or(0);

        c_kv_mem + q_rope_mem + k_rope_mem
    }

    pub fn compression_ratio(&self) -> f32 {
        let standard_kv = 2 * self.config.kv_dim();
        let compressed_kv = 2 * self.config.latent_dim;
        1.0 - (compressed_kv as f32 / standard_kv as f32)
    }

    pub fn standard_kv_memory(&self, seq_len: usize) -> usize {
        seq_len * self.config.kv_dim() * 2 * 4
    }

    pub fn compressed_kv_memory(&self, seq_len: usize) -> usize {
        seq_len * self.config.latent_dim * 4
    }
}

impl Clone for MLALatentCache {
    fn clone(&self) -> Self {
        let seq_len = *self.seq_len.read().unwrap_or_else(|e| e.into_inner());
        let c_kv = self.c_kv_cache.read().unwrap_or_else(|e| e.into_inner()).clone();
        let q_rope = self.q_rope_cache.read().unwrap_or_else(|e| e.into_inner()).clone();
        let k_rope = self.k_rope_cache.read().unwrap_or_else(|e| e.into_inner()).clone();
        let q_rope_dim = *self.q_rope_dim.read().unwrap_or_else(|e| e.into_inner());
        let k_rope_dim = *self.k_rope_dim.read().unwrap_or_else(|e| e.into_inner());

        Self {
            config: self.config.clone(),
            c_kv_cache: RwLock::new(c_kv),
            q_rope_cache: RwLock::new(q_rope),
            k_rope_cache: RwLock::new(k_rope),
            seq_len: RwLock::new(seq_len),
            max_seq_len: self.max_seq_len,
            q_rope_dim: RwLock::new(q_rope_dim),
            k_rope_dim: RwLock::new(k_rope_dim),
        }
    }
}

impl crate::hardware::kv_cache::KVCache for MLALatentCache {
    fn num_tokens(&self) -> usize {
        self.seq_len().unwrap_or(0)
    }

    fn clear_cache(&mut self) -> Result<(), crate::hardware::kv_cache::KVCacheError> {
        self.clear().map_err(|e| crate::hardware::kv_cache::KVCacheError::ClearError(e))
    }

    fn memory_usage(&self) -> usize {
        MLALatentCache::memory_usage(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> MLAConfig {
        MLAConfig::default().with_latent_dim(512)
    }

    #[test]
    fn test_cache_creation() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        assert_eq!(cache.seq_len().unwrap(), 0);
        assert!(cache.get_all_c_kv().unwrap().nrows() == 0);
    }

    #[test]
    fn test_cache_append() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let c_kv = Array2::zeros((1, 512));
        cache.append(&c_kv, None, None).unwrap();

        assert_eq!(cache.seq_len().unwrap(), 1);
    }

    #[test]
    fn test_cache_get() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let c_kv = Array2::ones((1, 512));
        cache.append(&c_kv, None, None).unwrap();

        let retrieved = cache.get_c_kv(0, 1).unwrap();
        assert_eq!(retrieved.nrows(), 1);
        assert_eq!(retrieved.ncols(), 512);
    }

    #[test]
    fn test_cache_clear() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let c_kv = Array2::ones((1, 512));
        cache.append(&c_kv, None, None).unwrap();
        cache.clear().unwrap();

        assert_eq!(cache.seq_len().unwrap(), 0);
    }

    #[test]
    fn test_memory_usage() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let mem = cache.memory_usage();
        assert_eq!(mem, 0);

        let c_kv = Array2::ones((1, 512));
        cache.append(&c_kv, None, None).unwrap();

        let mem = cache.memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_compression_ratio() {
        let config = MLAConfig::default();
        let cache = MLALatentCache::new(config);

        let ratio = cache.compression_ratio();
        assert!(ratio > 0.0);
        assert!(ratio < 1.0);

        let high_compress_config = MLAConfig::default().with_latent_dim(64);
        let high_compress_cache = MLALatentCache::new(high_compress_config);
        let high_ratio = high_compress_cache.compression_ratio();
        assert!(high_ratio > 0.9);
    }

    #[test]
    fn test_clone_preserves_data() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let c_kv = Array2::from_elem((1, 512), 1.5);
        let q_rope = Array2::from_elem((1, 128), 2.0);
        let k_rope = Array2::from_elem((1, 64), 3.0);

        cache.append(&c_kv, Some(&q_rope), Some(&k_rope)).unwrap();
        cache.append(&c_kv, Some(&q_rope), Some(&k_rope)).unwrap();

        let cloned = cache.clone();

        assert_eq!(cloned.seq_len().unwrap(), 2);

        let original_data = cache.get_c_kv(0, 2).unwrap();
        let cloned_data = cloned.get_c_kv(0, 2).unwrap();

        for i in 0..2 {
            for j in 0..512 {
                assert!((original_data[[i, j]] - cloned_data[[i, j]]).abs() < 1e-6);
            }
        }

        let original_q = cache.get_q_rope(0, 2).unwrap().unwrap();
        let cloned_q = cloned.get_q_rope(0, 2).unwrap().unwrap();

        for i in 0..2 {
            for j in 0..128 {
                assert!((original_q[[i, j]] - cloned_q[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let wrong_dim_c_kv = Array2::zeros((1, 256));
        let result = cache.append(&wrong_dim_c_kv, None, None);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn test_rope_dimension_consistency() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let c_kv = Array2::zeros((1, 512));
        let q_rope_1 = Array2::zeros((1, 128));
        let q_rope_2 = Array2::zeros((1, 64));

        cache.append(&c_kv, Some(&q_rope_1), None).unwrap();

        let result = cache.append(&c_kv, Some(&q_rope_2), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn test_with_capacity() {
        let config = create_test_config();
        let cache = MLALatentCache::with_capacity(100, &config);

        assert_eq!(cache.max_seq_len, 100);
    }

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 append 多行输入错误（覆盖第60-62行 nrows != 1 分支）
    #[test]
    fn test_append_multi_row_error() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        // 覆盖：尝试追加多行数据（仅支持单token追加）
        let multi_row_c_kv = Array2::zeros((3, 512));
        let result = cache.append(&multi_row_c_kv, None, None);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Only single token"));
    }

    /// 测试缓存满错误（覆盖第75-77行 Cache full 分支）
    #[test]
    fn test_append_cache_full_error() {
        let config = MLAConfig {
            hidden_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 32,
            latent_dim: 512,
            use_decoupled_rope: false,
            rope_theta: 10000.0,
            max_seq_len: 2,  // 小的max_seq_len用于测试缓存满
        };
        let cache = MLALatentCache::new(config);

        let c_kv = Array2::zeros((1, 512));

        // 填满缓存
        cache.append(&c_kv, None, None).unwrap();
        cache.append(&c_kv, None, None).unwrap();
        assert_eq!(cache.seq_len().unwrap(), 2);

        // 第三次应失败
        let result = cache.append(&c_kv, None, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("full"));
    }

    /// 测试 get_c_kv 越界错误（覆盖第145-150行 Out of bounds 分支）
    #[test]
    fn test_get_c_kv_out_of_bounds() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let c_kv = Array2::ones((1, 512));
        cache.append(&c_kv, None, None).unwrap();

        // 覆盖：start + len > total_len
        let result = cache.get_c_kv(0, 5);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Out of bounds"));

        // start 超出范围
        let result = cache.get_c_kv(1, 1);
        assert!(result.is_err());
    }

    /// 测试 get_all_c_kv 空缓存（覆盖第167-169行 total_len==0 分支）
    #[test]
    fn test_get_all_c_kv_empty() {
        let config = create_test_config();
        let latent_dim = config.latent_dim;  // 保存需要的字段
        let cache = MLALatentCache::new(config);

        // 覆盖：空缓存返回零行矩阵
        let result = cache.get_all_c_kv().unwrap();
        assert_eq!(result.nrows(), 0);
        assert_eq!(result.ncols(), latent_dim);
    }

    /// 测试 get_q_rope/k_rope 未初始化（覆盖第194-196行 None 分支）
    #[test]
    fn test_get_rope_uninitialized() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let c_kv = Array2::zeros((1, 512));
        cache.append(&c_kv, None, None).unwrap();

        // 覆盖：未初始化 q_rope 时返回 None
        let q_result = cache.get_q_rope(0, 1).unwrap();
        assert!(q_result.is_none());

        // 覆盖：未初始化 k_rope 时返回 None
        let k_result = cache.get_k_rope(0, 1).unwrap();
        assert!(k_result.is_none());
    }

    /// 测试 max_memory_usage 和 compression_ratio（覆盖第271-303行）
    #[test]
    fn test_memory_statistics() {
        // 使用合理的配置：latent_dim < kv_dim 以获得正的压缩比
        let config = MLAConfig {
            hidden_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 64,      // kv_dim = 1*64 = 64
            latent_dim: 16,     // latent_dim=16 < kv_dim=64 -> 压缩比 > 0
            use_decoupled_rope: false,
            rope_theta: 10000.0,
            max_seq_len: 100,
        };
        let cache = MLALatentCache::new(config.clone());

        // max_memory_usage 应基于 max_seq_len 计算
        let max_mem = cache.max_memory_usage();
        assert!(max_mem > 0);

        // compression_ratio 应在 (0, 1) 范围内（因为 latent_dim < kv_dim）
        let ratio = cache.compression_ratio();
        assert!(ratio > 0.0 && ratio <= 1.0, "compression_ratio应在(0,1]范围内，实际: {}", ratio);

        // standard_kv_memory vs compressed_kv_memory
        let seq_len = 10;
        let standard_mem = cache.standard_kv_memory(seq_len);
        let compressed_mem = cache.compressed_kv_memory(seq_len);
        
        assert!(standard_mem > compressed_mem, "压缩后内存应该更小");
        assert!(compressed_mem == seq_len * config.latent_dim * 4);
    }

    /// 测试 KVCache trait 方法（覆盖第328-340行 trait 实现）
    #[test]
    fn test_kv_cache_trait() {
        use crate::hardware::kv_cache::KVCache;

        let config = create_test_config();
        let cache: Box<dyn KVCache> = Box::new(MLALatentCache::new(config.clone()));

        // 覆盖 trait 的 num_tokens
        assert_eq!(cache.num_tokens(), 0);

        // 覆盖 trait 的 memory_usage
        assert_eq!(cache.memory_usage(), 0);
    }

    /// 测试 clear 后重新使用（覆盖 clear 重置所有状态分支）
    #[test]
    fn test_clear_and_reuse() {
        let config = create_test_config();
        let cache = MLALatentCache::new(config);

        let c_kv = Array2::from_elem((1, 512), 42.0);
        let q_rope = Array2::from_elem((1, 64), 1.0);
        let k_rope = Array2::from_elem((1, 32), 2.0);

        // 追加带 RoPE 数据
        cache.append(&c_kv, Some(&q_rope), Some(&k_rope)).unwrap();
        assert_eq!(cache.seq_len().unwrap(), 1);

        // 清除
        cache.clear().unwrap();
        assert_eq!(cache.seq_len().unwrap(), 0);

        // 验证 RoPE 缓存也被清除
        let q_result = cache.get_q_rope(0, 1).unwrap();
        assert!(q_result.is_none(), "clear后RoPE缓存应为None");

        // 重新追加应正常工作
        cache.append(&c_kv, Some(&q_rope), Some(&k_rope)).unwrap();
        assert_eq!(cache.seq_len().unwrap(), 1);
    }
}
