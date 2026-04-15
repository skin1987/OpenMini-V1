//! 分页KV Cache实现
//!
//! 核心KV Cache管理，支持：
//! - 分页内存管理
//! - 多请求并发
//! - 动态内存分配
//! - Copy-on-Write (COW) 支持

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2};

use super::block::{BlockId, KVCacheConfig};
use super::block_manager::BlockManager;
use super::page_table::PageTable;

/// 请求ID类型
pub type RequestId = u64;

/// KV Cache槽位信息
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct KVSlot {
    /// 请求ID
    pub request_id: RequestId,
    /// 起始位置
    pub start_pos: usize,
    /// 当前长度
    pub length: usize,
}

/// 单个块的KV数据
#[derive(Debug, Clone)]
struct BlockKVData {
    /// K数据: [block_size, num_heads * head_dim]
    k: Array1<f32>,
    /// V数据: [block_size, num_heads * head_dim]
    v: Array1<f32>,
}

impl BlockKVData {
    fn new(block_size: usize, kv_dim: usize) -> Self {
        Self {
            k: Array1::zeros(block_size * kv_dim),
            v: Array1::zeros(block_size * kv_dim),
        }
    }
}

/// 分页KV Cache
#[derive(Debug)]
#[allow(dead_code)]
pub struct PagedKVCache {
    /// 内存块管理器
    block_manager: BlockManager,
    /// 请求到页表的映射
    page_tables: HashMap<RequestId, PageTable>,
    /// KV数据存储 [layer][block_id] -> KV数据
    kv_data: Vec<Vec<Option<BlockKVData>>>,
    /// 配置
    config: KVCacheConfig,
    /// 每个token的KV维度 (num_heads * head_dim)
    kv_dim: usize,
    /// 每个块存储的token数
    block_size: usize,
    /// 当前活跃请求数
    num_active_requests: AtomicUsize,
    /// 总token数
    total_tokens: AtomicUsize,
}

#[allow(dead_code)]
impl PagedKVCache {
    /// 创建新的分页KV Cache
    pub fn new(config: KVCacheConfig) -> Self {
        let block_manager = BlockManager::new(&config);
        let num_layers = config.num_layers;
        let num_blocks = config.max_blocks;
        let kv_dim = config.num_heads * config.head_dim;
        let block_size = config.block_size;

        let kv_data: Vec<Vec<Option<BlockKVData>>> = (0..num_layers)
            .map(|_| (0..num_blocks).map(|_| None).collect())
            .collect();

        Self {
            block_manager,
            page_tables: HashMap::new(),
            kv_data,
            config,
            kv_dim,
            block_size,
            num_active_requests: AtomicUsize::new(0),
            total_tokens: AtomicUsize::new(0),
        }
    }

    /// 使用默认配置创建
    pub fn with_capacity(num_blocks: usize, block_size: usize) -> Self {
        let config = KVCacheConfig {
            max_blocks: num_blocks,
            block_size,
            ..Default::default()
        };
        Self::new(config)
    }

    /// 为请求分配槽位
    pub fn allocate_slots(
        &mut self,
        request_id: RequestId,
        num_tokens: usize,
    ) -> Result<(), String> {
        let num_blocks_needed = num_tokens.div_ceil(self.block_size);

        let block_ids = self
            .block_manager
            .allocate(num_blocks_needed, Some(request_id))?;

        let page_table = PageTable::from_blocks(block_ids);
        self.page_tables.insert(request_id, page_table);
        self.num_active_requests.fetch_add(1, Ordering::SeqCst);
        self.total_tokens.fetch_add(num_tokens, Ordering::SeqCst);

        Ok(())
    }

    /// 追加槽位
    pub fn append_slots(&mut self, request_id: RequestId, num_tokens: usize) -> Result<(), String> {
        let num_blocks_needed = num_tokens.div_ceil(self.block_size);

        let block_ids = self
            .block_manager
            .allocate(num_blocks_needed, Some(request_id))?;

        if let Some(page_table) = self.page_tables.get_mut(&request_id) {
            page_table.reserve(num_blocks_needed);
            page_table.append_blocks(block_ids)?;
            self.total_tokens.fetch_add(num_tokens, Ordering::SeqCst);
        } else {
            self.block_manager.free(&block_ids);
            return Err(format!("Request {} not found", request_id));
        }

        Ok(())
    }

    /// 检查并执行COW：如果块的引用计数>1，则复制数据到新块
    fn ensure_cow(
        &mut self,
        request_id: RequestId,
        block_idx: usize,
        block_id: BlockId,
    ) -> Result<BlockId, String> {
        let ref_count = self
            .block_manager
            .get_block(block_id)
            .map(|b| b.ref_count())
            .unwrap_or(1);

        if ref_count <= 1 {
            return Ok(block_id);
        }

        let new_block_id = self
            .block_manager
            .allocate_one(Some(request_id))
            .map_err(|e| format!("Failed to allocate block for COW: {}", e))?;

        for layer in 0..self.config.num_layers {
            if let Some(ref kv_data) = self.kv_data[layer][block_id] {
                self.kv_data[layer][new_block_id] = Some(kv_data.clone());
            }
        }

        self.block_manager.dec_ref(block_id);

        if let Some(page_table) = self.page_tables.get_mut(&request_id) {
            let _ = page_table.set(block_idx, new_block_id);
        }

        Ok(new_block_id)
    }

    /// 写入KV数据
    pub fn write_kv(
        &mut self,
        request_id: RequestId,
        layer: usize,
        start_pos: usize,
        k: &Array2<f32>,
        v: &Array2<f32>,
    ) -> Result<(), String> {
        if !self.page_tables.contains_key(&request_id) {
            return Err(format!("Request {} not found", request_id));
        }

        if layer >= self.config.num_layers {
            return Err(format!("Layer {} out of range", layer));
        }

        let num_tokens = k.nrows();
        let input_dim = k.ncols();

        if v.nrows() != num_tokens || v.ncols() != input_dim {
            return Err(format!(
                "K and V shape mismatch: K is ({}, {}), V is ({}, {})",
                num_tokens,
                input_dim,
                v.nrows(),
                v.ncols()
            ));
        }

        let mut block_mapping: HashMap<usize, BlockId> = HashMap::new();

        let first_block_idx = start_pos / self.block_size;
        let last_block_idx = (start_pos + num_tokens - 1).div_ceil(self.block_size);

        for block_idx in first_block_idx..=last_block_idx {
            let page_table_ref = self.page_tables.get(&request_id).unwrap();
            if let Some(current_block_id) = page_table_ref.get(block_idx) {
                let actual_block_id = self.ensure_cow(request_id, block_idx, current_block_id)?;
                block_mapping.insert(block_idx, actual_block_id);
            }
        }

        for token_idx in 0..num_tokens {
            let global_pos = start_pos + token_idx;
            let block_idx = global_pos / self.block_size;
            let local_pos = global_pos % self.block_size;

            let actual_block_id = *block_mapping.get(&block_idx).ok_or_else(|| {
                format!("Block {} not found for request {}", block_idx, request_id)
            })?;

            if self.kv_data[layer][actual_block_id].is_none() {
                self.kv_data[layer][actual_block_id] =
                    Some(BlockKVData::new(self.block_size, self.kv_dim));
            }

            if let Some(ref mut kv_data) = self.kv_data[layer][actual_block_id] {
                let offset = local_pos * self.kv_dim;

                let copy_dim = input_dim.min(self.kv_dim);

                for d in 0..copy_dim {
                    kv_data.k[offset + d] = k[[token_idx, d]];
                    kv_data.v[offset + d] = v[[token_idx, d]];
                }

                for d in copy_dim..self.kv_dim {
                    kv_data.k[offset + d] = 0.0;
                    kv_data.v[offset + d] = 0.0;
                }
            }
        }

        Ok(())
    }

    /// 读取KV数据
    pub fn read_kv(
        &self,
        request_id: RequestId,
        layer: usize,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let page_table = self.page_tables.get(&request_id)?;

        if layer >= self.config.num_layers {
            return None;
        }

        let num_blocks = page_table.len();
        let total_tokens = num_blocks * self.block_size;

        let mut k = Array2::zeros((total_tokens, self.kv_dim));
        let mut v = Array2::zeros((total_tokens, self.kv_dim));

        for block_idx in 0..num_blocks {
            if let Some(block_id) = page_table.get(block_idx) {
                if let Some(ref kv_data) = self.kv_data[layer][block_id] {
                    let start_token = block_idx * self.block_size;

                    for local_pos in 0..self.block_size {
                        let token_pos = start_token + local_pos;
                        let offset = local_pos * self.kv_dim;

                        for d in 0..self.kv_dim {
                            k[[token_pos, d]] = kv_data.k[offset + d];
                            v[[token_pos, d]] = kv_data.v[offset + d];
                        }
                    }
                }
            }
        }

        Some((k, v))
    }

    /// 读取指定范围的KV数据
    pub fn read_kv_range(
        &self,
        request_id: RequestId,
        layer: usize,
        start: usize,
        len: usize,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let page_table = self.page_tables.get(&request_id)?;

        if layer >= self.config.num_layers {
            return None;
        }

        let max_tokens = page_table.len() * self.block_size;
        if start >= max_tokens {
            return None;
        }

        let actual_len = len.min(max_tokens - start);

        let mut k = Array2::zeros((actual_len, self.kv_dim));
        let mut v = Array2::zeros((actual_len, self.kv_dim));

        for i in 0..actual_len {
            let global_pos = start + i;
            let block_idx = global_pos / self.block_size;
            let local_pos = global_pos % self.block_size;

            if let Some(block_id) = page_table.get(block_idx) {
                if let Some(ref kv_data) = self.kv_data[layer][block_id] {
                    let offset = local_pos * self.kv_dim;

                    for d in 0..self.kv_dim {
                        k[[i, d]] = kv_data.k[offset + d];
                        v[[i, d]] = kv_data.v[offset + d];
                    }
                }
            }
        }

        Some((k, v))
    }

    /// 释放请求的所有资源
    pub fn free_request(&mut self, request_id: &RequestId) {
        if let Some(page_table) = self.page_tables.remove(request_id) {
            let blocks = page_table.get_all_blocks();
            let num_tokens = blocks.len() * self.block_size;

            self.block_manager.free(&blocks);
            self.num_active_requests.fetch_sub(1, Ordering::SeqCst);
            self.total_tokens.fetch_sub(num_tokens, Ordering::SeqCst);
        }
    }

    /// 获取请求的序列长度
    pub fn get_seq_len(&self, request_id: RequestId) -> Option<usize> {
        self.page_tables
            .get(&request_id)
            .map(|pt| pt.len() * self.block_size)
    }

    /// 获取请求的块数
    pub fn get_num_blocks(&self, request_id: RequestId) -> Option<usize> {
        self.page_tables.get(&request_id).map(|pt| pt.len())
    }

    /// 检查请求是否存在
    pub fn contains_request(&self, request_id: RequestId) -> bool {
        self.page_tables.contains_key(&request_id)
    }

    /// 获取活跃请求数
    pub fn num_active_requests(&self) -> usize {
        self.num_active_requests.load(Ordering::SeqCst)
    }

    /// 获取总token数
    pub fn total_tokens(&self) -> usize {
        self.total_tokens.load(Ordering::SeqCst)
    }

    /// 获取可用块数
    pub fn available_blocks(&self) -> usize {
        self.block_manager.available_blocks()
    }

    /// 获取已分配块数
    pub fn allocated_blocks(&self) -> usize {
        self.block_manager.allocated_blocks()
    }

    /// 获取内存使用率
    pub fn utilization(&self) -> f32 {
        self.block_manager.utilization()
    }

    /// 获取配置
    pub fn config(&self) -> &KVCacheConfig {
        &self.config
    }

    /// 获取所有请求ID
    pub fn request_ids(&self) -> Vec<RequestId> {
        self.page_tables.keys().copied().collect()
    }

    /// 清空所有请求
    pub fn clear(&mut self) {
        for request_id in self.request_ids() {
            self.free_request(&request_id);
        }
    }

    /// Fork请求（Copy-on-Write）
    pub fn fork_request(&mut self, source_id: RequestId, new_id: RequestId) -> Result<(), String> {
        let source_pt = self
            .page_tables
            .get(&source_id)
            .ok_or_else(|| format!("Source request {} not found", source_id))?;

        let blocks = source_pt.get_all_blocks();
        let new_blocks = self.block_manager.fork(&blocks, Some(new_id))?;

        let new_page_table = PageTable::from_blocks(new_blocks);
        self.page_tables.insert(new_id, new_page_table);
        self.num_active_requests.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    /// 获取块管理器引用
    pub fn block_manager(&self) -> &BlockManager {
        &self.block_manager
    }

    /// 获取块管理器可变引用
    pub fn block_manager_mut(&mut self) -> &mut BlockManager {
        &mut self.block_manager
    }
}

impl Clone for PagedKVCache {
    fn clone(&self) -> Self {
        Self {
            block_manager: self.block_manager.clone(),
            page_tables: self.page_tables.clone(),
            kv_data: self.kv_data.clone(),
            config: self.config.clone(),
            kv_dim: self.kv_dim,
            block_size: self.block_size,
            num_active_requests: AtomicUsize::new(self.num_active_requests.load(Ordering::SeqCst)),
            total_tokens: AtomicUsize::new(self.total_tokens.load(Ordering::SeqCst)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cache() -> PagedKVCache {
        PagedKVCache::with_capacity(100, 16)
    }

    #[test]
    fn test_new_cache() {
        let cache = create_test_cache();
        assert_eq!(cache.num_active_requests(), 0);
        assert_eq!(cache.available_blocks(), 100);
        assert_eq!(cache.allocated_blocks(), 0);
    }

    #[test]
    fn test_allocate_slots() {
        let mut cache = create_test_cache();

        cache.allocate_slots(1, 32).unwrap();

        assert_eq!(cache.num_active_requests(), 1);
        assert_eq!(cache.allocated_blocks(), 2);
        assert!(cache.contains_request(1));
        assert_eq!(cache.get_num_blocks(1), Some(2));
    }

    #[test]
    fn test_free_request() {
        let mut cache = create_test_cache();

        cache.allocate_slots(1, 32).unwrap();
        cache.free_request(&1);

        assert_eq!(cache.num_active_requests(), 0);
        assert_eq!(cache.allocated_blocks(), 0);
        assert!(!cache.contains_request(1));
    }

    #[test]
    fn test_write_read_kv() {
        let mut cache = PagedKVCache::with_capacity(100, 16);

        cache.allocate_slots(1, 16).unwrap();

        let k = Array2::from_shape_fn((16, 128), |(i, j)| (i + j) as f32);
        let v = Array2::from_shape_fn((16, 128), |(i, j)| (i * 2 + j) as f32);

        cache.write_kv(1, 0, 0, &k, &v).unwrap();

        let (read_k, read_v) = cache.read_kv(1, 0).unwrap();

        assert_eq!(read_k.nrows(), 16);
        assert_eq!(read_v.nrows(), 16);

        for i in 0..16 {
            for j in 0..128 {
                assert!(
                    (read_k[[i, j]] - k[[i, j]]).abs() < 1e-6,
                    "K mismatch at ({}, {}): expected {}, got {}",
                    i,
                    j,
                    k[[i, j]],
                    read_k[[i, j]]
                );
                assert!(
                    (read_v[[i, j]] - v[[i, j]]).abs() < 1e-6,
                    "V mismatch at ({}, {}): expected {}, got {}",
                    i,
                    j,
                    v[[i, j]],
                    read_v[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_write_read_kv_different_values() {
        let mut cache = PagedKVCache::with_capacity(100, 16);

        cache.allocate_slots(1, 16).unwrap();

        let k = Array2::from_shape_fn((16, 128), |(i, j)| (i + j) as f32);
        let v = Array2::from_shape_fn((16, 128), |(i, j)| (i + j + 1000) as f32);

        cache.write_kv(1, 0, 0, &k, &v).unwrap();

        let (read_k, read_v) = cache.read_kv(1, 0).unwrap();

        for i in 0..16 {
            for j in 0..128 {
                assert!((read_k[[i, j]] - k[[i, j]]).abs() < 1e-6);
                assert!((read_v[[i, j]] - v[[i, j]]).abs() < 1e-6);
                assert_ne!(
                    read_k[[i, j]],
                    read_v[[i, j]],
                    "K and V should be different!"
                );
            }
        }
    }

    #[test]
    fn test_multiple_requests() {
        let mut cache = create_test_cache();

        cache.allocate_slots(1, 16).unwrap();
        cache.allocate_slots(2, 32).unwrap();
        cache.allocate_slots(3, 48).unwrap();

        assert_eq!(cache.num_active_requests(), 3);
        assert_eq!(cache.allocated_blocks(), 6);

        cache.free_request(&2);

        assert_eq!(cache.num_active_requests(), 2);
        assert_eq!(cache.allocated_blocks(), 4);
    }

    #[test]
    fn test_append_slots() {
        let mut cache = create_test_cache();

        cache.allocate_slots(1, 16).unwrap();
        assert_eq!(cache.get_num_blocks(1), Some(1));

        cache.append_slots(1, 32).unwrap();
        assert_eq!(cache.get_num_blocks(1), Some(3));
    }

    #[test]
    fn test_utilization() {
        let mut cache = create_test_cache();

        assert!((cache.utilization() - 0.0).abs() < 0.001);

        cache.allocate_slots(1, 800).unwrap();
        assert!((cache.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_fork_request() {
        let mut cache = create_test_cache();

        cache.allocate_slots(1, 32).unwrap();
        cache.fork_request(1, 2).unwrap();

        assert_eq!(cache.num_active_requests(), 2);
        assert!(cache.contains_request(2));
    }

    #[test]
    fn test_allocate_insufficient() {
        let mut cache = PagedKVCache::with_capacity(10, 16);

        let result = cache.allocate_slots(1, 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_cow_on_write() {
        let mut cache = PagedKVCache::with_capacity(100, 16);

        cache.allocate_slots(1, 16).unwrap();

        let k1 = Array2::from_shape_fn((16, 128), |(i, j)| (i + j) as f32);
        let v1 = Array2::from_shape_fn((16, 128), |(i, j)| (i + j + 100) as f32);
        cache.write_kv(1, 0, 0, &k1, &v1).unwrap();

        cache.fork_request(1, 2).unwrap();

        let k2 = Array2::from_shape_fn((16, 128), |(i, j)| (i * 2 + j) as f32);
        let v2 = Array2::from_shape_fn((16, 128), |(i, j)| (i * 2 + j + 200) as f32);
        cache.write_kv(2, 0, 0, &k2, &v2).unwrap();

        let (read_k1, read_v1) = cache.read_kv(1, 0).unwrap();
        let (read_k2, read_v2) = cache.read_kv(2, 0).unwrap();

        for i in 0..16 {
            for j in 0..128 {
                assert!(
                    (read_k1[[i, j]] - k1[[i, j]]).abs() < 1e-6,
                    "Original request K should not be modified! Expected {}, got {}",
                    k1[[i, j]],
                    read_k1[[i, j]]
                );
                assert!(
                    (read_v1[[i, j]] - v1[[i, j]]).abs() < 1e-6,
                    "Original request V should not be modified! Expected {}, got {}",
                    v1[[i, j]],
                    read_v1[[i, j]]
                );
                assert!(
                    (read_k2[[i, j]] - k2[[i, j]]).abs() < 1e-6,
                    "Forked request K should have new value! Expected {}, got {}",
                    k2[[i, j]],
                    read_k2[[i, j]]
                );
                assert!(
                    (read_v2[[i, j]] - v2[[i, j]]).abs() < 1e-6,
                    "Forked request V should have new value! Expected {}, got {}",
                    v2[[i, j]],
                    read_v2[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_read_kv_range() {
        let mut cache = PagedKVCache::with_capacity(100, 16);

        cache.allocate_slots(1, 32).unwrap();

        let k = Array2::from_shape_fn((32, 128), |(i, j)| (i + j) as f32);
        let v = Array2::from_shape_fn((32, 128), |(i, j)| (i + j + 1000) as f32);
        cache.write_kv(1, 0, 0, &k, &v).unwrap();

        let (read_k, read_v) = cache.read_kv_range(1, 0, 8, 16).unwrap();

        assert_eq!(read_k.nrows(), 16);
        assert_eq!(read_v.nrows(), 16);

        for i in 0..16 {
            for j in 0..128 {
                assert!((read_k[[i, j]] - k[[i + 8, j]]).abs() < 1e-6);
                assert!((read_v[[i, j]] - v[[i + 8, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_read_kv_range_out_of_bounds() {
        let mut cache = PagedKVCache::with_capacity(100, 16);

        cache.allocate_slots(1, 16).unwrap();

        let k = Array2::ones((16, 128));
        let v = Array2::ones((16, 128)) * 2.0;
        cache.write_kv(1, 0, 0, &k, &v).unwrap();

        let result = cache.read_kv_range(1, 0, 100, 10);
        assert!(
            result.is_none(),
            "Should return None for out of bounds start"
        );

        let (read_k, read_v) = cache.read_kv_range(1, 0, 8, 100).unwrap();
        assert_eq!(read_k.nrows(), 8, "Should clamp to available tokens");
        assert_eq!(read_v.nrows(), 8);
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_paged_kv_cache_clear_all_requests() {
        // 测试清空所有请求
        let mut cache = PagedKVCache::with_capacity(100, 16);

        // 分配多个请求
        for i in 1..=5 {
            cache.allocate_slots(i, 32).unwrap();
        }

        assert_eq!(cache.num_active_requests(), 5);

        // 清空所有请求
        cache.clear();

        // 验证所有请求已释放
        assert_eq!(cache.num_active_requests(), 0);
        assert_eq!(cache.allocated_blocks(), 0);
        assert_eq!(cache.available_blocks(), 100);
        assert!(cache.request_ids().is_empty());
    }

    #[test]
    fn test_paged_kv_cache_request_ids() {
        // 测试获取所有请求ID
        let mut cache = PagedKVCache::with_capacity(100, 16);

        // 初始为空
        assert!(cache.request_ids().is_empty());

        // 添加请求
        cache.allocate_slots(10, 16).unwrap();
        cache.allocate_slots(20, 32).unwrap();
        cache.allocate_slots(30, 48).unwrap();

        let ids = cache.request_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&10));
        assert!(ids.contains(&20));
        assert!(ids.contains(&30));
    }

    #[test]
    fn test_paged_kv_cache_get_seq_len() {
        // 测试获取请求的序列长度
        let mut cache = PagedKVCache::with_capacity(100, 16);

        // 未分配的请求返回None
        assert!(cache.get_seq_len(999).is_none());

        // 分配16个token（1个块）
        cache.allocate_slots(1, 16).unwrap();
        assert_eq!(cache.get_seq_len(1), Some(16));

        // 追加32个token（2个块）
        cache.append_slots(1, 32).unwrap();
        assert_eq!(cache.get_seq_len(1), Some(48)); // 16 + 32 = 48 tokens (3 blocks)
    }

    #[test]
    fn test_paged_kv_cache_contains_request() {
        // 测试检查请求是否存在
        let mut cache = PagedKVCache::with_capacity(100, 16);

        assert!(!cache.contains_request(1));

        cache.allocate_slots(1, 16).unwrap();
        assert!(cache.contains_request(1));
        assert!(!cache.contains_request(2));

        cache.free_request(&1);
        assert!(!cache.contains_request(1));
    }

    #[test]
    fn test_paged_kv_cache_total_tokens_tracking() {
        // 测试总token数跟踪
        let mut cache = PagedKVCache::with_capacity(100, 16);

        assert_eq!(cache.total_tokens(), 0);

        cache.allocate_slots(1, 48).unwrap(); // 3 blocks, 48 tokens
        assert_eq!(cache.total_tokens(), 48);

        cache.allocate_slots(2, 32).unwrap(); // 2 blocks, 32 tokens
        assert_eq!(cache.total_tokens(), 80); // 48 + 32

        cache.free_request(&1);
        assert_eq!(cache.total_tokens(), 32); // 只剩请求2

        cache.free_request(&2);
        assert_eq!(cache.total_tokens(), 0);
    }

    #[test]
    fn test_paged_kv_cache_config_access() {
        // 测试配置访问接口
        let cache = PagedKVCache::with_capacity(50, 16);

        let config = cache.config();
        assert_eq!(config.max_blocks, 50);
        assert_eq!(config.block_size, 16);
    }

    #[test]
    fn test_paged_kv_cache_block_manager_access() {
        // 测试块管理器访问接口
        let mut cache = PagedKVCache::with_capacity(100, 16);

        // 通过block_manager访问
        assert_eq!(cache.block_manager().total_blocks(), 100);
        assert_eq!(cache.block_manager().available_blocks(), 100);

        // 通过可变引用修改
        cache.block_manager_mut().allocate(5, None).unwrap();
        assert_eq!(cache.block_manager().available_blocks(), 95);
    }

    #[test]
    fn test_paged_kv_cache_append_nonexistent_request() {
        // 测试向不存在的请求追加槽位
        let mut cache = PagedKVCache::with_capacity(100, 16);

        let result = cache.append_slots(999, 16);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_paged_kv_cache_write_to_nonexistent_request() {
        // 测试向不存在的请求写入数据
        let mut cache = PagedKVCache::with_capacity(100, 16);

        let k = Array2::zeros((16, 128));
        let v = Array2::zeros((16, 128));
        let result = cache.write_kv(999, 0, 0, &k, &v);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_paged_kv_cache_write_layer_out_of_range() {
        // 测试写入超出范围的层
        let mut cache = PagedKVCache::with_capacity(100, 16);

        cache.allocate_slots(1, 16).unwrap();

        let k = Array2::zeros((16, 128));
        let v = Array2::zeros((16, 128));
        let result = cache.write_kv(1, 999, 0, &k, &v); // 默认只有1层
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of range"));
    }

    #[test]
    fn test_paged_kv_cache_read_nonexistent_request() {
        // 测试读取不存在的请求数据
        let cache = PagedKVCache::with_capacity(100, 16);

        let result = cache.read_kv(999, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_paged_kv_cache_clone() {
        // 测试克隆功能
        let mut cache = PagedKVCache::with_capacity(100, 16);

        cache.allocate_slots(1, 16).unwrap();

        let k = Array2::from_shape_fn((16, 128), |(i, j)| (i + j) as f32);
        let v = Array2::from_shape_fn((16, 128), |(i, j)| (i * 2 + j) as f32);
        cache.write_kv(1, 0, 0, &k, &v).unwrap();

        // 克隆
        let cloned = cache.clone();

        // 验证克隆后的数据一致
        assert_eq!(cloned.num_active_requests(), 1);
        assert_eq!(cloned.available_blocks(), cache.available_blocks());

        let (read_k, read_v) = cloned.read_kv(1, 0).unwrap();
        for i in 0..16 {
            for j in 0..128 {
                assert!((read_k[[i, j]] - k[[i, j]]).abs() < 1e-6);
                assert!((read_v[[i, j]] - v[[i, j]]).abs() < 1e-6);
            }
        }
    }
}
