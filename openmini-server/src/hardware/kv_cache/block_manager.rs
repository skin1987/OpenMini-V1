//! 内存块管理器
//!
//! 提供KV Cache的内存块分配、释放和管理功能。
//!
//! ## 设计原理
//!
//! 借鉴vLLM的PagedAttention设计：
//! - 内存按固定大小的Block组织
//! - 支持Copy-on-Write实现前缀共享
//! - 引用计数管理Block生命周期
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           BlockManager                  │
//! │  ┌─────┬─────┬─────┬─────┬─────┐      │
//! │  │ B0  │ B1  │ B2  │ B3  │ B4  │ ...  │
//! │  │Free │Used │Used │Free │Used │      │
//! │  └─────┴─────┴─────┴─────┴─────┘      │
//! │                                        │
//! │  Free List: [B0, B3, ...]             │
//! │  Used Blocks: {B1: req1, B2: req1, B4: req2} │
//! └─────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::block::{Block, BlockId, BlockState, KVCacheConfig};

/// 内存块管理器
#[derive(Debug)]
#[allow(dead_code)]
pub struct BlockManager {
    /// 所有内存块
    blocks: Vec<Block>,
    /// 空闲块列表
    free_blocks: VecDeque<BlockId>,
    /// 块大小（token数）
    block_size: usize,
    /// 最大块数
    max_blocks: usize,
    /// 已分配块数
    num_allocated: AtomicUsize,
    /// 请求到块的映射
    request_blocks: HashMap<u64, Vec<BlockId>>,
    /// 下一个请求ID
    next_request_id: AtomicUsize,
}

#[allow(dead_code)]
impl BlockManager {
    /// 创建新的内存块管理器
    pub fn new(config: &KVCacheConfig) -> Self {
        let max_blocks = config.max_blocks;
        let mut blocks = Vec::with_capacity(max_blocks);
        let mut free_blocks = VecDeque::with_capacity(max_blocks);
        
        let block_memory = config.block_memory_size();
        
        for i in 0..max_blocks {
            blocks.push(Block::new(i, i * block_memory));
            free_blocks.push_back(i);
        }
        
        Self {
            blocks,
            free_blocks,
            block_size: config.block_size,
            max_blocks,
            num_allocated: AtomicUsize::new(0),
            request_blocks: HashMap::new(),
            next_request_id: AtomicUsize::new(1),
        }
    }

    /// 创建指定大小的管理器
    pub fn with_capacity(num_blocks: usize, block_size: usize) -> Self {
        let config = KVCacheConfig {
            max_blocks: num_blocks,
            block_size,
            ..Default::default()
        };
        Self::new(&config)
    }

    /// 分配指定数量的块
    pub fn allocate(&mut self, num_blocks: usize, owner_id: Option<u64>) -> Result<Vec<BlockId>, String> {
        if self.free_blocks.len() < num_blocks {
            return Err(format!(
                "Not enough free blocks: requested {}, available {}",
                num_blocks,
                self.free_blocks.len()
            ));
        }
        
        let mut allocated = Vec::with_capacity(num_blocks);
        
        for _ in 0..num_blocks {
            if let Some(block_id) = self.free_blocks.pop_front() {
                self.blocks[block_id].allocate(owner_id);
                allocated.push(block_id);
                self.num_allocated.fetch_add(1, Ordering::SeqCst);
            }
        }
        
        if let Some(id) = owner_id {
            self.request_blocks.entry(id).or_default().extend(allocated.iter().cloned());
        }
        
        Ok(allocated)
    }

    /// 分配单个块
    pub fn allocate_one(&mut self, owner_id: Option<u64>) -> Result<BlockId, String> {
        let blocks = self.allocate(1, owner_id)?;
        Ok(blocks[0])
    }

    /// 释放指定的块
    ///
    /// 注意：此方法会减少引用计数，仅当引用计数归零时才真正回收块。
    /// 对于共享块（通过 fork 创建），只有当所有引用都释放后才会回收。
    pub fn free(&mut self, block_ids: &[BlockId]) {
        for &block_id in block_ids {
            self.dec_ref(block_id);
        }
    }

    /// 释放请求的所有块
    ///
    /// 注意：此方法会减少每个块的引用计数，仅当引用计数归零时才真正回收块。
    pub fn free_request(&mut self, request_id: u64) {
        if let Some(block_ids) = self.request_blocks.remove(&request_id) {
            for block_id in block_ids {
                self.dec_ref(block_id);
            }
        }
    }

    /// 复制块（Copy-on-Write）
    /// 返回新分配的块ID列表
    pub fn fork(&mut self, block_ids: &[BlockId], owner_id: Option<u64>) -> Result<Vec<BlockId>, String> {
        let num_blocks = block_ids.len();
        
        if self.free_blocks.len() < num_blocks {
            return Err(format!(
                "Not enough free blocks for fork: requested {}, available {}",
                num_blocks,
                self.free_blocks.len()
            ));
        }
        
        let mut new_blocks = Vec::with_capacity(num_blocks);
        
        for &block_id in block_ids {
            if block_id >= self.blocks.len() {
                return Err(format!("Invalid block id: {}", block_id));
            }
            
            if let Some(new_block_id) = self.free_blocks.pop_front() {
                let src_offset = self.blocks[block_id].physical_offset;
                self.blocks[new_block_id].allocate(owner_id);
                self.blocks[new_block_id].physical_offset = src_offset;
                new_blocks.push(new_block_id);
                self.num_allocated.fetch_add(1, Ordering::SeqCst);
            }
        }
        
        for &block_id in block_ids {
            self.blocks[block_id].inc_ref();
        }
        
        if let Some(id) = owner_id {
            self.request_blocks.entry(id).or_default().extend(new_blocks.iter().cloned());
        }
        
        Ok(new_blocks)
    }

    /// 增加块的引用计数
    pub fn inc_ref(&self, block_id: BlockId) -> usize {
        if block_id < self.blocks.len() {
            self.blocks[block_id].inc_ref()
        } else {
            0
        }
    }

    /// 减少块的引用计数
    pub fn dec_ref(&mut self, block_id: BlockId) -> usize {
        if block_id < self.blocks.len() {
            let count = self.blocks[block_id].dec_ref();
            if count == 0 && self.blocks[block_id].state != BlockState::Free {
                self.blocks[block_id].free();
                self.free_blocks.push_back(block_id);
                self.num_allocated.fetch_sub(1, Ordering::SeqCst);
            }
            count
        } else {
            0
        }
    }

    /// 获取可用块数量
    pub fn available_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// 获取已分配块数量
    pub fn allocated_blocks(&self) -> usize {
        self.num_allocated.load(Ordering::SeqCst)
    }

    /// 获取总块数
    pub fn total_blocks(&self) -> usize {
        self.max_blocks
    }

    /// 获取块大小
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// 检查是否可以分配指定数量的块
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.free_blocks.len() >= num_blocks
    }

    /// 获取内存使用率
    pub fn utilization(&self) -> f32 {
        if self.max_blocks == 0 {
            return 0.0;
        }
        self.num_allocated.load(Ordering::SeqCst) as f32 / self.max_blocks as f32
    }

    /// 生成新的请求ID
    pub fn new_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::SeqCst) as u64
    }

    /// 获取块信息
    pub fn get_block(&self, block_id: BlockId) -> Option<&Block> {
        self.blocks.get(block_id)
    }

    /// 获取请求的块列表
    pub fn get_request_blocks(&self, request_id: u64) -> Option<&Vec<BlockId>> {
        self.request_blocks.get(&request_id)
    }

    /// 重置管理器
    pub fn reset(&mut self) {
        for block in &mut self.blocks {
            block.free();
        }
        self.free_blocks.clear();
        for i in 0..self.max_blocks {
            self.free_blocks.push_back(i);
        }
        self.num_allocated.store(0, Ordering::SeqCst);
        self.request_blocks.clear();
    }
}

impl Clone for BlockManager {
    fn clone(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            free_blocks: self.free_blocks.clone(),
            block_size: self.block_size,
            max_blocks: self.max_blocks,
            num_allocated: AtomicUsize::new(self.num_allocated.load(Ordering::SeqCst)),
            request_blocks: self.request_blocks.clone(),
            next_request_id: AtomicUsize::new(self.next_request_id.load(Ordering::SeqCst)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_manager() -> BlockManager {
        BlockManager::with_capacity(100, 16)
    }

    #[test]
    fn test_new_manager() {
        let manager = create_test_manager();
        assert_eq!(manager.total_blocks(), 100);
        assert_eq!(manager.available_blocks(), 100);
        assert_eq!(manager.allocated_blocks(), 0);
    }

    #[test]
    fn test_allocate_one() {
        let mut manager = create_test_manager();
        let block_id = manager.allocate_one(None).unwrap();
        
        assert_eq!(manager.allocated_blocks(), 1);
        assert_eq!(manager.available_blocks(), 99);
        assert!(!manager.get_block(block_id).unwrap().is_free());
    }

    #[test]
    fn test_allocate_multiple() {
        let mut manager = create_test_manager();
        let blocks = manager.allocate(10, None).unwrap();
        
        assert_eq!(blocks.len(), 10);
        assert_eq!(manager.allocated_blocks(), 10);
        assert_eq!(manager.available_blocks(), 90);
    }

    #[test]
    fn test_free() {
        let mut manager = create_test_manager();
        let blocks = manager.allocate(5, None).unwrap();
        
        manager.free(&blocks);
        
        assert_eq!(manager.allocated_blocks(), 0);
        assert_eq!(manager.available_blocks(), 100);
    }

    #[test]
    fn test_allocate_insufficient() {
        let mut manager = create_test_manager();
        let result = manager.allocate(200, None);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_fork() {
        let mut manager = create_test_manager();
        let original = manager.allocate(3, None).unwrap();
        
        let forked = manager.fork(&original, None).unwrap();
        
        assert_eq!(forked.len(), 3);
        assert_eq!(manager.allocated_blocks(), 6);
        
        for &block_id in &original {
            assert_eq!(manager.get_block(block_id).unwrap().ref_count(), 2);
        }
    }

    #[test]
    fn test_ref_count() {
        let mut manager = create_test_manager();
        let block_id = manager.allocate_one(None).unwrap();
        
        assert_eq!(manager.get_block(block_id).unwrap().ref_count(), 1);
        
        manager.inc_ref(block_id);
        assert_eq!(manager.get_block(block_id).unwrap().ref_count(), 2);
        
        manager.dec_ref(block_id);
        assert_eq!(manager.get_block(block_id).unwrap().ref_count(), 1);
    }

    #[test]
    fn test_utilization() {
        let mut manager = create_test_manager();
        
        assert!((manager.utilization() - 0.0).abs() < 0.001);
        
        manager.allocate(50, None).unwrap();
        assert!((manager.utilization() - 0.5).abs() < 0.001);
        
        manager.allocate(50, None).unwrap();
        assert!((manager.utilization() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_request_tracking() {
        let mut manager = create_test_manager();
        let request_id = manager.new_request_id();
        
        let _blocks = manager.allocate(5, Some(request_id)).unwrap();
        
        assert!(manager.get_request_blocks(request_id).is_some());
        assert_eq!(manager.get_request_blocks(request_id).unwrap().len(), 5);
        
        manager.free_request(request_id);
        
        assert!(manager.get_request_blocks(request_id).is_none());
        assert_eq!(manager.allocated_blocks(), 0);
    }

    #[test]
    fn test_reset() {
        let mut manager = create_test_manager();
        manager.allocate(50, None).unwrap();

        manager.reset();

        assert_eq!(manager.allocated_blocks(), 0);
        assert_eq!(manager.available_blocks(), 100);
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_block_manager_block_size() {
        // 测试块大小访问
        let manager = BlockManager::with_capacity(100, 32);
        assert_eq!(manager.block_size(), 32);

        let manager2 = BlockManager::with_capacity(50, 64);
        assert_eq!(manager2.block_size(), 64);
    }

    #[test]
    fn test_block_manager_can_allocate() {
        // 测试预检查分配能力
        let mut manager = create_test_manager();

        assert!(manager.can_allocate(100));
        assert!(!manager.can_allocate(101));

        // 分配一些块后
        manager.allocate(30, None).unwrap();
        assert!(manager.can_allocate(70));
        assert!(!manager.can_allocate(71));
    }

    #[test]
    fn test_block_manager_get_block_out_of_range() {
        // 测试获取越界块
        let manager = create_test_manager();

        assert!(manager.get_block(999).is_none());
        assert!(manager.get_block(100).is_none()); // 刚好越界
    }

    #[test]
    fn test_block_manager_inc_ref_out_of_range() {
        // 测试对越界块增加引用计数
        let manager = create_test_manager();

        let count = manager.inc_ref(999);
        assert_eq!(count, 0); // 越界返回0
    }

    #[test]
    fn test_block_manager_dec_ref_out_of_range() {
        // 测试对越界块减少引用计数
        let mut manager = create_test_manager();

        let count = manager.dec_ref(999);
        assert_eq!(count, 0); // 越界返回0
    }

    #[test]
    fn test_block_manager_free_request_nonexistent() {
        // 测试释放不存在的请求（不应panic）
        let mut manager = create_test_manager();

        // 释放不存在的请求应该安全地什么都不做
        manager.free_request(99999);
        assert_eq!(manager.allocated_blocks(), 0);
        assert_eq!(manager.available_blocks(), 100);
    }

    #[test]
    fn test_block_manager_new_request_id_increments() {
        // 测试请求ID生成器递增
        let manager = create_test_manager();

        let id1 = manager.new_request_id();
        let id2 = manager.new_request_id();
        let id3 = manager.new_request_id();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert!(id3 > id2);
        assert!(id2 > id1);
    }

    #[test]
    fn test_block_manager_utilization_zero_blocks() {
        // 测试零块时的使用率
        let manager = BlockManager::with_capacity(0, 16);
        
        assert!((manager.utilization() - 0.0).abs() < 0.001);
        assert_eq!(manager.total_blocks(), 0);
        assert_eq!(manager.available_blocks(), 0);
    }

    #[test]
    fn test_block_manager_fork_insufficient_blocks() {
        // 测试fork时块不足
        let mut manager = BlockManager::with_capacity(5, 16);

        let original = manager.allocate(3, None).unwrap();

        // 尝试fork需要3个新块，但只剩2个
        let result = manager.fork(&original, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Not enough free blocks"));
    }

    #[test]
    fn test_block_manager_fork_invalid_block_id() {
        // 测试使用无效block ID进行fork
        let mut manager = create_test_manager();

        let invalid_ids = vec![999, 1000];
        let result = manager.fork(&invalid_ids, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid block id"));
    }

    #[test]
    fn test_block_manager_clone() {
        // 测试克隆功能
        let mut manager1 = create_test_manager();

        manager1.allocate(10, Some(1)).unwrap();
        manager1.allocate(20, Some(2)).unwrap();

        let manager2 = manager1.clone();

        // 验证克隆后的状态一致
        assert_eq!(manager2.total_blocks(), 100);
        assert_eq!(manager2.allocated_blocks(), 30);
        assert_eq!(manager2.available_blocks(), 70);
    }
}
