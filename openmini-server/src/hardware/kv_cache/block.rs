//! KV Cache 内存块定义
//!
//! 实现分页KV Cache的基础内存块单元：
//! - BlockId: 内存块唯一标识
//! - Block: 内存块结构，支持引用计数
//! - Copy-on-Write语义

use std::sync::atomic::{AtomicUsize, Ordering};

/// 内存块ID类型
pub type BlockId = usize;

/// 默认块大小（每个块存储的token数量）
#[allow(dead_code)]
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// 最大块数量
#[allow(dead_code)]
pub const MAX_BLOCKS: usize = 1 << 24;

/// 内存块状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BlockState {
    /// 空闲状态
    Free,
    /// 已分配状态
    Allocated,
    /// 已换出状态
    Swapped,
}

/// 内存块结构
#[derive(Debug)]
#[allow(dead_code)]
pub struct Block {
    /// 块ID
    pub block_id: BlockId,
    /// 物理内存偏移
    pub physical_offset: usize,
    /// 引用计数
    pub ref_count: AtomicUsize,
    /// 块状态
    pub state: BlockState,
    /// 所属请求ID（用于调试）
    pub owner_id: Option<u64>,
}

#[allow(dead_code)]
impl Block {
    /// 创建新的内存块
    pub fn new(block_id: BlockId, physical_offset: usize) -> Self {
        Self {
            block_id,
            physical_offset,
            ref_count: AtomicUsize::new(0),
            state: BlockState::Free,
            owner_id: None,
        }
    }

    /// 分配块
    pub fn allocate(&mut self, owner_id: Option<u64>) {
        self.state = BlockState::Allocated;
        self.owner_id = owner_id;
        self.ref_count.store(1, Ordering::SeqCst);
    }

    /// 增加引用计数
    pub fn inc_ref(&self) -> usize {
        self.ref_count.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// 减少引用计数
    pub fn dec_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::SeqCst) - 1
    }

    /// 获取引用计数
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// 检查是否空闲
    pub fn is_free(&self) -> bool {
        self.state == BlockState::Free || self.ref_count() == 0
    }

    /// 释放块
    pub fn free(&mut self) {
        self.state = BlockState::Free;
        self.owner_id = None;
        self.ref_count.store(0, Ordering::SeqCst);
    }

    /// 换出块
    pub fn swap_out(&mut self) {
        self.state = BlockState::Swapped;
    }

    /// 换入块
    pub fn swap_in(&mut self) {
        self.state = BlockState::Allocated;
    }
}

impl Clone for Block {
    fn clone(&self) -> Self {
        Self {
            block_id: self.block_id,
            physical_offset: self.physical_offset,
            ref_count: AtomicUsize::new(self.ref_count.load(Ordering::SeqCst)),
            state: self.state,
            owner_id: self.owner_id,
        }
    }
}

/// KV Cache配置
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct KVCacheConfig {
    /// 块大小（token数）
    pub block_size: usize,
    /// 最大块数
    pub max_blocks: usize,
    /// 层数
    pub num_layers: usize,
    /// 注意力头数
    pub num_heads: usize,
    /// 头维度
    pub head_dim: usize,
    /// 数据类型大小（字节）
    pub dtype_size: usize,
    /// 是否启用前缀缓存
    pub enable_prefix_cache: bool,
    /// 是否启用换出
    pub enable_swap: bool,
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            max_blocks: 1024,
            num_layers: 28,
            num_heads: 32,
            head_dim: 128,
            dtype_size: 2, // FP16
            enable_prefix_cache: true,
            enable_swap: false,
        }
    }
}

#[allow(dead_code)]
impl KVCacheConfig {
    /// 创建新配置
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_layers,
            num_heads,
            head_dim,
            ..Default::default()
        }
    }

    /// 计算单个块的内存大小（字节）
    pub fn block_memory_size(&self) -> usize {
        self.block_size * self.num_layers * 2 * self.num_heads * self.head_dim * self.dtype_size
    }

    /// 计算总内存大小（字节）
    pub fn total_memory_size(&self) -> usize {
        self.block_memory_size() * self.max_blocks
    }

    /// 设置最大内存（自动计算块数）
    pub fn with_max_memory(mut self, max_memory_mb: usize) -> Self {
        let block_size = self.block_memory_size();
        if block_size > 0 {
            let blocks = (max_memory_mb * 1024 * 1024) / block_size;
            self.max_blocks = blocks.min(MAX_BLOCKS);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let block = Block::new(0, 0);
        assert_eq!(block.block_id, 0);
        assert_eq!(block.ref_count(), 0);
        assert!(block.is_free());
    }

    #[test]
    fn test_block_allocation() {
        let mut block = Block::new(1, 1024);
        block.allocate(Some(42));

        assert_eq!(block.state, BlockState::Allocated);
        assert_eq!(block.ref_count(), 1);
        assert_eq!(block.owner_id, Some(42));
        assert!(!block.is_free());
    }

    #[test]
    fn test_block_ref_count() {
        let mut block = Block::new(2, 2048);
        block.allocate(None);

        assert_eq!(block.ref_count(), 1);
        assert_eq!(block.inc_ref(), 2);
        assert_eq!(block.inc_ref(), 3);
        assert_eq!(block.dec_ref(), 2);
        assert_eq!(block.ref_count(), 2);
    }

    #[test]
    fn test_block_free() {
        let mut block = Block::new(3, 3072);
        block.allocate(Some(100));
        block.free();

        assert_eq!(block.state, BlockState::Free);
        assert_eq!(block.ref_count(), 0);
        assert!(block.is_free());
    }

    #[test]
    fn test_kvcache_config() {
        let config = KVCacheConfig::default();

        assert_eq!(config.block_size, DEFAULT_BLOCK_SIZE);
        assert!(config.block_memory_size() > 0);
        assert!(config.total_memory_size() > 0);
    }

    #[test]
    fn test_kvcache_config_with_memory() {
        let config = KVCacheConfig::default().with_max_memory(1024);

        assert!(config.max_blocks > 0);
        assert!(config.max_blocks <= MAX_BLOCKS);
    }

    // ==================== 新增测试开始 ====================

    /// 测试Block的swap_out和swap_in操作
    /// 覆盖分支：换出/换入状态转换
    #[test]
    fn test_block_swap_operations() {
        let mut block = Block::new(10, 4096);

        // 初始状态应该是Free
        assert_eq!(block.state, BlockState::Free);

        // 分配后变为Allocated
        block.allocate(Some(123));
        assert_eq!(block.state, BlockState::Allocated);

        // 换出后变为Swapped
        block.swap_out();
        assert_eq!(block.state, BlockState::Swapped);

        // 换入后变回Allocated
        block.swap_in();
        assert_eq!(block.state, BlockState::Allocated);

        // owner_id在swap操作中应该保持不变
        assert_eq!(block.owner_id, Some(123));
    }

    /// 测试Block的所有状态变体
    /// 覆盖分支：BlockState枚举的完整覆盖
    #[test]
    fn test_block_state_variants() {
        let states = vec![BlockState::Free, BlockState::Allocated, BlockState::Swapped];

        for state in &states {
            // 验证Debug trait实现
            let _debug_str = format!("{:?}", state);

            // 验证Clone和Copy
            let state_copy = *state;
            assert_eq!(*state, state_copy);

            // 验证PartialEq
            assert_eq!(*state, state_copy);
        }

        // 验证不同状态不相等
        assert_ne!(BlockState::Free, BlockState::Allocated);
        assert_ne!(BlockState::Free, BlockState::Swapped);
        assert_ne!(BlockState::Allocated, BlockState::Swapped);
    }

    /// 测试Block的Clone行为
    /// 覆盖分支：Clone trait的实现细节
    #[test]
    fn test_block_clone() {
        let mut original = Block::new(42, 8192);
        original.allocate(Some(99));
        original.inc_ref();
        original.inc_ref(); // ref_count = 3

        let cloned = original.clone();

        // 验证基本字段复制
        assert_eq!(cloned.block_id, original.block_id);
        assert_eq!(cloned.physical_offset, original.physical_offset);
        assert_eq!(cloned.state, original.state);
        assert_eq!(cloned.owner_id, original.owner_id);

        // 验证引用计数被正确复制（但现在是独立的）
        assert_eq!(cloned.ref_count(), original.ref_count());

        // 验证克隆后的独立性
        cloned.inc_ref();
        assert_eq!(cloned.ref_count(), original.ref_count() + 1);
    }

    /// 测试Block分配时无owner_id
    /// 覆盖分支：allocate(None)路径
    #[test]
    fn test_block_allocate_without_owner() {
        let mut block = Block::new(5, 256);
        block.allocate(None);

        assert_eq!(block.state, BlockState::Allocated);
        assert_eq!(block.owner_id, None);
        assert_eq!(block.ref_count(), 1);
    }

    /// 测试Block的is_free在不同引用计数下的行为
    /// 覆盖分支：is_free的多种条件判断
    #[test]
    fn test_block_is_free_conditions() {
        let mut block = Block::new(6, 512);

        // 初始状态：Free且ref_count=0，应该返回true
        assert!(block.is_free());

        // 分配后：Allocated且ref_count=1，应该返回false
        block.allocate(Some(1));
        assert!(!block.is_free());

        // 减少引用计数到0：Allocated且ref_count=0，根据实现应该检查ref_count
        block.dec_ref();
        // 注意：此时ref_count可能为0，但state仍然是Allocated
        // is_free()会检查 state == Free || ref_count == 0

        // 释放后：Free且ref_count=0，应该返回true
        block.free();
        assert!(block.is_free());

        // 手动增加引用计数但不改变状态
        block.inc_ref();
        block.inc_ref();
        // 此时state=Free但ref_count>0，is_free()仍然返回true因为state==Free
        assert!(block.is_free());
    }

    /// 测试KVCacheConfig的new方法
    /// 覆盖分支：自定义配置创建
    #[test]
    fn test_kvcache_config_new() {
        let config = KVCacheConfig::new(32, 16, 128); // 32层，16个头，128维头维度

        assert_eq!(config.num_layers, 32);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.block_size, DEFAULT_BLOCK_SIZE); // 使用默认值
        assert!(config.enable_prefix_cache); // 默认启用
        assert!(!config.enable_swap); // 默认禁用
        assert_eq!(config.dtype_size, 2); // FP16默认
    }

    /// 测试KVCacheConfig的block_memory_size计算
    /// 覆盖分支：内存大小计算公式
    #[test]
    fn test_kvcache_config_memory_calculation() {
        // 创建一个简单的配置用于验证计算
        let config = KVCacheConfig {
            block_size: 4,
            max_blocks: 100,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            dtype_size: 2, // FP16
            enable_prefix_cache: true,
            enable_swap: false,
        };

        // 单个块内存 = block_size * num_layers * 2 (K+V) * num_heads * head_dim * dtype_size
        // = 4 * 2 * 2 * 2 * 8 * 2 = 256 字节
        let expected_block_size = 4 * 2 * 2 * 2 * 8 * 2;
        assert_eq!(config.block_memory_size(), expected_block_size);

        // 总内存 = block_memory_size * max_blocks
        // = 256 * 100 = 25600 字节
        assert_eq!(config.total_memory_size(), expected_block_size * 100);
    }

    /// 测试KVCacheConfig的with_max_memory边界条件
    /// 覆盖分支：with_max_memory的各种输入
    #[test]
    fn test_kvcache_config_with_max_memory_boundaries() {
        // 极小内存配置
        let tiny_config = KVCacheConfig::default().with_max_memory(1); // 1MB
        assert!(tiny_config.max_blocks <= MAX_BLOCKS);

        // 大内存配置
        let large_config = KVCacheConfig::default().with_max_memory(1024 * 1024); // 1TB（理论值）
        assert!(large_config.max_blocks <= MAX_BLOCKS); // 不应超过MAX_BLOCKS限制

        // 零内存配置（边界情况）
        let zero_config = KVCacheConfig {
            ..Default::default()
        };
        zero_config.with_max_memory(0); // 0 MB
                                        // 应该不会panic，max_blocks可能是0或某个最小值
    }

    /// 测试BlockId类型别名
    /// 覆盖分支：BlockId类型定义
    #[test]
    fn test_block_id_type() {
        let id: BlockId = 100;
        assert_eq!(id, 100);

        // 测试大ID值（接近MAX_BLOCKS）
        let large_id: BlockId = MAX_BLOCKS - 1;
        assert_eq!(large_id, MAX_BLOCKS - 1);
    }

    /// 测试Block的Debug trait实现
    /// 覆盖分支：Debug格式化输出
    #[test]
    fn test_block_debug_format() {
        let mut block = Block::new(99, 9999);
        block.allocate(Some(12345));

        let debug_str = format!("{:?}", block);

        // 验证包含关键字段信息
        assert!(debug_str.contains("99")); // block_id
        assert!(debug_str.contains("9999")); // physical_offset
    }

    /// 测试KVCacheConfig的Debug和Clone trait
    /// 覆盖分支：派生trait的实现
    #[test]
    fn test_kvcache_config_traits() {
        let config = KVCacheConfig::default();

        // Debug trait
        let debug_str = format!("{:?}", config);
        assert!(!debug_str.is_empty());

        // Clone trait
        let cloned = config.clone();
        assert_eq!(config.block_size, cloned.block_size);
        assert_eq!(config.max_blocks, cloned.max_blocks);
        assert_eq!(config.num_layers, cloned.num_layers);
    }

    /// 测试Block分配后多次释放和重新分配
    /// 覆盖分支：重复分配/释放循环
    #[test]
    fn test_block_allocate_free_cycle() {
        let mut block = Block::new(7, 1024);

        // 第一次分配
        block.allocate(Some(1));
        assert_eq!(block.ref_count(), 1);
        assert!(!block.is_free());

        // 释放
        block.free();
        assert_eq!(block.ref_count(), 0);
        assert!(block.is_free());

        // 第二次分配（不同owner）
        block.allocate(Some(2));
        assert_eq!(block.ref_count(), 1);
        assert_eq!(block.owner_id, Some(2));

        // 再次释放
        block.free();
        assert!(block.is_free());
    }

    /// 测试Block在Swapped状态下的行为
    /// 覆盖分支：Swapped状态的各种操作
    #[test]
    fn test_block_swapped_state_behavior() {
        let mut block = Block::new(8, 2048);
        block.allocate(Some(10));

        // 换出
        block.swap_out();
        assert_eq!(block.state, BlockState::Swapped);

        // Swapped状态下ref_count应该保持不变
        assert_eq!(block.ref_count(), 1);

        // Swapped状态下owner_id应该保持不变
        assert_eq!(block.owner_id, Some(10));

        // 换入后恢复Allocated状态
        block.swap_in();
        assert_eq!(block.state, BlockState::Allocated);
    }
}
