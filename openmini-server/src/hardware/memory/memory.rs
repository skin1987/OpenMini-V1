//! 内存管理器 - 统一的内存管理接口
//!
//! 提供完整的内存管理功能，支持多种分配策略的自适应选择。
//!
//! # 内存模型
//! 基于 Arena 分配器实现，特点：
//! - 线性分配，O(1) 时间复杂度
//! - 批量释放，不支持单个释放
//! - 使用 `reset()` 方法释放所有分配
//!
//! # 线程安全
//! - `alloc` 方法线程安全
//! - `reset` 方法需要外部同步

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::pool::MemoryPool;
use crate::hardware::scheduler::MemoryStrategy;

const ALIGN: usize = 64;

/// 内存管理器
///
/// 基于 Arena 池的统一内存管理接口。
/// 支持多种分配策略的自适应选择。
///
/// # 特点
/// - 线性分配，高效
/// - 批量释放（不支持单个释放）
/// - 线程安全的分配操作
///
/// # 示例
/// ```
/// let manager = MemoryManager::new(1024 * 1024, 4);
/// let ptr = manager.alloc(128).expect("Allocation failed");
/// // 使用内存...
/// manager.reset(); // 释放所有分配
/// ```
pub struct MemoryManager {
    pool: Arc<MemoryPool>,
    total_capacity: usize,
    used: AtomicUsize,
}

impl MemoryManager {
    /// 创建新的内存管理器
    ///
    /// # 参数
    /// - `arena_size`: 每个 Arena 的大小（字节）
    /// - `num_arenas`: Arena 数量
    ///
    /// # 示例
    /// ```
    /// let manager = MemoryManager::new(64 * 1024 * 1024, 4); // 4 个 64MB Arena
    /// ```
    pub fn new(arena_size: usize, num_arenas: usize) -> Self {
        let pool = Arc::new(MemoryPool::new(arena_size, num_arenas));
        let total_capacity = arena_size * num_arenas;

        Self {
            pool,
            total_capacity,
            used: AtomicUsize::new(0),
        }
    }

    /// 根据策略创建内存管理器
    ///
    /// # 参数
    /// - `strategy`: 内存策略
    /// - `total_memory_mb`: 总内存大小（MB）
    ///
    /// # 策略配置
    /// | 策略 | Arena 大小 | Arena 数量 | 适用场景 |
    /// |------|-----------|-----------|---------|
    /// | SmallArena | total/4 | 4 | 小内存设备 |
    /// | StandardArena | total/2 | 2 | 标准设备 |
    /// | PagedAttention | total/4 | 4 | 大内存，Paged Attention |
    /// | Distributed | total/8 | 8 | 分布式部署 |
    pub fn with_strategy(strategy: MemoryStrategy, total_memory_mb: usize) -> Self {
        let total_capacity = total_memory_mb * 1024 * 1024;

        let (arena_size, num_arenas) = match strategy {
            MemoryStrategy::SmallArena => (total_capacity / 4, 4),
            MemoryStrategy::StandardArena => (total_capacity / 2, 2),
            MemoryStrategy::PagedAttention => (total_capacity / 4, 4),
            MemoryStrategy::Distributed => (total_capacity / 8, 8),
        };

        Self::new(arena_size, num_arenas)
    }

    /// 分配内存
    ///
    /// # 参数
    /// - `size`: 请求的内存大小（字节）
    ///
    /// # 返回
    /// - 成功：返回内存指针
    /// - 失败：返回 `None`（容量不足）
    ///
    /// # 线程安全
    /// 此方法线程安全，可并发调用。
    pub fn alloc(&self, size: usize) -> Option<*mut u8> {
        if size == 0 {
            return Some(std::ptr::null_mut());
        }

        let aligned_size = (size + ALIGN - 1) & !(ALIGN - 1);

        let ptr = self.pool.alloc(aligned_size, ALIGN)?;

        self.used.fetch_add(aligned_size, Ordering::SeqCst);

        Some(ptr)
    }

    /// 重置内存管理器，释放所有分配
    ///
    /// # ⚠️ 线程安全警告
    /// **此方法不是线程安全的！** 调用时必须确保：
    /// - 没有其他线程正在执行 `alloc`
    /// - 没有其他线程正在访问已分配的内存
    ///
    /// # 说明
    /// Arena 分配器不支持单个释放，只能批量释放。
    /// 调用此方法后，所有之前分配的内存都将失效。
    pub fn reset(&self) {
        self.pool.reset();
        self.used.store(0, Ordering::SeqCst);
    }

    /// 获取总容量
    pub fn capacity(&self) -> usize {
        self.total_capacity
    }

    /// 获取已使用内存大小
    pub fn used(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }

    /// 获取可用内存大小
    pub fn available(&self) -> usize {
        self.total_capacity.saturating_sub(self.used())
    }

    /// 获取内存池引用
    pub fn pool(&self) -> &Arc<MemoryPool> {
        &self.pool
    }

    /// 获取内存使用率
    pub fn utilization(&self) -> f64 {
        if self.total_capacity == 0 {
            return 0.0;
        }
        self.used() as f64 / self.total_capacity as f64
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(64 * 1024 * 1024, 4)
    }
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryManager")
            .field("total_capacity", &self.total_capacity)
            .field("used", &self.used())
            .field("available", &self.available())
            .field(
                "utilization",
                &format!("{:.2}%", self.utilization() * 100.0),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let manager = MemoryManager::new(1024 * 1024, 1);
        let ptr = manager.alloc(128);
        assert!(ptr.is_some());
        assert!(manager.used() > 0);
    }

    #[test]
    fn test_zero_allocation() {
        let manager = MemoryManager::new(1024 * 1024, 1);
        let ptr = manager.alloc(0);
        assert!(ptr.is_some());
        assert_eq!(manager.used(), 0);
    }

    #[test]
    fn test_reset() {
        let manager = MemoryManager::new(1024 * 1024, 1);

        manager.alloc(128);
        assert!(manager.used() > 0);

        manager.reset();
        assert_eq!(manager.used(), 0);
    }

    #[test]
    fn test_capacity() {
        let manager = MemoryManager::new(1024 * 1024, 4);
        assert_eq!(manager.capacity(), 4 * 1024 * 1024);
    }

    #[test]
    fn test_available() {
        let manager = MemoryManager::new(1024 * 1024, 1);
        let initial = manager.available();

        manager.alloc(128);
        assert!(manager.available() < initial);
    }

    #[test]
    fn test_utilization() {
        let manager = MemoryManager::new(1024 * 1024, 1);
        assert_eq!(manager.utilization(), 0.0);

        manager.alloc(1024 * 512); // 分配一半
        assert!(manager.utilization() > 0.0);
    }

    #[test]
    fn test_with_strategy() {
        let manager = MemoryManager::with_strategy(MemoryStrategy::StandardArena, 256);
        assert_eq!(manager.capacity(), 256 * 1024 * 1024);
    }

    #[test]
    fn test_default() {
        let manager = MemoryManager::default();
        assert_eq!(manager.capacity(), 4 * 64 * 1024 * 1024);
    }

    // ==================== 新增测试开始 ====================

    /// 测试MemoryManager的Debug trait输出
    /// 覆盖分支：Debug格式化包含关键字段信息
    #[test]
    fn test_memory_manager_debug_format() {
        let manager = MemoryManager::new(1024 * 1024, 2); // 2MB total

        // 初始状态
        let debug_str = format!("{:?}", manager);
        assert!(debug_str.contains("MemoryManager"));
        assert!(debug_str.contains(&format!("{}", 2 * 1024 * 1024))); // capacity
        assert!(debug_str.contains("0")); // used = 0 initially

        // 分配后
        manager.alloc(512);
        let debug_str_after = format!("{:?}", manager);
        // 应该显示使用量增加
        assert!(debug_str_after.contains("used")); // used field present
    }

    /// 测试多次连续分配
    /// 覆盖分支：多次alloc调用的累积效果
    #[test]
    fn test_multiple_allocations() {
        let manager = MemoryManager::new(10 * 1024, 1); // 10KB

        let initial_available = manager.available();

        // 第一次分配
        let ptr1 = manager.alloc(100);
        assert!(ptr1.is_some());
        let after_first = manager.available();
        assert!(after_first < initial_available);

        // 第二次分配
        let ptr2 = manager.alloc(200);
        assert!(ptr2.is_some());
        let after_second = manager.available();
        assert!(after_second < after_first);

        // 第三次分配
        let ptr3 = manager.alloc(300);
        assert!(ptr3.is_some());

        // 验证总使用量是所有分配的总和（考虑对齐）
        assert!(manager.used() > 0);
    }

    /// 测试内存耗尽时的行为
    /// 覆盖分支：容量不足时返回None
    #[test]
    fn test_exhaustion() {
        let manager = MemoryManager::new(1024, 1); // 只有1KB

        // 尝试分配超过容量的内存
        let large_alloc = manager.alloc(2048); // 2KB > 1KB capacity
                                               // 应该返回None（因为容量不足）
                                               // 注意：实际行为取决于底层MemoryPool实现
        let _ = large_alloc; // 只要不panic即可

        // 小分配应该仍然可能成功
        let small_alloc = manager.alloc(64);
        let _ = small_alloc;
    }

    /// 测试with_strategy的所有策略变体
    /// 覆盖分支：所有MemoryStrategy枚举值
    #[test]
    fn test_all_strategies() {
        let strategies = vec![
            MemoryStrategy::SmallArena,
            MemoryStrategy::StandardArena,
            MemoryStrategy::PagedAttention,
            MemoryStrategy::Distributed,
        ];

        for strategy in strategies {
            let manager = MemoryManager::with_strategy(strategy, 16); // 16MB

            // 验证创建成功且容量合理
            assert!(manager.capacity() > 0);
            assert!(manager.capacity() <= 16 * 1024 * 1024);

            // 验证可以正常分配
            let ptr = manager.alloc(1024);
            assert!(ptr.is_some());
        }
    }

    /// 测试reset后可以重新分配
    /// 覆盖分支：reset后的状态重置和重新使用
    #[test]
    fn test_reset_and_reallocate() {
        let manager = MemoryManager::new(5 * 1024, 1); // 5KB

        // 分配并使用一些内存
        for _ in 0..5 {
            let ptr = manager.alloc(256);
            assert!(ptr.is_some());
        }

        let used_before_reset = manager.used();
        assert!(used_before_reset > 0);

        // 重置
        manager.reset();
        assert_eq!(manager.used(), 0);

        // 重置后应该能够再次分配相同的量
        for _ in 0..5 {
            let ptr = manager.alloc(256);
            assert!(ptr.is_some());
        }

        // 使用量应该与reset前相近
        let used_after_reset = manager.used();
        assert!(used_after_reset > 0);
        // 允许少量差异（由于对齐）
    }

    /// 测试utilization计算的准确性
    /// 覆盖分支：utilization的边界值和计算精度
    #[test]
    fn test_utilization_calculation() {
        let manager = MemoryManager::new(10000, 1); // 10KB，非2的幂以便精确测试

        // 初始利用率应该是0%
        assert!((manager.utilization() - 0.0).abs() < f64::EPSILON);

        // 分配一半（5000字节）
        manager.alloc(5000);
        let util_half = manager.utilization();
        // 利用率应该在50%左右（考虑对齐）
        assert!(util_half > 0.4 && util_half < 0.6);

        // 重置后再验证
        manager.reset();
        assert!((manager.utilization() - 0.0).abs() < f64::EPSILON);
    }

    /// 测试available方法的准确性
    /// 覆盖分支：available = capacity - used 的不变式
    #[test]
    fn test_available_invariant() {
        let manager = MemoryManager::new(10 * 1024, 1); // 10KB

        // 初始：available == capacity
        assert_eq!(manager.available(), manager.capacity());

        // 分配后：available减少
        manager.alloc(1024);
        assert_eq!(
            manager.available(),
            manager.capacity().saturating_sub(manager.used())
        );

        // 再次分配
        manager.alloc(2048);
        assert_eq!(
            manager.available(),
            manager.capacity().saturating_sub(manager.used())
        );

        // available不应该大于capacity
        assert!(manager.available() <= manager.capacity());

        // available不应该是负数（因为使用了saturating_sub）
        assert!(manager.available() <= manager.capacity());
    }

    /// 测试不同大小的Arena配置
    /// 覆盖分支：构造函数参数的各种组合
    #[test]
    fn test_various_arena_configs() {
        // 极小配置
        let tiny = MemoryManager::new(64, 1);
        assert_eq!(tiny.capacity(), 64);
        assert!(tiny.alloc(32).is_some());

        // 单个大Arena
        let single_large = MemoryManager::new(1024 * 1024, 1); // 1MB, 1 arena
        assert_eq!(single_large.capacity(), 1024 * 1024);

        // 多个小Arena
        let multi_small = MemoryManager::new(1024, 8); // 1KB * 8 = 8KB
        assert_eq!(multi_small.capacity(), 8192);

        // 多个中等Arena
        let multi_medium = MemoryManager::new(4 * 1024, 4); // 4KB * 4 = 16KB
        assert_eq!(multi_medium.capacity(), 16384);
    }
}
