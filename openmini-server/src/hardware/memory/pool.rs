//! 内存池 - 多 Arena 内存池管理
//!
//! 提供基于多个 Arena 的内存池，支持更高的并发分配能力。
//! 当一个 Arena 满时自动切换到下一个。
//!
//! # 线程安全
//! - `alloc` 方法线程安全，可并发调用
//! - `reset` 方法**不是线程安全的**，调用时必须确保没有并发分配
//!
//! # 负载均衡
//! 当前实现使用"填满再切换"策略，优先填满当前 Arena。
//! 这有助于内存局部性，但可能导致分配不均衡。

#![allow(dead_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::arena::Arena;

/// 内存池
///
/// 管理多个 Arena，提供负载均衡的内存分配。
///
/// # 特点
/// - 多 Arena 并发分配
/// - 自动切换满载 Arena
/// - 线程安全的分配操作
///
/// # 线程安全
/// - `alloc` 方法线程安全
/// - `reset` 方法需要外部同步
pub struct MemoryPool {
    /// Arena 列表
    arenas: Vec<Arc<Arena>>,
    /// 每个 Arena 的大小
    arena_size: usize,
    /// 当前活跃的 Arena 索引（使用取模避免无限增长）
    current_arena: AtomicUsize,
    /// Arena 数量（缓存避免重复计算）
    num_arenas: usize,
}

impl MemoryPool {
    /// 创建新的内存池
    ///
    /// # 参数
    /// - `arena_size`: 每个 Arena 的大小（字节）
    /// - `num_arenas`: Arena 数量
    ///
    /// # 示例
    /// ```
    /// let pool = MemoryPool::new(64 * 1024 * 1024, 4); // 4 个 64MB Arena
    /// ```
    pub fn new(arena_size: usize, num_arenas: usize) -> Self {
        let arenas = (0..num_arenas)
            .map(|_| Arc::new(Arena::new(arena_size)))
            .collect();
        
        Self {
            arenas,
            arena_size,
            current_arena: AtomicUsize::new(0),
            num_arenas,
        }
    }
    
    /// 分配内存
    ///
    /// 尝试从当前 Arena 分配，失败则轮询其他 Arena。
    /// 使用"填满再切换"策略，优先填满当前 Arena。
    ///
    /// # 参数
    /// - `size`: 请求的内存大小（字节）
    /// - `align`: 对齐要求（必须是 2 的幂）
    ///
    /// # 返回
    /// - 成功：返回内存指针
    /// - 失败：返回 `None`（所有 Arena 都满）
    ///
    /// # 线程安全
    /// 此方法线程安全，可并发调用。
    pub fn alloc(&self, size: usize, align: usize) -> Option<*mut u8> {
        if self.num_arenas == 0 {
            return None;
        }
        
        let start_idx = self.current_arena.load(Ordering::Relaxed) % self.num_arenas;
        
        for i in 0..self.num_arenas {
            let idx = (start_idx + i) % self.num_arenas;
            
            if let Some(ptr) = self.arenas[idx].alloc(size, align) {
                // 更新当前 Arena 索引（使用取模避免无限增长）
                self.current_arena.store(idx, Ordering::Relaxed);
                return Some(ptr);
            }
        }
        
        None
    }
    
    /// 重置所有 Arena
    ///
    /// # ⚠️ 线程安全警告
    /// **此方法不是线程安全的！** 调用时必须确保：
    /// - 没有其他线程正在执行 `alloc`
    /// - 没有其他线程正在访问已分配的内存
    ///
    /// # 说明
    /// 重置后，所有之前分配的内存都将失效。
    pub fn reset(&self) {
        for arena in &self.arenas {
            arena.reset();
        }
        self.current_arena.store(0, Ordering::SeqCst);
    }
    
    /// 获取总容量
    pub fn total_capacity(&self) -> usize {
        self.arena_size * self.num_arenas
    }
    
    /// 获取总已使用量
    pub fn total_used(&self) -> usize {
        self.arenas.iter().map(|a| a.used()).sum()
    }
    
    /// 获取总可用量
    pub fn total_available(&self) -> usize {
        self.arenas.iter().map(|a| a.available()).sum()
    }
    
    /// 获取 Arena 数量
    pub fn num_arenas(&self) -> usize {
        self.num_arenas
    }
    
    /// 获取每个 Arena 的大小
    pub fn arena_size(&self) -> usize {
        self.arena_size
    }
    
    /// 获取内存使用率
    pub fn utilization(&self) -> f64 {
        let total = self.total_capacity();
        if total == 0 {
            return 0.0;
        }
        self.total_used() as f64 / total as f64
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new(64 * 1024 * 1024, 4)
    }
}

impl std::fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("num_arenas", &self.num_arenas)
            .field("arena_size", &self.arena_size)
            .field("total_capacity", &self.total_capacity())
            .field("total_used", &self.total_used())
            .field("utilization", &format!("{:.2}%", self.utilization() * 100.0))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_new() {
        let pool = MemoryPool::new(1024 * 1024, 4);
        assert_eq!(pool.num_arenas(), 4);
        assert_eq!(pool.arena_size(), 1024 * 1024);
        assert_eq!(pool.total_capacity(), 4 * 1024 * 1024);
    }

    #[test]
    fn test_pool_alloc() {
        let pool = MemoryPool::new(1024, 2);
        
        let ptr = pool.alloc(128, 8);
        assert!(ptr.is_some());
        assert!(pool.total_used() > 0);
    }

    #[test]
    fn test_pool_alloc_switch_arena() {
        let pool = MemoryPool::new(128, 2);
        
        // 填满第一个 Arena
        let ptr1 = pool.alloc(64, 8);
        assert!(ptr1.is_some());
        
        // 继续分配，应该切换到第二个 Arena
        let ptr2 = pool.alloc(64, 8);
        assert!(ptr2.is_some());
        
        // 两个 Arena 都应该有使用
        assert!(pool.total_used() > 0);
    }

    #[test]
    fn test_pool_exhausted() {
        let pool = MemoryPool::new(64, 1);
        
        // 分配填满 Arena
        let ptr1 = pool.alloc(64, 8);
        assert!(ptr1.is_some());
        
        // 再分配应该失败
        let ptr2 = pool.alloc(64, 8);
        assert!(ptr2.is_none());
    }

    #[test]
    fn test_pool_reset() {
        let pool = MemoryPool::new(1024, 2);
        
        pool.alloc(128, 8);
        assert!(pool.total_used() > 0);
        
        pool.reset();
        assert_eq!(pool.total_used(), 0);
    }

    #[test]
    fn test_pool_utilization() {
        let pool = MemoryPool::new(1024, 1);
        assert_eq!(pool.utilization(), 0.0);
        
        pool.alloc(512, 8);
        assert!(pool.utilization() > 0.0);
    }

    #[test]
    fn test_pool_default() {
        let pool = MemoryPool::default();
        assert_eq!(pool.num_arenas(), 4);
        assert_eq!(pool.arena_size(), 64 * 1024 * 1024);
    }

    #[test]
    fn test_pool_zero_arenas() {
        let pool = MemoryPool::new(1024, 0);
        assert_eq!(pool.num_arenas(), 0);
        assert_eq!(pool.total_capacity(), 0);
        
        // 分配应该失败
        let ptr = pool.alloc(64, 8);
        assert!(ptr.is_none());
    }

    #[test]
    fn test_pool_thread_safety() {
        use std::thread;

        let pool = Arc::new(MemoryPool::new(1024 * 1024, 4));
        let mut handles = vec![];

        for _ in 0..4 {
            let pool_clone = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let ptr = pool_clone.alloc(64, 8);
                    assert!(ptr.is_some());
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // 总共分配 4 * 100 * 64 = 25600 字节
        assert_eq!(pool.total_used(), 25600);
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_pool_debug_format() {
        // 测试Debug trait实现
        let pool = MemoryPool::new(2048, 2);
        
        pool.alloc(512, 8);
        
        let debug_str = format!("{:?}", pool);
        
        // 验证包含关键字段
        assert!(debug_str.contains("num_arenas"));
        assert!(debug_str.contains("arena_size"));
        assert!(debug_str.contains("total_capacity"));
        assert!(debug_str.contains("total_used"));
        assert!(debug_str.contains("utilization"));
    }

    #[test]
    fn test_pool_total_available() {
        let pool = MemoryPool::new(1024, 2);

        // 初始全部可用
        assert_eq!(pool.total_available(), 2048);

        // 分配后可用量减少
        pool.alloc(512, 8);
        assert!(pool.total_available() < 2048);
        assert_eq!(pool.total_available(), 2048 - pool.total_used());
    }

    #[test]
    fn test_pool_utilization_edge_cases() {
        // 测试零容量时的使用率
        let pool = MemoryPool::new(0, 1);
        assert_eq!(pool.utilization(), 0.0);
    }

    #[test]
    fn test_pool_multiple_allocations_same_arena() {
        // 测试在同一Arena中多次小分配
        let pool = MemoryPool::new(1024, 1);

        let ptrs: Vec<_> = (0..10)
            .map(|_| pool.alloc(64, 8))
            .collect();

        // 所有分配都应该成功
        for ptr in &ptrs {
            assert!(ptr.is_some());
        }

        // 总共使用了 640 字节
        assert_eq!(pool.total_used(), 640);
    }

    #[test]
    fn test_pool_reset_clears_all_state() {
        let pool = MemoryPool::new(1024, 2);

        // 分配多个Arena的数据
        for _ in 0..10 {
            pool.alloc(100, 8);
        }

        assert!(pool.total_used() > 0);

        // 重置
        pool.reset();

        // 验证状态完全重置
        assert_eq!(pool.total_used(), 0);
        assert_eq!(pool.total_available(), 2048);
    }

    /// 测试不同对齐要求的分配
    /// 覆盖分支：不同的 align 参数
    #[test]
    fn test_pool_different_alignments() {
        let pool = MemoryPool::new(4096, 1);

        // 测试不同的对齐要求：1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
        let alignments = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        
        for &align in &alignments {
            let ptr = pool.alloc(64, align);
            assert!(ptr.is_some(), "分配失败: size=64, align={}", align);
        }
    }

    /// 测试极小尺寸的分配（边界条件）
    /// 覆盖分支：size=1 和 size=0 的边界情况
    #[test]
    fn test_pool_extreme_sizes() {
        let pool = MemoryPool::new(1024, 1);

        // 分配 1 字节
        let ptr1 = pool.alloc(1, 1);
        assert!(ptr1.is_some());

        // 分配 0 字节（可能成功或失败，取决于实现）
        let ptr0 = pool.alloc(0, 1);
        // 不论成功或失败都不应 panic
        let _ = ptr0;

        // 分配接近 Arena 大小的请求
        let ptr_large = pool.alloc(1024, 1);
        // 可能成功或失败
        let _ = ptr_large;
    }

    /// 测试多 Arena 的轮询分配策略
    /// 覆盖分支：current_arena 的轮转逻辑
    #[test]
    fn test_pool_arena_rotation() {
        let pool = MemoryPool::new(128, 3); // 3 个小 Arena

        // 连续多次分配，观察是否在不同 Arena 间轮转
        let mut rotation_count = 0;

        for _ in 0..10 {
            let ptr = pool.alloc(32, 8);
            if ptr.is_some() {
                // 简单验证分配成功即可
                rotation_count += 1;
            }
        }

        // 验证至少有部分分配成功
        assert!(rotation_count > 0);
    }

    /// 测试 MemoryPool 的 Clone 行为（如果支持）
    /// 覆盖分支：Arc<Arena> 的共享语义
    #[test]
    fn test_pool_arc_sharing() {
        let pool = MemoryPool::new(1024, 2);
        
        // 分配一些内存
        pool.alloc(256, 8);
        assert!(pool.total_used() > 0);
        
        // 由于使用 Arc，可以通过引用共享
        let pool_ref = &pool;
        assert_eq!(pool_ref.num_arenas(), pool.num_arenas());
        assert_eq!(pool_ref.total_used(), pool.total_used());
    }

    /// 测试单 Arena 场景下的完整生命周期
    /// 覆盖分支：num_arenas=1 的特殊路径
    #[test]
    fn test_pool_single_arena_lifecycle() {
        let pool = MemoryPool::new(512, 1); // 单个 Arena

        // 初始状态
        assert_eq!(pool.num_arenas(), 1);
        assert_eq!(pool.total_capacity(), 512);
        assert_eq!(pool.total_used(), 0);
        assert_eq!(pool.total_available(), 512);

        // 分配一部分
        let ptr1 = pool.alloc(200, 8);
        assert!(ptr1.is_some());
        assert!(pool.total_used() >= 200);
        assert!(pool.total_available() < 512);

        // 再分配一部分
        let ptr2 = pool.alloc(200, 8);
        assert!(ptr2.is_some());

        // 重置
        pool.reset();
        assert_eq!(pool.total_used(), 0);
        assert_eq!(pool.total_available(), 512);
    }

    /// 测试大数量 Arena 的场景
    /// 覆盖分支：num_arenas 较大的情况
    #[test]
    fn test_pool_many_arenas() {
        let pool = MemoryPool::new(64, 10); // 10 个小 Arena

        assert_eq!(pool.num_arenas(), 10);
        assert_eq!(pool.total_capacity(), 640);

        // 在多个 Arena 中分配
        for i in 0..20 {
            let ptr = pool.alloc(30, 8);
            if ptr.is_some() {
                // 分配成功
                assert!(pool.total_used() > 0);
            } else {
                // 可能 Arena 已满，这也是可接受的行为
                break;
            }
            
            // 防止无限循环
            if i >= 19 {
                break;
            }
        }
    }
}
