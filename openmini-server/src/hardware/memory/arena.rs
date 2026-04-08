//! Arena 内存分配器 - 高性能线性内存分配
//!
//! 提供基于 Arena 的内存分配器，适用于需要频繁分配和批量释放的场景。
//!
//! # 特点
//! - 线程安全的无锁分配（`alloc` 方法）
//! - O(1) 时间复杂度的批量重置
//! - 64 字节对齐，优化缓存性能
//!
//! # 线程安全
//! - `Arena` 实现了 `Send` 和 `Sync`，可安全跨线程共享
//! - `alloc` 方法使用原子操作实现无锁分配，线程安全
//! - `reset` 方法**不是线程安全的**，调用时必须确保没有其他线程正在使用 Arena
//!
//! # 对齐要求
//! - 对齐值必须是 2 的幂（如 1, 2, 4, 8, 16, ...）
//! - 最小对齐为 8 字节（内部强制）
//!
//! # 示例
//! ```
//! use openmini_server::hardware::memory::arena::Arena;
//!
//! let arena = Arena::new(1024 * 1024); // 1MB
//! let ptr = arena.alloc(128, 8).expect("Allocation failed");
//! arena.reset(); // 释放所有分配（确保无其他线程访问）
//! ```

#![allow(dead_code)]

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// Arena 内存分配器
///
/// 线程安全的线性分配器，支持原子操作的无锁分配。
/// 所有分配的内存会在 Arena 被丢弃时一次性释放。
///
/// # 内存布局
/// - 基地址 64 字节对齐
/// - 每次分配按指定对齐值向上取整
/// - 最小对齐 8 字节
///
/// # 线程安全
/// - `alloc` 方法线程安全，可并发调用
/// - `reset` 方法**不是线程安全的**，调用时必须确保没有并发访问
///
/// # 安全性
/// - 分配的内存未初始化
/// - 重置后不应再访问之前分配的内存
/// - 对齐值必须是 2 的幂
pub struct Arena {
    /// 内存缓冲区指针（零容量时使用 dangling）
    buffer: NonNull<u8>,
    /// 缓冲区总容量
    capacity: usize,
    /// 当前分配偏移量(原子操作)
    offset: std::sync::atomic::AtomicUsize,
}

impl Arena {
    /// 创建新的 Arena 分配器
    ///
    /// # 参数
    /// - `capacity`: 缓冲区大小(字节)，0 表示零容量 Arena
    ///
    /// # Panics
    /// 当内存分配失败时 panic
    ///
    /// # 示例
    /// ```
    /// let arena = Arena::new(1024 * 1024); // 1MB
    /// let empty = Arena::new(0); // 零容量
    /// ```
    pub fn new(capacity: usize) -> Self {
        // 处理 0 容量的情况：使用 dangling 指针，无需实际分配
        if capacity == 0 {
            return Self {
                buffer: NonNull::dangling(),
                capacity: 0,
                offset: std::sync::atomic::AtomicUsize::new(0),
            };
        }
        
        let layout = Layout::from_size_align(capacity, 64).expect("Invalid layout");
        let ptr = unsafe { alloc(layout) };
        let buffer = NonNull::new(ptr).expect("Allocation failed");
        
        Self {
            buffer,
            capacity,
            offset: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// 分配指定大小的内存
    ///
    /// 使用 CAS (Compare-And-Swap) 实现无锁分配。
    /// 自动处理对齐要求，最小对齐为 8 字节。
    ///
    /// # 参数
    /// - `size`: 请求的内存大小（字节）
    /// - `align`: 对齐要求（必须是 2 的幂）
    ///
    /// # 返回
    /// - 成功：返回内存指针
    /// - 失败：返回 None（容量不足或参数无效）
    ///
    /// # 安全性
    /// - 返回的内存未初始化
    /// - 分配 0 字节返回有效指针，但不应解引用
    ///
    /// # Panics
    /// Debug 模式下，若 `align` 不是 2 的幂会 panic
    ///
    /// # 示例
    /// ```
    /// let ptr = arena.alloc(128, 8).expect("Allocation failed");
    /// ```
    pub fn alloc(&self, size: usize, align: usize) -> Option<*mut u8> {
        // 验证对齐值是 2 的幂
        debug_assert!(align == 0 || align.is_power_of_two(), 
            "Alignment must be power of two, got {}", align);
        
        if align == 0 || !align.is_power_of_two() {
            return None;
        }
        
        // 最小对齐 8 字节
        let align = align.max(8);
        
        // 计算对齐后的大小（防止溢出）
        let aligned_size = if size == 0 {
            0
        } else {
            // 使用 checked_mul 防止溢出
            let blocks = size.checked_add(align - 1)?.checked_div(align)?;
            blocks.checked_mul(align)?
        };
        
        loop {
            let current = self.offset.load(std::sync::atomic::Ordering::Relaxed);
            
            // 防止整数溢出：检查 size > capacity - current
            // 而不是 current + size > capacity
            let available = match self.capacity.checked_sub(current) {
                Some(avail) => avail,
                None => return None, // current > capacity，不应该发生
            };
            
            if aligned_size > available {
                return None;
            }
            
            // 现在可以安全计算 new_offset
            let new_offset = current + aligned_size;
            
            match self.offset.compare_exchange_weak(
                current,
                new_offset,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let ptr = unsafe { self.buffer.as_ptr().add(current) };
                    return Some(ptr);
                }
                Err(_) => continue,
            }
        }
    }
    
    /// 重置 Arena，释放所有已分配的内存
    ///
    /// O(1) 时间复杂度，只需重置偏移量。
    ///
    /// # ⚠️ 线程安全警告
    /// **此方法不是线程安全的！** 调用 `reset` 时必须确保：
    /// - 没有其他线程正在执行 `alloc`
    /// - 没有其他线程正在访问已分配的内存
    /// - 使用外部同步机制（如 `Mutex`）保护调用
    ///
    /// # 安全性
    /// - 重置后不应再访问之前分配的内存
    /// - 不会调用任何已分配对象的析构函数
    ///
    /// # 示例
    /// ```
    /// // 单线程场景：安全
    /// arena.reset();
    ///
    /// // 多线程场景：需要外部同步
    /// let mutex = std::sync::Mutex::new(arena);
    /// mutex.lock().unwrap().reset();
    /// ```
    pub fn reset(&self) {
        self.offset.store(0, std::sync::atomic::Ordering::SeqCst);
    }
    
    /// 获取已使用的内存大小
    pub fn used(&self) -> usize {
        self.offset.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// 获取可用内存大小
    pub fn available(&self) -> usize {
        self.capacity.saturating_sub(self.used())
    }
    
    /// 获取总容量
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// 检查 Arena 是否为空（无分配）
    pub fn is_empty(&self) -> bool {
        self.used() == 0
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // 零容量 Arena 使用 dangling 指针，无需释放
        if self.capacity == 0 {
            return;
        }
        
        // Layout 已在 new 中验证，这里应该不会失败
        let layout = Layout::from_size_align(self.capacity, 64);
        if let Ok(layout) = layout {
            unsafe {
                dealloc(self.buffer.as_ptr(), layout);
            }
        }
    }
}

unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_allocation() {
        let arena = Arena::new(1024);
        let ptr = arena.alloc(128, 8);
        assert!(ptr.is_some());
        assert_eq!(arena.used(), 128);
    }

    #[test]
    fn test_multiple_allocations() {
        let arena = Arena::new(1024);
        
        let ptr1 = arena.alloc(100, 8);
        assert!(ptr1.is_some());
        
        let ptr2 = arena.alloc(200, 8);
        assert!(ptr2.is_some());
        
        // 100 对齐到 8 = 104, 200 对齐到 8 = 200
        assert_eq!(arena.used(), 304);
    }

    #[test]
    fn test_capacity_exceeded() {
        let arena = Arena::new(100);
        
        let ptr1 = arena.alloc(50, 8);
        assert!(ptr1.is_some());
        
        let ptr2 = arena.alloc(100, 8);
        assert!(ptr2.is_none()); // 应该失败
    }

    #[test]
    fn test_reset() {
        let arena = Arena::new(1024);
        
        arena.alloc(100, 8);
        // 100 对齐到 8 = 104
        assert_eq!(arena.used(), 104);
        
        arena.reset();
        assert_eq!(arena.used(), 0);
        
        // 重置后可以重新分配
        let ptr = arena.alloc(100, 8);
        assert!(ptr.is_some());
    }

    #[test]
    fn test_alignment() {
        let arena = Arena::new(1024);
        
        // 分配 17 字节，8 字节对齐 -> 应该分配 24 字节
        let ptr = arena.alloc(17, 8);
        assert!(ptr.is_some());
        assert_eq!(arena.used(), 24);
        
        arena.reset();
        
        // 分配 10 字节，16 字节对齐 -> 应该分配 16 字节
        let ptr = arena.alloc(10, 16);
        assert!(ptr.is_some());
        assert_eq!(arena.used(), 16);
    }

    #[test]
    fn test_zero_capacity() {
        let arena = Arena::new(0);
        assert_eq!(arena.capacity(), 0);
        
        // 0 容量无法分配
        let ptr = arena.alloc(1, 8);
        assert!(ptr.is_none());
    }

    #[test]
    fn test_zero_size_allocation() {
        let arena = Arena::new(1024);
        
        // 分配 0 字节应该成功
        let ptr = arena.alloc(0, 8);
        assert!(ptr.is_some());
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_invalid_alignment() {
        let arena = Arena::new(1024);
        
        // 非 2 的幂对齐应该失败（不使用 debug_assert panic）
        // 注意：在 debug 模式下会 panic，所以这里测试 0 对齐
        let ptr = arena.alloc(10, 0);
        assert!(ptr.is_none());
    }

    #[test]
    fn test_integer_overflow_protection() {
        let arena = Arena::new(1024);
        
        // 尝试分配接近 usize::MAX 的大小
        let huge_size = usize::MAX - 100;
        let ptr = arena.alloc(huge_size, 8);
        assert!(ptr.is_none(), "Should fail due to overflow protection");
    }

    #[test]
    fn test_available() {
        let arena = Arena::new(1024);
        assert_eq!(arena.available(), 1024);
        
        arena.alloc(100, 8);
        // 100 对齐到 8 = 104
        assert_eq!(arena.available(), 920);
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let arena = Arc::new(Arena::new(1024 * 1024)); // 1MB
        let mut handles = vec![];

        for _ in 0..4 {
            let arena_clone = Arc::clone(&arena);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let ptr = arena_clone.alloc(64, 8);
                    assert!(ptr.is_some());
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // 总共分配 4 * 100 * 64 = 25600 字节
        assert_eq!(arena.used(), 25600);
    }

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 is_empty 方法：初始状态和分配后状态
    #[test]
    fn test_is_empty_state() {
        let arena = Arena::new(512);
        assert!(arena.is_empty(), "New arena should be empty");

        arena.alloc(64, 8);
        assert!(!arena.is_empty(), "Arena with allocation should not be empty");

        arena.reset();
        assert!(arena.is_empty(), "Arena after reset should be empty");
    }

    /// 测试非 2 的幂对齐值触发 debug_assert panic
    #[test]
    #[should_panic(expected = "Alignment must be power of two")]
    fn test_non_power_of_two_alignment() {
        let arena = Arena::new(1024);
        let _ = arena.alloc(10, 3);
    }

    /// 测试精确填满容量的边界情况（aligned_size == available）
    #[test]
    fn test_exact_capacity_fill() {
        let arena = Arena::new(128);
        
        // 精确分配整个容量（128 字节，8字节对齐 -> 128）
        let ptr = arena.alloc(128, 8);
        assert!(ptr.is_some());
        assert_eq!(arena.available(), 0);
        assert_eq!(arena.used(), 128);

        // 再次分配应失败
        assert!(arena.alloc(1, 8).is_none());
    }

    /// 测试最小对齐强制为 8（align=1 被提升到 8）
    #[test]
    fn test_minimum_alignment_enforcement() {
        let arena = Arena::new(256);

        // align=1 应被提升到 8
        let ptr = arena.alloc(1, 1);
        assert!(ptr.is_some());
        // 分配 1 字节，最小对齐 8 => aligned_size = 8
        assert_eq!(arena.used(), 8);

        arena.reset();

        // align=2 应被提升到 8
        let ptr2 = arena.alloc(1, 2);
        assert!(ptr2.is_some());
        assert_eq!(arena.used(), 8);
    }

    /// 测试大对齐值的正确处理（如 64、128 字节对齐）
    #[test]
    fn test_large_alignment_values() {
        let arena = Arena::new(4096); // 4KB

        // 64 字节对齐，分配 1 字节 -> 64 字节
        let ptr = arena.alloc(1, 64);
        assert!(ptr.is_some());
        assert_eq!(arena.used(), 64);

        // 验证指针对齐
        if let Some(p) = ptr {
            let addr = p as usize;
            assert_eq!(addr % 64, 0, "Pointer should be 64-byte aligned");
        }
    }

    /// 测试连续重置后重新分配的正确性
    #[test]
    fn test_multiple_reset_cycles() {
        let arena = Arena::new(512);

        for cycle in 0..3 {
            let ptr = arena.alloc(100, 8);
            assert!(ptr.is_some(), "Cycle {}: alloc should succeed", cycle);
            assert!(arena.used() > 0, "Cycle {}: used should > 0", cycle);

            arena.reset();
            assert_eq!(arena.used(), 0, "Cycle {}: used should be 0 after reset", cycle);
            assert_eq!(arena.available(), 512, "Cycle {}: available should be full", cycle);
        }
    }

    /// 测试零容量 Arena 的所有方法安全边界
    #[test]
    fn test_zero_capacity_all_methods() {
        let arena = Arena::new(0);

        assert_eq!(arena.capacity(), 0);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.available(), 0);
        assert!(arena.is_empty());

        // reset 在零容量上应该是安全的
        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    /// 测试 capacity() 方法的稳定性
    #[test]
    fn test_capacity_stability() {
        let arena = Arena::new(2048);
        assert_eq!(arena.capacity(), 2048);

        // 分配后 capacity 不变
        arena.alloc(500, 8);
        assert_eq!(arena.capacity(), 2048);

        // 重置后 capacity 不变
        arena.reset();
        assert_eq!(arena.capacity(), 2048);
    }

    /// 测试分配指针的唯一性（不同分配应返回不同地址）
    #[test]
    fn test_unique_pointers() {
        let arena = Arena::new(1024);

        let ptr1 = arena.alloc(32, 8).expect("alloc 1 failed");
        let ptr2 = arena.alloc(32, 8).expect("alloc 2 failed");
        let ptr3 = arena.alloc(32, 8).expect("alloc 3 failed");

        // 三个指针应该互不相同
        assert_ne!(ptr1, ptr2, "Pointers should be unique");
        assert_ne!(ptr2, ptr3, "Pointers should be unique");
        assert_ne!(ptr1, ptr3, "Pointers should be unique");

        // 地址应该递增
        assert!(ptr2 > ptr1, "Second pointer should be after first");
        assert!(ptr3 > ptr2, "Third pointer should be after second");
    }
}
