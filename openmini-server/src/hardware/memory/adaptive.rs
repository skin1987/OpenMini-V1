//! 自适应内存策略模块
//!
//! 根据硬件能力和可用内存自动选择最优内存分配策略。
//!
//! ## 策略类型
//! - SmallArena: 小内存设备（< 1GB），激进量化 + 流式推理
//! - StandardArena: 标准内存设备（1-4GB），Arena + 缓存
//! - PagedAttention: 大内存设备（4-16GB），Paged Attention
//! - Distributed: 分布式部署（> 16GB），分布式内存
//!
//! ## 线程安全
//! `AdaptiveMemoryManager` 内部的 `Arena` 未实现线程同步。
//! 若需在多线程环境使用，请确保：
//! - 使用 `Mutex<AdaptiveMemoryManager>` 或 `RwLock<AdaptiveMemoryManager>` 包装
//! - 或确保仅单线程访问

#![allow(dead_code)]

use super::arena::Arena;
use crate::hardware::scheduler::MemoryStrategy;

/// 自适应内存管理器
/// 
/// # 线程安全
/// 此类型未实现内部同步。若需跨线程共享，请使用 `Mutex` 或 `RwLock` 包装。
pub struct AdaptiveMemoryManager {
    /// 当前内存策略
    strategy: MemoryStrategy,
    /// Arena 分配器
    arena: Option<Arena>,
    /// 总内存容量（字节）
    total_capacity: usize,
    /// KV Cache 大小（字节）
    kv_cache_size: usize,
}

impl AdaptiveMemoryManager {
    /// 创建新的自适应内存管理器
    pub fn new(strategy: MemoryStrategy, total_memory_mb: usize) -> Self {
        let total_capacity = total_memory_mb * 1024 * 1024;
        
        let arena_size = match strategy {
            MemoryStrategy::SmallArena => total_capacity / 4,
            MemoryStrategy::StandardArena => total_capacity / 2,
            MemoryStrategy::PagedAttention => total_capacity * 3 / 4,
            MemoryStrategy::Distributed => total_capacity,
        };
        
        let arena = Some(Arena::new(arena_size));
        let kv_cache_size = match strategy {
            MemoryStrategy::SmallArena => total_capacity / 8,
            MemoryStrategy::StandardArena => total_capacity / 4,
            MemoryStrategy::PagedAttention => total_capacity / 2,
            MemoryStrategy::Distributed => total_capacity * 3 / 4,
        };
        
        Self {
            strategy,
            arena,
            total_capacity,
            kv_cache_size,
        }
    }
    
    /// 从调度器配置创建
    /// 
    /// # 参数说明
    /// - `config.memory`: 内存策略（MemoryStrategy 枚举）
    /// - `config.kv_cache_size`: KV Cache 大小，单位为 MB
    /// 
    /// # 内存计算
    /// 总内存 = kv_cache_size * 4（预留 4 倍空间用于其他开销）
    /// 
    /// # 示例
    /// ```
    /// let config = InferenceConfig {
    ///     memory: MemoryStrategy::StandardArena,
    ///     kv_cache_size: 1024, // 1GB KV Cache
    ///     ..Default::default()
    /// };
    /// let manager = AdaptiveMemoryManager::from_scheduler_config(&config);
    /// ```
    pub fn from_scheduler_config(config: &crate::hardware::InferenceConfig) -> Self {
        // kv_cache_size 单位为 MB，乘以 4 预留额外空间
        // 例如：kv_cache_size = 1024 MB -> 总内存 4096 MB
        let total_memory_mb = config.kv_cache_size.saturating_mul(4);
        Self::new(config.memory, total_memory_mb)
    }
    
    /// 获取当前策略
    pub fn strategy(&self) -> MemoryStrategy {
        self.strategy
    }
    
    /// 获取 KV Cache 大小
    pub fn kv_cache_size(&self) -> usize {
        self.kv_cache_size
    }
    
    /// 获取总容量
    pub fn total_capacity(&self) -> usize {
        self.total_capacity
    }
    
    /// 分配内存
    pub fn alloc(&self, size: usize, align: usize) -> Option<*mut u8> {
        self.arena.as_ref()?.alloc(size, align)
    }
    
    /// 重置内存
    pub fn reset(&self) {
        if let Some(ref arena) = self.arena {
            arena.reset();
        }
    }
    
    /// 获取已使用内存
    pub fn used(&self) -> usize {
        self.arena.as_ref().map(|a| a.used()).unwrap_or(0)
    }
    
    /// 获取可用内存
    pub fn available(&self) -> usize {
        self.arena.as_ref().map(|a| a.available()).unwrap_or(0)
    }
    
    /// 动态调整策略
    /// 
    /// 注意：此方法会创建新的 Arena，应在系统初始化或确保所有分配已释放后调用。
    /// 如果之前有未释放的分配，这些内存将无法再访问。
    pub fn adjust_strategy(&mut self, available_memory_mb: usize) {
        let new_strategy = if available_memory_mb < 1024 {
            MemoryStrategy::SmallArena
        } else if available_memory_mb < 4096 {
            MemoryStrategy::StandardArena
        } else if available_memory_mb < 16384 {
            MemoryStrategy::PagedAttention
        } else {
            MemoryStrategy::Distributed
        };
        
        if new_strategy != self.strategy {
            // 先重置旧 Arena，释放资源
            if let Some(ref arena) = self.arena {
                arena.reset();
            }
            
            self.strategy = new_strategy;
            let total_capacity = available_memory_mb * 1024 * 1024;
            
            let arena_size = match new_strategy {
                MemoryStrategy::SmallArena => total_capacity / 4,
                MemoryStrategy::StandardArena => total_capacity / 2,
                MemoryStrategy::PagedAttention => total_capacity * 3 / 4,
                MemoryStrategy::Distributed => total_capacity,
            };
            
            let kv_cache_size = match new_strategy {
                MemoryStrategy::SmallArena => total_capacity / 8,
                MemoryStrategy::StandardArena => total_capacity / 4,
                MemoryStrategy::PagedAttention => total_capacity / 2,
                MemoryStrategy::Distributed => total_capacity * 3 / 4,
            };
            
            self.arena = Some(Arena::new(arena_size));
            self.total_capacity = total_capacity;
            self.kv_cache_size = kv_cache_size;
        }
    }
}

impl std::fmt::Debug for AdaptiveMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveMemoryManager")
            .field("strategy", &self.strategy)
            .field("total_capacity", &self.total_capacity)
            .field("kv_cache_size", &self.kv_cache_size)
            .field("used", &self.used())
            .field("available", &self.available())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_small_arena_strategy() {
        let manager = AdaptiveMemoryManager::new(MemoryStrategy::SmallArena, 512);
        assert_eq!(manager.strategy(), MemoryStrategy::SmallArena);
        assert!(manager.kv_cache_size() > 0);
    }
    
    #[test]
    fn test_standard_arena_strategy() {
        let manager = AdaptiveMemoryManager::new(MemoryStrategy::StandardArena, 2048);
        assert_eq!(manager.strategy(), MemoryStrategy::StandardArena);
    }
    
    #[test]
    fn test_paged_attention_strategy() {
        let manager = AdaptiveMemoryManager::new(MemoryStrategy::PagedAttention, 8192);
        assert_eq!(manager.strategy(), MemoryStrategy::PagedAttention);
    }
    
    #[test]
    fn test_memory_allocation() {
        let manager = AdaptiveMemoryManager::new(MemoryStrategy::StandardArena, 1024);
        let ptr = manager.alloc(1024, 8);
        assert!(ptr.is_some());
    }
    
    #[test]
    fn test_strategy_adjustment() {
        let mut manager = AdaptiveMemoryManager::new(MemoryStrategy::StandardArena, 2048);
        manager.adjust_strategy(512);
        assert_eq!(manager.strategy(), MemoryStrategy::SmallArena);
        
        manager.adjust_strategy(8192);
        assert_eq!(manager.strategy(), MemoryStrategy::PagedAttention);
    }
    
    #[test]
    fn test_distributed_strategy() {
        let mut manager = AdaptiveMemoryManager::new(MemoryStrategy::PagedAttention, 8192);
        manager.adjust_strategy(16384);
        assert_eq!(manager.strategy(), MemoryStrategy::Distributed);
    }
    
    #[test]
    fn test_kv_cache_size_sync_on_adjust() {
        let mut manager = AdaptiveMemoryManager::new(MemoryStrategy::SmallArena, 512);
        let initial_kv = manager.kv_cache_size();
        
        manager.adjust_strategy(8192);
        let new_kv = manager.kv_cache_size();
        
        // PagedAttention 策略的 kv_cache_size 应该是 total_capacity / 2
        // total_capacity = 8192 * 1024 * 1024 = 8GB
        // kv_cache_size = 4GB
        assert!(new_kv > initial_kv, "KV cache size should increase");
        assert_eq!(new_kv, 8192 * 1024 * 1024 / 2);
    }
    
    #[test]
    fn test_arena_reset_on_strategy_change() {
        let mut manager = AdaptiveMemoryManager::new(MemoryStrategy::StandardArena, 2048);
        
        // 分配一些内存
        let ptr = manager.alloc(1024, 8);
        assert!(ptr.is_some());
        assert!(manager.used() > 0);
        
        // 调整策略应该重置 Arena
        manager.adjust_strategy(512);
        assert_eq!(manager.used(), 0, "Arena should be reset after strategy change");
    }
    
    #[test]
    fn test_saturating_mul_in_from_config() {
        use crate::hardware::scheduler::{InferenceConfig, ScheduleStrategy, AttentionStrategy, ParallelStrategy};
        
        let config = InferenceConfig {
            strategy: ScheduleStrategy::Standard,
            attention: AttentionStrategy::Standard,
            memory: MemoryStrategy::StandardArena,
            parallel: ParallelStrategy::Single,
            num_threads: 4,
            use_simd: true,
            use_gpu: false,
            kv_cache_size: 1024, // 1GB
            batch_size: 32,
        };
        
        let manager = AdaptiveMemoryManager::from_scheduler_config(&config);
        // 总内存 = 1024 * 4 = 4096 MB
        assert_eq!(manager.total_capacity(), 4096 * 1024 * 1024);
    }

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 Distributed 策略的 new() 构造（覆盖所有策略变体）
    #[test]
    fn test_distributed_new() {
        let manager = AdaptiveMemoryManager::new(MemoryStrategy::Distributed, 16384);
        assert_eq!(manager.strategy(), MemoryStrategy::Distributed);
        // Distributed: arena_size = total_capacity (全部内存)
        // kv_cache_size = total_capacity * 3 / 4
        let expected_total = 16384 * 1024 * 1024;
        assert_eq!(manager.total_capacity(), expected_total);
        assert_eq!(manager.kv_cache_size(), expected_total * 3 / 4);
    }

    /// 测试 adjust_strategy 各阈值边界（<1024, <4096, <16384, >=16384）
    #[test]
    fn test_adjust_strategy_all_thresholds() {
        // 边界1: 1023 -> SmallArena (<1024)
        let mut mgr = AdaptiveMemoryManager::new(MemoryStrategy::Distributed, 20000);
        mgr.adjust_strategy(1023);
        assert_eq!(mgr.strategy(), MemoryStrategy::SmallArena);

        // 边界2: 1024 -> StandardArena (>=1024 且 <4096)
        mgr.adjust_strategy(1024);
        assert_eq!(mgr.strategy(), MemoryStrategy::StandardArena);

        // 边界3: 4095 -> StandardArena (<4096)
        mgr.adjust_strategy(4095);
        assert_eq!(mgr.strategy(), MemoryStrategy::StandardArena);

        // 边界4: 4096 -> PagedAttention (>=4096 且 <16384)
        mgr.adjust_strategy(4096);
        assert_eq!(mgr.strategy(), MemoryStrategy::PagedAttention);

        // 边界5: 16383 -> PagedAttention (<16384)
        mgr.adjust_strategy(16383);
        assert_eq!(mgr.strategy(), MemoryStrategy::PagedAttention);

        // 边界6: 16384 -> Distributed (>=16384)
        mgr.adjust_strategy(16384);
        assert_eq!(mgr.strategy(), MemoryStrategy::Distributed);
    }

    /// 测试 adjust_strategy 相同策略时不重建 Arena（跳过分支）
    #[test]
    fn test_adjust_strategy_same_strategy_noop() {
        let mut manager = AdaptiveMemoryManager::new(MemoryStrategy::StandardArena, 2048);

        // 先分配一些数据
        let _ptr = manager.alloc(256, 8);
        let used_before = manager.used();
        assert!(used_before > 0);

        // 调整到相同内存范围（仍为 StandardArena），不应重置
        manager.adjust_strategy(3000); // 3000 在 [1024, 4096) => StandardArena
        assert_eq!(manager.strategy(), MemoryStrategy::StandardArena);
        // 因为策略没变，Arena 不应被重置
        assert_eq!(manager.used(), used_before, "Same strategy should not reset arena");
    }

    /// 测试各策略的 arena 和 kv_cache_size 比例关系
    #[test]
    fn test_strategy_ratios() {
        let base_mb = 8000; // 8GB

        // SmallArena: arena=total/4, kv=total/8
        let small = AdaptiveMemoryManager::new(MemoryStrategy::SmallArena, base_mb);
        let small_total = base_mb * 1024 * 1024;
        assert_eq!(small.kv_cache_size(), small_total / 8);

        // StandardArena: arena=total/2, kv=total/4
        let std_mgr = AdaptiveMemoryManager::new(MemoryStrategy::StandardArena, base_mb);
        assert_eq!(std_mgr.kv_cache_size(), small_total / 4);

        // PagedAttention: arena=total*3/4, kv=total/2
        let paged = AdaptiveMemoryManager::new(MemoryStrategy::PagedAttention, base_mb);
        assert_eq!(paged.kv_cache_size(), small_total / 2);

        // Distributed: arena=total, kv=total*3/4
        let dist = AdaptiveMemoryManager::new(MemoryStrategy::Distributed, base_mb);
        assert_eq!(dist.kv_cache_size(), small_total * 3 / 4);
    }

    /// 测试 reset 方法后可用内存恢复
    #[test]
    fn test_reset_restores_available() {
        let manager = AdaptiveMemoryManager::new(MemoryStrategy::StandardArena, 1024); // 1MB

        let avail_before = manager.available();
        let _ptr = manager.alloc(256, 8).expect("alloc failed");
        assert!(manager.available() < avail_before);

        manager.reset();
        assert_eq!(manager.used(), 0);
        // 可用内存应该恢复（至少接近原始值，考虑对齐）
        assert!(manager.available() >= avail_before - 16, "Available should be restored after reset");
    }

    /// 测试 alloc/used/available 联动一致性
    #[test]
    fn test_alloc_used_available_consistency() {
        let manager = AdaptiveMemoryManager::new(MemoryStrategy::StandardArena, 1024);
        let cap = manager.total_capacity();

        assert_eq!(manager.used(), 0);
        assert!(manager.available() <= cap);

        let _ptr1 = manager.alloc(128, 8);
        let used_after_1 = manager.used();
        assert!(used_after_1 > 0);
        assert!(used_after_1 + manager.available() <= cap);

        let _ptr2 = manager.alloc(64, 8);
        let used_after_2 = manager.used();
        assert!(used_after_2 > used_after_1);
        assert!(used_after_2 + manager.available() <= cap);
    }

    /// 测试 Debug trait 输出包含关键字段
    #[test]
    fn test_debug_output() {
        let manager = AdaptiveMemoryManager::new(MemoryStrategy::PagedAttention, 4096);
        let debug_str = format!("{:?}", manager);

        // Debug 输出应包含结构体名和关键字段
        assert!(debug_str.contains("AdaptiveMemoryManager"));
        assert!(debug_str.contains("strategy"));
        assert!(debug_str.contains("total_capacity"));
        assert!(debug_str.contains("kv_cache_size"));
    }

    /// 测试 from_scheduler_config 的 saturating_mul 溢出保护
    #[test]
    fn test_from_config_saturating_overflow_protection() {
        use crate::hardware::scheduler::{InferenceConfig, ScheduleStrategy, AttentionStrategy, ParallelStrategy};

        let config = InferenceConfig {
            strategy: ScheduleStrategy::Standard,
            attention: AttentionStrategy::Standard,
            memory: MemoryStrategy::SmallArena,
            parallel: ParallelStrategy::Single,
            num_threads: 1,
            use_simd: true,
            use_gpu: false,
            kv_cache_size: 100_000,
            batch_size: 1,
        };

        let manager = AdaptiveMemoryManager::from_scheduler_config(&config);
        assert!(manager.total_capacity() > 0);
        assert_eq!(manager.strategy(), MemoryStrategy::SmallArena);
    }

    /// 测试最小内存配置（1MB）的极端情况
    #[test]
    fn test_minimum_memory_config() {
        // 最小 SmallArena 配置：1MB
        let manager = AdaptiveMemoryManager::new(MemoryStrategy::SmallArena, 1);
        assert_eq!(manager.strategy(), MemoryStrategy::SmallArena);
        assert!(manager.total_capacity() > 0);
        assert!(manager.kv_cache_size() > 0);

        // 应该能进行小量分配
        let ptr = manager.alloc(64, 8);
        assert!(ptr.is_some(), "Should be able to allocate in minimum config");
    }
}
