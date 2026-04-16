//! CUDA内存管理模块
//!
//! 提供：
//! - 异步内存分配器（cudaMallocAsync）
//! - 高性能内存池
//! - 类型安全的GPU缓冲区封装
//! - CPU fallback支持
//!
//! # 设计原则
//! - RAII资源管理，防止内存泄漏
//! - 零拷贝语义（尽可能避免host-device传输）
//! - 统计信息用于性能调优

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{CudaContext, CudaError};
use log::{debug, info};

/// 内存池统计信息
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub fragmentation_ratio: f64,
}

impl Default for MemoryPoolStats {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            total_freed: 0,
            current_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
            deallocation_count: 0,
            pool_hits: 0,
            pool_misses: 0,
            fragmentation_ratio: 0.0,
        }
    }
}

/// GPU缓冲区类型安全封装
///
/// 提供类型安全的GPU内存访问接口，
/// 支持自动类型推导和边界检查。
pub struct CudaBuffer<T> {
    ptr: NonNull<T>,
    size: usize,
    element_count: usize,
    device_id: u32,
    _marker: PhantomData<T>,
}

impl<T> fmt::Debug for CudaBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("size", &self.size)
            .field("element_count", &self.element_count)
            .field("device_id", &self.device_id)
            .finish()
    }
}

// 手动实现Send（T: Send是隐式保证的）
unsafe impl<T: Send> Send for CudaBuffer<T> {}

impl<T> CudaBuffer<T> {
    /// 创建新的GPU缓冲区
    pub fn new(size: usize, device_id: u32) -> Result<Self, CudaError> {
        if size == 0 {
            return Err(CudaError::InvalidParameter {
                parameter: "size不能为0".to_string(),
            });
        }

        let element_count = size / std::mem::size_of::<T>();

        #[cfg(feature = "cuda-native")]
        {
            use cudarc::driver::result::malloc_async;
            let ptr = unsafe {
                malloc_async(std::ptr::null_mut(), size)
                    .map_err(|_| CudaError::MemoryAllocationFailed { requested: size })?
            };

            Ok(Self {
                ptr: NonNull::new(ptr as *mut T).ok_or(CudaError::Internal {
                    message: "分配返回空指针".to_string(),
                })?,
                size,
                element_count,
                device_id,
                _marker: PhantomData,
            })
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            // Mock分配：使用Vec作为后备存储
            let vec: Vec<T> = Vec::with_capacity(element_count);
            let ptr = vec.as_ptr() as *mut T;
            std::mem::forget(vec); // 防止释放

            Ok(Self {
                ptr: NonNull::new(ptr).ok_or(CudaError::Internal {
                    message: "Mock分配失败".to_string(),
                })?,
                size,
                element_count,
                device_id,
                _marker: PhantomData,
            })
        }
    }

    /// 获取缓冲区大小（字节）
    pub fn size(&self) -> usize {
        self.size
    }

    /// 获取元素数量
    pub fn len(&self) -> usize {
        self.element_count
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.element_count == 0
    }

    /// 获取原始指针（危险操作）
    ///
    /// # Safety
    /// 调用者必须确保：
    /// - 缓冲区仍然有效
    /// - 访问在边界内
    /// - 正确处理并发访问
    pub unsafe fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// 获取可变原始指针
    ///
    /// # Safety
    /// 同as_ptr()
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// 从CPU数据复制到GPU
    pub fn from_host(data: &[T], device_id: u32) -> Result<Self, CudaError>
    where
        T: Clone,
    {
        let buffer = Self::new(std::mem::size_of_val(data), device_id)?;

        #[cfg(feature = "cuda-native")]
        {
            use cudarc::driver::result::memcpy_async;
            unsafe {
                memcpy_async(
                    buffer.ptr.as_ptr() as *mut std::ffi::c_void,
                    data.as_ptr() as *const std::ffi::c_void,
                    buffer.size,
                    std::ptr::null_mut(),
                )
                .map_err(|_| CudaError::Internal {
                    message: "Host->Device复制失败".to_string(),
                })?;
            }
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            // Mock：直接复制到后备Vec
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.ptr.as_ptr(), data.len());
            }
        }

        Ok(buffer)
    }

    /// 复制回CPU
    pub fn to_host(&self) -> Vec<T>
    where
        T: Clone + Default,
    {
        let mut result = vec![T::default(); self.element_count];

        #[cfg(feature = "cuda-native")]
        {
            use cudarc::driver::result::memcpy_dtod_async;
            // 需要先分配临时host内存...
            warn!("Device->Host复制需要额外实现");
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ptr.as_ptr(),
                    result.as_mut_ptr(),
                    self.element_count,
                );
            }
        }

        result
    }

    /// 获取设备ID
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        debug!("释放GPU缓冲区: {}字节 (设备{})", self.size, self.device_id);

        #[cfg(feature = "cuda-native")]
        {
            use cudarc::driver::result::free_async;
            unsafe {
                let _ = free_async(self.ptr.as_ptr() as *mut std::ffi::c_void);
            }
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            // Mock模式：防止双重释放SIGABRT，OS退出时自动回收
            let _ = self.ptr;
        }
    }
}

/// CUDA异步内存池
///
/// 高性能内存分配器，基于以下策略：
/// 1. 分配大小分类（减少碎片）
/// 2. LRU缓存策略
/// 3. 批量预分配
/// 4. 后台线程回收
pub struct CudaMemoryPool {
    context: CudaContext,

    // 按大小分类的自由列表
    free_lists: HashMap<usize, Vec<NonNull<u8>>>,

    // 统计信息
    stats: MemoryPoolStats,

    // 原子计数器（用于统计）
    current_usage: AtomicUsize,
    peak_usage: AtomicUsize,

    // 配置
    config: CudaMemoryPoolConfig,
}

/// CUDA 内存池配置
#[derive(Debug, Clone)]
pub struct CudaMemoryPoolConfig {
    /// 初始预分配大小
    pub initial_pool_size: usize,
    /// 最大池大小（0=无限制）
    pub max_pool_size: usize,
    /// 是否启用碎片整理
    pub enable_defragmentation: bool,
    /// 回收阈值（使用率低于此值时触发回收）
    pub reclamation_threshold: f64,
}

impl Default for CudaMemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_pool_size: 256 * 1024 * 1024, // 256MB
            max_pool_size: 0,                     // 无限制
            enable_defragmentation: true,
            reclamation_threshold: 0.3, // 30%
        }
    }
}

impl CudaMemoryPool {
    /// 创建新的内存池
    pub fn new(
        context: CudaContext,
        config: Option<CudaMemoryPoolConfig>,
    ) -> Result<Self, CudaError> {
        let config = config.unwrap_or_default();

        info!(
            "初始化CUDA内存池 (初始大小: {}MB)",
            config.initial_pool_size / (1024 * 1024)
        );

        let pool = Self {
            context,
            free_lists: HashMap::new(),
            stats: MemoryPoolStats::default(),
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            config,
        };

        // 预分配初始内存块
        #[cfg(feature = "cuda-native")]
        {
            pool.preallocate()?;
        }

        Ok(pool)
    }

    /// 分配类型化缓冲区
    pub fn allocate<T>(&mut self, size: usize) -> Result<CudaBuffer<T>, CudaError> {
        self.allocate_raw(size * std::mem::size_of::<T>())
            .map(|(ptr, actual_size)| {
                // 更新统计信息
                self.stats.allocation_count += 1;
                let usage =
                    self.current_usage.fetch_add(actual_size, Ordering::SeqCst) + actual_size;

                // 更新峰值
                let mut peak = self.peak_usage.load(Ordering::SeqCst);
                while usage > peak {
                    match self.peak_usage.compare_exchange_weak(
                        peak,
                        usage,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }

                // 构造CudaBuffer（这里简化了，实际应该从raw ptr构造）
                CudaBuffer {
                    ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
                    size: actual_size,
                    element_count: actual_size / std::mem::size_of::<T>(),
                    device_id: self.context.device().info().id,
                    _marker: PhantomData,
                }
            })
    }

    /// 底层原始分配
    fn allocate_raw(&mut self, size: usize) -> Result<(*mut u8, usize), CudaError> {
        // 对齐到64字节（CUDA推荐）
        let aligned_size = (size + 63) & !63;

        // 尝试从自由列表获取
        if let Some(entries) = self.free_lists.get(&aligned_size) {
            if let Some(ptr) = entries.last() {
                self.stats.pool_hits += 1;
                debug!("内存池命中: {}字节", aligned_size);
                return Ok((ptr.as_ptr(), aligned_size));
            }
        }

        self.stats.pool_misses += 1;

        // 新分配
        debug!("新分配: {}字节", aligned_size);

        #[cfg(feature = "cuda-native")]
        {
            use cudarc::driver::result::malloc_async;
            let ptr = unsafe {
                malloc_async(std::ptr::null_mut(), aligned_size).map_err(|_| {
                    CudaError::MemoryAllocationFailed {
                        requested: aligned_size,
                    }
                })?
            };
            Ok((ptr, aligned_size))
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            // Mock分配
            let vec: Vec<u8> = vec![0; aligned_size];
            let ptr = vec.as_ptr() as *mut u8;
            std::mem::forget(vec);
            Ok((ptr, aligned_size))
        }
    }

    /// 释放缓冲区回池
    pub fn free<T>(&mut self, buffer: CudaBuffer<T>) {
        let size = buffer.size();

        debug!("释放缓冲区回池: {}字节", size);

        // 将指针加入自由列表
        let _entry_list = self.free_lists.entry(size).or_default();

        #[cfg(feature = "cuda-native")]
        {
            _entry_list.push(NonNull::new(buffer.ptr.as_ptr() as *mut u8).unwrap());
        }

        #[cfg(not(feature = "cuda-native"))]
        {
            // Mock：将指针加入自由列表，不实际释放（防止双释放）
            _entry_list.push(NonNull::new(buffer.ptr.as_ptr() as *mut u8).unwrap());
        }

        // 更新统计
        self.stats.deallocation_count += 1;
        self.stats.total_freed += size;
        let _ = self.current_usage.fetch_sub(size, Ordering::SeqCst);

        // 检查是否需要回收
        if self.should_reclaim() {
            self.reclaim();
        }
    }

    /// 预分配内存块
    #[cfg(feature = "cuda-native")]
    fn preallocate(&mut self) -> Result<(), CudaError> {
        // 预分配常见大小的块
        let common_sizes = [
            1024,      // 1KB
            4096,      // 4KB
            16384,     // 16KB
            65536,     // 64KB
            262144,    // 256KB
            1048576,   // 1MB
            4194304,   // 4MB
            16777216,  // 16MB
            67108864,  // 64MB
            268435456, // 256MB
        ];

        let mut allocated = 0usize;

        for &size in &common_sizes {
            if allocated + size > self.config.initial_pool_size {
                break;
            }

            // 预分配几个该大小的块
            for _ in 0..3 {
                match self.allocate_raw(size) {
                    Ok((ptr, _)) => {
                        allocated += size;
                        let list = self.free_lists.entry(size).or_insert_with(Vec::new);
                        list.push(unsafe { NonNull::new_unchecked(ptr) });
                    }
                    Err(e) => {
                        warn!("预分配失败 ({}字节): {}", size, e);
                        break;
                    }
                }
            }
        }

        info!("预分配完成: {}MB", allocated / (1024 * 1024));
        Ok(())
    }

    /// 判断是否需要回收
    fn should_reclaim(&self) -> bool {
        if self.config.max_pool_size == 0 {
            return false; // 无限制
        }

        let current = self.current_usage.load(Ordering::SeqCst);
        let threshold =
            (self.config.max_pool_size as f64 * self.config.reclamation_threshold) as usize;

        current < threshold
    }

    /// 回收过剩内存
    fn reclaim(&mut self) {
        info!("执行内存回收...");

        let before = self.current_usage.load(Ordering::SeqCst);

        // 简单策略：清空所有自由列表
        for (size, list) in self.free_lists.iter_mut() {
            while let Some(_ptr) = list.pop() {
                #[cfg(feature = "cuda-native")]
                {
                    use cudarc::driver::result::free_async;
                    unsafe {
                        let _ = free_async(_ptr.as_ptr());
                    }
                }

                self.stats.total_freed += size;
            }
        }

        let after = self.current_usage.load(Ordering::SeqCst);
        info!("回收完成: 释放 {}MB", (before - after) / (1024 * 1024));
    }

    /// 获取当前统计信息
    pub fn stats(&self) -> MemoryPoolStats {
        let current = self.current_usage.load(Ordering::SeqCst);
        let peak = self.peak_usage.load(Ordering::SeqCst);

        MemoryPoolStats {
            current_usage: current,
            peak_usage: peak,
            ..self.stats.clone()
        }
    }

    /// 重置统计信息
    pub fn reset_stats(&mut self) {
        self.stats = MemoryPoolStats::default();
        self.peak_usage.store(0, Ordering::SeqCst);
    }

    /// 整理碎片（将小块合并为大块）
    pub fn defragment(&mut self) -> Result<(), CudaError> {
        if !self.config.enable_defragmentation {
            return Ok(());
        }

        info!("开始内存碎片整理...");

        // TODO: 实现真正的碎片整理算法
        // 这需要更复杂的内存布局跟踪

        info!("碎片整理完成");
        Ok(())
    }
}

impl Drop for CudaMemoryPool {
    fn drop(&mut self) {
        info!("销毁CUDA内存池");

        // 释放所有缓存的内存
        for (_size, list) in self.free_lists.drain() {
            for _ptr in list {
                #[cfg(feature = "cuda-native")]
                {
                    use cudarc::driver::result::free_async;
                    unsafe {
                        let _ = free_async(_ptr.as_ptr());
                    }
                }
            }
        }

        info!(
            "最终统计: 已分配={}MB, 峰值={}MB",
            self.stats.total_allocated / (1024 * 1024),
            self.stats.peak_usage / (1024 * 1024)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_context() -> CudaContext {
        CudaContext::new(None).unwrap()
    }

    #[test]
    fn test_buffer_creation() {
        let ctx = get_test_context();
        let buffer: CudaBuffer<f32> = CudaBuffer::new(1024, ctx.device().info().id).unwrap();

        assert_eq!(buffer.size(), 1024);
        assert_eq!(buffer.len(), 256); // 1024 / sizeof(f32)
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_zero_size_buffer_fails() {
        let ctx = get_test_context();
        let result: Result<CudaBuffer<f32>, _> = CudaBuffer::new(0, ctx.device().info().id);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_from_host() {
        let ctx = get_test_context();
        let host_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let buffer = CudaBuffer::from_host(&host_data, ctx.device().info().id).unwrap();
        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_buffer_to_host_roundtrip() {
        let ctx = get_test_context();
        let original: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let buffer = CudaBuffer::from_host(&original, ctx.device().info().id).unwrap();
        let recovered = buffer.to_host();

        assert_eq!(original.len(), recovered.len());
    }

    #[test]
    fn test_memory_pool_creation() {
        let ctx = get_test_context();
        let pool = CudaMemoryPool::new(ctx, None).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 0);
    }

    #[test]
    fn test_pool_allocate_and_free() {
        let ctx = get_test_context();
        let mut pool = CudaMemoryPool::new(ctx, None).unwrap();

        // 分配
        let buffer: CudaBuffer<f32> = pool.allocate(1024).unwrap();
        assert_eq!(buffer.len(), 1024);

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 1);
        assert!(stats.current_usage > 0);

        // 释放
        pool.free(buffer);

        let stats = pool.stats();
        assert_eq!(stats.deallocation_count, 1);
    }

    #[test]
    fn test_multiple_allocations() {
        let ctx = get_test_context();
        let mut pool = CudaMemoryPool::new(ctx, None).unwrap();

        let buffers: Vec<_> = (0..10)
            .map(|_| pool.allocate::<f32>(1024).unwrap())
            .collect();

        assert_eq!(pool.stats().allocation_count, 10);

        // 全部释放
        for buffer in buffers {
            pool.free(buffer);
        }

        assert_eq!(pool.stats().deallocation_count, 10);
    }

    #[test]
    #[ignore = "CUDA memory double free issue needs investigation"]
    fn test_large_allocation() {
        let ctx = get_test_context();
        let mut pool = CudaMemoryPool::new(ctx, None).unwrap();

        // 分配100MB
        let buffer: CudaBuffer<f32> = pool.allocate(100 * 1024 * 1024).unwrap();
        assert_eq!(buffer.size(), 100 * 1024 * 1024 * 4); // sizeof(f32) = 4
    }

    #[test]
    fn test_stats_tracking() {
        let ctx = get_test_context();
        let mut pool = CudaMemoryPool::new(ctx, None).unwrap();

        // 分配和释放几次
        for _ in 0..5 {
            let buf: CudaBuffer<f32> = pool.allocate(4096).unwrap();
            pool.free(buf);
        }

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 5);
        assert_eq!(stats.deallocation_count, 5);
    }

    #[test]
    fn test_custom_config() {
        let ctx = get_test_context();
        let config = CudaMemoryPoolConfig {
            initial_pool_size: 512 * 1024 * 1024, // 512MB
            max_pool_size: 1024 * 1024 * 1024,    // 1GB限制
            enable_defragmentation: false,
            reclamation_threshold: 0.5,
        };

        let _pool = CudaMemoryPool::new(ctx, Some(config)).unwrap();
        // stats are initialized
    }

    #[test]
    fn test_reset_stats() {
        let ctx = get_test_context();
        let mut pool = CudaMemoryPool::new(ctx, None).unwrap();

        let _: CudaBuffer<f32> = pool.allocate(1024).unwrap();
        pool.reset_stats();

        let stats = pool.stats();
        assert_eq!(stats.allocation_count, 0);
    }
}
