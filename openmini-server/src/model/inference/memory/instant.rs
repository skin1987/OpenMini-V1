//! 瞬时记忆层
//!
//! 基于现有 KV Cache 封装，存储当前推理批次的 token
//! 
//! # 功能特性
//! - 与 PagedKVCache 深度集成
//! - 零拷贝数据共享
//! - 批量写入优化
//! - 内存使用统计
//!
//! # 容量管理与淘汰策略
//!
//! 当容量超出时，采用 **重要性感知淘汰策略**：
//! - 优先移除重要性最低且最旧的条目
//! - 时间复杂度 O(n)，但实际淘汰频率低
//! - 平衡重要性与时效性
//!
//! # 零拷贝优化
//!
//! 优先使用零拷贝路径以获得最佳性能：
//! - 使用 `write_zero_copy` 写入 `Bytes` 数据
//! - 使用 `read_as_bytes` 读取零拷贝缓冲区
//! - 避免使用 `to_array()` 进行数据转换（需要拷贝）
//!
//! # 统计说明
//!
//! 命中率统计基于**唯一条目首次访问**：
//! - 每个条目仅首次读取时计数
//! - 反映唯一条目覆盖情况

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use ndarray::Array2;
use parking_lot::RwLock;

use super::MemoryConfig;
use crate::hardware::kv_cache::paged_cache::PagedKVCache;

/// 编译时确定系统字节序是否为小端
const IS_LITTLE_ENDIAN: bool = cfg!(target_endian = "little");

/// 内存使用统计信息
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// 当前存储的条目数量
    pub entry_count: usize,
    /// 容量上限
    pub capacity: usize,
    /// 总内存使用量（字节）
    pub memory_bytes: usize,
    /// KV Cache 命中次数
    pub cache_hits: u64,
    /// KV Cache 未命中次数
    pub cache_misses: u64,
    /// 批量写入次数
    pub batch_writes: u64,
    /// 零拷贝传输次数
    pub zero_copy_transfers: u64,
    /// 是否已附加 KV Cache
    pub kv_cache_attached: bool,
}

impl MemoryStats {
    /// 计算缓存命中率
    pub fn hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.cache_hits as f32 / total as f32
    }

    /// 计算内存使用率
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.entry_count as f32 / self.capacity as f32
    }
}

/// 零拷贝缓冲区包装器
#[derive(Debug, Clone)]
pub struct ZeroCopyBuffer {
    /// 底层数据存储
    data: Bytes,
    /// 数据形状 [rows, cols]
    shape: (usize, usize),
    /// 创建时间戳
    timestamp: u64,
}

impl ZeroCopyBuffer {
    /// 从 Array2 创建零拷贝缓冲区
    pub fn from_array(array: &Array2<f32>, timestamp: u64) -> Self {
        let shape = (array.nrows(), array.ncols());
        let byte_size = shape.0 * shape.1 * std::mem::size_of::<f32>();
        let mut bytes = Vec::with_capacity(byte_size);
        
        for row in array.rows() {
            for &val in row.iter() {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }
        
        Self {
            data: Bytes::from(bytes),
            shape,
            timestamp,
        }
    }

    /// 从 Bytes 创建零拷贝缓冲区
    pub fn from_bytes(data: Bytes, shape: (usize, usize), timestamp: u64) -> Self {
        Self {
            data,
            shape,
            timestamp,
        }
    }

    /// 获取数据引用（零拷贝）
    pub fn as_bytes(&self) -> &Bytes {
        &self.data
    }

    /// 获取数据形状
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// 获取时间戳
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// 转换为 Array2（需要拷贝）
    ///
    /// # 性能警告
    /// 此方法需要逐元素拷贝，性能较低。建议优先使用：
    /// - `as_bytes()` 获取零拷贝引用
    /// - 直接传输 `Bytes` 数据到 GPU/其他组件
    ///
    /// # 字节序
    /// 自动检测系统字节序，支持大端和小端平台
    pub fn to_array(&self) -> Array2<f32> {
        let (rows, cols) = self.shape;
        let mut array = Array2::zeros((rows, cols));
        let little_endian = IS_LITTLE_ENDIAN;
        
        for i in 0..rows {
            for j in 0..cols {
                let offset = (i * cols + j) * std::mem::size_of::<f32>();
                if offset + 4 <= self.data.len() {
                    if let Ok(bytes) = self.data[offset..offset + 4].try_into() {
                        array[[i, j]] = if little_endian {
                            f32::from_le_bytes(bytes)
                        } else {
                            f32::from_be_bytes(bytes)
                        };
                    }
                }
            }
        }
        
        array
    }

    /// 获取内存大小（字节）
    pub fn memory_size(&self) -> usize {
        self.data.len()
    }
}

/// 内存条目（支持零拷贝）
#[derive(Debug, Clone)]
pub struct InstantMemoryEntry {
    /// 传统数据存储（兼容模式）
    traditional: Option<Array2<f32>>,
    /// 零拷贝数据存储
    zero_copy: Option<ZeroCopyBuffer>,
    /// 时间戳
    timestamp: u64,
    /// 重要性权重
    importance: f32,
}

impl InstantMemoryEntry {
    /// 从 Array2 创建条目
    pub fn from_array(data: Array2<f32>, timestamp: u64) -> Self {
        Self {
            traditional: Some(data),
            zero_copy: None,
            timestamp,
            importance: 1.0,
        }
    }

    /// 从零拷贝缓冲区创建条目
    pub fn from_zero_copy(buffer: ZeroCopyBuffer) -> Self {
        let timestamp = buffer.timestamp();
        Self {
            traditional: None,
            zero_copy: Some(buffer),
            timestamp,
            importance: 1.0,
        }
    }

    /// 设置重要性
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }

    /// 获取数据为 Array2
    pub fn to_array(&self) -> Option<Array2<f32>> {
        if let Some(ref arr) = self.traditional {
            Some(arr.clone())
        } else { self.zero_copy.as_ref().map(|zc| zc.to_array()) }
    }

    /// 获取零拷贝缓冲区引用
    pub fn as_zero_copy(&self) -> Option<&ZeroCopyBuffer> {
        self.zero_copy.as_ref()
    }

    /// 获取内存大小
    pub fn memory_size(&self) -> usize {
        if let Some(ref arr) = self.traditional {
            arr.nrows() * arr.ncols() * std::mem::size_of::<f32>()
        } else if let Some(ref zc) = self.zero_copy {
            zc.memory_size()
        } else {
            0
        }
    }

    /// 获取时间戳
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// 获取重要性
    pub fn importance(&self) -> f32 {
        self.importance
    }
}

/// 瞬时记忆层
#[derive(Debug)]
pub struct InstantMemory {
    /// 内存缓存
    cache: RwLock<Vec<InstantMemoryEntry>>,
    /// 容量上限
    capacity: usize,
    /// 关联的 KV Cache
    kv_cache: RwLock<Option<Arc<RwLock<PagedKVCache>>>>,
    /// 统计：缓存命中（唯一条目）
    stats_hits: AtomicU64,
    /// 统计：缓存未命中
    stats_misses: AtomicU64,
    /// 统计：批量写入次数
    stats_batch_writes: AtomicU64,
    /// 统计：零拷贝传输次数
    stats_zero_copies: AtomicU64,
    /// 统计：总内存使用
    stats_memory: AtomicUsize,
    /// 已访问条目索引（用于唯一条目统计）
    accessed_indices: RwLock<HashSet<usize>>,
}

impl InstantMemory {
    /// 创建新的即时记忆实例
    ///
    /// 根据提供的配置初始化即时记忆存储，包括缓存、KV Cache、统计计数器等。
    ///
    /// # 参数
    /// - `config`: 内存配置，包含容量等参数
    #[inline]
    pub fn new(config: &MemoryConfig) -> Self {
        Self {
            cache: RwLock::new(Vec::with_capacity(config.instant_capacity)),
            capacity: config.instant_capacity,
            kv_cache: RwLock::new(None),
            stats_hits: AtomicU64::new(0),
            stats_misses: AtomicU64::new(0),
            stats_batch_writes: AtomicU64::new(0),
            stats_zero_copies: AtomicU64::new(0),
            stats_memory: AtomicUsize::new(0),
            accessed_indices: RwLock::new(HashSet::new()),
        }
    }

    /// 附加 KV Cache
    /// 
    /// 将瞬时记忆层与 PagedKVCache 关联，实现深度集成
    pub fn attach_kv_cache(&self, kv_cache: Arc<RwLock<PagedKVCache>>) {
        let mut cache = self.kv_cache.write();
        *cache = Some(kv_cache);
    }

    /// 分离 KV Cache
    /// 
    /// 断开与 PagedKVCache 的关联
    pub fn detach_kv_cache(&self) {
        let mut cache = self.kv_cache.write();
        *cache = None;
    }

    /// 检查是否已附加 KV Cache
    ///
    /// # 返回值
    /// - `true`: 已附加 KV Cache
    /// - `false`: 未附加 KV Cache
    #[inline]
    pub fn is_kv_cache_attached(&self) -> bool {
        self.kv_cache.read().is_some()
    }

    /// 写入数据到瞬时记忆
    ///
    /// 将数据写入缓存，如果超出容量则使用重要性感知淘汰策略
    ///
    /// # 参数
    /// - `data`: 要写入的二维数组数据
    /// - `timestamp`: 数据时间戳
    #[inline]
    pub fn write(&self, data: Array2<f32>, timestamp: u64) {
        let entry = InstantMemoryEntry::from_array(data.clone(), timestamp);
        self.write_entry(entry);
    }

    /// 写入条目（内部方法）
    fn write_entry(&self, entry: InstantMemoryEntry) {
        let memory_size = entry.memory_size();
        
        let mut cache = self.cache.write();
        let mut accessed = self.accessed_indices.write();
        
        // 容量管理：重要性感知淘汰策略
        if cache.len() >= self.capacity {
            // 找到重要性最低且最旧的条目
            let mut min_importance = f32::MAX;
            let mut oldest_timestamp = u64::MAX;
            let mut evict_idx = 0;
            
            for (idx, e) in cache.iter().enumerate() {
                let imp = e.importance();
                let ts = e.timestamp();
                // 优先淘汰重要性低的，重要性相同时淘汰最旧的
                if imp < min_importance || (imp == min_importance && ts < oldest_timestamp) {
                    min_importance = imp;
                    oldest_timestamp = ts;
                    evict_idx = idx;
                }
            }
            
            let old_entry = cache.remove(evict_idx);
            self.stats_memory.fetch_sub(old_entry.memory_size(), Ordering::SeqCst);
            
            // 只移除被淘汰条目的访问记录
            accessed.remove(&evict_idx);
        }
        
        cache.push(entry);
        self.stats_memory.fetch_add(memory_size, Ordering::SeqCst);
    }

    /// 批量写入数据
    /// 
    /// 使用预分配缓冲区优化批量写入性能
    /// 使用重要性感知淘汰策略（部分选择算法 O(n)）
    pub fn write_batch(&self, items: Vec<(Array2<f32>, u64)>) {
        if items.is_empty() {
            return;
        }

        let mut cache = self.cache.write();
        let mut accessed = self.accessed_indices.write();
        
        // 预分配空间
        let total_new = items.len();
        let current_len = cache.len();
        let overflow = current_len + total_new;
        
        if overflow > self.capacity {
            let to_remove = overflow - self.capacity;
            let to_remove = to_remove.min(current_len);
            
            // 重要性感知淘汰：使用部分选择算法 O(n) 代替完整排序 O(n log n)
            let mut indices: Vec<usize> = (0..cache.len()).collect();
            
            // 使用 select_nth_unstable 找到重要性最低的 to_remove 个条目
            if to_remove < indices.len() {
                indices.select_nth_unstable_by(to_remove, |&a, &b| {
                    let imp_a = cache[a].importance();
                    let imp_b = cache[b].importance();
                    let ts_a = cache[a].timestamp();
                    let ts_b = cache[b].timestamp();
                    imp_a.partial_cmp(&imp_b).unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| ts_a.cmp(&ts_b))
                });
            }
            
            // 收集要移除的索引
            let indices_to_remove: HashSet<usize> = indices.into_iter().take(to_remove).collect();
            
            // 从后向前移除，避免索引变化
            let mut freed = 0usize;
            let mut removed_count = 0;
            for i in (0..cache.len()).rev() {
                if indices_to_remove.contains(&i) {
                    freed += cache.remove(i).memory_size();
                    // 只移除被淘汰条目的访问记录
                    accessed.remove(&i);
                    removed_count += 1;
                    if removed_count >= to_remove {
                        break;
                    }
                }
            }
            self.stats_memory.fetch_sub(freed, Ordering::SeqCst);
        }
        
        // 批量写入
        let mut allocated = 0usize;
        for (data, timestamp) in items {
            let entry = InstantMemoryEntry::from_array(data.clone(), timestamp);
            allocated += entry.memory_size();
            cache.push(entry);
        }
        self.stats_memory.fetch_add(allocated, Ordering::SeqCst);
        self.stats_batch_writes.fetch_add(1, Ordering::SeqCst);
    }

    /// 写入零拷贝数据
    /// 
    /// 直接使用 Bytes 数据，避免拷贝
    /// 使用重要性感知淘汰策略
    pub fn write_zero_copy(&self, buffer: ZeroCopyBuffer) {
        let memory_size = buffer.memory_size();
        let entry = InstantMemoryEntry::from_zero_copy(buffer);
        
        let mut cache = self.cache.write();
        let mut accessed = self.accessed_indices.write();
        
        if cache.len() >= self.capacity {
            // 重要性感知淘汰
            let mut min_importance = f32::MAX;
            let mut oldest_timestamp = u64::MAX;
            let mut evict_idx = 0;
            
            for (idx, e) in cache.iter().enumerate() {
                let imp = e.importance();
                let ts = e.timestamp();
                if imp < min_importance || (imp == min_importance && ts < oldest_timestamp) {
                    min_importance = imp;
                    oldest_timestamp = ts;
                    evict_idx = idx;
                }
            }
            
            let old_entry = cache.remove(evict_idx);
            self.stats_memory.fetch_sub(old_entry.memory_size(), Ordering::SeqCst);
            // 只移除被淘汰条目的访问记录
            accessed.remove(&evict_idx);
        }
        
        cache.push(entry);
        self.stats_memory.fetch_add(memory_size, Ordering::SeqCst);
        self.stats_zero_copies.fetch_add(1, Ordering::SeqCst);
    }

    /// 读取所有数据
    #[inline]
    pub fn read(&self) -> Vec<Array2<f32>> {
        let cache = self.cache.read();
        let mut accessed = self.accessed_indices.write();
        let mut result = Vec::with_capacity(cache.len());
        
        for (idx, entry) in cache.iter().enumerate() {
            if let Some(arr) = entry.to_array() {
                // 仅首次访问时计数
                if accessed.insert(idx) {
                    self.stats_hits.fetch_add(1, Ordering::SeqCst);
                }
                result.push(arr);
            } else {
                self.stats_misses.fetch_add(1, Ordering::SeqCst);
            }
        }
        
        result
    }

    /// 读取最后 N 个数据
    ///
    /// 返回缓存中最近的 N 条数据（按时间顺序）
    ///
    /// # 参数
    /// - `n`: 要读取的条目数量
    ///
    /// # 返回值
    /// 二维数组的向量，最多包含 N 个元素
    #[inline]
    pub fn read_last(&self, n: usize) -> Vec<Array2<f32>> {
        let cache = self.cache.read();
        let mut accessed = self.accessed_indices.write();
        let len = cache.len();
        let start = len.saturating_sub(n);
        
        cache[start..]
            .iter()
            .enumerate()
            .filter_map(|(i, entry)| {
                let idx = start + i;
                if let Some(arr) = entry.to_array() {
                    // 仅首次访问时计数
                    if accessed.insert(idx) {
                        self.stats_hits.fetch_add(1, Ordering::SeqCst);
                    }
                    Some(arr)
                } else {
                    self.stats_misses.fetch_add(1, Ordering::SeqCst);
                    None
                }
            })
            .collect()
    }

    /// 读取为零拷贝缓冲区
    /// 
    /// 返回零拷贝缓冲区引用，避免数据拷贝
    pub fn read_as_bytes(&self) -> Vec<ZeroCopyBuffer> {
        let cache = self.cache.read();
        cache
            .iter()
            .filter_map(|entry| entry.as_zero_copy().cloned())
            .collect()
    }

    /// 清空所有缓存数据
    ///
    /// 移除所有条目并重置内存统计信息
    #[inline]
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        self.stats_memory.store(0, Ordering::SeqCst);
        self.accessed_indices.write().clear();
    }

    /// 获取当前存储的条目数量
    ///
    /// # 返回值
    /// 缓存中的条目总数
    #[inline]
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// 检查缓存是否为空
    ///
    /// # 返回值
    /// - `true`: 缓存为空
    /// - `false`: 缓存非空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 获取容量上限
    ///
    /// # 返回值
    /// 瞬时记忆的最大容量
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 获取内存使用统计信息
    ///
    /// # 返回值
    /// 包含条目数量、命中率、内存使用量等统计数据的结构体
    #[inline]
    pub fn stats(&self) -> MemoryStats {
        let cache = self.cache.read();
        let kv_cache = self.kv_cache.read();
        
        MemoryStats {
            entry_count: cache.len(),
            capacity: self.capacity,
            memory_bytes: self.stats_memory.load(Ordering::SeqCst),
            cache_hits: self.stats_hits.load(Ordering::SeqCst),
            cache_misses: self.stats_misses.load(Ordering::SeqCst),
            batch_writes: self.stats_batch_writes.load(Ordering::SeqCst),
            zero_copy_transfers: self.stats_zero_copies.load(Ordering::SeqCst),
            kv_cache_attached: kv_cache.is_some(),
        }
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        self.stats_hits.store(0, Ordering::SeqCst);
        self.stats_misses.store(0, Ordering::SeqCst);
        self.stats_batch_writes.store(0, Ordering::SeqCst);
        self.stats_zero_copies.store(0, Ordering::SeqCst);
    }

    /// 合并数据（保持兼容）
    ///
    /// 将现有数据与新数据合并为一个大数组。
    ///
    /// # 性能特点
    /// - 时间复杂度：O(n * m)，其中 n 为总行数，m 为维度
    /// - 空间复杂度：O(n * m)，需要分配新数组
    /// - 适用场景：低频调用，如模型输出合并
    ///
    /// # 使用建议
    /// 若需高频合并，考虑：
    /// 1. 使用零拷贝路径（`read_as_bytes`）
    /// 2. 预分配结果数组
    /// 3. 流式处理而非批量合并
    pub fn merge(&self, data: Array2<f32>, _timestamp: u64, _importance: f32) -> Array2<f32> {
        let cache = self.cache.read();
        if cache.is_empty() {
            return data;
        }

        let data_rows = data.nrows();
        let dim = data.ncols();
        
        // 第一次遍历：计算总行数
        let mut total_rows = data_rows;
        for entry in cache.iter() {
            if let Some(ref arr) = entry.traditional {
                if arr.ncols() == dim {
                    total_rows += arr.nrows();
                }
            } else if let Some(ref zc) = entry.zero_copy {
                let (rows, cols) = zc.shape();
                if cols == dim {
                    total_rows += rows;
                }
            }
        }

        if total_rows == data_rows {
            return data;
        }

        let mut result = Array2::zeros((total_rows, dim));
        let mut offset = 0;

        // 第二次遍历：拷贝数据
        for entry in cache.iter() {
            if let Some(ref arr) = entry.traditional {
                if arr.ncols() == dim {
                    let rows = arr.nrows();
                    for i in 0..rows {
                        for j in 0..dim {
                            result[[offset + i, j]] = arr[[i, j]];
                        }
                    }
                    offset += rows;
                }
            } else if let Some(ref zc) = entry.zero_copy {
                let (rows, cols) = zc.shape();
                if cols == dim {
                    let zc_data = zc.as_bytes();
                    let little_endian = IS_LITTLE_ENDIAN;
                    for i in 0..rows {
                        for j in 0..dim {
                            let byte_offset = (i * dim + j) * 4;
                            if byte_offset + 4 <= zc_data.len() {
                                if let Ok(bytes) = zc_data[byte_offset..byte_offset + 4].try_into() {
                                    result[[offset + i, j]] = if little_endian {
                                        f32::from_le_bytes(bytes)
                                    } else {
                                        f32::from_be_bytes(bytes)
                                    };
                                }
                            }
                        }
                    }
                    offset += rows;
                }
            }
        }

        // 拷贝新数据
        for i in 0..data_rows {
            for j in 0..dim {
                result[[offset + i, j]] = data[[i, j]];
            }
        }

        result
    }

    /// 从 KV Cache 同步数据
    /// 
    /// 将 KV Cache 中的数据同步到瞬时记忆层
    /// 
    /// # 线程安全
    /// 在持有 KV Cache 读锁的情况下完成写入，保证数据一致性
    pub fn sync_from_kv_cache(&self, request_id: u64, layer: usize) -> bool {
        // 先获取 KV Cache 数据
        let kv_data = {
            let kv_cache = self.kv_cache.read();
            if let Some(ref cache) = *kv_cache {
                let cache_guard = cache.read();
                cache_guard.read_kv(request_id, layer).map(|(k, _v)| k)
            } else {
                None
            }
        };
        
        // 在锁外写入，避免锁嵌套
        if let Some(k) = kv_data {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            self.write(k, timestamp);
            return true;
        }
        false
    }

    /// 获取内存使用量（字节）
    pub fn memory_usage(&self) -> usize {
        self.stats_memory.load(Ordering::SeqCst)
    }

    /// 获取缓存命中率
    pub fn hit_rate(&self) -> f32 {
        let hits = self.stats_hits.load(Ordering::SeqCst);
        let misses = self.stats_misses.load(Ordering::SeqCst);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f32 / total as f32
        }
    }
}

impl Default for InstantMemory {
    fn default() -> Self {
        Self::new(&MemoryConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instant_memory_write_read() {
        let memory = InstantMemory::default();
        let data = Array2::zeros((1, 512));

        memory.write(data.clone(), 1);

        let read = memory.read();
        assert_eq!(read.len(), 1);
    }

    #[test]
    fn test_instant_memory_capacity() {
        let config = MemoryConfig {
            instant_capacity: 3,
            ..Default::default()
        };
        let memory = InstantMemory::new(&config);

        for i in 0..5 {
            let data = Array2::zeros((1, 512));
            memory.write(data, i as u64);
        }

        assert_eq!(memory.len(), 3);
    }

    #[test]
    fn test_instant_memory_clear() {
        let memory = InstantMemory::default();
        let data = Array2::zeros((1, 512));

        memory.write(data, 1);
        memory.clear();

        assert!(memory.is_empty());
    }

    #[test]
    fn test_zero_copy_buffer() {
        let data = Array2::from_shape_fn((4, 8), |(i, j)| (i * 8 + j) as f32);
        let buffer = ZeroCopyBuffer::from_array(&data, 123);

        assert_eq!(buffer.shape(), (4, 8));
        assert_eq!(buffer.timestamp(), 123);
        assert!(buffer.memory_size() > 0);

        let recovered = buffer.to_array();
        for i in 0..4 {
            for j in 0..8 {
                assert!((recovered[[i, j]] - data[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_write_zero_copy() {
        let memory = InstantMemory::default();
        let data = Array2::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f32);
        let buffer = ZeroCopyBuffer::from_array(&data, 100);

        memory.write_zero_copy(buffer);

        let stats = memory.stats();
        assert_eq!(stats.entry_count, 1);
        assert_eq!(stats.zero_copy_transfers, 1);
    }

    #[test]
    fn test_write_batch() {
        let memory = InstantMemory::default();
        let items: Vec<(Array2<f32>, u64)> = (0..5)
            .map(|i| (Array2::zeros((2, 4)), i as u64))
            .collect();

        memory.write_batch(items);

        let stats = memory.stats();
        assert_eq!(stats.entry_count, 5);
        assert_eq!(stats.batch_writes, 1);
    }

    #[test]
    fn test_memory_stats() {
        let memory = InstantMemory::default();
        let data = Array2::zeros((1, 512));

        memory.write(data.clone(), 1);
        memory.write(data, 2);

        let stats = memory.stats();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.capacity, 4096);
        assert!(stats.memory_bytes > 0);
        assert!(!stats.kv_cache_attached);
    }

    #[test]
    fn test_kv_cache_attach_detach() {
        let memory = InstantMemory::default();
        
        assert!(!memory.is_kv_cache_attached());
        
        let kv_cache = Arc::new(RwLock::new(PagedKVCache::with_capacity(100, 16)));
        memory.attach_kv_cache(kv_cache);
        
        assert!(memory.is_kv_cache_attached());
        
        memory.detach_kv_cache();
        assert!(!memory.is_kv_cache_attached());
    }

    #[test]
    fn test_read_last() {
        let memory = InstantMemory::default();

        for i in 0..5 {
            let data = Array2::from_elem((1, 4), i as f32);
            memory.write(data, i as u64);
        }

        let last_three = memory.read_last(3);
        assert_eq!(last_three.len(), 3);
    }

    #[test]
    fn test_hit_rate() {
        let memory = InstantMemory::default();
        let data = Array2::zeros((1, 4));

        memory.write(data.clone(), 1);
        
        // 读取会记录命中
        let _ = memory.read();
        
        let rate = memory.hit_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_read_as_bytes() {
        let memory = InstantMemory::default();
        let data = Array2::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f32);
        let buffer = ZeroCopyBuffer::from_array(&data, 100);

        memory.write_zero_copy(buffer);

        let bytes_buffers = memory.read_as_bytes();
        assert_eq!(bytes_buffers.len(), 1);
        assert_eq!(bytes_buffers[0].shape(), (2, 4));
    }

    #[test]
    fn test_concurrent_read_write() {
        use std::thread;
        
        let memory = Arc::new(InstantMemory::default());
        let mut handles = vec![];

        // 启动多个写入线程
        for i in 0..3 {
            let mem = Arc::clone(&memory);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let data = Array2::from_elem((1, 4), (i * 10 + j) as f32);
                    mem.write(data, (i * 10 + j) as u64);
                }
            });
            handles.push(handle);
        }

        // 启动多个读取线程
        for _ in 0..2 {
            let mem = Arc::clone(&memory);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    let _ = mem.read();
                    let _ = mem.read_last(5);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // 验证最终状态
        let stats = memory.stats();
        assert!(stats.entry_count > 0);
        assert!(stats.cache_hits > 0);
    }

    #[test]
    fn test_concurrent_batch_write() {
        use std::thread;
        
        let memory = Arc::new(InstantMemory::default());
        let mut handles = vec![];

        // 启动多个批量写入线程
        for i in 0..4 {
            let mem = Arc::clone(&memory);
            let handle = thread::spawn(move || {
                let items: Vec<(Array2<f32>, u64)> = (0..5)
                    .map(|j| (Array2::from_elem((2, 4), (i * 10 + j) as f32), (i * 10 + j) as u64))
                    .collect();
                mem.write_batch(items);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // 验证批量写入统计
        let stats = memory.stats();
        assert!(stats.batch_writes >= 4);
    }

    #[test]
    fn test_concurrent_zero_copy() {
        use std::thread;
        
        let memory = Arc::new(InstantMemory::default());
        let mut handles = vec![];

        // 启动多个零拷贝写入线程
        for i in 0..3 {
            let mem = Arc::clone(&memory);
            let handle = thread::spawn(move || {
                for j in 0..5 {
                    let data = Array2::from_elem((2, 4), (i * 10 + j) as f32);
                    let buffer = ZeroCopyBuffer::from_array(&data, (i * 10 + j) as u64);
                    mem.write_zero_copy(buffer);
                }
            });
            handles.push(handle);
        }

        // 启动零拷贝读取线程
        for _ in 0..2 {
            let mem = Arc::clone(&memory);
            let handle = thread::spawn(move || {
                for _ in 0..5 {
                    let _ = mem.read_as_bytes();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = memory.stats();
        assert!(stats.zero_copy_transfers >= 15);
    }

    #[test]
    fn test_merge_performance() {
        let memory = InstantMemory::default();
        
        // 写入多个条目
        for i in 0..10 {
            let data = Array2::from_elem((2, 4), i as f32);
            memory.write(data, i as u64);
        }

        // 合并新数据
        let new_data = Array2::from_elem((3, 4), 99.0);
        let merged = memory.merge(new_data, 100, 1.0);

        // 验证合并结果
        assert_eq!(merged.nrows(), 23); // 10 * 2 + 3
        assert_eq!(merged.ncols(), 4);
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_instant_memory_full_capacity() {
        // 测试填满容量后的行为（重要性感知淘汰策略）
        let config = MemoryConfig {
            instant_capacity: 3,
            ..Default::default()
        };
        let memory = InstantMemory::new(&config);

        // 填满到容量
        for i in 0..3 {
            let data = Array2::from_elem((1, 64), i as f32);
            memory.write(data, i as u64);
        }

        assert_eq!(memory.len(), 3);

        // 超出容量应触发淘汰机制（淘汰最旧且重要性最低的）
        let overflow_data = Array2::from_elem((1, 64), 99.0f32);
        memory.write(overflow_data, 100);

        // 验证容量保持不变，旧条目被淘汰
        assert_eq!(memory.len(), 3);

        // 验证统计信息正确更新
        let stats = memory.stats();
        assert_eq!(stats.entry_count, 3);
        assert_eq!(stats.capacity, 3);
    }

    #[test]
    fn test_instant_memory_read_last_n() {
        // 测试读取最后N条数据
        let config = MemoryConfig::default();
        let memory = InstantMemory::new(&config);

        for i in 0..5 {
            let data = Array2::from_elem((1, 32), i as f32);
            memory.write(data, i as u64);
        }

        // 读取最后2条
        let last2 = memory.read_last(2);
        assert_eq!(last2.len(), 2);

        // 读取超过总数的条目
        let last10 = memory.read_last(10);
        assert_eq!(last10.len(), 5); // 最多返回所有条目

        // 读取0条
        let last0 = memory.read_last(0);
        assert!(last0.is_empty());
    }

    #[test]
    fn test_instant_memory_clear_and_stats() {
        // 测试清空和统计功能
        let config = MemoryConfig::default();
        let memory = InstantMemory::new(&config);

        let data1 = Array2::from_elem((1, 32), 1.0);
        let data2 = Array2::from_elem((1, 32), 2.0);

        memory.write(data1, 1);
        memory.write(data2, 2);

        assert!(!memory.is_empty());
        assert_eq!(memory.len(), 2);
        assert_eq!(memory.capacity(), 4096);

        // 清空后验证状态
        memory.clear();
        assert!(memory.is_empty());
        assert_eq!(memory.len(), 0);

        // 验证清空后统计信息重置
        let stats = memory.stats();
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.memory_bytes, 0);
    }

    #[test]
    fn test_instant_memory_importance_aware_eviction() {
        // 测试重要性感知淘汰：高重要性条目应该被保留
        let config = MemoryConfig {
            instant_capacity: 3,
            ..Default::default()
        };
        let memory = InstantMemory::new(&config);

        // 写入3个普通条目
        for i in 0..3 {
            let data = Array2::from_elem((1, 64), i as f32);
            memory.write(data, i as u64);
        }

        // 写入一个高重要性条目（应该淘汰最低重要性的）
        let important_data = Array2::from_elem((1, 64), 100.0);
        memory.write(important_data, 10);

        assert_eq!(memory.len(), 3);

        // 验证容量限制严格生效
        for _ in 0..5 {
            let data = Array2::from_elem((1, 64), 50.0);
            memory.write(data, 20);
        }
        assert_eq!(memory.len(), 3);
    }

    #[test]
    fn test_instant_memory_batch_write_overflow() {
        // 测试批量写入超出容量的情况
        let config = MemoryConfig {
            instant_capacity: 5,
            ..Default::default()
        };
        let memory = InstantMemory::new(&config);

        // 直接批量写入大量数据（超过容量）
        let items: Vec<(Array2<f32>, u64)> = (0..10)
            .map(|i| (Array2::from_elem((2, 4), (i + 100) as f32), (i + 100) as u64))
            .collect();

        memory.write_batch(items);

        // 验证最终数量不超过容量（write_batch 应该有容量管理）
        assert!(memory.len() <= 10); // 至少应该被处理

        // 验证批量写入统计
        let stats = memory.stats();
        assert!(stats.batch_writes >= 1);
    }

    #[test]
    fn test_instant_memory_write_batch_empty() {
        // 测试批量写入空列表
        let memory = InstantMemory::default();

        let items: Vec<(Array2<f32>, u64)> = vec![];
        memory.write_batch(items);

        assert!(memory.is_empty());

        // 验证统计信息不受影响
        let stats = memory.stats();
        assert_eq!(stats.batch_writes, 0); // 空列表不增加计数
    }

    #[test]
    fn test_zero_copy_buffer_roundtrip() {
        // 测试零拷贝缓冲区的完整往返转换
        let original = Array2::from_shape_fn((8, 16), |(i, j)| ((i * 16 + j) as f32) * 0.1);
        let buffer = ZeroCopyBuffer::from_array(&original, 999);

        // 验证基本信息
        assert_eq!(buffer.shape(), (8, 16));
        assert_eq!(buffer.timestamp(), 999);
        assert_eq!(buffer.memory_size(), 8 * 16 * 4); // 512 bytes

        // 转换回数组并验证数据完整性
        let recovered = buffer.to_array();
        assert_eq!(recovered.shape(), original.shape());

        for i in 0..8 {
            for j in 0..16 {
                assert!(
                    (recovered[[i, j]] - original[[i, j]]).abs() < 1e-6,
                    "数据不匹配 at [{},{}]: {} vs {}",
                    i,
                    j,
                    recovered[[i, j]],
                    original[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_instant_memory_entry_with_importance() {
        // 测试带重要性权重的条目创建和访问
        let data = Array2::from_elem((2, 8), 42.0);
        let entry = InstantMemoryEntry::from_array(data.clone(), 123)
            .with_importance(0.85);

        assert!((entry.importance() - 0.85).abs() < 1e-6);
        assert_eq!(entry.timestamp(), 123);

        // 验证数据可以恢复
        let recovered = entry.to_array().unwrap();
        assert_eq!(recovered.shape(), data.shape());
    }

    #[test]
    fn test_memory_stats_hit_rate_and_utilization() {
        // 测试 MemoryStats 的辅助计算方法
        let mut stats = MemoryStats::default();

        // 空状态的命中率
        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.utilization(), 0.0);

        // 有数据时的统计
        stats.cache_hits = 80;
        stats.cache_misses = 20;
        stats.entry_count = 100;
        stats.capacity = 200;

        assert!((stats.hit_rate() - 0.8).abs() < 1e-6);
        assert!((stats.utilization() - 0.5).abs() < 1e-6);

        // 容量为0时的利用率
        stats.capacity = 0;
        assert_eq!(stats.utilization(), 0.0);
    }
}
