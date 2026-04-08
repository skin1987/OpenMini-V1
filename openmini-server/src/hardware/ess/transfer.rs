//! ESS 传输模块
//!
//! 注意：FlashTransfer 为模拟实现，仅进行内存复制操作，
//! 未体现真实硬件传输特性（如 DMA、PCIe 带宽限制）。
//! 生产环境应替换为真实的 GPU 传输 API（CUDA、Vulkan 等）。

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::collections::VecDeque;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    ToGPU,
    ToCPU,
}

#[derive(Debug, Clone)]
pub struct TransferTask {
    pub id: u64,
    pub data: Vec<u8>,
    pub direction: TransferDirection,
    pub timestamp: Instant,
}

/// 模拟的闪存传输器
///
/// 注意：此实现仅为模拟，所有传输操作都是简单的内存复制。
/// 未模拟真实硬件特性如：
/// - DMA 传输
/// - PCIe 带宽限制
/// - GPU 直接内存访问
pub struct FlashTransfer {
    batch_size: usize,
    min_chunk_size: usize,
    max_bandwidth_gbps: f64,
}

impl FlashTransfer {
    pub fn new() -> Self {
        Self {
            batch_size: 4096,
            min_chunk_size: 256,
            max_bandwidth_gbps: 32.0,
        }
    }

    pub fn with_config(batch_size: usize, min_chunk_size: usize) -> Self {
        Self {
            batch_size,
            min_chunk_size,
            max_bandwidth_gbps: 32.0,
        }
    }

    /// 模拟传输到设备（实际为内存复制）
    pub fn transfer_to_device(&self, data: &[u8]) -> Option<Vec<u8>> {
        Some(data.to_vec())
    }

    /// 模拟传输到主机（实际为内存复制）
    pub fn transfer_to_host(&self, data: &[u8]) -> Option<Vec<u8>> {
        Some(data.to_vec())
    }

    /// 模拟 UVA 传输（实际为内存复制）
    pub fn uva_transfer(&self, src: &[u8], dst: &mut [u8]) -> usize {
        let transfer_size = src.len().min(dst.len());
        dst[..transfer_size].copy_from_slice(&src[..transfer_size]);
        transfer_size
    }

    /// 模拟批量传输（实际为内存复制）
    pub fn batch_transfer(&self, chunks: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        chunks.into_iter().collect()
    }

    /// 计算最优块大小
    pub fn optimal_chunk_size(&self, total_size: usize) -> usize {
        if total_size < self.min_chunk_size * 4 {
            self.min_chunk_size
        } else if total_size < self.batch_size * 16 {
            self.batch_size
        } else {
            self.batch_size * 4
        }
    }

    /// 估算传输时间（基于理论带宽）
    pub fn estimate_transfer_time(&self, size: usize) -> f64 {
        let size_bits = (size * 8) as f64;
        size_bits / (self.max_bandwidth_gbps * 1e9) * 1000.0
    }
}

impl Default for FlashTransfer {
    fn default() -> Self {
        Self::new()
    }
}

/// 传输管理器
///
/// 同步执行模型：`process_batch` 执行传输并立即完成。
/// `batch_size` 限制每次调用处理的任务数量。
pub struct TransferManager {
    batch_size: usize,
    pending_transfers: Arc<RwLock<VecDeque<TransferTask>>>,
    completed_transfers: Arc<RwLock<VecDeque<TransferTask>>>,
    transfer_id: Arc<AtomicU64>,
}

impl TransferManager {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            pending_transfers: Arc::new(RwLock::new(VecDeque::new())),
            completed_transfers: Arc::new(RwLock::new(VecDeque::new())),
            transfer_id: Arc::new(AtomicU64::new(0)),
        }
    }

    /// 提交传输任务，返回任务 ID
    pub fn submit(&self, data: Vec<u8>, direction: TransferDirection) -> u64 {
        let id = self.transfer_id.fetch_add(1, Ordering::SeqCst) + 1;

        let task = TransferTask {
            id,
            data,
            direction,
            timestamp: Instant::now(),
        };

        self.pending_transfers.write().unwrap().push_back(task);
        id
    }

    /// 处理一批传输任务
    ///
    /// 同步执行模型：获取任务、执行传输、立即完成。
    /// `batch_size` 限制每次调用处理的任务数量。
    pub fn process_batch(&self) -> Vec<u64> {
        let mut completed_ids = Vec::new();
        let mut processed = 0;
        
        while processed < self.batch_size {
            let task = {
                let mut pending = self.pending_transfers.write().unwrap();
                pending.pop_front()
            };
            
            if let Some(task) = task {
                completed_ids.push(task.id);
                
                {
                    let mut completed = self.completed_transfers.write().unwrap();
                    completed.push_back(task);
                }
                
                processed += 1;
            } else {
                break;
            }
        }
        
        completed_ids
    }

    /// 获取已完成的传输任务
    pub fn get_completed(&self) -> Vec<TransferTask> {
        let mut completed = self.completed_transfers.write().unwrap();
        let mut result = Vec::new();
        
        while let Some(task) = completed.pop_front() {
            result.push(task);
        }
        
        result
    }

    /// 获取待处理任务数量
    pub fn pending_count(&self) -> usize {
        self.pending_transfers.read().unwrap().len()
    }
}

impl Default for TransferManager {
    fn default() -> Self {
        Self::new(4096)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_transfer_basic() {
        let transfer = FlashTransfer::new();
        
        let data = vec![1, 2, 3, 4, 5];
        let result = transfer.transfer_to_device(&data);
        assert_eq!(result, Some(data.clone()));
        
        let result = transfer.transfer_to_host(&data);
        assert_eq!(result, Some(data));
    }

    #[test]
    fn test_uva_transfer() {
        let transfer = FlashTransfer::new();
        
        let src = vec![1, 2, 3, 4, 5];
        let mut dst = vec![0u8; 5];
        
        let size = transfer.uva_transfer(&src, &mut dst);
        assert_eq!(size, 5);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_optimal_chunk_size() {
        let transfer = FlashTransfer::new();
        
        assert_eq!(transfer.optimal_chunk_size(100), 256);
        assert_eq!(transfer.optimal_chunk_size(10000), 4096);
        assert_eq!(transfer.optimal_chunk_size(1000000), 16384);
    }

    #[test]
    fn test_transfer_manager_submit() {
        let manager = TransferManager::new(10);
        
        let id1 = manager.submit(vec![1, 2, 3], TransferDirection::ToGPU);
        let id2 = manager.submit(vec![4, 5, 6], TransferDirection::ToCPU);
        
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(manager.pending_count(), 2);
    }

    #[test]
    fn test_transfer_manager_process_batch() {
        let manager = TransferManager::new(10);
        
        manager.submit(vec![1, 2, 3], TransferDirection::ToGPU);
        manager.submit(vec![4, 5, 6], TransferDirection::ToCPU);
        
        let completed = manager.process_batch();
        assert_eq!(completed.len(), 2);
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_transfer_manager_batch_limit() {
        let manager = TransferManager::new(2);
        
        for i in 0..5 {
            manager.submit(vec![i as u8], TransferDirection::ToGPU);
        }
        
        let completed = manager.process_batch();
        assert_eq!(completed.len(), 2);
        assert_eq!(manager.pending_count(), 3);
    }

    #[test]
    fn test_transfer_manager_get_completed() {
        let manager = TransferManager::new(10);
        
        manager.submit(vec![1, 2, 3], TransferDirection::ToGPU);
        manager.process_batch();
        
        let completed = manager.get_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].data, vec![1, 2, 3]);
        
        let completed2 = manager.get_completed();
        assert!(completed2.is_empty());
    }

    // 新增分支覆盖测试

    /// 测试 transfer_to_device 的空数据分支
    #[test]
    fn test_transfer_to_device_empty() {
        let transfer = FlashTransfer::new();
        let data: Vec<u8> = vec![];
        let result = transfer.transfer_to_device(&data);
        assert_eq!(result, Some(vec![]));
    }

    /// 测试 transfer_to_host 的空数据分支
    #[test]
    fn test_transfer_to_host_empty() {
        let transfer = FlashTransfer::new();
        let data: Vec<u8> = vec![];
        let result = transfer.transfer_to_host(&data);
        assert_eq!(result, Some(vec![]));
    }

    /// 测试 uva_transfer 的 src 为空分支
    #[test]
    fn test_uva_transfer_empty_src() {
        let transfer = FlashTransfer::new();
        let src: Vec<u8> = vec![];
        let mut dst = vec![0u8; 5];
        
        let size = transfer.uva_transfer(&src, &mut dst);
        assert_eq!(size, 0);
        // dst 不应被修改
        assert_eq!(dst, vec![0u8; 5]);
    }

    /// 测试 uva_transfer 的 dst 比 src 小（取最小值）
    #[test]
    fn test_uva_transfer_dst_smaller() {
        let transfer = FlashTransfer::new();
        let src = vec![1, 2, 3, 4, 5];
        let mut dst = vec![0u8; 3];
        
        let size = transfer.uva_transfer(&src, &mut dst);
        assert_eq!(size, 3); // 取 min(5, 3)
        assert_eq!(dst, vec![1, 2, 3]);
    }

    /// 测试 batch_transfer 的空列表分支
    #[test]
    fn test_batch_transfer_empty() {
        let transfer = FlashTransfer::new();
        let chunks: Vec<Vec<u8>> = vec![];
        let result = transfer.batch_transfer(chunks);
        assert!(result.is_empty());
    }

    /// 测试 optimal_chunk_size 的边界值（刚好等于 min_chunk_size * 4）
    #[test]
    fn test_optimal_chunk_size_boundary_low() {
        let transfer = FlashTransfer::new();
        // min_chunk_size * 4 = 256 * 4 = 1024
        // total_size >= 1024 时进入第二个分支
        assert_eq!(transfer.optimal_chunk_size(1023), 256); // < 1024
        assert_eq!(transfer.optimal_chunk_size(1024), 4096); // >= 1024 且 < 65536
    }

    /// 测试 optimal_chunk_size 的边界值（刚好等于 batch_size * 16）
    #[test]
    fn test_optimal_chunk_size_boundary_high() {
        let transfer = FlashTransfer::new();
        // batch_size * 16 = 4096 * 16 = 65536
        assert_eq!(transfer.optimal_chunk_size(65535), 4096); // < batch_size * 16
        assert_eq!(transfer.optimal_chunk_size(65536), 16384); // >= batch_size * 16
    }

    /// 测试 estimate_transfer_time 的零大小分支
    #[test]
    fn test_estimate_transfer_time_zero() {
        let transfer = FlashTransfer::new();
        let time = transfer.estimate_transfer_time(0);
        assert_eq!(time, 0.0, "零大小的传输时间应为 0");
    }

    /// 测试 estimate_transfer_time 的正常值
    #[test]
    fn test_estimate_transfer_time_normal() {
        let transfer = FlashTransfer::new();
        let time = transfer.estimate_transfer_time(1024);
        assert!(time > 0.0, "传输时间应大于 0");
    }

    /// 测试 with_config 构造函数
    #[test]
    fn test_with_config() {
        let transfer = FlashTransfer::with_config(2048, 512);
        assert_eq!(transfer.batch_size, 2048);
        assert_eq!(transfer.min_chunk_size, 512);
    }

    /// 测试 Default trait 实现
    #[test]
    fn test_flash_transfer_default() {
        let transfer = FlashTransfer::default();
        assert_eq!(transfer.batch_size, 4096);
        assert_eq!(transfer.min_chunk_size, 256);
    }

    /// 测试 TransferDirection 枚举的 ToGPU 和 ToCPU 变体
    #[test]
    fn test_transfer_direction_enum() {
        let dir_gpu = TransferDirection::ToGPU;
        let dir_cpu = TransferDirection::ToCPU;
        
        assert_eq!(dir_gpu, TransferDirection::ToGPU);
        assert_ne!(dir_gpu, dir_cpu);
        assert!(dir_gpu == TransferDirection::ToGPU || dir_gpu == TransferDirection::ToCPU);
    }

    /// 测试 TransferTask 结构体的字段
    #[test]
    fn test_transfer_task_fields() {
        let task = TransferTask {
            id: 42,
            data: vec![1, 2, 3],
            direction: TransferDirection::ToGPU,
            timestamp: Instant::now(),
        };
        
        assert_eq!(task.id, 42);
        assert_eq!(task.data, vec![1, 2, 3]);
        assert_eq!(task.direction, TransferDirection::ToGPU);
    }

    /// 测试 process_batch 当没有待处理任务时返回空列表
    #[test]
    fn test_process_batch_no_pending() {
        let manager = TransferManager::new(10);
        let completed = manager.process_batch();
        assert!(completed.is_empty());
    }

    /// 测试 get_completed 当没有已完成任务时返回空列表
    #[test]
    fn test_get_completed_no_completed() {
        let manager = TransferManager::new(10);
        let completed = manager.get_completed();
        assert!(completed.is_empty());
    }

    /// 测试 pending_count 初始值为 0
    #[test]
    fn test_pending_count_initial() {
        let manager = TransferManager::new(10);
        assert_eq!(manager.pending_count(), 0);
    }

    /// 测试 Default trait 实现
    #[test]
    fn test_transfer_manager_default() {
        let manager = TransferManager::default();
        assert_eq!(manager.pending_count(), 0);
    }
}
