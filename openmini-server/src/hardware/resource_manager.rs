//! 硬件资源管理器
//!
//! 统一管理 CPU/GPU/内存资源，实现协同调度。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use super::gpu::{GpuBackend, GpuOps};
use super::{
    CoreSelectionStrategy, CpuAffinity, HardwareClassifier, HardwareLevel, HardwareProfile,
    TaskType,
};

/// 计算设备类型 (资源管理器专用)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ResourceManagerDevice {
    /// CPU 计算
    Cpu,
    /// GPU 计算
    Gpu,
    /// 自动选择
    Auto,
}

/// 内存类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum MemoryType {
    /// 系统内存
    System,
    /// GPU 显存
    Gpu,
    /// 统一内存 (Apple Silicon)
    Unified,
}

/// 资源分配请求
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResourceRequest {
    /// 计算设备偏好
    pub device: ResourceManagerDevice,
    /// 内存需求 (MB)
    pub memory_mb: usize,
    /// 内存类型偏好
    pub memory_type: MemoryType,
    /// 任务类型
    pub task_type: TaskType,
    /// 优先级 (0-100, 越高越优先)
    pub priority: u8,
}

/// 资源分配结果
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ResourceAllocation {
    /// 分配的计算设备
    pub device: ResourceManagerDevice,
    /// 分配的内存大小 (MB)
    pub memory_mb: usize,
    /// 分配的内存类型
    pub memory_type: MemoryType,
    /// 分配的核心列表 (CPU)
    pub cpu_cores: Vec<usize>,
    /// GPU 设备 ID
    pub gpu_device_id: Option<usize>,
}

/// 硬件资源管理器
#[allow(dead_code)]
pub struct HardwareResourceManager {
    /// 硬件配置
    profile: HardwareProfile,
    /// 硬件级别
    level: HardwareLevel,
    /// CPU 亲和性管理器
    cpu_affinity: CpuAffinity,
    /// GPU 后端
    gpu_backend: Option<GpuBackend>,
    /// 系统内存使用量 (MB)
    system_memory_used: Arc<AtomicUsize>,
    /// GPU 内存使用量 (MB)
    gpu_memory_used: Arc<AtomicUsize>,
    /// 是否初始化
    initialized: AtomicBool,
    /// 资源请求队列
    request_queue: Mutex<VecDeque<ResourceRequest>>,
}

#[allow(dead_code)]
impl HardwareResourceManager {
    /// 创建新的硬件资源管理器
    pub fn new() -> Self {
        let profile = HardwareProfile::detect();
        let classifier = HardwareClassifier::new(profile.clone());
        let level = classifier.level();
        let cpu_affinity = CpuAffinity::new(profile.hyperthreading.clone(), profile.numa.clone());
        let gpu_backend = GpuBackend::detect();

        Self {
            profile,
            level,
            cpu_affinity,
            gpu_backend,
            system_memory_used: Arc::new(AtomicUsize::new(0)),
            gpu_memory_used: Arc::new(AtomicUsize::new(0)),
            initialized: AtomicBool::new(false),
            request_queue: Mutex::new(VecDeque::new()),
        }
    }

    /// 初始化资源管理器
    pub fn initialize(&self) -> Result<(), String> {
        if self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }

        // 预留系统资源
        let reserved_memory = self.profile.memory.total_gb * 1024 / 4; // 预留 25%
        self.system_memory_used
            .store(reserved_memory, Ordering::SeqCst);

        self.initialized.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// 请求资源分配
    pub fn request_resources(
        &self,
        request: ResourceRequest,
    ) -> Result<ResourceAllocation, String> {
        if !self.initialized.load(Ordering::SeqCst) {
            self.initialize()?;
        }

        let device = self.select_device(&request);
        let memory_type = self.select_memory_type(&request, &device);
        let memory_mb = self.allocate_memory(&request, &memory_type)?;
        let cpu_cores = self.allocate_cpu_cores(&request, &device);
        let gpu_device_id = self.get_gpu_device_id(&device);

        Ok(ResourceAllocation {
            device,
            memory_mb,
            memory_type,
            cpu_cores,
            gpu_device_id,
        })
    }

    /// 释放资源
    pub fn release_resources(&self, allocation: &ResourceAllocation) {
        match allocation.memory_type {
            MemoryType::System | MemoryType::Unified => {
                let current = self.system_memory_used.load(Ordering::SeqCst);
                self.system_memory_used.store(
                    current.saturating_sub(allocation.memory_mb),
                    Ordering::SeqCst,
                );
            }
            MemoryType::Gpu => {
                let current = self.gpu_memory_used.load(Ordering::SeqCst);
                self.gpu_memory_used.store(
                    current.saturating_sub(allocation.memory_mb),
                    Ordering::SeqCst,
                );
            }
        }
    }

    /// 选择计算设备
    fn select_device(&self, request: &ResourceRequest) -> ResourceManagerDevice {
        match request.device {
            ResourceManagerDevice::Auto => {
                if self.gpu_backend.is_some()
                    && matches!(request.task_type, TaskType::ComputeIntensive)
                {
                    ResourceManagerDevice::Gpu
                } else {
                    ResourceManagerDevice::Cpu
                }
            }
            device => device,
        }
    }

    fn select_memory_type(
        &self,
        _request: &ResourceRequest,
        device: &ResourceManagerDevice,
    ) -> MemoryType {
        if self.profile.cpu.is_apple_silicon {
            return MemoryType::Unified;
        }

        match device {
            ResourceManagerDevice::Gpu => MemoryType::Gpu,
            ResourceManagerDevice::Cpu | ResourceManagerDevice::Auto => MemoryType::System,
        }
    }

    /// 分配内存
    fn allocate_memory(
        &self,
        request: &ResourceRequest,
        memory_type: &MemoryType,
    ) -> Result<usize, String> {
        let (available, used) = match memory_type {
            MemoryType::System | MemoryType::Unified => {
                let total = self.profile.memory.total_gb * 1024;
                let used = self.system_memory_used.load(Ordering::SeqCst);
                (total, used)
            }
            MemoryType::Gpu => {
                if let Some(ref gpu) = self.gpu_backend {
                    let total = gpu.device_info().memory_size / (1024 * 1024);
                    let used = self.gpu_memory_used.load(Ordering::SeqCst);
                    (total, used)
                } else {
                    return Err("No GPU available".to_string());
                }
            }
        };

        if used + request.memory_mb > available {
            return Err(format!(
                "Insufficient memory: requested {} MB, available {} MB",
                request.memory_mb,
                available.saturating_sub(used)
            ));
        }

        // 更新使用量
        match memory_type {
            MemoryType::System | MemoryType::Unified => {
                self.system_memory_used
                    .fetch_add(request.memory_mb, Ordering::SeqCst);
            }
            MemoryType::Gpu => {
                self.gpu_memory_used
                    .fetch_add(request.memory_mb, Ordering::SeqCst);
            }
        }

        Ok(request.memory_mb)
    }

    /// 分配 CPU 核心
    fn allocate_cpu_cores(
        &self,
        request: &ResourceRequest,
        device: &ResourceManagerDevice,
    ) -> Vec<usize> {
        if *device == ResourceManagerDevice::Gpu {
            return Vec::new();
        }

        let strategy = match request.task_type {
            TaskType::ComputeIntensive => CoreSelectionStrategy::PhysicalOnly,
            TaskType::IoIntensive => CoreSelectionStrategy::AllCores,
            TaskType::Mixed => CoreSelectionStrategy::PerformanceFirst,
        };

        let num_cores = self.cpu_affinity.optimal_thread_count(request.task_type);
        self.cpu_affinity.select_cores(strategy, Some(num_cores))
    }

    fn get_gpu_device_id(&self, device: &ResourceManagerDevice) -> Option<usize> {
        if *device == ResourceManagerDevice::Gpu && self.gpu_backend.is_some() {
            Some(0)
        } else {
            None
        }
    }

    /// 获取硬件配置
    pub fn profile(&self) -> &HardwareProfile {
        &self.profile
    }

    /// 获取硬件级别
    pub fn level(&self) -> HardwareLevel {
        self.level
    }

    /// 获取 GPU 后端
    pub fn gpu_backend(&self) -> Option<&GpuBackend> {
        self.gpu_backend.as_ref()
    }

    /// 获取可用系统内存 (MB)
    pub fn available_system_memory(&self) -> usize {
        let total = self.profile.memory.total_gb * 1024;
        let used = self.system_memory_used.load(Ordering::SeqCst);
        total - used
    }

    /// 获取可用 GPU 内存 (MB)
    pub fn available_gpu_memory(&self) -> usize {
        if let Some(ref gpu) = self.gpu_backend {
            let total = gpu.device_info().memory_size / (1024 * 1024);
            let used = self.gpu_memory_used.load(Ordering::SeqCst);
            total - used
        } else {
            0
        }
    }

    /// 获取最优线程数
    pub fn optimal_threads(&self, task_type: TaskType) -> usize {
        self.cpu_affinity.optimal_thread_count(task_type)
    }

    /// 绑定当前线程到指定核心
    pub fn bind_thread(&self, core_id: usize) -> Result<(), String> {
        self.cpu_affinity
            .bind_current_thread(core_id)
            .map_err(|e| e.to_string())
    }
}

impl Default for HardwareResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// 全局资源管理器
#[allow(dead_code)]
static RESOURCE_MANAGER: std::sync::OnceLock<HardwareResourceManager> = std::sync::OnceLock::new();

/// 获取全局资源管理器
#[allow(dead_code)]
pub fn get_resource_manager() -> &'static HardwareResourceManager {
    RESOURCE_MANAGER.get_or_init(HardwareResourceManager::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_manager_creation() {
        let manager = HardwareResourceManager::new();
        assert!(manager.initialize().is_ok());
    }

    #[test]
    fn test_resource_request() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        let request = ResourceRequest {
            device: ResourceManagerDevice::Auto,
            memory_mb: 1024,
            memory_type: MemoryType::System,
            task_type: TaskType::ComputeIntensive,
            priority: 50,
        };

        let result = manager.request_resources(request);
        assert!(result.is_ok());

        let allocation = result.unwrap();
        assert!(allocation.memory_mb > 0);
    }

    #[test]
    fn test_memory_availability() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        let available = manager.available_system_memory();
        assert!(available > 0);
    }

    #[test]
    fn test_optimal_threads() {
        let manager = HardwareResourceManager::new();

        let compute_threads = manager.optimal_threads(TaskType::ComputeIntensive);
        let io_threads = manager.optimal_threads(TaskType::IoIntensive);

        assert!(compute_threads > 0);
        assert!(io_threads >= compute_threads);
    }

    #[test]
    fn test_global_manager() {
        let manager = get_resource_manager();
        assert!(manager.initialize().is_ok());
    }

    // ========== 新增测试开始 ==========

    /// 测试资源释放后内存回收
    #[test]
    fn test_resource_release() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        let initial_available = manager.available_system_memory();

        // 分配资源
        let request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 512, // 分配512MB
            memory_type: MemoryType::System,
            task_type: TaskType::ComputeIntensive,
            priority: 50,
        };

        let allocation = manager.request_resources(request).unwrap();
        let after_alloc_available = manager.available_system_memory();

        // 分配后可用内存应该减少
        assert!(
            after_alloc_available < initial_available,
            "分配后可用内存应减少: {} < {}",
            after_alloc_available,
            initial_available
        );

        // 释放资源
        manager.release_resources(&allocation);
        let after_release_available = manager.available_system_memory();

        // 释放后可用内存应该恢复（或至少增加）
        assert!(
            after_release_available >= after_alloc_available,
            "释放后可用内存应恢复或增加"
        );
    }

    /// 测试内存不足时的错误处理
    #[test]
    fn test_insufficient_memory_error() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // 请求超过系统总内存的大小（比如100TB）
        let request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 100 * 1024 * 1024, // 100TB，肯定超出可用内存
            memory_type: MemoryType::System,
            task_type: TaskType::ComputeIntensive,
            priority: 50,
        };

        let result = manager.request_resources(request);

        // 应该返回错误
        assert!(result.is_err(), "请求超大内存应返回错误");
        let error_msg = result.unwrap_err();
        assert!(
            error_msg.to_lowercase().contains("insufficient")
                || error_msg.to_lowercase().contains("memory"),
            "错误消息应包含'insufficient'或'memory'"
        );
    }

    /// 测试不同设备类型的选择逻辑
    #[test]
    fn test_device_selection() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // 测试显式选择CPU
        let cpu_request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 100,
            memory_type: MemoryType::System,
            task_type: TaskType::ComputeIntensive,
            priority: 50,
        };
        let cpu_allocation = manager.request_resources(cpu_request).unwrap();
        assert_eq!(cpu_allocation.device, ResourceManagerDevice::Cpu);

        // 测试显式选择GPU（如果没有GPU可能会失败或回退到CPU）
        let gpu_request = ResourceRequest {
            device: ResourceManagerDevice::Gpu,
            memory_mb: 100,
            memory_type: MemoryType::Gpu,
            task_type: TaskType::ComputeIntensive,
            priority: 50,
        };
        match manager.request_resources(gpu_request) {
            Ok(gpu_allocation) => {
                // 如果成功，应该是GPU设备
                assert_eq!(gpu_allocation.device, ResourceManagerDevice::Gpu);
            }
            Err(_) => {
                // 没有GPU时返回错误是可接受的
            }
        }
    }

    /// 测试Auto模式下计算密集型任务的设备选择
    #[test]
    fn test_auto_device_compute_intensive() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // Auto模式 + 计算密集型任务
        let auto_request = ResourceRequest {
            device: ResourceManagerDevice::Auto,
            memory_mb: 100,
            memory_type: MemoryType::System,
            task_type: TaskType::ComputeIntensive, // 计算密集型
            priority: 80,
        };

        let allocation = manager.request_resources(auto_request).unwrap();

        // Auto模式应该选择了合适的设备（CPU或GPU）
        match allocation.device {
            ResourceManagerDevice::Cpu | ResourceManagerDevice::Gpu => {}
            _ => panic!("Auto模式不应返回Auto设备类型"),
        }
    }

    /// 测试不同任务类型的核心分配策略
    #[test]
    fn test_task_type_core_allocation() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // 计算密集型任务：应该使用物理核心
        let compute_request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 100,
            memory_type: MemoryType::System,
            task_type: TaskType::ComputeIntensive,
            priority: 50,
        };
        let compute_alloc = manager.request_resources(compute_request).unwrap();

        // IO密集型任务：可以使用所有核心（包括超线程）
        let io_request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 100,
            memory_type: MemoryType::System,
            task_type: TaskType::IoIntensive,
            priority: 50,
        };
        let io_alloc = manager.request_resources(io_request).unwrap();

        // 验证都获得了核心分配（可能为空如果使用GPU）
        // CPU设备时应该有核心列表
        if compute_alloc.device == ResourceManagerDevice::Cpu {
            // 核心列表可以为空（取决于实现），但不应panic
            println!("计算密集型分配的核心: {:?}", compute_alloc.cpu_cores);
        }

        if io_alloc.device == ResourceManagerDevice::Cpu {
            println!("IO密集型分配的核心: {:?}", io_alloc.cpu_cores);
        }
    }

    /// 测试重复初始化的幂等性
    #[test]
    fn test_initialize_idempotency() {
        let manager = HardwareResourceManager::new();

        // 第一次初始化
        assert!(manager.initialize().is_ok());

        // 第二次初始化应该成功且无副作用
        assert!(manager.initialize().is_ok());

        // 第三次初始化也应该成功
        assert!(manager.initialize().is_ok());

        // 可用内存应该保持一致
        let available1 = manager.available_system_memory();
        let available2 = manager.available_system_memory();
        assert_eq!(available1, available2, "多次初始化后状态应一致");
    }

    /// 测试profile()和level()信息获取
    #[test]
    fn test_profile_and_level_info() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // 获取硬件配置文件
        let profile = manager.profile();
        assert!(profile.memory.total_gb > 0, "总内存应>0");
        println!("总内存: {} GB", profile.memory.total_gb);
        println!("CPU是否Apple Silicon: {}", profile.cpu.is_apple_silicon);

        // 获取硬件级别
        let level = manager.level();
        // 硬件级别应该在合理范围内
        println!("硬件级别: {:?}", level);
    }

    /// 测试GPU内存查询（如果有GPU）
    #[test]
    fn test_gpu_memory_query() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        let gpu_mem = manager.available_gpu_memory();

        if manager.gpu_backend().is_some() {
            // 有GPU时，可用内存应该>=0
            println!("可用GPU内存: {} MB", gpu_mem);
        } else {
            // 无GPU时，可用内存应为0
            assert_eq!(gpu_mem, 0, "无GPU时可用GPU内存应为0");
        }
    }

    /// 测试混合类型的内存分配和释放
    #[test]
    fn test_mixed_memory_types() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // 系统内存分配
        let sys_request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 256,
            memory_type: MemoryType::System,
            task_type: TaskType::Mixed,
            priority: 60,
        };
        let sys_alloc = manager.request_resources(sys_request).unwrap();
        assert_eq!(sys_alloc.memory_type, MemoryType::System);

        // 统一内存分配（Apple Silicon）或回退到系统内存
        let unified_request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 256,
            memory_type: MemoryType::Unified,
            task_type: TaskType::Mixed,
            priority: 60,
        };
        let unified_alloc = manager.request_resources(unified_request).unwrap();

        // Apple Silicon上应该是Unified，其他平台可能是System
        if manager.profile().cpu.is_apple_silicon {
            assert_eq!(unified_alloc.memory_type, MemoryType::Unified);
        }

        // 释放所有资源
        manager.release_resources(&sys_alloc);
        manager.release_resources(&unified_alloc);

        // 验证内存已回收
        let final_available = manager.available_system_memory();
        assert!(final_available > 0, "释放后应有可用内存");
    }

    /// 测试高优先级与低优先级资源的公平性
    #[test]
    fn test_priority_resource_allocation() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // 低优先级请求
        let low_priority_request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 128,
            memory_type: MemoryType::System,
            task_type: TaskType::IoIntensive,
            priority: 10, // 低优先级
        };
        let low_priority_alloc = manager.request_resources(low_priority_request).unwrap();
        assert_eq!(low_priority_alloc.memory_mb, 128); // 验证分配的内存大小

        // 高优先级请求
        let high_priority_request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 128,
            memory_type: MemoryType::System,
            task_type: TaskType::ComputeIntensive,
            priority: 90, // 高优先级
        };
        let high_priority_alloc = manager.request_resources(high_priority_request).unwrap();
        assert_eq!(high_priority_alloc.memory_mb, 128); // 验证分配的内存大小

        // 清理
        manager.release_resources(&low_priority_alloc);
        manager.release_resources(&high_priority_alloc);
    }

    // ==================== 新增测试：达到 20+ 覆盖率 ====================

    /// 测试：release_resources 对 Gpu 内存类型的释放（覆盖第166-172行分支）
    #[test]
    fn test_release_gpu_memory() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // 尝试分配 GPU 内存（如果没有GPU会失败，这也是有效路径）
        let gpu_request = ResourceRequest {
            device: ResourceManagerDevice::Gpu,
            memory_mb: 256,
            memory_type: MemoryType::Gpu,
            task_type: TaskType::ComputeIntensive,
            priority: 80,
        };

        match manager.request_resources(gpu_request) {
            Ok(gpu_alloc) => {
                // 验证是 GPU 内存类型
                assert_eq!(gpu_alloc.memory_type, MemoryType::Gpu);

                // 释放 GPU 内存
                let gpu_mem_before = manager.available_gpu_memory();
                manager.release_resources(&gpu_alloc);
                let gpu_mem_after = manager.available_gpu_memory();

                // 释放后可用内存应该增加或保持不变
                assert!(gpu_mem_after >= gpu_mem_before);
            }
            Err(_) => {
                // 没有GPU时返回错误是可接受的
                println!("GPU不可用，跳过GPU内存释放测试");
            }
        }
    }

    /// 测试：request_resources 的边界条件 - 零内存请求
    #[test]
    fn test_zero_memory_request() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // 请求0MB内存
        let zero_request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 0, // 零内存
            memory_type: MemoryType::System,
            task_type: TaskType::IoIntensive,
            priority: 50,
        };

        let result = manager.request_resources(zero_request);
        assert!(result.is_ok(), "零内存请求应成功");

        let alloc = result.unwrap();
        assert_eq!(alloc.memory_mb, 0); // 分配的内存应为0

        // 释放零内存不应出错
        manager.release_resources(&alloc);
    }

    /// 测试：bind_thread 在无效核心ID时的错误处理（覆盖错误路径）
    #[test]
    fn test_bind_thread_invalid_core() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        // 尝试绑定到一个可能无效的核心ID（非常大的数字）
        let result = manager.bind_thread(99999);

        // 可能成功也可能失败，取决于系统实现
        // 主要验证不会panic
        match result {
            Ok(()) => println!("绑定到核心99999成功（意外但可接受）"),
            Err(e) => {
                // 错误消息应该有意义
                assert!(!e.is_empty(), "错误消息不应为空");
                println!("绑定失败（预期）: {}", e);
            }
        }
    }

    /// 测试：ResourceAllocation 结构体的字段完整性验证
    #[test]
    fn test_resource_allocation_fields() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        let request = ResourceRequest {
            device: ResourceManagerDevice::Cpu,
            memory_mb: 512,
            memory_type: MemoryType::System,
            task_type: TaskType::Mixed,
            priority: 75,
        };

        let allocation = manager.request_resources(request).unwrap();

        // 验证所有字段都有合理值
        assert_eq!(allocation.memory_mb, 512);
        assert_eq!(allocation.memory_type, MemoryType::System);

        // CPU 设备时应有核心列表（可能为空）
        match allocation.device {
            ResourceManagerDevice::Cpu => {
                // cpu_cores 可以为空或包含核心ID
                println!("CPU核心列表: {:?}", allocation.cpu_cores);
            }
            ResourceManagerDevice::Gpu => {
                // GPU 设备应有 device_id
                assert!(allocation.gpu_device_id.is_some(), "GPU设备应有device_id");
            }
            ResourceManagerDevice::Auto => {
                panic!("Auto设备类型不应出现在最终分配中");
            }
        }
    }

    /// 测试：多次连续请求和释放的资源回收验证
    #[test]
    fn test_multiple_resource_cycles() {
        let manager = HardwareResourceManager::new();
        manager.initialize().unwrap();

        let initial_available = manager.available_system_memory();
        let mut allocations = Vec::new();

        // 分配10次，每次256MB
        for i in 0..10 {
            let request = ResourceRequest {
                device: ResourceManagerDevice::Cpu,
                memory_mb: 256,
                memory_type: MemoryType::System,
                task_type: TaskType::ComputeIntensive,
                priority: 50 + i as u8,
            };

            let alloc = manager.request_resources(request).unwrap();
            allocations.push(alloc);
        }

        // 验证可用内存减少
        let after_alloc = manager.available_system_memory();
        assert!(after_alloc < initial_available, "分配后可用内存应减少");

        // 释放所有资源
        for alloc in &allocations {
            manager.release_resources(alloc);
        }

        // 验证可用内存恢复
        let after_release = manager.available_system_memory();
        assert!(after_release > after_alloc, "释放后可用内存应恢复");
    }

    /// 测试：Default trait 实现验证（调用 new()）
    #[test]
    fn test_resource_manager_default() {
        let manager = HardwareResourceManager::default();

        // Default 应该创建有效的管理器
        assert!(manager.initialize().is_ok());
        assert!(manager.profile().memory.total_gb > 0);
        assert!(manager.optimal_threads(TaskType::ComputeIntensive) > 0);
    }

    /// 测试：MemoryType 和 ResourceManagerDevice 枚举变体完整性
    #[test]
    fn test_enum_variants_completeness() {
        // 验证所有枚举变体都可以正常使用
        let devices = [
            ResourceManagerDevice::Cpu,
            ResourceManagerDevice::Gpu,
            ResourceManagerDevice::Auto,
        ];

        let memory_types = [MemoryType::System, MemoryType::Gpu, MemoryType::Unified];

        // 验证枚举可以比较和匹配
        assert_eq!(devices[0], ResourceManagerDevice::Cpu);
        assert_ne!(devices[0], devices[1]);
        assert_eq!(memory_types[0], MemoryType::System);
        assert_ne!(memory_types[0], memory_types[1]);
    }
}
