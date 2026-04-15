//! Vulkan GPU 后端实现
//!
//! 为跨平台提供 Vulkan GPU 加速支持。
//! 在 macOS 上通过 MoltenVK 转换为 Metal。
//!
//! ## 功能
//! - 矩阵乘法 (优化分块实现)
//! - Softmax (数值稳定)
//! - Layer Normalization / RMS Normalization
//! - 注意力计算 (支持 KV Cache)
//!
//! ## 使用
//!
//! ```ignore
//! use openmini_server::hardware::gpu::vulkan::VulkanBackend;
//!
//! let vulkan = VulkanBackend::new()?;
//! let result = vulkan.matmul(&a, &b)?;
//! ```
//!
//! ## 编译要求
//!
//! 需要启用 `vulkan` feature。

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::RwLock;

use anyhow::{bail, Result};
use ash::vk;
use ndarray::Array2;

use super::{GpuDeviceInfo, GpuOps};

// ============================================================================
// 常量定义
// ============================================================================

/// 矩阵乘法分块大小
const MATMUL_BLOCK_SIZE: u32 = 16;

/// Softmax 分块大小
const SOFTMAX_BLOCK_SIZE: u32 = 256;

/// LayerNorm 分块大小
const LAYERNORM_BLOCK_SIZE: u32 = 256;

/// Attention 分块大小
const ATTENTION_BLOCK_SIZE: u32 = 16;

// ============================================================================
// Vulkan Compute Shader 源码
// ============================================================================

/// 矩阵乘法 Vulkan Compute Shader
const MATMUL_SHADER: &str = r#"
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) buffer A { float a[]; };
layout(binding = 1) buffer B { float b[]; };
layout(binding = 2) buffer C { float c[]; };
layout(binding = 3) buffer Dims { uvec3 dims; };

void main() {
    uint m = gl_GlobalInvocationID.x;
    uint n = gl_GlobalInvocationID.y;
    
    if (m >= dims.x || n >= dims.y) return;
    
    uint K = dims.z;
    float sum = 0.0;
    
    for (uint k = 0; k < K; k++) {
        sum += a[m * K + k] * b[k * dims.y + n];
    }
    
    c[m * dims.y + n] = sum;
}
"#;

/// 分块矩阵乘法 Vulkan Compute Shader (优化版本)
const MATMUL_BLOCKED_SHADER: &str = r#"
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) buffer A { float a[]; };
layout(binding = 1) buffer B { float b[]; };
layout(binding = 2) buffer C { float c[]; };
layout(binding = 3) buffer Dims { uvec3 dims; };

shared float A_tile[16][16];
shared float B_tile[16][16];

void main() {
    uint m = gl_WorkGroupID.x * 16 + gl_LocalInvocationID.x;
    uint n = gl_WorkGroupID.y * 16 + gl_LocalInvocationID.y;
    
    uint K = dims.z;
    float sum = 0.0;
    
    for (uint k_block = 0; k_block < (K + 15) / 16; k_block++) {
        uint k = k_block * 16 + gl_LocalInvocationID.y;
        if (m < dims.x && k < K) {
            A_tile[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = a[m * K + k];
        } else {
            A_tile[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = 0.0;
        }
        
        k = k_block * 16 + gl_LocalInvocationID.x;
        uint b_row = k;
        if (b_row < K && n < dims.y) {
            B_tile[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = b[b_row * dims.y + n];
        } else {
            B_tile[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = 0.0;
        }
        
        memoryBarrierShared();
        barrier();
        
        for (uint k_local = 0; k_local < 16; k_local++) {
            sum += A_tile[gl_LocalInvocationID.x][k_local] * B_tile[k_local][gl_LocalInvocationID.y];
        }
        
        memoryBarrierShared();
        barrier();
    }
    
    if (m < dims.x && n < dims.y) {
        c[m * dims.y + n] = sum;
    }
}
"#;

/// Softmax Vulkan Compute Shader
const SOFTMAX_SHADER: &str = r#"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Input { float input[]; };
layout(binding = 1) buffer Output { float output[]; };
layout(binding = 2) buffer Dims { uvec2 dims; };

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= dims.x) return;
    
    uint cols = dims.y;
    uint offset = row * cols;
    
    float max_val = input[offset];
    for (uint j = 1; j < cols; j++) {
        max_val = max(max_val, input[offset + j]);
    }
    
    float sum = 0.0;
    for (uint j = 0; j < cols; j++) {
        sum += exp(input[offset + j] - max_val);
    }
    
    float inv_sum = 1.0 / sum;
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = exp(input[offset + j] - max_val) * inv_sum;
    }
}
"#;

/// LayerNorm Vulkan Compute Shader
const LAYERNORM_SHADER: &str = r#"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Input { float input[]; };
layout(binding = 1) buffer Output { float output[]; };
layout(binding = 2) buffer Gamma { float gamma[]; };
layout(binding = 3) buffer Beta { float beta[]; };
layout(binding = 4) buffer Dims { uvec2 dims; };
layout(binding = 5) buffer Eps { float eps; };

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= dims.x) return;
    
    uint cols = dims.y;
    uint offset = row * cols;
    
    float mean = 0.0;
    for (uint j = 0; j < cols; j++) {
        mean += input[offset + j];
    }
    mean /= float(cols);
    
    float var = 0.0;
    for (uint j = 0; j < cols; j++) {
        float diff = input[offset + j] - mean;
        var += diff * diff;
    }
    var /= float(cols);
    
    float inv_std = 1.0 / sqrt(var + eps);
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = gamma[j] * (input[offset + j] - mean) * inv_std + beta[j];
    }
}
"#;

/// RMS Normalization Vulkan Compute Shader
const RMS_NORM_SHADER: &str = r#"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Input { float input[]; };
layout(binding = 1) buffer Output { float output[]; };
layout(binding = 2) buffer Gamma { float gamma[]; };
layout(binding = 3) buffer Dims { uvec2 dims; };
layout(binding = 4) buffer Eps { float eps; };

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= dims.x) return;
    
    uint cols = dims.y;
    uint offset = row * cols;
    
    float sum_sq = 0.0;
    for (uint j = 0; j < cols; j++) {
        sum_sq += input[offset + j] * input[offset + j];
    }
    
    float rms = sqrt(sum_sq / float(cols) + eps);
    float inv_rms = 1.0 / rms;
    
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = gamma[j] * input[offset + j] * inv_rms;
    }
}
"#;

/// Attention Vulkan Compute Shader (在线 Softmax，无共享内存竞争)
const ATTENTION_SHADER: &str = r#"
#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(binding = 0) buffer Query { float query[]; };
layout(binding = 1) buffer Key { float key[]; };
layout(binding = 2) buffer Value { float value[]; };
layout(binding = 3) buffer Output { float output[]; };
layout(binding = 4) buffer Mask { float mask[]; };
layout(binding = 5) buffer Dims { uvec3 dims; };
layout(binding = 6) buffer Scale { float scale; };
layout(binding = 7) buffer HasMask { uint has_mask; };

void main() {
    uint seq_idx = gl_GlobalInvocationID.x;
    uint head_idx = gl_GlobalInvocationID.y;
    
    if (seq_idx >= dims.x || head_idx >= dims.z) return;
    
    uint seq_len = dims.x;
    uint kv_len = dims.y;
    uint head_dim = dims.z;
    
    float output_val = 0.0;
    float max_score = -1e30;
    float sum = 0.0;
    
    for (uint kv_idx = 0; kv_idx < kv_len; kv_idx++) {
        if (has_mask == 1) {
            float mask_val = mask[seq_idx * kv_len + kv_idx];
            if (mask_val < -1e30) continue;
        }
        
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            dot += query[seq_idx * head_dim + d] * key[kv_idx * head_dim + d];
        }
        float score = dot * scale;
        
        if (has_mask == 1) {
            score += mask[seq_idx * kv_len + kv_idx];
        }
        
        float new_max = max(max_score, score);
        float scale_factor = exp(max_score - new_max);
        float score_exp = exp(score - new_max);
        
        output_val = output_val * scale_factor + score_exp * value[kv_idx * head_dim + head_idx];
        sum = sum * scale_factor + score_exp;
        max_score = new_max;
    }
    
    output[seq_idx * head_dim + head_idx] = (sum > 0.0) ? output_val / sum : 0.0;
}
"#;

/// 带 KV Cache 的 Attention Vulkan Compute Shader (在线 Softmax，无共享内存竞争)
const ATTENTION_KV_CACHE_SHADER: &str = r#"
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Query { float query[]; };
layout(binding = 1) buffer KeyCache { float key_cache[]; };
layout(binding = 2) buffer ValueCache { float value_cache[]; };
layout(binding = 3) buffer Output { float output[]; };
layout(binding = 4) buffer CacheInfo { uvec2 cache_info; };
layout(binding = 5) buffer Scale { float scale; };

void main() {
    uint head_idx = gl_GlobalInvocationID.x;
    uint kv_len = cache_info.x;
    uint head_dim = cache_info.y;
    
    if (head_idx >= head_dim) return;
    
    float output_val = 0.0;
    float max_score = -1e30;
    float sum = 0.0;
    
    for (uint kv_idx = 0; kv_idx < kv_len; kv_idx++) {
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) {
            dot += query[d] * key_cache[kv_idx * head_dim + d];
        }
        float score = dot * scale;
        
        float new_max = max(max_score, score);
        float scale_factor = exp(max_score - new_max);
        float score_exp = exp(score - new_max);
        
        output_val = output_val * scale_factor + score_exp * value_cache[kv_idx * head_dim + head_idx];
        sum = sum * scale_factor + score_exp;
        max_score = new_max;
    }
    
    output[head_idx] = (sum > 0.0) ? output_val / sum : 0.0;
}
"#;

// ============================================================================
// Vulkan 实例封装
// ============================================================================

/// Vulkan 实例封装
pub struct VulkanInstance {
    /// Vulkan 实例
    instance: ash::Instance,
    /// 入口点
    entry: ash::Entry,
}

impl VulkanInstance {
    /// 创建 Vulkan 实例
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };

        let app_info = vk::ApplicationInfo::default()
            .application_name(unsafe {
                std::ffi::CStr::from_bytes_with_nul_unchecked(b"OpenMini\0")
            })
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"OpenMini\0") })
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        Ok(Self { instance, entry })
    }

    /// 获取实例引用
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// 获取入口点引用
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

// ============================================================================
// Vulkan 设备封装
// ============================================================================

/// Vulkan 设备封装
pub struct VulkanDevice {
    /// Vulkan 设备
    device: ash::Device,
    /// 物理设备
    physical_device: vk::PhysicalDevice,
    /// 设备信息
    device_info: GpuDeviceInfo,
    /// 计算队列族索引
    compute_queue_family_index: u32,
    /// 实例引用
    instance: std::sync::Arc<VulkanInstance>,
}

impl VulkanDevice {
    /// 创建 Vulkan 设备
    pub fn new(instance: std::sync::Arc<VulkanInstance>) -> Result<Self> {
        let physical_device = Self::select_physical_device(instance.instance())?;

        let (device, compute_queue_family_index) =
            Self::create_logical_device(instance.instance(), physical_device)?;

        let device_info = Self::get_device_info(instance.instance(), physical_device)?;

        Ok(Self {
            device,
            physical_device,
            device_info,
            compute_queue_family_index,
            instance,
        })
    }

    /// 选择物理设备
    fn select_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        if physical_devices.is_empty() {
            bail!("未找到 Vulkan 物理设备");
        }

        for device in &physical_devices {
            let properties = unsafe { instance.get_physical_device_properties(*device) };

            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                || properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
            {
                return Ok(*device);
            }
        }

        Ok(physical_devices[0])
    }

    /// 创建逻辑设备
    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<(ash::Device, u32)> {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let compute_queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| anyhow::anyhow!("未找到计算队列族"))?;

        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_family_index)
            .queue_priorities(&queue_priorities);

        let device_features = vk::PhysicalDeviceFeatures::default();

        let queue_create_infos = [queue_create_info];
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features);

        let device = unsafe { instance.create_device(physical_device, &create_info, None)? };

        Ok((device, compute_queue_family_index))
    }

    /// 获取设备信息
    fn get_device_info(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<GpuDeviceInfo> {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let name = unsafe {
            std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };

        let mut memory_size = 0usize;
        for i in 0..memory_properties.memory_heap_count as usize {
            if memory_properties.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            {
                memory_size = memory_properties.memory_heaps[i].size as usize;
                break;
            }
        }

        let mut features = Vec::new();
        features.push("compute_shader".to_string());
        features.push("storage_buffer".to_string());

        if properties.api_version >= vk::make_api_version(0, 1, 2, 0) {
            features.push("vulkan_1_2".to_string());
        }

        Ok(GpuDeviceInfo {
            name,
            memory_size,
            compute_capability: Some((
                vk::api_version_major(properties.api_version),
                vk::api_version_minor(properties.api_version),
            )),
            features,
        })
    }

    /// 获取设备引用
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// 获取物理设备
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// 获取设备信息
    pub fn info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    /// 获取计算队列族索引
    pub fn compute_queue_family_index(&self) -> u32 {
        self.compute_queue_family_index
    }

    /// 获取实例引用
    pub fn instance(&self) -> &VulkanInstance {
        &self.instance
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

// ============================================================================
// Vulkan 命令队列封装
// ============================================================================

/// Vulkan 命令队列封装
pub struct VulkanQueue {
    /// Vulkan 队列
    queue: vk::Queue,
    /// 命令池
    command_pool: vk::CommandPool,
    /// 设备引用
    device: std::sync::Arc<VulkanDevice>,
}

impl VulkanQueue {
    /// 创建命令队列
    pub fn new(device: std::sync::Arc<VulkanDevice>) -> Result<Self> {
        let queue = unsafe {
            device
                .device()
                .get_device_queue(device.compute_queue_family_index(), 0)
        };

        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.compute_queue_family_index())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .device()
                .create_command_pool(&command_pool_create_info, None)?
        };

        Ok(Self {
            queue,
            command_pool,
            device,
        })
    }

    /// 分配命令缓冲区
    pub fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe {
            self.device
                .device()
                .allocate_command_buffers(&command_buffer_allocate_info)?
        };

        Ok(command_buffers[0])
    }

    /// 提交命令缓冲区
    pub fn submit(&self, command_buffer: vk::CommandBuffer, fence: vk::Fence) -> Result<()> {
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        unsafe {
            self.device
                .device()
                .queue_submit(self.queue, &[submit_info], fence)?;
        }

        Ok(())
    }

    /// 等待队列空闲
    pub fn wait_idle(&self) -> Result<()> {
        unsafe {
            self.device.device().queue_wait_idle(self.queue)?;
        }
        Ok(())
    }

    /// 获取队列
    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    /// 获取命令池
    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    /// 获取设备引用
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
}

impl Drop for VulkanQueue {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device()
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

// ============================================================================
// Vulkan Buffer 封装
// ============================================================================

/// Vulkan Buffer 封装
pub struct VulkanBuffer {
    /// Vulkan Buffer
    buffer: vk::Buffer,
    /// 设备内存
    memory: vk::DeviceMemory,
    /// 数据大小 (字节)
    size: usize,
    /// 设备引用
    device: std::sync::Arc<VulkanDevice>,
}

impl VulkanBuffer {
    /// 创建新的 Buffer
    pub fn new(device: std::sync::Arc<VulkanDevice>, size: usize) -> Result<Self> {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.device().create_buffer(&buffer_create_info, None)? };

        let memory_requirements = unsafe { device.device().get_buffer_memory_requirements(buffer) };

        let memory_properties = unsafe {
            device
                .instance()
                .instance()
                .get_physical_device_memory_properties(device.physical_device())
        };

        let memory_type_index = Self::find_memory_type(
            memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &memory_properties,
        )?;

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe {
            device
                .device()
                .allocate_memory(&memory_allocate_info, None)?
        };

        unsafe {
            device.device().bind_buffer_memory(buffer, memory, 0)?;
        }

        Ok(Self {
            buffer,
            memory,
            size,
            device,
        })
    }

    /// 从数据创建 Buffer
    pub fn from_data<T>(device: std::sync::Arc<VulkanDevice>, data: &[T]) -> Result<Self> {
        let size = std::mem::size_of_val(data);
        let buffer = Self::new(device, size)?;
        buffer.write(data)?;
        Ok(buffer)
    }

    /// 查找内存类型
    fn find_memory_type(
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Result<u32> {
        for i in 0..memory_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Ok(i);
            }
        }

        bail!("未找到合适的内存类型")
    }

    /// 写入数据
    pub fn write<T>(&self, data: &[T]) -> Result<()> {
        let size = std::mem::size_of_val(data);
        if size > self.size {
            bail!("Buffer 大小不足: {} > {}", size, self.size);
        }

        unsafe {
            let ptr = self.device.device().map_memory(
                self.memory,
                0,
                self.size as u64,
                vk::MemoryMapFlags::empty(),
            )?;

            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, size);

            self.device.device().unmap_memory(self.memory);
        }

        Ok(())
    }

    /// 读取数据
    pub fn read<T>(&self, data: &mut [T]) -> Result<()> {
        let size = std::mem::size_of_val(data);
        if size > self.size {
            bail!("Buffer 大小不足: {} > {}", size, self.size);
        }

        unsafe {
            let ptr = self.device.device().map_memory(
                self.memory,
                0,
                self.size as u64,
                vk::MemoryMapFlags::empty(),
            )?;

            std::ptr::copy_nonoverlapping(ptr as *const u8, data.as_mut_ptr() as *mut u8, size);

            self.device.device().unmap_memory(self.memory);
        }

        Ok(())
    }

    /// 获取 Buffer
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// 获取大小
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device().destroy_buffer(self.buffer, None);
            self.device.device().free_memory(self.memory, None);
        }
    }
}

// ============================================================================
// Shader 编译和 Pipeline 管理
// ============================================================================

/// Shader 类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    /// 矩阵乘法
    Matmul,
    /// 分块矩阵乘法
    MatmulBlocked,
    /// Softmax
    Softmax,
    /// Layer Normalization
    LayerNorm,
    /// RMS Normalization
    RmsNorm,
    /// Attention
    Attention,
    /// KV Cache Attention
    AttentionKvCache,
}

impl ShaderType {
    /// 获取 shader 源码
    fn source(&self) -> &'static str {
        match self {
            ShaderType::Matmul => MATMUL_SHADER,
            ShaderType::MatmulBlocked => MATMUL_BLOCKED_SHADER,
            ShaderType::Softmax => SOFTMAX_SHADER,
            ShaderType::LayerNorm => LAYERNORM_SHADER,
            ShaderType::RmsNorm => RMS_NORM_SHADER,
            ShaderType::Attention => ATTENTION_SHADER,
            ShaderType::AttentionKvCache => ATTENTION_KV_CACHE_SHADER,
        }
    }
}

/// Shader 编译器
pub struct ShaderCompiler;

impl ShaderCompiler {
    /// 创建 Shader 编译器
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// 编译 Shader
    pub fn compile(&mut self, shader_type: ShaderType) -> Result<Vec<u32>> {
        let source = shader_type.source();

        let mut parser = naga::front::glsl::Frontend::default();
        let options = naga::front::glsl::Options {
            stage: naga::ShaderStage::Compute,
            defines: Default::default(),
        };

        let module = parser
            .parse(&options, source)
            .map_err(|e| anyhow::anyhow!("GLSL 解析失败: {:?}", e))?;

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        );

        let info = validator
            .validate(&module)
            .map_err(|e| anyhow::anyhow!("Shader 验证失败: {:?}", e))?;

        let mut spirv = Vec::new();
        let mut writer = naga::back::spv::Writer::new(&naga::back::spv::Options::default())
            .map_err(|e| anyhow::anyhow!("创建 SPIR-V Writer 失败: {:?}", e))?;
        writer
            .write(&module, &info, None, &None, &mut spirv)
            .map_err(|e| anyhow::anyhow!("SPIR-V 生成失败: {:?}", e))?;

        Ok(spirv)
    }
}

/// Descriptor Set Layout 创建信息
struct DescriptorSetLayoutInfo {
    /// 绑定点数量
    binding_count: u32,
}

/// Pipeline 管理
pub struct PipelineManager {
    /// Shader 模块缓存
    shader_modules: HashMap<ShaderType, vk::ShaderModule>,
    /// Pipeline Layout 缓存
    pipeline_layouts: HashMap<ShaderType, vk::PipelineLayout>,
    /// Descriptor Set Layout 缓存
    descriptor_set_layouts: HashMap<ShaderType, vk::DescriptorSetLayout>,
    /// Pipeline 缓存
    pipelines: HashMap<ShaderType, vk::Pipeline>,
    /// 设备引用
    device: std::sync::Arc<VulkanDevice>,
}

impl PipelineManager {
    /// 创建 Pipeline 管理器
    pub fn new(device: std::sync::Arc<VulkanDevice>) -> Result<Self> {
        Ok(Self {
            shader_modules: HashMap::new(),
            pipeline_layouts: HashMap::new(),
            descriptor_set_layouts: HashMap::new(),
            pipelines: HashMap::new(),
            device,
        })
    }

    /// 获取已缓存的 Pipeline (不创建)
    pub fn get_pipeline_cached(
        &self,
        shader_type: ShaderType,
    ) -> Option<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout)> {
        if self.pipelines.contains_key(&shader_type) {
            Some((
                self.pipelines[&shader_type],
                self.pipeline_layouts[&shader_type],
                self.descriptor_set_layouts[&shader_type],
            ))
        } else {
            None
        }
    }

    /// 获取或创建 Pipeline
    pub fn get_pipeline(
        &mut self,
        shader_type: ShaderType,
        binding_count: u32,
    ) -> Result<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout)> {
        if !self.pipelines.contains_key(&shader_type) {
            self.create_pipeline(shader_type, binding_count)?;
        }

        Ok((
            self.pipelines[&shader_type],
            self.pipeline_layouts[&shader_type],
            self.descriptor_set_layouts[&shader_type],
        ))
    }

    /// 创建 Pipeline
    fn create_pipeline(&mut self, shader_type: ShaderType, binding_count: u32) -> Result<()> {
        let mut compiler = ShaderCompiler::new()?;
        let spirv = compiler.compile(shader_type)?;

        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(&spirv);

        let shader_module = unsafe {
            self.device
                .device()
                .create_shader_module(&shader_module_create_info, None)?
        };

        let descriptor_set_layout = self.create_descriptor_set_layout(binding_count)?;

        let pipeline_layout = self.create_pipeline_layout(descriptor_set_layout)?;

        let pipeline = self.create_compute_pipeline(shader_module, pipeline_layout)?;

        self.shader_modules.insert(shader_type, shader_module);
        self.descriptor_set_layouts
            .insert(shader_type, descriptor_set_layout);
        self.pipeline_layouts.insert(shader_type, pipeline_layout);
        self.pipelines.insert(shader_type, pipeline);

        Ok(())
    }

    /// 创建 Descriptor Set Layout
    fn create_descriptor_set_layout(&self, binding_count: u32) -> Result<vk::DescriptorSetLayout> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..binding_count)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe {
            Ok(self
                .device
                .device()
                .create_descriptor_set_layout(&create_info, None)?)
        }
    }

    /// 创建 Pipeline Layout
    fn create_pipeline_layout(
        &self,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout> {
        let set_layouts = [descriptor_set_layout];
        let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);

        unsafe {
            Ok(self
                .device
                .device()
                .create_pipeline_layout(&create_info, None)?)
        }
    }

    /// 创建 Compute Pipeline
    fn create_compute_pipeline(
        &self,
        shader_module: vk::ShaderModule,
        pipeline_layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline> {
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") });

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage)
            .layout(pipeline_layout);

        unsafe {
            let result = self.device.device().create_compute_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            );

            match result {
                Ok(pipelines) => Ok(pipelines[0]),
                Err((_, vk_result)) => bail!("创建 Compute Pipeline 失败: {:?}", vk_result),
            }
        }
    }
}

impl Drop for PipelineManager {
    fn drop(&mut self) {
        unsafe {
            for (_, pipeline) in self.pipelines.drain() {
                self.device.device().destroy_pipeline(pipeline, None);
            }

            for (_, layout) in self.pipeline_layouts.drain() {
                self.device.device().destroy_pipeline_layout(layout, None);
            }

            for (_, layout) in self.descriptor_set_layouts.drain() {
                self.device
                    .device()
                    .destroy_descriptor_set_layout(layout, None);
            }

            for (_, module) in self.shader_modules.drain() {
                self.device.device().destroy_shader_module(module, None);
            }
        }
    }
}

// ============================================================================
// Descriptor Pool 和 Descriptor Set 管理
// ============================================================================

/// Descriptor Pool
pub struct DescriptorPool {
    /// Descriptor Pool
    pool: vk::DescriptorPool,
    /// 设备引用
    device: std::sync::Arc<VulkanDevice>,
}

impl DescriptorPool {
    /// 创建 Descriptor Pool
    pub fn new(device: std::sync::Arc<VulkanDevice>, max_sets: u32) -> Result<Self> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: max_sets * 8,
        }];

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets);

        let pool = unsafe { device.device().create_descriptor_pool(&create_info, None)? };

        Ok(Self { pool, device })
    }

    /// 分配 Descriptor Set
    pub fn allocate_descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let set_layouts = [layout];
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&set_layouts);

        unsafe {
            Ok(self
                .device
                .device()
                .allocate_descriptor_sets(&allocate_info)?[0])
        }
    }

    /// 更新 Descriptor Set
    pub fn update_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        buffers: &[vk::DescriptorBufferInfo],
    ) -> Result<()> {
        let mut writes = Vec::with_capacity(buffers.len());

        for (i, buffer_info) in buffers.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(buffer_info)),
            );
        }

        unsafe {
            self.device.device().update_descriptor_sets(&writes, &[]);
        }

        Ok(())
    }

    /// 释放 Descriptor Set
    pub fn free_descriptor_set(&self, descriptor_set: vk::DescriptorSet) -> Result<()> {
        unsafe {
            self.device
                .device()
                .free_descriptor_sets(self.pool, &[descriptor_set])?;
        }
        Ok(())
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device()
                .destroy_descriptor_pool(self.pool, None);
        }
    }
}

// ============================================================================
// 资源池管理
// ============================================================================

/// 命令缓冲区池
pub struct CommandBufferPool {
    /// 可用的命令缓冲区列表
    available: RwLock<Vec<vk::CommandBuffer>>,
    /// 命令池
    command_pool: vk::CommandPool,
    /// 设备引用
    device: std::sync::Arc<VulkanDevice>,
}

impl CommandBufferPool {
    /// 创建命令缓冲区池
    pub fn new(
        device: std::sync::Arc<VulkanDevice>,
        command_pool: vk::CommandPool,
        initial_count: u32,
    ) -> Result<Self> {
        let mut available = Vec::with_capacity(initial_count as usize);

        if initial_count > 0 {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(initial_count);

            let buffers = unsafe { device.device().allocate_command_buffers(&allocate_info)? };
            available = buffers;
        }

        Ok(Self {
            available: RwLock::new(available),
            command_pool,
            device,
        })
    }

    /// 获取命令缓冲区
    pub fn acquire(&self) -> Result<vk::CommandBuffer> {
        let mut available = self
            .available
            .write()
            .map_err(|_| anyhow::anyhow!("获取命令缓冲区池写锁失败"))?;

        if let Some(buffer) = available.pop() {
            return Ok(buffer);
        }

        drop(available);

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let buffers = unsafe {
            self.device
                .device()
                .allocate_command_buffers(&allocate_info)?
        };

        Ok(buffers[0])
    }

    /// 归还命令缓冲区 (重置状态)
    pub fn release(&self, buffer: vk::CommandBuffer) {
        unsafe {
            let _ = self
                .device
                .device()
                .reset_command_buffer(buffer, vk::CommandBufferResetFlags::empty());
        }

        if let Ok(mut available) = self.available.write() {
            available.push(buffer);
        }
    }
}

/// Fence 池
pub struct FencePool {
    /// 可用的 Fence 列表
    available: RwLock<Vec<vk::Fence>>,
    /// 设备引用
    device: std::sync::Arc<VulkanDevice>,
}

impl FencePool {
    /// 创建 Fence 池
    pub fn new(device: std::sync::Arc<VulkanDevice>, initial_count: u32) -> Result<Self> {
        let mut available = Vec::with_capacity(initial_count as usize);

        for _ in 0..initial_count {
            let fence_create_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let fence = unsafe { device.device().create_fence(&fence_create_info, None)? };
            available.push(fence);
        }

        Ok(Self {
            available: RwLock::new(available),
            device,
        })
    }

    /// 获取 Fence
    pub fn acquire(&self) -> Result<vk::Fence> {
        let mut available = self
            .available
            .write()
            .map_err(|_| anyhow::anyhow!("获取 Fence 池写锁失败"))?;

        if let Some(fence) = available.pop() {
            unsafe {
                self.device.device().reset_fences(&[fence])?;
            }
            return Ok(fence);
        }

        drop(available);

        let fence_create_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            self.device
                .device()
                .create_fence(&fence_create_info, None)?
        };

        Ok(fence)
    }

    /// 归还 Fence
    pub fn release(&self, fence: vk::Fence) {
        if let Ok(mut available) = self.available.write() {
            available.push(fence);
        }
    }
}

impl Drop for FencePool {
    fn drop(&mut self) {
        if let Ok(available) = self.available.read() {
            unsafe {
                for fence in available.iter() {
                    self.device.device().destroy_fence(*fence, None);
                }
            }
        }
    }
}

/// DescriptorSet 池
pub struct DescriptorSetPool {
    /// 可用的 DescriptorSet 列表 (descriptor_set, layout)
    available: RwLock<Vec<(vk::DescriptorSet, vk::DescriptorSetLayout)>>,
    /// 设备引用
    device: std::sync::Arc<VulkanDevice>,
}

impl DescriptorSetPool {
    /// 创建 DescriptorSet 池
    pub fn new(device: std::sync::Arc<VulkanDevice>) -> Result<Self> {
        Ok(Self {
            available: RwLock::new(Vec::new()),
            device,
        })
    }

    /// 获取 DescriptorSet (如果没有匹配的则返回 None)
    pub fn acquire(&self, layout: vk::DescriptorSetLayout) -> Option<vk::DescriptorSet> {
        let mut available = self.available.write().ok()?;

        for i in 0..available.len() {
            if available[i].1 == layout {
                let (descriptor_set, _) = available.remove(i);
                return Some(descriptor_set);
            }
        }

        None
    }

    /// 归还 DescriptorSet
    pub fn release(&self, descriptor_set: vk::DescriptorSet, layout: vk::DescriptorSetLayout) {
        if let Ok(mut available) = self.available.write() {
            if available.len() < 50 {
                available.push((descriptor_set, layout));
            }
        }
    }
}

// ============================================================================
// Vulkan 设备能力查询和管理
// ============================================================================

/// Vulkan 设备能力信息
#[derive(Debug, Clone)]
pub struct VulkanDeviceCapabilities {
    /// 设备名称
    pub device_name: String,
    /// 驱动版本
    pub driver_version: String,
    /// API 版本
    pub api_version: u32,
    /// 最大计算工作组数量 [x, y, z]
    pub max_compute_work_group_count: [u32; 3],
    /// 最大计算工作组成员数
    pub max_compute_work_group_invocations: u32,
    /// 最大计算共享内存大小
    pub max_compute_shared_memory_size: u64,
    /// 子组大小
    pub subgroup_size: u32,
    /// 是否支持 int16
    pub supports_int16: bool,
    /// 是否支持 int8
    pub supports_int8: bool,
    /// 是否支持存储缓冲区
    pub supports_storage_buffer: bool,
}

/// Vulkan 设备信息（用于枚举）
#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    /// 物理设备句柄
    pub physical_device: vk::PhysicalDevice,
    /// 设备名称
    pub device_name: String,
    /// 显存大小（字节）
    pub memory_size: usize,
    /// 设备类型
    pub device_type: vk::PhysicalDeviceType,
    /// 能力信息
    pub capabilities: VulkanDeviceCapabilities,
}

/// Vulkan GPU 封装（提供简化的设备管理接口）
pub struct VulkanGpu {
    /// Vulkan 实例
    instance: std::sync::Arc<VulkanInstance>,
    /// Vulkan 设备
    device: std::sync::Arc<VulkanDevice>,
    /// 设备能力
    capabilities: VulkanDeviceCapabilities,
}

impl VulkanGpu {
    /// 创建 Vulkan GPU 实例（自动选择最佳设备）
    pub fn new(_preferred_device: Option<&str>) -> Result<Self, VulkanError> {
        let instance = std::sync::Arc::new(
            VulkanInstance::new().map_err(|e| VulkanError::InstanceCreation(e.to_string()))?,
        );
        let device = std::sync::Arc::new(
            VulkanDevice::new(instance.clone())
                .map_err(|e| VulkanError::DeviceCreation(e.to_string()))?,
        );
        let capabilities = Self::query_capabilities(instance.instance(), device.physical_device());

        Ok(Self {
            instance,
            device,
            capabilities,
        })
    }

    /// 枚举所有可用的 Vulkan 设备
    pub fn enumerate_devices() -> Result<Vec<VulkanDeviceInfo>, VulkanError> {
        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| VulkanError::EntryLoad(e.to_string()))?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(unsafe {
                std::ffi::CStr::from_bytes_with_nul_unchecked(b"OpenMini Enumerator\0")
            })
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"OpenMini\0") })
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 2, 0));

        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| VulkanError::InstanceCreation(format!("{:?}", e)))?;

        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| VulkanError::Enumeration(format!("{:?}", e)))?;

        let mut devices = Vec::new();

        for pd in physical_devices {
            let properties = unsafe { instance.get_physical_device_properties(pd) };
            let memory_properties = unsafe { instance.get_physical_device_memory_properties(pd) };

            let name = unsafe {
                std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                    .to_string_lossy()
                    .to_string()
            };

            let mut memory_size = 0usize;
            for i in 0..memory_properties.memory_heap_count as usize {
                if memory_properties.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                {
                    memory_size = memory_properties.memory_heaps[i].size as usize;
                    break;
                }
            }

            let capabilities = Self::query_capabilities(&instance, pd);

            devices.push(VulkanDeviceInfo {
                physical_device: pd,
                device_name: name,
                memory_size,
                device_type: properties.device_type,
                capabilities,
            });
        }

        unsafe {
            instance.destroy_instance(None);
        }

        Ok(devices)
    }

    /// 选择最佳计算设备（按 VRAM 大小排序）
    pub fn select_best_device() -> Result<Self, VulkanError> {
        let devices = Self::enumerate_devices()?;

        if devices.is_empty() {
            return Err(VulkanError::NoDeviceFound(
                "未找到可用的 Vulkan 设备".to_string(),
            ));
        }

        // 选择显存最大的设备
        let best_device = devices.into_iter().max_by_key(|d| d.memory_size).unwrap();

        let instance = std::sync::Arc::new(
            VulkanInstance::new().map_err(|e| VulkanError::InstanceCreation(e.to_string()))?,
        );

        // 创建指定物理设备的逻辑设备
        let device = VulkanDevice::new(instance.clone())
            .map_err(|e| VulkanError::DeviceCreation(e.to_string()))?;

        Ok(Self {
            instance,
            device: std::sync::Arc::new(device),
            capabilities: best_device.capabilities,
        })
    }

    /// 查询设备能力
    fn query_capabilities(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> VulkanDeviceCapabilities {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let features = unsafe { instance.get_physical_device_features(physical_device) };

        // 获取子组属性（如果支持）
        let subgroup_props = if properties.api_version >= vk::API_VERSION_1_1 {
            let mut props = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props);
            unsafe {
                instance.get_physical_device_properties2(physical_device, &mut props2);
            }
            Some(props)
        } else {
            None
        };

        VulkanDeviceCapabilities {
            device_name: unsafe {
                std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                    .to_string_lossy()
                    .to_string()
            },
            driver_version: format!(
                "{}.{}.{}",
                (properties.driver_version >> 22) & 0x3FF,
                (properties.driver_version >> 12) & 0x3FF,
                properties.driver_version & 0xFFF
            ),
            api_version: properties.api_version,
            max_compute_work_group_count: properties.limits.max_compute_work_group_count,
            max_compute_work_group_invocations: properties
                .limits
                .max_compute_work_group_invocations,
            max_compute_shared_memory_size: properties.limits.max_compute_shared_memory_size as u64,
            subgroup_size: subgroup_props.map_or(32, |p| p.subgroup_size), // 默认值 32
            supports_int16: features.shader_int16 == vk::TRUE,
            supports_int8: false,          // 需要 Vulkan 1.1+ 特性查询，简化处理
            supports_storage_buffer: true, // Vulkan 1.0+ 都支持
        }
    }

    /// 获取当前设备能力
    pub fn capabilities(&self) -> &VulkanDeviceCapabilities {
        &self.capabilities
    }

    /// 打印设备信息（用于调试）
    pub fn print_info(&self) {
        println!("=== Vulkan 设备信息 ===");
        println!("设备名称: {}", self.capabilities.device_name);
        println!("驱动版本: {}", self.capabilities.driver_version);
        println!(
            "API 版本: {}.{}.{}",
            vk::api_version_major(self.capabilities.api_version),
            vk::api_version_minor(self.capabilities.api_version),
            vk::api_version_patch(self.capabilities.api_version)
        );
        println!(
            "最大计算工作组数量: [{}, {}, {}]",
            self.capabilities.max_compute_work_group_count[0],
            self.capabilities.max_compute_work_group_count[1],
            self.capabilities.max_compute_work_group_count[2]
        );
        println!(
            "最大计算工作组成员: {}",
            self.capabilities.max_compute_work_group_invocations
        );
        println!(
            "最大共享内存: {} KB",
            self.capabilities.max_compute_shared_memory_size / 1024
        );
        println!("子组大小: {}", self.capabilities.subgroup_size);
        println!("支持 INT16: {}", self.capabilities.supports_int16);
        println!("支持 INT8: {}", self.capabilities.supports_int8);
        println!(
            "支持存储缓冲区: {}",
            self.capabilities.supports_storage_buffer
        );
        println!("======================");
    }

    /// 获取内部设备引用（用于高级操作）
    pub fn device(&self) -> &std::sync::Arc<VulkanDevice> {
        &self.device
    }

    /// 获取实例引用
    pub fn instance(&self) -> &std::sync::Arc<VulkanInstance> {
        &self.instance
    }

    /// 获取设备信息（用于 GpuOps 实现）
    pub fn device_info(&self) -> &GpuDeviceInfo {
        // TODO: 实现从 VulkanDevice 到 GpuDeviceInfo 的转换
        // 暂时返回一个默认值，避免编译错误
        static DEFAULT_INFO: GpuDeviceInfo = GpuDeviceInfo {
            name: String::new(),
            memory_size: 0,
            compute_capability: None,
            features: Vec::new(),
        };
        &DEFAULT_INFO
    }

    /// 检查设备是否支持计算操作
    pub fn is_compute_capable(&self) -> bool {
        // Vulkan 设备通常支持计算着色器
        // 这里可以根据具体能力返回
        true
    }
}

/// Vulkan 错误类型
#[derive(Debug)]
pub enum VulkanError {
    /// 入口点加载失败
    EntryLoad(String),
    /// 实例创建失败
    InstanceCreation(String),
    /// 设备创建失败
    DeviceCreation(String),
    /// 未找到设备
    NoDeviceFound(String),
    /// 枚举失败
    Enumeration(String),
    /// 缓冲区操作错误
    BufferError(String),
    /// 计算错误
    ComputeError(String),
    /// Shader 编译/链接错误
    ShaderError(String),
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::EntryLoad(msg) => write!(f, "Vulkan 入口点加载失败: {}", msg),
            VulkanError::InstanceCreation(msg) => write!(f, "Vulkan 实例创建失败: {}", msg),
            VulkanError::DeviceCreation(msg) => write!(f, "Vulkan 设备创建失败: {}", msg),
            VulkanError::NoDeviceFound(msg) => write!(f, "未找到 Vulkan 设备: {}", msg),
            VulkanError::Enumeration(msg) => write!(f, "Vulkan 设备枚举失败: {}", msg),
            VulkanError::BufferError(msg) => write!(f, "Vulkan 缓冲区错误: {}", msg),
            VulkanError::ComputeError(msg) => write!(f, "Vulkan 计算错误: {}", msg),
            VulkanError::ShaderError(msg) => write!(f, "Vulkan Shader 错误: {}", msg),
        }
    }
}

impl std::error::Error for VulkanError {}

// ============================================================================
// 增强的缓冲区管理（泛型版本）
// ============================================================================

use std::marker::PhantomData;

/// 缓冲区使用类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    /// 存储缓冲区
    StorageBuffer,
    /// 统一缓冲区
    UniformBuffer,
    /// 传输源
    TransferSrc,
    /// 传输目标
    TransferDst,
    /// 顶点缓冲区
    VertexBuffer,
    /// 索引缓冲区
    IndexBuffer,
}

impl BufferUsage {
    /// 转换为 Vulkan 标志
    pub fn to_vk_flags(&self) -> vk::BufferUsageFlags {
        match self {
            BufferUsage::StorageBuffer => vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferUsage::UniformBuffer => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferUsage::TransferSrc => vk::BufferUsageFlags::TRANSFER_SRC,
            BufferUsage::TransferDst => vk::BufferUsageFlags::TRANSFER_DST,
            BufferUsage::VertexBuffer => vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferUsage::IndexBuffer => vk::BufferUsageFlags::INDEX_BUFFER,
        }
    }
}

/// 泛型 Vulkan GPU 缓冲区
pub struct TypedVulkanBuffer<T: Copy> {
    /// 内部缓冲区
    buffer: VulkanBuffer,
    /// 元素数量
    count: usize,
    /// 类型标记
    _marker: PhantomData<T>,
}

impl<T: Copy + Default> TypedVulkanBuffer<T> {
    /// 创建新的 GPU 缓冲区
    pub fn new(
        device: &std::sync::Arc<VulkanDevice>,
        count: usize,
        _usage: BufferUsage,
    ) -> Result<Self, VulkanError> {
        let size = count * std::mem::size_of::<T>();
        let buffer = VulkanBuffer::new(device.clone(), size)
            .map_err(|e| VulkanError::BufferError(e.to_string()))?;

        Ok(Self {
            buffer,
            count,
            _marker: PhantomData,
        })
    }

    /// 从数据创建缓冲区
    pub fn from_data(
        device: &std::sync::Arc<VulkanDevice>,
        data: &[T],
    ) -> Result<Self, VulkanError> {
        let buffer = VulkanBuffer::from_data(device.clone(), data)
            .map_err(|e| VulkanError::BufferError(e.to_string()))?;

        Ok(Self {
            buffer,
            count: data.len(),
            _marker: PhantomData,
        })
    }

    /// 上传数据到 GPU
    pub fn upload(&self, data: &[T]) -> Result<(), VulkanError> {
        if data.len() > self.count {
            return Err(VulkanError::BufferError(format!(
                "数据大小超出缓冲区容量: {} > {}",
                data.len(),
                self.count
            )));
        }

        self.buffer
            .write(data)
            .map_err(|e| VulkanError::BufferError(e.to_string()))
    }

    /// 从 GPU 下载数据
    pub fn download(&self) -> Result<Vec<T>, VulkanError> {
        let mut data = vec![T::default(); self.count];
        self.buffer
            .read(&mut data)
            .map_err(|e| VulkanError::BufferError(e.to_string()))?;
        Ok(data)
    }

    /// 获取元素数量
    pub fn len(&self) -> usize {
        self.count
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// 获取 Vulkan buffer handle
    pub fn handle(&self) -> vk::Buffer {
        self.buffer.buffer()
    }

    /// 获取缓冲区大小（字节）
    pub fn size(&self) -> usize {
        self.buffer.size()
    }

    /// 获取内部缓冲区引用
    pub fn inner(&self) -> &VulkanBuffer {
        &self.buffer
    }
}

// ============================================================================
// Vulkan 后端实现
// ============================================================================

/// Vulkan 后端
pub struct VulkanBackend {
    /// Vulkan 设备
    device: std::sync::Arc<VulkanDevice>,
    /// 命令队列
    queue: std::sync::Arc<VulkanQueue>,
    /// Pipeline 管理器
    pipeline_manager: RwLock<PipelineManager>,
    /// Descriptor Pool
    descriptor_pool: RwLock<DescriptorPool>,
    /// 命令缓冲区池
    command_buffer_pool: CommandBufferPool,
    /// Fence 池
    fence_pool: FencePool,
    /// DescriptorSet 池
    descriptor_set_pool: DescriptorSetPool,
}

impl VulkanBackend {
    /// 创建 Vulkan 后端
    pub fn new() -> Result<Self> {
        let instance = std::sync::Arc::new(VulkanInstance::new()?);
        let device = std::sync::Arc::new(VulkanDevice::new(instance)?);
        let queue = std::sync::Arc::new(VulkanQueue::new(device.clone())?);
        let pipeline_manager = PipelineManager::new(device.clone())?;
        let descriptor_pool = DescriptorPool::new(device.clone(), 200)?;

        let command_buffer_pool = CommandBufferPool::new(device.clone(), queue.command_pool(), 16)?;

        let fence_pool = FencePool::new(device.clone(), 16)?;
        let descriptor_set_pool = DescriptorSetPool::new(device.clone())?;

        Ok(Self {
            device,
            queue,
            pipeline_manager: RwLock::new(pipeline_manager),
            descriptor_pool: RwLock::new(descriptor_pool),
            command_buffer_pool,
            fence_pool,
            descriptor_set_pool,
        })
    }

    /// 执行计算 kernel
    fn execute_kernel(
        &self,
        shader_type: ShaderType,
        buffers: &[&VulkanBuffer],
        binding_count: u32,
        group_count: (u32, u32, u32),
    ) -> Result<()> {
        let (pipeline, pipeline_layout, descriptor_set_layout) = {
            let manager = self
                .pipeline_manager
                .read()
                .map_err(|_| anyhow::anyhow!("获取 pipeline_manager 读锁失败"))?;

            if let Some(result) = manager.get_pipeline_cached(shader_type) {
                result
            } else {
                drop(manager);

                let mut manager = self
                    .pipeline_manager
                    .write()
                    .map_err(|_| anyhow::anyhow!("获取 pipeline_manager 写锁失败"))?;
                manager.get_pipeline(shader_type, binding_count)?
            }
        };

        let descriptor_set =
            if let Some(ds) = self.descriptor_set_pool.acquire(descriptor_set_layout) {
                ds
            } else {
                let mut descriptor_pool = self
                    .descriptor_pool
                    .write()
                    .map_err(|_| anyhow::anyhow!("获取 descriptor_pool 写锁失败"))?;
                descriptor_pool.allocate_descriptor_set(descriptor_set_layout)?
            };

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer())
                    .offset(0)
                    .range(b.size() as u64)
            })
            .collect();

        {
            let descriptor_pool = self
                .descriptor_pool
                .read()
                .map_err(|_| anyhow::anyhow!("获取 descriptor_pool 读锁失败"))?;
            descriptor_pool.update_descriptor_set(descriptor_set, &buffer_infos)?;
        }

        let command_buffer = self.command_buffer_pool.acquire()?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .device()
                .begin_command_buffer(command_buffer, &begin_info)?;

            self.device.device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );

            self.device.device().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            self.device.device().cmd_dispatch(
                command_buffer,
                group_count.0,
                group_count.1,
                group_count.2,
            );

            self.device.device().end_command_buffer(command_buffer)?;
        }

        let fence = self.fence_pool.acquire()?;

        self.queue.submit(command_buffer, fence)?;

        unsafe {
            self.device
                .device()
                .wait_for_fences(&[fence], true, u64::MAX)?;
        }

        self.fence_pool.release(fence);
        self.command_buffer_pool.release(command_buffer);
        self.descriptor_set_pool
            .release(descriptor_set, descriptor_set_layout);

        Ok(())
    }

    /// 矩阵乘法 (Vulkan 实现)
    fn matmul_vulkan(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            bail!("矩阵维度不匹配: {} vs {}", k1, k2);
        }

        let k = k1;

        let a_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            a.as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵 A 不是连续存储"))?,
        )?;
        let b_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            b.as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵 B 不是连续存储"))?,
        )?;
        let c_buffer = VulkanBuffer::new(self.device.clone(), m * n * std::mem::size_of::<f32>())?;

        let dims = [m as u32, n as u32, k as u32];
        let dims_buffer = VulkanBuffer::from_data(self.device.clone(), &dims)?;

        let shader_type = if m >= 32 && n >= 32 && k >= 32 {
            ShaderType::MatmulBlocked
        } else {
            ShaderType::Matmul
        };

        let group_count = (
            (m as u32).div_ceil(MATMUL_BLOCK_SIZE),
            (n as u32).div_ceil(MATMUL_BLOCK_SIZE),
            1,
        );

        self.execute_kernel(
            shader_type,
            &[&a_buffer, &b_buffer, &c_buffer, &dims_buffer],
            4,
            group_count,
        )?;

        let mut result = vec![0.0f32; m * n];
        c_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((m, n), result)?)
    }

    /// Softmax (Vulkan 实现)
    fn softmax_vulkan(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let (rows, cols) = input.dim();

        let input_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            input
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("输入矩阵不是连续存储"))?,
        )?;
        let output_buffer = VulkanBuffer::new(
            self.device.clone(),
            rows * cols * std::mem::size_of::<f32>(),
        )?;

        let dims = [rows as u32, cols as u32];
        let dims_buffer = VulkanBuffer::from_data(self.device.clone(), &dims)?;

        let group_count = ((rows as u32).div_ceil(SOFTMAX_BLOCK_SIZE), 1, 1);

        self.execute_kernel(
            ShaderType::Softmax,
            &[&input_buffer, &output_buffer, &dims_buffer],
            3,
            group_count,
        )?;

        let mut result = vec![0.0f32; rows * cols];
        output_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((rows, cols), result)?)
    }

    /// Layer Normalization (Vulkan 实现)
    fn layer_norm_vulkan(
        &self,
        input: &Array2<f32>,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<Array2<f32>> {
        let (rows, cols) = input.dim();

        if gamma.len() != cols || beta.len() != cols {
            bail!("Layer norm 参数大小不匹配");
        }

        let input_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            input
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("输入矩阵不是连续存储"))?,
        )?;
        let output_buffer = VulkanBuffer::new(
            self.device.clone(),
            rows * cols * std::mem::size_of::<f32>(),
        )?;
        let gamma_buffer = VulkanBuffer::from_data(self.device.clone(), gamma)?;
        let beta_buffer = VulkanBuffer::from_data(self.device.clone(), beta)?;

        let dims = [rows as u32, cols as u32];
        let dims_buffer = VulkanBuffer::from_data(self.device.clone(), &dims)?;
        let eps_buffer = VulkanBuffer::from_data(self.device.clone(), &[eps])?;

        let group_count = ((rows as u32).div_ceil(LAYERNORM_BLOCK_SIZE), 1, 1);

        self.execute_kernel(
            ShaderType::LayerNorm,
            &[
                &input_buffer,
                &output_buffer,
                &gamma_buffer,
                &beta_buffer,
                &dims_buffer,
                &eps_buffer,
            ],
            6,
            group_count,
        )?;

        let mut result = vec![0.0f32; rows * cols];
        output_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((rows, cols), result)?)
    }

    /// RMS Normalization (Vulkan 实现)
    pub fn rms_norm(&self, input: &Array2<f32>, gamma: &[f32], eps: f32) -> Result<Array2<f32>> {
        let (rows, cols) = input.dim();

        if gamma.len() != cols {
            bail!("RMS norm 参数大小不匹配");
        }

        let input_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            input
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("输入矩阵不是连续存储"))?,
        )?;
        let output_buffer = VulkanBuffer::new(
            self.device.clone(),
            rows * cols * std::mem::size_of::<f32>(),
        )?;
        let gamma_buffer = VulkanBuffer::from_data(self.device.clone(), gamma)?;

        let dims = [rows as u32, cols as u32];
        let dims_buffer = VulkanBuffer::from_data(self.device.clone(), &dims)?;
        let eps_buffer = VulkanBuffer::from_data(self.device.clone(), &[eps])?;

        let group_count = ((rows as u32).div_ceil(LAYERNORM_BLOCK_SIZE), 1, 1);

        self.execute_kernel(
            ShaderType::RmsNorm,
            &[
                &input_buffer,
                &output_buffer,
                &gamma_buffer,
                &dims_buffer,
                &eps_buffer,
            ],
            5,
            group_count,
        )?;

        let mut result = vec![0.0f32; rows * cols];
        output_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((rows, cols), result)?)
    }

    /// Attention (Vulkan 实现)
    fn attention_vulkan(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>> {
        let (seq_len, head_dim) = query.dim();
        let (kv_len, _) = key.dim();

        let scale = 1.0 / (head_dim as f32).sqrt();

        let query_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            query
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("query 矩阵不是连续存储"))?,
        )?;
        let key_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            key.as_slice()
                .ok_or_else(|| anyhow::anyhow!("key 矩阵不是连续存储"))?,
        )?;
        let value_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            value
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("value 矩阵不是连续存储"))?,
        )?;
        let output_buffer = VulkanBuffer::new(
            self.device.clone(),
            seq_len * head_dim * std::mem::size_of::<f32>(),
        )?;

        let mask_buffer = if let Some(m) = mask {
            VulkanBuffer::from_data(
                self.device.clone(),
                m.as_slice()
                    .ok_or_else(|| anyhow::anyhow!("mask 矩阵不是连续存储"))?,
            )?
        } else {
            VulkanBuffer::from_data(self.device.clone(), &vec![0.0f32; seq_len * kv_len])?
        };

        let dims = [seq_len as u32, kv_len as u32, head_dim as u32];
        let dims_buffer = VulkanBuffer::from_data(self.device.clone(), &dims)?;
        let scale_buffer = VulkanBuffer::from_data(self.device.clone(), &[scale])?;
        let has_mask: u32 = if mask.is_some() { 1 } else { 0 };
        let has_mask_buffer = VulkanBuffer::from_data(self.device.clone(), &[has_mask])?;

        let group_count = (
            (seq_len as u32).div_ceil(ATTENTION_BLOCK_SIZE),
            (head_dim as u32).div_ceil(ATTENTION_BLOCK_SIZE),
            1,
        );

        self.execute_kernel(
            ShaderType::Attention,
            &[
                &query_buffer,
                &key_buffer,
                &value_buffer,
                &output_buffer,
                &mask_buffer,
                &dims_buffer,
                &scale_buffer,
                &has_mask_buffer,
            ],
            8,
            group_count,
        )?;

        let mut result = vec![0.0f32; seq_len * head_dim];
        output_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((seq_len, head_dim), result)?)
    }

    /// 带 KV Cache 的 Attention
    pub fn attention_with_kv_cache(
        &self,
        query: &Array2<f32>,
        key_cache: &Array2<f32>,
        value_cache: &Array2<f32>,
        kv_len: usize,
    ) -> Result<Array2<f32>> {
        let (_, head_dim) = query.dim();

        let scale = 1.0 / (head_dim as f32).sqrt();

        let query_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            query
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("query 矩阵不是连续存储"))?,
        )?;
        let key_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            key_cache
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("key_cache 矩阵不是连续存储"))?,
        )?;
        let value_buffer = VulkanBuffer::from_data(
            self.device.clone(),
            value_cache
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("value_cache 矩阵不是连续存储"))?,
        )?;
        let output_buffer =
            VulkanBuffer::new(self.device.clone(), head_dim * std::mem::size_of::<f32>())?;

        let cache_info = [kv_len as u32, head_dim as u32];
        let cache_info_buffer = VulkanBuffer::from_data(self.device.clone(), &cache_info)?;
        let scale_buffer = VulkanBuffer::from_data(self.device.clone(), &[scale])?;

        let group_count = ((head_dim as u32 + 255) / 256, 1, 1);

        self.execute_kernel(
            ShaderType::AttentionKvCache,
            &[
                &query_buffer,
                &key_buffer,
                &value_buffer,
                &output_buffer,
                &cache_info_buffer,
                &scale_buffer,
            ],
            6,
            group_count,
        )?;

        let mut result = vec![0.0f32; head_dim];
        output_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((1, head_dim), result)?)
    }
}

impl GpuOps for VulkanBackend {
    fn device_info(&self) -> &GpuDeviceInfo {
        self.device.info()
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        self.matmul_vulkan(a, b)
    }

    fn batch_matmul(&self, a: &[Array2<f32>], b: &[Array2<f32>]) -> Result<Vec<Array2<f32>>> {
        if a.len() != b.len() {
            bail!("批量大小不匹配: {} vs {}", a.len(), b.len());
        }

        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| self.matmul(ai, bi))
            .collect()
    }

    fn softmax(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.softmax_vulkan(input)
    }

    fn layer_norm(
        &self,
        input: &Array2<f32>,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<Array2<f32>> {
        self.layer_norm_vulkan(input, gamma, beta, eps)
    }

    fn attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>> {
        self.attention_vulkan(query, key, value, mask)
    }

    fn synchronize(&self) -> Result<()> {
        self.queue.wait_idle()
    }

    fn available_memory(&self) -> Result<usize> {
        Ok(self.device.info().memory_size)
    }
}

impl VulkanBackend {
    /// 获取内部设备引用（用于高级操作和测试）
    pub fn device(&self) -> &std::sync::Arc<VulkanDevice> {
        &self.device
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn get_backend() -> VulkanBackend {
        VulkanBackend::new().expect("无法创建 Vulkan 后端")
    }

    #[test]
    fn test_vulkan_backend_creation() {
        let backend = VulkanBackend::new();
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        let info = backend.device_info();
        assert!(!info.name.is_empty());
        assert!(info.memory_size > 0);
    }

    #[test]
    fn test_vulkan_device_info() {
        let backend = get_backend();
        let info = backend.device_info();

        println!("设备名称: {}", info.name);
        println!("显存大小: {} GB", info.memory_size / (1024 * 1024 * 1024));
        println!("支持特性: {:?}", info.features);
    }

    #[test]
    fn test_matmul_small() {
        let backend = get_backend();

        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let result = backend.matmul(&a, &b).unwrap();

        assert_eq!(result.dim(), (2, 2));
        assert!((result[[0, 0]] - 58.0).abs() < 1e-3);
        assert!((result[[0, 1]] - 64.0).abs() < 1e-3);
        assert!((result[[1, 0]] - 139.0).abs() < 1e-3);
        assert!((result[[1, 1]] - 154.0).abs() < 1e-3);
    }

    #[test]
    fn test_matmul_large() {
        let backend = get_backend();

        let m = 64;
        let k = 128;
        let n = 64;

        let a = Array2::from_shape_fn((m, k), |(i, j)| ((i * k + j) % 10) as f32);
        let b = Array2::from_shape_fn((k, n), |(i, j)| ((i * n + j) % 10) as f32);

        let result = backend.matmul(&a, &b).unwrap();
        assert_eq!(result.dim(), (m, n));
    }

    #[test]
    fn test_softmax() {
        let backend = get_backend();

        let input =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let result = backend.softmax(&input).unwrap();

        let sum_row0: f32 = result.row(0).sum();
        let sum_row1: f32 = result.row(1).sum();

        assert!((sum_row0 - 1.0).abs() < 1e-5);
        assert!((sum_row1 - 1.0).abs() < 1e-5);

        for elem in result.iter() {
            assert!(*elem > 0.0 && *elem < 1.0);
        }
    }

    #[test]
    fn test_layer_norm() {
        let backend = get_backend();

        let input =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];
        let eps = 1e-5;

        let result = backend.layer_norm(&input, &gamma, &beta, eps).unwrap();

        let mean_row0: f32 = result.row(0).sum() / 4.0;
        let mean_row1: f32 = result.row(1).sum() / 4.0;

        assert!(mean_row0.abs() < 1e-5);
        assert!(mean_row1.abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm() {
        let backend = get_backend();

        let input =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-5;

        let result = backend.rms_norm(&input, &gamma, eps).unwrap();

        assert_eq!(result.dim(), (2, 4));
    }

    #[test]
    fn test_attention() {
        let backend = get_backend();

        let seq_len = 4;
        let kv_len = 4;
        let head_dim = 8;

        let query = Array2::zeros((seq_len, head_dim));
        let key = Array2::zeros((kv_len, head_dim));
        let value = Array2::zeros((kv_len, head_dim));

        let result = backend.attention(&query, &key, &value, None).unwrap();

        assert_eq!(result.dim(), (seq_len, head_dim));
    }

    #[test]
    fn test_attention_with_mask() {
        let backend = get_backend();

        let seq_len = 4;
        let kv_len = 4;
        let head_dim = 8;

        let query = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| ((i + j) as f32));
        let key = Array2::from_shape_fn((kv_len, head_dim), |(i, j)| ((i + j) as f32));
        let value = Array2::from_shape_fn((kv_len, head_dim), |(i, j)| ((i + j) as f32));

        let mask = Array2::from_shape_fn((seq_len, kv_len), |(i, j)| {
            if j <= i {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        });

        let result = backend
            .attention(&query, &key, &value, Some(&mask))
            .unwrap();

        assert_eq!(result.dim(), (seq_len, head_dim));
    }

    #[test]
    fn test_attention_kv_cache() {
        let backend = get_backend();

        let head_dim = 8;
        let max_seq = 16;
        let kv_len = 4;

        let query = Array2::zeros((1, head_dim));
        let key_cache = Array2::zeros((max_seq, head_dim));
        let value_cache = Array2::zeros((max_seq, head_dim));

        let result = backend
            .attention_with_kv_cache(&query, &key_cache, &value_cache, kv_len)
            .unwrap();

        assert_eq!(result.dim(), (1, head_dim));
    }

    // 新增分支覆盖测试

    /// 测试 matmul 的维度不匹配错误分支
    #[test]
    fn test_matmul_dimension_mismatch() {
        let backend = get_backend();

        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let b = Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap(); // K 不匹配

        let result = backend.matmul(&a, &b);
        assert!(result.is_err(), "矩阵维度不匹配应返回错误");
    }

    /// 测试 rms_norm 的参数不匹配错误分支
    #[test]
    fn test_rms_norm_dimension_mismatch() {
        let backend = get_backend();

        let input =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let gamma = vec![1.0, 2.0]; // 长度不匹配

        let result = backend.rms_norm(&input, &gamma, 1e-5);
        assert!(result.is_err(), "gamma 长度不匹配应返回错误");
    }

    /// 测试 layer_norm 的参数不匹配错误分支
    #[test]
    fn test_layer_norm_dimension_mismatch() {
        let backend = get_backend();

        let input =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let gamma = vec![1.0; 3]; // 长度不匹配
        let beta = vec![0.0; 3];

        let result = backend.layer_norm(&input, &gamma, &beta, 1e-5);
        assert!(result.is_err(), "gamma/beta 长度不匹配应返回错误");
    }

    /// 测试 attention_with_kv_cache 的正常情况（非零输出）
    #[test]
    fn test_attention_kv_cache_nonzero_output() {
        let backend = get_backend();

        let head_dim = 8;
        let max_seq = 16;
        let kv_len = 4;

        let query = Array2::from_shape_fn((1, head_dim), |(_, j)| (j as f32));
        let key_cache = Array2::from_shape_fn((max_seq, head_dim), |(i, j)| ((i + j) as f32));
        let value_cache = Array2::from_shape_fn((max_seq, head_dim), |(i, j)| ((i + j) as f32));

        let result = backend
            .attention_with_kv_cache(&query, &key_cache, &value_cache, kv_len)
            .unwrap();

        assert_eq!(result.dim(), (1, head_dim));
        // 结果不应全为零（因为输入非零）
        let sum: f32 = result.iter().sum();
        assert!(sum != 0.0 || sum.is_nan(), "非零输入应产生非零输出");
    }

    /// 测试 batch_matmul 方法
    #[test]
    fn test_batch_matmul() {
        let backend = get_backend();

        let a = vec![
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            Array2::from_shape_vec((2, 3), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
        ];

        let b = vec![
            Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            Array2::from_shape_vec((3, 2), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
        ];

        let results = backend.batch_matmul(&a, &b).unwrap();

        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.dim(), (2, 2));
        }
    }

    /// 测试 batch_matmul 的批量大小不匹配错误分支
    #[test]
    fn test_batch_matmul_size_mismatch() {
        let backend = get_backend();

        let a = vec![Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap()];

        let b = vec![
            Array2::from_shape_vec((3, 2), vec![1.0; 6]).unwrap(),
            Array2::from_shape_vec((3, 2), vec![2.0; 6]).unwrap(), // 多一个
        ];

        let result = backend.batch_matmul(&a, &b);
        assert!(result.is_err(), "批量大小不匹配应返回错误");
    }

    /// 测试 synchronize 方法
    #[test]
    fn test_synchronize() {
        let backend = get_backend();

        let result = backend.synchronize();
        assert!(result.is_ok(), "同步操作应成功");
    }

    /// 测试 available_memory 方法
    #[test]
    fn test_available_memory() {
        let backend = get_backend();

        let mem = backend.available_memory().unwrap();
        assert!(mem > 0, "可用内存应大于 0");
    }

    // ==================== Phase 1 新增测试 ====================

    /// 测试 Vulkan 设备枚举功能
    #[test]
    fn test_vulkan_device_enumeration() {
        // 测试设备枚举（如果有Vulkan设备）
        let devices = VulkanGpu::enumerate_devices();
        // 至少不应该panic
        assert!(devices.is_ok(), "设备枚举不应失败");

        if let Ok(devs) = devices {
            // 如果有设备，验证基本信息
            for device in &devs {
                assert!(!device.device_name.is_empty(), "设备名称不应为空");
                println!(
                    "发现设备: {} (显存: {} MB)",
                    device.device_name,
                    device.memory_size / (1024 * 1024)
                );
            }
        }
    }

    /// 测试 Vulkan 设备创建
    #[test]
    fn test_vulkan_device_creation() {
        // 尝试创建Vulkan实例（可能在没有GPU的环境失败）
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let caps = gpu.capabilities();
                println!("Vulkan Device: {}", caps.device_name);
                assert!(!caps.device_name.is_empty(), "设备名称不应为空");
                assert!(caps.api_version > 0, "API版本应大于0");
                assert!(
                    caps.max_compute_work_group_invocations > 0,
                    "最大工作组成员应大于0"
                );

                // 打印完整信息
                gpu.print_info();
            }
            Err(e) => {
                // 无Vulkan环境时跳过
                eprintln!("Skipping test: No Vulkan device available - {}", e);
            }
        }
    }

    /// 测试 Vulkan 设备能力查询
    #[test]
    fn test_vulkan_capabilities() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let caps = gpu.capabilities();

                // 验证能力信息的合理性
                assert!(!caps.device_name.is_empty());
                assert!(!caps.driver_version.is_empty());
                assert!(caps.api_version >= vk::make_api_version(0, 1, 0, 0));
                assert!(caps.max_compute_shared_memory_size > 0);
                assert!(caps.subgroup_size > 0);

                // 验证工作组限制的合理性
                for count in caps.max_compute_work_group_count.iter() {
                    assert!(*count > 0, "最大工作组数量应大于0");
                }
            }
            Err(_) => {
                eprintln!("Skipping test: No Vulkan device available");
            }
        }
    }

    /// 测试最佳设备选择
    #[test]
    fn test_select_best_device() {
        match VulkanGpu::select_best_device() {
            Ok(gpu) => {
                let caps = gpu.capabilities();
                println!("选择的最佳设备: {}", caps.device_name);
                assert!(!caps.device_name.is_empty());
            }
            Err(VulkanError::NoDeviceFound(_)) => {
                eprintln!("Skipping test: No Vulkan devices found");
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    /// 测试泛型缓冲区创建和操作
    #[test]
    fn test_typed_buffer_operations() {
        let backend = get_backend();
        let device = backend.device();

        // 创建 f32 缓冲区
        let buffer_result = TypedVulkanBuffer::<f32>::new(device, 10, BufferUsage::StorageBuffer);
        assert!(buffer_result.is_ok(), "缓冲区创建应成功");

        let buffer = buffer_result.unwrap();
        assert_eq!(buffer.len(), 10, "缓冲区长度应为10");
        assert!(!buffer.is_empty(), "缓冲区不应为空");

        // 从数据创建缓冲区
        let data: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let data_buffer = TypedVulkanBuffer::from_data(device, &data).unwrap();
        assert_eq!(data_buffer.len(), 5);

        // 上传数据
        let upload_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let upload_result = buffer.upload(&upload_data);
        assert!(upload_result.is_ok(), "数据上传应成功");

        // 下载数据
        let download_result = buffer.download();
        assert!(download_result.is_ok(), "数据下载应成功");

        let downloaded = download_result.unwrap();
        assert_eq!(downloaded.len(), 10, "下载数据长度应为10");
        // 验证前5个元素
        for i in 0..5 {
            assert!(
                (downloaded[i] - upload_data[i]).abs() < 1e-6,
                "数据不匹配: expected {}, got {}",
                upload_data[i],
                downloaded[i]
            );
        }
    }

    /// 测试缓冲区错误处理
    #[test]
    fn test_typed_buffer_errors() {
        let backend = get_backend();
        let device = backend.device();

        // 创建小缓冲区
        let buffer = TypedVulkanBuffer::<f32>::new(device, 3, BufferUsage::StorageBuffer).unwrap();

        // 尝试上传过多数据
        let too_much_data: Vec<f32> = vec![1.0; 5];
        let result = buffer.upload(&too_much_data);
        assert!(result.is_err(), "上传过多数据应返回错误");

        if let Err(VulkanError::BufferError(msg)) = result {
            assert!(msg.contains("超出"), "错误信息应包含'超出'");
        } else {
            panic!("期望 BufferError");
        }
    }

    /// 测试 BufferUsage 枚举
    #[test]
    fn test_buffer_usage_enum() {
        let usages = [
            BufferUsage::StorageBuffer,
            BufferUsage::UniformBuffer,
            BufferUsage::TransferSrc,
            BufferUsage::TransferDst,
            BufferUsage::VertexBuffer,
            BufferUsage::IndexBuffer,
        ];

        for usage in &usages {
            // 验证可以转换为 Vulkan 标志
            let _flags = usage.to_vk_flags();

            // 验证 Debug trait 实现
            let _ = format!("{:?}", usage);
        }

        // 验证相等性
        assert_eq!(BufferUsage::StorageBuffer, BufferUsage::StorageBuffer);
        assert_ne!(BufferUsage::StorageBuffer, BufferUsage::UniformBuffer);
    }

    /// 测试 VulkanError Display trait
    #[test]
    fn test_vulkan_error_display() {
        let errors = [
            VulkanError::EntryLoad("test".to_string()),
            VulkanError::InstanceCreation("test".to_string()),
            VulkanError::DeviceCreation("test".to_string()),
            VulkanError::NoDeviceFound("test".to_string()),
            VulkanError::Enumeration("test".to_string()),
            VulkanError::BufferError("test".to_string()),
            VulkanError::ComputeError("test".to_string()),
        ];

        for error in &errors {
            let display = format!("{}", error);
            assert!(!display.is_empty(), "Display 输出不应为空");
            assert!(display.contains("test"), "Display 应包含错误详情");
        }
    }

    /// 测试 VulkanDeviceCapabilities Debug trait
    #[test]
    fn test_capabilities_debug() {
        let caps = VulkanDeviceCapabilities {
            device_name: "Test GPU".to_string(),
            driver_version: "1.2.3".to_string(),
            api_version: vk::make_api_version(0, 1, 2, 0),
            max_compute_work_group_count: [65535, 65535, 65535],
            max_compute_work_group_invocations: 1024,
            max_compute_shared_memory_size: 65536,
            subgroup_size: 32,
            supports_int16: true,
            supports_int8: false,
            supports_storage_buffer: true,
        };

        let debug_str = format!("{:?}", caps);
        assert!(debug_str.contains("Test GPU"));
        assert!(debug_str.contains("1.2.3"));
    }

    /// 测试 VulkanDeviceInfo 完整字段
    #[test]
    fn test_device_info_fields() {
        match VulkanGpu::enumerate_devices() {
            Ok(devices) => {
                if !devices.is_empty() {
                    let device = &devices[0];
                    assert!(!device.device_name.is_empty());
                    assert!(
                        device.memory_size > 0 || device.device_type == vk::PhysicalDeviceType::CPU
                    );

                    // 验证能力信息存在
                    let _caps = &device.capabilities;
                }
            }
            Err(_) => {
                eprintln!("Skipping test: No Vulkan devices");
            }
        }
    }
}

// ============================================================================
// 性能基准测试
// ============================================================================

#[cfg(test)]
mod benches {
    use super::*;
    use std::time::Instant;

    fn benchmark_matmul(backend: &VulkanBackend, m: usize, k: usize, n: usize, iterations: usize) {
        let a = Array2::from_shape_fn((m, k), |(i, j)| ((i * k + j) % 100) as f32 / 100.0);
        let b = Array2::from_shape_fn((k, n), |(i, j)| ((i * n + j) % 100) as f32 / 100.0);

        let _ = backend.matmul(&a, &b);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = backend.matmul(&a, &b);
        }
        let elapsed = start.elapsed();

        let ops = 2.0 * m as f64 * k as f64 * n as f64 * iterations as f64;
        let gflops = ops / elapsed.as_secs_f64() / 1e9;

        println!(
            "Matmul {}x{}x{}: {:?} ({:.2} GFLOPS)",
            m,
            k,
            n,
            elapsed / iterations as u32,
            gflops
        );
    }

    fn benchmark_softmax(backend: &VulkanBackend, rows: usize, cols: usize, iterations: usize) {
        let input =
            Array2::from_shape_fn((rows, cols), |(i, j)| ((i * cols + j) % 100) as f32 / 100.0);

        let _ = backend.softmax(&input);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = backend.softmax(&input);
        }
        let elapsed = start.elapsed();

        println!(
            "Softmax {}x{}: {:?}",
            rows,
            cols,
            elapsed / iterations as u32
        );
    }

    fn benchmark_attention(
        backend: &VulkanBackend,
        seq_len: usize,
        head_dim: usize,
        iterations: usize,
    ) {
        let query = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| ((i + j) as f32));
        let key = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| ((i + j) as f32));
        let value = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| ((i + j) as f32));

        let _ = backend.attention(&query, &key, &value, None);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = backend.attention(&query, &key, &value, None);
        }
        let elapsed = start.elapsed();

        println!(
            "Attention seq_len={}, head_dim={}: {:?}",
            seq_len,
            head_dim,
            elapsed / iterations as u32
        );
    }

    #[test]
    fn run_benchmarks() {
        let backend = VulkanBackend::new().expect("无法创建 Vulkan 后端");

        println!("\n=== Vulkan GPU 性能基准测试 ===\n");

        println!("--- 矩阵乘法 ---");
        benchmark_matmul(&backend, 64, 64, 64, 100);
        benchmark_matmul(&backend, 128, 128, 128, 100);
        benchmark_matmul(&backend, 256, 256, 256, 50);
        benchmark_matmul(&backend, 512, 512, 512, 20);
        benchmark_matmul(&backend, 1024, 1024, 1024, 10);

        println!("\n--- Softmax ---");
        benchmark_softmax(&backend, 128, 128, 100);
        benchmark_softmax(&backend, 256, 256, 100);
        benchmark_softmax(&backend, 512, 512, 100);
        benchmark_softmax(&backend, 1024, 1024, 50);

        println!("\n--- Attention ---");
        benchmark_attention(&backend, 64, 64, 100);
        benchmark_attention(&backend, 128, 64, 100);
        benchmark_attention(&backend, 256, 64, 50);
        benchmark_attention(&backend, 512, 64, 20);

        println!("\n=== 基准测试完成 ===\n");
    }
}
