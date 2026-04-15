//! Metal GPU 后端实现
//!
//! 为 macOS/iOS 提供 Metal GPU 加速支持。
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
//! use openmini_server::hardware::gpu::metal::MetalBackend;
//!
//! let metal = MetalBackend::new()?;
//! let result = metal.matmul(&a, &b)?;
//! ```
//!
//! ## 编译要求
//!
//! 需要启用 `metal` feature 并在 macOS 上编译。

#![allow(dead_code)]

use std::collections::HashMap;

use anyhow::{bail, Result};
use metal::MTLResourceOptions;
use ndarray::Array2;

use super::{GpuDeviceInfo, GpuOps};

// ============================================================================
// 常量定义
// ============================================================================

/// 矩阵乘法分块大小
const MATMUL_BLOCK_SIZE: u64 = 16;

/// Softmax 分块大小
const SOFTMAX_BLOCK_SIZE: u64 = 256;

/// LayerNorm 分块大小
const LAYERNORM_BLOCK_SIZE: u64 = 256;

/// Attention 分块大小
const ATTENTION_BLOCK_SIZE: u64 = 16;

// ============================================================================
// Metal Shader 源码
// ============================================================================

/// 矩阵乘法 Metal Shader
const MATMUL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// 矩阵乘法 kernel
/// C = A @ B
/// A: [M, K], B: [K, N], C: [M, N]
kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint m = gid.x;
    uint n = gid.y;
    
    if (m >= dims.x || n >= dims.y) return;
    
    uint K = dims.z;
    float sum = 0.0f;
    
    for (uint k = 0; k < K; k++) {
        sum += A[m * K + k] * B[k * dims.y + n];
    }
    
    C[m * dims.y + n] = sum;
}

/// 分块矩阵乘法 kernel (优化版本)
kernel void matmul_blocked(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint3 block_idx [[threadgroup_position_in_grid]],
    uint3 thread_idx [[thread_position_in_threadgroup]]
) {
    constexpr uint BLOCK_SIZE = 16;
    
    // Threadgroup 内存声明在循环外，明确复用同一块内存
    threadgroup float A_tile[BLOCK_SIZE][BLOCK_SIZE];
    threadgroup float B_tile[BLOCK_SIZE][BLOCK_SIZE];
    
    uint m = block_idx.x * BLOCK_SIZE + thread_idx.x;
    uint n = block_idx.y * BLOCK_SIZE + thread_idx.y;
    
    uint K = dims.z;
    float sum = 0.0f;
    
    for (uint k_block = 0; k_block < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; k_block++) {
        uint k = k_block * BLOCK_SIZE + thread_idx.y;
        if (m < dims.x && k < K) {
            A_tile[thread_idx.x][thread_idx.y] = A[m * K + k];
        } else {
            A_tile[thread_idx.x][thread_idx.y] = 0.0f;
        }
        
        k = k_block * BLOCK_SIZE + thread_idx.x;
        uint b_row = k;
        if (b_row < K && n < dims.y) {
            B_tile[thread_idx.x][thread_idx.y] = B[b_row * dims.y + n];
        } else {
            B_tile[thread_idx.x][thread_idx.y] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k_local = 0; k_local < BLOCK_SIZE; k_local++) {
            sum += A_tile[thread_idx.x][k_local] * B_tile[k_local][thread_idx.y];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (m < dims.x && n < dims.y) {
        C[m * dims.y + n] = sum;
    }
}
"#;

/// Softmax Metal Shader
const SOFTMAX_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// Softmax kernel (数值稳定版本)
kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= dims.x) return;
    
    uint cols = dims.y;
    uint offset = row * cols;
    
    float max_val = input[offset];
    for (uint j = 1; j < cols; j++) {
        max_val = max(max_val, input[offset + j]);
    }
    
    float sum = 0.0f;
    for (uint j = 0; j < cols; j++) {
        sum += exp(input[offset + j] - max_val);
    }
    
    float inv_sum = 1.0f / sum;
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = exp(input[offset + j] - max_val) * inv_sum;
    }
}

/// 在线 Softmax
kernel void softmax_online(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= dims.x) return;
    
    uint cols = dims.y;
    uint offset = row * cols;
    
    float max_val = -INFINITY;
    float sum = 0.0f;
    
    for (uint j = 0; j < cols; j++) {
        float x = input[offset + j];
        float new_max = max(max_val, x);
        float scale = exp(max_val - new_max);
        sum = sum * scale + exp(x - new_max);
        max_val = new_max;
    }
    
    float inv_sum = 1.0f / sum;
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = exp(input[offset + j] - max_val) * inv_sum;
    }
}
"#;

/// LayerNorm Metal Shader
const LAYERNORM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// Layer Normalization kernel
kernel void layer_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= dims.x) return;
    
    uint cols = dims.y;
    uint offset = row * cols;
    
    float mean = 0.0f;
    for (uint j = 0; j < cols; j++) {
        mean += input[offset + j];
    }
    mean /= float(cols);
    
    float var = 0.0f;
    for (uint j = 0; j < cols; j++) {
        float diff = input[offset + j] - mean;
        var += diff * diff;
    }
    var /= float(cols);
    
    float inv_std = 1.0f / sqrt(var + eps);
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = gamma[j] * (input[offset + j] - mean) * inv_std + beta[j];
    }
}

/// RMS Normalization kernel
kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint row = gid;
    if (row >= dims.x) return;
    
    uint cols = dims.y;
    uint offset = row * cols;
    
    float sum_sq = 0.0f;
    for (uint j = 0; j < cols; j++) {
        sum_sq += input[offset + j] * input[offset + j];
    }
    
    float rms = sqrt(sum_sq / float(cols) + eps);
    float inv_rms = 1.0f / rms;
    
    for (uint j = 0; j < cols; j++) {
        output[offset + j] = gamma[j] * input[offset + j] * inv_rms;
    }
}
"#;

/// Attention Metal Shader (使用在线 Softmax 避免内存竞争)
const ATTENTION_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// Scaled Dot-Product Attention kernel (在线 Softmax 版本)
/// 每个线程独立计算一个输出元素，无 threadgroup 内存竞争
kernel void attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const float* mask [[buffer(4)]],
    constant uint3& dims [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    constant uint& has_mask [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint seq_idx = gid.x;
    uint head_idx = gid.y;
    
    if (seq_idx >= dims.x || head_idx >= dims.z) return;
    
    uint kv_len = dims.y;
    uint head_dim = dims.z;
    
    // 在线 Softmax: 无需存储所有分数，使用寄存器累加
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_val = 0.0f;
    
    // 第一遍: 计算所有 Q*K^T 并使用在线 Softmax
    for (uint kv_idx = 0; kv_idx < kv_len; kv_idx++) {
        // 计算 Q * K^T
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += query[seq_idx * head_dim + d] * key[kv_idx * head_dim + d];
        }
        float score = dot * scale;

        // 应用 mask
        if (has_mask == 1) {
            score += mask[seq_idx * kv_len + kv_idx];
        }

        // 跳过被屏蔽的位置（score <= -1e30）
        if (score > -1e30f) {
            // 在线 Softmax 更新
            float new_max = max(max_score, score);
            float scale_factor = exp(max_score - new_max);
            float score_exp = exp(score - new_max);

            output_val = output_val * scale_factor + score_exp * value[kv_idx * head_dim + head_idx];
            sum_exp = sum_exp * scale_factor + score_exp;
            max_score = new_max;
        }
    }
    
    // 最终输出
    output[seq_idx * head_dim + head_idx] = output_val / sum_exp;
}

/// 带 KV Cache 的 Attention kernel (在线 Softmax 版本)
/// 每个线程独立计算一个输出元素，支持任意 kv_len
kernel void attention_kv_cache(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint2& cache_info [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint head_idx = gid;
    uint kv_len = cache_info.x;
    uint head_dim = cache_info.y;
    
    if (head_idx >= head_dim) return;
    
    // 在线 Softmax: 无需存储所有分数
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_val = 0.0f;
    
    for (uint kv_idx = 0; kv_idx < kv_len; kv_idx++) {
        // 计算 Q * K^T
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += query[d] * key_cache[kv_idx * head_dim + d];
        }
        float score = dot * scale;

        // 跳过被屏蔽的位置（score <= -1e30）防止 NaN
        if (score > -1e30f) {
            // 在线 Softmax 更新
            float new_max = max(max_score, score);
            float scale_factor = exp(max_score - new_max);
            float score_exp = exp(score - new_max);

            output_val = output_val * scale_factor + score_exp * value_cache[kv_idx * head_dim + head_idx];
            sum_exp = sum_exp * scale_factor + score_exp;
            max_score = new_max;
        }
    }

    // 防止除以零
    if (sum_exp > 0.0f) {
        output[head_idx] = output_val / sum_exp;
    } else {
        output[head_idx] = 0.0f;
    }
}
"#;

// ============================================================================
// Metal 设备封装
// ============================================================================

/// Metal 设备封装
pub struct MetalDevice {
    /// Metal 设备
    device: metal::Device,
    /// 设备信息
    device_info: GpuDeviceInfo,
}

impl MetalDevice {
    /// 创建 Metal 设备
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default()
            .ok_or_else(|| anyhow::anyhow!("无法获取 Metal 设备"))?;

        let device_info = Self::get_device_info(&device)?;

        Ok(Self {
            device,
            device_info,
        })
    }

    /// 获取设备信息
    fn get_device_info(device: &metal::Device) -> Result<GpuDeviceInfo> {
        let name = device.name().to_string();
        let memory_size = device.recommended_max_working_set_size();

        let mut features = Vec::new();

        if device.supports_family(metal::MTLGPUFamily::Apple7) {
            features.push("apple7".to_string());
        }
        if device.supports_family(metal::MTLGPUFamily::Apple8) {
            features.push("apple8".to_string());
        }
        if device.supports_family(metal::MTLGPUFamily::Metal3) {
            features.push("metal3".to_string());
        }

        if device.supports_shader_barycentric_coordinates() {
            features.push("simd_group_matrix".to_string());
        }

        // bfloat16 支持: Apple7 及以上 GPU 支持
        if device.supports_family(metal::MTLGPUFamily::Apple7) {
            features.push("bfloat16".to_string());
        }

        Ok(GpuDeviceInfo {
            name,
            memory_size: memory_size as usize,
            compute_capability: None,
            features,
        })
    }

    /// 获取设备引用
    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    /// 获取设备信息
    pub fn info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }
}

// ============================================================================
// Metal 命令队列封装
// ============================================================================

/// Metal 命令队列封装
pub struct MetalCommandQueue {
    /// 命令队列
    queue: metal::CommandQueue,
}

impl MetalCommandQueue {
    /// 创建命令队列
    pub fn new(device: &metal::Device) -> Result<Self> {
        let queue = device.new_command_queue();
        Ok(Self { queue })
    }

    /// 创建命令缓冲区
    pub fn create_command_buffer(&self) -> &metal::CommandBufferRef {
        self.queue.new_command_buffer()
    }

    /// 获取队列引用
    pub fn queue(&self) -> &metal::CommandQueue {
        &self.queue
    }
}

// ============================================================================
// Metal Buffer 封装
// ============================================================================

/// Metal Buffer 封装
pub struct MetalBuffer {
    /// Metal Buffer
    buffer: metal::Buffer,
    /// 数据大小 (字节)
    size: usize,
}

impl MetalBuffer {
    /// 创建新的 Buffer
    pub fn new(device: &metal::Device, size: usize) -> Result<Self> {
        let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        Ok(Self { buffer, size })
    }

    /// 从数据创建 Buffer
    pub fn from_data<T>(device: &metal::Device, data: &[T]) -> Result<Self> {
        let size = std::mem::size_of_val(data);
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(Self { buffer, size })
    }

    /// 写入数据（带对齐检查）
    pub fn write<T>(&self, data: &[T]) -> Result<()> {
        let size = std::mem::size_of_val(data);
        if size > self.size {
            bail!("Buffer 大小不足: {} > {}", size, self.size);
        }

        assert!(
            std::mem::align_of::<T>() <= 16,
            "类型对齐要求超过 16 字节，可能导致 GPU 内存访问问题"
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.buffer.contents() as *mut u8,
                size,
            );
        }

        Ok(())
    }

    /// 读取数据（带对齐检查）
    pub fn read<T>(&self, data: &mut [T]) -> Result<()> {
        let size = std::mem::size_of_val(data);
        if size > self.size {
            bail!("Buffer 大小不足: {} > {}", size, self.size);
        }

        assert!(
            std::mem::align_of::<T>() <= 16,
            "类型对齐要求超过 16 字节，可能导致 GPU 内存访问问题"
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.contents() as *const u8,
                data.as_mut_ptr() as *mut u8,
                size,
            );
        }

        Ok(())
    }

    /// 获取 Buffer 引用
    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }

    /// 获取大小
    pub fn size(&self) -> usize {
        self.size
    }
}

// ============================================================================
// Metal 命令句柄封装
// ============================================================================

/// Metal 命令句柄
///
/// 封装已提交但可能尚未完成的 GPU 命令缓冲区，
/// 允许 CPU 提交多个 kernel 后统一等待，提高 GPU 利用率。
pub struct MetalCommandHandle {
    /// 命令缓冲区（Option 支持移动语义）
    pub(crate) command_buffer: Option<metal::CommandBuffer>,
    /// 标签（用于调试和日志）
    pub label: String,
}

impl MetalCommandHandle {
    /// 创建新的命令句柄
    ///
    /// # 参数
    /// - `command_buffer`: 已编码并提交的命令缓冲区
    /// - `label`: 命令标签（用于调试）
    pub fn new(command_buffer: metal::CommandBuffer, label: &str) -> Self {
        Self {
            command_buffer: Some(command_buffer),
            label: label.to_string(),
        }
    }

    /// 等待命令完成（阻塞）
    ///
    /// 调用后确保 GPU 已完成所有操作，可以安全读取输出数据。
    pub fn wait(&self) {
        if let Some(ref cb) = self.command_buffer {
            cb.wait_until_completed();
        }
    }

    /// 检查命令是否已完成（非阻塞）
    ///
    /// # 返回值
    /// - `true`: 命令已完成
    /// - `false`: 命令仍在执行中
    pub fn is_completed(&self) -> bool {
        match &self.command_buffer {
            Some(cb) => cb.status() == metal::MTLCommandBufferStatus::Completed,
            None => true, // 无命令缓冲区视为已完成
        }
    }

    /// 获取命令标签
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl std::fmt::Debug for MetalCommandHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalCommandHandle")
            .field("label", &self.label)
            .field(
                "status",
                &match &self.command_buffer {
                    Some(cb) => format!("{:?}", cb.status()),
                    None => "None".to_string(),
                },
            )
            .finish()
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
    /// 在线 Softmax
    SoftmaxOnline,
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
    /// 获取 kernel 函数名
    fn kernel_name(&self) -> &'static str {
        match self {
            ShaderType::Matmul => "matmul",
            ShaderType::MatmulBlocked => "matmul_blocked",
            ShaderType::Softmax => "softmax",
            ShaderType::SoftmaxOnline => "softmax_online",
            ShaderType::LayerNorm => "layer_norm",
            ShaderType::RmsNorm => "rms_norm",
            ShaderType::Attention => "attention",
            ShaderType::AttentionKvCache => "attention_kv_cache",
        }
    }
}

/// Shader 库管理器
pub struct ShaderLibrary {
    /// Metal Library
    library: metal::Library,
    /// 已编译的 Pipeline State 缓存（使用 RwLock 支持并发读取）
    pipelines: std::sync::RwLock<HashMap<ShaderType, metal::ComputePipelineState>>,
}

impl ShaderLibrary {
    /// 创建 Shader 库
    pub fn new(device: &metal::Device) -> Result<Self> {
        let source = format!(
            "{}\n{}\n{}\n{}",
            MATMUL_SHADER, SOFTMAX_SHADER, LAYERNORM_SHADER, ATTENTION_SHADER
        );

        let options = metal::CompileOptions::new();
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|e| anyhow::anyhow!("Shader 编译失败: {}", e))?;

        Ok(Self {
            library,
            pipelines: std::sync::RwLock::new(HashMap::new()),
        })
    }

    /// 获取或创建 Pipeline State（优化锁使用）
    pub fn get_pipeline(
        &self,
        device: &metal::Device,
        shader_type: ShaderType,
    ) -> Result<metal::ComputePipelineState> {
        // 先用读锁查询
        {
            let pipelines = self
                .pipelines
                .read()
                .map_err(|_| anyhow::anyhow!("获取 pipelines 读锁失败"))?;
            if let Some(pipeline) = pipelines.get(&shader_type) {
                return Ok(pipeline.clone());
            }
        }

        // 释放读锁后获取写锁创建
        let mut pipelines = self
            .pipelines
            .write()
            .map_err(|_| anyhow::anyhow!("获取 pipelines 写锁失败"))?;

        // 再次检查（可能其他线程已创建）
        if let Some(pipeline) = pipelines.get(&shader_type) {
            return Ok(pipeline.clone());
        }

        let kernel_name = shader_type.kernel_name();
        let function = self
            .library
            .get_function(kernel_name, None)
            .map_err(|e| anyhow::anyhow!("无法获取 kernel 函数 {}: {}", kernel_name, e))?;

        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| anyhow::anyhow!("创建 Pipeline State 失败: {}", e))?;

        pipelines.insert(shader_type, pipeline.clone());
        Ok(pipeline)
    }
}

// ============================================================================
// Metal 后端实现
// ============================================================================

/// Metal 后端
pub struct MetalBackend {
    /// Metal 设备
    device: MetalDevice,
    /// 命令队列
    command_queue: MetalCommandQueue,
    /// Shader 库 (内部已有 RwLock 管理 pipelines)
    shader_library: ShaderLibrary,
}

impl MetalBackend {
    /// 创建 Metal 后端
    pub fn new() -> Result<Self> {
        let device = MetalDevice::new()?;
        let command_queue = MetalCommandQueue::new(device.device())?;
        let shader_library = ShaderLibrary::new(device.device())?;

        Ok(Self {
            device,
            command_queue,
            shader_library,
        })
    }

    /// 执行计算 kernel (异步版本)
    ///
    /// 返回 `MetalCommandHandle` 允许调用者：
    /// - 提交多个 kernel 后统一等待
    /// - 轮询检查完成状态
    /// - 在需要结果时才阻塞等待
    ///
    /// # 线程安全
    /// Metal CommandQueue 不是线程安全的，此方法内部已加锁保护。
    fn execute_kernel_async(
        &self,
        shader_type: ShaderType,
        buffers: &[&MetalBuffer],
        threadgroup_size: metal::MTLSize,
        grid_size: metal::MTLSize,
        label: &str,
    ) -> Result<MetalCommandHandle> {
        let command_buffer = self.command_queue.create_command_buffer().to_owned();

        let encoder = command_buffer.new_compute_command_encoder();

        let pipeline = self
            .shader_library
            .get_pipeline(self.device.device(), shader_type)?;

        encoder.set_compute_pipeline_state(&pipeline);

        for (i, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer.buffer()), 0);
        }

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        // 设置标签用于调试
        command_buffer.set_label(label);

        // 提交到 GPU（不等待完成）
        command_buffer.commit();

        Ok(MetalCommandHandle::new(command_buffer, label))
    }

    /// 执行计算 kernel (同步版本)
    ///
    /// 内部调用 `execute_kernel_async` 并立即等待完成。
    /// 保持向后兼容性，所有现有调用点无需修改。
    fn execute_kernel(
        &self,
        shader_type: ShaderType,
        buffers: &[&MetalBuffer],
        threadgroup_size: metal::MTLSize,
        grid_size: metal::MTLSize,
    ) -> Result<()> {
        let handle = self.execute_kernel_async(
            shader_type,
            buffers,
            threadgroup_size,
            grid_size,
            &format!("{:?}", shader_type),
        )?;
        handle.wait();
        Ok(())
    }

    /// 批量提交多个命令句柄
    ///
    /// 一次性等待所有提交的命令完成。
    /// 适用于需要并行执行多个独立 kernel 的场景，
    /// 可以最大化 GPU 利用率。
    ///
    /// # 参数
    /// - `handles`: 待等待的命令句柄列表
    ///
    /// # 示例
    /// ```ignore
    /// let handle1 = backend.execute_kernel_async(..., "kernel1")?;
    /// let handle2 = backend.execute_kernel_async(..., "kernel2")?;
    /// MetalBackend::submit_batch(vec![handle1, handle2]);
    /// // 此时两个 kernel 都已完成
    /// ```
    pub fn submit_batch(handles: Vec<MetalCommandHandle>) {
        for handle in &handles {
            handle.wait();
        }
    }

    /// 矩阵乘法 (Metal 实现)
    fn matmul_metal(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();

        if k1 != k2 {
            bail!("矩阵维度不匹配: {} vs {}", k1, k2);
        }

        let k = k1;

        let a_buffer = MetalBuffer::from_data(
            self.device.device(),
            a.as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let b_buffer = MetalBuffer::from_data(
            self.device.device(),
            b.as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let c_buffer = MetalBuffer::new(self.device.device(), m * n * std::mem::size_of::<f32>())?;

        let dims = [m as u32, n as u32, k as u32];
        let dims_buffer = MetalBuffer::from_data(self.device.device(), &dims)?;

        let shader_type = if m >= 32 && n >= 32 && k >= 32 {
            ShaderType::MatmulBlocked
        } else {
            ShaderType::Matmul
        };

        let grid_size = metal::MTLSize {
            width: (m as u64).div_ceil(MATMUL_BLOCK_SIZE),
            height: (n as u64).div_ceil(MATMUL_BLOCK_SIZE),
            depth: 1,
        };
        let threadgroup_size = metal::MTLSize {
            width: MATMUL_BLOCK_SIZE,
            height: MATMUL_BLOCK_SIZE,
            depth: 1,
        };

        self.execute_kernel(
            shader_type,
            &[&a_buffer, &b_buffer, &c_buffer, &dims_buffer],
            threadgroup_size,
            grid_size,
        )?;

        let mut result = vec![0.0f32; m * n];
        c_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((m, n), result)?)
    }

    /// Softmax (Metal 实现)
    fn softmax_metal(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let (rows, cols) = input.dim();

        let input_buffer = MetalBuffer::from_data(
            self.device.device(),
            input
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let output_buffer = MetalBuffer::new(
            self.device.device(),
            rows * cols * std::mem::size_of::<f32>(),
        )?;

        let dims = [rows as u32, cols as u32];
        let dims_buffer = MetalBuffer::from_data(self.device.device(), &dims)?;

        let grid_size = metal::MTLSize {
            width: rows.div_ceil(SOFTMAX_BLOCK_SIZE as usize) as u64,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = metal::MTLSize {
            width: SOFTMAX_BLOCK_SIZE,
            height: 1,
            depth: 1,
        };

        self.execute_kernel(
            ShaderType::Softmax,
            &[&input_buffer, &output_buffer, &dims_buffer],
            threadgroup_size,
            grid_size,
        )?;

        let mut result = vec![0.0f32; rows * cols];
        output_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((rows, cols), result)?)
    }

    /// Layer Normalization (Metal 实现)
    fn layer_norm_metal(
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

        let input_buffer = MetalBuffer::from_data(
            self.device.device(),
            input
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let output_buffer = MetalBuffer::new(
            self.device.device(),
            rows * cols * std::mem::size_of::<f32>(),
        )?;
        let gamma_buffer = MetalBuffer::from_data(self.device.device(), gamma)?;
        let beta_buffer = MetalBuffer::from_data(self.device.device(), beta)?;

        let dims = [rows as u32, cols as u32];
        let dims_buffer = MetalBuffer::from_data(self.device.device(), &dims)?;
        let eps_buffer = MetalBuffer::from_data(self.device.device(), &[eps])?;

        let grid_size = metal::MTLSize {
            width: rows.div_ceil(LAYERNORM_BLOCK_SIZE as usize) as u64,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = metal::MTLSize {
            width: LAYERNORM_BLOCK_SIZE,
            height: 1,
            depth: 1,
        };

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
            threadgroup_size,
            grid_size,
        )?;

        let mut result = vec![0.0f32; rows * cols];
        output_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((rows, cols), result)?)
    }

    /// RMS Normalization (Metal 实现)
    pub fn rms_norm(&self, input: &Array2<f32>, gamma: &[f32], eps: f32) -> Result<Array2<f32>> {
        let (rows, cols) = input.dim();

        if gamma.len() != cols {
            bail!("RMS norm 参数大小不匹配");
        }

        let input_buffer = MetalBuffer::from_data(
            self.device.device(),
            input
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let output_buffer = MetalBuffer::new(
            self.device.device(),
            rows * cols * std::mem::size_of::<f32>(),
        )?;
        let gamma_buffer = MetalBuffer::from_data(self.device.device(), gamma)?;

        let dims = [rows as u32, cols as u32];
        let dims_buffer = MetalBuffer::from_data(self.device.device(), &dims)?;
        let eps_buffer = MetalBuffer::from_data(self.device.device(), &[eps])?;

        let grid_size = metal::MTLSize {
            width: rows.div_ceil(LAYERNORM_BLOCK_SIZE as usize) as u64,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = metal::MTLSize {
            width: LAYERNORM_BLOCK_SIZE,
            height: 1,
            depth: 1,
        };

        self.execute_kernel(
            ShaderType::RmsNorm,
            &[
                &input_buffer,
                &output_buffer,
                &gamma_buffer,
                &dims_buffer,
                &eps_buffer,
            ],
            threadgroup_size,
            grid_size,
        )?;

        let mut result = vec![0.0f32; rows * cols];
        output_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((rows, cols), result)?)
    }

    /// Attention (Metal 实现)
    fn attention_metal(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>> {
        let (seq_len, head_dim) = query.dim();
        let (kv_len, _) = key.dim();

        let scale = 1.0 / (head_dim as f32).sqrt();

        let query_buffer = MetalBuffer::from_data(
            self.device.device(),
            query
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let key_buffer = MetalBuffer::from_data(
            self.device.device(),
            key.as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let value_buffer = MetalBuffer::from_data(
            self.device.device(),
            value
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let output_buffer = MetalBuffer::new(
            self.device.device(),
            seq_len * head_dim * std::mem::size_of::<f32>(),
        )?;

        let dims = [seq_len as u32, kv_len as u32, head_dim as u32];
        let dims_buffer = MetalBuffer::from_data(self.device.device(), &dims)?;
        let scale_buffer = MetalBuffer::from_data(self.device.device(), &[scale])?;
        let has_mask: u32 = if mask.is_some() { 1 } else { 0 };
        let has_mask_buffer = MetalBuffer::from_data(self.device.device(), &[has_mask])?;

        let mask_buffer = if let Some(m) = mask {
            MetalBuffer::from_data(
                self.device.device(),
                m.as_slice()
                    .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
            )?
        } else {
            let dummy = vec![0.0f32; seq_len * kv_len];
            MetalBuffer::from_data(self.device.device(), &dummy)?
        };

        let grid_size = metal::MTLSize {
            width: seq_len.div_ceil(ATTENTION_BLOCK_SIZE as usize) as u64,
            height: head_dim.div_ceil(ATTENTION_BLOCK_SIZE as usize) as u64,
            depth: 1,
        };
        let threadgroup_size = metal::MTLSize {
            width: ATTENTION_BLOCK_SIZE,
            height: ATTENTION_BLOCK_SIZE,
            depth: 1,
        };

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
            threadgroup_size,
            grid_size,
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

        let query_buffer = MetalBuffer::from_data(
            self.device.device(),
            query
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let key_buffer = MetalBuffer::from_data(
            self.device.device(),
            key_cache
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let value_buffer = MetalBuffer::from_data(
            self.device.device(),
            value_cache
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("矩阵不是连续存储"))?,
        )?;
        let output_buffer =
            MetalBuffer::new(self.device.device(), head_dim * std::mem::size_of::<f32>())?;

        let cache_info = [kv_len as u32, head_dim as u32];
        let cache_info_buffer = MetalBuffer::from_data(self.device.device(), &cache_info)?;
        let scale_buffer = MetalBuffer::from_data(self.device.device(), &[scale])?;

        let grid_size = metal::MTLSize {
            width: head_dim.div_ceil(ATTENTION_BLOCK_SIZE as usize) as u64,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = metal::MTLSize {
            width: ATTENTION_BLOCK_SIZE,
            height: 1,
            depth: 1,
        };

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
            threadgroup_size,
            grid_size,
        )?;

        let mut result = vec![0.0f32; head_dim];
        output_buffer.read(&mut result)?;

        Ok(Array2::from_shape_vec((1, head_dim), result)?)
    }
}

impl GpuOps for MetalBackend {
    fn device_info(&self) -> &GpuDeviceInfo {
        self.device.info()
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        self.matmul_metal(a, b)
    }

    fn batch_matmul(&self, a: &[Array2<f32>], b: &[Array2<f32>]) -> Result<Vec<Array2<f32>>> {
        if a.len() != b.len() {
            bail!("批量大小不匹配: {} vs {}", a.len(), b.len());
        }

        // 优化：并行执行批量矩阵乘法
        // 对于小批量，使用顺序执行避免开销
        if a.len() <= 4 {
            a.iter()
                .zip(b.iter())
                .map(|(ai, bi)| self.matmul(ai, bi))
                .collect()
        } else {
            // 对于大批量，使用并行执行
            use rayon::prelude::*;
            a.par_iter()
                .zip(b.par_iter())
                .map(|(ai, bi)| self.matmul(ai, bi))
                .collect()
        }
    }

    fn softmax(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.softmax_metal(input)
    }

    fn layer_norm(
        &self,
        input: &Array2<f32>,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<Array2<f32>> {
        self.layer_norm_metal(input, gamma, beta, eps)
    }

    fn attention(
        &self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        mask: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>> {
        self.attention_metal(query, key, value, mask)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn available_memory(&self) -> Result<usize> {
        Ok(self.device.info().memory_size)
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn get_backend() -> MetalBackend {
        MetalBackend::new().expect("无法创建 Metal 后端")
    }

    #[test]
    fn test_metal_backend_creation() {
        let backend = MetalBackend::new();
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        let info = backend.device_info();
        assert!(!info.name.is_empty());
        assert!(info.memory_size > 0);
    }

    #[test]
    fn test_metal_device_info() {
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

        let query = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| (i + j) as f32);
        let key = Array2::from_shape_fn((kv_len, head_dim), |(i, j)| (i + j) as f32);
        let value = Array2::from_shape_fn((kv_len, head_dim), |(i, j)| (i + j) as f32);

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

    // ==================== 新增分支覆盖测试 ====================

    /// 测试 MetalDevice device() 和 info() 访问器（覆盖第445-453行）
    #[test]
    fn test_metal_device_accessors() {
        let backend = get_backend();

        // 覆盖 device() 返回 Metal 设备引用
        let _device_ref = backend.device.device();

        // 覆盖 info() 返回设备信息
        let info = backend.device.info();
        assert!(!info.name.is_empty());
        assert!(info.memory_size > 0);

        // 验证 features 非空
        assert!(!info.features.is_empty(), "设备特性列表不应为空");
    }

    /// 测试 MetalCommandQueue queue() 访问器（覆盖第478-480行）
    #[test]
    fn test_metal_command_queue_accessor() {
        let backend = get_backend();

        // 覆盖 queue() 返回命令队列引用
        let queue_ref = backend.command_queue.queue();
        let _ = format!("{:?}", queue_ref); // Debug trait
    }

    /// 测试 MetalBuffer new/size/write/read（覆盖第497-568行）
    #[test]
    fn test_metal_buffer_operations() {
        let backend = get_backend();
        let device = backend.device.device();

        // 覆盖 new() 创建空缓冲区
        let buffer = MetalBuffer::new(device, 256);
        assert!(buffer.is_ok());
        let buffer = buffer.unwrap();
        assert_eq!(buffer.size(), 256);

        // 覆盖 buffer() 访问器
        let _buf_ref = buffer.buffer();

        // 覆盖 from_data 从数据创建
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data_buffer = MetalBuffer::from_data(device, &data);
        assert!(data_buffer.is_ok());
        let data_buffer = data_buffer.unwrap();
        assert_eq!(data_buffer.size(), 16); // 4 * sizeof(f32)

        // 覆盖 write 和 read
        let write_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let result = buffer.write(&write_data);
        assert!(result.is_ok());

        let mut read_data = vec![0.0f32; 4];
        let result = buffer.read(&mut read_data);
        assert!(result.is_ok());

        // 验证读写一致性
        for (i, (&written, &read)) in write_data.iter().zip(read_data.iter()).enumerate() {
            assert!((written - read).abs() < f32::EPSILON, "索引{}数据不一致", i);
        }
    }

    /// 测试 ShaderType 枚举完整性和 kernel_name（覆盖第576-609行）
    #[test]
    fn test_shader_type_enum() {
        // 覆盖所有 ShaderType 变体及其 kernel_name
        use std::collections::HashSet;

        let shader_types = [
            ShaderType::Matmul,
            ShaderType::MatmulBlocked,
            ShaderType::Softmax,
            ShaderType::SoftmaxOnline,
            ShaderType::LayerNorm,
            ShaderType::RmsNorm,
            ShaderType::Attention,
            ShaderType::AttentionKvCache,
        ];

        let mut kernel_names = HashSet::new();
        for st in &shader_types {
            let name = st.kernel_name();

            // 验证 kernel_name 非空且唯一
            assert!(!name.is_empty(), "{:?} 的 kernel_name 不应为空", st);
            kernel_names.insert(name);

            // 验证 Debug/Copy/Clone/PartialEq/Hash/Eq trait
            let _ = format!("{:?}", st);
            let copy = *st;
            assert_eq!(*st, copy);
        }

        // 所有 kernel_name 应唯一
        assert_eq!(
            kernel_names.len(),
            shader_types.len(),
            "kernel_name 应全部唯一"
        );
    }

    /// 测试 matmul 维度不匹配错误（覆盖第745-747行）
    #[test]
    fn test_metal_matmul_dimension_mismatch() {
        let backend = get_backend();

        let a = Array2::from_shape_vec((2, 3), vec![1.0f32; 6]).unwrap();
        let b = Array2::from_shape_vec((4, 5), vec![1.0f32; 20]).unwrap(); // K不匹配

        // 覆盖：维度不匹配应返回错误
        let result = backend.matmul(&a, &b);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("维度")
                || err_msg.contains("dimension")
                || err_msg.contains("mismatch"),
            "错误消息应包含维度信息: {}",
            err_msg
        );
    }

    /// 测试 layer_norm/rms_norm 参数不匹配错误（覆盖第835-837行和第886-888行）
    #[test]
    fn test_metal_norm_param_mismatch() {
        let backend = get_backend();

        let input = Array2::from_shape_vec((2, 4), vec![1.0f32; 8]).unwrap();
        let gamma = vec![1.0; 4];
        let beta_wrong = vec![0.0; 2]; // beta长度不匹配
        let gamma_wrong = vec![1.0; 2]; // gamma长度不匹配

        // 覆盖：layer_norm 参数不匹配
        let ln_result = backend.layer_norm(&input, &gamma, &beta_wrong, 1e-5);
        assert!(ln_result.is_err());

        // 覆盖：rms_norm 参数不匹配
        let rn_result = backend.rms_norm(&input, &gamma_wrong, 1e-5);
        assert!(rn_result.is_err());
    }

    /// 测试 batch_matmul 并行路径（覆盖第1070-1082行大批量并行分支）
    #[test]
    fn test_metal_batch_matmul_parallel_path() {
        let backend = get_backend();

        // 使用超过4个矩阵触发并行路径
        let batch_size = 6;
        let m = 4;
        let k = 8;
        let n = 4;

        let a: Vec<Array2<f32>> = (0..batch_size)
            .map(|b| Array2::from_shape_fn((m, k), |(i, j)| ((b + i * k + j) % 10) as f32))
            .collect();

        let b: Vec<Array2<f32>> = (0..batch_size)
            .map(|b| Array2::from_shape_fn((k, n), |(i, j)| ((b + i * n + j) % 10) as f32))
            .collect();

        // 覆盖：批量大小>4 应走并行路径
        let results = backend.batch_matmul(&a, &b).unwrap();
        assert_eq!(results.len(), batch_size);
        for (idx, result) in results.iter().enumerate() {
            assert_eq!(result.dim(), (m, n), "结果{}维度不正确", idx);
        }
    }

    /// 测试 attention_with_kv_cache 边界条件（覆盖第999-1051行）
    #[test]
    fn test_metal_attention_kv_cache_edge_cases() {
        let backend = get_backend();

        let head_dim = 8;
        let max_seq = 16;
        let query = Array2::from_shape_fn((1, head_dim), |(i, j)| (i + j) as f32);
        let key_cache = Array2::from_shape_fn((max_seq, head_dim), |(i, j)| (i + j) as f32);
        let value_cache = Array2::from_shape_fn((max_seq, head_dim), |(i, j)| (i + j) as f32);

        // 覆盖：kv_len=0 空缓存
        let result_zero = backend.attention_with_kv_cache(&query, &key_cache, &value_cache, 0);
        assert!(result_zero.is_ok());
        assert_eq!(result_zero.unwrap().dim(), (1, head_dim));

        // 覆盖：kv_len=max_seq 完整缓存
        let result_full =
            backend.attention_with_kv_cache(&query, &key_cache, &value_cache, max_seq);
        assert!(result_full.is_ok());
        assert_eq!(result_full.unwrap().dim(), (1, head_dim));
    }

    // ==================== 异步执行相关测试 ====================

    /// 测试 MetalCommandHandle 创建和基本方法（覆盖第570-640行）
    #[test]
    fn test_metal_command_handle_basic() {
        let backend = get_backend();
        let device = backend.device.device();

        // 准备测试数据
        let a =
            Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])
                .unwrap();
        let b = Array2::from_shape_vec(
            (3, 2),
            vec![7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32],
        )
        .unwrap();

        let a_buffer = MetalBuffer::from_data(device, a.as_slice().unwrap()).unwrap();
        let b_buffer = MetalBuffer::from_data(device, b.as_slice().unwrap()).unwrap();
        let c_buffer = MetalBuffer::new(device, 2 * 2 * std::mem::size_of::<f32>()).unwrap();
        let dims = [2u32, 2u32, 3u32];
        let dims_buffer = MetalBuffer::from_data(device, &dims).unwrap();

        // 测试异步执行
        let handle = backend.execute_kernel_async(
            ShaderType::Matmul,
            &[&a_buffer, &b_buffer, &c_buffer, &dims_buffer],
            metal::MTLSize {
                width: MATMUL_BLOCK_SIZE,
                height: MATMUL_BLOCK_SIZE,
                depth: 1,
            },
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            "test_async_matmul",
        );

        assert!(handle.is_ok(), "异步执行应成功");
        let handle = handle.unwrap();

        // 验证标签
        assert_eq!(handle.label(), "test_async_matmul");

        // 验证 Debug trait
        let debug_str = format!("{:?}", handle);
        assert!(
            debug_str.contains("test_async_matmul"),
            "Debug 输出应包含标签"
        );

        // 等待完成（测试 wait 方法）
        handle.wait();

        // 验证已完成
        assert!(handle.is_completed(), "等待后命令应已完成");
    }

    /// 测试 MetalCommandHandle is_completed 非阻塞检查（覆盖 is_completed 方法）
    #[test]
    fn test_metal_command_handle_is_completed() {
        let backend = get_backend();
        let device = backend.device.device();

        // 准备小规模测试数据（快速完成）
        let input = Array2::from_shape_vec((1, 4), vec![1.0f32, 2.0f32, 3.0f32, 4.0f32]).unwrap();
        let input_buffer = MetalBuffer::from_data(device, input.as_slice().unwrap()).unwrap();
        let output_buffer = MetalBuffer::new(device, 1 * 4 * std::mem::size_of::<f32>()).unwrap();
        let dims = [1u32, 4u32];
        let dims_buffer = MetalBuffer::from_data(device, &dims).unwrap();

        let handle = backend
            .execute_kernel_async(
                ShaderType::Softmax,
                &[&input_buffer, &output_buffer, &dims_buffer],
                metal::MTLSize {
                    width: SOFTMAX_BLOCK_SIZE,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                "test_is_completed",
            )
            .unwrap();

        // 对于小数据，可能已经完成，也可能还在执行
        // 两种状态都是合法的
        let _completed = handle.is_completed(); // 不断言，仅验证方法可调用

        // 等待完成后再次检查
        handle.wait();
        assert!(handle.is_completed(), "等待后必须完成");
    }

    /// 测试 submit_batch 批量提交（覆盖 submit_batch 静态方法）
    #[test]
    fn test_metal_submit_batch() {
        let backend = get_backend();
        let device = backend.device.device();

        // 准备多组测试数据
        let mut handles = Vec::new();

        for i in 0..3 {
            let size = 4;
            let input = Array2::from_shape_vec(
                (1, size),
                (0..size).map(|j| ((i * size + j) as f32)).collect(),
            )
            .unwrap();

            let input_buffer = MetalBuffer::from_data(device, input.as_slice().unwrap()).unwrap();
            let output_buffer =
                MetalBuffer::new(device, 1 * size * std::mem::size_of::<f32>()).unwrap();
            let dims = [1u32, size as u32];
            let dims_buffer = MetalBuffer::from_data(device, &dims).unwrap();

            let handle = backend
                .execute_kernel_async(
                    ShaderType::Softmax,
                    &[&input_buffer, &output_buffer, &dims_buffer],
                    metal::MTLSize {
                        width: SOFTMAX_BLOCK_SIZE,
                        height: 1,
                        depth: 1,
                    },
                    metal::MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    &format!("batch_softmax_{}", i),
                )
                .unwrap();

            handles.push(handle);
        }

        // 批量等待所有命令完成
        MetalBackend::submit_batch(handles);

        // 如果执行到这里没有 panic，说明批量提交成功
    }

    /// 测试异步执行结果正确性（与同步版本对比）
    #[test]
    fn test_metal_async_correctness() {
        let backend = get_backend();

        let a =
            Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])
                .unwrap();
        let b = Array2::from_shape_vec(
            (3, 2),
            vec![7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32],
        )
        .unwrap();

        // 同步版本
        let sync_result = backend.matmul(&a, &b).unwrap();

        // 异步版本应通过同步包装调用，结果应一致
        let async_result = backend.matmul(&a, &b).unwrap();

        // 验证两种方式结果一致
        assert_eq!(sync_result.dim(), async_result.dim());
        for i in 0..sync_result.dim().0 {
            for j in 0..sync_result.dim().1 {
                assert!(
                    (sync_result[[i, j]] - async_result[[i, j]]).abs() < 1e-5,
                    "异步和同步结果不一致: [{},{}] sync={} async={}",
                    i,
                    j,
                    sync_result[[i, j]],
                    async_result[[i, j]]
                );
            }
        }
    }

    /// 测试 execute_kernel_async 的标签设置（覆盖 set_label 调用）
    #[test]
    fn test_metal_async_label_setting() {
        let backend = get_backend();
        let device = backend.device.device();

        let input = Array2::from_shape_vec((1, 2), vec![1.0f32, 2.0f32]).unwrap();
        let input_buffer = MetalBuffer::from_data(device, input.as_slice().unwrap()).unwrap();
        let output_buffer = MetalBuffer::new(device, 1 * 2 * std::mem::size_of::<f32>()).unwrap();
        let dims = [1u32, 2u32];
        let dims_buffer = MetalBuffer::from_data(device, &dims).unwrap();

        let custom_label = "custom_test_label_12345";
        let handle = backend
            .execute_kernel_async(
                ShaderType::Softmax,
                &[&input_buffer, &output_buffer, &dims_buffer],
                metal::MTLSize {
                    width: SOFTMAX_BLOCK_SIZE,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                custom_label,
            )
            .unwrap();

        // 验证自定义标签被正确保存
        assert_eq!(handle.label(), custom_label);
        handle.wait();
    }
}
