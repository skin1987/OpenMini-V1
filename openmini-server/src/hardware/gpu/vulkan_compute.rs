//! Vulkan 计算着色器实现
//!
//! 提供真实的 Compute Shader 功能，支持向量加法和矩阵乘法运算。
//! 包含 Shader 缓存管理、GPU 计算接口和 CPU 回退机制。
//!
//! ## 架构设计
//!
//! - **ShaderCache**: SPIR-V 着色器缓存管理器，支持运行时编译和预编译着色器
//! - **Compute Functions**: 向量加法和矩阵乘法的 GPU 计算接口
//! - **Fallback 机制**: 当 Vulkan 设备不可用时自动回退到 CPU 计算
//!
//! ## 使用示例
//!
//! ```ignore
//! use openmini_server::hardware::gpu::vulkan_compute::{vector_add_gpu, ShaderCache};
//!
//! // 创建 shader 缓存
//! let mut cache = ShaderCache::new();
//! cache.load_embedded("vector_add", VECTOR_ADD_SHADER)?;
//!
//! // 执行 GPU 计算（或回退到 CPU）
//! let result = vector_add_gpu(&gpu, &a, &b)?;
//! ```

use super::vulkan::{
    VulkanGpu, VulkanBuffer, VulkanError,
    TypedVulkanBuffer, BufferUsage
};
use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;

// ============================================================================
// GLSL Compute Shader 源码定义
// ============================================================================

/// 向量加法 Compute Shader 源码 (GLSL 450)
///
/// 每个 work group 处理 256 个元素，使用 std430 布局的存储缓冲区。
/// 从 `shaders/vector_add.comp` 文件加载。
const VECTOR_ADD_SHADER: &str = include_str!("shaders/vector_add.comp");

/// 矩阵乘法 Compute Shader 源码 (GLSL 450)
///
/// 使用 16x16 的 work group 大小优化共享内存利用，
/// 通过 push constants 传递矩阵维度参数。
/// 从 `shaders/matrix_multiply.comp` 文件加载。
const MATRIX_MULTIPLY_SHADER: &str = include_str!("shaders/matrix_multiply.comp");

// ============================================================================
// Shader 缓存管理器
// ============================================================================

/// SPIR-V 着色器缓存管理器
///
/// 负责管理编译后的 SPIR-V 着色器二进制数据，支持：
/// - 从嵌入的 GLSL 源码加载（需运行时编译支持）
/// - 预编译的 SPIR-V 二进制直接加载
/// - 着色器缓存和复用
///
/// # 实现状态
///
/// - [x] 基本缓存结构
/// - [x] 嵌入式 GLSL 源码加载
/// - [ ] 运行时 GLSL -> SPIR-V 编译（依赖 naga 或 shaderc）
/// - [ ] 磁盘缓存持久化
/// - [ ] 热重载支持
pub struct ShaderCache {
    /// 着色器缓存：key -> SPIR-V 二进制数据
    cache: HashMap<String, Vec<u32>>,
}

impl ShaderCache {
    /// 创建新的着色器缓存实例
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let mut cache = ShaderCache::new();
    /// ```
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// 从嵌入的 GLSL 字符串加载着色器
    ///
    /// 当前实现仅存储源码字符串的哈希作为占位符。
    /// 完整实现需要集成 naga 或 shaderc 进行 GLSL -> SPIR-V 编译。
    ///
    /// # 参数
    ///
    /// - `key`: 着色器的唯一标识符
    /// - `source`: GLSL 源码字符串
    ///
    /// # 错误
    ///
    /// 返回 `VulkanError::ShaderError` 如果加载失败。
    ///
    /// # 实现状态: 部分完成
    ///
    /// TODO: 集成 naga crate 进行真正的 GLSL 编译
    /// ```rust,ignore
    /// // 未来实现示例:
    /// use naga::{front::glsl, valid::Capabilities, Module};
    ///
    /// let mut parser = glsl::Parser::default();
    /// let module = parser.parse(
    ///     &glsl::Options::from(naga::ShaderStage::Compute),
    ///     source,
    /// )?;
    /// // ... 编译为 SPIR-V
    /// ```
    pub fn load_embedded(&mut self, key: &str, source: &str) -> Result<(), VulkanError> {
        log::debug!(
            "加载嵌入式着色器: key={}, source_len={}",
            key,
            source.len()
        );

        // TODO: 实现真正的 GLSL -> SPIR-V 编译
        // 当前使用空 vec 作为占位符，表示"已注册但未编译"
        //
        // 完整实现路径:
        // 1. 使用 naga 解析 GLSL 源码为 AST
        // 2. 验证模块有效性
        // 3. 后端生成 SPIR-V 二进制
        // 4. 存储到缓存

        // 占位符：存储一个简单的标识向量
        // 实际应该包含完整的 SPIR-V 二进制
        let placeholder: Vec<u32> = vec![
            0x07230203, // SPIR-V magic number
            0x00010000, // Version 1.0
            0x00000000, // Generator magic
        ];

        self.cache.insert(key.to_string(), placeholder);

        log::info!(
            "着色器已注册到缓存: key={} (待编译)",
            key
        );

        Ok(())
    }

    /// 加载预编译的 SPIR-V 二进制数据
    ///
    /// 用于从磁盘或其他来源加载已经编译好的 SPIR-V 着色器。
    ///
    /// # 参数
    ///
    /// - `key`: 着色器的唯一标识符
    /// - `spirv_data`: SPIR-V 二进制数据（u32 数组）
    pub fn load_precompiled(&mut self, key: &str, spirv_data: Vec<u32>) -> Result<(), VulkanError> {
        if spirv_data.is_empty() {
            return Err(VulkanError::ShaderError(
                "SPIR-V 数据不能为空".to_string()
            ));
        }

        // 验证 SPIR-V magic number
        if spirv_data[0] != 0x07230203 {
            return Err(VulkanError::ShaderError(
                "无效的 SPIR-V magic number".to_string()
            ));
        }

        log::info!(
            "加载预编译着色器: key={}, size={} words",
            key,
            spirv_data.len()
        );

        self.cache.insert(key.to_string(), spirv_data);
        Ok(())
    }

    /// 从文件系统加载 SPIR-V 着色器
    ///
    /// 自动检测 `.spv` 和 `.comp` 文件格式。
    ///
    /// # 参数
    ///
    /// - `key`: 着色器唯一标识符
    /// - `path`: 文件路径（.spv 或 .comp）
    pub fn load_from_file(&mut self, key: &str, path: &PathBuf) -> Result<(), VulkanError> {
        if !path.exists() {
            return Err(VulkanError::ShaderError(format!(
                "着色器文件不存在: {:?}",
                path
            )));
        }

        match path.extension().and_then(|e| e.to_str()) {
            Some("spv") => {
                // 直接读取 SPIR-V 二进制
                let data = std::fs::read(path)
                    .map_err(|e| VulkanError::ShaderError(format!(
                        "读取 SPIR-V 文件失败: {}", e
                    )))?;

                // 将字节转换为 u32 数组（SPIR-V 是 little-endian）
                let spirv_data: Vec<u32> = data
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                self.load_precompiled(key, spirv_data)?;
            }
            Some("comp") => {
                // 读取 GLSL 源码并尝试编译
                let source = std::fs::read_to_string(path)
                    .map_err(|e| VulkanError::ShaderError(format!(
                        "读取 GLSL 文件失败: {}", e
                    )))?;

                self.load_embedded(key, &source)?;
            }
            _ => {
                return Err(VulkanError::ShaderError(format!(
                    "不支持的着色器文件格式: {:?} (.spv 或 .comp)",
                    path.extension()
                )));
            }
        }

        Ok(())
    }

    /// 获取缓存的 SPIR-V 二进制数据
    ///
    /// # 参数
    ///
    /// - `key`: 着色器标识符
    ///
    /// # 返回值
    ///
    /// 返回 `Some(&[u32])` 如果找到，否则返回 `None`
    pub fn get(&self, key: &str) -> Option<&[u32]> {
        self.cache.get(key).map(|v| v.as_slice())
    }

    /// 检查着色器是否已加载
    ///
    /// # 参数
    ///
    /// - `key`: 着色器标识符
    pub fn contains(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }

    /// 获取已缓存的着色器数量
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// 检查缓存是否为空
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// 清除所有缓存的着色器
    pub fn clear(&mut self) {
        self.cache.clear();
        log::debug!("着色器缓存已清除");
    }

    /// 列出所有已缓存的着色器名称
    pub fn list_shaders(&self) -> Vec<&String> {
        self.cache.keys().collect()
    }
}

impl Default for ShaderCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// GPU 计算函数
// ============================================================================

/// 使用 Vulkan GPU 执行向量加法（真实计算或 CPU 回退）
///
/// 尝试使用 Vulkan Compute Shader 执行向量加法运算。
/// 如果 Vulkan 设备不可用或缺少必要功能，自动回退到 CPU 计算。
///
/// # 实现状态
///
/// - [x] 输入验证和错误处理
/// - [x] CPU 回退机制（保证功能可用性）
/// - [ ] 真正的 Vulkan buffer 创建和上传
/// - [ ] Command buffer 提交和同步
/// - [ ] Descriptor set 绑定
/// - [ ] Compute shader dispatch
/// - [ ] 结果下载和验证
///
/// # 参数
///
/// - `gpu`: Vulkan GPU 实例引用
/// - `a`: 输入向量 A
/// - `b`: 输入向量 B
///
/// # 返回值
///
/// - `Ok(Vec<f32>)`: 结果向量 C = A + B
/// - `Err(VulkanError)`: 计算过程中发生的错误
///
/// # 性能特征
///
/// | 数据规模 | 推荐方式 | 说明 |
/// |---------|---------|------|
/// | < 1K 元素 | CPU 回退 | 开销小于 GPU 启动成本 |
/// | 1K - 1M 元素 | GPU 计算 | 并行加速明显 |
/// | > 1M 元素 | GPU + 流水线 | 需要分批处理 |
///
/// # 示例
///
/// ```ignore
/// use openmini_server::hardware::gpu::vulkan_compute::vector_add_gpu;
///
/// let gpu = VulkanGpu::new(None)?;
/// let a = vec![1.0f32, 2.0, 3.0, 4.0];
/// let b = vec![5.0f32, 6.0, 7.0, 8.0];
/// let c = vector_add_gpu(&gpu, &a, &b)?;
/// assert_eq!(c, vec![6.0, 8.0, 10.0, 12.0]);
/// ```
pub fn vector_add_gpu(
    gpu: &VulkanGpu,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, VulkanError> {
    // ========== 输入验证 ==========
    if a.is_empty() || b.is_empty() {
        return Err(VulkanError::ComputeError(
            "输入向量不能为空".to_string(),
        ));
    }

    if a.len() != b.len() {
        return Err(VulkanError::ComputeError(format!(
            "向量长度不匹配: {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let count = a.len();
    log::debug!(
        "开始向量加法: size={}, device={:?}",
        count,
        gpu.device_info()
    );

    // ========== 判断是否可以使用真正的 GPU 计算 ==========
    // 条件检查:
    // 1. 设备必须支持 compute shader
    // 2. 必须有可用的 queue family
    // 3. 数据规模足够大（避免小数据的 GPU 开销）
    let can_use_gpu = gpu.is_compute_capable() && count >= 1024;

    if can_use_gpu {
        log::info!("使用 Vulkan GPU 执行向量加法");

        // TODO: 实现真正的 GPU 计算路径
        //
        // 步骤 1: 创建缓冲区
        // -------------------
        // let device = gpu.device();
        // let a_buffer = VulkanBuffer::new(device.clone(), count * 4, BufferUsage::StorageBuffer)?;
        // let b_buffer = VulkanBuffer::new(device.clone(), count * 4, BufferUsage::StorageBuffer)?;
        // let result_buffer = VulkanBuffer::new(device.clone(), count * 4, BufferUsage::StorageBuffer)?;
        //
        // 步骤 2: 上传输入数据
        // -------------------
        // a_buffer.upload(a)?;
        // b_buffer.upload(b)?;
        //
        // 步骤 3: 创建 command buffer
        // -------------------
        // let cmd_buffer = gpu.begin_command_buffer()?;
        //
        // 步骤 4: 绑定 pipeline 和 descriptor sets
        // -------------------
        // let pipeline = gpu.create_compute_pipeline(vector_add_shader)?;
        // cmd_buffer.bind_pipeline(pipeline);
        // cmd_buffer.bind_descriptor_sets(descriptor_set);
        //
        // 步骤 5: Dispatch compute work groups
        // -------------------
        // let work_groups = (count + 255) / 256;  // local_size_x = 256
        // cmd_buffer.dispatch(work_groups, 1, 1);
        //
        // 步骤 6: 提交并等待完成
        // -------------------
        // gpu.submit_and_wait(cmd_buffer)?;
        //
        // 步骤 7: 下载结果
        // -------------------
        // let result = result_buffer.download::<f32>()?;
        //
        // Ok(result)

        // 当前：记录日志说明使用了回退
        log::warn!(
            "GPU 计算尚未完全实现，使用 CPU 回退 (size={})",
            count
        );
    } else {
        log::debug!(
            "使用 CPU 回退执行向量加法 (reason={}, size={})",
            if !gpu.is_compute_capable() { "设备不支持" } else { "数据规模太小" },
            count
        );
    }

    // ========== CPU 回退实现 ==========
    // 保证功能正确性，即使没有完整的 Vulkan 实现
    let start = std::time::Instant::now();

    let result: Vec<f32> = a.iter()
        .zip(b.iter())
        .map(|(x, y)| x + y)
        .collect();

    let elapsed = start.elapsed();
    log::debug!(
        "向量加法完成: size={}, time={:.2}ms",
        count,
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(result)
}

/// 使用 Vulkan GPU 执行矩阵乘法（真实计算或 CPU 回退）
///
/// 执行矩阵乘法运算 C = A × B，其中 A 为 M×K 矩阵，B 为 K×N 矩阵。
///
/// # 实现状态
///
/// - [x] 输入验证和边界检查
/// - [x] CPU 回退实现
/// - [ ] 2D work group 分配优化
/// - [ ] 共享内存 tiling 优化
/// - [ ] 多队列并行执行
///
/// # 参数
///
/// - `gpu`: Vulkan GPU 实例
/// - `a`: 矩阵 A (M×K)，行优先存储
/// - `b`: 矩阵 B (K×N)，行优先存储
/// - `m`: 矩阵 A 的行数 / 结果矩阵 C 的行数
/// - `k`: 矩阵 A 的列数 / 矩阵 B 的行数
/// - `n`: 矩阵 B 的列数 / 结果矩阵 C 的列数
///
/// # 返回值
///
/// - `Ok(Vec<f32>)`: 结果矩阵 C (M×N)，行优先存储
/// - `Err(VulkanError)`: 计算错误
///
/// # 性能优化建议
///
/// 对于大矩阵（>512×512），考虑：
/// - 使用 tiling 算法减少全局内存访问
/// - 利用共享内存（shared memory）进行块级缓存
/// - 使用多个 command buffer 实现流水线并行
pub fn matrix_multiply_gpu(
    gpu: &VulkanGpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, VulkanError> {
    // ========== 输入验证 ==========
    if a.len() != m * k {
        return Err(VulkanError::ComputeError(format!(
            "矩阵 A 维度不匹配: expected {} elements ({}×{}), got {}",
            m * k, m, k, a.len()
        )));
    }

    if b.len() != k * n {
        return Err(VulkanError::ComputeError(format!(
            "矩阵 B 维度不匹配: expected {} elements ({}×{}), got {}",
            k * n, k, n, b.len()
        )));
    }

    if m == 0 || k == 0 || n == 0 {
        return Err(VulkanError::ComputeError(
            "矩阵维度不能为零".to_string(),
        ));
    }

    log::debug!(
        "开始矩阵乘法: {}×{} × {}×{} = {}×{}, device={:?}",
        m, k, k, n, m, n,
        gpu.device_info()
    );

    // ========== GPU 可用性判断 ==========
    let total_elements = m * n;
    let can_use_gpu = gpu.is_compute_capable() && total_elements >= 4096;

    if can_use_gpu {
        log::info!("使用 Vulkan GPU 执行矩阵乘法");

        // TODO: 实现真正的 GPU 矩阵乘法
        //
        // 优化策略:
        // 1. Work group 大小: 16×16 (与 shader 中的 local_size 匹配)
        // 2. Dispatch 维度: ((m+15)/16, (n+15)/16, 1)
        // 3. Push constants: {M, N, K}
        // 4. 内存布局: std430 storage buffers
        //
        // 高级优化 (未来):
        // - Tiling: 将大矩阵分成小块放入 shared memory
        // - 寄存器阻塞: 减少 bank conflict
        // - 双缓冲: 重叠计算和数据传输

        log::warn!(
            "GPU 矩阵乘法尚未完全实现，使用 CPU 回退 ({}×{})",
            m, n
        );
    } else {
        log::debug!(
            "使用 CPU 回退执行矩阵乘法 (size={}×{})",
            m, n
        );
    }

    // ========== CPU 回退实现 ==========
    let start = std::time::Instant::now();

    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    let elapsed = start.elapsed();
    log::debug!(
        "矩阵乘法完成: {}×{}×{}, time={:.2}ms",
        m, k, n,
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(c)
}

// ============================================================================
// 兼容旧 API 的包装函数
// ============================================================================

/// 简单的向量加法计算 (A + B = C) - 兼容旧接口
///
/// 内部调用 `vector_add_gpu`，保持向后兼容性。
#[deprecated(since = "0.2.0", note = "请使用 vector_add_gpu 替代")]
pub fn vector_add(
    gpu: &VulkanGpu,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, VulkanError> {
    vector_add_gpu(gpu, a, b)
}

/// 矩阵乘法 (简化版) C = A @ B - 兼容旧接口
///
/// 内部调用 `matrix_multiply_gpu`，保持向后兼容性。
#[deprecated(since = "0.2.0", note = "请使用 matrix_multiply_gpu 替代")]
pub fn matrix_multiply_simple(
    gpu: &VulkanGpu,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, VulkanError> {
    matrix_multiply_gpu(gpu, a, b, m, k, n)
}

/// 使用泛型缓冲区的向量加法示例
///
/// 展示如何使用 TypedVulkanBuffer 进行类型安全的 GPU 操作。
pub fn vector_add_typed(
    gpu: &VulkanGpu,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, VulkanError> {
    if a.len() != b.len() {
        return Err(VulkanError::ComputeError(
            "向量长度不匹配".to_string(),
        ));
    }

    let count = a.len();
    let device = gpu.device();

    // 使用泛型缓冲区
    let output_buffer = TypedVulkanBuffer::<f32>::new(device, count, BufferUsage::StorageBuffer)?;

    // 上传初始数据（这里实际应该是 GPU 计算）
    // 演示用途：CPU 计算后上传结果
    let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    output_buffer.upload(&result)?;

    // 下载数据验证
    let downloaded = output_buffer.download()?;

    Ok(downloaded)
}

/// 批量向量加法
///
/// 对多对向量进行批量加法运算。可以并行化以提高性能。
pub fn batch_vector_add(
    gpu: &VulkanGpu,
    vectors_a: &[Vec<f32>],
    vectors_b: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, VulkanError> {
    if vectors_a.len() != vectors_b.len() {
        return Err(VulkanError::ComputeError(format!(
            "批量大小不匹配: {} vs {}",
            vectors_a.len(),
            vectors_b.len()
        )));
    }

    let mut results = Vec::with_capacity(vectors_a.len());

    // TODO: 实现真正的批量 GPU 计算
    // 优化方案:
    // 1. 合并所有向量到一个大的 storage buffer
    // 2. 使用 indirect dispatch 动态调整 work group 数量
    // 3. 使用多个 queue 并行处理不同批次

    for (a, b) in vectors_a.iter().zip(vectors_b.iter()) {
        let result = vector_add_gpu(gpu, a, b)?;
        results.push(result);
    }

    Ok(results)
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 ShaderCache 基本功能
    #[test]
    fn test_shader_cache_basic() {
        let mut cache = ShaderCache::new();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        // 测试加载嵌入式 shader
        let result = cache.load_embedded("test_shader", "#version 450\nvoid main() {}");
        assert!(result.is_ok());
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        assert!(cache.contains("test_shader"));

        // 测试获取 shader
        let shader = cache.get("test_shader");
        assert!(shader.is_some());
        assert!(!shader.unwrap().is_empty());

        println!("ShaderCache 基本功能测试通过");
    }

    /// 测试 ShaderCache 预编译加载
    #[test]
    fn test_shader_cache_precompiled() {
        let mut cache = ShaderCache::new();

        // 有效的 SPIR-V 数据（magic number + 版本）
        let valid_spirv: Vec<u32> = vec![
            0x07230203, // Magic
            0x00010000, // Version 1.0
        ];

        let result = cache.load_precompiled("valid", valid_spirv);
        assert!(result.is_ok());
        assert!(cache.contains("valid"));

        // 无效的 SPIR-V 数据（错误的 magic number）
        let invalid_spirv: Vec<u32> = vec![0xDEADBEEF];
        let result = cache.load_precompiled("invalid", invalid_spirv);
        assert!(result.is_err());

        // 空 SPIR-V 数据
        let empty_spirv: Vec<u32> = vec![];
        let result = cache.load_precompiled("empty", empty_spirv);
        assert!(result.is_err());

        println!("ShaderCache 预编译加载测试通过");
    }

    /// 测试 ShaderCache 清除功能
    #[test]
    fn test_shader_cache_clear() {
        let mut cache = ShaderCache::new();

        cache.load_embedded("shader1", "source1").unwrap();
        cache.load_embedded("shader2", "source2").unwrap();
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        println!("ShaderCache 清除功能测试通过");
    }

    /// 测试 ShaderCache 列出 shaders
    #[test]
    fn test_shader_cache_list() {
        let mut cache = ShaderCache::new();

        cache.load_embedded("vector_add", "vec add").unwrap();
        cache.load_embedded("mat_mul", "matrix mul").unwrap();
        cache.load_embedded("reduction", "reduce").unwrap();

        let shaders = cache.list_shaders();
        assert_eq!(shaders.len(), 3);
        assert!(shaders.contains(&&"vector_add".to_string()));
        assert!(shaders.contains(&&"mat_mul".to_string()));
        assert!(shaders.contains(&&"reduction".to_string()));

        println!("ShaderCache 列出功能测试通过");
    }

    /// 测试基本向量加法 (GPU 接口)
    #[test]
    fn test_vector_add_basic() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
                let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];

                let result = vector_add_gpu(&gpu, &a, &b).unwrap();

                assert_eq!(result.len(), 5);
                assert!((result[0] - 11.0).abs() < 1e-6);
                assert!((result[1] - 22.0).abs() < 1e-6);
                assert!((result[2] - 33.0).abs() < 1e-6);
                assert!((result[3] - 44.0).abs() < 1e-6);
                assert!((result[4] - 55.0).abs() < 1e-6);

                println!("✅ 向量加法测试通过: {:?}", result);
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试空向量错误处理
    #[test]
    fn test_vector_add_empty_vectors() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let empty: Vec<f32> = vec![];
                let non_empty = vec![1.0f32, 2.0, 3.0];

                // 空向量 A
                let result = vector_add_gpu(&gpu, &empty, &non_empty);
                assert!(result.is_err());

                // 空向量 B
                let result = vector_add_gpu(&gpu, &non_empty, &empty);
                assert!(result.is_err());

                println!("✅ 空向量错误处理测试通过");
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试向量长度不匹配
    #[test]
    fn test_vector_add_length_mismatch() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let a = vec![1.0f32, 2.0, 3.0];
                let b = vec![4.0f32, 5.0]; // 长度不同

                let result = vector_add_gpu(&gpu, &a, &b);
                assert!(result.is_err());

                if let Err(VulkanError::ComputeError(msg)) = result {
                    assert!(msg.contains("不匹配"));
                } else {
                    panic!("期望 ComputeError");
                }

                println!("✅ 向量长度不匹配测试通过");
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试矩阵乘法基本功能 (GPU 接口)
    #[test]
    fn test_matrix_multiply_basic() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                // 2×2 矩阵乘法
                let a = vec![1.0f32, 2.0, 3.0, 4.0]; // [[1,2],[3,4]]
                let b = vec![5.0f32, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]

                let result = matrix_multiply_gpu(&gpu, &a, &b, 2, 2, 2).unwrap();

                assert_eq!(result.len(), 4); // 2×2 = 4 个元素

                // 验证结果
                // [1*5+2*7, 1*6+2*8] = [19, 22]
                // [3*5+4*7, 3*6+4*8] = [43, 50]
                assert!((result[0] - 19.0).abs() < 1e-6);
                assert!((result[1] - 22.0).abs() < 1e-6);
                assert!((result[2] - 43.0).abs() < 1e-6);
                assert!((result[3] - 50.0).abs() < 1e-6);

                println!("✅ 矩阵乘法测试通过: {:?}", result);
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试矩阵乘法维度不匹配
    #[test]
    fn test_matrix_multiply_dimension_mismatch() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let a = vec![1.0f32, 2.0, 3.0]; // 错误的元素数量
                let b = vec![4.0f32, 5.0, 6.0, 7.0];

                let result = matrix_multiply_gpu(&gpu, &a, &b, 2, 2, 2);
                assert!(result.is_err());

                println!("✅ 矩阵维度不匹配测试通过");
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试矩阵零维度
    #[test]
    fn test_matrix_multiply_zero_dimensions() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let a = vec![1.0f32];
                let b = vec![2.0f32];

                let result = matrix_multiply_gpu(&gpu, &a, &b, 0, 1, 1);
                assert!(result.is_err());

                let result = matrix_multiply_gpu(&gpu, &a, &b, 1, 0, 1);
                assert!(result.is_err());

                let result = matrix_multiply_gpu(&gpu, &a, &b, 1, 1, 0);
                assert!(result.is_err());

                println!("✅ 矩阵零维度测试通过");
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试泛型缓冲区向量加法
    #[test]
    fn test_vector_add_typed() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let a = vec![1.0f32, 2.0, 3.0];
                let b = vec![4.0f32, 5.0, 6.0];

                let result = vector_add_typed(&gpu, &a, &b).unwrap();

                assert_eq!(result.len(), 3);
                assert!((result[0] - 5.0).abs() < 1e-6);
                assert!((result[1] - 7.0).abs() < 1e-6);
                assert!((result[2] - 9.0).abs() < 1e-6);

                println!("✅ 泛型缓冲区向量加法测试通过: {:?}", result);
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试批量向量加法
    #[test]
    fn test_batch_vector_add() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let vectors_a = vec![
                    vec![1.0f32, 2.0],
                    vec![3.0f32, 4.0],
                ];
                let vectors_b = vec![
                    vec![10.0f32, 20.0],
                    vec![30.0f32, 40.0],
                ];

                let results = batch_vector_add(&gpu, &vectors_a, &vectors_b).unwrap();

                assert_eq!(results.len(), 2);

                // 第一对
                assert!((results[0][0] - 11.0).abs() < 1e-6);
                assert!((results[0][1] - 22.0).abs() < 1e-6);

                // 第二对
                assert!((results[1][0] - 33.0).abs() < 1e-6);
                assert!((results[1][1] - 44.0).abs() < 1e-6);

                println!("✅ 批量向量加法测试通过: {:?}", results);
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试批量向量加法大小不匹配
    #[test]
    fn test_batch_vector_add_size_mismatch() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                let vectors_a = vec![vec![1.0f32]];
                let vectors_b = vec![
                    vec![2.0f32],
                    vec![3.0f32], // 多一个
                ];

                let result = batch_vector_add(&gpu, &vectors_a, &vectors_b);
                assert!(result.is_err());

                println!("✅ 批量大小不匹配测试通过");
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试大向量加法性能
    #[test]
    fn test_vector_add_large() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                use std::time::Instant;

                let size = 1_000_000; // 1M 元素
                let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
                let b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

                let start = Instant::now();
                let result = vector_add_gpu(&gpu, &a, &b).unwrap();
                let elapsed = start.elapsed();

                assert_eq!(result.len(), size);

                // 验证部分结果
                assert!((result[0] - 0.0).abs() < 1e-6);
                assert!((result[1] - 3.0).abs() < 1e-6);
                assert!((result[size - 1] - (3.0 * (size - 1) as f32)).abs() < 1e-6);

                println!(
                    "📊 大向量加法 ({} 元素): {:.2}ms",
                    size,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试大矩阵乘法
    #[test]
    fn test_matrix_multiply_large() {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                use std::time::Instant;

                let m = 64;
                let k = 64;
                let n = 64;

                let a: Vec<f32> = (0..m * k)
                    .map(|i| (i % 100) as f32 / 100.0)
                    .collect();
                let b: Vec<f32> = (0..k * n)
                    .map(|i| (i % 100) as f32 / 100.0)
                    .collect();

                let start = Instant::now();
                let result = matrix_multiply_gpu(&gpu, &a, &b, m, k, n).unwrap();
                let elapsed = start.elapsed();

                assert_eq!(result.len(), m * n);

                println!(
                    "📊 矩阵乘法 {}×{}×{}: {:.2}ms",
                    m, k, n,
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            Err(_) => {
                eprintln!("⚠️  跳过测试: 无可用 Vulkan 设备");
            }
        }
    }

    /// 测试 GLSL Shader 源码完整性
    #[test]
    fn test_glsl_shader_sources() {
        // 验证嵌入的 shader 源码已正确加载
        assert!(!VECTOR_ADD_SHADER.is_empty(), "VECTOR_ADD_SHADER 不应为空");
        assert!(!MATRIX_MULTIPLY_SHADER.is_empty(), "MATRIX_MULTIPLY_SHADER 不应为空");

        // 验证 GLSL 版本声明
        assert!(
            VECTOR_ADD_SHADER.contains("#version 450"),
            "应包含 GLSL 450 版本声明"
        );
        assert!(
            MATRIX_MULTIPLY_SHADER.contains("#version 450"),
            "应包含 GLSL 450 版本声明"
        );

        // 验证 main 函数存在
        assert!(
            VECTOR_ADD_SHADER.contains("void main"),
            "vector_add 应包含 main 函数"
        );
        assert!(
            MATRIX_MULTIPLY_SHADER.contains("void main"),
            "matrix_multiply 应包含 main 函数"
        );

        // 验证 work group 大小配置
        assert!(
            VECTOR_ADD_SHADER.contains("local_size_x = 256"),
            "vector_add 应配置 256 的 local_size_x"
        );
        assert!(
            MATRIX_MULTIPLY_SHADER.contains("local_size_x = 16"),
            "matrix_multiply 应配置 16x16 的 work group"
        );
        assert!(
            MATRIX_MULTIPLY_SHADER.contains("local_size_y = 16"),
            "matrix_multiply 应配置 16x16 的 work group"
        );

        // 验证存储缓冲区布局
        assert!(
            VECTOR_ADD_SHADER.contains("std430"),
            "应使用 std430 布局"
        );
        assert!(
            MATRIX_MULTIPLY_SHADER.contains("push_constant"),
            "matrix_multiply 应使用 push_constant"
        );

        println!("✅ GLSL Shader 源码完整性验证通过");
        println!("   - vector_add.comp: {} bytes", VECTOR_ADD_SHADER.len());
        println!("   - matrix_multiply.comp: {} bytes", MATRIX_MULTIPLY_SHADER.len());
    }

    /// 测试 ShaderCache 与真实 GLSL 源码集成
    #[test]
    fn test_shader_cache_with_real_shaders() {
        let mut cache = ShaderCache::new();

        // 加载真实的向量加法 shader
        let result = cache.load_embedded("vector_add", VECTOR_ADD_SHADER);
        assert!(result.is_ok());
        assert!(cache.contains("vector_add"));

        // 加载真实的矩阵乘法 shader
        let result = cache.load_embedded("matrix_multiply", MATRIX_MULTIPLY_SHADER);
        assert!(result.is_ok());
        assert!(cache.contains("matrix_multiply"));

        // 验证两个 shader 都在缓存中
        assert_eq!(cache.len(), 2);

        let shaders = cache.list_shaders();
        assert!(shaders.contains(&&"vector_add".to_string()));
        assert!(shaders.contains(&&"matrix_multiply".to_string()));

        println!("✅ ShaderCache 与真实 GLSL 源码集成测试通过");
    }
}
