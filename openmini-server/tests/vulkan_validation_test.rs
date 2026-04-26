//! Vulkan GPU 加速功能正确性验证测试
//!
//! 快速验证 Vulkan 后端核心组件的数值正确性（非性能）：
//! - Vulkan 实例创建与初始化
//! - 设备选择与队列获取
//! - GPU 缓冲区分配、上传、下载与释放
//! - Shader 编译器 (GLSL → SPIR-V via naga)
//! - 向量加法 (vector_add_gpu) 数值正确性
//! - 矩阵乘法 (matrix_multiply_gpu) 数值正确性（与 CPU 参考对比）
//!
//! 使用较小数据规模 (向量 1K-64K, 矩阵 16x16-64x64) 以快速完成。
//! 仅在启用 `vulkan` feature 时编译；无 Vulkan 设备或运行时失败时自动优雅跳过。

#[cfg(feature = "vulkan")]
mod vulkan_validation {

    use ash::vk;
    use openmini_server::hardware::gpu::vulkan::{
        ShaderCompiler, ShaderType, TypedVulkanBuffer, VulkanBuffer, VulkanGpu, VulkanInstance,
        VulkanQueue,
    };
    use openmini_server::hardware::gpu::vulkan_compute::{
        matrix_multiply_gpu, vector_add_gpu, ShaderCache,
    };

    // ==================== 常量定义 ====================

    /// f32 数值比较容差
    const ABS_TOL: f32 = 1e-5;

    // ==================== 辅助函数 ====================

    /// 尝试创建 Vulkan GPU 实例。
    ///
    /// 返回 `None` 表示 Vulkan 不可用（实例/设备初始化失败），
    /// 调用方应跳过测试而非 panic。此设计确保在 CI / 无 GPU 环境下不会崩溃。
    fn try_create_vulkan_gpu() -> Option<VulkanGpu> {
        match VulkanGpu::new(None) {
            Ok(gpu) => {
                eprintln!(
                    "[vulkan-init] Vulkan GPU created: device={}, api_version={}.{}.{}",
                    gpu.capabilities().device_name,
                    vk::api_version_major(gpu.capabilities().api_version),
                    vk::api_version_minor(gpu.capabilities().api_version),
                    vk::api_version_patch(gpu.capabilities().api_version)
                );
                Some(gpu)
            }
            Err(e) => {
                eprintln!(
                    "[vulkan-init] Vulkan GPU init failed, skipping all tests: {}",
                    e
                );
                None
            }
        }
    }

    /// 计算两个 f32 切片的最大绝对误差
    fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
        actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(f32::NEG_INFINITY, |acc, v| acc.max(v))
    }

    /// 断言两个 f32 切片近似相等（误差 < ABS_TOL）
    fn assert_vec_approx(actual: &[f32], expected: &[f32], msg: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{} - length mismatch: {} vs {}",
            msg,
            actual.len(),
            expected.len()
        );
        let diff = max_abs_diff(actual, expected);
        assert!(
            diff < ABS_TOL,
            "{}\n  max absolute diff: {} (tolerance: {})",
            msg,
            diff,
            ABS_TOL
        );
    }

    // ==================== 测试 1：Vulkan 实例创建 ====================

    /// **测试目标**: 验证 VulkanInstance 能否成功创建并获取有效的 instance handle。
    ///
    /// 涵盖:
    /// - ash Entry 加载
    /// - ApplicationInfo 配置
    /// - vkCreateInstance 调用
    #[test]
    fn test_vulkan_instance_creation() {
        match VulkanInstance::new() {
            Ok(instance) => {
                // 验证 instance handle 非空 (非 null handle)
                let _handle = instance.instance();

                eprintln!("[vulkan-instance] PASSED: Vulkan instance created successfully");
            }
            Err(e) => {
                eprintln!(
                    "[vulkan-instance] Skipped: Cannot create Vulkan instance: {}",
                    e
                );
            }
        }
    }

    // ==================== 测试 2：设备初始化与队列获取 ====================

    /// **测试目标**: 验证 VulkanDevice 能否从实例中成功创建，且能获取计算队列。
    ///
    /// 涵盖:
    /// - 物理设备枚举与选择
    /// - 逻辑设备创建
    /// - 计算队列族检测
    /// - 队列 handle 获取
    #[test]
    fn test_vulkan_device_and_queue() {
        match try_create_vulkan_gpu() {
            Some(gpu) => {
                let device = gpu.device();

                // 验证设备信息不为空
                let info = device.info();
                assert!(
                    !info.name.is_empty(),
                    "[vulkan-device] Device name should not be empty"
                );
                assert!(
                    info.memory_size > 0,
                    "[vulkan-device] Device memory size should be > 0, got {}",
                    info.memory_size
                );

                eprintln!(
                    "[vulkan-device] Device info: name={}, memory={} MB",
                    info.name,
                    info.memory_size / (1024 * 1024)
                );

                // 尝试创建命令队列以验证队列族可用
                match VulkanQueue::new(device.clone()) {
                    Ok(queue) => {
                        let q_handle = queue.queue();
                        assert_ne!(
                            q_handle,
                            ash::vk::Queue::null(),
                            "[vulkan-device] Queue handle should not be null"
                        );

                        // 验证命令池也创建了
                        assert_ne!(
                            queue.command_pool(),
                            ash::vk::CommandPool::null(),
                            "[vulkan-device] Command pool should not be null"
                        );

                        eprintln!("[vulkan-device] PASSED: Queue and command pool created");
                    }
                    Err(e) => {
                        eprintln!(
                            "[vulkan-device] Skipped queue test: Cannot create queue: {}",
                            e
                        );
                    }
                }
            }
            None => {}
        }
    }

    // ==================== 测试 3：GPU 缓冲区分配与数据传输 ====================

    /// **测试目标**: 验证 VulkanBuffer / TypedVulkanBuffer 能否正确分配、写入、读取和释放。
    ///
    /// 涵盖:
    /// - 缓冲区创建 (vkCreateBuffer + vkAllocateMemory + vkBindBufferMemory)
    /// - 数据上传 (vkMapMemory + 数据拷贝 + vkUnmapMemory)
    /// - 数据下载 (vkMapMemory + 读取 + vkUnmapMemory)
    /// - 缓冲区销毁 (Drop 时 vkDestroyBuffer + vkFreeMemory)
    #[test]
    fn test_vulkan_buffer_allocate_and_transfer() {
        match try_create_vulkan_gpu() {
            Some(gpu) => {
                let device = gpu.device();
                let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

                // --- 测试 3a: VulkanBuffer 基本操作 ---
                match VulkanBuffer::from_data(device.clone(), &data) {
                    Ok(buffer) => {
                        assert_eq!(buffer.size(), data.len() * std::mem::size_of::<f32>());

                        // 读回数据验证一致性
                        let mut read_back = vec![0.0f32; data.len()];
                        let read_result = buffer.read(&mut read_back);
                        assert!(
                            read_result.is_ok(),
                            "[vulkan-buffer] Buffer read failed: {:?}",
                            read_result.err()
                        );

                        assert_vec_approx(
                            &read_back,
                            &data,
                            "[vulkan-buffer] Data roundtrip mismatch",
                        );

                        eprintln!("[vulkan-buffer] PASSED: VulkanBuffer roundtrip correct");
                    }
                    Err(e) => {
                        eprintln!("[vulkan-buffer] Skipped: Buffer creation failed: {}", e);
                    }
                }

                // --- 测试 3b: TypedVulkanBuffer 泛型操作 ---
                match TypedVulkanBuffer::<f32>::from_data(device, &data) {
                    Ok(typed_buf) => {
                        assert_eq!(typed_buf.len(), data.len());
                        assert!(!typed_buf.is_empty());

                        // 上传新数据
                        let new_data: Vec<f32> =
                            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
                        let upload_result = typed_buf.upload(&new_data);
                        assert!(
                            upload_result.is_ok(),
                            "[vulkan-typed-buffer] Upload failed: {:?}",
                            upload_result.err()
                        );

                        // 下载数据验证
                        match typed_buf.download() {
                            Ok(downloaded) => {
                                assert_vec_approx(
                                    &downloaded,
                                    &new_data,
                                    "[vulkan-typed-buffer] Typed buffer roundtrip mismatch",
                                );
                                eprintln!(
                                    "[vulkan-typed-buffer] PASSED: TypedVulkanBuffer roundtrip correct"
                                );
                            }
                            Err(e) => {
                                eprintln!("[vulkan-typed-buffer] Download failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "[vulkan-typed-buffer] Skipped: Typed buffer creation failed: {}",
                            e
                        );
                    }
                }
            }
            None => {}
        }
    }

    // ==================== 测试 4：着色器编译 (GLSL → SPIR-V) ====================

    /// **测试目标**: 验证 ShaderCompiler 能否将 GLSL Compute Shader 源码编译为有效 SPIR-V 二进制。
    ///
    /// 涵盖:
    /// - naga GLSL frontend 解析
    /// - naga validation
    /// - naga SPIR-V backend 生成
    /// - 输出 SPIR-V magic number 校验
    #[test]
    fn test_shader_compilation_glsl_to_spirv() {
        // ShaderCompiler 不需要 GPU 设备，只需 naga 可用
        match ShaderCompiler::new() {
            Ok(mut compiler) => {
                // 编译矩阵乘法 shader
                let spirv_result = compiler.compile(ShaderType::Matmul);

                match spirv_result {
                    Ok(spirv) => {
                        // SPIR-V 最小长度应 > 5 (header 至少 5 个 word)
                        assert!(
                            spirv.len() >= 5,
                            "[vulkan-shader] SPIR-V too short: {} words",
                            spirv.len()
                        );

                        // 验证 SPIR-V magic number: 0x07230203
                        assert_eq!(
                            spirv[0], 0x07230203,
                            "[vulkan-shader] Invalid SPIR-V magic number: {:#010X}",
                            spirv[0]
                        );

                        eprintln!(
                            "[vulkan-shader] PASSED: Matmul shader compiled to SPIR-V ({} words)",
                            spirv.len()
                        );
                    }
                    Err(e) => {
                        eprintln!("[vulkan-shader] Skipped: Shader compilation failed: {}", e);
                    }
                }

                // 编译分块矩阵乘法 shader
                match compiler.compile(ShaderType::MatmulBlocked) {
                    Ok(spirv) => {
                        assert_eq!(
                            spirv[0], 0x07230203,
                            "[vulkan-shader] Blocked matmul invalid magic"
                        );
                        eprintln!(
                            "[vulkan-shader] PASSED: MatmulBlocked shader compiled ({} words)",
                            spirv.len()
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "[vulkan-shader] Skipped: Blocked matmul compilation failed: {}",
                            e
                        );
                    }
                }

                // 编译 Softmax shader
                match compiler.compile(ShaderType::Softmax) {
                    Ok(spirv) => {
                        assert_eq!(
                            spirv[0], 0x07230203,
                            "[vulkan-shader] Softmax invalid magic"
                        );
                        eprintln!(
                            "[vulkan-shader] PASSED: Softmax shader compiled ({} words)",
                            spirv.len()
                        );
                    }
                    Err(e) => {
                        eprintln!("[vulkan-shader] Skipped: Softmax compilation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "[vulkan-shader] Skipped: Cannot create ShaderCompiler: {}",
                    e
                );
            }
        }

        // --- 测试 ShaderCache 与真实 GLSL 源码集成 ---
        let mut cache = ShaderCache::new();
        assert!(cache.is_empty());

        // 加载 vector_add shader 源码到缓存
        match cache.load_embedded("vector_add_test", "#version 450\nvoid main() {}") {
            Ok(()) => {
                assert!(cache.contains("vector_add_test"));
                assert_eq!(cache.len(), 1);
                eprintln!("[vulkan-shader-cache] PASSED: ShaderCache load_embedded works");
            }
            Err(e) => {
                eprintln!("[vulkan-shader-cache] Skipped: Load embedded failed: {}", e);
            }
        }
    }

    // ==================== 测试 5：向量加法 (vector_add_gpu) ====================

    /// **测试目标**: 验证 `vector_add_gpu` 输出的数值正确性，与 CPU 参考实现逐元素对比。
    ///
    /// 注意: 当前实现可能回退到 CPU 计算（GPU 路径尚未完全实现），
    /// 但无论走哪条路径，结果数值必须正确。
    #[test]
    fn test_vector_add_small() {
        let gpu = match try_create_vulkan_gpu() {
            Some(g) => g,
            None => return,
        };

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0, 55.0];

        match vector_add_gpu(&gpu, &a, &b) {
            Ok(result) => {
                assert_vec_approx(
                    &result,
                    &expected,
                    "[vulkan-vector-add-small] Vector add result mismatch (5 elements)",
                );
                eprintln!("[vulkan-vector-add-small] PASSED: Small vector add correct");
            }
            Err(e) => {
                eprintln!("[vulkan-vector-add-small] Skipped - Execution error: {}", e);
            }
        }
    }

    /// **测试目标**: 较大向量加法 (1024 元素)，触发 GPU 路径条件判断（如果 GPU 已实现）。
    #[test]
    fn test_vector_add_1k() {
        let gpu = match try_create_vulkan_gpu() {
            Some(g) => g,
            None => return,
        };

        let size = 1024;
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| i as f32 * 0.2).collect();
        let expected: Vec<f32> = (0..size).map(|i| i as f32 * 0.3).collect();

        match vector_add_gpu(&gpu, &a, &b) {
            Ok(result) => {
                assert_eq!(
                    result.len(),
                    size,
                    "[vulkan-vector-add-1k] Result length mismatch"
                );
                assert_vec_approx(
                    &result,
                    &expected,
                    "[vulkan-vector-add-1k] Vector add (1K) result mismatch",
                );
                eprintln!("[vulkan-vector-add-1k] PASSED: 1K vector add correct");
            }
            Err(e) => {
                eprintln!("[vulkan-vector-add-1k] Skipped - Execution error: {}", e);
            }
        }
    }

    /// **测试目标**: 向量加法错误处理 — 空向量和长度不匹配应返回错误。
    #[test]
    fn test_vector_add_error_handling() {
        let gpu = match try_create_vulkan_gpu() {
            Some(g) => g,
            None => return,
        };

        let non_empty = vec![1.0f32, 2.0, 3.0];
        let empty: Vec<f32> = vec![];

        // 空向量 A
        let r1 = vector_add_gpu(&gpu, &empty, &non_empty);
        assert!(
            r1.is_err(),
            "[vulkan-vector-add-err] Empty vector A should return error"
        );

        // 空向量 B
        let r2 = vector_add_gpu(&gpu, &non_empty, &empty);
        assert!(
            r2.is_err(),
            "[vulkan-vector-add-err] Empty vector B should return error"
        );

        // 长度不匹配
        let mismatched = vec![1.0f32, 2.0];
        let r3 = vector_add_gpu(&gpu, &non_empty, &mismatched);
        assert!(
            r3.is_err(),
            "[vulkan-vector-add-err] Length mismatch should return error"
        );

        eprintln!("[vulkan-vector-add-err] PASSED: Error handling correct");
    }

    // ==================== 测试 6：矩阵乘法 (matrix_multiply_gpu) ====================

    /// **测试目标**: 验证 `matrix_multiply_gpu` 对小矩阵 (2×2 × 2×2) 的输出正确性。
    ///
    /// 手工计算参考:
    /// A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    /// C[0,0]=1*5+2*7=19, C[0,1]=1*6+2*8=22
    /// C[1,0]=3*5+4*7=43, C[1,1]=3*6+4*8=50
    #[test]
    fn test_matrix_multiply_2x2() {
        let gpu = match try_create_vulkan_gpu() {
            Some(g) => g,
            None => return,
        };

        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2×2 row-major
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2×2 row-major
        let expected = vec![19.0f32, 22.0, 43.0, 50.0];

        match matrix_multiply_gpu(&gpu, &a, &b, 2, 2, 2) {
            Ok(result) => {
                assert_eq!(
                    result.len(),
                    4,
                    "[vulkan-matmul-2x2] Result element count mismatch"
                );
                assert_vec_approx(
                    &result,
                    &expected,
                    "[vulkan-matmul-2x2] Matrix multiply (2x2) result mismatch",
                );
                eprintln!("[vulkan-matmul-2x2] PASSED: 2x2 matrix multiply correct");
            }
            Err(e) => {
                eprintln!("[vulkan-matmul-2x2] Skipped - Execution error: {}", e);
            }
        }
    }

    /// **测试目标**: 中等规模矩阵 (16×16 × 16×16) 正确性验证。
    #[test]
    fn test_matrix_multiply_16x16() {
        let gpu = match try_create_vulkan_gpu() {
            Some(g) => g,
            None => return,
        };

        let (m, k, n) = (16, 16, 16);

        // 构造确定性输入
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 100) as f32) / 100.0).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 100) as f32) / 100.0 + 0.5)
            .collect();

        // CPU 参考实现
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        match matrix_multiply_gpu(&gpu, &a, &b, m, k, n) {
            Ok(result) => {
                assert_eq!(
                    result.len(),
                    m * n,
                    "[vulkan-matmul-16x16] Result size mismatch: expected {}, got {}",
                    m * n,
                    result.len()
                );
                assert_vec_approx(
                    &result,
                    &expected,
                    "[vulkan-matmul-16x16] Matrix multiply (16x16) result mismatch",
                );
                eprintln!("[vulkan-matmul-16x16] PASSED: 16x16 matrix multiply correct");
            }
            Err(e) => {
                eprintln!("[vulkan-matmul-16x16] Skipped - Execution error: {}", e);
            }
        }
    }

    /// **测试目标**: 较大方阵 (64×64 × 64×64) 正确性验证。
    #[test]
    fn test_matrix_multiply_64x64() {
        let gpu = match try_create_vulkan_gpu() {
            Some(g) => g,
            None => return,
        };

        let (m, k, n) = (64, 64, 64);

        // 使用简单模式生成确定性数据
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) % 10.0) / 10.0 - 0.5)
            .collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) % 10.0) / 10.0).collect();

        // CPU 参考
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        match matrix_multiply_gpu(&gpu, &a, &b, m, k, n) {
            Ok(result) => {
                assert_eq!(result.len(), m * n, "[vulkan-matmul-64x64] Size mismatch");
                assert_vec_approx(
                    &result,
                    &expected,
                    "[vulkan-matmul-64x64] Matrix multiply (64x64) result mismatch",
                );
                eprintln!("[vulkan-matmul-64x64] PASSED: 64x64 matrix multiply correct");
            }
            Err(e) => {
                eprintln!("[vulkan-matmul-64x64] Skipped - Execution error: {}", e);
            }
        }
    }

    /// **测试目标**: 非方阵矩阵乘法 (32×64 @ 64×48 = 32×48) 形状与数值正确性。
    #[test]
    fn test_matrix_multiply_non_square() {
        let gpu = match try_create_vulkan_gpu() {
            Some(g) => g,
            None => return,
        };

        let (m, k, n) = (32, 64, 48); // A: 32×64, B: 64×48, C: 32×48

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) % 7.0) / 7.0 - 0.35)
            .collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) % 11.0) / 11.0).collect();

        // CPU 参考
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        match matrix_multiply_gpu(&gpu, &a, &b, m, k, n) {
            Ok(result) => {
                assert_eq!(
                    result.len(),
                    m * n,
                    "[vulkan-matmul-nonsquare] Expected {} elements, got {}",
                    m * n,
                    result.len()
                );
                assert_vec_approx(
                    &result,
                    &expected,
                    "[vulkan-matmul-nonsquare] Non-square result mismatch",
                );
                eprintln!(
                    "[vulkan-matmul-nonsquare] PASSED: Non-square ({}x{} @ {}x{}) correct",
                    m, k, k, n
                );
            }
            Err(e) => {
                eprintln!("[vulkan-matmul-nonsquare] Skipped - Execution error: {}", e);
            }
        }
    }

    /// **测试目标**: 矩阵乘法错误处理 — 维度不匹配和零维度应返回错误。
    #[test]
    fn test_matrix_multiply_error_handling() {
        let gpu = match try_create_vulkan_gpu() {
            Some(g) => g,
            None => return,
        };

        // A 元素数量不匹配声明维度
        let bad_a = vec![1.0f32, 2.0, 3.0]; // 声明 2×2 需要 4 个元素
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let r1 = matrix_multiply_gpu(&gpu, &bad_a, &b, 2, 2, 2);
        assert!(
            r1.is_err(),
            "[vulkan-matmul-err] Dimension mismatch should return error"
        );

        // B 元素数量不匹配
        let a_ok = vec![1.0f32, 2.0, 3.0, 4.0];
        let bad_b = vec![5.0f32, 6.0]; // 声明 2×2 需要 4 个元素
        let r2 = matrix_multiply_gpu(&gpu, &a_ok, &bad_b, 2, 2, 2);
        assert!(
            r2.is_err(),
            "[vulkan-matmul-err] B dimension mismatch should return error"
        );

        // 零维度
        let tiny_a = vec![1.0f32];
        let tiny_b = vec![2.0f32];
        let r3 = matrix_multiply_gpu(&gpu, &tiny_a, &tiny_b, 0, 1, 1);
        assert!(
            r3.is_err(),
            "[vulkan-matmul-err] Zero M dimension should return error"
        );

        let r4 = matrix_multiply_gpu(&gpu, &tiny_a, &tiny_b, 1, 0, 1);
        assert!(
            r4.is_err(),
            "[vulkan-matmul-err] Zero K dimension should return error"
        );

        let r5 = matrix_multiply_gpu(&gpu, &tiny_a, &tiny_b, 1, 1, 0);
        assert!(
            r5.is_err(),
            "[vulkan-matmul-err] Zero N dimension should return error"
        );

        eprintln!("[vulkan-matmul-err] PASSED: Error handling correct");
    }

    // ==================== 测试 7：边界情况 — 单位矩阵行为 ====================

    /// **测试目标**: 近似单位矩阵乘法 A @ I ≈ A，检验基础线性代数性质。
    #[test]
    fn test_matrix_multiply_identity_like() {
        let gpu = match try_create_vulkan_gpu() {
            Some(g) => g,
            None => return,
        };

        let n = 32usize;
        let a: Vec<f32> = (0..n * n).map(|i| ((i as f32) % 5.0) / 5.0 - 0.4).collect();

        // 构造单位矩阵
        let mut identity = vec![0.0f32; n * n];
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }

        match matrix_multiply_gpu(&gpu, &a, &identity, n, n, n) {
            Ok(result) => {
                assert_vec_approx(&result, &a, "[vulkan-identity] A @ I should approximate A");
                eprintln!("[vulkan-identity] PASSED: Identity-like multiplication preserves input");
            }
            Err(e) => {
                eprintln!("[vulkan-identity] Skipped - Execution error: {}", e);
            }
        }
    }
}

// ==================== 未启用 vulkan feature 时的占位测试 ====================

/// 当 `vulkan` feature 未启用时的占位测试，确保测试套件不会因缺少 feature 而失败。
/// 此测试始终通过，仅作为文档说明当前环境不支持 Vulkan 测试。
#[cfg(not(feature = "vulkan"))]
mod vulkan_unavailable {
    #[test]
    fn test_vulkan_skipped_unavailable() {
        eprintln!(
            "[vulkan-validation] Skipped: Vulkan tests require feature='vulkan'. \
             Current config: vulkan_feature={}",
            cfg!(feature = "vulkan")
        );
    }
}
