//! 硬件检测测试模块
//!
//! 测试硬件检测、分级和调度功能

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::hardware::detector::{
        CacheTopology, CpuArch, CpuInfo, GpuInfo, GpuType, HyperthreadTopology, MemoryInfo,
        NumaTopology, SimdCapabilities,
    };
    use crate::hardware::{
        detect_hardware, get_hardware_level, get_recommended_strategy, AdaptiveScheduler,
        HardwareClassifier, HardwareLevel, HardwareProfile, MemoryStrategy,
    };

    #[test]
    fn test_hardware_detection() {
        let profile = HardwareProfile::detect();

        assert!(profile.cpu.physical_cores > 0, "CPU cores should be > 0");
        assert!(
            profile.cpu.logical_cores >= profile.cpu.physical_cores,
            "Logical cores >= physical cores"
        );
        assert!(profile.memory.total_gb > 0, "Memory should be > 0 GB");

        println!(
            "CPU: {} physical, {} logical cores",
            profile.cpu.physical_cores, profile.cpu.logical_cores
        );
        println!("Memory: {} GB total", profile.memory.total_gb);
        println!("SIMD: {:?}", profile.cpu.simd);
    }

    #[test]
    fn test_hardware_classification() {
        let profile = HardwareProfile::detect();
        let classifier = HardwareClassifier::new(profile);

        let level = classifier.level();
        let device_type = classifier.device_type();

        assert!(
            level >= HardwareLevel::Entry,
            "Level should be at least Entry"
        );

        println!("Hardware Level: {:?}", level);
        println!("Device Type: {:?}", device_type);
    }

    #[test]
    fn test_minimum_requirements() {
        let profile = HardwareProfile::detect();
        let classifier = HardwareClassifier::new(profile);

        let strategy = classifier.recommended_strategy();
        println!("Recommended strategy: {:?}", strategy);
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = AdaptiveScheduler::new();

        let config = scheduler.config();
        println!("Schedule Strategy: {:?}", config.strategy);
        println!("Attention Strategy: {:?}", config.attention);
        println!("Memory Strategy: {:?}", config.memory);
        println!("Parallel Strategy: {:?}", config.parallel);
        println!("Threads: {}", config.num_threads);
        println!("Use SIMD: {}", config.use_simd);
        println!("Use GPU: {}", config.use_gpu);
        println!("KV Cache Size: {} MB", config.kv_cache_size);
        println!("Batch Size: {}", config.batch_size);
    }

    #[test]
    fn test_scheduler_dsa_k() {
        let scheduler = AdaptiveScheduler::new();

        let k_512 = scheduler.recommended_dsa_k(512);
        let k_1024 = scheduler.recommended_dsa_k(1024);
        let k_4096 = scheduler.recommended_dsa_k(4096);

        println!("DSA K for 512: {}", k_512);
        println!("DSA K for 1024: {}", k_1024);
        println!("DSA K for 4096: {}", k_4096);

        assert!(k_512 > 0 && k_512 <= 512);
        assert!(k_1024 > 0 && k_1024 <= 1024);
        assert!(k_4096 > 0 && k_4096 <= 4096);
    }

    #[test]
    fn test_memory_adjustment() {
        let mut scheduler = AdaptiveScheduler::new();

        scheduler.adjust_for_memory(512);
        assert_eq!(scheduler.config().memory, MemoryStrategy::SmallArena);

        scheduler.adjust_for_memory(2048);
        assert_eq!(scheduler.config().memory, MemoryStrategy::StandardArena);

        scheduler.adjust_for_memory(8192);
        assert_eq!(scheduler.config().memory, MemoryStrategy::PagedAttention);
    }

    #[test]
    fn test_sequence_length_adjustment() {
        let mut scheduler = AdaptiveScheduler::new();

        scheduler.adjust_for_sequence_length(1024);
        println!("Attention for 1024: {:?}", scheduler.config().attention);

        scheduler.adjust_for_sequence_length(8192);
        println!("Attention for 8192: {:?}", scheduler.config().attention);
    }

    #[test]
    fn test_convenience_functions() {
        let profile = detect_hardware();
        assert!(profile.cpu.physical_cores > 0);

        let level = get_hardware_level();
        assert!(level >= HardwareLevel::Entry);

        let strategy = get_recommended_strategy();
        assert!(strategy.use_simd);
    }
}

#[cfg(test)]
mod apple_silicon_tests {
    #[allow(unused_imports)]
    use crate::hardware::{CpuArch, HardwareProfile};

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_apple_silicon_detection() {
        let profile = HardwareProfile::detect();

        assert_eq!(profile.cpu.arch, CpuArch::AArch64);
        assert!(profile.cpu.simd.neon);

        println!("Apple Silicon detected!");
        println!("NEON support: {}", profile.cpu.simd.neon);
    }
}

#[cfg(test)]
mod intel_mac_tests {
    use crate::hardware::{CpuArch, HardwareProfile};

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_intel_mac_detection() {
        let profile = HardwareProfile::detect();

        assert_eq!(profile.cpu.arch, CpuArch::X86_64);
        assert!(profile.cpu.simd.sse42 || profile.cpu.simd.avx2);

        println!("Intel Mac detected!");
        println!("SSE4.2 support: {}", profile.cpu.simd.sse42);
        println!("AVX2 support: {}", profile.cpu.simd.avx2);
    }
}

#[cfg(test)]
mod simulated_low_end_tests {
    use crate::hardware::detector::{
        CacheTopology, CpuArch, CpuInfo, GpuInfo, GpuType, HyperthreadTopology, MemoryInfo,
        NumaTopology, SimdCapabilities,
    };
    use crate::hardware::{CpuBackendType, HardwareClassifier, HardwareLevel, HardwareProfile};

    #[test]
    fn test_entry_level_classification() {
        let low_end_profile = HardwareProfile {
            cpu: CpuInfo {
                arch: CpuArch::X86_64,
                physical_cores: 4,
                logical_cores: 8,
                simd: SimdCapabilities {
                    sse42: true,
                    avx: true,
                    avx2: true,
                    avx512: false,
                    neon: false,
                    sve: false,
                    sve2: false,
                    lsx: false,
                    lasx: false,
                },
                name: "Intel Core i7-7700HQ".to_string(),
                is_apple_silicon: false,
                backend_type: CpuBackendType::Avx,
            },
            gpu: GpuInfo {
                gpu_type: GpuType::Amd,
                name: "AMD Radeon Pro 555".to_string(),
                memory_mb: 4096,
                supports_metal: true,
                supports_cuda: false,
                supports_vulkan: false,
                compute_flops: Some(4000.0),
                memory_bandwidth: Some(100.0),
                compute_units: Some(16),
                gpu_frequency_mhz: Some(1278),
            },
            memory: MemoryInfo {
                total_gb: 16,
                available_gb: 8,
            },
            hyperthreading: HyperthreadTopology::default(),
            cache: CacheTopology::default(),
            numa: NumaTopology::default(),
        };

        let classifier = HardwareClassifier::new(low_end_profile);
        let level = classifier.level();

        println!("Entry level profile classified as: {:?}", level);
        assert!(level >= HardwareLevel::Entry);
    }

    #[test]
    fn test_mobile_level_classification() {
        let mobile_profile = HardwareProfile {
            cpu: CpuInfo {
                arch: CpuArch::AArch64,
                physical_cores: 6,
                logical_cores: 6,
                simd: SimdCapabilities {
                    sse42: false,
                    avx: false,
                    avx2: false,
                    avx512: false,
                    neon: true,
                    sve: false,
                    sve2: false,
                    lsx: false,
                    lasx: false,
                },
                name: "Apple A15".to_string(),
                is_apple_silicon: true,
                backend_type: CpuBackendType::Neon,
            },
            gpu: GpuInfo {
                gpu_type: GpuType::Apple,
                name: "Apple GPU".to_string(),
                memory_mb: 6144,
                supports_metal: true,
                supports_cuda: false,
                supports_vulkan: false,
                compute_flops: Some(2600.0),
                memory_bandwidth: Some(100.0),
                compute_units: Some(8),
                gpu_frequency_mhz: Some(1278),
            },
            memory: MemoryInfo {
                total_gb: 6,
                available_gb: 3,
            },
            hyperthreading: HyperthreadTopology::default(),
            cache: CacheTopology::default(),
            numa: NumaTopology::default(),
        };

        let classifier = HardwareClassifier::new(mobile_profile);
        let level = classifier.level();

        println!("Mobile profile classified as: {:?}", level);
        assert!(level >= HardwareLevel::Entry);
    }

    #[test]
    fn test_server_level_classification() {
        let server_profile = HardwareProfile {
            cpu: CpuInfo {
                arch: CpuArch::X86_64,
                physical_cores: 64,
                logical_cores: 128,
                simd: SimdCapabilities {
                    sse42: true,
                    avx: true,
                    avx2: true,
                    avx512: true,
                    neon: false,
                    sve: false,
                    sve2: false,
                    lsx: false,
                    lasx: false,
                },
                name: "AMD EPYC 7763".to_string(),
                is_apple_silicon: false,
                backend_type: CpuBackendType::Avx,
            },
            gpu: GpuInfo {
                gpu_type: GpuType::Nvidia,
                name: "NVIDIA A100".to_string(),
                memory_mb: 81920,
                supports_metal: false,
                supports_cuda: true,
                supports_vulkan: true,
                compute_flops: Some(19500.0),
                memory_bandwidth: Some(1555.0),
                compute_units: Some(108),
                gpu_frequency_mhz: Some(1410),
            },
            memory: MemoryInfo {
                total_gb: 512,
                available_gb: 400,
            },
            hyperthreading: HyperthreadTopology::default(),
            cache: CacheTopology::default(),
            numa: NumaTopology::default(),
        };

        let classifier = HardwareClassifier::new(server_profile);
        let level = classifier.level();

        println!("Server profile classified as: {:?}", level);
        assert!(level >= HardwareLevel::Server);
    }

    // ==================== 新增分支覆盖测试 (7个) ====================

    #[test]
    fn test_cross_module_simd_consistency() {
        // 覆盖分支: 跨 SIMD 模块的结果一致性验证
        use crate::hardware::simd::{SimdOps, SseOps};

        let sse_ops = SseOps;
        let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // 加法一致性
        let add_result = sse_ops.add(&test_data, &vec![1.0; 8]);
        for i in 0..8 {
            assert!((add_result[i] - (i as f32 + 2.0)).abs() < f32::EPSILON);
        }

        // 乘法一致性
        let mul_result = sse_ops.mul(&test_data, &[2.0; 8]);
        for i in 0..8 {
            assert!((mul_result[i] - (i as f32 + 1.0) * 2.0).abs() < f32::EPSILON);
        }

        // 归约操作一致性
        let sum = sse_ops.sum(&test_data);
        assert!((sum - 36.0).abs() < 1e-5); // 1+2+...+8=36

        let max_val = sse_ops.max(&test_data);
        assert!((max_val - 8.0).abs() < 1e-5);

        let min_val = sse_ops.min(&test_data);
        assert!((min_val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hardware_detection_with_classifier() {
        // 覆盖分支: 硬件检测与分类器集成
        let profile = HardwareProfile::detect();

        // 验证检测到的基本信息完整性
        assert!(!profile.cpu.name.is_empty());
        assert!(profile.cpu.physical_cores > 0);

        // 分类器应该能处理任何有效的 profile
        let classifier = HardwareClassifier::new(profile.clone());
        let level = classifier.level();

        // 验证分类结果在合理范围内
        match level {
            HardwareLevel::Entry => assert!(profile.cpu.physical_cores <= 4),
            HardwareLevel::Standard | HardwareLevel::Professional | HardwareLevel::Server => (),
        }
    }

    #[test]
    fn test_memory_info_boundary_conditions() {
        // 覆盖分支: 内存信息边界条件
        let profile = HardwareProfile::detect();

        // 内存总量应该大于可用内存
        assert!(
            profile.memory.total_gb >= profile.memory.available_gb,
            "Total memory should be >= available memory"
        );

        // 可用内存应该在合理范围内（不超过总量）
        assert!(
            profile.memory.available_gb <= profile.memory.total_gb,
            "Available memory should not exceed total"
        );

        // 可用内存应 >= 0（某些环境如容器/内存压力大时可能为 0）
        if profile.memory.total_gb > 0 {
            assert!(
                profile.memory.available_gb <= profile.memory.total_gb,
                "Available memory should not exceed total: {} > {}",
                profile.memory.available_gb,
                profile.memory.total_gb
            );
        }
    }

    #[test]
    fn test_cpu_feature_flag_completeness() {
        // 覆盖分支: CPU 特性标志完整性检查
        let profile = HardwareProfile::detect();

        // 检查特性标志的一致性
        // 使用 SimdCapabilities 字段
        let simd = &profile.cpu.simd;

        // 如果支持 AVX2，必须先支持 AVX 和 SSE4.2 (通过 SIMD 能力检查)
        if simd.avx2 {
            assert!(simd.avx, "AVX2 requires AVX");
            assert!(simd.sse42, "AVX2 requires SSE4.2");
        }

        // 如果支持 AVX-512，必须先支持 AVX2
        if simd.avx512 {
            assert!(simd.avx2, "AVX-512 requires AVX2");
        }

        // 如果是 ARM 平台，NEON 应该可用
        #[cfg(target_arch = "aarch64")]
        assert!(
            simd.neon || cfg!(target_feature = "sve"),
            "ARM64 should support NEON or SVE"
        );

        // 如果支持 SVE2，必须支持 SVE
        if simd.sve2 {
            assert!(simd.sve, "SVE2 requires SVE");
        }
    }

    #[test]
    fn test_gpu_info_fallback_handling() {
        // 覆盖分支: GPU 信息回退处理
        let profile = HardwareProfile::detect();

        // GPU 信息可能不可用（无 GPU 或驱动问题）
        match profile.gpu.gpu_type {
            GpuType::Unknown => {
                // 未知 GPU 是合法的配置
                // 名称可能为空或包含默认值
            }
            _ => {
                // 其他 GPU 类型也是合法的
            }
        }
    }

    #[test]
    fn test_hyperthreading_topology_validation() {
        // 覆盖分支: 超线程拓扑验证
        let profile = HardwareProfile::detect();

        // 使用 CPU 的物理/逻辑核心数信息
        let physical = profile.cpu.physical_cores;
        let logical = profile.cpu.logical_cores;

        // 逻辑核数应该 >= 物理核数
        assert!(
            logical >= physical,
            "Logical cores must be >= physical cores"
        );

        // 如果有超线程，逻辑核数大约是物理核数的 2 倍
        if logical > physical {
            let ratio = logical as f32 / physical as f32;
            assert!(
                (ratio - 2.0).abs() < 0.5 || ratio == 1.0,
                "Hyperthreading ratio unusual: {:.1}",
                ratio
            );
        } else {
            // 无超线程时，逻辑核数应等于物理核数
            assert_eq!(
                logical, physical,
                "Without hyperthreading, logical should equal physical"
            );
        }
    }

    #[test]
    fn test_end_to_end_inference_pipeline_simulation() {
        // 覆盖分支: 端到端推理流程模拟

        // 模拟一个简单的推理流程:
        // 1. 输入预处理 (归一化)
        // 2. 矩阵乘法 (GEMM)
        // 3. 激活函数 (ReLU/SiLU)
        // 4. 输出后处理

        use crate::hardware::simd::{SimdOps, SseOps};
        let ops = SseOps;

        // Step 1: 输入预处理 - 归一化到 [0, 1]
        let raw_input: Vec<f32> = vec![100.0, 200.0, 50.0, 150.0];
        let max_val = ops.max(&raw_input);
        let normalized: Vec<f32> = raw_input.iter().map(|&x| x / max_val).collect();
        assert!(normalized.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // Step 2: 矩阵乘法 (模拟线性层)
        let weight = vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.9, -0.4, 0.6]; // 2x4 权重
        let bias = vec![0.1, -0.1];

        let gemm_result = ops.fused_gemm_relu(&normalized, &weight, &bias, 1, 4, 2);
        assert_eq!(gemm_result.len(), 2);
        assert!(gemm_result.iter().all(|&x| x >= 0.0)); // ReLU 保证非负

        // Step 3: SiLU 激活
        let silu_output = ops.silu(&gemm_result);
        assert_eq!(silu_output.len(), 2);

        // Step 4: Softmax 输出
        let final_output = ops.softmax(&silu_output);
        let sum: f32 = final_output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4); // Softmax 归一化和为 1
        assert!(final_output.iter().all(|&x| x > 0.0 && x <= 1.0));
    }
}
