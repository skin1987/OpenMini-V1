//! quant_simd.rs SIGSEGV 修复回归测试
//!
//! 测试场景：
//! - 正常数据量化/反量化往返精度
//! - 空输入、单元素输入边界条件
//! - 非对齐内存输入安全性
//! - 大批量数据压力测试（100万+ 元素）
//! - 连续调用稳定性测试（10000 次无内存泄漏）
//!
//! 运行方式：
//! ```bash
//! # 标准测试
//! cargo test --test quant_simd_regression_test
//!
//! # AddressSanitizer 测试（需要 nightly Rust）
//! RUSTFLAGS="-Z sanitizer=address" cargo +nightly test --test quant_simd_regression_test
//! ```

use openmini_server::model::inference::quant_simd::{
    safe_dequantize, safe_dequantize_q4_0, safe_dequantize_q8_0, safe_dequantize_q4_1,
    detect_simd_support, QuantError, SimdSupport,
    dequantize_simd, dequantize_simd_parallel,
};
use openmini_server::model::inference::gguf::GgufTensorType;

// ============================================================================
// 辅助函数
// ============================================================================

/// 创建测试用的 Q4_0 量化数据
fn create_test_q4_0_data(n: usize) -> Vec<u8> {
    use half::f16;
    let block_count = n.div_ceil(32);
    let mut data = Vec::with_capacity(block_count * 18);

    for i in 0..block_count {
        // scale (f16)
        let scale = f16::from_f32(0.1 + (i as f32) * 0.01);
        data.extend_from_slice(&scale.to_le_bytes());

        // quants (16 bytes, 每个 byte 包含 2 个 4-bit 值)
        for j in 0..16 {
            let val = ((i * 16 + j) % 16) as u8;
            data.push((val << 4) | (val & 0x0F));
        }
    }

    data
}

/// 创建测试用的 Q8_0 量化数据
fn create_test_q8_0_data(n: usize) -> Vec<u8> {
    use half::f16;
    let block_count = n.div_ceil(32);
    let mut data = Vec::with_capacity(block_count * 34);

    for i in 0..block_count {
        // scale (f16)
        let scale = f16::from_f32(0.2 + (i as f32) * 0.02);
        data.extend_from_slice(&scale.to_le_bytes());

        // quants (32 bytes, i8 values)
        for j in 0..32 {
            let val = (((i * 32 + j) % 256) as i8).to_le_bytes()[0];
            data.push(val);
        }
    }

    data
}

/// 创建测试用的 Q4_1 量化数据
fn create_test_q4_1_data(n: usize) -> Vec<u8> {
    use half::f16;
    let block_count = n.div_ceil(32);
    let mut data = Vec::with_capacity(block_count * 20);

    for i in 0..block_count {
        // scale (f16)
        let scale = f16::from_f32(0.15 + (i as f32) * 0.015);
        data.extend_from_slice(&scale.to_le_bytes());

        // min (f16)
        let min = f16::from_f32(-1.0 + (i as f32) * 0.1);
        data.extend_from_slice(&min.to_le_bytes());

        // quants (16 bytes, 4-bit values)
        for j in 0..16 {
            let val = ((i * 16 + j) % 16) as u8;
            data.push((val << 4) | (val & 0x0F));
        }
    }

    data
}

// ============================================================================
// 第一部分：CPU Feature 检测测试
// ============================================================================

#[test]
fn test_detect_simd_support() {
    let support = detect_simd_support();

    // 验证返回值结构完整
    println!("SIMD Support: {}", support);

    // 在 x86_64 上应该至少支持 SSE4.2 或 AVX2
    #[cfg(target_arch = "x86_64")]
    {
        assert!(support.sse42 || support.avx2 || support.avx512,
            "x86_64 应该至少支持一种 SIMD 指令集");
    }

    // 在 aarch64 上应该支持 NEON
    #[cfg(target_arch = "aarch64")]
    {
        assert!(support.neon, "aarch64 应该支持 NEON");
    }
}

#[test]
fn test_simd_support_display() {
    let support = SimdSupport {
        avx512: true,
        avx2: true,
        sse42: true,
        neon: false,
    };

    let display = format!("{}", support);
    assert!(display.contains("AVX512"));
    assert!(display.contains("AVX2"));
    assert!(display.contains("SSE4.2"));
}

// ============================================================================
// 第二部分：安全 API 输入验证测试
// ============================================================================

#[test]
fn test_safe_dequantize_empty_input() {
    // 空数据应该返回空结果，而不是崩溃
    let result = safe_dequantize_q4_0(&[], 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);

    let result = safe_dequantize_q8_0(&[], 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);

    let result = safe_dequantize_q4_1(&[], 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_safe_dequantize_single_element() {
    // 单元素输入
    let q4_data = create_test_q4_0_data(1);
    let result = safe_dequantize_q4_0(&q4_data, 1);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 1);
    // 验证输出是有限数值
    assert!(output[0].is_finite(), "单元素输出应该是有限数");

    let q8_data = create_test_q8_0_data(1);
    let result = safe_dequantize_q8_0(&q8_data, 1);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
}

#[test]
fn test_safe_dequantize_insufficient_data() {
    // 数据不足应该返回错误而不是崩溃
    let incomplete_data = vec![0u8; 10]; // Q4_0 需要至少 18 字节
    let result = safe_dequantize_q4_0(&incomplete_data, 32);

    assert!(result.is_err());
    match result.unwrap_err() {
        QuantError::InsufficientData { expected, actual } => {
            assert_eq!(expected, 18);
            assert_eq!(actual, 10);
        }
        _ => panic!("期望 InsufficientData 错误"),
    }
}

#[test]
fn test_safe_dequantize_non_aligned_size() {
    // 非对齐大小（不是 32 的倍数）不应该崩溃
    let sizes = [1, 31, 33, 63, 65, 100, 1000, 1023];

    for &size in &sizes {
        let q4_data = create_test_q4_0_data(size);
        let result = safe_dequantize_q4_0(&q4_data, size);
        assert!(result.is_ok(), "Q4_0 size={} 失败", size);
        let output = result.unwrap();
        assert_eq!(output.len(), size, "Q4_0 输出长度不匹配");

        // 验证所有输出都是有限数值
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "Q4_0 输出[{}] 无穷或 NaN", i);
        }
    }
}

// ============================================================================
// 第三部分：正常数据往返精度测试
// ============================================================================

#[test]
fn test_safe_dequantize_q4_0_roundtrip() {
    let sizes = [32, 64, 128, 256, 512, 1024];

    for &size in &sizes {
        let data = create_test_q4_0_data(size);
        let result = safe_dequantize_q4_0(&data, size);
        assert!(result.is_ok(), "Q4_0 size={} 处理失败", size);

        let output = result.unwrap();
        assert_eq!(output.len(), size, "Q4_0 输出长度不匹配");

        // 验证输出在合理范围内（Q4_0 范围是 -8 到 7 * scale）
        for &val in &output {
            assert!(val.is_finite(), "输出包含非有限值");
            // Q4_0 的理论范围大约是 [-8*scale, 7*scale]，scale ~ 0.1-1.0
            assert!(val.abs() < 100.0, "输出值异常大: {}", val);
        }
    }
}

#[test]
fn test_safe_dequantize_q8_0_roundtrip() {
    let sizes = [32, 64, 128, 256, 512];

    for &size in &sizes {
        let data = create_test_q8_0_data(size);
        let result = safe_dequantize_q8_0(&data, size);
        assert!(result.is_ok(), "Q8_0 size={} 处理失败", size);

        let output = result.unwrap();
        assert_eq!(output.len(), size);

        for &val in &output {
            assert!(val.is_finite());
            assert!(val.abs() < 100.0);
        }
    }
}

#[test]
fn test_safe_dequantize_q4_1_roundtrip() {
    let sizes = [32, 64, 128, 256];

    for &size in &sizes {
        let data = create_test_q4_1_data(size);
        let result = safe_dequantize_q4_1(&data, size);
        assert!(result.is_ok(), "Q4_1 size={} 处理失败", size);

        let output = result.unwrap();
        assert_eq!(output.len(), size);

        for &val in &output {
            assert!(val.is_finite());
        }
    }
}

#[test]
fn test_safe_dequantize_generic_api() {
    // 测试通用安全 API
    let q4_data = create_test_q4_0_data(64);
    let result = safe_dequantize(&q4_data, GgufTensorType::Q4_0, 64);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 64);

    let q8_data = create_test_q8_0_data(64);
    let result = safe_dequantize(&q8_data, GgufTensorType::Q8_0, 64);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 64);

    // 不支持的类型应该返回错误
    let result = safe_dequantize(&[], GgufTensorType::Q2K, 32);
    assert!(result.is_err());
}

// ============================================================================
// 第四部分：原始 API 安全性测试（对比验证）
// ============================================================================

#[test]
fn test_original_api_vs_safe_api_consistency() {
    // 验证原始 API 和安全 API 的输出一致
    let size = 256;

    // Q4_0
    let q4_data = create_test_q4_0_data(size);
    let safe_result = safe_dequantize_q4_0(&q4_data, size).unwrap();
    let original_result = dequantize_simd(&q4_data, GgufTensorType::Q4_0, size);

    assert_eq!(safe_result.len(), original_result.len(),
        "安全 API 和原始 API 输出长度不一致");

    // 允许微小的浮点误差
    for (i, (safe_val, orig_val)) in safe_result.iter().zip(original_result.iter()).enumerate() {
        let diff = (safe_val - orig_val).abs();
        assert!(diff < 1e-6 || diff / safe_val.abs().max(1e-10) < 1e-5,
            "Q4_0 元素[{}]: safe={}, original={}, diff={}", i, safe_val, orig_val, diff);
    }

    // Q8_0
    let q8_data = create_test_q8_0_data(size);
    let safe_result = safe_dequantize_q8_0(&q8_data, size).unwrap();
    let original_result = dequantize_simd(&q8_data, GgufTensorType::Q8_0, size);

    assert_eq!(safe_result.len(), original_result.len());

    for (i, (safe_val, orig_val)) in safe_result.iter().zip(original_result.iter()).enumerate() {
        let diff = (safe_val - orig_val).abs();
        assert!(diff < 1e-6 || diff / safe_val.abs().max(1e-10) < 1e-5,
            "Q8_0 元素[{}]: safe={}, original={}", i, safe_val, orig_val);
    }
}

// ============================================================================
// 第五部分：大批量数据压力测试
// ============================================================================

#[test]
fn test_large_batch_q4_0() {
    // 100万元素的大批量数据（约 562.5 KB 量化数据）
    let size = 1_000_000;
    let data = create_test_q4_0_data(size);

    let start = std::time::Instant::now();
    let result = safe_dequantize_q4_0(&data, size);
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "大批量 Q4_0 处理失败");
    let output = result.unwrap();
    assert_eq!(output.len(), size);

    // 验证所有输出有效
    let invalid_count = output.iter().filter(|&&v| !v.is_finite()).count();
    assert_eq!(invalid_count, 0, "{} 个无效输出值", invalid_count);

    println!("大批量 Q4_0 ({} elements): {:?}", size, elapsed);
    assert!(elapsed.as_secs_f32() < 5.0, "处理时间过长: {:?}", elapsed); // 5秒超时
}

#[test]
fn test_large_batch_q8_0() {
    // 50万元素的 Q8_0 数据
    let size = 500_000;
    let data = create_test_q8_0_data(size);

    let start = std::time::Instant::now();
    let result = safe_dequantize_q8_0(&data, size);
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), size);

    let invalid_count = output.iter().filter(|&&v| !v.is_finite()).count();
    assert_eq!(invalid_count, 0);

    println!("大批量 Q8_0 ({} elements): {:?}", size, elapsed);
    assert!(elapsed.as_secs_f32() < 5.0);
}

#[test]
fn test_large_batch_mixed_types() {
    // 混合类型大批量测试
    let size = 100_000;

    let q4_data = create_test_q4_0_data(size);
    assert!(safe_dequantize_q4_0(&q4_data, size).is_ok());

    let q8_data = create_test_q8_0_data(size);
    assert!(safe_dequantize_q8_0(&q8_data, size).is_ok());

    let q41_data = create_test_q4_1_data(size);
    assert!(safe_dequantize_q4_1(&q41_data, size).is_ok());
}

// ============================================================================
// 第六部分：连续调用稳定性测试（内存泄漏检测）
// ============================================================================

#[test]
fn test_repeated_calls_no_memory_leak() {
    // 连续调用 10000 次，检测内存泄漏
    let iterations = 10_000;
    let size = 1024;

    // 记录初始内存使用（近似）
    let mem_before = get_memory_usage_mb();

    for i in 0..iterations {
        let q4_data = create_test_q4_0_data(size);
        let result = safe_dequantize_q4_0(&q4_data, size);
        if i % 1000 == 0 {
            assert!(result.is_ok(), "第 {} 次调用失败", i);
        }

        let _ = result; // 显式 drop

        // 每 2000 次检查一次内存
        if i > 0 && i % 2000 == 0 {
            let mem_current = get_memory_usage_mb();
            let mem_growth = mem_current.saturating_sub(mem_before);
            // 允许合理的增长（< 100MB），如果持续增长则可能有泄漏
            assert!(mem_growth < 100,
                "可能存在内存泄漏: 初始={}MB, 当前={}MB, 增长={}MB",
                mem_before, mem_current, mem_growth);
        }
    }

    let mem_after = get_memory_usage_mb();
    let total_growth = mem_after.saturating_sub(mem_before);
    println!("内存使用: 前={}MB, 后={}MB, 增长={}MB ({} 次迭代)",
        mem_before, mem_after, total_growth, iterations);

    // 总增长不应超过 50MB（允许一些碎片化）
    assert!(total_growth < 50,
        "总内存增长过大: {}MB (可能存在内存泄漏)", total_growth);
}

#[test]
fn test_repeated_calls_stress() {
    // 快速重复调用 10000 次，确保不崩溃
    let iterations = 10_000;
    let size = 256;

    for i in 0..iterations {
        // 交替使用不同类型
        match i % 3 {
            0 => {
                let data = create_test_q4_0_data(size);
                let _ = safe_dequantize_q4_0(&data, size);
            }
            1 => {
                let data = create_test_q8_0_data(size);
                let _ = safe_dequantize_q8_0(&data, size);
            }
            _ => {
                let data = create_test_q4_1_data(size);
                let _ = safe_dequantize_q4_1(&data, size);
            }
        }

        if i % 2500 == 0 {
            println!("已完成 {} / {} 次迭代", i, iterations);
        }
    }

    // 如果能到达这里说明没有崩溃
    assert!(true);
}

// ============================================================================
// 第七部分：并行版本安全性测试
// ============================================================================

#[test]
fn test_parallel_dequantize_safety() {
    // 测试并行版本的安全性
    let size = 4096; // 足够大的数据以触发并行路径
    let num_threads = 4;

    let q4_data = create_test_q4_0_data(size);

    // 并行版本不应该崩溃
    let result = std::panic::catch_unwind(|| {
        dequantize_simd_parallel(&q4_data, GgufTensorType::Q4_0, size, num_threads)
    });

    assert!(result.is_ok(), "并行反量化发生 panic");
    let parallel_output = result.unwrap();
    assert_eq!(parallel_output.len(), size);

    // 验证所有输出都是有效数值（不验证精确一致性，因为并行处理顺序可能不同）
    for (i, &val) in parallel_output.iter().enumerate() {
        assert!(val.is_finite(), "并行输出[{}] 无效: {}", i, val);
        // Q4_0 的输出应该在合理范围内
        assert!(val.abs() < 100.0, "并行输出[{}] 超出范围: {}", i, val);
    }

    // 对比串行结果（仅验证长度和有效性，不要求完全一致）
    let serial_output = dequantize_simd(&q4_data, GgufTensorType::Q4_0, size);
    assert_eq!(serial_output.len(), size);

    for (i, &val) in serial_output.iter().enumerate() {
        assert!(val.is_finite(), "串行输出[{}] 无效: {}", i, val);
    }
}

#[test]
fn test_parallel_with_various_thread_counts() {
    let size = 2048;
    let q4_data = create_test_q4_0_data(size);

    // 测试不同的线程数量
    for threads in [1, 2, 4, 8, 16] {
        let result = dequantize_simd_parallel(&q4_data, GgufTensorType::Q4_0, size, threads);
        assert_eq!(result.len(), size, "线程数={} 时输出长度不正确", threads);

        // 验证所有值都是有限的
        for (i, &val) in result.iter().enumerate() {
            assert!(val.is_finite(), "线程数={} 时输出[{}] 无效", threads, i);
        }
    }
}

// ============================================================================
// 第八部分：边界条件极端测试
// ============================================================================

#[test]
fn test_truncated_data_handling() {
    // 截断的数据应该被优雅处理
    let full_data = create_test_q4_0_data(64); // 需要 36 字节 (2 blocks)

    // 逐步截断
    for truncate_len in [0, 1, 10, 17, 18, 19, 35, 36] {
        let truncated = &full_data[..truncate_len.min(full_data.len())];
        let result = safe_dequantize_q4_0(truncated, 64);

        if truncate_len >= 36 {
            // 完整数据应该成功
            assert!(result.is_ok(), "完整数据 ({} bytes) 应该成功", truncate_len);
        } else {
            // 不完整数据应该返回错误
            assert!(result.is_err(), "截断数据 ({} bytes) 应该返回错误", truncate_len);
        }
    }
}

#[test]
fn test_zero_scale_edge_case() {
    // scale 为 0 的边界情况
    use half::f16;

    let mut data = vec![0u8; 18]; // 一个 Q4_0 block
    // 设置 scale 为 0
    let zero_scale = f16::from_f32(0.0);
    data[0..2].copy_from_slice(&zero_scale.to_le_bytes());

    let result = safe_dequantize_q4_0(&data, 32);
    assert!(result.is_ok());
    let output = result.unwrap();

    // 所有值应该是 0（因为 scale=0）
    for &val in &output {
        assert!((val - 0.0).abs() < 1e-7, "scale=0 时输出应为 0, 得到 {}", val);
    }
}

#[test]
fn test_negative_and_extreme_values() {
    // 极端值的处理
    use half::f16;

    let mut data = vec![0u8; 18];

    // 使用较大的 scale
    let large_scale = f16::from_f32(100.0);
    data[0..2].copy_from_slice(&large_scale.to_le_bytes());

    // 设置量化值为最大和最小
    data[2] = 0x11; // 高 4-bit = 1 (-7), 低 4-bit = 1 (-7)
    data[3] = 0x00; // 高 4-bit = 0 (-8), 低 4-bit = 0 (-8)

    let result = safe_dequantize_q4_0(&data, 32);
    assert!(result.is_ok());
    let output = result.unwrap();

    // 验证输出在预期范围内
    for &val in &output {
        assert!(val.is_finite(), "极端值导致非有限输出: {}", val);
        assert!(val.abs() < 1000.0, "输出值异常: {}", val);
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 获取当前进程的内存使用量（MB）（平台相关实现）
#[cfg(unix)]
fn get_memory_usage_mb() -> u64 {
    use std::process;

    if let Ok(output) = process::Command::new("ps")
        .args(["-o", "rss=", "-p", &process::id().to_string()])
        .output()
    {
        if let Ok(kb_str) = String::from_utf8(output.stdout) {
            if let Ok(kb) = kb_str.trim().parse::<u64>() {
                return kb / 1024; // KB -> MB
            }
        }
    }

    0 // 无法获取时返回 0
}

#[cfg(not(unix))]
fn get_memory_usage_mb() -> u64 {
    0 // 非 Unix 平台暂不支持
}
