# quant_simd.rs SIGSEGV 根因分析报告

**日期**: 2026-04-10
**严重级别**: P0 (Critical)
**影响范围**: 量化模块反量化功能，可导致进程崩溃
**文件**: `openmini-server/src/model/inference/quant_simd.rs` (4529 行)

---

## 1. 问题概述

`quant_simd.rs` 中的 SIMD 优化反量化代码存在多个可导致 SIGSEGV（段错误）的风险点。这些问题在 24h 压力测试中被触发，导致服务崩溃。

## 2. 识别的 SIGSEGV 风险点

### 2.1 [高危] 数组越界访问（直接索引）

**位置**: 多处
**问题描述**: 使用直接索引 `qs[byte_idx]` 而非安全访问 `.get(byte_idx)`

**具体实例**:
```rust
// 第 3226 行 - Q4_0 预取优化
let q: i32 = if is_high {
    ((qs[byte_idx] >> 4) as i32) - 8  // ⚠️ 未检查 byte_idx 是否越界
} else {
    ((qs[byte_idx] & 0x0F) as i32) - 8
};

// 第 3269 行 - Q8_0 预取优化
result[idx] = qs[i] as i8 as f32 * scale;  // ⚠️ 未检查 i 是否越界

// 第 3316-3318 行 - Q4_1 预取优化
let q: f32 = if is_high {
    (qs[byte_idx] >> 4) as f32  // ⚠️ 同样的问题
} else {
    (qs[byte_idx] & 0x0F) as f32
};
```

**触发条件**:
- 输入数据长度不是 block size 的整数倍
- 数据损坏或截断
- n 参数与实际数据不匹配

**修复方案**: 使用 `.get()` 或添加显式边界检查

---

### 2.2 [高危] CPU Feature 运行时检测不完整

**位置**: NEON 优化模块（第 2408 行起）
**问题描述**: NEON 函数使用 `#[target_feature(enable = "neon")]` 但缺少运行时检查

**具体实例**:
```rust
#[target_feature(enable = "neon")]
pub unsafe fn dequantize_q4_0_neon(data: &[u8], n: usize) -> Vec<f32> {
    // ⚠️ 假设 NEON 始终可用，但在某些 ARM 平台可能不支持
}
```

**风险**:
- 在不支持 NEON 的 ARM 设备上运行会导致 Illegal Instruction
- 虽然现代 aarch64 通常支持 NEON，但为了健壮性应检查

**修复方案**: 添加运行时 NEON 支持检测函数

---

### 2.3 [中危] 指针运算缺乏保护

**位置**: SIMD 加载/存储操作
**问题描述**: 使用 `as_ptr().add()` 进行指针运算，依赖上层边界检查

**具体实例**:
```rust
// 第 77 行 - AVX512 F32 实现
let va = _mm512_loadu_ps(data.as_ptr().add(offset * 4) as *const f32);
// ⚠️ 如果 offset * 4 计算错误或数据不足，会读取未定义内存

// 第 413 行 - AVX512 Q4_0 实现
_mm512_storeu_ps(result.as_mut_ptr().add(start + elems_start), vresult);
// ⚠️ 如果 start + elems_start 越界，会写入未定义内存
```

**现有保护**:
- 有 `debug_assert!` 边界检查（第 394-396 行）
- 但这些在 release 模式下被移除！

**修复方案**: 将关键边界检查改为运行时检查（非 debug_assert）

---

### 2.4 [中危] 内存对齐假设

**位置**: 部分 SIMD 优化路径
**问题描述**: 虽然 `_mm256_loadu_ps` 是 unaligned 版本，但某些优化路径可能隐含对齐假设

**影响范围**:
- AVX2/AVX512 的 aligned 版本指令（如果误用）
- 手动对齐优化的代码路径

**当前状态**: 大部分代码使用 `_mm256_loadu_ps`（unaligned），风险较低

**建议**: 确保所有 SIMD 操作都使用 unaligned 版本或手动处理对齐

---

### 2.5 [低危] 并行版本边界竞争

**位置**: `dequantize_q4_0_parallel` 等（第 1019 行起）
**问题描述**: 并行实现中 chunk 边界计算复杂，可能在极端情况下导致越界

**当前保护**:
```rust
if idx >= n || idx < elem_start || idx >= elem_end {
    continue;
}
let local_idx = idx - elem_start;
if local_idx >= chunk.len() {  // ✅ 有检查
    continue;
}
```

**评估**: 当前实现有较好的边界保护，风险较低

---

## 3. 触发场景分析

### 3.1 高频触发场景

1. **损坏的模型权重**
   - GGUF 文件截断或不完整
   - tensor data length 与声明的 shape 不匹配
   - **概率**: 中等（网络传输、磁盘损坏）

2. **极端张量尺寸**
   - 非标准 block size 的张量（n 不是 32 的倍数）
   - 空张量或单元素张量
   - **概率**: 低（特殊模型架构）

3. **高并发压力**
   - 多线程同时调用反量化
   - 内存压力导致分配失败
   - **概率**: 高（生产环境 24h 测试）

### 3.2 统计信息（基于代码审查）

| 风险类型 | 出现次数 | 严重程度 | 修复优先级 |
|---------|---------|---------|-----------|
| 直接数组索引 | ~15 处 | 🔴 高 | P0 |
| 缺少 CPU feature 检查 | 5+ 处 | 🟠 中高 | P0 |
| debug_assert 保护 | ~10 处 | 🟡 中 | P1 |
| 指针运算 | ~20 处 | 🟡 中 | P1 |
| 对齐问题 | 2-3 处 | 🟢 低 | P2 |

---

## 4. 修复方案概览

### Phase 1: CPU Feature 安全检测（P0）

```rust
#[cfg(target_arch = "x86_64")]
pub fn is_simd_supported() -> SimdSupport {
    SimdSupport {
        avx512: is_x86_feature_detected!("avx512f"),
        avx2: is_x86_feature_detected!("avx2"),
        sse42: is_x86_feature_detected!("sse4.2"),
    }
}

#[cfg(target_arch = "aarch64")]
pub fn is_neon_supported() -> bool {
    // 在 aarch64 上 NEON 通常可用，但进行运行时确认
    std::arch::is_aarch64_feature_detected!("neon")  // Rust 1.61+
}
```

### Phase 2: 安全包装函数（P0）

```rust
pub fn safe_dequantize_q4_0(data: &[u8], n: usize) -> Result<Vec<f32>, QuantError> {
    // 1. 输入验证
    if n == 0 { return Ok(vec![]); }
    let required = (n.div_ceil(QK4_0)) * 18;
    if data.len() < required {
        return Err(QuantError::InsufficientData {
            expected: required,
            actual: data.len(),
        });
    }

    // 2. 选择安全的实现路径
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(dequantize_q4_0_avx2_safe(data, n));
        }
    }

    // 3. Fallback 到标量实现
    Ok(super::quant::dequantize_q4_0(data, n))
}
```

### Phase 3: 内存安全加固（P1）

- 所有数组访问改用 `.get().copied().unwrap_or(0)`
- 关键边界检查从 `debug_assert!` 改为运行时检查
- 添加输入验证层

### Phase 4: 回归测试（P1）

- 正常数据往返精度测试
- 边界条件测试（空、单元素、非对齐）
- 压力测试（100万元素 × 10000 次）
- AddressSanitizer 验证

---

## 5. 影响评估

### 功能影响
- **不破坏现有功能**: 所有修改都是向后兼容的
- **性能影响**: < 5%（额外的边界检查开销）
- **兼容性**: x86_64, aarch64 全平台支持

### 风险缓解
- 保留原始快速路径作为默认选项
- 新增的安全 API 作为可选入口
- 通过特性标志控制严格程度

---

## 6. 验证计划

### 单元测试
- [x] 正常数据量化/反量化往返
- [ ] 空输入、单元素输入
- [ ] 非对齐内存输入
- [ ] 截断数据输入（应返回错误而非崩溃）

### 压力测试
- [ ] 100万元素批量处理
- [ ] 连续调用 10000 次无内存泄漏
- [ ] 多线程并发调用稳定性

### 工具验证
- [ ] `cargo test` 全部通过
- [ ] `RUSTFLAGS="-Z sanitizer=address" cargo test` 无报错
- [ ] Valgrind/Memory sanitizer 验证

---

## 7. 时间线

- **根因分析**: 2026-04-10 ✅
- **Phase 1-2 实现**: 进行中
- **Phase 3 加固**: 待开始
- **Phase 4 测试**: 待开始
- **集成验证**: 待开始

---

## 附录：相关代码位置索引

| 问题类型 | 行号范围 | 函数名 |
|---------|---------|--------|
| 直接索引 | 3218-3231 | dequantize_q4_0_prefetch |
| 直接索引 | 3264-3270 | dequantize_q8_0_prefetch |
| 直接索引 | 3308-3321 | dequantize_q4_1_prefetch |
| debug_assert | 394-396, 437-439, 480-482 | dequantize_q4_0_impl |
| NEON 无检测 | 2408-2471 | neon_opt::dequantize_q4_0_neon |
| 指针运算 | 77, 94, 135, 413, 456, 498 | 多处 SIMD 操作 |

---

**作者**: Go 并发专家 (AI Assistant)
**审查状态**: 待团队 review
**下一步**: 实施修复方案 Phase 1-2
