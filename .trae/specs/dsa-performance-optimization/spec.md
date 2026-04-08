# DSA 性能优化规范

## Why

当前 DSA 实现虽然功能完整，但存在以下性能瓶颈：

1. **Lightning Indexer CPU 瓶颈** - Q@K^T 矩阵乘法使用 ndarray 纯 Rust 实现，长序列性能受限
2. **Top-K 选择未充分优化** - 使用标准库排序，未利用 GPU 并行能力
3. **内存访问模式低效** - 频繁的小块内存分配和访问
4. **未与 GPU 后端集成** - Metal/CUDA 后端未用于 DSA 加速

根据 OPTIMIZATION_SUMMARY.md，此优化预期带来 **5-10倍性能提升**。

## What Changes

### 1. GPU 加速 Lightning Indexer

- [ ] 创建 GPU 版本的 `lightning_indexer_gpu`
- [ ] 利用 Metal/CUDA 后端加速 Q@K^T 计算
- [ ] 自动检测并选择最优后端（GPU/CPU）
- [ ] 支持批量处理多个查询

### 2. 优化 Top-K 选择算法

- [ ] 实现 GPU 并行 Top-K 选择
- [ ] 使用堆算法优化 CPU Top-K
- [ ] 批量处理多个查询的 Top-K
- [ ] 减少内存分配开销

### 3. 内存访问模式优化

- [ ] 使用内存池管理临时缓冲区
- [ ] 优化数据布局提高缓存命中率
- [ ] 预分配输出缓冲区
- [ ] 减少不必要的数据拷贝

### 4. 自适应后端选择

- [ ] 根据序列长度自动选择 CPU/GPU
- [ ] 实现性能分析和基准测试
- [ ] 提供配置选项控制后端选择

## Impact

- Affected code: `openmini-server/src/model/inference/dsa.rs`
- Affected modules: `hardware/gpu/`, `kernel/`
- Performance: **5-10倍加速**（长序列场景）
- Memory: 减少临时内存分配

## ADDED Requirements

### Requirement: GPU 加速 Lightning Indexer

Lightning Indexer 必须支持 GPU 加速，自动选择最优后端。

#### Scenario: 长序列推理

- **WHEN** 序列长度 > 4096
- **THEN** 自动使用 GPU 后端加速 Q@K^T 计算
- **AND** 性能提升 5-10倍

### Requirement: 优化 Top-K 选择

Top-K 选择必须使用高效算法，支持 GPU 并行。

#### Scenario: 批量 Top-K 选择

- **WHEN** 需要为多个查询选择 Top-K
- **THEN** 使用批量并行算法
- **AND** 减少内存分配开销

### Requirement: 自适应后端选择

DSA 必须根据序列长度和硬件能力自动选择最优后端。

#### Scenario: 短序列推理

- **WHEN** 序列长度 < 1024
- **THEN** 使用 CPU 后端（避免 GPU 传输开销）

#### Scenario: 长序列推理

- **WHEN** 序列长度 >= 4096
- **THEN** 使用 GPU 后端（充分利用并行能力）

### Requirement: 内存优化

DSA 必须最小化内存分配和数据拷贝。

#### Scenario: 连续推理

- **WHEN** 执行多次 DSA 计算
- **THEN** 复用内存池中的缓冲区
- **AND** 减少分配开销

## Performance Targets

| 场景 | 当前性能 | 目标性能 | 提升倍数 |
|------|---------|---------|---------|
| 短序列 (1K) | 基准 | 1.2x | 1.2x |
| 中序列 (4K) | 基准 | 3-5x | 3-5x |
| 长序列 (16K) | 基准 | 5-10x | 5-10x |
| 超长序列 (32K) | 基准 | 8-12x | 8-12x |

## Implementation Plan

### Phase 1: GPU Lightning Indexer (P0)

1. 在 `kernel/metal/mod.rs` 中实现 `matmul_batch` 函数
2. 在 `dsa.rs` 中添加 `lightning_indexer_gpu` 函数
3. 实现自动后端选择逻辑
4. 添加性能基准测试

### Phase 2: Top-K 优化 (P0)

1. 实现堆算法优化的 CPU Top-K
2. 实现 GPU 并行 Top-K（Metal/CUDA）
3. 批量处理优化
4. 性能对比测试

### Phase 3: 内存优化 (P1)

1. 创建 DSA 内存池
2. 优化数据布局
3. 减少临时分配
4. 内存使用分析

### Phase 4: 集成测试 (P1)

1. 端到端性能测试
2. 正确性验证
3. 不同硬件平台测试
4. 文档更新

## Testing Strategy

### Unit Tests

- GPU Lightning Indexer 正确性
- Top-K 选择正确性
- 后端选择逻辑
- 内存池管理

### Performance Tests

- 不同序列长度的性能对比
- CPU vs GPU 性能对比
- 内存使用分析
- 吞吐量测试

### Integration Tests

- 与 FlashAttention-3 集成
- 与 Continuous Batching 集成
- 端到端推理测试

## Success Criteria

- ✅ GPU Lightning Indexer 实现并通过测试
- ✅ 长序列（16K+）性能提升 5-10倍
- ✅ 内存分配减少 50%+
- ✅ 所有单元测试通过
- ✅ 性能基准测试建立
- ✅ 文档完善

