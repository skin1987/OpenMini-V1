# Tasks

## Phase 1: GPU Lightning Indexer (P0)

### 1.1 Metal 后端矩阵乘法增强
- [ ] Task 1.1.1: 在 `kernel/metal/mod.rs` 中实现批量矩阵乘法
- [ ] Task 1.1.2: 优化 Metal shader 的矩阵乘法性能
- [ ] Task 1.1.3: 添加矩阵乘法性能测试

### 1.2 GPU Lightning Indexer 实现
- [ ] Task 1.2.1: 创建 `lightning_indexer_gpu` 函数
- [ ] Task 1.2.2: 实现 GPU 数据传输优化
- [ ] Task 1.2.3: 添加错误处理和回退机制

### 1.3 自适应后端选择
- [ ] Task 1.3.1: 实现后端选择策略
- [ ] Task 1.3.2: 添加性能阈值配置
- [ ] Task 1.3.3: 实现运行时后端切换

### 1.4 性能基准测试
- [ ] Task 1.4.1: 创建 Lightning Indexer 基准测试
- [ ] Task 1.4.2: 对比 CPU vs GPU 性能
- [ ] Task 1.4.3: 生成性能报告

## Phase 2: Top-K 优化 (P0)

### 2.1 CPU Top-K 优化
- [ ] Task 2.1.1: 实现堆算法优化的 Top-K
- [ ] Task 2.1.2: 批量处理优化
- [ ] Task 2.1.3: SIMD 加速（如果可行）

### 2.2 GPU Top-K 实现
- [ ] Task 2.2.1: 设计 GPU Top-K 算法
- [ ] Task 2.2.2: 实现 Metal Top-K kernel
- [ ] Task 2.2.3: 实现 CUDA Top-K kernel（可选）

### 2.3 Top-K 集成
- [ ] Task 2.3.1: 集成到 `sparse_attention_forward`
- [ ] Task 2.3.2: 添加正确性测试
- [ ] Task 2.3.3: 性能对比测试

## Phase 3: 内存优化 (P1)

### 3.1 内存池设计
- [ ] Task 3.1.1: 创建 DSA 内存池结构
- [ ] Task 3.1.2: 实现缓冲区复用机制
- [ ] Task 3.1.3: 添加内存使用统计

### 3.2 数据布局优化
- [ ] Task 3.2.1: 分析当前内存访问模式
- [ ] Task 3.2.2: 优化数据布局提高缓存命中率
- [ ] Task 3.2.3: 减少数据拷贝

### 3.3 预分配优化
- [ ] Task 3.3.1: 预分配输出缓冲区
- [ ] Task 3.3.2: 复用临时缓冲区
- [ ] Task 3.3.3: 内存使用分析

## Phase 4: 集成测试 (P1)

### 4.1 端到端测试
- [ ] Task 4.1.1: 创建端到端推理测试
- [ ] Task 4.1.2: 验证不同序列长度的正确性
- [ ] Task 4.1.3: 性能回归测试

### 4.2 跨平台测试
- [ ] Task 4.2.1: macOS Metal 测试
- [ ] Task 4.2.2: Linux CUDA 测试（如果可用）
- [ ] Task 4.2.3: CPU 回退测试

### 4.3 文档更新
- [ ] Task 4.3.1: 更新 DSA 模块文档
- [ ] Task 4.3.2: 添加性能优化指南
- [ ] Task 4.3.3: 更新 OPTIMIZATION_SUMMARY.md

## Phase 5: 性能验证 (P1)

### 5.1 性能基准测试
- [ ] Task 5.1.1: 运行完整性能基准测试套件
- [ ] Task 5.1.2: 生成性能对比报告
- [ ] Task 5.1.3: 验证 5-10倍性能提升目标

### 5.2 内存分析
- [ ] Task 5.2.1: 分析内存使用情况
- [ ] Task 5.2.2: 验证内存优化效果
- [ ] Task 5.2.3: 生成内存使用报告

### 5.3 生产验证
- [ ] Task 5.3.1: 在实际模型上测试
- [ ] Task 5.3.2: 验证稳定性
- [ ] Task 5.3.3: 收集用户反馈

# Task Dependencies

## Phase 1 Dependencies
- [Task 1.1.1] 是独立的，最高优先级
- [Task 1.1.2] depends on [Task 1.1.1]
- [Task 1.1.3] depends on [Task 1.1.2]
- [Task 1.2.1] depends on [Task 1.1.1]
- [Task 1.2.2] depends on [Task 1.2.1]
- [Task 1.2.3] depends on [Task 1.2.2]
- [Task 1.3.1] depends on [Task 1.2.1]
- [Task 1.3.2] can run in parallel with [Task 1.3.1]
- [Task 1.3.3] depends on [Task 1.3.1, 1.3.2]
- [Task 1.4] depends on [Task 1.2, 1.3]

## Phase 2 Dependencies
- [Task 2.1] can run in parallel with [Phase 1]
- [Task 2.2] depends on [Task 1.1] (需要 GPU 基础设施)
- [Task 2.3] depends on [Task 2.1, 2.2]

## Phase 3 Dependencies
- [Task 3.1] can run in parallel with [Phase 2]
- [Task 3.2] depends on [Task 3.1]
- [Task 3.3] depends on [Task 3.2]

## Phase 4 Dependencies
- [Task 4.1] depends on [Phase 1, 2, 3]
- [Task 4.2] depends on [Task 4.1]
- [Task 4.3] depends on [Task 4.1, 4.2]

## Phase 5 Dependencies
- [Task 5] depends on [Phase 1, 2, 3, 4]

# Priority Summary

| Phase | Priority | Status | Estimated Time |
|-------|----------|--------|----------------|
| Phase 1 | P0 | 🔴 Not Started | 2-3 days |
| Phase 2 | P0 | 🔴 Not Started | 2-3 days |
| Phase 3 | P1 | 🔴 Not Started | 1-2 days |
| Phase 4 | P1 | 🔴 Not Started | 1-2 days |
| Phase 5 | P1 | 🔴 Not Started | 1 day |

**Total Estimated Time**: 7-11 days

