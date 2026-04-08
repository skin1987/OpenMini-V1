# Tasks

## 1. 修复 Mask -INF 导致 NaN 问题 ✅
- [x] Task 1.1: 在 attention kernel 中添加 score 有效性检查
- [x] Task 1.2: 当 score <= -1e30 时跳过在线 softmax 更新
- [x] Task 1.3: 验证因果掩码场景无 NaN

## 2. 移除线程组自动缩放逻辑 ⏸️
- [ ] Task 2.1: 移除 execute_kernel 中的自动缩放代码
- [ ] Task 2.2: 使用固定的、硬件兼容的块大小
- [ ] Task 2.3: 验证矩阵乘法正确性

**状态**: 延迟处理。当前 Apple GPU 环境下不会触发（16x16=256 < 1024）。

## 3. 异步执行支持 ❌
- [ ] Task 3.1: 修改 execute_kernel 返回异步句柄
- [ ] Task 3.2: 添加同步方法 execute_kernel_sync
- [ ] Task 3.3: 验证异步执行正确性

**状态**: 未处理。需要更复杂的设计，属于可选优化。

## 4. 代码清理 ✅
- [x] Task 4.1: Threadgroup 内存声明检查 - 循环内声明是正确做法
- [x] Task 4.2: 基准测试移到独立模块 - 已改为 #[cfg(test)]
- [x] Task 4.3: test_layer_norm 完整性检查 - 测试完整无截断

## 5. 编译验证 ✅
- [x] Task 5.1: cargo check 无错误
- [x] Task 5.2: cargo test 通过

# Task Dependencies
- [Task 1.1] 是独立的 ✅
- [Task 2.1] 独立，但延迟 ⏸️
- [Task 3] 需要更多设计 ❌
- [Task 5] 依赖 [Task 1] ✅

# 完成状态总结

| 任务 | 状态 | 说明 |
|------|------|------|
| Task 1 | ✅ 完成 | NaN 防护已添加 |
| Task 2 | ⏸️ 延迟 | 当前环境不会触发 |
| Task 3 | ❌ 未处理 | 可选优化 |
| Task 4 | ✅ 完成 | 基准测试已隔离 |
| Task 5 | ✅ 完成 | 编译测试通过 |
