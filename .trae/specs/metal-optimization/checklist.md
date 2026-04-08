# Checklist - Metal GPU 后端优化

## Mask -INF NaN 防护 ✅
- [x] attention kernel 添加 score > -1e30 检查
- [x] attention_kv_cache kernel 添加 score > -1e30 检查
- [x] flash_attention kernel 添加 score > -1e30 检查
- [x] 添加 sum_exp > 0 检查防止除以零

## 线程组自动缩放 ⏸️
- [ ] 移除 execute_kernel 自动缩放逻辑
- [ ] 使用固定块大小

**原因**: 当前环境下不会触发（16x16=256 < 1024）

## 异步执行 ✅
- [x] execute_kernel_async 返回 MetalCommandHandle 句柄
- [x] execute_kernel 同步包装方法（内部调用 async + wait）
- [x] MetalCommandHandle 支持 wait / is_completed / label
- [x] submit_batch 批量提交支持
- [x] 5 个异步执行单元测试通过

## 代码清理 ✅
- [x] Threadgroup 内存声明 - 循环内声明是 Metal 正确做法
- [x] test_layer_norm 完整性 - 测试完整无截断
- [x] 基准测试移到独立模块 - 当前无 criterion 基准测试，全部为单元测试

## 编译验证 ✅
- [x] cargo check 无错误
- [x] cargo test 通过 (tokenizer: 31 passed, 0 failed)
