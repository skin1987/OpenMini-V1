# Checklist - Metal GPU 后端 Bug 修复

## Attention Kernel 修复
- [x] attention shader 使用在线 softmax
- [x] 无 threadgroup 内存竞争
- [x] 支持任意 kv_len 长度

## 矩阵乘法修复
- [x] grid_size 使用线程组数量
- [x] 无无效线程执行
- [x] 性能符合预期

## ShaderLibrary 锁优化
- [x] 读锁查询已存在 pipeline
- [x] 写锁仅用于创建新 pipeline
- [x] 多线程并发不阻塞

## 数组越界修复
- [x] 使用在线 softmax 避免固定数组
- [x] 支持任意 kv_len

## 设备特性检测
- [x] bfloat16 通过 API 实际检查
- [x] 无硬编码特性列表

## 代码质量
- [x] 移除未使用的 import
- [x] MetalBackend 直接持有 ShaderLibrary

## 测试验证
- [x] 编译无错误无警告
- [ ] attention kernel 测试通过 (需在 macOS 上运行)
- [ ] 矩阵乘法测试通过 (需在 macOS 上运行)
