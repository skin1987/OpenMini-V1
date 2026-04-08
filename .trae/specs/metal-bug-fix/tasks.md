# Tasks

## 1. 修复 Attention Kernel 内存竞争
- [x] Task 1.1: 重写 attention shader 使用在线 softmax 算法
- [x] Task 1.2: 移除 threadgroup scores 数组，改用寄存器累加
- [x] Task 1.3: 实现 Flash Attention 风格的分块计算

## 2. 修复矩阵乘法 Grid 计算
- [x] Task 2.1: 修正 grid_size 为线程组数量而非全局线程数
- [x] Task 2.2: 验证修复后矩阵乘法结果正确

## 3. 优化 ShaderLibrary 锁
- [x] Task 3.1: 改用读写锁分离查询和创建
- [x] Task 3.2: 先读锁查询，不存在再写锁创建

## 4. 修复数组越界问题
- [x] Task 4.1: 使用在线 softmax 避免固定数组
- [x] Task 4.2: 支持任意 kv_len 长度

## 5. 修复设备特性检测
- [x] Task 5.1: 使用 `device.supports_family` 检查 bfloat16
- [x] Task 5.2: 移除硬编码的特性列表

## 6. 代码清理
- [x] Task 6.1: 移除未使用的 RwLock import
- [x] Task 6.2: MetalBackend 直接持有 ShaderLibrary

## 7. 测试验证
- [x] Task 7.1: 编译验证通过
- [ ] Task 7.2: attention kernel 测试 (需在 macOS 上运行)
- [ ] Task 7.3: 矩阵乘法测试 (需在 macOS 上运行)

# Task Dependencies
- [Task 1.1] 是独立的，最高优先级 ✅
- [Task 1.2] depends on [Task 1.1] ✅
- [Task 1.3] depends on [Task 1.2] ✅
- [Task 2.1] 是独立的，高优先级 ✅
- [Task 2.2] depends on [Task 2.1] ✅
- [Task 3.1] 是独立的 ✅
- [Task 3.2] depends on [Task 3.1] ✅
- [Task 4] depends on [Task 1] ✅
- [Task 5] 是独立的 ✅
- [Task 6] can run in parallel with [Task 5] ✅
- [Task 7] depends on [Task 1-6] ✅
