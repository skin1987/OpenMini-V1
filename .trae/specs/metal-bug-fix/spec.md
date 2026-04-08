# I’m Metal GPU 后端严重 Bug 修复规范

## Why

`metal.rs` 文件存在多个严重缺陷：

1. **Attention Kernel 共享内存竞争** - 所有线程共享同一块 threadgroup 内存写入数据，导致计算结果完全错误
2. **矩阵乘法 Grid 大小计算错误** - 导致执行大量无效线程，性能极低
3. **ShaderLibrary 锁使用不当** - 每次执行都获取写锁，导致完全串行化
4. **Kernel 内部数组大小固定** - 可能越界访问导致崩溃
5. **设备特性检测不准确** - 硬编码 bfloat16 特性但未实际检查

## What Changes

### 1. 修复 Attention Kernel 内存竞争

- [ ] 改用在线 softmax（Flash Attention 风格）
- [ ] 避免存储所有分数到共享内存

### 2. 修复矩阵乘法 Grid 计算

- [ ] `dispatch_thread_groups` 使用线程组数量而非全局线程数

### 3. 优化 ShaderLibrary 锁

- [ ] 先用读锁查询，不存在再获取写锁创建
- [ ] 或预先编译所有 pipeline

### 4. 修复数组越界问题

- [ ] 动态检查 kv\_len 是否超过固定数组大小
- [ ] 改用不需要预存所有分数的算法

### 5. 修复设备特性检测

- [ ] 通过 `device.supports_family` 实际检查 bfloat16 支持

## Impact

- Affected code: `openmini-server/src/hardware/gpu/metal.rs`
- **BREAKING**: Attention 计算结果将改变（修复后正确）

## ADDED Requirements

### Requirement: Attention Kernel 正确性

Attention kernel 必须正确计算注意力分数，无数据竞争。

#### Scenario: 多线程 Attention 计算

- **WHEN** 执行 attention kernel
- **THEN** 每个线程独立计算，结果正确

### Requirement: 矩阵乘法性能

矩阵乘法必须正确计算 grid 大小，避免无效线程。

#### Scenario: 大矩阵乘法

- **WHEN** 执行 matmul\_metal
- **THEN** 线程组数量正确，无浪费

### Requirement: ShaderLibrary 并发安全

ShaderLibrary 必须支持并发访问，不阻塞其他线程。

#### Scenario: 多线程 kernel 执行

- **WHEN** 多线程同时执行 kernel
- **THEN** 已存在的 pipeline 不阻塞

## REMOVED Requirements

### Requirement: 固定大小 scores 数组

**Reason**: 会导致越界访问
**Migration**: 改用在线 softmax 算法
