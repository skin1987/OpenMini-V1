# Tasks

## 1. 修复 Top-K 索引越界
- [x] Task 1.1: 修改 top_k_selection 确保每行返回恰好 k 个元素
- [x] Task 1.2: 在 MoEWeights::forward 中正确处理变长专家选择

## 2. Attention 完全向量化
- [x] Task 2.1: 重写 AttentionWeights::forward 使用纯矩阵运算
- [x] Task 2.2: 移除所有嵌套循环，改用 q @ k.t() 形式
- [x] Task 2.3: 验证输出正确性

## 3. MLA 完全向量化
- [x] Task 3.1: 重写 mla_forward 注意力计算为矩阵乘法
- [x] Task 3.2: 消除三层嵌套循环
- [x] Task 3.3: 验证 MLA 输出与原实现一致

## 4. RMSNorm 完全向量化
- [x] Task 4.1: 使用广播操作替代逐元素循环

## 5. KV 缓存支持
- [x] Task 5.1: 设计 KV 缓存结构
- [x] Task 5.2: 在 forward 时更新缓存
- [x] Task 5.3: 生成时复用缓存

## 6. sample_token 优化
- [x] Task 6.1: 使用 select_nth_unstable 替代全排序
- [x] Task 6.2: 实现 top-p 截断采样

## 7. apply_rotary_emb 向量化
- [x] Task 7.1: 预计算 cos/sin 位置编码表
- [x] Task 7.2: 使用广播乘法应用 RoPE

## 8. VisionEncoderWeights 优化
- [x] Task 8.1: patch 嵌入向量化
- [x] Task 8.2: 注意力计算矩阵化

## 9. 代码清理
- [x] Task 9.1: 移除未使用变量
- [x] Task 9.2: 完善 from_gguf 函数
- [x] Task 9.3: 统一错误处理

## 10. 测试验证
- [x] Task 10.1: 添加数值正确性测试
- [x] Task 10.2: 编译验证
- [x] Task 10.3: 端到端生成测试

# Task Dependencies
- [Task 1.1] 是独立的
- [Task 1.2] depends on [Task 1.1]
- [Task 2.1] depends on [Task 1.1]
- [Task 2.2] depends on [Task 2.1]
- [Task 2.3] depends on [Task 2.2]
- [Task 3.1] can run in parallel with [Task 2.1]
- [Task 3.2] depends on [Task 3.1]
- [Task 3.3] depends on [Task 3.2]
- [Task 4.1] can run in parallel with [Task 2.1]
- [Task 5] depends on [Task 2, 3]
- [Task 6] is independent
- [Task 7] can run in parallel with [Task 6]
- [Task 8] can run in parallel with [Task 6]
- [Task 9] can run in parallel with [Task 8]
- [Task 10] depends on [Task 1-9]
