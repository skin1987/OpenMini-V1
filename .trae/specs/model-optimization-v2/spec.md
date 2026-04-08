# Model.rs 深度优化规范 V2

## Why
当前 `model.rs` 实现存在以下关键问题：
1. Top-K 选择可能导致索引越界（长度不匹配）
2. Attention/MLA 仍使用三重循环 O(N²·H·D)，未利用 BLAS
3. RMSNorm 仍有显式循环
4. 无 KV 缓存导致生成复杂度 O(L³)
5. sample_token 全词表排序开销大
6. apply_rotary_emb 未向量化
7. VisionEncoderWeights 未优化
8. 未使用变量和死代码
9. 错误处理不完善

## What Changes

### 1. 修复 Top-K 索引越界
- [ ] top_k_selection 确保每行返回恰好 k 个元素
- [ ] 使用默认值填充不足 k 个元素的情况
- [ ] 或在 MoEWeights::forward 中记录每行实际专家数

### 2. Attention 完全向量化
- [ ] 将 AttentionWeights::forward 改为纯矩阵运算
- [ ] `scores = q_h.dot(&k_h.t()) * scale`
- [ ] `attn = softmax(&scores)`
- [ ] `out_h = attn.dot(&v_h)`

### 3. MLA 完全向量化
- [ ] mla_forward 中的注意力计算改为矩阵乘法
- [ ] 消除三层嵌套循环

### 4. RMSNorm 完全向量化
- [ ] 使用广播操作替代逐元素循环
- [ ] 利用 ndarray 的广播语义

### 5. KV 缓存支持
- [ ] 集成 memory 字段实现缓存
- [ ] 避免重复计算整个序列

### 6. sample_token 优化
- [ ] 使用 select_nth_unstable 替代全排序
- [ ] 或使用多项式采样

### 7. apply_rotary_emb 向量化
- [ ] 预计算 cos/sin 位置编码表
- [ ] 使用广播乘法应用

### 8. VisionEncoderWeights 优化
- [ ] patch 嵌入向量化
- [ ] 注意力计算矩阵化

### 9. 代码清理
- [ ] 移除未使用变量
- [ ] 完善 from_gguf 占位函数
- [ ] 统一错误处理

## Impact
- Affected specs: unified-compute（SIMD 加速）
- Affected code: `openmini-server/src/model/inference/model.rs`

## ADDED Requirements

### Requirement: Top-K 索引安全
top_k_selection 函数必须返回每行恰好 k 个元素，不足时用默认值填充。

#### Scenario: 专家数不足
- **WHEN** 某行专家概率值数量 < k
- **THEN** 返回 k 个元素，缺失位置用 0.0 填充

### Requirement: Attention 完全向量化
Attention 计算必须使用纯矩阵运算，无任何嵌套循环。

#### Scenario: QKV 注意力
- **WHEN** 执行 AttentionWeights::forward
- **THEN** 使用 `q @ k.t()` 和 `attn @ v` 形式

### Requirement: KV 缓存
推理时必须缓存已计算的 Key/Value，避免重复计算。

#### Scenario: 自回归生成
- **WHEN** 生成第 N+1 个 token
- **THEN** 复用前 N 个 token 的 KV 缓存

## MODIFIED Requirements

### Requirement: RMSNorm 实现
RMSNorm 应完全向量化，使用 ndarray 广播操作，无显式循环。

### Requirement: sample_token 实现
应使用 O(V) 或 O(V log k) 算法替代 O(V log V) 排序。

## REMOVED Requirements

### Requirement: 三重循环 Attention
**Reason**: 性能极差，无法利用 BLAS
**Migration**: 改为批量矩阵运算
