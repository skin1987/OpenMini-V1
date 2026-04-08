# Checklist - Model.rs 深度优化 V2

## Top-K 索引安全
- [x] top_k_selection 每行返回恰好 k 个元素
- [x] 不足时用默认值 0.0 填充
- [x] MoEWeights::forward 正确处理变长专家选择

## Attention 完全向量化
- [x] AttentionWeights::forward 使用纯矩阵运算
- [x] 无任何嵌套循环 for i/j/d
- [x] scores = q @ k.t() / sqrt(d)
- [x] attn = softmax(scores)
- [x] output = attn @ v

## MLA 完全向量化
- [x] mla_forward 无三层嵌套循环
- [x] 注意力分数计算矩阵化
- [x] 加权和计算矩阵化

## RMSNorm 完全向量化
- [x] 无显式 for 循环
- [x] 使用 ndarray 广播操作

## KV 缓存支持
- [x] 缓存结构设计完成 (KVCache, LayerKVCache)
- [x] forward 时更新缓存
- [x] 生成时复用缓存 (generate_with_cache)
- [x] 因果掩码正确实现

## sample_token 优化
- [x] 使用 select_nth_unstable
- [x] 无全词表排序
- [x] top-p 采样正确

## apply_rotary_emb 向量化
- [x] 预计算位置编码表 (cos/sin)
- [x] 使用广播乘法

## VisionEncoderWeights 优化
- [x] patch 嵌入向量化
- [x] 注意力计算矩阵化

## 代码清理
- [x] 无未使用变量
- [x] from_gguf 有实际实现
- [x] 错误处理统一使用 ?

## 测试验证
- [x] 数值正确性测试通过
- [x] 性能基准测试显示提升
- [x] 端到端生成正常
