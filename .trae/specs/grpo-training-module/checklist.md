# 大模型训练与预训练模块 - 验证清单

## Phase 1: 核心基础设施验证

### Autograd 自动微分引擎

- [ ] **文件结构与基础类型**
  - [ ] `training/autograd.rs` 文件存在且可编译
  - [ ] `Tensor` 结构体包含 value, grad, requires_grad 字段
  - [ ] ComputationGraph 正确表示 DAG（有向无环图）
  - [ ] GraphNode 枚举覆盖 Input, Param, Op, Loss 四种节点类型
  - [ ] 所有公开结构体和函数有文档注释

- [ ] **算子前向+反向实现**
  - [ ] MatMul 前向：Y = A @ B 形状正确，反向梯度公式正确（dA = dY @ B^T）
  - [ ] Add 前向/反向正确（简单传递）
  - [ ] LayerNorm 反向正确（需缓存 mean 和 variance）
  - [ ] Softmax 反向正确（dY * Y * (1-Y) 对角线近似）
  - [ ] GELU 反向正确（GELU 激活函数的导数）
  - [ ] Embedding Lookup 反向正确（scatter add 到对应 embedding 行）

- [ ] **反向传播引擎**
  - [ ] 拓扑排序算法正确（后序 DFS 遍历）
  - [ ] 反向遍历顺序与拓扑排序逆序一致
  - [ ] 梯度在多个路径汇合时正确累加（而非覆盖）
  - [ ] Param 节点的 .grad 在 backward 后包含正确的梯度值
  - [ ] 非 Param 节点不保留梯度（或按设计决定）

- [ ] **数值验证测试**
  - [ ] 简单线性网络（Embedding → Linear → Softmax → CELoss）的梯度通过有限差分法验证
  - [ ] 数值梯度与分析梯度的最大误差 < 1e-4
  - [ ] 空图、单节点图等边界情况不 panic

- [ ] **梯度裁剪**
  - [ ] L2 范数计算正确
  - [ ] 超过 max_norm 时等比缩放所有梯度
  - [ ] 未超过时保持原样

### MultimodalTransformer 训练模式扩展

- [ ] **forward_train 方法**
  - [ ] `forward_train()` 方法签名正确（input_ids, labels → TrainForwardResult）
  - [ ] 返回的 logits 形状为 [seq_len, vocab_size]
  - [ ] 当提供 labels 时返回正确的 loss 值（CrossEntropy）
  - [ ] ActivationCache 包含每层的 hidden_states
  - [ ] ActivationCache 可选包含 attention_scores（用于可视化/调试）

- [ ] **参数管理**
  - [ ] `trainable_params()` 返回所有需要优化的参数引用
  - [ ] 参数列表包括：embedding, 各层 attention 权重, FFN/MoE 权重, LayerNorm, lm_head
  - [ ] 每个参数有对应的梯度存储空间
  - [ ] 参数总数量与模型规模匹配

- [ ] **权重保存/加载**
  - [ ] `save_weights()` 生成有效的 safetensors 文件
  - [ ] `load_weights()` 从 safetensors 文件恢复权重
  - [ ] 保存→加载循环后模型输出完全一致（误差 < 1e-6）
  - [ ] 权重名称与标准命名约定一致（如 "model.layers.0.self_attn.q_proj.weight"）
  - [ ] 不存在的权重文件返回明确错误

## Phase 2: 优化器与损失函数验证

### Optimizer 优化器系统

- [ ] **AdamW 实现**
  - [ ] m_buffer 和 v_buffer 初始化为全零
  - [ ] step() 后 m = β₁*m + (1-β₁)*g 公式正确
  - [ ] step() 后 v = β₂*v + (1-β₂)*g² 公式正确
  - [ ] bias correction 正确应用（除以 1-β^t）
  - [ ] weight decay 解耦应用（θ = θ - lr * (m̂/(√v̂+ε) + λ*θ)）
  - [ ] zero_grad() 将所有参数梯度归零

- [ ] **SGD with Momentum**
  - [ ] momentum buffer 正确更新
  - [ ] 学习率正确应用

- [ ] **学习率调度器**
  - [ ] CosineWithWarmup:
    - [ ] warmup 阶段线性增长（从 0 到 target_lr）
    - [ ] decay 阶段余弦衰减（从 target_lr 到 ~0）
    - [ ] warmup_steps + decay_steps = total_steps
  - [ ] LinearWithWarmup:
    - [ ] warmup 线性增长
    - [ ] decay 线性衰减到 min_lr
  - [ ] ConstantScheduler: 始终返回 target_lr

- [ ] **单元测试**
  - [ ] AdamW 更新结果与手动计算一致（3 步内迭代验证）
  - [ ] 学习率曲线采样点符合预期形状

### Loss Functions 损失函数

- [ ] **CrossEntropyLoss**
  - [ ] 基础版本：loss = -mean(log(softmax(logits)[labels]))
  - [ ] ignore_index=-100 时 padding 位置 loss 为 0 且不计入均值
  - [ ] label_smoothing > 0 时分布更平滑（loss 略高于无 smoothing）
  - [ ] reduction="sum" 时返回总和，"none" 时返回逐样本 loss
  - [ ] 梯度形状与 logits 一致
  - [ ] 边界条件：空输入、全相同 label、单样本

## Phase 3: 数据处理流水线验证

### DataLoader 数据加载器

- [ ] **数据解析**
  - [ ] JSONL Causal LM 格式解析正确（{"text": "..."}）
  - [ ] JSONL SFT 格式解析正确（instruction/input/output）
  - [ ] 纯文本格式（每行一个样本）解析正确
  - [ ] 格式错误行给出警告并跳过（不中断整个加载）
  - [ ] 空文件返回空迭代器
  - [ ] UTF-8 编码异常处理

- [ ] **Tokenize 流程**
  - [ ] 文本正确转换为 token IDs（集成现有 Tokenizer）
  - [ ] 超过 max_seq_length 的序列被截断
  - [ ] 不足长度的序列用 pad_token_id 填充
  - [ ] Labels = input_ids shifted right（首位置为 -100 或 pad_token_id）
  - [ ] Attention mask 正确标记有效位置（1）和 padding（0）
  - [ ] EOS token 处理正确

- [ ] **Batch 构建**
  - [ ] 固定 batch_size 打包：每个 batch 包含恰好 batch_size 个样本
  - [ ] 最后一个 batch：drop_last=true 时丢弃，false 时保留小 batch
  - [ ] Dynamic Batching：同长度或相近长度的样本分到同一 batch
  - [ ] Batch 的 input_ids 形状为 [batch_size, seq_len]
  - [ ] Batch 的 labels 形状为 [batch_size, seq_len]
  - [ ] Batch 的 attention_mask 形状为 [batch_size, seq_len]

- [ ] **Shuffle 与迭代**
  - [ ] 每个 epoch 开始时 shuffle（两次调用顺序不同）
  - [ ] Shuffle 是均匀随机排列（卡方检验 p > 0.05）
  - [ ] 实现 Iterator trait（支持 for 循环）
  - [ ] 多次迭代可遍历完整数据集
  - [ ] 流式读取大文件时内存占用稳定（不随文件大小线性增长）

## Phase 4: 训练核心管理验证

### Trainer 训练管理器

- [ ] **配置与初始化**
  - [ ] TrainingConfig 从 TOML 正确反序列化
  - [ ] 缺少必填字段时返回明确错误
  - [ ] 超出范围的值（负 learning_rate）被捕获
  - [ ] Trainer::new() 成功初始化所有子组件

- [ ] **Causal LM Pre-training 模式**
  - [ ] 输入纯文本数据，labels 自动生成（shifted input_ids）
  - [ ] 执行完整 epoch-step 循环
  - [ ] 每 step 的 loss 有记录
  - [ ] Perplexity = exp(loss) 计算正确

- [ ] **Continue Pre-training 模式**
  - [ ] 从 checkpoint 加载模型权重成功
  - [ ] 从 checkpoint 恢复优化器状态成功
  - [ ] global_step 从断点继续
  - [ ] 训练指标连续无断层

- [ ] **SFT 模式**
  - [ ] prompt_template 正确替换 {instruction}, {input} 占位符
  - [ ] mask_prompt_loss=true 时只计算 response 部分 loss
  - [ ] prompt 部分的 label 设为 -100（忽略）
  - [ ] instruction + input + response 拼接正确

- [ ] **训练循环完整性**
  - [ ] Epoch 循环执行指定次数
  - [ ] Step 循环遍历完整 dataloader
  - [ ] 单步逻辑顺序：forward → loss → backward → clip → optimizer.step → metrics
  - [ ] Gradient accumulation 正确工作（累积 N 步后再更新）
  - [ ] 验证集评估在每个 epoch 结束后执行
  - [ ] Checkpoint 按 save_strategy 条件保存
  - [ ] 早停机制在满足 patience 后触发

- [ ] **状态管理**
  - [ ] Pause 后训练挂起（不再消耗计算资源）
  - [ ] Resume 后从暂停位置继续
  - [ ] SIGINT/SIGTERM 信号触发优雅关闭（保存 checkpoint 后退出）
  - [ ] TrainingState 实时反映当前进度

## Phase 5: 持久化与监控验证

### CheckpointManager 检查点管理

- [ ] **保存功能**
  - [ ] 生成目录结构：checkpoints/model_XXXXXX/
  - [ ] model_weights.safetensors 包含所有参数（名称+张量）
  - [ ] optimizer_state.bin 可反序列化恢复 AdamW 状态
  - [ ] training_state.json 包含 epoch, global_step, best_loss 等
  - [ ] metadata.json 包含时间戳、版本号
  - [ ] 保存时间 < 5s（小型模型）

- [ ] **加载功能**
  - [ ] load_checkpoint() 成功恢复模型权重
  - [ ] 优化器 m/v buffers 完全恢复
  - [ ] random state 恢复保证可复现性
  - [ ] 文件损坏或不完整时返回明确错误

- [ ] **管理策略**
  - [ ] 每 save_steps 步自动保存一次
  - [ ] 保留 best_model 符号链接指向最佳 val_loss checkpoint
  - [ ] 最多保留 save_total_limit 个 checkpoint
  - [ ] 超额时自动删除最旧的 checkpoint
  - [ ] 断点续训后训练指标连续

### TrainingMonitor 监控系统

- [ ] **指标采集**
  - [ ] MetricsRecord 包含 timestamp 和所有必需字段
  - [ ] RingBuffer 维护最近 N 步记录（N 可配置）
  - [ ] 超过容量时最旧记录被覆盖
  - [ ] 高频调用（每步一次）不会导致性能问题

- [ ] **统计聚合**
  - [ ] 均值计算准确（滑动窗口内的平均值）
  - [ ] 标准差计算准确
  - [ ] 趋势方向判断正确（上升/下降/稳定）
  - [ ] 移动平均平滑噪声

- [ ] **输出格式**
  - [ ] 日志格式符合 spec 示例（含 Epoch, Step, Loss, PPL, LR, GradNorm, Throughput, Time, ETA）
  - [ ] JSON 序列化可通过 API 返回
  - [ ] Perplexity = exp(loss) 计算正确
  - [ ] ETA 估算合理（基于最近 N 步平均速度）

## Phase 6: API 与配置验证

### TrainingAPI 控制接口

- [ ] **端点功能**
  - [ ] POST /start 启动训练任务，返回 training_id
  - [ ] POST /pause 将状态切换为 paused
  - [ ] POST /resume 恢复运行
  - [ ] POST /stop 优雅停止并保存最终 checkpoint
  - [ ] GET /status 返回完整的实时状态 JSON
  - [ ] GET /metrics?last_n=100 返回历史指标数组

- [ ] **错误处理**
  - [ ] 未启动时调用 pause/resume/stop 返回 409 Conflict
  - [ ] 已运行时调用 start 返回 409 Conflict
  - [ ] 无效配置返回 400 Bad Request（含具体字段名和原因）
  - [ ] 并发请求安全（无数据竞争）

- [ ] **路由注册**
  - [ ] 所有路由已注册到 Axum Router
  - [ ] URL 路径符合 RESTful 规范

### 配置系统

- [ ] **TOML 配置**
  - [ ] server.toml 包含完整的 `[training]` 及其子段
  - [ ] 所有配置项都有默认值
  - [ ] 嵌套结构（training.model, training.optimizer 等）正确解析
  - [ ] 配置验证捕获非法值

- [ ] **模块整合**
  - [ ] training 模块已在 lib.rs 或合适位置声明
  - [ ] 所有公共类型已导出
  - [ ] Cargo.toml 包含必要依赖
  - [ ] `cargo check --release` 通过（零错误零 warning）
  - [ ] `cargo test` 全部通过（新增 + 原有无回归）

---

## 端到端场景验证

### E2E 场景 1: 小规模预训练

- [ ] 使用 100 条 mock 文本数据运行 Causal LM 预训练
- [ ] 配置：epochs=2, batch_size=4, max_seq_len=128
- [ ] 训练正常完成，最终输出 summary
- [ ] Checkpoint 目录包含预期文件
- [ ] Loss 曲线呈现下降趋势（非严格单调但整体向下）

### E2E 场景 2: 断点续训

- [ ] 训练在第 1 个 epoch 第 50% 处中断
- [ ] 从最新 checkpoint 恢复
- [ ] 继续完成剩余训练
- [ ] 最终结果与未中断的训练一致（误差在浮点精度范围内）

### E2E 场景 3: API 控制流程

- [ ] POST start → 训练开始
- [ ] GET status → 返回 running
- [ ] POST pause → 状态变为 paused
- [ ] 等待 5 秒
- [ ] POST resume → 状态恢复为 running
- [ ] POST stop → 保存 checkpoint 并停止
- [ ] GET status → 返回 stopped + final_checkpoint 路径

---

## 验收标准总结

### P0 - 必须满足（阻塞发布）
- [ ] Autograd 能正确计算简单网络的梯度（数值验证通过）
- [ ] MultimodalTransformer 支持 forward_train 并缓存中间激活值
- [ ] AdamW 优化器能正确更新参数（公式验证通过）
- [ ] DataLoader 能加载 JSONL 数据并构建正确形状的 Batch
- [ ] Trainer 能完成至少 1 个 epoch 的完整训练循环
- [ ] CheckpointManager 支持保存和加载（权重一致性验证）
- [ ] TrainingAPI 提供 start/pause/resume/stop/status 端点
- [ ] 编译零错误，clippy 零 warning，测试全通过

### P1 - 应该满足（影响质量）
- [ ] CrossEntropyLoss 支持 ignore_index 和 label smoothing
- [ ] 学习率调度器（Warmup + Cosine）工作正确
- [ ] SFT 模式支持 prompt template 和 mask_prompt_loss
- [ ] TrainingMonitor 提供格式化日志和 JSON 接口
- [ ] 早停机制按预期触发
- [ ] Graceful Shutdown 保存 checkpoint 后退出
- [ ] 配置系统完整且有文档

### P2 - 可以延后（增强功能）
- [ ] Dynamic Batching（按长度分桶）
- [ ] 流式读取 TB 级大文件（mmap）
- [ ] Gradient Accumulation（模拟大 batch）
- [ ] 混合精度训练（FP16/BF16）
- [ ] 分布式训练支持（DDP/FSDP）
- [ ] TensorBoard/WandB 对接
- [ ] 更多损失函数（对比学习、RLHF 等）
