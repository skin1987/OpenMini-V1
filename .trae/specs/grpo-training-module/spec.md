# 大模型训练与预训练模块 Spec

## Why

OpenMini 项目已实现完整的 **Multimodal MoE Transformer 推理引擎**（DeepSeek-V2 架构，支持 MLA 注意力 + MoE），但**完全缺少模型训练能力**。当前无法进行：
- 从头预训练（Pre-training from scratch）
- 继续预训练（Continue Pre-training）
- 有监督微调（SFT - Supervised Fine-Tuning）

需要构建端到端的模型训练系统，使 OpenMini 成为一个完整的大模型训练+推理平台。

## What Changes

### 新增核心组件

#### 1. Autograd 自动微分引擎 (`autograd.rs`)
- **计算图构建**: 追踪前向传播中的所有操作
- **反向传播**: 自动计算所有参数的梯度
- **算子库**: 支持矩阵乘法、LayerNorm、Softmax、GELU、RoPE 等关键操作的梯度
- **内存优化**: 支持梯度 checkpointing（节省显存）

#### 2. Optimizer 优化器系统 (`optimizer.rs`)
- **AdamW**: 主流选择（自适应学习率 + 权重衰减）
- **SGD with Momentum**: 基线对比
- **学习率调度器**:
  - Linear Warmup + Cosine Decay（默认）
  - Linear Warmup + Linear Decay
  - Constant（调试用）
- **梯度裁剪**: L2 范数裁剪防止梯度爆炸
- **权重衰减**: 解耦式权重 decay（AdamW 特性）

#### 3. DataLoader 数据加载器 (`dataloader.rs`)
- **文本预处理**:
  - Tokenization（集成现有 Tokenizer）
  - BPE/MPE 分词
  - 最大序列长度截断/Padding
- **数据格式支持**:
  - 文本文件（每行一个样本）
  - JSONL 格式（`{"text": "..."}` 或 `{"instruction": ..., "input": ..., "output": ...}`）
  - 内存映射大文件（mmap）支持 TB 级数据集
- **Batch 构建**:
  - 固定 batch_size 打包
  - Dynamic Batching（按长度分桶减少 padding）
  - 数据 shuffle（每个 epoch 重排）
- **分布式采样**（未来扩展）: 多节点数据并行

#### 4. Trainer 训练管理器 (`trainer.rs`)
- **训练模式**:
  ```
  ┌─────────────────────────────────────┐
  │         Training Modes              │
  ├─────────────────────────────────────┤
  │ 1. Causal LM Pre-training          │
  │    - 目标: 下一个 token 预测        │
  │    - 损失: CrossEntropyLoss        │
  │    - 数据: 无标注纯文本             │
  ├─────────────────────────────────────┤
  │ 2. Causal LM Continue Pre-training │
  │    - 目标: 领域适应                │
  │    - 损失: CrossEntropyLoss        │
  │    - 数据: 领域特定文本             │
  │    - 初始化: 从 checkpoint 加载     │
  ├─────────────────────────────────────┤
  │ 3. SFT (Supervised Fine-Tuning)    │
  │    - 目标: 指令遵循               │
  │    - 损失: CrossEntropyLoss        │
  │    - 数据: (prompt, response) 对   │
  │    - 格式: instruction tuning      │
  └─────────────────────────────────────┘
  ```
- **训练循环**:
  ```rust
  FOR epoch IN 0..total_epochs:
    FOR batch IN dataloader:
      // 前向传播
      logits = model.forward(batch.input_ids)
      loss = cross_entropy_loss(logits, batch.labels)

      // 反向传播
      gradients = autograd.backward(loss)

      // 参数更新
      optimizer.step(gradients)

      // 记录指标
      monitor.record(loss, lr, grad_norm)
    END FOR

    // 验证评估
    val_loss = evaluate(val_dataloader)
    monitor.record_validation(val_loss)

    // Checkpoint 管理
    if should_save_checkpoint():
      checkpoint_manager.save(model, optimizer, epoch, step)

    // 早停检查
    if early_stopping.should_stop(val_loss):
      break
  END FOR
  ```

- **状态管理**:
  - TrainingState（epoch, global_step, best_loss）
  - Pause/Resume 支持
  - Graceful shutdown（信号处理）

#### 5. Loss Functions 损失函数 (`loss.rs`)
- **CrossEntropyLoss**: 核心损失（下一个 token 预测）
- **Label Smoothing**: 正则化技巧（默认 0.0）
- **Ignore Index**: 忽略 padding token 的损失（通常为 -100）
- **Loss Reduction**: mean / sum / none

#### 6. CheckpointManager 检查点管理 (`checkpoint.rs`)
- **保存内容**:
  ```
  checkpoints/
  ├── model_000010/
  │   ├── model_weights.safetensors  # 模型参数（FP32/BF16）
  │   ├── optimizer_state.bin        # AdamW 的 m/v 缓冲区
  │   ├── training_state.json        # epoch, step, config hash
  │   └── metrics_snapshot.json      # 当前最佳指标
  ├── best_model/                    # 符号链接 → 最佳验证 loss
  └── latest_model/                  # 符号链接 → 最近保存
  ```
- **保存策略**:
  - 每 N 步保存一次（默认 500）
  - 保留最佳 K 个模型（按 val_loss 排序，默认 K=3）
  - 保留最近 M 个 checkpoint（默认 M=5）
  - 自动清理过期文件
- **恢复机制**:
  - 加载权重到 MultimodalTransformer
  - 恢复优化器状态（m/v buffers, step_count）
  - 设置 random seed 保证可复现性
  - 返回 TrainingState 继续训练

#### 7. TrainingMonitor 监控系统 (`monitor.rs`)
- **采集指标**:
  | 类别 | 指标 | 说明 |
  |------|------|------|
  | Loss | train_loss, val_loss | 训练/验证损失 |
  | Perplexity | train_ppl, val_ppl | 困惑度 = exp(loss) |
  | Learning Rate | current_lr | 当前学习率（含调度）|
  | Gradient | grad_norm, max_grad | 梯度范数和最大值 |
  | Throughput | tokens/sec, samples/sec | 吞吐量 |
  | Memory | gpu_memory_mb, cpu_memory_mb | 资源占用 |
  | Time | step_time_ms, epoch_time_s | 耗时统计 |

- **输出格式**:
  ```
  [2024-01-15 10:23:45] Epoch 3/10 | Step 156/5000 | LR: 8.5e-05
    Train Loss: 2.3456 (-0.12↓) | Val Loss: 2.4567 (-0.08↓)
    PPL: 10.44 | Grad Norm: 0.8543 | Tokens/sec: 12.5K
    Time: 125.3ms | GPU: 8192MB | ETA: 2h34m
  ```

- **API 接口**:
  - `GET /api/v1/training/status` → 实时状态 JSON
  - `GET /api/v1/training/metrics?last_n=100` → 历史指标

#### 8. TrainingAPI 训练接口 (`api/training.rs`)
```
POST /api/v1/training/start     → 启动训练任务
POST /api/v1/training/pause     → 暂停训练
POST /api/v1/training/resume    → 恢复训练
POST /api/v1/training/stop      → 停止训练（保存 checkpoint）
GET  /api/v1/training/status    → 查询状态
GET  /api/v1/training/config    → 查询当前配置
PUT  /api/v1/training/config    → 动态修改配置（部分支持）
```

### 修改现有代码

#### 9. Model 层改造 (`model/inference/model.rs`)

**现状**: MultimodalTransformer 仅支持推理（forward only）

**修改后**:

```rust
impl MultimodalTransformer {
    // === 现有方法（保持不变）===
    pub fn forward(&self, input_ids: &[usize]) -> InferenceResult<Array2<f32>> { ... }

    // === 新增方法 ===

    /// 前向传播（训练模式，返回中间激活值用于反向传播）
    pub fn forward_train(
        &self,
        input_ids: &[usize],
        labels: Option<&[usize]>,
    ) -> TrainForwardResult {
        // 1. 执行标准 forward
        // 2. 缓存所有中间激活值（attention weights, hidden states）
        // 3. 计算 cross-entropy loss（如果提供 labels）
        // 4. 返回 TrainForwardResult { logits, loss, cache }
    }

    /// 获取可训练参数及其梯度存储
    pub fn trainable_params(&mut self) -> Vec<ParamRef> {
        // 返回所有需要更新的参数的引用
        // 包括: embedding, attention layers, ffn/moe layers, norm, lm_head
    }

    /// 加载训练权重（从 safetensors 或 bin 格式）
    pub fn load_weights(&mut self, path: &Path) -> Result<(), TrainingError> { ... }

    /// 保存当前权重
    pub fn save_weights(&self, path: &Path) -> Result<(), TrainingError> { ... }
}

/// 训练模式的前向传播结果
pub struct TrainForwardResult {
    pub logits: Array2<f32>,           // [seq_len, vocab_size]
    pub loss: Option<f32>,              // 如果提供了 labels
    pub activation_cache: ActivationCache, // 中间值缓存
}

/// 中间激活值缓存（用于反向传播）
pub struct ActivationCache {
    pub hidden_states: Vec<Array2<f32>>,       // 每层的隐藏状态
    pub attention_scores: Vec<Option<Array3<f32>>>, // 注意力分数（可选）
    pub input_embeddings: Array2<f32>,         // 输入嵌入
}
```

## Impact

- Affected specs: 无（全新功能）
- Affected code:
  - `openmini-server/src/rl/` (复用 Tensor 类型定义)
  - `openmini-server/src/model/inference/model.rs` (新增 forward_train 方法)
  - `openmini-server/src/api/` (新增训练 API)
  - `config/server.toml` (新增 [training] 配置段)
  - `Cargo.toml` (可能新增依赖如 safetensors)

---

## ADDED Requirements

### Requirement: Autograd 自动微分

系统 SHALL 提供基于计算图的自动微分能力：

1. **支持的张量操作及梯度**
   | 操作 | 前向 | 反向（梯度） |
   |------|------|-------------|
   | MatMul | Y = A @ B | dA = dY @ B^T, dB = A^T @ dY |
   | Add | Y = A + B | dA = dY, dB = dY |
   | LayerNorm | Y = LN(X) | 复杂（需缓存均值方差） |
   | Softmax | Y = softmax(X) | dY * Y * (1-Y) （对角线简化）|
   | GELU | Y = gelu(X) | dY * gelu'(X) |
   | Embedding Lookup | Y = embedding[idx] | scatter add 到对应位置 |
   | RoPE | Y = rope(X, pos) | 需特殊处理位置编码 |
   | Reshape | Y = reshape(X) | reshape(dY) |
   | Transpose | Y = X^T | dY^T |

2. **计算图表示**
   ```rust
   struct ComputationGraph {
       nodes: Vec<GraphNode>,
       edges: Vec<(NodeId, NodeId)>,
   }

   enum GraphNode {
       Input { name: String, value: Tensor },
       Param { name: String, value: Tensor, grad: Tensor },
       Op {
           op_type: OpType,
           inputs: Vec<NodeId>,
           output: Tensor,
           backward_fn: Box<dyn Fn(&[&Tensor], &Tensor) -> Vec<Tensor>>,
       },
       Loss { value: f32 },
   }
   ```

3. **反向传播流程**
   ```
   输入: loss 标量
   1. 初始化: grad(loss) = 1.0
   2. 拓扑排序节点（后序遍历）
   3. 反向遍历每个 op 节点:
      - 收集输入梯度和输出梯度
      - 调用 backward_fn 计算输入梯度
      - 累加到各输入节点的 .grad
   4. 返回所有 Param 节点的梯度
   ```

#### Scenario: 简单网络梯度计算

- **WHEN** 构建一个 Embedding → Linear → Softmax → CrossEntropyLoss 网络
- **THEN** 调用 `autograd.backward(loss)` 后：
  1. Embedding 权重的梯度形状正确 `[vocab_size, embed_dim]`
  2. Linear 权重梯度形状正确 `[embed_dim, vocab_size]`
  3. 梯度数值通过有限差分法验证（误差 < 1e-4）

### Requirement: DataLoader 数据流水线

系统 SHALL 提供高效的大规模数据处理能力：

1. **配置示例**
   ```toml
   [training.data]
   format = "jsonl"           # text | jsonl | json
   path = "./data/train.jsonl"
   validation_path = "./data/val.jsonl"
   validation_split = 0.1     # 如果没有单独的验证集

   [training.data.preprocessing]
   max_seq_length = 2048
   max_prompt_length = 512    # SFT 用
   max_response_length = 1536 # SFT 用
   pad_token_id = 0
   eos_token_id = 2
   mlm_probability = 0.15     # 可选：MLM 任务

   [training.dataloader]
   batch_size = 8            # 根据显存调整
   num_workers = 4            # 并行数据加载线程
   shuffle = true
   pin_memory = false         # Rust 不需要
   drop_last = true           # 丢弃不完整的最后 batch
   ```

2. **数据格式示例**

   **Causal LM 预训练格式** (jsonl):
   ```json
   {"text": "人工智能是计算机科学的一个分支..."}
   {"text": "机器学习是人工智能的核心技术..."}
   ```

   **SFT 格式** (jsonl):
   ```json
   {
     "instruction": "解释量子计算的基本原理",
     "input": "",
     "output": "量子计算利用量子力学原理..."
   }
   ```

3. **Tokenize 流程**
   ```
   Raw Text
     ↓ Tokenizer.encode()
   Token IDs [1, 234, 567, 890, 2]
     ↓ Truncate to max_seq_length
   Token IDs (padded/truncated)
     ↓ Add special tokens (如果需要)
   Input IDs [1, 234, 567, 890, 2, 0, 0, 0]
     ↓ Labels = Input IDs shifted right
   Labels [-100, 1, 234, 567, 890, 2, 0, 0]
     ↓ (padding positions 使用 -100 忽略)
   Batch: (input_ids, labels, attention_mask)
   ```

#### Scenario: 处理 10GB 文本数据集

- **WHEN** 提供 1000 万行 JSONL 文本数据
- **THEN** 系统：
  1. 流式读取（非全量加载到内存）
  2. 按 batch_size=32 分批，每批返回 `(input_ids: [32, 2048], labels: [32, 2048])`
  3. 每个 epoch 开始时 shuffle（使用 reservoir sampling 或 Fisher-Yates）
  4. 数据加载不阻塞 GPU 计算（异步预取）

### Requirement: Trainer 训练核心

系统 SHALL 提供完整的模型训练能力：

1. **三种训练模式实现**

   **Mode 1: Causal LM Pre-training**
   ```rust
   let trainer = Trainer::new(TrainingConfig {
       mode: TrainingMode::CausalLM,
       model_config: ModelConfig::from_file("config/model.toml")?,
       optimizer: AdamWConfig::default(),
       scheduler: CosineWithWarmupScheduler {
           warmup_steps: 1000,
           total_steps: 100_000,
       },
       ..Default::default()
   });

   trainer.train(dataloader, val_dataloader)?;
   ```

   **Mode 2: Continue Pre-training**
   ```rust
   let mut trainer = Trainer::new(config);
   trainer.load_checkpoint("checkpoints/pretrained_epoch_10")?;
   trainer.train(continue_dataloader, val_dataloader)?;
   ```

   **Mode 3: SFT**
   ```rust
   let trainer = Trainer::new(TrainingConfig {
       mode: TrainingMode::SFT,
       sft_config: Some(SFTConfig {
           prompt_template: "{instruction}\n\n{input}\n\nResponse:",
           response_prefix: "",
           mask_prompt_loss: true,  // 只计算 response 部分的 loss
       }),
       ..
   });
   trainer.train(sft_dataloader, val_dataloader)?;
   ```

2. **单步训练逻辑**
   ```rust
   fn training_step(&mut self, batch: &Batch) -> TrainingMetrics {
       // 1. 前向传播（训练模式）
       let result = self.model.forward_train(&batch.input_ids, Some(&batch.labels));

       // 2. 反向传播
       let gradients = self.autograd.backward(result.loss.unwrap());

       // 3. 梯度裁剪
       let grad_norm = clip_grad_norm_(gradients, self.config.max_grad_norm);

       // 4. 参数更新
       self.optimizer.step(gradients);
       self.optimizer.zero_grad();

       // 5. 学习率调度
       let lr = self.scheduler.get_lr(self.global_step);

       // 6. 返回指标
       TrainingMetrics {
           loss: result.loss.unwrap(),
           grad_norm,
           learning_rate: lr,
           throughput: batch.num_tokens as f64 / step_time_secs,
       }
   }
   ```

3. **早停机制**
   ```rust
   struct EarlyStopping {
       patience: usize,           // 默认 5 个 epoch
       min_delta: f64,            // 最小改善阈值 (0.001)
       counter: usize,            // 连续未改善计数
       best_loss: f64,            // 最佳验证损失
   }

   impl EarlyStopping {
       fn should_stop(&mut self, val_loss: f64) -> bool {
           if val_loss < self.best_loss - self.min_delta {
               self.best_loss = val_loss;
               self.counter = 0;  // 重置计数器
               false
           } else {
               self.counter += 1;
               self.counter >= self.patience
           }
       }
   }
   ```

#### Scenario: 完成 3 epoch 预训练

- **WHEN** 配置 epochs=3, batch_size=4, data=1000 条文本
- **THEN** 系统：
  1. 每个 epoch 遍历全部数据 250 个 steps (1000/4)
  2. 总计执行 750 个训练步骤
  3. 每 step 输出 loss 和 ppl
  4. 每 epoch 结束输出验证 loss
  5. 自动保存最佳模型 checkpoint
  6. 最终输出训练总结（总耗时、最终 loss、最佳 loss）

### Requirement: CheckpointManager 持久化

系统 SHALL 提供可靠的模型状态持久化能力：

1. **权重格式支持**
   - **Safetensors**（推荐）: 安全快速，支持 lazy loading
   - **Bincode**（备选）: Rust 原生序列化
   - **NumPy .npy**（兼容性）: 可与其他框架互操作

2. **保存频率策略**
   ```toml
   [training.checkpoint]
   save_strategy = "steps"  # steps | epoch | best
   save_steps = 500         # 每 500 步保存
   save_total_limit = 5     # 最多保留 5 个
   save_best_only = false   # 是否只保留最佳
   resume_from_checkpoint = ""  # 断点续训路径
   ```

3. **Checkpoint 内容**
   ```python
   # checkpoint 结构伪代码
   checkpoint = {
       "model_state_dict": {
           "embedding.weight": Tensor[vocab_size, hidden_size],
           "layers.0.self_attn.q_proj.weight": Tensor[...],
           ...
       },
       "optimizer_state_dict": {
           "state": [{
               "step": 12345,
               "exp_avg": Tensor[...],   # AdamW m buffer
               "exp_avg_sq": Tensor[...]  # AdamW v buffer
           }],
           "param_groups": [{"lr": 1e-4, "weight_decay": 0.01}]
       },
       "training_state": {
           "epoch": 5,
           "global_step": 2500,
           "best_val_loss": 2.3456,
           "config_hash": "sha256:...",
           "random_state": {...},
           "rng_state": {...},
       },
       "metrics_history": [...],
       "created_at": "2024-01-15T10:23:45Z",
       "version": "1.0.0",
   }
   ```

#### Scenario: 断点续训验证

- **WHEN** 训练在 epoch=5, step=1234 时崩溃
- **THEN** 从 checkpoint 恢复后：
  1. 模型权重完全一致（checksum 验证通过）
  2. AdamW 的 m/v buffers 恢复正确
  3. global_step 从 1235 继续
  4. 数据加载器的 shuffle state 一致
  5. 验证 loss 曲线连续无断层

---

## MODIFIED Requirements

### Requirement: MultimodalTransformer 扩展

**现状**: 仅支持 `forward()` 推理方法

**修改后**: SHALL 支持训练所需的所有接口

详见上方"Model 层改造"章节。

### Requirement: GRPOConfig / TrainingConfig 合并

**现状**: GRPOConfig 仅包含 RL 相关超参数

**修改后**: 引入独立的 `TrainingConfig`，包含通用训练超参数

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    // === 模型相关 ===
    pub model_name_or_path: String,  // 模型路径或名称
    pub dtype: DataType,             // fp32 | bf16 | fp16

    // === 训练超参数 ===
    pub num_train_epochs: usize,
    pub max_steps: Option<usize>,    // 优先于 epochs
    pub per_device_train_batch_size: usize,
    pub gradient_accumulation_steps: usize,  // 模拟更大 batch
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub adam_beta1: f64,
    pub adam_beta2: f64,
    pub adam_epsilon: f64,
    pub max_grad_norm: f64,

    // === 调度器 ===
    pub lr_scheduler_type: LrSchedulerType,  // linear | cosine | constant
    pub warmup_ratio: f64,                   // 或 warmup_steps
    pub warmup_steps: Option<usize>,

    // === 数据相关 ===
    pub max_seq_length: usize,
    pub label_smoothing_factor: f64,

    // === Checkpoint ===
    pub output_dir: String,
    pub save_strategy: SaveStrategy,
    pub save_steps: usize,
    pub save_total_limit: usize,
    pub resume_from_checkpoint: Option<String>,

    // === 日志 ===
    pub logging_dir: String,
    pub logging_steps: usize,

    // === 早停 ===
    pub early_stopping_patience: usize,
    pub early_stopping_threshold: f64,

    // === 混合精度（可选）===
    pub fp16: bool,
    pub bf16: bool,

    // === SFT 专用 ===
    pub sft_config: Option<SFTConfig>,
}
```

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                            │
│                                                                  │
│  ┌────────────┐    ┌─────────────┐    ┌────────────────────┐   │
│  │ DataLoader │───▶│   Trainer    │───▶│ ParameterUpdater   │   │
│  │            │    │             │    │ (AdamW/SGD)        │   │
│  │- load()    │    │- train()    │    │                    │   │
│  │- tokenize  │    │- validate() │    │- step()            │   │
│  │- batch()   │    │- save()     │    │- zero_grad()      │   │
│  └────────────┘    └──────┬──────┘    └────────▲───────────┘   │
│                          │                      │               │
│                          ▼                      │               │
│  ┌────────────┐    ┌─────────────┐             │               │
│  │ Autograd   │◀───│   Model     │─────────────┘               │
│  │ Engine     │    │             │                             │
│  │            │    │Multimodal   │                             │
│  │- forward() │    │Transformer  │                             │
│  │- backward()│    │             │                             │
│  │- graph     │    │- forward()  │                             │
│  └──────┬─────┘    │- forward_train()                         │
│         │          │- parameters()│                            │
│         │          └─────────────┘                             │
│         ▼                                                       │
│  ┌────────────┐    ┌────────────────┐    ┌─────────────────┐   │
│  │ Loss Fn    │    │ CheckpointMgr  │    │ TrainingMonitor  │   │
│  │            │    │                │    │                 │   │
│  │- CE Loss   │    │- save()        │    │- record()       │   │
│  │- smoothing │    │- load()        │    │- get_status()   │   │
│  └────────────┘    │- cleanup()     │    │- emit_log()     │   │
│                    └────────────────┘    └────────┬────────┘   │
│                                                  │             │
│                                          ┌───────▼─────────┐  │
│                                          │  TrainingAPI     │  │
│                                          │                 │  │
│                                          │ POST /start     │  │
│                                          │ POST /pause     │  │
│                                          │ POST /resume    │  │
│                                          │ POST /stop      │  │
│                                          │ GET  /status    │  │
│                                          └─────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Configuration Example (server.toml)

```toml
# ============================================
# 大模型训练配置
# ============================================
[training]
enable = true

# --- 训练模式 ---
mode = "causal_lm"  # causal_lm | continue_pretrain | sft

# --- 模型配置 ---
[training.model]
name_or_path = "./models/base_model"  # 预训练模型路径（继续预训练/SFT 时必填）
dtype = "fp32"                        # fp32 | bf16 | fp16

# --- 训练超参数 ---
[training.hyperparams]
num_train_epochs = 10
per_device_train_batch_size = 8
gradient_accumulation_steps = 4       # 有效 batch = 8 * 4 = 32
learning_rate = 1e-4
weight_decay = 0.01
max_grad_norm = 1.0
max_seq_length = 2048
label_smoothing = 0.0

# --- 优化器 ---
[training.optimizer]
type = "adamw"                        # adamw | sgd
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# --- 学习率调度 ---
[training.scheduler]
type = "cosine"                       # cosine | linear | constant
warmup_ratio = 0.1                    # 前 10% steps 线性 warmup
# warmup_steps = 1000                  # 或指定绝对步数

# --- 数据配置 ---
[training.data]
train_path = "./data/train.jsonl"
validation_path = "./data/val.jsonl"
format = "jsonl"                      # jsonl | text
validation_split = 0.1                 # 无验证集时自动拆分
shuffle = true
num_workers = 4

# --- Checkpoint ---
[training.checkpoint]
output_dir = "./checkpoints"
save_strategy = "steps"               # steps | epoch | best
save_steps = 500
save_total_limit = 5                  # 最多保留 5 个
resume_from_checkpoint = ""           # 断点续训路径

# --- 日志与监控 ---
[training.logging]
log_every_n_steps = 10
logging_dir = "./logs"
save_metrics_history = true

# --- 早停 ---
[training.early_stopping]
enabled = true
patience = 5                          # 连续 5 个 epoch 不改善则停止
min_delta = 0.001

# --- SFT 专用配置（仅 mode=sft 时生效）---
[training.sft]
prompt_template = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
mask_prompt_loss = true               # 只计算 response 部分 loss
max_prompt_length = 512
max_response_length = 1536
```
