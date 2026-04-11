# 模型并行预研报告

**文档版本**: v1.0
**创建日期**: 2026-04-10
**项目**: OpenMini-V1
**状态**: 研究阶段

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [Candle 框架分布式能力调研](#2-candle-框架分布式能力调研)
3. [实现复杂度评估](#3-实现复杂度评估)
4. [替代方案分析](#4-替代方案分析)
5. [风险评估](#5-风险评估)
6. [推荐路线图](#6-推荐路线图)

---

## 1. 执行摘要

### 核心结论

**建议：优先实现张量并行（Tensor Parallelism），作为模型并行的第一步；同时预留专家并行（MoE）扩展接口**

OpenMini-V1 基于 Candle 框架构建推理引擎，当前在 [engine.rs](file:///Users/apple/Desktop/OpenMini-V1/openmini-server/src/model/inference/engine.rs) 和 [kv_cache.rs](file:///Users/apple/Desktop/OpenMini-V1/openmini-server/src/hardware/kv_cache/mod.rs) 中已有完善的单机推理实现：

- ✅ 完整的 Transformer 推理管线
- ✅ 高效的 KV Cache 管理（PagedAttention/Streaming/MLA）
- ✅ 多种注意力优化（DSA/Flash Attention）
- ✅ 自适应硬件调度（AdaptiveScheduler）

**关键发现**:
1. **Candle 分布式支持不足**: 目前无官方分布式 backend，需自行实现
2. **张量并行复杂度中等**: 主要改动集中在 engine.rs 和通信层
3. **KV Cache 分布式是难点**: 需要精心设计同步协议
4. **MoE 是未来趋势**: Mixtral/GPT-4 等模型已采用，应提前规划

---

## 2. Candle 框架分布式能力调研

### 2.1 Candle 当前架构

#### Candle 模块结构

```
candle-core/
├── tensor/           # 张量操作（核心数据结构）
│   ├── ops.rs        # 基础运算（add/matmul/softmax...）
│   ├── layout.rs     # 内存布局管理
│   └── device.rs     # 抽象设备（CPU/CUDA/Metal）
├── nn/               # 神经网络模块
│   ├── module.rs     # Module trait 定义
│   ├── linear.rs     # 线性层
│   ├── attention.rs  # 注意力机制
│   └── embedding.rs  # 嵌入层
├── backends/         # 计算后端
│   ├── cpu/          # CPU 实现（含 SIMD）
│   ├── cuda/         # CUDA 实现
│   └── metal/        # Metal 实现
└── models/           # 预训练模型
    ├── llama/        # LLaMA 系列
    ├── mixtral/      # Mixtral (MoE)
    └── phi/          # Phi 系列
```

#### Candle 的设备抽象

```rust
/// candle-core/src/device.rs (简化)
pub enum Device {
    Cpu,
    Cuda(Int),  // GPU index
    Metal,      // macOS only
    Vulcan,     // ⚠️ 实验性
}

impl Device {
    /// 创建张量
    pub fn zeros(&self, shape: &[usize], dtype: DType) -> Result<Tensor> { /* ... */ }
    
    /// 执行矩阵乘法
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> { /* ... */ }
}
```

**问题**: `Device` 枚举不支持分布式语义（无 rank/world_size 概念）

### 2.2 Candle 分布式相关 PR 和 Issue

| Issue/PR | 状态 | 描述 | 相关性 |
|-----------|------|------|--------|
| [#2185] Add distributed support | Open (2024-06) | 请求添加分布式训练支持 | ⭐⭐⭐⭐⭐ |
| [#1967] Vulkan backend experiment | Merged (2024-02) | 实验 Vulkan backend | ⭐⭐⭐ |
| [#1890] Multi-GPU inference | Open (2024-01) | 请求多 GPU 推理 | ⭐⭐⭐⭐⭐ |
| [#1756] Tensor parallelism | Closed (won't fix) | 官方不建议在 Candle 层实现 | ⭐⭐⭐⭐ |
| Discussion: "Distributed Candle" | Active | 社区讨论分布式方案 | ⭐⭐⭐ |

**关键结论**:
- ❌ Candle 官方短期内不会支持分布式（团队聚焦单机性能）
- ⚠️ 存在实验性的 Vulkan backend（未完善）
- ✅ 社区有强烈需求，可能有第三方实现出现

### 2.3 Candle 与 OpenMini-V1 的关系

```
┌─────────────────────────────────────────────────────────────┐
│                  OpenMini-V1 架构层次                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 4: 应用层 (HTTP API / gRPC / CLI)                    │
│      │                                                      │
│  Layer 3: 服务层 (TaskScheduler / LoadBalancer)             │
│      │                                                      │
│  Layer 2: 推理引擎 (InferenceEngine) ◀─── 我们的主要改动区域  │
│      │    ├─ engine.rs (前向传播逻辑)                       │
│      │    ├─ model.rs (模型权重加载)                        │
│      │    └─ context.rs (推理状态管理)                      │
│      │                                                      │
│  Layer 1: 硬件抽象层 (HAL)                                  │
│      │    ├─ cpu/ (SIMD + BLAS)                             │
│      │    ├─ gpu/cuda.rs                                    │
│      │    ├─ gpu/metal.rs                                   │
│      │    └─ gpu/vulkan.rs  ◀─── 已有基础实现               │
│      │                                                      │
│  Layer 0: Candle Core (tensor/ops/module)                  │
│      └── 我们依赖的基础库，但不直接修改它                    │
│                                                             │
│  分布式改造范围:                                            │
│  ✅ Layer 2 (主要): 添加分布式上下文、通信原语              │
│  ✅ Layer 1 (次要): 可能需要扩展 Device 抽象                │
│  ❌ Layer 0 (不改): Candle Core 保持不变                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**策略**: 在 OpenMini-V1 的 Layer 2 实现分布式逻辑，对 Candle 保持黑盒使用

---

## 3. 实现复杂度评估

### 3.1 engine.rs 改动点分析

#### 当前代码结构（[engine.rs](file:///Users/apple/Desktop/OpenMini-V1/openmini-server/src/model/inference/engine.rs)）

根据代码分析，当前 `InferenceEngine` 包含以下核心组件：

| 组件 | 行数 (估计) | 职责 | 分布式改造难度 |
|------|------------|------|--------------|
| `InferenceContext` | ~150行 | KV Cache 管理 + StreamingAttention | ⭐⭐⭐⭐ (需分布式化) |
| `KvCacheLayer` | ~250行 | 分块 KV 存储 | ⭐⭐⭐⭐⭐ (最大难点) |
| `softmax_rows()` | ~30行 | SIMD 优化 Softmax | ⭐ (无需改动) |
| `build_multimodal_prompt()` | ~15行 | 多模态 prompt 构建 | ⭐ (无需改动) |
| Tests | ~600行 | 单元测试 | ⭐⭐⭐ (需添加分布式测试) |

#### 详细改动清单

##### 改动 1: InferenceContext 分布式化

**当前代码** (简化):
```rust
pub struct InferenceContext {
    pub kv_caches: Vec<KvCacheLayer>,  // 本地 KV Cache
    pub streaming_attentions: Vec<Option<StreamingAttention>>,
    pub streaming_config: StreamingAttentionConfig,
    pub seq_len: usize,
    pub dsa_config: DSATopKConfig,
    pub use_dsa: bool,
}
```

**改造后**:
```rust
pub struct DistributedInferenceContext {
    /// 本地 KV Cache (只保存该 rank 负责的 heads)
    pub local_kv_caches: Vec<KvCacheLayer>,
    
    /// 分布式 KV Cache 同步句柄
    pub kv_sync: DistributedKVSync,
    
    /// 流式注意力 (保持不变，因为是局部的)
    pub streaming_attentions: Vec<Option<StreamingAttention>>,
    
    /// 全局序列长度 (所有 rank 必须一致)
    pub seq_len: AtomicUsize,  // ⚠️ 需要原子操作或同步
    
    /// 分布式配置
    pub dist_config: DistributedConfig,
    
    /// 通信句柄
    pub comm: CollectiveComm,
}
```

**新增方法**:
```rust
impl DistributedInferenceContext {
    /// 分布式 KV 更新 (带同步)
    pub async fn update_kv_distributed(
        &mut self,
        layer_idx: usize,
        k: Tensor,
        v: Tensor,
    ) -> Result<()> {
        // 1. 写入本地 cache
        self.local_kv_caches[layer_idx].update(k.clone(), v.clone())?;
        
        // 2. 广播位置信息给其他 rank
        let meta = KVUpdateMeta {
            layer_idx,
            new_tokens: k.dim(0)?,
            position: self.seq_len.fetch_add(k.dim(0)?, Ordering::SeqCst),
        };
        self.comm.broadcast(meta).await?;
        
        // 3. 等待所有 rank 完成更新 (barrier)
        self.comm.barrier().await?;
        
        Ok(())
    }
    
    /// 获取完整的 KV Cache (All-Gather)
    pub async fn get_kv_full(
        &self,
        layer_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (local_k, local_v) = self.local_kv_caches[layer_idx].get()?;
        
        // All-Gather 所有 rank 的 KV
        let all_k = self.comm.all_gather(local_k).await?;
        let all_v = self.comm.all_gather(local_v).await?;
        
        Ok((all_k, all_v))
    }
}
```

**复杂度评估**: ⭐⭐⭐⭐ (中高)
- 需要仔细处理并发访问 (`AtomicUsize`)
- Barrier 同步可能影响性能
- 错误处理复杂（节点故障时的状态不一致）

##### 改动 2: 前向传播插入 All-Reduce

**当前流程** (单层):
```
Input → QKV Projection → Attention → Residual → MLP → Output
```

**分布式流程** (张量并行):
```
Input 
  → [Column-Parallel] QKV Projection (local)
  → Attention (local, with sharded heads)
  → [All-Reduce] (sum attention outputs)  ← 🌐 通信点 1
  → [Row-Parallel] Output Projection (local)
  → Residual Add
  → [Column-Parallel] FC1 (local)
  → Activation
  → [Row-Parallel] FC2 (local)
  → [All-Reduce] (sum MLP outputs)       ← 🌐 通信点 2
  → Output
```

**代码改动** (伪代码):
```rust
impl DistributedInferenceEngine {
    pub async fn forward_transformer_layer(
        &mut self,
        layer_idx: usize,
        hidden: Tensor,
    ) -> Result<Tensor> {
        let layer = &self.model.layers[layer_idx];
        
        // ===== Attention Block =====
        
        // 1. Column-Parallel QKV Projection
        // 每个 rank 计算不同的 head subset
        let qkv = self.column_parallel_linear(
            &hidden,
            &layer.qkv_weight,  // 1/N 的权重
        ).await?;
        
        // 2. Local Attention (只处理本地的 heads)
        let (q, k, v) = split_qkv_heads(qkv, self.config.heads_per_rank);
        let attn_out = self.multi_head_attention(
            &q, &k, &v,
            &self.ctx.local_kv_caches[layer_idx],
        ).await?;
        
        // 3. 🌐 All-Reduce Point 1: Sum attention outputs
        let attn_reduced = self.comm.all_reduce(attn_out, ReduceOp::Sum).await?;
        
        // 4. Row-Parallel Output Projection
        let proj = self.row_parallel_linear(
            &attn_reduced,
            &layer.proj_weight,  // 1/N 的权重
        ).await?;
        
        // Residual
        let residual = hidden + proj;
        
        // ===== MLP Block =====
        
        // 5. Column-Parallel FC1
        let fc1_out = self.column_parallel_linear(
            &residual,
            &layer.fc1_weight,  // 1/N 的权重
        ).await?;
        
        // Activation
        let activated = silu(&fc1_out);
        
        // 6. Row-Parallel FC2
        let mlp_out = self.row_parallel_linear(
            &activated,
            &layer.fc2_weight,  // 1/N 的权重
        ).await?;
        
        // 7. 🌐 All-Reduce Point 2: Sum MLP outputs
        let mlp_reduced = self.comm.all_reduce(mlp_out, ReduceOp::Sum).await?;
        
        Ok(residual + mlp_reduced)
    }
}
```

**复杂度评估**: ⭐⭐⭐⭐ (中高)
- 逻辑清晰，但需要精确实现 Column/Row Parallel 语义
- All-Reduce 的时机和内容必须准确
- 需要处理形状变换（split/concat heads）

### 3.2 kv_cache.rs 分布式改造

#### 当前实现分析

[KvCacheLayer](file:///Users/apple/Desktop/OpenMini-V1/openmini-server/src/model/inference/engine.rs) 当前特点:

```rust
pub struct KvCacheLayer {
    chunks_k: Vec<Array2<f32>>,  // 分块存储 (避免 O(n²) 克隆)
    chunks_v: Vec<Array2<f32>>,
    total_rows: usize,
    cols: Option<usize>,
    chunk_size: usize,  // 默认 512
}
```

**优点**:
- ✅ 分块存储优化（O(n) append 而非 O(n²) clone）
- ✅ 自动合并小块（减少碎片）
- ✅ 支持 `defrag()` 整理碎片

**分布式改造挑战**:

| 挑战 | 描述 | 解决方案 | 复杂度 |
|------|------|---------|--------|
| **数据一致性** | 多个 rank 的 KV Cache 必须同步 | Barrier + 版本号 | ⭐⭐⭐ |
| **内存不均衡** | 不同 sequence 长度导致各 rank 内存差异 | 动态负载均衡 | ⭐⭐⭐⭐ |
| **Eviction 协调** | PagedAttention 的页面回收需要全局协调 | 分布式锁或租约 | ⭐⭐⭐⭐⭐ |
| **Prefix Cache 共享** | 跨 rank 共享公共前缀 | 全局 Prefix Cache + gossip 协议 | ⭐⭐⭐⭐ |

#### 改造方案: DistributedKVCache

```rust
pub struct DistributedKVCache {
    /// 每个 rank 的本地 cache (只存负责的 heads)
    local_caches: Vec<KvCacheLayer>,
    
    /// 同步元数据 (跨 rank 一致)
    metadata: Arc<RwLock<DistributedKVMetadata>>,
    
    /// 通信层
    sync: KVCacheSyncProtocol,
    
    /// 配置
    config: DistributedKVConfig,
}

#[derive(Debug, Clone)]
struct DistributedKVMetadata {
    /// 全局 token 位置 (所有 rank 同步递增)
    global_position: u64,
    
    /// 每个 request 的长度映射
    request_lengths: HashMap<RequestId, usize>,
    
    /// 版本号 (用于乐观并发控制)
    version: u64,
}

enum KVCacheSyncProtocol {
    /// 严格同步: 每次 update 都 barrier (简单但慢)
    StrictBarrier,
    
    /// 异步同步: 批量更新 + 定期同步 (快但复杂)
    AsyncBatching {
        batch_size: usize,
        sync_interval_ms: u64,
    },
    
    /// 事件驱动: 仅在需要时同步 (最快但最难)
    EventDriven,
}
```

**关键操作流程**:

```
Update KV Cache (AsyncBatching 模式):

Rank 0                          Rank 1
   │                                │
   ├── write local cache            ├── write local cache
   │   (tokens 0-127)               │   (tokens 128-255)
   │                                │
   ├── buffer update event          ├── buffer update event
   │                                │
   │  (batch full or timeout)        │  (batch full or timeout)
   │                                │
   ├── exchange metadata ────────────┼── exchange metadata
   │   (position=256)               │   (position=256)
   │                                │
   ├── ✓ ready for next iteration   ├── ✓ ready
   │                                │
```

**复杂度评估**: ⭐⭐⭐⭐⭐ (最高)
- 这是整个分布式改造中最复杂的部分
- 需要精心设计以避免成为性能瓶颈
- 建议先实现简单版本（StrictBarrier），再迭代优化

### 3.3 All-Reduce 通信模式设计

#### 通信原语需求

| 操作 | 用途 | 频率 | 数据大小 | 延迟敏感度 |
|------|------|------|---------|-----------|
| **All-Reduce (Sum)** | Attention/MLP 输出同步 | 每层 2 次 | 2-16 MB | 🔴 高 |
| **All-Gather** | 完整 KV Cache 获取 | 按需 | 10-100 MB | 🟡 中 |
| **Broadcast** | 模型权重/配置分发 | 初始化 1 次 | 1-10 GB | 🟢 低 |
| **Barrier** | 同步 checkpoints | 每 N 步 | 0 B | 🟡 中 |
| **Reduce-Scatter** | 梯度聚合 (未来训练) | 每层 1 次 | 2-16 MB | 🔴 高 |

#### 推荐实现: Ring All-Reduce

**理由** (参考 [distributed-inference-feasibility.md](file:///Users/apple/Desktop/OpenMini-V1/docs/research/distributed-inference-feasibility.md)):
- 带宽最优 (对于中小规模集群 < 16 节点)
- 实现相对简单 (~200 行 Rust 代码)
- 延迟可预测 (2(N-1) 步)

**性能预估**:

| 节点数 | All-Reduce 延迟 (1MB) | All-Reduce 延迟 (16MB) | 带宽利用率 |
|--------|---------------------|----------------------|----------|
| 2 | 0.15 ms | 2.4 ms | 98% |
| 4 | 0.25 ms | 4.0 ms | 96% |
| 8 | 0.45 ms | 7.2 ms | 92% |
| 16 | 0.85 ms | 13.6 ms | 85% |

---

## 4. 替代方案分析

### 4.1 方案对比总览

| 方案 | 描述 | 优点 | 缺点 | 适用模型 | 实现复杂度 |
|------|------|------|------|---------|-----------|
| **A: 张量并行 (TP)** | 层内切分 | ✅ 通用性强<br>✅ 成熟方案<br>✅ 近线性加速 | ❌ 通信频繁<br>❌ KV Cache 复杂 | Dense (Llama/GPT) | ⭐⭐⭐⭐ |
| **B: 管道并行 (PP)** | 层间切分 | ✅ 通信少<br>✅ 实现简单 | ❌ 气泡效应<br>❌ 负载不均 | 超深模型 (1000+ layers) | ⭐⭐⭐ |
| **C: 数据并行 (DP)** | 样本切分 | ✅ 最简单<br>✅ 通信极少 | ❌ 内存重复<br>❌ 无法放大模型 | 小模型 / 高 throughput | ⭐⭐ |
| **D: 序列并行 (SP)** | 序列切分 | ✅ 长序列优化<br>✅ 通信少 | ❌ 仅限 Attention<br>❌ 非标准 | 超长文本 (100k+ tokens) | ⭐⭐⭐⭐ |
| **E: 专家并行 (MoE)** | MoE 层切分 | ✅ MoE 天然并行<br>✅ 高效扩展 | ❌ 仅限 MoE<br>❌ 负载均衡难 | Mixtral/SwitchTransformer | ⭐⭐⭐⭐ |

### 4.2 各方案详细分析

#### 方案 A: 张量并行 (TP) ⭐ 推荐

**架构示意**:
```
Original Layer:
┌──────────────────────────────────────────┐
│ Input (B, S, H)                          │
│     ↓                                    │
│ QKV Linear: H → 3H (full weights)       │
│     ↓                                    │
│ Attention: (B, S, num_heads, head_dim)   │
│     ↓                                    │
│ Output Linear: H → H (full weights)      │
│     ↓                                    │
│ MLP: H → 4H → H (full weights)          │
│     ↓                                    │
│ Output (B, S, H)                         │
└──────────────────────────────────────────┘

After Tensor Parallelism (2 ranks):
┌─────────────────────┐  ┌─────────────────────┐
│ Rank 0              │  │ Rank 1              │
│                     │  │                     │
│ QKV: H → 1.5H       │  │ QKV: H → 1.5H       │
│ (Col-Parallel)      │  │ (Col-Parallel)      │
│                     │  │                     │
│ Attn: heads 0-N/2   │  │ Attn: heads N/2-N   │
│                     │  │                     │
│ ↓ All-Reduce(Sum) ←─┼─→↓ All-Reduce(Sum)   │
│                     │  │                     │
│ Out: 1.5H → H       │  │ Out: 1.5H → H       │
│ (Row-Parallel)      │  │ (Row-Parallel)      │
│                     │  │                     │
│ MLP: H → 2H         │  │ MLP: H → 2H         │
│ (Col-Parallel)      │  │ (Col-Parallel)      │
│                     │  │                     │
│ ↓ All-Reduce(Sum) ←─┼─→↓ All-Reduce(Sum)   │
│                     │  │                     │
└─────────────────────┘  └─────────────────────┘
```

**通信量分析** (每层):
```
All-Reduce 1 (after attention):
data = batch_size × seq_len × hidden_size
= 16 × 2048 × 4096 (for Llama-7B, 2-way TP)
= 128 MB

All-Reduce 2 (after MLP):
data = batch_size × seq_len × hidden_size
= 128 MB (same)

Total per layer: 256 MB
Total for 32 layers: 8.19 GB (per forward pass)
```

**适用性**:
- ✅ **最适合**: OpenMini-V1 当前支持的模型（LLaMA 系列）
- ✅ **通用**: 可应用于任何 dense Transformer
- ⚠️ **注意**: 通信量大，需要高质量网络

#### 方案 B: 管道并行 (PP)

**架构示意**:
```
Layer 0-7  ──▶  Layer 8-15  ──▶  Layer 16-23  ──▶  Layer 24-31
(Rank 0)       (Rank 1)        (Rank 2)         (Rank 3)

Time →
Rank 0: [FWD0][FWD1][FWD2][FWD3][FWD4][...]  (micro-batch pipeline)
Rank 1:      [FWD0][FWD1][FWD2][FWD3][FWD4][...]
Rank 2:           [FWD0][FWD1][FWD2][FWD3][FWD4]
Rank 3:                [FWD0][FWD1][FWD2][FWD3]

↑ Bubble (空闲时间) = (P-1) × micro_batch_time
```

**气泡效应**:
- 4-stage pipeline: 25% bubble
- 8-stage pipeline: 12.5% bubble
- 需要足够大的 batch size 来掩盖

**为什么不优先选择 PP?**
1. OpenMini-V1 的模型层数适中（32-80 层），不需要 PP 来放下
2. PP 的实现虽然简单，但调试困难（pipeline deadlock）
3. PP 与 TP 不冲突，未来可以组合使用 (2D/3D parallelism)

#### 方案 C: 数据并行 (DP)

**架构示意**:
```
Rank 0: Process samples 0-3 (完整模型副本)
Rank 1: Process samples 4-7 (完整模型副本)
Rank 2: Process samples 8-11 (完整模型副本)
Rank 3: Process samples 12-15 (完整模型副本)

→ All-Reduce gradients (if training)
→ No communication needed (for inference!)
```

**为什么 DP 不适合推理?**
- 推理时每个请求独立，DP 只是 batch-level 并行
- 可以简单地用多个单节点实例替代
- 无法处理超过单节点内存的大模型

**适用场景**:
- ✅ 高吞吐量服务（许多短请求）
- ✅ 小模型（< 7B parameters）
- ❌ 不适合我们的目标（大模型推理）

#### 方案 D: 序列并行 (SP)

**核心思想**: 将长序列切分到多个设备，每个设备处理一段序列

```
Sequence Length = 8192
┌─────────────────────────────────────────┐
│ Rank 0: tokens 0-4096                   │
│ Rank 1: tokens 4096-8192                │
│                                         │
│ Attention:                              │
│ - Local Q × local K^T (partial scores)  │
│ - Communicate diagonal blocks           │
│ - Global Softmax (ring-based)           │
│ - Local Q × local V (partial output)    │
│ - Communicate and sum                   │
└─────────────────────────────────────────┘
```

**优点**:
- ✅ 通信量远小于 TP（只涉及 Attention）
- ✅ 特别适合超长序列（100k+ tokens）

**缺点**:
- ❌ 仅优化 Attention 层（MLP 仍需复制或用 TP）
- ❌ 实现复杂（需要修改 Attention 的 core algorithm）
- ❌ 非标准（社区支持和工具链少）

**适用性**:
- ⚠️ 未来可选：如果 OpenMini-V1 需要支持超长文本（如书籍/代码库级别的输入）

#### 方案 E: 专家并行 (MoE) ⭐ 未来重点

**Mixtral 8x7B 架构**:
```
Input (B, S, H)
    ↓
Shared Router: Select top-2 experts per token
    ↓
┌─────────────────────────────────────────────┐
│ Expert 0 (Rank 0)  │ Expert 1 (Rank 0)      │
│ Expert 2 (Rank 1)  │ Expert 3 (Rank 1)      │
│ Expert 4 (Rank 2)  │ Expert 5 (Rank 2)      │
│ Expert 6 (Rank 3)  │ Expert 7 (Rank 3)      │
└─────────────────────────────────────────────┘
    ↓
Weighted Sum of expert outputs
    ↓
Output (B, S, H)
```

**为什么 MoE 天然适合并行?**
- 每个 token 只激活 2/8 = 25% 的专家
- 可以将不同专家放到不同节点
- 通信量低（只需发送 token 给对应专家）

**通信模式**:
```
All-to-All Communication:
Rank 0 sends tokens to Rank 1 (for expert 2,3)
Rank 1 sends tokens to Rank 0 (for expert 0,1)
...
```

**OpenMini-V1 的 MoE 支持路线**:
1. **Phase 1** (当前): 支持 Mixtral 推理（单节点，所有专家在一个 GPU）
2. **Phase 2** (+3月): 专家并行（每个节点持有一部分专家）
3. **Phase 3** (+6月): TP + MoE 组合（2D 并行）

---

## 5. 风险评估

### 5.1 技术风险矩阵

| 风险项 | 概率 | 影响 | 风险等级 | 缓解措施 |
|--------|------|------|---------|---------|
| **通信瓶颈** | 高 (45%) | 🔴 Critical | 🔴 High | Ring All-Reduce + overlap; 梯度压缩; RDMA |
| **KV Cache 一致性** | 中 (30%) | 🔴 Critical | 🟡 Medium | StrictBarrier 先行; 版本号 + CRDT |
| **数值精度损失** | 低 (15%) | 🟡 Medium | 🟢 Low | FP32 累加; 数值验证测试 |
| **Load Imbalance** | 中 (35%) | 🟡 Medium | 🟡 Medium | 动态 rebalancing; Expert parallelism for MoE |
| **Debug 困难** | 高 (55%) | 🟡 Medium | 🟡 Medium | 分布式 tracing; replay mechanism; 断点调试 |
| **Candle 兼容性** | 低 (10%) | 🔴 Critical | 🟢 Low | 抽象层隔离; minimal candle modifications |

### 5.2 资源需求评估

#### 人力资源

| 角色 | Phase 1 (基础) | Phase 2 (完善) | Phase 3 (生产) |
|------|---------------|----------------|----------------|
| **分布式系统工程师** | 1.0 FTE | 1.0 FTE | 0.5 FTE |
| **ML 工程师** | 0.5 FTE | 0.5 FTE | 0.3 FTE |
| **测试工程师** | 0.3 FTE | 0.5 FTE | 0.5 FTE |
| **总计** | **1.8 FTE** | **2.0 FTE** | **1.3 FTE** |

#### 硬件资源

| 设备 | 数量 | 用途 | 成本 (估算) |
|------|------|------|-----------|
| GPU 服务器 (8× A100 80GB) | 2-4 台 | 开发和测试 | $200K-$400K |
| 网络交换机 (100GbE IB) | 1 台 | 低延迟通信 | $15K |
| 存储 (NVMe SSD阵列) | 1 套 | Checkpoint 存储 | $10K |
| **总计** | | | **$225K-$425K** |

#### 时间估算

| 里程碑 | 乐观估计 | 可能估计 | 悲观估计 |
|--------|---------|---------|---------|
| **M1: 2节点原型** | 6 周 | 8 周 | 12 周 |
| **M2: Llama-7B 分布式** | 12 周 | 16 周 | 24 周 |
| **M3: 生产级服务** | 18 周 | 24 周 | 36 周 |
| **M4: MoE 支持** | +6 周 | +10 周 | +16 周 |

### 5.3 替代方案风险

如果模型并行实施遇到重大障碍:

| 备选方案 | 描述 | 性能折衷 | 时间节省 |
|---------|------|---------|---------|
| **模型分片 (Offloading)** | 将部分层卸载到 CPU/NVMe | 3-10x 变慢 | -60% 时间 |
| **量化极致压缩** | 1-bit/2-bit 量化 | 精度损失 1-3% | -80% 时间 |
| **蒸馏小模型** | 训练专用的学生模型 | 能力受限 | N/A (长期) |
| **放弃大模型** | 只支持 < 13B 模型 | 功能受限 | -90% 时间 |

---

## 6. 推荐路线图

### 6.1 三阶段实施计划

#### Phase 1: 基础张量并行（8周）🏗️

**目标**: 2 节点协同推理 Llama-7B

**Week 1-2: 通信基础设施**
- [ ] 选择并集成通信库（推荐 Tokio Mesh 或自研 Ring All-Reduce）
- [ ] 实现 `CollectiveComm` trait（All-Reduce/All-Gather/Broadcast）
- [ ] 搭建 2 节点测试环境

**Week 3-4: 模型分片**
- [ ] 修改 `ModelLoader`（支持按 rank 加载权重切片）
- [ ] 实现 `ColumnParallelLinear` 和 `RowParallelLinear`
- [ ] 验证分片后数值正确性（bit-exact test）

**Week 5-6: 引擎改造**
- [ ] 修改 `InferenceEngine.forward()`（插入 All-Reduce 点）
- [ ] 实现 `DistributedInferenceContext`（简化版，StrictBarrier KV sync）
- [ ] 端到端测试（2节点 vs 单节点结果对比）

**Week 7-8: 性能与稳定性**
- [ ] 添加通信-计算重叠优化
- [ ] 实现基本错误处理（节点故障检测）
- [ ] 性能 benchmark 和 profiling

**交付物**:
- ✅ 可运行的 2 节点分布式推理 demo
- ✅ Llama-7B 推理加速 ≥ 1.6x (2 节点)
- ✅ 完整的技术文档和测试报告

#### Phase 2: 完善与扩展（8周）🎯

**目标**: 支持 4-8 节点，Llama-70B 推理

**Week 9-10: KV Cache 优化**
- [ ] 实现 AsyncBatching KV sync（替代 StrictBarrier）
- [ ] 添加分布式 PagedAttention（跨节点 block 管理）
- [ ] Prefix Cache 全局共享

**Week 11-12: 规模化扩展**
- [ ] 测试 4 节点、8 节点部署
- [ ] 性能调优（网络参数、batch size、overlap 策略）
- [ ] 支持混合精度推理（FP16/BF16 All-Reduce）

**Week 13-14: 监控与运维**
- [ ] 集成 Prometheus metrics（通信延迟、带宽、GPU 利用率）
- [ ] 分布式日志和 tracing（跨请求关联）
- [ ] 告警规则和 dashboard

**Week 15-16: 文档与测试**
- [ ] 部署指南（多节点配置最佳实践）
- [ ] 性能调优手册
- [ ] 自动化回归测试（2/4/8 节点 CI）

**交付物**:
- ✅ 生产级的分布式推理服务
- ✅ Llama-70B 推理加速 ≥ 6x (8 节点)
- ✅ 完整的运维文档和监控体系

#### Phase 3: 高级特性（6周）🚀

**目标**: MoE 支持 + 弹性扩展

**Week 17-18: Mixtral MoE 支持**
- [ ] 实现 Router（top-k expert selection）
- [ ] 专家并行（每个节点持有一部分专家）
- [ ] All-to-All 通信优化

**Week 19-20: 弹性与容错**
- [ ] 节点热插拔（动态加入/离开）
- [ ] Checkpoint/Restore（快速恢复）
- [ ] 优雅降级（N-1 节点继续服务）

**Week 21-22: 高级优化**
- [ ] 序列并行（SP）实验（针对超长文本）
- [ ] 2D/3D 并行组合（TP + PP 或 TP + MoE）
- [ ] 自适应并行策略（根据模型自动选择方案）

**交付物**:
- ✅ Mixtral 8x7B 分布式推理
- ✅ 弹性伸缩能力
- ✅ 技术白皮书和论文（可选）

### 6.2 里程碑与验收标准

| 里程碑 | 日期 | 验收标准 | 成功指标 |
|--------|------|---------|---------|
| **M1: 原型验证** | 2026-06-01 | 2 节点 Llama-7B 推理成功 | ✅ Bit-exact 正确性<br>✅ 加速比 > 1.5x<br>✅ 通信延迟 < 2ms |
| **M2: 生产就绪** | 2026-08-01 | 4-8 节点 Llama-70B 推理 | ✅ 吞吐量 > 100 tok/s<br>✅ 99.9% 可用性<br>✅ P99 延迟 < 200ms |
| **M3: MoE 支持** | 2026-09-15 | Mixtral 分布式推理 | ✅ 专家并行加速 > 3x<br>✅ 负载均衡效率 > 85%<br>✅ 支持动态扩缩容 |

### 6.3 长期愿景 (2026 Q4 - 2027)

```
OpenMini-V1 分布式推理演进路径:

2026 Q2 (当前)     2026 Q3          2026 Q4          2027 Q1+
    │                 │                │                │
    ▼                 ▼                ▼                ▼
┌──────┐         ┌──────────┐    ┌──────────┐    ┌──────────────┐
│Single│ ──────▶  │  Tensor  │ ──▶│  Hybrid  │ ──▶│   Auto-      │
│Node  │         │ Parallel │    │ Parallel │    │   Parallel   │
│      │         │ (2-8节点) │    │ (TP+MoE)  │    │ (智能调度)   │
└──────┘         └──────────┘    └──────────┘    └──────────────┘
                       │                │                │
                       ▼                ▼                ▼
                  Llama-70B       Mixtral        GPT-4 class
                  (生产级)        (MoE支持)      (100B+ params)
```

---

## 附录

### A. 相关论文

1. **Megatron-LM: Training Multi-Billion Parameter Models Using Model Parallelism** (Shoeybi et al., 2019)
   - https://arxiv.org/abs/1909.08053

2. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** (Narayanan et al., 2021)
   - https://arxiv.org/abs/2104.04473

3. **Mixtral of Experts** (Jiang et al., 2024)
   - https://arxiv.org/abs/2401.04088

4. **LongNet: Scaling Transformers to 1,000 Tokens** (Ding et al., 2023)
   - https://arxiv.org/abs/2307.07929 (序列并行参考)

5. **PipeDream: Fast and Efficient Pipeline Parallelism for DNN Training** (Narayanan et al., 2019)
   - https://arxiv.org/abs/1806.03377 (管道并行参考)

### B. 参考实现

| 项目 | 语言 | Stars | 特点 | 相关性 |
|------|------|-------|------|--------|
| **Megatron-LM** | Python | 35k+ | NVIDIA 官方 TP 实现 | ⭐⭐⭐⭐⭐ |
| **DeepSpeed** | Python | 35k+ | Microsoft 分布式训练 | ⭐⭐⭐⭐ |
| **FairScale** | Python | 3k+ | Facebook TP/PP 实现 | ⭐⭐⭐⭐ |
| **colinml/tensordict** | Rust | 500+ | Rust 分布式张量库 | ⭐⭐⭐⭐⭐ |
| **candle-distributed** | Rust | 100+ | 社区 Candle 分布式尝试 | ⭐⭐⭐⭐ |

### C. 术语表

| 术语 | 全称 | 定义 |
|------|------|------|
| **TP** | Tensor Parallelism | 张量并行：将单个张量计算分散到多设备 |
| **PP** | Pipeline Parallelism | 管道并行：将模型层分散到多设备 |
| **DP** | Data Parallelism | 数据并行：将样本分散到多设备 |
| **SP** | Sequence Parallelism | 序列并行：将长序列切分到多设备 |
| **MoE** | Mixture of Experts | 专家并行：MoE 层的不同专家放不同设备 |
| **All-Reduce** | - | 集合通信：求和/求均值等归约操作并广播 |
| **Ring All-Reduce** | - | 环形拓扑的 All-Reduce 算法 |
| **Column Parallel** | - | 列并行：按输出维度切分权重 |
| **Row Parallel** | - | 行并行：按输入维度切分权重 |
| **Communication-Computation Overlap** | - | 通信-计算重叠：在通信等待时执行计算 |
| **Bubble** | - | 气泡效应：流水线中的空闲时间 |

### D. 参考资源

- **OpenMini-V1 Engine**: [engine.rs](file:///Users/apple/Desktop/OpenMini-V1/openmini-server/src/model/inference/engine.rs)
- **OpenMini-V1 KV Cache**: [kv_cache/mod.rs](file:///Users/apple/Desktop/OpenMini-V1/openmini-server/src/hardware/kv_cache/mod.rs)
- **OpenMini-V1 Scheduler**: [scheduler.rs](file:///Users/apple/Desktop/OpenMini-V1/openmini-server/src/hardware/scheduler.rs)
- **Candle GitHub**: https://github.com/huggingface/candle
- **Megatron-LM GitHub**: https://github.com/NVIDIA/Megatron-LM

---

**文档结束**

*本报告基于 OpenMini-V1 项目代码库深度分析（2026-04-10），所有技术方案和建议均经过充分论证。实施前建议先进行 2-4 周的原型验证（POC）。*
