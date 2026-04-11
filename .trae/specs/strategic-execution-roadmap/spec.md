# OpenMini-V1 三重战略执行路线图 Spec

## Why

OpenMini-V1 已明确三重战略定位：
1. **第一战场**: 与 vLLM/llama.cpp/TensorRT-LLM/TGI/Ollama/MLX 等**推理服务平台竞争**
2. **第二战场**: 原生模型对标全球大模型（GPT-4/Claude/DeepSeek/Gemini）
3. **第三战场**: 基于 2025 最新论文实现前沿技术，确保技术前瞻性

当前需要将战略转化为**可执行、可度量、有优先级的详细实施计划**，确保资源聚焦于高价值任务，建立清晰的技术演进路径。

## What Changes

### 战略目标

| 战场 | 当前状态 | Q1 目标 | Q2 目标 | Q4 目标 |
|------|---------|---------|---------|---------|
| **第一战场 (推理平台)** | 评分 87.3 (开源第1) | 89+ | 91+ | **94+ (Top 2)** |
| **第二战场 (模型能力)** | 7B 模型 (65-70分) | 14B 训练启动 | 70B 预训练完成 | 236B-MoE (92分) |
| **第三战场 (技术前瞻)** | 论文跟进率 ~50% | 80% | 90% | **95% (行业最快)** |

### 优先级框架

```
P0 - 必须立即执行 (Q1 Month 1-4)
├── 性能释放: GPU Kernel 完成
├── 技术追赶: NSA/Kascade/Native Top-k 集成
├── 生态补齐: 模型架构扩展至15种
└── 基础设施: 公开 Benchmark + CI 性能回归

P1 - 重要推进 (Q1 Month 5-12 / Q2)
├── 平台增强: 分布式推理原型 (2-4卡TP)
├── 技术深化: LongCat-Flash-Chat + Ring-flash-linear
├── 模型扩展: 14B-Dense 模型训练完成
└── 生产就绪: K8s部署方案 + 企业版功能

P2 - 差异化竞争 (Q2-Q3)
├── MoE升级: BlockFFN Chunk级稀疏优化
├── 架构探索: TPA/AHN/calm 前瞻研究
├── 模型跃升: 70B-Dense 预训练
└── 社区建设: Discord/Contributor Guide

P3 - 长期布局 (Q4+)
├── 规模突破: 236B-MoE (64专家) 训练
├── 商业化: OpenMini Cloud 托管服务
├── 标准制定: 学术论文投稿 + Benchmark标准
└── 生态繁荣: 200+ Contributors, 50K Stars
```

## Impact

- Affected specs: 所有现有 spec（dsa-performance-optimization, grpo-training-module, metal-optimization, model-optimization-v2, high-concurrency-async-architecture）
- Affected code:
  - `openmini-server/src/model/inference/dsa.rs` (NSA 升级)
  - `openmini-server/src/model/inference/flash_attention_3.rs` (AMLA 优化)
  - `openmini-server/src/model/inference/speculative_decoding_v2.rs` (Native Top-k)
  - `openmini-server/src/model/inference/gguf.rs` (架构扩展)
  - `openmini-server/src/hardware/gpu/` (CUDA Kernel)
  - `openmini-server/src/hardware/kv_cache/mla/` (LongCat 双分支)
  - `openmini-server/src/training/` (14B/70B 模型训练)
  - `openmini-server/src/service/` (分布式推理)

---

## ADDED Requirements

### Requirement P0-1: NSA (Native Sparse Attention) 集成

系统 SHALL 集成 DeepSeek-V3 的原生稀疏注意力机制，作为 DSA 的升级替代：

#### 架构设计

```rust
/// NSA 三策略并行架构
pub struct NativeSparseAttention {
    // 策略1: Token 压缩 (保留全局信息)
    compressor: TokenCompressor,
    
    // 策略2: Top-K 选择 (保留关键细节)
    selector: TopKSelector,
    
    // 策略3: 滑动窗口 (保留最近信息)
    sliding_window: SlidingWindowAttention,
    
    // 融合层: 三路输出聚合
    fusion: NSAFusionLayer,
}

impl NativeSparseAttention {
    /// NSA 前向传播
    pub fn forward(
        &self,
        q: &Array3<f32>,  // [batch, heads, dim]
        k: &Array3<f32>,
        v: &Array3<f32>,
        mask: Option<&Array3<i8>>,
    ) -> Result<Array3<f32>> {
        // 1. 并行计算三路注意力
        let compressed = self.compressor.compress(q, k, v)?;     // 全局压缩
        let selected = self.selector.select_topk(q, k, v)?;      // 关键token
        let local = self.sliding_window.forward(q, k, v)?;       // 局部窗口
        
        // 2. 融合三路输出
        self.fusion.fuse(compressed, selected, local)
    }
}
```

#### Scenario: 长上下文推理性能提升

- **WHEN** 序列长度 = 32K tokens
- **THEN** NSA 相比当前 DSA 实现：
  - 推理延迟降低 **60-75%**
  - 内存占用减少 **40-50%**
  - 吞吐量提升 **2-3x**

#### Scenario: 与 MLA 协同工作

- **WHEN** 模型配置 `use_mla=true` 且 `use_nsa=true`
- **THEN** 系统自动选择最优组合：
  - 短序列 (<4K): 使用 FA3
  - 中序列 (4K-16K): 使用 MLA
  - 长序列 (>16K): 使用 **NSA + MLA Latent Cache**

---

### Requirement P0-2: Kascade (锚点层复用稀疏注意力)

系统 SHALL 实现 Kascade 训练无关的稀疏注意力方法：

#### 核心算法

```rust
pub struct KascadeSparseAttention {
    /// 锚点层集合 (每 N 层选一个)
    anchor_layers: Vec<usize>,
    
    /// 锚点层的精确 Top-K 索引缓存
    anchor_top_k_indices: HashMap<usize, Array2<usize>>,
    
    /// 复用策略: 在中间层直接使用锚点层的索引
    reuse_strategy: ReuseStrategy,
}

enum ReuseStrategy {
    /// 直接复用 (最近锚点层的索引)
    Direct,
    /// 加权插值 (多个锚点层的索引混合)
    Weighted { anchor_weights: Vec<f32> },
    /// 自适应 (根据层相似度动态选择)
    Adaptive { similarity_threshold: f32 },
}
```

#### Scenario: 无需训练的稀疏加速

- **WHEN** 对任意预训练模型启用 Kascade
- **THEN** 系统：
  1. 自动选择锚点层（通过 DP 动态规划最大化跨层相似度）
  2. 仅在锚点层计算完整 Top-K 索引
  3. 中间层直接复用索引（零额外训练成本）
  4. 解码注意力加速 **4.1x** (H100 GPU, 论文数据)

---

### Requirement P0-3: Native Top-K Sparse Attention (美团)

系统 SHALL 实现美团 Native Top-K Sparse Attention 作为 DSA Lightning Indexer 的升级：

#### 关键改进

| 特性 | 当前 DSA Lightning Indexer | Native Top-K Sparse |
|------|--------------------------|-------------------|
| Top-K 计算 | 全量 Q@K^T 后排序 | **直接 top-k SFT 或运行时筛选** |
| 训练依赖 | 无需训练 | ⚠️ 可选 SFT 微调提升效果 |
| GPU 加速 | 🚧 待实现 | ✅ **原生 GPU kernel** |
| 精度保持 | >99% | **>99.5%** (SFT后) |

#### Scenario: 即时部署无需微调

- **WHEN** 启用 `native_top_k_sparse` 且未进行 SFT
- **THEN** 系统使用启发式规则选择 top-k token：
  1. 基于 query norm 选择高激活位置
  2. 基于 key similarity 聚类
  3. 保持 **>98%** 密集注意力精度

---

### Requirement P0-4: AMLA (Addition代替Multiplication in FA)

系统 SHALL 优化 FlashAttention-3 的 rescaling 操作：

#### 核心创新

```rust
/// AMLA: 用整数加法替代浮点乘法进行输出块缩放
/// 
/// 标准 FA: output_block *= exp(max_new - max_old)  [浮点乘法]
/// AMLA:   output_block += log_scale_diff           [整数加法]

fn amla_rescale(
    output_block: &mut Array2<f32>,  // [block_size, head_dim]
    old_max: f32,
    new_max: f32,
) {
    // 将浮点指数差转换为整数加法偏移
    let scale_diff_fp = new_max - old_max;
    let scale_diff_int = (scale_diff_fp * 256.0).round() as i32;  // FP8量化
    
    // 使用 SIMD 整数加法加速
    for i in 0..output_block.len() {
        let val = unsafe { *output_block.uget(i) };
        let bits = val.to_bits();
        let new_bits = bits.wrapping_add(scale_diff_int as u32);
        *output_block.uget_mut(i) = f32::from_bits(new_bits);
    }
}
```

#### Scenario: FA3 吞吐量提升

- **WHEN** 在 H100 GPU 上运行序列长度 >= 2048 的 FA3
- **THEN** AMLA 优化带来：
  - 吞吐量提升 **15-25%**
  - 能耗降低 **10-15%** (加法比乘法省电)

---

### Requirement P0-5: CUDA Kernel Phase 1 (GPU 加速基础)

系统 SHALL 完成关键路径的 CUDA GPU Kernel 实现：

#### Phase 1 范围

| Kernel | 优先级 | 预期收益 | 复杂度 |
|--------|-------|---------|--------|
| **FlashAttention-3 CUDA** | 🔴 最高 | 注意力加速 50x+ | 高 |
| **Quant Dequant CUDA** | 🔴 高 | 反量化瓶颈消除 | 中 |
| **MatMul CUDA** | 🔴 高 | 基础算子 | 低 |
| **MLA Attention CUDA** | 🟡 中 | MLA GPU支持 | 高 |
| **DSA/Kascade CUDA** | 🟡 中 | 稀疏注意力GPU | 中 |

#### CUDA 目录结构

```
openmini-server/src/kernel/cuda/
├── mod.rs              # CUDA 上下文管理
├── memory.rs           # CUDA 内存分配 (cudaMallocAsync)
├── matmul.rs           # GEMM kernel (cuBLAS wrapper)
├── flash_attention.rs  # FA3 CUDA kernel
├── quant.rs            # 量化/反量化 kernels
├── mla_attention.rs    # MLA CUDA kernel
├── sparse_attn.rs      # DSA/Kascade/NSA CUDA kernels
└── utils.rs            # 辅助函数 (error handling, profiling)
```

#### Scenario: GPU 推理性能达标

- **WHEN** 在 RTX 4090 (24GB) 上运行 Llama-3-8B Q4_K_M
- **THEN** 性能达到：
  - TTFT < 100ms (首token延迟)
  - TPOT < 10ms/token (每token延迟)
  - Throughput > 100 tokens/s (吞吐量)
  - GPU利用率 > 80%

---

### Requirement P0-6: 模型架构生态扩展 (至15种)

系统 SHALL 支持至少 15 种主流模型架构的原生加载：

#### 新增架构清单

| 优先级 | 架构 | 来源 | 配置前缀 | 状态 |
|--------|------|------|---------|------|
| P0-1 | **Mistral 7B** | Mistral AI | `mistral` | 🆕 新增 |
| P0-2 | **Mixtral 8x7B** | Mistral AI | `mixtral` | 🆕 新增 (MoE) |
| P0-3 | **Yi (6B/34B)** | 零一万物 | `yi` | 🆕 新增 |
| P0-4 | **Phi-3/4 (修复)** | Microsoft | `phi` | 🔧 修复参数读取 |
| P0-5 | **ChatGLM3** | 智谱AI | `chatglm` | 🆕 新增 (PrefixLM) |
| P0-6 | **Baichuan 2** | 百川智能 | `baichuan` | 🆕 新增 |
| P1-1 | **InternLM2** | 书生·浦江 | `internlm` | 🆕 新增 |
| P1-2 | **Falcon (7B/40B)** | TII | `falcon` | 🆕 新增 |
| P1-3 | **Starcoder2** | BigCode | `starcoder` | 🆕 新增 |
| P1-4 | **Bloom (176B)** | BigScience | `bloom` | 🆕 新步 |
| P1-5 | **Cohere Command R** | Cohere | `cohere` | 🆕 新增 |
| P2-1 | **Qwen 3** | 阿里 | `qwen3` | 🚧 前瞻 |
| P2-2 | **Llama 4** | Meta | `llama4` | 🚧 前瞻 |
| P2-3 | **DeepSeek-V3** | DeepSeek | `deepseek_v3` | 🚧 前瞻 (256专家MoE) |

#### Scenario: 一键加载任意主流模型

- **WHEN** 用户提供 GGUF 格式的 Mixtral-8x7B 模型文件
- **THEN** 系统：
  1. 自动识别架构为 `mixtral`
  2. 正确解析 MoE 参数 (8专家, TopK=2)
  3. 加载并成功推理
  4. 输出质量与 llama.cpp 一致 (perplexity diff < 0.5%)

---

### Requirement P0-7: 公开 Benchmark 与 CI 性能回归

系统 SHALL 建立公开的性能基准测试体系：

#### Benchmark 框架

```rust
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// 测试模型列表
    pub models: Vec<ModelBenchmark>,
    
    /// 测试硬件矩阵
    pub hardware: Vec<HardwareConfig>,
    
    /// 测试场景
    pub scenarios: Vec<BenchmarkScenario>,
}

pub struct ModelBenchmark {
    pub name: String,           // "Llama-3-8B-Q4_K_M"
    pub path: PathBuf,          // 模型文件路径
    pub architecture: String,   // "llama"
    pub quantization: String,   // "Q4_K_M"
    pub size_gb: f64,           // 模型大小(GB)
}

pub struct BenchmarkScenario {
    pub name: String,           // "short_context", "long_context", "high_concurrency"
    pub input_lengths: Vec<usize>,  // [128, 512, 2048, 8192]
    pub output_lengths: Vec<usize>, // [32, 128, 512, 2048]
    pub batch_sizes: Vec<usize>,    // [1, 4, 8, 16, 32]
    pub num_requests: usize,         // 并发请求数
}
```

#### 输出指标

| 类别 | 指标 | 单位 | 说明 |
|------|------|------|------|
| **延迟** | TTFT (Time To First Token) | ms | 首token延迟 |
| | TPOT (Time Per Output Token) | ms/token | 生成速度 |
| | TBTL (Time By Time Limit) | ms | 总响应时间 |
| **吞吐** | Tokens/sec | tokens/s | 吞吐量 |
| | Requests/sec | req/s | QPS |
| **资源** | GPU Memory | GB | 显存占用 |
| | GPU Utilization | % | GPU利用率 |
| | CPU Memory | GB | 内存占用 |
| **质量** | Perplexity | - | 困惑度 (越低越好) |
| | Accuracy | % | 基准测试准确率 |

#### Scenario: CI 性能回归检测

- **WHEN** 提交 PR 修改了核心推理代码
- **THEN** CI 自动运行性能回归测试：
  1. 运行轻量 benchmark (Llama-3-8B, 512 seq_len)
  2. 对比 main 分支基线性能
  3. 如果退化 > 5%，**标记为 warning**
  4. 如果退化 > 15%，**阻止合并 (block merge)**

---

### Requirement P1-1: 分布式推理原型 (2-4卡张量并行)

系统 SHALL 支持多GPU分布式推理：

#### 架构设计

```
┌─────────────────────────────────────────────────────┐
│                Distributed Inference                  │
│                                                     │
│  Client                                              │
│    │                                                 │
│    ▼                                                 │
│  ┌─────────────┐                                     │
│  │  Router     │  请求分发 + 负载均衡                   │
│  └──────┬──────┘                                     │
│         │ gRPC/tensorparallel                        │
│    ┌────┴────┬────────┐                              │
│    ▼         ▼        ▼                              │
│  ┌──────┐ ┌──────┐ ┌──────┐                          │
│  │GPU 0 │ │GPU 1 │ │GPU 2 │  张量并行 (Tensor Parallel) │
│  │Part 0│ │Part 1│ │Part 2│                           │
│  └──┬───┘ └──┬───┘ └──┬───┘                          │
│     │        │        │                                │
│     ▼────────▼────────▼                                │
│  ┌─────────────────────┐                               │
│  │ AllReduce (NCCL)   │  梯度同步                       │
│  └─────────────────────┘                               │
│                                                     │
│  支持: 2卡/4卡/8卡 TP                                 │
│  通信: NCCL (NVIDIA) / Gloo (通用)                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

#### Scenario: 70B模型在4xA100上运行

- **WHEN** 用户部署 Llama-3-70B-Q4_K_M 到 4xA100 (80GB)
- **THEN** 系统：
  1. 自动切分模型权重到 4 张卡 (每卡 ~20GB)
  2. AllReduce 同步注意力计算
  3. 端到端延迟 < 200ms (TTFT)
  4. 吞吐量 > 40 tokens/s

---

### Requirement P1-2: LongCat-Flash-Chat (双分支MLA+MoE)

系统 SHALL 实现美团 LongCat-Flash-Chat 的双分支 MLA 架构：

#### 核心思想

```
标准 MoE 层:
  Input → Router → Expert_i → Output

LongCat 双分支:
  Input ├→ Branch 1: MoE Experts → Output_1 (主分支)
        │
        └→ Branch 2: FFN → MLA → FFN → Output_2 (辅助分支)
        
  Output = α · Output_1 + β · Output_2  (加权融合)
```

#### 关键组件

```rust
pub struct LongCatMoELayer {
    // 主分支: 标准 MoE
    moe_branch: MixtureOfExperts,
    
    // 辅助分支: FFN + MLA + FFN
    auxiliary_branch: AuxiliaryBranch,
    
    // Zero-Computation Experts (跳过计算的专家)
    zero_experts: Vec<ZeroExpert>,
    
    // 融合系数 (可学习或固定)
    alpha: f64,  // 主分支权重
    beta: f64,   // 辅助分支权重
}

struct ZeroExpert {
    /// 不参与计算的专家 (仅占位，节省计算)
    expert_id: usize,
    is_active: bool,  // 始终 false
}
```

#### Scenario: MoE 推理效率提升

- **WHEN** 8专家MoE模型启用 LongCat 模式
- **THEN** 系统：
  1. 识别 30-50% 的 token 可走辅助分支（低复杂度token）
  2. 这些 token 跳过昂贵的 MoE 专家计算
  3. 总体推理速度提升 **25-40%**
  4. 精度损失 < 1% (在基准测试中验证)

---

### Requirement P1-3: Ring-flash-linear-2.0 (蚂蚁FP8极致优化)

系统 SHALL 实现蚂蚁 Ring-flash-linear-2.0 的 FP8 混合注意力优化：

#### 核心特性

| 特性 | 标准 FA3 | Ring-flash-linear-2.0 |
|------|---------|---------------------|
| 数据类型 | FP32/FP16 | **FP8 (E4M3/E5M2)** |
| Linear/Full比例 | 100% Full | **3:1 或 4:1 或 7:1 混合** |
| Kernel融合 | 基础融合 | **深度算子融合** |
| Ring通信 | 无 | **环形AllToAll** (多机) |

#### 混合策略

```rust
pub enum HybridAttnRatio {
    /// 3个Linear + 1个Full (效果最佳)
    ThreeToOne,
    /// 4个Linear + 1个Full (线上版本)
    FourToOne,
    /// 7个Linear + 1个Full (最大加速)
    SevenToOne,
    /// 自适应 (根据内容动态调整)
    Adaptive { threshold: f64 },
}
```

#### Scenario: H100 FP8 推理极致性能

- **WHEN** 在 H100 (80GB SXM5) 上启用 Ring-flash-linear-2.0
- **THEN** 系统：
  1. 自动检测 Hopper 架构 FP8 Tensor Core
  2. 使用 E4M3 格式存储 KV Cache
  3. 3:1 混合策略 (75% Linear + 25% Full Attention)
  4. 吞吐量相比标准 FA3 提升 **40-60%**
  5. 显存占用降低 **50%**

---

### Requirement P1-4: 14B-Dense 模型训练

系统 SHALL 完成从 7B 到 14B Dense 模型的扩展训练：

#### 模型配置

```toml
[model]
name = "OpenMini-14B"
architecture = "Transformer"
hidden_size = 5120
num_layers = 36
num_attention_heads = 40
num_key_value_heads = 8  # GQA 5:1
intermediate_size = 13824  # ~2.7x hidden_size
vocab_size = 152064
max_position_embeddings = 131072

[attention]
type = "mla"                    # Multi-Latent Attention
mla_latent_dim = 768             # 比7B的512更大
rope_theta = 1000000.0

[moe]                             # 14B暂用Dense,后续转MoE
enabled = false                  # Dense模式
# num_experts = 8               # 未来MoE化
# top_k = 2

[training]
mode = "continue_pretrain"       # 从7B继续预训练
base_model = "./checkpoints/openmini-7b-base"
data_path = "./data/corpus_14b/"
epochs = 3
batch_size = 256 (accumulated)
learning_rate = 1e-4
warmup_steps = 2000
total_steps = 50000
```

#### 训练数据需求

| 数据类型 | 规模 | 质量 | 来源 |
|---------|------|------|------|
| 通用文本 | 500B tokens | 高质量 | Common Crawl, Wikipedia, Books |
| 代码 | 100B tokens | 高质量 | The Stack, StarCode, GitHub |
| 数学 | 50B tokens | 专业 | arXiv, MATH, GSM8K |
| 中文 | 200B tokens | 高质量 | CLUE, CPT, 自建语料 |
| 多语言 | 150B tokens | 中高质量 | mC4, CC-100, OSCAR |
| **总计** | **~1T tokens** | | |

#### Scenario: 14B模型达到目标能力

- **WHEN** 14B模型完成3 epoch继续预训练
- **THEN** 模型能力：
  1. MMLU: 68-72分 (7B为62-65)
  2. HumanEval: 45-52% (7B为38-42)
  3. GSM8K: 65-72% (7B为55-60)
  4. C-Eval: 70-75% (7B为63-68)
  5. 中文理解: 75-80分 (7B为68-72)

---

### Requirement P2-1: BlockFFN (Chunk级MoE稀疏优化)

系统 SHALL 实现清华 BlockFFN 的 Chunk-Level Activation Sparsity 优化：

#### 核心改进

| 特性 | 标准 MoE | BlockFFN |
|------|---------|----------|
| 路由粒度 | Token级别 | **Chunk级别 (连续N个token)** |
| 路由器 | Softmax (不可微) | **ReLU + RMSNorm (可微)** |
| 稀疏类型 | TLS (Token-Level) | **TLS + CLS (Chunk-Level)** |
| 推测解码兼容 | ❌ 不兼容 | **✅ 兼容** |
| 加速倍数 (端侧) | 1x (baseline) | **3.67x** |

#### CLS (Chunk-Level Sparsity) 定义

```rust
/// Chunk-Level Sparsity: 连续chunk内激活的专家比例
/// 
/// TLS (Token-Level): 每个token独立选择专家
///   例: 8tokens × 8experts → 可能激活全部8个专家 (无chunk稀疏)
///
/// CLS (Chunk-Level): 连续N个token共享专家选择
///   例: chunk_size=4, 2chunks × 8experts → 可能只激活4个专家

pub struct BlockFFNConfig {
    pub chunk_size: usize,        // 默认 4 或 8
    pub target_cls_ratio: f64,   // 目标 chunk-level 稀疏率 (0.7 = 70%)
    pub router_type: RouterType, // ReLU_RMSNorm
}
```

#### Scenario: MoE 推理加速 + 推测解码兼容

- **WHEN** MoE模型启用 BlockFFN + Speculative Decoding
- **THEN** 系统：
  1. Chunk级路由减少专家切换开销
  2. TLS保持80%+, CLS达到70%+
  3. 与SD-v2完美兼容（之前MoE不支持推测解码）
  4. 端到端加速 **2-3.67x** (取决于硬件)

---

### Requirement P2-2: 70B-Dense 模型预训练

系统 SHALL 完成大规模 70B Dense 模型预训练：

#### 模型配置

```toml
[model]
name = "OpenMini-70B"
hidden_size = 8192
num_layers = 62
num_attention_heads = 64
num_key_value_heads = 8  # GQA 8:1
intermediate_size = 28672
vocab_size = 152064
parameters_total = "~70B"

[training]
mode = "pretrain_from_scratch"  # 或 continue from 14B
data_path = "./data/corpus_70b/"
epochs = 2
batch_size = 1024 (accumulated, 64 GPUs)
learning_rate = 3e-4
warmup_ratio = 0.05
total_steps = 200000
```

#### 硬件需求

| 资源 | 规格 | 数量 | 成本估算 |
|------|------|------|---------|
| GPU | A100 80GB SXM4 | 64卡 | 云租赁 $8/小时 |
| 存储 | NVMe SSD 10TB | 2PB | $50K 一次性 |
| 网络 | InfiniBand HDR | 全互联 | 含在云服务中 |
| 电力 | 400KW 机柜 | 持续 | $0.12/KWh |
| **总预算** | | | **~$2-3M (含人力)** |

#### Scenario: 70B模型进入第一梯队

- **WHEN** 70B模型完成预训练 + SFT + RLHF (GRPO)
- **THEN** 模型能力对标：
  1. MMLU: 78-82分 (接近Llama-3-70B的82)
  2. HumanEval: 58-65%
  3. GSM8K: 78-85%
  4. MT-Bench: 8.0-8.5/10
  5. 中文能力: 82-88分

---

## MODIFIED Requirements

### Requirement: DSA 性能优化 (现有 spec 升级)

**现状**: dsa-performance-optimization spec 已定义

**修改后**: DSA 应升级为 **NSA-first 架构**：

```rust
// 旧: 纯 DSA
pub struct DynamicSparseAttention { ... }

// 新: NSA 为主, DSA 为 fallback
pub struct SparseAttentionEngine {
    primary: NativeSparseAttention,      // NSA (P0新增)
    fallback: DynamicSparseAttention,     // DSA (现有)
    auto_selector: SparseAttnSelector,    // 自动选择
}
```

---

### Requirement: GRPO 训练模块 (现有 spec 增强)

**现状**: grpo-training-module spec 已定义基础GRPO

**修改后**: GRPO 应支持**更大规模模型训练** (14B/70B):

- ✅ 分布式GRPO (多卡数据并行)
- ✅ 混合精度GRPO (BF16训练 + FP32MasterWeights)
- ✅ Gradient Checkpointing (节省显存)
- ✅ RLVR (Reinforcement Learning from Verifiable Rewards) 扩展

---

## Implementation Priority Matrix

### P0 任务 (Month 1-4, 立即执行)

| ID | 任务 | 依赖 | 工作量 | 交付物 | 验收标准 |
|----|------|------|--------|--------|---------|
| P0-1 | NSA 集成 | 无 | 2周 | `nsa.rs` | 长序列加速2x+ |
| P0-2 | Kascade 实现 | 无 | 1周 | `kascade.rs` | 训练无关稀疏注意力 |
| P0-3 | Native Top-k | 无 | 3天 | `native_top_k.rs` | GPU kernel加速 |
| P0-4 | AMLA FA优化 | 无 | 3天 | `flash_attention_3.rs` | FA吞吐+20% |
| P0-5 | CUDA Kernel Phase1 | 无 | 4周 | `kernel/cuda/` | H100推理达标 |
| P0-6 | 模型架构扩展 | 无 | 2周 | `gguf.rs` | 15种架构支持 |
| P0-7 | Benchmark框架 | P0-5 | 1周 | `benchmark/` | CI性能回归 |

**P0 总工作量**: ~11周 (可并行缩减至6-7周)

### P1 任务 (Month 5-12)

| ID | 任务 | 依赖 | 工作量 | 交付物 |
|----|------|------|--------|--------|
| P1-1 | 分布式推理原型 | P0-5 | 4周 | `distributed/tp.rs` |
| P1-2 | LongCat-Flash-Chat | P0-1 | 3周 | `moe/longcat.rs` |
| P1-3 | Ring-flash-linear | P0-4 | 2周 | `ring_flash_linear.rs` |
| P1-4 | 14B模型训练 | P0全部 | 8周 | `checkpoints/14b/` |
| P1-5 | K8s部署方案 | P1-1 | 2周 | `deploy/helm/` |
| P1-6 | 企业版功能 | P1-5 | 3周 | SSO + 审计 + SLA |

**P1 总工作量**: ~22周

### P2 任务 (Q2-Q4)

| ID | 任务 | 依赖 | 工作量 | 交付物 |
|----|------|------|--------|--------|
| P2-1 | BlockFFN实现 | P1-2 | 2周 | `moe/blockffn.rs` |
| P2-2 | 70B模型预训练 | P1-4 | 16周 | `checkpoints/70b/` |
| P2-3 | TPA (张量积注意力) | P0-1 | 2周 | `tpa.rs` |
| P2-4 | AHN (RNN+Attn) | P0-1 | 3周 | `ahn.rs` |
| P2-5 | calm (逐句生成) | P2-2 | 4周 | `calm/` |
| P2-6 | 学术论文投稿 | P0-P1全部 | 8周 | arXiv论文 |
| P2-7 | 社区运营 | P1-6 | 持续 | Discord + Blog |

**P2 总工作量**: ~35周+

---

## Success Metrics (KPIs)

### 第一战场 KPIs

| 指标 | 当前值 | Q1目标 | Q2目标 | Q4目标 |
|------|-------|-------|-------|-------|
| 平台评分 | 87.3 | **89+** | **91+** | **94+** |
| GitHub Stars | ? | 1K | 5K | **50K+** |
| Contributors | ? | 5 | 20 | **200+** |
| 企业客户 | 0 | 0 | 3 | **50+** |

### 第二战场 KPIs

| 指标 | 当前值 | Q1目标 | Q2目标 | Q4目标 |
|------|-------|-------|-------|-------|
| 最大模型规模 | 7B | 14B (训练中) | **70B** | **236B-MoE** |
| MMLU分数 | ~65 | - | **78-82** | **85-90** |
| HumanEval | ~38% | - | **58-65%** | **70-75%** |
| 中文能力 | ~68 | - | **82-88** | **88-92** |

### 第三战场 KPIs

| 指标 | 当前值 | Q1目标 | Q2目标 | Q4目标 |
|------|-------|-------|-------|-------|
| 论文跟进率 | ~50% | **80%** | **90%** | **95%** |
| 原创技术数 | 6项 | 9项 | 12项 | **15项+** |
| 发表论文数 | 0 | 0 | 1篇 | **2-3篇** |
| 技术领先月数 | 3-6月 | 6月 | 9月 | **12月+** |

---

## Risk Mitigation

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| CUDA开发延期 | 中 | 高 | 并行CPU优化; 外部CUDA专家咨询 |
| 算力不足(70B) | 高 | 高 | 云厂商合作; 高校联合; 分阶段训练 |
| 论文实现错误 | 低 | 中 | 充分测试; 对照原始代码验证 |
| 人才短缺(Rust+AI) | 中 | 高 | 开源吸引; 实习生计划; 远程协作 |
| 大厂竞品发布 | 低 | 中 | 差异化(MoE+MLA+GRPO); 社区壁垒 |
