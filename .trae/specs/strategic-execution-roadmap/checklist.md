# OpenMini-V1 三重战略执行路线图 - 检查清单

## P0 验收检查清单

### P0-1: NSA (Native Sparse Attention) 集成

- [ ] NSA 核心结构体 `NativeSparseAttention` 已定义
- [ ] TokenCompressor 压缩策略实现完成 (全局信息保留率>95%)
- [ ] TopKSelector 选择策略实现完成 (关键细节保留率>98%)
- [ ] SlidingWindowAttention 滑动窗口策略实现完成 (最近信息保留)
- [ ] NSAFusionLayer 三路融合层实现完成 (加权融合可配置)
- [ ] 与MLA协同选择逻辑已集成到推理引擎
  - [ ] 短序列(<4K) → FA3 路径验证通过
  - [ ] 中序列(4K-16K) → MLA 路径验证通过
  - [ ] 长序列(>16K) → NSA 路径验证通过
- [ ] NSA 单元测试全部通过 (>10个测试用例)
- [ ] 性能基准测试通过:
  - [ ] 32K序列延迟降低 >60% (vs DSA baseline)
  - [ ] 内存占用减少 >40%
  - [ ] 吞吐量提升 >2x

### P0-2: Kascade (锚点层复用稀疏注意力)

- [ ] `KascadeSparseAttention` 结构体已定义
- [ ] 锚点层选择DP算法实现正确 (最大化跨层相似度)
- [ ] 精确Top-K索引计算与缓存机制工作正常
- [ ] 三种复用策略全部实现:
  - [ ] Direct (直接复用最近锚点层)
  - [ ] Weighted (多锚点层加权插值)
  - [ ] Adaptive (相似度阈值自适应)
- [ ] Kascade集成到 `inference.rs` 推理引擎
- [ ] 训练无关性验证:
  - [ ] 无需SFT/微调即可使用
  - [ ] 对任意预训练模型开箱即用
- [ ] 解码注意力加速测试:
  - [ ] H100 GPU加速 >3x (接近论文4.1x数据)
  - [ ] 精度损失 <0.5% (vs 密集注意力)

### P0-3: Native Top-K Sparse Attention (美团)

- [ ] `native_top_k.rs` GPU kernel版本已创建
- [ ] 启发式top-k选择模式实现 (无需SFT):
  - [ ] 基于query norm的高激活位置选择
  - [ ] 基于key similarity的聚类选择
  - [ ] 精度保持 >98% (vs 密集注意力)
- [ ] Lightning Indexer升级为Native Top-K:
  - [ ] dsa.rs中的调用点已更新
  - [ ] 向后兼容 (旧DSA配置仍可用)
- [ ] 可选SFT微调接口就绪:
  - [ ] top-k SFT训练脚本提供
  - [ ] SFT后精度提升至99.5%+
- [ ] GPU kernel性能测试通过:
  - [ ] Top-K计算加速 >5x (vs CPU排序)
  - [ ] 长序列(16K+)收益最明显

### P0-4: AMLA (Addition代替Multiplication in FA)

- [ ] `amla_rescale()` 函数实现在 `flash_attention_3.rs`
- [ ] FP8指数差→整数加法转换逻辑正确
- [ ] SIMD优化整数加法版本 (AVX2/NEON):
  - [ ] x86 AVX2 版本测试通过
  - [ ] ARM NEON 版本测试通过
- [ ] AMLA模式开关功能正常:
  - [ ] 可在运行时切换 AMLA / 标准 FA3
  - [ ] 配置文件支持 `amla_enabled = true/false`
- [ ] 性能测试通过:
  - [ ] H100吞吐量提升 15-25%
  - [ ] 能耗降低 10-15%
  - [ ] 精度损失 <0.01% (数值等价性验证)

### P0-5: CUDA Kernel Phase 1

- [ ] CUDA上下文初始化与管理模块 (`kernel/cuda/mod.rs`)
  - [ ] CUDA Driver API加载
  - [ ] Device检测与选择
  - [ ] Context创建与销毁
- [ ] CUDA内存管理器 (`kernel/cuda/memory.rs`)
  - [ ] cudaMallocAsync 分配器
  - [ ] 内存池管理 (预分配+复用)
  - [ ] Host↔Device异步传输
- [ ] cuBLAS GEMM wrapper (`kernel/cuda/matmul.rs`)
  - [ ] FP32/FP16/BF16 GEMM 支持
  - [ ] Batched GEMM (多查询并行)
  - [ ] Strided GEMM (非连续内存)
- [ ] FlashAttention-3 CUDA kernel (`kernel/cuda/flash_attention.rs`)
  - [ ] 在线Softmax (Online Softmax) GPU实现
  - [ ] 分块矩阵乘法 (Tiling优化)
  - [ ] Hopper架构FP8 Tensor Core利用
- [ ] 量化GPU kernels (`kernel/cuda/quant.rs`)
  - [ ] Q4_K/Q5_K/Q6_K/Q8_0 反量化
  - [ ] IQ系列超压缩格式反量化
  - [ ] FP8/FP4格式支持
- [ ] 硬件自动检测与后端选择:
  - [ ] NVIDIA GPU检测 (CUDA版本/显存/算力)
  - [ ] Apple Metal检测 (M-series芯片)
  - [ ] CPU fallback逻辑 (无GPU时降级)
- [ ] 目标硬件性能达标:
  - [ ] RTX 4090 + Llama-3-8B Q4_K_M:
    - [ ] TTFT < 100ms ✅
    - [ ] TPOT < 10ms/token ✅
    - [ ] Throughput > 100 tokens/s ✅
    - [ ] GPU利用率 > 80% ✅

### P0-6: 模型架构生态扩展 (至15种)

- [ ] Mistral 架构支持 (`mistral` 前缀):
  - [ ] 配置参数解析正确 (embedding/head/block/layer)
  - [ ] Mistral-7B-v0.3 GGUF加载测试通过
  - [ ] 推理输出质量验证 (perplexity合理)
- [ ] Mixtral MoE 架构支持 (`mixtral` 前缀):
  - [ ] MoE参数解析 (8专家, TopK=2)
  - [ ] Mixtral-8x7B GGUF加载测试通过
  - [ ] MoE路由逻辑正确 (专家激活统计)
- [ ] Yi 系列架构支持 (`yi` 前缀):
  - [ ] Yi-6B/Yi-34B 参数解析
  - [ ] Yi-34B长上下文(200K)配置支持
  - [ ] GGUF加载测试通过
- [ ] **Phi系列参数读取修复**:
  - [ ] Phi-3/Phi-4 使用正确的`phi.*`前缀 (不再回退到llama)
  - [ ] Phi-3.5-4B GGUF加载测试通过
  - [ ] 推理输出与llama.cpp一致 (diff<0.5ppl)
- [ ] ChatGLM3 架构支持 (`chatglm` 前缀):
  - [ ] PrefixLM特殊处理 (bidirectional attention mask)
  - [ ] ChatGLM3-6B GGUF加载测试通过
- [ ] Baichuan 2 架构支持 (`baichuan` 前缀):
  - [ ] Parallel Attention结构识别
  - [ ] Baichuan2-7B/13B 加载测试通过
- [ ] **总计**: ≥10种新架构支持验证通过

### P0-7: 公开 Benchmark 与 CI性能回归

- [ ] Benchmark框架目录结构创建:
  ```
  benchmark/
  ├── models/          # 模型定义
  ├── scenarios/       # 测试场景
  ├── runners/         # 执行引擎
  └── results/         # 结果存储
  ```
- [ ] 核心数据结构实现:
  - [ ] `BenchmarkConfig` 完整
  - [ ] `ModelBenchmark` 含字段校验
  - [ ] `HardwareConfig` 自动检测
  - [ ] `BenchmarkScenario` 参数化
- [ ] 指标采集实现:
  - [ ] TTFT (Time To First Token) 测量
  - [ ] TPOT (Time Per Output Token) 测量
  - [ ] TBTL (Total Time By Time Limit) 测量
  - [ ] Throughput (Tokens/sec) 计算
  - [ ] GPU/CPU Memory 占用监控
  - [ ] Perplexity 质量评估
- [ ] 结果导出格式:
  - [ ] JSON (机器可读)
  - [ ] CSV (Excel友好)
  - [ ] Prometheus metrics format
  - [ ] Markdown报告生成
- [ ] CI集成:
  - [ ] GitHub Actions workflow文件创建
  - [ ] PR触发自动运行
  - [ ] main分支基线自动对比
  - [ ] 退化告警规则 (>5% warning, >15% block)
- [ ] 首次基准建立:
  - [ ] Llama-3-8B-Q4_K_M 基线数据记录
  - [ ] M2 Ultra / RTX 4090 双平台数据
  - [ ] 公开发布到项目Wiki/GitHub Releases

## P1 验收检查清单

### P1-1: 分布式推理原型 (2-4卡TP)

- [ ] 张量并行模块 (`distributed/tp.rs`) 创建
- [ ] 模型权重分片逻辑:
  - [ ] 按hidden_dim维度切分 (ColumnParallel)
  - [ ] 按attention_head切分 (RowParallel)
  - [ ] Q/K/V/O 各层分片策略正确
- [ ] NCCL/Gloo AllReduce集成:
  - [ ] AllReduce通信原语封装
  - [ ] 梯度同步正确性验证 (数值误差<1e-5)
  - [ ] 通信重叠计算优化
- [ ] 分布式Router:
  - [ ] gRPC请求分发到多卡Worker
  - [ ] 负载均衡 (RoundRobin/LeastLoaded)
  - [ ] Worker健康检查
- [ ] 多卡部署模板:
  - [ ] 2卡TP配置示例 (RTX 4090×2)
  - [ ] 4卡TP配置示例 (A100×4)
  - [ ] docker-compose多GPU编排
- [ ] Llama-3-70B 4xA100端到端测试:
  - [ ] 模型加载成功 (每卡~20GB)
  - [ ] TTFT < 200ms
  - [ ] Throughput > 40 tokens/s
  - [ ] 无死锁/通信错误

### P1-2: LongCat-Flash-Chat

- [ ] 双分支MoE结构 (`moe/longcat.rs`) 定义
- [ ] AuxiliaryBranch实现:
  - [ ] FFN → MLA → FFN 三阶段计算
  - [ ] 与主分支MoE独立执行
- [ ] ZeroExpert实现:
  - [ ] 占位专家 (不参与前向计算)
  - [ ] Router返回ZeroExpert时跳过
- [ ] 加权融合层:
  - [ ] α·Output_1 + β·Output_2 可配置
  - [ ] 融合系数可学习选项
- [ ] 与现有MoE路由器无缝集成
- [ ] 推理效率测试:
  - [ ] 低复杂度token识别率 >30%
  - [ ] 整体推理加速 25-40%
  - [ ] 精度损失 <1%

### P1-3: Ring-flash-linear-2.0

- [ ] Ring-flash-linear模块创建
- [ ] HybridAttnRatio枚举实现:
  - [ ] ThreeToOne (3L:1F)
  - [ ] FourToOne (4L:1F)
  - [ ] SevenToOne (7L:1F)
  - [ ] Adaptive (动态调整)
- [ ] FP8 KV Cache存储 (E4M3/E5M2)
- [ ] Linear Attention快速路径:
  - [ ] 无Softmax近似
  - [ ] 硬件友好的元素级乘加
- [ ] H100 FP8性能测试:
  - [ ] 吞吐量提升 40-60%
  - [ ] 显存占用降低 50%
  - [ ] 精度损失 <0.5%

### P1-4: 14B-Dense模型训练

- [ ] 14B模型配置文件完成 (`config/model_14b.toml`)
- [ ] 训练数据准备:
  - [ ] 通用文本500B tokens (CommonCrawl/Wikipedia/Books)
  - [ ] 代码100B tokens (TheStack/StarCode)
  - [ ] 中文200B tokens (CLUE/CPT)
  - [ ] 数学50B tokens (arXiv/MATH/GSM8K)
  - [ ] 数据清洗去重完成
- [ ] 继续预训练Pipeline搭建:
  - [ ] 从7B Checkpoint加载成功
  - [ ] DataLoader流式读取正常
  - [ ] Optimizer/Scheduler配置正确
  - [ ] Gradient Accumulation工作 (batch_size=256)
- [ ] 训练执行:
  - [ ] 3 epoch完成 (50K steps)
  - [ ] Loss曲线收敛正常
  - [ ] Checkpoint保存/恢复正常
  - [ ] 训练监控指标采集完整
- [ ] SFT微调完成:
  - [ ] 指令遵循能力提升明显
  - [ ] 对话质量人工评估通过
- [ ] GRPO RLHF对齐:
  - [ ] 使用已有GRPO管线
  - [ ] Reward Model训练完成
  - [ ] 策略优化收敛
- [ ] 基准测试评估:
  - [ ] MMLU: 68-72分 ✅
  - [ ] HumanEval: 45-52% ✅
  - [ ] GSM8K: 65-72% ✅
  - [ ] C-Eval: 70-75% ✅
  - [ ] 中文能力: 75-80分 ✅

### P1-5: K8s部署方案

- [ ] Helm Chart创建 (`deploy/helm/openmini/`)
- [ ] Dockerfile多阶段构建:
  - [ ] 构建镜像 < 100MB
  - [ ] 基础镜像: Alpine/Rust静态编译
- [ ] K8s资源清单:
  - [ ] Deployment (Pod模板)
  - [ ] Service (ClusterIP/NodePort/LoadBalancer)
  - [ ] ConfigMap (server.toml注入)
  - [ ] Secret (API Keys/证书)
- [ ] HPA (Horizontal Pod Autoscaler):
  - [ ] 基于CPU/GPU利用率自动扩缩容
  - [ ] 最小/最大副本数配置
- [ ] Prometheus+Grafana子Chart:
  - [ ] ServiceMonitor配置
  - [ ] Grafana Dashboard模板
  - [ ] 告警规则 (P0/P1/P2级别)

### P1-6: 企业版功能

- [ ] OAuth2/OIDC认证:
  - [ ] 支持Google/GitHub/企业IdP
  - [ ] JWT Token签发与验证
  - [ ] Refresh Token机制
- [ ] RBAC权限控制:
  - [ ] 角色: admin/operator/viewer/user
  - [ ] 资源: model/apikey/config/log
  - [ ] 操作: read/write/delete/admin
- [ ] 审计日志:
  - [ ] 所有操作记录不可篡改
  - [ ] 时间戳+操作者+资源+动作
  - [ ] 日志保留策略 (90天/1年/永久)
- [ ] SLA保障:
  - [ ] 可用性监控 (99.9%/99.99%目标)
  - [ ] 延迟P50/P99/P999告警
  - [ ] 错误率告警 (>0.1%/1%/5%)
- [ ] 多租户隔离:
  - [ ] Namespace级资源隔离
  - [ ] 租户配额限制
  - [ ] 数据隔离 (可选物理隔离)

## P2 验收检查清单 (前瞻性任务)

### P2-1: BlockFFN

- [ ] Chunk级MoE稀疏模块创建
- [ ] ReLU+RMSNorm可微路由器
- [ ] CLS (Chunk-Level Sparsity) 统计 >70%
- [ ] TLS保持 >80%
- [ ] Speculative Decoding兼容验证
- [ ] 端侧设备加速 >3x

### P2-2: 70B模型预训练

- [ ] 70B架构设计文档评审通过
- [ ] 算力资源到位 (64xA100或等价)
- [ ] 1T tokens数据流水线就绪
- [ ] 分布式训练配置 (FSDP/DeepSpeed)
- [ ] 预训练2 epoch完成
- [ ] SFT + GRPO对齐完成
- [ ] 全面基准测试:
  - [ ] MMLU: 78-82分
  - [ ] HumanEval: 58-65%
  - [ ] MT-Bench: 8.0-8.5/10

### P2-3-P2-7: 前瞻探索任务

- [ ] TPA原型实现并性能对比
- [ ] AHN超长上下文(256K+)原型验证
- [ ] calm逐句生成概念验证
- [ ] 至少1篇arXiv预印本发布
- [ ] Discord社区建立 (100+成员)
- [ ] Contributor Guide发布 (20+ Good First Issues)

## 最终验收标准

### 第一战场目标达成

- [ ] 平台评分 ≥ 89 (Q1) / ≥91 (Q2) / ≥94 (Q4)
- [ ] GitHub Stars ≥ 1K (Q1) / ≥5K (Q2) / ≥50K (Q4)
- [ ] Contributors ≥5 (Q1) / ≥20 (Q2) / ≥200 (Q4)
- [ ] 企业客户 ≥3 (Q2) / ≥50 (Q4)

### 第二战场目标达成

- [ ] 最大模型规模: 14B (Q1) / 70B (Q2) / 236B-MoE (Q4)
- [ ] MMLU: ≥72 (14B) / ≥82 (70B) / ≥90 (236B-MoE)
- [ ] HumanEval: ≥52% (14B) / ≥65% (70B) / ≥75% (236B-MoE)
- [ ] 中文能力: ≥80 (14B) / ≥88 (70B) / ≥92 (236B-MoE)

### 第三战场目标达成

- [ ] 论文跟进率: ≥80% (Q1) / ≥90% (Q2) / ≥95% (Q4)
- [ ] 原创技术数: ≥9项 (Q1) / ≥12项 (Q2) / ≥15项 (Q4)
- [ ] 发表论文: 0 (Q1) / ≥1篇 (Q2) / ≥2-3篇 (Q4)
- [ ] 技术领先月数: ≥6月 (Q1) / ≥9月 (Q2) / ≥12月 (Q4)
