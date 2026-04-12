# OpenMIini-V1 Public Roadmap

> ⚠️ 本路线图为公开版本，可能不包含最新内部规划。完整版本见内部 Wiki。

## 已完成 (✅)

### Phase 0: 核心基础设施 (2025 Q1)

- [x] FlashAttention-3 CPU实现 (FP8, SIMD AVX2/NEON)
- [x] Multi-Latent Attention (MLA) KV Cache压缩93%
- [x] Dynamic Sparse Attention (DSA) + Lightning Indexer
- [x] Speculative Decoding v2 (70-90% acceptance rate)
- [x] Continuous Batching (4种调度算法)
- [x] Paged KV Cache (COW语义)
- [x] 18种量化格式 (含IQ系列超压缩)
- [x] GRPO训练管线 (105个测试)

### Phase 1: 技术深化 (2025 Q1-Q2)

- [x] NSA (Native Sparse Attention) - DeepSeek-V3三路稀疏
- [x] Kascade (锚点层复用) - ~4x解码加速
- [x] Native Top-K Sparse (美团) - 15x Top-K加速
- [x] AMLA (整数加法替代浮点乘法)
- [x] CUDA Kernel基础设施 (cuBLAS + FA3 kernel)
- [x] 模型生态扩展 (12种架构支持)
- [x] Benchmark框架 (13场景 + CI集成)
- [x] 分布式推理原型 (TP 2卡/4卡)
- [x] LongCat双分支MoE (25-40%长序列加速)
- [x] Ring-flash-linear FP8混合注意力
- [x] 14B-Dense训练Pipeline
- [x] K8s部署方案 (Helm Chart)
- [x] 企业版功能 (OAuth2/RBAC/Audit/SLA)

---

## 进行中 🔄

### Phase 2: 差异化竞争 (2025 Q2-Q3)

- [ ] BlockFFN Chunk级MoE稀疏优化 (目标3.67x端侧加速)
- [ ] 70B-Dense模型预训练 (64xH100, 16周)
- [ ] TPA张量积注意力 (MLA替代方案探索)
- [ ] AHN RNN压缩+局部注意力 (>256K上下文)
- [ ] calm逐句生成新范式 (前瞻研究)
- [ ] 学术论文投稿 (OSDI/MLSys)
- [ ] 社区运营与生态建设

---

## 规划中 🔮

### Phase 3: 生态扩展 (2025 Q4+)

#### 模型规模演进

| 时间 | 模型 | 参数量 | 说明 |
|------|------|--------|------|
| Now | 7B-MoE | 7B (2B active) | 当前版本 |
| Q2'25 | 14B-Dense | 14B | 训练中 |
| Q3'25 | 70B-Dense | 70B | 规划中 |
| Q4'25 | 236B-MoE | 236B (21B active) | 远期目标 |

#### 平台能力

- **多模态增强**: 图像理解 → 图像生成 → 视频/Audio
- **Agent能力**: Tool Use → Function Calling → Multi-Agent
- **边缘部署**: Mobile (iOS/Android) → IoT → Automotive
- **开发者生态**: SDK → Plugin System → Fine-tuning Service

#### 学术研究

- [ ] Efficient LLM Inference (持续)
- [ ] Novel Attention Mechanisms
- [ ] Training Efficiency Optimization
- [ ] Safety & Alignment

---

## 愿景 🎯

### 2026 目标

1. **技术领先**: 至少1项技术达到SOTA水平
2. **模型竞争力**: OpenMini-236B 进入全球Top 10开源模型
3. **生态繁荣**: 100+ Contributors, 1000+ GitHub Stars
4. **产业落地**: 10+ 企业客户, 日调用量 1B+

### 长期愿景

> **让每一个设备都能运行高性能AI**

通过极致的软件优化和硬件协同设计，打破AI算力壁垒。
