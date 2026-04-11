# Tasks: DeepSeek-3.2 + Gemma3/4 架构对齐与引擎优化

# Task Dependencies
- [Task 3] depends on [Task 1] (DeepSeekV3架构定义是MoE升级的前提)
- [Task 4] depends on [Task 1] (GEMM引擎抽象层需要先知道支持的架构)
- [Task 5] depends on [Task 2] (Arena部署依赖GEMM引擎)
- [Task 6] depends on [Task 2] (FP8依赖GEMM引擎)
- [Task 7] depends on [Task 1, 4] (视觉模块需要架构支持+GEMM后端)
- [Task 8] depends on [Task 5, 6] (设备适配依赖底层优化完成)
- [Task 9] depends on [Task 1, 2, 3, 4, 7] (配置统一依赖所有核心改动)

---

## Phase 1: 核心引擎优化（P0 - 立即执行）

- [x] **Task 1: DeepSeek-V3 架构定义与MoE策略配置化**
  - [x] 1.1 在 `gguf.rs` Architecture枚举中新增 `DeepSeekV3` 变体
  - [x] 1.2 实现 `parameter_prefix()` 返回 `"deepseek_v3"`
  - [x] 1.3 新增 `MoEStrategy` 枚举（Cyclic / FullLayer / Hybrid）
  - [x] 1.4 新增 `MoEWeightsV2` 结构体（共享专家+路由专家分离）
  - [x] 1.5 实现 `MoEWeightsV2::forward()` 方法（共享→路由→融合）
  - [x] 1.6 实现负载均衡损失计算函数 `compute_load_balance_loss()`
  - [x] 1.7 向后兼容：`MoEVersion` enum包装V1和V2
  - [x] 1.8 单元测试：DeepSeekV3枚举、MoE策略选择、权重加载

- [x] **Task 2: 高性能GEMM引擎抽象层**
  - [x] 2.1 定义 `GemmEngine` trait（matmul/batched_matmul/fused_gemm_relu/fused_gemm_silu）
  - [x] 2.2 实现 `CandleCpuBlasBackend`（基于candle-core + BLAS）
  - [x] 2.3 实现 `NdarrayFallbackBackend`（当前ndarray实现）
  - [x] 2.4 实现 `GemmEngineManager` 自动检测和选择逻辑
  - [x] 2.5 替换 `dsa.rs::lightning_indexer()` 使用新GEMM引擎
  - [x] 2.6 注册模块到 mod.rs

- [x] **Task 3: MLA Config动态化**
  - [x] 3.1 为 `ModelConfig` 新增 `latent_dim: Option<usize>` 字段
  - [x] 3.2 实现 `MLAConfig::from(&ModelConfig)` 转换
  - [x] 3.3 更新 `MLAProjection::new()` 接受动态config
  - [x] 3.4 更新 `RoPECache::new()` 使用动态参数
  - [x] 3.5 验证：不同规模模型（7B/14B/70B）的MLA config正确生成

---

## Phase 2: 内存与量化优化（P0）

- [x] **Task 4: Arena内存池热路径部署**
  - [x] 4.1 扩展 `Arena` trait 增加 `gemm()`, `gemm_compress()`, `fused_gemm_silu()` 方法
  - [x] 4.2 在 `TransformerModel` 中新增 `forward_with_arena()` 方法
  - [x] 4.3 重写Attention层使用arena分配临时张量
  - [x] 4.4 重写FFN层使用arena分配（含fused_gemm_silu）
  - [x] 4.5 重写MLA层使用arena分配（压缩KV缓存）
  - [x] 4.6 在推理入口 `inference.rs` 中创建并传递arena
  - [x] 4.7 性能测试：arena vs non-arena 的延迟和内存对比

- [x] **Task 5: FP8 KV Cache与激活量化**
  - [x] 5.1 定义 FP8 数据类型（E4M3/E5M2）
  - [x] 5.2 实现 FP32↔FP8 量化/反量化函数
  - [x] 5.3 修改 MLALatentCache 支持 FP8 存储 c_kv
  - [x] 5.4 修改 KVCache（标准attention）支持 FP8
  - [x] 5.5 在 FlashAttention-3 中启用 FP8 分支（检测Hopper架构）
  - [x] 5.6 配置项：`enable_fp8_kv_cache` 默认根据硬件自动决定
  - [x] 5.7 精度验证：FP8 vs FP32 的 perplexity 差异 < 0.5%

---

## Phase 3: 多模态引擎（P0-P1）

- [x] **Task 6: Gemma3 SigLIP 视觉编码器**
  - [x] 6.1 创建 `vision/mod.rs` 模块导出文件
  - [x] 6.2 实现 `SigLIPEncoderConfig` 配置结构体
  - [x] 6.3 实现 `SigLIPEncoder` 编码器结构体（patch_embedding + ViT layers）
  - [x] 6.4 实现 `SigLIPEncoder::encode()` 前向传播方法
  - [x] 6.5 实现 `ViTTransformerLayer`（self-attention + FFN + LayerNorm）
  - [x] 6.6 权重加载：从GGUF读取 `vision_model.*` 或 `siglip.*` 前缀权重
  - [x] 6.7 单元测试：编码器前向传播正确性

- [x] **Task 7: Gemma图像预处理器与多模态集成**
  - [x] 7.1 实现 `GemmaImageProcessorConfig`（896×896, bicubic, mean=[0.5,0.5,0.5]）
  - [x] 7.2 实现 `GemmaImageProcessor::preprocess()` （resize+normalize）
  - [x] 7.3 实现 `num_image_tokens()` 计算（(896/14)² = 4096 patches）
  - [x] 7.4 重构 `image_preprocess.rs` 支持多后端（ImageNet/Gemma/SigLIP）
  - [x] 7.5 修改 `inference.rs::generate_with_image()` 集成SigLIP pipeline
  - [x] 7.6 修改 `engine.rs::build_multimodal_prompt()` 支持视觉token数量可配
  - [x] 7.7 端到端测试：图像输入 → 预处理 → 编码 → 推理 → 输出

---

## Phase 4: 设备适配与配置统一（P1）

- [x] **Task 8: 个人终端设备自适应系统**
  - [x] 8.1 实现 `DeviceProfile` 检测（内存/GPU/CPU架构/OS）
  - [x] 8.2 实现 `DeviceType` 枚举（Desktop/Laptop/AppleSilicon/Mobile/Embedded）
  - [x] 8.3 实现各设备类型的推荐配置映射表
  - [x] 8.4 实现 layer offloading 策略（低内存时按需加载/卸载层权重）
  - [x] 8.5 Apple Silicon专项：Metal command buffer批处理优化
  - [x] 8.6 低内存模式自动降级逻辑（<8GB RAM触发）
  - [x] 8.7 设备检测单元测试 + 各平台集成测试

- [x] **Task 9: 统一配置系统**
  - [x] 9.1 扩展 `server.toml` 新增 `[architecture]`, `[moe]`, `[vision]`, `[engine]` 配置段
  - [x] 9.2 扩展 `settings.rs` ModelSettings 新增对应字段
  - [x] 9.3 实现配置到运行时的转换逻辑
  - [x] 9.4 配置验证：非法值拒绝启动并给出明确错误信息
  - [x] 9.5 向后兼容：旧配置格式仍可正常工作
  - [x] 9.6 配置文档更新（示例配置注释）

---

## Phase 5: 验证与性能基线（P1）

- [ ] **Task 10: 性能基准测试与回归防护**
  - [ ] 10.1 建立 multi-platform benchmark 测试矩阵（Mac M3/RTX4090/CPU-only）
  - [ ] 10.2 测试场景：短/中/长序列（128/512/2048/8192/16384 tokens）
  - [ ] 10.3 记录基线指标：TTFT/TPOT/Throughput/Memory/CPU&GPU利用率
  - [ ] 10.4 CI集成：PR自动跑轻量benchmark，退化>5%告警，>15%阻止合并
  - [ ] 10.5 与竞品对比数据收集（llama.cpp/MLX/Ollama同模型同硬件）
  - [ ] 10.6 性能报告模板生成
