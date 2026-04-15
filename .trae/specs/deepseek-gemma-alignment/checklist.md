# Checklist: DeepSeek-3.2 + Gemma3/4 架构对齐与引擎优化

## Phase 1: 核心引擎优化

* [x] DeepSeekV3 架构枚举在 gguf.rs 中正确定义，包含 parameter\_prefix() 返回 "deepseek\_v3"

* [x] MoEStrategy 枚举支持 Cyclic / FullLayer / Hybrid 三种模式

* [x] MoEWeightsV2 结构体正确实现共享专家+路由专家分离

* [x] MoEWeightsV2::forward() 方法实现：共享专家计算 → 路由专家top-k选择 → 加权融合

* [x] compute\_load\_balance\_loss() 函数正确计算辅助损失

* [x] MoEVersion enum 正确包装 V1（向后兼容）和 V2（新版）

* [x] GemmEngine trait 定义完整：matmul / batched\_matmul / fused\_gemm\_relu / fused\_gemm\_silu

* [x] CandleCpuBlasBackend 实现并可在 CPU+BLAS 环境正常工作

* [x] NdarrayFallbackBackend 实现作为兜底后端

* [x] GemmEngineManager 自动检测硬件并选择最优后端

* [x] dsa.rs 的 lightning\_indexer\_with\_engine() 已新增使用新 GEMM 引擎

* [x] model.rs 中关键路径的 .dot() 调用可通过引擎接口调用

* [x] MLAConfig 支持从 ModelConfig 动态生成（不再硬编码维度）

## Phase 2: 内存与量化优化

* [x] Arena 分配器扩展支持 gemm / gemm\_compress / fused\_gemm\_silu 方法

* [x] InferenceEngine 集成 Arena 字段并在 generate() 中使用 arena.reset()

* [x] Attention 层临时张量可使用 arena 分配（方法已就绪）

* [x] FFN 层可使用 arena 的 fused\_gemm\_silu（方法已就绪）

* [x] MLA 层压缩 KV 缓存可使用 arena 分配（方法已就绪）

* [x] 推理入口正确创建并传递 arena 实例（64MB）

* [x] FP8 数据类型定义完成（E4M3/E5M2）

* [x] FP32↔FP8 量化/反量化函数实现完毕

* [x] FP8 量化模块已注册到 inference/mod.rs

* [x] enable\_fp8\_kv\_cache 配置项已在 EngineSettings 中定义

## Phase 3: 多模态引擎

* [x] vision/mod.rs 模块导出文件创建完成

* [x] SigLIPEncoderConfig 配置结构体字段完整（image\_size=896, patch\_size=14, hidden\_size=1152, ...）

* [x] SigLIPEncoder 编码器结构体包含 patch\_embedding + position\_embedding + layers + layernorm

* [x] SigLIPEncoder::encode() 正确将 (H,W,3) u8 图像编码为 (num\_patches+1, hidden\_dim) f32 特征

* [x] ViTTransformerLayer 实现 self-attention + FFN(SwiGLU) + LayerNorm 前向传播

* [x] GemmaImageProcessorConfig 默认值正确（896×896, bicubic, mean=\[0.5,0.5,0.5]）

* [x] GemmaImageProcessor::preprocess() 正确执行 resize → normalize 流程

* [x] num\_image\_tokens() 对 896×896 图像返回 4096（(896/14)² patches）

* [x] image\_preprocess.rs 重构支持多后端选择（Standard/Gemma3）

* [x] inference.rs::generate\_with\_image() 集成 SigLIP 编码 pipeline

* [x] engine.rs::build\_multimodal\_prompt() 支持可配置视觉 token 数量

* [x] model.rs 新增 generate\_with\_visual\_features() 方法支持预计算视觉特征

## Phase 4: 设备适配与配置统一

* [x] DeviceProfile 正确检测设备类型、内存、CPU信息、OS、架构

* [x] DeviceType 覆盖 DesktopHighEnd / Laptop / AppleSilicon / MobileDevice / Embedded

* [x] 各设备类型有合理的推荐 RuntimeConfig 映射

* [x] 低内存设备自动降级策略（<8GB RAM 触发 fp8+offload+batch=1）

* [x] server.toml 新增 \[architecture] / \[moe] / \[vision] / \[engine] 配置段

* [x] settings.rs 包含 MoESettings / VisionSettings / EngineSettings 完整字段

* [x] 配置到运行时转换逻辑正确无误（serde default 向后兼容）

* [x] 旧格式配置文件仍可正常加载（向后兼容验证通过）

## Phase 5: 验证与性能基线

* [ ] multi-platform benchmark 测试矩阵建立（需实际硬件环境）

* [ ] 短/中/长序列基准数据记录（需实际运行）

* [ ] TTFT / TPOT / Throughput / Memory 指标采集（需实际运行）

* [ ] CI 自动 benchmark 集成（需 CI 环境配置）

* [ ] 与竞品对比数据收集（需同模型同硬件对比测试）

