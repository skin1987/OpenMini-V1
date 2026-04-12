# OpenMini-V1 混合架构重构 - 检查清单

## 📋 文档信息

| 属性 | 值 |
|------|-----|
| **关联Spec** | [spec.md](./spec.md) |
| **关联Tasks** | [tasks.md](./tasks.md) |
| **版本** | v1.0 |
| **用途** | 逐项检查完成情况 |

---

## Phase 1: 基础设施搭建 检查

### 1.1 项目结构初始化
- [ ] `native/` 目录已创建
- [ ] `native/include/` 目录存在
- [ ] `native/src/` 目录存在
- [ ] `native/kernels/cpu/` 目录存在
- [ ] `native/kernels/cuda/` 目录存在 (可选)
- [ ] `native/kernels/metal/` 目录存在
- [ ] `native/third_party/` 目录存在
- [ ] `native/tests/` 目录存在
- [ ] `native/.gitignore` 已配置

### 1.2 CMake 构建系统
- [ ] 根 `CMakeLists.txt` 存在且可编译
- [ ] 支持平台检测 (macOS/Linux)
- [ ] `OPENMINI_ENABLE_CUDA` 选项可用
- [ ] `OPENMINI_ENABLE_METAL` 选项可用
- [ ] `OPENMINI_ENABLE_BITNET` 选项可用
- [ ] Release 构建配置正确 (-O3)
- [ ] Debug 构建配置正确 (-g)
- [ ] macOS Metal 链接配置正确
- [ ] Linux 线程库链接正确

### 1.3 llama.cpp 集成
- [ ] llama.cpp submodule 已添加
- [ ] `git submodule update --init --recursive` 成功
- [ ] llama.cpp 可独立编译通过
- [ ] CMake 正确引用 llama.cpp 子目录
- [ ] llama.cpp 头文件可被 native 代码引用

### 1.4 C API 接口定义
- [ ] `include/openmini.h` 存在
- [ ] 所有类型定义完整:
  - [ ] `openmini_context_t`
  - [ ] `openmini_model_t`
  - [ ] `openmini_config_t`
  - [ ] `openmini_result_t`
  - [ ] `openmini_stats_t`
  - [ ] `openmini_error_t` 枚举
- [ ] 生命周期 API 定义:
  - [ ] `openmini_init()`
  - [ ] `openmini_cleanup()`
  - [ ] `openmini_version()`
- [ ] 模型操作 API 定义:
  - [ ] `openmini_load_model()`
  - [ ] `openmini_free_model()`
  - [ ] `openmini_model_info()`
- [ ] 推理 API 定义:
  - [ ] `openmini_new_context()`
  - [ ] `openmini_free_context()`
  - [ ] `openmini_generate()`
  - [ ] `openmini_decode()`
  - [ ] `openmini_sample_token()`
- [ ] 统计 API 定义:
  - [ ] `openmini_get_stats()`
  - [ ] `openmini_reset_stats()`
- [ ] API 实现骨架 (`src/openmini_api.cpp`) 存在
- [ ] 所有函数有参数验证逻辑

### 1.5 Rust FFI 绑定层
- [ ] `Cargo.toml` 已更新:
  - [ ] `bindgen` 在 build-dependencies 中
  - [ ] `cc` 在 build-dependencies 中
- [ ] `build.rs` 存在且功能完整:
  - [ ] 调用 cmake 构建 native 库
  - [ ] 调用 bindgen 生成绑定
  - [ ] 输出正确的链接参数
  - [ ] macOS Metal framework 链接
- [ ] `src/ffi/mod.rs` 存在:
  - [ ] FFI bindings 模块声明
  - [ ] 安全包装结构体定义
  - [ ] 错误转换实现
- [ ] `src/ffi/types.rs` 存在:
  - [ ] Config 结构体映射
  - [ ] Result 结构体映射
  - [ ] Stats 结构体映射
- [ ] `src/lib.rs` 或相应位置导出 ffi 模块

### 1.6 集成测试
- [ ] 测试文件存在: `tests/test_ffi_integration.rs`
- [ ] 编译测试: `cargo build --release` 通过
- [ ] 单元测试: `cargo test` 通过
- [ ] FFI 调用测试用例:
  - [ ] `test_init_cleanup`
  - [ ] `test_version`
  - [ ] `test_load_model` (需要模型文件)
  - [ ] `test_generate_simple`
- [ ] 内存泄漏检测 (valgrind):
  - [ ] 无 definitely lost
  - [ ] 无 possibly lost (或可接受)

#### ✅ Phase 1 完成标准
**全部勾选后方可进入 Phase 2**

---

## Phase 2: 核心引擎移植 检查

### 2.1 模型加载器
- [ ] `src/model_loader.cpp` 存在
- [ ] GGUF 格式解析:
  - [ ] 文件头读取 (magic, version)
  - [ ] 超参数解析 (n_embd, n_head, n_layer, n_vocab)
  - [ ] 张量信息解析 (name, dims, type)
  - [ ] 权重数据加载
- [ ] 支持的量化格式:
  - [ ] Q4_0 / Q4_1
  - [ ] Q4_K / Q5_K / Q6_K
  - [ ] Q8_0
  - [ ] I2_S (BitNet)
  - [ ] F16 / F32
- [ ] 模型元数据提取 API 工作正常
- [ ] 错误处理:
  - [ ] 文件不存在 → 返回错误码
  - [ ] 格式错误 → 返回错误码
  - [ ] 内存不足 → 返回错误码
- [ ] 单元测试覆盖 >80%

### 2.2 Tokenizer 集成
- [ ] sentencepiece 库已集成
- [ ] `src/tokenizer_wrapper.cpp` 存在
- [ ] encode 功能: text → Vec<token_id>
- [ ] decode 功能: Vec<token_id> → text
- [ ] 支持 BPE tokenizer
- [ ] 支持 SentencePiece tokenizer
- [ ] 特殊 token 处理 (BOS, EOS, PAD)
- [ ] 编解码一致性测试通过

### 2.3 Transformer 核心
- [ ] Embedding 层:
  - [ ] Token Embedding Lookup
  - [ ] RoPE 位置编码实现
- [ ] Attention 模块:
  - [ ] Q/K/V 投影矩阵乘法
  - [ ] Attention Score 计算 (Q @ K^T / sqrt(d))
  - [ ] Softmax (数值稳定版本)
  - [ ] Attention Output (A @ V)
  - [ ] 输出投影
  - [ ] 多头注意力支持
  - [ ] 因果掩码 (Causal Mask)
- [ ] FFN 模块:
  - [ ] Gate 投影 (SwiGLU)
  - [ ] Up 投影
  - [ ] SiLU 激活函数
  - [ ] Down 投影
  - [ ] 残差连接
- [ ] 归一化:
  - [ ] RMSNorm 实现
  - [ ] LayerNorm 实现 (可选)
- [ ] Transformer Block 组装:
  - [ ] Pre-Norm 或 Post-Norm
  - [ ] N 层循环执行
  - [ ] 层间残差连接

### 2.4 KV Cache
- [ ] `src/kv_cache.cpp` 存在
- [ ] 数据结构:
  - [ ] Key Cache 分配
  - [ ] Value Cache 分配
  - [ ] 序列长度跟踪
- [ ] 操作:
  - [ ] Cache 写入 (新 token)
  - [ ] Cache 读取 (attention 时)
  - [ ] Cache 清空 (reset context)
  - [ ] Cache 扩展 (动态长度)
- [ ] 内存管理:
  - [ ] 按需分配
  - [ ] 最大容量限制
  - [ ] 释放机制

### 2.5 推理循环
- [ ] Prompt Processing (Prefill):
  - [ ] Tokenize prompt
  - [ ] 批量处理所有 prompt tokens
  - [ ] 填充 KV Cache
  - [ ] 返回初始 hidden state
- [ ] Auto-regressive Decode:
  - [ ] 循环直到 max_tokens 或 EOS
  - [ ] Last Token → Transformer
  - [ ] Logits 提取
  - [ ] Sampling
  - [ ] Next Token 输出
  - [ ] KV Cache 更新
- [ ] Sampling 算法:
  - [ ] Temperature 缩放
  - [ ] Top-K 过滤
  - [ ] Top-P (Nucleus) 采样
  - [ ] Random Seed 支持
  - [ ] Greedy 解码 (temp=0)
- [ ] EOS 检测和终止条件
- [ ] Max Tokens 限制

### 2.6 Rust 服务层适配
- [ ] `InferenceEngine` trait 更新:
  - [ ] 新的 FFI 实现类
  - [ ] 向后兼容旧接口
- [ ] HTTP Handler 更新:
  - [ ] 使用新的 Engine
  - [ ] `/v1/completions` 端点工作
  - [ ] `/v1/chat/completions` 端点工作
- [ ] Worker Pool 适配:
  - [ ] spawn_blocking 包装
  - [ ] 超时处理
  - [ ] 错误传播
- [ ] gRPC Handler 适配 (如有)

### 2.7 端到端测试
- [ ] 功能测试:
  - [ ] 模型加载测试 ✓
  - [ ] 单次推理测试 ✓
  - [ ] 多轮对话测试 ✓
  - [ ] 流式输出测试 ✓
- [ ] 正确性测试:
  - [ ] 与 llama.cpp 输出对比 (<1% 差异)
  - [ ] Perplexity 计算 (wiki文本)
  - [ ] 特定 prompt 校验集
- [ ] 性能基线:
  - [ ] 吞吐量记录 (t/s)
  - [ ] 延迟记录 (TTFT, TPOT)
  - [ ] 内存占用记录
- [ ] 回归测试套件可运行

#### ✅ Phase 2 完成标准
**全部勾选后方可进入 Phase 3**

---

## Phase 3: 性能优化 检查

### 3.1 BitNet LUT 方法
- [ ] LUT 内核代码存在:
  - [ ] `src/quant_lut.cpp`
  - [ ] `kernels/cpu/lut_gemm_tl1.cpp` (ARM)
  - [ ] `kernels/cpu/lut_gemm_tl2.cpp` (x86)
- [ ] LUT 表生成算法:
  - [ ] {-1,0,+1} × activation 预计算
  - [ ] 查找表数组存储
  - [ ] 支持可配置分块大小
- [ ] I2_S 格式原生支持:
  - [ ] 直接从量化权重计算
  - [ ] 无需反量化到 F32
- [ ] 性能对比:
  - [ ] vs 标准 GEMM 提升 >30%
  - [ ] 内存带宽使用降低

### 3.2 平台专用内核
**CPU 内核**:
- [ ] AVX2 GEMM:
  - [ ] 分块大小可配置
  - [ ] 向量化内积 (8 floats/cycle)
  - [ ] 缓存友好访问模式
- [ ] NEON GEMM:
  - [ ] DOTPROD 扩展支持
  - [ ] ARM 优化分块策略
- [ ] SIMD Softmax/LayerNorm:
  - [ ] 数值稳定
  - [ ] 向量化实现

**GPU 内核**:
- [ ] CUDA 支持 (如启用):
  - [ ] cuBLAS 集成
  - [ ] 自定义 kernel (可选)
  - [ ] FP16/BF16 支持
- [ ] Metal 支持 (macOS):
  - [ ] MSL 着色器编译
  - [ ] Threadgroup 内存利用
  - [ ] Texture 缓存优化
- [ ] Flash Attention:
  - [ ] IO-aware 实现
  - [ ] 在线 Softmax
  - [ ] 因果掩码支持

### 3.3 量化深度支持
- [ ] Q2_K / Q3_K 支持
- [ ] IQ 系列 (IQ1_S, IQ2_S, IQ3_S)
- [ ] FP8 实验 (可选)
- [ ] 混合精度推理:
  - [ ] 不同层不同精度
  - [ ] 配置化精度选择
- [ ] Embedding 量化 (Q6_K)

### 3.4 内存优化
- [ ] mmap 权重加载:
  - [ ] 延迟加载
  - [ ] OS 级缓存
- [ ] In-place 操作:
  - [ ] 激活值复用
  - [ ] 减少内存分配
- [ ] 内存池管理:
  - [ ] 预分配缓冲区
  - [ ] 复用临时内存
- [ ] KV Cache 量化存储

### 3.5 并发优化
- [ ] Continuous Batching:
  - [ ] 批量 decode API
  - [ ] 动态批大小
- [ ] 多实例并行:
  - [ ] 多模型同时加载
  - [ ] GPU 显存管理
- [ ] 异步预处理:
  - [ ] Tokenize 异步
  - [ ] 后处理异步

### 3.6 性能基准
- [ ] Benchmark 套件建立
- [ ] 性能剖析报告:
  - [ ] Top 10 热点函数
  - [ ] 内存带宽分析
  - [ ] GPU 利用率
- [ ] 参数调优记录:
  - [ ] 最优分块大小
  - [ ] 最优线程数
  - [ ] 最优 batch size
- [ ] 最终性能报告

#### ✅ Phase 3 完成标准
**性能指标达标后方可进入 Phase 4**

**目标指标**:
- [ ] CPU ≥25 t/s (7B Q4, M2 Pro)
- [ ] GPU ≥80 t/s (7B Q4, A100)
- [ ] TTFT <100ms
- [ ] Memory <6GB (7B Q4)

---

## Phase 4: 生产就绪 检查

### 4.1 错误处理与健壮性
- [ ] 所有错误路径有明确错误码
- [ ] 错误消息有意义且包含上下文
- [ ] OOM 处理: 优雅降级而非崩溃
- [ ] 模型损坏检测: 加载时校验
- [ ] 超时保护: 单次请求超时设置
- [ ] Context 泄漏检测: 自动清理
- [ ] 线程安全: 无数据竞争 (ThreadSanitizer)

### 4.2 监控与可观测性
- [ ] Prometheus 指标:
  - [ ] `openmini_inference_duration_seconds` (直方图)
  - [ ] `openmini_tokens_per_second` (gauge)
  - [ ] `openmini_requests_total` (counter)
  - [ ] `openmini_errors_total` (counter by type)
  - [ ] `openmini_memory_used_bytes` (gauge)
  - [ ] `openmini_gpu_memory_used_bytes` (gauge)
- [ ] 结构化日志:
  - [ ] 请求 ID 追踪
  - [ ] 关键路径耗时日志
  - [ ] 错误堆栈记录
- [ ] 健康检查:
  - [ ] GET /health 返回详细状态
  - [ ] 就绪探针 GET /ready
  - [ ] 模型状态包含

### 4.3 安全加固
- [ ] 输入验证:
  - [ ] Prompt 长度限制 (可配置)
  - [ ] 特殊字符过滤
  - [ ] JSON 注入防护
- [ ] 资源限制:
  - [ ] 最大并发数 (可配置)
  - [ ] 内存配额上限
  - [ ] CPU/GPU 时间片
- [ ] 敏感信息保护:
  - [ ] 日志中无 API Key
  - [ ] 错误响应无内部细节
  [ ] Prompt 不记录到日志 (可配置)

### 4.4 文档完整性
- [ ] 用户文档:
  - [ ] README.md 快速开始
  - [ ] INSTALL.md 安装指南
  - [ ] CONFIGURATION.md 配置说明
  - [ ] FAQ.md 常见问题
- [ ] 开发者文档:
  - [ ] ARCHITECTURE.md 架构设计
  - [ ] API.md API 参考
  - [ ] EXTENDING.md 扩展指南
  - [ ] CONTRIBUTING.md 贡献指南
- [ ] 运维文档:
  - [ ] DEPLOYMENT.md 部署指南
  - [ ] PERFORMANCE.md 性能调优
  - [ ] TROUBLESHOOTING.md 故障排查
  - [ ] MONITORING.md 监控配置

### 4.5 压力测试结果
- [ ] 24小时稳定性测试通过:
  - [ ] 无崩溃
  - [ ] 无内存泄漏增长
  - [ ] 无性能衰减 (>10%)
- [ ] 高并发测试:
  - [ ] 100 并发 P99 <500ms
  - [ ] 500 并发不崩溃
  - [ ] 错误率 <0.1%
- [ ] 边界条件:
  - [ ] 超长输入 (100K tokens) 处理正确
  - [ ] 空输入返回合理错误
  - [ ] 特殊字符不导致崩溃
  - [ ] 极端参数 (temp=0, top_k=1) 正常

### 4.6 发布准备
- [ ] 版本号: v2.0.0 (语义化版本)
- [ ] CHANGELOG.md 完整:
  - [ ] New Features 列表
  - [ ] Breaking Changes 说明
  - [ ] Bug Fixes 列表
  - [ ] Performance Improvements
- [ ] Release Note:
  - [ ] 升级指南
  - [ ] 兼容性声明
  - [ ] 已知问题
- [ ] Dockerfile 更新:
  - [ ] 多阶段构建
  - [ ] 最小镜像体积
  - [ ] 安全基础镜像
- [ ] CI/CD 流水线:
  - [ ] macOS 构建 + 测试
  - [ ] Linux 构建 + 测试
  - [ ] Release 构建流程
  - [ ] Docker 镜像构建

#### ✅ Phase 4 完成标准 = 项目发布就绪

---

## 🎯 总体验收检查

### 必须项 (Must Have)
- [ ] 混合架构编译成功 (Rust + C/C++)
- [ ] 7B Q4 模型可加载并推理
- [ ] 输出正确性验证通过
- [ ] CPU 推理速度 ≥25 t/s (M2 Pro) 或 ≥20 t/s (x86)
- [ ] GPU 推理速度 ≥80 t/s (如有GPU)
- [ ] 首token延迟 <150ms
- [ ] 内存占用 <8GB (7B Q4)
- [ ] 24小时稳定运行
- [ ] 无内存泄漏
- [ ] HTTP API 兼容 OpenAI 格式

### 应该项 (Should Have)
- [ ] BitNet 1.58-bit 模型支持
- [ ] Continuous Batching
- [ ] Prometheus 监控集成
- [ ] 结构化日志
- [ ] Docker 一键部署
- [ ] 完整用户文档
- [ ] 性能基准报告

### 可以有 (Nice to Have)
- [ ] Vulkan 后端支持
- [ ] 分布式推理支持
- [ ] Web UI 管理
- [ ] Python SDK
- [ ] 模型热加载

---

## 📊 进度追踪模板

```
日期: _______
当前Phase: □ P1  □ P2  □ P3  □ P4
本周完成任务数: ___ / ___
阻塞问题: _________________________
下周计划: _________________________

签名: _______
```

---

**文档结束**
