# OpenMini-V1 混合架构重构 - 任务分解

## 📋 文档信息

| 属性 | 值 |
|------|-----|
| **关联Spec** | [spec.md](./spec.md) |
| **版本** | v1.0 |
| **总工期** | 14周 (约3.5个月) |
| **团队规模** | 建议2-3人 (1架构师+1-2后端) |

---

## Phase 1: 基础设施搭建 (第1-2周)

### 目标
建立 C/C++ 计算引擎的基础设施，验证 FFI 集成可行性

### 任务清单

#### 1.1 项目结构初始化
**优先级**: P0 | **预估**: 0.5天 | **负责人**: 架构师

- [ ] 创建 `openmini-server/native/` 目录
- [ ] 创建子目录结构:
  ```
  native/
  ├── include/          # 头文件
  ├── src/              # 源代码
  ├── kernels/          # 平台内核
  │   ├── cpu/
  │   ├── cuda/
  │   └── metal/
  ├── third_party/      # 第三方库
  │   └── llama.cpp/    # git submodule
  └── tests/            # 测试
  ```
- [ ] 初始化 `.gitignore` for native/

#### 1.2 CMake 构建系统配置
**优先级**: P0 | **预估**: 1天 | **负责人**: 架构师

- [ ] 编写根 `CMakeLists.txt`
  - 支持平台检测 (macOS/Linux/Windows)
  - 可选特性开关:
    - `OPENMINI_ENABLE_CUDA`
    - `OPENMINI_ENABLE_METAL`
    - `OPENMINI_ENABLE_BITNET`
- [ ] 配置编译选项
  - C++17 标准
  - Release + Debug 配置
  - 优化选项 (-O3, -march=native)
- [ ] 编写 macOS Metal 支持
- [ ] 编写 Linux CUDA 支持 (可选)

#### 1.3 集成 llama.cpp
**优先级**: P0 | **预估**: 2天 | **负责人**: 后端A

- [ ] 添加 llama.cpp 为 git submodule
  ```bash
  cd native/third_party
  git submodule add https://github.com/ggerganov/llama.cpp.git
  ```
- [ ] 配置 CMake 子目录引用
- [ ] 验证 llama.cpp 独立编译成功
- [ ] 测试基础模型加载功能

#### 1.4 定义 C API 接口
**优先级**: P0 | **预估**: 2天 | **负责人**: 架构师

- [ ] 编写 `include/openmini.h`
  - 类型定义 (句柄、错误码、配置)
  - 生命周期管理 API
  - 模型操作 API
  - 推理执行 API
  - 性能统计 API
- [ ] 编写 API 实现骨架 (`src/openmini_api.cpp`)
  - 函数桩实现 (返回 NOT_IMPLEMENTED)
  - 参数验证逻辑
  - 错误码定义

#### 1.5 Rust FFI 绑定层
**优先级**: P0 | **预估**: 2天 | **负责人**: 后端B

- [ ] 更新 `Cargo.toml`
  - 添加 build-dependencies: bindgen, cc
  - 移除 candle-core/candle-transformers (可选，渐进式)
- [ ] 编写 `build.rs`
  - 调用 cmake 构建 native 库
  - 调用 bindgen 生成 Rust 绑定
  - 输出链接参数
- [ ] 创建 `src/ffi/mod.rs`
  - FFI 绑定模块
  - 安全包装类型
  - 错误转换
- [ ] 创建 `src/ffi/types.rs`
  - Rust 类型映射
  - Config/Result 结构体

#### 1.6 集成测试: Hello World
**优先级**: P0 | **预估**: 1天 | **负责人**: 全员

- [ ] 编写端到端测试
  ```rust
  #[test]
  fn test_ffi_integration() {
      let engine = OpenMiniEngine::new("path/to/model.gguf", &config)?;
      let result = engine.generate("Hello", 10)?;
      assert!(!result.text.is_empty());
  }
  ```
- [ ] 验证编译通过 (cargo build)
- [ ] 验证运行成功 (cargo test)

### Phase 1 交付物
- ✅ `native/` 目录完整结构
- ✅ CMake 构建系统可工作
- ✅ llama.cpp 成功集成
- ✅ C API 定义完成
- ✅ Rust FFI 绑定可用
- ✅ 基础集成测试通过

### Phase 1 验收标准
- [ ] `cargo build --release` 编译成功
- [ ] `cmake --build native/build` 编译成功
- [ ] FFI 调用测试通过
- [ ] 内存无泄漏 (valgrind 检查)

---

## Phase 2: 核心引擎移植 (第3-6周)

### 目标
将核心推理功能移植到 C/C++ 引擎，实现完整的推理流程

### 任务清单

#### 2.1 模型加载器实现
**优先级**: P0 | **预估**: 3天 | **负责人**: 后端A

- [ ] 实现 GGUF 格式解析 (`src/model_loader.cpp`)
  - 读取文件头
  - 解析张量信息
  - 加载权重数据
  - 支持量化格式: Q4_0, Q4_K, Q8_0, I2_S
- [ ] 实现 Safetensors 格式支持 (可选)
- [ ] 实现模型元数据提取
  - 参数量、层数、头数等
  - 词表大小、特殊token
- [ ] 单元测试: 模型加载正确性

#### 2.2 Tokenizer 集成
**优先级**: P0 | **预估**: 2天 | **负责人**: 后端A

- [ ] 集成 sentencepiece 库
- [ ] 实现 tokenizer wrapper
  - encode: text → tokens
  - decode: tokens → text
- [ ] 支持 BPE / SentencePiece tokenizer
- [ ] 测试: 编解码一致性

#### 2.3 Transformer 核心层实现
**优先级**: P0 | **预估**: 5天 | **负责人**: 后端A

- [ ] 实现 Embedding 层
  - Token Embedding Lookup
  - Positional Encoding (RoPE)
- [ ] 实现 Attention 模块
  - Multi-Head Attention
  - Q/K/V 投影
  - Attention Score 计算
  - Softmax + Output 投影
- [ ] 实现 FFN 模块
  - Gate Projection (SwiGLU)
  - Up Projection
  - Down Projection
  - Activation Functions (SiLU, GeLU)
- [ ] 实现 RMSNorm / LayerNorm
- [ ] 组装 Transformer Block
- [ ] 循环执行 N 层

#### 2.4 KV Cache 管理
**优先级**: P0 | **预估**: 3天 | **负责人**: 后端A

- [ ] 实现 KV Cache 数据结构
  - Key Cache
  - Value Cache
  - Cache 序列长度管理
- [ ] 实现 PagedAttention (可选，Phase 3)
- [ ] 实现 Cache 更新逻辑
- [ ] 实现 Cache 清空/重置
- [ ] 内存优化: 按需分配

#### 2.5 推理循环实现
**优先级**: P0 | **预估**: 3天 | **负责人**: 后端A

- [ ] 实现 Prompt Processing (Prefill)
  - 批量处理 prompt tokens
  - 填充 KV Cache
  - 计算初始 hidden state
- [ ] 实现 Auto-regressive Decode
  - 循环: Last Token → Transformer → Logits → Sample → Next Token
  - EOS 检测
  - Max Tokens 限制
- [ ] 实现 Temperature / Top-K / Top-P Sampling
  - Logits 温度缩放
  - Top-K 过滤
  - Top-P (Nucleus) 采样
  - Random Seed 控制

#### 2.6 Rust 服务层适配
**优先级**: P0 | **预估**: 3天 | **负责人**: 后端B

- [ ] 重构 `InferenceEngine` trait
  - 适配新 FFI 接口
  - 保持向后兼容
- [ ] 更新 HTTP Handler
  - 使用新的 Engine 实现
  - 保持 API 兼容性
- [ ] 更新 Worker Pool
  - spawn_blocking 调用
  - 错误处理更新
- [ ] 更新 gRPC Handler (如有)

#### 2.7 端到端集成测试
**优先级**: P0 | **预估**: 2天 | **负责人**: 全员

- [ ] 功能测试
  - 模型加载测试
  - 单次推理测试
  - 多轮对话测试
  - 流式输出测试
- [ ] 正确性测试
  - 与 llama.cpp 输出对比
  - Perplexity 计算
  - 特定prompt输出校验
- [ ] 性能基线测试
  - 吞吐量 (tokens/s)
  - 延迟 (TTFT, TPOT)
  - 内存占用
- [ ] 回归测试套件建立

### Phase 2 交付物
- ✅ 完整的模型加载能力
- ✅ 可工作的推理流程
- ✅ 基本采样算法
- ✅ Rust 服务层集成
- ✅ 测试覆盖 >80%

### Phase 2 验收标准
- [ ] 可加载并推理 7B Q4 模型
- [ ] 输出与 llama.cpp 一致 (<1% 差异)
- [ ] CPU 推理速度 ≥15 t/s (7B Q4)
- [ ] 无内存泄漏 (长时间运行稳定)

---

## Phase 3: 性能优化 (第7-10周)

### 目标
引入高级优化技术，达到生产级性能水平

### 任务清单

#### 3.1 BitNet LUT 方法集成
**优先级**: P1 | **预估**: 4天 | **负责人**: 后端A

- [ ] 研究 BitNet TL1/TL2 内核实现
- [ ] 实现 LUT 查找表生成
  - 预计算 {-1,0,+1} × activation 的所有组合
  - 存储为查找表数组
- [ ] 实现 LUT GEMM 内核
  - ARM NEON 版本 (TL1)
  - x86 AVX2 版本 (TL2)
  - 通用回退版本
- [ ] 集成到量化管线
  - I2_S 格式原生支持
  - 无需反量化直接计算
- [ ] 性能对比测试

#### 3.2 平台专用内核优化
**优先级**: P1 | **预估**: 4天 | **负责人**: 后端A

**CPU 优化**:
- [ ] AVX2 GEMM 内核
  - 分块矩阵乘法
  - 缓存友好的内存访问
  - 向量化内积计算
- [ ] ARM NEON GEMM 内核
  - DOTPROD 扩展支持
  - 分块策略调优
- [ ] SIMD Softmax / LayerNorm
  - 数值稳定实现
  - 向量化归一化

**GPU 优化**:
- [ ] CUDA GEMM Kernel
  - cuBLAS 集成
  - 自定义 kernel (可选)
  - FP16/BF16 支持
- [ ] Flash Attention 实现
  - IO-aware attention
  - 在线 softmax
  - 因果掩码支持
- [ ] Metal Shader 优化
  - MSL 着色器编写
  - Threadgroup 内存优化
  - Texture 缓存利用

#### 3.3 量化深度支持
**优先级**: P1 | **预估**: 3天 | **负责人**: 后端A

- [ ] 扩展量化格式支持
  - Q2_K, Q3_K (极低比特)
  - IQ 系列 (IQ1_S, IQ2_S, IQ3_S)
  - FP8 量化 (实验性)
- [ ] 实现混合精度推理
  - 不同层使用不同精度
  - Attention: FP16/Q8
  - FFN: Q4/I2
  - Embedding: Q6_K
- [ ] 量化感知训练支持 (预留接口)

#### 3.4 内存优化
**优先级**: P1 | **预估**: 2天 | **负责人**: 后端A

- [ ] 权重内存映射 (mmap)
  - 延迟加载
  - 操作系统级缓存
- [ ] 激活值复用
  - In-place 操作
  - 内存池管理
- [ ] KV Cache 压缩
  - 量化存储 (Q8/F16)
  - 共享跨请求

#### 3.5 并发优化
**优先级**: P1 | **预估**: 2天 | **负责人**: 后端B

- [ ] Continuous Batching 支持
  - 批量 decode
  - 动态批大小
  - 请求调度优化
- [ ] 多实例并行
  - 多模型同时加载
  - GPU 显存分时复用
- [ ] 异步预处理
  - Tokenize 异步化
  - 结果后处理异步化

#### 3.6 性能基准测试与调优
**优先级**: P1 | **预估**: 3天 | **负责人**: 全员

- [ ] 建立标准 benchmark 套件
  - 参考 llama.cpp benchmark
  - 多模型多场景测试
- [ ] 性能剖析
  - 热点函数识别
  - 内存带宽分析
  - GPU利用率分析
- [ ] 参数调优
  - 分块大小
  - 线程数
  - Batch size
- [ ] 生成性能报告

### Phase 3 交付物
- ✅ BitNet LUT 支持
- ✅ 平台专用优化内核
- ✅ 完整量化支持
- ✅ 内存优化方案
- ✅ 并发处理能力
- ✅ 性能达到目标指标

### Phase 3 验收标准
- [ ] CPU 推理 ≥25 t/s (7B Q4, M2 Pro)
- [ ] GPU 推理 ≥80 t/s (7B Q4, A100)
- [ ] 首token延迟 <100ms
- [ ] 内存占用 <6GB (7B Q4)
- [ ] BitNet 2B 模型支持

---

## Phase 4: 生产就绪 (第11-14周)

### 目标
完善工程化细节，达到生产部署标准

### 任务清单

#### 4.1 错误处理与健壮性
**优先级**: P1 | **预估**: 2天 | **负责人**: 后端B

- [ ] 完善错误码体系
  - 所有错误路径覆盖
  - 有意义的错误消息
  - 日志关联
- [ ] 异常恢复机制
  - OOM 处理
  - 模型损坏检测
  - 超时保护
- [ ] 资源清理
  - Context 泄漏检测
  - 模型卸载安全
  - 线程安全保证

#### 4.2 监控与可观测性
**优先级**: P1 | **预估**: 2天 | **负责人**: 后端B

- [ ] Prometheus 指标导出
  - 推理延迟分布
  - 吞吐量统计
  - 错误率监控
  - 资源使用率
- [ ] 结构化日志
  - 请求追踪 ID
  - 性能关键路径日志
  - 错误堆栈记录
- [ ] 健康检查端点增强
  - 模型状态检查
  - 资源阈值告警
  - 就绪探针

#### 4.3 安全加固
**优先级**: P1 | **预估**: 2天 | **负责人**: 后端B

- [ ] 输入验证强化
  - Prompt 长度限制
  - 特殊字符过滤
  - 注入防护
- [ ] 资源限制
  - 最大并发数
  - 内存配额
  - CPU/GPU 时间片
- [ ] 敏感信息保护
  - 日志脱敏
  - 错误信息过滤

#### 4.4 文档编写
**优先级**: P2 | **预估**: 3天 | **负责人**: 全员

- [ ] 用户文档
  - 安装指南
  - 配置说明
  - 快速开始
  - FAQ
- [ ] 开发者文档
  - 架构设计文档
  - API 参考手册
  - 扩展指南
  - 贡献指南
- [ ] 运维文档
  - 部署指南
  - 性能调优手册
  - 故障排查指南
  - 监控配置

#### 4.5 压力测试与稳定性验证
**优先级**: P0 | **预估**: 3天 | **负责人**: 全员

- [ ] 长时间稳定性测试
  - 24小时连续运行
  - 内存泄漏检测
  - 性能衰减监测
- [ ] 高并发压力测试
  - 100+ 并发请求
  - 峰值负载测试
  - 恢复时间测试
- [ ] 边界条件测试
  - 超长输入
  - 空输入
  - 特殊字符
  - 极端参数

#### 4.6 发布准备
**优先级**: P0 | **预估**: 2天 | **负责人**: 架构师

- [ ] 版本号规划 (v2.0.0)
- [ ] CHANGELOG 编写
- [ ] Release Note 撰写
- [ ] 兼容性声明
- [ ] Docker 镜像构建
- [ ] CI/CD 流水线更新

### Phase 4 交付物
- ✅ 生产级错误处理
- ✅ 完善的监控系统
- ✅ 安全加固措施
- ✅ 完整文档体系
- ✅ 通过压力测试
- ✅ 发布就绪

### Phase 4 验收标准
- [ ] 24小时稳定运行无崩溃
- [ ] 100并发下P99延迟<500ms
- [ ] 内存泄漏=0
- [ ] 文档覆盖率>90%
- [ ] 安全扫描无高危问题

---

## 📊 总体进度视图

```
Week:  1   2   3   4   5   6   7   8   9   10  11  12  13  14
       ├───────┤─────────────────────┼────────────────┼──────────┤
Phase: │  P1   │        P2           │     P3         │    P4    │
       │基础设施│   核心引擎移植       │   性能优化      │ 生产就绪  │
       
Milestones:
  ▶ W2:  FFI集成验证通过
  ▶ W6:  核心推理功能可用  
  ▶ W10: 性能达到目标指标
  ▶ W14: v2.0.0 发布
```

---

## 👥 团队分工建议

| 角色 | 主要职责 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|---------|---------|---------|---------|---------|
| **架构师** | 设计、API定义、技术决策 | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ |
| **后端A (C/C++)** | 引擎开发、性能优化 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **后端B (Rust)** | 服务层集成、工程化 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 🔧 工具链要求

### 开发环境
- **操作系统**: macOS (Apple Silicon) 或 Linux (x86_64)
- **编译器**: Clang >= 15, GCC >= 11
- **构建工具**: CMake >= 3.22, Cargo latest
- **Python**: >= 3.9 (用于脚本和测试)
- **CUDA Toolkit**: >= 12.0 (如需GPU支持)

### 开发工具
- **IDE**: VS Code + rust-analyzer + C/C++ Extension
- **调试**: lldb/gdb, CUDA NSight (GPU)
- **性能分析**: perf, Instruments, nvprof
- **内存检测**: valgrind, AddressSanitizer
- **CI/CD**: GitHub Actions

### 依赖库
```yaml
C/C++依赖:
  - llama.cpp (submodule)
  - sentencepiece (tokenizer)
  - pthreads (线程)
  - dl (动态加载)
  -Metal.framework (macOS)
  - CUDA toolkit (NVIDIA GPU)

Rust依赖:
  - tokio (异步运行时)
  - axum (HTTP框架)
  - tonic (gRPC框架)
  - bindgen (FFI绑定生成)
  - serde (序列化)
```

---

## ⚠️ 风险与缓解

| 风险 | 影响 | 缓解措施 | 应急计划 |
|------|------|----------|----------|
| **llama.cpp API变更** | 高 | 锁定版本号 | Fork维护 |
| **FFI性能开销** | 中 | 批量操作 | Profiling优化 |
| **团队C++经验不足** | 高 | 培训+结对编程 | 外部咨询 |
| **构建复杂度增加** | 中 | CI自动化 | Docker统一环境 |
| **调试困难** | 中 | 详细日志 | Core dump 分析 |

---

**文档结束**
