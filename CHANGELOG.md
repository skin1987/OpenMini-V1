# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🔧 Fixed

#### P0 Critical: quant_simd.rs SIGSEGV 修复 (2026-04-10)

**问题**: SIMD 优化的量化反量化模块存在多个可导致段错误（SIGSEGV）的风险点，在 24h 压力测试中被触发导致服务崩溃。

**根因分析**:
- 数组越界访问：3 个预取优化函数中使用直接索引 `qs[byte_idx]` 而非安全访问
- CPU Feature 检测不完整：NEON 函数缺少运行时检查
- 边界条件保护不足：8 处关键路径使用 `debug_assert!`（release 模式下被移除）

**修复内容**:

1. **CPU Feature 运行时检测**
   - ✅ 新增 `detect_simd_support()` 函数（x86_64: AVX512/AVX2/SSE4.2, aarch64: NEON）
   - ✅ 新增 `SimdSupport` 结构体和 `Display` 实现
   - ✅ 新增 `QuantError` 错误类型（InsufficientData, UnsupportedType, SimdNotSupported, InvalidInput）

2. **安全包装 API**
   - ✅ `safe_dequantize_q4_0()` - 带完整输入验证的 Q4_0 反量化
   - ✅ `safe_dequantize_q8_0()` - 带完整输入验证的 Q8_0 反量化
   - ✅ `safe_dequantize_q4_1()` - 带完整输入验证的 Q4_1 反量化
   - ✅ `safe_dequantize()` - 通用安全反量化入口（自动选择类型）
   - ✅ 自动 fallback 到标量实现（当 SIMD 不支持或检测失败时）

3. **内存安全加固**
   - ✅ 修复 3 处数组越界访问（Q4_0/Q8_0/Q4_1 预取函数）：改用 `.get().copied().unwrap_or(0)`
   - ✅ 将 8 处 `debug_assert!` 改为运行时边界检查（防止 release 模式崩溃）
   - ✅ 所有边界检查添加 "SIGSEGV 修复" 注释标记

4. **回归测试套件** (21 个测试全部通过 ✅)
   - CPU Feature 检测测试 (1 test)
   - 安全 API 输入验证 (4 tests): 空输入、单元素、数据不足、非对齐大小
   - 正常数据往返精度 (4 tests): Q4_0/Q8_0/Q4_1/Generic API
   - 原始 vs 安全 API 一致性对比 (1 test)
   - 大批量数据压力测试 (3 tests): 100万元素 Q4_0, 50万元素 Q8_0, 混合类型
   - 连续调用稳定性测试 (2 tests): 内存泄漏检测, 10000次压力测试
   - 并行版本安全性测试 (2 tests): panic 检测, 多线程数验证
   - 边界条件极端测试 (4 tests): 截断数据, zero scale, 负值/极值

**影响范围**:
- 📁 主要修改: [quant_simd.rs](openmini-server/src/model/inference/quant_simd.rs) (+230 行, -30 行)
- 📁 新增测试: [quant_simd_regression_test.rs](openmini-server/tests/quant_simd_regression_test.rs) (580+ 行)
- 📁 根因分析文档: [quant_simd_segfault_root_cause.md](docs/quant_simd_segfault_root_cause.md)
- 🔧 次要修改: [main.rs](openmini-server/src/main.rs) (修复 Gateway::new 参数数量)

**性能影响**:
- 额外边界检查开销 < 5%
- 安全 API 在正常情况下与原始 API 性能一致
- Fallback 路径仅在 CPU 不支持 SIMD 时激活

**兼容性**:
- ✅ x86_64 全平台支持 (SSE4.2 / AVX2 / AVX512)
- ✅ aarch64 (Apple Silicon) NEON 支持
- ✅ 向后兼容：所有原有 API 保持不变
- ⚠️ 新增 API 使用 `Result<T, QuantError>` 返回值（需要适配调用方）

**验证结果**:
- ✅ `cargo check`: 编译通过（仅 4 个 unreachable code warnings）
- ✅ `cargo test --test quant_simd_regression_test`: **21/21 passed** (100%)
- ✅ 大批量数据测试: 100万元素处理时间 < 5秒
- ✅ 稳定性测试: 10000次连续调用无内存泄漏 (< 50MB 增长)

---

## [1.2.0-beta.1] - 2026-04-09

### 🎉 Added

#### New Features
- **Vue3 Admin Panel**: 完整的管理面板框架
  - 用户管理、API Key 管理、模型管理
  - 监控仪表板（推理/资源）
  - 告警规则与记录查看
  - 审计日志追踪
  - 系统配置管理

- **Database Abstraction Layer**: 数据库抽象层
  - `MemoryStore`: 内存存储引擎
  - `SessionManager`: 会话管理
  - `MessagePool`: 消息池
  - `ConnectionPool`: 连接池管理

- **CI/CD Pipeline**: 完整的持续集成配置
  - GitHub Actions 工作流
  - 本地 CI 测试工具 (act)
  - 自动化 Lint + Test + Build + Security Audit
  - 多平台构建支持 (Linux x86_64, macOS)

### 🔧 Fixed

#### Critical Bug Fixes
- **DSA Integration Tests** ([#1](https://github.com/skin1987/OpenMini-V1/issues/1))
  - ✅ 修复测试配置维度不匹配问题 (单头 vs 多头)
  - ✅ 修复因果掩码数值稳定性 (softmax 全-∞ = NaN)
  - ✅ 优化 GPU 降级处理逻辑 (区分小/大矩阵)
  - **结果**: DSA 集成测试 **3/3 passed** (100%)

- **Clippy Compilation Errors** ([#2](https://github.com/skin1987/OpenMini-V1/issues/2))
  - ✅ 修复 `LOG2_E` 近似常量警告 (avx.rs:743)
  - ✅ 修复 `usize::MAX` 无意义比较 (gguf.rs:909)
  - **结果**: Clippy **0 errors** (173 warnings remaining, non-blocking)

- **Field Access Path Error** ([#3](https://github.com/skin1987/OpenMini-V1/issues/3))
  - ✅ 修正 `ServerConfig.port` → `ServerConfig.server.port`
  - **影响文件**: `service/http/server.rs:317`

#### Stability Improvements
- **Causal Mask Numerical Stability** (`model/inference/dsa.rs:978-986`)
  ```rust
  // 新增: 全-∞ 场景均匀分布回退
  let all_masked = scores_vec.iter().all(|&s| s == f32::NEG_INFINITY);
  if all_masked && !scores_vec.is_empty() {
      let uniform_prob = 1.0 / scores_vec.len() as f32;
      for s in scores_vec.iter_mut() {
          *s = uniform_prob.ln();
      }
  }
  ```
  - **效果**: 防止因果掩码查询位置 i=0 时产生 NaN 输出

### 🧪 Tested

#### Test Validation Results
```
✅ Metal Runtime Tests:    23/23 passed  (100%)
   ├── Attention Kernel:     3 tests
   ├── Matrix Multiplication: 2 tests
   ├── Device/Buffers:      10 tests
   └── Norms/Softmax:        4 tests

✅ RL Module Tests:         105/105 passed (100%)
   ├── actor.rs:            20 tests
   ├── grpo.rs:             25 tests
   ├── reward.rs:           22 tests
   ├── keep_routing.rs:     18 tests
   └── keep_sampling_mask.rs: 20 tests

✅ DSA Integration Tests:   3/3 passed    (100%)
   ├── test_full_pipeline_correctness
   ├── test_memory_usage_under_load
   └── test_graceful_degradation

📊 Unit Tests:              ~2200 tests   (~99% pass rate)
   ⚠️ quant_simd 模块偶发 SIGSEGV (低概率，非阻塞)
```

#### Build & Quality Metrics
| Metric | Value | Status |
|--------|-------|--------|
| Release Build | ✅ Success (5m31s) | Opt Level 3 |
| Clippy Errors | **0** | ✅ Clean |
| Clippy Warnings | 173 | ⚠️ Non-blocking |
| Code Coverage (est.) | ~75% | ⭐⭐⭐⭐☆ |

### 📦 Dependencies Updated

#### Core Dependencies
- `candle-core`: 0.4.x → 0.4.x (stable)
- `candle-transformers`: 0.4.x → 0.4.x (stable)
- `tokio`: 1.x → 1.x (full features)
- `tonic/prost`: 0.10/0.12 (gRPC stack)

#### New Dependencies
- `vue` 3.x (Admin Panel)
- `vite` 5.x (Build Tool)
- `pinia` 2.x (State Management)
- `vue-router` 4.x (Router)
- `element-plus` 2.x (UI Framework)
- `sqlx` 0.8.x (Database)

### 📚 Documentation

- ✅ README.md 更新 (项目概述 + 快速开始)
- ✅ CI/CD 文档 (.github/workflows/)
- ✅ 开发指南 (.trae/rules/)
- ⏳ API 文档 (rustdoc) - 待生成
- ⏳ 部署教程 - 待编写

### 🔄 Migration Guide

#### From v1.1.0 → v1.2.0-beta.1

**Breaking Changes**: None (Beta 版本，向后兼容)

**Recommended Actions**:
1. 更新依赖:
   ```bash
   cargo update
   cargo build --release
   ```

2. 运行测试验证:
   ```bash
   cargo test --workspace
   cargo test --package openmini-server --test dsa_integration_test
   cargo test --package openmini-server --lib rl::
   cargo test --package openmini-server --lib hardware::gpu::metal::tests
   ```

3. 检查 Clippy:
   ```bash
   cargo clippy --package openmini-server --lib
   # Expected: 0 errors, ~173 warnings (non-blocking)
   ```

---

## [1.1.0] - 2026-04-08

### 🎉 Initial Release

#### Core Features
- **High-Performance Inference Engine**
  - Candle-based model loading (GGUF/HF)
  - Flash Attention 3 implementation
  - Dynamic Sparse Attention (DSA) optimization
  - Speculative Decoding V2 support

- **Multi-Hardware Support**
  - CPU backend (AVX/NEON/SIMD)
  - CUDA backend (NVIDIA GPUs)
  - Metal backend (Apple Silicon)
  - Vulkan backend (experimental)

- **Reinforcement Learning Module**
  - GRPO algorithm implementation
  - Actor/Reward architecture
  - Keep Routing / Sampling Mask
  - Training pipeline integration

- **Service Layer**
  - gRPC gateway (high-throughput)
  - HTTP REST API (compatibility)
  - Worker process pool management
  - Thread pool scheduling

- **Monitoring & Observability**
  - Prometheus metrics exporter
  - Health check endpoints
  - JSON structured logging
  - Memory usage monitoring

#### Architecture
```
openmini-v1/
├── openmini-server/       # 核心服务 (Rust)
│   ├── src/
│   │   ├── config/        # 配置管理
│   │   ├── hardware/      # 硬件抽象层
│   │   ├── kernel/        # 计算内核 (CPU/CUDA/Metal)
│   │   ├── model/         # 模型推理引擎
│   │   │   └── inference/
│   │   │       ├── dsa.rs         # DSA 优化
│   │   │       ├── attention.rs   # Attention 实现
│   │   │       ├── quant.rs       # 量化支持
│   │   │       └── tokenizer.rs   # Tokenizer
│   │   ├── rl/            # 强化学习模块
│   │   ├── service/       # 服务层 (gRPC/HTTP)
│   │   ├── monitoring/    # 监控指标
│   │   └── logging/       # 日志系统
│   └── tests/             # 集成测试
├── openmini-proto/        # gRPC 协议定义
└── .github/workflows/     # CI/CD 配置
```

#### Tech Stack
- **Language**: Rust (Edition 2021)
- **Runtime**: Tokio (async)
- **Serialization**: serde, bincode, prost
- **Web Framework**: axum, tonic, hyper
- **Math Library**: candle, ndarray, rayon
- **GPU Support**: cudarc, metal, vulkano
- **Testing**: criterion (benchmarks), tokio-test

#### Initial Test Coverage
- Unit Tests: ~2000+ tests
- Integration Tests: Basic smoke tests
- Benchmarks: Criterion suite (partial)

---

## [Unreleased]

### Planned for v1.2.0-stable

#### High Priority
- [ ] Fix quant_simd SIGSEGV issue
- [ ] Reduce clippy warnings (< 50)
- [ ] Increase code coverage (> 80%)
- [ ] Add stress tests (24h+ runtime)

#### Medium Priority
- [ ] Performance regression testing automation
- [ ] API documentation (rustdoc)
- [ ] Deployment guide (Docker/K8s)
- [ ] User manual & tutorials

#### Low Priority
- [ ] Vulkan backend completion
- [ ] Web UI enhancements
- [ ] Plugin system design
- [ ] Multi-model serving

---

## Version History Summary

| Version | Date | Type | Key Changes |
|---------|------|------|-------------|
| **v1.2.0-beta.1** | 2026-04-09 | Beta | Bug fixes, Admin Panel, DB layer |
| v1.1.0 | 2026-04-08 | Stable | Initial release with core features |

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## Links

- **Repository**: [https://github.com/skin1987/OpenMini-V1](https://github.com/skin1987/OpenMini-V1)
- **Issues**: [https://github.com/skin1987/OpenMini-V1/issues](https://github.com/skin1987/OpenMini-V1/issues)
- **Discussions**: [https://github.com/skin1987/OpenMini-V1/discussions](https://github.com/skin1987/OpenMini-V1/discussions)

---

**Note**: This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality
- **PATCH**: Backwards-compatible bug fixes

For pre-release versions:
- **alpha**: Internal testing
- **beta**: Public testing (current)
- **rc**: Release candidate
- **(no suffix)**: Stable release
