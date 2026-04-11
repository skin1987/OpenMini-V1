# 项目生产就绪度改进 Spec

## Why
OpenMini-V1 当前存在 **1547 处 unwrap() 调用**、**75% 测试覆盖率**、**173 个 Clippy 警告** 等关键问题，距离生产级部署有显著差距。本 Spec 旨在系统性地解决代码健壮性、测试完整性、架构优化等核心劣势，将项目从"技术演示"提升至"生产可用"水平。

## What Changes
- **P0-紧急修复**: 消除所有 panic 风险点（unwrap/expect），补充核心路径集成测试
- **P1-重要改进**: 简化架构复杂度，完善监控运维能力，更新依赖版本
- **P2-持续优化**: 性能调优，代码规范化，功能 Roadmap 对齐
- **P3-长期规划**: 前后端一致性，社区生态建设

### 影响范围
- **核心模块**: `openmini-server/src/` (89 个文件需修改)
- **测试套件**: `openmini-server/tests/` (新增 20+ 测试文件)
- **配置系统**: `config/server.toml`, CI/CD workflows
- **依赖管理**: `Cargo.toml`, `package.json`
- **文档体系**: README, API 文档, 运维手册

---

## ADDED Requirements

### Requirement: 错误处理规范化 (P0)
系统 SHALL 将所有 `unwrap()` / `expect()` 调用替换为安全的错误处理模式：
- 使用 `?` 操作符传播错误
- 使用 `ok_or_else()` / `context()` 提供上下文信息
- 对不可恢复错误使用 `panic!` 并添加明确日志
- 全局禁用 `#![allow(unused_imports)]` 等 lint 规则

#### Scenario: KV Cache 维度验证失败
- **WHEN** 调用 `KvCacheLayer::update()` 时传入维度不匹配的数据
- **THEN** 返回 `Result<(), EngineError>` 而非 panic
- **AND** 错误信息包含期望维度和实际维度

#### Scenario: Worker 进程通信失败
- **WHEN** Worker stdin/stdout pipe 断开
- **THEN** 返回 `WorkerPoolError::CommunicationError` 
- **AND** 自动标记 Worker 为 DEAD 状态并触发重启

---

### Requirement: 测试覆盖率提升至 85%+ (P0)
系统 SHALL 补充以下关键测试场景：

#### 单元测试补充
- [ ] `training/` 模块：autograd 反向传播、optimizer 梯度裁剪、checkpoint 序列化
- [ ] `hardware/gpu/` 模块：CUDA 内存分配、Metal shader 编译、Vulkan 命令缓冲
- [ ] `service/grpc/` 模块：流式响应、拦截器链、负载均衡策略
- [ ] `model/inference/` 模块：GGUF 加载边界、量化精度损失、tokenizer 特殊 token

#### 集成测试补充
- [ ] **并发压力测试**: 1000 QPS 持续 5 分钟，验证无内存泄漏
- [ ] **GPU 后端完整流程**: 模型加载 → 推理 → 卸载（CUDA/Metal）
- [ ] **Worker 故障恢复**: Kill Worker 进程后验证自动重启 + 任务迁移
- [ ] **配置容错测试**: 缺失配置项、非法值、权限不足等场景

#### E2E 测试补充
- [ ] 完整请求生命周期：HTTP → Gateway → Worker → 模型 → 响应
- [ ] gRPC 流式推理端到端延迟 < 100ms (P99)
- [ ] Admin Panel CRUD 操作完整性

#### 性能基准测试
- [ ] Token 吞吐量回归检测（baseline ±5%）
- [ ] 内存使用率监控（长时间运行无泄漏）
- [ ] KV Cache defrag 性能影响量化

---

### Requirement: 架构简化与性能优化 (P1)

#### 架构调整
**当前问题**: 4 层资源池（Worker Pool → Thread Pool → Core Router → Async Pool）导致调度复杂度高

**目标架构**:
```
Simplified Architecture:
┌─────────────┐
│  HTTP/gRPC  │ ← Axum/Tonic (异步 I/O)
└──────┬──────┘
       │
┌──────▼──────┐
│ Task Scheduler│ ← Tokio Runtime (工作窃取)
└──────┬──────┘
       │
┌──────▼──────┐
│ Inference   │ ← Candle (单进程多线程)
│ Engine      │
└─────────────┘
```

**变更内容**:
- 移除 `WorkerPool` 多进程模型，改用 `tokio::task::spawn_blocking` 处理 CPU 密集型任务
- 合并 `ThreadPool` 和 `CoreRouter` 为统一的 `TaskScheduler`
- 保留 `AsyncInferencePool` 用于请求排队和批处理
- **BREAKING**: 配置格式变更（移除 `[worker]` section）

#### 性能优化项
1. **Softmax SIMD 向量化**: 使用 `std::arch` intrinsics 或 `packed_simd` crate
2. **KV Cache 零拷贝读取**: 实现 `Cow<Array2<f32>>` 借用机制
3. **Bincode 替换**: 考虑使用 `capnp` 或 `flatbuffers` 减少序列化开销
4. **锁粒度优化**: 将 `Mutex<Option<Child>>` 拆分为读写锁或使用 `RwLock`

---

### Requirement: 依赖安全与版本更新 (P1)

#### Rust 依赖升级路径
```toml
# 当前版本 → 目标版本
candle-core = "0.4" → "0.7"        # 性能提升 + bug 修复
candle-transformers = "0.4" → "0.7"
tonic = "0.10" → "0.12"            # gRPC streaming 改进
prost = "0.12" → "0.13"
axum = "0.7" → "0.8"               # 中间件增强
tower-http = "0.5" → "0.6"
sqlx = "0.8" → "0.8"               # 已是最新 ✅
```

#### 前端依赖修正
```json
{
  "vite": "^8.0.4" → "^6.0.0",     // 版本号异常修正
  "typescript": "~6.0.2" → "~5.7.2", // 版本号异常修正
  "vue": "^3.5.32" → "^3.5.x"       // 保持最新稳定版
}
```

#### 安全加固
- [ ] 启用 `cargo-deny` 检查依赖许可证合规性
- [ ] CI/CD 中 `cargo audit` 失败时**阻断构建**（移除 `continue-on-error: true`）
- [ ] 定期自动 PR：Dependabot / Renovate Bot 配置

---

### Requirement: 监控运维能力增强 (P1)

#### 可观测性栈
1. **分布式追踪**
   - 集成 OpenTelemetry Rust SDK
   - 自动追踪 gRPC/HTTP 请求全链路
   - 导出至 Jaeger/Tempo 后端

2. **结构化日志增强**
   - 统一使用 `tracing` 的 `#[instrument]` 宏
   - 关键操作添加 `Span` 上下文（request_id, session_id, model_name）
   - 日志采样：开发环境 DEBUG，生产环境 INFO（ERROR 始终记录）

3. **业务指标**
   ```rust
   // 新增 Prometheus 指标
   openmini_inference_tokens_total{model, status}    // token 吞吐量
   openmini_request_duration_seconds{endpoint, method} // P50/P95/P99
   openmini_kv_cache_usage_bytes{layer}              // KV Cache 内存占用
   openmini_worker_queue_length                      // 请求队列深度
   ```

4. **健康检查增强**
   - `/health/ready`: 就绪探针（依赖服务可用性）
   - `/health/live`: 存活探针（进程状态）
   - `/health/model`: 模型加载状态（已加载/加载中/卸载）

#### 配置管理
- [ ] 引入 `config` crate 的 schema validation（使用 `schemars`）
- [ ] 支持环境变量覆盖（`OPENMINI__SERVER__PORT=8080`）
- [ ] 配置热重载：监听文件变化 + SIGHUP 信号触发
- [ ] 敏感信息脱敏：日志中不打印 API Key、密码等字段

---

### Requirement: 代码规范化 (P2)

#### Lint 规则严格化
```rust
// Cargo.toml workspace.lints
[workspace.lints.rust]
missing_docs = "warn"          // 从 allow 改为 warn
dead_code = "warn"             // 从 allow 改为 warn
unused_imports = "warn"        // 从 allow 改为 warn

[workspace.lints.clippy]
all = "warn"                   // 启用所有 clippy 规则
unwrap_used = "deny"           // 禁止 unwrap（核心规则）
expect_used = "deny"           // 禁止 expect
panic = "deny"                 // 禁止显式 panic（允许在 main.rs 和 tests 中）
```

#### 错误类型统一
- 创建 `crate::error::AppError` 枚举，统一所有模块错误
- 实现 `From<E> for AppError` trait 以便 `?` 自动转换
- 使用 `thiserror` 的 `#[error("...")]` 属性生成友好错误消息
- 公共 API 返回 `Result<T, AppError>` 而非 `Box<dyn Error>`

#### 文档注释补全
- 所有 `pub` 函数添加 `///` 文档注释
- 包含 `# Examples`、`# Panics`、`# Errors` 章节
- 运行 `cargo doc --no-deps` 确保文档编译通过

---

### Requirement: 功能完整性对齐 Roadmap (P2)

#### v1.2.0-stable 必须完成项
1. **修复 quant_simd SIGSEGV**
   - 根因分析：SIMD 指令集检测不充分或内存对齐问题
   - 修复方案：添加运行时 CPU feature detection + fallback path
   - 验证：24h 压力测试零崩溃

2. **Clippy Warnings < 50**
   - 当前 173 个警告，分批消除
   - 优先处理 `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic`

3. **Code Coverage > 80%**
   - 目标 85%，重点覆盖 service/ 和 model/ 目录
   - 排除 `#[cfg(test)]` 和 benchmark 代码

#### 未来版本预研
- Vulkan 后端：评估是否值得继续投入（vs 直接支持 ROCm）
- 分布式推理：调研 RayOn/Gloo 等框架可行性
- 模型并行ism：需要 Candle 上游支持（FSDP/DeepSpeed equivalent）

---

### Requirement: 前后端一致性保障 (P3)

#### 类型安全共享
- 方案 A（推荐）：使用 `ts-rs` crate 从 Rust 类型自动生成 TypeScript
  ```rust
  #[derive(Serialize, Deserialize, Type)]
  pub struct ModelInfo {
      pub name: String,
      pub quantization: QuantizationType,
      pub parameters: u64,
  }
  // 自动生成: interface ModelInfo { name: string; quantization: QuantizationType; parameters: number; }
  ```
- 方案 B：维护独立的 OpenAPI spec（Swagger/YAML），前后端各自生成代码

#### API 版本管理
- gRPC 服务添加 `api-version` header
- HTTP REST 使用 URL versioning (`/v1/completions`, `/v2/completions`)
- Breaking changes 通过 Feature Flag 渐进式发布

---

### Requirement: 社区生态建设 (P3)

#### 开发者体验
- [ ] 添加 `CONTRIBUTING.md`（修复 CHANGELOG 中引用的缺失文件）
- [ ] 创建 `examples/` 目录：Python SDK、curl 命令、Docker Compose 示例
- [ ] 编写 `docs/architecture.md`：架构决策记录（ADR）
- [ ] 设置 GitHub Discussions 模板：Q&A、Feature Request、Show & Tell

#### 用户文档
- [ ] 快速入门视频教程（5 分钟部署）
- [ ] 性能调优指南（针对不同硬件配置）
- [ ] 故障排查手册（常见错误码及解决方案）
- [ ] API 参考（自动生成的 rustdoc + Swagger UI）

---

## MODIFIED Requirements

### Requirement: CI/CD Pipeline 增强
原有 CI/CD 仅做基本的 lint/test/build，现扩展为：

#### 新增 Job
1. **Security Scan**（强制失败）
   ```yaml
   - name: Run security audit
     run: cargo audit --db https://crates.io/api/v1/crates
     # 移除 continue-on-error: true
   ```

2. **Code Coverage Gate**
   ```yaml
   - name: Check coverage threshold
     run: |
       COVERAGE=$(cargo tarpaulin --workspace --out Xml | grep 'Total')
       if (( $(echo "$COVERAGE < 80" | bc -l) )); then
         echo "Coverage $COVERAGE% is below 80% threshold"
         exit 1
       fi
   ```

3. **Performance Regression Test**
   ```yaml
   - name: Benchmark comparison
     if: github.event_name == 'pull_request'
     run: |
       cargo bench --bench inference_benchmark -- --save-baseline pr
       cargo critcmp pr main  # 允许 ±5% 波动
   ```

4. **Documentation Build**
   ```yaml
   - name: Build documentation
     run: cargo doc --no-deps --document-private-items
   ```

---

## REMOVED Requirements

### Requirement: 多进程 Worker Pool（已废弃）
**原因**: IPC 通信开销大（每次任务 ~2ms latency），调试困难，与 Tokio 异步模型冲突

**迁移方案**:
1. 保留 `WorkerPool` 代码但标记为 `#[deprecated]`
2. 新代码统一使用 `TaskScheduler`（基于 tokio::sync::mpsc）
3. 提供 `config.migration_guide.md` 说明配置变更
4. 在 v2.0.0 正式移除（遵循 SemVer breaking change 规范）

---

## Success Metrics

| 指标 | 当前值 | 目标值 | 测量方法 |
|------|--------|--------|----------|
| `unwrap()` 调用数 | 1547 | **0** | `grep -r "unwrap()" src/ \| wc -l` |
| Clippy Warnings | 173 | **< 50** | `cargo clippy 2>&1 \| grep "warning"` |
| Code Coverage | ~75% | **> 85%** | `cargo tarpaulin` |
| 集成测试数量 | 8 | **> 25** | `ls tests/*.rs \| wc -l` |
| 依赖过期数 | 5 | **0** | `cargo outdated` |
| 文档覆盖率 | ~40% | **> 90%** | `cargo doc --no-deps` 无 warning |
| P99 延迟 | 未测量 | **< 100ms** | Prometheus histogram |

---

## Risk Assessment

### 高风险项
1. **Candle 升级 0.4 → 0.7**: 可能引入 API breaking changes
   - 缓解措施：先在分支测试，保留降级方案
   
2. **架构简化（移除 Worker Pool）**: 可能影响现有用户配置
   - 缓解措施：提供自动配置迁移工具 + 详细 migration guide

3. **Clippy deny 规则**: 可能使构建暂时失败
   - 缓解措施：分阶段启用，先 warn 后 deny

### 时间估算
- **P0 任务**: 2-3 周（全职 1 人）
- **P1 任务**: 3-4 周（可与 P0 并行部分工作）
- **P2 任务**: 4-6 周（持续迭代）
- **P3 任务**: 长期（社区共建）

**总计**: 约 **10-13 周** 达到 production-ready 状态
