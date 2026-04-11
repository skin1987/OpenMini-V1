# 项目生产就绪度改进 - 验证清单

## 阶段一：P0 紧急修复验证

### 错误处理规范化
- [ ] `crate::error::AppError` 枚举已定义并包含所有模块错误变体
- [ ] `model/inference/engine.rs` 中 `KvCacheLayer::update()` 返回 `Result<(), AppError>`
- [ ] `service/worker/pool.rs` 中 0 处 unwrap() 调用（grep 验证）
- [ ] `training/` 模块中 0 处 unwrap() 调用（grep 验证）
- [ ] `hardware/` 模块中 0 处 unwrap() 调用（grep 验证）
- [ ] `main.rs` 中已移除 `#![allow(unused_imports)]` 等 lint allow
- [ ] 全项目 unwrap() 总数 < 50（仅允许 test 代码中使用）
- [ ] `cargo clippy --workspace --lib` 输出 **0 errors**, **< 100 warnings**

### 测试覆盖率提升
- [ ] `training/autograd.rs` 反向传播测试通过（梯度检查精度 < 1e-5）
- [ ] `training/optimizer.rs` AdamW 更新规则测试覆盖所有参数组合
- [ ] `training/checkpoint.rs` 序列化/反序列化往返测试（save → load → verify）
- [ ] `hardware/gpu/cuda.rs` CUDA 内存分配/释放无泄漏（valgrind/cuda-memcheck）
- [ ] `hardware/gpu/metal.rs` Metal shader 编译 + 执行测试通过
- [ ] `service/grpc/server.rs` 流式响应测试（多消息、错误中断、超时）
- [ ] `model/inference/gguf.rs` 边界测试：空文件返回错误、损坏文件 panic-free
- [ ] 并发压力测试：1000 QPS × 5min 无崩溃、内存稳定
- [ ] GPU E2E 测试：CUDA/Metal 完整流程（加载 → 推理 → 卸载）成功
- [ ] Worker 故障恢复测试：Kill 后 < 5s 自动重启 + 任务迁移成功
- [ ] 配置容错测试：缺失配置项使用默认值、非法值返回明确错误
- [ ] E2E 生命周期测试：HTTP/gRPC 完整请求链路 P99 < 200ms

### quant_simd SIGSEGV 修复
- [ ] 根因分析文档已完成（包含 AddressSanitizer 日志）
- [ ] CPU feature detection 运行时检查已实现
- [ ] 不支持 SIMD 时自动 fallback 至标量实现
- [ ] 24h 压力测试零崩溃（运行 `cargo test --test stress_test -- --nocapture`）
- [ ] CHANGELOG 已更新并引用相关 Issue

---

## 阶段二：P1 重要改进验证

### 架构简化验证
- [ ] ADR 文档 `docs/architecture/001-simplify-worker-pool.md` 已创建
- [ ] `service/scheduler/mod.rs` 已实现统一调度器接口
- [ ] ThreadPool 和 CoreRouter 已合并至 TaskScheduler
- [ ] AsyncInferencePool 使用 TaskScheduler 分发任务
- [ ] WorkerPool/WorkerHandle 标记为 `#[deprecated]`
- [ ] 配置文件 `[worker]` section 已移除或标记废弃
- [ ] `[thread_pool].size` 默认值已调整为 CPU 核心数
- [ ] 迁移指南 `docs/migration_worker_pool_to_scheduler.md` 已编写

### 性能优化验证
- [ ] Softmax SIMD 实现基准测试：比标量版本快 **≥3x**（AVX2）/**≥2x**（NEON）
- [ ] KV Cache get() 零拷贝读取：内存分配减少 **≥50%**
- [ ] 序列化方案 benchmark 报告已完成（bincode vs capnp vs flatbuffers）
- [ ] WorkerHandle 锁竞争降低：pprof 显示 Mutex 等待时间 **< 1ms**
- [ ] 整体吞吐量提升：criterion benchmark 对比 baseline **+15%~25%**

### 依赖版本升级验证
- [ ] candle-core 版本 ≥ 0.7.0
- [ ] tonic 版本 ≥ 0.12.0
- [ ] axum 版本 ≥ 0.8.0
- [ ] 前端 vite 版本 = ^6.x（非 ^8.x）
- [ ] 前端 typescript 版本 = ~5.x（非 ~6.x）
- [ ] `cargo test --workspace` 全部通过（无 breaking changes）

### 监控运维增强验证
- [ ] OpenTelemetry SDK 已集成（tracing-opentelemetry 在 Cargo.toml）
- [ ] gRPC/HTTP handler 已添加 `#[instrument]` 注解（覆盖率 > 80%）
- [ ] Prometheus 业务指标可查询：
  - `openmini_inference_tokens_total`
  - `openmini_request_duration_seconds`
  - `openmini_kv_cache_usage_bytes`
  - `openmini_worker_queue_length`
- [ ] Health check endpoints 响应正常：
  - GET `/health/ready` → 200 (依赖服务正常) / 503 (依赖异常)
  - GET `/health/live` → 200 (进程存活)
  - GET `/health/model` → 200 (模型已加载) / 503 (未加载)
- [ ] 配置 schema validation 生效：非法配置启动时报错并提示修正位置
- [ ] 敏感信息脱敏：日志中不出现 API Key、密码明文

---

## 阶段三：P2 持续优化验证

### Clippy Warnings 清除
- [ ] `clippy::unwrap_used` 规则已启用（deny 或 warn）
- [ ] `clippy::expect_used` 规则已启用
- [ ] Clippy warnings 总数 **< 50**
- [ ] 剩余 warnings 均为 false positives（有注释说明原因）

### 文档体系完善
- [ ] 所有 `pub fn` 均有 `///` 文档注释（`cargo doc --no-deps` 无 missing_docs warning）
- [ ] 关键函数包含 `# Examples` 代码示例（可通过 `cargo test --doc` 运行）
- [ ] rustdoc 已生成并可访问（GitHub Pages 或本地 `cargo doc --open`）
- [ ] `docs/architecture.md` 包含至少 3 条 ADR 记录
- [ ] `docs/performance_tuning.md` 覆盖 CPU/GPU/Metal 三种硬件场景
- [ ] `docs/troubleshooting.md` 包含 Top 10 常见错误及解决方案
- [ ] Docker Compose 生产配置文件存在且可一键部署

### CI/CD Pipeline 强化
- [ ] Security Audit Job 失败时会阻断 PR（无 continue-on-error: true）
- [ ] Code Coverage Gate 启用：< 80% 构建失败
- [ ] Performance Regression Test 存在：PR vs main ±5% 自动对比
- [ ] Documentation Build Job 存在：rustdoc 编译检查
- [ ] Dependabot 已启用（`.github/dependabot.yml` 存在）

### 功能 Roadmap 对齐
- [ ] Vulkan 后端评估报告已完成（结论：继续/放弃/延期）
- [ ] 分布式推理预研文档存在（技术选型、可行性分析）
- [ ] 模型并行ism 技术路线图存在（依赖上游进展说明）

---

## 阶段四：P3 长期规划验证

### 前后端类型安全共享
- [ ] `ts-rs` crate 已添加至 Cargo.toml
- [ ] API 类型定义包含 `#[derive(Serialize, Deserialize, Type)]`
- [ ] 构建脚本自动生成 `openmini-admin-web/src/types/generated.ts`
- [ ] CI step 验证前后端类型一致性（类型 diff 检查）

### 社区生态建设
- [ ] `CONTRIBUTING.md` 文件存在且内容完整（> 500 字）
- [ ] Python SDK 示例 `examples/python_client.py` 可独立运行
- [ ] 快速部署视频教程链接有效（YouTube/Bilibili）
- [ ] GitHub Discussions 已启用且包含模板分类
- [ ] 社区频道已建立（Discord invite link 或 Slack workspace）

---

## 最终质量门禁（Release 准入标准）

在发布 v1.2.0-stable 前，必须满足以下**全部条件**：

### 必须项（Blocking）
- [ ] ✅ **unwrap() 调用数 = 0**（排除 #[cfg(test)] 和 examples/）
- [ ] ✅ **Clippy errors = 0, Warnings < 50**
- [ ] ✅ **Code Coverage > 85%**（tarpaulin 测量）
- [ ] ✅ **全量测试通过**：`cargo test --workspace`（unit + integration + e2e）
- [ ] ✅ **24h 压力测试零崩溃**：`cargo test --test stress_test`
- [ ] ✅ **Security Audit 通过**：`cargo audit` 无已知 vulnerability
- [ ] ✅ **文档编译通过**：`cargo doc --no-deps` 无 warning

### 推荐项（Non-blocking 但强烈建议）
- [ ] ⭐ P99 延迟 < 100ms（推理请求）
- [ ] ⭐ 内存使用 < 80% 物理内存（单模型加载后）
- [ ] ⭐ 依赖版本均为最新稳定版（< 6 months old）
- [ ] ⭐ CHANGELOG 和 RELEASE_NOTES 已更新
- [ ] ⭐ README.md 快速开始指南可在 5 分钟内完成部署

---

## 验证命令速查

```bash
# 1. 检查 unwrap 使用情况
grep -r "unwrap()" openmini-server/src/ --include="*.rs" | grep -v "test" | wc -l

# 2. Clippy 检查
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -20

# 3. 测试覆盖率
cargo tarpaulin --workspace --out Xml --verbose

# 4. 全量测试
cargo test --workspace --lib          # 单元测试
cargo test --workspace --test '*'     # 集成测试

# 5. 安全审计
cargo audit --db https://crates.io/api/v1/crates

# 6. 文档生成
cargo doc --no-deps --document-private-items

# 7. 性能基准
cargo bench --workspace -- --save-baseline main

# 8. 压力测试
cargo test --workspace --test stress_test -- --nocapture --test-threads=1

# 9. 依赖版本检查
cargo outdated --root-deps-only -R

# 10. 构建大小检查
ls -lh target/release/openmini-server
```

---

## 验证签名

| 角色 | 姓名 | 日期 | 签署 |
|------|------|------|------|
| Spec Author | AI Assistant | 2026-04-10 | ✅ |
| Tech Lead | _____________ | ______ | ⬜ |
| QA Engineer | _____________ | ______ | ⬜ |
| Release Manager | _____________ | ______ | ⬜ |

---

**注意**: 
- 每个 checkpoint 必须有对应的自动化测试或手动验证步骤
- 失败的 checkpoint 需要创建新 task 并重新验证
- 所有 checkbox 勾选后才可进入 release 流程
