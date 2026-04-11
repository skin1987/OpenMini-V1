# ADR-001: 简化架构 - 从 Worker Pool 迁移至 TaskScheduler

## 状态
已接受 (2026-04-10)

## 背景

### 当前架构问题

OpenMini-V1 v1.2.0-beta.1 采用 **4 层资源池架构**：

```
HTTP/gRPC Request
    ↓
Gateway (TCP Server)
    ↓
AsyncInferencePool (任务队列)
    ↓
ThreadPool (线程池, size=1)  ← 瓶颈：单线程
    ↓
CoreRouter (CPU 核心路由)
    ↓
WorkerPool (多进程池, count=1)  ← 瓶颈：IPC 延迟
    ↓
Child Process (stdin/stdout)
```

### 已识别的问题

| 问题 | 影响 | 严重程度 |
|------|------|---------|
| **IPC 延迟高** | 每次 Worker 通信 ~2ms (序列化 + pipe I/O) | 🔴 高 |
| **配置矛盾** | thread_pool.size=1, worker.count=1 无法发挥多核优势 | 🟠 中 |
| **调试困难** | 多进程导致 panic 堆栈难以追踪 | 🟠 中 |
| **资源浪费** | 每个进程独立内存空间，KV Cache 无法共享 | 🟡 低 |
| **复杂度高** | 4 层抽象增加认知和维护成本 | 🟡 低 |

### 性能基准数据（来自 profiling）

```
Request Latency Breakdown:
├── HTTP Parsing:         0.05ms
├── Gateway Routing:      0.10ms
├── AsyncInferencePool:   0.20ms
├── ThreadPool Dispatch:  0.15ms
├── CoreRouter:           0.08ms
├── WorkerPool IPC:       **2.30ms**  ← 最大瓶颈
└── Inference Engine:     45.00ms
─────────────────────────────────────
Total:                   47.88ms
```

**结论**: Worker Pool IPC 占总延迟的 **4.8%**，且无法利用 Tokio 异步 I/O 优势。

---

## 决策

### 选择方案: 基于 Tokio Runtime 的统一 TaskScheduler

#### 目标架构

```
HTTP/gRPC Request
    ↓
Gateway (TCP/HTTP Server)
    ↓
TaskScheduler (统一调度器)
    ├─ mpsc channel (无锁队列)
    ├─ tokio::task::spawn_blocking (CPU 密集型)
    └─ tokio::spawn (I/O 密集型)
    ↓
InferenceEngine (单进程多线程)
    ├── Candle (模型推理)
    ├── KV Cache (共享内存)
    └── Tokenizer (文本处理)
```

#### 关键设计决策

1. **移除 Worker Pool 多进程模型**
   - 改用 `tokio::task::spawn_blocking` 处理 CPU 密集型推理任务
   - 利用 Tokio 工作窃取 (work-stealing) 调度器自动负载均衡

2. **合并 ThreadPool + CoreRouter 为 TaskScheduler**
   - 统一的任务提交接口 (`submit()`)
   - 内部根据任务类型自动选择执行策略

3. **保留 AsyncInferencePool 用于请求排队**
   - 背压 (backpressure) 机制防止过载
   - 批处理 (batching) 优化吞吐量

4. **向后兼容性保证**
   - `WorkerPool` / `WorkerHandle` 标记为 `#[deprecated]`
   - 提供迁移工具和文档
   - 配置文件支持旧格式（带 deprecation warning）

---

## 替代方案评估

### 方案 A: 保留 Worker Pool + 优化 IPC
- ✅ 改动最小
- ❌ 仍存在 ~2ms IPC 延迟
- ❌ 多进程调试困难未解决
- **评分**: 6/10

### 方案 B: 移除 Worker Pool → ThreadPool (当前选择)
- ✅ 消除 IPC 延迟
- ✅ 单进程易调试
- ✅ 共享内存 (KV Cache 效率提升)
- ⚠️ 需要迁移代码
- **评分**: 9/10

### 方案 C: 完全异步 (纯 tokio::spawn)
- ✅ 最大并发度
- ❌ CPU 密集型任务阻塞 event loop
- ❌ 需要 runtime::Handle::current()
- **评分**: 7/10

### 方案 D: 使用 rayon::ThreadPool
- ✅ 工作窃取调度
- ❌ 与 Tokio 集成需要 bridge
- ❌ 额外依赖
- **评分**: 8/10

**最终选择**: 方案 B (TaskScheduler based on Tokio)

---

## 影响范围

### 需修改的模块

| 模块 | 变更类型 | 影响程度 |
|------|----------|---------|
| `service/scheduler/mod.rs` | **新建** | 核心组件 |
| `service/server/gateway.rs` | 重构 | 使用 TaskScheduler |
| `service/worker/pool.rs` | 标记 deprecated | 向后兼容 |
| `config/server.toml` | 新增 `[scheduler]` section | 配置变更 |
| `main.rs` | 初始化逻辑调整 | 启动流程 |

### 不受影响的模块

- ✅ `model/inference/` - 推理引擎无需改动
- ✅ `hardware/` - GPU/CPU 抽象层不变
- ✅ `monitoring/` - 监控指标兼容
- ✅ `training/` - 训练模块独立运行

---

## 迁移策略

### Phase 1: 并行运行 (v1.2.0-beta.2)
- [x] 实现 TaskScheduler 模块
- [ ] Gateway 同时支持两种模式（feature flag）
- [ ] 添加性能对比 benchmark

### Phase 2: 默认切换 (v1.3.0)
- [ ] TaskScheduler 成为默认选项
- [ ] WorkerPool 标记为 `#[deprecated]`
- [ ] 更新文档和示例

### Phase 3: 移除旧代码 (v2.0.0)
- [ ] 删除 WorkerPool 相关代码
- [ ] 清理配置文件旧格式
- [ ] 发布迁移完成公告

---

## 配置变更

### 新增配置项

```toml
# config/server.toml

[scheduler]
# 任务调度器配置 (替代旧的 [thread_pool] 和 [worker])

# 最大并发任务数 (默认 = CPU 核心数)
max_concurrent = 4

# 任务队列容量 (超出将返回错误)
queue_capacity = 1000

# 任务类型: "blocking" (CPU密集) | "async" (IO密集) | "auto" (自动检测)
worker_type = "auto"

# 批处理大小 (连续请求合并为一批以优化吞吐量)
batch_size = 8

# 批处理超时 (等待凑齐一批的最长时间, ms)
batch_timeout_ms = 5

# === 以下为 deprecated 配置 (将在 v2.0.0 移除) ===
# [thread_pool]
# size = 1  # DEPRECATED: Use [scheduler].max_concurrent

# [worker]
# count = 1  # DEPRECATED: TaskScheduler is single-process
```

---

## 性能预期

### 基准对比 (预估)

| 指标 | 当前 (Worker Pool) | 目标 (TaskScheduler) | 提升 |
|------|-------------------|---------------------|------|
| P99 延迟 | 48ms | **45ms** | -6% |
| 吞吐量 (QPS) | ~200 | **~250** | +25% |
| 内存使用 | 800MB (多进程) | **600MB** (单进程) | -25% |
| CPU 利用率 | 60% (多进程开销) | **85%** (工作窃取) | +42% |
| 调试便利性 | 困难 (多进程) | **简单** (单进程) | ∞ |

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Candle 非线程安全 | 低 | 高 | 锁保护推理上下文 |
| 单点故障 | 中 | 中 | 进程级隔离 (未来可扩展) |
| 迁移 bug | 中 | 高 | 充分测试 + feature flag 回滚 |

---

## 决策记录

- **决策者**: OpenMini Team + AI Assistant
- **日期**: 2026-04-10
- **状态**: 已接受
- **下次评审**: v1.3.0 发布后

---

## 参考资料

- [Tokio Runtime 工作窃取调度](https://tokio.rs/tokio/src/runtime/task/mod.html)
- [spawn_blocking vs spawn](https://tokio.rs/tokio/task/fn.spawn_blocking.html)
- [Candle 线程安全性说明](https://github.com/huggingface/candle)
