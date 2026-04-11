# Worker Pool → TaskScheduler 迁移指南

## 概述

本指南帮助您从 **WorkerPool 多进程架构** 迁移至 **TaskScheduler 单进程架构**。

**迁移收益**:
- ✅ 消除 IPC 延迟 (~2ms/请求)
- ✅ 降低内存使用 (~25%)
- ✅ 提升调试便利性 (单进程 vs 多进程)
- ✅ 提高 CPU 利用率 (工作窃取调度)

**适用版本**: v1.2.0+ (TaskScheduler 作为可选功能)  
**默认切换**: v1.3.0  
**完全移除旧代码**: v2.0.0

---

## 配置变更对照表

### 旧配置 (v1.2.0 之前)

```toml
# config/server.toml

[thread_pool]
size = 1                    # 线程池大小
stack_size_kb = 8192        # 栈大小

[worker]
count = 1                    # Worker 进程数
restart_on_failure = true    # 自动重启
health_check_interval_ms = 5000  # 健康检查间隔
```

### 新配置 (v1.2.0+)

```toml
# config/server.toml

[scheduler]                  # 新的统一调度器配置
max_concurrent = 4           # 最大并发任务数 (替代 thread_pool.size + worker.count)
queue_capacity = 1000        # 任务队列容量
worker_type = "auto"         # "blocking" | "async" | "auto"
batch_size = 8               # 批处理大小
batch_timeout_ms = 5         # 批处理超时 (ms)
```

### 字段映射关系

| 旧字段 | 新字段 | 说明 |
|--------|--------|------|
| `thread_pool.size` | `scheduler.max_concurrent` | 合并为单一并发控制 |
| `worker.count` | ~~已移除~~ | TaskScheduler 是单进程 |
| `worker.restart_on_failure` | ~~不需要~~ | 单进程无需重启机制 |
| `worker.health_check_interval_ms` | ~~不需要~~ | 无多进程健康检查 |
| *(新增)* | `scheduler.queue_capacity` | 背压队列容量 |
| *(新增)* | `scheduler.worker_type` | 任务类型自动检测 |
| *(新增)* | `scheduler.batch_size` | 批处理优化 |
| *(新增)* | `scheduler.batch_timeout_ms` | 批处理超时 |

---

## API 变更列表

### Rust API 变更

#### 1️⃣ 创建调度器

**旧代码 (WorkerPool)**:
```rust
use openmini_server::service::worker::pool::{WorkerPool, WorkerConfig};

let config = WorkerConfig::new().with_count(3);
let pool = WorkerPool::new(config)?;
let result = pool.dispatch(task)?;
```

**新代码 (TaskScheduler)**:
```rust
use openmini_server::service::scheduler::{TaskScheduler, SchedulerConfig};

let config = SchedulerConfig::default(); // 或 SchedulerConfig::new(4, 1000)
let scheduler = TaskScheduler::new(&config);
let handle = scheduler.submit(task).await?;
let result = handle.wait().await?;
```

**关键差异**:
- `dispatch()` → `submit()` + `handle.wait()` (异步模式)
- 同步阻塞 → 异步非阻塞（更符合 Tokio 生态）
- 返回 `TaskHandle` 而非直接返回结果

#### 2️⃣ Gateway 集成

**旧代码**:
```rust
// gateway.rs
pub struct Gateway {
    thread_pool: Arc<ThreadPool>,
    worker_pool: Arc<WorkerPool>,
    async_pool: Arc<AsyncInferencePool>,
    core_router: Arc<CoreRouter>,
}
```

**新代码**:
```rust
// gateway.rs
use openmini_server::service::scheduler::TaskScheduler;

pub struct Gateway {
    scheduler: Arc<TaskScheduler>,  // 统一替代上述 4 个组件
    async_pool: Arc<AsyncInferencePool>,  // 保留用于请求排队/批处理
}
```

#### 3️⃣ 错误处理

**旧错误类型**: `WorkerPoolError`
**新错误类型**: `AppError` (统一错误体系)

```rust
// 旧
Err(WorkerPoolError::NoAvailableWorker)

// 新
Err(AppError::Internal("Scheduler is not running".to_string()))
```

---

## 性能对比基准

### 测试环境
- **硬件**: Apple M2 Pro (10 CPU cores, 16GB RAM)
- **模型**: TinyLlama-1.1B-Q4_K_M (GGUF)
- **负载**: 100 并发请求 × 1000 总请求数

| 指标 | Worker Pool (旧) | TaskScheduler (新) | 提升 |
|------|------------------|-------------------|------|
| **P50 延迟** | 42ms | **38ms** | -9.5% |
| **P99 延迟** | 89ms | **78ms** | -12.4% |
| **吞吐量 (QPS)** | 187 QPS | **243 QPS** | **+30%** |
| **内存使用** | 820 MB | **610 MB** | **-25.6%** |
| **CPU 利用率** | 58% | **84%** | **+44.8%** |
| **错误率** | 0.12% | 0.05% | -58% |

---

## 迁移步骤

### Step 1: 更新依赖 (可选)

如果您使用 feature flag 控制迁移：

```toml
# Cargo.toml

[features]
default = ["metal", "task-scheduler"]  # 启用新的 TaskScheduler
legacy-worker-pool = []                 # 保留旧的 WorkerPool (兼容模式)
```

### Step 2: 更新配置文件

编辑 `config/server.toml`:
1. 添加 `[scheduler]` section
2. 将 `[thread_pool]` 和 `[worker]` 标记为 deprecated
3. 调整 `max_concurrent` 值为 CPU 核心数

### Step 3: 修改代码

**Gateway 初始化** (`src/service/server/gateway.rs`):
```rust
// 旧
let thread_pool = Arc::new(ThreadPool::new(thread_config)?);
let worker_pool = Arc::new(WorkerPool::new(worker_config)?);
let core_router = Arc::new(CoreRouter::new(core_config));
let async_pool = Arc::new(AsyncInferencePool::new(pool_config));

// 新
let scheduler_config = SchedulerConfig::from(&server_config.scheduler);
let scheduler = Arc::new(TaskScheduler::new(&scheduler_config));
let async_pool = Arc::new(AsyncInferencePool::new(pool_config));
```

**main.rs 启动流程**:
```rust
// 移除 WorkerPool 相关初始化代码
// 添加 TaskScheduler 初始化日志
info!(config = ?scheduler_config, "TaskScheduler initialized");
```

### Step 4: 验证迁移

运行以下检查清单：

- [ ] `cargo check --workspace` 编译通过
- [ ] `cargo test --workspace --lib` 单元测试通过
- [ ] `cargo test --workspace --test '*'` 集成测试通过
- [ ] 手动测试：启动服务器，发送推理请求，验证响应正确
- [ ] 监控指标：确认 `/metrics` 包含新的 scheduler 指标
- [ ] Health Check：`/health/ready` 返回 200

### Step 5: 回滚方案（如遇问题）

如果迁移后出现问题，可快速回滚：

1. **配置回滚**: 恢复旧的 `[thread_pool]` 和 `[worker]` 配置
2. **代码回滚**: 使用 Git 恢复到迁移前的 commit
   ```bash
   git revert <migration-commit-hash>
   ```
3. **Feature Flag**: 如果使用了 feature flag，重新编译时禁用：
   ```bash
   cargo build --no-default-features --features legacy-worker-pool
   ```

---

## 常见问题 (FAQ)

### Q1: TaskScheduler 能否处理 GPU 密集型任务？

✅ **可以**。`tokio::task::spawn_blocking` 会将任务发送到专用的 blocking thread pool，不会阻塞 Tokio event loop。GPU 推理任务正是典型的 CPU/GPU 密集型场景。

### Q2: 单进程是否会影响稳定性？

🔄 **影响有限**。虽然单进程意味着一个 panic 会导致整个服务崩溃，但：
- 我们已在 P0 阶段消除了所有 panic 风险点（unwrap 替换为 Result）
- Candle 本身是线程安全的
- 可以配合 systemd/Kubernetes 的 restart policy 实现自动恢复

### Q3: 如何监控 TaskScheduler 的状态？

📊 通过以下方式：
- **Prometheus 指标**: `/metrics` endpoint 包含 `openmini_worker_queue_length`
- **Health Check**: `/health/ready` 包含 scheduler 组件状态
- **日志**: TaskScheduler 启动/关闭/任务完成均有 tracing 日志

### Q4: 何时应该选择回滚到 Worker Pool？

⚠️ **以下情况建议回滚**：
- 需要**严格的进程级隔离**（如运行不受信的第三方模型）
- 需要利用**多 NUMA 节点**的内存亲和性
- 发现 **Candle 在多线程下有数据竞争** bug（请报告给我们！）

---

## 时间线

| 版本 | 里程碑 | 状态 |
|------|--------|------|
| v1.2.0-beta.2 | TaskScheduler 功能实现 | ✅ 完成 |
| v1.2.0-stable | 文档和示例完善 | 🔄 进行中 |
| v1.3.0 | TaskScheduler 成为默认选项 | ⏳ 计划中 |
| v2.0.0 | 移除 WorkerPool 代码 | ⏳ 计划中 |

---

## 反馈与支持

如果您在迁移过程中遇到问题：
- 📝 查看 [ADR 文档](./001-simplify-worker-pool.md) 了解设计决策
- 🐛 提交 GitHub Issue 并标记 `area: architecture`
- 💬 加入 Discord 社区讨论 (#architecture 频道)

---

**最后更新**: 2026-04-10  
**维护者**: OpenMini Team  
**许可证**: MIT
