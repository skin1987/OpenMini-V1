# 高并发异步推理架构改造 Spec

## Why

OpenMini-V1 当前架构存在 6 个核心瓶颈，导致单实例并发能力被限制在 ~100 连接，无法发挥 Rust + Tokio 的异步高并发潜力。目标是将单实例并发能力提升至 **10,000+**，并为未来百万级并发扩展奠定基础。

**当前瓶颈总结：**

| # | 瓶颈 | 位置 | 影响 |
|---|------|------|------|
| 1 | 同步/异步桥接反模式 (`block_on`) | gateway.rs:538-541 | 每个请求创建临时运行时，延迟增加 10-100x |
| 2 | Worker 进程池阻塞通信 (stdin/stdout) | worker/pool.rs:159-207 | 同步阻塞管道，一次只能处理一个请求 |
| 3 | Tokio 多线程运行时未优化 | main.rs:73 | 单核场景下多线程调度器反而增加锁竞争 |
| 4 | Gateway 请求处理链路冗余 | gateway.rs:507-548 | Tokio → OS线程池 → block_on → Tokio（效率~30%） |
| 5 | 流式输出使用 blocking_send | grpc/server.rs:345 | 可能阻塞事件循环 |
| 6 | Semaphore 硬限制连接数 100 | gateway.rs:253 | 无法扩展到高并发 |

## What Changes

### Phase 1: 运行时与配置优化（立即见效）
- Tokio 运行时从多线程切换为 `current_thread` 单线程模式
- 调整 server.toml 配置参数（线程池大小、Worker 数量、连接限制）
- 移除 Gateway 层的 Semaphore 硬限制，改为原子计数器背压

### Phase 2: Gateway 层纯异步化
- **BREAKING**: 移除 `futures::executor::block_on` 反模式
- 将 `ThreadPool` (crossbeam OS线程池) 分发改为 `tokio::spawn` 异步任务
- 请求处理链路从 `Tokio → OS线程 → block_on → Tokio` 简化为纯 `Tokio` 异步链路
- 连接管理从每连接一个 task 改为轻量状态机 + buffer pool 复用

### Phase 3: Worker 进程池 → 异步推理任务池
- **BREAKING**: 将多进程 Worker Pool (stdin/stdout 管道) 替换为 `AsyncInferencePool`
- 使用 `mpsc::channel` + `tokio::task::spawn_blocking` 实现非阻塞推理
- 引入批量推理调度器 (batch collector)，提升 GPU 利用率到 85%+
- 推理引擎包装为 async 接口 (`generate_async`, `stream_generate_async`)

### Phase 4: Per-Core Actor 架构（弹性扩展）
- 实现 `PerCoreActor`: 每个 CPU 核心一个独立 Tokio 运行时 + 推理引擎实例
- 实现 `LoadBalancer` (Router): 支持轮询/最少连接/一致性哈希策略
- CPU 亲和性硬绑定 (core affinity binding)
- 模型权重通过 mmap 只读共享，KV Cache 各核心独立
- 配置化核心数量 `[core] count = N`，硬件升级只需改数字

## Impact

- Affected specs: 无直接依赖的现有 spec
- Affected code:
  - `openmini-server/src/main.rs` — Tokio 运行时改造
  - `openmini-server/src/service/server/gateway.rs` — 核心重构：消除 block_on、连接管理优化
  - `openmini-server/src/service/thread/pool.rs` — 可能废弃或降级为辅助角色
  - `openmini-server/src/service/worker/pool.rs` — 重写为 AsyncInferencePool
  - `openmini-server/src/service/worker/worker.rs` — 适配新架构
  - `openmini-server/src/service/grpc/server.rs` — blocking_send → send().await
  - `openmini-server/src/model/inference/engine.rs` — 新增 async 包装方法
  - `openmini-server/src/config/settings.rs` — 新增 [core] 配置段
  - `config/server.toml` — 参数调整

## ADDED Requirements

### Requirement: 单线程 Tokio 运行时
系统 SHALL 使用 `#[tokio::main(flavor = "current_thread")]` 单线程运行时作为默认模式，以消除多线程调度开销并最大化单核事件循环效率。

#### Scenario: 单核高效运行
- **WHEN** 服务器在单核或指定少数核心上运行
- **THEN** Tokio 运行时使用 current_thread flavor，所有异步任务在同一事件循环中调度

### Requirement: 无阻塞异步推理任务池
系统 SHALL 提供 `AsyncInferencePool`，替代现有的同步阻塞 Worker 进程池，支持：
- 通过 mpsc channel 提交推理任务，不阻塞调用方
- 批量收集多个请求后统一执行推理（可配置 batch_size 和 batch_timeout）
- 使用 `tokio::task::spawn_blocking` 将 CPU 密集型推理计算放到阻塞线程池
- 结果通过 oneshot channel 回传给原始请求方

#### Scenario: 非阻塞推理提交
- **WHEN** Gateway 收到聊天推理请求
- **THEN** 任务被提交到 AsyncInferencePool 后立即返回，不阻塞事件循环处理其他连接
- **AND** 推理完成后结果自动路由回对应连接

#### Scenario: 批量推理优化 GPU 利用率
- **WHEN** 多个推理请求在短时间窗口内到达（< 10ms）
- **THEN** 这些请求被打包为一个 batch 一次性执行
- **AND** GPU 利用率从 ~30%（batch=1）提升到 85%+

### Requirement: Per-Core Actor 弹性架构
系统 SHALL 支持 Per-Core Actor 架构，允许按需启动 N 个独立推理服务实例（N = 可用 CPU 核心数），每个 Actor：
- 绑定到指定的物理 CPU 核心（CPU affinity）
- 拥有独立的 Tokio 单线程运行时
- 持有独立的 KV Cache 和推理状态
- 通过 mmap 共享只读模型权重（零额外内存开销）
- 独立管理自己的并发连接（每个 Actor 可承载 5000+ 连接）

#### Scenario: 双核部署
- **WHEN** 配置 `core.count = 2`
- **THEN** 启动 2 个 PerCoreActor，分别绑定 Core-0 和 Core-1
- **AND** LoadBalancer 将客户端请求均匀分发到两个 Actor
- **AND** 总并发能力 ≈ 2 × 单核并发能力

#### Scenario: 硬件无缝升级
- **WHEN** 硬件从 2 核升级到 8 核
- **THEN** 仅需修改配置 `core.count = 8` 并重启
- **AND** 自动启动 8 个 Actor，无需修改任何代码

### Requirement: 高并发连接管理
系统 SHALL 支持万级以上并发连接，要求：
- 移除 Semaphore(100) 硬限制，改用原子计数器实现软限制和背压控制
- 每连接内存占用 < 4KB（通过状态机 + buffer pool 复用实现）
- 会话管理使用 DashMap 或无锁数据结构替代 RwLock<HashMap>
- 支持 idle connection timeout 自动清理

#### Scenario: 万级并发连接
- **WHEN** 10000 个客户端同时保持连接
- **THEN** 所有连接正常维护，总连接管理内存占用 < 100MB
- **AND** 新请求仍能正常处理，P99 响应延迟 < 200ms

### Requirement: 流式输出全异步
系统 SHALL 确保流式推理输出（chat stream / image stream / TTS 等）全程异步：
- 将 `tx.blocking_send()` 替换为 `tx.send().await`
- 确保 StreamGenerator 的回调不阻塞事件循环
- 流式 chunk 发送使用 mpsc channel 异步传递

## MODIFIED Requirements

### Requirement: Gateway TCP 服务网关
Gateway SHALL 从"同步分发到 OS 线程池"模式改造为"纯异步任务调度"模式：
- 移除对 ThreadPool (crossbeam) 的依赖用于常规请求分发
- ThreadPool 仅保留用于 CPU 密集型的预处理任务（如 tokenization）
- `handle_chat_request` / `handle_image_request` 直接在 Tokio async context 中执行
- 超时控制使用 `tokio::time::timeout` 而非阻塞等待

### Requirement: gRPC 服务端
OpenMiniService gRPC 服务 SHALL 适配异步架构：
- 流式接口中的 `blocking_send` 全部替换为 `.send().await`
- 推理调用通过 AsyncInferencePool 异步提交
- 数据库操作（记忆读写）保持现有 sqlx 异步方式不变

### Requirement: 服务器配置
ServerConfig SHALL 新增 `[core]` 配置段：
```toml
[core]
count = 2                    # 使用的 CPU 核心数
binding = true               # 是否启用 CPU 亲和性绑定
strategy = "round_robin"     # 负载均衡策略: round_robin / least_conn / hash

[core.per_core]
max_concurrent = 5000       # 每个 Core 最大并发连接数
kv_cache_mb = 2048          # 每个 Core KV Cache 大小(MB)
```

## REMOVED Requirements

### Requirement: 同步 Worker 进程池 (pool.rs)
**Reason**: stdin/stdout 阻塞管道通信是高并发场景的根本瓶颈，无法支撑千级以上并发。替换为 AsyncInferencePool。
**Migration**: 
- 保留 `WorkerPool` 结构体标记为 `#[deprecated]`，保留 API 兼容一个版本周期
- 新代码统一使用 `AsyncInferencePool`
- 配置中 `worker.enabled = false` 默认禁用旧 Worker

### Requirement: Gateway Semaphore 连接限制
**Reason**: `Semaphore::new(100)` 是硬性上限，无法动态调整，且在百万级并发场景下语义不正确。
**Migration**: 
- 替换为基于 AtomicU64 的 `ConnectionLimiter`，支持配置化上限和软限背压
- 保留 graceful degradation：超限时返回 503 Service Unavailable 而非直接拒绝
