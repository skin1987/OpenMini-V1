# Tasks

## Phase 1: 运行时与配置优化

- [x] Task 1.1: Tokio 运行时切换为 current_thread 模式
  - [x] 修改 `openmini-server/src/main.rs` 的 `#[tokio::main]` 为 `#[tokio::main(flavor = "current_thread")]`
  - [x] 验证服务正常启动和基本请求处理
  - [x] 确认无编译错误和运行时 panic

- [x] Task 1.2: 调整 server.toml 配置参数
  - [x] `thread_pool.size` 从 4 改为 1
  - [x] `worker.count` 从 3 改为 1
  - [x] `server.max_connections` 从 100 提升到 10000
  - [x] 新增 `[core]` 配置段（count、binding、strategy）

- [x] Task 1.3: 移除 Gateway Semaphore 硬限制，改为原子计数器
  - [x] 在 `gateway.rs` 中将 `Semaphore::new(max_concurrent)` 替换为基于 AtomicU64 的 ConnectionLimiter
  - [x] 实现 try_acquire() / release() 方法
  - [x] 添加超限时的 graceful degradation（返回错误响应而非直接断开）
  - [x] 编写单元测试验证连接计数逻辑

## Phase 2: Gateway 层纯异步化

- [x] Task 2.1: 消除 block_on 反模式 — 重构 handle_chat_request / handle_image_request
  - [x] 将 `pool.execute(move || { ... block_on(...) })` 改为纯 async 实现（spawn_blocking）
  - [x] 直接在 async fn 中调用 InferenceEngine，通过 spawn_blocking 包装 CPU 密集型部分
  - [x] 移除对 crossbeam ThreadPool 的常规请求分发依赖
  - [x] 编写异步集成测试验证请求-响应完整链路

- [x] Task 2.2: 连接管理优化 — 轻量化每连接开销
  - [x] 引入 BufferPool（BytesMut 复用池），全局分配归还
  - [x] 优化 handle_connection 中的 buffer 管理，避免不必要的拷贝
  - [x] Gateway 结构体新增 buffer_pool 字段

- [x] Task 2.3: 会话管理无锁化
  - [x] 将 `RwLock<HashMap<String, Sender<Response>>>` 替换为 DashMap
  - [x] 确保并发读写安全性
  - [x] 测试通过（25/25 gateway 测试全部通过）

## Phase 3: Worker 进程池 → 异步推理任务池

- [x] Task 3.1: 创建 AsyncInferencePool 模块
  - [x] 新建 `src/service/worker/async_pool.rs`
  - [x] 实现 mpsc channel 任务接收 + oneshot 结果回传模式
  - [x] 实现 submit() 异步提交方法（不阻塞调用方）
  - [x] 实现 batch_loop() 批量收集 + 定时触发逻辑
  - [x] 使用 tokio::task::spawn_blocking 包装引擎 generate() 调用
  - [x] 编写单元测试：任务提交 → 批量执行 → 结果回传完整流程（3/3 通过）

- [x] Task 3.2: 推理引擎 Async 包装层
  - [x] 在 `src/model/inference/inference.rs` 中新增 generate_async()
  - [x] 内部使用 `tokio::task::spawn_blocking` 委托给阻塞线程池
  - [x] 为 InferenceEngine 和 ImagePreprocessor 添加 Clone derive
  - [x] 保持同步 API 不变，async 版本为新增接口

- [x] Task 3.3: Gateway 对接 AsyncInferencePool
  - [x] 修改 gateway.rs 的 dispatch_request 方法，使用 AsyncInferencePool 替代 ThreadPool 分发
  - [x] 适配新的请求/响应类型（InferenceTask / InferenceResult）
  - [x] 保持现有 Request/Response 结构体兼容
  - [x] 删除硬编码的 process_chat_sync / process_image_sync 假数据方法

- [x] Task 3.4: gRPC 服务层异步适配
  - [x] 修改 `grpc/server.rs` 中所有 `tx.blocking_send()` 为 `tx.send().await`
  - [x] chat() / image_understanding_stream() 流式接口全链路异步化

## Phase 4: Per-Core Actor 弹性架构

- [x] Task 4.1: 实现 PerCoreActor 核心结构
  - [x] 新建 `src/service/core_actor.rs`
  - [x] 实现 PerCoreActor { core_id, request_rx, active_connections }
  - [x] 实现 `PerCoreActor::run()` — 绑定 CPU 核心 + 事件循环 + 接收分发
  - [x] 实现 CPU 亲和性绑定函数 bind_to_core(core_id)（跨平台支持）
  - [x] 单元测试：Actor 启动/停止/核心绑定验证（3/3 通过）

- [x] Task 4.2: 实现 LoadBalancer (Router)
  - [x] 新建 `src/service/router.rs`
  - [x] 实现 CoreRouter { actors, strategy, next_index }
  - [x] 支持 RoundRobin / LeastConnections / ConsistentHash 三种策略
  - [x] 实现 dispatch() 方法：选择目标 Actor 并转发请求
  - [x] 编写路由策略正确性测试（4/4 通过）

- [ ] Task 4.3: 模型权重 mmap 共享机制
  - [ ] 在 InferenceEngine 或独立模块中实现模型权重的 mmap 加载
  - [ ] 多个 PerCoreActor 共享同一份只读模型权重内存
  - [ ] 每个 Actor 独立分配 KV Cache 内存
  - [ ] 验证共享模式下内存占用（应接近单实例而非 N 倍）
  - **状态**: 延后实施（需要与实际模型加载流程深度整合）

- [x] Task 4.4: main.rs 启动器重构为 Multi-Core Launcher
  - [x] 根据 CPU 核心数启动 N 个 PerCoreActor
  - [x] 每个 Actor 绑定到对应物理核心
  - [x] 创建 CoreRouter 并注入各 Actor 的 sender
  - [x] Gateway 通过 Router 分发请求
  - [x] 优雅关闭：逐个停止 Actor，等待进行中请求完成

- [x] Task 4.5: 配置系统扩展
  - [x] 在 `config/settings.rs` 中新增 `CoreSettings` 结构体
  - [x] 在 ServerConfig 中新增 `core: CoreSettings` 字段
  - [x] 更新 `config/server.toml` 默认值和注释说明
  - [x] 配置验证：默认值合理（settings 18/18 测试通过）

## Phase 5: 验证与调优

- [ ] Task 5.1: 性能基准测试
  - [ ] 使用 wrk/hey/k6 工具进行并发压测
  - [ ] 目标：单核 5000+ 并发，P99 < 200ms，CPU > 90%
  - **状态**: 需要完整模型加载后才能进行真实压测

- [x] Task 5.2: 编译警告与错误清零
  - [x] 运行 `cargo check --lib` 确保零 compile error ✅
  - [x] 运行 `cargo test --lib` (gateway+router+actor+pool+settings) 65/65 全部通过 ✅
  - [x] 修复新引入的 compile error / warning / test failure ✅
  - **注意**: DSA 模块有 2 个预已有测试失败（与本改造无关）

# Task Dependencies
- [Task 1.1] ← 无依赖 ✅
- [Task 1.2] ← 依赖 [Task 1.1] ✅
- [Task 1.3] ← 依赖 [Task 1.1] ✅
- [Task 2.1] ← 依赖 [Phase 1 全部完成] ✅
- [Task 2.2] ← 依赖 [Phase 1 完成] ✅
- [Task 2.3] ← 依赖 [Phase 1 完成] ✅
- [Task 3.1] ← 依赖 [Phase 2 完成] ✅
- [Task 3.2] ← 可与 [Task 3.1] 并行 ✅
- [Task 3.3] ← 依赖 [Task 3.1] + [Task 3.2] ✅
- [Task 3.4] ← 依赖 [Task 3.2] ✅
- [Task 4.1] ← 依赖 [Phase 3 完成] ✅
- [Task 4.2] ← 依赖 [Task 4.1] ✅
- [Task 4.3] ← 延后实施（需深度整合）
- [Task 4.4] ← 依赖 [Task 4.1] + [Task 4.2] ✅
- [Task 4.5] ← 与 Phase 1 并行完成 ✅
- [Task 5.1] ← 需要完整环境压测
- [Task 5.2] ← 已完成 ✅
