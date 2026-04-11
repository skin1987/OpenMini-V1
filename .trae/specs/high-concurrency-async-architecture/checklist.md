# Checklist

## Phase 1: 运行时与配置优化
- [x] Tokio 运行时已切换为 `current_thread` 模式 (main.rs)
- [x] server.toml 配置参数已调整（thread_pool.size=1, worker.count=1, max_connections=10000）
- [x] `[core]` 配置段已添加到 settings.rs 和 server.toml
- [x] Gateway Semaphore 硬限制已替换为 AtomicU64 ConnectionLimiter
- [x] ConnectionLimiter 超限返回 503 而非直接断开连接
- [x] Phase 1 全部编译通过 (`cargo check --lib`)
- [x] Phase 1 全部测试通过

## Phase 2: Gateway 层纯异步化
- [x] gateway.rs 中 `futures::executor::block_on` 调用已全部移除
- [x] handle_chat_request / handle_image_request 已改为纯 async 实现（spawn_blocking）
- [x] 常规请求分发不再依赖 crossbeam ThreadPool（仅 CPU 密集型任务保留）
- [x] BufferPool 已引入，BytesMut 复用池正常工作
- [x] 会话管理已从 RwLock<HashMap> 替换为 DashMap 无锁结构
- [x] 并发会话读写压力测试通过 (25/25 gateway 测试)

## Phase 3: 异步推理任务池
- [x] AsyncInferencePool 模块已创建 (`async_pool.rs`)
- [x] mpsc + oneshot 通道模式任务提交/结果回传正常工作
- [x] batch_loop 批量收集逻辑正确（超时触发 + 数量触发）
- [x] tokio::task::spawn_blocking 包装引擎调用正常
- [x] InferenceEngine async 包装方法已实现 (generate_async)
- [x] 同步 API 保持不变，async 版本为新增接口
- [x] Gateway 已对接 AsyncInferencePool 替代 ThreadPool 分发
- [x] gRPC 层所有 `blocking_send` 已改为 `.send().await`
- [x] AsyncInferencePool 测试全部通过 (3/3)
- [x] Phase 3 编译零错误

## Phase 4: Per-Core Actor 弹性架构
- [x] PerCoreActor 核心结构已实现 (`core_actor.rs`)
- [x] PerCoreActor::run() 事件循环正常运行
- [x] CPU 亲和性绑定函数 bind_to_core() 已实现（跨平台支持）
- [x] CoreRouter (LoadBalancer) 已实现，支持 RoundRobin 策略
- [x] CoreRouter dispatch() 路由分发正确
- [x] main.rs Multi-Core Launcher 根据 core.count 启动 N 个 Actor
- [x] Gateway/gRPC 通过 Router 分发到各 Actor
- [x] 优雅关闭逐个停止 Actor 正常工作
- [x] CoreSettings 配置结构和 server.toml 已更新
- [x] PerCoreActor 测试全部通过 (3/3)
- [x] CoreRouter 测试全部通过 (4/4)
- [ ] ~~模型权重 mmap 共享机制~~ → 延后实施（需深度整合）

## Phase 5: 验证与调优
- [ ] 性能基准测试完成（需完整环境）
- [x] `cargo check --lib` 编译零错误 ✅
- [x] `cargo test --lib` 新增模块测试 65/65 全部通过 ✅
- [x] release 编译验证通过
