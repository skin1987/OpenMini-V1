//! # OpenMini 统一错误处理模块
//!
//! 本模块定义了整个系统的错误类型层次结构，采用 **分层错误设计模式**：
//!
//! ## 错误处理策略
//!
//! 1. **统一错误枚举** (`AppError`) - 作为所有公开 API 的错误返回类型
//! 2. **领域错误分类** - 按子系统划分：Engine、Worker、Training、Hardware、Config
//! 3. **自动错误转换** - 使用 `thiserror` 的 `#[from]` 属性实现自动转换
//! 4. **错误链保留** - 通过 `Source` trait 保留原始错误上下文
//!
//! ## 错误码体系
//!
//! | 前缀 | 子系统 | 示例 |
//! |------|--------|------|
//! | `ENG` | 推理引擎 | `ENG001` KV Cache 维度不匹配 |
//! | `WRK` | 工作线程 | `WRK001` 互斥锁中毒 |
//! | `TRN` | 训练模块 | `TRN001` 张量操作失败 |
//! | `HW` | 硬件层 | `HW001` 内存分配失败 |
//! | `CFG` | 配置管理 | `CFG001` 配置解析失败 |
//!
//! ## 使用示例
//!
//! ```rust,ignore
//! use openmini_server::error::{AppError, EngineError};
//!
//! // 方式1: 直接使用子错误（自动转换为 AppError）
//! fn load_model() -> Result<(), AppError> {
//!     // EngineError 会自动转换为 AppError::Engine(...)
//!     Err(EngineError::KvCacheDimensionMismatch { expected: 4096, actual: 2048 })?;
//!     Ok(())
//! }
//!
//! // 方式2: 使用 AppError::Internal 处理未知错误
//! fn process_request() -> Result<String, AppError> {
//!     let result = some_fallible_operation()
//!         .map_err(|e| AppError::Internal(format!("Processing failed: {}", e)))?;
//!     Ok(result)
//! }
//! ```
//!
//! ## 最佳实践
//!
//! - **不要吞掉错误** - 始终使用 `?` 操作符向上传播
//! - **提供上下文** - 使用 `AppError::Internal` 添加额外调试信息
//! - **避免 panic** - 所有可恢复的错误都应返回 `Result`
//! - **日志记录** - 在错误产生点记录详细日志，在传播时保持简洁

use thiserror::Error;
use ts_rs::TS;

/// OpenMini 应用程序统一错误类型
///
/// 这是所有公开 API 的错误返回类型，封装了各子系统的具体错误。
/// 通过 `From<T>` 实现自动从子错误类型转换。
///
/// # 错误变体说明
///
/// - `Engine` - 推理引擎相关错误（KV Cache、注意力计算等）
/// - `Worker` - 任务调度/工作线程相关错误
/// - `Training` - 模型训练相关错误
/// - `Hardware` - GPU/CPU/SIMD 硬件操作错误
/// - `Config` - 配置文件解析和验证错误
/// - `Io` - 文件 I/O 和网络 I/O 错误
/// - `Internal` - 内部逻辑错误（用于未知或临时性错误）
///
/// # 示例
///
/// ```rust,ignore
/// fn handle_inference() -> Result<Vec<f32>, AppError> {
///     // 自动转换 EngineError -> AppError::Engine
///     validate_kv_cache()?;
///     
///     // 自动转换 HardwareError -> AppError::Hardware
///     allocate_gpu_memory()?;
///     
///     Ok(compute_output())
/// }
/// ```
#[derive(Debug, Error, TS)]
#[ts(export)]
pub enum AppError {
    /// 推理引擎错误 (错误码前缀: ENG)
    ///
    /// 涵盖 KV Cache 管理、流式注意力计算等引擎内部操作失败。
    /// **恢复建议**: 检查模型配置是否正确，确认输入维度与模型匹配。
    #[error("Engine error: {0}")]
    Engine(#[from] EngineError),

    /// 工作线程/任务调度错误 (错误码前缀: WRK)
    ///
    /// 涵盖任务队列、线程池通信、任务执行等调度层错误。
    /// **恢复建议**: 检查系统资源（CPU/内存），重启服务可能解决问题。
    #[error("Worker error: {0}")]
    Worker(#[from] WorkerError),

    /// 模型训练错误 (错误码前缀: TRN)
    ///
    /// 涵盖张量操作、梯度计算、优化器步骤、检查点保存等训练流程错误。
    /// **恢复建议**: 检查训练数据格式，验证超参数设置，确认磁盘空间充足。
    #[error("Training error: {0}")]
    Training(#[from] TrainingError),

    /// 硬件操作错误 (错误码前缀: HW)
    ///
    /// 涵盖 GPU 内存分配、CUDA/Metal 操作、SIMD 指令支持检测等硬件层错误。
    /// **恢复建议**: 
    /// - GPU 内存不足：减小 batch size 或启用 KV Cache 分页
    /// - SIMD 不支持：编译时禁用 SIMD 或更新 CPU
    /// - CUDA 错误：检查驱动版本和 CUDA toolkit 兼容性
    #[error("Hardware error: {0}")]
    Hardware(#[from] HardwareError),

    /// 配置管理错误 (错误码前缀: CFG)
    ///
    /// 涵盖 TOML/YAML 解析、配置值验证、必填项缺失等配置问题。
    /// **恢复建议**: 检查 `server.toml` 语法，参考文档中的配置示例。
    #[error("Config error: {0}")]
    Config(#[from] ConfigError),

    /// I/O 操作错误
    ///
    /// 封装标准库 `std::io::Error`，涵盖文件读写、网络通信等。
    /// **恢复建议**: 检查文件权限、磁盘空间、网络连接状态。
    /// 
    /// **注意**: 此变体在 TypeScript 导出时被跳过，
    /// 因为 `std::io::Error` 不支持 ts-rs 序列化。
    #[error("IO error: {0}")]
    #[ts(skip)]
    Io(#[from] std::io::Error),

    /// 内部逻辑错误
    ///
    /// 用于未分类的内部错误或需要附加上下文的错误信息。
    /// **注意**: 应尽量避免使用此变体，优先定义具体的错误类型。
    #[error("Internal error: {0}")]
    Internal(String),
}

/// 推理引擎错误类型 (错误码前缀: ENG)
///
/// 涵盖模型推理过程中的核心计算错误，主要涉及：
/// - KV Cache 维度和行数验证
/// - 数组内存布局检查
/// - 流式注意力机制写入操作
///
/// # 常见场景
///
/// 1. **模型加载阶段** - KV Cache 维度不匹配通常表示模型权重与配置不一致
/// 2. **推理执行阶段** - 流式写入失败可能由内存不足或并发冲突引起
/// 3. **批量推理** - 行数不匹配常发生在连续批处理（Continuous Batching）中
#[derive(Debug, Error, TS)]
#[ts(export)]
pub enum EngineError {
    /// KV Cache 维度不匹配 (ENG001)
    ///
    /// 当请求的隐藏层维度与模型实际维度不符时触发。
    ///
    /// # 可能原因
    /// - 配置文件中 `hidden_size` 设置错误
    /// - 加载了错误的模型权重文件
    /// - 模型量化导致维度变化未正确处理
    ///
    /// # 恢复建议
    /// 1. 检查模型配置文件的 `hidden_size` 字段
    /// 2. 确认加载的模型权重与配置一致
    /// 3. 如果使用量化模型，确保量化配置正确
    #[error("KV cache dimension mismatch: expected {expected}, got {actual}")]
    KvCacheDimensionMismatch {
        /// 期望的维度大小
        expected: usize,
        /// 实际获得的维度大小
        actual: usize,
    },

    /// KV Cache 行数不匹配 (ENG002)
    ///
    /// 当序列长度或批大小与预分配的缓存空间不匹配时触发。
    ///
    /// # 可能原因
    /// - 超过最大序列长度限制
    /// - 批处理动态调整导致的行数变化
    /// - Prefix Cache 共享时的行数计算错误
    ///
    /// # 恢复建议
    /// 1. 减小输入序列长度至 `max_seq_len` 以内
    /// 2. 检查 Continuous Batching 配置
    /// 3. 查看 Prefix Cache 的共享策略是否合理
    #[error("KV cache row mismatch: expected {expected}, got {actual}")]
    KvCacheRowMismatch {
        /// 期望的行数
        expected: usize,
        /// 实际获得的行数
        actual: usize,
    },

    /// 数组非连续存储 (ENG003)
    ///
    /// 当传入的 ndarray 不是行主序连续内存布局时触发。
    /// SIMD 优化要求连续内存以实现高效向量化操作。
    ///
    /// # 恢复建议
    /// - 对输入数组调用 `.to_owned()` 或 `.into_owned()` 创建连续副本
    /// - 避免对数组进行切片后再传给引擎
    #[error("Array not contiguous")]
    ArrayNotContiguous,

    /// 流式注意力写入失败 (ENG004)
    ///
    /// 在流式推理（Streaming Inference）过程中写入 KV Cache 时发生错误。
    ///
    /// # 可能原因
    /// - KV Cache 内存已满且无法分页
    /// - 并发写入冲突（多 token 同时写入同一位置）
    /// - 底层硬件故障（GPU 显存损坏）
    ///
    /// # 恢复建议
    /// 1. 启用 KV Cache 分页到 CPU 内存
    /// 2. 减小并发请求数量
    /// 3. 重启服务释放显存碎片
    #[error("Streaming attention write failed: {0}")]
    StreamingAttentionWriteFailed(String),
}

/// 工作线程/任务调度错误类型 (错误码前缀: WRK)
///
/// 涵盖任务调度系统中的并发和通信问题，主要涉及：
/// - 互斥锁操作（锁中毒检测）
/// - 任务队列通信（channel 发送/接收）
/// - 线程创建和管理（spawn 失败）
///
/// # 常见场景
///
/// 1. **高并发场景** - 锁中毒通常由 panic 导致的锁未释放引起
/// 2. **资源耗尽** - spawn 失败可能因达到操作系统线程数限制
/// 3. **关闭期间** - 通信错误常发生在调度器关闭后的任务提交
#[derive(Debug, Error, TS)]
#[ts(export)]
pub enum WorkerError {
    /// 互斥锁中毒 (WRK001)
    ///
    /// 当持有 Mutex 锁的线程发生 panic 时，Mutex 进入"中毒"状态。
    /// 这是 Rust 的安全机制，防止访问可能处于不一致状态的数据。
    ///
    /// # 可能原因
    /// - 任务执行过程中发生不可恢复的 panic
    /// - 依赖的库在持有锁时 panic
    ///
    /// # 恢复建议
    /// 1. 检查日志定位 panic 发生位置
    /// 2. 重启服务以重置所有锁状态
    /// 3. 修复导致 panic 的根本原因（通常是 unwrap() 或 expect() 调用）
    #[error("Mutex lock poisoned")]
    LockPoisoned,

    /// 通道通信错误 (WRK002)
    ///
    /// 任务调度器内部消息传递失败，可能是：
    /// - 接收端已关闭（调度器正在 shutdown）
    /// - 发送端已关闭（工作线程已退出）
    /// - 通道容量已满且超时
    ///
    /// # 恢复建议
    /// 1. 检查调度器是否正常运行（`scheduler.is_running()`）
    /// 2. 如果在关闭期间提交任务，这是预期行为
    /// 3. 增加队列容量或减小批处理大小
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// 线程创建失败 (WRK003)
    ///
    /// 操作系统无法创建新的线程来执行任务。
    ///
    /// # 可能原因
    /// - 达到系统最大线程数限制（Linux 默认 ~32K）
    /// - 内存不足无法分配线程栈（默认 8MB/线程）
    /// - 达到进程文件描述符限制
    ///
    /// # 恢复建议
    /// 1. 减小 `max_concurrent` 配置值
    /// 2. 增加 `ulimit -u` 和 `ulimit -n` 限制
    /// 3. 检查系统内存使用情况
    #[error("Worker spawn failed: {0}")]
    SpawnFailed(String),
}

/// 模型训练错误类型 (错误码前缀: TRN)
///
/// 涵盖模型训练全流程中的错误，包括数据加载、前向/反向传播、优化器更新等。
///
/// # 架构说明
///
/// 训练模块支持：
/// - GRPO (Group Relative Policy Optimization) 强化学习训练
/// - AMP (Automatic Mixed Precision) 混合精度训练
/// - 分布式检查点保存/加载
/// - 自定义数据加载器
///
/// # 常见错误模式
///
/// 1. **形状不匹配** - 张量操作时维度不一致（最常见）
/// 2. **数值溢出** - 梯度爆炸或 NaN 出现
/// 3. **内存不足** - 大 batch 或长序列导致 OOM
#[derive(Debug, Error, TS)]
#[ts(export)]
pub enum TrainingError {
    /// 张量操作失败 (TRN001)
    ///
    /// 通用的张量计算错误，涵盖：
    /// - 形状不匹配（矩阵乘法维度不一致）
    /// - 数据类型转换失败
    /// - 广播规则违反
    ///
    /// # 恢复建议
    /// 1. 打印张量形状进行调试：`println!("shape: {:?}", tensor.shape())`
    /// 2. 检查模型配置中的维度参数
    /// 3. 验证输入数据的预处理流程
    #[error("Tensor operation failed: {0}")]
    TensorOperationFailed(String),

    /// 梯度计算失败 (TRN002)
    ///
    /// 反向传播过程中出现错误，可能涉及：
    /// - 自动微分图构建失败
    /// - 梯度为 NaN 或 Inf
    /// - 内存不足导致梯度计算中断
    ///
    /// # 恢复建议
    /// 1. 启用梯度裁剪（gradient clipping）
    /// 2. 降低学习率防止梯度爆炸
    /// 3. 使用 AMP 减少显存占用
    #[error("Gradient computation failed: {0}")]
    GradientComputationFailed(String),

    /// 优化器步骤失败 (TRN003)
    ///
    /// 参数更新阶段出错，常见原因：
    /// - 学习率设置不当导致数值不稳定
    /// - Adam/W 动量状态损坏
    /// - 权重衰减配置错误
    ///
    /// # 恢复建议
    /// 1. 从最近检查点恢复训练
    /// 2. 尝试降低学习率 10 倍
    /// 3. 检查优化器超参数配置
    #[error("Optimizer step failed: {0}")]
    OptimizerStepFailed(String),

    /// 检查点操作失败 (TRN004)
    ///
    /// 模型权重保存或加载失败，可能原因：
    /// - 磁盘空间不足
    /// - 文件权限问题
    /// - 检查点格式不兼容（版本升级后）
    ///
    /// # 恢复建议
    /// 1. 检查磁盘空间：`df -h`
    /// 2. 确认检查点目录权限：`ls -la checkpoints/`
    /// 3. 如版本不兼容，使用迁移工具转换格式
    #[error("Checkpoint save/load failed: {0}")]
    CheckpointFailed(String),

    /// 数据加载失败 (TRN005)
    ///
    /// 训练数据读取或预处理出错，涵盖：
    /// - 文件不存在或格式错误
    /// - 数据解析异常（JSON/TOML/BIN 格式）
    /// - Tokenizer 初始化失败
    ///
    /// # 恢复建议
    /// 1. 验证数据文件路径和格式
    /// 2. 检查 Tokenizer 配置（vocab_path, special_tokens）
    /// 3. 查看数据加载器的日志输出定位具体行号
    #[error("Data loading failed: {0}")]
    DataLoadingFailed(String),
}

/// 硬件层错误类型 (错误码前缀: HW)
///
/// 涵盖与底层硬件交互的所有错误，包括 GPU、CPU SIMD、缓存管理等。
///
/// # 平台支持
///
/// | 平台 | 支持的后端 | 备注 |
/// |------|-----------|------|
/// | NVIDIA GPU | CUDA | 需要 CUDA Toolkit >= 11.8 |
/// | Apple Silicon | Metal | macOS 13+ |
/// | AMD GPU | Vulkan | 实验性支持 |
/// | x86_64 CPU | AVX2/SSE4.2 | 自动特性检测 |
/// | ARM64 CPU | NEON | aarch64 默认支持 |
///
/// # 性能调优提示
///
/// - **GPU 内存**: 启用 KV Cache 分页可显著降低显存需求
/// - **SIMD 加速**: 确保 CPU 支持 AVX2 可获得 4x Softmax 加速
/// - **多 GPU**: 当前版本单卡优化，多卡支持开发中
#[derive(Debug, Error, TS)]
#[ts(export)]
pub enum HardwareError {
    /// 内存分配失败 (HW001)
    ///
    /// 无法分配所需的设备内存（GPU 显存或 CPU RAM）。
    ///
    /// # GPU 显存不足时的解决方案
    /// 1. 减小 `max_batch_size`（推荐从 8 降至 4）
    /// 2. 减小 `max_seq_len`（如从 4096 降至 2048）
    /// 3. 启用 KV Cache 分页：`kv_cache.paging = true`
    /// 4. 使用量化模型（Q4_K_M 比 FP16 节省 75% 显存）
    ///
    /// # CPU 内存不足时的解决方案
    /// 1. 增加 `kv_cache.cpu_offload_bytes` 上限
    /// 2. 减少 `num_workers` 并发数
    /// 3. 启用内存压缩：`memory.compression = "zstd"`
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),

    /// GPU 操作失败 (HW002)
    ///
    /// CUDA/Metal/Vulkan API 调用返回错误。
    ///
    /// # 常见 CUDA 错误码
    /// - `cudaErrorOutOfMemory`: 显存不足 → 参考 HW001 解决方案
    /// - `cudaErrorInvalidValue`: 无效参数 → 检查张量形状
    /// - `cudaErrorLaunchTimeout`: kernel 执行超时 → 减小 batch size
    /// - `cudaErrorNoDevice`: 未找到 GPU → 安装/更新驱动
    ///
    /// # Metal 特有错误
    /// - 设备不支持：需要 Apple M1 及以上芯片
    /// - 命令缓冲区满：减少并行 kernel 数量
    #[error("GPU operation failed: {0}")]
    GpuOperationFailed(String),

    /// SIMD 指令不支持 (HW003)
    ///
    /// 当前 CPU 不支持请求的 SIMD 指令集。
    ///
    /// # 受影响的操作
    /// - AVX2: x86_64 Softmax、矩阵乘法加速（4x 提速）
    /// - NEON: ARM64 Softmax 加速（2x 提速）
    /// - SSE4.2: 字符串处理优化
    ///
    /// # 解决方案
    /// 1. **运行时回退**: 系统会自动降级到标量实现（功能正确但较慢）
    /// 2. **编译时禁用**: 在 `Cargo.toml` 中设置 `features = []`
    /// 3. **硬件升级**: 对于生产环境，建议使用支持 AVX2 的 CPU
    ///
    /// # 检测方法
    /// ```rust,ignore
    /// let features = SimdFeatures::detect();
    /// println!("AVX2: {}, NEON: {}", features.avx2, features.neon);
    /// ```
    #[error("SIMD operation not supported: {0}")]
    SimdNotSupported(String),

    /// 缓存操作失败 (HW004)
    ///
    /// KV Cache 或其他硬件缓存的读写操作失败。
    ///
    /// # 涉及的缓存组件
    /// - PagedAttention 页表管理
    /// - Prefix Cache 共享缓存
    /// - MLA (Multi-Layer Attention) 潜变量缓存
    /// - ESS (Efficient Storage System) 缓存预取
    ///
    /// # 恢复建议
    /// 1. 检查缓存配置是否合理（大小、分页策略）
    /// 2. 清空缓存并重启服务
    /// 3. 如果持续失败，考虑禁用高级缓存功能
    #[error("Cache operation failed: {0}")]
    CacheOperationFailed(String),
}

/// 配置管理错误类型 (错误码前缀: CFG)
///
/// 涵盖应用配置的解析、验证和缺失等问题。
///
/// # 配置文件位置
///
/// 1. **默认路径**: `./config/server.toml`
/// 2. **环境变量覆盖**: `OPENMINI_CONFIG_PATH=/path/to/config.toml`
/// 3. **命令行参数**: `openmini-server --config /path/to/config.toml`
///
/// # 配置验证规则
///
/// - 必填字段缺失会立即报错（如 `model.path`）
/// - 数值范围会在启动时验证（如 `server.port > 0 && port < 65536`）
/// - 类型错误会在 TOML 解析阶段捕获
#[derive(Debug, Error, TS)]
#[ts(export)]
pub enum ConfigError {
    /// 配置解析失败 (CFG001)
    ///
    /// TOML/YAML 文件语法错误或格式无效。
    ///
    /// # 常见原因
    /// - 缺少引号的字符串值
    /// - 不正确的数组/表格语法
    /// - UTF-8 BOM 或特殊字符
    /// - 注释符号使用错误（TOML 用 `#`）
    ///
    /// # 调试方法
    /// 1. 使用在线 TOML 验证器检查语法
    /// 2. 对比文档中的示例配置
    /// 3. 查看完整错误信息中的行列号
    #[error("Configuration parsing failed: {0}")]
    ParsingFailed(String),

    /// 配置值无效 (CFG002)
    ///
    /// 配置项的值不符合业务规则要求。
    ///
    /// # 常见验证规则
    /// - `server.port`: 必须在 1-65535 范围内
    /// - `model.hidden_size`: 必须是 64 的倍数（为了 SIMD 对齐）
    /// - `kv_cache.max_pages`: 不能超过物理内存的 80%
    /// - `training.learning_rate`: 必须 > 0 且 < 10.0
    ///
    /// # 示例错误信息
    /// ```text
    /// Invalid configuration value: server.port must be in range [1, 65535], got 70000
    /// ```
    #[error("Invalid configuration value: {0}")]
    InvalidValue(String),

    /// 必填配置缺失 (CFG003)
    ///
    /// 运行所必需的配置项未被提供。
    ///
    /// # 关键必填项列表
    /// - `model.name`: 模型名称
    /// - `model.path`: 模型权重文件路径
    /// - `tokenizer.path`: Tokenizer 文件路径
    /// - `server.host`: 监听地址
    ///
    /// # 可选项（有默认值）
    /// - `server.port`: 默认 8000
    /// - `batch_size`: 默认 8
    /// - `max_seq_len`: 默认 2048
    ///
    /// # 快速修复
    /// 复制 `config/server.toml.example` 为 `server.toml` 并填入必要值
    #[error("Missing required configuration: {0}")]
    MissingConfig(String),
}
