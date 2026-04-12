# OpenMini-V1 混合架构重构设计方案

## 📋 文档信息

| 属性 | 值 |
|------|-----|
| **版本** | v1.0 |
| **日期** | 2026-04-12 |
| **状态** | 待审批 |
| **作者** | OpenMini 架构团队 |
| **类型** | 架构决策记录 (ADR) |

---

## 1. 背景与动机

### 1.1 问题陈述

OpenMini-V1 当前采用纯 Rust 实现，虽然具备内存安全、并发优势，但在 LLM 推理场景下存在显著性能差距：

```
性能对比 (7B模型, Apple M2 Pro / NVIDIA A100):

                    OpenMini-V1     llama.cpp     vLLM        差距倍数
CPU推理速度:        ~8-12 t/s       ~20-30 t/s    N/A         2-3x ⚠️
GPU推理速度:        ~25-40 t/s      ~80-120 t/s   ~100-150    3-4x 🔴
首token延迟:        ~200-400ms      ~50-100ms     ~30-60ms    4-6x 🔴
内存占用(7B Q4):    ~14GB           ~5-6GB        ~4-5GB      2-3x ⚠️
能效比:             基准            2-3x更好      3-4x更好    显著差距
```

### 1.2 根因分析

| 根因类别 | 具体问题 | 影响 |
|---------|---------|------|
| **抽象层过多** | Rust → Candle → BLAS/GPU Driver | 每层有10-20%开销累积 |
| **量化实现** | 反量化到F32再计算，非原生量化计算 | 内存和计算浪费 |
| **GPU控制** | 通过cudarc/metal-rs绑定，无法精细调优 | 无法利用硬件特性 |
| **SIMD优化** | 安全包装限制了激进优化 | 未达到硬件极限 |
| **内核粒度** | 通用算子，无平台专用优化 | 无法匹配llama.cpp级别性能 |

### 1.3 设计目标

**核心目标**: 构建 Rust + C/C++ 混合架构，实现：
- ✅ 推理速度提升 **3-5x**（达到主流框架水平）
- ✅ 内存占用降低 **50%+**
- ✅ 首token延迟 < **100ms**
- ✅ 保留 Rust 的服务层优势（安全、并发、开发效率）
- ✅ 获得 C/C++ 的计算性能优势（底层控制、成熟生态）

---

## 2. 目标架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OpenMini-V2 混合架构                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                     Layer 4: 应用服务层 (Rust)                         │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ HTTP REST   │  │ gRPC API    │  │ Admin Panel │  │ CLI Tools   │  │   │
│  │  │ (Axum:8080) │  │ (Tonic:50051│  │ (Vue3)      │  │             │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │   │
│  └─────────┼────────────────┼────────────────┼────────────────┼─────────┘   │
│            └────────────────┼────────────────┼────────────────┘             │
│                             ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                     Layer 3: 业务逻辑层 (Rust)                         │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Gateway     │  │ Scheduler   │  │ Worker Pool │  │ Auth/RBAC   │  │   │
│  │  │ Router      │  │ TaskQueue   │  │ Async Pool  │  │ Middleware  │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────────────┘  │   │
│  └─────────┼────────────────┼────────────────┼─────────────────────────┘   │
│            └────────────────┼────────────────┘                               │
│                             ▼                                              │
│  ════════════════════════════════════════════════════════════════════════    │
│                        FFI Boundary (C ABI)                                │
│  ════════════════════════════════════════════════════════════════════════    │
│                             ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                   Layer 2: 计算引擎层 (C/C++)                          │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │  │                  openmini-engine (C++)                          │ │   │
│  │  │                                                                 │ │   │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────────┐ │ │   │
│  │  │  │ Model     │ │ Inference │ │ Quant     │ │ KV Cache        │ │ │   │
│  │  │  │ Loader    │ │ Engine    │ │ Engine    │ │ Manager         │ │ │   │
│  │  │  │ (GGUF)    │ │           │ │ (LUT)     │ │ (Paged)         │ │ │   │
│  │  │  └───────────┘ └─────┬─────┘ └───────────┘ └─────────────────┘ │ │   │
│  │  │                       │                                         │ │   │
│  │  │  ┌────────────────────┼─────────────────────────────────────┐  │ │   │
│  │  │  │              Kernel Abstraction Layer                     │  │ │   │
│  │  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │  │ │   │
│  │  │  │  │ CPU     │ │ CUDA    │ │ Metal   │ │ Vulkan (Future) │ │  │ │   │
│  │  │  │  │ Backend │ │ Backend │ │ Backend │ │                 │ │  │ │   │
│  │  │  │  │ (BLAS/  │ │ (cuBLAS │ │ (MPS)   │ │                 │ │  │ │   │
│  │  │  │  │ SIMD)   │ │ +Custom)│ │ +Shader)│ │                 │ │  │ │   │
│  │  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘ │  │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │   │
│  │  └─────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                       │   │
│  │  可选集成:                                                             │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                           │   │
│  │  │ llama.cpp (子模块)│  │ bitnet.cpp (参考) │                           │   │
│  │  └──────────────────┘  └──────────────────┘                           │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                             ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                   Layer 1: 硬件驱动层 (OS/厂商)                        │   │
│  │                                                                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │   │
│  │  │ NVIDIA   │  │ Apple    │  │ Intel/AMD │  │ ARM Embedded         │  │   │
│  │  │ CUDA     │  │ Metal    │  │ OneAPI   │  │ (NEON/DOTPROD)       │  │   │
│  │  │ Driver   │  │ Framework│  │ /OpenCL  │  │                      │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 分层职责定义

#### **Layer 4: 应用服务层 (Rust)**

**职责**:
- HTTP/gRPC API 端点处理
- 请求验证、路由、中间件
- 响应格式化、错误处理
- 监控指标暴露
- 配置管理接口

**保留模块**:
```rust
openmini-server/src/
├── service/http/          // Axum HTTP服务 (保留)
├── service/grpc/          // Tonic gRPC服务 (保留)
├── service/server/        // Gateway路由 (保留)
├── monitoring/            // Prometheus监控 (保留)
├── config/                // 配置管理 (保留)
└── error.rs               // 统一错误处理 (保留)
```

#### **Layer 3: 业务逻辑层 (Rust)**

**职责**:
- 任务调度与队列管理
- Worker进程池管理
- 并发请求处理
- 会话管理
- 认证授权
- 缓存策略

**保留模块**:
```rust
openmini-server/src/
├── service/scheduler/     // 任务调度器 (保留)
├── service/worker/        // Worker池 (保留)
├── db/                    // 数据库抽象 (保留)
├── enterprise/            // 企业功能 (保留)
└── logging/               // 日志系统 (保留)
```

#### **Layer 2: 计算引擎层 (C/C++) ⭐ 新增**

**职责**:
- 模型加载与解析 (GGUF/Safetensors)
- 张量运算 (GEMM, GEMV, BiasAdd)
- 注意力计算 (Flash Attention, Multi-Head)
- 激活函数 (ReLU, SiLU, GeLU)
- 归一化 (LayerNorm, RMSNorm)
- 量化推理 (INT4/INT8/FP8/1.58-bit)
- KV Cache 管理 (Paged Attention)
- Sampling (Top-K, Top-P, Temperature)
- Tokenizer 绑定

**新增目录结构**:
```
openmini-server/native/              # C/C++ 计算引擎
├── CMakeLists.txt                   # 构建配置
├── include/
│   ├── openmini.h                   # 公共API头文件
│   ├── openmini_model.h             # 模型操作API
│   ├── openmini_inference.h         # 推理API
│   ├── openmini_quant.h             # 量化API
│   └── openmini_config.h            # 配置API
├── src/
│   ├── engine.cpp                   # 主引擎实现
│   ├── model_loader.cpp             # GGUF模型加载
│   ├── inference.cpp                # 推理执行
│   ├── quant_lut.cpp                # LUT量化方法
│   ├── kv_cache.cpp                 # Paged KV Cache
│   ├── sampling.cpp                 # 采样算法
│   └── tokenizer_wrapper.cpp        # Tokenizer绑定
├── kernels/                         # 平台专用内核
│   ├── cpu/
│   │   ├── gemm_avx2.cpp            # AVX2 GEMM
│   │   ├── gemm_neon.cpp            # NEON GEMM
│   │   ├── attention.cpp            # CPU Attention
│   │   └── quant_simd.cpp           # SIMD量化算子
│   ├── cuda/
│   │   ├── gemm.cu                  # CUDA GEMM Kernel
│   │   ├── attention.cu             # Flash Attention
│   │   ├── quant_kernels.cu         # 量化Kernel
│   │   └── kv_cache.cu              # GPU KV Cache
│   └── metal/
│       ├── gemm.metal               # Metal Shader
│       ├── attention.metal          # Metal Attention
│       └── quant.metal              # Metal量化
├── third_party/                     # 第三方依赖
│   ├── llama.cpp/                   # (git submodule)
│   └── ggml/                        # GGML张量库
└── tests/
    ├── test_engine.cpp
    ├── test_quant.cpp
    └── benchmark.cpp
```

#### **Layer 1: 硬件驱动层 (OS/厂商提供)**

由操作系统或硬件厂商提供的驱动程序，无需项目维护。

---

## 3. FFI 接口规范

### 3.1 接口设计原则

1. **最小化跨边界调用**: 减少FFI调用频率，批量传递数据
2. **零拷贝优先**: 使用共享内存或指针传递大数据
3. **错误码返回**: C层返回错误码，Rust层转换为Result
4. **生命周期明确**: 明确资源的所有权和释放时机
5. **线程安全**: C层内部处理线程同步，对外提供线程安全保证

### 3.2 核心 C API 定义

```cpp
// ============================================================================
// openmini.h - 公共API头文件
// ============================================================================

#ifndef OPENMINI_H
#define OPENMINI_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 类型定义
// ============================================================================

/// 不透明句柄类型 (OOP风格封装)
typedef struct openmini_context openmini_context_t;
typedef struct openmini_model openmini_model_t;
typedef struct openmini_batch openmini_batch_t;

/// 错误码
typedef enum {
    OPENMINI_OK = 0,
    OPENMINI_ERR_MEMORY = -1,
    OPENMINI_ERR_INVALID_PARAM = -2,
    OPENMINI_ERR_MODEL_LOAD = -3,
    OPENMINI_ERR_INFERENCE = -4,
    OPENMINI_ERR_UNSUPPORTED = -5,
    OPENMINI_ERR_CUDA = -6,
    OPENMINI_ERR_METAL = -7,
} openmini_error_t;

/// 配置结构体
typedef struct {
    // 模型配置
    uint32_t n_ctx;              // 上下文长度
    uint32_t n_batch;            // 批大小
    uint32_t n_threads;          // CPU线程数
    
    // 硬件配置
    int gpu_device_id;           // GPU设备ID (-1=CPU only)
    
    // 量化配置
    char quant_type[16];         // "f16", "q4_0", "q4_k", "i2_s", "tl1", "tl2"
    
    // 性能配置
    bool enable_kv_cache;        // 启用KV Cache
    bool enable_flash_attn;      // 启用Flash Attention
    uint32_t kv_cache_size_mb;   // KV Cache大小(MB)
    
    // BitNet特定配置
    bool use_bitnet_lut;         // 使用LUT方法
    uint32_t lut_block_size;     // LUT分块大小
} openmini_config_t;

/// 推理结果
typedef struct {
    int* tokens;                 // 生成的token数组
    size_t n_tokens;             // token数量
    float* logits;               // 最后一个logits (可选)
    size_t n_logits;             // logits数量
    char* text;                  // 解码后的文本 (可选)
    uint64_t duration_us;        // 推理耗时(微秒)
    uint64_t prompt_tokens;      // 处理的prompt token数
    uint64_t output_tokens;      // 生成的output token数
} openmini_result_t;

// ============================================================================
// 生命周期管理
// ============================================================================

/// 初始化全局状态 (调用一次)
openmini_error_t openmini_init(void);

/// 清理全局状态
void openmini_cleanup(void);

/// 获取版本信息
const char* openmini_version(void);

/// 获取支持的硬件后端列表
const char* const* openmini_get_backends(size_t* count);

// ============================================================================
// 模型操作
// ============================================================================

/// 加载模型
openmini_error_t openmini_load_model(
    const char* model_path,
    const openmini_config_t* config,
    openmini_model_t** out_model
);

/// 卸载模型
void openmini_free_model(openmini_model_t* model);

/// 获取模型信息
openmini_error_t openmini_model_info(
    const openmini_model_t* model,
    size_t* n_params,            // 参数量
    size_t* n_layers,            // 层数
    size_t* n_head,              // 注意力头数
    size_t* n_embd,              // 嵌入维度
    size_t* n_vocab              // 词表大小
);

// ============================================================================
// 推理上下文
// ============================================================================

/// 创建推理上下文
openmini_error_t openmini_new_context(
    const openmini_model_t* model,
    openmini_context_t** out_ctx
);

/// 释放推理上下文
void openmini_free_context(openmini_context_t* ctx);

/// 重置上下文 (清空KV Cache)
openmini_error_t openmini_reset_context(openmini_context_t* ctx);

// ============================================================================
// 推理执行
// ============================================================================

/// 执行单次推理 (最常用的高层API)
openmini_error_t openmini_generate(
    openmini_context_t* ctx,
    const char* prompt,          // 输入prompt文本
    int max_tokens,              // 最大生成token数
    float temperature,           // 温度参数
    float top_p,                 // Top-P采样
    int top_k,                   // Top-K采样
    openmini_result_t* result    // 输出结果
);

/// 执行Token级推理 (低层API，用于流式输出)
openmini_error_t openmini_decode(
    openmini_context_t* ctx,
    const int* tokens,           // 输入token数组
    size_t n_tokens,             // token数量
    bool preprocess_only         // 是否只做预处理(不生成新token)
);

/// 获取下一个token (配合decode使用)
int openmini_sample_token(
    openmini_context_t* ctx,
    float temperature,
    float top_p,
    int top_k,
    float* out_logit             // 可选: 输出logits
);

/// 批量推理 (用于Continuous Batching)
openmini_error_t openmini_decode_batch(
    openmini_context_t** contexts,
    size_t n_contexts,
    const int** tokens_array,
    const size_t* n_tokens_array
);

// ============================================================================
// 结果管理
// ============================================================================

/// 释放结果资源
void openmini_free_result(openmini_result_t* result);

// ============================================================================
// 性能统计
// ============================================================================

typedef struct {
    uint64_t total_prompt_us;    // Prompt处理总耗时
    uint64_t total_gen_us;       // 生成总耗时
    uint64_t total_tokens;       // 总token数
    float tokens_per_second;     // 吞吐量(t/s)
    float memory_used_mb;        // 内存使用(MB)
    float gpu_memory_used_mb;    // GPU显存使用(MB)
    uint32_t cache_hit_rate;     // KV Cache命中率(%)
} openmini_stats_t;

/// 获取性能统计
openmini_error_t openmini_get_stats(
    const openmini_context_t* ctx,
    openmini_stats_t* stats
);

/// 重置统计计数器
void openmini_reset_stats(openmini_context_t* ctx);

// ============================================================================
// BitNet扩展API (可选编译)
// ============================================================================

#ifdef OPENMINI_ENABLE_BITNET

/// 加载BitNet专用LUT内核
openmini_error_t openmini_bitnet_load_lut_kernel(
    const char* kernel_path,
    const char* kernel_type      // "tl1" or "tl2"
);

/// 设置BitNet LUT参数
openmini_error_t openmini_bitnet_set_lut_params(
    uint32_t row_block_size,
    uint32_t col_block_size,
    uint32_t parallel_size
);

#endif // OPENMINI_ENABLE_BITNET

#ifdef __cplusplus
}
#endif

#endif // OPENMINI_H
```

### 3.3 Rust FFI 绑定层

```rust
// ============================================================================
// src/ffi/mod.rs - Rust FFI 绑定层
// ============================================================================

use std::ffi::{CString, CStr};
use std::ptr;

mod bindings {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// ============================================================================
// 安全包装: 将C API包装为Rust惯用接口
// ============================================================================

pub struct OpenMiniEngine {
    inner: *mut bindings::openmini_context_t,
    model: *mut bindings::openmini_model_t,
}

impl OpenMiniEngine {
    /// 创建新的推理引擎实例
    pub fn new(model_path: &str, config: &EngineConfig) -> Result<Self> {
        unsafe {
            // 初始化全局状态
            let ret = bindings::openmini_init();
            if ret != bindings::openmini_error_t_OPENMINI_OK {
                return Err(Error::InitFailed);
            }
            
            let c_path = CString::new(model_path)?;
            let c_config = config.to_c_struct();
            
            let mut model: *mut bindings::openmini_model_t = ptr::null_mut();
            let ret = bindings::openmini_load_model(
                c_path.as_ptr(),
                &c_config,
                &mut model
            );
            
            if ret != bindings::openmini_error_t_OPENMINI_OK {
                return Err(Error::ModelLoadFailed);
            }
            
            let mut ctx: *mut bindings::openmini_context_t = ptr::null_mut();
            let ret = bindings::openmini_new_context(model, &mut ctx);
            
            if ret != bindings::openmi_error_t_OPENMINI_OK {
                return Err(Error::ContextCreationFailed);
            }
            
            Ok(Self { inner: ctx, model })
        }
    }
    
    /// 同步生成 (阻塞调用，应在spawn_blocking中使用)
    pub fn generate(&self, request: &GenerateRequest) -> Result<GenerateResponse> {
        unsafe {
            let prompt = CString::new(request.prompt.clone())?;
            let mut result = std::mem::zeroed();
            
            let ret = bindings::openmini_generate(
                self.inner,
                prompt.as_ptr(),
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.top_k,
                &mut result
            );
            
            if ret != bindings::openmini_error_t_OPENMINI_OK {
                return Err(Error::InferenceFailed);
            }
            
            let text = if !result.text.is_null() {
                Some(CStr::from_ptr(result.text).to_string_lossy().into_owned())
            } else {
                None
            };
            
            Ok(GenerateResponse {
                text,
                tokens_per_second: result.duration_us as f64 / result.output_tokens as f64 / 1e6,
                prompt_tokens: result.prompt_tokens,
                output_tokens: result.output_tokens,
            })
        }
    }
    
    /// 异步生成 (自动在spawn_blocking中执行)
    pub async fn generate_async(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let engine_ptr = self.inner as usize;
        let req = request.clone();
        
        tokio::task::spawn_blocking(move || {
            let engine = engine_ptr as *mut bindings::openmini_context_t;
            // ... 调用generate
        }).await?
    }
    
    /// 获取性能统计
    pub fn stats(&self) -> Result<EngineStats> {
        unsafe {
            let mut stats = std::mem::zeroed();
            let ret = bindings::openmini_get_stats(self.inner, &mut stats);
            
            if ret != bindings::openmini_error_t_OPENMINI_OK {
                return Err(Error::StatsFailed);
            }
            
            Ok(EngineStats {
                tokens_per_second: stats.tokens_per_second,
                memory_used_mb: stats.memory_used_mb,
                gpu_memory_used_mb: stats.gpu_memory_used_ms,
                cache_hit_rate: stats.cache_hit_rate,
            })
        }
    }
}

impl Drop for OpenMiniEngine {
    fn drop(&mut self) {
        unsafe {
            if !self.inner.is_null() {
                bindings::openmini_free_context(self.inner);
            }
            if !self.model.is_null() {
                bindings::openmini_free_model(self.model);
            }
        }
    }
}
```

---

## 4. 数据流设计

### 4.1 推理请求完整流程

```
用户请求 → HTTP POST /v1/completions
    ↓
[Layer 4] HTTP Handler (Rust/Axum)
    ├─ 解析JSON请求体
    ├─ 参数验证
    ├─ 身份认证检查
    └─ 构造 GenerateRequest
    ↓
[Layer 3] Scheduler (Rust/Tokio)
    ├─ 放入任务队列
    ├─ 选择可用Worker
    └─ 分配到Worker线程
    ↓
[FFI Boundary] spawn_blocking()
    ↓
[Layer 2] C/C++ Engine
    ├─ 1. Tokenization
    │   └─ 调用Tokenizer (sentencepiece/huggingface)
    │
    ├─ 2. Prompt Processing (预填充)
    │   ├─ Embedding Lookup
    │   ├─ Transformer Layers × N
    │   │   ├─ Attention (Flash/MHA)
    │   │   ├─ FFN (GEMM + Activation)
    │   │   └─ RMSNorm
    │   └─ 更新KV Cache
    │
    ├─ 3. Token Generation (自回归解码循环)
    │   ├─ Loop for i in 0..max_tokens:
    │   │   ├─ Last Token → Transformer
    │   │   ├─ Logits → Sampling (Top-K/Top-P)
    │   │   ├─ Sample Next Token
    │   │   └─ Update KV Cache
    │   └─ EOS check
    │
    └─ 4. Detokenization
        └─ Tokens → Text
    ↓
[FFI Boundary] 返回结果
    ↓
[Layer 3] Worker (Rust)
    ├─ 包装结果为 Response
    └─ 更新统计指标
    ↓
[Layer 4] HTTP Response
    └─ JSON序列化返回给客户端
```

### 4.2 内存布局优化

```
传统方式 (当前):
┌──────────────────────────────────────────────┐
│ Model Weights (F32)                          │  ← 4 bytes/param
│   ↓ 反量化                                   │
│ Activations (F32)                            │  ← 4 bytes/element  
│   ↓ 计算                                     │
│ Output (F32)                                 │
└──────────────────────────────────────────────┘
总内存: ~14GB for 7B model (F32)

优化后 (混合架构):
┌──────────────────────────────────────────────┐
│ Model Weights (Q4_K / I2_S)                  │  ← 0.75-1.58 bits/param
│   ↓ 直接查表/LUT计算 (无需反量化)              │
│ Activations (F16 / Q8_0)                     │  ← 2-1 bytes/element
│   ↓ 原生量化计算                              │
│ Output (F32)                                 │
└──────────────────────────────────────────────┘
总内存: ~4-6GB for 7B model (Q4)  ← 降低60%+
```

---

## 5. 构建系统集成

### 5.1 CMake 配置 (native/)

```cmake
# native/CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(openmini-native LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# 编译选项
option(OPENMINI_ENABLE_CUDA "Enable CUDA support" OFF)
option(OPENMINI_ENABLE_METAL "Enable Metal support" ON)
option(OPENMINI_ENABLE_BITNET "Enable BitNet LUT support" ON)

# 依赖: llama.cpp (作为submodule或独立库)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# 核心库
add_library(openmini-core STATIC
    src/engine.cpp
    src/model_loader.cpp
    src/inference.cpp
    src/quant_lut.cpp
    src/kv_cache.cpp
    src/sampling.cpp
)

target_include_directories(openmini-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
)

# CPU 内核
add_library(openmini-kernel-cpu STATIC
    kernels/cpu/gemm_avx2.cpp
    kernels/cpu/gemm_neon.cpp
    kernels/cpu/attention.cpp
    kernels/cpu/quant_simd.cpp
)

target_compile_options(openmini-kernel-cpu PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mavx2>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-mfma>
)

# CUDA 支持
if(OPENMINI_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(openmini-kernel-cuda STATIC
        kernels/cuda/gemm.cu
        kernels/cuda/attention.cu
        kernels/cuda/quant_kernels.cu
        kernels/cuda/kv_cache.cu
    )
endif()

# Metal 支持
if(OPENMINI_ENABLE_METAL AND APPLE)
    find_library(METAL_FRAMEWORK REQUIRED Metal)
    target_sources(openmini-kernel-cpu PRIVATE
        kernels/metal/gemm.metal
        kernels/metal/attention.metal
    )
endif()

# BitNet LUT 支持
if(OPENMINI_ENABLE_BITNET)
    target_compile_definitions(openmini-core PUBLIC OPENMINI_ENABLE_BITNET)
    target_sources(openmini-core PRIVATE src/bitnet_lut_impl.cpp)
endif()

# 共享库输出 (供Rust链接)
add_library(openmini SHARED
    src/openmini_api.cpp  # C API实现
)
target_link_libraries(openmini 
    openmini-core
    openmini-kernel-cpu
    llama          # llama.cpp
    pthread
    dl
)

if(OPENMINI_ENABLE_CUDA)
    target_link_libraries(openmini openmini-kernel-cuda cudart cublas)
endif()

# 安装规则
install(TARGETS openmini LIBRARY DESTINATION lib)
install(FILES include/openmini.h DESTINATION include)
```

### 5.2 Cargo.toml 更新

```toml
# openmini-server/Cargo.toml (更新部分)

[build-dependencies]
bindgen = "0.69"           # 自动生成Rust FFI绑定
cc = "1.0"                 # 编译C/C++代码

[dependencies]
# 移除: candle-core, candle-transformers (不再需要)
# 保留: tokio, axum, tonic 等服务层依赖

[[bin]]
name = "openmini-server"
path = "src/main.rs"

[build-script]
# build.rs 中调用cmake构建native库
```

### 5.3 build.rs 示例

```rust
// build.rs - 构建脚本
fn main() {
    // 检查环境变量
    println!("cargo:rerun-if-changed=native/");
    println!("cargo:rerun-if-changed=native/include/");
    
    // 运行cmake构建native库
    let dst = cmake::Config::new("native")
        .define("OPENMINI_ENABLE_METAL", "ON")
        .define("OPENMINI_ENABLE_BITNET", "ON")
        .build();
    
    // 输出链接路径
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=openmini");
    
    // macOS额外链接
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
    
    // 生成FFI绑定
    let bindings = bindgen::Builder::default()
        .header("native/include/openmini.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");
    
    bindings
        .write_to_file(std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
```

---

## 6. 性能预期与验证

### 6.1 目标性能指标

| 场景 | 当前 (纯Rust) | 目标 (混合架构) | 提升比例 |
|------|--------------|----------------|---------|
| **CPU推理 (7B Q4)** | 8-12 t/s | 25-35 t/s | **3x** |
| **GPU推理 (7B Q4, M2)** | 25-40 t/s | 80-120 t/s | **3x** |
| **首token延迟** | 200-400ms | 50-100ms | **4x** |
| **内存占用 (7B Q4)** | ~14GB | ~5-6GB | **60%↓** |
| **能效比** | 基准 | 2-3x提升 | **显著** |
| **BitNet 2B (W2A8)** | 不支持 | ~150 t/s (A100) | **新增能力** |

### 6.2 验证基准测试计划

```python
# tests/benchmark_suite.py

import time
import psutil
import subprocess

class BenchmarkSuite:
    """标准化的性能基准测试套件"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        
    def test_throughput(self, n_requests: int = 100):
        """吞吐量测试: tokens/sec"""
        
    def test_latency_p50_p99(self):
        """延迟分布: p50/p99/p999"""
        
    def test_memory_usage(self):
        """内存占用峰值"""
        
    def test_first_token_latency(self):
        """首token延迟"""
        
    def test_concurrent_users(self, n_users: int = 10):
        """并发用户测试"""
        
    def run_all(self) -> dict:
        """运行全部测试并生成报告"""
        results = {
            'throughput': self.test_throughput(),
            'latency': self.test_latency_p50_p99(),
            'memory': self.test_memory_usage(),
            'ttft': self.test_first_token_latency(),
            'concurrent': self.test_concurrent_users(),
        }
        return results
```

---

## 7. 风险评估与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| **FFI调用开销** | 中 | 中 | 批量操作、减少调用频率 |
| **内存安全问题** | 低 | 高 | C层严格测试+Rust安全包装 |
| **构建复杂度增加** | 高 | 中 | CI自动化+CMake模板化 |
| **调试困难** | 中 | 中 | 完善日志+coredump分析 |
| **团队技能缺口** | 中 | 高 | 培训+文档+渐进迁移 |
| **依赖版本冲突** | 中 | 低 | 固定版本+submodule锁定 |

---

## 8. 实施路线图概览

### Phase 1: 基础设施搭建 (第1-2周)
- [ ] 创建 `native/` 目录结构
- [ ] 实现 C API 头文件 (`openmini.h`)
- [ ] 配置 CMake 构建系统
- [ ] 集成 llama.cpp 作为基础
- [ ] 编写 Rust FFI 绑定层

### Phase 2: 核心引擎移植 (第3-6周)
- [ ] 实现模型加载 (GGUF解析)
- [ ] 实现基础推理流程 (Prompt Process + Decode)
- [ ] 移植量化支持 (Q4_K, I2_S)
- [ ] 实现 KV Cache 管理
- [ ] 端到端测试与性能对比

### Phase 3: 性能优化 (第7-10周)
- [ ] 引入 BitNet LUT 方法
- [ ] 实现平台专用内核 (TL1/TL2)
- [ ] GPU 加速 (CUDA/Metal)
- [ ] Flash Attention 集成
- [ ] 性能调优与基准测试

### Phase 4: 生产就绪 (第11-14周)
- [ ] 错误处理完善
- [ ] 监控与日志集成
- [ ] 文档编写
- [ ] 压力测试
- [ ] 发布准备

---

## 9. 决策记录

### ADR-001: 采用混合架构而非完全重写

**决策**: 采用 Rust + C/C++ 混合架构

**理由**:
1. 保留 Rust 在服务层的优势 (安全性、并发、开发效率)
2. 利用 C/C++ 在计算层的成熟生态 (llama.cpp, CUDA, 优化内核)
3. 渐进式迁移风险可控
4. 团队可复用现有 Rust 代码

**替代方案**:
- ❌ 完全重写为 C/C++: 成本高、丢失 Rust 优势
- ❌ 保持纯 Rust: 性能问题无法根本解决
- ⚠️ 仅通过子进程调用 llama.cpp: 集成度低、IPC开销大

### ADR-002: FFI边界位置选择

**决策**: FFI 边界位于 **业务逻辑层 与 计算引擎层** 之间

**理由**:
1. 上层保持纯 Rust，便于维护
2. 下层 C/C++ 可独立开发和测试
3. 接口稳定，变更频率低
4. 符合关注点分离原则

---

## 10. 附录

### A. 参考资源

- [llama.cpp](https://github.com/ggerganov/llama.cpp): 基础推理框架
- [bitnet.cpp](https://github.com/microsoft/BitNet): 1-bit LLM 推理
- [GGML](https://github.com/ggerganov/ggml): 张量库
- [Rust FFI Book](https://rust-lang.github.io/nomicon/ffi.html): FFI 指南
- [bindgen](https://rust-lang.github.io/bindgen/): 自动绑定生成

### B. 术语表

| 术语 | 解释 |
|------|------|
| FFI | Foreign Function Interface，外部函数接口 |
| LUT | Lookup Table，查找表 |
| GEMM | General Matrix Multiply，通用矩阵乘法 |
| GEMV | General Matrix-Vector Multiply，矩阵向量乘法 |
| KV Cache | Key-Value Cache，键值缓存 |
| TL1/TL2 | BitNet 的查找表内核变体 |
| W2A8 | 2-bit权重 × 8-bit激活量化 |

---

## 11. 审批签字

| 角色 | 姓名 | 日期 | 签字 |
|------|------|------|------|
| 架构师 | ______ | ____ | ____ |
| 技术负责人 | ______ | ____ | ____ |
| 项目经理 | ______ | ____ | ____ |

---

**文档结束**
