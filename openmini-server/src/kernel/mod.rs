//! OpenMini 内核层
//!
//! 提供高性能的底层算子实现，包括：
//! - CPU优化内核（SIMD）
//! - GPU内核（CUDA/Metal/Vulkan）
//! - 混合精度计算
//! - 内存优化

pub mod cpu;
pub mod ops;
pub mod executor;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

pub use ops::*;
pub use executor::*;
