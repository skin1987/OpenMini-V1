//! 内存模块入口
//!
//! 导出内存管理相关的子模块和类型

pub mod adaptive;
pub mod arena;
pub mod memory;
pub mod mmap;
pub mod monitor;
pub mod pool;

#[allow(unused_imports)]
pub use adaptive::AdaptiveMemoryManager;
pub use memory::MemoryManager;
pub use monitor::MemoryMonitor;
