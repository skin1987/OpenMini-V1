//! 内存模块入口
//!
//! 导出内存管理相关的子模块和类型

pub mod arena;
pub mod pool;
pub mod monitor;
pub mod mmap;
pub mod adaptive;
pub mod memory;

pub use monitor::MemoryMonitor;
#[allow(unused_imports)]
pub use adaptive::AdaptiveMemoryManager;
pub use memory::MemoryManager;
