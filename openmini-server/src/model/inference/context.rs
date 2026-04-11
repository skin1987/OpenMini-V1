//! 推理上下文
//!
//! 管理推理过程中的 KV Cache、MLA 缓存、记忆等状态

#![allow(dead_code)]

use std::collections::TryReserveError as StdTryReserveError;

use crate::hardware::kv_cache::{
    mla::{config::MLAConfig, latent_cache::MLALatentCache},
    KVCache, KVCacheError,
};

use super::memory::{MemoryConfig, MemoryManager};

/// KV 缓存追加错误类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendError {
    /// per_token_size 未初始化
    NotInitialized,
    /// K 数据长度不匹配
    KLengthMismatch {
        /// 期望长度
        expected: usize,
        /// 实际长度
        actual: usize,
    },
    /// V 数据长度不匹配
    VLengthMismatch {
        /// 期望长度
        expected: usize,
        /// 实际长度
        actual: usize,
    },
    /// 缓存容量不足
    ///
    /// 当设置了 `max_tokens` 且当前 token 数量已达到最大容量时返回。
    /// 达到最大容量后不允许继续追加。
    CapacityExceeded {
        /// 当前 token 数量
        current: usize,
        /// 最大容量
        max: usize,
    },
}

/// KV 缓存初始化错误类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitError {
    /// 缓存已有数据，无法初始化
    CacheNotEmpty,
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::CacheNotEmpty => {
                write!(f, "Cache already contains tokens, call clear() first")
            }
        }
    }
}

/// 上下文错误类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextError {
    /// 层数不匹配
    LayerCountMismatch {
        /// 期望层数
        expected: usize,
        /// 实际层数
        actual: usize,
    },
    /// 容量溢出
    CapacityOverflow {
        /// 请求容量
        requested: usize,
        /// 最大容量
        max: usize,
    },
}

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextError::LayerCountMismatch { expected, actual } => {
                write!(
                    f,
                    "Layer count mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            ContextError::CapacityOverflow { requested, max } => {
                write!(
                    f,
                    "Capacity overflow: requested {} exceeds max {}",
                    requested, max
                )
            }
        }
    }
}

impl std::error::Error for ContextError {}

impl std::fmt::Display for AppendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppendError::NotInitialized => write!(f, "per_token_size not initialized"),
            AppendError::KLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "K length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            AppendError::VLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "V length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            AppendError::CapacityExceeded { current, max } => {
                write!(f, "Capacity exceeded: {}/{}", current, max)
            }
        }
    }
}

impl std::error::Error for AppendError {}

/// 尝试追加错误类型
#[derive(Debug)]
pub enum TryAppendError {
    /// 内存分配失败
    AllocationFailed(StdTryReserveError),
    /// 业务逻辑错误
    UserError(AppendError),
}

impl std::fmt::Display for TryAppendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TryAppendError::AllocationFailed(e) => write!(f, "memory allocation failed: {}", e),
            TryAppendError::UserError(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for TryAppendError {}

impl From<StdTryReserveError> for TryAppendError {
    fn from(e: StdTryReserveError) -> Self {
        TryAppendError::AllocationFailed(e)
    }
}

impl From<AppendError> for TryAppendError {
    fn from(e: AppendError) -> Self {
        TryAppendError::UserError(e)
    }
}

/// 推理上下文
///
/// 包含多层缓存和记忆管理
///
/// # 并发安全
/// 此类型自动实现 `Send + Sync`，允许在线程间移动和共享只读引用。
/// 但内部没有同步机制，**多线程共享可变引用需要外部同步**。
///
/// ## 正确用法示例
/// ```ignore
/// use std::sync::{Arc, Mutex};
///
/// // 多线程共享可变上下文
/// let ctx = Arc::new(Mutex::new(InferenceContext::new(num_layers, &config)?));
///
/// // 线程 1: 追加数据
/// {
///     let mut ctx = ctx.lock().unwrap();
///     // 操作上下文
/// }
///
/// // 线程 2: 读取数据
/// {
///     let ctx = ctx.lock().unwrap();
///     // 读取上下文
/// }
/// ```
#[derive(Debug)]
pub struct InferenceContext {
    /// 层缓存列表
    layer_caches: Vec<LayerCache>,
    /// 下一个要处理的位置
    next_position: usize,
    /// 最大序列长度
    max_seq_len: usize,
    /// 记忆管理器
    memory: MemoryManager,
}

/// 层缓存类型
#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum LayerCache {
    /// MLA 潜在缓存
    MLA(MLALatentCache),
    /// 标准 KV 缓存
    Standard(StandardKVCache),
}

impl LayerCache {
    /// 获取标准 KV 缓存的不可变引用
    pub fn as_standard(&self) -> Option<&StandardKVCache> {
        match self {
            LayerCache::Standard(cache) => Some(cache),
            LayerCache::MLA(_) => None,
        }
    }

    /// 获取标准 KV 缓存的可变引用
    pub fn as_standard_mut(&mut self) -> Option<&mut StandardKVCache> {
        match self {
            LayerCache::Standard(cache) => Some(cache),
            LayerCache::MLA(_) => None,
        }
    }

    /// 获取 MLA 缓存的不可变引用
    pub fn as_mla(&self) -> Option<&MLALatentCache> {
        match self {
            LayerCache::MLA(cache) => Some(cache),
            LayerCache::Standard(_) => None,
        }
    }

    /// 获取 MLA 缓存的可变引用
    pub fn as_mla_mut(&mut self) -> Option<&mut MLALatentCache> {
        match self {
            LayerCache::MLA(cache) => Some(cache),
            LayerCache::Standard(_) => None,
        }
    }

    /// 检查是否为 MLA 缓存
    pub fn is_mla(&self) -> bool {
        matches!(self, LayerCache::MLA(_))
    }

    /// 检查是否为标准 KV 缓存
    pub fn is_standard(&self) -> bool {
        matches!(self, LayerCache::Standard(_))
    }
}

/// 标准 KV 缓存
///
/// # 并发安全
/// 此类型自动实现 `Send + Sync`，允许在线程间移动和共享只读引用。
/// 但内部没有同步机制，**多线程共享可变引用需要外部同步**。
///
/// ## 正确用法示例
/// ```ignore
/// use std::sync::{Arc, Mutex};
///
/// // 多线程共享可变缓存
/// let cache = Arc::new(Mutex::new(StandardKVCache::new(128)));
///
/// // 线程 1: 追加数据
/// {
///     let mut cache = cache.lock().unwrap();
///     cache.append(&k, &v).unwrap();
/// }
///
/// // 线程 2: 读取数据
/// {
///     let cache = cache.lock().unwrap();
///     let k_data = cache.get_k(0);
/// }
/// ```
///
/// # 内存布局
/// K/V 数据以扁平方式存储在 `k_cache` 和 `v_cache` 中，每个 token 占用 `per_token_size` 个连续元素。
/// 例如，第 i 个 token 的 K 数据位于 `k_cache[i*per_token_size..(i+1)*per_token_size]`。
pub struct StandardKVCache {
    /// K 缓存
    k_cache: Vec<f32>,
    /// V 缓存
    v_cache: Vec<f32>,
    /// 当前 token 数量
    num_tokens: usize,
    /// 每个 token 的数据大小
    per_token_size: usize,
    /// 最大 token 容量（0 表示无限制）
    max_tokens: usize,
}

impl std::fmt::Debug for StandardKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StandardKVCache")
            .field("num_tokens", &self.num_tokens)
            .field("per_token_size", &self.per_token_size)
            .field("max_tokens", &self.max_tokens)
            .field("k_cache_len", &self.k_cache.len())
            .field("k_cache_capacity", &self.k_cache.capacity())
            .field("v_cache_len", &self.v_cache.len())
            .field("v_cache_capacity", &self.v_cache.capacity())
            .finish_non_exhaustive()
    }
}

impl Clone for StandardKVCache {
    fn clone(&self) -> Self {
        Self {
            k_cache: self.k_cache.clone(),
            v_cache: self.v_cache.clone(),
            num_tokens: self.num_tokens,
            per_token_size: self.per_token_size,
            max_tokens: self.max_tokens,
        }
    }
}

// 注意：`StandardKVCache` 自动满足 `Send` 和 `Sync`，
// 因为所有字段都是 `Vec<f32>` 和 `usize`，它们本身就是 `Send + Sync`。
// 无需显式 `unsafe impl`。

impl Default for StandardKVCache {
    /// 创建默认的空 KV 缓存
    ///
    /// 默认缓存 `per_token_size = 0`，表示未初始化状态。
    /// 调用者需要通过 `with_capacity` 或 `with_max_tokens` 创建可用的缓存。
    fn default() -> Self {
        Self {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            num_tokens: 0,
            per_token_size: 0,
            max_tokens: 0,
        }
    }
}

impl StandardKVCache {
    /// 创建新的标准 KV 缓存
    /// 创建带指定元素容量的 KV 缓存
    ///
    /// # 参数
    /// - `capacity`: **元素数量**（f32 数量），而非 token 数量
    /// - `per_token_size`: 每个 token 的数据大小
    ///
    /// # 注意
    /// 此方法的 `capacity` 参数表示元素数量，与 `with_max_tokens` 的 token 数量参数不同。
    /// 推荐使用 `with_token_capacity` 以避免混淆。
    #[deprecated(
        since = "0.2.0",
        note = "Use `with_token_capacity` instead to avoid confusion between element count and token count"
    )]
    pub fn with_capacity(capacity: usize, per_token_size: usize) -> Self {
        Self {
            k_cache: Vec::with_capacity(capacity),
            v_cache: Vec::with_capacity(capacity),
            num_tokens: 0,
            per_token_size,
            max_tokens: 0,
        }
    }

    /// 创建带指定 token 容量的 KV 缓存
    ///
    /// # 参数
    /// - `token_capacity`: **token 数量**
    /// - `per_token_size`: 每个 token 的数据大小
    ///
    /// # 返回
    /// 成功返回 `Ok(Self)`，失败返回 `Err(ContextError::CapacityOverflow)`
    ///
    /// # Errors
    /// 当 `token_capacity * per_token_size` 发生整数溢出时返回 `ContextError::CapacityOverflow`。
    /// 这通常发生在两个参数都很大的情况下（如 `token_capacity > usize::MAX / per_token_size`）。
    ///
    /// # Example
    /// ```ignore
    /// let cache = StandardKVCache::with_token_capacity(1024, 128)?;
    /// ```
    pub fn with_token_capacity(
        token_capacity: usize,
        per_token_size: usize,
    ) -> Result<Self, ContextError> {
        let capacity = token_capacity.checked_mul(per_token_size).ok_or_else(|| {
            ContextError::CapacityOverflow {
                requested: token_capacity.saturating_mul(per_token_size),
                max: usize::MAX,
            }
        })?;
        Ok(Self {
            k_cache: Vec::with_capacity(capacity),
            v_cache: Vec::with_capacity(capacity),
            num_tokens: 0,
            per_token_size,
            max_tokens: 0,
        })
    }

    /// 创建带最大容量限制的 KV 缓存
    pub fn with_max_tokens(max_tokens: usize, per_token_size: usize) -> Self {
        let capacity = max_tokens.saturating_mul(per_token_size);
        Self {
            k_cache: Vec::with_capacity(capacity),
            v_cache: Vec::with_capacity(capacity),
            num_tokens: 0,
            per_token_size,
            max_tokens,
        }
    }

    /// 获取每个 token 的数据大小
    pub fn per_token_size(&self) -> usize {
        self.per_token_size
    }

    /// 获取当前 token 数量
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// 初始化缓存
    ///
    /// # 参数
    /// - `per_token_size`: 每个 token 的数据大小
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，如果缓存已有数据（`num_tokens > 0`）返回错误
    ///
    /// # 注意
    /// - 如果缓存已有数据，调用此方法会返回错误，避免数据解释错误。
    /// - 如需重新初始化，请先调用 `clear()` 清除数据，或使用 `reinit()` 方法。
    pub fn init(&mut self, per_token_size: usize) -> Result<(), InitError> {
        if self.num_tokens > 0 {
            return Err(InitError::CacheNotEmpty);
        }
        self.per_token_size = per_token_size;
        Ok(())
    }

    /// 重新初始化缓存
    ///
    /// 清除所有数据并设置新的 `per_token_size`。此方法会释放已分配的内存。
    ///
    /// # 参数
    /// - `per_token_size`: 新的每个 token 的数据大小
    ///
    /// # 返回
    /// 成功返回 `Ok(())`
    ///
    /// # 注意
    /// 此方法会清除所有数据并释放内存。如需保留容量，请使用 `clear()` + `init()`。
    pub fn reinit(&mut self, per_token_size: usize) {
        self.k_cache = Vec::new();
        self.v_cache = Vec::new();
        self.num_tokens = 0;
        self.per_token_size = per_token_size;
    }

    /// 检查缓存是否已初始化
    pub fn is_initialized(&self) -> bool {
        self.per_token_size > 0
    }

    /// 获取最大 token 容量
    ///
    /// 返回 0 表示无限制
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// 检查缓存是否为空
    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    /// 追加 KV 数据
    ///
    /// # 参数
    /// - `k`: K 数据切片
    /// - `v`: V 数据切片
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，失败返回 `AppendError`
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> Result<(), AppendError> {
        if self.per_token_size == 0 {
            return Err(AppendError::NotInitialized);
        }

        let expected_len = self.per_token_size;
        if k.len() != expected_len {
            return Err(AppendError::KLengthMismatch {
                expected: expected_len,
                actual: k.len(),
            });
        }
        if v.len() != expected_len {
            return Err(AppendError::VLengthMismatch {
                expected: expected_len,
                actual: v.len(),
            });
        }

        if self.max_tokens > 0 && self.num_tokens >= self.max_tokens {
            return Err(AppendError::CapacityExceeded {
                current: self.num_tokens,
                max: self.max_tokens,
            });
        }

        self.k_cache.extend_from_slice(k);
        self.v_cache.extend_from_slice(v);
        self.num_tokens += 1;

        Ok(())
    }

    /// 按需增长并追加数据
    ///
    /// 当容量不足时自动扩容（每次翻倍）
    ///
    /// # 注意
    /// 如果设置了 `max_tokens` 且当前 token 数量已达到最大容量，
    /// 将返回 `AppendError::CapacityExceeded`。
    ///
    /// # 扩容策略
    /// 采用翻倍扩容策略（×2），这可能导致内存占用瞬间翻倍。
    /// 对于长序列生成（如 2048 tokens），建议使用 `reserve` 预分配足够容量，
    /// 以避免频繁扩容造成的内存峰值。
    ///
    /// # Panic
    /// **内存分配失败时会 panic**。生产环境建议使用 `try_append_with_grow`，
    /// 它使用 `try_reserve` 优雅处理 OOM。
    pub fn append_with_grow(&mut self, k: &[f32], v: &[f32]) -> Result<(), AppendError> {
        if self.per_token_size == 0 {
            return Err(AppendError::NotInitialized);
        }

        if self.max_tokens > 0 && self.num_tokens >= self.max_tokens {
            return Err(AppendError::CapacityExceeded {
                current: self.num_tokens,
                max: self.max_tokens,
            });
        }

        let expected_len = self.per_token_size;
        if k.len() != expected_len {
            return Err(AppendError::KLengthMismatch {
                expected: expected_len,
                actual: k.len(),
            });
        }
        if v.len() != expected_len {
            return Err(AppendError::VLengthMismatch {
                expected: expected_len,
                actual: v.len(),
            });
        }

        let required = (self.num_tokens + 1).saturating_mul(self.per_token_size);
        if self.k_cache.capacity() < required {
            let base_capacity = self.k_cache.capacity().max(required);
            let mut new_capacity = base_capacity.checked_mul(2).unwrap_or(base_capacity);
            if self.max_tokens > 0 {
                let max_capacity = self.max_tokens.saturating_mul(self.per_token_size);
                new_capacity = new_capacity.min(max_capacity);
            }
            let additional = new_capacity.saturating_sub(self.k_cache.len());
            if additional > 0 {
                self.k_cache.reserve(additional);
                self.v_cache.reserve(additional);
            }
        }

        self.k_cache.extend_from_slice(k);
        self.v_cache.extend_from_slice(v);
        self.num_tokens += 1;

        Ok(())
    }

    /// 尝试追加 KV 数据（优雅处理内存不足）
    ///
    /// 与 `append` 类似，但使用 `try_reserve` 来优雅处理内存分配失败。
    ///
    /// # 参数
    /// - `k`: K 数据切片
    /// - `v`: V 数据切片
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，失败返回 `TryAppendError`
    pub fn try_append(&mut self, k: &[f32], v: &[f32]) -> Result<(), TryAppendError> {
        if self.per_token_size == 0 {
            return Err(AppendError::NotInitialized.into());
        }

        let expected_len = self.per_token_size;
        if k.len() != expected_len {
            return Err(AppendError::KLengthMismatch {
                expected: expected_len,
                actual: k.len(),
            }
            .into());
        }
        if v.len() != expected_len {
            return Err(AppendError::VLengthMismatch {
                expected: expected_len,
                actual: v.len(),
            }
            .into());
        }

        if self.max_tokens > 0 && self.num_tokens >= self.max_tokens {
            return Err(AppendError::CapacityExceeded {
                current: self.num_tokens,
                max: self.max_tokens,
            }
            .into());
        }

        self.k_cache.try_reserve(expected_len)?;
        self.v_cache.try_reserve(expected_len)?;

        self.k_cache.extend_from_slice(k);
        self.v_cache.extend_from_slice(v);
        self.num_tokens += 1;

        Ok(())
    }

    /// 尝试按需增长并追加数据（优雅处理内存不足）
    ///
    /// 与 `append_with_grow` 类似，但使用 `try_reserve` 来优雅处理内存分配失败。
    ///
    /// # 参数
    /// - `k`: K 数据切片
    /// - `v`: V 数据切片
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，失败返回 `TryAppendError`
    ///
    /// # 内存分配行为
    /// 当需要扩容时，会先预留 K 缓存，再预留 V 缓存。如果 K 缓存预留成功但 V 缓存预留失败，
    /// K 缓存的容量已增大，造成少量内存浪费。建议使用 `reserve` 预分配足够容量以避免此问题。
    pub fn try_append_with_grow(&mut self, k: &[f32], v: &[f32]) -> Result<(), TryAppendError> {
        if self.per_token_size == 0 {
            return Err(AppendError::NotInitialized.into());
        }

        if self.max_tokens > 0 && self.num_tokens >= self.max_tokens {
            return Err(AppendError::CapacityExceeded {
                current: self.num_tokens,
                max: self.max_tokens,
            }
            .into());
        }

        let expected_len = self.per_token_size;
        if k.len() != expected_len {
            return Err(AppendError::KLengthMismatch {
                expected: expected_len,
                actual: k.len(),
            }
            .into());
        }
        if v.len() != expected_len {
            return Err(AppendError::VLengthMismatch {
                expected: expected_len,
                actual: v.len(),
            }
            .into());
        }

        let required = (self.num_tokens + 1).saturating_mul(self.per_token_size);
        if self.k_cache.capacity() < required {
            let base_capacity = self.k_cache.capacity().max(required);
            let mut new_capacity = base_capacity.checked_mul(2).unwrap_or(base_capacity);
            if self.max_tokens > 0 {
                let max_capacity = self.max_tokens.saturating_mul(self.per_token_size);
                new_capacity = new_capacity.min(max_capacity);
            }
            let additional = new_capacity.saturating_sub(self.k_cache.len());
            if additional > 0 {
                self.k_cache.try_reserve(additional)?;
                self.v_cache.try_reserve(additional)?;
            }
        }

        self.k_cache.extend_from_slice(k);
        self.v_cache.extend_from_slice(v);
        self.num_tokens += 1;

        Ok(())
    }

    /// 清空缓存
    ///
    /// 清除所有 K/V 数据，重置 `num_tokens` 为 0，但保留 `per_token_size` 和容量。
    ///
    /// # 与 `KVCache::clear_cache` 的区别
    /// - `clear()`: 无返回值，不会失败（`StandardKVCache` 专用）
    /// - `KVCache::clear_cache()`: 返回 `Result<(), KVCacheError>`（trait 统一接口）
    ///
    /// 如需通过 trait 调用，请使用 `KVCache::clear_cache(&mut cache)`。
    pub fn clear(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
        self.num_tokens = 0;
    }

    /// 截断缓存到指定 token 数量
    ///
    /// # 参数
    /// - `new_len`: 新的 token 数量
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，如果缓存未初始化返回 `Err(AppendError::NotInitialized)`
    ///
    /// # 注意
    /// - 如果 `new_len` 大于当前 token 数量，不做任何操作。
    /// - 如果 `new_len` 为 0，效果等同于 `clear()`。
    pub fn truncate(&mut self, new_len: usize) -> Result<(), AppendError> {
        if self.per_token_size == 0 {
            return Err(AppendError::NotInitialized);
        }
        if new_len >= self.num_tokens {
            return Ok(());
        }
        let new_data_len = new_len.saturating_mul(self.per_token_size);
        self.k_cache.truncate(new_data_len);
        self.v_cache.truncate(new_data_len);
        self.num_tokens = new_len;
        Ok(())
    }

    /// 获取指定位置的 K 数据
    ///
    /// # 参数
    /// - `token_idx`: token 索引（从 0 开始）
    ///
    /// # 返回
    /// 成功返回 K 数据切片，失败返回 `None`
    pub fn get_k(&self, token_idx: usize) -> Option<&[f32]> {
        if self.per_token_size == 0 || token_idx >= self.num_tokens {
            return None;
        }
        let start = token_idx.saturating_mul(self.per_token_size);
        let end = start.saturating_add(self.per_token_size);
        Some(&self.k_cache[start..end])
    }

    /// 获取指定位置的 V 数据
    ///
    /// # 参数
    /// - `token_idx`: token 索引（从 0 开始）
    ///
    /// # 返回
    /// 成功返回 V 数据切片，失败返回 `None`
    pub fn get_v(&self, token_idx: usize) -> Option<&[f32]> {
        if self.per_token_size == 0 || token_idx >= self.num_tokens {
            return None;
        }
        let start = token_idx.saturating_mul(self.per_token_size);
        let end = start.saturating_add(self.per_token_size);
        Some(&self.v_cache[start..end])
    }

    /// 批量获取 K 数据（指定范围）
    ///
    /// # 参数
    /// - `start_idx`: 起始 token 索引
    /// - `count`: token 数量
    ///
    /// # 返回
    /// 成功返回 K 数据切片，失败返回 `None`
    ///
    /// # 边界情况
    /// - 当 `count == 0` 且 `start_idx <= num_tokens` 时，返回空切片
    /// - 当 `per_token_size == 0` 时，返回 `None`
    pub fn get_k_range(&self, start_idx: usize, count: usize) -> Option<&[f32]> {
        if self.per_token_size == 0 {
            return None;
        }
        if count == 0 {
            if start_idx > self.num_tokens {
                return None;
            }
            return Some(&[]);
        }
        let end_idx = start_idx.saturating_add(count);
        if end_idx > self.num_tokens {
            return None;
        }
        let start = start_idx.saturating_mul(self.per_token_size);
        let end = end_idx.saturating_mul(self.per_token_size);
        Some(&self.k_cache[start..end])
    }

    /// 批量获取 V 数据（指定范围）
    ///
    /// # 参数
    /// - `start_idx`: 起始 token 索引
    /// - `count`: token 数量
    ///
    /// # 返回
    /// 成功返回 V 数据切片，失败返回 `None`
    ///
    /// # 边界情况
    /// - 当 `count == 0` 且 `start_idx <= num_tokens` 时，返回空切片
    /// - 当 `per_token_size == 0` 时，返回 `None`
    pub fn get_v_range(&self, start_idx: usize, count: usize) -> Option<&[f32]> {
        if self.per_token_size == 0 {
            return None;
        }
        if count == 0 {
            if start_idx > self.num_tokens {
                return None;
            }
            return Some(&[]);
        }
        let end_idx = start_idx.saturating_add(count);
        if end_idx > self.num_tokens {
            return None;
        }
        let start = start_idx.saturating_mul(self.per_token_size);
        let end = end_idx.saturating_mul(self.per_token_size);
        Some(&self.v_cache[start..end])
    }

    /// 获取整个 K 缓存
    ///
    /// # 返回
    /// K 缓存的原始数据切片（扁平存储，每个 token 占用 `per_token_size` 个连续元素）。
    ///
    /// # 注意
    /// 切片是扁平存储的，未按 token 分割。如需按 token 访问，请使用 `get_k`、 `get_k_range` 等方法。
    pub fn k_cache(&self) -> &[f32] {
        &self.k_cache
    }

    /// 获取整个 V 缓存
    ///
    /// # 返回
    /// V 缓存的原始数据切片（扁平存储，每个 token 占用 `per_token_size` 个连续元素）。
    ///
    /// # 注意
    /// 切片是扁平存储的，未按 token 分割。如需按 token 访问，请使用 `get_v`、 `get_v_range` 等方法。
    pub fn v_cache(&self) -> &[f32] {
        &self.v_cache
    }

    /// 获取 K 缓存容量
    pub fn k_cache_capacity(&self) -> usize {
        self.k_cache.capacity()
    }

    /// 获取 V 缓存容量
    pub fn v_cache_capacity(&self) -> usize {
        self.v_cache.capacity()
    }

    /// 收缩缓存容量以匹配实际大小
    ///
    /// 释放预分配但未使用的内存
    pub fn shrink_to_fit(&mut self) {
        self.k_cache.shrink_to_fit();
        self.v_cache.shrink_to_fit();
    }

    /// 迭代所有 token 的 KV 对
    ///
    /// # 返回
    /// 返回迭代器，每个元素是 `(K 切片, V 切片)`
    ///
    /// # 注意
    /// 如果 `per_token_size` 为 0（未初始化），返回空迭代器
    pub fn iter_kv(&self) -> impl Iterator<Item = (&[f32], &[f32])> + '_ {
        let per_token_size = self.per_token_size;
        let num_tokens = self.num_tokens;
        (0..num_tokens).map(move |i| {
            let start = i.saturating_mul(per_token_size);
            let end = start.saturating_add(per_token_size);
            (&self.k_cache[start..end], &self.v_cache[start..end])
        })
    }

    /// 预留额外容量
    ///
    /// # 参数
    /// - `additional_tokens`: 额外的 token 数量
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，失败返回 `AppendError`
    ///
    /// # 注意
    /// - 如果 `per_token_size` 为 0（未初始化），返回 `Err(AppendError::NotInitialized)`。
    /// - 如果设置了 `max_tokens`，预留容量会被限制在 `max_tokens` 剩余容量内，
    ///   不会因请求超出而失败（静默限制到剩余容量）。
    /// - 如果 `additional_tokens` 为 0，此方法立即返回 `Ok(())`。
    ///
    /// # Panic
    /// 内存分配失败时会 panic。如需优雅处理 OOM，请使用 `try_reserve_tokens`。
    pub fn reserve(&mut self, additional_tokens: usize) -> Result<(), AppendError> {
        if additional_tokens == 0 {
            return Ok(());
        }
        if self.per_token_size == 0 {
            return Err(AppendError::NotInitialized);
        }
        let mut additional = additional_tokens.saturating_mul(self.per_token_size);
        if self.max_tokens > 0 {
            let max_capacity = self.max_tokens.saturating_mul(self.per_token_size);
            let current_len = self.k_cache.len();
            let max_additional = max_capacity.saturating_sub(current_len);
            additional = additional.min(max_additional);
        }
        self.k_cache.reserve(additional);
        self.v_cache.reserve(additional);
        Ok(())
    }

    /// 尝试预留额外容量（优雅处理内存不足）
    ///
    /// # 参数
    /// - `additional_tokens`: 额外的 token 数量
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，失败返回 `TryAppendError`
    ///
    /// # 注意
    /// - 如果 `per_token_size` 为 0（未初始化），返回错误。
    /// - 如果设置了 `max_tokens`，预留容量将被限制为不超过 `max_tokens` 对应的容量。
    pub fn try_reserve_tokens(&mut self, additional_tokens: usize) -> Result<(), TryAppendError> {
        if additional_tokens == 0 {
            return Ok(());
        }
        if self.per_token_size == 0 {
            return Err(AppendError::NotInitialized.into());
        }
        let mut additional = additional_tokens.saturating_mul(self.per_token_size);
        if self.max_tokens > 0 {
            let max_capacity = self.max_tokens.saturating_mul(self.per_token_size);
            let current_len = self.k_cache.len();
            let max_additional = max_capacity.saturating_sub(current_len);
            additional = additional.min(max_additional);
        }
        self.k_cache.try_reserve(additional)?;
        self.v_cache.try_reserve(additional)?;
        Ok(())
    }

    /// 尝试精确预留额外容量（优雅处理内存不足）
    ///
    /// # 参数
    /// - `additional_tokens`: 额外的 token 数量
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，失败返回 `TryAppendError`
    ///
    /// # 注意
    /// - 如果 `per_token_size` 为 0（未初始化），返回错误。
    /// - 如果设置了 `max_tokens`，预留容量将被限制为不超过 `max_tokens` 对应的容量。
    /// - 与 `try_reserve_tokens` 不同，此方法使用 `try_reserve_exact`，
    ///   会精确分配所需容量，不会额外分配。
    pub fn try_reserve_exact_tokens(
        &mut self,
        additional_tokens: usize,
    ) -> Result<(), TryAppendError> {
        if additional_tokens == 0 {
            return Ok(());
        }
        if self.per_token_size == 0 {
            return Err(AppendError::NotInitialized.into());
        }
        let mut additional = additional_tokens.saturating_mul(self.per_token_size);
        if self.max_tokens > 0 {
            let max_capacity = self.max_tokens.saturating_mul(self.per_token_size);
            let current_len = self.k_cache.len();
            let max_additional = max_capacity.saturating_sub(current_len);
            additional = additional.min(max_additional);
        }
        self.k_cache.try_reserve_exact(additional)?;
        self.v_cache.try_reserve_exact(additional)?;
        Ok(())
    }
}

impl Extend<(Vec<f32>, Vec<f32>)> for StandardKVCache {
    /// 批量追加多个 token 的 KV 对
    ///
    /// # Panics
    /// 如果任何一对数据追加失败（如长度不匹配、容量超限），此方法会 panic。
    /// 如需优雅处理错误，请使用 `try_extend` 方法。
    fn extend<T: IntoIterator<Item = (Vec<f32>, Vec<f32>)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.append(&k, &v).unwrap_or_else(|e| {
                panic!(
                    "Extend failed: {}. Use try_extend() for graceful error handling.",
                    e
                )
            });
        }
    }
}

impl StandardKVCache {
    /// 尝试批量追加多个 token 的 KV 对
    ///
    /// # 参数
    /// - `iter`: KV 对的迭代器
    ///
    /// # 返回
    /// 成功返回 `Ok(())`，失败返回第一个错误
    ///
    /// # 注意
    /// 如果中途失败，已追加的数据不会回滚。
    pub fn try_extend<T: IntoIterator<Item = (Vec<f32>, Vec<f32>)>>(
        &mut self,
        iter: T,
    ) -> Result<(), AppendError> {
        for (k, v) in iter {
            self.append(&k, &v)?;
        }
        Ok(())
    }
}

impl KVCache for StandardKVCache {
    fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    fn clear_cache(&mut self) -> Result<(), KVCacheError> {
        self.k_cache.clear();
        self.v_cache.clear();
        self.num_tokens = 0;
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        (self.k_cache.capacity() + self.v_cache.capacity()) * std::mem::size_of::<f32>()
    }
}

impl InferenceContext {
    /// 创建新的推理上下文
    ///
    /// # Panics
    /// - 当 `config.use_mla` 长度不等于 `num_layers` 时会 panic
    /// - 当容量计算结果超过 `isize::MAX` 时会 panic
    ///
    /// # Example
    /// ```no_run
    /// use openmini_server::model::inference::context::{InferenceContext, ModelRunConfig};
    /// // let config = ModelRunConfig::from_model_config(&model_config);
    /// // let ctx = InferenceContext::new(model_config.num_hidden_layers, &config)?;
    /// ```
    pub fn new(num_layers: usize, config: &ModelRunConfig) -> Result<Self, ContextError> {
        let layer_caches = Self::build_layer_caches(num_layers, config)?;
        let memory = MemoryManager::new(MemoryConfig::default());

        Ok(Self {
            layer_caches,
            next_position: 0,
            max_seq_len: config.max_seq_len,
            memory,
        })
    }

    /// 创建带自定义记忆配置的推理上下文
    ///
    /// # 参数
    /// - `num_layers`: 层数
    /// - `config`: 模型运行配置
    /// - `memory_config`: 记忆管理器配置
    ///
    /// # 返回
    /// 成功返回 `Ok(Self)`，失败返回 `ContextError`
    pub fn with_memory_config(
        num_layers: usize,
        config: &ModelRunConfig,
        memory_config: MemoryConfig,
    ) -> Result<Self, ContextError> {
        let layer_caches = Self::build_layer_caches(num_layers, config)?;
        let memory = MemoryManager::new(memory_config);

        Ok(Self {
            layer_caches,
            next_position: 0,
            max_seq_len: config.max_seq_len,
            memory,
        })
    }

    /// 构建层缓存列表
    ///
    /// # 返回
    /// 成功返回 `Ok(Vec<LayerCache>)`，失败返回 `ContextError`
    ///
    /// # Errors
    /// - `ContextError::LayerCountMismatch`: `use_mla` 长度不等于 `num_layers`
    /// - `ContextError::CapacityOverflow`: 容量超过 `isize::MAX`
    fn build_layer_caches(
        num_layers: usize,
        config: &ModelRunConfig,
    ) -> Result<Vec<LayerCache>, ContextError> {
        if config.use_mla.len() != num_layers {
            return Err(ContextError::LayerCountMismatch {
                expected: num_layers,
                actual: config.use_mla.len(),
            });
        }

        let per_token_size = config.num_key_value_heads.saturating_mul(config.head_dim);
        let capacity = if config.preallocate {
            let cap = config
                .max_seq_len
                .checked_mul(per_token_size)
                .ok_or_else(|| ContextError::CapacityOverflow {
                    requested: config.max_seq_len.saturating_mul(per_token_size),
                    max: usize::MAX,
                })?;
            if cap > isize::MAX as usize {
                return Err(ContextError::CapacityOverflow {
                    requested: cap,
                    max: isize::MAX as usize,
                });
            }
            cap
        } else {
            0
        };

        Ok((0..num_layers)
            .map(|i| {
                if config.use_mla[i] {
                    let mla_config = MLAConfig {
                        hidden_size: config.hidden_size,
                        num_attention_heads: config.num_attention_heads,
                        num_key_value_heads: config.num_key_value_heads,
                        head_dim: config.head_dim,
                        latent_dim: config.mla_latent_dim,
                        use_decoupled_rope: config.mla_decoupled_rope,
                        rope_theta: config.rope_theta,
                        max_seq_len: config.max_seq_len,
                    };
                    LayerCache::MLA(MLALatentCache::new(mla_config))
                } else {
                    LayerCache::Standard(
                        StandardKVCache::with_token_capacity(capacity, per_token_size)
                            .expect("Failed to create StandardKVCache"),
                    )
                }
            })
            .collect())
    }

    /// 重置上下文状态
    ///
    /// 清除所有缓存和记忆状态。
    ///
    /// # 注意
    /// - `memory.clear(None)` 中的 `None` 表示清除所有记忆级别（瞬时、短期、长期）
    /// - MLA 缓存清除错误在 debug 模式下会打印到 stderr，release 模式下静默忽略
    /// - 如需自定义错误处理，请使用 `reset_with_error_handler`
    pub fn reset(&mut self) {
        self.reset_with_error_handler(|_layer_idx, _error| {
            #[cfg(debug_assertions)]
            eprintln!(
                "Failed to clear MLA cache at layer {}: {}",
                _layer_idx, _error
            );
        });
    }

    /// 重置上下文状态（带错误处理）
    ///
    /// 清除所有缓存和记忆状态。
    ///
    /// # 参数
    /// - `error_handler`: 错误处理回调，接收 (层索引, 错误信息)
    ///
    /// # 注意
    /// `memory.clear(None)` 中的 `None` 表示清除所有记忆级别
    pub fn reset_with_error_handler<F>(&mut self, mut error_handler: F)
    where
        F: FnMut(usize, &str),
    {
        self.next_position = 0;
        self.memory.clear(None);
        for (layer_idx, layer_cache) in self.layer_caches.iter_mut().enumerate() {
            match layer_cache {
                LayerCache::MLA(cache) => {
                    if let Err(e) = cache.clear() {
                        error_handler(layer_idx, &e.to_string());
                    }
                }
                LayerCache::Standard(cache) => {
                    cache.clear();
                }
            }
        }
    }

    /// 获取 MLA 缓存（不可变引用）
    ///
    /// # Example
    /// ```ignore
    /// if let Some(mla_cache) = context.get_mla_cache(layer_idx) {
    ///     // 使用 MLA 缓存进行注意力计算
    /// }
    /// ```
    pub fn get_mla_cache(&self, layer_idx: usize) -> Option<&MLALatentCache> {
        match self.layer_caches.get(layer_idx)? {
            LayerCache::MLA(cache) => Some(cache),
            LayerCache::Standard(_) => None,
        }
    }

    /// 获取 MLA 缓存（可变引用）
    pub fn get_mla_cache_mut(&mut self, layer_idx: usize) -> Option<&mut MLALatentCache> {
        match self.layer_caches.get_mut(layer_idx)? {
            LayerCache::MLA(cache) => Some(cache),
            LayerCache::Standard(_) => None,
        }
    }

    /// 获取下一个要处理的位置
    pub fn next_position(&self) -> usize {
        self.next_position
    }

    /// 推进位置
    pub fn advance_position(&mut self, steps: usize) {
        self.next_position += steps;
    }

    /// 获取最大序列长度
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// 获取层数
    pub fn num_layers(&self) -> usize {
        self.layer_caches.len()
    }

    /// 获取记忆管理器（不可变引用）
    pub fn memory(&self) -> &MemoryManager {
        &self.memory
    }

    /// 获取记忆管理器（可变引用）
    pub fn memory_mut(&mut self) -> &mut MemoryManager {
        &mut self.memory
    }

    /// 获取层缓存（不可变引用）
    pub fn layer_cache(&self, layer_idx: usize) -> Option<&LayerCache> {
        self.layer_caches.get(layer_idx)
    }

    /// 获取层缓存（可变引用）
    pub fn layer_cache_mut(&mut self, layer_idx: usize) -> Option<&mut LayerCache> {
        self.layer_caches.get_mut(layer_idx)
    }

    /// 迭代所有层缓存（不可变引用）
    pub fn iter_layer_caches(&self) -> impl Iterator<Item = &LayerCache> {
        self.layer_caches.iter()
    }

    /// 迭代所有层缓存（可变引用）
    pub fn iter_layer_caches_mut(&mut self) -> impl Iterator<Item = &mut LayerCache> {
        self.layer_caches.iter_mut()
    }
}

/// 推理上下文构建器
///
/// 用于灵活配置推理上下文的各个参数。
///
/// # Example
/// ```ignore
/// let ctx = InferenceContextBuilder::new(num_layers, &config)
///     .with_memory_config(memory_config)
///     .build();
/// ```
pub struct InferenceContextBuilder {
    num_layers: usize,
    config: ModelRunConfig,
    memory_config: Option<MemoryConfig>,
}

impl InferenceContextBuilder {
    /// 创建新的构建器
    ///
    /// # 参数
    /// - `num_layers`: 层数
    /// - `config`: 模型运行配置
    pub fn new(num_layers: usize, config: ModelRunConfig) -> Self {
        Self {
            num_layers,
            config,
            memory_config: None,
        }
    }

    /// 设置自定义记忆配置
    pub fn with_memory_config(mut self, memory_config: MemoryConfig) -> Self {
        self.memory_config = Some(memory_config);
        self
    }

    /// 构建推理上下文
    ///
    /// # 返回
    /// 成功返回 `Ok(InferenceContext)`，失败返回 `ContextError`
    pub fn build(self) -> Result<InferenceContext, ContextError> {
        let memory = if let Some(mc) = self.memory_config {
            MemoryManager::new(mc)
        } else {
            MemoryManager::new(MemoryConfig::default())
        };

        let layer_caches = InferenceContext::build_layer_caches(self.num_layers, &self.config)?;

        Ok(InferenceContext {
            layer_caches,
            next_position: 0,
            max_seq_len: self.config.max_seq_len,
            memory,
        })
    }
}

/// 模型运行配置
#[derive(Debug, Clone)]
pub struct ModelRunConfig {
    /// 隐藏层大小
    pub hidden_size: usize,
    /// 注意力头数
    pub num_attention_heads: usize,
    /// KV 头数
    pub num_key_value_heads: usize,
    /// 头维度
    pub head_dim: usize,
    /// 最大序列长度
    pub max_seq_len: usize,
    /// MLA 潜在维度
    pub mla_latent_dim: usize,
    /// MLA 解耦 RoPE
    pub mla_decoupled_rope: bool,
    /// RoPE theta 参数
    pub rope_theta: f32,
    /// 每层是否使用 MLA
    ///
    /// 每层可独立配置是否使用 MLA。`from_model_config` 会将模型配置的单一布尔值复制到所有层。
    /// 如需每层独立配置，可直接修改此字段。
    pub use_mla: Vec<bool>,
    /// 是否预分配最大容量
    ///
    /// - `true`: 预分配 `max_seq_len * per_token_size` 容量（默认行为，适合长序列场景）
    /// - `false`: 按需增长，初始容量为 0（适合短序列或内存受限场景）
    ///
    /// # 内存影响
    /// 当 `preallocate=true` 且 `max_seq_len` 较大时，可能消耗大量内存。
    /// 估算公式：`num_layers * max_seq_len * per_token_size * 2 (K+V) * 4 (f32字节)`
    /// 例如：32层、2048序列长度、1024 per_token_size ≈ 512MB
    pub preallocate: bool,
}

impl Default for ModelRunConfig {
    /// 返回默认配置
    ///
    /// # 注意
    /// - `use_mla` 默认为 `vec![false]`，长度为 1
    /// - 使用 `InferenceContext::new(num_layers, &config)` 时，`use_mla.len()` 必须等于 `num_layers`
    /// - 生产环境建议使用 `ModelRunConfig::from_model_config()` 创建配置
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            max_seq_len: 2048,
            mla_latent_dim: 512,
            mla_decoupled_rope: false,
            rope_theta: 10000.0,
            use_mla: vec![false],
            preallocate: true,
        }
    }
}

impl ModelRunConfig {
    /// 从模型配置创建运行配置
    ///
    /// # 注意
    /// - `use_mla` 从 `ModelConfig.use_mla` 复制到所有层
    /// - 默认启用预分配（`preallocate: true`）
    pub fn from_model_config(config: &super::model::ModelConfig) -> Self {
        let use_mla = vec![config.use_mla; config.num_hidden_layers];

        Self {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            max_seq_len: config.max_position_embeddings,
            mla_latent_dim: config.mla_latent_dim,
            mla_decoupled_rope: config.mla_decoupled_rope,
            rope_theta: config.rope_theta,
            use_mla,
            preallocate: true,
        }
    }

    /// 设置层数，自动调整 `use_mla` 长度
    ///
    /// # 参数
    /// - `num_layers`: 层数
    /// - `use_mla_value`: 每层是否使用 MLA（所有层使用相同值）
    ///
    /// # 返回
    /// 修改后的配置
    ///
    /// # 示例
    /// ```ignore
    /// let config = ModelRunConfig::default()
    ///     .with_num_layers(32, false);
    /// ```
    pub fn with_num_layers(mut self, num_layers: usize, use_mla_value: bool) -> Self {
        self.use_mla = vec![use_mla_value; num_layers];
        self
    }

    /// 设置预分配开关
    ///
    /// # 参数
    /// - `preallocate`: 是否预分配最大容量
    ///
    /// # 返回
    /// 修改后的配置
    ///
    /// # 示例
    /// ```ignore
    /// let config = ModelRunConfig::default()
    ///     .with_num_layers(32, false)
    ///     .with_preallocate(false);
    /// ```
    pub fn with_preallocate(mut self, preallocate: bool) -> Self {
        self.preallocate = preallocate;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cache(per_token_size: usize) -> StandardKVCache {
        StandardKVCache::with_token_capacity(100, per_token_size).unwrap()
    }

    fn create_test_cache_with_max(max_tokens: usize, per_token_size: usize) -> StandardKVCache {
        StandardKVCache::with_max_tokens(max_tokens, per_token_size)
    }

    #[test]
    fn test_append_basic() {
        let mut cache = create_test_cache(4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        assert!(cache.append(&k, &v).is_ok());
        assert_eq!(cache.num_tokens(), 1);
        assert_eq!(cache.get_k(0), Some(&k[..]));
        assert_eq!(cache.get_v(0), Some(&v[..]));
    }

    #[test]
    fn test_append_not_initialized() {
        let mut cache = create_test_cache(0);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        let result = cache.append(&k, &v);
        assert_eq!(result, Err(AppendError::NotInitialized));
    }

    #[test]
    fn test_append_length_mismatch() {
        let mut cache = create_test_cache(4);
        let k = vec![1.0, 2.0, 3.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        let result = cache.append(&k, &v);
        assert_eq!(
            result,
            Err(AppendError::KLengthMismatch {
                expected: 4,
                actual: 3
            })
        );

        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0];

        let result = cache.append(&k, &v);
        assert_eq!(
            result,
            Err(AppendError::VLengthMismatch {
                expected: 4,
                actual: 3
            })
        );
    }

    #[test]
    fn test_append_capacity_exceeded() {
        let mut cache = create_test_cache_with_max(2, 4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        assert!(cache.append(&k, &v).is_ok());
        assert!(cache.append(&k, &v).is_ok());

        let result = cache.append(&k, &v);
        assert_eq!(
            result,
            Err(AppendError::CapacityExceeded { current: 2, max: 2 })
        );
    }

    #[test]
    fn test_append_with_grow_capacity_exceeded() {
        let mut cache = create_test_cache_with_max(2, 4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        assert!(cache.append_with_grow(&k, &v).is_ok());
        assert!(cache.append_with_grow(&k, &v).is_ok());

        let result = cache.append_with_grow(&k, &v);
        assert_eq!(
            result,
            Err(AppendError::CapacityExceeded { current: 2, max: 2 })
        );
    }

    #[test]
    fn test_append_with_grow_expansion() {
        let mut cache = StandardKVCache::with_token_capacity(4, 4).unwrap();

        for i in 0..10 {
            let k = vec![i as f32; 4];
            let v = vec![(i + 10) as f32; 4];
            assert!(cache.append_with_grow(&k, &v).is_ok());
        }

        assert_eq!(cache.num_tokens(), 10);
    }

    #[test]
    fn test_get_k_v_out_of_bounds() {
        let mut cache = create_test_cache(4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&k, &v).unwrap();

        assert!(cache.get_k(1).is_none());
        assert!(cache.get_v(1).is_none());
    }

    #[test]
    fn test_get_k_range_count_zero() {
        let mut cache = create_test_cache(4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&k, &v).unwrap();

        assert_eq!(cache.get_k_range(0, 0), Some(&[][..]));
        assert_eq!(cache.get_k_range(1, 0), Some(&[][..]));
        assert!(cache.get_k_range(2, 0).is_none());
    }

    #[test]
    fn test_get_v_range_count_zero() {
        let mut cache = create_test_cache(4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&k, &v).unwrap();

        assert_eq!(cache.get_v_range(0, 0), Some(&[][..]));
        assert_eq!(cache.get_v_range(1, 0), Some(&[][..]));
        assert!(cache.get_v_range(2, 0).is_none());
    }

    #[test]
    fn test_get_k_range_per_token_size_zero() {
        let cache = create_test_cache(0);
        assert!(cache.get_k_range(0, 0).is_none());
        assert!(cache.get_k_range(0, 1).is_none());
    }

    #[test]
    fn test_get_v_range_per_token_size_zero() {
        let cache = create_test_cache(0);
        assert!(cache.get_v_range(0, 0).is_none());
        assert!(cache.get_v_range(0, 1).is_none());
    }

    #[test]
    fn test_reserve_per_token_size_zero() {
        let mut cache = create_test_cache(0);
        let result = cache.reserve(100);
        assert_eq!(result, Err(AppendError::NotInitialized));
    }

    #[test]
    fn test_reserve_normal() {
        let mut cache = create_test_cache(4);
        cache.reserve(10).unwrap();
        assert!(cache.k_cache().is_empty());
    }

    #[test]
    fn test_clear() {
        let mut cache = create_test_cache(4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&k, &v).unwrap();

        cache.clear();
        assert_eq!(cache.num_tokens(), 0);
        assert!(cache.get_k(0).is_none());
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut cache = StandardKVCache::with_token_capacity(100, 4).unwrap();
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&k, &v).unwrap();

        cache.shrink_to_fit();
        assert_eq!(cache.k_cache().len(), 4);
    }

    #[test]
    fn test_max_tokens_zero_unlimited() {
        let mut cache = create_test_cache(4);
        assert_eq!(cache.max_tokens(), 0);

        for i in 0..100 {
            let k = vec![i as f32; 4];
            let v = vec![(i + 10) as f32; 4];
            assert!(cache.append(&k, &v).is_ok());
        }
        assert_eq!(cache.num_tokens(), 100);
    }

    #[test]
    fn test_append_error_display() {
        assert_eq!(
            AppendError::NotInitialized.to_string(),
            "per_token_size not initialized"
        );
        assert_eq!(
            AppendError::KLengthMismatch {
                expected: 4,
                actual: 3
            }
            .to_string(),
            "K length mismatch: expected 4, got 3"
        );
        assert_eq!(
            AppendError::VLengthMismatch {
                expected: 4,
                actual: 3
            }
            .to_string(),
            "V length mismatch: expected 4, got 3"
        );
        assert_eq!(
            AppendError::CapacityExceeded {
                current: 10,
                max: 10
            }
            .to_string(),
            "Capacity exceeded: 10/10"
        );
    }

    #[test]
    fn test_batch_read() {
        let mut cache = create_test_cache(4);
        for i in 0..5 {
            let k = vec![i as f32; 4];
            let v = vec![(i + 10) as f32; 4];
            cache.append(&k, &v).unwrap();
        }

        let k_range = cache.get_k_range(1, 3);
        assert!(k_range.is_some());
        let k_range = k_range.unwrap();
        assert_eq!(k_range.len(), 12);

        let v_range = cache.get_v_range(1, 3);
        assert!(v_range.is_some());
        let v_range = v_range.unwrap();
        assert_eq!(v_range.len(), 12);
    }

    #[test]
    fn test_append_with_grow_capacity_increase() {
        let mut cache = StandardKVCache::with_token_capacity(4, 4).unwrap();
        let initial_capacity = 4;

        for i in 0..10 {
            let k = vec![i as f32; 4];
            let v = vec![(i + 10) as f32; 4];
            cache.append_with_grow(&k, &v).unwrap();
        }

        assert!(cache.k_cache_capacity() > initial_capacity);
        assert_eq!(cache.num_tokens(), 10);
    }

    #[test]
    fn test_append_with_grow_respects_max_tokens_capacity() {
        let max_tokens = 5;
        let mut cache = StandardKVCache::with_max_tokens(max_tokens, 4);

        for i in 0..max_tokens {
            let k = vec![i as f32; 4];
            let v = vec![(i + 10) as f32; 4];
            cache.append_with_grow(&k, &v).unwrap();
        }

        assert_eq!(cache.num_tokens(), max_tokens);
        assert!(cache.k_cache().len() <= max_tokens * 4);
    }

    #[test]
    fn test_is_empty() {
        let mut cache = create_test_cache(4);
        assert!(cache.is_empty());

        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&k, &v).unwrap();
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_reserve_respects_max_tokens() {
        let max_tokens = 5;
        let mut cache = StandardKVCache::with_max_tokens(max_tokens, 4);

        let _ = cache.reserve(100);
        assert!(cache.k_cache_capacity() <= max_tokens * 4);
    }

    #[test]
    fn test_default() {
        let cache = StandardKVCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.num_tokens(), 0);
        assert_eq!(cache.per_token_size(), 0);
        assert_eq!(cache.max_tokens(), 0);
    }

    #[test]
    fn test_truncate() {
        let mut cache = create_test_cache(4);
        for i in 0..10 {
            let k = vec![i as f32; 4];
            let v = vec![(i + 10) as f32; 4];
            cache.append(&k, &v).unwrap();
        }
        assert_eq!(cache.num_tokens(), 10);

        let _ = cache.truncate(5);
        assert_eq!(cache.num_tokens(), 5);
        assert_eq!(cache.k_cache().len(), 20);

        let _ = cache.truncate(10);
        assert_eq!(cache.num_tokens(), 5);

        let _ = cache.truncate(0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_try_append() {
        let mut cache = create_test_cache(4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        assert!(cache.try_append(&k, &v).is_ok());
        assert_eq!(cache.num_tokens(), 1);
    }

    #[test]
    fn test_try_append_with_grow() {
        let mut cache = StandardKVCache::with_token_capacity(4, 4).unwrap();

        for i in 0..10 {
            let k = vec![i as f32; 4];
            let v = vec![(i + 10) as f32; 4];
            assert!(cache.try_append_with_grow(&k, &v).is_ok());
        }
        assert_eq!(cache.num_tokens(), 10);
    }

    #[test]
    fn test_extend() {
        let mut cache = create_test_cache(4);
        let items: Vec<(Vec<f32>, Vec<f32>)> = (0..5)
            .map(|i| (vec![i as f32; 4], vec![(i + 10) as f32; 4]))
            .collect();

        cache.extend(items);
        assert_eq!(cache.num_tokens(), 5);
    }

    #[test]
    fn test_layer_cache_helpers() {
        let standard_cache = StandardKVCache::with_token_capacity(100, 4).unwrap();
        let layer = LayerCache::Standard(standard_cache);

        assert!(layer.is_standard());
        assert!(!layer.is_mla());
        assert!(layer.as_standard().is_some());
        assert!(layer.as_mla().is_none());
    }

    #[test]
    fn test_try_extend() {
        let mut cache = create_test_cache(4);
        let items: Vec<(Vec<f32>, Vec<f32>)> = (0..5)
            .map(|i| (vec![i as f32; 4], vec![(i + 10) as f32; 4]))
            .collect();

        assert!(cache.try_extend(items).is_ok());
        assert_eq!(cache.num_tokens(), 5);
    }

    #[test]
    fn test_try_extend_with_error() {
        let mut cache = create_test_cache(4);
        let items: Vec<(Vec<f32>, Vec<f32>)> = vec![
            (vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]),
            (vec![1.0, 2.0, 3.0], vec![5.0, 6.0, 7.0, 8.0]), // 错误：长度不匹配
        ];

        let result = cache.try_extend(items);
        assert!(result.is_err());
        assert_eq!(cache.num_tokens(), 1); // 第一个成功追加
    }

    #[test]
    fn test_truncate_uninitialized() {
        let mut cache = StandardKVCache::default();
        let _ = cache.truncate(5);
        assert_eq!(cache.num_tokens(), 0);
    }

    #[test]
    fn test_init() {
        let mut cache = StandardKVCache::default();
        assert!(!cache.is_initialized());

        let _ = cache.init(4);
        assert!(cache.is_initialized());
        assert_eq!(cache.per_token_size(), 4);
    }

    #[test]
    fn test_try_reserve_tokens() {
        let mut cache = create_test_cache(4);

        assert!(cache.try_reserve_tokens(10).is_ok());
        assert!(cache.k_cache_capacity() >= 40);
    }

    #[test]
    fn test_try_reserve_tokens_uninitialized() {
        let mut cache = StandardKVCache::default();

        let result = cache.try_reserve_tokens(10);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_reserve_exact_tokens() {
        let mut cache = StandardKVCache::with_token_capacity(0, 4).unwrap();

        assert!(cache.try_reserve_exact_tokens(10).is_ok());
        assert_eq!(cache.k_cache_capacity(), 40);
    }

    #[test]
    fn test_iter_kv() {
        let mut cache = create_test_cache(4);
        for i in 0..5 {
            let k = vec![i as f32; 4];
            let v = vec![(i + 10) as f32; 4];
            cache.append(&k, &v).unwrap();
        }

        let kv_pairs: Vec<_> = cache.iter_kv().collect();
        assert_eq!(kv_pairs.len(), 5);
    }

    #[test]
    fn test_kv_cache_trait() {
        let mut cache = create_test_cache(4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&k, &v).unwrap();

        assert_eq!(KVCache::num_tokens(&cache), 1);
        assert!(!KVCache::is_empty(&cache));
        assert!(KVCache::memory_usage(&cache) > 0);

        KVCache::clear_cache(&mut cache).unwrap();
        assert!(KVCache::is_empty(&cache));
    }

    #[test]
    fn test_model_run_config_default() {
        let config = ModelRunConfig::default();
        assert!(config.preallocate);
        assert_eq!(config.use_mla.len(), 1);
    }

    #[test]
    fn test_preallocate_false() {
        let config = ModelRunConfig {
            preallocate: false,
            ..ModelRunConfig::default()
        };
        assert!(!config.preallocate);
    }

    #[test]
    fn test_with_num_layers() {
        let config = ModelRunConfig::default().with_num_layers(32, true);

        assert_eq!(config.use_mla.len(), 32);
        assert!(config.use_mla.iter().all(|&v| v));
    }

    #[test]
    fn test_with_preallocate_builder() {
        let config = ModelRunConfig::default()
            .with_num_layers(16, false)
            .with_preallocate(false);

        assert_eq!(config.use_mla.len(), 16);
        assert!(!config.preallocate);
    }

    #[test]
    fn test_reinit() {
        let mut cache = create_test_cache(4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(&k, &v).unwrap();

        assert_eq!(cache.num_tokens(), 1);

        // reinit 清除数据并设置新的 per_token_size
        cache.reinit(8);
        assert_eq!(cache.num_tokens(), 0);
        assert_eq!(cache.per_token_size, 8);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_inference_context_new() {
        let config = ModelRunConfig::default().with_num_layers(2, false);

        let ctx = InferenceContext::new(2, &config);
        assert!(ctx.is_ok());

        let ctx = ctx.unwrap();
        assert_eq!(ctx.layer_caches.len(), 2);
        assert_eq!(ctx.max_seq_len, config.max_seq_len);
    }

    #[test]
    fn test_inference_context_layer_count_mismatch() {
        let config = ModelRunConfig::default(); // use_mla.len() == 1

        let result = InferenceContext::new(3, &config);
        assert!(result.is_err());

        match result {
            Err(ContextError::LayerCountMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 1);
            }
            _ => panic!("Expected LayerCountMismatch error"),
        }
    }

    #[test]
    fn test_inference_context_builder() {
        let config = ModelRunConfig::default()
            .with_num_layers(4, false)
            .with_preallocate(false);

        let ctx = InferenceContextBuilder::new(4, config).build();

        assert!(ctx.is_ok());
        let ctx = ctx.unwrap();
        assert_eq!(ctx.layer_caches.len(), 4);
    }

    #[test]
    fn test_try_append_error_display() {
        let err = TryAppendError::UserError(AppendError::NotInitialized);
        let msg = format!("{}", err);
        assert!(msg.contains("not initialized"));
    }

    #[test]
    fn test_with_token_capacity_success() {
        let cache = StandardKVCache::with_token_capacity(1024, 128);
        assert!(cache.is_ok());
        let cache = cache.unwrap();
        assert_eq!(cache.per_token_size, 128);
        assert_eq!(cache.num_tokens(), 0);
    }

    #[test]
    fn test_with_token_capacity_overflow() {
        let result = StandardKVCache::with_token_capacity(usize::MAX, usize::MAX);
        assert!(result.is_err());
        match result {
            Err(ContextError::CapacityOverflow { .. }) => {}
            _ => panic!("Expected CapacityOverflow error"),
        }
    }

    #[test]
    fn test_inference_context_mixed_layers() {
        let config = ModelRunConfig {
            use_mla: vec![false, true, false],
            ..ModelRunConfig::default()
        };

        let ctx = InferenceContext::new(3, &config);
        assert!(ctx.is_ok());

        let ctx = ctx.unwrap();
        assert_eq!(ctx.layer_caches.len(), 3);

        assert!(ctx.layer_caches[0].is_standard());
        assert!(ctx.layer_caches[1].is_mla());
        assert!(ctx.layer_caches[2].is_standard());
    }

    #[test]
    fn test_inference_context_get_layer_cache() {
        let config = ModelRunConfig::default().with_num_layers(2, false);

        let ctx = InferenceContext::new(2, &config).unwrap();

        let layer0 = ctx.layer_cache(0);
        assert!(layer0.is_some());
        assert!(layer0.unwrap().is_standard());

        let layer_invalid = ctx.layer_cache(5);
        assert!(layer_invalid.is_none());
    }

    #[test]
    fn test_inference_context_advance_position() {
        let config = ModelRunConfig::default().with_num_layers(1, false);

        let mut ctx = InferenceContext::new(1, &config).unwrap();

        assert_eq!(ctx.next_position(), 0);
        ctx.advance_position(10);
        assert_eq!(ctx.next_position(), 10);
    }
}
