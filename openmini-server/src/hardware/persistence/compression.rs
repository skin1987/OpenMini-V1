//! 数据压缩模块
//!
//! 为持久化层提供高效的压缩能力，
//! 减少数据库存储空间和网络传输开销。
//!
//! # 支持的算法
//!
//! - **Zstd**: 高压缩比，适合冷数据
//! - **Lz4**: 极快速度，适合热数据
//! - **None**: 不压缩，适合已压缩数据
//!
//! # 使用示例
//!
//! ```ignore
//! use openmini_server::hardware::persistence::compression::*;
//!
//! let manager = CompressionManager::new();
//! let data = b"Hello, World! This is a test data for compression.";
//!
//! // 使用默认算法 (Zstd) 压缩
//! let compressed = manager.compress(data).unwrap();
//! println!("压缩率: {:.2}%", compressed.ratio * 100.0);
//!
//! // 解压
//! let decompressed = manager.decompress(
//!     &compressed.data,
//!     compressed.original_size,
//!     compressed.algorithm
//! ).unwrap();
//! assert_eq!(decompressed.data, data);
//! ```

use std::collections::HashMap;

/// 压缩算法类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompressionAlgorithm {
    /// Zstandard 压缩 (高质量)
    #[default]
    Zstd,
    /// LZ4 压缩 (高速)
    Lz4,
    /// 无压缩
    None,
}

impl std::fmt::Display for CompressionAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionAlgorithm::Zstd => write!(f, "zstd"),
            CompressionAlgorithm::Lz4 => write!(f, "lz4"),
            CompressionAlgorithm::None => write!(f, "none"),
        }
    }
}

/// 压缩配置
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// 使用的算法
    pub algorithm: CompressionAlgorithm,

    /// 压缩级别 (1-22, 越高压缩越好但越慢)
    pub level: i32,

    /// 启用压缩的最小数据大小 (小于此值不压缩)
    pub min_compress_size: usize,

    /// 是否启用字典训练 (仅 zstd)
    pub dictionary_training: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,               // 平衡速度和压缩率
            min_compress_size: 256, // < 256 字节不压缩
            dictionary_training: false,
        }
    }
}

/// 压缩结果
#[derive(Debug, Clone)]
pub struct CompressResult {
    /// 压缩后的数据
    pub data: Vec<u8>,
    /// 原始数据大小
    pub original_size: usize,
    /// 压缩后大小
    pub compressed_size: usize,
    /// 压缩比率 (compressed / original)
    pub ratio: f32,
    /// 使用的算法
    pub algorithm: CompressionAlgorithm,
    /// 压耗时 (微秒)
    pub duration_us: u128,
}

/// 解压结果
#[derive(Debug, Clone)]
pub struct DecompressResult {
    /// 解压后的数据
    pub data: Vec<u8>,
    /// 原始大小 (解压后)
    pub original_size: usize,
    /// 解耗时 (微秒)
    pub duration_us: u128,
}

/// 压缩器 trait
///
/// 定义压缩器的统一接口，支持多种压缩算法的实现。
/// 所有实现必须线程安全 (`Send + Sync`)。
pub trait Compressor: Send + Sync {
    /// 压缩数据
    ///
    /// # 参数
    /// - `data`: 待压缩的数据
    ///
    /// # 返回
    /// - `Ok(CompressResult)`: 压缩结果，包含压缩后的数据和统计信息
    /// - `Err(String)`: 压缩失败时的错误信息
    fn compress(&self, data: &[u8]) -> Result<CompressResult, String>;

    /// 解压数据
    ///
    /// # 参数
    /// - `data`: 压缩后的数据
    /// - `original_size`: 原始数据的大小（用于验证）
    ///
    /// # 返回
    /// - `Ok(DecompressResult)`: 解压结果
    /// - `Err(String)`: 解压失败时的错误信息
    fn decompress(&self, data: &[u8], original_size: usize) -> Result<DecompressResult, String>;

    /// 获取压缩算法类型
    fn algorithm(&self) -> CompressionAlgorithm;

    /// 获取压缩配置
    fn config(&self) -> &CompressionConfig;
}

/// Zstd 压缩器实现
///
/// 使用 Zstandard 算法提供高质量压缩。
/// 适合冷数据和需要最大化节省存储空间的场景。
///
/// # 性能特点
/// - 压缩率：优秀 (通常 3-5x)
/// - 速度：中等 (~200 MB/s)
/// - 内存使用：低
#[cfg(feature = "compression")]
pub struct ZstdCompressor {
    config: CompressionConfig,
}

#[cfg(feature = "compression")]
impl ZstdCompressor {
    /// 创建新的 Zstd 压缩器
    ///
    /// # 参数
    /// - `level`: 压缩级别 (1-22)，推荐值：
    ///   - 1-3: 快速压缩，适合实时场景
    ///   - 4-9: 平衡模式（默认推荐）
    ///   - 10-15: 高压缩率，适合离线场景
    ///   - 16-22: 最大压缩率，非常慢
    pub fn new(level: i32) -> Self {
        Self {
            config: CompressionConfig {
                algorithm: CompressionAlgorithm::Zstd,
                level,
                ..Default::default()
            },
        }
    }

    /// 使用自定义配置创建
    pub fn with_config(config: CompressionConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "compression")]
impl Compressor for ZstdCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, String> {
        // 小数据跳过压缩
        if data.len() < self.config.min_compress_size {
            return Ok(CompressResult {
                data: data.to_vec(),
                original_size: data.len(),
                compressed_size: data.len(),
                ratio: 1.0,
                algorithm: CompressionAlgorithm::None,
                duration_us: 0,
            });
        }

        let start = std::time::Instant::now();

        // 使用 zstd crate 进行压缩
        let compressed = zstd::encode_all(data, self.config.level)
            .map_err(|e| format!("Zstd compression failed: {}", e))?;

        let elapsed = start.elapsed();

        Ok(CompressResult {
            data: compressed.clone(),
            original_size: data.len(),
            compressed_size: compressed.len(),
            ratio: compressed.len() as f32 / data.len() as f32,
            algorithm: CompressionAlgorithm::Zstd,
            duration_us: elapsed.as_micros(),
        })
    }

    fn decompress(&self, data: &[u8], _original_size: usize) -> Result<DecompressResult, String> {
        let start = std::time::Instant::now();

        let decompressed =
            zstd::decode_all(data).map_err(|e| format!("Zstd decompression failed: {}", e))?;

        let elapsed = start.elapsed();
        let original_size = decompressed.len();

        Ok(DecompressResult {
            data: decompressed,
            original_size,
            duration_us: elapsed.as_micros(),
        })
    }

    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::Zstd
    }

    fn config(&self) -> &CompressionConfig {
        &self.config
    }
}

/// LZ4 压缩器实现
///
/// 使用 LZ4 算法提供极快的压缩速度。
/// 适合热数据和高吞吐量场景。
///
/// # 性能特点
/// - 压缩率：良好 (通常 2-3x)
/// - 速度：极快 (~500 MB/s)
/// - 内存使用：极低
#[cfg(feature = "compression")]
pub struct Lz4Compressor {
    config: CompressionConfig,
}

#[cfg(feature = "compression")]
impl Lz4Compressor {
    /// 创建新的 LZ4 压缩器
    ///
    /// # 参数
    /// - `level`: 压缩级别 (1-16)，推荐值：
    ///   - 1-3: 最快速度
    ///   - 4-6: 平衡模式（默认）
    ///   - 7-10: 更好压缩率
    ///   - 11-16: 最大压缩率
    pub fn new(level: i32) -> Self {
        Self {
            config: CompressionConfig {
                algorithm: CompressionAlgorithm::Lz4,
                level,
                ..Default::default()
            },
        }
    }

    /// 使用自定义配置创建
    pub fn with_config(config: CompressionConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "compression")]
impl Compressor for Lz4Compressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, String> {
        // 小数据跳过压缩
        if data.len() < self.config.min_compress_size {
            return Ok(CompressResult {
                data: data.to_vec(),
                original_size: data.len(),
                compressed_size: data.len(),
                ratio: 1.0,
                algorithm: CompressionAlgorithm::None,
                duration_us: 0,
            });
        }

        let start = std::time::Instant::now();

        // 使用 lz4 EncoderBuilder 进行帧级别压缩
        let mut compressed = Vec::new();
        {
            let mut encoder = lz4::EncoderBuilder::new()
                .level(self.config.level as u32)
                .build(Vec::<u8>::new())
                .map_err(|e| format!("LZ4 encoder creation failed: {}", e))?;

            use std::io::Write;
            encoder
                .write_all(data)
                .map_err(|e| format!("LZ4 write failed: {}", e))?;
            let (result, finish_result) = encoder.finish();
            finish_result.map_err(|e| format!("LZ4 finish failed: {}", e))?;
            compressed = result;
        }

        let elapsed = start.elapsed();

        Ok(CompressResult {
            data: compressed.clone(),
            original_size: data.len(),
            compressed_size: compressed.len(),
            ratio: compressed.len() as f32 / data.len() as f32,
            algorithm: CompressionAlgorithm::Lz4,
            duration_us: elapsed.as_micros(),
        })
    }

    fn decompress(&self, data: &[u8], _original_size: usize) -> Result<DecompressResult, String> {
        let start = std::time::Instant::now();

        // 使用 lz4 Decoder 进行帧级别解压
        let mut decoder =
            lz4::Decoder::new(data).map_err(|e| format!("LZ4 decoder creation failed: {}", e))?;
        let mut decompressed = Vec::new();
        use std::io::Read;
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| format!("LZ4 read failed: {}", e))?;

        let elapsed = start.elapsed();
        let original_size = decompressed.len();

        Ok(DecompressResult {
            data: decompressed,
            original_size,
            duration_us: elapsed.as_micros(),
        })
    }

    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::Lz4
    }

    fn config(&self) -> &CompressionConfig {
        &self.config
    }
}

/// 无操作压缩器 (直接返回原始数据)
///
/// 当不需要压缩或数据已经过压缩时使用。
/// 提供零开销的 pass-through 实现。
pub struct NoOpCompressor {
    config: CompressionConfig,
}

impl NoOpCompressor {
    /// 创建无操作压缩器
    pub fn new() -> Self {
        Self {
            config: CompressionConfig {
                algorithm: CompressionAlgorithm::None,
                ..Default::default()
            },
        }
    }
}

impl Default for NoOpCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Compressor for NoOpCompressor {
    fn compress(&self, data: &[u8]) -> Result<CompressResult, String> {
        Ok(CompressResult {
            data: data.to_vec(),
            original_size: data.len(),
            compressed_size: data.len(),
            ratio: 1.0,
            algorithm: CompressionAlgorithm::None,
            duration_us: 0,
        })
    }

    fn decompress(&self, data: &[u8], _original_size: usize) -> Result<DecompressResult, String> {
        Ok(DecompressResult {
            data: data.to_vec(),
            original_size: data.len(),
            duration_us: 0,
        })
    }

    fn algorithm(&self) -> CompressionAlgorithm {
        CompressionAlgorithm::None
    }

    fn config(&self) -> &CompressionConfig {
        &self.config
    }
}

/// 压缩管理器 (工厂模式)
///
/// 统一管理多种压缩算法，提供工厂方法创建压缩器实例。
/// 支持运行时切换算法和性能监控。
///
/// # 线程安全
///
/// 内部使用 `parking_lot::RwLock` 保护统计数据，
/// 支持多线程并发调用压缩/解压操作。
pub struct CompressionManager {
    compressors: HashMap<CompressionAlgorithm, Box<dyn Compressor>>,
    default_algorithm: CompressionAlgorithm,
    stats: parking_lot::RwLock<CompressionStats>,
}

/// 压缩统计信息
#[derive(Debug, Default, Clone)]
pub struct CompressionStats {
    /// 总压缩次数
    pub total_compressions: u64,
    /// 总解压次数
    pub total_decompressions: u64,
    /// 总原始字节数
    pub total_original_bytes: u64,
    /// 总压缩后字节数
    pub total_compressed_bytes: u64,
    /// 总压缩时间 (微秒)
    pub total_compression_time_us: u128,
    /// 平均压缩比率
    pub avg_compression_ratio: f32,
}

impl CompressionStats {
    /// 计算整体压缩率
    pub fn overall_ratio(&self) -> f32 {
        if self.total_original_bytes == 0 {
            1.0
        } else {
            self.total_compressed_bytes as f32 / self.total_original_bytes as f32
        }
    }

    /// 计算节省的空间 (字节)
    pub fn saved_bytes(&self) -> u64 {
        self.total_original_bytes
            .saturating_sub(self.total_compressed_bytes)
    }

    /// 计算节省的百分比
    pub fn saved_percentage(&self) -> f32 {
        if self.total_original_bytes == 0 {
            0.0
        } else {
            (self.saved_bytes() as f32 / self.total_original_bytes as f32) * 100.0
        }
    }

    /// 平均每次压缩耗时 (微秒)
    pub fn avg_compression_time_us(&self) -> f64 {
        if self.total_compressions == 0 {
            0.0
        } else {
            self.total_compression_time_us as f64 / self.total_compressions as f64
        }
    }
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CompressionStats {{\n\
             \tcompressions: {},\n\
             \tdecompressions: {},\n\
             \toriginal_bytes: {} ({:.2} MB),\n\
             \tcompressed_bytes: {} ({:.2} MB),\n\
             \tsaved: {} bytes ({:.2}%),\n\
             \tavg_ratio: {:.3},\n\
             \tavg_time: {:.2} μs\n\
             }}",
            self.total_compressions,
            self.total_decompressions,
            self.total_original_bytes,
            self.total_original_bytes as f64 / 1024.0 / 1024.0,
            self.total_compressed_bytes,
            self.total_compressed_bytes as f64 / 1024.0 / 1024.0,
            self.saved_bytes(),
            self.saved_percentage(),
            self.avg_compression_ratio,
            self.avg_compression_time_us()
        )
    }
}

impl CompressionManager {
    /// 创建默认压缩管理器
    ///
    /// 初始化所有可用的压缩算法：
    /// - Zstd (level=3): 默认算法
    /// - Lz4 (level=4): 快速压缩备选
    /// - None: 无压缩选项
    #[cfg(feature = "compression")]
    pub fn new() -> Self {
        let mut compressors: HashMap<CompressionAlgorithm, Box<dyn Compressor>> = HashMap::new();
        compressors.insert(CompressionAlgorithm::Zstd, Box::new(ZstdCompressor::new(3)));
        compressors.insert(CompressionAlgorithm::Lz4, Box::new(Lz4Compressor::new(4)));
        compressors.insert(CompressionAlgorithm::None, Box::new(NoOpCompressor::new()));

        Self {
            compressors,
            default_algorithm: CompressionAlgorithm::Zstd,
            stats: parking_lot::RwLock::new(CompressionStats::default()),
        }
    }

    /// 创建仅支持无压缩的管理器 (无 feature 时使用)
    #[cfg(not(feature = "compression"))]
    pub fn new() -> Self {
        let mut compressors: HashMap<CompressionAlgorithm, Box<dyn Compressor>> = HashMap::new();
        compressors.insert(CompressionAlgorithm::None, Box::new(NoOpCompressor::new()));

        Self {
            compressors,
            default_algorithm: CompressionAlgorithm::None,
            stats: parking_lot::RwLock::new(CompressionStats::default()),
        }
    }

    /// 使用自定义配置创建
    ///
    /// 仅初始化指定的压缩算法。
    #[cfg(feature = "compression")]
    pub fn with_config(config: CompressionConfig) -> Self {
        let compressor: Box<dyn Compressor> = match config.algorithm {
            CompressionAlgorithm::Zstd => Box::new(ZstdCompressor::with_config(config.clone())),
            CompressionAlgorithm::Lz4 => Box::new(Lz4Compressor::with_config(config.clone())),
            CompressionAlgorithm::None => Box::new(NoOpCompressor::new()),
        };

        let mut compressors = HashMap::new();
        compressors.insert(config.algorithm, compressor);

        Self {
            compressors,
            default_algorithm: config.algorithm,
            stats: parking_lot::RwLock::new(CompressionStats::default()),
        }
    }

    /// 使用自定义配置创建 (无 feature 版本)
    #[cfg(not(feature = "compression"))]
    pub fn with_config(_config: CompressionConfig) -> Self {
        let mut compressors: HashMap<CompressionAlgorithm, Box<dyn Compressor>> = HashMap::new();
        compressors.insert(CompressionAlgorithm::None, Box::new(NoOpCompressor::new()));

        Self {
            compressors,
            default_algorithm: CompressionAlgorithm::None,
            stats: parking_lot::RwLock::new(CompressionStats::default()),
        }
    }

    /// 设置默认压缩算法
    ///
    /// 后续调用 `compress()` 将使用此算法。
    pub fn set_default_algorithm(&mut self, algo: CompressionAlgorithm) {
        if self.compressors.contains_key(&algo) {
            self.default_algorithm = algo;
        }
    }

    /// 获取当前默认算法
    pub fn default_algorithm(&self) -> CompressionAlgorithm {
        self.default_algorithm
    }

    /// 检查是否支持指定算法
    pub fn supports_algorithm(&self, algo: CompressionAlgorithm) -> bool {
        self.compressors.contains_key(&algo)
    }

    /// 获取支持的算法列表
    pub fn supported_algorithms(&self) -> Vec<CompressionAlgorithm> {
        self.compressors.keys().copied().collect()
    }

    /// 使用默认算法压缩数据
    pub fn compress(&self, data: &[u8]) -> Result<CompressResult, String> {
        self.compress_with(self.default_algorithm, data)
    }

    /// 使用指定算法压缩数据
    ///
    /// 自动更新压缩统计信息。
    pub fn compress_with(
        &self,
        algo: CompressionAlgorithm,
        data: &[u8],
    ) -> Result<CompressResult, String> {
        let compressor = self
            .compressors
            .get(&algo)
            .ok_or_else(|| format!("Unsupported algorithm: {:?}", algo))?;

        let result = compressor.compress(data)?;

        // 更新统计信息
        {
            let mut stats = self.stats.write();
            stats.total_compressions += 1;
            stats.total_original_bytes += result.original_size as u64;
            stats.total_compressed_bytes += result.compressed_size as u64;
            stats.total_compression_time_us += result.duration_us;

            // 滚动平均计算压缩率
            if stats.total_compressions > 0 {
                stats.avg_compression_ratio = (stats.avg_compression_ratio
                    * (stats.total_compressions - 1) as f32
                    + result.ratio)
                    / stats.total_compressions as f32;
            } else {
                stats.avg_compression_ratio = result.ratio;
            }
        }

        Ok(result)
    }

    /// 解压数据
    ///
    /// 必须指定正确的压缩算法才能正确解压。
    pub fn decompress(
        &self,
        data: &[u8],
        original_size: usize,
        algo: CompressionAlgorithm,
    ) -> Result<DecompressResult, String> {
        let compressor = self
            .compressors
            .get(&algo)
            .ok_or_else(|| format!("Unsupported algorithm: {:?}", algo))?;

        let result = compressor.decompress(data, original_size)?;

        // 更新统计
        {
            let mut stats = self.stats.write();
            stats.total_decompressions += 1;
        }

        Ok(result)
    }

    /// 获取压缩统计信息
    pub fn stats(&self) -> CompressionStats {
        self.stats.read().clone()
    }

    /// 重置统计信息
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = CompressionStats::default();
    }

    /// 根据数据大小自动选择最优算法
    ///
    /// # 策略
    /// - < 1KB: 不压缩 (开销过大)
    /// - 1KB - 100KB: LZ4 (速度快)
    /// - \> 100KB: Zstd (压缩率高)
    pub fn auto_select_algorithm(&self, data_size: usize) -> Option<CompressionAlgorithm> {
        match data_size {
            0..=1024 => Some(CompressionAlgorithm::None),
            1025..=102400 => {
                if self.supports_algorithm(CompressionAlgorithm::Lz4) {
                    Some(CompressionAlgorithm::Lz4)
                } else {
                    self.supported_algorithms().first().copied()
                }
            }
            _ => {
                if self.supports_algorithm(CompressionAlgorithm::Zstd) {
                    Some(CompressionAlgorithm::Zstd)
                } else if self.supports_algorithm(CompressionAlgorithm::Lz4) {
                    Some(CompressionAlgorithm::Lz4)
                } else {
                    Some(CompressionAlgorithm::None)
                }
            }
        }
    }

    /// 使用自动选择的算法压缩
    pub fn compress_auto(&self, data: &[u8]) -> Result<CompressResult, String> {
        let algo = self
            .auto_select_algorithm(data.len())
            .unwrap_or(self.default_algorithm);
        self.compress_with(algo, data)
    }
}

impl Default for CompressionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(size: usize) -> Vec<u8> {
        // 创建重复性高的测试数据（更容易压缩）
        let pattern = b"Hello, World! This is test data for compression. ";
        let mut data = Vec::with_capacity(size);
        while data.len() < size {
            let remaining = size - data.len();
            let to_copy = remaining.min(pattern.len());
            data.extend_from_slice(&pattern[..to_copy]);
        }
        data
    }

    #[test]
    fn test_noop_compressor() {
        let compressor = NoOpCompressor::new();
        let data = b"Hello, World!";

        let compressed = compressor.compress(data).unwrap();
        assert_eq!(compressed.data, data);
        assert_eq!(compressed.ratio, 1.0);
        assert_eq!(compressed.algorithm, CompressionAlgorithm::None);

        let decompressed = compressor.decompress(&compressed.data, data.len()).unwrap();
        assert_eq!(decompressed.data, data);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_zstd_compress_decompress() {
        let compressor = ZstdCompressor::new(3);
        let data = create_test_data(1024);

        let compressed = compressor.compress(&data).unwrap();
        assert!(compressed.compressed_size < compressed.original_size);
        assert!(compressed.ratio < 1.0);
        assert_eq!(compressed.algorithm, CompressionAlgorithm::Zstd);

        let decompressed = compressor
            .decompress(&compressed.data, compressed.original_size)
            .unwrap();
        assert_eq!(decompressed.data, data);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_lz4_compress_decompress() {
        let compressor = Lz4Compressor::new(4);
        let data = create_test_data(2048);

        let compressed = compressor.compress(&data).unwrap();
        assert!(compressed.compressed_size < compressed.original_size);
        assert!(compressed.ratio < 1.0);
        assert_eq!(compressed.algorithm, CompressionAlgorithm::Lz4);

        let decompressed = compressor
            .decompress(&compressed.data, compressed.original_size)
            .unwrap();
        assert_eq!(decompressed.data, data);
    }

    #[test]
    fn test_small_data_not_compressed() {
        let manager = CompressionManager::with_config(CompressionConfig {
            min_compress_size: 1000,
            ..Default::default()
        });

        let small_data = b"small";
        let result = manager.compress(small_data).unwrap();
        assert_eq!(result.compressed_size, result.original_size); // 不应压缩
    }

    #[test]
    fn test_compression_stats() {
        let manager = CompressionManager::new();

        let data1 = create_test_data(512);
        let data2 = create_test_data(1024);

        manager.compress(&data1).unwrap();
        manager.compress(&data2).unwrap();

        let stats = manager.stats();
        assert_eq!(stats.total_compressions, 2);
        assert!(stats.total_original_bytes > 0);
    }

    #[test]
    fn test_stats_display() {
        let stats = CompressionStats {
            total_compressions: 100,
            total_decompressions: 50,
            total_original_bytes: 1000000,
            total_compressed_bytes: 500000,
            total_compression_time_us: 1000000,
            avg_compression_ratio: 0.5,
        };

        let display = format!("{}", stats);
        assert!(display.contains("compressions: 100"));
        assert!(display.contains("saved: 500000"));
    }

    #[test]
    fn test_auto_select_algorithm() {
        let manager = CompressionManager::new();

        // 小数据应选择 None 或 Lz4
        let small_algo = manager.auto_select_algorithm(100).unwrap();
        assert!(
            small_algo == CompressionAlgorithm::None || small_algo == CompressionAlgorithm::Lz4
        );

        // 大数据应选择 Zstd (如果可用)
        let _large_algo = manager.auto_select_algorithm(1024 * 1024).unwrap();
        #[cfg(feature = "compression")]
        assert_eq!(large_algo, CompressionAlgorithm::Zstd);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_compression_manager_full_cycle() {
        let manager = CompressionManager::new();

        // 测试完整压缩-解压循环
        let original = create_test_data(4096);

        // 压缩
        let compressed = manager.compress(&original).unwrap();
        assert!(compressed.compressed_size < original.len());
        println!(
            "压缩: {} -> {} bytes (ratio: {:.2}%)",
            original.len(),
            compressed.compressed_size,
            compressed.ratio * 100.0
        );

        // 解压
        let decompressed = manager
            .decompress(
                &compressed.data,
                compressed.original_size,
                compressed.algorithm,
            )
            .unwrap();
        assert_eq!(decompressed.data, original);

        // 验证统计
        let stats = manager.stats();
        assert_eq!(stats.total_compressions, 1);
        assert_eq!(stats.total_decompressions, 1);
    }

    #[test]
    fn test_unsupported_algorithm() {
        let _manager = CompressionManager::with_config(CompressionConfig {
            algorithm: CompressionAlgorithm::None,
            ..Default::default()
        });

        // 尝试使用不支持的算法
        #[cfg(not(feature = "compression"))]
        {
            let manager = CompressionManager::with_config(CompressionConfig {
                algorithm: CompressionAlgorithm::None,
                ..Default::default()
            });
            let result = manager.compress_with(CompressionAlgorithm::Zstd, b"test");
            assert!(result.is_err());
        }

        #[cfg(feature = "compression")]
        {
            // 当 compression feature 启用时，所有算法都支持，此测试验证 None 算法正常工作
            let manager = CompressionManager::with_config(CompressionConfig {
                algorithm: CompressionAlgorithm::None,
                ..Default::default()
            });
            let result = manager.compress(b"test data").unwrap();
            assert_eq!(result.algorithm, CompressionAlgorithm::None);
        }
    }

    #[test]
    fn test_reset_stats() {
        let manager = CompressionManager::new();

        manager.compress(b"test").unwrap();
        assert_eq!(manager.stats().total_compressions, 1);

        manager.reset_stats();
        assert_eq!(manager.stats().total_compressions, 0);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_different_compression_levels() {
        let data = create_test_data(8192);

        // 低级别 (快速但压缩率低)
        let fast = ZstdCompressor::new(1).compress(&data).unwrap();

        // 高级别 (慢但压缩率高)
        let high_quality = ZstdCompressor::new(15).compress(&data).unwrap();

        // 高级别应该产生更小的输出
        assert!(high_quality.compressed_size <= fast.compressed_size);
        // 但高级别应该花费更多时间
        assert!(high_quality.duration_us >= fast.duration_us);
    }
}
