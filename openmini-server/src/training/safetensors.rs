//! Safetensors 权重格式支持
//!
//! 实现 safetensors 格式的读取和写入，
//! 兼容 HuggingFace 模型仓库标准。
//!
//! # 文件格式
//!
//! ```
//! +------------------+
//! |   Header (JSON)  |  <- 包含所有 tensor 的元数据
//! |   (size: u8)     |
//! +------------------+
//! |                  |
//! |   Tensor Data    |  <- 二进制张量数据
//! |   (binary blob)  |
//! +------------------+
//! ```
//!
//! # 示例
//!
//! ```ignore
//! use openmini_server::training::safetensors::*;
//! use ndarray::ArrayD;
//!
//! // 写入 safetensors 文件
//! let mut writer = SafeTensorsWriter::new("model.safetensors")?;
//! writer.add_tensor("embedding.weight", &embedding_array)?;
//! writer.finish()?;
//!
//! // 读取 safetensors 文件
//! let st = SafeTensorsFile::open("model.safetensors")?;
//! let tensor = st.load_tensor("embedding.weight")?;
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

/// Safetensors 操作错误类型
#[derive(Debug)]
pub enum SafeTensorsError {
    /// IO 错误
    Io(std::io::Error),
    /// 无效的 header
    InvalidHeader(String),
    /// 无效的 JSON
    InvalidJson(String),
    /// 形状不匹配
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Tensor 未找到
    TensorNotFound(String),
    /// Header 过大（超过 100MB）
    HeaderTooLarge(usize),
    /// 元数据不完整
    MetadataIncomplete,
    /// 其他错误
    Other(String),
}

impl From<std::io::Error> for SafeTensorsError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl std::fmt::Display for SafeTensorsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
            Self::InvalidJson(msg) => write!(f, "Invalid JSON: {}", msg),
            Self::ShapeMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "Shape mismatch for '{}': expected {:?}, got {:?}",
                name, expected, got
            ),
            Self::TensorNotFound(name) => write!(f, "Tensor '{}' not found", name),
            Self::HeaderTooLarge(size) => write!(f, "Header too large: {} bytes", size),
            Self::MetadataIncomplete => write!(f, "Metadata incomplete"),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for SafeTensorsError {}

/// Tensor 元数据（从 header 解析）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// 数据类型
    #[serde(rename = "type")]
    pub dtype: Dtype,
    /// 张量形状
    pub shape: Vec<usize>,
    /// 数据偏移量 [start, end]
    pub data_offsets: [usize; 2],
}

/// 数据类型枚举
///
/// 支持 HuggingFace safetensors 标准中定义的数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dtype {
    /// 32 位浮点数
    F32,
    /// 16 位浮点数
    F16,
    /// BFloat16
    BF16,
    /// 32 位整数
    I32,
    /// 64 位整数
    I64,
    /// 8 位无符号整数
    U8,
    /// 布尔值
    Bool,
}

impl Dtype {
    /// 返回每个元素的字节数
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
            Self::Bool => 1,
        }
    }

    /// 返回类型名称（safetensors 标准）
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
            Self::Bool => "BOOL",
        }
    }

    /// 从字符串解析 Dtype
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            "I32" => Some(Self::I32),
            "I64" => Some(Self::I64),
            "U8" => Some(Self::U8),
            "BOOL" => Some(Self::Bool),
            _ => None,
        }
    }
}

/// Safetensors 文件读取器
///
/// 用于打开和读取 safetensors 格式的权重文件。
/// 支持按需加载单个 tensor 或批量加载所有 tensor。
///
/// # 性能特性
///
/// - 延迟加载：只在调用 `load_tensor()` 时才读取实际数据
/// - 内存高效：元数据占用极小内存
/// - 支持大文件：可处理超过内存大小的文件
pub struct SafeTensorsFile {
    path: PathBuf,
    metadata: HashMap<String, TensorInfo>,
    data_offset: usize,
}

impl SafeTensorsFile {
    /// 打开一个 safetensors 文件并解析元数据
    ///
    /// 此操作只读取文件头部的 JSON 元数据，不会加载实际的张量数据。
    ///
    /// # 参数
    ///
    /// * `path` - safetensors 文件的路径
    ///
    /// # 错误
    ///
    /// 返回错误的情况：
    /// - 文件不存在或无法读取
    /// - header 大小超过 100MB 安全限制
    /// - JSON 解析失败
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let st = SafeTensorsFile::open("model.safetensors")?;
    /// println!("包含 {} 个 tensor", st.len());
    /// ```
    pub fn open(path: &Path) -> Result<Self, SafeTensorsError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // 读取 header 长度（前 8 字节，小端序 u64）
        let mut header_size_bytes = [0u8; 8];
        reader.read_exact(&mut header_size_bytes)?;
        let header_size = u64::from_le_bytes(header_size_bytes) as usize;

        // 安全检查：header 不能超过 100MB
        if header_size > 100 * 1024 * 1024 {
            return Err(SafeTensorsError::HeaderTooLarge(header_size));
        }

        // 读取 header JSON
        let mut header_json = vec![0u8; header_size];
        reader.read_exact(&mut header_json)?;
        let header_str = String::from_utf8(header_json)
            .map_err(|e| SafeTensorsError::InvalidHeader(format!("UTF-8 error: {}", e)))?;

        // 解析 JSON 为 HashMap<String, TensorInfo>
        let metadata: HashMap<String, TensorInfo> =
            serde_json::from_str(&header_str).map_err(|e| {
                SafeTensorsError::InvalidJson(format!("Parse error at {}: {}", e.line(), e))
            })?;

        // 验证元数据完整性
        for (name, info) in &metadata {
            if info.data_offsets[0] >= info.data_offsets[1] {
                return Err(SafeTensorsError::InvalidHeader(format!(
                    "Invalid data offsets for '{}': [{}, {}]",
                    name, info.data_offsets[0], info.data_offsets[1]
                )));
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            metadata,
            data_offset: 8 + header_size,
        })
    }

    /// 获取所有 tensor 名称
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.metadata.keys()
    }

    /// 获取指定 tensor 的元信息
    ///
    /// 返回 `None` 表示该 tensor 不存在
    pub fn get_info(&self, name: &str) -> Option<&TensorInfo> {
        self.metadata.get(name)
    }

    /// 加载指定的 tensor（返回 f32 类型）
    ///
    /// 自动处理 F16 → F32 的类型转换。
    ///
    /// # 参数
    ///
    /// * `name` - tensor 名称
    ///
    /// # 错误
    ///
    /// - Tensor 不存在
    /// - 不支持的数据类型（目前仅支持 F32 和 F16）
    /// - IO 错误
    pub fn load_tensor(&self, name: &str) -> Result<ArrayD<f32>, SafeTensorsError> {
        let info = self
            .metadata
            .get(name)
            .ok_or_else(|| SafeTensorsError::TensorNotFound(name.to_string()))?;

        let mut file = File::open(&self.path)?;
        let offset = self.data_offset + info.data_offsets[0];
        let length = info.data_offsets[1] - info.data_offsets[0];

        match info.dtype {
            Dtype::F32 => {
                let num_elements: usize = info.shape.iter().product();
                let expected_bytes = num_elements * Dtype::F32.element_size();

                if length != expected_bytes {
                    return Err(SafeTensorsError::ShapeMismatch {
                        name: name.to_string(),
                        expected: vec![expected_bytes],
                        got: vec![length],
                    });
                }

                file.seek(SeekFrom::Start(offset as u64))?;
                let mut buffer = vec![0u8; length];
                file.read_exact(&mut buffer)?;

                let values: Vec<f32> = buffer
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                let len = values.len();
                ArrayD::from_shape_vec(IxDyn(&info.shape), values).map_err(|_| {
                    SafeTensorsError::ShapeMismatch {
                        name: name.to_string(),
                        expected: info.shape.clone(),
                        got: vec![len],
                    }
                })
            }

            Dtype::F16 => {
                // FP16 → F32 转换
                let num_elements: usize = info.shape.iter().product();
                let expected_bytes = num_elements * Dtype::F16.element_size();

                if length != expected_bytes {
                    return Err(SafeTensorsError::ShapeMismatch {
                        name: name.to_string(),
                        expected: vec![expected_bytes],
                        got: vec![length],
                    });
                }

                file.seek(SeekFrom::Start(offset as u64))?;
                let mut buffer = vec![0u8; length];
                file.read_exact(&mut buffer)?;

                let values: Vec<f32> = buffer
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f16_to_f32(bits)
                    })
                    .collect();

                let len = values.len();
                ArrayD::from_shape_vec(IxDyn(&info.shape), values).map_err(|_| {
                    SafeTensorsError::ShapeMismatch {
                        name: name.to_string(),
                        expected: info.shape.clone(),
                        got: vec![len],
                    }
                })
            }

            _ => Err(SafeTensorsError::Other(format!(
                "Unsupported dtype: {} for tensor '{}'. Only F32 and F16 are supported",
                info.dtype.as_str(),
                name
            ))),
        }
    }

    /// 加载所有 tensors 到 HashMap
    ///
    /// 注意：这会将所有 tensor 数据加载到内存，对于大型模型可能消耗大量内存。
    /// 建议对于大型模型使用 `load_tensor()` 按需加载。
    pub fn load_all(&self) -> Result<HashMap<String, ArrayD<f32>>, SafeTensorsError> {
        let mut result = HashMap::with_capacity(self.metadata.len());
        for name in self.metadata.keys() {
            let tensor = self.load_tensor(name)?;
            result.insert(name.clone(), tensor);
        }
        Ok(result)
    }

    /// 获取 tensor 总数
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// 检查是否为空文件
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// 检查是否包含指定 tensor
    pub fn contains(&self, name: &str) -> bool {
        self.metadata.contains_key(name)
    }

    /// 获取文件路径
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Safetensors 文件写入器
///
/// 用于创建新的 safetensors 文件或覆盖现有文件。
/// 支持逐步添加 tensor，最后统一写入磁盘。
///
/// # 使用模式
///
/// ```ignore
/// let mut writer = SafeTensorsWriter::new("output.safetensors")?;
/// for (name, tensor) in &model_weights {
///     writer.add_tensor(name, tensor)?;
/// }
/// writer.finish()?;
/// ```
pub struct SafeTensorsWriter {
    file: BufWriter<File>,
    metadata: HashMap<String, TensorInfo>,
    data_buffer: Vec<u8>,
}

impl SafeTensorsWriter {
    /// 创建新文件准备写入
    ///
    /// 如果文件已存在，将被截断为空文件。
    ///
    /// # 参数
    ///
    /// * `path` - 输出文件的路径
    pub fn new(path: &Path) -> Result<Self, SafeTensorsError> {
        let file = File::create(path)?;
        Ok(Self {
            file: BufWriter::new(file),
            metadata: HashMap::new(),
            data_buffer: Vec::new(),
        })
    }

    /// 添加一个 f32 tensor 到文件
    ///
    /// 数据会被转换为小端序字节流存储在内部缓冲区，
    /// 在调用 `finish()` 时才会真正写入磁盘。
    ///
    /// # 参数
    ///
    /// * `name` - tensor 名称（建议使用点号分隔的层级命名，如 "layer.0.weight"）
    /// * `data` - 要存储的张量数据
    ///
    /// # 错误
    ///
    /// 如果同名 tensor 已添加过，会返回错误
    pub fn add_tensor(&mut self, name: &str, data: &ArrayD<f32>) -> Result<(), SafeTensorsError> {
        if self.metadata.contains_key(name) {
            return Err(SafeTensorsError::Other(format!(
                "Tensor '{}' already exists",
                name
            )));
        }

        let shape = data.shape().to_vec();
        let start_offset = self.data_buffer.len();

        // 写入 F32 数据（小端序）
        for &value in data.iter() {
            let bytes = value.to_le_bytes();
            self.data_buffer.extend_from_slice(&bytes);
        }

        let end_offset = self.data_buffer.len();

        self.metadata.insert(
            name.to_string(),
            TensorInfo {
                dtype: Dtype::F32,
                shape,
                data_offsets: [start_offset, end_offset],
            },
        );

        Ok(())
    }

    /// 完成写入并保存文件
    ///
    /// 此方法会：
    /// 1. 将元数据序列化为 JSON
    /// 2. 写入 8 字节的 header 长度（小端序）
    /// 3. 写入 JSON header
    /// 4. 写入所有 tensor 二进制数据
    /// 5. 刷新缓冲区到磁盘
    ///
    /// **重要**: 调用此方法后，writer 将被消费，不能再使用
    pub fn finish(mut self) -> Result<(), SafeTensorsError> {
        // 构建 header JSON（使用紧凑格式以减小文件大小）
        let header_json = serde_json::to_string(&self.metadata)
            .map_err(|e| SafeTensorsError::InvalidJson(format!("Serialization error: {}", e)))?;

        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        // 写入 header 长度（8 字节，小端序）
        self.file.write_all(&header_len.to_le_bytes())?;

        // 写入 header JSON
        self.file.write_all(header_bytes)?;

        // 写入 tensor 数据
        self.file.write_all(&self.data_buffer)?;

        // 确保所有数据写入磁盘
        self.file.flush()?;

        Ok(())
    }

    /// 获取已添加的 tensor 数量
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// 检查是否有 tensor 已添加
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }
}

/// FP16 → F32 转换函数
///
/// 实现 IEEE 754 半精度浮点到单精度浮点的转换。
/// 使用手动位运算实现，避免依赖外部 crate。
///
/// # 参数
///
/// * `bits` - FP16 的 16 位表示
///
/// # 返回值
///
/// 对应的 F32 值
fn f16_to_f32(bits: u16) -> f32 {
    // IEEE 754 半精度格式:
    // [1 sign bit][5 exponent bits][10 fraction bits]
    let sign = ((bits >> 15) & 1) as i32;
    let exponent = ((bits >> 10) & 0x1F) as i32;
    let fraction = (bits & 0x3FF) as u32;

    if exponent == 0 {
        // 非规格化数或零
        if fraction == 0 {
            // 正负零
            if sign != 0 {
                -0.0
            } else {
                0.0
            }
        } else {
            // 非规格化数
            let e = -14;
            let f = fraction as f32 / (1 << 10) as f32;
            let result = f * 2.0_f32.powi(e);
            if sign != 0 {
                -result
            } else {
                result
            }
        }
    } else if exponent == 31 {
        // 无穷大或 NaN
        if fraction == 0 {
            // 正负无穷
            if sign != 0 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            // NaN（保留部分有效位）
            f32::NAN
        }
    } else {
        // 规格化数
        let e = exponent - 15;
        let f = 1.0 + (fraction as f32) / (1 << 10) as f32;
        let result = f * 2.0_f32.powi(e);
        if sign != 0 {
            -result
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_load_safetensors() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("model.safetensors");

        // 创建文件
        let mut writer = SafeTensorsWriter::new(&path).unwrap();

        let embedding = ndarray::Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0])
            .into_shape_with_order([2, 2])
            .unwrap()
            .into_dyn();
        let linear = ndarray::Array::from_vec(vec![0.5f32, 0.25, 0.125, 0.0625])
            .into_shape_with_order([2, 2])
            .unwrap()
            .into_dyn();

        writer.add_tensor("embedding.weight", &embedding).unwrap();
        writer.add_tensor("linear.weight", &linear).unwrap();
        writer.finish().unwrap();

        // 加载验证
        let st = SafeTensorsFile::open(&path).unwrap();
        assert_eq!(st.len(), 2);

        let loaded_embedding = st.load_tensor("embedding.weight").unwrap();
        assert_eq!(loaded_embedding.shape(), &[2, 2]);
        assert!((loaded_embedding[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((loaded_embedding[[0, 1]] - 2.0).abs() < 1e-6);
        assert!((loaded_embedding[[1, 0]] - 3.0).abs() < 1e-6);
        assert!((loaded_embedding[[1, 1]] - 4.0).abs() < 1e-6);

        let loaded_linear = st.load_tensor("linear.weight").unwrap();
        assert_eq!(loaded_linear.shape(), &[2, 2]);
        assert!((loaded_linear[[0, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_not_found() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.safetensors");

        let mut writer = SafeTensorsWriter::new(&path).unwrap();
        let data = ndarray::Array::from_vec(vec![1.0f32]).into_dyn();
        writer.add_tensor("a", &data).unwrap();
        writer.finish().unwrap();

        let st = SafeTensorsFile::open(&path).unwrap();
        let result = st.load_tensor("nonexistent");
        assert!(result.is_err());

        match result.unwrap_err() {
            SafeTensorsError::TensorNotFound(name) => {
                assert_eq!(name, "nonexistent");
            }
            other => panic!("Expected TensorNotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_metadata_parsing() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("meta.safetensors");

        let mut writer = SafeTensorsWriter::new(&path).unwrap();
        let data = ndarray::Array::from_vec(vec![1.0f32, 2.0, 3.0])
            .into_shape_with_order([3])
            .unwrap()
            .into_dyn();
        writer.add_tensor("layer.weight", &data).unwrap();
        writer.finish().unwrap();

        let st = SafeTensorsFile::open(&path).unwrap();
        let info = st.get_info("layer.weight").unwrap();
        assert_eq!(info.shape, vec![3]);
        assert_eq!(info.dtype, Dtype::F32);
        assert!(st.contains("layer.weight"));
        assert!(!st.contains("nonexistent"));
    }

    #[test]
    fn test_large_tensor_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("large.safetensors");

        let mut writer = SafeTensorsWriter::new(&path).unwrap();
        let large_data: Vec<f32> = (0..10000).map(|i| i as f32 * 0.01).collect();
        let arr = ndarray::Array::from_vec(large_data)
            .into_shape_with_order([100, 100])
            .unwrap()
            .into_dyn();
        writer.add_tensor("big_matrix", &arr).unwrap();
        writer.finish().unwrap();

        let st = SafeTensorsFile::open(&path).unwrap();
        let loaded = st.load_tensor("big_matrix").unwrap();
        assert_eq!(loaded.shape(), &[100, 100]);

        // 验证特定位置的值
        assert!((loaded[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((loaded[[50, 50]] - 50.5).abs() < 1e-6);
        assert!((loaded[[99, 99]] - 99.99).abs() < 1e-6);
    }

    #[test]
    fn test_duplicate_tensor_error() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("dup.safetensors");

        let mut writer = SafeTensorsWriter::new(&path).unwrap();
        let data = ndarray::Array::from_vec(vec![1.0f32]).into_dyn();
        writer.add_tensor("tensor", &data).unwrap();

        let result = writer.add_tensor("tensor", &data);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("empty.safetensors");

        let writer = SafeTensorsWriter::new(&path).unwrap();
        assert!(writer.is_empty());
        writer.finish().unwrap();

        let st = SafeTensorsFile::open(&path).unwrap();
        assert!(st.is_empty());
        assert_eq!(st.len(), 0);
    }

    #[test]
    fn test_keys_iterator() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("keys.safetensors");

        let mut writer = SafeTensorsWriter::new(&path).unwrap();
        let data = ndarray::Array::from_vec(vec![1.0f32]).into_dyn();
        writer.add_tensor("a", &data).unwrap();
        writer.add_tensor("b", &data).unwrap();
        writer.add_tensor("c", &data).unwrap();
        writer.finish().unwrap();

        let st = SafeTensorsFile::open(&path).unwrap();
        let keys: Vec<&String> = st.keys().collect();
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn test_load_all_tensors() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("all.safetensors");

        let mut writer = SafeTensorsWriter::new(&path).unwrap();
        let data1 = ndarray::Array::from_vec(vec![1.0f32, 2.0])
            .into_shape_with_order([2])
            .unwrap()
            .into_dyn();
        let data2 = ndarray::Array::from_vec(vec![3.0f32, 4.0, 5.0])
            .into_shape_with_order([3])
            .unwrap()
            .into_dyn();
        writer.add_tensor("x", &data1).unwrap();
        writer.add_tensor("y", &data2).unwrap();
        writer.finish().unwrap();

        let st = SafeTensorsFile::open(&path).unwrap();
        let all = st.load_all().unwrap();
        assert_eq!(all.len(), 2);
        assert!(all.contains_key("x"));
        assert!(all.contains_key("y"));
    }

    #[test]
    fn test_f16_conversion_special_values() {
        // 测试 FP16 特殊值的转换
        assert_eq!(f16_to_f32(0x0000), 0.0); // 正零
        assert_eq!(f16_to_f32(0x8000), -0.0); // 负零
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY); // 正无穷
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY); // 负无穷
        assert!(f16_to_f32(0x7C01).is_nan()); // NaN
    }

    #[test]
    fn test_file_path_access() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("path_test.safetensors");

        let writer = SafeTensorsWriter::new(&path).unwrap();
        writer.finish().unwrap();

        let st = SafeTensorsFile::open(&path).unwrap();
        assert_eq!(st.path(), path);
    }
}
