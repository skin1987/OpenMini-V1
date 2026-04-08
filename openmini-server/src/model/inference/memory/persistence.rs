//! 记忆持久化存储模块
//!
//! 支持记忆的磁盘存储和恢复，包括：
//! - 同步/异步写入接口
//! - 索引文件管理
//! - 完整性校验
//!
//! # 文件格式
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │            File Header              │
//! │  - Magic Number (4 bytes)           │
//! │  - Version (4 bytes)                │
//! │  - Metadata Size (8 bytes)          │
//! │  - Data Size (8 bytes)              │
//! ├─────────────────────────────────────┤
//! │          Metadata (JSON)            │
//! │  - 配置信息                         │
//! │  - 统计信息                         │
//! ├─────────────────────────────────────┤
//! │          Index Data                 │
//! │  - 索引条目数组                     │
//! │  - 偏移量、长度                     │
//! ├─────────────────────────────────────┤
//! │          Memory Data                │
//! │  - 序列化的记忆数据                 │
//! └─────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{MemoryItem, MemoryLevel};

const MAGIC_NUMBER: u32 = 0x4D45_4D52; // "MEMR"
const CURRENT_VERSION: u32 = 1;
const INDEX_EXTENSION: &str = "idx";
const DATA_EXTENSION: &str = "mem";

/// 持久化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// 存储路径
    pub path: PathBuf,
    /// 是否启用异步写入
    pub async_write: bool,
    /// 刷新间隔（毫秒）
    pub flush_interval_ms: u64,
    /// 最大文件大小（MB）
    pub max_file_size_mb: usize,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./memory_store"),
            async_write: true,
            flush_interval_ms: 1000,
            max_file_size_mb: 1024,
        }
    }
}

impl PersistenceConfig {
    /// 使用指定存储路径创建持久化配置
    ///
    /// # 参数
    /// - `path`: 记忆数据存储目录路径
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }

    /// 设置是否启用异步写入模式
    ///
    /// # 参数
    /// - `async_write`: 是否异步写入
    pub fn with_async_write(mut self, async_write: bool) -> Self {
        self.async_write = async_write;
        self
    }

    /// 设置磁盘刷新间隔（毫秒）
    ///
    /// # 参数
    /// - `interval_ms`: 刷新间隔，单位毫秒
    pub fn with_flush_interval(mut self, interval_ms: u64) -> Self {
        self.flush_interval_ms = interval_ms;
        self
    }

    /// 设置最大文件大小限制（MB）
    ///
    /// # 参数
    /// - `size_mb`: 最大文件大小，单位 MB
    pub fn with_max_file_size(mut self, size_mb: usize) -> Self {
        self.max_file_size_mb = size_mb;
        self
    }
}

/// 文件头
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileHeader {
    magic: u32,
    version: u32,
    metadata_size: u64,
    data_size: u64,
}

impl Default for FileHeader {
    fn default() -> Self {
        Self {
            magic: MAGIC_NUMBER,
            version: CURRENT_VERSION,
            metadata_size: 0,
            data_size: 0,
        }
    }
}

impl FileHeader {
    const SIZE: usize = 4 + 4 + 8 + 8; // 24 bytes

    fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::SIZE);
        buf.extend_from_slice(&self.magic.to_le_bytes());
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.metadata_size.to_le_bytes());
        buf.extend_from_slice(&self.data_size.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Header too short",
            ));
        }

        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let metadata_size = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let data_size = u64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
        ]);

        Ok(Self {
            magic,
            version,
            metadata_size,
            data_size,
        })
    }
}

/// 元数据
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Metadata {
    /// 创建时间
    created_at: u64,
    /// 最后更新时间
    updated_at: u64,
    /// 记忆数量
    memory_count: usize,
    /// 索引数量
    index_count: usize,
    /// 配置信息
    config: PersistenceConfig,
}

/// 索引条目
///
/// 记录单个记忆数据在文件中的位置信息，用于快速定位和加载。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    /// 记忆ID
    pub memory_id: String,
    /// 数据偏移量
    pub offset: u64,
    /// 数据长度
    pub length: u64,
    /// 时间戳
    pub timestamp: u64,
    /// 重要性
    pub importance: f32,
    /// 记忆级别
    pub level: MemoryLevel,
}

impl IndexEntry {
    /// 创建新的索引条目
    ///
    /// # 参数
    /// - `memory_id`: 记忆唯一标识
    /// - `offset`: 数据在文件中的偏移量
    /// - `length`: 数据长度（字节）
    /// - `timestamp`: 创建时间戳
    /// - `importance`: 重要性分数
    /// - `level`: 记忆级别
    pub fn new(
        memory_id: String,
        offset: u64,
        length: u64,
        timestamp: u64,
        importance: f32,
        level: MemoryLevel,
    ) -> Self {
        Self {
            memory_id,
            offset,
            length,
            timestamp,
            importance,
            level,
        }
    }
}

/// 序列化的记忆数据
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedMemory {
    /// 数据形状 (rows, cols)
    shape: (usize, usize),
    /// 数据内容（行优先）
    data: Vec<f32>,
    /// 时间戳
    timestamp: u64,
    /// 重要性
    importance: f32,
    /// 会话ID
    session_id: Option<u64>,
}

impl From<&MemoryItem> for SerializedMemory {
    fn from(item: &MemoryItem) -> Self {
        let shape = (item.data.nrows(), item.data.ncols());
        let data = item.data.iter().copied().collect();
        Self {
            shape,
            data,
            timestamp: item.timestamp,
            importance: item.importance,
            session_id: item.session_id,
        }
    }
}

impl From<SerializedMemory> for MemoryItem {
    fn from(serialized: SerializedMemory) -> MemoryItem {
        use ndarray::Array2;

        let mut arr = Array2::zeros(serialized.shape);
        for (i, &val) in serialized.data.iter().enumerate() {
            let row = i / serialized.shape.1;
            let col = i % serialized.shape.1;
            arr[[row, col]] = val;
        }

        MemoryItem {
            data: arr,
            timestamp: serialized.timestamp,
            importance: serialized.importance,
            session_id: serialized.session_id,
        }
    }
}

/// 持久化存储引擎
///
/// 负责记忆数据的磁盘读写，支持：
/// - 单条/批量写入
/// - 按索引加载
/// - 完整性校验
/// - 原子写入事务（保证崩溃安全）
/// - 数据压缩（回收已删除空间）
///
/// # 文件布局
///
/// 存储目录包含两个文件：
/// - `data.mem`: 头部 + 元数据 + 记忆数据
/// - `index.idx`: 二进制索引文件
#[derive(Debug)]
pub struct Persistence {
    config: PersistenceConfig,
    data_file: Option<BufWriter<File>>,
    index_file: Option<BufWriter<File>>,
    header: FileHeader,
    metadata: Metadata,
    index: HashMap<String, IndexEntry>,
    current_offset: u64,
    /// 原子写入状态
    atomic_write_active: bool,
    /// 原子写入前的索引备份（用于回滚）
    saved_index: Option<HashMap<String, IndexEntry>>,
    /// 原子写入前的 header 备份（用于回滚）
    saved_header: Option<FileHeader>,
    /// 原子写入临时数据文件
    temp_data_file: Option<BufWriter<File>>,
    /// 原子写入临时索引文件
    temp_index_file: Option<BufWriter<File>>,
    /// 原子写入临时数据路径
    temp_data_path: Option<PathBuf>,
    /// 原子写入临时索引路径
    temp_index_path: Option<PathBuf>,
    /// 是否已关闭
    closed: bool,
}

impl Persistence {
    /// 创建新的持久化存储
    pub fn new(config: PersistenceConfig) -> io::Result<Self> {
        fs::create_dir_all(&config.path)?;

        Self::cleanup_temp_files(&config.path);

        let data_path = Self::data_path(&config.path);
        let index_path = Self::index_path(&config.path);

        let (header, metadata, index, current_offset) = if data_path.exists() && index_path.exists()
        {
            Self::load_existing(&config)?
        } else {
            let metadata = Metadata {
                created_at: current_timestamp(),
                updated_at: current_timestamp(),
                config: config.clone(),
                ..Default::default()
            };
            let metadata_bytes = serde_json::to_vec(&metadata).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Metadata error: {}", e))
            })?;
            let current_offset = (FileHeader::SIZE + metadata_bytes.len()) as u64;
            (
                FileHeader {
                    metadata_size: metadata_bytes.len() as u64,
                    ..Default::default()
                },
                metadata,
                HashMap::new(),
                current_offset,
            )
        };

        // 不使用 truncate(true)，因为需要保留现有数据用于追加和恢复
        #[allow(clippy::suspicious_open_options)]
        let data_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&data_path)
            .map(BufWriter::new)?;

        #[allow(clippy::suspicious_open_options)]
        let index_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&index_path)
            .map(BufWriter::new)?;

        Ok(Self {
            config,
            data_file: Some(data_file),
            index_file: Some(index_file),
            header,
            metadata,
            index,
            current_offset,
            atomic_write_active: false,
            saved_index: None,
            saved_header: None,
            temp_data_file: None,
            temp_index_file: None,
            temp_data_path: None,
            temp_index_path: None,
            closed: false,
        })
    }

    fn data_path(base: &Path) -> PathBuf {
        base.join("data.mem")
    }

    fn index_path(base: &Path) -> PathBuf {
        base.join("index.idx")
    }

    fn cleanup_temp_files(base: &Path) {
        let temp_patterns = ["mem.tmp", "idx.tmp", "compact.tmp"];
        for pattern in temp_patterns {
            if let Ok(entries) = fs::read_dir(base) {
                for entry in entries.flatten() {
                    if let Some(name) = entry.file_name().to_str() {
                        if name.ends_with(pattern) {
                            let _ = fs::remove_file(entry.path());
                        }
                    }
                }
            }
        }
    }

    fn load_existing(
        config: &PersistenceConfig,
    ) -> io::Result<(FileHeader, Metadata, HashMap<String, IndexEntry>, u64)> {
        let data_path = Self::data_path(&config.path);
        let index_path = Self::index_path(&config.path);

        let header: FileHeader = {
            let mut file = File::open(&data_path)?;
            let mut buf = vec![0u8; FileHeader::SIZE];
            file.read_exact(&mut buf)?;
            FileHeader::from_bytes(&buf)?
        };

        if header.magic != MAGIC_NUMBER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic number",
            ));
        }

        if header.version != CURRENT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: {}", header.version),
            ));
        }

        let metadata: Metadata = {
            let mut file = File::open(&data_path)?;
            file.seek(SeekFrom::Start(FileHeader::SIZE as u64))?;
            let mut buf = vec![0u8; header.metadata_size as usize];
            file.read_exact(&mut buf)?;
            serde_json::from_slice(&buf).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid metadata: {}", e),
                )
            })?
        };

        let index: HashMap<String, IndexEntry> = {
            let file = File::open(&index_path)?;
            let mut reader = BufReader::new(file);
            let mut buf = Vec::new();
            reader.read_to_end(&mut buf)?;
            let entries: Vec<IndexEntry> = bincode::deserialize(&buf).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Invalid index: {}", e))
            })?;
            entries
                .into_iter()
                .map(|e| (e.memory_id.clone(), e))
                .collect()
        };

        let current_offset = FileHeader::SIZE as u64 + header.metadata_size + header.data_size;

        Ok((header, metadata, index, current_offset))
    }

    /// 写入单个记忆到存储
    ///
    /// 将记忆数据序列化后追加写入数据文件，并更新索引。
    ///
    /// # 参数
    /// - `memory_id`: 记忆唯一标识
    /// - `item`: 记忆数据
    /// - `level`: 记忆级别
    pub fn write_memory(
        &mut self,
        memory_id: &str,
        item: &MemoryItem,
        level: MemoryLevel,
    ) -> io::Result<()> {
        self.ensure_not_closed()?;
        let serialized = SerializedMemory::from(item);
        let data = serde_json::to_vec(&serialized).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Serialization error: {}", e),
            )
        })?;

        let offset = self.current_offset;
        let length = data.len() as u64;

        if self.atomic_write_active {
            if let Some(ref mut file) = self.temp_data_file {
                file.seek(SeekFrom::Start(offset))?;
                file.write_all(&data)?;
            }
        } else if let Some(ref mut file) = self.data_file {
            file.seek(SeekFrom::Start(offset))?;
            file.write_all(&data)?;
        }

        let entry = IndexEntry::new(
            memory_id.to_string(),
            offset,
            length,
            item.timestamp,
            item.importance,
            level,
        );

        self.index.insert(memory_id.to_string(), entry);
        self.current_offset += length;
        self.header.data_size += length;
        self.metadata.memory_count = self.index.len();
        self.metadata.index_count = self.index.len();
        self.metadata.updated_at = current_timestamp();

        Ok(())
    }

    /// 批量写入多个记忆
    ///
    /// 逐条调用 `write_memory`，适用于一次性写入多个记忆的场景。
    ///
    /// # 参数
    /// - `memories`: 待写入的记忆列表，每项为 (ID, 数据, 级别) 元组
    pub fn write_batch(
        &mut self,
        memories: &[(String, MemoryItem, MemoryLevel)],
    ) -> io::Result<()> {
        for (id, item, level) in memories {
            self.write_memory(id, item, *level)?;
        }
        Ok(())
    }

    /// 将缓冲区数据刷新到磁盘
    ///
    /// 写入文件头、元数据和索引，确保所有数据持久化。
    /// 建议在重要操作后调用以保证数据安全。
    pub fn flush(&mut self) -> io::Result<()> {
        self.ensure_not_closed()?;
        self.write_header_and_metadata()?;

        self.write_index()?;

        if let Some(ref mut file) = self.data_file {
            file.flush()?;
        }

        if let Some(ref mut file) = self.index_file {
            file.flush()?;
        }

        Ok(())
    }

    fn write_header_and_metadata(&mut self) -> io::Result<()> {
        let metadata_bytes = serde_json::to_vec(&self.metadata).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Metadata error: {}", e))
        })?;

        self.header.metadata_size = metadata_bytes.len() as u64;

        if let Some(ref mut file) = self.data_file {
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&self.header.to_bytes())?;
            file.write_all(&metadata_bytes)?;
        }

        Ok(())
    }

    fn write_index(&mut self) -> io::Result<()> {
        let entries: Vec<&IndexEntry> = self.index.values().collect();

        if let Some(ref mut file) = self.index_file {
            file.seek(SeekFrom::Start(0))?;
            let data = bincode::serialize(&entries).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Index serialization error: {}", e),
                )
            })?;
            file.write_all(&data)?;
        }

        Ok(())
    }

    /// 加载所有已存储的记忆
    ///
    /// 从数据文件中按索引读取所有记忆数据。
    ///
    /// # 返回
    /// 所有记忆的列表，每项为 (ID, 数据, 级别) 元组
    pub fn load(&self) -> io::Result<Vec<(String, MemoryItem, MemoryLevel)>> {
        self.ensure_not_closed()?;
        let data_path = Self::data_path(&self.config.path);
        let file = File::open(&data_path)?;
        let mut reader = BufReader::new(file);

        reader.seek(SeekFrom::Start(
            FileHeader::SIZE as u64 + self.header.metadata_size,
        ))?;

        let mut result = Vec::with_capacity(self.index.len());

        let mut entries: Vec<&IndexEntry> = self.index.values().collect();
        entries.sort_by_key(|e| e.offset);

        for entry in entries {
            reader.seek(SeekFrom::Start(entry.offset))?;
            let mut buf = vec![0u8; entry.length as usize];
            reader.read_exact(&mut buf)?;

            let serialized: SerializedMemory = serde_json::from_slice(&buf).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Deserialize error: {}", e),
                )
            })?;

            let item: MemoryItem = serialized.into();
            result.push((entry.memory_id.clone(), item, entry.level));
        }

        Ok(result)
    }

    /// 获取内存中的索引映射（只读引用）
    ///
    /// # 返回
    /// 从 memory_id 到 IndexEntry 的 HashMap 引用
    pub fn load_index(&self) -> &HashMap<String, IndexEntry> {
        &self.index
    }

    /// 根据记忆 ID 加载单个记忆
    ///
    /// # 参数
    /// - `memory_id`: 记忆唯一标识
    ///
    /// # 返回
    /// 找到返回 `Some(MemoryItem)`，未找到返回 `None`
    pub fn load_by_id(&self, memory_id: &str) -> io::Result<Option<MemoryItem>> {
        self.ensure_not_closed()?;
        let entry = match self.index.get(memory_id) {
            Some(e) => e,
            None => return Ok(None),
        };

        let data_path = Self::data_path(&self.config.path);
        let file = File::open(&data_path)?;
        let mut reader = BufReader::new(file);

        reader.seek(SeekFrom::Start(entry.offset))?;
        let mut buf = vec![0u8; entry.length as usize];
        reader.read_exact(&mut buf)?;

        let serialized: SerializedMemory = serde_json::from_slice(&buf).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Deserialize error: {}", e),
            )
        })?;

        Ok(Some(serialized.into()))
    }

    /// 执行存储完整性校验
    ///
    /// 检查文件头、元数据、索引和数据的一致性。
    ///
    /// # 返回
    /// 包含各项校验结果和错误信息的 `VerifyResult`
    pub fn verify(&self) -> io::Result<VerifyResult> {
        self.ensure_not_closed()?;
        let mut result = VerifyResult {
            valid: true,
            header_valid: true,
            metadata_valid: true,
            index_valid: true,
            data_valid: true,
            errors: Vec::new(),
        };

        if self.header.magic != MAGIC_NUMBER {
            result.header_valid = false;
            result.errors.push("Invalid magic number".to_string());
        }

        if self.header.version != CURRENT_VERSION {
            result.header_valid = false;
            result
                .errors
                .push(format!("Unsupported version: {}", self.header.version));
        }

        if self.metadata.memory_count != self.index.len() {
            result.metadata_valid = false;
            result.errors.push(format!(
                "Memory count mismatch: metadata={}, index={}",
                self.metadata.memory_count,
                self.index.len()
            ));
        }

        let index_path = Self::index_path(&self.config.path);
        match File::open(&index_path) {
            Ok(file) => {
                let mut reader = BufReader::new(file);
                let mut buf = Vec::new();
                if let Err(e) = reader.read_to_end(&mut buf) {
                    result.index_valid = false;
                    result.errors.push(format!("Index file read error: {}", e));
                } else {
                    match bincode::deserialize::<Vec<IndexEntry>>(&buf) {
                        Ok(entries) => {
                            if entries.len() != self.index.len() {
                                result.index_valid = false;
                                result.errors.push(format!(
                                    "Index file entry count mismatch: file={}, memory={}",
                                    entries.len(),
                                    self.index.len()
                                ));
                            }
                        }
                        Err(e) => {
                            result.index_valid = false;
                            result.errors.push(format!("Index file parse error: {}", e));
                        }
                    }
                }
            }
            Err(e) => {
                if !self.index.is_empty() {
                    result.index_valid = false;
                    result.errors.push(format!("Index file open error: {}", e));
                }
            }
        }

        let data_path = Self::data_path(&self.config.path);
        let file = File::open(&data_path)?;
        let file_size = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        for entry in self.index.values() {
            if entry.offset < FileHeader::SIZE as u64 + self.header.metadata_size {
                result.data_valid = false;
                result.errors.push(format!(
                    "Entry {} offset {} is before data start",
                    entry.memory_id, entry.offset
                ));
                continue;
            }

            if entry.offset + entry.length > file_size {
                result.data_valid = false;
                result.errors.push(format!(
                    "Entry {} offset {} + length {} exceeds file size {}",
                    entry.memory_id, entry.offset, entry.length, file_size
                ));
                continue;
            }

            reader.seek(SeekFrom::Start(entry.offset))?;
            let mut buf = vec![0u8; entry.length as usize];

            match reader.read_exact(&mut buf) {
                Ok(_) => {
                    if serde_json::from_slice::<SerializedMemory>(&buf).is_err() {
                        result.data_valid = false;
                        result
                            .errors
                            .push(format!("Invalid data for memory: {}", entry.memory_id));
                    }
                }
                Err(e) => {
                    result.data_valid = false;
                    result
                        .errors
                        .push(format!("Failed to read memory {}: {}", entry.memory_id, e));
                }
            }
        }

        result.valid =
            result.header_valid && result.metadata_valid && result.index_valid && result.data_valid;

        Ok(result)
    }

    /// 删除指定 ID 的记忆
    ///
    /// 仅从索引中移除条目，不释放磁盘空间（需调用 `compact` 回收）。
    ///
    /// # 参数
    /// - `memory_id`: 待删除的记忆 ID
    ///
    /// # 返回
    /// `true` 表示找到并删除，`false` 表示未找到
    pub fn remove(&mut self, memory_id: &str) -> io::Result<bool> {
        self.ensure_not_closed()?;
        if self.index.remove(memory_id).is_some() {
            self.metadata.memory_count = self.index.len();
            self.metadata.index_count = self.index.len();
            self.metadata.updated_at = current_timestamp();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 压缩数据文件，回收已删除记忆占用的空间
    ///
    /// 重写数据文件，移除已删除记忆的数据，更新索引偏移量。
    /// 这是一个昂贵的操作，建议在删除大量记忆后手动调用。
    pub fn compact(&mut self) -> io::Result<()> {
        self.ensure_not_closed()?;

        if self.atomic_write_active {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot compact during atomic write transaction",
            ));
        }

        if self.index.is_empty() {
            return self.clear();
        }

        let data_path = Self::data_path(&self.config.path);
        let index_path = Self::index_path(&self.config.path);

        let temp_data_path = data_path.with_extension("compact.tmp");
        let temp_index_path = index_path.with_extension("compact.tmp");

        let mut entries: Vec<&mut IndexEntry> = self.index.values_mut().collect();
        entries.sort_by_key(|e| e.offset);

        let temp_data_file = File::create(&temp_data_path)?;
        let mut writer = BufWriter::new(temp_data_file);

        writer.seek(SeekFrom::Start(0))?;
        writer.write_all(&self.header.to_bytes())?;
        let metadata_bytes = serde_json::to_vec(&self.metadata).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Metadata error: {}", e))
        })?;
        writer.write_all(&metadata_bytes)?;

        let mut new_offset = (FileHeader::SIZE + metadata_bytes.len()) as u64;
        let mut new_data_size = 0u64;

        {
            let src_file = File::open(&data_path)?;
            let mut reader = BufReader::new(src_file);

            for entry in &mut entries {
                reader.seek(SeekFrom::Start((*entry).offset))?;
                let mut buf = vec![0u8; (*entry).length as usize];
                reader.read_exact(&mut buf)?;

                writer.write_all(&buf)?;

                (*entry).offset = new_offset;
                new_offset += (*entry).length;
                new_data_size += (*entry).length;
            }
        }

        writer.flush()?;

        {
            let entries: Vec<&IndexEntry> = self.index.values().collect();
            let temp_index_file = File::create(&temp_index_path)?;
            let mut index_writer = BufWriter::new(temp_index_file);
            let data = bincode::serialize(&entries).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Index serialization error: {}", e),
                )
            })?;
            index_writer.write_all(&data)?;
            index_writer.flush()?;
        }

        fs::rename(&temp_index_path, &index_path)?;
        fs::rename(&temp_data_path, &data_path)?;

        self.header.data_size = new_data_size;
        self.current_offset = new_offset;

        {
            let data_file = OpenOptions::new()
                .write(true)
                .read(true)
                .open(&data_path)
                .map(BufWriter::new)?;
            let index_file = OpenOptions::new()
                .write(true)
                .read(true)
                .open(&index_path)
                .map(BufWriter::new)?;
            self.data_file = Some(data_file);
            self.index_file = Some(index_file);
        }

        Ok(())
    }

    /// 清空所有记忆数据
    ///
    /// 清除索引并截断数据文件，释放所有磁盘空间。
    pub fn clear(&mut self) -> io::Result<()> {
        self.ensure_not_closed()?;
        self.index.clear();
        self.header.data_size = 0;
        self.metadata.memory_count = 0;
        self.metadata.index_count = 0;
        self.metadata.updated_at = current_timestamp();

        let metadata_bytes = serde_json::to_vec(&self.metadata).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Metadata error: {}", e))
        })?;
        self.header.metadata_size = metadata_bytes.len() as u64;
        self.current_offset = (FileHeader::SIZE + metadata_bytes.len()) as u64;

        if let Some(ref mut file) = self.data_file {
            file.get_mut().set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&self.header.to_bytes())?;
            file.write_all(&metadata_bytes)?;
            file.flush()?;
        }

        if let Some(ref mut file) = self.index_file {
            file.get_mut().set_len(0)?;
            file.flush()?;
        }

        Ok(())
    }

    /// 获取存储统计信息
    ///
    /// # 返回
    /// 包含记忆数量、数据大小、创建/更新时间的统计信息
    pub fn stats(&self) -> PersistenceStats {
        PersistenceStats {
            memory_count: self.index.len(),
            data_size: self.header.data_size,
            created_at: self.metadata.created_at,
            updated_at: self.metadata.updated_at,
        }
    }

    /// 检查数据文件大小是否超过配置的限制
    ///
    /// # 返回
    /// `true` 表示已超过 `max_file_size_mb` 限制
    pub fn is_size_exceeded(&self) -> bool {
        let size_mb = self.header.data_size as f64 / (1024.0 * 1024.0);
        size_mb >= self.config.max_file_size_mb as f64
    }

    /// 关闭存储并释放文件句柄
    ///
    /// 先刷新缓冲区数据，然后关闭所有文件。
    /// 关闭后不能再进行读写操作。
    pub fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        self.flush()?;
        self.data_file = None;
        self.index_file = None;
        self.closed = true;
        Ok(())
    }

    fn ensure_not_closed(&self) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Persistence is closed",
            ));
        }
        Ok(())
    }

    /// 开始原子写入事务
    ///
    /// 创建临时文件，后续所有写入都写入临时文件。
    /// 完成后调用 `commit_atomic_write` 原子替换原文件。
    /// 如果中途崩溃或调用 `rollback_atomic_write`，原有数据仍然保留。
    pub fn begin_atomic_write(&mut self) -> io::Result<()> {
        self.ensure_not_closed()?;

        let data_path = Self::data_path(&self.config.path);
        let index_path = Self::index_path(&self.config.path);

        let temp_data_path = data_path.with_extension("mem.tmp");
        let temp_index_path = index_path.with_extension("idx.tmp");

        let temp_data_file = File::create(&temp_data_path)?;
        let mut temp_data_writer = BufWriter::new(temp_data_file);

        self.saved_index = Some(self.index.clone());
        self.saved_header = Some(self.header.clone());

        self.index.clear();
        self.header.data_size = 0;
        self.current_offset = FileHeader::SIZE as u64 + self.header.metadata_size;

        temp_data_writer.seek(SeekFrom::Start(0))?;
        temp_data_writer.write_all(&self.header.to_bytes())?;
        let metadata_bytes = serde_json::to_vec(&self.metadata).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Metadata error: {}", e))
        })?;
        temp_data_writer.write_all(&metadata_bytes)?;
        temp_data_writer.flush()?;

        self.temp_data_file = Some(temp_data_writer);
        self.temp_index_file = None;
        self.temp_data_path = Some(temp_data_path);
        self.temp_index_path = Some(temp_index_path);
        self.atomic_write_active = true;

        Ok(())
    }

    /// 提交原子写入事务
    ///
    /// 刷新临时文件，写入索引，然后使用 fs::rename 原子替换原文件。
    pub fn commit_atomic_write(&mut self) -> io::Result<()> {
        if !self.atomic_write_active {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "No atomic write in progress",
            ));
        }

        self.ensure_not_closed()?;
        self.metadata.memory_count = self.index.len();
        self.metadata.index_count = self.index.len();
        self.metadata.updated_at = current_timestamp();

        if let Some(ref mut file) = self.temp_data_file {
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&self.header.to_bytes())?;
            let metadata_bytes = serde_json::to_vec(&self.metadata).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Metadata error: {}", e))
            })?;
            file.write_all(&metadata_bytes)?;
            file.flush()?;
        }

        let temp_index_path = self.temp_index_path.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Temp index path not set")
        })?;
        let temp_data_path = self
            .temp_data_path
            .take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Temp data path not set"))?;

        {
            let entries: Vec<&IndexEntry> = self.index.values().collect();
            let temp_index_file = File::create(&temp_index_path)?;
            let mut writer = BufWriter::new(temp_index_file);
            let data = bincode::serialize(&entries).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Index serialization error: {}", e),
                )
            })?;
            writer.write_all(&data)?;
            writer.flush()?;
        }

        let data_path = Self::data_path(&self.config.path);
        let index_path = Self::index_path(&self.config.path);

        fs::rename(&temp_index_path, &index_path)?;
        fs::rename(&temp_data_path, &data_path)?;

        self.temp_data_file = None;
        self.temp_index_file = None;
        self.atomic_write_active = false;

        {
            let data_file = OpenOptions::new()
                .write(true)
                .read(true)
                .open(&data_path)
                .map(BufWriter::new)?;
            let index_file = OpenOptions::new()
                .write(true)
                .read(true)
                .open(&index_path)
                .map(BufWriter::new)?;
            self.data_file = Some(data_file);
            self.index_file = Some(index_file);
        }

        Ok(())
    }

    /// 回滚原子写入事务
    ///
    /// 放弃当前原子写入，删除临时文件，恢复原状态。
    pub fn rollback_atomic_write(&mut self) -> io::Result<()> {
        if !self.atomic_write_active {
            return Ok(());
        }

        self.temp_data_file = None;
        self.temp_index_file = None;

        if let Some(ref path) = self.temp_data_path {
            let _ = fs::remove_file(path);
        }
        if let Some(ref path) = self.temp_index_path {
            let _ = fs::remove_file(path);
        }

        if let Some(saved_index) = self.saved_index.take() {
            self.index = saved_index;
        }
        if let Some(saved_header) = self.saved_header.take() {
            self.header = saved_header;
        }

        self.current_offset =
            FileHeader::SIZE as u64 + self.header.metadata_size + self.header.data_size;

        self.temp_data_path = None;
        self.temp_index_path = None;
        self.saved_index = None;
        self.saved_header = None;
        self.atomic_write_active = false;

        Ok(())
    }

    /// 检查是否处于原子写入状态
    pub fn is_atomic_write_active(&self) -> bool {
        self.atomic_write_active
    }
}

impl Drop for Persistence {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// 存储完整性校验结果
///
/// 包含各项校验的通过/失败状态和详细的错误信息列表。
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// 整体是否通过校验（所有子项均为 true 时为 true）
    pub valid: bool,
    /// 文件头是否有效（magic number + 版本号）
    pub header_valid: bool,
    /// 元数据是否一致（记忆数量匹配）
    pub metadata_valid: bool,
    /// 索引文件是否可正确解析
    pub index_valid: bool,
    /// 所有记忆数据是否可正确反序列化
    pub data_valid: bool,
    /// 校验过程中发现的错误信息列表
    pub errors: Vec<String>,
}

/// 存储统计信息
///
/// 提供当前存储的使用情况概览。
#[derive(Debug, Clone)]
pub struct PersistenceStats {
    /// 当前已存储的记忆数量
    pub memory_count: usize,
    /// 数据文件大小（字节）
    pub data_size: u64,
    /// 存储创建时间戳（毫秒级 UNIX 时间）
    pub created_at: u64,
    /// 最后更新时间戳（毫秒级 UNIX 时间）
    pub updated_at: u64,
}

/// 线程安全的持久化存储包装器
///
/// 使用 RwLock 保护内部 Persistence 实例，支持多线程并发读取和写入。
/// 注意：写入操作需要获取写锁，会阻塞其他所有操作。
pub struct ThreadSafePersistence {
    inner: std::sync::RwLock<Persistence>,
}

impl ThreadSafePersistence {
    /// 创建线程安全的持久化存储实例
    ///
    /// # 参数
    /// - `config`: 持久化配置
    pub fn new(config: PersistenceConfig) -> io::Result<Self> {
        Ok(Self {
            inner: std::sync::RwLock::new(Persistence::new(config)?),
        })
    }

    /// 线程安全地写入单个记忆（需要写锁）
    pub fn write_memory(
        &self,
        memory_id: &str,
        item: &MemoryItem,
        level: MemoryLevel,
    ) -> io::Result<()> {
        self.inner
            .write()
            .unwrap()
            .write_memory(memory_id, item, level)
    }

    /// 线程安全地批量写入记忆（需要写锁）
    pub fn write_batch(&self, memories: &[(String, MemoryItem, MemoryLevel)]) -> io::Result<()> {
        self.inner.write().unwrap().write_batch(memories)
    }

    /// 线程安全地刷新数据到磁盘（需要写锁）
    pub fn flush(&self) -> io::Result<()> {
        self.inner.write().unwrap().flush()
    }

    /// 线程安全地加载所有记忆（需要读锁）
    pub fn load(&self) -> io::Result<Vec<(String, MemoryItem, MemoryLevel)>> {
        self.inner.read().unwrap().load()
    }

    /// 线程安全地按 ID 加载单个记忆（需要读锁）
    pub fn load_by_id(&self, memory_id: &str) -> io::Result<Option<MemoryItem>> {
        self.inner.read().unwrap().load_by_id(memory_id)
    }

    /// 线程安全地执行完整性校验（需要读锁）
    pub fn verify(&self) -> io::Result<VerifyResult> {
        self.inner.read().unwrap().verify()
    }

    /// 线程安全地删除记忆（需要写锁）
    pub fn remove(&self, memory_id: &str) -> io::Result<bool> {
        self.inner.write().unwrap().remove(memory_id)
    }

    /// 线程安全地清空所有记忆（需要写锁）
    pub fn clear(&self) -> io::Result<()> {
        self.inner.write().unwrap().clear()
    }

    /// 线程安全地压缩数据文件（需要写锁）
    pub fn compact(&self) -> io::Result<()> {
        self.inner.write().unwrap().compact()
    }

    /// 获取存储统计信息（需要读锁）
    pub fn stats(&self) -> PersistenceStats {
        self.inner.read().unwrap().stats()
    }

    /// 检查文件大小是否超限（需要读锁）
    pub fn is_size_exceeded(&self) -> bool {
        self.inner.read().unwrap().is_size_exceeded()
    }

    /// 关闭存储（需要写锁）
    pub fn close(&self) -> io::Result<()> {
        self.inner.write().unwrap().close()
    }

    /// 开始原子写入事务（需要写锁）
    pub fn begin_atomic_write(&self) -> io::Result<()> {
        self.inner.write().unwrap().begin_atomic_write()
    }

    /// 提交原子写入事务（需要写锁）
    pub fn commit_atomic_write(&self) -> io::Result<()> {
        self.inner.write().unwrap().commit_atomic_write()
    }

    /// 回滚原子写入事务（需要写锁）
    pub fn rollback_atomic_write(&self) -> io::Result<()> {
        self.inner.write().unwrap().rollback_atomic_write()
    }

    /// 检查是否处于原子写入状态（需要读锁）
    pub fn is_atomic_write_active(&self) -> bool {
        self.inner.read().unwrap().is_atomic_write_active()
    }
}

/// 异步写入命令
enum AsyncCommand {
    Write {
        memory_id: String,
        item: MemoryItem,
        level: MemoryLevel,
        response: std::sync::mpsc::Sender<io::Result<()>>,
    },
    WriteBatch {
        memories: Vec<(String, MemoryItem, MemoryLevel)>,
        response: std::sync::mpsc::Sender<io::Result<()>>,
    },
    Flush {
        response: std::sync::mpsc::Sender<io::Result<()>>,
    },
    Close,
}

/// 异步写入器
///
/// 在后台线程中执行磁盘写入操作，通过 channel 接收写入命令。
/// 支持同步等待（`*_sync` 方法）和异步发送（不等待）两种模式。
///
/// 内部按配置的 `flush_interval_ms` 自动刷新缓冲区到磁盘。
pub struct AsyncWriter {
    sender: Sender<AsyncCommand>,
    handle: Option<JoinHandle<io::Result<()>>>,
}

impl AsyncWriter {
    /// 创建异步写入器
    pub fn new(config: PersistenceConfig) -> io::Result<Self> {
        let (sender, receiver) = mpsc::channel();
        let flush_interval = Duration::from_millis(config.flush_interval_ms);

        let handle = thread::spawn(move || {
            let mut persistence = Persistence::new(config)?;
            let mut pending_flush = false;

            loop {
                match receiver.recv_timeout(flush_interval) {
                    Ok(AsyncCommand::Write {
                        memory_id,
                        item,
                        level,
                        response,
                    }) => {
                        let result = persistence.write_memory(&memory_id, &item, level);
                        let _ = response.send(result);
                        pending_flush = true;
                    }
                    Ok(AsyncCommand::WriteBatch { memories, response }) => {
                        let result = persistence.write_batch(&memories);
                        let _ = response.send(result);
                        pending_flush = true;
                    }
                    Ok(AsyncCommand::Flush { response }) => {
                        let result = persistence.flush();
                        let _ = response.send(result);
                        pending_flush = false;
                    }
                    Ok(AsyncCommand::Close) => {
                        if pending_flush {
                            let _ = persistence.flush();
                        }
                        persistence.close()?;
                        return Ok(());
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if pending_flush {
                            let _ = persistence.flush();
                            pending_flush = false;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        if pending_flush {
                            let _ = persistence.flush();
                        }
                        let _ = persistence.close();
                        return Ok(());
                    }
                }
            }
        });

        Ok(Self {
            sender,
            handle: Some(handle),
        })
    }

    /// 异步写入单个记忆，等待写入完成并返回结果
    pub fn write_sync(
        &self,
        memory_id: String,
        item: MemoryItem,
        level: MemoryLevel,
    ) -> io::Result<()> {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(AsyncCommand::Write {
                memory_id,
                item,
                level,
                response: tx,
            })
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))?;
        rx.recv()
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))?
    }

    /// 异步写入单个记忆（不等待结果）
    pub fn write(&self, memory_id: String, item: MemoryItem, level: MemoryLevel) -> io::Result<()> {
        let (tx, _) = mpsc::channel();
        self.sender
            .send(AsyncCommand::Write {
                memory_id,
                item,
                level,
                response: tx,
            })
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))
    }

    /// 异步批量写入（等待结果）
    pub fn write_batch_sync(
        &self,
        memories: Vec<(String, MemoryItem, MemoryLevel)>,
    ) -> io::Result<()> {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(AsyncCommand::WriteBatch {
                memories,
                response: tx,
            })
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))?;
        rx.recv()
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))?
    }

    /// 异步批量写入（不等待结果）
    pub fn write_batch(&self, memories: Vec<(String, MemoryItem, MemoryLevel)>) -> io::Result<()> {
        let (tx, _) = mpsc::channel();
        self.sender
            .send(AsyncCommand::WriteBatch {
                memories,
                response: tx,
            })
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))
    }

    /// 请求刷新并等待完成
    pub fn flush_sync(&self) -> io::Result<()> {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(AsyncCommand::Flush { response: tx })
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))?;
        rx.recv()
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))?
    }

    /// 请求刷新（不等待结果）
    pub fn flush(&self) -> io::Result<()> {
        let (tx, _) = mpsc::channel();
        self.sender
            .send(AsyncCommand::Flush { response: tx })
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))
    }

    /// 关闭异步写入器
    pub fn close(mut self) -> io::Result<()> {
        self.sender
            .send(AsyncCommand::Close)
            .map_err(|e| io::Error::new(io::ErrorKind::BrokenPipe, e.to_string()))?;

        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|e| io::Error::other(format!("Thread error: {:?}", e)))?
        } else {
            Ok(())
        }
    }

    /// 检查后台线程是否健康
    pub fn is_healthy(&self) -> bool {
        self.handle.is_some()
    }

    /// 尝试关闭异步写入器
    ///
    /// 发送关闭命令并阻塞等待后台线程结束。
    /// 如果线程已 panic 或已关闭，返回错误。
    ///
    /// 注意：此方法会阻塞调用线程直到后台线程完成关闭。
    /// 如需非阻塞关闭，请直接丢弃 AsyncWriter 实例。
    pub fn try_close(&mut self) -> io::Result<()> {
        let send_result = self.sender.send(AsyncCommand::Close);

        if send_result.is_err() {
            self.handle = None;
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "AsyncWriter channel already closed",
            ));
        }

        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(result) => result,
                Err(_) => Err(io::Error::other("AsyncWriter thread panic")),
            }
        } else {
            Ok(())
        }
    }
}

/// 获取当前时间戳（毫秒级精度）
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use tempfile::TempDir;

    fn create_test_memory(rows: usize, cols: usize) -> MemoryItem {
        let data = Array2::from_shape_fn((rows, cols), |(i, j)| (i * cols + j) as f32);
        MemoryItem::new(data, current_timestamp()).with_importance(0.5)
    }

    #[test]
    fn test_persistence_config() {
        let config = PersistenceConfig::new("/tmp/test")
            .with_async_write(false)
            .with_flush_interval(2000)
            .with_max_file_size(512);

        assert_eq!(config.path, PathBuf::from("/tmp/test"));
        assert!(!config.async_write);
        assert_eq!(config.flush_interval_ms, 2000);
        assert_eq!(config.max_file_size_mb, 512);
    }

    #[test]
    fn test_file_header() {
        let header = FileHeader::default();
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), FileHeader::SIZE);

        let parsed = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.magic, MAGIC_NUMBER);
        assert_eq!(parsed.version, CURRENT_VERSION);
    }

    #[test]
    fn test_serialized_memory() {
        let item = create_test_memory(10, 128);
        let serialized = SerializedMemory::from(&item);

        assert_eq!(serialized.shape, (10, 128));
        assert_eq!(serialized.data.len(), 10 * 128);

        let recovered: MemoryItem = serialized.into();
        assert_eq!(recovered.data.nrows(), 10);
        assert_eq!(recovered.data.ncols(), 128);
    }

    #[test]
    fn test_persistence_write_read() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            PersistenceConfig::new(temp_dir.path().join("test_memory")).with_async_write(false);

        let mut persistence = Persistence::new(config).unwrap();

        let item1 = create_test_memory(10, 128);
        let item2 = create_test_memory(5, 64);

        persistence
            .write_memory("mem1", &item1, MemoryLevel::LongTerm)
            .unwrap();
        persistence
            .write_memory("mem2", &item2, MemoryLevel::ShortTerm)
            .unwrap();
        persistence.flush().unwrap();

        let loaded = persistence.load().unwrap();
        assert_eq!(loaded.len(), 2);

        let mem1 = persistence.load_by_id("mem1").unwrap();
        assert!(mem1.is_some());
        assert_eq!(mem1.unwrap().data.nrows(), 10);

        persistence.close().unwrap();
    }

    #[test]
    fn test_persistence_batch_write() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            PersistenceConfig::new(temp_dir.path().join("batch_test")).with_async_write(false);

        let mut persistence = Persistence::new(config).unwrap();

        let memories: Vec<(String, MemoryItem, MemoryLevel)> = (0..5)
            .map(|i| {
                let item = create_test_memory(i + 1, 32);
                (format!("batch_{}", i), item, MemoryLevel::LongTerm)
            })
            .collect();

        persistence.write_batch(&memories).unwrap();
        persistence.flush().unwrap();

        let loaded = persistence.load().unwrap();
        assert_eq!(loaded.len(), 5);

        persistence.close().unwrap();
    }

    #[test]
    fn test_persistence_verify() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            PersistenceConfig::new(temp_dir.path().join("verify_test")).with_async_write(false);

        let mut persistence = Persistence::new(config).unwrap();

        let item = create_test_memory(10, 64);
        persistence
            .write_memory("verify_mem", &item, MemoryLevel::LongTerm)
            .unwrap();
        persistence.flush().unwrap();

        let result = persistence.verify().unwrap();
        assert!(result.valid);
        assert!(result.header_valid);
        assert!(result.data_valid);

        persistence.close().unwrap();
    }

    #[test]
    fn test_persistence_remove() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            PersistenceConfig::new(temp_dir.path().join("remove_test")).with_async_write(false);

        let mut persistence = Persistence::new(config).unwrap();

        let item = create_test_memory(5, 32);
        persistence
            .write_memory("to_remove", &item, MemoryLevel::LongTerm)
            .unwrap();

        let removed = persistence.remove("to_remove").unwrap();
        assert!(removed);

        let removed_again = persistence.remove("to_remove").unwrap();
        assert!(!removed_again);

        persistence.flush().unwrap();
        persistence.close().unwrap();
    }

    #[test]
    fn test_persistence_clear() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            PersistenceConfig::new(temp_dir.path().join("clear_test")).with_async_write(false);

        let mut persistence = Persistence::new(config).unwrap();

        for i in 0..3 {
            let item = create_test_memory(5, 32);
            persistence
                .write_memory(&format!("clear_{}", i), &item, MemoryLevel::LongTerm)
                .unwrap();
        }

        persistence.clear().unwrap();

        let loaded = persistence.load().unwrap();
        assert!(loaded.is_empty());

        persistence.close().unwrap();
    }

    #[test]
    fn test_persistence_stats() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            PersistenceConfig::new(temp_dir.path().join("stats_test")).with_async_write(false);

        let mut persistence = Persistence::new(config).unwrap();

        let item = create_test_memory(10, 64);
        persistence
            .write_memory("stats_mem", &item, MemoryLevel::LongTerm)
            .unwrap();

        let stats = persistence.stats();
        assert_eq!(stats.memory_count, 1);
        assert!(stats.data_size > 0);

        persistence.close().unwrap();
    }

    #[test]
    fn test_async_writer() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig::new(temp_dir.path().join("async_test"))
            .with_async_write(true)
            .with_flush_interval(100);

        let writer = AsyncWriter::new(config).unwrap();

        let item = create_test_memory(5, 32);
        writer
            .write("async_mem".to_string(), item, MemoryLevel::LongTerm)
            .unwrap();

        writer.flush().unwrap();

        let result = writer.close();
        assert!(result.is_ok());
    }

    #[test]
    fn test_persistence_reload() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("reload_test");

        {
            let config = PersistenceConfig::new(&path).with_async_write(false);
            let mut persistence = Persistence::new(config).unwrap();

            let item = create_test_memory(10, 64);
            persistence
                .write_memory("reload_mem", &item, MemoryLevel::LongTerm)
                .unwrap();
            persistence.flush().unwrap();
            persistence.close().unwrap();
        }

        {
            let config = PersistenceConfig::new(&path).with_async_write(false);
            let persistence = Persistence::new(config).unwrap();

            let loaded = persistence.load().unwrap();
            assert_eq!(loaded.len(), 1);
            assert_eq!(loaded[0].0, "reload_mem");
        }
    }

    #[test]
    fn test_index_entry() {
        let entry = IndexEntry::new(
            "test_id".to_string(),
            100,
            200,
            12345,
            0.8,
            MemoryLevel::LongTerm,
        );

        assert_eq!(entry.memory_id, "test_id");
        assert_eq!(entry.offset, 100);
        assert_eq!(entry.length, 200);
        assert_eq!(entry.timestamp, 12345);
        assert_eq!(entry.importance, 0.8);
    }

    #[test]
    fn test_atomic_write() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            PersistenceConfig::new(temp_dir.path().join("atomic_test")).with_async_write(false);

        let mut persistence = Persistence::new(config).unwrap();

        persistence.begin_atomic_write().unwrap();
        assert!(persistence.is_atomic_write_active());

        let item = create_test_memory(5, 32);
        persistence
            .write_memory("atomic_mem", &item, MemoryLevel::LongTerm)
            .unwrap();

        persistence.commit_atomic_write().unwrap();
        assert!(!persistence.is_atomic_write_active());

        let loaded = persistence.load().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "atomic_mem");

        persistence.close().unwrap();
    }

    #[test]
    fn test_atomic_write_rollback() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            PersistenceConfig::new(temp_dir.path().join("rollback_test")).with_async_write(false);

        let mut persistence = Persistence::new(config).unwrap();

        let item = create_test_memory(5, 32);
        persistence
            .write_memory("existing_mem", &item, MemoryLevel::LongTerm)
            .unwrap();
        persistence.flush().unwrap();

        persistence.begin_atomic_write().unwrap();
        persistence
            .write_memory("temp_mem", &item, MemoryLevel::LongTerm)
            .unwrap();

        persistence.rollback_atomic_write().unwrap();
        assert!(!persistence.is_atomic_write_active());

        let loaded = persistence.load().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "existing_mem");

        persistence.close().unwrap();
    }

    #[test]
    fn test_thread_safe_persistence() {
        use std::sync::Arc;
        use std::thread;

        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig::new(temp_dir.path().join("thread_safe_test"))
            .with_async_write(false);

        let persistence = Arc::new(ThreadSafePersistence::new(config).unwrap());

        let mut handles = vec![];

        for i in 0..4 {
            let p = Arc::clone(&persistence);
            let handle = thread::spawn(move || {
                let item = create_test_memory(5, 32);
                p.write_memory(&format!("thread_{}", i), &item, MemoryLevel::LongTerm)
                    .unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        persistence.flush().unwrap();

        let loaded = persistence.load().unwrap();
        assert_eq!(loaded.len(), 4);

        persistence.close().unwrap();
    }

    #[test]
    fn test_compact() {
        let temp_dir = TempDir::new().unwrap();
        let config =
            PersistenceConfig::new(temp_dir.path().join("compact_test")).with_async_write(false);

        let mut persistence = Persistence::new(config).unwrap();

        for i in 0..5 {
            let item = create_test_memory(5, 32);
            persistence
                .write_memory(&format!("mem_{}", i), &item, MemoryLevel::LongTerm)
                .unwrap();
        }
        persistence.flush().unwrap();

        let stats_before = persistence.stats();
        assert_eq!(stats_before.memory_count, 5);

        persistence.remove("mem_1").unwrap();
        persistence.remove("mem_3").unwrap();

        persistence.compact().unwrap();

        let stats_after = persistence.stats();
        assert_eq!(stats_after.memory_count, 3);
        assert!(stats_after.data_size < stats_before.data_size);

        let loaded = persistence.load().unwrap();
        assert_eq!(loaded.len(), 3);

        let ids: Vec<&str> = loaded.iter().map(|(id, _, _)| id.as_str()).collect();
        assert!(ids.contains(&"mem_0"));
        assert!(ids.contains(&"mem_2"));
        assert!(ids.contains(&"mem_4"));

        persistence.close().unwrap();
    }

    #[test]
    fn test_async_writer_sync() {
        let temp_dir = TempDir::new().unwrap();
        let config = PersistenceConfig::new(temp_dir.path().join("async_sync_test"))
            .with_async_write(true)
            .with_flush_interval(100);

        let writer = AsyncWriter::new(config).unwrap();

        let item = create_test_memory(5, 32);
        writer
            .write_sync("sync_mem".to_string(), item, MemoryLevel::LongTerm)
            .unwrap();

        writer.flush_sync().unwrap();

        let result = writer.close();
        assert!(result.is_ok());
    }
}
