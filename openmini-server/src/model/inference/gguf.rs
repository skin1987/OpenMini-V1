//! GGUF 文件格式解析模块
//!
//! 本模块实现 GGUF (GGML Universal Format) 文件格式的解析，包括：
//! - 文件头解析
//! - 元数据解析
//! - 张量信息解析
//! - 张量数据读取
//!
//! # GGUF 文件格式
//!
//! GGUF 是一种高效的模型权重存储格式，支持：
//! - 多种数据类型（F32, F16, 量化类型等）
//! - 结构化元数据
//! - 内存映射访问
//!
//! # 支持的量化类型
//!
//! - Q4_0, Q4_1: 4位量化
//! - Q8_0: 8位量化
//! - Q2K - Q8K: K-量化系列

#![allow(dead_code)]
//! - IQ 系列: 智能量化


// ============================================================================
// 标准库和外部依赖导入
// ============================================================================

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Result};
use bytemuck::cast_slice;
use memmap2::Mmap;

// ============================================================================
// GGUF 常量定义
// ============================================================================

/// GGUF 文件魔数 ("GGUF" 的小端序)
pub const GGUF_MAGIC: u32 = 0x46554747;

// ============================================================================
// 默认模型配置常量
// ============================================================================

/// 默认隐藏层大小
pub const DEFAULT_HIDDEN_SIZE: usize = 3584;
/// 默认隐藏层数量
pub const DEFAULT_NUM_HIDDEN_LAYERS: usize = 28;
/// 默认注意力头数量
pub const DEFAULT_NUM_ATTENTION_HEADS: usize = 28;
/// 默认中间层大小
pub const DEFAULT_INTERMEDIATE_SIZE: usize = 18944;
/// 默认词表大小
pub const DEFAULT_VOCAB_SIZE: usize = 152064;
/// 默认最大位置编码长度
pub const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 131072;
/// 默认 RoPE 基础频率
pub const DEFAULT_ROPE_THETA: f32 = 1000000.0;
/// 默认 RMS 归一化 epsilon
pub const DEFAULT_RMS_NORM_EPS: f32 = 1e-6;

// ============================================================================
// GGUF 值类型定义
// ============================================================================

/// GGUF 值类型枚举
///
/// 定义 GGUF 文件中支持的所有数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    /// 无符号 8 位整数
    UInt8 = 0,
    /// 有符号 8 位整数
    Int8 = 1,
    /// 无符号 16 位整数
    UInt16 = 2,
    /// 有符号 16 位整数
    Int16 = 3,
    /// 无符号 32 位整数
    UInt32 = 4,
    /// 有符号 32 位整数
    Int32 = 5,
    /// 32 位浮点数
    Float32 = 6,
    /// 布尔值
    Bool = 7,
    /// 字符串
    String = 8,
    /// 数组
    Array = 9,
    /// 无符号 64 位整数
    UInt64 = 10,
    /// 有符号 64 位整数
    Int64 = 11,
    /// 64 位浮点数
    Float64 = 12,
}

/// GGUF 值枚举
///
/// 存储实际的元数据值
#[derive(Debug, Clone)]
pub enum GgufValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

// ============================================================================
// GGUF 元数据
// ============================================================================

/// GGUF 元数据结构体
///
/// 存储模型的所有元数据键值对
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    /// 键值对映射
    pub kv: HashMap<String, GgufValue>,
}

impl GgufMetadata {
    /// 获取字符串类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// 字符串值（如果存在且类型正确）
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.kv.get(key) {
            Some(GgufValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// 获取 u32 类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// u32 值（如果存在且类型兼容）
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.kv.get(key) {
            Some(GgufValue::UInt32(v)) => Some(*v),
            Some(GgufValue::Int32(v)) => Some(*v as u32),
            _ => None,
        }
    }

    /// 获取 u64 类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// u64 值（如果存在且类型兼容）
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.kv.get(key) {
            Some(GgufValue::UInt64(v)) => Some(*v),
            Some(GgufValue::Int64(v)) => Some(*v as u64),
            Some(GgufValue::UInt32(v)) => Some(*v as u64),
            Some(GgufValue::Int32(v)) => Some(*v as u64),
            _ => None,
        }
    }

    /// 获取 f32 类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// f32 值（如果存在且类型正确）
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.kv.get(key) {
            Some(GgufValue::Float32(v)) => Some(*v),
            _ => None,
        }
    }

    /// 获取数组类型的元数据值
    ///
    /// # Parameters
    /// - `key`: 元数据键名
    ///
    /// # Returns
    /// 数组切片（如果存在且类型正确）
    pub fn get_array(&self, key: &str) -> Option<&[GgufValue]> {
        match self.kv.get(key) {
            Some(GgufValue::Array(arr)) => Some(arr),
            _ => None,
        }
    }
}

// ============================================================================
// GGUF 张量类型定义
// ============================================================================

/// GGUF 张量类型枚举
///
/// 定义支持的所有张量数据类型，包括浮点和量化类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    /// 32 位浮点数
    F32 = 0,
    /// 16 位浮点数
    F16 = 1,
    /// 4 位量化 (类型 0)
    Q4_0 = 2,
    /// 4 位量化 (类型 1)
    Q4_1 = 3,
    /// 4 位量化 (类型 2)
    Q4_2 = 4,
    /// 4 位量化 (类型 3)
    Q4_3 = 5,
    /// 8 位量化 (类型 0)
    Q8_0 = 7,
    /// 8 位量化 (类型 1)
    Q8_1 = 8,
    /// 2 位 K-量化
    Q2K = 10,
    /// 3 位 K-量化
    Q3K = 11,
    /// 4 位 K-量化
    Q4K = 12,
    /// 5 位 K-量化
    Q5K = 13,
    /// 6 位 K-量化
    Q6K = 14,
    /// 8 位 K-量化
    Q8K = 15,
    /// 智能量化 2XXS
    Iq2Xxs = 16,
    /// 智能量化 2XS
    Iq2Xs = 17,
    /// 智能量化 3XXS
    Iq3Xxs = 18,
    /// 智能量化 1S
    Iq1S = 19,
    /// 智能量化 4NL
    Iq4Nl = 20,
    /// 智能量化 3S
    Iq3S = 21,
    /// 智能量化 2S
    Iq2S = 22,
    /// 智能量化 4XS
    Iq4Xs = 23,
    /// 8 位整数
    I8 = 24,
    /// 16 位整数
    I16 = 25,
    /// 32 位整数
    I32 = 26,
    /// 64 位整数
    I64 = 27,
    /// 64 位浮点数
    F64 = 28,
    /// BF16 浮点数
    BF16 = 29,
}

impl GgufTensorType {
    /// 从 u8 值创建张量类型
    ///
    /// # Parameters
    /// - `v`: 类型编号
    ///
    /// # Returns
    /// 张量类型（如果有效）
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            4 => Some(Self::Q4_2),
            5 => Some(Self::Q4_3),
            // 类型值 6 在 GGUF 规范中未定义/保留
            7 => Some(Self::Q8_0),
            8 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::Iq2Xxs),
            17 => Some(Self::Iq2Xs),
            18 => Some(Self::Iq3Xxs),
            19 => Some(Self::Iq1S),
            20 => Some(Self::Iq4Nl),
            21 => Some(Self::Iq3S),
            22 => Some(Self::Iq2S),
            23 => Some(Self::Iq4Xs),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::BF16),
            _ => None,
        }
    }

    /// 获取量化块大小
    ///
    /// 量化类型按块进行，此方法返回每个块的元素数量
    ///
    /// # Returns
    /// 每个量化块的元素数量
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q4_2 | Self::Q4_3 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::Iq2Xxs | Self::Iq2Xs | Self::Iq2S => 256,
            Self::Iq3Xxs | Self::Iq3S => 256,
            Self::Iq1S => 256,
            Self::Iq4Nl | Self::Iq4Xs => 256,
        }
    }

    /// 获取每个量化块的字节数
    ///
    /// # Returns
    /// 每个量化块占用的字节数
    ///
    /// # Note
    /// Q4_2 和 Q4_3 已在 GGUF 规范中标记为废弃，仅用于兼容旧模型
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,  // 2 (scale) + 16 (32 * 4bit / 8)
            Self::Q4_1 => 20,  // 2 (scale) + 2 (min) + 16 (32 * 4bit / 8)
            Self::Q4_2 => 18,  // 已废弃: 2 (scale) + 16 (32 * 4bit / 8)
            Self::Q4_3 => 20,  // 已废弃: 2 (scale) + 2 (min) + 16 (32 * 4bit / 8)
            Self::Q8_0 => 34,  // 2 (scale) + 32 (32 * 8bit)
            Self::Q8_1 => 36,  // 2 (scale) + 32 (32 * 8bit) + 2 (sum)
            Self::Q2K => 68,
            Self::Q3K => 162,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 258,
            Self::Iq1S => 132,
            Self::Iq2Xxs => 64,
            Self::Iq2Xs => 64,
            Self::Iq2S => 64,
            Self::Iq3Xxs => 96,
            Self::Iq3S => 96,
            Self::Iq4Nl => 130,
            Self::Iq4Xs => 130,
        }
    }

    /// 获取单个元素的字节数（非量化类型）
    ///
    /// # Returns
    /// 单个元素占用的字节数
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            _ => self.bytes_per_block() / self.block_size(),
        }
    }
}

// ============================================================================
// GGUF 张量信息
// ============================================================================

/// GGUF 张量信息结构体
///
/// 存储单个张量的元数据信息
#[derive(Debug, Clone)]
pub struct GgufTensor {
    /// 张量名称
    pub name: String,
    /// 张量维度
    pub dims: Vec<usize>,
    /// 张量数据类型
    pub tensor_type: GgufTensorType,
    /// 数据在文件中的偏移量
    pub offset: u64,
}

impl GgufTensor {
    /// 计算张量的元素总数
    ///
    /// # Returns
    /// 所有维度的乘积
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// 计算张量数据的大小（字节）
    ///
    /// # Returns
    /// 张量数据占用的字节数
    pub fn data_size(&self) -> usize {
        let n = self.num_elements();
        let block_size = self.tensor_type.block_size();
        let bytes_per_block = self.tensor_type.bytes_per_block();
        let num_blocks = n.div_ceil(block_size);
        num_blocks * bytes_per_block
    }
}

// ============================================================================
// 多模态配置结构体
// ============================================================================

/// 视觉编码器配置
///
/// 从 GGUF 文件中提取的视觉模型配置
#[derive(Debug, Clone)]
pub struct VisionConfigGGUF {
    /// 输入图像大小
    pub image_size: usize,
    /// 图像块大小
    pub patch_size: usize,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// 层数
    pub num_layers: usize,
    /// 注意力头数
    pub num_heads: usize,
    /// 中间层大小
    pub intermediate_size: usize,
}

/// 音频编码器配置
///
/// 从 GGUF 文件中提取的音频模型配置
#[derive(Debug, Clone)]
pub struct AudioConfigGGUF {
    /// 采样率
    pub sample_rate: usize,
    /// 梅尔频谱 bin 数
    pub num_mel_bins: usize,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// 层数
    pub num_layers: usize,
    /// 注意力头数
    pub num_heads: usize,
}

/// TTS 配置
///
/// 从 GGUF 文件中提取的语音合成模型配置
#[derive(Debug, Clone)]
pub struct TTSConfigGGUF {
    /// 采样率
    pub sample_rate: usize,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// Flow 层数
    pub num_flow_layers: usize,
}

// ============================================================================
// GGUF 文件头
// ============================================================================

/// GGUF 文件头结构体
///
/// 存储文件的基本信息
#[derive(Debug)]
pub struct GgufHeader {
    /// GGUF 版本号
    pub version: u32,
    /// 张量数量
    pub tensor_count: u64,
    /// 元数据键值对数量
    pub metadata_kv_count: u64,
}

// ============================================================================
// 模型配置
// ============================================================================

/// 模型配置结构体
///
/// 存储语言模型的完整配置信息
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// 词表大小
    pub vocab_size: usize,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// 中间层大小
    pub intermediate_size: usize,
    /// 隐藏层数量
    pub num_hidden_layers: usize,
    /// 注意力头数量
    pub num_attention_heads: usize,
    /// KV 头数量（用于 GQA）
    pub num_key_value_heads: usize,
    /// 最大位置编码长度
    pub max_position_embeddings: usize,
    /// RMS 归一化 epsilon
    pub rms_norm_eps: f32,
    /// RoPE 基础频率
    pub rope_theta: f32,
}

impl From<super::model::ModelConfig> for ModelConfig {
    fn from(config: super::model::ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
        }
    }
}

impl From<ModelConfig> for super::model::ModelConfig {
    fn from(config: ModelConfig) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            ..Default::default()
        }
    }
}

// ============================================================================
// GGUF 文件结构体
// ============================================================================

/// GGUF 文件结构体
///
/// 代表一个已解析的 GGUF 文件，提供访问元数据和张量数据的接口
#[derive(Debug)]
pub struct GgufFile {
    /// 文件头
    pub header: GgufHeader,
    /// 元数据
    pub metadata: GgufMetadata,
    /// 张量信息映射
    pub tensors: HashMap<String, GgufTensor>,
    /// 内存映射
    mmap: Mmap,
}

impl GgufFile {
    /// 打开 GGUF 文件
    ///
    /// # Parameters
    /// - `path`: 文件路径
    ///
    /// # Returns
    /// 解析后的 GGUF 文件实例
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // 使用内存映射提高大文件读取效率
        let mmap = unsafe { Mmap::map(&file)? };
        
        Self::parse(mmap)
    }

    /// 获取视觉编码器张量
    ///
    /// 筛选出属于视觉模型的张量
    ///
    /// # Returns
    /// 视觉张量名称到张量信息的映射
    pub fn get_vision_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.iter_vision_tensors()
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 迭代视觉编码器张量
    ///
    /// 返回迭代器，避免分配 HashMap
    pub fn iter_vision_tensors(&self) -> impl Iterator<Item = (&String, &GgufTensor)> {
        self.tensors.iter().filter(|(name, _)| {
            name.starts_with("vpm.") || 
            name.starts_with("vision_model.") ||
            name.starts_with("vision_encoder.") ||
            name.starts_with("visual.")
        })
    }

    /// 获取重采样器张量
    ///
    /// 筛选出属于多模态投影层的张量
    ///
    /// # Returns
    /// 重采样器张量名称到张量信息的映射
    pub fn get_resampler_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.iter_resampler_tensors()
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 迭代重采样器张量
    ///
    /// 返回迭代器，避免分配 HashMap
    pub fn iter_resampler_tensors(&self) -> impl Iterator<Item = (&String, &GgufTensor)> {
        self.tensors.iter().filter(|(name, _)| {
            name.starts_with("resampler.") ||
            name.starts_with("mm_projector.") ||
            name.starts_with("vision_proj.") ||
            name.starts_with("multi_modal_projector.")
        })
    }

    /// 获取语言模型张量
    ///
    /// 筛选出属于语言模型的张量
    ///
    /// # Returns
    /// 语言模型张量名称到张量信息的映射
    pub fn get_llm_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.iter_llm_tensors()
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 迭代语言模型张量
    ///
    /// 返回迭代器，避免分配 HashMap
    pub fn iter_llm_tensors(&self) -> impl Iterator<Item = (&String, &GgufTensor)> {
        self.tensors.iter().filter(|(name, _)| {
            name.starts_with("llm.") || 
            name.starts_with("model.") ||
            name.starts_with("language_model.") ||
            (!name.starts_with("vpm.") && 
             !name.starts_with("vision_model.") &&
             !name.starts_with("resampler.") &&
             !name.starts_with("mm_projector."))
        })
    }

    /// 检查是否包含视觉张量
    ///
    /// # Returns
    /// 如果文件包含视觉编码器权重则返回 true
    pub fn has_vision_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("vpm.") || 
            name.starts_with("vision_model.") ||
            name.starts_with("vision_encoder.") ||
            name.starts_with("visual.")
        })
    }

    /// 检查是否包含重采样器张量
    ///
    /// # Returns
    /// 如果文件包含多模态投影层权重则返回 true
    pub fn has_resampler_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("resampler.") ||
            name.starts_with("mm_projector.") ||
            name.starts_with("vision_proj.")
        })
    }

    /// 获取音频编码器张量
    ///
    /// # Returns
    /// 音频编码器张量名称到张量信息的映射
    pub fn get_audio_encoder_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.tensors.iter()
            .filter(|(name, _)| {
                name.starts_with("audio_encoder.") ||
                name.starts_with("encoder.audio.") ||
                name.starts_with("whisper.") ||
                name.starts_with("audio_model.")
            })
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 获取 TTS 张量
    ///
    /// # Returns
    /// TTS 张量名称到张量信息的映射
    pub fn get_tts_tensors(&self) -> HashMap<String, &GgufTensor> {
        self.tensors.iter()
            .filter(|(name, _)| {
                name.starts_with("tts.") ||
                name.starts_with("flow.") ||
                name.starts_with("vocoder.") ||
                name.starts_with("speech_decoder.")
            })
            .map(|(k, v)| (k.clone(), v))
            .collect()
    }

    /// 检查是否包含音频编码器张量
    ///
    /// # Returns
    /// 如果文件包含音频编码器权重则返回 true
    pub fn has_audio_encoder_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("audio_encoder.") ||
            name.starts_with("encoder.audio.") ||
            name.starts_with("whisper.") ||
            name.starts_with("audio_model.")
        })
    }

    /// 检查是否包含 TTS 张量
    ///
    /// # Returns
    /// 如果文件包含语音合成权重则返回 true
    pub fn has_tts_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("tts.") ||
            name.starts_with("flow.") ||
            name.starts_with("vocoder.") ||
            name.starts_with("speech_decoder.")
        })
    }

    /// 检查是否包含语言模型张量
    ///
    /// # Returns
    /// 如果文件包含语言模型权重则返回 true
    pub fn has_llm_tensors(&self) -> bool {
        self.tensors.keys().any(|name| {
            name.starts_with("model.") ||
            name.starts_with("llm.") ||
            name.starts_with("language_model.") ||
            name.starts_with("transformer.") ||
            (name.starts_with("tok_embeddings.") && !name.starts_with("vision_") && !name.starts_with("audio_"))
        })
    }

    /// 获取音频编码器配置
    ///
    /// 从元数据中提取音频编码器配置
    ///
    /// # Returns
    /// 音频配置（如果存在）
    pub fn get_audio_config(&self) -> Option<AudioConfigGGUF> {
        if !self.has_audio_encoder_tensors() {
            return None;
        }

        let sample_rate = self.metadata.get_u32("audio.sample_rate")
            .or_else(|| self.metadata.get_u32("whisper.sample_rate"))
            .unwrap_or(16000) as usize;

        let num_mel_bins = self.metadata.get_u32("audio.num_mel_bins")
            .or_else(|| self.metadata.get_u32("whisper.num_mel_bins"))
            .unwrap_or(80) as usize;

        let hidden_size = self.metadata.get_u32("audio.hidden_size")
            .or_else(|| self.metadata.get_u32("whisper.hidden_size"))
            .unwrap_or(1024) as usize;

        let num_layers = self.metadata.get_u32("audio.num_layers")
            .or_else(|| self.metadata.get_u32("whisper.num_layers"))
            .unwrap_or(24) as usize;

        let num_heads = self.metadata.get_u32("audio.num_heads")
            .or_else(|| self.metadata.get_u32("whisper.num_heads"))
            .unwrap_or(16) as usize;

        Some(AudioConfigGGUF {
            sample_rate,
            num_mel_bins,
            hidden_size,
            num_layers,
            num_heads,
        })
    }

    /// 获取视觉编码器配置
    ///
    /// 从元数据中提取视觉编码器配置
    ///
    /// # Returns
    /// 视觉配置（如果存在）
    pub fn get_vision_config(&self) -> Option<VisionConfigGGUF> {
        if !self.has_vision_tensors() {
            return None;
        }

        let image_size = self.metadata.get_u32("vision.image_size")
            .or_else(|| self.metadata.get_u32("minicpm.vision.image_size"))
            .unwrap_or(448) as usize;

        let patch_size = self.metadata.get_u32("vision.patch_size")
            .or_else(|| self.metadata.get_u32("minicpm.vision.patch_size"))
            .unwrap_or(14) as usize;

        let hidden_size = self.metadata.get_u32("vision.hidden_size")
            .or_else(|| self.metadata.get_u32("minicpm.vision.hidden_size"))
            .unwrap_or(1152) as usize;

        let num_layers = self.metadata.get_u32("vision.num_layers")
            .or_else(|| self.metadata.get_u32("minicpm.vision.num_layers"))
            .unwrap_or(27) as usize;

        let num_heads = self.metadata.get_u32("vision.num_heads")
            .or_else(|| self.metadata.get_u32("minicpm.vision.num_heads"))
            .unwrap_or(16) as usize;

        let intermediate_size = self.metadata.get_u32("vision.intermediate_size")
            .or_else(|| self.metadata.get_u32("minicpm.vision.intermediate_size"))
            .unwrap_or(4304) as usize;

        Some(VisionConfigGGUF {
            image_size,
            patch_size,
            hidden_size,
            num_layers,
            num_heads,
            intermediate_size,
        })
    }

    /// 解析 GGUF 文件
    ///
    /// 从内存映射解析 GGUF 文件结构
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    ///
    /// # Returns
    /// 解析后的 GGUF 文件实例
    fn parse(mmap: Mmap) -> Result<Self> {
        // 检查文件是否为空
        if mmap.len() < 24 {
            return Err(anyhow!("File too small to be a valid GGUF file"));
        }

        // 检查文件大小是否超过 usize 上限（32位系统兼容性）
        if mmap.len() > usize::MAX {
            return Err(anyhow!("File too large to map on this platform ({} bytes)", mmap.len()));
        }

        let mut offset: usize = 0;

        // 解析魔数（小端序）
        let magic = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
        offset += 4;
        
        if magic != GGUF_MAGIC {
            return Err(anyhow!("Invalid GGUF magic number: expected {:#x}, got {:#x}", GGUF_MAGIC, magic));
        }

        // 解析版本号
        let version = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?);
        offset += 4;

        // 解析张量数量
        let tensor_count = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
        offset += 8;

        // 解析元数据数量
        let metadata_kv_count = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
        offset += 8;

        let header = GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
        };

        // 解析元数据
        let (metadata, new_offset) = Self::parse_metadata(&mmap, offset, metadata_kv_count)?;
        offset = new_offset;

        // 解析张量信息
        let (tensors, _) = Self::parse_tensor_info(&mmap, offset, tensor_count)?;

        Ok(Self {
            header,
            metadata,
            tensors,
            mmap,
        })
    }

    /// 解析元数据
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移
    /// - `count`: 元数据数量
    ///
    /// # Returns
    /// 元数据和新的偏移
    fn parse_metadata(mmap: &[u8], mut offset: usize, count: u64) -> Result<(GgufMetadata, usize)> {
        let mut kv = HashMap::new();

        for _ in 0..count {
            // 解析键名
            let key = Self::parse_string(mmap, &mut offset)?;
            
            // 解析值
            let value = Self::parse_value(mmap, &mut offset)?;
            
            kv.insert(key, value);
        }

        Ok((GgufMetadata { kv }, offset))
    }

    /// 解析字符串
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移（可变引用）
    ///
    /// # Returns
    /// 解析的字符串
    fn parse_string(mmap: &[u8], offset: &mut usize) -> Result<String> {
        // 边界检查
        if *offset + 8 > mmap.len() {
            return Err(anyhow!("Unexpected EOF while reading string length"));
        }
        
        let len = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?) as usize;
        *offset += 8;
        
        // 限制最大字符串长度（防止 DoS）
        const MAX_STRING_LEN: usize = 1024 * 1024; // 1MB
        if len > MAX_STRING_LEN {
            return Err(anyhow!("String too long: {} bytes (max {})", len, MAX_STRING_LEN));
        }
        
        // 边界检查
        if *offset + len > mmap.len() {
            return Err(anyhow!("Unexpected EOF while reading string data"));
        }
        
        let bytes = &mmap[*offset..*offset + len];
        *offset += len;
        
        String::from_utf8(bytes.to_vec()).map_err(|e| anyhow!("Invalid UTF-8 string: {}", e))
    }

    /// 解析值
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移（可变引用）
    ///
    /// # Returns
    /// 解析的值
    fn parse_value(mmap: &[u8], offset: &mut usize) -> Result<GgufValue> {
        let value_type = mmap[*offset];
        *offset += 1;

        let value_type = GgufValueType::try_from(value_type as u32)
            .map_err(|_| anyhow!("Unknown value type: {}", value_type))?;

        match value_type {
            GgufValueType::UInt8 => {
                let v = mmap[*offset];
                *offset += 1;
                Ok(GgufValue::UInt8(v))
            }
            GgufValueType::Int8 => {
                let v = mmap[*offset] as i8;
                *offset += 1;
                Ok(GgufValue::Int8(v))
            }
            GgufValueType::UInt16 => {
                let v = u16::from_le_bytes(mmap[*offset..*offset + 2].try_into()?);
                *offset += 2;
                Ok(GgufValue::UInt16(v))
            }
            GgufValueType::Int16 => {
                let v = i16::from_le_bytes(mmap[*offset..*offset + 2].try_into()?);
                *offset += 2;
                Ok(GgufValue::Int16(v))
            }
            GgufValueType::UInt32 => {
                let v = u32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::UInt32(v))
            }
            GgufValueType::Int32 => {
                let v = i32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::Int32(v))
            }
            GgufValueType::Float32 => {
                let v = f32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::Float32(v))
            }
            GgufValueType::Bool => {
                let v = mmap[*offset] != 0;
                *offset += 1;
                Ok(GgufValue::Bool(v))
            }
            GgufValueType::String => {
                let v = Self::parse_string(mmap, offset)?;
                Ok(GgufValue::String(v))
            }
            GgufValueType::Array => {
                let element_type = mmap[*offset];
                *offset += 1;
                
                let count = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?) as usize;
                *offset += 8;

                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    // 数组元素不带类型前缀
                    arr.push(Self::parse_value_of_type(mmap, offset, element_type)?);
                }
                
                Ok(GgufValue::Array(arr))
            }
            GgufValueType::UInt64 => {
                let v = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::UInt64(v))
            }
            GgufValueType::Int64 => {
                let v = i64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::Int64(v))
            }
            GgufValueType::Float64 => {
                let v = f64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::Float64(v))
            }
        }
    }

    /// 解析指定类型的值
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移（可变引用）
    /// - `value_type`: 值类型编号
    ///
    /// # Returns
    /// 解析的值
    fn parse_value_of_type(mmap: &[u8], offset: &mut usize, value_type: u8) -> Result<GgufValue> {
        let value_type = GgufValueType::try_from(value_type as u32)
            .map_err(|_| anyhow!("Unknown value type: {}", value_type))?;

        match value_type {
            GgufValueType::UInt8 => {
                let v = mmap[*offset];
                *offset += 1;
                Ok(GgufValue::UInt8(v))
            }
            GgufValueType::Int8 => {
                let v = mmap[*offset] as i8;
                *offset += 1;
                Ok(GgufValue::Int8(v))
            }
            GgufValueType::UInt16 => {
                let v = u16::from_le_bytes(mmap[*offset..*offset + 2].try_into()?);
                *offset += 2;
                Ok(GgufValue::UInt16(v))
            }
            GgufValueType::Int16 => {
                let v = i16::from_le_bytes(mmap[*offset..*offset + 2].try_into()?);
                *offset += 2;
                Ok(GgufValue::Int16(v))
            }
            GgufValueType::UInt32 => {
                let v = u32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::UInt32(v))
            }
            GgufValueType::Int32 => {
                let v = i32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::Int32(v))
            }
            GgufValueType::Float32 => {
                let v = f32::from_le_bytes(mmap[*offset..*offset + 4].try_into()?);
                *offset += 4;
                Ok(GgufValue::Float32(v))
            }
            GgufValueType::Bool => {
                let v = mmap[*offset] != 0;
                *offset += 1;
                Ok(GgufValue::Bool(v))
            }
            GgufValueType::String => {
                let v = Self::parse_string(mmap, offset)?;
                Ok(GgufValue::String(v))
            }
            GgufValueType::UInt64 => {
                let v = u64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::UInt64(v))
            }
            GgufValueType::Int64 => {
                let v = i64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::Int64(v))
            }
            GgufValueType::Float64 => {
                let v = f64::from_le_bytes(mmap[*offset..*offset + 8].try_into()?);
                *offset += 8;
                Ok(GgufValue::Float64(v))
            }
            GgufValueType::Array => {
                Err(anyhow!("Nested arrays are not supported by GGUF specification"))
            }
        }
    }

    /// 解析张量信息
    ///
    /// # Parameters
    /// - `mmap`: 内存映射
    /// - `offset`: 当前偏移
    /// - `count`: 张量数量
    ///
    /// # Returns
    /// 张量映射和新的偏移
    fn parse_tensor_info(mmap: &[u8], mut offset: usize, count: u64) -> Result<(HashMap<String, GgufTensor>, usize)> {
        let mut tensors = HashMap::new();

        for _ in 0..count {
            // 解析张量名称
            let name = Self::parse_string(mmap, &mut offset)?;
            
            // 解析维度数量
            let n_dims = u32::from_le_bytes(mmap[offset..offset + 4].try_into()?) as usize;
            offset += 4;

            // GGUF 规范通常维度数 <= 4，设置上限防止恶意文件
            const MAX_DIMS: usize = 8;
            if n_dims > MAX_DIMS {
                return Err(anyhow!("Too many dimensions: {} (max {})", n_dims, MAX_DIMS));
            }
            
            // 解析各维度大小
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let dim = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?) as usize;
                offset += 8;
                dims.push(dim);
            }
            
            // 解析张量类型
            let tensor_type_byte = mmap[offset];
            offset += 1;
            
            let tensor_type = GgufTensorType::from_u8(tensor_type_byte)
                .ok_or_else(|| anyhow!("Unknown tensor type: {}", tensor_type_byte))?;
            
            // 解析数据偏移
            let tensor_offset = u64::from_le_bytes(mmap[offset..offset + 8].try_into()?);
            offset += 8;

            let tensor = GgufTensor {
                name: name.clone(),
                dims,
                tensor_type,
                offset: tensor_offset,
            };

            tensors.insert(name, tensor);
        }

        Ok((tensors, offset))
    }

    /// 获取张量数据
    ///
    /// 通过名称获取张量的原始字节数据
    /// 获取张量原始数据
    ///
    /// # Parameters
    /// - `name`: 张量名称
    ///
    /// # Returns
    /// 张量数据的字节切片
    ///
    /// # Note
    /// `tensor.offset` 是从文件起始处的绝对偏移量（GGUF 规范）
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensors.get(name)?;
        // tensor.offset 是绝对偏移（GGUF 规范）
        let start = tensor.offset as usize;
        let end = start + tensor.data_size();
        
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// 通过张量引用获取数据
    ///
    /// 将张量数据反序列化为 f32 向量
    ///
    /// # Parameters
    /// - `tensor`: 张量信息引用
    ///
    /// # Returns
    /// 反序列化后的 f32 向量
    ///
    /// # Note
    /// `tensor.offset` 是从文件起始处的绝对偏移量（GGUF 规范）
    /// 
    /// # Safety
    /// 内存映射通常满足对齐要求，但某些平台可能需要额外处理
    pub fn get_tensor_data_by_ref(&self, tensor: &GgufTensor) -> Result<Vec<f32>> {
        // tensor.offset 是绝对偏移（GGUF 规范）
        let start = tensor.offset as usize;
        let end = start + tensor.data_size();

        if end > self.mmap.len() {
            return Err(anyhow!("Tensor data out of bounds"));
        }

        let data = &self.mmap[start..end];
        let num_elements = tensor.dims.iter().product::<usize>();

        match tensor.tensor_type {
            GgufTensorType::F32 => {
                // 检查对齐：f32 需要 4 字节对齐
                debug_assert_eq!(
                    data.as_ptr() as usize % std::mem::align_of::<f32>(),
                    0,
                    "F32 data is not properly aligned"
                );
                let f32_slice: &[f32] = cast_slice(data);
                if f32_slice.len() != num_elements {
                    return Err(anyhow!("F32 tensor size mismatch: expected {}, got {}", 
                        num_elements, f32_slice.len()));
                }
                Ok(f32_slice.to_vec())
            }
            GgufTensorType::F16 => {
                // 检查对齐：f16 需要 2 字节对齐
                debug_assert_eq!(
                    data.as_ptr() as usize % std::mem::align_of::<half::f16>(),
                    0,
                    "F16 data is not properly aligned"
                );
                let f16_slice: &[half::f16] = cast_slice(data);
                if f16_slice.len() != num_elements {
                    return Err(anyhow!("F16 tensor size mismatch: expected {}, got {}", 
                        num_elements, f16_slice.len()));
                }
                Ok(f16_slice.iter().map(|&f| f.to_f32()).collect())
            }
            _ => {
                let result = super::quant::dequantize(data, tensor.tensor_type, num_elements);
                Ok(result)
            }
        }
    }

    /// 获取模型配置
    ///
    /// 从元数据中提取语言模型配置
    ///
    /// # Returns
    /// 模型配置
    pub fn get_model_config(&self) -> Result<ModelConfig> {
        // 首先读取架构名称，决定键名前缀
        let architecture = self.metadata.get_string("general.architecture")
            .unwrap_or("llama");
        
        // 根据架构选择键名前缀（支持多种模型架构）
        let prefix = match architecture {
            "minicpm" => "minicpm",
            "qwen2" => "qwen2",
            "gemma" => "gemma",
            "phi" | "phi3" => "phi",
            "llama" | _ => "llama",
        };
        
        let vocab_size = self.metadata.get_u64("general.vocab_size")
            .unwrap_or(DEFAULT_VOCAB_SIZE as u64) as usize;
        
        // 优先使用架构对应的前缀，然后尝试其他常见前缀
        let hidden_size = self.metadata.get_u32(&format!("{}.embedding_length", prefix))
            .or_else(|| self.metadata.get_u32("llama.embedding_length"))
            .or_else(|| self.metadata.get_u32("qwen2.embedding_length"))
            .or_else(|| self.metadata.get_u32("gemma.embedding_length"))
            .unwrap_or(DEFAULT_HIDDEN_SIZE as u32) as usize;
        
        let intermediate_size = self.metadata.get_u32(&format!("{}.feed_forward_length", prefix))
            .or_else(|| self.metadata.get_u32("llama.feed_forward_length"))
            .or_else(|| self.metadata.get_u32("qwen2.feed_forward_length"))
            .or_else(|| self.metadata.get_u32("gemma.feed_forward_length"))
            .unwrap_or(DEFAULT_INTERMEDIATE_SIZE as u32) as usize;
        
        let num_hidden_layers = self.metadata.get_u32(&format!("{}.block_count", prefix))
            .or_else(|| self.metadata.get_u32("llama.block_count"))
            .or_else(|| self.metadata.get_u32("qwen2.block_count"))
            .or_else(|| self.metadata.get_u32("gemma.block_count"))
            .unwrap_or(DEFAULT_NUM_HIDDEN_LAYERS as u32) as usize;
        
        let num_attention_heads = self.metadata.get_u32(&format!("{}.attention.head_count", prefix))
            .or_else(|| self.metadata.get_u32("llama.attention.head_count"))
            .or_else(|| self.metadata.get_u32("qwen2.attention.head_count"))
            .or_else(|| self.metadata.get_u32("gemma.attention.head_count"))
            .unwrap_or(DEFAULT_NUM_ATTENTION_HEADS as u32) as usize;
        
        let num_key_value_heads = self.metadata.get_u32(&format!("{}.attention.head_count_kv", prefix))
            .or_else(|| self.metadata.get_u32("llama.attention.head_count_kv"))
            .or_else(|| self.metadata.get_u32("qwen2.attention.head_count_kv"))
            .or_else(|| self.metadata.get_u32("gemma.attention.head_count_kv"))
            .unwrap_or(num_attention_heads as u32) as usize;
        
        let max_position_embeddings = self.metadata.get_u32(&format!("{}.context_length", prefix))
            .or_else(|| self.metadata.get_u32("llama.context_length"))
            .or_else(|| self.metadata.get_u32("qwen2.context_length"))
            .or_else(|| self.metadata.get_u32("gemma.context_length"))
            .unwrap_or(DEFAULT_MAX_POSITION_EMBEDDINGS as u32) as usize;
        
        let rms_norm_eps = self.metadata.get_f32(&format!("{}.attention.layer_norm_rms_epsilon", prefix))
            .or_else(|| self.metadata.get_f32("llama.attention.layer_norm_rms_epsilon"))
            .or_else(|| self.metadata.get_f32("qwen2.attention.layer_norm_rms_epsilon"))
            .or_else(|| self.metadata.get_f32("gemma.attention.layer_norm_rms_epsilon"))
            .unwrap_or(DEFAULT_RMS_NORM_EPS);
        
        let rope_theta = self.metadata.get_f32(&format!("{}.rope.freq_base", prefix))
            .or_else(|| self.metadata.get_f32("llama.rope.freq_base"))
            .or_else(|| self.metadata.get_f32("qwen2.rope.freq_base"))
            .or_else(|| self.metadata.get_f32("gemma.rope.freq_base"))
            .unwrap_or(DEFAULT_ROPE_THETA);

        Ok(ModelConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
        })
    }

    /// 通过名称获取张量数据
    ///
    /// # Parameters
    /// - `name`: 张量名称
    ///
    /// # Returns
    /// 反序列化后的 f32 向量
    pub fn get_tensor_data_by_name(&self, name: &str) -> Result<Vec<f32>> {
        let tensor = self.tensors.get(name)
            .ok_or_else(|| anyhow!("Tensor not found: {}", name))?;
        self.get_tensor_data_by_ref(tensor)
    }

    /// 获取 TTS 配置
    ///
    /// 从元数据中提取语音合成模型配置
    ///
    /// # Returns
    /// TTS 配置（如果存在）
    pub fn get_tts_config(&self) -> Option<TTSConfigGGUF> {
        if !self.has_tts_tensors() {
            return None;
        }

        let sample_rate = self.metadata.get_u32("tts.sample_rate")
            .unwrap_or(24000) as usize;

        let hidden_size = self.metadata.get_u32("tts.hidden_size")
            .unwrap_or(1024) as usize;

        let num_flow_layers = self.metadata.get_u32("tts.num_flow_layers")
            .unwrap_or(12) as usize;

        Some(TTSConfigGGUF {
            sample_rate,
            hidden_size,
            num_flow_layers,
        })
    }
}

// ============================================================================
// 类型转换实现
// ============================================================================

impl TryFrom<u32> for GgufValueType {
    type Error = anyhow::Error;

    fn try_from(v: u32) -> Result<Self> {
        match v {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(anyhow!("Unknown value type: {}", v)),
        }
    }
}

impl TryFrom<u8> for GgufTensorType {
    type Error = anyhow::Error;

    fn try_from(v: u8) -> Result<Self> {
        Self::from_u8(v).ok_or_else(|| anyhow!("Unknown tensor type: {}", v))
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_magic() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
    }

    #[test]
    fn test_tensor_type_from_u8() {
        assert_eq!(GgufTensorType::from_u8(0), Some(GgufTensorType::F32));
        assert_eq!(GgufTensorType::from_u8(1), Some(GgufTensorType::F16));
        assert_eq!(GgufTensorType::from_u8(2), Some(GgufTensorType::Q4_0));
        assert_eq!(GgufTensorType::from_u8(3), Some(GgufTensorType::Q4_1));
        assert_eq!(GgufTensorType::from_u8(12), Some(GgufTensorType::Q4K));
    }

    #[test]
    fn test_value_type_try_from() {
        assert!(matches!(GgufValueType::try_from(0u32), Ok(GgufValueType::UInt8)));
        assert!(matches!(GgufValueType::try_from(6u32), Ok(GgufValueType::Float32)));
        assert!(matches!(GgufValueType::try_from(8u32), Ok(GgufValueType::String)));
        assert!(GgufValueType::try_from(100u32).is_err());
    }

    #[test]
    fn test_model_config_defaults() {
        assert_eq!(DEFAULT_VOCAB_SIZE, 152064);
        assert_eq!(DEFAULT_HIDDEN_SIZE, 3584);
        assert_eq!(DEFAULT_NUM_HIDDEN_LAYERS, 28);
        assert_eq!(DEFAULT_NUM_ATTENTION_HEADS, 28);
        assert_eq!(DEFAULT_INTERMEDIATE_SIZE, 18944);
        assert_eq!(DEFAULT_MAX_POSITION_EMBEDDINGS, 131072);
        assert!((DEFAULT_ROPE_THETA - 1000000.0).abs() < 1e-6);
        assert!((DEFAULT_RMS_NORM_EPS - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_type_block_sizes() {
        assert_eq!(GgufTensorType::F32.block_size(), 1);
        assert_eq!(GgufTensorType::F16.block_size(), 1);
        assert_eq!(GgufTensorType::Q4_0.block_size(), 32);
        assert_eq!(GgufTensorType::Q4_1.block_size(), 32);
        assert_eq!(GgufTensorType::Q8_0.block_size(), 32);
        assert_eq!(GgufTensorType::Q4K.block_size(), 256);
        assert_eq!(GgufTensorType::Q5K.block_size(), 256);
        assert_eq!(GgufTensorType::Q6K.block_size(), 256);
    }

    #[test]
    fn test_vision_config_gguf() {
        let config = VisionConfigGGUF {
            image_size: 448,
            patch_size: 14,
            hidden_size: 1152,
            num_layers: 27,
            num_heads: 16,
            intermediate_size: 4304,
        };
        assert_eq!(config.image_size, 448);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.hidden_size, 1152);
    }

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig {
            vocab_size: 152064,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 28,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
        };
        assert_eq!(config.vocab_size, 152064);
        assert_eq!(config.hidden_size, 3584);
    }

    #[test]
    fn test_gguf_tensor_creation() {
        let tensor = GgufTensor {
            name: "test_tensor".to_string(),
            dims: vec![10, 20],
            tensor_type: GgufTensorType::F32,
            offset: 0,
        };
        assert_eq!(tensor.name, "test_tensor");
        assert_eq!(tensor.dims, vec![10, 20]);
        assert_eq!(tensor.tensor_type, GgufTensorType::F32);
    }

    #[test]
    fn test_tensor_type_from_u8_all() {
        // 测试所有有效类型映射
        let type_mappings: [(u8, GgufTensorType); 24] = [
            (0, GgufTensorType::F32),
            (1, GgufTensorType::F16),
            (2, GgufTensorType::Q4_0),
            (3, GgufTensorType::Q4_1),
            (4, GgufTensorType::Q4_2),
            (5, GgufTensorType::Q4_3),
            (7, GgufTensorType::Q8_0),
            (8, GgufTensorType::Q8_1),
            (10, GgufTensorType::Q2K),
            (11, GgufTensorType::Q3K),
            (12, GgufTensorType::Q4K),
            (13, GgufTensorType::Q5K),
            (14, GgufTensorType::Q6K),
            (15, GgufTensorType::Q8K),
            (16, GgufTensorType::Iq2Xxs),
            (17, GgufTensorType::Iq2Xs),
            (18, GgufTensorType::Iq3Xxs),
            (19, GgufTensorType::Iq1S),
            (20, GgufTensorType::Iq4Nl),
            (21, GgufTensorType::Iq3S),
            (22, GgufTensorType::Iq2S),
            (23, GgufTensorType::Iq4Xs),
            (24, GgufTensorType::I8),
            (25, GgufTensorType::I16),
        ];

        for (value, expected_type) in type_mappings {
            assert_eq!(
                GgufTensorType::from_u8(value),
                Some(expected_type),
                "Failed for value {}",
                value
            );
        }

        // 测试更多整数类型
        assert_eq!(GgufTensorType::from_u8(26), Some(GgufTensorType::I32));
        assert_eq!(GgufTensorType::from_u8(27), Some(GgufTensorType::I64));
        assert_eq!(GgufTensorType::from_u8(28), Some(GgufTensorType::F64));
        assert_eq!(GgufTensorType::from_u8(29), Some(GgufTensorType::BF16));

        // 测试无效值
        assert_eq!(GgufTensorType::from_u8(6), None);  // 保留值
        assert_eq!(GgufTensorType::from_u8(9), None);  // 未定义
        assert_eq!(GgufTensorType::from_u8(100), None);
    }

    #[test]
    fn test_bytes_per_block_all() {
        // 测试非量化类型
        assert_eq!(GgufTensorType::F32.bytes_per_block(), 4);
        assert_eq!(GgufTensorType::F16.bytes_per_block(), 2);
        assert_eq!(GgufTensorType::BF16.bytes_per_block(), 2);
        assert_eq!(GgufTensorType::F64.bytes_per_block(), 8);
        assert_eq!(GgufTensorType::I8.bytes_per_block(), 1);
        assert_eq!(GgufTensorType::I16.bytes_per_block(), 2);
        assert_eq!(GgufTensorType::I32.bytes_per_block(), 4);
        assert_eq!(GgufTensorType::I64.bytes_per_block(), 8);

        // 测试量化类型
        assert_eq!(GgufTensorType::Q4_0.bytes_per_block(), 18);
        assert_eq!(GgufTensorType::Q4_1.bytes_per_block(), 20);
        assert_eq!(GgufTensorType::Q4_2.bytes_per_block(), 18);
        assert_eq!(GgufTensorType::Q4_3.bytes_per_block(), 20);
        assert_eq!(GgufTensorType::Q8_0.bytes_per_block(), 34);
        assert_eq!(GgufTensorType::Q8_1.bytes_per_block(), 36);
    }

    #[test]
    fn test_element_size() {
        // 非量化类型
        assert_eq!(GgufTensorType::F32.element_size(), 4);
        assert_eq!(GgufTensorType::F16.element_size(), 2);
        assert_eq!(GgufTensorType::I8.element_size(), 1);
        assert_eq!(GgufTensorType::I16.element_size(), 2);
        assert_eq!(GgufTensorType::I32.element_size(), 4);
        assert_eq!(GgufTensorType::I64.element_size(), 8);

        // 量化类型（通过 bytes_per_block / block_size 计算）
        assert_eq!(GgufTensorType::Q4_0.element_size(), 18 / 32);
        assert_eq!(GgufTensorType::Q8_0.element_size(), 34 / 32);
    }

    #[test]
    fn test_tensor_type_try_from_u8() {
        use std::convert::TryFrom;
        
        assert!(matches!(GgufTensorType::try_from(0u8), Ok(GgufTensorType::F32)));
        assert!(matches!(GgufTensorType::try_from(1u8), Ok(GgufTensorType::F16)));
        assert!(matches!(GgufTensorType::try_from(2u8), Ok(GgufTensorType::Q4_0)));
        assert!(matches!(GgufTensorType::try_from(7u8), Ok(GgufTensorType::Q8_0)));
        assert!(GgufTensorType::try_from(6u8).is_err());  // 保留值
        assert!(GgufTensorType::try_from(100u8).is_err()); // 无效值
    }
}

// ============================================================================
// 集成测试 - 最小 GGUF 文件解析
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// 构建最小化的 GGUF 文件字节
    fn build_minimal_gguf() -> Vec<u8> {
        let mut buf = Vec::new();

        // 魔数 "GGUF"
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());

        // 版本号 (3)
        buf.extend_from_slice(&3u32.to_le_bytes());

        // 张量数量 (1)
        buf.extend_from_slice(&1u64.to_le_bytes());

        // 元数据数量 (3)
        buf.extend_from_slice(&3u64.to_le_bytes());

        // 元数据 1: general.architecture = "llama"
        let key1 = "general.architecture";
        buf.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        buf.extend_from_slice(key1.as_bytes());
        buf.push(8); // String type
        let val1 = "llama";
        buf.extend_from_slice(&(val1.len() as u64).to_le_bytes());
        buf.extend_from_slice(val1.as_bytes());

        // 元数据 2: llama.embedding_length = 4096
        let key2 = "llama.embedding_length";
        buf.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        buf.extend_from_slice(key2.as_bytes());
        buf.push(4); // UInt32 type
        buf.extend_from_slice(&4096u32.to_le_bytes());

        // 元数据 3: llama.block_count = 32
        let key3 = "llama.block_count";
        buf.extend_from_slice(&(key3.len() as u64).to_le_bytes());
        buf.extend_from_slice(key3.as_bytes());
        buf.push(4); // UInt32 type
        buf.extend_from_slice(&32u32.to_le_bytes());

        // 张量信息: "model.token_embd.weight" [4096] F32
        let tensor_name = "model.token_embd.weight";
        buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(tensor_name.as_bytes());

        // 维度数量
        buf.extend_from_slice(&1u32.to_le_bytes());
        // 维度大小
        buf.extend_from_slice(&4096u64.to_le_bytes());
        // 张量类型 F32
        buf.push(0);
        // 偏移量 (从文件头开始计算，需要对齐)
        let tensor_offset = buf.len() as u64 + 8; // +8 for offset field
        // 对齐到 32 字节
        let aligned_offset = (tensor_offset + 31) / 32 * 32;
        buf.extend_from_slice(&aligned_offset.to_le_bytes());

        // 填充到对齐位置
        let padding = aligned_offset as usize - buf.len();
        buf.extend(std::iter::repeat(0u8).take(padding));

        // 张量数据: 4096 个 f32 (全部为 0.5)
        for _ in 0..4096 {
            buf.extend_from_slice(&0.5f32.to_le_bytes());
        }

        buf
    }

    #[test]
    fn test_gguf_file_parsing() {
        let gguf_bytes = build_minimal_gguf();

        // 写入临时文件
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&gguf_bytes).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        // 解析文件
        let path = temp_file.path();
        let gguf = GgufFile::open(path).expect("Failed to parse GGUF file");

        // 验证文件头
        assert_eq!(gguf.header.version, 3);
        assert_eq!(gguf.header.tensor_count, 1);
        assert_eq!(gguf.header.metadata_kv_count, 3);

        // 验证元数据
        assert_eq!(gguf.metadata.get_string("general.architecture"), Some("llama"));
        assert_eq!(gguf.metadata.get_u32("llama.embedding_length"), Some(4096));
        assert_eq!(gguf.metadata.get_u32("llama.block_count"), Some(32));

        // 验证张量信息
        assert!(gguf.tensors.contains_key("model.token_embd.weight"));
        let tensor = gguf.tensors.get("model.token_embd.weight").unwrap();
        assert_eq!(tensor.dims, vec![4096]);
        assert_eq!(tensor.tensor_type, GgufTensorType::F32);
    }

    #[test]
    fn test_gguf_tensor_data_access() {
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&gguf_bytes).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 获取张量数据
        let data = gguf.get_tensor_data("model.token_embd.weight")
            .expect("Failed to get tensor data");

        // 验证数据大小 (4096 * 4 bytes)
        assert_eq!(data.len(), 4096 * 4);

        // 验证数据内容
        let f32_slice: &[f32] = bytemuck::cast_slice(data);
        assert_eq!(f32_slice.len(), 4096);
        assert!((f32_slice[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gguf_model_config_extraction() {
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&gguf_bytes).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 提取模型配置
        let config = gguf.get_model_config().expect("Failed to get model config");

        // 验证配置值
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
    }

    #[test]
    fn test_gguf_invalid_magic() {
        let mut buf = Vec::new();
        // 错误的魔数
        buf.extend_from_slice(&0x12345678u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&buf).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let result = GgufFile::open(temp_file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid GGUF magic"));
    }

    #[test]
    fn test_gguf_file_too_small() {
        let buf = vec![0u8; 10]; // 太小

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&buf).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let result = GgufFile::open(temp_file.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_gguf_tensor_iteration() {
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&gguf_bytes).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 测试 LLM 张量迭代器
        let llm_count = gguf.iter_llm_tensors().count();
        assert_eq!(llm_count, 1);

        // 测试视觉张量迭代器（应该为空）
        let vision_count = gguf.iter_vision_tensors().count();
        assert_eq!(vision_count, 0);

        // 测试 has_* 方法
        assert!(gguf.has_llm_tensors());
        assert!(!gguf.has_vision_tensors());
        assert!(!gguf.has_audio_encoder_tensors());
    }

    #[test]
    fn test_gguf_header_parse_valid() {
        // 有效GGUF头解析
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&gguf_bytes).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 测试版本号、架构等字段
        assert_eq!(gguf.header.version, 3);
        assert_eq!(gguf.header.tensor_count, 1);
        assert_eq!(gguf.header.metadata_kv_count, 3);

        // 验证魔数
        assert_eq!(GGUF_MAGIC, 0x46554747);
    }

    #[test]
    fn test_gguf_tensor_info_extraction() {
        // 张量信息提取
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&gguf_bytes).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 验证名称、维度、类型等
        let tensor = gguf.tensors.get("model.token_embd.weight")
            .expect("Tensor should exist");
        
        assert_eq!(tensor.name, "model.token_embd.weight");
        assert_eq!(tensor.dims, vec![4096]);
        assert_eq!(tensor.tensor_type, GgufTensorType::F32);
        
        // 验证张量计算方法
        assert_eq!(tensor.num_elements(), 4096);
        assert_eq!(tensor.data_size(), 4096 * 4); // F32 = 4 bytes per element
        
        // 测试get_tensor_data方法
        let data = gguf.get_tensor_data("model.token_embd.weight")
            .expect("Should get tensor data");
        assert!(!data.is_empty());
        assert_eq!(data.len(), 16384); // 4096 * 4 bytes
    }

    #[test]
    fn test_gguf_invalid_file_error() {
        // 无效文件错误处理 - 不存在的路径
        let result = GgufFile::open("/nonexistent/file.gguf");
        assert!(result.is_err(), "Should fail for nonexistent file");
        
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.to_lowercase().contains("no such file") || 
               err_msg.to_lowercase().contains("not found") ||
               err_msg.to_lowercase().contains("os error"),
               "Error should indicate file not found: {}", err_msg);
    }

    #[test]
    fn test_gguf_truncated_file_handling() {
        // 截断文件处理
        let temp_dir = std::env::temp_dir();
        let truncated_path = temp_dir.join("truncated.gguf");

        // 创建截断的文件(只有部分头部)
        std::fs::write(&truncated_path, b"GGUF partial data...").ok();

        let result = GgufFile::open(&truncated_path);
        assert!(result.is_err(), "Should fail for truncated file");
        
        let err_msg = result.unwrap_err().to_string();
        // 应该报告文件太小或解析错误
        assert!(err_msg.to_lowercase().contains("too small") ||
               err_msg.to_lowercase().contains("unexpected eof") ||
               err_msg.contains("Invalid GGUF magic"),
               "Error should indicate truncation issue: {}", err_msg);

        std::fs::remove_file(truncated_path).ok();
    }

    #[test]
    fn test_gguf_invalid_magic_number() {
        // 无效魔数测试
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x12345678u32.to_le_bytes()); // 错误的魔数
        buf.extend_from_slice(&3u32.to_le_bytes());           // 版本号
        buf.extend_from_slice(&0u64.to_le_bytes());            // 张量数量
        buf.extend_from_slice(&0u64.to_le_bytes());            // 元数据数量

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&buf).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let result = GgufFile::open(temp_file.path());
        assert!(result.is_err(), "Should fail for invalid magic number");
        
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid GGUF magic"),
            "Error should mention invalid magic: {}", err_msg);
    }

    #[test]
    fn test_gguf_metadata_accessors() {
        // 元数据访问器测试
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&gguf_bytes).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 测试字符串访问器
        let arch = gguf.metadata.get_string("general.architecture");
        assert_eq!(arch, Some("llama"));

        // 测试u32访问器
        let embed_len = gguf.metadata.get_u32("llama.embedding_length");
        assert_eq!(embed_len, Some(4096));

        // 测试不存在的键
        let missing = gguf.metadata.get_string("nonexistent.key");
        assert_eq!(missing, None);

        let missing_u32 = gguf.metadata.get_u32("nonexistent.u32");
        assert_eq!(missing_u32, None);
    }

    #[test]
    fn test_gguf_value_type_variants() {
        // 值类型枚举变体测试
        use GgufValue::*;
        
        // 创建各种类型的值
        let values = vec![
            UInt8(42),
            Int8(-10),
            UInt16(1000),
            Int16(-500),
            UInt32(12345),
            Int32(-99999),
            Float32(3.14),
            Bool(true),
            String("hello".to_string()),
            Array(vec![UInt8(1), UInt8(2), UInt8(3)]),
            UInt64(123456789),
            Int64(-987654321),
            Float64(2.71828),
        ];
        
        // 所有值都应该能创建和克隆
        for value in &values {
            let _clone = value.clone();
        }
    }

    #[test]
    fn test_gguf_tensor_type_comprehensive() {
        // 张量类型综合测试
        // 测试所有类型的block_size和bytes_per_block的一致性
        let types_to_test = [
            (GgufTensorType::F32, 1, 4),
            (GgufTensorType::F16, 1, 2),
            (GgufTensorType::BF16, 1, 2),
            (GgufTensorType::F64, 1, 8),
            (GgufTensorType::I8, 1, 1),
            (GgufTensorType::I16, 1, 2),
            (GgufTensorType::I32, 1, 4),
            (GgufTensorType::I64, 1, 8),
            (GgufTensorType::Q4_0, 32, 18),
            (GgufTensorType::Q4_1, 32, 20),
            (GgufTensorType::Q8_0, 32, 34),
            (GgufTensorType::Q8_1, 32, 36),
            (GgufTensorType::Q2K, 256, 68),
            (GgufTensorType::Q3K, 256, 162),
            (GgufTensorType::Q4K, 256, 144),
            (GgufTensorType::Q5K, 256, 176),
            (GgufTensorType::Q6K, 256, 210),
            (GgufTensorType::Q8K, 256, 258),
        ];

        for (tt, expected_block_size, expected_bytes) in types_to_test {
            assert_eq!(tt.block_size(), expected_block_size,
                "Block size mismatch for {:?}", tt);
            assert_eq!(tt.bytes_per_block(), expected_bytes,
                "Bytes per block mismatch for {:?}", tt);
            
            // element_size应该是bytes_per_block / block_size
            if expected_block_size > 0 {
                let expected_element_size = expected_bytes / expected_block_size;
                assert_eq!(tt.element_size(), expected_element_size,
                    "Element size mismatch for {:?}", tt);
            }
        }
    }

    #[test]
    fn test_gguf_multimodal_tensor_detection() {
        // 多模态张量检测测试
        let gguf_bytes = build_minimal_gguf();

        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(&gguf_bytes).expect("Failed to write temp file");
        temp_file.flush().expect("Failed to flush");

        let gguf = GgufFile::open(temp_file.path()).expect("Failed to parse GGUF file");

        // 最小GGUF文件只有LLM张量,没有多模态张量
        assert!(!gguf.has_vision_tensors());
        assert!(!gguf.has_resampler_tensors());
        assert!(!gguf.has_audio_encoder_tensors());
        assert!(!gguf.has_tts_tensors());

        // 对应的配置应该返回None
        assert!(gguf.get_vision_config().is_none());
        assert!(gguf.get_audio_config().is_none());
        assert!(gguf.get_tts_config().is_none());

        // 张量集合也应该为空
        assert!(gguf.get_vision_tensors().is_empty());
        assert!(gguf.get_resampler_tensors().is_empty());
        assert!(gguf.get_audio_encoder_tensors().is_empty());
        assert!(gguf.get_tts_tensors().is_empty());
    }
}
