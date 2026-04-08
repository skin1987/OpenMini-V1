//! 分词器模块
//!
//! 本模块实现文本的编码和解码功能，包括：
//! - BPE (Byte-Pair Encoding) 分词算法
//! - 从 GGUF 文件加载词表
//! - 从 HuggingFace tokenizer.json 加载词表
//! - 文本编码（字符串 → Token ID）
//! - 文本解码（Token ID → 字符串）
//! - 特殊 Token 处理（BOS/EOS/PAD/UNK）
//!
//! # 特殊 Token
//!
//! - `<image>`: 图像标记
//! - `<im_start>`: 图像块开始标记
//! - `<im_end>`: 图像块结束标记
//! - `<im_patch>`: 图像块标记

#![allow(dead_code)]

// ============================================================================
// 标准库和外部依赖导入
// ============================================================================

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::gguf::{GgufFile, GgufValue};

// ============================================================================
// 常量定义
// ============================================================================

/// 默认词表大小
pub const VOCAB_SIZE: usize = 152064;

/// 未知 Token ID (用于未登录词表的字符)
pub const UNK_TOKEN_ID: u32 = 0;
/// 填充 Token ID (用于序列对齐)
pub const PAD_TOKEN_ID: u32 = 151642;
/// 结束序列 Token ID
pub const EOS_TOKEN_ID: u32 = 151643;
/// 开始序列 Token ID
pub const BOS_TOKEN_ID: u32 = 151644;

/// 未知 Token 字符串
pub const UNK_TOKEN: &str = "<|unk|>";
/// 填充 Token 字符串
pub const PAD_TOKEN: &str = "<|pad|>";
/// 结束序列 Token 字符串
pub const EOS_TOKEN: &str = "<|endoftext|>";
/// 开始序列 Token 字符串
pub const BOS_TOKEN: &str = "<|im_start|>";

/// 图像 Token 字符串
pub const IMAGE_TOKEN: &str = "<image>";
/// 图像块开始 Token 字符串
pub const IM_START_TOKEN: &str = "<im_start>";
/// 图像块结束 Token 字符串
pub const IM_END_TOKEN: &str = "<im_end>";
/// 图像块 Token 字符串
pub const IM_PATCH_TOKEN: &str = "<im_patch>";

/// 音频 Token 字符串
pub const AUDIO_TOKEN: &str = "<audio>";
/// 音频块开始 Token 字符串
pub const AUDIO_START_TOKEN: &str = "<audio_start>";
/// 音频块结束 Token 字符串
pub const AUDIO_END_TOKEN: &str = "<audio_end>";
/// 音频块 Token 字符串
pub const AUDIO_PATCH_TOKEN: &str = "<audio_patch>";

/// 视频 Token 字符串
pub const VIDEO_TOKEN: &str = "<video>";
/// 视频块开始 Token 字符串
pub const VIDEO_START_TOKEN: &str = "<video_start>";
/// 视频块结束 Token 字符串
pub const VIDEO_END_TOKEN: &str = "<video_end>";
/// 视频帧 Token 字符串
pub const VIDEO_FRAME_TOKEN: &str = "<video_frame>";

/// 图像 Token ID
pub const IMAGE_TOKEN_ID: u32 = 151645;
/// 图像块开始 Token ID
pub const IM_START_TOKEN_ID: u32 = 151646;
/// 图像块结束 Token ID
pub const IM_END_TOKEN_ID: u32 = 151647;
/// 图像块 Token ID
pub const IM_PATCH_TOKEN_ID: u32 = 151648;

/// 音频 Token ID
pub const AUDIO_TOKEN_ID: u32 = 151649;
/// 音频块开始 Token ID
pub const AUDIO_START_TOKEN_ID: u32 = 151650;
/// 音频块结束 Token ID
pub const AUDIO_END_TOKEN_ID: u32 = 151651;
/// 音频块 Token ID
pub const AUDIO_PATCH_TOKEN_ID: u32 = 151652;

/// 视频 Token ID
pub const VIDEO_TOKEN_ID: u32 = 151653;
/// 视频块开始 Token ID
pub const VIDEO_START_TOKEN_ID: u32 = 151654;
/// 视频块结束 Token ID
pub const VIDEO_END_TOKEN_ID: u32 = 151655;
/// 视频帧 Token ID
pub const VIDEO_FRAME_TOKEN_ID: u32 = 151656;

// ============================================================================
// tokenizer.json 格式定义 (HuggingFace)
// ============================================================================

/// HuggingFace tokenizer.json 根结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenizerJson {
    /// 版本
    version: String,
    /// 分词器模型
    model: Option<TokenizerModel>,
    /// 预处理器
    pre_tokenizer: Option<PreTokenizer>,
    /// 后处理器
    post_processor: Option<PostProcessor>,
    /// 解码器
    decoder: Option<TokenizerDecoder>,
    /// 添加的特殊 Token
    added_tokens: Option<Vec<AddedToken>>,
}

/// 分词器模型
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenizerModel {
    /// 模型类型 (BPE, WordPiece, Unigram)
    #[serde(rename = "type")]
    model_type: String,
    /// 词汇表 (token -> id)
    vocab: Option<HashMap<String, u32>>,
    /// 合并规则列表
    merges: Option<Vec<String>>,
    /// 未知 Token ID
    unk_id: Option<u32>,
}

/// 预处理器
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreTokenizer {
    /// 预处理器类型
    #[serde(rename = "type")]
    pre_type: String,
}

/// 后处理器
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PostProcessor {
    /// 后处理器类型
    #[serde(rename = "type")]
    post_type: String,
}

/// 解码器
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenizerDecoder {
    /// 解码器类型
    #[serde(rename = "type")]
    decoder_type: String,
}

/// 添加的 Token
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AddedToken {
    /// Token ID
    id: u32,
    /// Token 内容
    content: String,
    /// 是否为单词开头
    single_word: Option<bool>,
    /// 是否去除左侧空格
    lstrip: Option<bool>,
    /// 是否去除右侧空格
    rstrip: Option<bool>,
    /// 是否规范化
    normalized: Option<bool>,
    /// 是否为特殊 Token
    special: Option<bool>,
}

// ============================================================================
// 编码结果结构体
// ============================================================================

/// 编码结果
///
/// 包含 Token ID 序列和注意力掩码
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Encoding {
    /// Token ID 序列
    pub input_ids: Vec<u32>,
    /// 注意力掩码 (1 表示有效 Token, 0 表示 PAD)
    pub attention_mask: Vec<u32>,
}

impl Encoding {
    /// 创建新的编码结果
    pub fn new(input_ids: Vec<u32>, attention_mask: Vec<u32>) -> Self {
        Self {
            input_ids,
            attention_mask,
        }
    }

    /// 获取序列长度
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

// ============================================================================
// BPE 分词器核心实现
// ============================================================================

/// BPE 合并规则的优先级索引
///
/// 用于快速查找合并规则的优先级
#[derive(Debug, Clone)]
struct BpeMergeIndex {
    /// 合并规则: (first_token, second_token) -> 合并后的 Token ID
    merges: HashMap<(u32, u32), u32>,
    /// 合并优先级: (first_token, second_token) -> 优先级 (越小越优先)
    priorities: HashMap<(u32, u32), usize>,
}

impl BpeMergeIndex {
    fn new() -> Self {
        Self {
            merges: HashMap::new(),
            priorities: HashMap::new(),
        }
    }
}

/// 分词器结构体
///
/// 管理词表、特殊 Token 和 BPE 合并规则
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Token 字符串到 ID 的映射
    vocab: HashMap<String, u32>,
    /// Token ID 到字符串的映射
    id_to_token: HashMap<u32, String>,
    /// 词表大小
    vocab_size: usize,
    /// 开始序列 Token ID
    bos_token_id: u32,
    /// 结束序列 Token ID
    eos_token_id: u32,
    /// 填充 Token ID
    pad_token_id: u32,
    /// 未知 Token ID
    unk_token_id: u32,
    /// 特殊 Token 字符串到 ID 的映射
    special_tokens: HashMap<String, u32>,
    /// 特殊 Token ID 到字符串的映射
    special_token_ids: HashMap<u32, String>,
    /// BPE 合并规则索引
    bpe_index: BpeMergeIndex,
    /// 字节到 Unicode 字符的映射 (用于 BPE 字节编码)
    byte_encoder: HashMap<u8, char>,
    /// Unicode 字符到字节的映射
    byte_decoder: HashMap<char, u8>,
}

/// Tokenizer 是线程安全的，因为所有操作都是只读的
/// encode/decode 方法只读取内部 HashMap，不修改状态
unsafe impl Sync for Tokenizer {}
unsafe impl Send for Tokenizer {}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    /// 创建新的分词器实例
    ///
    /// 初始化默认的特殊 Token 和字节编码映射
    ///
    /// # Returns
    /// 分词器实例
    pub fn new() -> Self {
        let mut special_tokens = HashMap::new();
        special_tokens.insert(UNK_TOKEN.to_string(), UNK_TOKEN_ID);
        special_tokens.insert(PAD_TOKEN.to_string(), PAD_TOKEN_ID);
        special_tokens.insert(EOS_TOKEN.to_string(), EOS_TOKEN_ID);
        special_tokens.insert(BOS_TOKEN.to_string(), BOS_TOKEN_ID);
        special_tokens.insert(IMAGE_TOKEN.to_string(), IMAGE_TOKEN_ID);
        special_tokens.insert(IM_START_TOKEN.to_string(), IM_START_TOKEN_ID);
        special_tokens.insert(IM_END_TOKEN.to_string(), IM_END_TOKEN_ID);
        special_tokens.insert(IM_PATCH_TOKEN.to_string(), IM_PATCH_TOKEN_ID);
        special_tokens.insert(AUDIO_TOKEN.to_string(), AUDIO_TOKEN_ID);
        special_tokens.insert(AUDIO_START_TOKEN.to_string(), AUDIO_START_TOKEN_ID);
        special_tokens.insert(AUDIO_END_TOKEN.to_string(), AUDIO_END_TOKEN_ID);
        special_tokens.insert(AUDIO_PATCH_TOKEN.to_string(), AUDIO_PATCH_TOKEN_ID);
        special_tokens.insert(VIDEO_TOKEN.to_string(), VIDEO_TOKEN_ID);
        special_tokens.insert(VIDEO_START_TOKEN.to_string(), VIDEO_START_TOKEN_ID);
        special_tokens.insert(VIDEO_END_TOKEN.to_string(), VIDEO_END_TOKEN_ID);
        special_tokens.insert(VIDEO_FRAME_TOKEN.to_string(), VIDEO_FRAME_TOKEN_ID);

        let mut special_token_ids = HashMap::new();
        special_token_ids.insert(UNK_TOKEN_ID, UNK_TOKEN.to_string());
        special_token_ids.insert(PAD_TOKEN_ID, PAD_TOKEN.to_string());
        special_token_ids.insert(EOS_TOKEN_ID, EOS_TOKEN.to_string());
        special_token_ids.insert(BOS_TOKEN_ID, BOS_TOKEN.to_string());
        special_token_ids.insert(IMAGE_TOKEN_ID, IMAGE_TOKEN.to_string());
        special_token_ids.insert(IM_START_TOKEN_ID, IM_START_TOKEN.to_string());
        special_token_ids.insert(IM_END_TOKEN_ID, IM_END_TOKEN.to_string());
        special_token_ids.insert(IM_PATCH_TOKEN_ID, IM_PATCH_TOKEN.to_string());
        special_token_ids.insert(AUDIO_TOKEN_ID, AUDIO_TOKEN.to_string());
        special_token_ids.insert(AUDIO_START_TOKEN_ID, AUDIO_START_TOKEN.to_string());
        special_token_ids.insert(AUDIO_END_TOKEN_ID, AUDIO_END_TOKEN.to_string());
        special_token_ids.insert(AUDIO_PATCH_TOKEN_ID, AUDIO_PATCH_TOKEN.to_string());
        special_token_ids.insert(VIDEO_TOKEN_ID, VIDEO_TOKEN.to_string());
        special_token_ids.insert(VIDEO_START_TOKEN_ID, VIDEO_START_TOKEN.to_string());
        special_token_ids.insert(VIDEO_END_TOKEN_ID, VIDEO_END_TOKEN.to_string());
        special_token_ids.insert(VIDEO_FRAME_TOKEN_ID, VIDEO_FRAME_TOKEN.to_string());

        let (byte_encoder, byte_decoder) = Self::build_byte_maps();

        Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            vocab_size: VOCAB_SIZE,
            bos_token_id: BOS_TOKEN_ID,
            eos_token_id: EOS_TOKEN_ID,
            pad_token_id: PAD_TOKEN_ID,
            unk_token_id: UNK_TOKEN_ID,
            special_tokens,
            special_token_ids,
            bpe_index: BpeMergeIndex::new(),
            byte_encoder,
            byte_decoder,
        }
    }

    /// 构建 BPE 字节编码映射
    ///
    /// GPT-2 风格的 BPE 使用特殊的字节到 Unicode 映射，
    /// 避免控制字符和空白字符直接出现在词表中。
    ///
    /// # Returns
    /// (字节到字符映射, 字符到字节映射)
    fn build_byte_maps() -> (HashMap<u8, char>, HashMap<char, u8>) {
        let mut byte_encoder = HashMap::new();
        let mut byte_decoder = HashMap::new();

        // 可打印 ASCII 字符范围: '!' (33) 到 '~' (126)
        // 以及扩展 Latin-1 字符: ¡ (161) 到 ¬ (172) 和 ® (174) 到 ÿ (255)
        let mut n = 0;
        for b in 0u8..=255 {
            let is_printable = (b'!'..=b'~').contains(&b) || (161..=172).contains(&b) || b >= 174;

            if is_printable {
                byte_encoder.insert(b, b as char);
                byte_decoder.insert(b as char, b);
            } else {
                // 非可打印字符映射到 Unicode 私有区域
                let c = char::from_u32(256 + n as u32).unwrap();
                byte_encoder.insert(b, c);
                byte_decoder.insert(c, b);
                n += 1;
            }
        }

        (byte_encoder, byte_decoder)
    }

    /// 获取词表大小
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// 获取开始序列 Token ID
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// 获取结束序列 Token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// 获取填充 Token ID
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// 获取未知 Token ID
    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    /// 设置开始序列 Token ID
    pub fn set_bos_token_id(&mut self, id: u32) {
        self.bos_token_id = id;
    }

    /// 设置结束序列 Token ID
    pub fn set_eos_token_id(&mut self, id: u32) {
        self.eos_token_id = id;
    }

    /// 设置填充 Token ID
    pub fn set_pad_token_id(&mut self, id: u32) {
        self.pad_token_id = id;
    }

    /// 设置未知 Token ID
    pub fn set_unk_token_id(&mut self, id: u32) {
        self.unk_token_id = id;
    }

    /// 从 GGUF 文件加载分词器
    ///
    /// # Parameters
    /// - `path`: GGUF 文件路径
    ///
    /// # Returns
    /// 加载的分词器实例
    pub fn load_from_gguf<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let mut tokenizer = Self::new();

        // 加载词表大小
        let vocab_size = gguf
            .metadata
            .get_u64("general.vocab_size")
            .unwrap_or(VOCAB_SIZE as u64) as usize;
        tokenizer.vocab_size = vocab_size;

        // 加载词表
        if let Some(GgufValue::Array(tokens)) = gguf.metadata.kv.get("tokenizer.ggml.tokens") {
            for (id, token_value) in tokens.iter().enumerate() {
                if let GgufValue::String(token) = token_value {
                    let token_id = id as u32;
                    tokenizer.vocab.insert(token.clone(), token_id);
                    tokenizer.id_to_token.insert(token_id, token.clone());
                }
            }
        }

        // 加载 BPE 合并规则并构建索引
        if let Some(GgufValue::Array(merges)) = gguf.metadata.kv.get("tokenizer.ggml.merges") {
            let mut merge_list = Vec::new();
            for merge_value in merges {
                if let GgufValue::String(merge) = merge_value {
                    let parts: Vec<&str> = merge.split(' ').collect();
                    if parts.len() == 2 {
                        merge_list.push((parts[0].to_string(), parts[1].to_string()));
                    }
                }
            }
            tokenizer.build_bpe_index(&merge_list);
        }

        // 加载额外添加的特殊 Token
        if let Some(GgufValue::Array(special_tokens)) =
            gguf.metadata.kv.get("tokenizer.ggml.added_tokens")
        {
            for (id, token_value) in special_tokens.iter().enumerate() {
                if let GgufValue::String(token) = token_value {
                    let token_id = id as u32;
                    tokenizer.special_tokens.insert(token.clone(), token_id);
                    tokenizer.special_token_ids.insert(token_id, token.clone());
                }
            }
        }

        // 加载 BOS Token ID
        if let Some(bos_id) = gguf.metadata.get_u32("tokenizer.ggml.bos_token_id") {
            tokenizer.bos_token_id = bos_id;
        }

        // 加载 EOS Token ID
        if let Some(eos_id) = gguf.metadata.get_u32("tokenizer.ggml.eos_token_id") {
            tokenizer.eos_token_id = eos_id;
        }

        // 加载 UNK Token ID
        if let Some(unk_id) = gguf.metadata.get_u32("tokenizer.ggml.unknown_token_id") {
            tokenizer.unk_token_id = unk_id;
        }

        // 加载 PAD Token ID
        if let Some(pad_id) = gguf.metadata.get_u32("tokenizer.ggml.padding_token_id") {
            tokenizer.pad_token_id = pad_id;
        }

        Ok(tokenizer)
    }

    /// 从 HuggingFace tokenizer.json 文件加载分词器
    ///
    /// # Parameters
    /// - `path`: tokenizer.json 文件路径
    ///
    /// # Returns
    /// 加载的分词器实例
    pub fn load_from_tokenizer_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let tokenizer_json: TokenizerJson = serde_json::from_str(&content)?;
        let mut tokenizer = Self::new();

        // 加载模型配置
        if let Some(model) = tokenizer_json.model {
            // 加载词表
            if let Some(vocab) = model.vocab {
                tokenizer.vocab_size = vocab.len();
                for (token, id) in &vocab {
                    tokenizer.vocab.insert(token.clone(), *id);
                    tokenizer.id_to_token.insert(*id, token.clone());
                }
            }

            // 加载合并规则
            if let Some(merges) = model.merges {
                let mut merge_list = Vec::new();
                for merge in &merges {
                    let parts: Vec<&str> = merge.split(' ').collect();
                    if parts.len() == 2 {
                        merge_list.push((parts[0].to_string(), parts[1].to_string()));
                    }
                }
                tokenizer.build_bpe_index(&merge_list);
            }

            // 加载未知 Token ID
            if let Some(unk_id) = model.unk_id {
                tokenizer.unk_token_id = unk_id;
            }
        }

        // 加载添加的特殊 Token
        if let Some(added_tokens) = tokenizer_json.added_tokens {
            for token in added_tokens {
                tokenizer
                    .special_tokens
                    .insert(token.content.clone(), token.id);
                tokenizer
                    .special_token_ids
                    .insert(token.id, token.content.clone());
            }
        }

        Ok(tokenizer)
    }

    /// 构建 BPE 合并索引
    ///
    /// 将合并规则转换为高效的查找结构
    ///
    /// # Parameters
    /// - `merges`: 合并规则列表 [(first, second), ...]
    fn build_bpe_index(&mut self, merges: &[(String, String)]) {
        self.bpe_index = BpeMergeIndex::new();

        for (priority, (first, second)) in merges.iter().enumerate() {
            // 获取两个 Token 的 ID
            let first_id = match self.vocab.get(first) {
                Some(&id) => id,
                None => continue,
            };
            let second_id = match self.vocab.get(second) {
                Some(&id) => id,
                None => continue,
            };

            // 合并后的 Token
            let merged = format!("{}{}", first, second);
            let merged_id = match self.vocab.get(&merged) {
                Some(&id) => id,
                None => continue,
            };

            // 添加到索引
            let key = (first_id, second_id);
            self.bpe_index.merges.insert(key, merged_id);
            self.bpe_index.priorities.insert(key, priority);
        }
    }

    /// 编码文本为 Token ID 序列
    ///
    /// # Parameters
    /// - `text`: 输入文本
    ///
    /// # Returns
    /// Token ID 序列
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut tokens = Vec::new();
        let mut remaining = text;

        // 循环处理文本，优先匹配特殊 Token
        while !remaining.is_empty() {
            let mut found_special = false;

            // 查找最早出现的特殊 Token
            let mut earliest_pos = usize::MAX;
            let mut earliest_token = String::new();
            let mut earliest_id = 0u32;

            for (special_token, &token_id) in &self.special_tokens {
                if let Some(pos) = remaining.find(special_token) {
                    if pos < earliest_pos {
                        earliest_pos = pos;
                        earliest_token = special_token.clone();
                        earliest_id = token_id;
                        found_special = true;
                    }
                }
            }

            if found_special {
                // 先编码特殊 Token 之前的普通文本
                if earliest_pos > 0 {
                    let before = &remaining[..earliest_pos];
                    let mut sub_tokens = self.encode_bpe(before)?;
                    tokens.append(&mut sub_tokens);
                }

                // 添加特殊 Token
                tokens.push(earliest_id);
                remaining = &remaining[earliest_pos + earliest_token.len()..];
            } else {
                // 没有特殊 Token，直接编码剩余文本
                let mut sub_tokens = self.encode_bpe(remaining)?;
                tokens.append(&mut sub_tokens);
                break;
            }
        }

        Ok(tokens)
    }

    /// 编码文本并返回完整编码结果
    ///
    /// # Parameters
    /// - `text`: 输入文本
    /// - `add_special_tokens`: 是否添加 BOS/EOS Token
    /// - `max_length`: 最大长度 (可选，超过则截断)
    /// - `padding`: 是否填充到 max_length
    ///
    /// # Returns
    /// 编码结果 (input_ids 和 attention_mask)
    pub fn encode_with_options(
        &self,
        text: &str,
        add_special_tokens: bool,
        max_length: Option<usize>,
        padding: bool,
    ) -> Result<Encoding> {
        let mut input_ids = Vec::new();

        // 添加 BOS Token
        if add_special_tokens {
            input_ids.push(self.bos_token_id);
        }

        // 编码文本
        let mut tokens = self.encode(text)?;
        input_ids.append(&mut tokens);

        // 添加 EOS Token
        if add_special_tokens {
            input_ids.push(self.eos_token_id);
        }

        // 截断
        if let Some(max_len) = max_length {
            if input_ids.len() > max_len {
                input_ids.truncate(max_len);
            }
        }

        // 填充
        if padding {
            if let Some(max_len) = max_length {
                let pad_count = max_len.saturating_sub(input_ids.len());
                for _ in 0..pad_count {
                    input_ids.push(self.pad_token_id);
                }
            }
        }

        // 更新 attention mask (填充部分为 0)
        let attention_mask: Vec<u32> = input_ids
            .iter()
            .map(|&id| if id == self.pad_token_id { 0 } else { 1 })
            .collect();

        Ok(Encoding::new(input_ids, attention_mask))
    }

    /// 使用 BPE 算法编码普通文本
    ///
    /// # Parameters
    /// - `text`: 输入文本
    ///
    /// # Returns
    /// Token ID 序列
    fn encode_bpe(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // vocab 为空时直接返回 unk token
        if self.vocab.is_empty() {
            return Ok(vec![self.unk_token_id; text.chars().count()]);
        }

        // 将文本转换为字节，再映射到 BPE 字符
        let bpe_text: String = text
            .bytes()
            .filter_map(|b| self.byte_encoder.get(&b).copied())
            .collect();

        // 使用贪心最长匹配进行初始分词
        let mut tokens = Vec::new();
        let chars: Vec<char> = bpe_text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut matched = false;

            // 从剩余字符开始，尝试最长匹配
            let remaining: String = chars[i..].iter().collect();
            let remaining_chars: Vec<char> = remaining.chars().collect();

            // 尝试从长到短匹配（按字符数，而非字节数）
            for char_len in (1..=remaining_chars.len()).rev() {
                let substr: String = remaining_chars[..char_len].iter().collect();

                if let Some(&token_id) = self.vocab.get(&substr) {
                    tokens.push(token_id);
                    i += char_len;
                    matched = true;
                    break;
                }
            }

            if !matched {
                // 未找到匹配，使用 unk token
                tokens.push(self.unk_token_id);
                i += 1;
            }
        }

        // 应用 BPE 合并
        tokens = self.apply_bpe_merges(&tokens);

        Ok(tokens)
    }

    /// 应用 BPE 合并规则
    ///
    /// 迭代应用合并规则，直到无法继续合并
    ///
    /// # Parameters
    /// - `tokens`: 当前 Token 序列
    ///
    /// # Returns
    /// 合并后的 Token 序列
    fn apply_bpe_merges(&self, tokens: &[u32]) -> Vec<u32> {
        if tokens.len() < 2 || self.bpe_index.merges.is_empty() {
            return tokens.to_vec();
        }

        let mut result = tokens.to_vec();

        loop {
            // 查找最高优先级的可合并对
            let mut best_merge: Option<(usize, (u32, u32), u32, usize)> = None;

            for i in 0..result.len().saturating_sub(1) {
                let pair = (result[i], result[i + 1]);
                if let Some(&priority) = self.bpe_index.priorities.get(&pair) {
                    if let Some(&merged_id) = self.bpe_index.merges.get(&pair) {
                        match best_merge {
                            None => best_merge = Some((i, pair, merged_id, priority)),
                            Some((_, _, _, best_priority)) if priority < best_priority => {
                                best_merge = Some((i, pair, merged_id, priority));
                            }
                            _ => {}
                        }
                    }
                }
            }

            // 应用最佳合并
            if let Some((pos, _, merged_id, _)) = best_merge {
                result.splice(pos..=pos + 1, [merged_id]);
            } else {
                break;
            }
        }

        result
    }

    /// 解码 Token ID 序列为文本
    ///
    /// # Parameters
    /// - `token_ids`: Token ID 序列
    ///
    /// # Returns
    /// 解码后的文本
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.decode_with_options(token_ids, false, true)
    }

    /// 解码 Token ID 序列为文本（带选项）
    ///
    /// # Parameters
    /// - `token_ids`: Token ID 序列
    /// - `skip_special_tokens`: 是否跳过特殊 Token
    /// - `skip_pad`: 是否跳过 PAD Token
    ///
    /// # Returns
    /// 解码后的文本
    pub fn decode_with_options(
        &self,
        token_ids: &[u32],
        skip_special_tokens: bool,
        skip_pad: bool,
    ) -> Result<String> {
        let mut result = String::new();

        for &id in token_ids {
            // 跳过 PAD Token
            if skip_pad && id == self.pad_token_id {
                continue;
            }

            // 处理特殊 Token
            if let Some(token) = self.special_token_ids.get(&id) {
                if !skip_special_tokens {
                    result.push_str(token);
                }
                continue;
            }

            // 处理普通 Token
            if let Some(token) = self.id_to_token.get(&id) {
                // 处理字节 Token（如 <0xXX>）
                if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
                    if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
                        result.push(byte as char);
                        continue;
                    }
                }

                // 处理 BPE 字节编码
                let decoded = self.decode_bpe_token(token);
                result.push_str(&decoded);
            } else {
                // 未知的 Token ID
                result.push_str(&format!("<unk:{}>", id));
            }
        }

        Ok(result)
    }

    /// 解码单个 BPE Token
    ///
    /// 将 BPE 字符转换回原始字节
    ///
    /// # Parameters
    /// - `token`: Token 字符串
    ///
    /// # Returns
    /// 解码后的字符串
    fn decode_bpe_token(&self, token: &str) -> String {
        let mut bytes = Vec::new();

        for c in token.chars() {
            if let Some(&b) = self.byte_decoder.get(&c) {
                bytes.push(b);
            } else {
                // 非 BPE 编码字符，直接使用原始字符
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }

        // 尝试转换为 UTF-8 字符串
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// 获取 Token ID
    ///
    /// # Parameters
    /// - `token`: Token 字符串
    ///
    /// # Returns
    /// Token ID（如果存在）
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.vocab
            .get(token)
            .copied()
            .or_else(|| self.special_tokens.get(token).copied())
    }

    /// 获取 Token 字符串
    ///
    /// # Parameters
    /// - `id`: Token ID
    ///
    /// # Returns
    /// Token 字符串（如果存在）
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token
            .get(&id)
            .map(|s| s.as_str())
            .or_else(|| self.special_token_ids.get(&id).map(|s| s.as_str()))
    }

    /// 检查是否为特殊 Token
    ///
    /// # Parameters
    /// - `id`: Token ID
    ///
    /// # Returns
    /// 如果是特殊 Token 返回 true
    pub fn is_special_token(&self, id: u32) -> bool {
        self.special_token_ids.contains_key(&id)
    }

    /// 添加自定义特殊 Token
    ///
    /// # Parameters
    /// - `token`: Token 字符串
    /// - `id`: Token ID
    pub fn add_special_token(&mut self, token: &str, id: u32) {
        self.special_tokens.insert(token.to_string(), id);
        self.special_token_ids.insert(id, token.to_string());
    }

    /// 添加 Token 到词表
    ///
    /// # Parameters
    /// - `token`: Token 字符串
    /// - `id`: Token ID
    pub fn add_token(&mut self, token: &str, id: u32) {
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
    }

    /// 批量编码文本
    ///
    /// # Parameters
    /// - `texts`: 文本列表
    /// - `add_special_tokens`: 是否添加特殊 Token
    /// - `max_length`: 最大长度
    /// - `padding`: 是否填充
    ///
    /// # Returns
    /// 编码结果列表
    pub fn batch_encode(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
        max_length: Option<usize>,
        padding: bool,
    ) -> Result<Vec<Encoding>> {
        let mut encodings = Vec::new();

        for text in texts {
            let encoding =
                self.encode_with_options(text, add_special_tokens, max_length, padding)?;
            encodings.push(encoding);
        }

        Ok(encodings)
    }

    /// 批量解码
    ///
    /// # Parameters
    /// - `batch_ids`: Token ID 序列列表
    /// - `skip_special_tokens`: 是否跳过特殊 Token
    ///
    /// # Returns
    /// 解码后的文本列表
    pub fn batch_decode(
        &self,
        batch_ids: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        let mut results = Vec::new();

        for ids in batch_ids {
            let text = self.decode_with_options(ids, skip_special_tokens, true)?;
            results.push(text);
        }

        Ok(results)
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_new() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.vocab_size(), VOCAB_SIZE);
        assert_eq!(tokenizer.bos_token_id(), BOS_TOKEN_ID);
        assert_eq!(tokenizer.eos_token_id(), EOS_TOKEN_ID);
        assert_eq!(tokenizer.pad_token_id(), PAD_TOKEN_ID);
        assert_eq!(tokenizer.unk_token_id(), UNK_TOKEN_ID);
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.get_token_id(IMAGE_TOKEN), Some(IMAGE_TOKEN_ID));
        assert_eq!(
            tokenizer.get_token_id(IM_START_TOKEN),
            Some(IM_START_TOKEN_ID)
        );
        assert_eq!(tokenizer.get_token_id(IM_END_TOKEN), Some(IM_END_TOKEN_ID));
        assert_eq!(
            tokenizer.get_token_id(IM_PATCH_TOKEN),
            Some(IM_PATCH_TOKEN_ID)
        );
        assert_eq!(tokenizer.get_token_id(UNK_TOKEN), Some(UNK_TOKEN_ID));
        assert_eq!(tokenizer.get_token_id(PAD_TOKEN), Some(PAD_TOKEN_ID));
    }

    #[test]
    fn test_is_special_token() {
        let tokenizer = Tokenizer::new();
        assert!(tokenizer.is_special_token(IMAGE_TOKEN_ID));
        assert!(tokenizer.is_special_token(IM_START_TOKEN_ID));
        assert!(tokenizer.is_special_token(PAD_TOKEN_ID));
        assert!(tokenizer.is_special_token(UNK_TOKEN_ID));
        assert!(!tokenizer.is_special_token(100));
    }

    #[test]
    fn test_decode_empty() {
        let tokenizer = Tokenizer::new();
        let result = tokenizer.decode(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_special_tokens() {
        let tokenizer = Tokenizer::new();
        let result = tokenizer.decode(&[IMAGE_TOKEN_ID]).unwrap();
        assert_eq!(result, IMAGE_TOKEN);
    }

    #[test]
    fn test_decode_skip_special_tokens() {
        let tokenizer = Tokenizer::new();
        let result = tokenizer
            .decode_with_options(&[IMAGE_TOKEN_ID], true, true)
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_skip_pad() {
        let tokenizer = Tokenizer::new();
        let result = tokenizer
            .decode_with_options(&[PAD_TOKEN_ID, PAD_TOKEN_ID], false, true)
            .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_token_ids_constants() {
        assert_eq!(UNK_TOKEN_ID, 0);
        assert_eq!(PAD_TOKEN_ID, 151642);
        assert_eq!(EOS_TOKEN_ID, 151643);
        assert_eq!(BOS_TOKEN_ID, 151644);
        assert_eq!(IMAGE_TOKEN_ID, 151645);
        assert_eq!(IM_START_TOKEN_ID, 151646);
        assert_eq!(IM_END_TOKEN_ID, 151647);
        assert_eq!(IM_PATCH_TOKEN_ID, 151648);
    }

    #[test]
    fn test_tokenizer_add_special_token() {
        let mut tokenizer = Tokenizer::new();
        tokenizer.add_special_token("<test_token>", 99999);
        assert_eq!(tokenizer.get_token_id("<test_token>"), Some(99999));
        assert!(tokenizer.is_special_token(99999));
    }

    #[test]
    fn test_tokenizer_get_token() {
        let tokenizer = Tokenizer::new();
        assert_eq!(tokenizer.get_token(IMAGE_TOKEN_ID), Some(IMAGE_TOKEN));
        assert_eq!(tokenizer.get_token(IM_START_TOKEN_ID), Some(IM_START_TOKEN));
        assert_eq!(tokenizer.get_token(IM_END_TOKEN_ID), Some(IM_END_TOKEN));
        assert_eq!(tokenizer.get_token(UNK_TOKEN_ID), Some(UNK_TOKEN));
        assert_eq!(tokenizer.get_token(PAD_TOKEN_ID), Some(PAD_TOKEN));
    }

    #[test]
    fn test_decode_multiple_special_tokens() {
        let tokenizer = Tokenizer::new();
        let result = tokenizer
            .decode(&[IM_START_TOKEN_ID, IM_PATCH_TOKEN_ID, IM_END_TOKEN_ID])
            .unwrap();
        assert!(result.contains(IM_START_TOKEN));
        assert!(result.contains(IM_PATCH_TOKEN));
        assert!(result.contains(IM_END_TOKEN));
    }

    #[test]
    fn test_vocab_size_constant() {
        assert_eq!(VOCAB_SIZE, 152064);
    }

    #[test]
    fn test_tokenizer_encode_empty() {
        let tokenizer = Tokenizer::new();
        let result = tokenizer.encode("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_tokenizer_decode_roundtrip() {
        let tokenizer = Tokenizer::new();
        let tokens = vec![IMAGE_TOKEN_ID, IM_START_TOKEN_ID, IM_END_TOKEN_ID];
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_special_token_strings() {
        assert_eq!(UNK_TOKEN, "<|unk|>");
        assert_eq!(PAD_TOKEN, "<|pad|>");
        assert_eq!(EOS_TOKEN, "<|endoftext|>");
        assert_eq!(BOS_TOKEN, "<|im_start|>");
        assert_eq!(IMAGE_TOKEN, "<image>");
        assert_eq!(IM_START_TOKEN, "<im_start>");
        assert_eq!(IM_END_TOKEN, "<im_end>");
        assert_eq!(IM_PATCH_TOKEN, "<im_patch>");
    }

    #[test]
    fn test_tokenizer_encode_with_vocab() {
        let mut tokenizer = Tokenizer::new();
        tokenizer.add_token("hello", 100);
        tokenizer.add_token("world", 101);

        let result = tokenizer.encode("hello world").unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tokenizer_encode_empty_vocab() {
        let tokenizer = Tokenizer::new();
        let result = tokenizer.encode("hello").unwrap();
        // vocab 为空时应返回 unk token
        assert!(!result.is_empty());
        assert!(result.contains(&UNK_TOKEN_ID));
    }

    #[test]
    fn test_tokenizer_encode_special_token_first() {
        let tokenizer = Tokenizer::new();
        let result = tokenizer.encode("<image>test").unwrap();
        // 应该优先识别特殊 token
        assert!(!result.is_empty());
        assert_eq!(result[0], IMAGE_TOKEN_ID);
    }

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        let mut tokenizer = Tokenizer::new();

        // 构建简单词表
        tokenizer.add_token("hello", 100);
        tokenizer.add_token("world", 101);
        tokenizer.add_token(" ", 102);

        // 编码
        let tokens = tokenizer.encode("hello world").unwrap();
        assert!(!tokens.is_empty());

        // 解码
        let decoded = tokenizer.decode(&tokens).unwrap();

        // 验证包含原始文本
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_tokenizer_bpe_merge_rules() {
        let mut tokenizer = Tokenizer::new();

        // 模拟 BPE 合并规则
        tokenizer.add_token("h", 1);
        tokenizer.add_token("e", 2);
        tokenizer.add_token("l", 3);
        tokenizer.add_token("o", 4);
        tokenizer.add_token("he", 10);

        // 添加合并规则: "h" + "e" -> "he"
        tokenizer.build_bpe_index(&[("h".to_string(), "e".to_string())]);

        let tokens = tokenizer.encode("hello").unwrap();
        // 应该能识别 "he" 这个合并 token
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_encode_with_options() {
        let mut tokenizer = Tokenizer::new();
        tokenizer.add_token("test", 100);

        // 测试添加特殊 token
        let encoding = tokenizer
            .encode_with_options("test", true, Some(10), true)
            .unwrap();
        assert!(encoding.input_ids.starts_with(&[BOS_TOKEN_ID]));
        assert!(encoding.input_ids.contains(&100));
        assert!(encoding.input_ids.contains(&EOS_TOKEN_ID));

        // 验证 attention mask
        assert!(encoding.attention_mask.iter().all(|&m| m == 1
            || encoding.input_ids[encoding
                .attention_mask
                .iter()
                .position(|&x| x == 0)
                .unwrap_or(0)]
                == PAD_TOKEN_ID));
    }

    #[test]
    fn test_encode_with_padding() {
        let mut tokenizer = Tokenizer::new();
        tokenizer.add_token("a", 100);

        let encoding = tokenizer
            .encode_with_options("a", false, Some(10), true)
            .unwrap();
        assert_eq!(encoding.len(), 10);

        // 验证填充
        let pad_count = encoding
            .input_ids
            .iter()
            .filter(|&&id| id == PAD_TOKEN_ID)
            .count();
        assert!(pad_count > 0);

        // 验证 attention mask
        let valid_count = encoding.attention_mask.iter().filter(|&&m| m == 1).count();
        let pad_mask_count = encoding.attention_mask.iter().filter(|&&m| m == 0).count();
        assert_eq!(pad_count, pad_mask_count);
        assert_eq!(valid_count + pad_mask_count, 10);
    }

    #[test]
    fn test_encode_truncation() {
        let mut tokenizer = Tokenizer::new();
        for i in 0..20 {
            tokenizer.add_token(&format!("t{}", i), 100 + i as u32);
        }

        let encoding = tokenizer
            .encode_with_options(
                "t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 t15 t16 t17 t18 t19",
                false,
                Some(5),
                false,
            )
            .unwrap();
        assert_eq!(encoding.len(), 5);
    }

    #[test]
    fn test_batch_encode() {
        let mut tokenizer = Tokenizer::new();
        tokenizer.add_token("a", 100);
        tokenizer.add_token("b", 101);

        let texts = vec!["a", "b", "a b"];
        let encodings = tokenizer.batch_encode(&texts, false, None, false).unwrap();

        assert_eq!(encodings.len(), 3);
        assert!(encodings[0].input_ids.contains(&100));
        assert!(encodings[1].input_ids.contains(&101));
    }

    #[test]
    fn test_batch_decode() {
        let tokenizer = Tokenizer::new();

        let batch_ids = vec![
            vec![IMAGE_TOKEN_ID],
            vec![IM_START_TOKEN_ID, IM_END_TOKEN_ID],
        ];

        let results = tokenizer.batch_decode(&batch_ids, false).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].contains(IMAGE_TOKEN));
    }

    #[test]
    fn test_byte_maps() {
        let tokenizer = Tokenizer::new();

        // 验证字节映射完整性
        assert_eq!(tokenizer.byte_encoder.len(), 256);
        assert_eq!(tokenizer.byte_decoder.len(), 256);

        // 验证可打印字符映射
        for b in b'!'..=b'~' {
            assert_eq!(tokenizer.byte_encoder[&b], b as char);
        }
    }

    #[test]
    fn test_set_token_ids() {
        let mut tokenizer = Tokenizer::new();

        tokenizer.set_bos_token_id(1);
        tokenizer.set_eos_token_id(2);
        tokenizer.set_pad_token_id(3);
        tokenizer.set_unk_token_id(4);

        assert_eq!(tokenizer.bos_token_id(), 1);
        assert_eq!(tokenizer.eos_token_id(), 2);
        assert_eq!(tokenizer.pad_token_id(), 3);
        assert_eq!(tokenizer.unk_token_id(), 4);
    }

    #[test]
    fn test_encoding_struct() {
        let encoding = Encoding::new(vec![1, 2, 3], vec![1, 1, 1]);
        assert_eq!(encoding.len(), 3);
        assert!(!encoding.is_empty());

        let empty_encoding = Encoding::new(vec![], vec![]);
        assert!(empty_encoding.is_empty());
    }

    #[test]
    fn test_decode_unknown_token() {
        let tokenizer = Tokenizer::new();
        let result = tokenizer.decode(&[999999]).unwrap();
        assert!(result.contains("<unk:"));
    }

    #[test]
    fn test_decode_byte_token() {
        let mut tokenizer = Tokenizer::new();
        tokenizer.add_token("<0x41>", 200);

        let result = tokenizer.decode(&[200]).unwrap();
        assert!(result.contains('A'));
    }

    #[test]
    fn test_add_token() {
        let mut tokenizer = Tokenizer::new();
        tokenizer.add_token("new_token", 12345);

        assert_eq!(tokenizer.get_token_id("new_token"), Some(12345));
        assert_eq!(tokenizer.get_token(12345), Some("new_token"));
    }
}
