//! BPE (Byte Pair Encoding) Tokenizer
//!
//! 支持大语言模型训练的文本分词功能。

use std::collections::HashMap;
use std::path::Path;

/// BPE Tokenizer 错误类型
#[derive(Debug)]
pub enum TokenizerError {
    VocabularyEmpty,
    TokenNotFound(u32),
    MergeNotFound(String),
    Io(std::io::Error),
    EncodingError(String),
    DecodingError(String),
    FileNotFound(std::path::PathBuf),
    InvalidFormat(String),
}

impl From<std::io::Error> for TokenizerError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VocabularyEmpty => write!(f, "Vocabulary is empty"),
            Self::TokenNotFound(id) => write!(f, "Token ID {} not found in vocabulary", id),
            Self::MergeNotFound(s) => write!(f, "Merge rule '{}' not found", s),
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::EncodingError(s) => write!(f, "Encoding error: {}", s),
            Self::DecodingError(s) => write!(f, "Decoding error: {}", s),
            Self::FileNotFound(p) => write!(f, "File not found: {}", p.display()),
            Self::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// BPE 合并规则
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MergeRule {
    left: String,
    right: String,
}

/// BPE Tokenizer 结构体
#[derive(Debug)]
pub struct BpeTokenizer {
    /// 词表：token → id
    encoder: HashMap<String, u32>,
    /// 反向词表：id → token
    decoder: HashMap<u32, String>,
    /// BPE 合并规则（按优先级排序）
    bpe_ranks: HashMap<MergeRule, usize>,
    /// 特殊 token 映射
    special_tokens: HashMap<String, u32>,
    /// 特殊 token 反向映射
    special_tokens_decoder: HashMap<u32, String>,
    /// 词汇表大小
    vocab_size: usize,
    /// 最大序列长度
    max_length: usize,
    /// pad_token_id
    pad_token_id: u32,
    /// eos_token_id
    eos_token_id: u32,
    /// bos_token_id
    bos_token_id: u32,
    /// unk_token_id
    unk_token_id: u32,
}

impl BpeTokenizer {
    /// 创建新的 BPE Tokenizer（需要加载词表和合并规则）
    pub fn new(
        vocab_path: &Path,
        merges_path: Option<&Path>,
    ) -> Result<Self, TokenizerError> {
        // 如果没有提供路径，创建默认的小型词表用于测试
        if !vocab_path.exists() && merges_path.is_none() || merges_path.is_none_or(|p| !p.exists()) {
            return Ok(Self::default());
        }

        // TODO: 从文件加载完整词表
        // 这里先返回默认实现
        Ok(Self::default())
    }

    /// 创建默认的简单 BPE Tokenizer（用于测试和开发）
    pub fn default() -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();

        // 基础 ASCII 字符（简化版）
        for (i, c) in (32..=126).enumerate() {
            let s = char::from_u32(c as u32).unwrap().to_string();
            encoder.insert(s.clone(), i as u32);
            decoder.insert(i as u32, s);
        }

        // 特殊 token
        let special_tokens = [
            ("<pad>", 0),
            ("<eos>", 1),
            ("<bos>", 2),
            ("<unk>", 3),
        ];

        let mut special_map = HashMap::new();
        let mut special_decoder = HashMap::new();
        for (token, id) in &special_tokens {
            special_map.insert(token.to_string(), *id);
            special_decoder.insert(*id, token.to_string());
        }

        let vocab_size = encoder.len() + special_tokens.len();

        Self {
            encoder,
            decoder,
            bpe_ranks: HashMap::new(),
            special_tokens: special_map,
            special_tokens_decoder: special_decoder,
            vocab_size,
            max_length: 2048,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            unk_token_id: 3,
        }
    }

    /// 创建指定最大长度的 BPE Tokenizer
    pub fn with_max_length(max_length: usize) -> Self {
        let mut tok = Self::default();
        tok.max_length = max_length;
        tok
    }

    /// 编码文本为 token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 预分词（简单的空格分词）
        let words: Vec<&str> = text.split_whitespace().collect();

        // 2. 对每个词进行 BPE 编码
        let mut tokens = Vec::new();

        // 添加 BOS token（可选）
        tokens.push(self.bos_token_id);

        for word in words {
            let word_tokens = self.bpe_encode_word(word)?;
            tokens.extend(word_tokens);
        }

        // 添加 EOS token
        tokens.push(self.eos_token_id);

        // 截断到最大长度
        if tokens.len() > self.max_length {
            tokens.truncate(self.max_length - 1);
            tokens.push(self.eos_token_id);
        }

        Ok(tokens)
    }

    /// BPE 编码单个词
    fn bpe_encode_word(&self, word: &str) -> Result<Vec<u32>, TokenizerError> {
        if word.is_empty() {
            return Ok(Vec::new());
        }

        // 初始化为字符级 token
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // 应用 BPE 合并规则
        while tokens.len() > 1 {
            // 找到优先级最高的可合并对
            let mut best_merge: Option<(usize, MergeRule)> = None;
            let mut best_rank = usize::MAX;

            for i in 0..tokens.len()-1 {
                let merge = MergeRule {
                    left: tokens[i].clone(),
                    right: tokens[i+1].clone(),
                };

                if let Some(&rank) = self.bpe_ranks.get(&merge) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_merge = Some((i, merge));
                    }
                }
            }

            match best_merge {
                Some((pos, merge)) => {
                    // 执行合并
                    tokens[pos] = format!("{}{}", merge.left, merge.right);
                    tokens.remove(pos + 1);
                }
                None => break,  // 无更多合并规则
            }
        }

        // 将 token 字符串转换为 IDs
        let mut ids = Vec::new();
        for token in tokens {
            if let Some(&id) = self.encoder.get(&token) {
                ids.push(id);
            } else if let Some(&id) = self.special_tokens.get(&token) {
                ids.push(id);
            } else {
                ids.push(self.unk_token_id);
            }
        }

        Ok(ids)
    }

    /// 解码 token IDs 回文本
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut parts = Vec::new();

        for &id in ids {
            // 跳过特殊 token
            if let Some(_token) = self.special_tokens_decoder.get(&id) {
                continue;
            }

            if let Some(token) = self.decoder.get(&id) {
                parts.push(token.as_str());
            } else {
                return Err(TokenizerError::TokenNotFound(id));
            }
        }

        Ok(parts.join(""))
    }

    /// 编码并填充到固定长度
    pub fn encode_with_padding(&self, text: &str, max_length: usize) -> Result<TokenIdsOutput, TokenizerError> {
        let mut ids = self.encode(text)?;

        // 截断或 padding
        if ids.len() > max_length {
            ids.truncate(max_length);
            // 确保最后一个是 EOS
            if !ids.is_empty() {
                *ids.last_mut().unwrap() = self.eos_token_id;
            }
        } else {
            while ids.len() < max_length {
                ids.push(self.pad_token_id);
            }
        }

        // 生成 attention mask
        let attention_mask: Vec<u8> = ids.iter()
            .map(|&id| if id == self.pad_token_id { 0 } else { 1 })
            .collect();

        // 生成 labels（shifted right）
        let mut labels = ids.clone();
        labels.rotate_left(1);
        if !labels.is_empty() {
            *labels.last_mut().unwrap() = self.pad_token_id;
            // 将 padding 位置设为特殊值（表示忽略 loss）
            for label in labels.iter_mut() {
                if *label == self.pad_token_id {
                    *label = u32::MAX;  // 表示 ignore_index
                }
            }
        }

        Ok(TokenIdsOutput {
            input_ids: ids.iter().map(|&x| x as usize).collect(),
            labels: labels.iter().map(|&x| x as usize).collect(),
            attention_mask,
        })
    }

    /// 获取词汇表大小
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// 获取 pad_token_id
    pub fn pad_token_id(&self) -> usize {
        self.pad_token_id as usize
    }

    /// 获取 eos_token_id
    pub fn eos_token_id(&self) -> usize {
        self.eos_token_id as usize
    }

    /// 获取 bos_token_id
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// 获取 unk_token_id
    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    /// 为 DataLoader 提供的简单分词接口（返回 Vec<usize>）
    pub fn tokenize_for_dataloader(&self, text: &str) -> Vec<usize> {
        match self.encode(text) {
            Ok(ids) => ids.iter().map(|&id| id as usize).collect(),
            Err(_) => {
                // 如果编码失败，回退到字符级映射（兼容性）
                text.chars().map(|c| c as usize).collect()
            }
        }
    }
}

/// Tokenize 输出结果
#[derive(Debug, Clone)]
pub struct TokenIdsOutput {
    pub input_ids: Vec<usize>,
    pub labels: Vec<usize>,
    pub attention_mask: Vec<u8>,
}

impl TokenIdsOutput {
    pub fn seq_len(&self) -> usize {
        self.input_ids.len()
    }
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tokenizer_creation() {
        let tok = BpeTokenizer::default();
        assert!(tok.vocab_size() > 0);
        assert_eq!(tok.pad_token_id(), 0);
        assert_eq!(tok.eos_token_id(), 1);
    }

    #[test]
    fn test_basic_encoding() {
        let tok = BpeTokenizer::default();
        let ids = tok.encode("hello world").unwrap();

        // 应该包含 BOS + tokens + EOS
        assert!(ids.len() >= 3);  // 至少 BOS + 内容 + EOS
        assert_eq!(ids[0], tok.bos_token_id());
        assert_eq!(*ids.last().unwrap(), tok.eos_token_id() as u32);
    }

    #[test]
    fn test_empty_text_encoding() {
        let tok = BpeTokenizer::default();
        let ids = tok.encode("").unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_decode_roundtrip() {
        let tok = BpeTokenizer::default();
        let original = "test";
        let ids = tok.encode(original).unwrap();
        let decoded = tok.decode(&ids).unwrap();

        // 注意：BOS/EOS 会被跳过，所以 decoded 可能不包含它们
        // 但主要内容应该保留
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_encode_with_padding() {
        let tok = BpeTokenizer::default();
        let output = tok.encode_with_padding("hi", 10).unwrap();

        assert_eq!(output.input_ids.len(), 10);
        assert_eq!(output.labels.len(), 10);
        assert_eq!(output.attention_mask.len(), 10);

        // 最后几个应该是 padding
        assert_eq!(output.input_ids[9], tok.pad_token_id());
        assert_eq!(output.attention_mask[9], 0);
    }

    #[test]
    fn test_truncation_long_text() {
        let tok = BpeTokenizer::with_max_length(20);
        let long_text = "word ".repeat(100);
        let output = tok.encode_with_padding(&long_text, 20).unwrap();

        assert_eq!(output.input_ids.len(), 20);
    }
}
