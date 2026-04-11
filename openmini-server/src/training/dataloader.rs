//! 训练数据加载器
//!
//! 支持多种数据格式（JSONL、纯文本）的高效数据加载与预处理，
//! 提供 Causal LM 和 SFT 两种训练模式的数据处理能力。

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use super::tokenizer::{BpeTokenizer, TokenizerError};

/// 单条训练样本（原始格式）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSample {
    #[serde(default)]
    pub text: Option<String>,

    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub input: Option<String>,
    #[serde(default)]
    pub output: Option<String>,

    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// 处理后的训练样本（Tokenize 后）
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub input_ids: Vec<usize>,
    pub labels: Vec<usize>,
    pub attention_mask: Vec<u8>,
    pub seq_len: usize,
}

/// 一个 Batch 的数据
#[derive(Debug, Clone)]
pub struct Batch {
    pub input_ids: Vec<Vec<usize>>,
    pub labels: Vec<Vec<usize>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub batch_size: usize,
    pub seq_len: usize,
    pub num_tokens: usize,
}

/// DataLoader 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    pub train_path: PathBuf,
    pub validation_path: Option<PathBuf>,
    pub format: DataFormat,

    pub max_seq_length: usize,
    pub pad_token_id: usize,
    pub eos_token_id: usize,

    pub batch_size: usize,
    pub drop_last: bool,
    pub shuffle: bool,

    #[serde(default)]
    pub sft_config: Option<SFTConfig>,
}

/// SFT 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SFTConfig {
    pub prompt_template: String,
    pub mask_prompt_loss: bool,
    pub max_prompt_length: Option<usize>,
    pub max_response_length: Option<usize>,
}

/// 数据格式枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    Jsonl,
    Text,
}

/// DataLoader 错误类型
#[derive(Debug)]
pub enum DataLoaderError {
    FileNotFound(PathBuf),
    ParseError { line: usize, error: String },
    InvalidFormat(String),
    TokenizationError(String),
    EmptyDataset,
    Io(std::io::Error),
    Tokenizer(TokenizerError),
}

impl std::fmt::Display for DataLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataLoaderError::FileNotFound(path) => write!(f, "文件未找到: {}", path.display()),
            DataLoaderError::ParseError { line, error } => {
                write!(f, "解析错误 (行 {}): {}", line, error)
            }
            DataLoaderError::InvalidFormat(msg) => write!(f, "无效的数据格式: {}", msg),
            DataLoaderError::TokenizationError(msg) => write!(f, "分词错误: {}", msg),
            DataLoaderError::EmptyDataset => write!(f, "数据集为空"),
            DataLoaderError::Io(err) => write!(f, "IO 错误: {}", err),
            DataLoaderError::Tokenizer(err) => write!(f, "Tokenizer 错误: {}", err),
        }
    }
}

impl std::error::Error for DataLoaderError {}

impl From<std::io::Error> for DataLoaderError {
    fn from(err: std::io::Error) -> Self {
        DataLoaderError::Io(err)
    }
}

impl From<TokenizerError> for DataLoaderError {
    fn from(err: TokenizerError) -> Self {
        DataLoaderError::Tokenizer(err)
    }
}

/// DataLoader 主结构
#[derive(Debug)]
pub struct DataLoader {
    config: DataLoaderConfig,
    samples: Vec<TrainingSample>,
    epoch_samples: Vec<usize>,
    current_position: usize,
    rng: rand::rngs::StdRng,
    tokenizer: BpeTokenizer,
}

impl DataLoader {
    /// 从配置创建 DataLoader 并加载数据
    pub fn new(config: DataLoaderConfig) -> Result<Self, DataLoaderError> {
        let path = &config.train_path;
        if !path.exists() {
            return Err(DataLoaderError::FileNotFound(path.clone()));
        }

        let raw_samples = match config.format {
            DataFormat::Jsonl => Self::load_jsonl(path)?,
            DataFormat::Text => Self::load_text(path)?,
        };

        if raw_samples.is_empty() {
            return Err(DataLoaderError::EmptyDataset);
        }

        // 初始化 BPE Tokenizer（使用默认配置）
        let tokenizer = BpeTokenizer::default();

        let mut samples = Vec::with_capacity(raw_samples.len());
        for (i, raw) in raw_samples.iter().enumerate() {
            match Self::preprocess_with_tokenizer(&tokenizer, &config, raw) {
                Ok(sample) => samples.push(sample),
                Err(e) => {
                    tracing::warn!("预处理样本 {} 失败: {}, 跳过", i, e);
                }
            }
        }

        if samples.is_empty() {
            return Err(DataLoaderError::EmptyDataset);
        }

        let sample_count = samples.len();
        let epoch_samples: Vec<usize> = (0..sample_count).collect();
        let rng = rand::rngs::StdRng::from_entropy();

        let mut loader = Self {
            config,
            samples,
            epoch_samples,
            current_position: 0,
            rng,
            tokenizer,
        };

        if loader.config.shuffle {
            loader.shuffle();
        }

        Ok(loader)
    }

    fn load_jsonl(path: &Path) -> Result<Vec<RawSample>, DataLoaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(DataLoaderError::Io)?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<RawSample>(&line) {
                Ok(sample) => samples.push(sample),
                Err(e) => {
                    return Err(DataLoaderError::ParseError {
                        line: line_num + 1,
                        error: e.to_string(),
                    });
                }
            }
        }

        Ok(samples)
    }

    fn load_text(path: &Path) -> Result<Vec<RawSample>, DataLoaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut samples = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                samples.push(RawSample {
                    text: Some(line),
                    instruction: None,
                    input: None,
                    output: None,
                    metadata: None,
                });
            }
        }

        Ok(samples)
    }

    /// 使用 tokenizer 预处理样本
    fn preprocess_with_tokenizer(
        tokenizer: &BpeTokenizer,
        config: &DataLoaderConfig,
        raw: &RawSample,
    ) -> Result<TrainingSample, DataLoaderError> {
        if config.sft_config.is_some()
            && (raw.instruction.is_some() || raw.output.is_some())
        {
            Self::preprocess_sft_with_tokenizer(tokenizer, config, raw)
        } else if let Some(ref text) = raw.text {
            Ok(Self::preprocess_causal_lm_with_tokenizer(tokenizer, config, text))
        } else {
            Err(DataLoaderError::InvalidFormat(
                "无法确定样本格式：缺少必要字段".to_string(),
            ))
        }
    }

    fn preprocess_causal_lm_with_tokenizer(
        tokenizer: &BpeTokenizer,
        config: &DataLoaderConfig,
        text: &str,
    ) -> TrainingSample {
        let tokens = tokenizer.tokenize_for_dataloader(text);
        let max_len = config.max_seq_length;

        let truncated_tokens: Vec<usize> = if tokens.len() > max_len {
            tokens[..max_len].to_vec()
        } else {
            tokens
        };

        let seq_len = truncated_tokens.len();
        let (input_ids, labels, attention_mask) =
            Self::pad_sequence(&truncated_tokens, max_len, config.pad_token_id);

        TrainingSample {
            input_ids,
            labels,
            attention_mask,
            seq_len,
        }
    }

    fn preprocess_sft_with_tokenizer(
        tokenizer: &BpeTokenizer,
        config: &DataLoaderConfig,
        sample: &RawSample,
    ) -> Result<TrainingSample, DataLoaderError> {
        let sft_config = config
            .sft_config
            .as_ref()
            .ok_or_else(|| DataLoaderError::InvalidFormat("缺少 SFT 配置".to_string()))?;

        let instruction = sample
            .instruction
            .as_deref()
            .unwrap_or("");
        let input = sample.input.as_deref().unwrap_or("");
        let output = sample
            .output
            .as_ref()
            .ok_or_else(|| {
                DataLoaderError::InvalidFormat("SFT 模式需要 output 字段".to_string())
            })?;

        let prompt = sft_config
            .prompt_template
            .replace("{instruction}", instruction)
            .replace("{input}", input);

        let prompt_tokens = tokenizer.tokenize_for_dataloader(&prompt);
        let response_tokens = tokenizer.tokenize_for_dataloader(output);

        let max_prompt_len = sft_config.max_prompt_length.unwrap_or(usize::MAX);
        let max_response_len = sft_config.max_response_length.unwrap_or(usize::MAX);

        let truncated_prompt: Vec<usize> = if prompt_tokens.len() > max_prompt_len {
            prompt_tokens[prompt_tokens.len() - max_prompt_len..].to_vec()
        } else {
            prompt_tokens
        };

        let truncated_response: Vec<usize> = if response_tokens.len() > max_response_len {
            response_tokens[..max_response_len].to_vec()
        } else {
            response_tokens
        };

        let mut all_tokens = truncated_prompt.clone();
        all_tokens.extend(truncated_response.iter());
        all_tokens.push(config.eos_token_id);

        let max_len = config.max_seq_length;
        let truncated_tokens: Vec<usize> = if all_tokens.len() > max_len {
            all_tokens[..max_len].to_vec()
        } else {
            all_tokens
        };

        let prompt_len = truncated_prompt.len();
        let (input_ids, _labels, _attention_mask) = if sft_config.mask_prompt_loss {
            let mut labels_vec = truncated_tokens.clone();
            for i in 0..prompt_len.min(labels_vec.len()) {
                labels_vec[i] = usize::MAX;
            }
            (
                truncated_tokens.clone(),
                labels_vec,
                vec![1u8; truncated_tokens.len()],
            )
        } else {
            (
                truncated_tokens.clone(),
                truncated_tokens.clone(),
                vec![1u8; truncated_tokens.len()],
            )
        };

        let (padded_input_ids, padded_labels, padded_attention_mask) =
            Self::pad_sequence(&input_ids, max_len, config.pad_token_id);

        let final_labels = if sft_config.mask_prompt_loss {
            let mut final_labels_vec = padded_labels;
            for i in 0..prompt_len.min(final_labels_vec.len()) {
                if padded_attention_mask[i] == 1 {
                    final_labels_vec[i] = usize::MAX;
                }
            }
            final_labels_vec
        } else {
            padded_labels
        };

        let seq_len = truncated_tokens.len();

        Ok(TrainingSample {
            input_ids: padded_input_ids,
            labels: final_labels,
            attention_mask: padded_attention_mask,
            seq_len,
        })
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        // 使用 BPE Tokenizer 进行分词
        match self.tokenizer.encode(text) {
            Ok(ids) => ids.iter().map(|&id| id as usize).collect(),
            Err(_) => {
                // 如果编码失败，回退到字符级映射（兼容性）
                text.chars().map(|c| c as usize).collect()
            }
        }
    }

    fn pad_sequence(
        ids: &[usize],
        length: usize,
        pad_token_id: usize,
    ) -> (Vec<usize>, Vec<usize>, Vec<u8>) {
        let mut input_ids = vec![pad_token_id; length];
        let mut labels = vec![usize::MAX; length];
        let mut attention_mask = vec![0u8; length];

        let copy_len = ids.len().min(length);
        input_ids[..copy_len].copy_from_slice(&ids[..copy_len]);
        labels[..copy_len].copy_from_slice(&ids[..copy_len]);
        for mask_val in attention_mask[..copy_len].iter_mut() {
            *mask_val = 1;
        }

        (input_ids, labels, attention_mask)
    }

    fn shuffle(&mut self) {
        self.epoch_samples.shuffle(&mut self.rng);
    }

    pub fn reset_epoch(&mut self) {
        self.current_position = 0;
        if self.config.shuffle {
            self.shuffle();
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl Iterator for DataLoader {
    type Item = Batch;

    fn next(&mut self) -> Option<Batch> {
        if self.current_position >= self.epoch_samples.len() {
            return None;
        }

        let remaining = self.epoch_samples.len() - self.current_position;

        if remaining < self.config.batch_size && self.config.drop_last {
            self.current_position = self.epoch_samples.len();
            return None;
        }

        let actual_batch_size = remaining.min(self.config.batch_size);
        let mut batch_samples: Vec<&TrainingSample> = Vec::with_capacity(actual_batch_size);

        for _ in 0..actual_batch_size {
            if self.current_position < self.epoch_samples.len() {
                let idx = self.epoch_samples[self.current_position];
                batch_samples.push(&self.samples[idx]);
                self.current_position += 1;
            }
        }

        if batch_samples.is_empty() {
            return None;
        }

        Some(Batch::from_samples(&batch_samples))
    }
}

impl Batch {
    pub fn from_samples(samples: &[&TrainingSample]) -> Self {
        assert!(!samples.is_empty(), "Batch 不能为空");

        let max_seq_len = samples
            .iter()
            .map(|s| s.seq_len)
            .max()
            .unwrap_or(0);

        let batch_size = samples.len();
        let mut input_ids = Vec::with_capacity(batch_size);
        let mut labels = Vec::with_capacity(batch_size);
        let mut attention_mask = Vec::with_capacity(batch_size);
        let mut num_tokens = 0;

        for sample in samples {
            let effective_len = sample.seq_len.min(max_seq_len);

            let mut batch_input_ids = vec![0usize; max_seq_len];
            let mut batch_labels = vec![usize::MAX; max_seq_len];
            let mut batch_attention_mask = vec![0u8; max_seq_len];

            if effective_len > 0 {
                let copy_len = effective_len.min(sample.input_ids.len());
                batch_input_ids[..copy_len].copy_from_slice(&sample.input_ids[..copy_len]);

                let label_copy_len = effective_len.min(sample.labels.len());
                batch_labels[..label_copy_len].copy_from_slice(&sample.labels[..label_copy_len]);

                for i in 0..effective_len.min(sample.attention_mask.len()) {
                    batch_attention_mask[i] = sample.attention_mask[i];
                }

                num_tokens += effective_len;
            }

            input_ids.push(batch_input_ids);
            labels.push(batch_labels);
            attention_mask.push(batch_attention_mask);
        }

        Batch {
            input_ids,
            labels,
            attention_mask,
            batch_size,
            seq_len: max_seq_len,
            num_tokens,
        }
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_jsonl(dir: &TempDir, content: &str) -> PathBuf {
        let path = dir.path().join("test.jsonl");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "{}", content).unwrap();
        path
    }

    fn create_test_text(dir: &TempDir, content: &str) -> PathBuf {
        let path = dir.path().join("test.txt");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "{}", content).unwrap();
        path
    }

    #[test]
    fn test_load_jsonl_causal_lm() {
        let dir = TempDir::new().unwrap();
        let content = r#"{"text": "Hello world"}
{"text": "Test data"}"#;
        let path = create_test_jsonl(&dir, content);

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 1024,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 2,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let loader = DataLoader::new(config).unwrap();
        assert_eq!(loader.len(), 2);
    }

    #[test]
    fn test_batch_construction() {
        let samples = vec![
            TrainingSample {
                input_ids: vec![1, 2, 3, 0, 0],
                labels: vec![1, 2, 3, usize::MAX, usize::MAX],
                attention_mask: vec![1, 1, 1, 0, 0],
                seq_len: 3,
            },
            TrainingSample {
                input_ids: vec![4, 5, 0, 0, 0],
                labels: vec![4, 5, usize::MAX, usize::MAX, usize::MAX],
                attention_mask: vec![1, 1, 0, 0, 0],
                seq_len: 2,
            },
        ];

        let batch = Batch::from_samples(&samples.iter().collect::<Vec<_>>());

        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.seq_len, 3);
        assert_eq!(batch.num_tokens(), 5);
        assert_eq!(batch.input_ids.len(), 2);
        assert_eq!(batch.labels.len(), 2);
        assert_eq!(batch.attention_mask.len(), 2);
    }

    #[test]
    fn test_shuffle_deterministic() {
        let dir = TempDir::new().unwrap();
        let content = (0..10)
            .map(|i| format!(r#"{{"text": "Sample {}"}}"#, i))
            .collect::<Vec<_>>()
            .join("\n");
        let path = create_test_jsonl(&dir, &content);

        let make_loader = || {
            let config = DataLoaderConfig {
                train_path: path.clone(),
                validation_path: None,
                format: DataFormat::Jsonl,
                max_seq_length: 128,
                pad_token_id: 0,
                eos_token_id: 2,
                batch_size: 10,
                drop_last: false,
                shuffle: false, // 不在 new() 时 shuffle
                sft_config: None,
            };
            let mut loader = DataLoader::new(config).unwrap();
            loader.rng = rand::rngs::StdRng::seed_from_u64(42);
            loader.shuffle();
            loader
        };

        let loader1 = make_loader();
        let loader2 = make_loader();

        assert_eq!(
            loader1.epoch_samples,
            loader2.epoch_samples,
            "相同 seed 应产生相同的 shuffle 结果"
        );
    }

    #[test]
    fn test_iterator_full_epoch() {
        let dir = TempDir::new().unwrap();
        let content = (0..10)
            .map(|i| format!(r#"{{"text": "Sample {}"}}"#, i))
            .collect::<Vec<_>>()
            .join("\n");
        let path = create_test_jsonl(&dir, &content);

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 128,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 3,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let loader = DataLoader::new(config).unwrap();
        let batches: Vec<_> = loader.collect();

        assert_eq!(batches.len(), 4); // 3 + 3 + 3 + 1

        let total_samples: usize = batches.iter().map(|b| b.batch_size).sum();
        assert_eq!(total_samples, 10);
    }

    #[test]
    fn test_iterator_drop_last() {
        let dir = TempDir::new().unwrap();
        let content = (0..10)
            .map(|i| format!(r#"{{"text": "Sample {}"}}"#, i))
            .collect::<Vec<_>>()
            .join("\n");
        let path = create_test_jsonl(&dir, &content);

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 128,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 3,
            drop_last: true,
            shuffle: false,
            sft_config: None,
        };

        let loader = DataLoader::new(config).unwrap();
        let batches: Vec<_> = loader.collect();

        assert_eq!(batches.len(), 3); // 3 + 3 + 3，最后一个被丢弃

        for batch in &batches {
            assert_eq!(batch.batch_size, 3);
        }
    }

    #[test]
    fn test_pad_sequence() {
        let ids = vec![1, 2, 3];
        let (padded_ids, padded_labels, mask) = DataLoader::pad_sequence(&ids, 6, 0);

        assert_eq!(padded_ids, vec![1, 2, 3, 0, 0, 0]);
        assert_eq!(padded_labels, vec![1, 2, 3, usize::MAX, usize::MAX, usize::MAX]);
        assert_eq!(mask, vec![1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_sft_preprocessing() {
        let dir = TempDir::new().unwrap();
        let content = r#"{"instruction": "Translate to Chinese", "input": "Hello", "output": "你好"}"#;
        let path = create_test_jsonl(&dir, content);

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 256,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 1,
            drop_last: false,
            shuffle: false,
            sft_config: Some(SFTConfig {
                prompt_template: "Instruction: {instruction}\nInput: {input}\nResponse: ".to_string(),
                mask_prompt_loss: true,
                max_prompt_length: None,
                max_response_length: None,
            }),
        };

        let loader = DataLoader::new(config).unwrap();
        assert_eq!(loader.len(), 1);

        let batch = loader.into_iter().next().unwrap();
        assert_eq!(batch.batch_size, 1);

        let has_masked_labels = batch.labels[0]
            .iter()
            .any(|&label| label == usize::MAX);
        assert!(
            has_masked_labels,
            "mask_prompt_loss=true 时应有部分 label 为 -100"
        );
    }

    #[test]
    fn test_sft_without_masking() {
        let dir = TempDir::new().unwrap();
        let content = r#"{"instruction": "Summarize", "output": "Summary here"}"#;
        let path = create_test_jsonl(&dir, content);

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 256,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 1,
            drop_last: false,
            shuffle: false,
            sft_config: Some(SFTConfig {
                prompt_template: "{instruction}: ".to_string(),
                mask_prompt_loss: false,
                max_prompt_length: None,
                max_response_length: None,
            }),
        };

        let loader = DataLoader::new(config).unwrap();
        let batch = loader.into_iter().next().unwrap();

        let all_valid = batch.labels[0]
            .iter()
            .zip(batch.attention_mask[0].iter())
            .all(|(&label, &mask)| mask == 0 || label != usize::MAX);
        assert!(
            all_valid,
            "mask_prompt_loss=false 时所有有效位置不应有 masked label"
        );
    }

    #[test]
    fn test_empty_file_handling() {
        let dir = TempDir::new().unwrap();
        let path = create_test_jsonl(&dir, "");

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 128,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 2,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let result = DataLoader::new(config);
        assert!(
            result.is_err(),
            "空文件应返回错误"
        );
        matches!(result.unwrap_err(), DataLoaderError::EmptyDataset);
    }

    #[test]
    fn test_max_seq_length_truncation() {
        let long_text = "A".repeat(2000);
        let dir = TempDir::new().unwrap();
        let content = format!(r#"{{"text": "{}"}}"#, long_text);
        let path = create_test_jsonl(&dir, &content);

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 512,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 1,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let loader = DataLoader::new(config).unwrap();
        let batch = loader.into_iter().next().unwrap();

        assert_eq!(
            batch.seq_len, 512,
            "序列长度应被截断到 max_seq_length"
        );

        let valid_tokens: usize = batch.attention_mask[0].iter().map(|&x| x as usize).sum();
        assert_eq!(
            valid_tokens, 512,
            "有效 token 数应为 512"
        );
    }

    #[test]
    fn test_load_text_format() {
        let dir = TempDir::new().unwrap();
        let content = "Line 1\nLine 2\nLine 3\n";
        let path = create_test_text(&dir, content);

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Text,
            max_seq_length: 128,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 3,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let loader = DataLoader::new(config).unwrap();
        assert_eq!(loader.len(), 3);

        let batch = loader.into_iter().next().unwrap();
        assert_eq!(batch.batch_size, 3);
    }

    #[test]
    fn test_file_not_found_error() {
        let config = DataLoaderConfig {
            train_path: PathBuf::from("/nonexistent/path/data.jsonl"),
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 128,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 2,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let result = DataLoader::new(config);
        assert!(result.is_err());
        matches!(result.unwrap_err(), DataLoaderError::FileNotFound(_));
    }

    #[test]
    fn test_reset_epoch() {
        let dir = TempDir::new().unwrap();
        let content = (0..5)
            .map(|i| format!(r#"{{"text": "Sample {}"}}"#, i))
            .collect::<Vec<_>>()
            .join("\n");
        let path = create_test_jsonl(&dir, &content);

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 128,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 5,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let mut loader = DataLoader::new(config).unwrap();

        let _batch1 = loader.next();
        assert_eq!(loader.current_position, 5);

        loader.reset_epoch();
        assert_eq!(loader.current_position, 0, "reset_epoch 后位置应重置");

        let batch2 = loader.next();
        assert!(batch2.is_some(), "重置后应能继续迭代");
    }

    #[test]
    fn test_parse_error_reporting() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("invalid.jsonl");
        let mut file = File::create(&path).unwrap();
        writeln!(file, r#"{{"text": "valid"}}"#).unwrap();
        writeln!(file, r#"{{invalid json}}"#).unwrap();

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 128,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 2,
            drop_last: false,
            shuffle: false,
            sft_config: None,
        };

        let result = DataLoader::new(config);
        assert!(result.is_err());
        if let Err(DataLoaderError::ParseError { line, .. }) = result {
            assert_eq!(line, 2, "应在第 2 行报告错误");
        }
    }

    #[test]
    fn test_batch_with_different_lengths() {
        let samples = vec![
            TrainingSample {
                input_ids: vec![1, 2],
                labels: vec![1, 2],
                attention_mask: vec![1, 1],
                seq_len: 2,
            },
            TrainingSample {
                input_ids: vec![3, 4, 5, 6],
                labels: vec![3, 4, 5, 6],
                attention_mask: vec![1, 1, 1, 1],
                seq_len: 4,
            },
            TrainingSample {
                input_ids: vec![7],
                labels: vec![7],
                attention_mask: vec![1],
                seq_len: 1,
            },
        ];

        let batch = Batch::from_samples(&samples.iter().collect::<Vec<_>>());

        assert_eq!(batch.batch_size, 3);
        assert_eq!(batch.seq_len, 4, "Batch 长度应等于最长样本");
        assert_eq!(batch.num_tokens(), 7, "总有效 token 数应为 7");

        assert_eq!(batch.input_ids[0].len(), 4);
        assert_eq!(batch.input_ids[1].len(), 4);
        assert_eq!(batch.input_ids[2].len(), 4);
    }

    #[test]
    fn test_sft_max_lengths() {
        let dir = TempDir::new().unwrap();
        let content = r#"{"instruction": "Long instruction repeated many times", "output": "Short output"}"#;
        let path = create_test_jsonl(&dir, content);

        let config = DataLoaderConfig {
            train_path: path,
            validation_path: None,
            format: DataFormat::Jsonl,
            max_seq_length: 100,
            pad_token_id: 0,
            eos_token_id: 2,
            batch_size: 1,
            drop_last: false,
            shuffle: false,
            sft_config: Some(SFTConfig {
                prompt_template: "{instruction}: ".to_string(),
                mask_prompt_loss: true,
                max_prompt_length: Some(10),
                max_response_length: Some(20),
            }),
        };

        let loader = DataLoader::new(config).unwrap();
        let batch = loader.into_iter().next().unwrap();

        assert!(
            batch.seq_len <= 100,
            "总长度不应超过 max_seq_length"
        );
    }
}
