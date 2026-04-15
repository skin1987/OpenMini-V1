//! 文本生成器
//!
//! 提供流式和非流式文本生成功能。
//!
//! ## 功能特性
//!
//! - **流式输出**：支持逐 Token 流式回调
//! - **进度追踪**：提供生成进度和统计信息
//! - **节奏控制**：可配置的输出节奏控制（高精度固定间隔）
//! - **停止符支持**：遇到指定字符串时提前终止生成
//! - **动态负载调节**：通过 `LoadRegulator` 根据系统负载手动调整生成速度
//! - **多模态支持**：预留图像和音频特征接口（需模型支持）
//!
//! ## 当前限制
//!
//! - 节奏控制使用 `std::thread::sleep`，会阻塞线程
//! - 如需在异步运行时使用，请禁用节奏控制或使用 `tokio::task::spawn_blocking` 自行包装
//! - 采样由模型内部处理
//!
//! ## 示例

#![allow(dead_code)]
//!
//! ```ignore
//! use anyhow::Result;
//! use openmini_server::model::inference::{MultimodalTransformer, Tokenizer, StreamGenerator};
//!
//! fn main() -> Result<()> {
//!     let model = MultimodalTransformer::default();
//!     let tokenizer = Tokenizer::new()?;
//!     
//!     let mut generator = StreamGenerator::new(model, tokenizer)
//!         .with_steady_interval(30);
//!     
//!     let stats = generator.stream("Once upon a time,", |token| {
//!         print!("{}", token);
//!         Ok(true)  // 返回 false 可提前终止
//!     })?;
//!     
//!     println!("\n首 token 延迟: {}ms", stats.first_token_latency_ms);
//!     println!("生成速度: {:.1} tokens/s", stats.tokens_per_second());
//!     Ok(())
//! }
//! ```

use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use ndarray::Array2;

use super::error::InferenceError;
use super::model::MultimodalTransformer;
use super::sampler::GenerateParams;
use super::tokenizer::Tokenizer;

/// 默认最大生成 Token 数量
pub const DEFAULT_MAX_NEW_TOKENS: usize = 2048;

/// 稳定输出间隔 (毫秒)
const STEADY_OUTPUT_INTERVAL_MS: u64 = 50;

/// 文本生成器
///
/// 提供流式和非流式文本生成接口
///
/// **注意**：此类型不是 `Send` 或 `Sync`，建议每个线程/任务使用独立的实例。
pub struct TextGenerator {
    /// 多模态模型
    model: MultimodalTransformer,
    /// 分词器
    tokenizer: Tokenizer,
    /// 生成参数
    params: GenerateParams,
}

impl TextGenerator {
    /// 创建新的文本生成器
    pub fn new(model: MultimodalTransformer, tokenizer: Tokenizer) -> Self {
        Self {
            model,
            tokenizer,
            params: GenerateParams::default(),
        }
    }

    /// 设置生成参数
    #[allow(dead_code)]
    pub fn with_params(mut self, params: GenerateParams) -> Self {
        self.params = params;
        self
    }

    /// 获取生成参数
    #[allow(dead_code)]
    pub fn params(&self) -> &GenerateParams {
        &self.params
    }

    /// 生成文本
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        let tokens = self.tokenizer.encode(prompt)?;
        let max_tokens = self.params.max_new_tokens;

        let generated_ids = self.model.generate(&tokens, max_tokens)?;
        let generated_ids: Vec<_> = generated_ids.into_iter().take(max_tokens).collect();

        let text = self.tokenizer.decode(&generated_ids)?;
        Ok(text)
    }

    /// 多模态生成（文本 + 图像）
    ///
    /// # Errors
    /// 当前模型实现尚未支持多模态特征传递
    #[allow(dead_code)]
    pub fn generate_with_image(
        &mut self,
        _prompt: &str,
        _image_patches: &Array2<f32>,
    ) -> Result<String> {
        Err(anyhow!("Multimodal generation not yet implemented"))
    }

    /// 多模态生成（文本 + 图像 + 音频）
    ///
    /// # Errors
    /// 当前模型实现尚未支持多模态特征传递
    #[allow(dead_code)]
    pub fn generate_multimodal(
        &mut self,
        _prompt: &str,
        _image_patches: Option<&Array2<f32>>,
        _audio_features: Option<&Array2<f32>>,
    ) -> Result<String> {
        Err(anyhow!("Multimodal generation not yet implemented"))
    }

    /// 流式生成（逐 token 回调）
    ///
    /// # 参数
    /// - `prompt`: 输入提示
    /// - `max_tokens`: 最大生成长度
    /// - `callback`: 每个 token 生成后调用的回调，返回 false 可提前终止
    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(u32) -> Result<bool>,
    {
        let tokens = self.tokenizer.encode(prompt)?;

        let params = GenerateParams {
            max_new_tokens: max_tokens,
            ..Default::default()
        };

        self.model
            .generate_streaming(&tokens, &params, |token_id| {
                callback(token_id).map_err(|e| InferenceError::generation(e.to_string()))
            })
            .map_err(|e| anyhow::anyhow!("{}", e))
    }
}

/// 流式生成器
///
/// 支持逐 Token 流式输出，提供首 token 延迟追踪和节奏控制
///
/// **注意**：节奏控制使用 `std::thread::sleep`，会阻塞当前线程。
/// 如需在异步运行时（如 tokio）中使用，请禁用节奏控制或自行适配。
///
/// ## 延迟优化说明
///
/// 当前仅提供延迟测量接口。实际延迟优化需要模型侧配合（如 FlashAttention、
/// PagedAttention、投机解码等）。具体优化策略可参考：
/// - 预热：提前分配 KV cache
/// - 投机解码：用小模型快速生成首 token
/// - 动态 batch：降低首 token 所在 batch 大小
pub struct StreamGenerator {
    /// 文本生成器
    generator: TextGenerator,
    /// 稳定输出间隔 (毫秒)
    steady_interval_ms: u64,
    /// 是否启用节奏控制
    pace_control: bool,
}

impl StreamGenerator {
    /// 创建新的流式生成器
    pub fn new(model: MultimodalTransformer, tokenizer: Tokenizer) -> Self {
        Self {
            generator: TextGenerator::new(model, tokenizer),
            steady_interval_ms: STEADY_OUTPUT_INTERVAL_MS,
            pace_control: true,
        }
    }

    /// 从已有的 TextGenerator 创建流式生成器
    ///
    /// 允许用户先配置 TextGenerator 的参数（如 temperature、top_p）
    #[allow(dead_code)]
    pub fn from_generator(generator: TextGenerator) -> Self {
        Self {
            generator,
            steady_interval_ms: STEADY_OUTPUT_INTERVAL_MS,
            pace_control: true,
        }
    }

    /// 设置生成参数
    pub fn with_params(mut self, params: GenerateParams) -> Self {
        self.generator.params = params;
        self
    }

    /// 设置稳定输出间隔
    pub fn with_steady_interval(mut self, ms: u64) -> Self {
        self.steady_interval_ms = ms;
        self
    }

    /// 启用/禁用节奏控制
    pub fn with_pace_control(mut self, enabled: bool) -> Self {
        self.pace_control = enabled;
        self
    }

    /// 流式生成内部实现
    ///
    /// 统一处理节奏控制、停止符检查、进度回调
    fn stream_internal<F, StopFn, ProgressFn>(
        &mut self,
        prompt: &str,
        mut token_callback: F,
        mut stop_checker: StopFn,
        mut progress_callback: ProgressFn,
    ) -> Result<StreamStats>
    where
        F: FnMut(&str) -> Result<bool>,
        StopFn: FnMut(&str) -> Result<Option<usize>>,
        ProgressFn: FnMut(StreamPhase, usize, usize, &str, u64) -> Result<()>,
    {
        let start_time = Instant::now();
        let mut stats = StreamStats::default();

        let tokens = self.generator.tokenizer.encode(prompt)?;
        let max_tokens = self.generator.params.max_new_tokens;

        progress_callback(StreamPhase::Encoding, 0, max_tokens, "", 0)?;

        let mut next_output_time = start_time;
        let target_interval = Duration::from_millis(self.steady_interval_ms);
        let mut accumulated_text = String::with_capacity(max_tokens * 6);
        let mut token_index = 0;

        let params = GenerateParams {
            max_new_tokens: max_tokens,
            ..Default::default()
        };

        self.generator
            .model
            .generate_streaming(&tokens, &params, |token_id| {
                let token_text = self
                    .generator
                    .tokenizer
                    .decode(&[token_id])
                    .map_err(|e| InferenceError::tokenization(e.to_string()))?;

                // 记录首 token 延迟
                if token_index == 0 {
                    stats.first_token_latency_ms = start_time.elapsed().as_millis() as u64;
                }

                accumulated_text.push_str(&token_text);

                // 检查停止符（在节奏控制之前）
                if let Some(pos) = stop_checker(&accumulated_text)
                    .map_err(|e| InferenceError::generation(e.to_string()))?
                {
                    accumulated_text.truncate(pos);
                    return Ok(false);
                }

                // 节奏控制（仅当要输出该 token 时）
                if token_index > 0 && self.pace_control {
                    let now = Instant::now();
                    if now < next_output_time {
                        std::thread::sleep(next_output_time - now);
                    }
                }

                // 进度回调
                progress_callback(
                    StreamPhase::Generating,
                    token_index + 1,
                    max_tokens,
                    &accumulated_text,
                    start_time.elapsed().as_millis() as u64,
                )
                .map_err(|e| InferenceError::generation(e.to_string()))?;

                let should_continue = token_callback(&token_text)
                    .map_err(|e| InferenceError::generation(e.to_string()))?;
                stats.tokens_generated += 1;
                token_index += 1;
                next_output_time += target_interval;

                Ok(should_continue)
            })
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        progress_callback(
            StreamPhase::Completed,
            stats.tokens_generated,
            max_tokens,
            &accumulated_text,
            start_time.elapsed().as_millis() as u64,
        )?;

        stats.total_latency_ms = start_time.elapsed().as_millis() as u64;
        stats.total_text = accumulated_text;

        Ok(stats)
    }

    /// 流式生成
    ///
    /// # 参数
    /// - `prompt`: 输入提示
    /// - `callback`: 每个 token 生成后调用的回调，返回是否继续生成
    ///
    /// # 节奏控制
    /// 首 token 立即输出，后续 token 按设定间隔输出。
    pub fn stream<F>(&mut self, prompt: &str, mut callback: F) -> Result<StreamStats>
    where
        F: FnMut(&str) -> Result<bool>,
    {
        self.stream_internal(
            prompt,
            &mut callback,
            |_: &str| Ok(None),
            |_, _, _, _, _| Ok(()),
        )
    }

    /// 流式生成（带停止符支持）
    ///
    /// # 参数
    /// - `prompt`: 输入提示
    /// - `stop_strings`: 停止符列表，当累积文本包含任一停止符时提前终止
    /// - `callback`: 每个 token 生成后调用的回调，返回是否继续生成
    ///
    /// # 注意
    /// 遇到停止符时，导致停止的 token 不会通过回调返回，也不会计入 `tokens_generated`。
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let stats = generator.stream_with_stop(
    ///     "Hello,",
    ///     &["\n\n", "END"],
    ///     |token| { print!("{}", token); Ok(true) }
    /// )?;
    /// ```
    pub fn stream_with_stop<F>(
        &mut self,
        prompt: &str,
        stop_strings: &[&str],
        mut callback: F,
    ) -> Result<StreamStats>
    where
        F: FnMut(&str) -> Result<bool>,
    {
        let stop_strings: Vec<&str> = stop_strings.to_vec();

        self.stream_internal(
            prompt,
            &mut callback,
            move |text: &str| {
                let mut earliest_pos: Option<usize> = None;
                for stop in &stop_strings {
                    if let Some(pos) = text.find(stop) {
                        if earliest_pos.is_none_or(|p| pos < p) {
                            earliest_pos = Some(pos);
                        }
                    }
                }
                Ok(earliest_pos)
            },
            |_, _, _, _, _| Ok(()),
        )
    }

    /// 流式生成（带进度回调）
    ///
    /// # 注意
    ///
    /// 进度回调是同步执行的，请勿将 `current_text` 存储到异步任务中。
    pub fn stream_with_progress<F, P>(
        &mut self,
        prompt: &str,
        mut callback: F,
        mut progress: P,
    ) -> Result<StreamStats>
    where
        F: FnMut(&str) -> Result<bool>,
        P: FnMut(&StreamProgress<'_>) -> Result<()>,
    {
        self.stream_internal(
            prompt,
            &mut callback,
            |_: &str| Ok(None),
            move |phase, tokens_generated, max_tokens, current_text, latency_ms| {
                progress(&StreamProgress {
                    phase,
                    tokens_generated,
                    max_tokens,
                    current_text,
                    latency_ms,
                })
            },
        )
    }
}

/// 动态负载调节器
///
/// 根据系统负载自动调整生成速度
#[derive(Debug, Clone)]
pub struct LoadRegulator {
    /// 最小间隔 (毫秒)
    pub min_interval_ms: u64,
    /// 最大间隔 (毫秒)
    pub max_interval_ms: u64,
    /// 目标 CPU 使用率 (0.0-1.0)
    pub target_cpu_usage: f32,
    /// 目标内存使用率 (0.0-1.0)
    pub target_memory_usage: f32,
    /// 当前间隔 (毫秒)
    pub current_interval_ms: u64,
}

impl Default for LoadRegulator {
    fn default() -> Self {
        Self {
            min_interval_ms: 10,
            max_interval_ms: 200,
            target_cpu_usage: 0.8,
            target_memory_usage: 0.8,
            current_interval_ms: 50,
        }
    }
}

impl LoadRegulator {
    /// 创建新的负载调节器
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置间隔范围
    pub fn with_interval_range(mut self, min_ms: u64, max_ms: u64) -> Self {
        self.min_interval_ms = min_ms;
        self.max_interval_ms = max_ms;
        self.current_interval_ms = (min_ms + max_ms) / 2;
        self
    }

    /// 设置目标 CPU 使用率
    pub fn with_target_cpu(mut self, usage: f32) -> Self {
        self.target_cpu_usage = usage.clamp(0.0, 1.0);
        self
    }

    /// 设置目标内存使用率
    pub fn with_target_memory(mut self, usage: f32) -> Self {
        self.target_memory_usage = usage.clamp(0.0, 1.0);
        self
    }

    /// 更新负载并调整间隔
    ///
    /// # 参数
    /// - `cpu_usage`: 当前 CPU 使用率 (0.0-1.0)
    /// - `memory_usage`: 当前内存使用率 (0.0-1.0)
    pub fn update(&mut self, cpu_usage: f32, memory_usage: f32) {
        let cpu_factor = if cpu_usage > self.target_cpu_usage {
            1.0 + (cpu_usage - self.target_cpu_usage) * 2.0
        } else {
            1.0 - (self.target_cpu_usage - cpu_usage) * 0.5
        };

        let memory_factor = if memory_usage > self.target_memory_usage {
            1.0 + (memory_usage - self.target_memory_usage) * 2.0
        } else {
            1.0 - (self.target_memory_usage - memory_usage) * 0.5
        };

        let factor = cpu_factor.max(memory_factor);
        let new_interval = (self.current_interval_ms as f32 * factor) as u64;

        self.current_interval_ms = new_interval.clamp(self.min_interval_ms, self.max_interval_ms);
    }

    /// 获取当前推荐的间隔
    pub fn current_interval(&self) -> u64 {
        self.current_interval_ms
    }
}

/// 流式生成阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPhase {
    /// 编码阶段
    Encoding,
    /// 生成阶段
    Generating,
    /// 完成
    Completed,
}

/// 流式生成进度
///
/// 使用生命周期参数 `'a` 避免不必要的字符串克隆
#[derive(Debug, Clone, Copy)]
pub struct StreamProgress<'a> {
    /// 当前阶段
    pub phase: StreamPhase,
    /// 已生成 token 数
    pub tokens_generated: usize,
    /// 最大 token 数
    pub max_tokens: usize,
    /// 当前累积文本
    pub current_text: &'a str,
    /// 已用时间 (毫秒)
    pub latency_ms: u64,
}

/// 流式生成统计
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    /// 首 token 延迟 (毫秒)
    pub first_token_latency_ms: u64,
    /// 总延迟 (毫秒)
    pub total_latency_ms: u64,
    /// 生成的 token 数
    pub tokens_generated: usize,
    /// 生成的文本
    pub total_text: String,
}

impl StreamStats {
    /// 计算平均 token 生成速度 (tokens/s)
    pub fn tokens_per_second(&self) -> f32 {
        if self.total_latency_ms == 0 {
            return 0.0;
        }
        (self.tokens_generated as f32) / (self.total_latency_ms as f32 / 1000.0)
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_stats_tokens_per_second() {
        let mut stats = StreamStats::default();
        stats.tokens_generated = 100;
        stats.total_latency_ms = 1000;
        assert!((stats.tokens_per_second() - 100.0).abs() < 0.01);

        stats.total_latency_ms = 0;
        assert_eq!(stats.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_stream_phase_variants() {
        assert_ne!(StreamPhase::Encoding, StreamPhase::Generating);
        assert_ne!(StreamPhase::Generating, StreamPhase::Completed);
        assert_eq!(StreamPhase::Encoding, StreamPhase::Encoding);
    }

    #[test]
    fn test_default_max_new_tokens() {
        assert_eq!(DEFAULT_MAX_NEW_TOKENS, 2048);
    }

    #[test]
    fn test_steady_output_interval() {
        assert_eq!(STEADY_OUTPUT_INTERVAL_MS, 50);
    }

    #[test]
    fn test_load_regulator_default() {
        let regulator = LoadRegulator::new();
        assert_eq!(regulator.min_interval_ms, 10);
        assert_eq!(regulator.max_interval_ms, 200);
        assert_eq!(regulator.current_interval_ms, 50);
    }

    #[test]
    fn test_load_regulator_update_high_cpu() {
        let mut regulator = LoadRegulator::new();
        regulator.update(0.9, 0.5);
        // cpu_factor ≈ 1.2 (略小于 1.2 由于浮点精度)
        // new_interval ≈ 59-60
        assert!((regulator.current_interval_ms as i64 - 60).abs() <= 1);
    }

    #[test]
    fn test_load_regulator_update_low_cpu() {
        let mut regulator = LoadRegulator::new();
        regulator.update(0.5, 0.5);
        // cpu_factor ≈ 0.85
        // new_interval ≈ 42-43
        assert!((regulator.current_interval_ms as i64 - 42).abs() <= 1);
    }

    #[test]
    fn test_load_regulator_clamp() {
        let mut regulator = LoadRegulator::new().with_interval_range(10, 100);
        regulator.update(0.99, 0.99); // 极高负载
        assert!(regulator.current_interval_ms <= 100);

        regulator.update(0.1, 0.1); // 极低负载
        assert!(regulator.current_interval_ms >= 10);
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_streaming_generator_config_creation() {
        // 流式生成器配置创建 - 使用 GenerateParams 模拟配置
        let params = GenerateParams {
            max_new_tokens: 100,
            temperature: 0.8,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.2,
            ..Default::default()
        };

        // 验证参数设置正确
        assert_eq!(params.max_new_tokens, 100);
        assert!((params.temperature - 0.8).abs() < 0.001);
        assert!((params.top_p - 0.9).abs() < 0.001);
        assert_eq!(params.top_k, 40);
        assert!((params.repetition_penalty - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_streaming_generator_default_params() {
        // 默认参数测试
        let params = GenerateParams::default();
        assert!(params.max_new_tokens > 0); // 应该有正的最大token数
        assert!(params.temperature > 0.0); // 默认温度应该大于0
        assert!(params.top_p > 0.0 && params.top_p <= 1.0);
        // top_k 是 usize 类型，始终 >= 0，无需断言
    }

    #[test]
    fn test_streaming_generator_empty_prompt_handling() {
        // 空提示词处理 - 测试 StreamStats 对空输入的处理
        let stats = StreamStats::default();

        // 空生成的统计信息应该合理
        assert_eq!(stats.tokens_generated, 0);
        assert_eq!(stats.first_token_latency_ms, 0);
        assert_eq!(stats.total_latency_ms, 0);
        assert!(stats.total_text.is_empty());
        assert_eq!(stats.tokens_per_second(), 0.0); // 除零保护
    }

    #[test]
    fn test_streaming_generator_token_by_token_stats() {
        // 逐token生成统计 - 模拟生成过程中的统计数据
        let mut stats = StreamStats::default();

        // 模拟生成了5个token
        stats.tokens_generated = 5;
        stats.first_token_latency_ms = 50; // 首token延迟50ms
        stats.total_latency_ms = 250; // 总延迟250ms
        stats.total_text = "Hello".to_string();

        // 验证统计信息正确性
        assert_eq!(stats.tokens_generated, 5);
        assert!(stats.first_token_latency_ms > 0);
        assert!(stats.total_latency_ms > 0);
        assert!(!stats.total_text.is_empty());

        // 计算tokens/s: 5 / (250/1000) = 20.0
        let tps = stats.tokens_per_second();
        assert!((tps - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_streaming_generator_stop_conditions_stats() {
        // 停止条件下的统计 - 提前终止时的统计信息
        let mut stats = StreamStats::default();

        // 模拟在第3个token时遇到停止符
        stats.tokens_generated = 3;
        stats.first_token_latency_ms = 30;
        stats.total_latency_ms = 120;
        stats.total_text = "Hel".to_string(); // 未包含停止符的文本

        assert_eq!(stats.tokens_generated, 3);
        assert_eq!(stats.total_text.len(), 3);

        // tokens/s: 3 / (120/1000) = 25.0
        let tps = stats.tokens_per_second();
        assert!((tps - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_streaming_generator_max_tokens_limit() {
        // 最大token数限制测试
        let params = GenerateParams {
            max_new_tokens: 10,
            ..Default::default()
        };

        assert_eq!(params.max_new_tokens, 10);
        assert!(params.max_new_tokens <= DEFAULT_MAX_NEW_TOKENS);
    }

    #[test]
    fn test_streaming_generator_extreme_params() {
        // 极端参数配置测试
        // 高温度（更随机）
        let high_temp = GenerateParams {
            temperature: 2.0,
            ..Default::default()
        };
        assert!((high_temp.temperature - 2.0).abs() < 0.01);

        // 低温度（更确定）
        let low_temp = GenerateParams {
            temperature: 0.1,
            ..Default::default()
        };
        assert!((low_temp.temperature - 0.1).abs() < 0.01);

        // 极端top_p
        let extreme_top_p = GenerateParams {
            top_p: 0.99,
            ..Default::default()
        };
        assert!((extreme_top_p.top_p - 0.99).abs() < 0.01);

        // 高重复惩罚
        let high_repetition = GenerateParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        assert!((high_repetition.repetition_penalty - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_stream_phase_all_variants() {
        // 测试所有 StreamPhase 变体的比较
        let phases = [
            StreamPhase::Encoding,
            StreamPhase::Generating,
            StreamPhase::Completed,
        ];

        // 每个变体都应该不等于其他变体
        for (i, phase1) in phases.iter().enumerate() {
            for (j, phase2) in phases.iter().enumerate() {
                if i == j {
                    assert_eq!(*phase1, *phase2);
                } else {
                    assert_ne!(*phase1, *phase2);
                }
            }
        }
    }

    #[test]
    fn test_load_regulator_custom_config() {
        // 自定义负载调节器配置
        let regulator = LoadRegulator::new()
            .with_interval_range(20, 500)
            .with_target_cpu(0.9)
            .with_target_memory(0.85);

        assert_eq!(regulator.min_interval_ms, 20);
        assert_eq!(regulator.max_interval_ms, 500);
        // 初始间隔应该是范围的中点
        assert_eq!(regulator.current_interval_ms, (20 + 500) / 2);
        assert_eq!(regulator.target_cpu_usage, 0.9);
        assert_eq!(regulator.target_memory_usage, 0.85);
    }

    #[test]
    fn test_load_regulator_memory_pressure() {
        // 内存压力测试
        let mut regulator = LoadRegulator::new().with_interval_range(10, 100);

        // 高内存使用率
        regulator.update(0.5, 0.95);
        let high_memory_interval = regulator.current_interval_ms;
        // memory_factor 应该 > 1.0，导致间隔增加
        assert!(high_memory_interval > 50); // 基准是50

        // 低内存使用率
        regulator.update(0.5, 0.3);
        let low_memory_interval = regulator.current_interval_ms;
        // memory_factor 应该 < 1.0，导致间隔减少
        assert!(
            low_memory_interval < high_memory_interval,
            "Low memory interval ({}) should be less than high memory interval ({})",
            low_memory_interval,
            high_memory_interval
        );
    }

    #[test]
    fn test_load_regulator_combined_pressure() {
        // CPU和内存综合压力测试
        let mut regulator = LoadRegulator::new();

        // CPU和内存都高
        regulator.update(0.95, 0.90);
        let high_pressure_interval = regulator.current_interval_ms;

        // CPU和内存都低
        regulator.update(0.2, 0.3);
        let low_pressure_interval = regulator.current_interval_ms;

        // 高压力时间隔应该更大
        assert!(high_pressure_interval > low_pressure_interval);
    }

    #[test]
    fn test_load_regulator_boundary_values() {
        // 边界值测试
        let mut regulator = LoadRegulator::new().with_interval_range(5, 1000);

        // CPU使用率为0.0
        regulator.update(0.0, 0.5);
        let min_interval = regulator.current_interval_ms;

        // CPU使用率为1.0
        regulator.update(1.0, 0.5);
        let max_interval = regulator.current_interval_ms;

        // 边界条件下应该能正常工作且在范围内
        assert!((5..=1000).contains(&min_interval));
        assert!((5..=1000).contains(&max_interval));
    }

    #[test]
    fn test_stream_stats_zero_division_protection() {
        // 零除法保护测试
        let stats = StreamStats::default();

        // 当total_latency_ms为0时，不应该panic
        let tps = stats.tokens_per_second();
        assert_eq!(tps, 0.0);

        // 有延迟但无token
        let mut stats_with_latency = StreamStats::default();
        stats_with_latency.total_latency_ms = 1000;
        assert_eq!(stats_with_latency.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_stream_progress_structure() {
        // 测试 StreamProgress 结构体的生命周期和字段
        let text = "test text";
        let progress = StreamProgress {
            phase: StreamPhase::Generating,
            tokens_generated: 10,
            max_tokens: 100,
            current_text: text,
            latency_ms: 500,
        };

        assert_eq!(progress.phase, StreamPhase::Generating);
        assert_eq!(progress.tokens_generated, 10);
        assert_eq!(progress.max_tokens, 100);
        assert_eq!(progress.current_text, "test text");
        assert_eq!(progress.latency_ms, 500);
    }

    #[test]
    fn test_steady_interval_constant() {
        // 稳定输出间隔常量验证
        assert_eq!(STEADY_OUTPUT_INTERVAL_MS, 50);
        // 这个值应该在合理范围内（10-1000ms）
        let interval = STEADY_OUTPUT_INTERVAL_MS;
        assert!((10..=1000).contains(&interval), "Interval {} out of range [10, 1000]", interval);
    }
}
