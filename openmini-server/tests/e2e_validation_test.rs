//! 端到端推理验证测试 (Legacy - 参考版本)
//!
//! 使用真实GGUF模型验证完整的推理流程：
//! 1. 模型加载 (GGUF)
//! 2. Tokenization
//! 3. 推理执行
//! 4. 输出生成与验证
//!
//! ⚠️ 此文件为参考实现，部分功能待完善
//! ✅ 推荐使用 e2e_validation_simple.rs 进行日常验证

#![allow(dead_code)]
#![allow(unused_imports)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

// 用于异步测试的运行时 (Legacy兼容)
#[allow(dead_code)]
static TOKIO_RT: std::sync::OnceLock<tokio::runtime::Runtime> = 
    std::sync::OnceLock::new();

fn tokio_runtime() -> &'static tokio::runtime::Runtime {
    TOKIO_RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime")
    })
}

use openmini_server::model::inference::{
    model::{ModelConfig, MultimodalTransformer},
    sampler::{GenerateParams, Sampler},
    tokenizer::Tokenizer,
    InferenceEngine,
};

/// 端到端测试结果报告
#[derive(Debug)]
pub struct E2ETestResult {
    pub test_name: String,
    pub backend_name: String,
    pub success: bool,
    pub error_message: Option<String>,
    pub stats: Option<InferenceStats>,
    pub output_text: Option<String>,
    pub duration_ms: u64,
}

impl std::fmt::Display for E2ETestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.success { "✅ PASS" } else { "❌ FAIL" };
        write!(
            f,
            "[{}] {} ({}) - {}ms",
            status, self.test_name, self.backend_name, self.duration_ms
        )?;

        if let Some(ref stats) = self.stats {
            write!(
                f,
                "\n  Tokens: {} | Speed: {:.1} t/s",
                stats.total_tokens, stats.tokens_per_second
            )?;
        }

        if let Some(ref err) = self.error_message {
            write!(f, "\n  Error: {}", err)?;
        }

        Ok(())
    }
}

/// 端到端测试引擎
pub struct E2ETestEngine {
    model_path: PathBuf,
    test_prompts: Vec<String>,
    max_new_tokens: usize,
    results: Vec<E2ETestResult>,
}

impl E2ETestEngine {
    /// 创建新的端到端测试引擎
    ///
    /// # 参数
    /// - `model_path`: GGUF模型文件路径
    /// - `test_prompts`: 测试提示词列表
    /// - `max_new_tokens`: 最大生成token数
    pub fn new(
        model_path: impl AsRef<Path>,
        test_prompts: Vec<String>,
        max_new_tokens: usize,
    ) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            test_prompts,
            max_new_tokens,
            results: Vec::new(),
        }
    }

    /// 运行所有端到端测试
    pub fn run_all_tests(&mut self) -> Vec<E2ETestResult> {
        println!("\n🚀 开始端到端验证测试");
        println!("========================================");
        println!("模型: {:?}", self.model_path);
        println!("提示词数: {}", self.test_prompts.len());
        println!("最大生成tokens: {}", self.max_new_tokens);
        println!("========================================\n");

        // Test 1: 模型加载验证
        self.run_test("模型加载验证", || self.test_model_loading());

        // Test 2: Tokenization验证
        self.run_test("Tokenization验证", || self.test_tokenization());

        // Test 3: 单轮文本生成
        for (i, prompt) in self.test_prompts.iter().enumerate() {
            let name = format!(
                "文本生成[{}]: {}",
                i + 1,
                prompt.chars().take(50).collect::<String>()
            );
            let p = prompt.clone();
            self.run_test(&name, move || self.test_single_generation(&p));
        }

        // Test 4: 批量生成性能
        if !self.test_prompts.is_empty() {
            self.run_test("批量生成性能", || self.test_batch_generation());
        }

        // Test 5: 不同采样策略
        self.run_test("贪婪解码", || self.test_greedy_decoding());
        self.run_test("温度采样(0.7)", || self.test_temperature_sampling(0.7));

        // 输出总结报告
        self.print_summary();

        self.results.clone()
    }

    /// 运行单个测试并记录结果
    fn run_test<F>(&mut self, name: &str, test_fn: F)
    where
        F: FnOnce() -> Result<E2ETestResult, String>,
    {
        print!("  测试: {:.<60}", name);

        let start = Instant::now();

        match test_fn() {
            Ok(mut result) => {
                result.duration_ms = start.elapsed().as_millis() as u64;
                self.results.push(result.clone());

                if result.success {
                    println!("✅ PASS ({}ms)", result.duration_ms);
                } else {
                    println!("❌ FAIL ({}ms)", result.duration_ms);
                    if let Some(ref err) = result.error_message {
                        println!("       错误: {}", err);
                    }
                }
            }
            Err(err) => {
                let duration = start.elapsed().as_millis() as u64;
                let result = E2ETestResult {
                    test_name: name.to_string(),
                    backend_name: "Unknown".to_string(),
                    success: false,
                    error_message: Some(err),
                    stats: None,
                    output_text: None,
                    duration_ms: duration,
                };
                self.results.push(result.clone());
                println!("❌ ERROR ({}ms): {}", duration, err);
            }
        }
    }

    /// Test 1: 模型加载验证
    fn test_model_loading(&self) -> Result<E2ETestResult, String> {
        let start = Instant::now();

        match InferenceEngine::from_gguf(&self.model_path) {
            Ok(engine) => {
                let load_time = start.elapsed();
                Ok(E2ETestResult {
                    test_name: "模型加载验证".to_string(),
                    backend_name: engine.get_backend_name().unwrap_or("unknown".to_string()),
                    success: true,
                    error_message: None,
                    stats: None,
                    output_text: Some(format!(
                        "加载时间: {:.2}s | 配置: {:?}",
                        load_time.as_secs_f32(),
                        engine.config()
                    )),
                    duration_ms: 0,
                })
            }
            Err(e) => Err(format!("模型加载失败: {}", e)),
        }
    }

    /// Test 2: Tokenization验证
    fn test_tokenization(&self) -> Result<E2ETestResult, String> {
        let tokenizer = Tokenizer::new();
        let test_texts = vec![
            "Hello, world!",
            "你好，世界！",
            "The quick brown fox jumps over the lazy dog.",
        ];

        let mut token_counts = Vec::new();
        for text in &test_texts {
            match tokenizer.encode(text) {
                Ok(tokens) => token_counts.push((text.clone(), tokens.len())),
                Err(e) => return Err(format!("Tokenization失败 '{}': {}", text, e)),
            }
        }

        Ok(E2ETestResult {
            test_name: "Tokenization验证".to_string(),
            backend_name: "Tokenizer".to_string(),
            success: true,
            error_message: None,
            stats: None,
            output_text: Some(
                token_counts
                    .iter()
                    .map(|(t, c)| format!("'{}': {} tokens", t, c))
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            duration_ms: 0,
        })
    }

    /// Test 3: 单轮文本生成
    fn test_single_generation(&self, prompt: &str) -> Result<E2ETestResult, String> {
        let start = Instant::now();

        let engine = InferenceEngine::from_gguf(&self.model_path)
            .map_err(|e| format!("引擎创建失败: {}", e))?;

        let params = GenerateParams::default()
            .with_max_new_tokens(self.max_new_tokens)
            .with_sampling(false); // 贪婪解码，可复现

        let generated = tokio_runtime()
            .block_on(engine.generate_async(prompt, &params))
            .map_err(|e| format!("生成失败: {}", e))?;

        let elapsed = start.elapsed();
        let stats = InferenceStats::with_timing(
            elapsed.as_millis() as u64,
            prompt.len(),    // 近似prompt tokens
            generated.len(), // 近似generated tokens
        );

        Ok(E2ETestResult {
            test_name: format!("单轮生成: {}", prompt.chars().take(30).collect::<String>()),
            backend_name: engine.get_backend_name().unwrap_or("unknown".to_string()),
            success: !generated.is_empty(),
            error_message: if generated.is_empty() {
                Some("生成为空".to_string())
            } else {
                None
            },
            stats: Some(stats),
            output_text: Some(if generated.len() > 100 {
                format!("{}...(截断,共{}字符)", &generated[..100], generated.len())
            } else {
                generated.clone()
            }),
            duration_ms: elapsed.as_millis() as u64,
        })
    }

    /// Test 4: 批量生成性能
    fn test_batch_generation(&self) -> Result<E2ETestResult, String> {
        let start = Instant::now();
        let mut all_outputs = Vec::new();

        let engine = InferenceEngine::from_gguf(&self.model_path)
            .map_err(|e| format!("引擎创建失败: {}", e))?;

        for prompt in &self.test_prompts {
            let params = GenerateParams::default().with_max_new_tokens(self.max_new_tokens);

            let output = tokio_runtime()
                .block_on(engine.generate_async(prompt, &params))
                .map_err(|e| format!("批量生成失败: {}", e))?;

            all_outputs.push(output);
        }

        let total_time = start.elapsed();
        let avg_time_per_prompt =
            total_time.as_millis() as f64 / self.test_prompts.len().max(1) as f64;

        Ok(E2ETestResult {
            test_name: "批量生成性能".to_string(),
            backend_name: engine.get_backend_name().unwrap_or("unknown".to_string()),
            success: true,
            error_message: None,
            stats: Some(InferenceStats::with_timing(
                total_time.as_millis() as u64,
                0,
                all_outputs.iter().map(|s| s.len()).sum(),
            )),
            output_text: Some(format!(
                "总时间: {:.2}s | 平均: {:.0}ms/prompt | 完成: {}/{}",
                total_time.as_secs_f32(),
                avg_time_per_prompt,
                all_outputs.len(),
                self.test_prompts.len()
            )),
            duration_ms: total_time.as_millis() as u64,
        })
    }

    /// Test 5: 贪婪解码
    fn test_greedy_decoding(&self) -> Result<E2ETestResult, String> {
        let prompt = self
            .test_prompts
            .first()
            .cloned()
            .unwrap_or_else(|| "Once upon a time".to_string());

        let engine = InferenceEngine::from_gguf(&self.model_path)
            .map_err(|e| format!("引擎创建失败: {}", e))?;

        let params = GenerateParams::default()
            .with_max_new_tokens(self.max_new_tokens)
            .with_sampling(false);

        let output = tokio_runtime()
            .block_on(engine.generate_async(&prompt, &params))
            .map_err(|e| format!("贪婪解码失败: {}", e))?;

        Ok(E2ETestResult {
            test_name: "贪婪解码".to_string(),
            backend_name: engine.get_backend_name().unwrap_or("unknown".to_string()),
            success: true,
            error_message: None,
            stats: None,
            output_text: Some(output),
            duration_ms: 0,
        })
    }

    /// Test 6: 温度采样
    fn test_temperature_sampling(&self, temp: f64) -> Result<E2ETestResult, String> {
        let prompt = self
            .test_prompts
            .first()
            .cloned()
            .unwrap_or_else(|| "The meaning of life is".to_string());

        let engine = InferenceEngine::from_gguf(&self.model_path)
            .map_err(|e| format!("引擎创建失败: {}", e))?;

        let params = GenerateParams::default()
            .with_max_new_tokens(self.max_new_tokens)
            .with_sampling(true)
            .with_temperature(temp);

        let output = tokio_runtime()
            .block_on(engine.generate_async(&prompt, &params))
            .map_err(|e| format!("温度采样失败: {}", e))?;

        Ok(E2ETestResult {
            test_name: format!("温度采样({})", temp),
            backend_name: engine.get_backend_name().unwrap_or("unknown".to_string()),
            success: true,
            error_message: None,
            stats: None,
            output_text: Some(output),
            duration_ms: 0,
        })
    }

    /// 打印测试总结报告
    fn print_summary(&self) {
        println!("\n==========================================");
        println!("📊 端到端验证测试总结");
        println!("==========================================\n");

        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.success).count();
        let failed = total - passed;

        println!("总测试数: {}", total);
        println!(
            "通过: {} ({:.1}%)",
            passed,
            passed as f64 / total.max(1) as f64 * 100.0
        );
        println!(
            "失败: {} ({:.1}%)",
            failed,
            failed as f64 / total.max(1) as f64 * 100.0
        );

        if failed > 0 {
            println!("\n❌ 失败的测试:");
            for result in &self.results {
                if !result.success {
                    println!("  ✗ {}: {:?}", result.test_name, result.error_message);
                }
            }
        }

        // 性能统计
        let successful_stats: Vec<&InferenceStats> = self
            .results
            .iter()
            .filter_map(|r| r.stats.as_ref())
            .collect();

        if !successful_stats.is_empty() {
            let avg_speed: f32 = successful_stats
                .iter()
                .map(|s| s.tokens_per_second)
                .sum::<f32>()
                / successful_stats.len() as f32;

            println!("\n⚡ 性能统计:");
            println!("  平均速度: {:.1} tokens/s", avg_speed);
        }

        println!("\n==========================================\n");
    }
}

// ============================================
// 辅助模块
// ============================================
mod e2e_test_utils {
    //! 异步测试辅助工具

    #[allow(dead_code)]
    pub struct TokioTestRuntime;

    #[allow(dead_code)]
    impl TokioTestRuntime {
        pub fn block_on<F, T>(future: F) -> T
        where
            F: std::future::Future<Output = T>,
        {
            // 同步运行异步代码（简化版）
            // 在实际项目中应该使用 tokio::runtime::Runtime
            unimplemented!("需要tokio runtime支持")
        }
    }
}
