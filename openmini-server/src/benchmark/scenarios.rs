//! Scenario Definitions Module
//!
//! 定义预置的基准测试场景，包括：
//! - 标准场景：短/中/长/超长上下文
//! - 负载场景：单用户、多用户并发、突发流量
//! - 特定能力：代码生成、数学推理、多模态输入
//! - 回归测试：基线建立与回归检测

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Scenario {
    ShortContext,
    MediumContext,
    LongContext,
    UltraLongContext,

    SingleUser,
    MultiUser { concurrency: usize },
    BurstTraffic { rps: usize },

    CodeGeneration,
    MathReasoning,
    MultimodalInput,

    RegressionBaseline,
    RegressionCheck,
}

impl Scenario {
    pub fn name(&self) -> &'static str {
        match self {
            Scenario::ShortContext => "ShortContext",
            Scenario::MediumContext => "MediumContext",
            Scenario::LongContext => "LongContext",
            Scenario::UltraLongContext => "UltraLongContext",
            Scenario::SingleUser => "SingleUser",
            Scenario::MultiUser { .. } => "MultiUser",
            Scenario::BurstTraffic { .. } => "BurstTraffic",
            Scenario::CodeGeneration => "CodeGeneration",
            Scenario::MathReasoning => "MathReasoning",
            Scenario::MultimodalInput => "MultimodalInput",
            Scenario::RegressionBaseline => "RegressionBaseline",
            Scenario::RegressionCheck => "RegressionCheck",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Scenario::ShortContext => "512 tokens, 快速响应测试",
            Scenario::MediumContext => "2048 tokens, 平衡场景",
            Scenario::LongContext => "8192/16384 tokens, 长文档理解",
            Scenario::UltraLongContext => "32768/65536 tokens, 超长上下文",
            Scenario::SingleUser => "单用户连续请求",
            Scenario::MultiUser { concurrency } => {
                Box::leak(format!("{} 并发用户模拟", concurrency).into_boxed_str()) as &str
            }
            Scenario::BurstTraffic { rps } => {
                Box::leak(format!("突发流量 ({} RPS)", rps).into_boxed_str()) as &str
            }
            Scenario::CodeGeneration => "代码生成质量+速度",
            Scenario::MathReasoning => "数学推理 (长CoT)",
            Scenario::MultimodalInput => "图像/音频输入",
            Scenario::RegressionBaseline => "建立基线",
            Scenario::RegressionCheck => "对比基线检测退化",
        }
    }

    pub fn default_sequence_length(&self) -> usize {
        match self {
            Scenario::ShortContext => 512,
            Scenario::MediumContext => 2048,
            Scenario::LongContext => 8192,
            Scenario::UltraLongContext => 32768,
            Scenario::SingleUser => 1024,
            Scenario::MultiUser { .. } => 1024,
            Scenario::BurstTraffic { .. } => 1024,
            Scenario::CodeGeneration => 2048,
            Scenario::MathReasoning => 4096,
            Scenario::MultimodalInput => 1536,
            Scenario::RegressionBaseline => 2048,
            Scenario::RegressionCheck => 2048,
        }
    }

    pub fn default_max_new_tokens(&self) -> usize {
        match self {
            Scenario::ShortContext => 128,
            Scenario::MediumContext => 256,
            Scenario::LongContext => 512,
            Scenario::UltraLongContext => 1024,
            Scenario::SingleUser => 256,
            Scenario::MultiUser { .. } => 128,
            Scenario::BurstTraffic { .. } => 64,
            Scenario::CodeGeneration => 512,
            Scenario::MathReasoning => 1024,
            Scenario::MultimodalInput => 256,
            Scenario::RegressionBaseline => 256,
            Scenario::RegressionCheck => 256,
        }
    }

    pub fn is_regression_scenario(&self) -> bool {
        matches!(
            self,
            Scenario::RegressionBaseline | Scenario::RegressionCheck
        )
    }

    pub fn all_standard_scenarios() -> Vec<Scenario> {
        vec![
            Scenario::ShortContext,
            Scenario::MediumContext,
            Scenario::LongContext,
            Scenario::UltraLongContext,
        ]
    }

    pub fn all_load_scenarios() -> Vec<Scenario> {
        vec![
            Scenario::SingleUser,
            Scenario::MultiUser { concurrency: 2 },
            Scenario::MultiUser { concurrency: 4 },
            Scenario::MultiUser { concurrency: 8 },
            Scenario::BurstTraffic { rps: 100 },
        ]
    }

    pub fn all_capability_scenarios() -> Vec<Scenario> {
        vec![
            Scenario::CodeGeneration,
            Scenario::MathReasoning,
            Scenario::MultimodalInput,
        ]
    }

    pub fn all_scenarios() -> Vec<Scenario> {
        let mut scenarios = Vec::new();
        scenarios.extend(Self::all_standard_scenarios());
        scenarios.extend(Self::all_load_scenarios());
        scenarios.extend(Self::all_capability_scenarios());
        scenarios.push(Scenario::RegressionBaseline);
        scenarios.push(Scenario::RegressionCheck);
        scenarios
    }
}

impl std::fmt::Display for Scenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_names() {
        assert_eq!(Scenario::ShortContext.name(), "ShortContext");
        assert_eq!(Scenario::MediumContext.name(), "MediumContext");
        assert_eq!(Scenario::CodeGeneration.name(), "CodeGeneration");
    }

    #[test]
    fn test_scenario_descriptions() {
        assert!(!Scenario::ShortContext.description().is_empty());
        assert!(!Scenario::LongContext.description().is_empty());
    }

    #[test]
    fn test_default_sequence_lengths() {
        assert_eq!(Scenario::ShortContext.default_sequence_length(), 512);
        assert_eq!(Scenario::MediumContext.default_sequence_length(), 2048);
        assert_eq!(Scenario::LongContext.default_sequence_length(), 8192);
        assert_eq!(Scenario::UltraLongContext.default_sequence_length(), 32768);
    }

    #[test]
    fn test_default_max_new_tokens() {
        assert_eq!(Scenario::ShortContext.default_max_new_tokens(), 128);
        assert_eq!(Scenario::MediumContext.default_max_new_tokens(), 256);
    }

    #[test]
    fn test_multi_user_scenario() {
        let scenario = Scenario::MultiUser { concurrency: 8 };
        assert_eq!(scenario.name(), "MultiUser");
        assert_eq!(scenario.default_sequence_length(), 1024);
    }

    #[test]
    fn test_burst_traffic_scenario() {
        let scenario = Scenario::BurstTraffic { rps: 200 };
        assert_eq!(scenario.name(), "BurstTraffic");
        assert_eq!(scenario.default_max_new_tokens(), 64);
    }

    #[test]
    fn test_regression_scenarios() {
        assert!(Scenario::RegressionBaseline.is_regression_scenario());
        assert!(Scenario::RegressionCheck.is_regression_scenario());
        assert!(!Scenario::ShortContext.is_regression_scenario());
    }

    #[test]
    fn test_all_scenarios_collections() {
        let standard = Scenario::all_standard_scenarios();
        assert_eq!(standard.len(), 4);

        let load = Scenario::all_load_scenarios();
        assert_eq!(load.len(), 5);

        let capability = Scenario::all_capability_scenarios();
        assert_eq!(capability.len(), 3);

        let all = Scenario::all_scenarios();
        assert!(all.len() >= 13);
    }

    #[test]
    fn test_serialization() {
        let scenario = Scenario::MediumContext;
        let json = serde_json::to_string(&scenario).unwrap();
        let deserialized: Scenario = serde_json::from_str(&json).unwrap();
        assert_eq!(scenario, deserialized);
    }

    #[test]
    fn test_display_trait() {
        assert_eq!(format!("{}", Scenario::ShortContext), "ShortContext");
        assert_eq!(format!("{}", Scenario::CodeGeneration), "CodeGeneration");
    }
}
