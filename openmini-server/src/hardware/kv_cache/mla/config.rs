//! MLA 多头潜在注意力配置
//!
//! MLA 通过低秩分解压缩 KV 缓存，显著减少内存占用

#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub struct MLAConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub latent_dim: usize,
    pub use_decoupled_rope: bool,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl Default for MLAConfig {
    fn default() -> Self {
        Self {
            hidden_size: 3584,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            latent_dim: 512,
            use_decoupled_rope: true,
            rope_theta: 1000000.0,
            max_seq_len: 32768,
        }
    }
}

impl MLAConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    pub fn with_num_heads(mut self, num_attention_heads: usize, num_key_value_heads: usize) -> Self {
        self.num_attention_heads = num_attention_heads;
        self.num_key_value_heads = num_key_value_heads;
        self
    }

    pub fn with_latent_dim(mut self, latent_dim: usize) -> Self {
        self.latent_dim = latent_dim;
        self
    }

    pub fn with_decoupled_rope(mut self, use_decoupled_rope: bool) -> Self {
        self.use_decoupled_rope = use_decoupled_rope;
        self
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn q_latent_dim(&self) -> usize {
        if self.use_decoupled_rope {
            self.num_attention_heads * self.head_dim
        } else {
            self.num_attention_heads * self.head_dim
        }
    }

    pub fn kv_latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn compress_ratio(&self) -> f32 {
        let standard_kv = 2 * self.kv_dim();
        let compressed_kv = 2 * self.latent_dim;
        1.0 - (compressed_kv as f32 / standard_kv as f32)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err("hidden_size cannot be zero".to_string());
        }
        if self.num_attention_heads == 0 {
            return Err("num_attention_heads cannot be zero".to_string());
        }
        if self.num_key_value_heads == 0 {
            return Err("num_key_value_heads cannot be zero".to_string());
        }
        if self.head_dim == 0 {
            return Err("head_dim cannot be zero".to_string());
        }
        if self.latent_dim == 0 {
            return Err("latent_dim cannot be zero".to_string());
        }
        if self.latent_dim > self.kv_dim() {
            return Err("latent_dim should be smaller than kv_dim for compression".to_string());
        }
        if self.num_attention_heads < self.num_key_value_heads {
            return Err("num_attention_heads should be >= num_key_value_heads".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_config_default() {
        let config = MLAConfig::default();
        assert_eq!(config.hidden_size, 3584);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.latent_dim, 512);
        assert!(config.use_decoupled_rope);
    }

    #[test]
    fn test_mla_config_validation() {
        let config = MLAConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = MLAConfig::default().with_hidden_size(0);
        assert!(invalid_config.validate().is_err());

        let invalid_config2 = MLAConfig::default().with_latent_dim(10240);
        assert!(invalid_config2.validate().is_err());
    }

    #[test]
    fn test_compress_ratio() {
        let config = MLAConfig::default();
        let ratio = config.compress_ratio();
        assert!(ratio > 0.0);
        assert!(ratio < 1.0);

        let high_compress = MLAConfig::default().with_latent_dim(64);
        let high_ratio = high_compress.compress_ratio();
        assert!(high_ratio > 0.9);
    }

    #[test]
    fn test_kv_dim() {
        let config = MLAConfig::default();
        assert_eq!(config.kv_dim(), 8 * 128);
    }

    #[test]
    fn test_builder_pattern() {
        let config = MLAConfig::new()
            .with_hidden_size(2048)
            .with_num_heads(16, 4)
            .with_latent_dim(256)
            .with_decoupled_rope(true);

        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.latent_dim, 256);
        assert!(config.use_decoupled_rope);
    }

    // 新增分支覆盖测试

    /// 测试 new 方法（等同于 default）
    #[test]
    fn test_new() {
        let config = MLAConfig::new();
        assert_eq!(config, MLAConfig::default());
    }

    /// 测试 validate 的 num_attention_heads=0 错误分支
    #[test]
    fn test_validate_zero_attention_heads() {
        let config = MLAConfig::new()
            .with_num_heads(0, 8);
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("num_attention_heads"));
    }

    /// 测试 validate 的 num_key_value_heads=0 错误分支
    #[test]
    fn test_validate_zero_kv_heads() {
        let config = MLAConfig::new()
            .with_num_heads(32, 0);
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("num_key_value_heads"));
    }

    /// 测试 validate 的 head_dim=0 错误分支
    #[test]
    fn test_validate_zero_head_dim() {
        // 需要手动构造，因为 builder 没有提供 head_dim 设置方法
        let config = MLAConfig {
            hidden_size: 3584,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 0,
            latent_dim: 512,
            use_decoupled_rope: true,
            rope_theta: 1000000.0,
            max_seq_len: 32768,
        };
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("head_dim"));
    }

    /// 测试 validate 的 latent_dim=0 错误分支
    #[test]
    fn test_validate_zero_latent_dim() {
        let config = MLAConfig::new()
            .with_latent_dim(0);
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("latent_dim"));
    }

    /// 测试 validate 的 latent_dim > kv_dim 错误分支
    #[test]
    fn test_validate_latent_dim_exceeds_kv_dim() {
        // kv_dim = num_key_value_heads * head_dim = 8 * 128 = 1024
        let config = MLAConfig::new()
            .with_latent_dim(2048); // > 1024
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("latent_dim should be smaller"));
    }

    /// 测试 validate 的 num_attention_heads < num_key_value_heads 错误分支
    #[test]
    fn test_validate_attention_heads_less_than_kv_heads() {
        let config = MLAConfig::new()
            .with_num_heads(4, 8); // attention_heads < key_value_heads
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("num_attention_heads should be >="));
    }

    /// 测试 q_latent_dim 方法
    #[test]
    fn test_q_latent_dim() {
        let config_with_rope = MLAConfig::new()
            .with_decoupled_rope(true);
        
        let config_without_rope = MLAConfig::new()
            .with_decoupled_rope(false);
        
        // 两种情况下应该相同
        assert_eq!(config_with_rope.q_latent_dim(), 32 * 128);
        assert_eq!(config_without_rope.q_latent_dim(), 32 * 128);
    }

    /// 测试 kv_latent_dim 方法
    #[test]
    fn test_kv_latent_dim() {
        let config = MLAConfig::new();
        assert_eq!(config.kv_latent_dim(), config.latent_dim);
    }

    /// 测试 compress_ratio 在边界情况下的值
    #[test]
    fn test_compress_ratio_boundary() {
        // 当 latent_dim 接近 kv_dim 时，压缩率接近 0
        let low_compress = MLAConfig::new()
            .with_latent_dim(1023); // kv_dim - 1
        
        let ratio = low_compress.compress_ratio();
        assert!(ratio >= 0.0 && ratio < 1.0);
    }

    /// 测试 MLAConfig 的 PartialEq trait（完整相等性比较）
    /// 覆盖分支：PartialEq 的所有字段比较
    #[test]
    fn test_mla_config_partial_eq() {
        let config1 = MLAConfig::default();
        let config2 = MLAConfig::default();
        
        // 两个默认配置应该相等
        assert_eq!(config1, config2);
        
        // 修改一个字段后应该不等
        let config3 = config1.clone().with_hidden_size(2048);
        assert_ne!(config1, config3);
        
        let config4 = config1.clone().with_num_heads(16, 4);
        assert_ne!(config1, config4);
        
        let config5 = config1.clone().with_latent_dim(256);
        assert_ne!(config1, config5);
        
        let config6 = config1.clone().with_decoupled_rope(false);
        assert_ne!(config1, config6);
    }

    /// 测试 MLAConfig 的 Clone trait 独立性
    /// 覆盖分支：Clone 后的修改不影响原对象
    #[test]
    fn test_mla_config_clone_independence() {
        let config1 = MLAConfig::default();
        let mut config2 = config1.clone();
        
        // 修改克隆后的对象
        config2.hidden_size = 9999;
        config2.latent_dim = 100;
        
        // 原对象应该不受影响
        assert_eq!(config1.hidden_size, 3584);
        assert_eq!(config1.latent_dim, 512);
        
        // 克隆后的对象应该反映修改
        assert_eq!(config2.hidden_size, 9999);
        assert_eq!(config2.latent_dim, 100);
    }

    /// 测试 validate 成功路径的完整配置验证
    /// 覆盖分支：validate 返回 Ok 的各种有效配置
    #[test]
    fn test_validate_valid_configs() {
        // 默认配置应该通过验证
        let default_config = MLAConfig::default();
        assert!(default_config.validate().is_ok());
        
        // 自定义但有效的配置
        let custom_config = MLAConfig::new()
            .with_hidden_size(2048)
            .with_num_heads(16, 8) // attention_heads >= kv_heads
            .with_latent_dim(256);  // latent_dim < kv_dim (kv_dim=8*128=1024)
        assert!(custom_config.validate().is_ok());
        
        // 极小但有效的配置
        let tiny_config = MLAConfig::new()
            .with_hidden_size(1)
            .with_num_heads(1, 1)
            .with_latent_dim(1);
        assert!(tiny_config.validate().is_ok());
    }

    /// 测试 builder pattern 链式调用的顺序无关性
    /// 覆盖分支：不同顺序的链式调用产生相同结果
    #[test]
    fn test_builder_pattern_order_independence() {
        // 顺序1：按声明顺序调用
        let config1 = MLAConfig::new()
            .with_hidden_size(2048)
            .with_num_heads(16, 8)
            .with_latent_dim(512)
            .with_decoupled_rope(false);
        
        // 顺序2：逆序调用
        let config2 = MLAConfig::new()
            .with_decoupled_rope(false)
            .with_latent_dim(512)
            .with_num_heads(16, 8)
            .with_hidden_size(2048);
        
        // 两种顺序应该产生相同的配置
        assert_eq!(config1, config2);
    }

    /// 测试 kv_dim 和 q_latent_dim 在不同配置下的值
    /// 覆盖分支：维度计算公式的正确性
    #[test]
    fn test_dimension_calculations() {
        // 配置1：标准配置
        let config1 = MLAConfig::default();
        assert_eq!(config1.kv_dim(), 8 * 128); // num_kv_heads * head_dim
        assert_eq!(config1.q_latent_dim(), 32 * 128); // num_attention_heads * head_dim
        
        // 配置2：自定义头数
        let config2 = MLAConfig::new()
            .with_num_heads(64, 16); // 64个注意力头，16个KV头
        assert_eq!(config2.kv_dim(), 16 * 128);
        assert_eq!(config2.q_latent_dim(), 64 * 128);
        
        // 配置3：不同头维度（需要手动构造）
        let config3 = MLAConfig {
            head_dim: 64,
            ..MLAConfig::new()
        };
        assert_eq!(config3.kv_dim(), 8 * 64);
        assert_eq!(config3.q_latent_dim(), 32 * 64);
    }
}
