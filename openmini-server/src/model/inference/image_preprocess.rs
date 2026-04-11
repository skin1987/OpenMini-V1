//! 图像预处理模块
//!
//! 提供多模态推理所需的图像预处理功能：
//! - 尺寸调整 (resize)
//! - 归一化 (normalize)
//! - 通道转换

#![allow(dead_code)]

use anyhow::{anyhow, Result};
use ndarray::{Array3, Array4};

/// 图像预处理器类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessorType {
    /// 标准预处理器 (224×224, ImageNet)
    Standard,
    /// Gemma3 预处理器 (896×896, SigLIP)
    Gemma3,
}

impl Default for ProcessorType {
    fn default() -> Self {
        ProcessorType::Standard
    }
}

/// 图像预处理器配置
#[derive(Debug, Clone)]
pub struct ImagePreprocessorConfig {
    /// 预处理器类型
    pub processor_type: ProcessorType,
    /// 目标高度
    pub target_height: usize,
    /// 目标宽度
    pub target_width: usize,
    /// 是否归一化到 [0, 1]
    pub normalize: bool,
    /// 均值（用于标准化）
    pub mean: [f32; 3],
    /// 标准差（用于标准化）
    pub std: [f32; 3],
}

impl Default for ImagePreprocessorConfig {
    fn default() -> Self {
        Self {
            processor_type: ProcessorType::Standard,
            target_height: 224,
            target_width: 224,
            normalize: true,
            mean: [0.485, 0.456, 0.406], // ImageNet 均值
            std: [0.229, 0.224, 0.225],  // ImageNet 标准差
        }
    }
}

impl ImagePreprocessorConfig {
    pub fn gemma3_config() -> Self {
        Self {
            processor_type: ProcessorType::Gemma3,
            target_height: 896,
            target_width: 896,
            normalize: true,
            mean: [0.5, 0.5, 0.5], // Gemma3 均值
            std: [0.5, 0.5, 0.5],  // Gemma3 标准差
        }
    }
}

/// 图像预处理器
#[derive(Clone)]
pub struct ImagePreprocessor {
    config: ImagePreprocessorConfig,
}

impl ImagePreprocessor {
    /// 创建新的图像预处理器
    pub fn new(config: ImagePreprocessorConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建预处理器
    pub fn default_preprocessor() -> Self {
        Self::new(ImagePreprocessorConfig::default())
    }

    /// 预处理图像
    ///
    /// # 参数
    /// - `image`: 输入图像 (H, W, 3)，RGB 格式，u8 类型
    ///
    /// # 返回
    /// 预处理后的图像 (H, W, 3)，f32 类型，已归一化
    pub fn preprocess(&self, image: &Array3<u8>) -> Result<Array3<f32>> {
        let shape = image.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);

        // 校验通道数
        if c != 3 {
            return Err(anyhow!("Image must have 3 channels (RGB), got {}", c));
        }

        // 调整尺寸（如果需要）
        let resized = if h != self.config.target_height || w != self.config.target_width {
            self.resize(image)?
        } else {
            image.clone()
        };

        // 转换为 f32 并归一化
        let normalized = if self.config.normalize {
            self.normalize(&resized)
        } else {
            self.to_float(&resized)
        };

        Ok(normalized)
    }

    /// 仅调整图像尺寸（不归一化）
    ///
    /// # 参数
    /// - `image`: 输入图像 (H, W, 3)，RGB 格式，u8 类型
    ///
    /// # 返回
    /// 调整尺寸后的图像 (H, W, 3)，u8 类型
    pub fn resize_only(&self, image: &Array3<u8>) -> Result<Array3<u8>> {
        let shape = image.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);

        // 校验通道数
        if c != 3 {
            return Err(anyhow!("Image must have 3 channels (RGB), got {}", c));
        }

        // 调整尺寸（如果需要）
        let resized = if h != self.config.target_height || w != self.config.target_width {
            self.resize(image)?
        } else {
            image.clone()
        };

        Ok(resized)
    }

    /// 调整图像尺寸（双线性插值）
    fn resize(&self, image: &Array3<u8>) -> Result<Array3<u8>> {
        let shape = image.shape();
        let (src_h, src_w, c) = (shape[0], shape[1], shape[2]);

        let dst_h = self.config.target_height;
        let dst_w = self.config.target_width;

        let mut result = Array3::<u8>::zeros((dst_h, dst_w, c));

        let scale_h = src_h as f32 / dst_h as f32;
        let scale_w = src_w as f32 / dst_w as f32;

        for y in 0..dst_h {
            for x in 0..dst_w {
                // 计算源图像中的对应位置
                let src_y = y as f32 * scale_h;
                let src_x = x as f32 * scale_w;

                // 双线性插值
                let y0 = src_y.floor() as usize;
                let x0 = src_x.floor() as usize;
                let y1 = (y0 + 1).min(src_h - 1);
                let x1 = (x0 + 1).min(src_w - 1);

                let dy = src_y - y0 as f32;
                let dx = src_x - x0 as f32;

                for ch in 0..c {
                    let v00 = image[[y0, x0, ch]] as f32;
                    let v01 = image[[y0, x1, ch]] as f32;
                    let v10 = image[[y1, x0, ch]] as f32;
                    let v11 = image[[y1, x1, ch]] as f32;

                    let v = v00 * (1.0 - dx) * (1.0 - dy)
                        + v01 * dx * (1.0 - dy)
                        + v10 * (1.0 - dx) * dy
                        + v11 * dx * dy;

                    result[[y, x, ch]] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(result)
    }

    /// 归一化图像（ImageNet 标准化）
    fn normalize(&self, image: &Array3<u8>) -> Array3<f32> {
        let shape = image.shape();
        let (h, w, _c) = (shape[0], shape[1], shape[2]);

        let mut result = Array3::<f32>::zeros((h, w, 3));

        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let pixel = image[[y, x, c]] as f32 / 255.0;
                    result[[y, x, c]] = (pixel - self.config.mean[c]) / self.config.std[c];
                }
            }
        }

        result
    }

    /// 转换为 f32（不归一化）
    fn to_float(&self, image: &Array3<u8>) -> Array3<f32> {
        let shape = image.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);

        let mut result = Array3::<f32>::zeros((h, w, c));

        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    result[[y, x, ch]] = image[[y, x, ch]] as f32;
                }
            }
        }

        result
    }

    /// 将图像转换为模型输入格式 (1, C, H, W)
    pub fn to_model_format(&self, image: &Array3<f32>) -> Array4<f32> {
        let shape = image.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);

        let mut result = Array4::<f32>::zeros((1, c, h, w));

        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    result[[0, ch, y, x]] = image[[y, x, ch]];
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocessor_default() {
        let preprocessor = ImagePreprocessor::default_preprocessor();
        assert_eq!(preprocessor.config.target_height, 224);
        assert_eq!(preprocessor.config.target_width, 224);
    }

    #[test]
    fn test_preprocess_no_resize() {
        let preprocessor = ImagePreprocessor::default_preprocessor();
        let image = Array3::<u8>::zeros((224, 224, 3));
        let result = preprocessor.preprocess(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_preprocess_with_resize() {
        let preprocessor = ImagePreprocessor::default_preprocessor();
        let image = Array3::<u8>::zeros((256, 256, 3));
        let result = preprocessor.preprocess(&image).unwrap();
        assert_eq!(result.shape(), &[224, 224, 3]);
    }

    #[test]
    fn test_invalid_channels() {
        let preprocessor = ImagePreprocessor::default_preprocessor();
        let image = Array3::<u8>::zeros((224, 224, 4));
        let result = preprocessor.preprocess(&image);
        assert!(result.is_err());
    }

    // ==================== 分支覆盖率补充测试 ====================

    #[test]
    fn test_image_preprocess_rgb_to_tensor() {
        // RGB图像转Tensor - 测试2x2 RGB图像的完整预处理流程
        let config = ImagePreprocessorConfig {
            processor_type: ProcessorType::Standard,
            target_height: 2,
            target_width: 2,
            normalize: false,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        let preprocessor = ImagePreprocessor::new(config);

        // 创建2x2 RGB图像数据 (H, W, C)
        let mut image = Array3::<u8>::zeros((2, 2, 3));
        // 填充像素值：红、绿、蓝、灰
        image[[0, 0, 0]] = 255;
        image[[0, 0, 1]] = 0;
        image[[0, 0, 2]] = 0; // 红
        image[[0, 1, 0]] = 0;
        image[[0, 1, 1]] = 255;
        image[[0, 1, 2]] = 0; // 绿
        image[[1, 0, 0]] = 0;
        image[[1, 0, 1]] = 0;
        image[[1, 0, 2]] = 255; // 蓝
        image[[1, 1, 0]] = 128;
        image[[1, 1, 1]] = 128;
        image[[1, 1, 2]] = 128; // 灰

        let tensor = preprocessor.preprocess(&image);
        assert!(tensor.is_ok());
        let t = tensor.unwrap();
        assert_eq!(t.shape(), &[2, 2, 3]); // HWC格式
                                           // 验证像素值正确转换（无归一化时应该等于原始值）
        assert!((t[[0, 0, 0]] - 255.0).abs() < 0.01); // 红色通道
    }

    #[test]
    fn test_image_preprocess_with_normalization() {
        // 测试带ImageNet标准化的预处理
        let preprocessor = ImagePreprocessor::default_preprocessor();
        let mut image = Array3::<u8>::zeros((224, 224, 3));
        // 设置一个白色像素
        image[[0, 0, 0]] = 255;
        image[[0, 0, 1]] = 255;
        image[[0, 0, 2]] = 255;

        let result = preprocessor.preprocess(&image).unwrap();
        // 验证归一化后的值在合理范围内 (pixel/255 - mean) / std
        // 白色像素：(1.0 - 0.485) / 0.229 ≈ 2.24
        assert!((result[[0, 0, 0]] - 2.24).abs() < 0.01);
    }

    #[test]
    fn test_image_preprocess_grayscale_not_supported() {
        // 当前实现仅支持RGB，灰度图像应返回错误
        let preprocessor = ImagePreprocessor::default_preprocessor();
        // 尝试传入单通道图像（灰度）
        let gray_image = Array3::<u8>::zeros((2, 1, 1)); // 单通道
        let result = preprocessor.preprocess(&gray_image);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("3 channels"));
    }

    #[test]
    fn test_image_preprocess_invalid_dimensions() {
        // 测试无效尺寸的处理（通过resize_only测试）
        let preprocessor = ImagePreprocessor::default_preprocessor();
        // 创建一个非标准尺寸的RGB图像
        let image = Array3::<u8>::zeros((100, 100, 3));
        let result = preprocessor.resize_only(&image);
        assert!(result.is_ok()); // 应该能成功resize到224x224
        let resized = result.unwrap();
        assert_eq!(resized.shape(), &[224, 224, 3]);
    }

    #[test]
    fn test_image_preprocess_empty_input() {
        // 空输入测试（0尺寸图像）
        let config = ImagePreprocessorConfig {
            processor_type: ProcessorType::Standard,
            target_height: 0,
            target_width: 0,
            normalize: true,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        };
        let preprocessor = ImagePreprocessor::new(config);

        // 0x0的空图像
        let empty_image = Array3::<u8>::zeros((0, 0, 3));
        let result = preprocessor.preprocess(&empty_image);
        // 0尺寸图像可能返回错误或空结果
        if result.is_ok() {
            let t = result.unwrap();
            assert_eq!(t.len(), 0);
        }
    }

    #[test]
    fn test_image_resize_bilinear() {
        // 双线性插值缩放测试 - 从小尺寸放大到大尺寸
        let config = ImagePreprocessorConfig {
            processor_type: ProcessorType::Standard,
            target_height: 8,
            target_width: 8,
            normalize: false,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        let preprocessor = ImagePreprocessor::new(config);

        // 创建4x4 RGB图像并填充渐变模式
        let mut image = Array3::<u8>::zeros((4, 4, 3));
        for y in 0..4 {
            for x in 0..4 {
                for c in 0..3 {
                    image[[y, x, c]] = ((y * 64 + x * 16 + c * 32) % 256) as u8;
                }
            }
        }

        let result = preprocessor.resize_only(&image);
        assert!(result.is_ok());
        let resized = result.unwrap();
        assert_eq!(resized.shape(), &[8, 8, 3]);
        // 验证插值后的值在有效范围内
        // val 是 u8 类型，始终 <= 255，无需断言
    }

    #[test]
    fn test_image_normalize() {
        // 图像归一化验证
        let config = ImagePreprocessorConfig {
            processor_type: ProcessorType::Standard,
            target_height: 16,
            target_width: 16,
            normalize: true,
            mean: [0.5, 0.5, 0.5], // mean=0.5
            std: [0.5, 0.5, 0.5],  // std=0.5
        };
        let preprocessor = ImagePreprocessor::new(config);

        // 创建16x16图像，填充固定值128
        let mut image = Array3::<u8>::zeros((16, 16, 3));
        image.fill(128); // 中等亮度

        let normalized = preprocessor.preprocess(&image).unwrap();

        // 验证归一化范围合理：(128/255 - 0.5) / 0.5 ≈ 0.0039
        for &val in normalized.iter() {
            assert!(
                val >= -3.0 && val <= 3.0,
                "Normalized value {} out of range",
                val
            );
        }
    }

    #[test]
    fn test_to_model_format() {
        // 测试转换为模型输入格式 (1, C, H, W)
        let config = ImagePreprocessorConfig {
            processor_type: ProcessorType::Standard,
            target_height: 2,
            target_width: 2,
            normalize: false,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        let preprocessor = ImagePreprocessor::new(config);

        let image = Array3::<f32>::zeros((2, 2, 3));
        let model_input = preprocessor.to_model_format(&image);

        assert_eq!(model_input.shape(), &[1, 3, 2, 2]); // NCHW格式
    }

    #[test]
    fn test_custom_config() {
        // 测试自定义配置
        let custom_config = ImagePreprocessorConfig {
            processor_type: ProcessorType::Standard,
            target_height: 512,
            target_width: 512,
            normalize: false,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };

        let preprocessor = ImagePreprocessor::new(custom_config.clone());
        assert_eq!(preprocessor.config.target_height, 512);
        assert_eq!(preprocessor.config.target_width, 512);
        assert!(!preprocessor.config.normalize);

        let image = Array3::<u8>::zeros((512, 512, 3));
        let result = preprocessor.preprocess(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resize_downsample() {
        // 测试缩小图像
        let config = ImagePreprocessorConfig {
            processor_type: ProcessorType::Standard,
            target_height: 112,
            target_width: 112,
            normalize: false,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        let preprocessor = ImagePreprocessor::new(config);

        let image = Array3::<u8>::zeros((224, 224, 3));
        let result = preprocessor.resize_only(&image);
        assert!(result.is_ok());
        let resized = result.unwrap();
        assert_eq!(resized.shape(), &[112, 112, 3]);
    }

    #[test]
    fn test_preprocess_rgba_channels() {
        // 测试RGBA图像（4通道）应失败
        let preprocessor = ImagePreprocessor::default_preprocessor();
        let rgba_image = Array3::<u8>::zeros((224, 224, 4));
        let result = preprocessor.preprocess(&rgba_image);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_normalize_mode() {
        // 测试不归一化模式
        let config = ImagePreprocessorConfig {
            processor_type: ProcessorType::Standard,
            target_height: 10,
            target_width: 10,
            normalize: false,
            mean: [0.0; 3],
            std: [1.0; 3],
        };
        let preprocessor = ImagePreprocessor::new(config);

        let mut image = Array3::<u8>::zeros((10, 10, 3));
        image.fill(200);

        let result = preprocessor.preprocess(&image).unwrap();
        // 不归一化时，值应该直接转换为f32
        assert!((result[[0, 0, 0]] - 200.0).abs() < 0.01);
    }
}
