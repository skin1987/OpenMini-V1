//! Pan & Scan 高分辨率图像处理
//!
//! 将超大/非正方形图像分割为多个固定尺寸的patch，
//! 分别编码后拼接为完整的视觉特征序列。

use super::image_processor::GemmaImageProcessor;
use ndarray::Array3;

#[derive(Clone)]
pub struct PanScanConfig {
    /// 基础 patch 尺寸 (正方形)
    pub base_size: usize,
    /// 最大 token 数量限制
    pub max_tokens: usize,
    /// 重叠像素数（用于边界平滑）
    pub overlap: usize,
}

impl Default for PanScanConfig {
    fn default() -> Self {
        Self {
            base_size: 896,
            max_tokens: 16384,
            overlap: 14,
        }
    }
}

/// 表示一个图像 patch 的位置和尺寸
#[derive(Debug, Clone)]
pub struct ImagePatch {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// 计算图像需要的 patch 布局
pub fn calculate_patch_layout(
    image_width: usize,
    image_height: usize,
    config: &PanScanConfig,
) -> Vec<ImagePatch> {
    let mut patches = Vec::new();

    // 如果图像尺寸不超过 base_size，直接作为单个 patch 处理
    if image_width <= config.base_size && image_height <= config.base_size {
        patches.push(ImagePatch {
            x: 0,
            y: 0,
            width: image_width,
            height: image_height,
        });
        return patches;
    }

    let mut y = 0;
    while y < image_height {
        let h = config.base_size.min(image_height - y);

        let mut x = 0;
        while x < image_width {
            let w = config.base_size.min(image_width - x);

            if w > 0 && h > 0 {
                patches.push(ImagePatch {
                    x,
                    y,
                    width: w,
                    height: h,
                });
            }

            x += config.base_size - config.overlap;
        }

        y += config.base_size - config.overlap;
    }

    patches
}

/// 计算总 token 数量
pub fn estimate_total_tokens(patches: &[ImagePatch], patch_size: usize) -> usize {
    let tokens_per_patch = (patch_size / 14).pow(2);
    patches.len() * tokens_per_patch + 1 // +1 for [CLS]
}

/// 从原始图像中提取一个 patch
pub fn extract_patch(image: &Array3<u8>, patch: &ImagePatch, config: &PanScanConfig) -> Array3<u8> {
    let (h, w, _ch) = image.dim();
    let ph = patch.height.min(h - patch.y);
    let pw = patch.width.min(w - patch.x);

    let mut result = Array3::<u8>::zeros((config.base_size, config.base_size, 3));

    for y in 0..ph {
        for x in 0..pw {
            for c in 0..3 {
                let src_val = image[[patch.y + y, patch.x + x, c]];
                result[[y, x, c]] = src_val;
            }
        }
    }

    result
}

/// Pan & Scan 处理器
pub struct PanScanProcessor {
    config: PanScanConfig,
    processor: GemmaImageProcessor,
}

impl PanScanProcessor {
    pub fn new(config: PanScanConfig) -> Self {
        use super::image_processor::GemmaImageProcessorConfig;
        Self {
            config: config.clone(),
            processor: GemmaImageProcessor::new(GemmaImageProcessorConfig {
                image_size: config.base_size,
                ..Default::default()
            }),
        }
    }

    /// 处理任意尺寸的图像，返回预处理后的 patch 列表
    pub fn process(&self, image: &Array3<u8>) -> Result<Vec<(Array3<f32>, ImagePatch)>, String> {
        let (h, w, _) = image.dim();
        let layout = calculate_patch_layout(w, h, &self.config);

        let total_tokens = estimate_total_tokens(&layout, self.config.base_size);
        if total_tokens > self.config.max_tokens {
            return Err(format!(
                "Image too large: {}x{} requires {} tokens (max {})",
                w, h, total_tokens, self.config.max_tokens
            ));
        }

        let mut processed = Vec::with_capacity(layout.len());
        for patch in &layout {
            let patch_image = extract_patch(image, patch, &self.config);
            match self.processor.preprocess(&patch_image) {
                Ok(normalized) => processed.push((normalized, patch.clone())),
                Err(e) => return Err(format!("Patch preprocessing failed: {}", e)),
            }
        }

        Ok(processed)
    }

    /// 获取处理后的总 token 数量
    pub fn total_token_count(&self, image_width: usize, image_height: usize) -> usize {
        let layout = calculate_patch_layout(image_width, image_height, &self.config);
        estimate_total_tokens(&layout, self.config.base_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_image_single_patch() {
        let config = PanScanConfig::default();
        let patches = calculate_patch_layout(896, 896, &config);
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].width, 896);
        assert_eq!(patches[0].height, 896);
    }

    #[test]
    fn test_large_image_multiple_patches() {
        let config = PanScanConfig::default();
        let patches = calculate_patch_layout(2000, 1500, &config);
        assert!(patches.len() >= 4);

        let first = &patches[0];
        assert_eq!(first.x, 0);
        assert_eq!(first.y, 0);
    }

    #[test]
    fn test_token_estimation() {
        let _config = PanScanConfig::default();
        let patches = vec![
            ImagePatch {
                x: 0,
                y: 0,
                width: 896,
                height: 896,
            },
            ImagePatch {
                x: 882,
                y: 0,
                width: 896,
                height: 896,
            },
        ];
        let tokens = estimate_total_tokens(&patches, 896);
        assert_eq!(tokens, 2 * 4096 + 1); // 2 patches * 4096 + CLS
    }
}
