use crate::model::inference::error::{InferenceError, InferenceResult};
use ndarray::Array3;

#[derive(Debug, Clone)]
pub enum ResampleMethod {
    Bicubic,
    Bilinear,
    Nearest,
}

#[derive(Debug, Clone)]
pub struct GemmaImageProcessorConfig {
    pub image_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub do_resize: bool,
    pub do_normalize: bool,
    pub resample: ResampleMethod,
}

impl Default for GemmaImageProcessorConfig {
    fn default() -> Self {
        Self {
            image_size: 896,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            do_resize: true,
            do_normalize: true,
            resample: ResampleMethod::Bilinear,
        }
    }
}

#[derive(Clone)]
pub struct GemmaImageProcessor {
    config: GemmaImageProcessorConfig,
}

impl GemmaImageProcessor {
    pub fn new(config: GemmaImageProcessorConfig) -> Self {
        Self { config }
    }

    pub fn preprocess(&self, image: &Array3<u8>) -> InferenceResult<Array3<f32>> {
        let shape = image.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);

        if c != 3 {
            return Err(InferenceError::image_preprocess(format!(
                "Expected 3 channels (RGB), got {}",
                c
            )));
        }

        let resized = if self.config.do_resize && (h != self.config.image_size || w != self.config.image_size)
        {
            self.resize(image)?
        } else {
            image.clone()
        };

        let result = if self.config.do_normalize {
            self.normalize(&resized)
        } else {
            self.to_float(&resized)
        };

        Ok(result)
    }

    fn resize(&self, image: &Array3<u8>) -> InferenceResult<Array3<u8>> {
        let shape = image.shape();
        let (src_h, src_w, c) = (shape[0], shape[1], shape[2]);
        let dst_h = self.config.image_size;
        let dst_w = self.config.image_size;

        if src_h == 0 || src_w == 0 {
            return Err(InferenceError::image_preprocess(
                "Source image has zero dimension",
            ));
        }

        let mut result = Array3::<u8>::zeros((dst_h, dst_w, c));

        match &self.config.resample {
            ResampleMethod::Bilinear => {
                self.bilinear_interpolation(image, &mut result, src_h, src_w, dst_h, dst_w, c);
            }
            ResampleMethod::Nearest => {
                self.nearest_neighbor(image, &mut result, src_h, src_w, dst_h, dst_w, c);
            }
            ResampleMethod::Bicubic => {
                self.bilinear_interpolation(image, &mut result, src_h, src_w, dst_h, dst_w, c);
            }
        }

        Ok(result)
    }

    fn bilinear_interpolation(
        &self,
        src: &Array3<u8>,
        dst: &mut Array3<u8>,
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
        c: usize,
    ) {
        let scale_h = src_h as f32 / dst_h as f32;
        let scale_w = src_w as f32 / dst_w as f32;

        for y in 0..dst_h {
            for x in 0..dst_w {
                let src_y = y as f32 * scale_h;
                let src_x = x as f32 * scale_w;

                let y0 = src_y.floor() as usize;
                let x0 = src_x.floor() as usize;
                let y1 = (y0 + 1).min(src_h - 1);
                let x1 = (x0 + 1).min(src_w - 1);

                let dy = src_y - y0 as f32;
                let dx = src_x - x0 as f32;

                for ch in 0..c {
                    let v00 = src[[y0, x0, ch]] as f32;
                    let v01 = src[[y0, x1, ch]] as f32;
                    let v10 = src[[y1, x0, ch]] as f32;
                    let v11 = src[[y1, x1, ch]] as f32;

                    let v = v00 * (1.0 - dx) * (1.0 - dy)
                        + v01 * dx * (1.0 - dy)
                        + v10 * (1.0 - dx) * dy
                        + v11 * dx * dy;

                    dst[[y, x, ch]] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    fn nearest_neighbor(
        &self,
        src: &Array3<u8>,
        dst: &mut Array3<u8>,
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
        c: usize,
    ) {
        let scale_h = src_h as f32 / dst_h as f32;
        let scale_w = src_w as f32 / dst_w as f32;

        for y in 0..dst_h {
            for x in 0..dst_w {
                let src_y = ((y as f32 * scale_h).round() as usize).min(src_h - 1);
                let src_x = ((x as f32 * scale_w).round() as usize).min(src_w - 1);

                for ch in 0..c {
                    dst[[y, x, ch]] = src[[src_y, src_x, ch]];
                }
            }
        }
    }

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

    pub fn num_image_tokens(&self, _image_h: usize, _image_w: usize) -> usize {
        let patches_per_side = self.config.image_size / 14; // patch_size is fixed at 14 for SigLIP
        patches_per_side.pow(2)
    }

    pub fn config(&self) -> &GemmaImageProcessorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GemmaImageProcessorConfig::default();
        assert_eq!(config.image_size, 896);
        assert_eq!(config.mean, [0.5, 0.5, 0.5]);
        assert_eq!(config.std, [0.5, 0.5, 0.5]);
        assert!(config.do_resize);
        assert!(config.do_normalize);
        assert!(matches!(config.resample, ResampleMethod::Bilinear));
    }

    #[test]
    fn test_processor_creation() {
        let config = GemmaImageProcessorConfig::default();
        let processor = GemmaImageProcessor::new(config);
        assert_eq!(processor.config().image_size, 896);
    }

    #[test]
    fn test_preprocess_no_resize_needed() {
        let config = GemmaImageProcessorConfig {
            image_size: 56,
            ..Default::default()
        };
        let processor = GemmaImageProcessor::new(config);

        let image = Array3::<u8>::zeros((56, 56, 3));
        let result = processor.preprocess(&image);

        assert!(result.is_ok());
        let processed = result.unwrap();
        assert_eq!(processed.shape(), &[56, 56, 3]);
    }

    #[test]
    fn test_preprocess_with_resize() {
        let config = GemmaImageProcessorConfig {
            image_size: 28,
            ..Default::default()
        };
        let processor = GemmaImageProcessor::new(config);

        let image = Array3::<u8>::zeros((56, 56, 3));
        let result = processor.preprocess(&image);

        assert!(result.is_ok());
        let processed = result.unwrap();
        assert_eq!(processed.shape(), &[28, 28, 3]);
    }

    #[test]
    fn test_invalid_channels() {
        let config = GemmaImageProcessorConfig::default();
        let processor = GemmaImageProcessor::new(config);

        let rgba_image = Array3::<u8>::zeros((896, 896, 4));
        let result = processor.preprocess(&rgba_image);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("3 channels"));
    }

    #[test]
    fn test_num_image_tokens() {
        let config = GemmaImageProcessorConfig {
            image_size: 896,
            ..Default::default()
        };
        let processor = GemmaImageProcessor::new(config);

        let num_tokens = processor.num_image_tokens(1024, 768);
        assert_eq!(num_tokens, 4096); // (896/14)^2 = 64^2 = 4096
    }

    #[test]
    fn test_bilinear_resampling() {
        let config = GemmaImageProcessorConfig {
            image_size: 16,
            resample: ResampleMethod::Bilinear,
            ..Default::default()
        };
        let processor = GemmaImageProcessor::new(config);

        let mut image = Array3::<u8>::zeros((4, 4, 3));
        image.fill(128);

        let result = processor.preprocess(&image).unwrap();

        assert_eq!(result.shape(), &[16, 16, 3]);
        for &val in result.iter() {
            assert!(
                val >= -3.0 && val <= 3.0,
                "Normalized value {} out of expected range",
                val
            );
        }
    }

    #[test]
    fn test_nearest_neighbor_resampling() {
        let config = GemmaImageProcessorConfig {
            image_size: 8,
            resample: ResampleMethod::Nearest,
            do_normalize: false,
            ..Default::default()
        };
        let processor = GemmaImageProcessor::new(config);

        let mut image = Array3::<u8>::zeros((4, 4, 3));
        image.fill(200);

        let result = processor.preprocess(&image).unwrap();

        assert_eq!(result.shape(), &[8, 8, 3]);
        assert!((result[[0, 0, 0]] - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_no_resize_option() {
        let config = GemmaImageProcessorConfig {
            do_resize: false,
            do_normalize: false,
            ..Default::default()
        };
        let processor = GemmaImageProcessor::new(config);

        let image = Array3::<u8>::zeros((100, 100, 3));
        let result = processor.preprocess(&image).unwrap();

        assert_eq!(result.shape(), &[100, 100, 3]);
        assert!((result[[50, 50, 0]] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_normalization_values() {
        let config = GemmaImageProcessorConfig {
            image_size: 4,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            do_resize: false,
            ..Default::default()
        };
        let processor = GemmaImageProcessor::new(config);

        let mut image = Array3::<u8>::zeros((4, 4, 3));
        image.fill(255); // white pixel

        let result = processor.preprocess(&image).unwrap();

        // white pixel: (255/255 - 0.5) / 0.5 = (1.0 - 0.5) / 0.5 = 1.0
        assert!((result[[0, 0, 0]] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_image_error() {
        let config = GemmaImageProcessorConfig::default();
        let processor = GemmaImageProcessor::new(config);

        let empty_image = Array3::<u8>::zeros((0, 0, 3));
        let result = processor.preprocess(&empty_image);

        assert!(result.is_err());
    }
}
