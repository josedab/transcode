//! Input preprocessing for neural inference.
//!
//! This module handles:
//! - Image normalization (0-1 or -1 to 1 range)
//! - Color space conversion (BGR to RGB)
//! - Padding for model input size requirements
//! - Batch dimension handling
//! - HWC to NCHW layout conversion

use crate::{NeuralError, NeuralFrame, Result};

/// Normalization mode for input data.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum NormalizationMode {
    /// Normalize to [0, 1] range.
    #[default]
    ZeroOne,
    /// Normalize to [-1, 1] range.
    NegOneOne,
    /// Normalize with ImageNet mean and std.
    ImageNet,
    /// Custom mean and standard deviation per channel.
    Custom { mean: [f32; 3], std: [f32; 3] },
    /// No normalization (data already in expected range).
    None,
}


/// Color channel order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelOrder {
    /// RGB order (default for most models).
    #[default]
    Rgb,
    /// BGR order (OpenCV default).
    Bgr,
}

/// Padding mode for image alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingMode {
    /// Reflect padding (mirror at edges).
    #[default]
    Reflect,
    /// Replicate edge pixels.
    Replicate,
    /// Zero padding.
    Zero,
    /// Constant value padding.
    Constant(u8),
}

/// Preprocessing configuration.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Normalization mode.
    pub normalization: NormalizationMode,
    /// Input channel order.
    pub input_order: ChannelOrder,
    /// Output channel order (model expectation).
    pub output_order: ChannelOrder,
    /// Minimum input size (for padding).
    pub min_size: Option<(u32, u32)>,
    /// Size must be multiple of this value.
    pub size_multiple: Option<u32>,
    /// Padding mode.
    pub padding_mode: PaddingMode,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            normalization: NormalizationMode::ZeroOne,
            input_order: ChannelOrder::Rgb,
            output_order: ChannelOrder::Rgb,
            min_size: None,
            size_multiple: Some(8), // Many models require multiples of 8
            padding_mode: PaddingMode::Reflect,
        }
    }
}

/// Preprocessor for neural network input.
pub struct Preprocessor {
    config: PreprocessConfig,
}

impl Preprocessor {
    /// Create a new preprocessor with the given configuration.
    pub fn new(config: PreprocessConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PreprocessConfig::default())
    }

    /// Process a NeuralFrame for inference.
    pub fn process(&self, frame: &NeuralFrame) -> Result<PreprocessedInput> {
        // Calculate padded dimensions
        let (padded_width, padded_height) = self.calculate_padded_size(frame.width, frame.height);
        let pad_right = padded_width - frame.width;
        let pad_bottom = padded_height - frame.height;

        // Allocate output buffer
        let output_size = (padded_width * padded_height * 3) as usize;
        let mut output = vec![0.0f32; output_size];

        // Process pixels
        for y in 0..padded_height {
            for x in 0..padded_width {
                // Get source coordinates (with padding logic)
                let (src_x, src_y) = self.get_source_coords(
                    x,
                    y,
                    frame.width,
                    frame.height,
                    padded_width,
                    padded_height,
                );

                for c in 0..3 {
                    let src_channel = self.map_channel(c);
                    let src_idx = ((src_y * frame.width + src_x) * 3 + src_channel) as usize;
                    let dst_idx = ((y * padded_width + x) * 3 + c) as usize;

                    let value = frame.data.get(src_idx).copied().unwrap_or(0.0);
                    output[dst_idx] = self.normalize(value, c);
                }
            }
        }

        Ok(PreprocessedInput {
            data: output,
            width: padded_width,
            height: padded_height,
            original_width: frame.width,
            original_height: frame.height,
            pad_right,
            pad_bottom,
        })
    }

    /// Process raw u8 image data.
    pub fn process_u8(&self, data: &[u8], width: u32, height: u32) -> Result<PreprocessedInput> {
        let expected_size = (width * height * 3) as usize;
        if data.len() != expected_size {
            return Err(NeuralError::InvalidInput(format!(
                "Expected {} bytes, got {}",
                expected_size,
                data.len()
            )));
        }

        // Convert to float first
        let float_data: Vec<f32> = data.iter().map(|&v| v as f32 / 255.0).collect();
        let frame = NeuralFrame::from_rgb(float_data, width, height)?;
        self.process(&frame)
    }

    /// Process a batch of frames.
    pub fn process_batch(&self, frames: &[&NeuralFrame]) -> Result<BatchPreprocessedInput> {
        if frames.is_empty() {
            return Err(NeuralError::InvalidInput("Empty batch".to_string()));
        }

        // All frames must have same dimensions
        let first = frames[0];
        for frame in frames.iter().skip(1) {
            if frame.width != first.width || frame.height != first.height {
                return Err(NeuralError::InvalidInput(
                    "All frames in batch must have same dimensions".to_string(),
                ));
            }
        }

        let processed: Vec<PreprocessedInput> = frames
            .iter()
            .map(|f| self.process(f))
            .collect::<Result<Vec<_>>>()?;

        let (padded_width, padded_height) = self.calculate_padded_size(first.width, first.height);

        Ok(BatchPreprocessedInput {
            inputs: processed,
            batch_size: frames.len(),
            width: padded_width,
            height: padded_height,
        })
    }

    /// Calculate padded dimensions.
    fn calculate_padded_size(&self, width: u32, height: u32) -> (u32, u32) {
        let mut w = width;
        let mut h = height;

        // Apply minimum size
        if let Some((min_w, min_h)) = self.config.min_size {
            w = w.max(min_w);
            h = h.max(min_h);
        }

        // Apply size multiple
        if let Some(multiple) = self.config.size_multiple {
            w = w.div_ceil(multiple) * multiple;
            h = h.div_ceil(multiple) * multiple;
        }

        (w, h)
    }

    /// Get source coordinates for a destination pixel (handles padding).
    fn get_source_coords(
        &self,
        x: u32,
        y: u32,
        src_width: u32,
        src_height: u32,
        _dst_width: u32,
        _dst_height: u32,
    ) -> (u32, u32) {
        let src_x = if x < src_width {
            x
        } else {
            match self.config.padding_mode {
                PaddingMode::Reflect => {
                    let overflow = x - src_width + 1;
                    src_width.saturating_sub(overflow).saturating_sub(1)
                }
                PaddingMode::Replicate => src_width - 1,
                PaddingMode::Zero | PaddingMode::Constant(_) => src_width - 1,
            }
        };

        let src_y = if y < src_height {
            y
        } else {
            match self.config.padding_mode {
                PaddingMode::Reflect => {
                    let overflow = y - src_height + 1;
                    src_height.saturating_sub(overflow).saturating_sub(1)
                }
                PaddingMode::Replicate => src_height - 1,
                PaddingMode::Zero | PaddingMode::Constant(_) => src_height - 1,
            }
        };

        (src_x, src_y)
    }

    /// Map output channel to input channel (for BGR/RGB conversion).
    fn map_channel(&self, output_channel: u32) -> u32 {
        if self.config.input_order != self.config.output_order {
            // Swap R and B channels
            match output_channel {
                0 => 2, // R -> B
                2 => 0, // B -> R
                c => c,
            }
        } else {
            output_channel
        }
    }

    /// Normalize a pixel value.
    fn normalize(&self, value: f32, channel: u32) -> f32 {
        match self.config.normalization {
            NormalizationMode::ZeroOne => value, // Already in 0-1
            NormalizationMode::NegOneOne => value * 2.0 - 1.0,
            NormalizationMode::ImageNet => {
                const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
                const STD: [f32; 3] = [0.229, 0.224, 0.225];
                (value - MEAN[channel as usize]) / STD[channel as usize]
            }
            NormalizationMode::Custom { mean, std } => {
                (value - mean[channel as usize]) / std[channel as usize]
            }
            NormalizationMode::None => value,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &PreprocessConfig {
        &self.config
    }
}

/// Preprocessed input ready for inference.
#[derive(Debug, Clone)]
pub struct PreprocessedInput {
    /// Preprocessed data (HWC layout, normalized).
    pub data: Vec<f32>,
    /// Padded width.
    pub width: u32,
    /// Padded height.
    pub height: u32,
    /// Original width before padding.
    pub original_width: u32,
    /// Original height before padding.
    pub original_height: u32,
    /// Padding added on right.
    pub pad_right: u32,
    /// Padding added on bottom.
    pub pad_bottom: u32,
}

impl PreprocessedInput {
    /// Convert to NCHW tensor format.
    pub fn to_nchw(&self) -> Vec<f32> {
        let channels = 3usize;
        let height = self.height as usize;
        let width = self.width as usize;
        let size = channels * height * width;
        let mut nchw = vec![0.0f32; size];

        for h in 0..height {
            for w in 0..width {
                for c in 0..channels {
                    let hwc_idx = h * width * channels + w * channels + c;
                    let nchw_idx = c * height * width + h * width + w;
                    nchw[nchw_idx] = self.data[hwc_idx];
                }
            }
        }

        nchw
    }

    /// Get tensor shape as [1, C, H, W].
    pub fn shape(&self) -> [usize; 4] {
        [1, 3, self.height as usize, self.width as usize]
    }

    /// Check if padding was applied.
    pub fn is_padded(&self) -> bool {
        self.pad_right > 0 || self.pad_bottom > 0
    }
}

/// Batch of preprocessed inputs.
#[derive(Debug, Clone)]
pub struct BatchPreprocessedInput {
    /// Individual preprocessed inputs.
    pub inputs: Vec<PreprocessedInput>,
    /// Batch size.
    pub batch_size: usize,
    /// Common width (all same after padding).
    pub width: u32,
    /// Common height.
    pub height: u32,
}

impl BatchPreprocessedInput {
    /// Convert to NCHW tensor format with batch dimension.
    pub fn to_nchw(&self) -> Vec<f32> {
        let channels = 3usize;
        let height = self.height as usize;
        let width = self.width as usize;
        let frame_size = channels * height * width;
        let mut nchw = vec![0.0f32; self.batch_size * frame_size];

        for (b, input) in self.inputs.iter().enumerate() {
            let batch_offset = b * frame_size;
            for h in 0..height {
                for w in 0..width {
                    for c in 0..channels {
                        let hwc_idx = h * width * channels + w * channels + c;
                        let nchw_idx = batch_offset + c * height * width + h * width + w;
                        if hwc_idx < input.data.len() {
                            nchw[nchw_idx] = input.data[hwc_idx];
                        }
                    }
                }
            }
        }

        nchw
    }

    /// Get tensor shape as [N, C, H, W].
    pub fn shape(&self) -> [usize; 4] {
        [self.batch_size, 3, self.height as usize, self.width as usize]
    }
}

/// Convert BGR u8 image to RGB f32 normalized.
pub fn bgr_to_rgb_f32(bgr: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
    let expected = (width * height * 3) as usize;
    if bgr.len() != expected {
        return Err(NeuralError::InvalidInput(format!(
            "Expected {} bytes, got {}",
            expected,
            bgr.len()
        )));
    }

    let mut rgb = vec![0.0f32; expected];
    for i in 0..(width * height) as usize {
        let idx = i * 3;
        rgb[idx] = bgr[idx + 2] as f32 / 255.0;     // B -> R
        rgb[idx + 1] = bgr[idx + 1] as f32 / 255.0; // G -> G
        rgb[idx + 2] = bgr[idx] as f32 / 255.0;     // R -> B
    }

    Ok(rgb)
}

/// Pad image to specified size.
pub fn pad_image(
    data: &[f32],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    mode: PaddingMode,
) -> Result<Vec<f32>> {
    if target_width < width || target_height < height {
        return Err(NeuralError::InvalidInput(
            "Target size must be >= source size".to_string(),
        ));
    }

    let channels = 3u32;
    let output_size = (target_width * target_height * channels) as usize;
    let mut output = vec![0.0f32; output_size];

    let pad_value = match mode {
        PaddingMode::Constant(v) => v as f32 / 255.0,
        _ => 0.0,
    };

    for y in 0..target_height {
        for x in 0..target_width {
            let (src_x, src_y) = if x < width && y < height {
                (x, y)
            } else {
                match mode {
                    PaddingMode::Zero => {
                        for c in 0..channels {
                            let idx = ((y * target_width + x) * channels + c) as usize;
                            output[idx] = 0.0;
                        }
                        continue;
                    }
                    PaddingMode::Constant(_) => {
                        for c in 0..channels {
                            let idx = ((y * target_width + x) * channels + c) as usize;
                            output[idx] = pad_value;
                        }
                        continue;
                    }
                    PaddingMode::Replicate => (x.min(width - 1), y.min(height - 1)),
                    PaddingMode::Reflect => {
                        let sx = if x >= width {
                            width - 1 - (x - width).min(width - 1)
                        } else {
                            x
                        };
                        let sy = if y >= height {
                            height - 1 - (y - height).min(height - 1)
                        } else {
                            y
                        };
                        (sx, sy)
                    }
                }
            };

            for c in 0..channels {
                let src_idx = ((src_y * width + src_x) * channels + c) as usize;
                let dst_idx = ((y * target_width + x) * channels + c) as usize;
                output[dst_idx] = data.get(src_idx).copied().unwrap_or(0.0);
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_zero_one() {
        let config = PreprocessConfig {
            normalization: NormalizationMode::ZeroOne,
            ..Default::default()
        };
        let preprocessor = Preprocessor::new(config);

        assert!((preprocessor.normalize(0.5, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalization_neg_one_one() {
        let config = PreprocessConfig {
            normalization: NormalizationMode::NegOneOne,
            ..Default::default()
        };
        let preprocessor = Preprocessor::new(config);

        assert!((preprocessor.normalize(0.0, 0) - (-1.0)).abs() < 1e-6);
        assert!((preprocessor.normalize(0.5, 0) - 0.0).abs() < 1e-6);
        assert!((preprocessor.normalize(1.0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bgr_rgb_conversion() {
        let preprocessor = Preprocessor::new(PreprocessConfig {
            input_order: ChannelOrder::Bgr,
            output_order: ChannelOrder::Rgb,
            ..Default::default()
        });

        assert_eq!(preprocessor.map_channel(0), 2); // R gets from B
        assert_eq!(preprocessor.map_channel(1), 1); // G stays
        assert_eq!(preprocessor.map_channel(2), 0); // B gets from R
    }

    #[test]
    fn test_calculate_padded_size() {
        let config = PreprocessConfig {
            size_multiple: Some(16),
            min_size: Some((32, 32)),
            ..Default::default()
        };
        let preprocessor = Preprocessor::new(config);

        // Small image gets padded to minimum
        let (w, h) = preprocessor.calculate_padded_size(10, 10);
        assert_eq!(w, 32);
        assert_eq!(h, 32);

        // Non-multiple gets padded up
        let (w, h) = preprocessor.calculate_padded_size(50, 50);
        assert_eq!(w, 64);
        assert_eq!(h, 64);

        // Already multiple stays same
        let (w, h) = preprocessor.calculate_padded_size(64, 64);
        assert_eq!(w, 64);
        assert_eq!(h, 64);
    }

    #[test]
    fn test_process_frame() {
        let config = PreprocessConfig {
            size_multiple: Some(8),
            ..Default::default()
        };
        let preprocessor = Preprocessor::new(config);

        let mut frame = NeuralFrame::new(10, 10);
        for i in 0..frame.data.len() {
            frame.data[i] = 0.5;
        }

        let result = preprocessor.process(&frame).unwrap();

        // Should be padded to multiple of 8
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
        assert_eq!(result.original_width, 10);
        assert_eq!(result.original_height, 10);
        assert_eq!(result.pad_right, 6);
        assert_eq!(result.pad_bottom, 6);
    }

    #[test]
    fn test_preprocessed_to_nchw() {
        let input = PreprocessedInput {
            data: vec![
                1.0, 2.0, 3.0, // (0,0)
                4.0, 5.0, 6.0, // (0,1)
                7.0, 8.0, 9.0, // (1,0)
                10.0, 11.0, 12.0, // (1,1)
            ],
            width: 2,
            height: 2,
            original_width: 2,
            original_height: 2,
            pad_right: 0,
            pad_bottom: 0,
        };

        let nchw = input.to_nchw();

        // Channel 0 (R): [1, 4, 7, 10]
        assert_eq!(nchw[0], 1.0);
        assert_eq!(nchw[1], 4.0);
        assert_eq!(nchw[2], 7.0);
        assert_eq!(nchw[3], 10.0);

        // Channel 1 (G): [2, 5, 8, 11]
        assert_eq!(nchw[4], 2.0);
        assert_eq!(nchw[5], 5.0);
    }

    #[test]
    fn test_bgr_to_rgb_f32() {
        let bgr = vec![255, 128, 0, 255, 128, 0]; // 2 BGR pixels
        let rgb = bgr_to_rgb_f32(&bgr, 2, 1).unwrap();

        // First pixel: BGR(255, 128, 0) -> RGB(0, 128, 255)
        assert!((rgb[0] - 0.0).abs() < 0.01); // R
        assert!((rgb[1] - 0.502).abs() < 0.01); // G
        assert!((rgb[2] - 1.0).abs() < 0.01); // B
    }

    #[test]
    fn test_pad_image_replicate() {
        let data = vec![1.0, 2.0, 3.0]; // 1x1 RGB pixel
        let padded = pad_image(&data, 1, 1, 2, 2, PaddingMode::Replicate).unwrap();

        // All pixels should be the same (replicated)
        assert_eq!(padded.len(), 12); // 2x2x3
        assert_eq!(padded[0..3], [1.0, 2.0, 3.0]);
        assert_eq!(padded[3..6], [1.0, 2.0, 3.0]);
        assert_eq!(padded[6..9], [1.0, 2.0, 3.0]);
        assert_eq!(padded[9..12], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_batch_preprocess() {
        let preprocessor = Preprocessor::new(PreprocessConfig {
            size_multiple: Some(4),
            ..Default::default()
        });

        let frame1 = NeuralFrame::new(4, 4);
        let frame2 = NeuralFrame::new(4, 4);

        let batch = preprocessor.process_batch(&[&frame1, &frame2]).unwrap();

        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.shape(), [2, 3, 4, 4]);
    }
}
