//! Output postprocessing for neural inference.
//!
//! This module handles:
//! - Denormalization (reversing input normalization)
//! - Clipping to valid pixel range
//! - Color space restoration
//! - Padding removal
//! - NCHW to HWC conversion

use crate::{NeuralError, NeuralFrame, Result};
use crate::preprocessing::{NormalizationMode, ChannelOrder, PreprocessedInput};

/// Postprocessing configuration.
#[derive(Debug, Clone)]
pub struct PostprocessConfig {
    /// Normalization mode used during preprocessing (for denormalization).
    pub normalization: NormalizationMode,
    /// Output channel order from model.
    pub model_output_order: ChannelOrder,
    /// Desired output channel order.
    pub output_order: ChannelOrder,
    /// Clip output to [0, 1] range.
    pub clip_output: bool,
    /// Apply gamma correction.
    pub gamma: Option<f32>,
}

impl Default for PostprocessConfig {
    fn default() -> Self {
        Self {
            normalization: NormalizationMode::ZeroOne,
            model_output_order: ChannelOrder::Rgb,
            output_order: ChannelOrder::Rgb,
            clip_output: true,
            gamma: None,
        }
    }
}

/// Postprocessor for neural network output.
pub struct Postprocessor {
    config: PostprocessConfig,
}

impl Postprocessor {
    /// Create a new postprocessor with the given configuration.
    pub fn new(config: PostprocessConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PostprocessConfig::default())
    }

    /// Process NCHW output tensor to NeuralFrame.
    #[allow(clippy::too_many_arguments)]
    pub fn process_nchw(
        &self,
        data: &[f32],
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        preprocess_info: Option<&PreprocessedInput>,
        scale: u32,
    ) -> Result<Vec<NeuralFrame>> {
        if channels != 3 {
            return Err(NeuralError::InvalidInput(format!(
                "Expected 3 channels, got {}",
                channels
            )));
        }

        let expected_size = batch * channels * height * width;
        if data.len() != expected_size {
            return Err(NeuralError::InvalidInput(format!(
                "Expected {} elements, got {}",
                expected_size,
                data.len()
            )));
        }

        let frame_size = channels * height * width;
        let mut frames = Vec::with_capacity(batch);

        for b in 0..batch {
            let batch_offset = b * frame_size;

            // Calculate output dimensions (remove padding if needed)
            let (out_width, out_height) = if let Some(info) = preprocess_info {
                (
                    (info.original_width * scale) as usize,
                    (info.original_height * scale) as usize,
                )
            } else {
                (width, height)
            };

            let mut frame = NeuralFrame::new(out_width as u32, out_height as u32);

            for y in 0..out_height {
                for x in 0..out_width {
                    for c in 0..3 {
                        let src_channel = self.map_channel(c);
                        let nchw_idx =
                            batch_offset + src_channel * height * width + y * width + x;
                        let hwc_idx = y * out_width * 3 + x * 3 + c;

                        let value = data.get(nchw_idx).copied().unwrap_or(0.0);
                        let processed = self.postprocess_value(value, c);
                        frame.data[hwc_idx] = processed;
                    }
                }
            }

            frames.push(frame);
        }

        Ok(frames)
    }

    /// Process a single output tensor.
    pub fn process_single(
        &self,
        data: &[f32],
        height: usize,
        width: usize,
        preprocess_info: Option<&PreprocessedInput>,
        scale: u32,
    ) -> Result<NeuralFrame> {
        let frames = self.process_nchw(data, 1, 3, height, width, preprocess_info, scale)?;
        frames.into_iter().next().ok_or_else(|| {
            NeuralError::Inference("No output frame generated".to_string())
        })
    }

    /// Process HWC output data.
    pub fn process_hwc(
        &self,
        data: &[f32],
        height: usize,
        width: usize,
        preprocess_info: Option<&PreprocessedInput>,
        scale: u32,
    ) -> Result<NeuralFrame> {
        let (out_width, out_height) = if let Some(info) = preprocess_info {
            (
                (info.original_width * scale) as usize,
                (info.original_height * scale) as usize,
            )
        } else {
            (width, height)
        };

        let mut frame = NeuralFrame::new(out_width as u32, out_height as u32);

        for y in 0..out_height {
            for x in 0..out_width {
                for c in 0..3 {
                    let src_channel = self.map_channel(c);
                    let src_idx = y * width * 3 + x * 3 + src_channel;
                    let dst_idx = y * out_width * 3 + x * 3 + c;

                    let value = data.get(src_idx).copied().unwrap_or(0.0);
                    let processed = self.postprocess_value(value, c);
                    frame.data[dst_idx] = processed;
                }
            }
        }

        Ok(frame)
    }

    /// Postprocess a single value.
    fn postprocess_value(&self, value: f32, channel: usize) -> f32 {
        let mut v = self.denormalize(value, channel);

        if self.config.clip_output {
            v = v.clamp(0.0, 1.0);
        }

        if let Some(gamma) = self.config.gamma {
            v = v.powf(1.0 / gamma);
        }

        v
    }

    /// Denormalize a value.
    fn denormalize(&self, value: f32, channel: usize) -> f32 {
        match self.config.normalization {
            NormalizationMode::ZeroOne => value,
            NormalizationMode::NegOneOne => (value + 1.0) / 2.0,
            NormalizationMode::ImageNet => {
                const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
                const STD: [f32; 3] = [0.229, 0.224, 0.225];
                value * STD[channel] + MEAN[channel]
            }
            NormalizationMode::Custom { mean, std } => {
                value * std[channel] + mean[channel]
            }
            NormalizationMode::None => value,
        }
    }

    /// Map output channel for color space conversion.
    fn map_channel(&self, output_channel: usize) -> usize {
        if self.config.model_output_order != self.config.output_order {
            // Swap R and B
            match output_channel {
                0 => 2,
                2 => 0,
                c => c,
            }
        } else {
            output_channel
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &PostprocessConfig {
        &self.config
    }
}

/// Convert NCHW tensor to HWC format.
pub fn nchw_to_hwc(data: &[f32], channels: usize, height: usize, width: usize) -> Vec<f32> {
    let size = channels * height * width;
    let mut hwc = vec![0.0f32; size];

    for c in 0..channels {
        for h in 0..height {
            for w in 0..width {
                let nchw_idx = c * height * width + h * width + w;
                let hwc_idx = h * width * channels + w * channels + c;
                if nchw_idx < data.len() {
                    hwc[hwc_idx] = data[nchw_idx];
                }
            }
        }
    }

    hwc
}

/// Remove padding from output image.
pub fn remove_padding(
    data: &[f32],
    padded_width: usize,
    padded_height: usize,
    original_width: usize,
    original_height: usize,
    channels: usize,
) -> Result<Vec<f32>> {
    if original_width > padded_width || original_height > padded_height {
        return Err(NeuralError::InvalidInput(
            "Original size cannot be larger than padded size".to_string(),
        ));
    }

    let output_size = original_width * original_height * channels;
    let mut output = vec![0.0f32; output_size];

    for y in 0..original_height {
        for x in 0..original_width {
            for c in 0..channels {
                let src_idx = y * padded_width * channels + x * channels + c;
                let dst_idx = y * original_width * channels + x * channels + c;
                output[dst_idx] = data.get(src_idx).copied().unwrap_or(0.0);
            }
        }
    }

    Ok(output)
}

/// Convert f32 [0-1] to u8 [0-255].
pub fn to_u8(data: &[f32]) -> Vec<u8> {
    data.iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// Convert f32 [0-1] to u16 [0-65535].
pub fn to_u16(data: &[f32]) -> Vec<u16> {
    data.iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16)
        .collect()
}

/// Apply gamma correction.
pub fn apply_gamma(data: &mut [f32], gamma: f32) {
    let inv_gamma = 1.0 / gamma;
    for v in data.iter_mut() {
        if *v > 0.0 {
            *v = v.powf(inv_gamma);
        }
    }
}

/// Clip values to [min, max] range.
pub fn clip_range(data: &mut [f32], min: f32, max: f32) {
    for v in data.iter_mut() {
        *v = v.clamp(min, max);
    }
}

/// Output result combining frame data with metadata.
#[derive(Debug, Clone)]
pub struct PostprocessedOutput {
    /// The output frame.
    pub frame: NeuralFrame,
    /// Whether padding was removed.
    pub padding_removed: bool,
    /// Scale factor applied.
    pub scale: u32,
    /// Processing time in milliseconds.
    pub processing_time_ms: Option<f64>,
}

impl PostprocessedOutput {
    /// Create a new postprocessed output.
    pub fn new(frame: NeuralFrame, padding_removed: bool, scale: u32) -> Self {
        Self {
            frame,
            padding_removed,
            scale,
            processing_time_ms: None,
        }
    }

    /// Get output as u8 bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        to_u8(&self.frame.data)
    }

    /// Get output dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.frame.width, self.frame.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denormalize_zero_one() {
        let config = PostprocessConfig {
            normalization: NormalizationMode::ZeroOne,
            ..Default::default()
        };
        let postprocessor = Postprocessor::new(config);

        assert!((postprocessor.denormalize(0.5, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_denormalize_neg_one_one() {
        let config = PostprocessConfig {
            normalization: NormalizationMode::NegOneOne,
            ..Default::default()
        };
        let postprocessor = Postprocessor::new(config);

        assert!((postprocessor.denormalize(-1.0, 0) - 0.0).abs() < 1e-6);
        assert!((postprocessor.denormalize(0.0, 0) - 0.5).abs() < 1e-6);
        assert!((postprocessor.denormalize(1.0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nchw_to_hwc() {
        // NCHW: 1x3x2x2
        let nchw = vec![
            1.0, 2.0, 3.0, 4.0, // Channel 0
            5.0, 6.0, 7.0, 8.0, // Channel 1
            9.0, 10.0, 11.0, 12.0, // Channel 2
        ];

        let hwc = nchw_to_hwc(&nchw, 3, 2, 2);

        // HWC format: [(0,0)RGB, (0,1)RGB, (1,0)RGB, (1,1)RGB]
        assert_eq!(hwc[0], 1.0); // (0,0) C0
        assert_eq!(hwc[1], 5.0); // (0,0) C1
        assert_eq!(hwc[2], 9.0); // (0,0) C2
        assert_eq!(hwc[3], 2.0); // (0,1) C0
    }

    #[test]
    fn test_remove_padding() {
        let padded = vec![
            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, // Row 0: 2 pixels + 1 pad
            3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, // Row 1: 2 pixels + 1 pad
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Row 2: all pad
        ];

        let unpadded = remove_padding(&padded, 3, 3, 2, 2, 3).unwrap();

        assert_eq!(unpadded.len(), 12); // 2x2x3
        assert_eq!(unpadded[0..3], [1.0, 1.0, 1.0]);
        assert_eq!(unpadded[3..6], [2.0, 2.0, 2.0]);
        assert_eq!(unpadded[6..9], [3.0, 3.0, 3.0]);
        assert_eq!(unpadded[9..12], [4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_to_u8() {
        let data = vec![0.0, 0.5, 1.0, -0.1, 1.5];
        let bytes = to_u8(&data);

        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 128);
        assert_eq!(bytes[2], 255);
        assert_eq!(bytes[3], 0); // Clamped
        assert_eq!(bytes[4], 255); // Clamped
    }

    #[test]
    fn test_clip_range() {
        let mut data = vec![-0.5, 0.0, 0.5, 1.0, 1.5];
        clip_range(&mut data, 0.0, 1.0);

        assert_eq!(data, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn test_apply_gamma() {
        let mut data = vec![0.0, 0.25, 1.0];
        apply_gamma(&mut data, 2.2);

        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!(data[1] > 0.25); // Gamma corrected should be brighter
        assert!((data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_process_nchw_single() {
        let config = PostprocessConfig::default();
        let postprocessor = Postprocessor::new(config);

        // 1x3x2x2 NCHW tensor
        let data = vec![
            0.1, 0.2, 0.3, 0.4, // R
            0.5, 0.6, 0.7, 0.8, // G
            0.9, 1.0, 0.1, 0.2, // B
        ];

        let frame = postprocessor.process_single(&data, 2, 2, None, 1).unwrap();

        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.data.len(), 12);
    }

    #[test]
    fn test_postprocessed_output() {
        let frame = NeuralFrame::new(10, 10);
        let output = PostprocessedOutput::new(frame, false, 4);

        assert_eq!(output.dimensions(), (10, 10));
        assert_eq!(output.scale, 4);
        assert!(!output.padding_removed);
    }

    #[test]
    fn test_color_space_conversion() {
        let config = PostprocessConfig {
            model_output_order: ChannelOrder::Bgr,
            output_order: ChannelOrder::Rgb,
            ..Default::default()
        };
        let postprocessor = Postprocessor::new(config);

        assert_eq!(postprocessor.map_channel(0), 2); // R from B
        assert_eq!(postprocessor.map_channel(1), 1); // G stays
        assert_eq!(postprocessor.map_channel(2), 0); // B from R
    }
}
