//! Video processing pipelines for GenAI operations.

use crate::models::{ModelBackend, ModelInfo, ModelPrecision, ModelRegistry, ModelType};
use crate::{GenAiError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Frame data for processing.
#[derive(Debug, Clone)]
pub struct FrameData {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Pixel data (RGB or RGBA).
    pub data: Vec<u8>,
    /// Number of channels.
    pub channels: u8,
    /// Frame timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl FrameData {
    /// Create new frame data.
    pub fn new(width: u32, height: u32, channels: u8) -> Self {
        let size = (width * height * channels as u32) as usize;
        Self {
            width,
            height,
            data: vec![0; size],
            channels,
            timestamp_ms: 0,
        }
    }

    /// Create from raw data.
    pub fn from_raw(width: u32, height: u32, channels: u8, data: Vec<u8>) -> Result<Self> {
        let expected_size = (width * height * channels as u32) as usize;
        if data.len() != expected_size {
            return Err(GenAiError::InvalidInput(format!(
                "Data size mismatch: expected {}, got {}",
                expected_size,
                data.len()
            )));
        }

        Ok(Self {
            width,
            height,
            data,
            channels,
            timestamp_ms: 0,
        })
    }

    /// Set timestamp.
    pub fn with_timestamp(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = timestamp_ms;
        self
    }

    /// Get pixel at coordinates.
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<&[u8]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y * self.width + x) * self.channels as u32) as usize;
        Some(&self.data[idx..idx + self.channels as usize])
    }

    /// Set pixel at coordinates.
    pub fn set_pixel(&mut self, x: u32, y: u32, pixel: &[u8]) {
        if x >= self.width || y >= self.height || pixel.len() != self.channels as usize {
            return;
        }
        let idx = ((y * self.width + x) * self.channels as u32) as usize;
        self.data[idx..idx + self.channels as usize].copy_from_slice(pixel);
    }

    /// Convert to normalized float tensor [0, 1].
    pub fn to_float_tensor(&self) -> Vec<f32> {
        self.data.iter().map(|&b| b as f32 / 255.0).collect()
    }

    /// Create from normalized float tensor.
    pub fn from_float_tensor(
        width: u32,
        height: u32,
        channels: u8,
        tensor: &[f32],
    ) -> Result<Self> {
        let expected_size = (width * height * channels as u32) as usize;
        if tensor.len() != expected_size {
            return Err(GenAiError::InvalidInput(format!(
                "Tensor size mismatch: expected {}, got {}",
                expected_size,
                tensor.len()
            )));
        }

        let data: Vec<u8> = tensor
            .iter()
            .map(|&f| (f.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        Ok(Self {
            width,
            height,
            data,
            channels,
            timestamp_ms: 0,
        })
    }
}

/// Processor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Model ID to use.
    pub model_id: String,
    /// Preferred backend.
    pub backend: ModelBackend,
    /// Model precision.
    pub precision: ModelPrecision,
    /// Batch size.
    pub batch_size: usize,
    /// Device ID (for GPU).
    pub device_id: i32,
    /// Enable model caching.
    pub cache_model: bool,
    /// Custom parameters.
    pub params: HashMap<String, String>,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            backend: ModelBackend::Onnx,
            precision: ModelPrecision::Fp16,
            batch_size: 1,
            device_id: 0,
            cache_model: true,
            params: HashMap::new(),
        }
    }
}

impl ProcessorConfig {
    /// Create config for a specific model.
    pub fn for_model(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Default::default()
        }
    }

    /// Set backend.
    pub fn backend(mut self, backend: ModelBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set precision.
    pub fn precision(mut self, precision: ModelPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set device ID.
    pub fn device(mut self, id: i32) -> Self {
        self.device_id = id;
        self
    }

    /// Add custom parameter.
    pub fn param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
}

/// Trait for frame processors.
pub trait FrameProcessor: Send + Sync {
    /// Process a single frame.
    fn process(&self, frame: &FrameData) -> Result<FrameData>;

    /// Process multiple frames.
    fn process_batch(&self, frames: &[FrameData]) -> Result<Vec<FrameData>> {
        frames.iter().map(|f| self.process(f)).collect()
    }

    /// Get model info.
    fn model_info(&self) -> &ModelInfo;

    /// Get processor name.
    fn name(&self) -> &str;
}

/// Super resolution processor.
pub struct SuperResolutionProcessor {
    config: ProcessorConfig,
    model_info: ModelInfo,
    scale_factor: u32,
}

impl SuperResolutionProcessor {
    /// Create a new super resolution processor.
    pub fn new(config: ProcessorConfig, registry: &ModelRegistry) -> Result<Self> {
        let model_info = registry
            .get(&config.model_id)
            .ok_or_else(|| GenAiError::ModelNotFound(config.model_id.clone()))?
            .clone();

        if model_info.model_type != ModelType::SuperResolution {
            return Err(GenAiError::InvalidInput(format!(
                "Model {} is not a super resolution model",
                config.model_id
            )));
        }

        // Determine scale factor from model ID
        let scale_factor = if config.model_id.contains("x4") {
            4
        } else {
            2 // Default (also for x2)
        };

        Ok(Self {
            config,
            model_info,
            scale_factor,
        })
    }

    /// Get scale factor.
    pub fn scale_factor(&self) -> u32 {
        self.scale_factor
    }
}

impl FrameProcessor for SuperResolutionProcessor {
    fn process(&self, frame: &FrameData) -> Result<FrameData> {
        // Simulated processing - in production this would run the actual model
        let new_width = frame.width * self.scale_factor;
        let new_height = frame.height * self.scale_factor;

        tracing::debug!(
            model = %self.config.model_id,
            input_size = %format!("{}x{}", frame.width, frame.height),
            output_size = %format!("{}x{}", new_width, new_height),
            "Processing super resolution"
        );

        // Simple bilinear interpolation as placeholder
        let mut output = FrameData::new(new_width, new_height, frame.channels);
        output.timestamp_ms = frame.timestamp_ms;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 / self.scale_factor as f32) as u32;
                let src_y = (y as f32 / self.scale_factor as f32) as u32;

                if let Some(pixel) = frame.get_pixel(src_x.min(frame.width - 1), src_y.min(frame.height - 1)) {
                    output.set_pixel(x, y, pixel);
                }
            }
        }

        Ok(output)
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn name(&self) -> &str {
        "SuperResolutionProcessor"
    }
}

/// Style transfer processor.
pub struct StyleTransferProcessor {
    config: ProcessorConfig,
    model_info: ModelInfo,
    style_strength: f32,
}

impl StyleTransferProcessor {
    /// Create a new style transfer processor.
    pub fn new(config: ProcessorConfig, registry: &ModelRegistry) -> Result<Self> {
        let model_info = registry
            .get(&config.model_id)
            .ok_or_else(|| GenAiError::ModelNotFound(config.model_id.clone()))?
            .clone();

        if model_info.model_type != ModelType::StyleTransfer {
            return Err(GenAiError::InvalidInput(format!(
                "Model {} is not a style transfer model",
                config.model_id
            )));
        }

        let style_strength = config
            .params
            .get("style_strength")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        Ok(Self {
            config,
            model_info,
            style_strength,
        })
    }

    /// Set style strength (0.0 - 1.0).
    pub fn set_style_strength(&mut self, strength: f32) {
        self.style_strength = strength.clamp(0.0, 1.0);
    }
}

impl FrameProcessor for StyleTransferProcessor {
    fn process(&self, frame: &FrameData) -> Result<FrameData> {
        tracing::debug!(
            model = %self.config.model_id,
            style_strength = %self.style_strength,
            "Processing style transfer"
        );

        // Simulated style transfer - just copies frame for now
        let mut output = frame.clone();

        // Apply simple color shift as placeholder effect
        for pixel in output.data.chunks_mut(frame.channels as usize) {
            if pixel.len() >= 3 {
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;

                // Simple artistic effect placeholder
                // Clamp values before casting to u8 to avoid overflow
                pixel[0] = (r * (1.0 - self.style_strength * 0.2)).clamp(0.0, 255.0) as u8;
                pixel[1] = (g * (1.0 + self.style_strength * 0.1)).clamp(0.0, 255.0) as u8;
                pixel[2] = (b * (1.0 + self.style_strength * 0.15)).clamp(0.0, 255.0) as u8;
            }
        }

        Ok(output)
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn name(&self) -> &str {
        "StyleTransferProcessor"
    }
}

/// Frame interpolation processor.
pub struct FrameInterpolationProcessor {
    config: ProcessorConfig,
    model_info: ModelInfo,
}

impl FrameInterpolationProcessor {
    /// Create a new frame interpolation processor.
    pub fn new(config: ProcessorConfig, registry: &ModelRegistry) -> Result<Self> {
        let model_info = registry
            .get(&config.model_id)
            .ok_or_else(|| GenAiError::ModelNotFound(config.model_id.clone()))?
            .clone();

        if model_info.model_type != ModelType::FrameInterpolation {
            return Err(GenAiError::InvalidInput(format!(
                "Model {} is not a frame interpolation model",
                config.model_id
            )));
        }

        Ok(Self { config, model_info })
    }

    /// Interpolate between two frames.
    pub fn interpolate(&self, frame1: &FrameData, frame2: &FrameData, t: f32) -> Result<FrameData> {
        if frame1.width != frame2.width || frame1.height != frame2.height {
            return Err(GenAiError::InvalidInput(
                "Frame dimensions must match".into(),
            ));
        }

        let t = t.clamp(0.0, 1.0);

        tracing::debug!(
            model = %self.config.model_id,
            t = %t,
            "Interpolating frames"
        );

        // Linear interpolation as placeholder
        let mut output = FrameData::new(frame1.width, frame1.height, frame1.channels);
        output.timestamp_ms =
            frame1.timestamp_ms + ((frame2.timestamp_ms - frame1.timestamp_ms) as f32 * t) as u64;

        for (i, (p1, p2)) in frame1.data.iter().zip(frame2.data.iter()).enumerate() {
            output.data[i] = ((*p1 as f32) * (1.0 - t) + (*p2 as f32) * t) as u8;
        }

        Ok(output)
    }

    /// Generate N interpolated frames between two frames.
    pub fn interpolate_n(
        &self,
        frame1: &FrameData,
        frame2: &FrameData,
        n: usize,
    ) -> Result<Vec<FrameData>> {
        let mut frames = Vec::with_capacity(n);

        for i in 1..=n {
            let t = i as f32 / (n + 1) as f32;
            frames.push(self.interpolate(frame1, frame2, t)?);
        }

        Ok(frames)
    }
}

impl FrameProcessor for FrameInterpolationProcessor {
    fn process(&self, frame: &FrameData) -> Result<FrameData> {
        // Single frame pass-through
        Ok(frame.clone())
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn name(&self) -> &str {
        "FrameInterpolationProcessor"
    }
}

/// Background removal processor.
pub struct BackgroundRemovalProcessor {
    config: ProcessorConfig,
    model_info: ModelInfo,
    threshold: f32,
}

impl BackgroundRemovalProcessor {
    /// Create a new background removal processor.
    pub fn new(config: ProcessorConfig, registry: &ModelRegistry) -> Result<Self> {
        let model_info = registry
            .get(&config.model_id)
            .ok_or_else(|| GenAiError::ModelNotFound(config.model_id.clone()))?
            .clone();

        if model_info.model_type != ModelType::BackgroundRemoval {
            return Err(GenAiError::InvalidInput(format!(
                "Model {} is not a background removal model",
                config.model_id
            )));
        }

        let threshold = config
            .params
            .get("threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);

        Ok(Self {
            config,
            model_info,
            threshold,
        })
    }

    /// Set mask threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Process and return frame with alpha channel.
    pub fn process_with_alpha(&self, frame: &FrameData) -> Result<FrameData> {
        tracing::debug!(
            model = %self.config.model_id,
            threshold = %self.threshold,
            "Processing background removal"
        );

        // Create RGBA output
        let mut output = FrameData::new(frame.width, frame.height, 4);
        output.timestamp_ms = frame.timestamp_ms;

        // Simulated segmentation - simple luminance-based threshold
        for y in 0..frame.height {
            for x in 0..frame.width {
                if let Some(pixel) = frame.get_pixel(x, y) {
                    let luminance = if pixel.len() >= 3 {
                        (pixel[0] as f32 * 0.299
                            + pixel[1] as f32 * 0.587
                            + pixel[2] as f32 * 0.114)
                            / 255.0
                    } else {
                        pixel[0] as f32 / 255.0
                    };

                    // Placeholder: use luminance as alpha
                    let alpha = if luminance > self.threshold { 255 } else { 0 };

                    let rgba = if pixel.len() >= 3 {
                        [pixel[0], pixel[1], pixel[2], alpha]
                    } else {
                        [pixel[0], pixel[0], pixel[0], alpha]
                    };

                    output.set_pixel(x, y, &rgba);
                }
            }
        }

        Ok(output)
    }
}

impl FrameProcessor for BackgroundRemovalProcessor {
    fn process(&self, frame: &FrameData) -> Result<FrameData> {
        self.process_with_alpha(frame)
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn name(&self) -> &str {
        "BackgroundRemovalProcessor"
    }
}

/// Face enhancement processor.
pub struct FaceEnhancementProcessor {
    config: ProcessorConfig,
    model_info: ModelInfo,
    enhancement_strength: f32,
}

impl FaceEnhancementProcessor {
    /// Create a new face enhancement processor.
    pub fn new(config: ProcessorConfig, registry: &ModelRegistry) -> Result<Self> {
        let model_info = registry
            .get(&config.model_id)
            .ok_or_else(|| GenAiError::ModelNotFound(config.model_id.clone()))?
            .clone();

        if model_info.model_type != ModelType::FaceEnhancement {
            return Err(GenAiError::InvalidInput(format!(
                "Model {} is not a face enhancement model",
                config.model_id
            )));
        }

        let enhancement_strength = config
            .params
            .get("enhancement_strength")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        Ok(Self {
            config,
            model_info,
            enhancement_strength,
        })
    }

    /// Set enhancement strength (0.0 - 1.0).
    pub fn set_enhancement_strength(&mut self, strength: f32) {
        self.enhancement_strength = strength.clamp(0.0, 1.0);
    }

    /// Get enhancement strength.
    pub fn enhancement_strength(&self) -> f32 {
        self.enhancement_strength
    }
}

impl FrameProcessor for FaceEnhancementProcessor {
    fn process(&self, frame: &FrameData) -> Result<FrameData> {
        tracing::debug!(
            model = %self.config.model_id,
            enhancement_strength = %self.enhancement_strength,
            "Processing face enhancement"
        );

        // Simulated face enhancement - applies sharpening and skin smoothing
        let mut output = frame.clone();

        // Simple enhancement placeholder: slight contrast boost and smoothing
        for pixel in output.data.chunks_mut(frame.channels as usize) {
            if pixel.len() >= 3 {
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;

                // Apply gentle contrast enhancement
                let factor = 1.0 + (self.enhancement_strength * 0.1);
                let mid = 128.0;

                pixel[0] = ((r - mid) * factor + mid).clamp(0.0, 255.0) as u8;
                pixel[1] = ((g - mid) * factor + mid).clamp(0.0, 255.0) as u8;
                pixel[2] = ((b - mid) * factor + mid).clamp(0.0, 255.0) as u8;
            }
        }

        Ok(output)
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn name(&self) -> &str {
        "FaceEnhancementProcessor"
    }
}

/// Denoising processor.
pub struct DenoisingProcessor {
    config: ProcessorConfig,
    model_info: ModelInfo,
    denoise_strength: f32,
}

impl DenoisingProcessor {
    /// Create a new denoising processor.
    pub fn new(config: ProcessorConfig, registry: &ModelRegistry) -> Result<Self> {
        let model_info = registry
            .get(&config.model_id)
            .ok_or_else(|| GenAiError::ModelNotFound(config.model_id.clone()))?
            .clone();

        if model_info.model_type != ModelType::Denoising {
            return Err(GenAiError::InvalidInput(format!(
                "Model {} is not a denoising model",
                config.model_id
            )));
        }

        let denoise_strength = config
            .params
            .get("denoise_strength")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);

        Ok(Self {
            config,
            model_info,
            denoise_strength,
        })
    }

    /// Set denoise strength (0.0 - 1.0).
    pub fn set_denoise_strength(&mut self, strength: f32) {
        self.denoise_strength = strength.clamp(0.0, 1.0);
    }

    /// Get denoise strength.
    pub fn denoise_strength(&self) -> f32 {
        self.denoise_strength
    }
}

impl FrameProcessor for DenoisingProcessor {
    fn process(&self, frame: &FrameData) -> Result<FrameData> {
        tracing::debug!(
            model = %self.config.model_id,
            denoise_strength = %self.denoise_strength,
            "Processing denoising"
        );

        // Simulated denoising using simple box blur
        let mut output = FrameData::new(frame.width, frame.height, frame.channels);
        output.timestamp_ms = frame.timestamp_ms;

        let kernel_size = (self.denoise_strength * 2.0 + 1.0) as i32; // 1, 2, or 3
        let half_kernel = kernel_size / 2;

        for y in 0..frame.height {
            for x in 0..frame.width {
                let mut sum = vec![0u32; frame.channels as usize];
                let mut count = 0u32;

                // Sample neighborhood
                for ky in -half_kernel..=half_kernel {
                    for kx in -half_kernel..=half_kernel {
                        let nx = x as i32 + kx;
                        let ny = y as i32 + ky;

                        if nx >= 0 && nx < frame.width as i32 && ny >= 0 && ny < frame.height as i32 {
                            if let Some(pixel) = frame.get_pixel(nx as u32, ny as u32) {
                                for (i, &val) in pixel.iter().enumerate() {
                                    sum[i] += val as u32;
                                }
                                count += 1;
                            }
                        }
                    }
                }

                // Average the neighborhood
                if count > 0 {
                    let averaged: Vec<u8> = sum.iter().map(|&s| (s / count) as u8).collect();

                    // Blend between original and denoised based on strength
                    if let Some(original) = frame.get_pixel(x, y) {
                        let blended: Vec<u8> = averaged
                            .iter()
                            .zip(original.iter())
                            .map(|(&avg, &orig)| {
                                let avg_f = avg as f32;
                                let orig_f = orig as f32;
                                (avg_f * self.denoise_strength + orig_f * (1.0 - self.denoise_strength)) as u8
                            })
                            .collect();
                        output.set_pixel(x, y, &blended);
                    }
                }
            }
        }

        Ok(output)
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn name(&self) -> &str {
        "DenoisingProcessor"
    }
}

/// Color correction processor.
pub struct ColorCorrectionProcessor {
    config: ProcessorConfig,
    model_info: ModelInfo,
    brightness: f32,
    contrast: f32,
    saturation: f32,
}

impl ColorCorrectionProcessor {
    /// Create a new color correction processor.
    pub fn new(config: ProcessorConfig, registry: &ModelRegistry) -> Result<Self> {
        let model_info = registry
            .get(&config.model_id)
            .ok_or_else(|| GenAiError::ModelNotFound(config.model_id.clone()))?
            .clone();

        if model_info.model_type != ModelType::ColorCorrection {
            return Err(GenAiError::InvalidInput(format!(
                "Model {} is not a color correction model",
                config.model_id
            )));
        }

        let brightness = config
            .params
            .get("brightness")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        let contrast = config
            .params
            .get("contrast")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        let saturation = config
            .params
            .get("saturation")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        Ok(Self {
            config,
            model_info,
            brightness,
            contrast,
            saturation,
        })
    }

    /// Set brightness adjustment (-1.0 to 1.0).
    pub fn set_brightness(&mut self, brightness: f32) {
        self.brightness = brightness.clamp(-1.0, 1.0);
    }

    /// Set contrast multiplier (0.0 to 3.0).
    pub fn set_contrast(&mut self, contrast: f32) {
        self.contrast = contrast.clamp(0.0, 3.0);
    }

    /// Set saturation multiplier (0.0 to 3.0).
    pub fn set_saturation(&mut self, saturation: f32) {
        self.saturation = saturation.clamp(0.0, 3.0);
    }

    /// Get brightness.
    pub fn brightness(&self) -> f32 {
        self.brightness
    }

    /// Get contrast.
    pub fn contrast(&self) -> f32 {
        self.contrast
    }

    /// Get saturation.
    pub fn saturation(&self) -> f32 {
        self.saturation
    }
}

impl FrameProcessor for ColorCorrectionProcessor {
    fn process(&self, frame: &FrameData) -> Result<FrameData> {
        tracing::debug!(
            model = %self.config.model_id,
            brightness = %self.brightness,
            contrast = %self.contrast,
            saturation = %self.saturation,
            "Processing color correction"
        );

        let mut output = frame.clone();

        for pixel in output.data.chunks_mut(frame.channels as usize) {
            if pixel.len() >= 3 {
                let mut r = pixel[0] as f32;
                let mut g = pixel[1] as f32;
                let mut b = pixel[2] as f32;

                // Apply brightness
                r += self.brightness * 255.0;
                g += self.brightness * 255.0;
                b += self.brightness * 255.0;

                // Apply contrast
                let mid = 127.5;
                r = (r - mid) * self.contrast + mid;
                g = (g - mid) * self.contrast + mid;
                b = (b - mid) * self.contrast + mid;

                // Apply saturation (convert to grayscale and blend)
                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                r = gray + (r - gray) * self.saturation;
                g = gray + (g - gray) * self.saturation;
                b = gray + (b - gray) * self.saturation;

                pixel[0] = r.clamp(0.0, 255.0) as u8;
                pixel[1] = g.clamp(0.0, 255.0) as u8;
                pixel[2] = b.clamp(0.0, 255.0) as u8;
            }
        }

        Ok(output)
    }

    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn name(&self) -> &str {
        "ColorCorrectionProcessor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_data() {
        let mut frame = FrameData::new(100, 100, 3);
        assert_eq!(frame.data.len(), 100 * 100 * 3);

        frame.set_pixel(50, 50, &[255, 0, 0]);
        let pixel = frame.get_pixel(50, 50).unwrap();
        assert_eq!(pixel, &[255, 0, 0]);
    }

    #[test]
    fn test_float_tensor_conversion() {
        let data = vec![0, 128, 255];
        let frame = FrameData::from_raw(1, 1, 3, data).unwrap();

        let tensor = frame.to_float_tensor();
        assert!((tensor[0] - 0.0).abs() < 0.01);
        assert!((tensor[1] - 0.502).abs() < 0.01);
        assert!((tensor[2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_super_resolution_processor() {
        let registry = ModelRegistry::new();
        let config = ProcessorConfig::for_model("esrgan-x4");

        let processor = SuperResolutionProcessor::new(config, &registry).unwrap();
        assert_eq!(processor.scale_factor(), 4);

        let frame = FrameData::new(100, 100, 3);
        let output = processor.process(&frame).unwrap();
        assert_eq!(output.width, 400);
        assert_eq!(output.height, 400);
    }

    #[test]
    fn test_style_transfer_processor() {
        let registry = ModelRegistry::new();
        let config = ProcessorConfig::for_model("fast-style-transfer")
            .param("style_strength", "0.8");

        let processor = StyleTransferProcessor::new(config, &registry).unwrap();

        let frame = FrameData::new(100, 100, 3);
        let output = processor.process(&frame).unwrap();
        assert_eq!(output.width, frame.width);
        assert_eq!(output.height, frame.height);
    }

    #[test]
    fn test_frame_interpolation() {
        let registry = ModelRegistry::new();
        let config = ProcessorConfig::for_model("rife-v4");

        let processor = FrameInterpolationProcessor::new(config, &registry).unwrap();

        let frame1 = FrameData::new(100, 100, 3).with_timestamp(0);
        let frame2 = FrameData::new(100, 100, 3).with_timestamp(100);

        let middle = processor.interpolate(&frame1, &frame2, 0.5).unwrap();
        assert_eq!(middle.timestamp_ms, 50);

        let frames = processor.interpolate_n(&frame1, &frame2, 3).unwrap();
        assert_eq!(frames.len(), 3);
    }

    #[test]
    fn test_background_removal() {
        let registry = ModelRegistry::new();
        let config = ProcessorConfig::for_model("rembg-u2net");

        let processor = BackgroundRemovalProcessor::new(config, &registry).unwrap();

        let frame = FrameData::new(100, 100, 3);
        let output = processor.process(&frame).unwrap();
        assert_eq!(output.channels, 4); // RGBA output
    }

    #[test]
    fn test_face_enhancement_processor() {
        let registry = ModelRegistry::new();
        let config = ProcessorConfig::for_model("gfpgan-v1.4")
            .param("enhancement_strength", "0.8");

        let processor = FaceEnhancementProcessor::new(config, &registry).unwrap();
        assert!((processor.enhancement_strength() - 0.8).abs() < 0.01);

        let frame = FrameData::new(100, 100, 3);
        let output = processor.process(&frame).unwrap();
        assert_eq!(output.width, frame.width);
        assert_eq!(output.height, frame.height);
    }

    #[test]
    fn test_denoising_processor() {
        let registry = ModelRegistry::new();
        let config = ProcessorConfig::for_model("denoise-cnn")
            .param("denoise_strength", "0.7");

        let processor = DenoisingProcessor::new(config, &registry).unwrap();
        assert!((processor.denoise_strength() - 0.7).abs() < 0.01);

        let frame = FrameData::new(50, 50, 3);
        let output = processor.process(&frame).unwrap();
        assert_eq!(output.width, frame.width);
        assert_eq!(output.height, frame.height);
    }

    #[test]
    fn test_color_correction_processor() {
        let registry = ModelRegistry::new();
        let config = ProcessorConfig::for_model("color-ai")
            .param("brightness", "0.1")
            .param("contrast", "1.2")
            .param("saturation", "1.1");

        let processor = ColorCorrectionProcessor::new(config, &registry).unwrap();
        assert!((processor.brightness() - 0.1).abs() < 0.01);
        assert!((processor.contrast() - 1.2).abs() < 0.01);
        assert!((processor.saturation() - 1.1).abs() < 0.01);

        let frame = FrameData::new(100, 100, 3);
        let output = processor.process(&frame).unwrap();
        assert_eq!(output.width, frame.width);
        assert_eq!(output.height, frame.height);
    }
}
