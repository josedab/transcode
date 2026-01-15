//! Frame interpolation for increasing frame rate.

use crate::error::{AiError, Result};
use crate::model::{InferenceSession, ModelBackend, ModelLoader};
use crate::Frame;
use std::path::PathBuf;
use tracing::debug;

/// Interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationMode {
    /// Simple linear blending (fast, low quality).
    Linear,
    /// Motion-compensated interpolation.
    #[default]
    MotionCompensated,
    /// Optical flow based interpolation.
    OpticalFlow,
    /// AI-based (RIFE, FILM, etc.)
    Neural,
}

/// Frame interpolation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum InterpolationModel {
    /// RIFE - Real-Time Intermediate Flow Estimation.
    #[default]
    Rife,
    /// FILM - Frame Interpolation for Large Motion.
    Film,
    /// IFRNet - Intermediate Feature Refine Network.
    IfrNet,
    /// CAIN - Channel Attention Is All You Need.
    Cain,
    /// Linear blend (non-AI fallback).
    LinearBlend,
}

impl InterpolationModel {
    /// Get model name for file lookup.
    pub fn model_name(self) -> String {
        match self {
            Self::Rife => "rife".to_string(),
            Self::Film => "film".to_string(),
            Self::IfrNet => "ifrnet".to_string(),
            Self::Cain => "cain".to_string(),
            Self::LinearBlend => "".to_string(),
        }
    }

    /// Check if this model requires a neural network.
    pub fn requires_nn(self) -> bool {
        !matches!(self, Self::LinearBlend)
    }
}

/// Interpolator configuration.
#[derive(Debug, Clone, Default)]
pub struct InterpolatorConfig {
    /// Interpolation mode.
    pub mode: InterpolationMode,
    /// Model to use.
    pub model: InterpolationModel,
    /// Inference backend.
    pub backend: ModelBackend,
    /// Custom model path.
    pub model_path: Option<PathBuf>,
    /// Target frame rate multiplier (2 = double fps, 4 = quadruple).
    pub multiplier: u32,
    /// Scene change detection threshold (0.0-1.0).
    pub scene_threshold: f32,
}

impl InterpolatorConfig {
    /// Set mode.
    pub fn with_mode(mut self, mode: InterpolationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set model.
    pub fn with_model(mut self, model: InterpolationModel) -> Self {
        self.model = model;
        self
    }

    /// Set backend.
    pub fn with_backend(mut self, backend: ModelBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set frame rate multiplier.
    pub fn with_multiplier(mut self, multiplier: u32) -> Self {
        self.multiplier = multiplier.max(2);
        self
    }

    /// Set scene change threshold.
    pub fn with_scene_threshold(mut self, threshold: f32) -> Self {
        self.scene_threshold = threshold.clamp(0.0, 1.0);
        self
    }
}

/// Frame interpolator.
pub struct FrameInterpolator {
    /// Configuration.
    config: InterpolatorConfig,
    /// Inference session (if using neural network).
    session: Option<InferenceSession>,
}

impl std::fmt::Debug for FrameInterpolator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameInterpolator")
            .field("config", &self.config)
            .field("has_session", &self.session.is_some())
            .finish()
    }
}

impl FrameInterpolator {
    /// Create a new frame interpolator.
    pub fn new(config: InterpolatorConfig) -> Result<Self> {
        let session = if config.model.requires_nn() {
            let loader = ModelLoader::new().with_backend(config.backend);

            if let Some(ref path) = config.model_path {
                Some(InferenceSession::new(path, config.backend)?)
            } else {
                let model_name = config.model.model_name();
                match loader.find_model(&model_name) {
                    Ok(path) => Some(InferenceSession::new(&path, config.backend)?),
                    Err(_) => {
                        debug!(
                            "Model '{}' not found, using fallback interpolation",
                            model_name
                        );
                        None
                    }
                }
            }
        } else {
            None
        };

        Ok(Self { config, session })
    }

    /// Interpolate frames between two input frames.
    pub fn interpolate(&self, frame1: &Frame, frame2: &Frame, _factor: f32) -> Result<Vec<Frame>> {
        frame1.validate()?;
        frame2.validate()?;

        // Check dimensions match
        if frame1.width != frame2.width || frame1.height != frame2.height {
            return Err(AiError::DimensionMismatch {
                expected: format!("{}x{}", frame1.width, frame1.height),
                actual: format!("{}x{}", frame2.width, frame2.height),
            });
        }

        // Check for scene change
        if self.detect_scene_change(frame1, frame2) {
            debug!("Scene change detected, skipping interpolation");
            return Ok(vec![frame1.clone()]);
        }

        let num_intermediate = (self.config.multiplier - 1) as usize;
        let mut results = Vec::with_capacity(num_intermediate);

        for i in 1..=num_intermediate {
            let t = i as f32 / self.config.multiplier as f32;
            let interpolated = self.interpolate_single(frame1, frame2, t)?;
            results.push(interpolated);
        }

        Ok(results)
    }

    /// Interpolate a single frame at time t (0.0-1.0).
    pub fn interpolate_single(&self, frame1: &Frame, frame2: &Frame, t: f32) -> Result<Frame> {
        if let Some(ref _session) = self.session {
            // In real implementation, would use neural network
            self.interpolate_motion_compensated(frame1, frame2, t)
        } else {
            match self.config.mode {
                InterpolationMode::Linear => self.interpolate_linear(frame1, frame2, t),
                InterpolationMode::MotionCompensated | InterpolationMode::OpticalFlow => {
                    self.interpolate_motion_compensated(frame1, frame2, t)
                }
                InterpolationMode::Neural => {
                    // Fall back to motion compensated if no neural network
                    self.interpolate_motion_compensated(frame1, frame2, t)
                }
            }
        }
    }

    /// Simple linear interpolation (blending).
    fn interpolate_linear(&self, frame1: &Frame, frame2: &Frame, t: f32) -> Result<Frame> {
        let t_inv = 1.0 - t;

        let data: Vec<u8> = frame1
            .data
            .iter()
            .zip(frame2.data.iter())
            .map(|(&a, &b)| {
                let result = a as f32 * t_inv + b as f32 * t;
                result.clamp(0.0, 255.0) as u8
            })
            .collect();

        // Interpolate PTS
        let pts = (frame1.pts as f64 * (1.0 - t as f64) + frame2.pts as f64 * t as f64) as i64;

        Ok(Frame::new(data, frame1.width, frame1.height, frame1.channels).with_pts(pts))
    }

    /// Motion-compensated interpolation.
    fn interpolate_motion_compensated(
        &self,
        frame1: &Frame,
        frame2: &Frame,
        t: f32,
    ) -> Result<Frame> {
        let width = frame1.width as usize;
        let height = frame1.height as usize;
        let channels = frame1.channels as usize;

        // Block matching parameters
        let block_size = 8;
        let search_range = 16;

        // Estimate motion vectors using block matching
        let motion_vectors = self.estimate_motion(frame1, frame2, block_size, search_range);

        // Interpolate using motion vectors
        let mut output = vec![0u8; frame1.data.len()];

        for by in 0..height / block_size {
            for bx in 0..width / block_size {
                let mv = motion_vectors[by * (width / block_size) + bx];

                // Motion vector at time t
                let mvx_t = mv.0 * t;
                let mvy_t = mv.1 * t;

                // Fill block
                for y in 0..block_size {
                    for x in 0..block_size {
                        let px = bx * block_size + x;
                        let py = by * block_size + y;

                        if px >= width || py >= height {
                            continue;
                        }

                        // Sample from frame1 with forward motion
                        let src1_x = (px as f32 + mvx_t).clamp(0.0, (width - 1) as f32);
                        let src1_y = (py as f32 + mvy_t).clamp(0.0, (height - 1) as f32);

                        // Sample from frame2 with backward motion
                        let src2_x = (px as f32 - (mv.0 - mvx_t)).clamp(0.0, (width - 1) as f32);
                        let src2_y = (py as f32 - (mv.1 - mvy_t)).clamp(0.0, (height - 1) as f32);

                        for c in 0..channels {
                            let val1 = bilinear_sample(frame1, src1_x, src1_y, c);
                            let val2 = bilinear_sample(frame2, src2_x, src2_y, c);

                            let blended = val1 * (1.0 - t) + val2 * t;
                            output[(py * width + px) * channels + c] = blended.clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }

        let pts = (frame1.pts as f64 * (1.0 - t as f64) + frame2.pts as f64 * t as f64) as i64;

        Ok(Frame::new(output, frame1.width, frame1.height, frame1.channels).with_pts(pts))
    }

    /// Estimate motion vectors using block matching.
    fn estimate_motion(
        &self,
        frame1: &Frame,
        frame2: &Frame,
        block_size: usize,
        search_range: usize,
    ) -> Vec<(f32, f32)> {
        let width = frame1.width as usize;
        let height = frame1.height as usize;
        let channels = frame1.channels as usize;

        let blocks_x = width / block_size;
        let blocks_y = height / block_size;

        let mut motion_vectors = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let base_x = bx * block_size;
                let base_y = by * block_size;

                let mut best_mv = (0.0f32, 0.0f32);
                let mut best_sad = f32::MAX;

                // Search for best matching block
                for dy in -(search_range as i32)..=(search_range as i32) {
                    for dx in -(search_range as i32)..=(search_range as i32) {
                        let search_x = (base_x as i32 + dx).clamp(0, width as i32 - block_size as i32) as usize;
                        let search_y = (base_y as i32 + dy).clamp(0, height as i32 - block_size as i32) as usize;

                        // Calculate SAD (Sum of Absolute Differences)
                        let mut sad = 0.0f32;
                        for y in 0..block_size {
                            for x in 0..block_size {
                                for c in 0..channels {
                                    let idx1 = ((base_y + y) * width + base_x + x) * channels + c;
                                    let idx2 = ((search_y + y) * width + search_x + x) * channels + c;

                                    let diff = frame1.data[idx1] as f32 - frame2.data[idx2] as f32;
                                    sad += diff.abs();
                                }
                            }
                        }

                        if sad < best_sad {
                            best_sad = sad;
                            best_mv = (dx as f32, dy as f32);
                        }
                    }
                }

                motion_vectors.push(best_mv);
            }
        }

        motion_vectors
    }

    /// Detect scene change between frames.
    fn detect_scene_change(&self, frame1: &Frame, frame2: &Frame) -> bool {
        if self.config.scene_threshold <= 0.0 {
            return false;
        }

        // Calculate histogram difference
        let mut hist1 = [0u32; 256];
        let mut hist2 = [0u32; 256];

        for val in &frame1.data {
            hist1[*val as usize] += 1;
        }
        for val in &frame2.data {
            hist2[*val as usize] += 1;
        }

        // Normalize and calculate difference
        let total = frame1.data.len() as f32;
        let mut diff = 0.0f32;

        for i in 0..256 {
            let h1 = hist1[i] as f32 / total;
            let h2 = hist2[i] as f32 / total;
            diff += (h1 - h2).abs();
        }

        diff > self.config.scene_threshold
    }

    /// Get configuration.
    pub fn config(&self) -> &InterpolatorConfig {
        &self.config
    }

    /// Check if using neural network.
    pub fn is_using_nn(&self) -> bool {
        self.session.is_some()
    }
}

/// Bilinear sample from frame.
fn bilinear_sample(frame: &Frame, x: f32, y: f32, channel: usize) -> f32 {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let channels = frame.channels as usize;

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let v00 = frame.data[(y0 * width + x0) * channels + channel] as f32;
    let v01 = frame.data[(y0 * width + x1) * channels + channel] as f32;
    let v10 = frame.data[(y1 * width + x0) * channels + channel] as f32;
    let v11 = frame.data[(y1 * width + x1) * channels + channel] as f32;

    (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01) + fy * ((1.0 - fx) * v10 + fx * v11)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, value: u8) -> Frame {
        let channels = 3u8;
        let data = vec![value; (width * height * channels as u32) as usize];
        Frame::new(data, width, height, channels)
    }

    fn create_gradient_frame(width: u32, height: u32) -> Frame {
        let channels = 3u8;
        let size = (width * height * channels as u32) as usize;
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            data.push((i % 256) as u8);
        }
        Frame::new(data, width, height, channels)
    }

    #[test]
    fn test_interpolation_model_names() {
        assert_eq!(InterpolationModel::Rife.model_name(), "rife");
        assert_eq!(InterpolationModel::Film.model_name(), "film");
        assert_eq!(InterpolationModel::LinearBlend.model_name(), "");
    }

    #[test]
    fn test_interpolation_model_requires_nn() {
        assert!(InterpolationModel::Rife.requires_nn());
        assert!(InterpolationModel::Film.requires_nn());
        assert!(!InterpolationModel::LinearBlend.requires_nn());
    }

    #[test]
    fn test_interpolator_config_builder() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Linear)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(4)
            .with_scene_threshold(0.3);

        assert_eq!(config.mode, InterpolationMode::Linear);
        assert_eq!(config.model, InterpolationModel::LinearBlend);
        assert_eq!(config.multiplier, 4);
        assert!((config.scene_threshold - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_interpolator_config_multiplier_min() {
        let config = InterpolatorConfig::default()
            .with_multiplier(1); // Should be clamped to 2
        assert_eq!(config.multiplier, 2);
    }

    #[test]
    fn test_interpolator_config_scene_threshold_clamped() {
        let config = InterpolatorConfig::default()
            .with_scene_threshold(1.5);
        assert!((config.scene_threshold - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_interpolate_frame_count_2x() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Linear)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(2)
            .with_scene_threshold(0.0); // Disable scene detection

        let interpolator = FrameInterpolator::new(config).unwrap();
        let frame1 = create_test_frame(32, 32, 0);
        let frame2 = create_test_frame(32, 32, 255);

        let results = interpolator.interpolate(&frame1, &frame2, 0.5).unwrap();
        // 2x multiplier means 1 intermediate frame
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_interpolate_frame_count_4x() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Linear)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(4)
            .with_scene_threshold(0.0);

        let interpolator = FrameInterpolator::new(config).unwrap();
        let frame1 = create_test_frame(32, 32, 0);
        let frame2 = create_test_frame(32, 32, 255);

        let results = interpolator.interpolate(&frame1, &frame2, 0.5).unwrap();
        // 4x multiplier means 3 intermediate frames
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_interpolate_preserves_dimensions() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Linear)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(2)
            .with_scene_threshold(0.0);

        let interpolator = FrameInterpolator::new(config).unwrap();
        let frame1 = create_test_frame(64, 48, 100);
        let frame2 = create_test_frame(64, 48, 200);

        let results = interpolator.interpolate(&frame1, &frame2, 0.5).unwrap();
        assert_eq!(results[0].width, 64);
        assert_eq!(results[0].height, 48);
        assert_eq!(results[0].channels, 3);
    }

    #[test]
    fn test_interpolate_linear_midpoint() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Linear)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(2)
            .with_scene_threshold(0.0);

        let interpolator = FrameInterpolator::new(config).unwrap();
        let frame1 = create_test_frame(8, 8, 0);
        let frame2 = create_test_frame(8, 8, 200);

        let results = interpolator.interpolate(&frame1, &frame2, 0.5).unwrap();
        // At t=0.5, linear blend should give ~100
        let mid_value = results[0].data[0];
        assert!(mid_value >= 95 && mid_value <= 105, "Expected ~100, got {}", mid_value);
    }

    #[test]
    fn test_interpolate_pts_interpolation() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Linear)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(2)
            .with_scene_threshold(0.0);

        let interpolator = FrameInterpolator::new(config).unwrap();
        let frame1 = create_test_frame(16, 16, 0).with_pts(0);
        let frame2 = create_test_frame(16, 16, 255).with_pts(1000);

        let results = interpolator.interpolate(&frame1, &frame2, 0.5).unwrap();
        // At t=0.5, PTS should be ~500
        assert!(results[0].pts >= 400 && results[0].pts <= 600);
    }

    #[test]
    fn test_interpolate_dimension_mismatch() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Linear)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(2);

        let interpolator = FrameInterpolator::new(config).unwrap();
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(32, 32, 255); // Different size

        let result = interpolator.interpolate(&frame1, &frame2, 0.5);
        assert!(result.is_err());
        match result {
            Err(AiError::DimensionMismatch { .. }) => (),
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_interpolate_invalid_frame() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Linear)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(2);

        let interpolator = FrameInterpolator::new(config).unwrap();

        let frame1 = Frame {
            data: vec![0u8; 10], // Wrong size
            width: 32,
            height: 32,
            channels: 3,
            pts: 0,
        };
        let frame2 = create_test_frame(32, 32, 255);

        let result = interpolator.interpolate(&frame1, &frame2, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_scene_change_detection() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Linear)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(2)
            .with_scene_threshold(0.1); // Low threshold to trigger detection

        let interpolator = FrameInterpolator::new(config).unwrap();

        // Very different frames (scene change)
        let frame1 = create_test_frame(32, 32, 0);
        let frame2 = create_test_frame(32, 32, 255);

        let results = interpolator.interpolate(&frame1, &frame2, 0.5).unwrap();
        // Scene change should return just the first frame
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data[0], 0); // Should be frame1
    }

    #[test]
    fn test_motion_compensated_interpolation() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::MotionCompensated)
            .with_model(InterpolationModel::LinearBlend)
            .with_multiplier(2)
            .with_scene_threshold(0.0);

        let interpolator = FrameInterpolator::new(config).unwrap();
        let frame1 = create_gradient_frame(32, 32);
        let frame2 = create_gradient_frame(32, 32);

        let results = interpolator.interpolate(&frame1, &frame2, 0.5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].width, 32);
        assert_eq!(results[0].height, 32);
    }

    #[test]
    fn test_bilinear_sample_center() {
        let frame = create_test_frame(4, 4, 100);
        let value = bilinear_sample(&frame, 1.5, 1.5, 0);
        assert!((value - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_interpolator_fallback_when_no_model() {
        let config = InterpolatorConfig::default()
            .with_mode(InterpolationMode::Neural)
            .with_model(InterpolationModel::Rife); // Requires NN

        let interpolator = FrameInterpolator::new(config).unwrap();
        // Should fall back when model not found
        assert!(!interpolator.is_using_nn());
    }
}
