//! Frame blending for frame rate conversion.
//!
//! This module provides frame blending algorithms that create smooth transitions
//! between frames by computing weighted averages of pixel values.

use crate::error::{FrameRateError, Result};
use transcode_core::{Frame, FrameBuffer};

/// Blending mode for combining frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Linear interpolation between frames.
    #[default]
    Linear,
    /// Quadratic ease-in blending.
    EaseIn,
    /// Quadratic ease-out blending.
    EaseOut,
    /// Smooth step (Hermite) interpolation.
    SmoothStep,
    /// Cosine interpolation for smoother motion.
    Cosine,
}

/// Frame blender configuration.
#[derive(Debug, Clone)]
pub struct BlendConfig {
    /// Blending mode to use.
    pub mode: BlendMode,
    /// Enable dithering to reduce banding artifacts.
    pub dithering: bool,
    /// Strength of the blend (0.0 = first frame, 1.0 = second frame).
    pub default_weight: f32,
}

impl Default for BlendConfig {
    fn default() -> Self {
        Self {
            mode: BlendMode::Linear,
            dithering: false,
            default_weight: 0.5,
        }
    }
}

/// Frame blender for creating smooth transitions.
pub struct FrameBlender {
    config: BlendConfig,
}

impl FrameBlender {
    /// Create a new frame blender with default configuration.
    pub fn new() -> Self {
        Self {
            config: BlendConfig::default(),
        }
    }

    /// Create a frame blender with custom configuration.
    pub fn with_config(config: BlendConfig) -> Self {
        Self { config }
    }

    /// Set the blend mode.
    pub fn set_mode(&mut self, mode: BlendMode) {
        self.config.mode = mode;
    }

    /// Enable or disable dithering.
    pub fn set_dithering(&mut self, enabled: bool) {
        self.config.dithering = enabled;
    }

    /// Blend two frames with the specified weight.
    ///
    /// The weight parameter controls the blend ratio:
    /// - 0.0 = 100% first frame
    /// - 0.5 = 50% each frame
    /// - 1.0 = 100% second frame
    pub fn blend(&self, frame1: &Frame, frame2: &Frame, weight: f32) -> Result<Frame> {
        self.validate_frames(frame1, frame2)?;

        let adjusted_weight = self.adjust_weight(weight);
        let mut output = Frame::from_buffer(FrameBuffer::new(
            frame1.width(),
            frame1.height(),
            frame1.format(),
        ));

        // Blend each plane
        for plane_idx in 0..frame1.format().num_planes() {
            self.blend_plane(
                frame1.plane(plane_idx).unwrap(),
                frame2.plane(plane_idx).unwrap(),
                output.plane_mut(plane_idx).unwrap(),
                adjusted_weight,
            );
        }

        // Interpolate timestamps
        output.pts = frame1.pts;
        output.dts = frame1.dts;
        output.duration = frame1.duration;
        output.flags = frame1.flags;

        Ok(output)
    }

    /// Blend multiple frames with specified weights.
    ///
    /// The weights should sum to 1.0 for proper blending.
    pub fn blend_multiple(&self, frames: &[&Frame], weights: &[f32]) -> Result<Frame> {
        if frames.is_empty() {
            return Err(FrameRateError::InsufficientFrames {
                needed: 1,
                available: 0,
            });
        }

        if frames.len() != weights.len() {
            return Err(FrameRateError::invalid_params(format!(
                "Frame count ({}) doesn't match weight count ({})",
                frames.len(),
                weights.len()
            )));
        }

        // Validate all frames have same dimensions
        let first = frames[0];
        for frame in frames.iter().skip(1) {
            self.validate_frames(first, frame)?;
        }

        let mut output = Frame::from_buffer(FrameBuffer::new(
            first.width(),
            first.height(),
            first.format(),
        ));

        // Blend each plane
        for plane_idx in 0..first.format().num_planes() {
            let planes: Vec<&[u8]> = frames
                .iter()
                .map(|f| f.plane(plane_idx).unwrap())
                .collect();
            self.blend_plane_multiple(&planes, weights, output.plane_mut(plane_idx).unwrap());
        }

        output.pts = first.pts;
        output.dts = first.dts;
        output.duration = first.duration;
        output.flags = first.flags;

        Ok(output)
    }

    /// Validate that two frames are compatible for blending.
    fn validate_frames(&self, frame1: &Frame, frame2: &Frame) -> Result<()> {
        if frame1.width() != frame2.width() || frame1.height() != frame2.height() {
            return Err(FrameRateError::DimensionMismatch {
                expected_width: frame1.width(),
                expected_height: frame1.height(),
                actual_width: frame2.width(),
                actual_height: frame2.height(),
            });
        }

        if frame1.format() != frame2.format() {
            return Err(FrameRateError::unsupported_format(format!(
                "Cannot blend {} with {}",
                frame1.format(),
                frame2.format()
            )));
        }

        Ok(())
    }

    /// Adjust weight based on blend mode.
    fn adjust_weight(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self.config.mode {
            BlendMode::Linear => t,
            BlendMode::EaseIn => t * t,
            BlendMode::EaseOut => t * (2.0 - t),
            BlendMode::SmoothStep => t * t * (3.0 - 2.0 * t),
            BlendMode::Cosine => (1.0 - (t * std::f32::consts::PI).cos()) / 2.0,
        }
    }

    /// Blend a single plane.
    fn blend_plane(&self, plane1: &[u8], plane2: &[u8], output: &mut [u8], weight: f32) {
        let w1 = ((1.0 - weight) * 256.0) as u32;
        let w2 = (weight * 256.0) as u32;

        for (i, out) in output.iter_mut().enumerate() {
            if i < plane1.len() && i < plane2.len() {
                let v1 = plane1[i] as u32;
                let v2 = plane2[i] as u32;
                let blended = (v1 * w1 + v2 * w2 + 128) >> 8;

                *out = if self.config.dithering {
                    // Simple ordered dithering
                    let dither = ((i ^ (i >> 4)) & 1) as u32;
                    (blended + dither).min(255) as u8
                } else {
                    blended.min(255) as u8
                };
            }
        }
    }

    /// Blend multiple planes with weights.
    fn blend_plane_multiple(&self, planes: &[&[u8]], weights: &[f32], output: &mut [u8]) {
        let scaled_weights: Vec<u32> = weights.iter().map(|&w| (w * 256.0) as u32).collect();

        for (i, out) in output.iter_mut().enumerate() {
            let mut sum = 0u32;
            for (plane, &weight) in planes.iter().zip(scaled_weights.iter()) {
                if i < plane.len() {
                    sum += plane[i] as u32 * weight;
                }
            }
            *out = ((sum + 128) >> 8).min(255) as u8;
        }
    }
}

impl Default for FrameBlender {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a ghosting effect by blending current frame with previous frames.
pub struct GhostBlender {
    /// Number of ghost frames to keep.
    ghost_count: usize,
    /// Decay factor for ghost intensity (0.0-1.0).
    decay: f32,
    /// Previous frames buffer.
    history: Vec<Frame>,
}

impl GhostBlender {
    /// Create a new ghost blender.
    pub fn new(ghost_count: usize, decay: f32) -> Self {
        Self {
            ghost_count,
            decay: decay.clamp(0.0, 1.0),
            history: Vec::with_capacity(ghost_count),
        }
    }

    /// Process a frame, adding ghost effect from previous frames.
    pub fn process(&mut self, frame: &Frame) -> Result<Frame> {
        if self.history.is_empty() {
            self.history.push(frame.clone());
            return Ok(frame.clone());
        }

        // Calculate weights with exponential decay
        let mut weights = Vec::with_capacity(self.history.len() + 1);
        let mut total_weight = 1.0f32;

        // Current frame weight
        weights.push(1.0 - self.decay * self.history.len() as f32 / (self.ghost_count as f32 + 1.0));

        // Ghost frame weights
        for i in 0..self.history.len() {
            let ghost_weight =
                self.decay * (self.history.len() - i) as f32 / (self.ghost_count as f32 + 1.0)
                    / self.history.len() as f32;
            weights.push(ghost_weight);
            total_weight += ghost_weight;
        }

        // Normalize weights
        for w in &mut weights {
            *w /= total_weight;
        }

        // Build frame references
        let mut frames: Vec<&Frame> = vec![frame];
        for hist_frame in self.history.iter().rev() {
            frames.push(hist_frame);
        }

        // Blend frames
        let blender = FrameBlender::new();
        let result = blender.blend_multiple(&frames, &weights)?;

        // Update history
        self.history.push(frame.clone());
        if self.history.len() > self.ghost_count {
            self.history.remove(0);
        }

        Ok(result)
    }

    /// Reset the ghost blender, clearing history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::{timestamp::TimeBase, PixelFormat};

    fn create_test_frame(width: u32, height: u32, value: u8) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);
        // Fill Y plane with value
        if let Some(plane) = frame.plane_mut(0) {
            plane.fill(value);
        }
        frame
    }

    #[test]
    fn test_blend_equal_weight() {
        let frame1 = create_test_frame(16, 16, 100);
        let frame2 = create_test_frame(16, 16, 200);

        let blender = FrameBlender::new();
        let result = blender.blend(&frame1, &frame2, 0.5).unwrap();

        // Check Y plane values are approximately 150 (average of 100 and 200)
        let y_plane = result.plane(0).unwrap();
        for &v in y_plane.iter().take(16 * 16) {
            assert!((v as i32 - 150).abs() <= 1);
        }
    }

    #[test]
    fn test_blend_zero_weight() {
        let frame1 = create_test_frame(16, 16, 100);
        let frame2 = create_test_frame(16, 16, 200);

        let blender = FrameBlender::new();
        let result = blender.blend(&frame1, &frame2, 0.0).unwrap();

        let y_plane = result.plane(0).unwrap();
        for &v in y_plane.iter().take(16 * 16) {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_blend_full_weight() {
        let frame1 = create_test_frame(16, 16, 100);
        let frame2 = create_test_frame(16, 16, 200);

        let blender = FrameBlender::new();
        let result = blender.blend(&frame1, &frame2, 1.0).unwrap();

        let y_plane = result.plane(0).unwrap();
        for &v in y_plane.iter().take(16 * 16) {
            assert_eq!(v, 200);
        }
    }

    #[test]
    fn test_blend_dimension_mismatch() {
        let frame1 = create_test_frame(16, 16, 100);
        let frame2 = create_test_frame(32, 32, 200);

        let blender = FrameBlender::new();
        let result = blender.blend(&frame1, &frame2, 0.5);

        assert!(matches!(result, Err(FrameRateError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_blend_modes() {
        let frame1 = create_test_frame(16, 16, 0);
        let frame2 = create_test_frame(16, 16, 255);

        for mode in [
            BlendMode::Linear,
            BlendMode::EaseIn,
            BlendMode::EaseOut,
            BlendMode::SmoothStep,
            BlendMode::Cosine,
        ] {
            let blender = FrameBlender::with_config(BlendConfig {
                mode,
                ..Default::default()
            });
            let result = blender.blend(&frame1, &frame2, 0.5);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_blend_multiple() {
        let frame1 = create_test_frame(16, 16, 0);
        let frame2 = create_test_frame(16, 16, 100);
        let frame3 = create_test_frame(16, 16, 200);

        let blender = FrameBlender::new();
        let frames: Vec<&Frame> = vec![&frame1, &frame2, &frame3];
        let weights = vec![0.25, 0.5, 0.25];

        let result = blender.blend_multiple(&frames, &weights).unwrap();
        let y_plane = result.plane(0).unwrap();

        // Expected: 0*0.25 + 100*0.5 + 200*0.25 = 0 + 50 + 50 = 100
        for &v in y_plane.iter().take(16 * 16) {
            assert!((v as i32 - 100).abs() <= 1);
        }
    }

    #[test]
    fn test_ghost_blender() {
        let mut ghost = GhostBlender::new(3, 0.3);

        let frame1 = create_test_frame(16, 16, 100);
        let result1 = ghost.process(&frame1).unwrap();
        assert!(result1.plane(0).is_some());

        let frame2 = create_test_frame(16, 16, 150);
        let result2 = ghost.process(&frame2).unwrap();
        assert!(result2.plane(0).is_some());

        ghost.reset();
        assert!(ghost.history.is_empty());
    }
}
