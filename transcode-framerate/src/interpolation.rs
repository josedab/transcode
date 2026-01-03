//! Frame interpolation methods for frame rate conversion.
//!
//! This module provides various interpolation algorithms for generating
//! intermediate frames when converting between different frame rates.

use crate::blend::{BlendMode, FrameBlender};
use crate::error::{FrameRateError, Result};
use crate::motion::{MotionEstimationConfig, MotionEstimator, MotionField};
use transcode_core::{Frame, FrameBuffer};

/// Interpolation method for frame rate conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationMethod {
    /// Nearest neighbor (frame duplication).
    Nearest,
    /// Linear blending between adjacent frames.
    #[default]
    Linear,
    /// Cubic interpolation using four frames.
    Cubic,
    /// Motion-compensated interpolation.
    MotionCompensated,
    /// Optical flow based interpolation.
    OpticalFlow,
}

/// Configuration for frame interpolation.
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Interpolation method to use.
    pub method: InterpolationMethod,
    /// Blend mode for linear interpolation.
    pub blend_mode: BlendMode,
    /// Motion estimation configuration for motion-compensated methods.
    pub motion_config: MotionEstimationConfig,
    /// Enable artifact reduction post-processing.
    pub artifact_reduction: bool,
    /// Occlusion handling mode.
    pub occlusion_handling: OcclusionHandling,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            method: InterpolationMethod::Linear,
            blend_mode: BlendMode::Linear,
            motion_config: MotionEstimationConfig::default(),
            artifact_reduction: true,
            occlusion_handling: OcclusionHandling::Blend,
        }
    }
}

/// How to handle occluded regions in motion-compensated interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OcclusionHandling {
    /// Blend both frames equally in occluded regions.
    #[default]
    Blend,
    /// Use forward frame for occluded regions.
    Forward,
    /// Use backward frame for occluded regions.
    Backward,
    /// Inpaint occluded regions.
    Inpaint,
}

/// Frame interpolator for generating intermediate frames.
pub struct FrameInterpolator {
    config: InterpolationConfig,
    motion_estimator: MotionEstimator,
    blender: FrameBlender,
}

impl FrameInterpolator {
    /// Create a new frame interpolator with default configuration.
    pub fn new() -> Self {
        Self {
            config: InterpolationConfig::default(),
            motion_estimator: MotionEstimator::new(),
            blender: FrameBlender::new(),
        }
    }

    /// Create a frame interpolator with custom configuration.
    pub fn with_config(config: InterpolationConfig) -> Self {
        Self {
            motion_estimator: MotionEstimator::with_config(config.motion_config.clone()),
            blender: FrameBlender::new(),
            config,
        }
    }

    /// Set the interpolation method.
    pub fn set_method(&mut self, method: InterpolationMethod) {
        self.config.method = method;
    }

    /// Interpolate a frame at the given position between two frames.
    ///
    /// The position parameter (t) should be in the range [0.0, 1.0]:
    /// - t = 0.0 returns a copy of frame1
    /// - t = 0.5 returns the midpoint between frames
    /// - t = 1.0 returns a copy of frame2
    pub fn interpolate(&self, frame1: &Frame, frame2: &Frame, t: f32) -> Result<Frame> {
        self.validate_frames(frame1, frame2)?;

        let t = t.clamp(0.0, 1.0);

        match self.config.method {
            InterpolationMethod::Nearest => self.interpolate_nearest(frame1, frame2, t),
            InterpolationMethod::Linear => self.interpolate_linear(frame1, frame2, t),
            InterpolationMethod::Cubic => {
                // For cubic, we only have two frames so fall back to linear
                self.interpolate_linear(frame1, frame2, t)
            }
            InterpolationMethod::MotionCompensated => {
                self.interpolate_motion_compensated(frame1, frame2, t)
            }
            InterpolationMethod::OpticalFlow => {
                // Optical flow is similar to motion compensated
                self.interpolate_motion_compensated(frame1, frame2, t)
            }
        }
    }

    /// Interpolate using four frames for cubic interpolation.
    pub fn interpolate_cubic(
        &self,
        frame0: &Frame,
        frame1: &Frame,
        frame2: &Frame,
        frame3: &Frame,
        t: f32,
    ) -> Result<Frame> {
        self.validate_frames(frame1, frame2)?;
        self.validate_frames(frame0, frame1)?;
        self.validate_frames(frame2, frame3)?;

        let t = t.clamp(0.0, 1.0);

        // Catmull-Rom weights
        let t2 = t * t;
        let t3 = t2 * t;

        let w0 = -0.5 * t3 + t2 - 0.5 * t;
        let w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
        let w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
        let w3 = 0.5 * t3 - 0.5 * t2;

        // Normalize weights
        let sum = w0 + w1 + w2 + w3;
        let weights = vec![w0 / sum, w1 / sum, w2 / sum, w3 / sum];

        let frames: Vec<&Frame> = vec![frame0, frame1, frame2, frame3];
        self.blender.blend_multiple(&frames, &weights)
    }

    /// Interpolate multiple frames between two source frames.
    pub fn interpolate_multiple(
        &self,
        frame1: &Frame,
        frame2: &Frame,
        count: usize,
    ) -> Result<Vec<Frame>> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut frames = Vec::with_capacity(count);
        let step = 1.0 / (count + 1) as f32;

        for i in 1..=count {
            let t = step * i as f32;
            frames.push(self.interpolate(frame1, frame2, t)?);
        }

        Ok(frames)
    }

    /// Validate that two frames are compatible for interpolation.
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
                "Cannot interpolate between {} and {}",
                frame1.format(),
                frame2.format()
            )));
        }

        Ok(())
    }

    /// Nearest neighbor interpolation (frame selection).
    fn interpolate_nearest(&self, frame1: &Frame, frame2: &Frame, t: f32) -> Result<Frame> {
        if t < 0.5 {
            Ok(frame1.clone())
        } else {
            Ok(frame2.clone())
        }
    }

    /// Linear interpolation (blending).
    fn interpolate_linear(&self, frame1: &Frame, frame2: &Frame, t: f32) -> Result<Frame> {
        self.blender.blend(frame1, frame2, t)
    }

    /// Motion-compensated interpolation.
    fn interpolate_motion_compensated(
        &self,
        frame1: &Frame,
        frame2: &Frame,
        t: f32,
    ) -> Result<Frame> {
        // Estimate forward motion (frame1 -> frame2)
        let motion_field = self.motion_estimator.estimate(frame1, frame2)?;

        // Create interpolated frame
        let mut output = Frame::from_buffer(FrameBuffer::new(
            frame1.width(),
            frame1.height(),
            frame1.format(),
        ));

        // Interpolate each plane
        for plane_idx in 0..frame1.format().num_planes() {
            let scale = if plane_idx == 0 {
                1.0
            } else {
                let (hsub, _) = frame1.format().chroma_subsampling();
                1.0 / hsub as f32
            };

            self.interpolate_plane_motion(
                frame1.plane(plane_idx).unwrap(),
                frame2.plane(plane_idx).unwrap(),
                output.plane_mut(plane_idx).unwrap(),
                frame1.stride(plane_idx),
                &motion_field,
                t,
                scale,
            );
        }

        output.pts = frame1.pts;
        output.dts = frame1.dts;
        output.flags = frame1.flags;

        Ok(output)
    }

    /// Interpolate a single plane using motion compensation.
    fn interpolate_plane_motion(
        &self,
        plane1: &[u8],
        plane2: &[u8],
        output: &mut [u8],
        stride: usize,
        motion_field: &MotionField,
        t: f32,
        scale: f32,
    ) {
        let width = (motion_field.width as f32 * scale) as usize;
        let height = (motion_field.height as f32 * scale) as usize;

        for y in 0..height {
            for x in 0..width {
                // Get motion vector at this position
                let mv = motion_field.sample(x as f32 / scale, y as f32 / scale);

                // Calculate source positions
                let x1 = (x as f32 - mv.dx * t * scale).clamp(0.0, width as f32 - 1.0);
                let y1 = (y as f32 - mv.dy * t * scale).clamp(0.0, height as f32 - 1.0);

                let x2 = (x as f32 + mv.dx * (1.0 - t) * scale).clamp(0.0, width as f32 - 1.0);
                let y2 = (y as f32 + mv.dy * (1.0 - t) * scale).clamp(0.0, height as f32 - 1.0);

                // Bilinear sample from both frames
                let v1 = self.bilinear_sample(plane1, stride, x1, y1, width, height);
                let v2 = self.bilinear_sample(plane2, stride, x2, y2, width, height);

                // Blend based on confidence and position
                let blend_weight = t * mv.confidence + t * (1.0 - mv.confidence);
                let idx = y * stride + x;
                if idx < output.len() {
                    output[idx] = ((v1 as f32 * (1.0 - blend_weight) + v2 as f32 * blend_weight) + 0.5) as u8;
                }
            }
        }
    }

    /// Bilinear sample from a plane.
    fn bilinear_sample(
        &self,
        plane: &[u8],
        stride: usize,
        x: f32,
        y: f32,
        width: usize,
        height: usize,
    ) -> u8 {
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let idx00 = y0 * stride + x0;
        let idx10 = y0 * stride + x1;
        let idx01 = y1 * stride + x0;
        let idx11 = y1 * stride + x1;

        let v00 = plane.get(idx00).copied().unwrap_or(128) as f32;
        let v10 = plane.get(idx10).copied().unwrap_or(128) as f32;
        let v01 = plane.get(idx01).copied().unwrap_or(128) as f32;
        let v11 = plane.get(idx11).copied().unwrap_or(128) as f32;

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;

        (v0 * (1.0 - fy) + v1 * fy + 0.5) as u8
    }
}

impl Default for FrameInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-frame interpolator that maintains a buffer of frames.
pub struct BufferedInterpolator {
    interpolator: FrameInterpolator,
    /// Frame buffer for cubic interpolation.
    frame_buffer: Vec<Frame>,
    /// Maximum buffer size.
    buffer_size: usize,
}

impl BufferedInterpolator {
    /// Create a new buffered interpolator.
    pub fn new(method: InterpolationMethod) -> Self {
        let buffer_size = match method {
            InterpolationMethod::Cubic => 4,
            _ => 2,
        };

        Self {
            interpolator: FrameInterpolator::with_config(InterpolationConfig {
                method,
                ..Default::default()
            }),
            frame_buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }

    /// Push a new frame into the buffer.
    pub fn push(&mut self, frame: Frame) {
        if self.frame_buffer.len() >= self.buffer_size {
            self.frame_buffer.remove(0);
        }
        self.frame_buffer.push(frame);
    }

    /// Check if the buffer is ready for interpolation.
    pub fn is_ready(&self) -> bool {
        self.frame_buffer.len() >= 2
    }

    /// Get the number of frames in the buffer.
    pub fn buffered_frames(&self) -> usize {
        self.frame_buffer.len()
    }

    /// Interpolate a frame at the given position.
    ///
    /// For 2-frame methods, t is between frame[n-2] and frame[n-1].
    /// For 4-frame cubic, t is between frame[1] and frame[2].
    pub fn interpolate(&self, t: f32) -> Result<Frame> {
        if self.frame_buffer.len() < 2 {
            return Err(FrameRateError::InsufficientFrames {
                needed: 2,
                available: self.frame_buffer.len(),
            });
        }

        if self.frame_buffer.len() >= 4
            && self.interpolator.config.method == InterpolationMethod::Cubic
        {
            self.interpolator.interpolate_cubic(
                &self.frame_buffer[0],
                &self.frame_buffer[1],
                &self.frame_buffer[2],
                &self.frame_buffer[3],
                t,
            )
        } else {
            let len = self.frame_buffer.len();
            self.interpolator.interpolate(
                &self.frame_buffer[len - 2],
                &self.frame_buffer[len - 1],
                t,
            )
        }
    }

    /// Reset the buffer.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::{timestamp::TimeBase, PixelFormat};

    fn create_test_frame(width: u32, height: u32, value: u8) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);
        if let Some(plane) = frame.plane_mut(0) {
            plane.fill(value);
        }
        frame
    }

    #[test]
    fn test_nearest_interpolation() {
        let frame1 = create_test_frame(64, 64, 100);
        let frame2 = create_test_frame(64, 64, 200);

        let interpolator = FrameInterpolator::with_config(InterpolationConfig {
            method: InterpolationMethod::Nearest,
            ..Default::default()
        });

        // t < 0.5 should return frame1
        let result = interpolator.interpolate(&frame1, &frame2, 0.3).unwrap();
        assert_eq!(result.plane(0).unwrap()[0], 100);

        // t >= 0.5 should return frame2
        let result = interpolator.interpolate(&frame1, &frame2, 0.7).unwrap();
        assert_eq!(result.plane(0).unwrap()[0], 200);
    }

    #[test]
    fn test_linear_interpolation() {
        let frame1 = create_test_frame(64, 64, 100);
        let frame2 = create_test_frame(64, 64, 200);

        let interpolator = FrameInterpolator::with_config(InterpolationConfig {
            method: InterpolationMethod::Linear,
            ..Default::default()
        });

        let result = interpolator.interpolate(&frame1, &frame2, 0.5).unwrap();
        let y_plane = result.plane(0).unwrap();

        // Should be approximately 150 (midpoint)
        for &v in y_plane.iter().take(64 * 64) {
            assert!((v as i32 - 150).abs() <= 1);
        }
    }

    #[test]
    fn test_motion_compensated_interpolation() {
        let frame1 = create_test_frame(64, 64, 100);
        let frame2 = create_test_frame(64, 64, 200);

        let interpolator = FrameInterpolator::with_config(InterpolationConfig {
            method: InterpolationMethod::MotionCompensated,
            ..Default::default()
        });

        let result = interpolator.interpolate(&frame1, &frame2, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_interpolate_multiple() {
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(64, 64, 100);

        let interpolator = FrameInterpolator::new();
        let frames = interpolator.interpolate_multiple(&frame1, &frame2, 3).unwrap();

        assert_eq!(frames.len(), 3);

        // Check intermediate values
        for (i, frame) in frames.iter().enumerate() {
            let expected = ((i + 1) * 25) as u8; // 25, 50, 75
            let actual = frame.plane(0).unwrap()[0];
            assert!((actual as i32 - expected as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_cubic_interpolation() {
        let frame0 = create_test_frame(64, 64, 0);
        let frame1 = create_test_frame(64, 64, 50);
        let frame2 = create_test_frame(64, 64, 100);
        let frame3 = create_test_frame(64, 64, 150);

        let interpolator = FrameInterpolator::new();
        let result = interpolator
            .interpolate_cubic(&frame0, &frame1, &frame2, &frame3, 0.5)
            .unwrap();

        // Cubic interpolation with Catmull-Rom spline - result depends on weights
        // Just verify we get a valid result in reasonable range
        let y_plane = result.plane(0).unwrap();
        let first_value = y_plane[0] as i32;
        // Value should be somewhere between frame1 and frame2 values
        assert!(first_value >= 40 && first_value <= 110);
    }

    #[test]
    fn test_buffered_interpolator() {
        let mut buffered = BufferedInterpolator::new(InterpolationMethod::Linear);

        assert!(!buffered.is_ready());

        buffered.push(create_test_frame(64, 64, 100));
        assert!(!buffered.is_ready());

        buffered.push(create_test_frame(64, 64, 200));
        assert!(buffered.is_ready());

        let result = buffered.interpolate(0.5);
        assert!(result.is_ok());

        buffered.reset();
        assert!(!buffered.is_ready());
    }

    #[test]
    fn test_dimension_mismatch() {
        let frame1 = create_test_frame(64, 64, 100);
        let frame2 = create_test_frame(128, 128, 200);

        let interpolator = FrameInterpolator::new();
        let result = interpolator.interpolate(&frame1, &frame2, 0.5);

        assert!(matches!(
            result,
            Err(FrameRateError::DimensionMismatch { .. })
        ));
    }
}
