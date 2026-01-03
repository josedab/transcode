//! # Transcode Frame Rate
//!
//! Frame rate conversion library for the Transcode project.
//!
//! This crate provides comprehensive frame rate conversion functionality including:
//!
//! - **Simple conversions**: Frame dropping and duplication
//! - **Frame blending**: Weighted averaging of adjacent frames
//! - **Motion-compensated interpolation**: Smooth frame interpolation using motion estimation
//! - **Telecine patterns**: 3:2 pulldown, 2:2 pulldown, and Euro pulldown
//! - **Inverse telecine**: Recovery of original film frames from telecined content
//! - **Variable frame rate handling**: Conversion from VFR to CFR
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use transcode_framerate::{FrameRateConverter, ConversionConfig, StandardFrameRate};
//! use transcode_core::Rational;
//!
//! // Create a converter for 24->30 fps conversion
//! let mut converter = FrameRateConverter::standard(
//!     StandardFrameRate::Film24,
//!     StandardFrameRate::Ntsc30,
//! );
//!
//! // Process frames
//! for frame in input_frames {
//!     let output_frames = converter.process(&frame)?;
//!     for output in output_frames {
//!         // Use output frames
//!     }
//! }
//!
//! // Flush remaining frames
//! let remaining = converter.flush();
//! ```
//!
//! ## Frame Blending
//!
//! For smooth transitions between frame rates:
//!
//! ```rust,ignore
//! use transcode_framerate::{FrameBlender, BlendMode, BlendConfig};
//!
//! let blender = FrameBlender::with_config(BlendConfig {
//!     mode: BlendMode::SmoothStep,
//!     dithering: true,
//!     ..Default::default()
//! });
//!
//! let blended = blender.blend(&frame1, &frame2, 0.5)?;
//! ```
//!
//! ## Motion-Compensated Interpolation
//!
//! For high-quality frame interpolation:
//!
//! ```rust,ignore
//! use transcode_framerate::{FrameInterpolator, InterpolationMethod, InterpolationConfig};
//!
//! let interpolator = FrameInterpolator::with_config(InterpolationConfig {
//!     method: InterpolationMethod::MotionCompensated,
//!     ..Default::default()
//! });
//!
//! let interpolated = interpolator.interpolate(&frame1, &frame2, 0.5)?;
//! ```
//!
//! ## Telecine Handling
//!
//! For film-to-video conversions:
//!
//! ```rust,ignore
//! use transcode_framerate::{TelecineApplier, InverseTelecine, TelecinePattern};
//!
//! // Apply 3:2 pulldown (24 -> 30 fps)
//! let mut telecine = TelecineApplier::new(TelecinePattern::Pulldown32);
//! let output_frames = telecine.process(&input_frame)?;
//!
//! // Remove telecine (30 -> 24 fps)
//! let mut ivtc = InverseTelecine::with_pattern(TelecinePattern::Pulldown32);
//! let recovered_frames = ivtc.process(&telecined_frame)?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod blend;
pub mod converter;
pub mod error;
pub mod interpolation;
pub mod motion;
pub mod telecine;

// Re-export main types
pub use blend::{BlendConfig, BlendMode, FrameBlender, GhostBlender};
pub use converter::{
    ConversionConfig, ConversionMethod, FrameRateConverter, StandardFrameRate, VfrHandler,
};
pub use error::{FrameRateError, Result};
pub use interpolation::{
    BufferedInterpolator, FrameInterpolator, InterpolationConfig, InterpolationMethod,
    OcclusionHandling,
};
pub use motion::{
    MotionEstimationAlgorithm, MotionEstimationConfig, MotionEstimator, MotionField, MotionVector,
};
pub use telecine::{InverseTelecine, TelecineApplier, TelecineInfo, TelecinePattern};

/// Convenience function to create a frame rate converter for common conversions.
pub fn create_converter(
    source_fps: transcode_core::Rational,
    target_fps: transcode_core::Rational,
) -> FrameRateConverter {
    let method = converter::suggest_conversion_method(source_fps, target_fps);

    let telecine_pattern = if method == ConversionMethod::Telecine
        || method == ConversionMethod::InverseTelecine
    {
        TelecinePattern::Pulldown32
    } else {
        TelecinePattern::None
    };

    FrameRateConverter::new(ConversionConfig {
        source_fps,
        target_fps,
        method,
        telecine_pattern,
        ..Default::default()
    })
}

/// Convenience function to blend two frames.
pub fn blend_frames(
    frame1: &transcode_core::Frame,
    frame2: &transcode_core::Frame,
    weight: f32,
) -> Result<transcode_core::Frame> {
    let blender = FrameBlender::new();
    blender.blend(frame1, frame2, weight)
}

/// Convenience function to interpolate between two frames.
pub fn interpolate_frames(
    frame1: &transcode_core::Frame,
    frame2: &transcode_core::Frame,
    t: f32,
    method: InterpolationMethod,
) -> Result<transcode_core::Frame> {
    let interpolator = FrameInterpolator::with_config(InterpolationConfig {
        method,
        ..Default::default()
    });
    interpolator.interpolate(frame1, frame2, t)
}

/// Detect telecine pattern in a sequence of frames.
pub fn detect_telecine_pattern(frames: &[transcode_core::Frame]) -> Result<TelecineInfo> {
    telecine::detect_telecine(frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::{timestamp::TimeBase, Frame, PixelFormat, Rational};

    fn create_test_frame(value: u8) -> Frame {
        let mut frame = Frame::new(64, 64, PixelFormat::Yuv420p, TimeBase::MPEG);
        if let Some(plane) = frame.plane_mut(0) {
            plane.fill(value);
        }
        frame
    }

    #[test]
    fn test_create_converter() {
        let converter = create_converter(Rational::new(24, 1), Rational::new(30, 1));
        assert_eq!(converter.config().method, ConversionMethod::Telecine);
    }

    #[test]
    fn test_blend_frames_convenience() {
        let frame1 = create_test_frame(100);
        let frame2 = create_test_frame(200);

        let result = blend_frames(&frame1, &frame2, 0.5);
        assert!(result.is_ok());

        let blended = result.unwrap();
        let y_plane = blended.plane(0).unwrap();
        for &v in y_plane.iter().take(64 * 64) {
            assert!((v as i32 - 150).abs() <= 1);
        }
    }

    #[test]
    fn test_interpolate_frames_convenience() {
        let frame1 = create_test_frame(0);
        let frame2 = create_test_frame(100);

        let result = interpolate_frames(&frame1, &frame2, 0.5, InterpolationMethod::Linear);
        assert!(result.is_ok());
    }

    #[test]
    fn test_detect_telecine_convenience() {
        let frames: Vec<Frame> = (0..5).map(|i| create_test_frame(i * 20)).collect();
        let result = detect_telecine_pattern(&frames);
        // Should fail due to insufficient frames
        assert!(result.is_err());

        let frames: Vec<Frame> = (0..15).map(|i| create_test_frame(i as u8 * 10)).collect();
        let result = detect_telecine_pattern(&frames);
        assert!(result.is_ok());
    }

    #[test]
    fn test_full_conversion_pipeline() {
        // Test a complete 24->30 conversion with telecine
        let mut converter = create_converter(Rational::new(24, 1), Rational::new(30, 1));

        let mut total_output = 0;
        for i in 0..24 {
            let output = converter.process(&create_test_frame(i as u8 * 10)).unwrap();
            total_output += output.len();
        }

        let flushed = converter.flush();
        total_output += flushed.len();

        // 24 input frames with 3:2 pulldown produce 60 output frames
        // Pattern: 2+3+2+3 = 10 frames per 4 input frames
        // 24 input frames = 6 cycles * 10 = 60 output frames
        assert!(
            total_output >= 55 && total_output <= 65,
            "Expected ~60 output frames, got {}",
            total_output
        );
    }

    #[test]
    fn test_interpolation_methods() {
        let frame1 = create_test_frame(50);
        let frame2 = create_test_frame(150);

        for method in [
            InterpolationMethod::Nearest,
            InterpolationMethod::Linear,
            InterpolationMethod::MotionCompensated,
        ] {
            let result = interpolate_frames(&frame1, &frame2, 0.5, method);
            assert!(result.is_ok(), "Failed for method {:?}", method);
        }
    }

    #[test]
    fn test_standard_conversions() {
        // Test common frame rate conversions
        let conversions = [
            (StandardFrameRate::Film24, StandardFrameRate::Ntsc30),
            (StandardFrameRate::Ntsc30, StandardFrameRate::Film24),
            (StandardFrameRate::Pal25, StandardFrameRate::Ntsc30),
            (StandardFrameRate::NtscFilm, StandardFrameRate::Hfr60),
        ];

        for (source, target) in conversions {
            let mut converter = FrameRateConverter::standard(source, target);
            for i in 0..10 {
                let result = converter.process(&create_test_frame(i * 25));
                assert!(
                    result.is_ok(),
                    "Failed for {:?} -> {:?}",
                    source,
                    target
                );
            }
        }
    }
}
