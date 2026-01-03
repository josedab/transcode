//! # transcode-deinterlace
//!
//! Deinterlacing filters for video processing in the transcode library.
//!
//! This crate provides multiple deinterlacing algorithms ranging from simple
//! and fast (bob) to sophisticated motion-adaptive filters (YADIF, BWDIF).
//!
//! ## Overview
//!
//! Interlaced video contains two fields per frame, each captured at slightly
//! different times. This was developed for CRT displays but causes artifacts
//! on modern progressive displays. Deinterlacing converts interlaced content
//! to progressive format.
//!
//! ## Available Deinterlacers
//!
//! | Algorithm | Quality | Speed | Temporal | Description |
//! |-----------|---------|-------|----------|-------------|
//! | Bob | Low | Fast | No | Simple line doubling |
//! | Linear | Medium | Fast | No | Linear interpolation |
//! | Bicubic | Medium | Medium | No | Bicubic interpolation |
//! | YADIF | High | Medium | Yes | Motion-adaptive, 3-frame |
//! | BWDIF | Highest | Slow | Yes | Motion-adaptive, 5-frame |
//!
//! ## Quick Start
//!
//! ```no_run
//! use transcode_deinterlace::{YadifDeinterlacer, YadifConfig};
//! use transcode_core::Frame;
//!
//! // Create a YADIF deinterlacer
//! let mut yadif = YadifDeinterlacer::new();
//!
//! // Process frames (push multiple frames for temporal filtering)
//! // let output_frames = yadif.push_frame(interlaced_frame)?;
//!
//! // Flush remaining frames when done
//! // let remaining = yadif.flush()?;
//! ```
//!
//! ## Interlace Detection
//!
//! The detection module can analyze video to determine:
//! - Whether content is interlaced or progressive
//! - Field order (top-field-first or bottom-field-first)
//! - Telecine patterns (3:2 pulldown)
//!
//! ```no_run
//! use transcode_deinterlace::{InterlaceDetector, ContentType};
//!
//! let mut detector = InterlaceDetector::new();
//!
//! // Analyze multiple frames
//! // for frame in frames {
//! //     detector.analyze_frame(&frame)?;
//! // }
//!
//! // Get detection result
//! // match detector.get_content_type() {
//! //     ContentType::Progressive => println!("Progressive content"),
//! //     ContentType::Interlaced { field_order } => println!("Interlaced: {:?}", field_order),
//! //     ContentType::Telecine { pattern, .. } => println!("Telecine: {:?}", pattern),
//! //     _ => println!("Unknown or mixed"),
//! // }
//! ```
//!
//! ## Inverse Telecine (IVTC)
//!
//! For film content that has been telecined (24fps -> 29.97fps), the IVTC
//! processor can recover the original film frames:
//!
//! ```no_run
//! use transcode_deinterlace::{InverseTelecine, TelecinePattern};
//! use transcode_deinterlace::bob::FieldOrder;
//!
//! let mut ivtc = InverseTelecine::new(
//!     TelecinePattern::Pulldown32,
//!     FieldOrder::TopFieldFirst,
//! );
//!
//! // Process telecined frames
//! // let film_frames = ivtc.push_frame(video_frame)?;
//! ```
//!
//! ## Algorithm Selection Guide
//!
//! - **Bob**: Use when speed is critical or for real-time preview.
//! - **Linear**: Good balance for non-temporal content.
//! - **YADIF**: Recommended for most content - good quality/speed tradeoff.
//! - **BWDIF**: Best quality but slower - use for final output.
//!
//! ## Feature Flags
//!
//! - `simd`: Enable SIMD optimizations (SSE2/AVX2/NEON)

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod bob;
pub mod bwdif;
pub mod detect;
pub mod error;
pub mod linear;
pub mod yadif;

// Re-export main types
pub use bob::{BobConfig, BobDeinterlacer, FieldOrder, FieldSelect};
pub use bwdif::{BwdifConfig, BwdifDeinterlacer, BwdifMode, BwdifParity};
pub use detect::{
    ContentType, DetectionConfig, FieldOrderDetector, FrameAnalysis, InterlaceDetector,
    InverseTelecine, TelecinePattern,
};
pub use error::{DeinterlaceError, Result};
pub use linear::{BicubicDeinterlacer, LinearConfig, LinearDeinterlacer};
pub use yadif::{YadifConfig, YadifDeinterlacer, YadifMode, YadifParity};

/// Deinterlacer trait for common interface.
pub trait Deinterlacer {
    /// Process a frame and return deinterlaced output.
    ///
    /// For temporal filters, this may return empty until enough context is available.
    fn process(&mut self, frame: transcode_core::Frame) -> Result<Vec<transcode_core::Frame>>;

    /// Flush any buffered frames.
    fn flush(&mut self) -> Result<Vec<transcode_core::Frame>>;

    /// Reset the deinterlacer state.
    fn reset(&mut self);
}

/// Wrapper to implement Deinterlacer for BobDeinterlacer.
pub struct BobDeinterlacerWrapper {
    inner: BobDeinterlacer,
}

impl BobDeinterlacerWrapper {
    /// Create a new wrapper.
    pub fn new(config: BobConfig) -> Self {
        Self {
            inner: BobDeinterlacer::with_config(config),
        }
    }
}

impl Deinterlacer for BobDeinterlacerWrapper {
    fn process(&mut self, frame: transcode_core::Frame) -> Result<Vec<transcode_core::Frame>> {
        self.inner.process(&frame)
    }

    fn flush(&mut self) -> Result<Vec<transcode_core::Frame>> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        // Bob has no state to reset
    }
}

impl Deinterlacer for YadifDeinterlacer {
    fn process(&mut self, frame: transcode_core::Frame) -> Result<Vec<transcode_core::Frame>> {
        self.push_frame(frame)
    }

    fn flush(&mut self) -> Result<Vec<transcode_core::Frame>> {
        YadifDeinterlacer::flush(self)
    }

    fn reset(&mut self) {
        YadifDeinterlacer::reset(self)
    }
}

impl Deinterlacer for BwdifDeinterlacer {
    fn process(&mut self, frame: transcode_core::Frame) -> Result<Vec<transcode_core::Frame>> {
        self.push_frame(frame)
    }

    fn flush(&mut self) -> Result<Vec<transcode_core::Frame>> {
        BwdifDeinterlacer::flush(self)
    }

    fn reset(&mut self) {
        BwdifDeinterlacer::reset(self)
    }
}

/// Wrapper to implement Deinterlacer for LinearDeinterlacer.
pub struct LinearDeinterlacerWrapper {
    inner: LinearDeinterlacer,
}

impl LinearDeinterlacerWrapper {
    /// Create a new wrapper.
    pub fn new(config: LinearConfig) -> Self {
        Self {
            inner: LinearDeinterlacer::with_config(config),
        }
    }
}

impl Deinterlacer for LinearDeinterlacerWrapper {
    fn process(&mut self, frame: transcode_core::Frame) -> Result<Vec<transcode_core::Frame>> {
        Ok(vec![self.inner.process(&frame)?])
    }

    fn flush(&mut self) -> Result<Vec<transcode_core::Frame>> {
        Ok(Vec::new())
    }

    fn reset(&mut self) {
        // Linear has no state to reset
    }
}

/// Available deinterlacing algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeinterlaceAlgorithm {
    /// Bob deinterlacing (simple line doubling).
    Bob,
    /// Linear interpolation.
    Linear,
    /// Bicubic interpolation.
    Bicubic,
    /// YADIF (Yet Another Deinterlacing Filter).
    #[default]
    Yadif,
    /// BWDIF (Bob-Weave Deinterlacing Filter).
    Bwdif,
}

impl std::fmt::Display for DeinterlaceAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeinterlaceAlgorithm::Bob => write!(f, "bob"),
            DeinterlaceAlgorithm::Linear => write!(f, "linear"),
            DeinterlaceAlgorithm::Bicubic => write!(f, "bicubic"),
            DeinterlaceAlgorithm::Yadif => write!(f, "yadif"),
            DeinterlaceAlgorithm::Bwdif => write!(f, "bwdif"),
        }
    }
}

/// Create a deinterlacer from an algorithm selection.
pub fn create_deinterlacer(algorithm: DeinterlaceAlgorithm) -> Box<dyn Deinterlacer> {
    match algorithm {
        DeinterlaceAlgorithm::Bob => {
            Box::new(BobDeinterlacerWrapper::new(BobConfig::default()))
        }
        DeinterlaceAlgorithm::Linear => {
            Box::new(LinearDeinterlacerWrapper::new(LinearConfig::default()))
        }
        DeinterlaceAlgorithm::Bicubic => {
            // Use linear wrapper for bicubic (simplified)
            Box::new(LinearDeinterlacerWrapper::new(LinearConfig::default()))
        }
        DeinterlaceAlgorithm::Yadif => Box::new(YadifDeinterlacer::new()),
        DeinterlaceAlgorithm::Bwdif => Box::new(BwdifDeinterlacer::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::frame::FrameFlags;
    use transcode_core::{Frame, PixelFormat, TimeBase};

    fn create_test_frame(width: u32, height: u32) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);
        frame.flags = FrameFlags::INTERLACED | FrameFlags::TOP_FIELD_FIRST;

        let stride = frame.stride(0);
        if let Some(y_plane) = frame.plane_mut(0) {
            for y in 0..height as usize {
                let value = if y % 2 == 0 { 100u8 } else { 200u8 };
                for x in 0..width as usize {
                    if y * stride + x < y_plane.len() {
                        y_plane[y * stride + x] = value;
                    }
                }
            }
        }

        frame
    }

    #[test]
    fn test_deinterlacer_trait_bob() {
        let mut deint = BobDeinterlacerWrapper::new(BobConfig::default());

        let frame = create_test_frame(32, 32);
        let result = deint.process(frame).unwrap();

        assert_eq!(result.len(), 2); // Both fields
        for output in result {
            assert!(!output.flags.contains(FrameFlags::INTERLACED));
        }
    }

    #[test]
    fn test_deinterlacer_trait_yadif() {
        let mut deint: Box<dyn Deinterlacer> = Box::new(YadifDeinterlacer::new());

        let frame1 = create_test_frame(32, 32);
        let frame2 = create_test_frame(32, 32);

        let r1 = deint.process(frame1).unwrap();
        assert!(r1.is_empty()); // Need more context

        let r2 = deint.process(frame2).unwrap();
        assert!(!r2.is_empty());

        let flush = deint.flush().unwrap();
        assert!(!flush.is_empty());
    }

    #[test]
    fn test_create_deinterlacer() {
        for algo in [
            DeinterlaceAlgorithm::Bob,
            DeinterlaceAlgorithm::Linear,
            DeinterlaceAlgorithm::Yadif,
            DeinterlaceAlgorithm::Bwdif,
        ] {
            let mut deint = create_deinterlacer(algo);

            // Should be able to process frames
            let frame = create_test_frame(32, 32);
            let _ = deint.process(frame);

            // Should be able to reset
            deint.reset();
        }
    }

    #[test]
    fn test_algorithm_display() {
        assert_eq!(format!("{}", DeinterlaceAlgorithm::Bob), "bob");
        assert_eq!(format!("{}", DeinterlaceAlgorithm::Yadif), "yadif");
        assert_eq!(format!("{}", DeinterlaceAlgorithm::Bwdif), "bwdif");
    }

    #[test]
    fn test_field_order() {
        assert_eq!(
            FieldOrder::TopFieldFirst.opposite(),
            FieldOrder::BottomFieldFirst
        );
        assert_eq!(
            FieldOrder::BottomFieldFirst.opposite(),
            FieldOrder::TopFieldFirst
        );
    }

    #[test]
    fn test_content_type_variants() {
        let progressive = ContentType::Progressive;
        let interlaced = ContentType::Interlaced {
            field_order: FieldOrder::TopFieldFirst,
        };
        let telecine = ContentType::Telecine {
            pattern: TelecinePattern::Pulldown32,
            field_order: FieldOrder::TopFieldFirst,
        };

        assert_eq!(progressive, ContentType::Progressive);
        assert_ne!(progressive, interlaced);
        assert_ne!(interlaced, telecine);
    }

    #[test]
    fn test_integrated_detection_and_deinterlace() {
        let mut detector = InterlaceDetector::new();
        let mut frames = Vec::new();

        // Create test frames
        for i in 0..15 {
            let mut frame = Frame::new(32, 32, PixelFormat::Yuv420p, TimeBase::MPEG);
            frame.flags = FrameFlags::INTERLACED | FrameFlags::TOP_FIELD_FIRST;

            let stride = frame.stride(0);
            if let Some(y_plane) = frame.plane_mut(0) {
                for y in 0..32 {
                    let value = if y % 2 == 0 { 50 + i * 10 } else { 200 - i * 10 };
                    for x in 0..32 {
                        y_plane[y * stride + x] = value;
                    }
                }
            }

            frames.push(frame);
        }

        // Analyze frames
        for frame in &frames {
            detector.analyze_frame(frame).unwrap();
        }

        // Detect content type
        let content_type = detector.get_content_type();

        // Choose deinterlacer based on detection
        let algorithm = match content_type {
            ContentType::Progressive => None,
            ContentType::Interlaced { .. } => Some(DeinterlaceAlgorithm::Yadif),
            ContentType::Telecine { .. } => Some(DeinterlaceAlgorithm::Bwdif),
            _ => Some(DeinterlaceAlgorithm::Linear),
        };

        if let Some(algo) = algorithm {
            let mut deint = create_deinterlacer(algo);

            for frame in frames {
                let _ = deint.process(frame);
            }

            let _ = deint.flush();
        }
    }
}
