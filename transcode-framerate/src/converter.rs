//! Frame rate converter implementation.
//!
//! This module provides the main frame rate conversion functionality,
//! supporting various conversion methods and frame rate combinations.

use crate::blend::FrameBlender;
use crate::error::{FrameRateError, Result};
use crate::interpolation::{FrameInterpolator, InterpolationConfig};
use crate::telecine::{InverseTelecine, TelecineApplier, TelecinePattern};
use transcode_core::{Frame, Rational};
use tracing::{debug, trace};

/// Common frame rates used in video production.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StandardFrameRate {
    /// Film: 24 fps (24/1)
    Film24,
    /// NTSC Film: 23.976 fps (24000/1001)
    NtscFilm,
    /// PAL: 25 fps (25/1)
    Pal25,
    /// NTSC: 29.97 fps (30000/1001)
    Ntsc30,
    /// Web/Games: 30 fps (30/1)
    Web30,
    /// PAL Interlaced: 50i (50/1)
    Pal50i,
    /// NTSC Interlaced: 59.94i (60000/1001)
    Ntsc60i,
    /// High Frame Rate: 48 fps (48/1)
    Hfr48,
    /// High Frame Rate: 60 fps (60/1)
    Hfr60,
    /// High Frame Rate: 120 fps (120/1)
    Hfr120,
}

impl StandardFrameRate {
    /// Convert to rational frame rate.
    pub fn to_rational(&self) -> Rational {
        match self {
            Self::Film24 => Rational::new(24, 1),
            Self::NtscFilm => Rational::new(24000, 1001),
            Self::Pal25 => Rational::new(25, 1),
            Self::Ntsc30 => Rational::new(30000, 1001),
            Self::Web30 => Rational::new(30, 1),
            Self::Pal50i => Rational::new(50, 1),
            Self::Ntsc60i => Rational::new(60000, 1001),
            Self::Hfr48 => Rational::new(48, 1),
            Self::Hfr60 => Rational::new(60, 1),
            Self::Hfr120 => Rational::new(120, 1),
        }
    }

    /// Get standard frame rate from rational, if it matches.
    pub fn from_rational(fps: Rational) -> Option<Self> {
        let reduced = fps.reduce();
        match (reduced.num, reduced.den) {
            (24, 1) => Some(Self::Film24),
            (24000, 1001) => Some(Self::NtscFilm),
            (25, 1) => Some(Self::Pal25),
            (30000, 1001) => Some(Self::Ntsc30),
            (30, 1) => Some(Self::Web30),
            (50, 1) => Some(Self::Pal50i),
            (60000, 1001) => Some(Self::Ntsc60i),
            (48, 1) => Some(Self::Hfr48),
            (60, 1) => Some(Self::Hfr60),
            (120, 1) => Some(Self::Hfr120),
            _ => None,
        }
    }
}

/// Frame rate conversion method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConversionMethod {
    /// Drop or duplicate frames as needed.
    DropDuplicate,
    /// Blend adjacent frames.
    #[default]
    Blend,
    /// Motion-compensated interpolation.
    MotionCompensated,
    /// Apply telecine pattern (24 -> 30).
    Telecine,
    /// Remove telecine pattern (30 -> 24).
    InverseTelecine,
    /// Speed change (affects audio sync).
    SpeedChange,
}

/// Configuration for frame rate conversion.
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Source frame rate.
    pub source_fps: Rational,
    /// Target frame rate.
    pub target_fps: Rational,
    /// Conversion method.
    pub method: ConversionMethod,
    /// Interpolation configuration.
    pub interpolation: InterpolationConfig,
    /// Telecine pattern (for telecine/inverse telecine).
    pub telecine_pattern: TelecinePattern,
    /// Drop threshold for frame dropping (0.0-1.0).
    pub drop_threshold: f32,
    /// Maximum frame buffer size.
    pub max_buffer_size: usize,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            source_fps: Rational::new(24, 1),
            target_fps: Rational::new(30, 1),
            method: ConversionMethod::Blend,
            interpolation: InterpolationConfig::default(),
            telecine_pattern: TelecinePattern::None,
            drop_threshold: 0.5,
            max_buffer_size: 8,
        }
    }
}

impl ConversionConfig {
    /// Create configuration for a standard conversion.
    pub fn standard(source: StandardFrameRate, target: StandardFrameRate) -> Self {
        let source_fps = source.to_rational();
        let target_fps = target.to_rational();

        // Determine best method based on source/target
        let (method, telecine_pattern) = Self::detect_best_method(source, target);

        Self {
            source_fps,
            target_fps,
            method,
            telecine_pattern,
            ..Default::default()
        }
    }

    /// Detect the best conversion method for given frame rates.
    fn detect_best_method(
        source: StandardFrameRate,
        target: StandardFrameRate,
    ) -> (ConversionMethod, TelecinePattern) {
        use StandardFrameRate::*;

        match (source, target) {
            // Telecine: 24 -> 30
            (Film24 | NtscFilm, Ntsc30 | Web30) => {
                (ConversionMethod::Telecine, TelecinePattern::Pulldown32)
            }
            // Inverse telecine: 30 -> 24
            (Ntsc30 | Web30, Film24 | NtscFilm) => {
                (ConversionMethod::InverseTelecine, TelecinePattern::Pulldown32)
            }
            // PAL speedup: 24 -> 25
            (Film24, Pal25) => (ConversionMethod::SpeedChange, TelecinePattern::EuroPulldown),
            // PAL slowdown: 25 -> 24
            (Pal25, Film24) => (ConversionMethod::SpeedChange, TelecinePattern::EuroPulldown),
            // Same or similar rates
            _ if source == target => (ConversionMethod::DropDuplicate, TelecinePattern::None),
            // Default to blending
            _ => (ConversionMethod::Blend, TelecinePattern::None),
        }
    }

    /// Calculate the frame rate ratio.
    pub fn rate_ratio(&self) -> f64 {
        self.target_fps.to_f64() / self.source_fps.to_f64()
    }

    /// Check if this is an upconversion (increasing frame rate).
    pub fn is_upconversion(&self) -> bool {
        self.target_fps > self.source_fps
    }

    /// Check if this is a downconversion (decreasing frame rate).
    pub fn is_downconversion(&self) -> bool {
        self.target_fps < self.source_fps
    }
}

/// Frame rate converter state.
pub struct FrameRateConverter {
    config: ConversionConfig,
    /// Frame buffer for interpolation.
    frame_buffer: Vec<Frame>,
    /// Accumulator for frame timing.
    accumulator: f64,
    /// Source frame counter.
    source_count: u64,
    /// Output frame counter.
    output_count: u64,
    /// Frame interpolator.
    interpolator: FrameInterpolator,
    /// Frame blender.
    blender: FrameBlender,
    /// Telecine applier.
    telecine: Option<TelecineApplier>,
    /// Inverse telecine processor.
    inverse_telecine: Option<InverseTelecine>,
}

impl FrameRateConverter {
    /// Create a new frame rate converter.
    pub fn new(config: ConversionConfig) -> Self {
        let telecine = match config.method {
            ConversionMethod::Telecine => Some(TelecineApplier::new(config.telecine_pattern)),
            _ => None,
        };

        let inverse_telecine = match config.method {
            ConversionMethod::InverseTelecine => {
                Some(InverseTelecine::with_pattern(config.telecine_pattern))
            }
            _ => None,
        };

        let interpolator = FrameInterpolator::with_config(config.interpolation.clone());

        Self {
            config,
            frame_buffer: Vec::new(),
            accumulator: 0.0,
            source_count: 0,
            output_count: 0,
            interpolator,
            blender: FrameBlender::new(),
            telecine,
            inverse_telecine,
        }
    }

    /// Create a converter for standard frame rate conversion.
    pub fn standard(source: StandardFrameRate, target: StandardFrameRate) -> Self {
        Self::new(ConversionConfig::standard(source, target))
    }

    /// Get the conversion configuration.
    pub fn config(&self) -> &ConversionConfig {
        &self.config
    }

    /// Get the number of source frames processed.
    pub fn source_frame_count(&self) -> u64 {
        self.source_count
    }

    /// Get the number of output frames produced.
    pub fn output_frame_count(&self) -> u64 {
        self.output_count
    }

    /// Reset the converter state.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.accumulator = 0.0;
        self.source_count = 0;
        self.output_count = 0;

        if let Some(ref mut telecine) = self.telecine {
            telecine.reset();
        }
        if let Some(ref mut ivtc) = self.inverse_telecine {
            ivtc.reset();
        }
    }

    /// Process a single input frame and return output frames.
    pub fn process(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        self.source_count += 1;
        trace!("Processing source frame {}", self.source_count);

        let result = match self.config.method {
            ConversionMethod::DropDuplicate => self.process_drop_duplicate(frame),
            ConversionMethod::Blend => self.process_blend(frame),
            ConversionMethod::MotionCompensated => self.process_motion_compensated(frame),
            ConversionMethod::Telecine => self.process_telecine(frame),
            ConversionMethod::InverseTelecine => self.process_inverse_telecine(frame),
            ConversionMethod::SpeedChange => self.process_speed_change(frame),
        };

        if let Ok(ref frames) = result {
            self.output_count += frames.len() as u64;
        }

        result
    }

    /// Flush any remaining buffered frames.
    pub fn flush(&mut self) -> Vec<Frame> {
        debug!(
            "Flushing converter: {} frames in buffer",
            self.frame_buffer.len()
        );

        let mut output = Vec::new();

        // For telecine, flush is simple
        if let Some(ref mut ivtc) = self.inverse_telecine {
            output.extend(ivtc.flush());
        }

        // Output any remaining buffered frames
        output.append(&mut self.frame_buffer);

        output
    }

    /// Process frame using drop/duplicate method.
    fn process_drop_duplicate(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        let ratio = self.config.rate_ratio();

        // Calculate how many output frames this input should produce
        let prev_output = (self.accumulator * ratio).floor() as u64;
        self.accumulator += 1.0;
        let new_output = (self.accumulator * ratio).floor() as u64;

        let count = (new_output - prev_output) as usize;

        if count == 0 {
            // Drop this frame
            trace!("Dropping frame {}", self.source_count);
            Ok(Vec::new())
        } else {
            // Duplicate as needed
            trace!("Duplicating frame {} x{}", self.source_count, count);
            Ok(vec![frame.clone(); count])
        }
    }

    /// Process frame using blending method.
    fn process_blend(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        self.frame_buffer.push(frame.clone());

        // Need at least 2 frames for blending
        if self.frame_buffer.len() < 2 {
            return Ok(Vec::new());
        }

        // Limit buffer size
        while self.frame_buffer.len() > self.config.max_buffer_size {
            self.frame_buffer.remove(0);
        }

        let _ratio = self.config.rate_ratio();
        let mut output = Vec::new();

        // Calculate output frames for this input interval
        let source_interval = 1.0 / self.config.source_fps.to_f64();
        let target_interval = 1.0 / self.config.target_fps.to_f64();

        let prev_time = (self.source_count as f64 - 1.0) * source_interval;
        let curr_time = self.source_count as f64 * source_interval;

        // Generate output frames in this time interval
        let start_output = (prev_time / target_interval).floor() as u64;
        let end_output = (curr_time / target_interval).floor() as u64;

        for output_idx in start_output..end_output {
            let output_time = (output_idx as f64 + 1.0) * target_interval;
            let t = ((output_time - prev_time) / source_interval).clamp(0.0, 1.0) as f32;

            let len = self.frame_buffer.len();
            if len >= 2 {
                let blended = self.blender.blend(
                    &self.frame_buffer[len - 2],
                    &self.frame_buffer[len - 1],
                    t,
                )?;
                output.push(blended);
            }
        }

        // Clean up old frames
        if self.frame_buffer.len() > 2 {
            self.frame_buffer.remove(0);
        }

        Ok(output)
    }

    /// Process frame using motion-compensated interpolation.
    fn process_motion_compensated(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        self.frame_buffer.push(frame.clone());

        if self.frame_buffer.len() < 2 {
            return Ok(Vec::new());
        }

        // Limit buffer size
        while self.frame_buffer.len() > self.config.max_buffer_size {
            self.frame_buffer.remove(0);
        }

        let _ratio = self.config.rate_ratio();
        let mut output = Vec::new();

        // Similar timing logic as blend, but use motion-compensated interpolation
        let source_interval = 1.0 / self.config.source_fps.to_f64();
        let target_interval = 1.0 / self.config.target_fps.to_f64();

        let prev_time = (self.source_count as f64 - 1.0) * source_interval;
        let curr_time = self.source_count as f64 * source_interval;

        let start_output = (prev_time / target_interval).floor() as u64;
        let end_output = (curr_time / target_interval).floor() as u64;

        for output_idx in start_output..end_output {
            let output_time = (output_idx as f64 + 1.0) * target_interval;
            let t = ((output_time - prev_time) / source_interval).clamp(0.0, 1.0) as f32;

            let len = self.frame_buffer.len();
            if len >= 2 {
                let interpolated = self.interpolator.interpolate(
                    &self.frame_buffer[len - 2],
                    &self.frame_buffer[len - 1],
                    t,
                )?;
                output.push(interpolated);
            }
        }

        // Clean up old frames
        if self.frame_buffer.len() > 2 {
            self.frame_buffer.remove(0);
        }

        Ok(output)
    }

    /// Process frame using telecine.
    fn process_telecine(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        if let Some(ref mut telecine) = self.telecine {
            telecine.process(frame)
        } else {
            Err(FrameRateError::invalid_params("Telecine not configured"))
        }
    }

    /// Process frame using inverse telecine.
    fn process_inverse_telecine(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        if let Some(ref mut ivtc) = self.inverse_telecine {
            ivtc.process(frame)
        } else {
            Err(FrameRateError::invalid_params(
                "Inverse telecine not configured",
            ))
        }
    }

    /// Process frame using speed change (for PAL conversion).
    fn process_speed_change(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        // Speed change simply passes frames through
        // The actual speed change happens at the container/player level
        // For 24->25, playback is 4.1667% faster
        // For 25->24, playback is 4% slower
        Ok(vec![frame.clone()])
    }
}

/// Calculate the optimal conversion method for given frame rates.
pub fn suggest_conversion_method(source_fps: Rational, target_fps: Rational) -> ConversionMethod {
    let source_std = StandardFrameRate::from_rational(source_fps);
    let target_std = StandardFrameRate::from_rational(target_fps);

    if let (Some(src), Some(tgt)) = (source_std, target_std) {
        let (method, _) = ConversionConfig::detect_best_method(src, tgt);
        return method;
    }

    // For non-standard rates, use ratio to decide
    let ratio = target_fps.to_f64() / source_fps.to_f64();

    if (ratio - 1.0).abs() < 0.001 {
        // Nearly identical rates
        ConversionMethod::DropDuplicate
    } else if !(0.67..=1.5).contains(&ratio) {
        // Large difference - use motion compensation
        ConversionMethod::MotionCompensated
    } else {
        // Moderate difference - blend
        ConversionMethod::Blend
    }
}

/// Variable frame rate handler.
pub struct VfrHandler {
    /// Target frame rate.
    target_fps: Rational,
    /// Minimum source frame interval.
    min_interval: f64,
    /// Maximum source frame interval.
    max_interval: f64,
    /// Last output timestamp.
    last_output_time: f64,
    /// Frame buffer.
    frame_buffer: Vec<(Frame, f64)>,
    /// Interpolator.
    interpolator: FrameInterpolator,
}

impl VfrHandler {
    /// Create a new VFR handler.
    pub fn new(target_fps: Rational) -> Self {
        Self {
            target_fps,
            min_interval: f64::MAX,
            max_interval: 0.0,
            last_output_time: 0.0,
            frame_buffer: Vec::new(),
            interpolator: FrameInterpolator::new(),
        }
    }

    /// Process a frame with its presentation time.
    pub fn process(&mut self, frame: Frame, pts_seconds: f64) -> Result<Vec<Frame>> {
        // Update interval statistics
        if let Some((_, last_pts)) = self.frame_buffer.last() {
            let interval = pts_seconds - last_pts;
            self.min_interval = self.min_interval.min(interval);
            self.max_interval = self.max_interval.max(interval);
        }

        self.frame_buffer.push((frame, pts_seconds));

        // Limit buffer
        while self.frame_buffer.len() > 8 {
            self.frame_buffer.remove(0);
        }

        if self.frame_buffer.len() < 2 {
            return Ok(Vec::new());
        }

        let target_interval = 1.0 / self.target_fps.to_f64();
        let mut output = Vec::new();

        // Generate output frames at target rate
        let (_, first_pts) = &self.frame_buffer[self.frame_buffer.len() - 2];
        let (_, second_pts) = &self.frame_buffer[self.frame_buffer.len() - 1];

        let mut output_time = self.last_output_time + target_interval;
        while output_time <= *second_pts {
            // Calculate interpolation factor
            let t = ((output_time - first_pts) / (second_pts - first_pts)).clamp(0.0, 1.0) as f32;

            let len = self.frame_buffer.len();
            let interpolated = self.interpolator.interpolate(
                &self.frame_buffer[len - 2].0,
                &self.frame_buffer[len - 1].0,
                t,
            )?;
            output.push(interpolated);

            self.last_output_time = output_time;
            output_time += target_interval;
        }

        Ok(output)
    }

    /// Get estimated source frame rate range.
    pub fn estimated_fps_range(&self) -> (f64, f64) {
        if self.min_interval == f64::MAX || self.max_interval == 0.0 {
            return (0.0, 0.0);
        }
        (1.0 / self.max_interval, 1.0 / self.min_interval)
    }

    /// Reset the handler.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.last_output_time = 0.0;
        self.min_interval = f64::MAX;
        self.max_interval = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::{timestamp::TimeBase, PixelFormat};

    fn create_test_frame(value: u8) -> Frame {
        let mut frame = Frame::new(64, 64, PixelFormat::Yuv420p, TimeBase::MPEG);
        if let Some(plane) = frame.plane_mut(0) {
            plane.fill(value);
        }
        frame
    }

    #[test]
    fn test_standard_frame_rate_conversion() {
        let fps = StandardFrameRate::Ntsc30;
        let rational = fps.to_rational();
        assert_eq!(rational.num, 30000);
        assert_eq!(rational.den, 1001);

        let detected = StandardFrameRate::from_rational(rational);
        assert_eq!(detected, Some(StandardFrameRate::Ntsc30));
    }

    #[test]
    fn test_conversion_config_ratio() {
        let config = ConversionConfig {
            source_fps: Rational::new(24, 1),
            target_fps: Rational::new(30, 1),
            ..Default::default()
        };

        assert!((config.rate_ratio() - 1.25).abs() < 0.001);
        assert!(config.is_upconversion());
        assert!(!config.is_downconversion());
    }

    #[test]
    fn test_drop_duplicate_downconversion() {
        let config = ConversionConfig {
            source_fps: Rational::new(30, 1),
            target_fps: Rational::new(24, 1),
            method: ConversionMethod::DropDuplicate,
            ..Default::default()
        };

        let mut converter = FrameRateConverter::new(config);
        let mut total_output = 0;

        for i in 0..30 {
            let output = converter.process(&create_test_frame(i)).unwrap();
            total_output += output.len();
        }

        // 30 input frames at 30fps should produce ~24 output frames at 24fps
        assert!((total_output as i32 - 24).abs() <= 1);
    }

    #[test]
    fn test_drop_duplicate_upconversion() {
        let config = ConversionConfig {
            source_fps: Rational::new(24, 1),
            target_fps: Rational::new(30, 1),
            method: ConversionMethod::DropDuplicate,
            ..Default::default()
        };

        let mut converter = FrameRateConverter::new(config);
        let mut total_output = 0;

        for i in 0..24 {
            let output = converter.process(&create_test_frame(i)).unwrap();
            total_output += output.len();
        }

        // 24 input frames at 24fps should produce ~30 output frames at 30fps
        assert!((total_output as i32 - 30).abs() <= 1);
    }

    #[test]
    fn test_blend_conversion() {
        let config = ConversionConfig {
            source_fps: Rational::new(24, 1),
            target_fps: Rational::new(30, 1),
            method: ConversionMethod::Blend,
            ..Default::default()
        };

        let mut converter = FrameRateConverter::new(config);

        // Process several frames
        for i in 0..10 {
            let result = converter.process(&create_test_frame(i * 10));
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_telecine_conversion() {
        let config = ConversionConfig {
            source_fps: Rational::new(24, 1),
            target_fps: Rational::new(30, 1),
            method: ConversionMethod::Telecine,
            telecine_pattern: TelecinePattern::Pulldown32,
            ..Default::default()
        };

        let mut converter = FrameRateConverter::new(config);
        let mut total_output = 0;

        for i in 0..4 {
            let output = converter.process(&create_test_frame(i * 50)).unwrap();
            total_output += output.len();
        }

        // 4 input frames with 3:2 pulldown should produce 10 output frames (2+3+2+3)
        assert_eq!(total_output, 10);
    }

    #[test]
    fn test_converter_reset() {
        let config = ConversionConfig::default();
        let mut converter = FrameRateConverter::new(config);

        converter.process(&create_test_frame(0)).unwrap();
        converter.process(&create_test_frame(1)).unwrap();

        assert!(converter.source_frame_count() > 0);

        converter.reset();

        assert_eq!(converter.source_frame_count(), 0);
        assert_eq!(converter.output_frame_count(), 0);
    }

    #[test]
    fn test_suggest_conversion_method() {
        // 24 -> 30 should suggest telecine
        let method = suggest_conversion_method(Rational::new(24, 1), Rational::new(30, 1));
        assert_eq!(method, ConversionMethod::Telecine);

        // 30 -> 24 should suggest inverse telecine
        let method = suggest_conversion_method(Rational::new(30, 1), Rational::new(24, 1));
        assert_eq!(method, ConversionMethod::InverseTelecine);

        // Same rate should use drop/duplicate
        let method = suggest_conversion_method(Rational::new(30, 1), Rational::new(30, 1));
        assert_eq!(method, ConversionMethod::DropDuplicate);
    }

    #[test]
    fn test_vfr_handler() {
        let mut handler = VfrHandler::new(Rational::new(30, 1));

        // Simulate variable frame rate input
        let _result1 = handler.process(create_test_frame(0), 0.0);
        let _result2 = handler.process(create_test_frame(1), 0.04);
        let _result3 = handler.process(create_test_frame(2), 0.08);

        let (min_fps, max_fps) = handler.estimated_fps_range();
        assert!(min_fps > 0.0);
        assert!(max_fps >= min_fps);

        handler.reset();
        assert_eq!(handler.estimated_fps_range(), (0.0, 0.0));
    }

    #[test]
    fn test_motion_compensated_conversion() {
        let config = ConversionConfig {
            source_fps: Rational::new(24, 1),
            target_fps: Rational::new(60, 1),
            method: ConversionMethod::MotionCompensated,
            ..Default::default()
        };

        let mut converter = FrameRateConverter::new(config);

        for i in 0..5 {
            let result = converter.process(&create_test_frame(i * 20));
            assert!(result.is_ok());
        }

        let flushed = converter.flush();
        // Should have some frames
        assert!(converter.output_frame_count() > 0 || !flushed.is_empty());
    }

    #[test]
    fn test_standard_conversion_preset() {
        let converter =
            FrameRateConverter::standard(StandardFrameRate::NtscFilm, StandardFrameRate::Ntsc30);

        assert_eq!(
            converter.config().method,
            ConversionMethod::Telecine
        );
        assert_eq!(
            converter.config().telecine_pattern,
            TelecinePattern::Pulldown32
        );
    }
}
