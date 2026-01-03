//! Interlace detection and telecine analysis.
//!
//! This module provides tools for:
//! - Detecting whether content is interlaced or progressive
//! - Determining field order (TFF/BFF)
//! - Detecting telecine patterns (3:2 pulldown, 2:2 pulldown)
//! - Performing inverse telecine (IVTC)
//!
//! # Telecine Overview
//!
//! Telecine is the process of converting film (24fps) to video (29.97fps).
//! The most common pattern is 3:2 pulldown:
//!
//! ```text
//! Film frames:    A       B       C       D
//! Video fields:   Aa Ab   Ba Bb Bb'   Ca Cb   Da Db Db'
//! ```
//!
//! Where `Bb'` and `Db'` are repeated fields.
//!
//! # IVTC (Inverse Telecine)
//!
//! IVTC reverses the telecine process to recover the original film frames.
//! This is important for:
//! - Reducing file size (24fps vs 29.97fps)
//! - Avoiding deinterlacing artifacts
//! - Preserving original film motion

use crate::bob::FieldOrder;
use crate::error::{DeinterlaceError, Result};
use std::collections::VecDeque;
use transcode_core::frame::FrameFlags;
use transcode_core::{Frame, FrameBuffer, PixelFormat};

/// Content type detection result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Progressive (non-interlaced) content.
    Progressive,
    /// Interlaced content.
    Interlaced {
        /// Detected field order.
        field_order: FieldOrder,
    },
    /// Telecined content (film converted to video).
    Telecine {
        /// Type of telecine pattern.
        pattern: TelecinePattern,
        /// Detected field order.
        field_order: FieldOrder,
    },
    /// Mixed content (some interlaced, some progressive).
    Mixed,
    /// Unable to determine.
    Unknown,
}

/// Telecine pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelecinePattern {
    /// 3:2 pulldown (24fps -> 29.97fps).
    Pulldown32,
    /// 2:2 pulldown / PAL speedup (24fps -> 25fps).
    Pulldown22,
    /// 2:3 pulldown (reverse of 3:2).
    Pulldown23,
    /// Variable pattern.
    Variable,
}

/// Interlace detection configuration.
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Minimum number of frames to analyze.
    pub min_frames: usize,
    /// Maximum number of frames to analyze.
    pub max_frames: usize,
    /// Comb detection threshold (higher = less sensitive).
    pub comb_threshold: u32,
    /// Minimum interlaced ratio to classify as interlaced.
    pub interlace_ratio_threshold: f32,
    /// Enable telecine pattern detection.
    pub detect_telecine: bool,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            min_frames: 10,
            max_frames: 100,
            comb_threshold: 30,
            interlace_ratio_threshold: 0.5,
            detect_telecine: true,
        }
    }
}

/// Frame interlace analysis result.
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    /// Comb score (higher = more likely interlaced).
    pub comb_score: f32,
    /// Is this frame likely interlaced?
    pub is_interlaced: bool,
    /// Detected field order (if interlaced).
    pub field_order: Option<FieldOrder>,
    /// Top field comb score.
    pub top_field_score: f32,
    /// Bottom field comb score.
    pub bottom_field_score: f32,
}

/// Interlace detector.
///
/// Analyzes video frames to detect interlacing and telecine patterns.
pub struct InterlaceDetector {
    config: DetectionConfig,
    /// Frame analysis history.
    analyses: VecDeque<FrameAnalysis>,
    /// Field difference history for telecine detection.
    field_diffs: VecDeque<f32>,
    /// Frame count.
    frame_count: usize,
}

impl InterlaceDetector {
    /// Create a new interlace detector.
    pub fn new() -> Self {
        Self {
            config: DetectionConfig::default(),
            analyses: VecDeque::new(),
            field_diffs: VecDeque::new(),
            frame_count: 0,
        }
    }

    /// Create a new detector with custom configuration.
    pub fn with_config(config: DetectionConfig) -> Self {
        Self {
            config,
            analyses: VecDeque::new(),
            field_diffs: VecDeque::new(),
            frame_count: 0,
        }
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.analyses.clear();
        self.field_diffs.clear();
        self.frame_count = 0;
    }

    /// Analyze a single frame.
    pub fn analyze_frame(&mut self, frame: &Frame) -> Result<FrameAnalysis> {
        let analysis = self.compute_frame_analysis(frame)?;

        self.analyses.push_back(analysis.clone());
        if self.analyses.len() > self.config.max_frames {
            self.analyses.pop_front();
        }

        self.frame_count += 1;

        Ok(analysis)
    }

    /// Get the current content type detection result.
    pub fn get_content_type(&self) -> ContentType {
        if self.analyses.len() < self.config.min_frames {
            return ContentType::Unknown;
        }

        let interlaced_count = self.analyses.iter().filter(|a| a.is_interlaced).count();
        let total = self.analyses.len();
        let interlace_ratio = interlaced_count as f32 / total as f32;

        if interlace_ratio < 0.1 {
            return ContentType::Progressive;
        }

        if interlace_ratio > self.config.interlace_ratio_threshold {
            // Determine field order
            let tff_count = self
                .analyses
                .iter()
                .filter(|a| a.field_order == Some(FieldOrder::TopFieldFirst))
                .count();
            let field_order = if tff_count > total / 2 {
                FieldOrder::TopFieldFirst
            } else {
                FieldOrder::BottomFieldFirst
            };

            // Check for telecine pattern
            if self.config.detect_telecine {
                if let Some(pattern) = self.detect_telecine_pattern() {
                    return ContentType::Telecine {
                        pattern,
                        field_order,
                    };
                }
            }

            return ContentType::Interlaced { field_order };
        }

        ContentType::Mixed
    }

    /// Compute analysis for a single frame.
    fn compute_frame_analysis(&self, frame: &Frame) -> Result<FrameAnalysis> {
        let format = frame.format();
        if !matches!(
            format,
            PixelFormat::Yuv420p
                | PixelFormat::Yuv422p
                | PixelFormat::Yuv444p
                | PixelFormat::Gray8
        ) {
            return Err(DeinterlaceError::unsupported_format(format.to_string()));
        }

        let y_plane = frame
            .plane(0)
            .ok_or_else(|| DeinterlaceError::buffer_error("Y plane not found"))?;
        let stride = frame.stride(0);
        let width = frame.width() as usize;
        let height = frame.height() as usize;

        if height < 4 {
            return Err(DeinterlaceError::invalid_dimensions(
                frame.width(),
                frame.height(),
            ));
        }

        // Compute comb scores for each field
        let top_field_score = self.compute_field_comb(y_plane, stride, width, height, true);
        let bottom_field_score = self.compute_field_comb(y_plane, stride, width, height, false);

        let comb_score = (top_field_score + bottom_field_score) / 2.0;
        let threshold = self.config.comb_threshold as f32;

        let is_interlaced = comb_score > threshold;

        let field_order = if is_interlaced {
            // Field with higher comb score came first (motion between fields)
            if top_field_score > bottom_field_score {
                Some(FieldOrder::BottomFieldFirst)
            } else {
                Some(FieldOrder::TopFieldFirst)
            }
        } else {
            None
        };

        Ok(FrameAnalysis {
            comb_score,
            is_interlaced,
            field_order,
            top_field_score,
            bottom_field_score,
        })
    }

    /// Compute comb score for a field.
    fn compute_field_comb(
        &self,
        plane: &[u8],
        stride: usize,
        width: usize,
        height: usize,
        top_field: bool,
    ) -> f32 {
        let mut total_diff: u64 = 0;
        let mut count: u64 = 0;

        let start_y = if top_field { 1 } else { 2 };

        // Sample every other line pair to detect comb artifacts
        for y in (start_y..height - 1).step_by(2) {
            let above_offset = (y - 1) * stride;
            let current_offset = y * stride;
            let below_offset = (y + 1) * stride;

            for x in 1..width - 1 {
                if above_offset + x >= plane.len()
                    || current_offset + x >= plane.len()
                    || below_offset + x >= plane.len()
                {
                    continue;
                }

                let above = plane[above_offset + x] as i32;
                let current = plane[current_offset + x] as i32;
                let below = plane[below_offset + x] as i32;

                // Comb detection: look for lines that differ significantly from both neighbors
                let diff_above = (current - above).abs();
                let diff_below = (current - below).abs();

                // If current line differs from both neighbors in the same direction,
                // this is likely a comb artifact
                if (current > above && current > below) || (current < above && current < below) {
                    total_diff += (diff_above.min(diff_below)) as u64;
                }

                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total_diff as f32 / count as f32
        }
    }

    /// Detect telecine pattern from field differences.
    fn detect_telecine_pattern(&self) -> Option<TelecinePattern> {
        if self.analyses.len() < 10 {
            return None;
        }

        // Look for repeating pattern in comb scores
        let scores: Vec<f32> = self.analyses.iter().map(|a| a.comb_score).collect();

        // Check for 3:2 pulldown pattern (period of 5)
        if self.check_pattern(&scores, 5) {
            return Some(TelecinePattern::Pulldown32);
        }

        // Check for 2:2 pulldown pattern (period of 2)
        if self.check_pattern(&scores, 2) {
            return Some(TelecinePattern::Pulldown22);
        }

        None
    }

    /// Check if scores follow a repeating pattern.
    fn check_pattern(&self, scores: &[f32], period: usize) -> bool {
        if scores.len() < period * 3 {
            return false;
        }

        let threshold = self.config.comb_threshold as f32;

        // Count high/low pattern matches
        let mut matches = 0;
        let mut total = 0;

        for i in period..scores.len() {
            let current_high = scores[i] > threshold;
            let period_ago_high = scores[i - period] > threshold;

            if current_high == period_ago_high {
                matches += 1;
            }
            total += 1;
        }

        // Pattern should match at least 70% of the time
        total > 0 && (matches as f32 / total as f32) > 0.7
    }
}

impl Default for InterlaceDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Inverse telecine processor.
///
/// Removes 3:2 pulldown and reconstructs original film frames.
pub struct InverseTelecine {
    /// Frame buffer for pattern matching.
    frame_buffer: VecDeque<Frame>,
    /// Detected pattern phase.
    pattern_phase: usize,
    /// Pattern type being processed.
    pattern: TelecinePattern,
    /// Field order.
    field_order: FieldOrder,
}

impl InverseTelecine {
    /// Create a new inverse telecine processor.
    pub fn new(pattern: TelecinePattern, field_order: FieldOrder) -> Self {
        Self {
            frame_buffer: VecDeque::with_capacity(5),
            pattern_phase: 0,
            pattern,
            field_order,
        }
    }

    /// Reset the processor.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.pattern_phase = 0;
    }

    /// Push a frame and get output frames.
    pub fn push_frame(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        self.frame_buffer.push_back(frame);

        match self.pattern {
            TelecinePattern::Pulldown32 => self.process_32_pulldown(),
            TelecinePattern::Pulldown22 => self.process_22_pulldown(),
            TelecinePattern::Pulldown23 => self.process_23_pulldown(),
            TelecinePattern::Variable => {
                // For variable, just pass through
                Ok(self.frame_buffer.drain(..).collect())
            }
        }
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> Vec<Frame> {
        self.frame_buffer.drain(..).collect()
    }

    /// Process 3:2 pulldown pattern.
    ///
    /// Input pattern (5 video frames):  AA AB BB BC CC
    /// Output pattern (4 film frames):  A  B  B  C
    fn process_32_pulldown(&mut self) -> Result<Vec<Frame>> {
        let mut output = Vec::new();

        // Need 5 frames to process one cycle
        while self.frame_buffer.len() >= 5 {
            let frames: Vec<Frame> = self.frame_buffer.drain(..5).collect();

            // In 3:2 pulldown, frames 0, 2, 4 contain unique content
            // Frames 1, 3 contain mixed fields

            // Output frame 0 (pure A frame)
            output.push(frames[0].clone());

            // Output frame 1 (reconstruct B from frame 1 top + frame 2 bottom, or vice versa)
            output.push(self.reconstruct_frame(&frames[1], &frames[2])?);

            // Output frame 2 (pure B or C)
            output.push(frames[2].clone());

            // Output frame 3 (reconstruct from frames 3 and 4)
            output.push(self.reconstruct_frame(&frames[3], &frames[4])?);
        }

        Ok(output)
    }

    /// Process 2:2 pulldown pattern.
    fn process_22_pulldown(&mut self) -> Result<Vec<Frame>> {
        let mut output = Vec::new();

        // 2:2 pulldown just duplicates frames, so every other frame is unique
        while self.frame_buffer.len() >= 2 {
            let frame = self.frame_buffer.pop_front().unwrap();
            output.push(frame);
            // Skip the duplicate
            self.frame_buffer.pop_front();
        }

        Ok(output)
    }

    /// Process 2:3 pulldown pattern.
    fn process_23_pulldown(&mut self) -> Result<Vec<Frame>> {
        // 2:3 is similar to 3:2 but with different phase
        self.process_32_pulldown()
    }

    /// Reconstruct a frame from two interlaced frames.
    fn reconstruct_frame(&self, frame_a: &Frame, frame_b: &Frame) -> Result<Frame> {
        let width = frame_a.width();
        let height = frame_a.height();
        let format = frame_a.format();

        let mut output = Frame::from_buffer(FrameBuffer::new(width, height, format));
        output.pts = frame_a.pts;
        output.dts = frame_a.dts;
        output.duration = frame_a.duration;
        output.poc = frame_a.poc;
        output.flags = frame_a.flags & !FrameFlags::INTERLACED & !FrameFlags::TOP_FIELD_FIRST;

        let num_planes = format.num_planes();
        let (hsub, vsub) = format.chroma_subsampling();

        for plane_idx in 0..num_planes {
            let src_a = frame_a
                .plane(plane_idx)
                .ok_or_else(|| DeinterlaceError::buffer_error("Source A plane not found"))?;
            let src_b = frame_b
                .plane(plane_idx)
                .ok_or_else(|| DeinterlaceError::buffer_error("Source B plane not found"))?;

            let stride_a = frame_a.stride(plane_idx);
            let stride_b = frame_b.stride(plane_idx);
            let dst_stride = output.stride(plane_idx);

            let dst = output
                .plane_mut(plane_idx)
                .ok_or_else(|| DeinterlaceError::buffer_error("Dest plane not found"))?;

            let plane_height = if plane_idx == 0 {
                height as usize
            } else {
                height as usize / vsub as usize
            };

            let plane_width = if plane_idx == 0 {
                width as usize
            } else {
                width as usize / hsub as usize
            };

            let use_a_for_even = matches!(self.field_order, FieldOrder::TopFieldFirst);

            for y in 0..plane_height {
                let (src, src_stride) = if (y % 2 == 0) == use_a_for_even {
                    (src_a, stride_a)
                } else {
                    (src_b, stride_b)
                };

                let src_offset = y * src_stride;
                let dst_offset = y * dst_stride;

                if src_offset + plane_width <= src.len() && dst_offset + plane_width <= dst.len() {
                    dst[dst_offset..dst_offset + plane_width]
                        .copy_from_slice(&src[src_offset..src_offset + plane_width]);
                }
            }
        }

        Ok(output)
    }
}

/// Field order detector.
///
/// Determines the field order by analyzing motion between fields.
pub struct FieldOrderDetector {
    /// Top field first confidence.
    tff_score: f32,
    /// Bottom field first confidence.
    bff_score: f32,
    /// Number of frames analyzed.
    frame_count: usize,
}

impl FieldOrderDetector {
    /// Create a new field order detector.
    pub fn new() -> Self {
        Self {
            tff_score: 0.0,
            bff_score: 0.0,
            frame_count: 0,
        }
    }

    /// Analyze a frame pair to determine field order.
    pub fn analyze(&mut self, prev: &Frame, cur: &Frame) -> FieldOrder {
        let format = cur.format();
        if !matches!(
            format,
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p
        ) {
            return FieldOrder::TopFieldFirst;
        }

        let prev_y = prev.plane(0);
        let cur_y = cur.plane(0);

        if prev_y.is_none() || cur_y.is_none() {
            return FieldOrder::TopFieldFirst;
        }

        let prev_y = prev_y.unwrap();
        let cur_y = cur_y.unwrap();
        let stride = cur.stride(0);
        let width = cur.width() as usize;
        let height = cur.height() as usize;

        // Calculate motion correlation for TFF and BFF hypotheses
        let tff_motion = self.calculate_field_motion(prev_y, cur_y, stride, width, height, true);
        let bff_motion = self.calculate_field_motion(prev_y, cur_y, stride, width, height, false);

        // Lower motion means the hypothesis is correct (fields align properly)
        if tff_motion < bff_motion {
            self.tff_score += 1.0;
        } else {
            self.bff_score += 1.0;
        }

        self.frame_count += 1;

        self.get_result()
    }

    /// Get the current detection result.
    pub fn get_result(&self) -> FieldOrder {
        if self.tff_score >= self.bff_score {
            FieldOrder::TopFieldFirst
        } else {
            FieldOrder::BottomFieldFirst
        }
    }

    /// Get confidence level (0.0 to 1.0).
    pub fn confidence(&self) -> f32 {
        if self.frame_count == 0 {
            return 0.0;
        }
        let total = self.tff_score + self.bff_score;
        if total == 0.0 {
            return 0.5;
        }
        (self.tff_score.max(self.bff_score) / total).min(1.0)
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.tff_score = 0.0;
        self.bff_score = 0.0;
        self.frame_count = 0;
    }

    /// Calculate field motion assuming a specific field order.
    fn calculate_field_motion(
        &self,
        prev: &[u8],
        cur: &[u8],
        stride: usize,
        width: usize,
        height: usize,
        tff: bool,
    ) -> f32 {
        let mut motion: u64 = 0;
        let mut count: u64 = 0;

        // Compare fields that should match if the hypothesis is correct
        let first_field_start = if tff { 0 } else { 1 };

        for y in (first_field_start..height - 1).step_by(2) {
            let prev_offset = y * stride;
            let cur_offset = y * stride;

            for x in 0..width {
                if prev_offset + x < prev.len() && cur_offset + x < cur.len() {
                    let diff = (prev[prev_offset + x] as i32 - cur[cur_offset + x] as i32).abs();
                    motion += diff as u64;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            motion as f32 / count as f32
        }
    }
}

impl Default for FieldOrderDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::TimeBase;

    fn create_interlaced_frame(width: u32, height: u32, pattern: u8) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);
        frame.flags = FrameFlags::INTERLACED | FrameFlags::TOP_FIELD_FIRST;

        let stride = frame.stride(0);
        if let Some(y_plane) = frame.plane_mut(0) {
            for y in 0..height as usize {
                let value = if y % 2 == 0 {
                    pattern
                } else {
                    255 - pattern
                };
                for x in 0..width as usize {
                    y_plane[y * stride + x] = value;
                }
            }
        }

        frame
    }

    fn create_progressive_frame(width: u32, height: u32, value: u8) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);

        if let Some(y_plane) = frame.plane_mut(0) {
            y_plane.fill(value);
        }

        frame
    }

    #[test]
    fn test_interlace_detection_interlaced() {
        let mut detector = InterlaceDetector::new();

        for i in 0..15 {
            let frame = create_interlaced_frame(64, 64, 50 + i * 10);
            detector.analyze_frame(&frame).unwrap();
        }

        let result = detector.get_content_type();
        assert!(matches!(result, ContentType::Interlaced { .. }));
    }

    #[test]
    fn test_interlace_detection_progressive() {
        let mut detector = InterlaceDetector::new();

        for i in 0..15 {
            let frame = create_progressive_frame(64, 64, 50 + i * 10);
            detector.analyze_frame(&frame).unwrap();
        }

        let result = detector.get_content_type();
        assert_eq!(result, ContentType::Progressive);
    }

    #[test]
    fn test_frame_analysis() {
        let detector = InterlaceDetector::new();

        let interlaced = create_interlaced_frame(64, 64, 100);
        let analysis = detector.compute_frame_analysis(&interlaced).unwrap();
        assert!(analysis.comb_score > 0.0);
    }

    #[test]
    fn test_field_order_detector() {
        let mut detector = FieldOrderDetector::new();

        let frame1 = create_progressive_frame(64, 64, 100);
        let frame2 = create_progressive_frame(64, 64, 110);

        let result = detector.analyze(&frame1, &frame2);
        assert!(matches!(
            result,
            FieldOrder::TopFieldFirst | FieldOrder::BottomFieldFirst
        ));
    }

    #[test]
    fn test_ivtc_32_pulldown() {
        let mut ivtc = InverseTelecine::new(TelecinePattern::Pulldown32, FieldOrder::TopFieldFirst);

        // Create 5 frames (one cycle of 3:2 pulldown)
        let mut all_output = Vec::new();
        for i in 0..5 {
            let frame = create_progressive_frame(64, 64, 50 + i * 20);
            let output = ivtc.push_frame(frame).unwrap();
            all_output.extend(output);
        }

        // Flush any remaining frames
        let flushed = ivtc.flush();
        all_output.extend(flushed);

        // After processing 5 input frames, we should get 4 output frames (3:2 pulldown conversion)
        assert!(!all_output.is_empty());
        // 3:2 pulldown converts 5 video frames to 4 film frames
        assert_eq!(all_output.len(), 4);
    }

    #[test]
    fn test_detector_reset() {
        let mut detector = InterlaceDetector::new();

        for _ in 0..5 {
            let frame = create_interlaced_frame(64, 64, 100);
            detector.analyze_frame(&frame).unwrap();
        }

        detector.reset();
        assert_eq!(detector.analyses.len(), 0);
        assert_eq!(detector.frame_count, 0);
    }

    #[test]
    fn test_content_type_unknown_few_frames() {
        let mut detector = InterlaceDetector::new();

        // Only analyze 2 frames (below minimum)
        for _ in 0..2 {
            let frame = create_interlaced_frame(64, 64, 100);
            detector.analyze_frame(&frame).unwrap();
        }

        assert_eq!(detector.get_content_type(), ContentType::Unknown);
    }
}
