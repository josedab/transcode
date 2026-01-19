//! Telecine and pulldown pattern handling.
//!
//! This module provides support for telecine patterns used to convert
//! film content (24 fps) to video frame rates (25/30/50/60 fps).

use crate::error::{FrameRateError, Result};
use transcode_core::{Frame, Rational};

/// Telecine/pulldown pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum TelecinePattern {
    /// 3:2 pulldown (24 -> 30 fps, also 24 -> 60i).
    /// Pattern: 2-3-2-3-2-3-2-3 (A A B B B C C D D D)
    Pulldown32,
    /// 2:3 pulldown (variant of 3:2).
    /// Pattern: 3-2-3-2-3-2-3-2 (A A A B B C C C D D)
    Pulldown23,
    /// 2:2 pulldown (24 -> 48/50 fps).
    /// Each frame shown twice.
    Pulldown22,
    /// 2:2:2:4 pulldown (25 -> 30 fps, or 24 -> 30 fps euro variant).
    Pulldown2224,
    /// 3:2:3:2:2 pulldown (24 -> 25 fps, euro pulldown).
    /// Also known as 2:2:2:2:2:2:2:2:2:2:2:3 (12-frame pattern).
    EuroPulldown,
    /// No telecine pattern.
    #[default]
    None,
}


impl TelecinePattern {
    /// Get the source frame rate for this pattern.
    pub fn source_fps(&self) -> Rational {
        match self {
            Self::Pulldown32 | Self::Pulldown23 | Self::Pulldown22 => Rational::new(24000, 1001),
            Self::Pulldown2224 => Rational::new(25, 1),
            Self::EuroPulldown => Rational::new(24, 1),
            Self::None => Rational::new(24, 1),
        }
    }

    /// Get the target frame rate for this pattern.
    pub fn target_fps(&self) -> Rational {
        match self {
            Self::Pulldown32 | Self::Pulldown23 => Rational::new(30000, 1001),
            Self::Pulldown22 => Rational::new(48000, 1001),
            Self::Pulldown2224 => Rational::new(30, 1),
            Self::EuroPulldown => Rational::new(25, 1),
            Self::None => Rational::new(24, 1),
        }
    }

    /// Get the pattern length in output frames.
    pub fn pattern_length(&self) -> usize {
        match self {
            Self::Pulldown32 | Self::Pulldown23 => 5,
            Self::Pulldown22 => 2,
            Self::Pulldown2224 => 5,
            Self::EuroPulldown => 25,
            Self::None => 1,
        }
    }

    /// Get the number of source frames in one pattern cycle.
    pub fn source_frames_per_cycle(&self) -> usize {
        match self {
            Self::Pulldown32 | Self::Pulldown23 => 4,
            Self::Pulldown22 => 1,
            Self::Pulldown2224 => 4,
            Self::EuroPulldown => 24,
            Self::None => 1,
        }
    }
}

/// Telecine applier for converting film to video frame rates.
pub struct TelecineApplier {
    pattern: TelecinePattern,
    /// Current position in pattern.
    pattern_pos: usize,
    /// Source frame index.
    source_index: usize,
    /// Pending frames buffer.
    pending: Vec<Frame>,
}

impl TelecineApplier {
    /// Create a new telecine applier.
    pub fn new(pattern: TelecinePattern) -> Self {
        Self {
            pattern,
            pattern_pos: 0,
            source_index: 0,
            pending: Vec::new(),
        }
    }

    /// Reset the telecine state.
    pub fn reset(&mut self) {
        self.pattern_pos = 0;
        self.source_index = 0;
        self.pending.clear();
    }

    /// Get the current telecine pattern.
    pub fn pattern(&self) -> TelecinePattern {
        self.pattern
    }

    /// Process a source frame and return telecined output frames.
    pub fn process(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        match self.pattern {
            TelecinePattern::Pulldown32 => self.apply_32_pulldown(frame),
            TelecinePattern::Pulldown23 => self.apply_23_pulldown(frame),
            TelecinePattern::Pulldown22 => self.apply_22_pulldown(frame),
            TelecinePattern::Pulldown2224 => self.apply_2224_pulldown(frame),
            TelecinePattern::EuroPulldown => self.apply_euro_pulldown(frame),
            TelecinePattern::None => Ok(vec![frame.clone()]),
        }
    }

    /// Apply 3:2 pulldown pattern.
    fn apply_32_pulldown(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        // Pattern: A A B B B C C D D D (repeating)
        // Frame 0: 2 copies, Frame 1: 3 copies, Frame 2: 2 copies, Frame 3: 3 copies
        let copies = match self.source_index % 4 {
            0 | 2 => 2,
            1 | 3 => 3,
            _ => unreachable!(),
        };

        self.source_index += 1;
        Ok(vec![frame.clone(); copies])
    }

    /// Apply 2:3 pulldown pattern.
    fn apply_23_pulldown(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        // Pattern: A A A B B C C C D D (repeating)
        let copies = match self.source_index % 4 {
            0 | 2 => 3,
            1 | 3 => 2,
            _ => unreachable!(),
        };

        self.source_index += 1;
        Ok(vec![frame.clone(); copies])
    }

    /// Apply 2:2 pulldown pattern.
    fn apply_22_pulldown(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        self.source_index += 1;
        Ok(vec![frame.clone(), frame.clone()])
    }

    /// Apply 2:2:2:4 pulldown pattern (25 -> 30 fps).
    fn apply_2224_pulldown(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        // Pattern over 4 source frames: 2, 2, 2, 4 copies = 10 output frames
        // But for 25->30, we use: skip every 5th output
        let copies = match self.source_index % 5 {
            0..=3 => 1,
            4 => 2, // Extra frame for 5th source frame
            _ => unreachable!(),
        };

        self.source_index += 1;
        Ok(vec![frame.clone(); copies])
    }

    /// Apply Euro pulldown pattern (24 -> 25 fps).
    fn apply_euro_pulldown(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        // Speed up by 4.1667% - every 24th source frame maps to 25 output frames
        // Simple approach: drop every 25th source frame
        self.source_index += 1;

        // Every 24 input frames produce 25 output frames
        // We duplicate one frame in every 24
        if self.source_index % 24 == 12 {
            Ok(vec![frame.clone(), frame.clone()])
        } else {
            Ok(vec![frame.clone()])
        }
    }
}

/// Inverse telecine for recovering original film frames.
pub struct InverseTelecine {
    pattern: TelecinePattern,
    /// Frame buffer for pattern detection and recovery.
    buffer: Vec<Frame>,
    /// Detected pattern phase.
    phase: usize,
    /// Pattern detection confidence.
    confidence: f32,
}

impl InverseTelecine {
    /// Create a new inverse telecine processor.
    pub fn new() -> Self {
        Self {
            pattern: TelecinePattern::None,
            buffer: Vec::new(),
            phase: 0,
            confidence: 0.0,
        }
    }

    /// Create with a known pattern.
    pub fn with_pattern(pattern: TelecinePattern) -> Self {
        Self {
            pattern,
            buffer: Vec::new(),
            phase: 0,
            confidence: 1.0,
        }
    }

    /// Reset the inverse telecine state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.phase = 0;
        self.confidence = 0.0;
    }

    /// Get the detected pattern.
    pub fn detected_pattern(&self) -> TelecinePattern {
        self.pattern
    }

    /// Get detection confidence (0.0 - 1.0).
    pub fn detection_confidence(&self) -> f32 {
        self.confidence
    }

    /// Process an input frame and return recovered film frames.
    pub fn process(&mut self, frame: &Frame) -> Result<Vec<Frame>> {
        self.buffer.push(frame.clone());

        // Need at least 5 frames for 3:2 pattern detection
        if self.buffer.len() < 5 {
            return Ok(Vec::new());
        }

        // Detect pattern if not already known
        if self.pattern == TelecinePattern::None {
            self.detect_pattern()?;
        }

        // Apply inverse telecine based on pattern
        match self.pattern {
            TelecinePattern::Pulldown32 | TelecinePattern::Pulldown23 => {
                self.inverse_32_pulldown()
            }
            TelecinePattern::Pulldown22 => self.inverse_22_pulldown(),
            TelecinePattern::EuroPulldown => self.inverse_euro_pulldown(),
            _ => {
                // No pattern - pass through
                let frame = self.buffer.remove(0);
                Ok(vec![frame])
            }
        }
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> Vec<Frame> {
        std::mem::take(&mut self.buffer)
    }

    /// Detect telecine pattern from buffered frames.
    fn detect_pattern(&mut self) -> Result<()> {
        if self.buffer.len() < 5 {
            return Err(FrameRateError::InsufficientFrames {
                needed: 5,
                available: self.buffer.len(),
            });
        }

        // Calculate frame differences
        let mut diffs = Vec::new();
        for i in 1..self.buffer.len() {
            let diff = self.frame_difference(&self.buffer[i - 1], &self.buffer[i]);
            diffs.push(diff);
        }

        // Look for 3:2 pattern (alternating similar/different frames)
        // In 3:2 pulldown, we expect pairs of identical frames
        let threshold = 0.02; // 2% difference threshold
        let mut pattern_32_score = 0.0;
        let mut pattern_22_score = 0.0;

        // Check for 3:2 pattern (low, low, high, low, high pattern)
        for (i, &diff) in diffs.iter().enumerate() {
            let expected_low = matches!(i % 5, 0 | 2);
            if (diff < threshold) == expected_low {
                pattern_32_score += 1.0;
            }
        }
        pattern_32_score /= diffs.len() as f32;

        // Check for 2:2 pattern (alternating low/high)
        for (i, &diff) in diffs.iter().enumerate() {
            if (i % 2 == 0 && diff < threshold) || (i % 2 == 1 && diff >= threshold) {
                pattern_22_score += 1.0;
            }
        }
        pattern_22_score /= diffs.len() as f32;

        // Select best pattern
        if pattern_32_score > 0.7 && pattern_32_score > pattern_22_score {
            self.pattern = TelecinePattern::Pulldown32;
            self.confidence = pattern_32_score;
        } else if pattern_22_score > 0.7 {
            self.pattern = TelecinePattern::Pulldown22;
            self.confidence = pattern_22_score;
        }

        Ok(())
    }

    /// Calculate normalized difference between two frames.
    fn frame_difference(&self, frame1: &Frame, frame2: &Frame) -> f32 {
        let plane1 = match frame1.plane(0) {
            Some(p) => p,
            None => return 1.0,
        };
        let plane2 = match frame2.plane(0) {
            Some(p) => p,
            None => return 1.0,
        };

        let len = plane1.len().min(plane2.len());
        if len == 0 {
            return 1.0;
        }

        let mut diff_sum = 0u64;
        for i in 0..len {
            diff_sum += (plane1[i] as i32 - plane2[i] as i32).unsigned_abs() as u64;
        }

        diff_sum as f32 / (len as f32 * 255.0)
    }

    /// Inverse 3:2 pulldown.
    fn inverse_32_pulldown(&mut self) -> Result<Vec<Frame>> {
        // 3:2 pattern: A A B B B -> extract A, B
        // Need 5 input frames to produce 4 output frames
        if self.buffer.len() < 5 {
            return Ok(Vec::new());
        }

        let mut output = Vec::new();

        // Based on phase, extract unique frames
        // Phase 0: frames 0, 2, 3, 4 are unique
        // We pick the first of each pair of duplicates
        match self.phase {
            0 => {
                output.push(self.buffer.remove(0)); // Frame 0 (first of A A)
                self.buffer.remove(0); // Skip duplicate
                output.push(self.buffer.remove(0)); // Frame 2 (first of B B B)
                self.buffer.remove(0); // Skip duplicate
                self.buffer.remove(0); // Skip duplicate
            }
            _ => {
                // Simplified - just output first frame
                output.push(self.buffer.remove(0));
            }
        }

        self.phase = (self.phase + output.len()) % 4;
        Ok(output)
    }

    /// Inverse 2:2 pulldown.
    fn inverse_22_pulldown(&mut self) -> Result<Vec<Frame>> {
        // 2:2 pattern: A A B B -> extract A, B
        if self.buffer.len() < 2 {
            return Ok(Vec::new());
        }

        let frame = self.buffer.remove(0);
        self.buffer.remove(0); // Skip duplicate

        Ok(vec![frame])
    }

    /// Inverse Euro pulldown (25 -> 24 fps).
    fn inverse_euro_pulldown(&mut self) -> Result<Vec<Frame>> {
        // Euro pulldown: 25 output frames from 24 source
        // We need to detect and remove the duplicated frame
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        // Simple approach: every 25th frame is a duplicate
        self.phase += 1;
        let frame = self.buffer.remove(0);

        if self.phase % 25 == 0 {
            // Skip this frame (it's the duplicate)
            Ok(Vec::new())
        } else {
            Ok(vec![frame])
        }
    }
}

impl Default for InverseTelecine {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame rate info for telecine detection.
#[derive(Debug, Clone)]
pub struct TelecineInfo {
    /// Detected pattern.
    pub pattern: TelecinePattern,
    /// Detection confidence.
    pub confidence: f32,
    /// Estimated original frame rate.
    pub original_fps: Rational,
    /// Current frame rate.
    pub current_fps: Rational,
    /// Phase offset in pattern.
    pub phase: usize,
}

/// Analyze frames for telecine pattern.
pub fn detect_telecine(frames: &[Frame]) -> Result<TelecineInfo> {
    if frames.len() < 10 {
        return Err(FrameRateError::InsufficientFrames {
            needed: 10,
            available: frames.len(),
        });
    }

    let mut ivtc = InverseTelecine::new();
    for frame in frames.iter().take(10) {
        let _ = ivtc.process(frame);
    }

    Ok(TelecineInfo {
        pattern: ivtc.detected_pattern(),
        confidence: ivtc.detection_confidence(),
        original_fps: ivtc.detected_pattern().source_fps(),
        current_fps: ivtc.detected_pattern().target_fps(),
        phase: ivtc.phase,
    })
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
    fn test_telecine_pattern_properties() {
        let pattern = TelecinePattern::Pulldown32;
        assert_eq!(pattern.pattern_length(), 5);
        assert_eq!(pattern.source_frames_per_cycle(), 4);

        let pattern = TelecinePattern::Pulldown22;
        assert_eq!(pattern.pattern_length(), 2);
        assert_eq!(pattern.source_frames_per_cycle(), 1);
    }

    #[test]
    fn test_32_pulldown() {
        let mut applier = TelecineApplier::new(TelecinePattern::Pulldown32);

        // Frame 0: 2 copies
        let result = applier.process(&create_test_frame(0)).unwrap();
        assert_eq!(result.len(), 2);

        // Frame 1: 3 copies
        let result = applier.process(&create_test_frame(1)).unwrap();
        assert_eq!(result.len(), 3);

        // Frame 2: 2 copies
        let result = applier.process(&create_test_frame(2)).unwrap();
        assert_eq!(result.len(), 2);

        // Frame 3: 3 copies
        let result = applier.process(&create_test_frame(3)).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_22_pulldown() {
        let mut applier = TelecineApplier::new(TelecinePattern::Pulldown22);

        let result = applier.process(&create_test_frame(0)).unwrap();
        assert_eq!(result.len(), 2);

        let result = applier.process(&create_test_frame(1)).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_inverse_telecine_creation() {
        let ivtc = InverseTelecine::new();
        assert_eq!(ivtc.detected_pattern(), TelecinePattern::None);
        assert_eq!(ivtc.detection_confidence(), 0.0);
    }

    #[test]
    fn test_inverse_telecine_with_pattern() {
        let ivtc = InverseTelecine::with_pattern(TelecinePattern::Pulldown32);
        assert_eq!(ivtc.detected_pattern(), TelecinePattern::Pulldown32);
        assert_eq!(ivtc.detection_confidence(), 1.0);
    }

    #[test]
    fn test_inverse_22_pulldown() {
        let mut ivtc = InverseTelecine::with_pattern(TelecinePattern::Pulldown22);

        // Create pairs of identical frames
        let frame1 = create_test_frame(100);
        let frame2 = create_test_frame(100);
        let frame3 = create_test_frame(200);
        let frame4 = create_test_frame(200);

        // Need to push enough frames to start processing
        ivtc.buffer.push(frame1);
        ivtc.buffer.push(frame2);
        ivtc.buffer.push(frame3);
        ivtc.buffer.push(frame4);

        // Minimum frames reached, now process
        ivtc.buffer.push(create_test_frame(100));

        let result = ivtc.inverse_22_pulldown().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].plane(0).unwrap()[0], 100);
    }

    #[test]
    fn test_telecine_applier_reset() {
        let mut applier = TelecineApplier::new(TelecinePattern::Pulldown32);

        applier.process(&create_test_frame(0)).unwrap();
        applier.process(&create_test_frame(1)).unwrap();

        applier.reset();
        assert_eq!(applier.source_index, 0);
        assert_eq!(applier.pattern_pos, 0);

        // Should start pattern from beginning
        let result = applier.process(&create_test_frame(0)).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_frame_difference() {
        let ivtc = InverseTelecine::new();

        let frame1 = create_test_frame(100);
        let frame2 = create_test_frame(100);
        let diff = ivtc.frame_difference(&frame1, &frame2);
        assert!(diff < 0.01); // Should be nearly identical

        let frame3 = create_test_frame(200);
        let diff = ivtc.frame_difference(&frame1, &frame3);
        assert!(diff > 0.3); // Should be significantly different
    }

    #[test]
    fn test_euro_pulldown() {
        let mut applier = TelecineApplier::new(TelecinePattern::EuroPulldown);

        let mut total_output = 0;
        for i in 0..24 {
            let result = applier.process(&create_test_frame(i as u8)).unwrap();
            total_output += result.len();
        }

        // 24 input frames should produce 25 output frames
        assert_eq!(total_output, 25);
    }

    #[test]
    fn test_detect_telecine_insufficient_frames() {
        let frames: Vec<Frame> = (0..5).map(|i| create_test_frame(i as u8)).collect();
        let result = detect_telecine(&frames);
        assert!(matches!(
            result,
            Err(FrameRateError::InsufficientFrames { .. })
        ));
    }
}
