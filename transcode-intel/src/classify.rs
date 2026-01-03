//! Content classification for video frames.
//!
//! Classifies video content into categories based on visual features:
//! - Shot types (close-up, medium, wide, etc.)
//! - Content types (action, dialogue, landscape, etc.)
//! - Motion levels (static, slow, fast)

#![allow(clippy::needless_range_loop)]

use crate::error::{IntelError, Result};
use crate::Frame;
use serde::{Deserialize, Serialize};

/// Shot type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShotType {
    /// Extreme close-up (face detail, eyes).
    ExtremeCloseUp,
    /// Close-up (head and shoulders).
    CloseUp,
    /// Medium close-up (waist up).
    MediumCloseUp,
    /// Medium shot (knees up).
    Medium,
    /// Medium long shot (full body with some environment).
    MediumLong,
    /// Long shot (full body, environment visible).
    Long,
    /// Extreme long shot (wide landscape, small figures).
    ExtremeLong,
    /// Overhead/bird's eye view.
    Overhead,
    /// Unknown shot type.
    Unknown,
}

/// Content type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentType {
    /// Dialogue/talking head.
    Dialogue,
    /// Action sequence.
    Action,
    /// Landscape/scenery.
    Landscape,
    /// Static image/graphic.
    Static,
    /// Animation/cartoon.
    Animation,
    /// Text/credits.
    Text,
    /// Unknown content.
    Unknown,
}

/// Motion level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionLevel {
    /// Static (no motion).
    Static,
    /// Slow motion.
    Slow,
    /// Normal motion.
    Normal,
    /// Fast motion.
    Fast,
    /// Very fast motion (action scenes).
    VeryFast,
}

impl MotionLevel {
    /// Get recommended encoding parameters based on motion level.
    pub fn recommended_bitrate_factor(&self) -> f64 {
        match self {
            Self::Static => 0.5,
            Self::Slow => 0.7,
            Self::Normal => 1.0,
            Self::Fast => 1.3,
            Self::VeryFast => 1.5,
        }
    }
}

/// Classification result for a frame or sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Classification {
    /// Shot type.
    pub shot_type: ShotType,
    /// Shot type confidence (0.0 - 1.0).
    pub shot_confidence: f64,
    /// Content type.
    pub content_type: ContentType,
    /// Content type confidence (0.0 - 1.0).
    pub content_confidence: f64,
    /// Motion level.
    pub motion_level: MotionLevel,
    /// Motion score (0.0 - 1.0).
    pub motion_score: f64,
    /// Visual complexity (0.0 - 1.0).
    pub complexity: f64,
    /// Color saturation (0.0 - 1.0).
    pub saturation: f64,
    /// Brightness (0.0 - 1.0).
    pub brightness: f64,
}

impl Default for Classification {
    fn default() -> Self {
        Self {
            shot_type: ShotType::Unknown,
            shot_confidence: 0.0,
            content_type: ContentType::Unknown,
            content_confidence: 0.0,
            motion_level: MotionLevel::Normal,
            motion_score: 0.5,
            complexity: 0.5,
            saturation: 0.5,
            brightness: 0.5,
        }
    }
}

/// Content classifier.
#[derive(Debug)]
pub struct ContentClassifier {
    /// Previous frame for motion analysis.
    prev_frame: Option<FrameStats>,
    /// Running motion average.
    motion_avg: f64,
    /// Sample count.
    sample_count: usize,
}

/// Frame statistics for classification.
#[derive(Debug, Clone)]
struct FrameStats {
    /// Average brightness.
    brightness: f64,
    /// Color saturation.
    saturation: f64,
    /// Edge density.
    edge_density: f64,
    /// Center weight (for shot type).
    center_weight: f64,
    /// Horizontal distribution variance.
    h_variance: f64,
    /// Vertical distribution variance.
    v_variance: f64,
    /// Downsampled luminance for motion.
    thumbnail: Vec<u8>,
}

impl Default for ContentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentClassifier {
    /// Create a new content classifier.
    pub fn new() -> Self {
        Self {
            prev_frame: None,
            motion_avg: 0.0,
            sample_count: 0,
        }
    }

    /// Reset classifier state.
    pub fn reset(&mut self) {
        self.prev_frame = None;
        self.motion_avg = 0.0;
        self.sample_count = 0;
    }

    /// Classify a frame.
    pub fn classify(&mut self, frame: &Frame) -> Result<Classification> {
        let stats = self.analyze_frame(frame)?;

        // Calculate motion
        let motion_score = if let Some(ref prev) = self.prev_frame {
            self.calculate_motion(prev, &stats)
        } else {
            0.5
        };

        // Update motion average
        self.sample_count += 1;
        self.motion_avg += (motion_score - self.motion_avg) / self.sample_count as f64;

        // Classify shot type
        let (shot_type, shot_confidence) = self.classify_shot(&stats);

        // Classify content type
        let (content_type, content_confidence) = self.classify_content(&stats, motion_score);

        // Determine motion level
        let motion_level = self.determine_motion_level(motion_score);

        // Calculate complexity (based on edge density and variance)
        let complexity = (stats.edge_density + stats.h_variance + stats.v_variance) / 3.0;

        self.prev_frame = Some(stats.clone());

        Ok(Classification {
            shot_type,
            shot_confidence,
            content_type,
            content_confidence,
            motion_level,
            motion_score,
            complexity,
            saturation: stats.saturation,
            brightness: stats.brightness,
        })
    }

    /// Analyze frame and extract statistics.
    fn analyze_frame(&self, frame: &Frame) -> Result<FrameStats> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        if channels < 3 {
            return Err(IntelError::InvalidFrame("Expected RGB frame".to_string()));
        }

        let pixel_count = (width * height) as f64;

        // Calculate brightness and saturation
        let mut sum_brightness = 0.0f64;
        let mut sum_saturation = 0.0f64;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * channels;
                let r = frame.data[idx] as f64 / 255.0;
                let g = frame.data[idx + 1] as f64 / 255.0;
                let b = frame.data[idx + 2] as f64 / 255.0;

                // Brightness
                let brightness = 0.299 * r + 0.587 * g + 0.114 * b;
                sum_brightness += brightness;

                // Saturation (simplified)
                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                let saturation = if max > 0.0 { (max - min) / max } else { 0.0 };
                sum_saturation += saturation;
            }
        }

        let brightness = sum_brightness / pixel_count;
        let saturation = sum_saturation / pixel_count;

        // Edge density
        let edge_density = self.calculate_edge_density(frame);

        // Center weight (ratio of detail in center vs edges)
        let center_weight = self.calculate_center_weight(frame);

        // Distribution variances
        let (h_variance, v_variance) = self.calculate_distribution_variance(frame);

        // Create thumbnail for motion detection
        let thumbnail = self.create_thumbnail(frame, 16, 16);

        Ok(FrameStats {
            brightness,
            saturation,
            edge_density,
            center_weight,
            h_variance,
            v_variance,
            thumbnail,
        })
    }

    /// Calculate edge density.
    fn calculate_edge_density(&self, frame: &Frame) -> f64 {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        if width < 3 || height < 3 {
            return 0.0;
        }

        let mut edge_sum = 0.0;
        let sample_step = 2;

        for y in (1..height - 1).step_by(sample_step) {
            for x in (1..width - 1).step_by(sample_step) {
                let get_lum = |dx: i32, dy: i32| -> f64 {
                    let px = (x as i32 + dx) as usize;
                    let py = (y as i32 + dy) as usize;
                    let idx = (py * width + px) * channels;
                    let r = frame.data[idx] as f64;
                    let g = frame.data[idx + 1] as f64;
                    let b = frame.data[idx + 2] as f64;
                    0.299 * r + 0.587 * g + 0.114 * b
                };

                let gx = get_lum(1, 0) - get_lum(-1, 0);
                let gy = get_lum(0, 1) - get_lum(0, -1);
                let gradient = (gx * gx + gy * gy).sqrt();

                edge_sum += if gradient > 30.0 { 1.0 } else { 0.0 };
            }
        }

        let samples =
            ((width - 2) / sample_step) as f64 * ((height - 2) / sample_step) as f64;
        edge_sum / samples
    }

    /// Calculate center weight (how much detail is in the center).
    fn calculate_center_weight(&self, frame: &Frame) -> f64 {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        // Define center region (middle 50%)
        let cx_start = width / 4;
        let cx_end = 3 * width / 4;
        let cy_start = height / 4;
        let cy_end = 3 * height / 4;

        let mut center_var = 0.0;
        let mut edge_var = 0.0;
        let mut center_count = 0;
        let mut edge_count = 0;

        let sample_step = 4;

        for y in (0..height).step_by(sample_step) {
            for x in (0..width).step_by(sample_step) {
                let idx = (y * width + x) * channels;
                let lum = 0.299 * frame.data[idx] as f64
                    + 0.587 * frame.data[idx + 1] as f64
                    + 0.114 * frame.data[idx + 2] as f64;

                let is_center = x >= cx_start && x < cx_end && y >= cy_start && y < cy_end;

                if is_center {
                    center_var += lum;
                    center_count += 1;
                } else {
                    edge_var += lum;
                    edge_count += 1;
                }
            }
        }

        let center_avg = if center_count > 0 {
            center_var / center_count as f64
        } else {
            128.0
        };
        let edge_avg = if edge_count > 0 {
            edge_var / edge_count as f64
        } else {
            128.0
        };

        // Higher means more activity in center
        let diff = (center_avg - edge_avg).abs();
        (diff / 128.0).min(1.0)
    }

    /// Calculate horizontal and vertical distribution variance.
    fn calculate_distribution_variance(&self, frame: &Frame) -> (f64, f64) {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        // Horizontal projection
        let mut h_proj = vec![0.0f64; width];
        for x in 0..width {
            for y in 0..height {
                let idx = (y * width + x) * channels;
                let lum = 0.299 * frame.data[idx] as f64
                    + 0.587 * frame.data[idx + 1] as f64
                    + 0.114 * frame.data[idx + 2] as f64;
                h_proj[x] += lum;
            }
        }

        // Vertical projection
        let mut v_proj = vec![0.0f64; height];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * channels;
                let lum = 0.299 * frame.data[idx] as f64
                    + 0.587 * frame.data[idx + 1] as f64
                    + 0.114 * frame.data[idx + 2] as f64;
                v_proj[y] += lum;
            }
        }

        let h_mean: f64 = h_proj.iter().sum::<f64>() / width as f64;
        let v_mean: f64 = v_proj.iter().sum::<f64>() / height as f64;

        let h_var = h_proj.iter().map(|&x| (x - h_mean).powi(2)).sum::<f64>() / width as f64;
        let v_var = v_proj.iter().map(|&x| (x - v_mean).powi(2)).sum::<f64>() / height as f64;

        // Normalize
        let max_var = 255.0 * 255.0 * height as f64;
        (
            (h_var / max_var).sqrt().min(1.0),
            (v_var / max_var).sqrt().min(1.0),
        )
    }

    /// Create a small thumbnail for motion detection.
    fn create_thumbnail(&self, frame: &Frame, tw: usize, th: usize) -> Vec<u8> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        let block_w = width / tw;
        let block_h = height / th;

        let mut thumbnail = vec![0u8; tw * th];

        for ty in 0..th {
            for tx in 0..tw {
                let mut sum = 0u32;
                let mut count = 0;

                for y in (ty * block_h)..((ty + 1) * block_h).min(height) {
                    for x in (tx * block_w)..((tx + 1) * block_w).min(width) {
                        let idx = (y * width + x) * channels;
                        let lum = (0.299 * frame.data[idx] as f64
                            + 0.587 * frame.data[idx + 1] as f64
                            + 0.114 * frame.data[idx + 2] as f64) as u32;
                        sum += lum;
                        count += 1;
                    }
                }

                thumbnail[ty * tw + tx] = if count > 0 { (sum / count) as u8 } else { 0 };
            }
        }

        thumbnail
    }

    /// Calculate motion between frames.
    fn calculate_motion(&self, prev: &FrameStats, current: &FrameStats) -> f64 {
        if prev.thumbnail.len() != current.thumbnail.len() {
            return 0.5;
        }

        let mut diff_sum = 0u32;
        for (p, c) in prev.thumbnail.iter().zip(current.thumbnail.iter()) {
            diff_sum += (*p as i32 - *c as i32).unsigned_abs();
        }

        let max_diff = (prev.thumbnail.len() * 255) as f64;
        (diff_sum as f64 / max_diff).min(1.0)
    }

    /// Classify shot type based on frame statistics.
    fn classify_shot(&self, stats: &FrameStats) -> (ShotType, f64) {
        // High center weight + high edge density = close-up
        // Low center weight + low edge density = wide shot

        if stats.center_weight > 0.5 && stats.edge_density > 0.3 {
            (ShotType::CloseUp, 0.7 + stats.center_weight * 0.2)
        } else if stats.center_weight > 0.3 && stats.edge_density > 0.2 {
            (ShotType::Medium, 0.6 + stats.center_weight * 0.2)
        } else if stats.edge_density < 0.1 && stats.h_variance < 0.2 {
            (ShotType::ExtremeLong, 0.6)
        } else if stats.edge_density < 0.2 {
            (ShotType::Long, 0.5 + (0.2 - stats.edge_density) * 2.0)
        } else {
            (ShotType::Medium, 0.4)
        }
    }

    /// Classify content type.
    fn classify_content(&self, stats: &FrameStats, motion: f64) -> (ContentType, f64) {
        // High edge density + low motion = dialogue/static
        // High motion + high saturation = action
        // Low saturation + low motion = text/graphics

        if motion < 0.05 && stats.edge_density < 0.1 {
            (ContentType::Static, 0.8)
        } else if motion < 0.1 && stats.center_weight > 0.4 {
            (ContentType::Dialogue, 0.6 + stats.center_weight * 0.3)
        } else if motion > 0.3 {
            (ContentType::Action, 0.5 + motion * 0.5)
        } else if stats.h_variance > 0.5 && stats.edge_density < 0.2 {
            (ContentType::Landscape, 0.5 + stats.h_variance * 0.3)
        } else if stats.saturation < 0.1 && stats.edge_density > 0.4 {
            (ContentType::Text, 0.6)
        } else {
            (ContentType::Unknown, 0.3)
        }
    }

    /// Determine motion level from score.
    fn determine_motion_level(&self, motion: f64) -> MotionLevel {
        if motion < 0.02 {
            MotionLevel::Static
        } else if motion < 0.1 {
            MotionLevel::Slow
        } else if motion < 0.25 {
            MotionLevel::Normal
        } else if motion < 0.5 {
            MotionLevel::Fast
        } else {
            MotionLevel::VeryFast
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, value: u8) -> Frame {
        let data = vec![value; (width * height * 3) as usize];
        Frame::new(data, width, height, 3)
    }

    #[test]
    fn test_classifier_static_frame() {
        let mut classifier = ContentClassifier::new();
        let frame = create_test_frame(64, 64, 128);

        let result = classifier.classify(&frame).unwrap();
        assert!(result.brightness > 0.0);
        assert!(result.brightness < 1.0);
    }

    #[test]
    fn test_motion_detection() {
        let mut classifier = ContentClassifier::new();

        let frame1 = create_test_frame(64, 64, 50);
        let frame2 = create_test_frame(64, 64, 200);

        classifier.classify(&frame1).unwrap();
        let result2 = classifier.classify(&frame2).unwrap();

        // Should detect motion between very different frames
        assert!(result2.motion_score > 0.3);
    }

    #[test]
    fn test_motion_level() {
        let mut classifier = ContentClassifier::new();

        // Same frame twice - should be static
        let frame = create_test_frame(64, 64, 128);
        classifier.classify(&frame).unwrap();
        let result = classifier.classify(&frame).unwrap();

        assert!(matches!(
            result.motion_level,
            MotionLevel::Static | MotionLevel::Slow
        ));
    }
}
