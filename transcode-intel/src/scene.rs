//! Scene detection for video content.
//!
//! Detects scene changes (cuts) in video using multiple methods:
//! - Histogram difference
//! - Edge detection
//! - Content-based detection
//! - Motion analysis

use crate::error::Result;
use crate::Frame;
use serde::{Deserialize, Serialize};

/// Scene change detection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Histogram-based detection (fast).
    Histogram,
    /// Content-based using luminance difference.
    #[default]
    ContentDiff,
    /// Edge-based detection.
    Edge,
    /// Combined methods (most accurate).
    Combined,
}

/// Scene detection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneConfig {
    /// Detection method.
    pub method: DetectionMethod,
    /// Threshold for scene change (0.0 - 1.0).
    pub threshold: f64,
    /// Minimum frames between scene changes.
    pub min_scene_length: usize,
    /// Adaptive threshold based on content.
    pub adaptive_threshold: bool,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            method: DetectionMethod::default(),
            threshold: 0.3,
            min_scene_length: 12, // ~0.5 seconds at 24fps
            adaptive_threshold: true,
        }
    }
}

impl SceneConfig {
    /// Set detection method.
    pub fn with_method(mut self, method: DetectionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set minimum scene length.
    pub fn with_min_scene_length(mut self, frames: usize) -> Self {
        self.min_scene_length = frames;
        self
    }
}

/// A detected scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Scene index.
    pub index: usize,
    /// Start frame number.
    pub start_frame: usize,
    /// End frame number.
    pub end_frame: usize,
    /// Scene change confidence (0.0 - 1.0).
    pub confidence: f64,
    /// Scene type (if classified).
    pub scene_type: Option<SceneType>,
}

impl Scene {
    /// Get frame count.
    pub fn frame_count(&self) -> usize {
        self.end_frame - self.start_frame + 1
    }

    /// Get duration in seconds at given frame rate.
    pub fn duration(&self, fps: f64) -> f64 {
        self.frame_count() as f64 / fps
    }
}

/// Scene types for classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneType {
    /// Hard cut (instant scene change).
    HardCut,
    /// Dissolve transition.
    Dissolve,
    /// Fade in.
    FadeIn,
    /// Fade out.
    FadeOut,
    /// Wipe transition.
    Wipe,
    /// Unknown transition.
    Unknown,
}

/// Scene detector.
#[derive(Debug)]
pub struct SceneDetector {
    config: SceneConfig,
    /// Previous frame for comparison.
    prev_frame: Option<FrameData>,
    /// Frame index.
    frame_index: usize,
    /// Last scene change frame.
    last_scene_frame: usize,
    /// Running average for adaptive threshold.
    running_avg: f64,
    /// Running variance for adaptive threshold.
    running_var: f64,
    /// Sample count for running statistics.
    sample_count: usize,
}

/// Internal frame data for analysis.
#[derive(Debug, Clone)]
struct FrameData {
    /// Histogram bins.
    histogram: [u32; 256],
    /// Average luminance.
    avg_luminance: f64,
    /// Edge strength.
    edge_strength: f64,
    /// Frame content hash (for fast comparison).
    content_hash: u64,
}

impl Default for SceneDetector {
    fn default() -> Self {
        Self::new(SceneConfig::default())
    }
}

impl SceneDetector {
    /// Create a new scene detector.
    pub fn new(config: SceneConfig) -> Self {
        Self {
            config,
            prev_frame: None,
            frame_index: 0,
            last_scene_frame: 0,
            running_avg: 0.0,
            running_var: 0.0,
            sample_count: 0,
        }
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.prev_frame = None;
        self.frame_index = 0;
        self.last_scene_frame = 0;
        self.running_avg = 0.0;
        self.running_var = 0.0;
        self.sample_count = 0;
    }

    /// Process a frame and detect if it's a scene change.
    pub fn process_frame(&mut self, frame: &Frame) -> Result<Option<f64>> {
        let current = self.analyze_frame(frame)?;
        self.frame_index += 1;

        let result = if let Some(ref prev) = self.prev_frame {
            let score = self.calculate_change_score(prev, &current);

            // Update running statistics
            self.update_statistics(score);

            // Determine threshold
            let threshold = if self.config.adaptive_threshold {
                self.adaptive_threshold()
            } else {
                self.config.threshold
            };

            // Check for scene change
            let is_scene_change = score > threshold
                && (self.frame_index - self.last_scene_frame) >= self.config.min_scene_length;

            if is_scene_change {
                self.last_scene_frame = self.frame_index;
                Some(score)
            } else {
                None
            }
        } else {
            None
        };

        self.prev_frame = Some(current);
        Ok(result)
    }

    /// Analyze a frame and extract features.
    fn analyze_frame(&self, frame: &Frame) -> Result<FrameData> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        // Calculate histogram (luminance)
        let mut histogram = [0u32; 256];
        let mut sum_lum = 0u64;

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * channels;
                let lum = if channels >= 3 {
                    // RGB to luminance
                    let r = frame.data[idx] as f64;
                    let g = frame.data[idx + 1] as f64;
                    let b = frame.data[idx + 2] as f64;
                    (0.299 * r + 0.587 * g + 0.114 * b) as u8
                } else {
                    frame.data[idx]
                };

                histogram[lum as usize] += 1;
                sum_lum += lum as u64;
            }
        }

        let pixel_count = (width * height) as f64;
        let avg_luminance = sum_lum as f64 / pixel_count;

        // Calculate edge strength using Sobel
        let edge_strength = self.calculate_edge_strength(frame);

        // Calculate content hash
        let content_hash = self.calculate_content_hash(frame);

        Ok(FrameData {
            histogram,
            avg_luminance,
            edge_strength,
            content_hash,
        })
    }

    /// Calculate change score between frames.
    fn calculate_change_score(&self, prev: &FrameData, current: &FrameData) -> f64 {
        match self.config.method {
            DetectionMethod::Histogram => self.histogram_diff(prev, current),
            DetectionMethod::ContentDiff => self.content_diff(prev, current),
            DetectionMethod::Edge => self.edge_diff(prev, current),
            DetectionMethod::Combined => {
                let hist = self.histogram_diff(prev, current);
                let content = self.content_diff(prev, current);
                let edge = self.edge_diff(prev, current);

                // Weighted combination
                hist * 0.4 + content * 0.4 + edge * 0.2
            }
        }
    }

    /// Calculate histogram difference.
    fn histogram_diff(&self, prev: &FrameData, current: &FrameData) -> f64 {
        let mut diff = 0.0;
        let mut total = 0.0;

        for i in 0..256 {
            let p = prev.histogram[i] as f64;
            let c = current.histogram[i] as f64;
            diff += (p - c).abs();
            total += p + c;
        }

        if total > 0.0 {
            diff / total
        } else {
            0.0
        }
    }

    /// Calculate content difference.
    fn content_diff(&self, prev: &FrameData, current: &FrameData) -> f64 {
        // Luminance difference
        let lum_diff = (prev.avg_luminance - current.avg_luminance).abs() / 255.0;

        // Hash difference (Hamming distance)
        let hash_xor = prev.content_hash ^ current.content_hash;
        let hash_diff = hash_xor.count_ones() as f64 / 64.0;

        (lum_diff + hash_diff) / 2.0
    }

    /// Calculate edge difference.
    fn edge_diff(&self, prev: &FrameData, current: &FrameData) -> f64 {
        let edge_max = prev.edge_strength.max(current.edge_strength);
        if edge_max > 0.0 {
            (prev.edge_strength - current.edge_strength).abs() / edge_max
        } else {
            0.0
        }
    }

    /// Calculate edge strength using Sobel operator.
    fn calculate_edge_strength(&self, frame: &Frame) -> f64 {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        if width < 3 || height < 3 {
            return 0.0;
        }

        let mut edge_sum = 0.0;
        let sample_step = 4; // Sample every 4th pixel for speed

        for y in (1..height - 1).step_by(sample_step) {
            for x in (1..width - 1).step_by(sample_step) {
                let get_lum = |dx: i32, dy: i32| -> f64 {
                    let px = (x as i32 + dx) as usize;
                    let py = (y as i32 + dy) as usize;
                    let idx = (py * width + px) * channels;
                    if channels >= 3 {
                        let r = frame.data[idx] as f64;
                        let g = frame.data[idx + 1] as f64;
                        let b = frame.data[idx + 2] as f64;
                        0.299 * r + 0.587 * g + 0.114 * b
                    } else {
                        frame.data[idx] as f64
                    }
                };

                // Sobel X
                let gx = -get_lum(-1, -1) - 2.0 * get_lum(-1, 0) - get_lum(-1, 1)
                    + get_lum(1, -1)
                    + 2.0 * get_lum(1, 0)
                    + get_lum(1, 1);

                // Sobel Y
                let gy = -get_lum(-1, -1) - 2.0 * get_lum(0, -1) - get_lum(1, -1)
                    + get_lum(-1, 1)
                    + 2.0 * get_lum(0, 1)
                    + get_lum(1, 1);

                edge_sum += (gx * gx + gy * gy).sqrt();
            }
        }

        let samples = ((width - 2) / sample_step) * ((height - 2) / sample_step);
        edge_sum / (samples as f64 * 255.0)
    }

    /// Calculate a simple content hash (perceptual hash).
    fn calculate_content_hash(&self, frame: &Frame) -> u64 {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        // Downsample to 8x8
        let block_w = width / 8;
        let block_h = height / 8;

        let mut values = [0.0f64; 64];

        for by in 0..8 {
            for bx in 0..8 {
                let mut sum = 0.0;
                let mut count = 0;

                for y in (by * block_h)..((by + 1) * block_h).min(height) {
                    for x in (bx * block_w)..((bx + 1) * block_w).min(width) {
                        let idx = (y * width + x) * channels;
                        let lum = if channels >= 3 {
                            let r = frame.data[idx] as f64;
                            let g = frame.data[idx + 1] as f64;
                            let b = frame.data[idx + 2] as f64;
                            0.299 * r + 0.587 * g + 0.114 * b
                        } else {
                            frame.data[idx] as f64
                        };
                        sum += lum;
                        count += 1;
                    }
                }

                values[by * 8 + bx] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }

        // Calculate average
        let avg: f64 = values.iter().sum::<f64>() / 64.0;

        // Create hash
        let mut hash = 0u64;
        for (i, &val) in values.iter().enumerate() {
            if val > avg {
                hash |= 1 << i;
            }
        }

        hash
    }

    /// Update running statistics for adaptive threshold.
    fn update_statistics(&mut self, score: f64) {
        self.sample_count += 1;

        // Welford's online algorithm
        let delta = score - self.running_avg;
        self.running_avg += delta / self.sample_count as f64;
        let delta2 = score - self.running_avg;
        self.running_var += delta * delta2;
    }

    /// Calculate adaptive threshold.
    fn adaptive_threshold(&self) -> f64 {
        if self.sample_count < 10 {
            return self.config.threshold;
        }

        let variance = self.running_var / self.sample_count as f64;
        let std_dev = variance.sqrt();

        // Threshold = mean + 2 * std_dev
        let adaptive = self.running_avg + 2.0 * std_dev;

        // Clamp to reasonable range
        adaptive.clamp(0.1, 0.8)
    }
}

/// Batch scene detection for a sequence of frames.
pub fn detect_scenes(frames: &[Frame], config: SceneConfig) -> Result<Vec<Scene>> {
    if frames.is_empty() {
        return Ok(Vec::new());
    }

    let mut detector = SceneDetector::new(config);
    let mut scenes = Vec::new();
    let mut current_scene_start = 0;

    for (i, frame) in frames.iter().enumerate() {
        if let Some(confidence) = detector.process_frame(frame)? {
            // End previous scene
            if i > 0 {
                scenes.push(Scene {
                    index: scenes.len(),
                    start_frame: current_scene_start,
                    end_frame: i - 1,
                    confidence,
                    scene_type: Some(SceneType::HardCut),
                });
            }
            current_scene_start = i;
        }
    }

    // Add final scene
    scenes.push(Scene {
        index: scenes.len(),
        start_frame: current_scene_start,
        end_frame: frames.len() - 1,
        confidence: 1.0,
        scene_type: None,
    });

    Ok(scenes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32, value: u8) -> Frame {
        let data = vec![value; (width * height * 3) as usize];
        Frame::new(data, width, height, 3)
    }

    #[test]
    fn test_scene_detector_same_frame() {
        let mut detector = SceneDetector::default();
        let frame = create_test_frame(64, 64, 128);

        // First frame
        let result1 = detector.process_frame(&frame).unwrap();
        assert!(result1.is_none());

        // Same frame again - no scene change
        let result2 = detector.process_frame(&frame).unwrap();
        assert!(result2.is_none());
    }

    #[test]
    fn test_scene_detector_different_frames() {
        let mut detector = SceneDetector::new(SceneConfig {
            threshold: 0.1,
            min_scene_length: 1,
            adaptive_threshold: false,
            ..Default::default()
        });

        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(64, 64, 255);

        detector.process_frame(&frame1).unwrap();
        let result = detector.process_frame(&frame2).unwrap();

        // Should detect scene change
        assert!(result.is_some());
    }

    #[test]
    fn test_batch_scene_detection() {
        let frames: Vec<Frame> = (0..10)
            .map(|i| {
                let value = if i < 5 { 50 } else { 200 };
                create_test_frame(64, 64, value)
            })
            .collect();

        let config = SceneConfig {
            threshold: 0.1,
            min_scene_length: 1,
            adaptive_threshold: false,
            ..Default::default()
        };

        let scenes = detect_scenes(&frames, config).unwrap();
        assert!(scenes.len() >= 2); // At least 2 scenes
    }
}
