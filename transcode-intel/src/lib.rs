//! Content intelligence for transcode.
//!
//! This crate provides content analysis capabilities for video:
//!
//! - **Scene Detection**: Detect scene changes and cuts
//! - **Content Classification**: Classify shot types and content types
//! - **Motion Analysis**: Analyze motion levels for adaptive encoding
//!
//! # Example
//!
//! ```ignore
//! use transcode_intel::{SceneDetector, ContentClassifier, Frame};
//!
//! // Scene detection
//! let mut detector = SceneDetector::default();
//! for frame in frames {
//!     if let Some(confidence) = detector.process_frame(&frame)? {
//!         println!("Scene change detected! Confidence: {:.2}", confidence);
//!     }
//! }
//!
//! // Content classification
//! let mut classifier = ContentClassifier::new();
//! let classification = classifier.classify(&frame)?;
//! println!("Shot type: {:?}", classification.shot_type);
//! println!("Motion level: {:?}", classification.motion_level);
//! ```
//!
//! # Use Cases
//!
//! - **Adaptive bitrate**: Adjust encoding based on content complexity
//! - **Scene-based segmentation**: Split video at scene boundaries
//! - **Thumbnail selection**: Choose representative frames from each scene
//! - **Content-aware processing**: Apply different filters based on content

pub mod classify;
pub mod error;
pub mod scene;

pub use classify::{
    Classification, ContentClassifier, ContentType, MotionLevel, ShotType,
};
pub use error::{IntelError, Result};
pub use scene::{
    detect_scenes, DetectionMethod, Scene, SceneConfig, SceneDetector, SceneType,
};

/// A video frame for analysis.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Raw pixel data (RGB).
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of channels.
    pub channels: u8,
}

impl Frame {
    /// Create a new frame.
    pub fn new(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Self {
        Self {
            data,
            width,
            height,
            channels,
        }
    }
}

/// Video analyzer combining scene detection and classification.
#[derive(Debug)]
pub struct VideoAnalyzer {
    scene_detector: SceneDetector,
    classifier: ContentClassifier,
}

impl Default for VideoAnalyzer {
    fn default() -> Self {
        Self::new(SceneConfig::default())
    }
}

impl VideoAnalyzer {
    /// Create a new video analyzer.
    pub fn new(scene_config: SceneConfig) -> Self {
        Self {
            scene_detector: SceneDetector::new(scene_config),
            classifier: ContentClassifier::new(),
        }
    }

    /// Reset analyzer state.
    pub fn reset(&mut self) {
        self.scene_detector.reset();
        self.classifier.reset();
    }

    /// Analyze a frame.
    pub fn analyze(&mut self, frame: &Frame) -> Result<FrameAnalysis> {
        let scene_change = self.scene_detector.process_frame(frame)?;
        let classification = self.classifier.classify(frame)?;

        Ok(FrameAnalysis {
            is_scene_change: scene_change.is_some(),
            scene_confidence: scene_change.unwrap_or(0.0),
            classification,
        })
    }

    /// Analyze a sequence of frames.
    pub fn analyze_sequence(&mut self, frames: &[Frame]) -> Result<SequenceAnalysis> {
        let mut frame_analyses = Vec::with_capacity(frames.len());
        let mut scenes = Vec::new();
        let mut current_scene_start = 0;

        for (i, frame) in frames.iter().enumerate() {
            let analysis = self.analyze(frame)?;

            if analysis.is_scene_change && i > 0 {
                scenes.push(Scene {
                    index: scenes.len(),
                    start_frame: current_scene_start,
                    end_frame: i - 1,
                    confidence: analysis.scene_confidence,
                    scene_type: Some(SceneType::HardCut),
                });
                current_scene_start = i;
            }

            frame_analyses.push(analysis);
        }

        // Add final scene
        if !frames.is_empty() {
            scenes.push(Scene {
                index: scenes.len(),
                start_frame: current_scene_start,
                end_frame: frames.len() - 1,
                confidence: 1.0,
                scene_type: None,
            });
        }

        // Calculate averages
        let avg_motion = frame_analyses
            .iter()
            .map(|a| a.classification.motion_score)
            .sum::<f64>()
            / frame_analyses.len().max(1) as f64;

        let avg_complexity = frame_analyses
            .iter()
            .map(|a| a.classification.complexity)
            .sum::<f64>()
            / frame_analyses.len().max(1) as f64;

        // Determine dominant content type
        let mut content_counts = std::collections::HashMap::new();
        for analysis in &frame_analyses {
            *content_counts
                .entry(analysis.classification.content_type)
                .or_insert(0) += 1;
        }
        let dominant_content = content_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(content, _)| content)
            .unwrap_or(ContentType::Unknown);

        Ok(SequenceAnalysis {
            frame_count: frames.len(),
            scenes,
            dominant_content,
            avg_motion,
            avg_complexity,
        })
    }
}

/// Analysis result for a single frame.
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    /// Whether this frame is a scene change.
    pub is_scene_change: bool,
    /// Scene change confidence (if applicable).
    pub scene_confidence: f64,
    /// Content classification.
    pub classification: Classification,
}

/// Analysis result for a sequence of frames.
#[derive(Debug, Clone)]
pub struct SequenceAnalysis {
    /// Total frame count.
    pub frame_count: usize,
    /// Detected scenes.
    pub scenes: Vec<Scene>,
    /// Dominant content type.
    pub dominant_content: ContentType,
    /// Average motion score.
    pub avg_motion: f64,
    /// Average complexity.
    pub avg_complexity: f64,
}

impl SequenceAnalysis {
    /// Get scene count.
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Get average scene length in frames.
    pub fn avg_scene_length(&self) -> f64 {
        if self.scenes.is_empty() {
            0.0
        } else {
            self.frame_count as f64 / self.scenes.len() as f64
        }
    }

    /// Get recommended bitrate factor based on content.
    pub fn recommended_bitrate_factor(&self) -> f64 {
        // Base factor from motion
        let motion_factor = if self.avg_motion < 0.1 {
            0.7
        } else if self.avg_motion < 0.3 {
            1.0
        } else {
            1.3
        };

        // Complexity factor
        let complexity_factor = 0.8 + self.avg_complexity * 0.4;

        motion_factor * complexity_factor
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
    fn test_video_analyzer() {
        let mut analyzer = VideoAnalyzer::default();
        let frame = create_test_frame(64, 64, 128);

        let analysis = analyzer.analyze(&frame).unwrap();
        assert!(!analysis.is_scene_change); // First frame not a scene change
    }

    #[test]
    fn test_sequence_analysis() {
        let mut analyzer = VideoAnalyzer::new(SceneConfig {
            threshold: 0.1,
            min_scene_length: 1,
            adaptive_threshold: false,
            ..Default::default()
        });

        // Create a sequence with a clear scene change
        let frames: Vec<Frame> = (0..10)
            .map(|i| {
                let value = if i < 5 { 50 } else { 200 };
                create_test_frame(64, 64, value)
            })
            .collect();

        let analysis = analyzer.analyze_sequence(&frames).unwrap();
        assert_eq!(analysis.frame_count, 10);
        assert!(analysis.scenes.len() >= 2);
    }
}
