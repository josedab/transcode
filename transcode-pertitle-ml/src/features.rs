//! Feature extraction for ML-based per-title encoding.

use serde::{Deserialize, Serialize};

/// Content complexity features extracted from video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFeatures {
    /// Spatial complexity (0.0-1.0).
    pub spatial_complexity: f64,
    /// Temporal complexity (0.0-1.0).
    pub temporal_complexity: f64,
    /// Average motion magnitude.
    pub motion_magnitude: f64,
    /// Scene change frequency (changes per second).
    pub scene_change_rate: f64,
    /// Color variance.
    pub color_variance: f64,
    /// Edge density.
    pub edge_density: f64,
    /// Texture complexity.
    pub texture_complexity: f64,
    /// Dark content ratio (0.0-1.0).
    pub dark_ratio: f64,
    /// High detail ratio (0.0-1.0).
    pub detail_ratio: f64,
    /// Film grain level (0.0-1.0).
    pub grain_level: f64,
}

impl Default for ContentFeatures {
    fn default() -> Self {
        Self {
            spatial_complexity: 0.5,
            temporal_complexity: 0.5,
            motion_magnitude: 0.5,
            scene_change_rate: 0.1,
            color_variance: 0.5,
            edge_density: 0.5,
            texture_complexity: 0.5,
            dark_ratio: 0.2,
            detail_ratio: 0.5,
            grain_level: 0.1,
        }
    }
}

impl ContentFeatures {
    /// Convert features to vector for ML model input.
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.spatial_complexity,
            self.temporal_complexity,
            self.motion_magnitude,
            self.scene_change_rate,
            self.color_variance,
            self.edge_density,
            self.texture_complexity,
            self.dark_ratio,
            self.detail_ratio,
            self.grain_level,
        ]
    }

    /// Create from vector.
    pub fn from_vector(v: &[f64]) -> Option<Self> {
        if v.len() < 10 {
            return None;
        }
        Some(Self {
            spatial_complexity: v[0],
            temporal_complexity: v[1],
            motion_magnitude: v[2],
            scene_change_rate: v[3],
            color_variance: v[4],
            edge_density: v[5],
            texture_complexity: v[6],
            dark_ratio: v[7],
            detail_ratio: v[8],
            grain_level: v[9],
        })
    }

    /// Calculate overall complexity score (0.0-1.0).
    pub fn complexity_score(&self) -> f64 {
        // Weighted average of complexity factors
        let spatial_weight = 0.3;
        let temporal_weight = 0.3;
        let motion_weight = 0.15;
        let detail_weight = 0.15;
        let grain_weight = 0.1;

        (self.spatial_complexity * spatial_weight
            + self.temporal_complexity * temporal_weight
            + self.motion_magnitude * motion_weight
            + self.detail_ratio * detail_weight
            + self.grain_level * grain_weight)
            .clamp(0.0, 1.0)
    }
}

/// Video metadata features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Frame rate.
    pub frame_rate: f64,
    /// Duration in seconds.
    pub duration: f64,
    /// Total frames.
    pub total_frames: u64,
    /// Bit depth.
    pub bit_depth: u8,
    /// Color space (e.g., "bt709", "bt2020").
    pub color_space: String,
    /// HDR type if applicable.
    pub hdr_type: Option<String>,
}

impl Default for VideoMetadata {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: 24.0,
            duration: 0.0,
            total_frames: 0,
            bit_depth: 8,
            color_space: "bt709".to_string(),
            hdr_type: None,
        }
    }
}

impl VideoMetadata {
    /// Get resolution category.
    pub fn resolution_category(&self) -> ResolutionCategory {
        let pixels = self.width * self.height;
        if pixels >= 3840 * 2160 {
            ResolutionCategory::Uhd4K
        } else if pixels >= 1920 * 1080 {
            ResolutionCategory::Fhd1080P
        } else if pixels >= 1280 * 720 {
            ResolutionCategory::Hd720P
        } else if pixels >= 854 * 480 {
            ResolutionCategory::Sd480P
        } else {
            ResolutionCategory::Low
        }
    }

    /// Get total pixels.
    pub fn total_pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Convert to feature vector.
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.width as f64 / 3840.0,  // Normalize to 4K
            self.height as f64 / 2160.0,
            self.frame_rate / 60.0,
            self.bit_depth as f64 / 12.0,
            if self.hdr_type.is_some() { 1.0 } else { 0.0 },
        ]
    }
}

/// Resolution categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResolutionCategory {
    Low,
    Sd480P,
    Hd720P,
    Fhd1080P,
    Uhd4K,
}

impl ResolutionCategory {
    /// Get default bitrate range for this resolution (min, max) in kbps.
    pub fn bitrate_range_kbps(&self) -> (u32, u32) {
        match self {
            ResolutionCategory::Low => (200, 800),
            ResolutionCategory::Sd480P => (500, 2000),
            ResolutionCategory::Hd720P => (1500, 5000),
            ResolutionCategory::Fhd1080P => (3000, 10000),
            ResolutionCategory::Uhd4K => (10000, 40000),
        }
    }
}

/// Scene-level features for shot-based encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFeatures {
    /// Scene start frame.
    pub start_frame: u64,
    /// Scene end frame.
    pub end_frame: u64,
    /// Scene duration in seconds.
    pub duration: f64,
    /// Content features for this scene.
    pub content: ContentFeatures,
    /// Scene type classification.
    pub scene_type: SceneType,
    /// Confidence of scene type classification.
    pub confidence: f64,
}

impl SceneFeatures {
    /// Calculate frame count.
    pub fn frame_count(&self) -> u64 {
        self.end_frame - self.start_frame
    }
}

/// Scene type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SceneType {
    /// Static scene with minimal motion.
    Static,
    /// Talking head / interview.
    TalkingHead,
    /// Action / fast motion.
    Action,
    /// Nature / landscape.
    Nature,
    /// Animation / cartoon.
    Animation,
    /// Sports footage.
    Sports,
    /// Text / graphics overlay.
    Graphics,
    /// Unknown / mixed content.
    Unknown,
}

impl SceneType {
    /// Get recommended CRF adjustment for this scene type.
    pub fn crf_adjustment(&self) -> i32 {
        match self {
            SceneType::Static => 2,      // Can use higher CRF (lower quality ok)
            SceneType::TalkingHead => 1,
            SceneType::Action => -2,     // Need lower CRF (higher quality)
            SceneType::Nature => 0,
            SceneType::Animation => 1,
            SceneType::Sports => -2,
            SceneType::Graphics => -1,   // Text needs clarity
            SceneType::Unknown => 0,
        }
    }
}

/// Feature extractor for content analysis.
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Number of frames to sample for analysis.
    pub sample_frames: usize,
    /// Enable motion analysis.
    pub analyze_motion: bool,
    /// Enable texture analysis.
    pub analyze_texture: bool,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self {
            sample_frames: 30,
            analyze_motion: true,
            analyze_texture: true,
        }
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract content features from frame data.
    ///
    /// This is a simplified implementation. In production, would use
    /// actual frame pixel data for DCT-based analysis, motion vectors, etc.
    pub fn extract_features(&self, frame_stats: &[FrameStats]) -> ContentFeatures {
        if frame_stats.is_empty() {
            return ContentFeatures::default();
        }

        let n = frame_stats.len() as f64;

        // Spatial complexity: average variance
        let spatial_complexity = frame_stats.iter().map(|f| f.variance).sum::<f64>() / n;

        // Temporal complexity: difference between consecutive frames
        let temporal_complexity = if frame_stats.len() > 1 {
            let diffs: Vec<f64> = frame_stats
                .windows(2)
                .map(|w| (w[1].mean - w[0].mean).abs())
                .collect();
            diffs.iter().sum::<f64>() / diffs.len() as f64
        } else {
            0.0
        };

        // Motion magnitude
        let motion_magnitude = frame_stats.iter().map(|f| f.motion).sum::<f64>() / n;

        // Scene change rate
        let scene_changes = frame_stats.windows(2).filter(|w| w[1].is_scene_change).count();
        let duration = frame_stats.len() as f64 / 30.0; // Assume 30fps
        let scene_change_rate = scene_changes as f64 / duration.max(1.0);

        // Edge density
        let edge_density = frame_stats.iter().map(|f| f.edge_density).sum::<f64>() / n;

        // Detail ratio
        let detail_ratio = frame_stats.iter().map(|f| f.detail_score).sum::<f64>() / n;

        ContentFeatures {
            spatial_complexity: normalize(spatial_complexity, 0.0, 100.0),
            temporal_complexity: normalize(temporal_complexity, 0.0, 50.0),
            motion_magnitude: normalize(motion_magnitude, 0.0, 1.0),
            scene_change_rate: normalize(scene_change_rate, 0.0, 2.0),
            color_variance: 0.5, // Would need actual color analysis
            edge_density: normalize(edge_density, 0.0, 1.0),
            texture_complexity: normalize(spatial_complexity * edge_density, 0.0, 50.0),
            dark_ratio: frame_stats.iter().filter(|f| f.mean < 50.0).count() as f64 / n,
            detail_ratio: normalize(detail_ratio, 0.0, 1.0),
            grain_level: 0.1, // Would need actual noise estimation
        }
    }

    /// Classify scene type from features.
    pub fn classify_scene(&self, features: &ContentFeatures) -> (SceneType, f64) {
        // Simple rule-based classification
        // In production, would use ML model

        if features.motion_magnitude < 0.1 && features.temporal_complexity < 0.1 {
            return (SceneType::Static, 0.9);
        }

        if features.motion_magnitude > 0.7 {
            return (SceneType::Action, 0.8);
        }

        if features.edge_density > 0.8 && features.spatial_complexity < 0.3 {
            return (SceneType::Graphics, 0.7);
        }

        if features.texture_complexity > 0.6 && features.color_variance > 0.6 {
            return (SceneType::Nature, 0.6);
        }

        if features.motion_magnitude < 0.3 && features.spatial_complexity < 0.4 {
            return (SceneType::TalkingHead, 0.6);
        }

        (SceneType::Unknown, 0.5)
    }
}

/// Statistics for a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameStats {
    /// Frame number.
    pub frame_number: u64,
    /// Mean pixel value.
    pub mean: f64,
    /// Variance.
    pub variance: f64,
    /// Motion estimate (0.0-1.0).
    pub motion: f64,
    /// Edge density (0.0-1.0).
    pub edge_density: f64,
    /// Detail score (0.0-1.0).
    pub detail_score: f64,
    /// Is this a scene change.
    pub is_scene_change: bool,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            frame_number: 0,
            mean: 128.0,
            variance: 50.0,
            motion: 0.0,
            edge_density: 0.5,
            detail_score: 0.5,
            is_scene_change: false,
        }
    }
}

fn normalize(value: f64, min: f64, max: f64) -> f64 {
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_features_vector() {
        let features = ContentFeatures::default();
        let vec = features.to_vector();
        assert_eq!(vec.len(), 10);

        let restored = ContentFeatures::from_vector(&vec).unwrap();
        assert!((restored.spatial_complexity - features.spatial_complexity).abs() < 0.001);
    }

    #[test]
    fn test_complexity_score() {
        let simple = ContentFeatures {
            spatial_complexity: 0.1,
            temporal_complexity: 0.1,
            motion_magnitude: 0.1,
            detail_ratio: 0.1,
            grain_level: 0.1,
            ..Default::default()
        };
        assert!(simple.complexity_score() < 0.2);

        let complex = ContentFeatures {
            spatial_complexity: 0.9,
            temporal_complexity: 0.9,
            motion_magnitude: 0.9,
            detail_ratio: 0.9,
            grain_level: 0.9,
            ..Default::default()
        };
        assert!(complex.complexity_score() > 0.8);
    }

    #[test]
    fn test_resolution_category() {
        let meta_4k = VideoMetadata { width: 3840, height: 2160, ..Default::default() };
        assert_eq!(meta_4k.resolution_category(), ResolutionCategory::Uhd4K);

        let meta_1080 = VideoMetadata { width: 1920, height: 1080, ..Default::default() };
        assert_eq!(meta_1080.resolution_category(), ResolutionCategory::Fhd1080P);
    }

    #[test]
    fn test_scene_classification() {
        let extractor = FeatureExtractor::default();

        let static_features = ContentFeatures {
            motion_magnitude: 0.05,
            temporal_complexity: 0.05,
            ..Default::default()
        };
        let (scene_type, _) = extractor.classify_scene(&static_features);
        assert_eq!(scene_type, SceneType::Static);

        let action_features = ContentFeatures {
            motion_magnitude: 0.9,
            ..Default::default()
        };
        let (scene_type, _) = extractor.classify_scene(&action_features);
        assert_eq!(scene_type, SceneType::Action);
    }
}
