//! Content analysis profile for ABR optimization.

use serde::{Deserialize, Serialize};

/// Resolution of a video source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    pub fn pixels(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Standard resolution presets for ABR ladders.
    pub fn standard_ladder() -> Vec<Resolution> {
        vec![
            Resolution { width: 426, height: 240 },
            Resolution { width: 640, height: 360 },
            Resolution { width: 854, height: 480 },
            Resolution { width: 1280, height: 720 },
            Resolution { width: 1920, height: 1080 },
            Resolution { width: 2560, height: 1440 },
            Resolution { width: 3840, height: 2160 },
        ]
    }
}

/// Content characteristics extracted from video analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentProfile {
    /// Spatial complexity (0.0 = flat, 1.0 = highly detailed). From edge/texture analysis.
    pub spatial_complexity: f64,
    /// Temporal complexity (0.0 = static, 1.0 = rapid changes). From frame differencing.
    pub temporal_complexity: f64,
    /// Motion intensity (0.0 = still, 1.0 = extreme motion).
    pub motion_intensity: f64,
    /// Rate of scene changes (fraction of total frames).
    pub scene_change_rate: f64,
    /// Film grain / noise level (0.0 = clean, 1.0 = heavy grain).
    pub grain_level: f64,
    /// Source resolution.
    pub source_resolution: Resolution,
    /// Source frame rate.
    pub source_framerate: f64,
    /// Duration in seconds.
    pub duration_secs: f64,
}

impl ContentProfile {
    /// Compute an overall complexity score (0.0 – 1.0).
    pub fn complexity_score(&self) -> f64 {
        let weighted = self.spatial_complexity * 0.35
            + self.temporal_complexity * 0.30
            + self.motion_intensity * 0.20
            + self.grain_level * 0.15;
        weighted.clamp(0.0, 1.0)
    }

    pub fn classify(&self) -> ContentClassification {
        let score = self.complexity_score();
        if score < 0.25 {
            ContentClassification::Simple
        } else if score < 0.50 {
            ContentClassification::Moderate
        } else if score < 0.75 {
            ContentClassification::Complex
        } else {
            ContentClassification::Extreme
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentClassification {
    Simple,
    Moderate,
    Complex,
    Extreme,
}

impl ContentClassification {
    /// Bitrate multiplier relative to the "moderate" baseline.
    pub fn bitrate_factor(&self) -> f64 {
        match self {
            Self::Simple => 0.65,
            Self::Moderate => 1.0,
            Self::Complex => 1.40,
            Self::Extreme => 1.85,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_score_range() {
        let profile = ContentProfile {
            spatial_complexity: 0.5,
            temporal_complexity: 0.5,
            motion_intensity: 0.5,
            scene_change_rate: 0.03,
            grain_level: 0.5,
            source_resolution: Resolution { width: 1920, height: 1080 },
            source_framerate: 30.0,
            duration_secs: 60.0,
        };
        let score = profile.complexity_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_classification_ordering() {
        assert!(ContentClassification::Simple.bitrate_factor()
            < ContentClassification::Moderate.bitrate_factor());
        assert!(ContentClassification::Moderate.bitrate_factor()
            < ContentClassification::Complex.bitrate_factor());
        assert!(ContentClassification::Complex.bitrate_factor()
            < ContentClassification::Extreme.bitrate_factor());
    }

    #[test]
    fn test_resolution_pixels() {
        let r = Resolution { width: 1920, height: 1080 };
        assert_eq!(r.pixels(), 2_073_600);
    }
}
