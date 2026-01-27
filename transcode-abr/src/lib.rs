//! Adaptive Bitrate (ABR) ladder generation with content-aware optimization.
//!
//! This crate provides automatic generation of optimal ABR encoding ladders
//! based on content analysis, replacing static bitrate profiles with
//! content-aware presets for streaming delivery.
//!
//! # Overview
//!
//! Traditional ABR ladders use fixed resolution/bitrate pairs regardless of
//! content complexity. This crate analyzes video content to find
//! Pareto-optimal encoding points that minimize bitrate while maximizing
//! perceptual quality (SSIM/VMAF).
//!
//! # Example
//!
//! ```
//! use transcode_abr::{LadderGenerator, ContentProfile, LadderConfig, Resolution};
//!
//! let profile = ContentProfile {
//!     spatial_complexity: 0.7,
//!     temporal_complexity: 0.4,
//!     motion_intensity: 0.5,
//!     scene_change_rate: 0.02,
//!     grain_level: 0.1,
//!     source_resolution: Resolution { width: 1920, height: 1080 },
//!     source_framerate: 30.0,
//!     duration_secs: 120.0,
//! };
//!
//! let generator = LadderGenerator::new(LadderConfig::default());
//! let ladder = generator.generate(&profile).unwrap();
//!
//! for rung in &ladder.rungs {
//!     println!("{}x{} @ {} kbps (predicted VMAF: {:.1})",
//!         rung.width, rung.height, rung.bitrate_kbps, rung.predicted_vmaf);
//! }
//! ```

#![allow(dead_code)]

mod error;
mod profile;
mod ladder;
mod optimizer;

pub use error::{Error, Result};
pub use profile::{ContentProfile, Resolution, ContentClassification};
pub use ladder::{
    LadderGenerator, LadderConfig, AbrLadder, LadderRung,
    Codec, RateControlMode,
};
pub use optimizer::{ConvexHullOptimizer, QualityPoint, ParetoFront};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_end_to_end_ladder_generation() {
        let profile = ContentProfile {
            spatial_complexity: 0.6,
            temporal_complexity: 0.5,
            motion_intensity: 0.4,
            scene_change_rate: 0.03,
            grain_level: 0.1,
            source_resolution: Resolution { width: 1920, height: 1080 },
            source_framerate: 30.0,
            duration_secs: 120.0,
        };

        let generator = LadderGenerator::new(LadderConfig::default());
        let ladder = generator.generate(&profile).unwrap();

        // Should produce multiple rungs
        assert!(ladder.rungs.len() >= 3);
        assert!(ladder.rungs.len() <= 8);

        // Rungs should be sorted by bitrate ascending
        for window in ladder.rungs.windows(2) {
            assert!(window[0].bitrate_kbps <= window[1].bitrate_kbps);
        }

        // Top rung should not exceed source resolution
        let top = ladder.rungs.last().unwrap();
        assert!(top.width <= 1920);
        assert!(top.height <= 1080);

        // Bottom rung should be significantly lower bitrate
        let bottom = ladder.rungs.first().unwrap();
        assert!(bottom.bitrate_kbps < top.bitrate_kbps);
    }

    #[test]
    fn test_high_complexity_gets_higher_bitrates() {
        let generator = LadderGenerator::new(LadderConfig::default());

        let simple = ContentProfile {
            spatial_complexity: 0.2,
            temporal_complexity: 0.2,
            motion_intensity: 0.1,
            scene_change_rate: 0.01,
            grain_level: 0.0,
            source_resolution: Resolution { width: 1920, height: 1080 },
            source_framerate: 30.0,
            duration_secs: 60.0,
        };

        let complex = ContentProfile {
            spatial_complexity: 0.9,
            temporal_complexity: 0.9,
            motion_intensity: 0.8,
            scene_change_rate: 0.1,
            grain_level: 0.3,
            ..simple.clone()
        };

        let simple_ladder = generator.generate(&simple).unwrap();
        let complex_ladder = generator.generate(&complex).unwrap();

        let simple_max = simple_ladder.rungs.last().unwrap().bitrate_kbps;
        let complex_max = complex_ladder.rungs.last().unwrap().bitrate_kbps;
        assert!(complex_max > simple_max);
    }

    #[test]
    fn test_estimated_savings() {
        let profile = ContentProfile {
            spatial_complexity: 0.4,
            temporal_complexity: 0.3,
            motion_intensity: 0.3,
            scene_change_rate: 0.02,
            grain_level: 0.05,
            source_resolution: Resolution { width: 1920, height: 1080 },
            source_framerate: 24.0,
            duration_secs: 7200.0,
        };

        let generator = LadderGenerator::new(LadderConfig::default());
        let ladder = generator.generate(&profile).unwrap();
        assert!(ladder.estimated_savings_percent > 0.0);
        assert!(ladder.estimated_savings_percent < 60.0);
    }
}
