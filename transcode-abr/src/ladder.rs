//! ABR ladder generation from content profiles.

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::error::{Error, Result};
use crate::optimizer::{ConvexHullOptimizer, QualityPoint};
use crate::profile::{ContentProfile, Resolution};

/// Codec selection for encoding ladder rungs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Codec {
    #[default]
    H264,
    H265,
    Av1,
    Vp9,
}

/// Rate control mode for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RateControlMode {
    #[default]
    Cbr,
    Vbr,
    Crf,
    Constrained,
}

/// Configuration for the ladder generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LadderConfig {
    pub codec: Codec,
    pub rate_control: RateControlMode,
    pub min_bitrate_kbps: u32,
    pub max_bitrate_kbps: u32,
    pub target_vmaf: f64,
    pub min_vmaf: f64,
    pub max_rungs: usize,
    pub min_rungs: usize,
    /// Include audio bitrate in calculations.
    pub audio_bitrate_kbps: u32,
    /// Segment duration for streaming (seconds).
    pub segment_duration_secs: f64,
}

impl Default for LadderConfig {
    fn default() -> Self {
        Self {
            codec: Codec::H264,
            rate_control: RateControlMode::Cbr,
            min_bitrate_kbps: 200,
            max_bitrate_kbps: 12_000,
            target_vmaf: 93.0,
            min_vmaf: 70.0,
            max_rungs: 7,
            min_rungs: 3,
            audio_bitrate_kbps: 128,
            segment_duration_secs: 6.0,
        }
    }
}

/// A single rung in the ABR ladder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LadderRung {
    pub width: u32,
    pub height: u32,
    pub bitrate_kbps: u32,
    pub framerate: f64,
    pub codec: Codec,
    pub predicted_vmaf: f64,
    pub predicted_ssim: f64,
}

/// The complete ABR ladder with all encoding variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbrLadder {
    pub rungs: Vec<LadderRung>,
    pub codec: Codec,
    pub estimated_savings_percent: f64,
    pub content_complexity: f64,
}

/// Generates optimal ABR encoding ladders from content profiles.
pub struct LadderGenerator {
    config: LadderConfig,
    optimizer: ConvexHullOptimizer,
}

impl LadderGenerator {
    pub fn new(config: LadderConfig) -> Self {
        Self {
            optimizer: ConvexHullOptimizer::new(config.min_vmaf, config.target_vmaf),
            config,
        }
    }

    /// Generate an ABR ladder for the given content profile.
    pub fn generate(&self, profile: &ContentProfile) -> Result<AbrLadder> {
        if profile.source_framerate <= 0.0 {
            return Err(Error::InvalidProfile {
                message: "Frame rate must be positive".into(),
            });
        }

        let complexity = profile.complexity_score();
        let classification = profile.classify();
        let factor = classification.bitrate_factor();

        debug!(
            complexity = complexity,
            classification = ?classification,
            factor = factor,
            "Content analyzed"
        );

        // Generate candidate quality points across resolutions and bitrates
        let candidates = self.generate_candidates(profile, factor);

        // Find Pareto-optimal points via convex hull
        let front = self.optimizer.find_pareto_front(&candidates);
        let selected = front.select_rungs(self.config.min_rungs, self.config.max_rungs);

        if selected.is_empty() {
            return Err(Error::Optimization {
                message: "Could not find valid encoding points".into(),
            });
        }

        // Build rungs from selected points
        let mut rungs: Vec<LadderRung> = selected
            .iter()
            .map(|point| LadderRung {
                width: point.width,
                height: point.height,
                bitrate_kbps: point.bitrate_kbps,
                framerate: point.framerate,
                codec: self.config.codec,
                predicted_vmaf: point.vmaf,
                predicted_ssim: point.ssim,
            })
            .collect();

        rungs.sort_by_key(|r| r.bitrate_kbps);

        // Estimate savings vs static ladder
        let static_bitrate: u32 = self.static_ladder_bitrate(profile);
        let optimized_bitrate: u32 = rungs.iter().map(|r| r.bitrate_kbps).sum();
        let savings = if static_bitrate > 0 {
            ((static_bitrate as f64 - optimized_bitrate as f64) / static_bitrate as f64) * 100.0
        } else {
            0.0
        };

        Ok(AbrLadder {
            rungs,
            codec: self.config.codec,
            estimated_savings_percent: savings.max(0.0),
            content_complexity: complexity,
        })
    }

    fn generate_candidates(&self, profile: &ContentProfile, factor: f64) -> Vec<QualityPoint> {
        let mut candidates = Vec::new();
        let resolutions = Resolution::standard_ladder();

        for res in &resolutions {
            if res.pixels() > profile.source_resolution.pixels() {
                continue;
            }

            // Scale factor relative to source
            let scale = (res.pixels() as f64 / profile.source_resolution.pixels() as f64).sqrt();

            // Base bitrate for this resolution (empirical model)
            let base_kbps = self.base_bitrate_for_resolution(res, profile.source_framerate);

            // Test multiple bitrate points around the base
            for multiplier in &[0.5, 0.75, 1.0, 1.25, 1.5] {
                let bitrate = (base_kbps as f64 * multiplier * factor) as u32;
                if bitrate < self.config.min_bitrate_kbps
                    || bitrate > self.config.max_bitrate_kbps
                {
                    continue;
                }

                // Predict quality (simplified model based on bits-per-pixel)
                let bpp = bitrate as f64 * 1000.0
                    / (res.pixels() as f64 * profile.source_framerate);
                let vmaf = self.predict_vmaf(bpp, profile.complexity_score(), scale);
                let ssim = self.predict_ssim(bpp, profile.complexity_score());

                if vmaf >= self.config.min_vmaf {
                    candidates.push(QualityPoint {
                        width: res.width,
                        height: res.height,
                        bitrate_kbps: bitrate,
                        framerate: profile.source_framerate,
                        vmaf,
                        ssim,
                    });
                }
            }
        }

        candidates
    }

    fn base_bitrate_for_resolution(&self, res: &Resolution, fps: f64) -> u32 {
        let pixels = res.pixels() as f64;
        // Empirical: ~1.0 bits per pixel at 30fps for medium quality
        let bpp_target = 1.0 * (fps / 30.0).sqrt();
        (pixels * bpp_target / 1000.0) as u32
    }

    fn predict_vmaf(&self, bpp: f64, complexity: f64, scale: f64) -> f64 {
        // Simplified VMAF prediction model
        // Higher bpp = higher quality; higher complexity = harder to encode
        let base = 100.0 * (1.0 - (-bpp * 500.0).exp());
        let complexity_penalty = complexity * 10.0;
        let scale_penalty = (1.0 - scale) * 5.0;
        (base - complexity_penalty - scale_penalty).clamp(0.0, 100.0)
    }

    fn predict_ssim(&self, bpp: f64, complexity: f64) -> f64 {
        let base = 1.0 - (-bpp * 300.0).exp();
        let penalty = complexity * 0.03;
        (base - penalty).clamp(0.0, 1.0)
    }

    fn static_ladder_bitrate(&self, profile: &ContentProfile) -> u32 {
        // Traditional fixed ladder total bitrate
        let resolutions = Resolution::standard_ladder();
        let mut total = 0u32;
        for res in &resolutions {
            if res.pixels() > profile.source_resolution.pixels() {
                continue;
            }
            total += self.base_bitrate_for_resolution(res, profile.source_framerate);
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::ContentProfile;

    fn test_profile() -> ContentProfile {
        ContentProfile {
            spatial_complexity: 0.5,
            temporal_complexity: 0.5,
            motion_intensity: 0.4,
            scene_change_rate: 0.03,
            grain_level: 0.1,
            source_resolution: Resolution { width: 1920, height: 1080 },
            source_framerate: 30.0,
            duration_secs: 120.0,
        }
    }

    #[test]
    fn test_generate_produces_valid_ladder() {
        let gen = LadderGenerator::new(LadderConfig::default());
        let ladder = gen.generate(&test_profile()).unwrap();
        assert!(!ladder.rungs.is_empty());

        for rung in &ladder.rungs {
            assert!(rung.bitrate_kbps >= 200);
            assert!(rung.predicted_vmaf >= 70.0);
            assert!(rung.width <= 1920);
        }
    }

    #[test]
    fn test_invalid_framerate() {
        let mut profile = test_profile();
        profile.source_framerate = 0.0;
        let gen = LadderGenerator::new(LadderConfig::default());
        assert!(gen.generate(&profile).is_err());
    }

    #[test]
    fn test_rungs_sorted_by_bitrate() {
        let gen = LadderGenerator::new(LadderConfig::default());
        let ladder = gen.generate(&test_profile()).unwrap();
        for w in ladder.rungs.windows(2) {
            assert!(w[0].bitrate_kbps <= w[1].bitrate_kbps);
        }
    }
}
