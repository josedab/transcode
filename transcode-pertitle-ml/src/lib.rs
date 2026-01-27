//! ML-optimized per-title encoding for the transcode library.
//!
//! This crate provides machine learning-based encoding optimization that
//! analyzes video content to predict optimal encoding parameters, achieving
//! significant bitrate savings while maintaining quality targets.
//!
//! # Features
//!
//! - **Content Analysis**: Extract spatial, temporal, and motion complexity features
//! - **Bitrate Prediction**: ML models to predict optimal CRF/bitrate for target quality
//! - **Ladder Generation**: Automatically generate optimal ABR encoding ladders
//! - **Scene-Aware Encoding**: Per-scene encoding recommendations
//!
//! # Example
//!
//! ```rust
//! use transcode_pertitle_ml::{
//!     ContentFeatures, VideoMetadata, ModelInput,
//!     HeuristicPredictor, BitratePredictor,
//!     LadderGenerator, LadderConfig,
//! };
//!
//! // Analyze content features (would come from video analysis)
//! let content = ContentFeatures {
//!     spatial_complexity: 0.6,
//!     temporal_complexity: 0.4,
//!     motion_magnitude: 0.3,
//!     ..Default::default()
//! };
//!
//! let metadata = VideoMetadata {
//!     width: 1920,
//!     height: 1080,
//!     frame_rate: 24.0,
//!     duration: 120.0,
//!     ..Default::default()
//! };
//!
//! // Get encoding prediction
//! let predictor = HeuristicPredictor::new();
//! let input = ModelInput::new(content.clone(), metadata.clone())
//!     .with_target_vmaf(93.0)
//!     .with_codec("h264");
//!
//! let prediction = predictor.predict(&input).unwrap();
//! println!("Recommended CRF: {:.1}", prediction.crf);
//! println!("Predicted bitrate: {} kbps", prediction.bitrate_kbps);
//! println!("Expected VMAF: {:.1}", prediction.predicted_vmaf);
//!
//! // Generate full ABR ladder
//! let generator = LadderGenerator::new(predictor, LadderConfig::streaming());
//! let ladder = generator.generate(&content, &metadata).unwrap();
//!
//! println!("Generated {} rungs:", ladder.rungs.len());
//! for rung in &ladder.rungs {
//!     println!("  {}: {}x{} @ {} kbps (VMAF ~{:.0})",
//!         rung.label, rung.width, rung.height,
//!         rung.bitrate_kbps, rung.expected_vmaf);
//! }
//! println!("Estimated savings: {:.1}%", ladder.savings_percent);
//! ```
//!
//! # Models
//!
//! The crate provides several prediction models:
//!
//! - **HeuristicPredictor**: Rule-based predictor (no ML, always available)
//! - **LinearRegressionModel**: Simple linear model for fast inference
//! - **GradientBoostingModel**: Ensemble model for higher accuracy
//!
//! ## Using Pre-trained Models
//!
//! ```rust
//! use transcode_pertitle_ml::{LinearRegressionModel, BitratePredictor};
//!
//! // Load pre-trained weights
//! let model = LinearRegressionModel::pretrained();
//! println!("Model: {} v{}", model.model_info().name, model.model_info().version);
//! ```
//!
//! # Integration with Transcode Pipeline
//!
//! ```ignore
//! use transcode_pipeline::{Pipeline, Encoder};
//! use transcode_pertitle_ml::{LadderGenerator, HeuristicPredictor};
//!
//! // Analyze input video
//! let analysis = analyze_video("input.mp4")?;
//!
//! // Generate optimized ladder
//! let generator = LadderGenerator::new(HeuristicPredictor::new(), LadderConfig::streaming());
//! let ladder = generator.generate(&analysis.content, &analysis.metadata)?;
//!
//! // Encode each rung
//! for rung in ladder.rungs {
//!     Pipeline::new()
//!         .input("input.mp4")
//!         .scale(rung.width, rung.height)
//!         .encode(Encoder::h264().crf(rung.crf as u32))
//!         .output(format!("output_{}.mp4", rung.label))
//!         .run()?;
//! }
//! ```

#![allow(dead_code)]

mod error;
mod features;
mod ladder;
mod model;
pub mod training;

pub use error::{PerTitleMlError, Result};
pub use features::{
    ContentFeatures, FeatureExtractor, FrameStats, ResolutionCategory, SceneFeatures, SceneType,
    VideoMetadata,
};
pub use ladder::{EncodingLadder, LadderConfig, LadderGenerator, LadderRung};
pub use model::{
    BitratePredictor, EncodingPrediction, GradientBoostingModel, HeuristicPredictor,
    LinearRegressionModel, ModelInfo, ModelInput, SceneRecommendation,
};
pub use training::{
    convex_hull_optimize, train, train_with_cv, TrainingConfig, TrainingDataset, TrainingResult,
    TrainingSample,
};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Analyze a sequence of frames and return content features.
///
/// This is a convenience function that creates a FeatureExtractor
/// and processes the frame statistics.
pub fn analyze_content(frame_stats: &[FrameStats]) -> ContentFeatures {
    let extractor = FeatureExtractor::default();
    extractor.extract_features(frame_stats)
}

/// Quick prediction using the default heuristic model.
///
/// This is a convenience function for simple use cases where
/// you don't need to configure the model.
pub fn quick_predict(
    content: &ContentFeatures,
    metadata: &VideoMetadata,
    target_vmaf: f64,
) -> Result<EncodingPrediction> {
    let predictor = HeuristicPredictor::new();
    let input = ModelInput::new(content.clone(), metadata.clone()).with_target_vmaf(target_vmaf);
    predictor.predict(&input)
}

/// Generate a quick encoding ladder using default settings.
pub fn quick_ladder(
    content: &ContentFeatures,
    metadata: &VideoMetadata,
) -> Result<EncodingLadder> {
    let predictor = HeuristicPredictor::new();
    let config = LadderConfig::streaming();
    let generator = LadderGenerator::new(predictor, config);
    generator.generate(content, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_predict() {
        let content = ContentFeatures::default();
        let metadata = VideoMetadata::default();

        let prediction = quick_predict(&content, &metadata, 93.0).unwrap();
        assert!(prediction.crf > 0.0);
        assert!(prediction.bitrate_kbps > 0);
    }

    #[test]
    fn test_quick_ladder() {
        let content = ContentFeatures::default();
        let metadata = VideoMetadata::default();

        let ladder = quick_ladder(&content, &metadata).unwrap();
        assert!(!ladder.rungs.is_empty());
    }

    #[test]
    fn test_analyze_content() {
        let stats = vec![
            FrameStats { frame_number: 0, mean: 128.0, variance: 50.0, ..Default::default() },
            FrameStats { frame_number: 1, mean: 130.0, variance: 52.0, ..Default::default() },
            FrameStats { frame_number: 2, mean: 125.0, variance: 48.0, ..Default::default() },
        ];

        let features = analyze_content(&stats);
        assert!(features.spatial_complexity >= 0.0 && features.spatial_complexity <= 1.0);
    }
}
