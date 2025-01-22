//! ML model for bitrate prediction.

use crate::error::{PerTitleMlError, Result};
use crate::features::{ContentFeatures, ResolutionCategory, SceneType, VideoMetadata};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Encoding parameters predicted by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingPrediction {
    /// Recommended CRF value.
    pub crf: f64,
    /// Recommended bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Minimum bitrate for acceptable quality.
    pub min_bitrate_kbps: u32,
    /// Maximum useful bitrate.
    pub max_bitrate_kbps: u32,
    /// Predicted VMAF score at recommended bitrate.
    pub predicted_vmaf: f64,
    /// Confidence of prediction (0.0-1.0).
    pub confidence: f64,
    /// Scene-level recommendations if available.
    pub scene_recommendations: Vec<SceneRecommendation>,
}

impl EncodingPrediction {
    /// Get bitrate savings compared to reference.
    pub fn bitrate_savings(&self, reference_bitrate: u32) -> f64 {
        if reference_bitrate == 0 {
            return 0.0;
        }
        1.0 - (self.bitrate_kbps as f64 / reference_bitrate as f64)
    }
}

/// Per-scene encoding recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneRecommendation {
    /// Scene start time in seconds.
    pub start_time: f64,
    /// Scene end time in seconds.
    pub end_time: f64,
    /// Scene type.
    pub scene_type: SceneType,
    /// CRF adjustment relative to base.
    pub crf_delta: i32,
    /// Bitrate multiplier.
    pub bitrate_multiplier: f64,
}

/// Model input combining all features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInput {
    pub content: ContentFeatures,
    pub metadata: VideoMetadata,
    pub target_vmaf: f64,
    pub codec: String,
}

impl ModelInput {
    /// Create new model input.
    pub fn new(content: ContentFeatures, metadata: VideoMetadata) -> Self {
        Self {
            content,
            metadata,
            target_vmaf: 93.0, // Netflix-quality default
            codec: "h264".to_string(),
        }
    }

    /// Set target VMAF.
    pub fn with_target_vmaf(mut self, vmaf: f64) -> Self {
        self.target_vmaf = vmaf;
        self
    }

    /// Set codec.
    pub fn with_codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = codec.into();
        self
    }

    /// Convert to feature vector for model inference.
    pub fn to_vector(&self) -> Vec<f64> {
        let mut features = self.content.to_vector();
        features.extend(self.metadata.to_vector());
        features.push(self.target_vmaf / 100.0);
        features.push(codec_to_factor(&self.codec));
        features
    }
}

/// Bitrate predictor model.
pub trait BitratePredictor: Send + Sync {
    /// Predict encoding parameters.
    fn predict(&self, input: &ModelInput) -> Result<EncodingPrediction>;

    /// Get model name/version.
    fn model_info(&self) -> ModelInfo;
}

/// Model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub trained_on: String,
    pub accuracy_mae: f64,
}

/// Heuristic-based predictor (rule-based, no ML).
///
/// This serves as a baseline and fallback when ML model is not available.
#[derive(Debug, Clone)]
pub struct HeuristicPredictor {
    /// Base CRF values per resolution.
    base_crf: HashMap<ResolutionCategory, f64>,
    /// Base bitrates per resolution (kbps).
    base_bitrates: HashMap<ResolutionCategory, u32>,
    /// VMAF-to-CRF mapping coefficients.
    vmaf_crf_slope: f64,
}

impl Default for HeuristicPredictor {
    fn default() -> Self {
        let mut base_crf = HashMap::new();
        base_crf.insert(ResolutionCategory::Low, 28.0);
        base_crf.insert(ResolutionCategory::Sd480P, 26.0);
        base_crf.insert(ResolutionCategory::Hd720P, 24.0);
        base_crf.insert(ResolutionCategory::Fhd1080P, 23.0);
        base_crf.insert(ResolutionCategory::Uhd4K, 22.0);

        let mut base_bitrates = HashMap::new();
        base_bitrates.insert(ResolutionCategory::Low, 500);
        base_bitrates.insert(ResolutionCategory::Sd480P, 1500);
        base_bitrates.insert(ResolutionCategory::Hd720P, 3000);
        base_bitrates.insert(ResolutionCategory::Fhd1080P, 5000);
        base_bitrates.insert(ResolutionCategory::Uhd4K, 15000);

        Self {
            base_crf,
            base_bitrates,
            vmaf_crf_slope: -0.5,
        }
    }
}

impl HeuristicPredictor {
    /// Create a new heuristic predictor.
    pub fn new() -> Self {
        Self::default()
    }
}

impl BitratePredictor for HeuristicPredictor {
    fn predict(&self, input: &ModelInput) -> Result<EncodingPrediction> {
        let resolution = input.metadata.resolution_category();

        // Get base values
        let base_crf = *self.base_crf.get(&resolution).unwrap_or(&24.0);
        let base_bitrate = *self.base_bitrates.get(&resolution).unwrap_or(&5000);

        // Adjust for content complexity
        let complexity = input.content.complexity_score();
        let complexity_adjustment = (complexity - 0.5) * 4.0; // -2 to +2

        // Adjust for target VMAF
        let vmaf_adjustment = (input.target_vmaf - 93.0) * self.vmaf_crf_slope;

        // Adjust for codec efficiency
        let codec_factor = codec_to_factor(&input.codec);
        let codec_bitrate_multiplier = 1.0 / codec_factor;

        // Calculate final CRF
        let crf = (base_crf - complexity_adjustment + vmaf_adjustment).clamp(15.0, 35.0);

        // Estimate bitrate from CRF using empirical model
        // bitrate ≈ base_bitrate * 2^((base_crf - crf) / 6)
        let bitrate_factor = 2.0_f64.powf((base_crf - crf) / 6.0);
        let bitrate_kbps =
            ((base_bitrate as f64 * bitrate_factor * codec_bitrate_multiplier) as u32).max(100);

        // Calculate min/max
        let (range_min, range_max) = resolution.bitrate_range_kbps();
        let min_bitrate_kbps = (bitrate_kbps as f64 * 0.6) as u32;
        let max_bitrate_kbps = (bitrate_kbps as f64 * 1.5) as u32;

        // Predict VMAF (rough estimate)
        let predicted_vmaf = estimate_vmaf(bitrate_kbps, &input.metadata, &input.content);

        Ok(EncodingPrediction {
            crf,
            bitrate_kbps: bitrate_kbps.clamp(range_min, range_max),
            min_bitrate_kbps: min_bitrate_kbps.clamp(range_min, range_max),
            max_bitrate_kbps: max_bitrate_kbps.clamp(range_min, range_max),
            predicted_vmaf,
            confidence: 0.7, // Heuristic has moderate confidence
            scene_recommendations: Vec::new(),
        })
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "HeuristicPredictor".to_string(),
            version: "1.0.0".to_string(),
            trained_on: "N/A (rule-based)".to_string(),
            accuracy_mae: 15.0, // Rough estimate
        }
    }
}

/// Linear regression model trained on encoding data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionModel {
    /// Weights for CRF prediction.
    crf_weights: Vec<f64>,
    /// Bias for CRF prediction.
    crf_bias: f64,
    /// Weights for bitrate prediction.
    bitrate_weights: Vec<f64>,
    /// Bias for bitrate prediction.
    bitrate_bias: f64,
    /// Model info.
    info: ModelInfo,
}

impl LinearRegressionModel {
    /// Create a new model with given weights.
    pub fn new(
        crf_weights: Vec<f64>,
        crf_bias: f64,
        bitrate_weights: Vec<f64>,
        bitrate_bias: f64,
    ) -> Self {
        Self {
            crf_weights,
            crf_bias,
            bitrate_weights,
            bitrate_bias,
            info: ModelInfo {
                name: "LinearRegressionModel".to_string(),
                version: "1.0.0".to_string(),
                trained_on: "synthetic".to_string(),
                accuracy_mae: 8.0,
            },
        }
    }

    /// Create a default model with pre-trained weights.
    ///
    /// These weights are illustrative. In production, would be trained
    /// on actual encoding data.
    pub fn pretrained() -> Self {
        // Feature order: content (10) + metadata (5) + target_vmaf + codec = 17
        let crf_weights = vec![
            -2.0,  // spatial_complexity
            -1.5,  // temporal_complexity
            -1.0,  // motion_magnitude
            0.5,   // scene_change_rate
            -0.5,  // color_variance
            -0.8,  // edge_density
            -1.2,  // texture_complexity
            0.3,   // dark_ratio
            -0.7,  // detail_ratio
            0.2,   // grain_level
            -3.0,  // width (normalized)
            -3.0,  // height (normalized)
            0.2,   // frame_rate
            -0.5,  // bit_depth
            -1.0,  // is_hdr
            -8.0,  // target_vmaf (normalized)
            2.0,   // codec_factor
        ];

        let bitrate_weights = vec![
            800.0,   // spatial_complexity
            600.0,   // temporal_complexity
            400.0,   // motion_magnitude
            -100.0,  // scene_change_rate
            200.0,   // color_variance
            300.0,   // edge_density
            500.0,   // texture_complexity
            -200.0,  // dark_ratio
            400.0,   // detail_ratio
            100.0,   // grain_level
            4000.0,  // width
            4000.0,  // height
            50.0,    // frame_rate
            200.0,   // bit_depth
            2000.0,  // is_hdr
            50.0,    // target_vmaf
            -2000.0, // codec_factor (more efficient = lower bitrate)
        ];

        Self::new(crf_weights, 28.0, bitrate_weights, 1000.0)
    }

    /// Load model from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| PerTitleMlError::ModelError(e.to_string()))
    }

    /// Save model to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| e.into())
    }
}

impl BitratePredictor for LinearRegressionModel {
    fn predict(&self, input: &ModelInput) -> Result<EncodingPrediction> {
        let features = input.to_vector();

        if features.len() != self.crf_weights.len() {
            return Err(PerTitleMlError::InvalidFeatures(format!(
                "Expected {} features, got {}",
                self.crf_weights.len(),
                features.len()
            )));
        }

        // Linear prediction: y = Σ(w_i * x_i) + b
        let crf: f64 = features
            .iter()
            .zip(self.crf_weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.crf_bias;

        let bitrate: f64 = features
            .iter()
            .zip(self.bitrate_weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.bitrate_bias;

        let resolution = input.metadata.resolution_category();
        let (range_min, range_max) = resolution.bitrate_range_kbps();

        let crf = crf.clamp(15.0, 35.0);
        let bitrate_kbps = (bitrate as u32).clamp(range_min, range_max);

        let predicted_vmaf = estimate_vmaf(bitrate_kbps, &input.metadata, &input.content);

        Ok(EncodingPrediction {
            crf,
            bitrate_kbps,
            min_bitrate_kbps: (bitrate_kbps as f64 * 0.7) as u32,
            max_bitrate_kbps: (bitrate_kbps as f64 * 1.3) as u32,
            predicted_vmaf,
            confidence: 0.85,
            scene_recommendations: Vec::new(),
        })
    }

    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }
}

/// Gradient boosting model for more accurate predictions.
#[derive(Debug, Clone)]
pub struct GradientBoostingModel {
    /// Ensemble of decision trees (simplified representation).
    trees: Vec<DecisionTree>,
    /// Learning rate.
    learning_rate: f64,
    /// Model info.
    info: ModelInfo,
}

impl Default for GradientBoostingModel {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientBoostingModel {
    /// Create a new gradient boosting model.
    pub fn new() -> Self {
        // Create a simple ensemble with pre-defined trees
        let trees = vec![
            DecisionTree::resolution_tree(),
            DecisionTree::complexity_tree(),
            DecisionTree::motion_tree(),
        ];

        Self {
            trees,
            learning_rate: 0.1,
            info: ModelInfo {
                name: "GradientBoostingModel".to_string(),
                version: "1.0.0".to_string(),
                trained_on: "synthetic".to_string(),
                accuracy_mae: 5.0,
            },
        }
    }
}

impl BitratePredictor for GradientBoostingModel {
    fn predict(&self, input: &ModelInput) -> Result<EncodingPrediction> {
        // Start with base prediction
        let base_crf = 23.0;
        let base_bitrate = 5000.0;

        // Accumulate tree predictions
        let crf_adjustment: f64 = self
            .trees
            .iter()
            .map(|tree| tree.predict_crf(input) * self.learning_rate)
            .sum();

        let bitrate_adjustment: f64 = self
            .trees
            .iter()
            .map(|tree| tree.predict_bitrate(input) * self.learning_rate)
            .sum();

        let crf = (base_crf + crf_adjustment).clamp(15.0, 35.0);
        let bitrate_kbps = ((base_bitrate + bitrate_adjustment) as u32).max(100);

        let resolution = input.metadata.resolution_category();
        let (range_min, range_max) = resolution.bitrate_range_kbps();

        let predicted_vmaf = estimate_vmaf(bitrate_kbps, &input.metadata, &input.content);

        Ok(EncodingPrediction {
            crf,
            bitrate_kbps: bitrate_kbps.clamp(range_min, range_max),
            min_bitrate_kbps: (bitrate_kbps as f64 * 0.65) as u32,
            max_bitrate_kbps: (bitrate_kbps as f64 * 1.35) as u32,
            predicted_vmaf,
            confidence: 0.9,
            scene_recommendations: Vec::new(),
        })
    }

    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }
}

/// Simplified decision tree for ensemble.
#[derive(Debug, Clone)]
struct DecisionTree {
    name: String,
}

impl DecisionTree {
    fn resolution_tree() -> Self {
        Self { name: "resolution".to_string() }
    }

    fn complexity_tree() -> Self {
        Self { name: "complexity".to_string() }
    }

    fn motion_tree() -> Self {
        Self { name: "motion".to_string() }
    }

    fn predict_crf(&self, input: &ModelInput) -> f64 {
        match self.name.as_str() {
            "resolution" => {
                let pixels = input.metadata.total_pixels() as f64;
                // Higher resolution -> lower CRF needed
                if pixels > 8_000_000.0 { -2.0 }
                else if pixels > 2_000_000.0 { -1.0 }
                else { 1.0 }
            }
            "complexity" => {
                let c = input.content.complexity_score();
                // Higher complexity -> lower CRF needed
                (c - 0.5) * -4.0
            }
            "motion" => {
                let m = input.content.motion_magnitude;
                // Higher motion -> lower CRF needed
                (m - 0.5) * -2.0
            }
            _ => 0.0,
        }
    }

    fn predict_bitrate(&self, input: &ModelInput) -> f64 {
        match self.name.as_str() {
            "resolution" => {
                let pixels = input.metadata.total_pixels() as f64;
                // Scale bitrate with pixels
                (pixels / 2_073_600.0 - 1.0) * 3000.0 // 1080p as baseline
            }
            "complexity" => {
                let c = input.content.complexity_score();
                (c - 0.5) * 2000.0
            }
            "motion" => {
                let m = input.content.motion_magnitude;
                (m - 0.5) * 1000.0
            }
            _ => 0.0,
        }
    }
}

// Helper functions

fn codec_to_factor(codec: &str) -> f64 {
    match codec.to_lowercase().as_str() {
        "av1" => 1.4,      // Most efficient
        "hevc" | "h265" => 1.3,
        "vp9" => 1.25,
        "h264" | "avc" => 1.0, // Baseline
        "vp8" => 0.9,
        "mpeg2" => 0.6,
        _ => 1.0,
    }
}

fn estimate_vmaf(bitrate_kbps: u32, metadata: &VideoMetadata, content: &ContentFeatures) -> f64 {
    // Rough VMAF estimation based on bitrate and content
    let pixels = metadata.total_pixels() as f64;
    let bpp = (bitrate_kbps as f64 * 1000.0) / (pixels * metadata.frame_rate);

    // Higher bpp = higher quality
    let base_vmaf = 70.0 + (bpp * 500.0).min(25.0);

    // Adjust for content complexity (complex content harder to encode)
    let complexity_penalty = content.complexity_score() * 5.0;

    (base_vmaf - complexity_penalty).clamp(40.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_predictor() {
        let predictor = HeuristicPredictor::new();
        let input = ModelInput::new(ContentFeatures::default(), VideoMetadata::default());

        let prediction = predictor.predict(&input).unwrap();
        assert!(prediction.crf >= 15.0 && prediction.crf <= 35.0);
        assert!(prediction.bitrate_kbps > 0);
        assert!(prediction.predicted_vmaf >= 40.0);
    }

    #[test]
    fn test_linear_model() {
        let model = LinearRegressionModel::pretrained();
        let input = ModelInput::new(ContentFeatures::default(), VideoMetadata::default());

        let prediction = model.predict(&input).unwrap();
        assert!(prediction.crf >= 15.0 && prediction.crf <= 35.0);
    }

    #[test]
    fn test_gradient_boosting_model() {
        let model = GradientBoostingModel::new();
        let input = ModelInput::new(ContentFeatures::default(), VideoMetadata::default());

        let prediction = model.predict(&input).unwrap();
        assert!(prediction.crf >= 15.0 && prediction.crf <= 35.0);
        assert!(prediction.confidence > 0.8);
    }

    #[test]
    fn test_complexity_affects_prediction() {
        let predictor = HeuristicPredictor::new();

        let simple_content = ContentFeatures {
            spatial_complexity: 0.1,
            temporal_complexity: 0.1,
            ..Default::default()
        };
        let complex_content = ContentFeatures {
            spatial_complexity: 0.9,
            temporal_complexity: 0.9,
            ..Default::default()
        };

        let simple_input = ModelInput::new(simple_content, VideoMetadata::default());
        let complex_input = ModelInput::new(complex_content, VideoMetadata::default());

        let simple_pred = predictor.predict(&simple_input).unwrap();
        let complex_pred = predictor.predict(&complex_input).unwrap();

        // Complex content should need higher bitrate
        assert!(complex_pred.bitrate_kbps > simple_pred.bitrate_kbps);
    }

    #[test]
    fn test_codec_efficiency() {
        assert!(codec_to_factor("av1") > codec_to_factor("h264"));
        assert!(codec_to_factor("hevc") > codec_to_factor("h264"));
    }
}
