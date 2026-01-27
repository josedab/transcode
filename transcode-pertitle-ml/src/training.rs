//! Model training pipeline with cross-validation and convex hull optimization.

use crate::error::{PerTitleMlError, Result};
use crate::features::{ContentFeatures, VideoMetadata};
use crate::model::{LinearRegressionModel, ModelInput};
use serde::{Deserialize, Serialize};

/// A single training sample: features + observed encoding results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub content: ContentFeatures,
    pub metadata: VideoMetadata,
    pub codec: String,
    pub target_vmaf: f64,
    /// Observed optimal CRF for target quality.
    pub observed_crf: f64,
    /// Observed bitrate at optimal CRF (kbps).
    pub observed_bitrate_kbps: u32,
}

impl TrainingSample {
    pub fn to_input(&self) -> ModelInput {
        ModelInput::new(self.content.clone(), self.metadata.clone())
            .with_target_vmaf(self.target_vmaf)
            .with_codec(&self.codec)
    }
}

/// Training dataset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingDataset {
    pub samples: Vec<TrainingSample>,
}

impl TrainingDataset {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    pub fn add(&mut self, sample: TrainingSample) {
        self.samples.push(sample);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Split dataset into training and validation sets.
    pub fn split(&self, train_ratio: f64) -> (TrainingDataset, TrainingDataset) {
        let split_idx = (self.samples.len() as f64 * train_ratio) as usize;
        let (train, val) = self.samples.split_at(split_idx.min(self.samples.len()));
        (
            TrainingDataset { samples: train.to_vec() },
            TrainingDataset { samples: val.to_vec() },
        )
    }

    /// Generate a synthetic dataset for testing/bootstrapping.
    pub fn synthetic(count: usize) -> Self {
        let mut samples = Vec::with_capacity(count);
        for i in 0..count {
            let complexity = (i as f64 / count as f64).clamp(0.0, 1.0);
            let resolution_factor = if i % 3 == 0 {
                1.0
            } else if i % 3 == 1 {
                0.5
            } else {
                0.25
            };

            let content = ContentFeatures {
                spatial_complexity: complexity,
                temporal_complexity: complexity * 0.8,
                motion_magnitude: complexity * 0.6,
                edge_density: 0.3 + complexity * 0.4,
                detail_ratio: 0.2 + complexity * 0.5,
                ..Default::default()
            };

            let metadata = VideoMetadata {
                width: (1920.0 * resolution_factor) as u32,
                height: (1080.0 * resolution_factor) as u32,
                frame_rate: 24.0,
                ..Default::default()
            };

            // Simulate observed encoding results
            let base_crf = 23.0 - complexity * 4.0 + (1.0 - resolution_factor) * 2.0;
            let base_bitrate = (5000.0 * resolution_factor * (0.8 + complexity * 0.4)) as u32;

            samples.push(TrainingSample {
                content,
                metadata,
                codec: "h264".to_string(),
                target_vmaf: 93.0,
                observed_crf: base_crf.clamp(18.0, 30.0),
                observed_bitrate_kbps: base_bitrate.max(200),
            });
        }
        Self { samples }
    }
}

/// Configuration for training.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate for gradient updates.
    pub learning_rate: f64,
    /// Number of training iterations.
    pub max_iterations: usize,
    /// L2 regularization strength.
    pub l2_lambda: f64,
    /// Convergence threshold.
    pub convergence_threshold: f64,
    /// Number of cross-validation folds.
    pub cv_folds: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            max_iterations: 1000,
            l2_lambda: 0.01,
            convergence_threshold: 1e-6,
            cv_folds: 5,
        }
    }
}

/// Training result with metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Trained model.
    pub model: LinearRegressionModel,
    /// Mean absolute error on training set.
    pub train_mae_crf: f64,
    pub train_mae_bitrate: f64,
    /// Mean absolute error on validation set (if available).
    pub val_mae_crf: Option<f64>,
    pub val_mae_bitrate: Option<f64>,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Final loss value.
    pub final_loss: f64,
}

/// Train a LinearRegressionModel using ordinary least squares with L2 regularization.
pub fn train(dataset: &TrainingDataset, config: &TrainingConfig) -> Result<TrainingResult> {
    if dataset.is_empty() {
        return Err(PerTitleMlError::ModelError("Empty training dataset".into()));
    }

    // Extract feature matrix and targets
    let n = dataset.len();
    let features: Vec<Vec<f64>> = dataset
        .samples
        .iter()
        .map(|s| s.to_input().to_vector())
        .collect();
    let dim = features[0].len();

    let crf_targets: Vec<f64> = dataset.samples.iter().map(|s| s.observed_crf).collect();
    let bitrate_targets: Vec<f64> = dataset
        .samples
        .iter()
        .map(|s| s.observed_bitrate_kbps as f64)
        .collect();

    // Initialize weights
    let mut crf_weights = vec![0.0; dim];
    let mut crf_bias = 23.0; // Start near typical CRF
    let mut bitrate_weights = vec![0.0; dim];
    let mut bitrate_bias = 3000.0; // Start near typical bitrate

    let mut prev_loss = f64::MAX;

    // Gradient descent with L2 regularization
    let mut final_iter = config.max_iterations;
    for iter in 0..config.max_iterations {
        let mut crf_grad = vec![0.0; dim];
        let mut crf_bias_grad = 0.0;
        let mut bitrate_grad = vec![0.0; dim];
        let mut bitrate_bias_grad = 0.0;
        let mut total_loss = 0.0;

        for i in 0..n {
            // CRF prediction
            let crf_pred: f64 = features[i]
                .iter()
                .zip(crf_weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f64>()
                + crf_bias;
            let crf_err = crf_pred - crf_targets[i];

            // Bitrate prediction
            let br_pred: f64 = features[i]
                .iter()
                .zip(bitrate_weights.iter())
                .map(|(x, w)| x * w)
                .sum::<f64>()
                + bitrate_bias;
            let br_err = br_pred - bitrate_targets[i];

            total_loss += crf_err * crf_err + (br_err / 1000.0) * (br_err / 1000.0);

            // Accumulate gradients
            for j in 0..dim {
                crf_grad[j] += crf_err * features[i][j];
                bitrate_grad[j] += br_err * features[i][j];
            }
            crf_bias_grad += crf_err;
            bitrate_bias_grad += br_err;
        }

        total_loss /= n as f64;

        // Apply gradients with L2 regularization
        let lr = config.learning_rate;
        let lambda = config.l2_lambda;
        for j in 0..dim {
            crf_weights[j] -= lr * (crf_grad[j] / n as f64 + lambda * crf_weights[j]);
            bitrate_weights[j] -= lr * (bitrate_grad[j] / n as f64 + lambda * bitrate_weights[j]);
        }
        crf_bias -= lr * crf_bias_grad / n as f64;
        bitrate_bias -= lr * bitrate_bias_grad / n as f64;

        // Check convergence
        if (prev_loss - total_loss).abs() < config.convergence_threshold {
            final_iter = iter + 1;
            break;
        }
        prev_loss = total_loss;
    }

    let model = LinearRegressionModel::new(crf_weights, crf_bias, bitrate_weights, bitrate_bias);

    // Calculate training MAE
    let train_mae_crf = calculate_mae_crf(&model, dataset)?;
    let train_mae_bitrate = calculate_mae_bitrate(&model, dataset)?;

    Ok(TrainingResult {
        model,
        train_mae_crf,
        train_mae_bitrate,
        val_mae_crf: None,
        val_mae_bitrate: None,
        iterations: final_iter,
        final_loss: prev_loss,
    })
}

/// Train with cross-validation.
pub fn train_with_cv(
    dataset: &TrainingDataset,
    config: &TrainingConfig,
) -> Result<TrainingResult> {
    if dataset.len() < config.cv_folds {
        return train(dataset, config);
    }

    let fold_size = dataset.len() / config.cv_folds;
    let mut cv_crf_maes = Vec::new();
    let mut cv_bitrate_maes = Vec::new();

    for fold in 0..config.cv_folds {
        let start = fold * fold_size;
        let end = if fold == config.cv_folds - 1 {
            dataset.len()
        } else {
            start + fold_size
        };

        let train_samples: Vec<_> = dataset.samples[..start]
            .iter()
            .chain(dataset.samples[end..].iter())
            .cloned()
            .collect();
        let val_samples: Vec<_> = dataset.samples[start..end].to_vec();

        let train_set = TrainingDataset {
            samples: train_samples,
        };
        let val_set = TrainingDataset {
            samples: val_samples,
        };

        let result = train(&train_set, config)?;
        cv_crf_maes.push(calculate_mae_crf(&result.model, &val_set)?);
        cv_bitrate_maes.push(calculate_mae_bitrate(&result.model, &val_set)?);
    }

    // Train final model on full dataset
    let mut result = train(dataset, config)?;
    result.val_mae_crf = Some(cv_crf_maes.iter().sum::<f64>() / cv_crf_maes.len() as f64);
    result.val_mae_bitrate =
        Some(cv_bitrate_maes.iter().sum::<f64>() / cv_bitrate_maes.len() as f64);

    Ok(result)
}

fn calculate_mae_crf(model: &LinearRegressionModel, dataset: &TrainingDataset) -> Result<f64> {
    use crate::model::BitratePredictor;
    let mut total_error = 0.0;
    for sample in &dataset.samples {
        let pred = model.predict(&sample.to_input())?;
        total_error += (pred.crf - sample.observed_crf).abs();
    }
    Ok(total_error / dataset.len() as f64)
}

fn calculate_mae_bitrate(
    model: &LinearRegressionModel,
    dataset: &TrainingDataset,
) -> Result<f64> {
    use crate::model::BitratePredictor;
    let mut total_error = 0.0;
    for sample in &dataset.samples {
        let pred = model.predict(&sample.to_input())?;
        total_error += (pred.bitrate_kbps as f64 - sample.observed_bitrate_kbps as f64).abs();
    }
    Ok(total_error / dataset.len() as f64)
}

/// Convex hull optimization for ABR ladder.
/// Removes rungs that are dominated (higher bitrate for same or lower VMAF).
pub fn convex_hull_optimize(rungs: &mut Vec<crate::ladder::LadderRung>) {
    if rungs.len() <= 2 {
        return;
    }

    // Sort by bitrate ascending
    rungs.sort_by(|a, b| a.bitrate_kbps.cmp(&b.bitrate_kbps));

    // Remove dominated points: a rung is dominated if there's another rung
    // with lower bitrate and equal-or-higher VMAF
    let mut hull: Vec<crate::ladder::LadderRung> = Vec::new();
    let mut max_vmaf_so_far = 0.0_f64;

    for rung in rungs.iter() {
        if rung.expected_vmaf > max_vmaf_so_far {
            hull.push(rung.clone());
            max_vmaf_so_far = rung.expected_vmaf;
        }
    }

    // Restore highest-to-lowest order
    hull.reverse();
    *rungs = hull;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_dataset() {
        let dataset = TrainingDataset::synthetic(50);
        assert_eq!(dataset.len(), 50);
        for sample in &dataset.samples {
            assert!(sample.observed_crf >= 18.0 && sample.observed_crf <= 30.0);
            assert!(sample.observed_bitrate_kbps >= 200);
        }
    }

    #[test]
    fn test_dataset_split() {
        let dataset = TrainingDataset::synthetic(100);
        let (train, val) = dataset.split(0.8);
        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    #[test]
    fn test_train_model() {
        let dataset = TrainingDataset::synthetic(50);
        let config = TrainingConfig {
            max_iterations: 100,
            learning_rate: 0.0001,
            ..Default::default()
        };
        let result = train(&dataset, &config).unwrap();
        assert!(result.train_mae_crf < 20.0);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_train_with_cv() {
        let dataset = TrainingDataset::synthetic(50);
        let config = TrainingConfig {
            max_iterations: 100,
            learning_rate: 0.0001,
            cv_folds: 5,
            ..Default::default()
        };
        let result = train_with_cv(&dataset, &config).unwrap();
        assert!(result.val_mae_crf.is_some());
        assert!(result.val_mae_bitrate.is_some());
    }

    #[test]
    fn test_convex_hull_optimize() {
        use crate::ladder::LadderRung;
        let mut rungs = vec![
            LadderRung {
                width: 1920,
                height: 1080,
                bitrate_kbps: 5000,
                crf: 23.0,
                expected_vmaf: 93.0,
                frame_rate: None,
                label: "1080p".into(),
            },
            LadderRung {
                width: 1280,
                height: 720,
                bitrate_kbps: 3000,
                crf: 24.0,
                expected_vmaf: 88.0,
                frame_rate: None,
                label: "720p".into(),
            },
            // Dominated: higher bitrate than 720p but lower VMAF
            LadderRung {
                width: 960,
                height: 540,
                bitrate_kbps: 3500,
                crf: 25.0,
                expected_vmaf: 85.0,
                frame_rate: None,
                label: "540p".into(),
            },
            LadderRung {
                width: 640,
                height: 360,
                bitrate_kbps: 1000,
                crf: 26.0,
                expected_vmaf: 75.0,
                frame_rate: None,
                label: "360p".into(),
            },
        ];

        convex_hull_optimize(&mut rungs);
        // The 540p rung should be removed (dominated by 720p: less bitrate, higher VMAF)
        assert_eq!(rungs.len(), 3);
        assert_eq!(rungs[0].label, "1080p");
        assert_eq!(rungs[1].label, "720p");
        assert_eq!(rungs[2].label, "360p");
    }

    #[test]
    fn test_empty_dataset_error() {
        let dataset = TrainingDataset::new();
        let config = TrainingConfig::default();
        assert!(train(&dataset, &config).is_err());
    }
}
