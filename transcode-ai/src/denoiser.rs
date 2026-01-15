//! AI-powered video denoising.

use crate::error::Result;
use crate::model::{InferenceSession, ModelBackend, ModelLoader};
use crate::Frame;
use std::path::PathBuf;
use tracing::debug;

/// Noise level estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NoiseLevel {
    /// Auto-detect noise level.
    #[default]
    Auto,
    /// Low noise (sigma ~5-15).
    Low,
    /// Medium noise (sigma ~15-25).
    Medium,
    /// High noise (sigma ~25-50).
    High,
    /// Very high noise (sigma >50).
    VeryHigh,
    /// Custom sigma value.
    Custom(u32),
}

impl NoiseLevel {
    /// Get estimated sigma value.
    pub fn sigma(self) -> f32 {
        match self {
            Self::Auto => 15.0, // Default
            Self::Low => 10.0,
            Self::Medium => 20.0,
            Self::High => 35.0,
            Self::VeryHigh => 50.0,
            Self::Custom(s) => s as f32,
        }
    }
}

/// Denoising model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DenoiseModel {
    /// DnCNN - denoising convolutional neural network.
    #[default]
    DnCNN,
    /// FFDNet - fast and flexible denoising.
    FFDNet,
    /// NAFNet - nonlinear activation free network.
    NAFNet,
    /// Restormer - efficient transformer for image restoration.
    Restormer,
    /// Non-local means (traditional, non-AI).
    NLMeans,
    /// Bilateral filter (traditional, non-AI).
    Bilateral,
}

impl DenoiseModel {
    /// Get model name for file lookup.
    pub fn model_name(self) -> String {
        match self {
            Self::DnCNN => "dncnn".to_string(),
            Self::FFDNet => "ffdnet".to_string(),
            Self::NAFNet => "nafnet".to_string(),
            Self::Restormer => "restormer".to_string(),
            Self::NLMeans | Self::Bilateral => "".to_string(),
        }
    }

    /// Check if this model requires a neural network.
    pub fn requires_nn(self) -> bool {
        !matches!(self, Self::NLMeans | Self::Bilateral)
    }
}

/// Denoiser configuration.
#[derive(Debug, Clone, Default)]
pub struct DenoiserConfig {
    /// Noise level.
    pub noise_level: NoiseLevel,
    /// Model to use.
    pub model: DenoiseModel,
    /// Inference backend.
    pub backend: ModelBackend,
    /// Custom model path.
    pub model_path: Option<PathBuf>,
    /// Preserve details strength (0.0-1.0).
    pub detail_preservation: f32,
    /// Temporal denoising (for video sequences).
    pub temporal: bool,
}

impl DenoiserConfig {
    /// Set noise level.
    pub fn with_noise_level(mut self, level: NoiseLevel) -> Self {
        self.noise_level = level;
        self
    }

    /// Set model.
    pub fn with_model(mut self, model: DenoiseModel) -> Self {
        self.model = model;
        self
    }

    /// Set backend.
    pub fn with_backend(mut self, backend: ModelBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set detail preservation.
    pub fn with_detail_preservation(mut self, strength: f32) -> Self {
        self.detail_preservation = strength.clamp(0.0, 1.0);
        self
    }

    /// Enable temporal denoising.
    pub fn with_temporal(mut self, enabled: bool) -> Self {
        self.temporal = enabled;
        self
    }
}

/// AI denoiser.
pub struct Denoiser {
    /// Configuration.
    config: DenoiserConfig,
    /// Inference session (if using neural network).
    session: Option<InferenceSession>,
    /// Previous frame for temporal denoising.
    #[allow(dead_code)]
    prev_frame: Option<Frame>,
}

impl std::fmt::Debug for Denoiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Denoiser")
            .field("config", &self.config)
            .field("has_session", &self.session.is_some())
            .finish()
    }
}

impl Denoiser {
    /// Create a new denoiser.
    pub fn new(config: DenoiserConfig) -> Result<Self> {
        let session = if config.model.requires_nn() {
            let loader = ModelLoader::new().with_backend(config.backend);

            if let Some(ref path) = config.model_path {
                Some(InferenceSession::new(path, config.backend)?)
            } else {
                let model_name = config.model.model_name();
                match loader.find_model(&model_name) {
                    Ok(path) => Some(InferenceSession::new(&path, config.backend)?),
                    Err(_) => {
                        debug!(
                            "Model '{}' not found, using fallback denoising",
                            model_name
                        );
                        None
                    }
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            session,
            prev_frame: None,
        })
    }

    /// Process a frame.
    pub fn process(&self, frame: &Frame) -> Result<Frame> {
        frame.validate()?;

        if let Some(ref session) = self.session {
            self.process_with_nn(frame, session)
        } else {
            self.process_fallback(frame)
        }
    }

    /// Process using neural network.
    fn process_with_nn(&self, frame: &Frame, _session: &InferenceSession) -> Result<Frame> {
        // Placeholder: In real implementation, would run the NN model
        // For now, apply a simple noise reduction

        let sigma = self.config.noise_level.sigma();

        // Simple bilateral-like filter as placeholder
        self.apply_bilateral_filter(frame, sigma)
    }

    /// Fallback denoising using bilateral filter.
    fn process_fallback(&self, frame: &Frame) -> Result<Frame> {
        let sigma = self.config.noise_level.sigma();

        match self.config.model {
            DenoiseModel::NLMeans => self.apply_nlmeans(frame, sigma),
            _ => self.apply_bilateral_filter(frame, sigma),
        }
    }

    /// Apply bilateral filter.
    fn apply_bilateral_filter(&self, frame: &Frame, sigma: f32) -> Result<Frame> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        // Kernel radius based on sigma
        let radius = (sigma / 5.0).ceil() as i32;
        let radius = radius.clamp(1, 5);

        let spatial_sigma = sigma;
        let range_sigma = sigma * 2.0;

        let mut output = vec![0u8; frame.data.len()];

        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let center_idx = (y * width + x) * channels + c;
                    let center_val = frame.data[center_idx] as f32;

                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let ny = (y as i32 + dy).clamp(0, height as i32 - 1) as usize;
                            let nx = (x as i32 + dx).clamp(0, width as i32 - 1) as usize;

                            let neighbor_idx = (ny * width + nx) * channels + c;
                            let neighbor_val = frame.data[neighbor_idx] as f32;

                            // Spatial weight
                            let spatial_dist = (dx * dx + dy * dy) as f32;
                            let spatial_weight = (-spatial_dist / (2.0 * spatial_sigma * spatial_sigma)).exp();

                            // Range weight
                            let range_dist = (center_val - neighbor_val).powi(2);
                            let range_weight = (-range_dist / (2.0 * range_sigma * range_sigma)).exp();

                            let weight = spatial_weight * range_weight;
                            sum += neighbor_val * weight;
                            weight_sum += weight;
                        }
                    }

                    let filtered = if weight_sum > 0.0 {
                        sum / weight_sum
                    } else {
                        center_val
                    };

                    // Blend with original based on detail preservation
                    let dp = self.config.detail_preservation;
                    let result = filtered * (1.0 - dp) + center_val * dp;

                    output[center_idx] = result.clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(Frame::new(output, frame.width, frame.height, frame.channels).with_pts(frame.pts))
    }

    /// Apply non-local means denoising.
    fn apply_nlmeans(&self, frame: &Frame, sigma: f32) -> Result<Frame> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let channels = frame.channels as usize;

        // NLM parameters
        let patch_size = 7;
        let search_window = 21;
        let h = sigma; // Filtering parameter

        let half_patch = patch_size / 2;
        let half_search = search_window / 2;

        let mut output = vec![0u8; frame.data.len()];

        for y in 0..height {
            for x in 0..width {
                for c in 0..channels {
                    let center_idx = (y * width + x) * channels + c;

                    let mut sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    // Search window
                    for sy in (y as i32 - half_search).max(0)..((y as i32 + half_search + 1).min(height as i32)) {
                        for sx in (x as i32 - half_search).max(0)..((x as i32 + half_search + 1).min(width as i32)) {
                            // Compute patch distance
                            let mut dist = 0.0f32;
                            let mut count = 0;

                            for py in -half_patch..=half_patch {
                                for px in -half_patch..=half_patch {
                                    let y1 = (y as i32 + py).clamp(0, height as i32 - 1) as usize;
                                    let x1 = (x as i32 + px).clamp(0, width as i32 - 1) as usize;
                                    let y2 = (sy + py).clamp(0, height as i32 - 1) as usize;
                                    let x2 = (sx + px).clamp(0, width as i32 - 1) as usize;

                                    let idx1 = (y1 * width + x1) * channels + c;
                                    let idx2 = (y2 * width + x2) * channels + c;

                                    let diff = frame.data[idx1] as f32 - frame.data[idx2] as f32;
                                    dist += diff * diff;
                                    count += 1;
                                }
                            }

                            dist /= count as f32;

                            // Weight based on patch similarity
                            let weight = (-dist / (h * h)).exp();

                            let neighbor_idx = (sy as usize * width + sx as usize) * channels + c;
                            sum += frame.data[neighbor_idx] as f32 * weight;
                            weight_sum += weight;
                        }
                    }

                    let filtered = if weight_sum > 0.0 {
                        sum / weight_sum
                    } else {
                        frame.data[center_idx] as f32
                    };

                    output[center_idx] = filtered.clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(Frame::new(output, frame.width, frame.height, frame.channels).with_pts(frame.pts))
    }

    /// Get configuration.
    pub fn config(&self) -> &DenoiserConfig {
        &self.config
    }

    /// Check if using neural network.
    pub fn is_using_nn(&self) -> bool {
        self.session.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(width: u32, height: u32) -> crate::Frame {
        let channels = 3u8;
        let data = vec![128u8; (width * height * channels as u32) as usize];
        crate::Frame::new(data, width, height, channels)
    }

    fn create_noisy_frame(width: u32, height: u32) -> crate::Frame {
        let channels = 3u8;
        let size = (width * height * channels as u32) as usize;
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            // Add some variation to simulate noise
            data.push(((i % 256) as u8).wrapping_add((i / 256) as u8));
        }
        crate::Frame::new(data, width, height, channels)
    }

    #[test]
    fn test_noise_level_sigma_values() {
        assert_eq!(NoiseLevel::Low.sigma(), 10.0);
        assert_eq!(NoiseLevel::Medium.sigma(), 20.0);
        assert_eq!(NoiseLevel::High.sigma(), 35.0);
        assert_eq!(NoiseLevel::VeryHigh.sigma(), 50.0);
        assert_eq!(NoiseLevel::Custom(25).sigma(), 25.0);
        assert_eq!(NoiseLevel::Auto.sigma(), 15.0);
    }

    #[test]
    fn test_denoise_model_names() {
        assert_eq!(DenoiseModel::DnCNN.model_name(), "dncnn");
        assert_eq!(DenoiseModel::FFDNet.model_name(), "ffdnet");
        assert_eq!(DenoiseModel::NLMeans.model_name(), "");
        assert_eq!(DenoiseModel::Bilateral.model_name(), "");
    }

    #[test]
    fn test_denoise_model_requires_nn() {
        assert!(DenoiseModel::DnCNN.requires_nn());
        assert!(DenoiseModel::FFDNet.requires_nn());
        assert!(!DenoiseModel::NLMeans.requires_nn());
        assert!(!DenoiseModel::Bilateral.requires_nn());
    }

    #[test]
    fn test_denoiser_config_builder() {
        let config = DenoiserConfig::default()
            .with_noise_level(NoiseLevel::High)
            .with_model(DenoiseModel::Bilateral)
            .with_detail_preservation(0.5)
            .with_temporal(true);

        assert_eq!(config.noise_level, NoiseLevel::High);
        assert_eq!(config.model, DenoiseModel::Bilateral);
        assert!((config.detail_preservation - 0.5).abs() < f32::EPSILON);
        assert!(config.temporal);
    }

    #[test]
    fn test_denoiser_config_detail_preservation_clamped() {
        let config = DenoiserConfig::default()
            .with_detail_preservation(1.5); // Should be clamped to 1.0
        assert!((config.detail_preservation - 1.0).abs() < f32::EPSILON);

        let config = DenoiserConfig::default()
            .with_detail_preservation(-0.5); // Should be clamped to 0.0
        assert!(config.detail_preservation.abs() < f32::EPSILON);
    }

    #[test]
    fn test_denoiser_preserves_dimensions() {
        let config = DenoiserConfig::default()
            .with_model(DenoiseModel::Bilateral)
            .with_noise_level(NoiseLevel::Medium);

        let denoiser = Denoiser::new(config).unwrap();
        let input = create_test_frame(64, 64);
        let output = denoiser.process(&input).unwrap();

        assert_eq!(output.width, input.width);
        assert_eq!(output.height, input.height);
        assert_eq!(output.channels, input.channels);
    }

    #[test]
    fn test_denoiser_preserves_pts() {
        let config = DenoiserConfig::default()
            .with_model(DenoiseModel::Bilateral);

        let denoiser = Denoiser::new(config).unwrap();
        let input = create_test_frame(32, 32).with_pts(98765);
        let output = denoiser.process(&input).unwrap();

        assert_eq!(output.pts, 98765);
    }

    #[test]
    fn test_denoiser_bilateral_filter() {
        let config = DenoiserConfig::default()
            .with_model(DenoiseModel::Bilateral)
            .with_noise_level(NoiseLevel::Low);

        let denoiser = Denoiser::new(config).unwrap();
        let input = create_noisy_frame(32, 32);
        let output = denoiser.process(&input).unwrap();

        // Output should have valid data
        assert_eq!(output.data.len(), input.data.len());
    }

    #[test]
    fn test_denoiser_nlmeans_filter() {
        let config = DenoiserConfig::default()
            .with_model(DenoiseModel::NLMeans)
            .with_noise_level(NoiseLevel::Low);

        let denoiser = Denoiser::new(config).unwrap();
        // Use small frame for NLMeans (it's slow)
        let input = create_noisy_frame(16, 16);
        let output = denoiser.process(&input).unwrap();

        assert_eq!(output.width, input.width);
        assert_eq!(output.height, input.height);
    }

    #[test]
    fn test_denoiser_fallback_when_no_model() {
        let config = DenoiserConfig::default()
            .with_model(DenoiseModel::DnCNN); // Requires NN

        let denoiser = Denoiser::new(config).unwrap();
        // Should fall back when model not found
        assert!(!denoiser.is_using_nn());

        let input = create_test_frame(32, 32);
        let output = denoiser.process(&input).unwrap();
        assert_eq!(output.width, 32);
    }

    #[test]
    fn test_denoiser_invalid_frame() {
        let config = DenoiserConfig::default()
            .with_model(DenoiseModel::Bilateral);

        let denoiser = Denoiser::new(config).unwrap();

        // Create frame with mismatched data size
        let frame = crate::Frame {
            data: vec![0u8; 50], // Wrong size
            width: 32,
            height: 32,
            channels: 3,
            pts: 0,
        };

        let result = denoiser.process(&frame);
        assert!(result.is_err());
    }
}
