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
