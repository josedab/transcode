//! AI-powered video enhancement.
//!
//! This crate provides neural network-based video enhancement capabilities:
//!
//! - **Super Resolution**: Upscale video to higher resolutions (2x, 4x)
//! - **Denoising**: Remove noise while preserving details
//! - **Frame Interpolation**: Generate intermediate frames for higher frame rates
//! - **Edge Enhancement**: Sharpen and enhance edges
//!
//! # Example
//!
//! ```ignore
//! use transcode_ai::{Upscaler, UpscaleConfig, ScaleFactor};
//!
//! // Create upscaler
//! let config = UpscaleConfig::default()
//!     .with_scale_factor(ScaleFactor::X2)
//!     .with_model(UpscaleModel::RealESRGAN);
//!
//! let upscaler = Upscaler::new(config)?;
//!
//! // Upscale a frame
//! let output = upscaler.process(&input_frame)?;
//! ```

#![allow(dead_code)]

mod denoiser;
mod error;
mod interpolator;
mod model;
pub mod onnx;
mod upscaler;

pub use denoiser::{Denoiser, DenoiserConfig, NoiseLevel};
pub use error::{AiError, Result};
pub use interpolator::{FrameInterpolator, InterpolatorConfig, InterpolationMode};
pub use model::{ModelBackend, ModelInfo, ModelLoader, InferenceSession};
pub use onnx::{ExecutionProvider, OnnxConfig, OnnxSession, OptimizationLevel, TensorInfo, ModelMetadata};
pub use upscaler::{ScaleFactor, UpscaleModel, Upscaler, UpscalerConfig};

/// AI enhancement pipeline combining multiple enhancers.
#[derive(Debug)]
pub struct EnhancementPipeline {
    /// Optional upscaler.
    upscaler: Option<Upscaler>,
    /// Optional denoiser.
    denoiser: Option<Denoiser>,
    /// Optional interpolator.
    interpolator: Option<FrameInterpolator>,
    /// Pipeline configuration.
    config: PipelineConfig,
}

/// Enhancement pipeline configuration.
#[derive(Debug, Clone, Default)]
pub struct PipelineConfig {
    /// Enable upscaling.
    pub enable_upscale: bool,
    /// Upscale configuration.
    pub upscale_config: Option<UpscalerConfig>,
    /// Enable denoising.
    pub enable_denoise: bool,
    /// Denoise configuration.
    pub denoise_config: Option<DenoiserConfig>,
    /// Enable frame interpolation.
    pub enable_interpolation: bool,
    /// Interpolation configuration.
    pub interpolation_config: Option<InterpolatorConfig>,
    /// Processing order.
    pub order: ProcessingOrder,
}

/// Order of enhancement operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProcessingOrder {
    /// Denoise -> Upscale -> Interpolate (default, best quality).
    #[default]
    DenoiseUpscaleInterpolate,
    /// Upscale -> Denoise -> Interpolate.
    UpscaleDenoiseInterpolate,
    /// Custom order based on config.
    Custom,
}

impl EnhancementPipeline {
    /// Create a new enhancement pipeline.
    pub fn new(config: PipelineConfig) -> Result<Self> {
        let upscaler = if config.enable_upscale {
            Some(Upscaler::new(
                config.upscale_config.clone().unwrap_or_default(),
            )?)
        } else {
            None
        };

        let denoiser = if config.enable_denoise {
            Some(Denoiser::new(
                config.denoise_config.clone().unwrap_or_default(),
            )?)
        } else {
            None
        };

        let interpolator = if config.enable_interpolation {
            Some(FrameInterpolator::new(
                config.interpolation_config.clone().unwrap_or_default(),
            )?)
        } else {
            None
        };

        Ok(Self {
            upscaler,
            denoiser,
            interpolator,
            config,
        })
    }

    /// Create a pipeline for upscaling only.
    pub fn upscale_only(scale: ScaleFactor) -> Result<Self> {
        Self::new(PipelineConfig {
            enable_upscale: true,
            upscale_config: Some(UpscalerConfig::default().with_scale_factor(scale)),
            ..Default::default()
        })
    }

    /// Create a pipeline for denoising only.
    pub fn denoise_only(level: NoiseLevel) -> Result<Self> {
        Self::new(PipelineConfig {
            enable_denoise: true,
            denoise_config: Some(DenoiserConfig::default().with_noise_level(level)),
            ..Default::default()
        })
    }

    /// Process a single frame through the pipeline.
    pub fn process_frame(&self, frame: &Frame) -> Result<Frame> {
        let mut result = frame.clone();

        match self.config.order {
            ProcessingOrder::DenoiseUpscaleInterpolate => {
                if let Some(ref denoiser) = self.denoiser {
                    result = denoiser.process(&result)?;
                }
                if let Some(ref upscaler) = self.upscaler {
                    result = upscaler.process(&result)?;
                }
            }
            ProcessingOrder::UpscaleDenoiseInterpolate => {
                if let Some(ref upscaler) = self.upscaler {
                    result = upscaler.process(&result)?;
                }
                if let Some(ref denoiser) = self.denoiser {
                    result = denoiser.process(&result)?;
                }
            }
            ProcessingOrder::Custom => {
                // Apply in config order
                if let Some(ref denoiser) = self.denoiser {
                    result = denoiser.process(&result)?;
                }
                if let Some(ref upscaler) = self.upscaler {
                    result = upscaler.process(&result)?;
                }
            }
        }

        Ok(result)
    }

    /// Interpolate between two frames.
    pub fn interpolate_frames(
        &self,
        frame1: &Frame,
        frame2: &Frame,
        factor: f32,
    ) -> Result<Vec<Frame>> {
        if let Some(ref interpolator) = self.interpolator {
            interpolator.interpolate(frame1, frame2, factor)
        } else {
            Err(AiError::NotConfigured("Interpolation not enabled".into()))
        }
    }

    /// Check if upscaling is enabled.
    pub fn has_upscaler(&self) -> bool {
        self.upscaler.is_some()
    }

    /// Check if denoising is enabled.
    pub fn has_denoiser(&self) -> bool {
        self.denoiser.is_some()
    }

    /// Check if interpolation is enabled.
    pub fn has_interpolator(&self) -> bool {
        self.interpolator.is_some()
    }
}

/// A video frame for AI processing.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Image data in RGB format.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of channels (3 for RGB, 4 for RGBA).
    pub channels: u8,
    /// Presentation timestamp.
    pub pts: i64,
}

impl Frame {
    /// Create a new frame.
    pub fn new(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Self {
        Self {
            data,
            width,
            height,
            channels,
            pts: 0,
        }
    }

    /// Create frame with timestamp.
    pub fn with_pts(mut self, pts: i64) -> Self {
        self.pts = pts;
        self
    }

    /// Get expected data size.
    pub fn expected_size(&self) -> usize {
        (self.width * self.height * self.channels as u32) as usize
    }

    /// Validate frame data.
    pub fn validate(&self) -> Result<()> {
        let expected = self.expected_size();
        if self.data.len() != expected {
            return Err(AiError::InvalidFrame(format!(
                "Data size mismatch: expected {}, got {}",
                expected,
                self.data.len()
            )));
        }
        Ok(())
    }

    /// Convert to ndarray for processing.
    pub fn to_ndarray(&self) -> ndarray::Array3<f32> {
        let shape = (self.height as usize, self.width as usize, self.channels as usize);
        ndarray::Array3::from_shape_fn(shape, |(y, x, c)| {
            let idx = (y * self.width as usize + x) * self.channels as usize + c;
            self.data[idx] as f32 / 255.0
        })
    }

    /// Create from ndarray.
    pub fn from_ndarray(arr: &ndarray::Array3<f32>, pts: i64) -> Self {
        let shape = arr.dim();
        let height = shape.0 as u32;
        let width = shape.1 as u32;
        let channels = shape.2 as u8;

        let data: Vec<u8> = arr
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        Self {
            data,
            width,
            height,
            channels,
            pts,
        }
    }
}
