//! AI-powered video upscaling (super resolution).

use crate::error::{AiError, Result};
use crate::model::{InferenceSession, ModelBackend, ModelLoader};
use crate::Frame;
use ndarray::Array4;
use std::path::PathBuf;
use tracing::debug;

/// Upscale factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleFactor {
    /// 2x upscaling (e.g., 1080p -> 4K).
    #[default]
    X2,
    /// 4x upscaling (e.g., 480p -> 4K).
    X4,
}

impl ScaleFactor {
    /// Get numeric scale factor.
    pub fn as_u32(self) -> u32 {
        match self {
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }
}

/// Available upscale models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpscaleModel {
    /// Real-ESRGAN - best for anime and real-world images.
    #[default]
    RealESRGAN,
    /// ESPCN - fast, efficient upscaling.
    ESPCN,
    /// FSRCNN - fast super resolution CNN.
    FSRCNN,
    /// EDSR - enhanced deep residual networks.
    EDSR,
    /// SwinIR - Swin Transformer for image restoration.
    SwinIR,
    /// Anime4K - optimized for anime content.
    Anime4K,
    /// Lanczos - traditional non-AI fallback.
    Lanczos,
}

impl UpscaleModel {
    /// Get model name for file lookup.
    pub fn model_name(self, scale: ScaleFactor) -> String {
        let scale_str = match scale {
            ScaleFactor::X2 => "x2",
            ScaleFactor::X4 => "x4",
        };

        match self {
            Self::RealESRGAN => format!("realesrgan_{}", scale_str),
            Self::ESPCN => format!("espcn_{}", scale_str),
            Self::FSRCNN => format!("fsrcnn_{}", scale_str),
            Self::EDSR => format!("edsr_{}", scale_str),
            Self::SwinIR => format!("swinir_{}", scale_str),
            Self::Anime4K => format!("anime4k_{}", scale_str),
            Self::Lanczos => "lanczos".to_string(),
        }
    }

    /// Check if this model requires a neural network.
    pub fn requires_nn(self) -> bool {
        !matches!(self, Self::Lanczos)
    }
}

/// Upscaler configuration.
#[derive(Debug, Clone, Default)]
pub struct UpscalerConfig {
    /// Scale factor.
    pub scale_factor: ScaleFactor,
    /// Model to use.
    pub model: UpscaleModel,
    /// Inference backend.
    pub backend: ModelBackend,
    /// Custom model path.
    pub model_path: Option<PathBuf>,
    /// Tile size for large images (0 for no tiling).
    pub tile_size: u32,
    /// Tile overlap.
    pub tile_overlap: u32,
}

impl UpscalerConfig {
    /// Set scale factor.
    pub fn with_scale_factor(mut self, scale: ScaleFactor) -> Self {
        self.scale_factor = scale;
        self
    }

    /// Set model.
    pub fn with_model(mut self, model: UpscaleModel) -> Self {
        self.model = model;
        self
    }

    /// Set inference backend.
    pub fn with_backend(mut self, backend: ModelBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set custom model path.
    pub fn with_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set tile size for processing large images.
    pub fn with_tiling(mut self, tile_size: u32, overlap: u32) -> Self {
        self.tile_size = tile_size;
        self.tile_overlap = overlap;
        self
    }
}

/// AI upscaler.
pub struct Upscaler {
    /// Configuration.
    config: UpscalerConfig,
    /// Inference session (if using neural network).
    session: Option<InferenceSession>,
}

impl std::fmt::Debug for Upscaler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Upscaler")
            .field("config", &self.config)
            .field("has_session", &self.session.is_some())
            .finish()
    }
}

impl Upscaler {
    /// Create a new upscaler.
    pub fn new(config: UpscalerConfig) -> Result<Self> {
        let session = if config.model.requires_nn() {
            // Try to load the model
            let loader = ModelLoader::new().with_backend(config.backend);

            if let Some(ref path) = config.model_path {
                Some(InferenceSession::new(path, config.backend)?)
            } else {
                // Try to find the model
                let model_name = config.model.model_name(config.scale_factor);
                match loader.find_model(&model_name) {
                    Ok(path) => Some(InferenceSession::new(&path, config.backend)?),
                    Err(_) => {
                        debug!(
                            "Model '{}' not found, using fallback upscaling",
                            model_name
                        );
                        None
                    }
                }
            }
        } else {
            None
        };

        Ok(Self { config, session })
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
    fn process_with_nn(&self, frame: &Frame, session: &InferenceSession) -> Result<Frame> {
        let _scale = self.config.scale_factor.as_u32();

        // Convert frame to NCHW format
        let input = frame_to_nchw(frame)?;

        // Run inference
        let output = if self.config.tile_size > 0 {
            self.process_tiled(&input, session)?
        } else {
            session.run(&input)?
        };

        // Convert back to frame
        nchw_to_frame(&output, frame.pts)
    }

    /// Process using tiled inference for large images.
    fn process_tiled(&self, input: &Array4<f32>, session: &InferenceSession) -> Result<Array4<f32>> {
        let (batch, channels, height, width) = input.dim();
        let tile_size = self.config.tile_size as usize;
        let overlap = self.config.tile_overlap as usize;
        let scale = self.config.scale_factor.as_u32() as usize;

        let out_height = height * scale;
        let out_width = width * scale;
        let out_tile_size = tile_size * scale;
        let out_overlap = overlap * scale;

        let mut output: Array4<f32> = Array4::zeros((batch, channels, out_height, out_width));
        let mut weights: Array4<f32> = Array4::zeros((batch, channels, out_height, out_width));

        let step = tile_size - overlap;
        let _out_step = out_tile_size - out_overlap;

        for y in (0..height).step_by(step) {
            for x in (0..width).step_by(step) {
                let y_end = (y + tile_size).min(height);
                let x_end = (x + tile_size).min(width);

                // Extract tile
                let tile = input
                    .slice(ndarray::s![.., .., y..y_end, x..x_end])
                    .to_owned();

                // Process tile
                let processed = session.run(&tile)?;

                // Calculate output position
                let out_y = y * scale;
                let out_x = x * scale;
                let out_y_end = y_end * scale;
                let out_x_end = x_end * scale;

                // Blend into output with linear weights for overlap
                for (oy, py) in (out_y..out_y_end).enumerate() {
                    for (ox, px) in (out_x..out_x_end).enumerate() {
                        // Calculate blend weight (1.0 in center, decreasing at edges)
                        let wy = if oy < out_overlap {
                            oy as f32 / out_overlap as f32
                        } else if oy >= out_tile_size - out_overlap {
                            (out_tile_size - oy) as f32 / out_overlap as f32
                        } else {
                            1.0
                        };

                        let wx = if ox < out_overlap {
                            ox as f32 / out_overlap as f32
                        } else if ox >= out_tile_size - out_overlap {
                            (out_tile_size - ox) as f32 / out_overlap as f32
                        } else {
                            1.0
                        };

                        let w = wy * wx;

                        for b in 0..batch {
                            for c in 0..channels {
                                if py < out_height && px < out_width {
                                    output[[b, c, py, px]] += processed[[b, c, oy, ox]] * w;
                                    weights[[b, c, py, px]] += w;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize by weights
        for i in 0..output.len() {
            if weights.as_slice().unwrap()[i] > 0.0 {
                output.as_slice_mut().unwrap()[i] /= weights.as_slice().unwrap()[i];
            }
        }

        Ok(output)
    }

    /// Fallback upscaling using Lanczos interpolation.
    fn process_fallback(&self, frame: &Frame) -> Result<Frame> {
        let scale = self.config.scale_factor.as_u32();
        let new_width = frame.width * scale;
        let new_height = frame.height * scale;

        // Use image crate for Lanczos resize
        let img = image::RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
            .ok_or_else(|| AiError::InvalidFrame("Failed to create image".into()))?;

        let resized = image::imageops::resize(
            &img,
            new_width,
            new_height,
            image::imageops::FilterType::Lanczos3,
        );

        Ok(Frame::new(
            resized.into_raw(),
            new_width,
            new_height,
            3,
        )
        .with_pts(frame.pts))
    }

    /// Get configuration.
    pub fn config(&self) -> &UpscalerConfig {
        &self.config
    }

    /// Check if using neural network.
    pub fn is_using_nn(&self) -> bool {
        self.session.is_some()
    }
}

/// Convert frame to NCHW format for neural network.
fn frame_to_nchw(frame: &Frame) -> Result<Array4<f32>> {
    let height = frame.height as usize;
    let width = frame.width as usize;
    let channels = frame.channels as usize;

    // Create NCHW array (batch=1, channels, height, width)
    let array = Array4::from_shape_fn((1, channels, height, width), |(_, c, y, x)| {
        let idx = (y * width + x) * channels + c;
        frame.data[idx] as f32 / 255.0
    });

    Ok(array)
}

/// Convert NCHW array back to frame.
fn nchw_to_frame(array: &Array4<f32>, pts: i64) -> Result<Frame> {
    let (_, channels, height, width) = array.dim();

    let mut data = vec![0u8; width * height * channels];

    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * channels + c;
                let value = (array[[0, c, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
                data[idx] = value;
            }
        }
    }

    Ok(Frame::new(data, width as u32, height as u32, channels as u8).with_pts(pts))
}
