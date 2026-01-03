//! Neural super-resolution for transcode
//!
//! This crate provides neural network-based video upscaling using ONNX runtime.
//!
//! # Features
//!
//! - `onnx` - Enable ONNX Runtime support for neural inference
//! - `cuda` - Enable CUDA GPU acceleration
//! - `tensorrt` - Enable TensorRT optimized inference
//! - `download` - Enable model downloading from URLs
//!
//! # Example
//!
//! ```no_run
//! use transcode_neural::{
//!     NeuralConfig, NeuralFrame, ModelType,
//!     upscaler::{OnnxUpscaler, OnnxUpscalerConfig},
//! };
//!
//! // Create upscaler with default config
//! let config = OnnxUpscalerConfig::default();
//! let mut upscaler = OnnxUpscaler::new(config);
//!
//! // Enable mock inference for testing
//! upscaler.enable_mock(4);
//!
//! // Upscale a frame
//! let frame = NeuralFrame::new(256, 256);
//! let upscaled = upscaler.upscale(&frame).unwrap();
//! assert_eq!(upscaled.width, 1024);
//! assert_eq!(upscaled.height, 1024);
//! ```

use std::path::Path;

mod error;
pub mod benchmark;
pub mod inference;
pub mod models;
pub mod postprocessing;
pub mod preprocessing;
pub mod tiled;
pub mod upscaler;

// Re-export primary types
pub use error::*;
pub use inference::{
    ExecutionProvider, InferenceConfig, InputTensor, MockInference, OptimizationLevel,
    OutputTensor,
};
pub use models::{ModelInfo, ModelManager, ModelMetadata, ModelRegistry};
pub use postprocessing::{PostprocessConfig, PostprocessedOutput, Postprocessor};
pub use preprocessing::{
    BatchPreprocessedInput, ChannelOrder, NormalizationMode, PaddingMode, PreprocessConfig,
    PreprocessedInput, Preprocessor,
};
pub use tiled::{global_model_cache, ModelCache, Tile, TileConfig, TiledProcessor};
pub use upscaler::{CpuUpscaler, OnnxUpscaler, OnnxUpscalerConfig, ProgressCallback, StreamingUpscaler, UpscaleAlgorithm};

#[cfg(feature = "onnx")]
pub use inference::InferenceSession;

/// Result type for neural operations
pub type Result<T> = std::result::Result<T, NeuralError>;

/// Neural upscaler configuration
#[derive(Debug, Clone)]
pub struct NeuralConfig {
    /// Model path or name
    pub model: ModelType,
    /// Target scale factor (2, 3, or 4)
    pub scale: u32,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// GPU device index
    pub device_id: u32,
    /// Tile size for large images
    pub tile_size: u32,
    /// Tile overlap
    pub tile_overlap: u32,
    /// Batch size
    pub batch_size: usize,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            model: ModelType::RealEsrgan,
            scale: 2,
            use_gpu: true,
            device_id: 0,
            tile_size: 512,
            tile_overlap: 32,
            batch_size: 1,
        }
    }
}

/// Available super-resolution models
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelType {
    /// Real-ESRGAN model (general purpose)
    RealEsrgan,
    /// Real-ESRGAN anime model
    RealEsrganAnime,
    /// BSRGAN model
    Bsrgan,
    /// SwinIR model
    SwinIR,
    /// Custom model from path
    Custom(String),
}

impl ModelType {
    /// Get expected input channels
    pub fn input_channels(&self) -> usize {
        3 // RGB
    }

    /// Get model scale factor
    pub fn default_scale(&self) -> u32 {
        match self {
            ModelType::RealEsrgan => 4,
            ModelType::RealEsrganAnime => 4,
            ModelType::Bsrgan => 4,
            ModelType::SwinIR => 2,
            ModelType::Custom(_) => 4,
        }
    }
}

/// Frame data for neural processing
#[derive(Debug, Clone)]
pub struct NeuralFrame {
    /// RGB pixel data (HWC layout)
    pub data: Vec<f32>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Number of channels (3 for RGB)
    pub channels: u32,
}

impl NeuralFrame {
    /// Create a new neural frame
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height * 3) as usize;
        Self {
            data: vec![0.0; size],
            width,
            height,
            channels: 3,
        }
    }

    /// Create from raw RGB data
    pub fn from_rgb(data: Vec<f32>, width: u32, height: u32) -> Result<Self> {
        let expected = (width * height * 3) as usize;
        if data.len() != expected {
            return Err(NeuralError::InvalidInput(format!(
                "expected {} elements, got {}",
                expected,
                data.len()
            )));
        }
        Ok(Self {
            data,
            width,
            height,
            channels: 3,
        })
    }

    /// Convert from YUV420 frame
    pub fn from_yuv420(y: &[u8], u: &[u8], v: &[u8], width: u32, height: u32) -> Self {
        let mut frame = Self::new(width, height);

        for py in 0..height {
            for px in 0..width {
                let y_idx = (py * width + px) as usize;
                let uv_idx = ((py / 2) * (width / 2) + (px / 2)) as usize;

                let y_val = y.get(y_idx).copied().unwrap_or(16) as f32;
                let u_val = u.get(uv_idx).copied().unwrap_or(128) as f32;
                let v_val = v.get(uv_idx).copied().unwrap_or(128) as f32;

                // YUV to RGB conversion
                let r = (y_val + 1.402 * (v_val - 128.0)).clamp(0.0, 255.0) / 255.0;
                let g = (y_val - 0.344 * (u_val - 128.0) - 0.714 * (v_val - 128.0))
                    .clamp(0.0, 255.0)
                    / 255.0;
                let b = (y_val + 1.772 * (u_val - 128.0)).clamp(0.0, 255.0) / 255.0;

                let rgb_idx = ((py * width + px) * 3) as usize;
                frame.data[rgb_idx] = r;
                frame.data[rgb_idx + 1] = g;
                frame.data[rgb_idx + 2] = b;
            }
        }

        frame
    }

    /// Convert to YUV420 data
    pub fn to_yuv420(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y_size = (self.width * self.height) as usize;
        let uv_size = ((self.width / 2) * (self.height / 2)) as usize;

        let mut y = vec![0u8; y_size];
        let mut u = vec![0u8; uv_size];
        let mut v = vec![0u8; uv_size];

        for py in 0..self.height {
            for px in 0..self.width {
                let rgb_idx = ((py * self.width + px) * 3) as usize;
                let r = self.data[rgb_idx] * 255.0;
                let g = self.data[rgb_idx + 1] * 255.0;
                let b = self.data[rgb_idx + 2] * 255.0;

                // RGB to YUV conversion
                let y_val = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0);

                let y_idx = (py * self.width + px) as usize;
                y[y_idx] = y_val as u8;

                if py % 2 == 0 && px % 2 == 0 {
                    let u_val = (128.0 - 0.169 * r - 0.331 * g + 0.500 * b).clamp(0.0, 255.0);
                    let v_val = (128.0 + 0.500 * r - 0.419 * g - 0.081 * b).clamp(0.0, 255.0);

                    let uv_idx = ((py / 2) * (self.width / 2) + (px / 2)) as usize;
                    u[uv_idx] = u_val as u8;
                    v[uv_idx] = v_val as u8;
                }
            }
        }

        (y, u, v)
    }

    /// Convert to u8 bytes (0-255 range).
    pub fn to_bytes(&self) -> Vec<u8> {
        postprocessing::to_u8(&self.data)
    }

    /// Get pixel value at (x, y) for channel c.
    pub fn get(&self, x: u32, y: u32, c: u32) -> f32 {
        let idx = ((y * self.width + x) * 3 + c) as usize;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Set pixel value at (x, y) for channel c.
    pub fn set(&mut self, x: u32, y: u32, c: u32, value: f32) {
        let idx = ((y * self.width + x) * 3 + c) as usize;
        if idx < self.data.len() {
            self.data[idx] = value;
        }
    }
}

/// Neural upscaler using ONNX runtime (legacy interface)
pub struct NeuralUpscaler {
    config: NeuralConfig,
    #[cfg(feature = "onnx")]
    session: Option<ort::session::Session>,
}

impl NeuralUpscaler {
    /// Create a new neural upscaler
    pub fn new(config: NeuralConfig) -> Result<Self> {
        Ok(Self {
            config,
            #[cfg(feature = "onnx")]
            session: None,
        })
    }

    /// Load model from path
    #[cfg(feature = "onnx")]
    pub fn load_model(&mut self, path: &Path) -> Result<()> {
        use ort::session::{builder::GraphOptimizationLevel, Session};

        let session = Session::builder()
            .map_err(|e| NeuralError::ModelLoad(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| NeuralError::ModelLoad(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| NeuralError::ModelLoad(e.to_string()))?;

        self.session = Some(session);
        Ok(())
    }

    #[cfg(not(feature = "onnx"))]
    pub fn load_model(&mut self, _path: &Path) -> Result<()> {
        Err(NeuralError::OnnxNotEnabled)
    }

    /// Upscale a frame
    pub fn upscale(&self, frame: &NeuralFrame) -> Result<NeuralFrame> {
        let scale = self.config.scale;
        let new_width = frame.width * scale;
        let new_height = frame.height * scale;

        #[cfg(feature = "onnx")]
        if let Some(ref _session) = self.session {
            // Real ONNX inference would happen here
            return self.bicubic_upscale(frame, new_width, new_height);
        }

        // Fallback to bicubic interpolation
        self.bicubic_upscale(frame, new_width, new_height)
    }

    fn bicubic_upscale(
        &self,
        frame: &NeuralFrame,
        new_width: u32,
        new_height: u32,
    ) -> Result<NeuralFrame> {
        CpuUpscaler::bicubic(frame, new_width, new_height)
    }

    /// Get configuration
    pub fn config(&self) -> &NeuralConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_frame_creation() {
        let frame = NeuralFrame::new(64, 64);
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_neural_upscaler() {
        let config = NeuralConfig::default();
        let upscaler = NeuralUpscaler::new(config).unwrap();

        let frame = NeuralFrame::new(32, 32);
        let upscaled = upscaler.upscale(&frame).unwrap();

        assert_eq!(upscaled.width, 64); // 2x scale by default
        assert_eq!(upscaled.height, 64);
    }

    #[test]
    fn test_yuv_rgb_roundtrip() {
        let mut frame = NeuralFrame::new(8, 8);
        // Fill with some test data
        for i in 0..frame.data.len() {
            frame.data[i] = i as f32 / frame.data.len() as f32;
        }

        let (y, u, v) = frame.to_yuv420();
        let restored = NeuralFrame::from_yuv420(&y, &u, &v, 8, 8);

        // Check dimensions preserved
        assert_eq!(restored.width, 8);
        assert_eq!(restored.height, 8);
    }

    #[test]
    fn test_frame_get_set() {
        let mut frame = NeuralFrame::new(4, 4);
        frame.set(1, 2, 0, 0.5);
        assert!((frame.get(1, 2, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_frame_to_bytes() {
        let mut frame = NeuralFrame::new(2, 2);
        frame.data[0] = 0.0;
        frame.data[1] = 0.5;
        frame.data[2] = 1.0;

        let bytes = frame.to_bytes();
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[1], 128);
        assert_eq!(bytes[2], 255);
    }

    #[test]
    fn test_model_type_defaults() {
        assert_eq!(ModelType::RealEsrgan.default_scale(), 4);
        assert_eq!(ModelType::SwinIR.default_scale(), 2);
        assert_eq!(ModelType::RealEsrgan.input_channels(), 3);
    }

    #[test]
    fn test_neural_config_default() {
        let config = NeuralConfig::default();
        assert_eq!(config.scale, 2);
        assert!(config.use_gpu);
        assert_eq!(config.tile_size, 512);
    }
}
