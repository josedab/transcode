//! Neural network model management.

use crate::error::{AiError, Result};
#[cfg(feature = "onnx")]
use crate::onnx::{OnnxConfig, OnnxSession};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Backend for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelBackend {
    /// CPU inference.
    #[default]
    Cpu,
    /// GPU inference via CUDA.
    Cuda,
    /// GPU inference via DirectML (Windows).
    DirectML,
    /// GPU inference via CoreML (macOS).
    CoreML,
    /// GPU inference via TensorRT.
    TensorRT,
}

impl ModelBackend {
    /// Check if this backend is available on the current system.
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Cuda => cfg!(feature = "onnx"),
            Self::DirectML => cfg!(target_os = "windows") && cfg!(feature = "onnx"),
            Self::CoreML => cfg!(target_os = "macos") && cfg!(feature = "onnx"),
            Self::TensorRT => cfg!(feature = "onnx"),
        }
    }

    /// Get the best available backend.
    pub fn best_available() -> Self {
        #[cfg(target_os = "macos")]
        if Self::CoreML.is_available() {
            return Self::CoreML;
        }

        #[cfg(target_os = "windows")]
        if Self::DirectML.is_available() {
            return Self::DirectML;
        }

        if Self::Cuda.is_available() {
            return Self::Cuda;
        }

        Self::Cpu
    }
}

/// Model information.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name.
    pub name: String,
    /// Model version.
    pub version: String,
    /// Expected input dimensions [batch, channels, height, width].
    pub input_shape: Vec<i64>,
    /// Output dimensions.
    pub output_shape: Vec<i64>,
    /// Scale factor for upscaling models.
    pub scale_factor: Option<u32>,
    /// Model file path.
    pub path: PathBuf,
}

/// Model loader and manager.
pub struct ModelLoader {
    /// Model search paths.
    search_paths: Vec<PathBuf>,
    /// Preferred backend.
    backend: ModelBackend,
}

impl ModelLoader {
    /// Create a new model loader.
    pub fn new() -> Self {
        let mut search_paths = vec![
            PathBuf::from("models"),
            PathBuf::from("./models"),
        ];

        // Add system paths
        if let Ok(home) = std::env::var("HOME") {
            search_paths.push(PathBuf::from(format!("{}/.transcode/models", home)));
        }

        #[cfg(target_os = "linux")]
        {
            search_paths.push(PathBuf::from("/usr/share/transcode/models"));
            search_paths.push(PathBuf::from("/usr/local/share/transcode/models"));
        }

        #[cfg(target_os = "macos")]
        {
            search_paths.push(PathBuf::from("/usr/local/share/transcode/models"));
        }

        Self {
            search_paths,
            backend: ModelBackend::best_available(),
        }
    }

    /// Set the inference backend.
    pub fn with_backend(mut self, backend: ModelBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Add a search path.
    pub fn with_search_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.search_paths.push(path.into());
        self
    }

    /// Find a model file by name.
    pub fn find_model(&self, name: &str) -> Result<PathBuf> {
        let extensions = ["onnx", "pt", "pb"];

        for base_path in &self.search_paths {
            for ext in &extensions {
                let path = base_path.join(format!("{}.{}", name, ext));
                if path.exists() {
                    debug!("Found model: {:?}", path);
                    return Ok(path);
                }
            }
        }

        Err(AiError::ModelNotFound(format!(
            "Model '{}' not found in search paths: {:?}",
            name, self.search_paths
        )))
    }

    /// Load model information.
    pub fn load_info(&self, path: &Path) -> Result<ModelInfo> {
        if !path.exists() {
            return Err(AiError::ModelNotFound(path.display().to_string()));
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Determine scale factor from model name
        let scale_factor = if name.contains("x2") || name.contains("2x") {
            Some(2)
        } else if name.contains("x4") || name.contains("4x") {
            Some(4)
        } else {
            None
        };

        info!("Loading model info: {}", name);

        Ok(ModelInfo {
            name,
            version: "1.0".to_string(),
            input_shape: vec![-1, 3, -1, -1], // Dynamic batch, 3 channels, dynamic HxW
            output_shape: vec![-1, 3, -1, -1],
            scale_factor,
            path: path.to_path_buf(),
        })
    }

    /// Get current backend.
    pub fn backend(&self) -> ModelBackend {
        self.backend
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Inference session wrapper with optional ONNX runtime support.
pub struct InferenceSession {
    /// Model info.
    info: ModelInfo,
    /// Backend being used.
    #[allow(dead_code)]
    backend: ModelBackend,
    /// ONNX session (when onnx feature is enabled).
    #[cfg(feature = "onnx")]
    onnx_session: Option<OnnxSession>,
}

impl InferenceSession {
    /// Create a new inference session.
    pub fn new(model_path: &Path, backend: ModelBackend) -> Result<Self> {
        let loader = ModelLoader::new().with_backend(backend);
        let info = loader.load_info(model_path)?;

        info!("Created inference session for: {}", info.name);

        // Try to create ONNX session if feature is enabled
        #[cfg(feature = "onnx")]
        let onnx_session = if model_path.extension().map_or(false, |ext| ext == "onnx") {
            let config = Self::backend_to_onnx_config(backend);
            match OnnxSession::load(model_path, config) {
                Ok(session) => {
                    info!("ONNX session created with provider: {}", session.active_provider().name());
                    Some(session)
                }
                Err(e) => {
                    tracing::warn!("Failed to create ONNX session, using fallback: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            info,
            backend,
            #[cfg(feature = "onnx")]
            onnx_session,
        })
    }

    /// Create an ONNX config from a ModelBackend.
    #[cfg(feature = "onnx")]
    fn backend_to_onnx_config(backend: ModelBackend) -> OnnxConfig {
        use crate::onnx::ExecutionProvider;

        let provider = match backend {
            ModelBackend::Cpu => ExecutionProvider::Cpu,
            ModelBackend::Cuda => ExecutionProvider::Cuda { device_id: 0 },
            ModelBackend::TensorRT => ExecutionProvider::TensorRT {
                device_id: 0,
                fp16: true,
                max_workspace_size: 1 << 30,
            },
            ModelBackend::CoreML => ExecutionProvider::CoreML {
                use_neural_engine: true,
                only_neural_engine: false,
            },
            ModelBackend::DirectML => ExecutionProvider::DirectML { device_id: 0 },
        };

        OnnxConfig::new().with_provider(provider)
    }

    /// Get model info.
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Check if ONNX runtime is being used.
    pub fn is_using_onnx(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            self.onnx_session.is_some()
        }
        #[cfg(not(feature = "onnx"))]
        {
            false
        }
    }

    /// Get the active execution provider name.
    pub fn active_provider(&self) -> &str {
        #[cfg(feature = "onnx")]
        if let Some(ref session) = self.onnx_session {
            return session.active_provider().name();
        }
        "Fallback"
    }

    /// Run inference.
    pub fn run(&self, input: &ndarray::Array4<f32>) -> Result<ndarray::Array4<f32>> {
        // Try ONNX session first if available
        #[cfg(feature = "onnx")]
        if let Some(ref session) = self.onnx_session {
            return session.run_image(input);
        }

        // Fallback implementation using bilinear interpolation
        self.run_fallback(input)
    }

    /// Fallback implementation when ONNX is not available.
    fn run_fallback(&self, input: &ndarray::Array4<f32>) -> Result<ndarray::Array4<f32>> {
        let (batch, channels, height, width) = input.dim();

        // Determine output size based on scale factor
        let scale = self.info.scale_factor.unwrap_or(1) as usize;
        let out_height = height * scale;
        let out_width = width * scale;

        // Use bilinear interpolation as a fallback
        let output = ndarray::Array4::from_shape_fn(
            (batch, channels, out_height, out_width),
            |(b, c, y, x)| {
                let src_y = y as f32 / scale as f32;
                let src_x = x as f32 / scale as f32;

                // Bilinear interpolation
                let y0 = src_y.floor() as usize;
                let x0 = src_x.floor() as usize;
                let y1 = (y0 + 1).min(height - 1);
                let x1 = (x0 + 1).min(width - 1);

                let fy = src_y - y0 as f32;
                let fx = src_x - x0 as f32;

                let v00 = input[[b, c, y0, x0]];
                let v01 = input[[b, c, y0, x1]];
                let v10 = input[[b, c, y1, x0]];
                let v11 = input[[b, c, y1, x1]];

                (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01)
                    + fy * ((1.0 - fx) * v10 + fx * v11)
            },
        );

        Ok(output)
    }

    /// Run inference with raw byte input (u8 image data).
    pub fn run_image_u8(
        &self,
        data: &[u8],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Result<(Vec<u8>, usize, usize)> {
        use crate::onnx::{preprocess_image, postprocess_image};

        let input = preprocess_image(data, width, height, channels);
        let output = self.run(&input)?;
        Ok(postprocess_image(&output))
    }
}
