//! Model management for neural super-resolution.
//!
//! This module provides:
//! - Model registry with supported models
//! - Model downloading from URLs
//! - Model validation (input/output shapes)
//! - Model metadata extraction
//! - Local model caching

use crate::{ModelType, NeuralError, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Information about a super-resolution model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Model type.
    pub model_type: ModelType,
    /// Default scale factor.
    pub scale: u32,
    /// Input shape (NCHW, None for dynamic).
    pub input_shape: [Option<usize>; 4],
    /// Output shape (NCHW, None for dynamic).
    pub output_shape: [Option<usize>; 4],
    /// Download URL.
    pub url: Option<String>,
    /// SHA256 checksum for validation.
    pub sha256: Option<String>,
    /// File size in bytes.
    pub file_size: Option<u64>,
    /// Description.
    pub description: String,
}

impl ModelInfo {
    /// Validate that input dimensions are compatible.
    pub fn validate_input(&self, batch: usize, channels: usize, _height: usize, _width: usize) -> Result<()> {
        if let Some(expected_batch) = self.input_shape[0] {
            if batch != expected_batch {
                return Err(NeuralError::InvalidInput(format!(
                    "Expected batch size {}, got {}",
                    expected_batch, batch
                )));
            }
        }
        if let Some(expected_channels) = self.input_shape[1] {
            if channels != expected_channels {
                return Err(NeuralError::InvalidInput(format!(
                    "Expected {} channels, got {}",
                    expected_channels, channels
                )));
            }
        }
        // Height and width are typically dynamic
        Ok(())
    }

    /// Calculate expected output dimensions.
    pub fn output_dimensions(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        (
            input_height * self.scale as usize,
            input_width * self.scale as usize,
        )
    }
}

/// Registry of supported super-resolution models.
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    /// Create a new model registry with built-in models.
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
        };
        registry.register_builtin_models();
        registry
    }

    /// Register built-in super-resolution models.
    fn register_builtin_models(&mut self) {
        // Real-ESRGAN 4x
        self.register(ModelInfo {
            id: "realesrgan-x4".to_string(),
            name: "Real-ESRGAN x4".to_string(),
            model_type: ModelType::RealEsrgan,
            scale: 4,
            input_shape: [Some(1), Some(3), None, None],
            output_shape: [Some(1), Some(3), None, None],
            url: Some("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.onnx".to_string()),
            sha256: None,
            file_size: Some(64_000_000),
            description: "General-purpose 4x upscaling model, good for photos and realistic content".to_string(),
        });

        // Real-ESRGAN Anime 4x
        self.register(ModelInfo {
            id: "realesrgan-anime-x4".to_string(),
            name: "Real-ESRGAN Anime x4".to_string(),
            model_type: ModelType::RealEsrganAnime,
            scale: 4,
            input_shape: [Some(1), Some(3), None, None],
            output_shape: [Some(1), Some(3), None, None],
            url: Some("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.onnx".to_string()),
            sha256: None,
            file_size: Some(17_000_000),
            description: "Optimized for anime and illustration content".to_string(),
        });

        // BSRGAN 4x
        self.register(ModelInfo {
            id: "bsrgan-x4".to_string(),
            name: "BSRGAN x4".to_string(),
            model_type: ModelType::Bsrgan,
            scale: 4,
            input_shape: [Some(1), Some(3), None, None],
            output_shape: [Some(1), Some(3), None, None],
            url: None, // User must provide
            sha256: None,
            file_size: None,
            description: "Blind super-resolution for unknown degradations".to_string(),
        });

        // SwinIR 2x
        self.register(ModelInfo {
            id: "swinir-x2".to_string(),
            name: "SwinIR x2".to_string(),
            model_type: ModelType::SwinIR,
            scale: 2,
            input_shape: [Some(1), Some(3), None, None],
            output_shape: [Some(1), Some(3), None, None],
            url: None,
            sha256: None,
            file_size: None,
            description: "Transformer-based SR with excellent quality".to_string(),
        });

        // SwinIR 4x
        self.register(ModelInfo {
            id: "swinir-x4".to_string(),
            name: "SwinIR x4".to_string(),
            model_type: ModelType::SwinIR,
            scale: 4,
            input_shape: [Some(1), Some(3), None, None],
            output_shape: [Some(1), Some(3), None, None],
            url: None,
            sha256: None,
            file_size: None,
            description: "Transformer-based 4x SR with excellent quality".to_string(),
        });

        // RealBasicVSR (Video SR)
        self.register(ModelInfo {
            id: "realbasicvsr".to_string(),
            name: "RealBasicVSR".to_string(),
            model_type: ModelType::Custom("realbasicvsr.onnx".to_string()),
            scale: 4,
            input_shape: [None, Some(3), None, None], // Dynamic batch for temporal
            output_shape: [None, Some(3), None, None],
            url: None,
            sha256: None,
            file_size: None,
            description: "Video super-resolution with temporal consistency".to_string(),
        });
    }

    /// Register a model.
    pub fn register(&mut self, info: ModelInfo) {
        self.models.insert(info.id.clone(), info);
    }

    /// Get model info by ID.
    pub fn get(&self, id: &str) -> Option<&ModelInfo> {
        self.models.get(id)
    }

    /// Get model info by type.
    pub fn get_by_type(&self, model_type: &ModelType) -> Option<&ModelInfo> {
        self.models.values().find(|m| &m.model_type == model_type)
    }

    /// List all registered models.
    pub fn list(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// List models with given scale factor.
    pub fn list_by_scale(&self, scale: u32) -> Vec<&ModelInfo> {
        self.models.values().filter(|m| m.scale == scale).collect()
    }
}

/// Model manager for downloading and caching models.
pub struct ModelManager {
    cache_dir: PathBuf,
    registry: ModelRegistry,
}

impl ModelManager {
    /// Create a new model manager with the given cache directory.
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            registry: ModelRegistry::new(),
        }
    }

    /// Create with default cache directory.
    pub fn with_default_cache() -> Result<Self> {
        let cache_dir = dirs_cache_dir()
            .ok_or_else(|| NeuralError::ModelLoad("Could not determine cache directory".to_string()))?;
        Ok(Self::new(cache_dir))
    }

    /// Get the cache directory.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get the model registry.
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    /// Get mutable registry for adding custom models.
    pub fn registry_mut(&mut self) -> &mut ModelRegistry {
        &mut self.registry
    }

    /// Get path for a model (downloading if necessary).
    pub fn get_model_path(&self, model_type: &ModelType) -> Result<PathBuf> {
        let filename = self.model_filename(model_type);
        let path = self.cache_dir.join(&filename);

        if path.exists() {
            return Ok(path);
        }

        // Check if we have a URL to download from
        if let Some(info) = self.registry.get_by_type(model_type) {
            if info.url.is_some() {
                #[cfg(feature = "download")]
                {
                    // Would trigger async download
                    return Err(NeuralError::ModelLoad(format!(
                        "Model {} not found. Use download_model() to fetch it.",
                        filename
                    )));
                }
                #[cfg(not(feature = "download"))]
                {
                    return Err(NeuralError::ModelLoad(format!(
                        "Model {} not found. Enable 'download' feature or download manually.",
                        filename
                    )));
                }
            }
        }

        Err(NeuralError::ModelLoad(format!(
            "Model not found: {:?}. Place the model file at {:?}",
            model_type, path
        )))
    }

    /// Get model filename for a model type.
    fn model_filename(&self, model_type: &ModelType) -> String {
        match model_type {
            ModelType::RealEsrgan => "realesrgan_x4.onnx".to_string(),
            ModelType::RealEsrganAnime => "realesrgan_anime_x4.onnx".to_string(),
            ModelType::Bsrgan => "bsrgan_x4.onnx".to_string(),
            ModelType::SwinIR => "swinir_x2.onnx".to_string(),
            ModelType::Custom(name) => name.clone(),
        }
    }

    /// Download a model by ID.
    #[cfg(feature = "download")]
    pub async fn download_model(&self, model_id: &str) -> Result<PathBuf> {
        use sha2::{Digest, Sha256};
        use tokio::io::AsyncWriteExt;

        let info = self
            .registry
            .get(model_id)
            .ok_or_else(|| NeuralError::ModelLoad(format!("Unknown model: {}", model_id)))?;

        let url = info
            .url
            .as_ref()
            .ok_or_else(|| NeuralError::ModelLoad(format!("No download URL for model: {}", model_id)))?;

        let filename = self.model_filename(&info.model_type);
        let path = self.cache_dir.join(&filename);

        // Create cache directory
        tokio::fs::create_dir_all(&self.cache_dir)
            .await
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to create cache dir: {}", e)))?;

        tracing::info!("Downloading model {} from {}", model_id, url);

        // Download file
        let response = reqwest::get(url)
            .await
            .map_err(|e| NeuralError::ModelLoad(format!("Download failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(NeuralError::ModelLoad(format!(
                "Download failed with status: {}",
                response.status()
            )));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to read response: {}", e)))?;

        // Verify checksum if provided
        if let Some(expected_sha256) = &info.sha256 {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let hash = format!("{:x}", hasher.finalize());
            if &hash != expected_sha256 {
                return Err(NeuralError::ModelLoad(format!(
                    "Checksum mismatch: expected {}, got {}",
                    expected_sha256, hash
                )));
            }
        }

        // Write to file
        let mut file = tokio::fs::File::create(&path)
            .await
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to create file: {}", e)))?;

        file.write_all(&bytes)
            .await
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to write file: {}", e)))?;

        tracing::info!("Model downloaded to {:?}", path);

        Ok(path)
    }

    /// Validate a model file.
    #[cfg(feature = "onnx")]
    pub fn validate_model(&self, path: &Path) -> Result<ModelMetadata> {
        use ort::session::Session;

        let session = Session::builder()
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to create session: {}", e)))?
            .commit_from_file(path)
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to load model: {}", e)))?;

        let input_info = session.inputs.first().ok_or_else(|| {
            NeuralError::ModelLoad("Model has no inputs".to_string())
        })?;

        let output_info = session.outputs.first().ok_or_else(|| {
            NeuralError::ModelLoad("Model has no outputs".to_string())
        })?;

        let input_shape = match &input_info.input_type {
            ort::session::input::Input::Tensor { dimensions, .. } => {
                dimensions.iter().map(|d| d.map(|v| v as usize)).collect()
            }
        };

        let output_shape = match &output_info.output_type {
            ort::session::output::Output::Tensor { dimensions, .. } => {
                dimensions.iter().map(|d| d.map(|v| v as usize)).collect()
            }
        };

        Ok(ModelMetadata {
            input_name: input_info.name.clone(),
            output_name: output_info.name.clone(),
            input_shape,
            output_shape,
            num_inputs: session.inputs.len(),
            num_outputs: session.outputs.len(),
        })
    }

    /// Validate without ONNX (stub).
    #[cfg(not(feature = "onnx"))]
    pub fn validate_model(&self, _path: &Path) -> Result<ModelMetadata> {
        Err(NeuralError::OnnxNotEnabled)
    }

    /// List cached models.
    pub fn list_cached(&self) -> Result<Vec<PathBuf>> {
        if !self.cache_dir.exists() {
            return Ok(Vec::new());
        }

        let entries: Vec<PathBuf> = std::fs::read_dir(&self.cache_dir)
            .map_err(|e| NeuralError::ModelLoad(format!("Failed to read cache dir: {}", e)))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .map(|ext| ext == "onnx")
                    .unwrap_or(false)
            })
            .map(|entry| entry.path())
            .collect();

        Ok(entries)
    }

    /// Clear the model cache.
    pub fn clear_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)
                .map_err(|e| NeuralError::ModelLoad(format!("Failed to clear cache: {}", e)))?;
        }
        Ok(())
    }
}

/// Model metadata extracted from ONNX file.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Input tensor name.
    pub input_name: String,
    /// Output tensor name.
    pub output_name: String,
    /// Input shape (None for dynamic dimensions).
    pub input_shape: Vec<Option<usize>>,
    /// Output shape (None for dynamic dimensions).
    pub output_shape: Vec<Option<usize>>,
    /// Number of inputs.
    pub num_inputs: usize,
    /// Number of outputs.
    pub num_outputs: usize,
}

impl ModelMetadata {
    /// Infer scale factor from input/output shapes.
    pub fn infer_scale(&self) -> Option<u32> {
        if self.input_shape.len() != 4 || self.output_shape.len() != 4 {
            return None;
        }

        // Compare height dimensions (index 2)
        match (self.input_shape[2], self.output_shape[2]) {
            (Some(in_h), Some(out_h)) if in_h > 0 => {
                Some((out_h / in_h) as u32)
            }
            _ => None,
        }
    }
}

/// Get platform-specific cache directory.
fn dirs_cache_dir() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join("Library/Caches/transcode/models"))
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var("LOCALAPPDATA")
            .ok()
            .map(|d| PathBuf::from(d).join("transcode/cache/models"))
    }
    #[cfg(target_os = "linux")]
    {
        std::env::var("XDG_CACHE_HOME")
            .ok()
            .or_else(|| std::env::var("HOME").ok().map(|h| format!("{}/.cache", h)))
            .map(|d| PathBuf::from(d).join("transcode/models"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".transcode/models"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_registry() {
        let registry = ModelRegistry::new();

        // Should have builtin models
        assert!(!registry.list().is_empty());

        // Check Real-ESRGAN
        let esrgan = registry.get("realesrgan-x4");
        assert!(esrgan.is_some());
        let esrgan = esrgan.unwrap();
        assert_eq!(esrgan.scale, 4);
        assert_eq!(esrgan.model_type, ModelType::RealEsrgan);
    }

    #[test]
    fn test_model_info_validation() {
        let info = ModelInfo {
            id: "test".to_string(),
            name: "Test Model".to_string(),
            model_type: ModelType::RealEsrgan,
            scale: 4,
            input_shape: [Some(1), Some(3), None, None],
            output_shape: [Some(1), Some(3), None, None],
            url: None,
            sha256: None,
            file_size: None,
            description: "Test".to_string(),
        };

        // Valid input
        assert!(info.validate_input(1, 3, 64, 64).is_ok());

        // Wrong batch
        assert!(info.validate_input(2, 3, 64, 64).is_err());

        // Wrong channels
        assert!(info.validate_input(1, 4, 64, 64).is_err());
    }

    #[test]
    fn test_output_dimensions() {
        let info = ModelInfo {
            id: "test".to_string(),
            name: "Test".to_string(),
            model_type: ModelType::RealEsrgan,
            scale: 4,
            input_shape: [Some(1), Some(3), None, None],
            output_shape: [Some(1), Some(3), None, None],
            url: None,
            sha256: None,
            file_size: None,
            description: "Test".to_string(),
        };

        let (h, w) = info.output_dimensions(128, 256);
        assert_eq!(h, 512);
        assert_eq!(w, 1024);
    }

    #[test]
    fn test_list_by_scale() {
        let registry = ModelRegistry::new();

        let x4_models = registry.list_by_scale(4);
        assert!(!x4_models.is_empty());
        for model in x4_models {
            assert_eq!(model.scale, 4);
        }
    }

    #[test]
    fn test_model_manager() {
        let temp_dir = std::env::temp_dir().join("transcode_test_models");
        let manager = ModelManager::new(temp_dir.clone());

        assert_eq!(manager.cache_dir(), &temp_dir);
        assert!(!manager.registry().list().is_empty());
    }

    #[test]
    fn test_model_metadata_infer_scale() {
        let meta = ModelMetadata {
            input_name: "input".to_string(),
            output_name: "output".to_string(),
            input_shape: vec![Some(1), Some(3), Some(64), Some(64)],
            output_shape: vec![Some(1), Some(3), Some(256), Some(256)],
            num_inputs: 1,
            num_outputs: 1,
        };

        assert_eq!(meta.infer_scale(), Some(4));
    }
}
