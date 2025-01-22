//! Model definitions and management for GenAI operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Model type categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    /// Super resolution / upscaling.
    SuperResolution,
    /// Style transfer.
    StyleTransfer,
    /// Video generation.
    VideoGeneration,
    /// Frame interpolation.
    FrameInterpolation,
    /// Face enhancement.
    FaceEnhancement,
    /// Background removal/replacement.
    BackgroundRemoval,
    /// Object detection/tracking.
    ObjectDetection,
    /// Content generation.
    ContentGeneration,
    /// Denoising.
    Denoising,
    /// Color correction.
    ColorCorrection,
}

/// Model precision/quantization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelPrecision {
    /// Full 32-bit floating point.
    #[default]
    Fp32,
    /// 16-bit floating point.
    Fp16,
    /// BFloat16.
    Bf16,
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit quantization.
    Int4,
}

/// Model backend/runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelBackend {
    /// ONNX Runtime.
    #[default]
    Onnx,
    /// TensorRT (NVIDIA).
    TensorRt,
    /// CoreML (Apple).
    CoreMl,
    /// OpenVINO (Intel).
    OpenVino,
    /// DirectML (Windows).
    DirectMl,
    /// WebNN (Browser).
    WebNn,
    /// Custom/native Rust.
    Native,
}

/// Model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Model type.
    pub model_type: ModelType,
    /// Model version.
    pub version: String,
    /// Description.
    pub description: Option<String>,
    /// Input specification.
    pub input_spec: TensorSpec,
    /// Output specification.
    pub output_spec: TensorSpec,
    /// Supported backends.
    pub supported_backends: Vec<ModelBackend>,
    /// Default precision.
    pub default_precision: ModelPrecision,
    /// Memory requirements (MB).
    pub memory_mb: u64,
    /// License information.
    pub license: Option<String>,
}

/// Tensor specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Shape (dimensions).
    pub shape: Vec<i64>,
    /// Data type.
    pub dtype: TensorDType,
    /// Channel format.
    pub format: TensorFormat,
}

/// Tensor data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TensorDType {
    Float32,
    Float16,
    Int32,
    Int64,
    Uint8,
    Int8,
}

/// Tensor format (channel ordering).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TensorFormat {
    /// Batch, Channels, Height, Width.
    #[default]
    Nchw,
    /// Batch, Height, Width, Channels.
    Nhwc,
    /// Batch, Time, Channels, Height, Width.
    Ntchw,
    /// Batch, Time, Height, Width, Channels.
    Nthwc,
}

/// Model registry for managing available models.
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
    model_paths: HashMap<String, PathBuf>,
}

impl ModelRegistry {
    /// Create a new model registry.
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
            model_paths: HashMap::new(),
        };
        registry.register_builtin_models();
        registry
    }

    /// Register a model.
    pub fn register(&mut self, info: ModelInfo, path: Option<PathBuf>) {
        let id = info.id.clone();
        self.models.insert(id.clone(), info);
        if let Some(path) = path {
            self.model_paths.insert(id, path);
        }
    }

    /// Get model info by ID.
    pub fn get(&self, id: &str) -> Option<&ModelInfo> {
        self.models.get(id)
    }

    /// Get model path.
    pub fn get_path(&self, id: &str) -> Option<&PathBuf> {
        self.model_paths.get(id)
    }

    /// List all models.
    pub fn list(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// List models by type.
    pub fn list_by_type(&self, model_type: ModelType) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|m| m.model_type == model_type)
            .collect()
    }

    /// Check if a model is registered.
    pub fn contains(&self, id: &str) -> bool {
        self.models.contains_key(id)
    }

    fn register_builtin_models(&mut self) {
        // Super resolution models
        self.register(
            ModelInfo {
                id: "esrgan-x4".to_string(),
                name: "ESRGAN 4x Upscaler".to_string(),
                model_type: ModelType::SuperResolution,
                version: "1.0.0".to_string(),
                description: Some("Enhanced Super-Resolution GAN for 4x upscaling".to_string()),
                input_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx, ModelBackend::TensorRt],
                default_precision: ModelPrecision::Fp16,
                memory_mb: 512,
                license: Some("BSD-3-Clause".to_string()),
            },
            None,
        );

        self.register(
            ModelInfo {
                id: "realesrgan-x2".to_string(),
                name: "Real-ESRGAN 2x".to_string(),
                model_type: ModelType::SuperResolution,
                version: "1.0.0".to_string(),
                description: Some("Real-ESRGAN for 2x photo-realistic upscaling".to_string()),
                input_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx, ModelBackend::TensorRt],
                default_precision: ModelPrecision::Fp16,
                memory_mb: 256,
                license: Some("BSD-3-Clause".to_string()),
            },
            None,
        );

        // Style transfer models
        self.register(
            ModelInfo {
                id: "fast-style-transfer".to_string(),
                name: "Fast Neural Style Transfer".to_string(),
                model_type: ModelType::StyleTransfer,
                version: "1.0.0".to_string(),
                description: Some("Real-time arbitrary style transfer".to_string()),
                input_spec: TensorSpec {
                    shape: vec![-1, 3, 512, 512],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, 512, 512],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx, ModelBackend::CoreMl],
                default_precision: ModelPrecision::Fp32,
                memory_mb: 128,
                license: Some("MIT".to_string()),
            },
            None,
        );

        // Frame interpolation
        self.register(
            ModelInfo {
                id: "rife-v4".to_string(),
                name: "RIFE v4 Frame Interpolation".to_string(),
                model_type: ModelType::FrameInterpolation,
                version: "4.0.0".to_string(),
                description: Some(
                    "Real-Time Intermediate Flow Estimation for frame interpolation".to_string(),
                ),
                input_spec: TensorSpec {
                    shape: vec![-1, 6, -1, -1], // Two concatenated frames
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx, ModelBackend::TensorRt],
                default_precision: ModelPrecision::Fp16,
                memory_mb: 384,
                license: Some("MIT".to_string()),
            },
            None,
        );

        // Background removal
        self.register(
            ModelInfo {
                id: "rembg-u2net".to_string(),
                name: "U2-Net Background Removal".to_string(),
                model_type: ModelType::BackgroundRemoval,
                version: "1.0.0".to_string(),
                description: Some("U2-Net based background removal".to_string()),
                input_spec: TensorSpec {
                    shape: vec![-1, 3, 320, 320],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 1, 320, 320],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx],
                default_precision: ModelPrecision::Fp32,
                memory_mb: 176,
                license: Some("Apache-2.0".to_string()),
            },
            None,
        );

        // Face enhancement
        self.register(
            ModelInfo {
                id: "gfpgan-v1.4".to_string(),
                name: "GFPGAN Face Enhancement".to_string(),
                model_type: ModelType::FaceEnhancement,
                version: "1.4.0".to_string(),
                description: Some(
                    "GAN for face restoration and enhancement".to_string(),
                ),
                input_spec: TensorSpec {
                    shape: vec![-1, 3, 512, 512],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, 512, 512],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx, ModelBackend::TensorRt],
                default_precision: ModelPrecision::Fp16,
                memory_mb: 512,
                license: Some("Apache-2.0".to_string()),
            },
            None,
        );

        // Denoising models
        self.register(
            ModelInfo {
                id: "denoise-cnn".to_string(),
                name: "CNN Video Denoiser".to_string(),
                model_type: ModelType::Denoising,
                version: "1.0.0".to_string(),
                description: Some(
                    "CNN-based video denoising for noise reduction".to_string(),
                ),
                input_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx, ModelBackend::TensorRt],
                default_precision: ModelPrecision::Fp16,
                memory_mb: 256,
                license: Some("MIT".to_string()),
            },
            None,
        );

        self.register(
            ModelInfo {
                id: "denoise-nlm".to_string(),
                name: "Non-Local Means Denoiser".to_string(),
                model_type: ModelType::Denoising,
                version: "1.0.0".to_string(),
                description: Some(
                    "Neural non-local means denoising".to_string(),
                ),
                input_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx],
                default_precision: ModelPrecision::Fp32,
                memory_mb: 128,
                license: Some("MIT".to_string()),
            },
            None,
        );

        // Color correction models
        self.register(
            ModelInfo {
                id: "color-ai".to_string(),
                name: "AI Color Correction".to_string(),
                model_type: ModelType::ColorCorrection,
                version: "1.0.0".to_string(),
                description: Some(
                    "AI-powered automatic color correction and grading".to_string(),
                ),
                input_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, -1, -1],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx, ModelBackend::CoreMl],
                default_precision: ModelPrecision::Fp16,
                memory_mb: 192,
                license: Some("MIT".to_string()),
            },
            None,
        );

        self.register(
            ModelInfo {
                id: "color-match".to_string(),
                name: "Color Matching Network".to_string(),
                model_type: ModelType::ColorCorrection,
                version: "1.0.0".to_string(),
                description: Some(
                    "Neural color matching for consistent grading across clips".to_string(),
                ),
                input_spec: TensorSpec {
                    shape: vec![-1, 3, 256, 256],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, 256, 256],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx],
                default_precision: ModelPrecision::Fp32,
                memory_mb: 96,
                license: Some("Apache-2.0".to_string()),
            },
            None,
        );
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new();
        assert!(registry.contains("esrgan-x4"));
        assert!(registry.contains("fast-style-transfer"));
        assert!(registry.contains("rife-v4"));
    }

    #[test]
    fn test_list_by_type() {
        let registry = ModelRegistry::new();

        let sr_models = registry.list_by_type(ModelType::SuperResolution);
        assert!(!sr_models.is_empty());

        let style_models = registry.list_by_type(ModelType::StyleTransfer);
        assert!(!style_models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let registry = ModelRegistry::new();
        let model = registry.get("esrgan-x4").unwrap();

        assert_eq!(model.model_type, ModelType::SuperResolution);
        assert!(model.memory_mb > 0);
    }

    #[test]
    fn test_custom_registration() {
        let mut registry = ModelRegistry::new();

        registry.register(
            ModelInfo {
                id: "custom-model".to_string(),
                name: "Custom Model".to_string(),
                model_type: ModelType::Denoising,
                version: "1.0.0".to_string(),
                description: None,
                input_spec: TensorSpec {
                    shape: vec![-1, 3, 256, 256],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                output_spec: TensorSpec {
                    shape: vec![-1, 3, 256, 256],
                    dtype: TensorDType::Float32,
                    format: TensorFormat::Nchw,
                },
                supported_backends: vec![ModelBackend::Onnx],
                default_precision: ModelPrecision::Fp32,
                memory_mb: 64,
                license: None,
            },
            None,
        );

        assert!(registry.contains("custom-model"));
    }
}
