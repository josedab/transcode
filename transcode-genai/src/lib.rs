//! Generative AI extensions for video transcoding.
//!
//! This crate provides AI-powered video enhancement capabilities including:
//!
//! - **Super Resolution**: Upscale video using neural networks (ESRGAN, Real-ESRGAN)
//! - **Style Transfer**: Apply artistic styles to video frames
//! - **Frame Interpolation**: Generate intermediate frames for smooth slow-mo (RIFE)
//! - **Background Removal**: Remove or replace video backgrounds (U2-Net)
//! - **Face Enhancement**: Restore and enhance faces (GFPGAN)
//!
//! # Example
//!
//! ```ignore
//! use transcode_genai::{
//!     GenAiPipeline, PipelinePresets, ModelRegistry,
//!     FrameData, ProcessorConfig,
//! };
//! use std::sync::Arc;
//!
//! // Create model registry
//! let registry = Arc::new(ModelRegistry::new());
//!
//! // Create a 4x upscaling pipeline
//! let pipeline = PipelinePresets::upscale_4x(registry.clone())?;
//!
//! // Process a frame
//! let input = FrameData::new(480, 270, 3);
//! let output = pipeline.process_frame(input)?;
//!
//! // Output is 1920x1080
//! assert_eq!(output.width, 1920);
//! assert_eq!(output.height, 1080);
//! ```
//!
//! # Custom Pipelines
//!
//! You can create custom pipelines with multiple stages:
//!
//! ```ignore
//! let pipeline = GenAiPipeline::builder(registry)
//!     .super_resolution("realesrgan-x2")
//!     .style_transfer("fast-style-transfer", 0.5)
//!     .background_removal("rembg-u2net")
//!     .build()?;
//! ```
//!
//! # Model Management
//!
//! The crate includes a model registry with built-in model definitions:
//!
//! ```ignore
//! let registry = ModelRegistry::new();
//!
//! // List all super resolution models
//! for model in registry.list_by_type(ModelType::SuperResolution) {
//!     println!("{}: {}", model.id, model.name);
//! }
//!
//! // Register a custom model
//! registry.register(custom_model_info, Some(model_path));
//! ```
//!
//! # Supported Models
//!
//! ## Super Resolution
//! - `esrgan-x4`: ESRGAN 4x upscaler
//! - `realesrgan-x2`: Real-ESRGAN 2x photo-realistic upscaling
//!
//! ## Style Transfer
//! - `fast-style-transfer`: Real-time arbitrary style transfer
//!
//! ## Frame Interpolation
//! - `rife-v4`: RIFE v4 for frame rate conversion
//!
//! ## Background Removal
//! - `rembg-u2net`: U2-Net based background removal
//!
//! ## Face Enhancement
//! - `gfpgan-v1.4`: GFPGAN face restoration

#![allow(dead_code)]

mod error;
mod inference;
mod models;
mod pipeline;
mod processors;

pub use error::{GenAiError, Result};
pub use inference::{
    BatchInference, InferenceResult, InferenceSession, ModelCache, QualityValidator,
    ValidationResult,
};
pub use models::{
    ModelBackend, ModelInfo, ModelPrecision, ModelRegistry, ModelType, TensorDType, TensorFormat,
    TensorSpec,
};
pub use pipeline::{
    GenAiPipeline, PipelineBuilder, PipelinePresets, PipelineStage, PipelineStats, ProcessorType,
};
pub use processors::{
    BackgroundRemovalProcessor, ColorCorrectionProcessor, DenoisingProcessor,
    FaceEnhancementProcessor, FrameData, FrameInterpolationProcessor, FrameProcessor,
    ProcessorConfig, StyleTransferProcessor, SuperResolutionProcessor,
};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_end_to_end_upscaling() {
        let registry = Arc::new(ModelRegistry::new());

        // Create 4x upscaling pipeline
        let pipeline = PipelinePresets::upscale_4x(registry).unwrap();

        // Create input frame
        let input = FrameData::new(480, 270, 3);

        // Process
        let output = pipeline.process_frame(input).unwrap();

        // Verify dimensions
        assert_eq!(output.width, 1920);
        assert_eq!(output.height, 1080);
    }

    #[test]
    fn test_multi_stage_processing() {
        let registry = Arc::new(ModelRegistry::new());

        let pipeline = GenAiPipeline::builder(registry)
            .super_resolution("realesrgan-x2")
            .style_transfer("fast-style-transfer", 0.7)
            .build()
            .unwrap();

        let input = FrameData::new(100, 100, 3);
        let output = pipeline.process_frame(input).unwrap();

        // 2x upscale
        assert_eq!(output.width, 200);
        assert_eq!(output.height, 200);
    }

    #[test]
    fn test_frame_interpolation() {
        let registry = Arc::new(ModelRegistry::new());
        let config = ProcessorConfig::for_model("rife-v4");

        let processor = FrameInterpolationProcessor::new(config, &registry).unwrap();

        // Create two frames
        let mut frame1 = FrameData::new(100, 100, 3).with_timestamp(0);
        let mut frame2 = FrameData::new(100, 100, 3).with_timestamp(100);

        // Set different pixel values
        frame1.data.fill(100);
        frame2.data.fill(200);

        // Interpolate
        let middle = processor.interpolate(&frame1, &frame2, 0.5).unwrap();

        assert_eq!(middle.timestamp_ms, 50);
        // Check pixels are blended
        assert!(middle.data[0] > 100 && middle.data[0] < 200);
    }

    #[test]
    fn test_model_registry() {
        let registry = ModelRegistry::new();

        // Check built-in models
        assert!(registry.contains("esrgan-x4"));
        assert!(registry.contains("fast-style-transfer"));
        assert!(registry.contains("rife-v4"));
        assert!(registry.contains("rembg-u2net"));
        assert!(registry.contains("gfpgan-v1.4"));

        // List by type
        let sr_models = registry.list_by_type(ModelType::SuperResolution);
        assert!(sr_models.len() >= 2);
    }

    #[test]
    fn test_pipeline_stats() {
        let registry = Arc::new(ModelRegistry::new());

        let pipeline = GenAiPipeline::builder(registry)
            .super_resolution("esrgan-x4")
            .style_transfer("fast-style-transfer", 0.5)
            .build()
            .unwrap();

        let stats = pipeline.stats();
        assert_eq!(stats.total_stages, 2);
        assert_eq!(stats.enabled_stages, 2);
        assert!(stats.estimated_memory_mb > 0);
    }
}
