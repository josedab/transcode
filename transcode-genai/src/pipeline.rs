//! GenAI processing pipeline for video enhancement.

use crate::models::ModelRegistry;
use crate::processors::{FrameData, FrameProcessor, ProcessorConfig};
use crate::{GenAiError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Pipeline stage definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Stage name.
    pub name: String,
    /// Processor type.
    pub processor_type: ProcessorType,
    /// Configuration.
    pub config: ProcessorConfig,
    /// Whether stage is enabled.
    pub enabled: bool,
}

/// Processor type for pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorType {
    SuperResolution,
    StyleTransfer,
    FrameInterpolation,
    BackgroundRemoval,
    FaceEnhancement,
    Denoising,
    ColorCorrection,
}

/// GenAI processing pipeline.
pub struct GenAiPipeline {
    stages: Vec<PipelineStage>,
    registry: Arc<ModelRegistry>,
    processors: HashMap<String, Box<dyn FrameProcessor>>,
}

impl GenAiPipeline {
    /// Create a new pipeline.
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self {
            stages: Vec::new(),
            registry,
            processors: HashMap::new(),
        }
    }

    /// Create a pipeline builder.
    pub fn builder(registry: Arc<ModelRegistry>) -> PipelineBuilder {
        PipelineBuilder::new(registry)
    }

    /// Add a stage to the pipeline.
    pub fn add_stage(&mut self, stage: PipelineStage) -> Result<()> {
        // Validate model exists
        if !self.registry.contains(&stage.config.model_id) {
            return Err(GenAiError::ModelNotFound(stage.config.model_id.clone()));
        }

        // Create processor
        let processor = self.create_processor(&stage)?;
        self.processors.insert(stage.name.clone(), processor);
        self.stages.push(stage);

        Ok(())
    }

    /// Remove a stage by name.
    pub fn remove_stage(&mut self, name: &str) -> bool {
        if let Some(pos) = self.stages.iter().position(|s| s.name == name) {
            self.stages.remove(pos);
            self.processors.remove(name);
            true
        } else {
            false
        }
    }

    /// Enable/disable a stage.
    pub fn set_stage_enabled(&mut self, name: &str, enabled: bool) {
        if let Some(stage) = self.stages.iter_mut().find(|s| s.name == name) {
            stage.enabled = enabled;
        }
    }

    /// Get all stages.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Process a single frame through the pipeline.
    pub fn process_frame(&self, frame: FrameData) -> Result<FrameData> {
        let mut current = frame;

        for stage in &self.stages {
            if !stage.enabled {
                continue;
            }

            let processor = self
                .processors
                .get(&stage.name)
                .ok_or_else(|| GenAiError::ConfigError(format!("Processor not found: {}", stage.name)))?;

            tracing::debug!(
                stage = %stage.name,
                processor = %processor.name(),
                "Processing pipeline stage"
            );

            current = processor.process(&current)?;
        }

        Ok(current)
    }

    /// Process multiple frames.
    pub fn process_frames(&self, frames: Vec<FrameData>) -> Result<Vec<FrameData>> {
        frames.into_iter().map(|f| self.process_frame(f)).collect()
    }

    /// Get pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        let enabled_stages = self.stages.iter().filter(|s| s.enabled).count();
        let total_memory: u64 = self
            .stages
            .iter()
            .filter(|s| s.enabled)
            .filter_map(|s| self.registry.get(&s.config.model_id))
            .map(|m| m.memory_mb)
            .sum();

        PipelineStats {
            total_stages: self.stages.len(),
            enabled_stages,
            estimated_memory_mb: total_memory,
        }
    }

    fn create_processor(&self, stage: &PipelineStage) -> Result<Box<dyn FrameProcessor>> {
        use crate::processors::*;

        match stage.processor_type {
            ProcessorType::SuperResolution => Ok(Box::new(SuperResolutionProcessor::new(
                stage.config.clone(),
                &self.registry,
            )?)),
            ProcessorType::StyleTransfer => Ok(Box::new(StyleTransferProcessor::new(
                stage.config.clone(),
                &self.registry,
            )?)),
            ProcessorType::FrameInterpolation => Ok(Box::new(FrameInterpolationProcessor::new(
                stage.config.clone(),
                &self.registry,
            )?)),
            ProcessorType::BackgroundRemoval => Ok(Box::new(BackgroundRemovalProcessor::new(
                stage.config.clone(),
                &self.registry,
            )?)),
            ProcessorType::FaceEnhancement => Ok(Box::new(FaceEnhancementProcessor::new(
                stage.config.clone(),
                &self.registry,
            )?)),
            ProcessorType::Denoising => Ok(Box::new(DenoisingProcessor::new(
                stage.config.clone(),
                &self.registry,
            )?)),
            ProcessorType::ColorCorrection => Ok(Box::new(ColorCorrectionProcessor::new(
                stage.config.clone(),
                &self.registry,
            )?)),
        }
    }
}

/// Pipeline builder.
pub struct PipelineBuilder {
    registry: Arc<ModelRegistry>,
    stages: Vec<PipelineStage>,
}

impl PipelineBuilder {
    /// Create a new builder.
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        Self {
            registry,
            stages: Vec::new(),
        }
    }

    /// Add super resolution stage.
    pub fn super_resolution(mut self, model_id: impl Into<String>) -> Self {
        self.stages.push(PipelineStage {
            name: format!("sr_{}", self.stages.len()),
            processor_type: ProcessorType::SuperResolution,
            config: ProcessorConfig::for_model(model_id),
            enabled: true,
        });
        self
    }

    /// Add style transfer stage.
    pub fn style_transfer(mut self, model_id: impl Into<String>, strength: f32) -> Self {
        self.stages.push(PipelineStage {
            name: format!("style_{}", self.stages.len()),
            processor_type: ProcessorType::StyleTransfer,
            config: ProcessorConfig::for_model(model_id)
                .param("style_strength", strength.to_string()),
            enabled: true,
        });
        self
    }

    /// Add frame interpolation stage.
    pub fn frame_interpolation(mut self, model_id: impl Into<String>) -> Self {
        self.stages.push(PipelineStage {
            name: format!("interp_{}", self.stages.len()),
            processor_type: ProcessorType::FrameInterpolation,
            config: ProcessorConfig::for_model(model_id),
            enabled: true,
        });
        self
    }

    /// Add background removal stage.
    pub fn background_removal(mut self, model_id: impl Into<String>) -> Self {
        self.stages.push(PipelineStage {
            name: format!("bg_{}", self.stages.len()),
            processor_type: ProcessorType::BackgroundRemoval,
            config: ProcessorConfig::for_model(model_id),
            enabled: true,
        });
        self
    }

    /// Add face enhancement stage.
    pub fn face_enhancement(mut self, model_id: impl Into<String>, strength: f32) -> Self {
        self.stages.push(PipelineStage {
            name: format!("face_{}", self.stages.len()),
            processor_type: ProcessorType::FaceEnhancement,
            config: ProcessorConfig::for_model(model_id)
                .param("enhancement_strength", strength.to_string()),
            enabled: true,
        });
        self
    }

    /// Add denoising stage.
    pub fn denoising(mut self, model_id: impl Into<String>, strength: f32) -> Self {
        self.stages.push(PipelineStage {
            name: format!("denoise_{}", self.stages.len()),
            processor_type: ProcessorType::Denoising,
            config: ProcessorConfig::for_model(model_id)
                .param("denoise_strength", strength.to_string()),
            enabled: true,
        });
        self
    }

    /// Add color correction stage.
    pub fn color_correction(
        mut self,
        model_id: impl Into<String>,
        brightness: f32,
        contrast: f32,
        saturation: f32,
    ) -> Self {
        self.stages.push(PipelineStage {
            name: format!("color_{}", self.stages.len()),
            processor_type: ProcessorType::ColorCorrection,
            config: ProcessorConfig::for_model(model_id)
                .param("brightness", brightness.to_string())
                .param("contrast", contrast.to_string())
                .param("saturation", saturation.to_string()),
            enabled: true,
        });
        self
    }

    /// Add a custom stage.
    pub fn stage(mut self, stage: PipelineStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> Result<GenAiPipeline> {
        let mut pipeline = GenAiPipeline::new(self.registry);

        for stage in self.stages {
            pipeline.add_stage(stage)?;
        }

        Ok(pipeline)
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Total number of stages.
    pub total_stages: usize,
    /// Number of enabled stages.
    pub enabled_stages: usize,
    /// Estimated total memory usage (MB).
    pub estimated_memory_mb: u64,
}

/// Preset pipeline configurations.
pub struct PipelinePresets;

impl PipelinePresets {
    /// Create an upscaling pipeline.
    pub fn upscale_4x(registry: Arc<ModelRegistry>) -> Result<GenAiPipeline> {
        GenAiPipeline::builder(registry)
            .super_resolution("esrgan-x4")
            .build()
    }

    /// Create an upscaling pipeline with face enhancement.
    pub fn upscale_with_faces(registry: Arc<ModelRegistry>) -> Result<GenAiPipeline> {
        GenAiPipeline::builder(registry)
            .super_resolution("realesrgan-x2")
            .stage(PipelineStage {
                name: "face_enhance".to_string(),
                processor_type: ProcessorType::FaceEnhancement,
                config: ProcessorConfig::for_model("gfpgan-v1.4"),
                enabled: true,
            })
            .build()
    }

    /// Create a frame rate doubling pipeline.
    pub fn frame_rate_2x(registry: Arc<ModelRegistry>) -> Result<GenAiPipeline> {
        GenAiPipeline::builder(registry)
            .frame_interpolation("rife-v4")
            .build()
    }

    /// Create an artistic style pipeline.
    pub fn artistic_style(registry: Arc<ModelRegistry>, strength: f32) -> Result<GenAiPipeline> {
        GenAiPipeline::builder(registry)
            .style_transfer("fast-style-transfer", strength)
            .build()
    }

    /// Create a green screen pipeline.
    pub fn green_screen(registry: Arc<ModelRegistry>) -> Result<GenAiPipeline> {
        GenAiPipeline::builder(registry)
            .background_removal("rembg-u2net")
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_builder() {
        let registry = Arc::new(ModelRegistry::new());

        let pipeline = GenAiPipeline::builder(registry)
            .super_resolution("esrgan-x4")
            .build()
            .unwrap();

        assert_eq!(pipeline.stages().len(), 1);
    }

    #[test]
    fn test_pipeline_processing() {
        let registry = Arc::new(ModelRegistry::new());

        let pipeline = GenAiPipeline::builder(registry)
            .super_resolution("esrgan-x4")
            .build()
            .unwrap();

        let frame = FrameData::new(100, 100, 3);
        let output = pipeline.process_frame(frame).unwrap();

        assert_eq!(output.width, 400);
        assert_eq!(output.height, 400);
    }

    #[test]
    fn test_multi_stage_pipeline() {
        let registry = Arc::new(ModelRegistry::new());

        let pipeline = GenAiPipeline::builder(registry)
            .super_resolution("realesrgan-x2")
            .style_transfer("fast-style-transfer", 0.5)
            .build()
            .unwrap();

        assert_eq!(pipeline.stages().len(), 2);

        let stats = pipeline.stats();
        assert_eq!(stats.enabled_stages, 2);
    }

    #[test]
    fn test_disable_stage() {
        let registry = Arc::new(ModelRegistry::new());

        let mut pipeline = GenAiPipeline::builder(registry)
            .super_resolution("esrgan-x4")
            .build()
            .unwrap();

        pipeline.set_stage_enabled("sr_0", false);

        let stats = pipeline.stats();
        assert_eq!(stats.enabled_stages, 0);

        // Should pass through unchanged
        let frame = FrameData::new(100, 100, 3);
        let output = pipeline.process_frame(frame).unwrap();
        assert_eq!(output.width, 100);
    }

    #[test]
    fn test_presets() {
        let registry = Arc::new(ModelRegistry::new());

        let pipeline = PipelinePresets::upscale_4x(registry.clone()).unwrap();
        assert_eq!(pipeline.stages().len(), 1);

        let pipeline = PipelinePresets::frame_rate_2x(registry.clone()).unwrap();
        assert_eq!(pipeline.stages().len(), 1);

        let pipeline = PipelinePresets::artistic_style(registry.clone(), 0.8).unwrap();
        assert_eq!(pipeline.stages().len(), 1);
    }
}
