//! Compute pipeline management.

use crate::error::Result;
use crate::shaders::ShaderKind;
use crate::GpuContext;
use bytemuck::{Pod, Zeroable};

/// Compute pipeline wrapper.
pub struct ComputePipeline {
    /// wgpu compute pipeline.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Pipeline configuration.
    config: PipelineConfig,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Shader kind.
    pub shader_kind: ShaderKind,
    /// Entry point function name.
    pub entry_point: String,
    /// Workgroup size.
    pub workgroup_size: [u32; 3],
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            shader_kind: ShaderKind::ScaleBilinear,
            entry_point: "main".to_string(),
            workgroup_size: [16, 16, 1],
        }
    }
}

impl ComputePipeline {
    /// Create a new compute pipeline.
    pub fn new(
        context: &GpuContext,
        shader_module: &wgpu::ShaderModule,
        config: PipelineConfig,
        layout_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<Self> {
        // Create bind group layout
        let bind_group_layout =
            context
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Compute Bind Group Layout"),
                    entries: layout_entries,
                });

        // Create pipeline layout
        let pipeline_layout =
            context
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Compute Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create compute pipeline
        let pipeline = context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{:?} Pipeline", config.shader_kind)),
                layout: Some(&pipeline_layout),
                module: shader_module,
                entry_point: Some(&config.entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Ok(Self {
            pipeline,
            bind_group_layout,
            config,
        })
    }

    /// Get the pipeline.
    pub fn pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Get bind group layout.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create a bind group from entries.
    pub fn create_bind_group(
        &self,
        context: &GpuContext,
        entries: &[wgpu::BindGroupEntry],
    ) -> wgpu::BindGroup {
        context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Bind Group"),
                layout: &self.bind_group_layout,
                entries,
            })
    }

    /// Calculate dispatch size for given dimensions.
    pub fn dispatch_size(&self, width: u32, height: u32) -> (u32, u32, u32) {
        let x = width.div_ceil(self.config.workgroup_size[0]);
        let y = height.div_ceil(self.config.workgroup_size[1]);
        let z = 1;
        (x, y, z)
    }
}

/// Parameters for scaling operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ScaleParams {
    /// Source width.
    pub src_width: u32,
    /// Source height.
    pub src_height: u32,
    /// Destination width.
    pub dst_width: u32,
    /// Destination height.
    pub dst_height: u32,
}

/// Parameters for YUV conversion.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct YuvParams {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Color space (0=BT.601, 1=BT.709, 2=BT.2020).
    pub color_space: u32,
    /// Padding for alignment.
    pub _padding: u32,
}

/// Parameters for blur operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BlurParams {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Blur radius.
    pub radius: u32,
    /// Gaussian sigma.
    pub sigma: f32,
}

/// Parameters for sharpen operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SharpenParams {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Sharpening strength.
    pub strength: f32,
    /// Padding.
    pub _padding: u32,
}

/// Parameters for color adjustment.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ColorAdjustParams {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Brightness adjustment (-1 to 1).
    pub brightness: f32,
    /// Contrast multiplier (0 to 2).
    pub contrast: f32,
    /// Saturation multiplier (0 to 2).
    pub saturation: f32,
    /// Gamma value.
    pub gamma: f32,
    /// Padding.
    pub _padding: [f32; 2],
}

impl Default for ColorAdjustParams {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            gamma: 1.0,
            _padding: [0.0; 2],
        }
    }
}

/// Simple dimension parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DimensionParams {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Padding.
    pub _padding: [u32; 2],
}

/// Create bind group layout entries for standard texture processing.
pub fn texture_processing_layout() -> Vec<wgpu::BindGroupLayoutEntry> {
    vec![
        // Input texture
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        // Output texture (storage)
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        // Uniform buffer for parameters
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
}

/// Create bind group layout entries for scaling with sampler.
pub fn scaling_layout() -> Vec<wgpu::BindGroupLayoutEntry> {
    vec![
        // Input texture
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        // Output texture (storage)
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
        // Uniform buffer for parameters
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        // Sampler for bilinear filtering
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ]
}
