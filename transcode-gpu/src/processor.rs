//! High-level GPU frame processor.

use crate::error::{GpuError, Result};
use crate::pipeline::{
    scaling_layout, texture_processing_layout, BlurParams, ColorAdjustParams, ComputePipeline,
    DimensionParams, PipelineConfig, ScaleParams, SharpenParams,
};
use crate::shaders::{ShaderKind, ShaderRegistry};
use crate::texture::{GpuTexture, TextureFormat};
use crate::{ColorSpace, GpuContext, ScaleMode};
use std::collections::HashMap;
use tracing::debug;
use wgpu::util::DeviceExt;

/// GPU processor configuration.
#[derive(Debug, Clone, Default)]
pub struct ProcessorConfig {
    /// Default scale mode.
    pub default_scale_mode: ScaleMode,
    /// Default color space.
    pub default_color_space: ColorSpace,
}

/// High-level GPU frame processor.
pub struct GpuProcessor {
    /// GPU context.
    context: GpuContext,
    /// Cached pipelines.
    pipelines: HashMap<ShaderKind, ComputePipeline>,
    /// Configuration.
    #[allow(dead_code)]
    config: ProcessorConfig,
    /// Bilinear sampler.
    bilinear_sampler: wgpu::Sampler,
}

impl GpuProcessor {
    /// Create a new GPU processor.
    pub async fn new(config: ProcessorConfig) -> Result<Self> {
        let context = GpuContext::new().await?;
        Self::with_context(context, config)
    }

    /// Create a processor with existing context.
    pub fn with_context(context: GpuContext, config: ProcessorConfig) -> Result<Self> {
        let shaders = ShaderRegistry::load_all(&context)?;

        // Create bilinear sampler
        let bilinear_sampler = context.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bilinear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Pre-compile all pipelines
        let mut pipelines = HashMap::new();

        // Compile scaling pipelines
        for kind in [ShaderKind::ScaleBilinear, ShaderKind::ScaleBicubic] {
            let shader = shaders.get(kind)?;
            let pipeline_config = PipelineConfig {
                shader_kind: kind,
                entry_point: "main".to_string(),
                workgroup_size: [16, 16, 1],
            };
            let layout = scaling_layout();
            let pipeline = ComputePipeline::new(&context, shader, pipeline_config, &layout)?;
            pipelines.insert(kind, pipeline);
        }

        // Compile texture processing pipelines
        for kind in [
            ShaderKind::GaussianBlur,
            ShaderKind::Sharpen,
            ShaderKind::ColorAdjust,
            ShaderKind::Grayscale,
        ] {
            let shader = shaders.get(kind)?;
            let pipeline_config = PipelineConfig {
                shader_kind: kind,
                entry_point: "main".to_string(),
                workgroup_size: [16, 16, 1],
            };
            let layout = texture_processing_layout();
            let pipeline = ComputePipeline::new(&context, shader, pipeline_config, &layout)?;
            pipelines.insert(kind, pipeline);
        }

        Ok(Self {
            context,
            pipelines,
            config,
            bilinear_sampler,
        })
    }

    /// Get a pre-compiled pipeline.
    fn get_pipeline(&self, kind: ShaderKind) -> Result<&ComputePipeline> {
        self.pipelines
            .get(&kind)
            .ok_or_else(|| GpuError::PipelineCreationFailed(format!("{:?} pipeline not found", kind)))
    }

    /// Create a uniform buffer with data.
    fn create_uniform_buffer<T: bytemuck::Pod>(&self, label: &str, data: &T) -> wgpu::Buffer {
        self.context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Scale a frame using bilinear filtering.
    pub fn scale_bilinear(
        &self,
        input: &GpuTexture,
        output_width: u32,
        output_height: u32,
    ) -> Result<GpuTexture> {
        debug!(
            "GPU scale: {}x{} -> {}x{}",
            input.width(),
            input.height(),
            output_width,
            output_height
        );

        // Create output texture
        let output = GpuTexture::new_output(
            &self.context,
            output_width,
            output_height,
            TextureFormat::Rgba8Unorm,
        )?;

        // Create parameters buffer
        let params = ScaleParams {
            src_width: input.width(),
            src_height: input.height(),
            dst_width: output_width,
            dst_height: output_height,
        };
        let params_buffer = self.create_uniform_buffer("Scale Params", &params);

        // Get pipeline and create bind group
        let pipeline = self.get_pipeline(ShaderKind::ScaleBilinear)?;
        let bind_group = pipeline.create_bind_group(
            &self.context,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.bilinear_sampler),
                },
            ],
        );

        // Dispatch compute shader
        let dispatch = pipeline.dispatch_size(output_width, output_height);
        let mut encoder = self.context.create_encoder();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scale Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline.pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
        }
        self.context.submit(vec![encoder.finish()]);

        Ok(output)
    }

    /// Scale a frame with specified mode.
    pub fn scale(
        &self,
        input: &GpuTexture,
        output_width: u32,
        output_height: u32,
        mode: ScaleMode,
    ) -> Result<GpuTexture> {
        match mode {
            ScaleMode::Nearest | ScaleMode::Bilinear => {
                self.scale_bilinear(input, output_width, output_height)
            }
            ScaleMode::Bicubic | ScaleMode::Lanczos => {
                self.scale_bicubic(input, output_width, output_height)
            }
        }
    }

    /// Scale using bicubic interpolation.
    pub fn scale_bicubic(
        &self,
        input: &GpuTexture,
        output_width: u32,
        output_height: u32,
    ) -> Result<GpuTexture> {
        let output = GpuTexture::new_output(
            &self.context,
            output_width,
            output_height,
            TextureFormat::Rgba8Unorm,
        )?;

        let params = ScaleParams {
            src_width: input.width(),
            src_height: input.height(),
            dst_width: output_width,
            dst_height: output_height,
        };
        let params_buffer = self.create_uniform_buffer("Scale Params", &params);

        let pipeline = self.get_pipeline(ShaderKind::ScaleBicubic)?;
        let bind_group = pipeline.create_bind_group(
            &self.context,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let dispatch = pipeline.dispatch_size(output_width, output_height);
        let mut encoder = self.context.create_encoder();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bicubic Scale Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline.pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
        }
        self.context.submit(vec![encoder.finish()]);

        Ok(output)
    }

    /// Apply Gaussian blur.
    pub fn blur(&self, input: &GpuTexture, radius: u32, sigma: f32) -> Result<GpuTexture> {
        debug!(
            "GPU blur: {}x{} radius={} sigma={}",
            input.width(),
            input.height(),
            radius,
            sigma
        );

        let output = GpuTexture::new_output(
            &self.context,
            input.width(),
            input.height(),
            TextureFormat::Rgba8Unorm,
        )?;

        let params = BlurParams {
            width: input.width(),
            height: input.height(),
            radius,
            sigma,
        };
        let params_buffer = self.create_uniform_buffer("Blur Params", &params);

        let pipeline = self.get_pipeline(ShaderKind::GaussianBlur)?;
        let bind_group = pipeline.create_bind_group(
            &self.context,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let dispatch = pipeline.dispatch_size(input.width(), input.height());
        let mut encoder = self.context.create_encoder();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Blur Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline.pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
        }
        self.context.submit(vec![encoder.finish()]);

        Ok(output)
    }

    /// Apply sharpening filter.
    pub fn sharpen(&self, input: &GpuTexture, strength: f32) -> Result<GpuTexture> {
        debug!(
            "GPU sharpen: {}x{} strength={}",
            input.width(),
            input.height(),
            strength
        );

        let output = GpuTexture::new_output(
            &self.context,
            input.width(),
            input.height(),
            TextureFormat::Rgba8Unorm,
        )?;

        let params = SharpenParams {
            width: input.width(),
            height: input.height(),
            strength,
            _padding: 0,
        };
        let params_buffer = self.create_uniform_buffer("Sharpen Params", &params);

        let pipeline = self.get_pipeline(ShaderKind::Sharpen)?;
        let bind_group = pipeline.create_bind_group(
            &self.context,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let dispatch = pipeline.dispatch_size(input.width(), input.height());
        let mut encoder = self.context.create_encoder();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sharpen Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline.pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
        }
        self.context.submit(vec![encoder.finish()]);

        Ok(output)
    }

    /// Adjust color properties.
    pub fn color_adjust(
        &self,
        input: &GpuTexture,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        gamma: f32,
    ) -> Result<GpuTexture> {
        let output = GpuTexture::new_output(
            &self.context,
            input.width(),
            input.height(),
            TextureFormat::Rgba8Unorm,
        )?;

        let params = ColorAdjustParams {
            width: input.width(),
            height: input.height(),
            brightness,
            contrast,
            saturation,
            gamma,
            _padding: [0.0; 2],
        };
        let params_buffer = self.create_uniform_buffer("Color Adjust Params", &params);

        let pipeline = self.get_pipeline(ShaderKind::ColorAdjust)?;
        let bind_group = pipeline.create_bind_group(
            &self.context,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let dispatch = pipeline.dispatch_size(input.width(), input.height());
        let mut encoder = self.context.create_encoder();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Color Adjust Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline.pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
        }
        self.context.submit(vec![encoder.finish()]);

        Ok(output)
    }

    /// Convert to grayscale.
    pub fn grayscale(&self, input: &GpuTexture) -> Result<GpuTexture> {
        let output = GpuTexture::new_output(
            &self.context,
            input.width(),
            input.height(),
            TextureFormat::Rgba8Unorm,
        )?;

        let params = DimensionParams {
            width: input.width(),
            height: input.height(),
            _padding: [0; 2],
        };
        let params_buffer = self.create_uniform_buffer("Grayscale Params", &params);

        let pipeline = self.get_pipeline(ShaderKind::Grayscale)?;
        let bind_group = pipeline.create_bind_group(
            &self.context,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        );

        let dispatch = pipeline.dispatch_size(input.width(), input.height());
        let mut encoder = self.context.create_encoder();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Grayscale Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline.pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
        }
        self.context.submit(vec![encoder.finish()]);

        Ok(output)
    }

    /// Upload RGBA data to GPU texture.
    pub fn upload_rgba(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<GpuTexture> {
        let texture = GpuTexture::new_input(&self.context, width, height, TextureFormat::Rgba8Unorm)?;
        texture.write_data(&self.context, data)?;
        Ok(texture)
    }

    /// Download texture data from GPU.
    pub fn download(&self, texture: &GpuTexture) -> Result<Vec<u8>> {
        let buffer_size = texture.buffer_size();
        let bytes_per_row = texture.bytes_per_row();

        // Create staging buffer
        let staging = self.context.create_staging_buffer(
            bytes_per_row as u64 * texture.height() as u64,
        );

        // Copy texture to buffer
        let mut encoder = self.context.create_encoder();
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: texture.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(texture.height()),
                },
            },
            wgpu::Extent3d {
                width: texture.width(),
                height: texture.height(),
                depth_or_array_layers: 1,
            },
        );
        self.context.submit(vec![encoder.finish()]);

        // Map buffer and read data
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.context.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| GpuError::MappingFailed("Channel closed".into()))?
            .map_err(|e| GpuError::MappingFailed(e.to_string()))?;

        let mapped = slice.get_mapped_range();

        // Handle potential row padding
        let actual_row_bytes = texture.width() * texture.format().bytes_per_pixel();
        let mut data = Vec::with_capacity(buffer_size as usize);

        for row in 0..texture.height() {
            let start = (row * bytes_per_row) as usize;
            let end = start + actual_row_bytes as usize;
            data.extend_from_slice(&mapped[start..end]);
        }

        drop(mapped);
        staging.unmap();

        Ok(data)
    }

    /// Get GPU capabilities.
    pub fn capabilities(&self) -> &crate::GpuCapabilities {
        self.context.capabilities()
    }

    /// Get the GPU context.
    pub fn context(&self) -> &GpuContext {
        &self.context
    }
}
