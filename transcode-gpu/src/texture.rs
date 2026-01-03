//! GPU texture management.

use crate::error::{GpuError, Result};
use crate::{GpuContext, PixelFormat};

/// Texture format wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// RGBA 8-bit unsigned normalized.
    Rgba8Unorm,
    /// RGBA 8-bit sRGB.
    Rgba8UnormSrgb,
    /// BGRA 8-bit unsigned normalized.
    Bgra8Unorm,
    /// R8 single channel.
    R8Unorm,
    /// RG8 two channel.
    Rg8Unorm,
    /// RGBA 16-bit float.
    Rgba16Float,
    /// RGBA 32-bit float.
    Rgba32Float,
}

impl TextureFormat {
    /// Convert to wgpu format.
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            Self::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            Self::R8Unorm => wgpu::TextureFormat::R8Unorm,
            Self::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
            Self::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            Self::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        }
    }

    /// Bytes per pixel.
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::R8Unorm => 1,
            Self::Rg8Unorm => 2,
            Self::Rgba8Unorm | Self::Rgba8UnormSrgb | Self::Bgra8Unorm => 4,
            Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
        }
    }

    /// Create from pixel format.
    pub fn from_pixel_format(format: PixelFormat) -> Result<Self> {
        match format {
            PixelFormat::Rgba8 => Ok(Self::Rgba8Unorm),
            PixelFormat::Bgra8 => Ok(Self::Bgra8Unorm),
            PixelFormat::Rgba16 => Ok(Self::Rgba16Float),
            PixelFormat::Rgba32f => Ok(Self::Rgba32Float),
            _ => Err(GpuError::UnsupportedFormat(format!("{:?}", format))),
        }
    }
}

/// GPU texture for video frame data.
pub struct GpuTexture {
    /// The wgpu texture.
    texture: wgpu::Texture,
    /// Texture view for binding.
    view: wgpu::TextureView,
    /// Texture format.
    format: TextureFormat,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
}

impl GpuTexture {
    /// Create a new GPU texture.
    pub fn new(
        context: &GpuContext,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: wgpu::TextureUsages,
        label: Option<&str>,
    ) -> Result<Self> {
        context.validate_texture_size(width, height)?;

        let texture = context.device().create_texture(&wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.to_wgpu(),
            usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self {
            texture,
            view,
            format,
            width,
            height,
        })
    }

    /// Create a texture for reading (shader input).
    pub fn new_input(
        context: &GpuContext,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Self> {
        Self::new(
            context,
            width,
            height,
            format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            Some("Input Texture"),
        )
    }

    /// Create a texture for writing (compute output).
    pub fn new_output(
        context: &GpuContext,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Self> {
        Self::new(
            context,
            width,
            height,
            format,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            Some("Output Texture"),
        )
    }

    /// Create a texture for both reading and writing.
    pub fn new_storage(
        context: &GpuContext,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Self> {
        Self::new(
            context,
            width,
            height,
            format,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            Some("Storage Texture"),
        )
    }

    /// Upload data to texture.
    pub fn write_data(&self, context: &GpuContext, data: &[u8]) -> Result<()> {
        let expected_size = (self.width * self.height * self.format.bytes_per_pixel()) as usize;
        if data.len() != expected_size {
            return Err(GpuError::BufferSizeMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }

        context.queue().write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.width * self.format.bytes_per_pixel()),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    /// Get the underlying wgpu texture.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Get the texture view.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get texture format.
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// Get width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Calculate buffer size for reading back.
    pub fn buffer_size(&self) -> u64 {
        (self.width * self.height * self.format.bytes_per_pixel()) as u64
    }

    /// Calculate bytes per row (with alignment).
    pub fn bytes_per_row(&self) -> u32 {
        let unpadded = self.width * self.format.bytes_per_pixel();
        // wgpu requires alignment to 256 bytes for buffer copies
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        unpadded.div_ceil(align) * align
    }
}

/// Builder for creating multiple related textures.
pub struct TextureSet {
    /// Input texture.
    pub input: GpuTexture,
    /// Output texture.
    pub output: GpuTexture,
}

impl TextureSet {
    /// Create input/output texture pair of same dimensions.
    pub fn new_pair(
        context: &GpuContext,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<Self> {
        Ok(Self {
            input: GpuTexture::new_input(context, width, height, format)?,
            output: GpuTexture::new_output(context, width, height, format)?,
        })
    }

    /// Create pair with different output dimensions (for scaling).
    pub fn new_scaling_pair(
        context: &GpuContext,
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
        format: TextureFormat,
    ) -> Result<Self> {
        Ok(Self {
            input: GpuTexture::new_input(context, input_width, input_height, format)?,
            output: GpuTexture::new_output(context, output_width, output_height, format)?,
        })
    }
}
