//! GPU-accelerated video processing using wgpu compute shaders.
//!
//! This crate provides hardware-accelerated video processing operations:
//!
//! - Color space conversion (YUV ↔ RGB)
//! - Scaling and resizing
//! - Frame filtering (blur, sharpen, denoise)
//! - Pixel format conversion
//!
//! # Example
//!
//! ```ignore
//! use transcode_gpu::{GpuContext, GpuProcessor, ScaleMode};
//!
//! async fn process() -> Result<(), transcode_gpu::GpuError> {
//!     // Initialize GPU context
//!     let context = GpuContext::new().await?;
//!
//!     // Create processor
//!     let processor = GpuProcessor::new(&context)?;
//!
//!     // Process frames
//!     let output = processor.scale(&input_frame, 1920, 1080, ScaleMode::Bilinear)?;
//!     Ok(())
//! }
//! ```

// Allow dead_code: This crate exposes a public API for GPU-accelerated video processing.
// Many internal types and functions are building blocks for external consumers and may
// not be used within the crate itself. The public API surface is intentionally larger
// than internal usage to support various user workflows.
#![allow(dead_code)]

mod context;
mod error;
mod pipeline;
mod processor;
mod shaders;
mod texture;

pub use context::GpuContext;
pub use error::{GpuError, Result};
pub use pipeline::{ComputePipeline, PipelineConfig};
pub use processor::{GpuProcessor, ProcessorConfig};
pub use shaders::{ShaderKind, ShaderRegistry};
pub use texture::{GpuTexture, TextureFormat};

/// Scaling algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ScaleMode {
    /// Nearest neighbor - fast, blocky
    Nearest,
    /// Bilinear interpolation - good balance
    #[default]
    Bilinear,
    /// Bicubic interpolation - higher quality
    Bicubic,
    /// Lanczos - best quality, slower
    Lanczos,
}

/// Color space for conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ColorSpace {
    /// BT.601 (SD video)
    Bt601,
    /// BT.709 (HD video)
    #[default]
    Bt709,
    /// BT.2020 (UHD/HDR video)
    Bt2020,
    /// sRGB
    Srgb,
}

/// Pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PixelFormat {
    /// RGBA 8-bit per channel
    #[default]
    Rgba8,
    /// BGRA 8-bit per channel
    Bgra8,
    /// YUV 4:2:0 planar (NV12)
    Nv12,
    /// YUV 4:2:0 planar (I420)
    I420,
    /// YUV 4:2:2 packed (YUYV)
    Yuyv,
    /// 10-bit per channel
    Rgba16,
    /// 32-bit float per channel
    Rgba32f,
}

impl PixelFormat {
    /// Get bytes per pixel for packed formats.
    pub fn bytes_per_pixel(&self) -> Option<usize> {
        match self {
            Self::Rgba8 | Self::Bgra8 => Some(4),
            Self::Yuyv => Some(2),
            Self::Rgba16 => Some(8),
            Self::Rgba32f => Some(16),
            Self::Nv12 | Self::I420 => None, // Planar formats
        }
    }

    /// Check if format is planar (separate Y, U, V planes).
    pub fn is_planar(&self) -> bool {
        matches!(self, Self::Nv12 | Self::I420)
    }
}

/// GPU capabilities and limits.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// GPU device name.
    pub device_name: String,
    /// Driver info.
    pub driver_info: String,
    /// Backend type (Vulkan, Metal, DX12, etc.)
    pub backend: String,
    /// Maximum texture dimension.
    pub max_texture_dimension: u32,
    /// Maximum buffer size.
    pub max_buffer_size: u64,
    /// Maximum compute workgroup size.
    pub max_workgroup_size: [u32; 3],
    /// Maximum compute workgroups per dimension.
    pub max_workgroups_per_dimension: u32,
    /// Whether 16-bit floats are supported.
    pub supports_f16: bool,
    /// Whether 64-bit atomics are supported.
    pub supports_64bit_atomics: bool,
}

impl GpuCapabilities {
    /// Create capabilities from wgpu adapter info and limits.
    pub fn from_adapter(info: &wgpu::AdapterInfo, limits: &wgpu::Limits) -> Self {
        Self {
            device_name: info.name.clone(),
            driver_info: info.driver_info.clone(),
            backend: format!("{:?}", info.backend),
            max_texture_dimension: limits.max_texture_dimension_2d,
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            supports_f16: false, // Check features if needed
            supports_64bit_atomics: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== ScaleMode tests =====

    #[test]
    fn test_scale_mode_default() {
        let mode: ScaleMode = Default::default();
        assert_eq!(mode, ScaleMode::Bilinear);
    }

    #[test]
    fn test_scale_mode_variants() {
        // Ensure all variants can be constructed
        let modes = [
            ScaleMode::Nearest,
            ScaleMode::Bilinear,
            ScaleMode::Bicubic,
            ScaleMode::Lanczos,
        ];
        assert_eq!(modes.len(), 4);
    }

    // ===== ColorSpace tests =====

    #[test]
    fn test_color_space_default() {
        let space: ColorSpace = Default::default();
        assert_eq!(space, ColorSpace::Bt709);
    }

    #[test]
    fn test_color_space_variants() {
        let spaces = [
            ColorSpace::Bt601,
            ColorSpace::Bt709,
            ColorSpace::Bt2020,
            ColorSpace::Srgb,
        ];
        assert_eq!(spaces.len(), 4);
    }

    // ===== PixelFormat tests =====

    #[test]
    fn test_pixel_format_default() {
        let format: PixelFormat = Default::default();
        assert_eq!(format, PixelFormat::Rgba8);
    }

    #[test]
    fn test_pixel_format_bytes_per_pixel_packed() {
        assert_eq!(PixelFormat::Rgba8.bytes_per_pixel(), Some(4));
        assert_eq!(PixelFormat::Bgra8.bytes_per_pixel(), Some(4));
        assert_eq!(PixelFormat::Yuyv.bytes_per_pixel(), Some(2));
        assert_eq!(PixelFormat::Rgba16.bytes_per_pixel(), Some(8));
        assert_eq!(PixelFormat::Rgba32f.bytes_per_pixel(), Some(16));
    }

    #[test]
    fn test_pixel_format_bytes_per_pixel_planar() {
        // Planar formats return None
        assert_eq!(PixelFormat::Nv12.bytes_per_pixel(), None);
        assert_eq!(PixelFormat::I420.bytes_per_pixel(), None);
    }

    #[test]
    fn test_pixel_format_is_planar() {
        assert!(!PixelFormat::Rgba8.is_planar());
        assert!(!PixelFormat::Bgra8.is_planar());
        assert!(!PixelFormat::Yuyv.is_planar());
        assert!(PixelFormat::Nv12.is_planar());
        assert!(PixelFormat::I420.is_planar());
    }

    // ===== GpuCapabilities tests =====

    #[test]
    fn test_gpu_capabilities_construction() {
        let caps = GpuCapabilities {
            device_name: "Test GPU".to_string(),
            driver_info: "Test Driver 1.0".to_string(),
            backend: "Vulkan".to_string(),
            max_texture_dimension: 16384,
            max_buffer_size: 256 * 1024 * 1024,
            max_workgroup_size: [256, 256, 64],
            max_workgroups_per_dimension: 65535,
            supports_f16: true,
            supports_64bit_atomics: false,
        };
        assert_eq!(caps.device_name, "Test GPU");
        assert_eq!(caps.max_texture_dimension, 16384);
        assert_eq!(caps.max_workgroup_size[0], 256);
    }
}
