//! Common types for hardware acceleration.

use crate::HwAccelType;

/// Hardware surface format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwSurfaceFormat {
    /// NV12 (Y plane + interleaved UV).
    Nv12,
    /// P010 (10-bit NV12).
    P010,
    /// YUV420P (planar YUV 4:2:0).
    Yuv420p,
    /// YUV420P10 (10-bit planar).
    Yuv420p10,
    /// RGBA.
    Rgba,
    /// BGRA.
    Bgra,
}

impl HwSurfaceFormat {
    /// Get bits per pixel.
    pub fn bits_per_pixel(&self) -> u32 {
        match self {
            HwSurfaceFormat::Nv12 | HwSurfaceFormat::Yuv420p => 12,
            HwSurfaceFormat::P010 | HwSurfaceFormat::Yuv420p10 => 15,
            HwSurfaceFormat::Rgba | HwSurfaceFormat::Bgra => 32,
        }
    }

    /// Check if this is a 10-bit format.
    pub fn is_10bit(&self) -> bool {
        matches!(self, HwSurfaceFormat::P010 | HwSurfaceFormat::Yuv420p10)
    }
}

/// Hardware frame/surface.
#[derive(Debug)]
pub struct HwFrame {
    /// Surface format.
    pub format: HwSurfaceFormat,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Presentation timestamp.
    pub pts: i64,
    /// Hardware-specific handle.
    pub handle: HwFrameHandle,
}

/// Hardware-specific frame handle.
#[derive(Debug)]
pub enum HwFrameHandle {
    /// CPU memory (for software fallback).
    Cpu(Vec<u8>),
    /// VA-API surface ID.
    #[cfg(target_os = "linux")]
    VaSurface(u32),
    /// VideoToolbox CVPixelBuffer (opaque pointer).
    #[cfg(target_os = "macos")]
    CvPixelBuffer(usize),
    /// CUDA device pointer.
    CudaPtr(u64),
    /// D3D11 texture (opaque pointer).
    #[cfg(target_os = "windows")]
    D3d11Texture(usize),
    /// Null/invalid handle.
    Null,
}

impl HwFrame {
    /// Create a new CPU frame.
    pub fn new_cpu(data: Vec<u8>, width: u32, height: u32, format: HwSurfaceFormat) -> Self {
        Self {
            format,
            width,
            height,
            pts: 0,
            handle: HwFrameHandle::Cpu(data),
        }
    }

    /// Check if this is a CPU frame.
    pub fn is_cpu(&self) -> bool {
        matches!(self.handle, HwFrameHandle::Cpu(_))
    }

    /// Get CPU data if available.
    pub fn cpu_data(&self) -> Option<&[u8]> {
        match &self.handle {
            HwFrameHandle::Cpu(data) => Some(data),
            _ => None,
        }
    }
}

/// Rate control mode for hardware encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwRateControl {
    /// Constant QP.
    Cqp(u32),
    /// Constant bitrate.
    Cbr(u32),
    /// Variable bitrate.
    Vbr { target: u32, max: u32 },
    /// Constant quality (CRF-like).
    Cq(u32),
}

/// Encoder preset for hardware encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HwPreset {
    /// Fastest encoding, lowest quality.
    Fastest,
    /// Fast encoding.
    Fast,
    /// Balanced quality and speed.
    #[default]
    Medium,
    /// Slower encoding, higher quality.
    Slow,
    /// Slowest encoding, highest quality.
    Slowest,
}

/// Profile for hardware encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwProfile {
    /// H.264 Baseline.
    H264Baseline,
    /// H.264 Main.
    H264Main,
    /// H.264 High.
    H264High,
    /// HEVC Main.
    HevcMain,
    /// HEVC Main 10.
    HevcMain10,
    /// AV1 Main (8-bit and 10-bit 4:2:0).
    Av1Main,
    /// AV1 High (8-bit and 10-bit 4:4:4).
    Av1High,
    /// AV1 Professional (12-bit support).
    Av1Professional,
}

/// Hardware device context.
#[derive(Debug)]
pub struct HwDeviceContext {
    /// Accelerator type.
    pub accel_type: HwAccelType,
    /// Device index.
    pub device_index: u32,
    /// Device name.
    pub device_name: String,
    /// Whether the device is initialized.
    pub initialized: bool,
}

impl HwDeviceContext {
    /// Create a new device context.
    pub fn new(accel_type: HwAccelType, device_index: u32) -> Self {
        Self {
            accel_type,
            device_index,
            device_name: format!("{}:{}", accel_type.name(), device_index),
            initialized: false,
        }
    }
}
