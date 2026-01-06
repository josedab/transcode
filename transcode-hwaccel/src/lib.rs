//! Hardware-accelerated encoding and decoding for transcode.
//!
//! This crate provides a unified interface to hardware video acceleration APIs:
//!
//! - **VAAPI** (Video Acceleration API) - Linux/Intel
//! - **VideoToolbox** - macOS/Apple Silicon
//! - **NVENC/NVDEC** - NVIDIA GPUs
//! - **QSV** (Quick Sync Video) - Intel integrated graphics
//! - **AMF** (Advanced Media Framework) - AMD GPUs
//!
//! # Example
//!
//! ```ignore
//! use transcode_hwaccel::{HwAccel, HwEncoder, HwDecoder};
//!
//! // Detect available hardware accelerators
//! let accelerators = HwAccel::detect()?;
//! println!("Available: {:?}", accelerators);
//!
//! // Create a hardware encoder
//! let encoder = HwEncoder::new(HwAccelType::Nvenc, config)?;
//! let packet = encoder.encode(&frame)?;
//! ```

pub mod error;
pub mod types;
pub mod detect;
pub mod encoder;
pub mod decoder;
pub mod av1_hw;
pub mod av1_obu;

#[cfg(target_os = "linux")]
pub mod vaapi;

#[cfg(target_os = "macos")]
pub mod videotoolbox;

#[cfg(feature = "nvenc")]
pub mod nvenc;

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub mod qsv;

pub use error::{HwAccelError, Result};
pub use types::*;
pub use detect::{detect_accelerators, HwAccelInfo};
pub use encoder::{HwEncoder, HwEncoderConfig};
pub use decoder::{HwDecoder, HwDecoderConfig};
pub use av1_hw::{Av1Profile, Av1Level, Av1TileConfig, Av1HwEncoderConfig};
pub use av1_obu::{ObuType, Obu, ObuHeader, SequenceHeader, FrameHeader};

/// Hardware acceleration type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HwAccelType {
    /// VA-API (Linux).
    Vaapi,
    /// VideoToolbox (macOS).
    VideoToolbox,
    /// NVIDIA NVENC/NVDEC.
    Nvenc,
    /// Intel Quick Sync Video.
    Qsv,
    /// AMD Advanced Media Framework.
    Amf,
    /// Direct3D 11 Video Acceleration.
    D3d11va,
    /// VDPAU (older Linux API).
    Vdpau,
    /// Software fallback.
    Software,
}

impl HwAccelType {
    /// Check if this accelerator is available on the current platform.
    pub fn is_available(&self) -> bool {
        match self {
            #[cfg(target_os = "linux")]
            HwAccelType::Vaapi => true,
            #[cfg(target_os = "macos")]
            HwAccelType::VideoToolbox => true,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            HwAccelType::Qsv => qsv::is_available(),
            HwAccelType::Software => true,
            _ => false,
        }
    }

    /// Get the display name.
    pub fn name(&self) -> &'static str {
        match self {
            HwAccelType::Vaapi => "VA-API",
            HwAccelType::VideoToolbox => "VideoToolbox",
            HwAccelType::Nvenc => "NVENC",
            HwAccelType::Qsv => "Quick Sync",
            HwAccelType::Amf => "AMD AMF",
            HwAccelType::D3d11va => "D3D11VA",
            HwAccelType::Vdpau => "VDPAU",
            HwAccelType::Software => "Software",
        }
    }
}

/// Supported hardware-accelerated codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HwCodec {
    H264,
    Hevc,
    Av1,
    Vp9,
    Vp8,
    Mpeg2,
    Vc1,
    Jpeg,
}

impl HwCodec {
    /// Get codec name.
    pub fn name(&self) -> &'static str {
        match self {
            HwCodec::H264 => "H.264/AVC",
            HwCodec::Hevc => "H.265/HEVC",
            HwCodec::Av1 => "AV1",
            HwCodec::Vp9 => "VP9",
            HwCodec::Vp8 => "VP8",
            HwCodec::Mpeg2 => "MPEG-2",
            HwCodec::Vc1 => "VC-1",
            HwCodec::Jpeg => "JPEG",
        }
    }
}

/// Hardware acceleration capabilities.
#[derive(Debug, Clone)]
pub struct HwCapabilities {
    /// Accelerator type.
    pub accel_type: HwAccelType,
    /// Supported codecs for encoding.
    pub encode_codecs: Vec<HwCodec>,
    /// Supported codecs for decoding.
    pub decode_codecs: Vec<HwCodec>,
    /// Maximum supported width.
    pub max_width: u32,
    /// Maximum supported height.
    pub max_height: u32,
    /// Supports B-frames.
    pub supports_bframes: bool,
    /// Supports 10-bit encoding.
    pub supports_10bit: bool,
    /// Supports HDR.
    pub supports_hdr: bool,
    /// Device name/description.
    pub device_name: String,
}

impl Default for HwCapabilities {
    fn default() -> Self {
        Self {
            accel_type: HwAccelType::Software,
            encode_codecs: Vec::new(),
            decode_codecs: Vec::new(),
            max_width: 4096,
            max_height: 2160,
            supports_bframes: true,
            supports_10bit: false,
            supports_hdr: false,
            device_name: "Software".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_accel_type() {
        assert_eq!(HwAccelType::Nvenc.name(), "NVENC");
        assert_eq!(HwAccelType::VideoToolbox.name(), "VideoToolbox");
        assert!(HwAccelType::Software.is_available());
    }

    #[test]
    fn test_hw_codec() {
        assert_eq!(HwCodec::H264.name(), "H.264/AVC");
        assert_eq!(HwCodec::Hevc.name(), "H.265/HEVC");
    }
}
