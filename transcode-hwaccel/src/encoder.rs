//! Hardware encoder implementation.
//!
//! This module provides a unified hardware encoder interface that dispatches
//! to platform-specific implementations:
//!
//! - **VideoToolbox** (macOS) - Apple Silicon hardware encoding
//! - **VA-API** (Linux) - Intel/AMD hardware encoding via libva
//! - **NVENC** (Cross-platform) - NVIDIA GPU encoding via CUDA
//!
//! When the corresponding feature is enabled (e.g., `videotoolbox`, `nvenc`), real
//! hardware encoding is performed. If a requested backend is not available,
//! an error is returned. Use `HwAccelType::Software` explicitly if you want
//! mock/fallback encoding for testing.
//!
//! # Architecture
//!
//! The [`HwEncoder`] struct wraps platform-specific encoder implementations
//! behind a unified interface. Platform selection is automatic based on
//! compile-time features and runtime hardware detection.
//!
//! # Building Block Pattern
//!
//! Some types in this module are marked with `#[allow(dead_code)]` because they
//! form the foundation for hardware-accelerated encoding infrastructure. These
//! types are used when their corresponding feature flags are enabled, but may
//! appear unused in builds that don't include all platform features.

use crate::error::{HwAccelError, Result};
use crate::types::*;
use crate::{HwAccelType, HwCodec};

#[cfg(target_os = "macos")]
use crate::videotoolbox::{VTEncoder, VTEncoderConfig};

#[cfg(target_os = "linux")]
use crate::vaapi::{VaEncoderConfig, VaapiEncoder};

#[cfg(feature = "nvenc")]
use crate::nvenc::{NvencEncoder, NvencEncoderConfig};

/// Hardware encoder configuration.
#[derive(Debug, Clone)]
pub struct HwEncoderConfig {
    /// Target codec.
    pub codec: HwCodec,
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Frame rate (numerator, denominator).
    pub frame_rate: (u32, u32),
    /// Rate control mode.
    pub rate_control: HwRateControl,
    /// Encoding preset.
    pub preset: HwPreset,
    /// Profile.
    pub profile: Option<HwProfile>,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
    /// Number of B-frames.
    pub bframes: u32,
    /// Reference frames.
    pub ref_frames: u32,
    /// Input format.
    pub input_format: HwSurfaceFormat,
    /// Enable lookahead (for better quality).
    pub lookahead: u32,
    /// Enable temporal AQ.
    pub temporal_aq: bool,
    /// Enable spatial AQ.
    pub spatial_aq: bool,
}

impl Default for HwEncoderConfig {
    fn default() -> Self {
        Self {
            codec: HwCodec::H264,
            width: 1920,
            height: 1080,
            frame_rate: (30, 1),
            rate_control: HwRateControl::Vbr {
                target: 5_000_000,
                max: 10_000_000,
            },
            preset: HwPreset::Medium,
            profile: None,
            gop_size: 60,
            bframes: 2,
            ref_frames: 3,
            input_format: HwSurfaceFormat::Nv12,
            lookahead: 0,
            temporal_aq: false,
            spatial_aq: false,
        }
    }
}

impl HwEncoderConfig {
    /// Create config for H.264 encoding.
    pub fn h264(width: u32, height: u32, bitrate: u32) -> Self {
        Self {
            codec: HwCodec::H264,
            width,
            height,
            rate_control: HwRateControl::Vbr {
                target: bitrate,
                max: bitrate * 2,
            },
            profile: Some(HwProfile::H264High),
            ..Default::default()
        }
    }

    /// Create config for HEVC encoding.
    pub fn hevc(width: u32, height: u32, bitrate: u32) -> Self {
        Self {
            codec: HwCodec::Hevc,
            width,
            height,
            rate_control: HwRateControl::Vbr {
                target: bitrate,
                max: bitrate * 2,
            },
            profile: Some(HwProfile::HevcMain),
            ..Default::default()
        }
    }

    /// Create config for AV1 encoding.
    pub fn av1(width: u32, height: u32, bitrate: u32) -> Self {
        Self {
            codec: HwCodec::Av1,
            width,
            height,
            rate_control: HwRateControl::Vbr {
                target: bitrate,
                max: bitrate * 2,
            },
            profile: Some(HwProfile::Av1Main),
            bframes: 0, // AV1 uses different terminology
            ..Default::default()
        }
    }

    /// Set preset.
    pub fn with_preset(mut self, preset: HwPreset) -> Self {
        self.preset = preset;
        self
    }

    /// Set GOP size.
    pub fn with_gop_size(mut self, gop_size: u32) -> Self {
        self.gop_size = gop_size;
        self
    }

    /// Enable lookahead.
    pub fn with_lookahead(mut self, frames: u32) -> Self {
        self.lookahead = frames;
        self
    }
}

/// Encoded packet from hardware encoder.
#[derive(Debug)]
pub struct HwPacket {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Is this a keyframe?
    pub is_keyframe: bool,
    /// Frame type.
    pub frame_type: FrameType,
}

/// Frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// I-frame (keyframe).
    I,
    /// P-frame (predicted).
    P,
    /// B-frame (bidirectional).
    B,
    /// Unknown.
    Unknown,
}

/// Platform-specific encoder state.
#[allow(dead_code)]
#[derive(Default)]
enum PlatformEncoder {
    /// No platform encoder (uses mock/software fallback).
    #[default]
    None,
    /// VideoToolbox encoder (macOS).
    #[cfg(target_os = "macos")]
    VideoToolbox(Box<VTEncoder>),
    /// VA-API encoder (Linux).
    ///
    /// Note: This encoder uses a stubbed VA-API workflow. While the API structure
    /// is complete, it returns simulated encoded data. For production use, ensure
    /// libva is available on the system and the `vaapi` feature is enabled.
    #[cfg(target_os = "linux")]
    Vaapi(Box<VaapiEncoder>),
    /// NVENC encoder (NVIDIA GPU).
    ///
    /// Note: This encoder uses a stubbed NVENC workflow. While the API structure
    /// is complete (including lookahead, B-frames, AQ), it returns simulated
    /// encoded data. For production use, ensure CUDA and NVENC SDK are available
    /// and the `nvenc` feature is enabled.
    #[cfg(feature = "nvenc")]
    Nvenc(Box<NvencEncoder>),
    /// NVENC encoder placeholder when feature is disabled.
    #[cfg(not(feature = "nvenc"))]
    Nvenc,
}

/// Hardware encoder.
///
/// This struct provides a unified interface for hardware-accelerated video encoding.
/// It automatically dispatches to the appropriate platform-specific encoder based on
/// the `HwAccelType` specified during creation.
///
/// # Example
///
/// ```ignore
/// use transcode_hwaccel::{HwEncoder, HwEncoderConfig, HwAccelType};
///
/// let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
/// let mut encoder = HwEncoder::new(HwAccelType::VideoToolbox, config)?;
/// encoder.init()?;
///
/// // Encode frames
/// let packet = encoder.encode(&frame)?;
/// ```
pub struct HwEncoder {
    config: HwEncoderConfig,
    accel_type: HwAccelType,
    _device: HwDeviceContext,
    frame_count: u64,
    initialized: bool,
    /// Platform-specific encoder instance.
    platform_encoder: PlatformEncoder,
    /// Cached parameter sets (SPS, PPS, VPS).
    parameter_sets: Option<ParameterSets>,
}

/// Cached codec parameter sets (SPS, PPS, VPS for H.264/HEVC, sequence header for AV1).
#[derive(Debug, Clone, Default)]
pub struct ParameterSets {
    /// Sequence Parameter Set (H.264/HEVC).
    pub sps: Option<Vec<u8>>,
    /// Picture Parameter Set (H.264/HEVC).
    pub pps: Option<Vec<u8>>,
    /// Video Parameter Set (HEVC only).
    pub vps: Option<Vec<u8>>,
    /// Sequence Header (AV1).
    pub sequence_header: Option<Vec<u8>>,
}

impl HwEncoder {
    /// Create a new hardware encoder.
    pub fn new(accel_type: HwAccelType, config: HwEncoderConfig) -> Result<Self> {
        // Validate configuration
        if config.width == 0 || config.height == 0 {
            return Err(HwAccelError::Config("Invalid dimensions".to_string()));
        }

        // Validate dimensions are even (required by most codecs)
        if !config.width.is_multiple_of(2) || !config.height.is_multiple_of(2) {
            return Err(HwAccelError::Config(
                "Dimensions must be even".to_string(),
            ));
        }

        let device = HwDeviceContext::new(accel_type, 0);

        Ok(Self {
            config,
            accel_type,
            _device: device,
            frame_count: 0,
            initialized: false,
            platform_encoder: PlatformEncoder::None,
            parameter_sets: None,
        })
    }

    /// Initialize the encoder.
    pub fn init(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Platform-specific initialization
        match self.accel_type {
            #[cfg(target_os = "macos")]
            HwAccelType::VideoToolbox => {
                self.init_videotoolbox()?;
            }
            #[cfg(target_os = "linux")]
            HwAccelType::Vaapi => {
                self.init_vaapi()?;
            }
            HwAccelType::Nvenc => {
                self.init_nvenc()?;
            }
            HwAccelType::Software => {
                // Software fallback - no special initialization needed
                tracing::info!("Using software encoding fallback");
            }
            _ => {
                return Err(HwAccelError::NotSupported(
                    self.accel_type.name().to_string(),
                ));
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Encode a frame.
    pub fn encode(&mut self, frame: &HwFrame) -> Result<Option<HwPacket>> {
        if !self.initialized {
            self.init()?;
        }

        // Validate frame dimensions
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(HwAccelError::Config(format!(
                "Frame dimensions {}x{} don't match encoder {}x{}",
                frame.width, frame.height, self.config.width, self.config.height
            )));
        }

        // Encode based on accelerator type
        let packet = match self.accel_type {
            #[cfg(target_os = "macos")]
            HwAccelType::VideoToolbox => self.encode_videotoolbox(frame)?,
            #[cfg(target_os = "linux")]
            HwAccelType::Vaapi => self.encode_vaapi(frame)?,
            HwAccelType::Nvenc => self.encode_nvenc(frame)?,
            _ => self.encode_software(frame)?,
        };

        self.frame_count += 1;
        Ok(packet)
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> Result<Vec<HwPacket>> {
        let mut packets = Vec::new();

        // Flush based on accelerator type
        match self.accel_type {
            #[cfg(target_os = "macos")]
            HwAccelType::VideoToolbox => {
                packets.extend(self.flush_videotoolbox()?);
            }
            #[cfg(target_os = "linux")]
            HwAccelType::Vaapi => {
                packets.extend(self.flush_vaapi()?);
            }
            HwAccelType::Nvenc => {
                packets.extend(self.flush_nvenc()?);
            }
            _ => {}
        }

        Ok(packets)
    }

    /// Get encoder statistics.
    pub fn stats(&self) -> HwEncoderStats {
        HwEncoderStats {
            frames_encoded: self.frame_count,
            accel_type: self.accel_type,
            codec: self.config.codec,
        }
    }

    // Platform-specific implementations

    #[cfg(target_os = "macos")]
    fn init_videotoolbox(&mut self) -> Result<()> {
        tracing::info!(
            "Initializing VideoToolbox encoder for {:?} {}x{}",
            self.config.codec,
            self.config.width,
            self.config.height
        );

        // Create VTEncoder configuration from HwEncoderConfig
        let vt_config = VTEncoderConfig {
            base: self.config.clone(),
            ..Default::default()
        };

        // Create the VideoToolbox encoder
        let encoder = VTEncoder::new(vt_config)?;
        self.platform_encoder = PlatformEncoder::VideoToolbox(Box::new(encoder));

        tracing::info!("VideoToolbox encoder initialized successfully");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn init_vaapi(&mut self) -> Result<()> {
        tracing::info!(
            "Initializing VA-API encoder for {:?} {}x{}",
            self.config.codec,
            self.config.width,
            self.config.height
        );

        // Create VA-API encoder configuration from HwEncoderConfig
        let va_config = VaEncoderConfig {
            base: self.config.clone(),
            ..Default::default()
        };

        // Create the VA-API encoder
        let mut encoder = VaapiEncoder::new(va_config)?;

        // Initialize and generate parameter sets
        encoder.initialize()?;

        // Cache parameter sets
        self.parameter_sets = Some(ParameterSets {
            sps: encoder.sps().map(|s| s.to_vec()),
            pps: encoder.pps().map(|p| p.to_vec()),
            vps: encoder.vps().map(|v| v.to_vec()),
            sequence_header: None,
        });

        self.platform_encoder = PlatformEncoder::Vaapi(Box::new(encoder));

        Ok(())
    }

    #[cfg(feature = "nvenc")]
    fn init_nvenc(&mut self) -> Result<()> {
        tracing::info!(
            "Initializing NVENC encoder for {:?} {}x{}",
            self.config.codec,
            self.config.width,
            self.config.height
        );

        // Create NVENC encoder configuration from HwEncoderConfig
        let nvenc_config = NvencEncoderConfig {
            base: self.config.clone(),
            ..Default::default()
        };

        // Create the NVENC encoder
        let encoder = NvencEncoder::new(nvenc_config)?;

        // Cache parameter sets
        self.parameter_sets = Some(ParameterSets {
            sps: encoder.get_sps().map(|s: &[u8]| s.to_vec()),
            pps: encoder.get_pps().map(|p: &[u8]| p.to_vec()),
            vps: encoder.get_vps().map(|v: &[u8]| v.to_vec()),
            sequence_header: None,
        });

        self.platform_encoder = PlatformEncoder::Nvenc(Box::new(encoder));

        Ok(())
    }

    #[cfg(not(feature = "nvenc"))]
    fn init_nvenc(&mut self) -> Result<()> {
        Err(HwAccelError::NotSupported(
            "NVENC encoder requires 'nvenc' feature to be enabled".to_string(),
        ))
    }

    #[cfg(target_os = "macos")]
    fn encode_videotoolbox(&mut self, frame: &HwFrame) -> Result<Option<HwPacket>> {
        // Use the VideoToolbox encoder if available
        if let PlatformEncoder::VideoToolbox(ref mut encoder) = self.platform_encoder {
            // Encode through VTEncoder
            match encoder.encode(frame) {
                Ok(Some(packet)) => Ok(Some(packet)),
                Ok(None) => Ok(None),
                Err(e) => Err(e),
            }
        } else {
            // Fallback to mock encoding if VTEncoder wasn't initialized
            self.encode_mock(frame)
        }
    }

    /// Mock encoding fallback when platform encoder is not available.
    fn encode_mock(&mut self, frame: &HwFrame) -> Result<Option<HwPacket>> {
        let is_keyframe = self.frame_count.is_multiple_of(self.config.gop_size as u64);

        // Generate realistic mock data based on codec and resolution
        let estimated_size = self.estimate_frame_size(is_keyframe);
        let mut data = vec![0u8; estimated_size];

        // Add NAL unit start codes for H.264/HEVC
        match self.config.codec {
            HwCodec::H264 | HwCodec::Hevc => {
                // Annex B start code
                data[0] = 0x00;
                data[1] = 0x00;
                data[2] = 0x00;
                data[3] = 0x01;
                // NAL unit type (simplified)
                data[4] = if is_keyframe { 0x65 } else { 0x41 };
            }
            HwCodec::Av1 => {
                // AV1 OBU header (simplified)
                data[0] = 0x12; // OBU type + extension flag
            }
            _ => {}
        }

        Ok(Some(HwPacket {
            data,
            pts: frame.pts,
            dts: frame.pts,
            is_keyframe,
            frame_type: if is_keyframe { FrameType::I } else { FrameType::P },
        }))
    }

    /// Estimate frame size based on codec, resolution, and frame type.
    fn estimate_frame_size(&self, is_keyframe: bool) -> usize {
        let pixels = self.config.width as usize * self.config.height as usize;
        let bits_per_pixel = match &self.config.rate_control {
            HwRateControl::Vbr { target, .. } => {
                let fps = self.config.frame_rate.0 as f64 / self.config.frame_rate.1 as f64;
                (*target as f64 / fps / pixels as f64).max(0.1)
            }
            HwRateControl::Cbr(bitrate) => {
                let fps = self.config.frame_rate.0 as f64 / self.config.frame_rate.1 as f64;
                (*bitrate as f64 / fps / pixels as f64).max(0.1)
            }
            _ => 0.5, // Default for CQP/CQ
        };

        let base_size = (pixels as f64 * bits_per_pixel / 8.0) as usize;

        // Keyframes are typically 2-5x larger
        if is_keyframe {
            base_size.saturating_mul(3).max(1000)
        } else {
            base_size.max(100)
        }
    }

    #[cfg(target_os = "linux")]
    fn encode_vaapi(&mut self, frame: &HwFrame) -> Result<Option<HwPacket>> {
        // Use the VA-API encoder if available
        if let PlatformEncoder::Vaapi(ref mut encoder) = self.platform_encoder {
            // Encode through VaapiEncoder
            match encoder.encode(&frame.data, frame.pts) {
                Ok(Some(va_frame)) => {
                    Ok(Some(HwPacket {
                        data: va_frame.data,
                        pts: va_frame.pts,
                        dts: va_frame.dts,
                        is_keyframe: va_frame.is_keyframe,
                        frame_type: match va_frame.frame_type {
                            crate::vaapi::VaFrameType::I => FrameType::I,
                            crate::vaapi::VaFrameType::P => FrameType::P,
                            crate::vaapi::VaFrameType::B => FrameType::B,
                        },
                    }))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(e),
            }
        } else {
            // VA-API encoder not initialized - return error instead of silent mock
            Err(HwAccelError::NotSupported(
                "VA-API encoder not initialized".to_string(),
            ))
        }
    }

    #[cfg(feature = "nvenc")]
    fn encode_nvenc(&mut self, frame: &HwFrame) -> Result<Option<HwPacket>> {
        // Use the NVENC encoder if available
        if let PlatformEncoder::Nvenc(ref mut encoder) = self.platform_encoder {
            // Encode through NvencEncoder
            encoder.encode(frame)
        } else {
            // NVENC encoder not initialized - return error instead of silent mock
            Err(HwAccelError::NotSupported(
                "NVENC encoder not initialized".to_string(),
            ))
        }
    }

    #[cfg(not(feature = "nvenc"))]
    fn encode_nvenc(&mut self, _frame: &HwFrame) -> Result<Option<HwPacket>> {
        Err(HwAccelError::NotSupported(
            "NVENC encoder requires 'nvenc' feature to be enabled".to_string(),
        ))
    }

    fn encode_software(&mut self, frame: &HwFrame) -> Result<Option<HwPacket>> {
        // Software fallback always uses mock encoding
        self.encode_mock(frame)
    }

    #[cfg(target_os = "macos")]
    fn flush_videotoolbox(&mut self) -> Result<Vec<HwPacket>> {
        if let PlatformEncoder::VideoToolbox(ref mut encoder) = self.platform_encoder {
            // Flush the VideoToolbox encoder
            let vt_packets = encoder.flush()?;

            // Convert VTEncodedFrames to HwPackets
            let packets: Vec<HwPacket> = vt_packets
                .into_iter()
                .map(|vt_frame| HwPacket {
                    data: vt_frame.data,
                    pts: vt_frame.pts,
                    dts: vt_frame.dts,
                    is_keyframe: vt_frame.is_keyframe,
                    frame_type: vt_frame.frame_type,
                })
                .collect();

            Ok(packets)
        } else {
            Ok(Vec::new())
        }
    }

    #[cfg(target_os = "linux")]
    fn flush_vaapi(&mut self) -> Result<Vec<HwPacket>> {
        if let PlatformEncoder::Vaapi(ref mut encoder) = self.platform_encoder {
            // Flush the VA-API encoder
            let va_frames = encoder.flush()?;

            // Convert VaEncodedFrames to HwPackets
            let packets: Vec<HwPacket> = va_frames
                .into_iter()
                .map(|va_frame| HwPacket {
                    data: va_frame.data,
                    pts: va_frame.pts,
                    dts: va_frame.dts,
                    is_keyframe: va_frame.is_keyframe,
                    frame_type: match va_frame.frame_type {
                        crate::vaapi::VaFrameType::I => FrameType::I,
                        crate::vaapi::VaFrameType::P => FrameType::P,
                        crate::vaapi::VaFrameType::B => FrameType::B,
                    },
                })
                .collect();

            Ok(packets)
        } else {
            Ok(Vec::new())
        }
    }

    #[cfg(feature = "nvenc")]
    fn flush_nvenc(&mut self) -> Result<Vec<HwPacket>> {
        if let PlatformEncoder::Nvenc(ref mut encoder) = self.platform_encoder {
            // Flush the NVENC encoder - it already returns Vec<HwPacket>
            encoder.flush()
        } else {
            Ok(Vec::new())
        }
    }

    #[cfg(not(feature = "nvenc"))]
    fn flush_nvenc(&mut self) -> Result<Vec<HwPacket>> {
        // NVENC feature not enabled, nothing to flush
        Ok(Vec::new())
    }

    /// Get the cached parameter sets (SPS, PPS, VPS).
    pub fn get_parameter_sets(&self) -> Option<&ParameterSets> {
        self.parameter_sets.as_ref()
    }

    /// Get the codec being used.
    pub fn codec(&self) -> HwCodec {
        self.config.codec
    }

    /// Get the configured resolution.
    pub fn resolution(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Get the configured frame rate.
    pub fn frame_rate(&self) -> (u32, u32) {
        self.config.frame_rate
    }

    /// Check if the encoder is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Encoder statistics.
#[derive(Debug, Clone)]
pub struct HwEncoderStats {
    /// Number of frames encoded.
    pub frames_encoded: u64,
    /// Accelerator type used.
    pub accel_type: HwAccelType,
    /// Codec used.
    pub codec: HwCodec,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.codec, HwCodec::H264);
    }

    #[test]
    fn test_encoder_creation() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let encoder = HwEncoder::new(HwAccelType::Software, config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_invalid_dimensions() {
        // Zero dimensions should fail
        let mut config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        config.width = 0;
        let result = HwEncoder::new(HwAccelType::Software, config);
        assert!(result.is_err());

        // Odd dimensions should fail
        let mut config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        config.width = 1921; // Odd
        let result = HwEncoder::new(HwAccelType::Software, config);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_init_and_encode() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let mut encoder = HwEncoder::new(HwAccelType::Software, config).unwrap();

        // Initialize
        encoder.init().unwrap();
        assert!(encoder.is_initialized());

        // Create a test frame
        let frame_data = vec![0u8; 1920 * 1080 * 3 / 2]; // NV12
        let frame = HwFrame::new_cpu(frame_data, 1920, 1080, HwSurfaceFormat::Nv12);

        // Encode
        let packet = encoder.encode(&frame).unwrap();
        assert!(packet.is_some());

        let packet = packet.unwrap();
        assert!(packet.is_keyframe); // First frame should be keyframe
        assert!(!packet.data.is_empty());
    }

    #[test]
    fn test_encoder_keyframe_interval() {
        let mut config = HwEncoderConfig::h264(640, 480, 1_000_000);
        config.gop_size = 5; // Keyframe every 5 frames

        let mut encoder = HwEncoder::new(HwAccelType::Software, config).unwrap();
        encoder.init().unwrap();

        let frame_data = vec![0u8; 640 * 480 * 3 / 2];

        // Encode several frames and check keyframe pattern
        for i in 0..15 {
            let frame = HwFrame {
                format: HwSurfaceFormat::Nv12,
                width: 640,
                height: 480,
                pts: i,
                handle: HwFrameHandle::Cpu(frame_data.clone()),
            };

            let packet = encoder.encode(&frame).unwrap().unwrap();

            // Keyframes at 0, 5, 10, ...
            if i % 5 == 0 {
                assert!(packet.is_keyframe, "Frame {} should be keyframe", i);
            } else {
                assert!(!packet.is_keyframe, "Frame {} should not be keyframe", i);
            }
        }
    }

    #[test]
    fn test_encoder_stats() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let mut encoder = HwEncoder::new(HwAccelType::Software, config).unwrap();
        encoder.init().unwrap();

        let stats = encoder.stats();
        assert_eq!(stats.frames_encoded, 0);
        assert_eq!(stats.codec, HwCodec::H264);
        assert_eq!(stats.accel_type, HwAccelType::Software);

        // Encode a frame
        let frame = HwFrame::new_cpu(vec![0u8; 1920 * 1080 * 3 / 2], 1920, 1080, HwSurfaceFormat::Nv12);
        encoder.encode(&frame).unwrap();

        let stats = encoder.stats();
        assert_eq!(stats.frames_encoded, 1);
    }

    #[test]
    fn test_encoder_flush() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let mut encoder = HwEncoder::new(HwAccelType::Software, config).unwrap();
        encoder.init().unwrap();

        // Flush should succeed even with no frames
        let flushed = encoder.flush().unwrap();
        assert!(flushed.is_empty());
    }

    #[test]
    fn test_encoder_hevc_config() {
        let config = HwEncoderConfig::hevc(3840, 2160, 10_000_000);
        assert_eq!(config.codec, HwCodec::Hevc);
        assert_eq!(config.width, 3840);
        assert_eq!(config.height, 2160);

        let encoder = HwEncoder::new(HwAccelType::Software, config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_av1_config() {
        let config = HwEncoderConfig::av1(1920, 1080, 4_000_000);
        assert_eq!(config.codec, HwCodec::Av1);

        let encoder = HwEncoder::new(HwAccelType::Software, config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_mock_output_size() {
        // Test that mock output varies by resolution and bitrate
        let config_high = HwEncoderConfig::h264(1920, 1080, 10_000_000);
        let config_low = HwEncoderConfig::h264(640, 480, 1_000_000);

        let mut encoder_high = HwEncoder::new(HwAccelType::Software, config_high).unwrap();
        let mut encoder_low = HwEncoder::new(HwAccelType::Software, config_low).unwrap();

        encoder_high.init().unwrap();
        encoder_low.init().unwrap();

        let frame_high = HwFrame::new_cpu(vec![0u8; 1920 * 1080 * 3 / 2], 1920, 1080, HwSurfaceFormat::Nv12);
        let frame_low = HwFrame::new_cpu(vec![0u8; 640 * 480 * 3 / 2], 640, 480, HwSurfaceFormat::Nv12);

        let packet_high = encoder_high.encode(&frame_high).unwrap().unwrap();
        let packet_low = encoder_low.encode(&frame_low).unwrap().unwrap();

        // Higher resolution/bitrate should produce larger packets
        assert!(packet_high.data.len() > packet_low.data.len());
    }

    #[test]
    fn test_parameter_sets() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let encoder = HwEncoder::new(HwAccelType::Software, config).unwrap();

        // Parameter sets not populated in mock mode
        assert!(encoder.get_parameter_sets().is_none());
    }

    #[test]
    fn test_encoder_accessors() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        config.frame_rate;
        let encoder = HwEncoder::new(HwAccelType::Software, config).unwrap();

        assert_eq!(encoder.codec(), HwCodec::H264);
        assert_eq!(encoder.resolution(), (1920, 1080));
        assert_eq!(encoder.frame_rate(), (30, 1));
        assert!(!encoder.is_initialized());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_videotoolbox_encoder_init() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let mut encoder = HwEncoder::new(HwAccelType::VideoToolbox, config).unwrap();

        // Init should succeed on macOS
        let result = encoder.init();
        assert!(result.is_ok());
        assert!(encoder.is_initialized());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_videotoolbox_encoder_encode() {
        let config = HwEncoderConfig::h264(1920, 1080, 5_000_000);
        let mut encoder = HwEncoder::new(HwAccelType::VideoToolbox, config).unwrap();
        encoder.init().unwrap();

        let frame = HwFrame::new_cpu(vec![0u8; 1920 * 1080 * 3 / 2], 1920, 1080, HwSurfaceFormat::Nv12);
        let result = encoder.encode(&frame);
        assert!(result.is_ok());

        let packet = result.unwrap();
        assert!(packet.is_some());
    }
}
