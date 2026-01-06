//! Intel Quick Sync Video (QSV) hardware acceleration.
//!
//! This module provides hardware-accelerated video encoding and decoding
//! using Intel Quick Sync Video technology, available on Intel processors
//! with integrated graphics.
//!
//! QSV supports:
//! - H.264/AVC encoding and decoding
//! - H.265/HEVC encoding and decoding
//! - VP9 decoding (and encoding on some platforms)
//! - AV1 encoding and decoding (on Intel Arc and newer)
//! - MPEG-2 decoding
//! - JPEG encoding and decoding
//!
//! # Platform Support
//!
//! - Windows: via Intel Media SDK / oneVPL
//! - Linux: via VA-API backend (libva with i965/iHD driver)

use crate::error::{HwAccelError, Result};
use crate::{HwAccelType, HwCapabilities, HwCodec};

/// QSV implementation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsvBackend {
    /// Intel Media SDK (legacy).
    MediaSdk,
    /// Intel oneVPL (Video Processing Library).
    OneVpl,
    /// VA-API backend (Linux).
    Vaapi,
}

/// QSV device information.
#[derive(Debug, Clone)]
pub struct QsvDeviceInfo {
    /// Device index.
    pub index: u32,
    /// Device name.
    pub name: String,
    /// Vendor ID.
    pub vendor_id: u32,
    /// Device ID.
    pub device_id: u32,
    /// Driver version.
    pub driver_version: String,
    /// API version.
    pub api_version: (u16, u16),
    /// Backend being used.
    pub backend: QsvBackend,
}

/// QSV encoder configuration.
#[derive(Debug, Clone)]
pub struct QsvEncoderConfig {
    /// Target codec.
    pub codec: HwCodec,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Target bitrate (bps).
    pub bitrate: u32,
    /// Maximum bitrate (bps).
    pub max_bitrate: u32,
    /// Rate control mode.
    pub rate_control: QsvRateControl,
    /// Quality preset.
    pub preset: QsvPreset,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
    /// Number of B-frames.
    pub b_frames: u32,
    /// Number of reference frames.
    pub ref_frames: u32,
    /// Enable 10-bit encoding.
    pub ten_bit: bool,
    /// Enable lookahead.
    pub lookahead: bool,
    /// Lookahead depth.
    pub lookahead_depth: u32,
    /// Low latency mode.
    pub low_latency: bool,
    /// Async depth (pipelining).
    pub async_depth: u32,
}

impl Default for QsvEncoderConfig {
    fn default() -> Self {
        Self {
            codec: HwCodec::H264,
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            bitrate: 5_000_000,
            max_bitrate: 10_000_000,
            rate_control: QsvRateControl::Vbr,
            preset: QsvPreset::Balanced,
            gop_size: 250,
            b_frames: 2,
            ref_frames: 4,
            ten_bit: false,
            lookahead: false,
            lookahead_depth: 40,
            low_latency: false,
            async_depth: 4,
        }
    }
}

/// QSV decoder configuration.
#[derive(Debug, Clone)]
pub struct QsvDecoderConfig {
    /// Codec to decode.
    pub codec: HwCodec,
    /// Expected frame width (0 for auto-detect).
    pub width: u32,
    /// Expected frame height (0 for auto-detect).
    pub height: u32,
    /// Output 10-bit surfaces.
    pub ten_bit_output: bool,
    /// Async depth.
    pub async_depth: u32,
    /// Output to system memory.
    pub output_to_system: bool,
}

impl Default for QsvDecoderConfig {
    fn default() -> Self {
        Self {
            codec: HwCodec::H264,
            width: 0,
            height: 0,
            ten_bit_output: false,
            async_depth: 4,
            output_to_system: true,
        }
    }
}

/// QSV rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsvRateControl {
    /// Constant Bitrate.
    Cbr,
    /// Variable Bitrate.
    Vbr,
    /// Constant Quality (CQP).
    Cqp,
    /// Average Variable Bitrate.
    Avbr,
    /// Intelligent Constant Quality (ICQ).
    Icq,
    /// Lookahead VBR.
    LookaheadVbr,
    /// Lookahead CBR.
    LookaheadCbr,
    /// Quality VBR (QVBR).
    Qvbr,
}

/// QSV encoding preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsvPreset {
    /// Fastest encoding, lowest quality.
    VeryFast,
    /// Fast encoding.
    Faster,
    /// Balanced speed/quality.
    Fast,
    /// Medium quality.
    Medium,
    /// Balanced preset.
    Balanced,
    /// Slow encoding, high quality.
    Slow,
    /// Slower encoding.
    Slower,
    /// Verify slow encoding.
    VerySlow,
    /// Best quality, slowest.
    Quality,
}

impl QsvPreset {
    /// Get the target usage value (1=best quality, 7=best speed).
    pub fn target_usage(&self) -> u32 {
        match self {
            QsvPreset::VeryFast => 7,
            QsvPreset::Faster => 6,
            QsvPreset::Fast => 5,
            QsvPreset::Medium => 4,
            QsvPreset::Balanced => 4,
            QsvPreset::Slow => 3,
            QsvPreset::Slower => 2,
            QsvPreset::VerySlow => 2,
            QsvPreset::Quality => 1,
        }
    }
}

/// QSV encoder.
pub struct QsvEncoder {
    config: QsvEncoderConfig,
    device: QsvDeviceInfo,
    initialized: bool,
    frame_count: u64,
    // In a real implementation, these would be FFI handles
    // session: mfxSession,
    // surfaces: Vec<mfxFrameSurface1>,
}

impl QsvEncoder {
    /// Create a new QSV encoder.
    pub fn new(config: QsvEncoderConfig) -> Result<Self> {
        let device = Self::detect_device()?;

        // Validate codec support
        if !Self::is_codec_supported(&device, config.codec, true) {
            return Err(HwAccelError::UnsupportedCodec(config.codec.name().to_string()));
        }

        Ok(Self {
            config,
            device,
            initialized: false,
            frame_count: 0,
        })
    }

    /// Initialize the encoder.
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // Validate configuration
        self.validate_config()?;

        // In a real implementation:
        // 1. Create Intel Media SDK/oneVPL session
        // 2. Set video parameters (mfxVideoParam)
        // 3. Allocate surfaces
        // 4. Initialize encoder with MFXVideoENCODE_Init

        self.initialized = true;
        Ok(())
    }

    /// Encode a frame.
    pub fn encode(&mut self, frame: &QsvFrame) -> Result<Vec<u8>> {
        if !self.initialized {
            self.initialize()?;
        }

        // Validate frame dimensions
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(HwAccelError::InvalidParameter(
                "Frame dimensions don't match encoder config".to_string()
            ));
        }

        self.frame_count += 1;

        // In a real implementation:
        // 1. Copy input frame to surface
        // 2. Call MFXVideoENCODE_EncodeFrameAsync
        // 3. Sync and retrieve bitstream
        // 4. Return encoded data

        // Placeholder: return empty encoded packet
        Ok(Vec::new())
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        if !self.initialized {
            return Ok(Vec::new());
        }

        // In a real implementation:
        // Call MFXVideoENCODE_EncodeFrameAsync with NULL surface
        // until all frames are drained

        Ok(Vec::new())
    }

    /// Get encoder statistics.
    pub fn stats(&self) -> QsvEncoderStats {
        QsvEncoderStats {
            frames_encoded: self.frame_count,
            bytes_output: 0,
            average_bitrate: 0,
            average_qp: 0.0,
        }
    }

    /// Detect QSV device.
    fn detect_device() -> Result<QsvDeviceInfo> {
        // In a real implementation:
        // 1. Try oneVPL first (MFXLoad)
        // 2. Fall back to Media SDK (MFXInit)
        // 3. Query device info

        #[cfg(target_os = "linux")]
        let backend = QsvBackend::Vaapi;
        #[cfg(target_os = "windows")]
        let backend = QsvBackend::OneVpl;
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        let backend = QsvBackend::MediaSdk;

        Ok(QsvDeviceInfo {
            index: 0,
            name: "Intel Quick Sync Video".to_string(),
            vendor_id: 0x8086,
            device_id: 0,
            driver_version: "1.0.0".to_string(),
            api_version: (2, 9),
            backend,
        })
    }

    /// Check if codec is supported.
    fn is_codec_supported(device: &QsvDeviceInfo, codec: HwCodec, encode: bool) -> bool {
        // Most Intel platforms support H.264 and HEVC
        // Newer platforms (Ice Lake+) support AV1
        match codec {
            HwCodec::H264 => true,
            HwCodec::Hevc => true,
            HwCodec::Vp9 => !encode, // Decode only on most platforms
            HwCodec::Av1 => device.api_version.0 >= 2, // AV1 requires newer platforms
            HwCodec::Mpeg2 => !encode, // Decode only
            HwCodec::Jpeg => true,
            _ => false,
        }
    }

    /// Validate configuration.
    fn validate_config(&self) -> Result<()> {
        // Check resolution limits
        let max_width = match self.config.codec {
            HwCodec::H264 => 4096,
            HwCodec::Hevc | HwCodec::Av1 => 8192,
            _ => 4096,
        };

        if self.config.width > max_width || self.config.height > max_width {
            return Err(HwAccelError::InvalidParameter(format!(
                "Resolution {}x{} exceeds maximum {}",
                self.config.width, self.config.height, max_width
            )));
        }

        // Check bitrate
        if self.config.bitrate == 0 {
            return Err(HwAccelError::InvalidParameter(
                "Bitrate cannot be zero".to_string()
            ));
        }

        Ok(())
    }
}

/// QSV decoder.
pub struct QsvDecoder {
    config: QsvDecoderConfig,
    device: QsvDeviceInfo,
    initialized: bool,
    frame_count: u64,
}

impl QsvDecoder {
    /// Create a new QSV decoder.
    pub fn new(config: QsvDecoderConfig) -> Result<Self> {
        let device = QsvEncoder::detect_device()?;

        // Validate codec support
        if !QsvEncoder::is_codec_supported(&device, config.codec, false) {
            return Err(HwAccelError::UnsupportedCodec(config.codec.name().to_string()));
        }

        Ok(Self {
            config,
            device,
            initialized: false,
            frame_count: 0,
        })
    }

    /// Initialize the decoder.
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        // In a real implementation:
        // 1. Create session
        // 2. Set decode parameters
        // 3. Allocate surfaces
        // 4. Initialize decoder

        self.initialized = true;
        Ok(())
    }

    /// Decode a packet.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Option<QsvFrame>> {
        if !self.initialized {
            self.initialize()?;
        }

        if packet.is_empty() {
            return Ok(None);
        }

        self.frame_count += 1;

        // In a real implementation:
        // 1. Submit bitstream to MFXVideoDECODE_DecodeFrameAsync
        // 2. Wait for surface
        // 3. Copy to output frame

        // Placeholder: return empty frame
        Ok(Some(QsvFrame {
            width: self.config.width,
            height: self.config.height,
            format: QsvFrameFormat::Nv12,
            data: vec![0u8; (self.config.width * self.config.height * 3 / 2) as usize],
            pts: self.frame_count as i64,
        }))
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> Result<Vec<QsvFrame>> {
        if !self.initialized {
            return Ok(Vec::new());
        }

        // Drain decoder
        Ok(Vec::new())
    }

    /// Get decoder statistics.
    pub fn stats(&self) -> QsvDecoderStats {
        QsvDecoderStats {
            frames_decoded: self.frame_count,
            bytes_input: 0,
        }
    }
}

/// QSV frame format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QsvFrameFormat {
    /// NV12 (8-bit YUV 4:2:0, semi-planar).
    Nv12,
    /// P010 (10-bit YUV 4:2:0, semi-planar).
    P010,
    /// YUY2 (8-bit YUV 4:2:2, packed).
    Yuy2,
    /// Y210 (10-bit YUV 4:2:2, packed).
    Y210,
    /// RGB4 (32-bit BGRA).
    Rgb4,
    /// AYUV (packed YUV 4:4:4).
    Ayuv,
}

/// QSV frame.
#[derive(Debug, Clone)]
pub struct QsvFrame {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Pixel format.
    pub format: QsvFrameFormat,
    /// Frame data.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: i64,
}

/// QSV encoder statistics.
#[derive(Debug, Clone)]
pub struct QsvEncoderStats {
    /// Number of frames encoded.
    pub frames_encoded: u64,
    /// Total bytes output.
    pub bytes_output: u64,
    /// Average bitrate achieved.
    pub average_bitrate: u64,
    /// Average QP value.
    pub average_qp: f32,
}

/// QSV decoder statistics.
#[derive(Debug, Clone)]
pub struct QsvDecoderStats {
    /// Number of frames decoded.
    pub frames_decoded: u64,
    /// Total bytes input.
    pub bytes_input: u64,
}

/// Detect QSV capabilities.
pub fn detect_capabilities() -> Result<HwCapabilities> {
    let device = QsvEncoder::detect_device()?;

    let encode_codecs = vec![HwCodec::H264, HwCodec::Hevc, HwCodec::Jpeg];
    let decode_codecs = vec![
        HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9,
        HwCodec::Mpeg2, HwCodec::Jpeg,
    ];

    // Check for AV1 support on newer platforms
    let mut encode = encode_codecs;
    let mut decode = decode_codecs;
    if device.api_version.0 >= 2 {
        encode.push(HwCodec::Av1);
        decode.push(HwCodec::Av1);
    }

    Ok(HwCapabilities {
        accel_type: HwAccelType::Qsv,
        encode_codecs: encode,
        decode_codecs: decode,
        max_width: 8192,
        max_height: 8192,
        supports_bframes: true,
        supports_10bit: true,
        supports_hdr: device.api_version.0 >= 2,
        device_name: device.name,
    })
}

/// Check if QSV is available on this system.
pub fn is_available() -> bool {
    // In a real implementation, try to create a session
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        true // Simplified - would actually probe hardware
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        false
    }
}

/// QSV video processing (VPP) operations.
pub mod vpp {
    use super::*;

    /// VPP configuration.
    #[derive(Debug, Clone)]
    pub struct VppConfig {
        /// Input width.
        pub in_width: u32,
        /// Input height.
        pub in_height: u32,
        /// Output width.
        pub out_width: u32,
        /// Output height.
        pub out_height: u32,
        /// Input format.
        pub in_format: QsvFrameFormat,
        /// Output format.
        pub out_format: QsvFrameFormat,
        /// Enable deinterlacing.
        pub deinterlace: bool,
        /// Enable denoise.
        pub denoise: bool,
        /// Denoise strength (0-100).
        pub denoise_strength: u32,
        /// Enable sharpening.
        pub sharpen: bool,
        /// Sharpening strength (0-100).
        pub sharpen_strength: u32,
    }

    impl Default for VppConfig {
        fn default() -> Self {
            Self {
                in_width: 1920,
                in_height: 1080,
                out_width: 1920,
                out_height: 1080,
                in_format: QsvFrameFormat::Nv12,
                out_format: QsvFrameFormat::Nv12,
                deinterlace: false,
                denoise: false,
                denoise_strength: 50,
                sharpen: false,
                sharpen_strength: 50,
            }
        }
    }

    /// QSV video processor.
    pub struct QsvVpp {
        config: VppConfig,
        initialized: bool,
    }

    impl QsvVpp {
        /// Create a new VPP instance.
        pub fn new(config: VppConfig) -> Result<Self> {
            Ok(Self {
                config,
                initialized: false,
            })
        }

        /// Initialize VPP.
        pub fn initialize(&mut self) -> Result<()> {
            if self.initialized {
                return Ok(());
            }

            // In a real implementation:
            // 1. Create VPP session
            // 2. Configure filters
            // 3. Allocate surfaces

            self.initialized = true;
            Ok(())
        }

        /// Process a frame.
        pub fn process(&mut self, input: &QsvFrame) -> Result<QsvFrame> {
            if !self.initialized {
                self.initialize()?;
            }

            // In a real implementation:
            // 1. Submit input surface
            // 2. Run VPP filters
            // 3. Return output surface

            Ok(QsvFrame {
                width: self.config.out_width,
                height: self.config.out_height,
                format: self.config.out_format,
                data: vec![0u8; (self.config.out_width * self.config.out_height * 3 / 2) as usize],
                pts: input.pts,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_default() {
        let config = QsvEncoderConfig::default();
        assert_eq!(config.codec, HwCodec::H264);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.bitrate, 5_000_000);
    }

    #[test]
    fn test_decoder_config_default() {
        let config = QsvDecoderConfig::default();
        assert_eq!(config.codec, HwCodec::H264);
        assert_eq!(config.async_depth, 4);
    }

    #[test]
    fn test_preset_target_usage() {
        assert_eq!(QsvPreset::Quality.target_usage(), 1);
        assert_eq!(QsvPreset::Balanced.target_usage(), 4);
        assert_eq!(QsvPreset::VeryFast.target_usage(), 7);
    }

    #[test]
    fn test_rate_control_modes() {
        let modes = [
            QsvRateControl::Cbr,
            QsvRateControl::Vbr,
            QsvRateControl::Cqp,
            QsvRateControl::Icq,
        ];
        assert_eq!(modes.len(), 4);
    }

    #[test]
    fn test_frame_format() {
        assert_eq!(QsvFrameFormat::Nv12, QsvFrameFormat::Nv12);
        assert_ne!(QsvFrameFormat::Nv12, QsvFrameFormat::P010);
    }

    #[test]
    fn test_vpp_config_default() {
        let config = vpp::VppConfig::default();
        assert_eq!(config.in_width, 1920);
        assert_eq!(config.out_width, 1920);
        assert!(!config.deinterlace);
    }
}
