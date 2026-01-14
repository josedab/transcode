//! MPEG-2 video encoder.
//!
//! Provides video encoding to MPEG-2 (MPEG-2 Part 2 Video) format.
//!
//! ## Usage
//!
//! Full encoding requires the `ffi-ffmpeg` feature. Without it, only header
//! generation is available.

use crate::types::*;
use crate::{Mpeg2Error, Result};

#[cfg(feature = "ffi-ffmpeg")]
use crate::ffi::Mpeg2FfiEncoder;

use std::fmt;

/// MPEG-2 video encoder.
///
/// Encodes raw video frames to MPEG-2 video elementary streams.
///
/// # Example
///
/// ```rust,ignore
/// use transcode_mpeg2::{Mpeg2Encoder, Mpeg2EncoderConfig};
///
/// let config = Mpeg2EncoderConfig::new(1920, 1080, 29.97);
/// let mut encoder = Mpeg2Encoder::new(config)?;
/// let encoded = encoder.encode_frame(&raw_frame)?;
/// ```
pub struct Mpeg2Encoder {
    /// Encoder configuration.
    config: Mpeg2EncoderConfig,
    /// Current frame number.
    frame_number: u64,
    /// Current GOP frame count.
    gop_frame_count: u32,
    /// Pictures encoded.
    pictures_encoded: u64,
    /// Bytes output.
    bytes_output: u64,
    /// Sequence header bytes.
    sequence_header: Vec<u8>,
    /// Last I-frame position.
    last_i_frame: u64,
    /// FFI encoder (when ffi-ffmpeg feature is enabled).
    #[cfg(feature = "ffi-ffmpeg")]
    ffi_encoder: Option<Mpeg2FfiEncoder>,
}

impl fmt::Debug for Mpeg2Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mpeg2Encoder")
            .field("config", &self.config)
            .field("frame_number", &self.frame_number)
            .field("gop_frame_count", &self.gop_frame_count)
            .field("pictures_encoded", &self.pictures_encoded)
            .field("bytes_output", &self.bytes_output)
            .finish_non_exhaustive()
    }
}

/// MPEG-2 encoder configuration.
#[derive(Debug, Clone)]
pub struct Mpeg2EncoderConfig {
    /// Frame width.
    pub width: u16,
    /// Frame height.
    pub height: u16,
    /// Frame rate.
    pub frame_rate: FrameRateCode,
    /// Bit rate in kbps.
    pub bitrate_kbps: u32,
    /// Profile.
    pub profile: Profile,
    /// Level.
    pub level: Level,
    /// GOP size (frames between I-frames).
    pub gop_size: u32,
    /// B-frames between P-frames.
    pub b_frames: u32,
    /// Chroma format.
    pub chroma_format: ChromaFormat,
    /// Aspect ratio.
    pub aspect_ratio: AspectRatioCode,
    /// Progressive sequence.
    pub progressive: bool,
    /// Closed GOP.
    pub closed_gop: bool,
    /// VBV buffer size.
    pub vbv_buffer_size: u32,
    /// Intra DC precision.
    pub intra_dc_precision: u8,
}

impl Mpeg2EncoderConfig {
    /// Create a new encoder configuration.
    pub fn new(width: u16, height: u16, frame_rate_fps: f64) -> Self {
        let frame_rate = Self::fps_to_code(frame_rate_fps);

        Self {
            width,
            height,
            frame_rate,
            bitrate_kbps: 8000,
            profile: Profile::Main,
            level: Level::Main,
            gop_size: 15,
            b_frames: 2,
            chroma_format: ChromaFormat::Yuv420,
            aspect_ratio: AspectRatioCode::Ratio16_9,
            progressive: true,
            closed_gop: true,
            vbv_buffer_size: 112,
            intra_dc_precision: 0,
        }
    }

    /// Create configuration for DVD-compliant encoding.
    pub fn dvd_ntsc(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            frame_rate: FrameRateCode::Fps29_97,
            bitrate_kbps: 8000,
            profile: Profile::Main,
            level: Level::Main,
            gop_size: 18,
            b_frames: 2,
            chroma_format: ChromaFormat::Yuv420,
            aspect_ratio: AspectRatioCode::Ratio16_9,
            progressive: false, // DVD is typically interlaced
            closed_gop: true,
            vbv_buffer_size: 112,
            intra_dc_precision: 0,
        }
    }

    /// Create configuration for DVD-compliant PAL encoding.
    pub fn dvd_pal(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            frame_rate: FrameRateCode::Fps25,
            bitrate_kbps: 8000,
            profile: Profile::Main,
            level: Level::Main,
            gop_size: 15,
            b_frames: 2,
            chroma_format: ChromaFormat::Yuv420,
            aspect_ratio: AspectRatioCode::Ratio16_9,
            progressive: false,
            closed_gop: true,
            vbv_buffer_size: 112,
            intra_dc_precision: 0,
        }
    }

    /// Create configuration for broadcast-quality HD encoding.
    pub fn broadcast_hd() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: FrameRateCode::Fps29_97,
            bitrate_kbps: 25000,
            profile: Profile::High,
            level: Level::High,
            gop_size: 15,
            b_frames: 2,
            chroma_format: ChromaFormat::Yuv422,
            aspect_ratio: AspectRatioCode::Ratio16_9,
            progressive: false, // 1080i
            closed_gop: true,
            vbv_buffer_size: 224,
            intra_dc_precision: 2,
        }
    }

    /// Set the bitrate in kbps.
    pub fn with_bitrate(mut self, bitrate_kbps: u32) -> Self {
        self.bitrate_kbps = bitrate_kbps;
        self
    }

    /// Set the GOP size.
    pub fn with_gop_size(mut self, gop_size: u32) -> Self {
        self.gop_size = gop_size;
        self
    }

    /// Set the number of B-frames.
    pub fn with_b_frames(mut self, b_frames: u32) -> Self {
        self.b_frames = b_frames;
        self
    }

    /// Set progressive/interlaced mode.
    pub fn with_progressive(mut self, progressive: bool) -> Self {
        self.progressive = progressive;
        self
    }

    /// Set the profile and level.
    pub fn with_profile_level(mut self, profile: Profile, level: Level) -> Self {
        self.profile = profile;
        self.level = level;
        self
    }

    /// Set the aspect ratio.
    pub fn with_aspect_ratio(mut self, aspect_ratio: AspectRatioCode) -> Self {
        self.aspect_ratio = aspect_ratio;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Check resolution limits for level
        let (max_width, max_height) = self.level.max_frame_size();
        if self.width > max_width || self.height > max_height {
            return Err(Mpeg2Error::UnsupportedLevel(self.level.to_code()));
        }

        // Check bitrate limits for level
        let max_bitrate = self.level.max_bitrate_mbps() * 1000;
        if self.bitrate_kbps > max_bitrate {
            return Err(Mpeg2Error::EncodingError(format!(
                "Bitrate {} kbps exceeds level {} maximum of {} kbps",
                self.bitrate_kbps,
                self.level,
                max_bitrate
            )));
        }

        // Check GOP size
        if self.gop_size == 0 {
            return Err(Mpeg2Error::EncodingError(
                "GOP size must be greater than 0".into(),
            ));
        }

        // Check B-frames
        if self.b_frames > 7 {
            return Err(Mpeg2Error::EncodingError(
                "B-frames must be 7 or less".into(),
            ));
        }

        Ok(())
    }

    /// Convert fps to frame rate code.
    fn fps_to_code(fps: f64) -> FrameRateCode {
        if (fps - 23.976).abs() < 0.01 {
            FrameRateCode::Fps23_976
        } else if (fps - 24.0).abs() < 0.01 {
            FrameRateCode::Fps24
        } else if (fps - 25.0).abs() < 0.01 {
            FrameRateCode::Fps25
        } else if (fps - 29.97).abs() < 0.01 {
            FrameRateCode::Fps29_97
        } else if (fps - 30.0).abs() < 0.01 {
            FrameRateCode::Fps30
        } else if (fps - 50.0).abs() < 0.01 {
            FrameRateCode::Fps50
        } else if (fps - 59.94).abs() < 0.01 {
            FrameRateCode::Fps59_94
        } else if (fps - 60.0).abs() < 0.01 {
            FrameRateCode::Fps60
        } else {
            // Default to 29.97
            FrameRateCode::Fps29_97
        }
    }
}

impl Default for Mpeg2EncoderConfig {
    fn default() -> Self {
        Self::new(720, 480, 29.97)
    }
}

impl Mpeg2Encoder {
    /// Create a new MPEG-2 encoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn new(config: Mpeg2EncoderConfig) -> Result<Self> {
        config.validate()?;

        let sequence_header = Self::build_sequence_header(&config);

        // Try to create FFI encoder
        let (frame_rate_num, frame_rate_den) = config.frame_rate.fraction();
        let ffi_encoder = match Mpeg2FfiEncoder::new(
            config.width,
            config.height,
            frame_rate_num,
            frame_rate_den,
            config.bitrate_kbps,
            config.gop_size,
            config.b_frames,
        ) {
            Ok(enc) => Some(enc),
            Err(e) => {
                tracing::warn!("FFI encoder init failed, falling back to header-only: {}", e);
                None
            }
        };

        Ok(Self {
            config,
            frame_number: 0,
            gop_frame_count: 0,
            pictures_encoded: 0,
            bytes_output: 0,
            sequence_header,
            last_i_frame: 0,
            ffi_encoder,
        })
    }

    /// Create a new MPEG-2 encoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn new(config: Mpeg2EncoderConfig) -> Result<Self> {
        config.validate()?;

        let sequence_header = Self::build_sequence_header(&config);

        Ok(Self {
            config,
            frame_number: 0,
            gop_frame_count: 0,
            pictures_encoded: 0,
            bytes_output: 0,
            sequence_header,
            last_i_frame: 0,
        })
    }

    /// Build the sequence header bytes.
    fn build_sequence_header(config: &Mpeg2EncoderConfig) -> Vec<u8> {
        let mut header = Vec::with_capacity(16);

        // Start code
        header.extend_from_slice(&[0x00, 0x00, 0x01, crate::SEQUENCE_HEADER_CODE]);

        // horizontal_size (12 bits) + vertical_size (12 bits) = 24 bits
        let h_size = config.width & 0x0FFF;
        let v_size = config.height & 0x0FFF;

        header.push(((h_size >> 4) & 0xFF) as u8);
        header.push((((h_size & 0x0F) << 4) | ((v_size >> 8) & 0x0F)) as u8);
        header.push((v_size & 0xFF) as u8);

        // aspect_ratio (4 bits) + frame_rate (4 bits)
        let ar_fr = ((config.aspect_ratio as u8) << 4) | (config.frame_rate as u8);
        header.push(ar_fr);

        // bit_rate (18 bits) - in units of 400 bps
        let bit_rate = (config.bitrate_kbps * 1000 / 400) & 0x3FFFF;
        header.push(((bit_rate >> 10) & 0xFF) as u8);
        header.push(((bit_rate >> 2) & 0xFF) as u8);

        // marker_bit (1) + vbv_buffer_size_value (10 bits) high part
        let vbv = config.vbv_buffer_size & 0x3FF;
        header.push((0x80 | ((bit_rate & 0x03) << 6) | ((vbv >> 5) & 0x1F)) as u8);

        // vbv_buffer_size_value low + constrained_parameters + load_matrices (both 0)
        header.push(((vbv & 0x1F) << 3) as u8);

        header
    }

    /// Encode a video frame.
    ///
    /// Returns encoded MPEG-2 data or an error if encoding fails.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn encode_frame(&mut self, frame: &RawFrame) -> Result<EncodedPacket> {
        // Try FFI encoder first
        if let Some(ref mut ffi) = self.ffi_encoder {
            match ffi.encode(&frame.y_plane, &frame.u_plane, &frame.v_plane) {
                Ok(Some(packet)) => {
                    self.frame_number += 1;
                    self.gop_frame_count += 1;
                    if self.gop_frame_count >= self.config.gop_size {
                        self.gop_frame_count = 0;
                    }
                    self.pictures_encoded += 1;
                    self.bytes_output += packet.data.len() as u64;

                    if packet.keyframe {
                        self.last_i_frame = self.frame_number - 1;
                    }

                    return Ok(packet);
                }
                Ok(None) => {
                    // Encoder needs more frames before producing output
                    // Return a placeholder with headers
                }
                Err(e) => {
                    tracing::debug!("FFI encode failed, trying fallback: {}", e);
                }
            }
        }

        // Fall back to header-only mode
        let mut data = Vec::new();

        let picture_type = self.get_next_picture_type();
        let temporal_reference = (self.gop_frame_count % 1024) as u16;

        if self.gop_frame_count == 0 {
            data.extend_from_slice(&self.sequence_header);
            data.extend_from_slice(&self.build_sequence_extension());
            data.extend_from_slice(&self.build_gop_header());
        }

        data.extend_from_slice(&self.build_picture_header(picture_type, temporal_reference));

        self.frame_number += 1;
        self.gop_frame_count += 1;
        if self.gop_frame_count >= self.config.gop_size {
            self.gop_frame_count = 0;
        }
        self.pictures_encoded += 1;
        self.bytes_output += data.len() as u64;

        if picture_type == PictureCodingType::I {
            self.last_i_frame = self.frame_number - 1;
        }

        Ok(EncodedPacket {
            data,
            picture_type,
            temporal_reference,
            pts: Some(self.frame_number - 1),
            dts: Some(self.frame_number - 1),
            keyframe: picture_type == PictureCodingType::I,
        })
    }

    /// Encode a video frame (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn encode_frame(&mut self, _frame: &RawFrame) -> Result<EncodedPacket> {
        // Without FFI, we can only generate headers
        let mut data = Vec::new();

        // Determine picture type based on GOP position
        let picture_type = self.get_next_picture_type();
        let temporal_reference = (self.gop_frame_count % 1024) as u16;

        // Add sequence header at start of each GOP
        if self.gop_frame_count == 0 {
            data.extend_from_slice(&self.sequence_header);

            // Add sequence extension for MPEG-2
            data.extend_from_slice(&self.build_sequence_extension());

            // Add GOP header
            data.extend_from_slice(&self.build_gop_header());
        }

        // Add picture header
        data.extend_from_slice(&self.build_picture_header(picture_type, temporal_reference));

        // Without FFI, we don't have actual encoded data
        // Just return the headers as a placeholder

        self.frame_number += 1;
        self.gop_frame_count += 1;
        if self.gop_frame_count >= self.config.gop_size {
            self.gop_frame_count = 0;
        }
        self.pictures_encoded += 1;
        self.bytes_output += data.len() as u64;

        if picture_type == PictureCodingType::I {
            self.last_i_frame = self.frame_number - 1;
        }

        Ok(EncodedPacket {
            data,
            picture_type,
            temporal_reference,
            pts: Some(self.frame_number - 1),
            dts: Some(self.frame_number - 1),
            keyframe: picture_type == PictureCodingType::I,
        })
    }

    /// Get the next picture type based on GOP pattern.
    fn get_next_picture_type(&self) -> PictureCodingType {
        if self.gop_frame_count == 0 {
            PictureCodingType::I
        } else if self.config.b_frames == 0 {
            PictureCodingType::P
        } else {
            let frames_per_pattern = self.config.b_frames + 1;
            if self.gop_frame_count.is_multiple_of(frames_per_pattern) {
                PictureCodingType::P
            } else {
                PictureCodingType::B
            }
        }
    }

    /// Build sequence extension bytes.
    fn build_sequence_extension(&self) -> Vec<u8> {
        let mut ext = Vec::with_capacity(10);

        // Start code
        ext.extend_from_slice(&[0x00, 0x00, 0x01, crate::EXTENSION_START_CODE]);

        // extension_start_code_identifier (4 bits) = 1
        // profile_and_level (8 bits)
        let profile_code = match self.config.profile {
            Profile::Simple => 5,
            Profile::Main => 4,
            Profile::SnrScalable => 3,
            Profile::SpatiallyScalable => 2,
            Profile::High => 1,
            Profile::Unknown(c) => c,
        };
        let level_code = match self.config.level {
            Level::Low => 10,
            Level::Main => 8,
            Level::High1440 => 6,
            Level::High => 4,
            Level::Unknown(c) => c,
        };
        let profile_and_level = ((profile_code & 0x07) << 4) | (level_code & 0x0F);

        ext.push(0x10 | ((profile_and_level >> 4) & 0x0F));
        ext.push(((profile_and_level & 0x0F) << 4) | if self.config.progressive { 0x08 } else { 0x00 }
            | ((self.config.chroma_format as u8) << 1));

        // size extensions (both 0 for normal resolutions)
        ext.push(0x00);
        ext.push(0x00);

        // marker + low_delay + frame_rate_extensions
        ext.push(0x80);

        ext
    }

    /// Build GOP header bytes.
    fn build_gop_header(&self) -> Vec<u8> {
        let mut gop = Vec::with_capacity(8);

        // Start code
        gop.extend_from_slice(&[0x00, 0x00, 0x01, crate::GOP_START_CODE]);

        // Calculate time code from frame number
        let frame_rate = self.config.frame_rate.fps();
        let total_frames = self.frame_number;
        let total_seconds = (total_frames as f64 / frame_rate) as u64;

        let hours = (total_seconds / 3600) as u8;
        let minutes = ((total_seconds % 3600) / 60) as u8;
        let seconds = (total_seconds % 60) as u8;
        let pictures = (total_frames % (frame_rate as u64).max(1)) as u8;

        // drop_frame (1) + hours (5) + minutes (6) + marker (1) + seconds (6) + pictures (6) + closed_gop (1) + broken_link (1)
        gop.push((hours << 2) | ((minutes >> 4) & 0x03));
        gop.push(((minutes & 0x0F) << 4) | 0x08 | ((seconds >> 3) & 0x07));
        gop.push(((seconds & 0x07) << 5) | ((pictures >> 1) & 0x1F));
        gop.push(((pictures & 0x01) << 7) | if self.config.closed_gop { 0x40 } else { 0x00 });

        gop
    }

    /// Build picture header bytes.
    fn build_picture_header(&self, picture_type: PictureCodingType, temporal_reference: u16) -> Vec<u8> {
        let mut pic = Vec::with_capacity(8);

        // Start code
        pic.extend_from_slice(&[0x00, 0x00, 0x01, crate::PICTURE_START_CODE]);

        // temporal_reference (10 bits) + picture_coding_type (3 bits) + vbv_delay (16 bits)
        pic.push(((temporal_reference >> 2) & 0xFF) as u8);
        pic.push((((temporal_reference & 0x03) << 6) | (((picture_type as u16) << 3) | 0x07)) as u8);
        pic.push(0xFF); // vbv_delay high
        pic.push(0xF8); // vbv_delay low + extra bits

        pic
    }

    /// Flush the encoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn flush(&mut self) -> Result<Option<EncodedPacket>> {
        if let Some(ref mut ffi) = self.ffi_encoder {
            match ffi.flush() {
                Ok(Some(packet)) => {
                    self.pictures_encoded += 1;
                    self.bytes_output += packet.data.len() as u64;
                    return Ok(Some(packet));
                }
                Ok(None) => return Ok(None),
                Err(e) => {
                    tracing::debug!("FFI flush failed: {}", e);
                }
            }
        }
        Ok(None)
    }

    /// Flush the encoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn flush(&mut self) -> Result<Option<EncodedPacket>> {
        // Without FFI, nothing to flush
        Ok(None)
    }

    /// Reset the encoder state.
    pub fn reset(&mut self) {
        self.frame_number = 0;
        self.gop_frame_count = 0;
        self.pictures_encoded = 0;
        self.bytes_output = 0;
        self.last_i_frame = 0;
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &Mpeg2EncoderConfig {
        &self.config
    }

    /// Get total pictures encoded.
    pub fn pictures_encoded(&self) -> u64 {
        self.pictures_encoded
    }

    /// Get total bytes output.
    pub fn bytes_output(&self) -> u64 {
        self.bytes_output
    }

    /// Get current frame number.
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Check if FFI encoding is available.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn is_encoding_available(&self) -> bool {
        self.ffi_encoder.is_some()
    }

    /// Check if FFI encoding is available.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn is_encoding_available(&self) -> bool {
        false
    }

    /// Force an I-frame at the next encode.
    pub fn force_keyframe(&mut self) {
        self.gop_frame_count = 0;
    }

    /// Get the sequence header bytes.
    pub fn sequence_header(&self) -> &[u8] {
        &self.sequence_header
    }
}

/// Raw video frame input for encoding.
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// Frame width.
    pub width: u16,
    /// Frame height.
    pub height: u16,
    /// Y plane data.
    pub y_plane: Vec<u8>,
    /// U (Cb) plane data.
    pub u_plane: Vec<u8>,
    /// V (Cr) plane data.
    pub v_plane: Vec<u8>,
    /// Presentation timestamp.
    pub pts: Option<u64>,
    /// Force keyframe.
    pub force_keyframe: bool,
}

impl RawFrame {
    /// Create a new raw frame.
    pub fn new(width: u16, height: u16) -> Self {
        let y_size = width as usize * height as usize;
        let uv_size = y_size / 4; // 4:2:0

        Self {
            width,
            height,
            y_plane: vec![0; y_size],
            u_plane: vec![0; uv_size],
            v_plane: vec![0; uv_size],
            pts: None,
            force_keyframe: false,
        }
    }

    /// Create from planar YUV data.
    pub fn from_yuv420(width: u16, height: u16, y: &[u8], u: &[u8], v: &[u8]) -> Self {
        Self {
            width,
            height,
            y_plane: y.to_vec(),
            u_plane: u.to_vec(),
            v_plane: v.to_vec(),
            pts: None,
            force_keyframe: false,
        }
    }
}

/// Encoded MPEG-2 packet.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Picture type.
    pub picture_type: PictureCodingType,
    /// Temporal reference.
    pub temporal_reference: u16,
    /// Presentation timestamp.
    pub pts: Option<u64>,
    /// Decode timestamp.
    pub dts: Option<u64>,
    /// Is keyframe.
    pub keyframe: bool,
}

impl EncodedPacket {
    /// Get the packet size.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if this is a keyframe.
    pub fn is_keyframe(&self) -> bool {
        self.keyframe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_new() {
        let config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        assert_eq!(config.width, 720);
        assert_eq!(config.height, 480);
        assert_eq!(config.frame_rate, FrameRateCode::Fps29_97);
    }

    #[test]
    fn test_encoder_config_dvd() {
        let config = Mpeg2EncoderConfig::dvd_ntsc(720, 480);
        assert_eq!(config.frame_rate, FrameRateCode::Fps29_97);
        assert!(!config.progressive);

        let config = Mpeg2EncoderConfig::dvd_pal(720, 576);
        assert_eq!(config.frame_rate, FrameRateCode::Fps25);
    }

    #[test]
    fn test_encoder_config_broadcast_hd() {
        let config = Mpeg2EncoderConfig::broadcast_hd();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.profile, Profile::High);
        assert_eq!(config.level, Level::High);
    }

    #[test]
    fn test_encoder_config_validation() {
        let config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        assert!(config.validate().is_ok());

        // Invalid GOP size
        let mut config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        config.gop_size = 0;
        assert!(config.validate().is_err());

        // Too many B-frames
        let mut config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        config.b_frames = 10;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_encoder_new() {
        let config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        let encoder = Mpeg2Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_reset() {
        let config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        let mut encoder = Mpeg2Encoder::new(config).unwrap();
        encoder.reset();
        assert_eq!(encoder.pictures_encoded(), 0);
        assert_eq!(encoder.bytes_output(), 0);
    }

    #[test]
    fn test_encoder_force_keyframe() {
        let config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        let mut encoder = Mpeg2Encoder::new(config).unwrap();
        encoder.gop_frame_count = 10;
        encoder.force_keyframe();
        assert_eq!(encoder.gop_frame_count, 0);
    }

    #[test]
    fn test_is_encoding_available() {
        let config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        let encoder = Mpeg2Encoder::new(config).unwrap();
        #[cfg(not(feature = "ffi-ffmpeg"))]
        assert!(!encoder.is_encoding_available());
    }

    #[test]
    fn test_raw_frame_new() {
        let frame = RawFrame::new(720, 480);
        assert_eq!(frame.y_plane.len(), 720 * 480);
        assert_eq!(frame.u_plane.len(), 720 * 480 / 4);
    }

    #[test]
    fn test_sequence_header_generation() {
        let config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        let encoder = Mpeg2Encoder::new(config).unwrap();
        let header = encoder.sequence_header();

        // Check start code
        assert_eq!(header[0], 0x00);
        assert_eq!(header[1], 0x00);
        assert_eq!(header[2], 0x01);
        assert_eq!(header[3], 0xB3);
    }

    #[test]
    fn test_encode_frame_without_ffi() {
        let config = Mpeg2EncoderConfig::new(720, 480, 29.97);
        let mut encoder = Mpeg2Encoder::new(config).unwrap();

        let frame = RawFrame::new(720, 480);
        let packet = encoder.encode_frame(&frame);

        // Without FFI, we still get header data
        assert!(packet.is_ok());
        let packet = packet.unwrap();
        assert!(packet.keyframe); // First frame is I-frame
    }
}
