//! Hardware decoder implementation.

use crate::error::{HwAccelError, Result};
use crate::types::*;
use crate::{HwAccelType, HwCodec};

/// Hardware decoder configuration.
#[derive(Debug, Clone)]
pub struct HwDecoderConfig {
    /// Target codec.
    pub codec: HwCodec,
    /// Output format.
    pub output_format: HwSurfaceFormat,
    /// Number of surfaces in pool.
    pub surface_count: u32,
    /// Extra data (codec-specific headers).
    pub extra_data: Option<Vec<u8>>,
}

impl Default for HwDecoderConfig {
    fn default() -> Self {
        Self {
            codec: HwCodec::H264,
            output_format: HwSurfaceFormat::Nv12,
            surface_count: 8,
            extra_data: None,
        }
    }
}

impl HwDecoderConfig {
    /// Create config for H.264 decoding.
    pub fn h264() -> Self {
        Self {
            codec: HwCodec::H264,
            ..Default::default()
        }
    }

    /// Create config for HEVC decoding.
    pub fn hevc() -> Self {
        Self {
            codec: HwCodec::Hevc,
            ..Default::default()
        }
    }

    /// Create config for AV1 decoding.
    pub fn av1() -> Self {
        Self {
            codec: HwCodec::Av1,
            ..Default::default()
        }
    }

    /// Create config for VP9 decoding.
    pub fn vp9() -> Self {
        Self {
            codec: HwCodec::Vp9,
            ..Default::default()
        }
    }

    /// Set output format.
    pub fn with_output_format(mut self, format: HwSurfaceFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set extra data.
    pub fn with_extra_data(mut self, data: Vec<u8>) -> Self {
        self.extra_data = Some(data);
        self
    }
}

/// Encoded packet for decoding.
#[derive(Debug)]
pub struct DecoderPacket {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: i64,
    /// Decode timestamp.
    pub dts: i64,
    /// Is this a keyframe?
    pub is_keyframe: bool,
}

impl DecoderPacket {
    /// Create a new packet.
    pub fn new(data: Vec<u8>, pts: i64) -> Self {
        Self {
            data,
            pts,
            dts: pts,
            is_keyframe: false,
        }
    }

    /// Mark as keyframe.
    pub fn keyframe(mut self) -> Self {
        self.is_keyframe = true;
        self
    }
}

/// Hardware decoder.
pub struct HwDecoder {
    config: HwDecoderConfig,
    accel_type: HwAccelType,
    _device: HwDeviceContext,
    frame_count: u64,
    initialized: bool,
    width: u32,
    height: u32,
}

impl HwDecoder {
    /// Create a new hardware decoder.
    pub fn new(accel_type: HwAccelType, config: HwDecoderConfig) -> Result<Self> {
        let device = HwDeviceContext::new(accel_type, 0);

        Ok(Self {
            config,
            accel_type,
            _device: device,
            frame_count: 0,
            initialized: false,
            width: 0,
            height: 0,
        })
    }

    /// Initialize the decoder.
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
                self.init_nvdec()?;
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

    /// Decode a packet.
    pub fn decode(&mut self, packet: &DecoderPacket) -> Result<Option<HwFrame>> {
        if !self.initialized {
            self.init()?;
        }

        // Decode based on accelerator type
        let frame = match self.accel_type {
            #[cfg(target_os = "macos")]
            HwAccelType::VideoToolbox => self.decode_videotoolbox(packet)?,
            #[cfg(target_os = "linux")]
            HwAccelType::Vaapi => self.decode_vaapi(packet)?,
            HwAccelType::Nvenc => self.decode_nvdec(packet)?,
            _ => self.decode_software(packet)?,
        };

        if frame.is_some() {
            self.frame_count += 1;
        }

        Ok(frame)
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> Result<Vec<HwFrame>> {
        let mut frames = Vec::new();

        // Keep decoding until no more frames
        loop {
            let empty_packet = DecoderPacket {
                data: Vec::new(),
                pts: -1,
                dts: -1,
                is_keyframe: false,
            };

            match self.decode(&empty_packet)? {
                Some(frame) => frames.push(frame),
                None => break,
            }
        }

        Ok(frames)
    }

    /// Get decoder statistics.
    pub fn stats(&self) -> HwDecoderStats {
        HwDecoderStats {
            frames_decoded: self.frame_count,
            accel_type: self.accel_type,
            codec: self.config.codec,
            width: self.width,
            height: self.height,
        }
    }

    /// Get decoded video dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    // Platform-specific implementations

    #[cfg(target_os = "macos")]
    fn init_videotoolbox(&mut self) -> Result<()> {
        tracing::info!("Initializing VideoToolbox decoder");
        // In a real implementation, this would create a VTDecompressionSession
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn init_vaapi(&mut self) -> Result<()> {
        tracing::info!("Initializing VA-API decoder");
        Ok(())
    }

    fn init_nvdec(&mut self) -> Result<()> {
        tracing::info!("Initializing NVDEC decoder");
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn decode_videotoolbox(&mut self, packet: &DecoderPacket) -> Result<Option<HwFrame>> {
        if packet.data.is_empty() {
            return Ok(None);
        }

        // Simulate decoding
        self.width = 1920;
        self.height = 1080;

        Ok(Some(HwFrame {
            format: self.config.output_format,
            width: self.width,
            height: self.height,
            pts: packet.pts,
            handle: HwFrameHandle::Cpu(vec![0u8; (self.width * self.height * 3 / 2) as usize]),
        }))
    }

    #[cfg(target_os = "linux")]
    fn decode_vaapi(&mut self, packet: &DecoderPacket) -> Result<Option<HwFrame>> {
        if packet.data.is_empty() {
            return Ok(None);
        }

        self.width = 1920;
        self.height = 1080;

        Ok(Some(HwFrame {
            format: self.config.output_format,
            width: self.width,
            height: self.height,
            pts: packet.pts,
            handle: HwFrameHandle::Cpu(vec![0u8; (self.width * self.height * 3 / 2) as usize]),
        }))
    }

    fn decode_nvdec(&mut self, packet: &DecoderPacket) -> Result<Option<HwFrame>> {
        if packet.data.is_empty() {
            return Ok(None);
        }

        self.width = 1920;
        self.height = 1080;

        Ok(Some(HwFrame {
            format: self.config.output_format,
            width: self.width,
            height: self.height,
            pts: packet.pts,
            handle: HwFrameHandle::Cpu(vec![0u8; (self.width * self.height * 3 / 2) as usize]),
        }))
    }

    fn decode_software(&mut self, packet: &DecoderPacket) -> Result<Option<HwFrame>> {
        if packet.data.is_empty() {
            return Ok(None);
        }

        self.width = 1920;
        self.height = 1080;

        Ok(Some(HwFrame {
            format: self.config.output_format,
            width: self.width,
            height: self.height,
            pts: packet.pts,
            handle: HwFrameHandle::Cpu(vec![0u8; (self.width * self.height * 3 / 2) as usize]),
        }))
    }
}

/// Decoder statistics.
#[derive(Debug, Clone)]
pub struct HwDecoderStats {
    /// Number of frames decoded.
    pub frames_decoded: u64,
    /// Accelerator type used.
    pub accel_type: HwAccelType,
    /// Codec used.
    pub codec: HwCodec,
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_config() {
        let config = HwDecoderConfig::h264();
        assert_eq!(config.codec, HwCodec::H264);
    }

    #[test]
    fn test_decoder_creation() {
        let config = HwDecoderConfig::h264();
        let decoder = HwDecoder::new(HwAccelType::Software, config);
        assert!(decoder.is_ok());
    }
}
