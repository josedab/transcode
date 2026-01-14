//! Apple Lossless Audio Codec (ALAC) implementation.
//!
//! ALAC is a lossless audio codec developed by Apple. It supports
//! bit depths of 16, 20, 24, and 32 bits, with sample rates up to 384 kHz.
//!
//! ## Features
//!
//! - Lossless audio compression
//! - Multi-channel support (1-8 channels)
//! - Multiple bit depths (16, 20, 24, 32)
//! - Adaptive Linear Prediction
//!
//! ## Example
//!
//! ```ignore
//! use transcode_alac::{AlacDecoder, AlacConfig};
//!
//! let config = AlacConfig::from_magic_cookie(&cookie)?;
//! let mut decoder = AlacDecoder::new(config)?;
//! let samples = decoder.decode(&compressed_packet)?;
//! ```

#![warn(missing_docs)]

pub mod error;
pub mod bitstream;
pub mod decoder;

#[cfg(feature = "encoder")]
pub mod encoder;

pub use error::{AlacError, Result};
pub use decoder::{AlacDecoder, DecodedPacket};
pub use bitstream::BitReader;

#[cfg(feature = "encoder")]
pub use bitstream::BitWriter;

#[cfg(feature = "encoder")]
pub use encoder::{AlacEncoder, AlacEncoderConfig, EncodedPacket};

/// ALAC magic cookie header identifier.
pub const ALAC_MAGIC: u32 = 0x616C6163; // 'alac'

/// Maximum frame size.
pub const MAX_FRAME_SIZE: usize = 4096;

/// Default frames per packet.
pub const DEFAULT_FRAMES_PER_PACKET: u32 = 4096;

/// ALAC configuration from magic cookie.
#[derive(Debug, Clone)]
pub struct AlacConfig {
    /// Frame length (samples per packet).
    pub frame_length: u32,
    /// Compatible version.
    pub compatible_version: u8,
    /// Bit depth.
    pub bit_depth: u8,
    /// Tuning parameter: pb.
    pub pb: u8,
    /// Tuning parameter: mb.
    pub mb: u8,
    /// Tuning parameter: kb.
    pub kb: u8,
    /// Number of channels.
    pub num_channels: u8,
    /// Maximum run (for Rice coding).
    pub max_run: u16,
    /// Maximum frame bytes.
    pub max_frame_bytes: u32,
    /// Average bit rate.
    pub avg_bit_rate: u32,
    /// Sample rate.
    pub sample_rate: u32,
}

impl AlacConfig {
    /// Create from magic cookie data.
    pub fn from_magic_cookie(data: &[u8]) -> Result<Self> {
        if data.len() < 24 {
            return Err(AlacError::InvalidMagicCookie);
        }

        // Parse ALACSpecificConfig
        // Format: frame_length(4) + compatible_version(1) + bit_depth(1) +
        //         pb(1) + mb(1) + kb(1) + num_channels(1) + max_run(2) +
        //         max_frame_bytes(4) + avg_bit_rate(4) + sample_rate(4)
        let frame_length = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let compatible_version = data[4];
        let bit_depth = data[5];
        let pb = data[6];
        let mb = data[7];
        let kb = data[8];
        let num_channels = data[9];
        let max_run = u16::from_be_bytes([data[10], data[11]]);
        let max_frame_bytes = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        let avg_bit_rate = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let sample_rate = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);

        // Validate bit depth
        if !matches!(bit_depth, 16 | 20 | 24 | 32) {
            return Err(AlacError::UnsupportedBitDepth(bit_depth));
        }

        // Validate channels
        if num_channels == 0 || num_channels > 8 {
            return Err(AlacError::UnsupportedChannels(num_channels));
        }

        Ok(Self {
            frame_length,
            compatible_version,
            bit_depth,
            pb,
            mb,
            kb,
            num_channels,
            max_run,
            max_frame_bytes,
            avg_bit_rate,
            sample_rate,
        })
    }

    /// Create default config for given parameters.
    pub fn new(sample_rate: u32, channels: u8, bit_depth: u8) -> Result<Self> {
        if !matches!(bit_depth, 16 | 20 | 24 | 32) {
            return Err(AlacError::UnsupportedBitDepth(bit_depth));
        }

        if channels == 0 || channels > 8 {
            return Err(AlacError::UnsupportedChannels(channels));
        }

        Ok(Self {
            frame_length: DEFAULT_FRAMES_PER_PACKET,
            compatible_version: 0,
            bit_depth,
            pb: 40,
            mb: 10,
            kb: 14,
            num_channels: channels,
            max_run: 255,
            max_frame_bytes: 0,
            avg_bit_rate: 0,
            sample_rate,
        })
    }

    /// Get bytes per sample (all channels).
    pub fn bytes_per_frame(&self) -> usize {
        self.num_channels as usize * (self.bit_depth as usize / 8)
    }

    /// Convert to magic cookie bytes.
    pub fn to_magic_cookie(&self) -> Vec<u8> {
        let mut cookie = Vec::with_capacity(24);
        cookie.extend_from_slice(&self.frame_length.to_be_bytes());
        cookie.push(self.compatible_version);
        cookie.push(self.bit_depth);
        cookie.push(self.pb);
        cookie.push(self.mb);
        cookie.push(self.kb);
        cookie.push(self.num_channels);
        cookie.extend_from_slice(&self.max_run.to_be_bytes());
        cookie.extend_from_slice(&self.max_frame_bytes.to_be_bytes());
        cookie.extend_from_slice(&self.avg_bit_rate.to_be_bytes());
        cookie.extend_from_slice(&self.sample_rate.to_be_bytes());
        cookie
    }
}

/// Channel layout for multi-channel audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Mono (1 channel).
    Mono,
    /// Stereo (2 channels).
    Stereo,
    /// 3.0 (L, R, C).
    Layout30,
    /// 4.0 (L, R, Ls, Rs).
    Layout40,
    /// 5.0 (L, R, C, Ls, Rs).
    Layout50,
    /// 5.1 (L, R, C, LFE, Ls, Rs).
    Layout51,
    /// 6.1 (L, R, C, LFE, Ls, Rs, Cs).
    Layout61,
    /// 7.1 (L, R, C, LFE, Ls, Rs, Lrs, Rrs).
    Layout71,
}

impl ChannelLayout {
    /// Get channel count for layout.
    pub fn channel_count(&self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Layout30 => 3,
            Self::Layout40 => 4,
            Self::Layout50 => 5,
            Self::Layout51 => 6,
            Self::Layout61 => 7,
            Self::Layout71 => 8,
        }
    }

    /// Create from channel count.
    pub fn from_channels(channels: u8) -> Option<Self> {
        match channels {
            1 => Some(Self::Mono),
            2 => Some(Self::Stereo),
            3 => Some(Self::Layout30),
            4 => Some(Self::Layout40),
            5 => Some(Self::Layout50),
            6 => Some(Self::Layout51),
            7 => Some(Self::Layout61),
            8 => Some(Self::Layout71),
            _ => None,
        }
    }
}

/// ALAC predictor coefficients.
#[derive(Debug, Clone, Default)]
pub struct PredictorInfo {
    /// Prediction type.
    pub prediction_type: u16,
    /// Quantization shift.
    pub quant_shift: u16,
    /// Rice modifier.
    pub rice_modifier: u16,
    /// Predictor coefficients count.
    pub predictor_coef_num: u16,
    /// Predictor coefficients.
    pub predictor_coef: [i16; 32],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = AlacConfig::new(44100, 2, 16).unwrap();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.num_channels, 2);
        assert_eq!(config.bit_depth, 16);
    }

    #[test]
    fn test_config_invalid_bit_depth() {
        let result = AlacConfig::new(44100, 2, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_invalid_channels() {
        let result = AlacConfig::new(44100, 0, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_magic_cookie_roundtrip() {
        let config = AlacConfig::new(48000, 6, 24).unwrap();
        let cookie = config.to_magic_cookie();
        let parsed = AlacConfig::from_magic_cookie(&cookie).unwrap();

        assert_eq!(parsed.sample_rate, 48000);
        assert_eq!(parsed.num_channels, 6);
        assert_eq!(parsed.bit_depth, 24);
    }

    #[test]
    fn test_channel_layout() {
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Layout51.channel_count(), 6);
        assert_eq!(ChannelLayout::from_channels(2), Some(ChannelLayout::Stereo));
    }
}
