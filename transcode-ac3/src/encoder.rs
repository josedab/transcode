//! AC-3 and E-AC-3 encoders.
//!
//! Provides audio encoding for AC-3 (Dolby Digital) and E-AC-3 (Dolby Digital Plus).
//!
//! ## Important Notice
//!
//! AC-3 and E-AC-3 encoding requires the `ffi-ffmpeg` feature and appropriate
//! Dolby licensing for commercial use.

use crate::types::*;
use crate::{Ac3Error, Result};

/// AC-3 encoder configuration.
#[derive(Debug, Clone)]
pub struct Ac3EncoderConfig {
    /// Sample rate (must be 48000, 44100, or 32000).
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Channel layout.
    pub channel_layout: AudioCodingMode,
    /// Include LFE channel.
    pub lfe: bool,
    /// Target bitrate in bits per second.
    pub bitrate: u32,
    /// Dialogue normalization (-1 to -31 dB).
    pub dialnorm: i8,
    /// Bitstream mode.
    pub bsmod: BitstreamMode,
    /// Audio production information.
    pub audio_production_info: Option<AudioProductionInfo>,
    /// Compression profile.
    pub compression_profile: CompressionProfile,
}

impl Ac3EncoderConfig {
    /// Create a new default configuration.
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        let channel_layout = match channels {
            1 => AudioCodingMode::Mono,
            2 => AudioCodingMode::Stereo,
            3 => AudioCodingMode::ThreeChannelFront,
            4 => AudioCodingMode::TwoChannelPlusTwoSurround,
            5 => AudioCodingMode::FiveChannel,
            6 => AudioCodingMode::FiveChannel, // 5.1
            _ => AudioCodingMode::Stereo,
        };

        Self {
            sample_rate,
            channels,
            channel_layout,
            lfe: channels >= 6,
            bitrate: 384000,
            dialnorm: -31,
            bsmod: BitstreamMode::CompleteMain,
            audio_production_info: None,
            compression_profile: CompressionProfile::None,
        }
    }

    /// Create configuration for 5.1 surround.
    pub fn surround_5_1(sample_rate: u32, bitrate: u32) -> Self {
        Self {
            sample_rate,
            channels: 6,
            channel_layout: AudioCodingMode::FiveChannel,
            lfe: true,
            bitrate,
            dialnorm: -31,
            bsmod: BitstreamMode::CompleteMain,
            audio_production_info: None,
            compression_profile: CompressionProfile::None,
        }
    }

    /// Create configuration for stereo.
    pub fn stereo(sample_rate: u32, bitrate: u32) -> Self {
        Self {
            sample_rate,
            channels: 2,
            channel_layout: AudioCodingMode::Stereo,
            lfe: false,
            bitrate,
            dialnorm: -31,
            bsmod: BitstreamMode::CompleteMain,
            audio_production_info: None,
            compression_profile: CompressionProfile::None,
        }
    }

    /// Set the dialogue normalization value.
    pub fn with_dialnorm(mut self, dialnorm: i8) -> Self {
        self.dialnorm = dialnorm.clamp(-31, -1);
        self
    }

    /// Set the bitstream mode.
    pub fn with_bsmod(mut self, bsmod: BitstreamMode) -> Self {
        self.bsmod = bsmod;
        self
    }

    /// Set the compression profile.
    pub fn with_compression(mut self, profile: CompressionProfile) -> Self {
        self.compression_profile = profile;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Validate sample rate
        if ![48000, 44100, 32000].contains(&self.sample_rate) {
            return Err(Ac3Error::EncodingError(format!(
                "Invalid sample rate: {} (must be 48000, 44100, or 32000)",
                self.sample_rate
            )));
        }

        // Validate bitrate
        let valid_bitrates = [
            32000, 40000, 48000, 56000, 64000, 80000, 96000, 112000, 128000, 160000, 192000,
            224000, 256000, 320000, 384000, 448000, 512000, 576000, 640000,
        ];
        if !valid_bitrates.contains(&self.bitrate) {
            return Err(Ac3Error::EncodingError(format!(
                "Invalid bitrate: {} (must be one of: {:?})",
                self.bitrate, valid_bitrates
            )));
        }

        // Validate dialnorm
        if !(-31..=-1).contains(&self.dialnorm) {
            return Err(Ac3Error::EncodingError(format!(
                "Invalid dialnorm: {} (must be -31 to -1)",
                self.dialnorm
            )));
        }

        Ok(())
    }
}

impl Default for Ac3EncoderConfig {
    fn default() -> Self {
        Self::new(48000, 2)
    }
}

/// E-AC-3 encoder configuration.
#[derive(Debug, Clone)]
pub struct Eac3EncoderConfig {
    /// Sample rate.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Channel layout.
    pub channel_layout: AudioCodingMode,
    /// Include LFE channel.
    pub lfe: bool,
    /// Target bitrate in bits per second.
    pub bitrate: u32,
    /// Dialogue normalization (-1 to -31 dB).
    pub dialnorm: i8,
    /// Bitstream mode.
    pub bsmod: BitstreamMode,
    /// Number of audio blocks per frame (1, 2, 3, or 6).
    pub num_blocks: u8,
    /// Enable coupling.
    pub coupling: bool,
    /// Enable channel extension (7.1).
    pub channel_extension: bool,
    /// Compression profile.
    pub compression_profile: CompressionProfile,
}

impl Eac3EncoderConfig {
    /// Create a new default configuration.
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        let channel_layout = match channels {
            1 => AudioCodingMode::Mono,
            2 => AudioCodingMode::Stereo,
            3 => AudioCodingMode::ThreeChannelFront,
            4 => AudioCodingMode::TwoChannelPlusTwoSurround,
            5 | 6 => AudioCodingMode::FiveChannel,
            _ => AudioCodingMode::FiveChannel,
        };

        Self {
            sample_rate,
            channels,
            channel_layout,
            lfe: channels >= 6,
            bitrate: 448000,
            dialnorm: -31,
            bsmod: BitstreamMode::CompleteMain,
            num_blocks: 6,
            coupling: true,
            channel_extension: false,
            compression_profile: CompressionProfile::None,
        }
    }

    /// Create configuration for 5.1 surround.
    pub fn surround_5_1(sample_rate: u32, bitrate: u32) -> Self {
        Self {
            sample_rate,
            channels: 6,
            channel_layout: AudioCodingMode::FiveChannel,
            lfe: true,
            bitrate,
            dialnorm: -31,
            bsmod: BitstreamMode::CompleteMain,
            num_blocks: 6,
            coupling: true,
            channel_extension: false,
            compression_profile: CompressionProfile::None,
        }
    }

    /// Create configuration for 7.1 surround (with channel extension).
    pub fn surround_7_1(sample_rate: u32, bitrate: u32) -> Self {
        Self {
            sample_rate,
            channels: 8,
            channel_layout: AudioCodingMode::FiveChannel,
            lfe: true,
            bitrate,
            dialnorm: -31,
            bsmod: BitstreamMode::CompleteMain,
            num_blocks: 6,
            coupling: true,
            channel_extension: true,
            compression_profile: CompressionProfile::None,
        }
    }

    /// Set the dialogue normalization value.
    pub fn with_dialnorm(mut self, dialnorm: i8) -> Self {
        self.dialnorm = dialnorm.clamp(-31, -1);
        self
    }

    /// Set the number of audio blocks.
    pub fn with_num_blocks(mut self, num_blocks: u8) -> Self {
        self.num_blocks = match num_blocks {
            1 => 1,
            2 => 2,
            3 => 3,
            _ => 6,
        };
        self
    }

    /// Set the compression profile.
    pub fn with_compression(mut self, profile: CompressionProfile) -> Self {
        self.compression_profile = profile;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // E-AC-3 supports more sample rates
        let valid_rates = [48000, 44100, 32000, 24000, 22050, 16000];
        if !valid_rates.contains(&self.sample_rate) {
            return Err(Ac3Error::EncodingError(format!(
                "Invalid sample rate: {}",
                self.sample_rate
            )));
        }

        // Validate num_blocks
        if ![1, 2, 3, 6].contains(&self.num_blocks) {
            return Err(Ac3Error::EncodingError(format!(
                "Invalid num_blocks: {} (must be 1, 2, 3, or 6)",
                self.num_blocks
            )));
        }

        Ok(())
    }
}

impl Default for Eac3EncoderConfig {
    fn default() -> Self {
        Self::new(48000, 2)
    }
}

/// Audio production information.
#[derive(Debug, Clone)]
pub struct AudioProductionInfo {
    /// Mixing level (0-31).
    pub mixing_level: u8,
    /// Room type.
    pub room_type: RoomType,
}

/// Compression profile for dynamic range control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionProfile {
    /// No compression.
    None,
    /// Film standard.
    FilmStandard,
    /// Film light.
    FilmLight,
    /// Music standard.
    MusicStandard,
    /// Music light.
    MusicLight,
    /// Speech.
    Speech,
}

/// AC-3 encoder.
///
/// Encodes PCM audio to AC-3 (Dolby Digital) format.
///
/// # Important
///
/// Encoding requires the `ffi-ffmpeg` feature and appropriate Dolby licensing
/// for commercial use.
#[derive(Debug)]
pub struct Ac3Encoder {
    /// Encoder configuration.
    config: Ac3EncoderConfig,
    /// Total frames encoded.
    frames_encoded: u64,
    /// Total samples encoded.
    samples_encoded: u64,
    /// Initialized flag.
    initialized: bool,
}

impl Ac3Encoder {
    /// Create a new AC-3 encoder.
    pub fn new(config: Ac3EncoderConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            frames_encoded: 0,
            samples_encoded: 0,
            initialized: false,
        })
    }

    /// Initialize the encoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn initialize(&mut self) -> Result<()> {
        // FFI initialization would go here
        Err(Ac3Error::FfiNotAvailable)
    }

    /// Initialize the encoder (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn initialize(&mut self) -> Result<()> {
        Err(Ac3Error::FfiNotAvailable)
    }

    /// Encode a frame of PCM samples.
    ///
    /// Input samples should be interleaved floats in range [-1.0, 1.0].
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn encode_frame(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        // FFI encoding would go here
        Err(Ac3Error::FfiNotAvailable)
    }

    /// Encode a frame (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn encode_frame(&mut self, _samples: &[f32]) -> Result<Vec<u8>> {
        Err(Ac3Error::FfiNotAvailable)
    }

    /// Flush remaining samples.
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        Ok(Vec::new())
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &Ac3EncoderConfig {
        &self.config
    }

    /// Get total frames encoded.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }

    /// Get total samples encoded.
    pub fn samples_encoded(&self) -> u64 {
        self.samples_encoded
    }

    /// Check if encoding is available.
    pub fn is_encoding_available(&self) -> bool {
        cfg!(feature = "ffi-ffmpeg")
    }
}

/// E-AC-3 encoder.
///
/// Encodes PCM audio to E-AC-3 (Dolby Digital Plus) format.
#[derive(Debug)]
pub struct Eac3Encoder {
    /// Encoder configuration.
    config: Eac3EncoderConfig,
    /// Total frames encoded.
    frames_encoded: u64,
    /// Total samples encoded.
    samples_encoded: u64,
    /// Initialized flag.
    initialized: bool,
}

impl Eac3Encoder {
    /// Create a new E-AC-3 encoder.
    pub fn new(config: Eac3EncoderConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            frames_encoded: 0,
            samples_encoded: 0,
            initialized: false,
        })
    }

    /// Initialize the encoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn initialize(&mut self) -> Result<()> {
        // FFI initialization would go here
        Err(Ac3Error::FfiNotAvailable)
    }

    /// Initialize the encoder (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn initialize(&mut self) -> Result<()> {
        Err(Ac3Error::FfiNotAvailable)
    }

    /// Encode a frame of PCM samples.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn encode_frame(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        // FFI encoding would go here
        Err(Ac3Error::FfiNotAvailable)
    }

    /// Encode a frame (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn encode_frame(&mut self, _samples: &[f32]) -> Result<Vec<u8>> {
        Err(Ac3Error::FfiNotAvailable)
    }

    /// Flush remaining samples.
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        Ok(Vec::new())
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &Eac3EncoderConfig {
        &self.config
    }

    /// Get total frames encoded.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }

    /// Get total samples encoded.
    pub fn samples_encoded(&self) -> u64 {
        self.samples_encoded
    }

    /// Check if encoding is available.
    pub fn is_encoding_available(&self) -> bool {
        cfg!(feature = "ffi-ffmpeg")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ac3_encoder_config_default() {
        let config = Ac3EncoderConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bitrate, 384000);
    }

    #[test]
    fn test_ac3_encoder_config_surround() {
        let config = Ac3EncoderConfig::surround_5_1(48000, 640000);
        assert_eq!(config.channels, 6);
        assert!(config.lfe);
        assert_eq!(config.channel_layout, AudioCodingMode::FiveChannel);
    }

    #[test]
    fn test_ac3_encoder_config_validate() {
        let config = Ac3EncoderConfig::default();
        assert!(config.validate().is_ok());

        let invalid_config = Ac3EncoderConfig {
            sample_rate: 96000, // Invalid
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_eac3_encoder_config_default() {
        let config = Eac3EncoderConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.num_blocks, 6);
    }

    #[test]
    fn test_eac3_encoder_config_7_1() {
        let config = Eac3EncoderConfig::surround_7_1(48000, 1024000);
        assert_eq!(config.channels, 8);
        assert!(config.channel_extension);
    }

    #[test]
    fn test_ac3_encoder_new() {
        let config = Ac3EncoderConfig::default();
        let encoder = Ac3Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_eac3_encoder_new() {
        let config = Eac3EncoderConfig::default();
        let encoder = Eac3Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_not_available() {
        let config = Ac3EncoderConfig::default();
        let encoder = Ac3Encoder::new(config).unwrap();

        #[cfg(not(feature = "ffi-ffmpeg"))]
        assert!(!encoder.is_encoding_available());
    }

    #[test]
    fn test_dialnorm_clamping() {
        let config = Ac3EncoderConfig::default()
            .with_dialnorm(-50); // Out of range

        assert_eq!(config.dialnorm, -31);

        let config2 = Ac3EncoderConfig::default()
            .with_dialnorm(5); // Out of range

        assert_eq!(config2.dialnorm, -1);
    }

    #[test]
    fn test_compression_profiles() {
        let config = Ac3EncoderConfig::default()
            .with_compression(CompressionProfile::FilmStandard);

        assert_eq!(config.compression_profile, CompressionProfile::FilmStandard);
    }
}
