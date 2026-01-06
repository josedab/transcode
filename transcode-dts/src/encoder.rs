//! DTS audio encoder.
//!
//! Provides audio encoding to DTS format.
//!
//! ## Usage
//!
//! Full encoding requires the `ffi-ffmpeg` feature. Without it, only header
//! generation is available.
//!
//! ## Note
//!
//! TrueHD encoding is not typically available as it's a Dolby proprietary format.
//! Only DTS encoding is provided here.

use crate::types::*;
use crate::{DtsError, Result};

/// DTS audio encoder.
///
/// Encodes PCM audio to DTS core format.
///
/// # Example
///
/// ```rust,ignore
/// use transcode_dts::{DtsEncoder, DtsEncoderConfig};
///
/// let config = DtsEncoderConfig::new(48000, 6);
/// let mut encoder = DtsEncoder::new(config)?;
/// let encoded = encoder.encode_frame(&pcm_samples)?;
/// ```
#[derive(Debug)]
pub struct DtsEncoder {
    /// Encoder configuration.
    config: DtsEncoderConfig,
    /// Current frame number.
    frame_number: u64,
    /// Frames encoded.
    frames_encoded: u64,
    /// Bytes output.
    bytes_output: u64,
}

/// DTS encoder configuration.
#[derive(Debug, Clone)]
pub struct DtsEncoderConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bit rate in kbps.
    pub bitrate_kbps: u32,
    /// Audio mode.
    pub audio_mode: AudioMode,
    /// LFE channel present.
    pub lfe: bool,
    /// Target profile.
    pub profile: DtsProfile,
    /// Enable dialog normalization.
    pub dialog_norm: bool,
    /// Dialog normalization value (dB).
    pub dialog_norm_value: i8,
    /// Enable dynamic range compression.
    pub drc: bool,
}

/// DTS encoder profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtsProfile {
    /// DTS Digital Surround (core).
    Core,
    /// DTS-HD High Resolution.
    HdHighResolution,
    /// DTS-HD Master Audio.
    HdMasterAudio,
    /// DTS Express (low bitrate).
    Express,
}

impl DtsEncoderConfig {
    /// Create a new encoder configuration.
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        let audio_mode = match channels {
            1 => AudioMode::Mono,
            2 => AudioMode::Stereo,
            3 => AudioMode::Front3_0,
            4 => AudioMode::Front2_Rear2,
            5 => AudioMode::Front3_Rear2,
            6 => AudioMode::Front3_Rear2,
            _ => AudioMode::Front3_Rear2,
        };

        let lfe = channels >= 6;

        Self {
            sample_rate,
            channels,
            bitrate_kbps: 768,
            audio_mode,
            lfe,
            profile: DtsProfile::Core,
            dialog_norm: true,
            dialog_norm_value: -31,
            drc: false,
        }
    }

    /// Create configuration for 5.1 surround.
    pub fn surround_5_1(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            channels: 6,
            bitrate_kbps: 1509,
            audio_mode: AudioMode::Front3_Rear2,
            lfe: true,
            profile: DtsProfile::Core,
            dialog_norm: true,
            dialog_norm_value: -31,
            drc: false,
        }
    }

    /// Create configuration for 7.1 surround (DTS-HD).
    pub fn surround_7_1(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            channels: 8,
            bitrate_kbps: 3000,
            audio_mode: AudioMode::Front3_Rear2_Height2,
            lfe: true,
            profile: DtsProfile::HdMasterAudio,
            dialog_norm: true,
            dialog_norm_value: -31,
            drc: false,
        }
    }

    /// Create configuration for stereo.
    pub fn stereo(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            channels: 2,
            bitrate_kbps: 384,
            audio_mode: AudioMode::Stereo,
            lfe: false,
            profile: DtsProfile::Core,
            dialog_norm: true,
            dialog_norm_value: -31,
            drc: false,
        }
    }

    /// Set the bitrate.
    pub fn with_bitrate(mut self, bitrate_kbps: u32) -> Self {
        self.bitrate_kbps = bitrate_kbps;
        self
    }

    /// Set the profile.
    pub fn with_profile(mut self, profile: DtsProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Enable or disable LFE channel.
    pub fn with_lfe(mut self, lfe: bool) -> Self {
        self.lfe = lfe;
        self
    }

    /// Set dialog normalization.
    pub fn with_dialog_norm(mut self, enabled: bool, value: i8) -> Self {
        self.dialog_norm = enabled;
        self.dialog_norm_value = value;
        self
    }

    /// Enable or disable dynamic range compression.
    pub fn with_drc(mut self, drc: bool) -> Self {
        self.drc = drc;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Check sample rate
        let valid_rates = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000, 192000];
        if !valid_rates.contains(&self.sample_rate) {
            return Err(DtsError::UnsupportedSampleRate(self.sample_rate));
        }

        // Check channel count
        if self.channels == 0 || self.channels > 8 {
            return Err(DtsError::UnsupportedChannelConfig(self.channels));
        }

        // Check bitrate for profile
        match self.profile {
            DtsProfile::Core => {
                if self.bitrate_kbps > 1509 {
                    return Err(DtsError::EncodingError(
                        "DTS Core bitrate cannot exceed 1509 kbps".into(),
                    ));
                }
            }
            DtsProfile::Express => {
                if self.bitrate_kbps > 256 {
                    return Err(DtsError::EncodingError(
                        "DTS Express bitrate cannot exceed 256 kbps".into(),
                    ));
                }
            }
            DtsProfile::HdHighResolution | DtsProfile::HdMasterAudio => {
                // HD profiles have higher limits
            }
        }

        // Check dialog normalization value
        if self.dialog_norm && (self.dialog_norm_value < -31 || self.dialog_norm_value > 0) {
            return Err(DtsError::EncodingError(
                "Dialog normalization must be between -31 and 0 dB".into(),
            ));
        }

        Ok(())
    }

    /// Get the frame size in samples.
    pub fn frame_size(&self) -> usize {
        match self.profile {
            DtsProfile::Core | DtsProfile::Express => 512,
            DtsProfile::HdHighResolution | DtsProfile::HdMasterAudio => 4096,
        }
    }
}

impl Default for DtsEncoderConfig {
    fn default() -> Self {
        Self::new(48000, 6)
    }
}

impl DtsEncoder {
    /// Create a new DTS encoder.
    pub fn new(config: DtsEncoderConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            frame_number: 0,
            frames_encoded: 0,
            bytes_output: 0,
        })
    }

    /// Encode PCM samples to DTS.
    ///
    /// Returns encoded DTS data or an error if encoding fails.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn encode_frame(&mut self, samples: &[f32]) -> Result<EncodedDtsPacket> {
        // FFI implementation would go here
        Err(DtsError::FfiNotAvailable)
    }

    /// Encode PCM samples to DTS (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn encode_frame(&mut self, _samples: &[f32]) -> Result<EncodedDtsPacket> {
        // Without FFI, we can only generate a placeholder header
        let mut data = Vec::new();

        // DTS sync word (big endian 14-bit format)
        data.extend_from_slice(&[0x7F, 0xFE, 0x80, 0x01]);

        // Frame header (simplified placeholder)
        // In reality, this would need proper encoding
        let header = self.build_frame_header();
        data.extend_from_slice(&header);

        self.frame_number += 1;
        self.frames_encoded += 1;
        self.bytes_output += data.len() as u64;

        Ok(EncodedDtsPacket {
            data,
            samples_per_channel: self.config.frame_size(),
            pts: Some(self.frame_number - 1),
            duration_samples: self.config.frame_size(),
        })
    }

    /// Build frame header bytes.
    fn build_frame_header(&self) -> Vec<u8> {
        let mut header = Vec::with_capacity(12);

        // FTYPE (1) + SHORT (5) + CPF (1) + NBLKS (7)
        // NBLKS = (samples / 32) - 1 = 15 for 512 samples
        let nblks = (self.config.frame_size() / 32) - 1;
        header.push(0x80 | ((nblks >> 2) as u8));

        // NBLKS (cont) + FSIZE (14 bits)
        let fsize = 100; // Placeholder frame size
        header.push(((nblks as u8 & 0x03) << 6) | ((fsize >> 8) as u8));
        header.push(fsize as u8);

        // AMODE (6) + SFREQ (4)
        let amode = self.config.audio_mode as u8;
        let sfreq = self.sample_rate_code();
        header.push((amode << 2) | (sfreq >> 2));

        // SFREQ (cont) + RATE (5)
        let rate = self.bitrate_code();
        header.push(((sfreq & 0x03) << 6) | (rate << 1));

        // Additional flags
        header.push(0x00);
        header.push(0x00);

        header
    }

    /// Get sample rate code for header.
    fn sample_rate_code(&self) -> u8 {
        match self.config.sample_rate {
            8000 => 1,
            16000 => 2,
            32000 => 3,
            11025 => 6,
            22050 => 7,
            44100 => 8,
            12000 => 11,
            24000 => 12,
            48000 => 13,
            96000 => 14,
            192000 => 15,
            _ => 13, // Default to 48kHz
        }
    }

    /// Get bitrate code for header.
    fn bitrate_code(&self) -> u8 {
        // Find closest bitrate in table
        const RATES: [(u32, u8); 24] = [
            (32, 0), (56, 1), (64, 2), (96, 3), (112, 4), (128, 5),
            (192, 6), (224, 7), (256, 8), (320, 9), (384, 10), (448, 11),
            (512, 12), (576, 13), (640, 14), (768, 15), (896, 16), (1024, 17),
            (1152, 18), (1280, 19), (1344, 20), (1408, 21), (1411, 22), (1472, 23),
        ];

        for (rate, code) in RATES.iter() {
            if self.config.bitrate_kbps <= *rate {
                return *code;
            }
        }
        24 // 1536 kbps
    }

    /// Flush the encoder.
    pub fn flush(&mut self) -> Result<Option<EncodedDtsPacket>> {
        // Without FFI, nothing to flush
        Ok(None)
    }

    /// Reset the encoder state.
    pub fn reset(&mut self) {
        self.frame_number = 0;
        self.frames_encoded = 0;
        self.bytes_output = 0;
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &DtsEncoderConfig {
        &self.config
    }

    /// Get total frames encoded.
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
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
    pub fn is_encoding_available(&self) -> bool {
        cfg!(feature = "ffi-ffmpeg")
    }
}

/// Encoded DTS packet.
#[derive(Debug, Clone)]
pub struct EncodedDtsPacket {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Number of samples per channel in this packet.
    pub samples_per_channel: usize,
    /// Presentation timestamp (in samples).
    pub pts: Option<u64>,
    /// Duration in samples.
    pub duration_samples: usize,
}

impl EncodedDtsPacket {
    /// Get the packet size in bytes.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_new() {
        let config = DtsEncoderConfig::new(48000, 6);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 6);
        assert!(config.lfe);
    }

    #[test]
    fn test_encoder_config_stereo() {
        let config = DtsEncoderConfig::stereo(48000);
        assert_eq!(config.channels, 2);
        assert!(!config.lfe);
        assert_eq!(config.audio_mode, AudioMode::Stereo);
    }

    #[test]
    fn test_encoder_config_5_1() {
        let config = DtsEncoderConfig::surround_5_1(48000);
        assert_eq!(config.channels, 6);
        assert!(config.lfe);
        assert_eq!(config.bitrate_kbps, 1509);
    }

    #[test]
    fn test_encoder_config_7_1() {
        let config = DtsEncoderConfig::surround_7_1(48000);
        assert_eq!(config.channels, 8);
        assert_eq!(config.profile, DtsProfile::HdMasterAudio);
    }

    #[test]
    fn test_encoder_config_validation() {
        let config = DtsEncoderConfig::new(48000, 6);
        assert!(config.validate().is_ok());

        // Invalid sample rate
        let mut config = DtsEncoderConfig::new(48000, 6);
        config.sample_rate = 12345;
        assert!(config.validate().is_err());

        // Invalid channel count
        let mut config = DtsEncoderConfig::new(48000, 6);
        config.channels = 0;
        assert!(config.validate().is_err());

        // Too high bitrate for core
        let mut config = DtsEncoderConfig::new(48000, 6);
        config.bitrate_kbps = 2000;
        config.profile = DtsProfile::Core;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_encoder_new() {
        let config = DtsEncoderConfig::new(48000, 6);
        let encoder = DtsEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_reset() {
        let config = DtsEncoderConfig::new(48000, 6);
        let mut encoder = DtsEncoder::new(config).unwrap();
        encoder.reset();
        assert_eq!(encoder.frames_encoded(), 0);
        assert_eq!(encoder.bytes_output(), 0);
    }

    #[test]
    fn test_is_encoding_available() {
        let config = DtsEncoderConfig::new(48000, 6);
        let encoder = DtsEncoder::new(config).unwrap();
        #[cfg(not(feature = "ffi-ffmpeg"))]
        assert!(!encoder.is_encoding_available());
    }

    #[test]
    fn test_frame_size() {
        let config = DtsEncoderConfig::new(48000, 6);
        assert_eq!(config.frame_size(), 512);

        let config = DtsEncoderConfig::new(48000, 6).with_profile(DtsProfile::HdMasterAudio);
        assert_eq!(config.frame_size(), 4096);
    }

    #[test]
    fn test_encode_frame_without_ffi() {
        let config = DtsEncoderConfig::new(48000, 6);
        let mut encoder = DtsEncoder::new(config).unwrap();

        let samples = vec![0.0f32; 512 * 6];
        let packet = encoder.encode_frame(&samples);

        assert!(packet.is_ok());
        let packet = packet.unwrap();
        assert!(!packet.data.is_empty());
        assert_eq!(packet.samples_per_channel, 512);
    }

    #[test]
    fn test_dts_profile() {
        assert_eq!(format!("{:?}", DtsProfile::Core), "Core");
        assert_eq!(format!("{:?}", DtsProfile::HdMasterAudio), "HdMasterAudio");
    }
}
