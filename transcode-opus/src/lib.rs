//! # Transcode Opus
//!
//! Opus audio codec implementation for the Transcode project.
//!
//! Opus is a versatile audio codec that combines two coding modes:
//! - **SILK**: Optimized for speech (6-40 kbps)
//! - **CELT**: Optimized for music (12-256 kbps)
//! - **Hybrid**: Combines both for high-quality voice (16-128 kbps)
//!
//! ## Features
//!
//! - Variable bitrate (VBR) support
//! - Sample rates: 8, 12, 16, 24, 48 kHz
//! - Mono and stereo
//! - Frame sizes: 2.5, 5, 10, 20, 40, 60 ms
//! - Packet loss concealment (PLC)
//! - Low latency (down to 2.5 ms)
//!
//! ## Usage
//!
//! ### Decoding
//!
//! ```ignore
//! use transcode_opus::OpusDecoder;
//!
//! let mut decoder = OpusDecoder::new(48000, 2)?;
//! let samples = decoder.decode_packet(&opus_packet)?;
//! ```
//!
//! ### Encoding
//!
//! ```ignore
//! use transcode_opus::{OpusEncoder, OpusEncoderConfig};
//!
//! let config = OpusEncoderConfig::for_music(48000, 2);
//! let mut encoder = OpusEncoder::new(config)?;
//! let packet = encoder.encode(&samples)?;
//! ```
//!
//! ## Codec Modes
//!
//! The encoder automatically selects the best mode based on:
//! - Signal content (voice vs. music)
//! - Target bitrate
//! - Application type (VoIP, Audio, Low-delay)
//!
//! You can also hint the mode using `SignalType::Voice` or `SignalType::Music`.

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod celt;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod range_coder;
pub mod silk;

pub use decoder::OpusDecoder;
pub use encoder::{OpusEncoder, OpusEncoderConfig};
pub use error::{OpusError, Result};

use transcode_core::error::{CodecError, Error as CoreError};
use transcode_core::packet::Packet;
use transcode_core::sample::Sample;

/// Opus supported sample rates.
pub const SAMPLE_RATES: [u32; 5] = [8000, 12000, 16000, 24000, 48000];

/// Opus supported frame sizes in milliseconds.
pub const FRAME_SIZES_MS: [f32; 6] = [2.5, 5.0, 10.0, 20.0, 40.0, 60.0];

/// Maximum Opus packet size in bytes.
pub const MAX_PACKET_SIZE: usize = 1275 * 3 + 7;

/// Maximum frame size in samples at 48 kHz.
pub const MAX_FRAME_SIZE: usize = 2880; // 60ms at 48kHz

/// Opus operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpusMode {
    /// SILK mode for speech.
    Silk,
    /// CELT mode for music.
    #[default]
    Celt,
    /// Hybrid mode (SILK + CELT).
    Hybrid,
}

impl std::fmt::Display for OpusMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Silk => write!(f, "SILK"),
            Self::Celt => write!(f, "CELT"),
            Self::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Opus bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Bandwidth {
    /// Narrowband (4 kHz).
    Narrowband = 0,
    /// Mediumband (6 kHz).
    Mediumband = 1,
    /// Wideband (8 kHz).
    Wideband = 2,
    /// Super-wideband (12 kHz).
    SuperWideband = 3,
    /// Fullband (20 kHz).
    #[default]
    Fullband = 4,
}

impl Bandwidth {
    /// Get the maximum frequency for this bandwidth.
    pub fn max_freq(&self) -> u32 {
        match self {
            Self::Narrowband => 4000,
            Self::Mediumband => 6000,
            Self::Wideband => 8000,
            Self::SuperWideband => 12000,
            Self::Fullband => 20000,
        }
    }

    /// Get the name of this bandwidth.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Narrowband => "Narrowband",
            Self::Mediumband => "Mediumband",
            Self::Wideband => "Wideband",
            Self::SuperWideband => "Super-wideband",
            Self::Fullband => "Fullband",
        }
    }
}

impl std::fmt::Display for Bandwidth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Frame size configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameSize {
    /// 2.5 ms frame.
    Ms2_5,
    /// 5 ms frame.
    Ms5,
    /// 10 ms frame.
    Ms10,
    /// 20 ms frame (default).
    #[default]
    Ms20,
    /// 40 ms frame.
    Ms40,
    /// 60 ms frame.
    Ms60,
}

impl FrameSize {
    /// Get the frame size in samples at 48 kHz.
    pub fn samples_48k(&self) -> usize {
        match self {
            Self::Ms2_5 => 120,
            Self::Ms5 => 240,
            Self::Ms10 => 480,
            Self::Ms20 => 960,
            Self::Ms40 => 1920,
            Self::Ms60 => 2880,
        }
    }

    /// Get the frame size in milliseconds.
    pub fn milliseconds(&self) -> f32 {
        match self {
            Self::Ms2_5 => 2.5,
            Self::Ms5 => 5.0,
            Self::Ms10 => 10.0,
            Self::Ms20 => 20.0,
            Self::Ms40 => 40.0,
            Self::Ms60 => 60.0,
        }
    }

    /// Get the frame size in samples at a given sample rate.
    pub fn samples_at_rate(&self, sample_rate: u32) -> usize {
        self.samples_48k() * sample_rate as usize / 48000
    }
}

impl std::fmt::Display for FrameSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ms", self.milliseconds())
    }
}

/// Application type hint for encoder optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Application {
    /// Voice over IP: optimized for voice at low bitrates.
    Voip,
    /// Generic audio: best for music and mixed content.
    #[default]
    Audio,
    /// Low delay: minimum latency mode.
    LowDelay,
}

impl std::fmt::Display for Application {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Voip => write!(f, "VoIP"),
            Self::Audio => write!(f, "Audio"),
            Self::LowDelay => write!(f, "Low-delay"),
        }
    }
}

/// Signal type hint for encoder optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SignalType {
    /// Automatic detection.
    #[default]
    Auto,
    /// Voice signal.
    Voice,
    /// Music signal.
    Music,
}

/// VBR (Variable Bit Rate) mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VbrMode {
    /// Variable bit rate (default).
    #[default]
    Vbr,
    /// Constrained VBR.
    ConstrainedVbr,
    /// Constant bit rate.
    Cbr,
}

/// Channel mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelMode {
    /// Mono (1 channel).
    Mono,
    /// Stereo (2 channels).
    #[default]
    Stereo,
}

impl ChannelMode {
    /// Get the number of channels.
    pub fn channels(&self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

/// Opus codec configuration.
#[derive(Debug, Clone)]
pub struct OpusConfig {
    /// Sample rate.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Target bitrate.
    pub bitrate: u32,
    /// Application type.
    pub application: Application,
    /// Frame size.
    pub frame_size: FrameSize,
    /// Maximum bandwidth.
    pub max_bandwidth: Bandwidth,
    /// VBR mode.
    pub vbr_mode: VbrMode,
    /// Complexity (0-10).
    pub complexity: u8,
    /// Enable DTX.
    pub dtx: bool,
    /// Enable FEC.
    pub fec: bool,
}

impl Default for OpusConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            bitrate: 64000,
            application: Application::Audio,
            frame_size: FrameSize::Ms20,
            max_bandwidth: Bandwidth::Fullband,
            vbr_mode: VbrMode::Vbr,
            complexity: 10,
            dtx: false,
            fec: false,
        }
    }
}

/// Opus packet info (parsed from TOC byte).
#[derive(Debug, Clone)]
pub struct OpusPacketInfo {
    /// Opus mode.
    pub mode: OpusMode,
    /// Bandwidth.
    pub bandwidth: Bandwidth,
    /// Number of frames in packet.
    pub frame_count: usize,
    /// Frame size in samples at 48 kHz.
    pub frame_size: usize,
    /// Stereo flag.
    pub stereo: bool,
}

impl OpusPacketInfo {
    /// Parse packet info from an Opus packet.
    pub fn parse(packet: &[u8]) -> Option<Self> {
        if packet.is_empty() {
            return None;
        }

        let toc = packet[0];
        let config = (toc >> 3) & 0x1F;
        let stereo = (toc >> 2) & 1 != 0;
        let frame_count_code = toc & 0x03;

        let mode = match config {
            0..=11 => OpusMode::Silk,
            12..=15 => OpusMode::Hybrid,
            16..=31 => OpusMode::Celt,
            _ => OpusMode::Celt,
        };

        let bandwidth = match config {
            0..=3 => Bandwidth::Narrowband,
            4..=7 => Bandwidth::Mediumband,
            8..=11 => Bandwidth::Wideband,
            12..=13 => Bandwidth::SuperWideband,
            14..=15 => Bandwidth::Fullband,
            16..=19 => Bandwidth::Narrowband,
            20..=23 => Bandwidth::Wideband,
            24..=27 => Bandwidth::SuperWideband,
            28..=31 => Bandwidth::Fullband,
            _ => Bandwidth::Fullband,
        };

        let frame_size_code = match config {
            0..=11 => config % 4,
            12..=15 => (config - 12) % 2 + 2,
            16..=31 => (config - 16) % 4,
            _ => 3,
        };

        let frame_size = match frame_size_code {
            0 => 120,
            1 => 240,
            2 => 480,
            3 => 960,
            _ => 960,
        };

        let frame_count = match frame_count_code {
            0 => 1,
            1 | 2 => 2,
            3 => {
                if packet.len() > 1 {
                    (packet[1] & 0x3F) as usize
                } else {
                    0
                }
            }
            _ => 1,
        };

        Some(Self {
            mode,
            bandwidth,
            frame_count,
            frame_size,
            stereo,
        })
    }

    /// Get the total duration in samples at 48 kHz.
    pub fn duration_samples(&self) -> usize {
        self.frame_size * self.frame_count
    }

    /// Get the duration in milliseconds.
    pub fn duration_ms(&self) -> f32 {
        self.duration_samples() as f32 / 48.0
    }
}

/// Trait for Opus audio decoder (for integration with transcode-codecs).
pub trait OpusAudioDecoder: Send {
    /// Decode a packet.
    fn decode_opus(&mut self, packet: &Packet<'_>) -> transcode_core::Result<Vec<Sample>>;

    /// Perform packet loss concealment.
    fn conceal_loss(&mut self) -> transcode_core::Result<Sample>;

    /// Reset the decoder.
    fn reset(&mut self);

    /// Get sample rate.
    fn sample_rate(&self) -> u32;

    /// Get number of channels.
    fn channels(&self) -> u8;
}

impl OpusAudioDecoder for OpusDecoder {
    fn decode_opus(&mut self, packet: &Packet<'_>) -> transcode_core::Result<Vec<Sample>> {
        let buffer = self.decode_packet(packet.data()).map_err(|e| {
            CoreError::Codec(CodecError::Other(e.to_string()))
        })?;

        let sample = Sample::from_buffer(buffer);
        Ok(vec![sample])
    }

    fn conceal_loss(&mut self) -> transcode_core::Result<Sample> {
        let buffer = self.conceal_packet_loss().map_err(|e| {
            CoreError::Codec(CodecError::Other(e.to_string()))
        })?;

        Ok(Sample::from_buffer(buffer))
    }

    fn reset(&mut self) {
        OpusDecoder::reset(self);
    }

    fn sample_rate(&self) -> u32 {
        OpusDecoder::sample_rate(self)
    }

    fn channels(&self) -> u8 {
        OpusDecoder::channels(self)
    }
}

/// Trait for Opus audio encoder (for integration with transcode-codecs).
pub trait OpusAudioEncoder: Send {
    /// Encode samples.
    fn encode_opus(&mut self, sample: &Sample) -> transcode_core::Result<Vec<Packet<'static>>>;

    /// Reset the encoder.
    fn reset(&mut self);

    /// Get sample rate.
    fn sample_rate(&self) -> u32;

    /// Get number of channels.
    fn channels(&self) -> u8;
}

impl OpusAudioEncoder for OpusEncoder {
    fn encode_opus(&mut self, sample: &Sample) -> transcode_core::Result<Vec<Packet<'static>>> {
        let data = self.encode(sample.buffer()).map_err(|e| {
            CoreError::Codec(CodecError::Other(e.to_string()))
        })?;

        let packet = Packet::new(data);
        Ok(vec![packet.into_owned()])
    }

    fn reset(&mut self) {
        OpusEncoder::reset(self);
    }

    fn sample_rate(&self) -> u32 {
        OpusEncoder::sample_rate(self)
    }

    fn channels(&self) -> u8 {
        OpusEncoder::channels(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_rates() {
        assert!(SAMPLE_RATES.contains(&48000));
        assert!(SAMPLE_RATES.contains(&16000));
        assert!(!SAMPLE_RATES.contains(&44100));
    }

    #[test]
    fn test_frame_sizes() {
        assert_eq!(FrameSize::Ms20.samples_48k(), 960);
        assert_eq!(FrameSize::Ms10.samples_48k(), 480);
        assert_eq!(FrameSize::Ms2_5.samples_48k(), 120);
    }

    #[test]
    fn test_frame_size_at_rate() {
        assert_eq!(FrameSize::Ms20.samples_at_rate(48000), 960);
        assert_eq!(FrameSize::Ms20.samples_at_rate(16000), 320);
    }

    #[test]
    fn test_bandwidth() {
        assert_eq!(Bandwidth::Fullband.max_freq(), 20000);
        assert_eq!(Bandwidth::Narrowband.max_freq(), 4000);
    }

    #[test]
    fn test_opus_mode_display() {
        assert_eq!(format!("{}", OpusMode::Silk), "SILK");
        assert_eq!(format!("{}", OpusMode::Celt), "CELT");
        assert_eq!(format!("{}", OpusMode::Hybrid), "Hybrid");
    }

    #[test]
    fn test_packet_info_parsing() {
        // CELT fullband stereo 20ms
        let packet = [0b11111100u8, 0, 0, 0];
        let info = OpusPacketInfo::parse(&packet).unwrap();
        assert_eq!(info.mode, OpusMode::Celt);
        assert_eq!(info.bandwidth, Bandwidth::Fullband);
        assert!(info.stereo);
    }

    #[test]
    fn test_config_default() {
        let config = OpusConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bitrate, 64000);
    }

    #[test]
    fn test_decoder_and_encoder() {
        // Create decoder
        let decoder_result = OpusDecoder::new(48000, 2);
        assert!(decoder_result.is_ok());

        // Create encoder
        let config = OpusEncoderConfig::default();
        let encoder_result = OpusEncoder::new(config);
        assert!(encoder_result.is_ok());
    }

    #[test]
    fn test_encoder_config() {
        // Create encoder with different configs
        let config_voice = OpusEncoderConfig::for_voice(48000, 1);
        assert_eq!(config_voice.application, Application::Voip);
        assert_eq!(config_voice.signal_type, SignalType::Voice);

        let config_music = OpusEncoderConfig::for_music(48000, 2);
        assert_eq!(config_music.application, Application::Audio);
        assert_eq!(config_music.signal_type, SignalType::Music);

        // Verify encoder creation
        let encoder_voice = OpusEncoder::new(config_voice);
        assert!(encoder_voice.is_ok());

        let encoder_music = OpusEncoder::new(config_music);
        assert!(encoder_music.is_ok());
    }

    #[test]
    fn test_packet_loss_concealment() {
        let mut decoder = OpusDecoder::new(48000, 2).unwrap();

        // Simulate packet loss
        let plc_result = decoder.conceal_packet_loss();
        assert!(plc_result.is_ok());

        let plc_buffer = plc_result.unwrap();
        assert!(plc_buffer.num_samples > 0);
    }
}
