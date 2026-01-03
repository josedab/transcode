//! AAC (Advanced Audio Coding) codec implementation.
//!
//! Supports AAC-LC (Low Complexity) and HE-AAC profiles.

// Allow common patterns in signal processing codec implementation
#![allow(
    dead_code,
    unused_must_use,
    unused_variables,
    clippy::needless_range_loop,
    clippy::manual_find,
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::cast_abs_to_unsigned
)]

mod adts;
mod decoder;
mod encoder;
mod huffman;
mod mdct;
mod psy;
mod tables;
mod tns;

pub use adts::*;
pub use decoder::AacDecoder;
pub use encoder::{AacEncoder, AacEncoderConfig};

/// AAC profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AacProfile {
    /// Main profile.
    Main = 0,
    /// Low Complexity profile.
    Lc = 1,
    /// Scalable Sampling Rate profile.
    Ssr = 2,
    /// Long Term Prediction profile.
    Ltp = 3,
    /// High Efficiency AAC (SBR).
    HeAac = 4,
    /// High Efficiency AAC v2 (SBR + PS).
    HeAacV2 = 5,
}

impl AacProfile {
    /// Get profile from object type.
    pub fn from_object_type(object_type: u8) -> Option<Self> {
        match object_type {
            1 => Some(AacProfile::Main),
            2 => Some(AacProfile::Lc),
            3 => Some(AacProfile::Ssr),
            4 => Some(AacProfile::Ltp),
            5 => Some(AacProfile::HeAac),
            29 => Some(AacProfile::HeAacV2),
            _ => None,
        }
    }

    /// Get object type from profile.
    pub fn object_type(&self) -> u8 {
        match self {
            AacProfile::Main => 1,
            AacProfile::Lc => 2,
            AacProfile::Ssr => 3,
            AacProfile::Ltp => 4,
            AacProfile::HeAac => 5,
            AacProfile::HeAacV2 => 29,
        }
    }
}

/// AAC sample rate index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SampleRateIndex {
    Rate96000 = 0,
    Rate88200 = 1,
    Rate64000 = 2,
    Rate48000 = 3,
    Rate44100 = 4,
    Rate32000 = 5,
    Rate24000 = 6,
    Rate22050 = 7,
    Rate16000 = 8,
    Rate12000 = 9,
    Rate11025 = 10,
    Rate8000 = 11,
    Rate7350 = 12,
    Explicit = 15,
}

impl SampleRateIndex {
    /// Get sample rate from index.
    pub fn to_sample_rate(self) -> Option<u32> {
        match self {
            SampleRateIndex::Rate96000 => Some(96000),
            SampleRateIndex::Rate88200 => Some(88200),
            SampleRateIndex::Rate64000 => Some(64000),
            SampleRateIndex::Rate48000 => Some(48000),
            SampleRateIndex::Rate44100 => Some(44100),
            SampleRateIndex::Rate32000 => Some(32000),
            SampleRateIndex::Rate24000 => Some(24000),
            SampleRateIndex::Rate22050 => Some(22050),
            SampleRateIndex::Rate16000 => Some(16000),
            SampleRateIndex::Rate12000 => Some(12000),
            SampleRateIndex::Rate11025 => Some(11025),
            SampleRateIndex::Rate8000 => Some(8000),
            SampleRateIndex::Rate7350 => Some(7350),
            SampleRateIndex::Explicit => None,
        }
    }

    /// Get index from sample rate.
    pub fn from_sample_rate(rate: u32) -> Self {
        match rate {
            96000 => SampleRateIndex::Rate96000,
            88200 => SampleRateIndex::Rate88200,
            64000 => SampleRateIndex::Rate64000,
            48000 => SampleRateIndex::Rate48000,
            44100 => SampleRateIndex::Rate44100,
            32000 => SampleRateIndex::Rate32000,
            24000 => SampleRateIndex::Rate24000,
            22050 => SampleRateIndex::Rate22050,
            16000 => SampleRateIndex::Rate16000,
            12000 => SampleRateIndex::Rate12000,
            11025 => SampleRateIndex::Rate11025,
            8000 => SampleRateIndex::Rate8000,
            7350 => SampleRateIndex::Rate7350,
            _ => SampleRateIndex::Explicit,
        }
    }
}

/// Channel configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChannelConfig {
    /// Defined in program config element.
    Specific = 0,
    /// Mono.
    Mono = 1,
    /// Stereo.
    Stereo = 2,
    /// 3.0 (L, R, C).
    Surround3 = 3,
    /// 4.0 (L, R, C, Cs).
    Surround4 = 4,
    /// 5.0 (L, R, C, Ls, Rs).
    Surround5 = 5,
    /// 5.1 (L, R, C, LFE, Ls, Rs).
    Surround51 = 6,
    /// 7.1 (L, R, C, LFE, Ls, Rs, Lrs, Rrs).
    Surround71 = 7,
}

impl ChannelConfig {
    /// Get number of channels.
    pub fn channels(&self) -> u8 {
        match self {
            ChannelConfig::Specific => 0,
            ChannelConfig::Mono => 1,
            ChannelConfig::Stereo => 2,
            ChannelConfig::Surround3 => 3,
            ChannelConfig::Surround4 => 4,
            ChannelConfig::Surround5 => 5,
            ChannelConfig::Surround51 => 6,
            ChannelConfig::Surround71 => 8,
        }
    }

    /// Create from channel count.
    pub fn from_channels(channels: u8) -> Option<Self> {
        match channels {
            1 => Some(ChannelConfig::Mono),
            2 => Some(ChannelConfig::Stereo),
            3 => Some(ChannelConfig::Surround3),
            4 => Some(ChannelConfig::Surround4),
            5 => Some(ChannelConfig::Surround5),
            6 => Some(ChannelConfig::Surround51),
            8 => Some(ChannelConfig::Surround71),
            _ => None,
        }
    }
}

/// Audio Specific Config.
#[derive(Debug, Clone)]
pub struct AudioSpecificConfig {
    /// Audio object type.
    pub object_type: u8,
    /// Sample rate index.
    pub sample_rate_index: u8,
    /// Explicit sample rate (if index == 15).
    pub sample_rate: u32,
    /// Channel configuration.
    pub channel_config: u8,
    /// Frame length (960 or 1024).
    pub frame_length: u16,
    /// Depends on core coder.
    pub depends_on_core_coder: bool,
    /// Extension audio object type.
    pub extension_object_type: Option<u8>,
    /// Extension sample rate.
    pub extension_sample_rate: Option<u32>,
    /// SBR present.
    pub sbr_present: bool,
    /// PS present.
    pub ps_present: bool,
}

impl AudioSpecificConfig {
    /// Parse from bytes.
    pub fn parse(data: &[u8]) -> Option<Self> {
        use transcode_core::bitstream::BitReader;

        let mut reader = BitReader::new(data);

        // Audio object type (5 bits, or 5 + 6 if 31)
        let mut object_type = reader.read_bits(5).ok()? as u8;
        if object_type == 31 {
            object_type = 32 + reader.read_bits(6).ok()? as u8;
        }

        // Sample rate index (4 bits)
        let sample_rate_index = reader.read_bits(4).ok()? as u8;
        let sample_rate = if sample_rate_index == 15 {
            reader.read_bits(24).ok()?
        } else {
            SampleRateIndex::from_sample_rate(0).to_sample_rate().unwrap_or(44100)
        };

        // Channel configuration (4 bits)
        let channel_config = reader.read_bits(4).ok()? as u8;

        Some(Self {
            object_type,
            sample_rate_index,
            sample_rate,
            channel_config,
            frame_length: 1024,
            depends_on_core_coder: false,
            extension_object_type: None,
            extension_sample_rate: None,
            sbr_present: object_type == 5 || object_type == 29,
            ps_present: object_type == 29,
        })
    }

    /// Encode to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(4);

        // Simple 2-byte ASC for AAC-LC
        let byte1 = (self.object_type << 3) | (self.sample_rate_index >> 1);
        let byte2 = ((self.sample_rate_index & 1) << 7) | (self.channel_config << 3);

        data.push(byte1);
        data.push(byte2);

        data
    }

    /// Get sample rate.
    pub fn get_sample_rate(&self) -> u32 {
        if self.sample_rate_index == 15 {
            self.sample_rate
        } else {
            match self.sample_rate_index {
                0 => 96000,
                1 => 88200,
                2 => 64000,
                3 => 48000,
                4 => 44100,
                5 => 32000,
                6 => 24000,
                7 => 22050,
                8 => 16000,
                9 => 12000,
                10 => 11025,
                11 => 8000,
                12 => 7350,
                _ => 44100,
            }
        }
    }

    /// Get number of channels.
    pub fn get_channels(&self) -> u8 {
        match self.channel_config {
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 5,
            6 => 6,
            7 => 8,
            _ => 2,
        }
    }
}
