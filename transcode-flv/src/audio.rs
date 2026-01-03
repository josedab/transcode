//! FLV audio codec support.
//!
//! FLV audio tags contain a one-byte header followed by audio data:
//! - Sound format (4 bits)
//! - Sound rate (2 bits)
//! - Sound size (1 bit)
//! - Sound type (1 bit)
//!
//! For AAC, an additional byte specifies the AAC packet type.

use crate::error::{FlvError, Result};
use std::io::{Read, Write};

/// Sound format: Linear PCM, platform endian.
pub const SOUND_FORMAT_LINEAR_PCM_PE: u8 = 0;
/// Sound format: ADPCM.
pub const SOUND_FORMAT_ADPCM: u8 = 1;
/// Sound format: MP3.
pub const SOUND_FORMAT_MP3: u8 = 2;
/// Sound format: Linear PCM, little endian.
pub const SOUND_FORMAT_LINEAR_PCM_LE: u8 = 3;
/// Sound format: Nellymoser 16kHz mono.
pub const SOUND_FORMAT_NELLYMOSER_16K: u8 = 4;
/// Sound format: Nellymoser 8kHz mono.
pub const SOUND_FORMAT_NELLYMOSER_8K: u8 = 5;
/// Sound format: Nellymoser.
pub const SOUND_FORMAT_NELLYMOSER: u8 = 6;
/// Sound format: G.711 A-law.
pub const SOUND_FORMAT_G711_ALAW: u8 = 7;
/// Sound format: G.711 mu-law.
pub const SOUND_FORMAT_G711_MULAW: u8 = 8;
/// Sound format: AAC.
pub const SOUND_FORMAT_AAC: u8 = 10;
/// Sound format: Speex.
pub const SOUND_FORMAT_SPEEX: u8 = 11;
/// Sound format: MP3 8kHz.
pub const SOUND_FORMAT_MP3_8K: u8 = 14;
/// Sound format: Device-specific.
pub const SOUND_FORMAT_DEVICE_SPECIFIC: u8 = 15;

/// AAC packet type: Sequence header (AudioSpecificConfig).
pub const AAC_PACKET_SEQUENCE_HEADER: u8 = 0;
/// AAC packet type: Raw AAC data.
pub const AAC_PACKET_RAW: u8 = 1;

/// Audio codec/format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SoundFormat {
    /// Linear PCM, platform endian.
    LinearPcmPe = SOUND_FORMAT_LINEAR_PCM_PE,
    /// ADPCM.
    Adpcm = SOUND_FORMAT_ADPCM,
    /// MP3.
    Mp3 = SOUND_FORMAT_MP3,
    /// Linear PCM, little endian.
    LinearPcmLe = SOUND_FORMAT_LINEAR_PCM_LE,
    /// Nellymoser 16kHz mono.
    Nellymoser16k = SOUND_FORMAT_NELLYMOSER_16K,
    /// Nellymoser 8kHz mono.
    Nellymoser8k = SOUND_FORMAT_NELLYMOSER_8K,
    /// Nellymoser.
    Nellymoser = SOUND_FORMAT_NELLYMOSER,
    /// G.711 A-law.
    G711Alaw = SOUND_FORMAT_G711_ALAW,
    /// G.711 mu-law.
    G711Mulaw = SOUND_FORMAT_G711_MULAW,
    /// AAC.
    Aac = SOUND_FORMAT_AAC,
    /// Speex.
    Speex = SOUND_FORMAT_SPEEX,
    /// MP3 8kHz.
    Mp38k = SOUND_FORMAT_MP3_8K,
    /// Device-specific.
    DeviceSpecific = SOUND_FORMAT_DEVICE_SPECIFIC,
}

impl SoundFormat {
    /// Create from raw byte value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            SOUND_FORMAT_LINEAR_PCM_PE => Some(Self::LinearPcmPe),
            SOUND_FORMAT_ADPCM => Some(Self::Adpcm),
            SOUND_FORMAT_MP3 => Some(Self::Mp3),
            SOUND_FORMAT_LINEAR_PCM_LE => Some(Self::LinearPcmLe),
            SOUND_FORMAT_NELLYMOSER_16K => Some(Self::Nellymoser16k),
            SOUND_FORMAT_NELLYMOSER_8K => Some(Self::Nellymoser8k),
            SOUND_FORMAT_NELLYMOSER => Some(Self::Nellymoser),
            SOUND_FORMAT_G711_ALAW => Some(Self::G711Alaw),
            SOUND_FORMAT_G711_MULAW => Some(Self::G711Mulaw),
            SOUND_FORMAT_AAC => Some(Self::Aac),
            SOUND_FORMAT_SPEEX => Some(Self::Speex),
            SOUND_FORMAT_MP3_8K => Some(Self::Mp38k),
            SOUND_FORMAT_DEVICE_SPECIFIC => Some(Self::DeviceSpecific),
            _ => None,
        }
    }

    /// Convert to raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Get the codec name.
    pub fn name(self) -> &'static str {
        match self {
            Self::LinearPcmPe | Self::LinearPcmLe => "PCM",
            Self::Adpcm => "ADPCM",
            Self::Mp3 | Self::Mp38k => "MP3",
            Self::Nellymoser16k | Self::Nellymoser8k | Self::Nellymoser => "Nellymoser",
            Self::G711Alaw => "G.711 A-law",
            Self::G711Mulaw => "G.711 mu-law",
            Self::Aac => "AAC",
            Self::Speex => "Speex",
            Self::DeviceSpecific => "Device-specific",
        }
    }

    /// Check if this format requires an AAC packet type byte.
    pub fn has_packet_type(self) -> bool {
        self == Self::Aac
    }
}

/// Sound sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SoundRate {
    /// 5.5 kHz.
    Rate5500Hz = 0,
    /// 11 kHz.
    Rate11000Hz = 1,
    /// 22 kHz.
    Rate22000Hz = 2,
    /// 44 kHz.
    Rate44000Hz = 3,
}

impl SoundRate {
    /// Create from raw 2-bit value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value & 0x03 {
            0 => Some(Self::Rate5500Hz),
            1 => Some(Self::Rate11000Hz),
            2 => Some(Self::Rate22000Hz),
            3 => Some(Self::Rate44000Hz),
            _ => None,
        }
    }

    /// Convert to raw 2-bit value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Get the sample rate in Hz.
    pub fn hz(self) -> u32 {
        match self {
            Self::Rate5500Hz => 5500,
            Self::Rate11000Hz => 11025,
            Self::Rate22000Hz => 22050,
            Self::Rate44000Hz => 44100,
        }
    }

    /// Create from sample rate in Hz (approximate match).
    pub fn from_hz(hz: u32) -> Self {
        if hz <= 5500 {
            Self::Rate5500Hz
        } else if hz <= 11025 {
            Self::Rate11000Hz
        } else if hz <= 22050 {
            Self::Rate22000Hz
        } else {
            Self::Rate44000Hz
        }
    }
}

/// Sound sample size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SoundSize {
    /// 8-bit samples.
    Bits8 = 0,
    /// 16-bit samples.
    Bits16 = 1,
}

impl SoundSize {
    /// Create from raw 1-bit value.
    pub fn from_u8(value: u8) -> Self {
        if value & 0x01 == 0 {
            Self::Bits8
        } else {
            Self::Bits16
        }
    }

    /// Convert to raw 1-bit value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Get the bit depth.
    pub fn bits(self) -> u8 {
        match self {
            Self::Bits8 => 8,
            Self::Bits16 => 16,
        }
    }
}

/// Sound channel configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SoundType {
    /// Mono.
    Mono = 0,
    /// Stereo.
    Stereo = 1,
}

impl SoundType {
    /// Create from raw 1-bit value.
    pub fn from_u8(value: u8) -> Self {
        if value & 0x01 == 0 {
            Self::Mono
        } else {
            Self::Stereo
        }
    }

    /// Convert to raw 1-bit value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Get the number of channels.
    pub fn channels(self) -> u8 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
        }
    }
}

/// AAC packet type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AacPacketType {
    /// AAC sequence header (AudioSpecificConfig).
    SequenceHeader = AAC_PACKET_SEQUENCE_HEADER,
    /// Raw AAC frame data.
    Raw = AAC_PACKET_RAW,
}

impl AacPacketType {
    /// Create from raw byte value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            AAC_PACKET_SEQUENCE_HEADER => Some(Self::SequenceHeader),
            AAC_PACKET_RAW => Some(Self::Raw),
            _ => None,
        }
    }

    /// Convert to raw byte value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Check if this is a sequence header.
    pub fn is_sequence_header(self) -> bool {
        self == Self::SequenceHeader
    }
}

/// FLV audio tag header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioTagHeader {
    /// Sound format/codec.
    pub sound_format: SoundFormat,
    /// Sample rate.
    pub sound_rate: SoundRate,
    /// Sample size (8 or 16 bit).
    pub sound_size: SoundSize,
    /// Channel configuration (mono or stereo).
    pub sound_type: SoundType,
    /// AAC packet type (only for AAC).
    pub aac_packet_type: Option<AacPacketType>,
}

impl AudioTagHeader {
    /// Create a new audio tag header.
    pub fn new(
        sound_format: SoundFormat,
        sound_rate: SoundRate,
        sound_size: SoundSize,
        sound_type: SoundType,
    ) -> Self {
        Self {
            sound_format,
            sound_rate,
            sound_size,
            sound_type,
            aac_packet_type: None,
        }
    }

    /// Create an AAC audio tag header.
    pub fn aac(packet_type: AacPacketType) -> Self {
        Self {
            sound_format: SoundFormat::Aac,
            sound_rate: SoundRate::Rate44000Hz,
            sound_size: SoundSize::Bits16,
            sound_type: SoundType::Stereo,
            aac_packet_type: Some(packet_type),
        }
    }

    /// Create an AAC sequence header tag header.
    pub fn aac_sequence_header() -> Self {
        Self::aac(AacPacketType::SequenceHeader)
    }

    /// Create an AAC raw data tag header.
    pub fn aac_raw() -> Self {
        Self::aac(AacPacketType::Raw)
    }

    /// Create an MP3 audio tag header.
    pub fn mp3(sample_rate: u32, stereo: bool) -> Self {
        Self {
            sound_format: SoundFormat::Mp3,
            sound_rate: SoundRate::from_hz(sample_rate),
            sound_size: SoundSize::Bits16,
            sound_type: if stereo {
                SoundType::Stereo
            } else {
                SoundType::Mono
            },
            aac_packet_type: None,
        }
    }

    /// Get the header size in bytes.
    pub fn size(&self) -> usize {
        if self.sound_format.has_packet_type() {
            2
        } else {
            1
        }
    }

    /// Parse from a reader.
    pub fn parse<R: Read>(reader: &mut R) -> Result<Self> {
        // Read first byte
        let mut first_byte = [0u8; 1];
        reader.read_exact(&mut first_byte).map_err(|_| FlvError::UnexpectedEnd { offset: 0 })?;
        let byte = first_byte[0];

        let format_value = (byte >> 4) & 0x0F;
        let rate_value = (byte >> 2) & 0x03;
        let size_value = (byte >> 1) & 0x01;
        let type_value = byte & 0x01;

        let sound_format = SoundFormat::from_u8(format_value)
            .ok_or(FlvError::InvalidAudioFormat(format_value))?;
        let sound_rate = SoundRate::from_u8(rate_value).unwrap();
        let sound_size = SoundSize::from_u8(size_value);
        let sound_type = SoundType::from_u8(type_value);

        let aac_packet_type = if sound_format.has_packet_type() {
            let mut packet_type_byte = [0u8; 1];
            reader.read_exact(&mut packet_type_byte)?;
            Some(
                AacPacketType::from_u8(packet_type_byte[0])
                    .ok_or(FlvError::InvalidAacPacketType(packet_type_byte[0]))?,
            )
        } else {
            None
        };

        Ok(Self {
            sound_format,
            sound_rate,
            sound_size,
            sound_type,
            aac_packet_type,
        })
    }

    /// Write to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> Result<usize> {
        let byte = (self.sound_format.as_u8() << 4)
            | (self.sound_rate.as_u8() << 2)
            | (self.sound_size.as_u8() << 1)
            | self.sound_type.as_u8();

        writer.write_all(&[byte])?;

        if let Some(aac_type) = self.aac_packet_type {
            writer.write_all(&[aac_type.as_u8()])?;
            Ok(2)
        } else {
            Ok(1)
        }
    }

    /// Check if this is an AAC sequence header.
    pub fn is_aac_sequence_header(&self) -> bool {
        self.aac_packet_type == Some(AacPacketType::SequenceHeader)
    }
}

/// AAC AudioSpecificConfig.
///
/// This is sent as the AAC sequence header and contains configuration
/// for the AAC decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AacConfig {
    /// Audio object type (1 = AAC Main, 2 = AAC LC, etc.).
    pub audio_object_type: u8,
    /// Sample rate index (0-12, or 15 for explicit).
    pub sample_rate_index: u8,
    /// Explicit sample rate (if sample_rate_index == 15).
    pub sample_rate: Option<u32>,
    /// Channel configuration.
    pub channel_config: u8,
    /// Raw AudioSpecificConfig bytes.
    pub raw: Vec<u8>,
}

impl AacConfig {
    /// Standard AAC sample rates.
    pub const SAMPLE_RATES: [u32; 13] = [
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
    ];

    /// Create a new AAC config.
    pub fn new(audio_object_type: u8, sample_rate: u32, channels: u8) -> Self {
        let sample_rate_index = Self::SAMPLE_RATES
            .iter()
            .position(|&r| r == sample_rate)
            .map(|i| i as u8)
            .unwrap_or(15);

        let explicit_rate = if sample_rate_index == 15 {
            Some(sample_rate)
        } else {
            None
        };

        let raw = Self::build_raw(audio_object_type, sample_rate_index, explicit_rate, channels);

        Self {
            audio_object_type,
            sample_rate_index,
            sample_rate: explicit_rate,
            channel_config: channels,
            raw,
        }
    }

    /// Create an AAC-LC config.
    pub fn aac_lc(sample_rate: u32, channels: u8) -> Self {
        Self::new(2, sample_rate, channels) // 2 = AAC-LC
    }

    /// Create an HE-AAC config.
    pub fn he_aac(sample_rate: u32, channels: u8) -> Self {
        Self::new(5, sample_rate, channels) // 5 = HE-AAC
    }

    /// Build raw AudioSpecificConfig bytes.
    fn build_raw(
        audio_object_type: u8,
        sample_rate_index: u8,
        explicit_rate: Option<u32>,
        channel_config: u8,
    ) -> Vec<u8> {
        let mut raw = Vec::with_capacity(4);

        // AudioObjectType (5 bits) + SamplingFrequencyIndex (4 bits) = 9 bits
        // First byte: 5 bits of AOT + 3 bits of SFI
        // Second byte: 1 bit of SFI + 4 bits of channel config + 3 bits of padding

        let byte1 = (audio_object_type << 3) | (sample_rate_index >> 1);
        raw.push(byte1);

        if sample_rate_index == 15 {
            // Need to include explicit sample rate (24 bits)
            let rate = explicit_rate.unwrap_or(44100);
            let byte2 = ((sample_rate_index & 0x01) << 7) | ((rate >> 17) & 0x7F) as u8;
            let byte3 = ((rate >> 9) & 0xFF) as u8;
            let byte4 = ((rate >> 1) & 0xFF) as u8;
            let byte5 = ((rate & 0x01) << 7) as u8 | ((channel_config & 0x0F) << 3);
            raw.push(byte2);
            raw.push(byte3);
            raw.push(byte4);
            raw.push(byte5);
        } else {
            let byte2 = ((sample_rate_index & 0x01) << 7) | ((channel_config & 0x0F) << 3);
            raw.push(byte2);
        }

        raw
    }

    /// Parse from raw AudioSpecificConfig bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(FlvError::InvalidSequenceHeader {
                codec: "AAC".to_string(),
                message: "AudioSpecificConfig too short".to_string(),
            });
        }

        let audio_object_type = (data[0] >> 3) & 0x1F;
        let sample_rate_index = ((data[0] & 0x07) << 1) | ((data[1] >> 7) & 0x01);

        let (sample_rate, channel_config) = if sample_rate_index == 15 {
            if data.len() < 5 {
                return Err(FlvError::InvalidSequenceHeader {
                    codec: "AAC".to_string(),
                    message: "AudioSpecificConfig too short for explicit sample rate".to_string(),
                });
            }
            let rate = ((data[1] as u32 & 0x7F) << 17)
                | ((data[2] as u32) << 9)
                | ((data[3] as u32) << 1)
                | ((data[4] as u32) >> 7);
            let channels = (data[4] >> 3) & 0x0F;
            (Some(rate), channels)
        } else {
            let channels = (data[1] >> 3) & 0x0F;
            (None, channels)
        };

        Ok(Self {
            audio_object_type,
            sample_rate_index,
            sample_rate,
            channel_config,
            raw: data.to_vec(),
        })
    }

    /// Get the sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        if let Some(rate) = self.sample_rate {
            rate
        } else if (self.sample_rate_index as usize) < Self::SAMPLE_RATES.len() {
            Self::SAMPLE_RATES[self.sample_rate_index as usize]
        } else {
            44100 // Default
        }
    }

    /// Get the number of channels.
    pub fn channels(&self) -> u8 {
        match self.channel_config {
            0 => 0, // Defined in stream
            1 => 1, // Mono
            2 => 2, // Stereo
            3 => 3, // 3.0
            4 => 4, // 4.0
            5 => 5, // 5.0
            6 => 6, // 5.1
            7 => 8, // 7.1
            _ => 2, // Default to stereo
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_sound_format() {
        assert_eq!(SoundFormat::from_u8(2), Some(SoundFormat::Mp3));
        assert_eq!(SoundFormat::from_u8(10), Some(SoundFormat::Aac));
        assert_eq!(SoundFormat::from_u8(99), None);

        assert_eq!(SoundFormat::Aac.name(), "AAC");
        assert_eq!(SoundFormat::Mp3.name(), "MP3");
    }

    #[test]
    fn test_sound_rate() {
        assert_eq!(SoundRate::Rate44000Hz.hz(), 44100);
        assert_eq!(SoundRate::from_hz(48000), SoundRate::Rate44000Hz);
        assert_eq!(SoundRate::from_hz(8000), SoundRate::Rate11000Hz);
        assert_eq!(SoundRate::from_hz(5500), SoundRate::Rate5500Hz);
    }

    #[test]
    fn test_sound_size() {
        assert_eq!(SoundSize::from_u8(0).bits(), 8);
        assert_eq!(SoundSize::from_u8(1).bits(), 16);
    }

    #[test]
    fn test_sound_type() {
        assert_eq!(SoundType::Mono.channels(), 1);
        assert_eq!(SoundType::Stereo.channels(), 2);
    }

    #[test]
    fn test_audio_tag_header_roundtrip() {
        let original = AudioTagHeader::new(
            SoundFormat::Mp3,
            SoundRate::Rate44000Hz,
            SoundSize::Bits16,
            SoundType::Stereo,
        );

        let mut buffer = Vec::new();
        original.write(&mut buffer).unwrap();

        assert_eq!(buffer.len(), 1);

        let mut cursor = Cursor::new(&buffer);
        let parsed = AudioTagHeader::parse(&mut cursor).unwrap();

        assert_eq!(original, parsed);
    }

    #[test]
    fn test_aac_audio_tag_header_roundtrip() {
        let original = AudioTagHeader::aac_raw();

        let mut buffer = Vec::new();
        original.write(&mut buffer).unwrap();

        assert_eq!(buffer.len(), 2);

        let mut cursor = Cursor::new(&buffer);
        let parsed = AudioTagHeader::parse(&mut cursor).unwrap();

        assert_eq!(original, parsed);
    }

    #[test]
    fn test_aac_config() {
        let config = AacConfig::aac_lc(44100, 2);
        assert_eq!(config.audio_object_type, 2);
        assert_eq!(config.sample_rate(), 44100);
        assert_eq!(config.channels(), 2);
    }

    #[test]
    fn test_aac_config_parse() {
        // AAC-LC, 44100 Hz, stereo
        let data = [0x12, 0x10];
        let config = AacConfig::parse(&data).unwrap();

        assert_eq!(config.audio_object_type, 2); // AAC-LC
        assert_eq!(config.sample_rate_index, 4); // 44100
        assert_eq!(config.channel_config, 2); // Stereo
    }

    #[test]
    fn test_mp3_header() {
        let header = AudioTagHeader::mp3(44100, true);
        assert_eq!(header.sound_format, SoundFormat::Mp3);
        assert_eq!(header.sound_rate, SoundRate::Rate44000Hz);
        assert_eq!(header.sound_type, SoundType::Stereo);
    }

    #[test]
    fn test_aac_sequence_header_check() {
        let header = AudioTagHeader::aac_sequence_header();
        assert!(header.is_aac_sequence_header());

        let header = AudioTagHeader::aac_raw();
        assert!(!header.is_aac_sequence_header());
    }

    #[test]
    fn test_sound_format_has_packet_type() {
        assert!(SoundFormat::Aac.has_packet_type());
        assert!(!SoundFormat::Mp3.has_packet_type());
    }

    #[test]
    fn test_audio_tag_header_size() {
        let mp3_header = AudioTagHeader::mp3(44100, true);
        assert_eq!(mp3_header.size(), 1);

        let aac_header = AudioTagHeader::aac_raw();
        assert_eq!(aac_header.size(), 2);
    }
}
