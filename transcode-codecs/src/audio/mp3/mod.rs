//! MP3 (MPEG-1 Layer III) encoder and decoder implementation.

// Allow common patterns in signal processing codec implementation
#![allow(
    dead_code,
    unused_must_use,
    unused_variables,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::cast_abs_to_unsigned
)]

mod decoder;
mod encoder;
mod huffman;
mod tables;

pub use decoder::Mp3Decoder;
pub use encoder::{Mp3Encoder, Mp3EncoderConfig};

use transcode_core::error::{Error, Result};

/// MPEG audio version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MpegVersion {
    /// MPEG Version 2.5 (unofficial extension).
    Mpeg25 = 0,
    /// Reserved.
    Reserved = 1,
    /// MPEG Version 2.
    Mpeg2 = 2,
    /// MPEG Version 1.
    Mpeg1 = 3,
}

impl MpegVersion {
    /// Get sample rate modifier.
    pub fn sample_rate_index(&self) -> usize {
        match self {
            MpegVersion::Mpeg1 => 0,
            MpegVersion::Mpeg2 => 1,
            MpegVersion::Mpeg25 => 2,
            MpegVersion::Reserved => 0,
        }
    }
}

/// MPEG audio layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Layer {
    /// Reserved.
    Reserved = 0,
    /// Layer III.
    Layer3 = 1,
    /// Layer II.
    Layer2 = 2,
    /// Layer I.
    Layer1 = 3,
}

/// Channel mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChannelMode {
    /// Stereo.
    Stereo = 0,
    /// Joint stereo.
    JointStereo = 1,
    /// Dual channel (two mono channels).
    DualChannel = 2,
    /// Mono.
    Mono = 3,
}

impl ChannelMode {
    /// Get number of channels.
    pub fn channels(&self) -> u8 {
        match self {
            ChannelMode::Mono => 1,
            _ => 2,
        }
    }
}

/// MP3 frame header.
#[derive(Debug, Clone)]
pub struct Mp3FrameHeader {
    /// MPEG version.
    pub version: MpegVersion,
    /// Layer.
    pub layer: Layer,
    /// CRC protection.
    pub crc_protected: bool,
    /// Bitrate index.
    pub bitrate_index: u8,
    /// Sample rate index.
    pub sample_rate_index: u8,
    /// Padding bit.
    pub padding: bool,
    /// Private bit.
    pub private: bool,
    /// Channel mode.
    pub channel_mode: ChannelMode,
    /// Mode extension (for joint stereo).
    pub mode_extension: u8,
    /// Copyright flag.
    pub copyright: bool,
    /// Original flag.
    pub original: bool,
    /// Emphasis.
    pub emphasis: u8,
}

impl Mp3FrameHeader {
    /// Sync word for MP3.
    pub const SYNC_WORD: u16 = 0xFFE0;

    /// Parse header from bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(Error::Bitstream("MP3 header too short".into()));
        }

        // Check sync word (11 bits = 0x7FF)
        if data[0] != 0xFF || (data[1] & 0xE0) != 0xE0 {
            return Err(Error::Bitstream("Invalid MP3 sync word".into()));
        }

        let version = match (data[1] >> 3) & 3 {
            0 => MpegVersion::Mpeg25,
            1 => MpegVersion::Reserved,
            2 => MpegVersion::Mpeg2,
            3 => MpegVersion::Mpeg1,
            _ => unreachable!("2-bit value masked with & 3 can only be 0-3"),
        };

        let layer = match (data[1] >> 1) & 3 {
            0 => Layer::Reserved,
            1 => Layer::Layer3,
            2 => Layer::Layer2,
            3 => Layer::Layer1,
            _ => unreachable!("2-bit value masked with & 3 can only be 0-3"),
        };

        let crc_protected = (data[1] & 1) == 0;
        let bitrate_index = (data[2] >> 4) & 0xF;
        let sample_rate_index = (data[2] >> 2) & 3;
        let padding = (data[2] >> 1) & 1 == 1;
        let private = data[2] & 1 == 1;

        let channel_mode = match (data[3] >> 6) & 3 {
            0 => ChannelMode::Stereo,
            1 => ChannelMode::JointStereo,
            2 => ChannelMode::DualChannel,
            3 => ChannelMode::Mono,
            _ => unreachable!("2-bit value masked with & 3 can only be 0-3"),
        };

        let mode_extension = (data[3] >> 4) & 3;
        let copyright = (data[3] >> 3) & 1 == 1;
        let original = (data[3] >> 2) & 1 == 1;
        let emphasis = data[3] & 3;

        Ok(Self {
            version,
            layer,
            crc_protected,
            bitrate_index,
            sample_rate_index,
            padding,
            private,
            channel_mode,
            mode_extension,
            copyright,
            original,
            emphasis,
        })
    }

    /// Get bitrate in kbps.
    pub fn bitrate(&self) -> u32 {
        const BITRATES_V1_L3: [u32; 16] = [
            0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
        ];
        const BITRATES_V2_L3: [u32; 16] = [
            0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
        ];

        let table = match self.version {
            MpegVersion::Mpeg1 => &BITRATES_V1_L3,
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => &BITRATES_V2_L3,
            MpegVersion::Reserved => &BITRATES_V1_L3,
        };

        table[self.bitrate_index as usize] * 1000
    }

    /// Get sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        const SAMPLE_RATES: [[u32; 3]; 3] = [
            [44100, 48000, 32000], // MPEG1
            [22050, 24000, 16000], // MPEG2
            [11025, 12000, 8000],  // MPEG2.5
        ];

        let version_idx = self.version.sample_rate_index();
        SAMPLE_RATES[version_idx][self.sample_rate_index as usize]
    }

    /// Get number of samples per frame.
    pub fn samples_per_frame(&self) -> usize {
        match self.version {
            MpegVersion::Mpeg1 => 1152,
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => 576,
            MpegVersion::Reserved => 1152,
        }
    }

    /// Get frame size in bytes.
    pub fn frame_size(&self) -> usize {
        let bitrate = self.bitrate();
        let sample_rate = self.sample_rate();

        if bitrate == 0 || sample_rate == 0 {
            return 0;
        }

        let padding = if self.padding { 1 } else { 0 };

        match self.version {
            MpegVersion::Mpeg1 => (144 * bitrate / sample_rate + padding) as usize,
            MpegVersion::Mpeg2 | MpegVersion::Mpeg25 => (72 * bitrate / sample_rate + padding) as usize,
            MpegVersion::Reserved => 0,
        }
    }

    /// Get side info size.
    pub fn side_info_size(&self) -> usize {
        match (self.version, self.channel_mode) {
            (MpegVersion::Mpeg1, ChannelMode::Mono) => 17,
            (MpegVersion::Mpeg1, _) => 32,
            (_, ChannelMode::Mono) => 9,
            _ => 17,
        }
    }

    /// Get header size (including CRC if present).
    pub fn header_size(&self) -> usize {
        if self.crc_protected { 6 } else { 4 }
    }
}

/// Side information for Layer III.
#[derive(Debug, Clone, Default)]
pub struct SideInfo {
    /// Main data begin pointer.
    pub main_data_begin: u16,
    /// Private bits.
    pub private_bits: u8,
    /// Scalefactor selection info.
    pub scfsi: [[bool; 4]; 2],
    /// Granule information.
    pub granules: [[GranuleInfo; 2]; 2],
}

/// Granule information.
#[derive(Debug, Clone, Default)]
pub struct GranuleInfo {
    /// Part2-3 length (bits).
    pub part2_3_length: u16,
    /// Big values.
    pub big_values: u16,
    /// Global gain.
    pub global_gain: u8,
    /// Scalefactor compress.
    pub scalefac_compress: u8,
    /// Window switching flag.
    pub window_switching: bool,
    /// Block type.
    pub block_type: u8,
    /// Mixed block flag.
    pub mixed_block: bool,
    /// Table select.
    pub table_select: [u8; 3],
    /// Subblock gain.
    pub subblock_gain: [u8; 3],
    /// Region 0 count.
    pub region0_count: u8,
    /// Region 1 count.
    pub region1_count: u8,
    /// Preflag.
    pub preflag: bool,
    /// Scalefactor scale.
    pub scalefac_scale: bool,
    /// Count1 table select.
    pub count1_table: bool,
}

/// Find next MP3 sync in a byte stream.
pub fn find_mp3_sync(data: &[u8]) -> Option<usize> {
    for i in 0..data.len().saturating_sub(1) {
        if data[i] == 0xFF && (data[i + 1] & 0xE0) == 0xE0 {
            // Validate frame header
            if Mp3FrameHeader::parse(&data[i..]).is_ok() {
                return Some(i);
            }
        }
    }
    None
}

/// Iterator over MP3 frames.
pub struct Mp3FrameIterator<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Mp3FrameIterator<'a> {
    /// Create a new MP3 frame iterator.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl<'a> Iterator for Mp3FrameIterator<'a> {
    type Item = Result<(&'a [u8], Mp3FrameHeader)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }

        let remaining = &self.data[self.offset..];

        // Find sync
        let sync_offset = find_mp3_sync(remaining)?;
        let frame_start = &remaining[sync_offset..];

        // Parse header
        let header = match Mp3FrameHeader::parse(frame_start) {
            Ok(h) => h,
            Err(e) => return Some(Err(e)),
        };

        let frame_size = header.frame_size();
        if frame_size == 0 || frame_start.len() < frame_size {
            return Some(Err(Error::Bitstream("Incomplete MP3 frame".into())));
        }

        let frame_data = &frame_start[..frame_size];
        self.offset += sync_offset + frame_size;

        Some(Ok((frame_data, header)))
    }
}
