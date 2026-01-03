//! MP4/ISOBMFF container format implementation.
//!
//! Supports reading and writing MP4 files with H.264/H.265 video
//! and AAC/MP3 audio.

mod atoms;
mod demuxer;
mod muxer;

pub use demuxer::Mp4Demuxer;
pub use muxer::Mp4Muxer;

use transcode_core::error::{Error, Result};

/// MP4 brand types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mp4Brand {
    /// ISO Base Media File.
    Isom,
    /// ISO Base Media File version 2.
    Iso2,
    /// AVC file.
    Avc1,
    /// MP4 version 1.
    Mp41,
    /// MP4 version 2.
    Mp42,
    /// 3GPP.
    Gp3,
    /// QuickTime.
    Qt,
    /// DASH.
    Dash,
    /// Unknown brand.
    Unknown([u8; 4]),
}

impl Mp4Brand {
    /// Parse brand from 4 bytes.
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        match bytes {
            b"isom" => Mp4Brand::Isom,
            b"iso2" => Mp4Brand::Iso2,
            b"avc1" => Mp4Brand::Avc1,
            b"mp41" => Mp4Brand::Mp41,
            b"mp42" => Mp4Brand::Mp42,
            b"3gp4" | b"3gp5" | b"3gp6" => Mp4Brand::Gp3,
            b"qt  " => Mp4Brand::Qt,
            b"dash" => Mp4Brand::Dash,
            _ => Mp4Brand::Unknown(*bytes),
        }
    }

    /// Convert to bytes.
    pub fn to_bytes(&self) -> [u8; 4] {
        match self {
            Mp4Brand::Isom => *b"isom",
            Mp4Brand::Iso2 => *b"iso2",
            Mp4Brand::Avc1 => *b"avc1",
            Mp4Brand::Mp41 => *b"mp41",
            Mp4Brand::Mp42 => *b"mp42",
            Mp4Brand::Gp3 => *b"3gp5",
            Mp4Brand::Qt => *b"qt  ",
            Mp4Brand::Dash => *b"dash",
            Mp4Brand::Unknown(b) => *b,
        }
    }
}

/// Sample entry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleEntryType {
    /// AVC/H.264.
    Avc1,
    /// AVC/H.264 (parameter sets in-band).
    Avc3,
    /// HEVC/H.265.
    Hev1,
    /// HEVC/H.265 (parameter sets in-band).
    Hvc1,
    /// VP9.
    Vp09,
    /// AV1.
    Av01,
    /// AAC.
    Mp4a,
    /// AC-3.
    Ac3,
    /// E-AC-3.
    Ec3,
    /// Opus.
    Opus,
    /// FLAC.
    Flac,
    /// Unknown.
    Unknown([u8; 4]),
}

impl SampleEntryType {
    /// Parse from 4 bytes.
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        match bytes {
            b"avc1" => SampleEntryType::Avc1,
            b"avc3" => SampleEntryType::Avc3,
            b"hev1" => SampleEntryType::Hev1,
            b"hvc1" => SampleEntryType::Hvc1,
            b"vp09" => SampleEntryType::Vp09,
            b"av01" => SampleEntryType::Av01,
            b"mp4a" => SampleEntryType::Mp4a,
            b"ac-3" => SampleEntryType::Ac3,
            b"ec-3" => SampleEntryType::Ec3,
            b"Opus" => SampleEntryType::Opus,
            b"fLaC" => SampleEntryType::Flac,
            _ => SampleEntryType::Unknown(*bytes),
        }
    }

    /// Convert to bytes.
    pub fn to_bytes(&self) -> [u8; 4] {
        match self {
            SampleEntryType::Avc1 => *b"avc1",
            SampleEntryType::Avc3 => *b"avc3",
            SampleEntryType::Hev1 => *b"hev1",
            SampleEntryType::Hvc1 => *b"hvc1",
            SampleEntryType::Vp09 => *b"vp09",
            SampleEntryType::Av01 => *b"av01",
            SampleEntryType::Mp4a => *b"mp4a",
            SampleEntryType::Ac3 => *b"ac-3",
            SampleEntryType::Ec3 => *b"ec-3",
            SampleEntryType::Opus => *b"Opus",
            SampleEntryType::Flac => *b"fLaC",
            SampleEntryType::Unknown(b) => *b,
        }
    }

    /// Check if this is a video type.
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            SampleEntryType::Avc1
                | SampleEntryType::Avc3
                | SampleEntryType::Hev1
                | SampleEntryType::Hvc1
                | SampleEntryType::Vp09
                | SampleEntryType::Av01
        )
    }

    /// Check if this is an audio type.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            SampleEntryType::Mp4a
                | SampleEntryType::Ac3
                | SampleEntryType::Ec3
                | SampleEntryType::Opus
                | SampleEntryType::Flac
        )
    }
}

/// Read a 32-bit big-endian integer.
fn read_u32_be(data: &[u8]) -> Result<u32> {
    if data.len() < 4 {
        return Err(Error::Bitstream("Not enough data for u32".into()));
    }
    Ok(u32::from_be_bytes([data[0], data[1], data[2], data[3]]))
}

/// Read a 64-bit big-endian integer.
fn read_u64_be(data: &[u8]) -> Result<u64> {
    if data.len() < 8 {
        return Err(Error::Bitstream("Not enough data for u64".into()));
    }
    Ok(u64::from_be_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]))
}

/// Write a 32-bit big-endian integer.
fn write_u32_be(value: u32) -> [u8; 4] {
    value.to_be_bytes()
}

/// Write a 64-bit big-endian integer.
fn write_u64_be(value: u64) -> [u8; 8] {
    value.to_be_bytes()
}
