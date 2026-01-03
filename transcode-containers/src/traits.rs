//! Container format traits for demuxing and muxing.

use transcode_core::error::Result;
use transcode_core::packet::Packet;
use transcode_core::rational::Rational;
use std::io::{Read, Seek, Write};

/// Seek target for demuxing operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeekTarget {
    /// Seek to a timestamp in microseconds.
    /// The demuxer will seek to the nearest keyframe at or before this timestamp.
    Timestamp(i64),
    /// Seek to a byte offset in the file.
    /// This is useful for resuming from a known position.
    ByteOffset(u64),
    /// Seek to a specific sample/frame number for a given track.
    Sample { track_index: usize, sample_number: usize },
}

impl SeekTarget {
    /// Create a timestamp-based seek target from microseconds.
    pub fn from_micros(micros: i64) -> Self {
        SeekTarget::Timestamp(micros)
    }

    /// Create a timestamp-based seek target from milliseconds.
    pub fn from_millis(millis: i64) -> Self {
        SeekTarget::Timestamp(millis.saturating_mul(1000))
    }

    /// Create a timestamp-based seek target from seconds.
    pub fn from_secs(secs: f64) -> Self {
        SeekTarget::Timestamp((secs * 1_000_000.0) as i64)
    }

    /// Create a byte offset seek target.
    pub fn from_byte_offset(offset: u64) -> Self {
        SeekTarget::ByteOffset(offset)
    }

    /// Create a sample-based seek target.
    pub fn from_sample(track_index: usize, sample_number: usize) -> Self {
        SeekTarget::Sample { track_index, sample_number }
    }
}

/// Seek mode options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeekMode {
    /// Seek to the nearest keyframe at or before the target (default).
    #[default]
    Backward,
    /// Seek to the nearest keyframe at or after the target.
    Forward,
    /// Seek to the exact position (may land on non-keyframe).
    Exact,
}

/// Result of a seek operation, providing information about where we actually landed.
#[derive(Debug, Clone)]
pub struct SeekResult {
    /// The actual timestamp we seeked to (in microseconds).
    pub timestamp_us: i64,
    /// Whether we landed on a keyframe.
    pub is_keyframe: bool,
    /// The sample index we seeked to for each track.
    pub sample_indices: Vec<usize>,
}

/// Track type in a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle track.
    Subtitle,
    /// Data track.
    Data,
    /// Unknown track type.
    Unknown,
}

/// Codec identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecId {
    /// H.264/AVC.
    H264,
    /// H.265/HEVC.
    H265,
    /// VP9.
    Vp9,
    /// AV1.
    Av1,
    /// AAC.
    Aac,
    /// MP3.
    Mp3,
    /// Opus.
    Opus,
    /// FLAC.
    Flac,
    /// Unknown codec.
    Unknown(String),
}

impl CodecId {
    /// Get FourCC for the codec.
    pub fn fourcc(&self) -> Option<[u8; 4]> {
        match self {
            CodecId::H264 => Some(*b"avc1"),
            CodecId::H265 => Some(*b"hvc1"),
            CodecId::Vp9 => Some(*b"vp09"),
            CodecId::Av1 => Some(*b"av01"),
            CodecId::Aac => Some(*b"mp4a"),
            CodecId::Mp3 => Some(*b"mp4a"),
            CodecId::Opus => Some(*b"Opus"),
            CodecId::Flac => Some(*b"fLaC"),
            CodecId::Unknown(_) => None,
        }
    }
}

/// Stream information.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Stream index.
    pub index: usize,
    /// Track type.
    pub track_type: TrackType,
    /// Codec ID.
    pub codec_id: CodecId,
    /// Time base.
    pub time_base: Rational,
    /// Duration in time base units.
    pub duration: Option<i64>,
    /// Codec-specific extra data.
    pub extra_data: Option<Vec<u8>>,
    /// Video-specific info.
    pub video: Option<VideoStreamInfo>,
    /// Audio-specific info.
    pub audio: Option<AudioStreamInfo>,
}

/// Video stream information.
#[derive(Debug, Clone)]
pub struct VideoStreamInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate.
    pub frame_rate: Option<Rational>,
    /// Pixel aspect ratio.
    pub pixel_aspect_ratio: Option<Rational>,
    /// Bit depth.
    pub bit_depth: u8,
}

/// Audio stream information.
#[derive(Debug, Clone)]
pub struct AudioStreamInfo {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bits per sample.
    pub bits_per_sample: u8,
}

/// Demuxer trait for reading container formats.
pub trait Demuxer {
    /// Open a container for reading.
    fn open<R: Read + Seek + Send + 'static>(&mut self, reader: R) -> Result<()>;

    /// Get container format name.
    fn format_name(&self) -> &str;

    /// Get duration in microseconds.
    fn duration(&self) -> Option<i64>;

    /// Get number of streams.
    fn num_streams(&self) -> usize;

    /// Get stream information.
    fn stream_info(&self, index: usize) -> Option<&StreamInfo>;

    /// Read the next packet.
    fn read_packet(&mut self) -> Result<Option<Packet<'static>>>;

    /// Seek to a timestamp (in microseconds).
    /// This is a convenience method that calls `seek_to` with `SeekTarget::Timestamp`.
    fn seek(&mut self, timestamp_us: i64) -> Result<()> {
        self.seek_to(SeekTarget::Timestamp(timestamp_us), SeekMode::Backward)?;
        Ok(())
    }

    /// Seek to a specific target with the given mode.
    ///
    /// Supports seeking by:
    /// - Timestamp (in microseconds)
    /// - Byte offset in the file
    /// - Sample/frame number for a specific track
    ///
    /// The seek mode controls how the demuxer handles non-keyframe positions:
    /// - `Backward`: Seek to the nearest keyframe at or before the target
    /// - `Forward`: Seek to the nearest keyframe at or after the target
    /// - `Exact`: Seek to the exact position (may result in decode artifacts)
    ///
    /// Returns a `SeekResult` with information about where we actually landed.
    fn seek_to(&mut self, target: SeekTarget, mode: SeekMode) -> Result<SeekResult>;

    /// Check if seeking is supported by this demuxer.
    fn can_seek(&self) -> bool {
        true
    }

    /// Get the current position as a byte offset.
    fn position(&self) -> Option<u64>;

    /// Close the demuxer.
    fn close(&mut self);
}

/// Muxer trait for writing container formats.
pub trait Muxer {
    /// Create a new container for writing.
    fn create<W: Write + Seek + Send + 'static>(&mut self, writer: W) -> Result<()>;

    /// Get container format name.
    fn format_name(&self) -> &str;

    /// Add a stream.
    fn add_stream(&mut self, info: StreamInfo) -> Result<usize>;

    /// Write the header.
    fn write_header(&mut self) -> Result<()>;

    /// Write a packet.
    fn write_packet(&mut self, packet: &Packet) -> Result<()>;

    /// Write the trailer and finalize.
    fn write_trailer(&mut self) -> Result<()>;

    /// Close the muxer.
    fn close(&mut self);
}

/// Probe result for container format detection.
#[derive(Debug)]
pub struct ProbeResult {
    /// Format name.
    pub format_name: String,
    /// Confidence score (0.0 - 1.0).
    pub score: f32,
}

/// Probe a file to detect its container format.
pub fn probe<R: Read>(reader: &mut R, size: usize) -> Result<Option<ProbeResult>> {
    let mut buffer = vec![0u8; size.min(4096)];
    let bytes_read = reader.read(&mut buffer)?;
    buffer.truncate(bytes_read);

    // Check for various container signatures
    if bytes_read >= 8 {
        // Check for MP4/MOV (ftyp box)
        if &buffer[4..8] == b"ftyp" {
            return Ok(Some(ProbeResult {
                format_name: "mp4".to_string(),
                score: 1.0,
            }));
        }

        // Check for WebM/MKV (EBML header)
        if buffer.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
            let is_webm = buffer.windows(4).any(|w| w == b"webm");
            return Ok(Some(ProbeResult {
                format_name: if is_webm { "webm" } else { "mkv" }.to_string(),
                score: 0.9,
            }));
        }

        // Check for MPEG-TS (sync byte pattern)
        if buffer[0] == 0x47 && (buffer.len() < 188 || buffer[188] == 0x47) {
            return Ok(Some(ProbeResult {
                format_name: "mpegts".to_string(),
                score: 0.8,
            }));
        }
    }

    // Check for MP3 (sync word)
    if bytes_read >= 2 && buffer[0] == 0xFF && (buffer[1] & 0xE0) == 0xE0 {
        return Ok(Some(ProbeResult {
            format_name: "mp3".to_string(),
            score: 0.7,
        }));
    }

    Ok(None)
}
