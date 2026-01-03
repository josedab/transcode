//! # transcode-flac
//!
//! A pure Rust FLAC (Free Lossless Audio Codec) implementation with streaming support.
//!
//! ## Features
//!
//! - Full FLAC decoder with metadata parsing and all prediction types
//! - FLAC encoder with configurable compression levels (0-8)
//! - Streaming support for real-time decoding
//! - CRC validation for data integrity
//!
//! ## Example
//!
//! ```no_run
//! use transcode_flac::{FlacDecoder, FlacEncoder, CompressionLevel};
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! // Decoding
//! let file = File::open("audio.flac").unwrap();
//! let mut decoder = FlacDecoder::new(BufReader::new(file)).unwrap();
//! while let Some(frame) = decoder.next_frame().unwrap() {
//!     // Process decoded audio samples
//! }
//!
//! // Encoding
//! let output = File::create("output.flac").unwrap();
//! let mut encoder = FlacEncoder::new(output, 44100, 2, 16, CompressionLevel::Default).unwrap();
//! // encoder.encode_samples(&samples).unwrap();
//! // encoder.finish().unwrap();
//! ```

pub mod decoder;
pub mod encoder;

pub use decoder::{FlacDecoder, StreamingDecoder};
pub use encoder::{FlacEncoder, CompressionLevel};

use thiserror::Error;

/// FLAC codec error types
#[derive(Error, Debug)]
pub enum FlacError {
    #[error("Invalid FLAC stream marker")]
    InvalidMarker,

    #[error("Invalid metadata block")]
    InvalidMetadata,

    #[error("Invalid frame header")]
    InvalidFrameHeader,

    #[error("Invalid subframe")]
    InvalidSubframe,

    #[error("CRC mismatch: expected {expected:#06x}, got {actual:#06x}")]
    CrcMismatch { expected: u16, actual: u16 },

    #[error("CRC-16 mismatch: expected {expected:#06x}, got {actual:#06x}")]
    Crc16Mismatch { expected: u16, actual: u16 },

    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    #[error("Invalid Rice partition")]
    InvalidRicePartition,

    #[error("Invalid LPC order: {0}")]
    InvalidLpcOrder(u8),

    #[error("Buffer too small")]
    BufferTooSmall,

    #[error("Unexpected end of stream")]
    UnexpectedEof,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Encoding error: {0}")]
    EncodingError(String),
}

pub type Result<T> = std::result::Result<T, FlacError>;

/// FLAC metadata block types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataBlockType {
    StreamInfo,
    Padding,
    Application,
    SeekTable,
    VorbisComment,
    CueSheet,
    Picture,
    Reserved(u8),
}

impl From<u8> for MetadataBlockType {
    fn from(value: u8) -> Self {
        match value {
            0 => MetadataBlockType::StreamInfo,
            1 => MetadataBlockType::Padding,
            2 => MetadataBlockType::Application,
            3 => MetadataBlockType::SeekTable,
            4 => MetadataBlockType::VorbisComment,
            5 => MetadataBlockType::CueSheet,
            6 => MetadataBlockType::Picture,
            n => MetadataBlockType::Reserved(n),
        }
    }
}

/// STREAMINFO metadata block
#[derive(Debug, Clone)]
pub struct StreamInfo {
    /// Minimum block size in samples
    pub min_block_size: u16,
    /// Maximum block size in samples
    pub max_block_size: u16,
    /// Minimum frame size in bytes (0 = unknown)
    pub min_frame_size: u32,
    /// Maximum frame size in bytes (0 = unknown)
    pub max_frame_size: u32,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1-8)
    pub channels: u8,
    /// Bits per sample (4-32)
    pub bits_per_sample: u8,
    /// Total samples in stream (0 = unknown)
    pub total_samples: u64,
    /// MD5 signature of unencoded audio data
    pub md5_signature: [u8; 16],
}

impl Default for StreamInfo {
    fn default() -> Self {
        Self {
            min_block_size: 4096,
            max_block_size: 4096,
            min_frame_size: 0,
            max_frame_size: 0,
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
            total_samples: 0,
            md5_signature: [0; 16],
        }
    }
}

/// Vorbis comment (tag) metadata
#[derive(Debug, Clone, Default)]
pub struct VorbisComment {
    /// Vendor string
    pub vendor: String,
    /// Comment key-value pairs
    pub comments: Vec<(String, String)>,
}

/// Picture metadata block
#[derive(Debug, Clone)]
pub struct Picture {
    /// Picture type (3 = front cover, etc.)
    pub picture_type: u32,
    /// MIME type
    pub mime_type: String,
    /// Description
    pub description: String,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Color depth in bits per pixel
    pub color_depth: u32,
    /// Number of colors (for indexed images)
    pub colors: u32,
    /// Picture data
    pub data: Vec<u8>,
}

/// Seek point for random access
#[derive(Debug, Clone, Copy)]
pub struct SeekPoint {
    /// Sample number of first sample in target frame
    pub sample_number: u64,
    /// Offset in bytes from first frame header
    pub stream_offset: u64,
    /// Number of samples in target frame
    pub frame_samples: u16,
}

/// All metadata blocks from a FLAC stream
#[derive(Debug, Clone, Default)]
pub struct FlacMetadata {
    pub stream_info: Option<StreamInfo>,
    pub vorbis_comment: Option<VorbisComment>,
    pub pictures: Vec<Picture>,
    pub seek_table: Vec<SeekPoint>,
    pub application_data: Vec<(String, Vec<u8>)>,
}

/// A decoded FLAC audio frame
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// Sample rate of this frame
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Bits per sample
    pub bits_per_sample: u8,
    /// Block size (samples per channel)
    pub block_size: u32,
    /// Frame number or sample number
    pub frame_number: u64,
    /// Decoded samples (interleaved)
    pub samples: Vec<i32>,
}

/// Channel assignment types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelAssignment {
    /// Independent channels
    Independent(u8),
    /// Left/side stereo
    LeftSide,
    /// Right/side stereo
    RightSide,
    /// Mid/side stereo
    MidSide,
}

/// Subframe prediction types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubframeType {
    /// Constant value
    Constant,
    /// Verbatim (uncompressed)
    Verbatim,
    /// Fixed linear prediction
    Fixed(u8),
    /// Linear predictive coding
    Lpc(u8),
}
