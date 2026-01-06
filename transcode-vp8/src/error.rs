//! VP8 codec error types.

use thiserror::Error;

/// VP8 codec error.
#[derive(Debug, Error)]
pub enum Vp8Error {
    /// Invalid bitstream.
    #[error("Invalid VP8 bitstream: {0}")]
    InvalidBitstream(String),

    /// Unsupported feature.
    #[error("Unsupported VP8 feature: {0}")]
    UnsupportedFeature(String),

    /// Invalid frame header.
    #[error("Invalid frame header: {0}")]
    InvalidFrameHeader(String),

    /// Invalid partition.
    #[error("Invalid partition: {0}")]
    InvalidPartition(String),

    /// Decoding error.
    #[error("VP8 decode error: {0}")]
    DecodeError(String),

    /// Encoding error.
    #[error("VP8 encode error: {0}")]
    EncodeError(String),

    /// Invalid dimensions.
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// Reference frame error.
    #[error("Reference frame error: {0}")]
    ReferenceFrameError(String),

    /// End of stream.
    #[error("End of VP8 stream")]
    EndOfStream,

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// VP8 result type.
pub type Result<T> = std::result::Result<T, Vp8Error>;
