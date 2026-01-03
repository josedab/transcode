//! Error types for the WebP decoder

use std::io;
use thiserror::Error;

/// Result type for WebP operations
pub type Result<T> = std::result::Result<T, WebPError>;

/// Errors that can occur during WebP decoding
#[derive(Error, Debug)]
pub enum WebPError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Invalid RIFF format
    #[error("Invalid RIFF format: {0}")]
    InvalidRiff(String),

    /// Invalid WebP format
    #[error("Invalid WebP format: {0}")]
    InvalidFormat(String),

    /// Invalid VP8 bitstream
    #[error("Invalid VP8 bitstream: {0}")]
    InvalidVp8(String),

    /// Invalid VP8L bitstream
    #[error("Invalid VP8L bitstream: {0}")]
    InvalidVp8l(String),

    /// Invalid alpha channel data
    #[error("Invalid alpha data: {0}")]
    InvalidAlpha(String),

    /// Invalid animation data
    #[error("Invalid animation data: {0}")]
    InvalidAnimation(String),

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// Buffer too small
    #[error("Buffer too small: expected {expected}, got {actual}")]
    BufferTooSmall { expected: usize, actual: usize },

    /// Unexpected end of data
    #[error("Unexpected end of data")]
    UnexpectedEof,

    /// Checksum mismatch
    #[error("Checksum mismatch")]
    ChecksumMismatch,

    /// Invalid metadata
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),
}

impl From<WebPError> for io::Error {
    fn from(err: WebPError) -> Self {
        io::Error::new(io::ErrorKind::InvalidData, err)
    }
}
