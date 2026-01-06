//! ALAC codec error types.

use thiserror::Error;

/// ALAC codec error.
#[derive(Debug, Error)]
pub enum AlacError {
    /// Invalid magic cookie.
    #[error("Invalid ALAC magic cookie")]
    InvalidMagicCookie,

    /// Invalid frame header.
    #[error("Invalid frame header: {0}")]
    InvalidFrameHeader(String),

    /// Unsupported bit depth.
    #[error("Unsupported bit depth: {0}")]
    UnsupportedBitDepth(u8),

    /// Unsupported channel configuration.
    #[error("Unsupported channel configuration: {0}")]
    UnsupportedChannels(u8),

    /// Invalid sample count.
    #[error("Invalid sample count")]
    InvalidSampleCount,

    /// Decoding error.
    #[error("ALAC decode error: {0}")]
    DecodeError(String),

    /// Encoding error.
    #[error("ALAC encode error: {0}")]
    EncodeError(String),

    /// Bitstream error.
    #[error("Bitstream error: {0}")]
    BitstreamError(String),

    /// End of stream.
    #[error("End of ALAC stream")]
    EndOfStream,

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// ALAC result type.
pub type Result<T> = std::result::Result<T, AlacError>;
