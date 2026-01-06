//! FFV1 codec error types.

use thiserror::Error;

/// FFV1 codec error.
#[derive(Debug, Error)]
pub enum Ffv1Error {
    /// Invalid configuration record.
    #[error("Invalid FFV1 configuration record")]
    InvalidConfigRecord,

    /// Invalid frame header.
    #[error("Invalid frame header: {0}")]
    InvalidFrameHeader(String),

    /// Unsupported version.
    #[error("Unsupported FFV1 version: {0}")]
    UnsupportedVersion(u8),

    /// Unsupported color space.
    #[error("Unsupported color space: {0}")]
    UnsupportedColorSpace(u8),

    /// Invalid slice.
    #[error("Invalid slice: {0}")]
    InvalidSlice(String),

    /// Range coder error.
    #[error("Range coder error: {0}")]
    RangeCoderError(String),

    /// Decoding error.
    #[error("FFV1 decode error: {0}")]
    DecodeError(String),

    /// Encoding error.
    #[error("FFV1 encode error: {0}")]
    EncodeError(String),

    /// End of stream.
    #[error("End of FFV1 stream")]
    EndOfStream,

    /// CRC mismatch.
    #[error("CRC mismatch: expected {expected:08x}, got {actual:08x}")]
    CrcMismatch {
        /// Expected CRC value.
        expected: u32,
        /// Actual computed CRC value.
        actual: u32,
    },

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// FFV1 result type.
pub type Result<T> = std::result::Result<T, Ffv1Error>;
