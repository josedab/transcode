//! Error types for Theora codec.

use std::fmt;

/// Theora codec error types.
#[derive(Debug, Clone)]
pub enum TheoraError {
    /// Invalid header data.
    InvalidHeader(String),
    /// Invalid comment header.
    InvalidComment(String),
    /// Invalid setup header.
    InvalidSetup(String),
    /// Unsupported version.
    UnsupportedVersion { major: u8, minor: u8 },
    /// Invalid frame data.
    InvalidFrame(String),
    /// Bitstream error.
    BitstreamError(String),
    /// Invalid dimensions.
    InvalidDimensions { width: u32, height: u32 },
    /// Decode error.
    DecodeError(String),
    /// Encode error.
    EncodeError(String),
    /// Buffer too small.
    BufferTooSmall { required: usize, available: usize },
    /// End of stream.
    EndOfStream,
    /// Not initialized.
    NotInitialized,
}

impl fmt::Display for TheoraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
            Self::InvalidComment(msg) => write!(f, "Invalid comment: {}", msg),
            Self::InvalidSetup(msg) => write!(f, "Invalid setup: {}", msg),
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "Unsupported version: {}.{}", major, minor)
            }
            Self::InvalidFrame(msg) => write!(f, "Invalid frame: {}", msg),
            Self::BitstreamError(msg) => write!(f, "Bitstream error: {}", msg),
            Self::InvalidDimensions { width, height } => {
                write!(f, "Invalid dimensions: {}x{}", width, height)
            }
            Self::DecodeError(msg) => write!(f, "Decode error: {}", msg),
            Self::EncodeError(msg) => write!(f, "Encode error: {}", msg),
            Self::BufferTooSmall {
                required,
                available,
            } => {
                write!(
                    f,
                    "Buffer too small: required {}, available {}",
                    required, available
                )
            }
            Self::EndOfStream => write!(f, "End of stream"),
            Self::NotInitialized => write!(f, "Decoder not initialized"),
        }
    }
}

impl std::error::Error for TheoraError {}

/// Result type for Theora operations.
pub type Result<T> = std::result::Result<T, TheoraError>;
