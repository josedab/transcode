//! Error types for DNxHD/DNxHR codec

use std::fmt;
use std::io;

/// Result type for DNxHD operations
pub type Result<T> = std::result::Result<T, DnxError>;

/// Errors that can occur during DNxHD encoding/decoding
#[derive(Debug)]
pub enum DnxError {
    /// IO error during read/write
    Io(io::Error),
    /// Invalid frame signature
    InvalidSignature([u8; 4]),
    /// Invalid frame header
    InvalidHeader(String),
    /// Unknown or unsupported profile
    UnknownProfile(u32),
    /// Bitstream decoding error
    BitstreamError(String),
    /// Huffman decoding error
    HuffmanError(String),
    /// Invalid slice data
    InvalidSlice(String),
    /// Insufficient data for operation
    InsufficientData {
        needed: usize,
        available: usize,
    },
    /// Unsupported feature
    Unsupported(String),
}

impl fmt::Display for DnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DnxError::Io(e) => write!(f, "IO error: {}", e),
            DnxError::InvalidSignature(sig) => {
                write!(f, "Invalid DNxHD signature: {:?}", sig)
            }
            DnxError::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
            DnxError::UnknownProfile(id) => write!(f, "Unknown profile ID: {}", id),
            DnxError::BitstreamError(msg) => write!(f, "Bitstream error: {}", msg),
            DnxError::HuffmanError(msg) => write!(f, "Huffman error: {}", msg),
            DnxError::InvalidSlice(msg) => write!(f, "Invalid slice: {}", msg),
            DnxError::InsufficientData { needed, available } => {
                write!(f, "Insufficient data: need {} bytes, have {}", needed, available)
            }
            DnxError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for DnxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DnxError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for DnxError {
    fn from(e: io::Error) -> Self {
        DnxError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DnxError::InvalidHeader("test".into());
        assert!(err.to_string().contains("Invalid header"));

        let err = DnxError::InsufficientData { needed: 100, available: 50 };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }
}
