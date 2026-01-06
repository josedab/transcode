//! Error types for CineForm codec

use std::fmt;
use std::io;

/// Result type for CineForm operations
pub type Result<T> = std::result::Result<T, CineformError>;

/// Errors that can occur during CineForm encoding/decoding
#[derive(Debug)]
pub enum CineformError {
    /// IO error during read/write
    Io(io::Error),
    /// Invalid frame signature
    InvalidSignature([u8; 4]),
    /// Invalid frame header
    InvalidHeader(String),
    /// Invalid tag in bitstream
    InvalidTag {
        tag: u16,
        value: u16,
    },
    /// Bitstream decoding error
    BitstreamError(String),
    /// Wavelet transform error
    WaveletError(String),
    /// Quantization error
    QuantizationError(String),
    /// Insufficient data for operation
    InsufficientData {
        needed: usize,
        available: usize,
    },
    /// Unsupported feature
    Unsupported(String),
}

impl fmt::Display for CineformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CineformError::Io(e) => write!(f, "IO error: {}", e),
            CineformError::InvalidSignature(sig) => {
                write!(f, "Invalid CineForm signature: {:?}", sig)
            }
            CineformError::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
            CineformError::InvalidTag { tag, value } => {
                write!(f, "Invalid tag: 0x{:04X} = 0x{:04X}", tag, value)
            }
            CineformError::BitstreamError(msg) => write!(f, "Bitstream error: {}", msg),
            CineformError::WaveletError(msg) => write!(f, "Wavelet error: {}", msg),
            CineformError::QuantizationError(msg) => write!(f, "Quantization error: {}", msg),
            CineformError::InsufficientData { needed, available } => {
                write!(
                    f,
                    "Insufficient data: need {} bytes, have {}",
                    needed, available
                )
            }
            CineformError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for CineformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CineformError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for CineformError {
    fn from(e: io::Error) -> Self {
        CineformError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CineformError::InvalidHeader("test".into());
        assert!(err.to_string().contains("Invalid header"));

        let err = CineformError::InsufficientData {
            needed: 100,
            available: 50,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_invalid_tag_display() {
        let err = CineformError::InvalidTag {
            tag: 0x1234,
            value: 0x5678,
        };
        let s = err.to_string();
        assert!(s.contains("1234"));
        assert!(s.contains("5678"));
    }
}
