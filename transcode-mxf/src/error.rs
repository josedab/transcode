//! Error types for MXF container

use std::fmt;
use std::io;

/// Result type for MXF operations
pub type Result<T> = std::result::Result<T, MxfError>;

/// Errors that can occur during MXF operations
#[derive(Debug)]
pub enum MxfError {
    /// IO error during read/write
    Io(io::Error),
    /// Invalid MXF file structure
    InvalidMxf(String),
    /// Invalid KLV structure
    InvalidKlv {
        message: String,
        offset: u64,
    },
    /// Unknown or unsupported Universal Label
    UnknownUL([u8; 16]),
    /// Missing required metadata
    MissingMetadata(&'static str),
    /// Invalid partition structure
    InvalidPartition(String),
    /// Insufficient data for operation
    InsufficientData {
        needed: usize,
        available: usize,
    },
    /// Unsupported feature
    Unsupported(String),
    /// Invalid track index
    InvalidTrack(u32),
    /// BER encoding error
    BerError(String),
}

impl fmt::Display for MxfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MxfError::Io(e) => write!(f, "IO error: {}", e),
            MxfError::InvalidMxf(msg) => write!(f, "Invalid MXF: {}", msg),
            MxfError::InvalidKlv { message, offset } => {
                write!(f, "Invalid KLV at offset {}: {}", offset, message)
            }
            MxfError::UnknownUL(ul) => {
                write!(f, "Unknown Universal Label: {:02x?}", ul)
            }
            MxfError::MissingMetadata(name) => {
                write!(f, "Missing required metadata: {}", name)
            }
            MxfError::InvalidPartition(msg) => {
                write!(f, "Invalid partition: {}", msg)
            }
            MxfError::InsufficientData { needed, available } => {
                write!(
                    f,
                    "Insufficient data: need {} bytes, have {}",
                    needed, available
                )
            }
            MxfError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
            MxfError::InvalidTrack(idx) => write!(f, "Invalid track index: {}", idx),
            MxfError::BerError(msg) => write!(f, "BER encoding error: {}", msg),
        }
    }
}

impl std::error::Error for MxfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MxfError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for MxfError {
    fn from(e: io::Error) -> Self {
        MxfError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MxfError::InvalidMxf("bad header".into());
        assert!(err.to_string().contains("Invalid MXF"));

        let err = MxfError::InvalidKlv {
            message: "truncated".into(),
            offset: 1000,
        };
        assert!(err.to_string().contains("1000"));

        let err = MxfError::InsufficientData {
            needed: 100,
            available: 50,
        };
        assert!(err.to_string().contains("100"));
    }
}
