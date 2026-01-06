//! Error types for AVI container

use std::fmt;
use std::io;

/// Result type for AVI operations
pub type Result<T> = std::result::Result<T, AviError>;

/// Errors that can occur during AVI operations
#[derive(Debug)]
pub enum AviError {
    /// IO error during read/write
    Io(io::Error),
    /// Invalid RIFF header
    InvalidRiff,
    /// Invalid AVI signature
    InvalidAvi,
    /// Invalid chunk structure
    InvalidChunk {
        id: [u8; 4],
        message: String,
    },
    /// Missing required chunk
    MissingChunk(&'static str),
    /// Invalid stream index
    InvalidStream(u32),
    /// Insufficient data for operation
    InsufficientData {
        needed: usize,
        available: usize,
    },
    /// Unsupported feature
    Unsupported(String),
}

impl fmt::Display for AviError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AviError::Io(e) => write!(f, "IO error: {}", e),
            AviError::InvalidRiff => write!(f, "Invalid RIFF header"),
            AviError::InvalidAvi => write!(f, "Invalid AVI signature"),
            AviError::InvalidChunk { id, message } => {
                let id_str = String::from_utf8_lossy(id);
                write!(f, "Invalid chunk '{}': {}", id_str, message)
            }
            AviError::MissingChunk(name) => write!(f, "Missing required chunk: {}", name),
            AviError::InvalidStream(idx) => write!(f, "Invalid stream index: {}", idx),
            AviError::InsufficientData { needed, available } => {
                write!(
                    f,
                    "Insufficient data: need {} bytes, have {}",
                    needed, available
                )
            }
            AviError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for AviError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AviError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for AviError {
    fn from(e: io::Error) -> Self {
        AviError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AviError::InvalidRiff;
        assert!(err.to_string().contains("RIFF"));

        let err = AviError::InvalidChunk {
            id: *b"test",
            message: "bad data".into(),
        };
        assert!(err.to_string().contains("test"));

        let err = AviError::InsufficientData {
            needed: 100,
            available: 50,
        };
        assert!(err.to_string().contains("100"));
    }
}
