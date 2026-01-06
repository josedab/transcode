//! OpenEXR error types

use thiserror::Error;

/// Result type for OpenEXR operations
pub type Result<T> = std::result::Result<T, ExrError>;

/// OpenEXR error types
#[derive(Error, Debug)]
pub enum ExrError {
    /// Invalid magic number
    #[error("Invalid EXR magic number")]
    InvalidMagic,

    /// Invalid version
    #[error("Unsupported EXR version: {version}")]
    UnsupportedVersion { version: u32 },

    /// Invalid header
    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    /// Invalid channel
    #[error("Invalid channel: {0}")]
    InvalidChannel(String),

    /// Invalid compression
    #[error("Unsupported compression: {0}")]
    UnsupportedCompression(u8),

    /// Compression error
    #[error("Compression error: {0}")]
    CompressionError(String),

    /// Decompression error
    #[error("Decompression error: {0}")]
    DecompressionError(String),

    /// Invalid data window
    #[error("Invalid data window")]
    InvalidDataWindow,

    /// Invalid display window
    #[error("Invalid display window")]
    InvalidDisplayWindow,

    /// Missing required attribute
    #[error("Missing required attribute: {0}")]
    MissingAttribute(String),

    /// Invalid attribute type
    #[error("Invalid attribute type: expected {expected}, got {actual}")]
    InvalidAttributeType { expected: String, actual: String },

    /// Insufficient data
    #[error("Insufficient data: needed {needed} bytes, got {available}")]
    InsufficientData { needed: usize, available: usize },

    /// Buffer too small
    #[error("Buffer too small: needed {needed} bytes, got {available}")]
    BufferTooSmall { needed: usize, available: usize },

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Encoding error
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Decoding error
    #[error("Decoding error: {0}")]
    DecodingError(String),

    /// Invalid pixel type
    #[error("Invalid pixel type: {0}")]
    InvalidPixelType(u32),

    /// Invalid tile description
    #[error("Invalid tile description")]
    InvalidTileDescription,

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ExrError::InvalidMagic;
        assert_eq!(format!("{}", err), "Invalid EXR magic number");

        let err = ExrError::UnsupportedVersion { version: 3 };
        assert!(format!("{}", err).contains("3"));

        let err = ExrError::MissingAttribute("compression".into());
        assert!(format!("{}", err).contains("compression"));
    }
}
