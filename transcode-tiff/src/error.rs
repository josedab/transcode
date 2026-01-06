//! TIFF error types

use thiserror::Error;

/// Result type for TIFF operations
pub type Result<T> = std::result::Result<T, TiffError>;

/// TIFF error types
#[derive(Error, Debug)]
pub enum TiffError {
    /// Invalid magic number
    #[error("Invalid TIFF magic number")]
    InvalidMagic,

    /// Invalid version
    #[error("Unsupported TIFF version: {version}")]
    UnsupportedVersion { version: u16 },

    /// Unsupported compression
    #[error("Unsupported compression: {0}")]
    UnsupportedCompression(u16),

    /// Invalid IFD
    #[error("Invalid IFD: {0}")]
    InvalidIfd(String),

    /// Invalid tag
    #[error("Invalid tag: {0}")]
    InvalidTag(u16),

    /// Missing required tag
    #[error("Missing required tag: {0}")]
    MissingTag(String),

    /// Invalid data type
    #[error("Invalid data type: {0}")]
    InvalidDataType(u16),

    /// Invalid image dimensions
    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// Insufficient data
    #[error("Insufficient data: needed {needed} bytes, got {available}")]
    InsufficientData { needed: usize, available: usize },

    /// Buffer too small
    #[error("Buffer too small: needed {needed} bytes, got {available}")]
    BufferTooSmall { needed: usize, available: usize },

    /// Compression error
    #[error("Compression error: {0}")]
    CompressionError(String),

    /// Decompression error
    #[error("Decompression error: {0}")]
    DecompressionError(String),

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Encoding error
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Decoding error
    #[error("Decoding error: {0}")]
    DecodingError(String),

    /// Invalid strip/tile configuration
    #[error("Invalid strip/tile configuration")]
    InvalidStripConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = TiffError::InvalidMagic;
        assert_eq!(format!("{}", err), "Invalid TIFF magic number");

        let err = TiffError::UnsupportedVersion { version: 43 };
        assert!(format!("{}", err).contains("43"));

        let err = TiffError::MissingTag("ImageWidth".into());
        assert!(format!("{}", err).contains("ImageWidth"));
    }
}
