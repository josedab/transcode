//! Image codec error types.

use thiserror::Error;

/// Image codec errors.
#[derive(Error, Debug)]
pub enum ImageError {
    /// Invalid image header.
    #[error("Invalid image header: {0}")]
    InvalidHeader(String),

    /// Unsupported format.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Invalid image data.
    #[error("Invalid image data: {0}")]
    InvalidData(String),

    /// Unsupported color space.
    #[error("Unsupported color space: {0}")]
    UnsupportedColorSpace(String),

    /// Dimension error.
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },

    /// Decoder error.
    #[error("Decoder error: {0}")]
    DecoderError(String),

    /// Encoder error.
    #[error("Encoder error: {0}")]
    EncoderError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Out of memory.
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Truncated data.
    #[error("Truncated data: expected {expected} bytes, got {actual}")]
    TruncatedData {
        /// Expected bytes.
        expected: usize,
        /// Actual bytes.
        actual: usize,
    },

    /// Corrupted data.
    #[error("Corrupted data: {0}")]
    CorruptedData(String),
}

/// Image codec result type.
pub type Result<T> = std::result::Result<T, ImageError>;

impl From<ImageError> for transcode_core::Error {
    fn from(err: ImageError) -> Self {
        transcode_core::Error::Codec(transcode_core::error::CodecError::Other(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ImageError::InvalidHeader("bad magic".to_string());
        assert!(err.to_string().contains("bad magic"));

        let err = ImageError::InvalidDimensions {
            width: 0,
            height: 100,
        };
        assert!(err.to_string().contains("0x100"));
    }
}
