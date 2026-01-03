//! HDR processing errors.
//!
//! This module defines error types specific to HDR processing operations.

use thiserror::Error;

/// Errors from HDR processing operations.
#[derive(Error, Debug)]
pub enum HdrError {
    /// Invalid or corrupted metadata
    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),

    /// Unsupported conversion between formats or color spaces
    #[error("unsupported conversion: {0}")]
    UnsupportedConversion(String),

    /// Processing error during frame conversion
    #[error("processing error: {0}")]
    Processing(String),

    /// Invalid configuration parameters
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Color space conversion error
    #[error("color space error: {0}")]
    ColorSpace(String),

    /// Transfer function error
    #[error("transfer function error: {0}")]
    TransferFunction(String),

    /// Gamut mapping error
    #[error("gamut mapping error: {0}")]
    GamutMapping(String),

    /// Tone mapping error
    #[error("tone mapping error: {0}")]
    ToneMapping(String),

    /// Metadata parsing error
    #[error("metadata parsing error: {0}")]
    MetadataParsing(String),

    /// HDR10+ specific error
    #[error("HDR10+ error: {0}")]
    Hdr10Plus(String),

    /// Dolby Vision specific error
    #[error("Dolby Vision error: {0}")]
    DolbyVision(String),

    /// Buffer size mismatch
    #[error("buffer size error: expected {expected}, got {actual}")]
    BufferSize {
        expected: usize,
        actual: usize,
    },

    /// Invalid pixel format
    #[error("invalid pixel format: {0}")]
    InvalidPixelFormat(String),

    /// Out of range value
    #[error("value out of range: {value} (expected {min} to {max})")]
    OutOfRange {
        value: f64,
        min: f64,
        max: f64,
    },

    /// Feature not supported
    #[error("feature not supported: {0}")]
    NotSupported(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl HdrError {
    /// Create an invalid metadata error.
    pub fn invalid_metadata(msg: impl Into<String>) -> Self {
        Self::InvalidMetadata(msg.into())
    }

    /// Create an unsupported conversion error.
    pub fn unsupported_conversion(msg: impl Into<String>) -> Self {
        Self::UnsupportedConversion(msg.into())
    }

    /// Create a processing error.
    pub fn processing(msg: impl Into<String>) -> Self {
        Self::Processing(msg.into())
    }

    /// Create an invalid configuration error.
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }

    /// Create a buffer size error.
    pub fn buffer_size(expected: usize, actual: usize) -> Self {
        Self::BufferSize { expected, actual }
    }

    /// Create an out of range error.
    pub fn out_of_range(value: f64, min: f64, max: f64) -> Self {
        Self::OutOfRange { value, min, max }
    }

    /// Check if this is a recoverable error.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            HdrError::InvalidMetadata(_)
                | HdrError::OutOfRange { .. }
                | HdrError::BufferSize { .. }
        )
    }

    /// Check if this is a configuration error.
    pub fn is_config_error(&self) -> bool {
        matches!(
            self,
            HdrError::InvalidConfig(_)
                | HdrError::UnsupportedConversion(_)
                | HdrError::NotSupported(_)
        )
    }
}

/// Result type for HDR operations.
pub type Result<T> = std::result::Result<T, HdrError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = HdrError::InvalidMetadata("test error".to_string());
        assert_eq!(err.to_string(), "invalid metadata: test error");

        let err = HdrError::BufferSize {
            expected: 100,
            actual: 50,
        };
        assert_eq!(err.to_string(), "buffer size error: expected 100, got 50");

        let err = HdrError::OutOfRange {
            value: 1.5,
            min: 0.0,
            max: 1.0,
        };
        assert_eq!(err.to_string(), "value out of range: 1.5 (expected 0 to 1)");
    }

    #[test]
    fn test_error_constructors() {
        let err = HdrError::invalid_metadata("test");
        assert!(matches!(err, HdrError::InvalidMetadata(_)));

        let err = HdrError::unsupported_conversion("test");
        assert!(matches!(err, HdrError::UnsupportedConversion(_)));

        let err = HdrError::processing("test");
        assert!(matches!(err, HdrError::Processing(_)));

        let err = HdrError::buffer_size(100, 50);
        assert!(matches!(err, HdrError::BufferSize { .. }));

        let err = HdrError::out_of_range(1.5, 0.0, 1.0);
        assert!(matches!(err, HdrError::OutOfRange { .. }));
    }

    #[test]
    fn test_is_recoverable() {
        assert!(HdrError::InvalidMetadata("test".into()).is_recoverable());
        assert!(HdrError::OutOfRange { value: 1.5, min: 0.0, max: 1.0 }.is_recoverable());
        assert!(HdrError::BufferSize { expected: 100, actual: 50 }.is_recoverable());

        assert!(!HdrError::UnsupportedConversion("test".into()).is_recoverable());
        assert!(!HdrError::NotSupported("test".into()).is_recoverable());
    }

    #[test]
    fn test_is_config_error() {
        assert!(HdrError::InvalidConfig("test".into()).is_config_error());
        assert!(HdrError::UnsupportedConversion("test".into()).is_config_error());
        assert!(HdrError::NotSupported("test".into()).is_config_error());

        assert!(!HdrError::InvalidMetadata("test".into()).is_config_error());
        assert!(!HdrError::Processing("test".into()).is_config_error());
    }
}
