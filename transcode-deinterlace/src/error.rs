//! Error types for deinterlacing operations.
//!
//! This module provides comprehensive error handling for all deinterlacing
//! filters and detection algorithms.

use thiserror::Error;

/// Error type for deinterlacing operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum DeinterlaceError {
    /// Invalid frame dimensions for deinterlacing.
    #[error("Invalid frame dimensions: {width}x{height} (minimum 2x2 required)")]
    InvalidDimensions { width: u32, height: u32 },

    /// Unsupported pixel format for the deinterlacer.
    #[error("Unsupported pixel format: {format}")]
    UnsupportedFormat { format: String },

    /// Insufficient frames for temporal filtering.
    #[error("Insufficient frames: need {needed}, have {available}")]
    InsufficientFrames { needed: usize, available: usize },

    /// Frame mismatch in temporal buffer (different dimensions or format).
    #[error("Frame mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}")]
    FrameMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },

    /// Invalid field configuration.
    #[error("Invalid field configuration: {message}")]
    InvalidFieldConfig { message: String },

    /// Buffer access error.
    #[error("Buffer access error: {message}")]
    BufferError { message: String },

    /// Invalid telecine pattern.
    #[error("Invalid telecine pattern: {pattern}")]
    InvalidTelecinePattern { pattern: String },

    /// Detection failed.
    #[error("Detection failed: {message}")]
    DetectionFailed { message: String },

    /// Internal algorithm error.
    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Result type for deinterlacing operations.
pub type Result<T> = std::result::Result<T, DeinterlaceError>;

impl DeinterlaceError {
    /// Create an invalid dimensions error.
    pub fn invalid_dimensions(width: u32, height: u32) -> Self {
        Self::InvalidDimensions { width, height }
    }

    /// Create an unsupported format error.
    pub fn unsupported_format(format: impl Into<String>) -> Self {
        Self::UnsupportedFormat {
            format: format.into(),
        }
    }

    /// Create an insufficient frames error.
    pub fn insufficient_frames(needed: usize, available: usize) -> Self {
        Self::InsufficientFrames { needed, available }
    }

    /// Create a frame mismatch error.
    pub fn frame_mismatch(
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    ) -> Self {
        Self::FrameMismatch {
            expected_width,
            expected_height,
            actual_width,
            actual_height,
        }
    }

    /// Create an invalid field config error.
    pub fn invalid_field_config(message: impl Into<String>) -> Self {
        Self::InvalidFieldConfig {
            message: message.into(),
        }
    }

    /// Create a buffer error.
    pub fn buffer_error(message: impl Into<String>) -> Self {
        Self::BufferError {
            message: message.into(),
        }
    }

    /// Create an internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DeinterlaceError::invalid_dimensions(100, 1);
        assert!(err.to_string().contains("100x1"));

        let err = DeinterlaceError::unsupported_format("unknown");
        assert!(err.to_string().contains("unknown"));

        let err = DeinterlaceError::insufficient_frames(3, 1);
        assert!(err.to_string().contains("need 3"));
        assert!(err.to_string().contains("have 1"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = DeinterlaceError::invalid_dimensions(100, 100);
        let err2 = DeinterlaceError::invalid_dimensions(100, 100);
        let err3 = DeinterlaceError::invalid_dimensions(200, 200);

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}
