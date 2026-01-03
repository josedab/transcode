//! Error types for frame rate conversion.

use thiserror::Error;

/// Frame rate conversion error types.
#[derive(Error, Debug)]
pub enum FrameRateError {
    /// Invalid frame rate specified.
    #[error("Invalid frame rate: {0}")]
    InvalidFrameRate(String),

    /// Invalid conversion parameters.
    #[error("Invalid conversion parameters: {0}")]
    InvalidParameters(String),

    /// Frame dimensions mismatch.
    #[error("Frame dimensions mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}")]
    DimensionMismatch {
        /// Expected width in pixels.
        expected_width: u32,
        /// Expected height in pixels.
        expected_height: u32,
        /// Actual width in pixels.
        actual_width: u32,
        /// Actual height in pixels.
        actual_height: u32,
    },

    /// Pixel format not supported.
    #[error("Unsupported pixel format: {0}")]
    UnsupportedFormat(String),

    /// Not enough frames for operation.
    #[error("Not enough frames: need {needed}, have {available}")]
    InsufficientFrames {
        /// Number of frames needed.
        needed: usize,
        /// Number of frames available.
        available: usize,
    },

    /// Telecine pattern detection failed.
    #[error("Failed to detect telecine pattern: {0}")]
    TelecineDetectionFailed(String),

    /// Motion estimation failed.
    #[error("Motion estimation failed: {0}")]
    MotionEstimationFailed(String),

    /// Interpolation failed.
    #[error("Interpolation failed: {0}")]
    InterpolationFailed(String),

    /// Buffer allocation failed.
    #[error("Buffer allocation failed: {0}")]
    AllocationFailed(String),

    /// Invalid timestamp.
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    /// Core library error.
    #[error("Core error: {0}")]
    Core(#[from] transcode_core::Error),
}

/// Result type for frame rate conversion operations.
pub type Result<T> = std::result::Result<T, FrameRateError>;

impl FrameRateError {
    /// Create an invalid frame rate error.
    pub fn invalid_frame_rate(msg: impl Into<String>) -> Self {
        Self::InvalidFrameRate(msg.into())
    }

    /// Create an invalid parameters error.
    pub fn invalid_params(msg: impl Into<String>) -> Self {
        Self::InvalidParameters(msg.into())
    }

    /// Create an unsupported format error.
    pub fn unsupported_format(msg: impl Into<String>) -> Self {
        Self::UnsupportedFormat(msg.into())
    }

    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::InsufficientFrames { .. } | Self::TelecineDetectionFailed(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FrameRateError::InvalidFrameRate("0/1 fps".into());
        assert_eq!(err.to_string(), "Invalid frame rate: 0/1 fps");
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = FrameRateError::DimensionMismatch {
            expected_width: 1920,
            expected_height: 1080,
            actual_width: 1280,
            actual_height: 720,
        };
        assert!(err.to_string().contains("1920x1080"));
        assert!(err.to_string().contains("1280x720"));
    }

    #[test]
    fn test_is_recoverable() {
        let recoverable = FrameRateError::InsufficientFrames {
            needed: 2,
            available: 1,
        };
        assert!(recoverable.is_recoverable());

        let not_recoverable = FrameRateError::InvalidFrameRate("test".into());
        assert!(!not_recoverable.is_recoverable());
    }
}
