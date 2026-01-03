//! VP9 codec error types.
//!
//! This module provides VP9-specific error types for decoding operations.

use thiserror::Error;

/// VP9-specific error types.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum Vp9Error {
    /// Invalid frame sync code (should be 0x498342).
    #[error("Invalid sync code: expected 0x498342, got {0:#06x}")]
    InvalidSyncCode(u32),

    /// Invalid frame marker (should be 2).
    #[error("Invalid frame marker: expected 2, got {0}")]
    InvalidFrameMarker(u8),

    /// Unsupported VP9 profile.
    #[error("Unsupported profile: {0}")]
    UnsupportedProfile(u8),

    /// Unsupported bit depth.
    #[error("Unsupported bit depth: {0}")]
    UnsupportedBitDepth(u8),

    /// Unsupported color space.
    #[error("Unsupported color space: {0}")]
    UnsupportedColorSpace(u8),

    /// Invalid frame dimensions.
    #[error("Invalid frame dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },

    /// Frame dimensions too large.
    #[error("Frame dimensions {width}x{height} exceed maximum {max_width}x{max_height}")]
    DimensionsTooLarge {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
        /// Maximum allowed width.
        max_width: u32,
        /// Maximum allowed height.
        max_height: u32,
    },

    /// Invalid reference frame index.
    #[error("Invalid reference frame index: {0}")]
    InvalidRefFrameIndex(u8),

    /// Missing reference frame.
    #[error("Missing reference frame: {0}")]
    MissingRefFrame(u8),

    /// Invalid segmentation parameters.
    #[error("Invalid segmentation parameters: {0}")]
    InvalidSegmentation(String),

    /// Invalid loop filter parameters.
    #[error("Invalid loop filter parameters: {0}")]
    InvalidLoopFilter(String),

    /// Invalid quantization parameters.
    #[error("Invalid quantization parameters: {0}")]
    InvalidQuantization(String),

    /// Invalid tile configuration.
    #[error("Invalid tile configuration: {0}")]
    InvalidTileConfig(String),

    /// Boolean entropy decoding error.
    #[error("Boolean decoder error: {0}")]
    BoolDecoderError(String),

    /// Bitstream exhausted unexpectedly.
    #[error("Unexpected end of bitstream")]
    UnexpectedEndOfStream,

    /// Invalid syntax element.
    #[error("Invalid syntax element: {element} = {value}")]
    InvalidSyntax {
        /// Syntax element name.
        element: String,
        /// Invalid value.
        value: i64,
    },

    /// Transform size not allowed.
    #[error("Transform size {0}x{0} not allowed")]
    InvalidTransformSize(u8),

    /// Invalid partition type.
    #[error("Invalid partition type: {0}")]
    InvalidPartitionType(u8),

    /// Invalid prediction mode.
    #[error("Invalid prediction mode: {0}")]
    InvalidPredictionMode(u8),

    /// Invalid interpolation filter.
    #[error("Invalid interpolation filter: {0}")]
    InvalidInterpFilter(u8),

    /// Decoder not initialized.
    #[error("Decoder not initialized")]
    NotInitialized,

    /// Decoder configuration error.
    #[error("Decoder configuration error: {0}")]
    ConfigError(String),

    /// Superframe parsing error.
    #[error("Superframe parsing error: {0}")]
    SuperframeError(String),

    /// Compressed header parsing error.
    #[error("Compressed header error: {0}")]
    CompressedHeaderError(String),

    /// Uncompressed header parsing error.
    #[error("Uncompressed header error: {0}")]
    UncompressedHeaderError(String),
}

/// VP9 codec result type.
pub type Result<T> = std::result::Result<T, Vp9Error>;

impl Vp9Error {
    /// Create an invalid syntax error.
    pub fn invalid_syntax(element: impl Into<String>, value: i64) -> Self {
        Self::InvalidSyntax {
            element: element.into(),
            value,
        }
    }

    /// Check if this error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::MissingRefFrame(_)
                | Self::BoolDecoderError(_)
                | Self::InvalidSyntax { .. }
        )
    }
}

impl From<Vp9Error> for transcode_core::Error {
    fn from(err: Vp9Error) -> Self {
        transcode_core::Error::Codec(transcode_core::error::CodecError::Other(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Vp9Error::InvalidSyncCode(0x123456);
        assert_eq!(err.to_string(), "Invalid sync code: expected 0x498342, got 0x123456");
    }

    #[test]
    fn test_invalid_dimensions() {
        let err = Vp9Error::InvalidDimensions { width: 0, height: 0 };
        assert_eq!(err.to_string(), "Invalid frame dimensions: 0x0");
    }

    #[test]
    fn test_dimensions_too_large() {
        let err = Vp9Error::DimensionsTooLarge {
            width: 8192,
            height: 4320,
            max_width: 4096,
            max_height: 2160,
        };
        assert!(err.to_string().contains("8192x4320"));
    }

    #[test]
    fn test_invalid_syntax() {
        let err = Vp9Error::invalid_syntax("base_q_idx", 256);
        assert!(err.to_string().contains("base_q_idx"));
    }

    #[test]
    fn test_is_recoverable() {
        assert!(Vp9Error::MissingRefFrame(1).is_recoverable());
        assert!(!Vp9Error::NotInitialized.is_recoverable());
        assert!(!Vp9Error::InvalidSyncCode(0).is_recoverable());
    }

    #[test]
    fn test_conversion_to_core_error() {
        let vp9_err = Vp9Error::UnsupportedProfile(4);
        let core_err: transcode_core::Error = vp9_err.into();
        assert!(core_err.to_string().contains("Unsupported profile"));
    }
}
