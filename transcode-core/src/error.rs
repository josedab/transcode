//! Error types for the Transcode library.
//!
//! This module provides a comprehensive error hierarchy for all components of the library.

use thiserror::Error;

/// Main error type for the Transcode library.
#[derive(Error, Debug)]
pub enum Error {
    /// Container format errors (demuxing/muxing).
    #[error("Container error: {0}")]
    Container(#[from] ContainerError),

    /// Codec errors (encoding/decoding).
    #[error("Codec error: {0}")]
    Codec(#[from] CodecError),

    /// Bitstream parsing errors.
    #[error("Bitstream error: {0}")]
    Bitstream(#[from] BitstreamError),

    /// I/O errors.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid parameter provided.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Unsupported feature or format.
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Resource exhausted (memory, buffers, etc.).
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Operation was cancelled.
    #[error("Operation cancelled")]
    Cancelled,

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// End of stream reached.
    #[error("End of stream")]
    EndOfStream,

    /// Buffer too small for operation.
    #[error("Buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall { needed: usize, available: usize },
}

/// Container format errors.
#[derive(Error, Debug)]
pub enum ContainerError {
    /// Invalid or corrupted container structure.
    #[error("Invalid container structure: {0}")]
    InvalidStructure(String),

    /// Unknown or unsupported container format.
    #[error("Unknown container format")]
    UnknownFormat,

    /// Missing required atom/box/element.
    #[error("Missing required element: {0}")]
    MissingElement(String),

    /// Invalid atom/box size.
    #[error("Invalid element size at offset {offset}: {message}")]
    InvalidSize { offset: u64, message: String },

    /// Recursion limit exceeded during parsing.
    #[error("Recursion limit exceeded at depth {depth}")]
    RecursionLimit { depth: u32 },

    /// Timeout during parsing operation.
    #[error("Parsing timeout exceeded")]
    Timeout,

    /// Stream not found in container.
    #[error("Stream {index} not found")]
    StreamNotFound { index: u32 },

    /// Seek operation failed.
    #[error("Seek failed: {0}")]
    SeekFailed(String),

    /// Track configuration error.
    #[error("Track configuration error: {0}")]
    TrackConfig(String),

    /// Generic container error message.
    #[error("{0}")]
    Other(String),
}

impl From<String> for ContainerError {
    fn from(s: String) -> Self {
        ContainerError::Other(s)
    }
}

impl From<&str> for ContainerError {
    fn from(s: &str) -> Self {
        ContainerError::Other(s.to_string())
    }
}

/// Codec errors.
#[derive(Error, Debug)]
pub enum CodecError {
    /// Unsupported codec profile.
    #[error("Unsupported profile: {0}")]
    UnsupportedProfile(String),

    /// Unsupported codec level.
    #[error("Unsupported level: {0}")]
    UnsupportedLevel(String),

    /// Bitstream corruption detected.
    #[error("Bitstream corruption at offset {offset}")]
    BitstreamCorruption { offset: u64 },

    /// Missing reference frame.
    #[error("Missing reference frame: {frame_num}")]
    MissingReference { frame_num: u32 },

    /// Decoder not initialized.
    #[error("Decoder not initialized")]
    NotInitialized,

    /// Encoder configuration error.
    #[error("Encoder configuration error: {0}")]
    EncoderConfig(String),

    /// Decoder configuration error.
    #[error("Decoder configuration error: {0}")]
    DecoderConfig(String),

    /// Invalid NAL unit.
    #[error("Invalid NAL unit: {0}")]
    InvalidNalUnit(String),

    /// Invalid parameter set.
    #[error("Invalid parameter set: {0}")]
    InvalidParameterSet(String),

    /// Slice parsing error.
    #[error("Slice error: {0}")]
    SliceError(String),

    /// Frame dimensions exceed limits.
    #[error("Frame dimensions {width}x{height} exceed maximum {max_width}x{max_height}")]
    DimensionsExceeded {
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },

    /// Resource limit exceeded (parameter sets, reference frames, etc.).
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Generic codec error message.
    #[error("{0}")]
    Other(String),
}

impl From<String> for CodecError {
    fn from(s: String) -> Self {
        CodecError::Other(s)
    }
}

impl From<&str> for CodecError {
    fn from(s: &str) -> Self {
        CodecError::Other(s.to_string())
    }
}

/// Bitstream parsing errors.
#[derive(Error, Debug)]
pub enum BitstreamError {
    /// Unexpected end of bitstream.
    #[error("Unexpected end of bitstream")]
    UnexpectedEnd,

    /// Invalid start code.
    #[error("Invalid start code at offset {offset}")]
    InvalidStartCode { offset: u64 },

    /// Invalid syntax element value.
    #[error("Invalid syntax element: {element} = {value}")]
    InvalidSyntax { element: String, value: i64 },

    /// Exp-Golomb decoding error.
    #[error("Exp-Golomb decoding error: value too large")]
    ExpGolombOverflow,

    /// Bit alignment error.
    #[error("Bit alignment error")]
    AlignmentError,

    /// Generic bitstream error message.
    #[error("{0}")]
    Other(String),
}

impl From<String> for BitstreamError {
    fn from(s: String) -> Self {
        BitstreamError::Other(s)
    }
}

impl From<&str> for BitstreamError {
    fn from(s: &str) -> Self {
        BitstreamError::Other(s.to_string())
    }
}

/// Result type alias using our Error type.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create an invalid parameter error.
    pub fn invalid_param(msg: impl Into<String>) -> Self {
        Error::InvalidParameter(msg.into())
    }

    /// Create an unsupported error.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }

    /// Check if this is an end-of-stream error.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        matches!(self, Error::EndOfStream)
    }

    /// Check if this error is recoverable (can continue processing).
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Error::Codec(CodecError::BitstreamCorruption { .. })
                | Error::Codec(CodecError::MissingReference { .. })
                | Error::Bitstream(BitstreamError::InvalidSyntax { .. })
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::InvalidParameter("test parameter".into());
        assert_eq!(err.to_string(), "Invalid parameter: test parameter");
    }

    #[test]
    fn test_container_error_conversion() {
        let container_err = ContainerError::UnknownFormat;
        let err: Error = container_err.into();
        assert!(matches!(err, Error::Container(ContainerError::UnknownFormat)));
    }

    #[test]
    fn test_is_eof() {
        assert!(Error::EndOfStream.is_eof());
        assert!(!Error::Cancelled.is_eof());
    }

    #[test]
    fn test_is_recoverable() {
        let recoverable = Error::Codec(CodecError::BitstreamCorruption { offset: 0 });
        assert!(recoverable.is_recoverable());

        let not_recoverable = Error::EndOfStream;
        assert!(!not_recoverable.is_recoverable());
    }
}
