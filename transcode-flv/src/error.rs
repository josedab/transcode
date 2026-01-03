//! FLV-specific error types.
//!
//! This module provides error types specific to FLV container parsing and writing.

use thiserror::Error;

/// FLV-specific error types.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum FlvError {
    /// Invalid FLV signature (expected "FLV").
    #[error("Invalid FLV signature: expected 'FLV', got '{0}'")]
    InvalidSignature(String),

    /// Invalid FLV version.
    #[error("Invalid FLV version: {0}")]
    InvalidVersion(u8),

    /// Invalid tag type.
    #[error("Invalid tag type: {0}")]
    InvalidTagType(u8),

    /// Invalid tag size.
    #[error("Invalid tag size at offset {offset}: {message}")]
    InvalidTagSize {
        /// Byte offset where the invalid size was found.
        offset: u64,
        /// Description of the size error.
        message: String,
    },

    /// Invalid audio format.
    #[error("Invalid or unsupported audio format: {0}")]
    InvalidAudioFormat(u8),

    /// Invalid video codec.
    #[error("Invalid or unsupported video codec: {0}")]
    InvalidVideoCodec(u8),

    /// Invalid video frame type.
    #[error("Invalid video frame type: {0}")]
    InvalidFrameType(u8),

    /// Invalid AVC packet type.
    #[error("Invalid AVC packet type: {0}")]
    InvalidAvcPacketType(u8),

    /// Invalid HEVC packet type.
    #[error("Invalid HEVC packet type: {0}")]
    InvalidHevcPacketType(u8),

    /// Invalid AAC packet type.
    #[error("Invalid AAC packet type: {0}")]
    InvalidAacPacketType(u8),

    /// Invalid AMF data.
    #[error("Invalid AMF data: {0}")]
    InvalidAmf(String),

    /// Invalid AMF type marker.
    #[error("Invalid AMF type marker: {0}")]
    InvalidAmfType(u8),

    /// AMF string too long.
    #[error("AMF string too long: {0} bytes (max 65535)")]
    AmfStringTooLong(usize),

    /// Invalid timestamp.
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    /// Timestamp wraparound detected.
    #[error("Timestamp wraparound detected at offset {offset}")]
    TimestampWraparound {
        /// Byte offset where wraparound was detected.
        offset: u64,
    },

    /// Missing sequence header.
    #[error("Missing {codec} sequence header")]
    MissingSequenceHeader {
        /// The codec that is missing its sequence header.
        codec: String,
    },

    /// Invalid sequence header.
    #[error("Invalid {codec} sequence header: {message}")]
    InvalidSequenceHeader {
        /// The codec with the invalid sequence header.
        codec: String,
        /// Description of the error.
        message: String,
    },

    /// Unexpected end of data.
    #[error("Unexpected end of data at offset {offset}")]
    UnexpectedEnd {
        /// Byte offset where the end was encountered.
        offset: u64,
    },

    /// Previous tag size mismatch.
    #[error("Previous tag size mismatch at offset {offset}: expected {expected}, got {actual}")]
    PreviousTagSizeMismatch {
        /// Byte offset where the mismatch was found.
        offset: u64,
        /// Expected previous tag size.
        expected: u32,
        /// Actual previous tag size value.
        actual: u32,
    },

    /// Seek failed.
    #[error("Seek failed: {0}")]
    SeekFailed(String),

    /// No streams found.
    #[error("No audio or video streams found in FLV")]
    NoStreamsFound,

    /// Stream configuration mismatch.
    #[error("Stream configuration mismatch: {0}")]
    StreamConfigMismatch(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(String),

    /// Generic error.
    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for FlvError {
    fn from(err: std::io::Error) -> Self {
        FlvError::Io(err.to_string())
    }
}

impl From<String> for FlvError {
    fn from(s: String) -> Self {
        FlvError::Other(s)
    }
}

impl From<&str> for FlvError {
    fn from(s: &str) -> Self {
        FlvError::Other(s.to_string())
    }
}

/// Result type for FLV operations.
pub type Result<T> = std::result::Result<T, FlvError>;

/// Convert FlvError to transcode_core::Error.
impl From<FlvError> for transcode_core::error::Error {
    fn from(err: FlvError) -> Self {
        match &err {
            FlvError::Io(msg) => {
                transcode_core::error::Error::Io(std::io::Error::other(msg.clone()))
            }
            FlvError::InvalidSignature(_) | FlvError::InvalidVersion(_) => {
                transcode_core::error::Error::Container(
                    transcode_core::error::ContainerError::InvalidStructure(err.to_string()),
                )
            }
            FlvError::MissingSequenceHeader { .. } => transcode_core::error::Error::Container(
                transcode_core::error::ContainerError::MissingElement(err.to_string()),
            ),
            FlvError::InvalidTagSize { offset, message } => {
                transcode_core::error::Error::Container(
                    transcode_core::error::ContainerError::InvalidSize {
                        offset: *offset,
                        message: message.clone(),
                    },
                )
            }
            FlvError::SeekFailed(msg) => transcode_core::error::Error::Container(
                transcode_core::error::ContainerError::SeekFailed(msg.clone()),
            ),
            _ => transcode_core::error::Error::Container(
                transcode_core::error::ContainerError::Other(err.to_string()),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FlvError::InvalidSignature("ABC".to_string());
        assert_eq!(
            err.to_string(),
            "Invalid FLV signature: expected 'FLV', got 'ABC'"
        );
    }

    #[test]
    fn test_error_from_string() {
        let err: FlvError = "test error".into();
        assert!(matches!(err, FlvError::Other(_)));
    }

    #[test]
    fn test_invalid_tag_type() {
        let err = FlvError::InvalidTagType(99);
        assert_eq!(err.to_string(), "Invalid tag type: 99");
    }

    #[test]
    fn test_missing_sequence_header() {
        let err = FlvError::MissingSequenceHeader {
            codec: "AVC".to_string(),
        };
        assert_eq!(err.to_string(), "Missing AVC sequence header");
    }

    #[test]
    fn test_conversion_to_core_error() {
        let flv_err = FlvError::InvalidSignature("ABC".to_string());
        let core_err: transcode_core::error::Error = flv_err.into();
        assert!(matches!(
            core_err,
            transcode_core::error::Error::Container(_)
        ));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let flv_err: FlvError = io_err.into();
        assert!(matches!(flv_err, FlvError::Io(_)));
    }
}
