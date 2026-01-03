//! WebM-specific error types.
//!
//! This module provides error types specific to WebM container parsing and writing.

use thiserror::Error;

/// WebM-specific error types.
#[derive(Error, Debug)]
pub enum WebmError {
    /// Invalid EBML header.
    #[error("Invalid EBML header: {0}")]
    InvalidEbmlHeader(String),

    /// Invalid element ID.
    #[error("Invalid element ID at offset {offset}")]
    InvalidElementId {
        /// Byte offset where the invalid ID was found.
        offset: u64,
    },

    /// Invalid element size.
    #[error("Invalid element size at offset {offset}: {message}")]
    InvalidElementSize {
        /// Byte offset where the invalid size was found.
        offset: u64,
        /// Description of the size error.
        message: String,
    },

    /// Unknown element ID.
    #[error("Unknown element ID: 0x{id:08X}")]
    UnknownElement {
        /// The unknown element ID value.
        id: u32,
    },

    /// Missing required element.
    #[error("Missing required element: {0}")]
    MissingElement(String),

    /// Invalid track type.
    #[error("Invalid track type: {0}")]
    InvalidTrackType(u8),

    /// Invalid codec ID.
    #[error("Invalid or unsupported codec ID: {0}")]
    InvalidCodecId(String),

    /// Invalid timestamp.
    #[error("Invalid timestamp at cluster offset {offset}")]
    InvalidTimestamp {
        /// Byte offset of the cluster with invalid timestamp.
        offset: u64,
    },

    /// Invalid block structure.
    #[error("Invalid block structure: {0}")]
    InvalidBlock(String),

    /// Invalid lacing structure.
    #[error("Invalid lacing: {0}")]
    InvalidLacing(String),

    /// Cluster without timestamp.
    #[error("Cluster missing timestamp at offset {offset}")]
    ClusterMissingTimestamp {
        /// Byte offset of the cluster missing a timestamp.
        offset: u64,
    },

    /// Track not found.
    #[error("Track {track_number} not found")]
    TrackNotFound {
        /// The track number that was not found.
        track_number: u64,
    },

    /// Invalid variable-length integer.
    #[error("Invalid VINT encoding at offset {offset}")]
    InvalidVint {
        /// Byte offset where the invalid VINT was found.
        offset: u64,
    },

    /// VINT overflow (value too large).
    #[error("VINT overflow: value exceeds maximum representable size")]
    VintOverflow,

    /// Recursion limit exceeded.
    #[error("Recursion limit exceeded at depth {depth}")]
    RecursionLimit {
        /// The depth at which recursion was limited.
        depth: u32,
    },

    /// Invalid WebM file (non-WebM codecs used).
    #[error("Invalid WebM file: {0}")]
    InvalidWebM(String),

    /// Seek failed.
    #[error("Seek failed: {0}")]
    SeekFailed(String),

    /// Cue point not found.
    #[error("No cue point found for timestamp {timestamp_ns}ns")]
    CueNotFound {
        /// The timestamp in nanoseconds that was not found.
        timestamp_ns: i64,
    },

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error.
    #[error("{0}")]
    Other(String),
}

impl From<String> for WebmError {
    fn from(s: String) -> Self {
        WebmError::Other(s)
    }
}

impl From<&str> for WebmError {
    fn from(s: &str) -> Self {
        WebmError::Other(s.to_string())
    }
}

/// Result type for WebM operations.
pub type Result<T> = std::result::Result<T, WebmError>;

/// Convert WebmError to transcode_core::Error.
impl From<WebmError> for transcode_core::Error {
    fn from(err: WebmError) -> Self {
        match err {
            WebmError::Io(e) => transcode_core::Error::Io(e),
            WebmError::InvalidEbmlHeader(msg) => {
                transcode_core::Error::Container(transcode_core::error::ContainerError::InvalidStructure(msg))
            }
            WebmError::MissingElement(name) => {
                transcode_core::Error::Container(transcode_core::error::ContainerError::MissingElement(name))
            }
            WebmError::InvalidElementSize { offset, message } => {
                transcode_core::Error::Container(transcode_core::error::ContainerError::InvalidSize {
                    offset,
                    message,
                })
            }
            WebmError::TrackNotFound { track_number } => {
                transcode_core::Error::Container(transcode_core::error::ContainerError::StreamNotFound {
                    index: track_number as u32,
                })
            }
            WebmError::SeekFailed(msg) => {
                transcode_core::Error::Container(transcode_core::error::ContainerError::SeekFailed(msg))
            }
            WebmError::RecursionLimit { depth } => {
                transcode_core::Error::Container(transcode_core::error::ContainerError::RecursionLimit { depth })
            }
            _ => transcode_core::Error::Container(transcode_core::error::ContainerError::Other(
                err.to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = WebmError::InvalidElementId { offset: 100 };
        assert_eq!(err.to_string(), "Invalid element ID at offset 100");
    }

    #[test]
    fn test_error_from_string() {
        let err: WebmError = "test error".into();
        assert!(matches!(err, WebmError::Other(_)));
    }

    #[test]
    fn test_conversion_to_core_error() {
        let webm_err = WebmError::TrackNotFound { track_number: 5 };
        let core_err: transcode_core::Error = webm_err.into();
        assert!(matches!(
            core_err,
            transcode_core::Error::Container(transcode_core::error::ContainerError::StreamNotFound { index: 5 })
        ));
    }

    #[test]
    fn test_invalid_webm_error() {
        let err = WebmError::InvalidWebM("H.264 not supported".to_string());
        assert!(err.to_string().contains("H.264"));
    }

    #[test]
    fn test_vint_overflow_error() {
        let err = WebmError::VintOverflow;
        assert!(err.to_string().contains("overflow"));
    }
}
