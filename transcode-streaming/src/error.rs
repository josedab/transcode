//! Streaming error types.

use thiserror::Error;

/// Errors that can occur during streaming operations.
#[derive(Error, Debug)]
pub enum StreamingError {
    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Segment error.
    #[error("Segment error: {0}")]
    SegmentError(String),

    /// Manifest generation error.
    #[error("Manifest error: {0}")]
    ManifestError(String),

    /// Invalid quality configuration.
    #[error("Invalid quality: {0}")]
    InvalidQuality(String),

    /// XML serialization error.
    #[error("XML error: {0}")]
    XmlError(String),

    /// JSON serialization error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// DRM configuration error.
    #[error("DRM error: {0}")]
    DrmError(String),
}

impl From<quick_xml::Error> for StreamingError {
    fn from(err: quick_xml::Error) -> Self {
        StreamingError::XmlError(err.to_string())
    }
}

impl From<quick_xml::DeError> for StreamingError {
    fn from(err: quick_xml::DeError) -> Self {
        StreamingError::XmlError(err.to_string())
    }
}

impl From<StreamingError> for transcode_core::Error {
    fn from(err: StreamingError) -> Self {
        transcode_core::Error::Config(err.to_string())
    }
}
