//! Content intelligence error types.

use thiserror::Error;

/// Content intelligence errors.
#[derive(Debug, Error)]
pub enum IntelError {
    /// Invalid frame data.
    #[error("Invalid frame: {0}")]
    InvalidFrame(String),

    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Analysis failed.
    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),

    /// Insufficient data.
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

impl From<IntelError> for transcode_core::error::Error {
    fn from(e: IntelError) -> Self {
        transcode_core::error::Error::Unsupported(e.to_string())
    }
}

/// Result type for content intelligence.
pub type Result<T> = std::result::Result<T, IntelError>;
