//! Quality metric error types.

use thiserror::Error;

/// Quality metric errors.
#[derive(Debug, Error)]
pub enum QualityError {
    /// Dimension mismatch between reference and distorted frames.
    #[error("Dimension mismatch: reference {reference}, distorted {distorted}")]
    DimensionMismatch {
        reference: String,
        distorted: String,
    },

    /// Invalid frame data.
    #[error("Invalid frame: {0}")]
    InvalidFrame(String),

    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// VMAF not available.
    #[error("VMAF not available: {0}")]
    VmafUnavailable(String),

    /// Computation error.
    #[error("Computation error: {0}")]
    ComputationError(String),
}

impl From<QualityError> for transcode_core::error::Error {
    fn from(e: QualityError) -> Self {
        transcode_core::error::Error::Unsupported(e.to_string())
    }
}

/// Result type for quality metrics.
pub type Result<T> = std::result::Result<T, QualityError>;
