//! AI error types.

use thiserror::Error;

/// Result type for AI operations.
pub type Result<T> = std::result::Result<T, AiError>;

/// AI-specific errors.
#[derive(Error, Debug)]
pub enum AiError {
    /// Model loading error.
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    /// Model inference error.
    #[error("Inference failed: {0}")]
    InferenceError(String),

    /// Invalid frame format or dimensions.
    #[error("Invalid frame: {0}")]
    InvalidFrame(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Feature not configured.
    #[error("Feature not configured: {0}")]
    NotConfigured(String),

    /// Unsupported model format.
    #[error("Unsupported model format: {0}")]
    UnsupportedFormat(String),

    /// Model not found.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Memory allocation error.
    #[error("Memory allocation failed: {0}")]
    AllocationError(String),

    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Image processing error.
    #[error("Image error: {0}")]
    ImageError(String),
}

impl From<image::ImageError> for AiError {
    fn from(err: image::ImageError) -> Self {
        AiError::ImageError(err.to_string())
    }
}

#[cfg(feature = "onnx")]
impl From<ort::Error> for AiError {
    fn from(err: ort::Error) -> Self {
        AiError::InferenceError(err.to_string())
    }
}

impl From<AiError> for transcode_core::Error {
    fn from(err: AiError) -> Self {
        transcode_core::Error::Unsupported(err.to_string())
    }
}
