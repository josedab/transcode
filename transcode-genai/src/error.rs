//! Error types for generative AI operations.

use thiserror::Error;

/// Generative AI errors.
#[derive(Error, Debug)]
pub enum GenAiError {
    /// Model not found.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Model loading failed.
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    /// Inference failed.
    #[error("Inference failed: {0}")]
    InferenceError(String),

    /// Invalid input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Unsupported operation.
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Resource exhausted.
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Processing timeout.
    #[error("Processing timeout after {0}ms")]
    Timeout(u64),

    /// Feature not enabled.
    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),
}

/// Result type for generative AI operations.
pub type Result<T> = std::result::Result<T, GenAiError>;
