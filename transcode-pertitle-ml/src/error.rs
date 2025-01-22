//! Error types for ML per-title encoding.

use thiserror::Error;

/// Error type for ML per-title encoding operations.
#[derive(Error, Debug)]
pub enum PerTitleMlError {
    /// Model not found or failed to load.
    #[error("Model error: {0}")]
    ModelError(String),

    /// Invalid input features.
    #[error("Invalid features: {0}")]
    InvalidFeatures(String),

    /// Inference failed.
    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    /// Training error.
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Data collection error.
    #[error("Data collection error: {0}")]
    DataCollectionError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Quality metric error.
    #[error("Quality error: {0}")]
    QualityError(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for ML per-title operations.
pub type Result<T> = std::result::Result<T, PerTitleMlError>;
