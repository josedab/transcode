//! Error types for observability operations.

use thiserror::Error;

/// Error type for observability operations.
#[derive(Error, Debug)]
pub enum OtelError {
    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Initialization error.
    #[error("Initialization error: {0}")]
    InitError(String),

    /// Export error.
    #[error("Export error: {0}")]
    ExportError(String),

    /// Metric recording error.
    #[error("Metric error: {0}")]
    MetricError(String),

    /// Span error.
    #[error("Span error: {0}")]
    SpanError(String),
}

/// Result type for observability operations.
pub type Result<T> = std::result::Result<T, OtelError>;
