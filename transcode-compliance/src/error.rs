//! Error types for compliance operations.

use thiserror::Error;

/// Compliance-related errors.
#[derive(Error, Debug)]
pub enum ComplianceError {
    /// Invalid caption format.
    #[error("Invalid caption format: {0}")]
    InvalidCaptionFormat(String),

    /// Caption sync error.
    #[error("Caption sync error: expected {expected}ms, got {actual}ms")]
    CaptionSyncError { expected: i64, actual: i64 },

    /// Missing required accessibility feature.
    #[error("Missing accessibility feature: {0}")]
    MissingAccessibility(String),

    /// Regulation violation.
    #[error("Regulation violation ({regulation}): {message}")]
    RegulationViolation { regulation: String, message: String },

    /// Validation failed.
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Parse error.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Encoding error.
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Unsupported format.
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for compliance operations.
pub type Result<T> = std::result::Result<T, ComplianceError>;
