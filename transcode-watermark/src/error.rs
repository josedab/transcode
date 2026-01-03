//! Watermark errors

use thiserror::Error;

/// Errors from watermark operations
#[derive(Error, Debug)]
pub enum WatermarkError {
    /// Invalid payload
    #[error("invalid payload: {0}")]
    InvalidPayload(String),

    /// Embedding failed
    #[error("embedding failed: {0}")]
    Embedding(String),

    /// Extraction failed
    #[error("extraction failed: {0}")]
    Extraction(String),

    /// Watermark not found
    #[error("watermark not found")]
    NotFound,

    /// Corrupted watermark
    #[error("watermark corrupted")]
    Corrupted,
}
