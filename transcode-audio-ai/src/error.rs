//! Audio AI errors

use thiserror::Error;

/// Errors from audio AI processing
#[derive(Error, Debug)]
pub enum AudioAiError {
    /// Invalid input
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Processing failed
    #[error("processing failed: {0}")]
    Processing(String),

    /// Model error
    #[error("model error: {0}")]
    Model(String),
}
