//! Distributed transcoding error types.

use thiserror::Error;

/// Distributed transcoding errors.
#[derive(Debug, Error)]
pub enum DistributedError {
    /// Task not found.
    #[error("Task not found: {0}")]
    TaskNotFound(String),

    /// Worker not found.
    #[error("Worker not found: {0}")]
    WorkerNotFound(String),

    /// Worker unavailable.
    #[error("No workers available")]
    NoWorkersAvailable,

    /// Task already exists.
    #[error("Task already exists: {0}")]
    TaskExists(String),

    /// Task failed.
    #[error("Task failed: {0}")]
    TaskFailed(String),

    /// Invalid task state transition.
    #[error("Invalid state transition from {from} to {to}")]
    InvalidStateTransition { from: String, to: String },

    /// Timeout.
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Network error.
    #[error("Network error: {0}")]
    Network(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Database error.
    #[error("Database error: {0}")]
    Database(String),

    /// Not found.
    #[error("Not found: {0}")]
    NotFound(String),

    /// Duplicate entry.
    #[error("Duplicate entry: {0}")]
    Duplicate(String),
}

impl From<serde_json::Error> for DistributedError {
    fn from(e: serde_json::Error) -> Self {
        DistributedError::Serialization(e.to_string())
    }
}

impl From<bincode::Error> for DistributedError {
    fn from(e: bincode::Error) -> Self {
        DistributedError::Serialization(e.to_string())
    }
}

impl From<tokio::sync::oneshot::error::RecvError> for DistributedError {
    fn from(e: tokio::sync::oneshot::error::RecvError) -> Self {
        DistributedError::Internal(e.to_string())
    }
}

impl From<DistributedError> for transcode_core::error::Error {
    fn from(e: DistributedError) -> Self {
        transcode_core::error::Error::Unsupported(e.to_string())
    }
}

/// Result type for distributed operations.
pub type Result<T> = std::result::Result<T, DistributedError>;
