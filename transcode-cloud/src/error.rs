//! Cloud storage errors

use thiserror::Error;

/// Errors from cloud operations
#[derive(Error, Debug)]
pub enum CloudError {
    /// Invalid URL
    #[error("invalid URL: {0}")]
    InvalidUrl(String),

    /// Object not found
    #[error("object not found")]
    NotFound,

    /// Access denied
    #[error("access denied")]
    AccessDenied,

    /// Unsupported provider
    #[error("unsupported provider: {0}")]
    UnsupportedProvider(String),

    /// Network error
    #[error("network error: {0}")]
    Network(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Provider error
    #[error("provider error: {0}")]
    Provider(String),

    /// Authentication failed
    #[error("authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Upload failed
    #[error("upload failed: {0}")]
    UploadFailed(String),

    /// Download failed
    #[error("download failed: {0}")]
    DownloadFailed(String),

    /// Delete failed
    #[error("delete failed: {0}")]
    DeleteFailed(String),

    /// Other error
    #[error("{0}")]
    Other(String),
}
