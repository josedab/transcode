//! Zero-copy errors

use thiserror::Error;

/// Errors from zero-copy operations
#[derive(Error, Debug)]
pub enum ZeroCopyError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Out of bounds access
    #[error("out of bounds")]
    OutOfBounds,

    /// Alignment error
    #[error("alignment error: {0}")]
    Alignment(String),

    /// Memory mapping failed
    #[error("mmap error: {0}")]
    Mmap(String),

    /// io_uring error
    #[error("io_uring error: {0}")]
    IoUring(String),
}
