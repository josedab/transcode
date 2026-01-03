//! Live streaming errors

use thiserror::Error;

/// Errors from live streaming
#[derive(Error, Debug)]
pub enum LiveError {
    /// Invalid URL
    #[error("invalid URL: {0}")]
    InvalidUrl(String),

    /// Connection failed
    #[error("connection failed: {0}")]
    Connection(String),

    /// Not connected
    #[error("not connected")]
    NotConnected,

    /// Timeout
    #[error("timeout")]
    Timeout,

    /// Protocol error
    #[error("protocol error: {0}")]
    Protocol(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error
    #[error("configuration error: {0}")]
    Configuration(String),

    /// Resource limit exceeded
    #[error("resource limit exceeded: {0}")]
    ResourceLimit(String),
}
