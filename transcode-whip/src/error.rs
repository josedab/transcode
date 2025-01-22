//! Error types for WHIP/WHEP operations.

use thiserror::Error;

/// Error type for WHIP/WHEP operations.
#[derive(Error, Debug)]
pub enum WhipError {
    /// Invalid SDP offer or answer.
    #[error("Invalid SDP: {0}")]
    InvalidSdp(String),

    /// Session not found.
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Session already exists.
    #[error("Session already exists: {0}")]
    SessionExists(String),

    /// Connection failed.
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// ICE negotiation failed.
    #[error("ICE negotiation failed: {0}")]
    IceNegotiationFailed(String),

    /// DTLS handshake failed.
    #[error("DTLS handshake failed: {0}")]
    DtlsFailed(String),

    /// Media processing error.
    #[error("Media error: {0}")]
    MediaError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Server error.
    #[error("Server error: {0}")]
    ServerError(String),

    /// HTTP error.
    #[error("HTTP error: {status} - {message}")]
    HttpError { status: u16, message: String },

    /// Unauthorized access.
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// Resource limit exceeded.
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),

    /// Timeout error.
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for WHIP operations.
pub type Result<T> = std::result::Result<T, WhipError>;
