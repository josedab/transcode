//! Error types for confidential transcoding.

use thiserror::Error;

/// Error type for confidential transcoding operations.
#[derive(Error, Debug)]
pub enum ConfidentialError {
    /// TEE not available or not supported.
    #[error("TEE not available: {0}")]
    TeeNotAvailable(String),

    /// Attestation failed.
    #[error("Attestation failed: {0}")]
    AttestationFailed(String),

    /// Key management error.
    #[error("Key error: {0}")]
    KeyError(String),

    /// Encryption/decryption error.
    #[error("Crypto error: {0}")]
    CryptoError(String),

    /// Secure channel error.
    #[error("Secure channel error: {0}")]
    SecureChannelError(String),

    /// Memory sealing error.
    #[error("Sealing error: {0}")]
    SealingError(String),

    /// Policy violation.
    #[error("Policy violation: {0}")]
    PolicyViolation(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for confidential transcoding.
pub type Result<T> = std::result::Result<T, ConfidentialError>;
