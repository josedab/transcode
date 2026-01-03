//! Error types for DRM and encryption operations.
//!
//! This module defines error types specific to DRM systems including
//! CENC encryption, key management, and DRM provider operations.

use thiserror::Error;

/// Main error type for DRM operations.
#[derive(Error, Debug)]
pub enum DrmError {
    /// Key-related errors.
    #[error("Key error: {0}")]
    Key(#[from] KeyError),

    /// Encryption operation errors.
    #[error("Encryption error: {0}")]
    Encryption(#[from] EncryptionError),

    /// PSSH box errors.
    #[error("PSSH error: {0}")]
    Pssh(#[from] PsshError),

    /// Invalid DRM configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Unsupported DRM feature.
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// I/O error during DRM operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization errors.
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Key management errors.
#[derive(Error, Debug)]
pub enum KeyError {
    /// Invalid key length (must be 16 bytes for AES-128).
    #[error("Invalid key length: expected {expected} bytes, got {actual}")]
    InvalidKeyLength {
        /// Expected key length in bytes.
        expected: usize,
        /// Actual key length provided.
        actual: usize,
    },

    /// Invalid key ID format.
    #[error("Invalid key ID: {0}")]
    InvalidKeyId(String),

    /// Invalid initialization vector length.
    #[error("Invalid IV length: expected {expected} bytes, got {actual}")]
    InvalidIvLength {
        /// Expected IV length in bytes.
        expected: usize,
        /// Actual IV length provided.
        actual: usize,
    },

    /// Key not found in key store.
    #[error("Key not found: {key_id}")]
    KeyNotFound {
        /// The key ID that was not found.
        key_id: String,
    },

    /// Duplicate key ID in key store.
    #[error("Duplicate key ID: {key_id}")]
    DuplicateKey {
        /// The duplicate key ID.
        key_id: String,
    },

    /// Key derivation failed.
    #[error("Key derivation failed: {0}")]
    DerivationFailed(String),

    /// Invalid base64 encoding.
    #[error("Invalid base64 encoding: {0}")]
    InvalidBase64(String),

    /// Key validation failed.
    #[error("Key validation failed: {0}")]
    ValidationFailed(String),
}

/// Encryption operation errors.
#[derive(Error, Debug)]
pub enum EncryptionError {
    /// Invalid encryption scheme.
    #[error("Invalid encryption scheme: {0}")]
    InvalidScheme(String),

    /// Encryption buffer too small.
    #[error("Buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall {
        /// Required buffer size in bytes.
        needed: usize,
        /// Available buffer size in bytes.
        available: usize,
    },

    /// Invalid subsample configuration.
    #[error("Invalid subsample: {0}")]
    InvalidSubsample(String),

    /// Pattern encryption configuration error.
    #[error("Invalid pattern: crypt={crypt_blocks}, skip={skip_blocks}")]
    InvalidPattern {
        /// Number of encrypted blocks in the pattern.
        crypt_blocks: u32,
        /// Number of clear blocks in the pattern.
        skip_blocks: u32,
    },

    /// Counter overflow during CTR mode.
    #[error("Counter overflow during encryption")]
    CounterOverflow,

    /// Block alignment error.
    #[error("Data not block aligned: {size} bytes is not a multiple of {block_size}")]
    BlockAlignment {
        /// Actual data size in bytes.
        size: usize,
        /// Required block size for alignment.
        block_size: usize,
    },

    /// Decryption verification failed.
    #[error("Decryption verification failed")]
    VerificationFailed,

    /// Clear lead configuration error.
    #[error("Invalid clear lead: {0}")]
    InvalidClearLead(String),

    /// Sample too large for encryption.
    #[error("Sample too large: {size} bytes exceeds maximum {max}")]
    SampleTooLarge {
        /// Actual sample size in bytes.
        size: usize,
        /// Maximum allowed sample size in bytes.
        max: usize,
    },
}

/// PSSH (Protection System Specific Header) box errors.
#[derive(Error, Debug)]
pub enum PsshError {
    /// Unknown system ID.
    #[error("Unknown system ID: {0}")]
    UnknownSystemId(String),

    /// Invalid PSSH box format.
    #[error("Invalid PSSH format: {0}")]
    InvalidFormat(String),

    /// PSSH data too large.
    #[error("PSSH data too large: {size} bytes exceeds maximum {max}")]
    DataTooLarge {
        /// Actual data size in bytes.
        size: usize,
        /// Maximum allowed size in bytes.
        max: usize,
    },

    /// Missing required PSSH field.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid PSSH version.
    #[error("Invalid PSSH version: {version} (expected {expected})")]
    InvalidVersion {
        /// Actual version found.
        version: u8,
        /// Expected version.
        expected: u8,
    },

    /// Protobuf encoding/decoding error.
    #[error("Protobuf error: {0}")]
    ProtobufError(String),
}

/// Result type alias for DRM operations.
pub type Result<T> = std::result::Result<T, DrmError>;

impl DrmError {
    /// Create an invalid configuration error.
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        DrmError::InvalidConfig(msg.into())
    }

    /// Create an unsupported feature error.
    pub fn unsupported(msg: impl Into<String>) -> Self {
        DrmError::Unsupported(msg.into())
    }

    /// Check if error is recoverable.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            DrmError::Key(KeyError::KeyNotFound { .. })
                | DrmError::Encryption(EncryptionError::BufferTooSmall { .. })
        )
    }
}

impl From<base64::DecodeError> for DrmError {
    fn from(err: base64::DecodeError) -> Self {
        DrmError::Key(KeyError::InvalidBase64(err.to_string()))
    }
}

impl From<uuid::Error> for DrmError {
    fn from(err: uuid::Error) -> Self {
        DrmError::Key(KeyError::InvalidKeyId(err.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_error_display() {
        let err = KeyError::InvalidKeyLength {
            expected: 16,
            actual: 8,
        };
        assert_eq!(
            err.to_string(),
            "Invalid key length: expected 16 bytes, got 8"
        );
    }

    #[test]
    fn test_encryption_error_display() {
        let err = EncryptionError::BufferTooSmall {
            needed: 256,
            available: 128,
        };
        assert_eq!(
            err.to_string(),
            "Buffer too small: need 256 bytes, have 128"
        );
    }

    #[test]
    fn test_pssh_error_display() {
        let err = PsshError::InvalidVersion {
            version: 2,
            expected: 1,
        };
        assert_eq!(
            err.to_string(),
            "Invalid PSSH version: 2 (expected 1)"
        );
    }

    #[test]
    fn test_drm_error_conversion() {
        let key_err = KeyError::KeyNotFound {
            key_id: "test-key".into(),
        };
        let drm_err: DrmError = key_err.into();
        assert!(matches!(drm_err, DrmError::Key(KeyError::KeyNotFound { .. })));
    }

    #[test]
    fn test_is_recoverable() {
        let recoverable = DrmError::Key(KeyError::KeyNotFound {
            key_id: "test".into(),
        });
        assert!(recoverable.is_recoverable());

        let not_recoverable = DrmError::InvalidConfig("test".into());
        assert!(!not_recoverable.is_recoverable());
    }
}
