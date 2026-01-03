//! Dolby Vision error types.
//!
//! This module defines all error types specific to Dolby Vision processing,
//! including RPU parsing errors, profile validation errors, and conversion errors.

// Error enum variants have many fields that are self-documenting via the error message
#![allow(missing_docs)]

use thiserror::Error;

/// Result type for Dolby Vision operations.
pub type Result<T> = std::result::Result<T, DolbyError>;

/// Errors that can occur during Dolby Vision processing.
#[derive(Debug, Error)]
pub enum DolbyError {
    /// Invalid or corrupted RPU data.
    #[error("Invalid RPU data: {message}")]
    InvalidRpu { message: String },

    /// RPU NAL unit parsing error.
    #[error("RPU NAL parsing error at byte {offset}: {message}")]
    RpuParseError { offset: usize, message: String },

    /// Invalid RPU header.
    #[error("Invalid RPU header: {message}")]
    InvalidRpuHeader { message: String },

    /// Unsupported RPU type.
    #[error("Unsupported RPU type: {rpu_type}")]
    UnsupportedRpuType { rpu_type: u8 },

    /// Invalid Dolby Vision profile.
    #[error("Invalid Dolby Vision profile: {profile}")]
    InvalidProfile { profile: u8 },

    /// Unsupported Dolby Vision profile.
    #[error("Unsupported Dolby Vision profile {profile} for operation: {operation}")]
    UnsupportedProfile { profile: u8, operation: String },

    /// Invalid Dolby Vision level.
    #[error("Invalid Dolby Vision level: {level}")]
    InvalidLevel { level: u8 },

    /// Profile/level constraint violation.
    #[error("Profile/level constraint violation: {message}")]
    ConstraintViolation { message: String },

    /// Invalid metadata block.
    #[error("Invalid metadata block L{level}: {message}")]
    InvalidMetadata { level: u8, message: String },

    /// Missing required metadata.
    #[error("Missing required metadata: {field}")]
    MissingMetadata { field: String },

    /// Invalid extension block.
    #[error("Invalid extension block type {block_type}: {message}")]
    InvalidExtensionBlock { block_type: u8, message: String },

    /// VDR (Video Dynamic Range) data error.
    #[error("VDR data error: {message}")]
    VdrError { message: String },

    /// Invalid coefficient table.
    #[error("Invalid coefficient table: {message}")]
    InvalidCoefficients { message: String },

    /// Polynomial coefficient error.
    #[error("Polynomial coefficient error: {message}")]
    PolynomialError { message: String },

    /// MMR (Multi-resolution Mapping) error.
    #[error("MMR processing error: {message}")]
    MmrError { message: String },

    /// NLQ (Non-Linear Quantization) error.
    #[error("NLQ processing error: {message}")]
    NlqError { message: String },

    /// Tone mapping error.
    #[error("Tone mapping error: {message}")]
    ToneMappingError { message: String },

    /// Gamut mapping error.
    #[error("Gamut mapping error: {message}")]
    GamutMappingError { message: String },

    /// Profile conversion error.
    #[error("Profile conversion error from profile {from} to {to}: {message}")]
    ConversionError {
        from: u8,
        to: u8,
        message: String,
    },

    /// Dual-layer to single-layer conversion error.
    #[error("Dual to single layer conversion error: {message}")]
    DualToSingleError { message: String },

    /// Layer extraction error.
    #[error("Layer extraction error: {message}")]
    ExtractionError { message: String },

    /// RPU injection error.
    #[error("RPU injection error: {message}")]
    InjectionError { message: String },

    /// HEVC stream error.
    #[error("HEVC stream error: {message}")]
    HevcStreamError { message: String },

    /// NAL unit error.
    #[error("NAL unit error: {message}")]
    NalError { message: String },

    /// Bitstream reading error.
    #[error("Bitstream error at bit {bit_offset}: {message}")]
    BitstreamError {
        /// Bit offset where the error occurred.
        bit_offset: usize,
        /// Error message.
        message: String,
    },

    /// Buffer too small.
    #[error("Buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall {
        /// Number of bytes needed.
        needed: usize,
        /// Number of bytes available.
        available: usize,
    },

    /// CRC check failed.
    #[error("CRC check failed: expected {expected:#010x}, got {actual:#010x}")]
    CrcMismatch {
        /// Expected CRC value.
        expected: u32,
        /// Actual computed CRC value.
        actual: u32,
    },

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Internal error.
    #[error("Internal error: {message}")]
    Internal {
        /// Error message.
        message: String,
    },
}

impl DolbyError {
    /// Create an invalid RPU error.
    pub fn invalid_rpu(message: impl Into<String>) -> Self {
        DolbyError::InvalidRpu {
            message: message.into(),
        }
    }

    /// Create an RPU parse error.
    pub fn rpu_parse(offset: usize, message: impl Into<String>) -> Self {
        DolbyError::RpuParseError {
            offset,
            message: message.into(),
        }
    }

    /// Create an invalid profile error.
    pub fn invalid_profile(profile: u8) -> Self {
        DolbyError::InvalidProfile { profile }
    }

    /// Create an unsupported profile error.
    pub fn unsupported_profile(profile: u8, operation: impl Into<String>) -> Self {
        DolbyError::UnsupportedProfile {
            profile,
            operation: operation.into(),
        }
    }

    /// Create an invalid metadata error.
    pub fn invalid_metadata(level: u8, message: impl Into<String>) -> Self {
        DolbyError::InvalidMetadata {
            level,
            message: message.into(),
        }
    }

    /// Create a conversion error.
    pub fn conversion(from: u8, to: u8, message: impl Into<String>) -> Self {
        DolbyError::ConversionError {
            from,
            to,
            message: message.into(),
        }
    }

    /// Create a bitstream error.
    pub fn bitstream(bit_offset: usize, message: impl Into<String>) -> Self {
        DolbyError::BitstreamError {
            bit_offset,
            message: message.into(),
        }
    }

    /// Create an internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        DolbyError::Internal {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DolbyError::InvalidProfile { profile: 99 };
        assert_eq!(err.to_string(), "Invalid Dolby Vision profile: 99");

        let err = DolbyError::RpuParseError {
            offset: 42,
            message: "unexpected end of data".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "RPU NAL parsing error at byte 42: unexpected end of data"
        );
    }

    #[test]
    fn test_error_constructors() {
        let err = DolbyError::invalid_rpu("test error");
        assert!(matches!(err, DolbyError::InvalidRpu { .. }));

        let err = DolbyError::conversion(7, 8, "metadata loss");
        assert!(matches!(err, DolbyError::ConversionError { from: 7, to: 8, .. }));
    }
}
