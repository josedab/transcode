//! Error types for ProRes decoding

use thiserror::Error;

/// Result type for ProRes operations
pub type Result<T> = std::result::Result<T, ProResError>;

/// Errors that can occur during ProRes decoding
#[derive(Error, Debug)]
pub enum ProResError {
    /// Invalid or unrecognized frame header signature
    #[error("Invalid frame signature: expected 'icpf', got {0:?}")]
    InvalidSignature([u8; 4]),

    /// Unsupported or unknown ProRes profile
    #[error("Unknown ProRes profile: {0:?}")]
    UnknownProfile([u8; 4]),

    /// Frame data is too short
    #[error("Insufficient data: need {needed} bytes, have {available}")]
    InsufficientData { needed: usize, available: usize },

    /// Invalid frame header
    #[error("Invalid frame header: {0}")]
    InvalidHeader(String),

    /// Invalid slice header
    #[error("Invalid slice header: {0}")]
    InvalidSliceHeader(String),

    /// Huffman decoding error
    #[error("Huffman decoding error: {0}")]
    HuffmanError(String),

    /// DCT coefficient error
    #[error("DCT coefficient error: {0}")]
    DctError(String),

    /// Bitstream error
    #[error("Bitstream error: {0}")]
    BitstreamError(String),

    /// Unsupported feature
    #[error("Unsupported feature: {0}")]
    Unsupported(String),

    /// Invalid quantization matrix
    #[error("Invalid quantization matrix")]
    InvalidQuantMatrix,

    /// Slice size mismatch
    #[error("Slice size mismatch: expected {expected}, got {actual}")]
    SliceSizeMismatch { expected: usize, actual: usize },

    /// Invalid alpha data
    #[error("Invalid alpha channel data: {0}")]
    InvalidAlpha(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
