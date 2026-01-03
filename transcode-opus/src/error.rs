//! Error types for the Opus codec.
//!
//! This module provides Opus-specific error types for encoding and decoding operations.

use thiserror::Error;

/// Opus-specific errors.
#[derive(Error, Debug)]
pub enum OpusError {
    /// Invalid packet structure.
    #[error("Invalid packet structure: {0}")]
    InvalidPacket(String),

    /// Invalid TOC byte.
    #[error("Invalid TOC byte: 0x{0:02x}")]
    InvalidToc(u8),

    /// Unsupported configuration.
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfig(String),

    /// Invalid frame size.
    #[error("Invalid frame size: {0} samples")]
    InvalidFrameSize(usize),

    /// Invalid sample rate.
    #[error("Invalid sample rate: {0} Hz (must be 8000, 12000, 16000, 24000, or 48000)")]
    InvalidSampleRate(u32),

    /// Invalid channel count.
    #[error("Invalid channel count: {0} (must be 1 or 2)")]
    InvalidChannels(u8),

    /// Invalid bandwidth.
    #[error("Invalid bandwidth: {0}")]
    InvalidBandwidth(String),

    /// Invalid mode.
    #[error("Invalid mode: {0}")]
    InvalidMode(String),

    /// Range coder error.
    #[error("Range coder error: {0}")]
    RangeCoder(String),

    /// SILK decoder error.
    #[error("SILK decoder error: {0}")]
    SilkDecoder(String),

    /// CELT decoder error.
    #[error("CELT decoder error: {0}")]
    CeltDecoder(String),

    /// Encoder configuration error.
    #[error("Encoder configuration error: {0}")]
    EncoderConfig(String),

    /// Decoder not initialized.
    #[error("Decoder not initialized")]
    NotInitialized,

    /// Buffer too small.
    #[error("Buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall {
        /// Number of bytes needed.
        needed: usize,
        /// Number of bytes available.
        available: usize,
    },

    /// Bitstream corruption.
    #[error("Bitstream corruption at offset {offset}")]
    BitstreamCorruption {
        /// Offset where corruption was detected.
        offset: usize,
    },

    /// Packet loss concealment failed.
    #[error("Packet loss concealment failed: {0}")]
    PlcFailed(String),

    /// Internal codec error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<OpusError> for transcode_core::Error {
    fn from(err: OpusError) -> Self {
        transcode_core::Error::Codec(transcode_core::error::CodecError::Other(err.to_string()))
    }
}

/// Result type for Opus operations.
pub type Result<T> = std::result::Result<T, OpusError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = OpusError::InvalidSampleRate(44100);
        assert!(err.to_string().contains("44100"));

        let err = OpusError::InvalidToc(0xFF);
        assert!(err.to_string().contains("0xff"));
    }

    #[test]
    fn test_error_conversion() {
        let opus_err = OpusError::NotInitialized;
        let core_err: transcode_core::Error = opus_err.into();
        assert!(core_err.to_string().contains("not initialized"));
    }
}
