//! # transcode-dts
//!
//! DTS and Dolby TrueHD audio codec support for decoding and encoding.
//!
//! ## Important Notice
//!
//! **DTS and TrueHD are patent-encumbered codecs with licensing requirements.**
//!
//! - DTS (Digital Theater Systems) is licensed by DTS, Inc.
//! - TrueHD (Dolby TrueHD) is licensed by Dolby Laboratories.
//!
//! This crate provides:
//! - Native bitstream parsing and sync frame detection
//! - Audio metadata extraction
//! - FFI wrapper support for encoding/decoding via external libraries
//!
//! For full encoding/decoding functionality, users must enable the `ffi-ffmpeg` feature
//! and have FFmpeg development libraries installed, along with appropriate licenses.
//!
//! ## Supported Formats
//!
//! - **DTS Core**: Original DTS surround sound (up to 5.1 channels)
//! - **DTS-HD High Resolution**: Extended bit depth and sample rates
//! - **DTS-HD Master Audio**: Lossless extension to DTS core
//! - **DTS:X**: Object-based immersive audio
//! - **TrueHD**: Dolby's lossless audio format for Blu-ray
//! - **Atmos via TrueHD**: Object-based audio layer on TrueHD
//!
//! ## Features
//!
//! - Parse DTS and TrueHD sync frames and headers
//! - Extract audio parameters (sample rate, channels, bit depth)
//! - Detect DTS-HD extensions and lossless streams
//! - Support for various channel configurations
//! - FFI wrapper for FFmpeg libavcodec (optional feature)
//!
//! ## Example
//!
//! ```rust
//! use transcode_dts::{DtsParser, DtsSyncFrame};
//!
//! // Parse DTS audio sync frame
//! let data = vec![0x7F, 0xFE, 0x80, 0x01, /* ... DTS frame data ... */];
//! let mut parser = DtsParser::new();
//!
//! if let Some(frame) = parser.parse_sync_frame(&data) {
//!     println!("Sample rate: {} Hz", frame.sample_rate);
//!     println!("Channels: {}", frame.channels);
//!     println!("Frame size: {} bytes", frame.frame_size);
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

pub mod decoder;
pub mod encoder;
#[cfg(feature = "ffi-ffmpeg")]
pub mod ffi;
pub mod parser;
pub mod types;

pub use decoder::{DtsDecoder, TrueHdDecoder};
pub use encoder::{DtsEncoder, DtsEncoderConfig};
pub use parser::{DtsParser, TrueHdParser};
pub use types::*;

use thiserror::Error;

/// DTS/TrueHD codec error types.
#[derive(Error, Debug)]
pub enum DtsError {
    /// Invalid sync word.
    #[error("Invalid sync word: expected DTS or TrueHD sync, got {0:#010x}")]
    InvalidSyncWord(u32),

    /// Invalid frame header.
    #[error("Invalid frame header: {0}")]
    InvalidFrameHeader(String),

    /// Invalid extension header.
    #[error("Invalid extension header: {0}")]
    InvalidExtensionHeader(String),

    /// Unsupported sample rate.
    #[error("Unsupported sample rate: {0} Hz")]
    UnsupportedSampleRate(u32),

    /// Unsupported channel configuration.
    #[error("Unsupported channel configuration: {0}")]
    UnsupportedChannelConfig(u8),

    /// Unsupported bit depth.
    #[error("Unsupported bit depth: {0} bits")]
    UnsupportedBitDepth(u8),

    /// Bitstream corruption detected.
    #[error("Bitstream corruption: {0}")]
    BitstreamCorruption(String),

    /// Insufficient data.
    #[error("Insufficient data: need {needed} bytes, have {available}")]
    InsufficientData {
        /// Bytes needed.
        needed: usize,
        /// Bytes available.
        available: usize,
    },

    /// FFI not available.
    #[error("FFI support not enabled - enable 'ffi-ffmpeg' feature for full decoding/encoding")]
    FfiNotAvailable,

    /// FFI initialization failed.
    #[error("FFI initialization failed: {0}")]
    FfiInitError(String),

    /// Encoding error.
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Decoding error.
    #[error("Decoding error: {0}")]
    DecodingError(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for DTS/TrueHD operations.
pub type Result<T> = std::result::Result<T, DtsError>;

/// DTS sync word (14-bit format, big endian).
pub const DTS_SYNC_WORD_BE: u32 = 0x7FFE8001;

/// DTS sync word (14-bit format, little endian).
pub const DTS_SYNC_WORD_LE: u32 = 0xFE7F0180;

/// DTS sync word (16-bit format, big endian).
pub const DTS_SYNC_WORD_16BE: u32 = 0x7FFE8001;

/// DTS sync word (16-bit format, little endian).
pub const DTS_SYNC_WORD_16LE: u32 = 0xFE7F0180;

/// DTS-HD sync word for extended streams.
pub const DTS_HD_SYNC: u32 = 0x64582025;

/// TrueHD sync word.
pub const TRUEHD_SYNC: u32 = 0xF8726FBA;

/// MLP (Meridian Lossless Packing) sync word used by TrueHD.
pub const MLP_SYNC: u32 = 0xF8726FBB;

/// Maximum DTS core frame size in bytes.
pub const DTS_MAX_FRAME_SIZE: usize = 16384;

/// DTS core samples per frame.
pub const DTS_SAMPLES_PER_FRAME: usize = 512;

/// DTS-HD MA maximum samples per frame.
pub const DTS_HD_SAMPLES_PER_FRAME: usize = 4096;

/// TrueHD samples per frame (for 48kHz).
pub const TRUEHD_SAMPLES_PER_FRAME: usize = 40;

/// Check if FFI support is available.
pub fn is_ffi_available() -> bool {
    cfg!(feature = "ffi-ffmpeg")
}

/// Get codec information string.
pub fn codec_info() -> &'static str {
    if is_ffi_available() {
        "DTS/TrueHD Audio (FFmpeg FFI)"
    } else {
        "DTS/TrueHD Audio (parser only)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_words() {
        assert_eq!(DTS_SYNC_WORD_BE, 0x7FFE8001);
        assert_eq!(TRUEHD_SYNC, 0xF8726FBA);
        assert_eq!(DTS_HD_SYNC, 0x64582025);
    }

    #[test]
    fn test_ffi_availability() {
        #[cfg(not(feature = "ffi-ffmpeg"))]
        assert!(!is_ffi_available());
    }

    #[test]
    fn test_codec_info() {
        let info = codec_info();
        assert!(info.contains("DTS") || info.contains("TrueHD"));
    }
}
