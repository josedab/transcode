//! # transcode-mpeg2
//!
//! MPEG-2 video codec support for decoding and encoding.
//!
//! ## Important Notice
//!
//! **MPEG-2 was patent-encumbered by MPEG-LA. Most patents have expired (2018-2019),
//! but some may still apply in certain jurisdictions.**
//!
//! This crate provides:
//! - Native bitstream parsing and sequence header detection
//! - Video metadata extraction
//! - FFI wrapper support for encoding/decoding via external libraries
//!
//! For full encoding/decoding functionality, users must enable the `ffi-ffmpeg` feature
//! and have FFmpeg development libraries installed.
//!
//! ## Features
//!
//! - Parse MPEG-2 video elementary stream headers
//! - Extract video parameters (resolution, frame rate, aspect ratio)
//! - Detect GOP structure and I/P/B frames
//! - Support for Main Profile and various levels
//! - FFI wrapper for FFmpeg libavcodec (optional feature)
//!
//! ## Example
//!
//! ```rust
//! use transcode_mpeg2::{Mpeg2Parser, SequenceHeader};
//!
//! // Parse MPEG-2 video elementary stream
//! let data = vec![0x00, 0x00, 0x01, 0xB3, /* ... sequence header data ... */];
//! let mut parser = Mpeg2Parser::new();
//!
//! if let Some(seq) = parser.parse_sequence_header(&data) {
//!     println!("Resolution: {}x{}", seq.horizontal_size, seq.vertical_size);
//!     println!("Frame rate: {:.3} fps", seq.frame_rate());
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

pub use decoder::Mpeg2Decoder;
pub use encoder::{Mpeg2Encoder, Mpeg2EncoderConfig};
pub use parser::Mpeg2Parser;
pub use types::*;

use thiserror::Error;

/// MPEG-2 codec error types.
#[derive(Error, Debug)]
pub enum Mpeg2Error {
    /// Invalid start code.
    #[error("Invalid start code: expected 0x000001XX, got {0:#010x}")]
    InvalidStartCode(u32),

    /// Invalid sequence header.
    #[error("Invalid sequence header: {0}")]
    InvalidSequenceHeader(String),

    /// Invalid picture header.
    #[error("Invalid picture header: {0}")]
    InvalidPictureHeader(String),

    /// Unsupported profile.
    #[error("Unsupported profile: {0}")]
    UnsupportedProfile(u8),

    /// Unsupported level.
    #[error("Unsupported level: {0}")]
    UnsupportedLevel(u8),

    /// Unsupported chroma format.
    #[error("Unsupported chroma format: {0}")]
    UnsupportedChromaFormat(u8),

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

/// Result type for MPEG-2 operations.
pub type Result<T> = std::result::Result<T, Mpeg2Error>;

/// MPEG-2 start code prefix (0x000001).
pub const START_CODE_PREFIX: u32 = 0x000001;

/// Sequence header start code.
pub const SEQUENCE_HEADER_CODE: u8 = 0xB3;

/// Sequence extension start code.
pub const EXTENSION_START_CODE: u8 = 0xB5;

/// Group of Pictures start code.
pub const GOP_START_CODE: u8 = 0xB8;

/// Picture start code.
pub const PICTURE_START_CODE: u8 = 0x00;

/// Sequence end code.
pub const SEQUENCE_END_CODE: u8 = 0xB7;

/// User data start code.
pub const USER_DATA_START_CODE: u8 = 0xB2;

/// Slice start code range (0x01 - 0xAF).
pub const SLICE_START_CODE_MIN: u8 = 0x01;
/// Maximum slice start code.
pub const SLICE_START_CODE_MAX: u8 = 0xAF;

/// Maximum frame width for Main Profile.
pub const MAX_FRAME_WIDTH: usize = 1920;

/// Maximum frame height for Main Profile.
pub const MAX_FRAME_HEIGHT: usize = 1152;

/// Check if FFI support is available.
pub fn is_ffi_available() -> bool {
    cfg!(feature = "ffi-ffmpeg")
}

/// Get codec information string.
pub fn codec_info() -> &'static str {
    if is_ffi_available() {
        "MPEG-2 Video (FFmpeg FFI)"
    } else {
        "MPEG-2 Video (parser only)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_codes() {
        assert_eq!(SEQUENCE_HEADER_CODE, 0xB3);
        assert_eq!(GOP_START_CODE, 0xB8);
        assert_eq!(PICTURE_START_CODE, 0x00);
    }

    #[test]
    fn test_ffi_availability() {
        #[cfg(not(feature = "ffi-ffmpeg"))]
        assert!(!is_ffi_available());
    }

    #[test]
    fn test_codec_info() {
        let info = codec_info();
        assert!(info.contains("MPEG-2"));
    }
}
