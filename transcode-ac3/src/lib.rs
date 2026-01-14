//! # transcode-ac3
//!
//! AC-3 (Dolby Digital) and E-AC-3 (Enhanced AC-3) audio codec support.
//!
//! ## Important Notice
//!
//! **AC-3 and E-AC-3 are patent-encumbered codecs owned by Dolby Laboratories.**
//!
//! This crate provides:
//! - Native bitstream parsing and sync frame detection
//! - Audio metadata extraction
//! - FFI wrapper support for encoding/decoding via external libraries
//!
//! For full encoding/decoding functionality, users must:
//! 1. Enable the `ffi-ffmpeg` feature and have FFmpeg development libraries installed
//! 2. Ensure proper licensing for commercial use (contact Dolby for licensing)
//!
//! ## Features
//!
//! - Parse AC-3/E-AC-3 bitstream headers and sync frames
//! - Extract audio parameters (sample rate, channels, bitrate)
//! - Detect Dolby metadata (dialnorm, compression profiles, mixing info)
//! - FFI wrapper for FFmpeg libavcodec (optional feature)
//!
//! ## Example
//!
//! ```rust
//! use transcode_ac3::{Ac3Parser, Ac3SyncFrame};
//!
//! // Parse AC-3 bitstream
//! let data = vec![0x0B, 0x77, /* ... AC-3 frame data ... */];
//! let mut parser = Ac3Parser::new();
//!
//! if let Some(frame) = parser.parse_sync_frame(&data) {
//!     println!("Sample rate: {} Hz", frame.sample_rate);
//!     println!("Channels: {}", frame.channels);
//!     println!("Bitrate: {} kbps", frame.bitrate / 1000);
//! }
//! ```
//!
//! ## E-AC-3 (Dolby Digital Plus)
//!
//! ```rust
//! use transcode_ac3::{Eac3Parser, Eac3SyncFrame};
//!
//! // Parse E-AC-3 bitstream
//! let data = vec![0x0B, 0x77, /* ... E-AC-3 frame data ... */];
//! let mut parser = Eac3Parser::new();
//!
//! if let Some(frame) = parser.parse_sync_frame(&data) {
//!     println!("Sample rate: {} Hz", frame.sample_rate);
//!     println!("Stream type: {:?}", frame.stream_type);
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

pub use decoder::{Ac3Decoder, Eac3Decoder};
pub use encoder::{Ac3Encoder, Ac3EncoderConfig, Eac3Encoder, Eac3EncoderConfig};
pub use parser::{Ac3Parser, Eac3Parser};
pub use types::*;

use thiserror::Error;

/// AC-3 codec error types.
#[derive(Error, Debug)]
pub enum Ac3Error {
    /// Invalid sync word in bitstream.
    #[error("Invalid sync word: expected 0x0B77, got {0:#06x}")]
    InvalidSyncWord(u16),

    /// Invalid bitstream ID.
    #[error("Invalid bitstream ID: {0}")]
    InvalidBsid(u8),

    /// Unsupported sample rate.
    #[error("Unsupported sample rate code: {0}")]
    UnsupportedSampleRate(u8),

    /// Unsupported frame size.
    #[error("Unsupported frame size code: {0}")]
    UnsupportedFrameSize(u8),

    /// Invalid audio coding mode.
    #[error("Invalid audio coding mode: {0}")]
    InvalidAcmod(u8),

    /// Bitstream corruption detected.
    #[error("Bitstream corruption: {0}")]
    BitstreamCorruption(String),

    /// CRC mismatch.
    #[error("CRC mismatch: expected {expected:#06x}, got {actual:#06x}")]
    CrcMismatch {
        /// Expected CRC value.
        expected: u16,
        /// Actual CRC value.
        actual: u16,
    },

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

/// Result type for AC-3 operations.
pub type Result<T> = std::result::Result<T, Ac3Error>;

/// AC-3 sync word (0x0B77).
pub const AC3_SYNC_WORD: u16 = 0x0B77;

/// Maximum frame size for AC-3 (3840 bytes for 640 kbps at 48 kHz).
pub const AC3_MAX_FRAME_SIZE: usize = 3840;

/// Maximum channels for AC-3.
pub const AC3_MAX_CHANNELS: usize = 6;

/// Maximum channels for E-AC-3.
pub const EAC3_MAX_CHANNELS: usize = 16;

/// AC-3 audio blocks per frame.
pub const AC3_BLOCKS_PER_FRAME: usize = 6;

/// Samples per audio block.
pub const AC3_SAMPLES_PER_BLOCK: usize = 256;

/// Total samples per AC-3 frame.
pub const AC3_SAMPLES_PER_FRAME: usize = AC3_BLOCKS_PER_FRAME * AC3_SAMPLES_PER_BLOCK;

/// Check if FFI support is available.
///
/// Returns `true` if the `ffi-ffmpeg` feature is enabled and FFmpeg
/// libraries were successfully loaded.
pub fn is_ffi_available() -> bool {
    #[cfg(feature = "ffi-ffmpeg")]
    {
        // Check if FFmpeg is available
        true
    }
    #[cfg(not(feature = "ffi-ffmpeg"))]
    {
        false
    }
}

/// Get codec information string.
pub fn codec_info() -> &'static str {
    if is_ffi_available() {
        "AC-3/E-AC-3 (FFmpeg FFI)"
    } else {
        "AC-3/E-AC-3 (parser only)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(AC3_SYNC_WORD, 0x0B77);
        assert_eq!(AC3_MAX_FRAME_SIZE, 3840);
        assert_eq!(AC3_SAMPLES_PER_FRAME, 1536);
    }

    #[test]
    fn test_ffi_availability() {
        // Without ffi-ffmpeg feature, should return false
        #[cfg(not(feature = "ffi-ffmpeg"))]
        assert!(!is_ffi_available());
    }

    #[test]
    fn test_codec_info() {
        let info = codec_info();
        assert!(info.contains("AC-3"));
    }
}
