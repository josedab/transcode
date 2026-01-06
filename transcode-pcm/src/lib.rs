//! PCM audio codec support for the transcode library.
//!
//! This crate provides PCM (Pulse Code Modulation) audio encoding and decoding,
//! supporting all common PCM variants used in audio processing.
//!
//! ## Supported Formats
//!
//! - **Signed Integer**: 8, 16, 24, 32-bit (little/big endian)
//! - **Unsigned Integer**: 8-bit
//! - **Floating Point**: 32-bit (f32), 64-bit (f64)
//! - **Compressed PCM**: A-law, mu-law (8-bit)
//! - **Packed Formats**: 20-bit, 24-bit in 32-bit containers
//!
//! ## Example
//!
//! ```no_run
//! use transcode_pcm::{PcmDecoder, PcmEncoder, PcmFormat};
//!
//! // Create a decoder for 16-bit signed little-endian PCM
//! let mut decoder = PcmDecoder::new(PcmFormat::S16Le, 44100, 2);
//!
//! // Create an encoder for 32-bit float PCM
//! let mut encoder = PcmEncoder::new(PcmFormat::F32Le, 48000, 2);
//! ```

#![warn(missing_docs)]

mod decoder;
mod encoder;
mod error;
mod format;

pub use decoder::PcmDecoder;
pub use encoder::PcmEncoder;
pub use error::{PcmError, Result};
pub use format::PcmFormat;

/// PCM codec information.
pub const CODEC_NAME: &str = "pcm";

/// PCM codec long name.
pub const CODEC_LONG_NAME: &str = "Uncompressed PCM Audio";

/// Get codec information for a specific PCM format.
pub fn codec_info(format: PcmFormat) -> CodecInfo {
    CodecInfo {
        name: format.codec_name(),
        long_name: format.codec_long_name(),
        format,
        bits_per_sample: format.bits_per_sample(),
        is_float: format.is_float(),
        is_signed: format.is_signed(),
        is_big_endian: format.is_big_endian(),
    }
}

/// Codec information structure.
#[derive(Debug, Clone)]
pub struct CodecInfo {
    /// Short codec name.
    pub name: &'static str,
    /// Long descriptive name.
    pub long_name: &'static str,
    /// PCM format.
    pub format: PcmFormat,
    /// Bits per sample.
    pub bits_per_sample: u8,
    /// Whether format uses floating point.
    pub is_float: bool,
    /// Whether format is signed.
    pub is_signed: bool,
    /// Whether format is big endian.
    pub is_big_endian: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_info() {
        let info = codec_info(PcmFormat::S16Le);
        assert_eq!(info.name, "pcm_s16le");
        assert_eq!(info.bits_per_sample, 16);
        assert!(info.is_signed);
        assert!(!info.is_big_endian);
        assert!(!info.is_float);
    }

    #[test]
    fn test_all_formats() {
        for format in PcmFormat::all() {
            let info = codec_info(format);
            assert!(info.bits_per_sample > 0);
        }
    }
}
