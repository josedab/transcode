//! Vorbis audio codec support for the transcode library.
//!
//! **⚠️ EXPERIMENTAL**: This implementation uses simplified floor decoding and
//! encoding algorithms. Output may not be bit-accurate with reference implementations.
//! For production use cases requiring full specification compliance, consider using
//! a binding to libvorbis instead.
//!
//! This crate provides a pure Rust implementation of the Vorbis audio codec,
//! the royalty-free audio compression format commonly used with OGG containers.
//!
//! ## Features
//!
//! - **Decoding**: Vorbis I specification support (simplified floor implementation)
//! - **Encoding**: VBR and ABR encoding modes (simplified floor curves)
//! - **Quality Levels**: -2 to 10 quality range
//! - **Multi-channel**: Up to 8 channels with proper coupling
//! - **Sample Rates**: 8kHz to 192kHz
//!
//! ## Example
//!
//! ```no_run
//! use transcode_vorbis::{VorbisDecoder, VorbisEncoder, VorbisConfig};
//!
//! // Create a decoder
//! let mut decoder = VorbisDecoder::new();
//!
//! // Create an encoder with quality 5
//! let config = VorbisConfig::new(44100, 2).with_quality(5.0);
//! let mut encoder = VorbisEncoder::new(config)?;
//! # Ok::<(), transcode_vorbis::VorbisError>(())
//! ```
//!
//! ## Vorbis Technical Overview
//!
//! Vorbis uses:
//! - Modified Discrete Cosine Transform (MDCT)
//! - Huffman and vector quantization codebooks
//! - Floor curves for spectral envelope
//! - Residue coding for spectral details
//! - Channel coupling for stereo/multi-channel efficiency

#![warn(missing_docs)]
#![allow(clippy::excessive_precision)]

mod decoder;
mod encoder;
mod error;
mod mdct;
mod codebook;
mod floor;
mod residue;

pub use decoder::{VorbisDecoder, VorbisInfo, VorbisComment};
pub use encoder::{VorbisEncoder, VorbisConfig, VorbisPacket, VorbisEncoderStats};
pub use error::{VorbisError, Result};

/// Vorbis codec name.
pub const CODEC_NAME: &str = "vorbis";

/// Vorbis codec long name.
pub const CODEC_LONG_NAME: &str = "Xiph.Org Vorbis";

/// Maximum supported sample rate.
pub const MAX_SAMPLE_RATE: u32 = 192000;

/// Minimum supported sample rate.
pub const MIN_SAMPLE_RATE: u32 = 8000;

/// Maximum number of channels.
pub const MAX_CHANNELS: u8 = 8;

/// Vorbis codec information.
#[derive(Debug, Clone)]
pub struct CodecInfo {
    /// Codec name.
    pub name: &'static str,
    /// Long name.
    pub long_name: &'static str,
    /// Supports encoding.
    pub can_encode: bool,
    /// Supports decoding.
    pub can_decode: bool,
}

/// Get codec information.
pub fn codec_info() -> CodecInfo {
    CodecInfo {
        name: CODEC_NAME,
        long_name: CODEC_LONG_NAME,
        can_encode: true,
        can_decode: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_info() {
        let info = codec_info();
        assert_eq!(info.name, "vorbis");
        assert!(info.can_encode);
        assert!(info.can_decode);
    }
}
