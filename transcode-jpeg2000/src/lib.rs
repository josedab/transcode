//! JPEG2000 codec support for DCP/cinema workflows.
//!
//! This crate provides JPEG2000 (J2K/JP2) codec support with optional FFI
//! integration via OpenJPEG for full encoding/decoding capabilities.
//!
//! ## Features
//!
//! - **Native parsing**: Codestream and file format parsing without external dependencies
//! - **FFI encoding/decoding**: Full JPEG2000 support via OpenJPEG (with `ffi-openjpeg` feature)
//! - **DCP support**: Digital Cinema Package workflows
//! - **Multiple profiles**: Cinema 2K/4K, Broadcast, Lossless
//!
//! ## Usage
//!
//! ### Basic parsing (no FFI required)
//!
//! ```rust,ignore
//! use transcode_jpeg2000::{Jpeg2000Parser, CodestreamInfo};
//!
//! let mut parser = Jpeg2000Parser::new();
//! // Parse codestream header
//! let info = parser.parse_header(&data)?;
//! println!("Image: {}x{}, {} components", info.width, info.height, info.num_components);
//! ```
//!
//! ### Full decoding (requires `ffi-openjpeg` feature)
//!
//! ```rust,ignore
//! use transcode_jpeg2000::{Jpeg2000Decoder, DecodedImage};
//!
//! let mut decoder = Jpeg2000Decoder::new()?;
//! let image = decoder.decode(&j2k_data)?;
//! ```
//!
//! ## JPEG2000 Format Overview
//!
//! JPEG2000 uses discrete wavelet transform (DWT) for compression:
//! - **Lossy**: 9/7 irreversible wavelet with CDF 9/7 filters
//! - **Lossless**: 5/3 reversible wavelet with Le Gall 5/3 filters
//!
//! The format supports:
//! - Tile-based encoding for large images
//! - Multiple resolution levels (precincts)
//! - Region of interest (ROI) coding
//! - Progressive decoding by quality, resolution, or position

pub mod decoder;
pub mod encoder;
pub mod parser;
pub mod types;

pub use decoder::{Jpeg2000Decoder, DecodedImage};
pub use encoder::{Jpeg2000Encoder, Jpeg2000EncoderConfig};
pub use parser::{Jpeg2000Parser, CodestreamInfo};
pub use types::*;

use thiserror::Error;

/// JPEG2000 codec errors.
#[derive(Debug, Error)]
pub enum Jpeg2000Error {
    /// Invalid or corrupted codestream.
    #[error("Invalid codestream: {0}")]
    InvalidCodestream(String),

    /// Unsupported feature.
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// Missing required marker.
    #[error("Missing required marker: {marker:?}")]
    MissingMarker { marker: MarkerType },

    /// Invalid marker segment.
    #[error("Invalid marker segment: {marker:?}")]
    InvalidMarkerSegment { marker: MarkerType },

    /// Invalid image dimensions.
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// Unsupported bit depth.
    #[error("Unsupported bit depth: {0}")]
    UnsupportedBitDepth(u8),

    /// Unsupported number of components.
    #[error("Unsupported component count: {0}")]
    UnsupportedComponentCount(u16),

    /// Encoding error.
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Decoding error.
    #[error("Decoding error: {0}")]
    DecodingError(String),

    /// FFI not available (feature not enabled).
    #[error("OpenJPEG FFI not available - enable 'ffi-openjpeg' feature")]
    FfiNotAvailable,

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Buffer too small.
    #[error("Buffer too small: need {needed} bytes, have {available}")]
    BufferTooSmall { needed: usize, available: usize },
}

/// Result type for JPEG2000 operations.
pub type Result<T> = std::result::Result<T, Jpeg2000Error>;

/// JPEG2000 marker types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerType {
    /// Start of codestream (SOC).
    Soc,
    /// Image and tile size (SIZ).
    Siz,
    /// Coding style default (COD).
    Cod,
    /// Coding style component (COC).
    Coc,
    /// Region of interest (RGN).
    Rgn,
    /// Quantization default (QCD).
    Qcd,
    /// Quantization component (QCC).
    Qcc,
    /// Progression order change (POC).
    Poc,
    /// Tile-part lengths (TLM).
    Tlm,
    /// Packet length, main header (PLM).
    Plm,
    /// Packet length, tile-part header (PLT).
    Plt,
    /// Packed packet headers, main header (PPM).
    Ppm,
    /// Packed packet headers, tile-part header (PPT).
    Ppt,
    /// Start of tile-part (SOT).
    Sot,
    /// Start of data (SOD).
    Sod,
    /// End of codestream (EOC).
    Eoc,
    /// Comment (COM).
    Com,
    /// Unknown marker.
    Unknown(u16),
}

impl MarkerType {
    /// Get the marker code.
    pub fn code(&self) -> u16 {
        match self {
            MarkerType::Soc => 0xFF4F,
            MarkerType::Siz => 0xFF51,
            MarkerType::Cod => 0xFF52,
            MarkerType::Coc => 0xFF53,
            MarkerType::Rgn => 0xFF5E,
            MarkerType::Qcd => 0xFF5C,
            MarkerType::Qcc => 0xFF5D,
            MarkerType::Poc => 0xFF5F,
            MarkerType::Tlm => 0xFF55,
            MarkerType::Plm => 0xFF57,
            MarkerType::Plt => 0xFF58,
            MarkerType::Ppm => 0xFF60,
            MarkerType::Ppt => 0xFF61,
            MarkerType::Sot => 0xFF90,
            MarkerType::Sod => 0xFF93,
            MarkerType::Eoc => 0xFFD9,
            MarkerType::Com => 0xFF64,
            MarkerType::Unknown(code) => *code,
        }
    }

    /// Create from marker code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0xFF4F => MarkerType::Soc,
            0xFF51 => MarkerType::Siz,
            0xFF52 => MarkerType::Cod,
            0xFF53 => MarkerType::Coc,
            0xFF5E => MarkerType::Rgn,
            0xFF5C => MarkerType::Qcd,
            0xFF5D => MarkerType::Qcc,
            0xFF5F => MarkerType::Poc,
            0xFF55 => MarkerType::Tlm,
            0xFF57 => MarkerType::Plm,
            0xFF58 => MarkerType::Plt,
            0xFF60 => MarkerType::Ppm,
            0xFF61 => MarkerType::Ppt,
            0xFF90 => MarkerType::Sot,
            0xFF93 => MarkerType::Sod,
            0xFFD9 => MarkerType::Eoc,
            0xFF64 => MarkerType::Com,
            _ => MarkerType::Unknown(code),
        }
    }

    /// Check if this marker has a length field.
    pub fn has_length(&self) -> bool {
        !matches!(self, MarkerType::Soc | MarkerType::Sod | MarkerType::Eoc)
    }
}

/// Codec information.
pub struct Jpeg2000CodecInfo;

impl Jpeg2000CodecInfo {
    /// Get the codec name.
    pub fn name() -> &'static str {
        "JPEG2000"
    }

    /// Get the codec short name.
    pub fn short_name() -> &'static str {
        "J2K"
    }

    /// Check if FFI decoding is available.
    pub fn is_decoding_available() -> bool {
        cfg!(feature = "ffi-openjpeg")
    }

    /// Check if FFI encoding is available.
    pub fn is_encoding_available() -> bool {
        cfg!(feature = "ffi-openjpeg")
    }

    /// Get supported file extensions.
    pub fn extensions() -> &'static [&'static str] {
        &["j2k", "j2c", "jp2", "jpx", "jpf"]
    }

    /// Get MIME types.
    pub fn mime_types() -> &'static [&'static str] {
        &["image/jp2", "image/jpx", "image/j2c"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_code() {
        assert_eq!(MarkerType::Soc.code(), 0xFF4F);
        assert_eq!(MarkerType::Siz.code(), 0xFF51);
        assert_eq!(MarkerType::Cod.code(), 0xFF52);
        assert_eq!(MarkerType::Eoc.code(), 0xFFD9);
    }

    #[test]
    fn test_marker_from_code() {
        assert_eq!(MarkerType::from_code(0xFF4F), MarkerType::Soc);
        assert_eq!(MarkerType::from_code(0xFF51), MarkerType::Siz);
        assert_eq!(MarkerType::from_code(0xFFD9), MarkerType::Eoc);
        assert!(matches!(MarkerType::from_code(0xFF00), MarkerType::Unknown(0xFF00)));
    }

    #[test]
    fn test_marker_has_length() {
        assert!(!MarkerType::Soc.has_length());
        assert!(!MarkerType::Sod.has_length());
        assert!(!MarkerType::Eoc.has_length());
        assert!(MarkerType::Siz.has_length());
        assert!(MarkerType::Cod.has_length());
    }

    #[test]
    fn test_codec_info() {
        assert_eq!(Jpeg2000CodecInfo::name(), "JPEG2000");
        assert_eq!(Jpeg2000CodecInfo::short_name(), "J2K");
        assert!(Jpeg2000CodecInfo::extensions().contains(&"j2k"));
        assert!(Jpeg2000CodecInfo::extensions().contains(&"jp2"));
    }

    #[test]
    fn test_ffi_availability() {
        #[cfg(feature = "ffi-openjpeg")]
        assert!(Jpeg2000CodecInfo::is_decoding_available());
        #[cfg(not(feature = "ffi-openjpeg"))]
        assert!(!Jpeg2000CodecInfo::is_decoding_available());
    }
}
