//! TIFF image codec implementation
//!
//! Tagged Image File Format (TIFF) is a flexible container format for storing
//! raster graphics images, commonly used in archival and professional workflows.
//!
//! # Features
//!
//! - Multiple compression methods (None, LZW, PackBits, Deflate)
//! - Support for various color spaces (RGB, RGBA, Grayscale, CMYK)
//! - Multiple bits per sample (8, 16, 32)
//! - Big and little endian support
//! - Multi-page TIFF support
//!
//! # Example
//!
//! ```ignore
//! use transcode_tiff::{TiffDecoder, TiffEncoder};
//!
//! // Decode a TIFF file
//! let decoder = TiffDecoder::new();
//! let image = decoder.decode(tiff_data)?;
//!
//! // Encode to TIFF
//! let encoder = TiffEncoder::new();
//! let output = encoder.encode(&image)?;
//! ```

pub mod compression;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod ifd;
pub mod tags;
pub mod types;

pub use compression::Compression;
pub use decoder::TiffDecoder;
pub use encoder::TiffEncoder;
pub use error::{TiffError, Result};
pub use ifd::{Ifd, IfdEntry};
pub use types::{ColorSpace, TiffImage};

/// TIFF magic number - little endian "II"
pub const TIFF_MAGIC_LE: [u8; 2] = [0x49, 0x49];

/// TIFF magic number - big endian "MM"
pub const TIFF_MAGIC_BE: [u8; 2] = [0x4D, 0x4D];

/// TIFF version (42)
pub const TIFF_VERSION: u16 = 42;

/// BigTIFF version (43)
pub const BIGTIFF_VERSION: u16 = 43;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_numbers() {
        assert_eq!(TIFF_MAGIC_LE, [b'I', b'I']);
        assert_eq!(TIFF_MAGIC_BE, [b'M', b'M']);
    }

    #[test]
    fn test_version() {
        assert_eq!(TIFF_VERSION, 42);
    }
}
