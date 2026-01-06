//! OpenEXR image codec implementation
//!
//! OpenEXR is a high dynamic range (HDR) image format developed by Industrial Light & Magic.
//! It's widely used in the VFX, animation, and film industry for its ability to store
//! high-precision floating-point image data.
//!
//! # Features
//!
//! - Half-float (16-bit) and float (32-bit) pixel data
//! - Multiple compression methods (PIZ, ZIP, ZIPS, RLE, PXR24)
//! - Multi-channel support (RGBA, deep images, arbitrary channels)
//! - Tiled and scanline storage
//!
//! # Example
//!
//! ```ignore
//! use transcode_openexr::{ExrDecoder, ExrEncoder};
//!
//! // Decode an EXR file
//! let decoder = ExrDecoder::new()?;
//! let image = decoder.decode(exr_data)?;
//!
//! // Encode to EXR
//! let encoder = ExrEncoder::new()?;
//! let output = encoder.encode(&image)?;
//! ```

pub mod channel;
pub mod compression;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod header;
pub mod types;

pub use channel::{Channel, ChannelList, PixelType};
pub use compression::Compression;
pub use decoder::ExrDecoder;
pub use encoder::ExrEncoder;
pub use error::{ExrError, Result};
pub use header::Header;
pub use types::{Box2i, DataWindow, DisplayWindow, V2f, V2i};

/// EXR magic number
pub const EXR_MAGIC: u32 = 0x01312F76;

/// Version field value for OpenEXR 2.0
pub const EXR_VERSION: u32 = 2;

/// Version field flags
pub mod version_flags {
    /// Tiled image
    pub const TILED: u32 = 0x200;
    /// Long names (>31 characters)
    pub const LONG_NAMES: u32 = 0x400;
    /// Non-image data (deep data)
    pub const NON_IMAGE: u32 = 0x800;
    /// Multi-part file
    pub const MULTI_PART: u32 = 0x1000;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_number() {
        assert_eq!(EXR_MAGIC, 0x01312F76);
    }

    #[test]
    fn test_version() {
        assert_eq!(EXR_VERSION, 2);
    }
}
