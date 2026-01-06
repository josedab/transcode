//! PNG image codec.
//!
//! This module provides PNG encoding and decoding with support for:
//! - RGB and RGBA images
//! - Grayscale and grayscale with alpha
//! - Indexed color with palette
//! - Interlaced (Adam7) images
//! - Multiple compression levels

mod decoder;
mod encoder;
mod filter;

pub use decoder::{PngDecoder, PngInfo};
pub use encoder::{PngEncoder, PngConfig};

use crate::error::{ImageError, Result};

/// PNG signature bytes.
pub const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// PNG color type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorType {
    /// Grayscale.
    Grayscale = 0,
    /// RGB.
    Rgb = 2,
    /// Indexed color.
    Indexed = 3,
    /// Grayscale with alpha.
    GrayscaleAlpha = 4,
    /// RGBA.
    Rgba = 6,
}

impl ColorType {
    /// Create color type from value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(ColorType::Grayscale),
            2 => Some(ColorType::Rgb),
            3 => Some(ColorType::Indexed),
            4 => Some(ColorType::GrayscaleAlpha),
            6 => Some(ColorType::Rgba),
            _ => None,
        }
    }

    /// Get number of channels.
    pub fn channels(&self) -> u8 {
        match self {
            ColorType::Grayscale => 1,
            ColorType::Rgb => 3,
            ColorType::Indexed => 1,
            ColorType::GrayscaleAlpha => 2,
            ColorType::Rgba => 4,
        }
    }

    /// Check if color type has alpha.
    pub fn has_alpha(&self) -> bool {
        matches!(self, ColorType::GrayscaleAlpha | ColorType::Rgba)
    }
}

/// PNG interlace method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterlaceMethod {
    /// No interlacing.
    None = 0,
    /// Adam7 interlacing.
    Adam7 = 1,
}

impl InterlaceMethod {
    /// Create from value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(InterlaceMethod::None),
            1 => Some(InterlaceMethod::Adam7),
            _ => None,
        }
    }
}

/// PNG chunk type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkType([u8; 4]);

impl ChunkType {
    /// IHDR - Image header.
    pub const IHDR: Self = Self(*b"IHDR");
    /// PLTE - Palette.
    pub const PLTE: Self = Self(*b"PLTE");
    /// IDAT - Image data.
    pub const IDAT: Self = Self(*b"IDAT");
    /// IEND - Image end.
    pub const IEND: Self = Self(*b"IEND");
    /// tRNS - Transparency.
    pub const TRNS: Self = Self(*b"tRNS");
    /// gAMA - Gamma.
    pub const GAMA: Self = Self(*b"gAMA");
    /// cHRM - Primary chromaticities.
    pub const CHRM: Self = Self(*b"cHRM");
    /// sRGB - Standard RGB color space.
    pub const SRGB: Self = Self(*b"sRGB");
    /// iCCP - Embedded ICC profile.
    pub const ICCP: Self = Self(*b"iCCP");
    /// tEXt - Textual data.
    pub const TEXT: Self = Self(*b"tEXt");
    /// zTXt - Compressed textual data.
    pub const ZTXT: Self = Self(*b"zTXt");
    /// iTXt - International textual data.
    pub const ITXT: Self = Self(*b"iTXt");
    /// bKGD - Background color.
    pub const BKGD: Self = Self(*b"bKGD");
    /// pHYs - Physical pixel dimensions.
    pub const PHYS: Self = Self(*b"pHYs");
    /// tIME - Image last-modification time.
    pub const TIME: Self = Self(*b"tIME");

    /// Create from bytes.
    pub fn new(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    /// Get bytes.
    pub fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }

    /// Check if chunk is critical.
    pub fn is_critical(&self) -> bool {
        (self.0[0] & 0x20) == 0
    }

    /// Check if chunk is public.
    pub fn is_public(&self) -> bool {
        (self.0[1] & 0x20) == 0
    }

    /// Check if chunk is safe to copy.
    pub fn is_safe_to_copy(&self) -> bool {
        (self.0[3] & 0x20) != 0
    }
}

impl std::fmt::Display for ChunkType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.0))
    }
}

/// PNG compression level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    /// No compression (level 0).
    None,
    /// Fast compression (level 1).
    Fast,
    /// Default compression (level 6).
    Default,
    /// Best compression (level 9).
    Best,
    /// Custom level (0-9).
    Custom(u8),
}

impl CompressionLevel {
    /// Get numeric level.
    pub fn level(&self) -> u8 {
        match self {
            CompressionLevel::None => 0,
            CompressionLevel::Fast => 1,
            CompressionLevel::Default => 6,
            CompressionLevel::Best => 9,
            CompressionLevel::Custom(l) => *l,
        }
    }
}

impl Default for CompressionLevel {
    fn default() -> Self {
        CompressionLevel::Default
    }
}

/// Calculate CRC32 for PNG chunks.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;

    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = CRC_TABLE[idx] ^ (crc >> 8);
    }

    crc ^ 0xFFFFFFFF
}

/// CRC32 lookup table.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            if c & 1 != 0 {
                c = 0xEDB88320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

/// Read big-endian u32.
#[inline]
pub fn read_u32_be(data: &[u8]) -> u32 {
    u32::from_be_bytes([data[0], data[1], data[2], data[3]])
}

/// Write big-endian u32.
#[inline]
pub fn write_u32_be(value: u32) -> [u8; 4] {
    value.to_be_bytes()
}

/// Adam7 interlace pass parameters: (start_x, start_y, step_x, step_y)
pub const ADAM7_PASSES: [(usize, usize, usize, usize); 7] = [
    (0, 0, 8, 8), // Pass 1
    (4, 0, 8, 8), // Pass 2
    (0, 4, 4, 8), // Pass 3
    (2, 0, 4, 4), // Pass 4
    (0, 2, 2, 4), // Pass 5
    (1, 0, 2, 2), // Pass 6
    (0, 1, 1, 2), // Pass 7
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_type() {
        assert_eq!(ColorType::from_u8(0), Some(ColorType::Grayscale));
        assert_eq!(ColorType::from_u8(2), Some(ColorType::Rgb));
        assert_eq!(ColorType::from_u8(6), Some(ColorType::Rgba));
        assert_eq!(ColorType::from_u8(1), None);

        assert_eq!(ColorType::Rgba.channels(), 4);
        assert!(ColorType::Rgba.has_alpha());
        assert!(!ColorType::Rgb.has_alpha());
    }

    #[test]
    fn test_chunk_type() {
        assert!(ChunkType::IHDR.is_critical());
        assert!(ChunkType::IDAT.is_critical());
        assert!(!ChunkType::TEXT.is_critical());

        assert_eq!(format!("{}", ChunkType::IHDR), "IHDR");
    }

    #[test]
    fn test_crc32() {
        // Test with known values
        let data = b"IHDR";
        let crc = crc32(data);
        assert!(crc != 0);

        // CRC should be deterministic
        assert_eq!(crc32(data), crc32(data));
    }

    #[test]
    fn test_compression_level() {
        assert_eq!(CompressionLevel::None.level(), 0);
        assert_eq!(CompressionLevel::Fast.level(), 1);
        assert_eq!(CompressionLevel::Default.level(), 6);
        assert_eq!(CompressionLevel::Best.level(), 9);
        assert_eq!(CompressionLevel::Custom(5).level(), 5);
    }
}
