//! JPEG image codec.
//!
//! This module provides JPEG encoding and decoding with support for:
//! - Baseline JPEG (sequential DCT)
//! - Progressive JPEG
//! - EXIF metadata
//! - Chroma subsampling (4:4:4, 4:2:2, 4:2:0)

mod decoder;
mod encoder;
mod huffman;
mod dct;

pub use decoder::{JpegDecoder, JpegInfo};
pub use encoder::{JpegEncoder, JpegConfig};

/// JPEG marker codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegMarker {
    /// Start of image.
    Soi = 0xD8,
    /// End of image.
    Eoi = 0xD9,
    /// Start of frame (baseline).
    Sof0 = 0xC0,
    /// Start of frame (progressive).
    Sof2 = 0xC2,
    /// Define Huffman table.
    Dht = 0xC4,
    /// Define quantization table.
    Dqt = 0xDB,
    /// Start of scan.
    Sos = 0xDA,
    /// Define restart interval.
    Dri = 0xDD,
    /// Application segment 0 (JFIF).
    App0 = 0xE0,
    /// Application segment 1 (EXIF).
    App1 = 0xE1,
    /// Comment.
    Com = 0xFE,
    /// Restart marker 0.
    Rst0 = 0xD0,
    /// Restart marker 7.
    Rst7 = 0xD7,
}

impl JpegMarker {
    /// Create marker from byte.
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0xD8 => Some(JpegMarker::Soi),
            0xD9 => Some(JpegMarker::Eoi),
            0xC0 => Some(JpegMarker::Sof0),
            0xC2 => Some(JpegMarker::Sof2),
            0xC4 => Some(JpegMarker::Dht),
            0xDB => Some(JpegMarker::Dqt),
            0xDA => Some(JpegMarker::Sos),
            0xDD => Some(JpegMarker::Dri),
            0xE0 => Some(JpegMarker::App0),
            0xE1 => Some(JpegMarker::App1),
            0xFE => Some(JpegMarker::Com),
            0xD0..=0xD7 => Some(JpegMarker::Rst0),
            _ => None,
        }
    }
}

/// JPEG chroma subsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:4:4 - No subsampling.
    Yuv444,
    /// 4:2:2 - Horizontal subsampling.
    Yuv422,
    /// 4:2:0 - Horizontal and vertical subsampling.
    Yuv420,
    /// Grayscale (no chroma).
    Gray,
}

impl Default for ChromaSubsampling {
    fn default() -> Self {
        ChromaSubsampling::Yuv420
    }
}

/// Standard JPEG quantization tables.
pub mod quantization {
    /// Luminance quantization table (quality 50).
    pub const LUMINANCE_50: [u8; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ];

    /// Chrominance quantization table (quality 50).
    pub const CHROMINANCE_50: [u8; 64] = [
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ];

    /// Scale quantization table for quality.
    pub fn scale_table(table: &[u8; 64], quality: u8) -> [u8; 64] {
        let quality = quality.clamp(1, 100);
        let scale = if quality < 50 {
            5000 / quality as u32
        } else {
            200 - quality as u32 * 2
        };

        let mut result = [0u8; 64];
        for i in 0..64 {
            let val = ((table[i] as u32 * scale + 50) / 100).clamp(1, 255);
            result[i] = val as u8;
        }
        result
    }
}

/// Zigzag scan order.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse zigzag order.
pub const IZIGZAG: [usize; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_from_byte() {
        assert_eq!(JpegMarker::from_byte(0xD8), Some(JpegMarker::Soi));
        assert_eq!(JpegMarker::from_byte(0xD9), Some(JpegMarker::Eoi));
        assert_eq!(JpegMarker::from_byte(0x00), None);
    }

    #[test]
    fn test_quantization_scaling() {
        let table = quantization::scale_table(&quantization::LUMINANCE_50, 90);
        // Higher quality = lower quantization values
        assert!(table[0] < quantization::LUMINANCE_50[0]);

        let table = quantization::scale_table(&quantization::LUMINANCE_50, 10);
        // Lower quality = higher quantization values
        assert!(table[0] > quantization::LUMINANCE_50[0]);
    }

    #[test]
    fn test_zigzag() {
        // First few should be DC, then low frequencies
        assert_eq!(ZIGZAG[0], 0);
        assert_eq!(ZIGZAG[1], 1);
        assert_eq!(ZIGZAG[2], 8);
    }
}
