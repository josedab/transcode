//! ProRes constant tables (zigzag, quantization matrices, etc.)

#![allow(dead_code)]

/// Zigzag scan order for 8x8 DCT blocks
pub const ZIGZAG_SCAN: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse zigzag scan order (position to zigzag index)
pub const INVERSE_ZIGZAG: [usize; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

/// Default luma quantization matrix for ProRes
pub const DEFAULT_LUMA_QUANT: [u8; 64] = [
    4, 7, 9, 11, 13, 14, 15, 16, 7, 7, 11, 12, 14, 15, 16, 16, 9, 11, 13, 14, 15, 16, 16, 17, 11,
    12, 14, 15, 16, 16, 17, 17, 13, 14, 15, 16, 16, 17, 17, 18, 14, 15, 16, 16, 17, 17, 18, 18, 15,
    16, 16, 17, 17, 18, 18, 19, 16, 16, 17, 17, 18, 18, 19, 19,
];

/// Default chroma quantization matrix for ProRes
pub const DEFAULT_CHROMA_QUANT: [u8; 64] = [
    4, 7, 9, 11, 13, 14, 15, 16, 7, 7, 11, 12, 14, 15, 16, 16, 9, 11, 13, 14, 15, 16, 16, 17, 11,
    12, 14, 15, 16, 16, 17, 17, 13, 14, 15, 16, 16, 17, 17, 18, 14, 15, 16, 16, 17, 17, 18, 18, 15,
    16, 16, 17, 17, 18, 18, 19, 16, 16, 17, 17, 18, 18, 19, 19,
];

/// ProRes level to quantization scale lookup table
/// This maps the 7-bit quantization level from slice header to actual scale
pub const QUANT_SCALE_TABLE: [u16; 128] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];

/// DC coefficient prediction modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcPredMode {
    /// Predict from left neighbor
    Left,
    /// Predict from top neighbor
    Top,
    /// Predict from top-left neighbor
    TopLeft,
    /// No prediction (use zero)
    None,
}

/// Cosine values for IDCT (scaled by 2^14)
/// cos(n * pi / 16) for n = 1..7
pub const COS_TABLE: [i32; 7] = [
    16069, // cos(pi/16) * 2^14
    15137, // cos(2*pi/16) * 2^14
    13623, // cos(3*pi/16) * 2^14
    11585, // cos(4*pi/16) * 2^14
    9102,  // cos(5*pi/16) * 2^14
    6270,  // cos(6*pi/16) * 2^14
    3196,  // cos(7*pi/16) * 2^14
];

/// IDCT scaling constants
pub const IDCT_SCALE: i32 = 1 << 14;
pub const IDCT_ROUND: i32 = 1 << 13;

/// Get the default quantization matrix for a given profile
pub fn get_default_quant_matrix(is_luma: bool) -> [u8; 64] {
    if is_luma {
        DEFAULT_LUMA_QUANT
    } else {
        DEFAULT_CHROMA_QUANT
    }
}

/// Convert zigzag-ordered coefficients to block order
pub fn dezigzag(zigzag: &[i16; 64]) -> [i16; 64] {
    let mut block = [0i16; 64];
    for (i, &idx) in ZIGZAG_SCAN.iter().enumerate() {
        block[idx] = zigzag[i];
    }
    block
}

/// Convert block-ordered coefficients to zigzag order
pub fn zigzag(block: &[i16; 64]) -> [i16; 64] {
    let mut zigzag = [0i16; 64];
    for (i, &idx) in ZIGZAG_SCAN.iter().enumerate() {
        zigzag[i] = block[idx];
    }
    zigzag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_roundtrip() {
        let original: [i16; 64] = std::array::from_fn(|i| i as i16);
        let zz = zigzag(&original);
        let back = dezigzag(&zz);
        assert_eq!(original, back);
    }

    #[test]
    fn test_zigzag_first_elements() {
        // First few zigzag positions should be: 0, 1, 8, 16, 9, 2, 3, 10...
        assert_eq!(ZIGZAG_SCAN[0], 0);
        assert_eq!(ZIGZAG_SCAN[1], 1);
        assert_eq!(ZIGZAG_SCAN[2], 8);
        assert_eq!(ZIGZAG_SCAN[3], 16);
        assert_eq!(ZIGZAG_SCAN[4], 9);
        assert_eq!(ZIGZAG_SCAN[5], 2);
    }
}
