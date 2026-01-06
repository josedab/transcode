//! DNxHD lookup tables and constants

/// Zigzag scan order for 8x8 blocks
pub const ZIGZAG_SCAN: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Inverse zigzag scan (position to zigzag index)
pub const ZIGZAG_INVERSE: [usize; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

/// DNxHD quantization matrix for luma (low quality profiles)
pub const QUANT_MATRIX_LUMA_LQ: [u16; 64] = [
    32, 32, 32, 33, 33, 33, 34, 34, 32, 32, 33, 33, 33, 34, 35, 36, 32, 33, 33, 33, 34, 35, 36, 38,
    33, 33, 33, 34, 35, 36, 38, 40, 33, 33, 34, 35, 36, 38, 40, 42, 33, 34, 35, 36, 38, 40, 42, 44,
    34, 35, 36, 38, 40, 42, 44, 46, 34, 36, 38, 40, 42, 44, 46, 48,
];

/// DNxHD quantization matrix for luma (high quality profiles)
pub const QUANT_MATRIX_LUMA_HQ: [u16; 64] = [
    16, 16, 16, 16, 17, 17, 18, 18, 16, 16, 16, 17, 17, 18, 18, 19, 16, 16, 17, 17, 18, 18, 19, 20,
    16, 17, 17, 18, 18, 19, 20, 21, 17, 17, 18, 18, 19, 20, 21, 22, 17, 18, 18, 19, 20, 21, 22, 23,
    18, 18, 19, 20, 21, 22, 23, 24, 18, 19, 20, 21, 22, 23, 24, 25,
];

/// DNxHD quantization matrix for chroma (low quality profiles)
pub const QUANT_MATRIX_CHROMA_LQ: [u16; 64] = [
    32, 33, 34, 35, 36, 38, 40, 42, 33, 33, 35, 36, 38, 40, 42, 44, 34, 35, 36, 38, 40, 42, 44, 46,
    35, 36, 38, 40, 42, 44, 46, 48, 36, 38, 40, 42, 44, 46, 48, 51, 38, 40, 42, 44, 46, 48, 51, 54,
    40, 42, 44, 46, 48, 51, 54, 57, 42, 44, 46, 48, 51, 54, 57, 60,
];

/// DNxHD quantization matrix for chroma (high quality profiles)
pub const QUANT_MATRIX_CHROMA_HQ: [u16; 64] = [
    16, 17, 17, 18, 18, 19, 20, 21, 17, 17, 18, 18, 19, 20, 21, 22, 17, 18, 18, 19, 20, 21, 22, 23,
    18, 18, 19, 20, 21, 22, 23, 24, 18, 19, 20, 21, 22, 23, 24, 25, 19, 20, 21, 22, 23, 24, 25, 26,
    20, 21, 22, 23, 24, 25, 26, 28, 21, 22, 23, 24, 25, 26, 28, 30,
];

/// DCT cosine coefficients scaled by 2^14
pub const DCT_COS: [i32; 8] = [
    16384, // cos(0 * pi/16) * 2^14
    16069, // cos(1 * pi/16) * 2^14
    15137, // cos(2 * pi/16) * 2^14
    13623, // cos(3 * pi/16) * 2^14
    11585, // cos(4 * pi/16) * 2^14
    9102,  // cos(5 * pi/16) * 2^14
    6270,  // cos(6 * pi/16) * 2^14
    3196,  // cos(7 * pi/16) * 2^14
];

/// IDCT scaling constant
pub const IDCT_SCALE: i32 = 16384;

/// Run-level escape code
pub const RUN_LEVEL_ESCAPE: u32 = 0x1FF;

/// DC prediction reset value for 8-bit
pub const DC_PRED_8BIT: i16 = 128;

/// DC prediction reset value for 10-bit
pub const DC_PRED_10BIT: i16 = 512;

/// DC prediction reset value for 12-bit
pub const DC_PRED_12BIT: i16 = 2048;

/// Macroblock size in pixels
pub const MB_SIZE: usize = 16;

/// DCT block size
pub const BLOCK_SIZE: usize = 8;

/// Number of coefficients in a block
pub const BLOCK_COEFFS: usize = 64;

/// Frame header size in bytes
pub const FRAME_HEADER_SIZE: usize = 640;

/// Slice header size in bytes (without coefficient data)
pub const SLICE_HEADER_SIZE: usize = 12;

/// Run-level table for DNxHD AC coefficients (simplified)
/// Format: (run, level) -> (code, bits)
pub struct RunLevelTable {
    /// Maximum run length
    pub max_run: usize,
    /// Maximum level
    pub max_level: usize,
}

impl RunLevelTable {
    /// Create default run-level table
    pub const fn new() -> Self {
        RunLevelTable {
            max_run: 63,
            max_level: 2047,
        }
    }

    /// Get the maximum run for a given level
    pub fn max_run_for_level(&self, level: i16) -> usize {
        let abs_level = level.unsigned_abs() as usize;
        if abs_level == 0 {
            0
        } else if abs_level <= 4 {
            63 - abs_level * 8
        } else {
            0
        }
    }
}

impl Default for RunLevelTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Weight matrix for DNxHD encoding quality control
#[derive(Debug, Clone)]
pub struct WeightMatrix {
    /// Luma weights
    pub luma: [u16; 64],
    /// Chroma weights
    pub chroma: [u16; 64],
}

impl WeightMatrix {
    /// Create weight matrix for a given quality level (1-100)
    pub fn for_quality(quality: u32) -> Self {
        let scale = if quality < 50 {
            5000 / quality.max(1)
        } else {
            200 - quality * 2
        };

        let mut luma = [0u16; 64];
        let mut chroma = [0u16; 64];

        for i in 0..64 {
            luma[i] = ((QUANT_MATRIX_LUMA_HQ[i] as u32 * scale + 50) / 100).clamp(1, 255) as u16;
            chroma[i] =
                ((QUANT_MATRIX_CHROMA_HQ[i] as u32 * scale + 50) / 100).clamp(1, 255) as u16;
        }

        WeightMatrix { luma, chroma }
    }
}

/// Get the quantization matrix based on profile
pub fn get_quant_matrix(is_luma: bool, high_quality: bool) -> &'static [u16; 64] {
    match (is_luma, high_quality) {
        (true, true) => &QUANT_MATRIX_LUMA_HQ,
        (true, false) => &QUANT_MATRIX_LUMA_LQ,
        (false, true) => &QUANT_MATRIX_CHROMA_HQ,
        (false, false) => &QUANT_MATRIX_CHROMA_LQ,
    }
}

/// Apply zigzag scan to a block
pub fn zigzag_block(block: &[i16; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];
    for (i, &idx) in ZIGZAG_SCAN.iter().enumerate() {
        output[i] = block[idx];
    }
    output
}

/// Apply inverse zigzag scan to a block
pub fn inverse_zigzag_block(zigzag: &[i16; 64]) -> [i16; 64] {
    let mut output = [0i16; 64];
    for (i, &idx) in ZIGZAG_SCAN.iter().enumerate() {
        output[idx] = zigzag[i];
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_inverse() {
        // Verify zigzag and inverse are correct inverses
        let mut block = [0i16; 64];
        for i in 0..64 {
            block[i] = i as i16;
        }

        let zigzagged = zigzag_block(&block);
        let restored = inverse_zigzag_block(&zigzagged);

        assert_eq!(block, restored);
    }

    #[test]
    fn test_zigzag_order() {
        // First few elements should be in correct zigzag order
        assert_eq!(ZIGZAG_SCAN[0], 0); // (0,0)
        assert_eq!(ZIGZAG_SCAN[1], 1); // (0,1)
        assert_eq!(ZIGZAG_SCAN[2], 8); // (1,0)
        assert_eq!(ZIGZAG_SCAN[3], 16); // (2,0)
        assert_eq!(ZIGZAG_SCAN[4], 9); // (1,1)
    }

    #[test]
    fn test_weight_matrix_quality() {
        let low = WeightMatrix::for_quality(10);
        let high = WeightMatrix::for_quality(90);

        // Lower quality should have higher quantization weights
        assert!(low.luma[0] > high.luma[0]);
    }

    #[test]
    fn test_run_level_table() {
        let table = RunLevelTable::new();
        assert_eq!(table.max_run, 63);
        assert_eq!(table.max_level, 2047);
    }
}
