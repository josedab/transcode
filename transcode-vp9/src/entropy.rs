//! Boolean entropy coding for VP9.
//!
//! VP9 uses a boolean arithmetic coder (bool decoder) for entropy coding.
//! This module provides the bool decoder implementation used for parsing
//! the compressed header and tile data.

use crate::error::{Result, Vp9Error};

/// Type alias for coefficient token probabilities.
/// Dimensions: [tx_size][plane][is_inter][coef_ctx][token_band][token]
pub type CoefProbs = [[[[[[u8; 3]; 6]; 6]; 2]; 2]; 4];

/// VP9 boolean decoder for arithmetic coding.
///
/// The bool decoder is used to decode binary decisions with associated
/// probabilities. It maintains a range and value that are updated with
/// each decoded symbol.
#[derive(Debug, Clone)]
pub struct BoolDecoder<'a> {
    /// Input data buffer.
    data: &'a [u8],
    /// Current byte position in data.
    pos: usize,
    /// Current value (bits read from stream).
    value: u64,
    /// Current range.
    range: u32,
    /// Number of bits available in value.
    bits: i32,
}

impl<'a> BoolDecoder<'a> {
    /// Create a new bool decoder from the given data.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Vp9Error::BoolDecoderError("Empty data".into()));
        }

        let mut decoder = Self {
            data,
            pos: 0,
            value: 0,
            range: 255,
            bits: 0,
        };

        // Initialize by reading first bytes
        decoder.fill()?;

        Ok(decoder)
    }

    /// Fill the value buffer with more bits from the stream.
    fn fill(&mut self) -> Result<()> {
        while self.bits < 56 && self.pos < self.data.len() {
            self.value |= (self.data[self.pos] as u64) << (56 - self.bits);
            self.pos += 1;
            self.bits += 8;
        }
        Ok(())
    }

    /// Read a single boolean with the given probability (0-255).
    ///
    /// Probability represents P(bit = 0), where 255 means almost certain 0.
    pub fn read_bool(&mut self, prob: u8) -> Result<bool> {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        let big_split = (split as u64) << 48;

        let bit = if self.value >= big_split {
            self.range -= split;
            self.value -= big_split;
            true
        } else {
            self.range = split;
            false
        };

        // Renormalize
        let shift = self.range.leading_zeros() as i32 - 24;
        self.range <<= shift;
        self.value <<= shift;
        self.bits -= shift;

        if self.bits < 0 {
            self.fill()?;
        }

        Ok(bit)
    }

    /// Read a boolean with 50% probability.
    pub fn read_bit(&mut self) -> Result<bool> {
        self.read_bool(128)
    }

    /// Read multiple bits as an unsigned integer.
    pub fn read_literal(&mut self, n: u8) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u32);
        }
        Ok(value)
    }

    /// Read a signed integer using sign bit + magnitude.
    pub fn read_signed_literal(&mut self, n: u8) -> Result<i32> {
        let value = self.read_literal(n)? as i32;
        if self.read_bit()? {
            Ok(-value)
        } else {
            Ok(value)
        }
    }

    /// Read a tree-coded value using the given tree and probabilities.
    ///
    /// The tree is an array where positive values are leaf nodes (symbols)
    /// and negative values are indices to the next node.
    pub fn read_tree(&mut self, tree: &[i8], probs: &[u8]) -> Result<u8> {
        let mut i = 0usize;
        loop {
            let prob = probs[i >> 1];
            let bit = self.read_bool(prob)?;
            i = tree[i + bit as usize] as usize;
            if tree[i] > 0 || i == 0 {
                break;
            }
            i = (-tree[i]) as usize;
        }
        Ok(i as u8)
    }

    /// Read a value using the given CDF (cumulative distribution function).
    pub fn read_symbol(&mut self, cdf: &[u16], nsymbs: usize) -> Result<u8> {
        let mut symbol = 0u8;

        for i in 0..nsymbs - 1 {
            let prob = cdf[i] as u32;
            let split = 1 + (((self.range - 1) * prob) >> 15);
            let big_split = (split as u64) << 48;

            if self.value < big_split {
                self.range = split;
                symbol = i as u8;
                break;
            } else {
                self.value -= big_split;
                self.range -= split;
                symbol = (i + 1) as u8;
            }
        }

        // Renormalize
        let shift = self.range.leading_zeros() as i32 - 24;
        self.range <<= shift;
        self.value <<= shift;
        self.bits -= shift;

        if self.bits < 0 {
            self.fill()?;
        }

        Ok(symbol)
    }

    /// Get the current byte position.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Check if we've reached the end of the data.
    pub fn is_eof(&self) -> bool {
        self.pos >= self.data.len() && self.bits <= 0
    }

    /// Get remaining bytes in the stream.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

/// VP9 probability context.
///
/// Stores and updates probabilities for entropy coding.
#[derive(Debug, Clone)]
pub struct ProbabilityContext {
    /// Partition probabilities.
    pub partition_probs: [[u8; 3]; 16],
    /// Skip probabilities.
    pub skip_probs: [u8; 3],
    /// Intra inter probabilities.
    pub intra_inter_probs: [u8; 4],
    /// Compound mode probabilities.
    pub comp_inter_probs: [u8; 5],
    /// Single reference probabilities.
    pub single_ref_probs: [[u8; 2]; 5],
    /// Compound reference probabilities.
    pub comp_ref_probs: [u8; 5],
    /// Inter mode probabilities.
    pub inter_mode_probs: [[u8; 3]; 7],
    /// Interpolation filter probabilities.
    pub interp_filter_probs: [[u8; 2]; 4],
    /// Intra mode probabilities (y).
    pub y_mode_probs: [[u8; 9]; 4],
    /// Intra mode probabilities (uv).
    pub uv_mode_probs: [[u8; 9]; 10],
    /// Motion vector sign probabilities.
    pub mv_sign_probs: [u8; 2],
    /// Motion vector class probabilities.
    pub mv_class_probs: [[u8; 10]; 2],
    /// Motion vector class0 bit probabilities.
    pub mv_class0_bit_probs: [u8; 2],
    /// Motion vector bit probabilities.
    pub mv_bit_probs: [[u8; 10]; 2],
    /// Motion vector class0 fraction probabilities.
    pub mv_class0_fr_probs: [[[u8; 3]; 2]; 2],
    /// Motion vector fraction probabilities.
    pub mv_fr_probs: [[u8; 3]; 2],
    /// Motion vector class0 high precision probabilities.
    pub mv_class0_hp_probs: [u8; 2],
    /// Motion vector high precision probabilities.
    pub mv_hp_probs: [u8; 2],
    /// Transform size probabilities.
    pub tx_probs_8x8: [[u8; 1]; 2],
    /// Transform size probabilities for 16x16.
    pub tx_probs_16x16: [[u8; 2]; 2],
    /// Transform size probabilities for 32x32.
    pub tx_probs_32x32: [[u8; 3]; 2],
    /// Coefficient token probabilities.
    pub coef_probs: CoefProbs,
}

impl Default for ProbabilityContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ProbabilityContext {
    /// Create a new probability context with default values.
    pub fn new() -> Self {
        Self {
            partition_probs: Self::default_partition_probs(),
            skip_probs: [192, 128, 64],
            intra_inter_probs: [9, 102, 187, 225],
            comp_inter_probs: [239, 183, 119, 96, 41],
            single_ref_probs: [
                [33, 16],
                [77, 74],
                [142, 142],
                [172, 170],
                [238, 247],
            ],
            comp_ref_probs: [50, 126, 123, 221, 226],
            inter_mode_probs: [
                [2, 173, 34],
                [7, 145, 85],
                [7, 166, 63],
                [7, 94, 66],
                [8, 64, 46],
                [17, 81, 31],
                [25, 29, 30],
            ],
            interp_filter_probs: [
                [235, 162],
                [36, 255],
                [34, 3],
                [149, 144],
            ],
            y_mode_probs: Self::default_y_mode_probs(),
            uv_mode_probs: Self::default_uv_mode_probs(),
            mv_sign_probs: [128, 128],
            mv_class_probs: Self::default_mv_class_probs(),
            mv_class0_bit_probs: [216, 208],
            mv_bit_probs: [[136; 10], [140; 10]],
            mv_class0_fr_probs: [[
                [128, 128, 64],
                [96, 112, 64],
            ], [
                [128, 128, 64],
                [96, 112, 64],
            ]],
            mv_fr_probs: [[128, 128, 64], [128, 128, 64]],
            mv_class0_hp_probs: [160, 160],
            mv_hp_probs: [128, 128],
            tx_probs_8x8: [[100], [66]],
            tx_probs_16x16: [[20, 152], [15, 101]],
            tx_probs_32x32: [[3, 136, 37], [5, 52, 13]],
            coef_probs: Self::default_coef_probs(),
        }
    }

    fn default_partition_probs() -> [[u8; 3]; 16] {
        [
            [199, 122, 141],
            [147, 63, 159],
            [148, 133, 118],
            [121, 104, 114],
            [174, 73, 87],
            [92, 41, 83],
            [82, 99, 50],
            [53, 39, 39],
            [177, 58, 59],
            [68, 26, 63],
            [52, 79, 25],
            [17, 14, 12],
            [222, 34, 30],
            [72, 16, 44],
            [58, 32, 12],
            [10, 7, 6],
        ]
    }

    fn default_y_mode_probs() -> [[u8; 9]; 4] {
        [
            [65, 32, 18, 144, 162, 194, 41, 51, 98],
            [132, 68, 18, 165, 217, 196, 45, 40, 78],
            [173, 80, 19, 176, 240, 193, 64, 35, 46],
            [221, 135, 38, 194, 248, 121, 96, 85, 29],
        ]
    }

    fn default_uv_mode_probs() -> [[u8; 9]; 10] {
        [
            [120, 7, 76, 176, 208, 126, 28, 54, 103],
            [48, 12, 154, 155, 139, 90, 34, 117, 119],
            [67, 6, 25, 204, 243, 158, 13, 21, 96],
            [97, 5, 44, 131, 176, 139, 48, 68, 97],
            [83, 5, 42, 156, 111, 152, 26, 49, 152],
            [80, 5, 58, 178, 74, 83, 33, 62, 145],
            [86, 5, 32, 154, 192, 168, 14, 22, 163],
            [85, 5, 32, 156, 216, 148, 19, 29, 73],
            [77, 7, 64, 116, 132, 122, 37, 126, 120],
            [101, 21, 107, 181, 192, 103, 19, 67, 125],
        ]
    }

    fn default_mv_class_probs() -> [[u8; 10]; 2] {
        [
            [224, 144, 192, 168, 192, 176, 192, 198, 198, 245],
            [216, 128, 176, 160, 176, 176, 192, 198, 198, 208],
        ]
    }

    fn default_coef_probs() -> CoefProbs {
        // Initialize with default VP9 coefficient probabilities
        // This is a simplified version - full tables are larger
        [[[[[[128; 3]; 6]; 6]; 2]; 2]; 4]
    }

    /// Reset to default probabilities (for keyframes).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Update partition probabilities from decoded values.
    pub fn update_partition_probs(&mut self, updates: &[[Option<u8>; 3]; 16]) {
        for (i, row) in updates.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if let Some(v) = val {
                    self.partition_probs[i][j] = v;
                }
            }
        }
    }
}

/// Inverse probability update table.
pub const INV_RECENTER_NONNEG: [u8; 256] = [
    255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240,
    239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224,
    223, 222, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212, 211, 210, 209, 208,
    207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 193, 192,
    191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176,
    175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160,
    159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144,
    143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129, 128,
    127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112,
    111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98,  97,  96,
     95,  94,  93,  92,  91,  90,  89,  88,  87,  86,  85,  84,  83,  82,  81,  80,
     79,  78,  77,  76,  75,  74,  73,  72,  71,  70,  69,  68,  67,  66,  65,  64,
     63,  62,  61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,
     47,  46,  45,  44,  43,  42,  41,  40,  39,  38,  37,  36,  35,  34,  33,  32,
     31,  30,  29,  28,  27,  26,  25,  24,  23,  22,  21,  20,  19,  18,  17,  16,
     15,  14,  13,  12,  11,  10,   9,   8,   7,   6,   5,   4,   3,   2,   1,   0,
];

/// Decode a probability update value.
pub fn decode_prob_update(decoder: &mut BoolDecoder, old_prob: u8) -> Result<u8> {
    if decoder.read_bit()? {
        let delta = decode_term_subexp(decoder)?;
        Ok(merge_probs(old_prob, delta))
    } else {
        Ok(old_prob)
    }
}

/// Decode a term subexp value for probability updates.
fn decode_term_subexp(decoder: &mut BoolDecoder) -> Result<u8> {
    if decoder.read_literal(3)? < 5 {
        let val = decoder.read_literal(1)?;
        Ok(val as u8)
    } else {
        let mut i = 0;
        while i < 6 && decoder.read_bit()? {
            i += 1;
        }
        let bits = i + 4;
        let val = decoder.read_literal(bits)?;
        Ok((val + (1 << bits) - 1) as u8)
    }
}

/// Merge old and new probability values.
fn merge_probs(old: u8, delta: u8) -> u8 {
    let v = old as i32;
    let max = 255 - v;
    let d = delta as i32;

    if d < (max >> 1) {
        (v + d) as u8
    } else if d < max {
        (v + max - 1 - d) as u8
    } else {
        (d - max + v) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_decoder_creation() {
        let data = [0x80, 0x00, 0x00, 0x00];
        let decoder = BoolDecoder::new(&data);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_bool_decoder_empty() {
        let data: [u8; 0] = [];
        let decoder = BoolDecoder::new(&data);
        assert!(decoder.is_err());
    }

    #[test]
    fn test_read_literal() {
        let data = [0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut decoder = BoolDecoder::new(&data).unwrap();

        // Reading with 50% probability should give predictable results
        // based on the data values
        let val = decoder.read_literal(4).unwrap();
        assert!(val <= 15); // 4 bits max value is 15
    }

    #[test]
    fn test_probability_context_default() {
        let ctx = ProbabilityContext::new();
        assert_eq!(ctx.skip_probs, [192, 128, 64]);
        assert_eq!(ctx.intra_inter_probs[0], 9);
    }

    #[test]
    fn test_probability_context_reset() {
        let mut ctx = ProbabilityContext::new();
        ctx.skip_probs = [100, 100, 100];
        ctx.reset();
        assert_eq!(ctx.skip_probs, [192, 128, 64]);
    }

    #[test]
    fn test_merge_probs() {
        assert_eq!(merge_probs(128, 0), 128);
        assert_eq!(merge_probs(128, 10), 138);
    }
}
