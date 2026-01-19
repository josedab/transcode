//! CAVLC (Context-Adaptive Variable-Length Coding) entropy coding.
//!
//! CAVLC is used in H.264 Baseline profile for entropy coding of residual data.

use transcode_core::bitstream::{BitReader, BitWriter};
use transcode_core::error::Result;

/// CAVLC decoder for residual coefficient decoding.
pub struct CavlcDecoder {
    /// Number of non-zero coefficients context.
    nc_cache: [[i8; 4]; 4],
}

impl CavlcDecoder {
    /// Create a new CAVLC decoder.
    pub fn new() -> Self {
        Self {
            nc_cache: [[0; 4]; 4],
        }
    }

    /// Decode a 4x4 residual block.
    pub fn decode_residual_block_4x4(
        &mut self,
        reader: &mut BitReader<'_>,
        _block_type: BlockType,
        max_coeffs: usize,
    ) -> Result<[i16; 16]> {
        let mut coeffs = [0i16; 16];

        // Get nC (predicted number of coefficients)
        let nc = self.get_nc(0, 0); // Simplified - would use actual position

        // Decode coeff_token
        let (total_coeff, trailing_ones) = self.decode_coeff_token(reader, nc)?;

        if total_coeff == 0 {
            return Ok(coeffs);
        }

        // Decode trailing ones signs
        let mut level = [0i16; 16];
        for i in 0..trailing_ones {
            let sign = reader.read_bit()?;
            level[i as usize] = if sign { -1 } else { 1 };
        }

        // Decode remaining levels
        let mut suffix_length = if total_coeff > 10 && trailing_ones < 3 { 1 } else { 0 };

        for i in trailing_ones..total_coeff {
            let level_val = self.decode_level(reader, suffix_length)?;
            level[i as usize] = level_val;

            if suffix_length == 0 {
                suffix_length = 1;
            }
            if level_val.abs() > (3 << (suffix_length - 1)) && suffix_length < 6 {
                suffix_length += 1;
            }
        }

        // Decode total_zeros
        let total_zeros = if total_coeff < max_coeffs as u8 {
            self.decode_total_zeros(reader, total_coeff, max_coeffs)?
        } else {
            0
        };

        // Decode run_before
        let mut run = [0u8; 16];
        let mut zeros_left = total_zeros;

        for i in 0..(total_coeff - 1) {
            if zeros_left > 0 {
                run[i as usize] = self.decode_run_before(reader, zeros_left)?;
                zeros_left -= run[i as usize];
            }
        }
        run[(total_coeff - 1) as usize] = zeros_left;

        // Place coefficients in zig-zag order
        let mut coeff_idx = 0usize;
        for i in (0..total_coeff).rev() {
            coeff_idx += run[i as usize] as usize;
            coeffs[coeff_idx] = level[i as usize];
            coeff_idx += 1;
        }

        Ok(coeffs)
    }

    /// Get predicted number of coefficients.
    fn get_nc(&self, _x: usize, _y: usize) -> i8 {
        // Simplified - would use actual left/top neighbors
        0
    }

    /// Decode coeff_token.
    fn decode_coeff_token(&self, reader: &mut BitReader<'_>, nc: i8) -> Result<(u8, u8)> {
        // Simplified VLC table lookup
        // Real implementation would use full VLC tables

        let _table_idx = if nc < 2 {
            0
        } else if nc < 4 {
            1
        } else if nc < 8 {
            2
        } else {
            3
        };

        // Read up to 16 bits for coeff_token
        let bits = reader.peek_bits(16)?;

        // Simplified: just return zeros for now
        // Real implementation needs full VLC tables
        if bits & 0x8000 != 0 {
            reader.skip(1)?;
            Ok((0, 0)) // total_coeff=0, trailing_ones=0
        } else {
            // Placeholder
            reader.skip(1)?;
            Ok((0, 0))
        }
    }

    /// Decode a level value.
    fn decode_level(&self, reader: &mut BitReader<'_>, suffix_length: u8) -> Result<i16> {
        // Decode level_prefix
        let mut prefix = 0u32;
        while !reader.read_bit()? {
            prefix += 1;
            if prefix > 15 {
                break;
            }
        }

        use std::cmp::Ordering;
        let level_code = match prefix.cmp(&14) {
            Ordering::Less => {
                if suffix_length > 0 {
                    let suffix = reader.read_bits(suffix_length)?;
                    (prefix << suffix_length) + suffix
                } else {
                    prefix
                }
            }
            Ordering::Equal => {
                let suffix = reader.read_bits(suffix_length.max(4))?;
                (prefix << suffix_length) + suffix
            }
            Ordering::Greater => {
                let suffix = reader.read_bits(12)?;
                (prefix << 12) + suffix
            }
        };

        let level = if level_code % 2 == 0 {
            (level_code / 2 + 1) as i16
        } else {
            -((level_code / 2 + 1) as i16)
        };

        Ok(level)
    }

    /// Decode total_zeros.
    fn decode_total_zeros(&self, _reader: &mut BitReader<'_>, _total_coeff: u8, _max_coeffs: usize) -> Result<u8> {
        // Simplified - would use VLC table
        Ok(0)
    }

    /// Decode run_before.
    fn decode_run_before(&self, _reader: &mut BitReader<'_>, _zeros_left: u8) -> Result<u8> {
        // Simplified - would use VLC table
        Ok(0)
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) {
        self.nc_cache = [[0; 4]; 4];
    }
}

impl Default for CavlcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Block type for residual decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    /// Luma DC (for 16x16 intra prediction).
    LumaDc,
    /// Luma AC (for 16x16 intra, excludes DC coefficient).
    LumaAc,
    /// Luma 4x4 (for I4x4/P/B macroblocks, includes all coefficients).
    Luma4x4,
    /// Chroma DC.
    ChromaDc,
    /// Chroma AC.
    ChromaAc,
}

/// CAVLC encoder for residual coefficient encoding.
pub struct CavlcEncoder {
    /// Number of non-zero coefficients context.
    nc_cache: [[i8; 4]; 4],
}

impl CavlcEncoder {
    /// Create a new CAVLC encoder.
    pub fn new() -> Self {
        Self {
            nc_cache: [[0; 4]; 4],
        }
    }

    /// Encode a 4x4 residual block.
    ///
    /// Returns the encoded data and updates the nC cache.
    pub fn encode_residual_block_4x4(
        &mut self,
        writer: &mut BitWriter,
        coeffs: &[i16; 16],
        block_type: BlockType,
        block_x: usize,
        block_y: usize,
    ) {
        // Count non-zero coefficients and trailing ones
        let (total_coeff, trailing_ones, levels, runs) = self.analyze_block(coeffs);

        // Get predicted nC for VLC table selection
        let nc = self.get_nc(block_x, block_y);

        // Encode coeff_token
        self.encode_coeff_token(writer, total_coeff, trailing_ones, nc);

        if total_coeff == 0 {
            // Update nC cache
            self.set_nc(block_x, block_y, 0);
            return;
        }

        // Encode trailing ones signs (in reverse order)
        for i in 0..trailing_ones {
            let level_idx = (total_coeff - 1 - i) as usize;
            writer.write_bit(levels[level_idx] < 0);
        }

        // Encode remaining levels
        let mut suffix_length = if total_coeff > 10 && trailing_ones < 3 { 1 } else { 0 };

        for i in trailing_ones..total_coeff {
            let level_idx = (total_coeff - 1 - i) as usize;
            let level = levels[level_idx];

            // Adjust level for first coefficient after trailing ones
            let adjusted_level = if i == trailing_ones && trailing_ones < 3 {
                if level > 0 { level - 1 } else { level + 1 }
            } else {
                level
            };

            self.encode_level(writer, adjusted_level, suffix_length);

            // Update suffix length
            if suffix_length == 0 {
                suffix_length = 1;
            }
            let threshold = 3 << (suffix_length - 1);
            if level.abs() > threshold && suffix_length < 6 {
                suffix_length += 1;
            }
        }

        // Encode total_zeros if there are zeros
        let max_coeffs = self.max_coeffs_for_block_type(block_type);
        if total_coeff < max_coeffs as u8 {
            let total_zeros: u8 = runs.iter().take(total_coeff as usize).sum();
            self.encode_total_zeros(writer, total_zeros, total_coeff, max_coeffs);
        }

        // Encode run_before
        let mut zeros_left: u8 = runs.iter().take(total_coeff as usize).sum();
        for i in 0..(total_coeff - 1) {
            if zeros_left == 0 {
                break;
            }
            let run = runs[(total_coeff - 1 - i) as usize];
            self.encode_run_before(writer, run, zeros_left);
            zeros_left -= run;
        }

        // Update nC cache
        self.set_nc(block_x, block_y, total_coeff as i8);
    }

    /// Analyze a block to extract coefficients and runs.
    fn analyze_block(&self, coeffs: &[i16; 16]) -> (u8, u8, [i16; 16], [u8; 16]) {
        let mut levels = [0i16; 16];
        let mut runs = [0u8; 16];
        let mut trailing_ones = 0u8;

        // Scan in reverse zig-zag order to find coefficients
        let mut coeff_idx = 0usize;
        let mut last_nonzero = 0usize;

        // Find all non-zero coefficients
        for i in 0..16 {
            if coeffs[i] != 0 {
                last_nonzero = i;
            }
        }

        // Extract levels and runs
        let mut run = 0u8;
        for i in 0..=last_nonzero {
            if coeffs[i] != 0 {
                if coeff_idx > 0 {
                    runs[coeff_idx - 1] = run;
                }
                levels[coeff_idx] = coeffs[i];
                coeff_idx += 1;
                run = 0;
            } else {
                run += 1;
            }
        }

        let total_coeff = coeff_idx as u8;
        if total_coeff > 0 {
            runs[coeff_idx - 1] = 0; // Last coefficient has no run after it
        }

        // Count trailing ones (up to 3, with value +1 or -1)
        for i in (0..total_coeff as usize).rev() {
            if levels[i] == 1 || levels[i] == -1 {
                trailing_ones += 1;
                if trailing_ones == 3 {
                    break;
                }
            } else {
                break;
            }
        }

        (total_coeff, trailing_ones, levels, runs)
    }

    /// Encode coeff_token using VLC tables.
    fn encode_coeff_token(&self, writer: &mut BitWriter, total_coeff: u8, trailing_ones: u8, nc: i8) {
        // Select VLC table based on nC
        let table_idx = if nc < 2 {
            0
        } else if nc < 4 {
            1
        } else if nc < 8 {
            2
        } else {
            3
        };

        // Get VLC code from table
        let (code, length) = get_coeff_token_vlc(table_idx, total_coeff, trailing_ones);
        writer.write_bits(code, length);
    }

    /// Encode a level value.
    fn encode_level(&self, writer: &mut BitWriter, level: i16, suffix_length: u8) {
        // Convert level to level_code
        let level_code = if level > 0 {
            (level as u32 - 1) * 2
        } else {
            (-level as u32 - 1) * 2 + 1
        };

        // Calculate prefix and suffix
        let (prefix, suffix, suffix_bits) = self.compute_level_prefix_suffix(level_code, suffix_length);

        // Write prefix (unary coded: prefix zeros followed by one)
        for _ in 0..prefix {
            writer.write_bit(false);
        }
        writer.write_bit(true);

        // Write suffix if present
        if suffix_bits > 0 {
            writer.write_bits(suffix, suffix_bits);
        }
    }

    /// Compute prefix and suffix for level encoding.
    fn compute_level_prefix_suffix(&self, level_code: u32, suffix_length: u8) -> (u32, u32, u8) {
        use std::cmp::Ordering;
        if suffix_length == 0 {
            // No suffix for suffix_length 0
            let prefix = level_code;
            match prefix.cmp(&14) {
                Ordering::Less => (prefix, 0, 0),
                Ordering::Equal => {
                    let suffix = level_code - 14;
                    (14, suffix, 4)
                }
                Ordering::Greater => {
                    let suffix = level_code - 15;
                    (15, suffix, 12)
                }
            }
        } else {
            let level_suffix_size = suffix_length as u32;
            let prefix = level_code >> level_suffix_size;
            let suffix = level_code & ((1 << level_suffix_size) - 1);

            match prefix.cmp(&14) {
                Ordering::Less => (prefix, suffix, suffix_length),
                Ordering::Equal => (14, suffix, suffix_length.max(4)),
                Ordering::Greater => {
                    let suffix = level_code - (15 << suffix_length);
                    (15, suffix, 12)
                }
            }
        }
    }

    /// Encode total_zeros.
    fn encode_total_zeros(&self, writer: &mut BitWriter, total_zeros: u8, total_coeff: u8, max_coeffs: usize) {
        let (code, length) = if max_coeffs == 4 {
            // Chroma DC
            get_total_zeros_chroma_dc_vlc(total_zeros, total_coeff)
        } else {
            // Luma
            get_total_zeros_vlc(total_zeros, total_coeff)
        };
        writer.write_bits(code, length);
    }

    /// Encode run_before.
    fn encode_run_before(&self, writer: &mut BitWriter, run_before: u8, zeros_left: u8) {
        let (code, length) = get_run_before_vlc(run_before, zeros_left);
        writer.write_bits(code, length);
    }

    /// Get predicted number of coefficients from neighbors.
    fn get_nc(&self, x: usize, y: usize) -> i8 {
        let x = x.min(3);
        let y = y.min(3);

        // Average of left and top neighbors
        let left = if x > 0 { self.nc_cache[y][x - 1] } else { 0 };
        let top = if y > 0 { self.nc_cache[y - 1][x] } else { 0 };

        if x == 0 && y == 0 {
            0
        } else if x == 0 {
            top
        } else if y == 0 {
            left
        } else {
            (left + top + 1) / 2
        }
    }

    /// Set nC value in cache.
    fn set_nc(&mut self, x: usize, y: usize, nc: i8) {
        let x = x.min(3);
        let y = y.min(3);
        self.nc_cache[y][x] = nc;
    }

    /// Get maximum coefficients for a block type.
    fn max_coeffs_for_block_type(&self, block_type: BlockType) -> usize {
        match block_type {
            BlockType::LumaDc => 16,
            BlockType::LumaAc => 15,
            BlockType::Luma4x4 => 16,
            BlockType::ChromaDc => 4,
            BlockType::ChromaAc => 15,
        }
    }

    /// Reset the encoder state for a new macroblock row.
    pub fn reset_row(&mut self) {
        // Reset only left column when starting new row
        for row in &mut self.nc_cache {
            row[0] = 0;
        }
    }

    /// Reset the encoder state completely.
    pub fn reset(&mut self) {
        self.nc_cache = [[0; 4]; 4];
    }
}

impl Default for CavlcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// VLC Tables
// =============================================================================

/// Get coeff_token VLC code.
/// Returns (code, length).
fn get_coeff_token_vlc(table_idx: usize, total_coeff: u8, trailing_ones: u8) -> (u32, u8) {
    // Simplified VLC table - full implementation would have complete tables
    // This covers the most common cases
    match table_idx {
        0 => {
            // nC < 2
            match (total_coeff, trailing_ones) {
                (0, 0) => (1, 1),      // 1
                (1, 0) => (5, 6),      // 000101
                (1, 1) => (1, 2),      // 01
                (2, 0) => (7, 8),      // 00000111
                (2, 1) => (4, 6),      // 000100
                (2, 2) => (1, 3),      // 001
                (3, 0) => (7, 9),      // 000000111
                (3, 1) => (6, 8),      // 00000110
                (3, 2) => (5, 7),      // 0000101
                (3, 3) => (3, 5),      // 00011
                (4, 0) => (7, 10),     // 0000000111
                (4, 1) => (6, 9),      // 000000110
                (4, 2) => (5, 8),      // 00000101
                (4, 3) => (3, 6),      // 000011
                _ => (1, 1),           // Default to 0,0
            }
        }
        1 => {
            // 2 <= nC < 4
            match (total_coeff, trailing_ones) {
                (0, 0) => (3, 2),
                (1, 0) => (11, 6),
                (1, 1) => (2, 2),
                (2, 0) => (7, 6),
                (2, 1) => (7, 5),
                (2, 2) => (3, 3),
                (3, 0) => (7, 7),
                (3, 1) => (10, 6),
                (3, 2) => (9, 6),
                (3, 3) => (5, 4),
                _ => (3, 2),
            }
        }
        2 => {
            // 4 <= nC < 8
            match (total_coeff, trailing_ones) {
                (0, 0) => (15, 4),
                (1, 0) => (15, 6),
                (1, 1) => (14, 4),
                (2, 0) => (11, 6),
                (2, 1) => (15, 5),
                (2, 2) => (13, 4),
                (3, 0) => (8, 6),
                (3, 1) => (12, 5),
                (3, 2) => (14, 5),
                (3, 3) => (12, 4),
                _ => (15, 4),
            }
        }
        _ => {
            // nC >= 8 - use fixed length code
            let code = ((total_coeff as u32) << 2) | (trailing_ones as u32);
            (code, 6)
        }
    }
}

/// Get total_zeros VLC code for luma.
fn get_total_zeros_vlc(total_zeros: u8, total_coeff: u8) -> (u32, u8) {
    // Simplified table - covers common cases
    match total_coeff {
        1 => match total_zeros {
            0 => (1, 1),
            1 => (3, 3),
            2 => (2, 3),
            3 => (3, 4),
            4 => (2, 4),
            5 => (3, 5),
            6 => (2, 5),
            7 => (3, 6),
            8 => (2, 6),
            9 => (3, 7),
            10 => (2, 7),
            11 => (3, 8),
            12 => (2, 8),
            13 => (3, 9),
            14 => (2, 9),
            15 => (1, 9),
            _ => (1, 1),
        },
        2 => match total_zeros {
            0 => (7, 3),
            1 => (6, 3),
            2 => (5, 3),
            3 => (4, 3),
            4 => (3, 3),
            5 => (5, 4),
            6 => (4, 4),
            7 => (3, 4),
            8 => (2, 4),
            9 => (3, 5),
            10 => (2, 5),
            11 => (3, 6),
            12 => (2, 6),
            13 => (1, 6),
            14 => (0, 6),
            _ => (7, 3),
        },
        _ => {
            // Simplified for higher total_coeff
            if total_zeros == 0 {
                (1, 1)
            } else {
                (0, (total_zeros + 1).min(9))
            }
        }
    }
}

/// Get total_zeros VLC code for chroma DC.
fn get_total_zeros_chroma_dc_vlc(total_zeros: u8, total_coeff: u8) -> (u32, u8) {
    match total_coeff {
        1 => match total_zeros {
            0 => (1, 1),
            1 => (1, 2),
            2 => (1, 3),
            3 => (0, 3),
            _ => (1, 1),
        },
        2 => match total_zeros {
            0 => (1, 1),
            1 => (1, 2),
            2 => (0, 2),
            _ => (1, 1),
        },
        3 => match total_zeros {
            0 => (1, 1),
            1 => (0, 1),
            _ => (1, 1),
        },
        _ => (1, 1),
    }
}

/// Get run_before VLC code.
fn get_run_before_vlc(run_before: u8, zeros_left: u8) -> (u32, u8) {
    if zeros_left <= 1 {
        if run_before == 0 { (1, 1) } else { (0, 1) }
    } else if zeros_left == 2 {
        match run_before {
            0 => (1, 1),
            1 => (1, 2),
            2 => (0, 2),
            _ => (1, 1),
        }
    } else if zeros_left == 3 {
        match run_before {
            0 => (3, 2),
            1 => (2, 2),
            2 => (1, 2),
            3 => (0, 2),
            _ => (3, 2),
        }
    } else if zeros_left == 4 {
        match run_before {
            0 => (3, 2),
            1 => (2, 2),
            2 => (1, 2),
            3 => (1, 3),
            4 => (0, 3),
            _ => (3, 2),
        }
    } else if zeros_left == 5 {
        match run_before {
            0 => (3, 2),
            1 => (2, 2),
            2 => (3, 3),
            3 => (2, 3),
            4 => (1, 3),
            5 => (0, 3),
            _ => (3, 2),
        }
    } else if zeros_left == 6 {
        match run_before {
            0 => (3, 2),
            1 => (0, 3),
            2 => (1, 3),
            3 => (3, 3),
            4 => (2, 3),
            5 => (5, 3),
            6 => (4, 3),
            _ => (3, 2),
        }
    } else {
        // zeros_left >= 7
        match run_before {
            0 => (7, 3),
            1 => (6, 3),
            2 => (5, 3),
            3 => (4, 3),
            4 => (3, 3),
            5 => (2, 3),
            6 => (1, 3),
            _ => {
                // Extended
                let prefix = run_before - 7;
                if prefix < 8 {
                    ((1 << prefix) - 1, prefix + 4)
                } else {
                    (0xFF, 11)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cavlc_encoder_zero_block() {
        let mut encoder = CavlcEncoder::new();
        let mut writer = BitWriter::new();
        let coeffs = [0i16; 16];

        encoder.encode_residual_block_4x4(&mut writer, &coeffs, BlockType::LumaAc, 0, 0);

        // Should encode coeff_token for 0 coefficients
        assert!(!writer.data().is_empty());
    }

    #[test]
    fn test_cavlc_encoder_single_coeff() {
        let mut encoder = CavlcEncoder::new();
        let mut writer = BitWriter::new();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 1;

        encoder.encode_residual_block_4x4(&mut writer, &coeffs, BlockType::LumaAc, 0, 0);

        assert!(!writer.data().is_empty());
    }

    #[test]
    fn test_cavlc_encoder_trailing_ones() {
        let mut encoder = CavlcEncoder::new();
        let mut writer = BitWriter::new();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 1;
        coeffs[1] = -1;
        coeffs[2] = 1;

        encoder.encode_residual_block_4x4(&mut writer, &coeffs, BlockType::LumaAc, 0, 0);

        assert!(!writer.data().is_empty());
    }

    #[test]
    fn test_cavlc_encoder_mixed_levels() {
        let mut encoder = CavlcEncoder::new();
        let mut writer = BitWriter::new();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 5;
        coeffs[1] = -3;
        coeffs[2] = 1;
        coeffs[3] = -1;

        encoder.encode_residual_block_4x4(&mut writer, &coeffs, BlockType::LumaAc, 0, 0);

        assert!(!writer.data().is_empty());
    }

    #[test]
    fn test_cavlc_nc_prediction() {
        let mut encoder = CavlcEncoder::new();

        // First block has no neighbors
        assert_eq!(encoder.get_nc(0, 0), 0);

        // Set some values in top row
        encoder.set_nc(0, 0, 5);
        encoder.set_nc(1, 0, 3);

        // Check prediction for second row - uses top values
        // (0, 1) uses top which is (0, 0) = 5
        assert_eq!(encoder.get_nc(0, 1), 5);

        // (1, 0) uses left which is (0, 0) = 5
        assert_eq!(encoder.get_nc(1, 0), 5);

        // For (1, 1), we need to set (0, 1) first to properly test averaging
        encoder.set_nc(0, 1, 4);
        // Now (1, 1) should average left (0, 1)=4 and top (1, 0)=3 -> (4+3+1)/2 = 4
        assert_eq!(encoder.get_nc(1, 1), 4);
    }

    #[test]
    fn test_analyze_block() {
        let encoder = CavlcEncoder::new();
        let mut coeffs = [0i16; 16];
        coeffs[0] = 3;
        coeffs[1] = 0;
        coeffs[2] = -1;
        coeffs[3] = 1;

        let (total_coeff, trailing_ones, levels, runs) = encoder.analyze_block(&coeffs);

        assert_eq!(total_coeff, 3);
        assert_eq!(trailing_ones, 2); // -1 and 1 are trailing ones
        assert_eq!(levels[0], 3);
        assert_eq!(levels[1], -1);
        assert_eq!(levels[2], 1);
        // runs[0] = zeros before second coeff (-1), which is 1
        // runs[1] = zeros before third coeff (1), which is 0
        // runs[2] = 0 (last coeff)
        assert_eq!(runs[0], 1); // One zero before -1
        assert_eq!(runs[1], 0); // No zeros before 1
    }
}
