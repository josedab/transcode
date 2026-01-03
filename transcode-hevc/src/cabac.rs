//! HEVC CABAC (Context-Adaptive Binary Arithmetic Coding) implementation.
//!
//! This module provides CABAC entropy coding for HEVC, including context modeling,
//! binary arithmetic decoding/encoding, and binarization methods.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(unused_variables)]

use crate::error::{HevcError, Result};
use crate::nal::{SliceSegmentHeader, SliceType};

/// CABAC state for a single context.
#[derive(Debug, Clone, Copy)]
pub struct CabacContext {
    /// State index (0-63).
    state: u8,
    /// Most probable symbol.
    mps: bool,
}

impl CabacContext {
    /// Create a new context with initial state.
    pub fn new(init_value: u8) -> Self {
        let slope_idx = (init_value >> 4) as i32;
        let offset_idx = (init_value & 0x0F) as i32;

        // Simplified initialization
        let pre_ctx_state = ((slope_idx * 5) - 45 + (offset_idx << 3) - 16).clamp(-126, 126);

        if pre_ctx_state <= 0 {
            Self {
                state: ((-pre_ctx_state + 1) >> 1).clamp(0, 63) as u8,
                mps: false,
            }
        } else {
            Self {
                state: ((pre_ctx_state - 1) >> 1).clamp(0, 63) as u8,
                mps: true,
            }
        }
    }

    /// Update context after coding a symbol.
    pub fn update(&mut self, symbol: bool) {
        if symbol == self.mps {
            // LPS->MPS transition probability update
            self.state = NEXT_STATE_MPS[self.state as usize];
        } else {
            // MPS->LPS transition probability update
            if self.state == 0 {
                self.mps = !self.mps;
            }
            self.state = NEXT_STATE_LPS[self.state as usize];
        }
    }

    /// Get the current state.
    pub fn state(&self) -> u8 {
        self.state
    }

    /// Get the MPS value.
    pub fn mps(&self) -> bool {
        self.mps
    }
}

// State transition tables for CABAC
const NEXT_STATE_MPS: [u8; 64] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];

const NEXT_STATE_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

// Range table for LPS (indexed by state and qRangeIdx)
const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216], [123, 150, 178, 205],
    [116, 142, 169, 195], [111, 135, 160, 185], [105, 128, 152, 175], [100, 122, 144, 166],
    [95, 116, 137, 158], [90, 110, 130, 150], [85, 104, 123, 142], [81, 99, 117, 135],
    [77, 94, 111, 128], [73, 89, 105, 122], [69, 85, 100, 116], [66, 80, 95, 110],
    [62, 76, 90, 104], [59, 72, 86, 99], [56, 69, 81, 94], [53, 65, 77, 89],
    [51, 62, 73, 85], [48, 59, 69, 80], [46, 56, 66, 76], [43, 53, 63, 72],
    [41, 50, 59, 69], [39, 48, 56, 65], [37, 45, 54, 62], [35, 43, 51, 59],
    [33, 41, 48, 56], [32, 39, 46, 53], [30, 37, 43, 50], [29, 35, 41, 48],
    [27, 33, 39, 45], [26, 31, 37, 43], [24, 30, 35, 41], [23, 28, 33, 39],
    [22, 27, 32, 37], [21, 26, 30, 35], [20, 24, 29, 33], [19, 23, 27, 31],
    [18, 22, 26, 30], [17, 21, 25, 28], [16, 20, 23, 27], [15, 19, 22, 25],
    [14, 18, 21, 24], [14, 17, 20, 23], [13, 16, 19, 22], [12, 15, 18, 21],
    [12, 14, 17, 20], [11, 14, 16, 19], [11, 13, 15, 18], [10, 12, 15, 17],
    [10, 12, 14, 16], [9, 11, 13, 15], [9, 11, 12, 14], [8, 10, 12, 14],
    [8, 9, 11, 13], [7, 9, 11, 12], [7, 9, 10, 12], [7, 8, 10, 11],
    [6, 8, 9, 11], [6, 7, 9, 10], [6, 7, 8, 9], [2, 2, 2, 2],
];

/// CABAC decoder.
#[derive(Debug)]
pub struct CabacDecoder<'a> {
    /// Input data.
    data: &'a [u8],
    /// Current byte position.
    byte_pos: usize,
    /// Bits left in current byte.
    bits_left: u8,
    /// Arithmetic coding range.
    range: u32,
    /// Arithmetic coding offset.
    offset: u32,
    /// Context models.
    contexts: Vec<CabacContext>,
}

impl<'a> CabacDecoder<'a> {
    /// Create a new CABAC decoder.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bits_left: 0,
            range: 510,
            offset: 0,
            contexts: Vec::new(),
        }
    }

    /// Initialize the decoder.
    pub fn init(&mut self) -> Result<()> {
        if self.data.len() < 2 {
            return Err(HevcError::Cabac("Not enough data for initialization".to_string()));
        }

        // Read initial bits
        self.offset = ((self.data[0] as u32) << 8) | (self.data[1] as u32);
        self.byte_pos = 2;
        self.bits_left = 0;
        self.range = 510;

        Ok(())
    }

    /// Initialize contexts for a slice.
    pub fn init_contexts(&mut self, slice_header: &SliceSegmentHeader, init_type: u8) {
        // Initialize contexts based on slice type and cabac_init_flag
        let _init_value = match slice_header.slice_type {
            SliceType::I => 0,
            SliceType::P => if slice_header.cabac_init_flag { 2 } else { 1 },
            SliceType::B => if slice_header.cabac_init_flag { 1 } else { 2 },
        };

        // Initialize all HEVC contexts
        self.contexts = HEVC_CONTEXT_INIT_VALUES[init_type as usize]
            .iter()
            .map(|&v| CabacContext::new(v))
            .collect();
    }

    /// Read a bit from the input.
    fn read_bit(&mut self) -> Result<bool> {
        if self.bits_left == 0 {
            if self.byte_pos >= self.data.len() {
                return Err(HevcError::Cabac("Unexpected end of data".to_string()));
            }
            self.bits_left = 8;
        }
        self.bits_left -= 1;
        let bit = (self.data[self.byte_pos] >> self.bits_left) & 1;
        if self.bits_left == 0 {
            self.byte_pos += 1;
        }
        Ok(bit != 0)
    }

    /// Renormalize the decoder state.
    fn renormalize(&mut self) -> Result<()> {
        while self.range < 256 {
            self.range <<= 1;
            self.offset = (self.offset << 1) | (self.read_bit()? as u32);
        }
        Ok(())
    }

    /// Decode a binary decision using a context.
    pub fn decode_decision(&mut self, ctx_idx: usize) -> Result<bool> {
        let ctx = &self.contexts[ctx_idx];
        let state = ctx.state() as usize;
        let mps = ctx.mps();

        // Calculate LPS range
        let q_range_idx = ((self.range >> 6) & 3) as usize;
        let lps_range = RANGE_TAB_LPS[state][q_range_idx] as u32;

        self.range -= lps_range;

        let symbol = if self.offset >= self.range {
            // LPS
            self.offset -= self.range;
            self.range = lps_range;
            !mps
        } else {
            // MPS
            mps
        };

        // Update context
        self.contexts[ctx_idx].update(symbol);

        // Renormalize
        self.renormalize()?;

        Ok(symbol)
    }

    /// Decode a bypass bin (equiprobable).
    pub fn decode_bypass(&mut self) -> Result<bool> {
        self.offset = (self.offset << 1) | (self.read_bit()? as u32);

        if self.offset >= self.range {
            self.offset -= self.range;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Decode multiple bypass bins.
    pub fn decode_bypass_bins(&mut self, count: u8) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | (self.decode_bypass()? as u32);
        }
        Ok(value)
    }

    /// Decode terminate bin.
    pub fn decode_terminate(&mut self) -> Result<bool> {
        self.range -= 2;

        if self.offset >= self.range {
            Ok(true)
        } else {
            self.renormalize()?;
            Ok(false)
        }
    }

    /// Decode unary value.
    pub fn decode_unary(&mut self, ctx_idx: usize, max_val: u32) -> Result<u32> {
        let mut value = 0u32;
        while value < max_val && self.decode_decision(ctx_idx)? {
            value += 1;
        }
        Ok(value)
    }

    /// Decode truncated unary value.
    pub fn decode_truncated_unary(&mut self, ctx_idx: usize, max_val: u32) -> Result<u32> {
        if max_val == 0 {
            return Ok(0);
        }

        let mut value = 0u32;
        while value < max_val {
            if !self.decode_decision(ctx_idx)? {
                break;
            }
            value += 1;
        }
        Ok(value)
    }

    /// Decode Exp-Golomb coded value.
    pub fn decode_exp_golomb(&mut self, k: u32) -> Result<u32> {
        // Count leading zeros
        let mut leading_zeros = 0u32;
        while !self.decode_bypass()? {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(HevcError::Cabac("Exp-Golomb overflow".to_string()));
            }
        }

        if leading_zeros == 0 {
            return Ok(0);
        }

        let suffix = self.decode_bypass_bins((leading_zeros + k) as u8)?;
        Ok(((1 << leading_zeros) - 1 + suffix) << k)
    }

    /// Decode split CU flag.
    pub fn decode_split_cu_flag(&mut self, ctx_idx: usize) -> Result<bool> {
        self.decode_decision(ctx_idx)
    }

    /// Decode CU skip flag.
    pub fn decode_cu_skip_flag(&mut self, ctx_idx: usize) -> Result<bool> {
        self.decode_decision(ctx_idx)
    }

    /// Decode prediction mode.
    pub fn decode_pred_mode_flag(&mut self) -> Result<bool> {
        self.decode_decision(PRED_MODE_CTX)
    }

    /// Decode part mode.
    pub fn decode_part_mode(&mut self, log2_cb_size: u8, is_intra: bool) -> Result<u8> {
        if is_intra {
            // Intra: 2Nx2N or NxN
            if log2_cb_size > 3 {
                Ok(0) // 2Nx2N only
            } else {
                let part_mode = self.decode_decision(PART_MODE_CTX)?;
                Ok(if part_mode { 0 } else { 1 })
            }
        } else {
            // Inter
            let mut part_mode = 0u8;
            if !self.decode_decision(PART_MODE_CTX)? {
                part_mode = 1;
                if log2_cb_size > 3 {
                    if !self.decode_decision(PART_MODE_CTX + 1)? {
                        part_mode = 2;
                        if !self.decode_decision(PART_MODE_CTX + 2)? {
                            part_mode = 3 + self.decode_bypass()? as u8;
                        }
                    }
                }
            }
            Ok(part_mode)
        }
    }

    /// Decode intra prediction mode.
    pub fn decode_prev_intra_luma_pred_flag(&mut self) -> Result<bool> {
        self.decode_decision(PREV_INTRA_LUMA_PRED_CTX)
    }

    /// Decode MPM index.
    pub fn decode_mpm_idx(&mut self) -> Result<u8> {
        if !self.decode_bypass()? {
            Ok(0)
        } else if !self.decode_bypass()? {
            Ok(1)
        } else {
            Ok(2)
        }
    }

    /// Decode remaining intra mode.
    pub fn decode_rem_intra_luma_pred_mode(&mut self) -> Result<u8> {
        self.decode_bypass_bins(5).map(|v| v as u8)
    }

    /// Decode merge flag.
    pub fn decode_merge_flag(&mut self) -> Result<bool> {
        self.decode_decision(MERGE_FLAG_CTX)
    }

    /// Decode merge index.
    pub fn decode_merge_idx(&mut self, max_num_merge_cand: u8) -> Result<u8> {
        if max_num_merge_cand <= 1 {
            return Ok(0);
        }

        let mut idx = 0u8;
        if self.decode_decision(MERGE_IDX_CTX)? {
            idx += 1;
            while idx < max_num_merge_cand - 1 && self.decode_bypass()? {
                idx += 1;
            }
        }
        Ok(idx)
    }

    /// Decode inter prediction direction.
    pub fn decode_inter_pred_idc(&mut self, ctx_idx: usize) -> Result<u8> {
        if self.decode_decision(ctx_idx)? {
            Ok(2) // Bi-prediction
        } else if self.decode_decision(ctx_idx + 1)? {
            Ok(1) // L1 prediction
        } else {
            Ok(0) // L0 prediction
        }
    }

    /// Decode reference index.
    pub fn decode_ref_idx(&mut self, num_ref_idx: u8) -> Result<u8> {
        if num_ref_idx == 1 {
            return Ok(0);
        }

        let mut idx = 0u8;
        while idx < num_ref_idx - 1 {
            let ctx_idx = if idx < 2 { REF_IDX_CTX + idx as usize } else { REF_IDX_CTX + 2 };
            if !self.decode_decision(ctx_idx)? {
                break;
            }
            idx += 1;
        }
        Ok(idx)
    }

    /// Decode MVD (motion vector difference).
    pub fn decode_mvd(&mut self) -> Result<(i32, i32)> {
        // Decode abs_mvd_greater0_flag
        let abs_mvd_x_gt0 = self.decode_decision(ABS_MVD_GREATER0_CTX)?;
        let abs_mvd_y_gt0 = self.decode_decision(ABS_MVD_GREATER0_CTX)?;

        // Decode abs_mvd_greater1_flag
        let abs_mvd_x_gt1 = if abs_mvd_x_gt0 {
            self.decode_decision(ABS_MVD_GREATER1_CTX)?
        } else {
            false
        };
        let abs_mvd_y_gt1 = if abs_mvd_y_gt0 {
            self.decode_decision(ABS_MVD_GREATER1_CTX)?
        } else {
            false
        };

        // Decode abs_mvd_minus2 if greater1
        let abs_mvd_x = if abs_mvd_x_gt1 {
            2 + self.decode_exp_golomb(1)? as i32
        } else if abs_mvd_x_gt0 {
            1
        } else {
            0
        };

        let abs_mvd_y = if abs_mvd_y_gt1 {
            2 + self.decode_exp_golomb(1)? as i32
        } else if abs_mvd_y_gt0 {
            1
        } else {
            0
        };

        // Decode sign flags
        let mvd_x = if abs_mvd_x > 0 && self.decode_bypass()? {
            -abs_mvd_x
        } else {
            abs_mvd_x
        };

        let mvd_y = if abs_mvd_y > 0 && self.decode_bypass()? {
            -abs_mvd_y
        } else {
            abs_mvd_y
        };

        Ok((mvd_x, mvd_y))
    }

    /// Decode CBF (coded block flag).
    pub fn decode_cbf(&mut self, ctx_idx: usize) -> Result<bool> {
        self.decode_decision(ctx_idx)
    }

    /// Decode transform skip flag.
    pub fn decode_transform_skip_flag(&mut self, ctx_idx: usize) -> Result<bool> {
        self.decode_decision(ctx_idx)
    }

    /// Decode last significant coefficient position.
    pub fn decode_last_sig_coeff(&mut self, log2_size: u8, is_chroma: bool) -> Result<(u32, u32)> {
        let ctx_offset = if is_chroma { 15 } else { 0 };
        let ctx_shift = if is_chroma { 0 } else { (log2_size + 1) >> 2 };

        let max_prefix = ((log2_size << 1) - 1) as u32;

        // Decode X prefix
        let mut last_x = 0u32;
        while last_x < max_prefix {
            let ctx_idx = LAST_SIG_COEFF_X_PREFIX_CTX + ctx_offset + ((last_x >> ctx_shift) as usize);
            if !self.decode_decision(ctx_idx)? {
                break;
            }
            last_x += 1;
        }

        // Decode X suffix
        if last_x > 3 {
            let suffix_length = ((last_x - 2) >> 1) as u8;
            last_x = ((2 + (last_x & 1)) << suffix_length) + self.decode_bypass_bins(suffix_length)?;
        }

        // Decode Y prefix
        let mut last_y = 0u32;
        while last_y < max_prefix {
            let ctx_idx = LAST_SIG_COEFF_Y_PREFIX_CTX + ctx_offset + ((last_y >> ctx_shift) as usize);
            if !self.decode_decision(ctx_idx)? {
                break;
            }
            last_y += 1;
        }

        // Decode Y suffix
        if last_y > 3 {
            let suffix_length = ((last_y - 2) >> 1) as u8;
            last_y = ((2 + (last_y & 1)) << suffix_length) + self.decode_bypass_bins(suffix_length)?;
        }

        Ok((last_x, last_y))
    }

    /// Decode significant coefficient flags for a subblock.
    pub fn decode_sig_coeff_flags(
        &mut self,
        coeffs: &mut [i32],
        log2_size: u8,
        last_x: u32,
        last_y: u32,
        is_luma: bool,
    ) -> Result<()> {
        let size = 1u32 << log2_size;
        let ctx_offset = if is_luma { 0 } else { 27 };

        for y in 0..size {
            for x in 0..size {
                // Skip positions after last significant
                if y > last_y || (y == last_y && x > last_x) {
                    continue;
                }

                // Last position is always significant
                if y == last_y && x == last_x {
                    coeffs[(y * size + x) as usize] = 1;
                    continue;
                }

                // Decode significance
                let ctx_idx = SIG_COEFF_FLAG_CTX + ctx_offset + self.get_sig_ctx(x, y, log2_size);
                if self.decode_decision(ctx_idx)? {
                    coeffs[(y * size + x) as usize] = 1;
                }
            }
        }

        Ok(())
    }

    /// Get context index for significance flag.
    fn get_sig_ctx(&self, x: u32, y: u32, _log2_size: u8) -> usize {
        // Simplified context derivation
        if x == 0 && y == 0 {
            0
        } else if x == 0 || y == 0 {
            1
        } else {
            2
        }
    }

    /// Decode coefficient levels.
    pub fn decode_coeff_abs_level(
        &mut self,
        coeffs: &mut [i32],
        size: usize,
        base_ctx: usize,
    ) -> Result<()> {
        let mut num_greater1 = 0;
        let mut last_greater1_idx = -1i32;

        // First pass: decode greater1 flags
        for (i, coeff) in coeffs.iter_mut().enumerate().take(size) {
            if *coeff != 0 {
                let ctx_idx = base_ctx + std::cmp::min(num_greater1, 3);
                if self.decode_decision(ctx_idx)? {
                    *coeff = 2;
                    num_greater1 += 1;
                    last_greater1_idx = i as i32;
                } else {
                    *coeff = 1;
                }
            }
        }

        // Second pass: decode greater2 flag for last greater1 position
        if last_greater1_idx >= 0 {
            let idx = last_greater1_idx as usize;
            let ctx_idx = base_ctx + 4;
            if self.decode_decision(ctx_idx)? {
                coeffs[idx] = 3;
            }
        }

        // Third pass: decode remaining levels
        for coeff in coeffs.iter_mut().take(size) {
            if *coeff == 3 {
                // Decode remaining level using bypass bins
                *coeff = 3 + self.decode_exp_golomb(0)? as i32;
            }
        }

        // Fourth pass: decode signs
        for coeff in coeffs.iter_mut().take(size) {
            if *coeff != 0 && self.decode_bypass()? {
                *coeff = -*coeff;
            }
        }

        Ok(())
    }

    /// Check if decoder has reached end of slice.
    pub fn is_end_of_slice(&mut self) -> Result<bool> {
        self.decode_terminate()
    }
}

/// CABAC encoder.
#[derive(Debug)]
pub struct CabacEncoder {
    /// Output buffer.
    output: Vec<u8>,
    /// Arithmetic coding low value.
    low: u64,
    /// Arithmetic coding range.
    range: u32,
    /// Bits to follow.
    bits_to_follow: u32,
    /// First bit flag.
    first_bit: bool,
    /// Context models.
    contexts: Vec<CabacContext>,
}

impl CabacEncoder {
    /// Create a new CABAC encoder.
    pub fn new() -> Self {
        Self {
            output: Vec::with_capacity(4096),
            low: 0,
            range: 510,
            bits_to_follow: 0,
            first_bit: true,
            contexts: Vec::new(),
        }
    }

    /// Initialize contexts for a slice.
    pub fn init_contexts(&mut self, slice_type: SliceType, cabac_init_flag: bool) {
        let init_type = match slice_type {
            SliceType::I => 0,
            SliceType::P => if cabac_init_flag { 2 } else { 1 },
            SliceType::B => if cabac_init_flag { 1 } else { 2 },
        };

        self.contexts = HEVC_CONTEXT_INIT_VALUES[init_type]
            .iter()
            .map(|&v| CabacContext::new(v))
            .collect();
    }

    /// Renormalize encoder state.
    fn renormalize(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put_bit_plus_follow(false);
            } else if self.low >= 512 {
                self.put_bit_plus_follow(true);
                self.low -= 512;
            } else {
                self.bits_to_follow += 1;
                self.low -= 256;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    /// Put bit plus following opposite bits.
    fn put_bit_plus_follow(&mut self, bit: bool) {
        if self.first_bit {
            self.first_bit = false;
        } else {
            self.put_bit(bit);
        }

        while self.bits_to_follow > 0 {
            self.put_bit(!bit);
            self.bits_to_follow -= 1;
        }
    }

    /// Put a single bit to output.
    fn put_bit(&mut self, bit: bool) {
        // Implementation would write to output buffer
        // Simplified for now
        if self.output.is_empty() || self.output.len() * 8 % 8 == 0 {
            self.output.push(0);
        }
        let idx = self.output.len() - 1;
        if bit {
            self.output[idx] |= 1 << (7 - (self.output.len() * 8 - 1) % 8);
        }
    }

    /// Encode a binary decision.
    pub fn encode_decision(&mut self, ctx_idx: usize, symbol: bool) -> Result<()> {
        let ctx = &self.contexts[ctx_idx];
        let state = ctx.state() as usize;
        let mps = ctx.mps();

        let q_range_idx = ((self.range >> 6) & 3) as usize;
        let lps_range = RANGE_TAB_LPS[state][q_range_idx] as u32;

        self.range -= lps_range;

        if symbol != mps {
            self.low += self.range as u64;
            self.range = lps_range;
        }

        self.contexts[ctx_idx].update(symbol);
        self.renormalize();

        Ok(())
    }

    /// Encode a bypass bin.
    pub fn encode_bypass(&mut self, symbol: bool) -> Result<()> {
        self.low <<= 1;
        if symbol {
            self.low += self.range as u64;
        }

        if self.low >= 1024 {
            self.put_bit_plus_follow(true);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put_bit_plus_follow(false);
        } else {
            self.bits_to_follow += 1;
            self.low -= 512;
        }

        Ok(())
    }

    /// Encode multiple bypass bins.
    pub fn encode_bypass_bins(&mut self, value: u32, count: u8) -> Result<()> {
        for i in (0..count).rev() {
            self.encode_bypass(((value >> i) & 1) != 0)?;
        }
        Ok(())
    }

    /// Encode terminate bin.
    pub fn encode_terminate(&mut self, symbol: bool) -> Result<()> {
        self.range -= 2;

        if symbol {
            self.low += self.range as u64;
            self.range = 2;
            self.renormalize();
            self.put_bit_plus_follow(((self.low >> 9) & 1) != 0);
            let bits = ((self.low >> 7) & 3) as u8;
            self.output.push(bits << 6);
        } else {
            self.renormalize();
        }

        Ok(())
    }

    /// Get the encoded data.
    pub fn data(&self) -> &[u8] {
        &self.output
    }

    /// Take the encoded data.
    pub fn into_data(self) -> Vec<u8> {
        self.output
    }
}

impl Default for CabacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// Context indices for various syntax elements
const PRED_MODE_CTX: usize = 0;
const PART_MODE_CTX: usize = 1;
const PREV_INTRA_LUMA_PRED_CTX: usize = 4;
const MERGE_FLAG_CTX: usize = 5;
const MERGE_IDX_CTX: usize = 6;
const REF_IDX_CTX: usize = 7;
const ABS_MVD_GREATER0_CTX: usize = 10;
const ABS_MVD_GREATER1_CTX: usize = 11;
const SIG_COEFF_FLAG_CTX: usize = 12;
const LAST_SIG_COEFF_X_PREFIX_CTX: usize = 50;
const LAST_SIG_COEFF_Y_PREFIX_CTX: usize = 70;

// HEVC context initialization values (simplified)
const HEVC_CONTEXT_INIT_VALUES: [[u8; 100]; 3] = [
    // I-slice
    [154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154],
    // P-slice
    [154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154],
    // B-slice
    [154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
     154, 154, 154, 154, 154, 154, 154, 154, 154, 154],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = CabacContext::new(154);
        assert!(ctx.state() <= 63);
    }

    #[test]
    fn test_context_update() {
        let mut ctx = CabacContext::new(154);
        let mps = ctx.mps();

        // Update with MPS
        ctx.update(mps);
        assert!(ctx.state() <= 63);

        // Update with LPS
        ctx.update(!mps);
        assert!(ctx.state() <= 63);
    }

    #[test]
    fn test_cabac_decoder_init() {
        let data = [0x00, 0xFF, 0xAA, 0x55];
        let mut decoder = CabacDecoder::new(&data);
        assert!(decoder.init().is_ok());
    }

    #[test]
    fn test_cabac_encoder_creation() {
        let encoder = CabacEncoder::new();
        assert!(encoder.data().is_empty());
    }

    #[test]
    fn test_state_transition_bounds() {
        for i in 0..64 {
            assert!(NEXT_STATE_MPS[i] <= 63);
            assert!(NEXT_STATE_LPS[i] <= 63);
        }
    }

    #[test]
    fn test_range_tab_bounds() {
        for state in 0..64 {
            for q in 0..4 {
                assert!(RANGE_TAB_LPS[state][q] > 0);
                assert!(RANGE_TAB_LPS[state][q] <= 255);
            }
        }
    }
}
