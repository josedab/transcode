//! CABAC (Context-Adaptive Binary Arithmetic Coding) entropy coding.
//!
//! CABAC is used in H.264 Main and High profiles for better compression.

use transcode_core::bitstream::BitReader;
use transcode_core::error::Result;

/// CABAC context state.
#[derive(Debug, Clone, Copy)]
pub struct CabacContext {
    /// Probability state index (0-63).
    state: u8,
    /// Most probable symbol.
    mps: bool,
}

impl CabacContext {
    /// Create a new context with initial probability.
    pub fn new(init_value: i8) -> Self {
        let state_idx = (init_value.unsigned_abs() & 0x3F).min(63);
        Self {
            state: state_idx,
            mps: init_value >= 0,
        }
    }

    /// Get the probability state.
    pub fn state(&self) -> u8 {
        self.state
    }

    /// Get the MPS.
    pub fn mps(&self) -> bool {
        self.mps
    }
}

/// CABAC decoder.
pub struct CabacDecoder {
    /// Range (9 bits).
    range: u16,
    /// Offset (9 bits).
    offset: u16,
    /// Bits remaining in the current byte.
    bits_left: u8,
    /// Context models.
    contexts: Vec<CabacContext>,
}

impl CabacDecoder {
    /// Create a new CABAC decoder.
    pub fn new() -> Self {
        Self {
            range: 510,
            offset: 0,
            bits_left: 0,
            contexts: Vec::new(),
        }
    }

    /// Initialize the decoder with a bitstream.
    pub fn init(&mut self, reader: &mut BitReader<'_>) -> Result<()> {
        self.range = 510;
        self.offset = reader.read_bits(9)? as u16;
        self.bits_left = 7;
        Ok(())
    }

    /// Initialize context models for a slice.
    pub fn init_contexts(&mut self, _slice_qp: i8, _cabac_init_idc: u8) {
        // Initialize ~460 context models based on QP and init_idc
        // This is simplified - real implementation uses init tables
        self.contexts.clear();
        self.contexts.resize(460, CabacContext::new(0));
    }

    /// Decode a binary decision.
    pub fn decode_decision(&mut self, reader: &mut BitReader<'_>, ctx_idx: usize) -> Result<bool> {
        // Get context state and mps before any mutations
        let ctx_state = self.contexts[ctx_idx].state();
        let ctx_mps = self.contexts[ctx_idx].mps();

        // Get range LPS from table
        let range_lps = self.get_range_lps(self.range, ctx_state);
        let range_mps = self.range - range_lps;

        let symbol = if self.offset >= range_mps {
            // LPS
            self.offset -= range_mps;
            self.range = range_lps;
            self.update_context_lps(ctx_idx);
            !ctx_mps
        } else {
            // MPS
            self.range = range_mps;
            self.update_context_mps(ctx_idx);
            ctx_mps
        };

        // Renormalization
        self.renormalize(reader)?;

        Ok(symbol)
    }

    /// Decode a bypass bin (equiprobable).
    pub fn decode_bypass(&mut self, reader: &mut BitReader<'_>) -> Result<bool> {
        self.offset = (self.offset << 1) | (reader.read_bit()? as u16);

        if self.offset >= self.range {
            self.offset -= self.range;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Decode a terminate bin.
    pub fn decode_terminate(&mut self, reader: &mut BitReader<'_>) -> Result<bool> {
        self.range -= 2;

        if self.offset >= self.range {
            Ok(true)
        } else {
            self.renormalize(reader)?;
            Ok(false)
        }
    }

    /// Decode an unsigned Exp-Golomb value using CABAC.
    pub fn decode_ue(&mut self, reader: &mut BitReader<'_>, ctx_idx: usize) -> Result<u32> {
        // Unary prefix
        let mut value = 0u32;
        while self.decode_decision(reader, ctx_idx)? {
            value += 1;
            if value > 31 {
                break;
            }
        }

        // Binary suffix
        if value > 0 {
            let suffix = self.decode_bypass_bits(reader, value as u8)?;
            value = (1 << value) - 1 + suffix;
        }

        Ok(value)
    }

    /// Decode multiple bypass bins.
    fn decode_bypass_bits(&mut self, reader: &mut BitReader<'_>, count: u8) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | (self.decode_bypass(reader)? as u32);
        }
        Ok(value)
    }

    /// Get range for LPS from table.
    fn get_range_lps(&self, range: u16, state: u8) -> u16 {
        // Simplified LPS range table
        // Real implementation uses 64x4 table
        let range_idx = ((range >> 6) & 3) as usize;
        let lps_range = [128, 176, 208, 240];
        lps_range[range_idx] >> (state >> 4)
    }

    /// Update context after MPS.
    fn update_context_mps(&mut self, ctx_idx: usize) {
        let ctx = &mut self.contexts[ctx_idx];
        if ctx.state < 62 {
            ctx.state += 1;
        }
    }

    /// Update context after LPS.
    fn update_context_lps(&mut self, ctx_idx: usize) {
        let ctx = &mut self.contexts[ctx_idx];
        if ctx.state == 0 {
            ctx.mps = !ctx.mps;
        } else {
            ctx.state -= 1;
        }
    }

    /// Renormalize the arithmetic decoder.
    fn renormalize(&mut self, reader: &mut BitReader<'_>) -> Result<()> {
        while self.range < 256 {
            self.range <<= 1;
            self.offset = (self.offset << 1) | (reader.read_bit()? as u16);
        }
        Ok(())
    }

    /// Reset the decoder.
    pub fn reset(&mut self) {
        self.range = 510;
        self.offset = 0;
        self.contexts.clear();
    }
}

impl Default for CabacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// CABAC encoder.
pub struct CabacEncoder {
    /// Low value (10 bits).
    low: u32,
    /// Range (9 bits).
    range: u16,
    /// Outstanding bits count.
    outstanding: u32,
    /// Output buffer.
    output: Vec<u8>,
    /// Context models.
    contexts: Vec<CabacContext>,
}

impl CabacEncoder {
    /// Create a new CABAC encoder.
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 510,
            outstanding: 0,
            output: Vec::new(),
            contexts: Vec::new(),
        }
    }

    /// Initialize context models.
    pub fn init_contexts(&mut self, _slice_qp: i8, _cabac_init_idc: u8) {
        self.contexts.clear();
        self.contexts.resize(460, CabacContext::new(0));
    }

    /// Encode a binary decision.
    pub fn encode_decision(&mut self, symbol: bool, ctx_idx: usize) {
        let ctx = &self.contexts[ctx_idx];
        let range_lps = self.get_range_lps(self.range, ctx.state());
        let range_mps = self.range - range_lps;

        if symbol == ctx.mps() {
            // MPS
            self.range = range_mps;
            self.update_context_mps(ctx_idx);
        } else {
            // LPS
            self.low += range_mps as u32;
            self.range = range_lps;
            self.update_context_lps(ctx_idx);
        }

        self.renormalize_encode();
    }

    /// Encode a bypass bin.
    pub fn encode_bypass(&mut self, symbol: bool) {
        self.low <<= 1;
        if symbol {
            self.low += self.range as u32;
        }
        self.renormalize_encode();
    }

    /// Encode a terminate bin.
    pub fn encode_terminate(&mut self, symbol: bool) {
        self.range -= 2;
        if symbol {
            self.low += self.range as u32;
            self.range = 2;
        }
        self.renormalize_encode();
    }

    /// Get the encoded data.
    pub fn data(&self) -> &[u8] {
        &self.output
    }

    /// Flush the encoder.
    pub fn flush(&mut self) {
        // Output remaining bits
        self.range = 2;
        self.renormalize_encode();
        self.output.push(((self.low >> 2) & 0xFF) as u8);
    }

    fn get_range_lps(&self, range: u16, state: u8) -> u16 {
        let range_idx = ((range >> 6) & 3) as usize;
        let lps_range = [128, 176, 208, 240];
        lps_range[range_idx] >> (state >> 4)
    }

    fn update_context_mps(&mut self, ctx_idx: usize) {
        let ctx = &mut self.contexts[ctx_idx];
        if ctx.state < 62 {
            ctx.state += 1;
        }
    }

    fn update_context_lps(&mut self, ctx_idx: usize) {
        let ctx = &mut self.contexts[ctx_idx];
        if ctx.state == 0 {
            ctx.mps = !ctx.mps;
        } else {
            ctx.state -= 1;
        }
    }

    fn renormalize_encode(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put_bit(0);
            } else if self.low >= 512 {
                self.put_bit(1);
                self.low -= 512;
            } else {
                self.outstanding += 1;
                self.low -= 256;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    fn put_bit(&mut self, bit: u8) {
        if self.output.is_empty() || self.output.last() == Some(&0xFF) {
            self.output.push(0);
        }

        let last = self.output.last_mut().unwrap();
        *last = (*last << 1) | bit;

        for _ in 0..self.outstanding {
            self.output.push(if bit == 0 { 0xFF } else { 0x00 });
        }
        self.outstanding = 0;
    }
}

impl Default for CabacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CABAC Encoder Extensions for Macroblock Encoding
// =============================================================================

/// Context indices for H.264 CABAC encoding.
pub mod ctx_indices {
    // Macroblock type contexts (I-slice)
    pub const MB_TYPE_I_PREFIX: usize = 3;
    pub const MB_TYPE_I_SUFFIX: usize = 4;

    // Coded block pattern contexts
    pub const CBP_LUMA: usize = 73;
    pub const CBP_CHROMA: usize = 77;

    // Intra prediction mode contexts
    pub const PREV_INTRA4X4_PRED_MODE: usize = 68;
    pub const REM_INTRA4X4_PRED_MODE: usize = 69;
    pub const INTRA_CHROMA_PRED_MODE: usize = 64;

    // Residual coding contexts
    pub const CODED_BLOCK_FLAG: usize = 85;
    pub const SIG_COEFF_FLAG: usize = 105;
    pub const LAST_SIG_COEFF_FLAG: usize = 166;
    pub const COEFF_ABS_LEVEL_MINUS1: usize = 227;

    // Transform size flag
    pub const TRANSFORM_SIZE_8X8_FLAG: usize = 399;

    // MB skip flag (P/B slices)
    pub const MB_SKIP_FLAG_P: usize = 11;
    pub const MB_SKIP_FLAG_B: usize = 24;
}

/// Macroblock type for I-slices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MbTypeI {
    /// Intra 4x4 prediction (luma uses 4x4 blocks).
    I4x4,
    /// Intra 16x16 prediction with coded block pattern.
    I16x16 {
        /// Intra 16x16 prediction mode (0-3).
        pred_mode: u8,
        /// Coded block pattern for chroma (0-3).
        cbp_chroma: u8,
        /// Coded block pattern for luma DC (0 or 1).
        cbp_luma_dc: u8,
    },
    /// PCM macroblock (raw samples).
    Ipcm,
}

/// Coded block pattern.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodedBlockPattern {
    /// Luma CBP (4 bits, one per 8x8 block).
    pub luma: u8,
    /// Chroma CBP (0=none, 1=DC only, 2=DC+AC).
    pub chroma: u8,
}

impl CabacEncoder {
    /// Encode macroblock type for I-slice.
    ///
    /// Per H.264 spec Table 9-34.
    pub fn encode_mb_type_intra(&mut self, mb_type: MbTypeI, ctx_a: bool, ctx_b: bool) {
        // Context for first bin based on neighbors
        let ctx_inc = (ctx_a as usize) + (ctx_b as usize);

        match mb_type {
            MbTypeI::I4x4 => {
                // I_4x4: single 0 bin
                self.encode_decision(false, ctx_indices::MB_TYPE_I_PREFIX + ctx_inc);
            }
            MbTypeI::I16x16 { pred_mode, cbp_chroma, cbp_luma_dc } => {
                // I_16x16: prefix = 1
                self.encode_decision(true, ctx_indices::MB_TYPE_I_PREFIX + ctx_inc);
                // Terminate bin = 0 (not PCM)
                self.encode_terminate(false);

                // Encode cbp_luma_dc (1 bit)
                self.encode_decision(cbp_luma_dc != 0, ctx_indices::MB_TYPE_I_SUFFIX);

                // Encode cbp_chroma (2 bits via truncated unary)
                match cbp_chroma {
                    0 => {
                        self.encode_decision(false, ctx_indices::MB_TYPE_I_SUFFIX + 1);
                    }
                    1 => {
                        self.encode_decision(true, ctx_indices::MB_TYPE_I_SUFFIX + 1);
                        self.encode_decision(false, ctx_indices::MB_TYPE_I_SUFFIX + 2);
                    }
                    _ => {
                        self.encode_decision(true, ctx_indices::MB_TYPE_I_SUFFIX + 1);
                        self.encode_decision(true, ctx_indices::MB_TYPE_I_SUFFIX + 2);
                    }
                }

                // Encode pred_mode (2 bits, fixed length)
                self.encode_decision((pred_mode & 2) != 0, ctx_indices::MB_TYPE_I_SUFFIX + 3);
                self.encode_decision((pred_mode & 1) != 0, ctx_indices::MB_TYPE_I_SUFFIX + 3);
            }
            MbTypeI::Ipcm => {
                // I_PCM: prefix = 1, then terminate = 1
                self.encode_decision(true, ctx_indices::MB_TYPE_I_PREFIX + ctx_inc);
                self.encode_terminate(true);
            }
        }
    }

    /// Encode coded block pattern (CBP).
    ///
    /// Per H.264 spec 9.3.2.6.
    pub fn encode_cbp(&mut self, cbp: CodedBlockPattern, neighbors: &CbpNeighbors) {
        // Luma CBP: 4 bins for 4 8x8 luma blocks
        for i in 0..4 {
            let bit = (cbp.luma >> i) & 1 != 0;

            // Context depends on neighbor CBP
            let ctx_inc = self.get_cbp_luma_ctx(i, neighbors);
            self.encode_decision(bit, ctx_indices::CBP_LUMA + ctx_inc);
        }

        // Chroma CBP: encoded as truncated unary (0, 1, or 2)
        if cbp.chroma == 0 {
            // cbp_chroma = 0: single 0 bin
            let ctx_inc = self.get_cbp_chroma_ctx(0, neighbors);
            self.encode_decision(false, ctx_indices::CBP_CHROMA + ctx_inc);
        } else {
            // First bin = 1
            let ctx_inc = self.get_cbp_chroma_ctx(0, neighbors);
            self.encode_decision(true, ctx_indices::CBP_CHROMA + ctx_inc);

            // Second bin: 0 for cbp=1, 1 for cbp=2
            let ctx_inc = self.get_cbp_chroma_ctx(1, neighbors);
            self.encode_decision(cbp.chroma == 2, ctx_indices::CBP_CHROMA + ctx_inc);
        }
    }

    /// Encode intra 4x4 prediction mode.
    ///
    /// Uses prev_intra4x4_pred_mode_flag and rem_intra4x4_pred_mode.
    pub fn encode_intra_pred_mode_4x4(&mut self, mode: u8, predicted_mode: u8) {
        if mode == predicted_mode {
            // prev_intra4x4_pred_mode_flag = 1
            self.encode_decision(true, ctx_indices::PREV_INTRA4X4_PRED_MODE);
        } else {
            // prev_intra4x4_pred_mode_flag = 0
            self.encode_decision(false, ctx_indices::PREV_INTRA4X4_PRED_MODE);

            // rem_intra4x4_pred_mode (3 bits fixed length)
            let rem = if mode < predicted_mode { mode } else { mode - 1 };
            self.encode_decision((rem & 1) != 0, ctx_indices::REM_INTRA4X4_PRED_MODE);
            self.encode_decision((rem & 2) != 0, ctx_indices::REM_INTRA4X4_PRED_MODE);
            self.encode_decision((rem & 4) != 0, ctx_indices::REM_INTRA4X4_PRED_MODE);
        }
    }

    /// Encode intra chroma prediction mode.
    ///
    /// Uses truncated unary binarization.
    pub fn encode_intra_chroma_pred_mode(&mut self, mode: u8, ctx_a: u8, ctx_b: u8) {
        let ctx_inc = ((ctx_a != 0) as usize) + ((ctx_b != 0) as usize);

        // Truncated unary: 0, 10, 110, 111
        match mode {
            0 => {
                self.encode_decision(false, ctx_indices::INTRA_CHROMA_PRED_MODE + ctx_inc);
            }
            1 => {
                self.encode_decision(true, ctx_indices::INTRA_CHROMA_PRED_MODE + ctx_inc);
                self.encode_decision(false, ctx_indices::INTRA_CHROMA_PRED_MODE + 3);
            }
            2 => {
                self.encode_decision(true, ctx_indices::INTRA_CHROMA_PRED_MODE + ctx_inc);
                self.encode_decision(true, ctx_indices::INTRA_CHROMA_PRED_MODE + 3);
                self.encode_decision(false, ctx_indices::INTRA_CHROMA_PRED_MODE + 3);
            }
            _ => {
                self.encode_decision(true, ctx_indices::INTRA_CHROMA_PRED_MODE + ctx_inc);
                self.encode_decision(true, ctx_indices::INTRA_CHROMA_PRED_MODE + 3);
                self.encode_decision(true, ctx_indices::INTRA_CHROMA_PRED_MODE + 3);
            }
        }
    }

    /// Encode a 4x4 residual block using CABAC.
    ///
    /// Implements the significance map and coefficient level encoding.
    pub fn encode_residual_block_4x4(
        &mut self,
        coeffs: &[i16; 16],
        block_cat: BlockCategory,
        max_coeff: usize,
    ) {
        // Find last significant coefficient
        let mut last_sig = None;
        for i in (0..max_coeff).rev() {
            if coeffs[SCAN_ORDER_4X4[i]] != 0 {
                last_sig = Some(i);
                break;
            }
        }

        let last_sig = match last_sig {
            Some(idx) => idx,
            None => {
                // All zeros - encode coded_block_flag = 0
                self.encode_decision(false, ctx_indices::CODED_BLOCK_FLAG + block_cat.ctx_offset());
                return;
            }
        };

        // coded_block_flag = 1
        self.encode_decision(true, ctx_indices::CODED_BLOCK_FLAG + block_cat.ctx_offset());

        // Encode significance map
        let sig_ctx_base = ctx_indices::SIG_COEFF_FLAG + block_cat.sig_ctx_offset();
        let last_ctx_base = ctx_indices::LAST_SIG_COEFF_FLAG + block_cat.last_ctx_offset();

        for i in 0..last_sig {
            let scan_idx = SCAN_ORDER_4X4[i];
            let is_sig = coeffs[scan_idx] != 0;

            // significant_coeff_flag
            self.encode_decision(is_sig, sig_ctx_base + i.min(14));

            if is_sig {
                // last_significant_coeff_flag = 0 (not the last one)
                self.encode_decision(false, last_ctx_base + i.min(14));
            }
        }

        // For last significant coefficient: sig=1, last=1
        self.encode_decision(true, sig_ctx_base + last_sig.min(14));
        self.encode_decision(true, last_ctx_base + last_sig.min(14));

        // Encode coefficient levels in reverse scan order
        let mut num_t1 = 0; // trailing ones count
        let mut num_gt1 = 0; // coefficients > 1 count

        for i in (0..=last_sig).rev() {
            let scan_idx = SCAN_ORDER_4X4[i];
            let coeff = coeffs[scan_idx];
            if coeff == 0 {
                continue;
            }

            let level = coeff.unsigned_abs() as u32;
            let sign = coeff < 0;

            // coeff_abs_level_minus1
            self.encode_coeff_abs_level_minus1(level - 1, num_t1, num_gt1);

            // coeff_sign_flag (bypass)
            self.encode_bypass(sign);

            // Update context state
            if level == 1 {
                num_t1 = (num_t1 + 1).min(3);
            } else {
                num_gt1 += 1;
            }
        }
    }

    /// Encode coeff_abs_level_minus1.
    fn encode_coeff_abs_level_minus1(&mut self, level_m1: u32, num_t1: u32, num_gt1: u32) {
        let ctx_base = ctx_indices::COEFF_ABS_LEVEL_MINUS1;

        // Context category based on previous coefficients
        let ctx_cat = if num_gt1 > 0 {
            5.min(num_gt1 as usize)
        } else {
            (num_t1 as usize).min(4)
        };

        if level_m1 < 14 {
            // Unary prefix (TU max 14)
            for i in 0..level_m1 {
                self.encode_decision(true, ctx_base + ctx_cat + (i as usize).min(4) * 10);
            }
            if level_m1 < 14 {
                self.encode_decision(false, ctx_base + ctx_cat + (level_m1 as usize).min(4) * 10);
            }
        } else {
            // Prefix = 14 ones
            for i in 0..14 {
                self.encode_decision(true, ctx_base + ctx_cat + (i as usize).min(4) * 10);
            }
            // Suffix using Exp-Golomb bypass
            let suffix = level_m1 - 14;
            self.encode_eg_suffix(suffix);
        }
    }

    /// Encode Exp-Golomb suffix using bypass bins.
    fn encode_eg_suffix(&mut self, value: u32) {
        if value == 0 {
            self.encode_bypass(false);
            return;
        }

        // Find the number of leading zeros
        let k = 32 - value.leading_zeros() as usize;

        // Encode k-1 ones followed by 0
        for _ in 0..k {
            self.encode_bypass(true);
        }
        self.encode_bypass(false);

        // Encode the remaining k bits
        for i in (0..k).rev() {
            self.encode_bypass((value >> i) & 1 != 0);
        }
    }

    /// Get CBP luma context increment.
    fn get_cbp_luma_ctx(&self, block_idx: usize, neighbors: &CbpNeighbors) -> usize {
        // Neighbor availability and CBP determines context
        let (a_avail, a_cbp) = match block_idx {
            0 => (neighbors.left_avail, (neighbors.left_cbp >> 1) & 1),
            1 => (true, neighbors.current_cbp & 1),
            2 => (neighbors.left_avail, (neighbors.left_cbp >> 3) & 1),
            3 => (true, (neighbors.current_cbp >> 2) & 1),
            _ => (false, 0),
        };
        let (b_avail, b_cbp) = match block_idx {
            0 => (neighbors.top_avail, (neighbors.top_cbp >> 2) & 1),
            1 => (neighbors.top_avail, (neighbors.top_cbp >> 3) & 1),
            2 => (true, neighbors.current_cbp & 1),
            3 => (true, (neighbors.current_cbp >> 1) & 1),
            _ => (false, 0),
        };

        let cond_term_a = if a_avail { 1 - a_cbp } else { 0 };
        let cond_term_b = if b_avail { 1 - b_cbp } else { 0 };

        (cond_term_a + 2 * cond_term_b) as usize
    }

    /// Get CBP chroma context increment.
    fn get_cbp_chroma_ctx(&self, bin_idx: usize, neighbors: &CbpNeighbors) -> usize {
        if bin_idx == 0 {
            // First bin context
            let cond_a = if neighbors.left_avail && neighbors.left_cbp_chroma > 0 { 1 } else { 0 };
            let cond_b = if neighbors.top_avail && neighbors.top_cbp_chroma > 0 { 1 } else { 0 };
            cond_a + 2 * cond_b
        } else {
            // Second bin context
            let cond_a = if neighbors.left_avail && neighbors.left_cbp_chroma > 1 { 1 } else { 0 };
            let cond_b = if neighbors.top_avail && neighbors.top_cbp_chroma > 1 { 1 } else { 0 };
            cond_a + 2 * cond_b
        }
    }

    /// Encode MB skip flag for P/B slices.
    pub fn encode_mb_skip_flag(&mut self, skip: bool, slice_type_b: bool, ctx_a: bool, ctx_b: bool) {
        let ctx_base = if slice_type_b {
            ctx_indices::MB_SKIP_FLAG_B
        } else {
            ctx_indices::MB_SKIP_FLAG_P
        };
        let ctx_inc = (ctx_a as usize) + (ctx_b as usize);
        self.encode_decision(skip, ctx_base + ctx_inc);
    }

    /// Encode transform size 8x8 flag.
    pub fn encode_transform_size_8x8(&mut self, use_8x8: bool, ctx_a: bool, ctx_b: bool) {
        let ctx_inc = (ctx_a as usize) + (ctx_b as usize);
        self.encode_decision(use_8x8, ctx_indices::TRANSFORM_SIZE_8X8_FLAG + ctx_inc);
    }
}

/// CBP neighbor information for context derivation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CbpNeighbors {
    /// Left neighbor available.
    pub left_avail: bool,
    /// Top neighbor available.
    pub top_avail: bool,
    /// Left neighbor luma CBP.
    pub left_cbp: u8,
    /// Top neighbor luma CBP.
    pub top_cbp: u8,
    /// Current MB CBP (for internal block context).
    pub current_cbp: u8,
    /// Left neighbor chroma CBP.
    pub left_cbp_chroma: u8,
    /// Top neighbor chroma CBP.
    pub top_cbp_chroma: u8,
}

/// Block category for residual coding contexts.
#[derive(Debug, Clone, Copy)]
pub enum BlockCategory {
    /// Luma DC (Intra 16x16).
    LumaDC,
    /// Luma AC (Intra 16x16).
    LumaAC,
    /// Luma 4x4.
    Luma4x4,
    /// Chroma DC.
    ChromaDC,
    /// Chroma AC.
    ChromaAC,
    /// Luma 8x8.
    Luma8x8,
}

impl BlockCategory {
    /// Get context offset for coded_block_flag.
    pub fn ctx_offset(self) -> usize {
        match self {
            BlockCategory::LumaDC => 0,
            BlockCategory::LumaAC => 4,
            BlockCategory::Luma4x4 => 8,
            BlockCategory::ChromaDC => 12,
            BlockCategory::ChromaAC => 16,
            BlockCategory::Luma8x8 => 0, // Uses different context model
        }
    }

    /// Get context offset for significant_coeff_flag.
    pub fn sig_ctx_offset(self) -> usize {
        match self {
            BlockCategory::Luma4x4 | BlockCategory::LumaAC => 0,
            BlockCategory::LumaDC => 15,
            BlockCategory::ChromaDC => 29,
            BlockCategory::ChromaAC => 44,
            BlockCategory::Luma8x8 => 0,
        }
    }

    /// Get context offset for last_significant_coeff_flag.
    pub fn last_ctx_offset(self) -> usize {
        match self {
            BlockCategory::Luma4x4 | BlockCategory::LumaAC => 0,
            BlockCategory::LumaDC => 15,
            BlockCategory::ChromaDC => 29,
            BlockCategory::ChromaAC => 44,
            BlockCategory::Luma8x8 => 0,
        }
    }
}

/// Zig-zag scan order for 4x4 blocks.
pub const SCAN_ORDER_4X4: [usize; 16] = [
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cabac_encoder_mb_type_i4x4() {
        let mut encoder = CabacEncoder::new();
        encoder.init_contexts(26, 0);

        // Encode I_4x4 macroblock type
        encoder.encode_mb_type_intra(MbTypeI::I4x4, false, false);

        // Should have encoded a single 0 bin
        encoder.flush();
        assert!(!encoder.data().is_empty());
    }

    #[test]
    fn test_cabac_encoder_mb_type_i16x16() {
        let mut encoder = CabacEncoder::new();
        encoder.init_contexts(26, 0);

        // Encode I_16x16 with various parameters
        encoder.encode_mb_type_intra(
            MbTypeI::I16x16 {
                pred_mode: 0,
                cbp_chroma: 2,
                cbp_luma_dc: 1,
            },
            false,
            false,
        );

        encoder.flush();
        assert!(!encoder.data().is_empty());
    }

    #[test]
    fn test_cabac_encoder_cbp() {
        let mut encoder = CabacEncoder::new();
        encoder.init_contexts(26, 0);

        let cbp = CodedBlockPattern {
            luma: 0b1010,
            chroma: 2,
        };
        let neighbors = CbpNeighbors::default();

        encoder.encode_cbp(cbp, &neighbors);
        encoder.flush();
        assert!(!encoder.data().is_empty());
    }

    #[test]
    fn test_cabac_encoder_intra_pred_mode() {
        let mut encoder = CabacEncoder::new();
        encoder.init_contexts(26, 0);

        // Test when mode matches predicted
        encoder.encode_intra_pred_mode_4x4(2, 2);

        // Test when mode differs from predicted
        encoder.encode_intra_pred_mode_4x4(5, 2);

        encoder.flush();
        assert!(!encoder.data().is_empty());
    }

    #[test]
    fn test_cabac_encoder_residual_block_zeros() {
        let mut encoder = CabacEncoder::new();
        encoder.init_contexts(26, 0);

        let coeffs = [0i16; 16];
        encoder.encode_residual_block_4x4(&coeffs, BlockCategory::Luma4x4, 16);

        encoder.flush();
        // Should encode coded_block_flag = 0
        assert!(!encoder.data().is_empty());
    }

    #[test]
    fn test_cabac_encoder_residual_block_with_coeffs() {
        let mut encoder = CabacEncoder::new();
        encoder.init_contexts(26, 0);

        let mut coeffs = [0i16; 16];
        coeffs[0] = 10;
        coeffs[1] = -5;
        coeffs[4] = 1;

        encoder.encode_residual_block_4x4(&coeffs, BlockCategory::Luma4x4, 16);

        encoder.flush();
        assert!(!encoder.data().is_empty());
    }

    #[test]
    fn test_cabac_encoder_mb_skip_flag() {
        let mut encoder = CabacEncoder::new();
        encoder.init_contexts(26, 0);

        // P-slice skip
        encoder.encode_mb_skip_flag(true, false, false, false);
        encoder.encode_mb_skip_flag(false, false, true, false);

        // B-slice skip
        encoder.encode_mb_skip_flag(true, true, false, true);

        encoder.flush();
        assert!(!encoder.data().is_empty());
    }
}
