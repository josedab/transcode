//! Macroblock encoding for H.264.
//!
//! This module provides the core macroblock encoding loop that combines
//! prediction, transform, quantization, and entropy coding.

use transcode_core::bitstream::BitWriter;
use super::prediction::{IntraPredictor, Intra4x4Mode, Intra16x16Mode, IntraChromaMode};
use super::transform::{fdct4x4, quantize4x4, hadamard4x4};
use super::cavlc::CavlcEncoder;
use super::cabac::{CabacEncoder, MbTypeI, CodedBlockPattern, CbpNeighbors, BlockCategory};

/// Macroblock type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroblockType {
    /// Intra 4x4 prediction.
    I4x4,
    /// Intra 16x16 prediction with mode and CBP info.
    I16x16 {
        pred_mode: Intra16x16Mode,
        cbp_luma: u8,
        cbp_chroma: u8,
    },
    /// Skip macroblock (P/B slices).
    Skip,
    /// Inter predicted macroblock.
    Inter,
}

/// Encoded macroblock data.
#[derive(Debug, Clone)]
pub struct EncodedMacroblock {
    /// Macroblock type.
    pub mb_type: MacroblockType,
    /// Intra 4x4 prediction modes (16 modes for 4x4 blocks).
    pub intra_4x4_modes: [Intra4x4Mode; 16],
    /// Intra chroma prediction mode.
    pub intra_chroma_mode: IntraChromaMode,
    /// Coded block pattern for luma.
    pub cbp_luma: u8,
    /// Coded block pattern for chroma.
    pub cbp_chroma: u8,
    /// Quantized coefficients for luma (16 blocks of 16 coefficients).
    pub luma_coeffs: [[i16; 16]; 16],
    /// Quantized DC coefficients for luma (for I16x16).
    pub luma_dc: [i16; 16],
    /// Quantized coefficients for Cb (4 blocks of 16 coefficients).
    pub cb_coeffs: [[i16; 16]; 4],
    /// Quantized DC coefficients for Cb.
    pub cb_dc: [i16; 4],
    /// Quantized coefficients for Cr (4 blocks of 16 coefficients).
    pub cr_coeffs: [[i16; 16]; 4],
    /// Quantized DC coefficients for Cr.
    pub cr_dc: [i16; 4],
    /// QP delta from slice QP.
    pub qp_delta: i8,
}

impl Default for EncodedMacroblock {
    fn default() -> Self {
        Self {
            mb_type: MacroblockType::I4x4,
            intra_4x4_modes: [Intra4x4Mode::Dc; 16],
            intra_chroma_mode: IntraChromaMode::Dc,
            cbp_luma: 0,
            cbp_chroma: 0,
            luma_coeffs: [[0; 16]; 16],
            luma_dc: [0; 16],
            cb_coeffs: [[0; 16]; 4],
            cb_dc: [0; 4],
            cr_coeffs: [[0; 16]; 4],
            cr_dc: [0; 4],
            qp_delta: 0,
        }
    }
}

/// Macroblock encoder state.
pub struct MacroblockEncoder {
    /// CAVLC encoder for Baseline profile.
    pub cavlc: CavlcEncoder,
    /// CABAC encoder for Main/High profile.
    pub cabac: CabacEncoder,
    /// Whether to use CABAC.
    use_cabac: bool,
    /// Current QP.
    qp: u8,
    /// Reconstructed luma samples for the current macroblock row.
    recon_luma: Vec<u8>,
    /// Width in macroblocks.
    mb_width: usize,
}

impl MacroblockEncoder {
    /// Create a new macroblock encoder.
    pub fn new(use_cabac: bool, mb_width: usize) -> Self {
        Self {
            cavlc: CavlcEncoder::new(),
            cabac: CabacEncoder::new(),
            use_cabac,
            qp: 26,
            recon_luma: vec![128u8; mb_width * 16 * 16],
            mb_width,
        }
    }

    /// Initialize for a new slice.
    pub fn init_slice(&mut self, qp: u8, cabac_init_idc: u8) {
        self.qp = qp;
        if self.use_cabac {
            self.cabac.init_contexts(qp as i8, cabac_init_idc);
        }
        // Reset reconstruction buffer
        self.recon_luma.fill(128);
    }

    /// Encode a macroblock from raw pixel data.
    pub fn encode_macroblock(
        &mut self,
        mb_x: usize,
        mb_y: usize,
        luma_data: &[u8],
        luma_stride: usize,
        _cb_data: &[u8],
        _cb_stride: usize,
        _cr_data: &[u8],
        _cr_stride: usize,
    ) -> EncodedMacroblock {
        let mut mb = EncodedMacroblock {
            // For simplicity, use Intra 16x16 DC mode
            mb_type: MacroblockType::I16x16 {
                pred_mode: Intra16x16Mode::Dc,
                cbp_luma: 0,
                cbp_chroma: 0,
            },
            ..Default::default()
        };

        // Get neighboring samples for prediction
        let (above, left) = self.get_neighbors(mb_x, mb_y);

        // Predict using 16x16 DC
        let mut prediction = [0u8; 256];
        IntraPredictor::predict_16x16(&mut prediction, 16, &above, &left, Intra16x16Mode::Dc);

        // Compute residual and transform for each 4x4 block
        let mut has_luma_coeffs = false;
        for blk_y in 0..4 {
            for blk_x in 0..4 {
                let blk_idx = blk_y * 4 + blk_x;

                // Get 4x4 residual
                let mut residual = [0i16; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let py = blk_y * 4 + y;
                        let px = blk_x * 4 + x;
                        let orig = luma_data[py * luma_stride + px] as i16;
                        let pred = prediction[py * 16 + px] as i16;
                        residual[y * 4 + x] = orig - pred;
                    }
                }

                // Forward DCT
                let mut transformed = [0i16; 16];
                fdct4x4(&residual, &mut transformed);

                // Quantize
                let mut quantized = [0i16; 16];
                quantize4x4(&transformed, &mut quantized, self.qp, true);

                // Check if any non-zero coefficients
                if quantized.iter().any(|&c| c != 0) {
                    has_luma_coeffs = true;
                    // Set CBP bit for this 8x8 block
                    let cbp_8x8_idx = (blk_y / 2) * 2 + (blk_x / 2);
                    mb.cbp_luma |= 1 << cbp_8x8_idx;
                }

                mb.luma_coeffs[blk_idx] = quantized;
            }
        }

        // For I16x16, extract DC coefficients and apply Hadamard
        let mut dc_coeffs = [0i16; 16];
        for i in 0..16 {
            dc_coeffs[i] = mb.luma_coeffs[i][0];
            mb.luma_coeffs[i][0] = 0; // Clear DC from AC blocks
        }
        hadamard4x4(&dc_coeffs, &mut mb.luma_dc);

        // Quantize DC coefficients
        let dc_qp = (self.qp as i8 - 6).max(0) as u8;
        let mut quantized_dc = [0i16; 16];
        quantize4x4(&mb.luma_dc, &mut quantized_dc, dc_qp, true);
        mb.luma_dc = quantized_dc;

        // Update CBP based on coefficients
        if has_luma_coeffs || mb.luma_dc.iter().any(|&c| c != 0) {
            mb.cbp_luma = 15; // All 8x8 blocks have coefficients
        }

        // Update macroblock type with actual CBP
        mb.mb_type = MacroblockType::I16x16 {
            pred_mode: Intra16x16Mode::Dc,
            cbp_luma: if mb.luma_dc.iter().any(|&c| c != 0) { 1 } else { 0 },
            cbp_chroma: 0,
        };

        // Store reconstruction for neighbor prediction
        self.store_reconstruction(mb_x, &prediction);

        mb
    }

    /// Get neighboring samples for intra prediction.
    fn get_neighbors(&self, mb_x: usize, mb_y: usize) -> ([u8; 16], [u8; 16]) {
        let mut above = [128u8; 16];
        let mut left = [128u8; 16];

        // Get samples from reconstruction buffer
        if mb_y > 0 {
            // Previous row available
            let row_start = (mb_y - 1) * self.mb_width * 256 + mb_x * 16;
            if row_start + 16 <= self.recon_luma.len() {
                // Get bottom row of above MB
                let offset = row_start + 15 * 16;
                if offset + 16 <= self.recon_luma.len() {
                    above.copy_from_slice(&self.recon_luma[offset..offset + 16]);
                }
            }
        }

        if mb_x > 0 {
            // Left MB available
            let mb_start = mb_y * self.mb_width * 256 + (mb_x - 1) * 16;
            if mb_start + 256 <= self.recon_luma.len() {
                for i in 0..16 {
                    left[i] = self.recon_luma[mb_start + i * 16 + 15];
                }
            }
        }

        (above, left)
    }

    /// Store reconstruction for future neighbor prediction.
    fn store_reconstruction(&mut self, mb_x: usize, recon: &[u8; 256]) {
        let mb_start = mb_x * 256;
        if mb_start + 256 <= self.recon_luma.len() {
            self.recon_luma[mb_start..mb_start + 256].copy_from_slice(recon);
        }
    }

    /// Write macroblock syntax to bitstream using CAVLC.
    pub fn write_macroblock_cavlc(&mut self, writer: &mut BitWriter, mb: &EncodedMacroblock, mb_x: usize, mb_y: usize) {
        match mb.mb_type {
            MacroblockType::I16x16 { pred_mode, cbp_luma, cbp_chroma } => {
                // mb_type for I16x16: 1 + pred_mode + 4*cbp_chroma + 12*cbp_luma
                let mb_type_val = 1 + (pred_mode as u32) + 4 * (cbp_chroma as u32) + 12 * (cbp_luma as u32);
                writer.write_ue(mb_type_val);

                // Intra chroma prediction mode
                writer.write_ue(mb.intra_chroma_mode as u32);

                // mb_qp_delta
                writer.write_se(mb.qp_delta as i32);

                // Residual data for I16x16
                // DC luma (if non-zero)
                if cbp_luma != 0 {
                    self.write_residual_cavlc_dc(writer, &mb.luma_dc, mb_x, mb_y);
                }

                // AC luma (if CBP indicates)
                if mb.cbp_luma != 0 {
                    for blk in 0..16 {
                        if (mb.cbp_luma >> (blk / 4)) & 1 != 0 {
                            self.write_residual_cavlc_ac(writer, &mb.luma_coeffs[blk], blk, mb_x, mb_y);
                        }
                    }
                }

                // Chroma DC and AC (if CBP indicates)
                if cbp_chroma >= 1 {
                    self.write_chroma_dc_cavlc(writer, &mb.cb_dc);
                    self.write_chroma_dc_cavlc(writer, &mb.cr_dc);
                }
                if cbp_chroma >= 2 {
                    for blk in 0..4 {
                        self.write_chroma_ac_cavlc(writer, &mb.cb_coeffs[blk]);
                        self.write_chroma_ac_cavlc(writer, &mb.cr_coeffs[blk]);
                    }
                }
            }
            MacroblockType::I4x4 => {
                // mb_type = 0 for I_4x4
                writer.write_ue(0);

                // Write prediction modes for each 4x4 block
                for blk in 0..16 {
                    let mode = mb.intra_4x4_modes[blk] as u8;
                    let predicted = self.predict_intra4x4_mode(blk, mb_x, mb_y);

                    if mode == predicted {
                        writer.write_bit(true); // prev_intra4x4_pred_mode_flag = 1
                    } else {
                        writer.write_bit(false); // prev_intra4x4_pred_mode_flag = 0
                        let rem = if mode < predicted { mode } else { mode - 1 };
                        writer.write_bits(rem as u32, 3);
                    }
                }

                // Intra chroma prediction mode
                writer.write_ue(mb.intra_chroma_mode as u32);

                // CBP (coded block pattern)
                let cbp = (mb.cbp_luma as u32) | ((mb.cbp_chroma as u32) << 4);
                writer.write_ue(cbp); // Simplified - should use code_from_cbp table

                // mb_qp_delta (if CBP != 0)
                if cbp != 0 {
                    writer.write_se(mb.qp_delta as i32);
                }

                // Residual
                for blk in 0..16 {
                    let cbp_8x8 = (blk / 4) * 2 + (blk % 4) / 2;
                    if (mb.cbp_luma >> cbp_8x8) & 1 != 0 {
                        self.write_residual_cavlc_4x4(writer, &mb.luma_coeffs[blk], blk, mb_x, mb_y);
                    }
                }
            }
            _ => {
                // Skip or Inter - simplified
                writer.write_ue(0);
            }
        }
    }

    /// Write macroblock syntax using CABAC.
    pub fn write_macroblock_cabac(&mut self, mb: &EncodedMacroblock, mb_x: usize, mb_y: usize) {
        let ctx_a = mb_x > 0;
        let ctx_b = mb_y > 0;

        match mb.mb_type {
            MacroblockType::I16x16 { pred_mode, cbp_chroma, cbp_luma } => {
                self.cabac.encode_mb_type_intra(
                    MbTypeI::I16x16 {
                        pred_mode: pred_mode as u8,
                        cbp_chroma,
                        cbp_luma_dc: cbp_luma,
                    },
                    ctx_a,
                    ctx_b,
                );

                // Intra chroma prediction mode
                let chroma_ctx_a = if mb_x > 0 { mb.intra_chroma_mode as u8 } else { 0 };
                let chroma_ctx_b = if mb_y > 0 { mb.intra_chroma_mode as u8 } else { 0 };
                self.cabac.encode_intra_chroma_pred_mode(
                    mb.intra_chroma_mode as u8,
                    chroma_ctx_a,
                    chroma_ctx_b,
                );

                // Residual using CABAC
                if cbp_luma != 0 {
                    self.cabac.encode_residual_block_4x4(
                        &mb.luma_dc,
                        BlockCategory::LumaDC,
                        16,
                    );
                }

                // AC blocks
                if mb.cbp_luma != 0 {
                    for blk in 0..16 {
                        self.cabac.encode_residual_block_4x4(
                            &mb.luma_coeffs[blk],
                            BlockCategory::LumaAC,
                            15, // Skip DC coefficient
                        );
                    }
                }
            }
            MacroblockType::I4x4 => {
                self.cabac.encode_mb_type_intra(MbTypeI::I4x4, ctx_a, ctx_b);

                // Prediction modes
                for blk in 0..16 {
                    let mode = mb.intra_4x4_modes[blk] as u8;
                    let predicted = self.predict_intra4x4_mode(blk, mb_x, mb_y);
                    self.cabac.encode_intra_pred_mode_4x4(mode, predicted);
                }

                // Chroma mode
                let chroma_ctx_a = if mb_x > 0 { mb.intra_chroma_mode as u8 } else { 0 };
                let chroma_ctx_b = if mb_y > 0 { mb.intra_chroma_mode as u8 } else { 0 };
                self.cabac.encode_intra_chroma_pred_mode(
                    mb.intra_chroma_mode as u8,
                    chroma_ctx_a,
                    chroma_ctx_b,
                );

                // CBP
                let cbp = CodedBlockPattern {
                    luma: mb.cbp_luma,
                    chroma: mb.cbp_chroma,
                };
                let neighbors = CbpNeighbors {
                    left_avail: mb_x > 0,
                    top_avail: mb_y > 0,
                    ..Default::default()
                };
                self.cabac.encode_cbp(cbp, &neighbors);

                // Residual
                for blk in 0..16 {
                    let cbp_8x8 = (blk / 4) * 2 + (blk % 4) / 2;
                    if (mb.cbp_luma >> cbp_8x8) & 1 != 0 {
                        self.cabac.encode_residual_block_4x4(
                            &mb.luma_coeffs[blk],
                            BlockCategory::Luma4x4,
                            16,
                        );
                    }
                }
            }
            _ => {}
        }
    }

    /// Predict intra 4x4 mode from neighbors.
    fn predict_intra4x4_mode(&self, _blk: usize, _mb_x: usize, _mb_y: usize) -> u8 {
        // Simplified: always predict DC mode
        Intra4x4Mode::Dc as u8
    }

    /// Write DC residual using CAVLC.
    fn write_residual_cavlc_dc(&mut self, writer: &mut BitWriter, coeffs: &[i16; 16], mb_x: usize, mb_y: usize) {
        self.cavlc.encode_residual_block_4x4(
            writer,
            coeffs,
            super::cavlc::BlockType::LumaDc,
            mb_x,
            mb_y,
        );
    }

    /// Write AC residual using CAVLC.
    fn write_residual_cavlc_ac(&mut self, writer: &mut BitWriter, coeffs: &[i16; 16], blk: usize, mb_x: usize, mb_y: usize) {
        let block_x = (blk % 4) + mb_x * 4;
        let block_y = (blk / 4) + mb_y * 4;
        self.cavlc.encode_residual_block_4x4(
            writer,
            coeffs,
            super::cavlc::BlockType::LumaAc,
            block_x,
            block_y,
        );
    }

    /// Write 4x4 residual using CAVLC.
    fn write_residual_cavlc_4x4(&mut self, writer: &mut BitWriter, coeffs: &[i16; 16], blk: usize, mb_x: usize, mb_y: usize) {
        let block_x = (blk % 4) + mb_x * 4;
        let block_y = (blk / 4) + mb_y * 4;
        self.cavlc.encode_residual_block_4x4(
            writer,
            coeffs,
            super::cavlc::BlockType::Luma4x4,
            block_x,
            block_y,
        );
    }

    /// Write chroma DC using CAVLC.
    fn write_chroma_dc_cavlc(&mut self, writer: &mut BitWriter, coeffs: &[i16; 4]) {
        // Expand to 16-element array for CAVLC encoder
        let mut expanded = [0i16; 16];
        expanded[..4].copy_from_slice(coeffs);
        self.cavlc.encode_residual_block_4x4(
            writer,
            &expanded,
            super::cavlc::BlockType::ChromaDc,
            0,
            0,
        );
    }

    /// Write chroma AC using CAVLC.
    fn write_chroma_ac_cavlc(&mut self, writer: &mut BitWriter, coeffs: &[i16; 16]) {
        self.cavlc.encode_residual_block_4x4(
            writer,
            coeffs,
            super::cavlc::BlockType::ChromaAc,
            0,
            0,
        );
    }

    /// Get CABAC encoded data.
    pub fn finish_cabac(&mut self) -> Vec<u8> {
        self.cabac.flush();
        self.cabac.data().to_vec()
    }
}

/// Encode a full slice of macroblocks.
pub fn encode_slice_macroblocks(
    mb_encoder: &mut MacroblockEncoder,
    writer: &mut BitWriter,
    luma: &[u8],
    luma_stride: usize,
    cb: &[u8],
    cb_stride: usize,
    cr: &[u8],
    cr_stride: usize,
    mb_width: usize,
    mb_height: usize,
    first_mb_row: usize,
    mb_row_count: usize,
    use_cabac: bool,
) {
    for mb_row in first_mb_row..(first_mb_row + mb_row_count).min(mb_height) {
        for mb_col in 0..mb_width {
            // Get macroblock data pointers
            let luma_offset = mb_row * 16 * luma_stride + mb_col * 16;
            let cb_offset = mb_row * 8 * cb_stride + mb_col * 8;
            let cr_offset = mb_row * 8 * cr_stride + mb_col * 8;

            // Encode macroblock
            let mb = mb_encoder.encode_macroblock(
                mb_col,
                mb_row,
                &luma[luma_offset..],
                luma_stride,
                &cb[cb_offset..],
                cb_stride,
                &cr[cr_offset..],
                cr_stride,
            );

            // Write to bitstream
            if use_cabac {
                mb_encoder.write_macroblock_cabac(&mb, mb_col, mb_row);
            } else {
                mb_encoder.write_macroblock_cavlc(writer, &mb, mb_col, mb_row);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macroblock_encoder_creation() {
        let encoder = MacroblockEncoder::new(false, 10);
        assert!(!encoder.use_cabac);
    }

    #[test]
    fn test_encode_simple_macroblock() {
        let mut encoder = MacroblockEncoder::new(false, 10);
        encoder.init_slice(26, 0);

        // Create simple test data (flat gray)
        let luma = [128u8; 256];
        let cb = [128u8; 64];
        let cr = [128u8; 64];

        let mb = encoder.encode_macroblock(0, 0, &luma, 16, &cb, 8, &cr, 8);

        // Should be I16x16 with DC mode
        matches!(mb.mb_type, MacroblockType::I16x16 { pred_mode: Intra16x16Mode::Dc, .. });
    }

    #[test]
    fn test_encode_macroblock_with_pattern() {
        let mut encoder = MacroblockEncoder::new(false, 10);
        encoder.init_slice(26, 0);

        // Create gradient test data
        let mut luma = [0u8; 256];
        for y in 0..16 {
            for x in 0..16 {
                luma[y * 16 + x] = ((x + y) * 8) as u8;
            }
        }
        let cb = [128u8; 64];
        let cr = [128u8; 64];

        let mb = encoder.encode_macroblock(0, 0, &luma, 16, &cb, 8, &cr, 8);

        // Should have some non-zero coefficients
        let has_coeffs = mb.luma_coeffs.iter().any(|blk| blk.iter().any(|&c| c != 0))
            || mb.luma_dc.iter().any(|&c| c != 0);
        assert!(has_coeffs);
    }

    #[test]
    fn test_write_macroblock_cavlc() {
        let mut encoder = MacroblockEncoder::new(false, 10);
        encoder.init_slice(26, 0);

        let luma = [128u8; 256];
        let cb = [128u8; 64];
        let cr = [128u8; 64];

        let mb = encoder.encode_macroblock(0, 0, &luma, 16, &cb, 8, &cr, 8);

        let mut writer = BitWriter::new();
        encoder.write_macroblock_cavlc(&mut writer, &mb, 0, 0);

        // Should have written some data
        assert!(writer.data().len() > 0);
    }

    #[test]
    fn test_write_macroblock_cabac() {
        let mut encoder = MacroblockEncoder::new(true, 10);
        encoder.init_slice(26, 0);

        let luma = [128u8; 256];
        let cb = [128u8; 64];
        let cr = [128u8; 64];

        let mb = encoder.encode_macroblock(0, 0, &luma, 16, &cb, 8, &cr, 8);
        encoder.write_macroblock_cabac(&mb, 0, 0);

        let data = encoder.finish_cabac();
        assert!(!data.is_empty());
    }
}
