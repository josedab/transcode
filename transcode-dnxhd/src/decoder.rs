//! DNxHD/DNxHR decoder

use crate::error::{DnxError, Result};
use crate::frame::{DnxFrame, FrameHeader, DNX_SIGNATURE};
use crate::huffman::{AcHuffmanTable, BitReader, DcHuffmanTable};
use crate::profile::DnxProfile;
use crate::tables::{
    inverse_zigzag_block, BLOCK_COEFFS, BLOCK_SIZE, DCT_COS, DC_PRED_10BIT, DC_PRED_8BIT,
    FRAME_HEADER_SIZE, IDCT_SCALE, MB_SIZE,
};
use crate::types::BitDepth;

/// DNxHD decoder configuration
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Enable threading for slice decoding
    pub threaded: bool,
    /// Number of threads (0 = auto)
    pub thread_count: usize,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        DecoderConfig {
            threaded: false,
            thread_count: 0,
        }
    }
}

/// DNxHD decoder
pub struct DnxDecoder {
    config: DecoderConfig,
    dc_table: DcHuffmanTable,
    ac_table: AcHuffmanTable,
    frame_count: u64,
}

impl Default for DnxDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl DnxDecoder {
    /// Create a new decoder with default config
    pub fn new() -> Self {
        Self::with_config(DecoderConfig::default())
    }

    /// Create a new decoder with custom config
    pub fn with_config(config: DecoderConfig) -> Self {
        DnxDecoder {
            config,
            dc_table: DcHuffmanTable::new(),
            ac_table: AcHuffmanTable::new(),
            frame_count: 0,
        }
    }

    /// Decode a single frame from raw DNxHD data
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<DnxFrame> {
        // Parse frame header
        let header = FrameHeader::parse(data)?;

        log::debug!(
            "Decoding DNxHD frame: {}x{}, profile {:?}",
            header.width,
            header.height,
            header.profile
        );

        // Create output frame
        let mut frame = DnxFrame::new(header.clone());

        // Decode slices
        self.decode_slices(data, &header, &mut frame)?;

        self.frame_count += 1;
        Ok(frame)
    }

    /// Decode all slices in the frame
    fn decode_slices(&self, data: &[u8], header: &FrameHeader, frame: &mut DnxFrame) -> Result<()> {
        // Skip header to get to slice data
        let slice_data = &data[FRAME_HEADER_SIZE..];

        // Calculate slice parameters
        let mb_width = header.mb_width as usize;
        let mb_height = header.mb_height as usize;
        let slices_per_row = 8.min(mb_width);

        // DC predictor reset value
        let dc_pred_reset = match header.bit_depth {
            BitDepth::Bit8 => DC_PRED_8BIT,
            BitDepth::Bit10 => DC_PRED_10BIT,
            BitDepth::Bit12 => 2048,
        };

        // Decode each slice row
        let mut slice_offset = 0;
        for mb_y in 0..mb_height {
            // Reset DC predictors at start of each row
            let mut dc_pred_y = [dc_pred_reset; 4];
            let mut dc_pred_cb = dc_pred_reset;
            let mut dc_pred_cr = dc_pred_reset;

            for slice_idx in 0..slices_per_row {
                let mb_x_start = slice_idx * mb_width / slices_per_row;
                let mb_x_end = (slice_idx + 1) * mb_width / slices_per_row;

                // Read slice header
                if slice_offset + 12 > slice_data.len() {
                    break;
                }

                let slice_size =
                    u32::from_be_bytes([
                        slice_data[slice_offset],
                        slice_data[slice_offset + 1],
                        slice_data[slice_offset + 2],
                        slice_data[slice_offset + 3],
                    ]) as usize;

                if slice_offset + slice_size > slice_data.len() {
                    log::warn!("Slice truncated, skipping");
                    break;
                }

                // Decode macroblocks in this slice
                let mut reader = BitReader::new(&slice_data[slice_offset + 12..slice_offset + slice_size]);

                for mb_x in mb_x_start..mb_x_end {
                    self.decode_macroblock(
                        &mut reader,
                        frame,
                        mb_x,
                        mb_y,
                        &mut dc_pred_y,
                        &mut dc_pred_cb,
                        &mut dc_pred_cr,
                        header,
                    )?;
                }

                slice_offset += slice_size;
            }
        }

        Ok(())
    }

    /// Decode a single macroblock
    #[allow(clippy::too_many_arguments)]
    fn decode_macroblock(
        &self,
        reader: &mut BitReader,
        frame: &mut DnxFrame,
        mb_x: usize,
        mb_y: usize,
        dc_pred_y: &mut [i16; 4],
        dc_pred_cb: &mut i16,
        dc_pred_cr: &mut i16,
        header: &FrameHeader,
    ) -> Result<()> {
        // Each macroblock has 4 luma blocks (2x2) and 2 chroma blocks (for 4:2:2)
        let mut y_blocks = [[0i16; BLOCK_COEFFS]; 4];
        let mut cb_block = [0i16; BLOCK_COEFFS];
        let mut cr_block = [0i16; BLOCK_COEFFS];

        // Decode luma blocks
        for i in 0..4 {
            self.decode_block(reader, &mut y_blocks[i], &mut dc_pred_y[i])?;
        }

        // Decode chroma blocks
        self.decode_block(reader, &mut cb_block, dc_pred_cb)?;
        self.decode_block(reader, &mut cr_block, dc_pred_cr)?;

        // Apply inverse DCT and store in frame
        self.store_macroblock(frame, mb_x, mb_y, &y_blocks, &cb_block, &cr_block, header);

        Ok(())
    }

    /// Decode a single 8x8 block
    fn decode_block(
        &self,
        reader: &mut BitReader,
        coeffs: &mut [i16; BLOCK_COEFFS],
        dc_pred: &mut i16,
    ) -> Result<()> {
        // Decode DC coefficient
        let dc_diff = self.dc_table.decode(reader)?;
        coeffs[0] = *dc_pred + dc_diff;
        *dc_pred = coeffs[0];

        // Decode AC coefficients
        let mut pos = 1;
        while pos < BLOCK_COEFFS {
            match self.ac_table.decode_run_level(reader) {
                Ok(Some((run, level))) => {
                    pos += run as usize;
                    if pos >= BLOCK_COEFFS {
                        break;
                    }
                    coeffs[pos] = level;
                    pos += 1;
                }
                Ok(None) => {
                    // End of block
                    break;
                }
                Err(_) => {
                    // Unknown code, skip rest of block
                    break;
                }
            }
        }

        // Inverse zigzag
        *coeffs = inverse_zigzag_block(coeffs);

        Ok(())
    }

    /// Apply IDCT and store macroblock in frame
    fn store_macroblock(
        &self,
        frame: &mut DnxFrame,
        mb_x: usize,
        mb_y: usize,
        y_blocks: &[[i16; BLOCK_COEFFS]; 4],
        cb_block: &[i16; BLOCK_COEFFS],
        cr_block: &[i16; BLOCK_COEFFS],
        header: &FrameHeader,
    ) {
        let width = header.width as usize;
        let chroma_width = (width + 1) / 2;

        // Process and store luma blocks
        for (i, block) in y_blocks.iter().enumerate() {
            let block_x = (i & 1) * BLOCK_SIZE;
            let block_y = (i >> 1) * BLOCK_SIZE;

            let mut idct_block = [0i16; BLOCK_COEFFS];
            inverse_dct_2d(block, &mut idct_block);

            for by in 0..BLOCK_SIZE {
                for bx in 0..BLOCK_SIZE {
                    let px = mb_x * MB_SIZE + block_x + bx;
                    let py = mb_y * MB_SIZE + block_y + by;

                    if px < width && py < header.height as usize {
                        let idx = py * width + px;
                        frame.y_plane[idx] = idct_block[by * BLOCK_SIZE + bx];
                    }
                }
            }
        }

        // Process and store chroma blocks (4:2:2 - full height, half width)
        let mut cb_idct = [0i16; BLOCK_COEFFS];
        let mut cr_idct = [0i16; BLOCK_COEFFS];
        inverse_dct_2d(cb_block, &mut cb_idct);
        inverse_dct_2d(cr_block, &mut cr_idct);

        // For 4:2:2, we have 8x16 chroma per macroblock (8 wide, 16 tall)
        for by in 0..MB_SIZE {
            for bx in 0..BLOCK_SIZE {
                let px = mb_x * BLOCK_SIZE + bx;
                let py = mb_y * MB_SIZE + by;

                if px < chroma_width && py < header.height as usize {
                    let idx = py * chroma_width + px;
                    // Use top or bottom half of block based on row
                    let block_idx = (by % BLOCK_SIZE) * BLOCK_SIZE + bx;
                    frame.cb_plane[idx] = cb_idct[block_idx];
                    frame.cr_plane[idx] = cr_idct[block_idx];
                }
            }
        }
    }

    /// Get number of frames decoded
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset decoder state
    pub fn reset(&mut self) {
        self.frame_count = 0;
    }
}

/// Inverse DCT for 8x8 block
fn inverse_dct_2d(input: &[i16; BLOCK_COEFFS], output: &mut [i16; BLOCK_COEFFS]) {
    // Temporary buffer for intermediate results
    let mut temp = [0i64; BLOCK_COEFFS];

    // Row IDCT
    for row in 0..BLOCK_SIZE {
        let row_offset = row * BLOCK_SIZE;
        idct_row(&input[row_offset..row_offset + BLOCK_SIZE], &mut temp[row_offset..row_offset + BLOCK_SIZE]);
    }

    // Column IDCT
    for col in 0..BLOCK_SIZE {
        idct_col(&temp, output, col);
    }
}

/// 1D IDCT on a row
fn idct_row(input: &[i16], output: &mut [i64]) {
    // Scaled IDCT using integer arithmetic
    let s0 = input[0] as i64 * DCT_COS[0] as i64;
    let s1 = input[1] as i64;
    let s2 = input[2] as i64;
    let s3 = input[3] as i64;
    let s4 = input[4] as i64 * DCT_COS[0] as i64;
    let s5 = input[5] as i64;
    let s6 = input[6] as i64;
    let s7 = input[7] as i64;

    // Butterfly operations
    let t0 = s0 + s4;
    let t1 = s0 - s4;

    let t2 = s2 * DCT_COS[2] as i64 + s6 * DCT_COS[6] as i64;
    let t3 = s2 * DCT_COS[6] as i64 - s6 * DCT_COS[2] as i64;

    let t4 = s1 * DCT_COS[1] as i64 + s7 * DCT_COS[7] as i64;
    let t5 = s1 * DCT_COS[7] as i64 - s7 * DCT_COS[1] as i64;
    let t6 = s3 * DCT_COS[3] as i64 + s5 * DCT_COS[5] as i64;
    let t7 = s3 * DCT_COS[5] as i64 - s5 * DCT_COS[3] as i64;

    // Combine stages
    let u0 = t0 + t2;
    let u1 = t1 + t3;
    let u2 = t1 - t3;
    let u3 = t0 - t2;

    let u4 = t4 + t6;
    let u5 = t5 + t7;
    let u6 = t4 - t6;
    let u7 = t5 - t7;

    // Output with scaling
    output[0] = (u0 + u4) >> 3;
    output[1] = (u1 + u5) >> 3;
    output[2] = (u2 + u6) >> 3;
    output[3] = (u3 + u7) >> 3;
    output[4] = (u3 - u7) >> 3;
    output[5] = (u2 - u6) >> 3;
    output[6] = (u1 - u5) >> 3;
    output[7] = (u0 - u4) >> 3;
}

/// 1D IDCT on a column
fn idct_col(input: &[i64; BLOCK_COEFFS], output: &mut [i16; BLOCK_COEFFS], col: usize) {
    // Extract column values
    let mut col_data = [0i64; BLOCK_SIZE];
    for row in 0..BLOCK_SIZE {
        col_data[row] = input[row * BLOCK_SIZE + col];
    }

    let s0 = col_data[0] * DCT_COS[0] as i64;
    let s1 = col_data[1];
    let s2 = col_data[2];
    let s3 = col_data[3];
    let s4 = col_data[4] * DCT_COS[0] as i64;
    let s5 = col_data[5];
    let s6 = col_data[6];
    let s7 = col_data[7];

    let t0 = s0 + s4;
    let t1 = s0 - s4;

    let t2 = s2 * DCT_COS[2] as i64 + s6 * DCT_COS[6] as i64;
    let t3 = s2 * DCT_COS[6] as i64 - s6 * DCT_COS[2] as i64;

    let t4 = s1 * DCT_COS[1] as i64 + s7 * DCT_COS[7] as i64;
    let t5 = s1 * DCT_COS[7] as i64 - s7 * DCT_COS[1] as i64;
    let t6 = s3 * DCT_COS[3] as i64 + s5 * DCT_COS[5] as i64;
    let t7 = s3 * DCT_COS[5] as i64 - s5 * DCT_COS[3] as i64;

    let u0 = t0 + t2;
    let u1 = t1 + t3;
    let u2 = t1 - t3;
    let u3 = t0 - t2;

    let u4 = t4 + t6;
    let u5 = t5 + t7;
    let u6 = t4 - t6;
    let u7 = t5 - t7;

    // Output with scaling and range clamp
    let scale = IDCT_SCALE as i64 * 8;
    output[0 * BLOCK_SIZE + col] = ((u0 + u4) / scale).clamp(-2048, 2047) as i16;
    output[1 * BLOCK_SIZE + col] = ((u1 + u5) / scale).clamp(-2048, 2047) as i16;
    output[2 * BLOCK_SIZE + col] = ((u2 + u6) / scale).clamp(-2048, 2047) as i16;
    output[3 * BLOCK_SIZE + col] = ((u3 + u7) / scale).clamp(-2048, 2047) as i16;
    output[4 * BLOCK_SIZE + col] = ((u3 - u7) / scale).clamp(-2048, 2047) as i16;
    output[5 * BLOCK_SIZE + col] = ((u2 - u6) / scale).clamp(-2048, 2047) as i16;
    output[6 * BLOCK_SIZE + col] = ((u1 - u5) / scale).clamp(-2048, 2047) as i16;
    output[7 * BLOCK_SIZE + col] = ((u0 - u4) / scale).clamp(-2048, 2047) as i16;
}

/// Probe data to check if it's DNxHD
pub fn probe_dnxhd(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    data[0..4] == DNX_SIGNATURE
}

/// Get profile from DNxHD data
pub fn get_profile(data: &[u8]) -> Option<DnxProfile> {
    if data.len() < 44 {
        return None;
    }
    let profile_id = u32::from_be_bytes([data[40], data[41], data[42], data[43]]);
    DnxProfile::from_id(profile_id)
}

/// Get dimensions from DNxHD data
pub fn get_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 24 {
        return None;
    }
    let width = u16::from_be_bytes([data[20], data[21]]) as u32;
    let height = u16::from_be_bytes([data[22], data[23]]) as u32;
    Some((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_dnxhd() {
        assert!(probe_dnxhd(&[0x00, 0x00, 0x02, 0x80, 0x00]));
        assert!(!probe_dnxhd(&[0x00, 0x00, 0x00, 0x00]));
        assert!(!probe_dnxhd(&[0x00, 0x00]));
    }

    #[test]
    fn test_decoder_new() {
        let decoder = DnxDecoder::new();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_idct_dc_only() {
        // DC-only block should produce uniform output
        let mut input = [0i16; BLOCK_COEFFS];
        input[0] = 1024;

        let mut output = [0i16; BLOCK_COEFFS];
        inverse_dct_2d(&input, &mut output);

        // All values should be similar (DC contribution)
        let first = output[0];
        for val in &output[1..] {
            assert!((val - first).abs() < 10, "DC-only should be uniform");
        }
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = DnxDecoder::new();
        decoder.frame_count = 100;
        decoder.reset();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_get_dimensions_insufficient_data() {
        assert!(get_dimensions(&[0; 10]).is_none());
    }

    #[test]
    fn test_get_profile_insufficient_data() {
        assert!(get_profile(&[0; 10]).is_none());
    }
}
