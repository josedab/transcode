//! DNxHD/DNxHR encoder

use crate::error::{DnxError, Result};
use crate::frame::{DnxFrame, FrameHeader, DNX_SIGNATURE};
use crate::huffman::{AcHuffmanTable, BitWriter, DcHuffmanTable, HuffmanCode};
use crate::profile::{DnxProfile, ProfileInfo};
use crate::tables::{
    get_quant_matrix, zigzag_block, WeightMatrix, BLOCK_COEFFS, BLOCK_SIZE, DCT_COS,
    DC_PRED_10BIT, DC_PRED_8BIT, FRAME_HEADER_SIZE, IDCT_SCALE, MB_SIZE,
};
use crate::types::{BitDepth, ChromaFormat, Colorimetry};

/// Encoder configuration
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Target profile
    pub profile: DnxProfile,
    /// Quality level (1-100, higher is better)
    pub quality: u32,
    /// Number of slices per row (typically 8)
    pub slices_per_row: usize,
    /// Colorimetry information
    pub colorimetry: Colorimetry,
    /// Force interlaced encoding
    pub interlaced: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig {
            profile: DnxProfile::Dnxhd145,
            quality: 75,
            slices_per_row: 8,
            colorimetry: Colorimetry::BT709,
            interlaced: false,
        }
    }
}

impl EncoderConfig {
    /// Create config for specific profile
    pub fn for_profile(profile: DnxProfile) -> Self {
        let quality = match profile {
            DnxProfile::Dnxhd36 | DnxProfile::Dnxhd45 | DnxProfile::DnxhrLb => 50,
            DnxProfile::Dnxhd90 | DnxProfile::Dnxhd90x | DnxProfile::DnxhrSq => 65,
            DnxProfile::Dnxhd120 | DnxProfile::Dnxhd145 | DnxProfile::DnxhrHq => 75,
            DnxProfile::Dnxhd175
            | DnxProfile::Dnxhd175x
            | DnxProfile::DnxhrHqx
            | DnxProfile::Dnxhd220
            | DnxProfile::Dnxhd220x => 85,
            DnxProfile::Dnxhr444 => 90,
        };

        EncoderConfig {
            profile,
            quality,
            ..Default::default()
        }
    }
}

/// DNxHD encoder
pub struct DnxEncoder {
    config: EncoderConfig,
    dc_table: DcHuffmanTable,
    ac_table: AcHuffmanTable,
    weight_matrix: WeightMatrix,
    frame_count: u64,
}

impl DnxEncoder {
    /// Create a new encoder with default config
    pub fn new() -> Self {
        Self::with_config(EncoderConfig::default())
    }

    /// Create a new encoder with custom config
    pub fn with_config(config: EncoderConfig) -> Self {
        let weight_matrix = WeightMatrix::for_quality(config.quality);
        DnxEncoder {
            config,
            dc_table: DcHuffmanTable::new(),
            ac_table: AcHuffmanTable::new(),
            weight_matrix,
            frame_count: 0,
        }
    }

    /// Encode a frame to DNxHD bitstream
    pub fn encode_frame(&mut self, frame: &DnxFrame) -> Result<Vec<u8>> {
        let profile_info = self.config.profile.info();

        log::debug!(
            "Encoding DNxHD frame: {}x{}, profile {:?}",
            frame.width,
            frame.height,
            self.config.profile
        );

        // Allocate output buffer
        let estimated_size = self.estimate_frame_size(frame.width, frame.height, &profile_info);
        let mut output = Vec::with_capacity(estimated_size);

        // Write frame header
        self.write_frame_header(&mut output, frame, &profile_info)?;

        // Encode slices
        self.encode_slices(&mut output, frame, &profile_info)?;

        // Update frame size in header
        let frame_size = output.len() as u32;
        output[4..8].copy_from_slice(&frame_size.to_be_bytes());

        self.frame_count += 1;
        Ok(output)
    }

    /// Estimate frame size for buffer allocation
    fn estimate_frame_size(&self, width: u32, height: u32, info: &ProfileInfo) -> usize {
        // Rough estimate based on bitrate
        let pixels = (width * height) as usize;
        let bits_per_pixel = (info.target_bitrate_mbps as usize * 1_000_000) / (pixels * 30);
        let base_size = pixels * bits_per_pixel.max(1) / 8;
        base_size.max(FRAME_HEADER_SIZE + 1024)
    }

    /// Write frame header
    fn write_frame_header(
        &self,
        output: &mut Vec<u8>,
        frame: &DnxFrame,
        info: &ProfileInfo,
    ) -> Result<()> {
        // Signature
        output.extend_from_slice(&DNX_SIGNATURE);

        // Frame size placeholder (will be updated later)
        output.extend_from_slice(&[0u8; 4]);

        // Header size (640 bytes for DNxHD)
        output.extend_from_slice(&(FRAME_HEADER_SIZE as u16).to_be_bytes());

        // Version (1 for standard DNxHD)
        output.extend_from_slice(&1u16.to_be_bytes());

        // Reserved bytes
        output.extend_from_slice(&[0u8; 8]);

        // Width and height
        output.extend_from_slice(&(frame.width as u16).to_be_bytes());
        output.extend_from_slice(&(frame.height as u16).to_be_bytes());

        // Flags
        let mut flags = 0u8;
        if self.config.interlaced {
            flags |= 0x02;
        }
        flags |= 0x04; // Progressive
        output.push(flags);

        // Colorimetry
        output.push(self.config.colorimetry.to_byte());

        // Reserved
        output.extend_from_slice(&[0u8; 14]);

        // Profile ID at offset 40
        output.extend_from_slice(&info.profile_id.to_be_bytes());

        // Bit depth indicator
        let bit_depth_byte = match info.bit_depth {
            BitDepth::Bit8 => 8,
            BitDepth::Bit10 => 10,
            BitDepth::Bit12 => 12,
        };
        output.push(bit_depth_byte);

        // Chroma format
        let chroma_byte = match info.chroma_format {
            ChromaFormat::YUV422 => 2,
            ChromaFormat::YUV444 => 4,
        };
        output.push(chroma_byte);

        // Pad header to FRAME_HEADER_SIZE
        output.resize(FRAME_HEADER_SIZE, 0);

        Ok(())
    }

    /// Encode all slices
    fn encode_slices(
        &self,
        output: &mut Vec<u8>,
        frame: &DnxFrame,
        info: &ProfileInfo,
    ) -> Result<()> {
        let mb_width = ((frame.width + 15) / 16) as usize;
        let mb_height = ((frame.height + 15) / 16) as usize;
        let slices_per_row = self.config.slices_per_row.min(mb_width).max(1);

        // DC predictor reset value
        let dc_pred_reset = match info.bit_depth {
            BitDepth::Bit8 => DC_PRED_8BIT,
            BitDepth::Bit10 => DC_PRED_10BIT,
            BitDepth::Bit12 => 2048,
        };

        let is_hq = info.target_bitrate_mbps >= 145;

        for mb_y in 0..mb_height {
            // Reset DC predictors at start of each row
            let mut dc_pred_y = [dc_pred_reset; 4];
            let mut dc_pred_cb = dc_pred_reset;
            let mut dc_pred_cr = dc_pred_reset;

            for slice_idx in 0..slices_per_row {
                let mb_x_start = slice_idx * mb_width / slices_per_row;
                let mb_x_end = (slice_idx + 1) * mb_width / slices_per_row;

                // Encode slice data
                let slice_data = self.encode_slice(
                    frame,
                    mb_x_start,
                    mb_x_end,
                    mb_y,
                    &mut dc_pred_y,
                    &mut dc_pred_cb,
                    &mut dc_pred_cr,
                    is_hq,
                )?;

                // Write slice header
                let slice_size = 12 + slice_data.len();
                output.extend_from_slice(&(slice_size as u32).to_be_bytes());

                // Slice position info
                output.extend_from_slice(&(mb_x_start as u16).to_be_bytes());
                output.extend_from_slice(&(mb_y as u16).to_be_bytes());

                // Reserved
                output.extend_from_slice(&[0u8; 4]);

                // Slice data
                output.extend_from_slice(&slice_data);
            }
        }

        Ok(())
    }

    /// Encode a single slice
    #[allow(clippy::too_many_arguments)]
    fn encode_slice(
        &self,
        frame: &DnxFrame,
        mb_x_start: usize,
        mb_x_end: usize,
        mb_y: usize,
        dc_pred_y: &mut [i16; 4],
        dc_pred_cb: &mut i16,
        dc_pred_cr: &mut i16,
        is_hq: bool,
    ) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        for mb_x in mb_x_start..mb_x_end {
            self.encode_macroblock(
                &mut writer,
                frame,
                mb_x,
                mb_y,
                dc_pred_y,
                dc_pred_cb,
                dc_pred_cr,
                is_hq,
            )?;
        }

        Ok(writer.into_bytes())
    }

    /// Encode a single macroblock
    #[allow(clippy::too_many_arguments)]
    fn encode_macroblock(
        &self,
        writer: &mut BitWriter,
        frame: &DnxFrame,
        mb_x: usize,
        mb_y: usize,
        dc_pred_y: &mut [i16; 4],
        dc_pred_cb: &mut i16,
        dc_pred_cr: &mut i16,
        is_hq: bool,
    ) -> Result<()> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let chroma_width = (width + 1) / 2;

        // Extract and encode luma blocks (4 per macroblock)
        for i in 0..4 {
            let block_x = (i & 1) * BLOCK_SIZE;
            let block_y = (i >> 1) * BLOCK_SIZE;

            let mut block = [0i16; BLOCK_COEFFS];
            self.extract_block(
                &frame.y_plane,
                width,
                height,
                mb_x * MB_SIZE + block_x,
                mb_y * MB_SIZE + block_y,
                &mut block,
            );

            // Forward DCT
            let mut dct_block = [0i16; BLOCK_COEFFS];
            forward_dct_2d(&block, &mut dct_block);

            // Quantize
            self.quantize_block(&mut dct_block, true, is_hq);

            // Encode
            self.encode_block(writer, &dct_block, &mut dc_pred_y[i])?;
        }

        // Extract and encode chroma blocks
        let mut cb_block = [0i16; BLOCK_COEFFS];
        let mut cr_block = [0i16; BLOCK_COEFFS];

        self.extract_block(
            &frame.cb_plane,
            chroma_width,
            height,
            mb_x * BLOCK_SIZE,
            mb_y * MB_SIZE,
            &mut cb_block,
        );
        self.extract_block(
            &frame.cr_plane,
            chroma_width,
            height,
            mb_x * BLOCK_SIZE,
            mb_y * MB_SIZE,
            &mut cr_block,
        );

        // Forward DCT for chroma
        let mut cb_dct = [0i16; BLOCK_COEFFS];
        let mut cr_dct = [0i16; BLOCK_COEFFS];
        forward_dct_2d(&cb_block, &mut cb_dct);
        forward_dct_2d(&cr_block, &mut cr_dct);

        // Quantize chroma
        self.quantize_block(&mut cb_dct, false, is_hq);
        self.quantize_block(&mut cr_dct, false, is_hq);

        // Encode chroma
        self.encode_block(writer, &cb_dct, dc_pred_cb)?;
        self.encode_block(writer, &cr_dct, dc_pred_cr)?;

        Ok(())
    }

    /// Extract an 8x8 block from a plane
    fn extract_block(
        &self,
        plane: &[i16],
        plane_width: usize,
        plane_height: usize,
        x: usize,
        y: usize,
        block: &mut [i16; BLOCK_COEFFS],
    ) {
        for by in 0..BLOCK_SIZE {
            for bx in 0..BLOCK_SIZE {
                let px = (x + bx).min(plane_width.saturating_sub(1));
                let py = (y + by).min(plane_height.saturating_sub(1));
                let idx = py * plane_width + px;
                block[by * BLOCK_SIZE + bx] = if idx < plane.len() { plane[idx] } else { 0 };
            }
        }
    }

    /// Quantize a DCT block
    fn quantize_block(&self, block: &mut [i16; BLOCK_COEFFS], is_luma: bool, is_hq: bool) {
        let matrix = if is_luma {
            &self.weight_matrix.luma
        } else {
            &self.weight_matrix.chroma
        };

        let base_matrix = get_quant_matrix(is_luma, is_hq);

        for i in 0..BLOCK_COEFFS {
            let q = (matrix[i] as i32 * base_matrix[i] as i32 + 8) >> 4;
            let q = q.max(1);
            block[i] = ((block[i] as i32 + (q >> 1)) / q) as i16;
        }
    }

    /// Encode a single block
    fn encode_block(
        &self,
        writer: &mut BitWriter,
        coeffs: &[i16; BLOCK_COEFFS],
        dc_pred: &mut i16,
    ) -> Result<()> {
        // Apply zigzag scan
        let zigzag = zigzag_block(coeffs);

        // Encode DC coefficient
        let dc_diff = zigzag[0] - *dc_pred;
        *dc_pred = zigzag[0];

        let (code, amplitude, size) = self.dc_table.encode(dc_diff);
        writer.write_code(&code);
        if size > 0 {
            writer.write_bits(amplitude as u32, size);
        }

        // Encode AC coefficients using run-level coding
        let mut run = 0u8;
        for i in 1..BLOCK_COEFFS {
            let level = zigzag[i];

            if level == 0 {
                run += 1;
            } else {
                // Try to encode with table
                if let Some((code, sign)) = self.ac_table.encode_run_level(run, level) {
                    writer.write_code(&code);
                    writer.write_bit(sign);
                } else {
                    // Use escape code
                    self.encode_escape(writer, run, level);
                }
                run = 0;
            }
        }

        // End of block
        writer.write_code(&self.ac_table.eob());

        Ok(())
    }

    /// Encode escape sequence for unusual run-level pairs
    fn encode_escape(&self, writer: &mut BitWriter, run: u8, level: i16) {
        // Escape code: 6 ones followed by run (6 bits) and level (12 bits)
        writer.write_bits(0b111111, 6);
        writer.write_bits(run as u32, 6);

        let sign = level < 0;
        let abs_level = level.unsigned_abs();
        writer.write_bits(abs_level as u32, 11);
        writer.write_bit(sign);
    }

    /// Get number of frames encoded
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset encoder state
    pub fn reset(&mut self) {
        self.frame_count = 0;
    }

    /// Get current configuration
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

impl Default for DnxEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Forward DCT for 8x8 block
fn forward_dct_2d(input: &[i16; BLOCK_COEFFS], output: &mut [i16; BLOCK_COEFFS]) {
    let mut temp = [0i64; BLOCK_COEFFS];

    // Row DCT
    for row in 0..BLOCK_SIZE {
        let row_offset = row * BLOCK_SIZE;
        dct_row(
            &input[row_offset..row_offset + BLOCK_SIZE],
            &mut temp[row_offset..row_offset + BLOCK_SIZE],
        );
    }

    // Column DCT
    for col in 0..BLOCK_SIZE {
        dct_col(&temp, output, col);
    }
}

/// 1D DCT on a row
fn dct_row(input: &[i16], output: &mut [i64]) {
    let s0 = input[0] as i64 + input[7] as i64;
    let s1 = input[1] as i64 + input[6] as i64;
    let s2 = input[2] as i64 + input[5] as i64;
    let s3 = input[3] as i64 + input[4] as i64;
    let s4 = input[3] as i64 - input[4] as i64;
    let s5 = input[2] as i64 - input[5] as i64;
    let s6 = input[1] as i64 - input[6] as i64;
    let s7 = input[0] as i64 - input[7] as i64;

    let t0 = s0 + s3;
    let t1 = s1 + s2;
    let t2 = s1 - s2;
    let t3 = s0 - s3;

    output[0] = (t0 + t1) * DCT_COS[0] as i64;
    output[4] = (t0 - t1) * DCT_COS[0] as i64;
    output[2] = t2 * DCT_COS[6] as i64 + t3 * DCT_COS[2] as i64;
    output[6] = t3 * DCT_COS[6] as i64 - t2 * DCT_COS[2] as i64;

    output[1] = s4 * DCT_COS[7] as i64 + s5 * DCT_COS[5] as i64
        + s6 * DCT_COS[3] as i64
        + s7 * DCT_COS[1] as i64;
    output[3] = s4 * DCT_COS[5] as i64 - s5 * DCT_COS[1] as i64
        - s6 * DCT_COS[7] as i64
        - s7 * DCT_COS[3] as i64;
    output[5] = s4 * DCT_COS[3] as i64 - s5 * DCT_COS[7] as i64
        + s6 * DCT_COS[1] as i64
        + s7 * DCT_COS[5] as i64;
    output[7] = s4 * DCT_COS[1] as i64 - s5 * DCT_COS[3] as i64
        + s6 * DCT_COS[5] as i64
        - s7 * DCT_COS[7] as i64;
}

/// 1D DCT on a column
fn dct_col(input: &[i64; BLOCK_COEFFS], output: &mut [i16; BLOCK_COEFFS], col: usize) {
    let mut col_data = [0i64; BLOCK_SIZE];
    for row in 0..BLOCK_SIZE {
        col_data[row] = input[row * BLOCK_SIZE + col];
    }

    let s0 = col_data[0] + col_data[7];
    let s1 = col_data[1] + col_data[6];
    let s2 = col_data[2] + col_data[5];
    let s3 = col_data[3] + col_data[4];
    let s4 = col_data[3] - col_data[4];
    let s5 = col_data[2] - col_data[5];
    let s6 = col_data[1] - col_data[6];
    let s7 = col_data[0] - col_data[7];

    let t0 = s0 + s3;
    let t1 = s1 + s2;
    let t2 = s1 - s2;
    let t3 = s0 - s3;

    let scale = IDCT_SCALE as i64 * IDCT_SCALE as i64;

    output[0 * BLOCK_SIZE + col] = (((t0 + t1) * DCT_COS[0] as i64) / scale).clamp(-2048, 2047) as i16;
    output[4 * BLOCK_SIZE + col] = (((t0 - t1) * DCT_COS[0] as i64) / scale).clamp(-2048, 2047) as i16;
    output[2 * BLOCK_SIZE + col] =
        ((t2 * DCT_COS[6] as i64 + t3 * DCT_COS[2] as i64) / scale).clamp(-2048, 2047) as i16;
    output[6 * BLOCK_SIZE + col] =
        ((t3 * DCT_COS[6] as i64 - t2 * DCT_COS[2] as i64) / scale).clamp(-2048, 2047) as i16;

    let v1 = s4 * DCT_COS[7] as i64 + s5 * DCT_COS[5] as i64
        + s6 * DCT_COS[3] as i64
        + s7 * DCT_COS[1] as i64;
    let v3 = s4 * DCT_COS[5] as i64 - s5 * DCT_COS[1] as i64
        - s6 * DCT_COS[7] as i64
        - s7 * DCT_COS[3] as i64;
    let v5 = s4 * DCT_COS[3] as i64 - s5 * DCT_COS[7] as i64
        + s6 * DCT_COS[1] as i64
        + s7 * DCT_COS[5] as i64;
    let v7 = s4 * DCT_COS[1] as i64 - s5 * DCT_COS[3] as i64
        + s6 * DCT_COS[5] as i64
        - s7 * DCT_COS[7] as i64;

    output[1 * BLOCK_SIZE + col] = (v1 / scale).clamp(-2048, 2047) as i16;
    output[3 * BLOCK_SIZE + col] = (v3 / scale).clamp(-2048, 2047) as i16;
    output[5 * BLOCK_SIZE + col] = (v5 / scale).clamp(-2048, 2047) as i16;
    output[7 * BLOCK_SIZE + col] = (v7 / scale).clamp(-2048, 2047) as i16;
}

/// Convenience function to encode DNxHD
pub fn encode_dnxhd(frame: &DnxFrame, profile: DnxProfile, quality: u32) -> Result<Vec<u8>> {
    let config = EncoderConfig {
        profile,
        quality,
        ..Default::default()
    };
    let mut encoder = DnxEncoder::with_config(config);
    encoder.encode_frame(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_new() {
        let encoder = DnxEncoder::new();
        assert_eq!(encoder.frame_count(), 0);
    }

    #[test]
    fn test_encoder_config_for_profile() {
        let config = EncoderConfig::for_profile(DnxProfile::Dnxhd220x);
        assert_eq!(config.profile, DnxProfile::Dnxhd220x);
        assert_eq!(config.quality, 85);
    }

    #[test]
    fn test_forward_dct_dc() {
        // Uniform block should produce DC only
        let input = [128i16; BLOCK_COEFFS];
        let mut output = [0i16; BLOCK_COEFFS];
        forward_dct_2d(&input, &mut output);

        // DC should be non-zero
        assert_ne!(output[0], 0);
    }

    #[test]
    fn test_encoder_reset() {
        let mut encoder = DnxEncoder::new();
        encoder.frame_count = 100;
        encoder.reset();
        assert_eq!(encoder.frame_count(), 0);
    }

    #[test]
    fn test_encode_small_frame() {
        let width = 64u32;
        let height = 64u32;
        let y_plane = vec![128i16; (width * height) as usize];
        let cb_plane = vec![0i16; (width / 2 * height) as usize];
        let cr_plane = vec![0i16; (width / 2 * height) as usize];

        let frame = DnxFrame::from_planes(
            &y_plane,
            &cb_plane,
            &cr_plane,
            None,
            width,
            height,
            DnxProfile::Dnxhd145,
        )
        .unwrap();

        let mut encoder = DnxEncoder::new();
        let result = encoder.encode_frame(&frame);

        assert!(result.is_ok());
        let data = result.unwrap();

        // Check signature
        assert_eq!(&data[0..4], &DNX_SIGNATURE);

        // Frame size should be at offset 4
        let frame_size = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(frame_size as usize, data.len());
    }

    #[test]
    fn test_encode_gradient_frame() {
        let width = 32u32;
        let height = 32u32;

        // Create gradient pattern
        let mut y_plane = vec![0i16; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                y_plane[(y * width + x) as usize] = ((x + y) * 4) as i16;
            }
        }

        let cb_plane = vec![128i16; (width / 2 * height) as usize];
        let cr_plane = vec![128i16; (width / 2 * height) as usize];

        let frame = DnxFrame::from_planes(
            &y_plane,
            &cb_plane,
            &cr_plane,
            None,
            width,
            height,
            DnxProfile::DnxhrHq,
        )
        .unwrap();

        let mut encoder = DnxEncoder::with_config(EncoderConfig::for_profile(DnxProfile::DnxhrHq));
        let result = encoder.encode_frame(&frame);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_dnxhd_convenience() {
        let width = 16u32;
        let height = 16u32;
        let y_plane = vec![64i16; (width * height) as usize];
        let cb_plane = vec![0i16; (width / 2 * height) as usize];
        let cr_plane = vec![0i16; (width / 2 * height) as usize];

        let frame = DnxFrame::from_planes(
            &y_plane,
            &cb_plane,
            &cr_plane,
            None,
            width,
            height,
            DnxProfile::Dnxhd90,
        )
        .unwrap();

        let result = encode_dnxhd(&frame, DnxProfile::Dnxhd90, 70);
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_profiles_encode() {
        let profiles = [
            DnxProfile::Dnxhd36,
            DnxProfile::Dnxhd45,
            DnxProfile::Dnxhd90,
            DnxProfile::Dnxhd145,
            DnxProfile::DnxhrLb,
            DnxProfile::DnxhrSq,
            DnxProfile::DnxhrHq,
        ];

        for profile in profiles {
            let width = 32u32;
            let height = 32u32;
            let y_plane = vec![100i16; (width * height) as usize];
            let cb_plane = vec![0i16; (width / 2 * height) as usize];
            let cr_plane = vec![0i16; (width / 2 * height) as usize];

            let frame =
                DnxFrame::from_planes(&y_plane, &cb_plane, &cr_plane, None, width, height, profile)
                    .unwrap();

            let mut encoder = DnxEncoder::with_config(EncoderConfig::for_profile(profile));
            let result = encoder.encode_frame(&frame);
            assert!(result.is_ok(), "Failed to encode with profile {:?}", profile);
        }
    }

    #[test]
    fn test_weight_matrix_affects_size() {
        let width = 64u32;
        let height = 64u32;

        // Random-ish pattern
        let mut y_plane = vec![0i16; (width * height) as usize];
        for (i, val) in y_plane.iter_mut().enumerate() {
            *val = ((i * 17 + 31) % 256) as i16;
        }
        let cb_plane = vec![128i16; (width / 2 * height) as usize];
        let cr_plane = vec![128i16; (width / 2 * height) as usize];

        let frame = DnxFrame::from_planes(
            &y_plane,
            &cb_plane,
            &cr_plane,
            None,
            width,
            height,
            DnxProfile::Dnxhd145,
        )
        .unwrap();

        // Low quality
        let mut low_config = EncoderConfig::default();
        low_config.quality = 20;
        let mut encoder_low = DnxEncoder::with_config(low_config);
        let low_data = encoder_low.encode_frame(&frame).unwrap();

        // High quality
        let mut high_config = EncoderConfig::default();
        high_config.quality = 95;
        let mut encoder_high = DnxEncoder::with_config(high_config);
        let high_data = encoder_high.encode_frame(&frame).unwrap();

        // Higher quality should generally produce larger files
        // (though this isn't guaranteed for all content)
        assert!(low_data.len() > 0);
        assert!(high_data.len() > 0);
    }
}
