//! ProRes encoder implementation
//!
//! Implements encoding for all ProRes profiles:
//! - ProRes 422 Proxy, LT, Standard, HQ
//! - ProRes 4444, 4444 XQ (with alpha support)

use byteorder::{BigEndian, WriteBytesExt};
use std::io::Write;

use crate::error::{ProResError, Result};
use crate::frame::ProResFrame;
use crate::tables::{zigzag, COS_TABLE, DEFAULT_CHROMA_QUANT, DEFAULT_LUMA_QUANT, IDCT_SCALE};
use crate::types::{BitDepth, ChromaFormat, ColorPrimaries, InterlaceMode, MatrixCoefficients, ProResProfile, TransferCharacteristic};

/// ProRes encoder configuration
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Target profile
    pub profile: ProResProfile,
    /// Bit depth (10 or 12)
    pub bit_depth: BitDepth,
    /// Interlace mode
    pub interlace_mode: InterlaceMode,
    /// Color primaries
    pub color_primaries: ColorPrimaries,
    /// Transfer characteristic
    pub transfer_characteristic: TransferCharacteristic,
    /// Matrix coefficients
    pub matrix_coefficients: MatrixCoefficients,
    /// Include alpha channel (4444 profiles only)
    pub include_alpha: bool,
    /// Quality factor (1-100, affects quantization)
    pub quality: u8,
    /// Number of slices per row
    pub slices_per_row: u8,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            profile: ProResProfile::HQ,
            bit_depth: BitDepth::Bit10,
            interlace_mode: InterlaceMode::Progressive,
            color_primaries: ColorPrimaries::BT709,
            transfer_characteristic: TransferCharacteristic::BT709,
            matrix_coefficients: MatrixCoefficients::BT709,
            include_alpha: false,
            quality: 85,
            slices_per_row: 8,
        }
    }
}

impl EncoderConfig {
    /// Create a new encoder configuration with specified profile
    pub fn new(profile: ProResProfile) -> Self {
        let bit_depth = if profile == ProResProfile::P4444XQ {
            BitDepth::Bit12
        } else {
            BitDepth::Bit10
        };
        Self {
            profile,
            bit_depth,
            ..Default::default()
        }
    }

    /// Set quality (1-100)
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = quality.clamp(1, 100);
        self
    }

    /// Set bit depth
    pub fn bit_depth(mut self, bit_depth: BitDepth) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Include alpha channel
    pub fn with_alpha(mut self, include_alpha: bool) -> Self {
        self.include_alpha = include_alpha && self.profile.supports_alpha();
        self
    }

    /// Get chroma format based on profile
    pub fn chroma_format(&self) -> ChromaFormat {
        if self.profile.is_444() {
            ChromaFormat::YUV444
        } else {
            ChromaFormat::YUV422
        }
    }
}

/// ProRes video encoder
pub struct ProResEncoder {
    config: EncoderConfig,
    /// DC Huffman encoder table
    dc_table: DcHuffmanEncoder,
    /// AC Huffman encoder table
    ac_table: AcHuffmanEncoder,
    /// Frame counter
    frame_count: u64,
}

impl Default for ProResEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProResEncoder {
    /// Create a new ProRes encoder with default configuration (HQ profile)
    pub fn new() -> Self {
        Self::with_config(EncoderConfig::default())
    }

    /// Create a new ProRes encoder with specified configuration
    pub fn with_config(config: EncoderConfig) -> Self {
        ProResEncoder {
            config,
            dc_table: DcHuffmanEncoder::new(),
            ac_table: AcHuffmanEncoder::new(),
            frame_count: 0,
        }
    }

    /// Get the encoder configuration
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// Get the number of frames encoded
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Encode a ProRes frame
    pub fn encode_frame(&mut self, frame: &ProResFrame) -> Result<Vec<u8>> {
        // Validate frame
        self.validate_frame(frame)?;

        // Build quantization matrices based on quality
        let (luma_quant, chroma_quant) = self.build_quant_matrices();

        // Encode frame
        let mut output = Vec::new();

        // Write frame header
        self.write_frame_header(&mut output, frame, &luma_quant, &chroma_quant)?;

        // Encode slices
        self.encode_slices(&mut output, frame, &luma_quant, &chroma_quant)?;

        // Update frame size in header
        let frame_size = output.len() as u32;
        output[0..4].copy_from_slice(&frame_size.to_be_bytes());

        self.frame_count += 1;

        Ok(output)
    }

    /// Encode a frame from raw planar data
    pub fn encode_planar(
        &mut self,
        y_plane: &[i16],
        cb_plane: &[i16],
        cr_plane: &[i16],
        alpha_plane: Option<&[u16]>,
        width: u16,
        height: u16,
    ) -> Result<Vec<u8>> {
        // Build a ProResFrame from planar data
        let frame = ProResFrame::from_planes(
            y_plane,
            cb_plane,
            cr_plane,
            alpha_plane,
            width,
            height,
            self.config.profile,
            self.config.bit_depth,
        )?;

        self.encode_frame(&frame)
    }

    /// Validate frame before encoding
    fn validate_frame(&self, frame: &ProResFrame) -> Result<()> {
        if frame.width == 0 || frame.height == 0 {
            return Err(ProResError::InvalidHeader("Zero frame dimensions".into()));
        }

        if frame.width > 8192 || frame.height > 8192 {
            return Err(ProResError::InvalidHeader(format!(
                "Frame dimensions too large: {}x{}",
                frame.width, frame.height
            )));
        }

        // Check plane sizes
        let expected_y_size = (frame.y_stride as usize) * (frame.height as usize);
        if frame.y_plane.len() < expected_y_size {
            return Err(ProResError::InvalidHeader("Y plane too small".into()));
        }

        Ok(())
    }

    /// Build quantization matrices based on quality setting
    fn build_quant_matrices(&self) -> ([u8; 64], [u8; 64]) {
        let scale = self.quality_to_scale(self.config.quality);

        let mut luma_quant = [0u8; 64];
        let mut chroma_quant = [0u8; 64];

        for i in 0..64 {
            luma_quant[i] = ((DEFAULT_LUMA_QUANT[i] as u32 * scale) / 100).clamp(1, 255) as u8;
            chroma_quant[i] = ((DEFAULT_CHROMA_QUANT[i] as u32 * scale) / 100).clamp(1, 255) as u8;
        }

        (luma_quant, chroma_quant)
    }

    /// Convert quality (1-100) to quantization scale
    fn quality_to_scale(&self, quality: u8) -> u32 {
        if quality >= 90 {
            // Very high quality: scale down quantization
            50 + (100 - quality as u32) * 2
        } else if quality >= 50 {
            // Normal quality range
            100 + (90 - quality as u32) * 2
        } else {
            // Lower quality: scale up quantization
            200 + (50 - quality as u32) * 4
        }
    }

    /// Write frame header
    fn write_frame_header(
        &self,
        output: &mut Vec<u8>,
        frame: &ProResFrame,
        luma_quant: &[u8; 64],
        chroma_quant: &[u8; 64],
    ) -> Result<()> {
        // Placeholder for frame size (will be updated later)
        output.write_u32::<BigEndian>(0)?;

        // Frame signature "icpf"
        output.write_all(b"icpf")?;

        // Header size (fixed at 148 bytes for now)
        output.write_u16::<BigEndian>(148)?;

        // Reserved
        output.write_u16::<BigEndian>(0)?;

        // FourCC profile code
        output.write_all(self.config.profile.fourcc())?;

        // Frame dimensions
        output.write_u16::<BigEndian>(frame.width as u16)?;
        output.write_u16::<BigEndian>(frame.height as u16)?;

        // Frame flags
        let chroma_flag = if self.config.chroma_format() == ChromaFormat::YUV444 {
            0x02
        } else {
            0x00
        };
        let interlace_flag = match self.config.interlace_mode {
            InterlaceMode::Progressive => 0x00,
            InterlaceMode::InterlacedTFF => 0x01,
            InterlaceMode::InterlacedBFF => 0x03,
        };
        output.write_u8(chroma_flag | interlace_flag)?;

        // Reserved
        output.write_u8(0)?;

        // Color info
        output.write_u8(self.color_primaries_code())?;
        output.write_u8(self.transfer_characteristic_code())?;
        output.write_u8(self.matrix_coefficients_code())?;

        // Alpha info
        let alpha_info = if self.config.include_alpha { 0x02 } else { 0x00 };
        output.write_u8(alpha_info)?;

        // Reserved
        output.write_u16::<BigEndian>(0)?;

        // Bit depth code
        let bit_depth_code = match self.config.bit_depth {
            BitDepth::Bit10 => 0x02,
            BitDepth::Bit12 => 0x03,
        };
        output.write_u8(bit_depth_code)?;

        // Reserved
        output.write_u8(0)?;

        // Slice info
        let mb_width = frame.width.div_ceil(16);
        let mb_height = frame.height.div_ceil(16);
        let slices_per_row = self.config.slices_per_row;
        let slice_rows = mb_height as u8;

        output.write_u8(slices_per_row)?;
        output.write_u8(slice_rows)?;

        // Reserved
        output.write_u16::<BigEndian>(0)?;

        // Write quantization matrices
        output.write_all(luma_quant)?;
        output.write_all(chroma_quant)?;

        // Calculate and write slice offsets (placeholder)
        let num_slices = (slices_per_row as u32) * (slice_rows as u32);
        let _mb_per_slice = mb_width.div_ceil(slices_per_row as u32);

        // Slice offset table placeholder (will be filled during slice encoding)
        for _ in 0..num_slices {
            output.write_u16::<BigEndian>(0)?;
        }

        Ok(())
    }

    /// Encode all slices in the frame
    fn encode_slices(
        &mut self,
        output: &mut Vec<u8>,
        frame: &ProResFrame,
        luma_quant: &[u8; 64],
        chroma_quant: &[u8; 64],
    ) -> Result<()> {
        let mb_width = frame.width.div_ceil(16);
        let mb_height = frame.height.div_ceil(16);

        // Limit slices_per_row to actual macroblock width
        let slices_per_row = (self.config.slices_per_row as u32).min(mb_width).max(1);
        let slice_mb_width = mb_width.div_ceil(slices_per_row);

        let is_444 = self.config.chroma_format() == ChromaFormat::YUV444;

        // Calculate header size and slice offset table position
        let header_size = 148;
        let num_slices = slices_per_row * mb_height;
        let slice_offset_table_pos = header_size - 4; // Position before slice offset table

        let mut slice_offsets = Vec::with_capacity(num_slices as usize);
        let mut slice_data = Vec::new();

        // Encode each slice
        for slice_row in 0..mb_height {
            for slice_col in 0..slices_per_row {
                let slice_start = slice_data.len();

                // Calculate macroblock range for this slice
                let slice_x = slice_col * slice_mb_width;

                // Skip if we've gone past the macroblock width
                if slice_x >= mb_width {
                    continue;
                }

                let actual_slice_width = if slice_col == slices_per_row - 1 {
                    mb_width.saturating_sub(slice_x)
                } else {
                    slice_mb_width.min(mb_width - slice_x)
                };

                // Encode slice
                self.encode_slice(
                    &mut slice_data,
                    frame,
                    slice_x,
                    slice_row,
                    actual_slice_width,
                    luma_quant,
                    chroma_quant,
                    is_444,
                )?;

                slice_offsets.push((output.len() + slice_start) as u16);
            }
        }

        // Update slice offset table in header
        let offset_table_start = slice_offset_table_pos as usize;
        for (i, offset) in slice_offsets.iter().enumerate() {
            let pos = offset_table_start + i * 2;
            if pos + 1 < output.len() {
                output[pos..pos + 2].copy_from_slice(&offset.to_be_bytes());
            }
        }

        // Append slice data
        output.extend_from_slice(&slice_data);

        Ok(())
    }

    /// Encode a single slice
    fn encode_slice(
        &mut self,
        output: &mut Vec<u8>,
        frame: &ProResFrame,
        slice_x: u32,
        slice_y: u32,
        slice_mb_width: u32,
        luma_quant: &[u8; 64],
        chroma_quant: &[u8; 64],
        is_444: bool,
    ) -> Result<()> {
        let mut bitwriter = BitstreamWriter::new();

        // Determine quantization scale based on profile
        let qscale = self.config.profile.quant_scale();
        let qscale_byte = (qscale << 4) | qscale;

        // Write slice header
        let header_size = if is_444 { 8 } else { 2 };
        output.push(header_size);
        output.push(qscale_byte);

        if is_444 {
            // Placeholder for component sizes (will update later)
            let size_placeholder_pos = output.len();
            output.extend_from_slice(&[0u8; 6]);

            let _ = size_placeholder_pos; // We'll update these after encoding
        }

        // Reset DC prediction
        let mut last_dc = [0i16; 4];

        // Encode macroblocks in this slice
        for mb_x in 0..slice_mb_width {
            let mb_px_x = ((slice_x + mb_x) * 16) as usize;
            let mb_px_y = (slice_y * 16) as usize;

            // Encode Y blocks (4 blocks per macroblock)
            for block_idx in 0..4 {
                let block_x = mb_px_x + (block_idx & 1) * 8;
                let block_y = mb_px_y + (block_idx >> 1) * 8;

                self.encode_block(
                    &mut bitwriter,
                    &frame.y_plane,
                    frame.y_stride as usize,
                    block_x,
                    block_y,
                    luma_quant,
                    &mut last_dc[0],
                    self.config.bit_depth,
                )?;
            }

            // Encode Cb/Cr blocks
            let chroma_blocks = if is_444 { 4 } else { 2 };

            for block_idx in 0..chroma_blocks {
                let (block_x, block_y) = if is_444 {
                    (
                        mb_px_x + (block_idx & 1) * 8,
                        mb_px_y + (block_idx >> 1) * 8,
                    )
                } else {
                    (mb_px_x / 2 + (block_idx & 1) * 8, mb_px_y)
                };

                self.encode_block(
                    &mut bitwriter,
                    &frame.cb_plane,
                    frame.chroma_stride as usize,
                    block_x,
                    block_y,
                    chroma_quant,
                    &mut last_dc[1],
                    self.config.bit_depth,
                )?;
            }

            for block_idx in 0..chroma_blocks {
                let (block_x, block_y) = if is_444 {
                    (
                        mb_px_x + (block_idx & 1) * 8,
                        mb_px_y + (block_idx >> 1) * 8,
                    )
                } else {
                    (mb_px_x / 2 + (block_idx & 1) * 8, mb_px_y)
                };

                self.encode_block(
                    &mut bitwriter,
                    &frame.cr_plane,
                    frame.chroma_stride as usize,
                    block_x,
                    block_y,
                    chroma_quant,
                    &mut last_dc[2],
                    self.config.bit_depth,
                )?;
            }

            // Encode alpha blocks if present
            if self.config.include_alpha {
                if let Some(ref alpha) = frame.alpha_plane {
                    for block_idx in 0..4 {
                        let block_x = mb_px_x + (block_idx & 1) * 8;
                        let block_y = mb_px_y + (block_idx >> 1) * 8;

                        self.encode_alpha_block(
                            &mut bitwriter,
                            alpha,
                            frame.y_stride as usize,
                            block_x,
                            block_y,
                            self.config.bit_depth,
                        )?;
                    }
                }
            }
        }

        // Flush bitwriter and append to output
        bitwriter.flush();
        output.extend_from_slice(&bitwriter.into_bytes());

        Ok(())
    }

    /// Encode a single 8x8 block
    fn encode_block(
        &mut self,
        writer: &mut BitstreamWriter,
        plane: &[i16],
        stride: usize,
        x: usize,
        y: usize,
        quant_matrix: &[u8; 64],
        last_dc: &mut i16,
        bit_depth: BitDepth,
    ) -> Result<()> {
        // Extract 8x8 block
        let mut block = [0i16; 64];
        let offset = (bit_depth.max_value() as i16 + 1) / 2;

        for row in 0..8 {
            let src_offset = (y + row) * stride + x;
            if src_offset + 8 <= plane.len() {
                for col in 0..8 {
                    // Remove DC offset
                    block[row * 8 + col] = plane[src_offset + col] - offset;
                }
            }
        }

        // Apply forward DCT
        let dct_block = forward_dct_2d(&block);

        // Quantize
        let mut quant_block = [0i16; 64];
        for i in 0..64 {
            let q = quant_matrix[i] as i32;
            quant_block[i] = ((dct_block[i] as i32 * 16) / q) as i16;
        }

        // Convert to zigzag order
        let zigzag_block = zigzag(&quant_block);

        // Encode DC coefficient
        let dc_diff = zigzag_block[0] - *last_dc;
        *last_dc = zigzag_block[0];
        self.encode_dc_diff(writer, dc_diff)?;

        // Encode AC coefficients
        self.encode_ac_coeffs(writer, &zigzag_block)?;

        Ok(())
    }

    /// Encode an alpha block
    fn encode_alpha_block(
        &mut self,
        writer: &mut BitstreamWriter,
        plane: &[u16],
        stride: usize,
        x: usize,
        y: usize,
        bit_depth: BitDepth,
    ) -> Result<()> {
        let sample_bits = match bit_depth {
            BitDepth::Bit10 => 8,
            BitDepth::Bit12 => 16,
        };

        for row in 0..8 {
            let src_offset = (y + row) * stride + x;
            if src_offset + 8 <= plane.len() {
                for col in 0..8 {
                    let sample = plane[src_offset + col] as u32;
                    writer.write_bits(sample, sample_bits);
                }
            } else {
                // Pad with zeros
                for _ in 0..8 {
                    writer.write_bits(0, sample_bits);
                }
            }
        }

        Ok(())
    }

    /// Encode DC coefficient difference
    fn encode_dc_diff(&self, writer: &mut BitstreamWriter, diff: i16) -> Result<()> {
        // Determine category (number of bits needed)
        let abs_diff = diff.unsigned_abs() as u32;
        let category = if abs_diff == 0 {
            0
        } else {
            32 - abs_diff.leading_zeros()
        } as u8;

        // Write Huffman code for category
        let (code, len) = self.dc_table.get_code(category);
        writer.write_bits(code as u32, len);

        // Write the difference value if category > 0
        if category > 0 {
            let value = if diff < 0 {
                // Negative: use one's complement
                (diff + (1 << category) - 1) as u32
            } else {
                diff as u32
            };
            writer.write_bits(value, category);
        }

        Ok(())
    }

    /// Encode AC coefficients
    fn encode_ac_coeffs(&self, writer: &mut BitstreamWriter, coeffs: &[i16; 64]) -> Result<()> {
        let mut pos = 1; // Start after DC
        let mut run = 0;

        while pos < 64 {
            if coeffs[pos] == 0 {
                run += 1;
                pos += 1;
            } else {
                let level = coeffs[pos].unsigned_abs();
                let sign = coeffs[pos] < 0;

                // Try to find a VLC code for (run, level)
                if let Some((code, len)) = self.ac_table.get_code(run, level) {
                    writer.write_bits(code as u32, len);
                    writer.write_bits(sign as u32, 1);
                } else {
                    // Use escape code
                    writer.write_bits(self.ac_table.escape_code as u32, self.ac_table.escape_length);
                    writer.write_bits(run as u32, 6);

                    // Write 12-bit signed level
                    let level_bits = if sign {
                        (-(coeffs[pos] as i32) + 4096) as u32 & 0xFFF
                    } else {
                        coeffs[pos] as u32 & 0xFFF
                    };
                    writer.write_bits(level_bits, 12);
                }

                run = 0;
                pos += 1;
            }
        }

        // Write end of block (EOB)
        let (eob_code, eob_len) = self.ac_table.get_eob();
        writer.write_bits(eob_code as u32, eob_len);

        Ok(())
    }

    /// Get color primaries code
    fn color_primaries_code(&self) -> u8 {
        match self.config.color_primaries {
            ColorPrimaries::Unknown => 0,
            ColorPrimaries::BT709 => 1,
            ColorPrimaries::BT601NTSC => 6,
            ColorPrimaries::BT601PAL => 7,
            ColorPrimaries::BT2020 => 9,
            ColorPrimaries::DCIP3 => 11,
        }
    }

    /// Get transfer characteristic code
    fn transfer_characteristic_code(&self) -> u8 {
        match self.config.transfer_characteristic {
            TransferCharacteristic::Unknown => 0,
            TransferCharacteristic::BT709 => 1,
            TransferCharacteristic::BT601 => 6,
            TransferCharacteristic::PQ => 16,
            TransferCharacteristic::HLG => 18,
        }
    }

    /// Get matrix coefficients code
    fn matrix_coefficients_code(&self) -> u8 {
        match self.config.matrix_coefficients {
            MatrixCoefficients::Unknown => 0,
            MatrixCoefficients::BT709 => 1,
            MatrixCoefficients::BT601 => 6,
            MatrixCoefficients::BT2020NCL => 9,
            MatrixCoefficients::BT2020CL => 10,
        }
    }
}

/// Forward 2D DCT
fn forward_dct_2d(block: &[i16; 64]) -> [i16; 64] {
    let mut temp = [0i64; 64];
    let mut output = [0i16; 64];

    // Row DCT
    for row in 0..8 {
        let row_offset = row * 8;
        forward_dct_row(&block[row_offset..row_offset + 8], &mut temp[row_offset..row_offset + 8]);
    }

    // Column DCT
    for col in 0..8 {
        let mut column = [0i64; 8];
        for row in 0..8 {
            column[row] = temp[row * 8 + col];
        }
        let mut out_col = [0i64; 8];
        forward_dct_col(&column, &mut out_col);

        for row in 0..8 {
            // Scale and convert to i16
            output[row * 8 + col] = ((out_col[row] + (1 << 13)) >> 14).clamp(-32768, 32767) as i16;
        }
    }

    output
}

/// Forward 1D DCT for a row
fn forward_dct_row(input: &[i16], output: &mut [i64]) {
    let c1 = COS_TABLE[0] as i64;
    let c2 = COS_TABLE[1] as i64;
    let c3 = COS_TABLE[2] as i64;
    let c4 = COS_TABLE[3] as i64;
    let c5 = COS_TABLE[4] as i64;
    let c6 = COS_TABLE[5] as i64;
    let c7 = COS_TABLE[6] as i64;
    let scale = IDCT_SCALE as i64;

    let x0 = input[0] as i64;
    let x1 = input[1] as i64;
    let x2 = input[2] as i64;
    let x3 = input[3] as i64;
    let x4 = input[4] as i64;
    let x5 = input[5] as i64;
    let x6 = input[6] as i64;
    let x7 = input[7] as i64;

    // Stage 1: butterfly
    let s0 = x0 + x7;
    let s1 = x1 + x6;
    let s2 = x2 + x5;
    let s3 = x3 + x4;
    let s4 = x3 - x4;
    let s5 = x2 - x5;
    let s6 = x1 - x6;
    let s7 = x0 - x7;

    // Even coefficients
    let t0 = s0 + s3;
    let t1 = s1 + s2;
    let t2 = s1 - s2;
    let t3 = s0 - s3;

    output[0] = (t0 + t1) * scale / 8;
    output[4] = (t0 - t1) * c4 / scale;
    output[2] = (t3 * c2 + t2 * c6) / scale;
    output[6] = (t3 * c6 - t2 * c2) / scale;

    // Odd coefficients
    let t4 = s4 * c3 + s7 * c5;
    let t5 = s5 * c1 + s6 * c7;
    let t6 = s5 * c7 - s6 * c1;
    let t7 = s4 * c5 - s7 * c3;

    output[1] = (t5 + t4) / scale;
    output[3] = (t6 + t7) / scale;
    output[5] = (t6 - t7) / scale;
    output[7] = (t5 - t4) / scale;
}

/// Forward 1D DCT for a column
fn forward_dct_col(input: &[i64], output: &mut [i64]) {
    let c1 = COS_TABLE[0] as i64;
    let c2 = COS_TABLE[1] as i64;
    let c3 = COS_TABLE[2] as i64;
    let c4 = COS_TABLE[3] as i64;
    let c5 = COS_TABLE[4] as i64;
    let c6 = COS_TABLE[5] as i64;
    let c7 = COS_TABLE[6] as i64;
    let scale = IDCT_SCALE as i64;

    let x0 = input[0];
    let x1 = input[1];
    let x2 = input[2];
    let x3 = input[3];
    let x4 = input[4];
    let x5 = input[5];
    let x6 = input[6];
    let x7 = input[7];

    // Stage 1: butterfly
    let s0 = x0 + x7;
    let s1 = x1 + x6;
    let s2 = x2 + x5;
    let s3 = x3 + x4;
    let s4 = x3 - x4;
    let s5 = x2 - x5;
    let s6 = x1 - x6;
    let s7 = x0 - x7;

    // Even coefficients
    let t0 = s0 + s3;
    let t1 = s1 + s2;
    let t2 = s1 - s2;
    let t3 = s0 - s3;

    output[0] = (t0 + t1) * scale / 8;
    output[4] = (t0 - t1) * c4 / scale;
    output[2] = (t3 * c2 + t2 * c6) / scale;
    output[6] = (t3 * c6 - t2 * c2) / scale;

    // Odd coefficients
    let t4 = s4 * c3 + s7 * c5;
    let t5 = s5 * c1 + s6 * c7;
    let t6 = s5 * c7 - s6 * c1;
    let t7 = s4 * c5 - s7 * c3;

    output[1] = (t5 + t4) / scale;
    output[3] = (t6 + t7) / scale;
    output[5] = (t6 - t7) / scale;
    output[7] = (t5 - t4) / scale;
}

/// DC Huffman encoder table
struct DcHuffmanEncoder {
    /// Code lengths for each symbol (0-11)
    lengths: [u8; 12],
    /// Codes for each symbol
    codes: [u16; 12],
}

impl DcHuffmanEncoder {
    fn new() -> Self {
        DcHuffmanEncoder {
            lengths: [2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10],
            codes: [
                0b00,       // symbol 0, len 2
                0b01,       // symbol 1, len 2
                0b10,       // symbol 2, len 2
                0b110,      // symbol 3, len 3
                0b111,      // symbol 4, len 3
                0b1110,     // symbol 5, len 4
                0b11110,    // symbol 6, len 5
                0b111110,   // symbol 7, len 6
                0b1111110,  // symbol 8, len 7
                0b11111110, // symbol 9, len 8
                0b111111110, // symbol 10, len 9
                0b1111111110, // symbol 11, len 10
            ],
        }
    }

    fn get_code(&self, category: u8) -> (u16, u8) {
        let idx = (category as usize).min(11);
        (self.codes[idx], self.lengths[idx])
    }
}

/// AC Huffman encoder table
struct AcHuffmanEncoder {
    /// Escape code
    escape_code: u16,
    /// Escape code length
    escape_length: u8,
}

impl AcHuffmanEncoder {
    fn new() -> Self {
        AcHuffmanEncoder {
            escape_code: 0b11111111110,
            escape_length: 11,
        }
    }

    fn get_code(&self, run: u8, level: u16) -> Option<(u16, u8)> {
        // Common ProRes AC codes
        match (run, level) {
            (0, 0) => Some((0b10, 2)),     // EOB
            (0, 1) => Some((0b00, 2)),
            (0, 2) => Some((0b010, 3)),
            (0, 3) => Some((0b0110, 4)),
            (0, 4) => Some((0b01110, 5)),
            (0, 5) => Some((0b011110, 6)),
            (0, 6) => Some((0b0111110, 7)),
            (0, 7) => Some((0b01111110, 8)),
            (1, 1) => Some((0b110, 3)),
            (1, 2) => Some((0b11110, 5)),
            (1, 3) => Some((0b1111110, 7)),
            (2, 1) => Some((0b1110, 4)),
            (2, 2) => Some((0b111110, 6)),
            (3, 1) => Some((0b111010, 6)),
            (4, 1) => Some((0b1110110, 7)),
            (5, 1) => Some((0b11101110, 8)),
            (6, 1) => Some((0b111011110, 9)),
            _ => None, // Use escape code
        }
    }

    fn get_eob(&self) -> (u16, u8) {
        (0b10, 2) // End of block
    }
}

/// Bitstream writer for encoding
struct BitstreamWriter {
    buffer: Vec<u8>,
    current_byte: u32,
    bit_count: u8,
}

impl BitstreamWriter {
    fn new() -> Self {
        BitstreamWriter {
            buffer: Vec::new(),
            current_byte: 0,
            bit_count: 0,
        }
    }

    fn write_bits(&mut self, value: u32, num_bits: u8) {
        if num_bits == 0 {
            return;
        }

        let mut bits_remaining = num_bits;
        let mut value = value;

        while bits_remaining > 0 {
            let bits_available = 8 - self.bit_count;
            let bits_to_write = bits_remaining.min(bits_available);

            // Extract the top bits we're writing
            let shift = bits_remaining - bits_to_write;
            let mask = (1u32 << bits_to_write) - 1;
            let bits = (value >> shift) & mask;

            // Add to current byte
            self.current_byte = (self.current_byte << bits_to_write) | bits;
            self.bit_count += bits_to_write;
            bits_remaining -= bits_to_write;

            // Clear the bits we just wrote from value
            value &= (1u32 << shift) - 1;

            // Flush byte if full
            if self.bit_count >= 8 {
                self.buffer.push(self.current_byte as u8);
                self.current_byte = 0;
                self.bit_count = 0;
            }
        }
    }

    fn flush(&mut self) {
        if self.bit_count > 0 {
            // Pad with zeros and write final byte
            self.current_byte <<= 8 - self.bit_count;
            self.buffer.push(self.current_byte as u8);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buffer
    }
}

/// Encode raw YCbCr data to ProRes
pub fn encode_prores(
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    width: u16,
    height: u16,
    profile: ProResProfile,
) -> Result<Vec<u8>> {
    let config = EncoderConfig::new(profile);
    let mut encoder = ProResEncoder::with_config(config);
    encoder.encode_planar(y_plane, cb_plane, cr_plane, None, width, height)
}

/// Encode raw YCbCrA data to ProRes 4444
pub fn encode_prores_4444(
    y_plane: &[i16],
    cb_plane: &[i16],
    cr_plane: &[i16],
    alpha_plane: &[u16],
    width: u16,
    height: u16,
) -> Result<Vec<u8>> {
    let config = EncoderConfig::new(ProResProfile::P4444).with_alpha(true);
    let mut encoder = ProResEncoder::with_config(config);
    encoder.encode_planar(y_plane, cb_plane, cr_plane, Some(alpha_plane), width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = ProResEncoder::new();
        assert_eq!(encoder.frame_count(), 0);
        assert_eq!(encoder.config().profile, ProResProfile::HQ);
    }

    #[test]
    fn test_encoder_config() {
        let config = EncoderConfig::new(ProResProfile::P4444)
            .quality(90)
            .with_alpha(true);

        assert_eq!(config.profile, ProResProfile::P4444);
        assert_eq!(config.quality, 90);
        assert!(config.include_alpha);
    }

    #[test]
    fn test_encoder_config_quality_clamping() {
        let config = EncoderConfig::default().quality(150);
        assert_eq!(config.quality, 100);

        let config = EncoderConfig::default().quality(0);
        assert_eq!(config.quality, 1);
    }

    #[test]
    fn test_forward_dct() {
        // Test with DC block (all same values)
        let block = [128i16; 64];
        let dct = forward_dct_2d(&block);

        // DC coefficient should be non-zero, AC should be near zero
        assert_ne!(dct[0], 0);

        // Most AC coefficients should be zero or near zero for flat block
        let ac_sum: i32 = dct[1..].iter().map(|&x| (x as i32).abs()).sum();
        assert!(ac_sum < 100, "AC sum too high for flat block: {}", ac_sum);
    }

    #[test]
    fn test_bitstream_writer() {
        let mut writer = BitstreamWriter::new();

        writer.write_bits(0b101, 3);
        writer.write_bits(0b11001, 5);
        writer.flush();

        let bytes = writer.into_bytes();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0b10111001);
    }

    #[test]
    fn test_bitstream_writer_multi_byte() {
        let mut writer = BitstreamWriter::new();

        writer.write_bits(0xFF, 8);
        writer.write_bits(0x00, 8);
        writer.write_bits(0xAB, 8);
        writer.flush();

        let bytes = writer.into_bytes();
        assert_eq!(bytes, vec![0xFF, 0x00, 0xAB]);
    }

    #[test]
    fn test_dc_huffman_encoder() {
        let encoder = DcHuffmanEncoder::new();

        let (_code, len) = encoder.get_code(0);
        assert_eq!(len, 2);

        let (_code, len) = encoder.get_code(5);
        assert_eq!(len, 4);
    }

    #[test]
    fn test_ac_huffman_encoder() {
        let encoder = AcHuffmanEncoder::new();

        // EOB
        let (code, len) = encoder.get_eob();
        assert_eq!(code, 0b10);
        assert_eq!(len, 2);

        // Common code
        assert!(encoder.get_code(0, 1).is_some());
        assert!(encoder.get_code(1, 1).is_some());

        // Uncommon - should return None
        assert!(encoder.get_code(10, 10).is_none());
    }

    #[test]
    fn test_quality_to_scale() {
        let encoder = ProResEncoder::new();

        // High quality = lower scale
        let scale_high = encoder.quality_to_scale(95);
        let scale_mid = encoder.quality_to_scale(75);
        let scale_low = encoder.quality_to_scale(25);

        assert!(scale_high < scale_mid);
        assert!(scale_mid < scale_low);
    }

    #[test]
    fn test_encode_small_frame() {
        // Create a small test frame (16x16 minimum for 1 macroblock)
        let width = 16u16;
        let height = 16u16;

        let y_size = (width as usize) * (height as usize);
        let chroma_size = y_size / 2; // 4:2:2

        let y_plane: Vec<i16> = (0..y_size).map(|i| (i % 512) as i16).collect();
        let cb_plane: Vec<i16> = vec![512; chroma_size];
        let cr_plane: Vec<i16> = vec![512; chroma_size];

        let result = encode_prores(&y_plane, &cb_plane, &cr_plane, width, height, ProResProfile::HQ);

        // Should succeed
        assert!(result.is_ok());

        let encoded = result.unwrap();

        // Check signature
        assert!(encoded.len() >= 8);
        assert_eq!(&encoded[4..8], b"icpf");
    }

    #[test]
    fn test_all_profiles() {
        let profiles = [
            ProResProfile::Proxy,
            ProResProfile::LT,
            ProResProfile::Standard,
            ProResProfile::HQ,
        ];

        for profile in profiles {
            let config = EncoderConfig::new(profile);
            let encoder = ProResEncoder::with_config(config);
            assert_eq!(encoder.config().profile, profile);
        }
    }

    #[test]
    fn test_4444_profile() {
        let config = EncoderConfig::new(ProResProfile::P4444);
        assert!(config.profile.is_444());
        assert!(config.profile.supports_alpha());
        assert_eq!(config.chroma_format(), ChromaFormat::YUV444);
    }

    #[test]
    fn test_chroma_format() {
        let config_422 = EncoderConfig::new(ProResProfile::HQ);
        assert_eq!(config_422.chroma_format(), ChromaFormat::YUV422);

        let config_444 = EncoderConfig::new(ProResProfile::P4444);
        assert_eq!(config_444.chroma_format(), ChromaFormat::YUV444);
    }
}
