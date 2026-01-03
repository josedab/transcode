//! ProRes slice decoding with DCT

use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;

use crate::error::{ProResError, Result};
use crate::frame::FrameHeader;
use crate::huffman::{
    decode_ac_coeffs, decode_dc_diff, AcHuffmanTable, BitstreamReader, DcHuffmanTable,
};
use crate::tables::{dezigzag, COS_TABLE, IDCT_ROUND, IDCT_SCALE};
use crate::types::{BitDepth, ChromaFormat};

/// Slice header information
#[derive(Debug, Clone)]
pub struct SliceHeader {
    /// Slice header size in bytes
    pub header_size: u8,
    /// Quantization scale index for Y
    pub qscale_y: u8,
    /// Quantization scale index for Cb
    pub qscale_cb: u8,
    /// Quantization scale index for Cr
    pub qscale_cr: u8,
    /// Y data size in bytes (for 4444)
    pub y_data_size: u16,
    /// Cb data size in bytes (for 4444)
    pub cb_data_size: u16,
    /// Cr data size in bytes (for 4444)
    pub cr_data_size: u16,
    /// Alpha data size in bytes
    pub alpha_data_size: u16,
}

impl SliceHeader {
    /// Parse slice header from data
    pub fn parse(data: &[u8], is_444: bool, has_alpha: bool) -> Result<Self> {
        if data.is_empty() {
            return Err(ProResError::InvalidSliceHeader("Empty slice data".into()));
        }

        let mut cursor = Cursor::new(data);
        let header_size = cursor.read_u8()?;

        if data.len() < header_size as usize {
            return Err(ProResError::InsufficientData {
                needed: header_size as usize,
                available: data.len(),
            });
        }

        // First byte after header size contains qscale info
        let qscale_byte = cursor.read_u8()?;
        let qscale_y = qscale_byte >> 4;
        let qscale_c = qscale_byte & 0x0F;

        // For 4444, read individual component sizes
        let (y_data_size, cb_data_size, cr_data_size, alpha_data_size) = if is_444 {
            let y_size = cursor.read_u16::<BigEndian>()?;
            let cb_size = cursor.read_u16::<BigEndian>()?;
            let cr_size = if header_size >= 8 {
                cursor.read_u16::<BigEndian>()?
            } else {
                0
            };
            let alpha_size = if has_alpha && header_size >= 10 {
                cursor.read_u16::<BigEndian>()?
            } else {
                0
            };
            (y_size, cb_size, cr_size, alpha_size)
        } else {
            // For 422, component sizes are calculated
            (0, 0, 0, 0)
        };

        Ok(SliceHeader {
            header_size,
            qscale_y,
            qscale_cb: qscale_c,
            qscale_cr: qscale_c,
            y_data_size,
            cb_data_size,
            cr_data_size,
            alpha_data_size,
        })
    }
}

/// Slice decoder
pub struct SliceDecoder {
    /// DC Huffman table
    dc_table: DcHuffmanTable,
    /// AC Huffman table
    ac_table: AcHuffmanTable,
    /// Last DC values for prediction
    last_dc: [i16; 4], // Y, Cb, Cr, Alpha
    /// Dequantization buffer
    dequant_buffer: [i16; 64],
}

impl Default for SliceDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SliceDecoder {
    /// Create a new slice decoder
    pub fn new() -> Self {
        SliceDecoder {
            dc_table: DcHuffmanTable::new(),
            ac_table: AcHuffmanTable::new(),
            last_dc: [0; 4],
            dequant_buffer: [0; 64],
        }
    }

    /// Reset DC prediction for a new picture
    pub fn reset_dc_prediction(&mut self) {
        self.last_dc = [0; 4];
    }

    /// Decode a slice
    pub fn decode_slice(
        &mut self,
        data: &[u8],
        header: &FrameHeader,
        slice_x: u32,
        slice_y: u32,
        slice_mb_width: u32,
        y_plane: &mut [i16],
        cb_plane: &mut [i16],
        cr_plane: &mut [i16],
        mut alpha_plane: Option<&mut [u16]>,
        y_stride: u32,
        chroma_stride: u32,
    ) -> Result<()> {
        let is_444 = header.chroma_format == ChromaFormat::YUV444;
        let has_alpha = header.has_alpha();

        // Parse slice header
        let slice_header = SliceHeader::parse(data, is_444, has_alpha)?;
        let coeff_data = &data[slice_header.header_size as usize..];

        // Reset DC prediction at slice boundary
        self.reset_dc_prediction();

        // Calculate quantization matrices with scale
        let y_quant = self.build_quant_matrix(&header.luma_quant_matrix, slice_header.qscale_y);
        let c_quant = self.build_quant_matrix(&header.chroma_quant_matrix, slice_header.qscale_cb);

        // Decode macroblocks in this slice
        let mut reader = BitstreamReader::new(coeff_data);

        for mb_x in 0..slice_mb_width {
            let mb_px_x = ((slice_x + mb_x) * 16) as usize;
            let mb_px_y = (slice_y * 16) as usize;

            // Decode Y blocks (4 blocks per macroblock: top-left, top-right, bottom-left, bottom-right)
            for block_idx in 0..4 {
                let block_x = mb_px_x + (block_idx & 1) * 8;
                let block_y = mb_px_y + (block_idx >> 1) * 8;

                self.decode_block(
                    &mut reader,
                    0, // Y component
                    &y_quant,
                    header.bit_depth,
                )?;

                // Copy block to Y plane
                self.copy_block_to_plane(
                    y_plane,
                    y_stride as usize,
                    block_x,
                    block_y,
                    header.bit_depth,
                );
            }

            // Decode Cb/Cr blocks
            let chroma_blocks = if is_444 { 4 } else { 2 }; // 4 blocks for 444, 2 for 422

            for block_idx in 0..chroma_blocks {
                let (block_x, block_y) = if is_444 {
                    (
                        mb_px_x + (block_idx & 1) * 8,
                        mb_px_y + (block_idx >> 1) * 8,
                    )
                } else {
                    (mb_px_x / 2 + (block_idx & 1) * 8, mb_px_y)
                };

                // Cb block
                self.decode_block(&mut reader, 1, &c_quant, header.bit_depth)?;
                self.copy_block_to_plane(
                    cb_plane,
                    chroma_stride as usize,
                    block_x,
                    block_y,
                    header.bit_depth,
                );
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

                // Cr block
                self.decode_block(&mut reader, 2, &c_quant, header.bit_depth)?;
                self.copy_block_to_plane(
                    cr_plane,
                    chroma_stride as usize,
                    block_x,
                    block_y,
                    header.bit_depth,
                );
            }

            // Decode alpha blocks if present
            if let Some(ref mut alpha) = alpha_plane {
                if has_alpha {
                    for block_idx in 0..4 {
                        let block_x = mb_px_x + (block_idx & 1) * 8;
                        let block_y = mb_px_y + (block_idx >> 1) * 8;

                        self.decode_alpha_block(&mut reader, header.bit_depth)?;
                        self.copy_alpha_block_to_plane(
                            alpha,
                            y_stride as usize,
                            block_x,
                            block_y,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Build quantization matrix with scale factor
    fn build_quant_matrix(&self, base_matrix: &[u8; 64], qscale: u8) -> [i32; 64] {
        let mut matrix = [0i32; 64];
        let scale = if qscale == 0 { 1 } else { qscale as i32 * 2 };

        for i in 0..64 {
            matrix[i] = (base_matrix[i] as i32) * scale;
        }
        matrix
    }

    /// Decode a single 8x8 block
    fn decode_block(
        &mut self,
        reader: &mut BitstreamReader,
        component: usize,
        quant_matrix: &[i32; 64],
        bit_depth: BitDepth,
    ) -> Result<()> {
        let mut coeffs = [0i16; 64];

        // Decode DC coefficient
        let dc_diff = decode_dc_diff(reader, &self.dc_table)?;
        self.last_dc[component] = self.last_dc[component].saturating_add(dc_diff);
        coeffs[0] = self.last_dc[component];

        // Decode AC coefficients
        decode_ac_coeffs(reader, &mut coeffs, &self.ac_table)?;

        // Convert from zigzag order
        let block_coeffs = dezigzag(&coeffs);

        // Dequantize
        for i in 0..64 {
            self.dequant_buffer[i] = ((block_coeffs[i] as i32 * quant_matrix[i]) >> 4) as i16;
        }

        // Apply IDCT
        self.idct_2d(bit_depth);

        Ok(())
    }

    /// Decode an alpha block (uses different coding)
    fn decode_alpha_block(
        &mut self,
        reader: &mut BitstreamReader,
        bit_depth: BitDepth,
    ) -> Result<()> {
        // Alpha uses simpler coding - direct samples or RLE
        // For now, decode as direct 8-bit or 16-bit samples

        let sample_bits = match bit_depth {
            BitDepth::Bit10 => 8,
            BitDepth::Bit12 => 16,
        };

        for i in 0..64 {
            let sample = reader.read_bits(sample_bits)?;
            self.dequant_buffer[i] = sample as i16;
        }

        Ok(())
    }

    /// Perform 2D IDCT on dequant_buffer
    fn idct_2d(&mut self, bit_depth: BitDepth) {
        let max_val = bit_depth.max_value() as i16;
        let offset = (max_val + 1) / 2;

        // Temporary buffer for row results
        let mut temp = [0i32; 64];

        // Row IDCT
        for row in 0..8 {
            let row_offset = row * 8;
            self.idct_row(&self.dequant_buffer[row_offset..row_offset + 8], &mut temp[row_offset..row_offset + 8]);
        }

        // Column IDCT
        let mut output = [0i16; 64];
        for col in 0..8 {
            let mut column = [0i32; 8];
            for row in 0..8 {
                column[row] = temp[row * 8 + col];
            }
            let mut out_col = [0i32; 8];
            self.idct_col(&column, &mut out_col);

            for row in 0..8 {
                // Scale, add DC offset, and clamp
                let val = ((out_col[row] + IDCT_ROUND) >> 14) + i32::from(offset);
                output[row * 8 + col] = val.clamp(0, i32::from(max_val)) as i16;
            }
        }

        self.dequant_buffer = output;
    }

    /// 1D IDCT for a row
    fn idct_row(&self, input: &[i16], output: &mut [i32]) {
        // Simplified IDCT using AAN algorithm
        let c1 = COS_TABLE[0];
        let c2 = COS_TABLE[1];
        let c3 = COS_TABLE[2];
        let c4 = COS_TABLE[3];
        let c5 = COS_TABLE[4];
        let c6 = COS_TABLE[5];
        let c7 = COS_TABLE[6];

        let x0 = input[0] as i32 * IDCT_SCALE;
        let x1 = input[1] as i32;
        let x2 = input[2] as i32;
        let x3 = input[3] as i32;
        let x4 = input[4] as i32;
        let x5 = input[5] as i32;
        let x6 = input[6] as i32;
        let x7 = input[7] as i32;

        // Even part
        let s0 = x0 + x4 * c4;
        let s1 = x0 - x4 * c4;
        let s2 = x2 * c6 - x6 * c2;
        let s3 = x2 * c2 + x6 * c6;

        let t0 = s0 + s3;
        let t1 = s1 + s2;
        let t2 = s1 - s2;
        let t3 = s0 - s3;

        // Odd part
        let s4 = x1 * c7 - x7 * c1;
        let s5 = x1 * c1 + x7 * c7;
        let s6 = x3 * c3 - x5 * c5;
        let s7 = x3 * c5 + x5 * c3;

        let t4 = s4 + s6;
        let t5 = s5 + s7;
        let t6 = s5 - s7;
        let t7 = s4 - s6;

        output[0] = t0 + t5;
        output[1] = t1 + t6;
        output[2] = t2 + t7;
        output[3] = t3 + t4;
        output[4] = t3 - t4;
        output[5] = t2 - t7;
        output[6] = t1 - t6;
        output[7] = t0 - t5;
    }

    /// 1D IDCT for a column
    fn idct_col(&self, input: &[i32], output: &mut [i32]) {
        let c1 = COS_TABLE[0];
        let c2 = COS_TABLE[1];
        let c3 = COS_TABLE[2];
        let c4 = COS_TABLE[3];
        let c5 = COS_TABLE[4];
        let c6 = COS_TABLE[5];
        let c7 = COS_TABLE[6];

        let x0 = input[0];
        let x1 = input[1];
        let x2 = input[2];
        let x3 = input[3];
        let x4 = input[4];
        let x5 = input[5];
        let x6 = input[6];
        let x7 = input[7];

        // Even part
        let s0 = x0 + (x4 * c4) / IDCT_SCALE;
        let s1 = x0 - (x4 * c4) / IDCT_SCALE;
        let s2 = (x2 * c6 - x6 * c2) / IDCT_SCALE;
        let s3 = (x2 * c2 + x6 * c6) / IDCT_SCALE;

        let t0 = s0 + s3;
        let t1 = s1 + s2;
        let t2 = s1 - s2;
        let t3 = s0 - s3;

        // Odd part
        let s4 = (x1 * c7 - x7 * c1) / IDCT_SCALE;
        let s5 = (x1 * c1 + x7 * c7) / IDCT_SCALE;
        let s6 = (x3 * c3 - x5 * c5) / IDCT_SCALE;
        let s7 = (x3 * c5 + x5 * c3) / IDCT_SCALE;

        let t4 = s4 + s6;
        let t5 = s5 + s7;
        let t6 = s5 - s7;
        let t7 = s4 - s6;

        output[0] = t0 + t5;
        output[1] = t1 + t6;
        output[2] = t2 + t7;
        output[3] = t3 + t4;
        output[4] = t3 - t4;
        output[5] = t2 - t7;
        output[6] = t1 - t6;
        output[7] = t0 - t5;
    }

    /// Copy decoded block to Y/Cb/Cr plane
    fn copy_block_to_plane(
        &self,
        plane: &mut [i16],
        stride: usize,
        x: usize,
        y: usize,
        _bit_depth: BitDepth,
    ) {
        for row in 0..8 {
            let src_offset = row * 8;
            let dst_offset = (y + row) * stride + x;

            if dst_offset + 8 <= plane.len() {
                plane[dst_offset..dst_offset + 8]
                    .copy_from_slice(&self.dequant_buffer[src_offset..src_offset + 8]);
            }
        }
    }

    /// Copy decoded alpha block to alpha plane
    fn copy_alpha_block_to_plane(
        &self,
        plane: &mut [u16],
        stride: usize,
        x: usize,
        y: usize,
    ) {
        for row in 0..8 {
            let src_offset = row * 8;
            let dst_offset = (y + row) * stride + x;

            if dst_offset + 8 <= plane.len() {
                for col in 0..8 {
                    plane[dst_offset + col] = self.dequant_buffer[src_offset + col] as u16;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_header_parse() {
        // Minimal slice header
        let data = [0x02, 0x44]; // header_size=2, qscale_y=4, qscale_c=4
        let header = SliceHeader::parse(&data, false, false).unwrap();

        assert_eq!(header.header_size, 2);
        assert_eq!(header.qscale_y, 4);
        assert_eq!(header.qscale_cb, 4);
    }

    #[test]
    fn test_slice_header_parse_444() {
        // 4444 slice header with component sizes
        let data = [
            0x0A, 0x44, // header_size=10, qscale
            0x01, 0x00, // y_size = 256
            0x00, 0x80, // cb_size = 128
            0x00, 0x80, // cr_size = 128
            0x00, 0x40, // alpha_size = 64
        ];
        let header = SliceHeader::parse(&data, true, true).unwrap();

        assert_eq!(header.header_size, 10);
        assert_eq!(header.y_data_size, 256);
        assert_eq!(header.cb_data_size, 128);
        assert_eq!(header.cr_data_size, 128);
        assert_eq!(header.alpha_data_size, 64);
    }

    #[test]
    fn test_slice_decoder_creation() {
        let decoder = SliceDecoder::new();
        assert_eq!(decoder.last_dc, [0; 4]);
    }

    #[test]
    fn test_quant_matrix_scaling() {
        let decoder = SliceDecoder::new();
        let base = [16u8; 64];
        let scaled = decoder.build_quant_matrix(&base, 2);

        // scale = 2 * 2 = 4
        assert_eq!(scaled[0], 16 * 4);
    }
}
