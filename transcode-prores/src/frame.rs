//! ProRes frame header parsing and frame data structures

use byteorder::{BigEndian, ReadBytesExt};
use std::io::{Cursor, Read};

use crate::error::{ProResError, Result};
use crate::types::{
    BitDepth, ChromaFormat, ColorPrimaries, InterlaceMode, MatrixCoefficients, ProResProfile,
    TransferCharacteristic,
};

/// ProRes frame signature
const FRAME_SIGNATURE: &[u8; 4] = b"icpf";

/// ProRes frame header
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Frame size in bytes (including header)
    pub frame_size: u32,
    /// ProRes profile
    pub profile: ProResProfile,
    /// Frame width in pixels
    pub width: u16,
    /// Frame height in pixels
    pub height: u16,
    /// Chroma format (4:2:2 or 4:4:4)
    pub chroma_format: ChromaFormat,
    /// Interlace mode
    pub interlace_mode: InterlaceMode,
    /// Bit depth
    pub bit_depth: BitDepth,
    /// Color primaries
    pub color_primaries: ColorPrimaries,
    /// Transfer characteristic
    pub transfer_characteristic: TransferCharacteristic,
    /// Matrix coefficients
    pub matrix_coefficients: MatrixCoefficients,
    /// Alpha channel info (0 = no alpha, 1 = 8-bit alpha, 2 = 16-bit alpha)
    pub alpha_info: u8,
    /// Luma quantization matrix (64 values in zigzag order)
    pub luma_quant_matrix: [u8; 64],
    /// Chroma quantization matrix (64 values in zigzag order)
    pub chroma_quant_matrix: [u8; 64],
    /// Number of slices per row
    pub slices_per_row: u16,
    /// Number of slice rows
    pub slice_rows: u16,
    /// Slice data offsets
    pub slice_offsets: Vec<u32>,
    /// Size of the header (to find where picture data starts)
    pub header_size: usize,
}

impl FrameHeader {
    /// Minimum frame header size (before quantization matrices)
    const MIN_HEADER_SIZE: usize = 28;

    /// Parse a ProRes frame header from raw bytes
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < Self::MIN_HEADER_SIZE {
            return Err(ProResError::InsufficientData {
                needed: Self::MIN_HEADER_SIZE,
                available: data.len(),
            });
        }

        let mut cursor = Cursor::new(data);

        // Read frame size (4 bytes)
        let frame_size = cursor.read_u32::<BigEndian>()?;

        // Read and verify signature (4 bytes)
        let mut signature = [0u8; 4];
        cursor.read_exact(&mut signature)?;

        if &signature != FRAME_SIGNATURE {
            return Err(ProResError::InvalidSignature(signature));
        }

        // Read header size (2 bytes)
        let header_size = cursor.read_u16::<BigEndian>()? as usize;

        // Read version (2 bytes, currently unused)
        let _version = cursor.read_u16::<BigEndian>()?;

        // Read FourCC (4 bytes)
        let mut fourcc = [0u8; 4];
        cursor.read_exact(&mut fourcc)?;

        let profile = ProResProfile::from_fourcc(&fourcc)
            .ok_or(ProResError::UnknownProfile(fourcc))?;

        // Read frame dimensions (4 bytes)
        let width = cursor.read_u16::<BigEndian>()?;
        let height = cursor.read_u16::<BigEndian>()?;

        // Read frame flags (2 bytes)
        let frame_flags = cursor.read_u16::<BigEndian>()?;

        // Parse frame flags
        let chroma_format = if (frame_flags >> 10) & 0x3 == 3 {
            ChromaFormat::YUV444
        } else {
            ChromaFormat::YUV422
        };

        let interlaced = (frame_flags >> 8) & 0x1 != 0;
        let top_field_first = (frame_flags >> 9) & 0x1 != 0;
        let interlace_mode = InterlaceMode::from_flags(interlaced, top_field_first);

        // Read reserved/aspect ratio (4 bytes, currently unused)
        let _reserved = cursor.read_u32::<BigEndian>()?;

        // Read color info (3 bytes)
        let color_primaries = ColorPrimaries::from_code(cursor.read_u8()?);
        let transfer_characteristic = TransferCharacteristic::from_code(cursor.read_u8()?);
        let matrix_coefficients = MatrixCoefficients::from_code(cursor.read_u8()?);

        // Read source format flags (1 byte)
        let src_format_flags = cursor.read_u8()?;
        let alpha_info = (src_format_flags >> 4) & 0xF;
        let bit_depth_code = src_format_flags & 0x3;
        let bit_depth = BitDepth::from_code(bit_depth_code)
            .ok_or_else(|| ProResError::InvalidHeader(format!("Invalid bit depth code: {}", bit_depth_code)))?;

        // Read quantization matrices (64 bytes each)
        let mut luma_quant_matrix = [0u8; 64];
        cursor.read_exact(&mut luma_quant_matrix)?;

        let mut chroma_quant_matrix = [0u8; 64];
        cursor.read_exact(&mut chroma_quant_matrix)?;

        // Calculate slice information
        // ProRes uses macroblocks of 16x16 pixels
        // Each slice contains a row of macroblocks
        let mb_width = (width as u32).div_ceil(16);
        let mb_height = (height as u32).div_ceil(16);

        // Read picture header (starts at header_size offset from frame start)
        if data.len() < header_size + 6 {
            return Err(ProResError::InsufficientData {
                needed: header_size + 6,
                available: data.len(),
            });
        }

        let picture_header_start = header_size;
        let picture_data = &data[picture_header_start..];
        let mut pic_cursor = Cursor::new(picture_data);

        // Picture header size (1 byte)
        let picture_header_size = pic_cursor.read_u8()? as usize;

        // Skip reserved (1 byte)
        let _reserved = pic_cursor.read_u8()?;

        // Read slice index table size info
        let slice_info = pic_cursor.read_u16::<BigEndian>()?;
        let slices_per_row = slice_info >> 8;
        let log2_slice_mb_width = slice_info & 0xFF;

        // Calculate number of slice rows
        let slice_rows = mb_height as u16;

        // Calculate total number of slices
        let num_slices = if slices_per_row > 0 {
            slices_per_row as u32 * slice_rows as u32
        } else {
            // Calculate based on log2_slice_mb_width
            let slice_mb_width = 1u32 << log2_slice_mb_width;
            let slices_per_row_calc = mb_width.div_ceil(slice_mb_width);
            slices_per_row_calc * slice_rows as u32
        };

        // Read slice offsets
        let mut slice_offsets = Vec::with_capacity(num_slices as usize + 1);

        // First offset is right after the slice index table
        let slice_index_table_size = 2 * num_slices as usize;
        let first_slice_offset = picture_header_start + picture_header_size + slice_index_table_size;

        slice_offsets.push(first_slice_offset as u32);

        // Read relative offsets from the slice index table
        for _ in 0..num_slices {
            let offset = pic_cursor.read_u16::<BigEndian>()? as u32;
            slice_offsets.push(first_slice_offset as u32 + offset);
        }

        Ok(FrameHeader {
            frame_size,
            profile,
            width,
            height,
            chroma_format,
            interlace_mode,
            bit_depth,
            color_primaries,
            transfer_characteristic,
            matrix_coefficients,
            alpha_info,
            luma_quant_matrix,
            chroma_quant_matrix,
            slices_per_row: if slices_per_row > 0 { slices_per_row } else { ((mb_width + (1 << log2_slice_mb_width) - 1) >> log2_slice_mb_width) as u16 },
            slice_rows,
            slice_offsets,
            header_size,
        })
    }

    /// Check if this frame has an alpha channel
    pub fn has_alpha(&self) -> bool {
        self.alpha_info != 0 && self.profile.supports_alpha()
    }

    /// Get the number of slices in this frame
    pub fn num_slices(&self) -> usize {
        if self.slice_offsets.is_empty() {
            0
        } else {
            self.slice_offsets.len() - 1
        }
    }

    /// Get the macroblock dimensions of the frame
    pub fn mb_dimensions(&self) -> (u32, u32) {
        let mb_width = (self.width as u32).div_ceil(16);
        let mb_height = (self.height as u32).div_ceil(16);
        (mb_width, mb_height)
    }
}

/// Decoded ProRes frame with pixel data
#[derive(Debug, Clone)]
pub struct ProResFrame {
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// ProRes profile
    pub profile: ProResProfile,
    /// Bit depth (10 or 12)
    pub bit_depth: BitDepth,
    /// Chroma format
    pub chroma_format: ChromaFormat,
    /// Y (luma) plane data
    pub y_plane: Vec<i16>,
    /// Cb (blue chroma) plane data
    pub cb_plane: Vec<i16>,
    /// Cr (red chroma) plane data
    pub cr_plane: Vec<i16>,
    /// Alpha plane data (only for 4444 profiles)
    pub alpha_plane: Option<Vec<u16>>,
    /// Y plane stride (bytes per row)
    pub y_stride: u32,
    /// Chroma plane stride (bytes per row)
    pub chroma_stride: u32,
}

impl ProResFrame {
    /// Create a new frame with allocated planes
    pub fn new(header: &FrameHeader) -> Self {
        let width = header.width as u32;
        let height = header.height as u32;

        // Round up to macroblock boundaries
        let padded_width = (width + 15) & !15;
        let padded_height = (height + 15) & !15;

        let y_size = (padded_width * padded_height) as usize;
        let chroma_h_shift = header.chroma_format.chroma_h_shift();
        let chroma_width = padded_width >> chroma_h_shift;
        let chroma_size = (chroma_width * padded_height) as usize;

        let alpha_plane = if header.has_alpha() {
            Some(vec![0u16; y_size])
        } else {
            None
        };

        ProResFrame {
            width,
            height,
            profile: header.profile,
            bit_depth: header.bit_depth,
            chroma_format: header.chroma_format,
            y_plane: vec![0i16; y_size],
            cb_plane: vec![0i16; chroma_size],
            cr_plane: vec![0i16; chroma_size],
            alpha_plane,
            y_stride: padded_width,
            chroma_stride: chroma_width,
        }
    }

    /// Get the padded dimensions (rounded up to macroblock boundaries)
    pub fn padded_dimensions(&self) -> (u32, u32) {
        let padded_width = (self.width + 15) & !15;
        let padded_height = (self.height + 15) & !15;
        (padded_width, padded_height)
    }

    /// Check if this frame has alpha
    pub fn has_alpha(&self) -> bool {
        self.alpha_plane.is_some()
    }

    /// Create a frame from raw planar data
    pub fn from_planes(
        y_plane: &[i16],
        cb_plane: &[i16],
        cr_plane: &[i16],
        alpha_plane: Option<&[u16]>,
        width: u16,
        height: u16,
        profile: ProResProfile,
        bit_depth: BitDepth,
    ) -> Result<Self> {
        let width = width as u32;
        let height = height as u32;

        // Round up to macroblock boundaries
        let padded_width = (width + 15) & !15;
        let padded_height = (height + 15) & !15;

        let chroma_format = if profile.is_444() {
            ChromaFormat::YUV444
        } else {
            ChromaFormat::YUV422
        };

        let chroma_h_shift = chroma_format.chroma_h_shift();
        let chroma_width = padded_width >> chroma_h_shift;

        let expected_y_size = (padded_width * padded_height) as usize;
        let expected_chroma_size = (chroma_width * padded_height) as usize;

        // Validate input sizes - allow for unpadded input too
        let min_y_size = (width * height) as usize;
        let min_chroma_size = ((width >> chroma_h_shift) * height) as usize;

        if y_plane.len() < min_y_size {
            return Err(ProResError::InvalidHeader(format!(
                "Y plane too small: {} < {}",
                y_plane.len(),
                min_y_size
            )));
        }

        if cb_plane.len() < min_chroma_size {
            return Err(ProResError::InvalidHeader(format!(
                "Cb plane too small: {} < {}",
                cb_plane.len(),
                min_chroma_size
            )));
        }

        if cr_plane.len() < min_chroma_size {
            return Err(ProResError::InvalidHeader(format!(
                "Cr plane too small: {} < {}",
                cr_plane.len(),
                min_chroma_size
            )));
        }

        // Copy data to padded buffers
        let mut y_data = vec![0i16; expected_y_size];
        let mut cb_data = vec![0i16; expected_chroma_size];
        let mut cr_data = vec![0i16; expected_chroma_size];

        // Copy Y plane row by row
        let input_y_stride = width as usize;
        let output_y_stride = padded_width as usize;
        for row in 0..height as usize {
            let src_start = row * input_y_stride;
            let dst_start = row * output_y_stride;
            let src_end = (src_start + width as usize).min(y_plane.len());
            let copy_len = src_end - src_start;
            y_data[dst_start..dst_start + copy_len].copy_from_slice(&y_plane[src_start..src_end]);
        }

        // Copy Cb/Cr planes row by row
        let input_chroma_width = width as usize >> chroma_h_shift;
        let output_chroma_stride = chroma_width as usize;
        for row in 0..height as usize {
            let src_start = row * input_chroma_width;
            let dst_start = row * output_chroma_stride;
            let src_end = (src_start + input_chroma_width).min(cb_plane.len());
            let copy_len = src_end - src_start;
            cb_data[dst_start..dst_start + copy_len].copy_from_slice(&cb_plane[src_start..src_end]);

            let src_end = (src_start + input_chroma_width).min(cr_plane.len());
            let copy_len = src_end - src_start;
            cr_data[dst_start..dst_start + copy_len].copy_from_slice(&cr_plane[src_start..src_end]);
        }

        // Handle alpha plane
        let alpha_data = if let Some(alpha) = alpha_plane {
            if alpha.len() < min_y_size {
                return Err(ProResError::InvalidHeader(format!(
                    "Alpha plane too small: {} < {}",
                    alpha.len(),
                    min_y_size
                )));
            }
            let mut alpha_buf = vec![0u16; expected_y_size];
            for row in 0..height as usize {
                let src_start = row * input_y_stride;
                let dst_start = row * output_y_stride;
                let src_end = (src_start + width as usize).min(alpha.len());
                let copy_len = src_end - src_start;
                alpha_buf[dst_start..dst_start + copy_len].copy_from_slice(&alpha[src_start..src_end]);
            }
            Some(alpha_buf)
        } else {
            None
        };

        Ok(ProResFrame {
            width,
            height,
            profile,
            bit_depth,
            chroma_format,
            y_plane: y_data,
            cb_plane: cb_data,
            cr_plane: cr_data,
            alpha_plane: alpha_data,
            y_stride: padded_width,
            chroma_stride: chroma_width,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_header_insufficient_data() {
        let data = [0u8; 10];
        let result = FrameHeader::parse(&data);
        assert!(matches!(result, Err(ProResError::InsufficientData { .. })));
    }

    #[test]
    fn test_frame_header_invalid_signature() {
        let mut data = vec![0u8; 200];
        // Frame size
        data[0] = 0;
        data[1] = 0;
        data[2] = 0;
        data[3] = 200;
        // Invalid signature
        data[4] = b'x';
        data[5] = b'y';
        data[6] = b'z';
        data[7] = b'w';

        let result = FrameHeader::parse(&data);
        assert!(matches!(result, Err(ProResError::InvalidSignature(_))));
    }

    #[test]
    fn test_prores_frame_creation() {
        // Create a mock header
        let header = FrameHeader {
            frame_size: 1000,
            profile: ProResProfile::HQ,
            width: 1920,
            height: 1080,
            chroma_format: ChromaFormat::YUV422,
            interlace_mode: InterlaceMode::Progressive,
            bit_depth: BitDepth::Bit10,
            color_primaries: ColorPrimaries::BT709,
            transfer_characteristic: TransferCharacteristic::BT709,
            matrix_coefficients: MatrixCoefficients::BT709,
            alpha_info: 0,
            luma_quant_matrix: [16; 64],
            chroma_quant_matrix: [16; 64],
            slices_per_row: 8,
            slice_rows: 68,
            slice_offsets: vec![0, 100, 200],
            header_size: 148,
        };

        let frame = ProResFrame::new(&header);

        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert_eq!(frame.profile, ProResProfile::HQ);
        assert!(!frame.has_alpha());

        // Check padded dimensions (1920 is already multiple of 16, 1080 rounds to 1088)
        let (pw, ph) = frame.padded_dimensions();
        assert_eq!(pw, 1920);
        assert_eq!(ph, 1088);
    }

    #[test]
    fn test_prores_frame_with_alpha() {
        let header = FrameHeader {
            frame_size: 1000,
            profile: ProResProfile::P4444,
            width: 1920,
            height: 1080,
            chroma_format: ChromaFormat::YUV444,
            interlace_mode: InterlaceMode::Progressive,
            bit_depth: BitDepth::Bit10,
            color_primaries: ColorPrimaries::BT709,
            transfer_characteristic: TransferCharacteristic::BT709,
            matrix_coefficients: MatrixCoefficients::BT709,
            alpha_info: 2,
            luma_quant_matrix: [16; 64],
            chroma_quant_matrix: [16; 64],
            slices_per_row: 8,
            slice_rows: 68,
            slice_offsets: vec![0, 100, 200],
            header_size: 148,
        };

        let frame = ProResFrame::new(&header);

        assert!(frame.has_alpha());
        assert!(frame.alpha_plane.is_some());
    }
}
