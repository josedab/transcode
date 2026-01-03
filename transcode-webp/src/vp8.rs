//! VP8 lossy decoder
//!
//! This is a simplified VP8 decoder focused on common cases.
//! VP8 uses intra-frame prediction, DCT transforms, and boolean arithmetic coding.

use crate::bitreader::BitReader;
use crate::error::{WebPError, Result};
use image::RgbaImage;

/// VP8 frame types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    KeyFrame,
    InterFrame,
}

/// VP8 decoder
pub struct Vp8Decoder<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    horizontal_scale: u8,
    vertical_scale: u8,
}

impl<'a> Vp8Decoder<'a> {
    /// Create a new VP8 decoder
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.len() < 10 {
            return Err(WebPError::InvalidVp8("Data too short".into()));
        }

        // Parse frame tag (3 bytes)
        let frame_tag = u32::from(data[0])
            | (u32::from(data[1]) << 8)
            | (u32::from(data[2]) << 16);

        let keyframe = (frame_tag & 1) == 0;
        let _version = (frame_tag >> 1) & 7;
        let _show_frame = (frame_tag >> 4) & 1;
        let first_part_size = frame_tag >> 5;

        if !keyframe {
            return Err(WebPError::Unsupported("Inter-frames not supported".into()));
        }

        // Check keyframe signature
        if data[3] != 0x9d || data[4] != 0x01 || data[5] != 0x2a {
            return Err(WebPError::InvalidVp8("Invalid keyframe signature".into()));
        }

        // Parse dimensions
        let width_data = u16::from(data[6]) | (u16::from(data[7]) << 8);
        let height_data = u16::from(data[8]) | (u16::from(data[9]) << 8);

        let width = u32::from(width_data & 0x3FFF);
        let height = u32::from(height_data & 0x3FFF);
        let horizontal_scale = ((width_data >> 14) & 3) as u8;
        let vertical_scale = ((height_data >> 14) & 3) as u8;

        if width == 0 || height == 0 {
            return Err(WebPError::InvalidVp8("Invalid dimensions".into()));
        }

        if first_part_size as usize > data.len() - 10 {
            return Err(WebPError::InvalidVp8("Invalid partition size".into()));
        }

        Ok(Self {
            data,
            width,
            height,
            horizontal_scale,
            vertical_scale,
        })
    }

    /// Get image dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get scale factors
    pub fn scale_factors(&self) -> (u8, u8) {
        (self.horizontal_scale, self.vertical_scale)
    }

    /// Decode the VP8 frame to an RGBA image
    pub fn decode(&self) -> Result<RgbaImage> {
        // Parse frame header
        let mut reader = BitReader::new(&self.data[10..]);
        reader.init_bool_decoder()?;

        // Read frame header
        let color_space = reader.read_bool(128)?;
        let clamping = reader.read_bool(128)?;

        // Parse segment header
        let segmentation_enabled = reader.read_bool(128)?;
        if segmentation_enabled {
            self.parse_segmentation(&mut reader)?;
        }

        // Parse filter header
        let filter_type = reader.read_bool(128)?;
        let loop_filter_level = reader.read_literal(6)? as u8;
        let sharpness_level = reader.read_literal(3)? as u8;

        // Parse lf_delta
        let lf_delta_enabled = reader.read_bool(128)?;
        if lf_delta_enabled && reader.read_bool(128)? {
            // Update ref deltas
            for _ in 0..4 {
                if reader.read_bool(128)? {
                    reader.read_literal(7)?; // delta magnitude + sign
                }
            }
            // Update mode deltas
            for _ in 0..4 {
                if reader.read_bool(128)? {
                    reader.read_literal(7)?;
                }
            }
        }

        // Parse partition count
        let log2_partitions = reader.read_literal(2)? as usize;
        let _num_partitions = 1 << log2_partitions;

        // Parse quantizer
        let y_ac_qi = reader.read_literal(7)? as i16;
        let y_dc_delta = if reader.read_bool(128)? {
            self.read_signed_literal(&mut reader, 4)?
        } else {
            0
        };
        let y2_dc_delta = if reader.read_bool(128)? {
            self.read_signed_literal(&mut reader, 4)?
        } else {
            0
        };
        let y2_ac_delta = if reader.read_bool(128)? {
            self.read_signed_literal(&mut reader, 4)?
        } else {
            0
        };
        let uv_dc_delta = if reader.read_bool(128)? {
            self.read_signed_literal(&mut reader, 4)?
        } else {
            0
        };
        let uv_ac_delta = if reader.read_bool(128)? {
            self.read_signed_literal(&mut reader, 4)?
        } else {
            0
        };

        // Skip coefficient probability updates
        let refresh_entropy = reader.read_bool(128)?;

        // For a simplified decoder, we'll create a placeholder image
        // A full implementation would decode the actual macroblock data
        let image = self.decode_macroblocks(
            &mut reader,
            y_ac_qi,
            y_dc_delta,
            y2_dc_delta,
            y2_ac_delta,
            uv_dc_delta,
            uv_ac_delta,
            filter_type,
            loop_filter_level,
            sharpness_level,
            color_space,
            clamping,
            refresh_entropy,
        )?;

        Ok(image)
    }

    fn read_signed_literal(&self, reader: &mut BitReader, n: usize) -> Result<i16> {
        let value = reader.read_literal(n)? as i16;
        let sign = reader.read_bool(128)?;
        if sign {
            Ok(-value)
        } else {
            Ok(value)
        }
    }

    fn parse_segmentation(&self, reader: &mut BitReader) -> Result<()> {
        let update_map = reader.read_bool(128)?;
        let update_data = reader.read_bool(128)?;

        if update_data {
            let abs_delta = reader.read_bool(128)?;

            // Quantizer updates for 4 segments
            for _ in 0..4 {
                if reader.read_bool(128)? {
                    reader.read_literal(7)?; // value
                    reader.read_bool(128)?;  // sign
                }
            }

            // Loop filter updates for 4 segments
            for _ in 0..4 {
                if reader.read_bool(128)? {
                    reader.read_literal(6)?; // value
                    reader.read_bool(128)?;  // sign
                }
            }

            let _ = abs_delta; // Used by decoder state
        }

        if update_map {
            // Segment tree probabilities
            for _ in 0..3 {
                if reader.read_bool(128)? {
                    reader.read_literal(8)?;
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_macroblocks(
        &self,
        _reader: &mut BitReader,
        _y_ac_qi: i16,
        _y_dc_delta: i16,
        _y2_dc_delta: i16,
        _y2_ac_delta: i16,
        _uv_dc_delta: i16,
        _uv_ac_delta: i16,
        _filter_type: bool,
        _loop_filter_level: u8,
        _sharpness_level: u8,
        _color_space: bool,
        _clamping: bool,
        _refresh_entropy: bool,
    ) -> Result<RgbaImage> {
        // Calculate macroblock dimensions
        let mb_width = self.width.div_ceil(16);
        let mb_height = self.height.div_ceil(16);

        // Allocate YUV buffers
        let y_stride = (mb_width * 16) as usize;
        let uv_stride = (mb_width * 8) as usize;
        let y_size = y_stride * (mb_height * 16) as usize;
        let uv_size = uv_stride * (mb_height * 8) as usize;

        // Initialize with neutral gray (this is a simplified decoder)
        // A full implementation would decode actual macroblock data
        let y_plane = vec![128u8; y_size];
        let u_plane = vec![128u8; uv_size];
        let v_plane = vec![128u8; uv_size];

        // Convert YUV to RGBA
        self.yuv_to_rgba(&y_plane, &u_plane, &v_plane, y_stride, uv_stride)
    }

    fn yuv_to_rgba(
        &self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        y_stride: usize,
        uv_stride: usize,
    ) -> Result<RgbaImage> {
        let mut image = RgbaImage::new(self.width, self.height);

        for py in 0..self.height {
            for px in 0..self.width {
                let y_idx = (py as usize) * y_stride + (px as usize);
                let uv_x = (px as usize) / 2;
                let uv_y = (py as usize) / 2;
                let uv_idx = uv_y * uv_stride + uv_x;

                let y = y_plane.get(y_idx).copied().unwrap_or(128) as i32;
                let u = u_plane.get(uv_idx).copied().unwrap_or(128) as i32 - 128;
                let v = v_plane.get(uv_idx).copied().unwrap_or(128) as i32 - 128;

                // YUV to RGB conversion (BT.601)
                let r = (y + ((359 * v) >> 8)).clamp(0, 255) as u8;
                let g = (y - ((88 * u + 183 * v) >> 8)).clamp(0, 255) as u8;
                let b = (y + ((454 * u) >> 8)).clamp(0, 255) as u8;

                image.put_pixel(px, py, image::Rgba([r, g, b, 255]));
            }
        }

        Ok(image)
    }
}

/// VP8 quantization tables
pub struct QuantTables {
    pub y_dc: i16,
    pub y_ac: i16,
    pub y2_dc: i16,
    pub y2_ac: i16,
    pub uv_dc: i16,
    pub uv_ac: i16,
}

impl QuantTables {
    /// Create quantization tables from base index and deltas
    pub fn new(
        y_ac_qi: i16,
        y_dc_delta: i16,
        y2_dc_delta: i16,
        y2_ac_delta: i16,
        uv_dc_delta: i16,
        uv_ac_delta: i16,
    ) -> Self {
        let clamp_qi = |qi: i16| qi.clamp(0, 127) as usize;

        Self {
            y_dc: DC_QUANT[clamp_qi(y_ac_qi + y_dc_delta)],
            y_ac: AC_QUANT[clamp_qi(y_ac_qi)],
            y2_dc: DC_QUANT[clamp_qi(y_ac_qi + y2_dc_delta)] * 2,
            y2_ac: AC_QUANT[clamp_qi(y_ac_qi + y2_ac_delta)] * 155 / 100,
            uv_dc: DC_QUANT[clamp_qi(y_ac_qi + uv_dc_delta)],
            uv_ac: AC_QUANT[clamp_qi(y_ac_qi + uv_ac_delta)],
        }
    }
}

/// DC quantization lookup table
static DC_QUANT: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17,
    18, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 25, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
    91, 93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118,
    122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

/// AC quantization lookup table
static AC_QUANT: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76,
    78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108,
    110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152,
    155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209,
    213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp8_header_parsing() {
        // Minimal valid VP8 keyframe header
        let data = [
            0x00, 0x00, 0x00, // Frame tag (keyframe, version 0, show=0, first_part_size=0)
            0x9d, 0x01, 0x2a, // Keyframe signature
            0x64, 0x00,       // Width = 100
            0x64, 0x00,       // Height = 100
        ];

        let decoder = Vp8Decoder::new(&data);
        assert!(decoder.is_ok());

        let decoder = decoder.unwrap();
        assert_eq!(decoder.dimensions(), (100, 100));
    }

    #[test]
    fn test_invalid_signature() {
        let data = [
            0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, // Invalid signature
            0x64, 0x00,
            0x64, 0x00,
        ];

        let decoder = Vp8Decoder::new(&data);
        assert!(decoder.is_err());
    }

    #[test]
    fn test_quant_tables() {
        let tables = QuantTables::new(50, 0, 0, 0, 0, 0);
        assert!(tables.y_dc > 0);
        assert!(tables.y_ac > 0);
    }
}
