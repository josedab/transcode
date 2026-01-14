//! JPEG decoder implementation.

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat, ColorSpace};
use super::huffman::{HuffmanTable, category_decode};
use super::dct::fast_inverse_dct;
use super::ZIGZAG;

/// JPEG image information.
#[derive(Debug, Clone, Default)]
pub struct JpegInfo {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Number of components.
    pub components: u8,
    /// Bits per sample.
    pub bits_per_sample: u8,
    /// Is progressive JPEG.
    pub progressive: bool,
    /// Has JFIF header.
    pub has_jfif: bool,
    /// Has EXIF data.
    pub has_exif: bool,
    /// JFIF version.
    pub jfif_version: (u8, u8),
}

/// JPEG component info.
#[derive(Debug, Clone, Default)]
struct Component {
    id: u8,
    h_sampling: u8,
    v_sampling: u8,
    quant_table: u8,
    dc_table: u8,
    ac_table: u8,
    dc_pred: i16,
}

/// JPEG decoder.
#[derive(Debug)]
pub struct JpegDecoder {
    /// Image information.
    info: JpegInfo,
    /// Quantization tables.
    quant_tables: [[u8; 64]; 4],
    /// Huffman tables (DC).
    dc_tables: [Option<HuffmanTable>; 4],
    /// Huffman tables (AC).
    ac_tables: [Option<HuffmanTable>; 4],
    /// Component information.
    components: Vec<Component>,
    /// Restart interval.
    restart_interval: u16,
}

impl JpegDecoder {
    /// Create a new JPEG decoder.
    pub fn new() -> Self {
        Self {
            info: JpegInfo::default(),
            quant_tables: [[16; 64]; 4],
            dc_tables: [None, None, None, None],
            ac_tables: [None, None, None, None],
            components: Vec::new(),
            restart_interval: 0,
        }
    }

    /// Get image information (after decode).
    pub fn info(&self) -> &JpegInfo {
        &self.info
    }

    /// Decode JPEG data to an image.
    pub fn decode(&mut self, data: &[u8]) -> Result<Image> {
        if data.len() < 2 {
            return Err(ImageError::TruncatedData {
                expected: 2,
                actual: data.len(),
            });
        }

        // Check SOI marker
        if data[0] != 0xFF || data[1] != 0xD8 {
            return Err(ImageError::InvalidHeader("Not a JPEG file".into()));
        }

        let mut offset = 2;

        // Parse markers
        while offset < data.len() - 1 {
            if data[offset] != 0xFF {
                offset += 1;
                continue;
            }

            let marker = data[offset + 1];
            offset += 2;

            match marker {
                0xD8 => {} // SOI
                0xD9 => break, // EOI
                0x00 => {} // Byte stuffing
                0xD0..=0xD7 => {} // RST markers
                0xE0 => offset = self.parse_app0(data, offset)?,
                0xE1 => offset = self.parse_app1(data, offset)?,
                0xDB => offset = self.parse_dqt(data, offset)?,
                0xC0 => offset = self.parse_sof0(data, offset)?,
                0xC2 => offset = self.parse_sof2(data, offset)?,
                0xC4 => offset = self.parse_dht(data, offset)?,
                0xDA => {
                    return self.decode_scan(data, offset);
                }
                0xDD => offset = self.parse_dri(data, offset)?,
                0xFE => offset = self.skip_segment(data, offset)?, // COM
                _ if (0xE0..=0xEF).contains(&marker) => {
                    offset = self.skip_segment(data, offset)?;
                }
                _ => {
                    offset = self.skip_segment(data, offset)?;
                }
            }
        }

        Err(ImageError::DecoderError("No image data found".into()))
    }

    /// Skip a segment.
    fn skip_segment(&self, data: &[u8], offset: usize) -> Result<usize> {
        if offset + 2 > data.len() {
            return Err(ImageError::TruncatedData {
                expected: offset + 2,
                actual: data.len(),
            });
        }

        let length = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        Ok(offset + length)
    }

    /// Parse APP0 (JFIF).
    fn parse_app0(&mut self, data: &[u8], offset: usize) -> Result<usize> {
        let length = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;

        if length >= 14 && &data[offset + 2..offset + 7] == b"JFIF\0" {
            self.info.has_jfif = true;
            self.info.jfif_version = (data[offset + 7], data[offset + 8]);
        }

        Ok(offset + length)
    }

    /// Parse APP1 (EXIF).
    fn parse_app1(&mut self, data: &[u8], offset: usize) -> Result<usize> {
        let length = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;

        if length >= 8 && &data[offset + 2..offset + 6] == b"Exif" {
            self.info.has_exif = true;
        }

        Ok(offset + length)
    }

    /// Parse DQT (quantization table).
    fn parse_dqt(&mut self, data: &[u8], offset: usize) -> Result<usize> {
        let length = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        let mut pos = offset + 2;
        let end = offset + length;

        while pos < end {
            let info = data[pos];
            let precision = (info >> 4) & 0x0F;
            let table_id = (info & 0x0F) as usize;
            pos += 1;

            if table_id > 3 {
                return Err(ImageError::InvalidData("Invalid quant table ID".into()));
            }

            let table_size = if precision == 0 { 64 } else { 128 };
            if pos + table_size > data.len() {
                return Err(ImageError::TruncatedData {
                    expected: pos + table_size,
                    actual: data.len(),
                });
            }

            if precision == 0 {
                for i in 0..64 {
                    self.quant_tables[table_id][ZIGZAG[i]] = data[pos + i];
                }
            } else {
                for i in 0..64 {
                    let val = u16::from_be_bytes([data[pos + i * 2], data[pos + i * 2 + 1]]);
                    self.quant_tables[table_id][ZIGZAG[i]] = val.min(255) as u8;
                }
            }

            pos += table_size;
        }

        Ok(end)
    }

    /// Parse SOF0 (baseline).
    fn parse_sof0(&mut self, data: &[u8], offset: usize) -> Result<usize> {
        self.parse_sof(data, offset, false)
    }

    /// Parse SOF2 (progressive).
    fn parse_sof2(&mut self, data: &[u8], offset: usize) -> Result<usize> {
        self.parse_sof(data, offset, true)
    }

    /// Parse SOF (Start of Frame).
    fn parse_sof(&mut self, data: &[u8], offset: usize, progressive: bool) -> Result<usize> {
        let length = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;

        self.info.bits_per_sample = data[offset + 2];
        self.info.height = u16::from_be_bytes([data[offset + 3], data[offset + 4]]) as u32;
        self.info.width = u16::from_be_bytes([data[offset + 5], data[offset + 6]]) as u32;
        self.info.components = data[offset + 7];
        self.info.progressive = progressive;

        if self.info.width == 0 || self.info.height == 0 {
            return Err(ImageError::InvalidDimensions {
                width: self.info.width,
                height: self.info.height,
            });
        }

        self.components.clear();
        let mut pos = offset + 8;

        for _ in 0..self.info.components {
            let id = data[pos];
            let sampling = data[pos + 1];
            let quant = data[pos + 2];

            self.components.push(Component {
                id,
                h_sampling: (sampling >> 4) & 0x0F,
                v_sampling: sampling & 0x0F,
                quant_table: quant,
                dc_table: 0,
                ac_table: 0,
                dc_pred: 0,
            });

            pos += 3;
        }

        Ok(offset + length)
    }

    /// Parse DHT (Huffman table).
    fn parse_dht(&mut self, data: &[u8], offset: usize) -> Result<usize> {
        let length = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        let mut pos = offset + 2;
        let end = offset + length;

        while pos < end {
            let info = data[pos];
            let table_class = (info >> 4) & 0x0F;
            let table_id = (info & 0x0F) as usize;
            pos += 1;

            if table_id > 3 {
                return Err(ImageError::InvalidData("Invalid Huffman table ID".into()));
            }

            let mut table = HuffmanTable::new(table_class, table_id as u8);

            // Read BITS
            let mut total_codes = 0;
            for i in 1..=16 {
                table.bits[i] = data[pos + i - 1];
                total_codes += table.bits[i] as usize;
            }
            pos += 16;

            // Read HUFFVAL
            table.huffval = data[pos..pos + total_codes].to_vec();
            pos += total_codes;

            table.build_lookup();

            if table_class == 0 {
                self.dc_tables[table_id] = Some(table);
            } else {
                self.ac_tables[table_id] = Some(table);
            }
        }

        Ok(end)
    }

    /// Parse DRI (restart interval).
    fn parse_dri(&mut self, data: &[u8], offset: usize) -> Result<usize> {
        self.restart_interval = u16::from_be_bytes([data[offset + 2], data[offset + 3]]);
        Ok(offset + 4)
    }

    /// Decode scan data.
    fn decode_scan(&mut self, data: &[u8], offset: usize) -> Result<Image> {
        let length = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        let num_components = data[offset + 2];

        // Parse component selector
        let mut pos = offset + 3;
        for _i in 0..num_components as usize {
            let component_id = data[pos];
            let tables = data[pos + 1];

            // Find matching component
            for comp in &mut self.components {
                if comp.id == component_id {
                    comp.dc_table = (tables >> 4) & 0x0F;
                    comp.ac_table = tables & 0x0F;
                    break;
                }
            }

            pos += 2;
        }

        // Skip spectral selection and successive approximation
        pos = offset + length;

        // Decode entropy-coded data
        let image = self.decode_entropy(data, pos)?;
        Ok(image)
    }

    /// Decode entropy-coded data.
    fn decode_entropy(&mut self, data: &[u8], offset: usize) -> Result<Image> {
        let width = self.info.width as usize;
        let height = self.info.height as usize;

        // Allocate output image
        let format = if self.info.components == 1 {
            PixelFormat::Gray8
        } else {
            PixelFormat::Rgb8
        };

        let mut image = Image::new(self.info.width, self.info.height, format)?;
        image.set_color_space(ColorSpace::YCbCr);

        // Get max sampling factors
        let max_h = self.components.iter().map(|c| c.h_sampling).max().unwrap_or(1);
        let max_v = self.components.iter().map(|c| c.v_sampling).max().unwrap_or(1);

        let mcu_width = max_h as usize * 8;
        let mcu_height = max_v as usize * 8;
        let mcu_cols = width.div_ceil(mcu_width);
        let mcu_rows = height.div_ceil(mcu_height);

        // Reset DC predictors
        for comp in &mut self.components {
            comp.dc_pred = 0;
        }

        // Decode MCUs
        let mut bit_pos = offset * 8;
        let mut restart_count = 0;

        for mcu_row in 0..mcu_rows {
            for mcu_col in 0..mcu_cols {
                // Check restart interval
                if self.restart_interval > 0 {
                    restart_count += 1;
                    if restart_count > self.restart_interval as usize {
                        // Skip to next marker
                        bit_pos = self.find_next_marker(data, bit_pos / 8)? * 8;
                        restart_count = 0;
                        for comp in &mut self.components {
                            comp.dc_pred = 0;
                        }
                    }
                }

                // Decode each component in MCU
                for comp_idx in 0..self.components.len() {
                    let comp = &self.components[comp_idx];
                    let h_blocks = comp.h_sampling as usize;
                    let v_blocks = comp.v_sampling as usize;

                    for v in 0..v_blocks {
                        for h in 0..h_blocks {
                            // Decode 8x8 block
                            let block = self.decode_block(
                                data,
                                &mut bit_pos,
                                comp_idx,
                            )?;

                            // Write block to image
                            let x = mcu_col * mcu_width + h * 8;
                            let y = mcu_row * mcu_height + v * 8;

                            self.write_block(&mut image, &block, x, y, comp_idx);
                        }
                    }
                }
            }
        }

        // Convert YCbCr to RGB if needed
        if self.info.components == 3 {
            self.ycbcr_to_rgb(&mut image);
        }

        Ok(image)
    }

    /// Decode an 8x8 block.
    fn decode_block(
        &mut self,
        data: &[u8],
        bit_pos: &mut usize,
        comp_idx: usize,
    ) -> Result<[i16; 64]> {
        let mut block = [0i16; 64];

        let comp = &self.components[comp_idx];
        let dc_table = self.dc_tables[comp.dc_table as usize]
            .as_ref()
            .ok_or_else(|| ImageError::DecoderError("Missing DC table".into()))?;
        let ac_table = self.ac_tables[comp.ac_table as usize]
            .as_ref()
            .ok_or_else(|| ImageError::DecoderError("Missing AC table".into()))?;
        let quant = &self.quant_tables[comp.quant_table as usize];

        // Decode DC coefficient
        let mut get_bit = || self.get_bit(data, bit_pos);
        let dc_category = dc_table.decode(&mut get_bit)?;
        let dc_bits = self.get_bits(data, bit_pos, dc_category)?;
        let dc_diff = category_decode(dc_category, dc_bits);

        // Update DC prediction
        let dc_pred = self.components[comp_idx].dc_pred;
        let dc = dc_pred + dc_diff;
        self.components[comp_idx].dc_pred = dc;
        block[0] = dc * quant[0] as i16;

        // Decode AC coefficients
        let mut k = 1;
        while k < 64 {
            let mut get_bit = || self.get_bit(data, bit_pos);
            let rs = ac_table.decode(&mut get_bit)?;

            if rs == 0 {
                // EOB
                break;
            }

            let rrrr = rs >> 4;
            let ssss = rs & 0x0F;

            k += rrrr as usize;

            if k >= 64 {
                break;
            }

            if ssss > 0 {
                let ac_bits = self.get_bits(data, bit_pos, ssss)?;
                let ac = category_decode(ssss, ac_bits);
                block[ZIGZAG[k]] = ac * quant[k] as i16;
            }

            k += 1;
        }

        // Inverse DCT
        fast_inverse_dct(&mut block);

        Ok(block)
    }

    /// Get a single bit.
    fn get_bit(&self, data: &[u8], bit_pos: &mut usize) -> Result<u8> {
        let byte_pos = *bit_pos / 8;
        let bit_offset = 7 - (*bit_pos % 8);

        if byte_pos >= data.len() {
            return Err(ImageError::TruncatedData {
                expected: byte_pos + 1,
                actual: data.len(),
            });
        }

        let byte = data[byte_pos];
        *bit_pos += 1;

        // Skip stuffed bytes
        if byte == 0xFF && byte_pos + 1 < data.len() && data[byte_pos + 1] == 0x00 {
            *bit_pos += 8; // Skip the 0x00
        }

        Ok((byte >> bit_offset) & 1)
    }

    /// Get multiple bits.
    fn get_bits(&self, data: &[u8], bit_pos: &mut usize, count: u8) -> Result<u16> {
        let mut value: u16 = 0;
        for _ in 0..count {
            value = (value << 1) | self.get_bit(data, bit_pos)? as u16;
        }
        Ok(value)
    }

    /// Find next marker.
    fn find_next_marker(&self, data: &[u8], offset: usize) -> Result<usize> {
        let mut pos = offset;
        while pos < data.len() - 1 {
            if data[pos] == 0xFF && data[pos + 1] >= 0xD0 && data[pos + 1] <= 0xD7 {
                return Ok(pos + 2);
            }
            pos += 1;
        }
        Err(ImageError::DecoderError("Restart marker not found".into()))
    }

    /// Write decoded block to image.
    fn write_block(&self, image: &mut Image, block: &[i16; 64], x: usize, y: usize, comp: usize) {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let channels = if self.info.components == 1 { 1 } else { 3 };

        for by in 0..8 {
            for bx in 0..8 {
                let px = x + bx;
                let py = y + by;

                if px < width && py < height {
                    let value = block[by * 8 + bx].clamp(0, 255) as u8;
                    let offset = (py * width + px) * channels + comp;

                    if offset < image.data().len() {
                        image.data_mut()[offset] = value;
                    }
                }
            }
        }
    }

    /// Convert YCbCr to RGB.
    fn ycbcr_to_rgb(&self, image: &mut Image) {
        let data = image.data_mut();
        let pixels = data.len() / 3;

        for i in 0..pixels {
            let offset = i * 3;
            let y = data[offset] as f32;
            let cb = data[offset + 1] as f32 - 128.0;
            let cr = data[offset + 2] as f32 - 128.0;

            let r = (y + 1.402 * cr).round().clamp(0.0, 255.0) as u8;
            let g = (y - 0.344136 * cb - 0.714136 * cr).round().clamp(0.0, 255.0) as u8;
            let b = (y + 1.772 * cb).round().clamp(0.0, 255.0) as u8;

            data[offset] = r;
            data[offset + 1] = g;
            data[offset + 2] = b;
        }
    }
}

impl Default for JpegDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = JpegDecoder::new();
        assert_eq!(decoder.info().width, 0);
        assert_eq!(decoder.info().height, 0);
    }

    #[test]
    fn test_invalid_header() {
        let mut decoder = JpegDecoder::new();
        let result = decoder.decode(&[0x00, 0x00]);
        assert!(matches!(result, Err(ImageError::InvalidHeader(_))));
    }

    #[test]
    fn test_truncated_data() {
        let mut decoder = JpegDecoder::new();
        let result = decoder.decode(&[0xFF]);
        assert!(matches!(result, Err(ImageError::TruncatedData { .. })));
    }
}
