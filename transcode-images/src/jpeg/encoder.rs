//! JPEG encoder implementation.

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};
use super::{ChromaSubsampling, ZIGZAG, quantization};
use super::dct::fast_forward_dct;
use super::huffman::{
    HuffmanEncoder, dc_luminance_table, dc_chrominance_table,
    ac_luminance_table, ac_chrominance_table, category_encode,
};

/// JPEG encoder configuration.
#[derive(Debug, Clone)]
pub struct JpegConfig {
    /// Quality (1-100).
    pub quality: u8,
    /// Chroma subsampling.
    pub subsampling: ChromaSubsampling,
    /// Include JFIF APP0 marker.
    pub jfif: bool,
    /// Optimize Huffman tables.
    pub optimize_coding: bool,
    /// Progressive encoding.
    pub progressive: bool,
}

impl Default for JpegConfig {
    fn default() -> Self {
        Self {
            quality: 85,
            subsampling: ChromaSubsampling::Yuv420,
            jfif: true,
            optimize_coding: false,
            progressive: false,
        }
    }
}

/// JPEG encoder.
pub struct JpegEncoder {
    config: JpegConfig,
    lum_quant: [u8; 64],
    chr_quant: [u8; 64],
    dc_lum_encoder: HuffmanEncoder,
    ac_lum_encoder: HuffmanEncoder,
    dc_chr_encoder: HuffmanEncoder,
    ac_chr_encoder: HuffmanEncoder,
}

impl JpegEncoder {
    /// Create a new JPEG encoder.
    pub fn new(config: JpegConfig) -> Self {
        let lum_quant = quantization::scale_table(&quantization::LUMINANCE_50, config.quality);
        let chr_quant = quantization::scale_table(&quantization::CHROMINANCE_50, config.quality);

        Self {
            config,
            lum_quant,
            chr_quant,
            dc_lum_encoder: HuffmanEncoder::from_table(&dc_luminance_table()),
            ac_lum_encoder: HuffmanEncoder::from_table(&ac_luminance_table()),
            dc_chr_encoder: HuffmanEncoder::from_table(&dc_chrominance_table()),
            ac_chr_encoder: HuffmanEncoder::from_table(&ac_chrominance_table()),
        }
    }

    /// Encode an image to JPEG.
    pub fn encode(&self, image: &Image) -> Result<Vec<u8>> {
        // Convert to RGB if needed
        let rgb_image = if image.format() == PixelFormat::Rgb8 {
            image.clone()
        } else if image.format() == PixelFormat::Rgba8 {
            image.convert(PixelFormat::Rgb8)?
        } else if image.format() == PixelFormat::Gray8 {
            image.clone()
        } else {
            return Err(ImageError::UnsupportedFormat(
                format!("Cannot encode {:?} to JPEG", image.format())
            ));
        };

        let is_grayscale = rgb_image.format() == PixelFormat::Gray8;
        let width = rgb_image.width() as usize;
        let height = rgb_image.height() as usize;

        // Convert to YCbCr
        let (y_plane, cb_plane, cr_plane) = if is_grayscale {
            let y = rgb_image.data().to_vec();
            (y, vec![], vec![])
        } else {
            self.rgb_to_ycbcr(&rgb_image)
        };

        let mut output = Vec::new();
        let mut bit_buffer = BitWriter::new();

        // Write SOI
        output.extend_from_slice(&[0xFF, 0xD8]);

        // Write JFIF APP0 if enabled
        if self.config.jfif {
            self.write_jfif(&mut output);
        }

        // Write quantization tables
        self.write_dqt(&mut output, is_grayscale);

        // Write SOF0 (baseline)
        self.write_sof0(&mut output, width, height, is_grayscale);

        // Write Huffman tables
        self.write_dht(&mut output, is_grayscale);

        // Write SOS
        self.write_sos(&mut output, is_grayscale);

        // Encode image data
        self.encode_scan(
            &y_plane, &cb_plane, &cr_plane,
            width, height, is_grayscale,
            &mut bit_buffer,
        )?;

        // Flush bit buffer and add to output
        bit_buffer.flush();
        output.extend_from_slice(&bit_buffer.bytes);

        // Write EOI
        output.extend_from_slice(&[0xFF, 0xD9]);

        Ok(output)
    }

    /// Convert RGB to YCbCr.
    fn rgb_to_ycbcr(&self, image: &Image) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let data = image.data();

        let mut y_plane = vec![0u8; width * height];
        let mut cb_plane = vec![0u8; width * height];
        let mut cr_plane = vec![0u8; width * height];

        for row in 0..height {
            for col in 0..width {
                let offset = (row * width + col) * 3;
                let r = data[offset] as f32;
                let g = data[offset + 1] as f32;
                let b = data[offset + 2] as f32;

                // ITU-R BT.601 conversion
                let y = 0.299 * r + 0.587 * g + 0.114 * b;
                let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
                let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;

                let idx = row * width + col;
                y_plane[idx] = y.clamp(0.0, 255.0) as u8;
                cb_plane[idx] = cb.clamp(0.0, 255.0) as u8;
                cr_plane[idx] = cr.clamp(0.0, 255.0) as u8;
            }
        }

        (y_plane, cb_plane, cr_plane)
    }

    /// Downsample chroma plane for subsampling.
    fn downsample_chroma(&self, plane: &[u8], width: usize, height: usize) -> Vec<u8> {
        match self.config.subsampling {
            ChromaSubsampling::Yuv444 | ChromaSubsampling::Gray => plane.to_vec(),
            ChromaSubsampling::Yuv422 => {
                // Horizontal 2:1
                let new_width = width.div_ceil(2);
                let mut result = vec![0u8; new_width * height];
                for y in 0..height {
                    for x in 0..new_width {
                        let x1 = x * 2;
                        let x2 = (x * 2 + 1).min(width - 1);
                        let avg = (plane[y * width + x1] as u16 + plane[y * width + x2] as u16) / 2;
                        result[y * new_width + x] = avg as u8;
                    }
                }
                result
            }
            ChromaSubsampling::Yuv420 => {
                // Horizontal and vertical 2:1
                let new_width = width.div_ceil(2);
                let new_height = height.div_ceil(2);
                let mut result = vec![0u8; new_width * new_height];
                for y in 0..new_height {
                    for x in 0..new_width {
                        let y1 = y * 2;
                        let y2 = (y * 2 + 1).min(height - 1);
                        let x1 = x * 2;
                        let x2 = (x * 2 + 1).min(width - 1);
                        let sum = plane[y1 * width + x1] as u32
                            + plane[y1 * width + x2] as u32
                            + plane[y2 * width + x1] as u32
                            + plane[y2 * width + x2] as u32;
                        result[y * new_width + x] = (sum / 4) as u8;
                    }
                }
                result
            }
        }
    }

    /// Write JFIF APP0 marker.
    fn write_jfif(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&[0xFF, 0xE0]); // APP0
        output.extend_from_slice(&[0x00, 0x10]); // Length = 16
        output.extend_from_slice(b"JFIF\0"); // Identifier
        output.extend_from_slice(&[0x01, 0x01]); // Version 1.1
        output.push(0x00); // Units = no units
        output.extend_from_slice(&[0x00, 0x01]); // X density = 1
        output.extend_from_slice(&[0x00, 0x01]); // Y density = 1
        output.extend_from_slice(&[0x00, 0x00]); // No thumbnail
    }

    /// Write DQT (quantization tables).
    fn write_dqt(&self, output: &mut Vec<u8>, is_grayscale: bool) {
        // Luminance table
        output.extend_from_slice(&[0xFF, 0xDB]); // DQT
        let length = if is_grayscale { 67 } else { 132 };
        output.extend_from_slice(&[(length >> 8) as u8, length as u8]);

        // Table 0 (luminance)
        output.push(0x00); // Precision 8-bit, table ID 0
        for i in 0..64 {
            output.push(self.lum_quant[ZIGZAG[i]]);
        }

        // Table 1 (chrominance) - only if not grayscale
        if !is_grayscale {
            output.push(0x01); // Precision 8-bit, table ID 1
            for i in 0..64 {
                output.push(self.chr_quant[ZIGZAG[i]]);
            }
        }
    }

    /// Write SOF0 (start of frame - baseline).
    fn write_sof0(&self, output: &mut Vec<u8>, width: usize, height: usize, is_grayscale: bool) {
        output.extend_from_slice(&[0xFF, 0xC0]); // SOF0

        let num_components = if is_grayscale { 1 } else { 3 };
        let length = 8 + num_components * 3;
        output.extend_from_slice(&[(length >> 8) as u8, length as u8]);

        output.push(8); // Precision
        output.extend_from_slice(&[(height >> 8) as u8, height as u8]);
        output.extend_from_slice(&[(width >> 8) as u8, width as u8]);
        output.push(num_components as u8);

        if is_grayscale {
            output.extend_from_slice(&[1, 0x11, 0]); // Y: 1x1, table 0
        } else {
            let (h_factor, v_factor) = match self.config.subsampling {
                ChromaSubsampling::Yuv444 => (1, 1),
                ChromaSubsampling::Yuv422 => (2, 1),
                ChromaSubsampling::Yuv420 => (2, 2),
                ChromaSubsampling::Gray => (1, 1),
            };
            let y_sampling = (h_factor << 4) | v_factor;
            output.extend_from_slice(&[1, y_sampling, 0]); // Y
            output.extend_from_slice(&[2, 0x11, 1]); // Cb: 1x1, table 1
            output.extend_from_slice(&[3, 0x11, 1]); // Cr: 1x1, table 1
        }
    }

    /// Write DHT (Huffman tables).
    fn write_dht(&self, output: &mut Vec<u8>, is_grayscale: bool) {
        // DC luminance
        self.write_huffman_table(output, 0x00, &dc_luminance_table().bits, &dc_luminance_table().huffval);

        // AC luminance
        self.write_huffman_table(output, 0x10, &ac_luminance_table().bits, &ac_luminance_table().huffval);

        if !is_grayscale {
            // DC chrominance
            self.write_huffman_table(output, 0x01, &dc_chrominance_table().bits, &dc_chrominance_table().huffval);

            // AC chrominance
            self.write_huffman_table(output, 0x11, &ac_chrominance_table().bits, &ac_chrominance_table().huffval);
        }
    }

    /// Write a single Huffman table.
    fn write_huffman_table(&self, output: &mut Vec<u8>, tc_th: u8, bits: &[u8; 17], huffval: &[u8]) {
        output.extend_from_slice(&[0xFF, 0xC4]); // DHT

        let length = 2 + 1 + 16 + huffval.len();
        output.extend_from_slice(&[(length >> 8) as u8, length as u8]);

        output.push(tc_th);

        // Number of codes for each length
        for i in 1..=16 {
            output.push(bits[i]);
        }

        // Symbol values
        output.extend_from_slice(huffval);
    }

    /// Write SOS (start of scan).
    fn write_sos(&self, output: &mut Vec<u8>, is_grayscale: bool) {
        output.extend_from_slice(&[0xFF, 0xDA]); // SOS

        let num_components = if is_grayscale { 1 } else { 3 };
        let length = 6 + num_components * 2;
        output.extend_from_slice(&[(length >> 8) as u8, length as u8]);

        output.push(num_components as u8);

        if is_grayscale {
            output.extend_from_slice(&[1, 0x00]); // Y: DC table 0, AC table 0
        } else {
            output.extend_from_slice(&[1, 0x00]); // Y: DC table 0, AC table 0
            output.extend_from_slice(&[2, 0x11]); // Cb: DC table 1, AC table 1
            output.extend_from_slice(&[3, 0x11]); // Cr: DC table 1, AC table 1
        }

        // Spectral selection and successive approximation (baseline)
        output.extend_from_slice(&[0x00, 0x3F, 0x00]); // Ss=0, Se=63, Ah=0, Al=0
    }

    /// Encode the image scan data.
    fn encode_scan(
        &self,
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        width: usize,
        height: usize,
        is_grayscale: bool,
        bit_writer: &mut BitWriter,
    ) -> Result<()> {
        let mcu_width = 8;
        let mcu_height = 8;

        let (h_factor, v_factor) = if is_grayscale {
            (1, 1)
        } else {
            match self.config.subsampling {
                ChromaSubsampling::Yuv444 => (1, 1),
                ChromaSubsampling::Yuv422 => (2, 1),
                ChromaSubsampling::Yuv420 => (2, 2),
                ChromaSubsampling::Gray => (1, 1),
            }
        };

        let block_width = mcu_width * h_factor;
        let block_height = mcu_height * v_factor;

        let mcus_x = width.div_ceil(block_width);
        let mcus_y = height.div_ceil(block_height);

        // Downsample chroma if needed
        let (cb_ds, cr_ds, chr_width, chr_height) = if is_grayscale {
            (vec![], vec![], 0, 0)
        } else {
            let cb = self.downsample_chroma(cb_plane, width, height);
            let cr = self.downsample_chroma(cr_plane, width, height);
            let cw = match self.config.subsampling {
                ChromaSubsampling::Yuv444 | ChromaSubsampling::Gray => width,
                ChromaSubsampling::Yuv422 | ChromaSubsampling::Yuv420 => width.div_ceil(2),
            };
            let ch = match self.config.subsampling {
                ChromaSubsampling::Yuv444 | ChromaSubsampling::Yuv422 | ChromaSubsampling::Gray => height,
                ChromaSubsampling::Yuv420 => height.div_ceil(2),
            };
            (cb, cr, cw, ch)
        };

        let mut dc_y = 0i16;
        let mut dc_cb = 0i16;
        let mut dc_cr = 0i16;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                // Encode Y blocks
                for v in 0..v_factor {
                    for h in 0..h_factor {
                        let block_x = mcu_x * block_width + h * 8;
                        let block_y = mcu_y * block_height + v * 8;

                        let mut block = self.extract_block(y_plane, width, height, block_x, block_y);
                        self.encode_block(
                            &mut block, &self.lum_quant, &mut dc_y,
                            &self.dc_lum_encoder, &self.ac_lum_encoder, bit_writer,
                        );
                    }
                }

                // Encode Cb and Cr blocks
                if !is_grayscale {
                    let cb_x = mcu_x * 8;
                    let cb_y = mcu_y * 8;

                    let mut cb_block = self.extract_block(&cb_ds, chr_width, chr_height, cb_x, cb_y);
                    self.encode_block(
                        &mut cb_block, &self.chr_quant, &mut dc_cb,
                        &self.dc_chr_encoder, &self.ac_chr_encoder, bit_writer,
                    );

                    let mut cr_block = self.extract_block(&cr_ds, chr_width, chr_height, cb_x, cb_y);
                    self.encode_block(
                        &mut cr_block, &self.chr_quant, &mut dc_cr,
                        &self.dc_chr_encoder, &self.ac_chr_encoder, bit_writer,
                    );
                }
            }
        }

        Ok(())
    }

    /// Extract an 8x8 block from a plane.
    fn extract_block(&self, plane: &[u8], width: usize, height: usize, x: usize, y: usize) -> [i16; 64] {
        let mut block = [0i16; 64];

        for row in 0..8 {
            for col in 0..8 {
                let px = (x + col).min(width.saturating_sub(1));
                let py = (y + row).min(height.saturating_sub(1));
                block[row * 8 + col] = plane[py * width + px] as i16 - 128;
            }
        }

        block
    }

    /// Encode a single 8x8 block.
    fn encode_block(
        &self,
        block: &mut [i16; 64],
        quant_table: &[u8; 64],
        dc_pred: &mut i16,
        dc_encoder: &HuffmanEncoder,
        ac_encoder: &HuffmanEncoder,
        bit_writer: &mut BitWriter,
    ) {
        // Forward DCT
        fast_forward_dct(block);

        // Quantize
        let mut quantized = [0i16; 64];
        for i in 0..64 {
            let q = quant_table[i] as i16;
            quantized[i] = (block[i] + q / 2) / q;
        }

        // Encode DC coefficient (DPCM)
        let dc_diff = quantized[0] - *dc_pred;
        *dc_pred = quantized[0];

        let (dc_cat, dc_bits) = category_encode(dc_diff);
        let (dc_code, dc_len) = dc_encoder.encode(dc_cat);
        bit_writer.write_bits(dc_code as u32, dc_len as u32);
        if dc_cat > 0 {
            bit_writer.write_bits(dc_bits as u32, dc_cat as u32);
        }

        // Encode AC coefficients
        let mut zero_run = 0;

        for i in 1..64 {
            let coef = quantized[ZIGZAG[i]];

            if coef == 0 {
                zero_run += 1;
            } else {
                // Encode any runs of 16 zeros
                while zero_run >= 16 {
                    let (code, len) = ac_encoder.encode(0xF0); // ZRL
                    bit_writer.write_bits(code as u32, len as u32);
                    zero_run -= 16;
                }

                let (ac_cat, ac_bits) = category_encode(coef);
                let rs = (zero_run << 4) | ac_cat;
                let (code, len) = ac_encoder.encode(rs);
                bit_writer.write_bits(code as u32, len as u32);
                bit_writer.write_bits(ac_bits as u32, ac_cat as u32);

                zero_run = 0;
            }
        }

        // EOB if there are trailing zeros
        if zero_run > 0 {
            let (code, len) = ac_encoder.encode(0x00); // EOB
            bit_writer.write_bits(code as u32, len as u32);
        }
    }
}

/// Bit writer for JPEG entropy coding.
struct BitWriter {
    bytes: Vec<u8>,
    bit_buffer: u32,
    bits_in_buffer: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_buffer: 0,
            bits_in_buffer: 0,
        }
    }

    fn write_bits(&mut self, value: u32, num_bits: u32) {
        if num_bits == 0 {
            return;
        }

        self.bit_buffer = (self.bit_buffer << num_bits) | (value & ((1 << num_bits) - 1));
        self.bits_in_buffer += num_bits;

        while self.bits_in_buffer >= 8 {
            self.bits_in_buffer -= 8;
            let byte = ((self.bit_buffer >> self.bits_in_buffer) & 0xFF) as u8;
            self.bytes.push(byte);

            // Byte stuffing: insert 0x00 after 0xFF
            if byte == 0xFF {
                self.bytes.push(0x00);
            }
        }
    }

    fn flush(&mut self) {
        if self.bits_in_buffer > 0 {
            // Pad with 1s
            let padding = 8 - self.bits_in_buffer;
            self.write_bits((1 << padding) - 1, padding);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let config = JpegConfig::default();
        let encoder = JpegEncoder::new(config);
        assert_eq!(encoder.config.quality, 85);
    }

    #[test]
    fn test_encode_grayscale() {
        let image = Image::new(8, 8, PixelFormat::Gray8).unwrap();
        let encoder = JpegEncoder::new(JpegConfig::default());
        let result = encoder.encode(&image);
        assert!(result.is_ok());

        let jpeg = result.unwrap();
        // Check SOI and EOI markers
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
        assert_eq!(&jpeg[jpeg.len()-2..], &[0xFF, 0xD9]);
    }

    #[test]
    fn test_encode_rgb() {
        let image = Image::new(16, 16, PixelFormat::Rgb8).unwrap();
        let encoder = JpegEncoder::new(JpegConfig::default());
        let result = encoder.encode(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quality_affects_quantization() {
        let low_quality = JpegEncoder::new(JpegConfig { quality: 10, ..Default::default() });
        let high_quality = JpegEncoder::new(JpegConfig { quality: 90, ..Default::default() });

        // Lower quality should have higher quantization values
        assert!(low_quality.lum_quant[0] > high_quality.lum_quant[0]);
    }

    #[test]
    fn test_bit_writer() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b11, 2);
        writer.write_bits(0b0000, 4);
        writer.write_bits(0b11, 2);
        writer.flush();

        assert_eq!(writer.bytes[0], 0b11000011);
    }

    #[test]
    fn test_byte_stuffing() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xFF, 8);
        writer.flush();

        // 0xFF should be followed by 0x00
        assert_eq!(writer.bytes[0], 0xFF);
        assert_eq!(writer.bytes[1], 0x00);
    }
}
