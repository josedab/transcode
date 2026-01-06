//! PNG encoder implementation.

use crate::error::{ImageError, Result};
use crate::image::{Image, PixelFormat};
use super::{
    ChunkType, ColorType, CompressionLevel, InterlaceMethod, PNG_SIGNATURE,
    crc32, write_u32_be,
};
use super::filter::{filter_row, select_filter, FilterType};

/// PNG encoder configuration.
#[derive(Debug, Clone)]
pub struct PngConfig {
    /// Compression level.
    pub compression: CompressionLevel,
    /// Filter selection strategy.
    pub filter: FilterStrategy,
    /// Include interlacing.
    pub interlace: bool,
    /// Include sRGB chunk.
    pub srgb: bool,
    /// Text metadata to include.
    pub text: Vec<(String, String)>,
}

impl Default for PngConfig {
    fn default() -> Self {
        Self {
            compression: CompressionLevel::Default,
            filter: FilterStrategy::Adaptive,
            interlace: false,
            srgb: true,
            text: Vec::new(),
        }
    }
}

/// Filter selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterStrategy {
    /// No filtering.
    None,
    /// Always use Sub filter.
    Sub,
    /// Always use Up filter.
    Up,
    /// Always use Average filter.
    Average,
    /// Always use Paeth filter.
    Paeth,
    /// Adaptively select best filter per row.
    Adaptive,
}

/// PNG encoder.
pub struct PngEncoder {
    config: PngConfig,
}

impl PngEncoder {
    /// Create a new PNG encoder.
    pub fn new() -> Self {
        Self {
            config: PngConfig::default(),
        }
    }

    /// Create encoder with configuration.
    pub fn with_config(config: PngConfig) -> Self {
        Self { config }
    }

    /// Encode an image to PNG.
    pub fn encode(&self, image: &Image) -> Result<Vec<u8>> {
        let mut output = Vec::new();

        // Write signature
        output.extend_from_slice(&PNG_SIGNATURE);

        // Determine color type
        let (color_type, data) = self.prepare_image_data(image)?;
        let bit_depth = 8u8;

        // Write IHDR
        self.write_ihdr(&mut output, image.width(), image.height(), bit_depth, color_type)?;

        // Write sRGB if requested
        if self.config.srgb {
            self.write_srgb(&mut output)?;
        }

        // Write text metadata
        for (keyword, value) in &self.config.text {
            self.write_text(&mut output, keyword, value)?;
        }

        // Filter and compress image data
        let bytes_per_pixel = match color_type {
            ColorType::Grayscale => 1,
            ColorType::GrayscaleAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
            ColorType::Indexed => 1,
        };

        let filtered = self.filter_image(&data, image.width() as usize, image.height() as usize, bytes_per_pixel)?;
        let compressed = self.compress(&filtered)?;

        // Write IDAT chunks
        self.write_idat(&mut output, &compressed)?;

        // Write IEND
        self.write_iend(&mut output)?;

        Ok(output)
    }

    /// Prepare image data for encoding.
    fn prepare_image_data(&self, image: &Image) -> Result<(ColorType, Vec<u8>)> {
        match image.format() {
            PixelFormat::Gray8 => Ok((ColorType::Grayscale, image.data().to_vec())),
            PixelFormat::GrayA8 => Ok((ColorType::GrayscaleAlpha, image.data().to_vec())),
            PixelFormat::Rgb8 => Ok((ColorType::Rgb, image.data().to_vec())),
            PixelFormat::Rgba8 => Ok((ColorType::Rgba, image.data().to_vec())),
            PixelFormat::Bgr8 => {
                // Convert BGR to RGB
                let data = image.data();
                let mut rgb = Vec::with_capacity(data.len());
                for chunk in data.chunks(3) {
                    rgb.push(chunk[2]);
                    rgb.push(chunk[1]);
                    rgb.push(chunk[0]);
                }
                Ok((ColorType::Rgb, rgb))
            }
            PixelFormat::Bgra8 => {
                // Convert BGRA to RGBA
                let data = image.data();
                let mut rgba = Vec::with_capacity(data.len());
                for chunk in data.chunks(4) {
                    rgba.push(chunk[2]);
                    rgba.push(chunk[1]);
                    rgba.push(chunk[0]);
                    rgba.push(chunk[3]);
                }
                Ok((ColorType::Rgba, rgba))
            }
            _ => Err(ImageError::UnsupportedFormat(format!(
                "Cannot encode {:?} to PNG",
                image.format()
            ))),
        }
    }

    /// Write IHDR chunk.
    fn write_ihdr(
        &self,
        output: &mut Vec<u8>,
        width: u32,
        height: u32,
        bit_depth: u8,
        color_type: ColorType,
    ) -> Result<()> {
        let mut data = Vec::with_capacity(13);
        data.extend_from_slice(&write_u32_be(width));
        data.extend_from_slice(&write_u32_be(height));
        data.push(bit_depth);
        data.push(color_type as u8);
        data.push(0); // Compression method
        data.push(0); // Filter method
        data.push(if self.config.interlace { 1 } else { 0 }); // Interlace method

        self.write_chunk(output, ChunkType::IHDR, &data)
    }

    /// Write sRGB chunk.
    fn write_srgb(&self, output: &mut Vec<u8>) -> Result<()> {
        self.write_chunk(output, ChunkType::SRGB, &[0]) // 0 = perceptual
    }

    /// Write tEXt chunk.
    fn write_text(&self, output: &mut Vec<u8>, keyword: &str, value: &str) -> Result<()> {
        let mut data = Vec::new();
        data.extend_from_slice(keyword.as_bytes());
        data.push(0); // Null separator
        data.extend_from_slice(value.as_bytes());

        self.write_chunk(output, ChunkType::TEXT, &data)
    }

    /// Write IDAT chunk(s).
    fn write_idat(&self, output: &mut Vec<u8>, data: &[u8]) -> Result<()> {
        // Split into chunks of max 8192 bytes
        const MAX_CHUNK_SIZE: usize = 8192;

        for chunk in data.chunks(MAX_CHUNK_SIZE) {
            self.write_chunk(output, ChunkType::IDAT, chunk)?;
        }

        Ok(())
    }

    /// Write IEND chunk.
    fn write_iend(&self, output: &mut Vec<u8>) -> Result<()> {
        self.write_chunk(output, ChunkType::IEND, &[])
    }

    /// Write a chunk.
    fn write_chunk(&self, output: &mut Vec<u8>, chunk_type: ChunkType, data: &[u8]) -> Result<()> {
        // Length
        output.extend_from_slice(&write_u32_be(data.len() as u32));

        // Type
        output.extend_from_slice(chunk_type.as_bytes());

        // Data
        output.extend_from_slice(data);

        // CRC (over type + data)
        let mut crc_data = Vec::with_capacity(4 + data.len());
        crc_data.extend_from_slice(chunk_type.as_bytes());
        crc_data.extend_from_slice(data);
        let crc = crc32(&crc_data);
        output.extend_from_slice(&write_u32_be(crc));

        Ok(())
    }

    /// Filter image data.
    fn filter_image(
        &self,
        data: &[u8],
        width: usize,
        height: usize,
        bytes_per_pixel: usize,
    ) -> Result<Vec<u8>> {
        let row_bytes = width * bytes_per_pixel;
        let mut filtered = Vec::with_capacity(height * (1 + row_bytes));

        let mut prev_row: Option<&[u8]> = None;

        for y in 0..height {
            let row_start = y * row_bytes;
            let row = &data[row_start..row_start + row_bytes];

            // Select filter type
            let filter_type = match self.config.filter {
                FilterStrategy::None => FilterType::None,
                FilterStrategy::Sub => FilterType::Sub,
                FilterStrategy::Up => FilterType::Up,
                FilterStrategy::Average => FilterType::Average,
                FilterStrategy::Paeth => FilterType::Paeth,
                FilterStrategy::Adaptive => select_filter(row, prev_row, bytes_per_pixel),
            };

            // Write filter type byte
            filtered.push(filter_type as u8);

            // Filter the row
            let mut filtered_row = vec![0u8; row_bytes];
            filter_row(filter_type, row, prev_row, bytes_per_pixel, &mut filtered_row);
            filtered.extend_from_slice(&filtered_row);

            prev_row = Some(row);
        }

        Ok(filtered)
    }

    /// Compress data using DEFLATE.
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let level = self.config.compression.level();

        // Simple DEFLATE compression
        let mut output = Vec::new();

        // Zlib header
        let cmf = 0x78; // CM=8 (deflate), CINFO=7 (32K window)
        let flg = match level {
            0 => 0x01,
            1..=5 => 0x5E,
            6..=7 => 0x9C,
            _ => 0xDA,
        };
        output.push(cmf);
        output.push(flg);

        // DEFLATE data
        let deflated = self.deflate(data, level)?;
        output.extend_from_slice(&deflated);

        // Adler-32 checksum
        let checksum = adler32(data);
        output.extend_from_slice(&checksum.to_be_bytes());

        Ok(output)
    }

    /// DEFLATE compression.
    fn deflate(&self, data: &[u8], level: u8) -> Result<Vec<u8>> {
        if level == 0 {
            // Store blocks (no compression)
            return self.deflate_store(data);
        }

        // Use fixed Huffman codes for simplicity
        self.deflate_fixed(data)
    }

    /// DEFLATE with stored blocks (no compression).
    fn deflate_store(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut output = Vec::new();

        for (i, chunk) in data.chunks(65535).enumerate() {
            let is_last = (i + 1) * 65535 >= data.len();

            // Block header: BFINAL (1 bit) + BTYPE (2 bits) = 0 for stored
            output.push(if is_last { 0x01 } else { 0x00 });

            // LEN and NLEN
            let len = chunk.len() as u16;
            output.extend_from_slice(&len.to_le_bytes());
            output.extend_from_slice(&(!len).to_le_bytes());

            // Data
            output.extend_from_slice(chunk);
        }

        Ok(output)
    }

    /// DEFLATE with fixed Huffman codes.
    fn deflate_fixed(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        // Single block with BFINAL=1, BTYPE=01 (fixed Huffman)
        writer.write_bits(1, 1); // BFINAL
        writer.write_bits(1, 2); // BTYPE = fixed Huffman

        // Encode literals
        for &byte in data {
            self.write_literal(&mut writer, byte as u16);
        }

        // End of block (symbol 256)
        self.write_literal(&mut writer, 256);

        writer.flush();
        Ok(writer.bytes)
    }

    /// Write a literal/length code with fixed Huffman.
    fn write_literal(&self, writer: &mut BitWriter, code: u16) {
        if code < 144 {
            // 0-143: 8 bits, codes 00110000-10111111
            writer.write_bits_rev(0x30 + code as u32, 8);
        } else if code < 256 {
            // 144-255: 9 bits, codes 110010000-111111111
            writer.write_bits_rev(0x190 + (code - 144) as u32, 9);
        } else if code < 280 {
            // 256-279: 7 bits, codes 0000000-0010111
            writer.write_bits_rev((code - 256) as u32, 7);
        } else {
            // 280-287: 8 bits, codes 11000000-11000111
            writer.write_bits_rev(0xC0 + (code - 280) as u32, 8);
        }
    }
}

impl Default for PngEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate Adler-32 checksum.
fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;

    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }

    (b << 16) | a
}

/// Bit writer for DEFLATE encoding.
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
        self.bit_buffer |= value << self.bits_in_buffer;
        self.bits_in_buffer += num_bits;

        while self.bits_in_buffer >= 8 {
            self.bytes.push((self.bit_buffer & 0xFF) as u8);
            self.bit_buffer >>= 8;
            self.bits_in_buffer -= 8;
        }
    }

    fn write_bits_rev(&mut self, value: u32, num_bits: u32) {
        // Reverse bit order for Huffman codes
        let mut reversed = 0u32;
        let mut v = value;
        for _ in 0..num_bits {
            reversed = (reversed << 1) | (v & 1);
            v >>= 1;
        }
        self.write_bits(reversed, num_bits);
    }

    fn flush(&mut self) {
        if self.bits_in_buffer > 0 {
            self.bytes.push((self.bit_buffer & 0xFF) as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = PngEncoder::new();
        assert_eq!(encoder.config.compression.level(), 6);
    }

    #[test]
    fn test_encode_grayscale() {
        let image = Image::new(8, 8, PixelFormat::Gray8).unwrap();
        let encoder = PngEncoder::new();
        let result = encoder.encode(&image);
        assert!(result.is_ok());

        let png = result.unwrap();
        // Check signature
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_encode_rgb() {
        let image = Image::new(16, 16, PixelFormat::Rgb8).unwrap();
        let encoder = PngEncoder::new();
        let result = encoder.encode(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_rgba() {
        let mut image = Image::new(4, 4, PixelFormat::Rgba8).unwrap();
        // Set some pixels with transparency
        image.set_pixel(0, 0, &[255, 0, 0, 128]); // Semi-transparent red
        image.set_pixel(1, 0, &[0, 255, 0, 255]); // Opaque green

        let encoder = PngEncoder::new();
        let result = encoder.encode(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adler32() {
        // Test with known value
        let data = b"Hello";
        let checksum = adler32(data);
        assert!(checksum != 0);

        // Empty data
        assert_eq!(adler32(&[]), 1);
    }

    #[test]
    fn test_bit_writer() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b110, 3);
        writer.write_bits(0b01, 2);
        writer.write_bits(0b111, 3);
        writer.flush();

        assert_eq!(writer.bytes[0], 0b11101110);
    }

    #[test]
    fn test_config_builder() {
        let config = PngConfig {
            compression: CompressionLevel::Best,
            filter: FilterStrategy::Paeth,
            interlace: true,
            srgb: false,
            text: vec![("Title".to_string(), "Test Image".to_string())],
        };

        let encoder = PngEncoder::with_config(config);
        assert_eq!(encoder.config.compression.level(), 9);
    }
}
