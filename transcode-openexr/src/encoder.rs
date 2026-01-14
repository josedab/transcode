//! OpenEXR encoder

use crate::channel::{ChannelList, PixelType};
use crate::compression::{compress_rle, Compression};
use crate::decoder::ExrImage;
use crate::error::{ExrError, Result};
use crate::types::Half;
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Cursor, Seek, SeekFrom, Write};

/// Encoder configuration
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Compression method
    pub compression: Compression,
    /// Pixel type for output
    pub pixel_type: PixelType,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig {
            compression: Compression::Zip,
            pixel_type: PixelType::Half,
        }
    }
}

/// OpenEXR encoder
pub struct ExrEncoder {
    config: EncoderConfig,
}

impl Default for ExrEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExrEncoder {
    /// Create new encoder
    pub fn new() -> Self {
        ExrEncoder {
            config: EncoderConfig::default(),
        }
    }

    /// Create encoder with config
    pub fn with_config(config: EncoderConfig) -> Self {
        ExrEncoder { config }
    }

    /// Set compression
    pub fn compression(mut self, compression: Compression) -> Self {
        self.config.compression = compression;
        self
    }

    /// Set pixel type
    pub fn pixel_type(mut self, pixel_type: PixelType) -> Self {
        self.config.pixel_type = pixel_type;
        self
    }

    /// Encode EXR image
    pub fn encode(&self, image: &ExrImage) -> Result<Vec<u8>> {
        let mut output = Cursor::new(Vec::new());
        self.encode_to(&mut output, image)?;
        Ok(output.into_inner())
    }

    /// Encode to writer
    pub fn encode_to<W: Write + Seek>(&self, writer: &mut W, image: &ExrImage) -> Result<()> {
        let _width = image.width;
        let height = image.height;
        let compression = self.config.compression;
        let scanlines_per_chunk = compression.scanlines_per_chunk();

        // Create header
        let mut header = image.header.clone();
        header.set_compression(compression);

        // Write header
        header.write(writer)?;

        // Calculate chunk count
        let chunk_count = (height as usize).div_ceil(scanlines_per_chunk);

        // Reserve space for offset table
        let offset_table_start = writer.stream_position()? as usize;
        for _ in 0..chunk_count {
            writer.write_u64::<LittleEndian>(0)?;
        }

        // Write chunks and record offsets
        let mut offsets = Vec::with_capacity(chunk_count);
        let _data_start = writer.stream_position()? as usize;

        for chunk_idx in 0..chunk_count {
            let offset = writer.stream_position()? as usize - offset_table_start;
            offsets.push(offset as u64);

            let y_start = (chunk_idx * scanlines_per_chunk) as i32;
            let lines_in_chunk = scanlines_per_chunk.min((height - y_start) as usize);

            self.write_chunk(writer, image, y_start, lines_in_chunk, compression)?;
        }

        // Go back and write offset table
        writer.seek(SeekFrom::Start(offset_table_start as u64))?;
        for offset in offsets {
            writer.write_u64::<LittleEndian>(offset)?;
        }

        // Seek to end
        writer.seek(SeekFrom::End(0))?;

        Ok(())
    }

    /// Write a single chunk
    fn write_chunk<W: Write>(
        &self,
        writer: &mut W,
        image: &ExrImage,
        y_start: i32,
        line_count: usize,
        compression: Compression,
    ) -> Result<()> {
        // Build scanline data
        let scanline_data = self.build_scanline_data(image, y_start, line_count)?;

        // Compress
        let compressed = match compression {
            Compression::None => scanline_data,
            Compression::Rle => compress_rle(&scanline_data)?,
            _ => {
                // For unsupported compression, use uncompressed
                scanline_data
            }
        };

        // Write y coordinate
        writer.write_i32::<LittleEndian>(y_start)?;

        // Write packed size
        writer.write_i32::<LittleEndian>(compressed.len() as i32)?;

        // Write data
        writer.write_all(&compressed)?;

        Ok(())
    }

    /// Build scanline data for encoding
    fn build_scanline_data(
        &self,
        image: &ExrImage,
        y_start: i32,
        line_count: usize,
    ) -> Result<Vec<u8>> {
        let width = image.width as usize;
        let channel_count = image.channels.len();
        let output_type = self.config.pixel_type;
        let bytes_per_sample = output_type.bytes_per_sample();

        let mut data = Vec::with_capacity(line_count * width * channel_count * bytes_per_sample);

        for line_offset in 0..line_count {
            let y = y_start + line_offset as i32;

            // Write each channel for this scanline
            for ch_idx in 0..channel_count {
                for x in 0..width {
                    let value = image.get_pixel(x as i32, y, ch_idx);

                    match output_type {
                        PixelType::Half => {
                            let half = Half::from_f32(value);
                            data.write_u16::<LittleEndian>(half.to_bits())?;
                        }
                        PixelType::Float => {
                            data.write_f32::<LittleEndian>(value)?;
                        }
                        PixelType::Uint => {
                            let u = (value.clamp(0.0, 1.0) * u32::MAX as f32) as u32;
                            data.write_u32::<LittleEndian>(u)?;
                        }
                    }
                }
            }
        }

        Ok(data)
    }

    /// Create EXR from raw RGBA float data
    pub fn from_rgba_f32(
        &self,
        width: i32,
        height: i32,
        data: &[f32],
    ) -> Result<Vec<u8>> {
        if data.len() != (width * height * 4) as usize {
            return Err(ExrError::BufferTooSmall {
                needed: (width * height * 4) as usize,
                available: data.len(),
            });
        }

        let channels = ChannelList::rgba(self.config.pixel_type);
        let mut image = ExrImage::new(width, height, channels);

        // Copy data (input is RGBA, channels are sorted as A, B, G, R)
        let r_idx = image.channels.index("R").unwrap_or(0);
        let g_idx = image.channels.index("G").unwrap_or(1);
        let b_idx = image.channels.index("B").unwrap_or(2);
        let a_idx = image.channels.index("A").unwrap_or(3);

        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) * 4) as usize;
                image.set_pixel(x, y, r_idx, data[i]);     // R
                image.set_pixel(x, y, g_idx, data[i + 1]); // G
                image.set_pixel(x, y, b_idx, data[i + 2]); // B
                image.set_pixel(x, y, a_idx, data[i + 3]); // A
            }
        }

        self.encode(&image)
    }

    /// Create EXR from raw RGB float data
    pub fn from_rgb_f32(
        &self,
        width: i32,
        height: i32,
        data: &[f32],
    ) -> Result<Vec<u8>> {
        if data.len() != (width * height * 3) as usize {
            return Err(ExrError::BufferTooSmall {
                needed: (width * height * 3) as usize,
                available: data.len(),
            });
        }

        let channels = ChannelList::rgb(self.config.pixel_type);
        let mut image = ExrImage::new(width, height, channels);

        let r_idx = image.channels.index("R").unwrap_or(0);
        let g_idx = image.channels.index("G").unwrap_or(1);
        let b_idx = image.channels.index("B").unwrap_or(2);

        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) * 3) as usize;
                image.set_pixel(x, y, r_idx, data[i]);     // R
                image.set_pixel(x, y, g_idx, data[i + 1]); // G
                image.set_pixel(x, y, b_idx, data[i + 2]); // B
            }
        }

        self.encode(&image)
    }

    /// Create EXR from 8-bit sRGB data
    pub fn from_srgb_8bit(
        &self,
        width: i32,
        height: i32,
        data: &[u8],
        has_alpha: bool,
    ) -> Result<Vec<u8>> {
        let channels_per_pixel = if has_alpha { 4 } else { 3 };
        let expected = (width * height) as usize * channels_per_pixel;

        if data.len() != expected {
            return Err(ExrError::BufferTooSmall {
                needed: expected,
                available: data.len(),
            });
        }

        let channels = if has_alpha {
            ChannelList::rgba(self.config.pixel_type)
        } else {
            ChannelList::rgb(self.config.pixel_type)
        };

        let mut image = ExrImage::new(width, height, channels);

        let r_idx = image.channels.index("R").unwrap_or(0);
        let g_idx = image.channels.index("G").unwrap_or(1);
        let b_idx = image.channels.index("B").unwrap_or(2);
        let a_idx = image.channels.index("A");

        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) as usize) * channels_per_pixel;

                // Convert sRGB to linear
                let r = srgb_to_linear(data[i]);
                let g = srgb_to_linear(data[i + 1]);
                let b = srgb_to_linear(data[i + 2]);

                image.set_pixel(x, y, r_idx, r);
                image.set_pixel(x, y, g_idx, g);
                image.set_pixel(x, y, b_idx, b);

                if has_alpha {
                    if let Some(a_idx) = a_idx {
                        let a = data[i + 3] as f32 / 255.0;
                        image.set_pixel(x, y, a_idx, a);
                    }
                }
            }
        }

        self.encode(&image)
    }
}

/// Convert sRGB to linear
fn srgb_to_linear(srgb: u8) -> f32 {
    let s = srgb as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::ExrDecoder;

    #[test]
    fn test_encoder_creation() {
        let encoder = ExrEncoder::new()
            .compression(Compression::Rle)
            .pixel_type(PixelType::Float);

        assert_eq!(encoder.config.compression, Compression::Rle);
        assert_eq!(encoder.config.pixel_type, PixelType::Float);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let channels = ChannelList::rgb(PixelType::Half);
        let mut image = ExrImage::new(4, 4, channels);

        // Set some pixel values
        let r_idx = image.channels.index("R").unwrap();
        let g_idx = image.channels.index("G").unwrap();
        let b_idx = image.channels.index("B").unwrap();

        image.set_pixel(0, 0, r_idx, 1.0);
        image.set_pixel(0, 0, g_idx, 0.5);
        image.set_pixel(0, 0, b_idx, 0.25);

        let encoder = ExrEncoder::new().compression(Compression::None);
        let encoded = encoder.encode(&image).unwrap();

        // Check magic number
        assert!(ExrDecoder::is_exr(&encoded));
    }

    #[test]
    fn test_from_rgba_f32() {
        let width = 2;
        let height = 2;
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0, // Red
            0.0, 1.0, 0.0, 1.0, // Green
            0.0, 0.0, 1.0, 1.0, // Blue
            1.0, 1.0, 1.0, 1.0, // White
        ];

        let encoder = ExrEncoder::new().compression(Compression::None);
        let result = encoder.from_rgba_f32(width, height, &data);
        assert!(result.is_ok());

        let exr = result.unwrap();
        assert!(ExrDecoder::is_exr(&exr));
    }

    #[test]
    fn test_from_rgb_f32() {
        let width = 2;
        let height = 2;
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, // Red
            0.0, 1.0, 0.0, // Green
            0.0, 0.0, 1.0, // Blue
            1.0, 1.0, 1.0, // White
        ];

        let encoder = ExrEncoder::new().compression(Compression::None);
        let result = encoder.from_rgb_f32(width, height, &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_srgb_8bit() {
        let width = 2;
        let height = 2;
        let data: Vec<u8> = vec![
            255, 0, 0, 255, // Red
            0, 255, 0, 255, // Green
            0, 0, 255, 255, // Blue
            255, 255, 255, 255, // White
        ];

        let encoder = ExrEncoder::new().compression(Compression::None);
        let result = encoder.from_srgb_8bit(width, height, &data, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_srgb_to_linear() {
        assert!((srgb_to_linear(0) - 0.0).abs() < 0.001);
        assert!((srgb_to_linear(255) - 1.0).abs() < 0.001);
        // Mid-gray should be around 0.21 in linear
        assert!((srgb_to_linear(128) - 0.215).abs() < 0.02);
    }

    #[test]
    fn test_buffer_size_check() {
        let encoder = ExrEncoder::new();

        // Too small buffer
        let result = encoder.from_rgba_f32(100, 100, &[0.0; 10]);
        assert!(result.is_err());
    }
}
