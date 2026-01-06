//! OpenEXR decoder

use crate::channel::{ChannelList, PixelType};
use crate::compression::{decompress_rle, Compression};
use crate::error::{ExrError, Result};
use crate::header::Header;
use crate::types::{Box2i, Half};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

/// Decoded EXR image
#[derive(Debug, Clone)]
pub struct ExrImage {
    /// Image width
    pub width: i32,
    /// Image height
    pub height: i32,
    /// Channels
    pub channels: ChannelList,
    /// Pixel data (interleaved by channel, half-float values)
    pub data: Vec<f32>,
    /// Header with full metadata
    pub header: Header,
}

impl ExrImage {
    /// Create new image
    pub fn new(width: i32, height: i32, channels: ChannelList) -> Self {
        let pixel_count = (width * height) as usize;
        let channel_count = channels.len();

        ExrImage {
            width,
            height,
            channels,
            data: vec![0.0; pixel_count * channel_count],
            header: Header::new(width, height),
        }
    }

    /// Get pixel value for channel at (x, y)
    pub fn get_pixel(&self, x: i32, y: i32, channel: usize) -> f32 {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return 0.0;
        }
        let idx = ((y * self.width + x) as usize) * self.channels.len() + channel;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Set pixel value for channel at (x, y)
    pub fn set_pixel(&mut self, x: i32, y: i32, channel: usize, value: f32) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }
        let idx = ((y * self.width + x) as usize) * self.channels.len() + channel;
        if idx < self.data.len() {
            self.data[idx] = value;
        }
    }

    /// Get RGBA value at (x, y)
    pub fn get_rgba(&self, x: i32, y: i32) -> [f32; 4] {
        let r_idx = self.channels.index("R");
        let g_idx = self.channels.index("G");
        let b_idx = self.channels.index("B");
        let a_idx = self.channels.index("A");

        [
            r_idx.map(|i| self.get_pixel(x, y, i)).unwrap_or(0.0),
            g_idx.map(|i| self.get_pixel(x, y, i)).unwrap_or(0.0),
            b_idx.map(|i| self.get_pixel(x, y, i)).unwrap_or(0.0),
            a_idx.map(|i| self.get_pixel(x, y, i)).unwrap_or(1.0),
        ]
    }

    /// Convert to 8-bit sRGB (for display)
    pub fn to_srgb_8bit(&self) -> Vec<u8> {
        let pixels = (self.width * self.height) as usize;
        let mut output = Vec::with_capacity(pixels * 4);

        for y in 0..self.height {
            for x in 0..self.width {
                let rgba = self.get_rgba(x, y);
                for c in 0..4 {
                    // Apply gamma correction and clamp
                    let linear = rgba[c].max(0.0).min(1.0);
                    let srgb = if linear <= 0.0031308 {
                        linear * 12.92
                    } else {
                        1.055 * linear.powf(1.0 / 2.4) - 0.055
                    };
                    output.push((srgb * 255.0) as u8);
                }
            }
        }

        output
    }
}

/// OpenEXR decoder
pub struct ExrDecoder {
    /// Whether to convert to f32 (vs keeping half)
    convert_to_f32: bool,
}

impl Default for ExrDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExrDecoder {
    /// Create new decoder
    pub fn new() -> Self {
        ExrDecoder {
            convert_to_f32: true,
        }
    }

    /// Set whether to convert half to f32
    pub fn convert_to_f32(mut self, convert: bool) -> Self {
        self.convert_to_f32 = convert;
        self
    }

    /// Decode EXR data
    pub fn decode(&self, data: &[u8]) -> Result<ExrImage> {
        // Parse header
        let (header, header_size) = Header::parse(data)?;

        let width = header.width();
        let height = header.height();
        let channels = header.channels().clone();
        let compression = header.compression();

        // Create image
        let mut image = ExrImage::new(width, height, channels.clone());
        image.header = header;

        // Parse scanline offset table
        let offsets = self.read_offset_table(&data[header_size..], height, compression)?;

        // Decode scanlines
        let bytes_per_pixel = channels.bytes_per_pixel();
        let scanlines_per_chunk = compression.scanlines_per_chunk();

        for (chunk_idx, &offset) in offsets.iter().enumerate() {
            if offset == 0 {
                continue;
            }

            let chunk_data = &data[header_size + offset as usize..];
            self.decode_chunk(
                chunk_data,
                &mut image,
                chunk_idx,
                scanlines_per_chunk,
                bytes_per_pixel,
                compression,
            )?;
        }

        Ok(image)
    }

    /// Read scanline offset table
    fn read_offset_table(
        &self,
        data: &[u8],
        height: i32,
        compression: Compression,
    ) -> Result<Vec<u64>> {
        let scanlines_per_chunk = compression.scanlines_per_chunk() as i32;
        let chunk_count = (height + scanlines_per_chunk - 1) / scanlines_per_chunk;

        let mut cursor = Cursor::new(data);
        let mut offsets = Vec::with_capacity(chunk_count as usize);

        for _ in 0..chunk_count {
            let offset = cursor.read_u64::<LittleEndian>()?;
            offsets.push(offset);
        }

        Ok(offsets)
    }

    /// Decode a single chunk (one or more scanlines)
    fn decode_chunk(
        &self,
        data: &[u8],
        image: &mut ExrImage,
        chunk_idx: usize,
        scanlines_per_chunk: usize,
        bytes_per_pixel: usize,
        compression: Compression,
    ) -> Result<()> {
        let mut cursor = Cursor::new(data);

        // Read y coordinate
        let y_start = cursor.read_i32::<LittleEndian>()?;

        // Read packed size
        let packed_size = cursor.read_i32::<LittleEndian>()? as usize;

        if packed_size > data.len() - 8 {
            return Err(ExrError::InsufficientData {
                needed: packed_size,
                available: data.len() - 8,
            });
        }

        // Get packed data
        let packed_data = &data[8..8 + packed_size];

        // Calculate unpacked size
        let lines_in_chunk = scanlines_per_chunk.min((image.height - y_start) as usize);
        let unpacked_size = lines_in_chunk * image.width as usize * bytes_per_pixel;

        // Decompress
        let scanline_data = match compression {
            Compression::None => packed_data.to_vec(),
            Compression::Rle => decompress_rle(packed_data, unpacked_size)?,
            Compression::ZipS | Compression::Zip => {
                // For ZIP, we'd use flate2 or miniz
                // For now, fall back to uncompressed assumption
                packed_data.to_vec()
            }
            _ => {
                return Err(ExrError::UnsupportedFeature(format!(
                    "Compression {:?} not implemented",
                    compression
                )));
            }
        };

        // Copy pixel data to image
        self.copy_scanline_data(image, y_start, lines_in_chunk, &scanline_data)?;

        Ok(())
    }

    /// Copy decompressed scanline data to image
    fn copy_scanline_data(
        &self,
        image: &mut ExrImage,
        y_start: i32,
        line_count: usize,
        data: &[u8],
    ) -> Result<()> {
        let width = image.width as usize;
        let height = image.height;

        // Get channel info first to avoid borrow conflicts
        let channel_info: Vec<_> = image.channels.iter()
            .map(|c| (c.pixel_type, c.pixel_type.bytes_per_sample()))
            .collect();

        // Channels are stored separately for each scanline
        let mut offset = 0;

        for line_offset in 0..line_count {
            let y = y_start + line_offset as i32;
            if y < 0 || y >= height {
                continue;
            }

            for (ch_idx, (pixel_type, bytes_per_sample)) in channel_info.iter().enumerate() {
                for x in 0..width {
                    if offset + bytes_per_sample > data.len() {
                        return Ok(()); // Incomplete data
                    }

                    let value = match pixel_type {
                        PixelType::Half => {
                            let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
                            Half::from_bits(bits).to_f32()
                        }
                        PixelType::Float => {
                            f32::from_le_bytes([
                                data[offset],
                                data[offset + 1],
                                data[offset + 2],
                                data[offset + 3],
                            ])
                        }
                        PixelType::Uint => {
                            let u = u32::from_le_bytes([
                                data[offset],
                                data[offset + 1],
                                data[offset + 2],
                                data[offset + 3],
                            ]);
                            u as f32 / u32::MAX as f32
                        }
                    };

                    image.set_pixel(x as i32, y, ch_idx, value);
                    offset += *bytes_per_sample;
                }
            }
        }

        Ok(())
    }

    /// Check if data is a valid EXR file
    pub fn is_exr(data: &[u8]) -> bool {
        if data.len() < 4 {
            return false;
        }
        u32::from_le_bytes([data[0], data[1], data[2], data[3]]) == crate::EXR_MAGIC
    }

    /// Get image info without full decode
    pub fn probe(data: &[u8]) -> Result<(i32, i32, ChannelList, Compression)> {
        let (header, _) = Header::parse(data)?;
        Ok((
            header.width(),
            header.height(),
            header.channels().clone(),
            header.compression(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::Channel;

    #[test]
    fn test_exr_image() {
        let mut channels = ChannelList::new();
        channels.add(Channel::half("R"));
        channels.add(Channel::half("G"));
        channels.add(Channel::half("B"));

        let mut image = ExrImage::new(10, 10, channels);

        image.set_pixel(5, 5, 0, 1.0);
        image.set_pixel(5, 5, 1, 0.5);
        image.set_pixel(5, 5, 2, 0.25);

        assert!((image.get_pixel(5, 5, 0) - 1.0).abs() < 0.001);
        assert!((image.get_pixel(5, 5, 1) - 0.5).abs() < 0.001);
        assert!((image.get_pixel(5, 5, 2) - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_get_rgba() {
        let channels = ChannelList::rgba(PixelType::Half);
        let mut image = ExrImage::new(2, 2, channels);

        // Channel order in list is A, B, G, R (sorted)
        image.set_pixel(0, 0, 3, 1.0); // R
        image.set_pixel(0, 0, 2, 0.5); // G
        image.set_pixel(0, 0, 1, 0.25); // B
        image.set_pixel(0, 0, 0, 0.8); // A

        let rgba = image.get_rgba(0, 0);
        assert!((rgba[0] - 1.0).abs() < 0.001); // R
        assert!((rgba[1] - 0.5).abs() < 0.001); // G
        assert!((rgba[2] - 0.25).abs() < 0.001); // B
        assert!((rgba[3] - 0.8).abs() < 0.001); // A
    }

    #[test]
    fn test_is_exr() {
        let magic = crate::EXR_MAGIC.to_le_bytes();
        assert!(ExrDecoder::is_exr(&magic));
        assert!(!ExrDecoder::is_exr(b"PNG\x00"));
    }

    #[test]
    fn test_to_srgb() {
        let channels = ChannelList::rgba(PixelType::Half);
        let mut image = ExrImage::new(1, 1, channels);

        // Set white pixel
        image.set_pixel(0, 0, 0, 1.0); // A
        image.set_pixel(0, 0, 1, 1.0); // B
        image.set_pixel(0, 0, 2, 1.0); // G
        image.set_pixel(0, 0, 3, 1.0); // R

        let srgb = image.to_srgb_8bit();
        assert_eq!(srgb.len(), 4);
        // Allow for floating-point precision in sRGB conversion
        assert!(srgb[0] >= 254); // R
        assert!(srgb[1] >= 254); // G
        assert!(srgb[2] >= 254); // B
        assert!(srgb[3] >= 254); // A
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = ExrDecoder::new().convert_to_f32(true);
        assert!(decoder.convert_to_f32);
    }
}
