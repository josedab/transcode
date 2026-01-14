//! TIFF encoder

use crate::compression::{apply_horizontal_predictor, compress, Compression};
use crate::error::Result;
use crate::ifd::{Ifd, IfdEntry, IfdValue};
use crate::tags::tag;
use crate::types::{ColorSpace, PhotometricInterpretation, TiffImage};
use crate::{TIFF_MAGIC_LE, TIFF_VERSION};
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::{Cursor, Seek, SeekFrom, Write};

/// TIFF encoder
pub struct TiffEncoder {
    /// Compression method
    pub compression: Compression,
    /// Use horizontal differencing predictor
    pub use_predictor: bool,
    /// Rows per strip (0 = single strip)
    pub rows_per_strip: u32,
}

impl Default for TiffEncoder {
    fn default() -> Self {
        TiffEncoder::new()
    }
}

impl TiffEncoder {
    /// Create new encoder with default settings
    pub fn new() -> Self {
        TiffEncoder {
            compression: Compression::None,
            use_predictor: false,
            rows_per_strip: 0,
        }
    }

    /// Set compression method
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Enable LZW compression
    pub fn lzw(mut self) -> Self {
        self.compression = Compression::Lzw;
        self.use_predictor = true;
        self
    }

    /// Enable PackBits compression
    pub fn packbits(mut self) -> Self {
        self.compression = Compression::PackBits;
        self
    }

    /// Set rows per strip
    pub fn rows_per_strip(mut self, rows: u32) -> Self {
        self.rows_per_strip = rows;
        self
    }

    /// Enable horizontal differencing predictor
    pub fn predictor(mut self, enable: bool) -> Self {
        self.use_predictor = enable;
        self
    }

    /// Encode image to TIFF data
    pub fn encode(&self, image: &TiffImage) -> Result<Vec<u8>> {
        let mut output = Cursor::new(Vec::new());
        self.encode_to(&mut output, image)?;
        Ok(output.into_inner())
    }

    /// Encode image to writer
    pub fn encode_to<W: Write + Seek>(&self, writer: &mut W, image: &TiffImage) -> Result<()> {
        // Write header
        writer.write_all(&TIFF_MAGIC_LE)?; // Little endian
        writer.write_u16::<LittleEndian>(TIFF_VERSION)?;

        // Placeholder for IFD offset
        let ifd_offset_pos = writer.stream_position()?;
        writer.write_u32::<LittleEndian>(0)?;

        // Prepare image data
        let rows_per_strip = if self.rows_per_strip == 0 {
            image.height
        } else {
            self.rows_per_strip
        };

        let bytes_per_row = image.bytes_per_row();
        let num_strips = image.height.div_ceil(rows_per_strip);

        // Write strips
        let mut strip_offsets = Vec::with_capacity(num_strips as usize);
        let mut strip_byte_counts = Vec::with_capacity(num_strips as usize);

        for strip_idx in 0..num_strips {
            let start_row = strip_idx * rows_per_strip;
            let end_row = ((strip_idx + 1) * rows_per_strip).min(image.height);

            // Extract strip data
            let strip_start = start_row as usize * bytes_per_row;
            let strip_end = end_row as usize * bytes_per_row;
            let mut strip_data = image.data[strip_start..strip_end].to_vec();

            // Apply predictor if enabled
            if self.use_predictor && self.compression == Compression::Lzw {
                apply_horizontal_predictor(
                    &mut strip_data,
                    image.width as usize,
                    image.samples_per_pixel as usize,
                );
            }

            // Compress strip
            let compressed = compress(self.compression, &strip_data)?;

            // Record offset and size
            let offset = writer.stream_position()? as u32;
            strip_offsets.push(offset);
            strip_byte_counts.push(compressed.len() as u32);

            // Write compressed data
            writer.write_all(&compressed)?;

            // Pad to word boundary
            if compressed.len() % 2 != 0 {
                writer.write_u8(0)?;
            }
        }

        // Build IFD
        let mut ifd = Ifd::new();

        // Required tags
        ifd.add(IfdEntry::long(tag::IMAGE_WIDTH, image.width));
        ifd.add(IfdEntry::long(tag::IMAGE_LENGTH, image.height));
        ifd.add(IfdEntry::new(
            tag::BITS_PER_SAMPLE,
            IfdValue::Shorts(image.bits_per_sample.clone()),
        ));
        ifd.add(IfdEntry::short(tag::COMPRESSION, self.compression.to_u16()));
        ifd.add(IfdEntry::short(
            tag::PHOTOMETRIC_INTERPRETATION,
            image.photometric.to_u16(),
        ));

        // Strip tags
        if strip_offsets.len() == 1 {
            ifd.add(IfdEntry::long(tag::STRIP_OFFSETS, strip_offsets[0]));
            ifd.add(IfdEntry::long(tag::STRIP_BYTE_COUNTS, strip_byte_counts[0]));
        } else {
            ifd.add(IfdEntry::new(
                tag::STRIP_OFFSETS,
                IfdValue::Longs(strip_offsets),
            ));
            ifd.add(IfdEntry::new(
                tag::STRIP_BYTE_COUNTS,
                IfdValue::Longs(strip_byte_counts),
            ));
        }

        ifd.add(IfdEntry::short(
            tag::SAMPLES_PER_PIXEL,
            image.samples_per_pixel,
        ));
        ifd.add(IfdEntry::long(tag::ROWS_PER_STRIP, rows_per_strip));

        // Resolution
        ifd.add(IfdEntry::rational(
            tag::X_RESOLUTION,
            (image.resolution_x * 1000.0) as u32,
            1000,
        ));
        ifd.add(IfdEntry::rational(
            tag::Y_RESOLUTION,
            (image.resolution_y * 1000.0) as u32,
            1000,
        ));
        ifd.add(IfdEntry::short(
            tag::RESOLUTION_UNIT,
            image.resolution_unit.to_u16(),
        ));

        // Planar configuration
        ifd.add(IfdEntry::short(
            tag::PLANAR_CONFIGURATION,
            image.planar_config.to_u16(),
        ));

        // Sample format if not default uint
        if image.sample_format != crate::types::SampleFormat::Uint {
            let formats = vec![image.sample_format.to_u16(); image.samples_per_pixel as usize];
            ifd.add(IfdEntry::new(tag::SAMPLE_FORMAT, IfdValue::Shorts(formats)));
        }

        // Predictor
        if self.use_predictor && self.compression == Compression::Lzw {
            ifd.add(IfdEntry::short(tag::PREDICTOR, 2)); // Horizontal differencing
        }

        // Extra samples for RGBA
        if image.color_space == ColorSpace::Rgba || image.color_space == ColorSpace::GrayscaleAlpha
        {
            ifd.add(IfdEntry::short(tag::EXTRA_SAMPLES, 2)); // Unassociated alpha
        }

        // Color palette
        if let Some(ref palette) = image.palette {
            let mut colormap = Vec::with_capacity(palette.len() * 3);
            // Red values
            for entry in palette {
                colormap.push(entry[0]);
            }
            // Green values
            for entry in palette {
                colormap.push(entry[1]);
            }
            // Blue values
            for entry in palette {
                colormap.push(entry[2]);
            }
            ifd.add(IfdEntry::new(tag::COLOR_MAP, IfdValue::Shorts(colormap)));
        }

        // Software tag
        ifd.add(IfdEntry::ascii(tag::SOFTWARE, "transcode-tiff"));

        // Write IFD
        let ifd_offset = writer.stream_position()? as u32;
        ifd.write::<_, LittleEndian>(writer)?;

        // Update IFD offset in header
        writer.seek(SeekFrom::Start(ifd_offset_pos))?;
        writer.write_u32::<LittleEndian>(ifd_offset)?;

        Ok(())
    }

    /// Create a grayscale image from 8-bit data
    pub fn from_grayscale_8bit(width: u32, height: u32, data: &[u8]) -> TiffImage {
        let mut image = TiffImage::new(width, height, ColorSpace::Grayscale);
        image.data = data.to_vec();
        image.photometric = PhotometricInterpretation::BlackIsZero;
        image
    }

    /// Create an RGB image from 8-bit data
    pub fn from_rgb_8bit(width: u32, height: u32, data: &[u8]) -> TiffImage {
        let mut image = TiffImage::new(width, height, ColorSpace::Rgb);
        image.data = data.to_vec();
        image
    }

    /// Create an RGBA image from 8-bit data
    pub fn from_rgba_8bit(width: u32, height: u32, data: &[u8]) -> TiffImage {
        let mut image = TiffImage::new(width, height, ColorSpace::Rgba);
        image.data = data.to_vec();
        image
    }

    /// Create a 16-bit grayscale image
    pub fn from_grayscale_16bit(width: u32, height: u32, data: &[u16]) -> TiffImage {
        let mut image = TiffImage::new(width, height, ColorSpace::Grayscale);
        image.bits_per_sample = vec![16];

        // Convert u16 to bytes (little endian)
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &val in data {
            bytes.push((val & 0xFF) as u8);
            bytes.push((val >> 8) as u8);
        }
        image.data = bytes;
        image.photometric = PhotometricInterpretation::BlackIsZero;
        image
    }

    /// Create a 16-bit RGB image
    pub fn from_rgb_16bit(width: u32, height: u32, data: &[u16]) -> TiffImage {
        let mut image = TiffImage::new(width, height, ColorSpace::Rgb);
        image.bits_per_sample = vec![16, 16, 16];

        // Convert u16 to bytes (little endian)
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &val in data {
            bytes.push((val & 0xFF) as u8);
            bytes.push((val >> 8) as u8);
        }
        image.data = bytes;
        image
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::TiffDecoder;

    #[test]
    fn test_encode_grayscale() {
        let width = 8;
        let height = 8;
        let data: Vec<u8> = (0..64).collect();

        let image = TiffEncoder::from_grayscale_8bit(width, height, &data);
        let encoder = TiffEncoder::new();
        let encoded = encoder.encode(&image).unwrap();

        assert!(TiffDecoder::probe(&encoded));

        let decoder = TiffDecoder::new();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.data, data);
    }

    #[test]
    fn test_encode_rgb() {
        let width = 4;
        let height = 4;
        let mut data = Vec::new();
        for y in 0..height {
            for x in 0..width {
                data.push((x * 64) as u8); // R
                data.push((y * 64) as u8); // G
                data.push(128);            // B
            }
        }

        let image = TiffEncoder::from_rgb_8bit(width, height, &data);
        let encoder = TiffEncoder::new();
        let encoded = encoder.encode(&image).unwrap();

        let decoder = TiffDecoder::new();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.color_space, ColorSpace::Rgb);
        assert_eq!(decoded.data, data);
    }

    #[test]
    fn test_encode_rgba() {
        let width = 2;
        let height = 2;
        let data = vec![
            255, 0, 0, 255,     // Red
            0, 255, 0, 128,     // Green semi-transparent
            0, 0, 255, 255,     // Blue
            255, 255, 0, 64,    // Yellow mostly transparent
        ];

        let image = TiffEncoder::from_rgba_8bit(width, height, &data);
        let encoder = TiffEncoder::new();
        let encoded = encoder.encode(&image).unwrap();

        let decoder = TiffDecoder::new();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.color_space, ColorSpace::Rgba);
        assert_eq!(decoded.data, data);
    }

    #[test]
    fn test_encode_with_packbits() {
        let width = 16;
        let height = 16;
        // Create repetitive data that compresses well
        let mut data = Vec::new();
        for _ in 0..height {
            for _ in 0..width {
                data.push(128);
                data.push(64);
                data.push(32);
            }
        }

        let image = TiffEncoder::from_rgb_8bit(width as u32, height as u32, &data);
        let encoder = TiffEncoder::new().packbits();
        let encoded = encoder.encode(&image).unwrap();

        let decoder = TiffDecoder::new();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded.data, data);
    }

    #[test]
    fn test_encode_with_lzw() {
        let width = 16;
        let height = 16;
        // Create data with patterns
        let mut data = Vec::new();
        for y in 0..height {
            for x in 0..width {
                data.push((x % 4 * 64) as u8);
                data.push((y % 4 * 64) as u8);
                data.push(128);
            }
        }

        let image = TiffEncoder::from_rgb_8bit(width as u32, height as u32, &data);
        let encoder = TiffEncoder::new().lzw();
        let encoded = encoder.encode(&image).unwrap();

        let decoder = TiffDecoder::new();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded.data, data);
    }

    #[test]
    fn test_encode_16bit() {
        let width = 4;
        let height = 4;
        let data: Vec<u16> = (0..16).map(|i| i * 4096).collect();

        let image = TiffEncoder::from_grayscale_16bit(width, height, &data);
        let encoder = TiffEncoder::new();
        let encoded = encoder.encode(&image).unwrap();

        let decoder = TiffDecoder::new();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.bits_per_sample, vec![16]);
    }

    #[test]
    fn test_encode_multiple_strips() {
        let width = 16;
        let height = 64;
        let data: Vec<u8> = (0..(width * height) as usize).map(|i| (i % 256) as u8).collect();

        let image = TiffEncoder::from_grayscale_8bit(width as u32, height as u32, &data);
        let encoder = TiffEncoder::new().rows_per_strip(16);
        let encoded = encoder.encode(&image).unwrap();

        let decoder = TiffDecoder::new();
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded.data, data);
    }
}
