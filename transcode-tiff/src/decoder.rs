//! TIFF decoder

use crate::compression::{decompress, reverse_horizontal_predictor, Compression};
use crate::error::{Result, TiffError};
use crate::ifd::{Ifd, IfdValue};
use crate::tags::tag;
use crate::types::{
    ColorSpace, PhotometricInterpretation, PlanarConfig, ResolutionUnit, SampleFormat, TiffImage,
};
use crate::{TIFF_MAGIC_BE, TIFF_MAGIC_LE, TIFF_VERSION};
use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read, Seek, SeekFrom};

/// TIFF decoder
pub struct TiffDecoder {
    /// Strict mode - fail on any issue
    pub strict: bool,
}

impl Default for TiffDecoder {
    fn default() -> Self {
        TiffDecoder::new()
    }
}

impl TiffDecoder {
    /// Create new decoder
    pub fn new() -> Self {
        TiffDecoder { strict: false }
    }

    /// Enable strict mode
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Decode TIFF data
    pub fn decode(&self, data: &[u8]) -> Result<TiffImage> {
        let mut cursor = Cursor::new(data);
        self.decode_from(&mut cursor)
    }

    /// Decode TIFF from reader
    pub fn decode_from<R: Read + Seek>(&self, reader: &mut R) -> Result<TiffImage> {
        // Read byte order marker
        let mut magic = [0u8; 2];
        reader.read_exact(&mut magic)?;

        let is_little_endian = match magic {
            [0x49, 0x49] => true,  // "II" - Intel
            [0x4D, 0x4D] => false, // "MM" - Motorola
            _ => return Err(TiffError::InvalidMagic),
        };

        if is_little_endian {
            self.decode_endian::<R, LittleEndian>(reader)
        } else {
            self.decode_endian::<R, BigEndian>(reader)
        }
    }

    /// Decode with specific endianness
    fn decode_endian<R: Read + Seek, B: ByteOrder + 'static>(&self, reader: &mut R) -> Result<TiffImage> {
        // Read version
        let version = reader.read_u16::<B>()?;
        if version != TIFF_VERSION {
            return Err(TiffError::UnsupportedVersion { version });
        }

        // Read IFD offset
        let ifd_offset = reader.read_u32::<B>()?;

        // Read first IFD
        let ifd = Ifd::read::<R, B>(reader, ifd_offset)?;

        // Extract image properties
        let width = ifd.get_required_u32(tag::IMAGE_WIDTH)?;
        let height = ifd.get_required_u32(tag::IMAGE_LENGTH)?;

        // Get bits per sample
        let bits_per_sample = match ifd.get_value(tag::BITS_PER_SAMPLE) {
            Some(v) => v.as_u16_vec().unwrap_or_else(|| vec![8]),
            None => vec![8],
        };

        // Get samples per pixel
        let samples_per_pixel = ifd.get_u16_or(tag::SAMPLES_PER_PIXEL, 1);

        // Get photometric interpretation
        let photometric_value = ifd.get_u16_or(tag::PHOTOMETRIC_INTERPRETATION, 2);
        let photometric = PhotometricInterpretation::from_u16(photometric_value)
            .unwrap_or(PhotometricInterpretation::Rgb);

        // Get compression
        let compression_value = ifd.get_u16_or(tag::COMPRESSION, 1);
        let compression = Compression::from_u16(compression_value)
            .ok_or(TiffError::UnsupportedCompression(compression_value))?;

        // Get planar configuration
        let planar_config = PlanarConfig::from_u16(ifd.get_u16_or(tag::PLANAR_CONFIGURATION, 1));

        // Get sample format
        let sample_format = SampleFormat::from_u16(ifd.get_u16_or(tag::SAMPLE_FORMAT, 1));

        // Get predictor
        let predictor = ifd.get_u16_or(tag::PREDICTOR, 1);

        // Determine color space
        let color_space = self.determine_color_space(photometric, samples_per_pixel);

        // Get resolution
        let resolution_x = ifd
            .get_value(tag::X_RESOLUTION)
            .and_then(|v| v.as_f64())
            .unwrap_or(72.0);
        let resolution_y = ifd
            .get_value(tag::Y_RESOLUTION)
            .and_then(|v| v.as_f64())
            .unwrap_or(72.0);
        let resolution_unit =
            ResolutionUnit::from_u16(ifd.get_u16_or(tag::RESOLUTION_UNIT, 2));

        // Read image data
        let data = if ifd.get_value(tag::TILE_WIDTH).is_some() {
            // Tiled image
            self.read_tiled_data::<R, B>(
                reader,
                &ifd,
                width,
                height,
                &bits_per_sample,
                samples_per_pixel,
                compression,
                predictor,
            )?
        } else {
            // Strip-based image
            self.read_strip_data::<R, B>(
                reader,
                &ifd,
                width,
                height,
                &bits_per_sample,
                samples_per_pixel,
                compression,
                predictor,
            )?
        };

        // Read color palette if present
        let palette = if photometric == PhotometricInterpretation::Palette {
            self.read_palette(&ifd)?
        } else {
            None
        };

        Ok(TiffImage {
            width,
            height,
            bits_per_sample,
            samples_per_pixel,
            color_space,
            sample_format,
            photometric,
            planar_config,
            resolution_x,
            resolution_y,
            resolution_unit,
            data,
            palette,
        })
    }

    /// Determine color space from photometric interpretation
    fn determine_color_space(
        &self,
        photometric: PhotometricInterpretation,
        samples_per_pixel: u16,
    ) -> ColorSpace {
        match photometric {
            PhotometricInterpretation::WhiteIsZero | PhotometricInterpretation::BlackIsZero => {
                if samples_per_pixel >= 2 {
                    ColorSpace::GrayscaleAlpha
                } else {
                    ColorSpace::Grayscale
                }
            }
            PhotometricInterpretation::Rgb => {
                if samples_per_pixel >= 4 {
                    ColorSpace::Rgba
                } else {
                    ColorSpace::Rgb
                }
            }
            PhotometricInterpretation::Palette => ColorSpace::Palette,
            PhotometricInterpretation::Cmyk => ColorSpace::Cmyk,
            PhotometricInterpretation::YCbCr => ColorSpace::YCbCr,
            _ => ColorSpace::Rgb,
        }
    }

    /// Read strip-based image data
    fn read_strip_data<R: Read + Seek, B: ByteOrder>(
        &self,
        reader: &mut R,
        ifd: &Ifd,
        width: u32,
        height: u32,
        bits_per_sample: &[u16],
        samples_per_pixel: u16,
        compression: Compression,
        predictor: u16,
    ) -> Result<Vec<u8>> {
        let rows_per_strip = ifd.get_u32_or(tag::ROWS_PER_STRIP, height);

        // Get strip offsets
        let strip_offsets = ifd
            .get_value(tag::STRIP_OFFSETS)
            .and_then(|v| v.as_u32_vec())
            .ok_or_else(|| TiffError::MissingTag("StripOffsets".into()))?;

        // Get strip byte counts
        let strip_byte_counts = ifd
            .get_value(tag::STRIP_BYTE_COUNTS)
            .and_then(|v| v.as_u32_vec())
            .ok_or_else(|| TiffError::MissingTag("StripByteCounts".into()))?;

        // Calculate expected size per row
        let bits_per_pixel: usize = bits_per_sample.iter().map(|&b| b as usize).sum();
        let bytes_per_row = (width as usize * bits_per_pixel + 7) / 8;
        let total_size = bytes_per_row * height as usize;

        let mut output = Vec::with_capacity(total_size);

        for (i, (&offset, &byte_count)) in
            strip_offsets.iter().zip(strip_byte_counts.iter()).enumerate()
        {
            // Calculate rows in this strip
            let strip_start_row = i as u32 * rows_per_strip;
            let rows_in_strip = rows_per_strip.min(height.saturating_sub(strip_start_row));
            let expected_strip_size = bytes_per_row * rows_in_strip as usize;

            // Read compressed data
            reader.seek(SeekFrom::Start(offset as u64))?;
            let mut compressed_data = vec![0u8; byte_count as usize];
            reader.read_exact(&mut compressed_data)?;

            // Decompress
            let mut strip_data = decompress(compression, &compressed_data, expected_strip_size)?;

            // Apply predictor reversal if needed
            if predictor == 2 {
                // Horizontal differencing
                reverse_horizontal_predictor(
                    &mut strip_data,
                    width as usize,
                    samples_per_pixel as usize,
                );
            }

            output.extend_from_slice(&strip_data);
        }

        // Truncate to expected size
        output.truncate(total_size);

        Ok(output)
    }

    /// Read tiled image data
    fn read_tiled_data<R: Read + Seek, B: ByteOrder>(
        &self,
        reader: &mut R,
        ifd: &Ifd,
        width: u32,
        height: u32,
        bits_per_sample: &[u16],
        samples_per_pixel: u16,
        compression: Compression,
        predictor: u16,
    ) -> Result<Vec<u8>> {
        let tile_width = ifd.get_required_u32(tag::TILE_WIDTH)?;
        let tile_height = ifd.get_required_u32(tag::TILE_LENGTH)?;

        // Get tile offsets and byte counts
        let tile_offsets = ifd
            .get_value(tag::TILE_OFFSETS)
            .and_then(|v| v.as_u32_vec())
            .ok_or_else(|| TiffError::MissingTag("TileOffsets".into()))?;

        let tile_byte_counts = ifd
            .get_value(tag::TILE_BYTE_COUNTS)
            .and_then(|v| v.as_u32_vec())
            .ok_or_else(|| TiffError::MissingTag("TileByteCounts".into()))?;

        // Calculate dimensions
        let bits_per_pixel: usize = bits_per_sample.iter().map(|&b| b as usize).sum();
        let bytes_per_pixel = (bits_per_pixel + 7) / 8;
        let tile_bytes_per_row = tile_width as usize * bytes_per_pixel;
        let image_bytes_per_row = width as usize * bytes_per_pixel;
        let tile_size = tile_bytes_per_row * tile_height as usize;

        let tiles_across = (width + tile_width - 1) / tile_width;
        let tiles_down = (height + tile_height - 1) / tile_height;

        let total_size = image_bytes_per_row * height as usize;
        let mut output = vec![0u8; total_size];

        for tile_y in 0..tiles_down {
            for tile_x in 0..tiles_across {
                let tile_idx = (tile_y * tiles_across + tile_x) as usize;

                if tile_idx >= tile_offsets.len() {
                    continue;
                }

                let offset = tile_offsets[tile_idx];
                let byte_count = tile_byte_counts[tile_idx];

                // Read compressed tile data
                reader.seek(SeekFrom::Start(offset as u64))?;
                let mut compressed_data = vec![0u8; byte_count as usize];
                reader.read_exact(&mut compressed_data)?;

                // Decompress
                let mut tile_data = decompress(compression, &compressed_data, tile_size)?;

                // Apply predictor
                if predictor == 2 {
                    reverse_horizontal_predictor(
                        &mut tile_data,
                        tile_width as usize,
                        samples_per_pixel as usize,
                    );
                }

                // Copy tile to output
                let start_x = tile_x * tile_width;
                let start_y = tile_y * tile_height;

                for row in 0..tile_height {
                    let y = start_y + row;
                    if y >= height {
                        break;
                    }

                    let tile_row_start = row as usize * tile_bytes_per_row;
                    let copy_width =
                        (tile_width.min(width - start_x) as usize) * bytes_per_pixel;

                    let src = &tile_data[tile_row_start..tile_row_start + copy_width];
                    let dst_offset = y as usize * image_bytes_per_row
                        + start_x as usize * bytes_per_pixel;

                    if dst_offset + copy_width <= output.len() {
                        output[dst_offset..dst_offset + copy_width].copy_from_slice(src);
                    }
                }
            }
        }

        Ok(output)
    }

    /// Read color palette
    fn read_palette(&self, ifd: &Ifd) -> Result<Option<Vec<[u16; 3]>>> {
        let colormap = match ifd.get_value(tag::COLOR_MAP) {
            Some(IfdValue::Shorts(v)) => v,
            _ => return Ok(None),
        };

        // Color map has 3 * 2^bits_per_sample entries
        // First third is red, second third is green, third is blue
        let entries = colormap.len() / 3;
        let mut palette = Vec::with_capacity(entries);

        for i in 0..entries {
            let r = colormap[i];
            let g = colormap[entries + i];
            let b = colormap[2 * entries + i];
            palette.push([r, g, b]);
        }

        Ok(Some(palette))
    }

    /// Probe if data is a valid TIFF
    pub fn probe(data: &[u8]) -> bool {
        if data.len() < 8 {
            return false;
        }

        // Check magic number
        let magic_ok = &data[0..2] == &TIFF_MAGIC_LE || &data[0..2] == &TIFF_MAGIC_BE;
        if !magic_ok {
            return false;
        }

        // Check version
        let version = if data[0] == 0x49 {
            LittleEndian::read_u16(&data[2..4])
        } else {
            BigEndian::read_u16(&data[2..4])
        };

        version == TIFF_VERSION
    }

    /// Check if this is a TIFF file
    pub fn is_tiff(data: &[u8]) -> bool {
        Self::probe(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_little_endian() {
        let data = [0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00];
        assert!(TiffDecoder::probe(&data));
    }

    #[test]
    fn test_probe_big_endian() {
        let data = [0x4D, 0x4D, 0x00, 0x2A, 0x00, 0x00, 0x00, 0x08];
        assert!(TiffDecoder::probe(&data));
    }

    #[test]
    fn test_probe_invalid() {
        let data = [0x89, 0x50, 0x4E, 0x47]; // PNG magic
        assert!(!TiffDecoder::probe(&data));
    }

    #[test]
    fn test_probe_too_short() {
        let data = [0x49, 0x49];
        assert!(!TiffDecoder::probe(&data));
    }

    #[test]
    fn test_color_space_detection() {
        let decoder = TiffDecoder::new();

        assert_eq!(
            decoder.determine_color_space(PhotometricInterpretation::BlackIsZero, 1),
            ColorSpace::Grayscale
        );
        assert_eq!(
            decoder.determine_color_space(PhotometricInterpretation::BlackIsZero, 2),
            ColorSpace::GrayscaleAlpha
        );
        assert_eq!(
            decoder.determine_color_space(PhotometricInterpretation::Rgb, 3),
            ColorSpace::Rgb
        );
        assert_eq!(
            decoder.determine_color_space(PhotometricInterpretation::Rgb, 4),
            ColorSpace::Rgba
        );
        assert_eq!(
            decoder.determine_color_space(PhotometricInterpretation::Cmyk, 4),
            ColorSpace::Cmyk
        );
    }
}
