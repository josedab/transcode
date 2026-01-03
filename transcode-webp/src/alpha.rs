//! Alpha channel decoder for WebP
//!
//! The ALPH chunk contains alpha channel data that can be:
//! - Uncompressed (raw bytes)
//! - Lossless compressed (VP8L without color)
//!
//! The alpha data has a 1-byte header followed by the compressed/uncompressed data.

use crate::error::{WebPError, Result};
use image::RgbaImage;

/// Alpha filtering methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaFilter {
    /// No filtering
    None,
    /// Horizontal filter
    Horizontal,
    /// Vertical filter
    Vertical,
    /// Gradient filter
    Gradient,
}

impl AlphaFilter {
    fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => AlphaFilter::None,
            1 => AlphaFilter::Horizontal,
            2 => AlphaFilter::Vertical,
            3 => AlphaFilter::Gradient,
            _ => unreachable!(),
        }
    }
}

/// Alpha compression methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaCompression {
    /// No compression (raw bytes)
    None,
    /// VP8L lossless compression
    Lossless,
}

impl AlphaCompression {
    fn from_bits(bits: u8) -> Result<Self> {
        match bits & 0x03 {
            0 => Ok(AlphaCompression::None),
            1 => Ok(AlphaCompression::Lossless),
            _ => Err(WebPError::InvalidAlpha("Invalid compression method".into())),
        }
    }
}

/// Alpha channel decoder
pub struct AlphaDecoder<'a> {
    data: &'a [u8],
    filter: AlphaFilter,
    compression: AlphaCompression,
    #[allow(dead_code)]
    pre_processing: bool,
}

impl<'a> AlphaDecoder<'a> {
    /// Create a new alpha decoder from ALPH chunk data
    pub fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(WebPError::InvalidAlpha("Empty alpha data".into()));
        }

        // Parse header byte
        let header = data[0];
        let filter = AlphaFilter::from_bits((header >> 4) & 0x03);
        let compression = AlphaCompression::from_bits(header & 0x03)?;
        let pre_processing = ((header >> 2) & 0x03) != 0;

        Ok(Self {
            data: &data[1..],
            filter,
            compression,
            pre_processing,
        })
    }

    /// Get the filter method
    pub fn filter(&self) -> AlphaFilter {
        self.filter
    }

    /// Get the compression method
    pub fn compression(&self) -> AlphaCompression {
        self.compression
    }

    /// Apply alpha channel to an existing RGBA image
    pub fn apply_alpha(&self, image: &mut RgbaImage) -> Result<()> {
        let width = image.width() as usize;
        let height = image.height() as usize;

        // Decode alpha values
        let alpha = self.decode_alpha(width, height)?;

        // Apply alpha to image
        for (i, pixel) in image.pixels_mut().enumerate() {
            if i < alpha.len() {
                pixel.0[3] = alpha[i];
            }
        }

        Ok(())
    }

    /// Decode alpha channel to raw bytes
    pub fn decode_alpha(&self, width: usize, height: usize) -> Result<Vec<u8>> {
        let size = width * height;

        let mut alpha = match self.compression {
            AlphaCompression::None => self.decode_uncompressed(size)?,
            AlphaCompression::Lossless => self.decode_lossless(width, height)?,
        };

        // Apply inverse filter
        self.apply_inverse_filter(&mut alpha, width, height)?;

        Ok(alpha)
    }

    fn decode_uncompressed(&self, size: usize) -> Result<Vec<u8>> {
        if self.data.len() < size {
            return Err(WebPError::BufferTooSmall {
                expected: size,
                actual: self.data.len(),
            });
        }

        Ok(self.data[..size].to_vec())
    }

    fn decode_lossless(&self, width: usize, height: usize) -> Result<Vec<u8>> {
        // VP8L lossless compression for alpha
        // The alpha is stored as the green channel of a VP8L-compressed image
        use crate::vp8l::Vp8lDecoder;

        // Build a minimal VP8L header
        let mut vp8l_data = Vec::with_capacity(5 + self.data.len());
        vp8l_data.push(0x2f); // VP8L signature

        // Pack width and height
        let w = (width - 1) as u32;
        let h = (height - 1) as u32;
        let bits = w | (h << 14);
        vp8l_data.push((bits & 0xFF) as u8);
        vp8l_data.push(((bits >> 8) & 0xFF) as u8);
        vp8l_data.push(((bits >> 16) & 0xFF) as u8);
        vp8l_data.push(((bits >> 24) & 0xFF) as u8);
        vp8l_data.extend_from_slice(self.data);

        let decoder = Vp8lDecoder::new(&vp8l_data)?;
        let image = decoder.decode()?;

        // Extract green channel as alpha
        let mut alpha = Vec::with_capacity(width * height);
        for pixel in image.pixels() {
            alpha.push(pixel.0[1]); // Green channel
        }

        Ok(alpha)
    }

    fn apply_inverse_filter(
        &self,
        alpha: &mut [u8],
        width: usize,
        height: usize,
    ) -> Result<()> {
        match self.filter {
            AlphaFilter::None => {}
            AlphaFilter::Horizontal => {
                for y in 0..height {
                    for x in 1..width {
                        let idx = y * width + x;
                        let left = alpha[idx - 1];
                        alpha[idx] = alpha[idx].wrapping_add(left);
                    }
                }
            }
            AlphaFilter::Vertical => {
                for y in 1..height {
                    for x in 0..width {
                        let idx = y * width + x;
                        let top = alpha[idx - width];
                        alpha[idx] = alpha[idx].wrapping_add(top);
                    }
                }
            }
            AlphaFilter::Gradient => {
                for y in 0..height {
                    for x in 0..width {
                        let idx = y * width + x;

                        let left = if x > 0 { alpha[idx - 1] } else { 0 };
                        let top = if y > 0 { alpha[idx - width] } else { 0 };
                        let top_left = if x > 0 && y > 0 {
                            alpha[idx - width - 1]
                        } else {
                            0
                        };

                        // Gradient prediction: left + top - top_left
                        let pred = gradient_predict(left, top, top_left);
                        alpha[idx] = alpha[idx].wrapping_add(pred);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Gradient prediction (clamped to [0, 255])
fn gradient_predict(left: u8, top: u8, top_left: u8) -> u8 {
    let prediction = i32::from(left) + i32::from(top) - i32::from(top_left);
    prediction.clamp(0, 255) as u8
}

/// Decode alpha from an ALPH chunk and dimensions
pub fn decode_alpha_chunk(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let decoder = AlphaDecoder::new(data)?;
    decoder.decode_alpha(width as usize, height as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_filter_parsing() {
        assert_eq!(AlphaFilter::from_bits(0), AlphaFilter::None);
        assert_eq!(AlphaFilter::from_bits(1), AlphaFilter::Horizontal);
        assert_eq!(AlphaFilter::from_bits(2), AlphaFilter::Vertical);
        assert_eq!(AlphaFilter::from_bits(3), AlphaFilter::Gradient);
    }

    #[test]
    fn test_alpha_compression_parsing() {
        assert_eq!(
            AlphaCompression::from_bits(0).unwrap(),
            AlphaCompression::None
        );
        assert_eq!(
            AlphaCompression::from_bits(1).unwrap(),
            AlphaCompression::Lossless
        );
        assert!(AlphaCompression::from_bits(2).is_err());
    }

    #[test]
    fn test_alpha_decoder_creation() {
        // Header byte: no filter, no compression, no pre-processing
        let data = [0x00, 0xFF, 0xFF, 0xFF, 0xFF];
        let decoder = AlphaDecoder::new(&data);
        assert!(decoder.is_ok());

        let decoder = decoder.unwrap();
        assert_eq!(decoder.filter(), AlphaFilter::None);
        assert_eq!(decoder.compression(), AlphaCompression::None);
    }

    #[test]
    fn test_decode_uncompressed() {
        // Header: no filter, no compression
        let data = [0x00, 0x80, 0x80, 0x80, 0x80];
        let decoder = AlphaDecoder::new(&data).unwrap();
        let alpha = decoder.decode_alpha(2, 2).unwrap();
        assert_eq!(alpha, vec![0x80, 0x80, 0x80, 0x80]);
    }

    #[test]
    fn test_gradient_predict() {
        assert_eq!(gradient_predict(100, 100, 100), 100);
        assert_eq!(gradient_predict(100, 50, 25), 125);
        assert_eq!(gradient_predict(255, 255, 0), 255); // Clamped
        assert_eq!(gradient_predict(0, 0, 255), 0); // Clamped
    }

    #[test]
    fn test_horizontal_filter() {
        // Header: horizontal filter, no compression
        let data = [0x10, 10, 5, 10, 5];
        let decoder = AlphaDecoder::new(&data).unwrap();
        let alpha = decoder.decode_alpha(2, 2).unwrap();
        // First row: 10, 10+5=15
        // Second row: 10, 10+5=15
        assert_eq!(alpha, vec![10, 15, 10, 15]);
    }
}
