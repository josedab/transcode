//! Core image types.

use crate::error::{ImageError, Result};

/// Pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Grayscale, 8 bits per pixel.
    Gray8,
    /// Grayscale with alpha, 16 bits per pixel.
    GrayA8,
    /// RGB, 24 bits per pixel.
    Rgb8,
    /// RGBA, 32 bits per pixel.
    Rgba8,
    /// BGR, 24 bits per pixel.
    Bgr8,
    /// BGRA, 32 bits per pixel.
    Bgra8,
    /// YUV 4:2:0 planar.
    Yuv420p,
    /// YUV 4:2:2 planar.
    Yuv422p,
    /// YUV 4:4:4 planar.
    Yuv444p,
    /// Indexed color (palette).
    Indexed8,
    /// 16-bit grayscale.
    Gray16,
    /// 16-bit RGB.
    Rgb16,
    /// 16-bit RGBA.
    Rgba16,
}

impl PixelFormat {
    /// Get bytes per pixel (for packed formats).
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::Gray8 | PixelFormat::Indexed8 => 1,
            PixelFormat::GrayA8 | PixelFormat::Gray16 => 2,
            PixelFormat::Rgb8 | PixelFormat::Bgr8 => 3,
            PixelFormat::Rgba8 | PixelFormat::Bgra8 => 4,
            PixelFormat::Rgb16 => 6,
            PixelFormat::Rgba16 => 8,
            // Planar formats don't have a simple bytes per pixel
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => 0,
        }
    }

    /// Get number of channels.
    pub fn channels(&self) -> u8 {
        match self {
            PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::Indexed8 => 1,
            PixelFormat::GrayA8 => 2,
            PixelFormat::Rgb8 | PixelFormat::Bgr8 | PixelFormat::Rgb16
            | PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => 3,
            PixelFormat::Rgba8 | PixelFormat::Bgra8 | PixelFormat::Rgba16 => 4,
        }
    }

    /// Check if format has alpha channel.
    pub fn has_alpha(&self) -> bool {
        matches!(
            self,
            PixelFormat::GrayA8 | PixelFormat::Rgba8 | PixelFormat::Bgra8
            | PixelFormat::Rgba16
        )
    }

    /// Check if format is planar.
    pub fn is_planar(&self) -> bool {
        matches!(
            self,
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p
        )
    }
}

/// Color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// Grayscale.
    Gray,
    /// Standard RGB.
    Srgb,
    /// Linear RGB.
    LinearRgb,
    /// YCbCr (JPEG default).
    YCbCr,
    /// Adobe RGB.
    AdobeRgb,
    /// Display P3.
    DisplayP3,
    /// BT.709.
    Bt709,
    /// BT.2020.
    Bt2020,
}

/// Image data structure.
#[derive(Debug, Clone)]
pub struct Image {
    /// Image width.
    width: u32,
    /// Image height.
    height: u32,
    /// Pixel format.
    format: PixelFormat,
    /// Color space.
    color_space: ColorSpace,
    /// Pixel data.
    data: Vec<u8>,
    /// Row stride (bytes per row).
    stride: usize,
    /// Color palette (for indexed images).
    palette: Option<Vec<[u8; 4]>>,
}

impl Image {
    /// Create a new image.
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions { width, height });
        }

        let bytes_per_pixel = format.bytes_per_pixel();
        let stride = if bytes_per_pixel > 0 {
            width as usize * bytes_per_pixel
        } else {
            // For planar formats, stride is width
            width as usize
        };

        let data_size = Self::calculate_size(width, height, format);
        let data = vec![0u8; data_size];

        Ok(Self {
            width,
            height,
            format,
            color_space: ColorSpace::Srgb,
            data,
            stride,
            palette: None,
        })
    }

    /// Create an image from existing data.
    pub fn from_data(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions { width, height });
        }

        let expected_size = Self::calculate_size(width, height, format);
        if data.len() < expected_size {
            return Err(ImageError::TruncatedData {
                expected: expected_size,
                actual: data.len(),
            });
        }

        let bytes_per_pixel = format.bytes_per_pixel();
        let stride = if bytes_per_pixel > 0 {
            width as usize * bytes_per_pixel
        } else {
            width as usize
        };

        Ok(Self {
            width,
            height,
            format,
            color_space: ColorSpace::Srgb,
            data,
            stride,
            palette: None,
        })
    }

    /// Calculate required data size.
    fn calculate_size(width: u32, height: u32, format: PixelFormat) -> usize {
        let w = width as usize;
        let h = height as usize;

        match format {
            PixelFormat::Yuv420p => w * h + 2 * w.div_ceil(2) * h.div_ceil(2),
            PixelFormat::Yuv422p => w * h + 2 * w.div_ceil(2) * h,
            PixelFormat::Yuv444p => 3 * w * h,
            _ => w * h * format.bytes_per_pixel(),
        }
    }

    /// Get image width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get image height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get pixel format.
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Get color space.
    pub fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    /// Set color space.
    pub fn set_color_space(&mut self, color_space: ColorSpace) {
        self.color_space = color_space;
    }

    /// Get row stride.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get pixel data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable pixel data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get palette (for indexed images).
    pub fn palette(&self) -> Option<&[[u8; 4]]> {
        self.palette.as_deref()
    }

    /// Set palette (for indexed images).
    pub fn set_palette(&mut self, palette: Vec<[u8; 4]>) {
        self.palette = Some(palette);
    }

    /// Get a row of pixels.
    pub fn row(&self, y: u32) -> &[u8] {
        let start = y as usize * self.stride;
        let end = start + self.stride;
        &self.data[start..end]
    }

    /// Get a mutable row of pixels.
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        let start = y as usize * self.stride;
        let end = start + self.stride;
        &mut self.data[start..end]
    }

    /// Get a pixel value.
    pub fn get_pixel(&self, x: u32, y: u32) -> &[u8] {
        let bpp = self.format.bytes_per_pixel();
        if bpp == 0 {
            return &[];
        }
        let offset = y as usize * self.stride + x as usize * bpp;
        &self.data[offset..offset + bpp]
    }

    /// Set a pixel value.
    pub fn set_pixel(&mut self, x: u32, y: u32, pixel: &[u8]) {
        let bpp = self.format.bytes_per_pixel();
        if bpp == 0 || pixel.len() < bpp {
            return;
        }
        let offset = y as usize * self.stride + x as usize * bpp;
        self.data[offset..offset + bpp].copy_from_slice(&pixel[..bpp]);
    }

    /// Convert to a different pixel format.
    pub fn convert(&self, target_format: PixelFormat) -> Result<Image> {
        if self.format == target_format {
            return Ok(self.clone());
        }

        let mut output = Image::new(self.width, self.height, target_format)?;

        // Simple conversions
        match (self.format, target_format) {
            (PixelFormat::Rgb8, PixelFormat::Rgba8) => {
                for y in 0..self.height {
                    for x in 0..self.width {
                        let src = self.get_pixel(x, y);
                        output.set_pixel(x, y, &[src[0], src[1], src[2], 255]);
                    }
                }
            }
            (PixelFormat::Rgba8, PixelFormat::Rgb8) => {
                for y in 0..self.height {
                    for x in 0..self.width {
                        let src = self.get_pixel(x, y);
                        output.set_pixel(x, y, &[src[0], src[1], src[2]]);
                    }
                }
            }
            (PixelFormat::Rgb8, PixelFormat::Gray8) => {
                for y in 0..self.height {
                    for x in 0..self.width {
                        let src = self.get_pixel(x, y);
                        let gray = (src[0] as u32 * 299 + src[1] as u32 * 587 + src[2] as u32 * 114) / 1000;
                        output.set_pixel(x, y, &[gray as u8]);
                    }
                }
            }
            (PixelFormat::Gray8, PixelFormat::Rgb8) => {
                for y in 0..self.height {
                    for x in 0..self.width {
                        let gray = self.get_pixel(x, y)[0];
                        output.set_pixel(x, y, &[gray, gray, gray]);
                    }
                }
            }
            _ => {
                return Err(ImageError::UnsupportedFormat(format!(
                    "Conversion from {:?} to {:?} not supported",
                    self.format, target_format
                )));
            }
        }

        Ok(output)
    }

    /// Flip image horizontally.
    pub fn flip_horizontal(&mut self) {
        let bpp = self.format.bytes_per_pixel();
        if bpp == 0 {
            return;
        }

        for y in 0..self.height {
            let row_start = y as usize * self.stride;
            for x in 0..self.width / 2 {
                let left = row_start + x as usize * bpp;
                let right = row_start + (self.width - 1 - x) as usize * bpp;

                for i in 0..bpp {
                    self.data.swap(left + i, right + i);
                }
            }
        }
    }

    /// Flip image vertically.
    pub fn flip_vertical(&mut self) {
        for y in 0..self.height / 2 {
            let top = y as usize * self.stride;
            let bottom = (self.height - 1 - y) as usize * self.stride;

            for x in 0..self.stride {
                self.data.swap(top + x, bottom + x);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_creation() {
        let image = Image::new(100, 100, PixelFormat::Rgb8).unwrap();
        assert_eq!(image.width(), 100);
        assert_eq!(image.height(), 100);
        assert_eq!(image.data().len(), 100 * 100 * 3);
    }

    #[test]
    fn test_pixel_format() {
        assert_eq!(PixelFormat::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Rgba8.bytes_per_pixel(), 4);
        assert!(PixelFormat::Rgba8.has_alpha());
        assert!(!PixelFormat::Rgb8.has_alpha());
    }

    #[test]
    fn test_pixel_access() {
        let mut image = Image::new(10, 10, PixelFormat::Rgb8).unwrap();
        image.set_pixel(5, 5, &[255, 128, 64]);
        let pixel = image.get_pixel(5, 5);
        assert_eq!(pixel, &[255, 128, 64]);
    }

    #[test]
    fn test_conversion() {
        let mut image = Image::new(10, 10, PixelFormat::Rgb8).unwrap();
        image.set_pixel(0, 0, &[255, 0, 0]); // Red

        let rgba = image.convert(PixelFormat::Rgba8).unwrap();
        let pixel = rgba.get_pixel(0, 0);
        assert_eq!(pixel, &[255, 0, 0, 255]);
    }

    #[test]
    fn test_invalid_dimensions() {
        let result = Image::new(0, 100, PixelFormat::Rgb8);
        assert!(matches!(result, Err(ImageError::InvalidDimensions { .. })));
    }
}
