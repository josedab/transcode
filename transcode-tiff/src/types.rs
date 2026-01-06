//! TIFF type definitions

use std::fmt;

/// Color space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// Grayscale (1 component)
    Grayscale,
    /// Grayscale with alpha (2 components)
    GrayscaleAlpha,
    /// RGB (3 components)
    Rgb,
    /// RGBA (4 components)
    Rgba,
    /// CMYK (4 components)
    Cmyk,
    /// Palette-indexed
    Palette,
    /// YCbCr
    YCbCr,
}

impl Default for ColorSpace {
    fn default() -> Self {
        ColorSpace::Rgb
    }
}

impl ColorSpace {
    /// Number of samples per pixel
    pub fn samples_per_pixel(&self) -> usize {
        match self {
            ColorSpace::Grayscale | ColorSpace::Palette => 1,
            ColorSpace::GrayscaleAlpha => 2,
            ColorSpace::Rgb | ColorSpace::YCbCr => 3,
            ColorSpace::Rgba | ColorSpace::Cmyk => 4,
        }
    }

    /// Has alpha channel
    pub fn has_alpha(&self) -> bool {
        matches!(self, ColorSpace::GrayscaleAlpha | ColorSpace::Rgba)
    }
}

/// Sample format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// Unsigned integer
    Uint,
    /// Signed integer
    Int,
    /// IEEE floating point
    Float,
    /// Undefined
    Undefined,
}

impl Default for SampleFormat {
    fn default() -> Self {
        SampleFormat::Uint
    }
}

impl SampleFormat {
    /// Create from TIFF value
    pub fn from_u16(value: u16) -> Self {
        match value {
            1 => SampleFormat::Uint,
            2 => SampleFormat::Int,
            3 => SampleFormat::Float,
            _ => SampleFormat::Undefined,
        }
    }

    /// Convert to TIFF value
    pub fn to_u16(self) -> u16 {
        match self {
            SampleFormat::Uint => 1,
            SampleFormat::Int => 2,
            SampleFormat::Float => 3,
            SampleFormat::Undefined => 4,
        }
    }
}

/// Photometric interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotometricInterpretation {
    /// WhiteIsZero - min value is white
    WhiteIsZero,
    /// BlackIsZero - min value is black
    BlackIsZero,
    /// RGB color
    Rgb,
    /// Palette color (indexed)
    Palette,
    /// Transparency mask
    TransparencyMask,
    /// CMYK
    Cmyk,
    /// YCbCr
    YCbCr,
    /// CIE L*a*b*
    CieLab,
}

impl Default for PhotometricInterpretation {
    fn default() -> Self {
        PhotometricInterpretation::Rgb
    }
}

impl PhotometricInterpretation {
    /// Create from TIFF value
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0 => Some(PhotometricInterpretation::WhiteIsZero),
            1 => Some(PhotometricInterpretation::BlackIsZero),
            2 => Some(PhotometricInterpretation::Rgb),
            3 => Some(PhotometricInterpretation::Palette),
            4 => Some(PhotometricInterpretation::TransparencyMask),
            5 => Some(PhotometricInterpretation::Cmyk),
            6 => Some(PhotometricInterpretation::YCbCr),
            8 => Some(PhotometricInterpretation::CieLab),
            _ => None,
        }
    }

    /// Convert to TIFF value
    pub fn to_u16(self) -> u16 {
        match self {
            PhotometricInterpretation::WhiteIsZero => 0,
            PhotometricInterpretation::BlackIsZero => 1,
            PhotometricInterpretation::Rgb => 2,
            PhotometricInterpretation::Palette => 3,
            PhotometricInterpretation::TransparencyMask => 4,
            PhotometricInterpretation::Cmyk => 5,
            PhotometricInterpretation::YCbCr => 6,
            PhotometricInterpretation::CieLab => 8,
        }
    }
}

/// Planar configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarConfig {
    /// Chunky format (RGBRGBRGB...)
    Chunky,
    /// Planar format (RRR...GGG...BBB...)
    Planar,
}

impl Default for PlanarConfig {
    fn default() -> Self {
        PlanarConfig::Chunky
    }
}

impl PlanarConfig {
    /// Create from TIFF value
    pub fn from_u16(value: u16) -> Self {
        match value {
            2 => PlanarConfig::Planar,
            _ => PlanarConfig::Chunky,
        }
    }

    /// Convert to TIFF value
    pub fn to_u16(self) -> u16 {
        match self {
            PlanarConfig::Chunky => 1,
            PlanarConfig::Planar => 2,
        }
    }
}

/// Resolution unit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionUnit {
    /// No unit
    None,
    /// Inches
    Inch,
    /// Centimeters
    Centimeter,
}

impl Default for ResolutionUnit {
    fn default() -> Self {
        ResolutionUnit::Inch
    }
}

impl ResolutionUnit {
    /// Create from TIFF value
    pub fn from_u16(value: u16) -> Self {
        match value {
            2 => ResolutionUnit::Inch,
            3 => ResolutionUnit::Centimeter,
            _ => ResolutionUnit::None,
        }
    }

    /// Convert to TIFF value
    pub fn to_u16(self) -> u16 {
        match self {
            ResolutionUnit::None => 1,
            ResolutionUnit::Inch => 2,
            ResolutionUnit::Centimeter => 3,
        }
    }
}

/// TIFF image
#[derive(Debug, Clone)]
pub struct TiffImage {
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
    /// Bits per sample
    pub bits_per_sample: Vec<u16>,
    /// Samples per pixel
    pub samples_per_pixel: u16,
    /// Color space
    pub color_space: ColorSpace,
    /// Sample format
    pub sample_format: SampleFormat,
    /// Photometric interpretation
    pub photometric: PhotometricInterpretation,
    /// Planar configuration
    pub planar_config: PlanarConfig,
    /// Resolution X
    pub resolution_x: f64,
    /// Resolution Y
    pub resolution_y: f64,
    /// Resolution unit
    pub resolution_unit: ResolutionUnit,
    /// Pixel data (raw bytes)
    pub data: Vec<u8>,
    /// Color palette (for indexed images)
    pub palette: Option<Vec<[u16; 3]>>,
}

impl Default for TiffImage {
    fn default() -> Self {
        TiffImage {
            width: 0,
            height: 0,
            bits_per_sample: vec![8],
            samples_per_pixel: 3,
            color_space: ColorSpace::Rgb,
            sample_format: SampleFormat::Uint,
            photometric: PhotometricInterpretation::Rgb,
            planar_config: PlanarConfig::Chunky,
            resolution_x: 72.0,
            resolution_y: 72.0,
            resolution_unit: ResolutionUnit::Inch,
            data: Vec::new(),
            palette: None,
        }
    }
}

impl TiffImage {
    /// Create new image
    pub fn new(width: u32, height: u32, color_space: ColorSpace) -> Self {
        let samples_per_pixel = color_space.samples_per_pixel() as u16;
        let bits_per_sample = vec![8; samples_per_pixel as usize];
        let photometric = match color_space {
            ColorSpace::Grayscale | ColorSpace::GrayscaleAlpha => {
                PhotometricInterpretation::BlackIsZero
            }
            ColorSpace::Rgb | ColorSpace::Rgba => PhotometricInterpretation::Rgb,
            ColorSpace::Cmyk => PhotometricInterpretation::Cmyk,
            ColorSpace::Palette => PhotometricInterpretation::Palette,
            ColorSpace::YCbCr => PhotometricInterpretation::YCbCr,
        };

        let bytes_per_row = (width as usize) * (samples_per_pixel as usize);
        let data = vec![0u8; bytes_per_row * (height as usize)];

        TiffImage {
            width,
            height,
            bits_per_sample,
            samples_per_pixel,
            color_space,
            sample_format: SampleFormat::Uint,
            photometric,
            planar_config: PlanarConfig::Chunky,
            resolution_x: 72.0,
            resolution_y: 72.0,
            resolution_unit: ResolutionUnit::Inch,
            data,
            palette: None,
        }
    }

    /// Bytes per pixel
    pub fn bytes_per_pixel(&self) -> usize {
        self.bits_per_sample.iter().map(|b| (*b as usize + 7) / 8).sum()
    }

    /// Bytes per row
    pub fn bytes_per_row(&self) -> usize {
        self.bytes_per_pixel() * self.width as usize
    }

    /// Total image size in bytes
    pub fn data_size(&self) -> usize {
        self.bytes_per_row() * self.height as usize
    }

    /// Get pixel at (x, y) as raw bytes
    pub fn get_pixel_bytes(&self, x: u32, y: u32) -> Option<&[u8]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let bpp = self.bytes_per_pixel();
        let offset = (y as usize * self.bytes_per_row()) + (x as usize * bpp);

        if offset + bpp <= self.data.len() {
            Some(&self.data[offset..offset + bpp])
        } else {
            None
        }
    }

    /// Set pixel at (x, y) from raw bytes
    pub fn set_pixel_bytes(&mut self, x: u32, y: u32, bytes: &[u8]) {
        if x >= self.width || y >= self.height {
            return;
        }

        let bpp = self.bytes_per_pixel();
        let offset = (y as usize * self.bytes_per_row()) + (x as usize * bpp);

        if offset + bpp <= self.data.len() && bytes.len() >= bpp {
            self.data[offset..offset + bpp].copy_from_slice(&bytes[..bpp]);
        }
    }

    /// Convert to 8-bit RGBA
    pub fn to_rgba_8bit(&self) -> Vec<u8> {
        let pixels = (self.width * self.height) as usize;
        let mut output = Vec::with_capacity(pixels * 4);

        for y in 0..self.height {
            for x in 0..self.width {
                let rgba = self.get_pixel_rgba(x, y);
                output.push(rgba[0]);
                output.push(rgba[1]);
                output.push(rgba[2]);
                output.push(rgba[3]);
            }
        }

        output
    }

    /// Get pixel as RGBA values
    pub fn get_pixel_rgba(&self, x: u32, y: u32) -> [u8; 4] {
        let Some(bytes) = self.get_pixel_bytes(x, y) else {
            return [0, 0, 0, 255];
        };

        match self.color_space {
            ColorSpace::Grayscale => {
                let v = bytes.first().copied().unwrap_or(0);
                [v, v, v, 255]
            }
            ColorSpace::GrayscaleAlpha => {
                let v = bytes.first().copied().unwrap_or(0);
                let a = bytes.get(1).copied().unwrap_or(255);
                [v, v, v, a]
            }
            ColorSpace::Rgb => {
                let r = bytes.first().copied().unwrap_or(0);
                let g = bytes.get(1).copied().unwrap_or(0);
                let b = bytes.get(2).copied().unwrap_or(0);
                [r, g, b, 255]
            }
            ColorSpace::Rgba => {
                let r = bytes.first().copied().unwrap_or(0);
                let g = bytes.get(1).copied().unwrap_or(0);
                let b = bytes.get(2).copied().unwrap_or(0);
                let a = bytes.get(3).copied().unwrap_or(255);
                [r, g, b, a]
            }
            ColorSpace::Cmyk => {
                // Simple CMYK to RGB conversion
                let c = bytes.first().copied().unwrap_or(0) as f32 / 255.0;
                let m = bytes.get(1).copied().unwrap_or(0) as f32 / 255.0;
                let yy = bytes.get(2).copied().unwrap_or(0) as f32 / 255.0;
                let k = bytes.get(3).copied().unwrap_or(0) as f32 / 255.0;

                let r = ((1.0 - c) * (1.0 - k) * 255.0) as u8;
                let g = ((1.0 - m) * (1.0 - k) * 255.0) as u8;
                let b = ((1.0 - yy) * (1.0 - k) * 255.0) as u8;
                [r, g, b, 255]
            }
            ColorSpace::Palette => {
                // Palette index - look up in palette
                let idx = bytes.first().copied().unwrap_or(0) as usize;
                if let Some(ref palette) = self.palette {
                    if let Some(entry) = palette.get(idx) {
                        let r = (entry[0] >> 8) as u8;
                        let g = (entry[1] >> 8) as u8;
                        let b = (entry[2] >> 8) as u8;
                        [r, g, b, 255]
                    } else {
                        [0, 0, 0, 255]
                    }
                } else {
                    [0, 0, 0, 255]
                }
            }
            ColorSpace::YCbCr => {
                // YCbCr to RGB conversion
                let y = bytes.first().copied().unwrap_or(0) as f32;
                let cb = bytes.get(1).copied().unwrap_or(128) as f32 - 128.0;
                let cr = bytes.get(2).copied().unwrap_or(128) as f32 - 128.0;

                let r = (y + 1.402 * cr).clamp(0.0, 255.0) as u8;
                let g = (y - 0.344136 * cb - 0.714136 * cr).clamp(0.0, 255.0) as u8;
                let b = (y + 1.772 * cb).clamp(0.0, 255.0) as u8;
                [r, g, b, 255]
            }
        }
    }
}

impl fmt::Display for TiffImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TIFF {}x{} {:?} {}bpp",
            self.width,
            self.height,
            self.color_space,
            self.bits_per_sample.first().copied().unwrap_or(8)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_space() {
        assert_eq!(ColorSpace::Rgb.samples_per_pixel(), 3);
        assert_eq!(ColorSpace::Rgba.samples_per_pixel(), 4);
        assert_eq!(ColorSpace::Grayscale.samples_per_pixel(), 1);
        assert!(ColorSpace::Rgba.has_alpha());
        assert!(!ColorSpace::Rgb.has_alpha());
    }

    #[test]
    fn test_tiff_image() {
        let image = TiffImage::new(100, 100, ColorSpace::Rgb);
        assert_eq!(image.width, 100);
        assert_eq!(image.height, 100);
        assert_eq!(image.bytes_per_pixel(), 3);
        assert_eq!(image.bytes_per_row(), 300);
    }

    #[test]
    fn test_pixel_access() {
        let mut image = TiffImage::new(10, 10, ColorSpace::Rgb);
        image.set_pixel_bytes(5, 5, &[255, 128, 64]);

        let pixel = image.get_pixel_bytes(5, 5).unwrap();
        assert_eq!(pixel, &[255, 128, 64]);
    }

    #[test]
    fn test_photometric() {
        assert_eq!(
            PhotometricInterpretation::from_u16(2),
            Some(PhotometricInterpretation::Rgb)
        );
        assert_eq!(PhotometricInterpretation::Rgb.to_u16(), 2);
    }

    #[test]
    fn test_sample_format() {
        assert_eq!(SampleFormat::from_u16(1), SampleFormat::Uint);
        assert_eq!(SampleFormat::from_u16(3), SampleFormat::Float);
    }
}
