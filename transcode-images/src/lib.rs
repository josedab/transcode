// Allow common patterns in multimedia/DSP code
#![allow(dead_code)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! Image codec support for the transcode library.
//!
//! This crate provides image encoding and decoding for common formats:
//! - JPEG (baseline and progressive)
//! - PNG (with alpha channel support)
//! - GIF (with animation support)
//!
//! ## Features
//!
//! - `jpeg` - JPEG encoder/decoder (default)
//! - `png` - PNG encoder/decoder (default)
//! - `gif` - GIF encoder/decoder
//!
//! ## Example
//!
//! ```no_run
//! use transcode_images::{JpegDecoder, JpegEncoder, PngDecoder, PngEncoder};
//! use transcode_images::{Image, PixelFormat};
//!
//! # let jpeg_data: Vec<u8> = vec![];
//! // Decode a JPEG
//! let mut decoder = JpegDecoder::new();
//! let image = decoder.decode(&jpeg_data)?;
//!
//! // Encode as PNG
//! let mut encoder = PngEncoder::new();
//! let png_data = encoder.encode(&image)?;
//! # Ok::<(), transcode_images::ImageError>(())
//! ```

#![warn(missing_docs)]

mod error;
mod image;

#[cfg(feature = "jpeg")]
pub mod jpeg;

#[cfg(feature = "png")]
pub mod png;

#[cfg(feature = "gif")]
pub mod gif;

pub use error::{ImageError, Result};
pub use image::{Image, PixelFormat, ColorSpace};

#[cfg(feature = "jpeg")]
pub use jpeg::{JpegDecoder, JpegEncoder, JpegConfig, JpegInfo};

#[cfg(feature = "png")]
pub use png::{PngDecoder, PngEncoder, PngConfig, PngInfo};

#[cfg(feature = "gif")]
pub use gif::{GifDecoder, GifEncoder, GifConfig, GifFrame};

/// Detect image format from magic bytes.
pub fn detect_format(data: &[u8]) -> Option<ImageFormat> {
    if data.len() < 8 {
        return None;
    }

    // JPEG: FF D8 FF
    if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
        return Some(ImageFormat::Jpeg);
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if &data[0..8] == b"\x89PNG\r\n\x1a\n" {
        return Some(ImageFormat::Png);
    }

    // GIF: GIF87a or GIF89a
    if &data[0..6] == b"GIF87a" || &data[0..6] == b"GIF89a" {
        return Some(ImageFormat::Gif);
    }

    // WebP: RIFF....WEBP
    if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        return Some(ImageFormat::WebP);
    }

    // BMP: BM
    if &data[0..2] == b"BM" {
        return Some(ImageFormat::Bmp);
    }

    None
}

/// Image format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// JPEG image.
    Jpeg,
    /// PNG image.
    Png,
    /// GIF image.
    Gif,
    /// WebP image.
    WebP,
    /// BMP image.
    Bmp,
    /// TIFF image.
    Tiff,
}

impl ImageFormat {
    /// Get file extension for format.
    pub fn extension(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Png => "png",
            ImageFormat::Gif => "gif",
            ImageFormat::WebP => "webp",
            ImageFormat::Bmp => "bmp",
            ImageFormat::Tiff => "tiff",
        }
    }

    /// Get MIME type for format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "image/jpeg",
            ImageFormat::Png => "image/png",
            ImageFormat::Gif => "image/gif",
            ImageFormat::WebP => "image/webp",
            ImageFormat::Bmp => "image/bmp",
            ImageFormat::Tiff => "image/tiff",
        }
    }

    /// Check if format supports transparency.
    pub fn supports_alpha(&self) -> bool {
        matches!(self, ImageFormat::Png | ImageFormat::Gif | ImageFormat::WebP)
    }

    /// Check if format supports animation.
    pub fn supports_animation(&self) -> bool {
        matches!(self, ImageFormat::Gif | ImageFormat::WebP | ImageFormat::Png)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_jpeg() {
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert_eq!(detect_format(&data), Some(ImageFormat::Jpeg));
    }

    #[test]
    fn test_detect_png() {
        let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(detect_format(&data), Some(ImageFormat::Png));
    }

    #[test]
    fn test_detect_gif() {
        let data = b"GIF89a\x00\x00";
        assert_eq!(detect_format(data), Some(ImageFormat::Gif));
    }

    #[test]
    fn test_format_properties() {
        assert!(ImageFormat::Png.supports_alpha());
        assert!(!ImageFormat::Jpeg.supports_alpha());
        assert!(ImageFormat::Gif.supports_animation());
        assert_eq!(ImageFormat::Jpeg.extension(), "jpg");
    }
}
