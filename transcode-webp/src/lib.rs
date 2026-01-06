//! # transcode-webp
//!
//! A WebP codec library supporting VP8 lossy, VP8L lossless encoding and decoding,
//! animation, and metadata extraction.
//!
//! ## Features
//!
//! - RIFF container parsing and writing
//! - VP8 lossy encoding and decoding
//! - VP8L lossless encoding and decoding
//! - Alpha channel support
//! - Animation support (ANIM, ANMF chunks)
//! - EXIF/XMP metadata extraction
//!
//! ## Decoding Example
//!
//! ```rust,no_run
//! use transcode_webp::WebPDecoder;
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! let file = File::open("image.webp").unwrap();
//! let reader = BufReader::new(file);
//! let decoder = WebPDecoder::new(reader).unwrap();
//! let image = decoder.decode().unwrap();
//! ```
//!
//! ## Encoding Example
//!
//! ```rust,no_run
//! use transcode_webp::{WebPEncoder, EncodingMode};
//!
//! // RGBA image data (4 bytes per pixel)
//! let rgba_data = vec![255u8; 100 * 100 * 4];
//!
//! // Lossless encoding
//! let encoder = WebPEncoder::new().mode(EncodingMode::Lossless);
//! let frame = encoder.encode_rgba(&rgba_data, 100, 100).unwrap();
//!
//! // Write to file
//! let mut output = Vec::new();
//! encoder.write_to_riff(&frame, &mut output).unwrap();
//! ```

pub mod error;
pub mod riff;
pub mod vp8;
pub mod vp8l;
pub mod alpha;
pub mod animation;
pub mod metadata;
pub mod encoder;
mod bitreader;
mod transform;

use std::io::{Read, Seek};

pub use error::{WebPError, Result};
pub use riff::{RiffContainer, WebPChunk, ChunkType};
pub use vp8::Vp8Decoder;
pub use vp8l::Vp8lDecoder;
pub use alpha::AlphaDecoder;
pub use animation::{AnimationDecoder, AnimationFrame, AnimationInfo};
pub use metadata::{Metadata, ExifData, XmpData};
pub use encoder::{
    WebPEncoder, WebPEncoderConfig, EncodingMode, EncodedFrame,
    encode_webp_lossless, encode_webp_lossy,
};

use image::DynamicImage;

/// Main WebP decoder
pub struct WebPDecoder<R: Read + Seek> {
    reader: R,
    container: Option<RiffContainer>,
}

impl<R: Read + Seek> WebPDecoder<R> {
    /// Create a new WebP decoder from a reader
    pub fn new(reader: R) -> Result<Self> {
        Ok(Self {
            reader,
            container: None,
        })
    }

    /// Parse the WebP container without decoding image data
    pub fn parse(&mut self) -> Result<&RiffContainer> {
        if self.container.is_none() {
            let container = riff::parse_riff(&mut self.reader)?;
            self.container = Some(container);
        }
        Ok(self.container.as_ref().unwrap())
    }

    /// Decode the WebP image to an RGBA image
    pub fn decode(mut self) -> Result<DynamicImage> {
        self.parse()?;
        let container = self.container.take().unwrap();

        decode_container(container, &mut self.reader)
    }

    /// Decode an animated WebP, returning all frames
    pub fn decode_animation(mut self) -> Result<Vec<AnimationFrame>> {
        self.parse()?;
        let container = self.container.take().unwrap();

        animation::decode_animation(container, &mut self.reader)
    }

    /// Get animation info without decoding frames
    pub fn animation_info(&mut self) -> Result<Option<AnimationInfo>> {
        self.parse()?;
        let container = self.container.as_ref().unwrap();

        Ok(animation::get_animation_info(container))
    }

    /// Extract metadata from the WebP file
    pub fn metadata(&mut self) -> Result<Metadata> {
        self.parse()?;
        let container = self.container.as_ref().unwrap();

        metadata::extract_metadata(container, &mut self.reader)
    }

    /// Check if the image is animated
    pub fn is_animated(&mut self) -> Result<bool> {
        self.parse()?;
        let container = self.container.as_ref().unwrap();

        Ok(container.is_animated())
    }

    /// Get image dimensions
    pub fn dimensions(&mut self) -> Result<(u32, u32)> {
        self.parse()?;
        let container = self.container.as_ref().unwrap();

        container.dimensions()
    }
}

/// Decode a parsed RIFF container to an image
fn decode_container<R: Read + Seek>(container: RiffContainer, reader: &mut R) -> Result<DynamicImage> {
    // Check for VP8X (extended format)
    if let Some(vp8x) = container.find_chunk(ChunkType::VP8X) {
        return decode_extended(&container, reader, vp8x);
    }

    // Check for simple VP8L (lossless)
    if let Some(vp8l) = container.find_chunk(ChunkType::VP8L) {
        let decoder = Vp8lDecoder::new(&vp8l.data)?;
        let image = decoder.decode()?;
        return Ok(DynamicImage::ImageRgba8(image));
    }

    // Check for simple VP8 (lossy)
    if let Some(vp8) = container.find_chunk(ChunkType::VP8) {
        let decoder = Vp8Decoder::new(&vp8.data)?;
        let image = decoder.decode()?;
        return Ok(DynamicImage::ImageRgba8(image));
    }

    Err(WebPError::InvalidFormat("No valid image data found".into()))
}

/// Decode extended WebP format (VP8X)
fn decode_extended<R: Read + Seek>(
    container: &RiffContainer,
    _reader: &mut R,
    _vp8x: &WebPChunk,
) -> Result<DynamicImage> {
    // Get alpha data if present
    let alpha_data = container.find_chunk(ChunkType::ALPH).map(|c| &c.data[..]);

    // Decode the image data
    let mut image = if let Some(vp8l) = container.find_chunk(ChunkType::VP8L) {
        let decoder = Vp8lDecoder::new(&vp8l.data)?;
        decoder.decode()?
    } else if let Some(vp8) = container.find_chunk(ChunkType::VP8) {
        let decoder = Vp8Decoder::new(&vp8.data)?;
        decoder.decode()?
    } else {
        return Err(WebPError::InvalidFormat("No image data in VP8X container".into()));
    };

    // Apply alpha channel if present
    if let Some(alpha) = alpha_data {
        let alpha_decoder = AlphaDecoder::new(alpha)?;
        alpha_decoder.apply_alpha(&mut image)?;
    }

    Ok(DynamicImage::ImageRgba8(image))
}

/// Decode WebP from a byte slice
pub fn decode_webp(data: &[u8]) -> Result<DynamicImage> {
    let cursor = std::io::Cursor::new(data);
    let decoder = WebPDecoder::new(cursor)?;
    decoder.decode()
}

/// Decode WebP from a file path
pub fn decode_webp_file(path: &str) -> Result<DynamicImage> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let decoder = WebPDecoder::new(reader)?;
    decoder.decode()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let data: &[u8] = &[];
        let cursor = std::io::Cursor::new(data);
        let decoder = WebPDecoder::new(cursor);
        assert!(decoder.is_ok());
    }
}
