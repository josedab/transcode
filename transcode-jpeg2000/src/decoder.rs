//! JPEG2000 decoder.
//!
//! Provides JPEG2000 image decoding with optional OpenJPEG FFI support.

use crate::parser::{CodestreamInfo, Jpeg2000Parser};
use crate::types::*;
use crate::{Jpeg2000Error, Result};

#[cfg(feature = "ffi-openjpeg")]
use crate::ffi::Jpeg2000FfiDecoder;

use std::fmt;

/// Decoded JPEG2000 image.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Number of components.
    pub num_components: u16,
    /// Bit depth per component.
    pub bit_depth: u8,
    /// Is signed data.
    pub is_signed: bool,
    /// Color space.
    pub color_space: ColorSpace,
    /// Component data (one Vec per component).
    pub components: Vec<Vec<i32>>,
    /// Profile.
    pub profile: Jpeg2000Profile,
}

impl DecodedImage {
    /// Create a new decoded image.
    pub fn new(width: u32, height: u32, num_components: u16, bit_depth: u8) -> Self {
        Self {
            width,
            height,
            num_components,
            bit_depth,
            is_signed: false,
            color_space: ColorSpace::Unknown,
            components: Vec::new(),
            profile: Jpeg2000Profile::Part1,
        }
    }

    /// Get total number of pixels per component.
    pub fn num_pixels(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Check if image is grayscale.
    pub fn is_grayscale(&self) -> bool {
        self.num_components == 1
    }

    /// Check if image has alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.num_components == 2 || self.num_components == 4
    }

    /// Convert to RGB byte array (8-bit).
    pub fn to_rgb_bytes(&self) -> Result<Vec<u8>> {
        if self.num_components < 3 {
            return Err(Jpeg2000Error::UnsupportedFeature(
                "Need at least 3 components for RGB".into(),
            ));
        }

        let num_pixels = self.num_pixels();
        let mut rgb = Vec::with_capacity(num_pixels * 3);

        let max_val = (1 << self.bit_depth) - 1;

        for i in 0..num_pixels {
            for c in 0..3 {
                let val = if c < self.components.len() && i < self.components[c].len() {
                    self.components[c][i]
                } else {
                    0
                };

                // Normalize to 8-bit
                let normalized = if self.bit_depth > 8 {
                    (val >> (self.bit_depth - 8)) as u8
                } else if self.bit_depth < 8 {
                    ((val * 255) / max_val) as u8
                } else {
                    val as u8
                };

                rgb.push(normalized);
            }
        }

        Ok(rgb)
    }

    /// Convert to RGBA byte array (8-bit).
    pub fn to_rgba_bytes(&self) -> Result<Vec<u8>> {
        let num_pixels = self.num_pixels();
        let mut rgba = Vec::with_capacity(num_pixels * 4);

        let max_val = (1 << self.bit_depth) - 1;

        for i in 0..num_pixels {
            for c in 0..4 {
                let val = if c < self.components.len() && i < self.components[c].len() {
                    self.components[c][i]
                } else if c == 3 {
                    max_val // Full opacity if no alpha
                } else {
                    0
                };

                // Normalize to 8-bit
                let normalized = if self.bit_depth > 8 {
                    (val >> (self.bit_depth - 8)) as u8
                } else if self.bit_depth < 8 {
                    ((val * 255) / max_val) as u8
                } else {
                    val as u8
                };

                rgba.push(normalized);
            }
        }

        Ok(rgba)
    }

    /// Convert grayscale to byte array.
    pub fn to_gray_bytes(&self) -> Result<Vec<u8>> {
        if self.components.is_empty() {
            return Err(Jpeg2000Error::DecodingError("No components".into()));
        }

        let max_val = (1 << self.bit_depth) - 1;
        let mut gray = Vec::with_capacity(self.num_pixels());

        for &val in &self.components[0] {
            let normalized = if self.bit_depth > 8 {
                (val >> (self.bit_depth - 8)) as u8
            } else if self.bit_depth < 8 {
                ((val * 255) / max_val) as u8
            } else {
                val as u8
            };
            gray.push(normalized);
        }

        Ok(gray)
    }
}

/// JPEG2000 decoder.
///
/// Decodes JPEG2000 codestreams and JP2 files.
///
/// # Example
///
/// ```rust,ignore
/// use transcode_jpeg2000::{Jpeg2000Decoder, DecodedImage};
///
/// let mut decoder = Jpeg2000Decoder::new()?;
/// let image = decoder.decode(&j2k_data)?;
/// println!("Decoded: {}x{}", image.width, image.height);
/// ```
pub struct Jpeg2000Decoder {
    /// Parser for header extraction.
    parser: Jpeg2000Parser,
    /// Last parsed info.
    info: Option<CodestreamInfo>,
    /// Total images decoded.
    images_decoded: u64,
    /// FFI decoder (when available).
    #[cfg(feature = "ffi-openjpeg")]
    ffi_decoder: Option<Jpeg2000FfiDecoder>,
}

impl fmt::Debug for Jpeg2000Decoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Jpeg2000Decoder");
        s.field("info", &self.info);
        s.field("images_decoded", &self.images_decoded);
        #[cfg(feature = "ffi-openjpeg")]
        s.field("ffi_decoder", &self.ffi_decoder.is_some());
        s.finish_non_exhaustive()
    }
}

impl Jpeg2000Decoder {
    /// Create a new decoder.
    #[cfg(feature = "ffi-openjpeg")]
    pub fn new() -> Result<Self> {
        let ffi_decoder = Jpeg2000FfiDecoder::new()?;
        Ok(Self {
            parser: Jpeg2000Parser::new(),
            info: None,
            images_decoded: 0,
            ffi_decoder: Some(ffi_decoder),
        })
    }

    /// Create a new decoder (without FFI).
    #[cfg(not(feature = "ffi-openjpeg"))]
    pub fn new() -> Result<Self> {
        Ok(Self {
            parser: Jpeg2000Parser::new(),
            info: None,
            images_decoded: 0,
        })
    }

    /// Check if FFI decoding is available.
    #[cfg(feature = "ffi-openjpeg")]
    pub fn is_decoding_available(&self) -> bool {
        self.ffi_decoder.is_some()
    }

    /// Check if FFI decoding is available (without FFI).
    #[cfg(not(feature = "ffi-openjpeg"))]
    pub fn is_decoding_available(&self) -> bool {
        false
    }

    /// Decode a JPEG2000 image using OpenJPEG.
    #[cfg(feature = "ffi-openjpeg")]
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedImage> {
        // Use FFI decoder if available
        if let Some(ref mut ffi) = self.ffi_decoder {
            let image = ffi.decode(data)?;
            self.images_decoded += 1;
            self.info = Some(CodestreamInfo {
                width: image.width,
                height: image.height,
                num_components: image.num_components,
                bit_depth: image.bit_depth,
                is_signed: image.is_signed,
                tile_width: image.width,
                tile_height: image.height,
                num_tiles: 1,
                num_decomposition_levels: 5,
                progression_order: crate::types::ProgressionOrder::Lrcp,
                profile: image.profile,
            });
            return Ok(image);
        }

        // Fallback to parsing only
        self.decode_header_only(data)
    }

    /// Decode a JPEG2000 image (stub without FFI).
    #[cfg(not(feature = "ffi-openjpeg"))]
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedImage> {
        self.decode_header_only(data)
    }

    /// Decode header only (no pixel data).
    fn decode_header_only(&mut self, data: &[u8]) -> Result<DecodedImage> {
        // Without FFI, we can only parse headers
        let info = self.parser.parse_header(data)?;
        self.info = Some(info.clone());

        // Create placeholder image with zero data
        // Full decoding requires OpenJPEG
        let mut image = DecodedImage::new(
            info.width,
            info.height,
            info.num_components,
            info.bit_depth,
        );

        image.is_signed = info.is_signed;
        image.profile = info.profile;

        // Without FFI, we cannot decode the actual image data
        // Return structure with metadata but no pixel data
        for _ in 0..info.num_components {
            image.components.push(Vec::new());
        }

        self.images_decoded += 1;

        Ok(image)
    }

    /// Parse header only (no full decode).
    pub fn parse_header(&mut self, data: &[u8]) -> Result<CodestreamInfo> {
        let info = self.parser.parse_header(data)?;
        self.info = Some(info.clone());
        Ok(info)
    }

    /// Get the last parsed codestream info.
    pub fn info(&self) -> Option<&CodestreamInfo> {
        self.info.as_ref()
    }

    /// Get total images decoded.
    pub fn images_decoded(&self) -> u64 {
        self.images_decoded
    }

    /// Reset the decoder.
    #[cfg(feature = "ffi-openjpeg")]
    pub fn reset(&mut self) {
        self.parser.reset();
        self.info = None;
        if let Some(ref mut ffi) = self.ffi_decoder {
            ffi.reset();
        }
    }

    /// Reset the decoder (without FFI).
    #[cfg(not(feature = "ffi-openjpeg"))]
    pub fn reset(&mut self) {
        self.parser.reset();
        self.info = None;
    }
}

impl Default for Jpeg2000Decoder {
    fn default() -> Self {
        Self::new().expect("Decoder creation should not fail")
    }
}

/// Streaming decoder for incremental data.
#[derive(Debug)]
pub struct StreamingDecoder {
    /// Internal buffer.
    buffer: Vec<u8>,
    /// Current state.
    state: StreamingState,
    /// Decoder instance.
    decoder: Jpeg2000Decoder,
}

/// Streaming decoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingState {
    /// Waiting for header.
    WaitingHeader,
    /// Header parsed, waiting for data.
    WaitingData,
    /// Ready to decode.
    Ready,
    /// Error state.
    Error,
}

impl StreamingDecoder {
    /// Create a new streaming decoder.
    pub fn new() -> Result<Self> {
        Ok(Self {
            buffer: Vec::new(),
            state: StreamingState::WaitingHeader,
            decoder: Jpeg2000Decoder::new()?,
        })
    }

    /// Feed data to the decoder.
    pub fn feed(&mut self, data: &[u8]) -> Result<Option<CodestreamInfo>> {
        if self.state == StreamingState::Error {
            return Err(Jpeg2000Error::DecodingError("Decoder in error state".into()));
        }

        self.buffer.extend_from_slice(data);

        // Try to parse header
        if self.state == StreamingState::WaitingHeader {
            match self.decoder.parse_header(&self.buffer) {
                Ok(info) => {
                    self.state = StreamingState::Ready;
                    return Ok(Some(info));
                }
                Err(Jpeg2000Error::BufferTooSmall { .. }) => {
                    // Need more data
                    return Ok(None);
                }
                Err(e) => {
                    self.state = StreamingState::Error;
                    return Err(e);
                }
            }
        }

        Ok(None)
    }

    /// Finish and decode the image.
    pub fn finish(&mut self) -> Result<DecodedImage> {
        if self.buffer.is_empty() {
            return Err(Jpeg2000Error::DecodingError("No data".into()));
        }

        self.decoder.decode(&self.buffer)
    }

    /// Get current state.
    pub fn state(&self) -> StreamingState {
        self.state
    }

    /// Reset the streaming decoder.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.state = StreamingState::WaitingHeader;
        self.decoder.reset();
    }

    /// Get buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for StreamingDecoder {
    fn default() -> Self {
        Self::new().expect("Streaming decoder creation should not fail")
    }
}

/// Tile decoder for region-of-interest decoding.
#[derive(Debug)]
pub struct TileDecoder {
    /// Base decoder.
    decoder: Jpeg2000Decoder,
    /// Parsed codestream info.
    info: Option<CodestreamInfo>,
}

impl TileDecoder {
    /// Create a new tile decoder.
    pub fn new() -> Result<Self> {
        Ok(Self {
            decoder: Jpeg2000Decoder::new()?,
            info: None,
        })
    }

    /// Parse headers to get tile info.
    pub fn parse(&mut self, data: &[u8]) -> Result<CodestreamInfo> {
        let info = self.decoder.parse_header(data)?;
        self.info = Some(info.clone());
        Ok(info)
    }

    /// Get number of tiles.
    pub fn num_tiles(&self) -> Option<u32> {
        self.info.as_ref().map(|i| i.num_tiles)
    }

    /// Get tile dimensions.
    pub fn tile_size(&self) -> Option<(u32, u32)> {
        self.info.as_ref().map(|i| (i.tile_width, i.tile_height))
    }

    /// Decode a specific tile (requires FFI).
    #[cfg(feature = "ffi-openjpeg")]
    pub fn decode_tile(&mut self, _data: &[u8], _tile_index: u32) -> Result<DecodedImage> {
        Err(Jpeg2000Error::FfiNotAvailable)
    }

    /// Decode a specific tile (stub without FFI).
    #[cfg(not(feature = "ffi-openjpeg"))]
    pub fn decode_tile(&mut self, _data: &[u8], _tile_index: u32) -> Result<DecodedImage> {
        Err(Jpeg2000Error::FfiNotAvailable)
    }

    /// Decode a region (requires FFI).
    #[cfg(feature = "ffi-openjpeg")]
    pub fn decode_region(
        &mut self,
        _data: &[u8],
        _x: u32,
        _y: u32,
        _width: u32,
        _height: u32,
    ) -> Result<DecodedImage> {
        Err(Jpeg2000Error::FfiNotAvailable)
    }

    /// Decode a region (stub without FFI).
    #[cfg(not(feature = "ffi-openjpeg"))]
    pub fn decode_region(
        &mut self,
        _data: &[u8],
        _x: u32,
        _y: u32,
        _width: u32,
        _height: u32,
    ) -> Result<DecodedImage> {
        Err(Jpeg2000Error::FfiNotAvailable)
    }

    /// Reset the tile decoder.
    pub fn reset(&mut self) {
        self.decoder.reset();
        self.info = None;
    }
}

impl Default for TileDecoder {
    fn default() -> Self {
        Self::new().expect("Tile decoder creation should not fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoded_image_new() {
        let image = DecodedImage::new(1920, 1080, 3, 8);
        assert_eq!(image.width, 1920);
        assert_eq!(image.height, 1080);
        assert_eq!(image.num_components, 3);
        assert_eq!(image.bit_depth, 8);
    }

    #[test]
    fn test_decoded_image_num_pixels() {
        let image = DecodedImage::new(100, 100, 3, 8);
        assert_eq!(image.num_pixels(), 10000);
    }

    #[test]
    fn test_decoded_image_is_grayscale() {
        let gray = DecodedImage::new(100, 100, 1, 8);
        let rgb = DecodedImage::new(100, 100, 3, 8);
        assert!(gray.is_grayscale());
        assert!(!rgb.is_grayscale());
    }

    #[test]
    fn test_decoded_image_has_alpha() {
        let rgb = DecodedImage::new(100, 100, 3, 8);
        let rgba = DecodedImage::new(100, 100, 4, 8);
        let gray_alpha = DecodedImage::new(100, 100, 2, 8);
        assert!(!rgb.has_alpha());
        assert!(rgba.has_alpha());
        assert!(gray_alpha.has_alpha());
    }

    #[test]
    fn test_decoder_new() {
        let decoder = Jpeg2000Decoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_is_decoding_available() {
        let decoder = Jpeg2000Decoder::new().unwrap();
        #[cfg(feature = "ffi-openjpeg")]
        assert!(decoder.is_decoding_available());
        #[cfg(not(feature = "ffi-openjpeg"))]
        assert!(!decoder.is_decoding_available());
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = Jpeg2000Decoder::new().unwrap();
        decoder.reset();
        assert!(decoder.info().is_none());
    }

    #[test]
    fn test_streaming_decoder_new() {
        let decoder = StreamingDecoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_streaming_decoder_state() {
        let decoder = StreamingDecoder::new().unwrap();
        assert_eq!(decoder.state(), StreamingState::WaitingHeader);
    }

    #[test]
    fn test_streaming_decoder_reset() {
        let mut decoder = StreamingDecoder::new().unwrap();
        decoder.buffer.extend_from_slice(&[0x00, 0x01, 0x02]);
        decoder.reset();
        assert_eq!(decoder.buffer_size(), 0);
        assert_eq!(decoder.state(), StreamingState::WaitingHeader);
    }

    #[test]
    fn test_tile_decoder_new() {
        let decoder = TileDecoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_tile_decoder_no_info() {
        let decoder = TileDecoder::new().unwrap();
        assert!(decoder.num_tiles().is_none());
        assert!(decoder.tile_size().is_none());
    }

    #[test]
    fn test_to_gray_bytes_empty() {
        let image = DecodedImage::new(10, 10, 1, 8);
        let result = image.to_gray_bytes();
        assert!(result.is_err());
    }

    #[test]
    fn test_to_rgb_bytes_insufficient_components() {
        let image = DecodedImage::new(10, 10, 1, 8);
        let result = image.to_rgb_bytes();
        assert!(result.is_err());
    }

    #[test]
    fn test_to_rgba_bytes() {
        let mut image = DecodedImage::new(2, 2, 3, 8);
        image.components = vec![
            vec![255, 0, 128, 64],
            vec![128, 255, 64, 32],
            vec![64, 128, 255, 16],
        ];
        let rgba = image.to_rgba_bytes().unwrap();
        assert_eq!(rgba.len(), 16); // 4 pixels * 4 components
        // First pixel: R=255, G=128, B=64, A=255
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[1], 128);
        assert_eq!(rgba[2], 64);
        assert_eq!(rgba[3], 255);
    }
}
