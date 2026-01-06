//! GIF image codec.
//!
//! Supports GIF87a and GIF89a formats with animation support.
//!
//! ## Features
//!
//! - LZW decompression/compression
//! - Animation support with frame delays
//! - Transparency support
//! - Interlaced images
//! - Local and global color tables

mod decoder;
mod encoder;

pub use decoder::GifDecoder;
pub use encoder::{GifEncoder, GifConfig};

/// A single frame in a GIF animation.
#[derive(Debug, Clone)]
pub struct GifFrame {
    /// Frame pixel data (indexed or RGBA).
    pub data: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frame X offset in canvas.
    pub x_offset: u16,
    /// Frame Y offset in canvas.
    pub y_offset: u16,
    /// Frame delay in centiseconds (1/100th second).
    pub delay: u16,
    /// Disposal method.
    pub disposal: DisposalMethod,
    /// Transparent color index (if any).
    pub transparent_index: Option<u8>,
    /// Local color table (if any).
    pub local_palette: Option<Vec<[u8; 3]>>,
    /// Is interlaced.
    pub interlaced: bool,
}

impl GifFrame {
    /// Create a new GIF frame.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0; (width * height) as usize],
            width,
            height,
            x_offset: 0,
            y_offset: 0,
            delay: 10, // 100ms default
            disposal: DisposalMethod::None,
            transparent_index: None,
            local_palette: None,
            interlaced: false,
        }
    }

    /// Get frame duration in milliseconds.
    pub fn duration_ms(&self) -> u32 {
        self.delay as u32 * 10
    }
}

/// GIF frame disposal method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisposalMethod {
    /// No disposal specified.
    #[default]
    None,
    /// Do not dispose.
    Keep,
    /// Restore to background color.
    RestoreBackground,
    /// Restore to previous frame.
    RestorePrevious,
}

impl DisposalMethod {
    /// Parse disposal method from byte.
    pub fn from_byte(byte: u8) -> Self {
        match (byte >> 2) & 0x07 {
            0 => DisposalMethod::None,
            1 => DisposalMethod::Keep,
            2 => DisposalMethod::RestoreBackground,
            3 => DisposalMethod::RestorePrevious,
            _ => DisposalMethod::None,
        }
    }

    /// Convert to byte representation.
    pub fn to_byte(self) -> u8 {
        match self {
            DisposalMethod::None => 0,
            DisposalMethod::Keep => 1,
            DisposalMethod::RestoreBackground => 2,
            DisposalMethod::RestorePrevious => 3,
        }
    }
}

/// GIF logical screen descriptor.
#[derive(Debug, Clone)]
pub struct LogicalScreenDescriptor {
    /// Canvas width.
    pub width: u16,
    /// Canvas height.
    pub height: u16,
    /// Has global color table.
    pub has_global_color_table: bool,
    /// Color resolution (bits per channel - 1).
    pub color_resolution: u8,
    /// Global color table is sorted.
    pub sorted: bool,
    /// Size of global color table (2^(n+1) entries).
    pub global_color_table_size: u8,
    /// Background color index.
    pub background_color_index: u8,
    /// Pixel aspect ratio.
    pub pixel_aspect_ratio: u8,
}

impl Default for LogicalScreenDescriptor {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            has_global_color_table: true,
            color_resolution: 7,
            sorted: false,
            global_color_table_size: 7,
            background_color_index: 0,
            pixel_aspect_ratio: 0,
        }
    }
}

/// LZW minimum code size validation.
const MIN_LZW_CODE_SIZE: u8 = 2;
const MAX_LZW_CODE_SIZE: u8 = 12;

/// GIF87a file signature.
pub const GIF87A_SIGNATURE: &[u8; 6] = b"GIF87a";
/// GIF89a file signature.
pub const GIF89A_SIGNATURE: &[u8; 6] = b"GIF89a";

/// Extension introducer byte.
pub const EXTENSION_INTRODUCER: u8 = 0x21;
/// Image separator byte.
pub const IMAGE_SEPARATOR: u8 = 0x2C;
/// File trailer byte.
pub const TRAILER: u8 = 0x3B;

/// Graphic control extension label.
pub const GRAPHIC_CONTROL_LABEL: u8 = 0xF9;
/// Comment extension label.
pub const COMMENT_LABEL: u8 = 0xFE;
/// Application extension label.
pub const APPLICATION_LABEL: u8 = 0xFF;
/// Plain text extension label.
pub const PLAIN_TEXT_LABEL: u8 = 0x01;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disposal_method() {
        assert_eq!(DisposalMethod::from_byte(0x00), DisposalMethod::None);
        assert_eq!(DisposalMethod::from_byte(0x04), DisposalMethod::Keep);
        assert_eq!(DisposalMethod::from_byte(0x08), DisposalMethod::RestoreBackground);
        assert_eq!(DisposalMethod::from_byte(0x0C), DisposalMethod::RestorePrevious);
    }

    #[test]
    fn test_gif_frame() {
        let frame = GifFrame::new(100, 100);
        assert_eq!(frame.width, 100);
        assert_eq!(frame.height, 100);
        assert_eq!(frame.duration_ms(), 100);
    }

    #[test]
    fn test_logical_screen_descriptor_default() {
        let lsd = LogicalScreenDescriptor::default();
        assert!(lsd.has_global_color_table);
        assert_eq!(lsd.color_resolution, 7);
    }
}
