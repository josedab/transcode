//! CineForm frame structures

use crate::error::{CineformError, Result};
use crate::types::{tags, BitDepth, PixelFormat, Quality};

/// CineForm file signature
pub const CFHD_SIGNATURE: [u8; 4] = [0x43, 0x46, 0x48, 0x44]; // "CFHD"

/// Frame header information
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Pixel format
    pub pixel_format: PixelFormat,
    /// Bit depth
    pub bit_depth: BitDepth,
    /// Quality level
    pub quality: Quality,
    /// Number of transform levels
    pub transform_levels: u8,
    /// Number of channels
    pub channel_count: u8,
    /// Frame index in sequence
    pub frame_index: u32,
}

impl FrameHeader {
    /// Parse frame header from bytes
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 24 {
            return Err(CineformError::InsufficientData {
                needed: 24,
                available: data.len(),
            });
        }

        // Check signature
        let sig: [u8; 4] = [data[0], data[1], data[2], data[3]];
        if sig != CFHD_SIGNATURE {
            return Err(CineformError::InvalidSignature(sig));
        }

        // Parse tag-value pairs
        let mut offset = 4;
        let mut width = 0u32;
        let mut height = 0u32;
        let mut pixel_format = PixelFormat::YUV422;
        let mut bit_depth = BitDepth::Bit10;
        let mut quality = Quality::High;
        let mut transform_levels = 3u8;
        let mut channel_count = 3u8;
        let mut frame_index = 0u32;

        while offset + 4 <= data.len() {
            let tag = u16::from_be_bytes([data[offset], data[offset + 1]]);
            let value = u16::from_be_bytes([data[offset + 2], data[offset + 3]]);
            offset += 4;

            match tag {
                tags::IMAGE_WIDTH => width = value as u32,
                tags::IMAGE_HEIGHT => height = value as u32,
                tags::PIXEL_FORMAT => {
                    pixel_format = match value {
                        0 => PixelFormat::YUV422,
                        1 => PixelFormat::YUV444,
                        2 => PixelFormat::RGBA,
                        3 => PixelFormat::BGRA,
                        4 => PixelFormat::Bayer,
                        _ => PixelFormat::YUV422,
                    };
                }
                tags::BITS_PER_COMPONENT => {
                    bit_depth = BitDepth::from_bits(value as u8).unwrap_or(BitDepth::Bit10);
                }
                tags::QUALITY_LEVEL => {
                    quality = Quality::from_level(value as u8);
                }
                tags::TRANSFORM_LEVELS => transform_levels = value as u8,
                tags::CHANNEL_COUNT => channel_count = value as u8,
                tags::FRAME_INDEX => frame_index = value as u32,
                tags::FRAME_HEADER => {
                    // Start of frame data, stop parsing header
                    break;
                }
                _ => {
                    // Skip unknown tags
                }
            }
        }

        if width == 0 || height == 0 {
            return Err(CineformError::InvalidHeader(
                "Missing width or height".into(),
            ));
        }

        Ok(FrameHeader {
            width,
            height,
            pixel_format,
            bit_depth,
            quality,
            transform_levels,
            channel_count,
            frame_index,
        })
    }

    /// Get the number of wavelet subbands
    pub fn subband_count(&self) -> u32 {
        // Each level produces 3 subbands (LH, HL, HH) plus final LL
        (self.transform_levels as u32 * 3) + 1
    }
}

/// Decoded CineForm frame
#[derive(Debug, Clone)]
pub struct CineformFrame {
    /// Frame header
    pub header: FrameHeader,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Pixel format
    pub pixel_format: PixelFormat,
    /// Quality level
    pub quality: Quality,
    /// Channel data (Y/Cb/Cr or R/G/B/A)
    pub channels: Vec<Vec<i16>>,
}

impl CineformFrame {
    /// Create a new frame with the given header
    pub fn new(header: FrameHeader) -> Self {
        let channel_size = (header.width * header.height) as usize;
        let channel_count = header.channel_count as usize;

        let mut channels = Vec::with_capacity(channel_count);
        for i in 0..channel_count {
            // For YUV422, chroma channels are half width
            let size = if header.pixel_format == PixelFormat::YUV422 && i > 0 {
                channel_size / 2
            } else {
                channel_size
            };
            channels.push(vec![0i16; size]);
        }

        CineformFrame {
            width: header.width,
            height: header.height,
            pixel_format: header.pixel_format,
            quality: header.quality,
            header,
            channels,
        }
    }

    /// Create frame from plane data
    pub fn from_planes(
        planes: &[&[i16]],
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        quality: Quality,
    ) -> Result<Self> {
        let expected_channels = pixel_format.components() as usize;
        if planes.len() < expected_channels {
            return Err(CineformError::InvalidHeader(format!(
                "Expected {} channels, got {}",
                expected_channels,
                planes.len()
            )));
        }

        let header = FrameHeader {
            width,
            height,
            pixel_format,
            bit_depth: BitDepth::Bit10,
            quality,
            transform_levels: 3,
            channel_count: expected_channels as u8,
            frame_index: 0,
        };

        let mut frame = CineformFrame::new(header);
        for (i, plane) in planes.iter().enumerate().take(expected_channels) {
            let target_size = frame.channels[i].len();
            let copy_size = plane.len().min(target_size);
            frame.channels[i][..copy_size].copy_from_slice(&plane[..copy_size]);
        }

        Ok(frame)
    }

    /// Get channel by index
    pub fn channel(&self, index: usize) -> Option<&[i16]> {
        self.channels.get(index).map(|c| c.as_slice())
    }

    /// Get mutable channel by index
    pub fn channel_mut(&mut self, index: usize) -> Option<&mut [i16]> {
        self.channels.get_mut(index).map(|c| c.as_mut_slice())
    }

    /// Get Y (luma) channel for YUV formats
    pub fn y(&self) -> Option<&[i16]> {
        match self.pixel_format {
            PixelFormat::YUV422 | PixelFormat::YUV444 => self.channel(0),
            _ => None,
        }
    }

    /// Get Cb (chroma blue) channel for YUV formats
    pub fn cb(&self) -> Option<&[i16]> {
        match self.pixel_format {
            PixelFormat::YUV422 | PixelFormat::YUV444 => self.channel(1),
            _ => None,
        }
    }

    /// Get Cr (chroma red) channel for YUV formats
    pub fn cr(&self) -> Option<&[i16]> {
        match self.pixel_format {
            PixelFormat::YUV422 | PixelFormat::YUV444 => self.channel(2),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_header_insufficient_data() {
        let data = vec![0u8; 10];
        let result = FrameHeader::parse(&data);
        assert!(matches!(result, Err(CineformError::InsufficientData { .. })));
    }

    #[test]
    fn test_frame_header_invalid_signature() {
        let mut data = vec![0u8; 100];
        data[0] = 0xFF; // Invalid signature
        let result = FrameHeader::parse(&data);
        assert!(matches!(result, Err(CineformError::InvalidSignature(_))));
    }

    #[test]
    fn test_cineform_frame_new() {
        let header = FrameHeader {
            width: 1920,
            height: 1080,
            pixel_format: PixelFormat::YUV422,
            bit_depth: BitDepth::Bit10,
            quality: Quality::High,
            transform_levels: 3,
            channel_count: 3,
            frame_index: 0,
        };

        let frame = CineformFrame::new(header);
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert_eq!(frame.channels.len(), 3);
        // Y channel is full size
        assert_eq!(frame.channels[0].len(), 1920 * 1080);
        // Cb/Cr channels are half width for YUV422
        assert_eq!(frame.channels[1].len(), 1920 * 1080 / 2);
    }

    #[test]
    fn test_cineform_frame_from_planes() {
        let width = 64u32;
        let height = 64u32;
        let y = vec![128i16; (width * height) as usize];
        let cb = vec![0i16; (width * height / 2) as usize];
        let cr = vec![0i16; (width * height / 2) as usize];

        let frame = CineformFrame::from_planes(
            &[&y, &cb, &cr],
            width,
            height,
            PixelFormat::YUV422,
            Quality::High,
        )
        .unwrap();

        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert!(frame.y().is_some());
    }

    #[test]
    fn test_subband_count() {
        let header = FrameHeader {
            width: 100,
            height: 100,
            pixel_format: PixelFormat::YUV422,
            bit_depth: BitDepth::Bit10,
            quality: Quality::High,
            transform_levels: 3,
            channel_count: 3,
            frame_index: 0,
        };

        // 3 levels * 3 subbands + 1 LL = 10
        assert_eq!(header.subband_count(), 10);
    }
}
