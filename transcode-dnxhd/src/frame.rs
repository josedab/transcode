//! DNxHD frame structures

use crate::error::{DnxError, Result};
use crate::profile::DnxProfile;
use crate::types::{BitDepth, ChromaFormat, Colorimetry};

/// DNxHD frame signature
pub const DNX_SIGNATURE: [u8; 4] = [0x00, 0x00, 0x02, 0x80];

/// Frame header information
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Profile used
    pub profile: DnxProfile,
    /// Bit depth
    pub bit_depth: BitDepth,
    /// Chroma format
    pub chroma_format: ChromaFormat,
    /// Colorimetry information
    pub colorimetry: Colorimetry,
    /// Frame size in bytes (including header)
    pub frame_size: u32,
    /// Number of macroblocks horizontally
    pub mb_width: u32,
    /// Number of macroblocks vertically
    pub mb_height: u32,
    /// Interlaced flag
    pub interlaced: bool,
    /// Progressive scan within frame
    pub progressive: bool,
    /// Alpha channel present
    pub has_alpha: bool,
}

impl FrameHeader {
    /// Parse frame header from bytes
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < 640 {
            return Err(DnxError::InsufficientData {
                needed: 640,
                available: data.len(),
            });
        }

        // Check signature
        let sig: [u8; 4] = [data[0], data[1], data[2], data[3]];
        if sig != DNX_SIGNATURE {
            return Err(DnxError::InvalidSignature(sig));
        }

        // Frame size (bytes 4-7, big endian)
        let frame_size = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);

        // Header version and size at offset 8-9
        let header_size = u16::from_be_bytes([data[8], data[9]]);
        if header_size < 640 {
            return Err(DnxError::InvalidHeader(format!(
                "Header size too small: {}",
                header_size
            )));
        }

        // Width and height at offset 20-23
        let width = u16::from_be_bytes([data[20], data[21]]) as u32;
        let height = u16::from_be_bytes([data[22], data[23]]) as u32;

        // Profile ID at offset 40-43
        let profile_id = u32::from_be_bytes([data[40], data[41], data[42], data[43]]);
        let profile = DnxProfile::from_id(profile_id)
            .ok_or(DnxError::UnknownProfile(profile_id))?;

        let profile_info = profile.info();

        // Parse flags byte at offset 24
        let flags = data[24];
        let interlaced = (flags & 0x02) != 0;
        let progressive = (flags & 0x04) != 0;

        // Alpha flag in profile or header
        let has_alpha = profile.is_444();

        // Calculate macroblock dimensions (16x16 for luma)
        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);

        Ok(FrameHeader {
            width,
            height,
            profile,
            bit_depth: profile_info.bit_depth,
            chroma_format: profile_info.chroma_format,
            colorimetry: Colorimetry::from_byte(data[25]),
            frame_size,
            mb_width,
            mb_height,
            interlaced,
            progressive,
            has_alpha,
        })
    }

    /// Get the number of slices in this frame
    pub fn slice_count(&self) -> u32 {
        // DNxHD uses 8 slices per row for HD content
        self.mb_height * 8
    }

    /// Get luma plane size in samples
    pub fn luma_size(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Get chroma plane size in samples
    pub fn chroma_size(&self) -> usize {
        let h_shift = self.chroma_format.chroma_h_shift();
        let chroma_width = (self.width + (1 << h_shift) - 1) >> h_shift;
        (chroma_width * self.height) as usize
    }
}

/// Decoded DNxHD frame
#[derive(Debug, Clone)]
pub struct DnxFrame {
    /// Frame header
    pub header: FrameHeader,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Profile
    pub profile: DnxProfile,
    /// Y (luma) plane
    pub y_plane: Vec<i16>,
    /// Cb (chroma blue) plane
    pub cb_plane: Vec<i16>,
    /// Cr (chroma red) plane
    pub cr_plane: Vec<i16>,
    /// Alpha plane (optional, for 4:4:4)
    pub alpha_plane: Option<Vec<u16>>,
}

impl DnxFrame {
    /// Create a new frame with the given dimensions
    pub fn new(header: FrameHeader) -> Self {
        let luma_size = header.luma_size();
        let chroma_size = header.chroma_size();

        let alpha_plane = if header.has_alpha {
            Some(vec![0u16; luma_size])
        } else {
            None
        };

        DnxFrame {
            width: header.width,
            height: header.height,
            profile: header.profile,
            y_plane: vec![0i16; luma_size],
            cb_plane: vec![0i16; chroma_size],
            cr_plane: vec![0i16; chroma_size],
            alpha_plane,
            header,
        }
    }

    /// Create frame from existing plane data
    pub fn from_planes(
        y_plane: &[i16],
        cb_plane: &[i16],
        cr_plane: &[i16],
        alpha_plane: Option<&[u16]>,
        width: u32,
        height: u32,
        profile: DnxProfile,
    ) -> Result<Self> {
        let profile_info = profile.info();
        let chroma_format = profile_info.chroma_format;
        let h_shift = chroma_format.chroma_h_shift();

        let expected_luma = (width * height) as usize;
        let chroma_width = (width + (1 << h_shift) - 1) >> h_shift;
        let expected_chroma = (chroma_width * height) as usize;

        if y_plane.len() < expected_luma {
            return Err(DnxError::InsufficientData {
                needed: expected_luma,
                available: y_plane.len(),
            });
        }

        if cb_plane.len() < expected_chroma || cr_plane.len() < expected_chroma {
            return Err(DnxError::InsufficientData {
                needed: expected_chroma,
                available: cb_plane.len().min(cr_plane.len()),
            });
        }

        let header = FrameHeader {
            width,
            height,
            profile,
            bit_depth: profile_info.bit_depth,
            chroma_format,
            colorimetry: Colorimetry::BT709,
            frame_size: 0, // Will be calculated during encoding
            mb_width: width.div_ceil(16),
            mb_height: height.div_ceil(16),
            interlaced: false,
            progressive: true,
            has_alpha: alpha_plane.is_some(),
        };

        Ok(DnxFrame {
            header,
            width,
            height,
            profile,
            y_plane: y_plane[..expected_luma].to_vec(),
            cb_plane: cb_plane[..expected_chroma].to_vec(),
            cr_plane: cr_plane[..expected_chroma].to_vec(),
            alpha_plane: alpha_plane.map(|a| a[..expected_luma].to_vec()),
        })
    }

    /// Get a reference to the Y plane
    pub fn y(&self) -> &[i16] {
        &self.y_plane
    }

    /// Get a reference to the Cb plane
    pub fn cb(&self) -> &[i16] {
        &self.cb_plane
    }

    /// Get a reference to the Cr plane
    pub fn cr(&self) -> &[i16] {
        &self.cr_plane
    }

    /// Get a reference to the alpha plane
    pub fn alpha(&self) -> Option<&[u16]> {
        self.alpha_plane.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_header_insufficient_data() {
        let data = vec![0u8; 100];
        let result = FrameHeader::parse(&data);
        assert!(matches!(result, Err(DnxError::InsufficientData { .. })));
    }

    #[test]
    fn test_frame_header_invalid_signature() {
        let mut data = vec![0u8; 640];
        data[0] = 0xFF; // Invalid signature
        let result = FrameHeader::parse(&data);
        assert!(matches!(result, Err(DnxError::InvalidSignature(_))));
    }

    #[test]
    fn test_dnx_frame_new() {
        let header = FrameHeader {
            width: 1920,
            height: 1080,
            profile: DnxProfile::Dnxhd145,
            bit_depth: BitDepth::Bit8,
            chroma_format: ChromaFormat::YUV422,
            colorimetry: Colorimetry::BT709,
            frame_size: 0,
            mb_width: 120,
            mb_height: 68,
            interlaced: false,
            progressive: true,
            has_alpha: false,
        };

        let frame = DnxFrame::new(header);
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert_eq!(frame.y_plane.len(), 1920 * 1080);
        // 4:2:2 means half width for chroma
        assert_eq!(frame.cb_plane.len(), 960 * 1080);
        assert!(frame.alpha_plane.is_none());
    }

    #[test]
    fn test_dnx_frame_from_planes() {
        let width = 64u32;
        let height = 64u32;
        let y_plane = vec![128i16; (width * height) as usize];
        let cb_plane = vec![0i16; (width / 2 * height) as usize];
        let cr_plane = vec![0i16; (width / 2 * height) as usize];

        let frame = DnxFrame::from_planes(
            &y_plane,
            &cb_plane,
            &cr_plane,
            None,
            width,
            height,
            DnxProfile::Dnxhd145,
        )
        .unwrap();

        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        assert_eq!(frame.y_plane.len(), 4096);
    }
}
