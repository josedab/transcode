//! Main ProRes decoder implementation

use crate::error::{ProResError, Result};
use crate::frame::{FrameHeader, ProResFrame};
use crate::slice::SliceDecoder;
use crate::types::{BitDepth, ChromaFormat, ProResProfile};

/// ProRes decoder configuration
#[derive(Debug, Clone, Default)]
pub struct DecoderConfig {
    /// Enable multi-threaded decoding
    pub multithreaded: bool,
    /// Number of threads (0 = auto-detect)
    pub num_threads: usize,
    /// Skip alpha channel decoding even if present
    pub skip_alpha: bool,
}

/// ProRes video decoder
///
/// Supports all ProRes profiles:
/// - ProRes 422 Proxy, LT, Standard, HQ
/// - ProRes 4444, 4444 XQ (with alpha support)
///
/// # Example
///
/// ```no_run
/// use transcode_prores::ProResDecoder;
///
/// let mut decoder = ProResDecoder::new();
/// let frame_data = std::fs::read("frame.prores").unwrap();
/// let frame = decoder.decode_frame(&frame_data).unwrap();
/// ```
pub struct ProResDecoder {
    /// Decoder configuration
    config: DecoderConfig,
    /// Slice decoder
    slice_decoder: SliceDecoder,
    /// Frame counter
    frame_count: u64,
}

impl Default for ProResDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProResDecoder {
    /// Create a new ProRes decoder with default configuration
    pub fn new() -> Self {
        ProResDecoder {
            config: DecoderConfig::default(),
            slice_decoder: SliceDecoder::new(),
            frame_count: 0,
        }
    }

    /// Create a new ProRes decoder with custom configuration
    pub fn with_config(config: DecoderConfig) -> Self {
        ProResDecoder {
            config,
            slice_decoder: SliceDecoder::new(),
            frame_count: 0,
        }
    }

    /// Decode a ProRes frame
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<ProResFrame> {
        // Parse frame header
        let header = FrameHeader::parse(data)?;

        // Validate frame
        self.validate_frame(&header)?;

        // Create output frame
        let mut frame = ProResFrame::new(&header);

        // Decode slices
        self.decode_slices(data, &header, &mut frame)?;

        self.frame_count += 1;

        Ok(frame)
    }

    /// Decode only the frame header without full decode
    pub fn decode_header(&self, data: &[u8]) -> Result<FrameHeader> {
        FrameHeader::parse(data)
    }

    /// Get the number of frames decoded
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.slice_decoder = SliceDecoder::new();
        self.frame_count = 0;
    }

    /// Get decoder configuration
    pub fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// Validate frame header
    fn validate_frame(&self, header: &FrameHeader) -> Result<()> {
        // Check dimensions
        if header.width == 0 || header.height == 0 {
            return Err(ProResError::InvalidHeader("Zero frame dimensions".into()));
        }

        if header.width > 8192 || header.height > 8192 {
            return Err(ProResError::InvalidHeader(format!(
                "Frame dimensions too large: {}x{}",
                header.width, header.height
            )));
        }

        // Validate profile/chroma format consistency
        match header.profile {
            ProResProfile::P4444 | ProResProfile::P4444XQ => {
                if header.chroma_format != ChromaFormat::YUV444 {
                    return Err(ProResError::InvalidHeader(
                        "4444 profile requires 4:4:4 chroma".into(),
                    ));
                }
            }
            _ => {
                if header.chroma_format != ChromaFormat::YUV422 {
                    return Err(ProResError::InvalidHeader(
                        "422 profile requires 4:2:2 chroma".into(),
                    ));
                }
            }
        }

        // Validate bit depth
        if header.profile == ProResProfile::P4444XQ && header.bit_depth != BitDepth::Bit12 {
            log::warn!("4444 XQ profile typically uses 12-bit");
        }

        Ok(())
    }

    /// Decode all slices in the frame
    fn decode_slices(&mut self, data: &[u8], header: &FrameHeader, frame: &mut ProResFrame) -> Result<()> {
        let (mb_width, _mb_height) = header.mb_dimensions();
        let num_slices = header.num_slices();

        if num_slices == 0 {
            return Err(ProResError::InvalidHeader("No slices in frame".into()));
        }

        // Calculate slice layout
        let slices_per_row = header.slices_per_row as u32;
        let slice_mb_width = mb_width.div_ceil(slices_per_row);

        // Reset DC prediction for new frame
        self.slice_decoder.reset_dc_prediction();

        // Decode each slice
        for slice_idx in 0..num_slices {
            let slice_row = slice_idx as u32 / slices_per_row;
            let slice_col = slice_idx as u32 % slices_per_row;

            // Get slice data boundaries
            let slice_start = header.slice_offsets[slice_idx] as usize;
            let slice_end = if slice_idx + 1 < header.slice_offsets.len() {
                header.slice_offsets[slice_idx + 1] as usize
            } else {
                data.len()
            };

            if slice_start >= data.len() || slice_end > data.len() {
                return Err(ProResError::SliceSizeMismatch {
                    expected: slice_end - slice_start,
                    actual: data.len().saturating_sub(slice_start),
                });
            }

            let slice_data = &data[slice_start..slice_end];

            // Calculate macroblock position
            let slice_x = slice_col * slice_mb_width;
            let slice_y = slice_row;

            // Calculate actual width of this slice (may be less at right edge)
            let actual_slice_mb_width = if slice_col == slices_per_row - 1 {
                mb_width - slice_x
            } else {
                slice_mb_width
            };

            // Get mutable references to planes
            let has_alpha = header.has_alpha() && !self.config.skip_alpha;

            // Decode the slice
            if has_alpha && frame.alpha_plane.is_some() {
                let alpha = frame.alpha_plane.as_mut().unwrap();
                self.slice_decoder.decode_slice(
                    slice_data,
                    header,
                    slice_x,
                    slice_y,
                    actual_slice_mb_width,
                    &mut frame.y_plane,
                    &mut frame.cb_plane,
                    &mut frame.cr_plane,
                    Some(alpha),
                    frame.y_stride,
                    frame.chroma_stride,
                )?;
            } else {
                self.slice_decoder.decode_slice(
                    slice_data,
                    header,
                    slice_x,
                    slice_y,
                    actual_slice_mb_width,
                    &mut frame.y_plane,
                    &mut frame.cb_plane,
                    &mut frame.cr_plane,
                    None,
                    frame.y_stride,
                    frame.chroma_stride,
                )?;
            }
        }

        Ok(())
    }
}

/// Probe data to check if it's a valid ProRes frame
pub fn probe_prores(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }

    // Check for frame signature
    &data[4..8] == b"icpf"
}

/// Get profile from FourCC without full decode
pub fn get_profile(data: &[u8]) -> Option<ProResProfile> {
    if data.len() < 20 {
        return None;
    }

    // FourCC is at offset 12-16
    ProResProfile::from_fourcc(&data[12..16])
}

/// Get frame dimensions without full decode
pub fn get_dimensions(data: &[u8]) -> Option<(u16, u16)> {
    if data.len() < 20 {
        return None;
    }

    let width = u16::from_be_bytes([data[16], data[17]]);
    let height = u16::from_be_bytes([data[18], data[19]]);

    if width > 0 && height > 0 {
        Some((width, height))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = ProResDecoder::new();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_decoder_with_config() {
        let config = DecoderConfig {
            multithreaded: true,
            num_threads: 4,
            skip_alpha: true,
        };
        let decoder = ProResDecoder::with_config(config);
        assert!(decoder.config().multithreaded);
        assert_eq!(decoder.config().num_threads, 4);
        assert!(decoder.config().skip_alpha);
    }

    #[test]
    fn test_probe_prores() {
        // Valid ProRes header start
        let valid_data = [
            0x00, 0x00, 0x10, 0x00, // frame size
            b'i', b'c', b'p', b'f', // signature
        ];
        assert!(probe_prores(&valid_data));

        // Invalid data
        let invalid_data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(!probe_prores(&invalid_data));

        // Too short
        let short_data = [b'i', b'c', b'p', b'f'];
        assert!(!probe_prores(&short_data));
    }

    #[test]
    fn test_get_profile() {
        let mut data = vec![0u8; 20];
        // Set signature
        data[4..8].copy_from_slice(b"icpf");
        // Set FourCC for HQ
        data[12..16].copy_from_slice(b"apch");

        assert_eq!(get_profile(&data), Some(ProResProfile::HQ));

        // Test 4444
        data[12..16].copy_from_slice(b"ap4h");
        assert_eq!(get_profile(&data), Some(ProResProfile::P4444));

        // Test 4444 XQ
        data[12..16].copy_from_slice(b"ap4x");
        assert_eq!(get_profile(&data), Some(ProResProfile::P4444XQ));
    }

    #[test]
    fn test_get_dimensions() {
        let mut data = vec![0u8; 20];
        // Set dimensions (1920x1080)
        data[16] = 0x07;
        data[17] = 0x80; // 1920
        data[18] = 0x04;
        data[19] = 0x38; // 1080

        let dims = get_dimensions(&data);
        assert_eq!(dims, Some((1920, 1080)));
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = ProResDecoder::new();
        // Simulate some state change
        decoder.frame_count = 10;
        decoder.reset();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_validate_frame() {
        let decoder = ProResDecoder::new();

        // Create a valid header
        let header = FrameHeader {
            frame_size: 1000,
            profile: ProResProfile::HQ,
            width: 1920,
            height: 1080,
            chroma_format: ChromaFormat::YUV422,
            interlace_mode: crate::types::InterlaceMode::Progressive,
            bit_depth: BitDepth::Bit10,
            color_primaries: crate::types::ColorPrimaries::BT709,
            transfer_characteristic: crate::types::TransferCharacteristic::BT709,
            matrix_coefficients: crate::types::MatrixCoefficients::BT709,
            alpha_info: 0,
            luma_quant_matrix: [16; 64],
            chroma_quant_matrix: [16; 64],
            slices_per_row: 8,
            slice_rows: 68,
            slice_offsets: vec![0, 100],
            header_size: 148,
        };

        assert!(decoder.validate_frame(&header).is_ok());

        // Test zero dimensions
        let mut bad_header = header.clone();
        bad_header.width = 0;
        assert!(decoder.validate_frame(&bad_header).is_err());

        // Test oversized dimensions
        let mut bad_header = header.clone();
        bad_header.width = 10000;
        assert!(decoder.validate_frame(&bad_header).is_err());

        // Test 4444 with wrong chroma format
        let mut bad_header = header.clone();
        bad_header.profile = ProResProfile::P4444;
        bad_header.chroma_format = ChromaFormat::YUV422;
        assert!(decoder.validate_frame(&bad_header).is_err());
    }

    #[test]
    fn test_all_profiles() {
        // Verify all profiles have unique FourCC codes
        let profiles = [
            ProResProfile::Proxy,
            ProResProfile::LT,
            ProResProfile::Standard,
            ProResProfile::HQ,
            ProResProfile::P4444,
            ProResProfile::P4444XQ,
        ];

        let fourccs: Vec<_> = profiles.iter().map(|p| p.fourcc()).collect();

        // Check uniqueness
        for i in 0..fourccs.len() {
            for j in (i + 1)..fourccs.len() {
                assert_ne!(fourccs[i], fourccs[j], "FourCC collision detected");
            }
        }

        // Check alpha support
        assert!(!ProResProfile::Proxy.supports_alpha());
        assert!(!ProResProfile::LT.supports_alpha());
        assert!(!ProResProfile::Standard.supports_alpha());
        assert!(!ProResProfile::HQ.supports_alpha());
        assert!(ProResProfile::P4444.supports_alpha());
        assert!(ProResProfile::P4444XQ.supports_alpha());

        // Check 444 detection
        assert!(!ProResProfile::HQ.is_444());
        assert!(ProResProfile::P4444.is_444());
        assert!(ProResProfile::P4444XQ.is_444());
    }

    #[test]
    fn test_bit_depth() {
        assert_eq!(BitDepth::Bit10.bits(), 10);
        assert_eq!(BitDepth::Bit12.bits(), 12);
        assert_eq!(BitDepth::Bit10.max_value(), 1023);
        assert_eq!(BitDepth::Bit12.max_value(), 4095);
    }
}
