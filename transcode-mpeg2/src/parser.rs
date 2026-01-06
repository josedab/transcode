//! MPEG-2 bitstream parser.
//!
//! Parses MPEG-2 elementary stream headers and extracts video metadata.

use crate::types::*;
use crate::{
    Mpeg2Error, Result, EXTENSION_START_CODE, GOP_START_CODE, PICTURE_START_CODE,
    SEQUENCE_HEADER_CODE, SLICE_START_CODE_MAX, SLICE_START_CODE_MIN,
};

/// MPEG-2 elementary stream parser.
///
/// Parses sequence headers, GOPs, pictures, and slices from MPEG-2 video elementary streams.
///
/// # Example
///
/// ```rust,ignore
/// use transcode_mpeg2::Mpeg2Parser;
///
/// let mut parser = Mpeg2Parser::new();
/// if let Some(seq) = parser.parse_sequence_header(&data) {
///     println!("Resolution: {}x{}", seq.horizontal_size, seq.vertical_size);
/// }
/// ```
#[derive(Debug)]
pub struct Mpeg2Parser {
    /// Current position in the stream.
    position: usize,
    /// Last parsed sequence header.
    sequence_header: Option<SequenceHeader>,
    /// Last parsed sequence extension.
    sequence_extension: Option<SequenceExtension>,
    /// Last parsed GOP header.
    gop_header: Option<GopHeader>,
    /// Total bytes parsed.
    bytes_parsed: u64,
}

impl Mpeg2Parser {
    /// Create a new MPEG-2 parser.
    pub fn new() -> Self {
        Self {
            position: 0,
            sequence_header: None,
            sequence_extension: None,
            gop_header: None,
            bytes_parsed: 0,
        }
    }

    /// Reset the parser state.
    pub fn reset(&mut self) {
        self.position = 0;
        self.sequence_header = None;
        self.sequence_extension = None;
        self.gop_header = None;
    }

    /// Find the next start code in the data.
    ///
    /// Start codes are 0x000001XX patterns.
    pub fn find_start_code(&self, data: &[u8]) -> Option<(usize, u8)> {
        if data.len() < 4 {
            return None;
        }

        for i in 0..data.len() - 3 {
            if data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01 {
                return Some((i, data[i + 3]));
            }
        }

        None
    }

    /// Find sequence header start code.
    pub fn find_sequence_header(&self, data: &[u8]) -> Option<usize> {
        if data.len() < 4 {
            return None;
        }

        (0..data.len() - 3).find(|&i| {
            data[i] == 0x00
                && data[i + 1] == 0x00
                && data[i + 2] == 0x01
                && data[i + 3] == SEQUENCE_HEADER_CODE
        })
    }

    /// Parse a sequence header from the data.
    ///
    /// Returns `Some(SequenceHeader)` if a valid sequence header is found.
    pub fn parse_sequence_header(&mut self, data: &[u8]) -> Option<SequenceHeader> {
        // Need at least 12 bytes for minimal sequence header
        // 4 bytes start code + 8 bytes minimum data
        if data.len() < 12 {
            return None;
        }

        // Check start code
        if data[0] != 0x00 || data[1] != 0x00 || data[2] != 0x01 || data[3] != SEQUENCE_HEADER_CODE
        {
            return None;
        }

        let mut reader = BitReader::new(&data[4..]);

        // horizontal_size_value (12 bits)
        let horizontal_size = reader.read_bits(12)? as u16;

        // vertical_size_value (12 bits)
        let vertical_size = reader.read_bits(12)? as u16;

        // aspect_ratio_information (4 bits)
        let aspect_ratio_code = AspectRatioCode::from_code(reader.read_bits(4)? as u8);

        // frame_rate_code (4 bits)
        let frame_rate_code = FrameRateCode::from_code(reader.read_bits(4)? as u8);

        // bit_rate_value (18 bits)
        let bit_rate = reader.read_bits(18)?;

        // marker_bit (1 bit) - should be 1
        let _ = reader.read_bit()?;

        // vbv_buffer_size_value (10 bits)
        let vbv_buffer_size = reader.read_bits(10)? as u16;

        // constrained_parameters_flag (1 bit)
        let constrained_parameters_flag = reader.read_bit()? == 1;

        // load_intra_quantizer_matrix (1 bit)
        let load_intra_quantizer_matrix = reader.read_bit()? == 1;

        // intra_quantizer_matrix (64 * 8 bits) - if loaded
        let intra_quantizer_matrix = if load_intra_quantizer_matrix {
            let mut matrix = [0u8; 64];
            for item in &mut matrix {
                *item = reader.read_bits(8)? as u8;
            }
            Some(matrix)
        } else {
            None
        };

        // load_non_intra_quantizer_matrix (1 bit)
        let load_non_intra_quantizer_matrix = reader.read_bit()? == 1;

        // non_intra_quantizer_matrix (64 * 8 bits) - if loaded
        let non_intra_quantizer_matrix = if load_non_intra_quantizer_matrix {
            let mut matrix = [0u8; 64];
            for item in &mut matrix {
                *item = reader.read_bits(8)? as u8;
            }
            Some(matrix)
        } else {
            None
        };

        let header = SequenceHeader {
            horizontal_size,
            vertical_size,
            aspect_ratio_code,
            frame_rate_code,
            bit_rate,
            vbv_buffer_size,
            constrained_parameters_flag,
            load_intra_quantizer_matrix,
            intra_quantizer_matrix,
            load_non_intra_quantizer_matrix,
            non_intra_quantizer_matrix,
        };

        self.sequence_header = Some(header.clone());
        self.bytes_parsed += reader.bytes_read() as u64 + 4;

        Some(header)
    }

    /// Parse a sequence extension from the data.
    pub fn parse_sequence_extension(&mut self, data: &[u8]) -> Option<SequenceExtension> {
        // Need at least 10 bytes
        if data.len() < 10 {
            return None;
        }

        // Check start code
        if data[0] != 0x00
            || data[1] != 0x00
            || data[2] != 0x01
            || data[3] != EXTENSION_START_CODE
        {
            return None;
        }

        let mut reader = BitReader::new(&data[4..]);

        // extension_start_code_identifier (4 bits) - should be 1 for sequence extension
        let ext_id = reader.read_bits(4)?;
        if ext_id != 1 {
            return None;
        }

        // profile_and_level_indication (8 bits)
        let profile_and_level = reader.read_bits(8)? as u8;

        // progressive_sequence (1 bit)
        let progressive_sequence = reader.read_bit()? == 1;

        // chroma_format (2 bits)
        let chroma_format_code = reader.read_bits(2)? as u8;
        let chroma_format = ChromaFormat::from_code(chroma_format_code)?;

        // horizontal_size_extension (2 bits)
        let horizontal_size_extension = reader.read_bits(2)? as u8;

        // vertical_size_extension (2 bits)
        let vertical_size_extension = reader.read_bits(2)? as u8;

        // bit_rate_extension (12 bits)
        let bit_rate_extension = reader.read_bits(12)? as u16;

        // marker_bit (1 bit)
        let _ = reader.read_bit()?;

        // vbv_buffer_size_extension (8 bits)
        let vbv_buffer_size_extension = reader.read_bits(8)? as u8;

        // low_delay (1 bit)
        let low_delay = reader.read_bit()? == 1;

        // frame_rate_extension_n (2 bits)
        let frame_rate_extension_n = reader.read_bits(2)? as u8;

        // frame_rate_extension_d (5 bits)
        let frame_rate_extension_d = reader.read_bits(5)? as u8;

        let ext = SequenceExtension {
            profile_and_level,
            progressive_sequence,
            chroma_format,
            horizontal_size_extension,
            vertical_size_extension,
            bit_rate_extension,
            vbv_buffer_size_extension,
            low_delay,
            frame_rate_extension_n,
            frame_rate_extension_d,
        };

        self.sequence_extension = Some(ext.clone());

        Some(ext)
    }

    /// Parse a GOP header from the data.
    pub fn parse_gop_header(&mut self, data: &[u8]) -> Option<GopHeader> {
        // Need at least 8 bytes
        if data.len() < 8 {
            return None;
        }

        // Check start code
        if data[0] != 0x00 || data[1] != 0x00 || data[2] != 0x01 || data[3] != GOP_START_CODE {
            return None;
        }

        let mut reader = BitReader::new(&data[4..]);

        // drop_frame_flag (1 bit)
        let drop_frame_flag = reader.read_bit()? == 1;

        // time_code_hours (5 bits)
        let hours = reader.read_bits(5)? as u8;

        // time_code_minutes (6 bits)
        let minutes = reader.read_bits(6)? as u8;

        // marker_bit (1 bit)
        let _ = reader.read_bit()?;

        // time_code_seconds (6 bits)
        let seconds = reader.read_bits(6)? as u8;

        // time_code_pictures (6 bits)
        let pictures = reader.read_bits(6)? as u8;

        // closed_gop (1 bit)
        let closed_gop = reader.read_bit()? == 1;

        // broken_link (1 bit)
        let broken_link = reader.read_bit()? == 1;

        let gop = GopHeader {
            hours,
            minutes,
            seconds,
            pictures,
            drop_frame_flag,
            closed_gop,
            broken_link,
        };

        self.gop_header = Some(gop.clone());

        Some(gop)
    }

    /// Parse a picture header from the data.
    pub fn parse_picture_header(&self, data: &[u8]) -> Option<PictureHeader> {
        // Need at least 8 bytes
        if data.len() < 8 {
            return None;
        }

        // Check start code
        if data[0] != 0x00 || data[1] != 0x00 || data[2] != 0x01 || data[3] != PICTURE_START_CODE {
            return None;
        }

        let mut reader = BitReader::new(&data[4..]);

        // temporal_reference (10 bits)
        let temporal_reference = reader.read_bits(10)? as u16;

        // picture_coding_type (3 bits)
        let picture_coding_type_code = reader.read_bits(3)? as u8;
        let picture_coding_type = PictureCodingType::from_code(picture_coding_type_code)?;

        // vbv_delay (16 bits)
        let vbv_delay = reader.read_bits(16)? as u16;

        // For P and B frames, there are additional fields
        let (full_pel_forward_vector, forward_f_code) =
            if picture_coding_type == PictureCodingType::P
                || picture_coding_type == PictureCodingType::B
            {
                let full_pel = reader.read_bit()? == 1;
                let f_code = reader.read_bits(3)? as u8;
                (full_pel, f_code)
            } else {
                (false, 0)
            };

        let (full_pel_backward_vector, backward_f_code) =
            if picture_coding_type == PictureCodingType::B {
                let full_pel = reader.read_bit()? == 1;
                let f_code = reader.read_bits(3)? as u8;
                (full_pel, f_code)
            } else {
                (false, 0)
            };

        Some(PictureHeader {
            temporal_reference,
            picture_coding_type,
            vbv_delay,
            full_pel_forward_vector,
            forward_f_code,
            full_pel_backward_vector,
            backward_f_code,
        })
    }

    /// Parse a picture coding extension from the data.
    pub fn parse_picture_coding_extension(&self, data: &[u8]) -> Option<PictureCodingExtension> {
        // Need at least 9 bytes
        if data.len() < 9 {
            return None;
        }

        // Check start code
        if data[0] != 0x00
            || data[1] != 0x00
            || data[2] != 0x01
            || data[3] != EXTENSION_START_CODE
        {
            return None;
        }

        let mut reader = BitReader::new(&data[4..]);

        // extension_start_code_identifier (4 bits) - should be 8 for picture coding extension
        let ext_id = reader.read_bits(4)?;
        if ext_id != 8 {
            return None;
        }

        // f_code[0][0] forward horizontal (4 bits)
        let f_code_00 = reader.read_bits(4)? as u8;

        // f_code[0][1] forward vertical (4 bits)
        let f_code_01 = reader.read_bits(4)? as u8;

        // f_code[1][0] backward horizontal (4 bits)
        let f_code_10 = reader.read_bits(4)? as u8;

        // f_code[1][1] backward vertical (4 bits)
        let f_code_11 = reader.read_bits(4)? as u8;

        // intra_dc_precision (2 bits)
        let intra_dc_precision = reader.read_bits(2)? as u8;

        // picture_structure (2 bits)
        let picture_structure_code = reader.read_bits(2)? as u8;
        let picture_structure = PictureStructure::from_code(picture_structure_code)?;

        // top_field_first (1 bit)
        let top_field_first = reader.read_bit()? == 1;

        // frame_pred_frame_dct (1 bit)
        let frame_pred_frame_dct = reader.read_bit()? == 1;

        // concealment_motion_vectors (1 bit)
        let concealment_motion_vectors = reader.read_bit()? == 1;

        // q_scale_type (1 bit)
        let q_scale_type = reader.read_bit()? == 1;

        // intra_vlc_format (1 bit)
        let intra_vlc_format = reader.read_bit()? == 1;

        // alternate_scan (1 bit)
        let alternate_scan = reader.read_bit()? == 1;

        // repeat_first_field (1 bit)
        let repeat_first_field = reader.read_bit()? == 1;

        // chroma_420_type (1 bit)
        let chroma_420_type = reader.read_bit()? == 1;

        // progressive_frame (1 bit)
        let progressive_frame = reader.read_bit()? == 1;

        // composite_display_flag (1 bit)
        let composite_display_flag = reader.read_bit()? == 1;

        Some(PictureCodingExtension {
            f_code_00,
            f_code_01,
            f_code_10,
            f_code_11,
            intra_dc_precision,
            picture_structure,
            top_field_first,
            frame_pred_frame_dct,
            concealment_motion_vectors,
            q_scale_type,
            intra_vlc_format,
            alternate_scan,
            repeat_first_field,
            chroma_420_type,
            progressive_frame,
            composite_display_flag,
        })
    }

    /// Check if a start code is a slice start code.
    pub fn is_slice_start_code(code: u8) -> bool {
        (SLICE_START_CODE_MIN..=SLICE_START_CODE_MAX).contains(&code)
    }

    /// Get the last parsed sequence header.
    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.sequence_header.as_ref()
    }

    /// Get the last parsed sequence extension.
    pub fn sequence_extension(&self) -> Option<&SequenceExtension> {
        self.sequence_extension.as_ref()
    }

    /// Get the last parsed GOP header.
    pub fn gop_header(&self) -> Option<&GopHeader> {
        self.gop_header.as_ref()
    }

    /// Get total bytes parsed.
    pub fn bytes_parsed(&self) -> u64 {
        self.bytes_parsed
    }

    /// Get full resolution considering extension.
    pub fn full_resolution(&self) -> Option<(u32, u32)> {
        let seq = self.sequence_header.as_ref()?;
        let ext = self.sequence_extension.as_ref();

        let width = if let Some(e) = ext {
            ((e.horizontal_size_extension as u32) << 12) | seq.horizontal_size as u32
        } else {
            seq.horizontal_size as u32
        };

        let height = if let Some(e) = ext {
            ((e.vertical_size_extension as u32) << 12) | seq.vertical_size as u32
        } else {
            seq.vertical_size as u32
        };

        Some((width, height))
    }

    /// Check if the stream is MPEG-2 (has sequence extension).
    pub fn is_mpeg2(&self) -> bool {
        self.sequence_extension.is_some()
    }
}

impl Default for Mpeg2Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple bit reader for MPEG-2 parsing.
#[derive(Debug)]
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// Create a new bit reader.
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit.
    fn read_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.data.len() {
            return None;
        }

        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }

        Some(bit)
    }

    /// Read multiple bits (up to 32).
    fn read_bits(&mut self, count: u8) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | self.read_bit()? as u32;
        }
        Some(value)
    }

    /// Get bytes read so far.
    fn bytes_read(&self) -> usize {
        if self.bit_pos > 0 {
            self.byte_pos + 1
        } else {
            self.byte_pos
        }
    }
}

/// Detect MPEG-2 video in data.
///
/// Returns true if the data starts with a valid MPEG-2 sequence header.
pub fn detect_mpeg2(data: &[u8]) -> bool {
    if data.len() < 12 {
        return false;
    }

    // Check for sequence header start code
    if data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x01 && data[3] == SEQUENCE_HEADER_CODE {
        return true;
    }

    // Search for sequence header within first 1KB
    let search_len = data.len().min(1024);
    for i in 0..search_len.saturating_sub(3) {
        if data[i] == 0x00
            && data[i + 1] == 0x00
            && data[i + 2] == 0x01
            && data[i + 3] == SEQUENCE_HEADER_CODE
        {
            return true;
        }
    }

    false
}

/// Get video info from MPEG-2 data.
pub fn get_video_info(data: &[u8]) -> Result<VideoInfo> {
    let mut parser = Mpeg2Parser::new();

    // Find and parse sequence header
    let seq_pos = parser
        .find_sequence_header(data)
        .ok_or_else(|| Mpeg2Error::InvalidSequenceHeader("No sequence header found".into()))?;

    let seq = parser
        .parse_sequence_header(&data[seq_pos..])
        .ok_or_else(|| Mpeg2Error::InvalidSequenceHeader("Failed to parse sequence header".into()))?;

    // Try to find and parse sequence extension (for MPEG-2)
    let is_mpeg2 = if seq_pos + 12 < data.len() {
        if let Some((pos, code)) = parser.find_start_code(&data[seq_pos + 12..]) {
            if code == EXTENSION_START_CODE {
                parser
                    .parse_sequence_extension(&data[seq_pos + 12 + pos..])
                    .is_some()
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };

    let (width, height) = if let Some((w, h)) = parser.full_resolution() {
        (w, h)
    } else {
        (seq.horizontal_size as u32, seq.vertical_size as u32)
    };

    Ok(VideoInfo {
        width: width as u16,
        height: height as u16,
        frame_rate: seq.frame_rate(),
        aspect_ratio: seq.aspect_ratio_code,
        bit_rate_bps: seq.bit_rate_bps(),
        is_mpeg2,
        progressive: parser
            .sequence_extension()
            .map(|e| e.progressive_sequence)
            .unwrap_or(true),
        chroma_format: parser
            .sequence_extension()
            .map(|e| e.chroma_format)
            .unwrap_or(ChromaFormat::Yuv420),
        profile: parser.sequence_extension().map(|e| e.profile()),
        level: parser.sequence_extension().map(|e| e.level()),
    })
}

/// Video information extracted from MPEG-2 stream.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    /// Video width.
    pub width: u16,
    /// Video height.
    pub height: u16,
    /// Frame rate in fps.
    pub frame_rate: f64,
    /// Aspect ratio.
    pub aspect_ratio: AspectRatioCode,
    /// Bit rate in bits per second.
    pub bit_rate_bps: u64,
    /// True if MPEG-2 (has sequence extension).
    pub is_mpeg2: bool,
    /// Progressive sequence.
    pub progressive: bool,
    /// Chroma format.
    pub chroma_format: ChromaFormat,
    /// Profile (MPEG-2 only).
    pub profile: Option<Profile>,
    /// Level (MPEG-2 only).
    pub level: Option<Level>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_new() {
        let parser = Mpeg2Parser::new();
        assert!(parser.sequence_header().is_none());
        assert!(parser.sequence_extension().is_none());
    }

    #[test]
    fn test_find_start_code() {
        let parser = Mpeg2Parser::new();

        // Valid start code
        let data = [0x00, 0x00, 0x01, 0xB3];
        assert_eq!(parser.find_start_code(&data), Some((0, 0xB3)));

        // Start code not at beginning
        let data = [0xFF, 0x00, 0x00, 0x01, 0xB3];
        assert_eq!(parser.find_start_code(&data), Some((1, 0xB3)));

        // No start code
        let data = [0x00, 0x00, 0x02, 0xB3];
        assert_eq!(parser.find_start_code(&data), None);
    }

    #[test]
    fn test_is_slice_start_code() {
        assert!(Mpeg2Parser::is_slice_start_code(0x01));
        assert!(Mpeg2Parser::is_slice_start_code(0xAF));
        assert!(!Mpeg2Parser::is_slice_start_code(0x00));
        assert!(!Mpeg2Parser::is_slice_start_code(0xB0));
    }

    #[test]
    fn test_parse_sequence_header() {
        // Minimal valid sequence header for 720x576 @ 25fps
        let data = [
            0x00, 0x00, 0x01, 0xB3, // Sequence header start code
            0x2D, 0x02, 0x40, // horizontal_size=720 (12 bits), vertical_size=576 (12 bits partial)
            0x33, // vertical_size (remaining), aspect_ratio=4:3 (3), frame_rate=25fps (3)
            0xFF, 0xFF, 0xE0, // bit_rate (18 bits high value)
            0x00, 0x00, // vbv_buffer_size, constrained_parameters, load matrices flags
        ];

        let mut parser = Mpeg2Parser::new();
        let header = parser.parse_sequence_header(&data);
        assert!(header.is_some());

        let header = header.unwrap();
        assert_eq!(header.horizontal_size, 720);
        assert_eq!(header.vertical_size, 576);
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = Mpeg2Parser::new();
        parser.reset();
        assert!(parser.sequence_header().is_none());
        assert!(parser.gop_header().is_none());
    }

    #[test]
    fn test_detect_mpeg2() {
        // Valid sequence header start
        let data = [0x00, 0x00, 0x01, 0xB3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(detect_mpeg2(&data));

        // Invalid data
        let data = [0x00, 0x00, 0x00, 0x00];
        assert!(!detect_mpeg2(&data));
    }

    #[test]
    fn test_bit_reader() {
        let data = [0b10110100, 0b01010101];
        let mut reader = BitReader::new(&data);

        // Read first bit
        assert_eq!(reader.read_bit(), Some(1));

        // Read next 4 bits
        assert_eq!(reader.read_bits(4), Some(0b0110));

        // Read more bits
        assert_eq!(reader.read_bits(3), Some(0b100));
    }
}
