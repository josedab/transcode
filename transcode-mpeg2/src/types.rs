//! MPEG-2 data types.

use std::fmt;

/// MPEG-2 sequence header information.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceHeader {
    /// Horizontal size in pixels.
    pub horizontal_size: u16,
    /// Vertical size in pixels.
    pub vertical_size: u16,
    /// Aspect ratio code.
    pub aspect_ratio_code: AspectRatioCode,
    /// Frame rate code.
    pub frame_rate_code: FrameRateCode,
    /// Bit rate (in units of 400 bits/second).
    pub bit_rate: u32,
    /// VBV buffer size.
    pub vbv_buffer_size: u16,
    /// Constrained parameters flag.
    pub constrained_parameters_flag: bool,
    /// Load intra quantizer matrix.
    pub load_intra_quantizer_matrix: bool,
    /// Intra quantizer matrix (if loaded).
    pub intra_quantizer_matrix: Option<[u8; 64]>,
    /// Load non-intra quantizer matrix.
    pub load_non_intra_quantizer_matrix: bool,
    /// Non-intra quantizer matrix (if loaded).
    pub non_intra_quantizer_matrix: Option<[u8; 64]>,
}

impl SequenceHeader {
    /// Create a new sequence header with default values.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            horizontal_size: width,
            vertical_size: height,
            aspect_ratio_code: AspectRatioCode::Square,
            frame_rate_code: FrameRateCode::Fps29_97,
            bit_rate: 0,
            vbv_buffer_size: 0,
            constrained_parameters_flag: false,
            load_intra_quantizer_matrix: false,
            intra_quantizer_matrix: None,
            load_non_intra_quantizer_matrix: false,
            non_intra_quantizer_matrix: None,
        }
    }

    /// Get the frame rate in frames per second.
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate_code.fps()
    }

    /// Get the aspect ratio as width:height.
    pub fn aspect_ratio(&self) -> (u16, u16) {
        self.aspect_ratio_code.ratio()
    }

    /// Get the bit rate in bits per second.
    pub fn bit_rate_bps(&self) -> u64 {
        self.bit_rate as u64 * 400
    }
}

/// MPEG-2 sequence extension (for MPEG-2 specific parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceExtension {
    /// Profile and level indication.
    pub profile_and_level: u8,
    /// Progressive sequence flag.
    pub progressive_sequence: bool,
    /// Chroma format.
    pub chroma_format: ChromaFormat,
    /// Horizontal size extension.
    pub horizontal_size_extension: u8,
    /// Vertical size extension.
    pub vertical_size_extension: u8,
    /// Bit rate extension.
    pub bit_rate_extension: u16,
    /// VBV buffer size extension.
    pub vbv_buffer_size_extension: u8,
    /// Low delay flag.
    pub low_delay: bool,
    /// Frame rate extension numerator.
    pub frame_rate_extension_n: u8,
    /// Frame rate extension denominator.
    pub frame_rate_extension_d: u8,
}

impl SequenceExtension {
    /// Get the profile.
    pub fn profile(&self) -> Profile {
        Profile::from_code((self.profile_and_level >> 4) & 0x07)
    }

    /// Get the level.
    pub fn level(&self) -> Level {
        Level::from_code(self.profile_and_level & 0x0F)
    }
}

impl Default for SequenceExtension {
    fn default() -> Self {
        Self {
            profile_and_level: 0x48, // Main Profile @ Main Level
            progressive_sequence: true,
            chroma_format: ChromaFormat::Yuv420,
            horizontal_size_extension: 0,
            vertical_size_extension: 0,
            bit_rate_extension: 0,
            vbv_buffer_size_extension: 0,
            low_delay: false,
            frame_rate_extension_n: 0,
            frame_rate_extension_d: 0,
        }
    }
}

/// Group of Pictures header.
#[derive(Debug, Clone, PartialEq)]
pub struct GopHeader {
    /// Time code - hours.
    pub hours: u8,
    /// Time code - minutes.
    pub minutes: u8,
    /// Time code - seconds.
    pub seconds: u8,
    /// Time code - pictures/frames.
    pub pictures: u8,
    /// Drop frame flag.
    pub drop_frame_flag: bool,
    /// Closed GOP flag.
    pub closed_gop: bool,
    /// Broken link flag.
    pub broken_link: bool,
}

impl GopHeader {
    /// Create a new GOP header.
    pub fn new() -> Self {
        Self {
            hours: 0,
            minutes: 0,
            seconds: 0,
            pictures: 0,
            drop_frame_flag: false,
            closed_gop: true,
            broken_link: false,
        }
    }

    /// Get time code as total frames.
    pub fn time_code_frames(&self, frame_rate: f64) -> u64 {
        let total_seconds = self.hours as u64 * 3600 + self.minutes as u64 * 60 + self.seconds as u64;
        (total_seconds as f64 * frame_rate) as u64 + self.pictures as u64
    }
}

impl Default for GopHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Picture header information.
#[derive(Debug, Clone, PartialEq)]
pub struct PictureHeader {
    /// Temporal reference (display order within GOP).
    pub temporal_reference: u16,
    /// Picture coding type.
    pub picture_coding_type: PictureCodingType,
    /// VBV delay.
    pub vbv_delay: u16,
    /// Full pel forward vector (for P/B frames).
    pub full_pel_forward_vector: bool,
    /// Forward f code (for P/B frames).
    pub forward_f_code: u8,
    /// Full pel backward vector (for B frames).
    pub full_pel_backward_vector: bool,
    /// Backward f code (for B frames).
    pub backward_f_code: u8,
}

impl PictureHeader {
    /// Create a new I-frame picture header.
    pub fn new_i_frame(temporal_reference: u16) -> Self {
        Self {
            temporal_reference,
            picture_coding_type: PictureCodingType::I,
            vbv_delay: 0xFFFF,
            full_pel_forward_vector: false,
            forward_f_code: 0,
            full_pel_backward_vector: false,
            backward_f_code: 0,
        }
    }

    /// Create a new P-frame picture header.
    pub fn new_p_frame(temporal_reference: u16) -> Self {
        Self {
            temporal_reference,
            picture_coding_type: PictureCodingType::P,
            vbv_delay: 0xFFFF,
            full_pel_forward_vector: false,
            forward_f_code: 7,
            full_pel_backward_vector: false,
            backward_f_code: 0,
        }
    }

    /// Create a new B-frame picture header.
    pub fn new_b_frame(temporal_reference: u16) -> Self {
        Self {
            temporal_reference,
            picture_coding_type: PictureCodingType::B,
            vbv_delay: 0xFFFF,
            full_pel_forward_vector: false,
            forward_f_code: 7,
            full_pel_backward_vector: false,
            backward_f_code: 7,
        }
    }
}

/// Picture coding extension (MPEG-2).
#[derive(Debug, Clone, PartialEq)]
pub struct PictureCodingExtension {
    /// Forward horizontal f code.
    pub f_code_00: u8,
    /// Forward vertical f code.
    pub f_code_01: u8,
    /// Backward horizontal f code.
    pub f_code_10: u8,
    /// Backward vertical f code.
    pub f_code_11: u8,
    /// Intra DC precision.
    pub intra_dc_precision: u8,
    /// Picture structure.
    pub picture_structure: PictureStructure,
    /// Top field first.
    pub top_field_first: bool,
    /// Frame prediction frame DCT.
    pub frame_pred_frame_dct: bool,
    /// Concealment motion vectors.
    pub concealment_motion_vectors: bool,
    /// Q scale type.
    pub q_scale_type: bool,
    /// Intra VLC format.
    pub intra_vlc_format: bool,
    /// Alternate scan.
    pub alternate_scan: bool,
    /// Repeat first field.
    pub repeat_first_field: bool,
    /// Chroma 420 type.
    pub chroma_420_type: bool,
    /// Progressive frame.
    pub progressive_frame: bool,
    /// Composite display flag.
    pub composite_display_flag: bool,
}

impl Default for PictureCodingExtension {
    fn default() -> Self {
        Self {
            f_code_00: 1,
            f_code_01: 1,
            f_code_10: 1,
            f_code_11: 1,
            intra_dc_precision: 0,
            picture_structure: PictureStructure::Frame,
            top_field_first: true,
            frame_pred_frame_dct: true,
            concealment_motion_vectors: false,
            q_scale_type: false,
            intra_vlc_format: false,
            alternate_scan: false,
            repeat_first_field: false,
            chroma_420_type: true,
            progressive_frame: true,
            composite_display_flag: false,
        }
    }
}

/// Picture coding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PictureCodingType {
    /// I-frame (Intra).
    I = 1,
    /// P-frame (Predictive).
    P = 2,
    /// B-frame (Bidirectional).
    B = 3,
    /// D-frame (DC Intra - MPEG-1 only).
    D = 4,
}

impl PictureCodingType {
    /// Parse from code value.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(PictureCodingType::I),
            2 => Some(PictureCodingType::P),
            3 => Some(PictureCodingType::B),
            4 => Some(PictureCodingType::D),
            _ => None,
        }
    }

    /// Check if this is a reference frame.
    pub fn is_reference(&self) -> bool {
        matches!(self, PictureCodingType::I | PictureCodingType::P)
    }
}

impl fmt::Display for PictureCodingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PictureCodingType::I => write!(f, "I"),
            PictureCodingType::P => write!(f, "P"),
            PictureCodingType::B => write!(f, "B"),
            PictureCodingType::D => write!(f, "D"),
        }
    }
}

/// Picture structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PictureStructure {
    /// Top field.
    TopField = 1,
    /// Bottom field.
    BottomField = 2,
    /// Frame.
    Frame = 3,
}

impl PictureStructure {
    /// Parse from code value.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(PictureStructure::TopField),
            2 => Some(PictureStructure::BottomField),
            3 => Some(PictureStructure::Frame),
            _ => None,
        }
    }

    /// Check if this is a field.
    pub fn is_field(&self) -> bool {
        !matches!(self, PictureStructure::Frame)
    }
}

impl fmt::Display for PictureStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PictureStructure::TopField => write!(f, "Top Field"),
            PictureStructure::BottomField => write!(f, "Bottom Field"),
            PictureStructure::Frame => write!(f, "Frame"),
        }
    }
}

/// Aspect ratio code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AspectRatioCode {
    /// Forbidden (0).
    Forbidden = 0,
    /// Square pixels (1:1).
    Square = 1,
    /// 4:3 display.
    Ratio4_3 = 2,
    /// 16:9 display.
    Ratio16_9 = 3,
    /// 2.21:1 display.
    Ratio221_1 = 4,
}

impl AspectRatioCode {
    /// Parse from code value.
    pub fn from_code(code: u8) -> Self {
        match code {
            1 => AspectRatioCode::Square,
            2 => AspectRatioCode::Ratio4_3,
            3 => AspectRatioCode::Ratio16_9,
            4 => AspectRatioCode::Ratio221_1,
            _ => AspectRatioCode::Forbidden,
        }
    }

    /// Get the aspect ratio as (width, height).
    pub fn ratio(&self) -> (u16, u16) {
        match self {
            AspectRatioCode::Square => (1, 1),
            AspectRatioCode::Ratio4_3 => (4, 3),
            AspectRatioCode::Ratio16_9 => (16, 9),
            AspectRatioCode::Ratio221_1 => (221, 100),
            AspectRatioCode::Forbidden => (0, 0),
        }
    }
}

impl fmt::Display for AspectRatioCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AspectRatioCode::Square => write!(f, "1:1"),
            AspectRatioCode::Ratio4_3 => write!(f, "4:3"),
            AspectRatioCode::Ratio16_9 => write!(f, "16:9"),
            AspectRatioCode::Ratio221_1 => write!(f, "2.21:1"),
            AspectRatioCode::Forbidden => write!(f, "Forbidden"),
        }
    }
}

/// Frame rate code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameRateCode {
    /// Forbidden (0).
    Forbidden = 0,
    /// 23.976 fps (24000/1001).
    Fps23_976 = 1,
    /// 24 fps.
    Fps24 = 2,
    /// 25 fps.
    Fps25 = 3,
    /// 29.97 fps (30000/1001).
    Fps29_97 = 4,
    /// 30 fps.
    Fps30 = 5,
    /// 50 fps.
    Fps50 = 6,
    /// 59.94 fps (60000/1001).
    Fps59_94 = 7,
    /// 60 fps.
    Fps60 = 8,
}

impl FrameRateCode {
    /// Parse from code value.
    pub fn from_code(code: u8) -> Self {
        match code {
            1 => FrameRateCode::Fps23_976,
            2 => FrameRateCode::Fps24,
            3 => FrameRateCode::Fps25,
            4 => FrameRateCode::Fps29_97,
            5 => FrameRateCode::Fps30,
            6 => FrameRateCode::Fps50,
            7 => FrameRateCode::Fps59_94,
            8 => FrameRateCode::Fps60,
            _ => FrameRateCode::Forbidden,
        }
    }

    /// Get the frame rate in fps.
    pub fn fps(&self) -> f64 {
        match self {
            FrameRateCode::Fps23_976 => 24000.0 / 1001.0,
            FrameRateCode::Fps24 => 24.0,
            FrameRateCode::Fps25 => 25.0,
            FrameRateCode::Fps29_97 => 30000.0 / 1001.0,
            FrameRateCode::Fps30 => 30.0,
            FrameRateCode::Fps50 => 50.0,
            FrameRateCode::Fps59_94 => 60000.0 / 1001.0,
            FrameRateCode::Fps60 => 60.0,
            FrameRateCode::Forbidden => 0.0,
        }
    }

    /// Get frame rate as fraction (num, den).
    pub fn fraction(&self) -> (u32, u32) {
        match self {
            FrameRateCode::Fps23_976 => (24000, 1001),
            FrameRateCode::Fps24 => (24, 1),
            FrameRateCode::Fps25 => (25, 1),
            FrameRateCode::Fps29_97 => (30000, 1001),
            FrameRateCode::Fps30 => (30, 1),
            FrameRateCode::Fps50 => (50, 1),
            FrameRateCode::Fps59_94 => (60000, 1001),
            FrameRateCode::Fps60 => (60, 1),
            FrameRateCode::Forbidden => (0, 1),
        }
    }
}

impl fmt::Display for FrameRateCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3} fps", self.fps())
    }
}

/// Chroma format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChromaFormat {
    /// 4:2:0 chroma subsampling.
    Yuv420 = 1,
    /// 4:2:2 chroma subsampling.
    Yuv422 = 2,
    /// 4:4:4 no chroma subsampling.
    Yuv444 = 3,
}

impl ChromaFormat {
    /// Parse from code value.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(ChromaFormat::Yuv420),
            2 => Some(ChromaFormat::Yuv422),
            3 => Some(ChromaFormat::Yuv444),
            _ => None,
        }
    }
}

impl fmt::Display for ChromaFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChromaFormat::Yuv420 => write!(f, "4:2:0"),
            ChromaFormat::Yuv422 => write!(f, "4:2:2"),
            ChromaFormat::Yuv444 => write!(f, "4:4:4"),
        }
    }
}

/// MPEG-2 profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Profile {
    /// Simple Profile.
    Simple,
    /// Main Profile.
    Main,
    /// SNR Scalable Profile.
    SnrScalable,
    /// Spatially Scalable Profile.
    SpatiallyScalable,
    /// High Profile.
    High,
    /// Unknown profile.
    Unknown(u8),
}

impl Profile {
    /// Parse from profile code.
    pub fn from_code(code: u8) -> Self {
        match code {
            5 => Profile::Simple,
            4 => Profile::Main,
            3 => Profile::SnrScalable,
            2 => Profile::SpatiallyScalable,
            1 => Profile::High,
            _ => Profile::Unknown(code),
        }
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Profile::Simple => write!(f, "Simple"),
            Profile::Main => write!(f, "Main"),
            Profile::SnrScalable => write!(f, "SNR Scalable"),
            Profile::SpatiallyScalable => write!(f, "Spatially Scalable"),
            Profile::High => write!(f, "High"),
            Profile::Unknown(code) => write!(f, "Unknown({})", code),
        }
    }
}

/// MPEG-2 level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Level {
    /// Low Level.
    Low,
    /// Main Level.
    Main,
    /// High 1440 Level.
    High1440,
    /// High Level.
    High,
    /// Unknown level.
    Unknown(u8),
}

impl Level {
    /// Parse from level code.
    pub fn from_code(code: u8) -> Self {
        match code {
            10 => Level::Low,
            8 => Level::Main,
            6 => Level::High1440,
            4 => Level::High,
            _ => Level::Unknown(code),
        }
    }

    /// Convert to level code.
    pub fn to_code(&self) -> u8 {
        match self {
            Level::Low => 10,
            Level::Main => 8,
            Level::High1440 => 6,
            Level::High => 4,
            Level::Unknown(code) => *code,
        }
    }

    /// Get maximum bit rate for this level (in Mbps).
    pub fn max_bitrate_mbps(&self) -> u32 {
        match self {
            Level::Low => 4,
            Level::Main => 15,
            Level::High1440 => 60,
            Level::High => 80,
            Level::Unknown(_) => 0,
        }
    }

    /// Get maximum frame size for this level.
    pub fn max_frame_size(&self) -> (u16, u16) {
        match self {
            Level::Low => (352, 288),
            Level::Main => (720, 576),
            Level::High1440 => (1440, 1152),
            Level::High => (1920, 1152),
            Level::Unknown(_) => (0, 0),
        }
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Level::Low => write!(f, "Low"),
            Level::Main => write!(f, "Main"),
            Level::High1440 => write!(f, "High-1440"),
            Level::High => write!(f, "High"),
            Level::Unknown(code) => write!(f, "Unknown({})", code),
        }
    }
}

/// Decoded video frame from MPEG-2.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Frame width.
    pub width: u16,
    /// Frame height.
    pub height: u16,
    /// Picture coding type.
    pub picture_type: PictureCodingType,
    /// Temporal reference.
    pub temporal_reference: u16,
    /// Progressive frame.
    pub progressive: bool,
    /// Top field first (for interlaced).
    pub top_field_first: bool,
    /// Y plane data.
    pub y_plane: Vec<u8>,
    /// U (Cb) plane data.
    pub u_plane: Vec<u8>,
    /// V (Cr) plane data.
    pub v_plane: Vec<u8>,
    /// Presentation timestamp.
    pub pts: Option<u64>,
    /// Decode timestamp.
    pub dts: Option<u64>,
}

impl DecodedFrame {
    /// Create a new empty decoded frame.
    pub fn new(width: u16, height: u16) -> Self {
        let y_size = width as usize * height as usize;
        let uv_size = y_size / 4; // 4:2:0

        Self {
            width,
            height,
            picture_type: PictureCodingType::I,
            temporal_reference: 0,
            progressive: true,
            top_field_first: true,
            y_plane: vec![0; y_size],
            u_plane: vec![0; uv_size],
            v_plane: vec![0; uv_size],
            pts: None,
            dts: None,
        }
    }

    /// Get total frame size in bytes.
    pub fn size(&self) -> usize {
        self.y_plane.len() + self.u_plane.len() + self.v_plane.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_rate_code() {
        assert!((FrameRateCode::Fps29_97.fps() - 29.97).abs() < 0.01);
        assert_eq!(FrameRateCode::Fps25.fraction(), (25, 1));
    }

    #[test]
    fn test_aspect_ratio_code() {
        assert_eq!(AspectRatioCode::Ratio16_9.ratio(), (16, 9));
    }

    #[test]
    fn test_picture_coding_type() {
        assert!(PictureCodingType::I.is_reference());
        assert!(PictureCodingType::P.is_reference());
        assert!(!PictureCodingType::B.is_reference());
    }

    #[test]
    fn test_profile_level() {
        assert_eq!(Profile::from_code(4), Profile::Main);
        assert_eq!(Level::from_code(8), Level::Main);
        assert_eq!(Level::Main.max_bitrate_mbps(), 15);
    }

    #[test]
    fn test_chroma_format() {
        assert_eq!(ChromaFormat::from_code(1), Some(ChromaFormat::Yuv420));
        assert_eq!(ChromaFormat::from_code(4), None);
    }

    #[test]
    fn test_sequence_header() {
        let header = SequenceHeader::new(720, 576);
        assert_eq!(header.horizontal_size, 720);
        assert_eq!(header.vertical_size, 576);
    }

    #[test]
    fn test_decoded_frame() {
        let frame = DecodedFrame::new(720, 576);
        // 4:2:0: Y=720*576, U=V=360*288
        assert_eq!(frame.y_plane.len(), 720 * 576);
        assert_eq!(frame.u_plane.len(), 720 * 576 / 4);
    }
}
