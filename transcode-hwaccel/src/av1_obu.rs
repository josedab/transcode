//! AV1 Open Bitstream Unit (OBU) utilities.
//!
//! This module provides types and functions for working with AV1 OBUs,
//! which are the fundamental units of an AV1 bitstream.
//!
//! # OBU Structure
//!
//! Each OBU consists of:
//! 1. OBU header (1-2 bytes)
//! 2. Optional OBU size (variable length using LEB128)
//! 3. OBU payload
//!
//! # OBU Types
//!
//! - Sequence Header: Contains codec configuration
//! - Temporal Delimiter: Marks frame boundaries
//! - Frame Header: Contains per-frame parameters
//! - Tile Group: Contains compressed tile data
//! - Metadata: Contains supplementary information
//! - Frame: Combined frame header and tile group
//! - Redundant Frame Header: For error resilience
//! - Tile List: For large-scale tile decoding
//! - Padding: For byte alignment

use crate::av1_hw::{Av1Level, Av1Profile, Av1TileConfig};
use crate::error::{HwAccelError, Result};

/// OBU type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ObuType {
    /// Reserved (0).
    Reserved = 0,
    /// Sequence header OBU.
    SequenceHeader = 1,
    /// Temporal delimiter OBU.
    TemporalDelimiter = 2,
    /// Frame header OBU.
    FrameHeader = 3,
    /// Tile group OBU.
    TileGroup = 4,
    /// Metadata OBU.
    Metadata = 5,
    /// Frame OBU (combined frame header and tile group).
    Frame = 6,
    /// Redundant frame header OBU.
    RedundantFrameHeader = 7,
    /// Tile list OBU.
    TileList = 8,
    /// Padding OBU.
    Padding = 15,
}

impl ObuType {
    /// Create OBU type from raw value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(ObuType::Reserved),
            1 => Some(ObuType::SequenceHeader),
            2 => Some(ObuType::TemporalDelimiter),
            3 => Some(ObuType::FrameHeader),
            4 => Some(ObuType::TileGroup),
            5 => Some(ObuType::Metadata),
            6 => Some(ObuType::Frame),
            7 => Some(ObuType::RedundantFrameHeader),
            8 => Some(ObuType::TileList),
            15 => Some(ObuType::Padding),
            _ => None,
        }
    }

    /// Get the name of this OBU type.
    pub fn name(&self) -> &'static str {
        match self {
            ObuType::Reserved => "Reserved",
            ObuType::SequenceHeader => "Sequence Header",
            ObuType::TemporalDelimiter => "Temporal Delimiter",
            ObuType::FrameHeader => "Frame Header",
            ObuType::TileGroup => "Tile Group",
            ObuType::Metadata => "Metadata",
            ObuType::Frame => "Frame",
            ObuType::RedundantFrameHeader => "Redundant Frame Header",
            ObuType::TileList => "Tile List",
            ObuType::Padding => "Padding",
        }
    }

    /// Check if this OBU type contains frame data.
    pub fn has_frame_data(&self) -> bool {
        matches!(self, ObuType::TileGroup | ObuType::Frame)
    }

    /// Check if this OBU type is a header.
    pub fn is_header(&self) -> bool {
        matches!(
            self,
            ObuType::SequenceHeader | ObuType::FrameHeader | ObuType::RedundantFrameHeader
        )
    }
}

/// OBU header structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObuHeader {
    /// OBU type.
    pub obu_type: ObuType,
    /// Has extension flag.
    pub has_extension: bool,
    /// Has size field flag.
    pub has_size_field: bool,
    /// Temporal ID (from extension, 0-7).
    pub temporal_id: u8,
    /// Spatial ID (from extension, 0-3).
    pub spatial_id: u8,
}

impl ObuHeader {
    /// Create a new OBU header.
    pub fn new(obu_type: ObuType) -> Self {
        Self {
            obu_type,
            has_extension: false,
            has_size_field: true, // Almost always true for streaming
            temporal_id: 0,
            spatial_id: 0,
        }
    }

    /// Create a new OBU header with extension.
    pub fn with_extension(obu_type: ObuType, temporal_id: u8, spatial_id: u8) -> Self {
        Self {
            obu_type,
            has_extension: true,
            has_size_field: true,
            temporal_id: temporal_id & 0x07,
            spatial_id: spatial_id & 0x03,
        }
    }

    /// Parse OBU header from bytes.
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        if data.is_empty() {
            return Err(HwAccelError::Decode("Empty OBU data".to_string()));
        }

        let first_byte = data[0];

        // forbidden bit (must be 0)
        if (first_byte & 0x80) != 0 {
            return Err(HwAccelError::Decode(
                "OBU forbidden bit is set".to_string(),
            ));
        }

        let obu_type_value = (first_byte >> 3) & 0x0F;
        let obu_type = ObuType::from_u8(obu_type_value)
            .ok_or_else(|| HwAccelError::Decode(format!("Invalid OBU type: {}", obu_type_value)))?;

        let has_extension = (first_byte & 0x04) != 0;
        let has_size_field = (first_byte & 0x02) != 0;
        // reserved bit at 0x01

        let mut header_size = 1;
        let mut temporal_id = 0;
        let mut spatial_id = 0;

        if has_extension {
            if data.len() < 2 {
                return Err(HwAccelError::Decode(
                    "OBU extension byte missing".to_string(),
                ));
            }
            let ext_byte = data[1];
            temporal_id = (ext_byte >> 5) & 0x07;
            spatial_id = (ext_byte >> 3) & 0x03;
            header_size = 2;
        }

        Ok((
            Self {
                obu_type,
                has_extension,
                has_size_field,
                temporal_id,
                spatial_id,
            },
            header_size,
        ))
    }

    /// Serialize OBU header to bytes.
    pub fn write(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(2);

        let mut first_byte = 0u8;
        // forbidden bit = 0
        first_byte |= (self.obu_type as u8 & 0x0F) << 3;
        if self.has_extension {
            first_byte |= 0x04;
        }
        if self.has_size_field {
            first_byte |= 0x02;
        }
        result.push(first_byte);

        if self.has_extension {
            let ext_byte = ((self.temporal_id & 0x07) << 5) | ((self.spatial_id & 0x03) << 3);
            result.push(ext_byte);
        }

        result
    }

    /// Get header size in bytes.
    pub fn size(&self) -> usize {
        if self.has_extension {
            2
        } else {
            1
        }
    }
}

/// Parsed OBU with header and payload.
#[derive(Debug, Clone)]
pub struct Obu {
    /// OBU header.
    pub header: ObuHeader,
    /// OBU payload data.
    pub payload: Vec<u8>,
}

impl Obu {
    /// Create a new OBU.
    pub fn new(obu_type: ObuType, payload: Vec<u8>) -> Self {
        Self {
            header: ObuHeader::new(obu_type),
            payload,
        }
    }

    /// Create a new OBU with extension.
    pub fn with_extension(
        obu_type: ObuType,
        temporal_id: u8,
        spatial_id: u8,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            header: ObuHeader::with_extension(obu_type, temporal_id, spatial_id),
            payload,
        }
    }

    /// Parse a single OBU from bytes.
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        let (header, header_size) = ObuHeader::parse(data)?;

        let remaining = &data[header_size..];
        let (payload_size, size_bytes) = if header.has_size_field {
            parse_leb128(remaining)?
        } else {
            (remaining.len() as u64, 0)
        };

        let total_header_size = header_size + size_bytes;
        let obu_end = total_header_size + payload_size as usize;

        if data.len() < obu_end {
            return Err(HwAccelError::Decode(format!(
                "OBU payload truncated: expected {} bytes, got {}",
                payload_size,
                data.len() - total_header_size
            )));
        }

        let payload = data[total_header_size..obu_end].to_vec();

        Ok((Self { header, payload }, obu_end))
    }

    /// Serialize OBU to bytes (with size field).
    pub fn write(&self) -> Vec<u8> {
        let header_bytes = self.header.write();
        let size_bytes = encode_leb128(self.payload.len() as u64);

        let mut result = Vec::with_capacity(header_bytes.len() + size_bytes.len() + self.payload.len());
        result.extend_from_slice(&header_bytes);
        if self.header.has_size_field {
            result.extend_from_slice(&size_bytes);
        }
        result.extend_from_slice(&self.payload);
        result
    }

    /// Get total OBU size in bytes.
    pub fn total_size(&self) -> usize {
        let size_field_len = if self.header.has_size_field {
            leb128_size(self.payload.len() as u64)
        } else {
            0
        };
        self.header.size() + size_field_len + self.payload.len()
    }
}

/// Temporal delimiter OBU.
///
/// A temporal delimiter marks the boundary between frames in the bitstream.
/// It contains no payload data.
#[derive(Debug, Clone, Copy, Default)]
pub struct TemporalDelimiter;

impl TemporalDelimiter {
    /// Create a new temporal delimiter.
    pub fn new() -> Self {
        Self
    }

    /// Create the OBU for this temporal delimiter.
    pub fn to_obu(&self) -> Obu {
        Obu::new(ObuType::TemporalDelimiter, Vec::new())
    }

    /// Write temporal delimiter to bytes.
    pub fn write(&self) -> Vec<u8> {
        self.to_obu().write()
    }
}

/// Color primaries for sequence header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ColorPrimaries {
    /// BT.709 (sRGB).
    #[default]
    Bt709 = 1,
    /// Unspecified.
    Unspecified = 2,
    /// BT.470M.
    Bt470M = 4,
    /// BT.470BG.
    Bt470Bg = 5,
    /// BT.601 / SMPTE 170M.
    Bt601 = 6,
    /// SMPTE 240M.
    Smpte240 = 7,
    /// Generic film.
    GenericFilm = 8,
    /// BT.2020.
    Bt2020 = 9,
    /// SMPTE ST 428 (XYZ).
    Xyz = 10,
    /// SMPTE RP 431 (DCI-P3).
    SmpteRp431 = 11,
    /// SMPTE EG 432 (Display P3).
    SmpteEg432 = 12,
    /// EBU Tech 3213.
    Ebu3213 = 22,
}

/// Transfer characteristics for sequence header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TransferCharacteristics {
    /// BT.709.
    #[default]
    Bt709 = 1,
    /// Unspecified.
    Unspecified = 2,
    /// BT.470M.
    Bt470M = 4,
    /// BT.470BG.
    Bt470Bg = 5,
    /// BT.601.
    Bt601 = 6,
    /// SMPTE 240M.
    Smpte240 = 7,
    /// Linear.
    Linear = 8,
    /// Logarithmic (100:1).
    Log100 = 9,
    /// Logarithmic (100 * sqrt(10):1).
    Log100Sqrt10 = 10,
    /// IEC 61966-2-4.
    Iec61966_2_4 = 11,
    /// BT.1361.
    Bt1361 = 12,
    /// sRGB.
    Srgb = 13,
    /// BT.2020 10-bit.
    Bt2020_10 = 14,
    /// BT.2020 12-bit.
    Bt2020_12 = 15,
    /// SMPTE ST 2084 (PQ / HDR10).
    Smpte2084 = 16,
    /// SMPTE ST 428.
    Smpte428 = 17,
    /// ARIB STD-B67 (HLG).
    AribStdB67 = 18,
}

/// Matrix coefficients for sequence header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum MatrixCoefficients {
    /// Identity (RGB/XYZ).
    Identity = 0,
    /// BT.709.
    #[default]
    Bt709 = 1,
    /// Unspecified.
    Unspecified = 2,
    /// US FCC.
    Fcc = 4,
    /// BT.470BG.
    Bt470Bg = 5,
    /// BT.601.
    Bt601 = 6,
    /// SMPTE 240M.
    Smpte240 = 7,
    /// YCgCo.
    YCgCo = 8,
    /// BT.2020 non-constant luminance.
    Bt2020Ncl = 9,
    /// BT.2020 constant luminance.
    Bt2020Cl = 10,
    /// SMPTE ST 2085.
    Smpte2085 = 11,
    /// Chromaticity-derived non-constant luminance.
    ChromaDerivedNcl = 12,
    /// Chromaticity-derived constant luminance.
    ChromaDerivedCl = 13,
    /// ICtCp.
    ICtCp = 14,
}

/// Color configuration for sequence header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorConfig {
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,
    /// Mono chrome flag.
    pub mono_chrome: bool,
    /// Color description present.
    pub color_description_present: bool,
    /// Color primaries.
    pub color_primaries: ColorPrimaries,
    /// Transfer characteristics.
    pub transfer_characteristics: TransferCharacteristics,
    /// Matrix coefficients.
    pub matrix_coefficients: MatrixCoefficients,
    /// Full color range.
    pub color_range: bool,
    /// Chroma subsampling X (0 for 4:4:4, 1 for 4:2:0/4:2:2).
    pub subsampling_x: bool,
    /// Chroma subsampling Y (0 for 4:4:4/4:2:2, 1 for 4:2:0).
    pub subsampling_y: bool,
    /// Chroma sample position.
    pub chroma_sample_position: u8,
    /// Separate UV delta Q.
    pub separate_uv_delta_q: bool,
}

impl ColorConfig {
    /// Create SDR (Standard Dynamic Range) configuration.
    pub fn sdr(bit_depth: u8) -> Self {
        Self {
            bit_depth,
            mono_chrome: false,
            color_description_present: true,
            color_primaries: ColorPrimaries::Bt709,
            transfer_characteristics: TransferCharacteristics::Bt709,
            matrix_coefficients: MatrixCoefficients::Bt709,
            color_range: false,
            subsampling_x: true,
            subsampling_y: true, // 4:2:0
            chroma_sample_position: 0,
            separate_uv_delta_q: false,
        }
    }

    /// Create HDR10 (PQ) configuration.
    pub fn hdr10() -> Self {
        Self {
            bit_depth: 10,
            mono_chrome: false,
            color_description_present: true,
            color_primaries: ColorPrimaries::Bt2020,
            transfer_characteristics: TransferCharacteristics::Smpte2084,
            matrix_coefficients: MatrixCoefficients::Bt2020Ncl,
            color_range: false,
            subsampling_x: true,
            subsampling_y: true, // 4:2:0
            chroma_sample_position: 0,
            separate_uv_delta_q: false,
        }
    }

    /// Create HLG (Hybrid Log-Gamma) configuration.
    pub fn hlg() -> Self {
        Self {
            bit_depth: 10,
            mono_chrome: false,
            color_description_present: true,
            color_primaries: ColorPrimaries::Bt2020,
            transfer_characteristics: TransferCharacteristics::AribStdB67,
            matrix_coefficients: MatrixCoefficients::Bt2020Ncl,
            color_range: false,
            subsampling_x: true,
            subsampling_y: true, // 4:2:0
            chroma_sample_position: 0,
            separate_uv_delta_q: false,
        }
    }
}

impl Default for ColorConfig {
    fn default() -> Self {
        Self::sdr(8)
    }
}

/// Timing information for sequence header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimingInfo {
    /// Number of time units in one second.
    pub num_units_in_display_tick: u32,
    /// Number of time units in one display tick.
    pub time_scale: u32,
    /// Equal picture interval flag.
    pub equal_picture_interval: bool,
    /// Number of ticks per picture (if equal_picture_interval).
    pub num_ticks_per_picture: u32,
}

impl TimingInfo {
    /// Create timing info for given frame rate.
    pub fn from_fps(fps_num: u32, fps_den: u32) -> Self {
        Self {
            num_units_in_display_tick: fps_den,
            time_scale: fps_num,
            equal_picture_interval: true,
            num_ticks_per_picture: 1,
        }
    }

    /// Create timing info for 30 fps.
    pub fn fps_30() -> Self {
        Self::from_fps(30, 1)
    }

    /// Create timing info for 60 fps.
    pub fn fps_60() -> Self {
        Self::from_fps(60, 1)
    }

    /// Create timing info for 23.976 fps (film).
    pub fn fps_23_976() -> Self {
        Self::from_fps(24000, 1001)
    }

    /// Create timing info for 29.97 fps (NTSC).
    pub fn fps_29_97() -> Self {
        Self::from_fps(30000, 1001)
    }
}

impl Default for TimingInfo {
    fn default() -> Self {
        Self::fps_30()
    }
}

/// Sequence header OBU parameters.
#[derive(Debug, Clone)]
pub struct SequenceHeader {
    /// Profile.
    pub profile: Av1Profile,
    /// Still picture flag.
    pub still_picture: bool,
    /// Reduced still picture header flag.
    pub reduced_still_picture_header: bool,
    /// Operating point count.
    pub operating_points: Vec<OperatingPoint>,
    /// Maximum frame width minus 1.
    pub max_frame_width_minus_1: u16,
    /// Maximum frame height minus 1.
    pub max_frame_height_minus_1: u16,
    /// Frame ID length (bits).
    pub frame_id_length: u8,
    /// Delta frame ID length (bits).
    pub delta_frame_id_length: u8,
    /// Use 128x128 superblock.
    pub use_128x128_superblock: bool,
    /// Enable filter intra.
    pub enable_filter_intra: bool,
    /// Enable intra edge filter.
    pub enable_intra_edge_filter: bool,
    /// Enable interintra compound.
    pub enable_interintra_compound: bool,
    /// Enable masked compound.
    pub enable_masked_compound: bool,
    /// Enable warped motion.
    pub enable_warped_motion: bool,
    /// Enable dual filter.
    pub enable_dual_filter: bool,
    /// Enable order hint.
    pub enable_order_hint: bool,
    /// Enable JNT_COMP.
    pub enable_jnt_comp: bool,
    /// Enable ref frame MVS.
    pub enable_ref_frame_mvs: bool,
    /// Sequence choose screen content tools.
    pub seq_choose_screen_content_tools: bool,
    /// Force screen content tools.
    pub seq_force_screen_content_tools: u8,
    /// Sequence choose integer MV.
    pub seq_choose_integer_mv: bool,
    /// Force integer MV.
    pub seq_force_integer_mv: u8,
    /// Order hint bits minus 1.
    pub order_hint_bits_minus_1: u8,
    /// Enable superres.
    pub enable_superres: bool,
    /// Enable CDEF.
    pub enable_cdef: bool,
    /// Enable restoration.
    pub enable_restoration: bool,
    /// Color configuration.
    pub color_config: ColorConfig,
    /// Film grain parameters present.
    pub film_grain_params_present: bool,
    /// Timing info.
    pub timing_info: Option<TimingInfo>,
}

/// Operating point in sequence header.
#[derive(Debug, Clone, Copy)]
pub struct OperatingPoint {
    /// Operating point IDC.
    pub idc: u16,
    /// Sequence level index.
    pub seq_level_idx: u8,
    /// Sequence tier.
    pub seq_tier: u8,
    /// Decoder model present for this operating point.
    pub decoder_model_present: bool,
    /// Initial display delay present.
    pub initial_display_delay_present: bool,
    /// Initial display delay minus 1.
    pub initial_display_delay_minus_1: u8,
}

impl OperatingPoint {
    /// Create a new operating point.
    pub fn new(level: Av1Level, tier: u8) -> Self {
        Self {
            idc: 0, // All spatial and temporal layers
            seq_level_idx: level.seq_level_idx(),
            seq_tier: tier,
            decoder_model_present: false,
            initial_display_delay_present: false,
            initial_display_delay_minus_1: 0,
        }
    }
}

impl SequenceHeader {
    /// Create a new sequence header with common defaults.
    pub fn new(width: u32, height: u32, profile: Av1Profile, level: Av1Level) -> Self {
        Self {
            profile,
            still_picture: false,
            reduced_still_picture_header: false,
            operating_points: vec![OperatingPoint::new(level, 0)],
            max_frame_width_minus_1: (width - 1) as u16,
            max_frame_height_minus_1: (height - 1) as u16,
            frame_id_length: 15,
            delta_frame_id_length: 14,
            use_128x128_superblock: true,
            enable_filter_intra: true,
            enable_intra_edge_filter: true,
            enable_interintra_compound: true,
            enable_masked_compound: true,
            enable_warped_motion: true,
            enable_dual_filter: true,
            enable_order_hint: true,
            enable_jnt_comp: true,
            enable_ref_frame_mvs: true,
            seq_choose_screen_content_tools: true,
            seq_force_screen_content_tools: 2, // SELECT_SCREEN_CONTENT_TOOLS
            seq_choose_integer_mv: true,
            seq_force_integer_mv: 2, // SELECT_INTEGER_MV
            order_hint_bits_minus_1: 6,
            enable_superres: false,
            enable_cdef: true,
            enable_restoration: true,
            color_config: ColorConfig::sdr(8),
            film_grain_params_present: false,
            timing_info: Some(TimingInfo::fps_30()),
        }
    }

    /// Set color configuration.
    pub fn with_color_config(mut self, config: ColorConfig) -> Self {
        self.color_config = config;
        self
    }

    /// Set timing info.
    pub fn with_timing(mut self, timing: TimingInfo) -> Self {
        self.timing_info = Some(timing);
        self
    }

    /// Enable film grain.
    pub fn with_film_grain(mut self, enabled: bool) -> Self {
        self.film_grain_params_present = enabled;
        self
    }

    /// Generate the OBU for this sequence header.
    ///
    /// Note: This is a simplified implementation. A full implementation
    /// would need bit-level writing of all fields.
    pub fn to_obu(&self) -> Obu {
        // Simplified sequence header generation
        // In a real implementation, this would write each field according to the AV1 spec
        let mut payload = Vec::new();

        // Simplified placeholder - real implementation needs bitstream writer
        // seq_profile (3 bits)
        let mut first_byte = self.profile.seq_profile() << 5;
        // still_picture (1 bit)
        if self.still_picture {
            first_byte |= 0x10;
        }
        // reduced_still_picture_header (1 bit)
        if self.reduced_still_picture_header {
            first_byte |= 0x08;
        }
        payload.push(first_byte);

        // This is a placeholder - actual implementation would continue with:
        // - timing_info_present_flag
        // - decoder_model_info_present_flag
        // - initial_display_delay_present_flag
        // - operating_points_cnt_minus_1
        // - operating_point_idc[i]
        // - seq_level_idx[i]
        // - seq_tier[i]
        // - decoder_model_present_for_this_op[i]
        // - initial_display_delay_present_for_this_op[i]
        // - frame_width_bits_minus_1
        // - frame_height_bits_minus_1
        // - max_frame_width_minus_1
        // - max_frame_height_minus_1
        // - etc.

        // Add minimal placeholder data
        payload.extend_from_slice(&[
            0x00, // placeholder for remaining header fields
            ((self.max_frame_width_minus_1 >> 8) & 0xFF) as u8,
            (self.max_frame_width_minus_1 & 0xFF) as u8,
            ((self.max_frame_height_minus_1 >> 8) & 0xFF) as u8,
            (self.max_frame_height_minus_1 & 0xFF) as u8,
        ]);

        Obu::new(ObuType::SequenceHeader, payload)
    }

    /// Get frame dimensions.
    pub fn frame_dimensions(&self) -> (u32, u32) {
        (
            self.max_frame_width_minus_1 as u32 + 1,
            self.max_frame_height_minus_1 as u32 + 1,
        )
    }
}

/// Frame type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1FrameType {
    /// Key frame (random access point).
    Key = 0,
    /// Inter frame (predicted from references).
    Inter = 1,
    /// Intra-only frame (not a random access point).
    IntraOnly = 2,
    /// Switch frame (for layer switching).
    Switch = 3,
}

impl Av1FrameType {
    /// Check if this is a keyframe.
    pub fn is_key(&self) -> bool {
        matches!(self, Av1FrameType::Key)
    }

    /// Check if this is an intra frame.
    pub fn is_intra(&self) -> bool {
        matches!(self, Av1FrameType::Key | Av1FrameType::IntraOnly)
    }

    /// Get frame type from value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Av1FrameType::Key),
            1 => Some(Av1FrameType::Inter),
            2 => Some(Av1FrameType::IntraOnly),
            3 => Some(Av1FrameType::Switch),
            _ => None,
        }
    }
}

/// Frame header parameters.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// Show existing frame flag.
    pub show_existing_frame: bool,
    /// Frame to show map index.
    pub frame_to_show_map_idx: u8,
    /// Frame type.
    pub frame_type: Av1FrameType,
    /// Show frame flag.
    pub show_frame: bool,
    /// Showable frame flag.
    pub showable_frame: bool,
    /// Error resilient mode.
    pub error_resilient_mode: bool,
    /// Disable CDF update.
    pub disable_cdf_update: bool,
    /// Allow screen content tools.
    pub allow_screen_content_tools: bool,
    /// Force integer MV.
    pub force_integer_mv: bool,
    /// Frame width minus 1.
    pub frame_width_minus_1: u16,
    /// Frame height minus 1.
    pub frame_height_minus_1: u16,
    /// Render width minus 1.
    pub render_width_minus_1: u16,
    /// Render height minus 1.
    pub render_height_minus_1: u16,
    /// Tile configuration.
    pub tiles: Av1TileConfig,
    /// Quantization base Q index.
    pub base_q_idx: u8,
    /// Segmentation enabled.
    pub segmentation_enabled: bool,
    /// Loop filter level.
    pub loop_filter_level: [u8; 4],
    /// CDEF damping minus 3.
    pub cdef_damping_minus_3: u8,
    /// CDEF bits.
    pub cdef_bits: u8,
    /// Loop restoration type.
    pub lr_type: [u8; 3],
}

impl FrameHeader {
    /// Create a new keyframe header.
    pub fn keyframe(width: u32, height: u32, q_index: u8) -> Self {
        Self {
            show_existing_frame: false,
            frame_to_show_map_idx: 0,
            frame_type: Av1FrameType::Key,
            show_frame: true,
            showable_frame: true,
            error_resilient_mode: false,
            disable_cdf_update: false,
            allow_screen_content_tools: false,
            force_integer_mv: false,
            frame_width_minus_1: (width - 1) as u16,
            frame_height_minus_1: (height - 1) as u16,
            render_width_minus_1: (width - 1) as u16,
            render_height_minus_1: (height - 1) as u16,
            tiles: Av1TileConfig::default(),
            base_q_idx: q_index,
            segmentation_enabled: false,
            loop_filter_level: [0, 0, 0, 0],
            cdef_damping_minus_3: 0,
            cdef_bits: 0,
            lr_type: [0, 0, 0],
        }
    }

    /// Create an inter frame header.
    pub fn inter_frame(width: u32, height: u32, q_index: u8) -> Self {
        let mut header = Self::keyframe(width, height, q_index);
        header.frame_type = Av1FrameType::Inter;
        header
    }

    /// Set tile configuration.
    pub fn with_tiles(mut self, tiles: Av1TileConfig) -> Self {
        self.tiles = tiles;
        self
    }

    /// Set loop filter levels.
    pub fn with_loop_filter(mut self, levels: [u8; 4]) -> Self {
        self.loop_filter_level = levels;
        self
    }
}

/// Metadata type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MetadataType {
    /// HDR CLL (Content Light Level).
    HdrCll = 1,
    /// HDR MDCV (Mastering Display Color Volume).
    HdrMdcv = 2,
    /// Scalability.
    Scalability = 3,
    /// ITUT T.35.
    ItutT35 = 4,
    /// Timecode.
    Timecode = 5,
}

/// HDR Content Light Level metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdrCll {
    /// Maximum content light level.
    pub max_cll: u16,
    /// Maximum frame average light level.
    pub max_fall: u16,
}

impl HdrCll {
    /// Create new HDR CLL metadata.
    pub fn new(max_cll: u16, max_fall: u16) -> Self {
        Self { max_cll, max_fall }
    }

    /// Create metadata OBU.
    pub fn to_obu(&self) -> Obu {
        let mut payload = vec![MetadataType::HdrCll as u8];
        payload.extend_from_slice(&self.max_cll.to_be_bytes());
        payload.extend_from_slice(&self.max_fall.to_be_bytes());
        Obu::new(ObuType::Metadata, payload)
    }
}

/// HDR Mastering Display Color Volume metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdrMdcv {
    /// Primary chromaticity coordinates (R, G, B, WP) - x, y pairs.
    pub primaries: [(u16, u16); 4],
    /// Luminance min (in 0.0001 cd/m^2 units).
    pub luminance_min: u32,
    /// Luminance max (in 0.0001 cd/m^2 units).
    pub luminance_max: u32,
}

impl HdrMdcv {
    /// Create standard HDR10 mastering display metadata.
    pub fn hdr10_standard() -> Self {
        Self {
            primaries: [
                (0x8A48, 0x3908), // R: 0.680, 0.320
                (0x4288, 0xB438), // G: 0.265, 0.690
                (0x1D50, 0x0BB8), // B: 0.150, 0.060
                (0x3D13, 0x4189), // WP: 0.3127, 0.3290
            ],
            luminance_min: 50,        // 0.005 cd/m^2
            luminance_max: 10000000,  // 1000 cd/m^2
        }
    }

    /// Create metadata OBU.
    pub fn to_obu(&self) -> Obu {
        let mut payload = vec![MetadataType::HdrMdcv as u8];

        for (x, y) in &self.primaries {
            payload.extend_from_slice(&x.to_be_bytes());
            payload.extend_from_slice(&y.to_be_bytes());
        }
        payload.extend_from_slice(&self.luminance_max.to_be_bytes());
        payload.extend_from_slice(&self.luminance_min.to_be_bytes());

        Obu::new(ObuType::Metadata, payload)
    }
}

/// Parse LEB128 encoded unsigned integer.
pub fn parse_leb128(data: &[u8]) -> Result<(u64, usize)> {
    let mut value = 0u64;
    let mut bytes_read = 0;

    for (i, &byte) in data.iter().enumerate() {
        if i >= 8 {
            return Err(HwAccelError::Decode("LEB128 overflow".to_string()));
        }

        value |= ((byte & 0x7F) as u64) << (i * 7);
        bytes_read = i + 1;

        if (byte & 0x80) == 0 {
            break;
        }
    }

    Ok((value, bytes_read))
}

/// Encode unsigned integer as LEB128.
pub fn encode_leb128(mut value: u64) -> Vec<u8> {
    let mut result = Vec::new();

    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        if value != 0 {
            byte |= 0x80;
        }

        result.push(byte);

        if value == 0 {
            break;
        }
    }

    result
}

/// Calculate LEB128 encoded size.
pub fn leb128_size(value: u64) -> usize {
    if value == 0 {
        return 1;
    }

    let bits = 64 - value.leading_zeros();
    bits.div_ceil(7) as usize
}

/// Parse all OBUs from a byte buffer.
pub fn parse_obus(data: &[u8]) -> Result<Vec<Obu>> {
    let mut obus = Vec::new();
    let mut offset = 0;

    while offset < data.len() {
        let (obu, consumed) = Obu::parse(&data[offset..])?;
        obus.push(obu);
        offset += consumed;
    }

    Ok(obus)
}

/// Calculate frame size in bytes from OBU list.
pub fn calculate_frame_size(obus: &[Obu]) -> usize {
    obus.iter().map(|obu| obu.total_size()).sum()
}

/// Extract sequence header from OBU list.
pub fn find_sequence_header(obus: &[Obu]) -> Option<&Obu> {
    obus.iter()
        .find(|obu| obu.header.obu_type == ObuType::SequenceHeader)
}

/// Check if OBU list starts with temporal delimiter.
pub fn has_temporal_delimiter(obus: &[Obu]) -> bool {
    obus.first()
        .is_some_and(|obu| obu.header.obu_type == ObuType::TemporalDelimiter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obu_type() {
        assert_eq!(ObuType::SequenceHeader as u8, 1);
        assert_eq!(ObuType::from_u8(1), Some(ObuType::SequenceHeader));
        assert_eq!(ObuType::from_u8(100), None);

        assert!(ObuType::TileGroup.has_frame_data());
        assert!(!ObuType::SequenceHeader.has_frame_data());
    }

    #[test]
    fn test_obu_header_parse() {
        // Simple header: type=1 (sequence header), no extension, has size
        let data = [0x0A]; // 0b00001010 = type 1, no ext, has size
        let (header, size) = ObuHeader::parse(&data).unwrap();

        assert_eq!(header.obu_type, ObuType::SequenceHeader);
        assert!(!header.has_extension);
        assert!(header.has_size_field);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_obu_header_with_extension() {
        // Header with extension
        let data = [0x0E, 0x60]; // type 1, has ext, has size; temporal=3, spatial=0
        let (header, size) = ObuHeader::parse(&data).unwrap();

        assert_eq!(header.obu_type, ObuType::SequenceHeader);
        assert!(header.has_extension);
        assert_eq!(header.temporal_id, 3);
        assert_eq!(header.spatial_id, 0);
        assert_eq!(size, 2);
    }

    #[test]
    fn test_obu_header_write() {
        let header = ObuHeader::new(ObuType::TemporalDelimiter);
        let bytes = header.write();

        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0] & 0x78, (ObuType::TemporalDelimiter as u8) << 3);
    }

    #[test]
    fn test_leb128() {
        // Test encoding
        assert_eq!(encode_leb128(0), vec![0x00]);
        assert_eq!(encode_leb128(127), vec![0x7F]);
        assert_eq!(encode_leb128(128), vec![0x80, 0x01]);
        assert_eq!(encode_leb128(300), vec![0xAC, 0x02]);

        // Test parsing
        assert_eq!(parse_leb128(&[0x00]).unwrap(), (0, 1));
        assert_eq!(parse_leb128(&[0x7F]).unwrap(), (127, 1));
        assert_eq!(parse_leb128(&[0x80, 0x01]).unwrap(), (128, 2));
        assert_eq!(parse_leb128(&[0xAC, 0x02]).unwrap(), (300, 2));
    }

    #[test]
    fn test_leb128_size() {
        assert_eq!(leb128_size(0), 1);
        assert_eq!(leb128_size(127), 1);
        assert_eq!(leb128_size(128), 2);
        assert_eq!(leb128_size(16383), 2);
        assert_eq!(leb128_size(16384), 3);
    }

    #[test]
    fn test_temporal_delimiter() {
        let td = TemporalDelimiter::new();
        let bytes = td.write();

        // Should be small (header + size=0)
        assert!(bytes.len() <= 3);
    }

    #[test]
    fn test_sequence_header() {
        let seq = SequenceHeader::new(1920, 1080, Av1Profile::Main, Av1Level::L4_1);

        assert_eq!(seq.frame_dimensions(), (1920, 1080));
        assert_eq!(seq.profile, Av1Profile::Main);
        assert!(seq.enable_cdef);
    }

    #[test]
    fn test_frame_header() {
        let header = FrameHeader::keyframe(1920, 1080, 30);
        assert!(header.frame_type.is_key());
        assert!(header.show_frame);

        let inter = FrameHeader::inter_frame(1920, 1080, 35);
        assert!(!inter.frame_type.is_key());
    }

    #[test]
    fn test_color_config() {
        let sdr = ColorConfig::sdr(8);
        assert_eq!(sdr.bit_depth, 8);
        assert!(!sdr.color_range);

        let hdr = ColorConfig::hdr10();
        assert_eq!(hdr.bit_depth, 10);
        assert_eq!(hdr.transfer_characteristics, TransferCharacteristics::Smpte2084);
    }

    #[test]
    fn test_hdr_metadata() {
        let cll = HdrCll::new(1000, 400);
        let obu = cll.to_obu();
        assert_eq!(obu.header.obu_type, ObuType::Metadata);

        let mdcv = HdrMdcv::hdr10_standard();
        let obu = mdcv.to_obu();
        assert_eq!(obu.header.obu_type, ObuType::Metadata);
    }

    #[test]
    fn test_obu_roundtrip() {
        let original = Obu::new(ObuType::SequenceHeader, vec![1, 2, 3, 4, 5]);
        let bytes = original.write();
        let (parsed, _) = Obu::parse(&bytes).unwrap();

        assert_eq!(parsed.header.obu_type, original.header.obu_type);
        assert_eq!(parsed.payload, original.payload);
    }
}
