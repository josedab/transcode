//! HEVC/H.265 codec support for the transcode library.
//!
//! This crate provides a complete HEVC (High Efficiency Video Coding) implementation
//! following the ITU-T H.265 specification. HEVC offers approximately 50% better
//! compression efficiency compared to H.264/AVC while maintaining similar visual quality.
//!
//! # Features
//!
//! - **NAL Unit Parsing**: Complete NAL unit type support including VPS, SPS, PPS,
//!   IDR, TRAIL_N/R, TSA, STSA, RADL, RASL, BLA, CRA, and SEI units
//! - **Profile/Tier/Level**: Support for Main, Main10, Main Still Picture, and Range
//!   Extensions profiles with proper tier and level constraints
//! - **CABAC Entropy Coding**: Full Context-Adaptive Binary Arithmetic Coding support
//!   with all required context models and bypass modes
//! - **Transform/Quantization**: DCT-II and DST-VII transforms (4x4 to 32x32) with
//!   proper scaling lists and quantization
//! - **Intra Prediction**: All 35 intra prediction modes including Planar (0),
//!   DC (1), and Angular modes (2-34)
//! - **Inter Prediction**: Motion compensation with quarter-sample precision,
//!   bi-directional prediction, and weighted prediction
//! - **In-Loop Filtering**: Deblocking filter and Sample Adaptive Offset (SAO)
//!   filtering with both band offset and edge offset modes
//!
//! # Architecture
//!
//! HEVC uses a hierarchical block structure:
//! - **CTU (Coding Tree Unit)**: Largest coding block (64x64, 32x32, or 16x16)
//! - **CU (Coding Unit)**: Variable size from 8x8 to CTU size
//! - **PU (Prediction Unit)**: Prediction partitioning within a CU
//! - **TU (Transform Unit)**: Transform partitioning within a CU
//!
//! # Example
//!
//! ```rust,ignore
//! use transcode_hevc::{HevcDecoder, HevcEncoder, HevcEncoderConfig, HevcPreset};
//!
//! // Create a decoder
//! let mut decoder = HevcDecoder::new()?;
//!
//! // Decode NAL units
//! for nal_data in nal_units {
//!     if let Some(frame) = decoder.decode_nal(&nal_data)? {
//!         // Process decoded frame
//!     }
//! }
//!
//! // Create an encoder with configuration
//! let config = HevcEncoderConfig::with_preset(HevcPreset::Medium)
//!     .with_resolution(1920, 1080)
//!     .with_framerate(30, 1);
//!
//! let mut encoder = HevcEncoder::new(config)?;
//!
//! // Encode frames
//! let packets = encoder.encode(&frame)?;
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_div_ceil)]

pub mod cabac;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod nal;
pub mod transform;

// Re-export main error types
pub use error::{HevcError, HevcLevel, HevcProfile, HevcTier, NalError};

// Re-export CABAC types
pub use cabac::{CabacContext, CabacDecoder, CabacEncoder};

// Re-export NAL types
pub use nal::{
    NalUnitHeader, NalUnitType, Pps, ProfileTierLevel, ShortTermRefPicSet,
    SliceSegmentHeader, SliceType, Sps, Vps,
};

// Re-export transform types
pub use transform::{HevcQuantizer, HevcTransform, QuantParams, ScalingList, TransformSize};

// Re-export decoder types
pub use decoder::{
    CodingTreeUnit, CodingUnit, HevcDecoder, HevcDecoderConfig,
    HevcDecoderInfo, IntraMode, PartMode, PredictionMode, SaoParams,
    TransformUnit,
};

// Re-export encoder types
pub use encoder::{
    FrameStats, HevcEncoder, HevcEncoderConfig, HevcPreset, MotionVector, RateControlMode,
};

/// HEVC codec information and capabilities.
#[derive(Debug, Clone)]
pub struct HevcInfo {
    /// Supported profiles
    pub profiles: Vec<HevcProfile>,
    /// Supported tiers
    pub tiers: Vec<HevcTier>,
    /// Supported levels
    pub levels: Vec<HevcLevel>,
    /// Maximum supported CTU size
    pub max_ctu_size: u32,
    /// Minimum supported CU size
    pub min_cu_size: u32,
    /// Maximum transform size
    pub max_transform_size: u32,
    /// Minimum transform size
    pub min_transform_size: u32,
    /// Maximum intra prediction depth
    pub max_intra_depth: u32,
    /// Maximum inter prediction depth
    pub max_inter_depth: u32,
    /// SAO support
    pub sao_enabled: bool,
    /// Deblocking filter support
    pub deblocking_enabled: bool,
    /// Weighted prediction support
    pub weighted_pred: bool,
    /// B-frame support
    pub b_frames: bool,
}

impl Default for HevcInfo {
    fn default() -> Self {
        Self {
            profiles: vec![HevcProfile::Main, HevcProfile::Main10],
            tiers: vec![HevcTier::Main, HevcTier::High],
            levels: vec![
                HevcLevel::L1,
                HevcLevel::L2,
                HevcLevel::L2_1,
                HevcLevel::L3,
                HevcLevel::L3_1,
                HevcLevel::L4,
                HevcLevel::L4_1,
                HevcLevel::L5,
                HevcLevel::L5_1,
                HevcLevel::L5_2,
                HevcLevel::L6,
                HevcLevel::L6_1,
                HevcLevel::L6_2,
            ],
            max_ctu_size: 64,
            min_cu_size: 8,
            max_transform_size: 32,
            min_transform_size: 4,
            max_intra_depth: 3,
            max_inter_depth: 3,
            sao_enabled: true,
            deblocking_enabled: true,
            weighted_pred: true,
            b_frames: true,
        }
    }
}

impl HevcInfo {
    /// Creates new HEVC codec info with default capabilities.
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks if the given profile is supported.
    pub fn supports_profile(&self, profile: HevcProfile) -> bool {
        self.profiles.contains(&profile)
    }

    /// Checks if the given level is supported.
    pub fn supports_level(&self, level: HevcLevel) -> bool {
        self.levels.contains(&level)
    }

    /// Gets the maximum supported resolution for a given level.
    pub fn max_resolution_for_level(level: HevcLevel) -> (u32, u32) {
        match level.level_idc {
            30 => (176, 144),        // Level 1
            60 => (352, 288),        // Level 2
            63 => (640, 360),        // Level 2.1
            90 => (960, 540),        // Level 3
            93 => (1280, 720),       // Level 3.1
            120 | 123 => (2048, 1080), // Level 4, 4.1
            150 | 153 | 156 => (4096, 2160), // Level 5, 5.1, 5.2
            180 | 183 | 186 => (8192, 4320), // Level 6, 6.1, 6.2
            _ => (176, 144),
        }
    }

    /// Gets the maximum luma sample rate for a given level.
    pub fn max_luma_sample_rate(level: HevcLevel) -> u64 {
        level.max_luma_samples_per_second()
    }

    /// Gets the maximum bitrate for a given level and tier.
    pub fn max_bitrate(level: HevcLevel, tier: HevcTier) -> u64 {
        level.max_bitrate(tier)
    }
}

/// Codec FourCC codes for HEVC.
pub mod fourcc {
    /// HEVC in MP4/MOV (ISO BMFF)
    pub const HVC1: [u8; 4] = *b"hvc1";
    /// HEVC in MP4/MOV with parameter sets in sample
    pub const HEV1: [u8; 4] = *b"hev1";
    /// HEVC in Matroska
    pub const V_MPEGH_ISO_HEVC: &str = "V_MPEGH/ISO/HEVC";
}

/// HEVC start code constants.
pub mod start_codes {
    /// 3-byte start code prefix
    pub const START_CODE_3: [u8; 3] = [0x00, 0x00, 0x01];
    /// 4-byte start code prefix
    pub const START_CODE_4: [u8; 4] = [0x00, 0x00, 0x00, 0x01];
    /// Emulation prevention byte
    pub const EMULATION_PREVENTION_BYTE: u8 = 0x03;
}

/// Parse Annex B byte stream into NAL units.
///
/// This function extracts individual NAL units from an Annex B formatted
/// byte stream by detecting start codes and separating the NAL unit data.
///
/// # Arguments
///
/// * `data` - Annex B formatted byte stream
///
/// # Returns
///
/// Vector of tuples containing (NalUnitHeader, NAL unit data)
pub fn parse_annexb_stream(data: &[u8]) -> Vec<(NalUnitHeader, Vec<u8>)> {
    nal::parse_annexb_nal_units(data)
}

/// Extract NAL unit type from NAL unit header.
///
/// # Arguments
///
/// * `nal_data` - NAL unit data (first byte is header)
///
/// # Returns
///
/// NAL unit type
pub fn get_nal_unit_type(nal_data: &[u8]) -> Option<NalUnitType> {
    if nal_data.len() < 2 {
        return None;
    }

    let type_value = (nal_data[0] >> 1) & 0x3F;
    Some(NalUnitType::from_raw(type_value))
}

/// Check if NAL unit is a parameter set (VPS, SPS, or PPS).
pub fn is_parameter_set(nal_type: NalUnitType) -> bool {
    matches!(
        nal_type,
        NalUnitType::VpsNut | NalUnitType::SpsNut | NalUnitType::PpsNut
    )
}

/// Check if NAL unit is an IDR picture.
pub fn is_idr_picture(nal_type: NalUnitType) -> bool {
    matches!(
        nal_type,
        NalUnitType::IdrWRadl | NalUnitType::IdrNLp
    )
}

/// Check if NAL unit is a RASL picture.
pub fn is_rasl_picture(nal_type: NalUnitType) -> bool {
    matches!(
        nal_type,
        NalUnitType::RaslN | NalUnitType::RaslR
    )
}

/// Check if NAL unit is a RADL picture.
pub fn is_radl_picture(nal_type: NalUnitType) -> bool {
    matches!(
        nal_type,
        NalUnitType::RadlN | NalUnitType::RadlR
    )
}

/// Check if NAL unit is a random access point (RAP).
pub fn is_random_access_point(nal_type: NalUnitType) -> bool {
    matches!(
        nal_type,
        NalUnitType::IdrWRadl
            | NalUnitType::IdrNLp
            | NalUnitType::CraNut
            | NalUnitType::BlaNLp
            | NalUnitType::BlaWRadl
            | NalUnitType::BlaWLp
    )
}

/// Check if NAL unit is a leading picture.
pub fn is_leading_picture(nal_type: NalUnitType) -> bool {
    is_rasl_picture(nal_type) || is_radl_picture(nal_type)
}

/// Check if NAL unit is a trailing picture.
pub fn is_trailing_picture(nal_type: NalUnitType) -> bool {
    matches!(
        nal_type,
        NalUnitType::TrailN | NalUnitType::TrailR
    )
}

/// Check if NAL unit is a VCL (Video Coding Layer) unit.
pub fn is_vcl_nal(nal_type: NalUnitType) -> bool {
    let type_val = nal_type.to_raw();
    type_val <= 31
}

/// Calculate CTU (Coding Tree Unit) count for a frame.
///
/// # Arguments
///
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `ctu_size` - CTU size (typically 64, 32, or 16)
///
/// # Returns
///
/// Total number of CTUs in the frame
pub fn calculate_ctu_count(width: u32, height: u32, ctu_size: u32) -> u32 {
    let ctus_x = (width + ctu_size - 1) / ctu_size;
    let ctus_y = (height + ctu_size - 1) / ctu_size;
    ctus_x * ctus_y
}

/// Calculate the minimum level required for given parameters.
///
/// # Arguments
///
/// * `width` - Frame width
/// * `height` - Frame height
/// * `framerate` - Frame rate
/// * `bitrate` - Target bitrate in bits per second
/// * `tier` - Tier (Main or High)
///
/// # Returns
///
/// Minimum level required, if any level can support it
pub fn calculate_min_level(
    width: u32,
    height: u32,
    framerate: f64,
    bitrate: u64,
    tier: HevcTier,
) -> Option<HevcLevel> {
    let sample_rate = (width as f64) * (height as f64) * framerate;

    let levels = [
        HevcLevel::L1,
        HevcLevel::L2,
        HevcLevel::L2_1,
        HevcLevel::L3,
        HevcLevel::L3_1,
        HevcLevel::L4,
        HevcLevel::L4_1,
        HevcLevel::L5,
        HevcLevel::L5_1,
        HevcLevel::L5_2,
        HevcLevel::L6,
        HevcLevel::L6_1,
        HevcLevel::L6_2,
    ];

    for level in levels {
        let max_rate = level.max_luma_samples_per_second() as f64;
        let max_bitrate = level.max_bitrate(tier);

        if sample_rate <= max_rate && bitrate <= max_bitrate {
            return Some(level);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hevc_info_default() {
        let info = HevcInfo::default();
        assert!(info.profiles.contains(&HevcProfile::Main));
        assert!(info.profiles.contains(&HevcProfile::Main10));
        assert_eq!(info.max_ctu_size, 64);
        assert_eq!(info.min_cu_size, 8);
        assert!(info.sao_enabled);
        assert!(info.deblocking_enabled);
    }

    #[test]
    fn test_supports_profile() {
        let info = HevcInfo::new();
        assert!(info.supports_profile(HevcProfile::Main));
        assert!(info.supports_profile(HevcProfile::Main10));
    }

    #[test]
    fn test_supports_level() {
        let info = HevcInfo::new();
        assert!(info.supports_level(HevcLevel::L4));
        assert!(info.supports_level(HevcLevel::L5_1));
    }

    #[test]
    fn test_max_resolution_for_level() {
        assert_eq!(HevcInfo::max_resolution_for_level(HevcLevel::L3_1), (1280, 720));
        assert_eq!(HevcInfo::max_resolution_for_level(HevcLevel::L4), (2048, 1080));
        assert_eq!(HevcInfo::max_resolution_for_level(HevcLevel::L5_1), (4096, 2160));
    }

    #[test]
    fn test_get_nal_unit_type() {
        // VPS NAL unit (type 32)
        let vps_header = [0x40, 0x01]; // (32 << 1) | 0, 0x01
        assert_eq!(get_nal_unit_type(&vps_header), Some(NalUnitType::VpsNut));

        // SPS NAL unit (type 33)
        let sps_header = [0x42, 0x01]; // (33 << 1) | 0, 0x01
        assert_eq!(get_nal_unit_type(&sps_header), Some(NalUnitType::SpsNut));

        // PPS NAL unit (type 34)
        let pps_header = [0x44, 0x01]; // (34 << 1) | 0, 0x01
        assert_eq!(get_nal_unit_type(&pps_header), Some(NalUnitType::PpsNut));

        // IDR NAL unit (type 19)
        let idr_header = [0x26, 0x01]; // (19 << 1) | 0, 0x01
        assert_eq!(get_nal_unit_type(&idr_header), Some(NalUnitType::IdrWRadl));
    }

    #[test]
    fn test_is_parameter_set() {
        assert!(is_parameter_set(NalUnitType::VpsNut));
        assert!(is_parameter_set(NalUnitType::SpsNut));
        assert!(is_parameter_set(NalUnitType::PpsNut));
        assert!(!is_parameter_set(NalUnitType::IdrWRadl));
        assert!(!is_parameter_set(NalUnitType::TrailR));
    }

    #[test]
    fn test_is_idr_picture() {
        assert!(is_idr_picture(NalUnitType::IdrWRadl));
        assert!(is_idr_picture(NalUnitType::IdrNLp));
        assert!(!is_idr_picture(NalUnitType::CraNut));
        assert!(!is_idr_picture(NalUnitType::TrailR));
    }

    #[test]
    fn test_is_random_access_point() {
        assert!(is_random_access_point(NalUnitType::IdrWRadl));
        assert!(is_random_access_point(NalUnitType::IdrNLp));
        assert!(is_random_access_point(NalUnitType::CraNut));
        assert!(is_random_access_point(NalUnitType::BlaNLp));
        assert!(!is_random_access_point(NalUnitType::TrailR));
    }

    #[test]
    fn test_is_leading_picture() {
        assert!(is_leading_picture(NalUnitType::RaslN));
        assert!(is_leading_picture(NalUnitType::RaslR));
        assert!(is_leading_picture(NalUnitType::RadlN));
        assert!(is_leading_picture(NalUnitType::RadlR));
        assert!(!is_leading_picture(NalUnitType::TrailR));
    }

    #[test]
    fn test_is_trailing_picture() {
        assert!(is_trailing_picture(NalUnitType::TrailN));
        assert!(is_trailing_picture(NalUnitType::TrailR));
        assert!(!is_trailing_picture(NalUnitType::IdrWRadl));
    }

    #[test]
    fn test_is_vcl_nal() {
        assert!(is_vcl_nal(NalUnitType::TrailN));
        assert!(is_vcl_nal(NalUnitType::TrailR));
        assert!(is_vcl_nal(NalUnitType::IdrWRadl));
        assert!(is_vcl_nal(NalUnitType::CraNut));
        assert!(!is_vcl_nal(NalUnitType::VpsNut));
        assert!(!is_vcl_nal(NalUnitType::SpsNut));
        assert!(!is_vcl_nal(NalUnitType::PpsNut));
    }

    #[test]
    fn test_calculate_ctu_count() {
        // 1920x1080 with 64x64 CTUs
        assert_eq!(calculate_ctu_count(1920, 1080, 64), 30 * 17);

        // 1280x720 with 64x64 CTUs
        assert_eq!(calculate_ctu_count(1280, 720, 64), 20 * 12);

        // 3840x2160 with 64x64 CTUs
        assert_eq!(calculate_ctu_count(3840, 2160, 64), 60 * 34);

        // 1920x1080 with 32x32 CTUs
        assert_eq!(calculate_ctu_count(1920, 1080, 32), 60 * 34);
    }

    #[test]
    fn test_calculate_min_level() {
        // 1080p30 at 8Mbps should be Level 4
        let level = calculate_min_level(1920, 1080, 30.0, 8_000_000_u64, HevcTier::Main);
        assert!(level.is_some());

        // 4K60 at 20Mbps should be Level 5.1 or higher
        let level = calculate_min_level(3840, 2160, 60.0, 20_000_000_u64, HevcTier::Main);
        assert!(level.is_some());
    }

    #[test]
    fn test_parse_annexb_empty() {
        let data: &[u8] = &[];
        let units = parse_annexb_stream(data);
        assert!(units.is_empty());
    }

    #[test]
    fn test_parse_annexb_single_nal() {
        // Single NAL unit with 4-byte start code
        // Format: start code (4 bytes) + header (2 bytes: 0x40, 0x01) + RBSP (2 bytes: 0x0c, 0x01)
        let data = [0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01];
        let units = parse_annexb_stream(&data);
        assert_eq!(units.len(), 1);
        // The RBSP is the payload after the 2-byte header
        assert_eq!(units[0].1, vec![0x0c, 0x01]);
        // Check the header type is VPS (type 32 = 0x40 >> 1)
        assert_eq!(units[0].0.nal_unit_type, NalUnitType::VpsNut);
    }

    #[test]
    fn test_parse_annexb_multiple_nals() {
        // Two NAL units with 4-byte start codes
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, // VPS
            0x00, 0x00, 0x00, 0x01, 0x42, 0x01, // SPS
        ];
        let units = parse_annexb_stream(&data);
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn test_fourcc_codes() {
        assert_eq!(&fourcc::HVC1, b"hvc1");
        assert_eq!(&fourcc::HEV1, b"hev1");
    }

    #[test]
    fn test_start_codes() {
        assert_eq!(start_codes::START_CODE_3, [0x00, 0x00, 0x01]);
        assert_eq!(start_codes::START_CODE_4, [0x00, 0x00, 0x00, 0x01]);
        assert_eq!(start_codes::EMULATION_PREVENTION_BYTE, 0x03);
    }
}
