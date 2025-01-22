//! VVC/H.266 codec support for the transcode library.
//!
//! This crate provides a complete VVC (Versatile Video Coding) implementation
//! following the ITU-T H.266 specification. VVC offers approximately 50% better
//! compression efficiency compared to HEVC/H.265 while maintaining similar visual quality.
//!
//! # Features
//!
//! - **NAL Unit Parsing**: Complete NAL unit type support including VPS, SPS, PPS,
//!   Picture Header, GDR, IDR, TRAIL, STSA, RADL, RASL, and SEI units
//! - **Profile/Tier/Level**: Support for Main10, Main10 4:4:4, Multilayer, and Still
//!   Picture profiles with proper tier and level constraints
//! - **CABAC Entropy Coding**: Full Context-Adaptive Binary Arithmetic Coding support
//!   with VVC-specific context models and bypass modes
//! - **Transform/Quantization**: DCT-II, DST-VII, and DCT-VIII transforms (4x4 to 64x64)
//!   with MTS (Multiple Transform Selection) and LFNST (Low-Frequency Non-Separable Transform)
//! - **Intra Prediction**: All 67 intra prediction modes including Planar (0),
//!   DC (1), and Angular modes (2-66) with MIP (Matrix Intra Prediction), ISP, and BDPCM
//! - **Inter Prediction**: Advanced motion compensation with MMVD, SMVD, Affine,
//!   GPM (Geometric Partition Mode), CIIP, BDOF, DMVR, and BCW
//! - **In-Loop Filtering**: Deblocking filter, SAO, ALF (Adaptive Loop Filter) with
//!   CCALF (Cross-Component ALF), and LMCS (Luma Mapping with Chroma Scaling)
//!
//! # Architecture
//!
//! VVC uses a hierarchical block structure with QTBT+MTT partitioning:
//! - **CTU (Coding Tree Unit)**: Largest coding block (up to 128x128)
//! - **CU (Coding Unit)**: Variable size from 4x4 to CTU size, using quad-tree,
//!   binary-tree, or ternary-tree splits
//! - **TU (Transform Unit)**: Transform partitioning within a CU with ISP support
//!
//! # Example
//!
//! ```rust,ignore
//! use transcode_vvc::{VvcDecoder, VvcEncoder, VvcEncoderConfig, VvcPreset};
//!
//! // Create a decoder
//! let mut decoder = VvcDecoder::new()?;
//!
//! // Decode NAL units
//! for nal_data in nal_units {
//!     if let Some(frame) = decoder.decode_nal(&nal_data)? {
//!         // Process decoded frame
//!     }
//! }
//!
//! // Create an encoder with configuration
//! let config = VvcEncoderConfig::with_preset(VvcPreset::Medium)
//!     .with_resolution(1920, 1080)
//!     .with_framerate(30, 1);
//!
//! let mut encoder = VvcEncoder::new(config)?;
//!
//! // Encode frames
//! let packets = encoder.encode(&frame)?;
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_div_ceil)]

pub mod decoder;
pub mod encoder;
pub mod error;
pub mod motion;
pub mod nal;
pub mod syntax;
pub mod transform;

// Re-export main error types
pub use error::{Result, VvcError, VvcLevel, VvcProfile, VvcTier};

// Re-export NAL types
pub use nal::{
    NalUnitHeader, NalUnitType, PictureHeader, Pps, ProfileTierLevel, SeiMessage, SeiType,
    SliceHeader, SliceType, Sps, Vps,
};

// Re-export syntax types
pub use syntax::{
    AlfCtuParams, BdpcmMode, CodingTreeUnit, CodingUnit, IntraChromaMode, IntraMode, IspMode,
    LmcsData, MergeMode, MipConfig, MipSizeClass, MotionVector, PredMode, SaoParams, SplitMode,
    TransformSizeClass, TransformUnit,
};

// Re-export decoder types
pub use decoder::{VvcDecoder, VvcDecoderConfig};

// Re-export encoder types
pub use encoder::{
    FrameStats, GopStructure, RateControlMode, TemporalLayerConfig, VvcEncoder, VvcEncoderConfig,
    VvcPreset,
};

// Re-export transform types
pub use transform::{
    LfnstIndex, LfnstTransform, QuantParams, TransformKernel, TransformSize, TransformType,
    VvcTransforms,
};

// Re-export motion compensation types
pub use motion::{
    BcwIndex, DmvrProcessor, MotionCompensator, MvPredictor, CHROMA_FILTER, LUMA_FILTER,
};

/// VVC codec information and capabilities.
#[derive(Debug, Clone)]
pub struct VvcInfo {
    /// Supported profiles
    pub profiles: Vec<VvcProfile>,
    /// Supported tiers
    pub tiers: Vec<VvcTier>,
    /// Supported levels
    pub levels: Vec<VvcLevel>,
    /// Maximum supported CTU size
    pub max_ctu_size: u32,
    /// Minimum supported CU size
    pub min_cu_size: u32,
    /// Maximum transform size
    pub max_transform_size: u32,
    /// Minimum transform size
    pub min_transform_size: u32,
    /// Maximum MTT depth for intra slices
    pub max_mtt_depth_intra: u32,
    /// Maximum MTT depth for inter slices
    pub max_mtt_depth_inter: u32,
    /// SAO support
    pub sao_enabled: bool,
    /// ALF support
    pub alf_enabled: bool,
    /// CCALF support
    pub ccalf_enabled: bool,
    /// LMCS support
    pub lmcs_enabled: bool,
    /// Deblocking filter support
    pub deblocking_enabled: bool,
    /// Weighted prediction support
    pub weighted_pred: bool,
    /// B-frame support
    pub b_frames: bool,
    /// MIP (Matrix Intra Prediction) support
    pub mip_enabled: bool,
    /// ISP (Intra Sub-Partitions) support
    pub isp_enabled: bool,
    /// LFNST support
    pub lfnst_enabled: bool,
    /// Affine motion compensation support
    pub affine_enabled: bool,
    /// GPM (Geometric Partition Mode) support
    pub gpm_enabled: bool,
}

impl Default for VvcInfo {
    fn default() -> Self {
        Self {
            profiles: vec![VvcProfile::Main10, VvcProfile::Main10_444],
            tiers: vec![VvcTier::Main, VvcTier::High],
            levels: vec![
                VvcLevel::L1_0,
                VvcLevel::L2_0,
                VvcLevel::L2_1,
                VvcLevel::L3_0,
                VvcLevel::L3_1,
                VvcLevel::L4_0,
                VvcLevel::L4_1,
                VvcLevel::L5_0,
                VvcLevel::L5_1,
                VvcLevel::L5_2,
                VvcLevel::L6_0,
                VvcLevel::L6_1,
                VvcLevel::L6_2,
                VvcLevel::L6_3,
            ],
            max_ctu_size: 128,
            min_cu_size: 4,
            max_transform_size: 64,
            min_transform_size: 4,
            max_mtt_depth_intra: 4,
            max_mtt_depth_inter: 4,
            sao_enabled: true,
            alf_enabled: true,
            ccalf_enabled: true,
            lmcs_enabled: true,
            deblocking_enabled: true,
            weighted_pred: true,
            b_frames: true,
            mip_enabled: true,
            isp_enabled: true,
            lfnst_enabled: true,
            affine_enabled: true,
            gpm_enabled: true,
        }
    }
}

impl VvcInfo {
    /// Creates new VVC codec info with default capabilities.
    pub fn new() -> Self {
        Self::default()
    }

    /// Checks if the given profile is supported.
    pub fn supports_profile(&self, profile: VvcProfile) -> bool {
        self.profiles.contains(&profile)
    }

    /// Checks if the given level is supported.
    pub fn supports_level(&self, level: VvcLevel) -> bool {
        self.levels.contains(&level)
    }

    /// Gets the maximum supported resolution for a given level.
    pub fn max_resolution_for_level(level: VvcLevel) -> (u32, u32) {
        // Resolution limits based on max luma picture size
        match level {
            VvcLevel::L1_0 => (176, 144),           // QCIF
            VvcLevel::L2_0 => (352, 288),           // CIF
            VvcLevel::L2_1 => (640, 360),           // 360p
            VvcLevel::L3_0 => (960, 540),           // 540p
            VvcLevel::L3_1 => (1280, 720),          // 720p
            VvcLevel::L4_0 | VvcLevel::L4_1 => (2048, 1080), // 1080p
            VvcLevel::L5_0 | VvcLevel::L5_1 | VvcLevel::L5_2 => (4096, 2160), // 4K
            VvcLevel::L6_0 | VvcLevel::L6_1 | VvcLevel::L6_2 => (8192, 4320), // 8K
            VvcLevel::L6_3 => (16384, 8640),        // 16K
        }
    }

    /// Gets the maximum luma sample rate for a given level.
    pub fn max_luma_sample_rate(level: VvcLevel) -> u64 {
        level.max_luma_ps()
    }

    /// Gets the maximum bitrate for a given level and tier.
    pub fn max_bitrate(level: VvcLevel, tier: VvcTier) -> u32 {
        match tier {
            VvcTier::Main => level.max_br_main(),
            VvcTier::High => level.max_br_high(),
        }
    }
}

/// VVC configuration for encoding/decoding.
#[derive(Debug, Clone)]
pub struct VvcConfig {
    /// Video profile
    pub profile: VvcProfile,
    /// Video tier
    pub tier: VvcTier,
    /// Video level
    pub level: VvcLevel,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Bit depth (8, 10, or 12)
    pub bit_depth: u8,
    /// Chroma format (0=mono, 1=4:2:0, 2=4:2:2, 3=4:4:4)
    pub chroma_format: u8,
    /// CTU size (32, 64, or 128)
    pub ctu_size: u32,
    /// Maximum MTT depth
    pub max_mtt_depth: u32,
    /// Enable SAO
    pub sao_enabled: bool,
    /// Enable ALF
    pub alf_enabled: bool,
    /// Enable CCALF
    pub ccalf_enabled: bool,
    /// Enable LMCS
    pub lmcs_enabled: bool,
    /// Enable deblocking
    pub deblocking_enabled: bool,
    /// Enable MIP
    pub mip_enabled: bool,
    /// Enable ISP
    pub isp_enabled: bool,
    /// Enable LFNST
    pub lfnst_enabled: bool,
    /// Enable MTS
    pub mts_enabled: bool,
    /// Enable affine motion compensation
    pub affine_enabled: bool,
    /// Enable GPM
    pub gpm_enabled: bool,
    /// Enable CIIP
    pub ciip_enabled: bool,
    /// Enable BDOF
    pub bdof_enabled: bool,
    /// Enable DMVR
    pub dmvr_enabled: bool,
}

impl Default for VvcConfig {
    fn default() -> Self {
        Self {
            profile: VvcProfile::Main10,
            tier: VvcTier::Main,
            level: VvcLevel::L4_1,
            width: 1920,
            height: 1080,
            bit_depth: 10,
            chroma_format: 1, // 4:2:0
            ctu_size: 128,
            max_mtt_depth: 4,
            sao_enabled: true,
            alf_enabled: true,
            ccalf_enabled: true,
            lmcs_enabled: false,
            deblocking_enabled: true,
            mip_enabled: true,
            isp_enabled: true,
            lfnst_enabled: true,
            mts_enabled: true,
            affine_enabled: true,
            gpm_enabled: true,
            ciip_enabled: true,
            bdof_enabled: true,
            dmvr_enabled: true,
        }
    }
}

impl VvcConfig {
    /// Creates a new VVC configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a configuration for the specified resolution.
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Sets the profile.
    pub fn with_profile(mut self, profile: VvcProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Sets the tier.
    pub fn with_tier(mut self, tier: VvcTier) -> Self {
        self.tier = tier;
        self
    }

    /// Sets the level.
    pub fn with_level(mut self, level: VvcLevel) -> Self {
        self.level = level;
        self
    }

    /// Sets the bit depth.
    pub fn with_bit_depth(mut self, bit_depth: u8) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Sets the chroma format.
    pub fn with_chroma_format(mut self, chroma_format: u8) -> Self {
        self.chroma_format = chroma_format;
        self
    }

    /// Sets the CTU size.
    pub fn with_ctu_size(mut self, ctu_size: u32) -> Self {
        self.ctu_size = ctu_size;
        self
    }

    /// Enables or disables all VVC tools.
    pub fn with_all_tools(mut self, enabled: bool) -> Self {
        self.sao_enabled = enabled;
        self.alf_enabled = enabled;
        self.ccalf_enabled = enabled;
        self.lmcs_enabled = enabled;
        self.mip_enabled = enabled;
        self.isp_enabled = enabled;
        self.lfnst_enabled = enabled;
        self.mts_enabled = enabled;
        self.affine_enabled = enabled;
        self.gpm_enabled = enabled;
        self.ciip_enabled = enabled;
        self.bdof_enabled = enabled;
        self.dmvr_enabled = enabled;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<()> {
        // Check resolution limits
        let (max_w, max_h) = VvcInfo::max_resolution_for_level(self.level);
        if self.width > max_w || self.height > max_h {
            return Err(VvcError::EncoderConfig(format!(
                "Resolution {}x{} exceeds level {} limit of {}x{}",
                self.width, self.height, self.level.level_idc(), max_w, max_h
            )));
        }

        // Check bit depth
        if self.bit_depth != 8 && self.bit_depth != 10 && self.bit_depth != 12 {
            return Err(VvcError::EncoderConfig(format!(
                "Invalid bit depth: {}. Must be 8, 10, or 12",
                self.bit_depth
            )));
        }

        // Check CTU size
        if self.ctu_size != 32 && self.ctu_size != 64 && self.ctu_size != 128 {
            return Err(VvcError::EncoderConfig(format!(
                "Invalid CTU size: {}. Must be 32, 64, or 128",
                self.ctu_size
            )));
        }

        // Check profile constraints
        if self.profile.is_444() && self.chroma_format != 3 {
            return Err(VvcError::EncoderConfig(
                "4:4:4 profile requires chroma_format=3".to_string(),
            ));
        }

        Ok(())
    }
}

/// Codec FourCC codes for VVC.
pub mod fourcc {
    /// VVC in MP4/MOV (ISO BMFF)
    pub const VVC1: [u8; 4] = *b"vvc1";
    /// VVC in MP4/MOV with parameter sets in sample
    pub const VVI1: [u8; 4] = *b"vvi1";
    /// VVC in Matroska
    pub const V_MPEGV_ISO_VVC: &str = "V_MPEGV/ISO/VVC";
}

/// VVC start code constants.
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
/// * `nal_data` - NAL unit data (first 2 bytes are header)
///
/// # Returns
///
/// NAL unit type
pub fn get_nal_unit_type(nal_data: &[u8]) -> Option<NalUnitType> {
    if nal_data.len() < 2 {
        return None;
    }

    // VVC NAL unit header is 2 bytes
    // Byte 0: forbidden_zero_bit (1) + nuh_reserved_zero_bit (1) + nuh_layer_id (6)
    // Byte 1: nal_unit_type (5) + nuh_temporal_id_plus1 (3)
    let type_value = (nal_data[1] >> 3) & 0x1F;
    Some(NalUnitType::from_raw(type_value))
}

/// Check if NAL unit is a parameter set (VPS, SPS, PPS, or APS).
pub fn is_parameter_set(nal_type: NalUnitType) -> bool {
    matches!(
        nal_type,
        NalUnitType::VpsNut
            | NalUnitType::SpsNut
            | NalUnitType::PpsNut
            | NalUnitType::ApsPrefix
            | NalUnitType::ApsNut
    )
}

/// Check if NAL unit is a Picture Header.
pub fn is_picture_header(nal_type: NalUnitType) -> bool {
    matches!(nal_type, NalUnitType::PhNut)
}

/// Check if NAL unit is an IDR picture.
pub fn is_idr_picture(nal_type: NalUnitType) -> bool {
    matches!(
        nal_type,
        NalUnitType::IdrWRadl | NalUnitType::IdrNLp
    )
}

/// Check if NAL unit is a GDR (Gradual Decoding Refresh) picture.
pub fn is_gdr_picture(nal_type: NalUnitType) -> bool {
    matches!(nal_type, NalUnitType::GdrNut)
}

/// Check if NAL unit is a CRA (Clean Random Access) picture.
pub fn is_cra_picture(nal_type: NalUnitType) -> bool {
    matches!(nal_type, NalUnitType::CraNut)
}

/// Check if NAL unit is a RASL picture.
pub fn is_rasl_picture(nal_type: NalUnitType) -> bool {
    matches!(nal_type, NalUnitType::RaslNut)
}

/// Check if NAL unit is a RADL picture.
pub fn is_radl_picture(nal_type: NalUnitType) -> bool {
    matches!(nal_type, NalUnitType::RadlNut)
}

/// Check if NAL unit is a random access point (RAP).
pub fn is_random_access_point(nal_type: NalUnitType) -> bool {
    matches!(
        nal_type,
        NalUnitType::IdrWRadl
            | NalUnitType::IdrNLp
            | NalUnitType::CraNut
            | NalUnitType::GdrNut
    )
}

/// Check if NAL unit is a leading picture.
pub fn is_leading_picture(nal_type: NalUnitType) -> bool {
    is_rasl_picture(nal_type) || is_radl_picture(nal_type)
}

/// Check if NAL unit is a trailing picture.
pub fn is_trailing_picture(nal_type: NalUnitType) -> bool {
    matches!(nal_type, NalUnitType::TrailNut)
}

/// Check if NAL unit is a STSA picture.
pub fn is_stsa_picture(nal_type: NalUnitType) -> bool {
    matches!(nal_type, NalUnitType::StsaNut)
}

/// Check if NAL unit is a VCL (Video Coding Layer) unit.
pub fn is_vcl_nal(nal_type: NalUnitType) -> bool {
    let type_val = nal_type.to_raw();
    type_val <= 12
}

/// Calculate CTU (Coding Tree Unit) count for a frame.
///
/// # Arguments
///
/// * `width` - Frame width in pixels
/// * `height` - Frame height in pixels
/// * `ctu_size` - CTU size (typically 128, 64, or 32)
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
/// * `bitrate` - Target bitrate in kbps
/// * `tier` - Tier (Main or High)
///
/// # Returns
///
/// Minimum level required, if any level can support it
pub fn calculate_min_level(
    width: u32,
    height: u32,
    framerate: f64,
    bitrate: u32,
    tier: VvcTier,
) -> Option<VvcLevel> {
    let luma_samples = (width as u64) * (height as u64);
    let sample_rate = (luma_samples as f64) * framerate;

    let levels = [
        VvcLevel::L1_0,
        VvcLevel::L2_0,
        VvcLevel::L2_1,
        VvcLevel::L3_0,
        VvcLevel::L3_1,
        VvcLevel::L4_0,
        VvcLevel::L4_1,
        VvcLevel::L5_0,
        VvcLevel::L5_1,
        VvcLevel::L5_2,
        VvcLevel::L6_0,
        VvcLevel::L6_1,
        VvcLevel::L6_2,
        VvcLevel::L6_3,
    ];

    for level in levels {
        let max_luma_ps = level.max_luma_ps();
        let max_rate = max_luma_ps as f64 * 60.0; // Approximate max sample rate
        let max_bitrate = match tier {
            VvcTier::Main => level.max_br_main(),
            VvcTier::High => level.max_br_high(),
        };

        // Check if level supports the resolution and sample rate
        if luma_samples <= max_luma_ps && sample_rate <= max_rate && bitrate <= max_bitrate {
            return Some(level);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vvc_info_default() {
        let info = VvcInfo::default();
        assert!(info.profiles.contains(&VvcProfile::Main10));
        assert!(info.profiles.contains(&VvcProfile::Main10_444));
        assert_eq!(info.max_ctu_size, 128);
        assert_eq!(info.min_cu_size, 4);
        assert!(info.sao_enabled);
        assert!(info.alf_enabled);
        assert!(info.ccalf_enabled);
        assert!(info.lmcs_enabled);
        assert!(info.deblocking_enabled);
        assert!(info.mip_enabled);
        assert!(info.isp_enabled);
        assert!(info.lfnst_enabled);
        assert!(info.affine_enabled);
        assert!(info.gpm_enabled);
    }

    #[test]
    fn test_supports_profile() {
        let info = VvcInfo::new();
        assert!(info.supports_profile(VvcProfile::Main10));
        assert!(info.supports_profile(VvcProfile::Main10_444));
        assert!(!info.supports_profile(VvcProfile::MultilayerMain10));
    }

    #[test]
    fn test_supports_level() {
        let info = VvcInfo::new();
        assert!(info.supports_level(VvcLevel::L4_0));
        assert!(info.supports_level(VvcLevel::L5_1));
        assert!(info.supports_level(VvcLevel::L6_3));
    }

    #[test]
    fn test_max_resolution_for_level() {
        assert_eq!(VvcInfo::max_resolution_for_level(VvcLevel::L3_1), (1280, 720));
        assert_eq!(VvcInfo::max_resolution_for_level(VvcLevel::L4_0), (2048, 1080));
        assert_eq!(VvcInfo::max_resolution_for_level(VvcLevel::L5_1), (4096, 2160));
        assert_eq!(VvcInfo::max_resolution_for_level(VvcLevel::L6_1), (8192, 4320));
    }

    #[test]
    fn test_vvc_config_default() {
        let config = VvcConfig::default();
        assert_eq!(config.profile, VvcProfile::Main10);
        assert_eq!(config.tier, VvcTier::Main);
        assert_eq!(config.level, VvcLevel::L4_1);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.bit_depth, 10);
        assert_eq!(config.ctu_size, 128);
        assert!(config.alf_enabled);
    }

    #[test]
    fn test_vvc_config_builder() {
        let config = VvcConfig::new()
            .with_resolution(3840, 2160)
            .with_profile(VvcProfile::Main10_444)
            .with_level(VvcLevel::L5_1)
            .with_bit_depth(10)
            .with_ctu_size(128);

        assert_eq!(config.width, 3840);
        assert_eq!(config.height, 2160);
        assert_eq!(config.profile, VvcProfile::Main10_444);
        assert_eq!(config.level, VvcLevel::L5_1);
    }

    #[test]
    fn test_vvc_config_validate() {
        // Valid config
        let config = VvcConfig::default();
        assert!(config.validate().is_ok());

        // Invalid bit depth
        let mut config = VvcConfig::default();
        config.bit_depth = 16;
        assert!(config.validate().is_err());

        // Invalid CTU size
        let mut config = VvcConfig::default();
        config.ctu_size = 256;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_get_nal_unit_type() {
        // VPS NAL unit (type 14)
        // Byte 0: 0x00 (forbidden=0, reserved=0, layer_id=0)
        // Byte 1: 0x71 ((14 << 3) | 1 = 0x71)
        let vps_header = [0x00, 0x71];
        assert_eq!(get_nal_unit_type(&vps_header), Some(NalUnitType::VpsNut));

        // SPS NAL unit (type 15)
        // Byte 1: (15 << 3) | 1 = 0x79
        let sps_header = [0x00, 0x79];
        assert_eq!(get_nal_unit_type(&sps_header), Some(NalUnitType::SpsNut));

        // PPS NAL unit (type 16)
        // Byte 1: (16 << 3) | 1 = 0x81
        let pps_header = [0x00, 0x81];
        assert_eq!(get_nal_unit_type(&pps_header), Some(NalUnitType::PpsNut));
    }

    #[test]
    fn test_is_parameter_set() {
        assert!(is_parameter_set(NalUnitType::VpsNut));
        assert!(is_parameter_set(NalUnitType::SpsNut));
        assert!(is_parameter_set(NalUnitType::PpsNut));
        assert!(is_parameter_set(NalUnitType::ApsPrefix));
        assert!(!is_parameter_set(NalUnitType::IdrWRadl));
        assert!(!is_parameter_set(NalUnitType::TrailNut));
    }

    #[test]
    fn test_is_picture_header() {
        assert!(is_picture_header(NalUnitType::PhNut));
        assert!(!is_picture_header(NalUnitType::SpsNut));
    }

    #[test]
    fn test_is_idr_picture() {
        assert!(is_idr_picture(NalUnitType::IdrWRadl));
        assert!(is_idr_picture(NalUnitType::IdrNLp));
        assert!(!is_idr_picture(NalUnitType::CraNut));
        assert!(!is_idr_picture(NalUnitType::TrailNut));
    }

    #[test]
    fn test_is_gdr_picture() {
        assert!(is_gdr_picture(NalUnitType::GdrNut));
        assert!(!is_gdr_picture(NalUnitType::CraNut));
    }

    #[test]
    fn test_is_random_access_point() {
        assert!(is_random_access_point(NalUnitType::IdrWRadl));
        assert!(is_random_access_point(NalUnitType::IdrNLp));
        assert!(is_random_access_point(NalUnitType::CraNut));
        assert!(is_random_access_point(NalUnitType::GdrNut));
        assert!(!is_random_access_point(NalUnitType::TrailNut));
    }

    #[test]
    fn test_is_leading_picture() {
        assert!(is_leading_picture(NalUnitType::RaslNut));
        assert!(is_leading_picture(NalUnitType::RadlNut));
        assert!(!is_leading_picture(NalUnitType::TrailNut));
    }

    #[test]
    fn test_is_trailing_picture() {
        assert!(is_trailing_picture(NalUnitType::TrailNut));
        assert!(!is_trailing_picture(NalUnitType::IdrWRadl));
    }

    #[test]
    fn test_is_vcl_nal() {
        assert!(is_vcl_nal(NalUnitType::TrailNut));
        assert!(is_vcl_nal(NalUnitType::IdrWRadl));
        assert!(is_vcl_nal(NalUnitType::CraNut));
        assert!(is_vcl_nal(NalUnitType::GdrNut));
        assert!(!is_vcl_nal(NalUnitType::VpsNut));
        assert!(!is_vcl_nal(NalUnitType::SpsNut));
        assert!(!is_vcl_nal(NalUnitType::PpsNut));
    }

    #[test]
    fn test_calculate_ctu_count() {
        // 1920x1080 with 128x128 CTUs
        assert_eq!(calculate_ctu_count(1920, 1080, 128), 15 * 9);

        // 1920x1080 with 64x64 CTUs
        assert_eq!(calculate_ctu_count(1920, 1080, 64), 30 * 17);

        // 3840x2160 with 128x128 CTUs
        assert_eq!(calculate_ctu_count(3840, 2160, 128), 30 * 17);

        // 3840x2160 with 64x64 CTUs
        assert_eq!(calculate_ctu_count(3840, 2160, 64), 60 * 34);
    }

    #[test]
    fn test_calculate_min_level() {
        // 1080p30 at 8Mbps should be Level 4
        let level = calculate_min_level(1920, 1080, 30.0, 8_000, VvcTier::Main);
        assert!(level.is_some());

        // 4K60 at 20Mbps should be Level 5.1 or higher
        let level = calculate_min_level(3840, 2160, 60.0, 20_000, VvcTier::Main);
        assert!(level.is_some());
    }

    #[test]
    fn test_fourcc_codes() {
        assert_eq!(&fourcc::VVC1, b"vvc1");
        assert_eq!(&fourcc::VVI1, b"vvi1");
    }

    #[test]
    fn test_start_codes() {
        assert_eq!(start_codes::START_CODE_3, [0x00, 0x00, 0x01]);
        assert_eq!(start_codes::START_CODE_4, [0x00, 0x00, 0x00, 0x01]);
        assert_eq!(start_codes::EMULATION_PREVENTION_BYTE, 0x03);
    }

    #[test]
    fn test_parse_annexb_empty() {
        let data: &[u8] = &[];
        let units = parse_annexb_stream(data);
        assert!(units.is_empty());
    }
}
