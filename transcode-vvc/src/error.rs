//! VVC-specific error types.
//!
//! This module defines error types specific to VVC/H.266 encoding and decoding operations.

use std::fmt;
use transcode_core::error::Error as CoreError;

/// Result type for VVC operations.
pub type Result<T> = std::result::Result<T, VvcError>;

/// VVC-specific errors.
#[derive(Debug, Clone)]
pub enum VvcError {
    /// Bitstream parsing error.
    Bitstream(String),
    /// VPS (Video Parameter Set) error.
    Vps(String),
    /// SPS (Sequence Parameter Set) error.
    Sps(String),
    /// PPS (Picture Parameter Set) error.
    Pps(String),
    /// PH (Picture Header) error.
    PictureHeader(String),
    /// Slice header error.
    SliceHeader(String),
    /// NAL unit error.
    NalUnit(String),
    /// Invalid profile.
    InvalidProfile(u8),
    /// Invalid level.
    InvalidLevel(u8),
    /// Decoder configuration error.
    DecoderConfig(String),
    /// Encoder configuration error.
    EncoderConfig(String),
    /// Transform error.
    Transform(String),
    /// Prediction error.
    Prediction(String),
    /// CABAC error.
    Cabac(String),
    /// Reference picture error.
    ReferencePicture(String),
    /// Invalid state error.
    InvalidState(String),
    /// Unsupported feature.
    Unsupported(String),
    /// ALF (Adaptive Loop Filter) error.
    Alf(String),
    /// LMCS (Luma Mapping with Chroma Scaling) error.
    Lmcs(String),
    /// Subpicture error.
    Subpicture(String),
    /// CTU (Coding Tree Unit) error.
    Ctu(String),
    /// Core library error.
    Core(String),
}

impl fmt::Display for VvcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bitstream(msg) => write!(f, "VVC bitstream error: {}", msg),
            Self::Vps(msg) => write!(f, "VVC VPS error: {}", msg),
            Self::Sps(msg) => write!(f, "VVC SPS error: {}", msg),
            Self::Pps(msg) => write!(f, "VVC PPS error: {}", msg),
            Self::PictureHeader(msg) => write!(f, "VVC picture header error: {}", msg),
            Self::SliceHeader(msg) => write!(f, "VVC slice header error: {}", msg),
            Self::NalUnit(msg) => write!(f, "VVC NAL unit error: {}", msg),
            Self::InvalidProfile(idc) => write!(f, "VVC invalid profile: {}", idc),
            Self::InvalidLevel(idc) => write!(f, "VVC invalid level: {}", idc),
            Self::DecoderConfig(msg) => write!(f, "VVC decoder config error: {}", msg),
            Self::EncoderConfig(msg) => write!(f, "VVC encoder config error: {}", msg),
            Self::Transform(msg) => write!(f, "VVC transform error: {}", msg),
            Self::Prediction(msg) => write!(f, "VVC prediction error: {}", msg),
            Self::Cabac(msg) => write!(f, "VVC CABAC error: {}", msg),
            Self::ReferencePicture(msg) => write!(f, "VVC reference picture error: {}", msg),
            Self::InvalidState(msg) => write!(f, "VVC invalid state: {}", msg),
            Self::Unsupported(msg) => write!(f, "VVC unsupported feature: {}", msg),
            Self::Alf(msg) => write!(f, "VVC ALF error: {}", msg),
            Self::Lmcs(msg) => write!(f, "VVC LMCS error: {}", msg),
            Self::Subpicture(msg) => write!(f, "VVC subpicture error: {}", msg),
            Self::Ctu(msg) => write!(f, "VVC CTU error: {}", msg),
            Self::Core(msg) => write!(f, "VVC core error: {}", msg),
        }
    }
}

impl std::error::Error for VvcError {}

impl From<CoreError> for VvcError {
    fn from(err: CoreError) -> Self {
        VvcError::Core(err.to_string())
    }
}

impl From<VvcError> for CoreError {
    fn from(err: VvcError) -> Self {
        CoreError::Codec(transcode_core::error::CodecError::Other(err.to_string()))
    }
}

/// VVC profile identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VvcProfile {
    /// Main 10 profile.
    #[default]
    Main10,
    /// Main 10 4:4:4 profile.
    Main10_444,
    /// Main 10 Still Picture profile.
    Main10StillPicture,
    /// Main 10 4:4:4 Still Picture profile.
    Main10_444StillPicture,
    /// Multilayer Main 10 profile.
    MultilayerMain10,
    /// Multilayer Main 10 4:4:4 profile.
    MultilayerMain10_444,
}

impl VvcProfile {
    /// Get the profile indicator.
    pub fn idc(&self) -> u8 {
        match self {
            Self::Main10 => 1,
            Self::Main10_444 => 33,
            Self::Main10StillPicture => 65,
            Self::Main10_444StillPicture => 97,
            Self::MultilayerMain10 => 17,
            Self::MultilayerMain10_444 => 49,
        }
    }

    /// Create from profile indicator.
    pub fn from_idc(idc: u8) -> Option<Self> {
        match idc {
            1 => Some(Self::Main10),
            33 => Some(Self::Main10_444),
            65 => Some(Self::Main10StillPicture),
            97 => Some(Self::Main10_444StillPicture),
            17 => Some(Self::MultilayerMain10),
            49 => Some(Self::MultilayerMain10_444),
            _ => None,
        }
    }

    /// Check if this is a 4:4:4 profile.
    pub fn is_444(&self) -> bool {
        matches!(self, Self::Main10_444 | Self::Main10_444StillPicture | Self::MultilayerMain10_444)
    }

    /// Check if this is a still picture profile.
    pub fn is_still_picture(&self) -> bool {
        matches!(self, Self::Main10StillPicture | Self::Main10_444StillPicture)
    }

    /// Check if this is a multilayer profile.
    pub fn is_multilayer(&self) -> bool {
        matches!(self, Self::MultilayerMain10 | Self::MultilayerMain10_444)
    }
}

/// VVC tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VvcTier {
    /// Main tier.
    #[default]
    Main,
    /// High tier.
    High,
}

impl VvcTier {
    /// Create from tier flag.
    pub fn from_flag(flag: bool) -> Self {
        if flag { Self::High } else { Self::Main }
    }

    /// Get the tier flag.
    pub fn flag(&self) -> bool {
        matches!(self, Self::High)
    }
}

/// VVC level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VvcLevel {
    /// Level 1.0
    L1_0,
    /// Level 2.0
    L2_0,
    /// Level 2.1
    L2_1,
    /// Level 3.0
    L3_0,
    /// Level 3.1
    L3_1,
    /// Level 4.0
    L4_0,
    /// Level 4.1
    #[default]
    L4_1,
    /// Level 5.0
    L5_0,
    /// Level 5.1
    L5_1,
    /// Level 5.2
    L5_2,
    /// Level 6.0
    L6_0,
    /// Level 6.1
    L6_1,
    /// Level 6.2
    L6_2,
    /// Level 6.3
    L6_3,
}

impl VvcLevel {
    /// Get the level indicator.
    pub fn level_idc(&self) -> u8 {
        match self {
            Self::L1_0 => 16,
            Self::L2_0 => 32,
            Self::L2_1 => 35,
            Self::L3_0 => 48,
            Self::L3_1 => 51,
            Self::L4_0 => 64,
            Self::L4_1 => 67,
            Self::L5_0 => 80,
            Self::L5_1 => 83,
            Self::L5_2 => 86,
            Self::L6_0 => 96,
            Self::L6_1 => 99,
            Self::L6_2 => 102,
            Self::L6_3 => 105,
        }
    }

    /// Create from level indicator.
    pub fn from_idc(idc: u8) -> Option<Self> {
        match idc {
            16 => Some(Self::L1_0),
            32 => Some(Self::L2_0),
            35 => Some(Self::L2_1),
            48 => Some(Self::L3_0),
            51 => Some(Self::L3_1),
            64 => Some(Self::L4_0),
            67 => Some(Self::L4_1),
            80 => Some(Self::L5_0),
            83 => Some(Self::L5_1),
            86 => Some(Self::L5_2),
            96 => Some(Self::L6_0),
            99 => Some(Self::L6_1),
            102 => Some(Self::L6_2),
            105 => Some(Self::L6_3),
            _ => None,
        }
    }

    /// Get maximum luma samples per second.
    pub fn max_luma_ps(&self) -> u64 {
        match self {
            Self::L1_0 => 36_864,
            Self::L2_0 => 122_880,
            Self::L2_1 => 245_760,
            Self::L3_0 => 552_960,
            Self::L3_1 => 983_040,
            Self::L4_0 => 2_228_224,
            Self::L4_1 => 2_228_224,
            Self::L5_0 => 8_912_896,
            Self::L5_1 => 8_912_896,
            Self::L5_2 => 8_912_896,
            Self::L6_0 => 35_651_584,
            Self::L6_1 => 35_651_584,
            Self::L6_2 => 35_651_584,
            Self::L6_3 => 80_216_064,
        }
    }

    /// Get maximum bitrate (kbps) for main tier.
    pub fn max_br_main(&self) -> u32 {
        match self {
            Self::L1_0 => 256,
            Self::L2_0 => 1_500,
            Self::L2_1 => 3_000,
            Self::L3_0 => 6_000,
            Self::L3_1 => 10_000,
            Self::L4_0 => 12_000,
            Self::L4_1 => 20_000,
            Self::L5_0 => 25_000,
            Self::L5_1 => 40_000,
            Self::L5_2 => 60_000,
            Self::L6_0 => 80_000,
            Self::L6_1 => 120_000,
            Self::L6_2 => 180_000,
            Self::L6_3 => 240_000,
        }
    }

    /// Get maximum bitrate (kbps) for high tier.
    pub fn max_br_high(&self) -> u32 {
        match self {
            Self::L1_0 => 256,
            Self::L2_0 => 1_500,
            Self::L2_1 => 3_000,
            Self::L3_0 => 6_000,
            Self::L3_1 => 10_000,
            Self::L4_0 => 30_000,
            Self::L4_1 => 50_000,
            Self::L5_0 => 100_000,
            Self::L5_1 => 160_000,
            Self::L5_2 => 240_000,
            Self::L6_0 => 240_000,
            Self::L6_1 => 480_000,
            Self::L6_2 => 800_000,
            Self::L6_3 => 1_600_000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vvc_error_display() {
        let err = VvcError::Bitstream("Invalid data".to_string());
        assert!(err.to_string().contains("bitstream"));

        let err = VvcError::Vps("Missing VPS".to_string());
        assert!(err.to_string().contains("VPS"));
    }

    #[test]
    fn test_vvc_profile() {
        assert_eq!(VvcProfile::Main10.idc(), 1);
        assert_eq!(VvcProfile::from_idc(1), Some(VvcProfile::Main10));
        assert_eq!(VvcProfile::from_idc(255), None);

        assert!(!VvcProfile::Main10.is_444());
        assert!(VvcProfile::Main10_444.is_444());

        assert!(!VvcProfile::Main10.is_still_picture());
        assert!(VvcProfile::Main10StillPicture.is_still_picture());

        assert!(!VvcProfile::Main10.is_multilayer());
        assert!(VvcProfile::MultilayerMain10.is_multilayer());
    }

    #[test]
    fn test_vvc_tier() {
        assert!(!VvcTier::Main.flag());
        assert!(VvcTier::High.flag());
        assert_eq!(VvcTier::from_flag(false), VvcTier::Main);
        assert_eq!(VvcTier::from_flag(true), VvcTier::High);
    }

    #[test]
    fn test_vvc_level() {
        assert_eq!(VvcLevel::L4_1.level_idc(), 67);
        assert_eq!(VvcLevel::from_idc(67), Some(VvcLevel::L4_1));
        assert_eq!(VvcLevel::from_idc(0), None);

        assert!(VvcLevel::L6_0.max_br_high() > VvcLevel::L6_0.max_br_main());
        assert!(VvcLevel::L6_0.max_luma_ps() > VvcLevel::L4_0.max_luma_ps());
    }

    #[test]
    fn test_default_values() {
        assert_eq!(VvcProfile::default(), VvcProfile::Main10);
        assert_eq!(VvcTier::default(), VvcTier::Main);
        assert_eq!(VvcLevel::default(), VvcLevel::L4_1);
    }
}
