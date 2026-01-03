//! HEVC-specific error types.
//!
//! This module provides comprehensive error handling for HEVC/H.265 codec operations.

use std::fmt;
use thiserror::Error;

/// HEVC-specific error type.
#[derive(Error, Debug)]
pub enum HevcError {
    /// NAL unit parsing error.
    #[error("NAL unit error: {0}")]
    NalUnit(#[from] NalError),

    /// Video Parameter Set error.
    #[error("VPS error: {0}")]
    Vps(String),

    /// Sequence Parameter Set error.
    #[error("SPS error: {0}")]
    Sps(String),

    /// Picture Parameter Set error.
    #[error("PPS error: {0}")]
    Pps(String),

    /// Slice header parsing error.
    #[error("Slice header error: {0}")]
    SliceHeader(String),

    /// CABAC decoding error.
    #[error("CABAC error: {0}")]
    Cabac(String),

    /// Transform/quantization error.
    #[error("Transform error: {0}")]
    Transform(String),

    /// Prediction error.
    #[error("Prediction error: {0}")]
    Prediction(String),

    /// SAO filter error.
    #[error("SAO filter error: {0}")]
    Sao(String),

    /// Deblocking filter error.
    #[error("Deblocking error: {0}")]
    Deblock(String),

    /// Profile/tier/level error.
    #[error("Profile/tier/level error: {0}")]
    ProfileTierLevel(String),

    /// Unsupported feature.
    #[error("Unsupported: {0}")]
    Unsupported(String),

    /// Reference picture error.
    #[error("Reference picture error: {0}")]
    ReferencePicture(String),

    /// Motion compensation error.
    #[error("Motion compensation error: {0}")]
    MotionCompensation(String),

    /// Coding tree unit error.
    #[error("CTU error: {0}")]
    Ctu(String),

    /// Coding unit error.
    #[error("CU error: {0}")]
    Cu(String),

    /// Bitstream error.
    #[error("Bitstream error: {0}")]
    Bitstream(String),

    /// Encoder configuration error.
    #[error("Encoder config error: {0}")]
    EncoderConfig(String),

    /// Decoder configuration error.
    #[error("Decoder config error: {0}")]
    DecoderConfig(String),

    /// Frame dimensions exceed limits.
    #[error("Frame dimensions {width}x{height} exceed maximum {max_width}x{max_height}")]
    DimensionsExceeded {
        width: u32,
        height: u32,
        max_width: u32,
        max_height: u32,
    },

    /// Invalid state.
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// End of stream.
    #[error("End of stream")]
    EndOfStream,

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Core library error.
    #[error("Core error: {0}")]
    Core(String),
}

/// NAL unit specific errors.
#[derive(Error, Debug)]
pub enum NalError {
    /// Invalid NAL unit type.
    #[error("Invalid NAL unit type: {0}")]
    InvalidType(u8),

    /// Invalid NAL unit header.
    #[error("Invalid NAL unit header")]
    InvalidHeader,

    /// Unexpected NAL unit type.
    #[error("Unexpected NAL unit type: expected {expected}, got {got}")]
    UnexpectedType { expected: String, got: String },

    /// Missing start code.
    #[error("Missing start code at offset {offset}")]
    MissingStartCode { offset: usize },

    /// Emulation prevention error.
    #[error("Emulation prevention error: {0}")]
    EmulationPrevention(String),

    /// Truncated NAL unit.
    #[error("Truncated NAL unit: expected {expected} bytes, got {got}")]
    Truncated { expected: usize, got: usize },

    /// RBSP trailing bits error.
    #[error("Invalid RBSP trailing bits")]
    InvalidTrailingBits,
}

/// HEVC profile definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HevcProfile {
    /// Main profile (8-bit, 4:2:0).
    Main = 1,
    /// Main 10 profile (10-bit, 4:2:0).
    Main10 = 2,
    /// Main Still Picture profile.
    MainStillPicture = 3,
    /// Range extensions.
    RangeExtensions = 4,
    /// High throughput profile.
    HighThroughput = 5,
    /// Multiview main profile.
    MultiviewMain = 6,
    /// Scalable main profile.
    ScalableMain = 7,
    /// 3D main profile.
    ThreeDMain = 8,
    /// Screen content coding.
    ScreenContentCoding = 9,
    /// Scalable range extensions.
    ScalableRangeExtensions = 10,
    /// High throughput screen content coding.
    HighThroughputScreenContent = 11,
}

impl HevcProfile {
    /// Create profile from IDC value.
    pub fn from_idc(idc: u8) -> Option<Self> {
        match idc {
            1 => Some(Self::Main),
            2 => Some(Self::Main10),
            3 => Some(Self::MainStillPicture),
            4 => Some(Self::RangeExtensions),
            5 => Some(Self::HighThroughput),
            6 => Some(Self::MultiviewMain),
            7 => Some(Self::ScalableMain),
            8 => Some(Self::ThreeDMain),
            9 => Some(Self::ScreenContentCoding),
            10 => Some(Self::ScalableRangeExtensions),
            11 => Some(Self::HighThroughputScreenContent),
            _ => None,
        }
    }

    /// Get the profile IDC value.
    pub fn idc(&self) -> u8 {
        *self as u8
    }

    /// Check if this profile supports 10-bit.
    pub fn supports_10bit(&self) -> bool {
        matches!(
            self,
            Self::Main10 | Self::RangeExtensions | Self::HighThroughput
        )
    }

    /// Check if this profile supports 4:2:2 chroma.
    pub fn supports_422(&self) -> bool {
        matches!(self, Self::RangeExtensions | Self::HighThroughput)
    }

    /// Check if this profile supports 4:4:4 chroma.
    pub fn supports_444(&self) -> bool {
        matches!(self, Self::RangeExtensions | Self::HighThroughput)
    }
}

impl fmt::Display for HevcProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Main => write!(f, "Main"),
            Self::Main10 => write!(f, "Main 10"),
            Self::MainStillPicture => write!(f, "Main Still Picture"),
            Self::RangeExtensions => write!(f, "Range Extensions"),
            Self::HighThroughput => write!(f, "High Throughput"),
            Self::MultiviewMain => write!(f, "Multiview Main"),
            Self::ScalableMain => write!(f, "Scalable Main"),
            Self::ThreeDMain => write!(f, "3D Main"),
            Self::ScreenContentCoding => write!(f, "Screen Content Coding"),
            Self::ScalableRangeExtensions => write!(f, "Scalable Range Extensions"),
            Self::HighThroughputScreenContent => write!(f, "High Throughput Screen Content"),
        }
    }
}

/// HEVC tier (Main or High).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum HevcTier {
    /// Main tier.
    #[default]
    Main,
    /// High tier.
    High,
}

impl fmt::Display for HevcTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Main => write!(f, "Main"),
            Self::High => write!(f, "High"),
        }
    }
}

/// HEVC level definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HevcLevel {
    /// Level value (e.g., 51 for level 5.1).
    pub level_idc: u8,
}

impl HevcLevel {
    /// Level 1.
    pub const L1: Self = Self { level_idc: 30 };
    /// Level 2.
    pub const L2: Self = Self { level_idc: 60 };
    /// Level 2.1.
    pub const L2_1: Self = Self { level_idc: 63 };
    /// Level 3.
    pub const L3: Self = Self { level_idc: 90 };
    /// Level 3.1.
    pub const L3_1: Self = Self { level_idc: 93 };
    /// Level 4.
    pub const L4: Self = Self { level_idc: 120 };
    /// Level 4.1.
    pub const L4_1: Self = Self { level_idc: 123 };
    /// Level 5.
    pub const L5: Self = Self { level_idc: 150 };
    /// Level 5.1.
    pub const L5_1: Self = Self { level_idc: 153 };
    /// Level 5.2.
    pub const L5_2: Self = Self { level_idc: 156 };
    /// Level 6.
    pub const L6: Self = Self { level_idc: 180 };
    /// Level 6.1.
    pub const L6_1: Self = Self { level_idc: 183 };
    /// Level 6.2.
    pub const L6_2: Self = Self { level_idc: 186 };

    /// Create a level from the IDC value.
    pub fn from_idc(idc: u8) -> Self {
        Self { level_idc: idc }
    }

    /// Get the major level number.
    pub fn major(&self) -> u8 {
        self.level_idc / 30
    }

    /// Get the minor level number.
    pub fn minor(&self) -> u8 {
        (self.level_idc % 30) / 3
    }

    /// Get the maximum luma samples per second for this level.
    pub fn max_luma_samples_per_second(&self) -> u64 {
        match self.level_idc {
            30 => 552_960,
            60 => 3_686_400,
            63 => 7_372_800,
            90 => 16_588_800,
            93 => 33_177_600,
            120 => 66_846_720,
            123 => 133_693_440,
            150 => 267_386_880,
            153 => 534_773_760,
            156 => 1_069_547_520,
            180 => 1_069_547_520,
            183 => 2_139_095_040,
            186 => 4_278_190_080,
            _ => 552_960, // Default to level 1
        }
    }

    /// Get the maximum luma picture size for this level.
    pub fn max_luma_picture_size(&self) -> u32 {
        match self.level_idc {
            30 => 36_864,
            60 => 122_880,
            63 => 245_760,
            90 => 552_960,
            93 => 983_040,
            120 => 2_228_224,
            123 => 2_228_224,
            150 => 8_912_896,
            153 => 8_912_896,
            156 => 8_912_896,
            180 => 35_651_584,
            183 => 35_651_584,
            186 => 35_651_584,
            _ => 36_864,
        }
    }

    /// Get the maximum bitrate in bits per second for this level and tier.
    pub fn max_bitrate(&self, tier: HevcTier) -> u64 {
        let (main, high) = match self.level_idc {
            30 => (128_000, 128_000),
            60 => (1_500_000, 1_500_000),
            63 => (3_000_000, 3_000_000),
            90 => (6_000_000, 6_000_000),
            93 => (10_000_000, 10_000_000),
            120 => (12_000_000, 30_000_000),
            123 => (20_000_000, 50_000_000),
            150 => (25_000_000, 100_000_000),
            153 => (40_000_000, 160_000_000),
            156 => (60_000_000, 240_000_000),
            180 => (60_000_000, 240_000_000),
            183 => (120_000_000, 480_000_000),
            186 => (240_000_000, 800_000_000),
            _ => (128_000, 128_000),
        };
        match tier {
            HevcTier::Main => main,
            HevcTier::High => high,
        }
    }
}

impl Default for HevcLevel {
    fn default() -> Self {
        Self::L4_1
    }
}

impl fmt::Display for HevcLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major(), self.minor())
    }
}

/// Result type for HEVC operations.
pub type Result<T> = std::result::Result<T, HevcError>;

impl From<transcode_core::Error> for HevcError {
    fn from(e: transcode_core::Error) -> Self {
        HevcError::Core(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hevc_profile() {
        assert_eq!(HevcProfile::Main.idc(), 1);
        assert_eq!(HevcProfile::Main10.idc(), 2);
        assert!(HevcProfile::Main10.supports_10bit());
        assert!(!HevcProfile::Main.supports_10bit());
        assert!(HevcProfile::RangeExtensions.supports_422());
        assert!(HevcProfile::RangeExtensions.supports_444());
    }

    #[test]
    fn test_hevc_level() {
        let level = HevcLevel::L5_1;
        assert_eq!(level.major(), 5);
        assert_eq!(level.minor(), 1);
        assert_eq!(level.to_string(), "5.1");
    }

    #[test]
    fn test_hevc_level_limits() {
        let level = HevcLevel::L4_1;
        assert!(level.max_luma_picture_size() > 0);
        assert!(level.max_bitrate(HevcTier::Main) < level.max_bitrate(HevcTier::High));
    }

    #[test]
    fn test_profile_from_idc() {
        assert_eq!(HevcProfile::from_idc(1), Some(HevcProfile::Main));
        assert_eq!(HevcProfile::from_idc(2), Some(HevcProfile::Main10));
        assert_eq!(HevcProfile::from_idc(99), None);
    }

    #[test]
    fn test_error_display() {
        let err = HevcError::Sps("invalid width".to_string());
        assert!(err.to_string().contains("SPS error"));
        assert!(err.to_string().contains("invalid width"));
    }

    #[test]
    fn test_nal_error() {
        let err = NalError::InvalidType(99);
        assert!(err.to_string().contains("99"));
    }
}
