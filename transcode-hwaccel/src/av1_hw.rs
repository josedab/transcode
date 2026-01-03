//! AV1 hardware encoding types and configuration.
//!
//! This module provides AV1-specific types and configuration for hardware-accelerated
//! AV1 encoding. AV1 is a royalty-free video codec that offers excellent compression
//! efficiency, especially at lower bitrates.
//!
//! # Features
//!
//! - AV1 profiles (Main, High, Professional)
//! - AV1 levels (2.0 through 6.3)
//! - Tile configuration for parallel encoding
//! - Film grain synthesis support
//! - Screen content coding tools
//! - Scalable Video Coding (SVC) support

use crate::error::{HwAccelError, Result};
use crate::types::*;
use crate::HwCodec;

/// AV1 profile for hardware encoding.
///
/// AV1 defines three profiles that specify the allowed bitstream features:
/// - Main: 8-bit and 10-bit 4:2:0
/// - High: 8-bit and 10-bit 4:2:0 and 4:4:4
/// - Professional: 8-bit, 10-bit, and 12-bit for all chroma subsampling formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Av1Profile {
    /// Main profile - 8-bit and 10-bit YUV 4:2:0.
    #[default]
    Main,
    /// High profile - adds 4:4:4 support.
    High,
    /// Professional profile - adds 12-bit and 4:2:2 support.
    Professional,
}

impl Av1Profile {
    /// Get the profile index value used in sequence headers.
    pub fn seq_profile(&self) -> u8 {
        match self {
            Av1Profile::Main => 0,
            Av1Profile::High => 1,
            Av1Profile::Professional => 2,
        }
    }

    /// Get the profile name.
    pub fn name(&self) -> &'static str {
        match self {
            Av1Profile::Main => "Main",
            Av1Profile::High => "High",
            Av1Profile::Professional => "Professional",
        }
    }

    /// Check if this profile supports the given bit depth.
    pub fn supports_bit_depth(&self, bit_depth: u8) -> bool {
        match self {
            Av1Profile::Main => bit_depth == 8 || bit_depth == 10,
            Av1Profile::High => bit_depth == 8 || bit_depth == 10,
            Av1Profile::Professional => bit_depth == 8 || bit_depth == 10 || bit_depth == 12,
        }
    }

    /// Check if this profile supports 4:4:4 chroma subsampling.
    pub fn supports_444(&self) -> bool {
        matches!(self, Av1Profile::High | Av1Profile::Professional)
    }

    /// Check if this profile supports 4:2:2 chroma subsampling.
    pub fn supports_422(&self) -> bool {
        matches!(self, Av1Profile::Professional)
    }

    /// Get the codec type for this profile.
    pub fn codec(&self) -> HwCodec {
        HwCodec::Av1
    }
}


/// AV1 level for hardware encoding.
///
/// AV1 levels define constraints on resolution, frame rate, and bitrate.
/// Levels are numbered X.Y where X is 2-6 and Y is 0-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Av1Level {
    /// Level 2.0 - up to 426x240 @ 30fps.
    L2_0,
    /// Level 2.1 - up to 426x240 @ 60fps.
    L2_1,
    /// Level 2.2 - reserved.
    L2_2,
    /// Level 2.3 - reserved.
    L2_3,
    /// Level 3.0 - up to 854x480 @ 30fps.
    L3_0,
    /// Level 3.1 - up to 854x480 @ 60fps.
    L3_1,
    /// Level 3.2 - reserved.
    L3_2,
    /// Level 3.3 - reserved.
    L3_3,
    /// Level 4.0 - up to 1920x1080 @ 30fps.
    L4_0,
    /// Level 4.1 - up to 1920x1080 @ 60fps (good default for 1080p60).
    #[default]
    L4_1,
    /// Level 4.2 - up to 1920x1080 @ 120fps.
    L4_2,
    /// Level 4.3 - up to 1920x1080 @ 120fps (higher bitrate).
    L4_3,
    /// Level 5.0 - up to 3840x2160 @ 30fps.
    L5_0,
    /// Level 5.1 - up to 3840x2160 @ 60fps.
    L5_1,
    /// Level 5.2 - up to 3840x2160 @ 120fps.
    L5_2,
    /// Level 5.3 - up to 3840x2160 @ 120fps (higher bitrate).
    L5_3,
    /// Level 6.0 - up to 7680x4320 @ 30fps.
    L6_0,
    /// Level 6.1 - up to 7680x4320 @ 60fps.
    L6_1,
    /// Level 6.2 - up to 7680x4320 @ 120fps.
    L6_2,
    /// Level 6.3 - up to 7680x4320 @ 120fps (higher bitrate).
    L6_3,
}

impl Av1Level {
    /// Get the level index value used in sequence headers.
    pub fn seq_level_idx(&self) -> u8 {
        match self {
            Av1Level::L2_0 => 0,
            Av1Level::L2_1 => 1,
            Av1Level::L2_2 => 2,
            Av1Level::L2_3 => 3,
            Av1Level::L3_0 => 4,
            Av1Level::L3_1 => 5,
            Av1Level::L3_2 => 6,
            Av1Level::L3_3 => 7,
            Av1Level::L4_0 => 8,
            Av1Level::L4_1 => 9,
            Av1Level::L4_2 => 10,
            Av1Level::L4_3 => 11,
            Av1Level::L5_0 => 12,
            Av1Level::L5_1 => 13,
            Av1Level::L5_2 => 14,
            Av1Level::L5_3 => 15,
            Av1Level::L6_0 => 16,
            Av1Level::L6_1 => 17,
            Av1Level::L6_2 => 18,
            Av1Level::L6_3 => 19,
        }
    }

    /// Get the level name string (e.g., "4.1").
    pub fn name(&self) -> &'static str {
        match self {
            Av1Level::L2_0 => "2.0",
            Av1Level::L2_1 => "2.1",
            Av1Level::L2_2 => "2.2",
            Av1Level::L2_3 => "2.3",
            Av1Level::L3_0 => "3.0",
            Av1Level::L3_1 => "3.1",
            Av1Level::L3_2 => "3.2",
            Av1Level::L3_3 => "3.3",
            Av1Level::L4_0 => "4.0",
            Av1Level::L4_1 => "4.1",
            Av1Level::L4_2 => "4.2",
            Av1Level::L4_3 => "4.3",
            Av1Level::L5_0 => "5.0",
            Av1Level::L5_1 => "5.1",
            Av1Level::L5_2 => "5.2",
            Av1Level::L5_3 => "5.3",
            Av1Level::L6_0 => "6.0",
            Av1Level::L6_1 => "6.1",
            Av1Level::L6_2 => "6.2",
            Av1Level::L6_3 => "6.3",
        }
    }

    /// Get maximum picture width for this level.
    pub fn max_width(&self) -> u32 {
        match self {
            Av1Level::L2_0 | Av1Level::L2_1 | Av1Level::L2_2 | Av1Level::L2_3 => 2048,
            Av1Level::L3_0 | Av1Level::L3_1 | Av1Level::L3_2 | Av1Level::L3_3 => 4096,
            Av1Level::L4_0 | Av1Level::L4_1 | Av1Level::L4_2 | Av1Level::L4_3 => 8192,
            Av1Level::L5_0 | Av1Level::L5_1 | Av1Level::L5_2 | Av1Level::L5_3 => 8192,
            Av1Level::L6_0 | Av1Level::L6_1 | Av1Level::L6_2 | Av1Level::L6_3 => 16384,
        }
    }

    /// Get maximum picture height for this level.
    pub fn max_height(&self) -> u32 {
        match self {
            Av1Level::L2_0 | Av1Level::L2_1 | Av1Level::L2_2 | Av1Level::L2_3 => 1152,
            Av1Level::L3_0 | Av1Level::L3_1 | Av1Level::L3_2 | Av1Level::L3_3 => 2176,
            Av1Level::L4_0 | Av1Level::L4_1 | Av1Level::L4_2 | Av1Level::L4_3 => 4352,
            Av1Level::L5_0 | Av1Level::L5_1 | Av1Level::L5_2 | Av1Level::L5_3 => 4352,
            Av1Level::L6_0 | Av1Level::L6_1 | Av1Level::L6_2 | Av1Level::L6_3 => 8704,
        }
    }

    /// Get maximum bitrate in kbps for Main profile Main tier.
    pub fn max_bitrate_main(&self) -> u32 {
        match self {
            Av1Level::L2_0 => 1_500,
            Av1Level::L2_1 => 3_000,
            Av1Level::L2_2 | Av1Level::L2_3 => 4_500, // Reserved, estimate
            Av1Level::L3_0 => 6_000,
            Av1Level::L3_1 => 10_000,
            Av1Level::L3_2 | Av1Level::L3_3 => 15_000, // Reserved, estimate
            Av1Level::L4_0 => 12_000,
            Av1Level::L4_1 => 20_000,
            Av1Level::L4_2 | Av1Level::L4_3 => 30_000,
            Av1Level::L5_0 => 30_000,
            Av1Level::L5_1 => 40_000,
            Av1Level::L5_2 | Av1Level::L5_3 => 60_000,
            Av1Level::L6_0 => 60_000,
            Av1Level::L6_1 => 100_000,
            Av1Level::L6_2 | Av1Level::L6_3 => 160_000,
        }
    }

    /// Get maximum bitrate in kbps for Main profile High tier.
    pub fn max_bitrate_high(&self) -> u32 {
        // High tier typically allows 2-3x the main tier bitrate
        self.max_bitrate_main() * 2
    }

    /// Create level from sequence level index.
    pub fn from_seq_level_idx(idx: u8) -> Option<Self> {
        match idx {
            0 => Some(Av1Level::L2_0),
            1 => Some(Av1Level::L2_1),
            2 => Some(Av1Level::L2_2),
            3 => Some(Av1Level::L2_3),
            4 => Some(Av1Level::L3_0),
            5 => Some(Av1Level::L3_1),
            6 => Some(Av1Level::L3_2),
            7 => Some(Av1Level::L3_3),
            8 => Some(Av1Level::L4_0),
            9 => Some(Av1Level::L4_1),
            10 => Some(Av1Level::L4_2),
            11 => Some(Av1Level::L4_3),
            12 => Some(Av1Level::L5_0),
            13 => Some(Av1Level::L5_1),
            14 => Some(Av1Level::L5_2),
            15 => Some(Av1Level::L5_3),
            16 => Some(Av1Level::L6_0),
            17 => Some(Av1Level::L6_1),
            18 => Some(Av1Level::L6_2),
            19 => Some(Av1Level::L6_3),
            _ => None,
        }
    }

    /// Find appropriate level for given dimensions and frame rate.
    ///
    /// AV1 levels are defined by MaxPicSize (max frame pixels) and
    /// MaxDisplayRate (max samples per second).
    pub fn for_resolution(width: u32, height: u32, fps: f64) -> Self {
        let pixels = width * height;
        let display_rate = (pixels as f64 * fps) as u64;

        // AV1 level constraints (from spec):
        // Level 2.0: MaxPicSize=147456, MaxDisplayRate=4,423,680
        // Level 2.1: MaxPicSize=278784, MaxDisplayRate=8,363,520
        // Level 3.0: MaxPicSize=665856, MaxDisplayRate=19,975,680
        // Level 3.1: MaxPicSize=665856, MaxDisplayRate=31,950,720
        // Level 4.0: MaxPicSize=2228224, MaxDisplayRate=66,846,720
        // Level 4.1: MaxPicSize=2228224, MaxDisplayRate=133,693,440
        // Level 5.0: MaxPicSize=8912896, MaxDisplayRate=267,386,880
        // Level 5.1: MaxPicSize=8912896, MaxDisplayRate=534,773,760
        // Level 5.2: MaxPicSize=8912896, MaxDisplayRate=1,069,547,520
        // Level 5.3: MaxPicSize=8912896, MaxDisplayRate=1,069,547,520
        // Level 6.0: MaxPicSize=35651584, MaxDisplayRate=1,069,547,520
        // Level 6.1: MaxPicSize=35651584, MaxDisplayRate=2,139,095,040
        // Level 6.2: MaxPicSize=35651584, MaxDisplayRate=4,278,190,080
        // Level 6.3: MaxPicSize=35651584, MaxDisplayRate=4,278,190,080

        if pixels <= 147_456 && display_rate <= 4_423_680 {
            Av1Level::L2_0
        } else if pixels <= 278_784 && display_rate <= 8_363_520 {
            Av1Level::L2_1
        } else if pixels <= 665_856 && display_rate <= 19_975_680 {
            Av1Level::L3_0
        } else if pixels <= 665_856 && display_rate <= 31_950_720 {
            Av1Level::L3_1
        } else if pixels <= 2_228_224 && display_rate <= 66_846_720 {
            Av1Level::L4_0
        } else if pixels <= 2_228_224 && display_rate <= 133_693_440 {
            Av1Level::L4_1
        } else if pixels <= 8_912_896 && display_rate <= 267_386_880 {
            Av1Level::L5_0
        } else if pixels <= 8_912_896 && display_rate <= 534_773_760 {
            Av1Level::L5_1
        } else if pixels <= 8_912_896 && display_rate <= 1_069_547_520 {
            Av1Level::L5_2
        } else if pixels <= 35_651_584 && display_rate <= 1_069_547_520 {
            Av1Level::L6_0
        } else if pixels <= 35_651_584 && display_rate <= 2_139_095_040 {
            Av1Level::L6_1
        } else if pixels <= 35_651_584 && display_rate <= 4_278_190_080 {
            Av1Level::L6_2
        } else {
            Av1Level::L6_3
        }
    }
}


/// AV1 tier for rate limiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Av1Tier {
    /// Main tier (lower bitrate limits).
    #[default]
    Main,
    /// High tier (higher bitrate limits).
    High,
}

impl Av1Tier {
    /// Get the tier flag value.
    pub fn flag(&self) -> bool {
        match self {
            Av1Tier::Main => false,
            Av1Tier::High => true,
        }
    }
}

/// AV1 rate control mode for hardware encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1RateControl {
    /// Constant quantizer (similar to CRF).
    Cq { q: u8 },
    /// Constant quality parameter.
    ConstantQuality { quality: u8 },
    /// Constant bitrate.
    Cbr { bitrate: u32 },
    /// Variable bitrate with target and max.
    Vbr { target: u32, max: u32 },
    /// Constrained quality (target bitrate with quality floor).
    ConstrainedQuality { target_bitrate: u32, min_q: u8, max_q: u8 },
}

impl Av1RateControl {
    /// Get the target bitrate if applicable.
    pub fn target_bitrate(&self) -> Option<u32> {
        match self {
            Av1RateControl::Cq { .. } | Av1RateControl::ConstantQuality { .. } => None,
            Av1RateControl::Cbr { bitrate } => Some(*bitrate),
            Av1RateControl::Vbr { target, .. } => Some(*target),
            Av1RateControl::ConstrainedQuality { target_bitrate, .. } => Some(*target_bitrate),
        }
    }

    /// Get the Q parameter if applicable.
    pub fn q_value(&self) -> Option<u8> {
        match self {
            Av1RateControl::Cq { q } => Some(*q),
            Av1RateControl::ConstantQuality { quality } => Some(*quality),
            _ => None,
        }
    }
}

impl Default for Av1RateControl {
    fn default() -> Self {
        Av1RateControl::Vbr {
            target: 5_000_000,
            max: 10_000_000,
        }
    }
}

/// AV1 tile configuration for parallel encoding.
///
/// Tiles allow parallel encoding and decoding of different regions
/// of the frame. More tiles = better parallelism but potentially
/// lower compression efficiency due to reduced prediction across tile boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1TileConfig {
    /// Number of tile columns (power of 2, 1-64).
    pub columns: u8,
    /// Number of tile rows (power of 2, 1-64).
    pub rows: u8,
    /// Enable uniform tile spacing.
    pub uniform_spacing: bool,
    /// Minimum tile width in superblocks (default 1).
    pub min_width_sb: u8,
    /// Maximum tile width in superblocks (default 4096/64 = 64).
    pub max_width_sb: u8,
}

impl Av1TileConfig {
    /// Create a new tile configuration.
    pub fn new(columns: u8, rows: u8) -> Self {
        Self {
            columns: columns.max(1),
            rows: rows.max(1),
            uniform_spacing: true,
            min_width_sb: 1,
            max_width_sb: 64,
        }
    }

    /// Create tile configuration for single tile (no parallelism).
    pub fn single() -> Self {
        Self::new(1, 1)
    }

    /// Create tile configuration optimized for encoding speed.
    pub fn for_speed(width: u32, _height: u32, thread_count: u32) -> Self {
        // Calculate number of tiles based on thread count and frame size
        let sb_size = 64u32;
        let width_sbs = width.div_ceil(sb_size);

        // Try to match tile count to thread count
        let mut cols = 1u8;
        while (cols as u32 * 2) <= thread_count && (cols as u32 * 2) <= width_sbs {
            cols *= 2;
        }

        // Typically use fewer rows than columns
        let rows = if cols >= 4 { 2 } else { 1 };

        Self::new(cols, rows)
    }

    /// Calculate total number of tiles.
    pub fn total_tiles(&self) -> u32 {
        self.columns as u32 * self.rows as u32
    }

    /// Validate tile configuration against frame dimensions.
    pub fn validate(&self, width: u32, height: u32) -> Result<()> {
        let sb_size = 64u32;
        let width_sbs = width.div_ceil(sb_size);
        let height_sbs = height.div_ceil(sb_size);

        if self.columns as u32 > width_sbs {
            return Err(HwAccelError::Config(format!(
                "Too many tile columns ({}) for frame width ({} superblocks)",
                self.columns, width_sbs
            )));
        }

        if self.rows as u32 > height_sbs {
            return Err(HwAccelError::Config(format!(
                "Too many tile rows ({}) for frame height ({} superblocks)",
                self.rows, height_sbs
            )));
        }

        Ok(())
    }
}

impl Default for Av1TileConfig {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

/// AV1 film grain synthesis configuration.
///
/// Film grain synthesis adds natural-looking film grain noise
/// to the decoded video. This can improve perceived quality
/// for content that originally had film grain but was smoothed
/// during encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1FilmGrainConfig {
    /// Enable film grain synthesis.
    pub enabled: bool,
    /// Film grain seed for random number generation.
    pub seed: u16,
    /// Grain scaling (0-3, higher = more grain).
    pub strength: u8,
    /// Apply grain to luma component.
    pub apply_luma: bool,
    /// Apply grain to chroma components.
    pub apply_chroma: bool,
    /// Use overlap processing between blocks.
    pub overlap: bool,
    /// Clip to restricted range.
    pub clip_to_restricted_range: bool,
}

impl Av1FilmGrainConfig {
    /// Create a new film grain configuration.
    pub fn new(enabled: bool, strength: u8) -> Self {
        Self {
            enabled,
            seed: 0,
            strength: strength.min(3),
            apply_luma: true,
            apply_chroma: true,
            overlap: true,
            clip_to_restricted_range: true,
        }
    }

    /// Create disabled film grain configuration.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            seed: 0,
            strength: 0,
            apply_luma: false,
            apply_chroma: false,
            overlap: false,
            clip_to_restricted_range: false,
        }
    }

    /// Create light film grain configuration.
    pub fn light() -> Self {
        Self::new(true, 1)
    }

    /// Create medium film grain configuration.
    pub fn medium() -> Self {
        Self::new(true, 2)
    }

    /// Create strong film grain configuration.
    pub fn strong() -> Self {
        Self::new(true, 3)
    }
}

impl Default for Av1FilmGrainConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// AV1 screen content coding tools configuration.
///
/// Screen content coding tools are optimized for content like
/// presentations, screen captures, and computer-generated graphics
/// that have different characteristics than natural video.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1ScreenContentConfig {
    /// Enable screen content coding tools.
    pub enabled: bool,
    /// Allow intra block copy (IBC) mode.
    pub intra_block_copy: bool,
    /// Enable palette mode for flat color regions.
    pub palette_mode: bool,
    /// Maximum palette size (2-8).
    pub max_palette_size: u8,
}

impl Av1ScreenContentConfig {
    /// Create a new screen content configuration.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            intra_block_copy: enabled,
            palette_mode: enabled,
            max_palette_size: 8,
        }
    }

    /// Create disabled screen content configuration (for natural video).
    pub fn disabled() -> Self {
        Self::new(false)
    }

    /// Create enabled screen content configuration (for screen recordings).
    pub fn enabled() -> Self {
        Self::new(true)
    }
}

impl Default for Av1ScreenContentConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// AV1 Scalable Video Coding (SVC) configuration.
///
/// SVC allows encoding multiple quality/resolution layers in a single bitstream,
/// enabling adaptive streaming without re-encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1SvcConfig {
    /// Enable SVC encoding.
    pub enabled: bool,
    /// Number of spatial layers (1-4).
    pub spatial_layers: u8,
    /// Number of temporal layers (1-8).
    pub temporal_layers: u8,
    /// Scale factors for each spatial layer (width, height).
    pub spatial_scale_factors: Vec<(u8, u8)>,
    /// Bitrate allocation per layer (as percentages).
    pub layer_bitrate_allocation: Vec<u8>,
}

impl Av1SvcConfig {
    /// Create a new SVC configuration.
    pub fn new(spatial_layers: u8, temporal_layers: u8) -> Self {
        let spatial_layers = spatial_layers.clamp(1, 4);
        let temporal_layers = temporal_layers.clamp(1, 8);

        // Default scale factors: 1/4, 1/2, 3/4, 1
        let spatial_scale_factors = (0..spatial_layers)
            .map(|i| {
                let scale = (i + 1) * (4 / spatial_layers.max(1));
                (scale, scale)
            })
            .collect();

        // Default bitrate allocation: roughly proportional to resolution
        let layer_bitrate_allocation = (0..spatial_layers)
            .map(|i| {
                let proportion = (i + 1) * 25 / spatial_layers;
                proportion.min(100)
            })
            .collect();

        Self {
            enabled: spatial_layers > 1 || temporal_layers > 1,
            spatial_layers,
            temporal_layers,
            spatial_scale_factors,
            layer_bitrate_allocation,
        }
    }

    /// Create disabled SVC configuration.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            spatial_layers: 1,
            temporal_layers: 1,
            spatial_scale_factors: vec![(4, 4)],
            layer_bitrate_allocation: vec![100],
        }
    }

    /// Create configuration for temporal scalability only.
    pub fn temporal_only(layers: u8) -> Self {
        Self::new(1, layers)
    }

    /// Create configuration for spatial scalability only.
    pub fn spatial_only(layers: u8) -> Self {
        Self::new(layers, 1)
    }

    /// Get total number of layers.
    pub fn total_layers(&self) -> u8 {
        self.spatial_layers * self.temporal_layers
    }
}

impl Default for Av1SvcConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// AV1 reference frame configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1ReferenceConfig {
    /// Number of reference frames to use (1-7).
    pub ref_frames: u8,
    /// Enable golden frame (long-term reference).
    pub golden_frame: bool,
    /// Golden frame interval (frames between golden frames).
    pub golden_interval: u32,
    /// Enable alternate reference frame.
    pub alt_ref_frame: bool,
    /// Enable backward reference frame.
    pub backward_ref: bool,
}

impl Av1ReferenceConfig {
    /// Create a new reference configuration.
    pub fn new(ref_frames: u8) -> Self {
        Self {
            ref_frames: ref_frames.clamp(1, 7),
            golden_frame: true,
            golden_interval: 16,
            alt_ref_frame: true,
            backward_ref: true,
        }
    }

    /// Create configuration for low-latency encoding (fewer references).
    pub fn low_latency() -> Self {
        Self {
            ref_frames: 2,
            golden_frame: false,
            golden_interval: 0,
            alt_ref_frame: false,
            backward_ref: false,
        }
    }

    /// Create configuration for high quality encoding (more references).
    pub fn high_quality() -> Self {
        Self {
            ref_frames: 7,
            golden_frame: true,
            golden_interval: 8,
            alt_ref_frame: true,
            backward_ref: true,
        }
    }
}

impl Default for Av1ReferenceConfig {
    fn default() -> Self {
        Self::new(4)
    }
}

/// Complete AV1 hardware encoder configuration.
#[derive(Debug, Clone)]
pub struct Av1HwEncoderConfig {
    /// Base encoder configuration.
    pub base: crate::encoder::HwEncoderConfig,
    /// AV1 profile.
    pub profile: Av1Profile,
    /// AV1 level.
    pub level: Av1Level,
    /// AV1 tier.
    pub tier: Av1Tier,
    /// Rate control configuration.
    pub rate_control: Av1RateControl,
    /// Tile configuration.
    pub tiles: Av1TileConfig,
    /// Film grain configuration.
    pub film_grain: Av1FilmGrainConfig,
    /// Screen content coding configuration.
    pub screen_content: Av1ScreenContentConfig,
    /// SVC configuration.
    pub svc: Av1SvcConfig,
    /// Reference frame configuration.
    pub references: Av1ReferenceConfig,
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,
    /// Enable CDEF (Constrained Directional Enhancement Filter).
    pub cdef: bool,
    /// Enable loop restoration filter.
    pub loop_restoration: bool,
    /// Superblock size (64 or 128).
    pub superblock_size: u8,
    /// Enable temporal filtering for AltRef frames.
    pub temporal_filtering: bool,
    /// Encoder speed preset (0-10, lower = slower/better quality).
    pub speed_preset: u8,
}

impl Av1HwEncoderConfig {
    /// Create a new AV1 encoder configuration.
    pub fn new(width: u32, height: u32, bitrate: u32) -> Self {
        let base = crate::encoder::HwEncoderConfig {
            codec: HwCodec::Av1,
            width,
            height,
            rate_control: HwRateControl::Vbr {
                target: bitrate,
                max: bitrate * 2,
            },
            profile: Some(HwProfile::Av1Main),
            bframes: 0, // AV1 doesn't use traditional B-frames
            ..Default::default()
        };

        let level = Av1Level::for_resolution(width, height, 30.0);

        Self {
            base,
            profile: Av1Profile::Main,
            level,
            tier: Av1Tier::Main,
            rate_control: Av1RateControl::Vbr {
                target: bitrate,
                max: bitrate * 2,
            },
            tiles: Av1TileConfig::default(),
            film_grain: Av1FilmGrainConfig::default(),
            screen_content: Av1ScreenContentConfig::default(),
            svc: Av1SvcConfig::default(),
            references: Av1ReferenceConfig::default(),
            bit_depth: 8,
            cdef: true,
            loop_restoration: true,
            superblock_size: 128,
            temporal_filtering: true,
            speed_preset: 5,
        }
    }

    /// Create configuration for 10-bit encoding.
    pub fn new_10bit(width: u32, height: u32, bitrate: u32) -> Self {
        let mut config = Self::new(width, height, bitrate);
        config.bit_depth = 10;
        config.base.input_format = HwSurfaceFormat::P010;
        config
    }

    /// Create configuration optimized for screen recording.
    pub fn screen_recording(width: u32, height: u32, bitrate: u32) -> Self {
        let mut config = Self::new(width, height, bitrate);
        config.screen_content = Av1ScreenContentConfig::enabled();
        config.speed_preset = 8; // Faster for real-time capture
        config
    }

    /// Create configuration optimized for streaming.
    pub fn streaming(width: u32, height: u32, bitrate: u32) -> Self {
        let mut config = Self::new(width, height, bitrate);
        config.references = Av1ReferenceConfig::low_latency();
        config.tiles = Av1TileConfig::for_speed(width, height, 4);
        config.speed_preset = 7;
        config
    }

    /// Create configuration optimized for archival quality.
    pub fn archival(width: u32, height: u32, bitrate: u32) -> Self {
        let mut config = Self::new(width, height, bitrate);
        config.references = Av1ReferenceConfig::high_quality();
        config.speed_preset = 2;
        config.temporal_filtering = true;
        config
    }

    /// Set profile.
    pub fn with_profile(mut self, profile: Av1Profile) -> Self {
        self.profile = profile;
        self
    }

    /// Set level.
    pub fn with_level(mut self, level: Av1Level) -> Self {
        self.level = level;
        self
    }

    /// Set tile configuration.
    pub fn with_tiles(mut self, tiles: Av1TileConfig) -> Self {
        self.tiles = tiles;
        self
    }

    /// Enable film grain synthesis.
    pub fn with_film_grain(mut self, config: Av1FilmGrainConfig) -> Self {
        self.film_grain = config;
        self
    }

    /// Enable screen content tools.
    pub fn with_screen_content(mut self, config: Av1ScreenContentConfig) -> Self {
        self.screen_content = config;
        self
    }

    /// Enable SVC encoding.
    pub fn with_svc(mut self, config: Av1SvcConfig) -> Self {
        self.svc = config;
        self
    }

    /// Set speed preset.
    pub fn with_speed(mut self, preset: u8) -> Self {
        self.speed_preset = preset.min(10);
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Check bit depth compatibility with profile
        if !self.profile.supports_bit_depth(self.bit_depth) {
            return Err(HwAccelError::Config(format!(
                "{}-bit encoding not supported by {} profile",
                self.bit_depth,
                self.profile.name()
            )));
        }

        // Validate tile configuration
        self.tiles.validate(self.base.width, self.base.height)?;

        // Check level constraints
        if self.base.width > self.level.max_width() {
            return Err(HwAccelError::Config(format!(
                "Width {} exceeds level {} maximum of {}",
                self.base.width,
                self.level.name(),
                self.level.max_width()
            )));
        }

        if self.base.height > self.level.max_height() {
            return Err(HwAccelError::Config(format!(
                "Height {} exceeds level {} maximum of {}",
                self.base.height,
                self.level.name(),
                self.level.max_height()
            )));
        }

        // Validate superblock size
        if self.superblock_size != 64 && self.superblock_size != 128 {
            return Err(HwAccelError::Config(
                "Superblock size must be 64 or 128".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for Av1HwEncoderConfig {
    fn default() -> Self {
        Self::new(1920, 1080, 5_000_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_av1_profile() {
        assert_eq!(Av1Profile::Main.seq_profile(), 0);
        assert_eq!(Av1Profile::High.seq_profile(), 1);
        assert_eq!(Av1Profile::Professional.seq_profile(), 2);

        assert!(Av1Profile::Main.supports_bit_depth(8));
        assert!(Av1Profile::Main.supports_bit_depth(10));
        assert!(!Av1Profile::Main.supports_bit_depth(12));
        assert!(Av1Profile::Professional.supports_bit_depth(12));
    }

    #[test]
    fn test_av1_level() {
        assert_eq!(Av1Level::L4_1.seq_level_idx(), 9);
        assert_eq!(Av1Level::L5_0.name(), "5.0");

        let level = Av1Level::for_resolution(1920, 1080, 30.0);
        assert!(matches!(level, Av1Level::L4_0 | Av1Level::L4_1));

        let level = Av1Level::for_resolution(3840, 2160, 60.0);
        assert!(matches!(level, Av1Level::L5_1));
    }

    #[test]
    fn test_av1_tile_config() {
        let tiles = Av1TileConfig::new(4, 2);
        assert_eq!(tiles.total_tiles(), 8);

        let tiles = Av1TileConfig::for_speed(1920, 1080, 8);
        assert!(tiles.columns >= 2);

        // Validate should pass for reasonable configuration
        let result = tiles.validate(1920, 1080);
        assert!(result.is_ok());
    }

    #[test]
    fn test_av1_film_grain() {
        let fg = Av1FilmGrainConfig::medium();
        assert!(fg.enabled);
        assert_eq!(fg.strength, 2);

        let fg = Av1FilmGrainConfig::disabled();
        assert!(!fg.enabled);
    }

    #[test]
    fn test_av1_screen_content() {
        let sc = Av1ScreenContentConfig::enabled();
        assert!(sc.enabled);
        assert!(sc.intra_block_copy);
        assert!(sc.palette_mode);
    }

    #[test]
    fn test_av1_svc() {
        let svc = Av1SvcConfig::new(2, 3);
        assert!(svc.enabled);
        assert_eq!(svc.spatial_layers, 2);
        assert_eq!(svc.temporal_layers, 3);
        assert_eq!(svc.total_layers(), 6);

        let svc = Av1SvcConfig::temporal_only(4);
        assert_eq!(svc.spatial_layers, 1);
        assert_eq!(svc.temporal_layers, 4);
    }

    #[test]
    fn test_av1_hw_encoder_config() {
        let config = Av1HwEncoderConfig::new(1920, 1080, 5_000_000);
        assert_eq!(config.profile, Av1Profile::Main);
        assert_eq!(config.bit_depth, 8);
        assert!(config.validate().is_ok());

        let config = Av1HwEncoderConfig::new_10bit(3840, 2160, 15_000_000);
        assert_eq!(config.bit_depth, 10);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_av1_rate_control() {
        let rc = Av1RateControl::Vbr {
            target: 5_000_000,
            max: 10_000_000,
        };
        assert_eq!(rc.target_bitrate(), Some(5_000_000));

        let rc = Av1RateControl::Cq { q: 30 };
        assert_eq!(rc.q_value(), Some(30));
        assert_eq!(rc.target_bitrate(), None);
    }

    #[test]
    fn test_av1_reference_config() {
        let refs = Av1ReferenceConfig::low_latency();
        assert_eq!(refs.ref_frames, 2);
        assert!(!refs.golden_frame);

        let refs = Av1ReferenceConfig::high_quality();
        assert_eq!(refs.ref_frames, 7);
        assert!(refs.golden_frame);
    }
}
