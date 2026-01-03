//! H.264/AVC codec implementation.
//!
//! This module provides H.264 decoding and encoding capabilities.

// Allow common patterns in signal processing codec implementation
#![allow(
    dead_code,
    unused_must_use,
    unused_variables,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::cast_abs_to_unsigned
)]

mod nal;
mod pps;
mod sps;
mod slice;
mod decoder;
mod encoder;
mod dpb;
mod cavlc;
mod cabac;
mod prediction;
mod transform;
mod deblock;
mod mb_encoder;
pub mod parallel;
pub mod presets;

pub use decoder::{H264Decoder, H264DecoderConfig};
pub use encoder::{H264Encoder, H264EncoderConfig, RateControlMode, EncoderPreset};
pub use nal::{NalUnit, NalUnitType, NalIterator, parse_avcc, parse_annex_b};
pub use sps::SequenceParameterSet;
pub use pps::PictureParameterSet;
pub use slice::{SliceHeader, SliceType};
pub use presets::{Preset, PresetBuilder, PresetCategory, all_presets, presets_for_category};
pub use parallel::{ThreadingConfig, FrameType};

/// H.264 profile definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264Profile {
    /// Baseline profile (no B-frames, CAVLC only).
    Baseline = 66,
    /// Main profile.
    Main = 77,
    /// Extended profile.
    Extended = 88,
    /// High profile.
    High = 100,
    /// High 10 profile (10-bit).
    High10 = 110,
    /// High 4:2:2 profile.
    High422 = 122,
    /// High 4:4:4 Predictive profile.
    High444 = 244,
}

impl H264Profile {
    /// Create from profile_idc value.
    pub fn from_idc(idc: u8) -> Option<Self> {
        match idc {
            66 => Some(Self::Baseline),
            77 => Some(Self::Main),
            88 => Some(Self::Extended),
            100 => Some(Self::High),
            110 => Some(Self::High10),
            122 => Some(Self::High422),
            244 => Some(Self::High444),
            _ => None,
        }
    }

    /// Check if B-frames are allowed in this profile.
    pub fn allows_b_frames(&self) -> bool {
        !matches!(self, Self::Baseline)
    }

    /// Check if CABAC is allowed in this profile.
    pub fn allows_cabac(&self) -> bool {
        !matches!(self, Self::Baseline)
    }
}

/// H.264 level definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H264Level {
    /// Level value (e.g., 31 = 3.1, 40 = 4.0).
    pub value: u8,
}

impl H264Level {
    /// Create a level from the level_idc value.
    pub fn from_idc(idc: u8) -> Self {
        Self { value: idc }
    }

    /// Get the maximum macroblocks per second for this level.
    pub fn max_mbps(&self) -> u32 {
        match self.value {
            10 => 1485,
            11 => 3000,
            12 => 6000,
            13 => 11880,
            20 => 11880,
            21 => 19800,
            22 => 20250,
            30 => 40500,
            31 => 108000,
            32 => 216000,
            40 => 245760,
            41 => 245760,
            42 => 522240,
            50 => 589824,
            51 => 983040,
            52 => 2073600,
            _ => 245760,
        }
    }

    /// Get the maximum frame size in macroblocks for this level.
    pub fn max_fs(&self) -> u32 {
        match self.value {
            10 => 99,
            11 => 396,
            12 => 396,
            13 => 396,
            20 => 396,
            21 => 792,
            22 => 1620,
            30 => 1620,
            31 => 3600,
            32 => 5120,
            40 | 41 => 8192,
            42 => 8704,
            50 => 22080,
            51 => 36864,
            52 => 36864,
            _ => 8192,
        }
    }

    /// Get the maximum decoded picture buffer size in frames.
    pub fn max_dpb_mbs(&self) -> u32 {
        match self.value {
            10 => 396,
            11 => 900,
            12 => 2376,
            13 => 2376,
            20 => 2376,
            21 => 4752,
            22 => 8100,
            30 => 8100,
            31 => 18000,
            32 => 20480,
            40 | 41 => 32768,
            42 => 34816,
            50 => 110400,
            51 | 52 => 184320,
            _ => 32768,
        }
    }
}
