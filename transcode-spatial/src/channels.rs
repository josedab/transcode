//! Channel layouts for spatial audio.
//!
//! This module provides comprehensive channel layout support including standard
//! layouts (stereo, 5.1, 7.1, 7.1.4 Atmos), channel positions, custom layouts,
//! and channel order conversion between different standards (WAV, SMPTE, Dolby).

use crate::error::{ChannelLayoutError, Result, SpatialError};
use std::fmt;

/// Individual channel position in a spatial audio layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChannelPosition {
    // Standard positions
    /// Front Left
    FrontLeft,
    /// Front Right
    FrontRight,
    /// Front Center
    FrontCenter,
    /// Low Frequency Effects (subwoofer)
    Lfe,
    /// Back Left (Rear Left)
    BackLeft,
    /// Back Right (Rear Right)
    BackRight,
    /// Side Left
    SideLeft,
    /// Side Right
    SideRight,
    /// Back Center (Rear Center)
    BackCenter,

    // Height/Top layer positions (for Atmos and immersive audio)
    /// Top Front Left
    TopFrontLeft,
    /// Top Front Right
    TopFrontRight,
    /// Top Front Center
    TopFrontCenter,
    /// Top Back Left
    TopBackLeft,
    /// Top Back Right
    TopBackRight,
    /// Top Back Center
    TopBackCenter,
    /// Top Side Left
    TopSideLeft,
    /// Top Side Right
    TopSideRight,
    /// Top Center (directly above)
    TopCenter,

    // Bottom layer (for full sphere)
    /// Bottom Front Left
    BottomFrontLeft,
    /// Bottom Front Right
    BottomFrontRight,
    /// Bottom Front Center
    BottomFrontCenter,

    // Special channels
    /// Mono (single channel)
    Mono,
    /// Left Total (Lt for matrix encoding)
    LeftTotal,
    /// Right Total (Rt for matrix encoding)
    RightTotal,
    /// LFE2 (second subwoofer)
    Lfe2,
    /// Wide Left
    WideLeft,
    /// Wide Right
    WideRight,
    /// Surround Left (generic surround)
    SurroundLeft,
    /// Surround Right (generic surround)
    SurroundRight,

    /// Unknown or custom position with index
    Unknown(u32),
}

impl ChannelPosition {
    /// Get the abbreviated name for this position.
    pub fn abbreviation(&self) -> &'static str {
        match self {
            Self::FrontLeft => "FL",
            Self::FrontRight => "FR",
            Self::FrontCenter => "FC",
            Self::Lfe => "LFE",
            Self::BackLeft => "BL",
            Self::BackRight => "BR",
            Self::SideLeft => "SL",
            Self::SideRight => "SR",
            Self::BackCenter => "BC",
            Self::TopFrontLeft => "TFL",
            Self::TopFrontRight => "TFR",
            Self::TopFrontCenter => "TFC",
            Self::TopBackLeft => "TBL",
            Self::TopBackRight => "TBR",
            Self::TopBackCenter => "TBC",
            Self::TopSideLeft => "TSL",
            Self::TopSideRight => "TSR",
            Self::TopCenter => "TC",
            Self::BottomFrontLeft => "BFL",
            Self::BottomFrontRight => "BFR",
            Self::BottomFrontCenter => "BFC",
            Self::Mono => "M",
            Self::LeftTotal => "Lt",
            Self::RightTotal => "Rt",
            Self::Lfe2 => "LFE2",
            Self::WideLeft => "WL",
            Self::WideRight => "WR",
            Self::SurroundLeft => "SuL",
            Self::SurroundRight => "SuR",
            Self::Unknown(i) => {
                // Return static str, can't format dynamically
                match i {
                    0 => "U0",
                    1 => "U1",
                    2 => "U2",
                    3 => "U3",
                    _ => "U?",
                }
            }
        }
    }

    /// Get the 3D position as (azimuth, elevation) in degrees.
    /// Azimuth: 0 = front, positive = left, negative = right.
    /// Elevation: 0 = ear level, positive = above, negative = below.
    pub fn position_degrees(&self) -> (f32, f32) {
        match self {
            Self::FrontCenter => (0.0, 0.0),
            Self::FrontLeft => (30.0, 0.0),
            Self::FrontRight => (-30.0, 0.0),
            Self::SideLeft => (90.0, 0.0),
            Self::SideRight => (-90.0, 0.0),
            Self::BackLeft => (135.0, 0.0),
            Self::BackRight => (-135.0, 0.0),
            Self::BackCenter => (180.0, 0.0),
            Self::Lfe => (0.0, -30.0), // Sub is typically below
            Self::Lfe2 => (0.0, -30.0),
            Self::TopFrontLeft => (45.0, 45.0),
            Self::TopFrontRight => (-45.0, 45.0),
            Self::TopFrontCenter => (0.0, 45.0),
            Self::TopBackLeft => (135.0, 45.0),
            Self::TopBackRight => (-135.0, 45.0),
            Self::TopBackCenter => (180.0, 45.0),
            Self::TopSideLeft => (90.0, 45.0),
            Self::TopSideRight => (-90.0, 45.0),
            Self::TopCenter => (0.0, 90.0),
            Self::BottomFrontLeft => (45.0, -30.0),
            Self::BottomFrontRight => (-45.0, -30.0),
            Self::BottomFrontCenter => (0.0, -30.0),
            Self::Mono => (0.0, 0.0),
            Self::LeftTotal => (30.0, 0.0),
            Self::RightTotal => (-30.0, 0.0),
            Self::WideLeft => (60.0, 0.0),
            Self::WideRight => (-60.0, 0.0),
            Self::SurroundLeft => (110.0, 0.0),
            Self::SurroundRight => (-110.0, 0.0),
            Self::Unknown(_) => (0.0, 0.0),
        }
    }

    /// Check if this is a height channel (top layer).
    pub fn is_height(&self) -> bool {
        matches!(
            self,
            Self::TopFrontLeft
                | Self::TopFrontRight
                | Self::TopFrontCenter
                | Self::TopBackLeft
                | Self::TopBackRight
                | Self::TopBackCenter
                | Self::TopSideLeft
                | Self::TopSideRight
                | Self::TopCenter
        )
    }

    /// Check if this is an LFE channel.
    pub fn is_lfe(&self) -> bool {
        matches!(self, Self::Lfe | Self::Lfe2)
    }

    /// Check if this is a surround/rear channel.
    pub fn is_surround(&self) -> bool {
        matches!(
            self,
            Self::BackLeft
                | Self::BackRight
                | Self::BackCenter
                | Self::SideLeft
                | Self::SideRight
                | Self::SurroundLeft
                | Self::SurroundRight
        )
    }

    /// Parse from abbreviation string.
    pub fn from_abbreviation(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "FL" | "L" => Some(Self::FrontLeft),
            "FR" | "R" => Some(Self::FrontRight),
            "FC" | "C" => Some(Self::FrontCenter),
            "LFE" | "SUB" => Some(Self::Lfe),
            "BL" | "RL" => Some(Self::BackLeft),
            "BR" | "RR" => Some(Self::BackRight),
            "SL" | "LS" => Some(Self::SideLeft),
            "SR" | "RS" => Some(Self::SideRight),
            "BC" | "RC" => Some(Self::BackCenter),
            "TFL" => Some(Self::TopFrontLeft),
            "TFR" => Some(Self::TopFrontRight),
            "TFC" => Some(Self::TopFrontCenter),
            "TBL" => Some(Self::TopBackLeft),
            "TBR" => Some(Self::TopBackRight),
            "TBC" => Some(Self::TopBackCenter),
            "TSL" => Some(Self::TopSideLeft),
            "TSR" => Some(Self::TopSideRight),
            "TC" => Some(Self::TopCenter),
            "M" | "MONO" => Some(Self::Mono),
            "LT" => Some(Self::LeftTotal),
            "RT" => Some(Self::RightTotal),
            "LFE2" => Some(Self::Lfe2),
            _ => None,
        }
    }
}

impl fmt::Display for ChannelPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.abbreviation())
    }
}

/// Channel ordering standards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ChannelOrder {
    /// Microsoft WAV/WAVEFORMATEXTENSIBLE order.
    #[default]
    Wav,
    /// SMPTE/ITU standard order.
    Smpte,
    /// Dolby order (used in AC-3, E-AC-3).
    Dolby,
    /// Film industry order.
    Film,
    /// AAC order.
    Aac,
    /// FLAC/Vorbis order.
    Flac,
    /// Custom order (as specified).
    Custom,
}

impl ChannelOrder {
    /// Get the channel positions for 5.1 surround in this order.
    pub fn surround_51(&self) -> Vec<ChannelPosition> {
        match self {
            Self::Wav | Self::Custom => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
            ],
            Self::Smpte => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
            ],
            Self::Dolby | Self::Aac => vec![
                ChannelPosition::FrontCenter,
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
                ChannelPosition::Lfe,
            ],
            Self::Film => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontCenter,
                ChannelPosition::FrontRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
                ChannelPosition::Lfe,
            ],
            Self::Flac => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
            ],
        }
    }

    /// Get the channel positions for 7.1 surround in this order.
    pub fn surround_71(&self) -> Vec<ChannelPosition> {
        match self {
            Self::Wav | Self::Custom => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
            ],
            Self::Smpte => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
            ],
            Self::Dolby | Self::Aac => vec![
                ChannelPosition::FrontCenter,
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::Lfe,
            ],
            Self::Film => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontCenter,
                ChannelPosition::FrontRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::Lfe,
            ],
            Self::Flac => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
            ],
        }
    }
}

/// Standard channel layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StandardLayout {
    /// Mono (1 channel).
    Mono,
    /// Stereo (2 channels: L, R).
    #[default]
    Stereo,
    /// 2.1 (3 channels: L, R, LFE).
    Surround21,
    /// 3.0 (3 channels: L, R, C).
    Surround30,
    /// 4.0 Quadraphonic (4 channels: FL, FR, BL, BR).
    Quad,
    /// 5.0 (5 channels: FL, FR, FC, BL, BR).
    Surround50,
    /// 5.1 (6 channels: FL, FR, FC, LFE, BL, BR).
    Surround51,
    /// 6.1 (7 channels: FL, FR, FC, LFE, BL, BR, BC).
    Surround61,
    /// 7.0 (7 channels: FL, FR, FC, BL, BR, SL, SR).
    Surround70,
    /// 7.1 (8 channels: FL, FR, FC, LFE, BL, BR, SL, SR).
    Surround71,
    /// 7.1.2 Atmos (10 channels: 7.1 + TFL, TFR).
    Atmos712,
    /// 7.1.4 Atmos (12 channels: 7.1 + TFL, TFR, TBL, TBR).
    Atmos714,
    /// 9.1.6 Atmos (16 channels: 9.1 + 6 height channels).
    Atmos916,
}

impl StandardLayout {
    /// Get the number of channels in this layout.
    pub fn channel_count(&self) -> u32 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround21 | Self::Surround30 => 3,
            Self::Quad => 4,
            Self::Surround50 => 5,
            Self::Surround51 => 6,
            Self::Surround61 | Self::Surround70 => 7,
            Self::Surround71 => 8,
            Self::Atmos712 => 10,
            Self::Atmos714 => 12,
            Self::Atmos916 => 16,
        }
    }

    /// Get the channel positions for this layout in WAV order.
    pub fn positions(&self) -> Vec<ChannelPosition> {
        match self {
            Self::Mono => vec![ChannelPosition::Mono],
            Self::Stereo => vec![ChannelPosition::FrontLeft, ChannelPosition::FrontRight],
            Self::Surround21 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::Lfe,
            ],
            Self::Surround30 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
            ],
            Self::Quad => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
            ],
            Self::Surround50 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
            ],
            Self::Surround51 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
            ],
            Self::Surround61 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::BackCenter,
            ],
            Self::Surround70 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
            ],
            Self::Surround71 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
            ],
            Self::Atmos712 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
                ChannelPosition::TopFrontLeft,
                ChannelPosition::TopFrontRight,
            ],
            Self::Atmos714 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
                ChannelPosition::TopFrontLeft,
                ChannelPosition::TopFrontRight,
                ChannelPosition::TopBackLeft,
                ChannelPosition::TopBackRight,
            ],
            Self::Atmos916 => vec![
                ChannelPosition::FrontLeft,
                ChannelPosition::FrontRight,
                ChannelPosition::FrontCenter,
                ChannelPosition::Lfe,
                ChannelPosition::BackLeft,
                ChannelPosition::BackRight,
                ChannelPosition::SideLeft,
                ChannelPosition::SideRight,
                ChannelPosition::WideLeft,
                ChannelPosition::WideRight,
                ChannelPosition::TopFrontLeft,
                ChannelPosition::TopFrontRight,
                ChannelPosition::TopFrontCenter,
                ChannelPosition::TopBackLeft,
                ChannelPosition::TopBackRight,
                ChannelPosition::TopBackCenter,
            ],
        }
    }

    /// Check if this layout has an LFE channel.
    pub fn has_lfe(&self) -> bool {
        matches!(
            self,
            Self::Surround21
                | Self::Surround51
                | Self::Surround61
                | Self::Surround71
                | Self::Atmos712
                | Self::Atmos714
                | Self::Atmos916
        )
    }

    /// Check if this layout has height channels.
    pub fn has_height(&self) -> bool {
        matches!(self, Self::Atmos712 | Self::Atmos714 | Self::Atmos916)
    }

    /// Get the number of height channels.
    pub fn height_channel_count(&self) -> u32 {
        match self {
            Self::Atmos712 => 2,
            Self::Atmos714 => 4,
            Self::Atmos916 => 6,
            _ => 0,
        }
    }

    /// Create from channel count with best guess.
    pub fn from_channel_count(count: u32) -> Option<Self> {
        match count {
            1 => Some(Self::Mono),
            2 => Some(Self::Stereo),
            3 => Some(Self::Surround30),
            4 => Some(Self::Quad),
            5 => Some(Self::Surround50),
            6 => Some(Self::Surround51),
            7 => Some(Self::Surround70),
            8 => Some(Self::Surround71),
            10 => Some(Self::Atmos712),
            12 => Some(Self::Atmos714),
            16 => Some(Self::Atmos916),
            _ => None,
        }
    }
}

impl fmt::Display for StandardLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mono => write!(f, "Mono"),
            Self::Stereo => write!(f, "Stereo"),
            Self::Surround21 => write!(f, "2.1"),
            Self::Surround30 => write!(f, "3.0"),
            Self::Quad => write!(f, "Quad"),
            Self::Surround50 => write!(f, "5.0"),
            Self::Surround51 => write!(f, "5.1"),
            Self::Surround61 => write!(f, "6.1"),
            Self::Surround70 => write!(f, "7.0"),
            Self::Surround71 => write!(f, "7.1"),
            Self::Atmos712 => write!(f, "7.1.2 Atmos"),
            Self::Atmos714 => write!(f, "7.1.4 Atmos"),
            Self::Atmos916 => write!(f, "9.1.6 Atmos"),
        }
    }
}

/// Custom channel layout with arbitrary positions.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChannelLayout {
    /// Channel positions in order.
    positions: Vec<ChannelPosition>,
    /// Channel ordering standard.
    order: ChannelOrder,
    /// Layout name (if any).
    name: Option<String>,
}

impl ChannelLayout {
    /// Create a new custom channel layout.
    pub fn new(positions: Vec<ChannelPosition>) -> Self {
        Self {
            positions,
            order: ChannelOrder::Custom,
            name: None,
        }
    }

    /// Create from a standard layout.
    pub fn from_standard(layout: StandardLayout) -> Self {
        Self {
            positions: layout.positions(),
            order: ChannelOrder::Wav,
            name: Some(layout.to_string()),
        }
    }

    /// Create from a standard layout with specific channel order.
    pub fn from_standard_with_order(layout: StandardLayout, order: ChannelOrder) -> Self {
        let positions = match layout {
            StandardLayout::Surround51 => order.surround_51(),
            StandardLayout::Surround71 => order.surround_71(),
            _ => layout.positions(),
        };
        Self {
            positions,
            order,
            name: Some(format!("{} ({})", layout, order_name(order))),
        }
    }

    /// Get the channel count.
    pub fn channel_count(&self) -> u32 {
        self.positions.len() as u32
    }

    /// Get the channel positions.
    pub fn positions(&self) -> &[ChannelPosition] {
        &self.positions
    }

    /// Get the channel order.
    pub fn order(&self) -> ChannelOrder {
        self.order
    }

    /// Get the layout name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Set the layout name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Get the position at a specific index.
    pub fn position_at(&self, index: usize) -> Option<ChannelPosition> {
        self.positions.get(index).copied()
    }

    /// Find the index of a channel position.
    pub fn index_of(&self, position: ChannelPosition) -> Option<usize> {
        self.positions.iter().position(|p| *p == position)
    }

    /// Check if the layout contains a specific position.
    pub fn contains(&self, position: ChannelPosition) -> bool {
        self.positions.contains(&position)
    }

    /// Check if this layout has an LFE channel.
    pub fn has_lfe(&self) -> bool {
        self.positions
            .iter()
            .any(|p| matches!(p, ChannelPosition::Lfe | ChannelPosition::Lfe2))
    }

    /// Check if this layout has height channels.
    pub fn has_height(&self) -> bool {
        self.positions.iter().any(|p| p.is_height())
    }

    /// Get the number of height channels.
    pub fn height_channel_count(&self) -> u32 {
        self.positions.iter().filter(|p| p.is_height()).count() as u32
    }

    /// Convert to a different channel order.
    pub fn convert_order(&self, target_order: ChannelOrder) -> Result<Self> {
        if self.order == target_order {
            return Ok(self.clone());
        }

        // Create a mapping from current order to target order
        let mut new_positions = self.positions.clone();

        // For standard layouts, we can reorder
        if let Some(standard) = self.detect_standard_layout() {
            new_positions = match standard {
                StandardLayout::Surround51 => target_order.surround_51(),
                StandardLayout::Surround71 => target_order.surround_71(),
                _ => self.positions.clone(),
            };
        }

        Ok(Self {
            positions: new_positions,
            order: target_order,
            name: self.name.clone(),
        })
    }

    /// Detect if this matches a standard layout.
    pub fn detect_standard_layout(&self) -> Option<StandardLayout> {
        let count = self.channel_count();
        let standard = StandardLayout::from_channel_count(count)?;

        // Check if positions match (in any order)
        let standard_positions = standard.positions();
        if self.positions.len() != standard_positions.len() {
            return None;
        }

        let all_match = self
            .positions
            .iter()
            .all(|p| standard_positions.contains(p));
        if all_match {
            Some(standard)
        } else {
            None
        }
    }

    /// Create channel reorder mapping from this layout to target.
    pub fn reorder_map(&self, target: &ChannelLayout) -> Result<Vec<usize>> {
        if self.channel_count() != target.channel_count() {
            return Err(SpatialError::ChannelLayout(
                ChannelLayoutError::InvalidChannelCount {
                    count: target.channel_count(),
                    expected: self.channel_count(),
                },
            ));
        }

        let mut map = Vec::with_capacity(target.channel_count() as usize);
        for target_pos in &target.positions {
            match self.index_of(*target_pos) {
                Some(idx) => map.push(idx),
                None => {
                    return Err(SpatialError::ChannelLayout(
                        ChannelLayoutError::MissingChannel(target_pos.abbreviation().to_string()),
                    ));
                }
            }
        }

        Ok(map)
    }
}

impl Default for ChannelLayout {
    fn default() -> Self {
        Self::from_standard(StandardLayout::Stereo)
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "{}", name)
        } else {
            let pos_str: Vec<&str> = self.positions.iter().map(|p| p.abbreviation()).collect();
            write!(f, "{}", pos_str.join(", "))
        }
    }
}

/// Get display name for channel order.
fn order_name(order: ChannelOrder) -> &'static str {
    match order {
        ChannelOrder::Wav => "WAV",
        ChannelOrder::Smpte => "SMPTE",
        ChannelOrder::Dolby => "Dolby",
        ChannelOrder::Film => "Film",
        ChannelOrder::Aac => "AAC",
        ChannelOrder::Flac => "FLAC",
        ChannelOrder::Custom => "Custom",
    }
}

/// Reorder audio samples between channel layouts.
pub fn reorder_channels<T: Copy>(
    input: &[T],
    output: &mut [T],
    samples_per_channel: usize,
    channel_map: &[usize],
) -> Result<()> {
    let num_channels = channel_map.len();
    let expected_len = samples_per_channel * num_channels;

    if input.len() < expected_len || output.len() < expected_len {
        return Err(SpatialError::ChannelLayout(
            ChannelLayoutError::InvalidChannelCount {
                count: (input.len() / samples_per_channel) as u32,
                expected: num_channels as u32,
            },
        ));
    }

    // Interleaved format: reorder samples
    for sample_idx in 0..samples_per_channel {
        for (out_ch, &in_ch) in channel_map.iter().enumerate() {
            let in_idx = sample_idx * num_channels + in_ch;
            let out_idx = sample_idx * num_channels + out_ch;
            output[out_idx] = input[in_idx];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_position_abbreviation() {
        assert_eq!(ChannelPosition::FrontLeft.abbreviation(), "FL");
        assert_eq!(ChannelPosition::Lfe.abbreviation(), "LFE");
        assert_eq!(ChannelPosition::TopFrontLeft.abbreviation(), "TFL");
    }

    #[test]
    fn test_channel_position_from_abbreviation() {
        assert_eq!(
            ChannelPosition::from_abbreviation("FL"),
            Some(ChannelPosition::FrontLeft)
        );
        assert_eq!(
            ChannelPosition::from_abbreviation("L"),
            Some(ChannelPosition::FrontLeft)
        );
        assert_eq!(
            ChannelPosition::from_abbreviation("LFE"),
            Some(ChannelPosition::Lfe)
        );
        assert_eq!(ChannelPosition::from_abbreviation("INVALID"), None);
    }

    #[test]
    fn test_channel_position_3d() {
        let (az, el) = ChannelPosition::FrontCenter.position_degrees();
        assert!((az - 0.0).abs() < 0.001);
        assert!((el - 0.0).abs() < 0.001);

        let (az, el) = ChannelPosition::FrontLeft.position_degrees();
        assert!((az - 30.0).abs() < 0.001);
        assert!((el - 0.0).abs() < 0.001);

        let (az, el) = ChannelPosition::TopFrontLeft.position_degrees();
        assert!((az - 45.0).abs() < 0.001);
        assert!((el - 45.0).abs() < 0.001);
    }

    #[test]
    fn test_standard_layout_channel_count() {
        assert_eq!(StandardLayout::Mono.channel_count(), 1);
        assert_eq!(StandardLayout::Stereo.channel_count(), 2);
        assert_eq!(StandardLayout::Surround51.channel_count(), 6);
        assert_eq!(StandardLayout::Surround71.channel_count(), 8);
        assert_eq!(StandardLayout::Atmos714.channel_count(), 12);
    }

    #[test]
    fn test_standard_layout_positions() {
        let positions = StandardLayout::Surround51.positions();
        assert_eq!(positions.len(), 6);
        assert!(positions.contains(&ChannelPosition::FrontLeft));
        assert!(positions.contains(&ChannelPosition::Lfe));
    }

    #[test]
    fn test_channel_layout_from_standard() {
        let layout = ChannelLayout::from_standard(StandardLayout::Surround51);
        assert_eq!(layout.channel_count(), 6);
        assert!(layout.has_lfe());
        assert!(!layout.has_height());
    }

    #[test]
    fn test_channel_layout_atmos() {
        let layout = ChannelLayout::from_standard(StandardLayout::Atmos714);
        assert_eq!(layout.channel_count(), 12);
        assert!(layout.has_lfe());
        assert!(layout.has_height());
        assert_eq!(layout.height_channel_count(), 4);
    }

    #[test]
    fn test_channel_order_51() {
        let wav = ChannelOrder::Wav.surround_51();
        let dolby = ChannelOrder::Dolby.surround_51();

        // WAV order starts with FL, FR
        assert_eq!(wav[0], ChannelPosition::FrontLeft);
        assert_eq!(wav[1], ChannelPosition::FrontRight);

        // Dolby order starts with FC
        assert_eq!(dolby[0], ChannelPosition::FrontCenter);
    }

    #[test]
    fn test_reorder_map() {
        // Use SMPTE and FLAC which both use the same channel positions (just different order)
        let smpte_layout = ChannelLayout::from_standard_with_order(StandardLayout::Surround51, ChannelOrder::Smpte);
        let flac_layout = ChannelLayout::from_standard_with_order(StandardLayout::Surround51, ChannelOrder::Flac);

        // SMPTE: FL, FR, FC, LFE, SL, SR
        // FLAC:  FL, FR, FC, LFE, BL, BR
        // These have different surround positions, so let's just test with stereo
        let stereo1 = ChannelLayout::from_standard(StandardLayout::Stereo);
        let stereo2 = ChannelLayout::new(vec![ChannelPosition::FrontRight, ChannelPosition::FrontLeft]);

        let map = stereo1.reorder_map(&stereo2).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map[0], 1); // FR in stereo1 is at index 1
        assert_eq!(map[1], 0); // FL in stereo1 is at index 0
    }

    #[test]
    fn test_reorder_channels() {
        // Simple stereo reorder (swap L and R)
        let input = [1.0f32, 2.0, 3.0, 4.0]; // L1, R1, L2, R2
        let mut output = [0.0f32; 4];
        let channel_map = [1, 0]; // Swap channels

        reorder_channels(&input, &mut output, 2, &channel_map).unwrap();

        assert!((output[0] - 2.0).abs() < 0.001); // R1 -> first
        assert!((output[1] - 1.0).abs() < 0.001); // L1 -> second
        assert!((output[2] - 4.0).abs() < 0.001); // R2 -> first
        assert!((output[3] - 3.0).abs() < 0.001); // L2 -> second
    }

    #[test]
    fn test_detect_standard_layout() {
        let layout = ChannelLayout::from_standard(StandardLayout::Surround51);
        assert_eq!(layout.detect_standard_layout(), Some(StandardLayout::Surround51));

        let custom = ChannelLayout::new(vec![
            ChannelPosition::FrontLeft,
            ChannelPosition::FrontRight,
            ChannelPosition::TopCenter,
        ]);
        assert_eq!(custom.detect_standard_layout(), None);
    }
}
