//! VVC-specific syntax elements.
//!
//! This module defines VVC syntax structures including:
//! - Coding tree structure (CTU, CU, PU, TU)
//! - Transform units
//! - Prediction units
//! - VVC-specific partition modes

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

// Error types would be used for future parsing implementations
#[allow(unused_imports)]
use crate::error::{Result, VvcError};

/// VVC partition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitMode {
    /// No split (leaf node).
    #[default]
    NoSplit,
    /// Quad-tree split (4 equal parts).
    QtSplit,
    /// Binary tree horizontal split (2 parts, top/bottom).
    BtHorSplit,
    /// Binary tree vertical split (2 parts, left/right).
    BtVerSplit,
    /// Ternary tree horizontal split (3 parts, 1:2:1 ratio).
    TtHorSplit,
    /// Ternary tree vertical split (3 parts, 1:2:1 ratio).
    TtVerSplit,
}

impl SplitMode {
    /// Get number of child partitions.
    pub fn num_children(&self) -> usize {
        match self {
            Self::NoSplit => 0,
            Self::QtSplit => 4,
            Self::BtHorSplit | Self::BtVerSplit => 2,
            Self::TtHorSplit | Self::TtVerSplit => 3,
        }
    }

    /// Check if this is a binary tree split.
    pub fn is_bt(&self) -> bool {
        matches!(self, Self::BtHorSplit | Self::BtVerSplit)
    }

    /// Check if this is a ternary tree split.
    pub fn is_tt(&self) -> bool {
        matches!(self, Self::TtHorSplit | Self::TtVerSplit)
    }

    /// Check if this is a multi-type tree split (BT or TT).
    pub fn is_mtt(&self) -> bool {
        self.is_bt() || self.is_tt()
    }

    /// Check if this is a horizontal split.
    pub fn is_horizontal(&self) -> bool {
        matches!(self, Self::BtHorSplit | Self::TtHorSplit)
    }

    /// Check if this is a vertical split.
    pub fn is_vertical(&self) -> bool {
        matches!(self, Self::BtVerSplit | Self::TtVerSplit)
    }
}

/// Coding Tree Unit (CTU) data.
#[derive(Debug, Clone)]
pub struct CodingTreeUnit {
    /// CTU address (raster scan).
    pub ctu_address: u32,
    /// X position in picture (luma samples).
    pub x: u32,
    /// Y position in picture (luma samples).
    pub y: u32,
    /// CTU size (log2).
    pub log2_size: u8,
    /// CTU size in samples.
    pub size: u32,
    /// Root coding unit.
    pub cu: Option<Box<CodingUnit>>,
}

impl CodingTreeUnit {
    /// Create a new CTU.
    pub fn new(address: u32, x: u32, y: u32, log2_size: u8) -> Self {
        Self {
            ctu_address: address,
            x,
            y,
            log2_size,
            size: 1 << log2_size,
            cu: None,
        }
    }

    /// Get the number of CTUs required for the given picture dimensions.
    pub fn count_ctus(pic_width: u32, pic_height: u32, log2_ctu_size: u8) -> u32 {
        let ctu_size = 1u32 << log2_ctu_size;
        let ctus_x = (pic_width + ctu_size - 1) / ctu_size;
        let ctus_y = (pic_height + ctu_size - 1) / ctu_size;
        ctus_x * ctus_y
    }
}

/// Coding Unit (CU) data.
#[derive(Debug, Clone)]
pub struct CodingUnit {
    /// X position.
    pub x: u32,
    /// Y position.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Log2 of width.
    pub log2_width: u8,
    /// Log2 of height.
    pub log2_height: u8,
    /// Depth in the coding tree.
    pub depth: u8,
    /// QT depth.
    pub qt_depth: u8,
    /// MTT depth.
    pub mtt_depth: u8,
    /// Split mode used to create this CU (NoSplit for leaf).
    pub split_mode: SplitMode,
    /// Is skip mode.
    pub cu_skip_flag: bool,
    /// Prediction mode.
    pub pred_mode: PredMode,
    /// Intra prediction mode for luma.
    pub intra_luma_mpm_flag: bool,
    /// Intra luma prediction mode.
    pub intra_luma_mode: u8,
    /// Intra chroma prediction mode.
    pub intra_chroma_mode: u8,
    /// MIP (Matrix Intra Prediction) flag.
    pub intra_mip_flag: bool,
    /// MIP mode.
    pub intra_mip_mode: u8,
    /// MIP transpose flag.
    pub intra_mip_transposed_flag: bool,
    /// ISP (Intra Sub-Partitions) mode.
    pub isp_mode: IspMode,
    /// BDPCM mode.
    pub bdpcm_mode: BdpcmMode,
    /// Inter merge flag.
    pub merge_flag: bool,
    /// Merge mode.
    pub merge_mode: MergeMode,
    /// Merge index.
    pub merge_idx: u8,
    /// MMVD (Merge Mode with Motion Vector Difference) flag.
    pub mmvd_flag: bool,
    /// MMVD merge flag.
    pub mmvd_merge_flag: bool,
    /// CIIP (Combined Inter Intra Prediction) flag.
    pub ciip_flag: bool,
    /// GPM (Geometric Partitioning Mode) flag.
    pub gpm_flag: bool,
    /// GPM split direction.
    pub gpm_split_dir: u8,
    /// GPM merge indices.
    pub gpm_merge_idx: [u8; 2],
    /// Motion vector L0.
    pub mv_l0: MotionVector,
    /// Motion vector L1.
    pub mv_l1: MotionVector,
    /// Reference index L0.
    pub ref_idx_l0: i8,
    /// Reference index L1.
    pub ref_idx_l1: i8,
    /// MVD (Motion Vector Difference) L0.
    pub mvd_l0: MotionVector,
    /// MVD L1.
    pub mvd_l1: MotionVector,
    /// Affine flag.
    pub affine_flag: bool,
    /// Affine type.
    pub affine_type: u8,
    /// Affine control point MVs.
    pub affine_cpmv: [MotionVector; 3],
    /// SBT (Sub-Block Transform) flag.
    pub sbt_flag: bool,
    /// SBT horizontal flag.
    pub sbt_horizontal_flag: bool,
    /// SBT pos flag.
    pub sbt_pos_flag: bool,
    /// BCW (Bi-prediction with CU-level Weights) index.
    pub bcw_idx: u8,
    /// Root coded block flag for luma.
    pub cu_cbf_y: bool,
    /// Root coded block flag for Cb.
    pub cu_cbf_cb: bool,
    /// Root coded block flag for Cr.
    pub cu_cbf_cr: bool,
    /// Joint CbCr flag.
    pub joint_cbcr_flag: bool,
    /// QP for this CU.
    pub qp: i8,
    /// QP for Cb.
    pub qp_cb: i8,
    /// QP for Cr.
    pub qp_cr: i8,
    /// Transform units.
    pub transform_units: Vec<TransformUnit>,
    /// Child CUs (if split).
    pub children: Vec<CodingUnit>,
}

impl Default for CodingUnit {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            log2_width: 0,
            log2_height: 0,
            depth: 0,
            qt_depth: 0,
            mtt_depth: 0,
            split_mode: SplitMode::NoSplit,
            cu_skip_flag: false,
            pred_mode: PredMode::Intra,
            intra_luma_mpm_flag: false,
            intra_luma_mode: 0,
            intra_chroma_mode: 0,
            intra_mip_flag: false,
            intra_mip_mode: 0,
            intra_mip_transposed_flag: false,
            isp_mode: IspMode::NoIsp,
            bdpcm_mode: BdpcmMode::None,
            merge_flag: false,
            merge_mode: MergeMode::Regular,
            merge_idx: 0,
            mmvd_flag: false,
            mmvd_merge_flag: false,
            ciip_flag: false,
            gpm_flag: false,
            gpm_split_dir: 0,
            gpm_merge_idx: [0, 0],
            mv_l0: MotionVector::default(),
            mv_l1: MotionVector::default(),
            ref_idx_l0: -1,
            ref_idx_l1: -1,
            mvd_l0: MotionVector::default(),
            mvd_l1: MotionVector::default(),
            affine_flag: false,
            affine_type: 0,
            affine_cpmv: [MotionVector::default(); 3],
            sbt_flag: false,
            sbt_horizontal_flag: false,
            sbt_pos_flag: false,
            bcw_idx: 0,
            cu_cbf_y: false,
            cu_cbf_cb: false,
            cu_cbf_cr: false,
            joint_cbcr_flag: false,
            qp: 0,
            qp_cb: 0,
            qp_cr: 0,
            transform_units: Vec::new(),
            children: Vec::new(),
        }
    }
}

impl CodingUnit {
    /// Create a new coding unit.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        let log2_width = (width as f32).log2() as u8;
        let log2_height = (height as f32).log2() as u8;
        Self {
            x,
            y,
            width,
            height,
            log2_width,
            log2_height,
            ..Default::default()
        }
    }

    /// Check if this is a leaf CU (no further split).
    pub fn is_leaf(&self) -> bool {
        self.split_mode == SplitMode::NoSplit
    }

    /// Check if this CU uses intra prediction.
    pub fn is_intra(&self) -> bool {
        self.pred_mode == PredMode::Intra
    }

    /// Check if this CU uses inter prediction.
    pub fn is_inter(&self) -> bool {
        self.pred_mode == PredMode::Inter
    }

    /// Check if this CU uses IBC (Intra Block Copy).
    pub fn is_ibc(&self) -> bool {
        self.pred_mode == PredMode::Ibc
    }

    /// Get the area of this CU.
    pub fn area(&self) -> u32 {
        self.width * self.height
    }

    /// Check if CU is square.
    pub fn is_square(&self) -> bool {
        self.width == self.height
    }

    /// Get split child positions and sizes.
    pub fn get_split_children(&self, split: SplitMode) -> Vec<(u32, u32, u32, u32)> {
        match split {
            SplitMode::NoSplit => vec![(self.x, self.y, self.width, self.height)],
            SplitMode::QtSplit => {
                let hw = self.width / 2;
                let hh = self.height / 2;
                vec![
                    (self.x, self.y, hw, hh),
                    (self.x + hw, self.y, hw, hh),
                    (self.x, self.y + hh, hw, hh),
                    (self.x + hw, self.y + hh, hw, hh),
                ]
            }
            SplitMode::BtHorSplit => {
                let hh = self.height / 2;
                vec![
                    (self.x, self.y, self.width, hh),
                    (self.x, self.y + hh, self.width, hh),
                ]
            }
            SplitMode::BtVerSplit => {
                let hw = self.width / 2;
                vec![
                    (self.x, self.y, hw, self.height),
                    (self.x + hw, self.y, hw, self.height),
                ]
            }
            SplitMode::TtHorSplit => {
                let h1 = self.height / 4;
                let h2 = self.height / 2;
                vec![
                    (self.x, self.y, self.width, h1),
                    (self.x, self.y + h1, self.width, h2),
                    (self.x, self.y + h1 + h2, self.width, h1),
                ]
            }
            SplitMode::TtVerSplit => {
                let w1 = self.width / 4;
                let w2 = self.width / 2;
                vec![
                    (self.x, self.y, w1, self.height),
                    (self.x + w1, self.y, w2, self.height),
                    (self.x + w1 + w2, self.y, w1, self.height),
                ]
            }
        }
    }
}

/// Prediction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PredMode {
    /// Intra prediction.
    #[default]
    Intra,
    /// Inter prediction.
    Inter,
    /// Intra Block Copy (IBC).
    Ibc,
    /// Palette mode.
    Palette,
}

/// ISP (Intra Sub-Partitions) mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IspMode {
    /// No ISP.
    #[default]
    NoIsp,
    /// Horizontal ISP.
    Horizontal,
    /// Vertical ISP.
    Vertical,
}

impl IspMode {
    /// Get number of sub-partitions.
    pub fn num_partitions(&self, width: u32, height: u32) -> u32 {
        match self {
            Self::NoIsp => 1,
            Self::Horizontal => {
                if height <= 8 { 2 } else { 4 }
            }
            Self::Vertical => {
                if width <= 8 { 2 } else { 4 }
            }
        }
    }

    /// Create from ISP mode value.
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::NoIsp),
            1 => Some(Self::Horizontal),
            2 => Some(Self::Vertical),
            _ => None,
        }
    }
}

/// BDPCM (Block-based Delta Pulse Code Modulation) mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BdpcmMode {
    /// No BDPCM.
    #[default]
    None,
    /// Horizontal BDPCM.
    Horizontal,
    /// Vertical BDPCM.
    Vertical,
}

impl BdpcmMode {
    /// Create from BDPCM mode value.
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::None),
            1 => Some(Self::Horizontal),
            2 => Some(Self::Vertical),
            _ => None,
        }
    }
}

/// Merge mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeMode {
    /// Regular merge.
    #[default]
    Regular,
    /// MMVD (Merge with MVD).
    Mmvd,
    /// Subblock merge (affine).
    Subblock,
    /// CIIP (Combined Inter-Intra Prediction).
    Ciip,
    /// GPM (Geometric Partitioning Mode).
    Gpm,
}

/// Motion vector.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotionVector {
    /// Horizontal component (1/16 pel).
    pub x: i16,
    /// Vertical component (1/16 pel).
    pub y: i16,
}

impl MotionVector {
    /// Create a new motion vector.
    pub fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Create a zero motion vector.
    pub fn zero() -> Self {
        Self { x: 0, y: 0 }
    }

    /// Add two motion vectors.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    /// Subtract two motion vectors.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    /// Scale motion vector.
    pub fn scale(&self, factor: i32, shift: u8) -> Self {
        Self {
            x: ((self.x as i32 * factor + (1 << (shift - 1))) >> shift) as i16,
            y: ((self.y as i32 * factor + (1 << (shift - 1))) >> shift) as i16,
        }
    }

    /// Round motion vector to specified precision.
    pub fn round(&self, shift: u8) -> Self {
        let add = 1 << (shift - 1);
        Self {
            x: ((self.x as i32 + add) >> shift) as i16,
            y: ((self.y as i32 + add) >> shift) as i16,
        }
    }

    /// Check if motion vector is zero.
    pub fn is_zero(&self) -> bool {
        self.x == 0 && self.y == 0
    }
}

/// Transform Unit (TU) data.
#[derive(Debug, Clone, Default)]
pub struct TransformUnit {
    /// X position.
    pub x: u32,
    /// Y position.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Log2 of width.
    pub log2_width: u8,
    /// Log2 of height.
    pub log2_height: u8,
    /// Depth in transform tree.
    pub depth: u8,
    /// Coded block flag for luma.
    pub cbf_luma: bool,
    /// Coded block flag for Cb.
    pub cbf_cb: bool,
    /// Coded block flag for Cr.
    pub cbf_cr: bool,
    /// Joint CbCr residual flag.
    pub tu_joint_cbcr_residual_flag: bool,
    /// Transform skip flag for luma.
    pub transform_skip_y: bool,
    /// Transform skip flag for Cb.
    pub transform_skip_cb: bool,
    /// Transform skip flag for Cr.
    pub transform_skip_cr: bool,
    /// MTS index.
    pub mts_idx: u8,
    /// LFNST index.
    pub lfnst_idx: u8,
    /// Luma coefficients.
    pub coeff_y: Vec<i32>,
    /// Cb coefficients.
    pub coeff_cb: Vec<i32>,
    /// Cr coefficients.
    pub coeff_cr: Vec<i32>,
    /// Last significant coefficient X position.
    pub last_sig_coeff_x_luma: u16,
    /// Last significant coefficient Y position.
    pub last_sig_coeff_y_luma: u16,
    /// Last significant coefficient X position for chroma.
    pub last_sig_coeff_x_chroma: u16,
    /// Last significant coefficient Y position for chroma.
    pub last_sig_coeff_y_chroma: u16,
}

impl TransformUnit {
    /// Create a new transform unit.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        let log2_width = (width as f32).log2() as u8;
        let log2_height = (height as f32).log2() as u8;
        Self {
            x,
            y,
            width,
            height,
            log2_width,
            log2_height,
            coeff_y: Vec::with_capacity((width * height) as usize),
            coeff_cb: Vec::new(),
            coeff_cr: Vec::new(),
            ..Default::default()
        }
    }

    /// Get the number of coefficients.
    pub fn num_coeffs(&self) -> usize {
        (self.width * self.height) as usize
    }

    /// Check if this TU has any non-zero coefficients.
    pub fn has_residual(&self) -> bool {
        self.cbf_luma || self.cbf_cb || self.cbf_cr
    }

    /// Get transform size class.
    pub fn size_class(&self) -> TransformSizeClass {
        match (self.width, self.height) {
            (2, 2) => TransformSizeClass::Size2x2,
            (4, 4) => TransformSizeClass::Size4x4,
            (8, 8) => TransformSizeClass::Size8x8,
            (16, 16) => TransformSizeClass::Size16x16,
            (32, 32) => TransformSizeClass::Size32x32,
            (64, 64) => TransformSizeClass::Size64x64,
            (4, 8) | (8, 4) => TransformSizeClass::Size4x8,
            (8, 16) | (16, 8) => TransformSizeClass::Size8x16,
            (16, 32) | (32, 16) => TransformSizeClass::Size16x32,
            (32, 64) | (64, 32) => TransformSizeClass::Size32x64,
            (4, 16) | (16, 4) => TransformSizeClass::Size4x16,
            (8, 32) | (32, 8) => TransformSizeClass::Size8x32,
            (16, 64) | (64, 16) => TransformSizeClass::Size16x64,
            (2, 8) | (8, 2) => TransformSizeClass::Size2x8,
            (2, 16) | (16, 2) => TransformSizeClass::Size2x16,
            (2, 32) | (32, 2) => TransformSizeClass::Size2x32,
            (4, 32) | (32, 4) => TransformSizeClass::Size4x32,
            (2, 4) | (4, 2) => TransformSizeClass::Size2x4,
            (2, 64) | (64, 2) => TransformSizeClass::Size2x64,
            (4, 64) | (64, 4) => TransformSizeClass::Size4x64,
            _ => TransformSizeClass::Other,
        }
    }
}

/// Transform size class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformSizeClass {
    /// 2x2 transform.
    Size2x2,
    /// 4x4 transform.
    Size4x4,
    /// 8x8 transform.
    Size8x8,
    /// 16x16 transform.
    Size16x16,
    /// 32x32 transform.
    Size32x32,
    /// 64x64 transform.
    Size64x64,
    /// 4x8 or 8x4 transform.
    Size4x8,
    /// 8x16 or 16x8 transform.
    Size8x16,
    /// 16x32 or 32x16 transform.
    Size16x32,
    /// 32x64 or 64x32 transform.
    Size32x64,
    /// 4x16 or 16x4 transform.
    Size4x16,
    /// 8x32 or 32x8 transform.
    Size8x32,
    /// 16x64 or 64x16 transform.
    Size16x64,
    /// 2x4 or 4x2 transform.
    Size2x4,
    /// 2x8 or 8x2 transform.
    Size2x8,
    /// 2x16 or 16x2 transform.
    Size2x16,
    /// 2x32 or 32x2 transform.
    Size2x32,
    /// 2x64 or 64x2 transform.
    Size2x64,
    /// 4x32 or 32x4 transform.
    Size4x32,
    /// 4x64 or 64x4 transform.
    Size4x64,
    /// Other non-standard size.
    Other,
}

/// VVC intra prediction modes (67 modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraMode {
    /// Planar (mode 0).
    Planar,
    /// DC (mode 1).
    Dc,
    /// Angular mode (2-66).
    Angular(u8),
}

impl IntraMode {
    /// Total number of intra modes in VVC.
    pub const NUM_MODES: u8 = 67;

    /// Create from mode index.
    pub fn from_index(idx: u8) -> Self {
        match idx {
            0 => Self::Planar,
            1 => Self::Dc,
            _ => Self::Angular(idx),
        }
    }

    /// Get the mode index.
    pub fn index(&self) -> u8 {
        match self {
            Self::Planar => 0,
            Self::Dc => 1,
            Self::Angular(idx) => *idx,
        }
    }

    /// Check if this is a diagonal mode.
    pub fn is_diagonal(&self) -> bool {
        match self {
            Self::Angular(idx) => *idx == 2 || *idx == 34 || *idx == 66,
            _ => false,
        }
    }

    /// Check if this is a horizontal-ish mode.
    pub fn is_horizontal(&self) -> bool {
        match self {
            Self::Angular(idx) => *idx >= 2 && *idx <= 34,
            _ => false,
        }
    }

    /// Check if this is a vertical-ish mode.
    pub fn is_vertical(&self) -> bool {
        match self {
            Self::Angular(idx) => *idx >= 34 && *idx <= 66,
            _ => false,
        }
    }

    /// Get the angle offset for this mode.
    pub fn angle_offset(&self) -> i8 {
        match self {
            Self::Planar | Self::Dc => 0,
            Self::Angular(idx) => {
                let base = if *idx <= 34 { 18i8 - *idx as i8 } else { *idx as i8 - 50 };
                base * 2
            }
        }
    }
}

/// VVC intra chroma prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraChromaMode {
    /// Planar.
    Planar,
    /// Vertical.
    Vertical,
    /// Horizontal.
    Horizontal,
    /// DC.
    Dc,
    /// LM (Linear Model).
    Lm,
    /// LM-L (Linear Model Left).
    LmL,
    /// LM-T (Linear Model Top).
    LmT,
    /// Derived from luma (DM).
    Dm,
}

impl IntraChromaMode {
    /// Create from mode value.
    pub fn from_u8(val: u8, has_lm: bool) -> Option<Self> {
        if has_lm {
            match val {
                0 => Some(Self::Planar),
                1 => Some(Self::Vertical),
                2 => Some(Self::Horizontal),
                3 => Some(Self::Dc),
                4 => Some(Self::Lm),
                5 => Some(Self::LmL),
                6 => Some(Self::LmT),
                7 => Some(Self::Dm),
                _ => None,
            }
        } else {
            match val {
                0 => Some(Self::Planar),
                1 => Some(Self::Vertical),
                2 => Some(Self::Horizontal),
                3 => Some(Self::Dc),
                4 => Some(Self::Dm),
                _ => None,
            }
        }
    }

    /// Check if this is a cross-component LM mode.
    pub fn is_lm(&self) -> bool {
        matches!(self, Self::Lm | Self::LmL | Self::LmT)
    }
}

/// MIP (Matrix-based Intra Prediction) configuration.
#[derive(Debug, Clone)]
pub struct MipConfig {
    /// MIP size class.
    pub size_class: MipSizeClass,
    /// Number of MIP modes available.
    pub num_modes: u8,
    /// Upsampling factor horizontal.
    pub up_hor: u8,
    /// Upsampling factor vertical.
    pub up_ver: u8,
    /// Boundary size.
    pub boundary_size: u8,
}

impl MipConfig {
    /// Create MIP configuration for given block size.
    pub fn for_size(width: u32, height: u32) -> Self {
        let size_class = MipSizeClass::from_size(width, height);
        let (num_modes, up_hor, up_ver, boundary_size) = match size_class {
            MipSizeClass::Size4x4 => (16, 1, 1, 2),
            MipSizeClass::Size4x8 | MipSizeClass::Size8x4 => (8, 2, 2, 2),
            MipSizeClass::Size8x8 => (6, 2, 2, 4),
            MipSizeClass::Large => (6, 4, 4, 4),
        };
        Self {
            size_class,
            num_modes,
            up_hor,
            up_ver,
            boundary_size,
        }
    }
}

/// MIP size class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipSizeClass {
    /// 4x4 blocks.
    Size4x4,
    /// 4x8 blocks.
    Size4x8,
    /// 8x4 blocks.
    Size8x4,
    /// 8x8 blocks.
    Size8x8,
    /// Large blocks (> 8x8).
    Large,
}

impl MipSizeClass {
    /// Determine MIP size class from dimensions.
    pub fn from_size(width: u32, height: u32) -> Self {
        match (width, height) {
            (4, 4) => Self::Size4x4,
            (4, 8) => Self::Size4x8,
            (8, 4) => Self::Size8x4,
            (8, 8) => Self::Size8x8,
            _ => Self::Large,
        }
    }
}

/// ALF (Adaptive Loop Filter) filter shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlfFilterShape {
    /// 5x5 diamond filter.
    #[default]
    Diamond5x5,
    /// 7x7 diamond filter.
    Diamond7x7,
}

/// SAO (Sample Adaptive Offset) parameters.
#[derive(Debug, Clone, Default)]
pub struct SaoParams {
    /// SAO merge left flag.
    pub merge_left: bool,
    /// SAO merge up flag.
    pub merge_up: bool,
    /// SAO type for each component (0=off, 1=band, 2=edge).
    pub type_idx: [u8; 3],
    /// SAO offsets.
    pub offset: [[i8; 4]; 3],
    /// SAO band position.
    pub band_position: [u8; 3],
    /// SAO edge class.
    pub eo_class: [u8; 3],
}

/// ALF (Adaptive Loop Filter) parameters for a CTU.
#[derive(Debug, Clone, Default)]
pub struct AlfCtuParams {
    /// ALF enabled for luma.
    pub alf_ctb_flag_luma: bool,
    /// ALF enabled for Cb.
    pub alf_ctb_flag_cb: bool,
    /// ALF enabled for Cr.
    pub alf_ctb_flag_cr: bool,
    /// ALF filter alternative luma.
    pub alf_ctb_filter_alt_idx_luma: u8,
    /// ALF filter alternative Cb.
    pub alf_ctb_filter_alt_idx_cb: u8,
    /// ALF filter alternative Cr.
    pub alf_ctb_filter_alt_idx_cr: u8,
    /// CC-ALF enabled for Cb.
    pub cc_alf_ctb_flag_cb: bool,
    /// CC-ALF enabled for Cr.
    pub cc_alf_ctb_flag_cr: bool,
    /// CC-ALF filter index for Cb.
    pub cc_alf_filter_idx_cb: u8,
    /// CC-ALF filter index for Cr.
    pub cc_alf_filter_idx_cr: u8,
}

/// LMCS (Luma Mapping with Chroma Scaling) data.
#[derive(Debug, Clone, Default)]
pub struct LmcsData {
    /// LMCS min bin index.
    pub lmcs_min_bin_idx: u8,
    /// LMCS delta max bin index.
    pub lmcs_delta_max_bin_idx: u8,
    /// LMCS delta CW prec minus 1.
    pub lmcs_delta_cw_prec_minus1: u8,
    /// LMCS delta absolute CW.
    pub lmcs_delta_abs_cw: Vec<u16>,
    /// LMCS delta sign CW flag.
    pub lmcs_delta_sign_cw_flag: Vec<bool>,
    /// LMCS delta absolute CRS.
    pub lmcs_delta_abs_crs: u8,
    /// LMCS delta sign CRS flag.
    pub lmcs_delta_sign_crs_flag: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_mode() {
        assert_eq!(SplitMode::NoSplit.num_children(), 0);
        assert_eq!(SplitMode::QtSplit.num_children(), 4);
        assert_eq!(SplitMode::BtHorSplit.num_children(), 2);
        assert_eq!(SplitMode::TtVerSplit.num_children(), 3);

        assert!(SplitMode::BtHorSplit.is_bt());
        assert!(SplitMode::TtHorSplit.is_tt());
        assert!(SplitMode::BtVerSplit.is_mtt());
        assert!(!SplitMode::QtSplit.is_mtt());
    }

    #[test]
    fn test_coding_unit_split() {
        let cu = CodingUnit::new(0, 0, 64, 64);

        let qt_children = cu.get_split_children(SplitMode::QtSplit);
        assert_eq!(qt_children.len(), 4);
        assert_eq!(qt_children[0], (0, 0, 32, 32));
        assert_eq!(qt_children[3], (32, 32, 32, 32));

        let bt_children = cu.get_split_children(SplitMode::BtHorSplit);
        assert_eq!(bt_children.len(), 2);
        assert_eq!(bt_children[0], (0, 0, 64, 32));
        assert_eq!(bt_children[1], (0, 32, 64, 32));

        let tt_children = cu.get_split_children(SplitMode::TtVerSplit);
        assert_eq!(tt_children.len(), 3);
        assert_eq!(tt_children[0], (0, 0, 16, 64));
        assert_eq!(tt_children[1], (16, 0, 32, 64));
        assert_eq!(tt_children[2], (48, 0, 16, 64));
    }

    #[test]
    fn test_motion_vector() {
        let mv1 = MotionVector::new(10, 20);
        let mv2 = MotionVector::new(5, -10);

        let sum = mv1.add(&mv2);
        assert_eq!(sum.x, 15);
        assert_eq!(sum.y, 10);

        let diff = mv1.sub(&mv2);
        assert_eq!(diff.x, 5);
        assert_eq!(diff.y, 30);

        assert!(MotionVector::zero().is_zero());
        assert!(!mv1.is_zero());
    }

    #[test]
    fn test_intra_mode() {
        assert_eq!(IntraMode::Planar.index(), 0);
        assert_eq!(IntraMode::Dc.index(), 1);
        assert_eq!(IntraMode::Angular(34).index(), 34);

        assert!(IntraMode::Angular(2).is_diagonal());
        assert!(IntraMode::Angular(10).is_horizontal());
        assert!(IntraMode::Angular(50).is_vertical());
    }

    #[test]
    fn test_transform_unit() {
        let tu = TransformUnit::new(0, 0, 16, 16);
        assert_eq!(tu.num_coeffs(), 256);
        assert!(!tu.has_residual());

        let mut tu2 = TransformUnit::new(0, 0, 8, 8);
        tu2.cbf_luma = true;
        assert!(tu2.has_residual());
    }

    #[test]
    fn test_isp_mode() {
        assert_eq!(IspMode::NoIsp.num_partitions(16, 16), 1);
        assert_eq!(IspMode::Horizontal.num_partitions(16, 8), 2);
        assert_eq!(IspMode::Horizontal.num_partitions(16, 16), 4);
        assert_eq!(IspMode::Vertical.num_partitions(8, 16), 2);
        assert_eq!(IspMode::Vertical.num_partitions(16, 16), 4);
    }

    #[test]
    fn test_mip_config() {
        let config_4x4 = MipConfig::for_size(4, 4);
        assert_eq!(config_4x4.size_class, MipSizeClass::Size4x4);
        assert_eq!(config_4x4.num_modes, 16);

        let config_8x8 = MipConfig::for_size(8, 8);
        assert_eq!(config_8x8.size_class, MipSizeClass::Size8x8);
        assert_eq!(config_8x8.num_modes, 6);

        let config_16x16 = MipConfig::for_size(16, 16);
        assert_eq!(config_16x16.size_class, MipSizeClass::Large);
    }

    #[test]
    fn test_ctu_count() {
        // 1920x1080 with 128x128 CTUs: 15 x 9 = 135
        assert_eq!(CodingTreeUnit::count_ctus(1920, 1080, 7), 135);
        // 1920x1080 with 64x64 CTUs: 30 x 17 = 510
        assert_eq!(CodingTreeUnit::count_ctus(1920, 1080, 6), 510);
    }
}
