//! VP9 intra and inter prediction.
//!
//! This module implements VP9 prediction modes:
//! - Intra prediction modes (DC, V, H, D45, D135, D117, D153, D207, D63, TM)
//! - Inter prediction with reference frames (LAST, GOLDEN, ALTREF)
//! - Transform types (DCT, ADST, identity)

use crate::frame_header::InterpFilter;

/// VP9 block sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockSize {
    /// 4x4 block.
    Block4x4 = 0,
    /// 4x8 block.
    Block4x8 = 1,
    /// 8x4 block.
    Block8x4 = 2,
    /// 8x8 block.
    Block8x8 = 3,
    /// 8x16 block.
    Block8x16 = 4,
    /// 16x8 block.
    Block16x8 = 5,
    /// 16x16 block.
    Block16x16 = 6,
    /// 16x32 block.
    Block16x32 = 7,
    /// 32x16 block.
    Block32x16 = 8,
    /// 32x32 block.
    Block32x32 = 9,
    /// 32x64 block.
    Block32x64 = 10,
    /// 64x32 block.
    Block64x32 = 11,
    /// 64x64 block (superblock).
    Block64x64 = 12,
}

impl BlockSize {
    /// Get the width of this block size.
    pub const fn width(&self) -> u32 {
        match self {
            Self::Block4x4 | Self::Block4x8 => 4,
            Self::Block8x4 | Self::Block8x8 | Self::Block8x16 => 8,
            Self::Block16x8 | Self::Block16x16 | Self::Block16x32 => 16,
            Self::Block32x16 | Self::Block32x32 | Self::Block32x64 => 32,
            Self::Block64x32 | Self::Block64x64 => 64,
        }
    }

    /// Get the height of this block size.
    pub const fn height(&self) -> u32 {
        match self {
            Self::Block4x4 | Self::Block8x4 => 4,
            Self::Block4x8 | Self::Block8x8 | Self::Block16x8 => 8,
            Self::Block8x16 | Self::Block16x16 | Self::Block32x16 => 16,
            Self::Block16x32 | Self::Block32x32 | Self::Block64x32 => 32,
            Self::Block32x64 | Self::Block64x64 => 64,
        }
    }

    /// Get width in mode info blocks (8x8).
    pub const fn mi_width(&self) -> u32 {
        self.width() >> 3
    }

    /// Get height in mode info blocks (8x8).
    pub const fn mi_height(&self) -> u32 {
        self.height() >> 3
    }

    /// Check if this is a square block.
    pub const fn is_square(&self) -> bool {
        matches!(
            self,
            Self::Block4x4
                | Self::Block8x8
                | Self::Block16x16
                | Self::Block32x32
                | Self::Block64x64
        )
    }
}

/// VP9 partition type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Partition {
    /// No partition.
    None = 0,
    /// Horizontal split.
    Horizontal = 1,
    /// Vertical split.
    Vertical = 2,
    /// Four-way split.
    Split = 3,
}

/// VP9 intra prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum IntraMode {
    /// DC prediction (average of neighbors).
    #[default]
    Dc = 0,
    /// Vertical prediction.
    V = 1,
    /// Horizontal prediction.
    H = 2,
    /// Diagonal 45 degrees prediction.
    D45 = 3,
    /// Diagonal 135 degrees prediction.
    D135 = 4,
    /// Diagonal 117 degrees prediction.
    D117 = 5,
    /// Diagonal 153 degrees prediction.
    D153 = 6,
    /// Diagonal 207 degrees prediction.
    D207 = 7,
    /// Diagonal 63 degrees prediction.
    D63 = 8,
    /// True motion prediction.
    Tm = 9,
}

impl IntraMode {
    /// Check if this mode needs the above neighbors.
    pub const fn needs_above(&self) -> bool {
        !matches!(self, Self::H | Self::D207)
    }

    /// Check if this mode needs the left neighbors.
    pub const fn needs_left(&self) -> bool {
        !matches!(self, Self::V | Self::D45 | Self::D63)
    }

    /// Check if this mode needs the above-left neighbor.
    pub const fn needs_above_left(&self) -> bool {
        matches!(
            self,
            Self::D135 | Self::D117 | Self::D153 | Self::Tm
        )
    }
}

/// VP9 inter prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum InterMode {
    /// Zero motion vector.
    #[default]
    ZeroMv = 0,
    /// Nearest motion vector.
    NearestMv = 1,
    /// Near motion vector.
    NearMv = 2,
    /// New motion vector.
    NewMv = 3,
}

/// Motion vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    /// Row component (quarter-pel).
    pub row: i16,
    /// Column component (quarter-pel).
    pub col: i16,
}

impl MotionVector {
    /// Create a new motion vector.
    pub const fn new(row: i16, col: i16) -> Self {
        Self { row, col }
    }

    /// Create a zero motion vector.
    pub const fn zero() -> Self {
        Self { row: 0, col: 0 }
    }

    /// Add two motion vectors.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            row: self.row.saturating_add(other.row),
            col: self.col.saturating_add(other.col),
        }
    }

    /// Scale motion vector.
    pub fn scale(&self, num: i32, den: i32) -> Self {
        if den == 0 {
            return Self::zero();
        }
        Self {
            row: ((self.row as i32 * num + den / 2) / den) as i16,
            col: ((self.col as i32 * num + den / 2) / den) as i16,
        }
    }

    /// Check if this is a zero vector.
    pub const fn is_zero(&self) -> bool {
        self.row == 0 && self.col == 0
    }
}

/// Transform type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TxType {
    /// DCT in both directions.
    #[default]
    DctDct = 0,
    /// ADST horizontal, DCT vertical.
    AdstDct = 1,
    /// DCT horizontal, ADST vertical.
    DctAdst = 2,
    /// ADST in both directions.
    AdstAdst = 3,
}

/// Transform size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TxSize {
    /// 4x4 transform.
    #[default]
    Tx4x4 = 0,
    /// 8x8 transform.
    Tx8x8 = 1,
    /// 16x16 transform.
    Tx16x16 = 2,
    /// 32x32 transform.
    Tx32x32 = 3,
}

impl TxSize {
    /// Get the size in pixels.
    pub const fn size(&self) -> u32 {
        match self {
            Self::Tx4x4 => 4,
            Self::Tx8x8 => 8,
            Self::Tx16x16 => 16,
            Self::Tx32x32 => 32,
        }
    }

    /// Get the log2 of size.
    pub const fn log2(&self) -> u8 {
        match self {
            Self::Tx4x4 => 2,
            Self::Tx8x8 => 3,
            Self::Tx16x16 => 4,
            Self::Tx32x32 => 5,
        }
    }
}

/// VP9 intra predictor.
pub struct IntraPredictor;

impl IntraPredictor {
    /// Predict a block using intra prediction.
    #[allow(clippy::too_many_arguments)]
    pub fn predict(
        dst: &mut [u8],
        dst_stride: usize,
        above: &[u8],
        left: &[u8],
        above_left: u8,
        mode: IntraMode,
        width: usize,
        height: usize,
        have_above: bool,
        have_left: bool,
    ) {
        match mode {
            IntraMode::Dc => {
                Self::predict_dc(dst, dst_stride, above, left, width, height, have_above, have_left)
            }
            IntraMode::V => Self::predict_v(dst, dst_stride, above, width, height),
            IntraMode::H => Self::predict_h(dst, dst_stride, left, width, height),
            IntraMode::D45 => Self::predict_d45(dst, dst_stride, above, width, height),
            IntraMode::D135 => {
                Self::predict_d135(dst, dst_stride, above, left, above_left, width, height)
            }
            IntraMode::D117 => {
                Self::predict_d117(dst, dst_stride, above, left, above_left, width, height)
            }
            IntraMode::D153 => {
                Self::predict_d153(dst, dst_stride, above, left, above_left, width, height)
            }
            IntraMode::D207 => Self::predict_d207(dst, dst_stride, left, width, height),
            IntraMode::D63 => Self::predict_d63(dst, dst_stride, above, width, height),
            IntraMode::Tm => {
                Self::predict_tm(dst, dst_stride, above, left, above_left, width, height)
            }
        }
    }

    fn predict_dc(
        dst: &mut [u8],
        dst_stride: usize,
        above: &[u8],
        left: &[u8],
        width: usize,
        height: usize,
        have_above: bool,
        have_left: bool,
    ) {
        let dc = match (have_above, have_left) {
            (true, true) => {
                let sum: u32 = above[..width].iter().map(|&x| x as u32).sum::<u32>()
                    + left[..height].iter().map(|&x| x as u32).sum::<u32>();
                let count = (width + height) as u32;
                ((sum + count / 2) / count) as u8
            }
            (true, false) => {
                let sum: u32 = above[..width].iter().map(|&x| x as u32).sum();
                ((sum + width as u32 / 2) / width as u32) as u8
            }
            (false, true) => {
                let sum: u32 = left[..height].iter().map(|&x| x as u32).sum();
                ((sum + height as u32 / 2) / height as u32) as u8
            }
            (false, false) => 128,
        };

        for row in 0..height {
            for col in 0..width {
                dst[row * dst_stride + col] = dc;
            }
        }
    }

    fn predict_v(dst: &mut [u8], dst_stride: usize, above: &[u8], width: usize, height: usize) {
        for row in 0..height {
            dst[row * dst_stride..row * dst_stride + width].copy_from_slice(&above[..width]);
        }
    }

    fn predict_h(dst: &mut [u8], dst_stride: usize, left: &[u8], width: usize, height: usize) {
        for row in 0..height {
            let val = left[row];
            for col in 0..width {
                dst[row * dst_stride + col] = val;
            }
        }
    }

    fn predict_d45(dst: &mut [u8], dst_stride: usize, above: &[u8], width: usize, height: usize) {
        for row in 0..height {
            for col in 0..width {
                let idx = row + col;
                if idx + 1 < above.len() && idx + 2 < above.len() {
                    let val = (above[idx] as u32 + 2 * above[idx + 1] as u32 + above[idx + 2] as u32 + 2) >> 2;
                    dst[row * dst_stride + col] = val as u8;
                } else if idx < above.len() {
                    dst[row * dst_stride + col] = above[above.len() - 1];
                } else {
                    dst[row * dst_stride + col] = 128;
                }
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn predict_d135(
        dst: &mut [u8],
        dst_stride: usize,
        above: &[u8],
        left: &[u8],
        above_left: u8,
        width: usize,
        height: usize,
    ) {
        // Build extended reference array
        let mut ref_buf = vec![0u8; width + height + 1];

        // Fill with left samples (reversed), above-left, and above samples
        for i in 0..height {
            ref_buf[height - 1 - i] = left.get(i).copied().unwrap_or(128);
        }
        ref_buf[height] = above_left;
        for i in 0..width {
            ref_buf[height + 1 + i] = above.get(i).copied().unwrap_or(128);
        }

        for row in 0..height {
            for col in 0..width {
                let idx = height + col - row;
                if idx > 0 && idx + 1 < ref_buf.len() {
                    let val = (ref_buf[idx - 1] as u32 + 2 * ref_buf[idx] as u32 + ref_buf[idx + 1] as u32 + 2) >> 2;
                    dst[row * dst_stride + col] = val as u8;
                } else {
                    dst[row * dst_stride + col] = 128;
                }
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn predict_d117(
        dst: &mut [u8],
        dst_stride: usize,
        above: &[u8],
        left: &[u8],
        above_left: u8,
        width: usize,
        height: usize,
    ) {
        // Simplified D117 prediction
        for row in 0..height {
            for col in 0..width {
                let r = row as i32;
                let c = col as i32;
                let idx = c - (r >> 1);

                if idx >= 0 && (idx as usize) < width {
                    dst[row * dst_stride + col] = above[idx as usize];
                } else if idx == -1 {
                    dst[row * dst_stride + col] = above_left;
                } else if (-idx - 1) < height as i32 {
                    dst[row * dst_stride + col] = left[(-idx - 2) as usize];
                } else {
                    dst[row * dst_stride + col] = 128;
                }
            }
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn predict_d153(
        dst: &mut [u8],
        dst_stride: usize,
        above: &[u8],
        left: &[u8],
        above_left: u8,
        width: usize,
        height: usize,
    ) {
        // Simplified D153 prediction
        for row in 0..height {
            for col in 0..width {
                let r = row as i32;
                let c = col as i32;
                let idx = r - (c >> 1);

                if idx >= 0 && (idx as usize) < height {
                    dst[row * dst_stride + col] = left[idx as usize];
                } else if idx == -1 {
                    dst[row * dst_stride + col] = above_left;
                } else if (-idx - 1) < width as i32 {
                    dst[row * dst_stride + col] = above[(-idx - 2) as usize];
                } else {
                    dst[row * dst_stride + col] = 128;
                }
            }
        }
    }

    fn predict_d207(dst: &mut [u8], dst_stride: usize, left: &[u8], width: usize, height: usize) {
        for row in 0..height {
            for col in 0..width {
                let idx = row + (col >> 1);
                if idx < height && idx + 1 < left.len() {
                    let val = if col & 1 == 0 {
                        (left[idx] as u32 + left[idx + 1] as u32 + 1) >> 1
                    } else {
                        (left[idx] as u32 + 2 * left[idx + 1] as u32 + left.get(idx + 2).copied().unwrap_or(left[left.len() - 1]) as u32 + 2) >> 2
                    };
                    dst[row * dst_stride + col] = val as u8;
                } else if idx < left.len() {
                    dst[row * dst_stride + col] = left[left.len() - 1];
                } else {
                    dst[row * dst_stride + col] = 128;
                }
            }
        }
    }

    fn predict_d63(dst: &mut [u8], dst_stride: usize, above: &[u8], width: usize, height: usize) {
        for row in 0..height {
            for col in 0..width {
                let idx = col + (row >> 1);
                if idx < width && idx + 1 < above.len() {
                    let val = if row & 1 == 0 {
                        (above[idx] as u32 + above[idx + 1] as u32 + 1) >> 1
                    } else {
                        (above[idx] as u32 + 2 * above[idx + 1] as u32 + above.get(idx + 2).copied().unwrap_or(above[above.len() - 1]) as u32 + 2) >> 2
                    };
                    dst[row * dst_stride + col] = val as u8;
                } else if idx < above.len() {
                    dst[row * dst_stride + col] = above[above.len() - 1];
                } else {
                    dst[row * dst_stride + col] = 128;
                }
            }
        }
    }

    fn predict_tm(
        dst: &mut [u8],
        dst_stride: usize,
        above: &[u8],
        left: &[u8],
        above_left: u8,
        width: usize,
        height: usize,
    ) {
        for row in 0..height {
            for col in 0..width {
                let val = above.get(col).copied().unwrap_or(128) as i32
                    + left.get(row).copied().unwrap_or(128) as i32
                    - above_left as i32;
                dst[row * dst_stride + col] = val.clamp(0, 255) as u8;
            }
        }
    }
}

/// VP9 inter predictor.
pub struct InterPredictor;

impl InterPredictor {
    /// 8-tap filter coefficients for subpixel interpolation.
    const SUBPEL_FILTERS_8TAP: [[i16; 8]; 16] = [
        [0, 0, 0, 128, 0, 0, 0, 0],      // 0/16
        [0, 1, -5, 126, 8, -3, 1, 0],    // 1/16
        [-1, 3, -10, 122, 18, -6, 2, 0], // 2/16
        [-1, 4, -13, 118, 27, -9, 3, -1],// 3/16
        [-1, 4, -16, 112, 37, -11, 4, -1],// 4/16
        [-1, 5, -18, 105, 48, -14, 4, -1],// 5/16
        [-1, 5, -19, 97, 58, -16, 5, -1],// 6/16
        [-1, 6, -19, 88, 68, -18, 5, -1],// 7/16
        [-1, 6, -19, 78, 78, -19, 6, -1],// 8/16
        [-1, 5, -18, 68, 88, -19, 6, -1],// 9/16
        [-1, 5, -16, 58, 97, -19, 5, -1],// 10/16
        [-1, 4, -14, 48, 105, -18, 5, -1],// 11/16
        [-1, 4, -11, 37, 112, -16, 4, -1],// 12/16
        [-1, 3, -9, 27, 118, -13, 4, -1],// 13/16
        [0, 2, -6, 18, 122, -10, 3, -1], // 14/16
        [0, 1, -3, 8, 126, -5, 1, 0],    // 15/16
    ];

    /// 8-tap smooth filter coefficients.
    const SUBPEL_FILTERS_8TAP_SMOOTH: [[i16; 8]; 16] = [
        [0, 0, 0, 128, 0, 0, 0, 0],
        [-3, -1, 32, 64, 38, 1, -3, 0],
        [-2, -2, 29, 63, 41, 2, -3, 0],
        [-2, -2, 26, 63, 44, 3, -4, 0],
        [-2, -3, 24, 62, 46, 4, -4, 1],
        [-2, -3, 21, 60, 49, 6, -4, 1],
        [-1, -4, 18, 59, 51, 7, -4, 2],
        [-1, -4, 16, 57, 53, 9, -4, 2],
        [-1, -4, 14, 55, 55, 14, -4, -1],
        [2, -4, 9, 53, 57, 16, -4, -1],
        [2, -4, 7, 51, 59, 18, -4, -1],
        [1, -4, 6, 49, 60, 21, -3, -2],
        [1, -4, 4, 46, 62, 24, -3, -2],
        [0, -4, 3, 44, 63, 26, -2, -2],
        [0, -3, 2, 41, 63, 29, -2, -2],
        [0, -3, 1, 38, 64, 32, -1, -3],
    ];

    /// Bilinear filter coefficients.
    const SUBPEL_FILTERS_BILINEAR: [[i16; 2]; 16] = [
        [128, 0],
        [120, 8],
        [112, 16],
        [104, 24],
        [96, 32],
        [88, 40],
        [80, 48],
        [72, 56],
        [64, 64],
        [56, 72],
        [48, 80],
        [40, 88],
        [32, 96],
        [24, 104],
        [16, 112],
        [8, 120],
    ];

    /// Perform motion compensation.
    #[allow(clippy::too_many_arguments)]
    pub fn motion_compensate(
        dst: &mut [u8],
        dst_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        ref_width: usize,
        ref_height: usize,
        mv: MotionVector,
        block_width: usize,
        block_height: usize,
        x: usize,
        y: usize,
        filter: InterpFilter,
    ) {
        let frac_x = (mv.col & 15) as usize;
        let frac_y = (mv.row & 15) as usize;
        let int_x = x as i32 + (mv.col >> 4) as i32;
        let int_y = y as i32 + (mv.row >> 4) as i32;

        if frac_x == 0 && frac_y == 0 {
            // Integer-pel copy
            Self::copy_block(
                dst, dst_stride,
                ref_frame, ref_stride,
                ref_width, ref_height,
                int_x, int_y,
                block_width, block_height,
            );
        } else if filter == InterpFilter::Bilinear {
            Self::interpolate_bilinear(
                dst, dst_stride,
                ref_frame, ref_stride,
                ref_width, ref_height,
                int_x, int_y,
                block_width, block_height,
                frac_x, frac_y,
            );
        } else {
            Self::interpolate_8tap(
                dst, dst_stride,
                ref_frame, ref_stride,
                ref_width, ref_height,
                int_x, int_y,
                block_width, block_height,
                frac_x, frac_y,
                filter,
            );
        }
    }

    fn copy_block(
        dst: &mut [u8],
        dst_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        ref_width: usize,
        ref_height: usize,
        int_x: i32,
        int_y: i32,
        block_width: usize,
        block_height: usize,
    ) {
        for row in 0..block_height {
            let src_y = (int_y + row as i32).clamp(0, ref_height as i32 - 1) as usize;
            for col in 0..block_width {
                let src_x = (int_x + col as i32).clamp(0, ref_width as i32 - 1) as usize;
                dst[row * dst_stride + col] = ref_frame[src_y * ref_stride + src_x];
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn interpolate_bilinear(
        dst: &mut [u8],
        dst_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        ref_width: usize,
        ref_height: usize,
        int_x: i32,
        int_y: i32,
        block_width: usize,
        block_height: usize,
        frac_x: usize,
        frac_y: usize,
    ) {
        let fx = Self::SUBPEL_FILTERS_BILINEAR[frac_x];
        let fy = Self::SUBPEL_FILTERS_BILINEAR[frac_y];

        for row in 0..block_height {
            for col in 0..block_width {
                let mut sum = 0i32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_x = (int_x + col as i32 + dx).clamp(0, ref_width as i32 - 1) as usize;
                        let src_y = (int_y + row as i32 + dy).clamp(0, ref_height as i32 - 1) as usize;
                        let pixel = ref_frame[src_y * ref_stride + src_x] as i32;
                        sum += pixel * fx[dx as usize] as i32 * fy[dy as usize] as i32;
                    }
                }
                dst[row * dst_stride + col] = ((sum + 8192) >> 14).clamp(0, 255) as u8;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn interpolate_8tap(
        dst: &mut [u8],
        dst_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        ref_width: usize,
        ref_height: usize,
        int_x: i32,
        int_y: i32,
        block_width: usize,
        block_height: usize,
        frac_x: usize,
        frac_y: usize,
        filter: InterpFilter,
    ) {
        let filters = match filter {
            InterpFilter::EightTapSmooth => &Self::SUBPEL_FILTERS_8TAP_SMOOTH,
            _ => &Self::SUBPEL_FILTERS_8TAP,
        };

        let fx = filters[frac_x];
        let fy = filters[frac_y];

        // Temporary buffer for horizontal filtering
        let temp_height = block_height + 7;
        let mut temp = vec![0i16; block_width * temp_height];

        // Horizontal filter
        for row in 0..temp_height {
            let src_y = (int_y + row as i32 - 3).clamp(0, ref_height as i32 - 1) as usize;
            for col in 0..block_width {
                let mut sum = 0i32;
                for tap in 0..8 {
                    let src_x = (int_x + col as i32 + tap as i32 - 3).clamp(0, ref_width as i32 - 1) as usize;
                    let pixel = ref_frame[src_y * ref_stride + src_x] as i32;
                    sum += pixel * fx[tap] as i32;
                }
                temp[row * block_width + col] = ((sum + 64) >> 7) as i16;
            }
        }

        // Vertical filter
        for row in 0..block_height {
            for col in 0..block_width {
                let mut sum = 0i32;
                for tap in 0..8 {
                    let temp_row = row + tap;
                    let pixel = temp[temp_row * block_width + col] as i32;
                    sum += pixel * fy[tap] as i32;
                }
                dst[row * dst_stride + col] = ((sum + 64) >> 7).clamp(0, 255) as u8;
            }
        }
    }

    /// Compute average of two prediction blocks.
    pub fn average_blocks(
        dst: &mut [u8],
        dst_stride: usize,
        src1: &[u8],
        src1_stride: usize,
        src2: &[u8],
        src2_stride: usize,
        width: usize,
        height: usize,
    ) {
        for row in 0..height {
            for col in 0..width {
                let a = src1[row * src1_stride + col] as u16;
                let b = src2[row * src2_stride + col] as u16;
                dst[row * dst_stride + col] = ((a + b + 1) >> 1) as u8;
            }
        }
    }
}

/// Loop filter implementation.
pub struct LoopFilter;

impl LoopFilter {
    /// Apply loop filter to a vertical edge (4 pixels wide).
    #[allow(clippy::too_many_arguments)]
    pub fn filter_vertical_edge(
        data: &mut [u8],
        stride: usize,
        x: usize,
        y: usize,
        height: usize,
        filter_level: u8,
        sharpness: u8,
    ) {
        if filter_level == 0 {
            return;
        }

        let limit = Self::calculate_limit(filter_level, sharpness);
        let thresh = Self::calculate_thresh(filter_level);

        for row in 0..height {
            let idx = (y + row) * stride + x;
            if idx >= 4 && idx + 4 <= data.len() {
                Self::filter_4_pixels(&mut data[idx - 4..idx + 4], 1, limit, thresh);
            }
        }
    }

    /// Apply loop filter to a horizontal edge (4 pixels tall).
    #[allow(clippy::too_many_arguments)]
    pub fn filter_horizontal_edge(
        data: &mut [u8],
        stride: usize,
        x: usize,
        y: usize,
        width: usize,
        filter_level: u8,
        sharpness: u8,
    ) {
        if filter_level == 0 {
            return;
        }

        let limit = Self::calculate_limit(filter_level, sharpness);
        let thresh = Self::calculate_thresh(filter_level);

        for col in 0..width {
            if y >= 4 {
                let mut pixels = [0u8; 8];
                for i in 0..8 {
                    let row = y - 4 + i;
                    pixels[i] = data[row * stride + x + col];
                }

                Self::filter_4_pixels(&mut pixels, 1, limit, thresh);

                for i in 0..8 {
                    let row = y - 4 + i;
                    data[row * stride + x + col] = pixels[i];
                }
            }
        }
    }

    fn calculate_limit(filter_level: u8, sharpness: u8) -> u8 {
        let shift = if sharpness > 4 { 2 } else if sharpness > 0 { 1 } else { 0 };
        let limit = (filter_level >> shift) as i32;
        let limit = if sharpness > 0 {
            limit.min(9 - sharpness as i32)
        } else {
            limit
        };
        limit.max(1) as u8
    }

    fn calculate_thresh(filter_level: u8) -> u8 {
        (filter_level >> 4) + 1
    }

    fn filter_4_pixels(pixels: &mut [u8], _step: usize, limit: u8, thresh: u8) {
        // Simplified 4-tap filter
        let p1 = pixels[2] as i32;
        let p0 = pixels[3] as i32;
        let q0 = pixels[4] as i32;
        let q1 = pixels[5] as i32;

        // Check filter mask
        let mask = (p1 - p0).abs() <= thresh as i32
            && (q1 - q0).abs() <= thresh as i32
            && (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 <= limit as i32;

        if !mask {
            return;
        }

        // Apply filter
        let filter = (p0 - q0).clamp(-128, 127);
        let filter = (3 * filter + 4) >> 3;

        pixels[3] = (p0 - filter).clamp(0, 255) as u8;
        pixels[4] = (q0 + filter).clamp(0, 255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_size_dimensions() {
        assert_eq!(BlockSize::Block4x4.width(), 4);
        assert_eq!(BlockSize::Block4x4.height(), 4);
        assert_eq!(BlockSize::Block64x64.width(), 64);
        assert_eq!(BlockSize::Block64x64.height(), 64);
        assert_eq!(BlockSize::Block8x16.width(), 8);
        assert_eq!(BlockSize::Block8x16.height(), 16);
    }

    #[test]
    fn test_block_size_mi() {
        assert_eq!(BlockSize::Block8x8.mi_width(), 1);
        assert_eq!(BlockSize::Block8x8.mi_height(), 1);
        assert_eq!(BlockSize::Block64x64.mi_width(), 8);
        assert_eq!(BlockSize::Block64x64.mi_height(), 8);
    }

    #[test]
    fn test_block_size_is_square() {
        assert!(BlockSize::Block4x4.is_square());
        assert!(BlockSize::Block64x64.is_square());
        assert!(!BlockSize::Block4x8.is_square());
        assert!(!BlockSize::Block32x64.is_square());
    }

    #[test]
    fn test_intra_mode_needs() {
        assert!(IntraMode::V.needs_above());
        assert!(!IntraMode::V.needs_left());
        assert!(IntraMode::H.needs_left());
        assert!(!IntraMode::H.needs_above());
        assert!(IntraMode::Tm.needs_above_left());
    }

    #[test]
    fn test_motion_vector() {
        let mv1 = MotionVector::new(4, 8);
        let mv2 = MotionVector::new(2, 4);
        let sum = mv1.add(&mv2);
        assert_eq!(sum.row, 6);
        assert_eq!(sum.col, 12);
    }

    #[test]
    fn test_motion_vector_scale() {
        let mv = MotionVector::new(10, 20);
        let scaled = mv.scale(2, 1);
        assert_eq!(scaled.row, 20);
        assert_eq!(scaled.col, 40);
    }

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector::zero();
        assert!(mv.is_zero());
        assert_eq!(mv.row, 0);
        assert_eq!(mv.col, 0);
    }

    #[test]
    fn test_tx_size() {
        assert_eq!(TxSize::Tx4x4.size(), 4);
        assert_eq!(TxSize::Tx4x4.log2(), 2);
        assert_eq!(TxSize::Tx32x32.size(), 32);
        assert_eq!(TxSize::Tx32x32.log2(), 5);
    }

    #[test]
    fn test_intra_dc_prediction() {
        let above = [100u8; 8];
        let left = [100u8; 8];
        let mut dst = [0u8; 64];

        IntraPredictor::predict(
            &mut dst, 8,
            &above, &left, 100,
            IntraMode::Dc,
            8, 8,
            true, true,
        );

        assert_eq!(dst[0], 100);
        assert_eq!(dst[63], 100);
    }

    #[test]
    fn test_intra_v_prediction() {
        let above = [50, 100, 150, 200, 50, 100, 150, 200];
        let left = [0u8; 8];
        let mut dst = [0u8; 64];

        IntraPredictor::predict(
            &mut dst, 8,
            &above, &left, 0,
            IntraMode::V,
            8, 8,
            true, true,
        );

        // All rows should be same as above
        assert_eq!(dst[0], 50);
        assert_eq!(dst[1], 100);
        assert_eq!(dst[56], 50); // Last row, first col
    }

    #[test]
    fn test_intra_h_prediction() {
        let above = [0u8; 8];
        let left = [50, 100, 150, 200, 50, 100, 150, 200];
        let mut dst = [0u8; 64];

        IntraPredictor::predict(
            &mut dst, 8,
            &above, &left, 0,
            IntraMode::H,
            8, 8,
            true, true,
        );

        // All columns in row i should equal left[i]
        assert_eq!(dst[0], 50);
        assert_eq!(dst[7], 50);
        assert_eq!(dst[8], 100);
        assert_eq!(dst[15], 100);
    }

    #[test]
    fn test_loop_filter_limit() {
        let limit = LoopFilter::calculate_limit(32, 0);
        assert!(limit > 0);

        let limit_sharp = LoopFilter::calculate_limit(32, 7);
        assert!(limit_sharp <= limit);
    }

    #[test]
    fn test_inter_average_blocks() {
        let src1 = [100u8; 16];
        let src2 = [200u8; 16];
        let mut dst = [0u8; 16];

        InterPredictor::average_blocks(
            &mut dst, 4,
            &src1, 4,
            &src2, 4,
            4, 4,
        );

        assert_eq!(dst[0], 150);
        assert_eq!(dst[15], 150);
    }
}
