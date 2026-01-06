//! VP8 intra and inter prediction.

use crate::frame::Vp8Frame;

/// Intra 4x4 prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Intra4x4Mode {
    /// DC prediction (average of above and left).
    #[default]
    DcPred = 0,
    /// True motion prediction.
    TmPred = 1,
    /// Vertical prediction.
    VePred = 2,
    /// Horizontal prediction.
    HePred = 3,
    /// Left-down diagonal prediction.
    LdPred = 4,
    /// Right-down diagonal prediction.
    RdPred = 5,
    /// Vertical-right prediction.
    VrPred = 6,
    /// Vertical-left prediction.
    VlPred = 7,
    /// Horizontal-down prediction.
    HdPred = 8,
    /// Horizontal-up prediction.
    HuPred = 9,
}

impl Intra4x4Mode {
    /// Create from raw value.
    pub fn from_raw(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::DcPred),
            1 => Some(Self::TmPred),
            2 => Some(Self::VePred),
            3 => Some(Self::HePred),
            4 => Some(Self::LdPred),
            5 => Some(Self::RdPred),
            6 => Some(Self::VrPred),
            7 => Some(Self::VlPred),
            8 => Some(Self::HdPred),
            9 => Some(Self::HuPred),
            _ => None,
        }
    }
}

/// Intra 16x16 (luma) prediction modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Intra16x16Mode {
    /// DC prediction.
    #[default]
    DcPred = 0,
    /// Vertical prediction.
    VPred = 1,
    /// Horizontal prediction.
    HPred = 2,
    /// TrueMotion prediction.
    TmPred = 3,
}

impl Intra16x16Mode {
    /// Create from raw value.
    pub fn from_raw(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::DcPred),
            1 => Some(Self::VPred),
            2 => Some(Self::HPred),
            3 => Some(Self::TmPred),
            _ => None,
        }
    }
}

/// Intra chroma prediction modes (same as 16x16).
pub type IntraChromaMode = Intra16x16Mode;

/// Prediction context for a macroblock.
#[derive(Debug, Clone)]
pub struct PredictionContext {
    /// Above pixels (17 pixels for 16x16 block + 1 corner).
    pub above: [u8; 17],
    /// Left pixels (16 pixels for 16x16 block).
    pub left: [u8; 16],
    /// Above-left corner pixel.
    pub above_left: u8,
    /// Whether above pixels are available.
    pub above_available: bool,
    /// Whether left pixels are available.
    pub left_available: bool,
}

impl Default for PredictionContext {
    fn default() -> Self {
        Self {
            above: [128u8; 17],
            left: [128u8; 16],
            above_left: 128,
            above_available: false,
            left_available: false,
        }
    }
}

impl PredictionContext {
    /// Create context from frame at macroblock position.
    pub fn from_frame(frame: &Vp8Frame, mb_x: usize, mb_y: usize) -> Self {
        let mut ctx = Self::default();
        let y_stride = frame.y_stride;
        let y_data = frame.y_data();

        let px = mb_x * 16;
        let py = mb_y * 16;

        // Above pixels available if not at top edge
        ctx.above_available = mb_y > 0;
        if ctx.above_available {
            let above_row = (py - 1) * y_stride + px;
            for i in 0..16 {
                if above_row + i < y_data.len() {
                    ctx.above[i + 1] = y_data[above_row + i];
                }
            }
        }

        // Left pixels available if not at left edge
        ctx.left_available = mb_x > 0;
        if ctx.left_available {
            for i in 0..16 {
                let idx = (py + i) * y_stride + px - 1;
                if idx < y_data.len() {
                    ctx.left[i] = y_data[idx];
                }
            }
        }

        // Above-left corner
        if ctx.above_available && ctx.left_available {
            let idx = (py - 1) * y_stride + px - 1;
            if idx < y_data.len() {
                ctx.above_left = y_data[idx];
                ctx.above[0] = ctx.above_left;
            }
        }

        ctx
    }
}

/// Perform 16x16 DC prediction.
pub fn predict_16x16_dc(ctx: &PredictionContext, output: &mut [u8; 256]) {
    let dc = if ctx.above_available && ctx.left_available {
        let sum_above: u32 = ctx.above[1..17].iter().map(|&x| x as u32).sum();
        let sum_left: u32 = ctx.left.iter().map(|&x| x as u32).sum();
        ((sum_above + sum_left + 16) >> 5) as u8
    } else if ctx.above_available {
        let sum: u32 = ctx.above[1..17].iter().map(|&x| x as u32).sum();
        ((sum + 8) >> 4) as u8
    } else if ctx.left_available {
        let sum: u32 = ctx.left.iter().map(|&x| x as u32).sum();
        ((sum + 8) >> 4) as u8
    } else {
        128
    };

    output.fill(dc);
}

/// Perform 16x16 vertical prediction.
pub fn predict_16x16_v(ctx: &PredictionContext, output: &mut [u8; 256]) {
    for y in 0..16 {
        for x in 0..16 {
            output[y * 16 + x] = ctx.above[x + 1];
        }
    }
}

/// Perform 16x16 horizontal prediction.
pub fn predict_16x16_h(ctx: &PredictionContext, output: &mut [u8; 256]) {
    for y in 0..16 {
        for x in 0..16 {
            output[y * 16 + x] = ctx.left[y];
        }
    }
}

/// Perform 16x16 TrueMotion prediction.
pub fn predict_16x16_tm(ctx: &PredictionContext, output: &mut [u8; 256]) {
    let tl = ctx.above_left as i16;
    for y in 0..16 {
        for x in 0..16 {
            let pred = ctx.above[x + 1] as i16 + ctx.left[y] as i16 - tl;
            output[y * 16 + x] = pred.clamp(0, 255) as u8;
        }
    }
}

/// Perform 16x16 intra prediction.
pub fn predict_16x16(mode: Intra16x16Mode, ctx: &PredictionContext, output: &mut [u8; 256]) {
    match mode {
        Intra16x16Mode::DcPred => predict_16x16_dc(ctx, output),
        Intra16x16Mode::VPred => predict_16x16_v(ctx, output),
        Intra16x16Mode::HPred => predict_16x16_h(ctx, output),
        Intra16x16Mode::TmPred => predict_16x16_tm(ctx, output),
    }
}

/// 4x4 prediction context.
#[derive(Debug, Clone)]
pub struct Pred4x4Context {
    /// Above pixels (4 + 4 for diagonal modes).
    pub above: [u8; 8],
    /// Left pixels (4).
    pub left: [u8; 4],
    /// Above-left corner.
    pub above_left: u8,
    /// Whether above is available.
    pub above_available: bool,
    /// Whether left is available.
    pub left_available: bool,
    /// Whether above-right is available.
    pub above_right_available: bool,
}

impl Default for Pred4x4Context {
    fn default() -> Self {
        Self {
            above: [128u8; 8],
            left: [128u8; 4],
            above_left: 128,
            above_available: false,
            left_available: false,
            above_right_available: false,
        }
    }
}

/// Perform 4x4 DC prediction.
pub fn predict_4x4_dc(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    let dc = if ctx.above_available && ctx.left_available {
        let sum_above: u32 = ctx.above[0..4].iter().map(|&x| x as u32).sum();
        let sum_left: u32 = ctx.left.iter().map(|&x| x as u32).sum();
        ((sum_above + sum_left + 4) >> 3) as u8
    } else if ctx.above_available {
        let sum: u32 = ctx.above[0..4].iter().map(|&x| x as u32).sum();
        ((sum + 2) >> 2) as u8
    } else if ctx.left_available {
        let sum: u32 = ctx.left.iter().map(|&x| x as u32).sum();
        ((sum + 2) >> 2) as u8
    } else {
        128
    };

    output.fill(dc);
}

/// Perform 4x4 TrueMotion prediction.
pub fn predict_4x4_tm(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    let tl = ctx.above_left as i16;
    for y in 0..4 {
        for x in 0..4 {
            let pred = ctx.above[x] as i16 + ctx.left[y] as i16 - tl;
            output[y * 4 + x] = pred.clamp(0, 255) as u8;
        }
    }
}

/// Perform 4x4 vertical prediction.
pub fn predict_4x4_ve(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    // Smooth vertical prediction
    let mut above_smoothed = [0u8; 4];

    above_smoothed[0] = ((ctx.above_left as u16 + 2 * ctx.above[0] as u16 + ctx.above[1] as u16 + 2) >> 2) as u8;
    above_smoothed[1] = ((ctx.above[0] as u16 + 2 * ctx.above[1] as u16 + ctx.above[2] as u16 + 2) >> 2) as u8;
    above_smoothed[2] = ((ctx.above[1] as u16 + 2 * ctx.above[2] as u16 + ctx.above[3] as u16 + 2) >> 2) as u8;
    above_smoothed[3] = ((ctx.above[2] as u16 + 2 * ctx.above[3] as u16 + ctx.above[4] as u16 + 2) >> 2) as u8;

    for y in 0..4 {
        for x in 0..4 {
            output[y * 4 + x] = above_smoothed[x];
        }
    }
}

/// Perform 4x4 horizontal prediction.
pub fn predict_4x4_he(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    // Smooth horizontal prediction
    let mut left_smoothed = [0u8; 4];

    left_smoothed[0] = ((ctx.above_left as u16 + 2 * ctx.left[0] as u16 + ctx.left[1] as u16 + 2) >> 2) as u8;
    left_smoothed[1] = ((ctx.left[0] as u16 + 2 * ctx.left[1] as u16 + ctx.left[2] as u16 + 2) >> 2) as u8;
    left_smoothed[2] = ((ctx.left[1] as u16 + 2 * ctx.left[2] as u16 + ctx.left[3] as u16 + 2) >> 2) as u8;
    left_smoothed[3] = ((ctx.left[2] as u16 + 2 * ctx.left[3] as u16 + ctx.left[3] as u16 + 2) >> 2) as u8;

    for y in 0..4 {
        for x in 0..4 {
            output[y * 4 + x] = left_smoothed[y];
        }
    }
}

/// Perform 4x4 left-down diagonal prediction.
pub fn predict_4x4_ld(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    let a = &ctx.above;

    // Pre-compute filtered values
    let mut f = [0u8; 8];
    for i in 0..7 {
        f[i] = ((a[i] as u16 + 2 * a[i + 1] as u16 + a.get(i + 2).copied().unwrap_or(a[7]) as u16 + 2) >> 2) as u8;
    }
    f[7] = a[7];

    output[0] = f[0]; output[1] = f[1]; output[2] = f[2]; output[3] = f[3];
    output[4] = f[1]; output[5] = f[2]; output[6] = f[3]; output[7] = f[4];
    output[8] = f[2]; output[9] = f[3]; output[10] = f[4]; output[11] = f[5];
    output[12] = f[3]; output[13] = f[4]; output[14] = f[5]; output[15] = f[6];
}

/// Perform 4x4 right-down diagonal prediction.
pub fn predict_4x4_rd(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    let a = &ctx.above;
    let l = &ctx.left;
    let tl = ctx.above_left;

    // Build prediction samples
    let x = |i: i32| -> u8 {
        if i >= 0 {
            a[i as usize]
        } else if i == -1 {
            tl
        } else {
            l[(-i - 2) as usize]
        }
    };

    // Filtered values
    let f = |i: i32| -> u8 {
        ((x(i - 1) as u16 + 2 * x(i) as u16 + x(i + 1) as u16 + 2) >> 2) as u8
    };

    output[0] = f(-1); output[1] = f(0); output[2] = f(1); output[3] = f(2);
    output[4] = f(-2); output[5] = f(-1); output[6] = f(0); output[7] = f(1);
    output[8] = f(-3); output[9] = f(-2); output[10] = f(-1); output[11] = f(0);
    output[12] = f(-4); output[13] = f(-3); output[14] = f(-2); output[15] = f(-1);
}

/// Perform 4x4 vertical-right prediction.
pub fn predict_4x4_vr(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    let a = &ctx.above;
    let l = &ctx.left;
    let tl = ctx.above_left;

    let avg2 = |x: u8, y: u8| ((x as u16 + y as u16 + 1) >> 1) as u8;
    let avg3 = |x: u8, y: u8, z: u8| ((x as u16 + 2 * y as u16 + z as u16 + 2) >> 2) as u8;

    output[0] = avg2(tl, a[0]);
    output[1] = avg2(a[0], a[1]);
    output[2] = avg2(a[1], a[2]);
    output[3] = avg2(a[2], a[3]);
    output[4] = avg3(l[0], tl, a[0]);
    output[5] = avg3(tl, a[0], a[1]);
    output[6] = avg3(a[0], a[1], a[2]);
    output[7] = avg3(a[1], a[2], a[3]);
    output[8] = avg3(l[1], l[0], tl);
    output[9] = output[0];
    output[10] = output[1];
    output[11] = output[2];
    output[12] = avg3(l[2], l[1], l[0]);
    output[13] = output[4];
    output[14] = output[5];
    output[15] = output[6];
}

/// Perform 4x4 vertical-left prediction.
pub fn predict_4x4_vl(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    let a = &ctx.above;

    let avg2 = |x: u8, y: u8| ((x as u16 + y as u16 + 1) >> 1) as u8;
    let avg3 = |x: u8, y: u8, z: u8| ((x as u16 + 2 * y as u16 + z as u16 + 2) >> 2) as u8;

    output[0] = avg2(a[0], a[1]);
    output[1] = avg2(a[1], a[2]);
    output[2] = avg2(a[2], a[3]);
    output[3] = avg2(a[3], a[4]);
    output[4] = avg3(a[0], a[1], a[2]);
    output[5] = avg3(a[1], a[2], a[3]);
    output[6] = avg3(a[2], a[3], a[4]);
    output[7] = avg3(a[3], a[4], a[5]);
    output[8] = avg2(a[1], a[2]);
    output[9] = avg2(a[2], a[3]);
    output[10] = avg2(a[3], a[4]);
    output[11] = avg2(a[4], a[5]);
    output[12] = avg3(a[1], a[2], a[3]);
    output[13] = avg3(a[2], a[3], a[4]);
    output[14] = avg3(a[3], a[4], a[5]);
    output[15] = avg3(a[4], a[5], a[6]);
}

/// Perform 4x4 horizontal-down prediction.
pub fn predict_4x4_hd(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    let a = &ctx.above;
    let l = &ctx.left;
    let tl = ctx.above_left;

    let avg2 = |x: u8, y: u8| ((x as u16 + y as u16 + 1) >> 1) as u8;
    let avg3 = |x: u8, y: u8, z: u8| ((x as u16 + 2 * y as u16 + z as u16 + 2) >> 2) as u8;

    output[0] = avg2(tl, l[0]);
    output[1] = avg3(a[0], tl, l[0]);
    output[2] = avg3(tl, a[0], a[1]);
    output[3] = avg3(a[0], a[1], a[2]);
    output[4] = avg2(l[0], l[1]);
    output[5] = avg3(tl, l[0], l[1]);
    output[6] = output[0];
    output[7] = output[1];
    output[8] = avg2(l[1], l[2]);
    output[9] = avg3(l[0], l[1], l[2]);
    output[10] = output[4];
    output[11] = output[5];
    output[12] = avg2(l[2], l[3]);
    output[13] = avg3(l[1], l[2], l[3]);
    output[14] = output[8];
    output[15] = output[9];
}

/// Perform 4x4 horizontal-up prediction.
pub fn predict_4x4_hu(ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    let l = &ctx.left;

    let avg2 = |x: u8, y: u8| ((x as u16 + y as u16 + 1) >> 1) as u8;
    let avg3 = |x: u8, y: u8, z: u8| ((x as u16 + 2 * y as u16 + z as u16 + 2) >> 2) as u8;

    output[0] = avg2(l[0], l[1]);
    output[1] = avg3(l[0], l[1], l[2]);
    output[2] = avg2(l[1], l[2]);
    output[3] = avg3(l[1], l[2], l[3]);
    output[4] = avg2(l[1], l[2]);
    output[5] = avg3(l[1], l[2], l[3]);
    output[6] = avg2(l[2], l[3]);
    output[7] = avg3(l[2], l[3], l[3]);
    output[8] = avg2(l[2], l[3]);
    output[9] = avg3(l[2], l[3], l[3]);
    output[10] = l[3];
    output[11] = l[3];
    output[12] = l[3];
    output[13] = l[3];
    output[14] = l[3];
    output[15] = l[3];
}

/// Perform 4x4 intra prediction.
pub fn predict_4x4(mode: Intra4x4Mode, ctx: &Pred4x4Context, output: &mut [u8; 16]) {
    match mode {
        Intra4x4Mode::DcPred => predict_4x4_dc(ctx, output),
        Intra4x4Mode::TmPred => predict_4x4_tm(ctx, output),
        Intra4x4Mode::VePred => predict_4x4_ve(ctx, output),
        Intra4x4Mode::HePred => predict_4x4_he(ctx, output),
        Intra4x4Mode::LdPred => predict_4x4_ld(ctx, output),
        Intra4x4Mode::RdPred => predict_4x4_rd(ctx, output),
        Intra4x4Mode::VrPred => predict_4x4_vr(ctx, output),
        Intra4x4Mode::VlPred => predict_4x4_vl(ctx, output),
        Intra4x4Mode::HdPred => predict_4x4_hd(ctx, output),
        Intra4x4Mode::HuPred => predict_4x4_hu(ctx, output),
    }
}

/// Chroma prediction context (8x8).
#[derive(Debug, Clone)]
pub struct ChromaPredContext {
    /// Above pixels.
    pub above: [u8; 8],
    /// Left pixels.
    pub left: [u8; 8],
    /// Above-left corner.
    pub above_left: u8,
    /// Whether above is available.
    pub above_available: bool,
    /// Whether left is available.
    pub left_available: bool,
}

impl Default for ChromaPredContext {
    fn default() -> Self {
        Self {
            above: [128u8; 8],
            left: [128u8; 8],
            above_left: 128,
            above_available: false,
            left_available: false,
        }
    }
}

/// Perform 8x8 chroma DC prediction.
pub fn predict_8x8_dc(ctx: &ChromaPredContext, output: &mut [u8; 64]) {
    let dc = if ctx.above_available && ctx.left_available {
        let sum_above: u32 = ctx.above.iter().map(|&x| x as u32).sum();
        let sum_left: u32 = ctx.left.iter().map(|&x| x as u32).sum();
        ((sum_above + sum_left + 8) >> 4) as u8
    } else if ctx.above_available {
        let sum: u32 = ctx.above.iter().map(|&x| x as u32).sum();
        ((sum + 4) >> 3) as u8
    } else if ctx.left_available {
        let sum: u32 = ctx.left.iter().map(|&x| x as u32).sum();
        ((sum + 4) >> 3) as u8
    } else {
        128
    };

    output.fill(dc);
}

/// Perform 8x8 chroma vertical prediction.
pub fn predict_8x8_v(ctx: &ChromaPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            output[y * 8 + x] = ctx.above[x];
        }
    }
}

/// Perform 8x8 chroma horizontal prediction.
pub fn predict_8x8_h(ctx: &ChromaPredContext, output: &mut [u8; 64]) {
    for y in 0..8 {
        for x in 0..8 {
            output[y * 8 + x] = ctx.left[y];
        }
    }
}

/// Perform 8x8 chroma TrueMotion prediction.
pub fn predict_8x8_tm(ctx: &ChromaPredContext, output: &mut [u8; 64]) {
    let tl = ctx.above_left as i16;
    for y in 0..8 {
        for x in 0..8 {
            let pred = ctx.above[x] as i16 + ctx.left[y] as i16 - tl;
            output[y * 8 + x] = pred.clamp(0, 255) as u8;
        }
    }
}

/// Perform 8x8 chroma prediction.
pub fn predict_8x8_chroma(mode: IntraChromaMode, ctx: &ChromaPredContext, output: &mut [u8; 64]) {
    match mode {
        IntraChromaMode::DcPred => predict_8x8_dc(ctx, output),
        IntraChromaMode::VPred => predict_8x8_v(ctx, output),
        IntraChromaMode::HPred => predict_8x8_h(ctx, output),
        IntraChromaMode::TmPred => predict_8x8_tm(ctx, output),
    }
}

/// Motion vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal component (in quarter-pixel units).
    pub x: i16,
    /// Vertical component (in quarter-pixel units).
    pub y: i16,
}

impl MotionVector {
    /// Create a new motion vector.
    pub const fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Zero motion vector.
    pub const fn zero() -> Self {
        Self { x: 0, y: 0 }
    }

    /// Add two motion vectors.
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

/// Inter prediction using motion compensation.
pub fn inter_predict_16x16(
    reference: &Vp8Frame,
    mv: MotionVector,
    mb_x: usize,
    mb_y: usize,
    output: &mut [u8; 256],
) {
    let ref_data = reference.y_data();
    let stride = reference.y_stride;

    let base_x = (mb_x * 16) as i32 + (mv.x as i32 >> 2);
    let base_y = (mb_y * 16) as i32 + (mv.y as i32 >> 2);

    let frac_x = (mv.x & 3) as usize;
    let frac_y = (mv.y & 3) as usize;

    // Simple integer-pel motion compensation for now
    // TODO: Add sub-pixel interpolation filters
    if frac_x == 0 && frac_y == 0 {
        // Integer pixel position
        for y in 0..16 {
            for x in 0..16 {
                let ref_x = (base_x + x as i32).clamp(0, reference.width as i32 - 1) as usize;
                let ref_y = (base_y + y as i32).clamp(0, reference.height as i32 - 1) as usize;
                output[y * 16 + x] = ref_data[ref_y * stride + ref_x];
            }
        }
    } else {
        // Sub-pixel interpolation (bilinear for simplicity)
        for y in 0..16 {
            for x in 0..16 {
                let ref_x0 = (base_x + x as i32).clamp(0, reference.width as i32 - 1) as usize;
                let ref_y0 = (base_y + y as i32).clamp(0, reference.height as i32 - 1) as usize;
                let ref_x1 = (ref_x0 + 1).min(reference.width as usize - 1);
                let ref_y1 = (ref_y0 + 1).min(reference.height as usize - 1);

                let p00 = ref_data[ref_y0 * stride + ref_x0] as u32;
                let p01 = ref_data[ref_y0 * stride + ref_x1] as u32;
                let p10 = ref_data[ref_y1 * stride + ref_x0] as u32;
                let p11 = ref_data[ref_y1 * stride + ref_x1] as u32;

                let fx = frac_x as u32;
                let fy = frac_y as u32;

                let h0 = p00 * (4 - fx) + p01 * fx;
                let h1 = p10 * (4 - fx) + p11 * fx;
                let val = (h0 * (4 - fy) + h1 * fy + 8) >> 4;

                output[y * 16 + x] = val as u8;
            }
        }
    }
}

/// VP8 6-tap filter coefficients for sub-pixel interpolation.
pub const SUBPEL_FILTERS: [[i16; 6]; 8] = [
    [0, 0, 128, 0, 0, 0],      // 0/8
    [0, -6, 123, 12, -1, 0],   // 1/8
    [2, -11, 108, 36, -8, 1],  // 2/8
    [0, -9, 93, 50, -6, 0],    // 3/8
    [3, -16, 77, 77, -16, 3],  // 4/8
    [0, -6, 50, 93, -9, 0],    // 5/8
    [1, -8, 36, 108, -11, 2],  // 6/8
    [0, -1, 12, 123, -6, 0],   // 7/8
];

/// Apply 6-tap filter for horizontal sub-pixel interpolation.
pub fn filter_h(src: &[u8], x: i32, frac: usize) -> u8 {
    let coeffs = &SUBPEL_FILTERS[frac];
    let mut sum: i32 = 0;

    for (i, &coeff) in coeffs.iter().enumerate() {
        let idx = (x + i as i32 - 2).max(0) as usize;
        sum += src.get(idx).copied().unwrap_or(128) as i32 * coeff as i32;
    }

    ((sum + 64) >> 7).clamp(0, 255) as u8
}

/// Apply 6-tap filter for vertical sub-pixel interpolation.
pub fn filter_v(src: &[u8], stride: usize, y: i32, frac: usize) -> u8 {
    let coeffs = &SUBPEL_FILTERS[frac];
    let mut sum: i32 = 0;

    for (i, &coeff) in coeffs.iter().enumerate() {
        let row = (y + i as i32 - 2).max(0) as usize;
        let idx = row * stride;
        sum += src.get(idx).copied().unwrap_or(128) as i32 * coeff as i32;
    }

    ((sum + 64) >> 7).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_16x16_dc_prediction() {
        let mut ctx = PredictionContext::default();
        ctx.above_available = true;
        ctx.left_available = true;
        ctx.above = [100u8; 17];
        ctx.left = [100u8; 16];

        let mut output = [0u8; 256];
        predict_16x16_dc(&ctx, &mut output);

        assert!(output.iter().all(|&x| x == 100));
    }

    #[test]
    fn test_16x16_v_prediction() {
        let mut ctx = PredictionContext::default();
        ctx.above_available = true;
        for i in 0..16 {
            ctx.above[i + 1] = i as u8;
        }

        let mut output = [0u8; 256];
        predict_16x16_v(&ctx, &mut output);

        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(output[y * 16 + x], x as u8);
            }
        }
    }

    #[test]
    fn test_motion_vector() {
        let mv1 = MotionVector::new(4, 8);
        let mv2 = MotionVector::new(-2, 3);
        let result = mv1.add(mv2);

        assert_eq!(result.x, 2);
        assert_eq!(result.y, 11);
    }
}
