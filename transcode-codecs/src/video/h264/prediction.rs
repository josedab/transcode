//! Intra and inter prediction for H.264.

/// Intra prediction modes for 4x4 blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Intra4x4Mode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    DiagonalDownLeft = 3,
    DiagonalDownRight = 4,
    VerticalRight = 5,
    HorizontalDown = 6,
    VerticalLeft = 7,
    HorizontalUp = 8,
}

/// Intra prediction modes for 16x16 blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Intra16x16Mode {
    Vertical = 0,
    Horizontal = 1,
    Dc = 2,
    Plane = 3,
}

/// Intra prediction modes for 8x8 chroma blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IntraChromaMode {
    Dc = 0,
    Horizontal = 1,
    Vertical = 2,
    Plane = 3,
}

/// Intra predictor.
pub struct IntraPredictor;

impl IntraPredictor {
    /// Predict a 4x4 luma block.
    pub fn predict_4x4(
        dst: &mut [u8],
        dst_stride: usize,
        above: &[u8; 8],
        left: &[u8; 8],
        mode: Intra4x4Mode,
    ) {
        match mode {
            Intra4x4Mode::Vertical => {
                for row in 0..4 {
                    for col in 0..4 {
                        dst[row * dst_stride + col] = above[col];
                    }
                }
            }
            Intra4x4Mode::Horizontal => {
                for row in 0..4 {
                    for col in 0..4 {
                        dst[row * dst_stride + col] = left[row];
                    }
                }
            }
            Intra4x4Mode::Dc => {
                let sum: u32 = above[..4].iter().map(|&x| x as u32).sum::<u32>()
                    + left[..4].iter().map(|&x| x as u32).sum::<u32>();
                let dc = ((sum + 4) >> 3) as u8;
                for row in 0..4 {
                    for col in 0..4 {
                        dst[row * dst_stride + col] = dc;
                    }
                }
            }
            Intra4x4Mode::DiagonalDownLeft => {
                for row in 0..4 {
                    for col in 0..4 {
                        let idx = row + col;
                        if idx < 6 {
                            dst[row * dst_stride + col] = ((above[idx] as u32
                                + 2 * above[idx + 1] as u32
                                + above[idx + 2] as u32
                                + 2)
                                >> 2) as u8;
                        } else {
                            dst[row * dst_stride + col] = above[7];
                        }
                    }
                }
            }
            _ => {
                // Other modes - use DC fallback
                let dc = 128u8;
                for row in 0..4 {
                    for col in 0..4 {
                        dst[row * dst_stride + col] = dc;
                    }
                }
            }
        }
    }

    /// Predict a 16x16 luma block.
    pub fn predict_16x16(
        dst: &mut [u8],
        dst_stride: usize,
        above: &[u8; 16],
        left: &[u8; 16],
        mode: Intra16x16Mode,
    ) {
        match mode {
            Intra16x16Mode::Vertical => {
                for row in 0..16 {
                    dst[row * dst_stride..row * dst_stride + 16].copy_from_slice(above);
                }
            }
            Intra16x16Mode::Horizontal => {
                for row in 0..16 {
                    for col in 0..16 {
                        dst[row * dst_stride + col] = left[row];
                    }
                }
            }
            Intra16x16Mode::Dc => {
                let sum: u32 = above.iter().map(|&x| x as u32).sum::<u32>()
                    + left.iter().map(|&x| x as u32).sum::<u32>();
                let dc = ((sum + 16) >> 5) as u8;
                for row in 0..16 {
                    for col in 0..16 {
                        dst[row * dst_stride + col] = dc;
                    }
                }
            }
            Intra16x16Mode::Plane => {
                // Plane prediction
                let h: i32 = (1i32..8).map(|i| i * (above[(7 + i) as usize] as i32 - above[(7 - i) as usize] as i32)).sum();
                let v: i32 = (1i32..8).map(|i| i * (left[(7 + i) as usize] as i32 - left[(7 - i) as usize] as i32)).sum();

                let a = 16 * (above[15] as i32 + left[15] as i32);
                let b = (5 * h + 32) >> 6;
                let c = (5 * v + 32) >> 6;

                for row in 0..16 {
                    for col in 0..16 {
                        let val = (a + b * (col as i32 - 7) + c * (row as i32 - 7) + 16) >> 5;
                        dst[row * dst_stride + col] = val.clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
}

/// Motion vector.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal component (quarter-pel).
    pub x: i16,
    /// Vertical component (quarter-pel).
    pub y: i16,
}

impl MotionVector {
    /// Create a new motion vector.
    pub fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Create a zero motion vector.
    pub const fn zero() -> Self {
        Self { x: 0, y: 0 }
    }

    /// Add two motion vectors.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    /// Scale motion vector.
    pub fn scale(&self, num: i32, den: i32) -> Self {
        Self {
            x: ((self.x as i32 * num + den / 2) / den) as i16,
            y: ((self.y as i32 * num + den / 2) / den) as i16,
        }
    }
}

/// Inter predictor for motion compensation.
pub struct InterPredictor;

impl InterPredictor {
    /// Perform motion compensation for a block.
    pub fn motion_compensate(
        dst: &mut [u8],
        dst_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        mv: MotionVector,
        width: usize,
        height: usize,
    ) {
        let frac_x = mv.x & 3;
        let frac_y = mv.y & 3;
        let int_x = mv.x >> 2;
        let int_y = mv.y >> 2;

        if frac_x == 0 && frac_y == 0 {
            // Integer-pel, direct copy
            for row in 0..height {
                let src_row = ((row as i32 + int_y as i32) as usize) * ref_stride
                    + int_x as usize;
                dst[row * dst_stride..row * dst_stride + width]
                    .copy_from_slice(&ref_frame[src_row..src_row + width]);
            }
        } else {
            // Fractional-pel interpolation
            Self::interpolate(
                dst,
                dst_stride,
                ref_frame,
                ref_stride,
                int_x,
                int_y,
                frac_x,
                frac_y,
                width,
                height,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn interpolate(
        dst: &mut [u8],
        dst_stride: usize,
        ref_frame: &[u8],
        ref_stride: usize,
        int_x: i16,
        int_y: i16,
        frac_x: i16,
        frac_y: i16,
        width: usize,
        height: usize,
    ) {
        // 6-tap filter coefficients for half-pel
        const FILTER: [i32; 6] = [1, -5, 20, 20, -5, 1];

        for row in 0..height {
            for col in 0..width {
                let ref_y = (row as i32 + int_y as i32) as usize;
                let ref_x = (col as i32 + int_x as i32) as usize;

                // Simplified bilinear interpolation
                let val = if frac_x == 0 {
                    // Vertical only
                    let a = ref_frame[ref_y * ref_stride + ref_x] as i32;
                    let b = ref_frame[(ref_y + 1) * ref_stride + ref_x] as i32;
                    ((4 - frac_y as i32) * a + frac_y as i32 * b + 2) >> 2
                } else if frac_y == 0 {
                    // Horizontal only
                    let a = ref_frame[ref_y * ref_stride + ref_x] as i32;
                    let b = ref_frame[ref_y * ref_stride + ref_x + 1] as i32;
                    ((4 - frac_x as i32) * a + frac_x as i32 * b + 2) >> 2
                } else {
                    // Both directions
                    let a = ref_frame[ref_y * ref_stride + ref_x] as i32;
                    let b = ref_frame[ref_y * ref_stride + ref_x + 1] as i32;
                    let c = ref_frame[(ref_y + 1) * ref_stride + ref_x] as i32;
                    let d = ref_frame[(ref_y + 1) * ref_stride + ref_x + 1] as i32;

                    let fx = frac_x as i32;
                    let fy = frac_y as i32;

                    ((4 - fx) * (4 - fy) * a
                        + fx * (4 - fy) * b
                        + (4 - fx) * fy * c
                        + fx * fy * d
                        + 8)
                        >> 4
                };

                dst[row * dst_stride + col] = val.clamp(0, 255) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector() {
        let mv1 = MotionVector::new(4, 8);
        let mv2 = MotionVector::new(2, 4);
        let sum = mv1.add(&mv2);
        assert_eq!(sum.x, 6);
        assert_eq!(sum.y, 12);
    }

    #[test]
    fn test_intra_dc_4x4() {
        let above = [100, 100, 100, 100, 0, 0, 0, 0];
        let left = [100, 100, 100, 100, 0, 0, 0, 0];
        let mut dst = [0u8; 16];

        IntraPredictor::predict_4x4(&mut dst, 4, &above, &left, Intra4x4Mode::Dc);
        assert_eq!(dst[0], 100);
    }
}
