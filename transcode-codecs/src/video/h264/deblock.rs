//! Deblocking filter for H.264.

/// Boundary strength for deblocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryStrength {
    /// No filtering.
    Bs0 = 0,
    /// Light filtering.
    Bs1 = 1,
    /// Medium filtering.
    Bs2 = 2,
    /// Strong filtering.
    Bs3 = 3,
    /// Maximum filtering (intra boundary).
    Bs4 = 4,
}

/// Deblocking filter.
pub struct DeblockFilter {
    /// Alpha offset.
    alpha_offset: i8,
    /// Beta offset.
    beta_offset: i8,
}

impl DeblockFilter {
    /// Create a new deblocking filter.
    pub fn new(alpha_offset: i8, beta_offset: i8) -> Self {
        Self {
            alpha_offset,
            beta_offset,
        }
    }

    /// Get alpha threshold for QP.
    fn alpha(&self, qp: u8) -> u8 {
        const ALPHA_TABLE: [u8; 52] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15,
            17, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162,
            182, 203, 226, 255, 255,
        ];
        let idx = (qp as i16 + self.alpha_offset as i16).clamp(0, 51) as usize;
        ALPHA_TABLE[idx]
    }

    /// Get beta threshold for QP.
    fn beta(&self, qp: u8) -> u8 {
        const BETA_TABLE: [u8; 52] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7,
            7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
        ];
        let idx = (qp as i16 + self.beta_offset as i16).clamp(0, 51) as usize;
        BETA_TABLE[idx]
    }

    /// Get tc0 threshold for QP and boundary strength.
    fn tc0(&self, qp: u8, bs: BoundaryStrength) -> u8 {
        const TC0_TABLE: [[u8; 52]; 3] = [
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13,
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17,
            ],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
                2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 23, 25,
            ],
        ];

        if bs == BoundaryStrength::Bs0 {
            return 0;
        }
        let bs_idx = (bs as usize).min(3) - 1;
        let qp_idx = (qp as i16 + self.alpha_offset as i16).clamp(0, 51) as usize;
        TC0_TABLE[bs_idx][qp_idx]
    }

    /// Filter a vertical edge (4 pixels).
    pub fn filter_vertical_edge(
        &self,
        data: &mut [u8],
        stride: usize,
        offset: usize,
        qp: u8,
        bs: BoundaryStrength,
    ) {
        if bs == BoundaryStrength::Bs0 {
            return;
        }

        let alpha = self.alpha(qp) as i32;
        let beta = self.beta(qp) as i32;

        for row in 0..4 {
            let idx = offset + row * stride;

            let p1 = data[idx - 2] as i32;
            let p0 = data[idx - 1] as i32;
            let q0 = data[idx] as i32;
            let q1 = data[idx + 1] as i32;

            // Check if filtering should be applied
            if (p0 - q0).abs() >= alpha
                || (p1 - p0).abs() >= beta
                || (q1 - q0).abs() >= beta
            {
                continue;
            }

            if bs == BoundaryStrength::Bs4 {
                // Strong filtering
                let p2 = data[idx - 3] as i32;
                let q2 = data[idx + 2] as i32;

                if (p2 - p0).abs() < beta && (p0 - q0).abs() < (alpha >> 2) + 2 {
                    data[idx - 1] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
                    data[idx - 2] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
                    data[idx - 3] = ((2 * p2 + p1 + p0 + q0 + 2) >> 2) as u8;
                } else {
                    data[idx - 1] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
                }

                if (q2 - q0).abs() < beta && (p0 - q0).abs() < (alpha >> 2) + 2 {
                    data[idx] = ((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3) as u8;
                    data[idx + 1] = ((q2 + q1 + q0 + p0 + 2) >> 2) as u8;
                    data[idx + 2] = ((2 * q2 + q1 + q0 + p0 + 2) >> 2) as u8;
                } else {
                    data[idx] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
                }
            } else {
                // Normal filtering
                let tc0 = self.tc0(qp, bs) as i32;
                let tc = tc0 + 1;

                let delta = ((4 * (q0 - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);

                data[idx - 1] = (p0 + delta).clamp(0, 255) as u8;
                data[idx] = (q0 - delta).clamp(0, 255) as u8;

                // Filter p1 and q1
                let p2 = data[idx - 3] as i32;
                let q2 = data[idx + 2] as i32;

                if (p2 - p0).abs() < beta {
                    let delta_p = ((p2 + ((p0 + q0 + 1) >> 1) - 2 * p1) >> 1).clamp(-tc0, tc0);
                    data[idx - 2] = (p1 + delta_p).clamp(0, 255) as u8;
                }

                if (q2 - q0).abs() < beta {
                    let delta_q = ((q2 + ((p0 + q0 + 1) >> 1) - 2 * q1) >> 1).clamp(-tc0, tc0);
                    data[idx + 1] = (q1 + delta_q).clamp(0, 255) as u8;
                }
            }
        }
    }

    /// Filter a horizontal edge (4 pixels).
    pub fn filter_horizontal_edge(
        &self,
        data: &mut [u8],
        stride: usize,
        offset: usize,
        qp: u8,
        bs: BoundaryStrength,
    ) {
        if bs == BoundaryStrength::Bs0 {
            return;
        }

        let alpha = self.alpha(qp) as i32;
        let beta = self.beta(qp) as i32;

        for col in 0..4 {
            let idx = offset + col;

            let p1 = data[idx - 2 * stride] as i32;
            let p0 = data[idx - stride] as i32;
            let q0 = data[idx] as i32;
            let q1 = data[idx + stride] as i32;

            if (p0 - q0).abs() >= alpha
                || (p1 - p0).abs() >= beta
                || (q1 - q0).abs() >= beta
            {
                continue;
            }

            if bs == BoundaryStrength::Bs4 {
                // Strong filtering (similar to vertical)
                data[idx - stride] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
                data[idx] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
            } else {
                // Normal filtering
                let tc = self.tc0(qp, bs) as i32 + 1;
                let delta = ((4 * (q0 - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);

                data[idx - stride] = (p0 + delta).clamp(0, 255) as u8;
                data[idx] = (q0 - delta).clamp(0, 255) as u8;
            }
        }
    }

    /// Filter an entire macroblock.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn filter_macroblock(
        &self,
        luma: &mut [u8],
        luma_stride: usize,
        chroma_u: &mut [u8],
        chroma_v: &mut [u8],
        chroma_stride: usize,
        qp_luma: u8,
        qp_chroma: u8,
        mb_x: usize,
        mb_y: usize,
        bs_v: &[[BoundaryStrength; 4]; 4],
        bs_h: &[[BoundaryStrength; 4]; 4],
    ) {
        let luma_offset = mb_y * 16 * luma_stride + mb_x * 16;
        let chroma_offset = mb_y * 8 * chroma_stride + mb_x * 8;

        // Filter luma vertical edges
        for block_y in 0..4 {
            for block_x in 0..4 {
                let offset = luma_offset + block_y * 4 * luma_stride + block_x * 4;
                if block_x > 0 || mb_x > 0 {
                    self.filter_vertical_edge(luma, luma_stride, offset, qp_luma, bs_v[block_y][block_x]);
                }
            }
        }

        // Filter luma horizontal edges
        for block_y in 0..4 {
            for block_x in 0..4 {
                let offset = luma_offset + block_y * 4 * luma_stride + block_x * 4;
                if block_y > 0 || mb_y > 0 {
                    self.filter_horizontal_edge(luma, luma_stride, offset, qp_luma, bs_h[block_y][block_x]);
                }
            }
        }

        // Filter chroma (simplified - every other block)
        for block_y in 0..2 {
            for block_x in 0..2 {
                let offset = chroma_offset + block_y * 4 * chroma_stride + block_x * 4;
                let bs_v_chroma = bs_v[block_y * 2][block_x * 2];
                let bs_h_chroma = bs_h[block_y * 2][block_x * 2];

                if block_x > 0 || mb_x > 0 {
                    self.filter_vertical_edge(chroma_u, chroma_stride, offset, qp_chroma, bs_v_chroma);
                    self.filter_vertical_edge(chroma_v, chroma_stride, offset, qp_chroma, bs_v_chroma);
                }
                if block_y > 0 || mb_y > 0 {
                    self.filter_horizontal_edge(chroma_u, chroma_stride, offset, qp_chroma, bs_h_chroma);
                    self.filter_horizontal_edge(chroma_v, chroma_stride, offset, qp_chroma, bs_h_chroma);
                }
            }
        }
    }
}

impl Default for DeblockFilter {
    fn default() -> Self {
        Self::new(0, 0)
    }
}
