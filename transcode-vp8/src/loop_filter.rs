//! VP8 loop filter (deblocking filter).
//!
//! VP8 uses a loop filter similar to H.264 to reduce blocking artifacts.
//! The filter operates on block edges and can be adjusted per-macroblock.

/// Loop filter parameters.
#[derive(Debug, Clone, Copy)]
pub struct LoopFilterParams {
    /// Filter level (0-63).
    pub level: u8,
    /// Sharpness level (0-7).
    pub sharpness: u8,
    /// Use simple filter (vs normal filter).
    pub simple: bool,
}

impl Default for LoopFilterParams {
    fn default() -> Self {
        Self {
            level: 0,
            sharpness: 0,
            simple: false,
        }
    }
}

impl LoopFilterParams {
    /// Create new loop filter params.
    pub fn new(level: u8, sharpness: u8, simple: bool) -> Self {
        Self {
            level: level.min(63),
            sharpness: sharpness.min(7),
            simple,
        }
    }
}

/// Per-macroblock loop filter adjustment.
#[derive(Debug, Clone, Copy, Default)]
pub struct MbLoopFilterAdjust {
    /// Reference frame adjustment.
    pub ref_delta: [i8; 4],
    /// Mode adjustment.
    pub mode_delta: [i8; 4],
}

/// Compute interior and edge filter limits.
fn compute_limits(level: u8, sharpness: u8) -> (u8, u8, u8) {
    let mut interior_limit = level;
    if sharpness > 0 {
        if sharpness > 4 {
            interior_limit >>= 1;
        }
        interior_limit = interior_limit.min(9 - sharpness);
    }
    interior_limit = interior_limit.max(1);

    let hev_threshold = if level >= 40 {
        2
    } else if level >= 15 {
        1
    } else {
        0
    };

    let edge_limit = (level * 2) + interior_limit;

    (interior_limit, edge_limit.min(255), hev_threshold)
}

/// Simple filter for low complexity deblocking.
fn simple_filter(p1: u8, p0: u8, q0: u8, q1: u8, limit: u8) -> (u8, u8) {
    let p1_i = p1 as i16;
    let p0_i = p0 as i16;
    let q0_i = q0 as i16;
    let q1_i = q1 as i16;

    // Check if filtering should be applied
    let mask = ((p0_i - q0_i).abs() * 2 + ((p1_i - q1_i).abs() >> 1)) <= limit as i16;

    if !mask {
        return (p0, q0);
    }

    // Compute filter value
    let filter = (3 * (q0_i - p0_i) + 0).clamp(-128, 127);
    let filter1 = ((filter + 4).clamp(-128, 127)) >> 3;
    let filter2 = ((filter + 3).clamp(-128, 127)) >> 3;

    let new_q0 = (q0_i - filter1).clamp(0, 255) as u8;
    let new_p0 = (p0_i + filter2).clamp(0, 255) as u8;

    (new_p0, new_q0)
}

/// Check if high-edge-variance condition is met.
fn hev(p1: u8, p0: u8, q0: u8, q1: u8, threshold: u8) -> bool {
    let t = threshold as i16;
    (p1 as i16 - p0 as i16).abs() > t || (q1 as i16 - q0 as i16).abs() > t
}

/// Normal filter for standard deblocking.
fn normal_filter(
    p3: u8, p2: u8, p1: u8, p0: u8,
    q0: u8, q1: u8, q2: u8, q3: u8,
    interior_limit: u8, edge_limit: u8, hev_threshold: u8,
) -> [u8; 8] {
    let _p3_i = p3 as i16;
    let p2_i = p2 as i16;
    let p1_i = p1 as i16;
    let p0_i = p0 as i16;
    let q0_i = q0 as i16;
    let q1_i = q1 as i16;
    let q2_i = q2 as i16;
    let _q3_i = q3 as i16;

    let i_limit = interior_limit as i16;
    let e_limit = edge_limit as i16;

    // Check filter mask
    let interior_mask = (p1_i - p0_i).abs() <= i_limit
        && (q1_i - q0_i).abs() <= i_limit
        && (p2_i - p1_i).abs() <= i_limit
        && (q2_i - q1_i).abs() <= i_limit;

    let edge_mask = ((p0_i - q0_i).abs() * 2 + ((p1_i - q1_i).abs() >> 1)) <= e_limit;

    if !interior_mask || !edge_mask {
        return [p3, p2, p1, p0, q0, q1, q2, q3];
    }

    let use_hev = hev(p1, p0, q0, q1, hev_threshold);

    if use_hev {
        // High edge variance - apply narrow filter
        let filter = (3 * (q0_i - p0_i)).clamp(-128, 127);
        let filter1 = ((filter + 4).clamp(-128, 127)) >> 3;
        let filter2 = ((filter + 3).clamp(-128, 127)) >> 3;

        let new_q0 = (q0_i - filter1).clamp(0, 255) as u8;
        let new_p0 = (p0_i + filter2).clamp(0, 255) as u8;

        [p3, p2, p1, new_p0, new_q0, q1, q2, q3]
    } else {
        // Low edge variance - apply wide filter
        let filter = (3 * (q0_i - p0_i)).clamp(-128, 127);
        let filter1 = ((filter + 4).clamp(-128, 127)) >> 3;
        let filter2 = ((filter + 3).clamp(-128, 127)) >> 3;
        let filter3 = (filter1 + 1) >> 1;

        let new_p0 = (p0_i + filter2).clamp(0, 255) as u8;
        let new_q0 = (q0_i - filter1).clamp(0, 255) as u8;
        let new_p1 = (p1_i + filter3).clamp(0, 255) as u8;
        let new_q1 = (q1_i - filter3).clamp(0, 255) as u8;

        [p3, p2, new_p1, new_p0, new_q0, new_q1, q2, q3]
    }
}

/// Subblock (4-tap) filter for subblock edges.
fn subblock_filter(
    p1: u8, p0: u8, q0: u8, q1: u8,
    interior_limit: u8, edge_limit: u8, hev_threshold: u8,
) -> [u8; 4] {
    let p1_i = p1 as i16;
    let p0_i = p0 as i16;
    let q0_i = q0 as i16;
    let q1_i = q1 as i16;

    let i_limit = interior_limit as i16;
    let e_limit = edge_limit as i16;

    // Check filter mask
    let interior_mask = (p1_i - p0_i).abs() <= i_limit && (q1_i - q0_i).abs() <= i_limit;
    let edge_mask = ((p0_i - q0_i).abs() * 2 + ((p1_i - q1_i).abs() >> 1)) <= e_limit;

    if !interior_mask || !edge_mask {
        return [p1, p0, q0, q1];
    }

    let use_hev = hev(p1, p0, q0, q1, hev_threshold);

    if use_hev {
        // Apply narrow filter
        let filter = (3 * (q0_i - p0_i)).clamp(-128, 127);
        let filter1 = ((filter + 4).clamp(-128, 127)) >> 3;
        let filter2 = ((filter + 3).clamp(-128, 127)) >> 3;

        let new_q0 = (q0_i - filter1).clamp(0, 255) as u8;
        let new_p0 = (p0_i + filter2).clamp(0, 255) as u8;

        [p1, new_p0, new_q0, q1]
    } else {
        // Apply normal filter
        let filter = (3 * (q0_i - p0_i)).clamp(-128, 127);
        let filter1 = ((filter + 4).clamp(-128, 127)) >> 3;
        let filter2 = ((filter + 3).clamp(-128, 127)) >> 3;
        let filter3 = (filter1 + 1) >> 1;

        let new_p0 = (p0_i + filter2).clamp(0, 255) as u8;
        let new_q0 = (q0_i - filter1).clamp(0, 255) as u8;
        let new_p1 = (p1_i + filter3).clamp(0, 255) as u8;
        let new_q1 = (q1_i - filter3).clamp(0, 255) as u8;

        [new_p1, new_p0, new_q0, new_q1]
    }
}

/// Loop filter for a macroblock's Y plane.
pub struct YPlaneFilter<'a> {
    data: &'a mut [u8],
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> YPlaneFilter<'a> {
    /// Create a new Y plane filter.
    pub fn new(data: &'a mut [u8], stride: usize, width: usize, height: usize) -> Self {
        Self { data, stride, width, height }
    }

    /// Apply horizontal edge filter at macroblock boundary.
    pub fn filter_mb_edge_h(&mut self, mb_x: usize, mb_y: usize, params: &LoopFilterParams) {
        if mb_y == 0 || params.level == 0 {
            return;
        }

        let (interior, edge, hev_thresh) = compute_limits(params.level, params.sharpness);
        let y = mb_y * 16;
        let x_start = mb_x * 16;

        for x in x_start..(x_start + 16).min(self.width) {
            if params.simple {
                let p1 = self.get_pixel(x, y - 2);
                let p0 = self.get_pixel(x, y - 1);
                let q0 = self.get_pixel(x, y);
                let q1 = self.get_pixel(x, y + 1);

                let (new_p0, new_q0) = simple_filter(p1, p0, q0, q1, edge);
                self.set_pixel(x, y - 1, new_p0);
                self.set_pixel(x, y, new_q0);
            } else {
                let p3 = self.get_pixel(x, y - 4);
                let p2 = self.get_pixel(x, y - 3);
                let p1 = self.get_pixel(x, y - 2);
                let p0 = self.get_pixel(x, y - 1);
                let q0 = self.get_pixel(x, y);
                let q1 = self.get_pixel(x, y + 1);
                let q2 = self.get_pixel(x, y + 2);
                let q3 = self.get_pixel(x, y + 3);

                let result = normal_filter(p3, p2, p1, p0, q0, q1, q2, q3, interior, edge, hev_thresh);
                self.set_pixel(x, y - 2, result[2]);
                self.set_pixel(x, y - 1, result[3]);
                self.set_pixel(x, y, result[4]);
                self.set_pixel(x, y + 1, result[5]);
            }
        }
    }

    /// Apply vertical edge filter at macroblock boundary.
    pub fn filter_mb_edge_v(&mut self, mb_x: usize, mb_y: usize, params: &LoopFilterParams) {
        if mb_x == 0 || params.level == 0 {
            return;
        }

        let (interior, edge, hev_thresh) = compute_limits(params.level, params.sharpness);
        let x = mb_x * 16;
        let y_start = mb_y * 16;

        for y in y_start..(y_start + 16).min(self.height) {
            if params.simple {
                let p1 = self.get_pixel(x - 2, y);
                let p0 = self.get_pixel(x - 1, y);
                let q0 = self.get_pixel(x, y);
                let q1 = self.get_pixel(x + 1, y);

                let (new_p0, new_q0) = simple_filter(p1, p0, q0, q1, edge);
                self.set_pixel(x - 1, y, new_p0);
                self.set_pixel(x, y, new_q0);
            } else {
                let p3 = self.get_pixel(x - 4, y);
                let p2 = self.get_pixel(x - 3, y);
                let p1 = self.get_pixel(x - 2, y);
                let p0 = self.get_pixel(x - 1, y);
                let q0 = self.get_pixel(x, y);
                let q1 = self.get_pixel(x + 1, y);
                let q2 = self.get_pixel(x + 2, y);
                let q3 = self.get_pixel(x + 3, y);

                let result = normal_filter(p3, p2, p1, p0, q0, q1, q2, q3, interior, edge, hev_thresh);
                self.set_pixel(x - 2, y, result[2]);
                self.set_pixel(x - 1, y, result[3]);
                self.set_pixel(x, y, result[4]);
                self.set_pixel(x + 1, y, result[5]);
            }
        }
    }

    /// Apply horizontal subblock edge filter.
    pub fn filter_subblock_edge_h(&mut self, mb_x: usize, mb_y: usize, sub_y: usize, params: &LoopFilterParams) {
        if params.level == 0 || sub_y == 0 {
            return;
        }

        let (interior, edge, hev_thresh) = compute_limits(params.level, params.sharpness);
        let y = mb_y * 16 + sub_y * 4;
        let x_start = mb_x * 16;

        for x in x_start..(x_start + 16).min(self.width) {
            let p1 = self.get_pixel(x, y - 2);
            let p0 = self.get_pixel(x, y - 1);
            let q0 = self.get_pixel(x, y);
            let q1 = self.get_pixel(x, y + 1);

            let result = subblock_filter(p1, p0, q0, q1, interior, edge, hev_thresh);
            self.set_pixel(x, y - 2, result[0]);
            self.set_pixel(x, y - 1, result[1]);
            self.set_pixel(x, y, result[2]);
            self.set_pixel(x, y + 1, result[3]);
        }
    }

    /// Apply vertical subblock edge filter.
    pub fn filter_subblock_edge_v(&mut self, mb_x: usize, mb_y: usize, sub_x: usize, params: &LoopFilterParams) {
        if params.level == 0 || sub_x == 0 {
            return;
        }

        let (interior, edge, hev_thresh) = compute_limits(params.level, params.sharpness);
        let x = mb_x * 16 + sub_x * 4;
        let y_start = mb_y * 16;

        for y in y_start..(y_start + 16).min(self.height) {
            let p1 = self.get_pixel(x - 2, y);
            let p0 = self.get_pixel(x - 1, y);
            let q0 = self.get_pixel(x, y);
            let q1 = self.get_pixel(x + 1, y);

            let result = subblock_filter(p1, p0, q0, q1, interior, edge, hev_thresh);
            self.set_pixel(x - 2, y, result[0]);
            self.set_pixel(x - 1, y, result[1]);
            self.set_pixel(x, y, result[2]);
            self.set_pixel(x + 1, y, result[3]);
        }
    }

    /// Get pixel value with bounds checking.
    fn get_pixel(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.data[y * self.stride + x]
        } else {
            128
        }
    }

    /// Set pixel value with bounds checking.
    fn set_pixel(&mut self, x: usize, y: usize, value: u8) {
        if x < self.width && y < self.height {
            self.data[y * self.stride + x] = value;
        }
    }
}

/// Loop filter for UV (chroma) planes.
pub struct UVPlaneFilter<'a> {
    u_data: &'a mut [u8],
    v_data: &'a mut [u8],
    stride: usize,
    width: usize,
    height: usize,
}

impl<'a> UVPlaneFilter<'a> {
    /// Create a new UV plane filter.
    pub fn new(
        u_data: &'a mut [u8],
        v_data: &'a mut [u8],
        stride: usize,
        width: usize,
        height: usize,
    ) -> Self {
        Self { u_data, v_data, stride, width, height }
    }

    /// Apply horizontal edge filter at macroblock boundary.
    pub fn filter_mb_edge_h(&mut self, mb_x: usize, mb_y: usize, params: &LoopFilterParams) {
        if mb_y == 0 || params.level == 0 {
            return;
        }

        let (interior, edge, hev_thresh) = compute_limits(params.level, params.sharpness);
        let y = mb_y * 8;
        let x_start = mb_x * 8;

        // Filter U plane
        for x in x_start..(x_start + 8).min(self.width) {
            let p1 = self.get_u_pixel(x, y - 2);
            let p0 = self.get_u_pixel(x, y - 1);
            let q0 = self.get_u_pixel(x, y);
            let q1 = self.get_u_pixel(x, y + 1);

            let result = subblock_filter(p1, p0, q0, q1, interior, edge, hev_thresh);
            self.set_u_pixel(x, y - 2, result[0]);
            self.set_u_pixel(x, y - 1, result[1]);
            self.set_u_pixel(x, y, result[2]);
            self.set_u_pixel(x, y + 1, result[3]);
        }

        // Filter V plane
        for x in x_start..(x_start + 8).min(self.width) {
            let p1 = self.get_v_pixel(x, y - 2);
            let p0 = self.get_v_pixel(x, y - 1);
            let q0 = self.get_v_pixel(x, y);
            let q1 = self.get_v_pixel(x, y + 1);

            let result = subblock_filter(p1, p0, q0, q1, interior, edge, hev_thresh);
            self.set_v_pixel(x, y - 2, result[0]);
            self.set_v_pixel(x, y - 1, result[1]);
            self.set_v_pixel(x, y, result[2]);
            self.set_v_pixel(x, y + 1, result[3]);
        }
    }

    /// Apply vertical edge filter at macroblock boundary.
    pub fn filter_mb_edge_v(&mut self, mb_x: usize, mb_y: usize, params: &LoopFilterParams) {
        if mb_x == 0 || params.level == 0 {
            return;
        }

        let (interior, edge, hev_thresh) = compute_limits(params.level, params.sharpness);
        let x = mb_x * 8;
        let y_start = mb_y * 8;

        // Filter U plane
        for y in y_start..(y_start + 8).min(self.height) {
            let p1 = self.get_u_pixel(x - 2, y);
            let p0 = self.get_u_pixel(x - 1, y);
            let q0 = self.get_u_pixel(x, y);
            let q1 = self.get_u_pixel(x + 1, y);

            let result = subblock_filter(p1, p0, q0, q1, interior, edge, hev_thresh);
            self.set_u_pixel(x - 2, y, result[0]);
            self.set_u_pixel(x - 1, y, result[1]);
            self.set_u_pixel(x, y, result[2]);
            self.set_u_pixel(x + 1, y, result[3]);
        }

        // Filter V plane
        for y in y_start..(y_start + 8).min(self.height) {
            let p1 = self.get_v_pixel(x - 2, y);
            let p0 = self.get_v_pixel(x - 1, y);
            let q0 = self.get_v_pixel(x, y);
            let q1 = self.get_v_pixel(x + 1, y);

            let result = subblock_filter(p1, p0, q0, q1, interior, edge, hev_thresh);
            self.set_v_pixel(x - 2, y, result[0]);
            self.set_v_pixel(x - 1, y, result[1]);
            self.set_v_pixel(x, y, result[2]);
            self.set_v_pixel(x + 1, y, result[3]);
        }
    }

    fn get_u_pixel(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.u_data[y * self.stride + x]
        } else {
            128
        }
    }

    fn set_u_pixel(&mut self, x: usize, y: usize, value: u8) {
        if x < self.width && y < self.height {
            self.u_data[y * self.stride + x] = value;
        }
    }

    fn get_v_pixel(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.v_data[y * self.stride + x]
        } else {
            128
        }
    }

    fn set_v_pixel(&mut self, x: usize, y: usize, value: u8) {
        if x < self.width && y < self.height {
            self.v_data[y * self.stride + x] = value;
        }
    }
}

/// Apply loop filter to entire frame.
pub fn filter_frame(
    y_data: &mut [u8],
    u_data: &mut [u8],
    v_data: &mut [u8],
    width: usize,
    height: usize,
    y_stride: usize,
    uv_stride: usize,
    params: &LoopFilterParams,
) {
    if params.level == 0 {
        return;
    }

    let mb_width = (width + 15) / 16;
    let mb_height = (height + 15) / 16;

    // Filter Y plane
    {
        let mut y_filter = YPlaneFilter::new(y_data, y_stride, width, height);

        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                // Macroblock edges
                y_filter.filter_mb_edge_h(mb_x, mb_y, params);
                y_filter.filter_mb_edge_v(mb_x, mb_y, params);

                // Subblock edges (if not simple filter)
                if !params.simple {
                    for sub_y in 1..4 {
                        y_filter.filter_subblock_edge_h(mb_x, mb_y, sub_y, params);
                    }
                    for sub_x in 1..4 {
                        y_filter.filter_subblock_edge_v(mb_x, mb_y, sub_x, params);
                    }
                }
            }
        }
    }

    // Filter UV planes
    {
        let uv_width = (width + 1) / 2;
        let uv_height = (height + 1) / 2;
        let mut uv_filter = UVPlaneFilter::new(u_data, v_data, uv_stride, uv_width, uv_height);

        for mb_y in 0..mb_height {
            for mb_x in 0..mb_width {
                uv_filter.filter_mb_edge_h(mb_x, mb_y, params);
                uv_filter.filter_mb_edge_v(mb_x, mb_y, params);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_limits() {
        let (interior, edge, hev) = compute_limits(32, 0);
        assert!(interior >= 1);
        assert!(edge > interior);
        assert!(hev >= 0);
    }

    #[test]
    fn test_simple_filter_no_change() {
        // Smooth edge - no filtering needed
        let (p0, q0) = simple_filter(100, 100, 100, 100, 10);
        assert_eq!(p0, 100);
        assert_eq!(q0, 100);
    }

    #[test]
    fn test_loop_filter_params() {
        let params = LoopFilterParams::new(32, 3, false);
        assert_eq!(params.level, 32);
        assert_eq!(params.sharpness, 3);
        assert!(!params.simple);

        // Test clamping
        let params = LoopFilterParams::new(100, 20, true);
        assert_eq!(params.level, 63);
        assert_eq!(params.sharpness, 7);
    }
}
