//! Motion estimation for frame interpolation.
//!
//! This module provides motion vector estimation algorithms used for
//! motion-compensated frame interpolation.

use crate::error::{FrameRateError, Result};
use transcode_core::Frame;

/// Motion vector representing displacement between frames.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotionVector {
    /// Horizontal displacement in pixels.
    pub dx: f32,
    /// Vertical displacement in pixels.
    pub dy: f32,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
}

impl MotionVector {
    /// Create a new motion vector.
    pub fn new(dx: f32, dy: f32) -> Self {
        Self {
            dx,
            dy,
            confidence: 1.0,
        }
    }

    /// Create a zero motion vector.
    pub const fn zero() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            confidence: 1.0,
        }
    }

    /// Calculate the magnitude of the motion vector.
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Interpolate motion vector for a given time position.
    pub fn interpolate(&self, t: f32) -> Self {
        Self {
            dx: self.dx * t,
            dy: self.dy * t,
            confidence: self.confidence,
        }
    }

    /// Reverse the motion vector direction.
    pub fn reverse(&self) -> Self {
        Self {
            dx: -self.dx,
            dy: -self.dy,
            confidence: self.confidence,
        }
    }
}

/// Motion vector field representing motion for entire frame.
#[derive(Debug, Clone)]
pub struct MotionField {
    /// Motion vectors, organized by block.
    pub vectors: Vec<MotionVector>,
    /// Block size used for estimation.
    pub block_size: u32,
    /// Number of blocks horizontally.
    pub blocks_x: u32,
    /// Number of blocks vertically.
    pub blocks_y: u32,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
}

impl MotionField {
    /// Create a new motion field.
    pub fn new(width: u32, height: u32, block_size: u32) -> Self {
        let blocks_x = width.div_ceil(block_size);
        let blocks_y = height.div_ceil(block_size);
        let vector_count = (blocks_x * blocks_y) as usize;

        Self {
            vectors: vec![MotionVector::zero(); vector_count],
            block_size,
            blocks_x,
            blocks_y,
            width,
            height,
        }
    }

    /// Get motion vector at block position.
    pub fn get(&self, bx: u32, by: u32) -> Option<&MotionVector> {
        if bx < self.blocks_x && by < self.blocks_y {
            let idx = (by * self.blocks_x + bx) as usize;
            self.vectors.get(idx)
        } else {
            None
        }
    }

    /// Set motion vector at block position.
    pub fn set(&mut self, bx: u32, by: u32, mv: MotionVector) {
        if bx < self.blocks_x && by < self.blocks_y {
            let idx = (by * self.blocks_x + bx) as usize;
            if idx < self.vectors.len() {
                self.vectors[idx] = mv;
            }
        }
    }

    /// Get interpolated motion vector at pixel position.
    pub fn sample(&self, x: f32, y: f32) -> MotionVector {
        let bx = (x / self.block_size as f32).floor();
        let by = (y / self.block_size as f32).floor();

        let bx0 = (bx as i32).clamp(0, self.blocks_x as i32 - 1) as u32;
        let by0 = (by as i32).clamp(0, self.blocks_y as i32 - 1) as u32;
        let bx1 = (bx0 + 1).min(self.blocks_x - 1);
        let by1 = (by0 + 1).min(self.blocks_y - 1);

        // Bilinear interpolation weights
        let fx = (x / self.block_size as f32) - bx;
        let fy = (y / self.block_size as f32) - by;

        let mv00 = self.get(bx0, by0).copied().unwrap_or_default();
        let mv10 = self.get(bx1, by0).copied().unwrap_or_default();
        let mv01 = self.get(bx0, by1).copied().unwrap_or_default();
        let mv11 = self.get(bx1, by1).copied().unwrap_or_default();

        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        MotionVector {
            dx: mv00.dx * w00 + mv10.dx * w10 + mv01.dx * w01 + mv11.dx * w11,
            dy: mv00.dy * w00 + mv10.dy * w10 + mv01.dy * w01 + mv11.dy * w11,
            confidence: mv00.confidence * w00
                + mv10.confidence * w10
                + mv01.confidence * w01
                + mv11.confidence * w11,
        }
    }

    /// Calculate average motion magnitude.
    pub fn average_magnitude(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.vectors.iter().map(|v| v.magnitude()).sum();
        sum / self.vectors.len() as f32
    }

    /// Interpolate motion field for a given time position.
    pub fn interpolate(&self, t: f32) -> Self {
        let mut result = self.clone();
        for mv in &mut result.vectors {
            *mv = mv.interpolate(t);
        }
        result
    }
}

/// Motion estimation algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum MotionEstimationAlgorithm {
    /// Three-step search (fast, moderate accuracy).
    ThreeStepSearch,
    /// Diamond search (fast, good accuracy).
    #[default]
    DiamondSearch,
    /// Hexagonal search (balanced speed/accuracy).
    HexagonSearch,
    /// Full search (slow, best accuracy).
    FullSearch,
    /// Zero motion (no search, fastest).
    ZeroMotion,
}


/// Configuration for motion estimation.
#[derive(Debug, Clone)]
pub struct MotionEstimationConfig {
    /// Block size for motion estimation.
    pub block_size: u32,
    /// Search range in pixels.
    pub search_range: u32,
    /// Algorithm to use.
    pub algorithm: MotionEstimationAlgorithm,
    /// Enable sub-pixel refinement.
    pub subpixel: bool,
    /// Minimum block variance to estimate motion (skip flat areas).
    pub min_variance: u32,
}

impl Default for MotionEstimationConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            search_range: 32,
            algorithm: MotionEstimationAlgorithm::DiamondSearch,
            subpixel: true,
            min_variance: 100,
        }
    }
}

/// Motion estimator for computing motion vectors between frames.
pub struct MotionEstimator {
    config: MotionEstimationConfig,
}

impl MotionEstimator {
    /// Create a new motion estimator with default configuration.
    pub fn new() -> Self {
        Self {
            config: MotionEstimationConfig::default(),
        }
    }

    /// Create a motion estimator with custom configuration.
    pub fn with_config(config: MotionEstimationConfig) -> Self {
        Self { config }
    }

    /// Estimate motion between two frames.
    pub fn estimate(&self, frame1: &Frame, frame2: &Frame) -> Result<MotionField> {
        if frame1.width() != frame2.width() || frame1.height() != frame2.height() {
            return Err(FrameRateError::DimensionMismatch {
                expected_width: frame1.width(),
                expected_height: frame1.height(),
                actual_width: frame2.width(),
                actual_height: frame2.height(),
            });
        }

        let mut field =
            MotionField::new(frame1.width(), frame1.height(), self.config.block_size);

        let plane1 = frame1.plane(0).ok_or_else(|| {
            FrameRateError::MotionEstimationFailed("Missing Y plane in frame 1".into())
        })?;
        let plane2 = frame2.plane(0).ok_or_else(|| {
            FrameRateError::MotionEstimationFailed("Missing Y plane in frame 2".into())
        })?;

        let stride = frame1.stride(0);

        for by in 0..field.blocks_y {
            for bx in 0..field.blocks_x {
                let mv = self.estimate_block(
                    plane1,
                    plane2,
                    frame1.width(),
                    frame1.height(),
                    stride,
                    bx * self.config.block_size,
                    by * self.config.block_size,
                );
                field.set(bx, by, mv);
            }
        }

        Ok(field)
    }

    /// Estimate motion for a single block.
    fn estimate_block(
        &self,
        plane1: &[u8],
        plane2: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        x: u32,
        y: u32,
    ) -> MotionVector {
        match self.config.algorithm {
            MotionEstimationAlgorithm::ZeroMotion => MotionVector::zero(),
            MotionEstimationAlgorithm::FullSearch => {
                self.full_search(plane1, plane2, width, height, stride, x, y)
            }
            MotionEstimationAlgorithm::ThreeStepSearch => {
                self.three_step_search(plane1, plane2, width, height, stride, x, y)
            }
            MotionEstimationAlgorithm::DiamondSearch => {
                self.diamond_search(plane1, plane2, width, height, stride, x, y)
            }
            MotionEstimationAlgorithm::HexagonSearch => {
                self.hexagon_search(plane1, plane2, width, height, stride, x, y)
            }
        }
    }

    /// Full exhaustive search (slow but accurate).
    fn full_search(
        &self,
        plane1: &[u8],
        plane2: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        x: u32,
        y: u32,
    ) -> MotionVector {
        let range = self.config.search_range as i32;
        let mut best_mv = MotionVector::zero();
        let mut best_sad = u32::MAX;

        for dy in -range..=range {
            for dx in -range..=range {
                let sad = self.compute_sad(plane1, plane2, width, height, stride, x, y, dx, dy);
                if sad < best_sad {
                    best_sad = sad;
                    best_mv = MotionVector::new(dx as f32, dy as f32);
                }
            }
        }

        best_mv.confidence = 1.0 - (best_sad as f32 / (self.config.block_size.pow(2) * 255) as f32);
        best_mv
    }

    /// Three-step search algorithm.
    fn three_step_search(
        &self,
        plane1: &[u8],
        plane2: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        x: u32,
        y: u32,
    ) -> MotionVector {
        let mut step = (self.config.search_range / 2).max(4) as i32;
        let mut cx = 0i32;
        let mut cy = 0i32;
        let mut best_sad = self.compute_sad(plane1, plane2, width, height, stride, x, y, 0, 0);

        while step >= 1 {
            let offsets = [
                (0, 0),
                (-step, 0),
                (step, 0),
                (0, -step),
                (0, step),
                (-step, -step),
                (step, -step),
                (-step, step),
                (step, step),
            ];

            for (dx, dy) in offsets {
                let nx = cx + dx;
                let ny = cy + dy;
                let sad = self.compute_sad(plane1, plane2, width, height, stride, x, y, nx, ny);
                if sad < best_sad {
                    best_sad = sad;
                    cx = nx;
                    cy = ny;
                }
            }

            step /= 2;
        }

        let mut mv = MotionVector::new(cx as f32, cy as f32);
        mv.confidence = 1.0 - (best_sad as f32 / (self.config.block_size.pow(2) * 255) as f32);
        mv
    }

    /// Diamond search algorithm.
    fn diamond_search(
        &self,
        plane1: &[u8],
        plane2: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        x: u32,
        y: u32,
    ) -> MotionVector {
        let large_diamond = [(0, -2), (-1, -1), (1, -1), (-2, 0), (2, 0), (-1, 1), (1, 1), (0, 2)];
        let small_diamond = [(0, -1), (-1, 0), (1, 0), (0, 1)];

        let mut cx = 0i32;
        let mut cy = 0i32;
        let mut best_sad = self.compute_sad(plane1, plane2, width, height, stride, x, y, 0, 0);
        let range = self.config.search_range as i32;

        // Large diamond search
        loop {
            let mut found_better = false;
            for (dx, dy) in large_diamond {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx.abs() <= range && ny.abs() <= range {
                    let sad = self.compute_sad(plane1, plane2, width, height, stride, x, y, nx, ny);
                    if sad < best_sad {
                        best_sad = sad;
                        cx = nx;
                        cy = ny;
                        found_better = true;
                    }
                }
            }
            if !found_better {
                break;
            }
        }

        // Small diamond refinement
        loop {
            let mut found_better = false;
            for (dx, dy) in small_diamond {
                let nx = cx + dx;
                let ny = cy + dy;
                let sad = self.compute_sad(plane1, plane2, width, height, stride, x, y, nx, ny);
                if sad < best_sad {
                    best_sad = sad;
                    cx = nx;
                    cy = ny;
                    found_better = true;
                }
            }
            if !found_better {
                break;
            }
        }

        let mut mv = MotionVector::new(cx as f32, cy as f32);
        mv.confidence = 1.0 - (best_sad as f32 / (self.config.block_size.pow(2) * 255) as f32);
        mv
    }

    /// Hexagonal search algorithm.
    fn hexagon_search(
        &self,
        plane1: &[u8],
        plane2: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        x: u32,
        y: u32,
    ) -> MotionVector {
        let hexagon = [(-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2)];
        let square = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

        let mut cx = 0i32;
        let mut cy = 0i32;
        let mut best_sad = self.compute_sad(plane1, plane2, width, height, stride, x, y, 0, 0);
        let range = self.config.search_range as i32;

        // Hexagon search
        loop {
            let mut found_better = false;
            for (dx, dy) in hexagon {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx.abs() <= range && ny.abs() <= range {
                    let sad = self.compute_sad(plane1, plane2, width, height, stride, x, y, nx, ny);
                    if sad < best_sad {
                        best_sad = sad;
                        cx = nx;
                        cy = ny;
                        found_better = true;
                    }
                }
            }
            if !found_better {
                break;
            }
        }

        // Square refinement
        for (dx, dy) in square {
            let nx = cx + dx;
            let ny = cy + dy;
            let sad = self.compute_sad(plane1, plane2, width, height, stride, x, y, nx, ny);
            if sad < best_sad {
                best_sad = sad;
                cx = nx;
                cy = ny;
            }
        }

        let mut mv = MotionVector::new(cx as f32, cy as f32);
        mv.confidence = 1.0 - (best_sad as f32 / (self.config.block_size.pow(2) * 255) as f32);
        mv
    }

    /// Compute Sum of Absolute Differences (SAD) for a block.
    fn compute_sad(
        &self,
        plane1: &[u8],
        plane2: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        x: u32,
        y: u32,
        dx: i32,
        dy: i32,
    ) -> u32 {
        let mut sad = 0u32;
        let block_size = self.config.block_size;

        for by in 0..block_size {
            for bx in 0..block_size {
                let x1 = x + bx;
                let y1 = y + by;

                let x2 = (x as i32 + bx as i32 + dx).clamp(0, width as i32 - 1) as u32;
                let y2 = (y as i32 + by as i32 + dy).clamp(0, height as i32 - 1) as u32;

                if x1 < width && y1 < height {
                    let idx1 = y1 as usize * stride + x1 as usize;
                    let idx2 = y2 as usize * stride + x2 as usize;

                    if idx1 < plane1.len() && idx2 < plane2.len() {
                        let diff = (plane1[idx1] as i32 - plane2[idx2] as i32).unsigned_abs();
                        sad += diff;
                    }
                }
            }
        }

        sad
    }
}

impl Default for MotionEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::{timestamp::TimeBase, PixelFormat};

    fn create_test_frame(width: u32, height: u32, pattern: u8) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);
        if let Some(plane) = frame.plane_mut(0) {
            for (i, p) in plane.iter_mut().enumerate() {
                *p = pattern.wrapping_add(i as u8);
            }
        }
        frame
    }

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector::new(3.0, 4.0);
        assert!((mv.magnitude() - 5.0).abs() < 0.001);

        let interpolated = mv.interpolate(0.5);
        assert!((interpolated.dx - 1.5).abs() < 0.001);
        assert!((interpolated.dy - 2.0).abs() < 0.001);

        let reversed = mv.reverse();
        assert!((reversed.dx + 3.0).abs() < 0.001);
        assert!((reversed.dy + 4.0).abs() < 0.001);
    }

    #[test]
    fn test_motion_field() {
        let mut field = MotionField::new(64, 64, 16);
        assert_eq!(field.blocks_x, 4);
        assert_eq!(field.blocks_y, 4);

        field.set(1, 1, MotionVector::new(2.0, 3.0));
        let mv = field.get(1, 1).unwrap();
        assert!((mv.dx - 2.0).abs() < 0.001);
        assert!((mv.dy - 3.0).abs() < 0.001);

        // Test sampling
        let sampled = field.sample(24.0, 24.0);
        assert!(sampled.dx.abs() < 10.0);
    }

    #[test]
    fn test_motion_estimation_same_frames() {
        let frame = create_test_frame(64, 64, 128);
        let estimator = MotionEstimator::with_config(MotionEstimationConfig {
            block_size: 16,
            search_range: 8,
            algorithm: MotionEstimationAlgorithm::DiamondSearch,
            ..Default::default()
        });

        let field = estimator.estimate(&frame, &frame).unwrap();

        // Same frames should have near-zero motion
        for mv in &field.vectors {
            assert!(mv.magnitude() < 1.0);
        }
    }

    #[test]
    fn test_motion_estimation_algorithms() {
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(64, 64, 10);

        for algorithm in [
            MotionEstimationAlgorithm::ZeroMotion,
            MotionEstimationAlgorithm::FullSearch,
            MotionEstimationAlgorithm::ThreeStepSearch,
            MotionEstimationAlgorithm::DiamondSearch,
            MotionEstimationAlgorithm::HexagonSearch,
        ] {
            let estimator = MotionEstimator::with_config(MotionEstimationConfig {
                block_size: 16,
                search_range: 8,
                algorithm,
                ..Default::default()
            });

            let field = estimator.estimate(&frame1, &frame2);
            assert!(field.is_ok());
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let frame1 = create_test_frame(64, 64, 0);
        let frame2 = create_test_frame(128, 128, 0);

        let estimator = MotionEstimator::new();
        let result = estimator.estimate(&frame1, &frame2);

        assert!(matches!(
            result,
            Err(FrameRateError::DimensionMismatch { .. })
        ));
    }
}
