//! Multi-threaded H.264 encoding support.
//!
//! This module provides parallel encoding capabilities for H.264:
//! - Slice-based parallelism: Split frames into slices and encode in parallel
//! - Frame-based parallelism: Parallel motion estimation with lookahead buffer
//! - Thread pool configuration and management

use std::sync::Arc;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use rayon::ThreadPool;
use transcode_core::Frame;

use super::prediction::MotionVector;

/// Thread pool configuration for parallel encoding.
#[derive(Debug, Clone)]
pub struct ThreadingConfig {
    /// Number of threads to use (0 = auto-detect based on CPU cores).
    pub num_threads: usize,
    /// Number of slices per frame for slice-based parallelism.
    pub slice_count: usize,
    /// Lookahead depth for B-frame decisions and parallel motion estimation.
    pub lookahead_depth: usize,
    /// Enable slice-based parallelism.
    pub enable_slice_parallel: bool,
    /// Enable frame-based parallelism (parallel motion estimation).
    pub enable_frame_parallel: bool,
}

impl Default for ThreadingConfig {
    fn default() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        Self {
            num_threads: 0, // Auto-detect
            slice_count: num_cpus.max(2), // At least 2 slices
            lookahead_depth: 40, // Default lookahead for B-frame decisions
            enable_slice_parallel: true,
            enable_frame_parallel: true,
        }
    }
}

impl ThreadingConfig {
    /// Create a new threading configuration with custom thread count.
    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            num_threads,
            ..Default::default()
        }
    }

    /// Set the number of slices per frame.
    pub fn with_slice_count(mut self, count: usize) -> Self {
        self.slice_count = count.max(1);
        self
    }

    /// Set the lookahead depth.
    pub fn with_lookahead_depth(mut self, depth: usize) -> Self {
        self.lookahead_depth = depth;
        self
    }

    /// Get the effective number of threads.
    pub fn effective_threads(&self) -> usize {
        if self.num_threads == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        } else {
            self.num_threads
        }
    }
}

/// Slice data for parallel encoding.
#[derive(Debug, Clone)]
pub struct SliceData {
    /// Slice index within the frame.
    pub slice_idx: usize,
    /// First macroblock row in this slice.
    pub first_mb_row: usize,
    /// Number of macroblock rows in this slice.
    pub mb_row_count: usize,
    /// Encoded slice data.
    pub encoded_data: Vec<u8>,
    /// Slice QP used.
    pub qp: u8,
}

/// Slice encoder context for thread-safe slice encoding.
pub struct SliceEncoderContext {
    /// Frame width in macroblocks.
    pub mb_width: usize,
    /// Frame height in macroblocks.
    pub mb_height: usize,
    /// Base QP for the frame.
    pub base_qp: u8,
    /// Whether this is an intra frame.
    pub is_intra: bool,
    /// CABAC mode enabled.
    pub cabac: bool,
    /// Reference frame data (read-only during encoding).
    pub reference_frames: Vec<Arc<RwLock<ReferenceFrame>>>,
}

/// Reference frame data.
#[derive(Debug, Clone)]
pub struct ReferenceFrame {
    /// Frame number.
    pub frame_num: u64,
    /// Picture order count.
    pub poc: i32,
    /// Luma plane data.
    pub luma: Vec<u8>,
    /// Chroma U plane data.
    pub chroma_u: Vec<u8>,
    /// Chroma V plane data.
    pub chroma_v: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Motion vectors for each macroblock.
    pub motion_vectors: Vec<MotionVector>,
}

impl ReferenceFrame {
    /// Create a new reference frame from a decoded frame.
    pub fn from_frame(frame: &Frame, frame_num: u64, poc: i32) -> Self {
        let width = frame.width();
        let height = frame.height();
        let mb_count = width.div_ceil(16) * height.div_ceil(16);

        Self {
            frame_num,
            poc,
            luma: frame.plane(0).map(|p| p.to_vec()).unwrap_or_default(),
            chroma_u: frame.plane(1).map(|p| p.to_vec()).unwrap_or_default(),
            chroma_v: frame.plane(2).map(|p| p.to_vec()).unwrap_or_default(),
            width,
            height,
            motion_vectors: vec![MotionVector::zero(); mb_count as usize],
        }
    }
}

/// Lookahead frame entry for B-frame decision making.
#[derive(Debug, Clone)]
pub struct LookaheadFrame {
    /// Original frame.
    pub frame: Frame,
    /// Frame display order.
    pub display_order: u64,
    /// Computed frame type (I, P, or B).
    pub frame_type: FrameType,
    /// Computed complexity score.
    pub complexity: f64,
    /// Motion estimation results (per macroblock).
    pub motion_results: Vec<MotionEstimationResult>,
}

/// Frame type for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Intra frame (keyframe).
    I,
    /// Predicted frame.
    P,
    /// Bidirectional predicted frame.
    B,
}

/// Motion estimation result for a single macroblock.
#[derive(Debug, Clone, Default)]
pub struct MotionEstimationResult {
    /// Best motion vector for L0 prediction.
    pub mv_l0: MotionVector,
    /// Best motion vector for L1 prediction (for B-frames).
    pub mv_l1: MotionVector,
    /// Best reference frame index for L0.
    pub ref_idx_l0: u8,
    /// Best reference frame index for L1.
    pub ref_idx_l1: u8,
    /// Motion estimation cost (SAD or SATD).
    pub cost: u32,
    /// Intra cost for mode decision.
    pub intra_cost: u32,
}

/// Parallel motion estimator.
pub struct ParallelMotionEstimator {
    /// Thread pool for parallel operations.
    thread_pool: ThreadPool,
    /// Search range for motion estimation.
    search_range: i16,
    /// Sub-pixel refinement enabled.
    subpel_refine: bool,
}

impl ParallelMotionEstimator {
    /// Create a new parallel motion estimator.
    pub fn new(config: &ThreadingConfig) -> Self {
        let num_threads = config.effective_threads();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|idx| format!("h264-me-{}", idx))
            .build()
            .expect("Failed to create motion estimation thread pool");

        Self {
            thread_pool,
            search_range: 64,
            subpel_refine: true,
        }
    }

    /// Set the search range for motion estimation.
    pub fn with_search_range(mut self, range: i16) -> Self {
        self.search_range = range;
        self
    }

    /// Perform parallel motion estimation for a frame.
    ///
    /// Returns an empty vector if the frame has zero dimensions.
    pub fn estimate_motion(
        &self,
        current_frame: &Frame,
        reference_frames: &[Arc<RwLock<ReferenceFrame>>],
    ) -> Vec<MotionEstimationResult> {
        let width = current_frame.width() as usize;
        let height = current_frame.height() as usize;

        // Handle zero-dimension frames gracefully
        if width == 0 || height == 0 {
            return Vec::new();
        }

        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        let mb_count = mb_width * mb_height;

        let current_luma = current_frame.plane(0).unwrap_or(&[]);
        let stride = current_frame.stride(0);
        let search_range = self.search_range;

        // Create read-only references for thread safety
        let ref_data: Vec<_> = reference_frames
            .iter()
            .map(|r| r.read().clone())
            .collect();

        self.thread_pool.install(|| {
            (0..mb_count)
                .into_par_iter()
                .map(|mb_idx| {
                    let mb_x = mb_idx % mb_width;
                    let mb_y = mb_idx / mb_width;

                    Self::estimate_mb_motion(
                        current_luma,
                        stride,
                        width,
                        height,
                        mb_x,
                        mb_y,
                        &ref_data,
                        search_range,
                    )
                })
                .collect()
        })
    }

    /// Estimate motion for a single macroblock.
    fn estimate_mb_motion(
        current_luma: &[u8],
        stride: usize,
        width: usize,
        height: usize,
        mb_x: usize,
        mb_y: usize,
        reference_frames: &[ReferenceFrame],
        search_range: i16,
    ) -> MotionEstimationResult {
        let mut result = MotionEstimationResult {
            cost: u32::MAX,
            ..Default::default()
        };

        let mb_px_x = mb_x * 16;
        let mb_px_y = mb_y * 16;

        // Calculate intra cost (DC prediction SAD)
        result.intra_cost = Self::calculate_intra_cost(
            current_luma,
            stride,
            mb_px_x,
            mb_px_y,
            width,
            height,
        );

        if reference_frames.is_empty() {
            result.cost = result.intra_cost;
            return result;
        }

        // Search in reference frames
        for (ref_idx, ref_frame) in reference_frames.iter().enumerate() {
            let ref_luma = &ref_frame.luma;
            let ref_stride = ref_frame.width as usize;

            // Diamond search for motion vector
            let (mv, cost) = Self::diamond_search(
                current_luma,
                stride,
                ref_luma,
                ref_stride,
                mb_px_x,
                mb_px_y,
                width,
                height,
                search_range,
            );

            if cost < result.cost {
                result.cost = cost;
                result.mv_l0 = mv;
                result.ref_idx_l0 = ref_idx as u8;
            }
        }

        result
    }

    /// Calculate intra cost using DC prediction SAD.
    fn calculate_intra_cost(
        luma: &[u8],
        stride: usize,
        mb_x: usize,
        mb_y: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        if mb_x + 16 > width || mb_y + 16 > height {
            return u32::MAX;
        }

        let mut sum: u32 = 0;
        for y in 0..16 {
            for x in 0..16 {
                let idx = (mb_y + y) * stride + mb_x + x;
                if idx < luma.len() {
                    sum += luma[idx] as u32;
                }
            }
        }
        let dc = (sum / 256) as u8;

        // Calculate SAD against DC prediction
        let mut sad: u32 = 0;
        for y in 0..16 {
            for x in 0..16 {
                let idx = (mb_y + y) * stride + mb_x + x;
                if idx < luma.len() {
                    sad += (luma[idx] as i32 - dc as i32).unsigned_abs();
                }
            }
        }

        sad
    }

    /// Diamond search motion estimation.
    fn diamond_search(
        current: &[u8],
        cur_stride: usize,
        reference: &[u8],
        ref_stride: usize,
        mb_x: usize,
        mb_y: usize,
        width: usize,
        height: usize,
        search_range: i16,
    ) -> (MotionVector, u32) {
        let mut best_mv = MotionVector::zero();
        let mut best_cost = Self::calculate_sad(
            current, cur_stride,
            reference, ref_stride,
            mb_x, mb_y, 0, 0,
            width, height,
        );

        // Large diamond pattern
        const LARGE_DIAMOND: [(i16, i16); 8] = [
            (-2, 0), (2, 0), (0, -2), (0, 2),
            (-1, -1), (1, -1), (-1, 1), (1, 1),
        ];

        // Small diamond pattern for refinement
        const SMALL_DIAMOND: [(i16, i16); 4] = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
        ];

        let mut center_x: i16 = 0;
        let mut center_y: i16 = 0;

        // Large diamond search
        for _ in 0..16 {
            let mut found_better = false;

            for &(dx, dy) in &LARGE_DIAMOND {
                let mv_x = center_x + dx;
                let mv_y = center_y + dy;

                if mv_x.abs() > search_range || mv_y.abs() > search_range {
                    continue;
                }

                let cost = Self::calculate_sad(
                    current, cur_stride,
                    reference, ref_stride,
                    mb_x, mb_y, mv_x, mv_y,
                    width, height,
                );

                if cost < best_cost {
                    best_cost = cost;
                    best_mv = MotionVector::new(mv_x * 4, mv_y * 4); // Quarter-pel
                    center_x = mv_x;
                    center_y = mv_y;
                    found_better = true;
                }
            }

            if !found_better {
                break;
            }
        }

        // Small diamond refinement
        loop {
            let mut found_better = false;

            for &(dx, dy) in &SMALL_DIAMOND {
                let mv_x = center_x + dx;
                let mv_y = center_y + dy;

                if mv_x.abs() > search_range || mv_y.abs() > search_range {
                    continue;
                }

                let cost = Self::calculate_sad(
                    current, cur_stride,
                    reference, ref_stride,
                    mb_x, mb_y, mv_x, mv_y,
                    width, height,
                );

                if cost < best_cost {
                    best_cost = cost;
                    best_mv = MotionVector::new(mv_x * 4, mv_y * 4);
                    center_x = mv_x;
                    center_y = mv_y;
                    found_better = true;
                }
            }

            if !found_better {
                break;
            }
        }

        (best_mv, best_cost)
    }

    /// Calculate SAD (Sum of Absolute Differences) between blocks.
    fn calculate_sad(
        current: &[u8],
        cur_stride: usize,
        reference: &[u8],
        ref_stride: usize,
        mb_x: usize,
        mb_y: usize,
        mv_x: i16,
        mv_y: i16,
        width: usize,
        height: usize,
    ) -> u32 {
        let ref_x = mb_x as i32 + mv_x as i32;
        let ref_y = mb_y as i32 + mv_y as i32;

        // Bounds check
        if ref_x < 0 || ref_y < 0
            || ref_x as usize + 16 > width
            || ref_y as usize + 16 > height
            || mb_x + 16 > width
            || mb_y + 16 > height
        {
            return u32::MAX;
        }

        let ref_x = ref_x as usize;
        let ref_y = ref_y as usize;

        let mut sad: u32 = 0;
        for y in 0..16 {
            for x in 0..16 {
                let cur_idx = (mb_y + y) * cur_stride + mb_x + x;
                let ref_idx = (ref_y + y) * ref_stride + ref_x + x;

                if cur_idx < current.len() && ref_idx < reference.len() {
                    sad += (current[cur_idx] as i32 - reference[ref_idx] as i32).unsigned_abs();
                }
            }
        }

        sad
    }
}

/// Parallel slice encoder.
pub struct ParallelSliceEncoder {
    /// Thread pool for parallel slice encoding.
    thread_pool: ThreadPool,
    /// Threading configuration.
    config: ThreadingConfig,
}

impl ParallelSliceEncoder {
    /// Create a new parallel slice encoder.
    pub fn new(config: &ThreadingConfig) -> Self {
        let num_threads = config.effective_threads();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|idx| format!("h264-slice-{}", idx))
            .build()
            .expect("Failed to create slice encoding thread pool");

        Self {
            thread_pool,
            config: config.clone(),
        }
    }

    /// Split a frame into slices for parallel encoding.
    pub fn split_into_slices(
        &self,
        frame_height: u32,
    ) -> Vec<(usize, usize)> {
        let mb_height = frame_height.div_ceil(16) as usize;
        let slice_count = self.config.slice_count.min(mb_height);

        let rows_per_slice = mb_height / slice_count;
        let extra_rows = mb_height % slice_count;

        let mut slices = Vec::with_capacity(slice_count);
        let mut current_row = 0;

        for i in 0..slice_count {
            let rows = rows_per_slice + if i < extra_rows { 1 } else { 0 };
            slices.push((current_row, rows));
            current_row += rows;
        }

        slices
    }

    /// Encode slices in parallel.
    pub fn encode_slices_parallel<F>(
        &self,
        frame: &Frame,
        context: &SliceEncoderContext,
        encode_slice_fn: F,
    ) -> Vec<SliceData>
    where
        F: Fn(&Frame, &SliceEncoderContext, usize, usize, usize) -> SliceData + Sync,
    {
        let slice_bounds = self.split_into_slices(frame.height());

        self.thread_pool.install(|| {
            slice_bounds
                .into_par_iter()
                .enumerate()
                .map(|(slice_idx, (first_mb_row, mb_row_count))| {
                    encode_slice_fn(frame, context, slice_idx, first_mb_row, mb_row_count)
                })
                .collect()
        })
    }

    /// Merge encoded slices into a single NAL unit stream.
    pub fn merge_slices(&self, slices: Vec<SliceData>) -> Vec<u8> {
        let total_size: usize = slices.iter().map(|s| s.encoded_data.len() + 4).sum();
        let mut result = Vec::with_capacity(total_size);

        for slice in slices {
            // Add start code
            result.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            result.extend(slice.encoded_data);
        }

        result
    }
}

/// Lookahead buffer for B-frame decision making.
pub struct LookaheadBuffer {
    /// Maximum lookahead depth.
    max_depth: usize,
    /// Buffered frames.
    frames: Vec<LookaheadFrame>,
    /// Thread pool for parallel analysis.
    thread_pool: ThreadPool,
    /// Maximum B-frames between I/P.
    max_bframes: u8,
    /// GOP size.
    gop_size: u32,
    /// Frame counter.
    frame_counter: u64,
}

impl LookaheadBuffer {
    /// Create a new lookahead buffer.
    pub fn new(config: &ThreadingConfig, max_bframes: u8, gop_size: u32) -> Self {
        let num_threads = config.effective_threads();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|idx| format!("h264-la-{}", idx))
            .build()
            .expect("Failed to create lookahead thread pool");

        Self {
            max_depth: config.lookahead_depth,
            frames: Vec::with_capacity(config.lookahead_depth),
            thread_pool,
            max_bframes,
            gop_size,
            frame_counter: 0,
        }
    }

    /// Add a frame to the lookahead buffer.
    pub fn push(&mut self, frame: Frame) -> Option<Vec<LookaheadFrame>> {
        let display_order = self.frame_counter;
        self.frame_counter += 1;

        let lookahead_frame = LookaheadFrame {
            frame,
            display_order,
            frame_type: FrameType::P, // Tentative, will be updated
            complexity: 0.0,
            motion_results: Vec::new(),
        };

        self.frames.push(lookahead_frame);

        // When buffer is full, analyze and return frames ready for encoding
        if self.frames.len() >= self.max_depth {
            Some(self.flush_ready_frames())
        } else {
            None
        }
    }

    /// Flush remaining frames (for end of stream).
    pub fn flush(&mut self) -> Vec<LookaheadFrame> {
        self.analyze_buffer();
        std::mem::take(&mut self.frames)
    }

    /// Analyze buffered frames and determine frame types.
    fn analyze_buffer(&mut self) {
        if self.frames.is_empty() {
            return;
        }

        // Calculate complexity for each frame in parallel
        let complexities: Vec<f64> = self.thread_pool.install(|| {
            self.frames
                .par_iter()
                .map(|f| Self::calculate_frame_complexity(&f.frame))
                .collect()
        });

        for (frame, complexity) in self.frames.iter_mut().zip(complexities) {
            frame.complexity = complexity;
        }

        // Determine frame types based on GOP structure and complexity
        self.determine_frame_types();
    }

    /// Calculate frame complexity (spatial variance).
    fn calculate_frame_complexity(frame: &Frame) -> f64 {
        let luma = match frame.plane(0) {
            Some(data) => data,
            None => return 0.0,
        };

        if luma.is_empty() {
            return 0.0;
        }

        // Calculate variance as complexity measure
        let mean: f64 = luma.iter().map(|&p| p as f64).sum::<f64>() / luma.len() as f64;
        let variance: f64 = luma.iter()
            .map(|&p| {
                let diff = p as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / luma.len() as f64;

        variance.sqrt()
    }

    /// Determine frame types based on GOP structure.
    fn determine_frame_types(&mut self) {
        for (idx, frame) in self.frames.iter_mut().enumerate() {
            let pos_in_gop = frame.display_order % self.gop_size as u64;

            if pos_in_gop == 0 {
                // IDR/I-frame
                frame.frame_type = FrameType::I;
            } else if self.max_bframes == 0 {
                // No B-frames
                frame.frame_type = FrameType::P;
            } else {
                // Determine if B or P frame
                let bframe_interval = self.max_bframes as u64 + 1;
                if pos_in_gop.is_multiple_of(bframe_interval) {
                    frame.frame_type = FrameType::P;
                } else {
                    frame.frame_type = FrameType::B;
                }
            }
        }

        // Scene change detection - force I-frame on high complexity change
        for i in 1..self.frames.len() {
            let prev_complexity = self.frames[i - 1].complexity;
            let curr_complexity = self.frames[i].complexity;

            // If complexity change is more than 50%, consider it a scene change
            if prev_complexity > 0.0
                && (curr_complexity - prev_complexity).abs() / prev_complexity > 0.5
                && curr_complexity > prev_complexity
            {
                self.frames[i].frame_type = FrameType::I;
            }
        }
    }

    /// Flush frames that are ready for encoding.
    fn flush_ready_frames(&mut self) -> Vec<LookaheadFrame> {
        self.analyze_buffer();

        // Return first batch of frames (mini-GOP)
        let batch_size = (self.max_bframes as usize + 1).min(self.frames.len());
        let ready: Vec<LookaheadFrame> = self.frames.drain(0..batch_size).collect();
        ready
    }
}

/// Thread-safe encoder state for parallel encoding.
pub struct ThreadSafeEncoderState {
    /// Current frame number.
    frame_num: Mutex<u64>,
    /// IDR frame counter.
    idr_count: Mutex<u64>,
    /// Reference frames (DPB).
    reference_frames: RwLock<Vec<Arc<RwLock<ReferenceFrame>>>>,
    /// Maximum reference frames.
    max_refs: usize,
    /// Rate control state.
    rate_control: Mutex<RateControlState>,
}

/// Rate control state for thread-safe updates.
#[derive(Debug, Clone)]
pub struct RateControlState {
    /// Target bits per frame.
    pub target_bits: f64,
    /// Accumulated bits.
    pub accumulated_bits: i64,
    /// Last QP used.
    pub last_qp: u8,
    /// Frame count.
    pub frame_count: u64,
}

impl Default for RateControlState {
    fn default() -> Self {
        Self {
            target_bits: 0.0,
            accumulated_bits: 0,
            last_qp: 26,
            frame_count: 0,
        }
    }
}

impl ThreadSafeEncoderState {
    /// Create new thread-safe encoder state.
    pub fn new(max_refs: usize) -> Self {
        Self {
            frame_num: Mutex::new(0),
            idr_count: Mutex::new(0),
            reference_frames: RwLock::new(Vec::with_capacity(max_refs)),
            max_refs,
            rate_control: Mutex::new(RateControlState::default()),
        }
    }

    /// Increment and get frame number.
    pub fn next_frame_num(&self) -> u64 {
        let mut num = self.frame_num.lock();
        let current = *num;
        *num += 1;
        current
    }

    /// Increment and get IDR count.
    pub fn next_idr_count(&self) -> u64 {
        let mut count = self.idr_count.lock();
        let current = *count;
        *count += 1;
        current
    }

    /// Reset frame number (after IDR).
    pub fn reset_frame_num(&self) {
        *self.frame_num.lock() = 0;
    }

    /// Add a reference frame to the DPB.
    pub fn add_reference_frame(&self, ref_frame: ReferenceFrame) {
        let mut refs = self.reference_frames.write();
        refs.push(Arc::new(RwLock::new(ref_frame)));

        // Maintain max reference frame count
        while refs.len() > self.max_refs {
            refs.remove(0);
        }
    }

    /// Get reference frames for motion estimation.
    pub fn get_reference_frames(&self) -> Vec<Arc<RwLock<ReferenceFrame>>> {
        self.reference_frames.read().clone()
    }

    /// Clear all reference frames.
    pub fn clear_references(&self) {
        self.reference_frames.write().clear();
    }

    /// Update rate control state.
    pub fn update_rate_control(&self, bits_used: u32, qp: u8) {
        let mut rc = self.rate_control.lock();
        rc.accumulated_bits += bits_used as i64;
        rc.last_qp = qp;
        rc.frame_count += 1;
    }

    /// Get recommended QP from rate control.
    pub fn get_recommended_qp(&self, base_qp: u8) -> u8 {
        let rc = self.rate_control.lock();

        if rc.frame_count == 0 || rc.target_bits <= 0.0 {
            return base_qp;
        }

        // Simple rate control: adjust QP based on buffer fullness
        let expected_bits = rc.target_bits * rc.frame_count as f64;
        let bit_error = rc.accumulated_bits as f64 - expected_bits;
        let qp_adjustment = (bit_error / rc.target_bits).clamp(-6.0, 6.0) as i8;

        (base_qp as i8 + qp_adjustment).clamp(0, 51) as u8
    }

    /// Set target bitrate for rate control.
    pub fn set_target_bitrate(&self, bitrate: u32, frame_rate: f64) {
        let mut rc = self.rate_control.lock();
        rc.target_bits = bitrate as f64 / frame_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::{PixelFormat, TimeBase, Timestamp};

    fn create_test_frame(width: u32, height: u32, frame_num: u32) -> Frame {
        let time_base = TimeBase::new(1, 30);
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, time_base);
        frame.pts = Timestamp::new(frame_num as i64, time_base);

        // Fill with gradient pattern
        if let Some(y_plane) = frame.plane_mut(0) {
            for (i, pixel) in y_plane.iter_mut().enumerate() {
                *pixel = ((i + frame_num as usize * 10) % 256) as u8;
            }
        }

        if let Some(u_plane) = frame.plane_mut(1) {
            for pixel in u_plane.iter_mut() {
                *pixel = 128;
            }
        }

        if let Some(v_plane) = frame.plane_mut(2) {
            for pixel in v_plane.iter_mut() {
                *pixel = 128;
            }
        }

        frame
    }

    #[test]
    fn test_threading_config_default() {
        let config = ThreadingConfig::default();
        assert!(config.effective_threads() >= 1);
        assert!(config.slice_count >= 2);
        assert_eq!(config.lookahead_depth, 40);
    }

    #[test]
    fn test_threading_config_with_threads() {
        let config = ThreadingConfig::with_threads(8);
        assert_eq!(config.effective_threads(), 8);
    }

    #[test]
    fn test_parallel_slice_encoder_split() {
        let config = ThreadingConfig::default().with_slice_count(4);
        let encoder = ParallelSliceEncoder::new(&config);

        let slices = encoder.split_into_slices(720);
        assert_eq!(slices.len(), 4);

        // Verify coverage
        let total_rows: usize = slices.iter().map(|(_, count)| *count).sum();
        assert_eq!(total_rows, (720 + 15) / 16);
    }

    #[test]
    fn test_lookahead_buffer() {
        let config = ThreadingConfig::default().with_lookahead_depth(8);
        let mut buffer = LookaheadBuffer::new(&config, 2, 30);

        // Add frames
        for i in 0..7 {
            let frame = create_test_frame(320, 240, i);
            let result = buffer.push(frame);
            assert!(result.is_none());
        }

        // 8th frame should trigger flush
        let frame = create_test_frame(320, 240, 7);
        let result = buffer.push(frame);
        assert!(result.is_some());
    }

    #[test]
    fn test_thread_safe_state() {
        let state = ThreadSafeEncoderState::new(4);

        assert_eq!(state.next_frame_num(), 0);
        assert_eq!(state.next_frame_num(), 1);
        assert_eq!(state.next_frame_num(), 2);

        state.reset_frame_num();
        assert_eq!(state.next_frame_num(), 0);
    }

    #[test]
    fn test_reference_frame_management() {
        let state = ThreadSafeEncoderState::new(2);
        let frame = create_test_frame(320, 240, 0);

        state.add_reference_frame(ReferenceFrame::from_frame(&frame, 0, 0));
        assert_eq!(state.get_reference_frames().len(), 1);

        state.add_reference_frame(ReferenceFrame::from_frame(&frame, 1, 2));
        assert_eq!(state.get_reference_frames().len(), 2);

        // Adding third should evict oldest
        state.add_reference_frame(ReferenceFrame::from_frame(&frame, 2, 4));
        assert_eq!(state.get_reference_frames().len(), 2);

        state.clear_references();
        assert_eq!(state.get_reference_frames().len(), 0);
    }

    #[test]
    fn test_parallel_motion_estimator() {
        let config = ThreadingConfig::with_threads(2);
        let estimator = ParallelMotionEstimator::new(&config);

        let frame = create_test_frame(320, 240, 0);
        let results = estimator.estimate_motion(&frame, &[]);

        // Should have one result per macroblock
        let mb_count = ((320 + 15) / 16) * ((240 + 15) / 16);
        assert_eq!(results.len(), mb_count);
    }

    #[test]
    fn test_frame_complexity() {
        let frame = create_test_frame(320, 240, 0);
        let complexity = LookaheadBuffer::calculate_frame_complexity(&frame);
        assert!(complexity > 0.0);
    }
}
