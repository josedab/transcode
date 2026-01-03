//! Shared parallel encoding infrastructure for video codecs.
//!
//! This module provides reusable components for parallel video encoding:
//! - Thread pool configuration and management
//! - Parallel motion estimation
//! - Slice/row-based parallelism
//! - Frame batch processing
//! - Lookahead buffer for frame type decisions

#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_is_multiple_of)]

use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use rayon::ThreadPool;
use std::sync::Arc;
use transcode_core::Frame;

/// Thread pool configuration for parallel encoding.
#[derive(Debug, Clone)]
pub struct ThreadConfig {
    /// Number of threads to use (0 = auto-detect based on CPU cores).
    pub num_threads: usize,
    /// Number of slices/rows per frame for parallel encoding.
    pub slice_count: usize,
    /// Lookahead depth for frame type decisions.
    pub lookahead_depth: usize,
    /// Enable slice-based parallelism.
    pub enable_slice_parallel: bool,
    /// Enable frame-level motion estimation parallelism.
    pub enable_frame_parallel: bool,
}

impl Default for ThreadConfig {
    fn default() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);

        Self {
            num_threads: 0, // Auto-detect
            slice_count: num_cpus.max(2),
            lookahead_depth: 40,
            enable_slice_parallel: true,
            enable_frame_parallel: true,
        }
    }
}

impl ThreadConfig {
    /// Create a new configuration with custom thread count.
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

/// Frame type for encoding decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameType {
    /// Intra frame (keyframe).
    I,
    /// Predicted frame (forward reference only).
    #[default]
    P,
    /// Bidirectional predicted frame.
    B,
}

/// Motion vector representation.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotionVector {
    /// Horizontal component (quarter-pel precision).
    pub x: i16,
    /// Vertical component (quarter-pel precision).
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
}

/// Motion estimation result for a single block.
#[derive(Debug, Clone, Default)]
pub struct MotionResult {
    /// Best motion vector for forward prediction (L0).
    pub mv_l0: MotionVector,
    /// Best motion vector for backward prediction (L1, for B-frames).
    pub mv_l1: MotionVector,
    /// Reference frame index for L0.
    pub ref_idx_l0: u8,
    /// Reference frame index for L1.
    pub ref_idx_l1: u8,
    /// Motion estimation cost (SAD/SATD).
    pub cost: u32,
    /// Intra prediction cost for mode decision.
    pub intra_cost: u32,
}

/// Reference frame data for motion estimation.
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
}

impl ReferenceFrame {
    /// Create a new reference frame from a decoded frame.
    pub fn from_frame(frame: &Frame, frame_num: u64, poc: i32) -> Self {
        Self {
            frame_num,
            poc,
            luma: frame.plane(0).map(|p| p.to_vec()).unwrap_or_default(),
            chroma_u: frame.plane(1).map(|p| p.to_vec()).unwrap_or_default(),
            chroma_v: frame.plane(2).map(|p| p.to_vec()).unwrap_or_default(),
            width: frame.width(),
            height: frame.height(),
        }
    }
}

/// Lookahead frame entry for frame type decisions.
#[derive(Debug, Clone)]
pub struct LookaheadFrame {
    /// Original frame.
    pub frame: Frame,
    /// Frame display order.
    pub display_order: u64,
    /// Computed frame type.
    pub frame_type: FrameType,
    /// Computed complexity score.
    pub complexity: f64,
    /// Motion estimation results (per block).
    pub motion_results: Vec<MotionResult>,
}

/// Generic parallel motion estimator.
pub struct ParallelMotionEstimator {
    /// Thread pool for parallel operations.
    thread_pool: ThreadPool,
    /// Search range for motion estimation.
    search_range: i16,
    /// Block size (16 for H.264 macroblocks, 64 for HEVC CTU).
    block_size: u32,
}

impl ParallelMotionEstimator {
    /// Create a new parallel motion estimator.
    pub fn new(config: &ThreadConfig, block_size: u32) -> Self {
        let num_threads = config.effective_threads();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|idx| format!("me-{}", idx))
            .build()
            .expect("Failed to create motion estimation thread pool");

        Self {
            thread_pool,
            search_range: 64,
            block_size,
        }
    }

    /// Set the search range for motion estimation.
    pub fn with_search_range(mut self, range: i16) -> Self {
        self.search_range = range;
        self
    }

    /// Perform parallel motion estimation for a frame.
    pub fn estimate_motion(
        &self,
        current_frame: &Frame,
        reference_frames: &[Arc<RwLock<ReferenceFrame>>],
    ) -> Vec<MotionResult> {
        let width = current_frame.width() as usize;
        let height = current_frame.height() as usize;
        let block_size = self.block_size as usize;
        let blocks_x = width.div_ceil(block_size);
        let blocks_y = height.div_ceil(block_size);
        let block_count = blocks_x * blocks_y;

        let current_luma = current_frame.plane(0).unwrap_or(&[]);
        let stride = current_frame.stride(0);
        let search_range = self.search_range;

        // Create read-only references for thread safety
        let ref_data: Vec<_> = reference_frames
            .iter()
            .map(|r| r.read().clone())
            .collect();

        self.thread_pool.install(|| {
            (0..block_count)
                .into_par_iter()
                .map(|block_idx| {
                    let block_x = block_idx % blocks_x;
                    let block_y = block_idx / blocks_x;

                    Self::estimate_block_motion(
                        current_luma,
                        stride,
                        width,
                        height,
                        block_x,
                        block_y,
                        block_size,
                        &ref_data,
                        search_range,
                    )
                })
                .collect()
        })
    }

    /// Estimate motion for a single block.
    fn estimate_block_motion(
        current_luma: &[u8],
        stride: usize,
        width: usize,
        height: usize,
        block_x: usize,
        block_y: usize,
        block_size: usize,
        reference_frames: &[ReferenceFrame],
        search_range: i16,
    ) -> MotionResult {
        let mut result = MotionResult {
            cost: u32::MAX,
            ..Default::default()
        };

        let px_x = block_x * block_size;
        let px_y = block_y * block_size;

        // Calculate intra cost
        result.intra_cost = Self::calculate_intra_cost(
            current_luma,
            stride,
            px_x,
            px_y,
            block_size,
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

            let (mv, cost) = Self::diamond_search(
                current_luma,
                stride,
                ref_luma,
                ref_stride,
                px_x,
                px_y,
                block_size,
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
        px_x: usize,
        px_y: usize,
        block_size: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        if px_x + block_size > width || px_y + block_size > height {
            return u32::MAX;
        }

        let mut sum: u32 = 0;
        for y in 0..block_size {
            for x in 0..block_size {
                let idx = (px_y + y) * stride + px_x + x;
                if idx < luma.len() {
                    sum += luma[idx] as u32;
                }
            }
        }
        let dc = (sum / (block_size * block_size) as u32) as u8;

        let mut sad: u32 = 0;
        for y in 0..block_size {
            for x in 0..block_size {
                let idx = (px_y + y) * stride + px_x + x;
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
        px_x: usize,
        px_y: usize,
        block_size: usize,
        width: usize,
        height: usize,
        search_range: i16,
    ) -> (MotionVector, u32) {
        let mut best_mv = MotionVector::zero();
        let mut best_cost = Self::calculate_sad(
            current, cur_stride,
            reference, ref_stride,
            px_x, px_y, 0, 0,
            block_size, width, height,
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
                    px_x, px_y, mv_x, mv_y,
                    block_size, width, height,
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
                    px_x, px_y, mv_x, mv_y,
                    block_size, width, height,
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

    /// Calculate SAD between blocks.
    fn calculate_sad(
        current: &[u8],
        cur_stride: usize,
        reference: &[u8],
        ref_stride: usize,
        px_x: usize,
        px_y: usize,
        mv_x: i16,
        mv_y: i16,
        block_size: usize,
        width: usize,
        height: usize,
    ) -> u32 {
        let ref_x = px_x as i32 + mv_x as i32;
        let ref_y = px_y as i32 + mv_y as i32;

        if ref_x < 0 || ref_y < 0
            || ref_x as usize + block_size > width
            || ref_y as usize + block_size > height
            || px_x + block_size > width
            || px_y + block_size > height
        {
            return u32::MAX;
        }

        let ref_x = ref_x as usize;
        let ref_y = ref_y as usize;

        let mut sad: u32 = 0;
        for y in 0..block_size {
            for x in 0..block_size {
                let cur_idx = (px_y + y) * cur_stride + px_x + x;
                let ref_idx = (ref_y + y) * ref_stride + ref_x + x;

                if cur_idx < current.len() && ref_idx < reference.len() {
                    sad += (current[cur_idx] as i32 - reference[ref_idx] as i32).unsigned_abs();
                }
            }
        }

        sad
    }
}

/// Parallel row/slice encoder.
pub struct ParallelRowEncoder {
    /// Thread pool for parallel encoding.
    thread_pool: ThreadPool,
    /// Number of slices/rows to encode in parallel.
    slice_count: usize,
}

impl ParallelRowEncoder {
    /// Create a new parallel row encoder.
    pub fn new(config: &ThreadConfig) -> Self {
        let num_threads = config.effective_threads();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|idx| format!("row-enc-{}", idx))
            .build()
            .expect("Failed to create row encoding thread pool");

        Self {
            thread_pool,
            slice_count: config.slice_count,
        }
    }

    /// Split a frame into row ranges for parallel encoding.
    pub fn split_into_rows(&self, frame_height: u32, row_size: u32) -> Vec<(usize, usize)> {
        let row_count = frame_height.div_ceil(row_size) as usize;
        let slice_count = self.slice_count.min(row_count);

        let rows_per_slice = row_count / slice_count;
        let extra_rows = row_count % slice_count;

        let mut slices = Vec::with_capacity(slice_count);
        let mut current_row = 0;

        for i in 0..slice_count {
            let rows = rows_per_slice + if i < extra_rows { 1 } else { 0 };
            slices.push((current_row, rows));
            current_row += rows;
        }

        slices
    }

    /// Encode rows in parallel using a custom encoding function.
    pub fn encode_rows_parallel<T, F>(
        &self,
        frame_height: u32,
        row_size: u32,
        encode_row_fn: F,
    ) -> Vec<T>
    where
        T: Send,
        F: Fn(usize, usize) -> T + Sync,
    {
        let row_bounds = self.split_into_rows(frame_height, row_size);

        self.thread_pool.install(|| {
            row_bounds
                .into_par_iter()
                .map(|(first_row, row_count)| encode_row_fn(first_row, row_count))
                .collect()
        })
    }
}

/// Lookahead buffer for frame type decisions.
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
    /// Scene change threshold (0.0-1.0).
    scene_change_threshold: f64,
}

impl LookaheadBuffer {
    /// Create a new lookahead buffer.
    pub fn new(config: &ThreadConfig, max_bframes: u8, gop_size: u32) -> Self {
        let num_threads = config.effective_threads();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|idx| format!("lookahead-{}", idx))
            .build()
            .expect("Failed to create lookahead thread pool");

        Self {
            max_depth: config.lookahead_depth,
            frames: Vec::with_capacity(config.lookahead_depth),
            thread_pool,
            max_bframes,
            gop_size,
            frame_counter: 0,
            scene_change_threshold: 0.5,
        }
    }

    /// Set the scene change detection threshold.
    pub fn with_scene_change_threshold(mut self, threshold: f64) -> Self {
        self.scene_change_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Add a frame to the lookahead buffer.
    pub fn push(&mut self, frame: Frame) -> Option<Vec<LookaheadFrame>> {
        let display_order = self.frame_counter;
        self.frame_counter += 1;

        let lookahead_frame = LookaheadFrame {
            frame,
            display_order,
            frame_type: FrameType::P,
            complexity: 0.0,
            motion_results: Vec::new(),
        };

        self.frames.push(lookahead_frame);

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

        let mean: f64 = luma.iter().map(|&p| p as f64).sum::<f64>() / luma.len() as f64;
        let variance: f64 = luma
            .iter()
            .map(|&p| {
                let diff = p as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / luma.len() as f64;

        variance.sqrt()
    }

    /// Determine frame types based on GOP structure.
    fn determine_frame_types(&mut self) {
        for frame in self.frames.iter_mut() {
            let pos_in_gop = frame.display_order % self.gop_size as u64;

            if pos_in_gop == 0 {
                frame.frame_type = FrameType::I;
            } else if self.max_bframes == 0 {
                frame.frame_type = FrameType::P;
            } else {
                let bframe_interval = self.max_bframes as u64 + 1;
                if pos_in_gop % bframe_interval == 0 {
                    frame.frame_type = FrameType::P;
                } else {
                    frame.frame_type = FrameType::B;
                }
            }
        }

        // Scene change detection
        for i in 1..self.frames.len() {
            let prev_complexity = self.frames[i - 1].complexity;
            let curr_complexity = self.frames[i].complexity;

            if prev_complexity > 0.0
                && (curr_complexity - prev_complexity).abs() / prev_complexity
                    > self.scene_change_threshold
                && curr_complexity > prev_complexity
            {
                self.frames[i].frame_type = FrameType::I;
            }
        }
    }

    /// Flush frames that are ready for encoding.
    fn flush_ready_frames(&mut self) -> Vec<LookaheadFrame> {
        self.analyze_buffer();
        let batch_size = (self.max_bframes as usize + 1).min(self.frames.len());
        self.frames.drain(0..batch_size).collect()
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

/// Frame batch processor for multi-frame parallel encoding.
pub struct FrameBatchProcessor {
    /// Thread pool for parallel processing.
    thread_pool: ThreadPool,
    /// Maximum batch size.
    max_batch_size: usize,
}

impl FrameBatchProcessor {
    /// Create a new frame batch processor.
    pub fn new(config: &ThreadConfig, max_batch_size: usize) -> Self {
        let num_threads = config.effective_threads();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|idx| format!("batch-{}", idx))
            .build()
            .expect("Failed to create batch processing thread pool");

        Self {
            thread_pool,
            max_batch_size,
        }
    }

    /// Process a batch of frames in parallel.
    #[allow(clippy::redundant_closure)]
    pub fn process_batch<T, F>(&self, frames: &[Frame], process_fn: F) -> Vec<T>
    where
        T: Send,
        F: Fn(&Frame) -> T + Sync,
    {
        self.thread_pool.install(|| {
            frames
                .par_iter()
                .map(|frame| process_fn(frame))
                .collect()
        })
    }

    /// Process frames with index.
    pub fn process_batch_indexed<T, F>(&self, frames: &[Frame], process_fn: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize, &Frame) -> T + Sync,
    {
        self.thread_pool.install(|| {
            frames
                .par_iter()
                .enumerate()
                .map(|(idx, frame)| process_fn(idx, frame))
                .collect()
        })
    }

    /// Get max batch size.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
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

        if let Some(y_plane) = frame.plane_mut(0) {
            for (i, pixel) in y_plane.iter_mut().enumerate() {
                *pixel = ((i + frame_num as usize * 10) % 256) as u8;
            }
        }

        frame
    }

    #[test]
    fn test_thread_config_default() {
        let config = ThreadConfig::default();
        assert!(config.effective_threads() >= 1);
        assert!(config.slice_count >= 2);
        assert_eq!(config.lookahead_depth, 40);
    }

    #[test]
    fn test_thread_config_with_threads() {
        let config = ThreadConfig::with_threads(8);
        assert_eq!(config.effective_threads(), 8);
    }

    #[test]
    fn test_parallel_row_encoder_split() {
        let config = ThreadConfig::default().with_slice_count(4);
        let encoder = ParallelRowEncoder::new(&config);

        let rows = encoder.split_into_rows(720, 16);
        assert_eq!(rows.len(), 4);

        let total_rows: usize = rows.iter().map(|(_, count)| *count).sum();
        assert_eq!(total_rows, (720 + 15) / 16);
    }

    #[test]
    fn test_lookahead_buffer() {
        let config = ThreadConfig::default().with_lookahead_depth(8);
        let mut buffer = LookaheadBuffer::new(&config, 2, 30);

        for i in 0..7 {
            let frame = create_test_frame(320, 240, i);
            let result = buffer.push(frame);
            assert!(result.is_none());
        }

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

        state.add_reference_frame(ReferenceFrame::from_frame(&frame, 2, 4));
        assert_eq!(state.get_reference_frames().len(), 2);

        state.clear_references();
        assert_eq!(state.get_reference_frames().len(), 0);
    }

    #[test]
    fn test_parallel_motion_estimator() {
        let config = ThreadConfig::with_threads(2);
        let estimator = ParallelMotionEstimator::new(&config, 16);

        let frame = create_test_frame(320, 240, 0);
        let results = estimator.estimate_motion(&frame, &[]);

        let block_count = ((320 + 15) / 16) * ((240 + 15) / 16);
        assert_eq!(results.len(), block_count);
    }

    #[test]
    fn test_frame_batch_processor() {
        let config = ThreadConfig::with_threads(2);
        let processor = FrameBatchProcessor::new(&config, 8);

        let frames: Vec<Frame> = (0..4).map(|i| create_test_frame(320, 240, i)).collect();

        let complexities: Vec<f64> = processor.process_batch(&frames, |f| {
            let luma = f.plane(0).unwrap_or(&[]);
            luma.iter().map(|&p| p as f64).sum::<f64>() / luma.len() as f64
        });

        assert_eq!(complexities.len(), 4);
    }

    #[test]
    fn test_frame_complexity() {
        let frame = create_test_frame(320, 240, 0);
        let complexity = LookaheadBuffer::calculate_frame_complexity(&frame);
        assert!(complexity > 0.0);
    }

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector::new(10, -5);
        assert_eq!(mv.x, 10);
        assert_eq!(mv.y, -5);

        let zero = MotionVector::zero();
        assert_eq!(zero.x, 0);
        assert_eq!(zero.y, 0);
    }
}
