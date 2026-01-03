//! YADIF (Yet Another Deinterlacing Filter) implementation.
//!
//! YADIF is a motion-adaptive deinterlacing filter that combines spatial
//! and temporal information to produce high-quality progressive frames.
//!
//! # Algorithm Overview
//!
//! YADIF operates on a sliding window of three consecutive frames:
//! - Previous frame
//! - Current frame (being processed)
//! - Next frame
//!
//! For each pixel position that needs to be interpolated:
//!
//! 1. **Spatial Prediction**: Calculate a spatial-only prediction using
//!    pixels from the current field (Catmull-Rom interpolation).
//!
//! 2. **Temporal Prediction**: Calculate predictions using corresponding
//!    pixels from the previous and next frames.
//!
//! 3. **Motion Detection**: Compare spatial and temporal predictions to
//!    detect motion.
//!
//! 4. **Adaptive Selection**: Use spatial prediction in areas with motion,
//!    temporal prediction in static areas.
//!
//! # References
//!
//! - Original implementation by Michael Niedermayer
//! - FFmpeg libavfilter/vf_yadif.c

use crate::error::{DeinterlaceError, Result};
use std::collections::VecDeque;
use transcode_core::frame::FrameFlags;
use transcode_core::{Frame, FrameBuffer, PixelFormat};

/// YADIF operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum YadifMode {
    /// Send one frame for each frame (frame rate preserved).
    #[default]
    Frame,
    /// Send one frame for each field (double frame rate).
    Field,
    /// Like Frame, but skip spatial interlacing check.
    FrameNoSpatial,
    /// Like Field, but skip spatial interlacing check.
    FieldNoSpatial,
}

/// YADIF parity (which field to keep).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum YadifParity {
    /// Top field first / keep top field.
    #[default]
    Top,
    /// Bottom field first / keep bottom field.
    Bottom,
    /// Automatic detection from frame flags.
    Auto,
}

/// YADIF configuration.
#[derive(Debug, Clone)]
pub struct YadifConfig {
    /// Operating mode.
    pub mode: YadifMode,
    /// Field parity.
    pub parity: YadifParity,
    /// Deinterlace all frames or only marked ones.
    pub deint_all: bool,
}

impl Default for YadifConfig {
    fn default() -> Self {
        Self {
            mode: YadifMode::Frame,
            parity: YadifParity::Auto,
            deint_all: true,
        }
    }
}

/// YADIF deinterlacer state.
pub struct YadifDeinterlacer {
    config: YadifConfig,
    /// Frame buffer for temporal processing (prev, cur, next).
    frame_buffer: VecDeque<Frame>,
    /// Expected dimensions.
    width: u32,
    height: u32,
    /// Expected format.
    format: PixelFormat,
    /// Whether the filter is initialized.
    initialized: bool,
}

impl YadifDeinterlacer {
    /// Create a new YADIF deinterlacer.
    pub fn new() -> Self {
        Self {
            config: YadifConfig::default(),
            frame_buffer: VecDeque::with_capacity(3),
            width: 0,
            height: 0,
            format: PixelFormat::Yuv420p,
            initialized: false,
        }
    }

    /// Create a new YADIF deinterlacer with custom configuration.
    pub fn with_config(config: YadifConfig) -> Self {
        Self {
            config,
            frame_buffer: VecDeque::with_capacity(3),
            width: 0,
            height: 0,
            format: PixelFormat::Yuv420p,
            initialized: false,
        }
    }

    /// Reset the deinterlacer state.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.initialized = false;
    }

    /// Push a new frame into the filter.
    ///
    /// Returns output frames when enough context is available.
    pub fn push_frame(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        self.validate_frame(&frame)?;

        // Initialize on first frame
        if !self.initialized {
            self.width = frame.width();
            self.height = frame.height();
            self.format = frame.format();
            self.initialized = true;
        }

        // Check frame compatibility
        if frame.width() != self.width || frame.height() != self.height {
            return Err(DeinterlaceError::frame_mismatch(
                self.width,
                self.height,
                frame.width(),
                frame.height(),
            ));
        }

        self.frame_buffer.push_back(frame);

        // Need at least 2 frames to start processing
        if self.frame_buffer.len() < 2 {
            return Ok(Vec::new());
        }

        // Keep only 3 frames maximum
        while self.frame_buffer.len() > 3 {
            self.frame_buffer.pop_front();
        }

        // Process the middle frame (or first if we only have 2)
        self.process_current()
    }

    /// Flush remaining frames from the buffer.
    pub fn flush(&mut self) -> Result<Vec<Frame>> {
        let mut output = Vec::new();

        while self.frame_buffer.len() >= 2 {
            output.extend(self.process_current()?);
            self.frame_buffer.pop_front();
        }

        // Process final frame if any remains
        if let Some(last) = self.frame_buffer.pop_front() {
            output.push(self.deinterlace_single(&last, true)?);
        }

        Ok(output)
    }

    /// Process the current frame in the buffer.
    fn process_current(&mut self) -> Result<Vec<Frame>> {
        let mut results = Vec::new();

        let frame_count = self.frame_buffer.len();
        if frame_count < 2 {
            return Ok(results);
        }

        // Get frame references
        let (prev, cur, next) = if frame_count >= 3 {
            (
                Some(&self.frame_buffer[0]),
                &self.frame_buffer[1],
                Some(&self.frame_buffer[2]),
            )
        } else {
            (None, &self.frame_buffer[0], Some(&self.frame_buffer[1]))
        };

        // Check if frame needs deinterlacing
        let is_interlaced = cur.flags.contains(FrameFlags::INTERLACED) || self.config.deint_all;

        if !is_interlaced {
            // Pass through progressive frames
            results.push(cur.clone());
            return Ok(results);
        }

        // Determine field order
        let tff = match self.config.parity {
            YadifParity::Top => true,
            YadifParity::Bottom => false,
            YadifParity::Auto => cur.flags.contains(FrameFlags::TOP_FIELD_FIRST),
        };

        match self.config.mode {
            YadifMode::Frame | YadifMode::FrameNoSpatial => {
                // Output one frame per input frame
                let output = self.filter_frame(prev, cur, next, tff, 0)?;
                results.push(output);
            }
            YadifMode::Field | YadifMode::FieldNoSpatial => {
                // Output two frames per input frame (one per field)
                let output1 = self.filter_frame(prev, cur, next, tff, 0)?;
                let output2 = self.filter_frame(prev, cur, next, tff, 1)?;
                results.push(output1);
                results.push(output2);
            }
        }

        Ok(results)
    }

    /// Filter a single frame using YADIF algorithm.
    fn filter_frame(
        &self,
        prev: Option<&Frame>,
        cur: &Frame,
        next: Option<&Frame>,
        tff: bool,
        field: usize,
    ) -> Result<Frame> {
        let width = cur.width();
        let height = cur.height();
        let format = cur.format();

        let mut output = Frame::from_buffer(FrameBuffer::new(width, height, format));
        output.pts = cur.pts;
        output.dts = cur.dts;
        output.duration = cur.duration;
        output.poc = cur.poc;
        output.flags = cur.flags & !FrameFlags::INTERLACED & !FrameFlags::TOP_FIELD_FIRST;

        let num_planes = format.num_planes();
        let (hsub, vsub) = format.chroma_subsampling();

        // Determine which field is being processed
        let parity = if field == 0 { tff } else { !tff };

        for plane_idx in 0..num_planes {
            let plane_height = if plane_idx == 0 {
                height as usize
            } else {
                height as usize / vsub as usize
            };

            let plane_width = if plane_idx == 0 {
                width as usize
            } else {
                width as usize / hsub as usize
            };

            self.filter_plane(
                prev,
                cur,
                next,
                &mut output,
                plane_idx,
                plane_width,
                plane_height,
                parity,
            )?;
        }

        Ok(output)
    }

    /// Filter a single plane.
    #[allow(clippy::too_many_arguments)]
    fn filter_plane(
        &self,
        prev: Option<&Frame>,
        cur: &Frame,
        next: Option<&Frame>,
        output: &mut Frame,
        plane_idx: usize,
        width: usize,
        height: usize,
        parity: bool,
    ) -> Result<()> {
        let cur_plane = cur
            .plane(plane_idx)
            .ok_or_else(|| DeinterlaceError::buffer_error("Current plane not found"))?;

        let cur_stride = cur.stride(plane_idx);
        let dst_stride = output.stride(plane_idx);

        let dst_plane = output
            .plane_mut(plane_idx)
            .ok_or_else(|| DeinterlaceError::buffer_error("Output plane not found"))?;

        // Get previous and next plane data
        let prev_plane = prev.and_then(|f| f.plane(plane_idx));
        let next_plane = next.and_then(|f| f.plane(plane_idx));
        let prev_stride = prev.map(|f| f.stride(plane_idx)).unwrap_or(cur_stride);
        let next_stride = next.map(|f| f.stride(plane_idx)).unwrap_or(cur_stride);

        let use_spatial = !matches!(
            self.config.mode,
            YadifMode::FrameNoSpatial | YadifMode::FieldNoSpatial
        );

        for y in 0..height {
            let dst_offset = y * dst_stride;
            let is_field_line = (y % 2 == 0) == parity;

            if is_field_line {
                // Copy lines from the kept field
                let src_offset = y * cur_stride;
                if src_offset + width <= cur_plane.len() && dst_offset + width <= dst_plane.len() {
                    dst_plane[dst_offset..dst_offset + width]
                        .copy_from_slice(&cur_plane[src_offset..src_offset + width]);
                }
            } else {
                // Interpolate missing lines
                for x in 0..width {
                    let pixel = self.filter_pixel(
                        cur_plane,
                        prev_plane,
                        next_plane,
                        cur_stride,
                        prev_stride,
                        next_stride,
                        x,
                        y,
                        width,
                        height,
                        parity,
                        use_spatial,
                    );

                    if dst_offset + x < dst_plane.len() {
                        dst_plane[dst_offset + x] = pixel;
                    }
                }
            }
        }

        Ok(())
    }

    /// Filter a single pixel using YADIF algorithm.
    #[allow(clippy::too_many_arguments)]
    fn filter_pixel(
        &self,
        cur: &[u8],
        prev: Option<&[u8]>,
        next: Option<&[u8]>,
        cur_stride: usize,
        prev_stride: usize,
        next_stride: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        parity: bool,
        use_spatial: bool,
    ) -> u8 {
        // Get pixels from current frame
        let c = self.get_pixel(cur, cur_stride, x, y, width, height);
        let d = self.get_pixel(cur, cur_stride, x, y.saturating_sub(1), width, height);
        let e = self.get_pixel(cur, cur_stride, x, (y + 1).min(height - 1), width, height);

        // Temporal neighbors (same position in prev and next frames)
        let p = prev
            .map(|p| self.get_pixel(p, prev_stride, x, y, width, height))
            .unwrap_or(c);
        let n = next
            .map(|n| self.get_pixel(n, next_stride, x, y, width, height))
            .unwrap_or(c);

        // Temporal prediction: average of previous and next frame at this position
        let temporal = ((p as i32 + n as i32 + 1) >> 1) as u8;

        if !use_spatial {
            return temporal;
        }

        // Spatial prediction using edge-directed interpolation
        let spatial = self.spatial_predict(cur, cur_stride, x, y, width, height, parity);

        // Calculate temporal difference for motion detection
        let temporal_diff0 = (d as i32 - p as i32).abs();
        let temporal_diff1 = (e as i32 - n as i32).abs();
        let temporal_diff2 = (c as i32 - temporal as i32).abs();

        let max_diff = temporal_diff0.max(temporal_diff1).max(temporal_diff2);

        // Spatial difference
        let spatial_diff = ((d as i32 + e as i32) / 2 - spatial as i32).abs();

        // Motion adaptive selection
        if spatial_diff > max_diff {
            // Motion detected - use spatial prediction
            spatial
        } else {
            // Static area - use temporal prediction with constraints
            let min_val = d.min(e).min(p).min(n);
            let max_val = d.max(e).max(p).max(n);
            temporal.clamp(min_val, max_val)
        }
    }

    /// Spatial prediction using edge-directed interpolation.
    fn spatial_predict(
        &self,
        plane: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        parity: bool,
    ) -> u8 {
        // Get lines above and below from the same field
        let y_above = if parity {
            y.saturating_sub(2)
        } else if y >= 2 {
            y - 2
        } else {
            1
        };

        let y_below = if parity {
            if y + 2 < height { y + 2 } else { height - 1 }
        } else if y + 2 < height {
            y + 2
        } else {
            height - 2
        };

        let above = self.get_pixel(plane, stride, x, y_above, width, height);
        let below = self.get_pixel(plane, stride, x, y_below, width, height);

        // Check for edges by comparing diagonal neighbors
        let score_center = (above as i32 - below as i32).abs();

        let left_above = self.get_pixel(plane, stride, x.saturating_sub(1), y_above, width, height);
        let right_below = self.get_pixel(plane, stride, (x + 1).min(width - 1), y_below, width, height);
        let score_left = (left_above as i32 - right_below as i32).abs();

        let right_above = self.get_pixel(plane, stride, (x + 1).min(width - 1), y_above, width, height);
        let left_below = self.get_pixel(plane, stride, x.saturating_sub(1), y_below, width, height);
        let score_right = (right_above as i32 - left_below as i32).abs();

        // Choose the interpolation direction with minimum difference
        if score_center <= score_left && score_center <= score_right {
            ((above as u32 + below as u32 + 1) >> 1) as u8
        } else if score_left < score_right {
            ((left_above as u32 + right_below as u32 + 1) >> 1) as u8
        } else {
            ((right_above as u32 + left_below as u32 + 1) >> 1) as u8
        }
    }

    /// Get a pixel value with bounds checking.
    fn get_pixel(
        &self,
        plane: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> u8 {
        let x = x.min(width - 1);
        let y = y.min(height - 1);
        let offset = y * stride + x;
        plane.get(offset).copied().unwrap_or(128)
    }

    /// Deinterlace a single frame without temporal context.
    fn deinterlace_single(&self, frame: &Frame, tff: bool) -> Result<Frame> {
        self.filter_frame(None, frame, None, tff, 0)
    }

    /// Validate frame for processing.
    fn validate_frame(&self, frame: &Frame) -> Result<()> {
        let width = frame.width();
        let height = frame.height();

        if width < 4 || height < 4 {
            return Err(DeinterlaceError::invalid_dimensions(width, height));
        }

        match frame.format() {
            PixelFormat::Yuv420p
            | PixelFormat::Yuv422p
            | PixelFormat::Yuv444p
            | PixelFormat::Gray8 => Ok(()),
            format => Err(DeinterlaceError::unsupported_format(format.to_string())),
        }
    }
}

impl Default for YadifDeinterlacer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::TimeBase;

    fn create_interlaced_frame(width: u32, height: u32, pattern: u8) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);
        frame.flags = FrameFlags::INTERLACED | FrameFlags::TOP_FIELD_FIRST;

        let stride = frame.stride(0);
        if let Some(y_plane) = frame.plane_mut(0) {
            for y in 0..height as usize {
                let value = if y % 2 == 0 {
                    pattern
                } else {
                    255 - pattern
                };
                for x in 0..width as usize {
                    y_plane[y * stride + x] = value;
                }
            }
        }

        frame
    }

    #[test]
    fn test_yadif_basic() {
        let mut yadif = YadifDeinterlacer::new();

        let frame1 = create_interlaced_frame(16, 16, 50);
        let frame2 = create_interlaced_frame(16, 16, 100);
        let frame3 = create_interlaced_frame(16, 16, 150);

        let result1 = yadif.push_frame(frame1).unwrap();
        assert!(result1.is_empty()); // Need more frames

        let result2 = yadif.push_frame(frame2).unwrap();
        assert_eq!(result2.len(), 1);

        let result3 = yadif.push_frame(frame3).unwrap();
        assert_eq!(result3.len(), 1);

        let flush = yadif.flush().unwrap();
        assert!(!flush.is_empty());
    }

    #[test]
    fn test_yadif_field_mode() {
        let mut yadif = YadifDeinterlacer::with_config(YadifConfig {
            mode: YadifMode::Field,
            ..Default::default()
        });

        let frame1 = create_interlaced_frame(16, 16, 50);
        let frame2 = create_interlaced_frame(16, 16, 100);

        yadif.push_frame(frame1).unwrap();
        let result = yadif.push_frame(frame2).unwrap();

        // Field mode should output 2 frames per input
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_yadif_reset() {
        let mut yadif = YadifDeinterlacer::new();

        let frame = create_interlaced_frame(16, 16, 50);
        yadif.push_frame(frame).unwrap();

        assert!(!yadif.frame_buffer.is_empty());

        yadif.reset();
        assert!(yadif.frame_buffer.is_empty());
        assert!(!yadif.initialized);
    }

    #[test]
    fn test_progressive_passthrough() {
        let mut yadif = YadifDeinterlacer::with_config(YadifConfig {
            deint_all: false,
            ..Default::default()
        });

        let frame1 = Frame::new(16, 16, PixelFormat::Yuv420p, TimeBase::MPEG);
        let frame2 = Frame::new(16, 16, PixelFormat::Yuv420p, TimeBase::MPEG);
        // Not setting INTERLACED flag - these are progressive

        yadif.push_frame(frame1).unwrap();
        let result = yadif.push_frame(frame2).unwrap();

        // Progressive frames should pass through unchanged
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_invalid_dimensions() {
        let mut yadif = YadifDeinterlacer::new();
        let frame = Frame::new(2, 2, PixelFormat::Yuv420p, TimeBase::MPEG);

        let result = yadif.push_frame(frame);
        assert!(matches!(
            result,
            Err(DeinterlaceError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_frame_mismatch() {
        let mut yadif = YadifDeinterlacer::new();

        let frame1 = create_interlaced_frame(16, 16, 50);
        let frame2 = create_interlaced_frame(32, 32, 100);

        yadif.push_frame(frame1).unwrap();
        let result = yadif.push_frame(frame2);

        assert!(matches!(result, Err(DeinterlaceError::FrameMismatch { .. })));
    }
}
