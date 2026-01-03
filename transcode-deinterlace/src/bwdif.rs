//! BWDIF (Bob-Weave Deinterlacing Filter) implementation.
//!
//! BWDIF is an advanced motion-adaptive deinterlacing filter that improves
//! upon YADIF by using a more sophisticated interpolation kernel.
//!
//! # Algorithm Overview
//!
//! BWDIF uses a combination of:
//! 1. **Spatial interpolation**: 5-tap filter for edge-directed interpolation
//! 2. **Temporal interpolation**: Uses 5 frames (prev2, prev, cur, next, next2)
//! 3. **Motion detection**: Refined motion detection using multiple frames
//!
//! The algorithm adapts between spatial and temporal predictions based on
//! detected motion, producing fewer artifacts than simpler methods.
//!
//! # Improvements over YADIF
//!
//! - Uses 5 frames instead of 3 for better temporal prediction
//! - More sophisticated spatial interpolation kernel
//! - Better handling of diagonal edges
//! - Reduced "bob" artifacts in motion areas
//!
//! # References
//!
//! - FFmpeg libavfilter/vf_bwdif.c by Thomas Mundt

use crate::error::{DeinterlaceError, Result};
use std::collections::VecDeque;
use transcode_core::frame::FrameFlags;
use transcode_core::{Frame, FrameBuffer, PixelFormat};

/// BWDIF operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BwdifMode {
    /// Send one frame for each frame (frame rate preserved).
    #[default]
    Frame,
    /// Send one frame for each field (double frame rate).
    Field,
}

/// BWDIF parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BwdifParity {
    /// Top field first.
    #[default]
    Top,
    /// Bottom field first.
    Bottom,
    /// Automatic detection.
    Auto,
}

/// BWDIF configuration.
#[derive(Debug, Clone)]
pub struct BwdifConfig {
    /// Operating mode.
    pub mode: BwdifMode,
    /// Field parity.
    pub parity: BwdifParity,
    /// Deinterlace all frames or only marked ones.
    pub deint_all: bool,
}

impl Default for BwdifConfig {
    fn default() -> Self {
        Self {
            mode: BwdifMode::Frame,
            parity: BwdifParity::Auto,
            deint_all: true,
        }
    }
}

/// BWDIF deinterlacer.
///
/// Advanced motion-adaptive deinterlacer using bob-weave algorithm.
pub struct BwdifDeinterlacer {
    config: BwdifConfig,
    /// Frame buffer (prev2, prev, cur, next, next2).
    frame_buffer: VecDeque<Frame>,
    /// Expected dimensions.
    width: u32,
    height: u32,
    /// Expected format.
    format: PixelFormat,
    /// Initialized flag.
    initialized: bool,
}

impl BwdifDeinterlacer {
    /// Create a new BWDIF deinterlacer.
    pub fn new() -> Self {
        Self {
            config: BwdifConfig::default(),
            frame_buffer: VecDeque::with_capacity(5),
            width: 0,
            height: 0,
            format: PixelFormat::Yuv420p,
            initialized: false,
        }
    }

    /// Create a new BWDIF deinterlacer with custom configuration.
    pub fn with_config(config: BwdifConfig) -> Self {
        Self {
            config,
            frame_buffer: VecDeque::with_capacity(5),
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
    pub fn push_frame(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        self.validate_frame(&frame)?;

        if !self.initialized {
            self.width = frame.width();
            self.height = frame.height();
            self.format = frame.format();
            self.initialized = true;
        }

        if frame.width() != self.width || frame.height() != self.height {
            return Err(DeinterlaceError::frame_mismatch(
                self.width,
                self.height,
                frame.width(),
                frame.height(),
            ));
        }

        self.frame_buffer.push_back(frame);

        // Need at least 3 frames to start processing
        if self.frame_buffer.len() < 3 {
            return Ok(Vec::new());
        }

        // Keep at most 5 frames
        while self.frame_buffer.len() > 5 {
            self.frame_buffer.pop_front();
        }

        self.process_current()
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> Result<Vec<Frame>> {
        let mut output = Vec::new();

        while self.frame_buffer.len() >= 3 {
            output.extend(self.process_current()?);
            self.frame_buffer.pop_front();
        }

        // Process remaining frames with reduced context
        while !self.frame_buffer.is_empty() {
            if let Some(frame) = self.frame_buffer.pop_front() {
                let tff = frame.flags.contains(FrameFlags::TOP_FIELD_FIRST);
                output.push(self.deinterlace_single(&frame, tff)?);
            }
        }

        Ok(output)
    }

    /// Process the current frame in the buffer.
    fn process_current(&mut self) -> Result<Vec<Frame>> {
        let mut results = Vec::new();

        if self.frame_buffer.len() < 3 {
            return Ok(results);
        }

        // Get frame indices
        let (prev2_idx, prev_idx, cur_idx, next_idx, next2_idx) = match self.frame_buffer.len() {
            3 => (0, 0, 1, 2, 2),
            4 => (0, 0, 1, 2, 3),
            5 => (0, 1, 2, 3, 4),
            _ => return Ok(results),
        };

        let cur = &self.frame_buffer[cur_idx];

        let is_interlaced = cur.flags.contains(FrameFlags::INTERLACED) || self.config.deint_all;

        if !is_interlaced {
            results.push(cur.clone());
            return Ok(results);
        }

        let tff = match self.config.parity {
            BwdifParity::Top => true,
            BwdifParity::Bottom => false,
            BwdifParity::Auto => cur.flags.contains(FrameFlags::TOP_FIELD_FIRST),
        };

        match self.config.mode {
            BwdifMode::Frame => {
                let output = self.filter_frame(
                    &self.frame_buffer[prev2_idx],
                    &self.frame_buffer[prev_idx],
                    &self.frame_buffer[cur_idx],
                    &self.frame_buffer[next_idx],
                    &self.frame_buffer[next2_idx],
                    tff,
                    0,
                )?;
                results.push(output);
            }
            BwdifMode::Field => {
                let output1 = self.filter_frame(
                    &self.frame_buffer[prev2_idx],
                    &self.frame_buffer[prev_idx],
                    &self.frame_buffer[cur_idx],
                    &self.frame_buffer[next_idx],
                    &self.frame_buffer[next2_idx],
                    tff,
                    0,
                )?;
                let output2 = self.filter_frame(
                    &self.frame_buffer[prev2_idx],
                    &self.frame_buffer[prev_idx],
                    &self.frame_buffer[cur_idx],
                    &self.frame_buffer[next_idx],
                    &self.frame_buffer[next2_idx],
                    tff,
                    1,
                )?;
                results.push(output1);
                results.push(output2);
            }
        }

        Ok(results)
    }

    /// Filter a frame using BWDIF algorithm.
    #[allow(clippy::too_many_arguments)]
    fn filter_frame(
        &self,
        prev2: &Frame,
        prev: &Frame,
        cur: &Frame,
        next: &Frame,
        next2: &Frame,
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
                prev2,
                prev,
                cur,
                next,
                next2,
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
        prev2: &Frame,
        prev: &Frame,
        cur: &Frame,
        next: &Frame,
        next2: &Frame,
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

        let prev2_plane = prev2.plane(plane_idx);
        let prev_plane = prev.plane(plane_idx);
        let next_plane = next.plane(plane_idx);
        let next2_plane = next2.plane(plane_idx);

        let prev2_stride = prev2.stride(plane_idx);
        let prev_stride = prev.stride(plane_idx);
        let next_stride = next.stride(plane_idx);
        let next2_stride = next2.stride(plane_idx);

        for y in 0..height {
            let dst_offset = y * dst_stride;
            let is_field_line = (y % 2 == 0) == parity;

            if is_field_line {
                let src_offset = y * cur_stride;
                if src_offset + width <= cur_plane.len() && dst_offset + width <= dst_plane.len() {
                    dst_plane[dst_offset..dst_offset + width]
                        .copy_from_slice(&cur_plane[src_offset..src_offset + width]);
                }
            } else {
                for x in 0..width {
                    let pixel = self.filter_pixel(
                        cur_plane,
                        prev2_plane,
                        prev_plane,
                        next_plane,
                        next2_plane,
                        cur_stride,
                        prev2_stride,
                        prev_stride,
                        next_stride,
                        next2_stride,
                        x,
                        y,
                        width,
                        height,
                        parity,
                    );

                    if dst_offset + x < dst_plane.len() {
                        dst_plane[dst_offset + x] = pixel;
                    }
                }
            }
        }

        Ok(())
    }

    /// Filter a single pixel using BWDIF algorithm.
    #[allow(clippy::too_many_arguments)]
    fn filter_pixel(
        &self,
        cur: &[u8],
        prev2: Option<&[u8]>,
        prev: Option<&[u8]>,
        next: Option<&[u8]>,
        next2: Option<&[u8]>,
        cur_stride: usize,
        prev2_stride: usize,
        prev_stride: usize,
        next_stride: usize,
        next2_stride: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        _parity: bool,
    ) -> u8 {
        // Get spatial neighbors from current field
        let y_above = y.saturating_sub(2);
        let y_below = if y + 2 < height { y + 2 } else { height - 1 };
        let y_above2 = y.saturating_sub(4);
        let y_below2 = if y + 4 < height { y + 4 } else { height - 1 };

        let c = self.get_pixel(cur, cur_stride, x, y, width, height);
        let d = self.get_pixel(cur, cur_stride, x, y_above, width, height);
        let e = self.get_pixel(cur, cur_stride, x, y_below, width, height);
        let f = self.get_pixel(cur, cur_stride, x, y_above2, width, height);
        let g = self.get_pixel(cur, cur_stride, x, y_below2, width, height);

        // Spatial prediction using 5-tap filter: (-f + 9d + 9e - g + 8) / 16
        let spatial = (-(f as i32) + 9 * (d as i32) + 9 * (e as i32) - (g as i32) + 8) >> 4;
        let spatial = spatial.clamp(0, 255) as u8;

        // Get temporal neighbors
        let p = prev
            .map(|p| self.get_pixel(p, prev_stride, x, y, width, height))
            .unwrap_or(c);
        let n = next
            .map(|n| self.get_pixel(n, next_stride, x, y, width, height))
            .unwrap_or(c);
        let pp = prev2
            .map(|p| self.get_pixel(p, prev2_stride, x, y, width, height))
            .unwrap_or(p);
        let nn = next2
            .map(|n| self.get_pixel(n, next2_stride, x, y, width, height))
            .unwrap_or(n);

        // Temporal prediction using 5-tap filter
        let temporal =
            (-(pp as i32) + 9 * (p as i32) + 9 * (n as i32) - (nn as i32) + 8) >> 4;
        let temporal = temporal.clamp(0, 255) as u8;

        // Calculate motion score
        let motion = self.calculate_motion(
            cur, prev, next, prev2, next2, cur_stride, prev_stride, next_stride,
            prev2_stride, next2_stride, x, y, width, height,
        );

        // Blend based on motion
        if motion < 10 {
            // Low motion - use temporal
            temporal
        } else if motion > 40 {
            // High motion - use spatial
            spatial
        } else {
            // Blend
            let weight = ((motion - 10) * 256 / 30) as u32;
            let result = ((temporal as u32) * (256 - weight) + (spatial as u32) * weight + 128) >> 8;
            result.min(255) as u8
        }
    }

    /// Calculate motion score at a pixel position.
    #[allow(clippy::too_many_arguments)]
    fn calculate_motion(
        &self,
        cur: &[u8],
        prev: Option<&[u8]>,
        next: Option<&[u8]>,
        prev2: Option<&[u8]>,
        next2: Option<&[u8]>,
        cur_stride: usize,
        prev_stride: usize,
        next_stride: usize,
        prev2_stride: usize,
        next2_stride: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> i32 {
        let c = self.get_pixel(cur, cur_stride, x, y, width, height) as i32;

        let mut diff = 0;

        if let Some(p) = prev {
            let pv = self.get_pixel(p, prev_stride, x, y, width, height) as i32;
            diff += (c - pv).abs();
        }

        if let Some(n) = next {
            let nv = self.get_pixel(n, next_stride, x, y, width, height) as i32;
            diff += (c - nv).abs();
        }

        if let Some(pp) = prev2 {
            let ppv = self.get_pixel(pp, prev2_stride, x, y, width, height) as i32;
            diff += (c - ppv).abs() / 2;
        }

        if let Some(nn) = next2 {
            let nnv = self.get_pixel(nn, next2_stride, x, y, width, height) as i32;
            diff += (c - nnv).abs() / 2;
        }

        diff / 3
    }

    /// Get a pixel with bounds checking.
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

    /// Deinterlace a single frame without full temporal context.
    fn deinterlace_single(&self, frame: &Frame, tff: bool) -> Result<Frame> {
        // Use simple linear interpolation when we don't have enough context
        let width = frame.width();
        let height = frame.height();
        let format = frame.format();

        let mut output = Frame::from_buffer(FrameBuffer::new(width, height, format));
        output.pts = frame.pts;
        output.dts = frame.dts;
        output.duration = frame.duration;
        output.poc = frame.poc;
        output.flags = frame.flags & !FrameFlags::INTERLACED & !FrameFlags::TOP_FIELD_FIRST;

        let num_planes = format.num_planes();
        let (hsub, vsub) = format.chroma_subsampling();

        for plane_idx in 0..num_planes {
            let src_plane = frame
                .plane(plane_idx)
                .ok_or_else(|| DeinterlaceError::buffer_error("Source plane not found"))?;

            let src_stride = frame.stride(plane_idx);
            let dst_stride = output.stride(plane_idx);

            let dst_plane = output
                .plane_mut(plane_idx)
                .ok_or_else(|| DeinterlaceError::buffer_error("Dest plane not found"))?;

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

            for y in 0..plane_height {
                let dst_offset = y * dst_stride;
                let is_field_line = (y % 2 == 0) == tff;

                if is_field_line {
                    let src_offset = y * src_stride;
                    if src_offset + plane_width <= src_plane.len()
                        && dst_offset + plane_width <= dst_plane.len()
                    {
                        dst_plane[dst_offset..dst_offset + plane_width]
                            .copy_from_slice(&src_plane[src_offset..src_offset + plane_width]);
                    }
                } else {
                    let above = if y > 0 { y - 1 } else { 1 };
                    let below = if y + 1 < plane_height { y + 1 } else { plane_height - 2 };

                    for x in 0..plane_width {
                        let a = src_plane
                            .get(above * src_stride + x)
                            .copied()
                            .unwrap_or(128);
                        let b = src_plane
                            .get(below * src_stride + x)
                            .copied()
                            .unwrap_or(128);
                        let interp = ((a as u32 + b as u32 + 1) >> 1) as u8;

                        if dst_offset + x < dst_plane.len() {
                            dst_plane[dst_offset + x] = interp;
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Validate frame for processing.
    fn validate_frame(&self, frame: &Frame) -> Result<()> {
        let width = frame.width();
        let height = frame.height();

        if width < 8 || height < 8 {
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

impl Default for BwdifDeinterlacer {
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
    fn test_bwdif_basic() {
        let mut bwdif = BwdifDeinterlacer::new();

        let frame1 = create_interlaced_frame(16, 16, 50);
        let frame2 = create_interlaced_frame(16, 16, 100);
        let frame3 = create_interlaced_frame(16, 16, 150);
        let frame4 = create_interlaced_frame(16, 16, 200);

        let r1 = bwdif.push_frame(frame1).unwrap();
        assert!(r1.is_empty());

        let r2 = bwdif.push_frame(frame2).unwrap();
        assert!(r2.is_empty());

        let r3 = bwdif.push_frame(frame3).unwrap();
        assert_eq!(r3.len(), 1);

        let r4 = bwdif.push_frame(frame4).unwrap();
        assert_eq!(r4.len(), 1);

        let flush = bwdif.flush().unwrap();
        assert!(!flush.is_empty());
    }

    #[test]
    fn test_bwdif_field_mode() {
        let mut bwdif = BwdifDeinterlacer::with_config(BwdifConfig {
            mode: BwdifMode::Field,
            ..Default::default()
        });

        for i in 0..5 {
            let frame = create_interlaced_frame(16, 16, 50 + i * 30);
            bwdif.push_frame(frame).unwrap();
        }

        let results = bwdif.flush().unwrap();
        // Field mode produces 2 outputs per input
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_bwdif_reset() {
        let mut bwdif = BwdifDeinterlacer::new();

        let frame = create_interlaced_frame(16, 16, 50);
        bwdif.push_frame(frame).unwrap();

        bwdif.reset();
        assert!(bwdif.frame_buffer.is_empty());
        assert!(!bwdif.initialized);
    }

    #[test]
    fn test_invalid_dimensions() {
        let mut bwdif = BwdifDeinterlacer::new();
        let frame = Frame::new(4, 4, PixelFormat::Yuv420p, TimeBase::MPEG);

        let result = bwdif.push_frame(frame);
        assert!(matches!(
            result,
            Err(DeinterlaceError::InvalidDimensions { .. })
        ));
    }
}
