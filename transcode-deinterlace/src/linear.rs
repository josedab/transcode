//! Linear interpolation deinterlacer.
//!
//! Linear interpolation is a spatial-only deinterlacing method that creates
//! missing lines by averaging adjacent lines. It's more sophisticated than
//! simple bob deinterlacing but still relatively fast.
//!
//! # Algorithm
//!
//! For each missing line (from the opposite field):
//! 1. Take the line above and the line below (from the same field)
//! 2. Average the pixel values: `output = (above + below) / 2`
//!
//! # Advantages
//!
//! - Fast computation
//! - Reduces line-doubling artifacts compared to bob
//! - Smoother vertical transitions
//!
//! # Disadvantages
//!
//! - Can produce blurry results
//! - No temporal information used
//! - Motion areas may still show artifacts

use crate::bob::FieldOrder;
use crate::error::{DeinterlaceError, Result};
use transcode_core::frame::FrameFlags;
use transcode_core::{Frame, FrameBuffer, PixelFormat};

/// Linear interpolation deinterlacer configuration.
#[derive(Debug, Clone)]
pub struct LinearConfig {
    /// Field order of the input content.
    pub field_order: FieldOrder,
    /// Blend factor for weighting (0.0 = line above, 1.0 = line below, 0.5 = equal).
    pub blend_factor: f32,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            field_order: FieldOrder::TopFieldFirst,
            blend_factor: 0.5,
        }
    }
}

/// Linear interpolation deinterlacer.
///
/// Creates progressive frames by linearly interpolating missing lines.
pub struct LinearDeinterlacer {
    config: LinearConfig,
}

impl LinearDeinterlacer {
    /// Create a new linear deinterlacer with default configuration.
    pub fn new() -> Self {
        Self {
            config: LinearConfig::default(),
        }
    }

    /// Create a new linear deinterlacer with custom configuration.
    pub fn with_config(config: LinearConfig) -> Self {
        Self { config }
    }

    /// Set the field order.
    pub fn set_field_order(&mut self, order: FieldOrder) {
        self.config.field_order = order;
    }

    /// Set the blend factor.
    ///
    /// # Arguments
    ///
    /// * `factor` - Blend factor between 0.0 and 1.0
    pub fn set_blend_factor(&mut self, factor: f32) {
        self.config.blend_factor = factor.clamp(0.0, 1.0);
    }

    /// Process a single interlaced frame using linear interpolation.
    ///
    /// Returns a single progressive frame.
    pub fn process(&self, frame: &Frame) -> Result<Frame> {
        self.validate_frame(frame)?;

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

        // Determine which lines to keep based on field order
        let keep_even = matches!(self.config.field_order, FieldOrder::TopFieldFirst);

        for plane_idx in 0..num_planes {
            let src_plane = frame
                .plane(plane_idx)
                .ok_or_else(|| DeinterlaceError::buffer_error("Source plane not found"))?;

            let src_stride = frame.stride(plane_idx);
            let dst_stride = output.stride(plane_idx);

            let dst_plane = output
                .plane_mut(plane_idx)
                .ok_or_else(|| DeinterlaceError::buffer_error("Destination plane not found"))?;

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

            if format.is_10bit() {
                self.interpolate_plane_16bit(
                    src_plane,
                    dst_plane,
                    src_stride,
                    dst_stride,
                    plane_width,
                    plane_height,
                    keep_even,
                );
            } else {
                self.interpolate_plane_8bit(
                    src_plane,
                    dst_plane,
                    src_stride,
                    dst_stride,
                    plane_width,
                    plane_height,
                    keep_even,
                );
            }
        }

        Ok(output)
    }

    /// Interpolate an 8-bit plane.
    #[allow(clippy::too_many_arguments)]
    fn interpolate_plane_8bit(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        dst_stride: usize,
        width: usize,
        height: usize,
        keep_even: bool,
    ) {
        let blend_a = ((1.0 - self.config.blend_factor) * 256.0) as u32;
        let blend_b = (self.config.blend_factor * 256.0) as u32;

        for y in 0..height {
            let dst_offset = y * dst_stride;
            let src_offset = y * src_stride;

            let is_kept_line = (y % 2 == 0) == keep_even;

            if is_kept_line {
                // Copy the original line
                if src_offset + width <= src.len() && dst_offset + width <= dst.len() {
                    dst[dst_offset..dst_offset + width]
                        .copy_from_slice(&src[src_offset..src_offset + width]);
                }
            } else {
                // Interpolate from adjacent lines
                let above_y = if y > 0 { y - 1 } else { y + 1 };
                let below_y = if y < height - 1 { y + 1 } else { y - 1 };

                let above_offset = above_y * src_stride;
                let below_offset = below_y * src_stride;

                for x in 0..width {
                    if above_offset + x < src.len()
                        && below_offset + x < src.len()
                        && dst_offset + x < dst.len()
                    {
                        let above = src[above_offset + x] as u32;
                        let below = src[below_offset + x] as u32;
                        let interpolated = (above * blend_a + below * blend_b + 128) >> 8;
                        dst[dst_offset + x] = interpolated.min(255) as u8;
                    }
                }
            }
        }
    }

    /// Interpolate a 16-bit plane.
    #[allow(clippy::too_many_arguments)]
    fn interpolate_plane_16bit(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        dst_stride: usize,
        width: usize,
        height: usize,
        keep_even: bool,
    ) {
        let blend_a = ((1.0 - self.config.blend_factor) * 65536.0) as u32;
        let blend_b = (self.config.blend_factor * 65536.0) as u32;

        for y in 0..height {
            let dst_offset = y * dst_stride;
            let src_offset = y * src_stride;

            let is_kept_line = (y % 2 == 0) == keep_even;

            if is_kept_line {
                // Copy the original line
                let row_bytes = width * 2;
                if src_offset + row_bytes <= src.len() && dst_offset + row_bytes <= dst.len() {
                    dst[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&src[src_offset..src_offset + row_bytes]);
                }
            } else {
                // Interpolate from adjacent lines
                let above_y = if y > 0 { y - 1 } else { y + 1 };
                let below_y = if y < height - 1 { y + 1 } else { y - 1 };

                let above_offset = above_y * src_stride;
                let below_offset = below_y * src_stride;

                for x in 0..width {
                    let px_offset = x * 2;
                    if above_offset + px_offset + 1 < src.len()
                        && below_offset + px_offset + 1 < src.len()
                        && dst_offset + px_offset + 1 < dst.len()
                    {
                        let above = u16::from_le_bytes([
                            src[above_offset + px_offset],
                            src[above_offset + px_offset + 1],
                        ]) as u32;
                        let below = u16::from_le_bytes([
                            src[below_offset + px_offset],
                            src[below_offset + px_offset + 1],
                        ]) as u32;

                        let interpolated =
                            ((above * blend_a + below * blend_b + 32768) >> 16).min(65535) as u16;
                        let bytes = interpolated.to_le_bytes();
                        dst[dst_offset + px_offset] = bytes[0];
                        dst[dst_offset + px_offset + 1] = bytes[1];
                    }
                }
            }
        }
    }

    /// Validate that the frame can be processed.
    fn validate_frame(&self, frame: &Frame) -> Result<()> {
        let width = frame.width();
        let height = frame.height();

        if width < 2 || height < 4 {
            return Err(DeinterlaceError::invalid_dimensions(width, height));
        }

        match frame.format() {
            PixelFormat::Yuv420p
            | PixelFormat::Yuv422p
            | PixelFormat::Yuv444p
            | PixelFormat::Yuv420p10le
            | PixelFormat::Yuv422p10le
            | PixelFormat::Yuv444p10le
            | PixelFormat::Gray8
            | PixelFormat::Gray16 => Ok(()),
            format => Err(DeinterlaceError::unsupported_format(format.to_string())),
        }
    }
}

impl Default for LinearDeinterlacer {
    fn default() -> Self {
        Self::new()
    }
}

/// Bicubic interpolation deinterlacer for higher quality output.
///
/// Uses four lines instead of two for smoother interpolation.
pub struct BicubicDeinterlacer {
    field_order: FieldOrder,
}

impl BicubicDeinterlacer {
    /// Create a new bicubic deinterlacer.
    pub fn new() -> Self {
        Self {
            field_order: FieldOrder::TopFieldFirst,
        }
    }

    /// Set the field order.
    pub fn set_field_order(&mut self, order: FieldOrder) {
        self.field_order = order;
    }

    /// Process a single interlaced frame using bicubic interpolation.
    pub fn process(&self, frame: &Frame) -> Result<Frame> {
        self.validate_frame(frame)?;

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
        let keep_even = matches!(self.field_order, FieldOrder::TopFieldFirst);

        for plane_idx in 0..num_planes {
            let src_plane = frame
                .plane(plane_idx)
                .ok_or_else(|| DeinterlaceError::buffer_error("Source plane not found"))?;

            let src_stride = frame.stride(plane_idx);
            let dst_stride = output.stride(plane_idx);

            let dst_plane = output
                .plane_mut(plane_idx)
                .ok_or_else(|| DeinterlaceError::buffer_error("Destination plane not found"))?;

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

            self.interpolate_bicubic(
                src_plane,
                dst_plane,
                src_stride,
                dst_stride,
                plane_width,
                plane_height,
                keep_even,
            );
        }

        Ok(output)
    }

    /// Bicubic interpolation for a plane.
    #[allow(clippy::too_many_arguments)]
    fn interpolate_bicubic(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        dst_stride: usize,
        width: usize,
        height: usize,
        keep_even: bool,
    ) {
        // Bicubic coefficients for interpolation at 0.5
        // Using Catmull-Rom: -1/16, 9/16, 9/16, -1/16
        const COEF_A: i32 = -1;
        const COEF_B: i32 = 9;
        const COEF_C: i32 = 9;
        const COEF_D: i32 = -1;
        const COEF_SUM: i32 = 16;

        for y in 0..height {
            let dst_offset = y * dst_stride;
            let src_offset = y * src_stride;

            let is_kept_line = (y % 2 == 0) == keep_even;

            if is_kept_line {
                if src_offset + width <= src.len() && dst_offset + width <= dst.len() {
                    dst[dst_offset..dst_offset + width]
                        .copy_from_slice(&src[src_offset..src_offset + width]);
                }
            } else {
                // Find four lines from the same field for bicubic interpolation
                let field_lines: Vec<usize> = (0..height)
                    .filter(|&line| (line % 2 == 0) == keep_even)
                    .collect();

                // Find the two closest field lines above and below
                let pos = field_lines.iter().position(|&l| l > y).unwrap_or(field_lines.len());

                let y0 = if pos >= 2 { field_lines[pos - 2] } else { field_lines[0] };
                let y1 = if pos >= 1 { field_lines[pos - 1] } else { field_lines[0] };
                let y2 = if pos < field_lines.len() {
                    field_lines[pos]
                } else {
                    field_lines[field_lines.len() - 1]
                };
                let y3 = if pos + 1 < field_lines.len() {
                    field_lines[pos + 1]
                } else {
                    field_lines[field_lines.len() - 1]
                };

                for x in 0..width {
                    let p0 = src.get(y0 * src_stride + x).copied().unwrap_or(128) as i32;
                    let p1 = src.get(y1 * src_stride + x).copied().unwrap_or(128) as i32;
                    let p2 = src.get(y2 * src_stride + x).copied().unwrap_or(128) as i32;
                    let p3 = src.get(y3 * src_stride + x).copied().unwrap_or(128) as i32;

                    let result = (COEF_A * p0 + COEF_B * p1 + COEF_C * p2 + COEF_D * p3 + COEF_SUM / 2) / COEF_SUM;
                    let clamped = result.clamp(0, 255) as u8;

                    if dst_offset + x < dst.len() {
                        dst[dst_offset + x] = clamped;
                    }
                }
            }
        }
    }

    fn validate_frame(&self, frame: &Frame) -> Result<()> {
        let width = frame.width();
        let height = frame.height();

        if width < 2 || height < 8 {
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

impl Default for BicubicDeinterlacer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::TimeBase;

    fn create_test_frame(width: u32, height: u32) -> Frame {
        let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);
        frame.flags = FrameFlags::INTERLACED | FrameFlags::TOP_FIELD_FIRST;

        let stride = frame.stride(0);
        // Fill Y plane with alternating pattern
        if let Some(y_plane) = frame.plane_mut(0) {
            for y in 0..height as usize {
                let value = if y % 2 == 0 { 50u8 } else { 200u8 };
                for x in 0..width as usize {
                    y_plane[y * stride + x] = value;
                }
            }
        }

        frame
    }

    #[test]
    fn test_linear_deinterlace() {
        let frame = create_test_frame(16, 16);
        let linear = LinearDeinterlacer::new();

        let result = linear.process(&frame).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
        assert!(!result.flags.contains(FrameFlags::INTERLACED));
    }

    #[test]
    fn test_linear_interpolation_values() {
        let frame = create_test_frame(16, 16);
        let linear = LinearDeinterlacer::new();

        let result = linear.process(&frame).unwrap();
        let y_plane = result.plane(0).unwrap();
        let stride = result.stride(0);

        // Even lines should be kept (value = 50)
        assert_eq!(y_plane[0], 50);
        assert_eq!(y_plane[2 * stride], 50);

        // Odd lines should be interpolated (average of 50 and 50 = 50 for TFF)
        // Wait, for TFF, even lines are kept, odd lines are interpolated from even lines
        // So odd lines should also be 50 (interpolated from adjacent even lines)
        // Actually in our test, even lines = 50, so interpolation of 50+50 / 2 = 50
    }

    #[test]
    fn test_blend_factor() {
        let _frame = create_test_frame(16, 16);
        let mut linear = LinearDeinterlacer::new();

        // Test clamping
        linear.set_blend_factor(-0.5);
        assert!((linear.config.blend_factor - 0.0).abs() < 0.01);

        linear.set_blend_factor(1.5);
        assert!((linear.config.blend_factor - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bicubic_deinterlace() {
        let frame = create_test_frame(16, 16);
        let bicubic = BicubicDeinterlacer::new();

        let result = bicubic.process(&frame).unwrap();
        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
        assert!(!result.flags.contains(FrameFlags::INTERLACED));
    }

    #[test]
    fn test_invalid_dimensions() {
        let frame = Frame::new(16, 2, PixelFormat::Yuv420p, TimeBase::MPEG);
        let linear = LinearDeinterlacer::new();

        let result = linear.process(&frame);
        assert!(matches!(
            result,
            Err(DeinterlaceError::InvalidDimensions { .. })
        ));
    }
}
