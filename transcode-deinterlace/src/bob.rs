//! Bob deinterlacer implementation.
//!
//! Bob deinterlacing is the simplest form of deinterlacing that works by
//! doubling each field vertically to create a full frame. Each field
//! becomes a separate output frame.
//!
//! # Algorithm
//!
//! 1. Extract either the top or bottom field from the interlaced frame
//! 2. Duplicate each line to fill the missing lines
//! 3. Output a full-height frame
//!
//! # Advantages
//!
//! - Very fast and simple
//! - Preserves all temporal information
//! - No motion artifacts
//!
//! # Disadvantages
//!
//! - Halves vertical resolution
//! - Can produce visible line-doubling artifacts
//! - Doubles the frame rate (may need to be addressed separately)

use crate::error::{DeinterlaceError, Result};
use transcode_core::frame::FrameFlags;
use transcode_core::{Frame, FrameBuffer, PixelFormat};

/// Field order for interlaced content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldOrder {
    /// Top field first (TFF) - even lines come first temporally.
    #[default]
    TopFieldFirst,
    /// Bottom field first (BFF) - odd lines come first temporally.
    BottomFieldFirst,
}

impl FieldOrder {
    /// Detect field order from frame flags.
    pub fn from_frame(frame: &Frame) -> Self {
        if frame.flags.contains(FrameFlags::TOP_FIELD_FIRST) {
            FieldOrder::TopFieldFirst
        } else {
            FieldOrder::BottomFieldFirst
        }
    }

    /// Get the opposite field order.
    pub fn opposite(&self) -> Self {
        match self {
            FieldOrder::TopFieldFirst => FieldOrder::BottomFieldFirst,
            FieldOrder::BottomFieldFirst => FieldOrder::TopFieldFirst,
        }
    }
}

/// Which field to extract and bob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FieldSelect {
    /// Extract and bob the top field (even lines: 0, 2, 4, ...).
    Top,
    /// Extract and bob the bottom field (odd lines: 1, 3, 5, ...).
    Bottom,
    /// Bob both fields, producing two output frames per input frame.
    #[default]
    Both,
}

/// Bob deinterlacer configuration.
#[derive(Debug, Clone)]
pub struct BobConfig {
    /// Which field(s) to process.
    pub field_select: FieldSelect,
    /// Field order of the input content.
    pub field_order: FieldOrder,
}

impl Default for BobConfig {
    fn default() -> Self {
        Self {
            field_select: FieldSelect::Both,
            field_order: FieldOrder::TopFieldFirst,
        }
    }
}

/// Bob deinterlacer.
///
/// Simple line-doubling deinterlacer that creates full frames from fields.
pub struct BobDeinterlacer {
    config: BobConfig,
}

impl BobDeinterlacer {
    /// Create a new bob deinterlacer with default configuration.
    pub fn new() -> Self {
        Self {
            config: BobConfig::default(),
        }
    }

    /// Create a new bob deinterlacer with custom configuration.
    pub fn with_config(config: BobConfig) -> Self {
        Self { config }
    }

    /// Set the field order.
    pub fn set_field_order(&mut self, order: FieldOrder) {
        self.config.field_order = order;
    }

    /// Set which field(s) to process.
    pub fn set_field_select(&mut self, select: FieldSelect) {
        self.config.field_select = select;
    }

    /// Process a single interlaced frame.
    ///
    /// Returns one or two deinterlaced frames depending on the field selection.
    pub fn process(&self, frame: &Frame) -> Result<Vec<Frame>> {
        self.validate_frame(frame)?;

        let mut result = Vec::new();

        match self.config.field_select {
            FieldSelect::Top => {
                result.push(self.bob_field(frame, true)?);
            }
            FieldSelect::Bottom => {
                result.push(self.bob_field(frame, false)?);
            }
            FieldSelect::Both => {
                // Output fields in temporal order based on field order
                match self.config.field_order {
                    FieldOrder::TopFieldFirst => {
                        result.push(self.bob_field(frame, true)?);
                        result.push(self.bob_field(frame, false)?);
                    }
                    FieldOrder::BottomFieldFirst => {
                        result.push(self.bob_field(frame, false)?);
                        result.push(self.bob_field(frame, true)?);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Bob a single field from the frame.
    ///
    /// # Arguments
    ///
    /// * `frame` - The interlaced source frame
    /// * `top_field` - If true, use the top field (even lines), otherwise bottom field
    fn bob_field(&self, frame: &Frame, top_field: bool) -> Result<Frame> {
        let width = frame.width();
        let height = frame.height();
        let format = frame.format();

        let mut output = Frame::from_buffer(FrameBuffer::new(width, height, format));
        output.pts = frame.pts;
        output.dts = frame.dts;
        output.duration = frame.duration;
        output.poc = frame.poc;
        // Clear the interlaced flag since the output is progressive
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
                .ok_or_else(|| DeinterlaceError::buffer_error("Destination plane not found"))?;

            // Calculate plane dimensions
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

            let bytes_per_pixel = if format.is_10bit() { 2 } else { 1 };
            let row_bytes = plane_width * bytes_per_pixel;

            self.bob_plane(
                src_plane,
                dst_plane,
                src_stride,
                dst_stride,
                plane_height,
                row_bytes,
                top_field,
            );
        }

        Ok(output)
    }

    /// Bob a single plane.
    #[allow(clippy::too_many_arguments)]
    fn bob_plane(
        &self,
        src: &[u8],
        dst: &mut [u8],
        src_stride: usize,
        dst_stride: usize,
        height: usize,
        row_bytes: usize,
        top_field: bool,
    ) {
        let field_offset = if top_field { 0 } else { 1 };

        for y in 0..height {
            // Determine source line (from the field)
            let src_y = if (y % 2) == field_offset {
                // This line is from our field - use it directly
                y
            } else {
                // This line is from the other field - duplicate from adjacent line
                if y == 0 {
                    1 - field_offset // Use first line of our field
                } else if y == height - 1 {
                    height - 2 + field_offset // Use last line of our field
                } else if top_field {
                    // For top field, interpolate from even lines
                    if y % 2 == 1 {
                        y - 1 // Use line above (which is from our field)
                    } else {
                        y
                    }
                } else {
                    // For bottom field, interpolate from odd lines
                    if y % 2 == 0 {
                        y + 1 // Use line below (which is from our field)
                    } else {
                        y
                    }
                }
            };

            let src_offset = src_y * src_stride;
            let dst_offset = y * dst_stride;

            if src_offset + row_bytes <= src.len() && dst_offset + row_bytes <= dst.len() {
                dst[dst_offset..dst_offset + row_bytes]
                    .copy_from_slice(&src[src_offset..src_offset + row_bytes]);
            }
        }
    }

    /// Validate that the frame can be processed.
    fn validate_frame(&self, frame: &Frame) -> Result<()> {
        let width = frame.width();
        let height = frame.height();

        if width < 2 || height < 2 {
            return Err(DeinterlaceError::invalid_dimensions(width, height));
        }

        // Check for supported formats
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

impl Default for BobDeinterlacer {
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
        // Fill Y plane with a pattern - even lines = 100, odd lines = 200
        if let Some(y_plane) = frame.plane_mut(0) {
            for y in 0..height as usize {
                let value = if y % 2 == 0 { 100u8 } else { 200u8 };
                for x in 0..width as usize {
                    y_plane[y * stride + x] = value;
                }
            }
        }

        frame
    }

    #[test]
    fn test_bob_top_field() {
        let frame = create_test_frame(16, 8);
        let bob = BobDeinterlacer::with_config(BobConfig {
            field_select: FieldSelect::Top,
            field_order: FieldOrder::TopFieldFirst,
        });

        let result = bob.process(&frame).unwrap();
        assert_eq!(result.len(), 1);

        let output = &result[0];
        assert_eq!(output.width(), 16);
        assert_eq!(output.height(), 8);
        assert!(!output.flags.contains(FrameFlags::INTERLACED));
    }

    #[test]
    fn test_bob_both_fields() {
        let frame = create_test_frame(16, 8);
        let bob = BobDeinterlacer::new();

        let result = bob.process(&frame).unwrap();
        assert_eq!(result.len(), 2);

        for output in &result {
            assert_eq!(output.width(), 16);
            assert_eq!(output.height(), 8);
            assert!(!output.flags.contains(FrameFlags::INTERLACED));
        }
    }

    #[test]
    fn test_field_order_from_frame() {
        let mut frame = Frame::new(16, 8, PixelFormat::Yuv420p, TimeBase::MPEG);

        frame.flags = FrameFlags::TOP_FIELD_FIRST;
        assert_eq!(FieldOrder::from_frame(&frame), FieldOrder::TopFieldFirst);

        frame.flags = FrameFlags::empty();
        assert_eq!(FieldOrder::from_frame(&frame), FieldOrder::BottomFieldFirst);
    }

    #[test]
    fn test_field_order_opposite() {
        assert_eq!(
            FieldOrder::TopFieldFirst.opposite(),
            FieldOrder::BottomFieldFirst
        );
        assert_eq!(
            FieldOrder::BottomFieldFirst.opposite(),
            FieldOrder::TopFieldFirst
        );
    }

    #[test]
    fn test_invalid_dimensions() {
        let frame = Frame::new(1, 1, PixelFormat::Yuv420p, TimeBase::MPEG);
        let bob = BobDeinterlacer::new();

        let result = bob.process(&frame);
        assert!(matches!(
            result,
            Err(DeinterlaceError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_unsupported_format() {
        let frame = Frame::new(16, 8, PixelFormat::Rgba, TimeBase::MPEG);
        let bob = BobDeinterlacer::new();

        let result = bob.process(&frame);
        assert!(matches!(
            result,
            Err(DeinterlaceError::UnsupportedFormat { .. })
        ));
    }
}
