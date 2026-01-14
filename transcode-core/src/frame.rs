//! Video frame buffer abstractions.
//!
//! Provides types for representing decoded video frames in various pixel formats.

use crate::timestamp::{Duration, TimeBase, Timestamp};
use bitflags::bitflags;
use std::fmt;
use std::sync::Arc;

/// Pixel format for video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelFormat {
    /// Planar YUV 4:2:0, 12bpp (1 Cr & Cb sample per 2x2 Y samples).
    Yuv420p,
    /// Planar YUV 4:2:2, 16bpp (1 Cr & Cb sample per 2x1 Y samples).
    Yuv422p,
    /// Planar YUV 4:4:4, 24bpp (no subsampling).
    Yuv444p,
    /// Planar YUV 4:2:0, 15bpp, 10-bit.
    Yuv420p10le,
    /// Planar YUV 4:2:2, 20bpp, 10-bit.
    Yuv422p10le,
    /// Planar YUV 4:4:4, 30bpp, 10-bit.
    Yuv444p10le,
    /// Packed NV12 (Y plane, interleaved UV plane).
    Nv12,
    /// Packed NV21 (Y plane, interleaved VU plane).
    Nv21,
    /// Packed RGB24, 24bpp.
    Rgb24,
    /// Packed BGR24, 24bpp.
    Bgr24,
    /// Packed RGBA, 32bpp.
    Rgba,
    /// Packed BGRA, 32bpp.
    Bgra,
    /// Grayscale, 8bpp.
    Gray8,
    /// Grayscale, 16bpp.
    Gray16,
}

impl PixelFormat {
    /// Get the number of planes for this pixel format.
    pub fn num_planes(&self) -> usize {
        match self {
            Self::Yuv420p
            | Self::Yuv422p
            | Self::Yuv444p
            | Self::Yuv420p10le
            | Self::Yuv422p10le
            | Self::Yuv444p10le => 3,
            Self::Nv12 | Self::Nv21 => 2,
            Self::Rgb24 | Self::Bgr24 | Self::Rgba | Self::Bgra | Self::Gray8 | Self::Gray16 => 1,
        }
    }

    /// Get the bits per pixel.
    pub fn bits_per_pixel(&self) -> u32 {
        match self {
            Self::Yuv420p | Self::Nv12 | Self::Nv21 => 12,
            Self::Yuv420p10le => 15,
            Self::Yuv422p => 16,
            Self::Yuv422p10le => 20,
            Self::Yuv444p | Self::Rgb24 | Self::Bgr24 => 24,
            Self::Yuv444p10le => 30,
            Self::Rgba | Self::Bgra => 32,
            Self::Gray8 => 8,
            Self::Gray16 => 16,
        }
    }

    /// Check if this is a planar format.
    pub fn is_planar(&self) -> bool {
        matches!(
            self,
            Self::Yuv420p
                | Self::Yuv422p
                | Self::Yuv444p
                | Self::Yuv420p10le
                | Self::Yuv422p10le
                | Self::Yuv444p10le
        )
    }

    /// Check if this is a 10-bit format.
    pub fn is_10bit(&self) -> bool {
        matches!(
            self,
            Self::Yuv420p10le | Self::Yuv422p10le | Self::Yuv444p10le | Self::Gray16
        )
    }

    /// Get chroma subsampling factors (horizontal, vertical).
    pub fn chroma_subsampling(&self) -> (u32, u32) {
        match self {
            Self::Yuv420p | Self::Yuv420p10le | Self::Nv12 | Self::Nv21 => (2, 2),
            Self::Yuv422p | Self::Yuv422p10le => (2, 1),
            Self::Yuv444p | Self::Yuv444p10le => (1, 1),
            _ => (1, 1),
        }
    }

    /// Calculate the size of a plane for given dimensions.
    pub fn plane_size(&self, plane: usize, width: u32, height: u32) -> usize {
        let (hsub, vsub) = self.chroma_subsampling();
        let bytes_per_sample = if self.is_10bit() { 2 } else { 1 };

        match self {
            Self::Yuv420p
            | Self::Yuv422p
            | Self::Yuv444p
            | Self::Yuv420p10le
            | Self::Yuv422p10le
            | Self::Yuv444p10le => {
                if plane == 0 {
                    (width as usize) * (height as usize) * bytes_per_sample
                } else {
                    (width as usize / hsub as usize)
                        * (height as usize / vsub as usize)
                        * bytes_per_sample
                }
            }
            Self::Nv12 | Self::Nv21 => {
                if plane == 0 {
                    (width as usize) * (height as usize)
                } else {
                    (width as usize) * (height as usize / 2)
                }
            }
            Self::Rgb24 | Self::Bgr24 => (width as usize) * (height as usize) * 3,
            Self::Rgba | Self::Bgra => (width as usize) * (height as usize) * 4,
            Self::Gray8 => (width as usize) * (height as usize),
            Self::Gray16 => (width as usize) * (height as usize) * 2,
        }
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Yuv420p => write!(f, "yuv420p"),
            Self::Yuv422p => write!(f, "yuv422p"),
            Self::Yuv444p => write!(f, "yuv444p"),
            Self::Yuv420p10le => write!(f, "yuv420p10le"),
            Self::Yuv422p10le => write!(f, "yuv422p10le"),
            Self::Yuv444p10le => write!(f, "yuv444p10le"),
            Self::Nv12 => write!(f, "nv12"),
            Self::Nv21 => write!(f, "nv21"),
            Self::Rgb24 => write!(f, "rgb24"),
            Self::Bgr24 => write!(f, "bgr24"),
            Self::Rgba => write!(f, "rgba"),
            Self::Bgra => write!(f, "bgra"),
            Self::Gray8 => write!(f, "gray8"),
            Self::Gray16 => write!(f, "gray16"),
        }
    }
}

/// Color space for video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorSpace {
    /// BT.601 (SD video).
    #[default]
    Bt601,
    /// BT.709 (HD video).
    Bt709,
    /// BT.2020 (UHD/HDR video).
    Bt2020,
    /// sRGB.
    Srgb,
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bt601 => write!(f, "BT.601"),
            Self::Bt709 => write!(f, "BT.709"),
            Self::Bt2020 => write!(f, "BT.2020"),
            Self::Srgb => write!(f, "sRGB"),
        }
    }
}

/// Color range (limited/full).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorRange {
    /// Limited/TV range (16-235 for Y, 16-240 for UV).
    #[default]
    Limited,
    /// Full/PC range (0-255).
    Full,
}

bitflags! {
    /// Frame flags indicating frame properties.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct FrameFlags: u32 {
        /// This is a keyframe (I-frame).
        const KEYFRAME = 0x0001;
        /// Frame is corrupted or incomplete.
        const CORRUPT = 0x0002;
        /// Frame should be discarded after decoding (used for reference only).
        const DISCARD = 0x0004;
        /// Interlaced frame.
        const INTERLACED = 0x0008;
        /// Top field first (for interlaced content).
        const TOP_FIELD_FIRST = 0x0010;
    }
}

impl Default for FrameFlags {
    fn default() -> Self {
        Self::empty()
    }
}

/// A decoded video frame.
#[derive(Clone)]
pub struct Frame {
    /// Frame data buffer.
    buffer: FrameBuffer,
    /// Presentation timestamp.
    pub pts: Timestamp,
    /// Decode timestamp.
    pub dts: Timestamp,
    /// Frame duration.
    pub duration: Duration,
    /// Frame flags.
    pub flags: FrameFlags,
    /// Picture order count (for B-frame reordering).
    pub poc: i32,
}

impl Frame {
    /// Create a new frame with the specified parameters.
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        time_base: TimeBase,
    ) -> Self {
        Self {
            buffer: FrameBuffer::new(width, height, format),
            pts: Timestamp::new(Timestamp::NONE, time_base),
            dts: Timestamp::new(Timestamp::NONE, time_base),
            duration: Duration::new(0, time_base),
            flags: FrameFlags::empty(),
            poc: 0,
        }
    }

    /// Create a frame from an existing buffer.
    pub fn from_buffer(buffer: FrameBuffer) -> Self {
        Self {
            buffer,
            pts: Timestamp::none(),
            dts: Timestamp::none(),
            duration: Duration::zero(),
            flags: FrameFlags::empty(),
            poc: 0,
        }
    }

    /// Get the frame width.
    pub fn width(&self) -> u32 {
        self.buffer.width
    }

    /// Get the frame height.
    pub fn height(&self) -> u32 {
        self.buffer.height
    }

    /// Get the pixel format.
    pub fn format(&self) -> PixelFormat {
        self.buffer.format
    }

    /// Get the color space.
    pub fn color_space(&self) -> ColorSpace {
        self.buffer.color_space
    }

    /// Get the color range.
    pub fn color_range(&self) -> ColorRange {
        self.buffer.color_range
    }

    /// Check if this is a keyframe.
    pub fn is_keyframe(&self) -> bool {
        self.flags.contains(FrameFlags::KEYFRAME)
    }

    /// Get the frame buffer.
    pub fn buffer(&self) -> &FrameBuffer {
        &self.buffer
    }

    /// Get a mutable reference to the frame buffer.
    pub fn buffer_mut(&mut self) -> &mut FrameBuffer {
        &mut self.buffer
    }

    /// Get a plane's data.
    pub fn plane(&self, index: usize) -> Option<&[u8]> {
        self.buffer.plane(index)
    }

    /// Get a mutable reference to a plane's data.
    pub fn plane_mut(&mut self, index: usize) -> Option<&mut [u8]> {
        self.buffer.plane_mut(index)
    }

    /// Get the stride (bytes per row) for a plane.
    pub fn stride(&self, plane: usize) -> usize {
        self.buffer.stride(plane)
    }
}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Frame")
            .field("width", &self.width())
            .field("height", &self.height())
            .field("format", &self.format())
            .field("pts", &self.pts)
            .field("flags", &self.flags)
            .finish()
    }
}

/// A buffer for storing frame pixel data.
#[derive(Clone)]
pub struct FrameBuffer {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Color space.
    pub color_space: ColorSpace,
    /// Color range.
    pub color_range: ColorRange,
    /// Plane data.
    planes: Vec<PlaneData>,
}

#[derive(Clone)]
struct PlaneData {
    data: Vec<u8>,
    stride: usize,
}

impl FrameBuffer {
    /// Create a new frame buffer.
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        let num_planes = format.num_planes();
        let (hsub, vsub) = format.chroma_subsampling();
        let bytes_per_sample = if format.is_10bit() { 2 } else { 1 };

        let mut planes = Vec::with_capacity(num_planes);

        for plane in 0..num_planes {
            let (plane_width, plane_height) = if plane == 0 {
                (width as usize, height as usize)
            } else {
                (
                    width as usize / hsub as usize,
                    height as usize / vsub as usize,
                )
            };

            let stride = if format == PixelFormat::Nv12 || format == PixelFormat::Nv21 {
                // Both Y and interleaved UV planes use width as stride
                width as usize
            } else if format == PixelFormat::Rgb24 || format == PixelFormat::Bgr24 {
                width as usize * 3
            } else if format == PixelFormat::Rgba || format == PixelFormat::Bgra {
                width as usize * 4
            } else {
                plane_width * bytes_per_sample
            };

            // Align stride to 32 bytes for SIMD optimization
            let aligned_stride = (stride + 31) & !31;
            let size = aligned_stride * plane_height;

            planes.push(PlaneData {
                data: vec![0u8; size],
                stride: aligned_stride,
            });
        }

        Self {
            width,
            height,
            format,
            color_space: ColorSpace::default(),
            color_range: ColorRange::default(),
            planes,
        }
    }

    /// Get the number of planes.
    pub fn num_planes(&self) -> usize {
        self.planes.len()
    }

    /// Get a plane's data.
    pub fn plane(&self, index: usize) -> Option<&[u8]> {
        self.planes.get(index).map(|p| p.data.as_slice())
    }

    /// Get a mutable reference to a plane's data.
    pub fn plane_mut(&mut self, index: usize) -> Option<&mut [u8]> {
        self.planes.get_mut(index).map(|p| p.data.as_mut_slice())
    }

    /// Get the stride for a plane.
    pub fn stride(&self, plane: usize) -> usize {
        self.planes.get(plane).map(|p| p.stride).unwrap_or(0)
    }

    /// Get the total size of all planes in bytes.
    pub fn total_size(&self) -> usize {
        self.planes.iter().map(|p| p.data.len()).sum()
    }

    /// Fill all planes with a value.
    pub fn fill(&mut self, value: u8) {
        for plane in &mut self.planes {
            plane.data.fill(value);
        }
    }

    /// Copy data from another frame buffer.
    pub fn copy_from(&mut self, other: &FrameBuffer) {
        assert_eq!(self.width, other.width);
        assert_eq!(self.height, other.height);
        assert_eq!(self.format, other.format);

        for (dst, src) in self.planes.iter_mut().zip(other.planes.iter()) {
            dst.data.copy_from_slice(&src.data);
        }
    }
}

impl fmt::Debug for FrameBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameBuffer")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("planes", &self.planes.len())
            .finish()
    }
}

/// A reference-counted frame for efficient sharing.
pub type SharedFrame = Arc<Frame>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_planes() {
        assert_eq!(PixelFormat::Yuv420p.num_planes(), 3);
        assert_eq!(PixelFormat::Nv12.num_planes(), 2);
        assert_eq!(PixelFormat::Rgb24.num_planes(), 1);
    }

    #[test]
    fn test_frame_buffer_creation() {
        let buffer = FrameBuffer::new(1920, 1080, PixelFormat::Yuv420p);
        assert_eq!(buffer.num_planes(), 3);
        assert!(buffer.plane(0).is_some());
        assert!(buffer.plane(1).is_some());
        assert!(buffer.plane(2).is_some());
        assert!(buffer.plane(3).is_none());
    }

    #[test]
    fn test_frame_creation() {
        let frame = Frame::new(1920, 1080, PixelFormat::Yuv420p, TimeBase::MPEG);
        assert_eq!(frame.width(), 1920);
        assert_eq!(frame.height(), 1080);
        assert_eq!(frame.format(), PixelFormat::Yuv420p);
    }

    #[test]
    fn test_stride_alignment() {
        let buffer = FrameBuffer::new(100, 100, PixelFormat::Yuv420p);
        // Stride should be aligned to 32 bytes
        assert_eq!(buffer.stride(0) % 32, 0);
    }
}
