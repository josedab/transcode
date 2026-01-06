//! VP8 frame types and structures.

/// VP8 frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp8FrameType {
    /// Key frame (intra-coded).
    KeyFrame = 0,
    /// Inter frame (predicted).
    InterFrame = 1,
}

/// VP8 color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp8ColorSpace {
    /// BT.601 (standard definition).
    Bt601 = 0,
    /// Reserved.
    Reserved = 1,
}

/// VP8 decoded frame.
#[derive(Debug, Clone)]
pub struct Vp8Frame {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frame type.
    pub frame_type: Vp8FrameType,
    /// Y plane (luma).
    pub y_plane: Vec<u8>,
    /// U plane (chroma Cb).
    pub u_plane: Vec<u8>,
    /// V plane (chroma Cr).
    pub v_plane: Vec<u8>,
    /// Y plane stride.
    pub y_stride: usize,
    /// UV plane stride.
    pub uv_stride: usize,
    /// Presentation timestamp.
    pub pts: i64,
    /// Is visible (show_frame).
    pub visible: bool,
}

impl Vp8Frame {
    /// Create a new frame.
    pub fn new(width: u32, height: u32, frame_type: Vp8FrameType) -> Self {
        let y_stride = ((width + 15) & !15) as usize;
        let uv_stride = ((width / 2 + 15) & !15) as usize;
        let y_height = ((height + 15) & !15) as usize;
        let uv_height = ((height / 2 + 15) & !15) as usize;

        Self {
            width,
            height,
            frame_type,
            y_plane: vec![128u8; y_stride * y_height],
            u_plane: vec![128u8; uv_stride * uv_height],
            v_plane: vec![128u8; uv_stride * uv_height],
            y_stride,
            uv_stride,
            pts: 0,
            visible: true,
        }
    }

    /// Get Y plane data.
    pub fn y_data(&self) -> &[u8] {
        &self.y_plane
    }

    /// Get U plane data.
    pub fn u_data(&self) -> &[u8] {
        &self.u_plane
    }

    /// Get V plane data.
    pub fn v_data(&self) -> &[u8] {
        &self.v_plane
    }

    /// Get mutable Y plane data.
    pub fn y_data_mut(&mut self) -> &mut [u8] {
        &mut self.y_plane
    }

    /// Get mutable U plane data.
    pub fn u_data_mut(&mut self) -> &mut [u8] {
        &mut self.u_plane
    }

    /// Get mutable V plane data.
    pub fn v_data_mut(&mut self) -> &mut [u8] {
        &mut self.v_plane
    }

    /// Get pixel at (x, y) in Y plane.
    pub fn get_y(&self, x: usize, y: usize) -> u8 {
        self.y_plane[y * self.y_stride + x]
    }

    /// Set pixel at (x, y) in Y plane.
    pub fn set_y(&mut self, x: usize, y: usize, value: u8) {
        self.y_plane[y * self.y_stride + x] = value;
    }

    /// Get pixel at (x, y) in U plane.
    pub fn get_u(&self, x: usize, y: usize) -> u8 {
        self.u_plane[y * self.uv_stride + x]
    }

    /// Set pixel at (x, y) in U plane.
    pub fn set_u(&mut self, x: usize, y: usize, value: u8) {
        self.u_plane[y * self.uv_stride + x] = value;
    }

    /// Get pixel at (x, y) in V plane.
    pub fn get_v(&self, x: usize, y: usize) -> u8 {
        self.v_plane[y * self.uv_stride + x]
    }

    /// Set pixel at (x, y) in V plane.
    pub fn set_v(&mut self, x: usize, y: usize, value: u8) {
        self.v_plane[y * self.uv_stride + x] = value;
    }

    /// Copy from another frame.
    pub fn copy_from(&mut self, other: &Vp8Frame) {
        self.y_plane.copy_from_slice(&other.y_plane);
        self.u_plane.copy_from_slice(&other.u_plane);
        self.v_plane.copy_from_slice(&other.v_plane);
    }

    /// Get total size in bytes.
    pub fn size(&self) -> usize {
        self.y_plane.len() + self.u_plane.len() + self.v_plane.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_creation() {
        let frame = Vp8Frame::new(320, 240, Vp8FrameType::KeyFrame);
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert!(frame.y_stride >= 320);
        assert!(frame.uv_stride >= 160);
    }

    #[test]
    fn test_pixel_access() {
        let mut frame = Vp8Frame::new(16, 16, Vp8FrameType::KeyFrame);
        frame.set_y(5, 5, 200);
        assert_eq!(frame.get_y(5, 5), 200);
    }
}
