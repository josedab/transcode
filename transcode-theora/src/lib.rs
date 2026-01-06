//! Theora video codec implementation.
//!
//! Theora is an open, royalty-free video codec developed by the Xiph.Org Foundation.
//! It is based on the VP3 codec and is commonly used with the OGG container format.
//!
//! # Features
//!
//! - DCT-based lossy compression
//! - Block-based motion compensation
//! - Three frame types: intra (I), predicted (P), and golden frames
//! - Huffman entropy coding
//! - YCbCr 4:2:0 color space
//!
//! # Example
//!
//! ```ignore
//! use transcode_theora::{TheoraDecoder, TheoraConfig};
//!
//! let config = TheoraConfig::new(1920, 1080)?;
//! let mut decoder = TheoraDecoder::new(config)?;
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod decoder;
#[cfg(feature = "encoder")]
pub mod encoder;
pub mod error;

pub use decoder::{DecodedFrame, TheoraDecoder};
pub use error::{Result, TheoraError};

#[cfg(feature = "encoder")]
pub use encoder::{EncodedPacket, TheoraEncoder, TheoraEncoderConfig};

/// Theora version constants.
pub const THEORA_VERSION_MAJOR: u8 = 3;
pub const THEORA_VERSION_MINOR: u8 = 2;
pub const THEORA_VERSION_SUBMINOR: u8 = 1;

/// Block size for DCT.
pub const BLOCK_SIZE: usize = 8;

/// Superblock size (4x4 blocks).
pub const SUPERBLOCK_SIZE: usize = 32;

/// Theora pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PixelFormat {
    /// YCbCr 4:2:0.
    #[default]
    Yuv420,
    /// YCbCr 4:2:2.
    Yuv422,
    /// YCbCr 4:4:4.
    Yuv444,
}

impl PixelFormat {
    /// Get chroma subsampling factors (horizontal, vertical).
    pub fn chroma_subsampling(&self) -> (u32, u32) {
        match self {
            Self::Yuv420 => (2, 2),
            Self::Yuv422 => (2, 1),
            Self::Yuv444 => (1, 1),
        }
    }

    /// Get number of planes.
    pub fn num_planes(&self) -> usize {
        3
    }
}

impl TryFrom<u8> for PixelFormat {
    type Error = TheoraError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Yuv420),
            1 => Ok(Self::Yuv422),
            2 => Ok(Self::Yuv444),
            _ => Err(TheoraError::InvalidHeader(format!(
                "Invalid pixel format: {}",
                value
            ))),
        }
    }
}

/// Theora color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// Unspecified.
    #[default]
    Unspecified,
    /// ITU-R BT.601.
    Bt601,
    /// ITU-R BT.709.
    Bt709,
}

impl TryFrom<u8> for ColorSpace {
    type Error = TheoraError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Unspecified),
            1 => Ok(Self::Bt601),
            2 => Ok(Self::Bt709),
            _ => Err(TheoraError::InvalidHeader(format!(
                "Invalid color space: {}",
                value
            ))),
        }
    }
}

/// Frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameType {
    /// Intra frame (keyframe).
    #[default]
    Intra,
    /// Predicted frame.
    Predicted,
}

/// Theora codec configuration.
#[derive(Debug, Clone)]
pub struct TheoraConfig {
    /// Picture width in pixels.
    pub pic_width: u32,
    /// Picture height in pixels.
    pub pic_height: u32,
    /// Frame width (multiple of 16).
    pub frame_width: u32,
    /// Frame height (multiple of 16).
    pub frame_height: u32,
    /// Picture X offset.
    pub pic_x: u32,
    /// Picture Y offset.
    pub pic_y: u32,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Pixel aspect ratio numerator.
    pub par_num: u32,
    /// Pixel aspect ratio denominator.
    pub par_den: u32,
    /// Pixel format.
    pub pixel_format: PixelFormat,
    /// Color space.
    pub color_space: ColorSpace,
    /// Nominal bitrate (0 = unspecified).
    pub bitrate: u32,
    /// Quality level (0-63).
    pub quality: u8,
    /// Version major.
    pub version_major: u8,
    /// Version minor.
    pub version_minor: u8,
    /// Version subminor.
    pub version_subminor: u8,
    /// Keyframe granule shift.
    pub keyframe_granule_shift: u8,
}

impl TheoraConfig {
    /// Create new configuration with defaults.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(TheoraError::InvalidDimensions { width, height });
        }

        // Frame dimensions must be multiples of 16
        let frame_width = (width + 15) & !15;
        let frame_height = (height + 15) & !15;

        Ok(Self {
            pic_width: width,
            pic_height: height,
            frame_width,
            frame_height,
            pic_x: 0,
            pic_y: frame_height - height, // Theora uses bottom-left origin
            fps_num: 30000,
            fps_den: 1001,
            par_num: 1,
            par_den: 1,
            pixel_format: PixelFormat::Yuv420,
            color_space: ColorSpace::Unspecified,
            bitrate: 0,
            quality: 48, // Default quality
            version_major: THEORA_VERSION_MAJOR,
            version_minor: THEORA_VERSION_MINOR,
            version_subminor: THEORA_VERSION_SUBMINOR,
            keyframe_granule_shift: 6,
        })
    }

    /// Set frame rate.
    pub fn set_framerate(&mut self, num: u32, den: u32) {
        self.fps_num = num.max(1);
        self.fps_den = den.max(1);
    }

    /// Set quality (0-63).
    pub fn set_quality(&mut self, quality: u8) {
        self.quality = quality.min(63);
    }

    /// Set target bitrate.
    pub fn set_bitrate(&mut self, bitrate: u32) {
        self.bitrate = bitrate;
    }

    /// Get number of macroblocks.
    pub fn num_macroblocks(&self) -> usize {
        let mb_width = (self.frame_width / 16) as usize;
        let mb_height = (self.frame_height / 16) as usize;
        mb_width * mb_height
    }

    /// Get number of superblocks.
    pub fn num_superblocks(&self) -> usize {
        let sb_width = (self.frame_width / SUPERBLOCK_SIZE as u32) as usize;
        let sb_height = (self.frame_height / SUPERBLOCK_SIZE as u32) as usize;
        (sb_width * sb_height).max(1)
    }

    /// Get number of blocks for a plane.
    pub fn num_blocks(&self, plane: usize) -> usize {
        let (h_sub, v_sub) = if plane == 0 {
            (1, 1)
        } else {
            self.pixel_format.chroma_subsampling()
        };

        let block_width = (self.frame_width / BLOCK_SIZE as u32 / h_sub) as usize;
        let block_height = (self.frame_height / BLOCK_SIZE as u32 / v_sub) as usize;
        block_width * block_height
    }

    /// Get plane dimensions.
    pub fn plane_dimensions(&self, plane: usize) -> (u32, u32) {
        if plane == 0 {
            (self.frame_width, self.frame_height)
        } else {
            let (h_sub, v_sub) = self.pixel_format.chroma_subsampling();
            (self.frame_width / h_sub, self.frame_height / v_sub)
        }
    }
}

impl Default for TheoraConfig {
    fn default() -> Self {
        Self::new(640, 480).unwrap()
    }
}

/// Loop filter limits per quality level.
pub const LOOP_FILTER_LIMITS: [u8; 64] = [
    30, 25, 20, 20, 15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 7, 7, 6, 6, 6,
    6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
];

/// AC scale factors.
pub const AC_SCALE: [u16; 64] = [
    500, 450, 400, 370, 340, 310, 285, 265, 245, 225, 210, 195, 185, 175, 165, 155, 145, 140, 135,
    130, 125, 120, 115, 110, 105, 100, 95, 93, 91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67,
    65, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42,
];

/// DC scale factors.
pub const DC_SCALE: [u16; 64] = [
    220, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 85, 80, 75, 70, 65, 60, 58, 56,
    54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,
    19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
];

/// Zigzag scan order for 8x8 blocks.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Base DCT matrices for intra Y blocks at each quality level.
pub const BASE_QUANT_INTRA_Y: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Base DCT matrices for intra UV blocks.
pub const BASE_QUANT_INTRA_UV: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// Base DCT matrices for inter blocks.
pub const BASE_QUANT_INTER: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
];

/// DCT coefficient for 8x8 transform.
fn dct_coeff(u: usize, x: usize) -> f32 {
    let n = BLOCK_SIZE as f32;
    let cu = if u == 0 {
        1.0 / 2.0_f32.sqrt()
    } else {
        1.0
    };
    cu * (std::f32::consts::PI * (2.0 * x as f32 + 1.0) * u as f32 / (2.0 * n)).cos()
        * (2.0 / n).sqrt()
}

/// Forward 8x8 DCT.
pub fn forward_dct(input: &[i16; 64], output: &mut [i32; 64]) {
    let mut temp = [0.0f32; 64];

    // Row DCT
    for y in 0..8 {
        for u in 0..8 {
            let mut sum = 0.0f32;
            for x in 0..8 {
                sum += input[y * 8 + x] as f32 * dct_coeff(u, x);
            }
            temp[y * 8 + u] = sum;
        }
    }

    // Column DCT
    for x in 0..8 {
        for v in 0..8 {
            let mut sum = 0.0f32;
            for y in 0..8 {
                sum += temp[y * 8 + x] * dct_coeff(v, y);
            }
            output[v * 8 + x] = sum.round() as i32;
        }
    }
}

/// Inverse 8x8 DCT.
pub fn inverse_dct(input: &[i32; 64], output: &mut [i16; 64]) {
    let mut temp = [0.0f32; 64];

    // Row IDCT
    for y in 0..8 {
        for x in 0..8 {
            let mut sum = 0.0f32;
            for u in 0..8 {
                sum += input[y * 8 + u] as f32 * dct_coeff(u, x);
            }
            temp[y * 8 + x] = sum;
        }
    }

    // Column IDCT
    for x in 0..8 {
        for y in 0..8 {
            let mut sum = 0.0f32;
            for v in 0..8 {
                sum += temp[v * 8 + x] * dct_coeff(v, y);
            }
            output[y * 8 + x] = sum.round().clamp(-32768.0, 32767.0) as i16;
        }
    }
}

/// Coding mode for macroblocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodingMode {
    /// Intra-coded.
    #[default]
    Intra,
    /// Inter, no motion.
    InterNoMv,
    /// Inter with motion vector.
    InterMv,
    /// Inter with motion from last frame.
    InterMvLast,
    /// Inter with motion from last2 frame.
    InterMvLast2,
    /// Inter using golden frame.
    InterGolden,
    /// Inter using golden frame with motion.
    InterGoldenMv,
    /// Four motion vectors.
    InterFourMv,
}

impl TryFrom<u8> for CodingMode {
    type Error = TheoraError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Intra),
            1 => Ok(Self::InterNoMv),
            2 => Ok(Self::InterMv),
            3 => Ok(Self::InterMvLast),
            4 => Ok(Self::InterMvLast2),
            5 => Ok(Self::InterGolden),
            6 => Ok(Self::InterGoldenMv),
            7 => Ok(Self::InterFourMv),
            _ => Err(TheoraError::BitstreamError(format!(
                "Invalid coding mode: {}",
                value
            ))),
        }
    }
}

/// Motion vector.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotionVector {
    /// X component.
    pub x: i16,
    /// Y component.
    pub y: i16,
}

impl MotionVector {
    /// Create zero motion vector.
    pub fn zero() -> Self {
        Self { x: 0, y: 0 }
    }

    /// Create new motion vector.
    pub fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Add two motion vectors.
    pub fn add(&self, other: &MotionVector) -> MotionVector {
        MotionVector {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = TheoraConfig::new(1920, 1080).unwrap();
        assert_eq!(config.pic_width, 1920);
        assert_eq!(config.pic_height, 1080);
        assert_eq!(config.frame_width, 1920);
        assert_eq!(config.frame_height, 1088); // Rounded up to multiple of 16
    }

    #[test]
    fn test_config_invalid_dimensions() {
        assert!(TheoraConfig::new(0, 480).is_err());
        assert!(TheoraConfig::new(640, 0).is_err());
    }

    #[test]
    fn test_config_default() {
        let config = TheoraConfig::default();
        assert_eq!(config.pic_width, 640);
        assert_eq!(config.pic_height, 480);
    }

    #[test]
    fn test_pixel_format() {
        assert_eq!(PixelFormat::Yuv420.chroma_subsampling(), (2, 2));
        assert_eq!(PixelFormat::Yuv422.chroma_subsampling(), (2, 1));
        assert_eq!(PixelFormat::Yuv444.chroma_subsampling(), (1, 1));
    }

    #[test]
    fn test_pixel_format_from_u8() {
        assert_eq!(PixelFormat::try_from(0u8).unwrap(), PixelFormat::Yuv420);
        assert_eq!(PixelFormat::try_from(1u8).unwrap(), PixelFormat::Yuv422);
        assert_eq!(PixelFormat::try_from(2u8).unwrap(), PixelFormat::Yuv444);
        assert!(PixelFormat::try_from(3u8).is_err());
    }

    #[test]
    fn test_color_space_from_u8() {
        assert_eq!(
            ColorSpace::try_from(0u8).unwrap(),
            ColorSpace::Unspecified
        );
        assert_eq!(ColorSpace::try_from(1u8).unwrap(), ColorSpace::Bt601);
        assert_eq!(ColorSpace::try_from(2u8).unwrap(), ColorSpace::Bt709);
        assert!(ColorSpace::try_from(3u8).is_err());
    }

    #[test]
    fn test_coding_mode_from_u8() {
        assert_eq!(CodingMode::try_from(0u8).unwrap(), CodingMode::Intra);
        assert_eq!(CodingMode::try_from(7u8).unwrap(), CodingMode::InterFourMv);
        assert!(CodingMode::try_from(8u8).is_err());
    }

    #[test]
    fn test_num_macroblocks() {
        let config = TheoraConfig::new(320, 240).unwrap();
        assert_eq!(config.num_macroblocks(), 20 * 15); // 300
    }

    #[test]
    fn test_num_superblocks() {
        let config = TheoraConfig::new(320, 240).unwrap();
        assert_eq!(config.num_superblocks(), 10 * 7); // 70 (320/32 * 240/32)
    }

    #[test]
    fn test_plane_dimensions() {
        let config = TheoraConfig::new(640, 480).unwrap();
        assert_eq!(config.plane_dimensions(0), (640, 480)); // Y
        assert_eq!(config.plane_dimensions(1), (320, 240)); // U
        assert_eq!(config.plane_dimensions(2), (320, 240)); // V
    }

    #[test]
    fn test_forward_inverse_dct() {
        let input: [i16; 64] = [
            52, 55, 61, 66, 70, 61, 64, 73, 63, 59, 55, 90, 109, 85, 69, 72, 62, 59, 68, 113, 144,
            104, 66, 73, 63, 58, 71, 122, 154, 106, 70, 69, 67, 61, 68, 104, 126, 88, 68, 70, 79,
            65, 60, 70, 77, 68, 58, 75, 85, 71, 64, 59, 55, 61, 65, 83, 87, 79, 69, 68, 65, 76, 78,
            94,
        ];

        let mut dct_output = [0i32; 64];
        forward_dct(&input, &mut dct_output);

        let mut idct_output = [0i16; 64];
        inverse_dct(&dct_output, &mut idct_output);

        // Check reconstruction is close to original
        for i in 0..64 {
            let diff = (input[i] as i32 - idct_output[i] as i32).abs();
            assert!(diff <= 2, "DCT/IDCT mismatch at {}: {} vs {}", i, input[i], idct_output[i]);
        }
    }

    #[test]
    fn test_motion_vector() {
        let mv1 = MotionVector::new(5, -3);
        let mv2 = MotionVector::new(-2, 4);
        let sum = mv1.add(&mv2);
        assert_eq!(sum.x, 3);
        assert_eq!(sum.y, 1);
    }

    #[test]
    fn test_zigzag_coverage() {
        // Verify zigzag covers all indices exactly once
        let mut seen = [false; 64];
        for &idx in &ZIGZAG {
            assert!(!seen[idx], "Duplicate index in zigzag: {}", idx);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s), "Missing indices in zigzag");
    }
}
