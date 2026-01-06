//! VP8 video codec implementation.
//!
//! This crate provides VP8 encoding and decoding for the transcode library.
//! VP8 is an open, royalty-free video codec developed by On2 Technologies
//! and now owned by Google. It is commonly used in the WebM container format.
//!
//! ## Features
//!
//! - Baseline VP8 decoding
//! - Keyframe and interframe support
//! - Multiple reference frames (Last, Golden, AltRef)
//! - Loop filtering
//! - Boolean arithmetic coder (BAC)
//!
//! ## Example
//!
//! ```ignore
//! use transcode_vp8::{Vp8Decoder, Vp8Frame};
//!
//! let mut decoder = Vp8Decoder::new()?;
//! let frame = decoder.decode(&compressed_data)?;
//! ```

#![warn(missing_docs)]

pub mod error;
mod bool_decoder;
mod decoder;
mod frame;
mod prediction;
mod transform;
mod loop_filter;

#[cfg(feature = "encoder")]
mod encoder;

pub use error::{Vp8Error, Result};
pub use decoder::{Vp8Decoder, Vp8DecoderConfig};
pub use frame::{Vp8Frame, Vp8FrameType, Vp8ColorSpace};
pub use bool_decoder::BoolDecoder;

#[cfg(feature = "encoder")]
pub use bool_decoder::BoolEncoder;

#[cfg(feature = "encoder")]
pub use encoder::{Vp8Encoder, Vp8EncoderConfig, RateControlMode, EncodedPacket};

/// VP8 profile (only profile 0 is commonly used).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vp8Profile {
    /// Simple profile (profile 0).
    Simple = 0,
    /// Profile 1 (with threading).
    Profile1 = 1,
    /// Profile 2 (bilinear MC).
    Profile2 = 2,
    /// Profile 3 (full).
    Profile3 = 3,
}

/// VP8 frame header.
#[derive(Debug, Clone)]
pub struct Vp8FrameHeader {
    /// Frame type (key or inter).
    pub frame_type: Vp8FrameType,
    /// Version (profile).
    pub version: u8,
    /// Show frame flag.
    pub show_frame: bool,
    /// First partition size.
    pub first_part_size: u32,
    /// Frame width.
    pub width: u16,
    /// Horizontal scale.
    pub horizontal_scale: u8,
    /// Frame height.
    pub height: u16,
    /// Vertical scale.
    pub vertical_scale: u8,
    /// Color space.
    pub color_space: Vp8ColorSpace,
    /// Clamping required.
    pub clamping_required: bool,
    /// Segmentation enabled.
    pub segmentation_enabled: bool,
    /// Loop filter type.
    pub filter_type: u8,
    /// Loop filter level.
    pub filter_level: u8,
    /// Loop filter sharpness.
    pub sharpness_level: u8,
    /// Number of token partitions (log2).
    pub log2_nbr_of_dct_partitions: u8,
    /// Quantizer update.
    pub q_index: u8,
    /// Refresh entropy probs.
    pub refresh_entropy_probs: bool,
    /// Refresh golden frame.
    pub refresh_golden_frame: bool,
    /// Refresh alt-ref frame.
    pub refresh_alt_ref_frame: bool,
    /// Golden frame copy flag.
    pub copy_buffer_to_golden: u8,
    /// Alt-ref copy flag.
    pub copy_buffer_to_alt_ref: u8,
    /// Reference frame sign bias.
    pub ref_frame_sign_bias: [bool; 4],
    /// Refresh last frame.
    pub refresh_last_frame: bool,
    /// MB no-skip coefficient.
    pub mb_no_skip_coeff: bool,
    /// Probability of MB being skipped.
    pub prob_skip_false: u8,
    /// Probability of using INTRA mode.
    pub prob_intra: u8,
    /// Probability of using LAST reference.
    pub prob_last: u8,
    /// Probability of using GOLDEN reference.
    pub prob_golden: u8,
}

impl Default for Vp8FrameHeader {
    fn default() -> Self {
        Self {
            frame_type: Vp8FrameType::KeyFrame,
            version: 0,
            show_frame: true,
            first_part_size: 0,
            width: 0,
            height: 0,
            horizontal_scale: 0,
            vertical_scale: 0,
            color_space: Vp8ColorSpace::Bt601,
            clamping_required: false,
            segmentation_enabled: false,
            filter_type: 0,
            filter_level: 0,
            sharpness_level: 0,
            log2_nbr_of_dct_partitions: 0,
            q_index: 0,
            refresh_entropy_probs: true,
            refresh_golden_frame: false,
            refresh_alt_ref_frame: false,
            copy_buffer_to_golden: 0,
            copy_buffer_to_alt_ref: 0,
            ref_frame_sign_bias: [false; 4],
            refresh_last_frame: true,
            mb_no_skip_coeff: true,
            prob_skip_false: 0,
            prob_intra: 0,
            prob_last: 0,
            prob_golden: 0,
        }
    }
}

/// Macroblock mode for luma.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MbLumaMode {
    /// DC prediction.
    #[default]
    DcPred = 0,
    /// Vertical prediction.
    VPred = 1,
    /// Horizontal prediction.
    HPred = 2,
    /// True motion prediction.
    TmPred = 3,
    /// Use sub-block modes (B_PRED).
    BPred = 4,
}

/// Macroblock mode for chroma.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MbChromaMode {
    /// DC prediction.
    #[default]
    DcPred = 0,
    /// Vertical prediction.
    VPred = 1,
    /// Horizontal prediction.
    HPred = 2,
    /// True motion prediction.
    TmPred = 3,
}

/// Sub-block mode (for B_PRED).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubBlockMode {
    /// DC prediction.
    BDcPred = 0,
    /// True motion prediction.
    BTmPred = 1,
    /// Vertical prediction.
    BVePred = 2,
    /// Horizontal prediction.
    BHePred = 3,
    /// Left-down prediction.
    BLdPred = 4,
    /// Right-down prediction.
    BRdPred = 5,
    /// Vertical-right prediction.
    BVrPred = 6,
    /// Vertical-left prediction.
    BVlPred = 7,
    /// Horizontal-down prediction.
    BHdPred = 8,
    /// Horizontal-up prediction.
    BHuPred = 9,
}

/// Motion vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionVector {
    /// Horizontal component (in 1/8 pixel units).
    pub x: i16,
    /// Vertical component (in 1/8 pixel units).
    pub y: i16,
}

impl MotionVector {
    /// Create a new motion vector.
    pub fn new(x: i16, y: i16) -> Self {
        Self { x, y }
    }

    /// Zero motion vector.
    pub const ZERO: Self = Self { x: 0, y: 0 };

    /// Add two motion vectors.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

/// VP8 macroblock.
#[derive(Debug, Clone)]
pub struct Macroblock {
    /// Luma prediction mode.
    pub y_mode: MbLumaMode,
    /// Chroma prediction mode.
    pub uv_mode: MbChromaMode,
    /// Reference frame.
    pub ref_frame: u8,
    /// Motion vectors (one per sub-block, 16 total).
    pub mvs: [MotionVector; 16],
    /// Sub-block modes (for B_PRED).
    pub sub_modes: [SubBlockMode; 16],
    /// Segment ID.
    pub segment_id: u8,
    /// Skip coefficient flag.
    pub skip_coeff: bool,
    /// Non-zero coefficient mask for luma.
    pub nz_luma: u16,
    /// Non-zero coefficient mask for chroma.
    pub nz_chroma: u8,
}

impl Default for Macroblock {
    fn default() -> Self {
        Self {
            y_mode: MbLumaMode::DcPred,
            uv_mode: MbChromaMode::DcPred,
            ref_frame: 0,
            mvs: [MotionVector::ZERO; 16],
            sub_modes: [SubBlockMode::BDcPred; 16],
            segment_id: 0,
            skip_coeff: false,
            nz_luma: 0,
            nz_chroma: 0,
        }
    }
}

/// Segment feature parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct SegmentFeature {
    /// Quantizer update value.
    pub quant_update: i8,
    /// Whether quantizer update is absolute.
    pub quant_absolute: bool,
    /// Loop filter update value.
    pub lf_update: i8,
    /// Whether loop filter update is absolute.
    pub lf_absolute: bool,
}

/// Default coefficient probabilities (simplified for common blocks).
pub const DEFAULT_COEFF_PROBS: [[[u8; 11]; 3]; 8] = [
    [[128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
     [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
     [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
    [[253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128],
     [189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128],
     [106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128]],
    [[1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128],
     [181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128],
     [78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128]],
    [[1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128],
     [184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128],
     [77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128]],
    [[1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128],
     [170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128],
     [37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128]],
    [[1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128],
     [207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128],
     [102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128]],
    [[1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128],
     [177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128],
     [80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128]],
    [[1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
     [246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
     [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
];

/// Default motion vector probabilities.
pub const DEFAULT_MV_PROBS: [[u8; 19]; 2] = [
    [162, 128, 225, 146, 172, 147, 214, 39, 156,
     128, 129, 132, 75, 145, 178, 206, 239, 254, 254],
    [164, 128, 204, 170, 119, 235, 140, 230, 228,
     128, 130, 130, 74, 148, 180, 203, 236, 254, 254],
];

/// Default Y mode probabilities.
pub const DEFAULT_Y_MODE_PROBS: [u8; 4] = [145, 156, 163, 128];

/// Default probability tables for VP8.
pub mod default_probs {
    /// Default coefficient probabilities.
    #[rustfmt::skip]
    pub const COEF_PROBS: [[[[u8; 11]; 3]; 8]; 4] = [
        // Block type 0 (Y after Y2)
        [
            [[128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
             [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
             [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
            [[253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128],
             [189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128],
             [106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128]],
            [[1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128],
             [181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128],
             [78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128]],
            [[1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128],
             [184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128],
             [77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128]],
            [[1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128],
             [170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128],
             [37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128]],
            [[1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128],
             [207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128],
             [102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128]],
            [[1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128],
             [177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128],
             [80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128]],
            [[1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
             [246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
             [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
        ],
        // Block types 1-3 follow similar pattern...
        [
            [[128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
             [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
             [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
            [[198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62],
             [131, 45, 198, 221, 172, 176, 220, 157, 252, 128, 128],
             [68, 47, 146, 208, 149, 167, 221, 162, 255, 128, 128]],
            [[1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128],
             [184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128],
             [81, 99, 181, 242, 176, 190, 249, 202, 255, 128, 128]],
            [[1, 129, 232, 253, 214, 197, 242, 196, 255, 128, 128],
             [99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128],
             [23, 91, 163, 242, 170, 187, 247, 210, 255, 128, 128]],
            [[1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128],
             [109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128],
             [44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128]],
            [[1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128],
             [94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128],
             [22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128]],
            [[1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128],
             [124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128],
             [35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128]],
            [[1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128],
             [121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128],
             [45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128]],
        ],
        [
            [[128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
             [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
             [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
            [[202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255],
             [126, 38, 182, 232, 169, 184, 228, 174, 255, 187, 128],
             [61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128]],
            [[1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128],
             [166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128],
             [39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128]],
            [[1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128],
             [124, 74, 191, 243, 183, 193, 250, 221, 255, 255, 128],
             [24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128]],
            [[1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128],
             [149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128],
             [28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128]],
            [[1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128],
             [123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128],
             [20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128]],
            [[1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128],
             [168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128],
             [47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128]],
            [[1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128],
             [141, 84, 213, 252, 201, 202, 255, 219, 128, 128, 128],
             [42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128]],
        ],
        [
            [[128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
             [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
             [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
            [[202, 40, 227, 251, 213, 181, 255, 171, 128, 128, 128],
             [152, 69, 192, 238, 185, 176, 255, 255, 128, 128, 128],
             [83, 40, 139, 224, 155, 152, 255, 255, 128, 128, 128]],
            [[1, 107, 238, 254, 213, 175, 255, 255, 128, 128, 128],
             [147, 108, 227, 254, 210, 194, 255, 255, 128, 128, 128],
             [28, 87, 164, 243, 170, 159, 255, 255, 128, 128, 128]],
            [[1, 49, 214, 252, 214, 243, 255, 191, 128, 128, 128],
             [142, 57, 193, 249, 191, 221, 255, 205, 128, 128, 128],
             [31, 59, 142, 236, 180, 211, 255, 255, 128, 128, 128]],
            [[1, 200, 252, 255, 241, 241, 128, 128, 128, 128, 128],
             [175, 163, 249, 255, 242, 243, 128, 128, 128, 128, 128],
             [39, 160, 219, 255, 221, 213, 128, 128, 128, 128, 128]],
            [[1, 107, 236, 254, 217, 211, 255, 255, 128, 128, 128],
             [134, 95, 210, 252, 195, 201, 255, 255, 128, 128, 128],
             [18, 73, 147, 236, 167, 191, 255, 224, 128, 128, 128]],
            [[1, 216, 255, 255, 250, 243, 128, 128, 128, 128, 128],
             [200, 163, 253, 255, 248, 232, 128, 128, 128, 128, 128],
             [85, 126, 241, 255, 238, 234, 128, 128, 128, 128, 128]],
            [[1, 148, 246, 255, 231, 218, 128, 128, 128, 128, 128],
             [143, 102, 231, 255, 221, 214, 255, 255, 128, 128, 128],
             [39, 103, 188, 255, 200, 199, 255, 255, 128, 128, 128]],
        ],
    ];

    /// Default MB intra mode probabilities.
    pub const MB_INTRA_MODE_PROBS: [u8; 4] = [145, 156, 163, 128];

    /// Default sub-block mode probabilities.
    pub const SUB_BLOCK_MODE_PROBS: [[u8; 9]; 10] = [
        [231, 120, 48, 89, 115, 113, 120, 152, 112],
        [152, 179, 64, 126, 170, 118, 46, 70, 95],
        [175, 69, 143, 80, 85, 82, 72, 155, 103],
        [56, 58, 10, 171, 218, 189, 17, 13, 152],
        [114, 26, 17, 163, 44, 195, 21, 10, 173],
        [121, 24, 80, 195, 26, 62, 44, 64, 85],
        [144, 71, 10, 38, 171, 213, 144, 34, 26],
        [170, 46, 55, 19, 136, 160, 33, 206, 71],
        [63, 20, 8, 114, 114, 208, 12, 9, 226],
        [81, 40, 11, 96, 182, 84, 29, 16, 36],
    ];

    /// Default MV probability updates.
    pub const MV_UPDATE_PROBS: [[u8; 19]; 2] = [
        [237, 246, 253, 253, 254, 254, 254, 254, 254,
         254, 254, 254, 254, 254, 250, 250, 252, 254, 254],
        [231, 243, 245, 253, 254, 254, 254, 254, 254,
         254, 254, 254, 254, 254, 251, 251, 254, 254, 254],
    ];

    /// Default MV probabilities.
    pub const MV_DEFAULT_PROBS: [[u8; 19]; 2] = [
        [162, 128, 225, 146, 172, 147, 214, 39, 156,
         128, 129, 132, 75, 145, 178, 206, 239, 254, 254],
        [164, 128, 204, 170, 119, 235, 140, 230, 228,
         128, 130, 130, 74, 148, 180, 203, 236, 254, 254],
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector() {
        let mv1 = MotionVector::new(10, -5);
        let mv2 = MotionVector::new(-3, 8);
        let sum = mv1.add(&mv2);
        assert_eq!(sum.x, 7);
        assert_eq!(sum.y, 3);
    }

    #[test]
    fn test_frame_header_default() {
        let header = Vp8FrameHeader::default();
        assert_eq!(header.frame_type, Vp8FrameType::KeyFrame);
        assert!(header.show_frame);
    }

    #[test]
    fn test_macroblock_default() {
        let mb = Macroblock::default();
        assert_eq!(mb.y_mode, MbLumaMode::DcPred);
        assert!(!mb.skip_coeff);
    }
}
