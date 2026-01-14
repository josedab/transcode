//! # transcode-vp9
//!
//! VP9 video codec implementation for the transcode library.
//!
//! This crate provides a pure Rust implementation of the VP9 video codec,
//! including decoder support for profiles 0-3.
//!
//! ## Features
//!
//! - **Profile Support**: Profiles 0, 1, 2, and 3 (8-bit, 10-bit, 12-bit)
//! - **Superblock Structure**: 64x64 superblocks with recursive partitioning
//! - **Reference Frames**: LAST, GOLDEN, ALTREF reference frame management
//! - **Transform Types**: DCT, ADST, and identity transforms
//! - **Loop Filter**: Deblocking filter with delta support
//! - **Segmentation**: Up to 8 segments with feature data
//! - **Boolean Entropy Coding**: Arithmetic coding for compressed data
//!
//! ## Example
//!
//! ```no_run
//! use transcode_vp9::{Vp9Decoder, Vp9DecoderConfig};
//! use transcode_core::Packet;
//!
//! let config = Vp9DecoderConfig::default();
//! let mut decoder = Vp9Decoder::new(config);
//!
//! // Decode VP9 data
//! # let vp9_data: &[u8] = &[];
//! let frames = decoder.decode_frame(vp9_data);
//! ```
//!
//! ## VP9 Bitstream Structure
//!
//! VP9 frames consist of:
//!
//! 1. **Uncompressed Header**: Fixed-length fields including frame type,
//!    dimensions, reference frame info, loop filter, quantization parameters
//!
//! 2. **Compressed Header**: Probability updates using boolean entropy coding
//!
//! 3. **Tile Data**: Actual coded video data split into tiles for parallel decoding
//!
//! ## Superframe Support
//!
//! VP9 supports superframes (multiple frames in a single packet) commonly used
//! for scalable coding. The decoder automatically handles superframe parsing.
//!
//! ## Color Spaces
//!
//! Supported color spaces:
//! - BT.601 (SD video)
//! - BT.709 (HD video)
//! - BT.2020 (UHD/HDR video)
//! - sRGB (for screen content)
//!
//! ## Chroma Subsampling
//!
//! - 4:2:0 (Profile 0/2)
//! - 4:2:2 and 4:4:4 (Profile 1/3)

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod decoder;
pub mod encoder;
pub mod entropy;
pub mod error;
pub mod frame_header;
pub mod prediction;

// Re-export main types
pub use decoder::{Vp9Decoder, Vp9DecoderConfig};
pub use encoder::{
    Vp9Encoder, Vp9EncoderConfig, Vp9RateControl, Vp9ContentType,
    Vp9Packet, Vp9FrameFlags, Vp9EncoderStats,
};
pub use entropy::{BoolDecoder, ProbabilityContext};
pub use error::{Result, Vp9Error};
pub use frame_header::{
    ChromaSubsampling, ColorRange, ColorSpace, FrameHeader, FrameType,
    InterpFilter, LoopFilterParams, Profile, QuantParams, RefFrame,
    SegmentationParams, TileInfo, TxMode,
};
pub use prediction::{
    BlockSize, IntraMode, IntraPredictor, InterMode, InterPredictor,
    LoopFilter, MotionVector, Partition, TxSize, TxType,
};

/// VP9 codec information.
pub const CODEC_NAME: &str = "vp9";

/// VP9 codec long name.
pub const CODEC_LONG_NAME: &str = "Google VP9";

/// VP9 frame sync code.
pub const VP9_SYNC_CODE: u32 = 0x498342;

/// Maximum supported frame width.
pub const VP9_MAX_WIDTH: u32 = 65536;

/// Maximum supported frame height.
pub const VP9_MAX_HEIGHT: u32 = 65536;

/// Superblock size (64x64).
pub const SUPERBLOCK_SIZE: u32 = 64;

/// Mode info block size (8x8).
pub const MI_SIZE: u32 = 8;

/// Number of reference frame slots.
pub const NUM_REF_FRAMES: usize = 8;

/// Number of active reference frames per inter frame.
pub const REFS_PER_FRAME: usize = 3;

/// Get codec information.
pub fn codec_info() -> CodecInfo {
    CodecInfo {
        name: CODEC_NAME,
        long_name: CODEC_LONG_NAME,
        can_encode: true,
        can_decode: true,
    }
}

/// Codec information structure.
#[derive(Debug, Clone)]
pub struct CodecInfo {
    /// Short codec name.
    pub name: &'static str,
    /// Long descriptive name.
    pub long_name: &'static str,
    /// Whether encoding is supported.
    pub can_encode: bool,
    /// Whether decoding is supported.
    pub can_decode: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_info() {
        let info = codec_info();
        assert_eq!(info.name, "vp9");
        assert_eq!(info.long_name, "Google VP9");
        assert!(info.can_encode);
        assert!(info.can_decode);
    }

    #[test]
    fn test_constants() {
        assert_eq!(VP9_SYNC_CODE, 0x498342);
        assert_eq!(SUPERBLOCK_SIZE, 64);
        assert_eq!(MI_SIZE, 8);
        assert_eq!(NUM_REF_FRAMES, 8);
        assert_eq!(REFS_PER_FRAME, 3);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = Vp9Decoder::new_default();
        assert!(decoder.width().is_none());
        assert!(decoder.height().is_none());
    }

    #[test]
    fn test_profile_enum() {
        assert_eq!(Profile::Profile0 as u8, 0);
        assert_eq!(Profile::Profile1 as u8, 1);
        assert_eq!(Profile::Profile2 as u8, 2);
        assert_eq!(Profile::Profile3 as u8, 3);
    }

    #[test]
    fn test_frame_type_enum() {
        assert!(matches!(FrameType::Keyframe, FrameType::Keyframe));
        assert!(matches!(FrameType::Inter, FrameType::Inter));
    }

    #[test]
    fn test_color_space_enum() {
        assert_eq!(ColorSpace::Unknown as u8, 0);
        assert_eq!(ColorSpace::Bt709 as u8, 2);
        assert_eq!(ColorSpace::Bt2020 as u8, 5);
        assert_eq!(ColorSpace::Srgb as u8, 7);
    }

    #[test]
    fn test_block_size_dimensions() {
        assert_eq!(BlockSize::Block64x64.width(), 64);
        assert_eq!(BlockSize::Block64x64.height(), 64);
        assert_eq!(BlockSize::Block4x4.width(), 4);
        assert_eq!(BlockSize::Block4x4.height(), 4);
    }

    #[test]
    fn test_motion_vector_operations() {
        let mv1 = MotionVector::new(10, 20);
        let mv2 = MotionVector::new(5, 10);

        let sum = mv1.add(&mv2);
        assert_eq!(sum.row, 15);
        assert_eq!(sum.col, 30);

        assert!(MotionVector::zero().is_zero());
        assert!(!mv1.is_zero());
    }

    #[test]
    fn test_intra_mode_properties() {
        // DC mode needs both neighbors
        assert!(IntraMode::Dc.needs_above());
        assert!(IntraMode::Dc.needs_left());

        // V mode only needs above
        assert!(IntraMode::V.needs_above());
        assert!(!IntraMode::V.needs_left());

        // H mode only needs left
        assert!(!IntraMode::H.needs_above());
        assert!(IntraMode::H.needs_left());

        // TM mode needs above-left
        assert!(IntraMode::Tm.needs_above_left());
    }

    #[test]
    fn test_tx_size_properties() {
        assert_eq!(TxSize::Tx4x4.size(), 4);
        assert_eq!(TxSize::Tx4x4.log2(), 2);
        assert_eq!(TxSize::Tx32x32.size(), 32);
        assert_eq!(TxSize::Tx32x32.log2(), 5);
    }

    #[test]
    fn test_error_types() {
        let err = Vp9Error::InvalidSyncCode(0x123456);
        assert!(err.to_string().contains("0x123456"));

        let err = Vp9Error::UnsupportedProfile(4);
        assert!(err.to_string().contains("4"));

        let err = Vp9Error::MissingRefFrame(1);
        assert!(err.is_recoverable());

        let err = Vp9Error::NotInitialized;
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_bool_decoder_creation() {
        let data = [0x80, 0x00, 0x00, 0x00];
        let decoder = BoolDecoder::new(&data);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_probability_context() {
        let ctx = ProbabilityContext::new();
        assert_eq!(ctx.skip_probs.len(), 3);
        assert_eq!(ctx.intra_inter_probs.len(), 4);
    }

    #[test]
    fn test_frame_header_defaults() {
        let header = FrameHeader::default();
        assert_eq!(header.profile, Profile::Profile0);
        assert_eq!(header.bit_depth, 8);
        assert!(!header.show_existing_frame);
        assert!(header.show_frame);
    }

    #[test]
    fn test_loop_filter_params() {
        let lf = LoopFilterParams::default();
        assert_eq!(lf.level, 0);
        assert_eq!(lf.sharpness, 0);
        assert!(!lf.delta_enabled);
        assert_eq!(lf.ref_deltas.len(), 4);
        assert_eq!(lf.mode_deltas.len(), 2);
    }

    #[test]
    fn test_segmentation_params() {
        let seg = SegmentationParams::default();
        assert!(!seg.enabled);
        assert!(!seg.update_map);
        assert_eq!(seg.tree_probs.len(), 7);
        assert_eq!(seg.pred_probs.len(), 3);
    }

    #[test]
    fn test_quant_params() {
        let quant = QuantParams::default();
        assert_eq!(quant.base_q_idx, 0);
        assert_eq!(quant.delta_q_y_dc, 0);
    }

    #[test]
    fn test_tile_info() {
        let tile = TileInfo::default();
        assert_eq!(tile.tile_cols_log2, 0);
        assert_eq!(tile.tile_rows_log2, 0);
    }

    #[test]
    fn test_interp_filter() {
        assert_eq!(InterpFilter::EightTapSmooth as u8, 0);
        assert_eq!(InterpFilter::Bilinear as u8, 3);
        assert_eq!(InterpFilter::Switchable as u8, 4);
    }

    #[test]
    fn test_decoder_config() {
        let config = Vp9DecoderConfig::default();
        assert_eq!(config.max_threads, 1);
        assert!(config.error_concealment);
        assert!(config.enable_loop_filter);
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = Vp9Decoder::new_default();
        decoder.reset_decoder();
        assert!(decoder.width().is_none());
        assert!(decoder.height().is_none());
    }
}
