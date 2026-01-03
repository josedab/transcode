//! VP9 frame header parsing.
//!
//! This module handles parsing of VP9 uncompressed and compressed headers.
//! VP9 frames consist of:
//! - Uncompressed header (fixed-length fields)
//! - Compressed header (probability updates, using bool decoder)
//! - Tile data

use crate::entropy::{BoolDecoder, ProbabilityContext};
use crate::error::{Result, Vp9Error};
use transcode_core::bitstream::BitReader;

/// VP9 frame sync code (0x498342).
pub const VP9_FRAME_SYNC_CODE: u32 = 0x498342;

/// Maximum frame width.
pub const VP9_MAX_WIDTH: u32 = 65536;
/// Maximum frame height.
pub const VP9_MAX_HEIGHT: u32 = 65536;

/// VP9 profile (0-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Profile {
    /// Profile 0: 8-bit, 4:2:0.
    #[default]
    Profile0 = 0,
    /// Profile 1: 8-bit, 4:2:2/4:4:4.
    Profile1 = 1,
    /// Profile 2: 10/12-bit, 4:2:0.
    Profile2 = 2,
    /// Profile 3: 10/12-bit, 4:2:2/4:4:4.
    Profile3 = 3,
}

impl TryFrom<u8> for Profile {
    type Error = Vp9Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Profile::Profile0),
            1 => Ok(Profile::Profile1),
            2 => Ok(Profile::Profile2),
            3 => Ok(Profile::Profile3),
            _ => Err(Vp9Error::UnsupportedProfile(value)),
        }
    }
}

/// VP9 color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ColorSpace {
    /// Unknown color space.
    Unknown = 0,
    /// BT.601.
    #[default]
    Bt601 = 1,
    /// BT.709.
    Bt709 = 2,
    /// SMPTE-170M.
    Smpte170 = 3,
    /// SMPTE-240M.
    Smpte240 = 4,
    /// BT.2020.
    Bt2020 = 5,
    /// Reserved.
    Reserved = 6,
    /// sRGB (RGB only).
    Srgb = 7,
}

impl TryFrom<u8> for ColorSpace {
    type Error = Vp9Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(ColorSpace::Unknown),
            1 => Ok(ColorSpace::Bt601),
            2 => Ok(ColorSpace::Bt709),
            3 => Ok(ColorSpace::Smpte170),
            4 => Ok(ColorSpace::Smpte240),
            5 => Ok(ColorSpace::Bt2020),
            6 => Ok(ColorSpace::Reserved),
            7 => Ok(ColorSpace::Srgb),
            _ => Err(Vp9Error::UnsupportedColorSpace(value)),
        }
    }
}

/// VP9 color range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorRange {
    /// Studio/TV range (16-235).
    #[default]
    Studio,
    /// Full range (0-255).
    Full,
}

/// VP9 chroma subsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromaSubsampling {
    /// 4:2:0 subsampling.
    #[default]
    Cs420,
    /// 4:2:2 subsampling.
    Cs422,
    /// 4:4:4 subsampling.
    Cs444,
}

/// VP9 frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameType {
    /// Keyframe (intra-only).
    #[default]
    Keyframe,
    /// Inter frame (predicted).
    Inter,
}

/// VP9 interpolation filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum InterpFilter {
    /// 8-tap smooth filter.
    #[default]
    EightTapSmooth = 0,
    /// 8-tap regular filter.
    EightTap = 1,
    /// 8-tap sharp filter.
    EightTapSharp = 2,
    /// Bilinear filter.
    Bilinear = 3,
    /// Switchable (per-block).
    Switchable = 4,
}

impl TryFrom<u8> for InterpFilter {
    type Error = Vp9Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(InterpFilter::EightTapSmooth),
            1 => Ok(InterpFilter::EightTap),
            2 => Ok(InterpFilter::EightTapSharp),
            3 => Ok(InterpFilter::Bilinear),
            4 => Ok(InterpFilter::Switchable),
            _ => Err(Vp9Error::InvalidInterpFilter(value)),
        }
    }
}

/// VP9 transform mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TxMode {
    /// Only 4x4 transforms.
    Only4x4 = 0,
    /// Allow up to 8x8 transforms.
    #[default]
    Allow8x8 = 1,
    /// Allow up to 16x16 transforms.
    Allow16x16 = 2,
    /// Allow up to 32x32 transforms.
    Allow32x32 = 3,
    /// Select transform size per block.
    Select = 4,
}

/// VP9 reference frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RefFrame {
    /// No reference (intra).
    None = 0,
    /// Last reference frame.
    Last = 1,
    /// Golden reference frame.
    Golden = 2,
    /// Altref reference frame.
    Altref = 3,
}

/// VP9 loop filter parameters.
#[derive(Debug, Clone, Default)]
pub struct LoopFilterParams {
    /// Base filter level (0-63).
    pub level: u8,
    /// Sharpness (0-7).
    pub sharpness: u8,
    /// Delta enabled.
    pub delta_enabled: bool,
    /// Delta update flag.
    pub delta_update: bool,
    /// Reference frame deltas.
    pub ref_deltas: [i8; 4],
    /// Mode deltas.
    pub mode_deltas: [i8; 2],
}

/// VP9 segmentation parameters.
#[derive(Debug, Clone, Default)]
pub struct SegmentationParams {
    /// Segmentation enabled.
    pub enabled: bool,
    /// Update map flag.
    pub update_map: bool,
    /// Temporal update flag.
    pub temporal_update: bool,
    /// Update data flag.
    pub update_data: bool,
    /// Absolute or delta values.
    pub abs_or_delta: bool,
    /// Segment feature data.
    pub features: [[i16; 4]; 8],
    /// Segment feature enabled.
    pub feature_enabled: [[bool; 4]; 8],
    /// Tree probabilities.
    pub tree_probs: [u8; 7],
    /// Prediction probabilities.
    pub pred_probs: [u8; 3],
}

/// VP9 quantization parameters.
#[derive(Debug, Clone, Default)]
pub struct QuantParams {
    /// Base Y DC delta.
    pub base_q_idx: u8,
    /// Y DC delta.
    pub delta_q_y_dc: i8,
    /// UV DC delta.
    pub delta_q_uv_dc: i8,
    /// UV AC delta.
    pub delta_q_uv_ac: i8,
    /// Lossless mode flag.
    pub lossless: bool,
}

/// VP9 tile configuration.
#[derive(Debug, Clone, Default)]
pub struct TileInfo {
    /// Log2 of tile columns.
    pub tile_cols_log2: u8,
    /// Log2 of tile rows.
    pub tile_rows_log2: u8,
    /// Number of tile columns.
    pub tile_cols: u32,
    /// Number of tile rows.
    pub tile_rows: u32,
}

/// VP9 frame header.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// VP9 profile (0-3).
    pub profile: Profile,
    /// Show existing frame flag.
    pub show_existing_frame: bool,
    /// Frame to show index.
    pub frame_to_show_map_idx: u8,
    /// Frame type (keyframe or inter).
    pub frame_type: FrameType,
    /// Show frame flag.
    pub show_frame: bool,
    /// Error resilient mode.
    pub error_resilient: bool,
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,
    /// Color space.
    pub color_space: ColorSpace,
    /// Color range.
    pub color_range: ColorRange,
    /// Chroma subsampling.
    pub subsampling: ChromaSubsampling,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Render width.
    pub render_width: u32,
    /// Render height.
    pub render_height: u32,
    /// Intra-only flag.
    pub intra_only: bool,
    /// Reset frame context.
    pub reset_frame_context: u8,
    /// Reference frame indices.
    pub ref_frame_idx: [u8; 3],
    /// Reference frame sign bias.
    pub ref_frame_sign_bias: [bool; 4],
    /// Allow high precision MV.
    pub allow_high_precision_mv: bool,
    /// Interpolation filter.
    pub interp_filter: InterpFilter,
    /// Refresh frame flags.
    pub refresh_frame_flags: u8,
    /// Loop filter parameters.
    pub loop_filter: LoopFilterParams,
    /// Quantization parameters.
    pub quant: QuantParams,
    /// Segmentation parameters.
    pub segmentation: SegmentationParams,
    /// Tile configuration.
    pub tile_info: TileInfo,
    /// Transform mode.
    pub tx_mode: TxMode,
    /// Reference mode.
    pub reference_mode: u8,
    /// Compressed header size.
    pub header_size: u16,
}

impl Default for FrameHeader {
    fn default() -> Self {
        Self {
            profile: Profile::default(),
            show_existing_frame: false,
            frame_to_show_map_idx: 0,
            frame_type: FrameType::default(),
            show_frame: true,
            error_resilient: false,
            bit_depth: 8,
            color_space: ColorSpace::default(),
            color_range: ColorRange::default(),
            subsampling: ChromaSubsampling::default(),
            width: 0,
            height: 0,
            render_width: 0,
            render_height: 0,
            intra_only: false,
            reset_frame_context: 0,
            ref_frame_idx: [0; 3],
            ref_frame_sign_bias: [false; 4],
            allow_high_precision_mv: false,
            interp_filter: InterpFilter::default(),
            refresh_frame_flags: 0,
            loop_filter: LoopFilterParams::default(),
            quant: QuantParams::default(),
            segmentation: SegmentationParams::default(),
            tile_info: TileInfo::default(),
            tx_mode: TxMode::default(),
            reference_mode: 0,
            header_size: 0,
        }
    }
}

impl FrameHeader {
    /// Parse the uncompressed frame header.
    pub fn parse(data: &[u8]) -> Result<(Self, usize)> {
        let mut reader = BitReader::new(data);
        let mut header = FrameHeader::default();

        // Frame marker (should be 2)
        let frame_marker = reader.read_bits(2).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;
        if frame_marker != 2 {
            return Err(Vp9Error::InvalidFrameMarker(frame_marker));
        }

        // Profile
        let profile_low = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
        let profile_high = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
        let profile = (profile_high as u8) << 1 | (profile_low as u8);
        if profile == 3 {
            // Reserved bit for profile 3
            let _reserved = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
        }
        header.profile = Profile::try_from(profile)?;

        // Show existing frame
        header.show_existing_frame = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
        if header.show_existing_frame {
            header.frame_to_show_map_idx = reader.read_bits(3).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;
            let bytes_read = reader.position().div_ceil(8);
            return Ok((header, bytes_read));
        }

        // Frame type
        header.frame_type = if reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? {
            FrameType::Inter
        } else {
            FrameType::Keyframe
        };

        // Show frame
        header.show_frame = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;

        // Error resilient
        header.error_resilient = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;

        if header.frame_type == FrameType::Keyframe {
            // Sync code
            let sync_code = reader.read_bits(24).map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
            if sync_code != VP9_FRAME_SYNC_CODE {
                return Err(Vp9Error::InvalidSyncCode(sync_code));
            }

            // Color config
            Self::parse_color_config(&mut reader, &mut header)?;

            // Frame size
            Self::parse_frame_size(&mut reader, &mut header)?;

            // Render size
            Self::parse_render_size(&mut reader, &mut header)?;

            header.refresh_frame_flags = 0xFF;
        } else {
            header.intra_only = if header.show_frame {
                false
            } else {
                reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?
            };

            if header.intra_only {
                // Sync code for intra-only
                let sync_code = reader.read_bits(24).map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
                if sync_code != VP9_FRAME_SYNC_CODE {
                    return Err(Vp9Error::InvalidSyncCode(sync_code));
                }

                if header.profile > Profile::Profile0 {
                    Self::parse_color_config(&mut reader, &mut header)?;
                } else {
                    header.color_space = ColorSpace::Bt601;
                    header.subsampling = ChromaSubsampling::Cs420;
                    header.bit_depth = 8;
                }

                header.refresh_frame_flags = reader.read_bits(8).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;
                Self::parse_frame_size(&mut reader, &mut header)?;
                Self::parse_render_size(&mut reader, &mut header)?;
            } else {
                header.reset_frame_context = if header.error_resilient {
                    0
                } else {
                    reader.read_bits(2).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8
                };

                header.refresh_frame_flags = reader.read_bits(8).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;

                // Reference frames
                for i in 0..3 {
                    header.ref_frame_idx[i] = reader.read_bits(3).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;
                    header.ref_frame_sign_bias[i + 1] = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
                }

                // Frame size with refs
                let found_ref = Self::parse_frame_size_with_refs(&mut reader, &mut header)?;
                if !found_ref {
                    Self::parse_frame_size(&mut reader, &mut header)?;
                }
                Self::parse_render_size(&mut reader, &mut header)?;

                header.allow_high_precision_mv = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;

                // Interpolation filter
                let is_filter_switchable = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
                if is_filter_switchable {
                    header.interp_filter = InterpFilter::Switchable;
                } else {
                    let raw_filter = reader.read_bits(2).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;
                    header.interp_filter = InterpFilter::try_from(raw_filter)?;
                }
            }
        }

        if !header.error_resilient {
            // Refresh frame context
            let _refresh_frame_context = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
            // Frame parallel decoding
            let _frame_parallel = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
        }

        // Frame context index
        let _frame_context_idx = reader.read_bits(2).map_err(|_| Vp9Error::UnexpectedEndOfStream)?;

        // Loop filter
        Self::parse_loop_filter(&mut reader, &mut header)?;

        // Quantization
        Self::parse_quantization(&mut reader, &mut header)?;

        // Segmentation
        Self::parse_segmentation(&mut reader, &mut header)?;

        // Tile info
        Self::parse_tile_info(&mut reader, &mut header)?;

        // Compressed header size
        header.header_size = reader.read_bits(16).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u16;

        let bytes_read = reader.position().div_ceil(8);
        Ok((header, bytes_read))
    }

    fn parse_color_config(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        if header.profile >= Profile::Profile2 {
            let ten_or_twelve_bit = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
            header.bit_depth = if ten_or_twelve_bit { 12 } else { 10 };
        } else {
            header.bit_depth = 8;
        }

        let color_space = reader.read_bits(3).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;
        header.color_space = ColorSpace::try_from(color_space)?;

        if header.color_space != ColorSpace::Srgb {
            header.color_range = if reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? {
                ColorRange::Full
            } else {
                ColorRange::Studio
            };

            if header.profile == Profile::Profile1 || header.profile == Profile::Profile3 {
                let subsampling_x = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
                let subsampling_y = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
                let _reserved = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;

                header.subsampling = match (subsampling_x, subsampling_y) {
                    (true, true) => ChromaSubsampling::Cs420,
                    (true, false) => ChromaSubsampling::Cs422,
                    (false, false) => ChromaSubsampling::Cs444,
                    (false, true) => return Err(Vp9Error::UnsupportedColorSpace(0)),
                };
            } else {
                header.subsampling = ChromaSubsampling::Cs420;
            }
        } else {
            header.color_range = ColorRange::Full;
            if header.profile == Profile::Profile1 || header.profile == Profile::Profile3 {
                header.subsampling = ChromaSubsampling::Cs444;
                let _reserved = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
            }
        }

        Ok(())
    }

    fn parse_frame_size(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        header.width = reader.read_bits(16).map_err(|_| Vp9Error::UnexpectedEndOfStream)? + 1;
        header.height = reader.read_bits(16).map_err(|_| Vp9Error::UnexpectedEndOfStream)? + 1;

        if header.width > VP9_MAX_WIDTH || header.height > VP9_MAX_HEIGHT {
            return Err(Vp9Error::DimensionsTooLarge {
                width: header.width,
                height: header.height,
                max_width: VP9_MAX_WIDTH,
                max_height: VP9_MAX_HEIGHT,
            });
        }

        if header.width == 0 || header.height == 0 {
            return Err(Vp9Error::InvalidDimensions {
                width: header.width,
                height: header.height,
            });
        }

        header.render_width = header.width;
        header.render_height = header.height;

        Ok(())
    }

    fn parse_render_size(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        let render_and_frame_size_different = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
        if render_and_frame_size_different {
            header.render_width = reader.read_bits(16).map_err(|_| Vp9Error::UnexpectedEndOfStream)? + 1;
            header.render_height = reader.read_bits(16).map_err(|_| Vp9Error::UnexpectedEndOfStream)? + 1;
        }
        Ok(())
    }

    fn parse_frame_size_with_refs(reader: &mut BitReader, _header: &mut FrameHeader) -> Result<bool> {
        for _i in 0..3 {
            let found_ref = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
            if found_ref {
                // Size comes from reference frame
                // In a real implementation, we'd look up the reference frame dimensions
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn parse_loop_filter(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        header.loop_filter.level = reader.read_bits(6).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;
        header.loop_filter.sharpness = reader.read_bits(3).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;

        header.loop_filter.delta_enabled = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
        if header.loop_filter.delta_enabled {
            header.loop_filter.delta_update = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
            if header.loop_filter.delta_update {
                for i in 0..4 {
                    if reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? {
                        let delta = reader.read_bits(6).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as i8;
                        let sign = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
                        header.loop_filter.ref_deltas[i] = if sign { -delta } else { delta };
                    }
                }
                for i in 0..2 {
                    if reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? {
                        let delta = reader.read_bits(6).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as i8;
                        let sign = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
                        header.loop_filter.mode_deltas[i] = if sign { -delta } else { delta };
                    }
                }
            }
        }

        Ok(())
    }

    fn parse_quantization(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        header.quant.base_q_idx = reader.read_bits(8).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;

        header.quant.delta_q_y_dc = Self::read_delta_q(reader)?;
        header.quant.delta_q_uv_dc = Self::read_delta_q(reader)?;
        header.quant.delta_q_uv_ac = Self::read_delta_q(reader)?;

        header.quant.lossless = header.quant.base_q_idx == 0
            && header.quant.delta_q_y_dc == 0
            && header.quant.delta_q_uv_dc == 0
            && header.quant.delta_q_uv_ac == 0;

        Ok(())
    }

    fn read_delta_q(reader: &mut BitReader) -> Result<i8> {
        if reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? {
            let value = reader.read_bits(4).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as i8;
            let sign = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
            Ok(if sign { -value } else { value })
        } else {
            Ok(0)
        }
    }

    fn parse_segmentation(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        header.segmentation.enabled = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;

        if !header.segmentation.enabled {
            return Ok(());
        }

        header.segmentation.update_map = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
        if header.segmentation.update_map {
            for i in 0..7 {
                if reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? {
                    header.segmentation.tree_probs[i] = reader.read_bits(8).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;
                } else {
                    header.segmentation.tree_probs[i] = 255;
                }
            }

            header.segmentation.temporal_update = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
            if header.segmentation.temporal_update {
                for i in 0..3 {
                    if reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? {
                        header.segmentation.pred_probs[i] = reader.read_bits(8).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8;
                    } else {
                        header.segmentation.pred_probs[i] = 255;
                    }
                }
            }
        }

        header.segmentation.update_data = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
        if header.segmentation.update_data {
            header.segmentation.abs_or_delta = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;

            // Feature bits per segment
            const SEG_LVL_MAX: usize = 4;
            const SEGMENTATION_FEATURE_BITS: [u8; SEG_LVL_MAX] = [8, 6, 2, 0];
            const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, false, false];

            for seg in 0..8 {
                for feature in 0..SEG_LVL_MAX {
                    header.segmentation.feature_enabled[seg][feature] = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
                    if header.segmentation.feature_enabled[seg][feature] {
                        let bits = SEGMENTATION_FEATURE_BITS[feature];
                        if bits > 0 {
                            let value = reader.read_bits(bits).map_err(|_| Vp9Error::UnexpectedEndOfStream)? as i16;
                            if SEGMENTATION_FEATURE_SIGNED[feature] {
                                let sign = reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)?;
                                header.segmentation.features[seg][feature] = if sign { -value } else { value };
                            } else {
                                header.segmentation.features[seg][feature] = value;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn parse_tile_info(reader: &mut BitReader, header: &mut FrameHeader) -> Result<()> {
        let mi_cols = (header.width + 7) >> 3;
        let sb64_cols = (mi_cols + 7) >> 3;

        let min_log2 = Self::calc_min_log2_tile_cols(sb64_cols);
        let max_log2 = Self::calc_max_log2_tile_cols(sb64_cols);

        header.tile_info.tile_cols_log2 = min_log2;
        while header.tile_info.tile_cols_log2 < max_log2 {
            if reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? {
                header.tile_info.tile_cols_log2 += 1;
            } else {
                break;
            }
        }

        header.tile_info.tile_rows_log2 = if reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? {
            1 + reader.read_bit().map_err(|_| Vp9Error::UnexpectedEndOfStream)? as u8
        } else {
            0
        };

        header.tile_info.tile_cols = 1 << header.tile_info.tile_cols_log2;
        header.tile_info.tile_rows = 1 << header.tile_info.tile_rows_log2;

        Ok(())
    }

    fn calc_min_log2_tile_cols(sb64_cols: u32) -> u8 {
        let mut min_log2 = 0u8;
        while (sb64_cols >> (min_log2 + 1)) >= 4 {
            min_log2 += 1;
        }
        min_log2
    }

    fn calc_max_log2_tile_cols(sb64_cols: u32) -> u8 {
        let mut max_log2 = 1u8;
        while (sb64_cols >> max_log2) >= 1 {
            max_log2 += 1;
        }
        max_log2.saturating_sub(1)
    }

    /// Parse the compressed header using the bool decoder.
    pub fn parse_compressed_header(&self, data: &[u8], _prob_ctx: &mut ProbabilityContext) -> Result<()> {
        if data.len() < self.header_size as usize {
            return Err(Vp9Error::UnexpectedEndOfStream);
        }

        let mut decoder = BoolDecoder::new(&data[..self.header_size as usize])?;

        // Transform mode
        if !self.quant.lossless {
            let tx_mode = decoder.read_literal(2)?;
            if tx_mode == 3 {
                let _tx_mode_select = decoder.read_literal(1)?;
            }
        }

        // Coefficient probability updates
        if self.frame_type == FrameType::Inter {
            // Reference mode
            let _comp_mode = if self.ref_frame_sign_bias[1] != self.ref_frame_sign_bias[2] {
                decoder.read_literal(1)?
            } else {
                0
            };
        }

        // Probability updates would happen here
        // This is a simplified implementation

        Ok(())
    }

    /// Check if this is a keyframe.
    pub fn is_keyframe(&self) -> bool {
        self.frame_type == FrameType::Keyframe
    }

    /// Check if this is an intra-only frame.
    pub fn is_intra_only(&self) -> bool {
        self.frame_type == FrameType::Keyframe || self.intra_only
    }

    /// Get the number of superblocks in width.
    pub fn sb_cols(&self) -> u32 {
        (self.width + 63) >> 6
    }

    /// Get the number of superblocks in height.
    pub fn sb_rows(&self) -> u32 {
        (self.height + 63) >> 6
    }

    /// Get the number of mode info blocks in width.
    pub fn mi_cols(&self) -> u32 {
        (self.width + 7) >> 3
    }

    /// Get the number of mode info blocks in height.
    pub fn mi_rows(&self) -> u32 {
        (self.height + 7) >> 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_conversion() {
        assert_eq!(Profile::try_from(0).unwrap(), Profile::Profile0);
        assert_eq!(Profile::try_from(1).unwrap(), Profile::Profile1);
        assert_eq!(Profile::try_from(2).unwrap(), Profile::Profile2);
        assert_eq!(Profile::try_from(3).unwrap(), Profile::Profile3);
        assert!(Profile::try_from(4).is_err());
    }

    #[test]
    fn test_color_space_conversion() {
        assert_eq!(ColorSpace::try_from(0).unwrap(), ColorSpace::Unknown);
        assert_eq!(ColorSpace::try_from(2).unwrap(), ColorSpace::Bt709);
        assert_eq!(ColorSpace::try_from(7).unwrap(), ColorSpace::Srgb);
    }

    #[test]
    fn test_interp_filter_conversion() {
        assert_eq!(InterpFilter::try_from(0).unwrap(), InterpFilter::EightTapSmooth);
        assert_eq!(InterpFilter::try_from(3).unwrap(), InterpFilter::Bilinear);
        assert!(InterpFilter::try_from(5).is_err());
    }

    #[test]
    fn test_frame_header_defaults() {
        let header = FrameHeader::default();
        assert_eq!(header.profile, Profile::Profile0);
        assert_eq!(header.frame_type, FrameType::Keyframe);
        assert_eq!(header.bit_depth, 8);
        assert_eq!(header.width, 0);
        assert_eq!(header.height, 0);
    }

    #[test]
    fn test_superblock_calculations() {
        let mut header = FrameHeader::default();
        header.width = 1920;
        header.height = 1080;

        assert_eq!(header.sb_cols(), 30); // ceil(1920/64)
        assert_eq!(header.sb_rows(), 17); // ceil(1080/64)
        assert_eq!(header.mi_cols(), 240); // ceil(1920/8)
        assert_eq!(header.mi_rows(), 135); // ceil(1080/8)
    }

    #[test]
    fn test_tile_log2_calculations() {
        // For 1920 width: mi_cols = 240, sb64_cols = 30
        let sb64_cols = 30;
        let min = FrameHeader::calc_min_log2_tile_cols(sb64_cols);
        let max = FrameHeader::calc_max_log2_tile_cols(sb64_cols);
        assert!(min <= max);
    }

    #[test]
    fn test_invalid_frame_marker() {
        // Frame with marker != 2
        let data = [0x40, 0x00]; // marker = 1
        let result = FrameHeader::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_loop_filter_defaults() {
        let lf = LoopFilterParams::default();
        assert_eq!(lf.level, 0);
        assert_eq!(lf.sharpness, 0);
        assert!(!lf.delta_enabled);
    }

    #[test]
    fn test_quant_params_lossless() {
        let quant = QuantParams {
            base_q_idx: 0,
            delta_q_y_dc: 0,
            delta_q_uv_dc: 0,
            delta_q_uv_ac: 0,
            lossless: true,
        };
        assert!(quant.lossless);
    }
}
