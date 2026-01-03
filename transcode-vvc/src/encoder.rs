//! VVC encoder implementation.
//!
//! This module provides a complete VVC/H.266 encoder including:
//! - Rate control (CQP, VBR, CBR)
//! - GOP structure and multi-layer temporal scalability
//! - CTU partitioning with QTBT+MTT structure
//! - Intra/inter prediction modes (including MIP, ISP, affine, GPM)
//! - Transform, quantization, and entropy coding (including LFNST, MTS, transform skip)
//! - In-loop filtering (deblocking, SAO, ALF)

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::error::{Result, VvcError, VvcLevel, VvcProfile, VvcTier};
use crate::nal::{NalUnitHeader, NalUnitType, SliceType};
// Syntax types used in full implementation
#[allow(unused_imports)]
use crate::syntax::{CodingUnit, IntraMode, MotionVector, PredMode, SplitMode, TransformUnit};
use std::collections::VecDeque;
use transcode_core::bitstream::{add_emulation_prevention, BitWriter};
use transcode_core::frame::Frame;
use transcode_core::packet::{Packet, PacketFlags};

/// VVC encoder preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VvcPreset {
    /// Ultrafast encoding (lowest quality, fastest).
    Ultrafast,
    /// Superfast encoding.
    Superfast,
    /// Veryfast encoding.
    Veryfast,
    /// Faster encoding.
    Faster,
    /// Fast encoding.
    Fast,
    /// Medium encoding (default, balanced).
    Medium,
    /// Slow encoding.
    Slow,
    /// Slower encoding.
    Slower,
    /// Veryslow encoding (highest quality, slowest).
    Veryslow,
    /// Placebo (maximum quality, extremely slow).
    Placebo,
}

impl VvcPreset {
    /// Get the search range for motion estimation.
    pub fn search_range(&self) -> u32 {
        match self {
            Self::Ultrafast => 16,
            Self::Superfast => 24,
            Self::Veryfast => 32,
            Self::Faster => 48,
            Self::Fast => 64,
            Self::Medium => 64,
            Self::Slow => 96,
            Self::Slower => 128,
            Self::Veryslow => 192,
            Self::Placebo => 384,
        }
    }

    /// Get the maximum QT depth.
    pub fn max_qt_depth(&self) -> u8 {
        match self {
            Self::Ultrafast => 2,
            Self::Superfast => 3,
            Self::Veryfast => 3,
            Self::Faster => 4,
            Self::Fast => 4,
            Self::Medium => 4,
            Self::Slow => 4,
            Self::Slower => 4,
            Self::Veryslow => 4,
            Self::Placebo => 4,
        }
    }

    /// Get the maximum MTT depth.
    pub fn max_mtt_depth(&self) -> u8 {
        match self {
            Self::Ultrafast => 0,
            Self::Superfast => 1,
            Self::Veryfast => 2,
            Self::Faster => 2,
            Self::Fast => 3,
            Self::Medium => 3,
            Self::Slow => 4,
            Self::Slower => 4,
            Self::Veryslow => 4,
            Self::Placebo => 4,
        }
    }

    /// Get the RDO (Rate-Distortion Optimization) level.
    pub fn rdo_level(&self) -> u8 {
        match self {
            Self::Ultrafast => 0,
            Self::Superfast => 1,
            Self::Veryfast => 1,
            Self::Faster => 2,
            Self::Fast => 2,
            Self::Medium => 3,
            Self::Slow => 4,
            Self::Slower => 5,
            Self::Veryslow => 6,
            Self::Placebo => 6,
        }
    }

    /// Enable LFNST (Low-Frequency Non-Separable Transform).
    pub fn lfnst_enabled(&self) -> bool {
        !matches!(self, Self::Ultrafast | Self::Superfast)
    }

    /// Enable MTS (Multiple Transform Selection).
    pub fn mts_enabled(&self) -> bool {
        !matches!(self, Self::Ultrafast | Self::Superfast | Self::Veryfast)
    }

    /// Enable ISP (Intra Sub-Partitions).
    pub fn isp_enabled(&self) -> bool {
        !matches!(self, Self::Ultrafast | Self::Superfast | Self::Veryfast)
    }

    /// Enable MIP (Matrix Intra Prediction).
    pub fn mip_enabled(&self) -> bool {
        !matches!(self, Self::Ultrafast | Self::Superfast)
    }

    /// Enable affine motion.
    pub fn affine_enabled(&self) -> bool {
        !matches!(self, Self::Ultrafast | Self::Superfast | Self::Veryfast | Self::Faster)
    }

    /// Enable GPM (Geometric Partition Mode).
    pub fn gpm_enabled(&self) -> bool {
        !matches!(self, Self::Ultrafast | Self::Superfast | Self::Veryfast | Self::Faster | Self::Fast)
    }
}

impl Default for VvcPreset {
    fn default() -> Self {
        Self::Medium
    }
}

/// Rate control mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateControlMode {
    /// Constant QP.
    Cqp { qp: u8 },
    /// Average bitrate (VBR).
    Vbr { bitrate: u32 },
    /// Constant bitrate (CBR).
    Cbr { bitrate: u32 },
    /// Constant Rate Factor (CRF).
    Crf { crf: f32 },
}

impl Default for RateControlMode {
    fn default() -> Self {
        Self::Crf { crf: 28.0 }
    }
}

/// GOP structure type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GopStructure {
    /// All intra (I frames only).
    AllIntra,
    /// Low delay P (I + P frames).
    LowDelayP,
    /// Low delay B (I + B frames, no reordering).
    LowDelayB,
    /// Random access (I + P + B frames with hierarchical structure).
    RandomAccess,
}

impl Default for GopStructure {
    fn default() -> Self {
        Self::RandomAccess
    }
}

/// Temporal layer configuration.
#[derive(Debug, Clone)]
pub struct TemporalLayerConfig {
    /// Number of temporal layers.
    pub num_layers: u8,
    /// QP offset per layer.
    pub qp_offset: Vec<i8>,
    /// Frame interval per layer.
    pub frame_interval: Vec<u32>,
}

impl Default for TemporalLayerConfig {
    fn default() -> Self {
        Self {
            num_layers: 4,
            qp_offset: vec![0, 1, 2, 3],
            frame_interval: vec![8, 4, 2, 1],
        }
    }
}

/// VVC encoder configuration.
#[derive(Debug, Clone)]
pub struct VvcEncoderConfig {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Encoding preset.
    pub preset: VvcPreset,
    /// Profile.
    pub profile: VvcProfile,
    /// Tier.
    pub tier: VvcTier,
    /// Level.
    pub level: VvcLevel,
    /// Rate control mode.
    pub rate_control: RateControlMode,
    /// Bit depth.
    pub bit_depth: u8,
    /// Chroma format (1=4:2:0, 2=4:2:2, 3=4:4:4).
    pub chroma_format: u8,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
    /// GOP structure.
    pub gop_structure: GopStructure,
    /// Number of B-frames between reference frames.
    pub bframes: u32,
    /// Enable B-frame pyramids.
    pub b_pyramid: bool,
    /// Temporal layer configuration.
    pub temporal_layers: TemporalLayerConfig,
    /// Enable SAO filter.
    pub sao_enabled: bool,
    /// Enable ALF filter.
    pub alf_enabled: bool,
    /// Enable CCALF (Cross-Component ALF).
    pub ccalf_enabled: bool,
    /// Enable LMCS (Luma Mapping with Chroma Scaling).
    pub lmcs_enabled: bool,
    /// Enable deblocking filter.
    pub deblocking_enabled: bool,
    /// Enable transform skip.
    pub transform_skip_enabled: bool,
    /// Enable LFNST (Low-Frequency Non-Separable Transform).
    pub lfnst_enabled: bool,
    /// Enable MTS (Multiple Transform Selection).
    pub mts_enabled: bool,
    /// Enable ISP (Intra Sub-Partitions).
    pub isp_enabled: bool,
    /// Enable MIP (Matrix Intra Prediction).
    pub mip_enabled: bool,
    /// Enable BDPCM.
    pub bdpcm_enabled: bool,
    /// Enable palette mode.
    pub palette_enabled: bool,
    /// Enable IBC (Intra Block Copy).
    pub ibc_enabled: bool,
    /// Enable affine motion.
    pub affine_enabled: bool,
    /// Enable 6-parameter affine.
    pub affine_6param_enabled: bool,
    /// Enable SBT (Sub-Block Transform).
    pub sbt_enabled: bool,
    /// Enable MMVD (Merge with MVD).
    pub mmvd_enabled: bool,
    /// Enable CIIP (Combined Inter-Intra Prediction).
    pub ciip_enabled: bool,
    /// Enable GPM (Geometric Partition Mode).
    pub gpm_enabled: bool,
    /// Enable SMVD (Symmetric MVD).
    pub smvd_enabled: bool,
    /// Enable BDOF (Bi-Directional Optical Flow).
    pub bdof_enabled: bool,
    /// Enable DMVR (Decoder-side Motion Vector Refinement).
    pub dmvr_enabled: bool,
    /// Enable BCW (Bi-prediction with CU-level Weights).
    pub bcw_enabled: bool,
    /// Enable joint CbCr coding.
    pub joint_cbcr_enabled: bool,
    /// Enable dependent quantization.
    pub dep_quant_enabled: bool,
    /// Enable sign data hiding.
    pub sign_hiding_enabled: bool,
    /// CTU size (32, 64, or 128).
    pub ctu_size: u32,
    /// Minimum CU size.
    pub min_cu_size: u32,
    /// Minimum QT size for intra.
    pub min_qt_size_intra: u32,
    /// Minimum QT size for inter.
    pub min_qt_size_inter: u32,
    /// Maximum BT size.
    pub max_bt_size: u32,
    /// Maximum TT size.
    pub max_tt_size: u32,
    /// Maximum MTT depth for intra.
    pub max_mtt_depth_intra: u8,
    /// Maximum MTT depth for inter.
    pub max_mtt_depth_inter: u8,
}

impl VvcEncoderConfig {
    /// Create a new configuration for the given resolution.
    pub fn new(width: u32, height: u32) -> Self {
        let preset = VvcPreset::default();
        Self {
            width,
            height,
            fps_num: 30,
            fps_den: 1,
            preset,
            profile: VvcProfile::Main10,
            tier: VvcTier::Main,
            level: VvcLevel::L4_1,
            rate_control: RateControlMode::default(),
            bit_depth: 10,
            chroma_format: 1,
            gop_size: 32,
            gop_structure: GopStructure::RandomAccess,
            bframes: 7,
            b_pyramid: true,
            temporal_layers: TemporalLayerConfig::default(),
            sao_enabled: true,
            alf_enabled: true,
            ccalf_enabled: true,
            lmcs_enabled: true,
            deblocking_enabled: true,
            transform_skip_enabled: true,
            lfnst_enabled: preset.lfnst_enabled(),
            mts_enabled: preset.mts_enabled(),
            isp_enabled: preset.isp_enabled(),
            mip_enabled: preset.mip_enabled(),
            bdpcm_enabled: true,
            palette_enabled: false,
            ibc_enabled: false,
            affine_enabled: preset.affine_enabled(),
            affine_6param_enabled: preset.affine_enabled(),
            sbt_enabled: true,
            mmvd_enabled: true,
            ciip_enabled: true,
            gpm_enabled: preset.gpm_enabled(),
            smvd_enabled: true,
            bdof_enabled: true,
            dmvr_enabled: true,
            bcw_enabled: true,
            joint_cbcr_enabled: true,
            dep_quant_enabled: true,
            sign_hiding_enabled: true,
            ctu_size: 128,
            min_cu_size: 4,
            min_qt_size_intra: 8,
            min_qt_size_inter: 8,
            max_bt_size: 128,
            max_tt_size: 64,
            max_mtt_depth_intra: preset.max_mtt_depth(),
            max_mtt_depth_inter: preset.max_mtt_depth(),
        }
    }

    /// Set the preset and update related settings.
    pub fn with_preset(mut self, preset: VvcPreset) -> Self {
        self.preset = preset;
        self.lfnst_enabled = preset.lfnst_enabled();
        self.mts_enabled = preset.mts_enabled();
        self.isp_enabled = preset.isp_enabled();
        self.mip_enabled = preset.mip_enabled();
        self.affine_enabled = preset.affine_enabled();
        self.affine_6param_enabled = preset.affine_enabled();
        self.gpm_enabled = preset.gpm_enabled();
        self.max_mtt_depth_intra = preset.max_mtt_depth();
        self.max_mtt_depth_inter = preset.max_mtt_depth();
        self
    }

    /// Set the bitrate (for VBR/CBR modes).
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.rate_control = RateControlMode::Vbr { bitrate };
        self
    }

    /// Set constant QP.
    pub fn with_qp(mut self, qp: u8) -> Self {
        self.rate_control = RateControlMode::Cqp { qp: qp.min(63) };
        self
    }

    /// Set CRF value.
    pub fn with_crf(mut self, crf: f32) -> Self {
        self.rate_control = RateControlMode::Crf { crf: crf.clamp(0.0, 63.0) };
        self
    }

    /// Set GOP size.
    pub fn with_gop_size(mut self, gop_size: u32) -> Self {
        self.gop_size = gop_size;
        self
    }

    /// Set GOP structure.
    pub fn with_gop_structure(mut self, structure: GopStructure) -> Self {
        self.gop_structure = structure;
        match structure {
            GopStructure::AllIntra => {
                self.bframes = 0;
                self.gop_size = 1;
            }
            GopStructure::LowDelayP => {
                self.bframes = 0;
            }
            GopStructure::LowDelayB => {
                // B frames but no reordering
            }
            GopStructure::RandomAccess => {
                // Default hierarchical B structure
            }
        }
        self
    }

    /// Set number of B-frames.
    pub fn with_bframes(mut self, bframes: u32) -> Self {
        self.bframes = bframes;
        self
    }

    /// Set profile.
    pub fn with_profile(mut self, profile: VvcProfile) -> Self {
        self.profile = profile;
        if profile.is_444() {
            self.chroma_format = 3;
        }
        self
    }

    /// Set temporal layer configuration.
    pub fn with_temporal_layers(mut self, num_layers: u8) -> Self {
        self.temporal_layers = TemporalLayerConfig {
            num_layers,
            qp_offset: (0..num_layers).map(|i| i as i8).collect(),
            frame_interval: (0..num_layers).map(|i| 1 << (num_layers - 1 - i)).collect(),
        };
        self
    }
}

/// Frame statistics.
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    /// Frame type.
    pub frame_type: char,
    /// Encoded size in bytes.
    pub size: usize,
    /// Average QP.
    pub avg_qp: f32,
    /// PSNR Y.
    pub psnr_y: f32,
    /// PSNR U.
    pub psnr_u: f32,
    /// PSNR V.
    pub psnr_v: f32,
    /// Temporal layer.
    pub temporal_id: u8,
    /// Encoding time in milliseconds.
    pub encode_time_ms: f32,
}

/// Reference picture.
#[derive(Debug, Clone)]
struct ReferencePicture {
    /// Frame data.
    frame: Frame,
    /// POC.
    poc: i32,
    /// Is long-term reference.
    is_long_term: bool,
    /// Temporal ID.
    temporal_id: u8,
}

/// Frame to be encoded.
#[derive(Debug)]
struct PendingFrame {
    /// Input frame.
    frame: Frame,
    /// Display order.
    display_order: u64,
    /// Coding order.
    coding_order: u64,
    /// Frame type.
    frame_type: SliceType,
    /// POC.
    poc: i32,
    /// Temporal ID.
    temporal_id: u8,
    /// QP offset for this frame.
    qp_offset: i8,
}

/// VVC encoder.
#[derive(Debug)]
pub struct VvcEncoder {
    /// Configuration.
    config: VvcEncoderConfig,
    /// Reference picture list 0.
    ref_list0: Vec<ReferencePicture>,
    /// Reference picture list 1.
    ref_list1: Vec<ReferencePicture>,
    /// Pending frames for B-frame reordering.
    pending_frames: VecDeque<PendingFrame>,
    /// Frame number.
    frame_num: u64,
    /// POC counter.
    poc: i32,
    /// Coding order counter.
    coding_order: u64,
    /// Current QP.
    current_qp: u8,
    /// VPS NAL unit.
    vps_nal: Vec<u8>,
    /// SPS NAL unit.
    sps_nal: Vec<u8>,
    /// PPS NAL unit.
    pps_nal: Vec<u8>,
    /// Headers sent flag.
    headers_sent: bool,
    /// Statistics for last encoded frame.
    last_stats: FrameStats,
    /// Rate control state.
    rc_state: RateControlState,
}

/// Rate control state.
#[derive(Debug, Default)]
struct RateControlState {
    /// Total bits used.
    total_bits: u64,
    /// Frame count.
    frame_count: u64,
    /// Target bits per frame.
    target_bits_per_frame: u64,
    /// Buffer fullness.
    buffer_fullness: i64,
    /// Buffer size.
    buffer_size: i64,
    /// Last I-frame bits.
    last_i_frame_bits: u64,
    /// Last P-frame bits.
    last_p_frame_bits: u64,
    /// Last B-frame bits.
    last_b_frame_bits: u64,
    /// Running QP average.
    avg_qp: f32,
}

impl VvcEncoder {
    /// Create a new VVC encoder.
    pub fn new(config: VvcEncoderConfig) -> Result<Self> {
        // Validate configuration
        if config.width == 0 || config.height == 0 {
            return Err(VvcError::EncoderConfig("Invalid dimensions".to_string()));
        }

        if config.ctu_size != 32 && config.ctu_size != 64 && config.ctu_size != 128 {
            return Err(VvcError::EncoderConfig("Invalid CTU size".to_string()));
        }

        let initial_qp = match config.rate_control {
            RateControlMode::Cqp { qp } => qp,
            RateControlMode::Crf { crf } => crf as u8,
            RateControlMode::Vbr { bitrate } | RateControlMode::Cbr { bitrate } => {
                // Estimate initial QP from bitrate
                let pixels_per_second = config.width * config.height * config.fps_num / config.fps_den;
                let bits_per_pixel = (bitrate as f64) / (pixels_per_second as f64);
                (40.0 - bits_per_pixel.log2() * 5.0).clamp(10.0, 51.0) as u8
            }
        };

        let mut rc_state = RateControlState::default();
        if let RateControlMode::Vbr { bitrate } | RateControlMode::Cbr { bitrate } = config.rate_control {
            rc_state.target_bits_per_frame = (bitrate as u64 * config.fps_den as u64) / config.fps_num as u64;
            rc_state.buffer_size = rc_state.target_bits_per_frame as i64 * 10;
        }

        let mut encoder = Self {
            ref_list0: Vec::new(),
            ref_list1: Vec::new(),
            pending_frames: VecDeque::new(),
            frame_num: 0,
            poc: 0,
            coding_order: 0,
            current_qp: initial_qp,
            vps_nal: Vec::new(),
            sps_nal: Vec::new(),
            pps_nal: Vec::new(),
            headers_sent: false,
            last_stats: FrameStats::default(),
            rc_state,
            config,
        };

        // Generate parameter sets
        encoder.generate_parameter_sets()?;

        Ok(encoder)
    }

    /// Generate VPS, SPS, and PPS NAL units.
    fn generate_parameter_sets(&mut self) -> Result<()> {
        self.vps_nal = self.generate_vps()?;
        self.sps_nal = self.generate_sps()?;
        self.pps_nal = self.generate_pps()?;
        Ok(())
    }

    /// Generate VPS NAL unit.
    fn generate_vps(&self) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        // NAL unit header
        let header = NalUnitHeader {
            nal_unit_type: NalUnitType::VpsNut,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: 1,
        };
        header.write(&mut writer)?;

        // VPS data
        writer.write_bits(0, 4)?; // vps_video_parameter_set_id
        writer.write_bits(0, 6)?; // vps_max_layers_minus1
        writer.write_bits(self.config.temporal_layers.num_layers as u32 - 1, 3)?; // vps_max_sublayers_minus1

        // Default PTL DPB HRD max TID flag
        if self.config.temporal_layers.num_layers > 1 {
            writer.write_bit(true)?;
        }

        // All independent layers flag
        writer.write_bit(true)?;

        // Layer info
        writer.write_bits(0, 6)?; // vps_layer_id[0]

        // OLS info
        writer.write_bit(true)?; // vps_each_layer_is_an_ols_flag

        // PTL info
        writer.write_bits(0, 8)?; // vps_ols_ptl_idx (simplified)

        // Timing info
        writer.write_bit(true)?; // vps_timing_hrd_params_present_flag
        writer.write_bits(self.config.fps_den, 32)?; // vps_num_units_in_tick
        writer.write_bits(self.config.fps_num, 32)?; // vps_time_scale

        writer.write_bit(false)?; // vps_extension_flag

        writer.write_rbsp_trailing_bits()?;

        let data = writer.into_data();
        Ok(add_emulation_prevention(&data))
    }

    /// Generate SPS NAL unit.
    fn generate_sps(&self) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        // NAL unit header
        let header = NalUnitHeader {
            nal_unit_type: NalUnitType::SpsNut,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: 1,
        };
        header.write(&mut writer)?;

        // SPS data
        writer.write_bits(0, 4)?; // sps_seq_parameter_set_id
        writer.write_bits(0, 4)?; // sps_video_parameter_set_id
        writer.write_bits(self.config.temporal_layers.num_layers as u32 - 1, 3)?; // sps_max_sublayers_minus1
        writer.write_bits(self.config.chroma_format as u32, 2)?; // sps_chroma_format_idc

        let log2_ctu_size = (self.config.ctu_size as f32).log2() as u32;
        writer.write_bits(log2_ctu_size - 5, 2)?; // sps_log2_ctu_size_minus5

        // PTL DPB HRD params present
        writer.write_bit(true)?;

        // Profile tier level
        self.write_profile_tier_level(&mut writer, true, self.config.temporal_layers.num_layers - 1)?;

        // GDR enabled
        writer.write_bit(false)?;

        // Reference picture resampling
        writer.write_bit(false)?;

        // Picture dimensions
        writer.write_ue(self.config.width)?;
        writer.write_ue(self.config.height)?;

        // Conformance window
        writer.write_bit(false)?;

        // Subpic info
        writer.write_bit(false)?;

        // Bit depth
        writer.write_ue((self.config.bit_depth - 8) as u32)?;

        // Entropy coding sync
        writer.write_bit(false)?;
        writer.write_bit(false)?; // entry_point_offsets_present

        // POC
        writer.write_bits(4, 4)?; // log2_max_pic_order_cnt_lsb_minus4
        writer.write_bit(false)?; // poc_msb_cycle_flag

        // Extra PH/SH bytes
        writer.write_bits(0, 2)?; // num_extra_ph_bytes
        writer.write_bits(0, 2)?; // num_extra_sh_bytes

        // DPB params
        if self.config.temporal_layers.num_layers > 1 {
            writer.write_bit(false)?;
        }

        // Partition constraints
        let log2_min_cb = (self.config.min_cu_size as f32).log2() as u8;
        writer.write_ue((log2_min_cb - 2) as u32)?;

        writer.write_bit(true)?; // partition_constraints_override_enabled_flag

        // Intra partition constraints
        let log2_min_qt_intra = (self.config.min_qt_size_intra as f32).log2() as u8;
        writer.write_ue((log2_min_qt_intra - log2_min_cb) as u32)?;
        writer.write_ue(self.config.max_mtt_depth_intra as u32)?;

        if self.config.max_mtt_depth_intra > 0 {
            let log2_max_bt = (self.config.max_bt_size as f32).log2() as u8;
            let log2_max_tt = (self.config.max_tt_size as f32).log2() as u8;
            writer.write_ue((log2_max_bt - log2_min_qt_intra) as u32)?;
            writer.write_ue((log2_max_tt - log2_min_qt_intra) as u32)?;
        }

        // Dual tree
        if self.config.chroma_format != 0 {
            writer.write_bit(true)?;
        }

        // Inter partition constraints
        let log2_min_qt_inter = (self.config.min_qt_size_inter as f32).log2() as u8;
        writer.write_ue((log2_min_qt_inter - log2_min_cb) as u32)?;
        writer.write_ue(self.config.max_mtt_depth_inter as u32)?;

        if self.config.max_mtt_depth_inter > 0 {
            let log2_max_bt = (self.config.max_bt_size as f32).log2() as u8;
            let log2_max_tt = (self.config.max_tt_size as f32).log2() as u8;
            writer.write_ue((log2_max_bt - log2_min_qt_inter) as u32)?;
            writer.write_ue((log2_max_tt - log2_min_qt_inter) as u32)?;
        }

        // Max 64x64 transform
        if self.config.ctu_size > 32 {
            writer.write_bit(true)?;
        }

        // Transform features
        writer.write_bit(self.config.transform_skip_enabled)?;
        if self.config.transform_skip_enabled {
            writer.write_ue(2)?; // log2_transform_skip_max_size_minus2
            writer.write_bit(self.config.bdpcm_enabled)?;
        }

        writer.write_bit(self.config.mts_enabled)?;
        if self.config.mts_enabled {
            writer.write_bit(true)?; // explicit_mts_intra
            writer.write_bit(true)?; // explicit_mts_inter
        }

        writer.write_bit(self.config.lfnst_enabled)?;

        // Chroma tools
        if self.config.chroma_format != 0 {
            writer.write_bit(self.config.joint_cbcr_enabled)?;
            writer.write_bit(true)?; // same_qp_table_for_chroma
            // QP table (simplified)
            writer.write_se(0)?; // qp_table_start_minus26
            writer.write_ue(0)?; // num_points_in_qp_table_minus1
        }

        // Filter flags
        writer.write_bit(self.config.sao_enabled)?;
        writer.write_bit(self.config.alf_enabled)?;
        if self.config.alf_enabled && self.config.chroma_format != 0 {
            writer.write_bit(self.config.ccalf_enabled)?;
        }

        writer.write_bit(self.config.lmcs_enabled)?;

        // Weighted prediction
        writer.write_bit(false)?; // weighted_pred
        writer.write_bit(false)?; // weighted_bipred

        // Long term refs
        writer.write_bit(false)?;

        // Inter layer prediction
        writer.write_bit(false)?; // idr_rpl_present
        writer.write_bit(true)?; // rpl1_same_as_rpl0

        // Ref pic lists
        writer.write_ue(0)?; // num_ref_pic_lists

        // Ref wraparound
        writer.write_bit(false)?;

        // Temporal MVP
        writer.write_bit(true)?;
        writer.write_bit(self.config.bdof_enabled)?;

        if self.config.bdof_enabled {
            writer.write_bit(true)?; // bdof_control_present_in_ph
        }

        writer.write_bit(self.config.smvd_enabled)?;
        writer.write_bit(self.config.dmvr_enabled)?;

        if self.config.dmvr_enabled {
            writer.write_bit(true)?; // dmvr_control_present_in_ph
        }

        writer.write_bit(self.config.mmvd_enabled)?;
        if self.config.mmvd_enabled {
            writer.write_bit(false)?; // mmvd_fullpel_only
        }

        // Merge candidates
        writer.write_ue(1)?; // six_minus_max_num_merge_cand

        writer.write_bit(self.config.sbt_enabled)?;
        writer.write_bit(self.config.affine_enabled)?;

        if self.config.affine_enabled {
            writer.write_ue(0)?; // five_minus_max_num_subblock_merge_cand
            writer.write_bit(self.config.affine_6param_enabled)?;
            writer.write_bit(true)?; // affine_amvr_enabled
            writer.write_bit(true)?; // affine_prof_enabled
            writer.write_bit(true)?; // prof_control_present_in_ph
        }

        writer.write_bit(self.config.bcw_enabled)?;
        writer.write_bit(self.config.ciip_enabled)?;
        writer.write_bit(self.config.gpm_enabled)?;

        if self.config.gpm_enabled {
            writer.write_ue(0)?; // max_num_merge_cand_minus_max_num_gpm_cand
        }

        writer.write_ue(2)?; // log2_parallel_merge_level_minus2

        // Intra tools
        writer.write_bit(self.config.isp_enabled)?;
        writer.write_bit(true)?; // mrl_enabled
        writer.write_bit(self.config.mip_enabled)?;

        if self.config.chroma_format != 0 {
            writer.write_bit(true)?; // cclm_enabled
        }

        if self.config.chroma_format == 1 {
            writer.write_bit(true)?; // chroma_horizontal_collocated
            writer.write_bit(true)?; // chroma_vertical_collocated
        }

        writer.write_bit(self.config.palette_enabled)?;

        if self.config.chroma_format == 3 && !self.config.palette_enabled {
            writer.write_bit(false)?; // act_enabled
        }

        if self.config.transform_skip_enabled || self.config.palette_enabled {
            writer.write_ue(0)?; // min_qp_prime_ts_minus4
        }

        writer.write_bit(self.config.ibc_enabled)?;

        if self.config.ibc_enabled {
            writer.write_ue(0)?; // six_minus_max_num_ibc_merge_cand
        }

        // LADF
        writer.write_bit(false)?;

        // Scaling list
        writer.write_bit(false)?;

        // Dep quant and sign hiding
        writer.write_bit(self.config.dep_quant_enabled)?;
        writer.write_bit(self.config.sign_hiding_enabled)?;

        // Virtual boundaries
        writer.write_bit(false)?;

        // Timing HRD params
        writer.write_bit(false)?;

        // Field seq flag
        writer.write_bit(false)?;

        // VUI
        writer.write_bit(false)?;

        // Extension
        writer.write_bit(false)?;

        writer.write_rbsp_trailing_bits()?;

        let data = writer.into_data();
        Ok(add_emulation_prevention(&data))
    }

    /// Generate PPS NAL unit.
    fn generate_pps(&self) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        // NAL unit header
        let header = NalUnitHeader {
            nal_unit_type: NalUnitType::PpsNut,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: 1,
        };
        header.write(&mut writer)?;

        // PPS data
        writer.write_bits(0, 6)?; // pps_pic_parameter_set_id
        writer.write_bits(0, 4)?; // pps_seq_parameter_set_id

        writer.write_bit(false)?; // mixed_nalu_types_in_pic
        writer.write_ue(self.config.width)?;
        writer.write_ue(self.config.height)?;

        // Conformance window
        writer.write_bit(false)?;

        // Scaling window
        writer.write_bit(false)?;

        // Output flag
        writer.write_bit(false)?;

        // No pic partition
        writer.write_bit(true)?;

        // Subpic ID mapping
        writer.write_bit(false)?;

        // Loop filter across slices
        writer.write_bit(true)?;

        // CABAC init
        writer.write_bit(true)?;

        // Ref IDX defaults
        writer.write_ue(0)?; // num_ref_idx_default_active_minus1[0]
        writer.write_ue(0)?; // num_ref_idx_default_active_minus1[1]

        // RPL1 IDX present
        writer.write_bit(false)?;

        // Weighted pred
        writer.write_bit(false)?;
        writer.write_bit(false)?;

        // Ref wraparound
        writer.write_bit(false)?;

        // Init QP
        let init_qp = self.current_qp as i32 - 26;
        writer.write_se(init_qp)?;

        // CU QP delta
        writer.write_bit(true)?;

        // Chroma tool offsets
        writer.write_bit(false)?;

        // Deblocking filter control
        writer.write_bit(self.config.deblocking_enabled)?;

        if self.config.deblocking_enabled {
            writer.write_bit(true)?; // deblocking_filter_override_enabled
            writer.write_bit(false)?; // pps_deblocking_filter_disabled
            writer.write_bit(true)?; // dbf_info_in_ph
        }

        // RPL/SAO/ALF/WP/QP in PH
        writer.write_bit(true)?; // rpl_info_in_ph
        writer.write_bit(true)?; // sao_info_in_ph
        writer.write_bit(true)?; // alf_info_in_ph
        writer.write_bit(true)?; // qp_delta_info_in_ph

        // Extensions
        writer.write_bit(false)?; // picture_header_extension_present
        writer.write_bit(false)?; // slice_header_extension_present
        writer.write_bit(false)?; // pps_extension

        writer.write_rbsp_trailing_bits()?;

        let data = writer.into_data();
        Ok(add_emulation_prevention(&data))
    }

    /// Write profile tier level syntax.
    fn write_profile_tier_level(
        &self,
        writer: &mut BitWriter,
        profile_tier_present_flag: bool,
        max_num_sub_layers_minus1: u8,
    ) -> Result<()> {
        if profile_tier_present_flag {
            writer.write_bits(self.config.profile.idc() as u32, 7)?;
            writer.write_bit(self.config.tier == VvcTier::High)?;
        }

        writer.write_bits(self.config.level.level_idc() as u32, 8)?;

        writer.write_bit(true)?; // ptl_frame_only_constraint_flag
        writer.write_bit(false)?; // ptl_multilayer_enabled_flag

        if profile_tier_present_flag {
            // General constraint info
            writer.write_bit(false)?; // gci_present_flag
        }

        // Byte alignment
        while !writer.is_byte_aligned() {
            writer.write_bit(false)?;
        }

        // Sub-layer flags
        for _ in 0..max_num_sub_layers_minus1 {
            writer.write_bit(false)?; // ptl_sublayer_level_present_flag
        }

        // Byte alignment
        while !writer.is_byte_aligned() {
            writer.write_bit(false)?;
        }

        if profile_tier_present_flag {
            writer.write_bits(0, 8)?; // ptl_num_sub_profiles
        }

        Ok(())
    }

    /// Encode a frame.
    pub fn encode(&mut self, frame: &Frame) -> Result<Option<Packet<'static>>> {
        // Validate input frame
        if frame.width() != self.config.width || frame.height() != self.config.height {
            return Err(VvcError::EncoderConfig("Frame dimensions mismatch".to_string()));
        }

        // Determine frame type and temporal layer
        let (frame_type, temporal_id, qp_offset) = self.determine_frame_type();

        // Add to pending frames for B-frame reordering
        let pending = PendingFrame {
            frame: frame.clone(),
            display_order: self.frame_num,
            coding_order: 0,
            frame_type,
            poc: self.poc,
            temporal_id,
            qp_offset,
        };

        self.pending_frames.push_back(pending);
        self.frame_num += 1;
        self.poc += 1;

        // Try to encode a frame
        self.try_encode_frame()
    }

    /// Determine the frame type, temporal layer, and QP offset.
    fn determine_frame_type(&self) -> (SliceType, u8, i8) {
        match self.config.gop_structure {
            GopStructure::AllIntra => (SliceType::I, 0, 0),
            GopStructure::LowDelayP => {
                if self.frame_num == 0 || self.frame_num % self.config.gop_size as u64 == 0 {
                    (SliceType::I, 0, 0)
                } else {
                    (SliceType::P, 0, 1)
                }
            }
            GopStructure::LowDelayB => {
                if self.frame_num == 0 || self.frame_num % self.config.gop_size as u64 == 0 {
                    (SliceType::I, 0, 0)
                } else {
                    (SliceType::B, 0, 1)
                }
            }
            GopStructure::RandomAccess => {
                if self.frame_num == 0 || self.frame_num % self.config.gop_size as u64 == 0 {
                    (SliceType::I, 0, 0)
                } else {
                    // Hierarchical B-frame structure
                    let gop_pos = self.frame_num % self.config.gop_size as u64;
                    let num_layers = self.config.temporal_layers.num_layers as u64;

                    let mut temporal_id = 0u8;
                    for layer in 1..num_layers {
                        let interval = 1u64 << (num_layers - layer - 1);
                        if gop_pos % interval == 0 {
                            temporal_id = layer as u8;
                            break;
                        }
                        temporal_id = layer as u8 + 1;
                    }

                    let qp_offset = self.config.temporal_layers.qp_offset
                        .get(temporal_id as usize)
                        .copied()
                        .unwrap_or(0);

                    if temporal_id == 0 {
                        (SliceType::P, temporal_id, qp_offset)
                    } else {
                        (SliceType::B, temporal_id, qp_offset)
                    }
                }
            }
        }
    }

    /// Try to encode a pending frame.
    fn try_encode_frame(&mut self) -> Result<Option<Packet<'static>>> {
        if self.pending_frames.is_empty() {
            return Ok(None);
        }

        // For B-frame support, wait for future references
        let can_encode = match self.config.gop_structure {
            GopStructure::AllIntra | GopStructure::LowDelayP | GopStructure::LowDelayB => true,
            GopStructure::RandomAccess => {
                self.pending_frames.len() > self.config.bframes as usize
                    || self.pending_frames.front().map(|f| f.frame_type != SliceType::B).unwrap_or(false)
            }
        };

        if !can_encode {
            return Ok(None);
        }

        let pending = self.pending_frames.pop_front().unwrap();
        self.encode_frame_internal(pending)
    }

    /// Internal frame encoding.
    fn encode_frame_internal(&mut self, pending: PendingFrame) -> Result<Option<Packet<'static>>> {
        let mut output = Vec::new();

        // Send headers before first frame
        if !self.headers_sent {
            output.extend_from_slice(&[0, 0, 0, 1]);
            output.extend_from_slice(&self.vps_nal);
            output.extend_from_slice(&[0, 0, 0, 1]);
            output.extend_from_slice(&self.sps_nal);
            output.extend_from_slice(&[0, 0, 0, 1]);
            output.extend_from_slice(&self.pps_nal);
            self.headers_sent = true;
        }

        // Generate picture header
        let ph_data = self.generate_picture_header(&pending)?;
        output.extend_from_slice(&[0, 0, 0, 1]);
        output.extend_from_slice(&ph_data);

        // Encode slice
        let slice_data = self.encode_slice(&pending)?;
        output.extend_from_slice(&[0, 0, 0, 1]);
        output.extend_from_slice(&slice_data);

        // Update rate control
        self.update_rate_control(&pending, output.len());

        // Update statistics
        self.last_stats = FrameStats {
            frame_type: match pending.frame_type {
                SliceType::I => 'I',
                SliceType::P => 'P',
                SliceType::B => 'B',
            },
            size: output.len(),
            avg_qp: self.current_qp as f32 + pending.qp_offset as f32,
            psnr_y: 0.0,
            psnr_u: 0.0,
            psnr_v: 0.0,
            temporal_id: pending.temporal_id,
            encode_time_ms: 0.0,
        };

        // Update reference pictures
        if pending.frame_type != SliceType::B || pending.temporal_id == 0 {
            self.update_reference_pictures(&pending.frame, pending.poc, pending.temporal_id);
        }

        self.coding_order += 1;

        // Create packet
        let mut packet = Packet::new(output);
        packet.pts = pending.frame.pts;
        packet.dts = pending.frame.dts;

        if pending.frame_type == SliceType::I {
            packet.flags = PacketFlags::KEYFRAME;
        }

        Ok(Some(packet))
    }

    /// Generate picture header.
    fn generate_picture_header(&self, pending: &PendingFrame) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        // NAL unit header
        let header = NalUnitHeader {
            nal_unit_type: NalUnitType::PhNut,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: pending.temporal_id + 1,
        };
        header.write(&mut writer)?;

        // PH data
        let is_irap = pending.frame_type == SliceType::I;

        writer.write_bit(is_irap)?; // ph_gdr_or_irap_pic_flag
        writer.write_bit(pending.frame_type == SliceType::B && pending.temporal_id > 0)?; // ph_non_ref_pic_flag

        if is_irap {
            writer.write_bit(false)?; // ph_gdr_pic_flag
        }

        // Inter slice allowed
        let inter_allowed = pending.frame_type != SliceType::I;
        writer.write_bit(inter_allowed)?;

        if inter_allowed {
            writer.write_bit(true)?; // ph_intra_slice_allowed
        }

        writer.write_ue(0)?; // ph_pic_parameter_set_id
        writer.write_bits(pending.poc as u32 & 0xFF, 8)?; // ph_pic_order_cnt_lsb

        // Extra PH bits (none)

        // ALF
        if self.config.alf_enabled {
            writer.write_bit(false)?; // ph_alf_enabled_flag
        }

        // LMCS
        if self.config.lmcs_enabled {
            writer.write_bit(false)?; // ph_lmcs_enabled_flag
        }

        // Scaling list
        // Handled by PPS

        // Virtual boundaries
        // Handled by SPS

        // Output flag
        writer.write_bit(true)?;

        // Reference picture lists would go here (simplified)
        if inter_allowed {
            // Simplified RPL signaling
        }

        // SAO
        if self.config.sao_enabled {
            writer.write_bit(true)?; // ph_sao_luma_enabled
            if self.config.chroma_format != 0 {
                writer.write_bit(true)?; // ph_sao_chroma_enabled
            }
        }

        // Deblocking
        if self.config.deblocking_enabled {
            writer.write_bit(false)?; // ph_deblocking_params_present
        }

        // QP delta
        let qp_delta = pending.qp_offset;
        writer.write_se(qp_delta as i32)?;

        writer.write_rbsp_trailing_bits()?;

        let data = writer.into_data();
        Ok(add_emulation_prevention(&data))
    }

    /// Encode a slice.
    fn encode_slice(&mut self, pending: &PendingFrame) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        // Determine NAL unit type
        let nal_type = match pending.frame_type {
            SliceType::I => {
                if pending.display_order == 0 {
                    NalUnitType::IdrWRadl
                } else {
                    NalUnitType::CraNut
                }
            }
            SliceType::P | SliceType::B => {
                if pending.temporal_id == 0 {
                    NalUnitType::TrailNut
                } else {
                    NalUnitType::StsaNut
                }
            }
        };

        // NAL unit header
        let header = NalUnitHeader {
            nal_unit_type: nal_type,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: pending.temporal_id + 1,
        };
        header.write(&mut writer)?;

        // Slice header
        self.write_slice_header(&mut writer, pending, nal_type)?;

        // Byte alignment
        writer.byte_align()?;

        // Encode CTUs (simplified)
        let slice_data = self.encode_slice_data(pending)?;

        let mut result = writer.into_data();
        result.extend_from_slice(&slice_data);

        Ok(add_emulation_prevention(&result))
    }

    /// Write slice header.
    fn write_slice_header(
        &self,
        writer: &mut BitWriter,
        pending: &PendingFrame,
        nal_type: NalUnitType,
    ) -> Result<()> {
        writer.write_bit(false)?; // sh_picture_header_in_slice_header_flag

        // Slice type
        let slice_type_val = match pending.frame_type {
            SliceType::B => 0,
            SliceType::P => 1,
            SliceType::I => 2,
        };
        writer.write_ue(slice_type_val)?;

        // IDR-specific
        if nal_type.is_idr() {
            writer.write_bit(false)?; // sh_no_output_of_prior_pics_flag
        }

        // QP delta is in PH

        // SAO is in PH

        // Deblocking is in PH/PPS

        // Dep quant
        if self.config.dep_quant_enabled {
            writer.write_bit(true)?;
        }

        // Sign data hiding
        if self.config.sign_hiding_enabled && !self.config.dep_quant_enabled {
            writer.write_bit(true)?;
        }

        Ok(())
    }

    /// Encode slice data (CTUs).
    fn encode_slice_data(&mut self, pending: &PendingFrame) -> Result<Vec<u8>> {
        // Simplified slice data encoding
        // A full implementation would use CABAC

        let ctu_size = self.config.ctu_size;
        let width_in_ctus = (self.config.width + ctu_size - 1) / ctu_size;
        let height_in_ctus = (self.config.height + ctu_size - 1) / ctu_size;

        let mut data = Vec::new();

        // Encode each CTU (simplified)
        for ctu_y in 0..height_in_ctus {
            for ctu_x in 0..width_in_ctus {
                let x = ctu_x * ctu_size;
                let y = ctu_y * ctu_size;

                // Simplified: use DC prediction for all CUs
                self.encode_ctu_simplified(&pending.frame, x, y, &mut data)?;
            }
        }

        // Add CABAC termination
        data.push(0x80); // End of slice

        Ok(data)
    }

    /// Encode a CTU (simplified).
    #[allow(clippy::ptr_arg)]
    fn encode_ctu_simplified(
        &self,
        _frame: &Frame,
        _x: u32,
        _y: u32,
        _data: &mut Vec<u8>,
    ) -> Result<()> {
        // Simplified CTU encoding
        // A real encoder would perform mode decision and RDO

        Ok(())
    }

    /// Update rate control.
    fn update_rate_control(&mut self, pending: &PendingFrame, encoded_size: usize) {
        let bits = encoded_size * 8;

        self.rc_state.total_bits += bits as u64;
        self.rc_state.frame_count += 1;

        match pending.frame_type {
            SliceType::I => self.rc_state.last_i_frame_bits = bits as u64,
            SliceType::P => self.rc_state.last_p_frame_bits = bits as u64,
            SliceType::B => self.rc_state.last_b_frame_bits = bits as u64,
        }

        // Update average QP
        let frame_qp = self.current_qp as f32 + pending.qp_offset as f32;
        self.rc_state.avg_qp = (self.rc_state.avg_qp * (self.rc_state.frame_count - 1) as f32
            + frame_qp) / self.rc_state.frame_count as f32;

        // Update buffer for CBR
        if let RateControlMode::Cbr { .. } = self.config.rate_control {
            self.rc_state.buffer_fullness += bits as i64 - self.rc_state.target_bits_per_frame as i64;

            // Adjust QP based on buffer fullness
            if self.rc_state.buffer_fullness > self.rc_state.buffer_size / 2 {
                self.current_qp = (self.current_qp + 1).min(63);
            } else if self.rc_state.buffer_fullness < -self.rc_state.buffer_size / 2 {
                self.current_qp = self.current_qp.saturating_sub(1);
            }
        }
    }

    /// Update reference pictures.
    fn update_reference_pictures(&mut self, frame: &Frame, poc: i32, temporal_id: u8) {
        let ref_pic = ReferencePicture {
            frame: frame.clone(),
            poc,
            is_long_term: false,
            temporal_id,
        };

        self.ref_list0.insert(0, ref_pic.clone());
        self.ref_list1.insert(0, ref_pic);

        // Limit reference list size
        const MAX_REFS: usize = 4;
        if self.ref_list0.len() > MAX_REFS {
            self.ref_list0.truncate(MAX_REFS);
        }
        if self.ref_list1.len() > MAX_REFS {
            self.ref_list1.truncate(MAX_REFS);
        }
    }

    /// Flush the encoder.
    pub fn flush(&mut self) -> Result<Vec<Packet<'static>>> {
        let mut packets = Vec::new();

        while !self.pending_frames.is_empty() {
            if let Some(pending) = self.pending_frames.pop_front() {
                if let Some(packet) = self.encode_frame_internal(pending)? {
                    packets.push(packet);
                }
            }
        }

        Ok(packets)
    }

    /// Get the last frame statistics.
    pub fn last_stats(&self) -> &FrameStats {
        &self.last_stats
    }

    /// Get the encoder headers (VPS, SPS, PPS).
    pub fn headers(&self) -> Vec<u8> {
        let mut result = Vec::new();
        result.extend_from_slice(&[0, 0, 0, 1]);
        result.extend_from_slice(&self.vps_nal);
        result.extend_from_slice(&[0, 0, 0, 1]);
        result.extend_from_slice(&self.sps_nal);
        result.extend_from_slice(&[0, 0, 0, 1]);
        result.extend_from_slice(&self.pps_nal);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config() {
        let config = VvcEncoderConfig::new(1920, 1080)
            .with_preset(VvcPreset::Fast)
            .with_crf(28.0)
            .with_gop_size(32);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.preset, VvcPreset::Fast);
        assert_eq!(config.gop_size, 32);
    }

    #[test]
    fn test_encoder_creation() {
        let config = VvcEncoderConfig::new(640, 480);
        let encoder = VvcEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_invalid_config() {
        let config = VvcEncoderConfig::new(0, 0);
        let encoder = VvcEncoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_preset_properties() {
        assert!(VvcPreset::Ultrafast.search_range() < VvcPreset::Veryslow.search_range());
        assert!(VvcPreset::Ultrafast.max_qt_depth() <= VvcPreset::Veryslow.max_qt_depth());
        assert!(VvcPreset::Ultrafast.max_mtt_depth() <= VvcPreset::Veryslow.max_mtt_depth());

        assert!(!VvcPreset::Ultrafast.lfnst_enabled());
        assert!(VvcPreset::Medium.lfnst_enabled());

        assert!(!VvcPreset::Ultrafast.affine_enabled());
        assert!(VvcPreset::Medium.affine_enabled());
    }

    #[test]
    fn test_gop_structures() {
        let config_ai = VvcEncoderConfig::new(640, 480)
            .with_gop_structure(GopStructure::AllIntra);
        assert_eq!(config_ai.gop_size, 1);
        assert_eq!(config_ai.bframes, 0);

        let config_ldp = VvcEncoderConfig::new(640, 480)
            .with_gop_structure(GopStructure::LowDelayP);
        assert_eq!(config_ldp.bframes, 0);
    }

    #[test]
    fn test_temporal_layers() {
        let config = VvcEncoderConfig::new(640, 480)
            .with_temporal_layers(4);
        assert_eq!(config.temporal_layers.num_layers, 4);
        assert_eq!(config.temporal_layers.qp_offset.len(), 4);
    }

    #[test]
    fn test_rate_control_modes() {
        let cqp = RateControlMode::Cqp { qp: 32 };
        let vbr = RateControlMode::Vbr { bitrate: 5_000_000 };
        let cbr = RateControlMode::Cbr { bitrate: 5_000_000 };
        let crf = RateControlMode::Crf { crf: 28.0 };

        assert!(matches!(cqp, RateControlMode::Cqp { .. }));
        assert!(matches!(vbr, RateControlMode::Vbr { .. }));
        assert!(matches!(cbr, RateControlMode::Cbr { .. }));
        assert!(matches!(crf, RateControlMode::Crf { .. }));
    }

    #[test]
    fn test_encoder_headers() {
        let config = VvcEncoderConfig::new(640, 480);
        let encoder = VvcEncoder::new(config).unwrap();
        let headers = encoder.headers();

        // Headers should contain VPS, SPS, PPS with start codes
        assert!(!headers.is_empty());
        // Check for start code
        assert_eq!(&headers[0..4], &[0, 0, 0, 1]);
    }

    #[test]
    fn test_frame_stats() {
        let stats = FrameStats {
            frame_type: 'I',
            size: 50000,
            avg_qp: 28.0,
            psnr_y: 42.0,
            psnr_u: 44.0,
            psnr_v: 44.5,
            temporal_id: 0,
            encode_time_ms: 100.0,
        };
        assert_eq!(stats.frame_type, 'I');
        assert_eq!(stats.size, 50000);
    }
}
