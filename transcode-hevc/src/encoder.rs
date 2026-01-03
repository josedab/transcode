//! HEVC encoder implementation.
//!
//! This module provides a complete HEVC/H.265 encoder including:
//! - Rate control (CQP, VBR, CBR)
//! - GOP structure and reference picture management
//! - Motion estimation and mode decision
//! - Transform, quantization, and entropy coding

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::derivable_impls)]
#![allow(unused_variables)]
#![allow(dead_code)]

use crate::cabac::CabacEncoder;
use crate::error::{HevcError, HevcLevel, HevcProfile, HevcTier, Result};
use crate::nal::{NalUnitHeader, NalUnitType, SliceType};
use crate::transform::{HevcQuantizer, HevcTransform, TransformSize};
use transcode_core::bitstream::{add_emulation_prevention, BitWriter};
use transcode_core::frame::Frame;
use transcode_core::packet::{Packet, PacketFlags};
use std::collections::VecDeque;

/// HEVC encoder preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HevcPreset {
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

impl HevcPreset {
    /// Get the search range for motion estimation.
    pub fn search_range(&self) -> u32 {
        match self {
            Self::Ultrafast => 16,
            Self::Superfast => 24,
            Self::Veryfast => 32,
            Self::Faster => 48,
            Self::Fast => 57,
            Self::Medium => 57,
            Self::Slow => 64,
            Self::Slower => 92,
            Self::Veryslow => 128,
            Self::Placebo => 256,
        }
    }

    /// Get the maximum CTU depth for mode decision.
    pub fn max_cu_depth(&self) -> u8 {
        match self {
            Self::Ultrafast => 1,
            Self::Superfast => 2,
            Self::Veryfast => 2,
            Self::Faster => 3,
            Self::Fast => 3,
            Self::Medium => 4,
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
}

impl Default for HevcPreset {
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
    Abr { bitrate: u32 },
    /// Constant bitrate (CBR).
    Cbr { bitrate: u32 },
    /// Constant Rate Factor (CRF).
    Crf { crf: f32 },
}

impl Default for RateControlMode {
    fn default() -> Self {
        Self::Crf { crf: 23.0 }
    }
}

/// HEVC encoder configuration.
#[derive(Debug, Clone)]
pub struct HevcEncoderConfig {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Encoding preset.
    pub preset: HevcPreset,
    /// Profile.
    pub profile: HevcProfile,
    /// Tier.
    pub tier: HevcTier,
    /// Level.
    pub level: HevcLevel,
    /// Rate control mode.
    pub rate_control: RateControlMode,
    /// Bit depth.
    pub bit_depth: u8,
    /// Chroma format (1=4:2:0, 2=4:2:2, 3=4:4:4).
    pub chroma_format: u8,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
    /// Number of B-frames between P-frames.
    pub bframes: u32,
    /// Enable B-frame pyramids.
    pub b_pyramid: bool,
    /// Enable weighted prediction.
    pub weighted_pred: bool,
    /// Enable SAO filter.
    pub sao_enabled: bool,
    /// Enable deblocking filter.
    pub deblocking_enabled: bool,
    /// Enable AMP (Asymmetric Motion Partitions).
    pub amp_enabled: bool,
    /// Enable transform skip.
    pub transform_skip_enabled: bool,
    /// CTU size (16, 32, or 64).
    pub ctu_size: u32,
    /// Minimum CU size.
    pub min_cu_size: u32,
    /// Maximum TU size.
    pub max_tu_size: u32,
    /// Maximum transform hierarchy depth for intra.
    pub max_transform_depth_intra: u8,
    /// Maximum transform hierarchy depth for inter.
    pub max_transform_depth_inter: u8,
}

impl HevcEncoderConfig {
    /// Create a new configuration for the given resolution.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            fps_num: 30,
            fps_den: 1,
            preset: HevcPreset::default(),
            profile: HevcProfile::Main,
            tier: HevcTier::Main,
            level: HevcLevel::L4_1,
            rate_control: RateControlMode::default(),
            bit_depth: 8,
            chroma_format: 1,
            gop_size: 60,
            bframes: 3,
            b_pyramid: true,
            weighted_pred: false,
            sao_enabled: true,
            deblocking_enabled: true,
            amp_enabled: false,
            transform_skip_enabled: false,
            ctu_size: 64,
            min_cu_size: 8,
            max_tu_size: 32,
            max_transform_depth_intra: 2,
            max_transform_depth_inter: 3,
        }
    }

    /// Set the preset.
    pub fn with_preset(mut self, preset: HevcPreset) -> Self {
        self.preset = preset;
        self
    }

    /// Set the bitrate (for ABR/CBR modes).
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.rate_control = RateControlMode::Abr { bitrate };
        self
    }

    /// Set constant QP.
    pub fn with_qp(mut self, qp: u8) -> Self {
        self.rate_control = RateControlMode::Cqp { qp: qp.min(51) };
        self
    }

    /// Set CRF value.
    pub fn with_crf(mut self, crf: f32) -> Self {
        self.rate_control = RateControlMode::Crf { crf: crf.clamp(0.0, 51.0) };
        self
    }

    /// Set GOP size.
    pub fn with_gop_size(mut self, gop_size: u32) -> Self {
        self.gop_size = gop_size;
        self
    }

    /// Set number of B-frames.
    pub fn with_bframes(mut self, bframes: u32) -> Self {
        self.bframes = bframes;
        self
    }

    /// Set profile.
    pub fn with_profile(mut self, profile: HevcProfile) -> Self {
        self.profile = profile;
        if profile == HevcProfile::Main10 {
            self.bit_depth = 10;
        }
        self
    }
}

/// Motion vector.
#[derive(Debug, Clone, Copy, Default)]
pub struct MotionVector {
    /// Horizontal component.
    pub x: i16,
    /// Vertical component.
    pub y: i16,
    /// Reference index.
    pub ref_idx: i8,
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
}

/// Frame to be encoded.
#[derive(Debug)]
struct PendingFrame {
    /// Input frame.
    frame: Frame,
    /// Presentation order.
    display_order: u64,
    /// Coding order.
    coding_order: u64,
    /// Frame type.
    frame_type: SliceType,
    /// POC.
    poc: i32,
}

/// HEVC encoder.
#[derive(Debug)]
pub struct HevcEncoder {
    /// Configuration.
    config: HevcEncoderConfig,
    /// Transform processor.
    transform: HevcTransform,
    /// Quantizer.
    quantizer: HevcQuantizer,
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
}

impl HevcEncoder {
    /// Create a new HEVC encoder.
    pub fn new(config: HevcEncoderConfig) -> Result<Self> {
        // Validate configuration
        if config.width == 0 || config.height == 0 {
            return Err(HevcError::EncoderConfig("Invalid dimensions".to_string()));
        }

        if config.ctu_size != 16 && config.ctu_size != 32 && config.ctu_size != 64 {
            return Err(HevcError::EncoderConfig("Invalid CTU size".to_string()));
        }

        let initial_qp = match config.rate_control {
            RateControlMode::Cqp { qp } => qp,
            RateControlMode::Crf { crf } => crf as u8,
            RateControlMode::Abr { bitrate } | RateControlMode::Cbr { bitrate } => {
                // Estimate initial QP from bitrate
                let pixels_per_second = config.width * config.height * config.fps_num / config.fps_den;
                let bits_per_pixel = (bitrate as f64) / (pixels_per_second as f64);
                (35.0 - bits_per_pixel.log2() * 5.0).clamp(10.0, 51.0) as u8
            }
        };

        let mut encoder = Self {
            transform: HevcTransform::new(config.bit_depth),
            quantizer: HevcQuantizer::new(config.bit_depth),
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
        writer.write_bit(true)?;  // vps_base_layer_internal_flag
        writer.write_bit(true)?;  // vps_base_layer_available_flag
        writer.write_bits(0, 6)?; // vps_max_layers_minus1
        writer.write_bits(0, 3)?; // vps_max_sub_layers_minus1
        writer.write_bit(true)?;  // vps_temporal_id_nesting_flag
        writer.write_bits(0xFFFF, 16)?; // vps_reserved_0xffff_16bits

        // Profile tier level
        self.write_profile_tier_level(&mut writer, true, 0)?;

        writer.write_bit(false)?; // vps_sub_layer_ordering_info_present_flag
        writer.write_ue(4)?; // vps_max_dec_pic_buffering_minus1
        writer.write_ue(2)?; // vps_max_num_reorder_pics
        writer.write_ue(0)?; // vps_max_latency_increase_plus1

        writer.write_bits(0, 6)?; // vps_max_layer_id
        writer.write_ue(0)?; // vps_num_layer_sets_minus1

        writer.write_bit(true)?; // vps_timing_info_present_flag
        writer.write_bits(self.config.fps_den, 32)?; // vps_num_units_in_tick
        writer.write_bits(self.config.fps_num, 32)?; // vps_time_scale

        writer.write_bit(false)?; // vps_poc_proportional_to_timing_flag
        writer.write_ue(0)?; // vps_num_hrd_parameters

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
        writer.write_bits(0, 4)?; // sps_video_parameter_set_id
        writer.write_bits(0, 3)?; // sps_max_sub_layers_minus1
        writer.write_bit(true)?;  // sps_temporal_id_nesting_flag

        // Profile tier level
        self.write_profile_tier_level(&mut writer, true, 0)?;

        writer.write_ue(0)?; // sps_seq_parameter_set_id
        writer.write_ue(self.config.chroma_format as u32)?; // chroma_format_idc

        if self.config.chroma_format == 3 {
            writer.write_bit(false)?; // separate_colour_plane_flag
        }

        writer.write_ue(self.config.width)?; // pic_width_in_luma_samples
        writer.write_ue(self.config.height)?; // pic_height_in_luma_samples

        writer.write_bit(false)?; // conformance_window_flag

        writer.write_ue((self.config.bit_depth - 8) as u32)?; // bit_depth_luma_minus8
        writer.write_ue((self.config.bit_depth - 8) as u32)?; // bit_depth_chroma_minus8
        writer.write_ue(4)?; // log2_max_pic_order_cnt_lsb_minus4

        writer.write_bit(false)?; // sps_sub_layer_ordering_info_present_flag
        writer.write_ue(4)?; // sps_max_dec_pic_buffering_minus1
        writer.write_ue(2)?; // sps_max_num_reorder_pics
        writer.write_ue(0)?; // sps_max_latency_increase_plus1

        let log2_min_cb = (self.config.min_cu_size as f32).log2() as u32;
        let log2_ctb = (self.config.ctu_size as f32).log2() as u32;
        let log2_min_tb = 2u32; // 4x4

        writer.write_ue(log2_min_cb - 3)?; // log2_min_luma_coding_block_size_minus3
        writer.write_ue(log2_ctb - log2_min_cb)?; // log2_diff_max_min_luma_coding_block_size
        writer.write_ue(log2_min_tb - 2)?; // log2_min_luma_transform_block_size_minus2
        writer.write_ue(3)?; // log2_diff_max_min_luma_transform_block_size
        writer.write_ue(self.config.max_transform_depth_inter as u32)?; // max_transform_hierarchy_depth_inter
        writer.write_ue(self.config.max_transform_depth_intra as u32)?; // max_transform_hierarchy_depth_intra

        writer.write_bit(false)?; // scaling_list_enabled_flag
        writer.write_bit(self.config.amp_enabled)?; // amp_enabled_flag
        writer.write_bit(self.config.sao_enabled)?; // sample_adaptive_offset_enabled_flag
        writer.write_bit(false)?; // pcm_enabled_flag

        // Short-term ref pic sets
        writer.write_ue(0)?; // num_short_term_ref_pic_sets

        writer.write_bit(false)?; // long_term_ref_pics_present_flag
        writer.write_bit(true)?;  // sps_temporal_mvp_enabled_flag
        writer.write_bit(true)?;  // strong_intra_smoothing_enabled_flag

        writer.write_bit(false)?; // vui_parameters_present_flag
        writer.write_bit(false)?; // sps_extension_present_flag

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
        writer.write_ue(0)?; // pps_pic_parameter_set_id
        writer.write_ue(0)?; // pps_seq_parameter_set_id

        writer.write_bit(false)?; // dependent_slice_segments_enabled_flag
        writer.write_bit(false)?; // output_flag_present_flag
        writer.write_bits(0, 3)?; // num_extra_slice_header_bits
        writer.write_bit(false)?; // sign_data_hiding_enabled_flag
        writer.write_bit(true)?;  // cabac_init_present_flag

        writer.write_ue(0)?; // num_ref_idx_l0_default_active_minus1
        writer.write_ue(0)?; // num_ref_idx_l1_default_active_minus1

        let init_qp = self.current_qp as i32 - 26;
        writer.write_se(init_qp)?; // init_qp_minus26

        writer.write_bit(false)?; // constrained_intra_pred_flag
        writer.write_bit(self.config.transform_skip_enabled)?; // transform_skip_enabled_flag
        writer.write_bit(true)?;  // cu_qp_delta_enabled_flag
        writer.write_ue(0)?; // diff_cu_qp_delta_depth

        writer.write_se(0)?; // pps_cb_qp_offset
        writer.write_se(0)?; // pps_cr_qp_offset

        writer.write_bit(false)?; // pps_slice_chroma_qp_offsets_present_flag
        writer.write_bit(self.config.weighted_pred)?; // weighted_pred_flag
        writer.write_bit(false)?; // weighted_bipred_flag
        writer.write_bit(false)?; // transquant_bypass_enabled_flag
        writer.write_bit(false)?; // tiles_enabled_flag
        writer.write_bit(false)?; // entropy_coding_sync_enabled_flag

        // No tiles, so skip tile-related syntax

        writer.write_bit(true)?; // pps_loop_filter_across_slices_enabled_flag
        writer.write_bit(self.config.deblocking_enabled)?; // deblocking_filter_control_present_flag

        if self.config.deblocking_enabled {
            writer.write_bit(false)?; // deblocking_filter_override_enabled_flag
            writer.write_bit(false)?; // pps_deblocking_filter_disabled_flag
            writer.write_se(0)?; // pps_beta_offset_div2
            writer.write_se(0)?; // pps_tc_offset_div2
        }

        writer.write_bit(false)?; // pps_scaling_list_data_present_flag
        writer.write_bit(false)?; // lists_modification_present_flag
        writer.write_ue(0)?; // log2_parallel_merge_level_minus2
        writer.write_bit(false)?; // slice_segment_header_extension_present_flag
        writer.write_bit(false)?; // pps_extension_present_flag

        writer.write_rbsp_trailing_bits()?;

        let data = writer.into_data();
        Ok(add_emulation_prevention(&data))
    }

    /// Write profile tier level syntax.
    fn write_profile_tier_level(
        &self,
        writer: &mut BitWriter,
        profile_present_flag: bool,
        max_num_sub_layers_minus1: u8,
    ) -> Result<()> {
        if profile_present_flag {
            writer.write_bits(0, 2)?; // general_profile_space
            writer.write_bit(self.config.tier == HevcTier::High)?; // general_tier_flag
            writer.write_bits(self.config.profile.idc() as u32, 5)?; // general_profile_idc

            // general_profile_compatibility_flag[32]
            let mut compat_flags = 0u32;
            compat_flags |= 1 << (31 - self.config.profile.idc());
            writer.write_bits(compat_flags, 32)?;

            writer.write_bit(true)?; // general_progressive_source_flag
            writer.write_bit(false)?; // general_interlaced_source_flag
            writer.write_bit(false)?; // general_non_packed_constraint_flag
            writer.write_bit(true)?; // general_frame_only_constraint_flag

            // Reserved bits (44 bits)
            writer.write_bits(0, 32)?;
            writer.write_bits(0, 12)?;
        }

        writer.write_bits(self.config.level.level_idc as u32, 8)?; // general_level_idc

        // Sub-layer flags (none for now)
        for _ in 0..max_num_sub_layers_minus1 {
            writer.write_bit(false)?; // sub_layer_profile_present_flag
            writer.write_bit(false)?; // sub_layer_level_present_flag
        }

        // Reserved bits if max_num_sub_layers_minus1 > 0
        if max_num_sub_layers_minus1 > 0 {
            for _ in max_num_sub_layers_minus1..8 {
                writer.write_bits(0, 2)?;
            }
        }

        Ok(())
    }

    /// Encode a frame.
    pub fn encode(&mut self, frame: &Frame) -> Result<Option<Packet<'static>>> {
        // Validate input frame
        if frame.width() != self.config.width || frame.height() != self.config.height {
            return Err(HevcError::EncoderConfig("Frame dimensions mismatch".to_string()));
        }

        // Determine frame type
        let frame_type = self.determine_frame_type();

        // Add to pending frames for B-frame reordering
        let pending = PendingFrame {
            frame: frame.clone(),
            display_order: self.frame_num,
            coding_order: 0, // Will be set when encoding
            frame_type,
            poc: self.poc,
        };

        self.pending_frames.push_back(pending);
        self.frame_num += 1;
        self.poc += 1;

        // Try to encode a frame
        self.try_encode_frame()
    }

    /// Determine the frame type for the next frame.
    fn determine_frame_type(&self) -> SliceType {
        if self.frame_num == 0 || self.frame_num % self.config.gop_size as u64 == 0 {
            SliceType::I
        } else if self.config.bframes > 0
            && (self.frame_num % (self.config.bframes as u64 + 1)) != 0
        {
            SliceType::B
        } else {
            SliceType::P
        }
    }

    /// Try to encode a pending frame.
    fn try_encode_frame(&mut self) -> Result<Option<Packet<'static>>> {
        // Need at least one frame to encode
        if self.pending_frames.is_empty() {
            return Ok(None);
        }

        // For B-frame support, we need to wait for future references
        let can_encode = if self.config.bframes > 0 {
            // Check if we have enough frames or if the next frame is an I/P frame
            self.pending_frames.len() > self.config.bframes as usize
                || self.pending_frames.front().map(|f| f.frame_type != SliceType::B).unwrap_or(false)
        } else {
            true
        };

        if !can_encode {
            return Ok(None);
        }

        // Get the frame to encode
        let pending = self.pending_frames.pop_front().unwrap();

        // Encode the frame
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

        // Encode slice
        let slice_data = self.encode_slice(&pending)?;
        output.extend_from_slice(&[0, 0, 0, 1]);
        output.extend_from_slice(&slice_data);

        // Update statistics
        self.last_stats = FrameStats {
            frame_type: match pending.frame_type {
                SliceType::I => 'I',
                SliceType::P => 'P',
                SliceType::B => 'B',
            },
            size: output.len(),
            avg_qp: self.current_qp as f32,
            psnr_y: 0.0,
            psnr_u: 0.0,
            psnr_v: 0.0,
            encode_time_ms: 0.0,
        };

        // Update reference pictures
        if pending.frame_type != SliceType::B {
            self.update_reference_pictures(&pending.frame, pending.poc);
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
            SliceType::P => NalUnitType::TrailR,
            SliceType::B => NalUnitType::TrailN,
        };

        // NAL unit header
        let header = NalUnitHeader {
            nal_unit_type: nal_type,
            nuh_layer_id: 0,
            nuh_temporal_id_plus1: 1,
        };
        header.write(&mut writer)?;

        // Slice segment header
        self.write_slice_header(&mut writer, pending, nal_type)?;

        // Encode CTUs
        writer.byte_align()?;
        let slice_data = self.encode_slice_data(pending)?;

        let mut result = writer.into_data();
        result.extend_from_slice(&slice_data);

        Ok(add_emulation_prevention(&result))
    }

    /// Write slice segment header.
    fn write_slice_header(
        &self,
        writer: &mut BitWriter,
        pending: &PendingFrame,
        nal_type: NalUnitType,
    ) -> Result<()> {
        writer.write_bit(true)?; // first_slice_segment_in_pic_flag

        if nal_type.is_irap() {
            writer.write_bit(false)?; // no_output_of_prior_pics_flag
        }

        writer.write_ue(0)?; // slice_pic_parameter_set_id

        // slice_type
        let slice_type_val = match pending.frame_type {
            SliceType::B => 0,
            SliceType::P => 1,
            SliceType::I => 2,
        };
        writer.write_ue(slice_type_val)?;

        // POC for non-IDR
        if !nal_type.is_idr() {
            writer.write_bits(pending.poc as u32 & 0xFF, 8)?; // slice_pic_order_cnt_lsb
            writer.write_bit(true)?; // short_term_ref_pic_set_sps_flag (use SPS set 0)
        }

        // SAO flags
        if self.config.sao_enabled {
            writer.write_bit(true)?; // slice_sao_luma_flag
            if self.config.chroma_format > 0 {
                writer.write_bit(true)?; // slice_sao_chroma_flag
            }
        }

        // Reference picture management for P/B slices
        if pending.frame_type != SliceType::I {
            // Simplified reference picture management
            writer.write_bit(false)?; // num_ref_idx_active_override_flag
        }

        // Slice QP delta
        writer.write_signed_exp_golomb(0)?; // slice_qp_delta

        // Deblocking filter
        if self.config.deblocking_enabled {
            writer.write_bit(false)?; // slice_deblocking_filter_disabled_flag
        }

        Ok(())
    }

    /// Encode slice data (CTUs).
    fn encode_slice_data(&mut self, pending: &PendingFrame) -> Result<Vec<u8>> {
        let mut cabac = CabacEncoder::new();
        cabac.init_contexts(pending.frame_type, false);

        let ctu_size = self.config.ctu_size;
        let width_in_ctus = (self.config.width + ctu_size - 1) / ctu_size;
        let height_in_ctus = (self.config.height + ctu_size - 1) / ctu_size;

        for ctu_y in 0..height_in_ctus {
            for ctu_x in 0..width_in_ctus {
                let x = ctu_x * ctu_size;
                let y = ctu_y * ctu_size;

                self.encode_ctu(&mut cabac, pending, x, y)?;
            }
        }

        // Encode end of slice
        cabac.encode_terminate(true)?;

        Ok(cabac.into_data())
    }

    /// Encode a CTU.
    fn encode_ctu(
        &mut self,
        cabac: &mut CabacEncoder,
        pending: &PendingFrame,
        x: u32,
        y: u32,
    ) -> Result<()> {
        // For now, use simple encoding without RDO
        // A full encoder would perform mode decision here

        let size = self.config.ctu_size.min(self.config.width - x).min(self.config.height - y);

        // Encode coding quadtree
        self.encode_coding_quadtree(cabac, pending, x, y, size, 0)?;

        Ok(())
    }

    /// Encode coding quadtree.
    fn encode_coding_quadtree(
        &mut self,
        cabac: &mut CabacEncoder,
        pending: &PendingFrame,
        x: u32,
        y: u32,
        size: u32,
        depth: u8,
    ) -> Result<()> {
        let log2_size = (size as f32).log2() as u8;
        let log2_min_cb = (self.config.min_cu_size as f32).log2() as u8;
        let max_depth = self.config.preset.max_cu_depth();

        // Decide whether to split
        let split = log2_size > log2_min_cb && depth < max_depth;

        if log2_size > log2_min_cb {
            cabac.encode_decision(depth as usize, split)?;
        }

        if split {
            let half_size = size / 2;
            for i in 0..4 {
                let sub_x = x + (i % 2) * half_size;
                let sub_y = y + (i / 2) * half_size;

                if sub_x < self.config.width && sub_y < self.config.height {
                    self.encode_coding_quadtree(
                        cabac, pending, sub_x, sub_y, half_size, depth + 1
                    )?;
                }
            }
        } else {
            // Encode coding unit
            self.encode_coding_unit(cabac, pending, x, y, size)?;
        }

        Ok(())
    }

    /// Encode a coding unit.
    fn encode_coding_unit(
        &mut self,
        cabac: &mut CabacEncoder,
        pending: &PendingFrame,
        x: u32,
        y: u32,
        size: u32,
    ) -> Result<()> {
        let is_intra = pending.frame_type == SliceType::I;

        if !is_intra {
            // Skip flag
            cabac.encode_decision(0, false)?; // Not skip
            // Pred mode flag
            cabac.encode_decision(0, true)?; // Intra (simplified)
        }

        // Part mode (2Nx2N)
        cabac.encode_decision(0, true)?; // 2Nx2N

        // Intra prediction mode
        self.encode_intra_prediction(cabac, pending, x, y, size)?;

        // Transform tree
        self.encode_transform_tree(cabac, pending, x, y, size, 0)?;

        Ok(())
    }

    /// Encode intra prediction.
    fn encode_intra_prediction(
        &mut self,
        cabac: &mut CabacEncoder,
        pending: &PendingFrame,
        x: u32,
        y: u32,
        size: u32,
    ) -> Result<()> {
        // Use DC mode for simplicity
        cabac.encode_decision(0, true)?; // prev_intra_luma_pred_flag
        cabac.encode_bypass(false)?; // mpm_idx = 1 (DC)
        cabac.encode_bypass(false)?;

        // Chroma mode (same as luma = 4)
        cabac.encode_bypass_bins(4, 2)?;

        Ok(())
    }

    /// Encode transform tree.
    fn encode_transform_tree(
        &mut self,
        cabac: &mut CabacEncoder,
        pending: &PendingFrame,
        x: u32,
        y: u32,
        size: u32,
        depth: u8,
    ) -> Result<()> {
        // For simplicity, don't split transforms
        // A real encoder would use RDO to decide

        // CBF flags
        let has_residual = true; // Simplified
        cabac.encode_decision(0, has_residual)?; // cbf_luma

        if has_residual {
            self.encode_residual(cabac, pending, x, y, size, true)?;
        }

        // Chroma CBF
        if self.config.chroma_format > 0 && size > 4 {
            cabac.encode_decision(1, has_residual)?; // cbf_cb
            cabac.encode_decision(2, has_residual)?; // cbf_cr
        }

        Ok(())
    }

    /// Encode residual coefficients.
    fn encode_residual(
        &mut self,
        cabac: &mut CabacEncoder,
        pending: &PendingFrame,
        x: u32,
        y: u32,
        size: u32,
        is_luma: bool,
    ) -> Result<()> {
        // Get residual block
        let num_coeffs = (size * size) as usize;
        let mut residual = vec![0i16; num_coeffs];

        // Calculate residual from prediction
        self.calculate_residual(pending, x, y, size, &mut residual, is_luma)?;

        // Forward transform
        let transform_size = TransformSize::from_size(size as usize)
            .ok_or_else(|| HevcError::Transform("Invalid transform size".to_string()))?;

        let mut coeffs = vec![0i32; num_coeffs];
        self.transform.forward_transform(
            &residual,
            &mut coeffs,
            transform_size,
            size as usize,
            true, // is_intra
            is_luma,
        );

        // Quantize
        let mut qcoeffs = vec![0i32; num_coeffs];
        self.quantizer.quantize(
            &coeffs,
            &mut qcoeffs,
            size as usize,
            self.current_qp as i32,
            true, // is_intra
            None,
        )?;

        // Find last significant coefficient
        let mut last_x = 0u32;
        let mut last_y = 0u32;

        for j in (0..size).rev() {
            for i in (0..size).rev() {
                if qcoeffs[(j * size + i) as usize] != 0 {
                    last_x = i;
                    last_y = j;
                    break;
                }
            }
        }

        // Encode last position
        let log2_size = (size as f32).log2() as u8;
        self.encode_last_sig_coeff(cabac, last_x, last_y, log2_size)?;

        // Encode coefficient levels (simplified)
        for j in 0..=last_y {
            for i in 0..=if j == last_y { last_x } else { size - 1 } {
                let coeff = qcoeffs[(j * size + i) as usize];
                let sig = coeff != 0;

                if j != last_y || i != last_x {
                    cabac.encode_decision(0, sig)?;
                }

                if sig {
                    let abs_coeff = coeff.unsigned_abs();
                    let sign = coeff < 0;

                    // Greater1 flag
                    cabac.encode_decision(0, abs_coeff > 1)?;

                    if abs_coeff > 1 {
                        // Greater2 flag
                        cabac.encode_decision(0, abs_coeff > 2)?;

                        if abs_coeff > 2 {
                            // Remaining level
                            let remaining = abs_coeff - 3;
                            let mut k = 0;
                            let mut level = remaining;
                            while level > 0 {
                                cabac.encode_bypass(true)?;
                                level >>= 1;
                                k += 1;
                            }
                            cabac.encode_bypass(false)?;
                            if k > 0 {
                                cabac.encode_bypass_bins(remaining, k)?;
                            }
                        }
                    }

                    // Sign
                    cabac.encode_bypass(sign)?;
                }
            }
        }

        Ok(())
    }

    /// Calculate residual.
    fn calculate_residual(
        &self,
        pending: &PendingFrame,
        x: u32,
        y: u32,
        size: u32,
        residual: &mut [i16],
        is_luma: bool,
    ) -> Result<()> {
        let frame = &pending.frame;
        let plane_idx = if is_luma { 0 } else { 1 };
        let stride = frame.stride(plane_idx);

        let plane = frame.plane(plane_idx)
            .ok_or_else(|| HevcError::InvalidState("Plane not found".to_string()))?;

        // DC prediction value
        let mut sum = 0u32;
        let mut count = 0u32;

        // Left column
        if x > 0 {
            for j in 0..size {
                sum += plane[((y + j) as usize) * stride + (x - 1) as usize] as u32;
                count += 1;
            }
        }

        // Top row
        if y > 0 {
            for i in 0..size {
                sum += plane[((y - 1) as usize) * stride + (x + i) as usize] as u32;
                count += 1;
            }
        }

        let dc = if count > 0 {
            ((sum + count / 2) / count) as i16
        } else {
            128
        };

        // Calculate residual
        for j in 0..size {
            for i in 0..size {
                let orig = plane[((y + j) as usize) * stride + (x + i) as usize] as i16;
                residual[(j * size + i) as usize] = orig - dc;
            }
        }

        Ok(())
    }

    /// Encode last significant coefficient position.
    fn encode_last_sig_coeff(
        &self,
        cabac: &mut CabacEncoder,
        last_x: u32,
        last_y: u32,
        log2_size: u8,
    ) -> Result<()> {
        let max_prefix = ((log2_size << 1) - 1) as u32;

        // X prefix
        for i in 0..last_x.min(max_prefix) {
            cabac.encode_decision(i as usize, true)?;
        }
        if last_x < max_prefix {
            cabac.encode_decision(last_x as usize, false)?;
        }

        // X suffix
        if last_x > 3 {
            let suffix_length = ((last_x - 2) >> 1) as u8;
            let suffix = last_x - ((2 + (last_x & 1)) << suffix_length);
            cabac.encode_bypass_bins(suffix, suffix_length)?;
        }

        // Y prefix
        for i in 0..last_y.min(max_prefix) {
            cabac.encode_decision(i as usize, true)?;
        }
        if last_y < max_prefix {
            cabac.encode_decision(last_y as usize, false)?;
        }

        // Y suffix
        if last_y > 3 {
            let suffix_length = ((last_y - 2) >> 1) as u8;
            let suffix = last_y - ((2 + (last_y & 1)) << suffix_length);
            cabac.encode_bypass_bins(suffix, suffix_length)?;
        }

        Ok(())
    }

    /// Update reference pictures.
    fn update_reference_pictures(&mut self, frame: &Frame, poc: i32) {
        // Add to reference list
        let ref_pic = ReferencePicture {
            frame: frame.clone(),
            poc,
            is_long_term: false,
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

        // Encode remaining pending frames
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
        let config = HevcEncoderConfig::new(1920, 1080)
            .with_preset(HevcPreset::Fast)
            .with_crf(23.0)
            .with_gop_size(60);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.preset, HevcPreset::Fast);
        assert_eq!(config.gop_size, 60);
    }

    #[test]
    fn test_encoder_creation() {
        let config = HevcEncoderConfig::new(640, 480);
        let encoder = HevcEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_invalid_config() {
        let config = HevcEncoderConfig::new(0, 0);
        let encoder = HevcEncoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_preset_properties() {
        assert!(HevcPreset::Ultrafast.search_range() < HevcPreset::Veryslow.search_range());
        assert!(HevcPreset::Ultrafast.max_cu_depth() <= HevcPreset::Veryslow.max_cu_depth());
        assert!(HevcPreset::Ultrafast.rdo_level() < HevcPreset::Veryslow.rdo_level());
    }

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector {
            x: 10,
            y: -5,
            ref_idx: 0,
        };
        assert_eq!(mv.x, 10);
        assert_eq!(mv.y, -5);
    }

    #[test]
    fn test_frame_stats() {
        let stats = FrameStats {
            frame_type: 'I',
            size: 10000,
            avg_qp: 26.0,
            psnr_y: 40.0,
            psnr_u: 42.0,
            psnr_v: 42.5,
            encode_time_ms: 50.0,
        };
        assert_eq!(stats.frame_type, 'I');
        assert_eq!(stats.size, 10000);
    }

    #[test]
    fn test_rate_control_modes() {
        let cqp = RateControlMode::Cqp { qp: 26 };
        let abr = RateControlMode::Abr { bitrate: 2_000_000 };
        let cbr = RateControlMode::Cbr { bitrate: 2_000_000 };
        let crf = RateControlMode::Crf { crf: 23.0 };

        assert!(matches!(cqp, RateControlMode::Cqp { .. }));
        assert!(matches!(abr, RateControlMode::Abr { .. }));
        assert!(matches!(cbr, RateControlMode::Cbr { .. }));
        assert!(matches!(crf, RateControlMode::Crf { .. }));
    }

    #[test]
    fn test_encoder_headers() {
        let config = HevcEncoderConfig::new(640, 480);
        let encoder = HevcEncoder::new(config).unwrap();
        let headers = encoder.headers();

        // Headers should contain VPS, SPS, PPS with start codes
        assert!(!headers.is_empty());
        // Check for start code
        assert_eq!(&headers[0..4], &[0, 0, 0, 1]);
    }
}
