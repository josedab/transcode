//! H.264 encoder implementation.

use std::sync::Arc;
use transcode_core::{Frame, Packet, Result, TimeBase};
use transcode_core::bitstream::BitWriter;
use crate::traits::{VideoEncoder, CodecInfo};
use super::{H264Profile, H264Level};
use super::presets::Preset;
use super::parallel::{
    ThreadingConfig, ThreadSafeEncoderState, ParallelSliceEncoder,
    ParallelMotionEstimator, LookaheadBuffer, SliceEncoderContext,
    SliceData, ReferenceFrame, LookaheadFrame, FrameType,
};

/// Rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant Rate Factor (quality-based).
    Crf(u8),
    /// Constant Bitrate.
    Cbr(u32),
    /// Variable Bitrate with target.
    Vbr { target: u32, max: u32 },
    /// Constant QP.
    Cqp(u8),
}

impl Default for RateControlMode {
    fn default() -> Self {
        Self::Crf(23)
    }
}

/// Encoding preset (speed vs quality tradeoff).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncoderPreset {
    /// Fastest encoding, lowest quality.
    Ultrafast,
    /// Very fast encoding.
    Superfast,
    /// Fast encoding.
    Veryfast,
    /// Faster encoding.
    Faster,
    /// Fast encoding.
    Fast,
    /// Balanced (default).
    #[default]
    Medium,
    /// Slower encoding, better quality.
    Slow,
    /// Even slower encoding.
    Slower,
    /// Very slow encoding.
    Veryslow,
    /// Placebo (maximum quality, impractical speed).
    Placebo,
}

/// H.264 encoder configuration.
#[derive(Debug, Clone)]
pub struct H264EncoderConfig {
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Frame rate (frames per second).
    pub frame_rate: (u32, u32),
    /// Target profile.
    pub profile: H264Profile,
    /// Target level.
    pub level: H264Level,
    /// Rate control mode.
    pub rate_control: RateControlMode,
    /// Encoding preset.
    pub preset: EncoderPreset,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
    /// Number of B-frames between I and P.
    pub bframes: u8,
    /// Number of reference frames.
    pub ref_frames: u8,
    /// Enable CABAC (vs CAVLC).
    pub cabac: bool,
    /// Threading configuration for parallel encoding.
    pub threading: ThreadingConfig,
}

impl H264EncoderConfig {
    /// Create a new config for the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            frame_rate: (30, 1),
            profile: H264Profile::Main,
            level: H264Level::from_idc(40),
            rate_control: RateControlMode::default(),
            preset: EncoderPreset::default(),
            gop_size: 250,
            bframes: 2,
            ref_frames: 3,
            cabac: true,
            threading: ThreadingConfig::default(),
        }
    }

    /// Set frame rate.
    pub fn with_frame_rate(mut self, fps: u32) -> Self {
        self.frame_rate = (fps, 1);
        self
    }

    /// Set bitrate (CBR mode).
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.rate_control = RateControlMode::Cbr(bitrate);
        self
    }

    /// Set CRF quality.
    pub fn with_crf(mut self, crf: u8) -> Self {
        self.rate_control = RateControlMode::Crf(crf);
        self
    }

    /// Set preset.
    pub fn with_preset(mut self, preset: EncoderPreset) -> Self {
        self.preset = preset;
        self
    }

    /// Set number of threads for parallel encoding.
    /// Use 0 for auto-detection based on CPU cores.
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.threading.num_threads = num_threads;
        self
    }

    /// Set number of slices per frame for slice-based parallelism.
    pub fn with_slice_count(mut self, slice_count: usize) -> Self {
        self.threading.slice_count = slice_count.max(1);
        self
    }

    /// Set lookahead depth for B-frame decisions.
    pub fn with_lookahead_depth(mut self, depth: usize) -> Self {
        self.threading.lookahead_depth = depth;
        self
    }

    /// Set threading configuration.
    pub fn with_threading(mut self, threading: ThreadingConfig) -> Self {
        self.threading = threading;
        self
    }

    /// Enable or disable slice-based parallel encoding.
    pub fn with_slice_parallel(mut self, enable: bool) -> Self {
        self.threading.enable_slice_parallel = enable;
        self
    }

    /// Enable or disable frame-based parallel encoding (motion estimation).
    pub fn with_frame_parallel(mut self, enable: bool) -> Self {
        self.threading.enable_frame_parallel = enable;
        self
    }
}

/// H.264 encoder.
pub struct H264Encoder {
    config: H264EncoderConfig,
    /// Frame counter.
    frame_count: u64,
    /// IDR frame counter.
    idr_count: u64,
    /// Generated SPS.
    sps: Vec<u8>,
    /// Generated PPS.
    pps: Vec<u8>,
    /// Pending output frames.
    pending_frames: Vec<Frame>,
    /// Time base.
    time_base: TimeBase,
    /// Thread-safe encoder state for parallel encoding.
    parallel_state: Option<Arc<ThreadSafeEncoderState>>,
    /// Parallel slice encoder.
    slice_encoder: Option<ParallelSliceEncoder>,
    /// Parallel motion estimator.
    motion_estimator: Option<ParallelMotionEstimator>,
    /// Lookahead buffer for B-frame decisions.
    lookahead: Option<LookaheadBuffer>,
}

impl H264Encoder {
    /// Create a new encoder.
    pub fn new(config: H264EncoderConfig) -> Result<Self> {
        // Initialize parallel components if threading is enabled
        let (parallel_state, slice_encoder, motion_estimator, lookahead) =
            if config.threading.enable_slice_parallel || config.threading.enable_frame_parallel {
                let state = Arc::new(ThreadSafeEncoderState::new(config.ref_frames as usize));

                // Set target bitrate for rate control
                if let RateControlMode::Cbr(bitrate) | RateControlMode::Vbr { target: bitrate, .. } = config.rate_control {
                    let frame_rate = config.frame_rate.0 as f64 / config.frame_rate.1 as f64;
                    state.set_target_bitrate(bitrate, frame_rate);
                }

                let slice_enc = if config.threading.enable_slice_parallel {
                    Some(ParallelSliceEncoder::new(&config.threading)?)
                } else {
                    None
                };

                let motion_est = if config.threading.enable_frame_parallel {
                    Some(ParallelMotionEstimator::new(&config.threading)?)
                } else {
                    None
                };

                let lookahead_buf = if config.threading.enable_frame_parallel && config.bframes > 0 {
                    Some(LookaheadBuffer::new(
                        &config.threading,
                        config.bframes,
                        config.gop_size,
                    )?)
                } else {
                    None
                };

                (Some(state), slice_enc, motion_est, lookahead_buf)
            } else {
                (None, None, None, None)
            };

        let mut encoder = Self {
            config,
            frame_count: 0,
            idr_count: 0,
            sps: Vec::new(),
            pps: Vec::new(),
            pending_frames: Vec::new(),
            time_base: TimeBase::MPEG,
            parallel_state,
            slice_encoder,
            motion_estimator,
            lookahead,
        };

        encoder.generate_parameter_sets()?;

        Ok(encoder)
    }

    /// Create a new encoder with parallel encoding disabled.
    pub fn new_single_threaded(width: u32, height: u32) -> Result<Self> {
        let mut config = H264EncoderConfig::new(width, height);
        config.threading.enable_slice_parallel = false;
        config.threading.enable_frame_parallel = false;
        Self::new(config)
    }

    /// Create a new encoder with specified thread count.
    pub fn new_with_threads(width: u32, height: u32, num_threads: usize) -> Result<Self> {
        let config = H264EncoderConfig::new(width, height)
            .with_threads(num_threads);
        Self::new(config)
    }

    /// Get the current threading configuration.
    pub fn threading_config(&self) -> &ThreadingConfig {
        &self.config.threading
    }

    /// Check if parallel encoding is enabled.
    pub fn is_parallel(&self) -> bool {
        self.parallel_state.is_some()
    }

    /// Get the number of threads being used.
    pub fn thread_count(&self) -> usize {
        self.config.threading.effective_threads()
    }

    /// Create a new encoder from a preset.
    ///
    /// This is a convenience method that configures the encoder with
    /// optimized settings for common use cases.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use transcode_codecs::video::h264::{H264Encoder, Preset};
    ///
    /// let encoder = H264Encoder::from_preset(Preset::YouTube1080p)?;
    /// let encoder = H264Encoder::from_preset(Preset::StreamingLowLatency)?;
    /// ```
    pub fn from_preset(preset: Preset) -> Result<Self> {
        Self::new(preset.to_config())
    }

    /// Create a new encoder from a preset with custom dimensions.
    ///
    /// This allows using a preset's settings while overriding the resolution.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use transcode_codecs::video::h264::{H264Encoder, Preset};
    ///
    /// // Use YouTube 1080p settings but at 4K resolution
    /// let encoder = H264Encoder::from_preset_with_dimensions(
    ///     Preset::YouTube1080p,
    ///     3840,
    ///     2160,
    /// )?;
    /// ```
    pub fn from_preset_with_dimensions(preset: Preset, width: u32, height: u32) -> Result<Self> {
        let mut config = preset.to_config();
        config.width = width;
        config.height = height;
        Self::new(config)
    }

    /// Generate SPS and PPS.
    fn generate_parameter_sets(&mut self) -> Result<()> {
        self.sps = self.generate_sps();
        self.pps = self.generate_pps();
        Ok(())
    }

    /// Generate SPS NAL unit.
    fn generate_sps(&self) -> Vec<u8> {
        let mut writer = BitWriter::new();

        // NAL header (nal_ref_idc=3, nal_unit_type=7)
        let nal_header: Vec<u8> = vec![0x67];

        // profile_idc
        writer.write_bits(self.config.profile as u32, 8);
        // constraint_set flags
        writer.write_bits(0, 8);
        // level_idc
        writer.write_bits(self.config.level.value as u32, 8);
        // sps_id
        writer.write_ue(0);
        // log2_max_frame_num_minus4
        writer.write_ue(0);
        // pic_order_cnt_type
        writer.write_ue(0);
        // log2_max_pic_order_cnt_lsb_minus4
        writer.write_ue(4);
        // max_num_ref_frames
        writer.write_ue(self.config.ref_frames as u32);
        // gaps_in_frame_num_value_allowed_flag
        writer.write_bit(false);
        // pic_width_in_mbs_minus1
        writer.write_ue(self.config.width.div_ceil(16) - 1);
        // pic_height_in_map_units_minus1
        writer.write_ue(self.config.height.div_ceil(16) - 1);
        // frame_mbs_only_flag
        writer.write_bit(true);
        // direct_8x8_inference_flag
        writer.write_bit(true);
        // frame_cropping_flag
        let crop = !self.config.width.is_multiple_of(16) || !self.config.height.is_multiple_of(16);
        writer.write_bit(crop);
        if crop {
            let crop_right = (16 - (self.config.width % 16)) % 16 / 2;
            let crop_bottom = (16 - (self.config.height % 16)) % 16 / 2;
            writer.write_ue(0); // left
            writer.write_ue(crop_right);
            writer.write_ue(0); // top
            writer.write_ue(crop_bottom);
        }
        // vui_parameters_present_flag
        writer.write_bit(true);
        // aspect_ratio_info_present_flag
        writer.write_bit(false);
        // overscan_info_present_flag
        writer.write_bit(false);
        // video_signal_type_present_flag
        writer.write_bit(false);
        // chroma_loc_info_present_flag
        writer.write_bit(false);
        // timing_info_present_flag
        writer.write_bit(true);
        // num_units_in_tick
        writer.write_bits(self.config.frame_rate.1, 32);
        // time_scale (2x for field-based timing)
        writer.write_bits(self.config.frame_rate.0 * 2, 32);
        // fixed_frame_rate_flag
        writer.write_bit(true);

        // nal_hrd_parameters_present_flag
        writer.write_bit(false);
        // vcl_hrd_parameters_present_flag
        writer.write_bit(false);
        // pic_struct_present_flag
        writer.write_bit(false);
        // bitstream_restriction_flag
        writer.write_bit(false);

        writer.write_rbsp_trailing_bits();

        let mut result = nal_header;
        result.extend(transcode_core::bitstream::add_emulation_prevention(writer.data()));
        result
    }

    /// Generate PPS NAL unit.
    fn generate_pps(&self) -> Vec<u8> {
        let mut writer = BitWriter::new();

        // NAL header (nal_ref_idc=3, nal_unit_type=8)
        let nal_header: Vec<u8> = vec![0x68];

        // pps_id
        writer.write_ue(0);
        // sps_id
        writer.write_ue(0);
        // entropy_coding_mode_flag (CABAC)
        writer.write_bit(self.config.cabac);
        // bottom_field_pic_order_in_frame_present_flag
        writer.write_bit(false);
        // num_slice_groups_minus1
        writer.write_ue(0);
        // num_ref_idx_l0_default_active_minus1
        writer.write_ue((self.config.ref_frames - 1) as u32);
        // num_ref_idx_l1_default_active_minus1
        writer.write_ue(0);
        // weighted_pred_flag
        writer.write_bit(false);
        // weighted_bipred_idc
        writer.write_bits(0, 2);
        // pic_init_qp_minus26
        writer.write_se(0);
        // pic_init_qs_minus26
        writer.write_se(0);
        // chroma_qp_index_offset
        writer.write_se(0);
        // deblocking_filter_control_present_flag
        writer.write_bit(true);
        // constrained_intra_pred_flag
        writer.write_bit(false);
        // redundant_pic_cnt_present_flag
        writer.write_bit(false);

        writer.write_rbsp_trailing_bits();

        let mut result = nal_header;
        result.extend(transcode_core::bitstream::add_emulation_prevention(writer.data()));
        result
    }

    /// Generate AVCC extra data.
    fn generate_avcc(&self) -> Vec<u8> {
        let mut avcc = vec![
            // configurationVersion
            1,
            // AVCProfileIndication
            self.config.profile as u8,
            // profile_compatibility
            0,
            // AVCLevelIndication
            self.config.level.value,
            // lengthSizeMinusOne (4 bytes - 1)
            0xFF,
            // numOfSequenceParameterSets
            0xE1,
        ];
        // SPS length
        avcc.extend_from_slice(&(self.sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&self.sps);
        // numOfPictureParameterSets
        avcc.push(1);
        // PPS length
        avcc.extend_from_slice(&(self.pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&self.pps);

        avcc
    }

    /// Encode a frame as an I-frame (intra only).
    fn encode_idr(&mut self, frame: &Frame) -> Result<Packet<'static>> {
        // This is a simplified placeholder
        // Real implementation would perform actual encoding

        let mut data = Vec::new();

        // Write start code and IDR slice
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        data.push(0x65); // IDR slice NAL header

        // Placeholder slice data
        let qp = match self.config.rate_control {
            RateControlMode::Crf(crf) => crf,
            RateControlMode::Cqp(qp) => qp,
            _ => 26,
        };

        // Simple slice header
        let mut writer = BitWriter::new();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(7); // I_8x8
        writer.write_ue(0); // pps_id
        writer.write_bits(0, 4); // frame_num (4 bits for log2_max_frame_num=4)
        writer.write_ue(self.idr_count as u32); // idr_pic_id
        writer.write_bits(0, 8); // pic_order_cnt_lsb
        writer.write_bit(false); // no_output_of_prior_pics_flag
        writer.write_bit(false); // long_term_reference_flag
        if self.config.cabac {
            writer.write_ue(0); // cabac_init_idc
        }
        writer.write_se(qp as i32 - 26); // slice_qp_delta
        writer.write_ue(0); // disable_deblocking_filter_idc
        writer.write_se(0); // slice_alpha_c0_offset_div2
        writer.write_se(0); // slice_beta_offset_div2
        writer.write_rbsp_trailing_bits();

        data.extend(transcode_core::bitstream::add_emulation_prevention(writer.data()));

        self.idr_count += 1;

        let mut packet = Packet::new(data);
        packet.pts = frame.pts;
        packet.dts = frame.dts;
        packet.set_keyframe(true);

        Ok(packet)
    }

    /// Encode a frame using parallel slice encoding.
    fn encode_parallel_slices(&mut self, frame: &Frame, is_keyframe: bool) -> Result<Packet<'static>> {
        let slice_encoder = self.slice_encoder.as_ref()
            .ok_or_else(|| transcode_core::error::CodecError::Other(
                "Parallel slice encoder not initialized".into()
            ))?;

        let parallel_state = self.parallel_state.as_ref()
            .ok_or_else(|| transcode_core::error::CodecError::Other(
                "Parallel state not initialized".into()
            ))?;

        let mb_width = (self.config.width as usize).div_ceil(16);
        let mb_height = (self.config.height as usize).div_ceil(16);

        let qp = match self.config.rate_control {
            RateControlMode::Crf(crf) => crf,
            RateControlMode::Cqp(qp) => qp,
            _ => parallel_state.get_recommended_qp(26),
        };

        let context = SliceEncoderContext {
            mb_width,
            mb_height,
            base_qp: qp,
            is_intra: is_keyframe,
            cabac: self.config.cabac,
            reference_frames: parallel_state.get_reference_frames(),
        };

        // Encode slices in parallel
        let slices = slice_encoder.encode_slices_parallel(
            frame,
            &context,
            |frame, ctx, slice_idx, first_mb_row, mb_row_count| {
                Self::encode_slice_data(frame, ctx, slice_idx, first_mb_row, mb_row_count)
            },
        );

        // Merge slices into single NAL stream
        let mut data = Vec::new();

        // Add start code and NAL header
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        if is_keyframe {
            data.push(0x65); // IDR slice
        } else {
            data.push(0x41); // Non-IDR slice
        }

        // Merge encoded slice data
        let merged = slice_encoder.merge_slices(slices);
        data.extend(merged);

        // Update rate control state
        parallel_state.update_rate_control(data.len() as u32 * 8, qp);

        // Add reconstructed frame to reference frames if this is a reference frame
        if is_keyframe {
            parallel_state.clear_references();
            self.idr_count += 1;
        }

        let poc = (self.frame_count * 2) as i32;
        let ref_frame = ReferenceFrame::from_frame(frame, self.frame_count, poc);
        parallel_state.add_reference_frame(ref_frame);

        let mut packet = Packet::new(data);
        packet.pts = frame.pts;
        packet.dts = frame.dts;
        packet.set_keyframe(is_keyframe);

        Ok(packet)
    }

    /// Encode a single slice (called from parallel context).
    fn encode_slice_data(
        _frame: &Frame,
        context: &SliceEncoderContext,
        slice_idx: usize,
        first_mb_row: usize,
        mb_row_count: usize,
    ) -> SliceData {
        let mut writer = BitWriter::new();

        // Calculate first macroblock address
        let first_mb_in_slice = first_mb_row * context.mb_width;

        // Write slice header
        writer.write_ue(first_mb_in_slice as u32);
        if context.is_intra {
            writer.write_ue(7); // I slice
        } else {
            writer.write_ue(5); // P slice
        }
        writer.write_ue(0); // pps_id

        // Frame number (simplified)
        writer.write_bits(0, 4);

        if context.is_intra && slice_idx == 0 {
            writer.write_ue(0); // idr_pic_id
        }

        // POC
        writer.write_bits(0, 8);

        if !context.is_intra {
            writer.write_bit(false); // num_ref_idx_active_override_flag
            writer.write_bit(false); // ref_pic_list_modification_flag
            writer.write_bit(false); // adaptive_ref_pic_marking_mode_flag
        } else if slice_idx == 0 {
            writer.write_bit(false); // no_output_of_prior_pics_flag
            writer.write_bit(false); // long_term_reference_flag
        }

        if context.cabac && !context.is_intra {
            writer.write_ue(0); // cabac_init_idc
        }

        writer.write_se(context.base_qp as i32 - 26); // slice_qp_delta
        writer.write_ue(0); // disable_deblocking_filter_idc
        writer.write_se(0); // slice_alpha
        writer.write_se(0); // slice_beta

        writer.write_rbsp_trailing_bits();

        let encoded_data = transcode_core::bitstream::add_emulation_prevention(writer.data());

        SliceData {
            slice_idx,
            first_mb_row,
            mb_row_count,
            encoded_data,
            qp: context.base_qp,
        }
    }

    /// Perform parallel motion estimation on a frame.
    pub fn parallel_motion_estimation(&self, frame: &Frame) -> Vec<super::parallel::MotionEstimationResult> {
        if let (Some(estimator), Some(state)) = (&self.motion_estimator, &self.parallel_state) {
            estimator.estimate_motion(frame, &state.get_reference_frames())
        } else {
            Vec::new()
        }
    }

    /// Encode with lookahead for B-frame decisions (parallel frame-level).
    pub fn encode_with_lookahead(&mut self, frame: Frame) -> Result<Vec<Packet<'static>>> {
        if let Some(ref mut lookahead) = self.lookahead {
            // Push frame to lookahead buffer
            if let Some(ready_frames) = lookahead.push(frame) {
                // Encode ready frames
                let mut packets = Vec::new();
                for la_frame in ready_frames {
                    let packet = self.encode_lookahead_frame(&la_frame)?;
                    packets.push(packet);
                }
                return Ok(packets);
            }
            Ok(Vec::new())
        } else {
            // No lookahead, encode directly
            self.encode(&frame)
        }
    }

    /// Encode a frame from the lookahead buffer.
    fn encode_lookahead_frame(&mut self, la_frame: &LookaheadFrame) -> Result<Packet<'static>> {
        let is_keyframe = la_frame.frame_type == FrameType::I;

        if self.slice_encoder.is_some() {
            self.encode_parallel_slices(&la_frame.frame, is_keyframe)
        } else if is_keyframe {
            self.encode_idr(&la_frame.frame)
        } else {
            // Use existing P-frame encoding for non-keyframes
            let mut data = Vec::new();
            data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            data.push(0x41);

            let mut writer = BitWriter::new();
            writer.write_ue(0);
            writer.write_ue(5);
            writer.write_ue(0);
            writer.write_bits((self.frame_count % 16) as u32, 4);
            writer.write_bits((self.frame_count * 2 % 256) as u32, 8);
            writer.write_bit(false);
            writer.write_bit(false);
            writer.write_bit(false);
            if self.config.cabac {
                writer.write_ue(0);
            }
            writer.write_se(0);
            writer.write_ue(0);
            writer.write_se(0);
            writer.write_se(0);
            writer.write_rbsp_trailing_bits();

            data.extend(transcode_core::bitstream::add_emulation_prevention(writer.data()));

            let mut packet = Packet::new(data);
            packet.pts = la_frame.frame.pts;
            packet.dts = la_frame.frame.dts;
            Ok(packet)
        }
    }

    /// Flush the lookahead buffer at end of stream.
    pub fn flush_lookahead(&mut self) -> Result<Vec<Packet<'static>>> {
        if let Some(ref mut lookahead) = self.lookahead {
            let remaining = lookahead.flush();
            let mut packets = Vec::new();
            for la_frame in remaining {
                let packet = self.encode_lookahead_frame(&la_frame)?;
                packets.push(packet);
            }
            Ok(packets)
        } else {
            Ok(Vec::new())
        }
    }
}

impl VideoEncoder for H264Encoder {
    fn codec_info(&self) -> CodecInfo {
        CodecInfo {
            name: "h264",
            long_name: "H.264 / AVC / MPEG-4 AVC",
            can_encode: true,
            can_decode: false,
        }
    }

    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet<'static>>> {
        let mut packets = Vec::new();

        // Determine if this should be a keyframe
        let is_keyframe = self.frame_count.is_multiple_of(self.config.gop_size as u64);

        // Use parallel slice encoding if available
        if self.slice_encoder.is_some() {
            let packet = self.encode_parallel_slices(frame, is_keyframe)?;
            packets.push(packet);
            self.frame_count += 1;
            return Ok(packets);
        }

        // Fall back to single-threaded encoding
        if is_keyframe {
            // Prepend SPS and PPS for keyframes
            let mut sps_data = vec![0x00, 0x00, 0x00, 0x01];
            sps_data.extend_from_slice(&self.sps);
            let mut sps_packet = Packet::new(sps_data);
            sps_packet.pts = frame.pts;
            packets.push(sps_packet);

            let mut pps_data = vec![0x00, 0x00, 0x00, 0x01];
            pps_data.extend_from_slice(&self.pps);
            let mut pps_packet = Packet::new(pps_data);
            pps_packet.pts = frame.pts;
            packets.push(pps_packet);

            // Encode IDR frame
            let idr_packet = self.encode_idr(frame)?;
            packets.push(idr_packet);
        } else {
            // Encode P-frame (simplified)
            let mut data = Vec::new();
            data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            data.push(0x41); // Non-IDR slice NAL header

            let mut writer = BitWriter::new();
            writer.write_ue(0); // first_mb_in_slice
            writer.write_ue(5); // P_8x8
            writer.write_ue(0); // pps_id
            writer.write_bits((self.frame_count % 16) as u32, 4); // frame_num
            writer.write_bits((self.frame_count * 2 % 256) as u32, 8); // poc_lsb
            writer.write_bit(false); // num_ref_idx_active_override_flag
            writer.write_bit(false); // ref_pic_list_modification_flag
            writer.write_bit(false); // adaptive_ref_pic_marking_mode_flag
            if self.config.cabac {
                writer.write_ue(0);
            }
            writer.write_se(0); // slice_qp_delta
            writer.write_ue(0); // disable_deblocking_filter_idc
            writer.write_se(0);
            writer.write_se(0);
            writer.write_rbsp_trailing_bits();

            data.extend(transcode_core::bitstream::add_emulation_prevention(writer.data()));

            let mut packet = Packet::new(data);
            packet.pts = frame.pts;
            packet.dts = frame.dts;
            packets.push(packet);
        }

        self.frame_count += 1;
        Ok(packets)
    }

    fn flush(&mut self) -> Result<Vec<Packet<'static>>> {
        // Flush lookahead buffer if present
        let mut packets = self.flush_lookahead()?;

        // Return any pending frames
        packets.extend(self.pending_frames.drain(..).filter_map(|_| None::<Packet<'static>>));

        Ok(packets)
    }

    fn reset(&mut self) {
        self.frame_count = 0;
        self.idr_count = 0;
        self.pending_frames.clear();

        // Reset parallel state
        if let Some(ref state) = self.parallel_state {
            state.clear_references();
        }
    }

    fn extra_data(&self) -> Option<&[u8]> {
        if self.sps.is_empty() {
            None
        } else {
            // Return AVCC format
            None // Would need to store generated AVCC
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config() {
        let config = H264EncoderConfig::new(1920, 1080)
            .with_frame_rate(30)
            .with_crf(23);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
    }

    #[test]
    fn test_encoder_creation() {
        let config = H264EncoderConfig::new(1920, 1080);
        let encoder = H264Encoder::new(config).unwrap();
        assert!(!encoder.sps.is_empty());
        assert!(!encoder.pps.is_empty());
    }

    #[test]
    fn test_encoder_from_preset_youtube_1080p() {
        let encoder = H264Encoder::from_preset(Preset::YouTube1080p).unwrap();
        assert_eq!(encoder.config.width, 1920);
        assert_eq!(encoder.config.height, 1080);
        assert_eq!(encoder.config.profile, H264Profile::Main);
        assert!(!encoder.sps.is_empty());
        assert!(!encoder.pps.is_empty());
    }

    #[test]
    fn test_encoder_from_preset_streaming_low_latency() {
        let encoder = H264Encoder::from_preset(Preset::StreamingLowLatency).unwrap();
        assert_eq!(encoder.config.width, 1280);
        assert_eq!(encoder.config.height, 720);
        assert_eq!(encoder.config.profile, H264Profile::Baseline);
        assert_eq!(encoder.config.bframes, 0);
        assert!(!encoder.config.cabac);
    }

    #[test]
    fn test_encoder_from_preset_archive_high_quality() {
        let encoder = H264Encoder::from_preset(Preset::ArchiveHighQuality).unwrap();
        assert_eq!(encoder.config.profile, H264Profile::High);
        assert!(matches!(encoder.config.rate_control, RateControlMode::Crf(18)));
    }

    #[test]
    fn test_encoder_from_preset_with_dimensions() {
        let encoder = H264Encoder::from_preset_with_dimensions(
            Preset::YouTube1080p,
            3840,
            2160,
        ).unwrap();
        assert_eq!(encoder.config.width, 3840);
        assert_eq!(encoder.config.height, 2160);
        // Other settings should come from the preset
        assert_eq!(encoder.config.profile, H264Profile::Main);
    }

    #[test]
    fn test_encoder_from_all_presets() {
        use super::super::presets::all_presets;

        for preset in all_presets() {
            let encoder = H264Encoder::from_preset(preset);
            assert!(
                encoder.is_ok(),
                "Failed to create encoder from preset {:?}",
                preset
            );
            let encoder = encoder.unwrap();
            assert!(!encoder.sps.is_empty());
            assert!(!encoder.pps.is_empty());
        }
    }

    // Threading and parallel encoding tests

    #[test]
    fn test_threading_config() {
        let config = H264EncoderConfig::new(1920, 1080)
            .with_threads(4)
            .with_slice_count(4)
            .with_lookahead_depth(40);

        assert_eq!(config.threading.num_threads, 4);
        assert_eq!(config.threading.slice_count, 4);
        assert_eq!(config.threading.lookahead_depth, 40);
    }

    #[test]
    fn test_threading_auto_detect() {
        let config = H264EncoderConfig::new(1920, 1080);

        // Auto-detect should use system cores
        assert_eq!(config.threading.num_threads, 0);
        assert!(config.threading.effective_threads() >= 1);
    }

    #[test]
    fn test_single_threaded_encoder() {
        let encoder = H264Encoder::new_single_threaded(1280, 720).unwrap();
        assert!(!encoder.is_parallel());
        assert!(!encoder.sps.is_empty());
    }

    #[test]
    fn test_multi_threaded_encoder() {
        let encoder = H264Encoder::new_with_threads(1280, 720, 4).unwrap();
        assert!(encoder.is_parallel());
        assert_eq!(encoder.thread_count(), 4);
    }

    #[test]
    fn test_parallel_slice_encoding() {
        use transcode_core::{PixelFormat, TimeBase, Timestamp};

        let config = H264EncoderConfig::new(640, 480)
            .with_threads(2)
            .with_slice_count(2)
            .with_slice_parallel(true)
            .with_frame_parallel(false);

        let mut encoder = H264Encoder::new(config).unwrap();
        assert!(encoder.is_parallel());

        // Create a test frame
        let time_base = TimeBase::new(1, 30);
        let mut frame = transcode_core::Frame::new(640, 480, PixelFormat::Yuv420p, time_base);
        frame.pts = Timestamp::new(0, time_base);

        // Fill with test pattern
        if let Some(y_plane) = frame.plane_mut(0) {
            for (i, pixel) in y_plane.iter_mut().enumerate() {
                *pixel = (i % 256) as u8;
            }
        }
        if let Some(u_plane) = frame.plane_mut(1) {
            for pixel in u_plane.iter_mut() {
                *pixel = 128;
            }
        }
        if let Some(v_plane) = frame.plane_mut(2) {
            for pixel in v_plane.iter_mut() {
                *pixel = 128;
            }
        }

        // Encode the frame
        let packets = encoder.encode(&frame).unwrap();
        assert!(!packets.is_empty());
        assert!(packets[0].is_keyframe());
    }

    #[test]
    fn test_encoder_with_threading_config() {
        let threading = ThreadingConfig::with_threads(2)
            .with_slice_count(4)
            .with_lookahead_depth(20);

        let config = H264EncoderConfig::new(1280, 720)
            .with_threading(threading);

        assert_eq!(config.threading.num_threads, 2);
        assert_eq!(config.threading.slice_count, 4);
        assert_eq!(config.threading.lookahead_depth, 20);

        let encoder = H264Encoder::new(config).unwrap();
        assert!(encoder.is_parallel());
    }

    #[test]
    fn test_encoder_slice_parallel_only() {
        let config = H264EncoderConfig::new(1280, 720)
            .with_slice_parallel(true)
            .with_frame_parallel(false);

        let encoder = H264Encoder::new(config).unwrap();
        assert!(encoder.is_parallel());
        assert!(encoder.slice_encoder.is_some());
        assert!(encoder.motion_estimator.is_none());
    }

    #[test]
    fn test_encoder_frame_parallel_only() {
        let config = H264EncoderConfig::new(1280, 720)
            .with_slice_parallel(false)
            .with_frame_parallel(true);

        let encoder = H264Encoder::new(config).unwrap();
        assert!(encoder.is_parallel());
        assert!(encoder.slice_encoder.is_none());
        assert!(encoder.motion_estimator.is_some());
    }

    #[test]
    fn test_encoder_reset_clears_parallel_state() {
        use transcode_core::{PixelFormat, TimeBase, Timestamp};

        let config = H264EncoderConfig::new(640, 480)
            .with_threads(2);

        let mut encoder = H264Encoder::new(config).unwrap();

        // Create and encode a test frame
        let time_base = TimeBase::new(1, 30);
        let mut frame = transcode_core::Frame::new(640, 480, PixelFormat::Yuv420p, time_base);
        frame.pts = Timestamp::new(0, time_base);

        if let Some(y_plane) = frame.plane_mut(0) {
            for (i, pixel) in y_plane.iter_mut().enumerate() {
                *pixel = (i % 256) as u8;
            }
        }
        if let Some(u_plane) = frame.plane_mut(1) {
            for pixel in u_plane.iter_mut() {
                *pixel = 128;
            }
        }
        if let Some(v_plane) = frame.plane_mut(2) {
            for pixel in v_plane.iter_mut() {
                *pixel = 128;
            }
        }

        let _ = encoder.encode(&frame);

        // Reset should clear state
        encoder.reset();
        assert_eq!(encoder.frame_count, 0);
        assert_eq!(encoder.idr_count, 0);
    }

    #[test]
    fn test_motion_estimation_api() {
        use transcode_core::{PixelFormat, TimeBase, Timestamp};

        let config = H264EncoderConfig::new(320, 240)
            .with_frame_parallel(true)
            .with_slice_parallel(false);

        let encoder = H264Encoder::new(config).unwrap();

        // Create a test frame
        let time_base = TimeBase::new(1, 30);
        let mut frame = transcode_core::Frame::new(320, 240, PixelFormat::Yuv420p, time_base);
        frame.pts = Timestamp::new(0, time_base);

        if let Some(y_plane) = frame.plane_mut(0) {
            for (i, pixel) in y_plane.iter_mut().enumerate() {
                *pixel = (i % 256) as u8;
            }
        }
        if let Some(u_plane) = frame.plane_mut(1) {
            for pixel in u_plane.iter_mut() {
                *pixel = 128;
            }
        }
        if let Some(v_plane) = frame.plane_mut(2) {
            for pixel in v_plane.iter_mut() {
                *pixel = 128;
            }
        }

        // Motion estimation should work (returns empty when no references)
        let results = encoder.parallel_motion_estimation(&frame);
        // Should have one result per macroblock
        let mb_count = ((320 + 15) / 16) * ((240 + 15) / 16);
        assert_eq!(results.len(), mb_count);
    }
}
