//! VP8 video decoder.

use crate::bool_decoder::BoolDecoder;
use crate::error::{Result, Vp8Error};
use crate::frame::{Vp8Frame, Vp8FrameType};
use crate::loop_filter::{filter_frame, LoopFilterParams};
use crate::prediction::{
    predict_16x16, predict_4x4, predict_8x8_chroma,
    ChromaPredContext, Intra16x16Mode, Intra4x4Mode, IntraChromaMode,
    Pred4x4Context, PredictionContext,
};
use crate::transform::{dequantize, idct4x4, quant_tables};
use crate::{
    DEFAULT_COEFF_PROBS, DEFAULT_MV_PROBS, DEFAULT_Y_MODE_PROBS,
    MbLumaMode, MbChromaMode, SegmentFeature,
};

/// VP8 decoder configuration.
#[derive(Debug, Clone)]
pub struct Vp8DecoderConfig {
    /// Maximum width.
    pub max_width: u32,
    /// Maximum height.
    pub max_height: u32,
    /// Number of threads for parallel decoding.
    pub threads: usize,
    /// Enable error concealment.
    pub error_concealment: bool,
}

impl Default for Vp8DecoderConfig {
    fn default() -> Self {
        Self {
            max_width: 4096,
            max_height: 4096,
            threads: 1,
            error_concealment: false,
        }
    }
}

/// Decoded macroblock info.
#[derive(Debug, Clone, Default)]
struct MacroblockInfo {
    /// Luma prediction mode.
    y_mode: MbLumaMode,
    /// Chroma prediction mode.
    uv_mode: MbChromaMode,
    /// Segment ID.
    segment_id: u8,
    /// Skip coefficient flag.
    skip_coeff: bool,
    /// 4x4 prediction modes (for B_PRED).
    sub_modes: [Intra4x4Mode; 16],
}

/// VP8 video decoder.
pub struct Vp8Decoder {
    /// Decoder configuration.
    config: Vp8DecoderConfig,
    /// Current frame width.
    width: u32,
    /// Current frame height.
    height: u32,
    /// Last decoded frame (golden reference).
    last_frame: Option<Vp8Frame>,
    /// Golden reference frame.
    golden_frame: Option<Vp8Frame>,
    /// Alternate reference frame.
    altref_frame: Option<Vp8Frame>,
    /// Coefficient probabilities.
    coeff_probs: [[[u8; 11]; 3]; 8],
    /// Motion vector probabilities.
    mv_probs: [[u8; 19]; 2],
    /// Y mode probabilities.
    y_mode_probs: [u8; 4],
    /// UV mode probabilities.
    uv_mode_probs: [u8; 3],
    /// Segmentation features.
    segment_features: [SegmentFeature; 4],
    /// Loop filter parameters.
    loop_filter: LoopFilterParams,
    /// Macroblock info storage.
    mb_info: Vec<MacroblockInfo>,
    /// Number of macroblocks in width.
    mb_width: usize,
    /// Number of macroblocks in height.
    mb_height: usize,
}

impl Vp8Decoder {
    /// Create a new VP8 decoder.
    pub fn new() -> Self {
        Self::with_config(Vp8DecoderConfig::default())
    }

    /// Create a new VP8 decoder with configuration.
    pub fn with_config(config: Vp8DecoderConfig) -> Self {
        Self {
            config,
            width: 0,
            height: 0,
            last_frame: None,
            golden_frame: None,
            altref_frame: None,
            coeff_probs: DEFAULT_COEFF_PROBS,
            mv_probs: DEFAULT_MV_PROBS,
            y_mode_probs: DEFAULT_Y_MODE_PROBS,
            uv_mode_probs: [142, 114, 183],
            segment_features: [SegmentFeature::default(); 4],
            loop_filter: LoopFilterParams::default(),
            mb_info: Vec::new(),
            mb_width: 0,
            mb_height: 0,
        }
    }

    /// Decode a VP8 frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<Vp8Frame> {
        if data.len() < 3 {
            return Err(Vp8Error::InvalidBitstream("Data too short".into()));
        }

        // Parse uncompressed header
        let header = self.parse_uncompressed_header(data)?;

        // Validate dimensions
        if header.width > self.config.max_width || header.height > self.config.max_height {
            return Err(Vp8Error::InvalidDimensions {
                width: header.width,
                height: header.height,
            });
        }

        // Update decoder state
        self.width = header.width;
        self.height = header.height;
        self.mb_width = ((header.width as usize) + 15) / 16;
        self.mb_height = ((header.height as usize) + 15) / 16;
        self.mb_info.resize(self.mb_width * self.mb_height, MacroblockInfo::default());

        // Create output frame
        let mut frame = Vp8Frame::new(header.width, header.height, header.frame_type);

        // Parse compressed header and decode
        let first_partition_offset = header.first_partition_offset;
        if first_partition_offset >= data.len() {
            return Err(Vp8Error::InvalidBitstream("Invalid partition offset".into()));
        }

        let compressed_data = &data[first_partition_offset..];
        self.decode_compressed(&mut frame, compressed_data, &header)?;

        // Apply loop filter
        if self.loop_filter.level > 0 {
            filter_frame(
                &mut frame.y_plane,
                &mut frame.u_plane,
                &mut frame.v_plane,
                frame.width as usize,
                frame.height as usize,
                frame.y_stride,
                frame.uv_stride,
                &self.loop_filter,
            );
        }

        // Update reference frames
        if header.frame_type == Vp8FrameType::KeyFrame {
            self.last_frame = Some(frame.clone());
            self.golden_frame = Some(frame.clone());
            self.altref_frame = Some(frame.clone());
        } else {
            // Update based on flags (simplified)
            self.last_frame = Some(frame.clone());
        }

        Ok(frame)
    }

    /// Parse uncompressed frame header.
    fn parse_uncompressed_header(&self, data: &[u8]) -> Result<FrameHeader> {
        if data.len() < 10 {
            return Err(Vp8Error::InvalidFrameHeader("Header too short".into()));
        }

        // First 3 bytes contain frame tag
        let frame_tag = u32::from_le_bytes([data[0], data[1], data[2], 0]);

        let frame_type = if frame_tag & 1 == 0 {
            Vp8FrameType::KeyFrame
        } else {
            Vp8FrameType::InterFrame
        };

        let version = ((frame_tag >> 1) & 7) as u8;
        let show_frame = (frame_tag >> 4) & 1 != 0;
        let first_partition_size = (frame_tag >> 5) as usize;

        let (width, height, first_partition_offset) = if frame_type == Vp8FrameType::KeyFrame {
            // Key frame has sync code and dimensions
            if data.len() < 10 {
                return Err(Vp8Error::InvalidFrameHeader("Key frame header too short".into()));
            }

            // Check sync code
            if data[3] != 0x9D || data[4] != 0x01 || data[5] != 0x2A {
                return Err(Vp8Error::InvalidFrameHeader("Invalid sync code".into()));
            }

            // Parse dimensions
            let width_bytes = u16::from_le_bytes([data[6], data[7]]);
            let height_bytes = u16::from_le_bytes([data[8], data[9]]);

            let width = (width_bytes & 0x3FFF) as u32;
            let height = (height_bytes & 0x3FFF) as u32;

            (width, height, 10)
        } else {
            // Inter frame uses previous dimensions
            (self.width, self.height, 3)
        };

        if width == 0 || height == 0 {
            return Err(Vp8Error::InvalidDimensions { width, height });
        }

        Ok(FrameHeader {
            frame_type,
            version,
            show_frame,
            first_partition_size,
            first_partition_offset,
            width,
            height,
        })
    }

    /// Decode compressed frame data.
    fn decode_compressed(
        &mut self,
        frame: &mut Vp8Frame,
        data: &[u8],
        header: &FrameHeader,
    ) -> Result<()> {
        let mut decoder = BoolDecoder::new(data)?;

        // Parse frame header from bool decoder
        if header.frame_type == Vp8FrameType::KeyFrame {
            // Color space and clamping (ignored for key frames)
            let _color_space = decoder.read_bit()?;
            let _clamping = decoder.read_bit()?;
        }

        // Segmentation
        let segmentation_enabled = decoder.read_bit()?;
        if segmentation_enabled {
            self.parse_segmentation(&mut decoder)?;
        }

        // Loop filter
        self.loop_filter.simple = decoder.read_bit()?;
        self.loop_filter.level = decoder.read_bits(6)? as u8;
        self.loop_filter.sharpness = decoder.read_bits(3)? as u8;

        // Loop filter adjustments
        let lf_adj_enable = decoder.read_bit()?;
        if lf_adj_enable {
            let _lf_delta_update = decoder.read_bit()?;
            // Skip delta parsing for simplicity
        }

        // Number of partitions
        let log2_partitions = decoder.read_bits(2)?;
        let _num_partitions = 1 << log2_partitions;

        // Quantization
        let y_ac_qi = decoder.read_bits(7)? as u8;
        let y_dc_delta = if decoder.read_bit()? {
            decoder.read_signed_literal(4)?
        } else {
            0
        };
        let y2_dc_delta = if decoder.read_bit()? {
            decoder.read_signed_literal(4)?
        } else {
            0
        };
        let y2_ac_delta = if decoder.read_bit()? {
            decoder.read_signed_literal(4)?
        } else {
            0
        };
        let uv_dc_delta = if decoder.read_bit()? {
            decoder.read_signed_literal(4)?
        } else {
            0
        };
        let uv_ac_delta = if decoder.read_bit()? {
            decoder.read_signed_literal(4)?
        } else {
            0
        };

        // Compute quantizers
        let quant = QuantizerSet {
            y_ac: y_ac_qi,
            y_dc: (y_ac_qi as i32 + y_dc_delta).clamp(0, 127) as u8,
            y2_dc: (y_ac_qi as i32 + y2_dc_delta).clamp(0, 127) as u8,
            y2_ac: (y_ac_qi as i32 + y2_ac_delta).clamp(0, 127) as u8,
            uv_dc: (y_ac_qi as i32 + uv_dc_delta).clamp(0, 127) as u8,
            uv_ac: (y_ac_qi as i32 + uv_ac_delta).clamp(0, 127) as u8,
        };

        // Reference frame updates (inter frames)
        if header.frame_type == Vp8FrameType::InterFrame {
            let _refresh_golden = decoder.read_bit()?;
            let _refresh_altref = decoder.read_bit()?;
            // Skip copy flags
        }

        // Coefficient probability updates
        let refresh_probs = decoder.read_bit()?;
        if refresh_probs {
            self.parse_coeff_prob_updates(&mut decoder)?;
        }

        // Skip coeff flag
        let mb_no_skip_coeff = decoder.read_bit()?;
        let skip_prob = if mb_no_skip_coeff {
            decoder.read_bits(8)? as u8
        } else {
            0
        };

        // Inter frame mode probabilities
        if header.frame_type == Vp8FrameType::InterFrame {
            let _intra_prob = decoder.read_bits(8)?;
            let _last_prob = decoder.read_bits(8)?;
            let _golden_prob = decoder.read_bits(8)?;

            // Y mode probability updates
            if decoder.read_bit()? {
                for prob in &mut self.y_mode_probs {
                    *prob = decoder.read_bits(8)? as u8;
                }
            }

            // UV mode probability updates
            if decoder.read_bit()? {
                for prob in &mut self.uv_mode_probs {
                    *prob = decoder.read_bits(8)? as u8;
                }
            }

            // MV probability updates
            self.parse_mv_prob_updates(&mut decoder)?;
        }

        // Decode macroblocks
        for mb_y in 0..self.mb_height {
            for mb_x in 0..self.mb_width {
                self.decode_macroblock(
                    frame,
                    &mut decoder,
                    mb_x,
                    mb_y,
                    header.frame_type,
                    &quant,
                    skip_prob,
                )?;
            }
        }

        Ok(())
    }

    /// Parse segmentation data.
    fn parse_segmentation(&mut self, decoder: &mut BoolDecoder) -> Result<()> {
        let update_map = decoder.read_bit()?;
        let update_data = decoder.read_bit()?;

        if update_data {
            let abs_delta = decoder.read_bit()?;

            // Quantizer updates
            for seg in &mut self.segment_features {
                if decoder.read_bit()? {
                    let value = decoder.read_bits(7)? as i8;
                    let sign = decoder.read_bit()?;
                    seg.quant_update = if sign { -value } else { value };
                    seg.quant_absolute = abs_delta;
                }
            }

            // Loop filter updates
            for seg in &mut self.segment_features {
                if decoder.read_bit()? {
                    let value = decoder.read_bits(6)? as i8;
                    let sign = decoder.read_bit()?;
                    seg.lf_update = if sign { -value } else { value };
                    seg.lf_absolute = abs_delta;
                }
            }
        }

        if update_map {
            // Segment tree probabilities
            for _ in 0..3 {
                if decoder.read_bit()? {
                    let _prob = decoder.read_bits(8)?;
                }
            }
        }

        Ok(())
    }

    /// Parse coefficient probability updates.
    fn parse_coeff_prob_updates(&mut self, decoder: &mut BoolDecoder) -> Result<()> {
        for i in 0..4 {
            for j in 0..8 {
                for k in 0..3 {
                    for l in 0..11 {
                        if decoder.read_bool(252)? {
                            self.coeff_probs[j][k][l] = decoder.read_bits(8)? as u8;
                        }
                    }
                }
            }
            let _ = i; // Suppress warning
        }
        Ok(())
    }

    /// Parse MV probability updates.
    fn parse_mv_prob_updates(&mut self, decoder: &mut BoolDecoder) -> Result<()> {
        for comp in 0..2 {
            for i in 0..19 {
                if decoder.read_bool(252)? {
                    self.mv_probs[comp][i] = decoder.read_bits(7)? as u8;
                }
            }
        }
        Ok(())
    }

    /// Decode a single macroblock.
    fn decode_macroblock(
        &mut self,
        frame: &mut Vp8Frame,
        decoder: &mut BoolDecoder,
        mb_x: usize,
        mb_y: usize,
        frame_type: Vp8FrameType,
        quant: &QuantizerSet,
        skip_prob: u8,
    ) -> Result<()> {
        let mb_idx = mb_y * self.mb_width + mb_x;

        // Read skip flag
        let skip_coeff = if skip_prob > 0 {
            decoder.read_bool(skip_prob)?
        } else {
            false
        };
        self.mb_info[mb_idx].skip_coeff = skip_coeff;

        // For key frames, always intra
        if frame_type == Vp8FrameType::KeyFrame {
            self.decode_intra_macroblock(frame, decoder, mb_x, mb_y, quant, skip_coeff)?;
        } else {
            // Inter frame - check if intra
            let is_intra = decoder.read_bool(128)?; // Simplified
            if is_intra {
                self.decode_intra_macroblock(frame, decoder, mb_x, mb_y, quant, skip_coeff)?;
            } else {
                self.decode_inter_macroblock(frame, decoder, mb_x, mb_y, quant, skip_coeff)?;
            }
        }

        Ok(())
    }

    /// Decode an intra macroblock.
    fn decode_intra_macroblock(
        &mut self,
        frame: &mut Vp8Frame,
        decoder: &mut BoolDecoder,
        mb_x: usize,
        mb_y: usize,
        quant: &QuantizerSet,
        skip_coeff: bool,
    ) -> Result<()> {
        let mb_idx = mb_y * self.mb_width + mb_x;

        // Read Y mode
        let y_mode = self.read_y_mode(decoder)?;
        self.mb_info[mb_idx].y_mode = y_mode;

        // Read UV mode
        let uv_mode = self.read_uv_mode(decoder)?;
        self.mb_info[mb_idx].uv_mode = uv_mode;

        // Build prediction context
        let pred_ctx = PredictionContext::from_frame(frame, mb_x, mb_y);

        // Decode Y plane
        if y_mode == MbLumaMode::BPred {
            // 4x4 prediction mode
            for sub_y in 0..4 {
                for sub_x in 0..4 {
                    let sub_idx = sub_y * 4 + sub_x;
                    let sub_mode = self.read_sub_mode(decoder)?;
                    self.mb_info[mb_idx].sub_modes[sub_idx] = sub_mode;

                    // Build 4x4 context
                    let sub_ctx = self.build_4x4_context(frame, mb_x, mb_y, sub_x, sub_y);

                    // Predict
                    let mut prediction = [0u8; 16];
                    predict_4x4(sub_mode, &sub_ctx, &mut prediction);

                    // Decode residual
                    let mut residual = [0i16; 16];
                    if !skip_coeff {
                        self.decode_block_coeffs(decoder, &mut residual, false)?;
                        dequantize(
                            &mut residual,
                            quant_tables::get_dc_quant(quant.y_dc),
                            quant_tables::get_ac_quant(quant.y_ac),
                        );
                        let mut transformed = [0i16; 16];
                        idct4x4(&residual, &mut transformed);
                        residual = transformed;
                    }

                    // Add residual to prediction
                    let px = mb_x * 16 + sub_x * 4;
                    let py = mb_y * 16 + sub_y * 4;
                    self.add_4x4_block(frame, px, py, &prediction, &residual);
                }
            }
        } else {
            // 16x16 prediction mode
            let mut prediction = [0u8; 256];
            let mode_16x16 = match y_mode {
                MbLumaMode::DcPred => Intra16x16Mode::DcPred,
                MbLumaMode::VPred => Intra16x16Mode::VPred,
                MbLumaMode::HPred => Intra16x16Mode::HPred,
                MbLumaMode::TmPred => Intra16x16Mode::TmPred,
                _ => Intra16x16Mode::DcPred,
            };
            predict_16x16(mode_16x16, &pred_ctx, &mut prediction);

            // Decode and add residuals for each 4x4 block
            for sub_y in 0..4 {
                for sub_x in 0..4 {
                    let mut residual = [0i16; 16];
                    if !skip_coeff {
                        self.decode_block_coeffs(decoder, &mut residual, false)?;
                        dequantize(
                            &mut residual,
                            quant_tables::get_dc_quant(quant.y_dc),
                            quant_tables::get_ac_quant(quant.y_ac),
                        );
                        let mut transformed = [0i16; 16];
                        idct4x4(&residual, &mut transformed);
                        residual = transformed;
                    }

                    // Extract 4x4 prediction block
                    let mut pred_4x4 = [0u8; 16];
                    for y in 0..4 {
                        for x in 0..4 {
                            pred_4x4[y * 4 + x] = prediction[(sub_y * 4 + y) * 16 + sub_x * 4 + x];
                        }
                    }

                    let px = mb_x * 16 + sub_x * 4;
                    let py = mb_y * 16 + sub_y * 4;
                    self.add_4x4_block(frame, px, py, &pred_4x4, &residual);
                }
            }
        }

        // Decode UV planes
        self.decode_uv_planes(frame, decoder, mb_x, mb_y, uv_mode, quant, skip_coeff)?;

        Ok(())
    }

    /// Decode an inter macroblock.
    fn decode_inter_macroblock(
        &mut self,
        frame: &mut Vp8Frame,
        _decoder: &mut BoolDecoder,
        mb_x: usize,
        mb_y: usize,
        _quant: &QuantizerSet,
        _skip_coeff: bool,
    ) -> Result<()> {
        // Simplified inter prediction - just copy from last frame
        if let Some(ref last) = self.last_frame {
            // Copy Y plane
            for y in 0..16 {
                for x in 0..16 {
                    let px = mb_x * 16 + x;
                    let py = mb_y * 16 + y;
                    if px < frame.width as usize && py < frame.height as usize {
                        let val = last.get_y(px, py);
                        frame.set_y(px, py, val);
                    }
                }
            }

            // Copy U plane
            for y in 0..8 {
                for x in 0..8 {
                    let px = mb_x * 8 + x;
                    let py = mb_y * 8 + y;
                    if px < (frame.width as usize + 1) / 2 && py < (frame.height as usize + 1) / 2 {
                        let val = last.get_u(px, py);
                        frame.set_u(px, py, val);
                    }
                }
            }

            // Copy V plane
            for y in 0..8 {
                for x in 0..8 {
                    let px = mb_x * 8 + x;
                    let py = mb_y * 8 + y;
                    if px < (frame.width as usize + 1) / 2 && py < (frame.height as usize + 1) / 2 {
                        let val = last.get_v(px, py);
                        frame.set_v(px, py, val);
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode UV (chroma) planes.
    fn decode_uv_planes(
        &mut self,
        frame: &mut Vp8Frame,
        decoder: &mut BoolDecoder,
        mb_x: usize,
        mb_y: usize,
        uv_mode: MbChromaMode,
        quant: &QuantizerSet,
        skip_coeff: bool,
    ) -> Result<()> {
        // Build chroma prediction context
        let u_ctx = self.build_chroma_context(frame, mb_x, mb_y, true);
        let v_ctx = self.build_chroma_context(frame, mb_x, mb_y, false);

        let mode = match uv_mode {
            MbChromaMode::DcPred => IntraChromaMode::DcPred,
            MbChromaMode::VPred => IntraChromaMode::VPred,
            MbChromaMode::HPred => IntraChromaMode::HPred,
            MbChromaMode::TmPred => IntraChromaMode::TmPred,
        };

        // Predict U plane
        let mut u_pred = [0u8; 64];
        predict_8x8_chroma(mode, &u_ctx, &mut u_pred);

        // Predict V plane
        let mut v_pred = [0u8; 64];
        predict_8x8_chroma(mode, &v_ctx, &mut v_pred);

        // Decode residuals and reconstruct
        for sub_y in 0..2 {
            for sub_x in 0..2 {
                // U residual
                let mut u_residual = [0i16; 16];
                if !skip_coeff {
                    self.decode_block_coeffs(decoder, &mut u_residual, true)?;
                    dequantize(
                        &mut u_residual,
                        quant_tables::get_dc_quant(quant.uv_dc),
                        quant_tables::get_ac_quant(quant.uv_ac),
                    );
                    let mut transformed = [0i16; 16];
                    idct4x4(&u_residual, &mut transformed);
                    u_residual = transformed;
                }

                // Extract 4x4 U prediction
                let mut u_pred_4x4 = [0u8; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        u_pred_4x4[y * 4 + x] = u_pred[(sub_y * 4 + y) * 8 + sub_x * 4 + x];
                    }
                }

                // Add U block
                let px = mb_x * 8 + sub_x * 4;
                let py = mb_y * 8 + sub_y * 4;
                self.add_4x4_chroma_block(frame, px, py, &u_pred_4x4, &u_residual, true);

                // V residual
                let mut v_residual = [0i16; 16];
                if !skip_coeff {
                    self.decode_block_coeffs(decoder, &mut v_residual, true)?;
                    dequantize(
                        &mut v_residual,
                        quant_tables::get_dc_quant(quant.uv_dc),
                        quant_tables::get_ac_quant(quant.uv_ac),
                    );
                    let mut transformed = [0i16; 16];
                    idct4x4(&v_residual, &mut transformed);
                    v_residual = transformed;
                }

                // Extract 4x4 V prediction
                let mut v_pred_4x4 = [0u8; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        v_pred_4x4[y * 4 + x] = v_pred[(sub_y * 4 + y) * 8 + sub_x * 4 + x];
                    }
                }

                // Add V block
                self.add_4x4_chroma_block(frame, px, py, &v_pred_4x4, &v_residual, false);
            }
        }

        Ok(())
    }

    /// Read Y prediction mode.
    fn read_y_mode(&self, decoder: &mut BoolDecoder) -> Result<MbLumaMode> {
        let mode = if !decoder.read_bool(self.y_mode_probs[0])? {
            MbLumaMode::DcPred
        } else if !decoder.read_bool(self.y_mode_probs[1])? {
            MbLumaMode::VPred
        } else if !decoder.read_bool(self.y_mode_probs[2])? {
            MbLumaMode::HPred
        } else if !decoder.read_bool(self.y_mode_probs[3])? {
            MbLumaMode::TmPred
        } else {
            MbLumaMode::BPred
        };
        Ok(mode)
    }

    /// Read UV prediction mode.
    fn read_uv_mode(&self, decoder: &mut BoolDecoder) -> Result<MbChromaMode> {
        let mode = if !decoder.read_bool(self.uv_mode_probs[0])? {
            MbChromaMode::DcPred
        } else if !decoder.read_bool(self.uv_mode_probs[1])? {
            MbChromaMode::VPred
        } else if !decoder.read_bool(self.uv_mode_probs[2])? {
            MbChromaMode::HPred
        } else {
            MbChromaMode::TmPred
        };
        Ok(mode)
    }

    /// Read sub-block prediction mode.
    fn read_sub_mode(&self, decoder: &mut BoolDecoder) -> Result<Intra4x4Mode> {
        // Simplified - just read 4 bits
        let mode_val = decoder.read_bits(4)? as u8;
        Ok(Intra4x4Mode::from_raw(mode_val.min(9)).unwrap_or(Intra4x4Mode::DcPred))
    }

    /// Build 4x4 prediction context.
    fn build_4x4_context(
        &self,
        frame: &Vp8Frame,
        mb_x: usize,
        mb_y: usize,
        sub_x: usize,
        sub_y: usize,
    ) -> Pred4x4Context {
        let mut ctx = Pred4x4Context::default();

        let px = mb_x * 16 + sub_x * 4;
        let py = mb_y * 16 + sub_y * 4;

        // Above pixels
        ctx.above_available = py > 0;
        if ctx.above_available {
            for i in 0..8 {
                let x = (px + i).min(frame.width as usize - 1);
                ctx.above[i] = frame.get_y(x, py - 1);
            }
        }

        // Left pixels
        ctx.left_available = px > 0;
        if ctx.left_available {
            for i in 0..4 {
                let y = (py + i).min(frame.height as usize - 1);
                ctx.left[i] = frame.get_y(px - 1, y);
            }
        }

        // Above-left
        if ctx.above_available && ctx.left_available {
            ctx.above_left = frame.get_y(px - 1, py - 1);
        }

        // Above-right available
        ctx.above_right_available = ctx.above_available && (px + 4) < frame.width as usize;

        ctx
    }

    /// Build chroma prediction context.
    fn build_chroma_context(
        &self,
        frame: &Vp8Frame,
        mb_x: usize,
        mb_y: usize,
        is_u: bool,
    ) -> ChromaPredContext {
        let mut ctx = ChromaPredContext::default();

        let px = mb_x * 8;
        let py = mb_y * 8;
        let width = (frame.width as usize + 1) / 2;
        let height = (frame.height as usize + 1) / 2;

        // Above pixels
        ctx.above_available = py > 0;
        if ctx.above_available {
            for i in 0..8 {
                let x = (px + i).min(width - 1);
                ctx.above[i] = if is_u {
                    frame.get_u(x, py - 1)
                } else {
                    frame.get_v(x, py - 1)
                };
            }
        }

        // Left pixels
        ctx.left_available = px > 0;
        if ctx.left_available {
            for i in 0..8 {
                let y = (py + i).min(height - 1);
                ctx.left[i] = if is_u {
                    frame.get_u(px - 1, y)
                } else {
                    frame.get_v(px - 1, y)
                };
            }
        }

        // Above-left
        if ctx.above_available && ctx.left_available {
            ctx.above_left = if is_u {
                frame.get_u(px - 1, py - 1)
            } else {
                frame.get_v(px - 1, py - 1)
            };
        }

        ctx
    }

    /// Decode block coefficients.
    fn decode_block_coeffs(
        &self,
        decoder: &mut BoolDecoder,
        coeffs: &mut [i16; 16],
        _is_chroma: bool,
    ) -> Result<()> {
        // Simplified coefficient decoding
        // In a full implementation, this would use the coefficient probability tables
        coeffs.fill(0);

        // Just decode a few coefficients for demonstration
        for i in 0..16 {
            if decoder.read_bool(200)? {
                break; // End of block
            }

            let magnitude = decoder.read_bits(4)? as i16;
            let sign = decoder.read_bit()?;
            coeffs[i] = if sign { -magnitude } else { magnitude };
        }

        Ok(())
    }

    /// Add 4x4 block to Y plane.
    fn add_4x4_block(
        &self,
        frame: &mut Vp8Frame,
        px: usize,
        py: usize,
        prediction: &[u8; 16],
        residual: &[i16; 16],
    ) {
        for y in 0..4 {
            for x in 0..4 {
                let fx = px + x;
                let fy = py + y;
                if fx < frame.width as usize && fy < frame.height as usize {
                    let pred = prediction[y * 4 + x] as i16;
                    let res = residual[y * 4 + x];
                    let val = (pred + res).clamp(0, 255) as u8;
                    frame.set_y(fx, fy, val);
                }
            }
        }
    }

    /// Add 4x4 block to chroma plane.
    fn add_4x4_chroma_block(
        &self,
        frame: &mut Vp8Frame,
        px: usize,
        py: usize,
        prediction: &[u8; 16],
        residual: &[i16; 16],
        is_u: bool,
    ) {
        let width = (frame.width as usize + 1) / 2;
        let height = (frame.height as usize + 1) / 2;

        for y in 0..4 {
            for x in 0..4 {
                let fx = px + x;
                let fy = py + y;
                if fx < width && fy < height {
                    let pred = prediction[y * 4 + x] as i16;
                    let res = residual[y * 4 + x];
                    let val = (pred + res).clamp(0, 255) as u8;
                    if is_u {
                        frame.set_u(fx, fy, val);
                    } else {
                        frame.set_v(fx, fy, val);
                    }
                }
            }
        }
    }

    /// Get decoder configuration.
    pub fn config(&self) -> &Vp8DecoderConfig {
        &self.config
    }

    /// Get current frame dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        self.last_frame = None;
        self.golden_frame = None;
        self.altref_frame = None;
        self.coeff_probs = DEFAULT_COEFF_PROBS;
        self.mv_probs = DEFAULT_MV_PROBS;
        self.y_mode_probs = DEFAULT_Y_MODE_PROBS;
        self.mb_info.clear();
    }
}

impl Default for Vp8Decoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Frame header information.
#[derive(Debug)]
struct FrameHeader {
    frame_type: Vp8FrameType,
    version: u8,
    show_frame: bool,
    first_partition_size: usize,
    first_partition_offset: usize,
    width: u32,
    height: u32,
}

/// Quantizer set for a frame.
#[derive(Debug, Clone, Copy)]
struct QuantizerSet {
    y_ac: u8,
    y_dc: u8,
    y2_dc: u8,
    y2_ac: u8,
    uv_dc: u8,
    uv_ac: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = Vp8Decoder::new();
        assert_eq!(decoder.dimensions(), (0, 0));
    }

    #[test]
    fn test_decoder_config() {
        let config = Vp8DecoderConfig {
            max_width: 1920,
            max_height: 1080,
            threads: 4,
            error_concealment: true,
        };
        let decoder = Vp8Decoder::with_config(config);
        assert_eq!(decoder.config().max_width, 1920);
        assert_eq!(decoder.config().max_height, 1080);
    }

    #[test]
    fn test_invalid_data() {
        let mut decoder = Vp8Decoder::new();
        let result = decoder.decode(&[0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let mut decoder = Vp8Decoder::new();
        decoder.reset();
        assert_eq!(decoder.dimensions(), (0, 0));
    }
}
