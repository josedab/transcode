//! VP8 video encoder.

use crate::bool_decoder::BoolEncoder;
use crate::error::{Result, Vp8Error};
use crate::frame::{Vp8Frame, Vp8FrameType};
use crate::prediction::{
    predict_16x16, predict_8x8_chroma,
    ChromaPredContext, Intra16x16Mode, Intra4x4Mode, IntraChromaMode,
    PredictionContext,
};
use crate::transform::{fdct4x4, quant_tables, sub_prediction};
use crate::{MbLumaMode, MbChromaMode};

/// VP8 encoder rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant quality (CQ) mode.
    ConstantQuality,
    /// Variable bitrate (VBR) mode.
    Vbr,
    /// Constant bitrate (CBR) mode.
    Cbr,
}

impl Default for RateControlMode {
    fn default() -> Self {
        Self::ConstantQuality
    }
}

/// VP8 encoder configuration.
#[derive(Debug, Clone)]
pub struct Vp8EncoderConfig {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Target bitrate in kbps (for VBR/CBR).
    pub bitrate: u32,
    /// Quantizer (0-63, lower = better quality).
    pub quantizer: u8,
    /// Key frame interval.
    pub keyframe_interval: u32,
    /// Rate control mode.
    pub rate_control: RateControlMode,
    /// Number of token partitions (0-3).
    pub token_partitions: u8,
    /// Enable noise sensitivity.
    pub noise_sensitivity: u8,
    /// Sharpness level (0-7).
    pub sharpness: u8,
    /// Loop filter level.
    pub filter_level: u8,
    /// Enable error resilient mode.
    pub error_resilient: bool,
    /// Number of threads.
    pub threads: usize,
    /// Speed preset (0-16, higher = faster).
    pub speed: u8,
}

impl Default for Vp8EncoderConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            bitrate: 1000,
            quantizer: 32,
            keyframe_interval: 250,
            rate_control: RateControlMode::default(),
            token_partitions: 0,
            noise_sensitivity: 0,
            sharpness: 0,
            filter_level: 28,
            error_resilient: false,
            threads: 1,
            speed: 8,
        }
    }
}

impl Vp8EncoderConfig {
    /// Create a new encoder config with dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ..Default::default()
        }
    }

    /// Set bitrate.
    pub fn bitrate(mut self, kbps: u32) -> Self {
        self.bitrate = kbps;
        self
    }

    /// Set quantizer.
    pub fn quantizer(mut self, q: u8) -> Self {
        self.quantizer = q.min(63);
        self
    }

    /// Set keyframe interval.
    pub fn keyframe_interval(mut self, interval: u32) -> Self {
        self.keyframe_interval = interval;
        self
    }

    /// Set rate control mode.
    pub fn rate_control(mut self, mode: RateControlMode) -> Self {
        self.rate_control = mode;
        self
    }

    /// Set speed preset.
    pub fn speed(mut self, speed: u8) -> Self {
        self.speed = speed.min(16);
        self
    }
}

/// Encoded packet output.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Frame type.
    pub frame_type: Vp8FrameType,
    /// Presentation timestamp.
    pub pts: i64,
    /// Duration.
    pub duration: i64,
    /// Is this a key frame?
    pub is_keyframe: bool,
}

/// Macroblock encoding info.
#[derive(Debug, Clone, Default)]
struct MbEncodeInfo {
    /// Best luma mode.
    y_mode: MbLumaMode,
    /// Best chroma mode.
    uv_mode: MbChromaMode,
    /// Sub-block modes (for B_PRED).
    sub_modes: [Intra4x4Mode; 16],
    /// Skip coefficient flag.
    skip_coeff: bool,
    /// Encoded cost/distortion.
    cost: u64,
}

/// VP8 video encoder.
pub struct Vp8Encoder {
    /// Encoder configuration.
    config: Vp8EncoderConfig,
    /// Frame counter.
    frame_count: u64,
    /// Last encoded frame.
    last_frame: Option<Vp8Frame>,
    /// Golden reference frame.
    golden_frame: Option<Vp8Frame>,
    /// Alternate reference frame.
    altref_frame: Option<Vp8Frame>,
    /// Macroblock info storage.
    mb_info: Vec<MbEncodeInfo>,
    /// Number of macroblocks in width.
    mb_width: usize,
    /// Number of macroblocks in height.
    mb_height: usize,
    /// Current quantizer.
    current_q: u8,
}

impl Vp8Encoder {
    /// Create a new VP8 encoder.
    pub fn new(config: Vp8EncoderConfig) -> Result<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(Vp8Error::InvalidDimensions {
                width: config.width,
                height: config.height,
            });
        }

        let mb_width = ((config.width as usize) + 15) / 16;
        let mb_height = ((config.height as usize) + 15) / 16;

        Ok(Self {
            config,
            frame_count: 0,
            last_frame: None,
            golden_frame: None,
            altref_frame: None,
            mb_info: vec![MbEncodeInfo::default(); mb_width * mb_height],
            mb_width,
            mb_height,
            current_q: 32,
        })
    }

    /// Encode a frame.
    pub fn encode(&mut self, frame: &Vp8Frame) -> Result<EncodedPacket> {
        // Validate dimensions
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(Vp8Error::InvalidDimensions {
                width: frame.width,
                height: frame.height,
            });
        }

        // Determine frame type
        let is_keyframe = self.frame_count == 0
            || self.frame_count % self.config.keyframe_interval as u64 == 0;

        let frame_type = if is_keyframe {
            Vp8FrameType::KeyFrame
        } else {
            Vp8FrameType::InterFrame
        };

        // Update quantizer based on rate control
        self.update_rate_control();

        // Encode the frame
        let data = self.encode_frame_data(frame, frame_type)?;

        // Update reference frames
        if is_keyframe {
            self.last_frame = Some(frame.clone());
            self.golden_frame = Some(frame.clone());
            self.altref_frame = Some(frame.clone());
        } else {
            self.last_frame = Some(frame.clone());
        }

        let packet = EncodedPacket {
            data,
            frame_type,
            pts: frame.pts,
            duration: 1,
            is_keyframe,
        };

        self.frame_count += 1;

        Ok(packet)
    }

    /// Encode frame data.
    fn encode_frame_data(&mut self, frame: &Vp8Frame, frame_type: Vp8FrameType) -> Result<Vec<u8>> {
        let mut output = Vec::new();

        // Write uncompressed header
        self.write_uncompressed_header(&mut output, frame_type)?;

        // Create bool encoder for compressed data
        let mut encoder = BoolEncoder::new();

        // Write compressed header
        self.write_compressed_header(&mut encoder, frame_type)?;

        // Encode macroblocks
        for mb_y in 0..self.mb_height {
            for mb_x in 0..self.mb_width {
                self.encode_macroblock(frame, &mut encoder, mb_x, mb_y, frame_type)?;
            }
        }

        // Finalize bool encoder
        let compressed = encoder.finalize();

        // Update partition size in header
        let partition_size = compressed.len() as u32;
        let frame_tag = if frame_type == Vp8FrameType::KeyFrame { 0u32 } else { 1u32 }
            | (0 << 1)  // version
            | (1 << 4)  // show_frame
            | ((partition_size as u32) << 5);

        output[0] = frame_tag as u8;
        output[1] = (frame_tag >> 8) as u8;
        output[2] = (frame_tag >> 16) as u8;

        // Append compressed data
        output.extend_from_slice(&compressed);

        Ok(output)
    }

    /// Write uncompressed frame header.
    fn write_uncompressed_header(
        &self,
        output: &mut Vec<u8>,
        frame_type: Vp8FrameType,
    ) -> Result<()> {
        // Frame tag (will be updated later with partition size)
        output.extend_from_slice(&[0, 0, 0]);

        if frame_type == Vp8FrameType::KeyFrame {
            // Sync code
            output.extend_from_slice(&[0x9D, 0x01, 0x2A]);

            // Width and height
            let width = self.config.width as u16;
            let height = self.config.height as u16;

            output.push(width as u8);
            output.push((width >> 8) as u8);
            output.push(height as u8);
            output.push((height >> 8) as u8);
        }

        Ok(())
    }

    /// Write compressed frame header.
    fn write_compressed_header(
        &self,
        encoder: &mut BoolEncoder,
        frame_type: Vp8FrameType,
    ) -> Result<()> {
        if frame_type == Vp8FrameType::KeyFrame {
            // Color space and clamping
            encoder.write_bit(false); // color space = 0
            encoder.write_bit(false); // clamping = 0
        }

        // Segmentation disabled
        encoder.write_bit(false);

        // Loop filter parameters
        encoder.write_bit(false); // simple filter
        encoder.write_bits(self.config.filter_level as u32, 6);
        encoder.write_bits(self.config.sharpness as u32, 3);

        // Loop filter adjustments disabled
        encoder.write_bit(false);

        // Number of token partitions
        encoder.write_bits(self.config.token_partitions as u32, 2);

        // Quantization parameters
        encoder.write_bits(self.current_q as u32, 7);
        encoder.write_bit(false); // y_dc_delta_present
        encoder.write_bit(false); // y2_dc_delta_present
        encoder.write_bit(false); // y2_ac_delta_present
        encoder.write_bit(false); // uv_dc_delta_present
        encoder.write_bit(false); // uv_ac_delta_present

        if frame_type == Vp8FrameType::InterFrame {
            // Reference frame update flags
            encoder.write_bit(true);  // refresh_golden
            encoder.write_bit(true);  // refresh_altref
        }

        // No coefficient probability updates
        encoder.write_bit(false);

        // Skip coeff flag
        encoder.write_bit(true);
        encoder.write_bits(128, 8); // skip_prob

        if frame_type == Vp8FrameType::InterFrame {
            // Intra probability
            encoder.write_bits(63, 8);
            // Last probability
            encoder.write_bits(128, 8);
            // Golden probability
            encoder.write_bits(128, 8);

            // No Y mode prob update
            encoder.write_bit(false);
            // No UV mode prob update
            encoder.write_bit(false);
            // No MV prob updates
            encoder.write_bit(false);
            encoder.write_bit(false);
        }

        Ok(())
    }

    /// Encode a single macroblock.
    fn encode_macroblock(
        &mut self,
        frame: &Vp8Frame,
        encoder: &mut BoolEncoder,
        mb_x: usize,
        mb_y: usize,
        frame_type: Vp8FrameType,
    ) -> Result<()> {
        let mb_idx = mb_y * self.mb_width + mb_x;

        // Find best prediction mode
        let (y_mode, uv_mode) = self.select_intra_modes(frame, mb_x, mb_y);
        self.mb_info[mb_idx].y_mode = y_mode;
        self.mb_info[mb_idx].uv_mode = uv_mode;

        // For key frames, always use intra prediction
        if frame_type == Vp8FrameType::KeyFrame {
            // Write skip flag
            encoder.write_bool(false, 128); // Don't skip

            // Write Y mode
            self.write_y_mode(encoder, y_mode);

            // Write UV mode
            self.write_uv_mode(encoder, uv_mode);

            // Encode Y residuals
            self.encode_y_residuals(frame, encoder, mb_x, mb_y, y_mode)?;

            // Encode UV residuals
            self.encode_uv_residuals(frame, encoder, mb_x, mb_y, uv_mode)?;
        } else {
            // Inter frame - simplified to just encode as intra
            encoder.write_bool(true, 128); // is_intra

            // Write modes
            self.write_y_mode(encoder, y_mode);
            self.write_uv_mode(encoder, uv_mode);

            // Encode residuals
            self.encode_y_residuals(frame, encoder, mb_x, mb_y, y_mode)?;
            self.encode_uv_residuals(frame, encoder, mb_x, mb_y, uv_mode)?;
        }

        Ok(())
    }

    /// Select best intra prediction modes.
    fn select_intra_modes(
        &self,
        frame: &Vp8Frame,
        mb_x: usize,
        mb_y: usize,
    ) -> (MbLumaMode, MbChromaMode) {
        // Build prediction context
        let ctx = PredictionContext::from_frame(frame, mb_x, mb_y);

        // Test all 16x16 modes and find best
        let modes = [
            (Intra16x16Mode::DcPred, MbLumaMode::DcPred),
            (Intra16x16Mode::VPred, MbLumaMode::VPred),
            (Intra16x16Mode::HPred, MbLumaMode::HPred),
            (Intra16x16Mode::TmPred, MbLumaMode::TmPred),
        ];

        let mut best_cost = u64::MAX;
        let mut best_y_mode = MbLumaMode::DcPred;

        for (pred_mode, mb_mode) in modes {
            // Skip modes that require unavailable pixels
            if pred_mode == Intra16x16Mode::VPred && !ctx.above_available {
                continue;
            }
            if pred_mode == Intra16x16Mode::HPred && !ctx.left_available {
                continue;
            }

            let mut prediction = [0u8; 256];
            predict_16x16(pred_mode, &ctx, &mut prediction);

            // Calculate SAD (Sum of Absolute Differences)
            let cost = self.calculate_sad_16x16(frame, mb_x, mb_y, &prediction);

            if cost < best_cost {
                best_cost = cost;
                best_y_mode = mb_mode;
            }
        }

        // For UV, just use DC prediction (simplified)
        let uv_mode = MbChromaMode::DcPred;

        (best_y_mode, uv_mode)
    }

    /// Calculate SAD for 16x16 block.
    fn calculate_sad_16x16(
        &self,
        frame: &Vp8Frame,
        mb_x: usize,
        mb_y: usize,
        prediction: &[u8; 256],
    ) -> u64 {
        let mut sad = 0u64;

        for y in 0..16 {
            for x in 0..16 {
                let px = mb_x * 16 + x;
                let py = mb_y * 16 + y;

                if px < frame.width as usize && py < frame.height as usize {
                    let orig = frame.get_y(px, py) as i32;
                    let pred = prediction[y * 16 + x] as i32;
                    sad += (orig - pred).unsigned_abs() as u64;
                }
            }
        }

        sad
    }

    /// Write Y prediction mode.
    fn write_y_mode(&self, encoder: &mut BoolEncoder, mode: MbLumaMode) {
        // Simplified mode encoding
        match mode {
            MbLumaMode::DcPred => {
                encoder.write_bool(false, 145);
            }
            MbLumaMode::VPred => {
                encoder.write_bool(true, 145);
                encoder.write_bool(false, 156);
            }
            MbLumaMode::HPred => {
                encoder.write_bool(true, 145);
                encoder.write_bool(true, 156);
                encoder.write_bool(false, 163);
            }
            MbLumaMode::TmPred => {
                encoder.write_bool(true, 145);
                encoder.write_bool(true, 156);
                encoder.write_bool(true, 163);
                encoder.write_bool(false, 128);
            }
            MbLumaMode::BPred => {
                encoder.write_bool(true, 145);
                encoder.write_bool(true, 156);
                encoder.write_bool(true, 163);
                encoder.write_bool(true, 128);
            }
        }
    }

    /// Write UV prediction mode.
    fn write_uv_mode(&self, encoder: &mut BoolEncoder, mode: MbChromaMode) {
        match mode {
            MbChromaMode::DcPred => {
                encoder.write_bool(false, 142);
            }
            MbChromaMode::VPred => {
                encoder.write_bool(true, 142);
                encoder.write_bool(false, 114);
            }
            MbChromaMode::HPred => {
                encoder.write_bool(true, 142);
                encoder.write_bool(true, 114);
                encoder.write_bool(false, 183);
            }
            MbChromaMode::TmPred => {
                encoder.write_bool(true, 142);
                encoder.write_bool(true, 114);
                encoder.write_bool(true, 183);
            }
        }
    }

    /// Encode Y plane residuals.
    fn encode_y_residuals(
        &self,
        frame: &Vp8Frame,
        encoder: &mut BoolEncoder,
        mb_x: usize,
        mb_y: usize,
        y_mode: MbLumaMode,
    ) -> Result<()> {
        // Build prediction
        let ctx = PredictionContext::from_frame(frame, mb_x, mb_y);
        let pred_mode = match y_mode {
            MbLumaMode::DcPred => Intra16x16Mode::DcPred,
            MbLumaMode::VPred => Intra16x16Mode::VPred,
            MbLumaMode::HPred => Intra16x16Mode::HPred,
            MbLumaMode::TmPred => Intra16x16Mode::TmPred,
            MbLumaMode::BPred => Intra16x16Mode::DcPred, // Handled separately
        };

        let mut prediction = [0u8; 256];
        predict_16x16(pred_mode, &ctx, &mut prediction);

        // Compute and encode residuals for each 4x4 block
        for sub_y in 0..4 {
            for sub_x in 0..4 {
                let px = mb_x * 16 + sub_x * 4;
                let py = mb_y * 16 + sub_y * 4;

                // Get original and prediction blocks
                let mut orig = [0u8; 16];
                let mut pred = [0u8; 16];

                for y in 0..4 {
                    for x in 0..4 {
                        let fx = px + x;
                        let fy = py + y;
                        if fx < frame.width as usize && fy < frame.height as usize {
                            orig[y * 4 + x] = frame.get_y(fx, fy);
                        }
                        pred[y * 4 + x] = prediction[(sub_y * 4 + y) * 16 + sub_x * 4 + x];
                    }
                }

                // Compute residual
                let mut residual = [0i16; 16];
                sub_prediction(&orig, &pred, &mut residual, 4);

                // Transform
                let mut transformed = [0i16; 16];
                fdct4x4(&residual, &mut transformed);

                // Quantize
                let dc_quant = quant_tables::get_dc_quant(self.current_q);
                let ac_quant = quant_tables::get_ac_quant(self.current_q);

                let mut quantized = [0i16; 16];
                quantized[0] = (transformed[0] + dc_quant / 2) / dc_quant;
                for i in 1..16 {
                    quantized[i] = (transformed[i] + ac_quant / 2) / ac_quant;
                }

                // Encode coefficients
                self.encode_block_coeffs(encoder, &quantized)?;
            }
        }

        Ok(())
    }

    /// Encode UV plane residuals.
    fn encode_uv_residuals(
        &self,
        frame: &Vp8Frame,
        encoder: &mut BoolEncoder,
        mb_x: usize,
        mb_y: usize,
        uv_mode: MbChromaMode,
    ) -> Result<()> {
        let mode = match uv_mode {
            MbChromaMode::DcPred => IntraChromaMode::DcPred,
            MbChromaMode::VPred => IntraChromaMode::VPred,
            MbChromaMode::HPred => IntraChromaMode::HPred,
            MbChromaMode::TmPred => IntraChromaMode::TmPred,
        };

        // Encode U plane
        let u_ctx = self.build_chroma_context(frame, mb_x, mb_y, true);
        let mut u_pred = [0u8; 64];
        predict_8x8_chroma(mode, &u_ctx, &mut u_pred);

        for sub_y in 0..2 {
            for sub_x in 0..2 {
                let px = mb_x * 8 + sub_x * 4;
                let py = mb_y * 8 + sub_y * 4;

                let mut orig = [0u8; 16];
                let mut pred = [0u8; 16];

                for y in 0..4 {
                    for x in 0..4 {
                        let fx = px + x;
                        let fy = py + y;
                        let w = (frame.width as usize + 1) / 2;
                        let h = (frame.height as usize + 1) / 2;
                        if fx < w && fy < h {
                            orig[y * 4 + x] = frame.get_u(fx, fy);
                        }
                        pred[y * 4 + x] = u_pred[(sub_y * 4 + y) * 8 + sub_x * 4 + x];
                    }
                }

                let mut residual = [0i16; 16];
                sub_prediction(&orig, &pred, &mut residual, 4);

                let mut transformed = [0i16; 16];
                fdct4x4(&residual, &mut transformed);

                let dc_quant = quant_tables::get_dc_quant(self.current_q);
                let ac_quant = quant_tables::get_ac_quant(self.current_q);

                let mut quantized = [0i16; 16];
                quantized[0] = (transformed[0] + dc_quant / 2) / dc_quant;
                for i in 1..16 {
                    quantized[i] = (transformed[i] + ac_quant / 2) / ac_quant;
                }

                self.encode_block_coeffs(encoder, &quantized)?;
            }
        }

        // Encode V plane
        let v_ctx = self.build_chroma_context(frame, mb_x, mb_y, false);
        let mut v_pred = [0u8; 64];
        predict_8x8_chroma(mode, &v_ctx, &mut v_pred);

        for sub_y in 0..2 {
            for sub_x in 0..2 {
                let px = mb_x * 8 + sub_x * 4;
                let py = mb_y * 8 + sub_y * 4;

                let mut orig = [0u8; 16];
                let mut pred = [0u8; 16];

                for y in 0..4 {
                    for x in 0..4 {
                        let fx = px + x;
                        let fy = py + y;
                        let w = (frame.width as usize + 1) / 2;
                        let h = (frame.height as usize + 1) / 2;
                        if fx < w && fy < h {
                            orig[y * 4 + x] = frame.get_v(fx, fy);
                        }
                        pred[y * 4 + x] = v_pred[(sub_y * 4 + y) * 8 + sub_x * 4 + x];
                    }
                }

                let mut residual = [0i16; 16];
                sub_prediction(&orig, &pred, &mut residual, 4);

                let mut transformed = [0i16; 16];
                fdct4x4(&residual, &mut transformed);

                let dc_quant = quant_tables::get_dc_quant(self.current_q);
                let ac_quant = quant_tables::get_ac_quant(self.current_q);

                let mut quantized = [0i16; 16];
                quantized[0] = (transformed[0] + dc_quant / 2) / dc_quant;
                for i in 1..16 {
                    quantized[i] = (transformed[i] + ac_quant / 2) / ac_quant;
                }

                self.encode_block_coeffs(encoder, &quantized)?;
            }
        }

        Ok(())
    }

    /// Encode a block of coefficients.
    fn encode_block_coeffs(&self, encoder: &mut BoolEncoder, coeffs: &[i16; 16]) -> Result<()> {
        // Simplified coefficient encoding
        for i in 0..16 {
            let coeff = coeffs[i];
            if coeff == 0 {
                encoder.write_bool(true, 200); // EOB marker for simplified version
                break;
            }

            encoder.write_bool(false, 200);

            // Write magnitude
            let mag = coeff.unsigned_abs().min(15) as u32;
            encoder.write_bits(mag, 4);

            // Write sign
            encoder.write_bit(coeff < 0);
        }

        Ok(())
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

        if ctx.above_available && ctx.left_available {
            ctx.above_left = if is_u {
                frame.get_u(px - 1, py - 1)
            } else {
                frame.get_v(px - 1, py - 1)
            };
        }

        ctx
    }

    /// Update rate control.
    fn update_rate_control(&mut self) {
        // Simplified rate control - just use config quantizer
        self.current_q = self.config.quantizer;
    }

    /// Get encoder configuration.
    pub fn config(&self) -> &Vp8EncoderConfig {
        &self.config
    }

    /// Get frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Flush encoder and get any remaining packets.
    pub fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        // VP8 doesn't have B-frames, so no delayed frames
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config() {
        let config = Vp8EncoderConfig::new(320, 240)
            .bitrate(500)
            .quantizer(28)
            .speed(6);

        assert_eq!(config.width, 320);
        assert_eq!(config.height, 240);
        assert_eq!(config.bitrate, 500);
        assert_eq!(config.quantizer, 28);
        assert_eq!(config.speed, 6);
    }

    #[test]
    fn test_encoder_creation() {
        let config = Vp8EncoderConfig::new(320, 240);
        let encoder = Vp8Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_invalid_dimensions() {
        let config = Vp8EncoderConfig::new(0, 0);
        let encoder = Vp8Encoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_encode_frame() {
        let config = Vp8EncoderConfig::new(16, 16).quantizer(32);
        let mut encoder = Vp8Encoder::new(config).unwrap();

        let frame = Vp8Frame::new(16, 16, Vp8FrameType::KeyFrame);
        let result = encoder.encode(&frame);

        assert!(result.is_ok());
        let packet = result.unwrap();
        assert!(packet.is_keyframe);
        assert!(!packet.data.is_empty());
    }
}
