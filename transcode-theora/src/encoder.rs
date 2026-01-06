//! Theora video encoder.

use crate::error::{Result, TheoraError};
use crate::{
    forward_dct, CodingMode, FrameType, MotionVector, TheoraConfig, BLOCK_SIZE, DC_SCALE, ZIGZAG,
};

/// Encoded Theora packet.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Frame type.
    pub frame_type: FrameType,
    /// Granule position.
    pub granule_pos: i64,
}

/// Theora encoder configuration.
#[derive(Debug, Clone)]
pub struct TheoraEncoderConfig {
    /// Base configuration.
    pub config: TheoraConfig,
    /// Keyframe interval.
    pub keyframe_interval: u32,
    /// Rate control mode.
    pub rate_control: RateControl,
}

/// Rate control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControl {
    /// Constant quality.
    ConstantQuality(u8),
    /// Variable bitrate with target.
    Vbr { target_kbps: u32 },
    /// Constant bitrate.
    Cbr { bitrate_kbps: u32 },
}

impl Default for RateControl {
    fn default() -> Self {
        Self::ConstantQuality(48)
    }
}

impl TheoraEncoderConfig {
    /// Create new encoder config.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Ok(Self {
            config: TheoraConfig::new(width, height)?,
            keyframe_interval: 250,
            rate_control: RateControl::default(),
        })
    }

    /// Set quality (0-63).
    pub fn set_quality(&mut self, quality: u8) {
        self.config.set_quality(quality);
        self.rate_control = RateControl::ConstantQuality(quality.min(63));
    }

    /// Set target bitrate.
    pub fn set_bitrate(&mut self, kbps: u32) {
        self.rate_control = RateControl::Vbr { target_kbps: kbps };
    }

    /// Set keyframe interval.
    pub fn set_keyframe_interval(&mut self, interval: u32) {
        self.keyframe_interval = interval.max(1);
    }
}

/// Bitstream writer for Theora.
struct BitWriter {
    data: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self::with_capacity(4096)
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_pos);
        }
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.data.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    fn write_bits(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1 != 0);
        }
    }

    fn write_signed_bits(&mut self, value: i32, n: u8) {
        let unsigned = if value < 0 {
            (value + (1 << n)) as u32
        } else {
            value as u32
        };
        self.write_bits(unsigned, n);
    }

    fn finalize(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.data.push(self.current_byte);
        }
        self.data
    }

    fn bytes_written(&self) -> usize {
        self.data.len() + if self.bit_pos > 0 { 1 } else { 0 }
    }
}

/// Quantize DCT coefficients.
fn quantize_coeffs(coeffs: &mut [i32; 64], quant: &[i16; 64]) {
    for i in 0..64 {
        let q = quant[i] as i32;
        coeffs[i] = if q > 0 {
            (coeffs[i] + q / 2) / q
        } else {
            0
        };
    }
}

/// Encode block coefficients.
fn encode_block_coeffs(
    writer: &mut BitWriter,
    coeffs: &[i32; 64],
    dc_pred: &mut [i32],
    block_idx: usize,
) {
    // DC coefficient (differential)
    let dc = coeffs[0];
    let dc_diff = dc - dc_pred.get(block_idx.saturating_sub(1)).copied().unwrap_or(0);
    dc_pred[block_idx] = dc;

    // Encode DC
    let sign = dc_diff < 0;
    writer.write_bit(sign);
    writer.write_bits(dc_diff.unsigned_abs().min(255), 8);

    // Encode AC coefficients with run-length encoding
    let mut last_nonzero = 0;
    for i in 1..64 {
        if coeffs[ZIGZAG[i]] != 0 {
            last_nonzero = i;
        }
    }

    let mut i = 1;
    while i <= last_nonzero {
        let zigzag_idx = ZIGZAG[i];
        let coeff = coeffs[zigzag_idx];

        if coeff == 0 {
            // Count run of zeros
            let mut run = 0;
            while i + run <= last_nonzero && coeffs[ZIGZAG[i + run]] == 0 {
                run += 1;
            }
            run = run.min(14);

            writer.write_bits(run as u32, 4);
            i += run;
        } else {
            // Non-zero coefficient
            writer.write_bits(0, 4); // Run = 0
            let sign = coeff < 0;
            writer.write_bit(sign);
            writer.write_bits(coeff.unsigned_abs().min(255), 8);
            i += 1;
        }
    }

    // End of block
    writer.write_bits(15, 4);
}

/// Theora encoder.
pub struct TheoraEncoder {
    config: TheoraEncoderConfig,
    /// Last frame for inter prediction.
    last_frame: Option<Vec<u8>>,
    /// Golden reference frame.
    golden_frame: Option<Vec<u8>>,
    /// DC predictors.
    dc_pred_y: Vec<i32>,
    dc_pred_u: Vec<i32>,
    dc_pred_v: Vec<i32>,
    /// Block coding modes.
    block_modes: Vec<CodingMode>,
    /// Block motion vectors.
    block_mvs: Vec<MotionVector>,
    /// Quantization matrices.
    quant_matrices: Vec<[i16; 64]>,
    /// Frame counter.
    frame_count: u64,
    /// Keyframe counter.
    keyframe_count: u64,
    /// Current quality index.
    qi: usize,
}

impl TheoraEncoder {
    /// Create a new Theora encoder.
    pub fn new(config: TheoraEncoderConfig) -> Result<Self> {
        let num_blocks_y = config.config.num_blocks(0);
        let num_blocks_uv = config.config.num_blocks(1);

        // Initialize quantization matrices
        let mut quant_matrices = Vec::with_capacity(64);
        for qi in 0..64 {
            let mut matrix = [0i16; 64];
            let dc_scale = DC_SCALE[qi] as i32;

            for i in 0..64 {
                let scale = if i == 0 { dc_scale } else { dc_scale };
                matrix[i] = (scale * 16 / 100).max(1) as i16;
            }

            quant_matrices.push(matrix);
        }

        let qi = match config.rate_control {
            RateControl::ConstantQuality(q) => q as usize,
            _ => 32,
        };

        Ok(Self {
            config,
            last_frame: None,
            golden_frame: None,
            dc_pred_y: vec![0; num_blocks_y],
            dc_pred_u: vec![0; num_blocks_uv],
            dc_pred_v: vec![0; num_blocks_uv],
            block_modes: vec![CodingMode::Intra; num_blocks_y],
            block_mvs: vec![MotionVector::zero(); num_blocks_y],
            quant_matrices,
            frame_count: 0,
            keyframe_count: 0,
            qi: qi.min(63),
        })
    }

    /// Get encoder configuration.
    pub fn config(&self) -> &TheoraEncoderConfig {
        &self.config
    }

    /// Generate identification header.
    pub fn id_header(&self) -> Vec<u8> {
        let mut writer = BitWriter::with_capacity(42);

        // Header type
        writer.write_bits(0x80, 8);

        // Signature
        for &b in b"theora" {
            writer.write_bits(b as u32, 8);
        }

        // Version
        writer.write_bits(self.config.config.version_major as u32, 8);
        writer.write_bits(self.config.config.version_minor as u32, 8);
        writer.write_bits(self.config.config.version_subminor as u32, 8);

        // Frame dimensions in macroblocks
        let fmbw = self.config.config.frame_width / 16;
        let fmbh = self.config.config.frame_height / 16;
        writer.write_bits(fmbw as u32, 16);
        writer.write_bits(fmbh as u32, 16);

        // Picture dimensions
        writer.write_bits(self.config.config.pic_width, 24);
        writer.write_bits(self.config.config.pic_height, 24);

        // Picture offsets
        writer.write_bits(self.config.config.pic_x, 8);
        writer.write_bits(self.config.config.pic_y, 8);

        // Frame rate
        writer.write_bits(self.config.config.fps_num, 32);
        writer.write_bits(self.config.config.fps_den, 32);

        // Pixel aspect ratio
        writer.write_bits(self.config.config.par_num, 24);
        writer.write_bits(self.config.config.par_den, 24);

        // Color space
        writer.write_bits(self.config.config.color_space as u32, 8);

        // Bitrate
        writer.write_bits(self.config.config.bitrate, 24);

        // Quality + keyframe granule shift
        writer.write_bits(self.config.config.quality as u32, 6);
        writer.write_bits(self.config.config.keyframe_granule_shift as u32, 5);

        // Pixel format
        writer.write_bits(self.config.config.pixel_format as u32, 2);

        // Reserved bits
        writer.write_bits(0, 3);

        writer.finalize()
    }

    /// Generate comment header.
    pub fn comment_header(&self) -> Vec<u8> {
        let mut writer = BitWriter::with_capacity(64);

        // Header type
        writer.write_bits(0x81, 8);

        // Signature
        for &b in b"theora" {
            writer.write_bits(b as u32, 8);
        }

        // Vendor string length
        let vendor = b"transcode-theora";
        writer.write_bits(vendor.len() as u32, 32);

        // Vendor string
        for &b in vendor {
            writer.write_bits(b as u32, 8);
        }

        // Number of user comments
        writer.write_bits(0, 32);

        writer.finalize()
    }

    /// Generate setup header.
    pub fn setup_header(&self) -> Vec<u8> {
        let mut writer = BitWriter::with_capacity(1024);

        // Header type
        writer.write_bits(0x82, 8);

        // Signature
        for &b in b"theora" {
            writer.write_bits(b as u32, 8);
        }

        // Loop filter limits (64 entries, 6 bits each)
        for limit in crate::LOOP_FILTER_LIMITS.iter().take(64) {
            writer.write_bits(*limit as u32, 6);
        }

        // AC scale (64 entries, 10 bits each)
        for scale in crate::AC_SCALE.iter().take(64) {
            writer.write_bits(*scale as u32, 10);
        }

        // DC scale (64 entries, 10 bits each)
        for scale in crate::DC_SCALE.iter().take(64) {
            writer.write_bits(*scale as u32, 10);
        }

        // Number of base matrices
        writer.write_bits(3, 9);

        // Base matrices (Y intra, UV intra, inter)
        for &val in &crate::BASE_QUANT_INTRA_Y {
            writer.write_bits(val as u32, 8);
        }
        for &val in &crate::BASE_QUANT_INTRA_UV {
            writer.write_bits(val as u32, 8);
        }
        for &val in &crate::BASE_QUANT_INTER {
            writer.write_bits(val as u32, 8);
        }

        // Huffman table configuration (simplified)
        writer.write_bits(0, 3); // Use default tables

        writer.finalize()
    }

    /// Encode a video frame.
    pub fn encode(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> Result<EncodedPacket> {
        let expected_y = (self.config.config.frame_width * self.config.config.frame_height) as usize;
        let (h_sub, v_sub) = self.config.config.pixel_format.chroma_subsampling();
        let expected_uv = ((self.config.config.frame_width / h_sub)
            * (self.config.config.frame_height / v_sub)) as usize;

        if y.len() < expected_y {
            return Err(TheoraError::BufferTooSmall {
                required: expected_y,
                available: y.len(),
            });
        }

        if u.len() < expected_uv || v.len() < expected_uv {
            return Err(TheoraError::BufferTooSmall {
                required: expected_uv,
                available: u.len().min(v.len()),
            });
        }

        // Determine frame type
        let is_keyframe = self.frame_count == 0
            || self.frame_count % self.config.keyframe_interval as u64 == 0;

        let frame_type = if is_keyframe {
            FrameType::Intra
        } else {
            FrameType::Predicted
        };

        // Reset predictors on keyframe
        if is_keyframe {
            self.dc_pred_y.fill(0);
            self.dc_pred_u.fill(0);
            self.dc_pred_v.fill(0);
            self.keyframe_count += 1;
        }

        // Encode frame
        let data = match frame_type {
            FrameType::Intra => self.encode_intra_frame(y, u, v)?,
            FrameType::Predicted => self.encode_predicted_frame(y, u, v)?,
        };

        // Update reference frames
        if is_keyframe {
            self.golden_frame = Some(y.to_vec());
        }
        self.last_frame = Some(y.to_vec());

        // Calculate granule position
        let granule_pos = self.calculate_granule_pos(is_keyframe);

        self.frame_count += 1;

        Ok(EncodedPacket {
            data,
            frame_type,
            granule_pos,
        })
    }

    /// Encode intra frame.
    fn encode_intra_frame(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> Result<Vec<u8>> {
        let mut writer = BitWriter::with_capacity(
            (self.config.config.frame_width * self.config.config.frame_height) as usize,
        );

        // Frame type (0 = intra)
        writer.write_bit(false);

        // Quality index
        writer.write_bits(self.qi as u32, 6);

        // Keyframe type bits
        writer.write_bits(0, 3);

        // Encode Y plane
        self.encode_plane(&mut writer, y, 0)?;

        // Encode U plane
        self.encode_plane(&mut writer, u, 1)?;

        // Encode V plane
        self.encode_plane(&mut writer, v, 2)?;

        Ok(writer.finalize())
    }

    /// Encode predicted frame.
    fn encode_predicted_frame(&mut self, y: &[u8], u: &[u8], v: &[u8]) -> Result<Vec<u8>> {
        let mut writer = BitWriter::with_capacity(
            (self.config.config.frame_width * self.config.config.frame_height / 2) as usize,
        );

        // Frame type (1 = predicted)
        writer.write_bit(true);

        // Quality index
        writer.write_bits(self.qi as u32, 6);

        // Analyze blocks for motion estimation
        self.analyze_blocks(y)?;

        // Write coding modes
        for mode in &self.block_modes {
            writer.write_bits(*mode as u32, 3);
        }

        // Write motion vectors
        for (i, mode) in self.block_modes.iter().enumerate() {
            match mode {
                CodingMode::InterMv | CodingMode::InterGoldenMv | CodingMode::InterFourMv => {
                    writer.write_signed_bits(self.block_mvs[i].x as i32, 8);
                    writer.write_signed_bits(self.block_mvs[i].y as i32, 8);
                }
                _ => {}
            }
        }

        // Encode residuals
        self.encode_residual_blocks(&mut writer, y, u, v)?;

        Ok(writer.finalize())
    }

    /// Encode a plane.
    fn encode_plane(&mut self, writer: &mut BitWriter, plane: &[u8], plane_idx: usize) -> Result<()> {
        let (width, height) = self.config.config.plane_dimensions(plane_idx);
        let blocks_x = (width / BLOCK_SIZE as u32) as usize;
        let blocks_y = (height / BLOCK_SIZE as u32) as usize;
        let stride = width as usize;
        let quant = self.quant_matrices[self.qi];

        let dc_pred = match plane_idx {
            0 => &mut self.dc_pred_y,
            1 => &mut self.dc_pred_u,
            2 => &mut self.dc_pred_v,
            _ => return Err(TheoraError::EncodeError("Invalid plane".into())),
        };

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;

                // Extract block
                let mut block = [0i16; 64];
                for y in 0..BLOCK_SIZE {
                    for x in 0..BLOCK_SIZE {
                        let px = bx * BLOCK_SIZE + x;
                        let py = by * BLOCK_SIZE + y;
                        block[y * BLOCK_SIZE + x] = plane[py * stride + px] as i16 - 128;
                    }
                }

                // Forward DCT
                let mut coeffs = [0i32; 64];
                forward_dct(&block, &mut coeffs);

                // Quantize
                quantize_coeffs(&mut coeffs, &quant);

                // Encode coefficients
                encode_block_coeffs(writer, &coeffs, dc_pred, block_idx);
            }
        }

        Ok(())
    }


    /// Analyze blocks for mode decision and motion estimation.
    fn analyze_blocks(&mut self, y: &[u8]) -> Result<()> {
        let blocks_x = (self.config.config.frame_width / BLOCK_SIZE as u32) as usize;
        let blocks_y = (self.config.config.frame_height / BLOCK_SIZE as u32) as usize;
        let stride = self.config.config.frame_width as usize;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;

                // Simple mode decision: compare with last frame
                if let Some(ref last) = self.last_frame {
                    let mut intra_cost = 0i64;
                    let mut inter_cost = 0i64;

                    for dy in 0..BLOCK_SIZE {
                        for dx in 0..BLOCK_SIZE {
                            let px = bx * BLOCK_SIZE + dx;
                            let py = by * BLOCK_SIZE + dy;
                            let idx = py * stride + px;

                            let curr = y[idx] as i64;
                            let prev = last[idx] as i64;

                            intra_cost += (curr - 128).abs();
                            inter_cost += (curr - prev).abs();
                        }
                    }

                    if inter_cost < intra_cost * 8 / 10 {
                        self.block_modes[block_idx] = CodingMode::InterNoMv;
                        self.block_mvs[block_idx] = MotionVector::zero();
                    } else {
                        self.block_modes[block_idx] = CodingMode::Intra;
                        self.block_mvs[block_idx] = MotionVector::zero();
                    }
                } else {
                    self.block_modes[block_idx] = CodingMode::Intra;
                    self.block_mvs[block_idx] = MotionVector::zero();
                }
            }
        }

        Ok(())
    }

    /// Encode residual blocks for predicted frame.
    fn encode_residual_blocks(
        &mut self,
        writer: &mut BitWriter,
        y: &[u8],
        _u: &[u8],
        _v: &[u8],
    ) -> Result<()> {
        let blocks_x = (self.config.config.frame_width / BLOCK_SIZE as u32) as usize;
        let blocks_y = (self.config.config.frame_height / BLOCK_SIZE as u32) as usize;
        let stride = self.config.config.frame_width as usize;
        let quant = self.quant_matrices[self.qi];

        // Clone last frame reference data to avoid borrow issues
        let last_frame_data = self.last_frame.clone();

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;
                let mode = self.block_modes[block_idx];

                if mode == CodingMode::Intra {
                    // Encode as intra block
                    let mut block = [0i16; 64];
                    for dy in 0..BLOCK_SIZE {
                        for dx in 0..BLOCK_SIZE {
                            let px = bx * BLOCK_SIZE + dx;
                            let py = by * BLOCK_SIZE + dy;
                            block[dy * BLOCK_SIZE + dx] = y[py * stride + px] as i16 - 128;
                        }
                    }

                    let mut coeffs = [0i32; 64];
                    forward_dct(&block, &mut coeffs);
                    quantize_coeffs(&mut coeffs, &quant);
                    encode_block_coeffs(writer, &coeffs, &mut self.dc_pred_y, block_idx);
                } else if let Some(ref last) = last_frame_data {
                    // Encode residual
                    let mut block = [0i16; 64];
                    for dy in 0..BLOCK_SIZE {
                        for dx in 0..BLOCK_SIZE {
                            let px = bx * BLOCK_SIZE + dx;
                            let py = by * BLOCK_SIZE + dy;
                            let idx = py * stride + px;
                            block[dy * BLOCK_SIZE + dx] = y[idx] as i16 - last[idx] as i16;
                        }
                    }

                    let mut coeffs = [0i32; 64];
                    forward_dct(&block, &mut coeffs);
                    quantize_coeffs(&mut coeffs, &quant);
                    encode_block_coeffs(writer, &coeffs, &mut self.dc_pred_y, block_idx);
                }
            }
        }

        Ok(())
    }

    /// Calculate granule position.
    fn calculate_granule_pos(&self, is_keyframe: bool) -> i64 {
        let shift = self.config.config.keyframe_granule_shift as u32;
        let frames_since_keyframe = if is_keyframe {
            0
        } else {
            self.frame_count - (self.keyframe_count - 1) * self.config.keyframe_interval as u64
        };

        ((self.keyframe_count as i64) << shift) | (frames_since_keyframe as i64)
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.last_frame = None;
        self.golden_frame = None;
        self.dc_pred_y.fill(0);
        self.dc_pred_u.fill(0);
        self.dc_pred_v.fill(0);
        self.block_modes.fill(CodingMode::Intra);
        self.block_mvs.fill(MotionVector::zero());
        self.frame_count = 0;
        self.keyframe_count = 0;
    }

    /// Get frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get output size estimate.
    pub fn estimated_size(&self) -> usize {
        (self.config.config.frame_width * self.config.config.frame_height) as usize / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> TheoraEncoderConfig {
        TheoraEncoderConfig::new(64, 64).unwrap()
    }

    #[test]
    fn test_encoder_creation() {
        let config = create_test_config();
        let encoder = TheoraEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_reset() {
        let config = create_test_config();
        let mut encoder = TheoraEncoder::new(config).unwrap();
        encoder.reset();
        assert_eq!(encoder.frame_count(), 0);
    }

    #[test]
    fn test_id_header() {
        let config = create_test_config();
        let encoder = TheoraEncoder::new(config).unwrap();
        let header = encoder.id_header();

        assert!(!header.is_empty());
        assert_eq!(header[0], 0x80);
        assert_eq!(&header[1..7], b"theora");
    }

    #[test]
    fn test_comment_header() {
        let config = create_test_config();
        let encoder = TheoraEncoder::new(config).unwrap();
        let header = encoder.comment_header();

        assert!(!header.is_empty());
        assert_eq!(header[0], 0x81);
        assert_eq!(&header[1..7], b"theora");
    }

    #[test]
    fn test_setup_header() {
        let config = create_test_config();
        let encoder = TheoraEncoder::new(config).unwrap();
        let header = encoder.setup_header();

        assert!(!header.is_empty());
        assert_eq!(header[0], 0x82);
        assert_eq!(&header[1..7], b"theora");
    }

    #[test]
    fn test_encode_intra_frame() {
        let config = TheoraEncoderConfig::new(16, 16).unwrap();
        let mut encoder = TheoraEncoder::new(config).unwrap();

        let y = vec![128u8; 16 * 16];
        let u = vec![128u8; 8 * 8];
        let v = vec![128u8; 8 * 8];

        let result = encoder.encode(&y, &u, &v);
        assert!(result.is_ok());

        let packet = result.unwrap();
        assert_eq!(packet.frame_type, FrameType::Intra);
        assert!(!packet.data.is_empty());
    }

    #[test]
    fn test_encode_predicted_frame() {
        let config = TheoraEncoderConfig::new(16, 16).unwrap();
        let mut encoder = TheoraEncoder::new(config).unwrap();

        let y = vec![128u8; 16 * 16];
        let u = vec![128u8; 8 * 8];
        let v = vec![128u8; 8 * 8];

        // First frame is keyframe
        let _ = encoder.encode(&y, &u, &v).unwrap();

        // Second frame is predicted
        let result = encoder.encode(&y, &u, &v);
        assert!(result.is_ok());

        let packet = result.unwrap();
        assert_eq!(packet.frame_type, FrameType::Predicted);
    }

    #[test]
    fn test_encode_gradient_frame() {
        let config = TheoraEncoderConfig::new(16, 16).unwrap();
        let mut encoder = TheoraEncoder::new(config).unwrap();

        // Create gradient
        let mut y = vec![0u8; 16 * 16];
        for row in 0..16 {
            for col in 0..16 {
                y[row * 16 + col] = ((col * 16) as u8).min(255);
            }
        }
        let u = vec![128u8; 8 * 8];
        let v = vec![128u8; 8 * 8];

        let result = encoder.encode(&y, &u, &v);
        assert!(result.is_ok());
    }

    #[test]
    fn test_buffer_too_small_error() {
        let config = TheoraEncoderConfig::new(64, 64).unwrap();
        let mut encoder = TheoraEncoder::new(config).unwrap();

        let y = vec![128u8; 32]; // Too small
        let u = vec![128u8; 32 * 32];
        let v = vec![128u8; 32 * 32];

        let result = encoder.encode(&y, &u, &v);
        assert!(matches!(result, Err(TheoraError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_rate_control_modes() {
        let mut config = TheoraEncoderConfig::new(64, 64).unwrap();

        config.set_quality(32);
        assert!(matches!(
            config.rate_control,
            RateControl::ConstantQuality(32)
        ));

        config.set_bitrate(1000);
        assert!(matches!(
            config.rate_control,
            RateControl::Vbr { target_kbps: 1000 }
        ));
    }

    #[test]
    fn test_keyframe_interval() {
        let mut config = TheoraEncoderConfig::new(16, 16).unwrap();
        config.set_keyframe_interval(5);

        let mut encoder = TheoraEncoder::new(config).unwrap();

        let y = vec![128u8; 16 * 16];
        let u = vec![128u8; 8 * 8];
        let v = vec![128u8; 8 * 8];

        // Encode multiple frames
        for i in 0..10 {
            let packet = encoder.encode(&y, &u, &v).unwrap();
            if i % 5 == 0 {
                assert_eq!(packet.frame_type, FrameType::Intra);
            } else {
                assert_eq!(packet.frame_type, FrameType::Predicted);
            }
        }
    }

    #[test]
    fn test_bitwriter() {
        let mut writer = BitWriter::new();

        writer.write_bit(true);
        writer.write_bit(false);
        writer.write_bits(0b1010, 4);
        writer.write_bits(0xFF, 8);

        let data = writer.finalize();
        assert!(!data.is_empty());
    }

    #[test]
    fn test_granule_position() {
        let config = TheoraEncoderConfig::new(16, 16).unwrap();
        let mut encoder = TheoraEncoder::new(config).unwrap();

        let y = vec![128u8; 16 * 16];
        let u = vec![128u8; 8 * 8];
        let v = vec![128u8; 8 * 8];

        let packet = encoder.encode(&y, &u, &v).unwrap();
        assert!(packet.granule_pos >= 0);
    }
}
