//! Theora video decoder.

use crate::error::{Result, TheoraError};
use crate::{
    inverse_dct, CodingMode, FrameType, MotionVector, PixelFormat, TheoraConfig, BLOCK_SIZE,
    DC_SCALE, ZIGZAG,
};

/// Decoded Theora frame.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Y plane data.
    pub y_plane: Vec<u8>,
    /// U (Cb) plane data.
    pub u_plane: Vec<u8>,
    /// V (Cr) plane data.
    pub v_plane: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frame type.
    pub frame_type: FrameType,
    /// Pixel format.
    pub pixel_format: PixelFormat,
}

impl DecodedFrame {
    /// Create new decoded frame.
    pub fn new(config: &TheoraConfig) -> Self {
        let y_size = (config.frame_width * config.frame_height) as usize;
        let (h_sub, v_sub) = config.pixel_format.chroma_subsampling();
        let uv_size = ((config.frame_width / h_sub) * (config.frame_height / v_sub)) as usize;

        Self {
            y_plane: vec![0; y_size],
            u_plane: vec![0; uv_size],
            v_plane: vec![0; uv_size],
            width: config.frame_width,
            height: config.frame_height,
            frame_type: FrameType::Intra,
            pixel_format: config.pixel_format,
        }
    }

    /// Get Y plane.
    pub fn y(&self) -> &[u8] {
        &self.y_plane
    }

    /// Get U plane.
    pub fn u(&self) -> &[u8] {
        &self.u_plane
    }

    /// Get V plane.
    pub fn v(&self) -> &[u8] {
        &self.v_plane
    }

    /// Get Y plane stride.
    pub fn y_stride(&self) -> usize {
        self.width as usize
    }

    /// Get UV plane stride.
    pub fn uv_stride(&self) -> usize {
        let (h_sub, _) = self.pixel_format.chroma_subsampling();
        (self.width / h_sub) as usize
    }

    /// Check if frame is keyframe.
    pub fn is_keyframe(&self) -> bool {
        self.frame_type == FrameType::Intra
    }

    /// Get plane reference by index.
    ///
    /// Returns `Some` for valid plane indices (0=Y, 1=U, 2=V), `None` otherwise.
    pub fn plane(&self, index: usize) -> Option<&[u8]> {
        match index {
            0 => Some(&self.y_plane),
            1 => Some(&self.u_plane),
            2 => Some(&self.v_plane),
            _ => None,
        }
    }

    /// Get plane mutable reference by index.
    ///
    /// Returns `Some` for valid plane indices (0=Y, 1=U, 2=V), `None` otherwise.
    pub fn plane_mut(&mut self, index: usize) -> Option<&mut [u8]> {
        match index {
            0 => Some(&mut self.y_plane),
            1 => Some(&mut self.u_plane),
            2 => Some(&mut self.v_plane),
            _ => None,
        }
    }
}

/// Bitstream reader for Theora.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_pos: u8,
    bits_left: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            bit_pos: 0,
            bits_left: data.len() * 8,
        }
    }

    fn read_bit(&mut self) -> Result<bool> {
        if self.bits_left == 0 {
            return Err(TheoraError::EndOfStream);
        }

        let byte = self.data[self.pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1 != 0;

        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.pos += 1;
        }
        self.bits_left -= 1;

        Ok(bit)
    }

    fn read_bits(&mut self, n: u8) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(TheoraError::BitstreamError("Too many bits requested".into()));
        }

        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | (self.read_bit()? as u32);
        }
        Ok(value)
    }

    fn read_signed_bits(&mut self, n: u8) -> Result<i32> {
        let unsigned = self.read_bits(n)?;
        let sign_bit = 1u32 << (n - 1);
        if unsigned & sign_bit != 0 {
            Ok(unsigned as i32 - (1 << n))
        } else {
            Ok(unsigned as i32)
        }
    }

    fn remaining_bits(&self) -> usize {
        self.bits_left
    }
}

/// Decode DC coefficient from bitstream.
fn decode_dc_coeff(reader: &mut BitReader) -> Result<i32> {
    let sign = reader.read_bit()?;
    let magnitude = reader.read_bits(8)? as i32;
    Ok(if sign { -magnitude } else { magnitude })
}

/// Decode AC coefficient from bitstream.
fn decode_ac_coeff(reader: &mut BitReader) -> Result<(u8, i16)> {
    let run = reader.read_bits(4)? as u8;

    if run == 15 {
        return Ok((0, 0));
    }

    let sign = reader.read_bit()?;
    let magnitude = reader.read_bits(8)? as i16;
    let level = if sign { -magnitude } else { magnitude };

    Ok((run, level))
}

/// Decode block coefficients from bitstream.
fn decode_block_coeffs(
    reader: &mut BitReader,
    dc_pred: &mut [i32],
    block_idx: usize,
    quant: &[[i16; 64]; 64],
) -> Result<[i32; 64]> {
    let mut coeffs = [0i32; 64];

    // Decode DC coefficient
    let dc_diff = decode_dc_coeff(reader)?;
    let dc = dc_pred.get(block_idx.saturating_sub(1)).copied().unwrap_or(0) + dc_diff;
    dc_pred[block_idx] = dc;

    // Dequantize DC
    coeffs[0] = dc * quant[0][0] as i32;

    // Decode AC coefficients
    let mut i = 1;
    while i < 64 {
        if reader.remaining_bits() < 4 {
            break;
        }

        let (run, level) = decode_ac_coeff(reader)?;

        if run == 0 && level == 0 {
            break;
        }

        i += run as usize;
        if i >= 64 {
            break;
        }

        let zigzag_idx = ZIGZAG[i];
        coeffs[zigzag_idx] = level as i32 * quant[0][i] as i32;
        i += 1;
    }

    Ok(coeffs)
}

/// Theora decoder state.
pub struct TheoraDecoder {
    /// Configuration.
    config: TheoraConfig,
    /// Current frame.
    current_frame: Option<DecodedFrame>,
    /// Last reference frame.
    last_frame: Option<DecodedFrame>,
    /// Golden reference frame.
    golden_frame: Option<DecodedFrame>,
    /// DC predictor for Y.
    dc_pred_y: Vec<i32>,
    /// DC predictor for U.
    dc_pred_u: Vec<i32>,
    /// DC predictor for V.
    dc_pred_v: Vec<i32>,
    /// Block coding modes.
    block_modes: Vec<CodingMode>,
    /// Block motion vectors.
    block_mvs: Vec<MotionVector>,
    /// Quantization matrices.
    quant_matrices: Vec<[[i16; 64]; 64]>,
    /// Headers received.
    headers_complete: bool,
    /// Frame count.
    frame_count: u64,
}

impl TheoraDecoder {
    /// Create a new Theora decoder.
    pub fn new(config: TheoraConfig) -> Result<Self> {
        let num_blocks_y = config.num_blocks(0);
        let num_blocks_uv = config.num_blocks(1);

        // Initialize quantization matrices for each quality level
        let mut quant_matrices = Vec::with_capacity(64);
        for qi in 0..64 {
            let mut matrix = [[0i16; 64]; 64];
            let dc_scale = DC_SCALE[qi] as i32;

            for i in 0..64 {
                let scale = dc_scale;
                matrix[0][i] = (scale * 16 / 100).max(1) as i16;
            }

            quant_matrices.push(matrix);
        }

        Ok(Self {
            config,
            current_frame: None,
            last_frame: None,
            golden_frame: None,
            dc_pred_y: vec![0; num_blocks_y],
            dc_pred_u: vec![0; num_blocks_uv],
            dc_pred_v: vec![0; num_blocks_uv],
            block_modes: vec![CodingMode::Intra; num_blocks_y],
            block_mvs: vec![MotionVector::zero(); num_blocks_y],
            quant_matrices,
            headers_complete: false,
            frame_count: 0,
        })
    }

    /// Get configuration.
    pub fn config(&self) -> &TheoraConfig {
        &self.config
    }

    /// Set configuration.
    pub fn set_config(&mut self, config: TheoraConfig) {
        self.config = config;
    }

    /// Parse identification header.
    pub fn parse_id_header(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 42 {
            return Err(TheoraError::InvalidHeader("ID header too short".into()));
        }

        if data[0] != 0x80 {
            return Err(TheoraError::InvalidHeader("Invalid header type".into()));
        }

        if &data[1..7] != b"theora" {
            return Err(TheoraError::InvalidHeader("Invalid signature".into()));
        }

        let mut reader = BitReader::new(&data[7..]);

        let vmaj = reader.read_bits(8)? as u8;
        let vmin = reader.read_bits(8)? as u8;
        let vrev = reader.read_bits(8)? as u8;

        if vmaj != 3 || vmin > 2 {
            return Err(TheoraError::UnsupportedVersion {
                major: vmaj,
                minor: vmin,
            });
        }

        self.config.version_major = vmaj;
        self.config.version_minor = vmin;
        self.config.version_subminor = vrev;

        let fmbw = reader.read_bits(16)?;
        let fmbh = reader.read_bits(16)?;
        self.config.frame_width = fmbw * 16;
        self.config.frame_height = fmbh * 16;

        self.config.pic_width = reader.read_bits(24)?;
        self.config.pic_height = reader.read_bits(24)?;

        self.config.pic_x = reader.read_bits(8)?;
        self.config.pic_y = reader.read_bits(8)?;

        self.config.fps_num = reader.read_bits(32)?;
        self.config.fps_den = reader.read_bits(32)?;

        self.config.par_num = reader.read_bits(24)?;
        self.config.par_den = reader.read_bits(24)?;

        let cs = reader.read_bits(8)? as u8;
        self.config.color_space = cs.try_into()?;

        self.config.bitrate = reader.read_bits(24)?;

        self.config.quality = reader.read_bits(6)? as u8;

        self.config.keyframe_granule_shift = reader.read_bits(5)? as u8;

        let pf = reader.read_bits(2)? as u8;
        self.config.pixel_format = pf.try_into()?;

        Ok(())
    }

    /// Parse comment header.
    pub fn parse_comment_header(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 7 {
            return Err(TheoraError::InvalidComment("Comment header too short".into()));
        }

        if data[0] != 0x81 {
            return Err(TheoraError::InvalidComment("Invalid header type".into()));
        }

        if &data[1..7] != b"theora" {
            return Err(TheoraError::InvalidComment("Invalid signature".into()));
        }

        Ok(())
    }

    /// Parse setup header.
    pub fn parse_setup_header(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 7 {
            return Err(TheoraError::InvalidSetup("Setup header too short".into()));
        }

        if data[0] != 0x82 {
            return Err(TheoraError::InvalidSetup("Invalid header type".into()));
        }

        if &data[1..7] != b"theora" {
            return Err(TheoraError::InvalidSetup("Invalid signature".into()));
        }

        self.headers_complete = true;

        Ok(())
    }

    /// Decode a video frame.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedFrame> {
        if data.is_empty() {
            return Err(TheoraError::DecodeError("Empty frame data".into()));
        }

        if !self.headers_complete {
            if !data.is_empty() && (data[0] & 0x80) != 0 {
                match data[0] {
                    0x80 => {
                        self.parse_id_header(data)?;
                        return Err(TheoraError::NotInitialized);
                    }
                    0x81 => {
                        self.parse_comment_header(data)?;
                        return Err(TheoraError::NotInitialized);
                    }
                    0x82 => {
                        self.parse_setup_header(data)?;
                        return Err(TheoraError::NotInitialized);
                    }
                    _ => {}
                }
            }
            return Err(TheoraError::NotInitialized);
        }

        let mut reader = BitReader::new(data);

        let frame_type = if reader.read_bit()? {
            FrameType::Predicted
        } else {
            FrameType::Intra
        };

        let qi = reader.read_bits(6)? as usize;

        if frame_type == FrameType::Intra {
            let _ = reader.read_bits(3)?;
        }

        let mut frame = DecodedFrame::new(&self.config);
        frame.frame_type = frame_type;

        match frame_type {
            FrameType::Intra => self.decode_intra_frame(&mut reader, &mut frame, qi)?,
            FrameType::Predicted => self.decode_predicted_frame(&mut reader, &mut frame, qi)?,
        }

        if frame_type == FrameType::Intra {
            self.golden_frame = Some(frame.clone());
        }
        self.last_frame = Some(frame.clone());
        self.current_frame = Some(frame.clone());
        self.frame_count += 1;

        Ok(frame)
    }

    /// Decode intra frame.
    fn decode_intra_frame(
        &mut self,
        reader: &mut BitReader,
        frame: &mut DecodedFrame,
        qi: usize,
    ) -> Result<()> {
        self.dc_pred_y.fill(0);
        self.dc_pred_u.fill(0);
        self.dc_pred_v.fill(0);

        self.decode_plane(reader, frame, 0, qi)?;
        self.decode_plane(reader, frame, 1, qi)?;
        self.decode_plane(reader, frame, 2, qi)?;

        Ok(())
    }

    /// Decode predicted frame.
    fn decode_predicted_frame(
        &mut self,
        reader: &mut BitReader,
        frame: &mut DecodedFrame,
        qi: usize,
    ) -> Result<()> {
        if let Some(ref last) = self.last_frame {
            frame.y_plane.copy_from_slice(&last.y_plane);
            frame.u_plane.copy_from_slice(&last.u_plane);
            frame.v_plane.copy_from_slice(&last.v_plane);
        }

        self.decode_coding_modes(reader)?;
        self.decode_motion_vectors(reader)?;
        self.decode_predicted_blocks(reader, frame, qi)?;

        Ok(())
    }

    /// Decode plane.
    fn decode_plane(
        &mut self,
        reader: &mut BitReader,
        frame: &mut DecodedFrame,
        plane_idx: usize,
        qi: usize,
    ) -> Result<()> {
        let (width, height) = self.config.plane_dimensions(plane_idx);
        let blocks_x = (width / BLOCK_SIZE as u32) as usize;
        let blocks_y = (height / BLOCK_SIZE as u32) as usize;
        let stride = width as usize;
        let quant = self.quant_matrices[qi.min(63)];

        let dc_pred = match plane_idx {
            0 => &mut self.dc_pred_y,
            1 => &mut self.dc_pred_u,
            2 => &mut self.dc_pred_v,
            _ => return Err(TheoraError::DecodeError("Invalid plane".into())),
        };

        let plane = frame
            .plane_mut(plane_idx)
            .ok_or_else(|| TheoraError::DecodeError("Invalid plane index".into()))?;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;

                let coeffs = decode_block_coeffs(reader, dc_pred, block_idx, &quant)?;

                let mut pixels = [0i16; 64];
                inverse_dct(&coeffs, &mut pixels);

                for y in 0..BLOCK_SIZE {
                    for x in 0..BLOCK_SIZE {
                        let px = bx * BLOCK_SIZE + x;
                        let py = by * BLOCK_SIZE + y;
                        let val = (pixels[y * BLOCK_SIZE + x] + 128).clamp(0, 255) as u8;
                        plane[py * stride + px] = val;
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode coding modes.
    fn decode_coding_modes(&mut self, reader: &mut BitReader) -> Result<()> {
        for mode in &mut self.block_modes {
            let mode_idx = reader.read_bits(3)? as u8;
            *mode = mode_idx.try_into()?;
        }
        Ok(())
    }

    /// Decode motion vectors.
    fn decode_motion_vectors(&mut self, reader: &mut BitReader) -> Result<()> {
        let modes_copy: Vec<CodingMode> = self.block_modes.clone();
        for (i, mode) in modes_copy.iter().enumerate() {
            match mode {
                CodingMode::InterMv | CodingMode::InterGoldenMv | CodingMode::InterFourMv => {
                    let x = reader.read_signed_bits(8)? as i16;
                    let y = reader.read_signed_bits(8)? as i16;
                    self.block_mvs[i] = MotionVector::new(x, y);
                }
                CodingMode::InterMvLast => {
                    if i > 0 {
                        self.block_mvs[i] = self.block_mvs[i - 1];
                    }
                }
                _ => {
                    self.block_mvs[i] = MotionVector::zero();
                }
            }
        }
        Ok(())
    }

    /// Decode predicted blocks.
    fn decode_predicted_blocks(
        &mut self,
        reader: &mut BitReader,
        frame: &mut DecodedFrame,
        qi: usize,
    ) -> Result<()> {
        let blocks_x = (self.config.frame_width / BLOCK_SIZE as u32) as usize;
        let blocks_y = (self.config.frame_height / BLOCK_SIZE as u32) as usize;
        let stride = self.config.frame_width as usize;
        let frame_height = self.config.frame_height;
        let quant = self.quant_matrices[qi.min(63)];

        // Clone data to avoid borrow issues
        let modes_copy = self.block_modes.clone();
        let mvs_copy = self.block_mvs.clone();
        let last_frame_data = self.last_frame.as_ref().map(|f| f.y_plane.clone());

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_idx = by * blocks_x + bx;
                let mode = modes_copy[block_idx];

                if mode == CodingMode::Intra {
                    let coeffs =
                        decode_block_coeffs(reader, &mut self.dc_pred_y, block_idx, &quant)?;
                    let mut pixels = [0i16; 64];
                    inverse_dct(&coeffs, &mut pixels);

                    for y in 0..BLOCK_SIZE {
                        for x in 0..BLOCK_SIZE {
                            let px = bx * BLOCK_SIZE + x;
                            let py = by * BLOCK_SIZE + y;
                            let val = (pixels[y * BLOCK_SIZE + x] + 128).clamp(0, 255) as u8;
                            frame.y_plane[py * stride + px] = val;
                        }
                    }
                } else if reader.remaining_bits() >= 12 {
                    let coeffs =
                        decode_block_coeffs(reader, &mut self.dc_pred_y, block_idx, &quant)?;
                    let mut residual = [0i16; 64];
                    inverse_dct(&coeffs, &mut residual);

                    let mv = mvs_copy[block_idx];

                    // Apply motion compensation
                    for y in 0..BLOCK_SIZE {
                        for x in 0..BLOCK_SIZE {
                            let px = bx * BLOCK_SIZE + x;
                            let py = by * BLOCK_SIZE + y;

                            let ref_x = (px as i32 + mv.x as i32).clamp(0, stride as i32 - 1) as usize;
                            let ref_y =
                                (py as i32 + mv.y as i32).clamp(0, frame_height as i32 - 1) as usize;

                            let ref_val = last_frame_data
                                .as_ref()
                                .map(|f| f[ref_y * stride + ref_x] as i16)
                                .unwrap_or(128);

                            let val = (ref_val + residual[y * BLOCK_SIZE + x]).clamp(0, 255) as u8;
                            frame.y_plane[py * stride + px] = val;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        self.current_frame = None;
        self.last_frame = None;
        self.golden_frame = None;
        self.dc_pred_y.fill(0);
        self.dc_pred_u.fill(0);
        self.dc_pred_v.fill(0);
        self.block_modes.fill(CodingMode::Intra);
        self.block_mvs.fill(MotionVector::zero());
        self.frame_count = 0;
    }

    /// Get current frame.
    pub fn current_frame(&self) -> Option<&DecodedFrame> {
        self.current_frame.as_ref()
    }

    /// Get frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Check if headers are complete.
    pub fn headers_complete(&self) -> bool {
        self.headers_complete
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> TheoraConfig {
        TheoraConfig::new(320, 240).unwrap()
    }

    #[test]
    fn test_decoder_creation() {
        let config = create_test_config();
        let decoder = TheoraDecoder::new(config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_reset() {
        let config = create_test_config();
        let mut decoder = TheoraDecoder::new(config).unwrap();
        decoder.reset();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_decoded_frame() {
        let config = create_test_config();
        let frame = DecodedFrame::new(&config);

        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_eq!(frame.y().len(), 320 * 240);
        assert_eq!(frame.u().len(), 160 * 120);
        assert_eq!(frame.v().len(), 160 * 120);
    }

    #[test]
    fn test_decoded_frame_strides() {
        let config = create_test_config();
        let frame = DecodedFrame::new(&config);

        assert_eq!(frame.y_stride(), 320);
        assert_eq!(frame.uv_stride(), 160);
    }

    #[test]
    fn test_parse_id_header() {
        let config = create_test_config();
        let mut decoder = TheoraDecoder::new(config).unwrap();

        let mut header = vec![0x80];
        header.extend_from_slice(b"theora");
        header.extend_from_slice(&[3, 2, 1]);
        header.extend_from_slice(&[0, 20, 0, 15]);
        header.extend_from_slice(&[0, 0, 140, 1, 0, 0xF0]);
        header.extend_from_slice(&[0, 0]);
        header.extend_from_slice(&[0, 0, 117, 48]);
        header.extend_from_slice(&[0, 0, 3, 233]);
        header.extend_from_slice(&[0, 0, 1, 0, 0, 1]);
        header.extend_from_slice(&[0]);
        header.extend_from_slice(&[0, 0, 0]);
        header.push(0b11_0000_00);
        header.push(0b00_00_0000);

        let _ = decoder.parse_id_header(&header);
    }

    #[test]
    fn test_parse_comment_header() {
        let config = create_test_config();
        let mut decoder = TheoraDecoder::new(config).unwrap();

        let mut header = vec![0x81];
        header.extend_from_slice(b"theora");
        header.extend_from_slice(&[0, 0, 0, 0]);

        let result = decoder.parse_comment_header(&header);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_setup_header() {
        let config = create_test_config();
        let mut decoder = TheoraDecoder::new(config).unwrap();

        let mut header = vec![0x82];
        header.extend_from_slice(b"theora");
        header.extend_from_slice(&[0; 100]);

        let result = decoder.parse_setup_header(&header);
        assert!(result.is_ok());
        assert!(decoder.headers_complete());
    }

    #[test]
    fn test_empty_frame_error() {
        let config = create_test_config();
        let mut decoder = TheoraDecoder::new(config).unwrap();

        let result = decoder.decode(&[]);
        assert!(matches!(result, Err(TheoraError::DecodeError(_))));
    }

    #[test]
    fn test_not_initialized_error() {
        let config = create_test_config();
        let mut decoder = TheoraDecoder::new(config).unwrap();

        let result = decoder.decode(&[0x00, 0x30]);
        assert!(matches!(result, Err(TheoraError::NotInitialized)));
    }

    #[test]
    fn test_bitreader() {
        let data = [0b10110100, 0b11001010];
        let mut reader = BitReader::new(&data);

        assert!(reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());

        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
    }

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector::new(10, -5);
        assert_eq!(mv.x, 10);
        assert_eq!(mv.y, -5);

        let mv2 = MotionVector::zero();
        assert_eq!(mv2.x, 0);
        assert_eq!(mv2.y, 0);
    }
}
