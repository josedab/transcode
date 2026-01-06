//! FFV1 encoder implementation.

use crate::error::{Ffv1Error, Result};
use crate::range_coder::RangeEncoder;
use crate::{crc32, Ffv1Config, PredictionContext, FFV1_VERSION_3};

/// FFV1 encoder configuration.
#[derive(Debug, Clone)]
pub struct Ffv1EncoderConfig {
    /// Base configuration.
    pub config: Ffv1Config,
    /// Context model (0 = small, 1 = large).
    pub context_model: u8,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
}

impl Ffv1EncoderConfig {
    /// Create encoder config with defaults.
    pub fn new(width: u32, height: u32, bits: u8) -> Result<Self> {
        let mut config = Ffv1Config::new(width, height, bits)?;
        config.version = FFV1_VERSION_3;
        config.crc_protection = true;

        Ok(Self {
            config,
            context_model: 0,
            gop_size: 1, // All keyframes by default
        })
    }

    /// Set number of slices.
    pub fn set_slices(&mut self, h_slices: u32, v_slices: u32) {
        self.config.num_h_slices = h_slices.max(1);
        self.config.num_v_slices = v_slices.max(1);
    }
}

/// Encoded FFV1 packet.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Is keyframe.
    pub keyframe: bool,
}

/// FFV1 encoder.
pub struct Ffv1Encoder {
    config: Ffv1EncoderConfig,
    /// Context states.
    context_states: Vec<Vec<[u8; 32]>>,
    /// Frame counter.
    frame_count: u64,
}

impl Ffv1Encoder {
    /// Create a new FFV1 encoder.
    pub fn new(config: Ffv1EncoderConfig) -> Result<Self> {
        let context_states = config
            .config
            .context_counts
            .iter()
            .map(|&count| vec![[128u8; 32]; count])
            .collect();

        Ok(Self {
            config,
            context_states,
            frame_count: 0,
        })
    }

    /// Get encoder configuration.
    pub fn config(&self) -> &Ffv1EncoderConfig {
        &self.config
    }

    /// Generate configuration record (for container).
    pub fn config_record(&self) -> Vec<u8> {
        let mut encoder = RangeEncoder::with_capacity(256);
        let mut state = [128u8; 128];

        // Write configuration parameters
        Self::put_symbol(&mut encoder, &mut state, self.config.config.version);
        if self.config.config.version >= 3 {
            Self::put_symbol(&mut encoder, &mut state, self.config.config.micro_version);
        }

        Self::put_symbol(&mut encoder, &mut state, self.config.config.colorspace as u8);
        if self.config.config.version >= 1 {
            Self::put_symbol(
                &mut encoder,
                &mut state,
                self.config.config.bits_per_raw_sample,
            );
        }

        // Chroma subsampling
        let (h_sub, v_sub) = match self.config.config.chroma_subsampling {
            crate::ChromaSubsampling::Yuv444 => (0, 0),
            crate::ChromaSubsampling::Yuv422 => (1, 0),
            crate::ChromaSubsampling::Yuv420 => (1, 1),
            crate::ChromaSubsampling::Yuv411 => (2, 0),
            crate::ChromaSubsampling::Yuv410 => (2, 2),
        };
        Self::put_symbol(&mut encoder, &mut state, h_sub);
        Self::put_symbol(&mut encoder, &mut state, v_sub);

        Self::put_symbol(
            &mut encoder,
            &mut state,
            self.config.config.has_alpha as u8,
        );

        if self.config.config.version >= 3 {
            Self::put_symbol(
                &mut encoder,
                &mut state,
                (self.config.config.num_h_slices - 1) as u8,
            );
            Self::put_symbol(
                &mut encoder,
                &mut state,
                (self.config.config.num_v_slices - 1) as u8,
            );
            Self::put_symbol(
                &mut encoder,
                &mut state,
                (self.config.config.quant_table_count - 1) as u8,
            );
        }

        // Write quantization tables
        for table in &self.config.config.quant_tables {
            Self::write_quant_table(&mut encoder, &mut state, table);
        }

        encoder.finalize()
    }

    /// Put a symbol.
    fn put_symbol(encoder: &mut RangeEncoder, state: &mut [u8], value: u8) {
        for i in 0..8 {
            encoder.put_bit((value >> i) & 1 != 0, &mut state[i]);
        }
    }

    /// Write quantization table.
    fn write_quant_table(encoder: &mut RangeEncoder, state: &mut [u8], table: &[i8; 256]) {
        let mut i = 0usize;
        while i < 256 {
            // Count run of same values
            let val = table[i];
            let mut len = 1usize;
            while i + len < 256 && table[i + len] == val && len < 255 {
                len += 1;
            }

            Self::put_symbol(encoder, state, len as u8);
            i += len;
        }
        Self::put_symbol(encoder, state, 0); // End marker
    }

    /// Encode a frame.
    pub fn encode(&mut self, planes: &[&[u16]], width: u32, height: u32) -> Result<EncodedPacket> {
        if planes.is_empty() || planes[0].is_empty() {
            return Err(Ffv1Error::EncodeError("Empty input".into()));
        }

        let keyframe = self.frame_count % self.config.gop_size as u64 == 0;

        // Reset states on keyframe
        if keyframe {
            self.reset_states();
        }

        // Encode frame
        let data = if self.config.config.version >= 3 {
            self.encode_frame_v3(planes, width, height, keyframe)?
        } else {
            self.encode_frame_v0(planes, width, height, keyframe)?
        };

        self.frame_count += 1;

        Ok(EncodedPacket { data, keyframe })
    }

    /// Encode frame (version 0/1).
    fn encode_frame_v0(
        &mut self,
        planes: &[&[u16]],
        width: u32,
        height: u32,
        keyframe: bool,
    ) -> Result<Vec<u8>> {
        let mut encoder = RangeEncoder::with_capacity(width as usize * height as usize * 2);

        // Write keyframe flag
        let mut state = 128u8;
        encoder.put_bit(!keyframe, &mut state);

        // Encode each plane
        for (plane_idx, &plane) in planes.iter().enumerate() {
            let (w, h) = self.config.config.plane_dimensions(plane_idx);
            self.encode_plane(&mut encoder, plane, w, h, 0)?;
        }

        Ok(encoder.finalize())
    }

    /// Encode frame (version 3).
    fn encode_frame_v3(
        &mut self,
        planes: &[&[u16]],
        width: u32,
        height: u32,
        keyframe: bool,
    ) -> Result<Vec<u8>> {
        let mut encoder =
            RangeEncoder::with_capacity(width as usize * height as usize * planes.len());

        // Frame header
        let header_byte = if keyframe { 0x00 } else { 0x80 };
        encoder.put_raw_bits(header_byte, 8);

        // Encode each plane
        for (plane_idx, &plane) in planes.iter().enumerate() {
            let (w, h) = self.config.config.plane_dimensions(plane_idx);
            self.encode_plane(&mut encoder, plane, w, h, 0)?;
        }

        let mut output = encoder.finalize();

        // Add CRC if enabled
        if self.config.config.crc_protection {
            let crc = crc32(&output);
            output.extend_from_slice(&crc.to_le_bytes());
        }

        Ok(output)
    }

    /// Encode a plane.
    fn encode_plane(
        &mut self,
        encoder: &mut RangeEncoder,
        plane: &[u16],
        width: u32,
        height: u32,
        quant_idx: usize,
    ) -> Result<()> {
        let quant_table = &self.config.config.quant_tables[quant_idx];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let sample = plane[idx] as i32;

                // Get prediction context
                let ctx = self.get_prediction_context(plane, width, x, y);

                // Calculate prediction
                let pred = ctx.median_pred();

                // Calculate residual
                let residual = sample - pred;

                // Get context index
                let context_idx = self.get_context_index(quant_table, &ctx);

                // Encode residual
                encoder.put_symbol(residual, &mut self.context_states[quant_idx][context_idx]);
            }
        }

        Ok(())
    }

    /// Get prediction context.
    fn get_prediction_context(
        &self,
        plane: &[u16],
        stride: u32,
        x: u32,
        y: u32,
    ) -> PredictionContext {
        let idx = |x: u32, y: u32| (y * stride + x) as usize;

        let c = if x > 0 && y > 0 {
            plane[idx(x - 1, y - 1)] as i32
        } else {
            0
        };

        let b = if y > 0 {
            plane[idx(x, y - 1)] as i32
        } else {
            0
        };

        let a = if x > 0 {
            plane[idx(x - 1, y)] as i32
        } else if y > 0 {
            plane[idx(x, y - 1)] as i32
        } else {
            1 << (self.config.config.bits_per_raw_sample - 1)
        };

        let d = if x + 1 < stride && y > 0 {
            plane[idx(x + 1, y - 1)] as i32
        } else {
            0
        };

        PredictionContext { a, b, c, d }
    }

    /// Get context index from gradients.
    fn get_context_index(&self, quant_table: &[i8; 256], ctx: &PredictionContext) -> usize {
        let dh = ctx.a - ctx.c;
        let dv = ctx.b - ctx.c;

        let qh = quant_table[((dh + 128).clamp(0, 255)) as usize] as i32;
        let qv = quant_table[((dv + 128).clamp(0, 255)) as usize] as i32;

        let context = qh + qv * 16;
        (context.unsigned_abs() as usize) % self.config.config.context_counts[0]
    }

    /// Reset context states.
    fn reset_states(&mut self) {
        for states in &mut self.context_states {
            for state in states.iter_mut() {
                state.fill(128);
            }
        }
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.reset_states();
        self.frame_count = 0;
    }

    /// Get frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> Ffv1EncoderConfig {
        Ffv1EncoderConfig::new(64, 64, 8).unwrap()
    }

    #[test]
    fn test_encoder_creation() {
        let config = create_test_config();
        let encoder = Ffv1Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_reset() {
        let config = create_test_config();
        let mut encoder = Ffv1Encoder::new(config).unwrap();
        encoder.reset();
        assert_eq!(encoder.frame_count(), 0);
    }

    #[test]
    fn test_config_record() {
        let config = create_test_config();
        let encoder = Ffv1Encoder::new(config).unwrap();
        let record = encoder.config_record();
        assert!(!record.is_empty());
    }

    #[test]
    fn test_encode_gray_frame() {
        let config = Ffv1EncoderConfig::new(8, 8, 8).unwrap();
        let mut encoder = Ffv1Encoder::new(config).unwrap();

        // Create small gray frame (YCbCr with gray Y)
        let y_plane = vec![128u16; 8 * 8];
        let cb_plane = vec![128u16; 4 * 4];
        let cr_plane = vec![128u16; 4 * 4];

        let planes: Vec<&[u16]> = vec![&y_plane, &cb_plane, &cr_plane];
        let result = encoder.encode(&planes, 8, 8);
        assert!(result.is_ok());

        let packet = result.unwrap();
        assert!(packet.keyframe);
        assert!(!packet.data.is_empty());
    }

    #[test]
    fn test_encode_gradient_frame() {
        let config = Ffv1EncoderConfig::new(8, 8, 8).unwrap();
        let mut encoder = Ffv1Encoder::new(config).unwrap();

        // Create small gradient frame
        let mut y_plane = vec![0u16; 8 * 8];
        for y in 0..8 {
            for x in 0..8 {
                y_plane[y * 8 + x] = ((x * 32) as u16).min(255);
            }
        }
        let cb_plane = vec![128u16; 4 * 4];
        let cr_plane = vec![128u16; 4 * 4];

        let planes: Vec<&[u16]> = vec![&y_plane, &cb_plane, &cr_plane];
        let result = encoder.encode(&planes, 8, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_multiple_frames() {
        let config = Ffv1EncoderConfig::new(8, 8, 8).unwrap();
        let mut encoder = Ffv1Encoder::new(config).unwrap();

        let y_plane = vec![128u16; 8 * 8];
        let cb_plane = vec![128u16; 4 * 4];
        let cr_plane = vec![128u16; 4 * 4];
        let planes: Vec<&[u16]> = vec![&y_plane, &cb_plane, &cr_plane];

        // First frame should be keyframe
        let packet1 = encoder.encode(&planes, 8, 8).unwrap();
        assert!(packet1.keyframe);

        // Second frame should also be keyframe (gop_size = 1)
        let packet2 = encoder.encode(&planes, 8, 8).unwrap();
        assert!(packet2.keyframe);
    }

    #[test]
    fn test_empty_input_error() {
        let config = create_test_config();
        let mut encoder = Ffv1Encoder::new(config).unwrap();

        let planes: Vec<&[u16]> = vec![];
        let result = encoder.encode(&planes, 64, 64);
        assert!(matches!(result, Err(Ffv1Error::EncodeError(_))));
    }

    #[test]
    fn test_set_slices() {
        let mut config = create_test_config();
        config.set_slices(4, 4);
        assert_eq!(config.config.num_h_slices, 4);
        assert_eq!(config.config.num_v_slices, 4);
    }
}
