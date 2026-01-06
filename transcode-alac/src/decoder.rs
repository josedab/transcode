//! ALAC decoder implementation.

use crate::bitstream::BitReader;
use crate::error::{AlacError, Result};
use crate::{AlacConfig, PredictorInfo, MAX_FRAME_SIZE};

/// Decoded ALAC packet.
#[derive(Debug, Clone)]
pub struct DecodedPacket {
    /// Sample data (interleaved).
    pub samples: Vec<i32>,
    /// Number of samples per channel.
    pub num_samples: usize,
    /// Number of channels.
    pub channels: u8,
    /// Bit depth.
    pub bit_depth: u8,
}

impl DecodedPacket {
    /// Convert samples to i16 format.
    pub fn to_i16(&self) -> Vec<i16> {
        let shift = self.bit_depth.saturating_sub(16) as i32;
        self.samples
            .iter()
            .map(|&s| {
                let shifted = if shift > 0 { s >> shift } else { s << (-shift) };
                shifted.clamp(i16::MIN as i32, i16::MAX as i32) as i16
            })
            .collect()
    }

    /// Convert samples to f32 format (normalized -1.0 to 1.0).
    pub fn to_f32(&self) -> Vec<f32> {
        let scale = 1.0 / (1i64 << (self.bit_depth - 1)) as f32;
        self.samples.iter().map(|&s| s as f32 * scale).collect()
    }
}

/// ALAC decoder.
pub struct AlacDecoder {
    config: AlacConfig,
    /// Mix buffer for stereo decorrelation.
    mix_buffer_u: Vec<i32>,
    mix_buffer_v: Vec<i32>,
    /// Predictor buffer.
    predictor_buffer: Vec<i32>,
    /// Previous samples for prediction.
    history: Vec<Vec<i32>>,
}

impl AlacDecoder {
    /// Create a new ALAC decoder.
    pub fn new(config: AlacConfig) -> Result<Self> {
        let frame_length = config.frame_length as usize;
        let channels = config.num_channels as usize;

        Ok(Self {
            config,
            mix_buffer_u: vec![0; frame_length],
            mix_buffer_v: vec![0; frame_length],
            predictor_buffer: vec![0; frame_length],
            history: vec![vec![0; 32]; channels],
        })
    }

    /// Get decoder configuration.
    pub fn config(&self) -> &AlacConfig {
        &self.config
    }

    /// Decode a packet.
    pub fn decode(&mut self, data: &[u8]) -> Result<DecodedPacket> {
        if data.is_empty() {
            return Err(AlacError::EndOfStream);
        }

        let mut reader = BitReader::new(data);

        // Read frame header
        let channels = reader.read_bits(3)? as u8 + 1;
        if channels != self.config.num_channels {
            // Could be end of stream indicator
            if channels == 8 {
                return Err(AlacError::EndOfStream);
            }
        }

        // Skip reserved bits
        let _reserved = reader.read_bits(4)?;

        // Sample count indicator
        let has_sample_count = reader.read_bit()?;
        let uncompressed = reader.read_bit()?;
        let has_extra_bits = reader.read_bit()?;

        // Read sample count
        let num_samples = if has_sample_count {
            reader.read_bits(32)? as usize
        } else {
            self.config.frame_length as usize
        };

        if num_samples > MAX_FRAME_SIZE {
            return Err(AlacError::InvalidSampleCount);
        }

        let samples = if uncompressed {
            self.decode_uncompressed(&mut reader, num_samples, channels)?
        } else {
            self.decode_compressed(&mut reader, num_samples, channels, has_extra_bits)?
        };

        Ok(DecodedPacket {
            samples,
            num_samples,
            channels,
            bit_depth: self.config.bit_depth,
        })
    }

    /// Decode uncompressed samples.
    fn decode_uncompressed(
        &mut self,
        reader: &mut BitReader,
        num_samples: usize,
        channels: u8,
    ) -> Result<Vec<i32>> {
        let bit_depth = self.config.bit_depth as u32;
        let total_samples = num_samples * channels as usize;
        let mut samples = Vec::with_capacity(total_samples);

        for _ in 0..total_samples {
            let sample = reader.read_signed(bit_depth)?;
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Decode compressed samples.
    fn decode_compressed(
        &mut self,
        reader: &mut BitReader,
        num_samples: usize,
        channels: u8,
        has_extra_bits: bool,
    ) -> Result<Vec<i32>> {
        let extra_bits = if has_extra_bits {
            reader.read_bits(2)? as u8 + 1
        } else {
            0
        };

        let bit_depth = self.config.bit_depth;
        let effective_bits = bit_depth - extra_bits;

        // For stereo, we may have interlacing
        if channels == 2 {
            self.decode_stereo(reader, num_samples, effective_bits, extra_bits)
        } else {
            self.decode_mono(reader, num_samples, effective_bits, extra_bits, 0)
                .map(|ch| ch)
        }
    }

    /// Decode mono channel.
    fn decode_mono(
        &mut self,
        reader: &mut BitReader,
        num_samples: usize,
        _bit_depth: u8,
        extra_bits: u8,
        channel: usize,
    ) -> Result<Vec<i32>> {
        // Read predictor info
        let pred_info = self.read_predictor_info(reader)?;

        // Read Rice-coded residuals
        let mut residuals = vec![0i32; num_samples];
        self.decode_residuals(reader, &mut residuals, &pred_info)?;

        // Apply prediction
        self.apply_prediction(&mut residuals, &pred_info, channel)?;

        // Apply extra bits shift if needed
        if extra_bits > 0 {
            let shift = extra_bits as i32;
            for sample in &mut residuals {
                // Read extra bits and combine
                let extra = reader.read_bits(extra_bits as u32)? as i32;
                *sample = (*sample << shift) | extra;
            }
        }

        Ok(residuals)
    }

    /// Decode stereo channels.
    fn decode_stereo(
        &mut self,
        reader: &mut BitReader,
        num_samples: usize,
        _bit_depth: u8,
        extra_bits: u8,
    ) -> Result<Vec<i32>> {
        // Read interlacing info
        let interlacing_shift = reader.read_bits(8)? as u8;
        let interlacing_leftweight = reader.read_bits(8)? as i32;

        // Read predictor info for both channels
        let pred_info_left = self.read_predictor_info(reader)?;
        let pred_info_right = self.read_predictor_info(reader)?;

        // Decode residuals for both channels
        let mut left = vec![0i32; num_samples];
        let mut right = vec![0i32; num_samples];

        self.decode_residuals(reader, &mut left, &pred_info_left)?;
        self.decode_residuals(reader, &mut right, &pred_info_right)?;

        // Apply prediction
        self.apply_prediction(&mut left, &pred_info_left, 0)?;
        self.apply_prediction(&mut right, &pred_info_right, 1)?;

        // Apply stereo decorrelation
        if interlacing_leftweight > 0 {
            self.decorrelate_stereo(
                &mut left,
                &mut right,
                interlacing_shift,
                interlacing_leftweight,
            );
        }

        // Apply extra bits shift if needed
        if extra_bits > 0 {
            let shift = extra_bits as i32;
            for i in 0..num_samples {
                let extra_l = reader.read_bits(extra_bits as u32)? as i32;
                let extra_r = reader.read_bits(extra_bits as u32)? as i32;
                left[i] = (left[i] << shift) | extra_l;
                right[i] = (right[i] << shift) | extra_r;
            }
        }

        // Interleave channels
        let mut samples = Vec::with_capacity(num_samples * 2);
        for i in 0..num_samples {
            samples.push(left[i]);
            samples.push(right[i]);
        }

        Ok(samples)
    }

    /// Read predictor information.
    fn read_predictor_info(&self, reader: &mut BitReader) -> Result<PredictorInfo> {
        let prediction_type = reader.read_bits(4)? as u16;
        let quant_shift = reader.read_bits(4)? as u16;
        let rice_modifier = reader.read_bits(3)? as u16;
        let predictor_coef_num = reader.read_bits(5)? as u16;

        let mut predictor_coef = [0i16; 32];
        for i in 0..predictor_coef_num as usize {
            predictor_coef[i] = reader.read_signed(16)? as i16;
        }

        Ok(PredictorInfo {
            prediction_type,
            quant_shift,
            rice_modifier,
            predictor_coef_num,
            predictor_coef,
        })
    }

    /// Decode Rice-coded residuals.
    fn decode_residuals(
        &self,
        reader: &mut BitReader,
        output: &mut [i32],
        _pred_info: &PredictorInfo,
    ) -> Result<()> {
        let kb = self.config.kb as u32;
        let mb = self.config.mb as u32;
        let pb = self.config.pb as u32;
        let output_len = output.len();

        let mut history = mb;
        let mut sign_modifier = 0i32;

        for sample in output.iter_mut() {
            // Calculate Rice parameter k
            let k = Self::calculate_k(history, kb);

            // Read Rice-coded value
            let mut value = reader.read_rice(k, kb, mb)?;

            // Apply sign modifier
            value += sign_modifier;
            sign_modifier = 0;

            *sample = value;

            // Update history
            history = Self::update_history(history, value.unsigned_abs(), pb, mb);

            // Check for block switch (run of zeros)
            if history < 128 && output_len > 1 {
                // Potential zero run
                let k_run = Self::calculate_k_run(history, kb);
                if k_run >= 0 {
                    // Skip handling zero runs for simplicity
                    // Full implementation would handle run-length encoding
                }
            }
        }

        Ok(())
    }

    /// Calculate Rice parameter k.
    fn calculate_k(history: u32, kb: u32) -> u32 {
        let x = (history >> 9) + 3;
        let k = 31u32.saturating_sub(x.leading_zeros());
        k.min(kb)
    }

    /// Calculate k for zero runs.
    fn calculate_k_run(history: u32, kb: u32) -> i32 {
        if history < 128 {
            let k = 7i32 - (history.leading_zeros() as i32);
            k.max(0).min(kb as i32)
        } else {
            -1
        }
    }

    /// Update adaptive history.
    fn update_history(history: u32, value: u32, pb: u32, mb: u32) -> u32 {
        let add = (value > 0xFFFF) as u32 * 0x10000;
        let val = value.min(0xFFFF);

        // History update formula
        let new_history = history.saturating_sub(((history * pb) >> 16) + 1) + val + add;
        new_history.min(mb * 4)
    }

    /// Apply linear prediction.
    fn apply_prediction(
        &mut self,
        samples: &mut [i32],
        pred_info: &PredictorInfo,
        _channel: usize,
    ) -> Result<()> {
        let order = pred_info.predictor_coef_num as usize;
        if order == 0 {
            return Ok(());
        }

        let shift = pred_info.quant_shift as i32;
        let coeffs = &pred_info.predictor_coef[..order];

        // First 'order' samples are unfiltered
        for i in order..samples.len() {
            let mut prediction: i64 = 0;

            // Apply FIR filter
            for (j, &coef) in coeffs.iter().enumerate() {
                prediction += samples[i - j - 1] as i64 * coef as i64;
            }

            // Scale and add to residual
            let scaled = (prediction >> shift) as i32;
            samples[i] = samples[i].wrapping_add(scaled);

            // Update coefficients (adaptive filter)
            let error = samples[i].signum();
            for j in 0..order {
                let sign = samples[i - j - 1].signum() * error;
                // Coefficient adaptation is typically small
                // Full implementation would update coefficients here
                let _ = sign; // Simplified - full ALAC updates coeffs adaptively
            }
        }

        Ok(())
    }

    /// Apply stereo decorrelation.
    fn decorrelate_stereo(
        &self,
        left: &mut [i32],
        right: &mut [i32],
        shift: u8,
        weight: i32,
    ) {
        for i in 0..left.len() {
            let l = left[i];
            let r = right[i];

            // Decorrelation formula
            let new_right = l - ((r * weight) >> shift);
            let new_left = new_right + r;

            left[i] = new_left;
            right[i] = new_right;
        }
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        for hist in &mut self.history {
            hist.fill(0);
        }
        self.mix_buffer_u.fill(0);
        self.mix_buffer_v.fill(0);
        self.predictor_buffer.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> AlacConfig {
        AlacConfig::new(44100, 2, 16).unwrap()
    }

    #[test]
    fn test_decoder_creation() {
        let config = create_test_config();
        let decoder = AlacDecoder::new(config);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_reset() {
        let config = create_test_config();
        let mut decoder = AlacDecoder::new(config).unwrap();
        decoder.reset();
        // Should not panic
    }

    #[test]
    fn test_calculate_k() {
        // k = min(kb, 31 - leading_zeros((history >> 9) + 3))
        // For history=0: x=3, k=31-30=1
        // For history=512: x=4, k=31-29=2
        // For history=1024: x=5, k=31-29=2
        assert_eq!(AlacDecoder::calculate_k(0, 14), 1);
        assert_eq!(AlacDecoder::calculate_k(512, 14), 2);
        assert_eq!(AlacDecoder::calculate_k(1024, 14), 2);
    }

    #[test]
    fn test_update_history() {
        let history = AlacDecoder::update_history(1000, 100, 40, 10);
        assert!(history > 0);
    }

    #[test]
    fn test_decoded_packet_to_i16() {
        let packet = DecodedPacket {
            samples: vec![0, 32767, -32768, 16384],
            num_samples: 2,
            channels: 2,
            bit_depth: 16,
        };

        let i16_samples = packet.to_i16();
        assert_eq!(i16_samples.len(), 4);
        assert_eq!(i16_samples[0], 0);
        assert_eq!(i16_samples[1], 32767);
        assert_eq!(i16_samples[2], -32768);
    }

    #[test]
    fn test_decoded_packet_to_f32() {
        let packet = DecodedPacket {
            samples: vec![0, 16384, -16384],
            num_samples: 3,
            channels: 1,
            bit_depth: 16,
        };

        let f32_samples = packet.to_f32();
        assert_eq!(f32_samples.len(), 3);
        assert!((f32_samples[0] - 0.0).abs() < 0.001);
        assert!((f32_samples[1] - 0.5).abs() < 0.001);
        assert!((f32_samples[2] + 0.5).abs() < 0.001);
    }

    #[test]
    fn test_empty_data_error() {
        let config = create_test_config();
        let mut decoder = AlacDecoder::new(config).unwrap();
        let result = decoder.decode(&[]);
        assert!(matches!(result, Err(AlacError::EndOfStream)));
    }

    #[test]
    fn test_decorrelate_stereo() {
        let config = create_test_config();
        let decoder = AlacDecoder::new(config).unwrap();

        let mut left = vec![100, 200, 300];
        let mut right = vec![10, 20, 30];

        decoder.decorrelate_stereo(&mut left, &mut right, 4, 8);

        // Values should be modified
        assert_ne!(left[0], 100);
        assert_ne!(right[0], 10);
    }
}
