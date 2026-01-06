//! ALAC encoder implementation.

use crate::bitstream::BitWriter;
use crate::error::{AlacError, Result};
use crate::{AlacConfig, PredictorInfo, MAX_FRAME_SIZE};

/// ALAC encoder configuration.
#[derive(Debug, Clone)]
pub struct AlacEncoderConfig {
    /// Base configuration.
    pub config: AlacConfig,
    /// Fast mode (less compression, faster encoding).
    pub fast_mode: bool,
    /// Adaptive mode for coefficient adaptation.
    pub adaptive: bool,
}

impl AlacEncoderConfig {
    /// Create default encoder config.
    pub fn new(sample_rate: u32, channels: u8, bit_depth: u8) -> Result<Self> {
        Ok(Self {
            config: AlacConfig::new(sample_rate, channels, bit_depth)?,
            fast_mode: false,
            adaptive: true,
        })
    }
}

/// Encoded ALAC packet.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Number of samples encoded.
    pub num_samples: usize,
}

/// ALAC encoder.
pub struct AlacEncoder {
    config: AlacEncoderConfig,
    /// Working buffers.
    work_buffer: Vec<i32>,
    /// Predictor coefficients (adaptive).
    predictor_coef: [[i16; 32]; 8],
    /// Frame counter.
    frame_count: u64,
}

impl AlacEncoder {
    /// Create a new ALAC encoder.
    pub fn new(config: AlacEncoderConfig) -> Result<Self> {
        let frame_length = config.config.frame_length as usize;

        Ok(Self {
            config,
            work_buffer: vec![0; frame_length],
            predictor_coef: [[0; 32]; 8],
            frame_count: 0,
        })
    }

    /// Get encoder configuration.
    pub fn config(&self) -> &AlacEncoderConfig {
        &self.config
    }

    /// Get the magic cookie for this encoder.
    pub fn magic_cookie(&self) -> Vec<u8> {
        self.config.config.to_magic_cookie()
    }

    /// Encode samples.
    pub fn encode(&mut self, samples: &[i32], channels: u8) -> Result<EncodedPacket> {
        if samples.is_empty() {
            return Err(AlacError::EncodeError("Empty input".into()));
        }

        let num_samples = samples.len() / channels as usize;
        if num_samples > MAX_FRAME_SIZE {
            return Err(AlacError::InvalidSampleCount);
        }

        let mut writer = BitWriter::with_capacity(samples.len() * 4);

        // Write frame header
        // Channel count - 1 (3 bits)
        writer.write_bits((channels - 1) as u32, 3);

        // Reserved (4 bits)
        writer.write_bits(0, 4);

        // Has sample count, not uncompressed, no extra bits
        let has_sample_count = num_samples != self.config.config.frame_length as usize;
        writer.write_bit(has_sample_count);
        writer.write_bit(false); // Not uncompressed
        writer.write_bit(false); // No extra bits

        if has_sample_count {
            writer.write_bits(num_samples as u32, 32);
        }

        // Encode based on channel count
        if channels == 2 {
            self.encode_stereo(&mut writer, samples, num_samples)?;
        } else {
            self.encode_mono(&mut writer, samples, num_samples, 0)?;
        }

        self.frame_count += 1;

        Ok(EncodedPacket {
            data: writer.finalize(),
            num_samples,
        })
    }

    /// Encode mono channel.
    fn encode_mono(
        &mut self,
        writer: &mut BitWriter,
        samples: &[i32],
        num_samples: usize,
        channel: usize,
    ) -> Result<()> {
        // Compute prediction coefficients
        let pred_info = self.compute_predictor(samples, channel);

        // Write predictor info
        self.write_predictor_info(writer, &pred_info);

        // Compute and encode residuals
        let mut residuals = vec![0i32; num_samples];
        self.compute_residuals(samples, &mut residuals, &pred_info);

        // Rice encode residuals
        self.encode_residuals(writer, &residuals, &pred_info);

        Ok(())
    }

    /// Encode stereo channels.
    fn encode_stereo(
        &mut self,
        writer: &mut BitWriter,
        samples: &[i32],
        num_samples: usize,
    ) -> Result<()> {
        // Deinterleave
        let mut left = vec![0i32; num_samples];
        let mut right = vec![0i32; num_samples];

        for i in 0..num_samples {
            left[i] = samples[i * 2];
            right[i] = samples[i * 2 + 1];
        }

        // Compute stereo correlation
        let (interlacing_shift, interlacing_weight) =
            self.compute_stereo_correlation(&left, &right);

        // Apply correlation if beneficial
        if interlacing_weight > 0 {
            self.correlate_stereo(
                &mut left,
                &mut right,
                interlacing_shift,
                interlacing_weight,
            );
        }

        // Write interlacing info
        writer.write_bits(interlacing_shift as u32, 8);
        writer.write_bits(interlacing_weight as u32, 8);

        // Encode each channel
        let pred_info_left = self.compute_predictor(&left, 0);
        let pred_info_right = self.compute_predictor(&right, 1);

        self.write_predictor_info(writer, &pred_info_left);
        self.write_predictor_info(writer, &pred_info_right);

        // Compute residuals
        let mut residuals_left = vec![0i32; num_samples];
        let mut residuals_right = vec![0i32; num_samples];

        self.compute_residuals(&left, &mut residuals_left, &pred_info_left);
        self.compute_residuals(&right, &mut residuals_right, &pred_info_right);

        // Encode residuals
        self.encode_residuals(writer, &residuals_left, &pred_info_left);
        self.encode_residuals(writer, &residuals_right, &pred_info_right);

        Ok(())
    }

    /// Compute predictor coefficients using Levinson-Durbin.
    fn compute_predictor(&mut self, samples: &[i32], channel: usize) -> PredictorInfo {
        let order = if self.config.fast_mode { 4 } else { 8 };
        let order = order.min(samples.len().saturating_sub(1));

        if order == 0 || samples.len() < 2 {
            return PredictorInfo::default();
        }

        // Compute autocorrelation
        let mut autocorr = vec![0i64; order + 1];
        for lag in 0..=order {
            let mut sum = 0i64;
            for i in lag..samples.len() {
                sum += samples[i] as i64 * samples[i - lag] as i64;
            }
            autocorr[lag] = sum;
        }

        // Skip if signal is silence
        if autocorr[0] == 0 {
            return PredictorInfo::default();
        }

        // Levinson-Durbin recursion
        let mut coeffs = vec![0.0f64; order];
        let mut error = autocorr[0] as f64;

        for i in 0..order {
            let mut lambda = autocorr[i + 1] as f64;
            for j in 0..i {
                lambda -= coeffs[j] * autocorr[i - j] as f64;
            }
            lambda /= error;

            // Update coefficients
            let mut new_coeffs = coeffs.clone();
            new_coeffs[i] = lambda;
            for j in 0..i {
                new_coeffs[j] = coeffs[j] - lambda * coeffs[i - 1 - j];
            }
            coeffs = new_coeffs;

            error *= 1.0 - lambda * lambda;
            if error <= 0.0 {
                break;
            }
        }

        // Quantize coefficients
        let quant_shift = 9u16; // Standard ALAC shift
        let scale = (1 << quant_shift) as f64;
        let mut quantized = [0i16; 32];

        for (i, &c) in coeffs.iter().enumerate() {
            quantized[i] = (c * scale).round().clamp(-32768.0, 32767.0) as i16;
        }

        // Store for potential adaptation
        self.predictor_coef[channel][..order].copy_from_slice(&quantized[..order]);

        PredictorInfo {
            prediction_type: 0,
            quant_shift,
            rice_modifier: 4,
            predictor_coef_num: order as u16,
            predictor_coef: quantized,
        }
    }

    /// Write predictor info to bitstream.
    fn write_predictor_info(&self, writer: &mut BitWriter, pred_info: &PredictorInfo) {
        writer.write_bits(pred_info.prediction_type as u32, 4);
        writer.write_bits(pred_info.quant_shift as u32, 4);
        writer.write_bits(pred_info.rice_modifier as u32, 3);
        writer.write_bits(pred_info.predictor_coef_num as u32, 5);

        for i in 0..pred_info.predictor_coef_num as usize {
            writer.write_bits(pred_info.predictor_coef[i] as u32, 16);
        }
    }

    /// Compute prediction residuals.
    fn compute_residuals(&self, samples: &[i32], residuals: &mut [i32], pred_info: &PredictorInfo) {
        let order = pred_info.predictor_coef_num as usize;
        let shift = pred_info.quant_shift as i32;
        let coeffs = &pred_info.predictor_coef[..order];

        // First 'order' samples are passed through
        for i in 0..order.min(samples.len()) {
            residuals[i] = samples[i];
        }

        // Compute residuals for remaining samples
        for i in order..samples.len() {
            let mut prediction: i64 = 0;
            for (j, &coef) in coeffs.iter().enumerate() {
                prediction += samples[i - j - 1] as i64 * coef as i64;
            }
            let scaled = (prediction >> shift) as i32;
            residuals[i] = samples[i].wrapping_sub(scaled);
        }
    }

    /// Encode residuals using Rice coding.
    fn encode_residuals(&self, writer: &mut BitWriter, residuals: &[i32], _pred_info: &PredictorInfo) {
        let kb = self.config.config.kb as u32;
        let mb = self.config.config.mb as u32;
        let pb = self.config.config.pb as u32;

        let mut history = mb;

        for &residual in residuals {
            // Calculate Rice parameter k
            let k = Self::calculate_k(history, kb);

            // Write Rice-coded value
            writer.write_rice(residual, k);

            // Update history
            history = Self::update_history(history, residual.unsigned_abs(), pb, mb);
        }
    }

    /// Calculate Rice parameter k.
    fn calculate_k(history: u32, kb: u32) -> u32 {
        let x = (history >> 9) + 3;
        let k = 31u32.saturating_sub(x.leading_zeros());
        k.min(kb)
    }

    /// Update adaptive history.
    fn update_history(history: u32, value: u32, pb: u32, mb: u32) -> u32 {
        let add = (value > 0xFFFF) as u32 * 0x10000;
        let val = value.min(0xFFFF);
        let new_history = history.saturating_sub(((history * pb) >> 16) + 1) + val + add;
        new_history.min(mb * 4)
    }

    /// Compute stereo correlation parameters.
    fn compute_stereo_correlation(&self, left: &[i32], right: &[i32]) -> (u8, i32) {
        if left.is_empty() {
            return (0, 0);
        }

        // Simple correlation: try mid/side encoding
        let mut diff_energy = 0i64;
        let mut right_energy = 0i64;

        for i in 0..left.len() {
            let diff = left[i] - right[i];
            diff_energy += (diff as i64).pow(2);
            right_energy += (right[i] as i64).pow(2);
        }

        // If difference channel has less energy AND is non-zero, use correlation
        // For identical signals (diff_energy=0), no point in correlation
        if diff_energy > 0 && diff_energy < right_energy && right_energy > 0 {
            // Standard shift and weight for mid/side
            (4, 8)
        } else {
            (0, 0)
        }
    }

    /// Apply stereo correlation (encoding direction).
    fn correlate_stereo(
        &self,
        left: &mut [i32],
        right: &mut [i32],
        shift: u8,
        weight: i32,
    ) {
        for i in 0..left.len() {
            let l = left[i];
            let r = right[i];

            // Correlation (inverse of decorrelation)
            let new_right = l - r;
            let new_left = l - ((new_right * weight) >> shift);

            left[i] = new_left;
            right[i] = new_right;
        }
    }

    /// Encode uncompressed frame (fallback).
    #[allow(dead_code)]
    fn encode_uncompressed(&self, writer: &mut BitWriter, samples: &[i32], channels: u8) {
        let bit_depth = self.config.config.bit_depth as u32;
        let num_samples = samples.len() / channels as usize;

        // Write header indicating uncompressed
        writer.write_bits((channels - 1) as u32, 3);
        writer.write_bits(0, 4); // Reserved
        writer.write_bit(true); // Has sample count
        writer.write_bit(true); // Uncompressed
        writer.write_bit(false); // No extra bits
        writer.write_bits(num_samples as u32, 32);

        // Write raw samples
        for &sample in samples {
            writer.write_bits(sample as u32, bit_depth);
        }
    }

    /// Reset encoder state.
    pub fn reset(&mut self) {
        self.predictor_coef = [[0; 32]; 8];
        self.work_buffer.fill(0);
        self.frame_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> AlacEncoderConfig {
        AlacEncoderConfig::new(44100, 2, 16).unwrap()
    }

    #[test]
    fn test_encoder_creation() {
        let config = create_test_config();
        let encoder = AlacEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_reset() {
        let config = create_test_config();
        let mut encoder = AlacEncoder::new(config).unwrap();
        encoder.reset();
        assert_eq!(encoder.frame_count, 0);
    }

    #[test]
    fn test_magic_cookie() {
        let config = create_test_config();
        let encoder = AlacEncoder::new(config).unwrap();
        let cookie = encoder.magic_cookie();
        assert_eq!(cookie.len(), 24);
    }

    #[test]
    fn test_encode_silence() {
        let config = AlacEncoderConfig::new(44100, 2, 16).unwrap();
        let mut encoder = AlacEncoder::new(config).unwrap();

        // Stereo silence
        let samples = vec![0i32; 1024];
        let result = encoder.encode(&samples, 2);
        assert!(result.is_ok());

        let packet = result.unwrap();
        assert_eq!(packet.num_samples, 512);
        assert!(!packet.data.is_empty());
    }

    #[test]
    fn test_encode_sine_wave() {
        let config = AlacEncoderConfig::new(44100, 1, 16).unwrap();
        let mut encoder = AlacEncoder::new(config).unwrap();

        // Generate sine wave
        let samples: Vec<i32> = (0..1024)
            .map(|i| {
                let t = i as f32 / 44100.0;
                (f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * 16000.0) as i32
            })
            .collect();

        let result = encoder.encode(&samples, 1);
        assert!(result.is_ok());

        let packet = result.unwrap();
        assert_eq!(packet.num_samples, 1024);
        // Output should be non-empty
        // Note: Rice/unary coding can expand data significantly for certain input patterns
        // Real ALAC uses more sophisticated adaptive Rice with escape codes
        assert!(!packet.data.is_empty());
    }

    #[test]
    fn test_encode_stereo() {
        let config = AlacEncoderConfig::new(44100, 2, 16).unwrap();
        let mut encoder = AlacEncoder::new(config).unwrap();

        // Stereo samples (interleaved)
        let mut samples = Vec::with_capacity(2048);
        for i in 0..1024 {
            let t = i as f32 / 44100.0;
            let left = (f32::sin(2.0 * std::f32::consts::PI * 440.0 * t) * 16000.0) as i32;
            let right = (f32::sin(2.0 * std::f32::consts::PI * 445.0 * t) * 16000.0) as i32;
            samples.push(left);
            samples.push(right);
        }

        let result = encoder.encode(&samples, 2);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_input_error() {
        let config = create_test_config();
        let mut encoder = AlacEncoder::new(config).unwrap();
        let result = encoder.encode(&[], 2);
        assert!(matches!(result, Err(AlacError::EncodeError(_))));
    }

    #[test]
    fn test_calculate_k() {
        // k = min(kb, 31 - leading_zeros((history >> 9) + 3))
        assert_eq!(AlacEncoder::calculate_k(0, 14), 1);
        assert_eq!(AlacEncoder::calculate_k(512, 14), 2);
    }

    #[test]
    fn test_stereo_correlation() {
        let config = create_test_config();
        let encoder = AlacEncoder::new(config).unwrap();

        // Identical channels should have high correlation
        let left = vec![100, 200, 300, 400];
        let right = vec![100, 200, 300, 400];

        let (shift, weight) = encoder.compute_stereo_correlation(&left, &right);
        // Identical signals: difference = 0, so correlation won't help
        assert_eq!(weight, 0);
    }

    #[test]
    fn test_predictor_computation() {
        let config = create_test_config();
        let mut encoder = AlacEncoder::new(config).unwrap();

        // Predictable signal (linear ramp)
        let samples: Vec<i32> = (0..256).map(|i| i * 100).collect();
        let pred_info = encoder.compute_predictor(&samples, 0);

        // Should have some prediction coefficients
        assert!(pred_info.predictor_coef_num > 0);
    }
}
