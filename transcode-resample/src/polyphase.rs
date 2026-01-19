//! Polyphase filter bank resampler.
//!
//! Efficient implementation for sample rate conversion using polyphase decomposition.
//! Particularly efficient when the conversion ratio can be expressed as a ratio of small integers.

use crate::error::{ResampleError, Result};
use crate::simd;
use crate::{gcd, ResamplerImpl};
use std::f64::consts::PI;

/// Polyphase filter bank resampler.
///
/// Uses polyphase decomposition to efficiently implement sample rate conversion.
/// The filter bank is constructed by decomposing a prototype lowpass filter
/// into multiple sub-filters (phases), which reduces computational complexity.
///
/// # Algorithm
///
/// For a conversion ratio L/M (where L is interpolation and M is decimation):
/// 1. Interpolate by factor L (insert zeros)
/// 2. Apply lowpass filter
/// 3. Decimate by factor M
///
/// The polyphase implementation avoids computing samples that would be discarded.
///
/// # Quality Characteristics
/// - Excellent quality for rational rate conversions
/// - Very efficient for common conversions (e.g., 44100 <-> 48000)
/// - Lower complexity than pure sinc interpolation
#[derive(Debug)]
#[allow(dead_code)]
pub struct PolyphaseResampler {
    input_rate: u32,
    output_rate: u32,
    /// Interpolation factor
    interp_factor: u32,
    /// Decimation factor
    decim_factor: u32,
    /// Number of taps per phase
    taps_per_phase: usize,
    /// Polyphase filter bank [phase][tap]
    filter_bank: Vec<Vec<f32>>,
    /// Input buffer (ring buffer)
    input_buffer: Vec<f32>,
    /// Write position in input buffer
    buffer_write_pos: usize,
    /// Current output phase
    output_phase: u64,
    /// Number of channels
    channels: usize,
}

impl PolyphaseResampler {
    /// Create a new polyphase resampler.
    ///
    /// # Arguments
    /// * `input_rate` - Input sample rate in Hz
    /// * `output_rate` - Output sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `filter_length` - Total filter length (will be distributed across phases)
    ///
    /// # Errors
    /// Returns an error if parameters are invalid.
    pub fn new(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        filter_length: usize,
    ) -> Result<Self> {
        if input_rate == 0 {
            return Err(ResampleError::InvalidSampleRate { rate: input_rate });
        }
        if output_rate == 0 {
            return Err(ResampleError::InvalidSampleRate { rate: output_rate });
        }
        if channels == 0 {
            return Err(ResampleError::InvalidChannelCount { count: channels });
        }

        // Find the simplified ratio L/M
        let g = gcd(input_rate, output_rate);
        let interp_factor = output_rate / g; // L
        let decim_factor = input_rate / g;   // M

        // Check for extreme ratios
        let ratio = output_rate as f64 / input_rate as f64;
        if !(1.0 / 256.0..=256.0).contains(&ratio) {
            return Err(ResampleError::RatioTooExtreme { ratio });
        }

        // Calculate taps per phase
        let taps_per_phase = filter_length.div_ceil(interp_factor as usize);

        // Create prototype lowpass filter
        let prototype = Self::design_lowpass_filter(
            taps_per_phase * interp_factor as usize,
            interp_factor,
            decim_factor,
        );

        // Decompose into polyphase filter bank
        let filter_bank = Self::create_polyphase_bank(&prototype, interp_factor as usize, taps_per_phase);

        // Input buffer size (enough for filter history)
        let buffer_size = taps_per_phase;
        let input_buffer = vec![0.0; buffer_size];

        Ok(Self {
            input_rate,
            output_rate,
            interp_factor,
            decim_factor,
            taps_per_phase,
            filter_bank,
            input_buffer,
            buffer_write_pos: 0,
            output_phase: 0,
            channels,
        })
    }

    /// Create with default filter length.
    pub fn with_defaults(input_rate: u32, output_rate: u32, channels: usize) -> Result<Self> {
        // Default to 256 taps for good quality
        Self::new(input_rate, output_rate, channels, 256)
    }

    /// Design a lowpass filter using a windowed sinc.
    fn design_lowpass_filter(length: usize, interp: u32, decim: u32) -> Vec<f32> {
        // Cutoff should be the lower of the two rates to avoid aliasing
        let cutoff = if interp > decim {
            1.0 / interp as f64
        } else {
            1.0 / decim as f64
        };

        let half_len = (length - 1) as f64 / 2.0;

        (0..length)
            .map(|i| {
                let x = i as f64 - half_len;

                // Sinc function
                let sinc = if x.abs() < 1e-10 {
                    2.0 * cutoff
                } else {
                    (2.0 * PI * cutoff * x).sin() / (PI * x)
                };

                // Blackman window
                let window = 0.42
                    - 0.5 * (2.0 * PI * i as f64 / (length - 1) as f64).cos()
                    + 0.08 * (4.0 * PI * i as f64 / (length - 1) as f64).cos();

                // Scale by interpolation factor for proper gain
                (sinc * window * interp as f64) as f32
            })
            .collect()
    }

    /// Decompose prototype filter into polyphase filter bank.
    fn create_polyphase_bank(
        prototype: &[f32],
        num_phases: usize,
        taps_per_phase: usize,
    ) -> Vec<Vec<f32>> {
        (0..num_phases)
            .map(|phase| {
                (0..taps_per_phase)
                    .map(|tap| {
                        let idx = phase + tap * num_phases;
                        if idx < prototype.len() {
                            prototype[idx]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Apply a single phase filter.
    #[inline]
    #[allow(unreachable_code)]
    fn apply_phase_filter(&self, phase: usize) -> f32 {
        let coeffs = &self.filter_bank[phase];

        // Get samples from ring buffer in correct order
        let mut samples = Vec::with_capacity(self.taps_per_phase);
        for i in 0..self.taps_per_phase {
            let idx = (self.buffer_write_pos + self.taps_per_phase - 1 - i) % self.taps_per_phase;
            samples.push(self.input_buffer[idx]);
        }

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return simd::apply_filter_avx2(&samples, 0, coeffs, self.taps_per_phase);
            }
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            return simd::apply_filter_neon(&samples, 0, coeffs, self.taps_per_phase);
        }

        // Scalar fallback
        samples.iter().zip(coeffs.iter()).map(|(&s, &c)| s * c).sum()
    }
}

impl ResamplerImpl for PolyphaseResampler {
    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate expected output size
        let output_len = ((input.len() as u64 * self.interp_factor as u64)
            / self.decim_factor as u64) as usize + 1;
        let mut output = Vec::with_capacity(output_len);

        for &sample in input {
            // Add sample to circular buffer
            self.input_buffer[self.buffer_write_pos] = sample;
            self.buffer_write_pos = (self.buffer_write_pos + 1) % self.taps_per_phase;

            // Generate output samples for this input
            // For every input sample, we may produce 0, 1, or more outputs
            // depending on the ratio
            while self.output_phase < self.interp_factor as u64 {
                let phase = self.output_phase as usize;
                let out_sample = self.apply_phase_filter(phase);
                output.push(out_sample);

                self.output_phase += self.decim_factor as u64;
            }
            self.output_phase -= self.interp_factor as u64;
        }

        Ok(output)
    }

    fn process_interleaved(&mut self, input: &[f32], channels: usize) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        if input.len() % channels != 0 {
            return Err(ResampleError::BufferSizeMismatch {
                actual: input.len(),
                channels,
            });
        }

        // For simplicity, process each channel separately
        let _input_frames = input.len() / channels;
        let mut channel_outputs: Vec<Vec<f32>> = Vec::with_capacity(channels);

        for ch in 0..channels {
            // Extract channel
            let channel_input: Vec<f32> = input.iter().skip(ch).step_by(channels).copied().collect();

            // Reset state for each channel (except first)
            if ch > 0 {
                self.reset();
            }

            // Process channel
            let channel_output = self.process(&channel_input)?;
            channel_outputs.push(channel_output);
        }

        // Interleave outputs
        let output_frames = channel_outputs.first().map(|v| v.len()).unwrap_or(0);
        let mut output = Vec::with_capacity(output_frames * channels);

        for frame in 0..output_frames {
            for ch in 0..channels {
                output.push(channel_outputs[ch].get(frame).copied().unwrap_or(0.0));
            }
        }

        Ok(output)
    }

    fn input_rate(&self) -> u32 {
        self.input_rate
    }

    fn output_rate(&self) -> u32 {
        self.output_rate
    }

    fn reset(&mut self) {
        self.input_buffer.fill(0.0);
        self.buffer_write_pos = 0;
        self.output_phase = 0;
    }

    fn latency(&self) -> usize {
        self.taps_per_phase / 2
    }

    fn flush(&mut self) -> Result<Vec<f32>> {
        // Process with zero padding
        let padding = vec![0.0f32; self.taps_per_phase];
        let result = self.process(&padding)?;
        self.reset();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polyphase_creation() {
        let resampler = PolyphaseResampler::new(44100, 48000, 2, 256).unwrap();
        assert_eq!(resampler.input_rate(), 44100);
        assert_eq!(resampler.output_rate(), 48000);

        // 44100 / 48000 = 147/160 (GCD = 300)
        assert_eq!(resampler.interp_factor, 160);
        assert_eq!(resampler.decim_factor, 147);
    }

    #[test]
    fn test_polyphase_simple_ratio() {
        // 1:2 upsampling
        let resampler = PolyphaseResampler::new(22050, 44100, 1, 64).unwrap();
        assert_eq!(resampler.interp_factor, 2);
        assert_eq!(resampler.decim_factor, 1);
    }

    #[test]
    fn test_polyphase_resample() {
        let mut resampler = PolyphaseResampler::with_defaults(44100, 48000, 1).unwrap();

        // Generate test signal
        let input: Vec<f32> = (0..1000)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / 44100.0).sin() as f32)
            .collect();

        let output = resampler.process(&input).unwrap();

        // Output should be roughly 48000/44100 times the input length
        let expected_len = (1000.0 * 48000.0 / 44100.0) as usize;
        assert!(
            (output.len() as i32 - expected_len as i32).abs() < 50,
            "Expected ~{} samples, got {}",
            expected_len,
            output.len()
        );
    }

    #[test]
    fn test_polyphase_downsample() {
        let mut resampler = PolyphaseResampler::with_defaults(48000, 44100, 1).unwrap();

        let input: Vec<f32> = (0..2000).map(|i| (i as f32 / 100.0).sin()).collect();
        let output = resampler.process(&input).unwrap();

        // Output should be roughly 44100/48000 times the input length
        let expected_len = (2000.0 * 44100.0 / 48000.0) as usize;
        assert!(
            (output.len() as i32 - expected_len as i32).abs() < 50,
            "Expected ~{} samples, got {}",
            expected_len,
            output.len()
        );
    }

    #[test]
    fn test_filter_bank_creation() {
        let prototype = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bank = PolyphaseResampler::create_polyphase_bank(&prototype, 3, 2);

        // Should have 3 phases
        assert_eq!(bank.len(), 3);

        // Phase 0 should have samples at indices 0, 3
        assert_eq!(bank[0], vec![1.0, 4.0]);
        // Phase 1 should have samples at indices 1, 4
        assert_eq!(bank[1], vec![2.0, 5.0]);
        // Phase 2 should have samples at indices 2, 5
        assert_eq!(bank[2], vec![3.0, 6.0]);
    }

    #[test]
    fn test_polyphase_reset() {
        let mut resampler = PolyphaseResampler::with_defaults(44100, 48000, 1).unwrap();

        let input = vec![1.0f32; 100];
        let _ = resampler.process(&input).unwrap();

        resampler.reset();
        assert_eq!(resampler.output_phase, 0);
        assert!(resampler.input_buffer.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_polyphase_2x_upsample() {
        let mut resampler = PolyphaseResampler::new(22050, 44100, 1, 64).unwrap();
        let input: Vec<f32> = (0..100).map(|i| (i as f32).sin()).collect();
        let output = resampler.process(&input).unwrap();

        // 2x upsampling should produce roughly 2x samples
        assert!(
            (output.len() as i32 - 200).abs() < 20,
            "Expected ~200, got {}",
            output.len()
        );
    }

    #[test]
    fn test_polyphase_2x_downsample() {
        let mut resampler = PolyphaseResampler::new(44100, 22050, 1, 64).unwrap();
        let input: Vec<f32> = (0..200).map(|i| (i as f32).sin()).collect();
        let output = resampler.process(&input).unwrap();

        // 2x downsampling should produce roughly half the samples
        assert!(
            (output.len() as i32 - 100).abs() < 20,
            "Expected ~100, got {}",
            output.len()
        );
    }
}
