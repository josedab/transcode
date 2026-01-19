//! Sinc interpolation resampler with configurable window functions.
//!
//! High-quality resampling using windowed sinc interpolation.
//! Supports various window functions for different quality/performance tradeoffs.

use crate::error::{ResampleError, Result};
use crate::simd;
use crate::ResamplerImpl;
use std::f64::consts::PI;

/// Window functions for sinc interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowFunction {
    /// Rectangular window (no windowing).
    Rectangular,
    /// Hann window - good balance of main lobe width and side lobe attenuation.
    Hann,
    /// Hamming window - slightly better side lobe attenuation than Hann.
    Hamming,
    /// Blackman window - excellent side lobe attenuation.
    #[default]
    Blackman,
    /// Kaiser window with configurable beta parameter.
    Kaiser { beta: u32 },
    /// Lanczos window (sinc window).
    Lanczos,
}


impl WindowFunction {
    /// Calculate the window value at position n for a window of size N.
    pub fn value(&self, n: usize, size: usize) -> f64 {
        let n = n as f64;
        let size = size as f64;

        match self {
            Self::Rectangular => 1.0,
            Self::Hann => 0.5 * (1.0 - (2.0 * PI * n / (size - 1.0)).cos()),
            Self::Hamming => 0.54 - 0.46 * (2.0 * PI * n / (size - 1.0)).cos(),
            Self::Blackman => {
                0.42 - 0.5 * (2.0 * PI * n / (size - 1.0)).cos()
                    + 0.08 * (4.0 * PI * n / (size - 1.0)).cos()
            }
            Self::Kaiser { beta } => {
                let beta = *beta as f64;
                let x = 2.0 * n / (size - 1.0) - 1.0;
                bessel_i0(beta * (1.0 - x * x).sqrt()) / bessel_i0(beta)
            }
            Self::Lanczos => {
                let x = 2.0 * n / (size - 1.0) - 1.0;
                if x.abs() < 1e-10 {
                    1.0
                } else {
                    (PI * x).sin() / (PI * x)
                }
            }
        }
    }
}

/// Approximate Bessel function I0 using a polynomial approximation.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y
            * (3.5156229
                + y * (3.0899424
                    + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + y * (0.01328592
                    + y * (0.00225319
                        + y * (-0.00157565
                            + y * (0.00916281
                                + y * (-0.02057706
                                    + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
    }
}

/// Sinc interpolation resampler.
///
/// Uses windowed sinc interpolation for high-quality resampling.
/// The window function and size can be configured for different quality levels.
///
/// # Quality Characteristics
/// - Very low aliasing with proper window selection
/// - Excellent frequency response
/// - Higher computational cost than linear interpolation
#[derive(Debug)]
#[allow(dead_code)]
pub struct SincResampler {
    input_rate: u32,
    output_rate: u32,
    ratio: f64,
    window_size: usize,
    window_function: WindowFunction,
    /// Pre-computed filter coefficients (window * sinc)
    filter_table: Vec<Vec<f32>>,
    /// Number of filter phases
    num_phases: usize,
    /// Input buffer for history
    input_buffer: Vec<f32>,
    /// Current position in input stream
    position: f64,
    /// Number of channels
    channels: usize,
}

impl SincResampler {
    /// Create a new sinc resampler.
    ///
    /// # Arguments
    /// * `input_rate` - Input sample rate in Hz
    /// * `output_rate` - Output sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `window_size` - Size of the sinc window (must be even, typically 32-256)
    /// * `window_function` - Window function to use
    ///
    /// # Errors
    /// Returns an error if parameters are invalid.
    pub fn new(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        window_size: usize,
        window_function: WindowFunction,
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
        if window_size == 0 || window_size % 2 != 0 {
            return Err(ResampleError::InvalidWindowSize { size: window_size });
        }

        let ratio = input_rate as f64 / output_rate as f64;

        // Check for extreme ratios
        if !(1.0 / 256.0..=256.0).contains(&ratio) {
            return Err(ResampleError::RatioTooExtreme { ratio });
        }

        // Calculate number of filter phases for interpolation
        // More phases = better interpolation quality
        let num_phases = 256;

        // Pre-compute filter table
        let filter_table = Self::compute_filter_table(
            window_size,
            num_phases,
            ratio,
            window_function,
        );

        // Initialize input buffer with zeros for history
        let input_buffer = vec![0.0; window_size * channels];

        Ok(Self {
            input_rate,
            output_rate,
            ratio,
            window_size,
            window_function,
            filter_table,
            num_phases,
            input_buffer,
            position: 0.0,
            channels,
        })
    }

    /// Create with default settings (Blackman window, 64 taps).
    pub fn with_defaults(input_rate: u32, output_rate: u32, channels: usize) -> Result<Self> {
        Self::new(input_rate, output_rate, channels, 64, WindowFunction::Blackman)
    }

    /// Compute the filter coefficient table.
    fn compute_filter_table(
        window_size: usize,
        num_phases: usize,
        ratio: f64,
        window_function: WindowFunction,
    ) -> Vec<Vec<f32>> {
        let half_size = window_size / 2;
        let cutoff = if ratio > 1.0 { 1.0 / ratio } else { 1.0 };

        (0..num_phases)
            .map(|phase| {
                let phase_offset = phase as f64 / num_phases as f64;

                (0..window_size)
                    .map(|i| {
                        let x = (i as f64 - half_size as f64) - phase_offset;

                        // Sinc function
                        let sinc = if x.abs() < 1e-10 {
                            cutoff
                        } else {
                            (PI * x * cutoff).sin() / (PI * x)
                        };

                        // Window function
                        let window = window_function.value(i, window_size);

                        (sinc * window * cutoff) as f32
                    })
                    .collect()
            })
            .collect()
    }

    /// Apply the sinc filter at the given position.
    #[inline]
    #[allow(unreachable_code)]
    fn apply_filter(&self, input: &[f32], start_idx: usize, phase: usize) -> f32 {
        let coeffs = &self.filter_table[phase];

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return simd::apply_filter_avx2(input, start_idx, coeffs, self.window_size);
            }
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        {
            return simd::apply_filter_neon(input, start_idx, coeffs, self.window_size);
        }

        // Scalar fallback
        Self::apply_filter_scalar(input, start_idx, coeffs, self.window_size)
    }

    /// Scalar implementation of filter application.
    #[inline]
    fn apply_filter_scalar(input: &[f32], start_idx: usize, coeffs: &[f32], window_size: usize) -> f32 {
        let mut sum = 0.0f32;
        let end = (start_idx + window_size).min(input.len());

        for i in start_idx..end {
            let coeff_idx = i - start_idx;
            if coeff_idx < coeffs.len() {
                sum += input[i] * coeffs[coeff_idx];
            }
        }

        sum
    }
}

impl ResamplerImpl for SincResampler {
    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let half_window = self.window_size / 2;

        // Prepend history buffer to input
        let mut extended_input = Vec::with_capacity(self.input_buffer.len() + input.len());
        extended_input.extend_from_slice(&self.input_buffer);
        extended_input.extend_from_slice(input);

        // Calculate output size
        let output_len = ((input.len() as f64) / self.ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        let mut pos = self.position + half_window as f64;
        let max_pos = extended_input.len() as f64 - half_window as f64;

        while pos < max_pos {
            let int_pos = pos.floor() as usize;
            let frac_pos = pos - int_pos as f64;

            // Select filter phase based on fractional position
            let phase = ((frac_pos * self.num_phases as f64) as usize).min(self.num_phases - 1);

            // Calculate start index for filter application
            let start_idx = int_pos.saturating_sub(half_window);

            // Apply filter
            let sample = self.apply_filter(&extended_input, start_idx, phase);
            output.push(sample);

            pos += self.ratio;
        }

        // Update position for next call
        self.position = pos - max_pos;

        // Update history buffer
        let history_start = if input.len() >= self.window_size {
            input.len() - self.window_size
        } else {
            0
        };

        self.input_buffer.clear();
        if input.len() >= self.window_size {
            self.input_buffer.extend_from_slice(&input[history_start..]);
        } else {
            // Keep some old history and add new input
            let keep = self.window_size - input.len();
            let old_start = self.input_buffer.len().saturating_sub(keep);
            let old_history: Vec<f32> = self.input_buffer[old_start..].to_vec();
            self.input_buffer.clear();
            self.input_buffer.extend_from_slice(&old_history);
            self.input_buffer.extend_from_slice(input);
        }
        self.input_buffer.resize(self.window_size, 0.0);

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

        let input_frames = input.len() / channels;
        let half_window = self.window_size / 2;
        let _history_frames = self.input_buffer.len() / channels;

        // Calculate output size
        let output_frames = ((input_frames as f64) / self.ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_frames * channels);

        // Process each channel separately for simplicity
        // (A more optimized version would process all channels together)
        for ch in 0..channels {
            // Extract channel data
            let channel_input: Vec<f32> = input.iter().skip(ch).step_by(channels).copied().collect();
            let channel_history: Vec<f32> = self.input_buffer.iter().skip(ch).step_by(channels).copied().collect();

            // Extend with history
            let mut extended: Vec<f32> = Vec::with_capacity(channel_history.len() + channel_input.len());
            extended.extend_from_slice(&channel_history);
            extended.extend_from_slice(&channel_input);

            let mut pos = self.position + half_window as f64;
            let max_pos = extended.len() as f64 - half_window as f64;

            let mut frame_idx = 0;
            while pos < max_pos {
                let int_pos = pos.floor() as usize;
                let frac_pos = pos - int_pos as f64;
                let phase = ((frac_pos * self.num_phases as f64) as usize).min(self.num_phases - 1);
                let start_idx = int_pos.saturating_sub(half_window);

                let sample = self.apply_filter(&extended, start_idx, phase);

                // Insert at correct interleaved position
                let output_pos = frame_idx * channels + ch;
                if output_pos >= output.len() {
                    output.resize(output_pos + 1, 0.0);
                }
                output[output_pos] = sample;

                pos += self.ratio;
                frame_idx += 1;
            }

            // Only update position on last channel
            if ch == channels - 1 {
                self.position = pos - max_pos;
            }
        }

        // Update history buffer
        let history_samples = self.window_size * channels;
        self.input_buffer.clear();
        if input.len() >= history_samples {
            self.input_buffer.extend_from_slice(&input[input.len() - history_samples..]);
        } else {
            let keep = history_samples - input.len();
            let old_history: Vec<f32> = self.input_buffer.iter().skip(self.input_buffer.len().saturating_sub(keep)).copied().collect();
            self.input_buffer.extend_from_slice(&old_history);
            self.input_buffer.extend_from_slice(input);
        }
        self.input_buffer.resize(history_samples, 0.0);

        Ok(output)
    }

    fn input_rate(&self) -> u32 {
        self.input_rate
    }

    fn output_rate(&self) -> u32 {
        self.output_rate
    }

    fn reset(&mut self) {
        self.position = 0.0;
        self.input_buffer.fill(0.0);
    }

    fn latency(&self) -> usize {
        self.window_size / 2
    }

    fn flush(&mut self) -> Result<Vec<f32>> {
        // Process remaining samples in buffer with zero-padding
        let padding = vec![0.0f32; self.window_size];
        let result = self.process(&padding)?;
        self.reset();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_functions() {
        let size = 64;

        // All windows should be 1.0 at center (or close to it)
        let hann = WindowFunction::Hann.value(size / 2, size);
        assert!((hann - 1.0).abs() < 0.1);

        let blackman = WindowFunction::Blackman.value(size / 2, size);
        assert!((blackman - 1.0).abs() < 0.1);

        // Windows should be 0 at edges (except rectangular)
        let hann_edge = WindowFunction::Hann.value(0, size);
        assert!(hann_edge < 0.01);
    }

    #[test]
    fn test_sinc_resampler_creation() {
        let resampler = SincResampler::new(44100, 48000, 2, 64, WindowFunction::Blackman).unwrap();
        assert_eq!(resampler.input_rate(), 44100);
        assert_eq!(resampler.output_rate(), 48000);
        assert_eq!(resampler.latency(), 32);
    }

    #[test]
    fn test_sinc_resampler_invalid_window() {
        assert!(SincResampler::new(44100, 48000, 2, 0, WindowFunction::Blackman).is_err());
        assert!(SincResampler::new(44100, 48000, 2, 63, WindowFunction::Blackman).is_err());
    }

    #[test]
    fn test_sinc_resample_upsample() {
        let mut resampler = SincResampler::with_defaults(22050, 44100, 1).unwrap();
        let input: Vec<f32> = (0..1000).map(|i| (i as f32 / 100.0).sin()).collect();
        let output = resampler.process(&input).unwrap();

        // Should roughly double the samples
        assert!(output.len() >= 1800 && output.len() <= 2200);
    }

    #[test]
    fn test_sinc_resample_downsample() {
        let mut resampler = SincResampler::with_defaults(48000, 24000, 1).unwrap();
        let input: Vec<f32> = (0..2000).map(|i| (i as f32 / 100.0).sin()).collect();
        let output = resampler.process(&input).unwrap();

        // Should roughly halve the samples
        assert!(output.len() >= 900 && output.len() <= 1100);
    }

    #[test]
    fn test_bessel_i0() {
        // Test against known values
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-6);
        assert!((bessel_i0(1.0) - 1.2660658).abs() < 1e-5);
        assert!((bessel_i0(2.0) - 2.2795853).abs() < 1e-5);
    }

    #[test]
    fn test_filter_table_generation() {
        let table = SincResampler::compute_filter_table(32, 64, 1.0, WindowFunction::Blackman);
        assert_eq!(table.len(), 64);
        assert_eq!(table[0].len(), 32);

        // Filter should be normalized (sum close to 1)
        let sum: f32 = table[0].iter().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }
}
