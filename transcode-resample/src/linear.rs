//! Linear interpolation resampler.
//!
//! Fast, low-quality resampling using linear interpolation between samples.
//! Best suited for previews or when quality is not critical.

use crate::error::{ResampleError, Result};
use crate::ResamplerImpl;

/// Linear interpolation resampler.
///
/// Uses simple linear interpolation between adjacent samples.
/// This is the fastest resampling method but has the lowest quality.
///
/// # Quality Characteristics
/// - Introduces aliasing artifacts
/// - High-frequency content is attenuated
/// - Suitable for previews or voice content
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct LinearResampler {
    input_rate: u32,
    output_rate: u32,
    ratio: f64,
    /// Fractional position within the input samples
    position: f64,
    /// Previous sample for interpolation (per channel)
    prev_samples: Vec<f32>,
    /// Number of channels
    channels: usize,
}

impl LinearResampler {
    /// Create a new linear resampler.
    ///
    /// # Arguments
    /// * `input_rate` - Input sample rate in Hz
    /// * `output_rate` - Output sample rate in Hz
    /// * `channels` - Number of audio channels
    ///
    /// # Errors
    /// Returns an error if sample rates or channel count are invalid.
    pub fn new(input_rate: u32, output_rate: u32, channels: usize) -> Result<Self> {
        if input_rate == 0 {
            return Err(ResampleError::InvalidSampleRate { rate: input_rate });
        }
        if output_rate == 0 {
            return Err(ResampleError::InvalidSampleRate { rate: output_rate });
        }
        if channels == 0 {
            return Err(ResampleError::InvalidChannelCount { count: channels });
        }

        let ratio = input_rate as f64 / output_rate as f64;

        Ok(Self {
            input_rate,
            output_rate,
            ratio,
            position: 0.0,
            prev_samples: vec![0.0; channels],
            channels,
        })
    }

    /// Perform linear interpolation between two samples.
    #[inline]
    fn interpolate(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
}

impl ResamplerImpl for LinearResampler {
    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate output size
        let output_len = ((input.len() as f64) / self.ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        let mut pos = self.position;
        let input_len = input.len();

        while (pos as usize) < input_len {
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;

            // Get samples for interpolation
            let s0 = if idx > 0 {
                input[idx - 1]
            } else {
                self.prev_samples[0]
            };
            let s1 = input[idx];

            output.push(Self::interpolate(s0, s1, frac));
            pos += self.ratio;
        }

        // Update state for next call
        self.position = pos - input_len as f64;
        if !input.is_empty() {
            self.prev_samples[0] = input[input_len - 1];
        }

        Ok(output)
    }

    fn process_interleaved(&mut self, input: &[f32], channels: usize) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        if !input.len().is_multiple_of(channels) {
            return Err(ResampleError::BufferSizeMismatch {
                actual: input.len(),
                channels,
            });
        }

        let input_frames = input.len() / channels;
        let output_frames = ((input_frames as f64) / self.ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_frames * channels);

        let mut pos = self.position;

        while (pos as usize) < input_frames {
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;

            for ch in 0..channels {
                let s0 = if idx > 0 {
                    input[(idx - 1) * channels + ch]
                } else {
                    self.prev_samples.get(ch).copied().unwrap_or(0.0)
                };
                let s1 = input[idx * channels + ch];

                output.push(Self::interpolate(s0, s1, frac));
            }

            pos += self.ratio;
        }

        // Update state for next call
        self.position = pos - input_frames as f64;
        if input_frames > 0 {
            for ch in 0..channels.min(self.prev_samples.len()) {
                self.prev_samples[ch] = input[(input_frames - 1) * channels + ch];
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
        self.position = 0.0;
        self.prev_samples.fill(0.0);
    }

    fn latency(&self) -> usize {
        // Linear interpolation has essentially 1 sample of latency
        1
    }

    fn flush(&mut self) -> Result<Vec<f32>> {
        // No buffered samples to flush in linear interpolation
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_resampler_creation() {
        let resampler = LinearResampler::new(44100, 48000, 2).unwrap();
        assert_eq!(resampler.input_rate(), 44100);
        assert_eq!(resampler.output_rate(), 48000);
    }

    #[test]
    fn test_linear_resampler_invalid_rates() {
        assert!(LinearResampler::new(0, 48000, 2).is_err());
        assert!(LinearResampler::new(44100, 0, 2).is_err());
        assert!(LinearResampler::new(44100, 48000, 0).is_err());
    }

    #[test]
    fn test_linear_interpolate() {
        assert_eq!(LinearResampler::interpolate(0.0, 1.0, 0.0), 0.0);
        assert_eq!(LinearResampler::interpolate(0.0, 1.0, 1.0), 1.0);
        assert_eq!(LinearResampler::interpolate(0.0, 1.0, 0.5), 0.5);
        assert_eq!(LinearResampler::interpolate(-1.0, 1.0, 0.5), 0.0);
    }

    #[test]
    fn test_linear_resample_upsample() {
        let mut resampler = LinearResampler::new(22050, 44100, 1).unwrap();
        let input: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0).sin()).collect();
        let output = resampler.process(&input).unwrap();

        // Should roughly double the samples
        assert!(output.len() >= 190 && output.len() <= 210);
    }

    #[test]
    fn test_linear_resample_downsample() {
        let mut resampler = LinearResampler::new(48000, 24000, 1).unwrap();
        let input: Vec<f32> = (0..200).map(|i| (i as f32 / 100.0).sin()).collect();
        let output = resampler.process(&input).unwrap();

        // Should roughly halve the samples
        assert!(output.len() >= 90 && output.len() <= 110);
    }

    #[test]
    fn test_linear_resample_interleaved() {
        let mut resampler = LinearResampler::new(44100, 48000, 2).unwrap();
        let input: Vec<f32> = (0..200)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let output = resampler.process_interleaved(&input, 2).unwrap();

        // Should have more frames than input
        assert!(output.len() > input.len());
        // Should be divisible by channel count
        assert_eq!(output.len() % 2, 0);
    }

    #[test]
    fn test_linear_reset() {
        let mut resampler = LinearResampler::new(44100, 48000, 2).unwrap();
        let input: Vec<f32> = vec![1.0; 100];
        let _ = resampler.process(&input).unwrap();

        resampler.reset();
        assert_eq!(resampler.position, 0.0);
        assert!(resampler.prev_samples.iter().all(|&s| s == 0.0));
    }
}
