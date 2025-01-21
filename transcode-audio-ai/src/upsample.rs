//! Audio upsampling algorithms
//!
//! Provides various upsampling methods including:
//! - Linear interpolation (fast, low quality)
//! - Windowed sinc interpolation (medium quality)
//! - Polyphase filter bank (high quality, efficient)

use crate::{AudioBuffer, Result};
use std::f64::consts::PI;

/// Upsampling quality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsampleQuality {
    /// Fast (linear interpolation)
    Fast,
    /// Medium (windowed sinc)
    Medium,
    /// High (polyphase)
    High,
    /// Best (neural network assisted)
    Best,
}

/// Polyphase filter bank for efficient resampling
///
/// A polyphase filter decomposes a lowpass FIR filter into L subfilters,
/// where L is the upsampling factor. Each subfilter produces one output
/// sample per input sample, making the computation much more efficient
/// than direct convolution.
struct PolyphaseFilterBank {
    /// Number of phases (upsampling factor)
    num_phases: usize,
    /// Filter coefficients for each phase
    /// Each phase has `taps_per_phase` coefficients
    coefficients: Vec<Vec<f64>>,
    /// Number of taps per phase
    taps_per_phase: usize,
}

impl PolyphaseFilterBank {
    /// Create a new polyphase filter bank
    ///
    /// # Arguments
    /// * `upsampling_factor` - The upsampling ratio L
    /// * `filter_order` - Number of taps per phase (higher = better quality)
    /// * `cutoff` - Cutoff frequency as fraction of Nyquist (typically 0.9-0.95)
    fn new(upsampling_factor: usize, filter_order: usize, cutoff: f64) -> Self {
        let num_phases = upsampling_factor;
        let taps_per_phase = filter_order;
        let total_taps = num_phases * taps_per_phase;

        // Design the prototype lowpass filter using windowed sinc
        let prototype = Self::design_prototype_filter(total_taps, cutoff / num_phases as f64);

        // Decompose into polyphase components
        let mut coefficients = vec![vec![0.0; taps_per_phase]; num_phases];

        for (i, &coeff) in prototype.iter().enumerate() {
            let phase = i % num_phases;
            let tap = i / num_phases;
            coefficients[phase][tap] = coeff;
        }

        Self {
            num_phases,
            coefficients,
            taps_per_phase,
        }
    }

    /// Design the prototype lowpass filter using Kaiser-windowed sinc
    fn design_prototype_filter(num_taps: usize, cutoff: f64) -> Vec<f64> {
        let mut filter = vec![0.0; num_taps];
        let center = (num_taps - 1) as f64 / 2.0;

        // Kaiser window parameter (beta = 5.0 gives good stopband attenuation)
        let beta = 5.0;

        // Compute sinc * window
        for (i, coeff) in filter.iter_mut().enumerate() {
            let n = i as f64 - center;

            // Sinc function
            let sinc = if n.abs() < 1e-10 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * n).sin() / (PI * n)
            };

            // Kaiser window
            let window = Self::kaiser_window(i, num_taps, beta);

            *coeff = sinc * window;
        }

        // Normalize so that DC gain = upsampling factor
        let sum: f64 = filter.iter().sum();
        if sum.abs() > 1e-10 {
            for coeff in &mut filter {
                *coeff /= sum;
            }
        }

        filter
    }

    /// Kaiser window function
    fn kaiser_window(n: usize, length: usize, beta: f64) -> f64 {
        let center = (length - 1) as f64 / 2.0;
        let x = (n as f64 - center) / center;
        let arg = beta * (1.0 - x * x).max(0.0).sqrt();
        Self::bessel_i0(arg) / Self::bessel_i0(beta)
    }

    /// Modified Bessel function of the first kind, order 0
    /// Used for Kaiser window calculation
    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x_sq = x * x / 4.0;

        for k in 1..50 {
            term *= x_sq / (k as f64 * k as f64);
            sum += term;
            if term < 1e-20 {
                break;
            }
        }

        sum
    }

    /// Apply the polyphase filter to upsample a single channel
    fn apply(&self, input: &[f32]) -> Vec<f32> {
        let output_len = input.len() * self.num_phases;

        // For each output sample, compute the convolution with the appropriate phase
        (0..output_len)
            .map(|out_idx| {
                let phase = out_idx % self.num_phases;
                let in_base = out_idx / self.num_phases;

                // Convolve with the phase's filter coefficients
                let sum: f64 = self.coefficients[phase]
                    .iter()
                    .enumerate()
                    .filter_map(|(tap, &coeff)| {
                        let in_idx = in_base as i64 - tap as i64 + (self.taps_per_phase / 2) as i64;
                        if in_idx >= 0 && (in_idx as usize) < input.len() {
                            Some(input[in_idx as usize] as f64 * coeff)
                        } else {
                            None
                        }
                    })
                    .sum();

                (sum * self.num_phases as f64) as f32
            })
            .collect()
    }
}

/// Rational resampler using polyphase filters
///
/// For arbitrary rate conversion, we use the formula:
/// output_rate / input_rate = L / M
/// where L is upsampling factor and M is downsampling factor
#[allow(dead_code)]
struct RationalResampler {
    /// Upsampling factor
    up_factor: usize,
    /// Downsampling factor
    down_factor: usize,
    /// Polyphase filter for upsampling
    filter: PolyphaseFilterBank,
}

impl RationalResampler {
    /// Create a resampler for the given rate conversion
    fn new(input_rate: u32, output_rate: u32, filter_order: usize) -> Self {
        let gcd = Self::gcd(input_rate, output_rate);
        let up_factor = (output_rate / gcd) as usize;
        let down_factor = (input_rate / gcd) as usize;

        // Cutoff at the lower of the two Nyquist frequencies
        let cutoff = 0.95_f64.min((input_rate as f64 / output_rate as f64) * 0.95);

        let filter = PolyphaseFilterBank::new(up_factor, filter_order, cutoff);

        Self {
            up_factor,
            down_factor,
            filter,
        }
    }

    /// Greatest common divisor using Euclidean algorithm
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a
        } else {
            Self::gcd(b, a % b)
        }
    }

    /// Resample a single channel
    fn resample_channel(&self, input: &[f32]) -> Vec<f32> {
        // First upsample by L
        let upsampled = self.filter.apply(input);

        // Then downsample by M (take every M-th sample)
        if self.down_factor == 1 {
            upsampled
        } else {
            upsampled
                .into_iter()
                .step_by(self.down_factor)
                .collect()
        }
    }
}

/// Audio upsampler
pub struct AudioUpsampler {
    quality: UpsampleQuality,
}

impl AudioUpsampler {
    /// Create a new upsampler
    pub fn new(quality: UpsampleQuality) -> Self {
        Self { quality }
    }

    /// Upsample audio to target sample rate
    pub fn upsample(&self, buffer: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        if target_rate <= buffer.sample_rate {
            return Ok(buffer.clone());
        }

        match self.quality {
            UpsampleQuality::Fast => self.linear_upsample(buffer, target_rate),
            UpsampleQuality::Medium => self.sinc_upsample(buffer, target_rate),
            UpsampleQuality::High => self.polyphase_upsample(buffer, target_rate),
            UpsampleQuality::Best => self.polyphase_upsample(buffer, target_rate), // Placeholder
        }
    }

    fn linear_upsample(&self, buffer: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        let ratio = target_rate as f64 / buffer.sample_rate as f64;
        let new_len = (buffer.num_frames() as f64 * ratio) as usize;

        let mut output = AudioBuffer::new(buffer.channels, target_rate);
        output.samples.reserve(new_len * buffer.channels);

        for ch in 0..buffer.channels {
            let channel = buffer.channel(ch);

            for i in 0..new_len {
                let pos = i as f64 / ratio;
                let idx = pos as usize;
                let frac = pos - idx as f64;

                let s0 = channel.get(idx).copied().unwrap_or(0.0);
                let s1 = channel.get(idx + 1).copied().unwrap_or(s0);

                let sample = s0 + (s1 - s0) * frac as f32;
                output.samples.push(sample);
            }
        }

        // Interleave channels
        if buffer.channels > 1 {
            output.samples = self.interleave(&output.samples, buffer.channels, new_len);
        }

        Ok(output)
    }

    fn sinc_upsample(&self, buffer: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        let ratio = target_rate as f64 / buffer.sample_rate as f64;
        let new_len = (buffer.num_frames() as f64 * ratio) as usize;

        let mut output = AudioBuffer::new(buffer.channels, target_rate);
        output.samples.reserve(new_len * buffer.channels);

        let filter_size = 32;

        for ch in 0..buffer.channels {
            let channel = buffer.channel(ch);

            for i in 0..new_len {
                let pos = i as f64 / ratio;
                let idx = pos as usize;
                let frac = pos - idx as f64;

                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for k in -filter_size..=filter_size {
                    let sample_idx = idx as i32 + k;
                    if sample_idx >= 0 && (sample_idx as usize) < channel.len() {
                        let x = k as f64 - frac;
                        let sinc = if x.abs() < 0.0001 {
                            1.0
                        } else {
                            (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                        };
                        let window = 0.5 + 0.5 * (std::f64::consts::PI * k as f64 / filter_size as f64).cos();
                        let weight = sinc * window;

                        sum += channel[sample_idx as usize] as f64 * weight;
                        weight_sum += weight;
                    }
                }

                let sample = if weight_sum > 0.0 {
                    (sum / weight_sum) as f32
                } else {
                    0.0
                };
                output.samples.push(sample);
            }
        }

        if buffer.channels > 1 {
            output.samples = self.interleave(&output.samples, buffer.channels, new_len);
        }

        Ok(output)
    }

    fn polyphase_upsample(&self, buffer: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        // Use higher filter order for better quality
        let filter_order = 32;
        let resampler = RationalResampler::new(buffer.sample_rate, target_rate, filter_order);

        let expected_frames =
            (buffer.num_frames() as u64 * target_rate as u64 / buffer.sample_rate as u64) as usize;
        let mut output = AudioBuffer::new(buffer.channels, target_rate);
        output.samples.reserve(expected_frames * buffer.channels);

        // Resample each channel
        let mut channels_data = Vec::with_capacity(buffer.channels);
        for ch in 0..buffer.channels {
            let channel = buffer.channel(ch);
            let resampled = resampler.resample_channel(&channel);
            channels_data.push(resampled);
        }

        // Interleave the resampled channels
        if buffer.channels == 1 {
            output.samples = channels_data.into_iter().next().unwrap_or_default();
        } else {
            let frame_count = channels_data
                .iter()
                .map(|c| c.len())
                .min()
                .unwrap_or(0);

            for f in 0..frame_count {
                for ch_data in &channels_data {
                    output.samples.push(ch_data[f]);
                }
            }
        }

        Ok(output)
    }

    fn interleave(&self, data: &[f32], channels: usize, frames: usize) -> Vec<f32> {
        let mut interleaved = vec![0.0; frames * channels];

        for ch in 0..channels {
            for f in 0..frames {
                interleaved[f * channels + ch] = data[ch * frames + f];
            }
        }

        interleaved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_upsample() {
        let upsampler = AudioUpsampler::new(UpsampleQuality::Fast);
        let buffer = AudioBuffer::from_samples(vec![0.0, 1.0, 0.0, -1.0], 1, 44100);

        let result = upsampler.upsample(&buffer, 88200).unwrap();
        assert_eq!(result.sample_rate, 88200);
        assert!(result.num_frames() > buffer.num_frames());
    }

    #[test]
    fn test_polyphase_upsample_2x() {
        let upsampler = AudioUpsampler::new(UpsampleQuality::High);

        // Create a simple test signal (DC offset)
        let samples: Vec<f32> = vec![0.5; 100];
        let buffer = AudioBuffer::from_samples(samples, 1, 44100);

        let result = upsampler.upsample(&buffer, 88200).unwrap();
        assert_eq!(result.sample_rate, 88200);
        // 2x upsampling should roughly double the frame count
        assert!(result.num_frames() >= 190 && result.num_frames() <= 210);
    }

    #[test]
    fn test_polyphase_upsample_preserves_dc() {
        let upsampler = AudioUpsampler::new(UpsampleQuality::High);

        // DC signal should be preserved
        let samples: Vec<f32> = vec![0.5; 200];
        let buffer = AudioBuffer::from_samples(samples, 1, 44100);

        let result = upsampler.upsample(&buffer, 88200).unwrap();

        // Check that output samples are close to the input DC value
        // (after filter settling)
        let mid_start = result.num_frames() / 4;
        let mid_end = 3 * result.num_frames() / 4;
        for sample in &result.samples[mid_start..mid_end] {
            assert!(
                (*sample - 0.5).abs() < 0.1,
                "DC level not preserved: expected ~0.5, got {}",
                sample
            );
        }
    }

    #[test]
    fn test_polyphase_upsample_stereo() {
        let upsampler = AudioUpsampler::new(UpsampleQuality::High);

        // Stereo signal with different values per channel
        let samples: Vec<f32> = (0..200)
            .flat_map(|_| [0.3, 0.7])
            .collect();
        let buffer = AudioBuffer::from_samples(samples, 2, 44100);

        let result = upsampler.upsample(&buffer, 88200).unwrap();
        assert_eq!(result.sample_rate, 88200);
        assert_eq!(result.channels, 2);

        // Check that channels are still interleaved correctly
        let left = result.channel(0);
        let right = result.channel(1);

        // Middle samples should preserve channel separation
        let mid = left.len() / 2;
        assert!(left[mid] < right[mid], "Channel separation not preserved");
    }

    #[test]
    fn test_polyphase_filter_bank_creation() {
        let filter = PolyphaseFilterBank::new(2, 16, 0.9);
        assert_eq!(filter.num_phases, 2);
        assert_eq!(filter.taps_per_phase, 16);
        assert_eq!(filter.coefficients.len(), 2);
        assert_eq!(filter.coefficients[0].len(), 16);
    }

    #[test]
    fn test_kaiser_window() {
        // Kaiser window should be 1.0 at center for any beta
        let center = 31; // For length 63
        let window = PolyphaseFilterBank::kaiser_window(center, 63, 5.0);
        assert!((window - 1.0).abs() < 0.001, "Kaiser window not 1.0 at center");

        // Window should be symmetric
        let w_start = PolyphaseFilterBank::kaiser_window(0, 63, 5.0);
        let w_end = PolyphaseFilterBank::kaiser_window(62, 63, 5.0);
        assert!((w_start - w_end).abs() < 0.001, "Kaiser window not symmetric");
    }

    #[test]
    fn test_bessel_i0() {
        // I0(0) = 1
        assert!((PolyphaseFilterBank::bessel_i0(0.0) - 1.0).abs() < 1e-10);

        // Known values
        // I0(1) ≈ 1.2661
        assert!((PolyphaseFilterBank::bessel_i0(1.0) - 1.2661).abs() < 0.001);

        // I0(2) ≈ 2.2796
        assert!((PolyphaseFilterBank::bessel_i0(2.0) - 2.2796).abs() < 0.001);
    }

    #[test]
    fn test_rational_resampler_gcd() {
        // 44100 to 48000: GCD = 300, up = 160, down = 147
        assert_eq!(RationalResampler::gcd(44100, 48000), 300);

        // 44100 to 88200: GCD = 44100, up = 2, down = 1
        assert_eq!(RationalResampler::gcd(44100, 88200), 44100);

        // 48000 to 96000: GCD = 48000, up = 2, down = 1
        assert_eq!(RationalResampler::gcd(48000, 96000), 48000);
    }

    #[test]
    fn test_sinc_upsample() {
        let upsampler = AudioUpsampler::new(UpsampleQuality::Medium);
        let buffer = AudioBuffer::from_samples(vec![0.0, 1.0, 0.0, -1.0], 1, 44100);

        let result = upsampler.upsample(&buffer, 88200).unwrap();
        assert_eq!(result.sample_rate, 88200);
        assert!(result.num_frames() > buffer.num_frames());
    }

    #[test]
    fn test_no_upsample_when_equal_or_lower() {
        let upsampler = AudioUpsampler::new(UpsampleQuality::High);
        let buffer = AudioBuffer::from_samples(vec![0.5; 100], 1, 44100);

        // Same rate
        let result = upsampler.upsample(&buffer, 44100).unwrap();
        assert_eq!(result.num_frames(), 100);

        // Lower rate (should just clone)
        let result = upsampler.upsample(&buffer, 22050).unwrap();
        assert_eq!(result.num_frames(), 100);
    }
}
