//! Spectral processing for audio enhancement.
//!
//! This module provides FFT-based audio processing including:
//! - Spectral noise reduction using Wiener filtering
//! - De-essing via dynamic frequency compression
//! - Spectral analysis utilities

use std::f32::consts::PI;

use crate::{AudioBuffer, Result};

/// FFT size constants.
pub const DEFAULT_FFT_SIZE: usize = 2048;
pub const MIN_FFT_SIZE: usize = 256;
pub const MAX_FFT_SIZE: usize = 8192;

/// Simple DFT implementation for spectral analysis.
/// In production, this would use a proper FFT library like rustfft.
pub struct SimpleDft {
    size: usize,
    twiddles: Vec<(f32, f32)>,
    window: Vec<f32>,
}

impl SimpleDft {
    /// Create a new DFT processor.
    pub fn new(size: usize) -> Self {
        let size = size.clamp(MIN_FFT_SIZE, MAX_FFT_SIZE);

        // Pre-compute twiddle factors
        let twiddles: Vec<(f32, f32)> = (0..size)
            .map(|k| {
                let angle = -2.0 * PI * k as f32 / size as f32;
                (angle.cos(), angle.sin())
            })
            .collect();

        // Create Hann window
        let window: Vec<f32> = (0..size)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (size - 1) as f32).cos()))
            .collect();

        Self {
            size,
            twiddles,
            window,
        }
    }

    /// Compute forward DFT, returning magnitude and phase.
    pub fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = self.size;
        let mut magnitudes = vec![0.0f32; n / 2 + 1];
        let mut phases = vec![0.0f32; n / 2 + 1];

        for k in 0..=n / 2 {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (i, &sample) in input.iter().take(n).enumerate() {
                let windowed = sample * self.window[i];
                let idx = (k * i) % n;
                real += windowed * self.twiddles[idx].0;
                imag += windowed * self.twiddles[idx].1;
            }

            magnitudes[k] = (real * real + imag * imag).sqrt();
            phases[k] = imag.atan2(real);
        }

        (magnitudes, phases)
    }

    /// Compute inverse DFT from magnitude and phase.
    pub fn inverse(&self, magnitudes: &[f32], phases: &[f32]) -> Vec<f32> {
        let n = self.size;
        let mut output = vec![0.0f32; n];

        for (i, out_sample) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;

            for (k, (&mag, &phase)) in magnitudes.iter().zip(phases.iter()).enumerate() {
                let angle = 2.0 * PI * k as f32 * i as f32 / n as f32 + phase;
                sum += mag * angle.cos();
            }

            // Also add negative frequencies for real signal
            for k in 1..magnitudes.len() - 1 {
                let mag = magnitudes[k];
                let phase = -phases[k];
                let freq = n - k;
                let angle = 2.0 * PI * freq as f32 * i as f32 / n as f32 + phase;
                sum += mag * angle.cos();
            }

            *out_sample = sum / n as f32;
        }

        output
    }

    /// Get the window function.
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// Get the FFT size.
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Spectral noise reducer using Wiener filtering.
pub struct SpectralNoiseReducer {
    fft: SimpleDft,
    hop_size: usize,
    noise_floor: Vec<f32>,
    noise_estimated: bool,
    oversubtraction: f32,
    spectral_floor: f32,
}

impl SpectralNoiseReducer {
    /// Create a new spectral noise reducer.
    pub fn new(fft_size: usize) -> Self {
        let fft = SimpleDft::new(fft_size);
        let hop_size = fft_size / 4;
        let num_bins = fft_size / 2 + 1;

        Self {
            fft,
            hop_size,
            noise_floor: vec![0.0; num_bins],
            noise_estimated: false,
            oversubtraction: 2.0,
            spectral_floor: 0.001,
        }
    }

    /// Set the oversubtraction factor (higher = more aggressive).
    pub fn set_oversubtraction(&mut self, factor: f32) {
        self.oversubtraction = factor.clamp(0.5, 10.0);
    }

    /// Set the spectral floor (minimum gain).
    pub fn set_spectral_floor(&mut self, floor: f32) {
        self.spectral_floor = floor.clamp(0.0001, 0.5);
    }

    /// Estimate noise from a silent segment.
    pub fn estimate_noise(&mut self, samples: &[f32]) {
        let frame_count = (samples.len().saturating_sub(self.fft.size())) / self.hop_size + 1;
        if frame_count == 0 {
            return;
        }

        // Average spectral magnitudes across frames
        let mut sum_mag = vec![0.0f32; self.noise_floor.len()];

        for frame_idx in 0..frame_count {
            let start = frame_idx * self.hop_size;
            let frame: Vec<f32> = samples
                .iter()
                .skip(start)
                .take(self.fft.size())
                .copied()
                .collect();

            if frame.len() < self.fft.size() {
                break;
            }

            let (magnitudes, _) = self.fft.forward(&frame);
            for (sum, mag) in sum_mag.iter_mut().zip(magnitudes.iter()) {
                *sum += mag;
            }
        }

        for (floor, sum) in self.noise_floor.iter_mut().zip(sum_mag.iter()) {
            *floor = sum / frame_count as f32;
        }

        self.noise_estimated = true;
    }

    /// Process an audio buffer to reduce noise.
    pub fn process(&self, buffer: &mut AudioBuffer) -> Result<ProcessStats> {
        if !self.noise_estimated {
            // Auto-estimate from first 50ms (assuming it's quiet)
            let estimate_samples = (buffer.sample_rate as f32 * 0.05) as usize;
            let channel = buffer.channel(0);
            let reducer = self.clone_with_noise_estimate(&channel[..estimate_samples.min(channel.len())]);
            return reducer.process(buffer);
        }

        let mut stats = ProcessStats::default();
        let mut total_reduction = 0.0f32;
        let mut frame_count = 0usize;

        // Process each channel
        for ch in 0..buffer.channels {
            let channel_samples: Vec<f32> = buffer.channel(ch);
            let mut processed = vec![0.0f32; channel_samples.len()];
            let mut overlap_buffer = vec![0.0f32; channel_samples.len()];

            let num_frames = (channel_samples.len().saturating_sub(self.fft.size())) / self.hop_size + 1;

            for frame_idx in 0..num_frames {
                let start = frame_idx * self.hop_size;
                let frame: Vec<f32> = channel_samples
                    .iter()
                    .skip(start)
                    .take(self.fft.size())
                    .copied()
                    .collect();

                if frame.len() < self.fft.size() {
                    break;
                }

                // Forward transform
                let (mut magnitudes, phases) = self.fft.forward(&frame);

                // Apply Wiener filter
                let mut frame_reduction = 0.0f32;
                for (k, mag) in magnitudes.iter_mut().enumerate() {
                    let noise = self.noise_floor[k] * self.oversubtraction;
                    let original_mag = *mag;

                    // Wiener filter gain
                    let signal_power = (*mag * *mag).max(0.0);
                    let noise_power = noise * noise;
                    let gain = if signal_power > noise_power {
                        ((signal_power - noise_power) / signal_power.max(0.0001)).sqrt()
                    } else {
                        self.spectral_floor
                    };

                    *mag *= gain;
                    frame_reduction += original_mag - *mag;
                }

                total_reduction += frame_reduction;
                frame_count += 1;

                // Inverse transform
                let output_frame = self.fft.inverse(&magnitudes, &phases);

                // Overlap-add
                for (i, &sample) in output_frame.iter().enumerate() {
                    let idx = start + i;
                    if idx < overlap_buffer.len() {
                        overlap_buffer[idx] += sample * self.fft.window()[i];
                    }
                }
            }

            // Normalize overlap-add
            let window_sum: f32 = (0..processed.len())
                .map(|i| {
                    let mut sum = 0.0f32;
                    let mut frame_idx = 0;
                    while frame_idx * self.hop_size <= i {
                        let local_i = i - frame_idx * self.hop_size;
                        if local_i < self.fft.size() {
                            sum += self.fft.window()[local_i] * self.fft.window()[local_i];
                        }
                        frame_idx += 1;
                    }
                    sum.max(0.001)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .sum::<f32>()
                / processed.len() as f32;

            for (i, sample) in processed.iter_mut().enumerate() {
                *sample = overlap_buffer[i] / window_sum.max(0.001);
            }

            // Write back to buffer
            for (i, frame_idx) in (0..buffer.num_frames()).enumerate() {
                let idx = frame_idx * buffer.channels + ch;
                if idx < buffer.samples.len() && i < processed.len() {
                    buffer.samples[idx] = processed[i];
                }
            }
        }

        stats.reduction_db = if frame_count > 0 {
            -20.0 * (total_reduction / frame_count as f32 + 0.001).log10()
        } else {
            0.0
        };

        Ok(stats)
    }

    fn clone_with_noise_estimate(&self, samples: &[f32]) -> Self {
        let mut new = Self::new(self.fft.size());
        new.oversubtraction = self.oversubtraction;
        new.spectral_floor = self.spectral_floor;
        new.estimate_noise(samples);
        new
    }
}

/// De-esser for reducing sibilance.
pub struct DeEsser {
    fft: SimpleDft,
    hop_size: usize,
    threshold: f32,
    reduction: f32,
    low_freq_hz: f32,
    high_freq_hz: f32,
}

impl DeEsser {
    /// Create a new de-esser.
    pub fn new(fft_size: usize) -> Self {
        Self {
            fft: SimpleDft::new(fft_size),
            hop_size: fft_size / 4,
            threshold: 0.3,
            reduction: 0.5,
            low_freq_hz: 4000.0,
            high_freq_hz: 9000.0,
        }
    }

    /// Set the detection threshold (0.0 to 1.0).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set the reduction amount (0.0 to 1.0).
    pub fn set_reduction(&mut self, reduction: f32) {
        self.reduction = reduction.clamp(0.0, 1.0);
    }

    /// Set the frequency range for sibilance detection.
    pub fn set_frequency_range(&mut self, low_hz: f32, high_hz: f32) {
        self.low_freq_hz = low_hz.clamp(1000.0, 8000.0);
        self.high_freq_hz = high_hz.clamp(self.low_freq_hz + 1000.0, 20000.0);
    }

    /// Process an audio buffer to reduce sibilance.
    pub fn process(&self, buffer: &mut AudioBuffer) -> Result<DeEsserStats> {
        let mut stats = DeEsserStats::default();
        let sample_rate = buffer.sample_rate as f32;

        // Calculate bin indices for sibilance range
        let bin_resolution = sample_rate / self.fft.size() as f32;
        let low_bin = (self.low_freq_hz / bin_resolution) as usize;
        let high_bin = ((self.high_freq_hz / bin_resolution) as usize).min(self.fft.size() / 2);

        let mut total_reduction_events = 0usize;

        for ch in 0..buffer.channels {
            let channel_samples: Vec<f32> = buffer.channel(ch);
            let mut processed = vec![0.0f32; channel_samples.len()];
            let mut overlap_buffer = vec![0.0f32; channel_samples.len()];

            let num_frames = (channel_samples.len().saturating_sub(self.fft.size())) / self.hop_size + 1;

            for frame_idx in 0..num_frames {
                let start = frame_idx * self.hop_size;
                let frame: Vec<f32> = channel_samples
                    .iter()
                    .skip(start)
                    .take(self.fft.size())
                    .copied()
                    .collect();

                if frame.len() < self.fft.size() {
                    break;
                }

                let (mut magnitudes, phases) = self.fft.forward(&frame);

                // Calculate energy in sibilance range
                let sibilance_energy: f32 = magnitudes[low_bin..high_bin]
                    .iter()
                    .map(|m| m * m)
                    .sum();

                // Calculate total energy
                let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();

                // Sibilance ratio
                let sibilance_ratio = if total_energy > 0.0001 {
                    sibilance_energy / total_energy
                } else {
                    0.0
                };

                // Apply de-essing if sibilance exceeds threshold
                if sibilance_ratio > self.threshold {
                    total_reduction_events += 1;

                    // Calculate gain reduction for sibilance frequencies
                    let gain = 1.0 - self.reduction * (sibilance_ratio - self.threshold) / (1.0 - self.threshold + 0.001);
                    let gain = gain.clamp(0.1, 1.0);

                    // Apply gain to sibilance frequency range with smooth transition
                    for (k, mag) in magnitudes.iter_mut().enumerate().take(high_bin).skip(low_bin) {
                        // Smooth transition at edges
                        let edge_factor = if k < low_bin + 5 {
                            (k - low_bin) as f32 / 5.0
                        } else if k > high_bin - 5 {
                            (high_bin - k) as f32 / 5.0
                        } else {
                            1.0
                        };

                        *mag *= 1.0 - (1.0 - gain) * edge_factor;
                    }
                }

                // Inverse transform
                let output_frame = self.fft.inverse(&magnitudes, &phases);

                // Overlap-add
                for (i, &sample) in output_frame.iter().enumerate() {
                    let idx = start + i;
                    if idx < overlap_buffer.len() {
                        overlap_buffer[idx] += sample * self.fft.window()[i];
                    }
                }
            }

            // Normalize overlap-add
            let window_sum = 0.375; // Approximate for Hann window with 75% overlap
            for (i, sample) in processed.iter_mut().enumerate() {
                *sample = overlap_buffer[i] / window_sum;
            }

            // Write back
            for (i, frame_idx) in (0..buffer.num_frames()).enumerate() {
                let idx = frame_idx * buffer.channels + ch;
                if idx < buffer.samples.len() && i < processed.len() {
                    buffer.samples[idx] = processed[i];
                }
            }
        }

        stats.reduction_events = total_reduction_events;
        Ok(stats)
    }
}

/// Statistics from noise reduction processing.
#[derive(Debug, Clone, Default)]
pub struct ProcessStats {
    /// Amount of noise reduction in dB.
    pub reduction_db: f32,
}

/// Statistics from de-essing processing.
#[derive(Debug, Clone, Default)]
pub struct DeEsserStats {
    /// Number of frames where de-essing was applied.
    pub reduction_events: usize,
}

/// Spectral analyzer for audio analysis.
pub struct SpectralAnalyzer {
    fft: SimpleDft,
}

impl SpectralAnalyzer {
    /// Create a new spectral analyzer.
    pub fn new(fft_size: usize) -> Self {
        Self {
            fft: SimpleDft::new(fft_size),
        }
    }

    /// Analyze spectrum of audio buffer.
    pub fn analyze(&self, buffer: &AudioBuffer) -> SpectrumAnalysis {
        let channel = buffer.channel(0);
        let frame: Vec<f32> = channel.iter().take(self.fft.size()).copied().collect();

        let (magnitudes, _) = self.fft.forward(&frame);

        // Calculate spectral centroid
        let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
        let weighted_sum: f32 = magnitudes
            .iter()
            .enumerate()
            .map(|(k, m)| k as f32 * m * m)
            .sum();

        let centroid_bin = if total_energy > 0.0 {
            weighted_sum / total_energy
        } else {
            0.0
        };

        let bin_resolution = buffer.sample_rate as f32 / self.fft.size() as f32;

        SpectrumAnalysis {
            magnitudes,
            spectral_centroid_hz: centroid_bin * bin_resolution,
            total_energy,
        }
    }
}

/// Result of spectral analysis.
#[derive(Debug, Clone)]
pub struct SpectrumAnalysis {
    /// Magnitude spectrum.
    pub magnitudes: Vec<f32>,
    /// Spectral centroid in Hz.
    pub spectral_centroid_hz: f32,
    /// Total energy.
    pub total_energy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_dft() {
        let dft = SimpleDft::new(256);

        // Test with a simple sine wave
        let freq = 10.0;
        let samples: Vec<f32> = (0..256)
            .map(|i| (2.0 * PI * freq * i as f32 / 256.0).sin())
            .collect();

        let (magnitudes, _phases) = dft.forward(&samples);

        // Peak should be around bin 10
        let max_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        assert!((max_bin as i32 - 10).abs() <= 1, "Peak at wrong bin: {}", max_bin);
    }

    #[test]
    fn test_noise_reducer() {
        let mut reducer = SpectralNoiseReducer::new(512);

        // Generate deterministic pseudo-noise samples for estimation
        // Using a simple hash-like pattern for reproducibility
        let noise: Vec<f32> = (0..1024)
            .map(|i| {
                // Use wrapping arithmetic to avoid overflow
                let x = (i as u32).wrapping_mul(2654435761) ^ 0x12345678;
                let normalized = (x % 65536) as f32 / 65536.0;
                (normalized - 0.5) * 0.1
            })
            .collect();

        reducer.estimate_noise(&noise);
        assert!(reducer.noise_estimated);
    }

    #[test]
    fn test_deesser_creation() {
        let mut deesser = DeEsser::new(1024);

        deesser.set_threshold(0.5);
        deesser.set_reduction(0.7);
        deesser.set_frequency_range(5000.0, 8000.0);

        assert!((deesser.threshold - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_spectral_analyzer() {
        let analyzer = SpectralAnalyzer::new(512);

        let buffer = AudioBuffer::from_samples(
            (0..1024).map(|i| (i as f32 * 0.1).sin()).collect(),
            1,
            44100,
        );

        let analysis = analyzer.analyze(&buffer);
        assert!(!analysis.magnitudes.is_empty());
        assert!(analysis.spectral_centroid_hz >= 0.0);
    }
}
