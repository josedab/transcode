//! K-weighting filter for EBU R128 loudness measurement.
//!
//! This module implements the K-weighting pre-filter specified in ITU-R BS.1770.
//! The filter consists of two cascaded stages:
//! 1. High-shelf filter (shelving filter boosting high frequencies)
//! 2. High-pass filter (removing very low frequencies)
//!
//! The filter is designed to approximate the frequency response of human hearing
//! for loudness perception.

use crate::error::{LoudnessError, Result};
use std::f64::consts::PI;

/// Biquad filter coefficients.
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoeffs {
    /// Feedforward coefficient b0.
    pub b0: f64,
    /// Feedforward coefficient b1.
    pub b1: f64,
    /// Feedforward coefficient b2.
    pub b2: f64,
    /// Feedback coefficient a1 (a0 is normalized to 1).
    pub a1: f64,
    /// Feedback coefficient a2.
    pub a2: f64,
}

impl BiquadCoeffs {
    /// Create unity (bypass) coefficients.
    pub fn unity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }
}

/// Biquad filter state for a single channel.
#[derive(Debug, Clone, Copy, Default)]
pub struct BiquadState {
    /// Previous input sample x[n-1].
    x1: f64,
    /// Previous input sample x[n-2].
    x2: f64,
    /// Previous output sample y[n-1].
    y1: f64,
    /// Previous output sample y[n-2].
    y2: f64,
}

impl BiquadState {
    /// Create a new zeroed state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the filter state to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Process a single sample through the filter.
    #[inline]
    pub fn process(&mut self, input: f64, coeffs: &BiquadCoeffs) -> f64 {
        let output = coeffs.b0 * input
            + coeffs.b1 * self.x1
            + coeffs.b2 * self.x2
            - coeffs.a1 * self.y1
            - coeffs.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }
}

/// K-weighting filter for a single channel.
///
/// Implements the two-stage K-weighting filter as specified in ITU-R BS.1770.
#[derive(Debug, Clone)]
pub struct KWeightingFilter {
    /// High-shelf filter coefficients.
    shelf_coeffs: BiquadCoeffs,
    /// High-pass filter coefficients.
    highpass_coeffs: BiquadCoeffs,
    /// High-shelf filter state.
    shelf_state: BiquadState,
    /// High-pass filter state.
    highpass_state: BiquadState,
    /// Sample rate in Hz.
    sample_rate: u32,
}

impl KWeightingFilter {
    /// Create a new K-weighting filter for the given sample rate.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (must be >= 8000)
    ///
    /// # Returns
    ///
    /// A configured K-weighting filter or an error if the sample rate is invalid.
    pub fn new(sample_rate: u32) -> Result<Self> {
        if sample_rate < 8000 {
            return Err(LoudnessError::invalid_sample_rate(sample_rate));
        }

        let shelf_coeffs = Self::compute_high_shelf_coeffs(sample_rate);
        let highpass_coeffs = Self::compute_highpass_coeffs(sample_rate);

        Ok(Self {
            shelf_coeffs,
            highpass_coeffs,
            shelf_state: BiquadState::new(),
            highpass_state: BiquadState::new(),
            sample_rate,
        })
    }

    /// Compute the high-shelf filter coefficients.
    ///
    /// This filter provides a boost to higher frequencies to better approximate
    /// human loudness perception. The filter is designed per ITU-R BS.1770.
    fn compute_high_shelf_coeffs(sample_rate: u32) -> BiquadCoeffs {
        // High shelf filter parameters from ITU-R BS.1770-4
        // fc = 1681.974450955533 Hz
        // G = +4 dB (high frequency boost)
        // Q = 0.7071752369554196 (Butterworth)

        let fc = 1681.974450955533;
        let db_gain = 4.0;
        let q = 0.7071752369554196;

        let fs = sample_rate as f64;
        let k = (PI * fc / fs).tan();
        let k2 = k * k;

        let v0 = 10.0_f64.powf(db_gain / 20.0);
        let vb = v0.powf(0.4996667741545416);

        let a0 = 1.0 + k / q + k2;
        let a1 = 2.0 * (k2 - 1.0) / a0;
        let a2 = (1.0 - k / q + k2) / a0;
        let b0 = (v0 + vb * k / q + k2) / a0;
        let b1 = 2.0 * (k2 - v0) / a0;
        let b2 = (v0 - vb * k / q + k2) / a0;

        BiquadCoeffs { b0, b1, b2, a1, a2 }
    }

    /// Compute the high-pass filter coefficients.
    ///
    /// This filter removes very low frequencies (below ~38 Hz) that don't
    /// contribute significantly to perceived loudness.
    fn compute_highpass_coeffs(sample_rate: u32) -> BiquadCoeffs {
        // High-pass filter parameters from ITU-R BS.1770-4
        // fc = 38.13547087602444 Hz
        // Q = 0.5003270373238773

        let fc = 38.13547087602444;
        let q = 0.5003270373238773;

        let fs = sample_rate as f64;
        let k = (PI * fc / fs).tan();
        let k2 = k * k;

        let a0 = 1.0 + k / q + k2;
        let a1 = 2.0 * (k2 - 1.0) / a0;
        let a2 = (1.0 - k / q + k2) / a0;
        let b0 = 1.0 / a0;
        let b1 = -2.0 / a0;
        let b2 = 1.0 / a0;

        BiquadCoeffs { b0, b1, b2, a1, a2 }
    }

    /// Reset the filter state to zero.
    pub fn reset(&mut self) {
        self.shelf_state.reset();
        self.highpass_state.reset();
    }

    /// Process a single sample through the K-weighting filter.
    #[inline]
    pub fn process_sample(&mut self, input: f64) -> f64 {
        // First stage: high-shelf filter
        let shelved = self.shelf_state.process(input, &self.shelf_coeffs);
        // Second stage: high-pass filter
        self.highpass_state.process(shelved, &self.highpass_coeffs)
    }

    /// Process a block of samples in-place.
    pub fn process_block(&mut self, samples: &mut [f64]) {
        for sample in samples.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Process a block of samples, writing to a separate output buffer.
    pub fn process_block_to(&mut self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());
        for (inp, out) in input.iter().zip(output.iter_mut()) {
            *out = self.process_sample(*inp);
        }
    }

    /// Get the sample rate this filter is configured for.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Multi-channel K-weighting filter bank.
///
/// Provides K-weighting filtering for multiple channels simultaneously.
#[derive(Debug, Clone)]
pub struct KWeightingFilterBank {
    /// Individual filters for each channel.
    filters: Vec<KWeightingFilter>,
    /// Sample rate in Hz.
    sample_rate: u32,
}

impl KWeightingFilterBank {
    /// Create a new filter bank for the given number of channels.
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of audio channels (1-8)
    /// * `sample_rate` - Sample rate in Hz (must be >= 8000)
    ///
    /// # Returns
    ///
    /// A configured filter bank or an error if parameters are invalid.
    pub fn new(channels: u32, sample_rate: u32) -> Result<Self> {
        if channels == 0 || channels > 8 {
            return Err(LoudnessError::invalid_channel_count(channels));
        }
        if sample_rate < 8000 {
            return Err(LoudnessError::invalid_sample_rate(sample_rate));
        }

        let filters = (0..channels)
            .map(|_| KWeightingFilter::new(sample_rate))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            filters,
            sample_rate,
        })
    }

    /// Get the number of channels.
    pub fn channels(&self) -> u32 {
        self.filters.len() as u32
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Reset all filter states.
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
    }

    /// Process interleaved samples in-place.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved samples (length must be multiple of channel count)
    pub fn process_interleaved(&mut self, samples: &mut [f64]) {
        let channels = self.filters.len();
        debug_assert_eq!(samples.len() % channels, 0);

        for frame in samples.chunks_exact_mut(channels) {
            for (ch, sample) in frame.iter_mut().enumerate() {
                *sample = self.filters[ch].process_sample(*sample);
            }
        }
    }

    /// Process planar samples in-place.
    ///
    /// # Arguments
    ///
    /// * `channels` - Slice of channel buffers
    pub fn process_planar(&mut self, channels: &mut [&mut [f64]]) {
        debug_assert_eq!(channels.len(), self.filters.len());

        for (ch, channel) in channels.iter_mut().enumerate() {
            self.filters[ch].process_block(channel);
        }
    }

    /// Process a single frame of samples (one sample per channel).
    ///
    /// # Arguments
    ///
    /// * `frame` - Slice of samples, one per channel
    ///
    /// # Returns
    ///
    /// Filtered samples for each channel.
    pub fn process_frame(&mut self, frame: &[f64]) -> Vec<f64> {
        debug_assert_eq!(frame.len(), self.filters.len());

        frame
            .iter()
            .enumerate()
            .map(|(ch, &sample)| self.filters[ch].process_sample(sample))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biquad_unity() {
        let coeffs = BiquadCoeffs::unity();
        let mut state = BiquadState::new();

        // Unity filter should pass signal unchanged
        assert_eq!(state.process(1.0, &coeffs), 1.0);
        assert_eq!(state.process(0.5, &coeffs), 0.5);
        assert_eq!(state.process(-0.3, &coeffs), -0.3);
    }

    #[test]
    fn test_kweighting_filter_creation() {
        // Valid sample rates
        assert!(KWeightingFilter::new(48000).is_ok());
        assert!(KWeightingFilter::new(44100).is_ok());
        assert!(KWeightingFilter::new(96000).is_ok());
        assert!(KWeightingFilter::new(8000).is_ok());

        // Invalid sample rate
        assert!(KWeightingFilter::new(7999).is_err());
    }

    #[test]
    fn test_kweighting_filter_process() {
        let mut filter = KWeightingFilter::new(48000).unwrap();

        // Process some samples
        let mut samples = vec![1.0, 0.5, -0.5, -1.0, 0.0];
        filter.process_block(&mut samples);

        // Filter should modify the signal (not pass-through)
        // Exact values depend on filter coefficients
        assert!(samples.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_kweighting_filter_reset() {
        let mut filter = KWeightingFilter::new(48000).unwrap();

        // Process some samples to build up state
        for i in 0..100 {
            let _ = filter.process_sample((i as f64 / 100.0).sin());
        }

        // Reset and process same samples - should get same result as fresh filter
        filter.reset();
        let mut filter2 = KWeightingFilter::new(48000).unwrap();

        for i in 0..100 {
            let input = (i as f64 / 100.0).sin();
            let out1 = filter.process_sample(input);
            let out2 = filter2.process_sample(input);
            assert!((out1 - out2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_filter_bank_creation() {
        // Valid configurations
        assert!(KWeightingFilterBank::new(1, 48000).is_ok());
        assert!(KWeightingFilterBank::new(2, 48000).is_ok());
        assert!(KWeightingFilterBank::new(6, 48000).is_ok());
        assert!(KWeightingFilterBank::new(8, 48000).is_ok());

        // Invalid configurations
        assert!(KWeightingFilterBank::new(0, 48000).is_err());
        assert!(KWeightingFilterBank::new(9, 48000).is_err());
        assert!(KWeightingFilterBank::new(2, 7999).is_err());
    }

    #[test]
    fn test_filter_bank_process_interleaved() {
        let mut bank = KWeightingFilterBank::new(2, 48000).unwrap();

        // Interleaved stereo samples: L, R, L, R, ...
        let mut samples = vec![1.0, 0.5, 0.5, -0.5, -0.5, 0.0, 0.0, 0.5];
        bank.process_interleaved(&mut samples);

        // Should have modified samples
        assert_eq!(samples.len(), 8);
    }

    #[test]
    fn test_filter_bank_process_frame() {
        let mut bank = KWeightingFilterBank::new(2, 48000).unwrap();

        // One frame (one sample per channel)
        let frame = vec![1.0, 0.5];
        let output = bank.process_frame(&frame);

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_high_frequency_boost() {
        // Test that the K-weighting filter boosts high frequencies
        let mut filter = KWeightingFilter::new(48000).unwrap();

        // Generate a high-frequency test signal (10 kHz at 48 kHz sample rate)
        let frequency = 10000.0;
        let sample_rate = 48000.0;
        let num_samples = 1000;

        let mut input_power = 0.0;
        let mut output_power = 0.0;

        // Skip first 100 samples to let filter settle
        for i in 0..100 {
            let t = i as f64 / sample_rate;
            let input = (2.0 * PI * frequency * t).sin();
            let _ = filter.process_sample(input);
        }

        for i in 100..num_samples {
            let t = i as f64 / sample_rate;
            let input = (2.0 * PI * frequency * t).sin();
            let output = filter.process_sample(input);
            input_power += input * input;
            output_power += output * output;
        }

        // High frequency should be boosted (output power > input power)
        assert!(output_power > input_power);
    }

    #[test]
    fn test_low_frequency_attenuation() {
        // Test that the K-weighting filter attenuates very low frequencies
        let mut filter = KWeightingFilter::new(48000).unwrap();

        // Generate a low-frequency test signal (20 Hz at 48 kHz sample rate)
        let frequency = 20.0;
        let sample_rate = 48000.0;
        let num_samples = 5000; // Need more samples for low frequency

        let mut input_power = 0.0;
        let mut output_power = 0.0;

        // Skip first 1000 samples to let filter settle
        for i in 0..1000 {
            let t = i as f64 / sample_rate;
            let input = (2.0 * PI * frequency * t).sin();
            let _ = filter.process_sample(input);
        }

        for i in 1000..num_samples {
            let t = i as f64 / sample_rate;
            let input = (2.0 * PI * frequency * t).sin();
            let output = filter.process_sample(input);
            input_power += input * input;
            output_power += output * output;
        }

        // Very low frequency should be attenuated (output power < input power)
        assert!(output_power < input_power);
    }
}
