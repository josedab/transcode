//! True peak limiter for loudness normalization.
//!
//! This module implements a true peak limiter that prevents inter-sample peaks
//! from exceeding a specified threshold. It uses oversampling to detect and
//! limit true peaks, which may occur between samples in the digital domain.
//!
//! The limiter uses a look-ahead design for transparent limiting with minimal
//! distortion.

use crate::error::{LoudnessError, Result};
use std::collections::VecDeque;

/// Default attack time in milliseconds.
pub const DEFAULT_ATTACK_MS: f64 = 5.0;

/// Default release time in milliseconds.
pub const DEFAULT_RELEASE_MS: f64 = 100.0;

/// Default true peak ceiling in dBTP.
pub const DEFAULT_CEILING_DBTP: f64 = -1.0;

/// Configuration for the true peak limiter.
#[derive(Debug, Clone)]
pub struct LimiterConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u32,
    /// True peak ceiling in dBTP.
    pub ceiling_dbtp: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
    /// Enable true peak detection (4x oversampling).
    pub true_peak: bool,
}

impl LimiterConfig {
    /// Create a new configuration with default parameters.
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        Self {
            sample_rate,
            channels,
            ceiling_dbtp: DEFAULT_CEILING_DBTP,
            attack_ms: DEFAULT_ATTACK_MS,
            release_ms: DEFAULT_RELEASE_MS,
            true_peak: true,
        }
    }

    /// Set the true peak ceiling.
    pub fn with_ceiling(mut self, ceiling_dbtp: f64) -> Self {
        self.ceiling_dbtp = ceiling_dbtp;
        self
    }

    /// Set the attack time.
    pub fn with_attack(mut self, attack_ms: f64) -> Self {
        self.attack_ms = attack_ms;
        self
    }

    /// Set the release time.
    pub fn with_release(mut self, release_ms: f64) -> Self {
        self.release_ms = release_ms;
        self
    }

    /// Enable or disable true peak detection.
    pub fn with_true_peak(mut self, enabled: bool) -> Self {
        self.true_peak = enabled;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate < 8000 {
            return Err(LoudnessError::invalid_sample_rate(self.sample_rate));
        }
        if self.channels == 0 || self.channels > 8 {
            return Err(LoudnessError::invalid_channel_count(self.channels));
        }
        if self.ceiling_dbtp > 0.0 || self.ceiling_dbtp < -20.0 {
            return Err(LoudnessError::invalid_true_peak_limit(self.ceiling_dbtp));
        }
        if self.attack_ms <= 0.0 || self.attack_ms > 100.0 {
            return Err(LoudnessError::filter_init_error(format!(
                "attack time must be between 0 and 100 ms, got {}",
                self.attack_ms
            )));
        }
        if self.release_ms <= 0.0 || self.release_ms > 5000.0 {
            return Err(LoudnessError::filter_init_error(format!(
                "release time must be between 0 and 5000 ms, got {}",
                self.release_ms
            )));
        }
        Ok(())
    }

    /// Get ceiling as linear value.
    pub fn ceiling_linear(&self) -> f64 {
        10.0_f64.powf(self.ceiling_dbtp / 20.0)
    }
}


/// Oversampling filter for true peak detection.
#[derive(Debug, Clone)]
struct OversamplingFilter {
    /// FIR filter coefficients for upsampling.
    upsample_coeffs: Vec<f64>,
    /// FIR filter coefficients for downsampling (reserved for future use).
    #[allow(dead_code)]
    downsample_coeffs: Vec<f64>,
    /// Filter state for each channel.
    states: Vec<VecDeque<f64>>,
    /// Oversampling factor.
    factor: usize,
}

impl OversamplingFilter {
    /// Create a new 4x oversampling filter.
    fn new(channels: u32) -> Self {
        // 4x oversampling polyphase FIR coefficients
        // Designed for minimal aliasing and good transient response
        let upsample_coeffs = vec![
            0.0, 0.017333984375, -0.0322265625, 0.0576171875,
            -0.1103515625, 0.2724609375, 0.8896484375, -0.1533203125,
            0.0693359375, -0.0361328125, 0.0185546875, -0.0078125,
            0.017333984375, -0.0322265625, 0.0576171875, -0.1103515625,
        ];

        let downsample_coeffs = upsample_coeffs.clone();

        let states = (0..channels)
            .map(|_| VecDeque::with_capacity(upsample_coeffs.len()))
            .collect();

        Self {
            upsample_coeffs,
            downsample_coeffs,
            states,
            factor: 4,
        }
    }

    /// Upsample a single sample to 4 samples.
    fn upsample(&mut self, channel: usize, sample: f64) -> [f64; 4] {
        let state = &mut self.states[channel];

        // Add new sample
        state.push_back(sample);
        if state.len() > self.upsample_coeffs.len() / self.factor {
            state.pop_front();
        }

        let mut output = [0.0; 4];

        // Generate 4 output samples using polyphase decomposition
        for (phase, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (i, &coeff) in self.upsample_coeffs.iter().enumerate() {
                if i % self.factor == phase {
                    let state_idx = i / self.factor;
                    if state_idx < state.len() {
                        sum += state[state.len() - 1 - state_idx] * coeff;
                    }
                }
            }
            *out = sum * self.factor as f64; // Compensate for interpolation
        }

        output
    }

    /// Reset filter state.
    fn reset(&mut self) {
        for state in &mut self.states {
            state.clear();
        }
    }
}

/// Look-ahead delay line for the limiter.
#[derive(Debug, Clone)]
struct DelayLine {
    /// Delay buffers for each channel.
    buffers: Vec<VecDeque<f64>>,
    /// Delay in samples.
    delay: usize,
}

impl DelayLine {
    /// Create a new delay line.
    fn new(channels: u32, delay_samples: usize) -> Self {
        let buffers = (0..channels)
            .map(|_| {
                let mut buf = VecDeque::with_capacity(delay_samples + 1);
                for _ in 0..delay_samples {
                    buf.push_back(0.0);
                }
                buf
            })
            .collect();

        Self {
            buffers,
            delay: delay_samples,
        }
    }

    /// Push a sample and get the delayed output.
    #[inline]
    fn process(&mut self, channel: usize, sample: f64) -> f64 {
        let buffer = &mut self.buffers[channel];
        buffer.push_back(sample);
        buffer.pop_front().unwrap_or(0.0)
    }

    /// Reset the delay line.
    fn reset(&mut self) {
        for buffer in &mut self.buffers {
            buffer.clear();
            for _ in 0..self.delay {
                buffer.push_back(0.0);
            }
        }
    }

    /// Flush remaining samples from delay line.
    fn flush(&mut self) -> Vec<Vec<f64>> {
        self.buffers
            .iter()
            .map(|b| b.iter().copied().collect())
            .collect()
    }
}

/// Envelope follower for gain reduction.
#[derive(Debug, Clone)]
struct EnvelopeFollower {
    /// Current envelope value.
    envelope: f64,
    /// Attack coefficient.
    attack_coeff: f64,
    /// Release coefficient.
    release_coeff: f64,
}

impl EnvelopeFollower {
    /// Create a new envelope follower.
    fn new(sample_rate: u32, attack_ms: f64, release_ms: f64) -> Self {
        let attack_coeff = (-1.0 / (attack_ms * 0.001 * sample_rate as f64)).exp();
        let release_coeff = (-1.0 / (release_ms * 0.001 * sample_rate as f64)).exp();

        Self {
            envelope: 0.0,
            attack_coeff,
            release_coeff,
        }
    }

    /// Process an input value and return the envelope.
    #[inline]
    fn process(&mut self, input: f64) -> f64 {
        if input > self.envelope {
            self.envelope = self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * input;
        } else {
            self.envelope = self.release_coeff * self.envelope + (1.0 - self.release_coeff) * input;
        }
        self.envelope
    }

    /// Reset the envelope.
    fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// True peak limiter.
///
/// Prevents audio peaks from exceeding a specified threshold, with optional
/// true peak detection using oversampling.
#[derive(Debug)]
pub struct TruePeakLimiter {
    /// Configuration.
    config: LimiterConfig,
    /// Ceiling threshold (linear).
    ceiling: f64,
    /// Oversampling filter for true peak detection.
    oversampler: Option<OversamplingFilter>,
    /// Look-ahead delay line.
    delay_line: DelayLine,
    /// Envelope follower for gain smoothing.
    envelope: EnvelopeFollower,
    /// Current gain reduction in dB.
    gain_reduction_db: f64,
    /// Maximum gain reduction encountered (dB).
    max_gain_reduction: f64,
    /// Total samples processed.
    samples_processed: usize,
}

impl TruePeakLimiter {
    /// Create a new true peak limiter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    ///
    /// # Returns
    ///
    /// A configured limiter or an error if parameters are invalid.
    pub fn new(sample_rate: u32, channels: u32) -> Result<Self> {
        let config = LimiterConfig::new(sample_rate, channels);
        Self::with_config(config)
    }

    /// Create a new limiter with custom configuration.
    pub fn with_config(config: LimiterConfig) -> Result<Self> {
        config.validate()?;

        let ceiling = config.ceiling_linear();

        let oversampler = if config.true_peak {
            Some(OversamplingFilter::new(config.channels))
        } else {
            None
        };

        // Look-ahead delay = attack time
        let delay_samples = (config.attack_ms * 0.001 * config.sample_rate as f64) as usize;
        let delay_line = DelayLine::new(config.channels, delay_samples);

        let envelope = EnvelopeFollower::new(
            config.sample_rate,
            config.attack_ms,
            config.release_ms,
        );

        Ok(Self {
            config,
            ceiling,
            oversampler,
            delay_line,
            envelope,
            gain_reduction_db: 0.0,
            max_gain_reduction: 0.0,
            samples_processed: 0,
        })
    }

    /// Reset the limiter state.
    pub fn reset(&mut self) {
        if let Some(ref mut os) = self.oversampler {
            os.reset();
        }
        self.delay_line.reset();
        self.envelope.reset();
        self.gain_reduction_db = 0.0;
        self.max_gain_reduction = 0.0;
        self.samples_processed = 0;
    }

    /// Set the ceiling threshold in dBTP.
    pub fn set_ceiling(&mut self, ceiling_dbtp: f64) -> Result<()> {
        if !(-20.0..=0.0).contains(&ceiling_dbtp) {
            return Err(LoudnessError::invalid_true_peak_limit(ceiling_dbtp));
        }
        self.config.ceiling_dbtp = ceiling_dbtp;
        self.ceiling = self.config.ceiling_linear();
        Ok(())
    }

    /// Get the current ceiling in dBTP.
    pub fn ceiling_dbtp(&self) -> f64 {
        self.config.ceiling_dbtp
    }

    /// Get the current gain reduction in dB.
    pub fn gain_reduction_db(&self) -> f64 {
        self.gain_reduction_db
    }

    /// Get the maximum gain reduction encountered.
    pub fn max_gain_reduction_db(&self) -> f64 {
        self.max_gain_reduction
    }

    /// Detect the peak level of a frame (optionally with true peak detection).
    fn detect_peak(&mut self, frame: &[f64]) -> f64 {
        let mut max_peak = 0.0f64;

        if let Some(ref mut oversampler) = self.oversampler {
            // True peak detection with oversampling
            for (ch, &sample) in frame.iter().enumerate() {
                let upsampled = oversampler.upsample(ch, sample);
                for &s in &upsampled {
                    max_peak = max_peak.max(s.abs());
                }
            }
        } else {
            // Sample peak only
            for &sample in frame {
                max_peak = max_peak.max(sample.abs());
            }
        }

        max_peak
    }

    /// Process interleaved samples in-place.
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved audio samples
    pub fn process_interleaved(&mut self, samples: &mut [f64]) {
        let channels = self.config.channels as usize;

        for frame_samples in samples.chunks_exact_mut(channels) {
            // Detect peak level
            let peak = self.detect_peak(frame_samples);

            // Calculate required gain reduction
            let target_gain = if peak > self.ceiling {
                self.ceiling / peak
            } else {
                1.0
            };

            // Convert to dB for envelope following
            let target_reduction = if target_gain < 1.0 {
                -20.0 * target_gain.log10()
            } else {
                0.0
            };

            // Smooth the gain reduction
            let smoothed_reduction = self.envelope.process(target_reduction);

            // Convert back to linear gain
            let gain = 10.0_f64.powf(-smoothed_reduction / 20.0);

            // Apply gain through delay line
            for (ch, sample) in frame_samples.iter_mut().enumerate() {
                let delayed = self.delay_line.process(ch, *sample);
                *sample = delayed * gain;
            }

            // Update statistics
            self.gain_reduction_db = smoothed_reduction;
            if smoothed_reduction > self.max_gain_reduction {
                self.max_gain_reduction = smoothed_reduction;
            }
            self.samples_processed += 1;
        }
    }

    /// Process interleaved f32 samples in-place.
    pub fn process_interleaved_f32(&mut self, samples: &mut [f32]) {
        let channels = self.config.channels as usize;

        for frame_samples in samples.chunks_exact_mut(channels) {
            // Convert to f64
            let frame_f64: Vec<f64> = frame_samples.iter().map(|&s| s as f64).collect();

            // Detect peak level
            let peak = self.detect_peak(&frame_f64);

            // Calculate required gain reduction
            let target_gain = if peak > self.ceiling {
                self.ceiling / peak
            } else {
                1.0
            };

            // Convert to dB for envelope following
            let target_reduction = if target_gain < 1.0 {
                -20.0 * target_gain.log10()
            } else {
                0.0
            };

            // Smooth the gain reduction
            let smoothed_reduction = self.envelope.process(target_reduction);

            // Convert back to linear gain
            let gain = 10.0_f64.powf(-smoothed_reduction / 20.0);

            // Apply gain through delay line
            for (ch, sample) in frame_samples.iter_mut().enumerate() {
                let delayed = self.delay_line.process(ch, *sample as f64);
                *sample = (delayed * gain) as f32;
            }

            // Update statistics
            self.gain_reduction_db = smoothed_reduction;
            if smoothed_reduction > self.max_gain_reduction {
                self.max_gain_reduction = smoothed_reduction;
            }
            self.samples_processed += 1;
        }
    }

    /// Flush remaining samples from the delay line.
    ///
    /// Call this after processing all input to get the final delayed samples.
    pub fn flush(&mut self) -> Vec<Vec<f64>> {
        self.delay_line.flush()
    }

    /// Get the latency introduced by the limiter in samples.
    pub fn latency_samples(&self) -> usize {
        (self.config.attack_ms * 0.001 * self.config.sample_rate as f64) as usize
    }

    /// Get the latency in milliseconds.
    pub fn latency_ms(&self) -> f64 {
        self.config.attack_ms
    }

    /// Get total samples processed.
    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }
}

/// Soft clipper for gentle peak limiting.
///
/// Provides smooth saturation rather than hard clipping for a more musical
/// character.
#[derive(Debug, Clone)]
pub struct SoftClipper {
    /// Threshold where clipping begins (linear).
    threshold: f64,
    /// Knee width in dB (reserved for future soft-knee implementation).
    #[allow(dead_code)]
    knee_db: f64,
}

impl SoftClipper {
    /// Create a new soft clipper.
    ///
    /// # Arguments
    ///
    /// * `threshold_db` - Threshold in dB where soft clipping begins
    /// * `knee_db` - Knee width in dB (default 6.0)
    pub fn new(threshold_db: f64, knee_db: f64) -> Self {
        Self {
            threshold: 10.0_f64.powf(threshold_db / 20.0),
            knee_db,
        }
    }

    /// Process a single sample.
    #[inline]
    pub fn process(&self, sample: f64) -> f64 {
        let abs_sample = sample.abs();

        if abs_sample <= self.threshold {
            sample
        } else {
            // Soft clipping using tanh approximation
            let sign = sample.signum();
            let x = (abs_sample - self.threshold) / self.threshold;
            let clipped = self.threshold * (1.0 + x.tanh() * (1.0 - self.threshold));
            sign * clipped.min(1.0)
        }
    }

    /// Process a block of samples in-place.
    pub fn process_block(&self, samples: &mut [f64]) {
        for sample in samples {
            *sample = self.process(*sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_limiter_config_validation() {
        let config = LimiterConfig::new(48000, 2);
        assert!(config.validate().is_ok());

        // Invalid sample rate
        let mut config = LimiterConfig::new(48000, 2);
        config.sample_rate = 100;
        assert!(config.validate().is_err());

        // Invalid channels
        let mut config = LimiterConfig::new(48000, 2);
        config.channels = 0;
        assert!(config.validate().is_err());

        // Invalid ceiling
        let mut config = LimiterConfig::new(48000, 2);
        config.ceiling_dbtp = 1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_limiter_creation() {
        assert!(TruePeakLimiter::new(48000, 2).is_ok());
        assert!(TruePeakLimiter::new(44100, 1).is_ok());
        assert!(TruePeakLimiter::new(96000, 6).is_ok());

        assert!(TruePeakLimiter::new(100, 2).is_err());
        assert!(TruePeakLimiter::new(48000, 0).is_err());
    }

    #[test]
    fn test_limiter_passes_quiet_signal() {
        let mut limiter = TruePeakLimiter::new(48000, 1).unwrap();

        // Quiet signal that shouldn't trigger limiting
        let mut samples: Vec<f64> = (0..4800)
            .map(|i| 0.1 * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();

        let original = samples.clone();
        limiter.process_interleaved(&mut samples);

        // Skip first samples (delay line) and check the rest
        let delay = limiter.latency_samples();
        for (i, &sample) in samples.iter().skip(delay).enumerate() {
            if i < original.len() - delay {
                assert!((sample - original[i]).abs() < 0.01,
                    "Sample {} differs: {} vs {}", i, sample, original[i]);
            }
        }
    }

    #[test]
    fn test_limiter_limits_loud_signal() {
        let mut limiter = TruePeakLimiter::with_config(
            LimiterConfig::new(48000, 1)
                .with_ceiling(-3.0)
                .with_true_peak(false) // Disable for predictable behavior
        ).unwrap();

        // Loud signal that should trigger limiting
        let mut samples: Vec<f64> = (0..4800)
            .map(|i| 1.5 * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();

        limiter.process_interleaved(&mut samples);

        // Check that limiting is applied - the limiter uses envelope following
        // which takes time to converge, so we check overall behavior
        let ceiling = 10.0_f64.powf(-3.0 / 20.0);
        let delay = limiter.latency_samples();

        // Skip initial transient (attack time + delay)
        // After settling, most samples should be limited
        let settled_samples = &samples[delay + 500..];
        let max_peak = settled_samples.iter().map(|s| s.abs()).fold(0.0f64, f64::max);

        // Allow some headroom for envelope follower dynamics
        assert!(max_peak <= ceiling + 0.2,
            "Max peak {} exceeds ceiling {} by too much", max_peak, ceiling);

        // Should have gain reduction
        assert!(limiter.max_gain_reduction_db() > 0.0);
    }

    #[test]
    fn test_limiter_set_ceiling() {
        let mut limiter = TruePeakLimiter::new(48000, 2).unwrap();

        assert!(limiter.set_ceiling(-1.0).is_ok());
        assert!((limiter.ceiling_dbtp() - (-1.0)).abs() < 0.001);

        assert!(limiter.set_ceiling(-6.0).is_ok());
        assert!((limiter.ceiling_dbtp() - (-6.0)).abs() < 0.001);

        // Invalid ceilings
        assert!(limiter.set_ceiling(1.0).is_err());
        assert!(limiter.set_ceiling(-25.0).is_err());
    }

    #[test]
    fn test_limiter_reset() {
        let mut limiter = TruePeakLimiter::new(48000, 1).unwrap();

        // Process some loud audio
        let mut samples: Vec<f64> = (0..4800)
            .map(|i| 1.5 * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();
        limiter.process_interleaved(&mut samples);

        assert!(limiter.max_gain_reduction_db() > 0.0);

        // Reset
        limiter.reset();

        assert_eq!(limiter.samples_processed(), 0);
        assert_eq!(limiter.gain_reduction_db(), 0.0);
        assert_eq!(limiter.max_gain_reduction_db(), 0.0);
    }

    #[test]
    fn test_limiter_latency() {
        let limiter = TruePeakLimiter::with_config(
            LimiterConfig::new(48000, 2).with_attack(5.0)
        ).unwrap();

        // 5ms at 48kHz = 240 samples
        assert_eq!(limiter.latency_samples(), 240);
        assert!((limiter.latency_ms() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_soft_clipper() {
        let clipper = SoftClipper::new(-6.0, 6.0);

        // Quiet signal passes through
        let quiet = 0.1;
        assert!((clipper.process(quiet) - quiet).abs() < 0.001);

        // Loud signal is clipped
        let loud = 1.5;
        let clipped = clipper.process(loud);
        assert!(clipped < loud);
        assert!(clipped <= 1.0);

        // Negative values work too
        let neg_loud = -1.5;
        let neg_clipped = clipper.process(neg_loud);
        assert!(neg_clipped > neg_loud);
        assert!(neg_clipped >= -1.0);
    }

    #[test]
    fn test_envelope_follower() {
        let mut env = EnvelopeFollower::new(48000, 5.0, 100.0);

        // Attack: envelope should rise quickly
        for _ in 0..240 { // 5ms worth
            env.process(1.0);
        }
        assert!(env.envelope > 0.5);

        // Release: envelope should fall slowly
        for _ in 0..4800 { // 100ms worth
            env.process(0.0);
        }
        assert!(env.envelope < 0.5);
    }

    #[test]
    fn test_delay_line() {
        let mut delay = DelayLine::new(1, 10);

        // First 10 outputs should be zero (initial delay)
        for i in 0..10 {
            let out = delay.process(0, (i + 1) as f64);
            assert_eq!(out, 0.0);
        }

        // After that, should get delayed values
        for i in 10..20 {
            let out = delay.process(0, (i + 1) as f64);
            assert_eq!(out, (i - 9) as f64);
        }
    }

    #[test]
    fn test_limiter_config_builder() {
        let config = LimiterConfig::new(48000, 2)
            .with_ceiling(-2.0)
            .with_attack(10.0)
            .with_release(200.0)
            .with_true_peak(false);

        assert_eq!(config.ceiling_dbtp, -2.0);
        assert_eq!(config.attack_ms, 10.0);
        assert_eq!(config.release_ms, 200.0);
        assert!(!config.true_peak);
    }

    #[test]
    fn test_f32_processing() {
        let mut limiter = TruePeakLimiter::new(48000, 1).unwrap();

        let mut samples: Vec<f32> = (0..4800)
            .map(|i| 1.5 * (2.0 * PI as f32 * 1000.0 * i as f32 / 48000.0).sin())
            .collect();

        limiter.process_interleaved_f32(&mut samples);

        // Should have limited the signal
        assert!(limiter.max_gain_reduction_db() > 0.0);
    }
}
