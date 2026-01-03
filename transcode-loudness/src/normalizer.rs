//! Loudness normalizer for EBU R128 compliance.
//!
//! This module provides loudness normalization to meet various broadcast
//! and streaming standards:
//! - EBU R128: -23 LUFS (European broadcast)
//! - ATSC A/85: -24 LKFS (US broadcast)
//! - Streaming: -14 LUFS (Spotify, YouTube)
//! - Apple Music: -16 LUFS
//!
//! The normalizer operates in two modes:
//! 1. Two-pass: First pass measures loudness, second pass applies gain
//! 2. Real-time: Adjusts gain based on running loudness measurement

use crate::error::{LoudnessError, Result};
use crate::limiter::{LimiterConfig, TruePeakLimiter};
use crate::meter::{LoudnessMeter, LoudnessResults, targets};
use transcode_core::{Sample, SampleFormat};

/// Normalization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalizationMode {
    /// EBU R128 broadcast (-23 LUFS).
    #[default]
    EbuR128,
    /// ATSC A/85 US broadcast (-24 LKFS).
    AtscA85,
    /// Streaming platforms (-14 LUFS).
    Streaming,
    /// Apple Music / iTunes (-16 LUFS).
    AppleMusic,
    /// Podcast (-16 LUFS).
    Podcast,
    /// Custom target loudness.
    Custom(i32), // LUFS * 10 to avoid floating point in enum
}

impl NormalizationMode {
    /// Get the target loudness in LUFS.
    pub fn target_lufs(&self) -> f64 {
        match self {
            Self::EbuR128 => targets::EBU_R128,
            Self::AtscA85 => targets::ATSC_A85,
            Self::Streaming => targets::STREAMING,
            Self::AppleMusic => targets::APPLE_MUSIC,
            Self::Podcast => targets::PODCAST,
            Self::Custom(lufs_x10) => *lufs_x10 as f64 / 10.0,
        }
    }

    /// Create a custom normalization mode.
    pub fn custom(lufs: f64) -> Self {
        Self::Custom((lufs * 10.0).round() as i32)
    }
}

/// Configuration for the loudness normalizer.
#[derive(Debug, Clone)]
pub struct NormalizerConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u32,
    /// Normalization mode (target loudness).
    pub mode: NormalizationMode,
    /// Maximum gain increase in dB.
    pub max_gain_db: f64,
    /// Enable true peak limiting.
    pub limit_true_peak: bool,
    /// True peak ceiling in dBTP.
    pub true_peak_ceiling: f64,
    /// Enable loudness range compression.
    pub compress_lra: bool,
    /// Target loudness range (LU).
    pub target_lra: f64,
}

impl NormalizerConfig {
    /// Create a new configuration with default EBU R128 parameters.
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        Self {
            sample_rate,
            channels,
            mode: NormalizationMode::EbuR128,
            max_gain_db: 20.0, // Maximum 20 dB boost
            limit_true_peak: true,
            true_peak_ceiling: -1.0, // -1 dBTP per EBU R128
            compress_lra: false,
            target_lra: 20.0, // 20 LU max range
        }
    }

    /// Set the normalization mode.
    pub fn with_mode(mut self, mode: NormalizationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the maximum gain increase.
    pub fn with_max_gain(mut self, max_gain_db: f64) -> Self {
        self.max_gain_db = max_gain_db;
        self
    }

    /// Enable or disable true peak limiting.
    pub fn with_limiter(mut self, enable: bool, ceiling_dbtp: f64) -> Self {
        self.limit_true_peak = enable;
        self.true_peak_ceiling = ceiling_dbtp;
        self
    }

    /// Enable loudness range compression.
    pub fn with_lra_compression(mut self, enable: bool, target_lra: f64) -> Self {
        self.compress_lra = enable;
        self.target_lra = target_lra;
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
        let target = self.mode.target_lufs();
        if !(-70.0..=0.0).contains(&target) {
            return Err(LoudnessError::invalid_target_loudness(target));
        }
        if !(-20.0..=0.0).contains(&self.true_peak_ceiling) {
            return Err(LoudnessError::invalid_true_peak_limit(self.true_peak_ceiling));
        }
        Ok(())
    }
}

/// Result of normalization analysis.
#[derive(Debug, Clone)]
pub struct NormalizationAnalysis {
    /// Original loudness measurements.
    pub original: LoudnessResults,
    /// Target loudness.
    pub target_lufs: f64,
    /// Calculated gain to apply (dB).
    pub gain_db: f64,
    /// Calculated gain to apply (linear).
    pub gain_linear: f64,
    /// Whether gain was clipped to max.
    pub gain_clipped: bool,
    /// Predicted output loudness.
    pub predicted_output_lufs: f64,
    /// Predicted output true peak.
    pub predicted_output_peak_dbtp: f64,
}

impl NormalizationAnalysis {
    /// Check if normalization is needed.
    pub fn needs_normalization(&self) -> bool {
        self.gain_db.abs() > 0.1
    }
}

/// Two-pass loudness normalizer.
///
/// First pass: analyze audio to measure loudness
/// Second pass: apply calculated gain with optional limiting
#[derive(Debug)]
pub struct TwoPassNormalizer {
    /// Configuration.
    config: NormalizerConfig,
    /// Loudness meter for analysis.
    meter: LoudnessMeter,
    /// True peak limiter (optional).
    limiter: Option<TruePeakLimiter>,
    /// Current analysis result.
    analysis: Option<NormalizationAnalysis>,
    /// Whether currently in analysis phase.
    analyzing: bool,
}

impl TwoPassNormalizer {
    /// Create a new two-pass normalizer.
    pub fn new(sample_rate: u32, channels: u32) -> Result<Self> {
        let config = NormalizerConfig::new(sample_rate, channels);
        Self::with_config(config)
    }

    /// Create a new normalizer with custom configuration.
    pub fn with_config(config: NormalizerConfig) -> Result<Self> {
        config.validate()?;

        let meter = LoudnessMeter::new(config.sample_rate, config.channels)?;

        let limiter = if config.limit_true_peak {
            let limiter_config = LimiterConfig::new(config.sample_rate, config.channels)
                .with_ceiling(config.true_peak_ceiling);
            Some(TruePeakLimiter::with_config(limiter_config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            meter,
            limiter,
            analysis: None,
            analyzing: true,
        })
    }

    /// Reset the normalizer to initial state.
    pub fn reset(&mut self) {
        self.meter.reset();
        if let Some(ref mut limiter) = self.limiter {
            limiter.reset();
        }
        self.analysis = None;
        self.analyzing = true;
    }

    /// Start the analysis phase.
    pub fn begin_analysis(&mut self) {
        self.meter.reset();
        self.analysis = None;
        self.analyzing = true;
    }

    /// Feed audio for analysis (first pass).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved f64 samples
    pub fn analyze_f64(&mut self, samples: &[f64]) {
        if self.analyzing {
            self.meter.process_interleaved_f64(samples);
        }
    }

    /// Feed audio for analysis (first pass).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved f32 samples
    pub fn analyze_f32(&mut self, samples: &[f32]) {
        if self.analyzing {
            self.meter.process_interleaved_f32(samples);
        }
    }

    /// Feed audio for analysis (first pass).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved i16 samples
    pub fn analyze_i16(&mut self, samples: &[i16]) {
        if self.analyzing {
            self.meter.process_interleaved_i16(samples);
        }
    }

    /// Complete analysis and prepare for normalization.
    ///
    /// # Returns
    ///
    /// Analysis results including calculated gain.
    pub fn finish_analysis(&mut self) -> Result<NormalizationAnalysis> {
        if !self.analyzing {
            if let Some(ref analysis) = self.analysis {
                return Ok(analysis.clone());
            }
        }

        let original = self.meter.results();

        if !original.is_valid() {
            return Err(LoudnessError::NoDataProcessed);
        }

        let target = self.config.mode.target_lufs();
        let mut gain_db = target - original.integrated;
        let mut gain_clipped = false;

        // Clip gain to max
        if gain_db > self.config.max_gain_db {
            gain_db = self.config.max_gain_db;
            gain_clipped = true;
        }

        // Check true peak headroom
        if self.config.limit_true_peak {
            let predicted_peak = original.true_peak + gain_db;
            if predicted_peak > self.config.true_peak_ceiling {
                // Reduce gain to stay within peak limit
                let max_gain = self.config.true_peak_ceiling - original.true_peak;
                if max_gain < gain_db {
                    gain_db = max_gain;
                }
            }
        }

        let gain_linear = 10.0_f64.powf(gain_db / 20.0);

        let analysis = NormalizationAnalysis {
            original,
            target_lufs: target,
            gain_db,
            gain_linear,
            gain_clipped,
            predicted_output_lufs: original.integrated + gain_db,
            predicted_output_peak_dbtp: original.true_peak + gain_db,
        };

        self.analysis = Some(analysis.clone());
        self.analyzing = false;

        Ok(analysis)
    }

    /// Get the current analysis (if completed).
    pub fn analysis(&self) -> Option<&NormalizationAnalysis> {
        self.analysis.as_ref()
    }

    /// Apply normalization to audio (second pass).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved f64 samples to normalize in-place
    ///
    /// # Returns
    ///
    /// Error if analysis hasn't been completed.
    pub fn normalize_f64(&mut self, samples: &mut [f64]) -> Result<()> {
        let gain = self
            .analysis
            .as_ref()
            .ok_or(LoudnessError::NoDataProcessed)?
            .gain_linear;

        // Apply gain
        for sample in samples.iter_mut() {
            *sample *= gain;
        }

        // Apply limiting if enabled
        if let Some(ref mut limiter) = self.limiter {
            limiter.process_interleaved(samples);
        }

        Ok(())
    }

    /// Apply normalization to audio (second pass).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved f32 samples to normalize in-place
    pub fn normalize_f32(&mut self, samples: &mut [f32]) -> Result<()> {
        let gain = self
            .analysis
            .as_ref()
            .ok_or(LoudnessError::NoDataProcessed)?
            .gain_linear as f32;

        // Apply gain
        for sample in samples.iter_mut() {
            *sample *= gain;
        }

        // Apply limiting if enabled
        if let Some(ref mut limiter) = self.limiter {
            limiter.process_interleaved_f32(samples);
        }

        Ok(())
    }

    /// Get target loudness.
    pub fn target_lufs(&self) -> f64 {
        self.config.mode.target_lufs()
    }

    /// Get the true peak ceiling.
    pub fn true_peak_ceiling(&self) -> f64 {
        self.config.true_peak_ceiling
    }
}

/// Real-time loudness normalizer.
///
/// Adjusts gain in real-time based on running loudness measurement.
/// Uses a slow-responding envelope to avoid pumping artifacts.
#[derive(Debug)]
pub struct RealtimeNormalizer {
    /// Configuration.
    config: NormalizerConfig,
    /// Loudness meter.
    meter: LoudnessMeter,
    /// True peak limiter.
    limiter: Option<TruePeakLimiter>,
    /// Current gain (linear).
    current_gain: f64,
    /// Target gain (linear).
    target_gain: f64,
    /// Gain smoothing coefficient.
    smoothing_coeff: f64,
    /// Total samples processed.
    samples_processed: usize,
    /// Minimum samples before adjusting gain.
    min_samples: usize,
}

impl RealtimeNormalizer {
    /// Create a new real-time normalizer.
    pub fn new(sample_rate: u32, channels: u32) -> Result<Self> {
        let config = NormalizerConfig::new(sample_rate, channels);
        Self::with_config(config)
    }

    /// Create with custom configuration.
    pub fn with_config(config: NormalizerConfig) -> Result<Self> {
        config.validate()?;

        let meter = LoudnessMeter::new(config.sample_rate, config.channels)?;

        let limiter = if config.limit_true_peak {
            let limiter_config = LimiterConfig::new(config.sample_rate, config.channels)
                .with_ceiling(config.true_peak_ceiling);
            Some(TruePeakLimiter::with_config(limiter_config)?)
        } else {
            None
        };

        // Smoothing time constant: ~3 seconds for stable normalization
        let smoothing_time_ms = 3000.0;
        let smoothing_coeff = (-1.0 / (smoothing_time_ms * 0.001 * config.sample_rate as f64)).exp();

        // Wait at least 3 seconds before adjusting gain (need short-term measurement)
        let min_samples = (config.sample_rate as f64 * 3.0) as usize;

        Ok(Self {
            config,
            meter,
            limiter,
            current_gain: 1.0,
            target_gain: 1.0,
            smoothing_coeff,
            samples_processed: 0,
            min_samples,
        })
    }

    /// Reset the normalizer.
    pub fn reset(&mut self) {
        self.meter.reset();
        if let Some(ref mut limiter) = self.limiter {
            limiter.reset();
        }
        self.current_gain = 1.0;
        self.target_gain = 1.0;
        self.samples_processed = 0;
    }

    /// Process audio in real-time (measure and normalize).
    ///
    /// # Arguments
    ///
    /// * `samples` - Interleaved f64 samples to process in-place
    pub fn process_f64(&mut self, samples: &mut [f64]) {
        let channels = self.config.channels as usize;

        // Update loudness measurement
        self.meter.process_interleaved_f64(samples);

        // Update gain target based on current loudness
        self.samples_processed += samples.len() / channels;
        self.update_target_gain();

        // Apply smoothed gain
        for sample in samples.iter_mut() {
            // Smooth gain transition
            self.current_gain = self.smoothing_coeff * self.current_gain
                + (1.0 - self.smoothing_coeff) * self.target_gain;
            *sample *= self.current_gain;
        }

        // Apply limiting
        if let Some(ref mut limiter) = self.limiter {
            limiter.process_interleaved(samples);
        }
    }

    /// Process audio in real-time.
    pub fn process_f32(&mut self, samples: &mut [f32]) {
        let channels = self.config.channels as usize;

        // Convert to f64 for measurement
        let samples_f64: Vec<f64> = samples.iter().map(|&s| s as f64).collect();
        self.meter.process_interleaved_f64(&samples_f64);

        // Update gain target
        self.samples_processed += samples.len() / channels;
        self.update_target_gain();

        // Apply smoothed gain
        for sample in samples.iter_mut() {
            self.current_gain = self.smoothing_coeff * self.current_gain
                + (1.0 - self.smoothing_coeff) * self.target_gain;
            *sample *= self.current_gain as f32;
        }

        // Apply limiting
        if let Some(ref mut limiter) = self.limiter {
            limiter.process_interleaved_f32(samples);
        }
    }

    /// Update the target gain based on current loudness.
    fn update_target_gain(&mut self) {
        if self.samples_processed < self.min_samples {
            return;
        }

        // Use short-term loudness for real-time adjustment
        let current_loudness = self.meter.shortterm_loudness();

        if current_loudness.is_finite() && current_loudness > -70.0 {
            let target = self.config.mode.target_lufs();
            let gain_db = target - current_loudness;

            // Clip to maximum gain
            let gain_db = gain_db.min(self.config.max_gain_db);

            self.target_gain = 10.0_f64.powf(gain_db / 20.0);
        }
    }

    /// Get the current gain in dB.
    pub fn current_gain_db(&self) -> f64 {
        20.0 * self.current_gain.log10()
    }

    /// Get the target gain in dB.
    pub fn target_gain_db(&self) -> f64 {
        20.0 * self.target_gain.log10()
    }

    /// Get the current loudness measurement.
    pub fn current_loudness(&self) -> LoudnessResults {
        self.meter.results()
    }
}

/// Normalize a Sample buffer from transcode_core.
///
/// This is a convenience function for integrating with the transcode pipeline.
pub fn normalize_sample(
    sample: &mut Sample,
    target_lufs: f64,
    true_peak_ceiling: f64,
) -> Result<NormalizationAnalysis> {
    let sample_rate = sample.sample_rate();
    let channels = sample.channels();

    // Create normalizer
    let config = NormalizerConfig::new(sample_rate, channels)
        .with_mode(NormalizationMode::custom(target_lufs))
        .with_limiter(true, true_peak_ceiling);

    let mut normalizer = TwoPassNormalizer::with_config(config)?;

    // Analyze
    let buffer = sample.buffer();
    match buffer.format {
        SampleFormat::F32 | SampleFormat::F32p => {
            if let Some(data) = buffer.as_f32() {
                normalizer.analyze_f32(data);
            } else {
                return Err(LoudnessError::unsupported_format(buffer.format.to_string()));
            }
        }
        SampleFormat::S16 | SampleFormat::S16p => {
            if let Some(data) = buffer.as_s16() {
                normalizer.analyze_i16(data);
            } else {
                return Err(LoudnessError::unsupported_format(buffer.format.to_string()));
            }
        }
        _ => {
            return Err(LoudnessError::unsupported_format(buffer.format.to_string()));
        }
    }

    let analysis = normalizer.finish_analysis()?;

    // Apply normalization
    let buffer_mut = sample.buffer_mut();
    match buffer_mut.format {
        SampleFormat::F32 | SampleFormat::F32p => {
            // Need to get raw bytes and reinterpret as f32
            let data = buffer_mut.data_mut();
            let samples: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_mut_ptr() as *mut f32,
                    data.len() / 4,
                )
            };
            normalizer.normalize_f32(samples)?;
        }
        SampleFormat::S16 | SampleFormat::S16p => {
            // Convert to f32, normalize, convert back
            let data = buffer_mut.data_mut();
            let samples: &mut [i16] = unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_mut_ptr() as *mut i16,
                    data.len() / 2,
                )
            };

            // Convert to f32
            let mut f32_samples: Vec<f32> = samples.iter()
                .map(|&s| s as f32 / 32768.0)
                .collect();

            normalizer.normalize_f32(&mut f32_samples)?;

            // Convert back to i16
            for (i, &s) in f32_samples.iter().enumerate() {
                samples[i] = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            }
        }
        _ => {
            return Err(LoudnessError::unsupported_format(buffer_mut.format.to_string()));
        }
    }

    Ok(analysis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_normalization_mode() {
        assert_eq!(NormalizationMode::EbuR128.target_lufs(), -23.0);
        assert_eq!(NormalizationMode::AtscA85.target_lufs(), -24.0);
        assert_eq!(NormalizationMode::Streaming.target_lufs(), -14.0);
        assert_eq!(NormalizationMode::AppleMusic.target_lufs(), -16.0);

        let custom = NormalizationMode::custom(-20.5);
        assert!((custom.target_lufs() - (-20.5)).abs() < 0.1);
    }

    #[test]
    fn test_config_validation() {
        let config = NormalizerConfig::new(48000, 2);
        assert!(config.validate().is_ok());

        // Invalid sample rate
        let mut config = NormalizerConfig::new(48000, 2);
        config.sample_rate = 100;
        assert!(config.validate().is_err());

        // Invalid channels
        let mut config = NormalizerConfig::new(48000, 2);
        config.channels = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_two_pass_normalizer_creation() {
        assert!(TwoPassNormalizer::new(48000, 2).is_ok());
        assert!(TwoPassNormalizer::new(44100, 1).is_ok());

        assert!(TwoPassNormalizer::new(100, 2).is_err());
    }

    #[test]
    fn test_two_pass_normalization() {
        let mut normalizer = TwoPassNormalizer::with_config(
            NormalizerConfig::new(48000, 1)
                .with_mode(NormalizationMode::custom(-20.0))
                .with_limiter(false, -1.0)
        ).unwrap();

        // Generate test signal at approximately -30 LUFS
        let amplitude = 0.01; // Low amplitude = quiet signal
        let samples: Vec<f64> = (0..48000)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();

        // First pass: analyze
        normalizer.analyze_f64(&samples);
        let analysis = normalizer.finish_analysis().unwrap();

        // Should need positive gain to reach -20 LUFS
        assert!(analysis.gain_db > 0.0);
        assert!(analysis.needs_normalization());

        // Second pass: normalize
        let mut output = samples.clone();
        normalizer.normalize_f64(&mut output).unwrap();

        // Output should be louder than input
        let input_rms: f64 = (samples.iter().map(|x| x * x).sum::<f64>() / samples.len() as f64).sqrt();
        let output_rms: f64 = (output.iter().map(|x| x * x).sum::<f64>() / output.len() as f64).sqrt();
        assert!(output_rms > input_rms);
    }

    #[test]
    fn test_analysis_before_normalization() {
        let mut normalizer = TwoPassNormalizer::new(48000, 2).unwrap();

        // Try to normalize without analysis
        let mut samples = vec![0.0; 1000];
        let result = normalizer.normalize_f64(&mut samples);

        // Should fail
        assert!(result.is_err());
    }

    #[test]
    fn test_gain_limiting() {
        let mut normalizer = TwoPassNormalizer::with_config(
            NormalizerConfig::new(48000, 1)
                .with_mode(NormalizationMode::EbuR128)
                .with_max_gain(10.0) // Limit to 10 dB
                .with_limiter(false, -1.0)
        ).unwrap();

        // Very quiet signal that would need >10 dB gain
        let amplitude = 0.001;
        let samples: Vec<f64> = (0..48000)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();

        normalizer.analyze_f64(&samples);
        let analysis = normalizer.finish_analysis().unwrap();

        // Gain should be clipped to max
        assert!(analysis.gain_db <= 10.0);
        assert!(analysis.gain_clipped);
    }

    #[test]
    fn test_realtime_normalizer() {
        let mut normalizer = RealtimeNormalizer::with_config(
            NormalizerConfig::new(48000, 1)
                .with_mode(NormalizationMode::Streaming)
                .with_limiter(false, -1.0)
        ).unwrap();

        // Process some audio
        let mut samples: Vec<f64> = (0..48000 * 5) // 5 seconds
            .map(|i| 0.1 * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();

        normalizer.process_f64(&mut samples);

        // Check that gain is being applied
        let gain_db = normalizer.current_gain_db();
        assert!(gain_db.is_finite());
    }

    #[test]
    fn test_realtime_normalizer_reset() {
        let mut normalizer = RealtimeNormalizer::new(48000, 1).unwrap();

        // Process some audio
        let mut samples: Vec<f64> = (0..48000)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();
        normalizer.process_f64(&mut samples);

        // Reset
        normalizer.reset();

        // Gain should be back to unity
        assert!((normalizer.current_gain_db()).abs() < 0.1);
    }

    #[test]
    fn test_config_builder() {
        let config = NormalizerConfig::new(48000, 2)
            .with_mode(NormalizationMode::Streaming)
            .with_max_gain(15.0)
            .with_limiter(true, -2.0)
            .with_lra_compression(true, 15.0);

        assert_eq!(config.mode.target_lufs(), -14.0);
        assert_eq!(config.max_gain_db, 15.0);
        assert!(config.limit_true_peak);
        assert_eq!(config.true_peak_ceiling, -2.0);
        assert!(config.compress_lra);
        assert_eq!(config.target_lra, 15.0);
    }

    #[test]
    fn test_f32_processing() {
        let mut normalizer = TwoPassNormalizer::with_config(
            NormalizerConfig::new(48000, 1)
                .with_limiter(false, -1.0)
        ).unwrap();

        let samples_f32: Vec<f32> = (0..48000)
            .map(|i| 0.1 * (2.0 * PI as f32 * 1000.0 * i as f32 / 48000.0).sin())
            .collect();

        normalizer.analyze_f32(&samples_f32);
        let analysis = normalizer.finish_analysis().unwrap();

        assert!(analysis.original.is_valid());

        let mut output = samples_f32.clone();
        normalizer.normalize_f32(&mut output).unwrap();
    }

    #[test]
    fn test_i16_processing() {
        let mut normalizer = TwoPassNormalizer::with_config(
            NormalizerConfig::new(48000, 1)
                .with_limiter(false, -1.0)
        ).unwrap();

        let samples_i16: Vec<i16> = (0..48000)
            .map(|i| (3000.0 * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin()) as i16)
            .collect();

        normalizer.analyze_i16(&samples_i16);
        let analysis = normalizer.finish_analysis().unwrap();

        assert!(analysis.original.is_valid());
    }

    #[test]
    fn test_normalization_analysis_helpers() {
        let analysis = NormalizationAnalysis {
            original: LoudnessResults {
                integrated: -25.0,
                momentary: -24.0,
                shortterm: -25.0,
                range: 10.0,
                true_peak: -3.0,
                max_momentary: -22.0,
                max_shortterm: -23.0,
            },
            target_lufs: -23.0,
            gain_db: 2.0,
            gain_linear: 1.26,
            gain_clipped: false,
            predicted_output_lufs: -23.0,
            predicted_output_peak_dbtp: -1.0,
        };

        assert!(analysis.needs_normalization());

        // Small gain shouldn't need normalization
        let analysis2 = NormalizationAnalysis {
            gain_db: 0.05,
            ..analysis
        };
        assert!(!analysis2.needs_normalization());
    }
}
