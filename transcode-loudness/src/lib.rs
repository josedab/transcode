//! # Transcode Loudness
//!
//! EBU R128 loudness normalization and measurement for the Transcode codec library.
//!
//! This crate provides comprehensive loudness processing following international
//! broadcast standards:
//!
//! - **EBU R128** (European Broadcasting Union) - Target: -23 LUFS
//! - **ITU-R BS.1770** (International Telecommunication Union) - Measurement algorithm
//! - **ATSC A/85** (Advanced Television Systems Committee) - Target: -24 LKFS
//!
//! ## Features
//!
//! - **Integrated loudness (I)**: Program loudness measurement in LUFS
//! - **Momentary loudness (M)**: 400ms sliding window measurement
//! - **Short-term loudness (S)**: 3s sliding window measurement
//! - **Loudness range (LRA)**: Dynamic range measurement in LU
//! - **True peak (TP)**: Inter-sample peak detection in dBTP
//! - **K-weighting filter**: ITU-R BS.1770 pre-filter
//! - **True peak limiter**: Prevents inter-sample peaks
//!
//! ## Example
//!
//! ```no_run
//! use transcode_loudness::{LoudnessMeter, TwoPassNormalizer, NormalizationMode};
//!
//! // Create a loudness meter
//! let mut meter = LoudnessMeter::new(48000, 2).unwrap();
//!
//! // Process audio samples
//! let samples: Vec<f64> = vec![0.0; 48000 * 2]; // 1 second stereo
//! meter.process_interleaved_f64(&samples);
//!
//! // Get results
//! let results = meter.results();
//! println!("Integrated loudness: {:.1} LUFS", results.integrated);
//! println!("True peak: {:.1} dBTP", results.true_peak);
//!
//! // Normalize to streaming target (-14 LUFS)
//! let mut normalizer = TwoPassNormalizer::new(48000, 2).unwrap();
//! // First pass: analyze
//! normalizer.analyze_f64(&samples);
//! let analysis = normalizer.finish_analysis().unwrap();
//! println!("Gain to apply: {:.1} dB", analysis.gain_db);
//!
//! // Second pass: normalize
//! let mut output = samples.clone();
//! normalizer.normalize_f64(&mut output).unwrap();
//! ```
//!
//! ## Normalization Targets
//!
//! | Standard | Target | Use Case |
//! |----------|--------|----------|
//! | EBU R128 | -23 LUFS | European broadcast |
//! | ATSC A/85 | -24 LKFS | US broadcast |
//! | Streaming | -14 LUFS | Spotify, YouTube |
//! | Apple Music | -16 LUFS | iTunes, Apple Music |
//! | Podcast | -16 LUFS | Podcast platforms |
//!
//! ## Technical Details
//!
//! ### K-Weighting Filter
//!
//! The K-weighting pre-filter consists of two cascaded biquad stages:
//! 1. High-shelf filter (fc=1681.97 Hz, +4 dB gain)
//! 2. High-pass filter (fc=38.14 Hz)
//!
//! This filter approximates the frequency response of human loudness perception.
//!
//! ### Gating
//!
//! Integrated loudness uses a two-stage gating approach:
//! 1. Absolute gate at -70 LUFS (blocks very quiet passages)
//! 2. Relative gate at -10 LU below ungated loudness
//!
//! ### Loudness Range
//!
//! LRA is calculated as the difference between the 95th and 10th percentiles
//! of short-term loudness measurements (after gating).

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod error;
pub mod filter;
pub mod limiter;
pub mod meter;
pub mod normalizer;

// Re-export main types
pub use error::{LoudnessError, Result};
pub use filter::{BiquadCoeffs, BiquadState, KWeightingFilter, KWeightingFilterBank};
pub use limiter::{
    LimiterConfig, SoftClipper, TruePeakLimiter, DEFAULT_ATTACK_MS, DEFAULT_CEILING_DBTP,
    DEFAULT_RELEASE_MS,
};
pub use meter::{LoudnessMeter, LoudnessResults, MeterConfig, TruePeakMeter, targets};
pub use normalizer::{
    NormalizationAnalysis, NormalizationMode, NormalizerConfig, RealtimeNormalizer,
    TwoPassNormalizer, normalize_sample,
};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimum supported sample rate (Hz).
pub const MIN_SAMPLE_RATE: u32 = 8000;

/// Maximum supported channels.
pub const MAX_CHANNELS: u32 = 8;

/// Convert linear amplitude to decibels.
///
/// # Arguments
///
/// * `linear` - Linear amplitude value
///
/// # Returns
///
/// Value in decibels (dB). Returns -infinity for zero or negative input.
#[inline]
pub fn linear_to_db(linear: f64) -> f64 {
    if linear > 0.0 {
        20.0 * linear.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Convert decibels to linear amplitude.
///
/// # Arguments
///
/// * `db` - Value in decibels
///
/// # Returns
///
/// Linear amplitude value.
#[inline]
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear power to LUFS.
///
/// # Arguments
///
/// * `power` - Mean square power value
///
/// # Returns
///
/// Value in LUFS. Returns -infinity for zero or negative input.
#[inline]
pub fn power_to_lufs(power: f64) -> f64 {
    if power > 0.0 {
        -0.691 + 10.0 * power.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Convert LUFS to linear power.
///
/// # Arguments
///
/// * `lufs` - Value in LUFS
///
/// # Returns
///
/// Mean square power value.
#[inline]
pub fn lufs_to_power(lufs: f64) -> f64 {
    10.0_f64.powf((lufs + 0.691) / 10.0)
}

/// Presets for common normalization scenarios.
pub mod presets {
    use super::*;

    /// Create a normalizer preset for European broadcast (EBU R128).
    ///
    /// - Target: -23 LUFS
    /// - True peak: -1 dBTP
    pub fn broadcast_europe(sample_rate: u32, channels: u32) -> Result<TwoPassNormalizer> {
        let config = NormalizerConfig::new(sample_rate, channels)
            .with_mode(NormalizationMode::EbuR128)
            .with_limiter(true, -1.0);
        TwoPassNormalizer::with_config(config)
    }

    /// Create a normalizer preset for US broadcast (ATSC A/85).
    ///
    /// - Target: -24 LKFS
    /// - True peak: -2 dBTP
    pub fn broadcast_us(sample_rate: u32, channels: u32) -> Result<TwoPassNormalizer> {
        let config = NormalizerConfig::new(sample_rate, channels)
            .with_mode(NormalizationMode::AtscA85)
            .with_limiter(true, -2.0);
        TwoPassNormalizer::with_config(config)
    }

    /// Create a normalizer preset for streaming platforms.
    ///
    /// - Target: -14 LUFS
    /// - True peak: -1 dBTP
    pub fn streaming(sample_rate: u32, channels: u32) -> Result<TwoPassNormalizer> {
        let config = NormalizerConfig::new(sample_rate, channels)
            .with_mode(NormalizationMode::Streaming)
            .with_limiter(true, -1.0);
        TwoPassNormalizer::with_config(config)
    }

    /// Create a normalizer preset for Apple Music / iTunes.
    ///
    /// - Target: -16 LUFS
    /// - True peak: -1 dBTP
    pub fn apple_music(sample_rate: u32, channels: u32) -> Result<TwoPassNormalizer> {
        let config = NormalizerConfig::new(sample_rate, channels)
            .with_mode(NormalizationMode::AppleMusic)
            .with_limiter(true, -1.0);
        TwoPassNormalizer::with_config(config)
    }

    /// Create a normalizer preset for podcasts.
    ///
    /// - Target: -16 LUFS
    /// - True peak: -1 dBTP
    /// - Max gain: 12 dB (to avoid excessive noise amplification)
    pub fn podcast(sample_rate: u32, channels: u32) -> Result<TwoPassNormalizer> {
        let config = NormalizerConfig::new(sample_rate, channels)
            .with_mode(NormalizationMode::Podcast)
            .with_max_gain(12.0)
            .with_limiter(true, -1.0);
        TwoPassNormalizer::with_config(config)
    }

    /// Create a real-time normalizer for live streaming.
    ///
    /// - Target: -14 LUFS
    /// - True peak: -1 dBTP
    pub fn live_streaming(sample_rate: u32, channels: u32) -> Result<RealtimeNormalizer> {
        let config = NormalizerConfig::new(sample_rate, channels)
            .with_mode(NormalizationMode::Streaming)
            .with_limiter(true, -1.0);
        RealtimeNormalizer::with_config(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_linear_to_db() {
        assert!((linear_to_db(1.0) - 0.0).abs() < 0.001);
        assert!((linear_to_db(0.5) - (-6.02)).abs() < 0.1);
        assert!((linear_to_db(2.0) - 6.02).abs() < 0.1);
        assert!(linear_to_db(0.0) == f64::NEG_INFINITY);
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 0.001);
        assert!((db_to_linear(-6.0) - 0.5).abs() < 0.02);
        assert!((db_to_linear(6.0) - 2.0).abs() < 0.02);
    }

    #[test]
    fn test_power_to_lufs() {
        // At power = 1.0, LUFS should be approximately -0.691
        assert!((power_to_lufs(1.0) - (-0.691)).abs() < 0.01);

        // Test zero
        assert!(power_to_lufs(0.0) == f64::NEG_INFINITY);
    }

    #[test]
    fn test_lufs_to_power() {
        // Round-trip test
        let lufs = -23.0;
        let power = lufs_to_power(lufs);
        let back = power_to_lufs(power);
        assert!((back - lufs).abs() < 0.001);
    }

    #[test]
    fn test_conversions_roundtrip() {
        // dB round-trip
        for db in [-60.0, -20.0, -6.0, 0.0, 6.0, 20.0] {
            let linear = db_to_linear(db);
            let back = linear_to_db(linear);
            assert!((back - db).abs() < 0.001, "db round-trip failed for {}", db);
        }

        // LUFS round-trip
        for lufs in [-60.0, -40.0, -23.0, -14.0, 0.0] {
            let power = lufs_to_power(lufs);
            let back = power_to_lufs(power);
            assert!((back - lufs).abs() < 0.001, "LUFS round-trip failed for {}", lufs);
        }
    }

    #[test]
    fn test_preset_creation() {
        assert!(presets::broadcast_europe(48000, 2).is_ok());
        assert!(presets::broadcast_us(48000, 2).is_ok());
        assert!(presets::streaming(48000, 2).is_ok());
        assert!(presets::apple_music(48000, 2).is_ok());
        assert!(presets::podcast(48000, 1).is_ok());
        assert!(presets::live_streaming(48000, 2).is_ok());
    }

    #[test]
    fn test_full_workflow() {
        // Generate test audio
        let sample_rate = 48000;
        let duration_secs = 2;
        let channels = 2;

        let samples: Vec<f64> = (0..sample_rate * duration_secs * channels)
            .map(|i| {
                let t = (i / channels) as f64 / sample_rate as f64;
                0.2 * (2.0 * PI * 440.0 * t).sin()
            })
            .collect();

        // Measure loudness
        let mut meter = LoudnessMeter::new(sample_rate as u32, channels as u32).unwrap();
        meter.process_interleaved_f64(&samples);
        let results = meter.results();

        assert!(results.integrated.is_finite());
        assert!(results.true_peak.is_finite());

        // Normalize
        let mut normalizer = presets::streaming(sample_rate as u32, channels as u32).unwrap();
        normalizer.analyze_f64(&samples);
        let analysis = normalizer.finish_analysis().unwrap();

        assert!(analysis.gain_db.is_finite());

        let mut output = samples.clone();
        normalizer.normalize_f64(&mut output).unwrap();

        // Verify output is different from input
        let differs = samples.iter().zip(output.iter())
            .any(|(&a, &b)| (a - b).abs() > 0.0001);
        assert!(differs || analysis.gain_db.abs() < 0.1);
    }

    #[test]
    fn test_constants() {
        assert_eq!(MIN_SAMPLE_RATE, 8000);
        assert_eq!(MAX_CHANNELS, 8);
    }

    #[test]
    fn test_meter_with_different_formats() {
        let mut meter = LoudnessMeter::new(48000, 1).unwrap();

        // f64
        let f64_samples: Vec<f64> = (0..1000).map(|i| 0.5 * (i as f64 / 100.0).sin()).collect();
        meter.process_interleaved_f64(&f64_samples);

        // f32
        let f32_samples: Vec<f32> = (0..1000).map(|i| 0.5 * (i as f32 / 100.0).sin()).collect();
        meter.process_interleaved_f32(&f32_samples);

        // i16
        let i16_samples: Vec<i16> = (0..1000).map(|i| (16000.0 * (i as f64 / 100.0).sin()) as i16).collect();
        meter.process_interleaved_i16(&i16_samples);

        let results = meter.results();
        assert!(results.integrated.is_finite() || results.integrated == f64::NEG_INFINITY);
    }

    #[test]
    fn test_true_peak_meter() {
        let mut meter = TruePeakMeter::new(2).unwrap();

        // Stereo signal
        let samples: Vec<f64> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.8 } else { 0.6 })
            .collect();

        meter.process_interleaved(&samples);

        assert!(meter.peak_linear() >= 0.79);
        assert!(meter.peak_dbtp() > -3.0);
    }

    #[test]
    fn test_limiter() {
        let mut limiter = TruePeakLimiter::new(48000, 1).unwrap();

        // Signal that exceeds the ceiling
        let mut samples: Vec<f64> = (0..4800)
            .map(|i| 1.2 * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();

        limiter.process_interleaved(&mut samples);

        // Should have limited the signal
        assert!(limiter.max_gain_reduction_db() > 0.0);
    }

    #[test]
    fn test_realtime_normalizer() {
        let mut normalizer = presets::live_streaming(48000, 2).unwrap();

        // Process several seconds of audio
        for _ in 0..10 {
            let mut samples: Vec<f64> = (0..4800)
                .map(|i| 0.3 * (2.0 * PI * 440.0 * i as f64 / 48000.0).sin())
                .collect();

            normalizer.process_f64(&mut samples);
        }

        // Should have stabilized to some gain value
        let gain = normalizer.current_gain_db();
        assert!(gain.is_finite());
    }
}
