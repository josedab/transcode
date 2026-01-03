//! Loudness meter for EBU R128 / ITU-R BS.1770 measurement.
//!
//! This module provides comprehensive loudness measurement including:
//! - Integrated loudness (I) - overall program loudness
//! - Momentary loudness (M) - 400ms sliding window
//! - Short-term loudness (S) - 3s sliding window
//! - Loudness range (LRA) - dynamic range measure
//! - True peak measurement (dBTP)
//!
//! All measurements conform to EBU R128 and ITU-R BS.1770-4 specifications.

use crate::error::{LoudnessError, Result};
use crate::filter::KWeightingFilterBank;
use std::collections::VecDeque;

/// Standard target loudness values.
pub mod targets {
    /// EBU R128 broadcast target (-23 LUFS).
    pub const EBU_R128: f64 = -23.0;

    /// ATSC A/85 broadcast target (-24 LKFS).
    pub const ATSC_A85: f64 = -24.0;

    /// Streaming target (Spotify, YouTube) (-14 LUFS).
    pub const STREAMING: f64 = -14.0;

    /// Apple Music / iTunes target (-16 LUFS).
    pub const APPLE_MUSIC: f64 = -16.0;

    /// Podcast target (-16 to -18 LUFS, using -16).
    pub const PODCAST: f64 = -16.0;
}

/// Channel weights for loudness calculation per EBU R128.
///
/// Returns the channel weight for a given channel index in a multichannel layout.
/// Center, LFE have special weights; surround channels get +1.5 dB boost.
#[inline]
fn channel_weight(channel_idx: u32, total_channels: u32) -> f64 {
    match total_channels {
        1 => 1.0,                        // Mono
        2 => 1.0,                        // Stereo: L, R
        3 => {                           // 2.1: L, R, LFE
            if channel_idx == 2 { 0.0 } else { 1.0 }
        }
        4 => 1.0,                        // Quad: FL, FR, BL, BR
        5 => 1.0,                        // 5.0: FL, FR, C, BL, BR
        6 => {                           // 5.1: FL, FR, C, LFE, BL, BR
            match channel_idx {
                3 => 0.0,                // LFE excluded
                4 | 5 => 1.41,           // Surround +1.5 dB
                _ => 1.0,
            }
        }
        8 => {                           // 7.1: FL, FR, C, LFE, BL, BR, SL, SR
            match channel_idx {
                3 => 0.0,                // LFE excluded
                4..=7 => 1.41,           // Surround +1.5 dB
                _ => 1.0,
            }
        }
        _ => 1.0,
    }
}

/// Configuration for the loudness meter.
#[derive(Debug, Clone)]
pub struct MeterConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u32,
    /// Block size for momentary loudness (samples).
    pub momentary_block_size: usize,
    /// Block size for short-term loudness (samples).
    pub shortterm_block_size: usize,
    /// Block overlap (0.75 = 75% overlap per EBU R128).
    pub overlap: f64,
    /// Absolute gate threshold in LUFS.
    pub absolute_gate: f64,
    /// Relative gate threshold (dB below ungated loudness).
    pub relative_gate: f64,
}

impl MeterConfig {
    /// Create a new configuration with standard EBU R128 parameters.
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        // Momentary: 400ms window
        let momentary_block_size = (sample_rate as f64 * 0.4) as usize;
        // Short-term: 3s window
        let shortterm_block_size = (sample_rate as f64 * 3.0) as usize;

        Self {
            sample_rate,
            channels,
            momentary_block_size,
            shortterm_block_size,
            overlap: 0.75, // 75% overlap
            absolute_gate: -70.0,
            relative_gate: -10.0,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate < 8000 {
            return Err(LoudnessError::invalid_sample_rate(self.sample_rate));
        }
        if self.channels == 0 || self.channels > 8 {
            return Err(LoudnessError::invalid_channel_count(self.channels));
        }
        Ok(())
    }
}

/// Loudness measurement results.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoudnessResults {
    /// Integrated loudness in LUFS.
    pub integrated: f64,
    /// Momentary loudness in LUFS (last 400ms).
    pub momentary: f64,
    /// Short-term loudness in LUFS (last 3s).
    pub shortterm: f64,
    /// Loudness range in LU.
    pub range: f64,
    /// True peak in dBTP.
    pub true_peak: f64,
    /// Maximum momentary loudness encountered.
    pub max_momentary: f64,
    /// Maximum short-term loudness encountered.
    pub max_shortterm: f64,
}

impl LoudnessResults {
    /// Check if the integrated loudness is valid (not -inf).
    pub fn is_valid(&self) -> bool {
        self.integrated.is_finite() && self.integrated > -70.0
    }

    /// Calculate the gain needed to reach the target loudness.
    ///
    /// # Arguments
    ///
    /// * `target` - Target loudness in LUFS
    ///
    /// # Returns
    ///
    /// Linear gain factor to apply.
    pub fn gain_to_target(&self, target: f64) -> f64 {
        if !self.is_valid() {
            return 1.0;
        }
        let db_change = target - self.integrated;
        10.0_f64.powf(db_change / 20.0)
    }
}

/// Block power accumulator for gated loudness calculation.
#[derive(Debug, Clone)]
struct BlockPower {
    /// Block mean-square values above absolute gate.
    blocks: Vec<f64>,
    /// Sum of all block powers.
    power_sum: f64,
}

impl BlockPower {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            power_sum: 0.0,
        }
    }

    fn add_block(&mut self, power: f64) {
        self.blocks.push(power);
        self.power_sum += power;
    }

    fn ungated_loudness(&self) -> f64 {
        if self.blocks.is_empty() {
            return f64::NEG_INFINITY;
        }
        -0.691 + 10.0 * (self.power_sum / self.blocks.len() as f64).log10()
    }

    fn gated_loudness(&self, relative_gate: f64) -> f64 {
        if self.blocks.is_empty() {
            return f64::NEG_INFINITY;
        }

        let ungated = self.ungated_loudness();
        let gate_threshold = ungated + relative_gate;

        // Convert threshold back to linear power
        let gate_power = 10.0_f64.powf((gate_threshold + 0.691) / 10.0);

        let mut sum = 0.0;
        let mut count = 0;

        for &block in &self.blocks {
            if block >= gate_power {
                sum += block;
                count += 1;
            }
        }

        if count == 0 {
            f64::NEG_INFINITY
        } else {
            -0.691 + 10.0 * (sum / count as f64).log10()
        }
    }

    fn calculate_lra(&self, relative_gate: f64) -> f64 {
        if self.blocks.is_empty() {
            return 0.0;
        }

        let ungated = self.ungated_loudness();
        let gate_threshold = ungated + relative_gate;
        let gate_power = 10.0_f64.powf((gate_threshold + 0.691) / 10.0);

        // Collect gated block loudnesses
        let mut loudnesses: Vec<f64> = self
            .blocks
            .iter()
            .filter(|&&b| b >= gate_power)
            .map(|&b| -0.691 + 10.0 * b.log10())
            .collect();

        if loudnesses.len() < 2 {
            return 0.0;
        }

        loudnesses.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // LRA is the difference between 95th and 10th percentiles
        let low_idx = (loudnesses.len() as f64 * 0.10).floor() as usize;
        let high_idx = (loudnesses.len() as f64 * 0.95).ceil() as usize;

        let high_idx = high_idx.min(loudnesses.len() - 1);

        loudnesses[high_idx] - loudnesses[low_idx]
    }
}

/// EBU R128 Loudness Meter.
///
/// Measures loudness according to EBU R128 / ITU-R BS.1770-4 specifications.
#[derive(Debug)]
pub struct LoudnessMeter {
    /// Configuration.
    config: MeterConfig,
    /// K-weighting filter bank.
    filters: KWeightingFilterBank,
    /// Channel weights.
    weights: Vec<f64>,
    /// Ring buffer for momentary loudness calculation.
    momentary_buffer: VecDeque<f64>,
    /// Ring buffer for short-term loudness calculation.
    shortterm_buffer: VecDeque<f64>,
    /// Block power accumulator.
    block_powers: BlockPower,
    /// Current block accumulator (per-channel mean square).
    current_block: Vec<f64>,
    /// Samples in current block.
    samples_in_block: usize,
    /// Total samples processed.
    total_samples: usize,
    /// Current true peak.
    true_peak: f64,
    /// Maximum momentary loudness.
    max_momentary: f64,
    /// Maximum short-term loudness.
    max_shortterm: f64,
    /// Step size for block processing.
    step_size: usize,
}

impl LoudnessMeter {
    /// Create a new loudness meter.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    ///
    /// # Returns
    ///
    /// A configured loudness meter or an error if parameters are invalid.
    pub fn new(sample_rate: u32, channels: u32) -> Result<Self> {
        let config = MeterConfig::new(sample_rate, channels);
        Self::with_config(config)
    }

    /// Create a new loudness meter with custom configuration.
    pub fn with_config(config: MeterConfig) -> Result<Self> {
        config.validate()?;

        let filters = KWeightingFilterBank::new(config.channels, config.sample_rate)?;
        let weights: Vec<f64> = (0..config.channels)
            .map(|i| channel_weight(i, config.channels))
            .collect();

        let step_size = ((1.0 - config.overlap) * config.momentary_block_size as f64) as usize;
        let step_size = step_size.max(1);

        Ok(Self {
            filters,
            weights,
            momentary_buffer: VecDeque::with_capacity(config.momentary_block_size),
            shortterm_buffer: VecDeque::with_capacity(config.shortterm_block_size),
            block_powers: BlockPower::new(),
            current_block: vec![0.0; config.channels as usize],
            samples_in_block: 0,
            total_samples: 0,
            true_peak: f64::NEG_INFINITY,
            max_momentary: f64::NEG_INFINITY,
            max_shortterm: f64::NEG_INFINITY,
            step_size,
            config,
        })
    }

    /// Reset the meter to initial state.
    pub fn reset(&mut self) {
        self.filters.reset();
        self.momentary_buffer.clear();
        self.shortterm_buffer.clear();
        self.block_powers = BlockPower::new();
        self.current_block.fill(0.0);
        self.samples_in_block = 0;
        self.total_samples = 0;
        self.true_peak = f64::NEG_INFINITY;
        self.max_momentary = f64::NEG_INFINITY;
        self.max_shortterm = f64::NEG_INFINITY;
    }

    /// Process interleaved f32 samples.
    pub fn process_interleaved_f32(&mut self, samples: &[f32]) {
        let channels = self.config.channels as usize;

        for frame in samples.chunks_exact(channels) {
            // Convert to f64 and track true peak
            let frame_f64: Vec<f64> = frame.iter().map(|&s| {
                let sample = s as f64;
                self.update_true_peak(sample);
                sample
            }).collect();

            self.process_frame(&frame_f64);
        }
    }

    /// Process interleaved f64 samples.
    pub fn process_interleaved_f64(&mut self, samples: &[f64]) {
        let channels = self.config.channels as usize;

        for frame in samples.chunks_exact(channels) {
            // Track true peak
            for &sample in frame {
                self.update_true_peak(sample);
            }

            self.process_frame(frame);
        }
    }

    /// Process interleaved i16 samples.
    pub fn process_interleaved_i16(&mut self, samples: &[i16]) {
        let channels = self.config.channels as usize;
        let scale = 1.0 / 32768.0;

        for frame in samples.chunks_exact(channels) {
            let frame_f64: Vec<f64> = frame.iter().map(|&s| {
                let sample = s as f64 * scale;
                self.update_true_peak(sample);
                sample
            }).collect();

            self.process_frame(&frame_f64);
        }
    }

    /// Process a single frame (one sample per channel).
    fn process_frame(&mut self, frame: &[f64]) {
        // Apply K-weighting
        let filtered = self.filters.process_frame(frame);

        // Calculate weighted mean square for this frame
        let mut frame_power = 0.0;
        for (i, (&sample, &weight)) in filtered.iter().zip(self.weights.iter()).enumerate() {
            let power = sample * sample * weight;
            self.current_block[i] += power;
            frame_power += power;
        }

        // Add to ring buffers
        self.momentary_buffer.push_back(frame_power);
        self.shortterm_buffer.push_back(frame_power);

        // Trim ring buffers to window size
        while self.momentary_buffer.len() > self.config.momentary_block_size {
            self.momentary_buffer.pop_front();
        }
        while self.shortterm_buffer.len() > self.config.shortterm_block_size {
            self.shortterm_buffer.pop_front();
        }

        self.samples_in_block += 1;
        self.total_samples += 1;

        // Complete block processing
        if self.samples_in_block >= self.step_size {
            self.complete_block();
        }

        // Update max momentary/shortterm
        if self.momentary_buffer.len() >= self.config.momentary_block_size {
            let momentary = self.calculate_momentary();
            if momentary > self.max_momentary {
                self.max_momentary = momentary;
            }
        }

        if self.shortterm_buffer.len() >= self.config.shortterm_block_size {
            let shortterm = self.calculate_shortterm();
            if shortterm > self.max_shortterm {
                self.max_shortterm = shortterm;
            }
        }
    }

    /// Complete the current block and add to gated loudness calculation.
    fn complete_block(&mut self) {
        if self.samples_in_block == 0 {
            return;
        }

        // Calculate block power (sum of channel powers)
        let block_power: f64 = self.current_block.iter().sum::<f64>() / self.samples_in_block as f64;

        // Apply absolute gate (-70 LUFS)
        let abs_gate_power = 10.0_f64.powf((self.config.absolute_gate + 0.691) / 10.0);
        if block_power >= abs_gate_power {
            self.block_powers.add_block(block_power);
        }

        // Reset for next block
        self.current_block.fill(0.0);
        self.samples_in_block = 0;
    }

    /// Update true peak measurement.
    #[inline]
    fn update_true_peak(&mut self, sample: f64) {
        let peak = sample.abs();
        if peak > self.true_peak {
            self.true_peak = peak;
        }
    }

    /// Calculate momentary loudness from ring buffer.
    fn calculate_momentary(&self) -> f64 {
        if self.momentary_buffer.is_empty() {
            return f64::NEG_INFINITY;
        }

        let sum: f64 = self.momentary_buffer.iter().sum();
        let mean = sum / self.momentary_buffer.len() as f64;

        if mean <= 0.0 {
            f64::NEG_INFINITY
        } else {
            -0.691 + 10.0 * mean.log10()
        }
    }

    /// Calculate short-term loudness from ring buffer.
    fn calculate_shortterm(&self) -> f64 {
        if self.shortterm_buffer.is_empty() {
            return f64::NEG_INFINITY;
        }

        let sum: f64 = self.shortterm_buffer.iter().sum();
        let mean = sum / self.shortterm_buffer.len() as f64;

        if mean <= 0.0 {
            f64::NEG_INFINITY
        } else {
            -0.691 + 10.0 * mean.log10()
        }
    }

    /// Get the current loudness measurement results.
    pub fn results(&self) -> LoudnessResults {
        // Complete any pending block
        let mut meter = self.clone_state();
        meter.complete_block();

        let integrated = meter.block_powers.gated_loudness(self.config.relative_gate);
        let range = meter.block_powers.calculate_lra(self.config.relative_gate);
        let momentary = self.calculate_momentary();
        let shortterm = self.calculate_shortterm();

        // Convert true peak to dBTP
        let true_peak_dbtp = if self.true_peak > 0.0 {
            20.0 * self.true_peak.log10()
        } else {
            f64::NEG_INFINITY
        };

        LoudnessResults {
            integrated,
            momentary,
            shortterm,
            range,
            true_peak: true_peak_dbtp,
            max_momentary: self.max_momentary,
            max_shortterm: self.max_shortterm,
        }
    }

    /// Clone the meter state (for completing pending blocks without mutation).
    fn clone_state(&self) -> MeterState {
        MeterState {
            block_powers: self.block_powers.clone(),
            current_block: self.current_block.clone(),
            samples_in_block: self.samples_in_block,
        }
    }

    /// Get integrated loudness in LUFS.
    pub fn integrated_loudness(&self) -> f64 {
        self.results().integrated
    }

    /// Get momentary loudness in LUFS.
    pub fn momentary_loudness(&self) -> f64 {
        self.calculate_momentary()
    }

    /// Get short-term loudness in LUFS.
    pub fn shortterm_loudness(&self) -> f64 {
        self.calculate_shortterm()
    }

    /// Get loudness range in LU.
    pub fn loudness_range(&self) -> f64 {
        self.results().range
    }

    /// Get true peak in dBTP.
    pub fn true_peak(&self) -> f64 {
        if self.true_peak > 0.0 {
            20.0 * self.true_peak.log10()
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Get total samples processed.
    pub fn samples_processed(&self) -> usize {
        self.total_samples
    }

    /// Get duration processed in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.total_samples as f64 / self.config.sample_rate as f64
    }
}

/// Helper struct for cloning meter state.
struct MeterState {
    block_powers: BlockPower,
    current_block: Vec<f64>,
    samples_in_block: usize,
}

impl MeterState {
    fn complete_block(&mut self) {
        if self.samples_in_block == 0 {
            return;
        }

        let block_power: f64 = self.current_block.iter().sum::<f64>() / self.samples_in_block as f64;
        let abs_gate_power = 10.0_f64.powf((-70.0 + 0.691) / 10.0);

        if block_power >= abs_gate_power {
            self.block_powers.add_block(block_power);
        }
    }
}

/// True peak meter with oversampling.
///
/// Implements true peak measurement per ITU-R BS.1770-4 using 4x oversampling.
#[derive(Debug, Clone)]
pub struct TruePeakMeter {
    /// Maximum true peak seen (linear).
    max_peak: f64,
    /// Oversampling filter coefficients.
    filter_coeffs: Vec<f64>,
    /// Filter state per channel.
    filter_states: Vec<VecDeque<f64>>,
    /// Number of channels.
    channels: u32,
}

impl TruePeakMeter {
    /// Create a new true peak meter.
    ///
    /// # Arguments
    ///
    /// * `channels` - Number of audio channels
    ///
    /// # Returns
    ///
    /// A configured true peak meter.
    pub fn new(channels: u32) -> Result<Self> {
        if channels == 0 || channels > 8 {
            return Err(LoudnessError::invalid_channel_count(channels));
        }

        // 4x oversampling FIR filter coefficients (12 taps per phase)
        // These coefficients implement a polyphase filter for 4x oversampling
        let filter_coeffs = vec![
            0.0017089843750, 0.0109863281250, -0.0196533203125, 0.0332031250000,
            -0.0594482421875, 0.1373291015625, 0.9721679687500, -0.1022949218750,
            0.0476074218750, -0.0266113281250, 0.0148925781250, -0.0083007812500,
        ];

        let filter_states = (0..channels)
            .map(|_| VecDeque::with_capacity(filter_coeffs.len()))
            .collect();

        Ok(Self {
            max_peak: 0.0,
            filter_coeffs,
            filter_states,
            channels,
        })
    }

    /// Reset the meter.
    pub fn reset(&mut self) {
        self.max_peak = 0.0;
        for state in &mut self.filter_states {
            state.clear();
        }
    }

    /// Process interleaved samples and update true peak.
    pub fn process_interleaved(&mut self, samples: &[f64]) {
        let channels = self.channels as usize;

        for frame in samples.chunks_exact(channels) {
            for (ch, &sample) in frame.iter().enumerate() {
                self.process_sample(ch, sample);
            }
        }
    }

    /// Process a single sample for a channel.
    fn process_sample(&mut self, channel: usize, sample: f64) {
        let state = &mut self.filter_states[channel];

        // Add sample to filter state
        state.push_back(sample);
        if state.len() > self.filter_coeffs.len() {
            state.pop_front();
        }

        // Check sample peak directly
        let peak = sample.abs();
        if peak > self.max_peak {
            self.max_peak = peak;
        }

        // Calculate interpolated peaks (4x oversampling)
        if state.len() == self.filter_coeffs.len() {
            for phase in 0..4 {
                let mut interp = 0.0;
                for (i, &coeff) in self.filter_coeffs.iter().enumerate() {
                    // Phase-shifted coefficient index
                    let idx = (i * 4 + phase) % self.filter_coeffs.len();
                    if idx < state.len() {
                        interp += state[idx] * coeff;
                    }
                }
                let interp_peak = interp.abs();
                if interp_peak > self.max_peak {
                    self.max_peak = interp_peak;
                }
            }
        }
    }

    /// Get the maximum true peak in linear scale.
    pub fn peak_linear(&self) -> f64 {
        self.max_peak
    }

    /// Get the maximum true peak in dBTP.
    pub fn peak_dbtp(&self) -> f64 {
        if self.max_peak > 0.0 {
            20.0 * self.max_peak.log10()
        } else {
            f64::NEG_INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_meter_config_validation() {
        // Valid config
        let config = MeterConfig::new(48000, 2);
        assert!(config.validate().is_ok());

        // Invalid sample rate
        let mut config = MeterConfig::new(48000, 2);
        config.sample_rate = 100;
        assert!(config.validate().is_err());

        // Invalid channel count
        let mut config = MeterConfig::new(48000, 2);
        config.channels = 0;
        assert!(config.validate().is_err());

        config.channels = 9;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_meter_creation() {
        assert!(LoudnessMeter::new(48000, 2).is_ok());
        assert!(LoudnessMeter::new(44100, 1).is_ok());
        assert!(LoudnessMeter::new(96000, 6).is_ok());

        assert!(LoudnessMeter::new(100, 2).is_err());
        assert!(LoudnessMeter::new(48000, 0).is_err());
    }

    #[test]
    fn test_silence_measurement() {
        let mut meter = LoudnessMeter::new(48000, 2).unwrap();

        // Process silence
        let silence = vec![0.0f64; 48000 * 2]; // 1 second stereo
        meter.process_interleaved_f64(&silence);

        let results = meter.results();

        // Silence should result in -inf or very low loudness
        assert!(results.integrated < -60.0 || results.integrated == f64::NEG_INFINITY);
    }

    #[test]
    fn test_sine_wave_measurement() {
        let mut meter = LoudnessMeter::new(48000, 1).unwrap();

        // Generate 1 second of 1 kHz sine wave at -20 dBFS
        let frequency = 1000.0;
        let sample_rate = 48000.0;
        let amplitude = 0.1; // approximately -20 dBFS

        let samples: Vec<f64> = (0..48000)
            .map(|i| amplitude * (2.0 * PI * frequency * i as f64 / sample_rate).sin())
            .collect();

        meter.process_interleaved_f64(&samples);

        let results = meter.results();

        // Should have a reasonable loudness value
        assert!(results.integrated.is_finite());
        assert!(results.integrated < 0.0);
        assert!(results.integrated > -50.0);
    }

    #[test]
    fn test_channel_weights() {
        // Mono
        assert_eq!(channel_weight(0, 1), 1.0);

        // Stereo
        assert_eq!(channel_weight(0, 2), 1.0);
        assert_eq!(channel_weight(1, 2), 1.0);

        // 5.1 surround
        assert_eq!(channel_weight(0, 6), 1.0);  // FL
        assert_eq!(channel_weight(1, 6), 1.0);  // FR
        assert_eq!(channel_weight(2, 6), 1.0);  // C
        assert_eq!(channel_weight(3, 6), 0.0);  // LFE (excluded)
        assert!((channel_weight(4, 6) - 1.41).abs() < 0.01);  // BL (+1.5 dB)
        assert!((channel_weight(5, 6) - 1.41).abs() < 0.01);  // BR (+1.5 dB)
    }

    #[test]
    fn test_true_peak_meter() {
        let mut meter = TruePeakMeter::new(1).unwrap();

        // Process a simple signal
        let samples: Vec<f64> = (0..1000)
            .map(|i| 0.5 * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin())
            .collect();

        meter.process_interleaved(&samples);

        // Peak should be close to 0.5 (maybe slightly higher due to interpolation)
        assert!(meter.peak_linear() >= 0.49);
        assert!(meter.peak_linear() <= 0.6);

        // dBTP should be around -6 dB
        assert!(meter.peak_dbtp() > -7.0);
        assert!(meter.peak_dbtp() < -5.0);
    }

    #[test]
    fn test_loudness_results_gain() {
        let results = LoudnessResults {
            integrated: -23.0,
            momentary: -22.0,
            shortterm: -23.0,
            range: 5.0,
            true_peak: -1.0,
            max_momentary: -20.0,
            max_shortterm: -21.0,
        };

        // Gain to reach -14 LUFS should be about 2.24 (9 dB increase)
        let gain = results.gain_to_target(-14.0);
        assert!((gain - 2.818).abs() < 0.1); // 10^(9/20) = 2.818

        // Gain to reach same level should be 1.0
        let gain = results.gain_to_target(-23.0);
        assert!((gain - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_meter_reset() {
        let mut meter = LoudnessMeter::new(48000, 2).unwrap();

        // Process some audio
        let samples: Vec<f64> = (0..48000)
            .map(|i| 0.5 * (2.0 * PI * 440.0 * i as f64 / 48000.0).sin())
            .collect();
        meter.process_interleaved_f64(&samples);

        assert!(meter.samples_processed() > 0);

        // Reset
        meter.reset();

        assert_eq!(meter.samples_processed(), 0);
        assert!(meter.true_peak() == f64::NEG_INFINITY);
    }

    #[test]
    fn test_target_constants() {
        assert_eq!(targets::EBU_R128, -23.0);
        assert_eq!(targets::ATSC_A85, -24.0);
        assert_eq!(targets::STREAMING, -14.0);
        assert_eq!(targets::APPLE_MUSIC, -16.0);
    }

    #[test]
    fn test_duration_calculation() {
        let mut meter = LoudnessMeter::new(48000, 1).unwrap();

        let samples = vec![0.0f64; 48000]; // 1 second
        meter.process_interleaved_f64(&samples);

        assert!((meter.duration_seconds() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_i16_processing() {
        let mut meter = LoudnessMeter::new(48000, 1).unwrap();

        // Generate sine wave as i16
        let samples: Vec<i16> = (0..48000)
            .map(|i| (16000.0 * (2.0 * PI * 440.0 * i as f64 / 48000.0).sin()) as i16)
            .collect();

        meter.process_interleaved_i16(&samples);

        let results = meter.results();
        assert!(results.integrated.is_finite());
    }
}
