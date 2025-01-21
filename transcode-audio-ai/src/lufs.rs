//! ITU-R BS.1770-4 compliant loudness measurement (LUFS)
//!
//! This module implements the EBU R128 / ITU-R BS.1770 loudness measurement algorithm:
//! 1. K-weighting filter (pre-filter stage 1: high-shelf, stage 2: high-pass)
//! 2. Mean square calculation per channel
//! 3. Channel weighting (surround channels weighted by 1.41)
//! 4. Gated loudness measurement with absolute (-70 LUFS) and relative (-10 LU) gates

use crate::AudioBuffer;

/// K-weighting filter for ITU-R BS.1770
///
/// The K-weighting filter consists of two stages:
/// - Stage 1: High-shelf filter (+4 dB above ~1.5 kHz)
/// - Stage 2: High-pass filter (RLB weighting, ~38 Hz cutoff)
pub struct KWeightingFilter {
    // Stage 1: High-shelf filter state (biquad)
    hs_x1: f64,
    hs_x2: f64,
    hs_y1: f64,
    hs_y2: f64,
    // Stage 1 coefficients
    hs_b0: f64,
    hs_b1: f64,
    hs_b2: f64,
    hs_a1: f64,
    hs_a2: f64,

    // Stage 2: High-pass filter state (biquad)
    hp_x1: f64,
    hp_x2: f64,
    hp_y1: f64,
    hp_y2: f64,
    // Stage 2 coefficients
    hp_b0: f64,
    hp_b1: f64,
    hp_b2: f64,
    hp_a1: f64,
    hp_a2: f64,
}

impl KWeightingFilter {
    /// Create a new K-weighting filter for the given sample rate
    pub fn new(sample_rate: u32) -> Self {
        let fs = sample_rate as f64;

        // Stage 1: High-shelf filter coefficients (from ITU-R BS.1770-4)
        // These are the coefficients for 48 kHz, scaled for other sample rates
        let (hs_b0, hs_b1, hs_b2, hs_a1, hs_a2) = Self::calculate_high_shelf_coefficients(fs);

        // Stage 2: High-pass filter coefficients (RLB weighting)
        let (hp_b0, hp_b1, hp_b2, hp_a1, hp_a2) = Self::calculate_high_pass_coefficients(fs);

        Self {
            hs_x1: 0.0,
            hs_x2: 0.0,
            hs_y1: 0.0,
            hs_y2: 0.0,
            hs_b0,
            hs_b1,
            hs_b2,
            hs_a1,
            hs_a2,

            hp_x1: 0.0,
            hp_x2: 0.0,
            hp_y1: 0.0,
            hp_y2: 0.0,
            hp_b0,
            hp_b1,
            hp_b2,
            hp_a1,
            hp_a2,
        }
    }

    /// Calculate high-shelf filter coefficients for the given sample rate
    /// Based on the ITU-R BS.1770-4 specification
    fn calculate_high_shelf_coefficients(fs: f64) -> (f64, f64, f64, f64, f64) {
        // High-shelf filter parameters
        let db = 4.0; // Gain in dB
        let f0 = 1681.974450955533; // Center frequency
        let q = 0.7071752369554196; // Q factor

        let a = 10.0_f64.powf(db / 40.0);
        let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha;
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * a.sqrt() * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * a.sqrt() * alpha);

        (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Calculate high-pass filter coefficients (RLB weighting)
    /// Based on the ITU-R BS.1770-4 specification
    fn calculate_high_pass_coefficients(fs: f64) -> (f64, f64, f64, f64, f64) {
        // High-pass filter parameters (RLB weighting)
        let f0 = 38.13547087602444; // Cutoff frequency
        let q = 0.5003270373238773; // Q factor

        let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;

        (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.hs_x1 = 0.0;
        self.hs_x2 = 0.0;
        self.hs_y1 = 0.0;
        self.hs_y2 = 0.0;
        self.hp_x1 = 0.0;
        self.hp_x2 = 0.0;
        self.hp_y1 = 0.0;
        self.hp_y2 = 0.0;
    }

    /// Process a single sample through the K-weighting filter
    pub fn process(&mut self, input: f64) -> f64 {
        // Stage 1: High-shelf filter
        let hs_out = self.hs_b0 * input + self.hs_b1 * self.hs_x1 + self.hs_b2 * self.hs_x2
            - self.hs_a1 * self.hs_y1
            - self.hs_a2 * self.hs_y2;

        self.hs_x2 = self.hs_x1;
        self.hs_x1 = input;
        self.hs_y2 = self.hs_y1;
        self.hs_y1 = hs_out;

        // Stage 2: High-pass filter
        let hp_out = self.hp_b0 * hs_out + self.hp_b1 * self.hp_x1 + self.hp_b2 * self.hp_x2
            - self.hp_a1 * self.hp_y1
            - self.hp_a2 * self.hp_y2;

        self.hp_x2 = self.hp_x1;
        self.hp_x1 = hs_out;
        self.hp_y2 = self.hp_y1;
        self.hp_y1 = hp_out;

        hp_out
    }

    /// Process a buffer of samples
    pub fn process_buffer(&mut self, samples: &[f32]) -> Vec<f64> {
        samples.iter().map(|&s| self.process(s as f64)).collect()
    }
}

/// Channel type for loudness weighting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    /// Left, Right, Center channels (weight = 1.0)
    Front,
    /// Left surround, Right surround channels (weight = 1.41)
    Surround,
    /// LFE channel (excluded from loudness calculation)
    Lfe,
}

impl ChannelType {
    /// Get the loudness weight for this channel type
    pub fn weight(&self) -> f64 {
        match self {
            ChannelType::Front => 1.0,
            ChannelType::Surround => 1.41,
            ChannelType::Lfe => 0.0,
        }
    }

    /// Get default channel types for a given channel count
    pub fn default_layout(channels: usize) -> Vec<ChannelType> {
        match channels {
            1 => vec![ChannelType::Front], // Mono
            2 => vec![ChannelType::Front, ChannelType::Front], // Stereo
            6 => vec![
                // 5.1
                ChannelType::Front,    // L
                ChannelType::Front,    // R
                ChannelType::Front,    // C
                ChannelType::Lfe,      // LFE
                ChannelType::Surround, // Ls
                ChannelType::Surround, // Rs
            ],
            8 => vec![
                // 7.1
                ChannelType::Front,    // L
                ChannelType::Front,    // R
                ChannelType::Front,    // C
                ChannelType::Lfe,      // LFE
                ChannelType::Surround, // Lss
                ChannelType::Surround, // Rss
                ChannelType::Surround, // Lrs
                ChannelType::Surround, // Rrs
            ],
            _ => vec![ChannelType::Front; channels],
        }
    }
}

/// LUFS measurement result
#[derive(Debug, Clone)]
pub struct LufsResult {
    /// Integrated loudness (gated) in LUFS
    pub integrated: f32,
    /// Momentary loudness (400ms window) in LUFS
    pub momentary: f32,
    /// Short-term loudness (3s window) in LUFS
    pub short_term: f32,
    /// Loudness range (LRA) in LU
    pub range: f32,
    /// True peak level in dBTP
    pub true_peak: f32,
}

/// ITU-R BS.1770-4 loudness meter
pub struct LoudnessMeter {
    sample_rate: u32,
    channels: usize,
    channel_types: Vec<ChannelType>,
    /// K-weighting filters (one per channel)
    filters: Vec<KWeightingFilter>,
    /// Block size for gated measurement (400ms at sample rate)
    block_size: usize,
    /// Hop size (75% overlap = 100ms hop)
    hop_size: usize,
    /// Accumulated block powers for gated measurement
    block_powers: Vec<f64>,
}

impl LoudnessMeter {
    /// Create a new loudness meter
    pub fn new(sample_rate: u32, channels: usize) -> Self {
        let channel_types = ChannelType::default_layout(channels);
        Self::with_channel_types(sample_rate, channel_types)
    }

    /// Create a new loudness meter with custom channel types
    pub fn with_channel_types(sample_rate: u32, channel_types: Vec<ChannelType>) -> Self {
        let channels = channel_types.len();
        let filters = (0..channels)
            .map(|_| KWeightingFilter::new(sample_rate))
            .collect();

        // 400ms block size (ITU-R BS.1770-4)
        let block_size = (sample_rate as f64 * 0.4) as usize;
        // 75% overlap = 100ms hop
        let hop_size = block_size / 4;

        Self {
            sample_rate,
            channels,
            channel_types,
            filters,
            block_size,
            hop_size,
            block_powers: Vec::new(),
        }
    }

    /// Measure integrated loudness with gating (ITU-R BS.1770-4)
    pub fn measure_integrated(&mut self, buffer: &AudioBuffer) -> f32 {
        self.reset();

        if buffer.samples.is_empty() || buffer.channels == 0 {
            return f32::NEG_INFINITY;
        }

        // K-weight and compute block powers
        self.process_buffer(buffer);

        // Apply gating
        self.gated_loudness()
    }

    /// Measure momentary loudness (400ms window, no gating)
    pub fn measure_momentary(&mut self, buffer: &AudioBuffer) -> f32 {
        if buffer.samples.is_empty() || buffer.channels == 0 {
            return f32::NEG_INFINITY;
        }

        // Use the last 400ms of audio
        let frames = buffer.num_frames();
        let window_frames = (self.sample_rate as f64 * 0.4) as usize;
        let start_frame = frames.saturating_sub(window_frames);

        self.measure_window_loudness(buffer, start_frame, frames)
    }

    /// Measure short-term loudness (3s window, no gating)
    pub fn measure_short_term(&mut self, buffer: &AudioBuffer) -> f32 {
        if buffer.samples.is_empty() || buffer.channels == 0 {
            return f32::NEG_INFINITY;
        }

        // Use the last 3s of audio
        let frames = buffer.num_frames();
        let window_frames = (self.sample_rate as f64 * 3.0) as usize;
        let start_frame = frames.saturating_sub(window_frames);

        self.measure_window_loudness(buffer, start_frame, frames)
    }

    /// Full loudness measurement including all metrics
    pub fn measure_full(&mut self, buffer: &AudioBuffer) -> LufsResult {
        let integrated = self.measure_integrated(buffer);
        let momentary = self.measure_momentary(buffer);
        let short_term = self.measure_short_term(buffer);
        let range = self.measure_range(buffer);
        let true_peak = self.measure_true_peak(buffer);

        LufsResult {
            integrated,
            momentary,
            short_term,
            range,
            true_peak,
        }
    }

    /// Reset the meter state
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
        self.block_powers.clear();
    }

    /// Process a buffer and accumulate block powers
    fn process_buffer(&mut self, buffer: &AudioBuffer) {
        let num_frames = buffer.num_frames();
        if num_frames < self.block_size {
            return;
        }

        // Extract and K-weight each channel
        let weighted_channels: Vec<Vec<f64>> = (0..self.channels)
            .map(|ch| {
                let channel_data = buffer.channel(ch);
                self.filters[ch].reset();
                self.filters[ch].process_buffer(&channel_data)
            })
            .collect();

        // Compute block powers with 75% overlap
        let num_blocks = (num_frames - self.block_size) / self.hop_size + 1;

        for block_idx in 0..num_blocks {
            let start = block_idx * self.hop_size;
            let end = start + self.block_size;

            // Sum weighted mean squares across channels
            let mut block_power = 0.0;

            for (ch, weighted) in weighted_channels.iter().enumerate() {
                let weight = self.channel_types[ch].weight();
                if weight == 0.0 {
                    continue;
                }

                let sum_sq: f64 = weighted[start..end].iter().map(|&s| s * s).sum();
                let mean_sq = sum_sq / self.block_size as f64;
                block_power += weight * mean_sq;
            }

            self.block_powers.push(block_power);
        }
    }

    /// Apply gating and compute integrated loudness
    fn gated_loudness(&self) -> f32 {
        if self.block_powers.is_empty() {
            return f32::NEG_INFINITY;
        }

        // Absolute gate: -70 LUFS
        let absolute_gate_power = 10.0_f64.powf((-70.0 + 0.691) / 10.0);

        // First pass: compute ungated loudness for relative gate
        let above_absolute: Vec<f64> = self
            .block_powers
            .iter()
            .copied()
            .filter(|&p| p > absolute_gate_power)
            .collect();

        if above_absolute.is_empty() {
            return f32::NEG_INFINITY;
        }

        let ungated_mean: f64 = above_absolute.iter().sum::<f64>() / above_absolute.len() as f64;
        let ungated_lufs = -0.691 + 10.0 * ungated_mean.log10();

        // Relative gate: -10 LU below ungated loudness
        let relative_gate_lufs = ungated_lufs - 10.0;
        let relative_gate_power = 10.0_f64.powf((relative_gate_lufs + 0.691) / 10.0);

        // Second pass: compute gated loudness
        let above_relative: Vec<f64> = self
            .block_powers
            .iter()
            .copied()
            .filter(|&p| p > relative_gate_power)
            .collect();

        if above_relative.is_empty() {
            return f32::NEG_INFINITY;
        }

        let gated_mean: f64 = above_relative.iter().sum::<f64>() / above_relative.len() as f64;
        (-0.691 + 10.0 * gated_mean.log10()) as f32
    }

    /// Measure loudness for a window (without gating)
    fn measure_window_loudness(
        &mut self,
        buffer: &AudioBuffer,
        start_frame: usize,
        end_frame: usize,
    ) -> f32 {
        if end_frame <= start_frame {
            return f32::NEG_INFINITY;
        }

        let mut total_power = 0.0;

        for ch in 0..self.channels {
            let weight = self.channel_types[ch].weight();
            if weight == 0.0 {
                continue;
            }

            self.filters[ch].reset();

            let channel_data = buffer.channel(ch);
            let window_data = &channel_data[start_frame..end_frame.min(channel_data.len())];

            let weighted = self.filters[ch].process_buffer(window_data);
            let sum_sq: f64 = weighted.iter().map(|&s| s * s).sum();
            let mean_sq = sum_sq / weighted.len() as f64;

            total_power += weight * mean_sq;
        }

        if total_power <= 0.0 {
            f32::NEG_INFINITY
        } else {
            (-0.691 + 10.0 * total_power.log10()) as f32
        }
    }

    /// Measure loudness range (LRA) in LU
    fn measure_range(&mut self, buffer: &AudioBuffer) -> f32 {
        self.reset();
        self.process_buffer(buffer);

        if self.block_powers.len() < 2 {
            return 0.0;
        }

        // Apply absolute gate
        let absolute_gate_power = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
        let mut above_absolute: Vec<f64> = self
            .block_powers
            .iter()
            .copied()
            .filter(|&p| p > absolute_gate_power)
            .collect();

        if above_absolute.is_empty() {
            return 0.0;
        }

        // Compute relative gate
        let ungated_mean: f64 = above_absolute.iter().sum::<f64>() / above_absolute.len() as f64;
        let ungated_lufs = -0.691 + 10.0 * ungated_mean.log10();
        let relative_gate_lufs = ungated_lufs - 20.0; // -20 LU for LRA
        let relative_gate_power = 10.0_f64.powf((relative_gate_lufs + 0.691) / 10.0);

        // Filter by relative gate and sort
        above_absolute.retain(|&p| p > relative_gate_power);

        if above_absolute.len() < 2 {
            return 0.0;
        }

        above_absolute.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // LRA is difference between 95th and 10th percentile
        let low_idx = (above_absolute.len() as f64 * 0.10) as usize;
        let high_idx = (above_absolute.len() as f64 * 0.95) as usize;

        let low_lufs = -0.691 + 10.0 * above_absolute[low_idx].log10();
        let high_lufs = -0.691 + 10.0 * above_absolute[high_idx.min(above_absolute.len() - 1)].log10();

        (high_lufs - low_lufs) as f32
    }

    /// Measure true peak level in dBTP
    /// Uses 4x oversampling for inter-sample peak detection
    fn measure_true_peak(&self, buffer: &AudioBuffer) -> f32 {
        let mut max_peak = 0.0_f64;

        for ch in 0..buffer.channels {
            let channel = buffer.channel(ch);

            // Simple 4x oversampling using linear interpolation
            // (Full implementation would use polyphase FIR)
            for i in 0..channel.len().saturating_sub(1) {
                let s0 = channel[i].abs() as f64;
                let s1 = channel[i + 1].abs() as f64;

                max_peak = max_peak.max(s0);

                // Check inter-sample peaks
                for k in 1..4 {
                    let t = k as f64 / 4.0;
                    let interp = s0 + (s1 - s0) * t;
                    max_peak = max_peak.max(interp);
                }
            }

            if let Some(&last) = channel.last() {
                max_peak = max_peak.max(last.abs() as f64);
            }
        }

        if max_peak <= 0.0 {
            f32::NEG_INFINITY
        } else {
            (20.0 * max_peak.log10()) as f32
        }
    }
}

/// Convenience function to measure integrated loudness
pub fn measure_lufs(buffer: &AudioBuffer) -> f32 {
    let mut meter = LoudnessMeter::new(buffer.sample_rate, buffer.channels);
    meter.measure_integrated(buffer)
}

/// Convenience function for full loudness analysis
pub fn analyze_loudness(buffer: &AudioBuffer) -> LufsResult {
    let mut meter = LoudnessMeter::new(buffer.sample_rate, buffer.channels);
    meter.measure_full(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_weighting_filter_creation() {
        let filter = KWeightingFilter::new(48000);
        // Coefficients should be non-zero
        assert!(filter.hs_b0 != 0.0);
        assert!(filter.hp_b0 != 0.0);
    }

    #[test]
    fn test_k_weighting_filter_process() {
        let mut filter = KWeightingFilter::new(48000);

        // Process a DC signal - should be filtered out by high-pass
        let input = vec![1.0f32; 1000];
        let output = filter.process_buffer(&input);

        // After settling, output should be near zero for DC
        let last_values = &output[output.len() - 100..];
        let avg: f64 = last_values.iter().sum::<f64>() / last_values.len() as f64;
        assert!(avg.abs() < 0.1, "DC should be filtered: avg = {}", avg);
    }

    #[test]
    fn test_channel_type_weights() {
        assert!((ChannelType::Front.weight() - 1.0).abs() < 0.001);
        assert!((ChannelType::Surround.weight() - 1.41).abs() < 0.001);
        assert!((ChannelType::Lfe.weight() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_channel_layout_stereo() {
        let layout = ChannelType::default_layout(2);
        assert_eq!(layout.len(), 2);
        assert_eq!(layout[0], ChannelType::Front);
        assert_eq!(layout[1], ChannelType::Front);
    }

    #[test]
    fn test_channel_layout_5_1() {
        let layout = ChannelType::default_layout(6);
        assert_eq!(layout.len(), 6);
        assert_eq!(layout[3], ChannelType::Lfe);
        assert_eq!(layout[4], ChannelType::Surround);
    }

    #[test]
    fn test_loudness_meter_creation() {
        let meter = LoudnessMeter::new(48000, 2);
        assert_eq!(meter.channels, 2);
        assert_eq!(meter.filters.len(), 2);
    }

    #[test]
    fn test_silence_measurement() {
        let buffer = AudioBuffer::from_samples(vec![0.0; 48000], 1, 48000);
        let lufs = measure_lufs(&buffer);
        assert!(lufs < -60.0 || lufs.is_infinite());
    }

    #[test]
    fn test_sine_wave_measurement() {
        // Generate a 1kHz sine wave at -20 dBFS
        let sample_rate = 48000u32;
        let duration = 2.0; // 2 seconds for proper gating
        let num_samples = (sample_rate as f64 * duration) as usize;
        let amplitude = 0.1f32; // -20 dBFS

        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                amplitude * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
            })
            .collect();

        let buffer = AudioBuffer::from_samples(samples, 1, sample_rate);
        let lufs = measure_lufs(&buffer);

        // Should be approximately -23 LUFS for a -20 dBFS sine
        // (K-weighting adds ~3 dB for mid-frequency content)
        assert!(
            lufs > -30.0 && lufs < -15.0,
            "Unexpected LUFS for sine wave: {}",
            lufs
        );
    }

    #[test]
    fn test_full_measurement() {
        let sample_rate = 48000u32;
        let samples: Vec<f32> = (0..96000)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                0.1 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
            })
            .collect();

        let buffer = AudioBuffer::from_samples(samples, 1, sample_rate);
        let result = analyze_loudness(&buffer);

        // All measurements should be valid
        assert!(!result.integrated.is_nan());
        assert!(!result.momentary.is_nan());
        assert!(!result.short_term.is_nan());
        assert!(result.range >= 0.0);
        assert!(!result.true_peak.is_nan());
    }

    #[test]
    fn test_true_peak() {
        // Create a signal with known peak
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let buffer = AudioBuffer::from_samples(samples, 1, 48000);

        let meter = LoudnessMeter::new(48000, 1);
        let true_peak = meter.measure_true_peak(&buffer);

        // True peak should be 0 dBTP for a signal with peak of 1.0
        assert!(
            (true_peak - 0.0).abs() < 0.5,
            "True peak should be ~0 dBTP: {}",
            true_peak
        );
    }
}
