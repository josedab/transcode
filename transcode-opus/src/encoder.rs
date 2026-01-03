//! Opus encoder implementation.
//!
//! This module provides the main Opus encoder that can encode using SILK, CELT, or hybrid mode.

use crate::celt::{CeltBandwidth, CeltEncoder};
use crate::error::{OpusError, Result};
use crate::range_coder::RangeEncoder;
use crate::silk::{SilkEncoder, SilkSampleRate};
use crate::{
    Application, Bandwidth, FrameSize, OpusMode, SignalType, VbrMode, SAMPLE_RATES,
};
use transcode_core::sample::{SampleBuffer, SampleFormat};

/// Opus encoder configuration.
#[derive(Debug, Clone)]
pub struct OpusEncoderConfig {
    /// Sample rate (must be 8000, 12000, 16000, 24000, or 48000).
    pub sample_rate: u32,
    /// Number of channels (1 or 2).
    pub channels: u8,
    /// Target bitrate in bits per second.
    pub bitrate: u32,
    /// Application type.
    pub application: Application,
    /// VBR mode.
    pub vbr_mode: VbrMode,
    /// Signal type hint.
    pub signal_type: SignalType,
    /// Maximum bandwidth.
    pub max_bandwidth: Bandwidth,
    /// Frame size.
    pub frame_size: FrameSize,
    /// Complexity (0-10).
    pub complexity: u8,
    /// Enable DTX (Discontinuous Transmission).
    pub dtx: bool,
    /// Enable in-band FEC.
    pub fec: bool,
    /// Expected packet loss percentage (0-100).
    pub packet_loss_perc: u8,
}

impl Default for OpusEncoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            bitrate: 64000,
            application: Application::Audio,
            vbr_mode: VbrMode::Vbr,
            signal_type: SignalType::Auto,
            max_bandwidth: Bandwidth::Fullband,
            frame_size: FrameSize::Ms20,
            complexity: 10,
            dtx: false,
            fec: false,
            packet_loss_perc: 0,
        }
    }
}

impl OpusEncoderConfig {
    /// Create config for voice/speech.
    pub fn for_voice(sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate: 24000,
            application: Application::Voip,
            signal_type: SignalType::Voice,
            max_bandwidth: Bandwidth::Wideband,
            ..Default::default()
        }
    }

    /// Create config for music.
    pub fn for_music(sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate: 96000,
            application: Application::Audio,
            signal_type: SignalType::Music,
            max_bandwidth: Bandwidth::Fullband,
            ..Default::default()
        }
    }

    /// Create config for low latency.
    pub fn for_low_latency(sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate: 48000,
            application: Application::LowDelay,
            frame_size: FrameSize::Ms5,
            ..Default::default()
        }
    }
}

/// Mode decision state.
#[derive(Debug)]
struct ModeDecision {
    /// Current mode.
    current_mode: OpusMode,
    /// Mode history for hysteresis.
    mode_history: [OpusMode; 8],
    /// History index.
    history_idx: usize,
    /// Voice activity detector state.
    vad_state: VadState,
    /// Speech/music classifier.
    classifier_state: ClassifierState,
}

impl ModeDecision {
    fn new() -> Self {
        Self {
            current_mode: OpusMode::Celt,
            mode_history: [OpusMode::Celt; 8],
            history_idx: 0,
            vad_state: VadState::new(),
            classifier_state: ClassifierState::new(),
        }
    }

    fn decide_mode(&mut self, samples: &[f32], config: &OpusEncoderConfig) -> OpusMode {
        // Forced mode based on signal type hint
        match config.signal_type {
            SignalType::Voice => return OpusMode::Silk,
            SignalType::Music => return OpusMode::Celt,
            SignalType::Auto => {}
        }

        // Application hint
        match config.application {
            Application::Voip => {
                // Prefer SILK for voice
                if self.vad_state.is_voice(samples) {
                    return OpusMode::Silk;
                }
            }
            Application::LowDelay => {
                // Always use CELT for lowest latency
                return OpusMode::Celt;
            }
            Application::Audio => {}
        }

        // Bandwidth constraint
        match config.max_bandwidth {
            Bandwidth::Narrowband | Bandwidth::Mediumband | Bandwidth::Wideband => {
                return OpusMode::Silk;
            }
            _ => {}
        }

        // Automatic classification
        let music_prob = self.classifier_state.music_probability(samples);

        let mode = if music_prob > 0.7 {
            OpusMode::Celt
        } else if music_prob < 0.3 {
            OpusMode::Silk
        } else {
            // Hybrid for mixed content at high bitrates
            if config.bitrate > 40000 {
                OpusMode::Hybrid
            } else {
                // Use history for hysteresis
                self.current_mode
            }
        };

        // Update history
        self.mode_history[self.history_idx] = mode;
        self.history_idx = (self.history_idx + 1) % 8;
        self.current_mode = mode;

        mode
    }

    fn reset(&mut self) {
        self.current_mode = OpusMode::Celt;
        self.mode_history = [OpusMode::Celt; 8];
        self.history_idx = 0;
        self.vad_state.reset();
        self.classifier_state.reset();
    }
}

/// Voice Activity Detector state.
#[derive(Debug)]
struct VadState {
    /// Energy history.
    energy_history: [f32; 16],
    /// History index.
    history_idx: usize,
    /// Noise floor estimate.
    noise_floor: f32,
}

impl VadState {
    fn new() -> Self {
        Self {
            energy_history: [0.0; 16],
            history_idx: 0,
            noise_floor: 0.0001,
        }
    }

    fn is_voice(&mut self, samples: &[f32]) -> bool {
        // Compute frame energy
        let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;

        // Update history
        self.energy_history[self.history_idx] = energy;
        self.history_idx = (self.history_idx + 1) % 16;

        // Update noise floor (slow adaptation)
        let min_energy = self.energy_history.iter().cloned().fold(f32::MAX, f32::min);
        self.noise_floor = self.noise_floor * 0.99 + min_energy * 0.01;

        // Voice if energy is significantly above noise floor
        energy > self.noise_floor * 4.0
    }

    fn reset(&mut self) {
        self.energy_history = [0.0; 16];
        self.history_idx = 0;
        self.noise_floor = 0.0001;
    }
}

/// Speech/Music classifier state.
#[derive(Debug)]
struct ClassifierState {
    /// Spectral flatness history.
    flatness_history: [f32; 8],
    /// History index.
    history_idx: usize,
    /// Zero crossing rate history.
    zcr_history: [f32; 8],
}

impl ClassifierState {
    fn new() -> Self {
        Self {
            flatness_history: [0.5; 8],
            history_idx: 0,
            zcr_history: [0.0; 8],
        }
    }

    fn music_probability(&mut self, samples: &[f32]) -> f32 {
        // Compute spectral flatness (geometric mean / arithmetic mean)
        // High flatness = more noise-like (speech), low flatness = more tonal (music)

        let mut sum = 0.0f64;
        let mut log_sum = 0.0f64;
        let mut count = 0;

        // Simple energy in frequency bands
        let window_size = 256.min(samples.len());
        for i in 0..window_size {
            let val = samples.get(i).copied().unwrap_or(0.0).abs() + 1e-10;
            sum += val as f64;
            log_sum += (val as f64).ln();
            count += 1;
        }

        let arithmetic_mean = sum / count as f64;
        let geometric_mean = (log_sum / count as f64).exp();
        let flatness = (geometric_mean / arithmetic_mean).clamp(0.0, 1.0) as f32;

        // Compute zero crossing rate
        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        let zcr = crossings as f32 / samples.len() as f32;

        // Update history
        self.flatness_history[self.history_idx] = flatness;
        self.zcr_history[self.history_idx] = zcr;
        self.history_idx = (self.history_idx + 1) % 8;

        // Average flatness and ZCR
        let avg_flatness: f32 = self.flatness_history.iter().sum::<f32>() / 8.0;
        let avg_zcr: f32 = self.zcr_history.iter().sum::<f32>() / 8.0;

        // Music: low flatness, low ZCR
        // Speech: higher flatness, higher ZCR
        let music_score = (1.0 - avg_flatness) * 0.6 + (1.0 - avg_zcr * 5.0).max(0.0) * 0.4;

        music_score.clamp(0.0, 1.0)
    }

    fn reset(&mut self) {
        self.flatness_history = [0.5; 8];
        self.zcr_history = [0.0; 8];
        self.history_idx = 0;
    }
}

/// Opus encoder.
pub struct OpusEncoder {
    /// Configuration.
    config: OpusEncoderConfig,
    /// SILK encoder.
    silk_encoder: Option<SilkEncoder>,
    /// CELT encoder.
    celt_encoder: CeltEncoder,
    /// Mode decision state.
    mode_decision: ModeDecision,
    /// Previous mode.
    prev_mode: OpusMode,
    /// Input resampler state.
    resampler: ResamplerState,
    /// Frame counter.
    frame_count: u64,
    /// Bits used in current frame.
    bits_used: u32,
}

impl OpusEncoder {
    /// Create a new Opus encoder.
    pub fn new(config: OpusEncoderConfig) -> Result<Self> {
        // Validate sample rate
        if !SAMPLE_RATES.contains(&config.sample_rate) {
            return Err(OpusError::InvalidSampleRate(config.sample_rate));
        }

        // Validate channels
        if config.channels != 1 && config.channels != 2 {
            return Err(OpusError::InvalidChannels(config.channels));
        }

        // Validate bitrate
        if config.bitrate < 500 || config.bitrate > 512000 {
            return Err(OpusError::EncoderConfig(format!(
                "Invalid bitrate: {} (must be 500-512000)",
                config.bitrate
            )));
        }

        // Create CELT encoder (always needed)
        let frame_size = config.frame_size.samples_48k();
        let celt_encoder = CeltEncoder::new(config.channels, frame_size, config.bitrate);

        // Create resampler
        let resampler = ResamplerState::new(config.sample_rate, 48000);

        Ok(Self {
            config,
            silk_encoder: None,
            celt_encoder,
            mode_decision: ModeDecision::new(),
            prev_mode: OpusMode::Celt,
            resampler,
            frame_count: 0,
            bits_used: 0,
        })
    }

    /// Encode audio samples to an Opus packet.
    pub fn encode(&mut self, samples: &SampleBuffer) -> Result<Vec<u8>> {
        // Convert to f32 samples
        let input_samples = self.extract_samples(samples)?;

        // Resample to 48kHz if needed
        let samples_48k = if self.config.sample_rate != 48000 {
            self.resampler.resample(&input_samples, self.config.channels)
        } else {
            input_samples
        };

        let frame_size = self.config.frame_size.samples_48k();

        if samples_48k.len() < frame_size * self.config.channels as usize {
            return Err(OpusError::InvalidFrameSize(samples_48k.len()));
        }

        // Decide encoding mode
        let mode = self.mode_decision.decide_mode(&samples_48k, &self.config);

        // Get bandwidth
        let bandwidth = self.decide_bandwidth(mode);

        // Create range encoder
        let mut writer = RangeEncoder::new();

        // Encode based on mode
        match mode {
            OpusMode::Silk => {
                self.encode_silk(&samples_48k, bandwidth, &mut writer)?;
            }
            OpusMode::Celt => {
                self.encode_celt(&samples_48k, bandwidth, &mut writer)?;
            }
            OpusMode::Hybrid => {
                self.encode_hybrid(&samples_48k, bandwidth, &mut writer)?;
            }
        }

        // Get encoded data
        let encoded_data = writer.finish()?;

        // Create TOC byte
        let toc = self.create_toc(mode, bandwidth);

        // Create final packet
        let mut packet = Vec::with_capacity(1 + encoded_data.len());
        packet.push(toc);
        packet.extend_from_slice(&encoded_data);

        self.prev_mode = mode;
        self.frame_count += 1;
        self.bits_used = (packet.len() * 8) as u32;

        Ok(packet)
    }

    /// Extract f32 samples from buffer.
    fn extract_samples(&self, buffer: &SampleBuffer) -> Result<Vec<f32>> {
        let data = buffer.data();

        let samples = match buffer.format {
            SampleFormat::F32 => {
                let floats: &[f32] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                };
                floats.to_vec()
            }
            SampleFormat::S16 => {
                let shorts: &[i16] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i16, data.len() / 2)
                };
                shorts.iter().map(|&s| s as f32 / 32768.0).collect()
            }
            _ => {
                return Err(OpusError::UnsupportedConfig(format!(
                    "Unsupported sample format: {:?}",
                    buffer.format
                )));
            }
        };

        Ok(samples)
    }

    /// Decide bandwidth based on mode and config.
    fn decide_bandwidth(&self, mode: OpusMode) -> Bandwidth {
        let max_bw = self.config.max_bandwidth;

        // SILK can't do fullband
        if mode == OpusMode::Silk && max_bw == Bandwidth::Fullband {
            return Bandwidth::SuperWideband;
        }

        // Limit by bitrate
        let bitrate_bw = if self.config.bitrate < 12000 {
            Bandwidth::Narrowband
        } else if self.config.bitrate < 20000 {
            Bandwidth::Wideband
        } else if self.config.bitrate < 40000 {
            Bandwidth::SuperWideband
        } else {
            Bandwidth::Fullband
        };

        // Return minimum of constraints
        if (max_bw as u8) < (bitrate_bw as u8) {
            max_bw
        } else {
            bitrate_bw
        }
    }

    /// Create TOC byte.
    fn create_toc(&self, mode: OpusMode, bandwidth: Bandwidth) -> u8 {
        let frame_size_code = match self.config.frame_size {
            FrameSize::Ms2_5 => 0,
            FrameSize::Ms5 => 1,
            FrameSize::Ms10 => 2,
            FrameSize::Ms20 => 3,
            FrameSize::Ms40 => 3, // Use 20ms config
            FrameSize::Ms60 => 3, // Use 20ms config
        };

        let config = match mode {
            OpusMode::Silk => {
                match bandwidth {
                    Bandwidth::Narrowband => frame_size_code,
                    Bandwidth::Mediumband => 4 + frame_size_code,
                    Bandwidth::Wideband | Bandwidth::SuperWideband | Bandwidth::Fullband => {
                        8 + frame_size_code
                    }
                }
            }
            OpusMode::Hybrid => {
                match bandwidth {
                    Bandwidth::SuperWideband => 12 + (frame_size_code / 2),
                    Bandwidth::Fullband => 14 + (frame_size_code / 2),
                    _ => 12, // Fallback
                }
            }
            OpusMode::Celt => {
                16 + match bandwidth {
                    Bandwidth::Narrowband => 0,
                    Bandwidth::Wideband | Bandwidth::Mediumband => 4,
                    Bandwidth::SuperWideband => 8,
                    Bandwidth::Fullband => 12,
                } + frame_size_code
            }
        };

        let stereo_bit = if self.config.channels == 2 { 0x04 } else { 0x00 };
        let frame_count_code = 0x00; // Single frame

        (config << 3) | stereo_bit | frame_count_code
    }

    /// Encode using SILK.
    fn encode_silk(
        &mut self,
        samples: &[f32],
        bandwidth: Bandwidth,
        writer: &mut RangeEncoder,
    ) -> Result<()> {
        // Get SILK sample rate
        let silk_rate = match bandwidth {
            Bandwidth::Narrowband => SilkSampleRate::Nb8000,
            Bandwidth::Mediumband => SilkSampleRate::Mb12000,
            Bandwidth::Wideband => SilkSampleRate::Wb16000,
            Bandwidth::SuperWideband | Bandwidth::Fullband => SilkSampleRate::Swb24000,
        };

        // Create SILK encoder if needed
        if self.silk_encoder.is_none() {
            let silk_bitrate = self.config.bitrate.min(40000);
            self.silk_encoder = Some(SilkEncoder::new(
                silk_rate,
                self.config.channels,
                silk_bitrate,
            ));
        }

        // Downsample to SILK rate
        let silk_samples = Self::downsample_for_silk_static(samples, silk_rate);

        // Encode
        let frame_ms = match self.config.frame_size {
            FrameSize::Ms2_5 | FrameSize::Ms5 => 10, // Minimum SILK frame
            FrameSize::Ms10 => 10,
            FrameSize::Ms20 => 20,
            FrameSize::Ms40 => 40,
            FrameSize::Ms60 => 60,
        };

        let silk = self.silk_encoder.as_mut().unwrap();
        silk.encode_frame(&silk_samples, frame_ms, writer)?;

        Ok(())
    }

    /// Downsample for SILK encoder (static version to avoid borrow issues).
    fn downsample_for_silk_static(samples: &[f32], rate: SilkSampleRate) -> Vec<f32> {
        let ratio = 48000 / (rate as u32);
        if ratio == 1 {
            return samples.to_vec();
        }

        let output_len = samples.len() / ratio as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            // Simple averaging filter
            let mut sum = 0.0f32;
            for j in 0..ratio as usize {
                sum += samples.get(i * ratio as usize + j).copied().unwrap_or(0.0);
            }
            output.push(sum / ratio as f32);
        }

        output
    }

    /// Encode using CELT.
    fn encode_celt(
        &mut self,
        samples: &[f32],
        bandwidth: Bandwidth,
        writer: &mut RangeEncoder,
    ) -> Result<()> {
        let celt_bandwidth = match bandwidth {
            Bandwidth::Narrowband => CeltBandwidth::Narrow,
            Bandwidth::Mediumband => CeltBandwidth::Medium,
            Bandwidth::Wideband => CeltBandwidth::Wide,
            Bandwidth::SuperWideband => CeltBandwidth::SuperWide,
            Bandwidth::Fullband => CeltBandwidth::Full,
        };

        self.celt_encoder.encode_frame(samples, celt_bandwidth, writer)?;

        Ok(())
    }

    /// Encode using hybrid mode (SILK + CELT).
    fn encode_hybrid(
        &mut self,
        samples: &[f32],
        bandwidth: Bandwidth,
        writer: &mut RangeEncoder,
    ) -> Result<()> {
        // Encode SILK for low frequencies
        self.encode_silk(samples, Bandwidth::Wideband, writer)?;

        // Encode CELT for high frequencies
        // Note: In real hybrid mode, CELT only encodes frequencies above 8kHz
        // This is a simplified implementation
        let celt_bandwidth = match bandwidth {
            Bandwidth::SuperWideband => CeltBandwidth::SuperWide,
            Bandwidth::Fullband => CeltBandwidth::Full,
            _ => CeltBandwidth::Wide,
        };

        // High-pass filter for CELT portion
        let hp_samples = self.high_pass_filter(samples);
        self.celt_encoder.encode_frame(&hp_samples, celt_bandwidth, writer)?;

        Ok(())
    }

    /// Simple high-pass filter for hybrid mode.
    fn high_pass_filter(&self, samples: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(samples.len());

        // Simple first-order HP at ~6kHz
        let alpha = 0.8f32;
        let mut prev_in = 0.0f32;
        let mut prev_out = 0.0f32;

        for &sample in samples {
            let out = alpha * (prev_out + sample - prev_in);
            output.push(out);
            prev_in = sample;
            prev_out = out;
        }

        output
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &OpusEncoderConfig {
        &self.config
    }

    /// Set the target bitrate.
    pub fn set_bitrate(&mut self, bitrate: u32) -> Result<()> {
        if !(500..=512000).contains(&bitrate) {
            return Err(OpusError::EncoderConfig(format!(
                "Invalid bitrate: {}",
                bitrate
            )));
        }
        self.config.bitrate = bitrate;
        Ok(())
    }

    /// Set the complexity.
    pub fn set_complexity(&mut self, complexity: u8) -> Result<()> {
        if complexity > 10 {
            return Err(OpusError::EncoderConfig(format!(
                "Invalid complexity: {} (must be 0-10)",
                complexity
            )));
        }
        self.config.complexity = complexity;
        Ok(())
    }

    /// Set the signal type hint.
    pub fn set_signal_type(&mut self, signal_type: SignalType) {
        self.config.signal_type = signal_type;
    }

    /// Set the maximum bandwidth.
    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.config.max_bandwidth = bandwidth;
    }

    /// Enable/disable FEC.
    pub fn set_fec(&mut self, enable: bool) {
        self.config.fec = enable;
    }

    /// Set expected packet loss percentage.
    pub fn set_packet_loss_perc(&mut self, perc: u8) {
        self.config.packet_loss_perc = perc.min(100);
    }

    /// Get the number of bits used in the last encoded frame.
    pub fn last_bits_used(&self) -> u32 {
        self.bits_used
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the number of channels.
    pub fn channels(&self) -> u8 {
        self.config.channels
    }

    /// Get the frame size in samples.
    pub fn frame_size(&self) -> usize {
        self.config.frame_size.samples_48k() * self.config.sample_rate as usize / 48000
    }

    /// Reset the encoder state.
    pub fn reset(&mut self) {
        if let Some(ref mut silk) = self.silk_encoder {
            silk.reset();
        }
        self.celt_encoder.reset();
        self.mode_decision.reset();
        self.resampler.reset();
        self.frame_count = 0;
        self.bits_used = 0;
        self.prev_mode = OpusMode::Celt;
    }
}

/// Simple resampler for input conversion.
#[derive(Debug)]
struct ResamplerState {
    /// Input sample rate.
    input_rate: u32,
    /// Output sample rate.
    output_rate: u32,
    /// Previous samples for interpolation.
    prev_samples: Vec<f32>,
}

impl ResamplerState {
    fn new(input_rate: u32, output_rate: u32) -> Self {
        Self {
            input_rate,
            output_rate,
            prev_samples: vec![0.0; 8],
        }
    }

    fn resample(&mut self, input: &[f32], channels: u8) -> Vec<f32> {
        if self.input_rate == self.output_rate {
            return input.to_vec();
        }

        let ratio = self.output_rate as f64 / self.input_rate as f64;
        let input_samples = input.len() / channels as usize;
        let output_samples = (input_samples as f64 * ratio).ceil() as usize;

        let mut output = Vec::with_capacity(output_samples * channels as usize);

        for out_idx in 0..output_samples {
            let in_pos = out_idx as f64 / ratio;
            let in_idx = in_pos.floor() as usize;
            let frac = in_pos - in_idx as f64;

            for ch in 0..channels as usize {
                let idx0 = (in_idx * channels as usize + ch).min(input.len().saturating_sub(1));
                let idx1 = ((in_idx + 1) * channels as usize + ch).min(input.len().saturating_sub(1));

                let s0 = input.get(idx0).copied().unwrap_or(0.0);
                let s1 = input.get(idx1).copied().unwrap_or(s0);

                // Linear interpolation
                output.push(s0 * (1.0 - frac as f32) + s1 * frac as f32);
            }
        }

        output
    }

    fn reset(&mut self) {
        self.prev_samples.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_default() {
        let config = OpusEncoderConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bitrate, 64000);
    }

    #[test]
    fn test_encoder_config_voice() {
        let config = OpusEncoderConfig::for_voice(16000, 1);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.application, Application::Voip);
    }

    #[test]
    fn test_encoder_config_music() {
        let config = OpusEncoderConfig::for_music(48000, 2);
        assert_eq!(config.bitrate, 96000);
        assert_eq!(config.application, Application::Audio);
    }

    #[test]
    fn test_encoder_creation() {
        let config = OpusEncoderConfig::default();
        let encoder = OpusEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_invalid_sample_rate() {
        let mut config = OpusEncoderConfig::default();
        config.sample_rate = 44100;
        let encoder = OpusEncoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_invalid_channels() {
        let mut config = OpusEncoderConfig::default();
        config.channels = 5;
        let encoder = OpusEncoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_toc_creation() {
        let config = OpusEncoderConfig::default();
        let encoder = OpusEncoder::new(config).unwrap();

        // CELT fullband
        let toc = encoder.create_toc(OpusMode::Celt, Bandwidth::Fullband);
        assert!((toc >> 3) >= 16); // CELT config range

        // SILK wideband
        let toc = encoder.create_toc(OpusMode::Silk, Bandwidth::Wideband);
        assert!((toc >> 3) <= 15); // SILK config range
    }

    #[test]
    fn test_encoder_components() {
        // Test encoder creation
        let config = OpusEncoderConfig::default();
        let encoder = OpusEncoder::new(config).unwrap();

        // Verify encoder is properly initialized
        assert_eq!(encoder.channels(), 2);
        assert_eq!(encoder.sample_rate(), 48000);

        // Test TOC byte creation
        let toc_silk = encoder.create_toc(crate::OpusMode::Silk, crate::Bandwidth::Wideband);
        assert!(toc_silk > 0);

        let toc_celt = encoder.create_toc(crate::OpusMode::Celt, crate::Bandwidth::Fullband);
        assert!(toc_celt > 0);
        assert!(toc_celt != toc_silk); // Different modes should give different TOC
    }

    #[test]
    fn test_mode_decision() {
        let config = OpusEncoderConfig::for_voice(48000, 1);
        let mut encoder = OpusEncoder::new(config).unwrap();

        // Voice config should prefer SILK
        let samples = vec![0.1f32; 960];
        let mode = encoder.mode_decision.decide_mode(&samples, &encoder.config);
        assert_eq!(mode, OpusMode::Silk);
    }

    #[test]
    fn test_set_bitrate() {
        let config = OpusEncoderConfig::default();
        let mut encoder = OpusEncoder::new(config).unwrap();

        assert!(encoder.set_bitrate(128000).is_ok());
        assert_eq!(encoder.config().bitrate, 128000);

        assert!(encoder.set_bitrate(100).is_err()); // Too low
    }

    #[test]
    fn test_resampler() {
        let mut resampler = ResamplerState::new(24000, 48000);
        let input: Vec<f32> = (0..240).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = resampler.resample(&input, 1);
        assert!(output.len() > 400 && output.len() < 500);
    }
}
