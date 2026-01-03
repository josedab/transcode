//! Audio intelligence for transcode
//!
//! This crate provides audio enhancement, upsampling, and classification.

mod error;
mod enhance;
mod upsample;
mod classify;

pub use error::*;
pub use enhance::*;
pub use upsample::*;
pub use classify::*;

/// Result type for audio AI operations
pub type Result<T> = std::result::Result<T, AudioAiError>;

/// Audio buffer
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Sample data (interleaved)
    pub samples: Vec<f32>,
    /// Number of channels
    pub channels: usize,
    /// Sample rate
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Create a new audio buffer
    pub fn new(channels: usize, sample_rate: u32) -> Self {
        Self {
            samples: Vec::new(),
            channels,
            sample_rate,
        }
    }

    /// Create from samples
    pub fn from_samples(samples: Vec<f32>, channels: usize, sample_rate: u32) -> Self {
        Self {
            samples,
            channels,
            sample_rate,
        }
    }

    /// Get number of frames
    pub fn num_frames(&self) -> usize {
        self.samples.len() / self.channels
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f64 {
        self.num_frames() as f64 / self.sample_rate as f64
    }

    /// Get channel data
    pub fn channel(&self, ch: usize) -> Vec<f32> {
        self.samples
            .iter()
            .skip(ch)
            .step_by(self.channels)
            .copied()
            .collect()
    }
}

/// Audio enhancement configuration
#[derive(Debug, Clone)]
pub struct AudioEnhanceConfig {
    /// Enable noise reduction
    pub noise_reduction: bool,
    /// Noise reduction strength (0.0-1.0)
    pub noise_reduction_strength: f32,
    /// Enable normalization
    pub normalize: bool,
    /// Target loudness in LUFS
    pub target_lufs: f32,
    /// Enable dynamic range compression
    pub compress: bool,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Enable de-essing
    pub deess: bool,
    /// De-essing threshold
    pub deess_threshold: f32,
}

impl Default for AudioEnhanceConfig {
    fn default() -> Self {
        Self {
            noise_reduction: true,
            noise_reduction_strength: 0.5,
            normalize: true,
            target_lufs: -14.0,
            compress: false,
            compression_ratio: 4.0,
            deess: false,
            deess_threshold: 0.5,
        }
    }
}

/// Audio enhancer
pub struct AudioEnhancer {
    config: AudioEnhanceConfig,
}

impl AudioEnhancer {
    /// Create a new audio enhancer
    pub fn new(config: AudioEnhanceConfig) -> Self {
        Self { config }
    }

    /// Enhance audio buffer
    #[allow(clippy::field_reassign_with_default)]
    pub fn enhance(&self, buffer: &mut AudioBuffer) -> Result<EnhanceStats> {
        let mut stats = EnhanceStats::default();

        // Measure input loudness
        stats.input_lufs = self.measure_lufs(buffer);

        // Apply noise reduction
        if self.config.noise_reduction {
            self.apply_noise_reduction(buffer)?;
        }

        // Apply normalization
        if self.config.normalize {
            self.apply_normalization(buffer, self.config.target_lufs)?;
        }

        // Apply compression
        if self.config.compress {
            self.apply_compression(buffer)?;
        }

        // Apply de-essing
        if self.config.deess {
            self.apply_deessing(buffer)?;
        }

        // Measure output loudness
        stats.output_lufs = self.measure_lufs(buffer);

        Ok(stats)
    }

    fn measure_lufs(&self, buffer: &AudioBuffer) -> f32 {
        // Simplified LUFS measurement
        let sum_squares: f32 = buffer.samples.iter().map(|s| s * s).sum();
        let rms = (sum_squares / buffer.samples.len() as f32).sqrt();

        // Convert to LUFS (simplified)
        20.0 * rms.log10() - 0.691
    }

    fn apply_noise_reduction(&self, buffer: &mut AudioBuffer) -> Result<()> {
        // Spectral subtraction noise reduction
        let strength = self.config.noise_reduction_strength;

        // Simple gate-based noise reduction
        let threshold = 0.01 * (1.0 - strength);
        for sample in buffer.samples.iter_mut() {
            if sample.abs() < threshold {
                *sample *= 0.1;
            }
        }

        Ok(())
    }

    fn apply_normalization(&self, buffer: &mut AudioBuffer, target_lufs: f32) -> Result<()> {
        let current_lufs = self.measure_lufs(buffer);
        let gain_db = target_lufs - current_lufs;
        let gain = 10.0_f32.powf(gain_db / 20.0);

        // Limit gain to prevent clipping
        let max_sample = buffer.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let safe_gain = gain.min(0.99 / max_sample.max(0.001));

        for sample in buffer.samples.iter_mut() {
            *sample *= safe_gain;
        }

        Ok(())
    }

    fn apply_compression(&self, buffer: &mut AudioBuffer) -> Result<()> {
        let threshold = 0.5;
        let ratio = self.config.compression_ratio;

        for sample in buffer.samples.iter_mut() {
            let abs = sample.abs();
            if abs > threshold {
                let over = abs - threshold;
                let compressed = threshold + over / ratio;
                *sample = compressed * sample.signum();
            }
        }

        Ok(())
    }

    fn apply_deessing(&self, _buffer: &mut AudioBuffer) -> Result<()> {
        // Simple high-frequency reduction for sibilance
        // Real implementation would use FFT
        let _threshold = self.config.deess_threshold;

        // Apply simple lowpass when signal is high-frequency heavy
        // This is a placeholder - real implementation needs spectral analysis

        Ok(())
    }
}

/// Enhancement statistics
#[derive(Debug, Clone, Default)]
pub struct EnhanceStats {
    /// Input loudness in LUFS
    pub input_lufs: f32,
    /// Output loudness in LUFS
    pub output_lufs: f32,
    /// Noise reduced (dB)
    pub noise_reduced_db: f32,
    /// Dynamic range before
    pub dynamic_range_before: f32,
    /// Dynamic range after
    pub dynamic_range_after: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer() {
        let buffer = AudioBuffer::from_samples(vec![0.0; 44100 * 2], 2, 44100);
        assert_eq!(buffer.num_frames(), 44100);
        assert!((buffer.duration() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_enhancer() {
        let config = AudioEnhanceConfig::default();
        let enhancer = AudioEnhancer::new(config);

        let mut buffer = AudioBuffer::from_samples(
            (0..44100).map(|i| (i as f32 * 0.01).sin() * 0.5).collect(),
            1,
            44100,
        );

        let stats = enhancer.enhance(&mut buffer).unwrap();
        assert!(stats.output_lufs > stats.input_lufs || stats.output_lufs <= -14.0);
    }
}
