//! Unified resampler interface.
//!
//! Provides a high-level API that wraps the various resampler implementations.

use crate::error::{ResampleError, Result};
use crate::linear::LinearResampler;
use crate::polyphase::PolyphaseResampler;
use crate::sinc::{SincResampler, WindowFunction};
use crate::ResamplerImpl;
use transcode_core::sample::{SampleBuffer, SampleFormat};

/// Type of resampling algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResamplerType {
    /// Linear interpolation - fast, low quality.
    Linear,
    /// Sinc interpolation with configurable window size.
    Sinc {
        /// Number of taps (must be even, typically 32-256).
        window_size: usize,
    },
    /// Polyphase filter bank - efficient for rational rate conversion.
    Polyphase {
        /// Total filter length.
        filter_length: usize,
    },
}

impl Default for ResamplerType {
    fn default() -> Self {
        Self::Sinc { window_size: 64 }
    }
}

/// Configuration for the resampler.
#[derive(Debug, Clone)]
pub struct ResamplerConfig {
    /// Input sample rate in Hz.
    pub input_rate: u32,
    /// Output sample rate in Hz.
    pub output_rate: u32,
    /// Number of channels.
    pub channels: usize,
    /// Resampler type and parameters.
    pub resampler_type: ResamplerType,
    /// Window function for sinc resampler.
    pub window_function: WindowFunction,
}

impl ResamplerConfig {
    /// Create a new configuration.
    pub fn new(input_rate: u32, output_rate: u32) -> Self {
        Self {
            input_rate,
            output_rate,
            channels: 2,
            resampler_type: ResamplerType::default(),
            window_function: WindowFunction::default(),
        }
    }

    /// Set the number of channels.
    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self
    }

    /// Set the resampler type.
    pub fn with_type(mut self, resampler_type: ResamplerType) -> Self {
        self.resampler_type = resampler_type;
        self
    }

    /// Set the window function (only used for sinc resampler).
    pub fn with_window(mut self, window: WindowFunction) -> Self {
        self.window_function = window;
        self
    }

    /// Create a config for fast, low-quality resampling.
    pub fn fast(input_rate: u32, output_rate: u32) -> Self {
        Self::new(input_rate, output_rate).with_type(ResamplerType::Linear)
    }

    /// Create a config for high-quality resampling.
    pub fn high_quality(input_rate: u32, output_rate: u32) -> Self {
        Self::new(input_rate, output_rate)
            .with_type(ResamplerType::Sinc { window_size: 128 })
            .with_window(WindowFunction::Blackman)
    }

    /// Create a config optimized for specific rate conversions.
    pub fn optimized(input_rate: u32, output_rate: u32) -> Self {
        Self::new(input_rate, output_rate).with_type(ResamplerType::Polyphase {
            filter_length: 256,
        })
    }
}

/// Unified resampler that wraps different implementations.
pub struct Resampler {
    inner: Box<dyn ResamplerImpl>,
    channels: usize,
}

impl Resampler {
    /// Create a new resampler from configuration.
    pub fn new(config: ResamplerConfig) -> Result<Self> {
        let inner: Box<dyn ResamplerImpl> = match config.resampler_type {
            ResamplerType::Linear => Box::new(LinearResampler::new(
                config.input_rate,
                config.output_rate,
                config.channels,
            )?),
            ResamplerType::Sinc { window_size } => Box::new(SincResampler::new(
                config.input_rate,
                config.output_rate,
                config.channels,
                window_size,
                config.window_function,
            )?),
            ResamplerType::Polyphase { filter_length } => Box::new(PolyphaseResampler::new(
                config.input_rate,
                config.output_rate,
                config.channels,
                filter_length,
            )?),
        };

        Ok(Self {
            inner,
            channels: config.channels,
        })
    }

    /// Process a single channel of samples.
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.inner.process(input)
    }

    /// Process interleaved multi-channel samples.
    pub fn process_interleaved(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.inner.process_interleaved(input, self.channels)
    }

    /// Process a sample buffer.
    pub fn process_buffer(&mut self, input: &SampleBuffer) -> Result<SampleBuffer> {
        // Convert input to f32
        let input_samples = self.buffer_to_f32(input)?;
        let channels = input.layout.channels() as usize;

        // Process
        let output_samples = self.inner.process_interleaved(&input_samples, channels)?;

        // Calculate output sample count
        let output_num_samples = output_samples.len() / channels;

        // Create output buffer
        let mut output = SampleBuffer::new(
            output_num_samples,
            SampleFormat::F32,
            input.layout,
            self.output_rate(),
        );

        // Copy output data
        let output_data = output.data_mut();
        let output_f32 = unsafe {
            std::slice::from_raw_parts_mut(
                output_data.as_mut_ptr() as *mut f32,
                output_samples.len(),
            )
        };
        output_f32.copy_from_slice(&output_samples);

        Ok(output)
    }

    /// Convert a sample buffer to f32 format.
    fn buffer_to_f32(&self, buffer: &SampleBuffer) -> Result<Vec<f32>> {
        let channels = buffer.layout.channels() as usize;
        let num_samples = buffer.num_samples;
        let total_samples = num_samples * channels;

        match buffer.format {
            SampleFormat::F32 => {
                let data = buffer.data();
                let f32_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, total_samples)
                };
                Ok(f32_data.to_vec())
            }
            SampleFormat::S16 => {
                let data = buffer.data();
                let s16_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i16, total_samples)
                };
                Ok(s16_data.iter().map(|&s| s as f32 / 32768.0).collect())
            }
            SampleFormat::S32 => {
                let data = buffer.data();
                let s32_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i32, total_samples)
                };
                Ok(s32_data
                    .iter()
                    .map(|&s| s as f32 / 2147483648.0)
                    .collect())
            }
            SampleFormat::F64 => {
                let data = buffer.data();
                let f64_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f64, total_samples)
                };
                Ok(f64_data.iter().map(|&s| s as f32).collect())
            }
            _ => Err(ResampleError::internal(format!(
                "Unsupported sample format: {:?}",
                buffer.format
            ))),
        }
    }

    /// Get the input sample rate.
    pub fn input_rate(&self) -> u32 {
        self.inner.input_rate()
    }

    /// Get the output sample rate.
    pub fn output_rate(&self) -> u32 {
        self.inner.output_rate()
    }

    /// Get the resampling ratio.
    pub fn ratio(&self) -> f64 {
        self.inner.ratio()
    }

    /// Reset the resampler state.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get the latency in samples.
    pub fn latency(&self) -> usize {
        self.inner.latency()
    }

    /// Flush remaining samples.
    pub fn flush(&mut self) -> Result<Vec<f32>> {
        self.inner.flush()
    }
}

impl std::fmt::Debug for Resampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resampler")
            .field("input_rate", &self.input_rate())
            .field("output_rate", &self.output_rate())
            .field("channels", &self.channels)
            .field("latency", &self.latency())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use transcode_core::sample::ChannelLayout;

    #[test]
    fn test_resampler_config() {
        let config = ResamplerConfig::new(44100, 48000)
            .with_channels(2)
            .with_type(ResamplerType::Sinc { window_size: 64 })
            .with_window(WindowFunction::Hann);

        assert_eq!(config.input_rate, 44100);
        assert_eq!(config.output_rate, 48000);
        assert_eq!(config.channels, 2);
    }

    #[test]
    fn test_resampler_creation() {
        let config = ResamplerConfig::new(44100, 48000);
        let resampler = Resampler::new(config).unwrap();

        assert_eq!(resampler.input_rate(), 44100);
        assert_eq!(resampler.output_rate(), 48000);
    }

    #[test]
    fn test_linear_resampler_via_unified() {
        let config = ResamplerConfig::fast(44100, 48000);
        let mut resampler = Resampler::new(config).unwrap();

        let input: Vec<f32> = (0..100).map(|i| (i as f32 / 50.0).sin()).collect();
        let output = resampler.process(&input).unwrap();

        assert!(!output.is_empty());
    }

    #[test]
    fn test_sinc_resampler_via_unified() {
        let config = ResamplerConfig::high_quality(44100, 48000);
        let mut resampler = Resampler::new(config).unwrap();

        let input: Vec<f32> = (0..1000).map(|i| (i as f32 / 50.0).sin()).collect();
        let output = resampler.process(&input).unwrap();

        assert!(!output.is_empty());
    }

    #[test]
    fn test_polyphase_resampler_via_unified() {
        let config = ResamplerConfig::optimized(44100, 48000);
        let mut resampler = Resampler::new(config).unwrap();

        let input: Vec<f32> = (0..1000).map(|i| (i as f32 / 50.0).sin()).collect();
        let output = resampler.process(&input).unwrap();

        assert!(!output.is_empty());
    }

    #[test]
    fn test_process_buffer() {
        let config = ResamplerConfig::new(44100, 48000).with_channels(2);
        let mut resampler = Resampler::new(config).unwrap();

        // Create a test buffer
        let mut buffer = SampleBuffer::new(
            1024,
            SampleFormat::F32,
            ChannelLayout::Stereo,
            44100,
        );

        // Fill with test data
        let data = buffer.data_mut();
        let f32_data = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, 2048)
        };
        for (i, sample) in f32_data.iter_mut().enumerate() {
            *sample = ((i as f32) / 100.0).sin();
        }

        let output = resampler.process_buffer(&buffer).unwrap();

        assert!(output.num_samples > 0);
        assert_eq!(output.sample_rate, 48000);
        assert_eq!(output.layout, ChannelLayout::Stereo);
    }

    #[test]
    fn test_common_conversions() {
        // 44.1kHz to 48kHz
        let config = ResamplerConfig::new(44100, 48000);
        assert!(Resampler::new(config).is_ok());

        // 48kHz to 44.1kHz
        let config = ResamplerConfig::new(48000, 44100);
        assert!(Resampler::new(config).is_ok());

        // 96kHz to 48kHz
        let config = ResamplerConfig::new(96000, 48000);
        assert!(Resampler::new(config).is_ok());

        // CD to DVD audio
        let config = ResamplerConfig::new(44100, 96000);
        assert!(Resampler::new(config).is_ok());
    }
}
