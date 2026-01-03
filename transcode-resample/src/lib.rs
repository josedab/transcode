//! # Transcode Resample
//!
//! High-quality audio resampling with SIMD optimization.
//!
//! This crate provides multiple resampling algorithms:
//! - **Linear interpolation**: Fast, low quality (suitable for previews)
//! - **Sinc interpolation**: High quality with configurable window functions
//! - **Polyphase filter bank**: Efficient implementation for fixed rate conversions
//!
//! ## Features
//!
//! - Arbitrary sample rate conversion (e.g., 44100 -> 48000 Hz)
//! - Multi-channel support
//! - SIMD optimization for x86_64 (AVX2) and aarch64 (NEON)
//! - Support for planar and interleaved audio formats
//!
//! ## Example
//!
//! ```ignore
//! use transcode_resample::{Resampler, ResamplerConfig, ResamplerType};
//! use transcode_core::sample::SampleBuffer;
//!
//! // Create a high-quality sinc resampler
//! let config = ResamplerConfig::new(44100, 48000)
//!     .with_type(ResamplerType::Sinc { window_size: 64 })
//!     .with_channels(2);
//!
//! let mut resampler = Resampler::new(config)?;
//!
//! // Process audio
//! let output = resampler.process(&input_buffer)?;
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod error;
pub mod linear;
pub mod sinc;
pub mod polyphase;
pub mod simd;
mod resampler;

pub use error::{ResampleError, Result};
pub use linear::LinearResampler;
pub use sinc::{SincResampler, WindowFunction};
pub use polyphase::PolyphaseResampler;
pub use resampler::{Resampler, ResamplerConfig, ResamplerType};

/// Trait for all resampling implementations.
pub trait ResamplerImpl: Send + Sync {
    /// Process input samples and return resampled output.
    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>>;

    /// Process multi-channel input samples.
    fn process_interleaved(&mut self, input: &[f32], channels: usize) -> Result<Vec<f32>>;

    /// Get the input sample rate.
    fn input_rate(&self) -> u32;

    /// Get the output sample rate.
    fn output_rate(&self) -> u32;

    /// Get the resampling ratio (output_rate / input_rate).
    fn ratio(&self) -> f64 {
        self.output_rate() as f64 / self.input_rate() as f64
    }

    /// Reset the resampler state (clear internal buffers).
    fn reset(&mut self);

    /// Get the latency in samples.
    fn latency(&self) -> usize;

    /// Flush any remaining samples.
    fn flush(&mut self) -> Result<Vec<f32>>;
}

/// Calculate the greatest common divisor.
pub(crate) fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Calculate the least common multiple.
#[allow(dead_code)]
pub(crate) fn lcm(a: u32, b: u32) -> u32 {
    (a / gcd(a, b)) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(44100, 48000), 300);
        assert_eq!(gcd(48000, 96000), 48000);
        assert_eq!(gcd(44100, 22050), 22050);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(44100, 48000), 7056000);
    }
}
