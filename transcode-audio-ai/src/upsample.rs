//! Audio upsampling algorithms

use crate::{AudioBuffer, Result};

/// Upsampling quality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsampleQuality {
    /// Fast (linear interpolation)
    Fast,
    /// Medium (windowed sinc)
    Medium,
    /// High (polyphase)
    High,
    /// Best (neural network assisted)
    Best,
}

/// Audio upsampler
pub struct AudioUpsampler {
    quality: UpsampleQuality,
}

impl AudioUpsampler {
    /// Create a new upsampler
    pub fn new(quality: UpsampleQuality) -> Self {
        Self { quality }
    }

    /// Upsample audio to target sample rate
    pub fn upsample(&self, buffer: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        if target_rate <= buffer.sample_rate {
            return Ok(buffer.clone());
        }

        match self.quality {
            UpsampleQuality::Fast => self.linear_upsample(buffer, target_rate),
            UpsampleQuality::Medium => self.sinc_upsample(buffer, target_rate),
            UpsampleQuality::High => self.polyphase_upsample(buffer, target_rate),
            UpsampleQuality::Best => self.polyphase_upsample(buffer, target_rate), // Placeholder
        }
    }

    fn linear_upsample(&self, buffer: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        let ratio = target_rate as f64 / buffer.sample_rate as f64;
        let new_len = (buffer.num_frames() as f64 * ratio) as usize;

        let mut output = AudioBuffer::new(buffer.channels, target_rate);
        output.samples.reserve(new_len * buffer.channels);

        for ch in 0..buffer.channels {
            let channel = buffer.channel(ch);

            for i in 0..new_len {
                let pos = i as f64 / ratio;
                let idx = pos as usize;
                let frac = pos - idx as f64;

                let s0 = channel.get(idx).copied().unwrap_or(0.0);
                let s1 = channel.get(idx + 1).copied().unwrap_or(s0);

                let sample = s0 + (s1 - s0) * frac as f32;
                output.samples.push(sample);
            }
        }

        // Interleave channels
        if buffer.channels > 1 {
            output.samples = self.interleave(&output.samples, buffer.channels, new_len);
        }

        Ok(output)
    }

    fn sinc_upsample(&self, buffer: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        let ratio = target_rate as f64 / buffer.sample_rate as f64;
        let new_len = (buffer.num_frames() as f64 * ratio) as usize;

        let mut output = AudioBuffer::new(buffer.channels, target_rate);
        output.samples.reserve(new_len * buffer.channels);

        let filter_size = 32;

        for ch in 0..buffer.channels {
            let channel = buffer.channel(ch);

            for i in 0..new_len {
                let pos = i as f64 / ratio;
                let idx = pos as usize;
                let frac = pos - idx as f64;

                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for k in -filter_size..=filter_size {
                    let sample_idx = idx as i32 + k;
                    if sample_idx >= 0 && (sample_idx as usize) < channel.len() {
                        let x = k as f64 - frac;
                        let sinc = if x.abs() < 0.0001 {
                            1.0
                        } else {
                            (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                        };
                        let window = 0.5 + 0.5 * (std::f64::consts::PI * k as f64 / filter_size as f64).cos();
                        let weight = sinc * window;

                        sum += channel[sample_idx as usize] as f64 * weight;
                        weight_sum += weight;
                    }
                }

                let sample = if weight_sum > 0.0 {
                    (sum / weight_sum) as f32
                } else {
                    0.0
                };
                output.samples.push(sample);
            }
        }

        if buffer.channels > 1 {
            output.samples = self.interleave(&output.samples, buffer.channels, new_len);
        }

        Ok(output)
    }

    fn polyphase_upsample(&self, buffer: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        // For now, use sinc as polyphase implementation
        self.sinc_upsample(buffer, target_rate)
    }

    fn interleave(&self, data: &[f32], channels: usize, frames: usize) -> Vec<f32> {
        let mut interleaved = vec![0.0; frames * channels];

        for ch in 0..channels {
            for f in 0..frames {
                interleaved[f * channels + ch] = data[ch * frames + f];
            }
        }

        interleaved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_upsample() {
        let upsampler = AudioUpsampler::new(UpsampleQuality::Fast);
        let buffer = AudioBuffer::from_samples(vec![0.0, 1.0, 0.0, -1.0], 1, 44100);

        let result = upsampler.upsample(&buffer, 88200).unwrap();
        assert_eq!(result.sample_rate, 88200);
        assert!(result.num_frames() > buffer.num_frames());
    }
}
