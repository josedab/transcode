//! PCM encoder implementation.

use crate::error::{PcmError, Result};
use crate::format::PcmFormat;
use byteorder::{BigEndian, ByteOrder, LittleEndian};

/// PCM encoder.
#[derive(Debug, Clone)]
pub struct PcmEncoder {
    format: PcmFormat,
    sample_rate: u32,
    channels: u8,
    samples_encoded: u64,
}

impl PcmEncoder {
    /// Create a new PCM encoder.
    pub fn new(format: PcmFormat, sample_rate: u32, channels: u8) -> Self {
        Self {
            format,
            sample_rate,
            channels,
            samples_encoded: 0,
        }
    }

    /// Get the PCM format.
    pub fn format(&self) -> PcmFormat {
        self.format
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the number of channels.
    pub fn channels(&self) -> u8 {
        self.channels
    }

    /// Get the number of samples encoded so far.
    pub fn samples_encoded(&self) -> u64 {
        self.samples_encoded
    }

    /// Encode f32 samples (normalized -1.0 to 1.0) to PCM data.
    pub fn encode_from_f32(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        let bytes_per_sample = self.format.bytes_per_sample() as usize;
        let mut output = Vec::with_capacity(samples.len() * bytes_per_sample);

        match self.format {
            PcmFormat::U8 | PcmFormat::U8Planar => {
                for &sample in samples {
                    let val = ((sample.clamp(-1.0, 1.0) * 128.0) + 128.0) as u8;
                    output.push(val);
                }
            }
            PcmFormat::S8 => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 127.0) as i8;
                    output.push(val as u8);
                }
            }
            PcmFormat::S16Le | PcmFormat::S16Planar => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    let mut buf = [0u8; 2];
                    LittleEndian::write_i16(&mut buf, val);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::S16Be => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    let mut buf = [0u8; 2];
                    BigEndian::write_i16(&mut buf, val);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::S24Le => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 8388607.0) as i32;
                    output.push((val & 0xFF) as u8);
                    output.push(((val >> 8) & 0xFF) as u8);
                    output.push(((val >> 16) & 0xFF) as u8);
                }
            }
            PcmFormat::S24Be => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 8388607.0) as i32;
                    output.push(((val >> 16) & 0xFF) as u8);
                    output.push(((val >> 8) & 0xFF) as u8);
                    output.push((val & 0xFF) as u8);
                }
            }
            PcmFormat::S32Le | PcmFormat::S32Planar => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 2147483647.0) as i32;
                    let mut buf = [0u8; 4];
                    LittleEndian::write_i32(&mut buf, val);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::S32Be => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 2147483647.0) as i32;
                    let mut buf = [0u8; 4];
                    BigEndian::write_i32(&mut buf, val);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::F32Le | PcmFormat::F32Planar => {
                for &sample in samples {
                    let mut buf = [0u8; 4];
                    LittleEndian::write_f32(&mut buf, sample);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::F32Be => {
                for &sample in samples {
                    let mut buf = [0u8; 4];
                    BigEndian::write_f32(&mut buf, sample);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::F64Le | PcmFormat::F64Planar => {
                for &sample in samples {
                    let mut buf = [0u8; 8];
                    LittleEndian::write_f64(&mut buf, sample as f64);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::F64Be => {
                for &sample in samples {
                    let mut buf = [0u8; 8];
                    BigEndian::write_f64(&mut buf, sample as f64);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::S24Le32 => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 8388607.0) as i32;
                    let mut buf = [0u8; 4];
                    LittleEndian::write_i32(&mut buf, val);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::S24Be32 => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 8388607.0) as i32;
                    let mut buf = [0u8; 4];
                    BigEndian::write_i32(&mut buf, val);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::S20Le32 => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 524287.0) as i32;
                    let mut buf = [0u8; 4];
                    LittleEndian::write_i32(&mut buf, val);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::S20Be32 => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 524287.0) as i32;
                    let mut buf = [0u8; 4];
                    BigEndian::write_i32(&mut buf, val);
                    output.extend_from_slice(&buf);
                }
            }
            PcmFormat::ALaw => {
                for &sample in samples {
                    let linear = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    output.push(linear_to_alaw(linear));
                }
            }
            PcmFormat::MuLaw => {
                for &sample in samples {
                    let linear = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    output.push(linear_to_mulaw(linear));
                }
            }
            PcmFormat::DvdLpcm | PcmFormat::BlurayLpcm => {
                for &sample in samples {
                    let val = (sample.clamp(-1.0, 1.0) * 8388607.0) as i32;
                    let mut buf = [0u8; 4];
                    BigEndian::write_i32(&mut buf, val << 8);
                    output.extend_from_slice(&buf);
                }
            }
        }

        self.samples_encoded += samples.len() as u64;
        Ok(output)
    }

    /// Encode i16 samples to PCM data.
    pub fn encode_from_i16(&mut self, samples: &[i16]) -> Result<Vec<u8>> {
        let f32_samples: Vec<f32> = samples
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();
        self.encode_from_f32(&f32_samples)
    }

    /// Encode i32 samples to PCM data.
    pub fn encode_from_i32(&mut self, samples: &[i32]) -> Result<Vec<u8>> {
        let f32_samples: Vec<f32> = samples
            .iter()
            .map(|&s| s as f32 / 2147483648.0)
            .collect();
        self.encode_from_f32(&f32_samples)
    }

    /// Encode interleaved f32 samples to planar PCM data.
    pub fn encode_interleaved_to_planar(&mut self, samples: &[f32]) -> Result<Vec<Vec<u8>>> {
        if !samples.len().is_multiple_of(self.channels as usize) {
            return Err(PcmError::InvalidData(
                "Sample count must be divisible by channel count".to_string(),
            ));
        }

        let samples_per_channel = samples.len() / self.channels as usize;
        let mut planes = Vec::with_capacity(self.channels as usize);

        for ch in 0..self.channels as usize {
            let channel_samples: Vec<f32> = (0..samples_per_channel)
                .map(|i| samples[i * self.channels as usize + ch])
                .collect();

            let mut temp_encoder = PcmEncoder::new(
                self.format,
                self.sample_rate,
                1, // Single channel for encoding
            );
            planes.push(temp_encoder.encode_from_f32(&channel_samples)?);
        }

        self.samples_encoded += samples.len() as u64;
        Ok(planes)
    }

    /// Reset the encoder state.
    pub fn reset(&mut self) {
        self.samples_encoded = 0;
    }

    /// Calculate the byte size for a given number of samples.
    pub fn bytes_for_samples(&self, sample_count: usize) -> usize {
        sample_count * self.format.bytes_per_sample() as usize
    }

    /// Calculate the number of samples that fit in a given byte count.
    pub fn samples_for_bytes(&self, byte_count: usize) -> usize {
        byte_count / self.format.bytes_per_sample() as usize
    }
}

/// Linear PCM to A-law conversion.
fn linear_to_alaw(linear: i16) -> u8 {
    const ALAW_MAX: i16 = 0x0FFF;

    let sign = if linear >= 0 { 0x80 } else { 0x00 };
    let mut pcm = if linear < 0 { (-linear).saturating_sub(1) } else { linear };

    if pcm > ALAW_MAX {
        pcm = ALAW_MAX;
    }

    let exponent: u8;
    let mantissa: u8;

    if pcm >= 256 {
        let mut exp = 1;
        let mut shifted = pcm >> 8;
        while shifted > 1 && exp < 7 {
            shifted >>= 1;
            exp += 1;
        }
        exponent = exp;
        mantissa = ((pcm >> (exponent + 3)) & 0x0F) as u8;
    } else {
        exponent = 0;
        mantissa = (pcm >> 4) as u8;
    }

    let alaw = sign | (exponent << 4) | mantissa;
    alaw ^ 0x55
}

/// Linear PCM to mu-law conversion.
fn linear_to_mulaw(linear: i16) -> u8 {
    const MULAW_MAX: i16 = 0x1FFF;
    const MULAW_BIAS: i16 = 132; // 0x84

    let sign = if linear >= 0 { 0xFF } else { 0x7F };
    let mut pcm = if linear < 0 { -linear } else { linear };

    pcm = (pcm + MULAW_BIAS).min(MULAW_MAX);

    let exponent = {
        let mut exp = 0;
        let mut shifted = pcm >> 7;
        while shifted > 0 && exp < 7 {
            shifted >>= 1;
            exp += 1;
        }
        exp
    };

    let mantissa = (pcm >> (exponent + 3)) & 0x0F;
    !(sign & (0x80 | (exponent << 4) | mantissa as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_s16le() {
        let mut encoder = PcmEncoder::new(PcmFormat::S16Le, 44100, 2);
        let samples = [0.0f32, 1.0, -1.0];
        let data = encoder.encode_from_f32(&samples).unwrap();
        assert_eq!(data.len(), 6);

        // Check zero
        assert_eq!(LittleEndian::read_i16(&data[0..2]), 0);
        // Check max positive (clamped to 32767)
        assert_eq!(LittleEndian::read_i16(&data[2..4]), 32767);
        // Check max negative
        assert_eq!(LittleEndian::read_i16(&data[4..6]), -32767);
    }

    #[test]
    fn test_encode_u8() {
        let mut encoder = PcmEncoder::new(PcmFormat::U8, 44100, 1);
        let samples = [0.0f32, 1.0, -1.0];
        let data = encoder.encode_from_f32(&samples).unwrap();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0], 128); // 0.0 -> 128
        assert_eq!(data[1], 255); // 1.0 -> 255 (clamped)
        assert_eq!(data[2], 0);   // -1.0 -> 0
    }

    #[test]
    fn test_encode_f32le() {
        let mut encoder = PcmEncoder::new(PcmFormat::F32Le, 44100, 1);
        let samples = [0.0f32, 0.5, -0.5];
        let data = encoder.encode_from_f32(&samples).unwrap();
        assert_eq!(data.len(), 12);

        assert!((LittleEndian::read_f32(&data[0..4]) - 0.0).abs() < 0.0001);
        assert!((LittleEndian::read_f32(&data[4..8]) - 0.5).abs() < 0.0001);
        assert!((LittleEndian::read_f32(&data[8..12]) + 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_roundtrip_s16le() {
        use crate::PcmDecoder;

        let original = [0.0f32, 0.5, -0.5, 0.25, -0.25];

        let mut encoder = PcmEncoder::new(PcmFormat::S16Le, 44100, 2);
        let encoded = encoder.encode_from_f32(&original).unwrap();

        let mut decoder = PcmDecoder::new(PcmFormat::S16Le, 44100, 2);
        let decoded = decoder.decode_to_f32(&encoded).unwrap();

        for (orig, dec) in original.iter().zip(decoded.iter()) {
            // Allow for quantization error in 16-bit
            assert!((orig - dec).abs() < 0.001);
        }
    }

    #[test]
    fn test_samples_counter() {
        let mut encoder = PcmEncoder::new(PcmFormat::S16Le, 44100, 2);
        let samples = [0.0f32, 0.0];
        encoder.encode_from_f32(&samples).unwrap();
        assert_eq!(encoder.samples_encoded(), 2);
        encoder.encode_from_f32(&samples).unwrap();
        assert_eq!(encoder.samples_encoded(), 4);
        encoder.reset();
        assert_eq!(encoder.samples_encoded(), 0);
    }

    #[test]
    fn test_bytes_for_samples() {
        let encoder = PcmEncoder::new(PcmFormat::S16Le, 44100, 2);
        assert_eq!(encoder.bytes_for_samples(100), 200);

        let encoder = PcmEncoder::new(PcmFormat::F32Le, 48000, 1);
        assert_eq!(encoder.bytes_for_samples(100), 400);

        let encoder = PcmEncoder::new(PcmFormat::S24Le, 48000, 2);
        assert_eq!(encoder.bytes_for_samples(100), 300);
    }

    #[test]
    fn test_encode_from_i16() {
        let mut encoder = PcmEncoder::new(PcmFormat::F32Le, 44100, 1);
        let samples = [0i16, 16383, -16384];
        let data = encoder.encode_from_i16(&samples).unwrap();

        // Check approximate values
        let f0 = LittleEndian::read_f32(&data[0..4]);
        let f1 = LittleEndian::read_f32(&data[4..8]);
        let f2 = LittleEndian::read_f32(&data[8..12]);

        assert!(f0.abs() < 0.001);
        assert!((f1 - 0.5).abs() < 0.001);
        assert!((f2 + 0.5).abs() < 0.001);
    }
}
