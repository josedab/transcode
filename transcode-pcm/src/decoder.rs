//! PCM decoder implementation.

use crate::error::{PcmError, Result};
use crate::format::PcmFormat;
use byteorder::{BigEndian, ByteOrder, LittleEndian};

/// PCM decoder.
#[derive(Debug, Clone)]
pub struct PcmDecoder {
    format: PcmFormat,
    sample_rate: u32,
    channels: u8,
    samples_decoded: u64,
}

impl PcmDecoder {
    /// Create a new PCM decoder.
    pub fn new(format: PcmFormat, sample_rate: u32, channels: u8) -> Self {
        Self {
            format,
            sample_rate,
            channels,
            samples_decoded: 0,
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

    /// Get the number of samples decoded so far.
    pub fn samples_decoded(&self) -> u64 {
        self.samples_decoded
    }

    /// Decode PCM data to f32 samples (normalized to -1.0 to 1.0 range).
    pub fn decode_to_f32(&mut self, data: &[u8]) -> Result<Vec<f32>> {
        let bytes_per_sample = self.format.bytes_per_sample() as usize;
        let total_samples = data.len() / bytes_per_sample;

        if data.len() % bytes_per_sample != 0 {
            return Err(PcmError::BufferSizeMismatch {
                actual: data.len(),
                expected: bytes_per_sample,
            });
        }

        let mut output = Vec::with_capacity(total_samples);

        match self.format {
            PcmFormat::U8 | PcmFormat::U8Planar => {
                for &byte in data {
                    // U8 is 0-255, convert to -1.0 to 1.0
                    output.push((byte as f32 - 128.0) / 128.0);
                }
            }
            PcmFormat::S8 => {
                for &byte in data {
                    output.push(byte as i8 as f32 / 128.0);
                }
            }
            PcmFormat::S16Le | PcmFormat::S16Planar => {
                for chunk in data.chunks_exact(2) {
                    let sample = LittleEndian::read_i16(chunk);
                    output.push(sample as f32 / 32768.0);
                }
            }
            PcmFormat::S16Be => {
                for chunk in data.chunks_exact(2) {
                    let sample = BigEndian::read_i16(chunk);
                    output.push(sample as f32 / 32768.0);
                }
            }
            PcmFormat::S24Le => {
                for chunk in data.chunks_exact(3) {
                    let sample = read_i24_le(chunk);
                    output.push(sample as f32 / 8388608.0);
                }
            }
            PcmFormat::S24Be => {
                for chunk in data.chunks_exact(3) {
                    let sample = read_i24_be(chunk);
                    output.push(sample as f32 / 8388608.0);
                }
            }
            PcmFormat::S32Le | PcmFormat::S32Planar => {
                for chunk in data.chunks_exact(4) {
                    let sample = LittleEndian::read_i32(chunk);
                    output.push(sample as f32 / 2147483648.0);
                }
            }
            PcmFormat::S32Be => {
                for chunk in data.chunks_exact(4) {
                    let sample = BigEndian::read_i32(chunk);
                    output.push(sample as f32 / 2147483648.0);
                }
            }
            PcmFormat::F32Le | PcmFormat::F32Planar => {
                for chunk in data.chunks_exact(4) {
                    output.push(LittleEndian::read_f32(chunk));
                }
            }
            PcmFormat::F32Be => {
                for chunk in data.chunks_exact(4) {
                    output.push(BigEndian::read_f32(chunk));
                }
            }
            PcmFormat::F64Le | PcmFormat::F64Planar => {
                for chunk in data.chunks_exact(8) {
                    output.push(LittleEndian::read_f64(chunk) as f32);
                }
            }
            PcmFormat::F64Be => {
                for chunk in data.chunks_exact(8) {
                    output.push(BigEndian::read_f64(chunk) as f32);
                }
            }
            PcmFormat::S24Le32 => {
                for chunk in data.chunks_exact(4) {
                    let raw = LittleEndian::read_i32(chunk);
                    // Sign-extend from 24-bit
                    let sample = (raw << 8) >> 8;
                    output.push(sample as f32 / 8388608.0);
                }
            }
            PcmFormat::S24Be32 => {
                for chunk in data.chunks_exact(4) {
                    let raw = BigEndian::read_i32(chunk);
                    let sample = (raw << 8) >> 8;
                    output.push(sample as f32 / 8388608.0);
                }
            }
            PcmFormat::S20Le32 => {
                for chunk in data.chunks_exact(4) {
                    let raw = LittleEndian::read_i32(chunk);
                    // Sign-extend from 20-bit
                    let sample = (raw << 12) >> 12;
                    output.push(sample as f32 / 524288.0);
                }
            }
            PcmFormat::S20Be32 => {
                for chunk in data.chunks_exact(4) {
                    let raw = BigEndian::read_i32(chunk);
                    let sample = (raw << 12) >> 12;
                    output.push(sample as f32 / 524288.0);
                }
            }
            PcmFormat::ALaw => {
                for &byte in data {
                    output.push(alaw_to_linear(byte) as f32 / 32768.0);
                }
            }
            PcmFormat::MuLaw => {
                for &byte in data {
                    output.push(mulaw_to_linear(byte) as f32 / 32768.0);
                }
            }
            PcmFormat::DvdLpcm | PcmFormat::BlurayLpcm => {
                // DVD/Blu-ray LPCM is big-endian 24-bit in 32-bit container
                for chunk in data.chunks_exact(4) {
                    let raw = BigEndian::read_i32(chunk);
                    let sample = raw >> 8;
                    output.push(sample as f32 / 8388608.0);
                }
            }
        }

        self.samples_decoded += output.len() as u64;
        Ok(output)
    }

    /// Decode PCM data to i16 samples.
    pub fn decode_to_i16(&mut self, data: &[u8]) -> Result<Vec<i16>> {
        let f32_samples = self.decode_to_f32(data)?;
        Ok(f32_samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect())
    }

    /// Decode PCM data to i32 samples.
    pub fn decode_to_i32(&mut self, data: &[u8]) -> Result<Vec<i32>> {
        let f32_samples = self.decode_to_f32(data)?;
        Ok(f32_samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 2147483647.0) as i32)
            .collect())
    }

    /// Decode planar data to interleaved f32 samples.
    pub fn decode_planar_to_interleaved(&mut self, planes: &[&[u8]]) -> Result<Vec<f32>> {
        if planes.len() != self.channels as usize {
            return Err(PcmError::InvalidChannelCount(planes.len() as u8));
        }

        let bytes_per_sample = self.format.bytes_per_sample() as usize;
        let samples_per_channel = planes[0].len() / bytes_per_sample;

        // Verify all planes have the same size
        for plane in planes.iter() {
            if plane.len() != planes[0].len() {
                return Err(PcmError::InvalidData(
                    "Plane sizes must be equal".to_string(),
                ));
            }
        }

        let mut channel_samples: Vec<Vec<f32>> = Vec::new();

        // Decode each plane
        for plane in planes {
            let mut temp_decoder = PcmDecoder::new(
                self.format,
                self.sample_rate,
                1, // Single channel for decoding
            );
            channel_samples.push(temp_decoder.decode_to_f32(plane)?);
        }

        // Interleave
        let mut output = Vec::with_capacity(samples_per_channel * self.channels as usize);
        for i in 0..samples_per_channel {
            for channel in &channel_samples {
                output.push(channel[i]);
            }
        }

        self.samples_decoded += output.len() as u64;
        Ok(output)
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) {
        self.samples_decoded = 0;
    }
}

/// Read a 24-bit signed integer in little-endian format.
fn read_i24_le(data: &[u8]) -> i32 {
    let val = (data[0] as u32) | ((data[1] as u32) << 8) | ((data[2] as u32) << 16);
    // Sign-extend from 24-bit
    if val & 0x800000 != 0 {
        (val | 0xFF000000) as i32
    } else {
        val as i32
    }
}

/// Read a 24-bit signed integer in big-endian format.
fn read_i24_be(data: &[u8]) -> i32 {
    let val = ((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32);
    // Sign-extend from 24-bit
    if val & 0x800000 != 0 {
        (val | 0xFF000000) as i32
    } else {
        val as i32
    }
}

/// A-law to linear PCM conversion.
fn alaw_to_linear(alaw: u8) -> i16 {
    // A-law decompression
    let a = alaw ^ 0x55;
    let sign = (a & 0x80) != 0;
    let segment = (a >> 4) & 0x07;
    let value = (a & 0x0F) << 1 | 0x21;

    let linear = if segment == 0 {
        (value << 4) as i16
    } else {
        ((value << segment) << 3) as i16
    };

    if sign {
        -linear
    } else {
        linear
    }
}

/// mu-law to linear PCM conversion.
fn mulaw_to_linear(mulaw: u8) -> i16 {
    // mu-law decompression
    static MULAW_TABLE: [i16; 256] = [
        -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
        -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
        -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
        -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
        -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
        -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
        -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
        -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
        -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
        -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
        -876, -844, -812, -780, -748, -716, -684, -652,
        -620, -588, -556, -524, -492, -460, -428, -396,
        -372, -356, -340, -324, -308, -292, -276, -260,
        -244, -228, -212, -196, -180, -164, -148, -132,
        -120, -112, -104, -96, -88, -80, -72, -64,
        -56, -48, -40, -32, -24, -16, -8, 0,
        32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
        23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
        15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
        11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
        7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
        5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
        3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
        2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
        1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
        1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
        876, 844, 812, 780, 748, 716, 684, 652,
        620, 588, 556, 524, 492, 460, 428, 396,
        372, 356, 340, 324, 308, 292, 276, 260,
        244, 228, 212, 196, 180, 164, 148, 132,
        120, 112, 104, 96, 88, 80, 72, 64,
        56, 48, 40, 32, 24, 16, 8, 0,
    ];
    MULAW_TABLE[mulaw as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_s16le() {
        let mut decoder = PcmDecoder::new(PcmFormat::S16Le, 44100, 2);
        let data = [0x00, 0x00, 0xFF, 0x7F, 0x00, 0x80]; // 0, 32767, -32768
        let samples = decoder.decode_to_f32(&data).unwrap();
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.0).abs() < 0.0001);
        assert!((samples[1] - 1.0).abs() < 0.0001);
        assert!((samples[2] + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_decode_u8() {
        let mut decoder = PcmDecoder::new(PcmFormat::U8, 44100, 1);
        let data = [128, 255, 0]; // 0, max, min
        let samples = decoder.decode_to_f32(&data).unwrap();
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.0).abs() < 0.01);
        assert!(samples[1] > 0.9);
        assert!(samples[2] < -0.9);
    }

    #[test]
    fn test_decode_f32le() {
        let mut decoder = PcmDecoder::new(PcmFormat::F32Le, 44100, 1);
        let mut data = [0u8; 12];
        LittleEndian::write_f32(&mut data[0..4], 0.0);
        LittleEndian::write_f32(&mut data[4..8], 1.0);
        LittleEndian::write_f32(&mut data[8..12], -1.0);

        let samples = decoder.decode_to_f32(&data).unwrap();
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.0).abs() < 0.0001);
        assert!((samples[1] - 1.0).abs() < 0.0001);
        assert!((samples[2] + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_buffer_alignment_error() {
        let mut decoder = PcmDecoder::new(PcmFormat::S16Le, 44100, 2);
        let data = [0x00, 0x00, 0xFF]; // 3 bytes, not aligned to 2
        let result = decoder.decode_to_f32(&data);
        assert!(matches!(result, Err(PcmError::BufferSizeMismatch { .. })));
    }

    #[test]
    fn test_24bit_decoding() {
        let mut decoder = PcmDecoder::new(PcmFormat::S24Le, 44100, 1);
        let data = [0x00, 0x00, 0x00, 0xFF, 0xFF, 0x7F]; // 0, 8388607
        let samples = decoder.decode_to_f32(&data).unwrap();
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.0).abs() < 0.0001);
        assert!((samples[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_samples_counter() {
        let mut decoder = PcmDecoder::new(PcmFormat::S16Le, 44100, 2);
        let data = [0x00, 0x00, 0x00, 0x00];
        decoder.decode_to_f32(&data).unwrap();
        assert_eq!(decoder.samples_decoded(), 2);
        decoder.decode_to_f32(&data).unwrap();
        assert_eq!(decoder.samples_decoded(), 4);
        decoder.reset();
        assert_eq!(decoder.samples_decoded(), 0);
    }
}
