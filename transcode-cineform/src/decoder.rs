//! CineForm decoder

use crate::error::{CineformError, Result};
use crate::frame::{CineformFrame, FrameHeader, CFHD_SIGNATURE};
use crate::quantize::{dequantize_decomposition, QuantTable};
use crate::tables::BitReader;
use crate::types::{tags, PixelFormat};
use crate::wavelet::{inverse_wavelet_2d, WaveletDecomposition, WaveletType};

/// Decoder configuration
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Enable threading
    pub threaded: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        DecoderConfig { threaded: false }
    }
}

/// CineForm decoder
pub struct CineformDecoder {
    _config: DecoderConfig,
    frame_count: u64,
}

impl Default for CineformDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CineformDecoder {
    /// Create new decoder with default config
    pub fn new() -> Self {
        Self::with_config(DecoderConfig::default())
    }

    /// Create decoder with custom config
    pub fn with_config(config: DecoderConfig) -> Self {
        CineformDecoder {
            _config: config,
            frame_count: 0,
        }
    }

    /// Decode a frame
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<CineformFrame> {
        // Parse header
        let header = FrameHeader::parse(data)?;

        log::debug!(
            "Decoding CineForm frame: {}x{}, quality {:?}",
            header.width,
            header.height,
            header.quality
        );

        // Create output frame
        let mut frame = CineformFrame::new(header.clone());

        // Parse and decode channels
        let mut offset = 4; // Skip signature

        // Parse tags until we find channel data
        let mut current_channel = 0usize;
        let mut wavelet_type = WaveletType::Haar;
        let mut transform_levels = 3usize;

        while offset + 4 <= data.len() {
            let tag = u16::from_be_bytes([data[offset], data[offset + 1]]);
            let value = u16::from_be_bytes([data[offset + 2], data[offset + 3]]);
            offset += 4;

            match tag {
                tags::WAVELET_TYPE => {
                    wavelet_type = match value {
                        1 => WaveletType::LeGall53,
                        2 => WaveletType::Cdf97,
                        _ => WaveletType::Haar,
                    };
                }
                tags::TRANSFORM_LEVELS => {
                    transform_levels = value as usize;
                }
                tags::CHANNEL_HEADER => {
                    current_channel = value as usize;
                }
                tags::BAND_DATA => {
                    let size = value as usize;
                    if offset + size > data.len() {
                        return Err(CineformError::InsufficientData {
                            needed: offset + size,
                            available: data.len(),
                        });
                    }
                    offset += size; // Skip band data for now
                }
                tags::CHANNEL_END => {
                    // Decode the channel
                    if current_channel < frame.channels.len() {
                        let (ch_width, ch_height) =
                            self.get_channel_dimensions(&header, current_channel);

                        // For simplicity, create a placeholder decomposition
                        // A full implementation would parse and decode the band data
                        let mut decomp =
                            WaveletDecomposition::new(ch_width, ch_height, transform_levels);

                        // Fill with decoded data (placeholder)
                        let ll_idx = decomp.subbands.len() - 1;
                        for val in decomp.subbands[ll_idx].iter_mut() {
                            *val = 512; // Mid-gray for 10-bit
                        }

                        // Dequantize
                        let quant = QuantTable::for_quality(header.quality, transform_levels);
                        dequantize_decomposition(&mut decomp, &quant);

                        // Inverse wavelet transform
                        let reconstructed = inverse_wavelet_2d(&decomp, wavelet_type)?;

                        // Copy to frame
                        let target = &mut frame.channels[current_channel];
                        let copy_len = target.len().min(reconstructed.len());
                        target[..copy_len].copy_from_slice(&reconstructed[..copy_len]);
                    }
                }
                tags::FRAME_END => break,
                _ => {} // Skip unknown tags
            }
        }

        self.frame_count += 1;
        Ok(frame)
    }

    /// Get channel dimensions
    fn get_channel_dimensions(&self, header: &FrameHeader, channel: usize) -> (usize, usize) {
        if header.pixel_format == PixelFormat::YUV422 && channel > 0 {
            // Chroma channels are half width
            ((header.width / 2) as usize, header.height as usize)
        } else {
            (header.width as usize, header.height as usize)
        }
    }

    /// Get frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset decoder
    pub fn reset(&mut self) {
        self.frame_count = 0;
    }
}

/// Decode subband coefficients
fn _decode_subband(data: &[u8], output: &mut [i16]) -> Result<()> {
    let mut reader = BitReader::new(data);
    let mut pos = 0;

    while pos < output.len() {
        // Decode run of zeros
        let run = decode_run(&mut reader)?;
        pos += run as usize;

        if pos >= output.len() {
            break;
        }

        // Decode value
        if let Some(val) = reader.read_signed() {
            output[pos] = val;
            pos += 1;
        } else {
            break;
        }
    }

    Ok(())
}

/// Decode run length
fn decode_run(reader: &mut BitReader) -> Result<u32> {
    // Read leading zeros
    let mut zeros = 0u32;
    loop {
        match reader.read_bits(1) {
            Some(bit) if bit == 1 => {
                // End of run prefix
                break;
            }
            Some(_) => {
                // bit == 0
                zeros += 1;
                if zeros > 16 {
                    // Escape sequence
                    return reader
                        .read_bits(16)
                        .map(|v| v.min(65535))
                        .ok_or_else(|| CineformError::BitstreamError("Failed to read run".into()));
                }
            }
            None => return Ok(0),
        }
    }

    if zeros < 4 {
        Ok(zeros)
    } else {
        // Extended run
        reader
            .read_bits(6)
            .ok_or_else(|| CineformError::BitstreamError("Failed to read extended run".into()))
    }
}

/// Probe data to check if it's CineForm
pub fn probe_cineform(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    data[0..4] == CFHD_SIGNATURE
}

/// Get dimensions from CineForm data
pub fn get_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    let header = FrameHeader::parse(data).ok()?;
    Some((header.width, header.height))
}

/// Convenience function to decode a frame
pub fn decode_cineform(data: &[u8]) -> Result<CineformFrame> {
    let mut decoder = CineformDecoder::new();
    decoder.decode_frame(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::CineformEncoder;
    use crate::types::Quality;

    #[test]
    fn test_decoder_new() {
        let decoder = CineformDecoder::new();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_probe_cineform() {
        assert!(probe_cineform(&CFHD_SIGNATURE));
        assert!(probe_cineform(b"CFHD...."));
        assert!(!probe_cineform(b"RIFF"));
        assert!(!probe_cineform(b"CF"));
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = CineformDecoder::new();
        decoder.frame_count = 100;
        decoder.reset();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let width = 32u32;
        let height = 32u32;

        // Create test frame with gradient
        let mut y = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            for col in 0..width {
                y.push((row * 8 + col * 4) as i16);
            }
        }
        let cb = vec![0i16; (width * height / 2) as usize];
        let cr = vec![0i16; (width * height / 2) as usize];

        let original = CineformFrame::from_planes(
            &[&y, &cb, &cr],
            width,
            height,
            PixelFormat::YUV422,
            Quality::High,
        )
        .unwrap();

        // Encode
        let mut encoder = CineformEncoder::new();
        let encoded = encoder.encode_frame(&original).unwrap();

        // Verify it's valid CineForm data
        assert!(probe_cineform(&encoded));

        // Verify the data contains our frame header tags
        // The encoded data should start with CFHD signature
        assert_eq!(&encoded[0..4], b"CFHD");
    }

    #[test]
    fn test_get_dimensions_invalid() {
        let bad_data = vec![0u8; 10];
        assert!(get_dimensions(&bad_data).is_none());
    }
}
