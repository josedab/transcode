//! CineForm encoder

use crate::error::Result;
use crate::frame::{CineformFrame, CFHD_SIGNATURE};
use crate::quantize::{quantize_decomposition, QuantTable};
use crate::tables::BitWriter;
use crate::types::{tags, BitDepth, PixelFormat, Quality};
use crate::wavelet::{forward_wavelet_2d, WaveletDecomposition, WaveletType};

/// Encoder configuration
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Quality level
    pub quality: Quality,
    /// Wavelet type
    pub wavelet_type: WaveletType,
    /// Number of transform levels
    pub transform_levels: usize,
    /// Bit depth
    pub bit_depth: BitDepth,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig {
            quality: Quality::High,
            wavelet_type: WaveletType::Haar,
            transform_levels: 3,
            bit_depth: BitDepth::Bit10,
        }
    }
}

impl EncoderConfig {
    /// Create config for specific quality
    pub fn for_quality(quality: Quality) -> Self {
        EncoderConfig {
            quality,
            ..Default::default()
        }
    }
}

/// CineForm encoder
pub struct CineformEncoder {
    config: EncoderConfig,
    frame_count: u64,
}

impl Default for CineformEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CineformEncoder {
    /// Create new encoder with default config
    pub fn new() -> Self {
        Self::with_config(EncoderConfig::default())
    }

    /// Create encoder with custom config
    pub fn with_config(config: EncoderConfig) -> Self {
        CineformEncoder {
            config,
            frame_count: 0,
        }
    }

    /// Encode a frame
    pub fn encode_frame(&mut self, frame: &CineformFrame) -> Result<Vec<u8>> {
        let mut output = Vec::new();

        // Write signature
        output.extend_from_slice(&CFHD_SIGNATURE);

        // Write header tags
        self.write_header_tags(&mut output, frame)?;

        // Encode each channel
        for (channel_idx, channel_data) in frame.channels.iter().enumerate() {
            // Get channel dimensions
            let (ch_width, ch_height) = if frame.pixel_format == PixelFormat::YUV422
                && channel_idx > 0
            {
                // Chroma channels are half width for YUV422
                ((frame.width / 2) as usize, frame.height as usize)
            } else {
                (frame.width as usize, frame.height as usize)
            };

            // Forward wavelet transform
            let mut decomp = forward_wavelet_2d(
                channel_data,
                ch_width,
                ch_height,
                self.config.transform_levels,
                self.config.wavelet_type,
            )?;

            // Quantize
            let quant = QuantTable::for_quality(self.config.quality, self.config.transform_levels);
            quantize_decomposition(&mut decomp, &quant);

            // Encode subbands
            self.encode_channel(&mut output, &decomp, channel_idx)?;
        }

        // Write frame end tag
        self.write_tag(&mut output, tags::FRAME_END, 0);

        self.frame_count += 1;
        Ok(output)
    }

    /// Write header tags
    fn write_header_tags(&self, output: &mut Vec<u8>, frame: &CineformFrame) -> Result<()> {
        // Frame header start
        self.write_tag(output, tags::FRAME_HEADER, 0);

        // Image dimensions
        self.write_tag(output, tags::IMAGE_WIDTH, frame.width as u16);
        self.write_tag(output, tags::IMAGE_HEIGHT, frame.height as u16);

        // Pixel format
        let pixel_format_val = match frame.pixel_format {
            PixelFormat::YUV422 => 0,
            PixelFormat::YUV444 => 1,
            PixelFormat::RGBA => 2,
            PixelFormat::BGRA => 3,
            PixelFormat::Bayer => 4,
        };
        self.write_tag(output, tags::PIXEL_FORMAT, pixel_format_val);

        // Bit depth
        self.write_tag(output, tags::BITS_PER_COMPONENT, self.config.bit_depth.bits() as u16);

        // Channel count
        self.write_tag(output, tags::CHANNEL_COUNT, frame.channels.len() as u16);

        // Quality level
        self.write_tag(output, tags::QUALITY_LEVEL, self.config.quality.to_level() as u16);

        // Transform info
        self.write_tag(output, tags::TRANSFORM_LEVELS, self.config.transform_levels as u16);
        self.write_tag(output, tags::WAVELET_TYPE, self.config.wavelet_type as u16);

        // Frame index
        self.write_tag(output, tags::FRAME_INDEX, self.frame_count as u16);

        Ok(())
    }

    /// Write a tag-value pair
    fn write_tag(&self, output: &mut Vec<u8>, tag: u16, value: u16) {
        output.extend_from_slice(&tag.to_be_bytes());
        output.extend_from_slice(&value.to_be_bytes());
    }

    /// Encode a channel
    fn encode_channel(
        &self,
        output: &mut Vec<u8>,
        decomp: &WaveletDecomposition,
        channel_idx: usize,
    ) -> Result<()> {
        // Channel header
        self.write_tag(output, tags::CHANNEL_HEADER, channel_idx as u16);

        // Encode each subband
        for (subband_idx, subband) in decomp.subbands.iter().enumerate() {
            // Band header
            self.write_tag(output, tags::BAND_HEADER, subband_idx as u16);

            // Encode band data
            let encoded = self.encode_subband(subband)?;

            // Band data tag with size
            self.write_tag(output, tags::BAND_DATA, encoded.len() as u16);
            output.extend_from_slice(&encoded);

            // Band end
            self.write_tag(output, tags::BAND_END, 0);
        }

        // Channel end
        self.write_tag(output, tags::CHANNEL_END, 0);

        Ok(())
    }

    /// Encode a subband using entropy coding
    fn encode_subband(&self, data: &[i16]) -> Result<Vec<u8>> {
        let mut writer = BitWriter::new();

        // Run-length encode zeros and encode non-zero values
        let mut run = 0u32;

        for &val in data {
            if val == 0 {
                run += 1;
            } else {
                // Encode run of zeros
                self.encode_run(&mut writer, run);
                run = 0;

                // Encode value
                writer.write_signed(val);
            }
        }

        // Encode final run if any
        if run > 0 {
            self.encode_run(&mut writer, run);
        }

        Ok(writer.into_bytes())
    }

    /// Encode a run of zeros
    fn encode_run(&self, writer: &mut BitWriter, run: u32) {
        if run == 0 {
            // No zeros before this value
            writer.write_bits(1, 1); // Single 1 bit
        } else if run < 4 {
            // Short run: unary code
            writer.write_bits(0, run as u8);
            writer.write_bits(1, 1);
        } else if run < 64 {
            // Medium run: escape + 6-bit value
            writer.write_bits(0, 4);
            writer.write_bits(run, 6);
        } else {
            // Long run: double escape + 16-bit value
            writer.write_bits(0, 8);
            writer.write_bits(run.min(65535), 16);
        }
    }

    /// Get frame count
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Reset encoder
    pub fn reset(&mut self) {
        self.frame_count = 0;
    }

    /// Get configuration
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

/// Convenience function to encode a frame
pub fn encode_cineform(frame: &CineformFrame, quality: Quality) -> Result<Vec<u8>> {
    let config = EncoderConfig::for_quality(quality);
    let mut encoder = CineformEncoder::with_config(config);
    encoder.encode_frame(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_new() {
        let encoder = CineformEncoder::new();
        assert_eq!(encoder.frame_count(), 0);
    }

    #[test]
    fn test_encoder_config() {
        let config = EncoderConfig::for_quality(Quality::FilmScan2);
        assert_eq!(config.quality, Quality::FilmScan2);
    }

    #[test]
    fn test_encode_small_frame() {
        let width = 32u32;
        let height = 32u32;

        let y = vec![128i16; (width * height) as usize];
        let cb = vec![0i16; (width * height / 2) as usize];
        let cr = vec![0i16; (width * height / 2) as usize];

        let frame = CineformFrame::from_planes(
            &[&y, &cb, &cr],
            width,
            height,
            PixelFormat::YUV422,
            Quality::High,
        )
        .unwrap();

        let mut encoder = CineformEncoder::new();
        let result = encoder.encode_frame(&frame);

        assert!(result.is_ok());
        let data = result.unwrap();

        // Check signature
        assert_eq!(&data[0..4], &CFHD_SIGNATURE);
    }

    #[test]
    fn test_encode_gradient() {
        let width = 16u32;
        let height = 16u32;

        // Gradient pattern
        let mut y = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            for col in 0..width {
                y.push((row * 16 + col * 8) as i16);
            }
        }

        let cb = vec![0i16; (width * height / 2) as usize];
        let cr = vec![0i16; (width * height / 2) as usize];

        let frame = CineformFrame::from_planes(
            &[&y, &cb, &cr],
            width,
            height,
            PixelFormat::YUV422,
            Quality::High,
        )
        .unwrap();

        let result = encode_cineform(&frame, Quality::High);
        assert!(result.is_ok());
    }

    #[test]
    fn test_all_qualities() {
        let qualities = [
            Quality::Low,
            Quality::Medium,
            Quality::High,
            Quality::FilmScan1,
            Quality::FilmScan2,
        ];

        for quality in qualities {
            let width = 16u32;
            let height = 16u32;
            let y = vec![64i16; (width * height) as usize];
            let cb = vec![0i16; (width * height / 2) as usize];
            let cr = vec![0i16; (width * height / 2) as usize];

            let frame = CineformFrame::from_planes(
                &[&y, &cb, &cr],
                width,
                height,
                PixelFormat::YUV422,
                quality,
            )
            .unwrap();

            let result = encode_cineform(&frame, quality);
            assert!(result.is_ok(), "Failed for quality {:?}", quality);
        }
    }

    #[test]
    fn test_encoder_reset() {
        let mut encoder = CineformEncoder::new();
        encoder.frame_count = 100;
        encoder.reset();
        assert_eq!(encoder.frame_count(), 0);
    }
}
