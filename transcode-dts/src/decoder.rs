//! DTS and TrueHD audio decoders.
//!
//! Provides audio decoding for DTS (including DTS-HD) and Dolby TrueHD.
//!
//! ## Usage
//!
//! Full decoding requires the `ffi-ffmpeg` feature. Without it, only parsing
//! and metadata extraction is available.

use std::fmt;

#[cfg(feature = "ffi-ffmpeg")]
use crate::ffi::{DtsFfiDecoder, TrueHdFfiDecoder};
use crate::parser::{DtsParser, TrueHdParser};
use crate::types::*;
use crate::{DtsError, Result};

/// DTS audio decoder.
///
/// Decodes DTS core, DTS-HD High Resolution, and DTS-HD Master Audio frames.
///
/// # Example
///
/// ```rust,ignore
/// use transcode_dts::DtsDecoder;
///
/// let mut decoder = DtsDecoder::new()?;
/// let decoded = decoder.decode_frame(&dts_frame)?;
/// println!("Decoded {} samples", decoded.samples_per_channel);
/// ```
pub struct DtsDecoder {
    /// Parser for extracting frame info.
    parser: DtsParser,
    /// Output sample rate (for resampling).
    output_sample_rate: Option<u32>,
    /// Output bit depth.
    output_bit_depth: u8,
    /// Total frames decoded.
    frames_decoded: u64,
    /// Total samples decoded.
    samples_decoded: u64,
    /// Decode DTS-HD extensions.
    decode_hd: bool,
    /// Downmix to stereo.
    downmix_stereo: bool,
    /// FFI decoder (when ffi-ffmpeg feature is enabled).
    #[cfg(feature = "ffi-ffmpeg")]
    ffi_decoder: Option<DtsFfiDecoder>,
}

impl fmt::Debug for DtsDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DtsDecoder")
            .field("output_sample_rate", &self.output_sample_rate)
            .field("output_bit_depth", &self.output_bit_depth)
            .field("frames_decoded", &self.frames_decoded)
            .field("samples_decoded", &self.samples_decoded)
            .field("decode_hd", &self.decode_hd)
            .field("downmix_stereo", &self.downmix_stereo)
            .finish_non_exhaustive()
    }
}

impl DtsDecoder {
    /// Create a new DTS decoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn new() -> Result<Self> {
        let ffi_decoder = DtsFfiDecoder::new().ok();
        Ok(Self {
            parser: DtsParser::new(),
            output_sample_rate: None,
            output_bit_depth: 24,
            frames_decoded: 0,
            samples_decoded: 0,
            decode_hd: true,
            downmix_stereo: false,
            ffi_decoder,
        })
    }

    /// Create a new DTS decoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn new() -> Result<Self> {
        Ok(Self {
            parser: DtsParser::new(),
            output_sample_rate: None,
            output_bit_depth: 24,
            frames_decoded: 0,
            samples_decoded: 0,
            decode_hd: true,
            downmix_stereo: false,
        })
    }

    /// Create decoder with specific output format.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn with_format(sample_rate: u32, bit_depth: u8) -> Result<Self> {
        let ffi_decoder = DtsFfiDecoder::new().ok();
        Ok(Self {
            parser: DtsParser::new(),
            output_sample_rate: Some(sample_rate),
            output_bit_depth: bit_depth,
            frames_decoded: 0,
            samples_decoded: 0,
            decode_hd: true,
            downmix_stereo: false,
            ffi_decoder,
        })
    }

    /// Create decoder with specific output format.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn with_format(sample_rate: u32, bit_depth: u8) -> Result<Self> {
        Ok(Self {
            parser: DtsParser::new(),
            output_sample_rate: Some(sample_rate),
            output_bit_depth: bit_depth,
            frames_decoded: 0,
            samples_decoded: 0,
            decode_hd: true,
            downmix_stereo: false,
        })
    }

    /// Set whether to decode DTS-HD extensions.
    pub fn set_decode_hd(&mut self, decode_hd: bool) {
        self.decode_hd = decode_hd;
    }

    /// Set whether to downmix to stereo.
    pub fn set_downmix_stereo(&mut self, downmix: bool) {
        self.downmix_stereo = downmix;
    }

    /// Decode a DTS frame.
    ///
    /// Returns decoded PCM audio or an error if decoding fails.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        if let Some(ref mut ffi) = self.ffi_decoder {
            let decoded = ffi.decode(data)?;
            self.frames_decoded += 1;
            self.samples_decoded += decoded.samples_per_channel as u64;
            Ok(decoded)
        } else {
            let frame = self.parse_frame(data)?;
            let layout = ChannelLayout::from_audio_mode(frame.amode, frame.lfe);

            Ok(DecodedAudio {
                sample_rate: frame.sample_rate,
                channels: frame.total_channels(),
                layout,
                samples: Vec::new(),
                samples_per_channel: 0,
                bit_depth: frame.pcm_resolution,
            })
        }
    }

    /// Decode a DTS frame (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        // Without FFI, we can only parse the frame
        let frame = self.parse_frame(data)?;

        // Create empty decoded audio with correct metadata
        let layout = ChannelLayout::from_audio_mode(frame.amode, frame.lfe);

        Ok(DecodedAudio {
            sample_rate: frame.sample_rate,
            channels: frame.total_channels(),
            layout,
            samples: Vec::new(), // No actual decoding without FFI
            samples_per_channel: 0,
            bit_depth: frame.pcm_resolution,
        })
    }

    /// Parse a frame without decoding.
    pub fn parse_frame(&mut self, data: &[u8]) -> Result<DtsSyncFrame> {
        self.parser.reset();
        self.parser
            .parse_sync_frame(data)
            .ok_or_else(|| DtsError::BitstreamCorruption("Could not parse sync frame".into()))
    }

    /// Get frame information without decoding.
    pub fn probe_frame(&self, data: &[u8]) -> Result<DtsSyncFrame> {
        let mut parser = DtsParser::new();
        parser
            .parse_sync_frame(data)
            .ok_or_else(|| DtsError::BitstreamCorruption("Could not parse sync frame".into()))
    }

    /// Check if the frame contains DTS-HD extension.
    pub fn has_hd_extension(&self, data: &[u8]) -> bool {
        if data.len() < 8 {
            return false;
        }

        // Check for DTS-HD sync word following core frame
        let sync = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        sync == crate::DTS_HD_SYNC
    }

    /// Flush the decoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn flush(&mut self) {
        self.parser.reset();
        if let Some(ref mut ffi) = self.ffi_decoder {
            ffi.flush();
        }
    }

    /// Flush the decoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn flush(&mut self) {
        self.parser.reset();
    }

    /// Reset the decoder state.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn reset(&mut self) {
        self.parser.reset();
        self.frames_decoded = 0;
        self.samples_decoded = 0;
        if let Some(ref mut ffi) = self.ffi_decoder {
            ffi.flush();
        }
    }

    /// Reset the decoder state.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn reset(&mut self) {
        self.parser.reset();
        self.frames_decoded = 0;
        self.samples_decoded = 0;
    }

    /// Get total frames decoded.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Get total samples decoded.
    pub fn samples_decoded(&self) -> u64 {
        self.samples_decoded
    }

    /// Get the output sample rate.
    pub fn output_sample_rate(&self) -> Option<u32> {
        self.output_sample_rate
    }

    /// Get the output bit depth.
    pub fn output_bit_depth(&self) -> u8 {
        self.output_bit_depth
    }

    /// Check if FFI decoding is available.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn is_decoding_available(&self) -> bool {
        self.ffi_decoder.is_some()
    }

    /// Check if FFI decoding is available.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn is_decoding_available(&self) -> bool {
        false
    }

    /// Check if HD decoding is enabled.
    pub fn is_hd_enabled(&self) -> bool {
        self.decode_hd
    }
}

impl Default for DtsDecoder {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// TrueHD audio decoder.
///
/// Decodes Dolby TrueHD (MLP) audio frames to PCM samples.
///
/// # Example
///
/// ```rust,ignore
/// use transcode_dts::TrueHdDecoder;
///
/// let mut decoder = TrueHdDecoder::new()?;
/// let decoded = decoder.decode_frame(&truehd_frame)?;
/// println!("Decoded {} samples at {} Hz", decoded.samples_per_channel, decoded.sample_rate);
/// ```
pub struct TrueHdDecoder {
    /// Parser for extracting frame info.
    parser: TrueHdParser,
    /// Output sample rate (for resampling).
    output_sample_rate: Option<u32>,
    /// Output bit depth.
    output_bit_depth: u8,
    /// Total frames decoded.
    frames_decoded: u64,
    /// Total samples decoded.
    samples_decoded: u64,
    /// Downmix to stereo.
    downmix_stereo: bool,
    /// Decode Atmos metadata.
    decode_atmos: bool,
    /// FFI decoder (when ffi-ffmpeg feature is enabled).
    #[cfg(feature = "ffi-ffmpeg")]
    ffi_decoder: Option<TrueHdFfiDecoder>,
}

impl fmt::Debug for TrueHdDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrueHdDecoder")
            .field("output_sample_rate", &self.output_sample_rate)
            .field("output_bit_depth", &self.output_bit_depth)
            .field("frames_decoded", &self.frames_decoded)
            .field("samples_decoded", &self.samples_decoded)
            .field("downmix_stereo", &self.downmix_stereo)
            .field("decode_atmos", &self.decode_atmos)
            .finish_non_exhaustive()
    }
}

impl TrueHdDecoder {
    /// Create a new TrueHD decoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn new() -> Result<Self> {
        let ffi_decoder = TrueHdFfiDecoder::new().ok();
        Ok(Self {
            parser: TrueHdParser::new(),
            output_sample_rate: None,
            output_bit_depth: 24,
            frames_decoded: 0,
            samples_decoded: 0,
            downmix_stereo: false,
            decode_atmos: true,
            ffi_decoder,
        })
    }

    /// Create a new TrueHD decoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn new() -> Result<Self> {
        Ok(Self {
            parser: TrueHdParser::new(),
            output_sample_rate: None,
            output_bit_depth: 24,
            frames_decoded: 0,
            samples_decoded: 0,
            downmix_stereo: false,
            decode_atmos: true,
        })
    }

    /// Create decoder with specific output format.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn with_format(sample_rate: u32, bit_depth: u8) -> Result<Self> {
        let ffi_decoder = TrueHdFfiDecoder::new().ok();
        Ok(Self {
            parser: TrueHdParser::new(),
            output_sample_rate: Some(sample_rate),
            output_bit_depth: bit_depth,
            frames_decoded: 0,
            samples_decoded: 0,
            downmix_stereo: false,
            decode_atmos: true,
            ffi_decoder,
        })
    }

    /// Create decoder with specific output format.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn with_format(sample_rate: u32, bit_depth: u8) -> Result<Self> {
        Ok(Self {
            parser: TrueHdParser::new(),
            output_sample_rate: Some(sample_rate),
            output_bit_depth: bit_depth,
            frames_decoded: 0,
            samples_decoded: 0,
            downmix_stereo: false,
            decode_atmos: true,
        })
    }

    /// Set whether to downmix to stereo.
    pub fn set_downmix_stereo(&mut self, downmix: bool) {
        self.downmix_stereo = downmix;
    }

    /// Set whether to decode Atmos metadata.
    pub fn set_decode_atmos(&mut self, decode_atmos: bool) {
        self.decode_atmos = decode_atmos;
    }

    /// Decode a TrueHD frame.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        if let Some(ref mut ffi) = self.ffi_decoder {
            let decoded = ffi.decode(data)?;
            self.frames_decoded += 1;
            self.samples_decoded += decoded.samples_per_channel as u64;
            Ok(decoded)
        } else {
            let frame = self.parse_frame(data)?;
            let layout = match frame.channel_assignment {
                TrueHdChannelAssignment::Mono => ChannelLayout::mono(),
                TrueHdChannelAssignment::Stereo => ChannelLayout::stereo(),
                TrueHdChannelAssignment::Layout5_1 => ChannelLayout::surround_5_1(),
                TrueHdChannelAssignment::Layout7_1 | TrueHdChannelAssignment::Atmos => {
                    ChannelLayout::surround_7_1()
                }
                _ => ChannelLayout::stereo(),
            };

            Ok(DecodedAudio {
                sample_rate: frame.sample_rate,
                channels: frame.channels,
                layout,
                samples: Vec::new(),
                samples_per_channel: 0,
                bit_depth: frame.bit_depth,
            })
        }
    }

    /// Decode a TrueHD frame (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        let frame = self.parse_frame(data)?;

        // Create channel layout based on channel assignment
        let layout = match frame.channel_assignment {
            TrueHdChannelAssignment::Mono => ChannelLayout::mono(),
            TrueHdChannelAssignment::Stereo => ChannelLayout::stereo(),
            TrueHdChannelAssignment::Layout5_1 => ChannelLayout::surround_5_1(),
            TrueHdChannelAssignment::Layout7_1 | TrueHdChannelAssignment::Atmos => {
                ChannelLayout::surround_7_1()
            }
            _ => ChannelLayout::stereo(),
        };

        Ok(DecodedAudio {
            sample_rate: frame.sample_rate,
            channels: frame.channels,
            layout,
            samples: Vec::new(),
            samples_per_channel: 0,
            bit_depth: frame.bit_depth,
        })
    }

    /// Parse a frame without decoding.
    pub fn parse_frame(&mut self, data: &[u8]) -> Result<TrueHdSyncFrame> {
        self.parser.reset();
        self.parser
            .parse_sync_frame(data)
            .ok_or_else(|| DtsError::BitstreamCorruption("Could not parse sync frame".into()))
    }

    /// Get frame information without decoding.
    pub fn probe_frame(&self, data: &[u8]) -> Result<TrueHdSyncFrame> {
        let mut parser = TrueHdParser::new();
        parser
            .parse_sync_frame(data)
            .ok_or_else(|| DtsError::BitstreamCorruption("Could not parse sync frame".into()))
    }

    /// Check if stream appears to be Dolby Atmos.
    pub fn is_atmos_stream(&self, data: &[u8]) -> Result<bool> {
        let frame = self.probe_frame(data)?;
        Ok(frame.is_atmos())
    }

    /// Flush the decoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn flush(&mut self) {
        self.parser.reset();
        if let Some(ref mut ffi) = self.ffi_decoder {
            ffi.flush();
        }
    }

    /// Flush the decoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn flush(&mut self) {
        self.parser.reset();
    }

    /// Reset the decoder state.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn reset(&mut self) {
        self.parser.reset();
        self.frames_decoded = 0;
        self.samples_decoded = 0;
        if let Some(ref mut ffi) = self.ffi_decoder {
            ffi.flush();
        }
    }

    /// Reset the decoder state.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn reset(&mut self) {
        self.parser.reset();
        self.frames_decoded = 0;
        self.samples_decoded = 0;
    }

    /// Get total frames decoded.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Get total samples decoded.
    pub fn samples_decoded(&self) -> u64 {
        self.samples_decoded
    }

    /// Get the output sample rate.
    pub fn output_sample_rate(&self) -> Option<u32> {
        self.output_sample_rate
    }

    /// Get the output bit depth.
    pub fn output_bit_depth(&self) -> u8 {
        self.output_bit_depth
    }

    /// Check if FFI decoding is available.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn is_decoding_available(&self) -> bool {
        self.ffi_decoder.is_some()
    }

    /// Check if FFI decoding is available.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn is_decoding_available(&self) -> bool {
        false
    }

    /// Check if Atmos decoding is enabled.
    pub fn is_atmos_enabled(&self) -> bool {
        self.decode_atmos
    }
}

impl Default for TrueHdDecoder {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Streaming decoder for processing DTS/TrueHD data incrementally.
#[derive(Debug)]
pub struct StreamingDecoder {
    /// Buffer for incoming data.
    buffer: Vec<u8>,
    /// Detected format.
    format: Option<crate::parser::AudioFormat>,
    /// DTS decoder.
    dts_decoder: DtsDecoder,
    /// TrueHD decoder.
    truehd_decoder: TrueHdDecoder,
}

impl StreamingDecoder {
    /// Create a new streaming decoder.
    pub fn new() -> Result<Self> {
        Ok(Self {
            buffer: Vec::new(),
            format: None,
            dts_decoder: DtsDecoder::new()?,
            truehd_decoder: TrueHdDecoder::new()?,
        })
    }

    /// Feed data to the decoder.
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Try to decode the next frame.
    pub fn decode_next(&mut self) -> Result<Option<DecodedAudio>> {
        // Need at least 4 bytes to detect format
        if self.buffer.len() < 4 {
            return Ok(None);
        }

        // Detect format if not already known
        if self.format.is_none() {
            self.format = Some(crate::parser::detect_format(&self.buffer));
        }

        match self.format {
            Some(crate::parser::AudioFormat::DtsCore) | Some(crate::parser::AudioFormat::DtsHd) => {
                match self.dts_decoder.decode_frame(&self.buffer) {
                    Ok(decoded) => {
                        let frame_info = self.dts_decoder.probe_frame(&self.buffer)?;
                        self.buffer.drain(..frame_info.frame_size);
                        Ok(Some(decoded))
                    }
                    Err(_) => Ok(None),
                }
            }
            Some(crate::parser::AudioFormat::TrueHd) => {
                match self.truehd_decoder.decode_frame(&self.buffer) {
                    Ok(decoded) => {
                        let frame_info = self.truehd_decoder.probe_frame(&self.buffer)?;
                        self.buffer.drain(..frame_info.frame_size);
                        Ok(Some(decoded))
                    }
                    Err(_) => Ok(None),
                }
            }
            _ => {
                // Try to find sync word
                let dts_parser = DtsParser::new();
                let truehd_parser = TrueHdParser::new();

                if let Some(pos) = dts_parser.find_sync(&self.buffer) {
                    self.buffer.drain(..pos);
                    self.format = Some(crate::parser::AudioFormat::DtsCore);
                } else if let Some(pos) = truehd_parser.find_sync(&self.buffer) {
                    self.buffer.drain(..pos);
                    self.format = Some(crate::parser::AudioFormat::TrueHd);
                }
                Ok(None)
            }
        }
    }

    /// Get the detected format.
    pub fn detected_format(&self) -> Option<crate::parser::AudioFormat> {
        self.format
    }

    /// Flush the decoder.
    pub fn flush(&mut self) {
        self.buffer.clear();
        self.dts_decoder.flush();
        self.truehd_decoder.flush();
    }

    /// Reset the decoder.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.format = None;
        self.dts_decoder.reset();
        self.truehd_decoder.reset();
    }

    /// Get buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for StreamingDecoder {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dts_decoder_new() {
        let decoder = DtsDecoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_dts_decoder_with_format() {
        let decoder = DtsDecoder::with_format(48000, 24);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.output_sample_rate(), Some(48000));
        assert_eq!(decoder.output_bit_depth(), 24);
    }

    #[test]
    fn test_truehd_decoder_new() {
        let decoder = TrueHdDecoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_streaming_decoder_new() {
        let decoder = StreamingDecoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_dts_decoder_reset() {
        let mut decoder = DtsDecoder::new().unwrap();
        decoder.reset();
        assert_eq!(decoder.frames_decoded(), 0);
        assert_eq!(decoder.samples_decoded(), 0);
    }

    #[test]
    fn test_truehd_decoder_reset() {
        let mut decoder = TrueHdDecoder::new().unwrap();
        decoder.reset();
        assert_eq!(decoder.frames_decoded(), 0);
        assert_eq!(decoder.samples_decoded(), 0);
    }

    #[test]
    fn test_is_decoding_available() {
        let decoder = DtsDecoder::new().unwrap();
        #[cfg(not(feature = "ffi-ffmpeg"))]
        assert!(!decoder.is_decoding_available());
    }

    #[test]
    fn test_streaming_decoder_feed() {
        let mut decoder = StreamingDecoder::new().unwrap();
        decoder.feed(&[0x7F, 0xFE, 0x80, 0x01]);
        assert_eq!(decoder.buffer_size(), 4);
    }

    #[test]
    fn test_streaming_decoder_flush() {
        let mut decoder = StreamingDecoder::new().unwrap();
        decoder.feed(&[0x7F, 0xFE, 0x80, 0x01]);
        decoder.flush();
        assert_eq!(decoder.buffer_size(), 0);
    }

    #[test]
    fn test_dts_decoder_hd_settings() {
        let mut decoder = DtsDecoder::new().unwrap();
        assert!(decoder.is_hd_enabled());
        decoder.set_decode_hd(false);
        assert!(!decoder.is_hd_enabled());
    }

    #[test]
    fn test_truehd_decoder_atmos_settings() {
        let mut decoder = TrueHdDecoder::new().unwrap();
        assert!(decoder.is_atmos_enabled());
        decoder.set_decode_atmos(false);
        assert!(!decoder.is_atmos_enabled());
    }
}
