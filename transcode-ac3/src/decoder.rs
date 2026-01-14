//! AC-3 and E-AC-3 decoders.
//!
//! Provides audio decoding for AC-3 (Dolby Digital) and E-AC-3 (Dolby Digital Plus).
//!
//! ## Usage
//!
//! Full decoding requires the `ffi-ffmpeg` feature. Without it, only parsing
//! and metadata extraction is available.

use std::fmt;

#[cfg(feature = "ffi-ffmpeg")]
use crate::ffi::{Ac3FfiDecoder, Eac3FfiDecoder};
use crate::parser::{Ac3Parser, Eac3Parser};
use crate::types::*;
use crate::{Ac3Error, Result};

/// AC-3 decoder.
///
/// Decodes AC-3 (Dolby Digital) audio frames to PCM samples.
///
/// # Example
///
/// ```rust,ignore
/// use transcode_ac3::Ac3Decoder;
///
/// let mut decoder = Ac3Decoder::new()?;
/// let decoded = decoder.decode_frame(&ac3_frame)?;
/// println!("Decoded {} samples", decoded.samples_per_channel);
/// ```
pub struct Ac3Decoder {
    /// Parser for extracting frame info.
    parser: Ac3Parser,
    /// Output sample rate.
    output_sample_rate: u32,
    /// Output channel layout.
    output_layout: Option<ChannelLayout>,
    /// Decoded sample format.
    sample_format: SampleFormat,
    /// Total frames decoded.
    frames_decoded: u64,
    /// Total samples decoded.
    samples_decoded: u64,
    /// FFI decoder (when ffi-ffmpeg feature is enabled).
    #[cfg(feature = "ffi-ffmpeg")]
    ffi_decoder: Option<Ac3FfiDecoder>,
}

impl fmt::Debug for Ac3Decoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ac3Decoder")
            .field("output_sample_rate", &self.output_sample_rate)
            .field("output_layout", &self.output_layout)
            .field("sample_format", &self.sample_format)
            .field("frames_decoded", &self.frames_decoded)
            .field("samples_decoded", &self.samples_decoded)
            .finish_non_exhaustive()
    }
}

/// Sample format for decoded audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 32-bit float samples.
    F32,
    /// 16-bit integer samples.
    S16,
    /// 32-bit integer samples.
    S32,
}

impl Ac3Decoder {
    /// Create a new AC-3 decoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn new() -> Result<Self> {
        let ffi_decoder = Ac3FfiDecoder::new().ok();
        Ok(Self {
            parser: Ac3Parser::new(),
            output_sample_rate: 48000,
            output_layout: None,
            sample_format: SampleFormat::F32,
            frames_decoded: 0,
            samples_decoded: 0,
            ffi_decoder,
        })
    }

    /// Create a new AC-3 decoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn new() -> Result<Self> {
        Ok(Self {
            parser: Ac3Parser::new(),
            output_sample_rate: 48000,
            output_layout: None,
            sample_format: SampleFormat::F32,
            frames_decoded: 0,
            samples_decoded: 0,
        })
    }

    /// Create decoder with specific output format.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn with_format(sample_rate: u32, sample_format: SampleFormat) -> Result<Self> {
        let ffi_decoder = Ac3FfiDecoder::new().ok();
        Ok(Self {
            parser: Ac3Parser::new(),
            output_sample_rate: sample_rate,
            output_layout: None,
            sample_format,
            frames_decoded: 0,
            samples_decoded: 0,
            ffi_decoder,
        })
    }

    /// Create decoder with specific output format.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn with_format(sample_rate: u32, sample_format: SampleFormat) -> Result<Self> {
        Ok(Self {
            parser: Ac3Parser::new(),
            output_sample_rate: sample_rate,
            output_layout: None,
            sample_format,
            frames_decoded: 0,
            samples_decoded: 0,
        })
    }

    /// Decode an AC-3 frame.
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
            // Fall back to parsing only
            let frame = self.parse_frame(data)?;
            let layout = ChannelLayout::from_acmod(frame.acmod, frame.lfe_on);

            Ok(DecodedAudio {
                sample_rate: frame.sample_rate,
                channels: frame.channels,
                layout,
                samples: Vec::new(),
                samples_per_channel: 0,
            })
        }
    }

    /// Decode an AC-3 frame (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        // Without FFI, we can only parse the frame
        let frame = self.parse_frame(data)?;

        // Create empty decoded audio with correct metadata
        let layout = ChannelLayout::from_acmod(frame.acmod, frame.lfe_on);

        Ok(DecodedAudio {
            sample_rate: frame.sample_rate,
            channels: frame.channels,
            layout,
            samples: Vec::new(), // No actual decoding without FFI
            samples_per_channel: 0,
        })
    }

    /// Parse a frame without decoding.
    pub fn parse_frame(&mut self, data: &[u8]) -> Result<Ac3SyncFrame> {
        self.parser.reset();
        self.parser
            .parse_sync_frame(data)
            .ok_or_else(|| Ac3Error::BitstreamCorruption("Could not parse sync frame".into()))
    }

    /// Get frame information without decoding.
    pub fn probe_frame(&self, data: &[u8]) -> Result<Ac3SyncFrame> {
        let mut parser = Ac3Parser::new();
        parser
            .parse_sync_frame(data)
            .ok_or_else(|| Ac3Error::BitstreamCorruption("Could not parse sync frame".into()))
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
    pub fn output_sample_rate(&self) -> u32 {
        self.output_sample_rate
    }

    /// Get the sample format.
    pub fn sample_format(&self) -> SampleFormat {
        self.sample_format
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
}

impl Default for Ac3Decoder {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// E-AC-3 decoder.
///
/// Decodes E-AC-3 (Dolby Digital Plus) audio frames to PCM samples.
pub struct Eac3Decoder {
    /// Parser for extracting frame info.
    parser: Eac3Parser,
    /// Output sample rate.
    output_sample_rate: u32,
    /// Output channel layout.
    output_layout: Option<ChannelLayout>,
    /// Decoded sample format.
    sample_format: SampleFormat,
    /// Total frames decoded.
    frames_decoded: u64,
    /// Total samples decoded.
    samples_decoded: u64,
    /// Substream to decode (for multi-program streams).
    target_substream: u8,
    /// FFI decoder (when ffi-ffmpeg feature is enabled).
    #[cfg(feature = "ffi-ffmpeg")]
    ffi_decoder: Option<Eac3FfiDecoder>,
}

impl fmt::Debug for Eac3Decoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Eac3Decoder")
            .field("output_sample_rate", &self.output_sample_rate)
            .field("output_layout", &self.output_layout)
            .field("sample_format", &self.sample_format)
            .field("frames_decoded", &self.frames_decoded)
            .field("samples_decoded", &self.samples_decoded)
            .field("target_substream", &self.target_substream)
            .finish_non_exhaustive()
    }
}

impl Eac3Decoder {
    /// Create a new E-AC-3 decoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn new() -> Result<Self> {
        let ffi_decoder = Eac3FfiDecoder::new().ok();
        Ok(Self {
            parser: Eac3Parser::new(),
            output_sample_rate: 48000,
            output_layout: None,
            sample_format: SampleFormat::F32,
            frames_decoded: 0,
            samples_decoded: 0,
            target_substream: 0,
            ffi_decoder,
        })
    }

    /// Create a new E-AC-3 decoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn new() -> Result<Self> {
        Ok(Self {
            parser: Eac3Parser::new(),
            output_sample_rate: 48000,
            output_layout: None,
            sample_format: SampleFormat::F32,
            frames_decoded: 0,
            samples_decoded: 0,
            target_substream: 0,
        })
    }

    /// Create decoder with specific output format.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn with_format(sample_rate: u32, sample_format: SampleFormat) -> Result<Self> {
        let ffi_decoder = Eac3FfiDecoder::new().ok();
        Ok(Self {
            parser: Eac3Parser::new(),
            output_sample_rate: sample_rate,
            output_layout: None,
            sample_format,
            frames_decoded: 0,
            samples_decoded: 0,
            target_substream: 0,
            ffi_decoder,
        })
    }

    /// Create decoder with specific output format.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn with_format(sample_rate: u32, sample_format: SampleFormat) -> Result<Self> {
        Ok(Self {
            parser: Eac3Parser::new(),
            output_sample_rate: sample_rate,
            output_layout: None,
            sample_format,
            frames_decoded: 0,
            samples_decoded: 0,
            target_substream: 0,
        })
    }

    /// Set the target substream for decoding.
    pub fn set_target_substream(&mut self, substream: u8) {
        self.target_substream = substream;
    }

    /// Decode an E-AC-3 frame.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        if let Some(ref mut ffi) = self.ffi_decoder {
            let decoded = ffi.decode(data)?;
            self.frames_decoded += 1;
            self.samples_decoded += decoded.samples_per_channel as u64;
            Ok(decoded)
        } else {
            let frame = self.parse_frame(data)?;
            let layout = ChannelLayout::from_acmod(frame.acmod, frame.lfe_on);

            Ok(DecodedAudio {
                sample_rate: frame.sample_rate,
                channels: frame.channels,
                layout,
                samples: Vec::new(),
                samples_per_channel: 0,
            })
        }
    }

    /// Decode an E-AC-3 frame (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<DecodedAudio> {
        let frame = self.parse_frame(data)?;
        let layout = ChannelLayout::from_acmod(frame.acmod, frame.lfe_on);

        Ok(DecodedAudio {
            sample_rate: frame.sample_rate,
            channels: frame.channels,
            layout,
            samples: Vec::new(),
            samples_per_channel: 0,
        })
    }

    /// Parse a frame without decoding.
    pub fn parse_frame(&mut self, data: &[u8]) -> Result<Eac3SyncFrame> {
        self.parser.reset();
        self.parser
            .parse_sync_frame(data)
            .ok_or_else(|| Ac3Error::BitstreamCorruption("Could not parse sync frame".into()))
    }

    /// Get frame information without decoding.
    pub fn probe_frame(&self, data: &[u8]) -> Result<Eac3SyncFrame> {
        let mut parser = Eac3Parser::new();
        parser
            .parse_sync_frame(data)
            .ok_or_else(|| Ac3Error::BitstreamCorruption("Could not parse sync frame".into()))
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
    pub fn output_sample_rate(&self) -> u32 {
        self.output_sample_rate
    }

    /// Get the sample format.
    pub fn sample_format(&self) -> SampleFormat {
        self.sample_format
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

    /// Check if stream appears to be Dolby Atmos.
    pub fn is_atmos_stream(&self, data: &[u8]) -> Result<bool> {
        let frame = self.probe_frame(data)?;
        Ok(frame.is_atmos())
    }
}

impl Default for Eac3Decoder {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Streaming decoder for processing AC-3/E-AC-3 data incrementally.
#[derive(Debug)]
pub struct StreamingDecoder {
    /// Buffer for incoming data.
    buffer: Vec<u8>,
    /// Detected format.
    format: Option<crate::parser::Ac3Format>,
    /// AC-3 decoder.
    ac3_decoder: Ac3Decoder,
    /// E-AC-3 decoder.
    eac3_decoder: Eac3Decoder,
}

impl StreamingDecoder {
    /// Create a new streaming decoder.
    pub fn new() -> Result<Self> {
        Ok(Self {
            buffer: Vec::new(),
            format: None,
            ac3_decoder: Ac3Decoder::new()?,
            eac3_decoder: Eac3Decoder::new()?,
        })
    }

    /// Feed data to the decoder.
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Try to decode the next frame.
    pub fn decode_next(&mut self) -> Result<Option<DecodedAudio>> {
        // Need at least 6 bytes to detect format
        if self.buffer.len() < 6 {
            return Ok(None);
        }

        // Detect format if not already known
        if self.format.is_none() {
            self.format = crate::parser::detect_format(&self.buffer);
        }

        match self.format {
            Some(crate::parser::Ac3Format::Ac3) => {
                // Try to decode AC-3 frame
                match self.ac3_decoder.decode_frame(&self.buffer) {
                    Ok(decoded) => {
                        // Remove consumed data
                        let frame_info = self.ac3_decoder.probe_frame(&self.buffer)?;
                        self.buffer.drain(..frame_info.frame_size);
                        Ok(Some(decoded))
                    }
                    Err(_) => Ok(None),
                }
            }
            Some(crate::parser::Ac3Format::Eac3) => {
                match self.eac3_decoder.decode_frame(&self.buffer) {
                    Ok(decoded) => {
                        let frame_info = self.eac3_decoder.probe_frame(&self.buffer)?;
                        self.buffer.drain(..frame_info.frame_size);
                        Ok(Some(decoded))
                    }
                    Err(_) => Ok(None),
                }
            }
            None => {
                // Try to find sync word
                if let Some(pos) = self.find_sync() {
                    self.buffer.drain(..pos);
                }
                Ok(None)
            }
        }
    }

    /// Find sync word position in buffer.
    fn find_sync(&self) -> Option<usize> {
        (0..self.buffer.len().saturating_sub(1))
            .find(|&i| self.buffer[i] == 0x0B && self.buffer[i + 1] == 0x77)
    }

    /// Get the detected format.
    pub fn detected_format(&self) -> Option<crate::parser::Ac3Format> {
        self.format
    }

    /// Flush the decoder.
    pub fn flush(&mut self) {
        self.buffer.clear();
        self.ac3_decoder.flush();
        self.eac3_decoder.flush();
    }

    /// Reset the decoder.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.format = None;
        self.ac3_decoder.reset();
        self.eac3_decoder.reset();
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
    fn test_ac3_decoder_new() {
        let decoder = Ac3Decoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_eac3_decoder_new() {
        let decoder = Eac3Decoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_streaming_decoder_new() {
        let decoder = StreamingDecoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = Ac3Decoder::new().unwrap();
        decoder.reset();
        assert_eq!(decoder.frames_decoded(), 0);
        assert_eq!(decoder.samples_decoded(), 0);
    }

    #[test]
    fn test_sample_format() {
        let decoder = Ac3Decoder::new().unwrap();
        assert_eq!(decoder.sample_format(), SampleFormat::F32);
    }

    #[test]
    fn test_is_decoding_available() {
        let decoder = Ac3Decoder::new().unwrap();
        // Without ffi-ffmpeg feature, should be false
        #[cfg(not(feature = "ffi-ffmpeg"))]
        assert!(!decoder.is_decoding_available());
    }

    #[test]
    fn test_streaming_decoder_feed() {
        let mut decoder = StreamingDecoder::new().unwrap();
        decoder.feed(&[0x0B, 0x77, 0x00, 0x00]);
        assert_eq!(decoder.buffer.len(), 4);
    }

    #[test]
    fn test_streaming_decoder_flush() {
        let mut decoder = StreamingDecoder::new().unwrap();
        decoder.feed(&[0x0B, 0x77, 0x00, 0x00]);
        decoder.flush();
        assert!(decoder.buffer.is_empty());
    }
}
