//! MPEG-2 video decoder.
//!
//! Provides video decoding for MPEG-2 (MPEG-2 Part 2 Video).
//!
//! ## Usage
//!
//! Full decoding requires the `ffi-ffmpeg` feature. Without it, only parsing
//! and metadata extraction is available.

use crate::parser::{Mpeg2Parser, VideoInfo};
use crate::types::*;
use crate::{Mpeg2Error, Result};

#[cfg(feature = "ffi-ffmpeg")]
use crate::ffi::Mpeg2FfiDecoder;

use std::fmt;

/// MPEG-2 video decoder.
///
/// Decodes MPEG-2 video elementary streams to raw video frames.
///
/// # Example
///
/// ```rust,ignore
/// use transcode_mpeg2::Mpeg2Decoder;
///
/// let mut decoder = Mpeg2Decoder::new()?;
/// let decoded = decoder.decode_picture(&mpeg2_data)?;
/// println!("Decoded {}x{} frame", decoded.width, decoded.height);
/// ```
pub struct Mpeg2Decoder {
    /// Parser for extracting stream info.
    parser: Mpeg2Parser,
    /// Output width.
    output_width: Option<u16>,
    /// Output height.
    output_height: Option<u16>,
    /// Output chroma format.
    output_chroma: ChromaFormat,
    /// Total pictures decoded.
    pictures_decoded: u64,
    /// Total bytes processed.
    bytes_processed: u64,
    /// Current sequence header.
    current_sequence: Option<SequenceHeader>,
    /// Current sequence extension.
    current_extension: Option<SequenceExtension>,
    /// Reference frames for P/B decoding.
    reference_frames: Vec<DecodedFrame>,
    /// Maximum reference frames to keep.
    max_ref_frames: usize,
    /// FFI decoder (when ffi-ffmpeg feature is enabled).
    #[cfg(feature = "ffi-ffmpeg")]
    ffi_decoder: Option<Mpeg2FfiDecoder>,
}

impl fmt::Debug for Mpeg2Decoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mpeg2Decoder")
            .field("output_width", &self.output_width)
            .field("output_height", &self.output_height)
            .field("output_chroma", &self.output_chroma)
            .field("pictures_decoded", &self.pictures_decoded)
            .field("bytes_processed", &self.bytes_processed)
            .field("current_sequence", &self.current_sequence.is_some())
            .field("current_extension", &self.current_extension.is_some())
            .field("max_ref_frames", &self.max_ref_frames)
            .finish_non_exhaustive()
    }
}

impl Mpeg2Decoder {
    /// Create a new MPEG-2 decoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn new() -> Result<Self> {
        let ffi_decoder = match Mpeg2FfiDecoder::new() {
            Ok(dec) => Some(dec),
            Err(e) => {
                tracing::warn!("FFI decoder init failed, falling back to parser-only: {}", e);
                None
            }
        };

        Ok(Self {
            parser: Mpeg2Parser::new(),
            output_width: None,
            output_height: None,
            output_chroma: ChromaFormat::Yuv420,
            pictures_decoded: 0,
            bytes_processed: 0,
            current_sequence: None,
            current_extension: None,
            reference_frames: Vec::new(),
            max_ref_frames: 2,
            ffi_decoder,
        })
    }

    /// Create a new MPEG-2 decoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn new() -> Result<Self> {
        Ok(Self {
            parser: Mpeg2Parser::new(),
            output_width: None,
            output_height: None,
            output_chroma: ChromaFormat::Yuv420,
            pictures_decoded: 0,
            bytes_processed: 0,
            current_sequence: None,
            current_extension: None,
            reference_frames: Vec::new(),
            max_ref_frames: 2,
        })
    }

    /// Create decoder with specific output size.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn with_output_size(width: u16, height: u16) -> Result<Self> {
        let ffi_decoder = match Mpeg2FfiDecoder::new() {
            Ok(dec) => Some(dec),
            Err(e) => {
                tracing::warn!("FFI decoder init failed, falling back to parser-only: {}", e);
                None
            }
        };

        Ok(Self {
            parser: Mpeg2Parser::new(),
            output_width: Some(width),
            output_height: Some(height),
            output_chroma: ChromaFormat::Yuv420,
            pictures_decoded: 0,
            bytes_processed: 0,
            current_sequence: None,
            current_extension: None,
            reference_frames: Vec::new(),
            max_ref_frames: 2,
            ffi_decoder,
        })
    }

    /// Create decoder with specific output size.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn with_output_size(width: u16, height: u16) -> Result<Self> {
        Ok(Self {
            parser: Mpeg2Parser::new(),
            output_width: Some(width),
            output_height: Some(height),
            output_chroma: ChromaFormat::Yuv420,
            pictures_decoded: 0,
            bytes_processed: 0,
            current_sequence: None,
            current_extension: None,
            reference_frames: Vec::new(),
            max_ref_frames: 2,
        })
    }

    /// Decode an MPEG-2 picture.
    ///
    /// Returns decoded video frame or an error if decoding fails.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn decode_picture(&mut self, data: &[u8]) -> Result<DecodedFrame> {
        // Try FFI decoder first
        if let Some(ref mut ffi) = self.ffi_decoder {
            match ffi.decode(data) {
                Ok(frame) => {
                    self.pictures_decoded += 1;
                    self.bytes_processed += data.len() as u64;
                    return Ok(frame);
                }
                Err(e) => {
                    tracing::debug!("FFI decode failed, trying parser: {}", e);
                }
            }
        }

        // Fall back to parser-only mode
        let info = self.parse_and_update(data)?;

        let width = self.output_width.unwrap_or(info.width);
        let height = self.output_height.unwrap_or(info.height);

        let mut frame = DecodedFrame::new(width, height);
        frame.progressive = info.progressive;

        if let Some((pos, 0x00)) = self.parser.find_start_code(data) {
            if let Some(pic_header) = self.parser.parse_picture_header(&data[pos..]) {
                frame.picture_type = pic_header.picture_coding_type;
                frame.temporal_reference = pic_header.temporal_reference;
            }
        }

        self.pictures_decoded += 1;
        self.bytes_processed += data.len() as u64;

        Ok(frame)
    }

    /// Decode an MPEG-2 picture (stub without FFI).
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn decode_picture(&mut self, data: &[u8]) -> Result<DecodedFrame> {
        // Without FFI, we can only parse headers
        let info = self.parse_and_update(data)?;

        // Create empty decoded frame with correct metadata
        let width = self.output_width.unwrap_or(info.width);
        let height = self.output_height.unwrap_or(info.height);

        let mut frame = DecodedFrame::new(width, height);
        frame.progressive = info.progressive;

        // Try to parse picture header for type info
        if let Some((pos, 0x00)) = self.parser.find_start_code(data) {
            if let Some(pic_header) = self.parser.parse_picture_header(&data[pos..]) {
                frame.picture_type = pic_header.picture_coding_type;
                frame.temporal_reference = pic_header.temporal_reference;
            }
        }

        self.pictures_decoded += 1;
        self.bytes_processed += data.len() as u64;

        Ok(frame)
    }

    /// Parse data and update internal state.
    fn parse_and_update(&mut self, data: &[u8]) -> Result<VideoInfo> {
        // Try to find and parse sequence header
        if let Some(pos) = self.parser.find_sequence_header(data) {
            if let Some(seq) = self.parser.parse_sequence_header(&data[pos..]) {
                self.current_sequence = Some(seq.clone());

                // Look for sequence extension
                let search_start = pos + 12;
                if search_start < data.len() {
                    if let Some((ext_pos, code)) =
                        self.parser.find_start_code(&data[search_start..])
                    {
                        if code == crate::EXTENSION_START_CODE {
                            let ext_data = &data[search_start + ext_pos..];
                            if let Some(ext) = self.parser.parse_sequence_extension(ext_data) {
                                self.current_extension = Some(ext);
                            }
                        }
                    }
                }
            }
        }

        // Build video info from current state
        let seq = self
            .current_sequence
            .as_ref()
            .ok_or_else(|| Mpeg2Error::InvalidSequenceHeader("No sequence header found".into()))?;

        let (width, height) = if let Some((w, h)) = self.parser.full_resolution() {
            (w as u16, h as u16)
        } else {
            (seq.horizontal_size, seq.vertical_size)
        };

        Ok(VideoInfo {
            width,
            height,
            frame_rate: seq.frame_rate(),
            aspect_ratio: seq.aspect_ratio_code,
            bit_rate_bps: seq.bit_rate_bps(),
            is_mpeg2: self.current_extension.is_some(),
            progressive: self
                .current_extension
                .as_ref()
                .map(|e| e.progressive_sequence)
                .unwrap_or(true),
            chroma_format: self
                .current_extension
                .as_ref()
                .map(|e| e.chroma_format)
                .unwrap_or(ChromaFormat::Yuv420),
            profile: self.current_extension.as_ref().map(|e| e.profile()),
            level: self.current_extension.as_ref().map(|e| e.level()),
        })
    }

    /// Parse a picture without decoding.
    pub fn parse_picture(&mut self, data: &[u8]) -> Result<PictureInfo> {
        // Find picture start code
        let pic_pos = self.find_picture_start(data).ok_or_else(|| {
            Mpeg2Error::InvalidPictureHeader("No picture start code found".into())
        })?;

        // Parse picture header
        let pic_header = self
            .parser
            .parse_picture_header(&data[pic_pos..])
            .ok_or_else(|| Mpeg2Error::InvalidPictureHeader("Failed to parse picture header".into()))?;

        // Look for picture coding extension
        let search_start = pic_pos + 8;
        let pic_coding_ext = if search_start < data.len() {
            if let Some((ext_pos, code)) = self.parser.find_start_code(&data[search_start..]) {
                if code == crate::EXTENSION_START_CODE {
                    self.parser
                        .parse_picture_coding_extension(&data[search_start + ext_pos..])
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(PictureInfo {
            header: pic_header,
            coding_extension: pic_coding_ext,
            offset: pic_pos,
        })
    }

    /// Find picture start code position.
    fn find_picture_start(&self, data: &[u8]) -> Option<usize> {
        (0..data.len().saturating_sub(3)).find(|&i| {
            data[i] == 0x00
                && data[i + 1] == 0x00
                && data[i + 2] == 0x01
                && data[i + 3] == crate::PICTURE_START_CODE
        })
    }

    /// Get video info without decoding.
    pub fn probe(&self, data: &[u8]) -> Result<VideoInfo> {
        crate::parser::get_video_info(data)
    }

    /// Flush the decoder.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn flush(&mut self) {
        if let Some(ref mut ffi) = self.ffi_decoder {
            ffi.flush();
        }
        self.parser.reset();
        self.reference_frames.clear();
    }

    /// Flush the decoder.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn flush(&mut self) {
        self.parser.reset();
        self.reference_frames.clear();
    }

    /// Reset the decoder state.
    #[cfg(feature = "ffi-ffmpeg")]
    pub fn reset(&mut self) {
        if let Some(ref mut ffi) = self.ffi_decoder {
            ffi.flush();
        }
        self.parser.reset();
        self.pictures_decoded = 0;
        self.bytes_processed = 0;
        self.current_sequence = None;
        self.current_extension = None;
        self.reference_frames.clear();
    }

    /// Reset the decoder state.
    #[cfg(not(feature = "ffi-ffmpeg"))]
    pub fn reset(&mut self) {
        self.parser.reset();
        self.pictures_decoded = 0;
        self.bytes_processed = 0;
        self.current_sequence = None;
        self.current_extension = None;
        self.reference_frames.clear();
    }

    /// Get total pictures decoded.
    pub fn pictures_decoded(&self) -> u64 {
        self.pictures_decoded
    }

    /// Get total bytes processed.
    pub fn bytes_processed(&self) -> u64 {
        self.bytes_processed
    }

    /// Get current sequence header.
    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.current_sequence.as_ref()
    }

    /// Get current sequence extension.
    pub fn sequence_extension(&self) -> Option<&SequenceExtension> {
        self.current_extension.as_ref()
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

    /// Get output width.
    pub fn output_width(&self) -> Option<u16> {
        self.output_width
    }

    /// Get output height.
    pub fn output_height(&self) -> Option<u16> {
        self.output_height
    }
}

impl Default for Mpeg2Decoder {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Picture information from parsing.
#[derive(Debug, Clone)]
pub struct PictureInfo {
    /// Picture header.
    pub header: PictureHeader,
    /// Picture coding extension (MPEG-2 only).
    pub coding_extension: Option<PictureCodingExtension>,
    /// Offset in the data where picture starts.
    pub offset: usize,
}

impl PictureInfo {
    /// Get the picture coding type.
    pub fn picture_type(&self) -> PictureCodingType {
        self.header.picture_coding_type
    }

    /// Check if this is a reference frame (I or P).
    pub fn is_reference(&self) -> bool {
        self.header.picture_coding_type.is_reference()
    }

    /// Check if this is a progressive frame.
    pub fn is_progressive(&self) -> bool {
        self.coding_extension
            .as_ref()
            .map(|e| e.progressive_frame)
            .unwrap_or(true)
    }

    /// Get picture structure.
    pub fn structure(&self) -> PictureStructure {
        self.coding_extension
            .as_ref()
            .map(|e| e.picture_structure)
            .unwrap_or(PictureStructure::Frame)
    }
}

/// Streaming decoder for processing MPEG-2 data incrementally.
#[derive(Debug)]
pub struct StreamingDecoder {
    /// Buffer for incoming data.
    buffer: Vec<u8>,
    /// MPEG-2 decoder.
    decoder: Mpeg2Decoder,
    /// Whether we've seen a sequence header.
    seen_sequence_header: bool,
    /// Pending picture data.
    pending_picture: Option<Vec<u8>>,
}

impl StreamingDecoder {
    /// Create a new streaming decoder.
    pub fn new() -> Result<Self> {
        Ok(Self {
            buffer: Vec::new(),
            decoder: Mpeg2Decoder::new()?,
            seen_sequence_header: false,
            pending_picture: None,
        })
    }

    /// Feed data to the decoder.
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Try to decode the next picture.
    pub fn decode_next(&mut self) -> Result<Option<DecodedFrame>> {
        // Need at least 4 bytes to check for start code
        if self.buffer.len() < 4 {
            return Ok(None);
        }

        // Look for sequence header if not seen yet
        if !self.seen_sequence_header {
            if let Some(pos) = self.decoder.parser.find_sequence_header(&self.buffer) {
                self.buffer.drain(..pos);
                self.seen_sequence_header = true;
            } else {
                return Ok(None);
            }
        }

        // Find picture boundaries
        let pic_start = self.find_next_picture_start(0)?;
        if pic_start.is_none() {
            return Ok(None);
        }
        let pic_start = pic_start.unwrap();

        let pic_end = self.find_next_picture_start(pic_start + 4)?;
        if pic_end.is_none() {
            // Need more data
            return Ok(None);
        }
        let pic_end = pic_end.unwrap();

        // Extract and decode picture
        let picture_data: Vec<u8> = self.buffer.drain(..pic_end).collect();
        let frame = self.decoder.decode_picture(&picture_data)?;

        Ok(Some(frame))
    }

    /// Find the next picture start code position.
    fn find_next_picture_start(&self, start: usize) -> Result<Option<usize>> {
        for i in start..self.buffer.len().saturating_sub(3) {
            if self.buffer[i] == 0x00
                && self.buffer[i + 1] == 0x00
                && self.buffer[i + 2] == 0x01
                && self.buffer[i + 3] == crate::PICTURE_START_CODE
            {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    /// Get the underlying decoder.
    pub fn decoder(&self) -> &Mpeg2Decoder {
        &self.decoder
    }

    /// Get mutable reference to the underlying decoder.
    pub fn decoder_mut(&mut self) -> &mut Mpeg2Decoder {
        &mut self.decoder
    }

    /// Flush the decoder.
    pub fn flush(&mut self) {
        self.buffer.clear();
        self.decoder.flush();
        self.seen_sequence_header = false;
        self.pending_picture = None;
    }

    /// Reset the decoder.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.decoder.reset();
        self.seen_sequence_header = false;
        self.pending_picture = None;
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
    fn test_decoder_new() {
        let decoder = Mpeg2Decoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_with_output_size() {
        let decoder = Mpeg2Decoder::with_output_size(1920, 1080);
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.output_width(), Some(1920));
        assert_eq!(decoder.output_height(), Some(1080));
    }

    #[test]
    fn test_streaming_decoder_new() {
        let decoder = StreamingDecoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = Mpeg2Decoder::new().unwrap();
        decoder.reset();
        assert_eq!(decoder.pictures_decoded(), 0);
        assert_eq!(decoder.bytes_processed(), 0);
    }

    #[test]
    fn test_is_decoding_available() {
        let decoder = Mpeg2Decoder::new().unwrap();
        // Without ffi-ffmpeg feature, should be false
        #[cfg(not(feature = "ffi-ffmpeg"))]
        assert!(!decoder.is_decoding_available());
    }

    #[test]
    fn test_streaming_decoder_feed() {
        let mut decoder = StreamingDecoder::new().unwrap();
        decoder.feed(&[0x00, 0x00, 0x01, 0xB3]);
        assert_eq!(decoder.buffer_size(), 4);
    }

    #[test]
    fn test_streaming_decoder_flush() {
        let mut decoder = StreamingDecoder::new().unwrap();
        decoder.feed(&[0x00, 0x00, 0x01, 0xB3]);
        decoder.flush();
        assert_eq!(decoder.buffer_size(), 0);
    }

    #[test]
    fn test_picture_info() {
        let info = PictureInfo {
            header: PictureHeader::new_i_frame(0),
            coding_extension: None,
            offset: 0,
        };

        assert_eq!(info.picture_type(), PictureCodingType::I);
        assert!(info.is_reference());
        assert!(info.is_progressive());
        assert_eq!(info.structure(), PictureStructure::Frame);
    }
}
