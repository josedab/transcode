//! Vorbis decoder implementation.

use crate::codebook::Codebook;
use crate::error::{VorbisError, Result};
use crate::floor::Floor;
use crate::mdct::Mdct;
use crate::residue::Residue;

/// Vorbis stream information.
#[derive(Debug, Clone, Default)]
pub struct VorbisInfo {
    /// Vorbis version (should be 0).
    pub version: u32,
    /// Number of channels.
    pub channels: u8,
    /// Sample rate.
    pub sample_rate: u32,
    /// Maximum bitrate (0 = unset).
    pub bitrate_maximum: i32,
    /// Nominal bitrate (0 = unset).
    pub bitrate_nominal: i32,
    /// Minimum bitrate (0 = unset).
    pub bitrate_minimum: i32,
    /// Block size 0 (short block).
    pub block_size_0: u16,
    /// Block size 1 (long block).
    pub block_size_1: u16,
}

/// Vorbis comment (metadata).
#[derive(Debug, Clone, Default)]
pub struct VorbisComment {
    /// Vendor string.
    pub vendor: String,
    /// User comments (key=value pairs).
    pub comments: Vec<(String, String)>,
}

impl VorbisComment {
    /// Get a comment by key (case-insensitive).
    pub fn get(&self, key: &str) -> Option<&str> {
        let key_lower = key.to_lowercase();
        self.comments
            .iter()
            .find(|(k, _)| k.to_lowercase() == key_lower)
            .map(|(_, v)| v.as_str())
    }

    /// Get title.
    pub fn title(&self) -> Option<&str> {
        self.get("title")
    }

    /// Get artist.
    pub fn artist(&self) -> Option<&str> {
        self.get("artist")
    }

    /// Get album.
    pub fn album(&self) -> Option<&str> {
        self.get("album")
    }

    /// Get track number.
    pub fn track_number(&self) -> Option<&str> {
        self.get("tracknumber")
    }
}

/// Vorbis decoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderState {
    /// Waiting for identification header.
    WaitingIdHeader,
    /// Waiting for comment header.
    WaitingCommentHeader,
    /// Waiting for setup header.
    WaitingSetupHeader,
    /// Ready to decode audio packets.
    Ready,
    /// End of stream.
    EndOfStream,
}

/// Vorbis decoder.
#[derive(Debug)]
pub struct VorbisDecoder {
    /// Decoder state.
    state: DecoderState,
    /// Stream information.
    info: VorbisInfo,
    /// Comments/metadata.
    comment: VorbisComment,
    /// Codebooks.
    codebooks: Vec<Codebook>,
    /// Floor configurations.
    floors: Vec<Floor>,
    /// Residue configurations.
    residues: Vec<Residue>,
    /// MDCT for short blocks.
    mdct_short: Option<Mdct>,
    /// MDCT for long blocks.
    mdct_long: Option<Mdct>,
    /// Previous window data for overlap-add.
    prev_window: Vec<Vec<f32>>,
    /// Previous block flag.
    prev_block_flag: bool,
    /// Total samples decoded.
    samples_decoded: u64,
}

impl VorbisDecoder {
    /// Create a new Vorbis decoder.
    pub fn new() -> Self {
        Self {
            state: DecoderState::WaitingIdHeader,
            info: VorbisInfo::default(),
            comment: VorbisComment::default(),
            codebooks: Vec::new(),
            floors: Vec::new(),
            residues: Vec::new(),
            mdct_short: None,
            mdct_long: None,
            prev_window: Vec::new(),
            prev_block_flag: false,
            samples_decoded: 0,
        }
    }

    /// Get decoder state.
    pub fn state(&self) -> DecoderState {
        self.state
    }

    /// Get stream info (only valid after headers are decoded).
    pub fn info(&self) -> &VorbisInfo {
        &self.info
    }

    /// Get comments (only valid after comment header is decoded).
    pub fn comment(&self) -> &VorbisComment {
        &self.comment
    }

    /// Get sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.info.sample_rate
    }

    /// Get number of channels.
    pub fn channels(&self) -> u8 {
        self.info.channels
    }

    /// Get total samples decoded.
    pub fn samples_decoded(&self) -> u64 {
        self.samples_decoded
    }

    /// Submit a Vorbis packet for decoding.
    pub fn decode_packet(&mut self, data: &[u8]) -> Result<Option<Vec<Vec<f32>>>> {
        if data.is_empty() {
            return Err(VorbisError::BitstreamError("Empty packet".into()));
        }

        // Check packet type (first byte & 1)
        let packet_type = data[0] & 1;

        if packet_type == 1 {
            // Header packet
            self.decode_header(data)?;
            Ok(None)
        } else {
            // Audio packet
            if self.state != DecoderState::Ready {
                return Err(VorbisError::NotInitialized);
            }
            self.decode_audio(data).map(Some)
        }
    }

    /// Decode a header packet.
    fn decode_header(&mut self, data: &[u8]) -> Result<()> {
        if data.len() < 7 {
            return Err(VorbisError::InvalidHeader("Packet too short".into()));
        }

        // Check "vorbis" signature
        if &data[1..7] != b"vorbis" {
            return Err(VorbisError::InvalidHeader("Invalid signature".into()));
        }

        let header_type = data[0];

        match header_type {
            1 => self.decode_id_header(data),
            3 => self.decode_comment_header(data),
            5 => self.decode_setup_header(data),
            _ => Err(VorbisError::InvalidHeader(format!(
                "Unknown header type: {}",
                header_type
            ))),
        }
    }

    /// Decode identification header.
    fn decode_id_header(&mut self, data: &[u8]) -> Result<()> {
        if self.state != DecoderState::WaitingIdHeader {
            return Err(VorbisError::InvalidIdHeader("Unexpected ID header".into()));
        }

        if data.len() < 30 {
            return Err(VorbisError::InvalidIdHeader("Header too short".into()));
        }

        // Parse header fields
        let version = u32::from_le_bytes([data[7], data[8], data[9], data[10]]);
        if version != 0 {
            return Err(VorbisError::InvalidIdHeader(format!(
                "Unsupported version: {}",
                version
            )));
        }

        let channels = data[11];
        if channels == 0 || channels > 8 {
            return Err(VorbisError::UnsupportedChannels(channels));
        }

        let sample_rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        if sample_rate < 8000 || sample_rate > 192000 {
            return Err(VorbisError::UnsupportedSampleRate(sample_rate));
        }

        let bitrate_maximum = i32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let bitrate_nominal = i32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let bitrate_minimum = i32::from_le_bytes([data[24], data[25], data[26], data[27]]);

        let block_sizes = data[28];
        let block_size_0 = 1u16 << (block_sizes & 0x0F);
        let block_size_1 = 1u16 << ((block_sizes >> 4) & 0x0F);

        // Validate block sizes
        if block_size_0 < 64 || block_size_0 > 8192 || block_size_1 < block_size_0 {
            return Err(VorbisError::InvalidIdHeader("Invalid block sizes".into()));
        }

        // Check framing bit
        if data[29] & 1 != 1 {
            return Err(VorbisError::InvalidIdHeader("Missing framing bit".into()));
        }

        self.info = VorbisInfo {
            version,
            channels,
            sample_rate,
            bitrate_maximum,
            bitrate_nominal,
            bitrate_minimum,
            block_size_0,
            block_size_1,
        };

        self.state = DecoderState::WaitingCommentHeader;
        Ok(())
    }

    /// Decode comment header.
    fn decode_comment_header(&mut self, data: &[u8]) -> Result<()> {
        if self.state != DecoderState::WaitingCommentHeader {
            return Err(VorbisError::InvalidCommentHeader(
                "Unexpected comment header".into(),
            ));
        }

        let mut offset = 7; // Skip type + "vorbis"

        if offset + 4 > data.len() {
            return Err(VorbisError::InvalidCommentHeader("Header too short".into()));
        }

        // Vendor string length
        let vendor_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + vendor_len > data.len() {
            return Err(VorbisError::InvalidCommentHeader("Invalid vendor length".into()));
        }

        let vendor = String::from_utf8_lossy(&data[offset..offset + vendor_len]).to_string();
        offset += vendor_len;

        if offset + 4 > data.len() {
            return Err(VorbisError::InvalidCommentHeader("Header too short".into()));
        }

        // Number of comments
        let comment_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut comments = Vec::with_capacity(comment_count);

        for _ in 0..comment_count {
            if offset + 4 > data.len() {
                break;
            }

            let comment_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + comment_len > data.len() {
                break;
            }

            let comment = String::from_utf8_lossy(&data[offset..offset + comment_len]);
            offset += comment_len;

            // Parse key=value
            if let Some(eq_pos) = comment.find('=') {
                let key = comment[..eq_pos].to_string();
                let value = comment[eq_pos + 1..].to_string();
                comments.push((key, value));
            }
        }

        self.comment = VorbisComment { vendor, comments };
        self.state = DecoderState::WaitingSetupHeader;
        Ok(())
    }

    /// Decode setup header.
    fn decode_setup_header(&mut self, data: &[u8]) -> Result<()> {
        if self.state != DecoderState::WaitingSetupHeader {
            return Err(VorbisError::InvalidSetupHeader(
                "Unexpected setup header".into(),
            ));
        }

        // Initialize MDCT
        self.mdct_short = Some(Mdct::new(self.info.block_size_0 as usize));
        self.mdct_long = Some(Mdct::new(self.info.block_size_1 as usize));

        // Initialize floors
        self.floors.push(Floor::new_type1(self.info.block_size_0 as usize));
        self.floors.push(Floor::new_type1(self.info.block_size_1 as usize));

        // Initialize residues
        self.residues.push(Residue::new(2, self.info.block_size_0 as usize));
        self.residues.push(Residue::new(2, self.info.block_size_1 as usize));

        // Initialize previous window buffer
        self.prev_window = vec![
            vec![0.0f32; self.info.block_size_1 as usize];
            self.info.channels as usize
        ];

        self.state = DecoderState::Ready;
        Ok(())
    }

    /// Decode an audio packet.
    fn decode_audio(&mut self, data: &[u8]) -> Result<Vec<Vec<f32>>> {
        if data.is_empty() {
            return Err(VorbisError::BitstreamError("Empty audio packet".into()));
        }

        // Determine block size from first bit
        let block_flag = (data[0] >> 1) & 1;
        let block_size = if block_flag == 1 {
            self.info.block_size_1
        } else {
            self.info.block_size_0
        } as usize;

        let mdct = if block_flag == 1 {
            self.mdct_long.as_ref()
        } else {
            self.mdct_short.as_ref()
        }.ok_or(VorbisError::NotInitialized)?;

        let n2 = block_size / 2;
        let channels = self.info.channels as usize;

        // Allocate channel buffers
        let mut channel_data: Vec<Vec<f32>> = vec![vec![0.0f32; block_size]; channels];

        // Decode floor and residue (simplified)
        // A full implementation would decode from the bitstream

        // Apply inverse MDCT
        for ch in 0..channels {
            let mut freq = vec![0.0f32; n2];
            let mut time = vec![0.0f32; block_size];

            // Get floor (using placeholder data)
            freq.iter_mut().for_each(|f| *f = 0.0);

            mdct.inverse(&freq, &mut time);
            channel_data[ch] = time;
        }

        // Overlap-add with previous block
        let output_samples = n2;
        let mut output: Vec<Vec<f32>> = vec![vec![0.0f32; output_samples]; channels];

        for ch in 0..channels {
            mdct.overlap_add(
                &channel_data[ch],
                &self.prev_window[ch],
                &mut output[ch],
            );
            // Save current window for next overlap
            self.prev_window[ch] = channel_data[ch].clone();
        }

        self.prev_block_flag = block_flag == 1;
        self.samples_decoded += output_samples as u64;

        Ok(output)
    }

    /// Reset the decoder.
    pub fn reset(&mut self) {
        self.state = DecoderState::WaitingIdHeader;
        self.info = VorbisInfo::default();
        self.comment = VorbisComment::default();
        self.codebooks.clear();
        self.floors.clear();
        self.residues.clear();
        self.mdct_short = None;
        self.mdct_long = None;
        self.prev_window.clear();
        self.prev_block_flag = false;
        self.samples_decoded = 0;
    }

    /// Flush decoder (for seeking).
    pub fn flush(&mut self) {
        for ch in &mut self.prev_window {
            ch.fill(0.0);
        }
        self.prev_block_flag = false;
    }
}

impl Default for VorbisDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = VorbisDecoder::new();
        assert_eq!(decoder.state(), DecoderState::WaitingIdHeader);
    }

    #[test]
    fn test_id_header_parsing() {
        let mut decoder = VorbisDecoder::new();

        // Valid ID header
        let mut header = vec![0u8; 30];
        header[0] = 1; // Type
        header[1..7].copy_from_slice(b"vorbis");
        header[7..11].copy_from_slice(&0u32.to_le_bytes()); // Version
        header[11] = 2; // Channels
        header[12..16].copy_from_slice(&44100u32.to_le_bytes()); // Sample rate
        header[28] = 0x88; // Block sizes: 256 (2^8) and 2048 (2^11)
        header[29] = 1; // Framing

        let result = decoder.decode_packet(&header);
        assert!(result.is_ok());
        assert_eq!(decoder.state(), DecoderState::WaitingCommentHeader);
        assert_eq!(decoder.info().channels, 2);
        assert_eq!(decoder.info().sample_rate, 44100);
    }

    #[test]
    fn test_comment_parsing() {
        let comment = VorbisComment {
            vendor: "Test".into(),
            comments: vec![
                ("TITLE".into(), "My Song".into()),
                ("ARTIST".into(), "Test Artist".into()),
            ],
        };

        assert_eq!(comment.title(), Some("My Song"));
        assert_eq!(comment.artist(), Some("Test Artist"));
        assert_eq!(comment.album(), None);
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = VorbisDecoder::new();
        decoder.reset();
        assert_eq!(decoder.state(), DecoderState::WaitingIdHeader);
        assert_eq!(decoder.samples_decoded(), 0);
    }
}
