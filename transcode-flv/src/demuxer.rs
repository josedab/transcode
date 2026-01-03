//! FLV demuxer implementation.
//!
//! This module provides a demuxer for reading FLV files and extracting
//! audio and video packets.
//!
//! ## Example
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::BufReader;
//! use transcode_flv::FlvDemuxer;
//!
//! let file = File::open("input.flv").unwrap();
//! let reader = BufReader::new(file);
//! let mut demuxer = FlvDemuxer::new(reader).unwrap();
//!
//! // Print stream info
//! if let Some(video) = demuxer.video_info() {
//!     println!("Video: {}x{} @ {}fps", video.width, video.height, video.frame_rate);
//! }
//!
//! if let Some(audio) = demuxer.audio_info() {
//!     println!("Audio: {} Hz, {} channels", audio.sample_rate, audio.channels);
//! }
//!
//! // Read packets
//! while let Ok(Some(packet)) = demuxer.read_packet() {
//!     println!("Packet: stream={}, pts={}ms, size={}",
//!              packet.stream_index, packet.timestamp_ms, packet.data.len());
//! }
//! ```

use crate::amf::{parse_on_metadata, AmfValue};
use crate::audio::{AacConfig, AudioTagHeader, SoundFormat};
use crate::error::{FlvError, Result};
use crate::header::{FlvHeader, FLV_HEADER_SIZE};
use crate::tag::{read_previous_tag_size, TagHeader, TagType, TAG_HEADER_SIZE};
use crate::video::{AvcConfig, HevcConfig, VideoCodec, VideoTagHeader};

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

/// Video stream information.
#[derive(Debug, Clone)]
pub struct VideoInfo {
    /// Video codec.
    pub codec: VideoCodec,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate (from metadata).
    pub frame_rate: f64,
    /// Video bitrate in kbps (from metadata).
    pub bitrate: Option<f64>,
    /// AVC decoder configuration.
    pub avc_config: Option<AvcConfig>,
    /// HEVC decoder configuration.
    pub hevc_config: Option<HevcConfig>,
}

/// Audio stream information.
#[derive(Debug, Clone)]
pub struct AudioInfo {
    /// Audio format/codec.
    pub format: SoundFormat,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bits per sample.
    pub bits_per_sample: u8,
    /// Audio bitrate in kbps (from metadata).
    pub bitrate: Option<f64>,
    /// AAC decoder configuration.
    pub aac_config: Option<AacConfig>,
}

/// Demuxed FLV packet.
#[derive(Debug, Clone)]
pub struct FlvPacket {
    /// Stream index (0 for video, 1 for audio).
    pub stream_index: u32,
    /// Timestamp in milliseconds.
    pub timestamp_ms: u32,
    /// Packet data (without FLV tag header).
    pub data: Vec<u8>,
    /// Whether this is a keyframe (video only).
    pub is_keyframe: bool,
    /// Composition time offset in milliseconds (video only).
    pub composition_time_ms: i32,
    /// Whether this is a sequence header.
    pub is_sequence_header: bool,
}

impl FlvPacket {
    /// Check if this is a video packet.
    pub fn is_video(&self) -> bool {
        self.stream_index == 0
    }

    /// Check if this is an audio packet.
    pub fn is_audio(&self) -> bool {
        self.stream_index == 1
    }
}

/// FLV demuxer.
pub struct FlvDemuxer<R: Read + Seek> {
    /// Input reader.
    reader: R,
    /// FLV header.
    header: FlvHeader,
    /// Metadata from onMetaData.
    metadata: HashMap<String, AmfValue>,
    /// Video stream info.
    video_info: Option<VideoInfo>,
    /// Audio stream info.
    audio_info: Option<AudioInfo>,
    /// Current file position.
    position: u64,
    /// File size (if known).
    file_size: Option<u64>,
    /// Duration in milliseconds (from metadata or last packet).
    duration_ms: Option<u64>,
    /// Last video timestamp for wraparound detection.
    last_video_timestamp: u32,
    /// Last audio timestamp for wraparound detection.
    last_audio_timestamp: u32,
    /// Timestamp wraparound offset for video.
    video_timestamp_offset: u64,
    /// Timestamp wraparound offset for audio.
    audio_timestamp_offset: u64,
    /// Whether we've finished reading.
    eof: bool,
}

impl<R: Read + Seek> FlvDemuxer<R> {
    /// Create a new FLV demuxer.
    pub fn new(mut reader: R) -> Result<Self> {
        // Get file size
        let file_size = reader.seek(SeekFrom::End(0)).ok();
        reader.seek(SeekFrom::Start(0))?;

        // Parse FLV header
        let header = FlvHeader::parse(&mut reader)?;

        // Skip any extra header bytes
        if header.header_size > FLV_HEADER_SIZE {
            let extra = header.header_size - FLV_HEADER_SIZE;
            let mut skip = vec![0u8; extra as usize];
            reader.read_exact(&mut skip)?;
        }

        // Read first previous tag size (should be 0)
        let _first_prev_tag_size = read_previous_tag_size(&mut reader)?;

        let position = header.header_size as u64 + 4;

        let mut demuxer = Self {
            reader,
            header,
            metadata: HashMap::new(),
            video_info: None,
            audio_info: None,
            position,
            file_size,
            duration_ms: None,
            last_video_timestamp: 0,
            last_audio_timestamp: 0,
            video_timestamp_offset: 0,
            audio_timestamp_offset: 0,
            eof: false,
        };

        // Probe for metadata and sequence headers
        demuxer.probe()?;

        Ok(demuxer)
    }

    /// Probe the file for metadata and codec information.
    fn probe(&mut self) -> Result<()> {
        let start_pos = self.position;

        // Read tags until we have metadata and sequence headers
        let mut found_video_seq = false;
        let mut found_audio_seq = false;
        let mut probe_count = 0;
        const MAX_PROBE_TAGS: u32 = 100;

        while probe_count < MAX_PROBE_TAGS && (!found_video_seq || !found_audio_seq) {
            match self.read_tag_header() {
                Ok(Some((tag_header, data))) => {
                    match tag_header.tag_type {
                        TagType::ScriptData => {
                            self.parse_script_data(&data)?;
                        }
                        TagType::Video if !found_video_seq => {
                            if let Some(info) = self.parse_video_config(&data)? {
                                self.video_info = Some(info);
                                found_video_seq = true;
                            }
                        }
                        TagType::Audio if !found_audio_seq => {
                            if let Some(info) = self.parse_audio_config(&data)? {
                                self.audio_info = Some(info);
                                found_audio_seq = true;
                            }
                        }
                        _ => {}
                    }
                    probe_count += 1;
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }

        // Seek back to start
        self.reader.seek(SeekFrom::Start(start_pos))?;
        self.position = start_pos;

        Ok(())
    }

    /// Read a tag header and data.
    fn read_tag_header(&mut self) -> Result<Option<(TagHeader, Vec<u8>)>> {
        if self.eof {
            return Ok(None);
        }

        // Check if we have enough data for a tag header
        if let Some(file_size) = self.file_size {
            if self.position + TAG_HEADER_SIZE as u64 > file_size {
                self.eof = true;
                return Ok(None);
            }
        }

        // Read tag header
        let tag_header = match TagHeader::parse(&mut self.reader) {
            Ok(h) => h,
            Err(FlvError::Io(_)) => {
                self.eof = true;
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        // Read tag data
        let mut data = vec![0u8; tag_header.data_size as usize];
        if self.reader.read_exact(&mut data).is_err() {
            self.eof = true;
            return Ok(None);
        }

        // Read previous tag size
        let _prev_tag_size = read_previous_tag_size(&mut self.reader)?;

        self.position += TAG_HEADER_SIZE as u64 + tag_header.data_size as u64 + 4;

        Ok(Some((tag_header, data)))
    }

    /// Parse script data (metadata).
    fn parse_script_data(&mut self, data: &[u8]) -> Result<()> {
        if let Ok(metadata) = parse_on_metadata(data) {
            // Extract duration
            if let Some(AmfValue::Number(duration)) = metadata.get("duration") {
                self.duration_ms = Some((*duration * 1000.0) as u64);
            }

            self.metadata = metadata;
        }
        Ok(())
    }

    /// Parse video configuration from first video tag.
    fn parse_video_config(&mut self, data: &[u8]) -> Result<Option<VideoInfo>> {
        if data.is_empty() {
            return Ok(None);
        }

        let mut cursor = std::io::Cursor::new(data);
        let video_header = VideoTagHeader::parse(&mut cursor)?;

        if !video_header.is_sequence_header() {
            return Ok(None);
        }

        let header_size = video_header.size();
        let config_data = &data[header_size..];

        let (avc_config, hevc_config) = match video_header.codec_id {
            VideoCodec::Avc => {
                let config = AvcConfig::parse(config_data)?;
                (Some(config), None)
            }
            VideoCodec::Hevc => {
                let config = HevcConfig::parse(config_data)?;
                (None, Some(config))
            }
            _ => (None, None),
        };

        // Get dimensions from metadata or config
        let width = self
            .metadata
            .get("width")
            .and_then(|v| v.as_number())
            .map(|n| n as u32)
            .unwrap_or(0);
        let height = self
            .metadata
            .get("height")
            .and_then(|v| v.as_number())
            .map(|n| n as u32)
            .unwrap_or(0);
        let frame_rate = self
            .metadata
            .get("framerate")
            .and_then(|v| v.as_number())
            .unwrap_or(0.0);
        let bitrate = self
            .metadata
            .get("videodatarate")
            .and_then(|v| v.as_number());

        Ok(Some(VideoInfo {
            codec: video_header.codec_id,
            width,
            height,
            frame_rate,
            bitrate,
            avc_config,
            hevc_config,
        }))
    }

    /// Parse audio configuration from first audio tag.
    fn parse_audio_config(&mut self, data: &[u8]) -> Result<Option<AudioInfo>> {
        if data.is_empty() {
            return Ok(None);
        }

        let mut cursor = std::io::Cursor::new(data);
        let audio_header = AudioTagHeader::parse(&mut cursor)?;

        // For AAC, check if this is a sequence header
        let aac_config = if audio_header.is_aac_sequence_header() {
            let header_size = audio_header.size();
            let config_data = &data[header_size..];
            AacConfig::parse(config_data).ok()
        } else {
            None
        };

        // Determine sample rate and channels
        let sample_rate = if let Some(ref config) = aac_config {
            config.sample_rate()
        } else {
            self.metadata
                .get("audiosamplerate")
                .and_then(|v| v.as_number())
                .map(|n| n as u32)
                .unwrap_or(audio_header.sound_rate.hz())
        };

        let channels = if let Some(ref config) = aac_config {
            config.channels()
        } else {
            audio_header.sound_type.channels()
        };

        let bits_per_sample = self
            .metadata
            .get("audiosamplesize")
            .and_then(|v| v.as_number())
            .map(|n| n as u8)
            .unwrap_or(audio_header.sound_size.bits());

        let bitrate = self
            .metadata
            .get("audiodatarate")
            .and_then(|v| v.as_number());

        Ok(Some(AudioInfo {
            format: audio_header.sound_format,
            sample_rate,
            channels,
            bits_per_sample,
            bitrate,
            aac_config,
        }))
    }

    /// Read the next packet.
    pub fn read_packet(&mut self) -> Result<Option<FlvPacket>> {
        loop {
            let (tag_header, data) = match self.read_tag_header()? {
                Some(t) => t,
                None => return Ok(None),
            };

            let timestamp_ms = tag_header.timestamp_ms();

            match tag_header.tag_type {
                TagType::Video => {
                    // Handle timestamp wraparound
                    if timestamp_ms < self.last_video_timestamp
                        && self.last_video_timestamp - timestamp_ms > 0x80000000
                    {
                        self.video_timestamp_offset += 0x100000000;
                    }
                    self.last_video_timestamp = timestamp_ms;

                    if let Ok(packet) = self.parse_video_packet(timestamp_ms, &data) {
                        return Ok(Some(packet));
                    }
                }
                TagType::Audio => {
                    // Handle timestamp wraparound
                    if timestamp_ms < self.last_audio_timestamp
                        && self.last_audio_timestamp - timestamp_ms > 0x80000000
                    {
                        self.audio_timestamp_offset += 0x100000000;
                    }
                    self.last_audio_timestamp = timestamp_ms;

                    if let Ok(packet) = self.parse_audio_packet(timestamp_ms, &data) {
                        return Ok(Some(packet));
                    }
                }
                TagType::ScriptData => {
                    // Skip script data during playback
                    continue;
                }
            }
        }
    }

    /// Parse a video packet.
    fn parse_video_packet(&self, timestamp_ms: u32, data: &[u8]) -> Result<FlvPacket> {
        if data.is_empty() {
            return Err(FlvError::InvalidTagSize {
                offset: self.position,
                message: "Empty video tag".to_string(),
            });
        }

        let mut cursor = std::io::Cursor::new(data);
        let header = VideoTagHeader::parse(&mut cursor)?;

        let header_size = header.size();
        let payload = data[header_size..].to_vec();

        Ok(FlvPacket {
            stream_index: 0,
            timestamp_ms,
            data: payload,
            is_keyframe: header.is_keyframe(),
            composition_time_ms: header.composition_time,
            is_sequence_header: header.is_sequence_header(),
        })
    }

    /// Parse an audio packet.
    fn parse_audio_packet(&self, timestamp_ms: u32, data: &[u8]) -> Result<FlvPacket> {
        if data.is_empty() {
            return Err(FlvError::InvalidTagSize {
                offset: self.position,
                message: "Empty audio tag".to_string(),
            });
        }

        let mut cursor = std::io::Cursor::new(data);
        let header = AudioTagHeader::parse(&mut cursor)?;

        let header_size = header.size();
        let payload = data[header_size..].to_vec();

        Ok(FlvPacket {
            stream_index: 1,
            timestamp_ms,
            data: payload,
            is_keyframe: false,
            composition_time_ms: 0,
            is_sequence_header: header.is_aac_sequence_header(),
        })
    }

    /// Get the FLV header.
    pub fn header(&self) -> &FlvHeader {
        &self.header
    }

    /// Get video stream information.
    pub fn video_info(&self) -> Option<&VideoInfo> {
        self.video_info.as_ref()
    }

    /// Get audio stream information.
    pub fn audio_info(&self) -> Option<&AudioInfo> {
        self.audio_info.as_ref()
    }

    /// Get metadata.
    pub fn metadata(&self) -> &HashMap<String, AmfValue> {
        &self.metadata
    }

    /// Get the duration in milliseconds.
    pub fn duration_ms(&self) -> Option<u64> {
        self.duration_ms
    }

    /// Get the duration in seconds.
    pub fn duration(&self) -> Option<f64> {
        self.duration_ms.map(|ms| ms as f64 / 1000.0)
    }

    /// Check if the file has video.
    pub fn has_video(&self) -> bool {
        self.header.has_video && self.video_info.is_some()
    }

    /// Check if the file has audio.
    pub fn has_audio(&self) -> bool {
        self.header.has_audio && self.audio_info.is_some()
    }

    /// Get the current position in the file.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get the file size.
    pub fn file_size(&self) -> Option<u64> {
        self.file_size
    }

    /// Seek to a timestamp.
    ///
    /// Note: FLV doesn't have a seek index, so this may seek to an approximate position.
    pub fn seek(&mut self, timestamp_ms: u64) -> Result<()> {
        // Simple seek: estimate position based on file size and duration
        if let (Some(file_size), Some(duration_ms)) = (self.file_size, self.duration_ms) {
            if duration_ms == 0 {
                return Ok(());
            }

            let estimated_pos = (file_size as f64 * timestamp_ms as f64 / duration_ms as f64) as u64;
            let seek_pos = estimated_pos.max(FLV_HEADER_SIZE as u64 + 4);

            self.reader.seek(SeekFrom::Start(seek_pos))?;
            self.position = seek_pos;

            // Find next sync point (tag start)
            self.resync()?;
        }

        Ok(())
    }

    /// Resync to the next valid tag.
    fn resync(&mut self) -> Result<()> {
        // Read until we find a valid tag
        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 10000;

        while attempts < MAX_ATTEMPTS {
            let current_pos = self.reader.stream_position()?;

            // Try to read a tag header
            if let Ok(Some(_)) = self.peek_tag_type() {
                self.position = current_pos;
                return Ok(());
            }

            // Advance by one byte
            self.reader.seek(SeekFrom::Current(1))?;
            attempts += 1;
        }

        Err(FlvError::SeekFailed("Could not find valid tag".to_string()))
    }

    /// Peek at the next tag type without consuming it.
    fn peek_tag_type(&mut self) -> Result<Option<TagType>> {
        let mut buf = [0u8; 1];
        match self.reader.read_exact(&mut buf) {
            Ok(()) => {
                self.reader.seek(SeekFrom::Current(-1))?;
                Ok(TagType::from_u8(buf[0]))
            }
            Err(_) => Ok(None),
        }
    }

    /// Get the underlying reader.
    pub fn into_inner(self) -> R {
        self.reader
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::muxer::{AudioStreamConfig, FlvMuxer, MuxerConfig, VideoStreamConfig};
    use std::io::Cursor;

    fn create_test_flv() -> Vec<u8> {
        let mut buffer = Vec::new();
        let config = MuxerConfig::new().with_metadata(true);
        let mut muxer = FlvMuxer::new(&mut buffer, config);

        muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));
        muxer.set_audio_config(AudioStreamConfig::aac(48000, 2));
        muxer.write_header().unwrap();

        // Write video sequence header
        let avc_config = vec![
            0x01, 0x64, 0x00, 0x1F, 0xFF, 0xE1, 0x00, 0x04, 0x67, 0x64, 0x00, 0x1F, 0x01, 0x00,
            0x02, 0x68, 0xEF,
        ];
        muxer.write_video_sequence_header(&avc_config).unwrap();

        // Write audio sequence header
        let aac_config = vec![0x12, 0x10]; // AAC-LC, 44100, stereo
        muxer.write_audio_sequence_header(&aac_config).unwrap();

        // Write some packets
        muxer
            .write_video_packet(0, &[0xAB; 100], true, 0)
            .unwrap();
        muxer.write_audio_packet(0, &[0xCD; 50]).unwrap();
        muxer
            .write_video_packet(33, &[0xEF; 80], false, 0)
            .unwrap();
        muxer.write_audio_packet(20, &[0x12; 50]).unwrap();

        muxer.finalize().unwrap();

        buffer
    }

    #[test]
    fn test_demuxer_creation() {
        let data = create_test_flv();
        let cursor = Cursor::new(data);

        let demuxer = FlvDemuxer::new(cursor).unwrap();

        assert!(demuxer.header().has_video);
        assert!(demuxer.header().has_audio);
    }

    #[test]
    fn test_demuxer_stream_info() {
        let data = create_test_flv();
        let cursor = Cursor::new(data);

        let demuxer = FlvDemuxer::new(cursor).unwrap();

        // Check video info
        let video = demuxer.video_info().unwrap();
        assert_eq!(video.codec, VideoCodec::Avc);
        assert!(video.avc_config.is_some());

        // Check audio info
        let audio = demuxer.audio_info().unwrap();
        assert_eq!(audio.format, SoundFormat::Aac);
        assert!(audio.aac_config.is_some());
    }

    #[test]
    fn test_demuxer_read_packets() {
        let data = create_test_flv();
        let cursor = Cursor::new(data);

        let mut demuxer = FlvDemuxer::new(cursor).unwrap();

        // Read all packets
        let mut video_count = 0;
        let mut audio_count = 0;

        while let Ok(Some(packet)) = demuxer.read_packet() {
            if packet.is_video() {
                video_count += 1;
            } else if packet.is_audio() {
                audio_count += 1;
            }
        }

        // Should have video and audio packets (including sequence headers)
        assert!(video_count >= 2);
        assert!(audio_count >= 2);
    }

    #[test]
    fn test_demuxer_metadata() {
        let data = create_test_flv();
        let cursor = Cursor::new(data);

        let demuxer = FlvDemuxer::new(cursor).unwrap();
        let metadata = demuxer.metadata();

        // Check that we have some metadata
        assert!(metadata.contains_key("width") || metadata.contains_key("encoder"));
    }

    #[test]
    fn test_demuxer_helpers() {
        let data = create_test_flv();
        let cursor = Cursor::new(data);

        let demuxer = FlvDemuxer::new(cursor).unwrap();

        assert!(demuxer.has_video());
        assert!(demuxer.has_audio());
        assert!(demuxer.file_size().is_some());
    }

    #[test]
    fn test_flv_packet() {
        let video_packet = FlvPacket {
            stream_index: 0,
            timestamp_ms: 100,
            data: vec![0xAB; 50],
            is_keyframe: true,
            composition_time_ms: 33,
            is_sequence_header: false,
        };

        assert!(video_packet.is_video());
        assert!(!video_packet.is_audio());

        let audio_packet = FlvPacket {
            stream_index: 1,
            timestamp_ms: 100,
            data: vec![0xCD; 30],
            is_keyframe: false,
            composition_time_ms: 0,
            is_sequence_header: false,
        };

        assert!(!audio_packet.is_video());
        assert!(audio_packet.is_audio());
    }

    #[test]
    fn test_invalid_flv_signature() {
        let data = b"ABC\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00";
        let cursor = Cursor::new(&data[..]);

        let result = FlvDemuxer::new(cursor);
        assert!(matches!(result, Err(FlvError::InvalidSignature(_))));
    }
}
