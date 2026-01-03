//! MKV/Matroska muxer implementation.
//!
//! This module provides muxing capabilities for Matroska and WebM containers.

use crate::ebml::{self, EbmlHeader};
use crate::elements::{self, *};
use crate::error::{MkvError, Result};

use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};

use transcode_core::{AudioCodec, Packet, PacketFlags, VideoCodec};

/// Default timecode scale (1 millisecond in nanoseconds).
const DEFAULT_TIMECODE_SCALE: u64 = 1_000_000;

/// Maximum cluster duration (5 seconds in timecode units).
const MAX_CLUSTER_DURATION: u64 = 5000;

/// Target cluster size in bytes.
const TARGET_CLUSTER_SIZE: usize = 5 * 1024 * 1024; // 5 MB

/// Video track configuration for muxer.
#[derive(Debug, Clone)]
pub struct VideoTrackConfig {
    /// Track number (1-based).
    pub track_number: u64,
    /// Video codec.
    pub codec: VideoCodec,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Display width (for aspect ratio).
    pub display_width: Option<u32>,
    /// Display height (for aspect ratio).
    pub display_height: Option<u32>,
    /// Frame rate (frames per second).
    pub frame_rate: Option<f64>,
    /// Codec private data (e.g., AV1 config, VP9 config).
    pub codec_private: Option<Vec<u8>>,
    /// Track name.
    pub name: Option<String>,
    /// Language code.
    pub language: Option<String>,
    /// Is default track.
    pub is_default: bool,
}

impl VideoTrackConfig {
    /// Create a new video track config.
    pub fn new(track_number: u64, codec: VideoCodec, width: u32, height: u32) -> Self {
        Self {
            track_number,
            codec,
            width,
            height,
            display_width: None,
            display_height: None,
            frame_rate: None,
            codec_private: None,
            name: None,
            language: None,
            is_default: true,
        }
    }
}

/// Audio track configuration for muxer.
#[derive(Debug, Clone)]
pub struct AudioTrackConfig {
    /// Track number (1-based).
    pub track_number: u64,
    /// Audio codec.
    pub codec: AudioCodec,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of channels.
    pub channels: u32,
    /// Bits per sample.
    pub bit_depth: Option<u32>,
    /// Codec private data (e.g., Opus headers, Vorbis headers).
    pub codec_private: Option<Vec<u8>>,
    /// Track name.
    pub name: Option<String>,
    /// Language code.
    pub language: Option<String>,
    /// Is default track.
    pub is_default: bool,
    /// Codec delay in nanoseconds.
    pub codec_delay: u64,
    /// Seek pre-roll in nanoseconds.
    pub seek_pre_roll: u64,
}

impl AudioTrackConfig {
    /// Create a new audio track config.
    pub fn new(track_number: u64, codec: AudioCodec, sample_rate: f64, channels: u32) -> Self {
        Self {
            track_number,
            codec,
            sample_rate,
            channels,
            bit_depth: None,
            codec_private: None,
            name: None,
            language: None,
            is_default: true,
            codec_delay: 0,
            seek_pre_roll: 0,
        }
    }
}

/// Subtitle track configuration for muxer.
#[derive(Debug, Clone)]
pub struct SubtitleTrackConfig {
    /// Track number (1-based).
    pub track_number: u64,
    /// Codec ID string.
    pub codec_id: String,
    /// Codec private data.
    pub codec_private: Option<Vec<u8>>,
    /// Track name.
    pub name: Option<String>,
    /// Language code.
    pub language: Option<String>,
    /// Is default track.
    pub is_default: bool,
    /// Is forced track.
    pub is_forced: bool,
}

impl SubtitleTrackConfig {
    /// Create a new subtitle track config.
    pub fn new(track_number: u64, codec_id: &str) -> Self {
        Self {
            track_number,
            codec_id: codec_id.to_string(),
            codec_private: None,
            name: None,
            language: None,
            is_default: false,
            is_forced: false,
        }
    }
}

/// Track configuration enum.
#[derive(Debug, Clone)]
pub enum TrackConfig {
    /// Video track.
    Video(VideoTrackConfig),
    /// Audio track.
    Audio(AudioTrackConfig),
    /// Subtitle track.
    Subtitle(SubtitleTrackConfig),
}

impl TrackConfig {
    /// Get the track number.
    pub fn track_number(&self) -> u64 {
        match self {
            TrackConfig::Video(v) => v.track_number,
            TrackConfig::Audio(a) => a.track_number,
            TrackConfig::Subtitle(s) => s.track_number,
        }
    }
}

/// A cue point for the muxer to write.
#[derive(Debug, Clone)]
struct MuxerCuePoint {
    /// Timestamp in timecode units.
    time: u64,
    /// Track number.
    track: u64,
    /// Cluster position (relative to segment).
    cluster_position: u64,
    /// Relative position within cluster.
    relative_position: Option<u64>,
}

/// Muxer configuration.
#[derive(Debug, Clone)]
pub struct MuxerConfig {
    /// Document type ("matroska" or "webm").
    pub doc_type: String,
    /// Timecode scale (nanoseconds per timecode unit).
    pub timecode_scale: u64,
    /// Segment title.
    pub title: Option<String>,
    /// Muxing application name.
    pub muxing_app: String,
    /// Writing application name.
    pub writing_app: String,
    /// Whether to write cues.
    pub write_cues: bool,
    /// Whether to write chapters.
    pub write_chapters: bool,
}

impl Default for MuxerConfig {
    fn default() -> Self {
        Self {
            doc_type: "matroska".to_string(),
            timecode_scale: DEFAULT_TIMECODE_SCALE,
            title: None,
            muxing_app: "transcode-mkv".to_string(),
            writing_app: "transcode-mkv".to_string(),
            write_cues: true,
            write_chapters: true,
        }
    }
}

impl MuxerConfig {
    /// Create a WebM muxer config.
    pub fn webm() -> Self {
        Self {
            doc_type: "webm".to_string(),
            ..Default::default()
        }
    }
}

/// Chapter entry for muxer.
#[derive(Debug, Clone)]
pub struct MuxerChapter {
    /// Chapter UID.
    pub uid: u64,
    /// Start time in nanoseconds.
    pub start_time_ns: u64,
    /// End time in nanoseconds.
    pub end_time_ns: Option<u64>,
    /// Chapter title.
    pub title: String,
    /// Language code.
    pub language: Option<String>,
}

/// MKV muxer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MuxerState {
    /// Initial state.
    Initial,
    /// Header written, ready for packets.
    Ready,
    /// Writing packets.
    Writing,
    /// Finalized.
    Finalized,
}

/// MKV muxer.
pub struct MkvMuxer<W: Write + Seek> {
    writer: W,
    config: MuxerConfig,
    state: MuxerState,
    /// Tracks to be written.
    tracks: Vec<TrackConfig>,
    /// Track number to stream index mapping.
    track_map: HashMap<u32, u64>,
    /// Chapters to write.
    chapters: Vec<MuxerChapter>,
    /// Cue points collected during muxing.
    cue_points: Vec<MuxerCuePoint>,
    /// Segment start position.
    segment_start: u64,
    /// Segment size placeholder position.
    segment_size_pos: u64,
    /// Current cluster start position.
    cluster_start: u64,
    /// Current cluster timestamp.
    cluster_timestamp: u64,
    /// Current cluster size.
    cluster_size: usize,
    /// First timestamp of current cluster.
    cluster_first_timestamp: Option<u64>,
    /// Whether we're in a cluster.
    in_cluster: bool,
    /// Maximum timestamp seen.
    max_timestamp: u64,
    /// Seek head placeholder position.
    seek_head_pos: u64,
    /// Seek head size reserved.
    seek_head_size: u64,
    /// Cues position (relative to segment).
    cues_position: Option<u64>,
}

impl<W: Write + Seek> MkvMuxer<W> {
    /// Create a new MKV muxer.
    pub fn new(writer: W, config: MuxerConfig) -> Self {
        Self {
            writer,
            config,
            state: MuxerState::Initial,
            tracks: Vec::new(),
            track_map: HashMap::new(),
            chapters: Vec::new(),
            cue_points: Vec::new(),
            segment_start: 0,
            segment_size_pos: 0,
            cluster_start: 0,
            cluster_timestamp: 0,
            cluster_size: 0,
            cluster_first_timestamp: None,
            in_cluster: false,
            max_timestamp: 0,
            seek_head_pos: 0,
            seek_head_size: 0,
            cues_position: None,
        }
    }

    /// Add a video track.
    pub fn add_video_track(&mut self, config: VideoTrackConfig) -> Result<()> {
        if self.state != MuxerState::Initial {
            return Err(MkvError::Other(
                "Cannot add tracks after header is written".to_string(),
            ));
        }

        // Validate WebM codec compatibility
        if self.config.doc_type == "webm" {
            let codec_id = elements::video_codec_to_mkv_id(config.codec);
            if !elements::is_webm_compatible_codec(codec_id) {
                return Err(MkvError::InvalidWebM(format!(
                    "Codec {} is not WebM-compatible",
                    codec_id
                )));
            }
        }

        self.track_map
            .insert(config.track_number as u32, config.track_number);
        self.tracks.push(TrackConfig::Video(config));
        Ok(())
    }

    /// Add an audio track.
    pub fn add_audio_track(&mut self, config: AudioTrackConfig) -> Result<()> {
        if self.state != MuxerState::Initial {
            return Err(MkvError::Other(
                "Cannot add tracks after header is written".to_string(),
            ));
        }

        // Validate WebM codec compatibility
        if self.config.doc_type == "webm" {
            let codec_id = elements::audio_codec_to_mkv_id(config.codec);
            if !elements::is_webm_compatible_codec(codec_id) {
                return Err(MkvError::InvalidWebM(format!(
                    "Codec {} is not WebM-compatible",
                    codec_id
                )));
            }
        }

        self.track_map
            .insert(config.track_number as u32, config.track_number);
        self.tracks.push(TrackConfig::Audio(config));
        Ok(())
    }

    /// Add a subtitle track.
    pub fn add_subtitle_track(&mut self, config: SubtitleTrackConfig) -> Result<()> {
        if self.state != MuxerState::Initial {
            return Err(MkvError::Other(
                "Cannot add tracks after header is written".to_string(),
            ));
        }

        if self.config.doc_type == "webm" {
            return Err(MkvError::InvalidWebM(
                "Subtitle tracks are not supported in WebM".to_string(),
            ));
        }

        self.track_map
            .insert(config.track_number as u32, config.track_number);
        self.tracks.push(TrackConfig::Subtitle(config));
        Ok(())
    }

    /// Add a chapter.
    pub fn add_chapter(&mut self, chapter: MuxerChapter) {
        self.chapters.push(chapter);
    }

    /// Write the file header.
    pub fn write_header(&mut self) -> Result<()> {
        if self.state != MuxerState::Initial {
            return Err(MkvError::Other("Header already written".to_string()));
        }

        if self.tracks.is_empty() {
            return Err(MkvError::Other("No tracks configured".to_string()));
        }

        // Write EBML header
        self.write_ebml_header()?;

        // Write Segment with unknown size
        self.write_element_id(SEGMENT)?;
        self.segment_size_pos = self.position()?;
        self.write_unknown_size(8)?;
        self.segment_start = self.position()?;

        // Reserve space for SeekHead
        self.seek_head_pos = self.position()? - self.segment_start;
        self.seek_head_size = 100; // Reserve 100 bytes
        self.write_void(self.seek_head_size as usize)?;

        // Write Info
        self.write_info()?;

        // Write Tracks
        self.write_tracks()?;

        self.state = MuxerState::Ready;
        Ok(())
    }

    /// Write the EBML header.
    fn write_ebml_header(&mut self) -> Result<()> {
        let header = EbmlHeader {
            doc_type: self.config.doc_type.clone(),
            doc_type_version: 4,
            doc_type_read_version: 2,
            ..Default::default()
        };

        // Build EBML header content
        let mut content = Vec::new();

        // EBMLVersion = 1
        self.write_element_to_vec(&mut content, EBML_VERSION, &[1])?;
        // EBMLReadVersion = 1
        self.write_element_to_vec(&mut content, EBML_READ_VERSION, &[1])?;
        // EBMLMaxIDLength = 4
        self.write_element_to_vec(&mut content, EBML_MAX_ID_LENGTH, &[4])?;
        // EBMLMaxSizeLength = 8
        self.write_element_to_vec(&mut content, EBML_MAX_SIZE_LENGTH, &[8])?;
        // DocType
        self.write_element_to_vec(&mut content, DOC_TYPE, header.doc_type.as_bytes())?;
        // DocTypeVersion
        self.write_element_to_vec(&mut content, DOC_TYPE_VERSION, &[header.doc_type_version as u8])?;
        // DocTypeReadVersion
        self.write_element_to_vec(
            &mut content,
            DOC_TYPE_READ_VERSION,
            &[header.doc_type_read_version as u8],
        )?;

        // Write EBML header element
        self.write_element_id(EBML)?;
        ebml::write_vint(&mut self.writer, content.len() as u64)?;
        self.writer.write_all(&content)?;

        Ok(())
    }

    /// Write Info element.
    fn write_info(&mut self) -> Result<()> {
        let mut content = Vec::new();

        // TimecodeScale
        self.write_uint_element_to_vec(&mut content, TIMECODE_SCALE, self.config.timecode_scale)?;

        // MuxingApp
        self.write_element_to_vec(&mut content, MUXING_APP, self.config.muxing_app.as_bytes())?;

        // WritingApp
        self.write_element_to_vec(&mut content, WRITING_APP, self.config.writing_app.as_bytes())?;

        // Title (optional)
        if let Some(ref title) = self.config.title {
            self.write_element_to_vec(&mut content, TITLE, title.as_bytes())?;
        }

        // Write Info element
        self.write_element_id(INFO)?;
        ebml::write_vint(&mut self.writer, content.len() as u64)?;
        self.writer.write_all(&content)?;

        Ok(())
    }

    /// Write Tracks element.
    fn write_tracks(&mut self) -> Result<()> {
        let mut content = Vec::new();

        for track in &self.tracks.clone() {
            let track_content = self.build_track_entry(track)?;
            self.write_element_to_vec(&mut content, TRACK_ENTRY, &track_content)?;
        }

        self.write_element_id(TRACKS)?;
        ebml::write_vint(&mut self.writer, content.len() as u64)?;
        self.writer.write_all(&content)?;

        Ok(())
    }

    /// Build a TrackEntry element.
    fn build_track_entry(&self, track: &TrackConfig) -> Result<Vec<u8>> {
        let mut content = Vec::new();

        match track {
            TrackConfig::Video(v) => {
                // TrackNumber
                self.write_uint_element_to_vec(&mut content, TRACK_NUMBER, v.track_number)?;
                // TrackUID
                self.write_uint_element_to_vec(&mut content, TRACK_UID, v.track_number)?;
                // TrackType = 1 (video)
                self.write_uint_element_to_vec(&mut content, TRACK_TYPE, TRACK_TYPE_VIDEO as u64)?;
                // CodecID
                let codec_id = elements::video_codec_to_mkv_id(v.codec);
                self.write_element_to_vec(&mut content, CODEC_ID, codec_id.as_bytes())?;
                // CodecPrivate (optional)
                if let Some(ref private) = v.codec_private {
                    self.write_element_to_vec(&mut content, CODEC_PRIVATE, private)?;
                }
                // FlagDefault
                self.write_uint_element_to_vec(&mut content, FLAG_DEFAULT, v.is_default as u64)?;
                // Name (optional)
                if let Some(ref name) = v.name {
                    self.write_element_to_vec(&mut content, NAME, name.as_bytes())?;
                }
                // Language (optional)
                if let Some(ref lang) = v.language {
                    self.write_element_to_vec(&mut content, LANGUAGE, lang.as_bytes())?;
                }
                // DefaultDuration (optional)
                if let Some(fps) = v.frame_rate {
                    let duration_ns = (1_000_000_000.0 / fps) as u64;
                    self.write_uint_element_to_vec(&mut content, DEFAULT_DURATION, duration_ns)?;
                }
                // Video element
                let video_content = self.build_video_element(v)?;
                self.write_element_to_vec(&mut content, VIDEO, &video_content)?;
            }
            TrackConfig::Audio(a) => {
                // TrackNumber
                self.write_uint_element_to_vec(&mut content, TRACK_NUMBER, a.track_number)?;
                // TrackUID
                self.write_uint_element_to_vec(&mut content, TRACK_UID, a.track_number)?;
                // TrackType = 2 (audio)
                self.write_uint_element_to_vec(&mut content, TRACK_TYPE, TRACK_TYPE_AUDIO as u64)?;
                // CodecID
                let codec_id = elements::audio_codec_to_mkv_id(a.codec);
                self.write_element_to_vec(&mut content, CODEC_ID, codec_id.as_bytes())?;
                // CodecPrivate (optional)
                if let Some(ref private) = a.codec_private {
                    self.write_element_to_vec(&mut content, CODEC_PRIVATE, private)?;
                }
                // FlagDefault
                self.write_uint_element_to_vec(&mut content, FLAG_DEFAULT, a.is_default as u64)?;
                // Name (optional)
                if let Some(ref name) = a.name {
                    self.write_element_to_vec(&mut content, NAME, name.as_bytes())?;
                }
                // Language (optional)
                if let Some(ref lang) = a.language {
                    self.write_element_to_vec(&mut content, LANGUAGE, lang.as_bytes())?;
                }
                // CodecDelay
                if a.codec_delay > 0 {
                    self.write_uint_element_to_vec(&mut content, CODEC_DELAY, a.codec_delay)?;
                }
                // SeekPreRoll
                if a.seek_pre_roll > 0 {
                    self.write_uint_element_to_vec(&mut content, SEEK_PRE_ROLL, a.seek_pre_roll)?;
                }
                // Audio element
                let audio_content = self.build_audio_element(a)?;
                self.write_element_to_vec(&mut content, AUDIO, &audio_content)?;
            }
            TrackConfig::Subtitle(s) => {
                // TrackNumber
                self.write_uint_element_to_vec(&mut content, TRACK_NUMBER, s.track_number)?;
                // TrackUID
                self.write_uint_element_to_vec(&mut content, TRACK_UID, s.track_number)?;
                // TrackType = 17 (subtitle)
                self.write_uint_element_to_vec(&mut content, TRACK_TYPE, TRACK_TYPE_SUBTITLE as u64)?;
                // CodecID
                self.write_element_to_vec(&mut content, CODEC_ID, s.codec_id.as_bytes())?;
                // CodecPrivate (optional)
                if let Some(ref private) = s.codec_private {
                    self.write_element_to_vec(&mut content, CODEC_PRIVATE, private)?;
                }
                // FlagDefault
                self.write_uint_element_to_vec(&mut content, FLAG_DEFAULT, s.is_default as u64)?;
                // FlagForced
                self.write_uint_element_to_vec(&mut content, FLAG_FORCED, s.is_forced as u64)?;
                // Name (optional)
                if let Some(ref name) = s.name {
                    self.write_element_to_vec(&mut content, NAME, name.as_bytes())?;
                }
                // Language (optional)
                if let Some(ref lang) = s.language {
                    self.write_element_to_vec(&mut content, LANGUAGE, lang.as_bytes())?;
                }
            }
        }

        Ok(content)
    }

    /// Build Video element content.
    fn build_video_element(&self, v: &VideoTrackConfig) -> Result<Vec<u8>> {
        let mut content = Vec::new();

        self.write_uint_element_to_vec(&mut content, PIXEL_WIDTH, v.width as u64)?;
        self.write_uint_element_to_vec(&mut content, PIXEL_HEIGHT, v.height as u64)?;

        if let Some(dw) = v.display_width {
            self.write_uint_element_to_vec(&mut content, DISPLAY_WIDTH, dw as u64)?;
        }
        if let Some(dh) = v.display_height {
            self.write_uint_element_to_vec(&mut content, DISPLAY_HEIGHT, dh as u64)?;
        }

        Ok(content)
    }

    /// Build Audio element content.
    fn build_audio_element(&self, a: &AudioTrackConfig) -> Result<Vec<u8>> {
        let mut content = Vec::new();

        self.write_float_element_to_vec(&mut content, SAMPLING_FREQUENCY, a.sample_rate)?;
        self.write_uint_element_to_vec(&mut content, CHANNELS, a.channels as u64)?;

        if let Some(depth) = a.bit_depth {
            self.write_uint_element_to_vec(&mut content, BIT_DEPTH, depth as u64)?;
        }

        Ok(content)
    }

    /// Write a packet to the muxer.
    pub fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.state == MuxerState::Initial {
            self.write_header()?;
        }

        if self.state == MuxerState::Finalized {
            return Err(MkvError::Other("Muxer already finalized".to_string()));
        }

        self.state = MuxerState::Writing;

        // Get track number from stream index
        let track_number = self
            .track_map
            .get(&packet.stream_index)
            .copied()
            .unwrap_or(packet.stream_index as u64 + 1);

        // Convert timestamp to timecode units
        let timestamp_ns = packet
            .pts
            .to_millis()
            .map(|ms| (ms as u64) * 1_000_000)
            .unwrap_or(0);
        let timecode = timestamp_ns / self.config.timecode_scale;

        // Update max timestamp
        self.max_timestamp = self.max_timestamp.max(timecode);

        // Check if we need a new cluster
        let need_new_cluster = !self.in_cluster
            || packet.is_keyframe()
            || self.cluster_size > TARGET_CLUSTER_SIZE
            || timecode.saturating_sub(self.cluster_timestamp) > MAX_CLUSTER_DURATION;

        if need_new_cluster {
            if self.in_cluster {
                self.finish_cluster()?;
            }
            self.start_cluster(timecode)?;

            // Add cue point for keyframes
            if packet.is_keyframe() && self.config.write_cues {
                self.cue_points.push(MuxerCuePoint {
                    time: timecode,
                    track: track_number,
                    cluster_position: self.cluster_start - self.segment_start,
                    relative_position: None,
                });
            }
        }

        // Write SimpleBlock
        self.write_simple_block(packet, track_number, timecode)?;

        Ok(())
    }

    /// Start a new cluster.
    fn start_cluster(&mut self, timestamp: u64) -> Result<()> {
        self.cluster_start = self.position()?;
        self.cluster_timestamp = timestamp;
        self.cluster_size = 0;
        self.cluster_first_timestamp = Some(timestamp);
        self.in_cluster = true;

        // Write Cluster element with unknown size
        self.write_element_id(CLUSTER)?;
        self.write_unknown_size(4)?;

        // Write Timestamp
        self.write_uint_element(TIMESTAMP, timestamp)?;

        Ok(())
    }

    /// Finish the current cluster.
    fn finish_cluster(&mut self) -> Result<()> {
        if !self.in_cluster {
            return Ok(());
        }

        let cluster_end = self.position()?;
        let cluster_content_size = cluster_end - self.cluster_start - 5 - 4; // ID + size header

        // Seek back and write actual size
        self.writer.seek(SeekFrom::Start(self.cluster_start + 4))?;
        // Write 4-byte size (was unknown)
        let size_bytes = [
            0x10 | ((cluster_content_size >> 24) & 0x0F) as u8,
            ((cluster_content_size >> 16) & 0xFF) as u8,
            ((cluster_content_size >> 8) & 0xFF) as u8,
            (cluster_content_size & 0xFF) as u8,
        ];
        self.writer.write_all(&size_bytes)?;

        // Seek back to end
        self.writer.seek(SeekFrom::Start(cluster_end))?;

        self.in_cluster = false;
        Ok(())
    }

    /// Write a SimpleBlock.
    fn write_simple_block(
        &mut self,
        packet: &Packet,
        track_number: u64,
        timecode: u64,
    ) -> Result<()> {
        // Calculate relative timestamp
        let relative_ts = (timecode as i64 - self.cluster_timestamp as i64) as i16;

        // Build block header
        let mut header = Vec::new();

        // Track number (VINT)
        ebml::write_vint(&mut header, track_number)?;

        // Relative timestamp (signed 16-bit big-endian)
        header.extend_from_slice(&relative_ts.to_be_bytes());

        // Flags
        let mut flags = 0u8;
        if packet.is_keyframe() {
            flags |= 0x80; // Keyframe
        }
        if packet.flags.contains(PacketFlags::DISPOSABLE) {
            flags |= 0x01; // Discardable
        }
        // No lacing (lacing = 0)
        header.push(flags);

        // Calculate total block size
        let block_size = header.len() + packet.size();

        // Write SimpleBlock element
        self.write_element_id(SIMPLE_BLOCK)?;
        ebml::write_vint(&mut self.writer, block_size as u64)?;
        self.writer.write_all(&header)?;
        self.writer.write_all(packet.data())?;

        self.cluster_size += 4 + ebml::vint_length(block_size as u64) + block_size;

        Ok(())
    }

    /// Finalize the muxer and write trailing data.
    pub fn finalize(&mut self) -> Result<()> {
        if self.state == MuxerState::Finalized {
            return Ok(());
        }

        if self.state == MuxerState::Initial {
            return Err(MkvError::Other("No data written".to_string()));
        }

        // Finish current cluster
        if self.in_cluster {
            self.finish_cluster()?;
        }

        // Write Cues
        if self.config.write_cues && !self.cue_points.is_empty() {
            self.cues_position = Some(self.position()? - self.segment_start);
            self.write_cues()?;
        }

        // Write Chapters
        if self.config.write_chapters && !self.chapters.is_empty() {
            self.write_chapters()?;
        }

        // Update segment size
        let segment_end = self.position()?;
        let segment_size = segment_end - self.segment_start;

        self.writer.seek(SeekFrom::Start(self.segment_size_pos))?;
        // Write 8-byte size
        let mut size_bytes = [0u8; 8];
        size_bytes[0] = 0x01; // 8-byte VINT marker
        for (i, byte) in segment_size.to_be_bytes()[1..].iter().enumerate() {
            size_bytes[i + 1] = *byte;
        }
        self.writer.write_all(&size_bytes)?;

        // Write SeekHead
        self.write_seek_head()?;

        self.state = MuxerState::Finalized;
        Ok(())
    }

    /// Write Cues element.
    fn write_cues(&mut self) -> Result<()> {
        let mut content = Vec::new();

        for cue in &self.cue_points.clone() {
            let mut cue_content = Vec::new();

            // CueTime
            self.write_uint_element_to_vec(&mut cue_content, CUE_TIME, cue.time)?;

            // CueTrackPositions
            let mut pos_content = Vec::new();
            self.write_uint_element_to_vec(&mut pos_content, CUE_TRACK, cue.track)?;
            self.write_uint_element_to_vec(
                &mut pos_content,
                CUE_CLUSTER_POSITION,
                cue.cluster_position,
            )?;
            if let Some(rel_pos) = cue.relative_position {
                self.write_uint_element_to_vec(&mut pos_content, CUE_RELATIVE_POSITION, rel_pos)?;
            }

            self.write_element_to_vec(&mut cue_content, CUE_TRACK_POSITIONS, &pos_content)?;
            self.write_element_to_vec(&mut content, CUE_POINT, &cue_content)?;
        }

        self.write_element_id(CUES)?;
        ebml::write_vint(&mut self.writer, content.len() as u64)?;
        self.writer.write_all(&content)?;

        Ok(())
    }

    /// Write Chapters element.
    fn write_chapters(&mut self) -> Result<()> {
        let mut edition_content = Vec::new();

        // EditionUID
        self.write_uint_element_to_vec(&mut edition_content, EDITION_UID, 1)?;
        // EditionFlagDefault
        self.write_uint_element_to_vec(&mut edition_content, EDITION_FLAG_DEFAULT, 1)?;

        for chapter in &self.chapters.clone() {
            let mut chapter_content = Vec::new();

            // ChapterUID
            self.write_uint_element_to_vec(&mut chapter_content, CHAPTER_UID, chapter.uid)?;
            // ChapterTimeStart
            self.write_uint_element_to_vec(
                &mut chapter_content,
                CHAPTER_TIME_START,
                chapter.start_time_ns,
            )?;
            // ChapterTimeEnd (optional)
            if let Some(end) = chapter.end_time_ns {
                self.write_uint_element_to_vec(&mut chapter_content, CHAPTER_TIME_END, end)?;
            }
            // ChapterFlagEnabled
            self.write_uint_element_to_vec(&mut chapter_content, CHAPTER_FLAG_ENABLED, 1)?;

            // ChapterDisplay
            let mut display_content = Vec::new();
            self.write_element_to_vec(&mut display_content, CHAP_STRING, chapter.title.as_bytes())?;
            if let Some(ref lang) = chapter.language {
                self.write_element_to_vec(&mut display_content, CHAP_LANGUAGE, lang.as_bytes())?;
            }
            self.write_element_to_vec(&mut chapter_content, CHAPTER_DISPLAY, &display_content)?;

            self.write_element_to_vec(&mut edition_content, CHAPTER_ATOM, &chapter_content)?;
        }

        let mut chapters_content = Vec::new();
        self.write_element_to_vec(&mut chapters_content, EDITION_ENTRY, &edition_content)?;

        self.write_element_id(CHAPTERS)?;
        ebml::write_vint(&mut self.writer, chapters_content.len() as u64)?;
        self.writer.write_all(&chapters_content)?;

        Ok(())
    }

    /// Write SeekHead element.
    fn write_seek_head(&mut self) -> Result<()> {
        // Build SeekHead content
        let mut content = Vec::new();

        // Add Seek entries for known positions
        // Info position (always after SeekHead)
        self.write_seek_entry(&mut content, INFO, self.seek_head_size)?;

        // Tracks position
        self.write_seek_entry(&mut content, TRACKS, self.seek_head_size + 50)?; // Approximate

        // Cues position (if available)
        if let Some(cues_pos) = self.cues_position {
            self.write_seek_entry(&mut content, CUES, cues_pos)?;
        }

        // Check if content fits in reserved space
        let total_size = 4 + ebml::vint_length(content.len() as u64) + content.len();
        if total_size > self.seek_head_size as usize {
            // SeekHead doesn't fit, skip it
            return Ok(());
        }

        // Seek to SeekHead position
        self.writer
            .seek(SeekFrom::Start(self.segment_start + self.seek_head_pos))?;

        // Write SeekHead
        self.write_element_id(SEEK_HEAD)?;
        ebml::write_vint(&mut self.writer, content.len() as u64)?;
        self.writer.write_all(&content)?;

        // Fill remaining space with Void
        let written = 4 + ebml::vint_length(content.len() as u64) + content.len();
        let remaining = self.seek_head_size as usize - written;
        if remaining > 2 {
            self.write_void(remaining)?;
        }

        Ok(())
    }

    /// Write a Seek entry.
    fn write_seek_entry(&self, output: &mut Vec<u8>, element_id: u32, position: u64) -> Result<()> {
        let mut seek_content = Vec::new();

        // SeekID
        let id_bytes = element_id.to_be_bytes();
        let start = id_bytes.iter().position(|&b| b != 0).unwrap_or(3);
        self.write_element_to_vec(&mut seek_content, SEEK_ID, &id_bytes[start..])?;

        // SeekPosition
        self.write_uint_element_to_vec(&mut seek_content, SEEK_POSITION, position)?;

        self.write_element_to_vec(output, SEEK, &seek_content)?;

        Ok(())
    }

    /// Write a Void element for padding.
    fn write_void(&mut self, size: usize) -> Result<()> {
        if size < 2 {
            return Ok(());
        }

        self.write_element_id(VOID)?;
        let content_size = size - 1 - ebml::vint_length((size - 2) as u64);
        ebml::write_vint(&mut self.writer, content_size as u64)?;

        let zeros = vec![0u8; content_size];
        self.writer.write_all(&zeros)?;

        Ok(())
    }

    /// Get the current position.
    fn position(&mut self) -> Result<u64> {
        Ok(self.writer.stream_position()?)
    }

    /// Write an element ID.
    fn write_element_id(&mut self, id: u32) -> Result<()> {
        ebml::write_element_id(&mut self.writer, id)?;
        Ok(())
    }

    /// Write unknown size marker.
    fn write_unknown_size(&mut self, length: usize) -> Result<()> {
        ebml::write_unknown_size(&mut self.writer, length)?;
        Ok(())
    }

    /// Write an unsigned integer element.
    fn write_uint_element(&mut self, id: u32, value: u64) -> Result<()> {
        self.write_element_id(id)?;
        ebml::write_unsigned_int(&mut self.writer, value)?;
        Ok(())
    }

    /// Write an element to a vector.
    fn write_element_to_vec(&self, output: &mut Vec<u8>, id: u32, data: &[u8]) -> Result<()> {
        ebml::write_element_id(output, id)?;
        ebml::write_vint(output, data.len() as u64)?;
        output.extend_from_slice(data);
        Ok(())
    }

    /// Write an unsigned integer element to a vector.
    fn write_uint_element_to_vec(&self, output: &mut Vec<u8>, id: u32, value: u64) -> Result<()> {
        ebml::write_element_id(output, id)?;
        ebml::write_unsigned_int(output, value)?;
        Ok(())
    }

    /// Write a float element to a vector.
    fn write_float_element_to_vec(&self, output: &mut Vec<u8>, id: u32, value: f64) -> Result<()> {
        ebml::write_element_id(output, id)?;
        ebml::write_float(output, value)?;
        Ok(())
    }

    /// Get the underlying writer.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_muxer_creation() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let muxer = MkvMuxer::new(buffer, config);
        assert_eq!(muxer.state, MuxerState::Initial);
    }

    #[test]
    fn test_webm_config() {
        let config = MuxerConfig::webm();
        assert_eq!(config.doc_type, "webm");
    }

    #[test]
    fn test_video_track_config() {
        let config = VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080);
        assert_eq!(config.track_number, 1);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert!(config.is_default);
    }

    #[test]
    fn test_audio_track_config() {
        let config = AudioTrackConfig::new(2, AudioCodec::Opus, 48000.0, 2);
        assert_eq!(config.track_number, 2);
        assert_eq!(config.sample_rate, 48000.0);
        assert_eq!(config.channels, 2);
    }

    #[test]
    fn test_add_tracks() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = MkvMuxer::new(buffer, config);

        let video = VideoTrackConfig::new(1, VideoCodec::H264, 1920, 1080);
        muxer.add_video_track(video).unwrap();

        let audio = AudioTrackConfig::new(2, AudioCodec::Aac, 48000.0, 2);
        muxer.add_audio_track(audio).unwrap();

        assert_eq!(muxer.tracks.len(), 2);
    }

    #[test]
    fn test_webm_codec_validation() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::webm();
        let mut muxer = MkvMuxer::new(buffer, config);

        // VP9 is WebM-compatible
        let vp9 = VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080);
        assert!(muxer.add_video_track(vp9).is_ok());

        // H.264 is not WebM-compatible
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::webm();
        let mut muxer = MkvMuxer::new(buffer, config);
        let h264 = VideoTrackConfig::new(1, VideoCodec::H264, 1920, 1080);
        assert!(muxer.add_video_track(h264).is_err());
    }

    #[test]
    fn test_write_header() {
        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let mut muxer = MkvMuxer::new(buffer, config);

        let video = VideoTrackConfig::new(1, VideoCodec::H264, 1920, 1080);
        muxer.add_video_track(video).unwrap();

        muxer.write_header().unwrap();

        let data = muxer.into_inner().into_inner();
        assert!(!data.is_empty());

        // Check EBML header signature
        assert_eq!(&data[0..4], &[0x1A, 0x45, 0xDF, 0xA3]);
    }

    #[test]
    fn test_muxer_chapter() {
        let chapter = MuxerChapter {
            uid: 1,
            start_time_ns: 0,
            end_time_ns: Some(60_000_000_000),
            title: "Chapter 1".to_string(),
            language: Some("eng".to_string()),
        };

        assert_eq!(chapter.title, "Chapter 1");
        assert_eq!(chapter.start_time_ns, 0);
    }
}
