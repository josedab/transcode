//! WebM muxer implementation.
//!
//! This module provides muxing capabilities for WebM containers.
//! WebM supports VP8, VP9, and AV1 video codecs, and Vorbis and Opus audio codecs.

use crate::ebml::{self, EbmlHeader, ElementHeader};
use crate::elements::{self, *};
use crate::error::{WebmError, Result};

use std::collections::HashMap;
use std::io::{Seek, SeekFrom, Write};

use transcode_core::timestamp::TimeBase;
use transcode_core::{AudioCodec, Packet, PacketFlags, VideoCodec};

/// Default timecode scale (1 millisecond = 1,000,000 nanoseconds).
pub const DEFAULT_TIMECODE_SCALE: u64 = 1_000_000;

/// Maximum cluster duration in milliseconds (5 seconds).
pub const MAX_CLUSTER_DURATION_MS: u64 = 5000;

/// WebM time base (nanoseconds).
pub const WEBM_TIME_BASE: TimeBase = TimeBase::NANOSECONDS;

/// Video track configuration.
#[derive(Debug, Clone)]
pub struct VideoTrackConfig {
    /// Track number (1-based).
    pub track_number: u64,
    /// Video codec (must be VP8, VP9, or AV1).
    pub codec: VideoCodec,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Display width (for aspect ratio).
    pub display_width: Option<u32>,
    /// Display height (for aspect ratio).
    pub display_height: Option<u32>,
    /// Codec private data.
    pub codec_private: Option<Vec<u8>>,
    /// Default frame duration in nanoseconds.
    pub default_duration: Option<u64>,
    /// Track name.
    pub name: Option<String>,
}

impl VideoTrackConfig {
    /// Create a new video track configuration.
    pub fn new(track_number: u64, codec: VideoCodec, width: u32, height: u32) -> Self {
        Self {
            track_number,
            codec,
            width,
            height,
            display_width: None,
            display_height: None,
            codec_private: None,
            default_duration: None,
            name: None,
        }
    }

    /// Set display dimensions for aspect ratio.
    pub fn with_display_size(mut self, width: u32, height: u32) -> Self {
        self.display_width = Some(width);
        self.display_height = Some(height);
        self
    }

    /// Set codec private data.
    pub fn with_codec_private(mut self, data: Vec<u8>) -> Self {
        self.codec_private = Some(data);
        self
    }

    /// Set default frame duration (in nanoseconds).
    pub fn with_default_duration(mut self, duration_ns: u64) -> Self {
        self.default_duration = Some(duration_ns);
        self
    }

    /// Set frame rate (converted to default duration).
    pub fn with_frame_rate(mut self, fps: f64) -> Self {
        self.default_duration = Some((1_000_000_000.0 / fps) as u64);
        self
    }

    /// Set track name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Audio track configuration.
#[derive(Debug, Clone)]
pub struct AudioTrackConfig {
    /// Track number (1-based).
    pub track_number: u64,
    /// Audio codec (must be Vorbis or Opus).
    pub codec: AudioCodec,
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of channels.
    pub channels: u32,
    /// Bits per sample.
    pub bit_depth: Option<u32>,
    /// Codec private data (required for Vorbis, optional for Opus).
    pub codec_private: Option<Vec<u8>>,
    /// Codec delay in nanoseconds (used for Opus).
    pub codec_delay: Option<u64>,
    /// Seek pre-roll in nanoseconds (used for Opus).
    pub seek_pre_roll: Option<u64>,
    /// Track name.
    pub name: Option<String>,
}

impl AudioTrackConfig {
    /// Create a new audio track configuration.
    pub fn new(track_number: u64, codec: AudioCodec, sample_rate: f64, channels: u32) -> Self {
        Self {
            track_number,
            codec,
            sample_rate,
            channels,
            bit_depth: None,
            codec_private: None,
            codec_delay: None,
            seek_pre_roll: None,
            name: None,
        }
    }

    /// Set bit depth.
    pub fn with_bit_depth(mut self, bit_depth: u32) -> Self {
        self.bit_depth = Some(bit_depth);
        self
    }

    /// Set codec private data.
    pub fn with_codec_private(mut self, data: Vec<u8>) -> Self {
        self.codec_private = Some(data);
        self
    }

    /// Set codec delay (for Opus).
    pub fn with_codec_delay(mut self, delay_ns: u64) -> Self {
        self.codec_delay = Some(delay_ns);
        self
    }

    /// Set seek pre-roll (for Opus).
    pub fn with_seek_pre_roll(mut self, pre_roll_ns: u64) -> Self {
        self.seek_pre_roll = Some(pre_roll_ns);
        self
    }

    /// Set track name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Muxer configuration.
#[derive(Debug, Clone)]
pub struct MuxerConfig {
    /// Timecode scale (nanoseconds per timecode unit).
    pub timecode_scale: u64,
    /// Maximum cluster duration in timecode units.
    pub max_cluster_duration: u64,
    /// Writing application name.
    pub writing_app: String,
    /// Title (optional).
    pub title: Option<String>,
    /// Generate cues (seeking index).
    pub generate_cues: bool,
}

impl Default for MuxerConfig {
    fn default() -> Self {
        Self {
            timecode_scale: DEFAULT_TIMECODE_SCALE,
            max_cluster_duration: MAX_CLUSTER_DURATION_MS,
            writing_app: "transcode-webm".to_string(),
            title: None,
            generate_cues: true,
        }
    }
}

impl MuxerConfig {
    /// Create default WebM configuration.
    pub fn webm() -> Self {
        Self::default()
    }

    /// Set title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set writing application name.
    pub fn with_writing_app(mut self, app: impl Into<String>) -> Self {
        self.writing_app = app.into();
        self
    }

    /// Enable or disable cue generation.
    pub fn with_cues(mut self, enabled: bool) -> Self {
        self.generate_cues = enabled;
        self
    }
}

/// A cue point entry for seeking.
#[derive(Debug, Clone)]
struct CueEntry {
    /// Timestamp in timecode units.
    time: u64,
    /// Track number.
    track: u64,
    /// Cluster position (byte offset from segment start).
    cluster_position: u64,
}

/// Track state during muxing.
#[derive(Debug, Clone)]
struct TrackState {
    /// Track type (video or audio).
    is_video: bool,
    /// Maximum timestamp seen.
    max_timestamp: u64,
}

/// Muxer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuxerState {
    /// Initial state.
    Initial,
    /// Header written.
    HeaderWritten,
    /// Segment opened.
    SegmentOpened,
    /// Writing clusters.
    WritingClusters,
    /// Finalized.
    Finalized,
}

/// WebM muxer.
pub struct WebmMuxer<W: Write + Seek> {
    writer: W,
    config: MuxerConfig,
    state: MuxerState,
    /// Video track configurations.
    video_tracks: Vec<VideoTrackConfig>,
    /// Audio track configurations.
    audio_tracks: Vec<AudioTrackConfig>,
    /// Track states (indexed by track number).
    track_states: HashMap<u64, TrackState>,
    /// Cue entries for seeking index.
    cue_entries: Vec<CueEntry>,
    /// Segment start position.
    segment_start: u64,
    /// Segment size placeholder position.
    segment_size_pos: u64,
    /// Current cluster start position.
    cluster_start: u64,
    /// Current cluster timestamp (in timecode units).
    cluster_timestamp: u64,
    /// Whether we're in a cluster.
    in_cluster: bool,
    /// First timestamp in current cluster.
    cluster_first_timestamp: Option<u64>,
    /// Duration placeholder position.
    duration_pos: Option<u64>,
    /// Total duration (max timestamp across all tracks).
    total_duration: u64,
    /// Cues placeholder position.
    cues_pos: u64,
}

impl<W: Write + Seek> WebmMuxer<W> {
    /// Create a new WebM muxer.
    pub fn new(writer: W) -> Self {
        Self::with_config(writer, MuxerConfig::default())
    }

    /// Create a new WebM muxer with custom configuration.
    pub fn with_config(writer: W, config: MuxerConfig) -> Self {
        Self {
            writer,
            config,
            state: MuxerState::Initial,
            video_tracks: Vec::new(),
            audio_tracks: Vec::new(),
            track_states: HashMap::new(),
            cue_entries: Vec::new(),
            segment_start: 0,
            segment_size_pos: 0,
            cluster_start: 0,
            cluster_timestamp: 0,
            in_cluster: false,
            cluster_first_timestamp: None,
            duration_pos: None,
            total_duration: 0,
            cues_pos: 0,
        }
    }

    /// Get the current position in the stream.
    fn position(&mut self) -> Result<u64> {
        Ok(self.writer.stream_position()?)
    }

    /// Add a video track.
    pub fn add_video_track(&mut self, config: VideoTrackConfig) -> Result<()> {
        if self.state != MuxerState::Initial {
            return Err(WebmError::Other("Cannot add tracks after header is written".to_string()));
        }

        // Validate WebM compatibility
        if !matches!(config.codec, VideoCodec::Vp8 | VideoCodec::Vp9 | VideoCodec::Av1) {
            return Err(WebmError::InvalidWebM(format!(
                "Video codec {:?} is not WebM-compatible. Use VP8, VP9, or AV1.",
                config.codec
            )));
        }

        self.track_states.insert(config.track_number, TrackState {
            is_video: true,
            max_timestamp: 0,
        });

        self.video_tracks.push(config);
        Ok(())
    }

    /// Add an audio track.
    pub fn add_audio_track(&mut self, config: AudioTrackConfig) -> Result<()> {
        if self.state != MuxerState::Initial {
            return Err(WebmError::Other("Cannot add tracks after header is written".to_string()));
        }

        // Validate WebM compatibility
        if !matches!(config.codec, AudioCodec::Vorbis | AudioCodec::Opus) {
            return Err(WebmError::InvalidWebM(format!(
                "Audio codec {:?} is not WebM-compatible. Use Vorbis or Opus.",
                config.codec
            )));
        }

        self.track_states.insert(config.track_number, TrackState {
            is_video: false,
            max_timestamp: 0,
        });

        self.audio_tracks.push(config);
        Ok(())
    }

    /// Write the file header (EBML header and segment start).
    pub fn write_header(&mut self) -> Result<()> {
        if self.state != MuxerState::Initial {
            return Err(WebmError::Other("Header already written".to_string()));
        }

        if self.video_tracks.is_empty() && self.audio_tracks.is_empty() {
            return Err(WebmError::Other("No tracks added".to_string()));
        }

        // Write EBML header
        self.write_ebml_header()?;

        // Write segment with unknown size (we'll fix it at finalization)
        ebml::write_element_id(&mut self.writer, SEGMENT)?;
        self.segment_size_pos = self.position()?;
        ebml::write_unknown_size(&mut self.writer, 8)?; // 8-byte unknown size

        self.segment_start = self.position()?;

        // Write segment info
        self.write_segment_info()?;

        // Write tracks
        self.write_tracks()?;

        self.state = MuxerState::SegmentOpened;
        Ok(())
    }

    /// Write the EBML header.
    fn write_ebml_header(&mut self) -> Result<()> {
        let header = EbmlHeader::webm();

        // Build header content
        let mut content = Vec::new();

        // EBMLVersion
        Self::write_element(&mut content, EBML_VERSION, |w| {
            ebml::write_unsigned_int(w, header.version)
        })?;

        // EBMLReadVersion
        Self::write_element(&mut content, EBML_READ_VERSION, |w| {
            ebml::write_unsigned_int(w, header.read_version)
        })?;

        // EBMLMaxIDLength
        Self::write_element(&mut content, EBML_MAX_ID_LENGTH, |w| {
            ebml::write_unsigned_int(w, header.max_id_length)
        })?;

        // EBMLMaxSizeLength
        Self::write_element(&mut content, EBML_MAX_SIZE_LENGTH, |w| {
            ebml::write_unsigned_int(w, header.max_size_length)
        })?;

        // DocType
        Self::write_element(&mut content, DOC_TYPE, |w| {
            ebml::write_string(w, &header.doc_type)
        })?;

        // DocTypeVersion
        Self::write_element(&mut content, DOC_TYPE_VERSION, |w| {
            ebml::write_unsigned_int(w, header.doc_type_version)
        })?;

        // DocTypeReadVersion
        Self::write_element(&mut content, DOC_TYPE_READ_VERSION, |w| {
            ebml::write_unsigned_int(w, header.doc_type_read_version)
        })?;

        // Write EBML master element
        let header = ElementHeader {
            id: EBML,
            size: Some(content.len() as u64),
            header_size: 0,
        };
        header.write(&mut self.writer)?;
        self.writer.write_all(&content)?;

        Ok(())
    }

    /// Write segment info.
    fn write_segment_info(&mut self) -> Result<()> {
        let mut content = Vec::new();

        // TimecodeScale
        Self::write_element(&mut content, TIMECODE_SCALE, |w| {
            ebml::write_unsigned_int(w, self.config.timecode_scale)
        })?;

        // MuxingApp
        Self::write_element(&mut content, MUXING_APP, |w| {
            ebml::write_string(w, "transcode-webm")
        })?;

        // WritingApp
        Self::write_element(&mut content, WRITING_APP, |w| {
            ebml::write_string(w, &self.config.writing_app)
        })?;

        // Title (optional)
        if let Some(ref title) = self.config.title {
            Self::write_element(&mut content, TITLE, |w| {
                ebml::write_string(w, title)
            })?;
        }

        // Duration placeholder (8-byte float, we'll update it later)
        let info_content_start = content.len();
        Self::write_element(&mut content, DURATION, |w| {
            ebml::write_float(w, 0.0)
        })?;

        // Write Info element
        let header = ElementHeader {
            id: INFO,
            size: Some(content.len() as u64),
            header_size: 0,
        };
        let header_pos = self.position()?;
        let actual_header_size = header.write(&mut self.writer)?;

        // Calculate duration position
        // Duration element is at: header_pos + actual_header_size + info_content_start + 3 (ID + size)
        self.duration_pos = Some(header_pos + actual_header_size as u64 + info_content_start as u64 + 3);

        self.writer.write_all(&content)?;

        Ok(())
    }

    /// Write tracks element.
    fn write_tracks(&mut self) -> Result<()> {
        let mut content = Vec::new();

        // Write video tracks
        for track in &self.video_tracks {
            self.write_video_track(&mut content, track)?;
        }

        // Write audio tracks
        for track in &self.audio_tracks {
            self.write_audio_track(&mut content, track)?;
        }

        // Write Tracks element
        let header = ElementHeader {
            id: TRACKS,
            size: Some(content.len() as u64),
            header_size: 0,
        };
        header.write(&mut self.writer)?;
        self.writer.write_all(&content)?;

        Ok(())
    }

    /// Write a video track entry.
    fn write_video_track(&self, content: &mut Vec<u8>, track: &VideoTrackConfig) -> Result<()> {
        let mut track_content = Vec::new();

        // TrackNumber
        Self::write_element(&mut track_content, TRACK_NUMBER, |w| {
            ebml::write_unsigned_int(w, track.track_number)
        })?;

        // TrackUID (use track number as UID)
        Self::write_element(&mut track_content, TRACK_UID, |w| {
            ebml::write_unsigned_int(w, track.track_number)
        })?;

        // TrackType (video = 1)
        Self::write_element(&mut track_content, TRACK_TYPE, |w| {
            ebml::write_unsigned_int(w, TRACK_TYPE_VIDEO as u64)
        })?;

        // FlagEnabled
        Self::write_element(&mut track_content, FLAG_ENABLED, |w| {
            ebml::write_unsigned_int(w, 1)
        })?;

        // FlagDefault
        Self::write_element(&mut track_content, FLAG_DEFAULT, |w| {
            ebml::write_unsigned_int(w, 1)
        })?;

        // FlagLacing
        Self::write_element(&mut track_content, FLAG_LACING, |w| {
            ebml::write_unsigned_int(w, 0)
        })?;

        // CodecID
        let codec_id = elements::video_codec_to_webm_id(track.codec)
            .ok_or_else(|| WebmError::InvalidCodecId(format!("{:?}", track.codec)))?;
        Self::write_element(&mut track_content, CODEC_ID, |w| {
            ebml::write_string(w, codec_id)
        })?;

        // CodecPrivate (if present)
        if let Some(ref data) = track.codec_private {
            Self::write_element(&mut track_content, CODEC_PRIVATE, |w| {
                w.write_all(data)?;
                Ok(data.len())
            })?;
        }

        // DefaultDuration (if present)
        if let Some(duration) = track.default_duration {
            Self::write_element(&mut track_content, DEFAULT_DURATION, |w| {
                ebml::write_unsigned_int(w, duration)
            })?;
        }

        // Name (if present)
        if let Some(ref name) = track.name {
            Self::write_element(&mut track_content, NAME, |w| {
                ebml::write_string(w, name)
            })?;
        }

        // Video settings
        let mut video_content = Vec::new();

        // PixelWidth
        Self::write_element(&mut video_content, PIXEL_WIDTH, |w| {
            ebml::write_unsigned_int(w, track.width as u64)
        })?;

        // PixelHeight
        Self::write_element(&mut video_content, PIXEL_HEIGHT, |w| {
            ebml::write_unsigned_int(w, track.height as u64)
        })?;

        // DisplayWidth (if different from pixel width)
        if let Some(dw) = track.display_width {
            Self::write_element(&mut video_content, DISPLAY_WIDTH, |w| {
                ebml::write_unsigned_int(w, dw as u64)
            })?;
        }

        // DisplayHeight (if different from pixel height)
        if let Some(dh) = track.display_height {
            Self::write_element(&mut video_content, DISPLAY_HEIGHT, |w| {
                ebml::write_unsigned_int(w, dh as u64)
            })?;
        }

        // Write Video element
        Self::write_master_element(&mut track_content, VIDEO, &video_content)?;

        // Write TrackEntry element
        Self::write_master_element(content, TRACK_ENTRY, &track_content)?;

        Ok(())
    }

    /// Write an audio track entry.
    fn write_audio_track(&self, content: &mut Vec<u8>, track: &AudioTrackConfig) -> Result<()> {
        let mut track_content = Vec::new();

        // TrackNumber
        Self::write_element(&mut track_content, TRACK_NUMBER, |w| {
            ebml::write_unsigned_int(w, track.track_number)
        })?;

        // TrackUID
        Self::write_element(&mut track_content, TRACK_UID, |w| {
            ebml::write_unsigned_int(w, track.track_number)
        })?;

        // TrackType (audio = 2)
        Self::write_element(&mut track_content, TRACK_TYPE, |w| {
            ebml::write_unsigned_int(w, TRACK_TYPE_AUDIO as u64)
        })?;

        // FlagEnabled
        Self::write_element(&mut track_content, FLAG_ENABLED, |w| {
            ebml::write_unsigned_int(w, 1)
        })?;

        // FlagDefault
        Self::write_element(&mut track_content, FLAG_DEFAULT, |w| {
            ebml::write_unsigned_int(w, 1)
        })?;

        // FlagLacing
        Self::write_element(&mut track_content, FLAG_LACING, |w| {
            ebml::write_unsigned_int(w, 0)
        })?;

        // CodecID
        let codec_id = elements::audio_codec_to_webm_id(track.codec)
            .ok_or_else(|| WebmError::InvalidCodecId(format!("{:?}", track.codec)))?;
        Self::write_element(&mut track_content, CODEC_ID, |w| {
            ebml::write_string(w, codec_id)
        })?;

        // CodecPrivate (if present)
        if let Some(ref data) = track.codec_private {
            Self::write_element(&mut track_content, CODEC_PRIVATE, |w| {
                w.write_all(data)?;
                Ok(data.len())
            })?;
        }

        // CodecDelay (for Opus)
        if let Some(delay) = track.codec_delay {
            Self::write_element(&mut track_content, CODEC_DELAY, |w| {
                ebml::write_unsigned_int(w, delay)
            })?;
        }

        // SeekPreRoll (for Opus)
        if let Some(pre_roll) = track.seek_pre_roll {
            Self::write_element(&mut track_content, SEEK_PRE_ROLL, |w| {
                ebml::write_unsigned_int(w, pre_roll)
            })?;
        }

        // Name (if present)
        if let Some(ref name) = track.name {
            Self::write_element(&mut track_content, NAME, |w| {
                ebml::write_string(w, name)
            })?;
        }

        // Audio settings
        let mut audio_content = Vec::new();

        // SamplingFrequency
        Self::write_element(&mut audio_content, SAMPLING_FREQUENCY, |w| {
            ebml::write_float(w, track.sample_rate)
        })?;

        // Channels
        Self::write_element(&mut audio_content, CHANNELS, |w| {
            ebml::write_unsigned_int(w, track.channels as u64)
        })?;

        // BitDepth (if present)
        if let Some(depth) = track.bit_depth {
            Self::write_element(&mut audio_content, BIT_DEPTH, |w| {
                ebml::write_unsigned_int(w, depth as u64)
            })?;
        }

        // Write Audio element
        Self::write_master_element(&mut track_content, AUDIO, &audio_content)?;

        // Write TrackEntry element
        Self::write_master_element(content, TRACK_ENTRY, &track_content)?;

        Ok(())
    }

    /// Write a packet to the file.
    pub fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.state == MuxerState::Initial {
            self.write_header()?;
        }

        if self.state == MuxerState::Finalized {
            return Err(WebmError::Other("Cannot write to finalized file".to_string()));
        }

        self.state = MuxerState::WritingClusters;

        // Get track number
        let track_number = packet.stream_index as u64;

        // Convert timestamp to timecode units
        let timestamp_ns = packet.pts.rescale(WEBM_TIME_BASE).value;
        let timestamp = (timestamp_ns as u64) / self.config.timecode_scale;

        // Update track state
        if let Some(state) = self.track_states.get_mut(&track_number) {
            state.max_timestamp = state.max_timestamp.max(timestamp);
        }
        self.total_duration = self.total_duration.max(timestamp);

        // Check if we need to start a new cluster
        let is_keyframe = packet.flags.contains(PacketFlags::KEYFRAME);
        let is_video = self.track_states.get(&track_number).is_some_and(|s| s.is_video);

        let start_new_cluster = if !self.in_cluster {
            true
        } else if is_video && is_keyframe {
            // Start new cluster on video keyframes
            true
        } else if let Some(first_ts) = self.cluster_first_timestamp {
            // Start new cluster if duration exceeded
            timestamp.saturating_sub(first_ts) >= self.config.max_cluster_duration
        } else {
            false
        };

        if start_new_cluster {
            // Close current cluster if open
            if self.in_cluster {
                self.close_cluster()?;
            }

            // Start new cluster
            self.start_cluster(timestamp)?;

            // Add cue entry for video keyframes
            if is_video && is_keyframe && self.config.generate_cues {
                self.cue_entries.push(CueEntry {
                    time: timestamp,
                    track: track_number,
                    cluster_position: self.cluster_start - self.segment_start,
                });
            }
        }

        // Write SimpleBlock
        self.write_simple_block(packet, track_number, timestamp)?;

        Ok(())
    }

    /// Start a new cluster.
    fn start_cluster(&mut self, timestamp: u64) -> Result<()> {
        self.cluster_start = self.position()?;
        self.cluster_timestamp = timestamp;
        self.cluster_first_timestamp = Some(timestamp);

        // Write cluster header with unknown size (we'll fix it later)
        ebml::write_element_id(&mut self.writer, CLUSTER)?;
        ebml::write_unknown_size(&mut self.writer, 4)?; // 4-byte unknown size

        // Write cluster timestamp
        let mut ts_content = Vec::new();
        ebml::write_unsigned_int(&mut ts_content, timestamp)?;
        Self::write_master_element(&mut self.writer, TIMESTAMP, &ts_content)?;

        self.in_cluster = true;
        Ok(())
    }

    /// Close the current cluster.
    fn close_cluster(&mut self) -> Result<()> {
        if !self.in_cluster {
            return Ok(());
        }

        // For WebM, we leave clusters with unknown size (which is valid)
        // Alternatively, we could go back and patch the size, but unknown size is simpler
        // and more compatible with streaming scenarios

        self.in_cluster = false;
        self.cluster_first_timestamp = None;
        Ok(())
    }

    /// Write a SimpleBlock.
    fn write_simple_block(&mut self, packet: &Packet, track_number: u64, timestamp: u64) -> Result<()> {
        // Calculate relative timestamp (signed 16-bit)
        let relative_ts = (timestamp as i64 - self.cluster_timestamp as i64)
            .clamp(i16::MIN as i64, i16::MAX as i64) as i16;

        // Build block data
        let mut block_data = Vec::new();

        // Track number (VINT)
        ebml::write_vint(&mut block_data, track_number)?;

        // Relative timestamp (big-endian signed 16-bit)
        block_data.extend_from_slice(&relative_ts.to_be_bytes());

        // Flags
        let mut flags = 0u8;
        if packet.flags.contains(PacketFlags::KEYFRAME) {
            flags |= 0x80; // Keyframe
        }
        if packet.flags.contains(PacketFlags::DISPOSABLE) {
            flags |= 0x01; // Discardable
        }
        // No lacing (bits 1-2 = 00)
        block_data.push(flags);

        // Frame data
        block_data.extend_from_slice(packet.data());

        // Write SimpleBlock element
        let header = ElementHeader {
            id: SIMPLE_BLOCK,
            size: Some(block_data.len() as u64),
            header_size: 0,
        };
        header.write(&mut self.writer)?;
        self.writer.write_all(&block_data)?;

        Ok(())
    }

    /// Finalize the file.
    pub fn finalize(&mut self) -> Result<()> {
        if self.state == MuxerState::Finalized {
            return Ok(());
        }

        // Close current cluster
        if self.in_cluster {
            self.close_cluster()?;
        }

        // Write cues
        if self.config.generate_cues && !self.cue_entries.is_empty() {
            self.write_cues()?;
        }

        // Update duration
        if let Some(duration_pos) = self.duration_pos {
            let duration_float = self.total_duration as f64;
            let current_pos = self.position()?;
            self.writer.seek(SeekFrom::Start(duration_pos))?;
            ebml::write_float(&mut self.writer, duration_float)?;
            self.writer.seek(SeekFrom::Start(current_pos))?;
        }

        self.state = MuxerState::Finalized;
        self.writer.flush()?;
        Ok(())
    }

    /// Write cues (seeking index).
    fn write_cues(&mut self) -> Result<()> {
        let mut content = Vec::new();

        for entry in &self.cue_entries {
            let mut point_content = Vec::new();

            // CueTime
            Self::write_element(&mut point_content, CUE_TIME, |w| {
                ebml::write_unsigned_int(w, entry.time)
            })?;

            // CueTrackPositions
            let mut positions_content = Vec::new();

            // CueTrack
            Self::write_element(&mut positions_content, CUE_TRACK, |w| {
                ebml::write_unsigned_int(w, entry.track)
            })?;

            // CueClusterPosition
            Self::write_element(&mut positions_content, CUE_CLUSTER_POSITION, |w| {
                ebml::write_unsigned_int(w, entry.cluster_position)
            })?;

            Self::write_master_element(&mut point_content, CUE_TRACK_POSITIONS, &positions_content)?;

            Self::write_master_element(&mut content, CUE_POINT, &point_content)?;
        }

        // Write Cues element
        let header = ElementHeader {
            id: CUES,
            size: Some(content.len() as u64),
            header_size: 0,
        };
        self.cues_pos = self.position()?;
        header.write(&mut self.writer)?;
        self.writer.write_all(&content)?;

        Ok(())
    }

    /// Get the underlying writer, consuming the muxer.
    /// Note: You should call `finalize()` before this if you want proper file structure.
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// Get a reference to the underlying writer.
    pub fn writer(&self) -> &W {
        &self.writer
    }

    /// Get the current state.
    pub fn state(&self) -> MuxerState {
        self.state
    }

    // Helper functions for writing EBML elements

    /// Write a simple element with a content writer function.
    fn write_element<W2: Write, F>(writer: &mut W2, id: u32, content_fn: F) -> Result<()>
    where
        F: FnOnce(&mut Vec<u8>) -> Result<usize>,
    {
        let mut content = Vec::new();
        content_fn(&mut content)?;

        let header = ElementHeader {
            id,
            size: Some(content.len() as u64),
            header_size: 0,
        };
        header.write(writer)?;
        writer.write_all(&content)?;

        Ok(())
    }

    /// Write a master element containing pre-built content.
    fn write_master_element<W2: Write>(writer: &mut W2, id: u32, content: &[u8]) -> Result<()> {
        let header = ElementHeader {
            id,
            size: Some(content.len() as u64),
            header_size: 0,
        };
        header.write(writer)?;
        writer.write_all(content)?;
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_muxer_config_default() {
        let config = MuxerConfig::default();
        assert_eq!(config.timecode_scale, DEFAULT_TIMECODE_SCALE);
        assert!(config.generate_cues);
    }

    #[test]
    fn test_video_track_config() {
        let config = VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080)
            .with_frame_rate(30.0)
            .with_name("Video");

        assert_eq!(config.track_number, 1);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert!(config.default_duration.is_some());
        assert_eq!(config.name, Some("Video".to_string()));
    }

    #[test]
    fn test_audio_track_config() {
        let config = AudioTrackConfig::new(2, AudioCodec::Opus, 48000.0, 2)
            .with_codec_delay(6500000)
            .with_name("Audio");

        assert_eq!(config.track_number, 2);
        assert_eq!(config.sample_rate, 48000.0);
        assert_eq!(config.channels, 2);
        assert_eq!(config.codec_delay, Some(6500000));
    }

    #[test]
    fn test_muxer_creation() {
        let buffer = Cursor::new(Vec::new());
        let muxer = WebmMuxer::new(buffer);
        assert_eq!(muxer.state(), MuxerState::Initial);
    }

    #[test]
    fn test_add_video_track() {
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        let config = VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080);
        assert!(muxer.add_video_track(config).is_ok());
    }

    #[test]
    fn test_add_audio_track() {
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        let config = AudioTrackConfig::new(1, AudioCodec::Opus, 48000.0, 2);
        assert!(muxer.add_audio_track(config).is_ok());
    }

    #[test]
    fn test_invalid_video_codec() {
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        let config = VideoTrackConfig::new(1, VideoCodec::H264, 1920, 1080);
        assert!(muxer.add_video_track(config).is_err());
    }

    #[test]
    fn test_invalid_audio_codec() {
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        let config = AudioTrackConfig::new(1, AudioCodec::Aac, 48000.0, 2);
        assert!(muxer.add_audio_track(config).is_err());
    }

    #[test]
    fn test_write_header() {
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        muxer.add_video_track(VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080)).unwrap();

        assert!(muxer.write_header().is_ok());
        assert_eq!(muxer.state(), MuxerState::SegmentOpened);
    }

    #[test]
    fn test_write_header_no_tracks() {
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        assert!(muxer.write_header().is_err());
    }

    #[test]
    fn test_mux_packets() {
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        muxer.add_video_track(VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080)).unwrap();
        muxer.write_header().unwrap();

        // Create a keyframe packet
        let mut packet = Packet::new(vec![0u8; 1000]);
        packet.stream_index = 1;
        packet.pts = transcode_core::Timestamp::new(0, WEBM_TIME_BASE);
        packet.flags.insert(PacketFlags::KEYFRAME);

        assert!(muxer.write_packet(&packet).is_ok());

        // Create a non-keyframe
        let mut packet2 = Packet::new(vec![0u8; 500]);
        packet2.stream_index = 1;
        packet2.pts = transcode_core::Timestamp::new(33_333_333, WEBM_TIME_BASE);

        assert!(muxer.write_packet(&packet2).is_ok());
        assert!(muxer.finalize().is_ok());
    }

    #[test]
    fn test_finalize() {
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        muxer.add_video_track(VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080)).unwrap();
        muxer.write_header().unwrap();

        assert!(muxer.finalize().is_ok());
        assert_eq!(muxer.state(), MuxerState::Finalized);

        // Second finalize should be no-op
        assert!(muxer.finalize().is_ok());
    }

    #[test]
    fn test_output_starts_with_ebml_header() {
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        muxer.add_video_track(VideoTrackConfig::new(1, VideoCodec::Vp9, 640, 480)).unwrap();
        muxer.write_header().unwrap();
        muxer.finalize().unwrap();

        let output = muxer.into_inner().into_inner();

        // EBML header ID
        assert_eq!(&output[0..4], &[0x1A, 0x45, 0xDF, 0xA3]);
    }
}
