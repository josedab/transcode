//! WebM demuxer implementation.
//!
//! This module provides demuxing capabilities for WebM containers.

use crate::ebml::{self, EbmlHeader, ElementHeader};
use crate::elements::{self, *};
use crate::error::{WebmError, Result};

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use transcode_core::format::StreamType;
use transcode_core::timestamp::{Duration, TimeBase, Timestamp};
use transcode_core::{AudioCodec, Packet, PacketFlags, VideoCodec};

/// Default timecode scale (1 millisecond in nanoseconds).
const DEFAULT_TIMECODE_SCALE: u64 = 1_000_000;

/// Maximum element size to prevent OOM from malformed files (256 MB).
const MAX_ELEMENT_SIZE: u64 = 256 * 1024 * 1024;

/// WebM time base (nanoseconds).
pub const WEBM_TIME_BASE: TimeBase = TimeBase::NANOSECONDS;

/// Lacing types in WebM.
mod lacing {
    /// No lacing - single frame per block.
    pub const NO_LACING: u8 = 0;
    /// Xiph-style lacing with variable-sized frames.
    pub const XIPH: u8 = 1;
    /// Fixed-size lacing - all frames have equal size.
    pub const FIXED: u8 = 2;
    /// EBML-style lacing with signed size deltas.
    pub const EBML: u8 = 3;
}

/// A buffered frame from a laced block.
#[derive(Debug, Clone)]
struct BufferedFrame {
    /// Frame data.
    data: Vec<u8>,
    /// Track number.
    track_number: u64,
    /// Presentation timestamp in nanoseconds.
    pts_ns: u64,
    /// Whether this is a keyframe.
    keyframe: bool,
    /// Whether this frame is discardable.
    discardable: bool,
    /// Duration in nanoseconds (if known).
    duration_ns: Option<u64>,
}

/// Video track information.
#[derive(Debug, Clone)]
pub struct VideoTrackInfo {
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Display width (for aspect ratio).
    pub display_width: Option<u32>,
    /// Display height (for aspect ratio).
    pub display_height: Option<u32>,
    /// Interlaced flag.
    pub interlaced: bool,
    /// Video codec.
    pub codec: Option<VideoCodec>,
}

/// Audio track information.
#[derive(Debug, Clone)]
pub struct AudioTrackInfo {
    /// Sampling frequency in Hz.
    pub sample_rate: f64,
    /// Output sampling frequency (for SBR).
    pub output_sample_rate: Option<f64>,
    /// Number of channels.
    pub channels: u32,
    /// Bits per sample.
    pub bit_depth: Option<u32>,
    /// Audio codec.
    pub codec: Option<AudioCodec>,
}

/// Track information.
#[derive(Debug, Clone)]
pub struct TrackInfo {
    /// Track number (1-based).
    pub number: u64,
    /// Track UID.
    pub uid: u64,
    /// Track type.
    pub track_type: StreamType,
    /// Codec ID string.
    pub codec_id: String,
    /// Codec private data.
    pub codec_private: Option<Vec<u8>>,
    /// Track name.
    pub name: Option<String>,
    /// Language (ISO 639-2).
    pub language: Option<String>,
    /// Default duration in nanoseconds.
    pub default_duration: Option<u64>,
    /// Codec delay in nanoseconds.
    pub codec_delay: u64,
    /// Seek pre-roll in nanoseconds.
    pub seek_pre_roll: u64,
    /// Is default track.
    pub is_default: bool,
    /// Is forced track.
    pub is_forced: bool,
    /// Is enabled track.
    pub is_enabled: bool,
    /// Video-specific info.
    pub video: Option<VideoTrackInfo>,
    /// Audio-specific info.
    pub audio: Option<AudioTrackInfo>,
}

impl Default for TrackInfo {
    fn default() -> Self {
        Self {
            number: 0,
            uid: 0,
            track_type: StreamType::Unknown,
            codec_id: String::new(),
            codec_private: None,
            name: None,
            language: None,
            default_duration: None,
            codec_delay: 0,
            seek_pre_roll: 0,
            is_default: true,
            is_forced: false,
            is_enabled: true,
            video: None,
            audio: None,
        }
    }
}

/// A cue point for seeking.
#[derive(Debug, Clone)]
pub struct CuePoint {
    /// Timestamp in timecode units.
    pub time: u64,
    /// Track positions.
    pub positions: Vec<CueTrackPosition>,
}

/// Track position within a cue point.
#[derive(Debug, Clone)]
pub struct CueTrackPosition {
    /// Track number.
    pub track: u64,
    /// Cluster position (byte offset from segment start).
    pub cluster_position: u64,
    /// Relative position within cluster.
    pub relative_position: Option<u64>,
    /// Block number within cluster.
    pub block_number: Option<u64>,
}

/// Segment information.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment UID.
    pub uid: Option<[u8; 16]>,
    /// Segment filename.
    pub filename: Option<String>,
    /// Title.
    pub title: Option<String>,
    /// Muxing application.
    pub muxing_app: Option<String>,
    /// Writing application.
    pub writing_app: Option<String>,
    /// Duration in nanoseconds.
    pub duration_ns: Option<u64>,
    /// Timecode scale (nanoseconds per timecode unit).
    pub timecode_scale: u64,
    /// Date in nanoseconds since 2001-01-01.
    pub date: Option<i64>,
}

impl Default for SegmentInfo {
    fn default() -> Self {
        Self {
            uid: None,
            filename: None,
            title: None,
            muxing_app: None,
            writing_app: None,
            duration_ns: None,
            timecode_scale: DEFAULT_TIMECODE_SCALE,
            date: None,
        }
    }
}

/// WebM demuxer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemuxerState {
    /// Initial state, reading EBML header.
    Header,
    /// Reading segment info.
    SegmentInfo,
    /// Reading tracks.
    Tracks,
    /// Reading clusters (media data).
    Clusters,
    /// End of file reached.
    Eof,
}

/// WebM demuxer.
pub struct WebmDemuxer<R: Read + Seek> {
    reader: R,
    state: DemuxerState,
    /// EBML header.
    pub ebml_header: Option<EbmlHeader>,
    /// Segment info.
    pub segment_info: SegmentInfo,
    /// Tracks indexed by track number.
    pub tracks: HashMap<u64, TrackInfo>,
    /// Cue points for seeking.
    pub cues: Vec<CuePoint>,
    /// Segment start position in bytes.
    segment_start: u64,
    /// Segment size (if known).
    segment_size: Option<u64>,
    /// Current cluster timestamp.
    current_cluster_timestamp: u64,
    /// Current position within cluster.
    current_cluster_end: u64,
    /// Whether we're inside a cluster.
    in_cluster: bool,
    /// Seek head positions.
    seek_positions: HashMap<u32, u64>,
    /// Buffered frames from laced blocks.
    frame_buffer: Vec<BufferedFrame>,
}

impl<R: Read + Seek> WebmDemuxer<R> {
    /// Create a new WebM demuxer.
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            state: DemuxerState::Header,
            ebml_header: None,
            segment_info: SegmentInfo::default(),
            tracks: HashMap::new(),
            cues: Vec::new(),
            segment_start: 0,
            segment_size: None,
            current_cluster_timestamp: 0,
            current_cluster_end: 0,
            in_cluster: false,
            seek_positions: HashMap::new(),
            frame_buffer: Vec::new(),
        }
    }

    /// Get the underlying reader.
    pub fn into_inner(self) -> R {
        self.reader
    }

    /// Get the current position in the stream.
    fn position(&mut self) -> Result<u64> {
        Ok(self.reader.stream_position()?)
    }

    /// Read the EBML header.
    pub fn read_header(&mut self) -> Result<()> {
        let header = ElementHeader::read(&mut self.reader)?;

        if header.id != EBML {
            return Err(WebmError::InvalidEbmlHeader(
                "Missing EBML header element".to_string(),
            ));
        }

        let mut ebml_header = EbmlHeader::default();
        let end_pos = self.position()? + header.size.unwrap_or(0);

        while self.position()? < end_pos {
            let child = ElementHeader::read(&mut self.reader)?;
            let size = child.size.unwrap_or(0);

            match child.id {
                EBML_VERSION => {
                    ebml_header.version = self.read_uint(size)?;
                }
                EBML_READ_VERSION => {
                    ebml_header.read_version = self.read_uint(size)?;
                }
                EBML_MAX_ID_LENGTH => {
                    ebml_header.max_id_length = self.read_uint(size)?;
                }
                EBML_MAX_SIZE_LENGTH => {
                    ebml_header.max_size_length = self.read_uint(size)?;
                }
                DOC_TYPE => {
                    ebml_header.doc_type = self.read_string(size)?;
                }
                DOC_TYPE_VERSION => {
                    ebml_header.doc_type_version = self.read_uint(size)?;
                }
                DOC_TYPE_READ_VERSION => {
                    ebml_header.doc_type_read_version = self.read_uint(size)?;
                }
                _ => {
                    self.skip(size)?;
                }
            }
        }

        // Validate doc type
        if ebml_header.doc_type != "webm" {
            return Err(WebmError::InvalidWebM(format!(
                "Document type is '{}', expected 'webm'",
                ebml_header.doc_type
            )));
        }

        self.ebml_header = Some(ebml_header);
        self.state = DemuxerState::SegmentInfo;
        Ok(())
    }

    /// Read the segment header and metadata.
    pub fn read_segment_info(&mut self) -> Result<()> {
        if self.state == DemuxerState::Header {
            self.read_header()?;
        }

        // Find segment element
        let header = ElementHeader::read(&mut self.reader)?;
        if header.id != SEGMENT {
            return Err(WebmError::MissingElement("Segment".to_string()));
        }

        self.segment_start = self.position()?;
        self.segment_size = header.size;

        // Parse segment children until we hit clusters
        self.parse_segment_metadata()?;

        // Validate WebM codec compatibility
        self.validate_webm_tracks()?;

        self.state = DemuxerState::Clusters;
        Ok(())
    }

    /// Validate that all tracks use WebM-compatible codecs.
    fn validate_webm_tracks(&self) -> Result<()> {
        for track in self.tracks.values() {
            if !elements::is_webm_compatible_codec(&track.codec_id) {
                return Err(WebmError::InvalidWebM(format!(
                    "Codec '{}' is not WebM-compatible",
                    track.codec_id
                )));
            }
        }
        Ok(())
    }

    /// Parse segment metadata (info, tracks, cues, etc.).
    fn parse_segment_metadata(&mut self) -> Result<()> {
        let segment_end = self
            .segment_size
            .map(|s| self.segment_start + s)
            .unwrap_or(u64::MAX);

        while self.position()? < segment_end {
            let pos = self.position()?;
            let header = match ElementHeader::read(&mut self.reader) {
                Ok(h) => h,
                Err(_) => break, // EOF
            };

            let size = header.size.unwrap_or(0);

            match header.id {
                SEEK_HEAD => {
                    self.parse_seek_head(size)?;
                }
                INFO => {
                    self.parse_info(size)?;
                }
                TRACKS => {
                    self.parse_tracks(size)?;
                }
                CUES => {
                    self.parse_cues(size)?;
                }
                CLUSTER => {
                    // Reached first cluster, seek back and stop
                    self.reader.seek(SeekFrom::Start(pos))?;
                    break;
                }
                VOID | CRC32 => {
                    self.skip(size)?;
                }
                _ => {
                    self.skip(size)?;
                }
            }
        }

        Ok(())
    }

    /// Parse SeekHead element.
    fn parse_seek_head(&mut self, size: u64) -> Result<()> {
        let end_pos = self.position()? + size;

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            if header.id == SEEK {
                self.parse_seek_entry(child_size)?;
            } else {
                self.skip(child_size)?;
            }
        }

        Ok(())
    }

    /// Parse a single Seek entry.
    fn parse_seek_entry(&mut self, size: u64) -> Result<()> {
        let end_pos = self.position()? + size;
        let mut seek_id: Option<u32> = None;
        let mut seek_position: Option<u64> = None;

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            match header.id {
                SEEK_ID => {
                    let data = self.read_bytes(child_size)?;
                    seek_id = Some(ebml::read_unsigned_int(&data) as u32);
                }
                SEEK_POSITION => {
                    seek_position = Some(self.read_uint(child_size)?);
                }
                _ => {
                    self.skip(child_size)?;
                }
            }
        }

        if let (Some(id), Some(pos)) = (seek_id, seek_position) {
            self.seek_positions.insert(id, pos);
        }

        Ok(())
    }

    /// Parse Info element.
    fn parse_info(&mut self, size: u64) -> Result<()> {
        let end_pos = self.position()? + size;

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            match header.id {
                SEGMENT_UID => {
                    let data = self.read_bytes(child_size)?;
                    if data.len() == 16 {
                        let mut uid = [0u8; 16];
                        uid.copy_from_slice(&data);
                        self.segment_info.uid = Some(uid);
                    }
                }
                SEGMENT_FILENAME => {
                    self.segment_info.filename = Some(self.read_string(child_size)?);
                }
                TITLE => {
                    self.segment_info.title = Some(self.read_string(child_size)?);
                }
                MUXING_APP => {
                    self.segment_info.muxing_app = Some(self.read_string(child_size)?);
                }
                WRITING_APP => {
                    self.segment_info.writing_app = Some(self.read_string(child_size)?);
                }
                TIMECODE_SCALE => {
                    self.segment_info.timecode_scale = self.read_uint(child_size)?;
                }
                DURATION => {
                    let duration = self.read_float(child_size)?;
                    self.segment_info.duration_ns =
                        Some((duration * self.segment_info.timecode_scale as f64) as u64);
                }
                DATE_UTC => {
                    self.segment_info.date = Some(self.read_int(child_size)?);
                }
                _ => {
                    self.skip(child_size)?;
                }
            }
        }

        Ok(())
    }

    /// Parse Tracks element.
    fn parse_tracks(&mut self, size: u64) -> Result<()> {
        let end_pos = self.position()? + size;

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            if header.id == TRACK_ENTRY {
                let track = self.parse_track_entry(child_size)?;
                self.tracks.insert(track.number, track);
            } else {
                self.skip(child_size)?;
            }
        }

        Ok(())
    }

    /// Parse a single TrackEntry.
    fn parse_track_entry(&mut self, size: u64) -> Result<TrackInfo> {
        let end_pos = self.position()? + size;
        let mut track = TrackInfo::default();

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            match header.id {
                TRACK_NUMBER => {
                    track.number = self.read_uint(child_size)?;
                }
                TRACK_UID => {
                    track.uid = self.read_uint(child_size)?;
                }
                TRACK_TYPE => {
                    let track_type = self.read_uint(child_size)? as u8;
                    track.track_type = match track_type {
                        TRACK_TYPE_VIDEO => StreamType::Video,
                        TRACK_TYPE_AUDIO => StreamType::Audio,
                        TRACK_TYPE_SUBTITLE => StreamType::Subtitle,
                        _ => StreamType::Data,
                    };
                }
                CODEC_ID => {
                    track.codec_id = self.read_string(child_size)?;
                }
                CODEC_PRIVATE => {
                    track.codec_private = Some(self.read_bytes(child_size)?);
                }
                NAME => {
                    track.name = Some(self.read_string(child_size)?);
                }
                LANGUAGE | LANGUAGE_IETF => {
                    track.language = Some(self.read_string(child_size)?);
                }
                DEFAULT_DURATION => {
                    track.default_duration = Some(self.read_uint(child_size)?);
                }
                CODEC_DELAY => {
                    track.codec_delay = self.read_uint(child_size)?;
                }
                SEEK_PRE_ROLL => {
                    track.seek_pre_roll = self.read_uint(child_size)?;
                }
                FLAG_DEFAULT => {
                    track.is_default = self.read_uint(child_size)? != 0;
                }
                FLAG_FORCED => {
                    track.is_forced = self.read_uint(child_size)? != 0;
                }
                FLAG_ENABLED => {
                    track.is_enabled = self.read_uint(child_size)? != 0;
                }
                VIDEO => {
                    track.video = Some(self.parse_video_settings(child_size)?);
                }
                AUDIO => {
                    track.audio = Some(self.parse_audio_settings(child_size)?);
                }
                _ => {
                    self.skip(child_size)?;
                }
            }
        }

        // Set codec type based on codec_id
        if track.track_type == StreamType::Video {
            if let Some(ref mut video) = track.video {
                video.codec = elements::video_codec_from_webm_id(&track.codec_id);
            }
        } else if track.track_type == StreamType::Audio {
            if let Some(ref mut audio) = track.audio {
                audio.codec = elements::audio_codec_from_webm_id(&track.codec_id);
            }
        }

        Ok(track)
    }

    /// Parse Video element.
    fn parse_video_settings(&mut self, size: u64) -> Result<VideoTrackInfo> {
        let end_pos = self.position()? + size;
        let mut video = VideoTrackInfo {
            width: 0,
            height: 0,
            display_width: None,
            display_height: None,
            interlaced: false,
            codec: None,
        };

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            match header.id {
                PIXEL_WIDTH => {
                    video.width = self.read_uint(child_size)? as u32;
                }
                PIXEL_HEIGHT => {
                    video.height = self.read_uint(child_size)? as u32;
                }
                DISPLAY_WIDTH => {
                    video.display_width = Some(self.read_uint(child_size)? as u32);
                }
                DISPLAY_HEIGHT => {
                    video.display_height = Some(self.read_uint(child_size)? as u32);
                }
                FLAG_INTERLACED => {
                    video.interlaced = self.read_uint(child_size)? != 0;
                }
                _ => {
                    self.skip(child_size)?;
                }
            }
        }

        Ok(video)
    }

    /// Parse Audio element.
    fn parse_audio_settings(&mut self, size: u64) -> Result<AudioTrackInfo> {
        let end_pos = self.position()? + size;
        let mut audio = AudioTrackInfo {
            sample_rate: 8000.0,
            output_sample_rate: None,
            channels: 1,
            bit_depth: None,
            codec: None,
        };

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            match header.id {
                SAMPLING_FREQUENCY => {
                    audio.sample_rate = self.read_float(child_size)?;
                }
                OUTPUT_SAMPLING_FREQUENCY => {
                    audio.output_sample_rate = Some(self.read_float(child_size)?);
                }
                CHANNELS => {
                    audio.channels = self.read_uint(child_size)? as u32;
                }
                BIT_DEPTH => {
                    audio.bit_depth = Some(self.read_uint(child_size)? as u32);
                }
                _ => {
                    self.skip(child_size)?;
                }
            }
        }

        Ok(audio)
    }

    /// Parse Cues element.
    fn parse_cues(&mut self, size: u64) -> Result<()> {
        let end_pos = self.position()? + size;

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            if header.id == CUE_POINT {
                if let Ok(cue) = self.parse_cue_point(child_size) {
                    self.cues.push(cue);
                }
            } else {
                self.skip(child_size)?;
            }
        }

        Ok(())
    }

    /// Parse a single CuePoint.
    fn parse_cue_point(&mut self, size: u64) -> Result<CuePoint> {
        let end_pos = self.position()? + size;
        let mut cue = CuePoint {
            time: 0,
            positions: Vec::new(),
        };

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            match header.id {
                CUE_TIME => {
                    cue.time = self.read_uint(child_size)?;
                }
                CUE_TRACK_POSITIONS => {
                    cue.positions.push(self.parse_cue_track_positions(child_size)?);
                }
                _ => {
                    self.skip(child_size)?;
                }
            }
        }

        Ok(cue)
    }

    /// Parse CueTrackPositions.
    fn parse_cue_track_positions(&mut self, size: u64) -> Result<CueTrackPosition> {
        let end_pos = self.position()? + size;
        let mut pos = CueTrackPosition {
            track: 0,
            cluster_position: 0,
            relative_position: None,
            block_number: None,
        };

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            match header.id {
                CUE_TRACK => {
                    pos.track = self.read_uint(child_size)?;
                }
                CUE_CLUSTER_POSITION => {
                    pos.cluster_position = self.read_uint(child_size)?;
                }
                CUE_RELATIVE_POSITION => {
                    pos.relative_position = Some(self.read_uint(child_size)?);
                }
                CUE_BLOCK_NUMBER => {
                    pos.block_number = Some(self.read_uint(child_size)?);
                }
                _ => {
                    self.skip(child_size)?;
                }
            }
        }

        Ok(pos)
    }

    /// Read the next packet from the stream.
    pub fn read_packet(&mut self) -> Result<Option<Packet<'static>>> {
        // First, check for buffered frames from a previous laced block
        if let Some(frame) = self.frame_buffer.pop() {
            return Ok(Some(Self::buffered_frame_to_packet(frame)));
        }

        if self.state != DemuxerState::Clusters {
            self.read_segment_info()?;
        }

        loop {
            // If we're not in a cluster, find the next one
            if !self.in_cluster && !self.find_next_cluster()? {
                self.state = DemuxerState::Eof;
                return Ok(None);
            }

            // Read blocks from current cluster
            if let Some(packet) = self.read_next_block()? {
                return Ok(Some(packet));
            }

            // No more blocks in cluster, move to next
            self.in_cluster = false;
        }
    }

    /// Convert a buffered frame to a packet.
    fn buffered_frame_to_packet(frame: BufferedFrame) -> Packet<'static> {
        let mut packet = Packet::new(frame.data);
        packet.pts = Timestamp::new(frame.pts_ns as i64, WEBM_TIME_BASE);
        packet.dts = packet.pts;
        packet.stream_index = frame.track_number as u32;

        if frame.keyframe {
            packet.flags.insert(PacketFlags::KEYFRAME);
        }
        if frame.discardable {
            packet.flags.insert(PacketFlags::DISPOSABLE);
        }
        if let Some(duration_ns) = frame.duration_ns {
            packet.duration = Duration::new(duration_ns as i64, WEBM_TIME_BASE);
        }

        packet
    }

    /// Find the next cluster element.
    fn find_next_cluster(&mut self) -> Result<bool> {
        let segment_end = self
            .segment_size
            .map(|s| self.segment_start + s)
            .unwrap_or(u64::MAX);

        while self.position()? < segment_end {
            let header = match ElementHeader::read(&mut self.reader) {
                Ok(h) => h,
                Err(_) => return Ok(false),
            };

            if header.id == CLUSTER {
                // For unknown size clusters, use segment_end as the boundary
                // We'll stop when we hit another top-level element in read_next_block
                self.current_cluster_end = match header.size {
                    Some(size) => self.position()? + size,
                    None => segment_end, // Unknown size - read until segment end or next cluster
                };
                self.in_cluster = true;
                self.current_cluster_timestamp = 0;
                return Ok(true);
            } else {
                // Skip non-cluster elements
                if let Some(size) = header.size {
                    self.skip(size)?;
                }
            }
        }

        Ok(false)
    }

    /// Read the next block from the current cluster.
    fn read_next_block(&mut self) -> Result<Option<Packet<'static>>> {
        while self.position()? < self.current_cluster_end {
            let pos = self.position()?;
            let header = match ElementHeader::read(&mut self.reader) {
                Ok(h) => h,
                Err(_) => return Ok(None),
            };

            let size = header.size.unwrap_or(0);

            match header.id {
                TIMESTAMP => {
                    self.current_cluster_timestamp = self.read_uint(size)?;
                }
                SIMPLE_BLOCK => {
                    return Ok(Some(self.parse_simple_block(size)?));
                }
                BLOCK_GROUP => {
                    return self.parse_block_group(size);
                }
                // If we hit a top-level element, we've reached the end of this cluster
                CLUSTER | CUES | TAGS | SEEK_HEAD => {
                    // Seek back to the start of this element so find_next_cluster can read it
                    self.reader.seek(SeekFrom::Start(pos))?;
                    return Ok(None);
                }
                _ => {
                    self.skip(size)?;
                }
            }
        }

        Ok(None)
    }

    /// Parse a SimpleBlock.
    fn parse_simple_block(&mut self, size: u64) -> Result<Packet<'static>> {
        let data = self.read_bytes(size)?;
        if data.len() < 4 {
            return Err(WebmError::InvalidBlock("SimpleBlock too small".to_string()));
        }

        // Parse block header
        // Track number (VINT)
        let mut cursor = std::io::Cursor::new(&data);
        let (track_number, vint_len) = ebml::read_vint(&mut cursor)?;

        // Relative timestamp (signed 16-bit)
        let pos = vint_len;
        if data.len() < pos + 3 {
            return Err(WebmError::InvalidBlock(
                "SimpleBlock header too small".to_string(),
            ));
        }
        let relative_ts = i16::from_be_bytes([data[pos], data[pos + 1]]);

        // Flags
        let flags = data[pos + 2];
        let keyframe = (flags & 0x80) != 0;
        let _invisible = (flags & 0x08) != 0;
        let lacing_type = (flags >> 1) & 0x03;
        let discardable = (flags & 0x01) != 0;

        // Calculate absolute timestamp in nanoseconds
        let abs_timestamp =
            (self.current_cluster_timestamp as i64 + relative_ts as i64).max(0) as u64;
        let timestamp_ns = abs_timestamp * self.segment_info.timecode_scale;

        // Get frame duration if known
        let frame_duration_ns = self
            .tracks
            .get(&track_number)
            .and_then(|t| t.default_duration);

        // Frame data starts after header
        let header_size = pos + 3;

        if lacing_type == lacing::NO_LACING {
            // No lacing - single frame
            let frame_data = data[header_size..].to_vec();

            let mut packet = Packet::new(frame_data);
            packet.pts = Timestamp::new(timestamp_ns as i64, WEBM_TIME_BASE);
            packet.dts = packet.pts;
            packet.stream_index = track_number as u32;

            if keyframe {
                packet.flags.insert(PacketFlags::KEYFRAME);
            }
            if discardable {
                packet.flags.insert(PacketFlags::DISPOSABLE);
            }
            if let Some(duration) = frame_duration_ns {
                packet.duration = Duration::new(duration as i64, WEBM_TIME_BASE);
            }

            Ok(packet)
        } else {
            // Handle lacing (Xiph, EBML, or fixed-size)
            let frames = Self::parse_laced_frames(&data, header_size, lacing_type)?;

            if frames.is_empty() {
                return Err(WebmError::InvalidLacing("No frames in laced block".to_string()));
            }

            // Buffer all frames except the first one (in reverse order for pop efficiency)
            let num_frames = frames.len();
            for (i, frame_data) in frames.into_iter().enumerate().rev() {
                let frame_pts_ns = if let Some(duration) = frame_duration_ns {
                    timestamp_ns + (i as u64 * duration)
                } else {
                    timestamp_ns // All frames get the same timestamp if no duration known
                };

                if i == 0 {
                    // Return the first frame directly
                    let mut packet = Packet::new(frame_data);
                    packet.pts = Timestamp::new(frame_pts_ns as i64, WEBM_TIME_BASE);
                    packet.dts = packet.pts;
                    packet.stream_index = track_number as u32;

                    if keyframe {
                        packet.flags.insert(PacketFlags::KEYFRAME);
                    }
                    if discardable {
                        packet.flags.insert(PacketFlags::DISPOSABLE);
                    }
                    if let Some(duration) = frame_duration_ns {
                        packet.duration = Duration::new(duration as i64, WEBM_TIME_BASE);
                    }

                    return Ok(packet);
                } else {
                    // Buffer remaining frames
                    self.frame_buffer.push(BufferedFrame {
                        data: frame_data,
                        track_number,
                        pts_ns: frame_pts_ns,
                        keyframe: keyframe && i == 0, // Only first frame is keyframe
                        discardable,
                        duration_ns: frame_duration_ns,
                    });
                }
            }

            // This should never happen, but handle it gracefully
            Err(WebmError::InvalidLacing(format!(
                "Failed to extract frames from laced block with {} frames",
                num_frames
            )))
        }
    }

    /// Parse a BlockGroup.
    fn parse_block_group(&mut self, size: u64) -> Result<Option<Packet<'static>>> {
        let end_pos = self.position()? + size;
        let mut block_data: Option<Vec<u8>> = None;
        let mut block_duration: Option<u64> = None;
        let mut reference_blocks: Vec<i64> = Vec::new();

        while self.position()? < end_pos {
            let header = ElementHeader::read(&mut self.reader)?;
            let child_size = header.size.unwrap_or(0);

            match header.id {
                BLOCK => {
                    block_data = Some(self.read_bytes(child_size)?);
                }
                BLOCK_DURATION => {
                    block_duration = Some(self.read_uint(child_size)?);
                }
                REFERENCE_BLOCK => {
                    reference_blocks.push(self.read_int(child_size)?);
                }
                _ => {
                    self.skip(child_size)?;
                }
            }
        }

        let data = match block_data {
            Some(d) => d,
            None => return Ok(None),
        };

        if data.len() < 4 {
            return Err(WebmError::InvalidBlock("Block too small".to_string()));
        }

        // Parse block header (same as SimpleBlock but without keyframe flag)
        let mut cursor = std::io::Cursor::new(&data);
        let (track_number, vint_len) = ebml::read_vint(&mut cursor)?;

        let pos = vint_len;
        if data.len() < pos + 3 {
            return Err(WebmError::InvalidBlock(
                "Block header too small".to_string(),
            ));
        }
        let relative_ts = i16::from_be_bytes([data[pos], data[pos + 1]]);
        let flags = data[pos + 2];
        let lacing_type = (flags >> 1) & 0x03;

        let header_size = pos + 3;

        // Calculate absolute timestamp in nanoseconds
        let abs_timestamp =
            (self.current_cluster_timestamp as i64 + relative_ts as i64).max(0) as u64;
        let timestamp_ns = abs_timestamp * self.segment_info.timecode_scale;

        // Keyframe if no reference blocks
        let keyframe = reference_blocks.is_empty();

        // Determine frame duration
        let frame_duration_ns = if let Some(duration) = block_duration {
            Some(duration * self.segment_info.timecode_scale)
        } else {
            self.tracks
                .get(&track_number)
                .and_then(|t| t.default_duration)
        };

        if lacing_type == lacing::NO_LACING {
            // No lacing - single frame
            let frame_data = data[header_size..].to_vec();

            let mut packet = Packet::new(frame_data);
            packet.pts = Timestamp::new(timestamp_ns as i64, WEBM_TIME_BASE);
            packet.dts = packet.pts;
            packet.stream_index = track_number as u32;

            if keyframe {
                packet.flags.insert(PacketFlags::KEYFRAME);
            }
            if let Some(duration) = frame_duration_ns {
                packet.duration = Duration::new(duration as i64, WEBM_TIME_BASE);
            }

            Ok(Some(packet))
        } else {
            // Handle lacing
            let frames = Self::parse_laced_frames(&data, header_size, lacing_type)?;

            if frames.is_empty() {
                return Err(WebmError::InvalidLacing("No frames in laced block".to_string()));
            }

            // For BlockGroup, block_duration applies to the entire block
            let per_frame_duration_ns = frame_duration_ns.map(|d| d / frames.len() as u64);

            // Buffer all frames except the first one
            let num_frames = frames.len();
            for (i, frame_data) in frames.into_iter().enumerate().rev() {
                let frame_pts_ns = if let Some(duration) = per_frame_duration_ns {
                    timestamp_ns + (i as u64 * duration)
                } else {
                    timestamp_ns
                };

                if i == 0 {
                    let mut packet = Packet::new(frame_data);
                    packet.pts = Timestamp::new(frame_pts_ns as i64, WEBM_TIME_BASE);
                    packet.dts = packet.pts;
                    packet.stream_index = track_number as u32;

                    if keyframe {
                        packet.flags.insert(PacketFlags::KEYFRAME);
                    }
                    if let Some(duration) = per_frame_duration_ns {
                        packet.duration = Duration::new(duration as i64, WEBM_TIME_BASE);
                    }

                    return Ok(Some(packet));
                } else {
                    self.frame_buffer.push(BufferedFrame {
                        data: frame_data,
                        track_number,
                        pts_ns: frame_pts_ns,
                        keyframe: keyframe && i == 0,
                        discardable: false,
                        duration_ns: per_frame_duration_ns,
                    });
                }
            }

            Err(WebmError::InvalidLacing(format!(
                "Failed to extract frames from laced block with {} frames",
                num_frames
            )))
        }
    }

    /// Seek to a specific timestamp.
    pub fn seek(&mut self, timestamp_ns: i64) -> Result<()> {
        // Convert to timecode units
        let timecode = (timestamp_ns as u64) / self.segment_info.timecode_scale;

        // Find the best cue point
        let cue = self
            .cues
            .iter()
            .filter(|c| c.time <= timecode)
            .max_by_key(|c| c.time);

        let cluster_pos = match cue {
            Some(c) if !c.positions.is_empty() => c.positions[0].cluster_position,
            _ => {
                // No cue found, seek to beginning
                self.reader.seek(SeekFrom::Start(self.segment_start))?;
                self.in_cluster = false;
                self.frame_buffer.clear();
                return Ok(());
            }
        };

        // Seek to cluster position (relative to segment start)
        self.reader
            .seek(SeekFrom::Start(self.segment_start + cluster_pos))?;
        self.in_cluster = false;
        self.frame_buffer.clear();

        Ok(())
    }

    /// Get the duration of the file in nanoseconds.
    pub fn duration_ns(&self) -> Option<u64> {
        self.segment_info.duration_ns
    }

    /// Get the number of tracks.
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Get track info by track number.
    pub fn track(&self, number: u64) -> Option<&TrackInfo> {
        self.tracks.get(&number)
    }

    // Helper methods for reading EBML data types

    fn read_bytes(&mut self, size: u64) -> Result<Vec<u8>> {
        if size > MAX_ELEMENT_SIZE {
            return Err(WebmError::InvalidElementSize {
                offset: 0, // Offset not available in this context
                message: format!(
                    "element size {} exceeds maximum allowed {}",
                    size, MAX_ELEMENT_SIZE
                ),
            });
        }
        let mut data = vec![0u8; size as usize];
        self.reader.read_exact(&mut data)?;
        Ok(data)
    }

    fn read_uint(&mut self, size: u64) -> Result<u64> {
        let data = self.read_bytes(size)?;
        Ok(ebml::read_unsigned_int(&data))
    }

    fn read_int(&mut self, size: u64) -> Result<i64> {
        let data = self.read_bytes(size)?;
        Ok(ebml::read_signed_int(&data))
    }

    fn read_float(&mut self, size: u64) -> Result<f64> {
        let data = self.read_bytes(size)?;
        Ok(ebml::read_float(&data))
    }

    fn read_string(&mut self, size: u64) -> Result<String> {
        let data = self.read_bytes(size)?;
        ebml::read_string(&data)
    }

    fn skip(&mut self, size: u64) -> Result<()> {
        self.reader.seek(SeekFrom::Current(size as i64))?;
        Ok(())
    }

    // Lacing parsing helper functions

    /// Parse frame sizes from Xiph-style lacing.
    fn parse_xiph_lacing(data: &[u8], num_frames: usize) -> Result<(Vec<usize>, usize)> {
        let mut frame_sizes = Vec::with_capacity(num_frames);
        let mut offset = 0;

        for _ in 0..num_frames - 1 {
            let mut size = 0usize;
            loop {
                if offset >= data.len() {
                    return Err(WebmError::InvalidLacing(
                        "Xiph lacing: unexpected end of data".to_string(),
                    ));
                }
                let byte = data[offset] as usize;
                offset += 1;
                size += byte;
                if byte < 255 {
                    break;
                }
            }
            frame_sizes.push(size);
        }

        Ok((frame_sizes, offset))
    }

    /// Parse frame sizes from EBML-style lacing.
    fn parse_ebml_lacing(data: &[u8], num_frames: usize) -> Result<(Vec<usize>, usize)> {
        use std::io::Cursor;

        let mut frame_sizes = Vec::with_capacity(num_frames);
        let mut cursor = Cursor::new(data);

        // First frame size is an unsigned VINT
        let (first_size, _) = ebml::read_vint(&mut cursor)?;
        frame_sizes.push(first_size as usize);

        let mut prev_size = first_size as i64;

        // Subsequent sizes are signed deltas
        for _ in 1..num_frames - 1 {
            let (raw_delta, vint_len) = ebml::read_vint(&mut cursor)?;
            let delta = Self::vint_to_signed_delta(raw_delta, vint_len);
            prev_size += delta;
            if prev_size < 0 {
                return Err(WebmError::InvalidLacing(
                    "EBML lacing: negative frame size".to_string(),
                ));
            }
            frame_sizes.push(prev_size as usize);
        }

        Ok((frame_sizes, cursor.position() as usize))
    }

    /// Convert an unsigned VINT value to a signed delta.
    fn vint_to_signed_delta(value: u64, vint_length: usize) -> i64 {
        let bits = 7 * vint_length - 1;
        let bias = (1_i64 << bits) - 1;
        (value as i64) - bias
    }

    /// Parse frame sizes from fixed-size lacing.
    fn parse_fixed_lacing(total_data_size: usize, num_frames: usize) -> Result<Vec<usize>> {
        if total_data_size % num_frames != 0 {
            return Err(WebmError::InvalidLacing(format!(
                "Fixed lacing: data size {} not evenly divisible by {} frames",
                total_data_size, num_frames
            )));
        }
        let frame_size = total_data_size / num_frames;
        Ok(vec![frame_size; num_frames])
    }

    /// Parse laced frames from block data.
    fn parse_laced_frames(
        data: &[u8],
        header_size: usize,
        lacing_type: u8,
    ) -> Result<Vec<Vec<u8>>> {
        if data.len() <= header_size {
            return Err(WebmError::InvalidLacing("No data after block header".to_string()));
        }

        // Number of frames = (value + 1)
        let num_frames = data[header_size] as usize + 1;
        let lacing_data_start = header_size + 1;

        if num_frames == 0 {
            return Err(WebmError::InvalidLacing("Zero frames in laced block".to_string()));
        }

        let frame_data = &data[lacing_data_start..];

        let frame_sizes: Vec<usize> = match lacing_type {
            lacing::XIPH => {
                let (mut sizes, sizes_len) = Self::parse_xiph_lacing(frame_data, num_frames)?;
                let used_data: usize = sizes.iter().sum();
                let remaining = frame_data.len().saturating_sub(sizes_len + used_data);
                sizes.push(remaining);
                let frames_start = lacing_data_start + sizes_len;
                return Self::extract_frames(&data[frames_start..], &sizes);
            }
            lacing::EBML => {
                let (mut sizes, sizes_len) = Self::parse_ebml_lacing(frame_data, num_frames)?;
                let used_data: usize = sizes.iter().sum();
                let remaining = frame_data.len().saturating_sub(sizes_len + used_data);
                sizes.push(remaining);
                let frames_start = lacing_data_start + sizes_len;
                return Self::extract_frames(&data[frames_start..], &sizes);
            }
            lacing::FIXED => {
                Self::parse_fixed_lacing(frame_data.len(), num_frames)?
            }
            _ => {
                return Err(WebmError::InvalidLacing(format!(
                    "Unknown lacing type: {}",
                    lacing_type
                )));
            }
        };

        Self::extract_frames(frame_data, &frame_sizes)
    }

    /// Extract frame data given sizes.
    fn extract_frames(data: &[u8], sizes: &[usize]) -> Result<Vec<Vec<u8>>> {
        let mut frames = Vec::with_capacity(sizes.len());
        let mut offset = 0;

        for &size in sizes {
            if offset + size > data.len() {
                return Err(WebmError::InvalidLacing(format!(
                    "Frame size {} exceeds remaining data {} at offset {}",
                    size,
                    data.len() - offset,
                    offset
                )));
            }
            frames.push(data[offset..offset + size].to_vec());
            offset += size;
        }

        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_minimal_webm() -> Vec<u8> {
        let mut data = Vec::new();

        // EBML header
        data.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]); // EBML ID
        data.push(0x8F); // Size (15 bytes)

        // EBMLVersion = 1
        data.extend_from_slice(&[0x42, 0x86, 0x81, 0x01]);
        // EBMLReadVersion = 1
        data.extend_from_slice(&[0x42, 0xF7, 0x81, 0x01]);
        // EBMLMaxIDLength = 4
        data.extend_from_slice(&[0x42, 0xF2, 0x81, 0x04]);
        // EBMLMaxSizeLength = 8
        data.extend_from_slice(&[0x42, 0xF3, 0x81, 0x08]);
        // DocType = "webm"
        data.extend_from_slice(&[0x42, 0x82, 0x84]);
        data.extend_from_slice(b"webm");
        // DocTypeVersion = 4
        data.extend_from_slice(&[0x42, 0x87, 0x81, 0x04]);
        // DocTypeReadVersion = 2
        data.extend_from_slice(&[0x42, 0x85, 0x81, 0x02]);

        // Segment
        data.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]); // Segment ID
        data.push(0xFF); // Unknown size

        data
    }

    #[test]
    fn test_demuxer_creation() {
        let data = create_minimal_webm();
        let cursor = Cursor::new(data);
        let demuxer = WebmDemuxer::new(cursor);
        assert_eq!(demuxer.state, DemuxerState::Header);
    }

    #[test]
    fn test_read_ebml_header() {
        let data = create_minimal_webm();
        let cursor = Cursor::new(data);
        let mut demuxer = WebmDemuxer::new(cursor);

        demuxer.read_header().unwrap();

        let header = demuxer.ebml_header.unwrap();
        assert_eq!(header.version, 1);
        assert_eq!(header.read_version, 1);
        assert_eq!(header.max_id_length, 4);
        assert_eq!(header.max_size_length, 8);
        assert_eq!(header.doc_type, "webm");
        assert_eq!(header.doc_type_version, 4);
        assert_eq!(header.doc_type_read_version, 2);
    }

    #[test]
    fn test_segment_info_default() {
        let info = SegmentInfo::default();
        assert_eq!(info.timecode_scale, DEFAULT_TIMECODE_SCALE);
        assert!(info.duration_ns.is_none());
    }

    #[test]
    fn test_track_info_default() {
        let track = TrackInfo::default();
        assert_eq!(track.number, 0);
        assert_eq!(track.track_type, StreamType::Unknown);
        assert!(track.is_default);
        assert!(track.is_enabled);
        assert!(!track.is_forced);
    }

    #[test]
    fn test_xiph_lacing_parsing() {
        let data = vec![
            100,        // Frame 1: 100 bytes
            255, 50,    // Frame 2: 305 bytes (255 + 50)
        ];
        let (sizes, offset) = WebmDemuxer::<Cursor<Vec<u8>>>::parse_xiph_lacing(&data, 3).unwrap();
        assert_eq!(sizes, vec![100, 305]);
        assert_eq!(offset, 3);
    }

    #[test]
    fn test_fixed_lacing_parsing() {
        let sizes = WebmDemuxer::<Cursor<Vec<u8>>>::parse_fixed_lacing(300, 3).unwrap();
        assert_eq!(sizes, vec![100, 100, 100]);
    }

    #[test]
    fn test_fixed_lacing_uneven_error() {
        let result = WebmDemuxer::<Cursor<Vec<u8>>>::parse_fixed_lacing(301, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_vint_to_signed_delta() {
        // Test bias calculation for 1-byte VINT (length=1)
        assert_eq!(WebmDemuxer::<Cursor<Vec<u8>>>::vint_to_signed_delta(63, 1), 0);
        assert_eq!(WebmDemuxer::<Cursor<Vec<u8>>>::vint_to_signed_delta(64, 1), 1);
        assert_eq!(WebmDemuxer::<Cursor<Vec<u8>>>::vint_to_signed_delta(62, 1), -1);

        // Test for 2-byte VINT (length=2)
        assert_eq!(WebmDemuxer::<Cursor<Vec<u8>>>::vint_to_signed_delta(8191, 2), 0);
        assert_eq!(WebmDemuxer::<Cursor<Vec<u8>>>::vint_to_signed_delta(8192, 2), 1);
        assert_eq!(WebmDemuxer::<Cursor<Vec<u8>>>::vint_to_signed_delta(8190, 2), -1);
    }

    #[test]
    fn test_extract_frames() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sizes = vec![3, 4, 3];
        let frames = WebmDemuxer::<Cursor<Vec<u8>>>::extract_frames(&data, &sizes).unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0], vec![1, 2, 3]);
        assert_eq!(frames[1], vec![4, 5, 6, 7]);
        assert_eq!(frames[2], vec![8, 9, 10]);
    }

    #[test]
    fn test_extract_frames_size_error() {
        let data = vec![1, 2, 3, 4, 5];
        let sizes = vec![3, 10]; // 10 exceeds remaining data
        let result = WebmDemuxer::<Cursor<Vec<u8>>>::extract_frames(&data, &sizes);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffered_frame_to_packet() {
        let frame = BufferedFrame {
            data: vec![1, 2, 3, 4],
            track_number: 1,
            pts_ns: 1000000,
            keyframe: true,
            discardable: false,
            duration_ns: Some(33333333),
        };
        let packet = WebmDemuxer::<Cursor<Vec<u8>>>::buffered_frame_to_packet(frame);
        assert_eq!(packet.data(), &[1, 2, 3, 4]);
        assert_eq!(packet.stream_index, 1);
        assert!(packet.flags.contains(PacketFlags::KEYFRAME));
    }
}
