//! # transcode-webm
//!
//! WebM container format support for the transcode library.
//!
//! WebM is a subset of the Matroska container format specifically designed for
//! web video. It supports:
//!
//! - **Video codecs**: VP8, VP9, AV1
//! - **Audio codecs**: Vorbis, Opus
//!
//! ## Features
//!
//! - Full EBML parsing with variable-length integer support
//! - Muxing with cluster management and keyframe-based segmentation
//! - Demuxing with seeking via cues
//! - Support for laced frames (Xiph, EBML, fixed-size)
//! - Codec private data handling for all supported codecs
//!
//! ## Example: Muxing a WebM file
//!
//! ```no_run
//! use transcode_webm::{WebmMuxer, VideoTrackConfig, AudioTrackConfig, MuxerConfig};
//! use transcode_core::{VideoCodec, AudioCodec, Packet, PacketFlags};
//! use std::io::Cursor;
//!
//! // Create a muxer
//! let buffer = Cursor::new(Vec::new());
//! let mut muxer = WebmMuxer::new(buffer);
//!
//! // Add tracks
//! muxer.add_video_track(
//!     VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080)
//!         .with_frame_rate(30.0)
//! ).unwrap();
//!
//! muxer.add_audio_track(
//!     AudioTrackConfig::new(2, AudioCodec::Opus, 48000.0, 2)
//! ).unwrap();
//!
//! // Write header
//! muxer.write_header().unwrap();
//!
//! // Write packets...
//!
//! // Finalize
//! muxer.finalize().unwrap();
//! ```
//!
//! ## Example: Demuxing a WebM file
//!
//! ```no_run
//! use transcode_webm::WebmDemuxer;
//! use std::fs::File;
//!
//! let file = File::open("video.webm").unwrap();
//! let mut demuxer = WebmDemuxer::new(file);
//!
//! // Read header and tracks
//! demuxer.read_segment_info().unwrap();
//!
//! // Print track info
//! for (num, track) in &demuxer.tracks {
//!     println!("Track {}: {:?} - {}", num, track.track_type, track.codec_id);
//! }
//!
//! // Read packets
//! while let Some(packet) = demuxer.read_packet().unwrap() {
//!     println!("Packet: track={}, pts={}, size={}",
//!         packet.stream_index, packet.pts.value, packet.data().len());
//! }
//! ```

pub mod ebml;
pub mod elements;
pub mod error;
pub mod muxer;
pub mod demuxer;

// Re-export main types
pub use ebml::{EbmlHeader, ElementHeader};
pub use elements::codec_ids;
pub use error::{WebmError, Result};
pub use muxer::{WebmMuxer, MuxerConfig, VideoTrackConfig, AudioTrackConfig, MuxerState};
pub use demuxer::{WebmDemuxer, DemuxerState, TrackInfo, VideoTrackInfo, AudioTrackInfo};
pub use demuxer::{SegmentInfo, CuePoint, CueTrackPosition};

use transcode_core::{AudioCodec, VideoCodec};

/// WebM-compatible video codecs.
pub const WEBM_VIDEO_CODECS: &[VideoCodec] = &[VideoCodec::Vp8, VideoCodec::Vp9, VideoCodec::Av1];

/// WebM-compatible audio codecs.
pub const WEBM_AUDIO_CODECS: &[AudioCodec] = &[AudioCodec::Vorbis, AudioCodec::Opus];

/// Check if a video codec is WebM-compatible.
pub fn is_webm_video_codec(codec: VideoCodec) -> bool {
    WEBM_VIDEO_CODECS.contains(&codec)
}

/// Check if an audio codec is WebM-compatible.
pub fn is_webm_audio_codec(codec: AudioCodec) -> bool {
    WEBM_AUDIO_CODECS.contains(&codec)
}

/// WebM configuration options.
#[derive(Debug, Clone)]
pub struct WebmConfig {
    /// Timecode scale in nanoseconds (default: 1,000,000 = 1ms).
    pub timecode_scale: u64,
    /// Maximum cluster duration in milliseconds (default: 5000ms = 5s).
    pub max_cluster_duration: u64,
    /// Whether to generate cues for seeking (default: true).
    pub generate_cues: bool,
    /// Writing application name.
    pub writing_app: String,
    /// Document title.
    pub title: Option<String>,
}

impl Default for WebmConfig {
    fn default() -> Self {
        Self {
            timecode_scale: 1_000_000, // 1 millisecond
            max_cluster_duration: 5000, // 5 seconds
            generate_cues: true,
            writing_app: "transcode-webm".to_string(),
            title: None,
        }
    }
}

impl WebmConfig {
    /// Create a new WebM configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the timecode scale (nanoseconds per timecode unit).
    pub fn with_timecode_scale(mut self, scale: u64) -> Self {
        self.timecode_scale = scale;
        self
    }

    /// Set the maximum cluster duration in milliseconds.
    pub fn with_max_cluster_duration(mut self, duration_ms: u64) -> Self {
        self.max_cluster_duration = duration_ms;
        self
    }

    /// Enable or disable cue generation.
    pub fn with_cues(mut self, enabled: bool) -> Self {
        self.generate_cues = enabled;
        self
    }

    /// Set the writing application name.
    pub fn with_writing_app(mut self, app: impl Into<String>) -> Self {
        self.writing_app = app.into();
        self
    }

    /// Set the document title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Convert to MuxerConfig.
    pub fn to_muxer_config(&self) -> MuxerConfig {
        MuxerConfig {
            timecode_scale: self.timecode_scale,
            max_cluster_duration: self.max_cluster_duration,
            generate_cues: self.generate_cues,
            writing_app: self.writing_app.clone(),
            title: self.title.clone(),
        }
    }
}

/// VP9 codec private data builder.
#[derive(Debug, Clone, Default)]
pub struct Vp9CodecPrivate {
    /// Profile (0-3).
    pub profile: u8,
    /// Level.
    pub level: u8,
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,
    /// Chroma subsampling.
    pub chroma_subsampling: u8,
}

impl Vp9CodecPrivate {
    /// Create a new VP9 codec private data builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the profile.
    pub fn with_profile(mut self, profile: u8) -> Self {
        self.profile = profile;
        self
    }

    /// Set the level.
    pub fn with_level(mut self, level: u8) -> Self {
        self.level = level;
        self
    }

    /// Set the bit depth.
    pub fn with_bit_depth(mut self, bit_depth: u8) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Set the chroma subsampling.
    pub fn with_chroma_subsampling(mut self, subsampling: u8) -> Self {
        self.chroma_subsampling = subsampling;
        self
    }

    /// Build the codec private data.
    pub fn build(&self) -> Vec<u8> {
        vec![self.profile, self.level, (self.bit_depth << 4) | self.chroma_subsampling]
    }
}

/// Opus codec private data builder.
#[derive(Debug, Clone)]
pub struct OpusCodecPrivate {
    /// Number of channels.
    pub channels: u8,
    /// Pre-skip samples.
    pub pre_skip: u16,
    /// Sample rate.
    pub sample_rate: u32,
    /// Output gain.
    pub output_gain: i16,
    /// Channel mapping family.
    pub channel_mapping_family: u8,
}

impl Default for OpusCodecPrivate {
    fn default() -> Self {
        Self {
            channels: 2,
            pre_skip: 0,
            sample_rate: 48000,
            output_gain: 0,
            channel_mapping_family: 0,
        }
    }
}

impl OpusCodecPrivate {
    /// Create a new Opus codec private data builder.
    pub fn new(channels: u8, sample_rate: u32) -> Self {
        Self {
            channels,
            sample_rate,
            ..Default::default()
        }
    }

    /// Set the pre-skip.
    pub fn with_pre_skip(mut self, pre_skip: u16) -> Self {
        self.pre_skip = pre_skip;
        self
    }

    /// Set the output gain.
    pub fn with_output_gain(mut self, gain: i16) -> Self {
        self.output_gain = gain;
        self
    }

    /// Set the channel mapping family.
    pub fn with_channel_mapping_family(mut self, family: u8) -> Self {
        self.channel_mapping_family = family;
        self
    }

    /// Build the Opus ID header (codec private data).
    pub fn build(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(19);

        // Magic signature
        data.extend_from_slice(b"OpusHead");

        // Version (must be 1)
        data.push(1);

        // Channel count
        data.push(self.channels);

        // Pre-skip (little-endian)
        data.extend_from_slice(&self.pre_skip.to_le_bytes());

        // Input sample rate (little-endian)
        data.extend_from_slice(&self.sample_rate.to_le_bytes());

        // Output gain (little-endian)
        data.extend_from_slice(&self.output_gain.to_le_bytes());

        // Channel mapping family
        data.push(self.channel_mapping_family);

        data
    }
}

/// Vorbis codec private data builder.
#[derive(Debug, Clone)]
pub struct VorbisCodecPrivate {
    /// Vorbis identification header.
    pub identification_header: Vec<u8>,
    /// Vorbis comment header.
    pub comment_header: Vec<u8>,
    /// Vorbis setup header.
    pub setup_header: Vec<u8>,
}

impl VorbisCodecPrivate {
    /// Create from raw Vorbis headers.
    pub fn from_headers(
        identification: Vec<u8>,
        comment: Vec<u8>,
        setup: Vec<u8>,
    ) -> Self {
        Self {
            identification_header: identification,
            comment_header: comment,
            setup_header: setup,
        }
    }

    /// Build the codec private data in Matroska/WebM format.
    pub fn build(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Number of packets - 1 = 2
        data.push(2);

        // Xiph lacing for identification header size
        let mut size = self.identification_header.len();
        while size >= 255 {
            data.push(255);
            size -= 255;
        }
        data.push(size as u8);

        // Xiph lacing for comment header size
        size = self.comment_header.len();
        while size >= 255 {
            data.push(255);
            size -= 255;
        }
        data.push(size as u8);

        // Headers
        data.extend_from_slice(&self.identification_header);
        data.extend_from_slice(&self.comment_header);
        data.extend_from_slice(&self.setup_header);

        data
    }
}

/// AV1 codec private data builder.
#[derive(Debug, Clone, Default)]
pub struct Av1CodecPrivate {
    /// AV1 configuration record.
    config_record: Vec<u8>,
}

impl Av1CodecPrivate {
    /// Create from an AV1 configuration record.
    pub fn from_config_record(config: Vec<u8>) -> Self {
        Self {
            config_record: config,
        }
    }

    /// Build the codec private data.
    pub fn build(&self) -> Vec<u8> {
        self.config_record.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_webm_video_codec_check() {
        assert!(is_webm_video_codec(VideoCodec::Vp8));
        assert!(is_webm_video_codec(VideoCodec::Vp9));
        assert!(is_webm_video_codec(VideoCodec::Av1));
        assert!(!is_webm_video_codec(VideoCodec::H264));
        assert!(!is_webm_video_codec(VideoCodec::H265));
    }

    #[test]
    fn test_webm_audio_codec_check() {
        assert!(is_webm_audio_codec(AudioCodec::Vorbis));
        assert!(is_webm_audio_codec(AudioCodec::Opus));
        assert!(!is_webm_audio_codec(AudioCodec::Aac));
        assert!(!is_webm_audio_codec(AudioCodec::Mp3));
    }

    #[test]
    fn test_webm_config_default() {
        let config = WebmConfig::default();
        assert_eq!(config.timecode_scale, 1_000_000);
        assert_eq!(config.max_cluster_duration, 5000);
        assert!(config.generate_cues);
    }

    #[test]
    fn test_webm_config_builder() {
        let config = WebmConfig::new()
            .with_timecode_scale(500_000)
            .with_max_cluster_duration(10000)
            .with_cues(false)
            .with_writing_app("test-app")
            .with_title("Test Video");

        assert_eq!(config.timecode_scale, 500_000);
        assert_eq!(config.max_cluster_duration, 10000);
        assert!(!config.generate_cues);
        assert_eq!(config.writing_app, "test-app");
        assert_eq!(config.title, Some("Test Video".to_string()));
    }

    #[test]
    fn test_vp9_codec_private() {
        let vp9 = Vp9CodecPrivate::new()
            .with_profile(0)
            .with_level(31)
            .with_bit_depth(8)
            .with_chroma_subsampling(1);
        let data = vp9.build();

        assert_eq!(data[0], 0); // Profile
        assert_eq!(data[1], 31); // Level
        assert_eq!(data[2] >> 4, 8); // Bit depth
        assert_eq!(data[2] & 0x0F, 1); // Chroma subsampling
    }

    #[test]
    fn test_opus_codec_private() {
        let opus = OpusCodecPrivate::new(2, 48000)
            .with_pre_skip(312)
            .with_output_gain(0);
        let data = opus.build();

        assert_eq!(&data[0..8], b"OpusHead");
        assert_eq!(data[8], 1); // Version
        assert_eq!(data[9], 2); // Channels
        assert_eq!(u16::from_le_bytes([data[10], data[11]]), 312); // Pre-skip
        assert_eq!(
            u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            48000
        ); // Sample rate
    }

    #[test]
    fn test_vorbis_codec_private() {
        let id_header = vec![0x01, 0x76, 0x6F, 0x72, 0x62, 0x69, 0x73];
        let comment_header = vec![0x03, 0x76, 0x6F, 0x72, 0x62, 0x69, 0x73];
        let setup_header = vec![0x05, 0x76, 0x6F, 0x72, 0x62, 0x69, 0x73];

        let vorbis = VorbisCodecPrivate::from_headers(
            id_header,
            comment_header,
            setup_header,
        );
        let data = vorbis.build();

        assert_eq!(data[0], 2); // Number of packets - 1
    }

    #[test]
    fn test_mux_demux_roundtrip() {
        use transcode_core::{Packet, PacketFlags, Timestamp};

        // Create a WebM file in memory
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        // Add a video track
        muxer.add_video_track(
            VideoTrackConfig::new(1, VideoCodec::Vp9, 640, 480)
                .with_frame_rate(30.0)
        ).unwrap();

        muxer.write_header().unwrap();

        // Write some packets
        for i in 0..5 {
            let mut packet = Packet::new(vec![0u8; 100 + i * 10]);
            packet.stream_index = 1;
            packet.pts = Timestamp::new((i as i64) * 33_333_333, muxer::WEBM_TIME_BASE);
            if i == 0 {
                packet.flags.insert(PacketFlags::KEYFRAME);
            }
            muxer.write_packet(&packet).unwrap();
        }

        muxer.finalize().unwrap();

        // Get the data
        let data = muxer.into_inner().into_inner();

        // Verify EBML header
        assert_eq!(&data[0..4], &[0x1A, 0x45, 0xDF, 0xA3]);

        // Now demux it
        let cursor = Cursor::new(data);
        let mut demuxer = WebmDemuxer::new(cursor);
        demuxer.read_segment_info().unwrap();

        // Verify track was parsed
        assert_eq!(demuxer.tracks.len(), 1);
        let track = demuxer.tracks.get(&1).unwrap();
        assert_eq!(track.codec_id, "V_VP9");

        // Read packets
        let mut packet_count = 0;
        while let Ok(Some(_packet)) = demuxer.read_packet() {
            packet_count += 1;
        }

        assert_eq!(packet_count, 5);
    }

    #[test]
    fn test_webm_config_to_muxer_config() {
        let webm_config = WebmConfig::new()
            .with_title("Test")
            .with_cues(false);

        let muxer_config = webm_config.to_muxer_config();

        assert_eq!(muxer_config.title, Some("Test".to_string()));
        assert!(!muxer_config.generate_cues);
    }
}
