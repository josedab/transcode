//! # transcode-flv
//!
//! FLV (Flash Video) container support for the transcode library.
//!
//! This crate provides demuxing and muxing capabilities for FLV files,
//! commonly used for streaming video over RTMP and HTTP.
//!
//! ## Features
//!
//! - **FLV Header Parsing**: Signature, version, and stream flags
//! - **Tag Parsing**: Audio, video, and script data tags
//! - **Audio Codecs**: AAC (with AudioSpecificConfig), MP3
//! - **Video Codecs**: H.264/AVC, H.265/HEVC (Enhanced FLV)
//! - **AMF0 Metadata**: onMetaData parsing and generation
//! - **Timestamp Handling**: Extended timestamps, wraparound detection
//! - **Seeking Support**: Basic seeking (without index)
//!
//! ## Example: Demuxing an FLV file
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
//! // Print stream information
//! if let Some(video) = demuxer.video_info() {
//!     println!("Video: {} {}x{} @ {}fps",
//!              video.codec.name(), video.width, video.height, video.frame_rate);
//! }
//!
//! if let Some(audio) = demuxer.audio_info() {
//!     println!("Audio: {} {} Hz, {} channels",
//!              audio.format.name(), audio.sample_rate, audio.channels);
//! }
//!
//! // Read packets
//! while let Ok(Some(packet)) = demuxer.read_packet() {
//!     if packet.is_video() && !packet.is_sequence_header {
//!         println!("Video: {}ms, keyframe={}, size={}",
//!                  packet.timestamp_ms, packet.is_keyframe, packet.data.len());
//!     } else if packet.is_audio() && !packet.is_sequence_header {
//!         println!("Audio: {}ms, size={}",
//!                  packet.timestamp_ms, packet.data.len());
//!     }
//! }
//! ```
//!
//! ## Example: Muxing an FLV file
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::BufWriter;
//! use transcode_flv::{FlvMuxer, MuxerConfig, VideoStreamConfig, AudioStreamConfig};
//!
//! let file = File::create("output.flv").unwrap();
//! let writer = BufWriter::new(file);
//!
//! let config = MuxerConfig::default();
//! let mut muxer = FlvMuxer::new(writer, config);
//!
//! // Configure streams
//! muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));
//! muxer.set_audio_config(AudioStreamConfig::aac(48000, 2));
//!
//! // Write header (includes metadata)
//! muxer.write_header().unwrap();
//!
//! // Write sequence headers (from codec configuration)
//! // muxer.write_video_sequence_header(&avc_config).unwrap();
//! // muxer.write_audio_sequence_header(&aac_config).unwrap();
//!
//! // Write packets
//! // muxer.write_video_packet(timestamp_ms, &data, is_keyframe, cts).unwrap();
//! // muxer.write_audio_packet(timestamp_ms, &data).unwrap();
//!
//! // Finalize
//! muxer.finalize().unwrap();
//! ```
//!
//! ## FLV File Structure
//!
//! ```text
//! FLV File
//! ├── Header (9 bytes)
//! │   ├── Signature: "FLV"
//! │   ├── Version: 1
//! │   ├── Flags: has audio, has video
//! │   └── Header size: 9
//! ├── Previous Tag Size 0 (4 bytes, always 0)
//! └── Tags (repeating)
//!     ├── Tag Header (11 bytes)
//!     │   ├── Tag type (8=audio, 9=video, 18=script)
//!     │   ├── Data size (3 bytes)
//!     │   ├── Timestamp (3 bytes + 1 extended)
//!     │   └── Stream ID (3 bytes, always 0)
//!     ├── Tag Data
//!     │   ├── Audio: 1-2 byte header + data
//!     │   ├── Video: 1-5 byte header + data
//!     │   └── Script: AMF0 encoded data
//!     └── Previous Tag Size (4 bytes)
//! ```
//!
//! ## Codec Support
//!
//! ### Video Codecs
//!
//! | CodecID | Codec | Notes |
//! |---------|-------|-------|
//! | 7 | H.264/AVC | Standard FLV |
//! | 12 | H.265/HEVC | Enhanced FLV |
//! | 2 | Sorenson H.263 | Legacy |
//! | 4-5 | VP6 | Legacy |
//!
//! ### Audio Codecs
//!
//! | SoundFormat | Codec | Notes |
//! |-------------|-------|-------|
//! | 10 | AAC | With AudioSpecificConfig |
//! | 2 | MP3 | Standard |
//! | 14 | MP3 8kHz | Low sample rate |
//! | 0-1 | PCM | Raw audio |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod amf;
pub mod audio;
pub mod demuxer;
pub mod error;
pub mod header;
pub mod muxer;
pub mod tag;
pub mod video;

// Re-export main types
pub use amf::{AmfValue, MetadataBuilder};
pub use audio::{AacConfig, AacPacketType, AudioTagHeader, SoundFormat, SoundRate, SoundSize, SoundType};
pub use demuxer::{AudioInfo, FlvDemuxer, FlvPacket, VideoInfo};
pub use error::{FlvError, Result};
pub use header::{is_flv_signature, probe_flv, FlvHeader, FLV_HEADER_SIZE, FLV_SIGNATURE};
pub use muxer::{AudioStreamConfig, FlvMuxer, MuxerConfig, TagWriter, VideoStreamConfig};
pub use tag::{
    read_previous_tag_size, write_previous_tag_size, ExtendedTimestamp, FlvTag, TagHeader,
    TagType, TAG_HEADER_SIZE, TAG_TYPE_AUDIO, TAG_TYPE_SCRIPT_DATA, TAG_TYPE_VIDEO,
};
pub use video::{
    AvcConfig, AvcPacketType, EnhancedPacketType, FrameType, HevcConfig, VideoCodec,
    VideoTagHeader, CODEC_ID_AVC, CODEC_ID_HEVC,
};

/// FLV configuration for the container.
///
/// This is a high-level configuration struct that can be used to set up
/// both muxing and demuxing operations.
#[derive(Debug, Clone, Default)]
pub struct FlvConfig {
    /// Whether to write/expect metadata.
    pub has_metadata: bool,
    /// Whether the file has video.
    pub has_video: bool,
    /// Whether the file has audio.
    pub has_audio: bool,
    /// Video codec (if video is present).
    pub video_codec: Option<VideoCodec>,
    /// Audio format (if audio is present).
    pub audio_format: Option<SoundFormat>,
}

impl FlvConfig {
    /// Create a new empty configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a video-only configuration.
    pub fn video_only(codec: VideoCodec) -> Self {
        Self {
            has_metadata: true,
            has_video: true,
            has_audio: false,
            video_codec: Some(codec),
            audio_format: None,
        }
    }

    /// Create an audio-only configuration.
    pub fn audio_only(format: SoundFormat) -> Self {
        Self {
            has_metadata: true,
            has_video: false,
            has_audio: true,
            video_codec: None,
            audio_format: Some(format),
        }
    }

    /// Create a configuration for AVC video with AAC audio.
    pub fn avc_aac() -> Self {
        Self {
            has_metadata: true,
            has_video: true,
            has_audio: true,
            video_codec: Some(VideoCodec::Avc),
            audio_format: Some(SoundFormat::Aac),
        }
    }

    /// Create a configuration for HEVC video with AAC audio.
    pub fn hevc_aac() -> Self {
        Self {
            has_metadata: true,
            has_video: true,
            has_audio: true,
            video_codec: Some(VideoCodec::Hevc),
            audio_format: Some(SoundFormat::Aac),
        }
    }

    /// Set whether to include metadata.
    pub fn with_metadata(mut self, has_metadata: bool) -> Self {
        self.has_metadata = has_metadata;
        self
    }

    /// Set the video codec.
    pub fn with_video(mut self, codec: VideoCodec) -> Self {
        self.has_video = true;
        self.video_codec = Some(codec);
        self
    }

    /// Set the audio format.
    pub fn with_audio(mut self, format: SoundFormat) -> Self {
        self.has_audio = true;
        self.audio_format = Some(format);
        self
    }
}

/// FLV time base (milliseconds).
pub const FLV_TIME_BASE_MS: u32 = 1000;

/// Maximum FLV timestamp before wraparound (~49.7 days).
pub const FLV_MAX_TIMESTAMP: u32 = 0xFFFF_FFFF;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_is_flv_signature() {
        assert!(is_flv_signature(b"FLV"));
        assert!(is_flv_signature(b"FLV\x01\x05"));
        assert!(!is_flv_signature(b"FL"));
        assert!(!is_flv_signature(b"ABC"));
    }

    #[test]
    fn test_flv_config() {
        let config = FlvConfig::avc_aac();
        assert!(config.has_video);
        assert!(config.has_audio);
        assert_eq!(config.video_codec, Some(VideoCodec::Avc));
        assert_eq!(config.audio_format, Some(SoundFormat::Aac));
    }

    #[test]
    fn test_flv_config_builder() {
        let config = FlvConfig::new()
            .with_video(VideoCodec::Hevc)
            .with_audio(SoundFormat::Mp3)
            .with_metadata(true);

        assert!(config.has_video);
        assert!(config.has_audio);
        assert_eq!(config.video_codec, Some(VideoCodec::Hevc));
        assert_eq!(config.audio_format, Some(SoundFormat::Mp3));
    }

    #[test]
    fn test_muxer_demuxer_roundtrip() {
        // Create FLV data
        let mut buffer = Vec::new();
        {
            let config = MuxerConfig::default();
            let mut muxer = FlvMuxer::new(&mut buffer, config);

            muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));
            muxer.set_audio_config(AudioStreamConfig::aac(48000, 2));
            muxer.write_header().unwrap();

            // Write sequence headers
            let avc_config = vec![
                0x01, 0x64, 0x00, 0x1F, 0xFF, 0xE1, 0x00, 0x04, 0x67, 0x64, 0x00, 0x1F, 0x01, 0x00,
                0x02, 0x68, 0xEF,
            ];
            muxer.write_video_sequence_header(&avc_config).unwrap();

            let aac_config = vec![0x12, 0x10];
            muxer.write_audio_sequence_header(&aac_config).unwrap();

            // Write video frames
            muxer
                .write_video_packet(0, &[0xAB; 100], true, 0)
                .unwrap();
            muxer
                .write_video_packet(33, &[0xCD; 80], false, 33)
                .unwrap();
            muxer
                .write_video_packet(66, &[0xEF; 90], false, 33)
                .unwrap();

            // Write audio frames
            muxer.write_audio_packet(0, &[0x11; 50]).unwrap();
            muxer.write_audio_packet(23, &[0x22; 50]).unwrap();

            muxer.finalize().unwrap();
        }

        // Parse with demuxer
        let cursor = Cursor::new(&buffer);
        let mut demuxer = FlvDemuxer::new(cursor).unwrap();

        assert!(demuxer.has_video());
        assert!(demuxer.has_audio());

        // Count packets
        let mut video_count = 0;
        let mut audio_count = 0;

        while let Ok(Some(packet)) = demuxer.read_packet() {
            if packet.is_video() {
                video_count += 1;
            } else if packet.is_audio() {
                audio_count += 1;
            }
        }

        // 3 video frames + 1 sequence header = 4
        assert_eq!(video_count, 4);
        // 2 audio frames + 1 sequence header = 3
        assert_eq!(audio_count, 3);
    }

    #[test]
    fn test_tag_types() {
        assert_eq!(TagType::Audio.as_u8(), TAG_TYPE_AUDIO);
        assert_eq!(TagType::Video.as_u8(), TAG_TYPE_VIDEO);
        assert_eq!(TagType::ScriptData.as_u8(), TAG_TYPE_SCRIPT_DATA);
    }

    #[test]
    fn test_video_codecs() {
        assert_eq!(VideoCodec::Avc.as_u8(), CODEC_ID_AVC);
        assert_eq!(VideoCodec::Hevc.as_u8(), CODEC_ID_HEVC);
        assert_eq!(VideoCodec::Avc.name(), "H.264/AVC");
        assert_eq!(VideoCodec::Hevc.name(), "H.265/HEVC");
    }

    #[test]
    fn test_audio_formats() {
        assert_eq!(SoundFormat::Aac.name(), "AAC");
        assert_eq!(SoundFormat::Mp3.name(), "MP3");
    }

    #[test]
    fn test_amf_metadata() {
        let script_data = MetadataBuilder::new()
            .duration(60.0)
            .width(1920)
            .height(1080)
            .frame_rate(30.0)
            .build_script_data();

        let metadata = amf::parse_on_metadata(&script_data).unwrap();
        assert_eq!(metadata.get("duration").unwrap().as_number(), Some(60.0));
        assert_eq!(metadata.get("width").unwrap().as_number(), Some(1920.0));
    }

    #[test]
    fn test_header_roundtrip() {
        let original = FlvHeader::audio_video();

        let mut buffer = Vec::new();
        original.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let parsed = FlvHeader::parse(&mut cursor).unwrap();

        assert_eq!(original, parsed);
    }

    #[test]
    fn test_extended_timestamp() {
        let ts = ExtendedTimestamp::from_ms(0x12345678);
        assert_eq!(ts.to_ms(), 0x12345678);
        assert!(ts.is_extended());

        let ts = ExtendedTimestamp::from_ms(0x00123456);
        assert_eq!(ts.to_ms(), 0x00123456);
        assert!(!ts.is_extended());
    }

    #[test]
    fn test_avc_config_parse() {
        let data = [
            0x01, 0x64, 0x00, 0x1F, 0xFF, 0xE1, 0x00, 0x04, 0x67, 0x64, 0x00, 0x1F, 0x01, 0x00,
            0x02, 0x68, 0xEF,
        ];
        let config = AvcConfig::parse(&data).unwrap();

        assert_eq!(config.avc_profile, 0x64);
        assert_eq!(config.avc_level, 0x1F);
        assert_eq!(config.sps.len(), 1);
        assert_eq!(config.pps.len(), 1);
    }

    #[test]
    fn test_aac_config() {
        let config = AacConfig::aac_lc(44100, 2);
        assert_eq!(config.audio_object_type, 2);
        assert_eq!(config.sample_rate(), 44100);
        assert_eq!(config.channels(), 2);
    }
}
