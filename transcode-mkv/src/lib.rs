//! # transcode-mkv
//!
//! MKV/Matroska and WebM container support for the transcode library.
//!
//! This crate provides demuxing and muxing capabilities for:
//! - **Matroska (.mkv)** - Full Matroska container support with all codecs
//! - **WebM (.webm)** - WebM subset with VP8/VP9/AV1 video and Vorbis/Opus audio
//!
//! ## Features
//!
//! - EBML (Extensible Binary Meta Language) parsing and writing
//! - Variable-length integer (VINT) encoding/decoding
//! - Multiple tracks (video, audio, subtitle)
//! - Seeking with cues
//! - Chapters and tags
//! - WebM codec validation
//!
//! ## Example: Demuxing an MKV file
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::BufReader;
//! use transcode_mkv::MkvDemuxer;
//!
//! let file = File::open("video.mkv").unwrap();
//! let reader = BufReader::new(file);
//! let mut demuxer = MkvDemuxer::new(reader);
//!
//! // Read header and segment info
//! demuxer.read_segment_info().unwrap();
//!
//! // Print track information
//! for (num, track) in &demuxer.tracks {
//!     println!("Track {}: {:?} - {}", num, track.track_type, track.codec_id);
//! }
//!
//! // Read packets
//! while let Some(packet) = demuxer.read_packet().unwrap() {
//!     println!("Packet: stream={}, pts={:?}, size={}",
//!              packet.stream_index, packet.pts.to_millis(), packet.size());
//! }
//! ```
//!
//! ## Example: Muxing a WebM file
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::BufWriter;
//! use transcode_mkv::{WebmMuxer, VideoTrackConfig, AudioTrackConfig};
//! use transcode_core::{VideoCodec, AudioCodec, Packet};
//!
//! let file = File::create("output.webm").unwrap();
//! let writer = BufWriter::new(file);
//! let mut muxer = WebmMuxer::new(writer);
//!
//! // Add VP9 video track
//! let video = VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080);
//! muxer.add_video_track(video).unwrap();
//!
//! // Add Opus audio track
//! let audio = AudioTrackConfig::new(2, AudioCodec::Opus, 48000.0, 2);
//! muxer.add_audio_track(audio).unwrap();
//!
//! // Write header
//! muxer.write_header().unwrap();
//!
//! // Write packets...
//! // muxer.write_packet(&packet).unwrap();
//!
//! // Finalize
//! muxer.finalize().unwrap();
//! ```
//!
//! ## Matroska Element Structure
//!
//! ```text
//! EBML Header
//! Segment
//! ├── SeekHead (index to other elements)
//! ├── Info (segment information)
//! ├── Tracks (track definitions)
//! │   └── TrackEntry
//! │       ├── Video
//! │       └── Audio
//! ├── Chapters (chapter markers)
//! ├── Cues (seeking index)
//! ├── Tags (metadata)
//! └── Cluster (media data)
//!     ├── Timestamp
//!     └── SimpleBlock / BlockGroup
//! ```
//!
//! ## Codec Support
//!
//! ### Video Codecs
//!
//! | Codec ID | Description | WebM |
//! |----------|-------------|------|
//! | V_VP8 | VP8 | Yes |
//! | V_VP9 | VP9 | Yes |
//! | V_AV1 | AV1 | Yes |
//! | V_MPEG4/ISO/AVC | H.264/AVC | No |
//! | V_MPEGH/ISO/HEVC | H.265/HEVC | No |
//!
//! ### Audio Codecs
//!
//! | Codec ID | Description | WebM |
//! |----------|-------------|------|
//! | A_OPUS | Opus | Yes |
//! | A_VORBIS | Vorbis | Yes |
//! | A_AAC | AAC | No |
//! | A_FLAC | FLAC | No |
//! | A_MPEG/L3 | MP3 | No |
//! | A_AC3 | AC-3 | No |
//! | A_EAC3 | E-AC-3 | No |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod ebml;
pub mod elements;
pub mod error;
pub mod demuxer;
pub mod muxer;
pub mod webm;

// Re-export main types
pub use demuxer::{
    AudioTrackInfo, Chapter, ChapterDisplay, CuePoint, CueTrackPosition, MkvDemuxer, SegmentInfo,
    SimpleTag, Tag, TrackInfo, VideoTrackInfo, MKV_TIME_BASE,
};

pub use muxer::{
    AudioTrackConfig, MkvMuxer, MuxerChapter, MuxerConfig, SubtitleTrackConfig, TrackConfig,
    VideoTrackConfig,
};

pub use webm::{
    is_webm_audio_codec, is_webm_codec_id, is_webm_video_codec, validate_webm_file,
    validate_webm_track, Av1CodecPrivate, OpusCodecPrivate, VorbisCodecPrivate, Vp9CodecPrivate,
    WebmDemuxer, WebmMuxer, WEBM_AUDIO_CODECS, WEBM_VIDEO_CODECS,
};

pub use ebml::{EbmlHeader, ElementHeader};

pub use error::{MkvError, Result};

/// Check if a file appears to be a valid Matroska/WebM file.
///
/// This performs a quick check by looking for the EBML header signature.
pub fn is_mkv_signature(data: &[u8]) -> bool {
    // EBML header element ID: 0x1A45DFA3
    data.len() >= 4 && data[0..4] == [0x1A, 0x45, 0xDF, 0xA3]
}

/// Detect if a file is MKV or WebM based on the EBML header.
///
/// Returns `Some("matroska")` or `Some("webm")` if detected, `None` otherwise.
pub fn detect_container_type<R: std::io::Read + std::io::Seek>(
    reader: &mut R,
) -> Option<String> {
    // Record start position (unused but kept for future seek-back functionality)
    let _start_pos = reader.stream_position().ok()?;

    // Try to read EBML header
    let mut demuxer = MkvDemuxer::new(reader);
    demuxer.read_header().ok()?;

    // Get doc type
    let doc_type = demuxer.ebml_header.as_ref()?.doc_type.clone();

    // Seek back to start
    let _reader = demuxer.into_inner();
    // Note: Can't seek back here since we consumed the reader
    // Caller should handle this

    Some(doc_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_mkv_signature() {
        // Valid EBML signature
        assert!(is_mkv_signature(&[0x1A, 0x45, 0xDF, 0xA3]));
        assert!(is_mkv_signature(&[0x1A, 0x45, 0xDF, 0xA3, 0x00, 0x00]));

        // Invalid signatures
        assert!(!is_mkv_signature(&[0x00, 0x00, 0x00, 0x00]));
        assert!(!is_mkv_signature(&[0x1A, 0x45, 0xDF])); // Too short
        assert!(!is_mkv_signature(&[]));
    }

    #[test]
    fn test_demuxer_creation() {
        use std::io::Cursor;

        let data = vec![0x1A, 0x45, 0xDF, 0xA3, 0x80]; // Minimal EBML header start
        let cursor = Cursor::new(data);
        let demuxer = MkvDemuxer::new(cursor);

        // Just test that creation works
        assert!(demuxer.tracks.is_empty());
    }

    #[test]
    fn test_muxer_creation() {
        use std::io::Cursor;

        let buffer = Cursor::new(Vec::new());
        let config = MuxerConfig::default();
        let muxer = MkvMuxer::new(buffer, config);

        // Just test that creation works
        let _ = muxer;
    }

    #[test]
    fn test_webm_muxer_creation() {
        use std::io::Cursor;

        let buffer = Cursor::new(Vec::new());
        let muxer = WebmMuxer::new(buffer);

        // Just test that creation works
        let _ = muxer;
    }

    #[test]
    fn test_ebml_header_default() {
        let header = EbmlHeader::default();
        assert_eq!(header.doc_type, "matroska");
        assert!(header.is_matroska());
        assert!(!header.is_webm());
    }

    #[test]
    fn test_ebml_header_webm() {
        let header = EbmlHeader::webm();
        assert_eq!(header.doc_type, "webm");
        assert!(header.is_webm());
        assert!(!header.is_matroska());
    }

    #[test]
    fn test_codec_id_mapping() {
        use elements::codec_ids;

        // Video codecs
        assert_eq!(
            elements::video_codec_to_mkv_id(transcode_core::VideoCodec::Vp9),
            codec_ids::V_VP9
        );
        assert_eq!(
            elements::video_codec_to_mkv_id(transcode_core::VideoCodec::Av1),
            codec_ids::V_AV1
        );
        assert_eq!(
            elements::video_codec_to_mkv_id(transcode_core::VideoCodec::H264),
            codec_ids::V_MPEG4_ISO_AVC
        );

        // Audio codecs
        assert_eq!(
            elements::audio_codec_to_mkv_id(transcode_core::AudioCodec::Opus),
            codec_ids::A_OPUS
        );
        assert_eq!(
            elements::audio_codec_to_mkv_id(transcode_core::AudioCodec::Vorbis),
            codec_ids::A_VORBIS
        );
    }

    #[test]
    fn test_webm_codec_validation() {
        // WebM-compatible
        assert!(is_webm_video_codec(transcode_core::VideoCodec::Vp8));
        assert!(is_webm_video_codec(transcode_core::VideoCodec::Vp9));
        assert!(is_webm_video_codec(transcode_core::VideoCodec::Av1));
        assert!(is_webm_audio_codec(transcode_core::AudioCodec::Opus));
        assert!(is_webm_audio_codec(transcode_core::AudioCodec::Vorbis));

        // Not WebM-compatible
        assert!(!is_webm_video_codec(transcode_core::VideoCodec::H264));
        assert!(!is_webm_video_codec(transcode_core::VideoCodec::H265));
        assert!(!is_webm_audio_codec(transcode_core::AudioCodec::Aac));
        assert!(!is_webm_audio_codec(transcode_core::AudioCodec::Mp3));
    }

    #[test]
    fn test_vint_roundtrip() {
        use std::io::Cursor;

        for value in [0u64, 1, 127, 128, 16383, 16384, 1_000_000, 0xFF_FFFF] {
            let (encoded, len) = ebml::encode_vint(value).unwrap();
            let mut cursor = Cursor::new(&encoded[..len]);
            let (decoded, decoded_len) = ebml::read_vint(&mut cursor).unwrap();
            assert_eq!(value, decoded, "Value {} failed roundtrip", value);
            assert_eq!(len, decoded_len);
        }
    }

    #[test]
    fn test_element_header_roundtrip() {
        use std::io::Cursor;

        let header = ElementHeader {
            id: elements::SEGMENT,
            size: Some(1000),
            header_size: 0,
        };

        let mut buffer = Vec::new();
        header.write(&mut buffer).unwrap();

        let mut cursor = Cursor::new(&buffer);
        let read_header = ElementHeader::read(&mut cursor).unwrap();

        assert_eq!(header.id, read_header.id);
        assert_eq!(header.size, read_header.size);
    }

    #[test]
    fn test_opus_codec_private_builder() {
        let opus = OpusCodecPrivate::new(2, 48000).with_pre_skip(312);
        let data = opus.build();

        // Verify OpusHead signature
        assert_eq!(&data[0..8], b"OpusHead");
        assert_eq!(data[8], 1); // Version
        assert_eq!(data[9], 2); // Channels
    }

    #[test]
    fn test_track_config_creation() {
        let video = VideoTrackConfig::new(1, transcode_core::VideoCodec::Vp9, 1920, 1080);
        assert_eq!(video.track_number, 1);
        assert_eq!(video.width, 1920);
        assert_eq!(video.height, 1080);

        let audio = AudioTrackConfig::new(2, transcode_core::AudioCodec::Opus, 48000.0, 2);
        assert_eq!(audio.track_number, 2);
        assert_eq!(audio.sample_rate, 48000.0);
        assert_eq!(audio.channels, 2);
    }

    #[test]
    fn test_muxer_config_default() {
        let config = MuxerConfig::default();
        assert_eq!(config.doc_type, "matroska");
        assert!(config.write_cues);
        assert!(config.write_chapters);
    }

    #[test]
    fn test_muxer_config_webm() {
        let config = MuxerConfig::webm();
        assert_eq!(config.doc_type, "webm");
    }

    #[test]
    fn test_error_types() {
        let err = MkvError::InvalidElementId { offset: 100 };
        assert!(err.to_string().contains("100"));

        let err = MkvError::TrackNotFound { track_number: 5 };
        assert!(err.to_string().contains("5"));

        let err = MkvError::InvalidWebM("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_segment_info_default() {
        let info = SegmentInfo::default();
        assert_eq!(info.timecode_scale, 1_000_000); // 1ms default
        assert!(info.duration_ns.is_none());
        assert!(info.title.is_none());
    }

    #[test]
    fn test_track_info_default() {
        let track = TrackInfo::default();
        assert_eq!(track.number, 0);
        assert!(track.is_default);
        assert!(track.is_enabled);
        assert!(!track.is_forced);
        assert_eq!(track.codec_delay, 0);
        assert_eq!(track.seek_pre_roll, 0);
    }
}
