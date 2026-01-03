//! WebM subset support.
//!
//! WebM is a subset of Matroska that only supports specific codecs:
//! - Video: VP8, VP9, AV1
//! - Audio: Vorbis, Opus
//!
//! This module provides WebM-specific validation and utilities.

use crate::demuxer::{MkvDemuxer, TrackInfo};
use crate::elements::{self, codec_ids};
use crate::error::{MkvError, Result};
use crate::muxer::{AudioTrackConfig, MkvMuxer, MuxerConfig, VideoTrackConfig};

use std::io::{Read, Seek, Write};

use transcode_core::format::StreamType;
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

/// Check if a Matroska codec ID is WebM-compatible.
pub fn is_webm_codec_id(codec_id: &str) -> bool {
    elements::is_webm_compatible_codec(codec_id)
}

/// Validate that a track is WebM-compatible.
pub fn validate_webm_track(track: &TrackInfo) -> Result<()> {
    match track.track_type {
        StreamType::Video => {
            if let Some(ref video) = track.video {
                if let Some(codec) = video.codec {
                    if !is_webm_video_codec(codec) {
                        return Err(MkvError::InvalidWebM(format!(
                            "Video codec {:?} is not WebM-compatible",
                            codec
                        )));
                    }
                } else if !is_webm_codec_id(&track.codec_id) {
                    return Err(MkvError::InvalidWebM(format!(
                        "Video codec {} is not WebM-compatible",
                        track.codec_id
                    )));
                }
            }
        }
        StreamType::Audio => {
            if let Some(ref audio) = track.audio {
                if let Some(codec) = audio.codec {
                    if !is_webm_audio_codec(codec) {
                        return Err(MkvError::InvalidWebM(format!(
                            "Audio codec {:?} is not WebM-compatible",
                            codec
                        )));
                    }
                } else if !is_webm_codec_id(&track.codec_id) {
                    return Err(MkvError::InvalidWebM(format!(
                        "Audio codec {} is not WebM-compatible",
                        track.codec_id
                    )));
                }
            }
        }
        StreamType::Subtitle => {
            // WebM technically supports WebVTT subtitles
            if track.codec_id != codec_ids::S_TEXT_WEBVTT {
                return Err(MkvError::InvalidWebM(
                    "Only WebVTT subtitles are supported in WebM".to_string(),
                ));
            }
        }
        _ => {
            return Err(MkvError::InvalidWebM(format!(
                "Track type {:?} is not supported in WebM",
                track.track_type
            )));
        }
    }

    Ok(())
}

/// Validate that all tracks in a demuxer are WebM-compatible.
pub fn validate_webm_file<R: Read + Seek>(demuxer: &MkvDemuxer<R>) -> Result<()> {
    for track in demuxer.tracks.values() {
        validate_webm_track(track)?;
    }
    Ok(())
}

/// WebM demuxer (wrapper around MkvDemuxer with validation).
pub struct WebmDemuxer<R: Read + Seek> {
    inner: MkvDemuxer<R>,
}

impl<R: Read + Seek> WebmDemuxer<R> {
    /// Create a new WebM demuxer.
    pub fn new(reader: R) -> Self {
        Self {
            inner: MkvDemuxer::new(reader),
        }
    }

    /// Read and validate the header.
    pub fn read_header(&mut self) -> Result<()> {
        self.inner.read_header()?;

        // Validate doc type
        if let Some(ref header) = self.inner.ebml_header {
            if !header.is_webm() {
                return Err(MkvError::InvalidWebM(format!(
                    "Document type is '{}', not 'webm'",
                    header.doc_type
                )));
            }
        }

        Ok(())
    }

    /// Read segment info and validate tracks.
    pub fn read_segment_info(&mut self) -> Result<()> {
        self.inner.read_segment_info()?;

        // Validate all tracks are WebM-compatible
        validate_webm_file(&self.inner)?;

        Ok(())
    }

    /// Read the next packet.
    pub fn read_packet(&mut self) -> Result<Option<transcode_core::Packet<'static>>> {
        self.inner.read_packet()
    }

    /// Seek to a timestamp.
    pub fn seek(&mut self, timestamp_ns: i64) -> Result<()> {
        self.inner.seek(timestamp_ns)
    }

    /// Get the inner demuxer.
    pub fn inner(&self) -> &MkvDemuxer<R> {
        &self.inner
    }

    /// Get the inner demuxer mutably.
    pub fn inner_mut(&mut self) -> &mut MkvDemuxer<R> {
        &mut self.inner
    }

    /// Consume and return the inner demuxer.
    pub fn into_inner(self) -> MkvDemuxer<R> {
        self.inner
    }
}

/// WebM muxer (wrapper around MkvMuxer with validation).
pub struct WebmMuxer<W: Write + Seek> {
    inner: MkvMuxer<W>,
}

impl<W: Write + Seek> WebmMuxer<W> {
    /// Create a new WebM muxer.
    pub fn new(writer: W) -> Self {
        Self {
            inner: MkvMuxer::new(writer, MuxerConfig::webm()),
        }
    }

    /// Create a new WebM muxer with custom config.
    pub fn with_config(writer: W, mut config: MuxerConfig) -> Self {
        config.doc_type = "webm".to_string();
        Self {
            inner: MkvMuxer::new(writer, config),
        }
    }

    /// Add a video track (must be VP8, VP9, or AV1).
    pub fn add_video_track(&mut self, config: VideoTrackConfig) -> Result<()> {
        if !is_webm_video_codec(config.codec) {
            return Err(MkvError::InvalidWebM(format!(
                "Video codec {:?} is not WebM-compatible. Use VP8, VP9, or AV1.",
                config.codec
            )));
        }
        self.inner.add_video_track(config)
    }

    /// Add an audio track (must be Vorbis or Opus).
    pub fn add_audio_track(&mut self, config: AudioTrackConfig) -> Result<()> {
        if !is_webm_audio_codec(config.codec) {
            return Err(MkvError::InvalidWebM(format!(
                "Audio codec {:?} is not WebM-compatible. Use Vorbis or Opus.",
                config.codec
            )));
        }
        self.inner.add_audio_track(config)
    }

    /// Write the header.
    pub fn write_header(&mut self) -> Result<()> {
        self.inner.write_header()
    }

    /// Write a packet.
    pub fn write_packet(&mut self, packet: &transcode_core::Packet) -> Result<()> {
        self.inner.write_packet(packet)
    }

    /// Finalize the file.
    pub fn finalize(&mut self) -> Result<()> {
        self.inner.finalize()
    }

    /// Get the inner muxer.
    pub fn inner(&self) -> &MkvMuxer<W> {
        &self.inner
    }

    /// Get the inner muxer mutably.
    pub fn inner_mut(&mut self) -> &mut MkvMuxer<W> {
        &mut self.inner
    }

    /// Consume and return the inner muxer.
    pub fn into_inner(self) -> MkvMuxer<W> {
        self.inner
    }
}

/// VP9 codec private data builder for WebM.
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

    /// Build the codec private data.
    pub fn build(&self) -> Vec<u8> {
        // VP9 codec private is optional in WebM
        // Format: Profile (1 byte) | Level (1 byte) | BitDepth (4 bits) | ChromaSubsampling (4 bits)
        vec![self.profile, self.level, (self.bit_depth << 4) | self.chroma_subsampling]
    }
}

/// Opus codec private data builder for WebM.
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

        // If channel_mapping_family > 0, additional channel mapping data would follow

        data
    }
}

/// Vorbis codec private data builder for WebM.
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

    /// Build the codec private data in Matroska format.
    pub fn build(&self) -> Vec<u8> {
        // Matroska Vorbis codec private format:
        // - Number of packets (1 byte, minus 1, so 2 for 3 headers)
        // - Size of first packet (Xiph lacing)
        // - Size of second packet (Xiph lacing)
        // - First packet (identification header)
        // - Second packet (comment header)
        // - Third packet (setup header)

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

/// AV1 codec private data builder for WebM.
#[derive(Debug, Clone, Default)]
pub struct Av1CodecPrivate {
    /// AV1 configuration record.
    config_record: Vec<u8>,
}

impl Av1CodecPrivate {
    /// Create from an AV1 configuration record (OBU sequence header).
    pub fn from_config_record(config: Vec<u8>) -> Self {
        Self {
            config_record: config,
        }
    }

    /// Build the codec private data.
    pub fn build(&self) -> Vec<u8> {
        // AV1CodecConfigurationRecord format:
        // marker (1) + version (7) = 1 byte
        // seq_profile (3) + seq_level_idx_0 (5) = 1 byte
        // seq_tier_0 (1) + high_bitdepth (1) + twelve_bit (1) + monochrome (1) + chroma_subsampling_x (1) + chroma_subsampling_y (1) + chroma_sample_position (2) = 1 byte
        // reserved (3) + initial_presentation_delay_present (1) + initial_presentation_delay_minus_one (4) = 1 byte
        // configOBUs

        // For now, just return the raw config record
        // A proper implementation would parse and build the AV1CodecConfigurationRecord
        self.config_record.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_webm_codec_id_check() {
        assert!(is_webm_codec_id(codec_ids::V_VP8));
        assert!(is_webm_codec_id(codec_ids::V_VP9));
        assert!(is_webm_codec_id(codec_ids::V_AV1));
        assert!(is_webm_codec_id(codec_ids::A_OPUS));
        assert!(is_webm_codec_id(codec_ids::A_VORBIS));
        assert!(!is_webm_codec_id(codec_ids::V_MPEG4_ISO_AVC));
        assert!(!is_webm_codec_id(codec_ids::A_AAC));
    }

    #[test]
    fn test_opus_codec_private() {
        let opus = OpusCodecPrivate::new(2, 48000).with_pre_skip(312);
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
    fn test_vp9_codec_private() {
        let vp9 = Vp9CodecPrivate::new()
            .with_profile(0)
            .with_level(31)
            .with_bit_depth(8);
        let data = vp9.build();

        assert_eq!(data[0], 0); // Profile
        assert_eq!(data[1], 31); // Level
        assert_eq!(data[2] >> 4, 8); // Bit depth
    }

    #[test]
    fn test_vorbis_codec_private() {
        let id_header = vec![0x01, 0x76, 0x6F, 0x72, 0x62, 0x69, 0x73]; // Minimal ID header
        let comment_header = vec![0x03, 0x76, 0x6F, 0x72, 0x62, 0x69, 0x73]; // Minimal comment
        let setup_header = vec![0x05, 0x76, 0x6F, 0x72, 0x62, 0x69, 0x73]; // Minimal setup

        let vorbis = VorbisCodecPrivate::from_headers(
            id_header.clone(),
            comment_header.clone(),
            setup_header.clone(),
        );
        let data = vorbis.build();

        // Should have: 1 byte count + lacing sizes + headers
        assert_eq!(data[0], 2); // Number of packets - 1
        assert!(data.len() > 3); // At least has the lacing info
    }

    #[test]
    fn test_webm_muxer_codec_validation() {
        use std::io::Cursor;

        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        // VP9 should work
        let vp9 = VideoTrackConfig::new(1, VideoCodec::Vp9, 1920, 1080);
        assert!(muxer.add_video_track(vp9).is_ok());

        // Create new muxer for H.264 test
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        // H.264 should fail
        let h264 = VideoTrackConfig::new(1, VideoCodec::H264, 1920, 1080);
        assert!(muxer.add_video_track(h264).is_err());
    }

    #[test]
    fn test_webm_muxer_audio_validation() {
        use std::io::Cursor;

        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        // Opus should work
        let opus = AudioTrackConfig::new(1, AudioCodec::Opus, 48000.0, 2);
        assert!(muxer.add_audio_track(opus).is_ok());

        // Create new muxer for AAC test
        let buffer = Cursor::new(Vec::new());
        let mut muxer = WebmMuxer::new(buffer);

        // AAC should fail
        let aac = AudioTrackConfig::new(1, AudioCodec::Aac, 48000.0, 2);
        assert!(muxer.add_audio_track(aac).is_err());
    }
}
