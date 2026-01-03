//! FLV muxer implementation.
//!
//! This module provides a muxer for creating FLV files from audio and video packets.
//!
//! ## Example
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::BufWriter;
//! use transcode_flv::{FlvMuxer, MuxerConfig, VideoStreamConfig, AudioStreamConfig};
//! use transcode_core::packet::Packet;
//!
//! let file = File::create("output.flv").unwrap();
//! let writer = BufWriter::new(file);
//!
//! let config = MuxerConfig::default();
//! let mut muxer = FlvMuxer::new(writer, config);
//!
//! // Configure streams
//! let video_config = VideoStreamConfig::avc(1920, 1080, 30.0);
//! muxer.set_video_config(video_config);
//!
//! // Write header and metadata
//! muxer.write_header().unwrap();
//!
//! // Write packets...
//! // muxer.write_video_packet(&packet).unwrap();
//!
//! // Finalize
//! muxer.finalize().unwrap();
//! ```

use crate::amf::MetadataBuilder;
use crate::audio::{AacConfig, AudioTagHeader, SoundFormat};
use crate::error::{FlvError, Result};
use crate::header::FlvHeader;
use crate::tag::{write_previous_tag_size, FlvTag, TagHeader, TagType, TAG_HEADER_SIZE};
use crate::video::{AvcConfig, FrameType, VideoCodec, VideoTagHeader};

use byteorder::{BigEndian, WriteBytesExt};
use std::io::Write;

/// Video stream configuration.
#[derive(Debug, Clone)]
pub struct VideoStreamConfig {
    /// Video codec.
    pub codec: VideoCodec,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub frame_rate: f64,
    /// Video bitrate in kbps.
    pub bitrate: Option<f64>,
    /// AVC decoder configuration (SPS/PPS).
    pub avc_config: Option<AvcConfig>,
}

impl VideoStreamConfig {
    /// Create an AVC/H.264 video configuration.
    pub fn avc(width: u32, height: u32, frame_rate: f64) -> Self {
        Self {
            codec: VideoCodec::Avc,
            width,
            height,
            frame_rate,
            bitrate: None,
            avc_config: None,
        }
    }

    /// Create an HEVC/H.265 video configuration.
    pub fn hevc(width: u32, height: u32, frame_rate: f64) -> Self {
        Self {
            codec: VideoCodec::Hevc,
            width,
            height,
            frame_rate,
            bitrate: None,
            avc_config: None,
        }
    }

    /// Set the video bitrate.
    pub fn with_bitrate(mut self, kbps: f64) -> Self {
        self.bitrate = Some(kbps);
        self
    }

    /// Set the AVC decoder configuration.
    pub fn with_avc_config(mut self, config: AvcConfig) -> Self {
        self.avc_config = Some(config);
        self
    }
}

/// Audio stream configuration.
#[derive(Debug, Clone)]
pub struct AudioStreamConfig {
    /// Audio codec/format.
    pub format: SoundFormat,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bits per sample.
    pub bits_per_sample: u8,
    /// Audio bitrate in kbps.
    pub bitrate: Option<f64>,
    /// AAC decoder configuration.
    pub aac_config: Option<AacConfig>,
}

impl AudioStreamConfig {
    /// Create an AAC audio configuration.
    pub fn aac(sample_rate: u32, channels: u8) -> Self {
        Self {
            format: SoundFormat::Aac,
            sample_rate,
            channels,
            bits_per_sample: 16,
            bitrate: None,
            aac_config: None,
        }
    }

    /// Create an MP3 audio configuration.
    pub fn mp3(sample_rate: u32, channels: u8) -> Self {
        Self {
            format: SoundFormat::Mp3,
            sample_rate,
            channels,
            bits_per_sample: 16,
            bitrate: None,
            aac_config: None,
        }
    }

    /// Set the audio bitrate.
    pub fn with_bitrate(mut self, kbps: f64) -> Self {
        self.bitrate = Some(kbps);
        self
    }

    /// Set the AAC decoder configuration.
    pub fn with_aac_config(mut self, config: AacConfig) -> Self {
        self.aac_config = Some(config);
        self
    }
}

/// FLV muxer configuration.
#[derive(Debug, Clone)]
pub struct MuxerConfig {
    /// Write metadata tag at the start.
    pub write_metadata: bool,
    /// Encoder name for metadata.
    pub encoder_name: String,
}

impl Default for MuxerConfig {
    fn default() -> Self {
        Self {
            write_metadata: true,
            encoder_name: "transcode-flv".to_string(),
        }
    }
}

impl MuxerConfig {
    /// Create a new muxer configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to write metadata.
    pub fn with_metadata(mut self, write: bool) -> Self {
        self.write_metadata = write;
        self
    }

    /// Set the encoder name.
    pub fn with_encoder_name(mut self, name: impl Into<String>) -> Self {
        self.encoder_name = name.into();
        self
    }
}

/// FLV muxer.
pub struct FlvMuxer<W: Write> {
    /// Output writer.
    writer: W,
    /// Muxer configuration.
    config: MuxerConfig,
    /// Video stream configuration.
    video_config: Option<VideoStreamConfig>,
    /// Audio stream configuration.
    audio_config: Option<AudioStreamConfig>,
    /// Whether the header has been written.
    header_written: bool,
    /// Whether the video sequence header has been written.
    video_sequence_header_written: bool,
    /// Whether the audio sequence header has been written.
    audio_sequence_header_written: bool,
    /// Last tag size written.
    last_tag_size: u32,
    /// Bytes written.
    bytes_written: u64,
    /// Last video timestamp.
    last_video_timestamp: u32,
    /// Last audio timestamp.
    last_audio_timestamp: u32,
    /// Duration tracking.
    duration_ms: u32,
}

impl<W: Write> FlvMuxer<W> {
    /// Create a new FLV muxer.
    pub fn new(writer: W, config: MuxerConfig) -> Self {
        Self {
            writer,
            config,
            video_config: None,
            audio_config: None,
            header_written: false,
            video_sequence_header_written: false,
            audio_sequence_header_written: false,
            last_tag_size: 0,
            bytes_written: 0,
            last_video_timestamp: 0,
            last_audio_timestamp: 0,
            duration_ms: 0,
        }
    }

    /// Set the video stream configuration.
    pub fn set_video_config(&mut self, config: VideoStreamConfig) {
        self.video_config = Some(config);
    }

    /// Set the audio stream configuration.
    pub fn set_audio_config(&mut self, config: AudioStreamConfig) {
        self.audio_config = Some(config);
    }

    /// Check if this muxer has video.
    pub fn has_video(&self) -> bool {
        self.video_config.is_some()
    }

    /// Check if this muxer has audio.
    pub fn has_audio(&self) -> bool {
        self.audio_config.is_some()
    }

    /// Write the FLV header.
    pub fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Ok(());
        }

        // Write FLV header
        let header = FlvHeader::new()
            .with_video(self.has_video())
            .with_audio(self.has_audio());

        header.write(&mut self.writer)?;
        self.bytes_written += 9;

        // Write initial previous tag size (0 after header)
        self.writer.write_u32::<BigEndian>(0)?;
        self.bytes_written += 4;

        self.header_written = true;

        // Write metadata if configured
        if self.config.write_metadata {
            self.write_metadata()?;
        }

        Ok(())
    }

    /// Write metadata tag.
    fn write_metadata(&mut self) -> Result<()> {
        let mut builder = MetadataBuilder::new().encoder(&self.config.encoder_name);

        if let Some(ref video) = self.video_config {
            builder = builder
                .width(video.width)
                .height(video.height)
                .frame_rate(video.frame_rate)
                .video_codec_id(video.codec.as_u8());

            if let Some(bitrate) = video.bitrate {
                builder = builder.video_data_rate(bitrate);
            }
        }

        if let Some(ref audio) = self.audio_config {
            builder = builder
                .audio_sample_rate(audio.sample_rate)
                .audio_sample_size(audio.bits_per_sample)
                .stereo(audio.channels > 1)
                .audio_codec_id(audio.format.as_u8());

            if let Some(bitrate) = audio.bitrate {
                builder = builder.audio_data_rate(bitrate);
            }
        }

        let script_data = builder.build_script_data();
        self.write_tag(TagType::ScriptData, 0, &script_data)?;

        Ok(())
    }

    /// Write a raw FLV tag.
    fn write_tag(&mut self, tag_type: TagType, timestamp_ms: u32, data: &[u8]) -> Result<()> {
        let header = TagHeader::new(tag_type, data.len() as u32, timestamp_ms);
        header.write(&mut self.writer)?;
        self.writer.write_all(data)?;

        let tag_size = TAG_HEADER_SIZE as u32 + data.len() as u32;
        write_previous_tag_size(&mut self.writer, tag_size)?;

        self.last_tag_size = tag_size;
        self.bytes_written += tag_size as u64 + 4;

        Ok(())
    }

    /// Write the video sequence header (AVC/HEVC decoder configuration).
    pub fn write_video_sequence_header(&mut self, config_data: &[u8]) -> Result<()> {
        if !self.header_written {
            self.write_header()?;
        }

        let video_config = self
            .video_config
            .as_ref()
            .ok_or(FlvError::NoStreamsFound)?;

        // Build video tag data
        let mut tag_data = Vec::with_capacity(config_data.len() + 5);

        let header = if video_config.codec == VideoCodec::Hevc {
            VideoTagHeader::hevc_sequence_header()
        } else {
            VideoTagHeader::avc_sequence_header()
        };

        header.write(&mut tag_data)?;
        tag_data.extend_from_slice(config_data);

        self.write_tag(TagType::Video, 0, &tag_data)?;
        self.video_sequence_header_written = true;

        Ok(())
    }

    /// Write the audio sequence header (AAC AudioSpecificConfig).
    pub fn write_audio_sequence_header(&mut self, config_data: &[u8]) -> Result<()> {
        if !self.header_written {
            self.write_header()?;
        }

        let audio_config = self
            .audio_config
            .as_ref()
            .ok_or(FlvError::NoStreamsFound)?;

        if audio_config.format != SoundFormat::Aac {
            return Ok(()); // Only AAC has sequence headers
        }

        // Build audio tag data
        let mut tag_data = Vec::with_capacity(config_data.len() + 2);

        let header = AudioTagHeader::aac_sequence_header();
        header.write(&mut tag_data)?;
        tag_data.extend_from_slice(config_data);

        self.write_tag(TagType::Audio, 0, &tag_data)?;
        self.audio_sequence_header_written = true;

        Ok(())
    }

    /// Write a video packet.
    ///
    /// # Arguments
    ///
    /// * `timestamp_ms` - Timestamp in milliseconds
    /// * `data` - Video NAL unit data (without start codes, with length prefixes)
    /// * `is_keyframe` - Whether this is a keyframe
    /// * `composition_time_ms` - PTS - DTS offset in milliseconds
    pub fn write_video_packet(
        &mut self,
        timestamp_ms: u32,
        data: &[u8],
        is_keyframe: bool,
        composition_time_ms: i32,
    ) -> Result<()> {
        if !self.header_written {
            self.write_header()?;
        }

        if !self.video_sequence_header_written {
            return Err(FlvError::MissingSequenceHeader {
                codec: "Video".to_string(),
            });
        }

        let video_config = self
            .video_config
            .as_ref()
            .ok_or(FlvError::NoStreamsFound)?;

        let frame_type = if is_keyframe {
            FrameType::Keyframe
        } else {
            FrameType::Inter
        };

        // Build video tag data
        let mut tag_data = Vec::with_capacity(data.len() + 5);

        let header = if video_config.codec == VideoCodec::Hevc {
            VideoTagHeader::hevc_coded_frames(frame_type, composition_time_ms)
        } else {
            VideoTagHeader::avc_nalu(frame_type, composition_time_ms)
        };

        header.write(&mut tag_data)?;
        tag_data.extend_from_slice(data);

        self.write_tag(TagType::Video, timestamp_ms, &tag_data)?;

        self.last_video_timestamp = timestamp_ms;
        if timestamp_ms > self.duration_ms {
            self.duration_ms = timestamp_ms;
        }

        Ok(())
    }

    /// Write an audio packet.
    ///
    /// # Arguments
    ///
    /// * `timestamp_ms` - Timestamp in milliseconds
    /// * `data` - Audio frame data
    pub fn write_audio_packet(&mut self, timestamp_ms: u32, data: &[u8]) -> Result<()> {
        if !self.header_written {
            self.write_header()?;
        }

        let audio_config = self
            .audio_config
            .as_ref()
            .ok_or(FlvError::NoStreamsFound)?;

        // Check if sequence header is needed for AAC
        if audio_config.format == SoundFormat::Aac && !self.audio_sequence_header_written {
            return Err(FlvError::MissingSequenceHeader {
                codec: "AAC".to_string(),
            });
        }

        // Build audio tag data
        let mut tag_data = Vec::with_capacity(data.len() + 2);

        let header = if audio_config.format == SoundFormat::Aac {
            AudioTagHeader::aac_raw()
        } else {
            AudioTagHeader::mp3(audio_config.sample_rate, audio_config.channels > 1)
        };

        header.write(&mut tag_data)?;
        tag_data.extend_from_slice(data);

        self.write_tag(TagType::Audio, timestamp_ms, &tag_data)?;

        self.last_audio_timestamp = timestamp_ms;
        if timestamp_ms > self.duration_ms {
            self.duration_ms = timestamp_ms;
        }

        Ok(())
    }

    /// Finalize the FLV file.
    pub fn finalize(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// Get the number of bytes written.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Get the current duration in milliseconds.
    pub fn duration_ms(&self) -> u32 {
        self.duration_ms
    }

    /// Get the underlying writer.
    pub fn into_inner(self) -> W {
        self.writer
    }

    /// Get a reference to the underlying writer.
    pub fn get_ref(&self) -> &W {
        &self.writer
    }

    /// Get a mutable reference to the underlying writer.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.writer
    }
}

/// Simple FLV tag writer for raw tag data.
pub struct TagWriter<W: Write> {
    writer: W,
    last_tag_size: u32,
}

impl<W: Write> TagWriter<W> {
    /// Create a new tag writer.
    pub fn new(mut writer: W, has_audio: bool, has_video: bool) -> Result<Self> {
        // Write header
        let header = FlvHeader::new()
            .with_audio(has_audio)
            .with_video(has_video);
        header.write(&mut writer)?;

        // Write initial previous tag size
        writer.write_u32::<BigEndian>(0)?;

        Ok(Self {
            writer,
            last_tag_size: 0,
        })
    }

    /// Write a complete FLV tag.
    pub fn write_tag(&mut self, tag: &FlvTag) -> Result<()> {
        tag.write(&mut self.writer)?;
        tag.write_previous_tag_size(&mut self.writer)?;
        self.last_tag_size = TAG_HEADER_SIZE as u32 + tag.data.len() as u32;
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

    #[test]
    fn test_muxer_creation() {
        let buffer = Vec::new();
        let config = MuxerConfig::default();
        let muxer = FlvMuxer::new(buffer, config);

        assert!(!muxer.has_video());
        assert!(!muxer.has_audio());
    }

    #[test]
    fn test_muxer_with_video() {
        let buffer = Vec::new();
        let config = MuxerConfig::default();
        let mut muxer = FlvMuxer::new(buffer, config);

        muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));

        assert!(muxer.has_video());
        assert!(!muxer.has_audio());
    }

    #[test]
    fn test_muxer_write_header() {
        let mut buffer = Vec::new();
        let config = MuxerConfig::new().with_metadata(false);
        let mut muxer = FlvMuxer::new(&mut buffer, config);

        muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));
        muxer.write_header().unwrap();

        // Check FLV signature
        assert_eq!(&buffer[0..3], b"FLV");
        // Check version
        assert_eq!(buffer[3], 1);
        // Check flags (video only = 0x01)
        assert_eq!(buffer[4], 0x01);
        // Check header size
        assert_eq!(&buffer[5..9], &[0, 0, 0, 9]);
        // Check initial previous tag size
        assert_eq!(&buffer[9..13], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_muxer_with_metadata() {
        let mut buffer = Vec::new();
        let config = MuxerConfig::default();
        let mut muxer = FlvMuxer::new(&mut buffer, config);

        muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));
        muxer.write_header().unwrap();

        // Should have written header (13 bytes) + metadata tag
        assert!(buffer.len() > 13);

        // First tag should be script data (type 18)
        assert_eq!(buffer[13], 18);
    }

    #[test]
    fn test_video_stream_config() {
        let config = VideoStreamConfig::avc(1920, 1080, 30.0).with_bitrate(5000.0);

        assert_eq!(config.codec, VideoCodec::Avc);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.frame_rate, 30.0);
        assert_eq!(config.bitrate, Some(5000.0));
    }

    #[test]
    fn test_audio_stream_config() {
        let config = AudioStreamConfig::aac(48000, 2).with_bitrate(128.0);

        assert_eq!(config.format, SoundFormat::Aac);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bitrate, Some(128.0));
    }

    #[test]
    fn test_muxer_config() {
        let config = MuxerConfig::new()
            .with_metadata(false)
            .with_encoder_name("test-encoder");

        assert!(!config.write_metadata);
        assert_eq!(config.encoder_name, "test-encoder");
    }

    #[test]
    fn test_tag_writer() {
        let mut buffer = Vec::new();
        let mut writer = TagWriter::new(&mut buffer, true, true).unwrap();

        let tag = FlvTag::script_data(0, vec![0xAB; 10]);
        writer.write_tag(&tag).unwrap();

        // Check output structure
        // Header: 9 bytes
        // Initial prev tag size: 4 bytes
        // Tag header: 11 bytes
        // Tag data: 10 bytes
        // Prev tag size: 4 bytes
        assert_eq!(buffer.len(), 9 + 4 + 11 + 10 + 4);
    }

    #[test]
    fn test_write_audio_video_headers() {
        let mut buffer = Vec::new();
        let config = MuxerConfig::new().with_metadata(false);
        let mut muxer = FlvMuxer::new(&mut buffer, config);

        muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));
        muxer.set_audio_config(AudioStreamConfig::aac(48000, 2));

        muxer.write_header().unwrap();

        // Verify header has both audio and video flags
        assert_eq!(buffer[4], 0x05); // Audio (0x04) + Video (0x01)
    }

    #[test]
    fn test_sequence_header_required() {
        let mut buffer = Vec::new();
        let config = MuxerConfig::new().with_metadata(false);
        let mut muxer = FlvMuxer::new(&mut buffer, config);

        muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));
        muxer.write_header().unwrap();

        // Should fail without sequence header
        let result = muxer.write_video_packet(0, &[0xAB; 100], true, 0);
        assert!(matches!(result, Err(FlvError::MissingSequenceHeader { .. })));
    }

    #[test]
    fn test_full_video_write() {
        let mut buffer = Vec::new();
        let duration: u32;
        {
            let config = MuxerConfig::new().with_metadata(false);
            let mut muxer = FlvMuxer::new(&mut buffer, config);

            muxer.set_video_config(VideoStreamConfig::avc(1920, 1080, 30.0));
            muxer.write_header().unwrap();

            // Write sequence header
            let avc_config = vec![
                0x01, 0x64, 0x00, 0x1F, 0xFF, 0xE1, 0x00, 0x04, 0x67, 0x64, 0x00, 0x1F, 0x01, 0x00,
                0x02, 0x68, 0xEF,
            ];
            muxer.write_video_sequence_header(&avc_config).unwrap();

            // Write video frame
            muxer
                .write_video_packet(0, &[0xAB; 100], true, 0)
                .unwrap();
            muxer
                .write_video_packet(33, &[0xCD; 50], false, 0)
                .unwrap();

            duration = muxer.duration_ms();
            muxer.finalize().unwrap();
        }

        // Verify we wrote some data
        assert!(buffer.len() > 100);
        assert_eq!(duration, 33);
    }
}
