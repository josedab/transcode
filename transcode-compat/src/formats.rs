//! Format and codec name mapping.
//!
//! This module provides mappings between FFmpeg codec/format names
//! and the internal transcode types.

use crate::error::{CompatError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use transcode_core::format::{AudioCodec, ContainerFormat, VideoCodec};

/// Video codec mapping from FFmpeg names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VideoCodecName {
    /// H.264 / AVC
    H264,
    /// H.265 / HEVC
    H265,
    /// VP8
    Vp8,
    /// VP9
    Vp9,
    /// AV1
    Av1,
    /// MJPEG
    Mjpeg,
    /// Copy (stream copy, no transcoding)
    Copy,
    /// Raw video
    Raw,
}

impl VideoCodecName {
    /// Parse an FFmpeg video codec name.
    ///
    /// Supported names:
    /// - `libx264`, `h264`, `avc`, `x264`
    /// - `libx265`, `h265`, `hevc`, `x265`
    /// - `libvpx`, `vp8`
    /// - `libvpx-vp9`, `vp9`
    /// - `libaom-av1`, `av1`, `libsvtav1`, `librav1e`
    /// - `mjpeg`, `mjpg`
    /// - `copy`
    /// - `rawvideo`
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase();

        match name.as_str() {
            // H.264 variants
            "libx264" | "h264" | "avc" | "x264" | "h264_nvenc" | "h264_qsv" | "h264_vaapi"
            | "h264_videotoolbox" | "h264_amf" | "h264_v4l2m2m" => Ok(Self::H264),

            // H.265 variants
            "libx265" | "h265" | "hevc" | "x265" | "hevc_nvenc" | "hevc_qsv" | "hevc_vaapi"
            | "hevc_videotoolbox" | "hevc_amf" | "hevc_v4l2m2m" => Ok(Self::H265),

            // VP8
            "libvpx" | "vp8" => Ok(Self::Vp8),

            // VP9
            "libvpx-vp9" | "vp9" => Ok(Self::Vp9),

            // AV1 variants
            "libaom-av1" | "av1" | "libsvtav1" | "svtav1" | "librav1e" | "rav1e"
            | "av1_nvenc" | "av1_qsv" | "av1_vaapi" | "av1_amf" => Ok(Self::Av1),

            // MJPEG
            "mjpeg" | "mjpg" => Ok(Self::Mjpeg),

            // Copy
            "copy" => Ok(Self::Copy),

            // Raw
            "rawvideo" | "raw" => Ok(Self::Raw),

            _ => Err(CompatError::UnknownCodec(name)),
        }
    }

    /// Convert to the internal video codec type.
    pub fn to_video_codec(&self) -> Option<VideoCodec> {
        match self {
            Self::H264 => Some(VideoCodec::H264),
            Self::H265 => Some(VideoCodec::H265),
            Self::Vp8 => Some(VideoCodec::Vp8),
            Self::Vp9 => Some(VideoCodec::Vp9),
            Self::Av1 => Some(VideoCodec::Av1),
            Self::Mjpeg => Some(VideoCodec::Mjpeg),
            Self::Raw => Some(VideoCodec::Raw),
            Self::Copy => None, // Copy doesn't map to a codec
        }
    }

    /// Get the canonical FFmpeg encoder name.
    pub fn ffmpeg_encoder(&self) -> &'static str {
        match self {
            Self::H264 => "libx264",
            Self::H265 => "libx265",
            Self::Vp8 => "libvpx",
            Self::Vp9 => "libvpx-vp9",
            Self::Av1 => "libaom-av1",
            Self::Mjpeg => "mjpeg",
            Self::Raw => "rawvideo",
            Self::Copy => "copy",
        }
    }

    /// Check if this is a copy operation.
    pub fn is_copy(&self) -> bool {
        matches!(self, Self::Copy)
    }
}

impl std::fmt::Display for VideoCodecName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ffmpeg_encoder())
    }
}

impl From<VideoCodec> for VideoCodecName {
    fn from(codec: VideoCodec) -> Self {
        match codec {
            VideoCodec::H264 => Self::H264,
            VideoCodec::H265 => Self::H265,
            VideoCodec::Vp8 => Self::Vp8,
            VideoCodec::Vp9 => Self::Vp9,
            VideoCodec::Av1 => Self::Av1,
            VideoCodec::Mjpeg => Self::Mjpeg,
            VideoCodec::Raw => Self::Raw,
        }
    }
}

/// Audio codec mapping from FFmpeg names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudioCodecName {
    /// AAC
    Aac,
    /// MP3
    Mp3,
    /// Opus
    Opus,
    /// Vorbis
    Vorbis,
    /// FLAC
    Flac,
    /// PCM
    Pcm,
    /// AC-3
    Ac3,
    /// E-AC-3
    Eac3,
    /// Copy (stream copy)
    Copy,
}

impl AudioCodecName {
    /// Parse an FFmpeg audio codec name.
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase();

        match name.as_str() {
            // AAC variants
            "aac" | "libfdk_aac" | "aac_at" => Ok(Self::Aac),

            // MP3
            "libmp3lame" | "mp3" | "mp3float" => Ok(Self::Mp3),

            // Opus
            "libopus" | "opus" => Ok(Self::Opus),

            // Vorbis
            "libvorbis" | "vorbis" => Ok(Self::Vorbis),

            // FLAC
            "flac" => Ok(Self::Flac),

            // PCM variants
            "pcm_s16le" | "pcm_s16be" | "pcm_s24le" | "pcm_s24be" | "pcm_s32le" | "pcm_s32be"
            | "pcm_f32le" | "pcm_f32be" | "pcm_f64le" | "pcm_f64be" | "pcm" => Ok(Self::Pcm),

            // AC-3
            "ac3" | "ac3_fixed" => Ok(Self::Ac3),

            // E-AC-3
            "eac3" => Ok(Self::Eac3),

            // Copy
            "copy" => Ok(Self::Copy),

            _ => Err(CompatError::UnknownCodec(name)),
        }
    }

    /// Convert to the internal audio codec type.
    pub fn to_audio_codec(&self) -> Option<AudioCodec> {
        match self {
            Self::Aac => Some(AudioCodec::Aac),
            Self::Mp3 => Some(AudioCodec::Mp3),
            Self::Opus => Some(AudioCodec::Opus),
            Self::Vorbis => Some(AudioCodec::Vorbis),
            Self::Flac => Some(AudioCodec::Flac),
            Self::Pcm => Some(AudioCodec::Pcm),
            Self::Ac3 => Some(AudioCodec::Ac3),
            Self::Eac3 => Some(AudioCodec::Eac3),
            Self::Copy => None,
        }
    }

    /// Get the canonical FFmpeg encoder name.
    pub fn ffmpeg_encoder(&self) -> &'static str {
        match self {
            Self::Aac => "aac",
            Self::Mp3 => "libmp3lame",
            Self::Opus => "libopus",
            Self::Vorbis => "libvorbis",
            Self::Flac => "flac",
            Self::Pcm => "pcm_s16le",
            Self::Ac3 => "ac3",
            Self::Eac3 => "eac3",
            Self::Copy => "copy",
        }
    }

    /// Check if this is a copy operation.
    pub fn is_copy(&self) -> bool {
        matches!(self, Self::Copy)
    }
}

impl std::fmt::Display for AudioCodecName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ffmpeg_encoder())
    }
}

impl From<AudioCodec> for AudioCodecName {
    fn from(codec: AudioCodec) -> Self {
        match codec {
            AudioCodec::Aac => Self::Aac,
            AudioCodec::Mp3 => Self::Mp3,
            AudioCodec::Opus => Self::Opus,
            AudioCodec::Vorbis => Self::Vorbis,
            AudioCodec::Flac => Self::Flac,
            AudioCodec::Pcm => Self::Pcm,
            AudioCodec::Ac3 => Self::Ac3,
            AudioCodec::Eac3 => Self::Eac3,
        }
    }
}

/// Container format mapping from FFmpeg names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContainerName {
    /// MP4
    Mp4,
    /// Matroska
    Mkv,
    /// WebM
    WebM,
    /// MPEG-TS
    MpegTs,
    /// FLV
    Flv,
    /// QuickTime
    Mov,
    /// Raw bitstream
    Raw,
    /// HLS (segmented)
    Hls,
    /// DASH (segmented)
    Dash,
}

impl ContainerName {
    /// Parse an FFmpeg format name.
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase();

        match name.as_str() {
            "mp4" | "m4v" | "m4a" | "ipod" => Ok(Self::Mp4),
            "matroska" | "mkv" | "mka" | "mks" => Ok(Self::Mkv),
            "webm" => Ok(Self::WebM),
            "mpegts" | "ts" | "m2ts" | "mts" => Ok(Self::MpegTs),
            "flv" | "f4v" => Ok(Self::Flv),
            "mov" | "qt" => Ok(Self::Mov),
            "rawvideo" | "raw" | "h264" | "hevc" | "avc" => Ok(Self::Raw),
            "hls" | "segment" => Ok(Self::Hls),
            "dash" => Ok(Self::Dash),
            _ => Err(CompatError::UnknownFormat(name)),
        }
    }

    /// Convert to the internal container format type.
    pub fn to_container_format(&self) -> Option<ContainerFormat> {
        match self {
            Self::Mp4 => Some(ContainerFormat::Mp4),
            Self::Mkv => Some(ContainerFormat::Mkv),
            Self::WebM => Some(ContainerFormat::WebM),
            Self::MpegTs => Some(ContainerFormat::MpegTs),
            Self::Flv => Some(ContainerFormat::Flv),
            Self::Mov => Some(ContainerFormat::Mov),
            Self::Raw => Some(ContainerFormat::Raw),
            Self::Hls | Self::Dash => None, // Streaming formats are handled differently
        }
    }

    /// Get the canonical FFmpeg format name.
    pub fn ffmpeg_format(&self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "matroska",
            Self::WebM => "webm",
            Self::MpegTs => "mpegts",
            Self::Flv => "flv",
            Self::Mov => "mov",
            Self::Raw => "rawvideo",
            Self::Hls => "hls",
            Self::Dash => "dash",
        }
    }

    /// Get the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
            Self::WebM => "webm",
            Self::MpegTs => "ts",
            Self::Flv => "flv",
            Self::Mov => "mov",
            Self::Raw => "raw",
            Self::Hls => "m3u8",
            Self::Dash => "mpd",
        }
    }

    /// Guess format from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        let ext = ext.trim().to_lowercase();
        let ext = ext.trim_start_matches('.');

        match ext {
            "mp4" | "m4v" | "m4a" => Some(Self::Mp4),
            "mkv" | "mka" | "mks" => Some(Self::Mkv),
            "webm" => Some(Self::WebM),
            "ts" | "m2ts" | "mts" => Some(Self::MpegTs),
            "flv" | "f4v" => Some(Self::Flv),
            "mov" | "qt" => Some(Self::Mov),
            "m3u8" => Some(Self::Hls),
            "mpd" => Some(Self::Dash),
            "h264" | "264" | "avc" => Some(Self::Raw),
            "h265" | "265" | "hevc" => Some(Self::Raw),
            _ => None,
        }
    }
}

impl std::fmt::Display for ContainerName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ffmpeg_format())
    }
}

impl From<ContainerFormat> for ContainerName {
    fn from(format: ContainerFormat) -> Self {
        match format {
            ContainerFormat::Mp4 => Self::Mp4,
            ContainerFormat::Mkv => Self::Mkv,
            ContainerFormat::WebM => Self::WebM,
            ContainerFormat::MpegTs => Self::MpegTs,
            ContainerFormat::Flv => Self::Flv,
            ContainerFormat::Mov => Self::Mov,
            ContainerFormat::Raw => Self::Raw,
        }
    }
}

/// Pixel format mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PixelFormatName {
    /// YUV 4:2:0 planar 8-bit
    Yuv420p,
    /// YUV 4:2:0 planar 10-bit LE
    Yuv420p10le,
    /// YUV 4:2:2 planar 8-bit
    Yuv422p,
    /// YUV 4:2:2 planar 10-bit LE
    Yuv422p10le,
    /// YUV 4:4:4 planar 8-bit
    Yuv444p,
    /// YUV 4:4:4 planar 10-bit LE
    Yuv444p10le,
    /// NV12 (YUV 4:2:0 semi-planar)
    Nv12,
    /// NV21 (YUV 4:2:0 semi-planar, reversed chroma)
    Nv21,
    /// RGB 24-bit
    Rgb24,
    /// BGR 24-bit
    Bgr24,
    /// RGBA 32-bit
    Rgba,
    /// BGRA 32-bit
    Bgra,
    /// Gray 8-bit
    Gray,
    /// Gray 16-bit
    Gray16,
}

impl PixelFormatName {
    /// Parse an FFmpeg pixel format name.
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase();

        match name.as_str() {
            "yuv420p" => Ok(Self::Yuv420p),
            "yuv420p10le" | "yuv420p10" => Ok(Self::Yuv420p10le),
            "yuv422p" => Ok(Self::Yuv422p),
            "yuv422p10le" | "yuv422p10" => Ok(Self::Yuv422p10le),
            "yuv444p" => Ok(Self::Yuv444p),
            "yuv444p10le" | "yuv444p10" => Ok(Self::Yuv444p10le),
            "nv12" => Ok(Self::Nv12),
            "nv21" => Ok(Self::Nv21),
            "rgb24" => Ok(Self::Rgb24),
            "bgr24" => Ok(Self::Bgr24),
            "rgba" | "rgb32" => Ok(Self::Rgba),
            "bgra" | "bgr32" => Ok(Self::Bgra),
            "gray" | "gray8" | "y8" => Ok(Self::Gray),
            "gray16" | "gray16le" | "y16" => Ok(Self::Gray16),
            _ => Err(CompatError::InvalidValue {
                option: "pix_fmt".to_string(),
                value: name,
                reason: "unknown pixel format".to_string(),
            }),
        }
    }

    /// Get the FFmpeg name.
    pub fn ffmpeg_name(&self) -> &'static str {
        match self {
            Self::Yuv420p => "yuv420p",
            Self::Yuv420p10le => "yuv420p10le",
            Self::Yuv422p => "yuv422p",
            Self::Yuv422p10le => "yuv422p10le",
            Self::Yuv444p => "yuv444p",
            Self::Yuv444p10le => "yuv444p10le",
            Self::Nv12 => "nv12",
            Self::Nv21 => "nv21",
            Self::Rgb24 => "rgb24",
            Self::Bgr24 => "bgr24",
            Self::Rgba => "rgba",
            Self::Bgra => "bgra",
            Self::Gray => "gray",
            Self::Gray16 => "gray16le",
        }
    }

    /// Get the bits per component.
    pub fn bits_per_component(&self) -> u8 {
        match self {
            Self::Yuv420p10le | Self::Yuv422p10le | Self::Yuv444p10le | Self::Gray16 => 10,
            _ => 8,
        }
    }

    /// Check if this is a 10-bit format.
    pub fn is_10bit(&self) -> bool {
        self.bits_per_component() == 10
    }
}

impl std::fmt::Display for PixelFormatName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ffmpeg_name())
    }
}

/// Sample format mapping for audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SampleFormatName {
    /// Unsigned 8-bit
    U8,
    /// Signed 16-bit
    S16,
    /// Signed 32-bit
    S32,
    /// Signed 64-bit
    S64,
    /// Float 32-bit
    Flt,
    /// Float 64-bit (double)
    Dbl,
    /// Unsigned 8-bit planar
    U8p,
    /// Signed 16-bit planar
    S16p,
    /// Signed 32-bit planar
    S32p,
    /// Float 32-bit planar
    Fltp,
    /// Float 64-bit planar
    Dblp,
}

impl SampleFormatName {
    /// Parse an FFmpeg sample format name.
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase();

        match name.as_str() {
            "u8" => Ok(Self::U8),
            "s16" => Ok(Self::S16),
            "s32" => Ok(Self::S32),
            "s64" => Ok(Self::S64),
            "flt" => Ok(Self::Flt),
            "dbl" => Ok(Self::Dbl),
            "u8p" => Ok(Self::U8p),
            "s16p" => Ok(Self::S16p),
            "s32p" => Ok(Self::S32p),
            "fltp" => Ok(Self::Fltp),
            "dblp" => Ok(Self::Dblp),
            _ => Err(CompatError::InvalidValue {
                option: "sample_fmt".to_string(),
                value: name,
                reason: "unknown sample format".to_string(),
            }),
        }
    }

    /// Get the FFmpeg name.
    pub fn ffmpeg_name(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::S16 => "s16",
            Self::S32 => "s32",
            Self::S64 => "s64",
            Self::Flt => "flt",
            Self::Dbl => "dbl",
            Self::U8p => "u8p",
            Self::S16p => "s16p",
            Self::S32p => "s32p",
            Self::Fltp => "fltp",
            Self::Dblp => "dblp",
        }
    }

    /// Check if this is a planar format.
    pub fn is_planar(&self) -> bool {
        matches!(
            self,
            Self::U8p | Self::S16p | Self::S32p | Self::Fltp | Self::Dblp
        )
    }
}

impl std::fmt::Display for SampleFormatName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ffmpeg_name())
    }
}

/// Codec name registry for looking up codecs by various names.
#[derive(Debug, Clone, Default)]
pub struct CodecRegistry {
    video_aliases: HashMap<String, VideoCodecName>,
    audio_aliases: HashMap<String, AudioCodecName>,
}

impl CodecRegistry {
    /// Create a new registry with default mappings.
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_defaults();
        registry
    }

    /// Register default codec mappings.
    fn register_defaults(&mut self) {
        // Additional video codec aliases
        self.video_aliases
            .insert("264".to_string(), VideoCodecName::H264);
        self.video_aliases
            .insert("265".to_string(), VideoCodecName::H265);

        // Additional audio codec aliases
        self.audio_aliases
            .insert("lame".to_string(), AudioCodecName::Mp3);
        self.audio_aliases
            .insert("faac".to_string(), AudioCodecName::Aac);
        self.audio_aliases
            .insert("fdk".to_string(), AudioCodecName::Aac);
    }

    /// Register a video codec alias.
    pub fn register_video_alias(&mut self, alias: impl Into<String>, codec: VideoCodecName) {
        self.video_aliases.insert(alias.into(), codec);
    }

    /// Register an audio codec alias.
    pub fn register_audio_alias(&mut self, alias: impl Into<String>, codec: AudioCodecName) {
        self.audio_aliases.insert(alias.into(), codec);
    }

    /// Look up a video codec by name.
    pub fn lookup_video(&self, name: &str) -> Result<VideoCodecName> {
        let name = name.trim().to_lowercase();

        // Try aliases first
        if let Some(codec) = self.video_aliases.get(&name) {
            return Ok(*codec);
        }

        // Fall back to standard parsing
        VideoCodecName::parse(&name)
    }

    /// Look up an audio codec by name.
    pub fn lookup_audio(&self, name: &str) -> Result<AudioCodecName> {
        let name = name.trim().to_lowercase();

        // Try aliases first
        if let Some(codec) = self.audio_aliases.get(&name) {
            return Ok(*codec);
        }

        // Fall back to standard parsing
        AudioCodecName::parse(&name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_codec_parse() {
        assert_eq!(
            VideoCodecName::parse("libx264").unwrap(),
            VideoCodecName::H264
        );
        assert_eq!(VideoCodecName::parse("h264").unwrap(), VideoCodecName::H264);
        assert_eq!(
            VideoCodecName::parse("libx265").unwrap(),
            VideoCodecName::H265
        );
        assert_eq!(
            VideoCodecName::parse("libaom-av1").unwrap(),
            VideoCodecName::Av1
        );
        assert_eq!(VideoCodecName::parse("copy").unwrap(), VideoCodecName::Copy);
    }

    #[test]
    fn test_video_codec_to_internal() {
        assert_eq!(
            VideoCodecName::H264.to_video_codec(),
            Some(VideoCodec::H264)
        );
        assert_eq!(VideoCodecName::Copy.to_video_codec(), None);
    }

    #[test]
    fn test_audio_codec_parse() {
        assert_eq!(AudioCodecName::parse("aac").unwrap(), AudioCodecName::Aac);
        assert_eq!(
            AudioCodecName::parse("libmp3lame").unwrap(),
            AudioCodecName::Mp3
        );
        assert_eq!(
            AudioCodecName::parse("libopus").unwrap(),
            AudioCodecName::Opus
        );
        assert_eq!(AudioCodecName::parse("copy").unwrap(), AudioCodecName::Copy);
    }

    #[test]
    fn test_container_parse() {
        assert_eq!(ContainerName::parse("mp4").unwrap(), ContainerName::Mp4);
        assert_eq!(
            ContainerName::parse("matroska").unwrap(),
            ContainerName::Mkv
        );
        assert_eq!(ContainerName::parse("webm").unwrap(), ContainerName::WebM);
        assert_eq!(ContainerName::parse("hls").unwrap(), ContainerName::Hls);
    }

    #[test]
    fn test_container_from_extension() {
        assert_eq!(ContainerName::from_extension("mp4"), Some(ContainerName::Mp4));
        assert_eq!(ContainerName::from_extension("mkv"), Some(ContainerName::Mkv));
        assert_eq!(
            ContainerName::from_extension(".webm"),
            Some(ContainerName::WebM)
        );
        assert_eq!(ContainerName::from_extension("xyz"), None);
    }

    #[test]
    fn test_pixel_format_parse() {
        assert_eq!(
            PixelFormatName::parse("yuv420p").unwrap(),
            PixelFormatName::Yuv420p
        );
        assert_eq!(
            PixelFormatName::parse("yuv420p10le").unwrap(),
            PixelFormatName::Yuv420p10le
        );
        assert!(PixelFormatName::Yuv420p10le.is_10bit());
        assert!(!PixelFormatName::Yuv420p.is_10bit());
    }

    #[test]
    fn test_sample_format_parse() {
        assert_eq!(
            SampleFormatName::parse("s16").unwrap(),
            SampleFormatName::S16
        );
        assert_eq!(
            SampleFormatName::parse("fltp").unwrap(),
            SampleFormatName::Fltp
        );
        assert!(SampleFormatName::Fltp.is_planar());
        assert!(!SampleFormatName::S16.is_planar());
    }

    #[test]
    fn test_codec_registry() {
        let registry = CodecRegistry::new();

        assert_eq!(registry.lookup_video("264").unwrap(), VideoCodecName::H264);
        assert_eq!(
            registry.lookup_video("libx264").unwrap(),
            VideoCodecName::H264
        );
        assert_eq!(registry.lookup_audio("lame").unwrap(), AudioCodecName::Mp3);
    }

    #[test]
    fn test_hardware_codec_names() {
        // Verify hardware-accelerated codec names are recognized
        assert_eq!(
            VideoCodecName::parse("h264_nvenc").unwrap(),
            VideoCodecName::H264
        );
        assert_eq!(
            VideoCodecName::parse("hevc_qsv").unwrap(),
            VideoCodecName::H265
        );
        assert_eq!(
            VideoCodecName::parse("av1_vaapi").unwrap(),
            VideoCodecName::Av1
        );
    }
}
