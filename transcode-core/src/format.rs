//! Container and codec format definitions.

use std::fmt;

/// Container format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ContainerFormat {
    /// ISO Base Media File Format (MP4, M4V, M4A).
    Mp4,
    /// Matroska container.
    Mkv,
    /// WebM (Matroska subset for web).
    WebM,
    /// MPEG Transport Stream.
    MpegTs,
    /// Flash Video (legacy).
    Flv,
    /// QuickTime Movie.
    Mov,
    /// Raw bitstream (no container).
    Raw,
}

impl ContainerFormat {
    /// Get the typical file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Mkv => "mkv",
            Self::WebM => "webm",
            Self::MpegTs => "ts",
            Self::Flv => "flv",
            Self::Mov => "mov",
            Self::Raw => "raw",
        }
    }

    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Mp4 => "video/mp4",
            Self::Mkv => "video/x-matroska",
            Self::WebM => "video/webm",
            Self::MpegTs => "video/mp2t",
            Self::Flv => "video/x-flv",
            Self::Mov => "video/quicktime",
            Self::Raw => "application/octet-stream",
        }
    }

    /// Try to detect format from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "mp4" | "m4v" | "m4a" => Some(Self::Mp4),
            "mkv" => Some(Self::Mkv),
            "webm" => Some(Self::WebM),
            "ts" | "mts" | "m2ts" => Some(Self::MpegTs),
            "flv" => Some(Self::Flv),
            "mov" => Some(Self::Mov),
            _ => None,
        }
    }
}

impl fmt::Display for ContainerFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mp4 => write!(f, "MP4"),
            Self::Mkv => write!(f, "Matroska"),
            Self::WebM => write!(f, "WebM"),
            Self::MpegTs => write!(f, "MPEG-TS"),
            Self::Flv => write!(f, "FLV"),
            Self::Mov => write!(f, "QuickTime"),
            Self::Raw => write!(f, "Raw"),
        }
    }
}

/// Video codec type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VideoCodec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    H265,
    /// VP8.
    Vp8,
    /// VP9.
    Vp9,
    /// AV1.
    Av1,
    /// MJPEG.
    Mjpeg,
    /// Raw video (uncompressed).
    Raw,
}

impl VideoCodec {
    /// Get the FourCC code for this codec.
    pub fn fourcc(&self) -> [u8; 4] {
        match self {
            Self::H264 => *b"avc1",
            Self::H265 => *b"hvc1",
            Self::Vp8 => *b"vp08",
            Self::Vp9 => *b"vp09",
            Self::Av1 => *b"av01",
            Self::Mjpeg => *b"mjpg",
            Self::Raw => *b"raw ",
        }
    }

    /// Check if this codec is royalty-free.
    pub fn is_royalty_free(&self) -> bool {
        matches!(self, Self::Vp8 | Self::Vp9 | Self::Av1)
    }
}

impl fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H264 => write!(f, "H.264/AVC"),
            Self::H265 => write!(f, "H.265/HEVC"),
            Self::Vp8 => write!(f, "VP8"),
            Self::Vp9 => write!(f, "VP9"),
            Self::Av1 => write!(f, "AV1"),
            Self::Mjpeg => write!(f, "MJPEG"),
            Self::Raw => write!(f, "Raw"),
        }
    }
}

/// Audio codec type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AudioCodec {
    /// AAC (Advanced Audio Coding).
    Aac,
    /// MP3 (MPEG Layer 3).
    Mp3,
    /// Opus.
    Opus,
    /// Vorbis.
    Vorbis,
    /// FLAC (Free Lossless Audio Codec).
    Flac,
    /// PCM (uncompressed).
    Pcm,
    /// AC-3 (Dolby Digital).
    Ac3,
    /// E-AC-3 (Enhanced AC-3).
    Eac3,
}

impl AudioCodec {
    /// Check if this is a lossless codec.
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::Flac | Self::Pcm)
    }

    /// Check if this codec is royalty-free.
    pub fn is_royalty_free(&self) -> bool {
        matches!(self, Self::Opus | Self::Vorbis | Self::Flac | Self::Pcm)
    }

    /// Get the typical sample rates for this codec.
    pub fn common_sample_rates(&self) -> &'static [u32] {
        match self {
            Self::Aac => &[44100, 48000, 96000],
            Self::Mp3 => &[32000, 44100, 48000],
            Self::Opus => &[48000], // Opus always uses 48kHz internally
            Self::Vorbis => &[44100, 48000],
            Self::Flac => &[44100, 48000, 96000, 192000],
            Self::Pcm => &[44100, 48000, 96000, 192000],
            Self::Ac3 | Self::Eac3 => &[48000],
        }
    }
}

impl fmt::Display for AudioCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aac => write!(f, "AAC"),
            Self::Mp3 => write!(f, "MP3"),
            Self::Opus => write!(f, "Opus"),
            Self::Vorbis => write!(f, "Vorbis"),
            Self::Flac => write!(f, "FLAC"),
            Self::Pcm => write!(f, "PCM"),
            Self::Ac3 => write!(f, "AC-3"),
            Self::Eac3 => write!(f, "E-AC-3"),
        }
    }
}

/// Stream type (video, audio, subtitle, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StreamType {
    /// Video stream.
    Video,
    /// Audio stream.
    Audio,
    /// Subtitle stream.
    Subtitle,
    /// Data stream.
    Data,
    /// Unknown stream type.
    Unknown,
}

impl fmt::Display for StreamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "Video"),
            Self::Audio => write!(f, "Audio"),
            Self::Subtitle => write!(f, "Subtitle"),
            Self::Data => write!(f, "Data"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_extension() {
        assert_eq!(ContainerFormat::Mp4.extension(), "mp4");
        assert_eq!(ContainerFormat::WebM.extension(), "webm");
    }

    #[test]
    fn test_from_extension() {
        assert_eq!(
            ContainerFormat::from_extension("mp4"),
            Some(ContainerFormat::Mp4)
        );
        assert_eq!(
            ContainerFormat::from_extension("MP4"),
            Some(ContainerFormat::Mp4)
        );
        assert_eq!(ContainerFormat::from_extension("xyz"), None);
    }

    #[test]
    fn test_video_codec_royalty_free() {
        assert!(!VideoCodec::H264.is_royalty_free());
        assert!(VideoCodec::Vp9.is_royalty_free());
        assert!(VideoCodec::Av1.is_royalty_free());
    }

    #[test]
    fn test_audio_codec_lossless() {
        assert!(!AudioCodec::Aac.is_lossless());
        assert!(AudioCodec::Flac.is_lossless());
    }
}
