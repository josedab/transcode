//! Encoding presets and quality levels.

use crate::options::{AudioConfig, VideoConfig};

/// Encoding preset for speed/quality tradeoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preset {
    /// Fastest encoding, lowest quality.
    UltraFast,
    /// Very fast encoding.
    SuperFast,
    /// Fast encoding.
    VeryFast,
    /// Faster than default.
    Faster,
    /// Fast encoding.
    Fast,
    /// Balanced speed and quality.
    #[default]
    Medium,
    /// Slower encoding, better quality.
    Slow,
    /// Very slow encoding.
    Slower,
    /// Slowest encoding, best quality.
    VerySlow,
    /// Placebo - extreme compression.
    Placebo,
}

impl Preset {
    /// Get preset name as string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Preset::UltraFast => "ultrafast",
            Preset::SuperFast => "superfast",
            Preset::VeryFast => "veryfast",
            Preset::Faster => "faster",
            Preset::Fast => "fast",
            Preset::Medium => "medium",
            Preset::Slow => "slow",
            Preset::Slower => "slower",
            Preset::VerySlow => "veryslow",
            Preset::Placebo => "placebo",
        }
    }

    /// Parse preset from string.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ultrafast" => Some(Preset::UltraFast),
            "superfast" => Some(Preset::SuperFast),
            "veryfast" => Some(Preset::VeryFast),
            "faster" => Some(Preset::Faster),
            "fast" => Some(Preset::Fast),
            "medium" => Some(Preset::Medium),
            "slow" => Some(Preset::Slow),
            "slower" => Some(Preset::Slower),
            "veryslow" => Some(Preset::VerySlow),
            "placebo" => Some(Preset::Placebo),
            _ => None,
        }
    }
}

/// Quality level for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quality {
    /// Low quality, small file size.
    Low,
    /// Medium quality, balanced.
    #[default]
    Medium,
    /// High quality.
    High,
    /// Very high quality.
    VeryHigh,
    /// Lossless quality.
    Lossless,
}

impl Quality {
    /// Get CRF value for H.264/H.265.
    pub fn crf(&self) -> u32 {
        match self {
            Quality::Low => 28,
            Quality::Medium => 23,
            Quality::High => 18,
            Quality::VeryHigh => 15,
            Quality::Lossless => 0,
        }
    }

    /// Get video bitrate for 1080p content.
    pub fn video_bitrate_1080p(&self) -> u64 {
        match self {
            Quality::Low => 2_000_000,      // 2 Mbps
            Quality::Medium => 5_000_000,   // 5 Mbps
            Quality::High => 8_000_000,     // 8 Mbps
            Quality::VeryHigh => 15_000_000, // 15 Mbps
            Quality::Lossless => 50_000_000, // 50 Mbps
        }
    }

    /// Get video bitrate for 720p content.
    pub fn video_bitrate_720p(&self) -> u64 {
        match self {
            Quality::Low => 1_000_000,      // 1 Mbps
            Quality::Medium => 2_500_000,   // 2.5 Mbps
            Quality::High => 5_000_000,     // 5 Mbps
            Quality::VeryHigh => 8_000_000, // 8 Mbps
            Quality::Lossless => 25_000_000, // 25 Mbps
        }
    }

    /// Get audio bitrate in bits per second.
    pub fn audio_bitrate(&self) -> u64 {
        match self {
            Quality::Low => 96_000,       // 96 kbps
            Quality::Medium => 128_000,   // 128 kbps
            Quality::High => 192_000,     // 192 kbps
            Quality::VeryHigh => 320_000, // 320 kbps
            Quality::Lossless => 1_411_000, // CD quality
        }
    }
}

/// Output format configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// MP4 container with H.264/AAC.
    Mp4,
    /// MP4 container with H.265/AAC.
    Mp4Hevc,
    /// MKV container.
    Mkv,
    /// WebM container with VP9/Opus.
    WebM,
    /// Audio only - AAC.
    Aac,
    /// Audio only - MP3.
    Mp3,
    /// Audio only - Opus.
    Opus,
    /// Audio only - FLAC.
    Flac,
}

impl Format {
    /// Get file extension.
    pub fn extension(&self) -> &'static str {
        match self {
            Format::Mp4 | Format::Mp4Hevc => "mp4",
            Format::Mkv => "mkv",
            Format::WebM => "webm",
            Format::Aac => "aac",
            Format::Mp3 => "mp3",
            Format::Opus => "opus",
            Format::Flac => "flac",
        }
    }

    /// Get MIME type.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Format::Mp4 | Format::Mp4Hevc => "video/mp4",
            Format::Mkv => "video/x-matroska",
            Format::WebM => "video/webm",
            Format::Aac => "audio/aac",
            Format::Mp3 => "audio/mpeg",
            Format::Opus => "audio/opus",
            Format::Flac => "audio/flac",
        }
    }

    /// Check if format is audio-only.
    pub fn is_audio_only(&self) -> bool {
        matches!(self, Format::Aac | Format::Mp3 | Format::Opus | Format::Flac)
    }

    /// Get recommended video codec.
    pub fn video_codec(&self) -> Option<&'static str> {
        match self {
            Format::Mp4 => Some("h264"),
            Format::Mp4Hevc => Some("h265"),
            Format::Mkv => Some("h264"),
            Format::WebM => Some("vp9"),
            _ => None,
        }
    }

    /// Get recommended audio codec.
    pub fn audio_codec(&self) -> &'static str {
        match self {
            Format::Mp4 | Format::Mp4Hevc | Format::Aac => "aac",
            Format::Mkv => "aac",
            Format::WebM | Format::Opus => "opus",
            Format::Mp3 => "mp3",
            Format::Flac => "flac",
        }
    }

    /// Get default video configuration for this format.
    pub fn default_video_config(&self, quality: Quality) -> Option<VideoConfig> {
        self.video_codec().map(|codec| {
            VideoConfig::new()
                .codec(codec)
                .crf(quality.crf())
                .preset("medium")
        })
    }

    /// Get default audio configuration for this format.
    pub fn default_audio_config(&self, quality: Quality) -> AudioConfig {
        AudioConfig::new()
            .codec(self.audio_codec())
            .bitrate(quality.audio_bitrate())
            .sample_rate(48000)
            .channels(2)
    }
}

/// Pre-configured encoding profile for common use cases.
#[derive(Debug, Clone)]
pub struct EncodingProfile {
    /// Profile name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Output format.
    pub format: Format,
    /// Quality level.
    pub quality: Quality,
    /// Encoding preset.
    pub preset: Preset,
    /// Video configuration.
    pub video: Option<VideoConfig>,
    /// Audio configuration.
    pub audio: AudioConfig,
}

impl EncodingProfile {
    /// Web streaming profile (H.264 + AAC, optimized for web playback).
    pub fn web_streaming() -> Self {
        Self {
            name: "web_streaming".into(),
            description: "Optimized for web streaming with broad compatibility".into(),
            format: Format::Mp4,
            quality: Quality::Medium,
            preset: Preset::Fast,
            video: Some(
                VideoConfig::new()
                    .codec("h264")
                    .crf(23)
                    .preset("fast")
                    .gop_size(60),
            ),
            audio: AudioConfig::new()
                .codec("aac")
                .bitrate(128_000)
                .sample_rate(48000)
                .channels(2),
        }
    }

    /// High quality archive profile.
    pub fn archive() -> Self {
        Self {
            name: "archive".into(),
            description: "High quality for archival purposes".into(),
            format: Format::Mkv,
            quality: Quality::VeryHigh,
            preset: Preset::Slow,
            video: Some(
                VideoConfig::new()
                    .codec("h265")
                    .crf(15)
                    .preset("slow"),
            ),
            audio: AudioConfig::new()
                .codec("flac")
                .sample_rate(48000)
                .channels(2),
        }
    }

    /// Mobile-optimized profile.
    pub fn mobile() -> Self {
        Self {
            name: "mobile".into(),
            description: "Optimized for mobile devices".into(),
            format: Format::Mp4,
            quality: Quality::Medium,
            preset: Preset::VeryFast,
            video: Some(
                VideoConfig::new()
                    .codec("h264")
                    .crf(26)
                    .preset("veryfast")
                    .resolution(1280, 720),
            ),
            audio: AudioConfig::new()
                .codec("aac")
                .bitrate(96_000)
                .sample_rate(44100)
                .channels(2),
        }
    }

    /// Audio podcast profile.
    pub fn podcast() -> Self {
        Self {
            name: "podcast".into(),
            description: "Optimized for voice/podcast content".into(),
            format: Format::Mp3,
            quality: Quality::Medium,
            preset: Preset::Medium,
            video: None,
            audio: AudioConfig::new()
                .codec("mp3")
                .bitrate(128_000)
                .sample_rate(44100)
                .channels(1),
        }
    }

    /// Music streaming profile.
    pub fn music() -> Self {
        Self {
            name: "music".into(),
            description: "Optimized for music content".into(),
            format: Format::Opus,
            quality: Quality::High,
            preset: Preset::Medium,
            video: None,
            audio: AudioConfig::new()
                .codec("opus")
                .bitrate(192_000)
                .sample_rate(48000)
                .channels(2),
        }
    }

    /// 4K UHD profile.
    pub fn uhd_4k() -> Self {
        Self {
            name: "4k_uhd".into(),
            description: "4K Ultra HD content".into(),
            format: Format::Mp4Hevc,
            quality: Quality::VeryHigh,
            preset: Preset::Medium,
            video: Some(
                VideoConfig::new()
                    .codec("h265")
                    .crf(18)
                    .preset("medium")
                    .resolution(3840, 2160),
            ),
            audio: AudioConfig::new()
                .codec("aac")
                .bitrate(256_000)
                .sample_rate(48000)
                .channels(6),
        }
    }
}
