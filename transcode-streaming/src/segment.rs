//! Segment types and quality definitions.

use serde::{Deserialize, Serialize};

/// Video quality level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quality {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Target bitrate in bits per second.
    pub bitrate: u64,
    /// Video codec (e.g., "avc1.64001f", "hvc1.1.6.L93.B0").
    pub video_codec: String,
    /// Audio codec (e.g., "mp4a.40.2").
    pub audio_codec: String,
    /// Frame rate.
    pub framerate: f32,
    /// Quality name/label.
    pub name: String,
}

impl Quality {
    /// Create a new quality level.
    pub fn new(width: u32, height: u32, bitrate: u64) -> Self {
        let name = format!("{}p", height);
        Self {
            width,
            height,
            bitrate,
            video_codec: "avc1.64001f".to_string(),
            audio_codec: "mp4a.40.2".to_string(),
            framerate: 30.0,
            name,
        }
    }

    /// Set video codec.
    pub fn with_video_codec(mut self, codec: impl Into<String>) -> Self {
        self.video_codec = codec.into();
        self
    }

    /// Set audio codec.
    pub fn with_audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.audio_codec = codec.into();
        self
    }

    /// Set frame rate.
    pub fn with_framerate(mut self, framerate: f32) -> Self {
        self.framerate = framerate;
        self
    }

    /// Set quality name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get combined codec string for HLS.
    pub fn codecs_string(&self) -> String {
        format!("{},{}", self.video_codec, self.audio_codec)
    }

    /// Get resolution string.
    pub fn resolution_string(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// Get bandwidth for HLS (video + audio + overhead).
    pub fn bandwidth(&self) -> u64 {
        // Add 10% overhead for container and audio
        (self.bitrate as f64 * 1.1) as u64
    }

    /// Create 4K quality.
    pub fn uhd_4k() -> Self {
        Self::new(3840, 2160, 15_000_000)
    }

    /// Create 1080p quality.
    pub fn fhd_1080p() -> Self {
        Self::new(1920, 1080, 5_000_000)
    }

    /// Create 720p quality.
    pub fn hd_720p() -> Self {
        Self::new(1280, 720, 2_500_000)
    }

    /// Create 480p quality.
    pub fn sd_480p() -> Self {
        Self::new(854, 480, 1_000_000)
    }

    /// Create 360p quality.
    pub fn low_360p() -> Self {
        Self::new(640, 360, 500_000)
    }
}

/// Segment type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentType {
    /// Initialization segment (moov).
    Init,
    /// Media segment (moof + mdat).
    Media,
    /// Combined init + media segment.
    Combined,
}

/// A streaming segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Segment sequence number.
    pub sequence: u64,
    /// Segment type.
    pub segment_type: SegmentType,
    /// Duration in seconds.
    pub duration: f64,
    /// Start time in seconds.
    pub start_time: f64,
    /// File path relative to output directory.
    pub path: String,
    /// File size in bytes.
    pub size: u64,
    /// Whether this is a keyframe segment.
    pub keyframe: bool,
    /// Quality level this segment belongs to.
    pub quality_name: String,
}

impl Segment {
    /// Create a new segment.
    pub fn new(
        sequence: u64,
        segment_type: SegmentType,
        duration: f64,
        start_time: f64,
        path: impl Into<String>,
        quality_name: impl Into<String>,
    ) -> Self {
        Self {
            sequence,
            segment_type,
            duration,
            start_time,
            path: path.into(),
            size: 0,
            keyframe: segment_type == SegmentType::Init || sequence == 0,
            quality_name: quality_name.into(),
        }
    }

    /// Set segment size.
    pub fn with_size(mut self, size: u64) -> Self {
        self.size = size;
        self
    }

    /// Set keyframe flag.
    pub fn with_keyframe(mut self, keyframe: bool) -> Self {
        self.keyframe = keyframe;
        self
    }

    /// Get end time.
    pub fn end_time(&self) -> f64 {
        self.start_time + self.duration
    }
}

/// Segment naming strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentNaming {
    /// Sequential numbers: segment_000.ts
    Sequential,
    /// Time-based: segment_00000.ts
    TimeBased,
    /// UUID-based: {uuid}.ts
    Uuid,
}

impl SegmentNaming {
    /// Generate segment filename.
    pub fn generate(&self, sequence: u64, extension: &str) -> String {
        match self {
            Self::Sequential => format!("segment_{:05}.{}", sequence, extension),
            Self::TimeBased => format!("segment_{:08}.{}", sequence, extension),
            Self::Uuid => format!("{}.{}", uuid::Uuid::new_v4(), extension),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_creation() {
        let quality = Quality::new(1920, 1080, 5_000_000);
        assert_eq!(quality.resolution_string(), "1920x1080");
        assert_eq!(quality.name, "1080p");
    }

    #[test]
    fn test_segment_naming() {
        assert_eq!(
            SegmentNaming::Sequential.generate(5, "ts"),
            "segment_00005.ts"
        );
        assert_eq!(
            SegmentNaming::TimeBased.generate(100, "m4s"),
            "segment_00000100.m4s"
        );
    }
}
