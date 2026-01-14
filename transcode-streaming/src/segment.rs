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

    // ===== Quality tests =====

    #[test]
    fn test_quality_creation() {
        let quality = Quality::new(1920, 1080, 5_000_000);
        assert_eq!(quality.resolution_string(), "1920x1080");
        assert_eq!(quality.name, "1080p");
        assert_eq!(quality.width, 1920);
        assert_eq!(quality.height, 1080);
        assert_eq!(quality.bitrate, 5_000_000);
    }

    #[test]
    fn test_quality_defaults() {
        let quality = Quality::new(1280, 720, 2_500_000);
        assert_eq!(quality.video_codec, "avc1.64001f");
        assert_eq!(quality.audio_codec, "mp4a.40.2");
        assert_eq!(quality.framerate, 30.0);
    }

    #[test]
    fn test_quality_builder_methods() {
        let quality = Quality::new(1920, 1080, 5_000_000)
            .with_video_codec("hvc1.1.6.L93.B0")
            .with_audio_codec("mp4a.40.5")
            .with_framerate(60.0)
            .with_name("1080p60");

        assert_eq!(quality.video_codec, "hvc1.1.6.L93.B0");
        assert_eq!(quality.audio_codec, "mp4a.40.5");
        assert_eq!(quality.framerate, 60.0);
        assert_eq!(quality.name, "1080p60");
    }

    #[test]
    fn test_quality_codecs_string() {
        let quality = Quality::new(1920, 1080, 5_000_000);
        assert_eq!(quality.codecs_string(), "avc1.64001f,mp4a.40.2");
    }

    #[test]
    fn test_quality_bandwidth() {
        let quality = Quality::new(1920, 1080, 5_000_000);
        // Should be 10% overhead
        assert_eq!(quality.bandwidth(), 5_500_000);
    }

    #[test]
    fn test_quality_presets() {
        let uhd = Quality::uhd_4k();
        assert_eq!(uhd.width, 3840);
        assert_eq!(uhd.height, 2160);
        assert_eq!(uhd.bitrate, 15_000_000);

        let fhd = Quality::fhd_1080p();
        assert_eq!(fhd.width, 1920);
        assert_eq!(fhd.height, 1080);
        assert_eq!(fhd.bitrate, 5_000_000);

        let hd = Quality::hd_720p();
        assert_eq!(hd.width, 1280);
        assert_eq!(hd.height, 720);
        assert_eq!(hd.bitrate, 2_500_000);

        let sd = Quality::sd_480p();
        assert_eq!(sd.width, 854);
        assert_eq!(sd.height, 480);
        assert_eq!(sd.bitrate, 1_000_000);

        let low = Quality::low_360p();
        assert_eq!(low.width, 640);
        assert_eq!(low.height, 360);
        assert_eq!(low.bitrate, 500_000);
    }

    #[test]
    fn test_quality_serialization() {
        let quality = Quality::new(1920, 1080, 5_000_000);
        let json = serde_json::to_string(&quality).unwrap();
        assert!(json.contains("\"width\":1920"));
        assert!(json.contains("\"height\":1080"));

        let deserialized: Quality = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, quality);
    }

    // ===== SegmentType tests =====

    #[test]
    fn test_segment_type_variants() {
        assert_eq!(SegmentType::Init, SegmentType::Init);
        assert_eq!(SegmentType::Media, SegmentType::Media);
        assert_eq!(SegmentType::Combined, SegmentType::Combined);
        assert_ne!(SegmentType::Init, SegmentType::Media);
    }

    // ===== Segment tests =====

    #[test]
    fn test_segment_creation() {
        let segment = Segment::new(
            0,
            SegmentType::Media,
            6.0,
            0.0,
            "segment_00000.ts",
            "1080p",
        );
        assert_eq!(segment.sequence, 0);
        assert_eq!(segment.segment_type, SegmentType::Media);
        assert_eq!(segment.duration, 6.0);
        assert_eq!(segment.start_time, 0.0);
        assert_eq!(segment.path, "segment_00000.ts");
        assert_eq!(segment.quality_name, "1080p");
        assert!(segment.keyframe); // First segment (sequence 0) should be keyframe
    }

    #[test]
    fn test_segment_init_is_keyframe() {
        let segment = Segment::new(
            5,
            SegmentType::Init,
            0.0,
            0.0,
            "init.mp4",
            "1080p",
        );
        assert!(segment.keyframe); // Init segments are always keyframes
    }

    #[test]
    fn test_segment_builder_methods() {
        let segment = Segment::new(
            1,
            SegmentType::Media,
            6.0,
            6.0,
            "segment_00001.ts",
            "720p",
        )
        .with_size(1024 * 1024)
        .with_keyframe(false);

        assert_eq!(segment.size, 1024 * 1024);
        assert!(!segment.keyframe);
    }

    #[test]
    fn test_segment_end_time() {
        let segment = Segment::new(
            0,
            SegmentType::Media,
            6.0,
            10.0,
            "segment.ts",
            "1080p",
        );
        assert_eq!(segment.end_time(), 16.0);
    }

    #[test]
    fn test_segment_serialization() {
        let segment = Segment::new(
            0,
            SegmentType::Media,
            6.0,
            0.0,
            "segment_00000.ts",
            "1080p",
        );
        let json = serde_json::to_string(&segment).unwrap();
        assert!(json.contains("\"sequence\":0"));
        assert!(json.contains("\"duration\":6.0"));

        let deserialized: Segment = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.sequence, segment.sequence);
        assert_eq!(deserialized.duration, segment.duration);
    }

    // ===== SegmentNaming tests =====

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

    #[test]
    fn test_segment_naming_uuid() {
        let name = SegmentNaming::Uuid.generate(0, "m4s");
        assert!(name.ends_with(".m4s"));
        // UUID format: 8-4-4-4-12 characters plus extension
        assert_eq!(name.len(), 36 + 4); // UUID + ".m4s"
    }

    #[test]
    fn test_segment_naming_sequential_edge_cases() {
        assert_eq!(
            SegmentNaming::Sequential.generate(0, "ts"),
            "segment_00000.ts"
        );
        assert_eq!(
            SegmentNaming::Sequential.generate(99999, "ts"),
            "segment_99999.ts"
        );
    }
}
