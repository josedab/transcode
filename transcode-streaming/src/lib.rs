//! HLS and DASH streaming output support.
//!
//! This crate provides adaptive bitrate streaming output generation
//! for the transcode library, supporting both HLS (HTTP Live Streaming)
//! and DASH (Dynamic Adaptive Streaming over HTTP) formats.
//!
//! # Features
//!
//! - HLS playlist generation (master and media playlists)
//! - DASH MPD manifest generation
//! - Multiple quality levels (adaptive bitrate)
//! - Segment management and naming
//! - DRM integration support
//!
//! # Example
//!
//! ```no_run
//! use transcode_streaming::{HlsConfig, HlsWriter, Quality};
//!
//! let config = HlsConfig::new("output")
//!     .with_segment_duration(6.0)
//!     .with_quality(Quality::new(1920, 1080, 5_000_000))
//!     .with_quality(Quality::new(1280, 720, 2_500_000))
//!     .with_quality(Quality::new(854, 480, 1_000_000));
//!
//! let mut writer = HlsWriter::new(config)?;
//! // Write segments...
//! # Ok::<(), transcode_streaming::StreamingError>(())
//! ```

#![allow(dead_code)]

mod cmaf;
mod dash;
mod error;
mod hls;
mod segment;

pub use cmaf::{
    CmafChunk, CmafConfig, CmafEncryption, CmafInitSegment, CmafMediaSegment,
    CmafOutputFormat, CmafTrackConfig, CmafWriter, DrmSystem, EncryptionScheme, FragmentType,
    SampleInfo, TrackSelector, TrackSwitchPoint, TrackType,
    ByteRange as CmafByteRange,
};
pub use dash::{
    AdaptationSet, AudioTrackConfig, ContentProtection, ContentType, DashConfig, DashDrmConfig,
    DashProfile, DashWriter, MpdManifest, MpdType, Period, Representation, SegmentTemplate,
};
pub use error::StreamingError;
pub use hls::{
    HlsConfig, HlsWriter, MasterPlaylist, MediaPlaylist, VariantStream, PlaylistType,
    // LL-HLS types
    LowLatencyConfig, PartialSegment, PreloadHint, PreloadHintType, ServerControl,
    SkipTag, RenditionReport, BlockingPlaylistRequest, SkipDirective, ByteRange,
    SessionData, AudioGroup, AudioTrack,
    // I-Frame playlist types
    IFramePlaylist, IFrameSegment,
    // Subtitle types
    SubtitleGroup, SubtitleTrack,
    // DRM key types
    HlsDrmConfig, HlsKey, HlsKeyFormat, HlsKeyMethod, HlsSessionKey,
};
pub use segment::{Quality, Segment, SegmentNaming, SegmentType};

/// Result type for streaming operations.
pub type Result<T> = std::result::Result<T, StreamingError>;

/// Common streaming configuration.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Output directory.
    pub output_dir: String,
    /// Segment duration in seconds.
    pub segment_duration: f64,
    /// Quality levels.
    pub qualities: Vec<Quality>,
    /// Enable DRM.
    pub drm_enabled: bool,
    /// DRM key ID (if enabled).
    pub drm_key_id: Option<String>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            output_dir: "output".to_string(),
            segment_duration: 6.0,
            qualities: vec![
                Quality::new(1920, 1080, 5_000_000),
                Quality::new(1280, 720, 2_500_000),
                Quality::new(854, 480, 1_000_000),
            ],
            drm_enabled: false,
            drm_key_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.output_dir, "output");
        assert_eq!(config.segment_duration, 6.0);
        assert_eq!(config.qualities.len(), 3);
        assert!(!config.drm_enabled);
        assert!(config.drm_key_id.is_none());
    }

    #[test]
    fn test_streaming_config_default_qualities() {
        let config = StreamingConfig::default();

        // First quality should be 1080p
        assert_eq!(config.qualities[0].width, 1920);
        assert_eq!(config.qualities[0].height, 1080);
        assert_eq!(config.qualities[0].bitrate, 5_000_000);

        // Second quality should be 720p
        assert_eq!(config.qualities[1].width, 1280);
        assert_eq!(config.qualities[1].height, 720);
        assert_eq!(config.qualities[1].bitrate, 2_500_000);

        // Third quality should be 480p
        assert_eq!(config.qualities[2].width, 854);
        assert_eq!(config.qualities[2].height, 480);
        assert_eq!(config.qualities[2].bitrate, 1_000_000);
    }

    #[test]
    fn test_streaming_config_custom() {
        let config = StreamingConfig {
            output_dir: "/tmp/output".to_string(),
            segment_duration: 10.0,
            qualities: vec![Quality::fhd_1080p()],
            drm_enabled: true,
            drm_key_id: Some("test-key-id".to_string()),
        };

        assert_eq!(config.output_dir, "/tmp/output");
        assert_eq!(config.segment_duration, 10.0);
        assert_eq!(config.qualities.len(), 1);
        assert!(config.drm_enabled);
        assert_eq!(config.drm_key_id.as_deref(), Some("test-key-id"));
    }

    #[test]
    fn test_result_type_alias() {
        // Verify the Result type alias works correctly
        fn test_fn() -> Result<u32> {
            Ok(42)
        }
        assert_eq!(test_fn().unwrap(), 42);
    }
}
