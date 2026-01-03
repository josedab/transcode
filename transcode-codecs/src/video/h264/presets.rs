//! Encoder presets for common use cases.
//!
//! This module provides pre-configured encoder settings for common video
//! encoding scenarios like YouTube uploads, streaming, archival, and social media.
//!
//! # Examples
//!
//! ```rust,ignore
//! use transcode_codecs::video::h264::{H264Encoder, Preset};
//!
//! // Create an encoder with YouTube 1080p preset
//! let encoder = H264Encoder::from_preset(Preset::YouTube1080p)?;
//!
//! // Create an encoder with low latency streaming preset
//! let encoder = H264Encoder::from_preset(Preset::StreamingLowLatency)?;
//!
//! // Customize a preset using the builder
//! let config = PresetBuilder::new(Preset::YouTube1080p)
//!     .with_frame_rate(60)
//!     .with_gop_size(120)
//!     .build();
//! let encoder = H264Encoder::new(config)?;
//! ```

use super::{H264EncoderConfig, H264Profile, H264Level, RateControlMode, EncoderPreset};
use super::parallel::ThreadingConfig;

/// Video encoding presets for common use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    // YouTube presets
    /// YouTube 1080p Full HD (1920x1080, 8 Mbps VBR, Main profile).
    YouTube1080p,
    /// YouTube 720p HD (1280x720, 5 Mbps VBR, Main profile).
    YouTube720p,
    /// YouTube 480p SD (854x480, 2.5 Mbps VBR, Main profile).
    YouTube480p,

    // Streaming presets
    /// Low latency streaming (reduced B-frames, smaller GOP, faster preset).
    StreamingLowLatency,
    /// Standard streaming (balanced quality and latency).
    StreamingStandard,

    // Archive presets
    /// High quality archival (CRF 18, High profile, slow preset).
    ArchiveHighQuality,
    /// Lossless archival (CQP 0, High 4:4:4 profile).
    ArchiveLossless,

    // Social media presets
    /// Instagram (1080x1080 square, optimized for feed).
    Instagram,
    /// TikTok (1080x1920 vertical, optimized for mobile).
    TikTok,
    /// Twitter/X (1280x720, optimized for timeline autoplay).
    Twitter,
}

impl Preset {
    /// Get the default width for this preset.
    pub fn width(&self) -> u32 {
        match self {
            Preset::YouTube1080p => 1920,
            Preset::YouTube720p => 1280,
            Preset::YouTube480p => 854,
            Preset::StreamingLowLatency => 1280,
            Preset::StreamingStandard => 1920,
            Preset::ArchiveHighQuality => 1920,
            Preset::ArchiveLossless => 1920,
            Preset::Instagram => 1080,
            Preset::TikTok => 1080,
            Preset::Twitter => 1280,
        }
    }

    /// Get the default height for this preset.
    pub fn height(&self) -> u32 {
        match self {
            Preset::YouTube1080p => 1080,
            Preset::YouTube720p => 720,
            Preset::YouTube480p => 480,
            Preset::StreamingLowLatency => 720,
            Preset::StreamingStandard => 1080,
            Preset::ArchiveHighQuality => 1080,
            Preset::ArchiveLossless => 1080,
            Preset::Instagram => 1080,
            Preset::TikTok => 1920,
            Preset::Twitter => 720,
        }
    }

    /// Get the default frame rate for this preset.
    pub fn frame_rate(&self) -> u32 {
        match self {
            Preset::YouTube1080p => 30,
            Preset::YouTube720p => 30,
            Preset::YouTube480p => 30,
            Preset::StreamingLowLatency => 30,
            Preset::StreamingStandard => 30,
            Preset::ArchiveHighQuality => 30,
            Preset::ArchiveLossless => 30,
            Preset::Instagram => 30,
            Preset::TikTok => 30,
            Preset::Twitter => 30,
        }
    }

    /// Get the recommended bitrate for this preset (in bits per second).
    pub fn bitrate(&self) -> Option<u32> {
        match self {
            Preset::YouTube1080p => Some(8_000_000),
            Preset::YouTube720p => Some(5_000_000),
            Preset::YouTube480p => Some(2_500_000),
            Preset::StreamingLowLatency => Some(3_000_000),
            Preset::StreamingStandard => Some(6_000_000),
            Preset::ArchiveHighQuality => None, // Uses CRF
            Preset::ArchiveLossless => None,    // Uses CQP
            Preset::Instagram => Some(3_500_000),
            Preset::TikTok => Some(6_000_000),
            Preset::Twitter => Some(5_000_000),
        }
    }

    /// Get the recommended CRF value for this preset (if applicable).
    pub fn crf(&self) -> Option<u8> {
        match self {
            Preset::ArchiveHighQuality => Some(18),
            _ => None,
        }
    }

    /// Get the recommended QP value for this preset (if applicable).
    pub fn qp(&self) -> Option<u8> {
        match self {
            Preset::ArchiveLossless => Some(0),
            _ => None,
        }
    }

    /// Get the H.264 profile for this preset.
    pub fn profile(&self) -> H264Profile {
        match self {
            Preset::YouTube1080p => H264Profile::Main,
            Preset::YouTube720p => H264Profile::Main,
            Preset::YouTube480p => H264Profile::Main,
            Preset::StreamingLowLatency => H264Profile::Baseline,
            Preset::StreamingStandard => H264Profile::Main,
            Preset::ArchiveHighQuality => H264Profile::High,
            Preset::ArchiveLossless => H264Profile::High444,
            Preset::Instagram => H264Profile::High,
            Preset::TikTok => H264Profile::High,
            Preset::Twitter => H264Profile::Main,
        }
    }

    /// Get the H.264 level for this preset.
    pub fn level(&self) -> H264Level {
        match self {
            Preset::YouTube1080p => H264Level::from_idc(40),   // 4.0
            Preset::YouTube720p => H264Level::from_idc(31),    // 3.1
            Preset::YouTube480p => H264Level::from_idc(30),    // 3.0
            Preset::StreamingLowLatency => H264Level::from_idc(31), // 3.1
            Preset::StreamingStandard => H264Level::from_idc(40),   // 4.0
            Preset::ArchiveHighQuality => H264Level::from_idc(41),  // 4.1
            Preset::ArchiveLossless => H264Level::from_idc(51),     // 5.1
            Preset::Instagram => H264Level::from_idc(40),      // 4.0
            Preset::TikTok => H264Level::from_idc(40),         // 4.0
            Preset::Twitter => H264Level::from_idc(31),        // 3.1
        }
    }

    /// Get the GOP size (keyframe interval) for this preset.
    pub fn gop_size(&self) -> u32 {
        match self {
            Preset::YouTube1080p => 250,
            Preset::YouTube720p => 250,
            Preset::YouTube480p => 250,
            Preset::StreamingLowLatency => 30, // 1 second at 30fps
            Preset::StreamingStandard => 60,  // 2 seconds at 30fps
            Preset::ArchiveHighQuality => 250,
            Preset::ArchiveLossless => 1,     // All I-frames
            Preset::Instagram => 60,
            Preset::TikTok => 60,
            Preset::Twitter => 60,
        }
    }

    /// Get the number of B-frames for this preset.
    pub fn bframes(&self) -> u8 {
        match self {
            Preset::YouTube1080p => 2,
            Preset::YouTube720p => 2,
            Preset::YouTube480p => 2,
            Preset::StreamingLowLatency => 0, // No B-frames for low latency
            Preset::StreamingStandard => 2,
            Preset::ArchiveHighQuality => 3,
            Preset::ArchiveLossless => 0,
            Preset::Instagram => 2,
            Preset::TikTok => 2,
            Preset::Twitter => 2,
        }
    }

    /// Get the number of reference frames for this preset.
    pub fn ref_frames(&self) -> u8 {
        match self {
            Preset::YouTube1080p => 4,
            Preset::YouTube720p => 4,
            Preset::YouTube480p => 3,
            Preset::StreamingLowLatency => 1,
            Preset::StreamingStandard => 3,
            Preset::ArchiveHighQuality => 5,
            Preset::ArchiveLossless => 1,
            Preset::Instagram => 3,
            Preset::TikTok => 3,
            Preset::Twitter => 3,
        }
    }

    /// Get whether CABAC entropy coding should be used.
    pub fn cabac(&self) -> bool {
        match self {
            Preset::StreamingLowLatency => false, // CAVLC for lower latency
            Preset::ArchiveLossless => true,
            _ => true,
        }
    }

    /// Get the encoder speed preset for this preset.
    pub fn encoder_preset(&self) -> EncoderPreset {
        match self {
            Preset::YouTube1080p => EncoderPreset::Slow,
            Preset::YouTube720p => EncoderPreset::Slow,
            Preset::YouTube480p => EncoderPreset::Medium,
            Preset::StreamingLowLatency => EncoderPreset::Ultrafast,
            Preset::StreamingStandard => EncoderPreset::Veryfast,
            Preset::ArchiveHighQuality => EncoderPreset::Slower,
            Preset::ArchiveLossless => EncoderPreset::Medium,
            Preset::Instagram => EncoderPreset::Medium,
            Preset::TikTok => EncoderPreset::Medium,
            Preset::Twitter => EncoderPreset::Fast,
        }
    }

    /// Get the threading configuration for this preset.
    pub fn threading_config(&self) -> ThreadingConfig {
        match self {
            // YouTube presets - use multi-threading for offline encoding
            Preset::YouTube1080p => ThreadingConfig {
                num_threads: 0, // Auto-detect
                slice_count: 4,
                lookahead_depth: 40,
                enable_slice_parallel: true,
                enable_frame_parallel: true,
            },
            Preset::YouTube720p => ThreadingConfig {
                num_threads: 0,
                slice_count: 4,
                lookahead_depth: 40,
                enable_slice_parallel: true,
                enable_frame_parallel: true,
            },
            Preset::YouTube480p => ThreadingConfig {
                num_threads: 0,
                slice_count: 2,
                lookahead_depth: 30,
                enable_slice_parallel: true,
                enable_frame_parallel: true,
            },
            // Streaming presets - optimize for latency
            Preset::StreamingLowLatency => ThreadingConfig {
                num_threads: 0,
                slice_count: 4, // More slices for faster encode
                lookahead_depth: 0, // No lookahead for minimum latency
                enable_slice_parallel: true,
                enable_frame_parallel: false, // Disable for lower latency
            },
            Preset::StreamingStandard => ThreadingConfig {
                num_threads: 0,
                slice_count: 4,
                lookahead_depth: 10, // Reduced lookahead for streaming
                enable_slice_parallel: true,
                enable_frame_parallel: true,
            },
            // Archive presets - maximize quality, don't care about speed
            Preset::ArchiveHighQuality => ThreadingConfig {
                num_threads: 0,
                slice_count: 1, // Single slice for best compression
                lookahead_depth: 60,
                enable_slice_parallel: false, // Disable for best quality
                enable_frame_parallel: true,
            },
            Preset::ArchiveLossless => ThreadingConfig {
                num_threads: 0,
                slice_count: 1,
                lookahead_depth: 0, // All I-frames, no lookahead needed
                enable_slice_parallel: false,
                enable_frame_parallel: false,
            },
            // Social media presets - balanced
            Preset::Instagram => ThreadingConfig {
                num_threads: 0,
                slice_count: 2,
                lookahead_depth: 20,
                enable_slice_parallel: true,
                enable_frame_parallel: true,
            },
            Preset::TikTok => ThreadingConfig {
                num_threads: 0,
                slice_count: 2,
                lookahead_depth: 20,
                enable_slice_parallel: true,
                enable_frame_parallel: true,
            },
            Preset::Twitter => ThreadingConfig {
                num_threads: 0,
                slice_count: 2,
                lookahead_depth: 20,
                enable_slice_parallel: true,
                enable_frame_parallel: true,
            },
        }
    }

    /// Convert the preset to an H264EncoderConfig.
    pub fn to_config(&self) -> H264EncoderConfig {
        let mut config = H264EncoderConfig::new(self.width(), self.height());
        config.frame_rate = (self.frame_rate(), 1);
        config.profile = self.profile();
        config.level = self.level();
        config.gop_size = self.gop_size();
        config.bframes = self.bframes();
        config.ref_frames = self.ref_frames();
        config.cabac = self.cabac();
        config.preset = self.encoder_preset();
        config.threading = self.threading_config();

        // Set rate control
        if let Some(crf) = self.crf() {
            config.rate_control = RateControlMode::Crf(crf);
        } else if let Some(qp) = self.qp() {
            config.rate_control = RateControlMode::Cqp(qp);
        } else if let Some(bitrate) = self.bitrate() {
            // Use VBR with 1.5x max for most presets
            let max_bitrate = (bitrate as f64 * 1.5) as u32;
            config.rate_control = RateControlMode::Vbr {
                target: bitrate,
                max: max_bitrate,
            };
        }

        config
    }

    /// Get a human-readable description of this preset.
    pub fn description(&self) -> &'static str {
        match self {
            Preset::YouTube1080p => "YouTube 1080p Full HD - optimized for YouTube uploads",
            Preset::YouTube720p => "YouTube 720p HD - optimized for YouTube uploads",
            Preset::YouTube480p => "YouTube 480p SD - optimized for YouTube uploads",
            Preset::StreamingLowLatency => "Low latency streaming - minimal delay for live streaming",
            Preset::StreamingStandard => "Standard streaming - balanced quality and latency",
            Preset::ArchiveHighQuality => "High quality archival - maximum quality for long-term storage",
            Preset::ArchiveLossless => "Lossless archival - no quality loss, large file size",
            Preset::Instagram => "Instagram - square format optimized for feed posts",
            Preset::TikTok => "TikTok - vertical format optimized for mobile viewing",
            Preset::Twitter => "Twitter/X - optimized for timeline autoplay",
        }
    }
}

/// Builder for customizing encoder presets.
///
/// Allows starting from a preset and modifying specific parameters.
///
/// # Examples
///
/// ```rust,ignore
/// use transcode_codecs::video::h264::{PresetBuilder, Preset};
///
/// let config = PresetBuilder::new(Preset::YouTube1080p)
///     .with_dimensions(1920, 1080)
///     .with_frame_rate(60)
///     .with_bitrate(12_000_000)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct PresetBuilder {
    config: H264EncoderConfig,
}

impl PresetBuilder {
    /// Create a new builder from a preset.
    pub fn new(preset: Preset) -> Self {
        Self {
            config: preset.to_config(),
        }
    }

    /// Create a builder from an existing config.
    pub fn from_config(config: H264EncoderConfig) -> Self {
        Self { config }
    }

    /// Set output dimensions.
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    /// Set frame rate.
    pub fn with_frame_rate(mut self, fps: u32) -> Self {
        self.config.frame_rate = (fps, 1);
        self
    }

    /// Set frame rate with denominator (for non-integer frame rates like 29.97).
    pub fn with_frame_rate_rational(mut self, num: u32, den: u32) -> Self {
        self.config.frame_rate = (num, den);
        self
    }

    /// Set target bitrate (CBR mode).
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.config.rate_control = RateControlMode::Cbr(bitrate);
        self
    }

    /// Set VBR mode with target and maximum bitrate.
    pub fn with_vbr(mut self, target: u32, max: u32) -> Self {
        self.config.rate_control = RateControlMode::Vbr { target, max };
        self
    }

    /// Set CRF quality value (0-51, lower is better quality).
    pub fn with_crf(mut self, crf: u8) -> Self {
        self.config.rate_control = RateControlMode::Crf(crf);
        self
    }

    /// Set constant QP value.
    pub fn with_qp(mut self, qp: u8) -> Self {
        self.config.rate_control = RateControlMode::Cqp(qp);
        self
    }

    /// Set H.264 profile.
    pub fn with_profile(mut self, profile: H264Profile) -> Self {
        self.config.profile = profile;
        self
    }

    /// Set H.264 level.
    pub fn with_level(mut self, level: H264Level) -> Self {
        self.config.level = level;
        self
    }

    /// Set GOP size (keyframe interval).
    pub fn with_gop_size(mut self, gop_size: u32) -> Self {
        self.config.gop_size = gop_size;
        self
    }

    /// Set number of B-frames.
    pub fn with_bframes(mut self, bframes: u8) -> Self {
        self.config.bframes = bframes;
        self
    }

    /// Set number of reference frames.
    pub fn with_ref_frames(mut self, ref_frames: u8) -> Self {
        self.config.ref_frames = ref_frames;
        self
    }

    /// Enable or disable CABAC entropy coding.
    pub fn with_cabac(mut self, cabac: bool) -> Self {
        self.config.cabac = cabac;
        self
    }

    /// Set encoder speed preset.
    pub fn with_encoder_preset(mut self, preset: EncoderPreset) -> Self {
        self.config.preset = preset;
        self
    }

    /// Set the threading configuration.
    pub fn with_threading(mut self, threading: ThreadingConfig) -> Self {
        self.config.threading = threading;
        self
    }

    /// Set the number of threads for parallel encoding.
    /// Use 0 for auto-detection based on CPU cores.
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.config.threading.num_threads = num_threads;
        self
    }

    /// Set the number of slices per frame for slice-based parallelism.
    pub fn with_slice_count(mut self, slice_count: usize) -> Self {
        self.config.threading.slice_count = slice_count.max(1);
        self
    }

    /// Set the lookahead depth for B-frame decisions.
    pub fn with_lookahead_depth(mut self, depth: usize) -> Self {
        self.config.threading.lookahead_depth = depth;
        self
    }

    /// Enable or disable slice-based parallel encoding.
    pub fn with_slice_parallel(mut self, enable: bool) -> Self {
        self.config.threading.enable_slice_parallel = enable;
        self
    }

    /// Enable or disable frame-based parallel encoding (motion estimation).
    pub fn with_frame_parallel(mut self, enable: bool) -> Self {
        self.config.threading.enable_frame_parallel = enable;
        self
    }

    /// Build the final encoder configuration.
    pub fn build(self) -> H264EncoderConfig {
        self.config
    }
}

/// Get all available presets.
pub fn all_presets() -> Vec<Preset> {
    vec![
        Preset::YouTube1080p,
        Preset::YouTube720p,
        Preset::YouTube480p,
        Preset::StreamingLowLatency,
        Preset::StreamingStandard,
        Preset::ArchiveHighQuality,
        Preset::ArchiveLossless,
        Preset::Instagram,
        Preset::TikTok,
        Preset::Twitter,
    ]
}

/// Get presets for a specific category.
pub fn presets_for_category(category: PresetCategory) -> Vec<Preset> {
    match category {
        PresetCategory::YouTube => vec![
            Preset::YouTube1080p,
            Preset::YouTube720p,
            Preset::YouTube480p,
        ],
        PresetCategory::Streaming => vec![
            Preset::StreamingLowLatency,
            Preset::StreamingStandard,
        ],
        PresetCategory::Archive => vec![
            Preset::ArchiveHighQuality,
            Preset::ArchiveLossless,
        ],
        PresetCategory::SocialMedia => vec![
            Preset::Instagram,
            Preset::TikTok,
            Preset::Twitter,
        ],
    }
}

/// Preset categories for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetCategory {
    /// YouTube upload presets.
    YouTube,
    /// Live streaming presets.
    Streaming,
    /// Archival presets.
    Archive,
    /// Social media presets.
    SocialMedia,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_youtube_1080p_preset() {
        let config = Preset::YouTube1080p.to_config();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.profile, H264Profile::Main);
        assert!(matches!(config.rate_control, RateControlMode::Vbr { target: 8_000_000, .. }));
        assert_eq!(config.gop_size, 250);
        assert_eq!(config.bframes, 2);
        assert!(config.cabac);
    }

    #[test]
    fn test_youtube_720p_preset() {
        let config = Preset::YouTube720p.to_config();
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.profile, H264Profile::Main);
        assert!(matches!(config.rate_control, RateControlMode::Vbr { target: 5_000_000, .. }));
    }

    #[test]
    fn test_youtube_480p_preset() {
        let config = Preset::YouTube480p.to_config();
        assert_eq!(config.width, 854);
        assert_eq!(config.height, 480);
        assert_eq!(config.profile, H264Profile::Main);
        assert!(matches!(config.rate_control, RateControlMode::Vbr { target: 2_500_000, .. }));
    }

    #[test]
    fn test_streaming_low_latency_preset() {
        let config = Preset::StreamingLowLatency.to_config();
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.profile, H264Profile::Baseline);
        assert_eq!(config.gop_size, 30); // 1 second GOP
        assert_eq!(config.bframes, 0);   // No B-frames for low latency
        assert_eq!(config.ref_frames, 1);
        assert!(!config.cabac);          // CAVLC for lower latency
        assert_eq!(config.preset, EncoderPreset::Ultrafast);
    }

    #[test]
    fn test_streaming_standard_preset() {
        let config = Preset::StreamingStandard.to_config();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.profile, H264Profile::Main);
        assert_eq!(config.gop_size, 60);
        assert_eq!(config.bframes, 2);
        assert!(config.cabac);
        assert_eq!(config.preset, EncoderPreset::Veryfast);
    }

    #[test]
    fn test_archive_high_quality_preset() {
        let config = Preset::ArchiveHighQuality.to_config();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.profile, H264Profile::High);
        assert!(matches!(config.rate_control, RateControlMode::Crf(18)));
        assert_eq!(config.bframes, 3);
        assert_eq!(config.ref_frames, 5);
        assert_eq!(config.preset, EncoderPreset::Slower);
    }

    #[test]
    fn test_archive_lossless_preset() {
        let config = Preset::ArchiveLossless.to_config();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.profile, H264Profile::High444);
        assert!(matches!(config.rate_control, RateControlMode::Cqp(0)));
        assert_eq!(config.gop_size, 1); // All I-frames
        assert_eq!(config.bframes, 0);
    }

    #[test]
    fn test_instagram_preset() {
        let config = Preset::Instagram.to_config();
        assert_eq!(config.width, 1080);
        assert_eq!(config.height, 1080); // Square
        assert_eq!(config.profile, H264Profile::High);
        assert!(matches!(config.rate_control, RateControlMode::Vbr { target: 3_500_000, .. }));
    }

    #[test]
    fn test_tiktok_preset() {
        let config = Preset::TikTok.to_config();
        assert_eq!(config.width, 1080);
        assert_eq!(config.height, 1920); // Vertical
        assert_eq!(config.profile, H264Profile::High);
        assert!(matches!(config.rate_control, RateControlMode::Vbr { target: 6_000_000, .. }));
    }

    #[test]
    fn test_twitter_preset() {
        let config = Preset::Twitter.to_config();
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.profile, H264Profile::Main);
        assert!(matches!(config.rate_control, RateControlMode::Vbr { target: 5_000_000, .. }));
        assert_eq!(config.preset, EncoderPreset::Fast);
    }

    #[test]
    fn test_preset_builder() {
        let config = PresetBuilder::new(Preset::YouTube1080p)
            .with_dimensions(3840, 2160)
            .with_frame_rate(60)
            .with_bitrate(20_000_000)
            .with_profile(H264Profile::High)
            .with_level(H264Level::from_idc(51))
            .build();

        assert_eq!(config.width, 3840);
        assert_eq!(config.height, 2160);
        assert_eq!(config.frame_rate, (60, 1));
        assert!(matches!(config.rate_control, RateControlMode::Cbr(20_000_000)));
        assert_eq!(config.profile, H264Profile::High);
        assert_eq!(config.level.value, 51);
    }

    #[test]
    fn test_preset_builder_vbr() {
        let config = PresetBuilder::new(Preset::YouTube720p)
            .with_vbr(5_000_000, 8_000_000)
            .build();

        assert!(matches!(
            config.rate_control,
            RateControlMode::Vbr { target: 5_000_000, max: 8_000_000 }
        ));
    }

    #[test]
    fn test_preset_builder_crf() {
        let config = PresetBuilder::new(Preset::YouTube1080p)
            .with_crf(20)
            .build();

        assert!(matches!(config.rate_control, RateControlMode::Crf(20)));
    }

    #[test]
    fn test_preset_builder_rational_frame_rate() {
        let config = PresetBuilder::new(Preset::YouTube1080p)
            .with_frame_rate_rational(30000, 1001) // 29.97 fps
            .build();

        assert_eq!(config.frame_rate, (30000, 1001));
    }

    #[test]
    fn test_preset_builder_gop_and_bframes() {
        let config = PresetBuilder::new(Preset::YouTube1080p)
            .with_gop_size(120)
            .with_bframes(4)
            .with_ref_frames(6)
            .build();

        assert_eq!(config.gop_size, 120);
        assert_eq!(config.bframes, 4);
        assert_eq!(config.ref_frames, 6);
    }

    #[test]
    fn test_preset_builder_cabac() {
        let config = PresetBuilder::new(Preset::StreamingLowLatency)
            .with_cabac(true)
            .build();

        assert!(config.cabac);
    }

    #[test]
    fn test_preset_builder_encoder_preset() {
        let config = PresetBuilder::new(Preset::YouTube1080p)
            .with_encoder_preset(EncoderPreset::Placebo)
            .build();

        assert_eq!(config.preset, EncoderPreset::Placebo);
    }

    #[test]
    fn test_preset_builder_from_config() {
        let original = H264EncoderConfig::new(1280, 720);
        let config = PresetBuilder::from_config(original)
            .with_frame_rate(60)
            .build();

        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.frame_rate, (60, 1));
    }

    #[test]
    fn test_all_presets() {
        let presets = all_presets();
        assert_eq!(presets.len(), 10);
        assert!(presets.contains(&Preset::YouTube1080p));
        assert!(presets.contains(&Preset::StreamingLowLatency));
        assert!(presets.contains(&Preset::ArchiveHighQuality));
        assert!(presets.contains(&Preset::Instagram));
    }

    #[test]
    fn test_presets_for_category() {
        let youtube = presets_for_category(PresetCategory::YouTube);
        assert_eq!(youtube.len(), 3);
        assert!(youtube.contains(&Preset::YouTube1080p));
        assert!(youtube.contains(&Preset::YouTube720p));
        assert!(youtube.contains(&Preset::YouTube480p));

        let streaming = presets_for_category(PresetCategory::Streaming);
        assert_eq!(streaming.len(), 2);
        assert!(streaming.contains(&Preset::StreamingLowLatency));
        assert!(streaming.contains(&Preset::StreamingStandard));

        let archive = presets_for_category(PresetCategory::Archive);
        assert_eq!(archive.len(), 2);
        assert!(archive.contains(&Preset::ArchiveHighQuality));
        assert!(archive.contains(&Preset::ArchiveLossless));

        let social = presets_for_category(PresetCategory::SocialMedia);
        assert_eq!(social.len(), 3);
        assert!(social.contains(&Preset::Instagram));
        assert!(social.contains(&Preset::TikTok));
        assert!(social.contains(&Preset::Twitter));
    }

    #[test]
    fn test_preset_descriptions() {
        assert!(!Preset::YouTube1080p.description().is_empty());
        assert!(!Preset::StreamingLowLatency.description().is_empty());
        assert!(!Preset::ArchiveHighQuality.description().is_empty());
        assert!(!Preset::Instagram.description().is_empty());
    }

    #[test]
    fn test_all_presets_produce_valid_configs() {
        for preset in all_presets() {
            let config = preset.to_config();

            // Verify dimensions are reasonable
            assert!(config.width > 0);
            assert!(config.height > 0);

            // Verify frame rate is set
            assert!(config.frame_rate.0 > 0);
            assert!(config.frame_rate.1 > 0);

            // Verify GOP size is positive
            assert!(config.gop_size > 0);

            // Verify ref_frames is reasonable
            assert!(config.ref_frames > 0);
            assert!(config.ref_frames <= 16);
        }
    }

    #[test]
    fn test_preset_profile_allows_bframes_when_used() {
        for preset in all_presets() {
            let config = preset.to_config();
            if config.bframes > 0 {
                assert!(
                    config.profile.allows_b_frames(),
                    "Preset {:?} uses B-frames but profile {:?} doesn't allow them",
                    preset,
                    config.profile
                );
            }
        }
    }

    #[test]
    fn test_preset_profile_allows_cabac_when_used() {
        for preset in all_presets() {
            let config = preset.to_config();
            if config.cabac {
                assert!(
                    config.profile.allows_cabac(),
                    "Preset {:?} uses CABAC but profile {:?} doesn't allow it",
                    preset,
                    config.profile
                );
            }
        }
    }
}
