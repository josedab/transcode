//! FFmpeg preset mapping.
//!
//! This module maps FFmpeg encoder presets to internal quality/speed settings
//! for various codecs.

use crate::error::{CompatError, Result};
use serde::{Deserialize, Serialize};

/// Encoding speed/quality preset.
///
/// These presets control the trade-off between encoding speed and quality.
/// Slower presets generally produce better quality at the same bitrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Preset {
    /// Fastest encoding, lowest quality.
    Ultrafast,
    /// Very fast encoding.
    Superfast,
    /// Very fast encoding.
    Veryfast,
    /// Fast encoding.
    Faster,
    /// Faster than default.
    Fast,
    /// Default/medium quality.
    #[default]
    Medium,
    /// Slower encoding, better quality.
    Slow,
    /// Very slow encoding, high quality.
    Slower,
    /// Slowest encoding, highest quality.
    Veryslow,
    /// Placebo - negligible quality improvement over veryslow.
    Placebo,
}

impl Preset {
    /// Parse an FFmpeg preset name.
    ///
    /// Supported presets:
    /// - x264/x265 presets: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo
    /// - VP9 presets: best, good, realtime (mapped to our presets)
    /// - AV1 presets: 0-13 (mapped to our presets)
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase();

        match name.as_str() {
            // Standard x264/x265 presets
            "ultrafast" => Ok(Self::Ultrafast),
            "superfast" => Ok(Self::Superfast),
            "veryfast" => Ok(Self::Veryfast),
            "faster" => Ok(Self::Faster),
            "fast" => Ok(Self::Fast),
            "medium" => Ok(Self::Medium),
            "slow" => Ok(Self::Slow),
            "slower" => Ok(Self::Slower),
            "veryslow" => Ok(Self::Veryslow),
            "placebo" => Ok(Self::Placebo),

            // VP9 presets (libvpx)
            "realtime" => Ok(Self::Ultrafast),
            "good" => Ok(Self::Medium),
            "best" => Ok(Self::Veryslow),

            // Numeric presets (AV1 style: 0=slowest, higher=faster)
            "0" => Ok(Self::Placebo),
            "1" => Ok(Self::Veryslow),
            "2" => Ok(Self::Slower),
            "3" | "4" => Ok(Self::Slow),
            "5" | "6" => Ok(Self::Medium),
            "7" | "8" => Ok(Self::Fast),
            "9" | "10" => Ok(Self::Veryfast),
            "11" | "12" => Ok(Self::Superfast),
            "13" => Ok(Self::Ultrafast),

            _ => Err(CompatError::UnknownPreset(name)),
        }
    }

    /// Get the FFmpeg preset name for x264/x265.
    pub fn x264_name(&self) -> &'static str {
        match self {
            Self::Ultrafast => "ultrafast",
            Self::Superfast => "superfast",
            Self::Veryfast => "veryfast",
            Self::Faster => "faster",
            Self::Fast => "fast",
            Self::Medium => "medium",
            Self::Slow => "slow",
            Self::Slower => "slower",
            Self::Veryslow => "veryslow",
            Self::Placebo => "placebo",
        }
    }

    /// Get the VP9 deadline/quality setting.
    pub fn vp9_deadline(&self) -> &'static str {
        match self {
            Self::Ultrafast | Self::Superfast => "realtime",
            Self::Veryfast | Self::Faster | Self::Fast | Self::Medium => "good",
            Self::Slow | Self::Slower | Self::Veryslow | Self::Placebo => "best",
        }
    }

    /// Get the AV1 CPU usage level (0-8, lower is slower/better quality).
    pub fn av1_cpu_used(&self) -> u8 {
        match self {
            Self::Placebo => 0,
            Self::Veryslow => 1,
            Self::Slower => 2,
            Self::Slow => 3,
            Self::Medium => 4,
            Self::Fast => 5,
            Self::Faster => 6,
            Self::Veryfast => 7,
            Self::Superfast | Self::Ultrafast => 8,
        }
    }

    /// Get a speed value from 0 (slowest) to 10 (fastest).
    pub fn speed_value(&self) -> u8 {
        match self {
            Self::Placebo => 0,
            Self::Veryslow => 1,
            Self::Slower => 2,
            Self::Slow => 3,
            Self::Medium => 5,
            Self::Fast => 6,
            Self::Faster => 7,
            Self::Veryfast => 8,
            Self::Superfast => 9,
            Self::Ultrafast => 10,
        }
    }

    /// Get a quality value from 0 (lowest quality) to 10 (highest quality).
    /// Inverse of speed value.
    pub fn quality_value(&self) -> u8 {
        10 - self.speed_value()
    }

    /// Check if this is a fast preset (faster, veryfast, superfast, ultrafast).
    pub fn is_fast(&self) -> bool {
        matches!(
            self,
            Self::Faster | Self::Veryfast | Self::Superfast | Self::Ultrafast
        )
    }

    /// Check if this is a slow preset (slow, slower, veryslow, placebo).
    pub fn is_slow(&self) -> bool {
        matches!(
            self,
            Self::Slow | Self::Slower | Self::Veryslow | Self::Placebo
        )
    }
}


impl std::fmt::Display for Preset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.x264_name())
    }
}

/// Encoding tune setting.
///
/// Tune settings optimize the encoder for specific content types or use cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tune {
    /// Film content (live action).
    Film,
    /// Animation content.
    Animation,
    /// High detail content (grain preservation).
    Grain,
    /// Still image or slideshow.
    Stillimage,
    /// Fast decode (hardware constrained devices).
    Fastdecode,
    /// Zero latency (streaming).
    Zerolatency,
    /// PSNR optimization.
    Psnr,
    /// SSIM optimization.
    Ssim,
}

impl Tune {
    /// Parse an FFmpeg tune name.
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase();

        match name.as_str() {
            "film" => Ok(Self::Film),
            "animation" => Ok(Self::Animation),
            "grain" => Ok(Self::Grain),
            "stillimage" => Ok(Self::Stillimage),
            "fastdecode" => Ok(Self::Fastdecode),
            "zerolatency" => Ok(Self::Zerolatency),
            "psnr" => Ok(Self::Psnr),
            "ssim" => Ok(Self::Ssim),
            _ => Err(CompatError::UnknownPreset(format!("tune: {}", name))),
        }
    }

    /// Get the FFmpeg tune name.
    pub fn ffmpeg_name(&self) -> &'static str {
        match self {
            Self::Film => "film",
            Self::Animation => "animation",
            Self::Grain => "grain",
            Self::Stillimage => "stillimage",
            Self::Fastdecode => "fastdecode",
            Self::Zerolatency => "zerolatency",
            Self::Psnr => "psnr",
            Self::Ssim => "ssim",
        }
    }

    /// Check if this tune enables low latency mode.
    pub fn is_low_latency(&self) -> bool {
        matches!(self, Self::Zerolatency)
    }
}

impl std::fmt::Display for Tune {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ffmpeg_name())
    }
}

/// H.264/AVC profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum H264Profile {
    /// Baseline profile (most compatible).
    Baseline,
    /// Main profile.
    Main,
    /// High profile (best quality).
    High,
    /// High 10-bit profile.
    High10,
    /// High 4:2:2 profile.
    High422,
    /// High 4:4:4 Predictive profile.
    High444,
}

impl H264Profile {
    /// Parse an FFmpeg profile name.
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase();

        match name.as_str() {
            "baseline" | "base" => Ok(Self::Baseline),
            "main" => Ok(Self::Main),
            "high" => Ok(Self::High),
            "high10" | "high_10" => Ok(Self::High10),
            "high422" | "high_422" => Ok(Self::High422),
            "high444" | "high_444" | "high444predictive" => Ok(Self::High444),
            _ => Err(CompatError::UnknownPreset(format!("profile: {}", name))),
        }
    }

    /// Get the FFmpeg profile name.
    pub fn ffmpeg_name(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Main => "main",
            Self::High => "high",
            Self::High10 => "high10",
            Self::High422 => "high422",
            Self::High444 => "high444",
        }
    }

    /// Check if this profile supports 10-bit.
    pub fn supports_10bit(&self) -> bool {
        matches!(self, Self::High10 | Self::High422 | Self::High444)
    }

    /// Check if this profile supports B-frames.
    pub fn supports_bframes(&self) -> bool {
        !matches!(self, Self::Baseline)
    }
}

impl std::fmt::Display for H264Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ffmpeg_name())
    }
}

/// H.265/HEVC profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum H265Profile {
    /// Main profile (8-bit 4:2:0).
    Main,
    /// Main 10 profile (10-bit 4:2:0).
    Main10,
    /// Main Still Picture profile.
    MainStillPicture,
    /// Main 4:2:2 10 profile.
    Main422_10,
    /// Main 4:4:4 profile.
    Main444,
    /// Main 4:4:4 10 profile.
    Main444_10,
}

impl H265Profile {
    /// Parse an FFmpeg profile name.
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase().replace(['-', '_'], "");

        match name.as_str() {
            "main" => Ok(Self::Main),
            "main10" => Ok(Self::Main10),
            "mainstillpicture" | "mainsp" | "stillpicture" => Ok(Self::MainStillPicture),
            "main42210" | "main422" => Ok(Self::Main422_10),
            "main444" => Ok(Self::Main444),
            "main44410" => Ok(Self::Main444_10),
            _ => Err(CompatError::UnknownPreset(format!("profile: {}", name))),
        }
    }

    /// Get the FFmpeg profile name.
    pub fn ffmpeg_name(&self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Main10 => "main10",
            Self::MainStillPicture => "mainstillpicture",
            Self::Main422_10 => "main422-10",
            Self::Main444 => "main444",
            Self::Main444_10 => "main444-10",
        }
    }

    /// Check if this profile supports 10-bit.
    pub fn supports_10bit(&self) -> bool {
        matches!(
            self,
            Self::Main10 | Self::Main422_10 | Self::Main444_10
        )
    }
}

impl std::fmt::Display for H265Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.ffmpeg_name())
    }
}

/// Encoding level (codec-specific).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Level(pub f32);

impl Level {
    /// Parse an FFmpeg level string.
    ///
    /// Supports formats like "4.0", "40", "4", "high".
    pub fn parse(name: &str) -> Result<Self> {
        let name = name.trim().to_lowercase();

        // Handle named levels
        match name.as_str() {
            "auto" | "none" => return Ok(Self(0.0)),
            _ => {}
        }

        // Try parsing as float
        if let Ok(level) = name.parse::<f32>() {
            // If it's a whole number >= 10, it's in the "40" format
            if level >= 10.0 && level == level.floor() {
                return Ok(Self(level / 10.0));
            }
            return Ok(Self(level));
        }

        Err(CompatError::InvalidValue {
            option: "level".to_string(),
            value: name,
            reason: "invalid level format".to_string(),
        })
    }

    /// Get the level as a float (e.g., 4.0).
    pub fn as_f32(&self) -> f32 {
        self.0
    }

    /// Get the level as an integer (e.g., 40 for level 4.0).
    pub fn as_int(&self) -> u32 {
        (self.0 * 10.0) as u32
    }

    /// H.264 Level 3.0.
    pub fn h264_3_0() -> Self {
        Self(3.0)
    }

    /// H.264 Level 3.1.
    pub fn h264_3_1() -> Self {
        Self(3.1)
    }

    /// H.264 Level 4.0.
    pub fn h264_4_0() -> Self {
        Self(4.0)
    }

    /// H.264 Level 4.1.
    pub fn h264_4_1() -> Self {
        Self(4.1)
    }

    /// H.264 Level 5.0.
    pub fn h264_5_0() -> Self {
        Self(5.0)
    }

    /// H.264 Level 5.1.
    pub fn h264_5_1() -> Self {
        Self(5.1)
    }
}

impl std::fmt::Display for Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0.0 {
            write!(f, "auto")
        } else {
            write!(f, "{:.1}", self.0)
        }
    }
}

/// Encoder settings combined.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncoderSettings {
    /// Speed/quality preset.
    pub preset: Option<Preset>,
    /// Content tune.
    pub tune: Option<Tune>,
    /// H.264 profile.
    pub h264_profile: Option<H264Profile>,
    /// H.265 profile.
    pub h265_profile: Option<H265Profile>,
    /// Encoding level.
    pub level: Option<Level>,
    /// CRF (Constant Rate Factor) value.
    pub crf: Option<f32>,
    /// QP (Quantization Parameter) value.
    pub qp: Option<u8>,
    /// Keyframe interval (GOP size).
    pub keyframe_interval: Option<u32>,
    /// Minimum keyframe interval.
    pub min_keyframe_interval: Option<u32>,
    /// Number of B-frames.
    pub bframes: Option<u8>,
    /// Reference frames.
    pub ref_frames: Option<u8>,
    /// Two-pass encoding.
    pub two_pass: bool,
    /// Lookahead frames.
    pub lookahead: Option<u32>,
}

impl EncoderSettings {
    /// Create new empty settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set preset.
    pub fn with_preset(mut self, preset: Preset) -> Self {
        self.preset = Some(preset);
        self
    }

    /// Set tune.
    pub fn with_tune(mut self, tune: Tune) -> Self {
        self.tune = Some(tune);
        self
    }

    /// Set CRF.
    pub fn with_crf(mut self, crf: f32) -> Self {
        self.crf = Some(crf);
        self
    }

    /// Set keyframe interval.
    pub fn with_keyframe_interval(mut self, interval: u32) -> Self {
        self.keyframe_interval = Some(interval);
        self
    }

    /// Get recommended CRF for a quality level (0-10, 10 being highest quality).
    pub fn crf_for_quality(quality: u8) -> f32 {
        // Map quality 0-10 to CRF ~35-18 (lower CRF = better quality)
        35.0 - (quality.min(10) as f32 * 1.7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_parse() {
        assert_eq!(Preset::parse("ultrafast").unwrap(), Preset::Ultrafast);
        assert_eq!(Preset::parse("medium").unwrap(), Preset::Medium);
        assert_eq!(Preset::parse("veryslow").unwrap(), Preset::Veryslow);
        assert_eq!(Preset::parse("MEDIUM").unwrap(), Preset::Medium);
    }

    #[test]
    fn test_preset_parse_numeric() {
        assert_eq!(Preset::parse("0").unwrap(), Preset::Placebo);
        assert_eq!(Preset::parse("5").unwrap(), Preset::Medium);
        assert_eq!(Preset::parse("13").unwrap(), Preset::Ultrafast);
    }

    #[test]
    fn test_preset_parse_vp9() {
        assert_eq!(Preset::parse("realtime").unwrap(), Preset::Ultrafast);
        assert_eq!(Preset::parse("good").unwrap(), Preset::Medium);
        assert_eq!(Preset::parse("best").unwrap(), Preset::Veryslow);
    }

    #[test]
    fn test_preset_values() {
        assert_eq!(Preset::Ultrafast.speed_value(), 10);
        assert_eq!(Preset::Medium.speed_value(), 5);
        assert_eq!(Preset::Placebo.speed_value(), 0);

        assert_eq!(Preset::Ultrafast.quality_value(), 0);
        assert_eq!(Preset::Medium.quality_value(), 5);
        assert_eq!(Preset::Placebo.quality_value(), 10);
    }

    #[test]
    fn test_tune_parse() {
        assert_eq!(Tune::parse("film").unwrap(), Tune::Film);
        assert_eq!(Tune::parse("animation").unwrap(), Tune::Animation);
        assert_eq!(Tune::parse("zerolatency").unwrap(), Tune::Zerolatency);
    }

    #[test]
    fn test_h264_profile_parse() {
        assert_eq!(
            H264Profile::parse("baseline").unwrap(),
            H264Profile::Baseline
        );
        assert_eq!(H264Profile::parse("main").unwrap(), H264Profile::Main);
        assert_eq!(H264Profile::parse("high").unwrap(), H264Profile::High);
        assert_eq!(H264Profile::parse("high10").unwrap(), H264Profile::High10);
    }

    #[test]
    fn test_h265_profile_parse() {
        assert_eq!(H265Profile::parse("main").unwrap(), H265Profile::Main);
        assert_eq!(H265Profile::parse("main10").unwrap(), H265Profile::Main10);
        assert_eq!(H265Profile::parse("main444").unwrap(), H265Profile::Main444);
    }

    #[test]
    fn test_level_parse() {
        assert!((Level::parse("4.0").unwrap().as_f32() - 4.0).abs() < 0.001);
        assert!((Level::parse("40").unwrap().as_f32() - 4.0).abs() < 0.001);
        assert!((Level::parse("5.1").unwrap().as_f32() - 5.1).abs() < 0.001);
    }

    #[test]
    fn test_encoder_settings() {
        let settings = EncoderSettings::new()
            .with_preset(Preset::Medium)
            .with_crf(23.0)
            .with_keyframe_interval(250);

        assert_eq!(settings.preset, Some(Preset::Medium));
        assert_eq!(settings.crf, Some(23.0));
        assert_eq!(settings.keyframe_interval, Some(250));
    }

    #[test]
    fn test_crf_for_quality() {
        // Highest quality should give lowest CRF
        let high_quality_crf = EncoderSettings::crf_for_quality(10);
        let low_quality_crf = EncoderSettings::crf_for_quality(0);

        assert!(high_quality_crf < low_quality_crf);
        assert!(high_quality_crf >= 18.0);
        assert!(low_quality_crf <= 35.0);
    }
}
