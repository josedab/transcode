//! FFmpeg-style option parsing.
//!
//! This module handles parsing of FFmpeg command-line options including:
//! - Stream-specific options with specifiers (-c:v, -b:a:0)
//! - Bitrate parsing (5M, 5000k, 5000000)
//! - Time parsing (00:01:30, 90, 1:30)
//! - Resolution parsing (1920x1080, hd1080, 4k)
//! - Aspect ratio parsing (16:9, 1.777)

use crate::error::{CompatError, Result, StreamSpecifier};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Parsed bitrate value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bitrate(pub u64);

impl Bitrate {
    /// Parse a bitrate string like "5M", "5000k", "5000000".
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Err(CompatError::InvalidBitrate("empty string".to_string()));
        }

        // Check for suffix
        let (num_str, multiplier) = if s.ends_with('k') || s.ends_with('K') {
            (&s[..s.len() - 1], 1_000u64)
        } else if s.ends_with('m') || s.ends_with('M') {
            (&s[..s.len() - 1], 1_000_000u64)
        } else if s.ends_with('g') || s.ends_with('G') {
            (&s[..s.len() - 1], 1_000_000_000u64)
        } else {
            (s, 1u64)
        };

        // Parse the numeric part
        let num: f64 = num_str
            .parse()
            .map_err(|_| CompatError::InvalidBitrate(s.to_string()))?;

        if num < 0.0 {
            return Err(CompatError::InvalidBitrate(format!(
                "negative bitrate: {}",
                s
            )));
        }

        Ok(Bitrate((num * multiplier as f64) as u64))
    }

    /// Get the bitrate in bits per second.
    pub fn bps(&self) -> u64 {
        self.0
    }

    /// Get the bitrate in kilobits per second.
    pub fn kbps(&self) -> f64 {
        self.0 as f64 / 1_000.0
    }

    /// Get the bitrate in megabits per second.
    pub fn mbps(&self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }
}

impl std::fmt::Display for Bitrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 >= 1_000_000_000 && self.0 % 1_000_000_000 == 0 {
            write!(f, "{}G", self.0 / 1_000_000_000)
        } else if self.0 >= 1_000_000 && self.0 % 1_000_000 == 0 {
            write!(f, "{}M", self.0 / 1_000_000)
        } else if self.0 >= 1_000 && self.0 % 1_000 == 0 {
            write!(f, "{}k", self.0 / 1_000)
        } else {
            write!(f, "{}", self.0)
        }
    }
}

/// Parsed time value in seconds (with fractional part).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeValue(pub f64);

impl TimeValue {
    /// Parse a time string.
    ///
    /// Supported formats:
    /// - `90` or `90.5` - seconds
    /// - `1:30` - minutes:seconds
    /// - `00:01:30` or `1:00:30` - hours:minutes:seconds
    /// - `00:01:30.500` - with milliseconds
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Err(CompatError::InvalidTime("empty string".to_string()));
        }

        let parts: Vec<&str> = s.split(':').collect();

        match parts.as_slice() {
            // Just seconds
            [secs] => {
                let seconds: f64 = secs
                    .parse()
                    .map_err(|_| CompatError::InvalidTime(s.to_string()))?;
                Ok(TimeValue(seconds))
            }
            // minutes:seconds
            [mins, secs] => {
                let minutes: f64 = mins
                    .parse()
                    .map_err(|_| CompatError::InvalidTime(s.to_string()))?;
                let seconds: f64 = secs
                    .parse()
                    .map_err(|_| CompatError::InvalidTime(s.to_string()))?;
                Ok(TimeValue(minutes * 60.0 + seconds))
            }
            // hours:minutes:seconds
            [hours, mins, secs] => {
                let hours: f64 = hours
                    .parse()
                    .map_err(|_| CompatError::InvalidTime(s.to_string()))?;
                let minutes: f64 = mins
                    .parse()
                    .map_err(|_| CompatError::InvalidTime(s.to_string()))?;
                let seconds: f64 = secs
                    .parse()
                    .map_err(|_| CompatError::InvalidTime(s.to_string()))?;
                Ok(TimeValue(hours * 3600.0 + minutes * 60.0 + seconds))
            }
            _ => Err(CompatError::InvalidTime(s.to_string())),
        }
    }

    /// Get time in seconds.
    pub fn seconds(&self) -> f64 {
        self.0
    }

    /// Get time in milliseconds.
    pub fn milliseconds(&self) -> u64 {
        (self.0 * 1000.0) as u64
    }

    /// Get time in microseconds.
    pub fn microseconds(&self) -> i64 {
        (self.0 * 1_000_000.0) as i64
    }

    /// Convert to std::time::Duration.
    pub fn to_duration(&self) -> Duration {
        Duration::from_secs_f64(self.0)
    }
}

impl std::fmt::Display for TimeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_secs = self.0;
        let hours = (total_secs / 3600.0).floor() as u64;
        let mins = ((total_secs % 3600.0) / 60.0).floor() as u64;
        let secs = total_secs % 60.0;

        if hours > 0 {
            write!(f, "{:02}:{:02}:{:06.3}", hours, mins, secs)
        } else if mins > 0 {
            write!(f, "{}:{:06.3}", mins, secs)
        } else {
            write!(f, "{:.3}", secs)
        }
    }
}

/// Parsed resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resolution {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Resolution {
    /// Create a new resolution.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Parse a resolution string.
    ///
    /// Supported formats:
    /// - `1920x1080`
    /// - `1920X1080`
    /// - `hd720`, `hd1080`
    /// - `4k`, `2k`
    /// - `vga`, `svga`, `xga`
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_lowercase();

        // Check for named resolutions
        match s.as_str() {
            "sqcif" => return Ok(Self::new(128, 96)),
            "qcif" => return Ok(Self::new(176, 144)),
            "cif" => return Ok(Self::new(352, 288)),
            "4cif" => return Ok(Self::new(704, 576)),
            "qqvga" => return Ok(Self::new(160, 120)),
            "qvga" => return Ok(Self::new(320, 240)),
            "vga" => return Ok(Self::new(640, 480)),
            "svga" => return Ok(Self::new(800, 600)),
            "xga" => return Ok(Self::new(1024, 768)),
            "sxga" => return Ok(Self::new(1280, 1024)),
            "wxga" => return Ok(Self::new(1366, 768)),
            "wsxga" => return Ok(Self::new(1600, 1024)),
            "wuxga" => return Ok(Self::new(1920, 1200)),
            "wqxga" => return Ok(Self::new(2560, 1600)),
            "wqsxga" => return Ok(Self::new(3200, 2048)),
            "wquxga" => return Ok(Self::new(3840, 2400)),
            "hd480" => return Ok(Self::new(852, 480)),
            "hd720" | "720p" => return Ok(Self::new(1280, 720)),
            "hd1080" | "1080p" => return Ok(Self::new(1920, 1080)),
            "2k" => return Ok(Self::new(2048, 1080)),
            "4k" | "uhd" | "2160p" => return Ok(Self::new(3840, 2160)),
            "8k" | "4320p" => return Ok(Self::new(7680, 4320)),
            "ntsc" => return Ok(Self::new(720, 480)),
            "pal" => return Ok(Self::new(720, 576)),
            _ => {}
        }

        // Try to parse as WxH
        if let Some(sep_pos) = s.find('x') {
            let width: u32 = s[..sep_pos]
                .parse()
                .map_err(|_| CompatError::InvalidResolution(s.clone()))?;
            let height: u32 = s[sep_pos + 1..]
                .parse()
                .map_err(|_| CompatError::InvalidResolution(s.clone()))?;
            return Ok(Self::new(width, height));
        }

        Err(CompatError::InvalidResolution(s))
    }

    /// Calculate aspect ratio as a float.
    pub fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}

impl std::fmt::Display for Resolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Parsed aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AspectRatio {
    /// Numerator of the aspect ratio.
    pub num: u32,
    /// Denominator of the aspect ratio.
    pub den: u32,
}

impl AspectRatio {
    /// Create a new aspect ratio.
    pub fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// Parse an aspect ratio string.
    ///
    /// Supported formats:
    /// - `16:9`
    /// - `4:3`
    /// - `1.777` (float)
    /// - `1.777777` (float)
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();

        // Try ratio format (num:den)
        if let Some(colon_pos) = s.find(':') {
            let num: u32 = s[..colon_pos]
                .parse()
                .map_err(|_| CompatError::InvalidAspectRatio(s.to_string()))?;
            let den: u32 = s[colon_pos + 1..]
                .parse()
                .map_err(|_| CompatError::InvalidAspectRatio(s.to_string()))?;
            if den == 0 {
                return Err(CompatError::InvalidAspectRatio(
                    "denominator cannot be zero".to_string(),
                ));
            }
            return Ok(Self::new(num, den));
        }

        // Try float format
        let ratio: f64 = s
            .parse()
            .map_err(|_| CompatError::InvalidAspectRatio(s.to_string()))?;

        // Convert to fraction (approximate)
        // Common ratios
        if (ratio - 16.0 / 9.0).abs() < 0.01 {
            return Ok(Self::new(16, 9));
        }
        if (ratio - 4.0 / 3.0).abs() < 0.01 {
            return Ok(Self::new(4, 3));
        }
        if (ratio - 21.0 / 9.0).abs() < 0.01 {
            return Ok(Self::new(21, 9));
        }
        if (ratio - 1.85).abs() < 0.01 {
            return Ok(Self::new(185, 100));
        }
        if (ratio - 2.35).abs() < 0.01 {
            return Ok(Self::new(235, 100));
        }
        if (ratio - 2.39).abs() < 0.01 {
            return Ok(Self::new(239, 100));
        }

        // General case: multiply by 1000 and reduce
        let num = (ratio * 1000.0).round() as u32;
        let den = 1000u32;
        let g = gcd(num, den);
        Ok(Self::new(num / g, den / g))
    }

    /// Get the ratio as a float.
    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

impl std::fmt::Display for AspectRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.num, self.den)
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// A single parsed option with optional stream specifier.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedOption {
    /// The option name (without leading dash).
    pub name: String,
    /// Optional stream specifier (e.g., ":v:0").
    pub specifier: Option<StreamSpecifier>,
    /// The option value (if any).
    pub value: Option<String>,
}

impl ParsedOption {
    /// Create a new parsed option.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            specifier: None,
            value: None,
        }
    }

    /// Add a stream specifier.
    pub fn with_specifier(mut self, spec: StreamSpecifier) -> Self {
        self.specifier = Some(spec);
        self
    }

    /// Add a value.
    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    /// Parse an option string like "-c:v" or "-b:a:0".
    pub fn parse(opt: &str, value: Option<&str>) -> Result<Self> {
        let opt = opt.trim_start_matches('-');

        // Find the colon for stream specifier
        if let Some(colon_pos) = opt.find(':') {
            let name = &opt[..colon_pos];
            let spec_str = &opt[colon_pos + 1..];
            let specifier = StreamSpecifier::parse(spec_str)?;

            Ok(Self {
                name: name.to_string(),
                specifier: Some(specifier),
                value: value.map(|s| s.to_string()),
            })
        } else {
            Ok(Self {
                name: opt.to_string(),
                specifier: None,
                value: value.map(|s| s.to_string()),
            })
        }
    }

    /// Get the value as a bitrate.
    pub fn as_bitrate(&self) -> Result<Bitrate> {
        self.value
            .as_ref()
            .ok_or_else(|| CompatError::MissingValue(self.name.clone()))
            .and_then(|v| Bitrate::parse(v))
    }

    /// Get the value as a time.
    pub fn as_time(&self) -> Result<TimeValue> {
        self.value
            .as_ref()
            .ok_or_else(|| CompatError::MissingValue(self.name.clone()))
            .and_then(|v| TimeValue::parse(v))
    }

    /// Get the value as a resolution.
    pub fn as_resolution(&self) -> Result<Resolution> {
        self.value
            .as_ref()
            .ok_or_else(|| CompatError::MissingValue(self.name.clone()))
            .and_then(|v| Resolution::parse(v))
    }

    /// Get the value as an aspect ratio.
    pub fn as_aspect_ratio(&self) -> Result<AspectRatio> {
        self.value
            .as_ref()
            .ok_or_else(|| CompatError::MissingValue(self.name.clone()))
            .and_then(|v| AspectRatio::parse(v))
    }

    /// Get the value as a u32.
    pub fn as_u32(&self) -> Result<u32> {
        self.value
            .as_ref()
            .ok_or_else(|| CompatError::MissingValue(self.name.clone()))
            .and_then(|v| {
                v.parse().map_err(|_| {
                    CompatError::invalid_value(&self.name, v, "expected unsigned integer")
                })
            })
    }

    /// Get the value as an i32.
    pub fn as_i32(&self) -> Result<i32> {
        self.value
            .as_ref()
            .ok_or_else(|| CompatError::MissingValue(self.name.clone()))
            .and_then(|v| {
                v.parse()
                    .map_err(|_| CompatError::invalid_value(&self.name, v, "expected integer"))
            })
    }

    /// Get the value as a float.
    pub fn as_f64(&self) -> Result<f64> {
        self.value
            .as_ref()
            .ok_or_else(|| CompatError::MissingValue(self.name.clone()))
            .and_then(|v| {
                v.parse()
                    .map_err(|_| CompatError::invalid_value(&self.name, v, "expected number"))
            })
    }
}

/// Collection of parsed options organized by type.
#[derive(Debug, Clone, Default)]
pub struct OptionSet {
    /// Input files.
    pub inputs: Vec<String>,
    /// Output files.
    pub outputs: Vec<String>,
    /// General options (not stream-specific).
    pub general: HashMap<String, String>,
    /// Video codec options.
    pub video_codec: HashMap<StreamSpecifier, String>,
    /// Audio codec options.
    pub audio_codec: HashMap<StreamSpecifier, String>,
    /// Subtitle codec options.
    pub subtitle_codec: HashMap<StreamSpecifier, String>,
    /// Video bitrate options.
    pub video_bitrate: HashMap<StreamSpecifier, Bitrate>,
    /// Audio bitrate options.
    pub audio_bitrate: HashMap<StreamSpecifier, Bitrate>,
    /// Video filters.
    pub video_filters: Vec<String>,
    /// Audio filters.
    pub audio_filters: Vec<String>,
    /// Complex filter graph.
    pub filter_complex: Option<String>,
    /// Preset name.
    pub preset: Option<String>,
    /// CRF/quality value.
    pub crf: Option<f64>,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// Resolution.
    pub resolution: Option<Resolution>,
    /// Aspect ratio.
    pub aspect_ratio: Option<AspectRatio>,
    /// Start time.
    pub start_time: Option<TimeValue>,
    /// Duration.
    pub duration: Option<TimeValue>,
    /// End time.
    pub end_time: Option<TimeValue>,
    /// Whether to overwrite output.
    pub overwrite: bool,
    /// Stream mappings.
    pub mappings: Vec<StreamSpecifier>,
    /// Copy mode for video.
    pub copy_video: bool,
    /// Copy mode for audio.
    pub copy_audio: bool,
    /// Disable video.
    pub no_video: bool,
    /// Disable audio.
    pub no_audio: bool,
    /// Other options not specifically parsed.
    pub other: Vec<ParsedOption>,
}

impl OptionSet {
    /// Create a new empty option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the effective video codec for a stream.
    pub fn get_video_codec(&self, spec: &StreamSpecifier) -> Option<&str> {
        // Try exact match first
        if let Some(codec) = self.video_codec.get(spec) {
            return Some(codec);
        }
        // Try generic video specifier
        if let Some(codec) = self.video_codec.get(&StreamSpecifier::Video) {
            return Some(codec);
        }
        // Try all specifier
        if let Some(codec) = self.video_codec.get(&StreamSpecifier::All) {
            return Some(codec);
        }
        None
    }

    /// Get the effective audio codec for a stream.
    pub fn get_audio_codec(&self, spec: &StreamSpecifier) -> Option<&str> {
        if let Some(codec) = self.audio_codec.get(spec) {
            return Some(codec);
        }
        if let Some(codec) = self.audio_codec.get(&StreamSpecifier::Audio) {
            return Some(codec);
        }
        if let Some(codec) = self.audio_codec.get(&StreamSpecifier::All) {
            return Some(codec);
        }
        None
    }

    /// Get the effective video bitrate for a stream.
    pub fn get_video_bitrate(&self, spec: &StreamSpecifier) -> Option<Bitrate> {
        if let Some(br) = self.video_bitrate.get(spec) {
            return Some(*br);
        }
        if let Some(br) = self.video_bitrate.get(&StreamSpecifier::Video) {
            return Some(*br);
        }
        if let Some(br) = self.video_bitrate.get(&StreamSpecifier::All) {
            return Some(*br);
        }
        None
    }

    /// Get the effective audio bitrate for a stream.
    pub fn get_audio_bitrate(&self, spec: &StreamSpecifier) -> Option<Bitrate> {
        if let Some(br) = self.audio_bitrate.get(spec) {
            return Some(*br);
        }
        if let Some(br) = self.audio_bitrate.get(&StreamSpecifier::Audio) {
            return Some(*br);
        }
        if let Some(br) = self.audio_bitrate.get(&StreamSpecifier::All) {
            return Some(*br);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitrate_parse() {
        assert_eq!(Bitrate::parse("5000000").unwrap().bps(), 5_000_000);
        assert_eq!(Bitrate::parse("5000k").unwrap().bps(), 5_000_000);
        assert_eq!(Bitrate::parse("5M").unwrap().bps(), 5_000_000);
        assert_eq!(Bitrate::parse("1G").unwrap().bps(), 1_000_000_000);
        assert_eq!(Bitrate::parse("1.5M").unwrap().bps(), 1_500_000);
    }

    #[test]
    fn test_bitrate_display() {
        assert_eq!(Bitrate(5_000_000).to_string(), "5M");
        assert_eq!(Bitrate(5_000).to_string(), "5k");
        assert_eq!(Bitrate(5_000_001).to_string(), "5000001");
    }

    #[test]
    fn test_time_parse() {
        assert!((TimeValue::parse("90").unwrap().seconds() - 90.0).abs() < 0.001);
        assert!((TimeValue::parse("1:30").unwrap().seconds() - 90.0).abs() < 0.001);
        assert!((TimeValue::parse("00:01:30").unwrap().seconds() - 90.0).abs() < 0.001);
        assert!((TimeValue::parse("1:00:30").unwrap().seconds() - 3630.0).abs() < 0.001);
        assert!((TimeValue::parse("00:01:30.500").unwrap().seconds() - 90.5).abs() < 0.001);
    }

    #[test]
    fn test_resolution_parse() {
        assert_eq!(
            Resolution::parse("1920x1080").unwrap(),
            Resolution::new(1920, 1080)
        );
        assert_eq!(
            Resolution::parse("1920X1080").unwrap(),
            Resolution::new(1920, 1080)
        );
        assert_eq!(
            Resolution::parse("hd1080").unwrap(),
            Resolution::new(1920, 1080)
        );
        assert_eq!(
            Resolution::parse("4k").unwrap(),
            Resolution::new(3840, 2160)
        );
        assert_eq!(
            Resolution::parse("720p").unwrap(),
            Resolution::new(1280, 720)
        );
    }

    #[test]
    fn test_aspect_ratio_parse() {
        assert_eq!(
            AspectRatio::parse("16:9").unwrap(),
            AspectRatio::new(16, 9)
        );
        assert_eq!(AspectRatio::parse("4:3").unwrap(), AspectRatio::new(4, 3));
        // Float format - should approximate to 16:9
        let ratio = AspectRatio::parse("1.777").unwrap();
        assert!((ratio.to_f64() - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_parsed_option() {
        let opt = ParsedOption::parse("-c:v", Some("libx264")).unwrap();
        assert_eq!(opt.name, "c");
        assert_eq!(opt.specifier, Some(StreamSpecifier::Video));
        assert_eq!(opt.value, Some("libx264".to_string()));

        let opt = ParsedOption::parse("-b:a:0", Some("128k")).unwrap();
        assert_eq!(opt.name, "b");
        assert_eq!(
            opt.specifier,
            Some(StreamSpecifier::TypeIndex {
                stream_type: crate::error::StreamType::Audio,
                index: 0
            })
        );
        assert_eq!(opt.as_bitrate().unwrap().bps(), 128_000);
    }

    #[test]
    fn test_parsed_option_conversions() {
        let opt = ParsedOption::new("b").with_value("5M");
        assert_eq!(opt.as_bitrate().unwrap().bps(), 5_000_000);

        let opt = ParsedOption::new("t").with_value("1:30");
        assert!((opt.as_time().unwrap().seconds() - 90.0).abs() < 0.001);

        let opt = ParsedOption::new("s").with_value("1920x1080");
        let res = opt.as_resolution().unwrap();
        assert_eq!(res.width, 1920);
        assert_eq!(res.height, 1080);
    }
}
