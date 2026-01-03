//! # Transcode Conformance Test Suite
//!
//! This crate provides H.264 conformance testing infrastructure for the Transcode
//! codec library. It supports testing against ITU-T H.264 reference test streams
//! and validates decoder/encoder output against known-good reference values.
//!
//! ## Features
//!
//! - **Test Infrastructure**: Framework for running conformance tests
//! - **Stream Management**: Download and cache ITU-T test streams or use local paths
//! - **Profile Testing**: Baseline, Main, and High profile conformance tests
//! - **Bitstream Validation**: Verify NAL unit structure and syntax elements
//! - **Frame Comparison**: MD5 checksum verification of decoded frames
//! - **Report Generation**: Generate detailed conformance reports
//!
//! ## Usage
//!
//! Tests requiring external files are marked with `#[ignore]` and can be run with:
//!
//! ```bash
//! cargo test -p transcode-conformance --ignored
//! ```
//!
//! To run with download capability:
//!
//! ```bash
//! cargo test -p transcode-conformance --features download --ignored
//! ```

pub mod cache;
pub mod checksum;
pub mod download;
pub mod profiles;
pub mod report;
pub mod runner;
pub mod stream;
pub mod validation;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Conformance test error types
#[derive(Error, Debug)]
pub enum ConformanceError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Test stream not found: {path}")]
    StreamNotFound { path: PathBuf },

    #[error("Download failed: {url}")]
    DownloadFailed { url: String },

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Bitstream validation failed: {message}")]
    BitstreamValidation { message: String },

    #[error("Decoding failed: {message}")]
    DecodingFailed { message: String },

    #[error("Frame count mismatch: expected {expected}, got {actual}")]
    FrameCountMismatch { expected: usize, actual: usize },

    #[error("Profile not supported: {profile}")]
    UnsupportedProfile { profile: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("Reference data missing: {path}")]
    ReferenceMissing { path: PathBuf },
}

/// Result type for conformance operations
pub type Result<T> = std::result::Result<T, ConformanceError>;

/// H.264 profile identifiers for conformance testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum H264Profile {
    /// Baseline Profile (profile_idc = 66)
    Baseline,
    /// Constrained Baseline Profile
    ConstrainedBaseline,
    /// Main Profile (profile_idc = 77)
    Main,
    /// Extended Profile (profile_idc = 88)
    Extended,
    /// High Profile (profile_idc = 100)
    High,
    /// High 10 Profile
    High10,
    /// High 4:2:2 Profile
    High422,
    /// High 4:4:4 Predictive Profile
    High444,
}

impl H264Profile {
    /// Get the profile_idc value
    pub fn profile_idc(&self) -> u8 {
        match self {
            H264Profile::Baseline | H264Profile::ConstrainedBaseline => 66,
            H264Profile::Main => 77,
            H264Profile::Extended => 88,
            H264Profile::High => 100,
            H264Profile::High10 => 110,
            H264Profile::High422 => 122,
            H264Profile::High444 => 244,
        }
    }

    /// Get human-readable profile name
    pub fn name(&self) -> &'static str {
        match self {
            H264Profile::Baseline => "Baseline",
            H264Profile::ConstrainedBaseline => "Constrained Baseline",
            H264Profile::Main => "Main",
            H264Profile::Extended => "Extended",
            H264Profile::High => "High",
            H264Profile::High10 => "High 10",
            H264Profile::High422 => "High 4:2:2",
            H264Profile::High444 => "High 4:4:4 Predictive",
        }
    }
}

/// H.264 level identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum H264Level {
    Level1,
    Level1b,
    Level11,
    Level12,
    Level13,
    Level2,
    Level21,
    Level22,
    Level3,
    Level31,
    Level32,
    Level4,
    Level41,
    Level42,
    Level5,
    Level51,
    Level52,
    Level6,
    Level61,
    Level62,
}

impl H264Level {
    /// Get the level_idc value
    pub fn level_idc(&self) -> u8 {
        match self {
            H264Level::Level1 => 10,
            H264Level::Level1b => 11,
            H264Level::Level11 => 11,
            H264Level::Level12 => 12,
            H264Level::Level13 => 13,
            H264Level::Level2 => 20,
            H264Level::Level21 => 21,
            H264Level::Level22 => 22,
            H264Level::Level3 => 30,
            H264Level::Level31 => 31,
            H264Level::Level32 => 32,
            H264Level::Level4 => 40,
            H264Level::Level41 => 41,
            H264Level::Level42 => 42,
            H264Level::Level5 => 50,
            H264Level::Level51 => 51,
            H264Level::Level52 => 52,
            H264Level::Level6 => 60,
            H264Level::Level61 => 61,
            H264Level::Level62 => 62,
        }
    }

    /// Get human-readable level name
    pub fn name(&self) -> &'static str {
        match self {
            H264Level::Level1 => "1",
            H264Level::Level1b => "1b",
            H264Level::Level11 => "1.1",
            H264Level::Level12 => "1.2",
            H264Level::Level13 => "1.3",
            H264Level::Level2 => "2",
            H264Level::Level21 => "2.1",
            H264Level::Level22 => "2.2",
            H264Level::Level3 => "3",
            H264Level::Level31 => "3.1",
            H264Level::Level32 => "3.2",
            H264Level::Level4 => "4",
            H264Level::Level41 => "4.1",
            H264Level::Level42 => "4.2",
            H264Level::Level5 => "5",
            H264Level::Level51 => "5.1",
            H264Level::Level52 => "5.2",
            H264Level::Level6 => "6",
            H264Level::Level61 => "6.1",
            H264Level::Level62 => "6.2",
        }
    }
}

/// Test stream metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStream {
    /// Unique identifier for the test stream
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of what this test validates
    pub description: String,
    /// H.264 profile
    pub profile: H264Profile,
    /// H.264 level
    pub level: Option<H264Level>,
    /// Source URL for downloading (ITU-T or other source)
    pub source_url: Option<String>,
    /// Local file path (if available)
    pub local_path: Option<PathBuf>,
    /// Expected frame count
    pub expected_frame_count: Option<usize>,
    /// Expected MD5 checksums for decoded frames
    pub frame_checksums: Vec<String>,
    /// Expected bitstream checksum
    pub bitstream_checksum: Option<String>,
    /// Resolution (width x height)
    pub resolution: Option<(u32, u32)>,
    /// Frame rate (numerator, denominator)
    pub frame_rate: Option<(u32, u32)>,
    /// Test categories
    pub categories: Vec<String>,
}

impl TestStream {
    /// Create a new test stream builder
    pub fn builder(id: &str) -> TestStreamBuilder {
        TestStreamBuilder::new(id)
    }

    /// Check if the stream is available (either locally or for download)
    pub fn is_available(&self) -> bool {
        self.local_path
            .as_ref()
            .map(|p| p.exists())
            .unwrap_or(false)
            || self.source_url.is_some()
    }

    /// Get the effective path (local or cached)
    pub fn effective_path(&self, cache_dir: &std::path::Path) -> PathBuf {
        if let Some(local) = &self.local_path {
            if local.exists() {
                return local.clone();
            }
        }
        cache_dir.join(&self.id).with_extension("264")
    }
}

/// Builder for TestStream
pub struct TestStreamBuilder {
    stream: TestStream,
}

impl TestStreamBuilder {
    pub fn new(id: &str) -> Self {
        Self {
            stream: TestStream {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
                profile: H264Profile::Baseline,
                level: None,
                source_url: None,
                local_path: None,
                expected_frame_count: None,
                frame_checksums: Vec::new(),
                bitstream_checksum: None,
                resolution: None,
                frame_rate: None,
                categories: Vec::new(),
            },
        }
    }

    pub fn name(mut self, name: &str) -> Self {
        self.stream.name = name.to_string();
        self
    }

    pub fn description(mut self, desc: &str) -> Self {
        self.stream.description = desc.to_string();
        self
    }

    pub fn profile(mut self, profile: H264Profile) -> Self {
        self.stream.profile = profile;
        self
    }

    pub fn level(mut self, level: H264Level) -> Self {
        self.stream.level = Some(level);
        self
    }

    pub fn source_url(mut self, url: &str) -> Self {
        self.stream.source_url = Some(url.to_string());
        self
    }

    pub fn local_path(mut self, path: PathBuf) -> Self {
        self.stream.local_path = Some(path);
        self
    }

    pub fn expected_frames(mut self, count: usize) -> Self {
        self.stream.expected_frame_count = Some(count);
        self
    }

    pub fn frame_checksum(mut self, checksum: &str) -> Self {
        self.stream.frame_checksums.push(checksum.to_string());
        self
    }

    pub fn frame_checksums(mut self, checksums: Vec<String>) -> Self {
        self.stream.frame_checksums = checksums;
        self
    }

    pub fn bitstream_checksum(mut self, checksum: &str) -> Self {
        self.stream.bitstream_checksum = Some(checksum.to_string());
        self
    }

    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.stream.resolution = Some((width, height));
        self
    }

    pub fn frame_rate(mut self, num: u32, den: u32) -> Self {
        self.stream.frame_rate = Some((num, den));
        self
    }

    pub fn category(mut self, cat: &str) -> Self {
        self.stream.categories.push(cat.to_string());
        self
    }

    pub fn categories(mut self, cats: Vec<&str>) -> Self {
        self.stream.categories = cats.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn build(self) -> TestStream {
        self.stream
    }
}

/// Test result status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestStatus {
    /// Test passed
    Passed,
    /// Test failed
    Failed,
    /// Test was skipped
    Skipped,
    /// Test encountered an error
    Error,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test stream ID
    pub stream_id: String,
    /// Test name
    pub test_name: String,
    /// Test status
    pub status: TestStatus,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Decoded frame count
    pub decoded_frames: Option<usize>,
    /// Checksum verification results
    pub checksum_results: Vec<ChecksumResult>,
    /// Additional notes
    pub notes: Vec<String>,
}

/// Checksum verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecksumResult {
    /// Frame index (or "bitstream" for bitstream checksum)
    pub target: String,
    /// Expected checksum
    pub expected: String,
    /// Actual checksum
    pub actual: String,
    /// Whether checksums match
    pub matches: bool,
}

/// Conformance test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceConfig {
    /// Directory for caching downloaded test streams
    pub cache_dir: PathBuf,
    /// Directory containing local test streams
    pub local_streams_dir: Option<PathBuf>,
    /// Whether to download missing streams
    pub allow_download: bool,
    /// Profiles to test
    pub profiles: Vec<H264Profile>,
    /// Whether to generate detailed reports
    pub generate_reports: bool,
    /// Report output directory
    pub report_dir: Option<PathBuf>,
    /// Maximum parallel tests
    pub max_parallel: usize,
    /// Timeout per test in seconds
    pub timeout_secs: u64,
}

impl Default for ConformanceConfig {
    fn default() -> Self {
        Self {
            cache_dir: std::env::temp_dir().join("transcode-conformance"),
            local_streams_dir: None,
            allow_download: false,
            profiles: vec![H264Profile::Baseline, H264Profile::Main, H264Profile::High],
            generate_reports: true,
            report_dir: None,
            max_parallel: 4,
            timeout_secs: 300,
        }
    }
}

impl ConformanceConfig {
    /// Create config for baseline profile only
    pub fn baseline_only() -> Self {
        Self {
            profiles: vec![H264Profile::Baseline],
            ..Default::default()
        }
    }

    /// Create config for main profile only
    pub fn main_only() -> Self {
        Self {
            profiles: vec![H264Profile::Main],
            ..Default::default()
        }
    }

    /// Create config for high profile only
    pub fn high_only() -> Self {
        Self {
            profiles: vec![H264Profile::High],
            ..Default::default()
        }
    }

    /// Enable downloading of test streams
    pub fn with_download(mut self, allow: bool) -> Self {
        self.allow_download = allow;
        self
    }

    /// Set cache directory
    pub fn with_cache_dir(mut self, dir: PathBuf) -> Self {
        self.cache_dir = dir;
        self
    }

    /// Set local streams directory
    pub fn with_local_streams(mut self, dir: PathBuf) -> Self {
        self.local_streams_dir = Some(dir);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_idc() {
        assert_eq!(H264Profile::Baseline.profile_idc(), 66);
        assert_eq!(H264Profile::Main.profile_idc(), 77);
        assert_eq!(H264Profile::High.profile_idc(), 100);
    }

    #[test]
    fn test_level_idc() {
        assert_eq!(H264Level::Level1.level_idc(), 10);
        assert_eq!(H264Level::Level31.level_idc(), 31);
        assert_eq!(H264Level::Level51.level_idc(), 51);
    }

    #[test]
    fn test_stream_builder() {
        let stream = TestStream::builder("test_001")
            .name("Basic Intra Test")
            .description("Tests basic I-frame decoding")
            .profile(H264Profile::Baseline)
            .level(H264Level::Level21)
            .resolution(352, 288)
            .expected_frames(10)
            .build();

        assert_eq!(stream.id, "test_001");
        assert_eq!(stream.name, "Basic Intra Test");
        assert_eq!(stream.profile, H264Profile::Baseline);
        assert_eq!(stream.resolution, Some((352, 288)));
    }

    #[test]
    fn test_config_profiles() {
        let config = ConformanceConfig::baseline_only();
        assert_eq!(config.profiles, vec![H264Profile::Baseline]);

        let config = ConformanceConfig::main_only();
        assert_eq!(config.profiles, vec![H264Profile::Main]);
    }
}
