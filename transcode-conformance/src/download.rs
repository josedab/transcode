//! Test stream download functionality
//!
//! Handles downloading ITU-T H.264 conformance test streams from various sources.

use crate::{ConformanceError, Result, TestStream};
use std::path::PathBuf;

#[cfg(feature = "download")]
use tracing::{debug, info, warn};

/// Known ITU-T test stream sources
pub mod sources {
    /// Base URL for ITU-T conformance streams (placeholder)
    pub const ITU_T_BASE: &str = "https://www.itu.int/wftp3/av-arch/jvt-site/draft_conformance";

    /// Alternative mirror sources
    pub const MIRRORS: &[&str] = &[
        // These are placeholder URLs - actual ITU-T streams require registration
        "https://ftp.example.com/h264-conformance",
    ];
}

/// Stream downloader
pub struct StreamDownloader {
    #[cfg(feature = "download")]
    client: reqwest::blocking::Client,
    /// Maximum retries per download
    max_retries: u32,
    /// Timeout in seconds
    timeout_secs: u64,
}

impl Default for StreamDownloader {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDownloader {
    /// Create a new downloader
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "download")]
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
            max_retries: 3,
            timeout_secs: 300,
        }
    }

    /// Set maximum retries
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set timeout in seconds
    pub fn timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Download a test stream
    #[cfg(feature = "download")]
    pub fn download(&self, stream: &TestStream, dest_dir: &std::path::Path) -> Result<PathBuf> {
        let url = stream.source_url.as_ref().ok_or_else(|| {
            ConformanceError::Configuration {
                message: format!("No source URL for stream {}", stream.id),
            }
        })?;

        let dest_path = dest_dir.join(&stream.id).with_extension("264");

        info!("Downloading {} from {}", stream.id, url);

        let mut last_error = None;
        for attempt in 1..=self.max_retries {
            match self.download_with_retry(url, &dest_path) {
                Ok(_) => {
                    info!("Successfully downloaded {} to {:?}", stream.id, dest_path);
                    return Ok(dest_path);
                }
                Err(e) => {
                    warn!(
                        "Download attempt {}/{} failed for {}: {}",
                        attempt, self.max_retries, stream.id, e
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| ConformanceError::DownloadFailed {
            url: url.clone(),
        }))
    }

    #[cfg(feature = "download")]
    fn download_with_retry(
        &self,
        url: &str,
        dest_path: &std::path::Path,
    ) -> Result<()> {
        use std::io::Write;

        let response = self.client.get(url).send().map_err(|e| {
            ConformanceError::DownloadFailed {
                url: format!("{}: {}", url, e),
            }
        })?;

        if !response.status().is_success() {
            return Err(ConformanceError::DownloadFailed {
                url: format!("{}: HTTP {}", url, response.status()),
            });
        }

        let bytes = response.bytes().map_err(|e| {
            ConformanceError::DownloadFailed {
                url: format!("{}: {}", url, e),
            }
        })?;

        // Create parent directories if needed
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = std::fs::File::create(dest_path)?;
        file.write_all(&bytes)?;

        debug!("Wrote {} bytes to {:?}", bytes.len(), dest_path);
        Ok(())
    }

    /// Download a test stream (stub when download feature is disabled)
    #[cfg(not(feature = "download"))]
    pub fn download(&self, stream: &TestStream, _dest_dir: &std::path::Path) -> Result<PathBuf> {
        Err(ConformanceError::Configuration {
            message: format!(
                "Download feature not enabled. Enable with --features download or provide local path for {}",
                stream.id
            ),
        })
    }

    /// Check if a URL is accessible
    #[cfg(feature = "download")]
    pub fn check_url(&self, url: &str) -> bool {
        match self.client.head(url).send() {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }

    #[cfg(not(feature = "download"))]
    pub fn check_url(&self, _url: &str) -> bool {
        false
    }
}

/// Download progress callback
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Batch download result
#[derive(Debug)]
pub struct BatchDownloadResult {
    /// Successfully downloaded streams
    pub succeeded: Vec<String>,
    /// Failed downloads with error messages
    pub failed: Vec<(String, String)>,
    /// Skipped (already cached)
    pub skipped: Vec<String>,
}

impl BatchDownloadResult {
    /// Check if all downloads succeeded
    pub fn all_succeeded(&self) -> bool {
        self.failed.is_empty()
    }

    /// Get total number of streams processed
    pub fn total(&self) -> usize {
        self.succeeded.len() + self.failed.len() + self.skipped.len()
    }
}

/// Download multiple streams
pub fn download_batch(
    downloader: &StreamDownloader,
    streams: &[TestStream],
    dest_dir: &std::path::Path,
    skip_existing: bool,
) -> BatchDownloadResult {
    let mut result = BatchDownloadResult {
        succeeded: Vec::new(),
        failed: Vec::new(),
        skipped: Vec::new(),
    };

    for stream in streams {
        let dest_path = dest_dir.join(&stream.id).with_extension("264");

        if skip_existing && dest_path.exists() {
            result.skipped.push(stream.id.clone());
            continue;
        }

        match downloader.download(stream, dest_dir) {
            Ok(_) => result.succeeded.push(stream.id.clone()),
            Err(e) => result.failed.push((stream.id.clone(), e.to_string())),
        }
    }

    result
}

/// ITU-T conformance stream index
/// This provides metadata about known conformance streams
pub struct ConformanceStreamIndex {
    streams: Vec<TestStream>,
}

impl Default for ConformanceStreamIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl ConformanceStreamIndex {
    /// Create a new index with known ITU-T streams
    pub fn new() -> Self {
        use crate::{H264Level, H264Profile};

        let streams = vec![
            // Baseline Profile streams
            TestStream::builder("BA1_Sony_D")
                .name("BA1_Sony_D - Basic I/P slice")
                .description("Basic I and P slice decoding for Baseline profile")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(176, 144)
                .expected_frames(17)
                .category("baseline")
                .category("basic")
                .build(),
            TestStream::builder("BA2_Sony_F")
                .name("BA2_Sony_F - Multiple slices")
                .description("Multiple slices in single frame")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level21)
                .resolution(176, 144)
                .expected_frames(300)
                .category("baseline")
                .category("slices")
                .build(),
            TestStream::builder("CVBS3_Sony_C")
                .name("CVBS3_Sony_C - CAVLC basic")
                .description("CAVLC entropy coding basic test")
                .profile(H264Profile::Baseline)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .category("baseline")
                .category("cavlc")
                .build(),

            // Main Profile streams
            TestStream::builder("CAPM3_Sony_D")
                .name("CAPM3_Sony_D - Main Profile CABAC")
                .description("CABAC entropy coding for Main profile")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 480)
                .category("main")
                .category("cabac")
                .build(),
            TestStream::builder("CAMP_MOT_MBAFF_L30")
                .name("CAMP_MOT_MBAFF_L30 - MBAFF")
                .description("Macroblock-Adaptive Frame-Field coding")
                .profile(H264Profile::Main)
                .level(H264Level::Level3)
                .resolution(720, 576)
                .category("main")
                .category("mbaff")
                .build(),
            TestStream::builder("CAWP1_TOSHIBA_E")
                .name("CAWP1_TOSHIBA_E - Weighted prediction")
                .description("Weighted prediction test")
                .profile(H264Profile::Main)
                .level(H264Level::Level21)
                .resolution(352, 288)
                .category("main")
                .category("weighted_pred")
                .build(),

            // High Profile streams
            TestStream::builder("CAPH_HP_B")
                .name("CAPH_HP_B - High Profile basic")
                .description("Basic High profile decoding test")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1920, 1080)
                .category("high")
                .category("basic")
                .build(),
            TestStream::builder("CAH1_Sony_B")
                .name("CAH1_Sony_B - 8x8 transform")
                .description("8x8 integer transform for High profile")
                .profile(H264Profile::High)
                .level(H264Level::Level4)
                .resolution(1280, 720)
                .category("high")
                .category("transform")
                .build(),
            TestStream::builder("CVHP_Toshiba_B")
                .name("CVHP_Toshiba_B - High Profile CAVLC")
                .description("CAVLC with High profile features")
                .profile(H264Profile::High)
                .level(H264Level::Level41)
                .resolution(1920, 1080)
                .category("high")
                .category("cavlc")
                .build(),
        ];

        Self { streams }
    }

    /// Get all streams
    pub fn all(&self) -> &[TestStream] {
        &self.streams
    }

    /// Get streams by profile
    pub fn by_profile(&self, profile: crate::H264Profile) -> Vec<&TestStream> {
        self.streams
            .iter()
            .filter(|s| s.profile == profile)
            .collect()
    }

    /// Get streams by category
    pub fn by_category(&self, category: &str) -> Vec<&TestStream> {
        self.streams
            .iter()
            .filter(|s| s.categories.iter().any(|c| c == category))
            .collect()
    }

    /// Get a specific stream by ID
    pub fn get(&self, id: &str) -> Option<&TestStream> {
        self.streams.iter().find(|s| s.id == id)
    }

    /// Add a custom stream to the index
    pub fn add(&mut self, stream: TestStream) {
        self.streams.push(stream);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downloader_creation() {
        let downloader = StreamDownloader::new().max_retries(5).timeout(600);
        assert_eq!(downloader.max_retries, 5);
        assert_eq!(downloader.timeout_secs, 600);
    }

    #[test]
    fn test_stream_index() {
        let index = ConformanceStreamIndex::new();
        assert!(!index.all().is_empty());

        let baseline = index.by_profile(crate::H264Profile::Baseline);
        assert!(!baseline.is_empty());

        let main_streams = index.by_profile(crate::H264Profile::Main);
        assert!(!main_streams.is_empty());

        let high = index.by_profile(crate::H264Profile::High);
        assert!(!high.is_empty());
    }

    #[test]
    fn test_stream_by_category() {
        let index = ConformanceStreamIndex::new();
        let cabac = index.by_category("cabac");
        assert!(!cabac.is_empty());
    }

    #[test]
    fn test_batch_download_result() {
        let result = BatchDownloadResult {
            succeeded: vec!["stream1".to_string()],
            failed: vec![("stream2".to_string(), "error".to_string())],
            skipped: vec!["stream3".to_string()],
        };

        assert!(!result.all_succeeded());
        assert_eq!(result.total(), 3);
    }
}
