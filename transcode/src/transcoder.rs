//! High-level transcoder API.

use crate::options::TranscodeOptions;
use crate::presets::EncodingProfile;
use crate::Result;
use std::path::Path;
use tracing::{debug, info};

/// Progress callback type.
pub type ProgressCallback = Box<dyn Fn(f64, u64) + Send>;

/// Transcoding statistics.
#[derive(Debug, Clone, Default)]
pub struct TranscodeStats {
    /// Total packets processed.
    pub packets_processed: u64,
    /// Total frames decoded.
    pub frames_decoded: u64,
    /// Total frames encoded.
    pub frames_encoded: u64,
    /// Duration processed in microseconds.
    pub duration_processed_us: i64,
    /// Total input duration in microseconds.
    pub total_duration_us: Option<i64>,
    /// Input file size in bytes.
    pub input_size: u64,
    /// Output file size in bytes.
    pub output_size: u64,
    /// Encoding speed (real-time multiplier).
    pub speed: f64,
    /// Average video bitrate.
    pub avg_video_bitrate: u64,
    /// Average audio bitrate.
    pub avg_audio_bitrate: u64,
}

impl TranscodeStats {
    /// Get progress as percentage (0.0 - 100.0).
    pub fn progress(&self) -> f64 {
        if let Some(total) = self.total_duration_us {
            if total > 0 {
                return (self.duration_processed_us as f64 / total as f64 * 100.0).min(100.0);
            }
        }
        0.0
    }

    /// Get compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.input_size > 0 && self.output_size > 0 {
            self.input_size as f64 / self.output_size as f64
        } else {
            1.0
        }
    }
}

/// High-level transcoder that orchestrates the transcoding process.
pub struct Transcoder {
    /// Transcoding options.
    options: TranscodeOptions,
    /// Progress callback.
    progress_callback: Option<ProgressCallback>,
    /// Transcoding statistics.
    stats: TranscodeStats,
    /// Whether transcoding has been initialized.
    initialized: bool,
}

impl Transcoder {
    /// Create a new transcoder with the given options.
    pub fn new(options: TranscodeOptions) -> Result<Self> {
        options.validate().map_err(|e| {
            transcode_core::Error::Config(e)
        })?;

        Ok(Self {
            options,
            progress_callback: None,
            stats: TranscodeStats::default(),
            initialized: false,
        })
    }

    /// Create a transcoder from an encoding profile.
    pub fn from_profile(
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
        profile: EncodingProfile,
    ) -> Result<Self> {
        let mut options = TranscodeOptions::new()
            .input(input.as_ref())
            .output(output.as_ref());

        if let Some(video) = profile.video {
            options = options.video_config(video);
        }

        options = options.audio_config(profile.audio);

        Self::new(options)
    }

    /// Set progress callback.
    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(f64, u64) + Send + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Get current statistics.
    pub fn stats(&self) -> &TranscodeStats {
        &self.stats
    }

    /// Get the options.
    pub fn options(&self) -> &TranscodeOptions {
        &self.options
    }

    /// Initialize the transcoder (probe input, setup pipeline).
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        let input_path = self.options.input.as_ref()
            .map(|c| &c.path)
            .ok_or_else(|| transcode_core::Error::Config("No input specified".into()))?;

        info!("Initializing transcoder for: {:?}", input_path);

        // Get input file size
        if let Ok(metadata) = std::fs::metadata(input_path) {
            self.stats.input_size = metadata.len();
        }

        self.initialized = true;
        debug!("Transcoder initialized");

        Ok(())
    }

    /// Run the transcoding process.
    pub fn run(&mut self) -> Result<()> {
        self.initialize()?;

        let input_path = self.options.input.as_ref()
            .map(|c| &c.path)
            .ok_or_else(|| transcode_core::Error::Config("No input specified".into()))?;

        let output_path = self.options.output.as_ref()
            .map(|c| c.path.clone())
            .ok_or_else(|| transcode_core::Error::Config("No output specified".into()))?;

        info!("Starting transcode: {:?} -> {:?}", input_path, output_path);

        // Check if output exists and overwrite is disabled
        if output_path.exists() && !self.options.overwrite {
            return Err(transcode_core::Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("Output file already exists: {:?}", output_path),
            )));
        }

        // Create output directory if needed
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Main transcoding loop simulation
        // In a real implementation, this would use the Pipeline
        self.simulate_transcode()?;

        // Get output file size
        if let Ok(metadata) = std::fs::metadata(&output_path) {
            self.stats.output_size = metadata.len();
        }

        info!(
            "Transcode complete: {} packets, {:.1}% compression",
            self.stats.packets_processed,
            (1.0 - 1.0 / self.stats.compression_ratio()) * 100.0
        );

        Ok(())
    }

    /// Simulate transcoding (placeholder for actual pipeline integration).
    fn simulate_transcode(&mut self) -> Result<()> {
        // This is a placeholder that simulates the transcoding process
        // In a real implementation, this would:
        // 1. Create demuxer from input file
        // 2. Create decoders based on input streams
        // 3. Create encoders based on output options
        // 4. Create muxer for output file
        // 5. Build and run the pipeline

        let total_steps = 100;
        self.stats.total_duration_us = Some(total_steps as i64 * 1_000_000);

        for i in 0..total_steps {
            self.stats.packets_processed += 10;
            self.stats.frames_decoded += 1;
            self.stats.frames_encoded += 1;
            self.stats.duration_processed_us = (i + 1) as i64 * 1_000_000;

            // Report progress
            if let Some(ref callback) = self.progress_callback {
                let progress = self.stats.progress();
                callback(progress, self.stats.packets_processed);
            }
        }

        // Create empty output file as placeholder
        let output_path = self.options.output.as_ref()
            .map(|c| &c.path)
            .ok_or_else(|| transcode_core::Error::Config("No output specified".into()))?;

        std::fs::write(output_path, b"")?;

        Ok(())
    }

    /// Get current progress (0.0 - 100.0).
    pub fn progress(&self) -> f64 {
        self.stats.progress()
    }

    /// Check if transcoding is complete.
    pub fn is_complete(&self) -> bool {
        self.stats.progress() >= 100.0
    }
}

/// Builder for creating transcoder with fluent API.
#[allow(dead_code)]
pub struct TranscoderBuilder {
    options: TranscodeOptions,
    progress_callback: Option<ProgressCallback>,
}

impl Default for TranscoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl TranscoderBuilder {
    /// Create a new transcoder builder.
    pub fn new() -> Self {
        Self {
            options: TranscodeOptions::new(),
            progress_callback: None,
        }
    }

    /// Set input file.
    pub fn input(mut self, path: impl AsRef<Path>) -> Self {
        self.options = self.options.input(path.as_ref());
        self
    }

    /// Set output file.
    pub fn output(mut self, path: impl AsRef<Path>) -> Self {
        self.options = self.options.output(path.as_ref());
        self
    }

    /// Set video codec.
    pub fn video_codec(mut self, codec: impl Into<String>) -> Self {
        self.options = self.options.video_codec(codec);
        self
    }

    /// Set audio codec.
    pub fn audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.options = self.options.audio_codec(codec);
        self
    }

    /// Set video resolution.
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.options = self.options.video_resolution(width, height);
        self
    }

    /// Set video bitrate.
    pub fn video_bitrate(mut self, bitrate: u64) -> Self {
        self.options = self.options.video_bitrate(bitrate);
        self
    }

    /// Set audio bitrate.
    pub fn audio_bitrate(mut self, bitrate: u64) -> Self {
        self.options = self.options.audio_bitrate(bitrate);
        self
    }

    /// Set number of threads.
    pub fn threads(mut self, threads: usize) -> Self {
        self.options = self.options.threads(threads);
        self
    }

    /// Enable hardware acceleration.
    pub fn hardware_acceleration(mut self, enable: bool) -> Self {
        self.options = self.options.hardware_acceleration(enable);
        self
    }

    /// Enable overwrite mode.
    pub fn overwrite(mut self, enable: bool) -> Self {
        self.options = self.options.overwrite(enable);
        self
    }

    /// Set progress callback.
    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(f64, u64) + Send + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Build the transcoder.
    pub fn build(self) -> Result<Transcoder> {
        let mut transcoder = Transcoder::new(self.options)?;
        transcoder.progress_callback = self.progress_callback;
        Ok(transcoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_progress() {
        let mut stats = TranscodeStats::default();
        stats.total_duration_us = Some(1_000_000);
        stats.duration_processed_us = 500_000;
        assert!((stats.progress() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_compression_ratio() {
        let mut stats = TranscodeStats::default();
        stats.input_size = 1000;
        stats.output_size = 500;
        assert!((stats.compression_ratio() - 2.0).abs() < 0.001);
    }
}
