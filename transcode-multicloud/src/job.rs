//! Job definitions for multi-cloud transcoding.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// A transcoding job to be executed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeJob {
    /// Unique job identifier.
    pub id: String,
    /// Input source URI.
    pub input: String,
    /// Output configuration.
    pub output: OutputConfig,
    /// Encoding preset.
    pub preset: Preset,
    /// Job priority (higher = more urgent).
    pub priority: u32,
    /// Maximum cost in cents.
    pub max_cost_cents: Option<u32>,
    /// Deadline for completion.
    pub deadline: Option<DateTime<Utc>>,
    /// Custom metadata.
    pub metadata: HashMap<String, String>,
    /// Created timestamp.
    pub created_at: DateTime<Utc>,
}

impl TranscodeJob {
    /// Create a new job builder.
    pub fn builder() -> TranscodeJobBuilder {
        TranscodeJobBuilder::default()
    }

    /// Create a new job with minimal configuration.
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            input: input.into(),
            output: OutputConfig {
                uri: output.into(),
                format: OutputFormat::Mp4,
                video: None,
                audio: None,
            },
            preset: Preset::default(),
            priority: 50,
            max_cost_cents: None,
            deadline: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// Set the output destination.
    pub fn output(mut self, uri: impl Into<String>) -> Self {
        self.output.uri = uri.into();
        self
    }

    /// Set the input source.
    pub fn input(mut self, uri: impl Into<String>) -> Self {
        self.input = uri.into();
        self
    }

    /// Set the encoding preset.
    pub fn preset(mut self, preset: impl Into<Preset>) -> Self {
        self.preset = preset.into();
        self
    }

    /// Set job priority.
    pub fn priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set maximum cost.
    pub fn max_cost(mut self, cents: u32) -> Self {
        self.max_cost_cents = Some(cents);
        self
    }

    /// Set completion deadline.
    pub fn deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Builder for TranscodeJob.
#[derive(Default)]
pub struct TranscodeJobBuilder {
    input: Option<String>,
    output: Option<OutputConfig>,
    preset: Option<Preset>,
    priority: Option<u32>,
    max_cost_cents: Option<u32>,
    deadline: Option<DateTime<Utc>>,
    metadata: HashMap<String, String>,
}

impl TranscodeJobBuilder {
    /// Set input URI.
    pub fn input(mut self, input: impl Into<String>) -> Self {
        self.input = Some(input.into());
        self
    }

    /// Set output configuration.
    pub fn output(mut self, output: OutputConfig) -> Self {
        self.output = Some(output);
        self
    }

    /// Set output URI with default format.
    pub fn output_uri(mut self, uri: impl Into<String>) -> Self {
        self.output = Some(OutputConfig {
            uri: uri.into(),
            format: OutputFormat::Mp4,
            video: None,
            audio: None,
        });
        self
    }

    /// Set encoding preset.
    pub fn preset(mut self, preset: Preset) -> Self {
        self.preset = Some(preset);
        self
    }

    /// Set priority.
    pub fn priority(mut self, priority: u32) -> Self {
        self.priority = Some(priority);
        self
    }

    /// Set maximum cost.
    pub fn max_cost(mut self, cents: u32) -> Self {
        self.max_cost_cents = Some(cents);
        self
    }

    /// Set deadline.
    pub fn deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the job.
    pub fn build(self) -> crate::Result<TranscodeJob> {
        let input = self.input.ok_or_else(|| {
            crate::MultiCloudError::InvalidInput("Input URI is required".into())
        })?;
        let output = self.output.ok_or_else(|| {
            crate::MultiCloudError::InvalidInput("Output configuration is required".into())
        })?;

        Ok(TranscodeJob {
            id: uuid::Uuid::new_v4().to_string(),
            input,
            output,
            preset: self.preset.unwrap_or_default(),
            priority: self.priority.unwrap_or(50),
            max_cost_cents: self.max_cost_cents,
            deadline: self.deadline,
            metadata: self.metadata,
            created_at: Utc::now(),
        })
    }
}

/// Output configuration for a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output URI (s3://, gs://, azure://, or local path).
    pub uri: String,
    /// Output container format.
    pub format: OutputFormat,
    /// Video encoding settings.
    pub video: Option<VideoOutput>,
    /// Audio encoding settings.
    pub audio: Option<AudioOutput>,
}

impl OutputConfig {
    /// Create a new output configuration.
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            format: OutputFormat::Mp4,
            video: None,
            audio: None,
        }
    }

    /// Set output format.
    pub fn format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    /// Set video output settings.
    pub fn video(mut self, video: VideoOutput) -> Self {
        self.video = Some(video);
        self
    }

    /// Set audio output settings.
    pub fn audio(mut self, audio: AudioOutput) -> Self {
        self.audio = Some(audio);
        self
    }
}

/// Output container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Mp4,
    Hls,
    Dash,
    Webm,
    Mkv,
    Mov,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Mp4
    }
}

/// Video output settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoOutput {
    /// Video codec.
    pub codec: VideoCodec,
    /// Target bitrate in kbps.
    pub bitrate_kbps: Option<u32>,
    /// Output width.
    pub width: Option<u32>,
    /// Output height.
    pub height: Option<u32>,
    /// Frame rate.
    pub framerate: Option<f64>,
    /// Quality preset.
    pub quality: Option<QualityPreset>,
}

impl Default for VideoOutput {
    fn default() -> Self {
        Self {
            codec: VideoCodec::H264,
            bitrate_kbps: None,
            width: None,
            height: None,
            framerate: None,
            quality: Some(QualityPreset::Medium),
        }
    }
}

/// Video codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoCodec {
    H264,
    H265,
    Vp9,
    Av1,
}

/// Audio output settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOutput {
    /// Audio codec.
    pub codec: AudioCodec,
    /// Target bitrate in kbps.
    pub bitrate_kbps: Option<u32>,
    /// Sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Number of channels.
    pub channels: Option<u8>,
}

impl Default for AudioOutput {
    fn default() -> Self {
        Self {
            codec: AudioCodec::Aac,
            bitrate_kbps: Some(128),
            sample_rate: Some(48000),
            channels: Some(2),
        }
    }
}

/// Audio codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioCodec {
    Aac,
    Mp3,
    Opus,
    Flac,
}

/// Quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityPreset {
    Low,
    Medium,
    High,
    Ultra,
}

/// Encoding preset definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preset {
    /// Preset name.
    pub name: String,
    /// Description.
    pub description: Option<String>,
    /// Video settings.
    pub video: VideoOutput,
    /// Audio settings.
    pub audio: AudioOutput,
    /// Output format.
    pub format: OutputFormat,
}

impl Default for Preset {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: Some("Default H.264/AAC preset".to_string()),
            video: VideoOutput::default(),
            audio: AudioOutput::default(),
            format: OutputFormat::Mp4,
        }
    }
}

impl From<&str> for Preset {
    fn from(name: &str) -> Self {
        match name {
            "hls_1080p" => Self::hls_1080p(),
            "hls_720p" => Self::hls_720p(),
            "hls_480p" => Self::hls_480p(),
            "web_high" => Self::web_high(),
            "web_medium" => Self::web_medium(),
            "archive" => Self::archive(),
            _ => Self::default(),
        }
    }
}

impl Preset {
    /// HLS 1080p preset.
    pub fn hls_1080p() -> Self {
        Self {
            name: "hls_1080p".to_string(),
            description: Some("HLS 1080p H.264".to_string()),
            video: VideoOutput {
                codec: VideoCodec::H264,
                bitrate_kbps: Some(5000),
                width: Some(1920),
                height: Some(1080),
                framerate: None,
                quality: Some(QualityPreset::High),
            },
            audio: AudioOutput::default(),
            format: OutputFormat::Hls,
        }
    }

    /// HLS 720p preset.
    pub fn hls_720p() -> Self {
        Self {
            name: "hls_720p".to_string(),
            description: Some("HLS 720p H.264".to_string()),
            video: VideoOutput {
                codec: VideoCodec::H264,
                bitrate_kbps: Some(2500),
                width: Some(1280),
                height: Some(720),
                framerate: None,
                quality: Some(QualityPreset::High),
            },
            audio: AudioOutput::default(),
            format: OutputFormat::Hls,
        }
    }

    /// HLS 480p preset.
    pub fn hls_480p() -> Self {
        Self {
            name: "hls_480p".to_string(),
            description: Some("HLS 480p H.264".to_string()),
            video: VideoOutput {
                codec: VideoCodec::H264,
                bitrate_kbps: Some(1200),
                width: Some(854),
                height: Some(480),
                framerate: None,
                quality: Some(QualityPreset::Medium),
            },
            audio: AudioOutput::default(),
            format: OutputFormat::Hls,
        }
    }

    /// Web high quality preset.
    pub fn web_high() -> Self {
        Self {
            name: "web_high".to_string(),
            description: Some("Web optimized high quality".to_string()),
            video: VideoOutput {
                codec: VideoCodec::H264,
                bitrate_kbps: Some(4000),
                width: Some(1920),
                height: Some(1080),
                framerate: Some(30.0),
                quality: Some(QualityPreset::High),
            },
            audio: AudioOutput::default(),
            format: OutputFormat::Mp4,
        }
    }

    /// Web medium quality preset.
    pub fn web_medium() -> Self {
        Self {
            name: "web_medium".to_string(),
            description: Some("Web optimized medium quality".to_string()),
            video: VideoOutput {
                codec: VideoCodec::H264,
                bitrate_kbps: Some(2000),
                width: Some(1280),
                height: Some(720),
                framerate: Some(30.0),
                quality: Some(QualityPreset::Medium),
            },
            audio: AudioOutput::default(),
            format: OutputFormat::Mp4,
        }
    }

    /// Archive quality preset.
    pub fn archive() -> Self {
        Self {
            name: "archive".to_string(),
            description: Some("High quality archive".to_string()),
            video: VideoOutput {
                codec: VideoCodec::H265,
                bitrate_kbps: Some(8000),
                width: None,
                height: None,
                framerate: None,
                quality: Some(QualityPreset::Ultra),
            },
            audio: AudioOutput {
                codec: AudioCodec::Flac,
                bitrate_kbps: None,
                sample_rate: Some(48000),
                channels: Some(2),
            },
            format: OutputFormat::Mkv,
        }
    }
}

/// Job status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// Job is queued waiting for a provider.
    Queued,
    /// Job is submitted to a provider.
    Submitted,
    /// Job is currently processing.
    Processing {
        /// Provider handling the job.
        provider: String,
        /// Progress percentage (0-100).
        progress: u8,
    },
    /// Job completed successfully.
    Completed,
    /// Job failed.
    Failed {
        /// Error message.
        error: String,
        /// Whether the job can be retried.
        retryable: bool,
    },
    /// Job was cancelled.
    Cancelled,
}

impl JobStatus {
    /// Check if job is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobStatus::Completed | JobStatus::Failed { .. } | JobStatus::Cancelled
        )
    }

    /// Check if job is currently active.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            JobStatus::Queued | JobStatus::Submitted | JobStatus::Processing { .. }
        )
    }
}

/// Result of a completed job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    /// Job ID.
    pub job_id: String,
    /// Final status.
    pub status: JobStatus,
    /// Provider that processed the job.
    pub provider: String,
    /// Output URIs.
    pub outputs: Vec<String>,
    /// Processing duration.
    pub duration: Duration,
    /// Actual cost in cents.
    pub cost_cents: Option<u32>,
    /// Job statistics.
    pub stats: JobStats,
    /// Timestamps.
    pub timestamps: JobTimestamps,
}

/// Job processing statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JobStats {
    /// Input file size in bytes.
    pub input_size_bytes: u64,
    /// Output file size in bytes.
    pub output_size_bytes: u64,
    /// Input duration in seconds.
    pub input_duration_secs: f64,
    /// Frames processed.
    pub frames_processed: u64,
    /// Average processing FPS.
    pub avg_fps: f64,
    /// Compression ratio.
    pub compression_ratio: f64,
}

/// Job timestamps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobTimestamps {
    /// When job was created.
    pub created_at: DateTime<Utc>,
    /// When job was submitted to provider.
    pub submitted_at: Option<DateTime<Utc>>,
    /// When job started processing.
    pub started_at: Option<DateTime<Utc>>,
    /// When job completed.
    pub completed_at: Option<DateTime<Utc>>,
}

impl Default for JobTimestamps {
    fn default() -> Self {
        Self {
            created_at: Utc::now(),
            submitted_at: None,
            started_at: None,
            completed_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_creation() {
        let job = TranscodeJob::new("s3://input/video.mp4", "s3://output/");
        assert!(!job.id.is_empty());
        assert_eq!(job.input, "s3://input/video.mp4");
        assert_eq!(job.output.uri, "s3://output/");
        assert_eq!(job.priority, 50);
    }

    #[test]
    fn test_job_builder() {
        let job = TranscodeJob::builder()
            .input("gs://bucket/input.mp4")
            .output_uri("gs://bucket/output/")
            .priority(100)
            .max_cost(500)
            .metadata("user_id", "12345")
            .build()
            .unwrap();

        assert_eq!(job.input, "gs://bucket/input.mp4");
        assert_eq!(job.priority, 100);
        assert_eq!(job.max_cost_cents, Some(500));
        assert_eq!(job.metadata.get("user_id"), Some(&"12345".to_string()));
    }

    #[test]
    fn test_preset_from_str() {
        let preset: Preset = "hls_1080p".into();
        assert_eq!(preset.name, "hls_1080p");
        assert_eq!(preset.video.width, Some(1920));
        assert_eq!(preset.video.height, Some(1080));
        assert_eq!(preset.format, OutputFormat::Hls);
    }

    #[test]
    fn test_job_status() {
        assert!(JobStatus::Completed.is_terminal());
        assert!(!JobStatus::Queued.is_terminal());
        assert!(JobStatus::Queued.is_active());
        assert!(!JobStatus::Completed.is_active());
    }
}
