//! Output format management for the ingest server.

use serde::{Deserialize, Serialize};

/// Output streaming format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Hls,
    Dash,
    Whep,
}

impl OutputFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Hls => "m3u8",
            Self::Dash => "mpd",
            Self::Whep => "sdp",
        }
    }

    pub fn content_type(&self) -> &'static str {
        match self {
            Self::Hls => "application/vnd.apple.mpegurl",
            Self::Dash => "application/dash+xml",
            Self::Whep => "application/sdp",
        }
    }
}

/// Configuration for stream output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: OutputFormat,
    pub directory: String,
    pub segment_duration_secs: f64,
    pub playlist_length: u32,
    pub delete_old_segments: bool,
    pub video_codec: String,
    pub audio_codec: String,
    pub video_bitrate_kbps: u32,
    pub audio_bitrate_kbps: u32,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Hls,
            directory: "/tmp/transcode-hls".into(),
            segment_duration_secs: 6.0,
            playlist_length: 5,
            delete_old_segments: true,
            video_codec: "h264".into(),
            audio_codec: "aac".into(),
            video_bitrate_kbps: 4000,
            audio_bitrate_kbps: 128,
        }
    }
}

/// Manages output writers for active streams.
pub struct OutputManager {
    config: OutputConfig,
    active_outputs: std::collections::HashMap<String, StreamOutput>,
}

struct StreamOutput {
    stream_key: String,
    segments_written: u64,
    bytes_written: u64,
}

impl OutputManager {
    pub fn new(config: OutputConfig) -> Self {
        Self {
            config,
            active_outputs: std::collections::HashMap::new(),
        }
    }

    /// Start output for a stream.
    pub fn start_output(&mut self, stream_key: &str) -> crate::Result<()> {
        self.active_outputs.insert(
            stream_key.to_string(),
            StreamOutput {
                stream_key: stream_key.into(),
                segments_written: 0,
                bytes_written: 0,
            },
        );
        Ok(())
    }

    /// Stop output for a stream.
    pub fn stop_output(&mut self, stream_key: &str) {
        self.active_outputs.remove(stream_key);
    }

    /// Get statistics for a stream's output.
    pub fn output_stats(&self, stream_key: &str) -> Option<(u64, u64)> {
        self.active_outputs
            .get(stream_key)
            .map(|o| (o.segments_written, o.bytes_written))
    }

    pub fn config(&self) -> &OutputConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format() {
        assert_eq!(OutputFormat::Hls.extension(), "m3u8");
        assert_eq!(OutputFormat::Dash.extension(), "mpd");
        assert_eq!(OutputFormat::Whep.extension(), "sdp");
    }

    #[test]
    fn test_output_manager() {
        let mut manager = OutputManager::new(OutputConfig::default());
        manager.start_output("stream-1").unwrap();
        assert!(manager.output_stats("stream-1").is_some());

        manager.stop_output("stream-1");
        assert!(manager.output_stats("stream-1").is_none());
    }
}
