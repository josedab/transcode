//! Automatic captioning for transcode
//!
//! This crate provides speech-to-text captioning with Whisper integration.

use serde::{Deserialize, Serialize};
use std::path::Path;

mod error;
mod subtitle;
mod timing;

pub use error::*;
pub use subtitle::*;
pub use timing::*;

/// Result type for caption operations
pub type Result<T> = std::result::Result<T, CaptionError>;

/// Caption configuration
#[derive(Debug, Clone)]
pub struct CaptionConfig {
    /// Model size
    pub model_size: ModelSize,
    /// Language code (None for auto-detect)
    pub language: Option<String>,
    /// Enable translation to English
    pub translate: bool,
    /// Enable word-level timestamps
    pub word_timestamps: bool,
    /// Maximum segment length in seconds
    pub max_segment_length: f32,
}

impl Default for CaptionConfig {
    fn default() -> Self {
        Self {
            model_size: ModelSize::Base,
            language: None,
            translate: false,
            word_timestamps: false,
            max_segment_length: 30.0,
        }
    }
}

/// Whisper model size
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSize {
    /// Tiny model (~39M params)
    Tiny,
    /// Base model (~74M params)
    Base,
    /// Small model (~244M params)
    Small,
    /// Medium model (~769M params)
    Medium,
    /// Large model (~1.5B params)
    Large,
}

impl ModelSize {
    /// Get model filename
    pub fn filename(&self) -> &str {
        match self {
            ModelSize::Tiny => "ggml-tiny.bin",
            ModelSize::Base => "ggml-base.bin",
            ModelSize::Small => "ggml-small.bin",
            ModelSize::Medium => "ggml-medium.bin",
            ModelSize::Large => "ggml-large.bin",
        }
    }

    /// Get approximate model size in MB
    pub fn size_mb(&self) -> u32 {
        match self {
            ModelSize::Tiny => 75,
            ModelSize::Base => 142,
            ModelSize::Small => 466,
            ModelSize::Medium => 1533,
            ModelSize::Large => 2952,
        }
    }
}

/// Transcription segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Start time in milliseconds
    pub start_ms: u64,
    /// End time in milliseconds
    pub end_ms: u64,
    /// Transcribed text
    pub text: String,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Word-level timestamps
    pub words: Vec<Word>,
    /// Detected language
    pub language: Option<String>,
}

impl Segment {
    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.end_ms - self.start_ms
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.duration_ms() as f32 / 1000.0
    }
}

/// Word with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    /// Start time in milliseconds
    pub start_ms: u64,
    /// End time in milliseconds
    pub end_ms: u64,
    /// Word text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
}

/// Transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcription {
    /// All segments
    pub segments: Vec<Segment>,
    /// Full text
    pub text: String,
    /// Detected language
    pub language: String,
    /// Language confidence
    pub language_confidence: f32,
}

impl Transcription {
    /// Get total duration
    pub fn duration_ms(&self) -> u64 {
        self.segments.last().map(|s| s.end_ms).unwrap_or(0)
    }

    /// Export to SRT format
    pub fn to_srt(&self) -> String {
        let mut srt = String::new();

        for (i, segment) in self.segments.iter().enumerate() {
            // Sequence number
            srt.push_str(&format!("{}\n", i + 1));

            // Timestamps
            srt.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(segment.start_ms),
                format_srt_time(segment.end_ms)
            ));

            // Text
            srt.push_str(&segment.text);
            srt.push_str("\n\n");
        }

        srt
    }

    /// Export to VTT format
    pub fn to_vtt(&self) -> String {
        let mut vtt = String::from("WEBVTT\n\n");

        for segment in &self.segments {
            vtt.push_str(&format!(
                "{} --> {}\n",
                format_vtt_time(segment.start_ms),
                format_vtt_time(segment.end_ms)
            ));
            vtt.push_str(&segment.text);
            vtt.push_str("\n\n");
        }

        vtt
    }
}

fn format_srt_time(ms: u64) -> String {
    let hours = ms / 3600000;
    let minutes = (ms % 3600000) / 60000;
    let seconds = (ms % 60000) / 1000;
    let millis = ms % 1000;

    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

fn format_vtt_time(ms: u64) -> String {
    let hours = ms / 3600000;
    let minutes = (ms % 3600000) / 60000;
    let seconds = (ms % 60000) / 1000;
    let millis = ms % 1000;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

/// Auto-captioner using Whisper
pub struct Captioner {
    config: CaptionConfig,
}

impl Captioner {
    /// Create a new captioner
    pub fn new(config: CaptionConfig) -> Self {
        Self { config }
    }

    /// Load Whisper model
    ///
    /// Note: Whisper integration requires building whisper.cpp separately
    /// and linking to the whisper-rs crate with compatible clang version.
    pub fn load_model(&mut self, _model_path: &Path) -> Result<()> {
        // Whisper model loading would be implemented here
        // Currently requires external whisper.cpp build
        Err(CaptionError::WhisperNotEnabled)
    }

    /// Transcribe audio samples (16kHz mono f32)
    pub fn transcribe(&self, audio: &[f32]) -> Result<Transcription> {
        self.fallback_transcribe(audio)
    }

    fn fallback_transcribe(&self, audio: &[f32]) -> Result<Transcription> {
        // Placeholder transcription for testing
        let duration_ms = (audio.len() as f64 / 16000.0 * 1000.0) as u64;

        Ok(Transcription {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: duration_ms,
                text: "[Transcription requires Whisper feature]".into(),
                confidence: 0.0,
                words: Vec::new(),
                language: Some("en".into()),
            }],
            text: "[Transcription requires Whisper feature]".into(),
            language: "en".into(),
            language_confidence: 0.0,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &CaptionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srt_format() {
        let trans = Transcription {
            segments: vec![
                Segment {
                    start_ms: 0,
                    end_ms: 2500,
                    text: "Hello world".into(),
                    confidence: 0.95,
                    words: Vec::new(),
                    language: Some("en".into()),
                },
                Segment {
                    start_ms: 3000,
                    end_ms: 5000,
                    text: "This is a test".into(),
                    confidence: 0.90,
                    words: Vec::new(),
                    language: Some("en".into()),
                },
            ],
            text: "Hello world This is a test".into(),
            language: "en".into(),
            language_confidence: 0.99,
        };

        let srt = trans.to_srt();
        assert!(srt.contains("00:00:00,000 --> 00:00:02,500"));
        assert!(srt.contains("Hello world"));
    }

    #[test]
    fn test_vtt_format() {
        let trans = Transcription {
            segments: vec![Segment {
                start_ms: 1000,
                end_ms: 3000,
                text: "Test caption".into(),
                confidence: 0.9,
                words: Vec::new(),
                language: None,
            }],
            text: "Test caption".into(),
            language: "en".into(),
            language_confidence: 0.9,
        };

        let vtt = trans.to_vtt();
        assert!(vtt.starts_with("WEBVTT"));
        assert!(vtt.contains("00:00:01.000 --> 00:00:03.000"));
    }

    #[test]
    fn test_model_sizes() {
        assert_eq!(ModelSize::Tiny.filename(), "ggml-tiny.bin");
        assert!(ModelSize::Large.size_mb() > ModelSize::Base.size_mb());
    }
}
