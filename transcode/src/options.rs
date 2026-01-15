//! Transcoding options and configuration.
//!
//! This module provides two builder patterns:
//!
//! 1. **Typestate Builder** ([`TranscodeOptionsBuilder`]): Compile-time guarantee that
//!    required fields (input and output) are set before building. Recommended for new code.
//!
//! 2. **Legacy Builder** ([`TranscodeOptions`]): Runtime validation via `validate()`.
//!    Maintained for backward compatibility.
//!
//! # Example (Typestate Builder)
//!
//! ```
//! use transcode::TranscodeOptionsBuilder;
//!
//! // This compiles - both input and output are set
//! let options = TranscodeOptionsBuilder::new()
//!     .input("input.mp4")
//!     .output("output.mp4")
//!     .video_codec("h264")
//!     .build();
//!
//! // This would NOT compile - output not set
//! // let options = TranscodeOptionsBuilder::new()
//! //     .input("input.mp4")
//! //     .build();  // Error: method `build` not found
//! ```

use std::marker::PhantomData;
use std::path::PathBuf;

// ============================================================================
// Typestate Builder Pattern
// ============================================================================

/// Marker type indicating a required field has not been set.
#[derive(Debug, Clone, Copy, Default)]
pub struct NotSet;

/// Marker type indicating input has been configured.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputSet;

/// Marker type indicating output has been configured.
#[derive(Debug, Clone, Copy, Default)]
pub struct OutputSet;

/// Typestate builder for `TranscodeOptions`.
///
/// This builder uses the typestate pattern to ensure at compile time that
/// required fields (input and output) are set before `build()` can be called.
///
/// # Type Parameters
///
/// - `I`: Input state - either `NotSet` or `InputSet`
/// - `O`: Output state - either `NotSet` or `OutputSet`
#[derive(Debug, Clone)]
pub struct TranscodeOptionsBuilder<I = NotSet, O = NotSet> {
    input: Option<InputConfig>,
    output: Option<OutputConfig>,
    video: Option<VideoConfig>,
    audio: Option<AudioConfig>,
    threads: Option<usize>,
    hardware_accel: bool,
    overwrite: bool,
    _input_state: PhantomData<I>,
    _output_state: PhantomData<O>,
}

impl Default for TranscodeOptionsBuilder<NotSet, NotSet> {
    fn default() -> Self {
        Self::new()
    }
}

impl TranscodeOptionsBuilder<NotSet, NotSet> {
    /// Create a new builder with no fields set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            input: None,
            output: None,
            video: None,
            audio: None,
            threads: None,
            hardware_accel: false,
            overwrite: false,
            _input_state: PhantomData,
            _output_state: PhantomData,
        }
    }
}

// Input setters - transition from NotSet to InputSet
impl<O> TranscodeOptionsBuilder<NotSet, O> {
    /// Set input file path.
    ///
    /// This transitions the builder from `NotSet` to `InputSet` state.
    #[must_use]
    pub fn input(self, path: impl Into<PathBuf>) -> TranscodeOptionsBuilder<InputSet, O> {
        TranscodeOptionsBuilder {
            input: Some(InputConfig {
                path: path.into(),
                start_time: None,
                duration: None,
                stream_index: None,
            }),
            output: self.output,
            video: self.video,
            audio: self.audio,
            threads: self.threads,
            hardware_accel: self.hardware_accel,
            overwrite: self.overwrite,
            _input_state: PhantomData,
            _output_state: PhantomData,
        }
    }

    /// Set input configuration.
    ///
    /// This transitions the builder from `NotSet` to `InputSet` state.
    #[must_use]
    pub fn input_config(self, config: InputConfig) -> TranscodeOptionsBuilder<InputSet, O> {
        TranscodeOptionsBuilder {
            input: Some(config),
            output: self.output,
            video: self.video,
            audio: self.audio,
            threads: self.threads,
            hardware_accel: self.hardware_accel,
            overwrite: self.overwrite,
            _input_state: PhantomData,
            _output_state: PhantomData,
        }
    }
}

// Output setters - transition from NotSet to OutputSet
impl<I> TranscodeOptionsBuilder<I, NotSet> {
    /// Set output file path.
    ///
    /// This transitions the builder from `NotSet` to `OutputSet` state.
    #[must_use]
    pub fn output(self, path: impl Into<PathBuf>) -> TranscodeOptionsBuilder<I, OutputSet> {
        TranscodeOptionsBuilder {
            input: self.input,
            output: Some(OutputConfig {
                path: path.into(),
                format: None,
            }),
            video: self.video,
            audio: self.audio,
            threads: self.threads,
            hardware_accel: self.hardware_accel,
            overwrite: self.overwrite,
            _input_state: PhantomData,
            _output_state: PhantomData,
        }
    }

    /// Set output configuration.
    ///
    /// This transitions the builder from `NotSet` to `OutputSet` state.
    #[must_use]
    pub fn output_config(self, config: OutputConfig) -> TranscodeOptionsBuilder<I, OutputSet> {
        TranscodeOptionsBuilder {
            input: self.input,
            output: Some(config),
            video: self.video,
            audio: self.audio,
            threads: self.threads,
            hardware_accel: self.hardware_accel,
            overwrite: self.overwrite,
            _input_state: PhantomData,
            _output_state: PhantomData,
        }
    }
}

// Optional configuration methods - available in any state
impl<I, O> TranscodeOptionsBuilder<I, O> {
    /// Set video codec.
    #[must_use]
    pub fn video_codec(mut self, codec: impl Into<String>) -> Self {
        if let Some(ref mut video) = self.video {
            video.codec = Some(codec.into());
        } else {
            self.video = Some(VideoConfig {
                codec: Some(codec.into()),
                ..Default::default()
            });
        }
        self
    }

    /// Set video configuration.
    #[must_use]
    pub fn video_config(mut self, config: VideoConfig) -> Self {
        self.video = Some(config);
        self
    }

    /// Set audio codec.
    #[must_use]
    pub fn audio_codec(mut self, codec: impl Into<String>) -> Self {
        if let Some(ref mut audio) = self.audio {
            audio.codec = Some(codec.into());
        } else {
            self.audio = Some(AudioConfig {
                codec: Some(codec.into()),
                ..Default::default()
            });
        }
        self
    }

    /// Set audio configuration.
    #[must_use]
    pub fn audio_config(mut self, config: AudioConfig) -> Self {
        self.audio = Some(config);
        self
    }

    /// Set number of threads.
    #[must_use]
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Enable hardware acceleration.
    #[must_use]
    pub fn hardware_acceleration(mut self, enable: bool) -> Self {
        self.hardware_accel = enable;
        self
    }

    /// Enable overwrite mode.
    #[must_use]
    pub fn overwrite(mut self, enable: bool) -> Self {
        self.overwrite = enable;
        self
    }

    /// Set video bitrate.
    #[must_use]
    pub fn video_bitrate(mut self, bitrate: u64) -> Self {
        if let Some(ref mut video) = self.video {
            video.bitrate = Some(bitrate);
        } else {
            self.video = Some(VideoConfig {
                bitrate: Some(bitrate),
                ..Default::default()
            });
        }
        self
    }

    /// Set video resolution.
    #[must_use]
    pub fn video_resolution(mut self, width: u32, height: u32) -> Self {
        if let Some(ref mut video) = self.video {
            video.width = Some(width);
            video.height = Some(height);
        } else {
            self.video = Some(VideoConfig {
                width: Some(width),
                height: Some(height),
                ..Default::default()
            });
        }
        self
    }

    /// Set video frame rate.
    #[must_use]
    pub fn video_framerate(mut self, fps: f64) -> Self {
        if let Some(ref mut video) = self.video {
            video.framerate = Some(fps);
        } else {
            self.video = Some(VideoConfig {
                framerate: Some(fps),
                ..Default::default()
            });
        }
        self
    }

    /// Set audio bitrate.
    #[must_use]
    pub fn audio_bitrate(mut self, bitrate: u64) -> Self {
        if let Some(ref mut audio) = self.audio {
            audio.bitrate = Some(bitrate);
        } else {
            self.audio = Some(AudioConfig {
                bitrate: Some(bitrate),
                ..Default::default()
            });
        }
        self
    }

    /// Set audio sample rate.
    #[must_use]
    pub fn audio_sample_rate(mut self, sample_rate: u32) -> Self {
        if let Some(ref mut audio) = self.audio {
            audio.sample_rate = Some(sample_rate);
        } else {
            self.audio = Some(AudioConfig {
                sample_rate: Some(sample_rate),
                ..Default::default()
            });
        }
        self
    }

    /// Set audio channels.
    #[must_use]
    pub fn audio_channels(mut self, channels: u32) -> Self {
        if let Some(ref mut audio) = self.audio {
            audio.channels = Some(channels);
        } else {
            self.audio = Some(AudioConfig {
                channels: Some(channels),
                ..Default::default()
            });
        }
        self
    }
}

// Build method - only available when both input and output are set
impl TranscodeOptionsBuilder<InputSet, OutputSet> {
    /// Build the final `TranscodeOptions`.
    ///
    /// This method is only available when both input and output have been set,
    /// providing compile-time guarantee that required fields are present.
    #[must_use]
    pub fn build(self) -> TranscodeOptions {
        TranscodeOptions {
            // Safe to unwrap - typestate guarantees these are Some
            input: self.input,
            output: self.output,
            video: self.video,
            audio: self.audio,
            threads: self.threads,
            hardware_accel: self.hardware_accel,
            overwrite: self.overwrite,
        }
    }
}

// ============================================================================
// Legacy Builder (maintained for backward compatibility)
// ============================================================================

/// High-level transcoding options using builder pattern.
///
/// **Note**: For new code, prefer [`TranscodeOptionsBuilder`] which provides
/// compile-time validation of required fields.
#[derive(Debug, Clone)]
pub struct TranscodeOptions {
    /// Input configuration.
    pub input: Option<InputConfig>,
    /// Output configuration.
    pub output: Option<OutputConfig>,
    /// Video configuration.
    pub video: Option<VideoConfig>,
    /// Audio configuration.
    pub audio: Option<AudioConfig>,
    /// Number of threads for parallel processing.
    pub threads: Option<usize>,
    /// Enable hardware acceleration.
    pub hardware_accel: bool,
    /// Overwrite output file if exists.
    pub overwrite: bool,
}

impl Default for TranscodeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl TranscodeOptions {
    /// Create new transcoding options.
    ///
    /// **Note**: For compile-time validation of required fields, use
    /// [`TranscodeOptionsBuilder::new()`] instead.
    #[must_use]
    pub fn new() -> Self {
        Self {
            input: None,
            output: None,
            video: None,
            audio: None,
            threads: None,
            hardware_accel: false,
            overwrite: false,
        }
    }

    /// Create a typestate builder for compile-time validation.
    ///
    /// This is the preferred way to construct `TranscodeOptions` as it
    /// ensures at compile time that required fields are set.
    #[must_use]
    pub fn builder() -> TranscodeOptionsBuilder<NotSet, NotSet> {
        TranscodeOptionsBuilder::new()
    }

    /// Set input file path.
    #[must_use]
    pub fn input(mut self, path: impl Into<PathBuf>) -> Self {
        self.input = Some(InputConfig {
            path: path.into(),
            start_time: None,
            duration: None,
            stream_index: None,
        });
        self
    }

    /// Set input configuration.
    #[must_use]
    pub fn input_config(mut self, config: InputConfig) -> Self {
        self.input = Some(config);
        self
    }

    /// Set output file path.
    #[must_use]
    pub fn output(mut self, path: impl Into<PathBuf>) -> Self {
        self.output = Some(OutputConfig {
            path: path.into(),
            format: None,
        });
        self
    }

    /// Set output configuration.
    #[must_use]
    pub fn output_config(mut self, config: OutputConfig) -> Self {
        self.output = Some(config);
        self
    }

    /// Set video codec.
    #[must_use]
    pub fn video_codec(mut self, codec: impl Into<String>) -> Self {
        if let Some(ref mut video) = self.video {
            video.codec = Some(codec.into());
        } else {
            self.video = Some(VideoConfig {
                codec: Some(codec.into()),
                ..Default::default()
            });
        }
        self
    }

    /// Set video configuration.
    #[must_use]
    pub fn video_config(mut self, config: VideoConfig) -> Self {
        self.video = Some(config);
        self
    }

    /// Set audio codec.
    #[must_use]
    pub fn audio_codec(mut self, codec: impl Into<String>) -> Self {
        if let Some(ref mut audio) = self.audio {
            audio.codec = Some(codec.into());
        } else {
            self.audio = Some(AudioConfig {
                codec: Some(codec.into()),
                ..Default::default()
            });
        }
        self
    }

    /// Set audio configuration.
    #[must_use]
    pub fn audio_config(mut self, config: AudioConfig) -> Self {
        self.audio = Some(config);
        self
    }

    /// Set number of threads.
    #[must_use]
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }

    /// Enable hardware acceleration.
    #[must_use]
    pub fn hardware_acceleration(mut self, enable: bool) -> Self {
        self.hardware_accel = enable;
        self
    }

    /// Enable overwrite mode.
    #[must_use]
    pub fn overwrite(mut self, enable: bool) -> Self {
        self.overwrite = enable;
        self
    }

    /// Set video bitrate.
    #[must_use]
    pub fn video_bitrate(mut self, bitrate: u64) -> Self {
        if let Some(ref mut video) = self.video {
            video.bitrate = Some(bitrate);
        } else {
            self.video = Some(VideoConfig {
                bitrate: Some(bitrate),
                ..Default::default()
            });
        }
        self
    }

    /// Set video resolution.
    #[must_use]
    pub fn video_resolution(mut self, width: u32, height: u32) -> Self {
        if let Some(ref mut video) = self.video {
            video.width = Some(width);
            video.height = Some(height);
        } else {
            self.video = Some(VideoConfig {
                width: Some(width),
                height: Some(height),
                ..Default::default()
            });
        }
        self
    }

    /// Set video frame rate.
    #[must_use]
    pub fn video_framerate(mut self, fps: f64) -> Self {
        if let Some(ref mut video) = self.video {
            video.framerate = Some(fps);
        } else {
            self.video = Some(VideoConfig {
                framerate: Some(fps),
                ..Default::default()
            });
        }
        self
    }

    /// Set audio bitrate.
    #[must_use]
    pub fn audio_bitrate(mut self, bitrate: u64) -> Self {
        if let Some(ref mut audio) = self.audio {
            audio.bitrate = Some(bitrate);
        } else {
            self.audio = Some(AudioConfig {
                bitrate: Some(bitrate),
                ..Default::default()
            });
        }
        self
    }

    /// Set audio sample rate.
    #[must_use]
    pub fn audio_sample_rate(mut self, sample_rate: u32) -> Self {
        if let Some(ref mut audio) = self.audio {
            audio.sample_rate = Some(sample_rate);
        } else {
            self.audio = Some(AudioConfig {
                sample_rate: Some(sample_rate),
                ..Default::default()
            });
        }
        self
    }

    /// Set audio channels.
    #[must_use]
    pub fn audio_channels(mut self, channels: u32) -> Self {
        if let Some(ref mut audio) = self.audio {
            audio.channels = Some(channels);
        } else {
            self.audio = Some(AudioConfig {
                channels: Some(channels),
                ..Default::default()
            });
        }
        self
    }

    /// Validate the options.
    ///
    /// **Note**: When using [`TranscodeOptionsBuilder`], validation is performed
    /// at compile time, making this method unnecessary.
    pub fn validate(&self) -> Result<(), String> {
        if self.input.is_none() {
            return Err("Input file not specified".into());
        }
        if self.output.is_none() {
            return Err("Output file not specified".into());
        }
        Ok(())
    }
}

/// Input file configuration.
#[derive(Debug, Clone)]
pub struct InputConfig {
    /// Input file path.
    pub path: PathBuf,
    /// Start time in seconds.
    pub start_time: Option<f64>,
    /// Duration in seconds.
    pub duration: Option<f64>,
    /// Specific stream index to process.
    pub stream_index: Option<usize>,
}

impl InputConfig {
    /// Create new input configuration.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            start_time: None,
            duration: None,
            stream_index: None,
        }
    }

    /// Set start time.
    #[must_use]
    pub fn start_time(mut self, seconds: f64) -> Self {
        self.start_time = Some(seconds);
        self
    }

    /// Set duration.
    #[must_use]
    pub fn duration(mut self, seconds: f64) -> Self {
        self.duration = Some(seconds);
        self
    }

    /// Set stream index.
    #[must_use]
    pub fn stream_index(mut self, index: usize) -> Self {
        self.stream_index = Some(index);
        self
    }
}

/// Output file configuration.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// Output file path.
    pub path: PathBuf,
    /// Output container format.
    pub format: Option<String>,
}

impl OutputConfig {
    /// Create new output configuration.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            format: None,
        }
    }

    /// Set output format.
    #[must_use]
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }
}

/// Video stream configuration.
#[derive(Debug, Clone, Default)]
pub struct VideoConfig {
    /// Video codec name.
    pub codec: Option<String>,
    /// Target bitrate in bits per second.
    pub bitrate: Option<u64>,
    /// Output width.
    pub width: Option<u32>,
    /// Output height.
    pub height: Option<u32>,
    /// Frame rate.
    pub framerate: Option<f64>,
    /// Pixel format.
    pub pixel_format: Option<String>,
    /// Encoder preset.
    pub preset: Option<String>,
    /// Encoder profile.
    pub profile: Option<String>,
    /// Encoder level.
    pub level: Option<String>,
    /// GOP size (keyframe interval).
    pub gop_size: Option<u32>,
    /// Number of B-frames.
    pub b_frames: Option<u32>,
    /// Constant Rate Factor (quality-based encoding).
    pub crf: Option<u32>,
}

impl VideoConfig {
    /// Create new video configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set codec.
    #[must_use]
    pub fn codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = Some(codec.into());
        self
    }

    /// Set bitrate.
    #[must_use]
    pub fn bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = Some(bitrate);
        self
    }

    /// Set resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set frame rate.
    #[must_use]
    pub fn framerate(mut self, fps: f64) -> Self {
        self.framerate = Some(fps);
        self
    }

    /// Set pixel format.
    #[must_use]
    pub fn pixel_format(mut self, format: impl Into<String>) -> Self {
        self.pixel_format = Some(format.into());
        self
    }

    /// Set encoder preset.
    #[must_use]
    pub fn preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Set CRF value.
    #[must_use]
    pub fn crf(mut self, crf: u32) -> Self {
        self.crf = Some(crf);
        self
    }

    /// Set GOP size.
    #[must_use]
    pub fn gop_size(mut self, size: u32) -> Self {
        self.gop_size = Some(size);
        self
    }
}

/// Audio stream configuration.
#[derive(Debug, Clone, Default)]
pub struct AudioConfig {
    /// Audio codec name.
    pub codec: Option<String>,
    /// Target bitrate in bits per second.
    pub bitrate: Option<u64>,
    /// Sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Number of channels.
    pub channels: Option<u32>,
    /// Sample format.
    pub sample_format: Option<String>,
    /// Channel layout.
    pub channel_layout: Option<String>,
}

impl AudioConfig {
    /// Create new audio configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set codec.
    #[must_use]
    pub fn codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = Some(codec.into());
        self
    }

    /// Set bitrate.
    #[must_use]
    pub fn bitrate(mut self, bitrate: u64) -> Self {
        self.bitrate = Some(bitrate);
        self
    }

    /// Set sample rate.
    #[must_use]
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Set number of channels.
    #[must_use]
    pub fn channels(mut self, channels: u32) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Set sample format.
    #[must_use]
    pub fn sample_format(mut self, format: impl Into<String>) -> Self {
        self.sample_format = Some(format.into());
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Tests for typestate builder

    #[test]
    fn typestate_builder_basic_build() {
        let options = TranscodeOptionsBuilder::new()
            .input("input.mp4")
            .output("output.mp4")
            .build();

        assert!(options.input.is_some());
        assert!(options.output.is_some());
        assert_eq!(options.input.unwrap().path.to_str().unwrap(), "input.mp4");
        assert_eq!(options.output.unwrap().path.to_str().unwrap(), "output.mp4");
    }

    #[test]
    fn typestate_builder_with_video_config() {
        let options = TranscodeOptionsBuilder::new()
            .input("input.mp4")
            .video_codec("h264")
            .video_bitrate(5_000_000)
            .video_resolution(1920, 1080)
            .output("output.mp4")
            .build();

        let video = options.video.unwrap();
        assert_eq!(video.codec, Some("h264".to_string()));
        assert_eq!(video.bitrate, Some(5_000_000));
        assert_eq!(video.width, Some(1920));
        assert_eq!(video.height, Some(1080));
    }

    #[test]
    fn typestate_builder_with_audio_config() {
        let options = TranscodeOptionsBuilder::new()
            .input("input.mp4")
            .output("output.mp4")
            .audio_codec("aac")
            .audio_bitrate(128_000)
            .audio_sample_rate(44100)
            .audio_channels(2)
            .build();

        let audio = options.audio.unwrap();
        assert_eq!(audio.codec, Some("aac".to_string()));
        assert_eq!(audio.bitrate, Some(128_000));
        assert_eq!(audio.sample_rate, Some(44100));
        assert_eq!(audio.channels, Some(2));
    }

    #[test]
    fn typestate_builder_any_order() {
        // Output before input
        let options = TranscodeOptionsBuilder::new()
            .output("output.mp4")
            .input("input.mp4")
            .build();

        assert!(options.input.is_some());
        assert!(options.output.is_some());
    }

    #[test]
    fn typestate_builder_with_input_config() {
        let input = InputConfig::new("input.mp4")
            .start_time(10.0)
            .duration(30.0);

        let options = TranscodeOptionsBuilder::new()
            .input_config(input)
            .output("output.mp4")
            .build();

        let input = options.input.unwrap();
        assert_eq!(input.start_time, Some(10.0));
        assert_eq!(input.duration, Some(30.0));
    }

    #[test]
    fn typestate_builder_with_output_config() {
        let output = OutputConfig::new("output.mp4")
            .format("mp4");

        let options = TranscodeOptionsBuilder::new()
            .input("input.mp4")
            .output_config(output)
            .build();

        let output = options.output.unwrap();
        assert_eq!(output.format, Some("mp4".to_string()));
    }

    #[test]
    fn typestate_builder_full_config() {
        let options = TranscodeOptionsBuilder::new()
            .input("input.mp4")
            .output("output.mp4")
            .video_codec("h264")
            .audio_codec("aac")
            .threads(4)
            .hardware_acceleration(true)
            .overwrite(true)
            .build();

        assert!(options.input.is_some());
        assert!(options.output.is_some());
        assert!(options.video.is_some());
        assert!(options.audio.is_some());
        assert_eq!(options.threads, Some(4));
        assert!(options.hardware_accel);
        assert!(options.overwrite);
    }

    #[test]
    fn typestate_builder_from_transcode_options() {
        // Test TranscodeOptions::builder() method
        let options = TranscodeOptions::builder()
            .input("input.mp4")
            .output("output.mp4")
            .build();

        assert!(options.input.is_some());
        assert!(options.output.is_some());
    }

    // Tests for legacy builder

    #[test]
    fn legacy_builder_validate_success() {
        let options = TranscodeOptions::new()
            .input("input.mp4")
            .output("output.mp4");

        assert!(options.validate().is_ok());
    }

    #[test]
    fn legacy_builder_validate_missing_input() {
        let options = TranscodeOptions::new()
            .output("output.mp4");

        assert!(options.validate().is_err());
        assert_eq!(options.validate().unwrap_err(), "Input file not specified");
    }

    #[test]
    fn legacy_builder_validate_missing_output() {
        let options = TranscodeOptions::new()
            .input("input.mp4");

        assert!(options.validate().is_err());
        assert_eq!(options.validate().unwrap_err(), "Output file not specified");
    }

    // Tests for config structs

    #[test]
    fn input_config_builder() {
        let config = InputConfig::new("input.mp4")
            .start_time(5.0)
            .duration(10.0)
            .stream_index(1);

        assert_eq!(config.path.to_str().unwrap(), "input.mp4");
        assert_eq!(config.start_time, Some(5.0));
        assert_eq!(config.duration, Some(10.0));
        assert_eq!(config.stream_index, Some(1));
    }

    #[test]
    fn output_config_builder() {
        let config = OutputConfig::new("output.mkv")
            .format("matroska");

        assert_eq!(config.path.to_str().unwrap(), "output.mkv");
        assert_eq!(config.format, Some("matroska".to_string()));
    }

    #[test]
    fn video_config_builder() {
        let config = VideoConfig::new()
            .codec("h264")
            .bitrate(5_000_000)
            .resolution(1280, 720)
            .framerate(30.0)
            .pixel_format("yuv420p")
            .preset("medium")
            .crf(23)
            .gop_size(250);

        assert_eq!(config.codec, Some("h264".to_string()));
        assert_eq!(config.bitrate, Some(5_000_000));
        assert_eq!(config.width, Some(1280));
        assert_eq!(config.height, Some(720));
        assert_eq!(config.framerate, Some(30.0));
        assert_eq!(config.pixel_format, Some("yuv420p".to_string()));
        assert_eq!(config.preset, Some("medium".to_string()));
        assert_eq!(config.crf, Some(23));
        assert_eq!(config.gop_size, Some(250));
    }

    #[test]
    fn audio_config_builder() {
        let config = AudioConfig::new()
            .codec("aac")
            .bitrate(128_000)
            .sample_rate(48000)
            .channels(2)
            .sample_format("fltp");

        assert_eq!(config.codec, Some("aac".to_string()));
        assert_eq!(config.bitrate, Some(128_000));
        assert_eq!(config.sample_rate, Some(48000));
        assert_eq!(config.channels, Some(2));
        assert_eq!(config.sample_format, Some("fltp".to_string()));
    }
}
