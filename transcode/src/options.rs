//! Transcoding options and configuration.

use std::path::PathBuf;

/// High-level transcoding options using builder pattern.
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
