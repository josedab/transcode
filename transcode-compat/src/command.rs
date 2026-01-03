//! FFmpeg command-line parser and pipeline builder.
//!
//! This module provides parsing of FFmpeg command-line arguments and
//! conversion to our internal pipeline configuration.

use crate::error::{CompatError, Result, StreamSpecifier, StreamType};
use crate::filter::{FilterChain, FilterGraph};
use crate::formats::{AudioCodecName, ContainerName, VideoCodecName};
use crate::options::{AspectRatio, Bitrate, OptionSet, ParsedOption, Resolution, TimeValue};
use crate::preset::{EncoderSettings, H264Profile, H265Profile, Level, Preset, Tune};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Parsed FFmpeg command representing a complete transcoding job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfmpegCommand {
    /// Input files with their options.
    pub inputs: Vec<InputFile>,
    /// Output files with their options.
    pub outputs: Vec<OutputFile>,
    /// Global options.
    pub global: GlobalOptions,
}

/// Input file specification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InputFile {
    /// File path.
    pub path: PathBuf,
    /// Seek position before reading.
    pub start_time: Option<TimeValue>,
    /// Duration to read.
    pub duration: Option<TimeValue>,
    /// Input format override.
    pub format: Option<ContainerName>,
    /// Stream loop count.
    pub stream_loop: Option<i32>,
    /// Frame rate for raw input.
    pub frame_rate: Option<f64>,
    /// Resolution for raw input.
    pub resolution: Option<Resolution>,
    /// Pixel format for raw input.
    pub pixel_format: Option<String>,
    /// Other input options.
    pub options: HashMap<String, String>,
}

/// Output file specification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputFile {
    /// File path.
    pub path: PathBuf,
    /// Output format.
    pub format: Option<ContainerName>,
    /// Video stream settings.
    pub video: Option<VideoOutputSettings>,
    /// Audio stream settings.
    pub audio: Option<AudioOutputSettings>,
    /// Stream mappings.
    pub mappings: Vec<StreamMapping>,
    /// Video filter chain.
    pub video_filters: FilterChain,
    /// Audio filter chain.
    pub audio_filters: FilterChain,
    /// Complex filter graph.
    pub filter_complex: Option<FilterGraph>,
    /// Start time (trim).
    pub start_time: Option<TimeValue>,
    /// Duration limit.
    pub duration: Option<TimeValue>,
    /// End time (trim).
    pub end_time: Option<TimeValue>,
    /// Shortest mode (end when shortest stream ends).
    pub shortest: bool,
    /// Metadata to set.
    pub metadata: HashMap<String, String>,
    /// Other output options.
    pub options: HashMap<String, String>,
}

/// Video output settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoOutputSettings {
    /// Video codec.
    pub codec: Option<VideoCodecName>,
    /// Bitrate.
    pub bitrate: Option<Bitrate>,
    /// Maximum bitrate for VBR.
    pub max_bitrate: Option<Bitrate>,
    /// Buffer size for rate control.
    pub buffer_size: Option<Bitrate>,
    /// Encoder settings (preset, tune, etc.).
    pub encoder: EncoderSettings,
    /// Target resolution.
    pub resolution: Option<Resolution>,
    /// Target frame rate.
    pub frame_rate: Option<f64>,
    /// Display aspect ratio.
    pub aspect_ratio: Option<AspectRatio>,
    /// Pixel format.
    pub pixel_format: Option<String>,
    /// Pass number for multi-pass encoding.
    pub pass: Option<u8>,
    /// Pass log file prefix.
    pub passlogfile: Option<String>,
    /// Copy mode (no transcoding).
    pub copy: bool,
    /// Disable video.
    pub disabled: bool,
}

/// Audio output settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioOutputSettings {
    /// Audio codec.
    pub codec: Option<AudioCodecName>,
    /// Bitrate.
    pub bitrate: Option<Bitrate>,
    /// Sample rate.
    pub sample_rate: Option<u32>,
    /// Number of channels.
    pub channels: Option<u8>,
    /// Channel layout.
    pub channel_layout: Option<String>,
    /// Sample format.
    pub sample_format: Option<String>,
    /// Audio quality (codec-specific).
    pub quality: Option<f32>,
    /// Copy mode (no transcoding).
    pub copy: bool,
    /// Disable audio.
    pub disabled: bool,
}

/// Stream mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMapping {
    /// Input file index.
    pub input_index: usize,
    /// Stream specifier.
    pub specifier: StreamSpecifier,
    /// Whether this is a negative mapping (exclude).
    pub negative: bool,
}

/// Global options.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalOptions {
    /// Overwrite output files.
    pub overwrite: bool,
    /// Don't overwrite output files.
    pub no_overwrite: bool,
    /// Number of threads.
    pub threads: Option<u32>,
    /// Hide banner.
    pub hide_banner: bool,
    /// Log level.
    pub loglevel: Option<String>,
    /// Statistics output.
    pub stats: bool,
    /// Progress output URL.
    pub progress: Option<String>,
}

impl FfmpegCommand {
    /// Create a new empty command.
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            global: GlobalOptions::default(),
        }
    }

    /// Parse FFmpeg command-line arguments.
    ///
    /// Expects arguments without the "ffmpeg" program name.
    pub fn parse<I, S>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let args: Vec<String> = args.into_iter().map(|s| s.as_ref().to_string()).collect();
        let mut parser = CommandParser::new(&args);
        parser.parse()
    }

    /// Add an input file.
    pub fn input(mut self, path: impl Into<PathBuf>) -> Self {
        self.inputs.push(InputFile {
            path: path.into(),
            ..Default::default()
        });
        self
    }

    /// Add an output file.
    pub fn output(mut self, path: impl Into<PathBuf>) -> Self {
        self.outputs.push(OutputFile {
            path: path.into(),
            ..Default::default()
        });
        self
    }

    /// Validate the command.
    pub fn validate(&self) -> Result<()> {
        if self.inputs.is_empty() {
            return Err(CompatError::MissingInput);
        }
        if self.outputs.is_empty() {
            return Err(CompatError::MissingOutput);
        }
        Ok(())
    }

    /// Convert to an OptionSet for easier processing.
    pub fn to_option_set(&self) -> OptionSet {
        let mut opts = OptionSet::new();

        // Collect inputs
        for input in &self.inputs {
            opts.inputs.push(input.path.to_string_lossy().to_string());
        }

        // Collect outputs
        for output in &self.outputs {
            opts.outputs.push(output.path.to_string_lossy().to_string());
        }

        // Collect video settings from first output
        if let Some(output) = self.outputs.first() {
            if let Some(video) = &output.video {
                if video.copy {
                    opts.copy_video = true;
                }
                if video.disabled {
                    opts.no_video = true;
                }
                if let Some(codec) = &video.codec {
                    opts.video_codec
                        .insert(StreamSpecifier::Video, codec.to_string());
                }
                if let Some(br) = video.bitrate {
                    opts.video_bitrate.insert(StreamSpecifier::Video, br);
                }
                opts.crf = video.encoder.crf.map(|f| f as f64);
                opts.preset = video.encoder.preset.map(|p| p.to_string());
                opts.resolution = video.resolution;
                opts.frame_rate = video.frame_rate;
                opts.aspect_ratio = video.aspect_ratio;
            }

            if let Some(audio) = &output.audio {
                if audio.copy {
                    opts.copy_audio = true;
                }
                if audio.disabled {
                    opts.no_audio = true;
                }
                if let Some(codec) = &audio.codec {
                    opts.audio_codec
                        .insert(StreamSpecifier::Audio, codec.to_string());
                }
                if let Some(br) = audio.bitrate {
                    opts.audio_bitrate.insert(StreamSpecifier::Audio, br);
                }
            }

            // Filters
            for filter in &output.video_filters.filters {
                opts.video_filters.push(filter.to_string());
            }
            for filter in &output.audio_filters.filters {
                opts.audio_filters.push(filter.to_string());
            }
            if let Some(fc) = &output.filter_complex {
                opts.filter_complex = Some(fc.to_string());
            }

            opts.start_time = output.start_time;
            opts.duration = output.duration;
            opts.end_time = output.end_time;
        }

        opts.overwrite = self.global.overwrite;

        opts
    }
}

impl Default for FfmpegCommand {
    fn default() -> Self {
        Self::new()
    }
}

/// Parser state for FFmpeg command-line arguments.
struct CommandParser<'a> {
    args: &'a [String],
    pos: usize,
    command: FfmpegCommand,
    pending_input_opts: InputFile,
    pending_output_opts: OutputFile,
}

impl<'a> CommandParser<'a> {
    fn new(args: &'a [String]) -> Self {
        Self {
            args,
            pos: 0,
            command: FfmpegCommand::new(),
            pending_input_opts: InputFile::default(),
            pending_output_opts: OutputFile::default(),
        }
    }

    fn parse(&mut self) -> Result<FfmpegCommand> {
        while self.pos < self.args.len() {
            let arg = &self.args[self.pos];

            if !arg.starts_with('-') {
                // This is an output file
                self.pending_output_opts.path = PathBuf::from(arg);
                self.command.outputs.push(std::mem::take(&mut self.pending_output_opts));
                self.pos += 1;
                continue;
            }

            match arg.as_str() {
                "-i" => {
                    self.parse_input()?;
                }
                "-f" => {
                    self.parse_format()?;
                }
                "-c" | "-codec" => {
                    self.parse_codec(arg)?;
                }
                "-b" | "-b:v" | "-b:a" => {
                    self.parse_bitrate(arg)?;
                }
                "-r" => {
                    self.parse_frame_rate()?;
                }
                "-s" => {
                    self.parse_resolution()?;
                }
                "-aspect" => {
                    self.parse_aspect()?;
                }
                "-ss" => {
                    self.parse_start_time()?;
                }
                "-t" => {
                    self.parse_duration()?;
                }
                "-to" => {
                    self.parse_end_time()?;
                }
                "-vf" | "-filter:v" => {
                    self.parse_video_filter()?;
                }
                "-af" | "-filter:a" => {
                    self.parse_audio_filter()?;
                }
                "-filter_complex" | "-lavfi" => {
                    self.parse_filter_complex()?;
                }
                "-preset" => {
                    self.parse_preset()?;
                }
                "-tune" => {
                    self.parse_tune()?;
                }
                "-crf" => {
                    self.parse_crf()?;
                }
                "-qp" => {
                    self.parse_qp()?;
                }
                "-profile" | "-profile:v" => {
                    self.parse_profile()?;
                }
                "-level" => {
                    self.parse_level()?;
                }
                "-g" => {
                    self.parse_gop()?;
                }
                "-bf" => {
                    self.parse_bframes()?;
                }
                "-refs" => {
                    self.parse_refs()?;
                }
                "-map" => {
                    self.parse_map()?;
                }
                "-an" => {
                    self.pending_output_opts.audio.get_or_insert_default().disabled = true;
                    self.pos += 1;
                }
                "-vn" => {
                    self.pending_output_opts.video.get_or_insert_default().disabled = true;
                    self.pos += 1;
                }
                "-y" => {
                    self.command.global.overwrite = true;
                    self.pos += 1;
                }
                "-n" => {
                    self.command.global.no_overwrite = true;
                    self.pos += 1;
                }
                "-threads" => {
                    self.parse_threads()?;
                }
                "-hide_banner" => {
                    self.command.global.hide_banner = true;
                    self.pos += 1;
                }
                "-loglevel" | "-v" => {
                    self.parse_loglevel()?;
                }
                "-stats" => {
                    self.command.global.stats = true;
                    self.pos += 1;
                }
                "-shortest" => {
                    self.pending_output_opts.shortest = true;
                    self.pos += 1;
                }
                "-ar" => {
                    self.parse_sample_rate()?;
                }
                "-ac" => {
                    self.parse_channels()?;
                }
                "-pix_fmt" => {
                    self.parse_pixel_format()?;
                }
                "-maxrate" => {
                    self.parse_max_bitrate()?;
                }
                "-bufsize" => {
                    self.parse_buffer_size()?;
                }
                "-pass" => {
                    self.parse_pass()?;
                }
                "-passlogfile" => {
                    self.parse_passlogfile()?;
                }
                "-metadata" => {
                    self.parse_metadata()?;
                }
                _ => {
                    // Check for stream-specific options like -c:v, -b:a:0
                    if arg.starts_with("-c:") {
                        self.parse_codec(arg)?;
                    } else if arg.starts_with("-b:") {
                        self.parse_bitrate(arg)?;
                    } else if arg.starts_with("-profile:") {
                        self.parse_profile()?;
                    } else {
                        // Unknown option, skip it and its value if present
                        self.pos += 1;
                        if self.pos < self.args.len() && !self.args[self.pos].starts_with('-') {
                            self.pos += 1;
                        }
                    }
                }
            }
        }

        Ok(std::mem::take(&mut self.command))
    }

    fn next_value(&mut self, option: &str) -> Result<String> {
        self.pos += 1;
        if self.pos >= self.args.len() {
            return Err(CompatError::MissingValue(option.to_string()));
        }
        Ok(self.args[self.pos].clone())
    }

    fn parse_input(&mut self) -> Result<()> {
        let path = self.next_value("-i")?;
        self.pending_input_opts.path = PathBuf::from(path);
        self.command.inputs.push(std::mem::take(&mut self.pending_input_opts));
        self.pos += 1;
        Ok(())
    }

    fn parse_format(&mut self) -> Result<()> {
        let format = self.next_value("-f")?;
        let container = ContainerName::parse(&format)?;

        // Apply to pending output
        self.pending_output_opts.format = Some(container);
        self.pos += 1;
        Ok(())
    }

    fn parse_codec(&mut self, arg: &str) -> Result<()> {
        let value = self.next_value(arg)?;
        let parsed = ParsedOption::parse(arg, Some(&value))?;

        let specifier = parsed.specifier.unwrap_or(StreamSpecifier::All);

        match specifier {
            StreamSpecifier::Video | StreamSpecifier::All => {
                if let Ok(codec) = VideoCodecName::parse(&value) {
                    let video = self.pending_output_opts.video.get_or_insert_default();
                    if codec.is_copy() {
                        video.copy = true;
                    } else {
                        video.codec = Some(codec);
                    }
                }
            }
            StreamSpecifier::Audio => {
                if let Ok(codec) = AudioCodecName::parse(&value) {
                    let audio = self.pending_output_opts.audio.get_or_insert_default();
                    if codec.is_copy() {
                        audio.copy = true;
                    } else {
                        audio.codec = Some(codec);
                    }
                }
            }
            StreamSpecifier::TypeIndex { stream_type, .. } => {
                match stream_type {
                    StreamType::Video => {
                        if let Ok(codec) = VideoCodecName::parse(&value) {
                            let video = self.pending_output_opts.video.get_or_insert_default();
                            if codec.is_copy() {
                                video.copy = true;
                            } else {
                                video.codec = Some(codec);
                            }
                        }
                    }
                    StreamType::Audio => {
                        if let Ok(codec) = AudioCodecName::parse(&value) {
                            let audio = self.pending_output_opts.audio.get_or_insert_default();
                            if codec.is_copy() {
                                audio.copy = true;
                            } else {
                                audio.codec = Some(codec);
                            }
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        self.pos += 1;
        Ok(())
    }

    fn parse_bitrate(&mut self, arg: &str) -> Result<()> {
        let value = self.next_value(arg)?;
        let bitrate = Bitrate::parse(&value)?;
        let parsed = ParsedOption::parse(arg, Some(&value))?;

        let specifier = parsed.specifier.unwrap_or(StreamSpecifier::All);

        match specifier {
            StreamSpecifier::Video | StreamSpecifier::All => {
                self.pending_output_opts.video.get_or_insert_default().bitrate = Some(bitrate);
            }
            StreamSpecifier::Audio => {
                self.pending_output_opts.audio.get_or_insert_default().bitrate = Some(bitrate);
            }
            StreamSpecifier::TypeIndex { stream_type, .. } => {
                match stream_type {
                    StreamType::Video => {
                        self.pending_output_opts.video.get_or_insert_default().bitrate = Some(bitrate);
                    }
                    StreamType::Audio => {
                        self.pending_output_opts.audio.get_or_insert_default().bitrate = Some(bitrate);
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        self.pos += 1;
        Ok(())
    }

    fn parse_frame_rate(&mut self) -> Result<()> {
        let value = self.next_value("-r")?;
        let fps: f64 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-r".to_string(),
            value: value.clone(),
            reason: "invalid frame rate".to_string(),
        })?;
        self.pending_output_opts.video.get_or_insert_default().frame_rate = Some(fps);
        self.pos += 1;
        Ok(())
    }

    fn parse_resolution(&mut self) -> Result<()> {
        let value = self.next_value("-s")?;
        let resolution = Resolution::parse(&value)?;
        self.pending_output_opts.video.get_or_insert_default().resolution = Some(resolution);
        self.pos += 1;
        Ok(())
    }

    fn parse_aspect(&mut self) -> Result<()> {
        let value = self.next_value("-aspect")?;
        let aspect = AspectRatio::parse(&value)?;
        self.pending_output_opts.video.get_or_insert_default().aspect_ratio = Some(aspect);
        self.pos += 1;
        Ok(())
    }

    fn parse_start_time(&mut self) -> Result<()> {
        let value = self.next_value("-ss")?;
        let time = TimeValue::parse(&value)?;

        // If no inputs yet, apply to next input; otherwise to output
        if self.command.inputs.is_empty() {
            self.pending_input_opts.start_time = Some(time);
        } else {
            self.pending_output_opts.start_time = Some(time);
        }

        self.pos += 1;
        Ok(())
    }

    fn parse_duration(&mut self) -> Result<()> {
        let value = self.next_value("-t")?;
        let time = TimeValue::parse(&value)?;

        if self.command.inputs.is_empty() {
            self.pending_input_opts.duration = Some(time);
        } else {
            self.pending_output_opts.duration = Some(time);
        }

        self.pos += 1;
        Ok(())
    }

    fn parse_end_time(&mut self) -> Result<()> {
        let value = self.next_value("-to")?;
        let time = TimeValue::parse(&value)?;
        self.pending_output_opts.end_time = Some(time);
        self.pos += 1;
        Ok(())
    }

    fn parse_video_filter(&mut self) -> Result<()> {
        let value = self.next_value("-vf")?;
        let chain = FilterChain::parse(&value)?;
        self.pending_output_opts.video_filters = chain;
        self.pos += 1;
        Ok(())
    }

    fn parse_audio_filter(&mut self) -> Result<()> {
        let value = self.next_value("-af")?;
        let chain = FilterChain::parse(&value)?;
        self.pending_output_opts.audio_filters = chain;
        self.pos += 1;
        Ok(())
    }

    fn parse_filter_complex(&mut self) -> Result<()> {
        let value = self.next_value("-filter_complex")?;
        let graph = FilterGraph::parse(&value)?;
        self.pending_output_opts.filter_complex = Some(graph);
        self.pos += 1;
        Ok(())
    }

    fn parse_preset(&mut self) -> Result<()> {
        let value = self.next_value("-preset")?;
        let preset = Preset::parse(&value)?;
        self.pending_output_opts.video.get_or_insert_default().encoder.preset = Some(preset);
        self.pos += 1;
        Ok(())
    }

    fn parse_tune(&mut self) -> Result<()> {
        let value = self.next_value("-tune")?;
        let tune = Tune::parse(&value)?;
        self.pending_output_opts.video.get_or_insert_default().encoder.tune = Some(tune);
        self.pos += 1;
        Ok(())
    }

    fn parse_crf(&mut self) -> Result<()> {
        let value = self.next_value("-crf")?;
        let crf: f32 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-crf".to_string(),
            value: value.clone(),
            reason: "invalid CRF value".to_string(),
        })?;
        self.pending_output_opts.video.get_or_insert_default().encoder.crf = Some(crf);
        self.pos += 1;
        Ok(())
    }

    fn parse_qp(&mut self) -> Result<()> {
        let value = self.next_value("-qp")?;
        let qp: u8 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-qp".to_string(),
            value: value.clone(),
            reason: "invalid QP value".to_string(),
        })?;
        self.pending_output_opts.video.get_or_insert_default().encoder.qp = Some(qp);
        self.pos += 1;
        Ok(())
    }

    fn parse_profile(&mut self) -> Result<()> {
        let value = self.next_value("-profile")?;

        // Try H.264 profile first
        if let Ok(profile) = H264Profile::parse(&value) {
            self.pending_output_opts.video.get_or_insert_default().encoder.h264_profile = Some(profile);
        } else if let Ok(profile) = H265Profile::parse(&value) {
            self.pending_output_opts.video.get_or_insert_default().encoder.h265_profile = Some(profile);
        }

        self.pos += 1;
        Ok(())
    }

    fn parse_level(&mut self) -> Result<()> {
        let value = self.next_value("-level")?;
        let level = Level::parse(&value)?;
        self.pending_output_opts.video.get_or_insert_default().encoder.level = Some(level);
        self.pos += 1;
        Ok(())
    }

    fn parse_gop(&mut self) -> Result<()> {
        let value = self.next_value("-g")?;
        let gop: u32 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-g".to_string(),
            value: value.clone(),
            reason: "invalid GOP size".to_string(),
        })?;
        self.pending_output_opts.video.get_or_insert_default().encoder.keyframe_interval = Some(gop);
        self.pos += 1;
        Ok(())
    }

    fn parse_bframes(&mut self) -> Result<()> {
        let value = self.next_value("-bf")?;
        let bframes: u8 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-bf".to_string(),
            value: value.clone(),
            reason: "invalid B-frames value".to_string(),
        })?;
        self.pending_output_opts.video.get_or_insert_default().encoder.bframes = Some(bframes);
        self.pos += 1;
        Ok(())
    }

    fn parse_refs(&mut self) -> Result<()> {
        let value = self.next_value("-refs")?;
        let refs: u8 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-refs".to_string(),
            value: value.clone(),
            reason: "invalid reference frames value".to_string(),
        })?;
        self.pending_output_opts.video.get_or_insert_default().encoder.ref_frames = Some(refs);
        self.pos += 1;
        Ok(())
    }

    fn parse_map(&mut self) -> Result<()> {
        let value = self.next_value("-map")?;

        let (negative, spec_str) = if let Some(stripped) = value.strip_prefix('-') {
            (true, stripped)
        } else {
            (false, value.as_str())
        };

        // Parse input_index:stream_specifier
        let parts: Vec<&str> = spec_str.splitn(2, ':').collect();
        let input_index: usize = parts[0].parse().unwrap_or(0);

        let specifier = if parts.len() > 1 {
            StreamSpecifier::parse(parts[1])?
        } else {
            StreamSpecifier::All
        };

        self.pending_output_opts.mappings.push(StreamMapping {
            input_index,
            specifier,
            negative,
        });

        self.pos += 1;
        Ok(())
    }

    fn parse_threads(&mut self) -> Result<()> {
        let value = self.next_value("-threads")?;
        let threads: u32 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-threads".to_string(),
            value: value.clone(),
            reason: "invalid thread count".to_string(),
        })?;
        self.command.global.threads = Some(threads);
        self.pos += 1;
        Ok(())
    }

    fn parse_loglevel(&mut self) -> Result<()> {
        let value = self.next_value("-loglevel")?;
        self.command.global.loglevel = Some(value);
        self.pos += 1;
        Ok(())
    }

    fn parse_sample_rate(&mut self) -> Result<()> {
        let value = self.next_value("-ar")?;
        let rate: u32 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-ar".to_string(),
            value: value.clone(),
            reason: "invalid sample rate".to_string(),
        })?;
        self.pending_output_opts.audio.get_or_insert_default().sample_rate = Some(rate);
        self.pos += 1;
        Ok(())
    }

    fn parse_channels(&mut self) -> Result<()> {
        let value = self.next_value("-ac")?;
        let channels: u8 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-ac".to_string(),
            value: value.clone(),
            reason: "invalid channel count".to_string(),
        })?;
        self.pending_output_opts.audio.get_or_insert_default().channels = Some(channels);
        self.pos += 1;
        Ok(())
    }

    fn parse_pixel_format(&mut self) -> Result<()> {
        let value = self.next_value("-pix_fmt")?;
        self.pending_output_opts.video.get_or_insert_default().pixel_format = Some(value);
        self.pos += 1;
        Ok(())
    }

    fn parse_max_bitrate(&mut self) -> Result<()> {
        let value = self.next_value("-maxrate")?;
        let bitrate = Bitrate::parse(&value)?;
        self.pending_output_opts.video.get_or_insert_default().max_bitrate = Some(bitrate);
        self.pos += 1;
        Ok(())
    }

    fn parse_buffer_size(&mut self) -> Result<()> {
        let value = self.next_value("-bufsize")?;
        let bitrate = Bitrate::parse(&value)?;
        self.pending_output_opts.video.get_or_insert_default().buffer_size = Some(bitrate);
        self.pos += 1;
        Ok(())
    }

    fn parse_pass(&mut self) -> Result<()> {
        let value = self.next_value("-pass")?;
        let pass: u8 = value.parse().map_err(|_| CompatError::InvalidValue {
            option: "-pass".to_string(),
            value: value.clone(),
            reason: "invalid pass number".to_string(),
        })?;
        self.pending_output_opts.video.get_or_insert_default().pass = Some(pass);
        if pass > 1 {
            self.pending_output_opts.video.get_or_insert_default().encoder.two_pass = true;
        }
        self.pos += 1;
        Ok(())
    }

    fn parse_passlogfile(&mut self) -> Result<()> {
        let value = self.next_value("-passlogfile")?;
        self.pending_output_opts.video.get_or_insert_default().passlogfile = Some(value);
        self.pos += 1;
        Ok(())
    }

    fn parse_metadata(&mut self) -> Result<()> {
        let value = self.next_value("-metadata")?;
        if let Some(eq_pos) = value.find('=') {
            let key = &value[..eq_pos];
            let val = &value[eq_pos + 1..];
            self.pending_output_opts.metadata.insert(key.to_string(), val.to_string());
        }
        self.pos += 1;
        Ok(())
    }
}

/// Builder for creating pipeline configurations from FFmpeg commands.
#[derive(Debug, Clone)]
pub struct CommandBuilder {
    command: FfmpegCommand,
}

impl CommandBuilder {
    /// Create a new builder from a parsed command.
    pub fn new(command: FfmpegCommand) -> Self {
        Self { command }
    }

    /// Parse FFmpeg arguments and create a builder.
    pub fn from_args<I, S>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let command = FfmpegCommand::parse(args)?;
        Ok(Self::new(command))
    }

    /// Get the parsed command.
    pub fn command(&self) -> &FfmpegCommand {
        &self.command
    }

    /// Get input files.
    pub fn inputs(&self) -> &[InputFile] {
        &self.command.inputs
    }

    /// Get output files.
    pub fn outputs(&self) -> &[OutputFile] {
        &self.command.outputs
    }

    /// Get video codec for output.
    pub fn video_codec(&self) -> Option<VideoCodecName> {
        self.command
            .outputs
            .first()
            .and_then(|o| o.video.as_ref())
            .and_then(|v| v.codec)
    }

    /// Get audio codec for output.
    pub fn audio_codec(&self) -> Option<AudioCodecName> {
        self.command
            .outputs
            .first()
            .and_then(|o| o.audio.as_ref())
            .and_then(|a| a.codec)
    }

    /// Get video bitrate.
    pub fn video_bitrate(&self) -> Option<Bitrate> {
        self.command
            .outputs
            .first()
            .and_then(|o| o.video.as_ref())
            .and_then(|v| v.bitrate)
    }

    /// Get audio bitrate.
    pub fn audio_bitrate(&self) -> Option<Bitrate> {
        self.command
            .outputs
            .first()
            .and_then(|o| o.audio.as_ref())
            .and_then(|a| a.bitrate)
    }

    /// Get video filters.
    pub fn video_filters(&self) -> Option<&FilterChain> {
        self.command
            .outputs
            .first()
            .map(|o| &o.video_filters)
    }

    /// Get audio filters.
    pub fn audio_filters(&self) -> Option<&FilterChain> {
        self.command
            .outputs
            .first()
            .map(|o| &o.audio_filters)
    }

    /// Get encoder settings.
    pub fn encoder_settings(&self) -> Option<&EncoderSettings> {
        self.command
            .outputs
            .first()
            .and_then(|o| o.video.as_ref())
            .map(|v| &v.encoder)
    }

    /// Check if video should be copied.
    pub fn is_video_copy(&self) -> bool {
        self.command
            .outputs
            .first()
            .and_then(|o| o.video.as_ref())
            .map(|v| v.copy)
            .unwrap_or(false)
    }

    /// Check if audio should be copied.
    pub fn is_audio_copy(&self) -> bool {
        self.command
            .outputs
            .first()
            .and_then(|o| o.audio.as_ref())
            .map(|a| a.copy)
            .unwrap_or(false)
    }

    /// Check if video is disabled.
    pub fn is_video_disabled(&self) -> bool {
        self.command
            .outputs
            .first()
            .and_then(|o| o.video.as_ref())
            .map(|v| v.disabled)
            .unwrap_or(false)
    }

    /// Check if audio is disabled.
    pub fn is_audio_disabled(&self) -> bool {
        self.command
            .outputs
            .first()
            .and_then(|o| o.audio.as_ref())
            .map(|a| a.disabled)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_command() {
        let args = ["-i", "input.mp4", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        assert_eq!(cmd.inputs.len(), 1);
        assert_eq!(cmd.inputs[0].path, PathBuf::from("input.mp4"));
        assert_eq!(cmd.outputs.len(), 1);
        assert_eq!(cmd.outputs[0].path, PathBuf::from("output.mp4"));
    }

    #[test]
    fn test_parse_with_codec() {
        let args = ["-i", "input.mp4", "-c:v", "libx264", "-c:a", "aac", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        let video = cmd.outputs[0].video.as_ref().unwrap();
        assert_eq!(video.codec, Some(VideoCodecName::H264));

        let audio = cmd.outputs[0].audio.as_ref().unwrap();
        assert_eq!(audio.codec, Some(AudioCodecName::Aac));
    }

    #[test]
    fn test_parse_with_bitrate() {
        let args = ["-i", "input.mp4", "-b:v", "5M", "-b:a", "128k", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        let video = cmd.outputs[0].video.as_ref().unwrap();
        assert_eq!(video.bitrate.unwrap().bps(), 5_000_000);

        let audio = cmd.outputs[0].audio.as_ref().unwrap();
        assert_eq!(audio.bitrate.unwrap().bps(), 128_000);
    }

    #[test]
    fn test_parse_with_filters() {
        let args = ["-i", "input.mp4", "-vf", "scale=1920:1080,fps=30", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        let filters = &cmd.outputs[0].video_filters;
        assert_eq!(filters.len(), 2);
        assert_eq!(filters.filters[0].name, "scale");
        assert_eq!(filters.filters[1].name, "fps");
    }

    #[test]
    fn test_parse_with_preset() {
        let args = ["-i", "input.mp4", "-c:v", "libx264", "-preset", "slow", "-crf", "23", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        let video = cmd.outputs[0].video.as_ref().unwrap();
        assert_eq!(video.encoder.preset, Some(Preset::Slow));
        assert_eq!(video.encoder.crf, Some(23.0));
    }

    #[test]
    fn test_parse_copy_mode() {
        let args = ["-i", "input.mp4", "-c:v", "copy", "-c:a", "copy", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        let video = cmd.outputs[0].video.as_ref().unwrap();
        assert!(video.copy);

        let audio = cmd.outputs[0].audio.as_ref().unwrap();
        assert!(audio.copy);
    }

    #[test]
    fn test_parse_disable_streams() {
        let args = ["-i", "input.mp4", "-an", "-c:v", "libx264", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        let audio = cmd.outputs[0].audio.as_ref().unwrap();
        assert!(audio.disabled);
    }

    #[test]
    fn test_parse_time_options() {
        let args = ["-ss", "1:30", "-i", "input.mp4", "-t", "60", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        // -ss before -i applies to input
        assert!(cmd.inputs[0].start_time.is_some());
        assert!((cmd.inputs[0].start_time.unwrap().seconds() - 90.0).abs() < 0.001);

        // -t after -i applies to output
        assert!(cmd.outputs[0].duration.is_some());
        assert!((cmd.outputs[0].duration.unwrap().seconds() - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_resolution() {
        let args = ["-i", "input.mp4", "-s", "1920x1080", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        let video = cmd.outputs[0].video.as_ref().unwrap();
        let res = video.resolution.unwrap();
        assert_eq!(res.width, 1920);
        assert_eq!(res.height, 1080);
    }

    #[test]
    fn test_parse_global_options() {
        let args = ["-y", "-threads", "4", "-i", "input.mp4", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        assert!(cmd.global.overwrite);
        assert_eq!(cmd.global.threads, Some(4));
    }

    #[test]
    fn test_command_builder() {
        let args = ["-i", "input.mp4", "-c:v", "libx264", "-b:v", "5M", "-preset", "slow", "output.mp4"];
        let builder = CommandBuilder::from_args(&args).unwrap();

        assert_eq!(builder.video_codec(), Some(VideoCodecName::H264));
        assert_eq!(builder.video_bitrate().unwrap().bps(), 5_000_000);
        assert_eq!(builder.encoder_settings().unwrap().preset, Some(Preset::Slow));
        assert!(!builder.is_video_copy());
    }

    #[test]
    fn test_parse_map() {
        let args = ["-i", "input.mp4", "-map", "0:v:0", "-map", "0:a:0", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();

        assert_eq!(cmd.outputs[0].mappings.len(), 2);
        assert_eq!(cmd.outputs[0].mappings[0].input_index, 0);
        assert!(!cmd.outputs[0].mappings[0].negative);
    }

    #[test]
    fn test_to_option_set() {
        let args = ["-i", "input.mp4", "-c:v", "libx264", "-b:v", "5M", "output.mp4"];
        let cmd = FfmpegCommand::parse(&args).unwrap();
        let opts = cmd.to_option_set();

        assert_eq!(opts.inputs.len(), 1);
        assert_eq!(opts.outputs.len(), 1);
        assert!(opts.video_codec.contains_key(&StreamSpecifier::Video));
    }
}
