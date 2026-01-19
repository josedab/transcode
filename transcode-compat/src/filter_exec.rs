//! Filter execution framework.
//!
//! This module provides the infrastructure for executing parsed filter graphs,
//! including frame format negotiation and filter chaining.

#![allow(clippy::type_complexity)]
#![allow(clippy::manual_strip)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::should_implement_trait)]
#![allow(dead_code)]

use crate::error::{CompatError, Result};
use crate::filter::{Filter, FilterChain};
use std::collections::HashMap;

/// Pixel format for video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// YUV 4:2:0 planar.
    Yuv420p,
    /// YUV 4:2:2 planar.
    Yuv422p,
    /// YUV 4:4:4 planar.
    Yuv444p,
    /// NV12 semi-planar (Y plane + interleaved UV).
    Nv12,
    /// RGB 24-bit.
    Rgb24,
    /// BGR 24-bit.
    Bgr24,
    /// RGBA 32-bit.
    Rgba,
    /// BGRA 32-bit.
    Bgra,
    /// Gray 8-bit.
    Gray8,
    /// Gray 16-bit.
    Gray16,
    /// 10-bit YUV 4:2:0.
    Yuv420p10,
    /// 10-bit YUV 4:2:2.
    Yuv422p10,
}

impl PixelFormat {
    /// Get the number of planes.
    pub fn num_planes(&self) -> usize {
        match self {
            Self::Yuv420p | Self::Yuv422p | Self::Yuv444p
            | Self::Yuv420p10 | Self::Yuv422p10 => 3,
            Self::Nv12 => 2,
            Self::Rgb24 | Self::Bgr24 => 1,
            Self::Rgba | Self::Bgra => 1,
            Self::Gray8 | Self::Gray16 => 1,
        }
    }

    /// Get bits per pixel.
    pub fn bits_per_pixel(&self) -> u32 {
        match self {
            Self::Yuv420p => 12,
            Self::Nv12 => 12,
            Self::Yuv422p => 16,
            Self::Yuv444p => 24,
            Self::Rgb24 | Self::Bgr24 => 24,
            Self::Rgba | Self::Bgra => 32,
            Self::Gray8 => 8,
            Self::Gray16 => 16,
            Self::Yuv420p10 => 15,
            Self::Yuv422p10 => 20,
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "yuv420p" => Some(Self::Yuv420p),
            "yuv422p" => Some(Self::Yuv422p),
            "yuv444p" => Some(Self::Yuv444p),
            "nv12" => Some(Self::Nv12),
            "rgb24" => Some(Self::Rgb24),
            "bgr24" => Some(Self::Bgr24),
            "rgba" => Some(Self::Rgba),
            "bgra" => Some(Self::Bgra),
            "gray" | "gray8" => Some(Self::Gray8),
            "gray16" | "gray16le" => Some(Self::Gray16),
            "yuv420p10" | "yuv420p10le" => Some(Self::Yuv420p10),
            "yuv422p10" | "yuv422p10le" => Some(Self::Yuv422p10),
            _ => None,
        }
    }

    /// Get FFmpeg format string.
    pub fn as_ffmpeg_str(&self) -> &str {
        match self {
            Self::Yuv420p => "yuv420p",
            Self::Yuv422p => "yuv422p",
            Self::Yuv444p => "yuv444p",
            Self::Nv12 => "nv12",
            Self::Rgb24 => "rgb24",
            Self::Bgr24 => "bgr24",
            Self::Rgba => "rgba",
            Self::Bgra => "bgra",
            Self::Gray8 => "gray",
            Self::Gray16 => "gray16le",
            Self::Yuv420p10 => "yuv420p10le",
            Self::Yuv422p10 => "yuv422p10le",
        }
    }
}

/// Sample format for audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    /// Unsigned 8-bit.
    U8,
    /// Signed 16-bit.
    S16,
    /// Signed 32-bit.
    S32,
    /// 32-bit float.
    Flt,
    /// 64-bit double.
    Dbl,
    /// Planar unsigned 8-bit.
    U8p,
    /// Planar signed 16-bit.
    S16p,
    /// Planar signed 32-bit.
    S32p,
    /// Planar 32-bit float.
    Fltp,
    /// Planar 64-bit double.
    Dblp,
}

impl SampleFormat {
    /// Check if format is planar.
    pub fn is_planar(&self) -> bool {
        matches!(self, Self::U8p | Self::S16p | Self::S32p | Self::Fltp | Self::Dblp)
    }

    /// Get bytes per sample.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::U8 | Self::U8p => 1,
            Self::S16 | Self::S16p => 2,
            Self::S32 | Self::S32p | Self::Flt | Self::Fltp => 4,
            Self::Dbl | Self::Dblp => 8,
        }
    }

    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "u8" => Some(Self::U8),
            "s16" => Some(Self::S16),
            "s32" => Some(Self::S32),
            "flt" => Some(Self::Flt),
            "dbl" => Some(Self::Dbl),
            "u8p" => Some(Self::U8p),
            "s16p" => Some(Self::S16p),
            "s32p" => Some(Self::S32p),
            "fltp" => Some(Self::Fltp),
            "dblp" => Some(Self::Dblp),
            _ => None,
        }
    }
}

/// Video frame properties for format negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFormat {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Pixel format.
    pub pixel_format: PixelFormat,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Sample aspect ratio numerator.
    pub sar_num: u32,
    /// Sample aspect ratio denominator.
    pub sar_den: u32,
    /// Color range (limited or full).
    pub full_range: bool,
}

impl Default for VideoFormat {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixel_format: PixelFormat::Yuv420p,
            fps_num: 30,
            fps_den: 1,
            sar_num: 1,
            sar_den: 1,
            full_range: false,
        }
    }
}

impl VideoFormat {
    /// Create a new video format.
    pub fn new(width: u32, height: u32, pixel_format: PixelFormat) -> Self {
        Self {
            width,
            height,
            pixel_format,
            ..Default::default()
        }
    }

    /// Set frame rate.
    pub fn with_fps(mut self, num: u32, den: u32) -> Self {
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    /// Set sample aspect ratio.
    pub fn with_sar(mut self, num: u32, den: u32) -> Self {
        self.sar_num = num;
        self.sar_den = den;
        self
    }

    /// Get frame rate as f64.
    pub fn fps(&self) -> f64 {
        self.fps_num as f64 / self.fps_den as f64
    }
}

/// Audio format properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFormat {
    /// Sample rate.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u32,
    /// Sample format.
    pub sample_format: SampleFormat,
    /// Channel layout (e.g., "stereo", "5.1").
    pub channel_layout: String,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            sample_format: SampleFormat::Fltp,
            channel_layout: "stereo".to_string(),
        }
    }
}

impl AudioFormat {
    /// Create a new audio format.
    pub fn new(sample_rate: u32, channels: u32, sample_format: SampleFormat) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format,
            channel_layout: Self::default_layout(channels),
        }
    }

    /// Get default channel layout for channel count.
    fn default_layout(channels: u32) -> String {
        match channels {
            1 => "mono".to_string(),
            2 => "stereo".to_string(),
            6 => "5.1".to_string(),
            8 => "7.1".to_string(),
            _ => format!("{}c", channels),
        }
    }
}

/// Video frame buffer.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Frame data planes.
    pub planes: Vec<Vec<u8>>,
    /// Stride (bytes per row) for each plane.
    pub strides: Vec<usize>,
    /// Frame format.
    pub format: VideoFormat,
    /// Presentation timestamp (in timebase units).
    pub pts: i64,
    /// Whether this is a keyframe.
    pub keyframe: bool,
}

impl VideoFrame {
    /// Create a new video frame.
    pub fn new(format: VideoFormat) -> Self {
        let num_planes = format.pixel_format.num_planes();
        let mut planes = Vec::with_capacity(num_planes);
        let mut strides = Vec::with_capacity(num_planes);

        // Allocate planes based on pixel format
        for i in 0..num_planes {
            let (plane_width, plane_height) = match format.pixel_format {
                PixelFormat::Yuv420p | PixelFormat::Yuv420p10 => {
                    if i == 0 {
                        (format.width as usize, format.height as usize)
                    } else {
                        ((format.width / 2) as usize, (format.height / 2) as usize)
                    }
                }
                PixelFormat::Yuv422p | PixelFormat::Yuv422p10 => {
                    if i == 0 {
                        (format.width as usize, format.height as usize)
                    } else {
                        ((format.width / 2) as usize, format.height as usize)
                    }
                }
                PixelFormat::Yuv444p => (format.width as usize, format.height as usize),
                PixelFormat::Nv12 => {
                    if i == 0 {
                        (format.width as usize, format.height as usize)
                    } else {
                        (format.width as usize, (format.height / 2) as usize)
                    }
                }
                _ => (format.width as usize, format.height as usize),
            };

            let bytes_per_component = match format.pixel_format {
                PixelFormat::Yuv420p10 | PixelFormat::Yuv422p10 | PixelFormat::Gray16 => 2,
                _ => 1,
            };

            let stride = plane_width * bytes_per_component;
            strides.push(stride);
            planes.push(vec![0u8; stride * plane_height]);
        }

        Self {
            planes,
            strides,
            format,
            pts: 0,
            keyframe: false,
        }
    }

    /// Set presentation timestamp.
    pub fn with_pts(mut self, pts: i64) -> Self {
        self.pts = pts;
        self
    }

    /// Set keyframe flag.
    pub fn with_keyframe(mut self, keyframe: bool) -> Self {
        self.keyframe = keyframe;
        self
    }
}

/// Audio samples buffer.
#[derive(Debug, Clone)]
pub struct AudioSamples {
    /// Sample data (interleaved or planar).
    pub data: Vec<Vec<u8>>,
    /// Number of samples.
    pub num_samples: usize,
    /// Audio format.
    pub format: AudioFormat,
    /// Presentation timestamp.
    pub pts: i64,
}

impl AudioSamples {
    /// Create new audio samples.
    pub fn new(format: AudioFormat, num_samples: usize) -> Self {
        let bytes_per_sample = format.sample_format.bytes_per_sample();
        let data = if format.sample_format.is_planar() {
            (0..format.channels as usize)
                .map(|_| vec![0u8; num_samples * bytes_per_sample])
                .collect()
        } else {
            vec![vec![0u8; num_samples * bytes_per_sample * format.channels as usize]]
        };

        Self {
            data,
            num_samples,
            format,
            pts: 0,
        }
    }

    /// Set presentation timestamp.
    pub fn with_pts(mut self, pts: i64) -> Self {
        self.pts = pts;
        self
    }
}

/// Filter executor trait for video filters.
pub trait VideoFilter: Send + Sync {
    /// Get filter name.
    fn name(&self) -> &str;

    /// Configure the filter with input format and return output format.
    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat>;

    /// Process a frame.
    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>>;

    /// Flush any buffered frames (for temporal filters).
    fn flush(&mut self) -> Result<Vec<VideoFrame>> {
        Ok(Vec::new())
    }

    /// Reset filter state.
    fn reset(&mut self) {}
}

/// Filter executor trait for audio filters.
pub trait AudioFilter: Send + Sync {
    /// Get filter name.
    fn name(&self) -> &str;

    /// Configure the filter with input format and return output format.
    fn configure(&mut self, input: &AudioFormat) -> Result<AudioFormat>;

    /// Process audio samples.
    fn process(&mut self, input: AudioSamples) -> Result<Vec<AudioSamples>>;

    /// Flush any buffered samples.
    fn flush(&mut self) -> Result<Vec<AudioSamples>> {
        Ok(Vec::new())
    }

    /// Reset filter state.
    fn reset(&mut self) {}
}

/// Registry of filter implementations.
#[derive(Default)]
pub struct FilterRegistry {
    video_factories: HashMap<String, Box<dyn Fn(&Filter) -> Result<Box<dyn VideoFilter>> + Send + Sync>>,
    audio_factories: HashMap<String, Box<dyn Fn(&Filter) -> Result<Box<dyn AudioFilter>> + Send + Sync>>,
}

impl FilterRegistry {
    /// Create a new filter registry.
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_builtin_filters();
        registry
    }

    /// Register built-in filters.
    fn register_builtin_filters(&mut self) {
        // Register core video filters
        self.register_video_filter("null", |_| Ok(Box::new(NullVideoFilter)));
        self.register_video_filter("scale", |f| ScaleFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("fps", |f| FpsFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("format", |f| FormatFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("crop", |f| CropFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("pad", |f| PadFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));

        // Register enhancement video filters
        self.register_video_filter("eq", |f| EqFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("unsharp", |f| UnsharpFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("noise", |f| NoiseFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("overlay", |f| OverlayFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("drawtext", |f| DrawtextFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));

        // Register transform video filters
        self.register_video_filter("transpose", |f| TransposeFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("hflip", |f| HflipFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("vflip", |f| VflipFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));
        self.register_video_filter("yadif", |f| DeinterlaceFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn VideoFilter>));

        // Register audio filters
        self.register_audio_filter("anull", |_| Ok(Box::new(NullAudioFilter)));
        self.register_audio_filter("volume", |f| VolumeFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn AudioFilter>));
        self.register_audio_filter("aresample", |f| AresampleFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn AudioFilter>));
        self.register_audio_filter("atempo", |f| AtempoFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn AudioFilter>));
        self.register_audio_filter("aformat", |f| AformatFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn AudioFilter>));
        self.register_audio_filter("loudnorm", |f| LoudnormFilter::from_filter(f).map(|f| Box::new(f) as Box<dyn AudioFilter>));
    }

    /// Register a video filter factory.
    pub fn register_video_filter<F>(&mut self, name: &str, factory: F)
    where
        F: Fn(&Filter) -> Result<Box<dyn VideoFilter>> + Send + Sync + 'static,
    {
        self.video_factories.insert(name.to_string(), Box::new(factory));
    }

    /// Register an audio filter factory.
    pub fn register_audio_filter<F>(&mut self, name: &str, factory: F)
    where
        F: Fn(&Filter) -> Result<Box<dyn AudioFilter>> + Send + Sync + 'static,
    {
        self.audio_factories.insert(name.to_string(), Box::new(factory));
    }

    /// Create a video filter from a parsed Filter.
    pub fn create_video_filter(&self, filter: &Filter) -> Result<Box<dyn VideoFilter>> {
        let factory = self.video_factories.get(&filter.name).ok_or_else(|| {
            CompatError::UnsupportedFilter(format!("unknown video filter: {}", filter.name))
        })?;
        factory(filter)
    }

    /// Create an audio filter from a parsed Filter.
    pub fn create_audio_filter(&self, filter: &Filter) -> Result<Box<dyn AudioFilter>> {
        let factory = self.audio_factories.get(&filter.name).ok_or_else(|| {
            CompatError::UnsupportedFilter(format!("unknown audio filter: {}", filter.name))
        })?;
        factory(filter)
    }
}

/// Video filter pipeline.
pub struct VideoFilterPipeline {
    filters: Vec<Box<dyn VideoFilter>>,
    configured: bool,
}

impl VideoFilterPipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            configured: false,
        }
    }

    /// Create pipeline from a filter chain.
    pub fn from_chain(chain: &FilterChain, registry: &FilterRegistry) -> Result<Self> {
        let mut pipeline = Self::new();
        for filter in &chain.filters {
            pipeline.filters.push(registry.create_video_filter(filter)?);
        }
        Ok(pipeline)
    }

    /// Add a filter to the pipeline.
    pub fn push(&mut self, filter: Box<dyn VideoFilter>) {
        self.filters.push(filter);
        self.configured = false;
    }

    /// Configure the pipeline with input format.
    pub fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        let mut current_format = input.clone();
        for filter in &mut self.filters {
            current_format = filter.configure(&current_format)?;
        }
        self.configured = true;
        Ok(current_format)
    }

    /// Process a frame through the pipeline.
    pub fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        if !self.configured {
            return Err(CompatError::InvalidFilter("pipeline not configured".to_string()));
        }

        let mut frames = vec![input];
        for filter in &mut self.filters {
            let mut output_frames = Vec::new();
            for frame in frames {
                output_frames.extend(filter.process(frame)?);
            }
            frames = output_frames;
        }
        Ok(frames)
    }

    /// Flush all filters.
    pub fn flush(&mut self) -> Result<Vec<VideoFrame>> {
        let mut frames = Vec::new();
        for filter in &mut self.filters {
            frames.extend(filter.flush()?);
        }
        Ok(frames)
    }

    /// Reset all filters.
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
        self.configured = false;
    }
}

impl Default for VideoFilterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio filter pipeline.
pub struct AudioFilterPipeline {
    filters: Vec<Box<dyn AudioFilter>>,
    configured: bool,
}

impl AudioFilterPipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            configured: false,
        }
    }

    /// Create pipeline from a filter chain.
    pub fn from_chain(chain: &FilterChain, registry: &FilterRegistry) -> Result<Self> {
        let mut pipeline = Self::new();
        for filter in &chain.filters {
            pipeline.filters.push(registry.create_audio_filter(filter)?);
        }
        Ok(pipeline)
    }

    /// Add a filter to the pipeline.
    pub fn push(&mut self, filter: Box<dyn AudioFilter>) {
        self.filters.push(filter);
        self.configured = false;
    }

    /// Configure the pipeline with input format.
    pub fn configure(&mut self, input: &AudioFormat) -> Result<AudioFormat> {
        let mut current_format = input.clone();
        for filter in &mut self.filters {
            current_format = filter.configure(&current_format)?;
        }
        self.configured = true;
        Ok(current_format)
    }

    /// Process samples through the pipeline.
    pub fn process(&mut self, input: AudioSamples) -> Result<Vec<AudioSamples>> {
        if !self.configured {
            return Err(CompatError::InvalidFilter("pipeline not configured".to_string()));
        }

        let mut samples = vec![input];
        for filter in &mut self.filters {
            let mut output_samples = Vec::new();
            for sample in samples {
                output_samples.extend(filter.process(sample)?);
            }
            samples = output_samples;
        }
        Ok(samples)
    }

    /// Flush all filters.
    pub fn flush(&mut self) -> Result<Vec<AudioSamples>> {
        let mut samples = Vec::new();
        for filter in &mut self.filters {
            samples.extend(filter.flush()?);
        }
        Ok(samples)
    }

    /// Reset all filters.
    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
        self.configured = false;
    }
}

impl Default for AudioFilterPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Built-in Filter Implementations
// =============================================================================

/// Null (passthrough) video filter.
pub struct NullVideoFilter;

impl VideoFilter for NullVideoFilter {
    fn name(&self) -> &str {
        "null"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        Ok(vec![input])
    }
}

/// Null (passthrough) audio filter.
pub struct NullAudioFilter;

impl AudioFilter for NullAudioFilter {
    fn name(&self) -> &str {
        "anull"
    }

    fn configure(&mut self, input: &AudioFormat) -> Result<AudioFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: AudioSamples) -> Result<Vec<AudioSamples>> {
        Ok(vec![input])
    }
}

/// Scale video filter (placeholder).
pub struct ScaleFilter {
    target_width: Option<u32>,
    target_height: Option<u32>,
    output_format: VideoFormat,
}

impl ScaleFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        let width = filter.get_positional(0)
            .or_else(|| filter.get_param("w"))
            .and_then(|s| s.parse().ok());
        let height = filter.get_positional(1)
            .or_else(|| filter.get_param("h"))
            .and_then(|s| s.parse().ok());

        Ok(Self {
            target_width: width,
            target_height: height,
            output_format: VideoFormat::default(),
        })
    }
}

impl VideoFilter for ScaleFilter {
    fn name(&self) -> &str {
        "scale"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        let width = self.target_width.unwrap_or(input.width);
        let height = self.target_height.unwrap_or(input.height);

        self.output_format = VideoFormat {
            width,
            height,
            pixel_format: input.pixel_format,
            fps_num: input.fps_num,
            fps_den: input.fps_den,
            sar_num: 1,
            sar_den: 1,
            full_range: input.full_range,
        };

        Ok(self.output_format.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would perform actual scaling
        // For now, just create a new frame with the output format
        let mut output = VideoFrame::new(self.output_format.clone());
        output.pts = input.pts;
        output.keyframe = input.keyframe;
        Ok(vec![output])
    }
}

/// FPS (frame rate) filter (placeholder).
pub struct FpsFilter {
    target_fps: f64,
    output_format: VideoFormat,
}

impl FpsFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        let fps = filter.get_positional(0)
            .or_else(|| filter.get_param("fps"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(30.0);

        Ok(Self {
            target_fps: fps,
            output_format: VideoFormat::default(),
        })
    }
}

impl VideoFilter for FpsFilter {
    fn name(&self) -> &str {
        "fps"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        // Simple rational approximation for fps
        let (num, den) = approximate_fraction(self.target_fps);

        self.output_format = VideoFormat {
            fps_num: num,
            fps_den: den,
            ..input.clone()
        };

        Ok(self.output_format.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would perform actual frame rate conversion
        Ok(vec![input])
    }
}

/// Format (pixel format) filter.
pub struct FormatFilter {
    target_format: PixelFormat,
    output_format: VideoFormat,
}

impl FormatFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        let format_str = filter.get_positional(0)
            .or_else(|| filter.get_param("pix_fmts"))
            .ok_or_else(|| CompatError::InvalidFilter("format filter requires pixel format".to_string()))?;

        let target_format = PixelFormat::from_str(format_str)
            .ok_or_else(|| CompatError::InvalidFilter(format!("unknown pixel format: {}", format_str)))?;

        Ok(Self {
            target_format,
            output_format: VideoFormat::default(),
        })
    }
}

impl VideoFilter for FormatFilter {
    fn name(&self) -> &str {
        "format"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        self.output_format = VideoFormat {
            pixel_format: self.target_format,
            ..input.clone()
        };
        Ok(self.output_format.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would perform actual pixel format conversion
        let mut output = VideoFrame::new(self.output_format.clone());
        output.pts = input.pts;
        output.keyframe = input.keyframe;
        Ok(vec![output])
    }
}

/// Crop video filter (placeholder).
pub struct CropFilter {
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    output_format: VideoFormat,
}

impl CropFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        let width = filter.get_param("w")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| CompatError::InvalidFilter("crop filter requires width".to_string()))?;
        let height = filter.get_param("h")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| CompatError::InvalidFilter("crop filter requires height".to_string()))?;
        let x = filter.get_param("x")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let y = filter.get_param("y")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        Ok(Self {
            width,
            height,
            x,
            y,
            output_format: VideoFormat::default(),
        })
    }
}

impl VideoFilter for CropFilter {
    fn name(&self) -> &str {
        "crop"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        if self.x + self.width > input.width || self.y + self.height > input.height {
            return Err(CompatError::InvalidFilter("crop region exceeds frame size".to_string()));
        }

        self.output_format = VideoFormat {
            width: self.width,
            height: self.height,
            ..input.clone()
        };
        Ok(self.output_format.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would perform actual cropping
        let mut output = VideoFrame::new(self.output_format.clone());
        output.pts = input.pts;
        output.keyframe = input.keyframe;
        Ok(vec![output])
    }
}

/// Pad video filter - add borders around the video.
pub struct PadFilter {
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    output_format: VideoFormat,
}

impl PadFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        let width = filter.get_positional(0)
            .or_else(|| filter.get_param("w"))
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| CompatError::InvalidFilter("pad filter requires width".to_string()))?;
        let height = filter.get_positional(1)
            .or_else(|| filter.get_param("h"))
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| CompatError::InvalidFilter("pad filter requires height".to_string()))?;
        let x = filter.get_positional(2)
            .or_else(|| filter.get_param("x"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let y = filter.get_positional(3)
            .or_else(|| filter.get_param("y"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        Ok(Self {
            width,
            height,
            x,
            y,
            output_format: VideoFormat::default(),
        })
    }
}

impl VideoFilter for PadFilter {
    fn name(&self) -> &str {
        "pad"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        if self.width < input.width || self.height < input.height {
            return Err(CompatError::InvalidFilter("pad size must be larger than input".to_string()));
        }

        self.output_format = VideoFormat {
            width: self.width,
            height: self.height,
            ..input.clone()
        };
        Ok(self.output_format.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would add padding around the frame
        let mut output = VideoFrame::new(self.output_format.clone());
        output.pts = input.pts;
        output.keyframe = input.keyframe;
        Ok(vec![output])
    }
}

/// EQ (equalizer) filter for brightness, contrast, saturation adjustment.
pub struct EqFilter {
    brightness: f64,
    contrast: f64,
    saturation: f64,
    gamma: f64,
}

impl EqFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        Ok(Self {
            brightness: filter.get_param("brightness")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            contrast: filter.get_param("contrast")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            saturation: filter.get_param("saturation")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            gamma: filter.get_param("gamma")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
        })
    }
}

impl VideoFilter for EqFilter {
    fn name(&self) -> &str {
        "eq"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would apply color adjustments
        Ok(vec![input])
    }
}

/// Unsharp masking filter for sharpening/blurring.
pub struct UnsharpFilter {
    luma_x: i32,
    luma_y: i32,
    luma_amount: f64,
    chroma_x: i32,
    chroma_y: i32,
    chroma_amount: f64,
}

impl UnsharpFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        Ok(Self {
            luma_x: filter.get_positional(0)
                .or_else(|| filter.get_param("lx"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
            luma_y: filter.get_positional(1)
                .or_else(|| filter.get_param("ly"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
            luma_amount: filter.get_positional(2)
                .or_else(|| filter.get_param("la"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            chroma_x: filter.get_positional(3)
                .or_else(|| filter.get_param("cx"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
            chroma_y: filter.get_positional(4)
                .or_else(|| filter.get_param("cy"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
            chroma_amount: filter.get_positional(5)
                .or_else(|| filter.get_param("ca"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
        })
    }
}

impl VideoFilter for UnsharpFilter {
    fn name(&self) -> &str {
        "unsharp"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would apply unsharp mask
        Ok(vec![input])
    }
}

/// Noise filter for adding or reducing noise.
pub struct NoiseFilter {
    /// Luma noise strength.
    luma_strength: i32,
    /// Chroma noise strength.
    chroma_strength: i32,
    /// Noise type: uniform, gaussian, averaged.
    noise_type: String,
}

impl NoiseFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        Ok(Self {
            luma_strength: filter.get_param("alls")
                .or_else(|| filter.get_param("c0s"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            chroma_strength: filter.get_param("c1s")
                .or_else(|| filter.get_param("c2s"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            noise_type: filter.get_param("allf")
                .or_else(|| filter.get_param("c0f"))
                .unwrap_or("u")
                .to_string(),
        })
    }
}

impl VideoFilter for NoiseFilter {
    fn name(&self) -> &str {
        "noise"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would apply noise
        Ok(vec![input])
    }
}

/// Overlay filter for compositing frames.
pub struct OverlayFilter {
    x: i32,
    y: i32,
    output_format: VideoFormat,
}

impl OverlayFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        Ok(Self {
            x: filter.get_positional(0)
                .or_else(|| filter.get_param("x"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            y: filter.get_positional(1)
                .or_else(|| filter.get_param("y"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            output_format: VideoFormat::default(),
        })
    }
}

impl VideoFilter for OverlayFilter {
    fn name(&self) -> &str {
        "overlay"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        // Overlay takes two inputs - using main input format as output
        self.output_format = input.clone();
        Ok(self.output_format.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would composite overlay onto main frame
        // Real implementation needs two inputs
        Ok(vec![input])
    }
}

/// Drawtext filter for text overlay.
pub struct DrawtextFilter {
    text: String,
    x: String,
    y: String,
    fontsize: u32,
    fontcolor: String,
}

impl DrawtextFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        Ok(Self {
            text: filter.get_param("text")
                .unwrap_or("").to_string(),
            x: filter.get_param("x")
                .unwrap_or("0").to_string(),
            y: filter.get_param("y")
                .unwrap_or("0").to_string(),
            fontsize: filter.get_param("fontsize")
                .and_then(|s| s.parse().ok())
                .unwrap_or(16),
            fontcolor: filter.get_param("fontcolor")
                .unwrap_or("white").to_string(),
        })
    }
}

impl VideoFilter for DrawtextFilter {
    fn name(&self) -> &str {
        "drawtext"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        // Placeholder: would draw text on frame
        Ok(vec![input])
    }
}

/// Transpose filter for rotation/flipping.
pub struct TransposeFilter {
    /// Direction: 0=90ccw+vflip, 1=90cw, 2=90ccw, 3=90cw+vflip
    direction: u32,
    output_format: VideoFormat,
}

impl TransposeFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        Ok(Self {
            direction: filter.get_positional(0)
                .or_else(|| filter.get_param("dir"))
                .and_then(|s| s.parse().ok())
                .unwrap_or(1),
            output_format: VideoFormat::default(),
        })
    }
}

impl VideoFilter for TransposeFilter {
    fn name(&self) -> &str {
        "transpose"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        // Transpose swaps width and height
        self.output_format = VideoFormat {
            width: input.height,
            height: input.width,
            ..input.clone()
        };
        Ok(self.output_format.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        let mut output = VideoFrame::new(self.output_format.clone());
        output.pts = input.pts;
        output.keyframe = input.keyframe;
        Ok(vec![output])
    }
}

/// Hflip filter for horizontal flip.
pub struct HflipFilter;

impl HflipFilter {
    fn from_filter(_filter: &Filter) -> Result<Self> {
        Ok(Self)
    }
}

impl VideoFilter for HflipFilter {
    fn name(&self) -> &str {
        "hflip"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        Ok(vec![input])
    }
}

/// Vflip filter for vertical flip.
pub struct VflipFilter;

impl VflipFilter {
    fn from_filter(_filter: &Filter) -> Result<Self> {
        Ok(Self)
    }
}

impl VideoFilter for VflipFilter {
    fn name(&self) -> &str {
        "vflip"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        Ok(vec![input])
    }
}

/// Deinterlace filter.
pub struct DeinterlaceFilter {
    mode: String,
}

impl DeinterlaceFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        Ok(Self {
            mode: filter.get_param("mode")
                .unwrap_or("send_frame").to_string(),
        })
    }
}

impl VideoFilter for DeinterlaceFilter {
    fn name(&self) -> &str {
        "yadif"
    }

    fn configure(&mut self, input: &VideoFormat) -> Result<VideoFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: VideoFrame) -> Result<Vec<VideoFrame>> {
        Ok(vec![input])
    }
}

/// Aresample filter for audio resampling.
pub struct AresampleFilter {
    sample_rate: u32,
    output_format: AudioFormat,
}

impl AresampleFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        let sample_rate = filter.get_positional(0)
            .and_then(|s| s.parse().ok())
            .unwrap_or(48000);

        Ok(Self {
            sample_rate,
            output_format: AudioFormat::default(),
        })
    }
}

impl AudioFilter for AresampleFilter {
    fn name(&self) -> &str {
        "aresample"
    }

    fn configure(&mut self, input: &AudioFormat) -> Result<AudioFormat> {
        self.output_format = AudioFormat {
            sample_rate: self.sample_rate,
            channels: input.channels,
            sample_format: input.sample_format,
            channel_layout: input.channel_layout.clone(),
        };
        Ok(self.output_format.clone())
    }

    fn process(&mut self, input: AudioSamples) -> Result<Vec<AudioSamples>> {
        // Placeholder: would perform actual resampling
        // Estimate output samples based on rate ratio
        let ratio = self.output_format.sample_rate as f64 / input.format.sample_rate as f64;
        let out_samples = (input.num_samples as f64 * ratio) as usize;
        let mut output = AudioSamples::new(self.output_format.clone(), out_samples);
        output.pts = input.pts;
        Ok(vec![output])
    }
}

/// Atempo filter for audio tempo adjustment.
pub struct AtempoFilter {
    tempo: f64,
}

impl AtempoFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        let tempo = filter.get_positional(0)
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);

        if !(0.5..=100.0).contains(&tempo) {
            return Err(CompatError::InvalidFilter("atempo must be between 0.5 and 100.0".to_string()));
        }

        Ok(Self { tempo })
    }
}

impl AudioFilter for AtempoFilter {
    fn name(&self) -> &str {
        "atempo"
    }

    fn configure(&mut self, input: &AudioFormat) -> Result<AudioFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: AudioSamples) -> Result<Vec<AudioSamples>> {
        // Placeholder: would adjust tempo
        Ok(vec![input])
    }
}

/// Aformat filter for audio format conversion.
pub struct AformatFilter {
    sample_format: SampleFormat,
    sample_rate: Option<u32>,
    channel_layout: Option<String>,
    output_format: AudioFormat,
}

impl AformatFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        let sample_fmt_str = filter.get_param("sample_fmts")
            .unwrap_or("s16");
        let sample_format = SampleFormat::from_str(sample_fmt_str)
            .ok_or_else(|| CompatError::InvalidFilter(format!("unknown sample format: {}", sample_fmt_str)))?;

        Ok(Self {
            sample_format,
            sample_rate: filter.get_param("sample_rates")
                .and_then(|s| s.parse().ok()),
            channel_layout: filter.get_param("channel_layouts")
                .map(|s| s.to_string()),
            output_format: AudioFormat::default(),
        })
    }
}

impl AudioFilter for AformatFilter {
    fn name(&self) -> &str {
        "aformat"
    }

    fn configure(&mut self, input: &AudioFormat) -> Result<AudioFormat> {
        self.output_format = AudioFormat {
            sample_format: self.sample_format,
            sample_rate: self.sample_rate.unwrap_or(input.sample_rate),
            channels: input.channels,
            channel_layout: self.channel_layout.clone().unwrap_or_else(|| input.channel_layout.clone()),
        };
        Ok(self.output_format.clone())
    }

    fn process(&mut self, input: AudioSamples) -> Result<Vec<AudioSamples>> {
        let mut output = AudioSamples::new(self.output_format.clone(), input.num_samples);
        output.pts = input.pts;
        Ok(vec![output])
    }
}

/// Loudnorm filter for EBU R128 loudness normalization.
pub struct LoudnormFilter {
    integrated: f64,
    true_peak: f64,
    lra: f64,
}

impl LoudnormFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        Ok(Self {
            integrated: filter.get_param("I")
                .and_then(|s| s.parse().ok())
                .unwrap_or(-24.0),
            true_peak: filter.get_param("TP")
                .and_then(|s| s.parse().ok())
                .unwrap_or(-2.0),
            lra: filter.get_param("LRA")
                .and_then(|s| s.parse().ok())
                .unwrap_or(7.0),
        })
    }
}

impl AudioFilter for LoudnormFilter {
    fn name(&self) -> &str {
        "loudnorm"
    }

    fn configure(&mut self, input: &AudioFormat) -> Result<AudioFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: AudioSamples) -> Result<Vec<AudioSamples>> {
        // Placeholder: would apply loudness normalization
        Ok(vec![input])
    }
}

/// Volume audio filter.
pub struct VolumeFilter {
    volume: f64,
}

impl VolumeFilter {
    fn from_filter(filter: &Filter) -> Result<Self> {
        let volume_str = filter.get_positional(0)
            .or_else(|| filter.get_param("volume"))
            .unwrap_or("1.0");

        // Handle dB notation
        let volume = if volume_str.ends_with("dB") {
            let db: f64 = volume_str[..volume_str.len() - 2].parse()
                .map_err(|_| CompatError::InvalidFilter(format!("invalid dB value: {}", volume_str)))?;
            10_f64.powf(db / 20.0)
        } else {
            volume_str.parse()
                .map_err(|_| CompatError::InvalidFilter(format!("invalid volume value: {}", volume_str)))?
        };

        Ok(Self { volume })
    }
}

impl AudioFilter for VolumeFilter {
    fn name(&self) -> &str {
        "volume"
    }

    fn configure(&mut self, input: &AudioFormat) -> Result<AudioFormat> {
        Ok(input.clone())
    }

    fn process(&mut self, input: AudioSamples) -> Result<Vec<AudioSamples>> {
        // Placeholder: would apply volume adjustment to samples
        Ok(vec![input])
    }
}

/// Approximate a floating-point value as a fraction.
fn approximate_fraction(value: f64) -> (u32, u32) {
    // Common frame rates
    let common_fps = [
        (24000, 1001, 23.976),
        (24, 1, 24.0),
        (25, 1, 25.0),
        (30000, 1001, 29.97),
        (30, 1, 30.0),
        (50, 1, 50.0),
        (60000, 1001, 59.94),
        (60, 1, 60.0),
    ];

    // Check for common frame rates
    for (num, den, fps) in common_fps {
        if (value - fps).abs() < 0.01 {
            return (num, den);
        }
    }

    // Default to simple approximation
    (value.round() as u32, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterChain;

    #[test]
    fn test_video_format() {
        let format = VideoFormat::new(1920, 1080, PixelFormat::Yuv420p)
            .with_fps(30, 1);
        assert_eq!(format.width, 1920);
        assert_eq!(format.height, 1080);
        assert_eq!(format.fps(), 30.0);
    }

    #[test]
    fn test_pixel_format_parse() {
        assert_eq!(PixelFormat::from_str("yuv420p"), Some(PixelFormat::Yuv420p));
        assert_eq!(PixelFormat::from_str("rgb24"), Some(PixelFormat::Rgb24));
        assert_eq!(PixelFormat::from_str("unknown"), None);
    }

    #[test]
    fn test_filter_registry() {
        let registry = FilterRegistry::new();
        let filter = Filter::parse("scale=1920:1080").unwrap();
        let scale = registry.create_video_filter(&filter);
        assert!(scale.is_ok());
    }

    #[test]
    fn test_video_pipeline() {
        let registry = FilterRegistry::new();
        let chain = FilterChain::parse("scale=1280:720,format=yuv420p").unwrap();
        let mut pipeline = VideoFilterPipeline::from_chain(&chain, &registry).unwrap();

        let input_format = VideoFormat::new(1920, 1080, PixelFormat::Yuv444p);
        let output_format = pipeline.configure(&input_format).unwrap();

        assert_eq!(output_format.width, 1280);
        assert_eq!(output_format.height, 720);
        assert_eq!(output_format.pixel_format, PixelFormat::Yuv420p);
    }

    #[test]
    fn test_volume_db_parsing() {
        let filter = Filter::parse("volume=-3dB").unwrap();
        let volume = VolumeFilter::from_filter(&filter).unwrap();
        assert!((volume.volume - 0.708).abs() < 0.01); // -3dB ≈ 0.708
    }
}
