//! Transcoding options for WASM.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Video codec options.
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoOptions {
    /// Video codec name (e.g., "h264", "av1").
    codec: String,
    /// Target width (0 for original).
    width: u32,
    /// Target height (0 for original).
    height: u32,
    /// Target bitrate in bits per second.
    bitrate: u32,
    /// Target framerate (0 for original).
    framerate: f32,
    /// Quality preset (0-9, where 0 is fastest and 9 is highest quality).
    quality: u8,
    /// Keyframe interval.
    keyframe_interval: u32,
}

#[wasm_bindgen]
impl VideoOptions {
    /// Create new video options with defaults.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set video codec.
    #[wasm_bindgen(js_name = withCodec)]
    pub fn with_codec(mut self, codec: &str) -> Self {
        self.codec = codec.to_string();
        self
    }

    /// Set resolution.
    #[wasm_bindgen(js_name = withResolution)]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set bitrate.
    #[wasm_bindgen(js_name = withBitrate)]
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set framerate.
    #[wasm_bindgen(js_name = withFramerate)]
    pub fn with_framerate(mut self, framerate: f32) -> Self {
        self.framerate = framerate;
        self
    }

    /// Set quality preset.
    #[wasm_bindgen(js_name = withQuality)]
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality.min(9);
        self
    }

    /// Set keyframe interval.
    #[wasm_bindgen(js_name = withKeyframeInterval)]
    pub fn with_keyframe_interval(mut self, interval: u32) -> Self {
        self.keyframe_interval = interval;
        self
    }

    // Getters
    #[wasm_bindgen(getter)]
    pub fn codec(&self) -> String {
        self.codec.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[wasm_bindgen(getter)]
    pub fn bitrate(&self) -> u32 {
        self.bitrate
    }

    #[wasm_bindgen(getter)]
    pub fn framerate(&self) -> f32 {
        self.framerate
    }

    #[wasm_bindgen(getter)]
    pub fn quality(&self) -> u8 {
        self.quality
    }

    #[wasm_bindgen(getter, js_name = keyframeInterval)]
    pub fn keyframe_interval(&self) -> u32 {
        self.keyframe_interval
    }
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            codec: "h264".to_string(),
            width: 0,
            height: 0,
            bitrate: 2_000_000,
            framerate: 0.0,
            quality: 5,
            keyframe_interval: 30,
        }
    }
}

/// Audio codec options.
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOptions {
    /// Audio codec name (e.g., "aac", "mp3").
    codec: String,
    /// Target bitrate in bits per second.
    bitrate: u32,
    /// Sample rate (0 for original).
    sample_rate: u32,
    /// Number of channels (0 for original).
    channels: u8,
}

#[wasm_bindgen]
impl AudioOptions {
    /// Create new audio options with defaults.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set audio codec.
    #[wasm_bindgen(js_name = withCodec)]
    pub fn with_codec(mut self, codec: &str) -> Self {
        self.codec = codec.to_string();
        self
    }

    /// Set bitrate.
    #[wasm_bindgen(js_name = withBitrate)]
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set sample rate.
    #[wasm_bindgen(js_name = withSampleRate)]
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set number of channels.
    #[wasm_bindgen(js_name = withChannels)]
    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = channels;
        self
    }

    // Getters
    #[wasm_bindgen(getter)]
    pub fn codec(&self) -> String {
        self.codec.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn bitrate(&self) -> u32 {
        self.bitrate
    }

    #[wasm_bindgen(getter, js_name = sampleRate)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> u8 {
        self.channels
    }
}

impl Default for AudioOptions {
    fn default() -> Self {
        Self {
            codec: "aac".to_string(),
            bitrate: 128_000,
            sample_rate: 0,
            channels: 0,
        }
    }
}

/// Complete transcoding options.
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeOptions {
    /// Video options (None to copy or disable).
    video: Option<VideoOptions>,
    /// Audio options (None to copy or disable).
    audio: Option<AudioOptions>,
    /// Output container format.
    container: String,
    /// Enable hardware acceleration if available.
    hardware_acceleration: bool,
    /// Number of worker threads.
    threads: u8,
}

#[wasm_bindgen]
impl TranscodeOptions {
    /// Create new transcode options with defaults.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set video options.
    #[wasm_bindgen(js_name = withVideo)]
    pub fn with_video(mut self, video: VideoOptions) -> Self {
        self.video = Some(video);
        self
    }

    /// Set audio options.
    #[wasm_bindgen(js_name = withAudio)]
    pub fn with_audio(mut self, audio: AudioOptions) -> Self {
        self.audio = Some(audio);
        self
    }

    /// Disable video.
    #[wasm_bindgen(js_name = withoutVideo)]
    pub fn without_video(mut self) -> Self {
        self.video = None;
        self
    }

    /// Disable audio.
    #[wasm_bindgen(js_name = withoutAudio)]
    pub fn without_audio(mut self) -> Self {
        self.audio = None;
        self
    }

    /// Set output container format.
    #[wasm_bindgen(js_name = withContainer)]
    pub fn with_container(mut self, container: &str) -> Self {
        self.container = container.to_string();
        self
    }

    /// Set video codec (convenience method).
    #[wasm_bindgen(js_name = withVideoCodec)]
    pub fn with_video_codec(mut self, codec: &str) -> Self {
        if let Some(ref mut video) = self.video {
            video.codec = codec.to_string();
        } else {
            self.video = Some(VideoOptions::new().with_codec(codec));
        }
        self
    }

    /// Set video bitrate (convenience method).
    #[wasm_bindgen(js_name = withVideoBitrate)]
    pub fn with_video_bitrate(mut self, bitrate: u32) -> Self {
        if let Some(ref mut video) = self.video {
            video.bitrate = bitrate;
        } else {
            self.video = Some(VideoOptions::new().with_bitrate(bitrate));
        }
        self
    }

    /// Set resolution (convenience method).
    #[wasm_bindgen(js_name = withResolution)]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        if let Some(ref mut video) = self.video {
            video.width = width;
            video.height = height;
        } else {
            self.video = Some(VideoOptions::new().with_resolution(width, height));
        }
        self
    }

    /// Enable or disable hardware acceleration.
    #[wasm_bindgen(js_name = withHardwareAcceleration)]
    pub fn with_hardware_acceleration(mut self, enabled: bool) -> Self {
        self.hardware_acceleration = enabled;
        self
    }

    /// Set number of worker threads.
    #[wasm_bindgen(js_name = withThreads)]
    pub fn with_threads(mut self, threads: u8) -> Self {
        self.threads = threads;
        self
    }

    // Getters
    #[wasm_bindgen(getter)]
    pub fn container(&self) -> String {
        self.container.clone()
    }

    #[wasm_bindgen(getter, js_name = hardwareAcceleration)]
    pub fn hardware_acceleration(&self) -> bool {
        self.hardware_acceleration
    }

    #[wasm_bindgen(getter)]
    pub fn threads(&self) -> u8 {
        self.threads
    }

    /// Check if video is enabled.
    #[wasm_bindgen(js_name = hasVideo)]
    pub fn has_video(&self) -> bool {
        self.video.is_some()
    }

    /// Check if audio is enabled.
    #[wasm_bindgen(js_name = hasAudio)]
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }
}

impl Default for TranscodeOptions {
    fn default() -> Self {
        Self {
            video: Some(VideoOptions::default()),
            audio: Some(AudioOptions::default()),
            container: "mp4".to_string(),
            hardware_acceleration: false,
            threads: 4,
        }
    }
}
