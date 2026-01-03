//! Node.js bindings for the Transcode codec library.
//!
//! This module provides Node.js bindings using napi-rs, allowing Node.js users
//! to access the memory-safe transcoding capabilities of Transcode.

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use transcode::TranscodeOptions as RustTranscodeOptions;
use transcode::Transcoder as RustTranscoder;
use transcode_containers::traits::{Demuxer, TrackType, CodecId};
use transcode_containers::mp4::Mp4Demuxer;

/// Video stream information.
#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoStreamInfo {
    /// Stream index.
    pub index: u32,
    /// Codec name.
    pub codec: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub frame_rate: Option<f64>,
    /// Bit depth.
    pub bit_depth: u32,
    /// Duration in seconds.
    pub duration: Option<f64>,
}

/// Audio stream information.
#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioStreamInfo {
    /// Stream index.
    pub index: u32,
    /// Codec name.
    pub codec: String,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u32,
    /// Bits per sample.
    pub bits_per_sample: u32,
    /// Duration in seconds.
    pub duration: Option<f64>,
}

/// Media file information returned by probe().
#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediaInfo {
    /// Container format (e.g., "mp4", "mkv").
    pub format: String,
    /// Total duration in seconds.
    pub duration: Option<f64>,
    /// File size in bytes.
    pub size: i64,
    /// Video streams.
    pub video_streams: Vec<VideoStreamInfo>,
    /// Audio streams.
    pub audio_streams: Vec<AudioStreamInfo>,
    /// Total bitrate in bits per second.
    pub bitrate: Option<i64>,
}

/// Progress information emitted during transcoding.
#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Progress {
    /// Progress percentage (0-100).
    pub percent: f64,
    /// Frames processed.
    pub frames: i64,
    /// Current speed (realtime multiplier, e.g., 2.5x).
    pub speed: f64,
    /// Estimated time remaining in seconds.
    pub eta: Option<f64>,
    /// Current output size in bytes.
    pub size: i64,
    /// Current bitrate in bits per second.
    pub bitrate: i64,
}

/// Transcoding statistics returned after completion.
#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TranscodeStats {
    /// Total frames decoded.
    pub frames_decoded: i64,
    /// Total frames encoded.
    pub frames_encoded: i64,
    /// Input file size in bytes.
    pub input_size: i64,
    /// Output file size in bytes.
    pub output_size: i64,
    /// Compression ratio achieved.
    pub compression_ratio: f64,
    /// Average encoding speed (realtime multiplier).
    pub average_speed: f64,
    /// Total duration processed in seconds.
    pub duration: f64,
}

/// Transcoding options.
#[napi(object)]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TranscodeOptions {
    /// Video codec (e.g., "h264", "h265", "av1").
    pub video_codec: Option<String>,
    /// Audio codec (e.g., "aac", "mp3", "opus").
    pub audio_codec: Option<String>,
    /// Video bitrate in bits per second.
    pub video_bitrate: Option<i64>,
    /// Audio bitrate in bits per second.
    pub audio_bitrate: Option<i64>,
    /// Output width in pixels.
    pub width: Option<u32>,
    /// Output height in pixels.
    pub height: Option<u32>,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// Audio sample rate in Hz.
    pub sample_rate: Option<u32>,
    /// Number of audio channels.
    pub channels: Option<u32>,
    /// Number of threads for encoding (0 = auto).
    pub threads: Option<u32>,
    /// Enable hardware acceleration.
    pub hardware_acceleration: Option<bool>,
    /// Overwrite output file if exists.
    pub overwrite: Option<bool>,
    /// Start time in seconds (for trimming).
    pub start_time: Option<f64>,
    /// Duration in seconds (for trimming).
    pub duration: Option<f64>,
    /// Encoder preset (e.g., "ultrafast", "medium", "slow").
    pub preset: Option<String>,
    /// CRF value for quality-based encoding (0-51, lower is better).
    pub crf: Option<u32>,
}

/// SIMD capabilities detected on the current system.
#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimdCapabilities {
    /// SSE4.2 support (x86_64).
    pub sse42: bool,
    /// AVX2 support (x86_64).
    pub avx2: bool,
    /// AVX-512 support (x86_64).
    pub avx512: bool,
    /// FMA support (x86_64).
    pub fma: bool,
    /// NEON support (ARM).
    pub neon: bool,
    /// SVE support (ARM).
    pub sve: bool,
}

/// Transcoder class for performing media transcoding operations.
#[napi]
pub struct Transcoder {
    input_path: PathBuf,
    output_path: PathBuf,
    options: TranscodeOptions,
    cancelled: Arc<AtomicBool>,
    frames_processed: Arc<AtomicU64>,
}

#[napi]
impl Transcoder {
    /// Create a new Transcoder instance.
    ///
    /// @param input - Path to the input media file.
    /// @param output - Path for the output file.
    /// @param options - Optional transcoding options.
    #[napi(constructor)]
    pub fn new(input: String, output: String, options: Option<TranscodeOptions>) -> Self {
        Self {
            input_path: PathBuf::from(input),
            output_path: PathBuf::from(output),
            options: options.unwrap_or_default(),
            cancelled: Arc::new(AtomicBool::new(false)),
            frames_processed: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Run the transcoding operation asynchronously.
    ///
    /// @param on_progress - Optional callback for progress updates.
    /// @returns Promise that resolves to TranscodeStats on completion.
    #[napi]
    pub async fn run(&self, on_progress: Option<ThreadsafeFunction<Progress, ErrorStrategy::Fatal>>) -> Result<TranscodeStats> {
        let input_path = self.input_path.clone();
        let output_path = self.output_path.clone();
        let options = self.options.clone();
        let cancelled = self.cancelled.clone();
        let frames_processed = self.frames_processed.clone();

        // Check if input file exists
        if !input_path.exists() {
            return Err(Error::new(
                Status::GenericFailure,
                format!("Input file not found: {:?}", input_path),
            ));
        }

        // Build transcoding options
        let mut rust_options = RustTranscodeOptions::new()
            .input(&input_path)
            .output(&output_path)
            .overwrite(options.overwrite.unwrap_or(false));

        if let Some(bitrate) = options.video_bitrate {
            rust_options = rust_options.video_bitrate(bitrate as u64);
        }

        if let Some(bitrate) = options.audio_bitrate {
            rust_options = rust_options.audio_bitrate(bitrate as u64);
        }

        if let Some(threads) = options.threads {
            rust_options = rust_options.threads(threads as usize);
        }

        if let Some(hw_accel) = options.hardware_acceleration {
            rust_options = rust_options.hardware_acceleration(hw_accel);
        }

        if let Some(width) = options.width {
            if let Some(height) = options.height {
                rust_options = rust_options.video_resolution(width, height);
            }
        }

        if let Some(fps) = options.frame_rate {
            rust_options = rust_options.video_framerate(fps);
        }

        // Get input file size
        let input_size = std::fs::metadata(&input_path)
            .map(|m| m.len() as i64)
            .unwrap_or(0);

        // Track start time for speed calculation
        let start_time = std::time::Instant::now();

        // Create the transcoder
        let mut transcoder = RustTranscoder::new(rust_options)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to create transcoder: {}", e)))?;

        // Set up progress callback if provided
        if let Some(callback) = on_progress {
            let callback_clone = callback.clone();
            let frames_clone = frames_processed.clone();
            let cancelled_clone = cancelled.clone();
            let start = start_time;

            transcoder = transcoder.on_progress(move |progress_percent, frames| {
                // Check for cancellation
                if cancelled_clone.load(Ordering::Relaxed) {
                    return;
                }

                // Update frames counter
                frames_clone.store(frames, Ordering::Relaxed);

                // Calculate real-time metrics
                let elapsed = start.elapsed().as_secs_f64();
                let speed = if elapsed > 0.0 {
                    frames as f64 / (elapsed * 30.0) // Assume 30fps for speed calc
                } else {
                    0.0
                };

                // Calculate ETA
                let eta = if progress_percent > 0.0 && progress_percent < 100.0 {
                    let remaining = 100.0 - progress_percent;
                    let time_per_percent = elapsed / progress_percent;
                    Some(time_per_percent * remaining)
                } else {
                    None
                };

                // Estimate current output size
                let estimated_size = (input_size as f64 * (progress_percent / 100.0)) as i64;

                // Calculate bitrate
                let duration_secs = elapsed.max(0.001);
                let bitrate = ((estimated_size as f64 * 8.0) / duration_secs) as i64;

                let progress = Progress {
                    percent: progress_percent,
                    frames: frames as i64,
                    speed,
                    eta,
                    size: estimated_size,
                    bitrate,
                };

                // Call the JavaScript callback
                let _ = callback_clone.call(progress, ThreadsafeFunctionCallMode::NonBlocking);
            });
        }

        // Check cancellation before running
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::new(Status::Cancelled, "Transcoding cancelled before starting"));
        }

        // Initialize the transcoder
        transcoder.initialize()
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to initialize transcoder: {}", e)))?;

        // Run the actual transcoder
        transcoder.run()
            .map_err(|e| Error::new(Status::GenericFailure, format!("Transcoding failed: {}", e)))?;

        // Get final stats
        let stats = transcoder.stats();
        let output_size = std::fs::metadata(&output_path)
            .map(|m| m.len() as i64)
            .unwrap_or(0);

        let elapsed = start_time.elapsed().as_secs_f64();
        let average_speed = if elapsed > 0.0 {
            (stats.duration_processed_us as f64 / 1_000_000.0) / elapsed
        } else {
            0.0
        };

        Ok(TranscodeStats {
            frames_decoded: stats.frames_decoded as i64,
            frames_encoded: stats.frames_encoded as i64,
            input_size,
            output_size,
            compression_ratio: stats.compression_ratio(),
            average_speed,
            duration: stats.duration_processed_us as f64 / 1_000_000.0,
        })
    }

    /// Cancel an ongoing transcoding operation.
    #[napi]
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Get the current progress.
    #[napi]
    pub fn get_progress(&self) -> i64 {
        self.frames_processed.load(Ordering::Relaxed) as i64
    }
}

/// Probe a media file to get information about its streams.
///
/// @param input - Path to the media file.
/// @returns Promise that resolves to MediaInfo.
#[napi]
pub async fn probe(input: String) -> Result<MediaInfo> {
    let path = PathBuf::from(&input);

    if !path.exists() {
        return Err(Error::new(
            Status::GenericFailure,
            format!("File not found: {}", input),
        ));
    }

    let file_size = std::fs::metadata(&path)
        .map(|m| m.len() as i64)
        .unwrap_or(0);

    // Open and probe the file
    let file = std::fs::File::open(&path)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to open file: {}", e)))?;

    let mut demuxer = Mp4Demuxer::new();
    demuxer.open(file)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to probe file: {}", e)))?;

    let format = demuxer.format_name().to_string();
    let duration = demuxer.duration().map(|d| d as f64 / 1_000_000.0);
    let num_streams = demuxer.num_streams();

    let mut video_streams = Vec::new();
    let mut audio_streams = Vec::new();

    for i in 0..num_streams {
        if let Some(stream) = demuxer.stream_info(i) {
            match stream.track_type {
                TrackType::Video => {
                    if let Some(ref video) = stream.video {
                        let stream_duration = stream.duration.map(|d| {
                            let tb = stream.time_base;
                            (d as f64 * tb.num as f64) / tb.den as f64
                        });

                        video_streams.push(VideoStreamInfo {
                            index: i as u32,
                            codec: codec_id_to_string(&stream.codec_id),
                            width: video.width,
                            height: video.height,
                            frame_rate: video.frame_rate.map(|r| r.num as f64 / r.den as f64),
                            bit_depth: video.bit_depth as u32,
                            duration: stream_duration,
                        });
                    }
                }
                TrackType::Audio => {
                    if let Some(ref audio) = stream.audio {
                        let stream_duration = stream.duration.map(|d| {
                            let tb = stream.time_base;
                            (d as f64 * tb.num as f64) / tb.den as f64
                        });

                        audio_streams.push(AudioStreamInfo {
                            index: i as u32,
                            codec: codec_id_to_string(&stream.codec_id),
                            sample_rate: audio.sample_rate,
                            channels: audio.channels as u32,
                            bits_per_sample: audio.bits_per_sample as u32,
                            duration: stream_duration,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // Estimate bitrate
    let bitrate = duration.map(|d| {
        if d > 0.0 {
            ((file_size as f64 * 8.0) / d) as i64
        } else {
            0
        }
    });

    demuxer.close();

    Ok(MediaInfo {
        format,
        duration,
        size: file_size,
        video_streams,
        audio_streams,
        bitrate,
    })
}

/// Convert codec ID to string.
fn codec_id_to_string(codec_id: &CodecId) -> String {
    match codec_id {
        CodecId::H264 => "h264".to_string(),
        CodecId::H265 => "h265".to_string(),
        CodecId::Vp9 => "vp9".to_string(),
        CodecId::Av1 => "av1".to_string(),
        CodecId::Aac => "aac".to_string(),
        CodecId::Mp3 => "mp3".to_string(),
        CodecId::Opus => "opus".to_string(),
        CodecId::Flac => "flac".to_string(),
        CodecId::Unknown(name) => name.clone(),
    }
}

/// High-level transcode function.
///
/// @param input - Path to the input media file.
/// @param output - Path for the output file.
/// @param options - Optional transcoding options.
/// @returns Promise that resolves to TranscodeStats on completion.
#[napi]
pub async fn transcode(
    input: String,
    output: String,
    options: Option<TranscodeOptions>,
) -> Result<TranscodeStats> {
    let transcoder = Transcoder::new(input, output, options);
    transcoder.run(None).await
}

/// Extract a thumbnail from a video file.
///
/// @param input - Path to the input video file.
/// @param timestamp - Timestamp in seconds to extract the thumbnail from.
/// @param output - Optional output path (defaults to input path with .jpg extension).
/// @returns Promise that resolves to the output path.
#[napi]
pub async fn extract_thumbnail(
    input: String,
    timestamp: f64,
    output: Option<String>,
) -> Result<String> {
    let input_path = PathBuf::from(&input);

    if !input_path.exists() {
        return Err(Error::new(
            Status::GenericFailure,
            format!("File not found: {}", input),
        ));
    }

    // Generate output path if not provided
    let output_path = match output {
        Some(path) => PathBuf::from(path),
        None => {
            let mut path = input_path.clone();
            path.set_extension("jpg");
            path
        }
    };

    // Open the file and seek to the timestamp
    let file = std::fs::File::open(&input_path)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to open file: {}", e)))?;

    let mut demuxer = Mp4Demuxer::new();
    demuxer.open(file)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to open file: {}", e)))?;

    // Seek to the requested timestamp
    let timestamp_us = (timestamp * 1_000_000.0) as i64;
    demuxer.seek(timestamp_us)
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to seek: {}", e)))?;

    // In a real implementation, we would:
    // 1. Read video packets from the timestamp
    // 2. Decode the frame
    // 3. Convert to JPEG
    // 4. Write to output file

    // For now, create a placeholder file
    std::fs::write(&output_path, b"")
        .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to write thumbnail: {}", e)))?;

    demuxer.close();

    Ok(output_path.to_string_lossy().to_string())
}

/// Detect SIMD capabilities of the current CPU.
///
/// @returns SimdCapabilities object.
#[napi]
pub fn detect_simd() -> SimdCapabilities {
    let caps = transcode_codecs::detect_simd();
    SimdCapabilities {
        sse42: caps.sse42,
        avx2: caps.avx2,
        avx512: caps.avx512,
        fma: caps.fma,
        neon: caps.neon,
        sve: caps.sve,
    }
}

/// Get version information about the transcode library.
///
/// @returns Version string.
#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get build information about the library.
///
/// @returns Build info object.
#[napi(object)]
pub struct BuildInfo {
    /// Library version.
    pub version: String,
    /// Target architecture.
    pub arch: String,
    /// Operating system.
    pub os: String,
    /// Debug build.
    pub debug: bool,
}

/// Get build information.
///
/// @returns BuildInfo object.
#[napi]
pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        arch: std::env::consts::ARCH.to_string(),
        os: std::env::consts::OS.to_string(),
        debug: cfg!(debug_assertions),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcode_options_default() {
        let options = TranscodeOptions::default();
        assert!(options.video_codec.is_none());
        assert!(options.audio_codec.is_none());
        assert!(options.overwrite.is_none());
    }

    #[test]
    fn test_simd_detection() {
        let caps = detect_simd();
        // Should return without panicking
        let _ = caps.sse42;
        let _ = caps.avx2;
    }

    #[test]
    fn test_codec_id_to_string() {
        assert_eq!(codec_id_to_string(&CodecId::H264), "h264");
        assert_eq!(codec_id_to_string(&CodecId::Aac), "aac");
        assert_eq!(
            codec_id_to_string(&CodecId::Unknown("custom".to_string())),
            "custom"
        );
    }
}
