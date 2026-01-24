//! Media file inspection command.

use clap::Args;
use console::style;
use serde::Serialize;
use std::fs::File;
use std::path::PathBuf;
use transcode::{Mp4Demuxer, Demuxer, TrackType};

/// Stream information for display.
#[derive(Debug, Clone, Serialize)]
pub struct StreamInfo {
    /// Stream index.
    pub index: usize,
    /// Stream type (video/audio).
    #[serde(rename = "type")]
    pub stream_type: String,
    /// Codec name.
    pub codec: String,
    /// Duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Bitrate in bps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitrate: Option<u64>,
    /// Video-specific: width.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Video-specific: height.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// Video-specific: frame rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_rate: Option<f64>,
    /// Audio-specific: sample rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    /// Audio-specific: channels.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<u32>,
}

/// Media file information.
#[derive(Debug, Clone, Serialize)]
pub struct MediaInfo {
    /// File path.
    pub file: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Container format.
    pub format: String,
    /// Total duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Overall bitrate in bps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bitrate: Option<u64>,
    /// Stream information.
    pub streams: Vec<StreamInfo>,
}

/// Inspect a media file.
#[derive(Args, Debug)]
pub struct CmdInfo {
    /// Path to the media file.
    pub file: PathBuf,

    /// Output in JSON format.
    #[arg(long)]
    pub json: bool,
}

impl CmdInfo {
    /// Execute the info command.
    pub fn run(&self) -> anyhow::Result<()> {
        if !self.file.exists() {
            anyhow::bail!("File not found: {}", self.file.display());
        }

        let metadata = std::fs::metadata(&self.file)?;
        let file_size = metadata.len();

        // Try to open as MP4
        let media_info = match self.try_analyze_mp4(file_size) {
            Ok(info) => info,
            Err(_) => {
                // Fallback: just show file info
                MediaInfo {
                    file: self.file.display().to_string(),
                    size_bytes: file_size,
                    format: self.guess_format(),
                    duration_seconds: None,
                    bitrate: None,
                    streams: vec![],
                }
            }
        };

        if self.json {
            println!("{}", serde_json::to_string_pretty(&media_info)?);
        } else {
            self.print_media_info(&media_info);
        }

        Ok(())
    }

    fn try_analyze_mp4(&self, file_size: u64) -> anyhow::Result<MediaInfo> {
        let file = File::open(&self.file)?;
        let mut demuxer = Mp4Demuxer::new();
        demuxer.open(file)?;

        let mut streams = Vec::new();

        for i in 0..demuxer.num_streams() {
            if let Some(stream) = demuxer.stream_info(i) {
                let duration_us = stream.duration.map(|d| {
                    let time_base = stream.time_base;
                    (d as f64 * time_base.to_f64()) * 1_000_000.0
                });
                let duration_seconds = duration_us.map(|us| us / 1_000_000.0);

                let stream_info = match stream.track_type {
                    TrackType::Video => {
                        let video = stream.video.as_ref();
                        StreamInfo {
                            index: stream.index,
                            stream_type: "video".to_string(),
                            codec: format!("{:?}", stream.codec_id).to_lowercase(),
                            duration_seconds,
                            bitrate: None,
                            width: video.map(|v| v.width),
                            height: video.map(|v| v.height),
                            frame_rate: video.and_then(|v| v.frame_rate.as_ref().map(|f| f.to_f64())),
                            sample_rate: None,
                            channels: None,
                        }
                    }
                    TrackType::Audio => {
                        let audio = stream.audio.as_ref();
                        StreamInfo {
                            index: stream.index,
                            stream_type: "audio".to_string(),
                            codec: format!("{:?}", stream.codec_id).to_lowercase(),
                            duration_seconds,
                            bitrate: None,
                            width: None,
                            height: None,
                            frame_rate: None,
                            sample_rate: audio.map(|a| a.sample_rate),
                            channels: audio.map(|a| a.channels as u32),
                        }
                    }
                    _ => continue,
                };
                streams.push(stream_info);
            }
        }

        // Calculate duration from demuxer or longest stream
        let duration_us = demuxer.duration();
        let duration = duration_us.map(|us| us as f64 / 1_000_000.0);

        // Calculate overall bitrate
        let bitrate = duration.map(|d| if d > 0.0 { ((file_size as f64 * 8.0) / d) as u64 } else { 0 });

        Ok(MediaInfo {
            file: self.file.display().to_string(),
            size_bytes: file_size,
            format: demuxer.format_name().to_string(),
            duration_seconds: duration,
            bitrate,
            streams,
        })
    }

    fn guess_format(&self) -> String {
        self.file
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_lowercase()
    }

    fn print_media_info(&self, info: &MediaInfo) {
        println!();
        println!("{}", style("Media Information").cyan().bold());
        println!();

        println!("  {:<16} {}", style("File:").white(), info.file);
        println!(
            "  {:<16} {}",
            style("Size:").white(),
            format_size(info.size_bytes)
        );
        println!("  {:<16} {}", style("Format:").white(), info.format);

        if let Some(duration) = info.duration_seconds {
            println!(
                "  {:<16} {}",
                style("Duration:").white(),
                format_duration(duration)
            );
        }

        if let Some(bitrate) = info.bitrate {
            println!(
                "  {:<16} {:.0} kbps",
                style("Bitrate:").white(),
                bitrate as f64 / 1000.0
            );
        }

        if !info.streams.is_empty() {
            println!();
            println!("{}", style("Streams:").cyan().bold());

            for stream in &info.streams {
                println!();
                println!(
                    "  {} #{} ({})",
                    style("Stream").white(),
                    stream.index,
                    style(&stream.stream_type).yellow()
                );
                println!("    {:<14} {}", style("Codec:").dim(), stream.codec);

                if let Some(duration) = stream.duration_seconds {
                    println!(
                        "    {:<14} {}",
                        style("Duration:").dim(),
                        format_duration(duration)
                    );
                }

                if let Some(bitrate) = stream.bitrate {
                    println!(
                        "    {:<14} {:.0} kbps",
                        style("Bitrate:").dim(),
                        bitrate as f64 / 1000.0
                    );
                }

                // Video-specific
                if let (Some(w), Some(h)) = (stream.width, stream.height) {
                    println!("    {:<14} {}x{}", style("Resolution:").dim(), w, h);
                }
                if let Some(fps) = stream.frame_rate {
                    println!("    {:<14} {:.2} fps", style("Frame Rate:").dim(), fps);
                }

                // Audio-specific
                if let Some(sr) = stream.sample_rate {
                    println!("    {:<14} {} Hz", style("Sample Rate:").dim(), sr);
                }
                if let Some(ch) = stream.channels {
                    let ch_str = match ch {
                        1 => "mono".to_string(),
                        2 => "stereo".to_string(),
                        6 => "5.1".to_string(),
                        8 => "7.1".to_string(),
                        _ => format!("{} channels", ch),
                    };
                    println!("    {:<14} {}", style("Channels:").dim(), ch_str);
                }
            }
        }

        println!();
    }
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds as u64;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let secs = total_seconds % 60;
    let millis = ((seconds - total_seconds as f64) * 1000.0) as u64;

    if hours > 0 {
        format!("{}:{:02}:{:02}.{:03}", hours, minutes, secs, millis)
    } else {
        format!("{}:{:02}.{:03}", minutes, secs, millis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.0), "0:00.000");
        assert_eq!(format_duration(61.5), "1:01.500");
        assert_eq!(format_duration(3661.0), "1:01:01.000");
    }

    #[test]
    fn test_media_info_serialization() {
        let info = MediaInfo {
            file: "test.mp4".to_string(),
            size_bytes: 1024,
            format: "mp4".to_string(),
            duration_seconds: Some(10.0),
            bitrate: Some(128000),
            streams: vec![],
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"file\":\"test.mp4\""));
    }
}
