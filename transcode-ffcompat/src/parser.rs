//! FFmpeg argument parser.

use crate::error::{Error, Result};
use crate::translator::{NativeArgs, TranslationResult, TranslationWarning};

/// A parsed FFmpeg argument.
#[derive(Debug, Clone, PartialEq)]
pub enum FfmpegArg {
    Input(String),
    Output(String),
    VideoCodec(String),
    AudioCodec(String),
    VideoBitrate(String),
    AudioBitrate(String),
    Crf(u32),
    Preset(String),
    FrameRate(String),
    Resolution(String),
    GopSize(u32),
    Duration(String),
    StartTime(String),
    EndTime(String),
    Threads(u32),
    Format(String),
    AudioSampleRate(u32),
    AudioChannels(u32),
    DisableAudio,
    DisableVideo,
    Overwrite,
    NoOverwrite,
    VideoFilter(String),
    AudioFilter(String),
    FilterComplex(String),
    Unsupported { flag: String, value: Option<String> },
}

/// Parsed FFmpeg command line.
#[derive(Debug, Clone)]
pub struct FfmpegArgs {
    pub args: Vec<FfmpegArg>,
}

impl FfmpegArgs {
    /// Parse FFmpeg-style arguments from a slice of strings.
    pub fn parse(args: &[&str]) -> Result<Self> {
        let mut parsed = Vec::new();
        let mut i = 0;

        while i < args.len() {
            let arg = args[i];
            match arg {
                "-i" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-i".into() })?;
                    parsed.push(FfmpegArg::Input(val.to_string()));
                }
                "-y" => parsed.push(FfmpegArg::Overwrite),
                "-n" => parsed.push(FfmpegArg::NoOverwrite),
                "-an" => parsed.push(FfmpegArg::DisableAudio),
                "-vn" => parsed.push(FfmpegArg::DisableVideo),

                "-c:v" | "-vcodec" | "-codec:v" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: arg.into() })?;
                    parsed.push(FfmpegArg::VideoCodec(val.to_string()));
                }
                "-c:a" | "-acodec" | "-codec:a" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: arg.into() })?;
                    parsed.push(FfmpegArg::AudioCodec(val.to_string()));
                }
                "-b:v" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-b:v".into() })?;
                    parsed.push(FfmpegArg::VideoBitrate(val.to_string()));
                }
                "-b:a" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-b:a".into() })?;
                    parsed.push(FfmpegArg::AudioBitrate(val.to_string()));
                }
                "-crf" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-crf".into() })?;
                    let crf = val.parse::<u32>().map_err(|_| Error::InvalidValue {
                        flag: "-crf".into(),
                        message: format!("'{}' is not a valid integer", val),
                    })?;
                    parsed.push(FfmpegArg::Crf(crf));
                }
                "-preset" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-preset".into() })?;
                    parsed.push(FfmpegArg::Preset(val.to_string()));
                }
                "-r" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-r".into() })?;
                    parsed.push(FfmpegArg::FrameRate(val.to_string()));
                }
                "-s" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-s".into() })?;
                    parsed.push(FfmpegArg::Resolution(val.to_string()));
                }
                "-g" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-g".into() })?;
                    let gop = val.parse::<u32>().map_err(|_| Error::InvalidValue {
                        flag: "-g".into(),
                        message: format!("'{}' is not a valid integer", val),
                    })?;
                    parsed.push(FfmpegArg::GopSize(gop));
                }
                "-t" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-t".into() })?;
                    parsed.push(FfmpegArg::Duration(val.to_string()));
                }
                "-ss" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-ss".into() })?;
                    parsed.push(FfmpegArg::StartTime(val.to_string()));
                }
                "-to" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-to".into() })?;
                    parsed.push(FfmpegArg::EndTime(val.to_string()));
                }
                "-threads" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-threads".into() })?;
                    let n = val.parse::<u32>().map_err(|_| Error::InvalidValue {
                        flag: "-threads".into(),
                        message: format!("'{}' is not a valid integer", val),
                    })?;
                    parsed.push(FfmpegArg::Threads(n));
                }
                "-f" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-f".into() })?;
                    parsed.push(FfmpegArg::Format(val.to_string()));
                }
                "-ar" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-ar".into() })?;
                    let sr = val.parse::<u32>().map_err(|_| Error::InvalidValue {
                        flag: "-ar".into(),
                        message: format!("'{}' is not a valid integer", val),
                    })?;
                    parsed.push(FfmpegArg::AudioSampleRate(sr));
                }
                "-ac" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: "-ac".into() })?;
                    let ch = val.parse::<u32>().map_err(|_| Error::InvalidValue {
                        flag: "-ac".into(),
                        message: format!("'{}' is not a valid integer", val),
                    })?;
                    parsed.push(FfmpegArg::AudioChannels(ch));
                }
                "-vf" | "-filter:v" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: arg.into() })?;
                    parsed.push(FfmpegArg::VideoFilter(val.to_string()));
                }
                "-af" | "-filter:a" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: arg.into() })?;
                    parsed.push(FfmpegArg::AudioFilter(val.to_string()));
                }
                "-filter_complex" | "-lavfi" => {
                    i += 1;
                    let val = args.get(i).ok_or(Error::MissingValue { flag: arg.into() })?;
                    parsed.push(FfmpegArg::FilterComplex(val.to_string()));
                }
                other if other.starts_with('-') => {
                    // Check if next arg is a value (doesn't start with -)
                    let value = if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                        i += 1;
                        Some(args[i].to_string())
                    } else {
                        None
                    };
                    parsed.push(FfmpegArg::Unsupported {
                        flag: other.into(),
                        value,
                    });
                }
                // Positional argument = output file
                _ => {
                    parsed.push(FfmpegArg::Output(arg.to_string()));
                }
            }
            i += 1;
        }

        Ok(FfmpegArgs { args: parsed })
    }

    /// Translate parsed args to native Transcode options.
    pub fn translate(&self) -> TranslationResult {
        let mut native = NativeArgs::default();
        let mut warnings = Vec::new();

        for arg in &self.args {
            match arg {
                FfmpegArg::Input(path) => native.input = Some(path.clone()),
                FfmpegArg::Output(path) => native.output = Some(path.clone()),
                FfmpegArg::Overwrite => native.overwrite = true,
                FfmpegArg::NoOverwrite => native.overwrite = false,
                FfmpegArg::VideoCodec(name) => {
                    native.video_codec = Some(translate_video_codec(name));
                }
                FfmpegArg::AudioCodec(name) => {
                    native.audio_codec = Some(translate_audio_codec(name));
                }
                FfmpegArg::VideoBitrate(val) => {
                    native.video_bitrate = Some(parse_bitrate(val));
                }
                FfmpegArg::AudioBitrate(val) => {
                    native.audio_bitrate = Some(parse_bitrate(val));
                }
                FfmpegArg::Crf(val) => native.crf = Some(*val),
                FfmpegArg::Preset(val) => native.preset = Some(val.clone()),
                FfmpegArg::FrameRate(val) => native.frame_rate = Some(val.clone()),
                FfmpegArg::Resolution(val) => native.resolution = Some(val.clone()),
                FfmpegArg::GopSize(val) => native.gop_size = Some(*val),
                FfmpegArg::Duration(val) => native.duration = Some(val.clone()),
                FfmpegArg::StartTime(val) => native.start_time = Some(val.clone()),
                FfmpegArg::EndTime(val) => native.end_time = Some(val.clone()),
                FfmpegArg::Threads(val) => native.threads = Some(*val),
                FfmpegArg::Format(val) => native.format = Some(val.clone()),
                FfmpegArg::AudioSampleRate(val) => native.audio_sample_rate = Some(*val),
                FfmpegArg::AudioChannels(val) => native.audio_channels = Some(*val),
                FfmpegArg::DisableAudio => native.disable_audio = true,
                FfmpegArg::DisableVideo => native.disable_video = true,
                FfmpegArg::VideoFilter(graph_str) | FfmpegArg::AudioFilter(graph_str) | FfmpegArg::FilterComplex(graph_str) => {
                    let graph = crate::filter::FilterGraph::parse(graph_str);
                    let filters = graph.translate();
                    for f in &filters {
                        if !f.supported {
                            warnings.push(TranslationWarning {
                                ffmpeg_arg: format!("-vf {}", f.name),
                                message: format!("Filter '{}' is not yet supported in Transcode", f.name),
                            });
                        }
                    }
                    native.filters = filters;
                }
                FfmpegArg::Unsupported { flag, value } => {
                    warnings.push(TranslationWarning {
                        ffmpeg_arg: flag.clone(),
                        message: format!(
                            "Unsupported FFmpeg argument '{}'{} — ignored",
                            flag,
                            value.as_ref().map(|v| format!(" '{}'", v)).unwrap_or_default()
                        ),
                    });
                }
            }
        }

        TranslationResult { native, warnings }
    }
}

fn translate_video_codec(name: &str) -> String {
    match name {
        "libx264" | "h264" => "h264",
        "libx265" | "h265" | "hevc" => "hevc",
        "libvpx-vp9" | "vp9" => "vp9",
        "libvpx" | "vp8" => "vp8",
        "libaom-av1" | "libsvtav1" | "librav1e" | "av1" => "av1",
        "libtheora" | "theora" => "theora",
        "prores" | "prores_ks" => "prores",
        "copy" => "copy",
        other => other,
    }
    .to_string()
}

fn translate_audio_codec(name: &str) -> String {
    match name {
        "aac" | "libfdk_aac" => "aac",
        "libmp3lame" | "mp3" => "mp3",
        "libopus" | "opus" => "opus",
        "flac" => "flac",
        "libvorbis" | "vorbis" => "vorbis",
        "ac3" | "eac3" => "ac3",
        "pcm_s16le" | "pcm_s24le" | "pcm_f32le" => "pcm",
        "copy" => "copy",
        other => other,
    }
    .to_string()
}

fn parse_bitrate(val: &str) -> u64 {
    let val = val.trim();
    if let Some(stripped) = val.strip_suffix('k').or_else(|| val.strip_suffix('K')) {
        stripped.parse::<u64>().unwrap_or(0) * 1000
    } else if let Some(stripped) = val.strip_suffix('M').or_else(|| val.strip_suffix('m')) {
        stripped.parse::<u64>().unwrap_or(0) * 1_000_000
    } else {
        val.parse::<u64>().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let args = vec!["-i", "in.mp4", "-c:v", "libx264", "out.mp4"];
        let parsed = FfmpegArgs::parse(&args).unwrap();
        assert!(parsed.args.iter().any(|a| matches!(a, FfmpegArg::Input(p) if p == "in.mp4")));
        assert!(parsed.args.iter().any(|a| matches!(a, FfmpegArg::Output(p) if p == "out.mp4")));
    }

    #[test]
    fn test_codec_translation() {
        assert_eq!(translate_video_codec("libx264"), "h264");
        assert_eq!(translate_video_codec("libx265"), "hevc");
        assert_eq!(translate_video_codec("libaom-av1"), "av1");
        assert_eq!(translate_audio_codec("libfdk_aac"), "aac");
        assert_eq!(translate_audio_codec("libopus"), "opus");
    }

    #[test]
    fn test_parse_bitrate() {
        assert_eq!(parse_bitrate("128k"), 128_000);
        assert_eq!(parse_bitrate("5M"), 5_000_000);
        assert_eq!(parse_bitrate("1000000"), 1_000_000);
    }

    #[test]
    fn test_missing_value_error() {
        let args = vec!["-i"];
        assert!(FfmpegArgs::parse(&args).is_err());
    }

    #[test]
    fn test_all_args_translated() {
        let args = vec![
            "-y", "-i", "in.mp4", "-c:v", "libx264", "-crf", "18",
            "-preset", "slow", "-b:a", "192k", "-c:a", "aac",
            "-r", "30", "-s", "1920x1080", "-g", "250",
            "-t", "60", "-ss", "10", "-threads", "8", "out.mp4",
        ];
        let parsed = FfmpegArgs::parse(&args).unwrap();
        let result = parsed.translate();
        assert!(result.warnings.is_empty());
        assert!(result.native.overwrite);
        assert_eq!(result.native.crf, Some(18));
        assert_eq!(result.native.threads, Some(8));
    }
}
