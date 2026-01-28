//! Translation result types.

use serde::{Deserialize, Serialize};

/// Native Transcode arguments translated from FFmpeg flags.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NativeArgs {
    pub input: Option<String>,
    pub output: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub video_bitrate: Option<u64>,
    pub audio_bitrate: Option<u64>,
    pub crf: Option<u32>,
    pub preset: Option<String>,
    pub frame_rate: Option<String>,
    pub resolution: Option<String>,
    pub gop_size: Option<u32>,
    pub duration: Option<String>,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub threads: Option<u32>,
    pub format: Option<String>,
    pub audio_sample_rate: Option<u32>,
    pub audio_channels: Option<u32>,
    pub disable_audio: bool,
    pub disable_video: bool,
    pub overwrite: bool,
    pub filters: Vec<crate::filter::NativeFilter>,
}

impl NativeArgs {
    /// Convert to transcode CLI argument strings.
    pub fn to_cli_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        if let Some(ref input) = self.input {
            args.extend_from_slice(&["-i".into(), input.clone()]);
        }
        if let Some(ref codec) = self.video_codec {
            args.extend_from_slice(&["--video-codec".into(), codec.clone()]);
        }
        if let Some(ref codec) = self.audio_codec {
            args.extend_from_slice(&["--audio-codec".into(), codec.clone()]);
        }
        if let Some(bitrate) = self.video_bitrate {
            args.extend_from_slice(&["--video-bitrate".into(), bitrate.to_string()]);
        }
        if let Some(bitrate) = self.audio_bitrate {
            args.extend_from_slice(&["--audio-bitrate".into(), bitrate.to_string()]);
        }
        if let Some(crf) = self.crf {
            args.extend_from_slice(&["--crf".into(), crf.to_string()]);
        }
        if let Some(ref preset) = self.preset {
            args.extend_from_slice(&["--preset".into(), preset.clone()]);
        }
        if self.overwrite {
            args.push("--overwrite".into());
        }
        for filter in &self.filters {
            if filter.supported {
                let params_str: String = filter.params.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(",");
                if params_str.is_empty() {
                    args.extend_from_slice(&["--filter".into(), filter.name.clone()]);
                } else {
                    args.extend_from_slice(&["--filter".into(), format!("{}:{}", filter.name, params_str)]);
                }
            }
        }
        if let Some(ref output) = self.output {
            args.extend_from_slice(&["-o".into(), output.clone()]);
        }

        args
    }
}

/// A warning generated during translation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationWarning {
    pub ffmpeg_arg: String,
    pub message: String,
}

/// Result of translating FFmpeg args to native format.
#[derive(Debug, Clone)]
pub struct TranslationResult {
    pub native: NativeArgs,
    pub warnings: Vec<TranslationWarning>,
}

impl TranslationResult {
    /// Whether the translation was fully supported (no warnings).
    pub fn fully_supported(&self) -> bool {
        self.warnings.is_empty()
    }

    /// Number of unsupported arguments.
    pub fn unsupported_count(&self) -> usize {
        self.warnings.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_args_to_cli() {
        let args = NativeArgs {
            input: Some("input.mp4".into()),
            output: Some("output.mp4".into()),
            video_codec: Some("h264".into()),
            video_bitrate: Some(5_000_000),
            overwrite: true,
            ..Default::default()
        };
        let cli = args.to_cli_args();
        assert!(cli.contains(&"-i".to_string()));
        assert!(cli.contains(&"--video-codec".to_string()));
        assert!(cli.contains(&"--overwrite".to_string()));
    }
}
