//! Compatibility report generation.

use crate::translator::TranslationResult;
use serde::{Deserialize, Serialize};

/// Summary of FFmpeg command compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    pub total_args: usize,
    pub supported_args: usize,
    pub unsupported_args: usize,
    pub compatibility_percent: f64,
    pub unsupported_details: Vec<String>,
}

impl CompatibilityReport {
    pub fn from_result(result: &TranslationResult) -> Self {
        let unsupported = result.warnings.len();
        // Count supported args by looking at non-None fields
        let supported = count_set_fields(&result.native);
        let total = supported + unsupported;

        Self {
            total_args: total,
            supported_args: supported,
            unsupported_args: unsupported,
            compatibility_percent: if total > 0 {
                (supported as f64 / total as f64) * 100.0
            } else {
                100.0
            },
            unsupported_details: result
                .warnings
                .iter()
                .map(|w| w.message.clone())
                .collect(),
        }
    }

    pub fn fully_supported(&self) -> bool {
        self.unsupported_args == 0
    }

    /// Format as human-readable text.
    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "FFmpeg Compatibility: {:.0}% ({}/{} args supported)\n",
            self.compatibility_percent, self.supported_args, self.total_args
        ));
        if !self.unsupported_details.is_empty() {
            out.push_str("\nUnsupported arguments:\n");
            for detail in &self.unsupported_details {
                out.push_str(&format!("  ⚠ {}\n", detail));
            }
        }
        out
    }
}

fn count_set_fields(native: &crate::translator::NativeArgs) -> usize {
    let mut count = 0;
    if native.input.is_some() { count += 1; }
    if native.output.is_some() { count += 1; }
    if native.video_codec.is_some() { count += 1; }
    if native.audio_codec.is_some() { count += 1; }
    if native.video_bitrate.is_some() { count += 1; }
    if native.audio_bitrate.is_some() { count += 1; }
    if native.crf.is_some() { count += 1; }
    if native.preset.is_some() { count += 1; }
    if native.frame_rate.is_some() { count += 1; }
    if native.resolution.is_some() { count += 1; }
    if native.gop_size.is_some() { count += 1; }
    if native.duration.is_some() { count += 1; }
    if native.start_time.is_some() { count += 1; }
    if native.end_time.is_some() { count += 1; }
    if native.threads.is_some() { count += 1; }
    if native.format.is_some() { count += 1; }
    if native.audio_sample_rate.is_some() { count += 1; }
    if native.audio_channels.is_some() { count += 1; }
    if native.disable_audio { count += 1; }
    if native.disable_video { count += 1; }
    if native.overwrite { count += 1; }
    count += native.filters.len();
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::FfmpegArgs;

    #[test]
    fn test_full_compatibility_report() {
        let args = vec!["-i", "in.mp4", "-c:v", "libx264", "out.mp4"];
        let parsed = FfmpegArgs::parse(&args).unwrap();
        let result = parsed.translate();
        let report = CompatibilityReport::from_result(&result);
        assert!(report.fully_supported());
        assert_eq!(report.compatibility_percent, 100.0);
    }

    #[test]
    fn test_partial_compatibility() {
        let args = vec![
            "-i", "in.mp4", "-c:v", "libx264",
            "-filter_complex", "overlay", "out.mp4",
        ];
        let parsed = FfmpegArgs::parse(&args).unwrap();
        let result = parsed.translate();
        let report = CompatibilityReport::from_result(&result);
        assert!(!report.fully_supported());
        assert!(report.compatibility_percent < 100.0);
    }
}
