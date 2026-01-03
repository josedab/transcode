//! Format conversion utilities for subtitle tracks.
//!
//! This module provides functions to convert between different subtitle formats
//! (SRT, ASS/SSA, WebVTT) while preserving as much information as possible.

use crate::types::{SubtitleFormat, SubtitleResult, SubtitleTrack};
use crate::{ass, srt, vtt};

/// Converts a subtitle track to the specified format string.
pub fn convert_to_format(track: &SubtitleTrack, format: SubtitleFormat) -> String {
    match format {
        SubtitleFormat::Srt => srt::write(track),
        SubtitleFormat::Ass => ass::write(track),
        SubtitleFormat::Ssa => ass::write_ssa(track),
        SubtitleFormat::WebVtt => vtt::write(track),
    }
}

/// Parses subtitle content from a string, auto-detecting the format.
pub fn parse_auto(content: &str) -> SubtitleResult<(SubtitleTrack, SubtitleFormat)> {
    let format = detect_format(content)?;
    let track = parse_with_format(content, format)?;
    Ok((track, format))
}

/// Parses subtitle content with the specified format.
pub fn parse_with_format(content: &str, format: SubtitleFormat) -> SubtitleResult<SubtitleTrack> {
    match format {
        SubtitleFormat::Srt => srt::parse(content),
        SubtitleFormat::Ass | SubtitleFormat::Ssa => ass::parse(content),
        SubtitleFormat::WebVtt => vtt::parse(content),
    }
}

/// Detects the subtitle format from content.
pub fn detect_format(content: &str) -> SubtitleResult<SubtitleFormat> {
    let content = content.trim_start();

    // Check for WebVTT header
    if content.starts_with("WEBVTT") {
        return Ok(SubtitleFormat::WebVtt);
    }

    // Check for ASS/SSA script info section
    if content.contains("[Script Info]") {
        // Distinguish between ASS and SSA by script type
        if content.contains("v4.00+") || content.contains("V4+ Styles") {
            return Ok(SubtitleFormat::Ass);
        } else if content.contains("v4.00") || content.contains("V4 Styles") {
            return Ok(SubtitleFormat::Ssa);
        }
        // Default to ASS if unclear
        return Ok(SubtitleFormat::Ass);
    }

    // Check for SRT format (numeric index followed by timing)
    let lines: Vec<&str> = content.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let line = line.trim();
        // Skip empty lines
        if line.is_empty() {
            continue;
        }
        // Check if first non-empty line looks like an index
        if line.parse::<u32>().is_ok() {
            // Check if next line contains SRT timing
            if let Some(next_line) = lines.get(i + 1) {
                if next_line.contains("-->") && next_line.contains(',') {
                    return Ok(SubtitleFormat::Srt);
                }
            }
        }
        // If first non-empty content doesn't match known formats, break
        break;
    }

    // Default to SRT as it's the most common and simplest format
    Ok(SubtitleFormat::Srt)
}

/// Converts subtitle content from one format to another.
pub fn convert(content: &str, from: SubtitleFormat, to: SubtitleFormat) -> SubtitleResult<String> {
    let track = parse_with_format(content, from)?;
    Ok(convert_to_format(&track, to))
}

/// Converts subtitle content, auto-detecting the source format.
pub fn convert_auto(content: &str, to: SubtitleFormat) -> SubtitleResult<String> {
    let (track, _) = parse_auto(content)?;
    Ok(convert_to_format(&track, to))
}

/// A builder for performing subtitle conversions with options.
#[derive(Debug, Clone)]
pub struct ConversionBuilder<'a> {
    content: &'a str,
    source_format: Option<SubtitleFormat>,
    strip_styling: bool,
    normalize_timing: bool,
}

impl<'a> ConversionBuilder<'a> {
    /// Creates a new conversion builder from source content.
    pub fn new(content: &'a str) -> Self {
        Self {
            content,
            source_format: None,
            strip_styling: false,
            normalize_timing: false,
        }
    }

    /// Sets the source format (otherwise auto-detected).
    pub fn from_format(mut self, format: SubtitleFormat) -> Self {
        self.source_format = Some(format);
        self
    }

    /// Strips all styling information during conversion.
    pub fn strip_styling(mut self) -> Self {
        self.strip_styling = true;
        self
    }

    /// Normalizes timing (sorts by start time, removes overlaps).
    pub fn normalize_timing(mut self) -> Self {
        self.normalize_timing = true;
        self
    }

    /// Converts to the specified format.
    pub fn to_format(self, format: SubtitleFormat) -> SubtitleResult<String> {
        let mut track = if let Some(source) = self.source_format {
            parse_with_format(self.content, source)?
        } else {
            parse_auto(self.content)?.0
        };

        if self.strip_styling {
            strip_track_styling(&mut track);
        }

        if self.normalize_timing {
            normalize_track_timing(&mut track);
        }

        Ok(convert_to_format(&track, format))
    }

    /// Converts to SRT format.
    pub fn to_srt(self) -> SubtitleResult<String> {
        self.to_format(SubtitleFormat::Srt)
    }

    /// Converts to ASS format.
    pub fn to_ass(self) -> SubtitleResult<String> {
        self.to_format(SubtitleFormat::Ass)
    }

    /// Converts to SSA format.
    pub fn to_ssa(self) -> SubtitleResult<String> {
        self.to_format(SubtitleFormat::Ssa)
    }

    /// Converts to WebVTT format.
    pub fn to_vtt(self) -> SubtitleResult<String> {
        self.to_format(SubtitleFormat::WebVtt)
    }
}

/// Strips all styling from a subtitle track.
fn strip_track_styling(track: &mut SubtitleTrack) {
    use crate::types::{StyledText, TextStyle};

    for event in &mut track.events {
        let plain_text = event.plain_text();
        event.text = vec![StyledText::new(plain_text, TextStyle::default())];
    }

    track.styles.clear();
}

/// Normalizes timing in a subtitle track.
fn normalize_track_timing(track: &mut SubtitleTrack) {
    // Sort by start time
    track.sort_by_time();

    // Remove overlaps by adjusting end times
    for i in 0..track.events.len().saturating_sub(1) {
        let next_start = track.events[i + 1].start.milliseconds;
        if track.events[i].end.milliseconds > next_start {
            track.events[i].end.milliseconds = next_start.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Timestamp;

    const SRT_SAMPLE: &str = r#"1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:08,500
Second subtitle.
"#;

    const VTT_SAMPLE: &str = r#"WEBVTT

00:00:01.000 --> 00:00:04.000
Hello, world!

00:00:05.000 --> 00:00:08.500
Second subtitle.
"#;

    const ASS_SAMPLE: &str = r#"[Script Info]
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello, world!
Dialogue: 0,0:00:05.00,0:00:08.50,Default,,0,0,0,,Second subtitle.
"#;

    #[test]
    fn test_detect_srt() {
        let format = detect_format(SRT_SAMPLE).unwrap();
        assert_eq!(format, SubtitleFormat::Srt);
    }

    #[test]
    fn test_detect_vtt() {
        let format = detect_format(VTT_SAMPLE).unwrap();
        assert_eq!(format, SubtitleFormat::WebVtt);
    }

    #[test]
    fn test_detect_ass() {
        let format = detect_format(ASS_SAMPLE).unwrap();
        assert_eq!(format, SubtitleFormat::Ass);
    }

    #[test]
    fn test_parse_auto() {
        let (track, format) = parse_auto(SRT_SAMPLE).unwrap();
        assert_eq!(format, SubtitleFormat::Srt);
        assert_eq!(track.events.len(), 2);
    }

    #[test]
    fn test_convert_srt_to_vtt() {
        let result = convert(SRT_SAMPLE, SubtitleFormat::Srt, SubtitleFormat::WebVtt).unwrap();
        assert!(result.starts_with("WEBVTT"));
        assert!(result.contains("00:00:01.000 --> 00:00:04.000"));
        assert!(result.contains("Hello, world!"));
    }

    #[test]
    fn test_convert_vtt_to_srt() {
        let result = convert(VTT_SAMPLE, SubtitleFormat::WebVtt, SubtitleFormat::Srt).unwrap();
        assert!(result.contains("00:00:01,000 --> 00:00:04,000"));
        assert!(result.contains("Hello, world!"));
    }

    #[test]
    fn test_convert_srt_to_ass() {
        let result = convert(SRT_SAMPLE, SubtitleFormat::Srt, SubtitleFormat::Ass).unwrap();
        assert!(result.contains("[Script Info]"));
        assert!(result.contains("[Events]"));
        assert!(result.contains("Hello, world!"));
    }

    #[test]
    fn test_convert_ass_to_srt() {
        let result = convert(ASS_SAMPLE, SubtitleFormat::Ass, SubtitleFormat::Srt).unwrap();
        assert!(result.contains("00:00:01,000 --> 00:00:04,000"));
        assert!(result.contains("Hello, world!"));
    }

    #[test]
    fn test_convert_auto() {
        let result = convert_auto(SRT_SAMPLE, SubtitleFormat::WebVtt).unwrap();
        assert!(result.starts_with("WEBVTT"));
        assert!(result.contains("Hello, world!"));
    }

    #[test]
    fn test_conversion_builder() {
        let result = ConversionBuilder::new(SRT_SAMPLE)
            .from_format(SubtitleFormat::Srt)
            .to_vtt()
            .unwrap();

        assert!(result.starts_with("WEBVTT"));
        assert!(result.contains("Hello, world!"));
    }

    #[test]
    fn test_conversion_builder_strip_styling() {
        let styled_srt = r#"1
00:00:01,000 --> 00:00:04,000
<b>Bold</b> text

"#;

        let result = ConversionBuilder::new(styled_srt)
            .strip_styling()
            .to_srt()
            .unwrap();

        // The output should not contain any styling tags
        assert!(!result.contains("<b>"));
        assert!(result.contains("Bold text"));
    }

    #[test]
    fn test_conversion_preserves_timing() {
        // Convert SRT -> VTT -> ASS -> SRT and verify timing is preserved
        let srt = convert_auto(SRT_SAMPLE, SubtitleFormat::Srt).unwrap();
        let vtt = convert(&srt, SubtitleFormat::Srt, SubtitleFormat::WebVtt).unwrap();
        let ass = convert(&vtt, SubtitleFormat::WebVtt, SubtitleFormat::Ass).unwrap();
        let final_srt = convert(&ass, SubtitleFormat::Ass, SubtitleFormat::Srt).unwrap();

        let original_track = srt::parse(SRT_SAMPLE).unwrap();
        let final_track = srt::parse(&final_srt).unwrap();

        assert_eq!(original_track.events.len(), final_track.events.len());

        for (orig, final_evt) in original_track.events.iter().zip(final_track.events.iter()) {
            // Allow small timing differences due to centisecond precision in ASS
            assert!(
                (orig.start.milliseconds as i64 - final_evt.start.milliseconds as i64).abs() <= 10
            );
            assert!(
                (orig.end.milliseconds as i64 - final_evt.end.milliseconds as i64).abs() <= 10
            );
        }
    }

    #[test]
    fn test_normalize_timing() {
        use crate::types::SubtitleEvent;

        let mut track = SubtitleTrack::new();

        // Add events out of order with overlaps
        track.add_event(SubtitleEvent::new(
            Timestamp::new(0, 0, 5, 0),
            Timestamp::new(0, 0, 10, 0),
            "Second",
        ));
        track.add_event(SubtitleEvent::new(
            Timestamp::new(0, 0, 1, 0),
            Timestamp::new(0, 0, 6, 0), // Overlaps with "Second"
            "First",
        ));

        normalize_track_timing(&mut track);

        // Should be sorted
        assert_eq!(track.events[0].plain_text(), "First");
        assert_eq!(track.events[1].plain_text(), "Second");

        // Overlap should be removed
        assert!(track.events[0].end.milliseconds < track.events[1].start.milliseconds);
    }
}
