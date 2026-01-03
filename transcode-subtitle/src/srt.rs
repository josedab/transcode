//! SRT (SubRip) subtitle format parser and writer.
//!
//! SRT is a simple and widely used subtitle format consisting of:
//! - A numeric index
//! - Timing in HH:MM:SS,mmm --> HH:MM:SS,mmm format
//! - Subtitle text (may contain basic HTML-like formatting)
//! - Blank line separator

use crate::types::{
    Color, StyledText, SubtitleError, SubtitleEvent, SubtitleResult, SubtitleTrack, TextStyle,
    Timestamp,
};
use regex::Regex;
use std::io::{BufRead, Write};

/// Parses SRT content from a string.
pub fn parse(content: &str) -> SubtitleResult<SubtitleTrack> {
    let mut track = SubtitleTrack::new();
    let mut lines = content.lines().peekable();

    while lines.peek().is_some() {
        // Skip empty lines
        while lines.peek().is_some_and(|l| l.trim().is_empty()) {
            lines.next();
        }

        if lines.peek().is_none() {
            break;
        }

        // Parse index (we don't strictly require it to be numeric)
        let _index_line = match lines.next() {
            Some(line) => line.trim(),
            None => break,
        };

        // Parse timing line
        let timing_line = match lines.next() {
            Some(line) => line.trim(),
            None => break,
        };

        let (start, end) = parse_timing_line(timing_line)?;

        // Collect text lines until empty line
        let mut text_lines = Vec::new();
        while let Some(line) = lines.peek() {
            if line.trim().is_empty() {
                break;
            }
            text_lines.push(lines.next().unwrap());
        }

        if !text_lines.is_empty() {
            let raw_text = text_lines.join("\n");
            let styled_text = parse_srt_formatting(&raw_text);

            let event = SubtitleEvent::with_styled_text(start, end, styled_text);
            track.add_event(event);
        }
    }

    Ok(track)
}

/// Parses SRT content from a reader.
pub fn parse_reader<R: BufRead>(reader: R) -> SubtitleResult<SubtitleTrack> {
    let content: String = reader
        .lines()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| SubtitleError::IoError(e.to_string()))?
        .join("\n");

    parse(&content)
}

/// Parses a timing line in SRT format.
fn parse_timing_line(line: &str) -> SubtitleResult<(Timestamp, Timestamp)> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() != 2 {
        return Err(SubtitleError::ParseError(format!(
            "Invalid timing line: {}",
            line
        )));
    }

    let start = Timestamp::parse_srt(parts[0].trim())?;
    // Handle potential position info after the end timestamp
    let end_part = parts[1].split_whitespace().next().unwrap_or("");
    let end = Timestamp::parse_srt(end_part)?;

    Ok((start, end))
}

/// Parses SRT text formatting tags into styled text segments.
fn parse_srt_formatting(text: &str) -> Vec<StyledText> {
    let mut segments = Vec::new();
    let mut current_text = String::new();
    let mut current_style = TextStyle::default();
    let mut style_stack: Vec<TextStyle> = Vec::new();

    let tag_regex = Regex::new(r"<(/?)([bius]|font[^>]*)>").unwrap();
    let font_color_regex = Regex::new(r#"color\s*=\s*["']?([^"'\s>]+)["']?"#).unwrap();

    let mut last_end = 0;

    for cap in tag_regex.captures_iter(text) {
        let full_match = cap.get(0).unwrap();
        let start = full_match.start();
        let end = full_match.end();

        // Add text before this tag
        if start > last_end {
            let text_before = &text[last_end..start];
            if !text_before.is_empty() {
                current_text.push_str(text_before);
            }
        }

        let is_closing = &cap[1] == "/";
        let tag_name = &cap[2];

        // If we have accumulated text, save it with current style
        if !current_text.is_empty() {
            segments.push(StyledText::new(
                std::mem::take(&mut current_text),
                current_style.clone(),
            ));
        }

        if is_closing {
            // Pop style from stack
            if let Some(prev_style) = style_stack.pop() {
                current_style = prev_style;
            }
        } else {
            // Push current style and apply new formatting
            style_stack.push(current_style.clone());

            match tag_name {
                "b" => current_style.bold = true,
                "i" => current_style.italic = true,
                "u" => current_style.underline = true,
                "s" => current_style.strikethrough = true,
                font_tag if font_tag.starts_with("font") => {
                    if let Some(color_cap) = font_color_regex.captures(font_tag) {
                        if let Ok(color) = Color::from_hex(&color_cap[1]) {
                            current_style.color = Some(color);
                        }
                    }
                }
                _ => {}
            }
        }

        last_end = end;
    }

    // Add remaining text
    if last_end < text.len() {
        current_text.push_str(&text[last_end..]);
    }

    if !current_text.is_empty() {
        segments.push(StyledText::new(current_text, current_style));
    }

    // If no formatting was found, return the original text as plain
    if segments.is_empty() {
        segments.push(StyledText::plain(text.to_string()));
    }

    segments
}

/// Writes a subtitle track to SRT format.
pub fn write(track: &SubtitleTrack) -> String {
    let mut output = String::new();

    for (index, event) in track.events.iter().enumerate() {
        // Write index
        output.push_str(&format!("{}\n", index + 1));

        // Write timing
        output.push_str(&format!(
            "{} --> {}\n",
            event.start.to_srt_string(),
            event.end.to_srt_string()
        ));

        // Write text with formatting
        output.push_str(&format_styled_text_srt(&event.text));
        output.push('\n');

        // Blank line separator
        output.push('\n');
    }

    output
}

/// Writes a subtitle track to an SRT writer.
pub fn write_to<W: Write>(track: &SubtitleTrack, mut writer: W) -> SubtitleResult<()> {
    let content = write(track);
    writer
        .write_all(content.as_bytes())
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;
    Ok(())
}

/// Formats styled text segments into SRT formatting tags.
fn format_styled_text_srt(segments: &[StyledText]) -> String {
    let mut output = String::new();

    for segment in segments {
        let mut text = segment.text.clone();

        // Apply formatting tags (innermost first)
        if let Some(color) = &segment.style.color {
            text = format!("<font color=\"{}\">{}</font>", color.to_hex(), text);
        }
        if segment.style.strikethrough {
            text = format!("<s>{}</s>", text);
        }
        if segment.style.underline {
            text = format!("<u>{}</u>", text);
        }
        if segment.style.italic {
            text = format!("<i>{}</i>", text);
        }
        if segment.style.bold {
            text = format!("<b>{}</b>", text);
        }

        output.push_str(&text);
    }

    output
}

/// Strips all SRT formatting tags from text.
pub fn strip_formatting(text: &str) -> String {
    let tag_regex = Regex::new(r"<[^>]+>").unwrap();
    tag_regex.replace_all(text, "").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SRT: &str = r#"1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:08,500
This is a <b>bold</b> test.

3
00:00:10,000 --> 00:00:15,000
Multiple lines
of text here.

"#;

    #[test]
    fn test_parse_simple_srt() {
        let track = parse(SAMPLE_SRT).unwrap();
        assert_eq!(track.events.len(), 3);

        assert_eq!(track.events[0].start, Timestamp::new(0, 0, 1, 0));
        assert_eq!(track.events[0].end, Timestamp::new(0, 0, 4, 0));
        assert_eq!(track.events[0].plain_text(), "Hello, world!");

        assert_eq!(track.events[1].start, Timestamp::new(0, 0, 5, 0));
        assert_eq!(track.events[1].end, Timestamp::new(0, 0, 8, 500));

        assert_eq!(track.events[2].plain_text(), "Multiple lines\nof text here.");
    }

    #[test]
    fn test_parse_formatting() {
        let srt = r#"1
00:00:00,000 --> 00:00:01,000
<b>Bold</b> and <i>italic</i>

"#;
        let track = parse(srt).unwrap();
        assert_eq!(track.events.len(), 1);

        let segments = &track.events[0].text;
        assert!(segments.len() >= 2);
    }

    #[test]
    fn test_parse_color() {
        let srt = r##"1
00:00:00,000 --> 00:00:01,000
<font color="#FF0000">Red text</font>

"##;
        let track = parse(srt).unwrap();
        assert_eq!(track.events.len(), 1);

        let segments = &track.events[0].text;
        let colored_segment = segments
            .iter()
            .find(|s| s.text.contains("Red text"))
            .unwrap();
        assert_eq!(colored_segment.style.color, Some(Color::RED));
    }

    #[test]
    fn test_write_srt() {
        let mut track = SubtitleTrack::new();
        track.add_event(SubtitleEvent::new(
            Timestamp::new(0, 0, 1, 0),
            Timestamp::new(0, 0, 4, 0),
            "Hello, world!",
        ));

        let output = write(&track);
        assert!(output.contains("1\n"));
        assert!(output.contains("00:00:01,000 --> 00:00:04,000"));
        assert!(output.contains("Hello, world!"));
    }

    #[test]
    fn test_write_with_formatting() {
        let mut track = SubtitleTrack::new();

        let styled_text = vec![
            StyledText::new("Bold text", TextStyle::new().with_bold(true)),
            StyledText::plain(" and "),
            StyledText::new("italic text", TextStyle::new().with_italic(true)),
        ];

        track.add_event(SubtitleEvent::with_styled_text(
            Timestamp::new(0, 0, 1, 0),
            Timestamp::new(0, 0, 4, 0),
            styled_text,
        ));

        let output = write(&track);
        assert!(output.contains("<b>Bold text</b>"));
        assert!(output.contains("<i>italic text</i>"));
    }

    #[test]
    fn test_roundtrip() {
        let track = parse(SAMPLE_SRT).unwrap();
        let output = write(&track);
        let reparsed = parse(&output).unwrap();

        assert_eq!(track.events.len(), reparsed.events.len());
        for (orig, new) in track.events.iter().zip(reparsed.events.iter()) {
            assert_eq!(orig.start, new.start);
            assert_eq!(orig.end, new.end);
        }
    }

    #[test]
    fn test_strip_formatting() {
        let text = "<b>Bold</b> and <i>italic</i>";
        assert_eq!(strip_formatting(text), "Bold and italic");
    }

    #[test]
    fn test_parse_timing_with_position() {
        // Some SRT files include position info after the timing
        let srt = r#"1
00:00:01,000 --> 00:00:04,000 X1:100 X2:200 Y1:50 Y2:100
Hello, world!

"#;
        let track = parse(srt).unwrap();
        assert_eq!(track.events.len(), 1);
        assert_eq!(track.events[0].start, Timestamp::new(0, 0, 1, 0));
        assert_eq!(track.events[0].end, Timestamp::new(0, 0, 4, 0));
    }
}
