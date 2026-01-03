//! WebVTT (Web Video Text Tracks) subtitle format parser and writer.
//!
//! WebVTT is a W3C standard for displaying timed text tracks.
//! It supports:
//! - Cue timing with optional settings
//! - Text styling with CSS-like tags
//! - Regions and positioning
//! - Voice spans and speaker annotations

use crate::types::{
    Alignment, Color, Position, StyledText, SubtitleError, SubtitleEvent, SubtitleResult,
    SubtitleTrack, TextStyle, Timestamp,
};
use regex::Regex;
use std::io::{BufRead, Write};

/// WebVTT file header.
const WEBVTT_HEADER: &str = "WEBVTT";

/// Parses WebVTT content from a string.
pub fn parse(content: &str) -> SubtitleResult<SubtitleTrack> {
    let mut track = SubtitleTrack::new();
    let mut lines = content.lines().peekable();

    // Check for WEBVTT header
    let first_line = lines.next().ok_or_else(|| {
        SubtitleError::InvalidFormat("Empty WebVTT file".to_string())
    })?;

    if !first_line.trim().starts_with(WEBVTT_HEADER) {
        return Err(SubtitleError::InvalidFormat(
            "WebVTT file must start with 'WEBVTT'".to_string(),
        ));
    }

    // Parse optional header metadata
    while let Some(line) = lines.peek() {
        let line = line.trim();
        if line.is_empty() {
            lines.next();
            break;
        }
        // Skip header lines (like "WEBVTT - Title" or header metadata)
        lines.next();
    }

    // Parse cues
    while lines.peek().is_some() {
        // Skip empty lines and comments
        while let Some(line) = lines.peek() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("NOTE") {
                lines.next();
                // Skip multi-line notes
                if line.starts_with("NOTE") {
                    while let Some(note_line) = lines.peek() {
                        if note_line.trim().is_empty() {
                            break;
                        }
                        lines.next();
                    }
                }
            } else if line.starts_with("STYLE") || line.starts_with("REGION") {
                // Skip style and region blocks for now
                lines.next();
                while let Some(block_line) = lines.peek() {
                    if block_line.trim().is_empty() {
                        break;
                    }
                    lines.next();
                }
            } else {
                break;
            }
        }

        if lines.peek().is_none() {
            break;
        }

        // Check if next line is a cue identifier or timing line
        let first_cue_line = match lines.next() {
            Some(line) => line.trim(),
            None => break,
        };

        let timing_line = if first_cue_line.contains("-->") {
            // This is the timing line directly
            first_cue_line
        } else {
            // This was a cue identifier, get the timing line
            match lines.next() {
                Some(line) => line.trim(),
                None => break,
            }
        };

        // Parse timing line
        let (start, end, settings) = match parse_timing_line(timing_line) {
            Ok(result) => result,
            Err(_) => continue, // Skip invalid cues
        };

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
            let styled_text = parse_vtt_formatting(&raw_text);

            let mut event = SubtitleEvent::with_styled_text(start, end, styled_text);

            // Apply cue settings
            if let Some(position) = parse_cue_settings(&settings) {
                event.position = Some(position);
            }

            track.add_event(event);
        }
    }

    Ok(track)
}

/// Parses WebVTT content from a reader.
pub fn parse_reader<R: BufRead>(reader: R) -> SubtitleResult<SubtitleTrack> {
    let content: String = reader
        .lines()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| SubtitleError::IoError(e.to_string()))?
        .join("\n");

    parse(&content)
}

/// Parses a timing line in WebVTT format.
fn parse_timing_line(line: &str) -> SubtitleResult<(Timestamp, Timestamp, String)> {
    let parts: Vec<&str> = line.split("-->").collect();
    if parts.len() != 2 {
        return Err(SubtitleError::ParseError(format!(
            "Invalid timing line: {}",
            line
        )));
    }

    let start = Timestamp::parse_vtt(parts[0].trim())?;

    // The end part may contain cue settings after the timestamp
    let end_and_settings = parts[1].trim();
    let end_parts: Vec<&str> = end_and_settings.splitn(2, char::is_whitespace).collect();

    let end = Timestamp::parse_vtt(end_parts[0])?;
    let settings = end_parts.get(1).unwrap_or(&"").to_string();

    Ok((start, end, settings))
}

/// Parses cue settings into a Position.
fn parse_cue_settings(settings: &str) -> Option<Position> {
    if settings.is_empty() {
        return None;
    }

    let mut x = 50.0;
    let mut y = 100.0;
    let mut alignment = Alignment::BottomCenter;

    for setting in settings.split_whitespace() {
        if let Some((key, value)) = setting.split_once(':') {
            match key {
                "position" => {
                    let value = value.trim_end_matches('%');
                    if let Ok(pos) = value.parse::<f32>() {
                        x = pos;
                    }
                }
                "line" => {
                    let value = value.trim_end_matches('%');
                    if let Ok(pos) = value.parse::<f32>() {
                        y = pos;
                    }
                }
                "align" => {
                    alignment = match value {
                        "start" | "left" => Alignment::BottomLeft,
                        "center" | "middle" => Alignment::BottomCenter,
                        "end" | "right" => Alignment::BottomRight,
                        _ => Alignment::BottomCenter,
                    };
                }
                _ => {}
            }
        }
    }

    Some(Position::new(x, y, alignment))
}

/// Parses WebVTT text formatting into styled text segments.
fn parse_vtt_formatting(text: &str) -> Vec<StyledText> {
    let mut segments = Vec::new();
    let mut current_text = String::new();
    let mut current_style = TextStyle::default();
    let mut style_stack: Vec<TextStyle> = Vec::new();
    let mut speaker: Option<String> = None;

    // Regex for WebVTT tags
    let tag_regex = Regex::new(r"<(/?)([a-zA-Z][a-zA-Z0-9]*(?:\.[^\s>]*)?)(?:\s+([^>]*))?>").unwrap();
    let class_regex = Regex::new(r"\.([^\s.>]+)").unwrap();

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

        // If we have accumulated text, save it with current style
        if !current_text.is_empty() {
            let mut styled = StyledText::new(
                std::mem::take(&mut current_text),
                current_style.clone(),
            );
            if let Some(ref s) = speaker {
                // Store speaker info in a way that can be retrieved
                styled.style.font_name = Some(format!("__speaker__{}", s));
            }
            segments.push(styled);
        }

        let is_closing = &cap[1] == "/";
        let tag_name = &cap[2];
        let base_tag: &str = tag_name.split('.').next().unwrap_or(tag_name);

        if is_closing {
            // Pop style from stack
            if let Some(prev_style) = style_stack.pop() {
                current_style = prev_style;
            }
            if base_tag == "v" {
                speaker = None;
            }
        } else {
            // Push current style and apply new formatting
            style_stack.push(current_style.clone());

            match base_tag {
                "b" => current_style.bold = true,
                "i" => current_style.italic = true,
                "u" => current_style.underline = true,
                "c" => {
                    // Check for color class
                    for class_cap in class_regex.captures_iter(tag_name) {
                        let class_name = &class_cap[1];
                        if let Some(color) = parse_color_class(class_name) {
                            current_style.color = Some(color);
                        }
                    }
                }
                "v" => {
                    // Voice tag - extract speaker name
                    if let Some(annotation) = cap.get(3) {
                        speaker = Some(annotation.as_str().to_string());
                    }
                }
                "lang" => {
                    // Language tag - we could store this as metadata
                }
                "ruby" | "rt" => {
                    // Ruby annotations - pass through
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

/// Parses a color class name into a Color.
fn parse_color_class(class: &str) -> Option<Color> {
    match class.to_lowercase().as_str() {
        "white" => Some(Color::WHITE),
        "black" => Some(Color::BLACK),
        "red" => Some(Color::RED),
        "green" => Some(Color::GREEN),
        "blue" => Some(Color::BLUE),
        "yellow" => Some(Color::YELLOW),
        "cyan" => Some(Color::CYAN),
        "magenta" => Some(Color::MAGENTA),
        "lime" => Some(Color::rgb(0, 255, 0)),
        "aqua" => Some(Color::CYAN),
        "fuchsia" => Some(Color::MAGENTA),
        "bg_white" | "bg_black" | "bg_red" | "bg_green" | "bg_blue" | "bg_yellow" | "bg_cyan"
        | "bg_magenta" => None, // Background colors not directly supported in our model
        _ => {
            // Try to parse as hex color
            if class.starts_with("color") || class.starts_with('#') {
                Color::from_hex(class.trim_start_matches("color")).ok()
            } else {
                None
            }
        }
    }
}

/// Writes a subtitle track to WebVTT format.
pub fn write(track: &SubtitleTrack) -> String {
    let mut output = String::new();

    // Write header
    output.push_str(WEBVTT_HEADER);
    if let Some(title) = &track.title {
        output.push_str(" - ");
        output.push_str(title);
    }
    output.push_str("\n\n");

    // Write cues
    for (index, event) in track.events.iter().enumerate() {
        // Optional cue identifier
        output.push_str(&format!("{}\n", index + 1));

        // Write timing
        output.push_str(&format!(
            "{} --> {}",
            event.start.to_vtt_string(),
            event.end.to_vtt_string()
        ));

        // Write position settings if present
        if let Some(pos) = &event.position {
            output.push_str(&format!(" position:{}%", pos.x as i32));
            if pos.y != 100.0 {
                output.push_str(&format!(" line:{}%", pos.y as i32));
            }
            let align = match pos.alignment {
                Alignment::BottomLeft | Alignment::MiddleLeft | Alignment::TopLeft => "start",
                Alignment::BottomCenter | Alignment::MiddleCenter | Alignment::TopCenter => {
                    "center"
                }
                Alignment::BottomRight | Alignment::MiddleRight | Alignment::TopRight => "end",
            };
            if align != "center" {
                output.push_str(&format!(" align:{}", align));
            }
        }

        output.push('\n');

        // Write text with formatting
        output.push_str(&format_styled_text_vtt(&event.text));
        output.push('\n');

        // Blank line separator
        output.push('\n');
    }

    output
}

/// Writes a subtitle track to a WebVTT writer.
pub fn write_to<W: Write>(track: &SubtitleTrack, mut writer: W) -> SubtitleResult<()> {
    let content = write(track);
    writer
        .write_all(content.as_bytes())
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;
    Ok(())
}

/// Formats styled text segments into WebVTT formatting tags.
fn format_styled_text_vtt(segments: &[StyledText]) -> String {
    let mut output = String::new();

    for segment in segments {
        let mut text = segment.text.clone();

        // Apply formatting tags (innermost first)
        if let Some(color) = &segment.style.color {
            // Use color class if it matches a standard color
            let class = match *color {
                Color::WHITE => "white",
                Color::BLACK => "black",
                Color::RED => "red",
                Color::GREEN => "green",
                Color::BLUE => "blue",
                Color::YELLOW => "yellow",
                Color::CYAN => "cyan",
                Color::MAGENTA => "magenta",
                _ => "",
            };
            if !class.is_empty() {
                text = format!("<c.{}>{}</c>", class, text);
            }
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

/// Strips all WebVTT formatting tags from text.
pub fn strip_formatting(text: &str) -> String {
    let tag_regex = Regex::new(r"<[^>]+>").unwrap();
    tag_regex.replace_all(text, "").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_VTT: &str = r#"WEBVTT

1
00:00:01.000 --> 00:00:04.000
Hello, world!

2
00:00:05.000 --> 00:00:08.500
This is a <b>bold</b> test.

3
00:00:10.000 --> 00:00:15.000
Multiple lines
of text here.

"#;

    #[test]
    fn test_parse_simple_vtt() {
        let track = parse(SAMPLE_VTT).unwrap();
        assert_eq!(track.events.len(), 3);

        assert_eq!(track.events[0].start, Timestamp::new(0, 0, 1, 0));
        assert_eq!(track.events[0].end, Timestamp::new(0, 0, 4, 0));
        assert_eq!(track.events[0].plain_text(), "Hello, world!");

        assert_eq!(track.events[1].start, Timestamp::new(0, 0, 5, 0));
        assert_eq!(track.events[1].end, Timestamp::new(0, 0, 8, 500));

        assert_eq!(track.events[2].plain_text(), "Multiple lines\nof text here.");
    }

    #[test]
    fn test_parse_with_title() {
        let vtt = "WEBVTT - My Subtitles\n\n1\n00:00:00.000 --> 00:00:01.000\nTest\n";
        let track = parse(vtt).unwrap();
        assert_eq!(track.events.len(), 1);
    }

    #[test]
    fn test_parse_short_timestamp() {
        let vtt = "WEBVTT\n\n00:01.000 --> 00:04.000\nShort format\n";
        let track = parse(vtt).unwrap();
        assert_eq!(track.events.len(), 1);
        assert_eq!(track.events[0].start, Timestamp::new(0, 0, 1, 0));
    }

    #[test]
    fn test_parse_formatting() {
        let vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<b>Bold</b> and <i>italic</i>\n";
        let track = parse(vtt).unwrap();
        assert_eq!(track.events.len(), 1);

        let segments = &track.events[0].text;
        assert!(segments.len() >= 2);
    }

    #[test]
    fn test_parse_color_class() {
        let vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<c.red>Red text</c>\n";
        let track = parse(vtt).unwrap();
        assert_eq!(track.events.len(), 1);

        let segments = &track.events[0].text;
        let colored_segment = segments
            .iter()
            .find(|s| s.text.contains("Red text"))
            .unwrap();
        assert_eq!(colored_segment.style.color, Some(Color::RED));
    }

    #[test]
    fn test_parse_with_position() {
        let vtt =
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000 position:25% line:50% align:start\nPositioned\n";
        let track = parse(vtt).unwrap();
        assert_eq!(track.events.len(), 1);

        let pos = track.events[0].position.as_ref().unwrap();
        assert_eq!(pos.x, 25.0);
        assert_eq!(pos.y, 50.0);
        assert_eq!(pos.alignment, Alignment::BottomLeft);
    }

    #[test]
    fn test_parse_voice_tag() {
        let vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<v Speaker>Hello there</v>\n";
        let track = parse(vtt).unwrap();
        assert_eq!(track.events.len(), 1);
        assert_eq!(track.events[0].plain_text(), "Hello there");
    }

    #[test]
    fn test_parse_with_notes() {
        let vtt = r#"WEBVTT

NOTE This is a comment
that spans multiple lines

00:00:00.000 --> 00:00:01.000
Subtitle after note
"#;
        let track = parse(vtt).unwrap();
        assert_eq!(track.events.len(), 1);
    }

    #[test]
    fn test_write_vtt() {
        let mut track = SubtitleTrack::new();
        track.title = Some("Test".to_string());
        track.add_event(SubtitleEvent::new(
            Timestamp::new(0, 0, 1, 0),
            Timestamp::new(0, 0, 4, 0),
            "Hello, world!",
        ));

        let output = write(&track);
        assert!(output.starts_with("WEBVTT - Test"));
        assert!(output.contains("00:00:01.000 --> 00:00:04.000"));
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
    fn test_write_with_color() {
        let mut track = SubtitleTrack::new();

        let styled_text = vec![StyledText::new(
            "Red text",
            TextStyle::new().with_color(Color::RED),
        )];

        track.add_event(SubtitleEvent::with_styled_text(
            Timestamp::new(0, 0, 1, 0),
            Timestamp::new(0, 0, 4, 0),
            styled_text,
        ));

        let output = write(&track);
        assert!(output.contains("<c.red>Red text</c>"));
    }

    #[test]
    fn test_roundtrip() {
        let track = parse(SAMPLE_VTT).unwrap();
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
    fn test_invalid_header() {
        let result = parse("NOT WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest\n");
        assert!(result.is_err());
    }
}
