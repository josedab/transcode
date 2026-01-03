//! ASS/SSA (Advanced SubStation Alpha / SubStation Alpha) subtitle format parser and writer.
//!
//! ASS/SSA is a feature-rich subtitle format commonly used for anime and karaoke.
//! It supports:
//! - Multiple named styles
//! - Rich text formatting with override tags
//! - Positioning and animation
//! - Script information and metadata

use crate::types::{
    Alignment, BorderStyle, Color, NamedStyle, StyledText, SubtitleError, SubtitleEvent,
    SubtitleResult, SubtitleTrack, TextStyle, Timestamp, TrackMetadata,
};
use regex::Regex;
use std::io::{BufRead, Write};

/// Parses ASS/SSA content from a string.
pub fn parse(content: &str) -> SubtitleResult<SubtitleTrack> {
    let mut track = SubtitleTrack::new();
    let mut current_section = String::new();
    let mut style_format: Vec<String> = Vec::new();
    let mut event_format: Vec<String> = Vec::new();

    for line in content.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with(';') {
            continue;
        }

        // Check for section headers
        if line.starts_with('[') && line.ends_with(']') {
            current_section = line[1..line.len() - 1].to_lowercase();
            continue;
        }

        match current_section.as_str() {
            "script info" => {
                parse_script_info(line, &mut track.metadata);
                if let Some((key, value)) = line.split_once(':') {
                    if key.trim().to_lowercase() == "title" {
                        track.title = Some(value.trim().to_string());
                    }
                }
            }
            "v4 styles" | "v4+ styles" => {
                if let Some(format_str) = line.strip_prefix("Format:") {
                    style_format = format_str.split(',').map(|s| s.trim().to_lowercase()).collect();
                } else if let Some(style_str) = line.strip_prefix("Style:") {
                    if let Ok(style) = parse_style_line(style_str, &style_format) {
                        track.add_style(style);
                    }
                }
            }
            "events" => {
                if let Some(format_str) = line.strip_prefix("Format:") {
                    event_format = format_str.split(',').map(|s| s.trim().to_lowercase()).collect();
                } else if let Some(dialogue_str) = line.strip_prefix("Dialogue:") {
                    if let Ok(event) = parse_dialogue_line(dialogue_str, &event_format) {
                        track.add_event(event);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(track)
}

/// Parses ASS/SSA content from a reader.
pub fn parse_reader<R: BufRead>(reader: R) -> SubtitleResult<SubtitleTrack> {
    let content: String = reader
        .lines()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| SubtitleError::IoError(e.to_string()))?
        .join("\n");

    parse(&content)
}

/// Parses script info section entries.
fn parse_script_info(line: &str, metadata: &mut TrackMetadata) {
    if let Some((key, value)) = line.split_once(':') {
        let key = key.trim().to_lowercase();
        let value = value.trim();

        match key.as_str() {
            "playresx" => metadata.play_res_x = value.parse().ok(),
            "playresy" => metadata.play_res_y = value.parse().ok(),
            "timer" => metadata.timer = value.parse().ok(),
            "wrapstyle" => metadata.wrap_style = value.parse().ok(),
            "collisions" => metadata.collisions = Some(value.to_string()),
            _ => {}
        }
    }
}

/// Parses a style definition line.
fn parse_style_line(line: &str, format: &[String]) -> SubtitleResult<NamedStyle> {
    let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if values.len() < format.len() {
        return Err(SubtitleError::ParseError(
            "Style line has fewer fields than format".to_string(),
        ));
    }

    let get_field = |name: &str| -> Option<&str> {
        format
            .iter()
            .position(|f| f == name)
            .and_then(|i| values.get(i).copied())
    };

    let name = get_field("name").unwrap_or("Default").to_string();
    let mut style = NamedStyle::new(name);

    if let Some(font) = get_field("fontname") {
        style.style.font_name = Some(font.to_string());
    }

    if let Some(size) = get_field("fontsize") {
        style.style.font_size = size.parse().ok();
    }

    if let Some(color) = get_field("primarycolour").or_else(|| get_field("primarycolor")) {
        style.style.color = Color::from_ass(color).ok();
    }

    if let Some(bold) = get_field("bold") {
        style.style.bold = bold != "0" && bold != "-1";
    }

    if let Some(italic) = get_field("italic") {
        style.style.italic = italic != "0" && italic != "-1";
    }

    if let Some(underline) = get_field("underline") {
        style.style.underline = underline != "0" && underline != "-1";
    }

    if let Some(strikeout) = get_field("strikeout") {
        style.style.strikethrough = strikeout != "0" && strikeout != "-1";
    }

    if let Some(borderstyle) = get_field("borderstyle") {
        style.border_style = match borderstyle {
            "3" => BorderStyle::OpaqueBox,
            _ => BorderStyle::OutlineAndShadow,
        };
    }

    if let Some(outline) = get_field("outline") {
        style.outline = outline.parse().unwrap_or(2.0);
    }

    if let Some(shadow) = get_field("shadow") {
        style.shadow = shadow.parse().unwrap_or(2.0);
    }

    if let Some(alignment) = get_field("alignment") {
        style.alignment = Alignment::from_ass_alignment(alignment.parse().unwrap_or(2));
    }

    let margin_l: i32 = get_field("marginl").and_then(|s| s.parse().ok()).unwrap_or(10);
    let margin_r: i32 = get_field("marginr").and_then(|s| s.parse().ok()).unwrap_or(10);
    let margin_v: i32 = get_field("marginv").and_then(|s| s.parse().ok()).unwrap_or(10);
    style.margins = (margin_l, margin_r, margin_v);

    Ok(style)
}

/// Parses a dialogue line.
fn parse_dialogue_line(line: &str, format: &[String]) -> SubtitleResult<SubtitleEvent> {
    // Split by comma, but the last field (Text) may contain commas
    let mut values: Vec<&str> = Vec::new();
    let mut remaining = line;

    for (i, _) in format.iter().enumerate() {
        if i == format.len() - 1 {
            // Last field takes everything remaining
            values.push(remaining.trim());
        } else if let Some(pos) = remaining.find(',') {
            values.push(remaining[..pos].trim());
            remaining = &remaining[pos + 1..];
        } else {
            values.push(remaining.trim());
            remaining = "";
        }
    }

    let get_field = |name: &str| -> Option<&str> {
        format
            .iter()
            .position(|f| f == name)
            .and_then(|i| values.get(i).copied())
    };

    let start_str = get_field("start").ok_or_else(|| {
        SubtitleError::ParseError("Missing start time in dialogue line".to_string())
    })?;
    let end_str = get_field("end")
        .ok_or_else(|| SubtitleError::ParseError("Missing end time in dialogue line".to_string()))?;
    let text = get_field("text").unwrap_or("");

    let start = Timestamp::parse_ass(start_str)?;
    let end = Timestamp::parse_ass(end_str)?;

    let styled_text = parse_ass_formatting(text);
    let mut event = SubtitleEvent::with_styled_text(start, end, styled_text);

    if let Some(layer) = get_field("layer") {
        event.layer = layer.parse().unwrap_or(0);
    }

    if let Some(style) = get_field("style") {
        event.style_name = Some(style.to_string());
    }

    if let Some(name) = get_field("name") {
        if !name.is_empty() {
            event.speaker = Some(name.to_string());
        }
    }

    Ok(event)
}

/// Parses ASS override tags into styled text segments.
fn parse_ass_formatting(text: &str) -> Vec<StyledText> {
    let mut segments = Vec::new();
    let mut current_text = String::new();
    let mut current_style = TextStyle::default();
    let mut style_stack: Vec<TextStyle> = vec![TextStyle::default()];

    // Replace \N and \n with actual newlines
    let text = text.replace("\\N", "\n").replace("\\n", "\n");

    let override_regex = Regex::new(r"\{([^}]*)\}").unwrap();
    let mut last_end = 0;

    for cap in override_regex.captures_iter(&text) {
        let full_match = cap.get(0).unwrap();
        let start = full_match.start();
        let end = full_match.end();

        // Add text before this override block
        if start > last_end {
            let text_before = &text[last_end..start];
            if !text_before.is_empty() {
                current_text.push_str(text_before);
            }
        }

        // If we have accumulated text, save it with current style
        if !current_text.is_empty() {
            segments.push(StyledText::new(
                std::mem::take(&mut current_text),
                current_style.clone(),
            ));
        }

        // Parse override tags
        let overrides = &cap[1];
        current_style = parse_override_tags(overrides, &style_stack);
        style_stack.push(current_style.clone());

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

/// Parses individual override tags.
fn parse_override_tags(overrides: &str, style_stack: &[TextStyle]) -> TextStyle {
    let base_style = style_stack.last().cloned().unwrap_or_default();
    let mut style = base_style;

    // Parse individual tags
    let tag_regex = Regex::new(r"\\([a-zA-Z]+)([^\\]*)").unwrap();

    for cap in tag_regex.captures_iter(overrides) {
        let tag = &cap[1];
        let value = cap[2].trim();

        match tag {
            "b" => {
                style.bold = value != "0";
            }
            "i" => {
                style.italic = value != "0";
            }
            "u" => {
                style.underline = value != "0";
            }
            "s" => {
                style.strikethrough = value != "0";
            }
            "c" | "1c" => {
                if let Ok(color) = Color::from_ass(value) {
                    style.color = Some(color);
                }
            }
            "fn" => {
                if !value.is_empty() {
                    style.font_name = Some(value.to_string());
                }
            }
            "fs" => {
                if let Ok(size) = value.parse::<f32>() {
                    style.font_size = Some(size);
                }
            }
            "r" => {
                // Reset to base style (or specified style)
                style = TextStyle::default();
            }
            _ => {}
        }
    }

    style
}

/// Writes a subtitle track to ASS format.
pub fn write(track: &SubtitleTrack) -> String {
    write_internal(track, true)
}

/// Writes a subtitle track to SSA format.
pub fn write_ssa(track: &SubtitleTrack) -> String {
    write_internal(track, false)
}

/// Internal writer for both ASS and SSA formats.
fn write_internal(track: &SubtitleTrack, is_ass: bool) -> String {
    let mut output = String::new();

    // Script Info section
    output.push_str("[Script Info]\n");
    output.push_str("; Script generated by transcode-subtitle\n");
    if let Some(title) = &track.title {
        output.push_str(&format!("Title: {}\n", title));
    }
    output.push_str(if is_ass {
        "ScriptType: v4.00+\n"
    } else {
        "ScriptType: v4.00\n"
    });

    if let Some(res_x) = track.metadata.play_res_x {
        output.push_str(&format!("PlayResX: {}\n", res_x));
    }
    if let Some(res_y) = track.metadata.play_res_y {
        output.push_str(&format!("PlayResY: {}\n", res_y));
    }
    if let Some(timer) = track.metadata.timer {
        output.push_str(&format!("Timer: {:.4}\n", timer));
    }
    if let Some(wrap) = track.metadata.wrap_style {
        output.push_str(&format!("WrapStyle: {}\n", wrap));
    }

    output.push('\n');

    // Styles section
    output.push_str(if is_ass {
        "[V4+ Styles]\n"
    } else {
        "[V4 Styles]\n"
    });
    output.push_str(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n",
    );

    if track.styles.is_empty() {
        // Write a default style
        output.push_str("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n");
    } else {
        for style in &track.styles {
            output.push_str(&format_style(style));
            output.push('\n');
        }
    }

    output.push('\n');

    // Events section
    output.push_str("[Events]\n");
    output.push_str("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n");

    for event in &track.events {
        output.push_str(&format_dialogue(event));
        output.push('\n');
    }

    output
}

/// Formats a style definition line.
fn format_style(style: &NamedStyle) -> String {
    let font_name = style.style.font_name.as_deref().unwrap_or("Arial");
    let font_size = style.style.font_size.unwrap_or(20.0) as i32;
    let primary_color = style.style.color.unwrap_or(Color::WHITE).to_ass();
    let bold = if style.style.bold { "-1" } else { "0" };
    let italic = if style.style.italic { "-1" } else { "0" };
    let underline = if style.style.underline { "-1" } else { "0" };
    let strikeout = if style.style.strikethrough { "-1" } else { "0" };
    let border_style = match style.border_style {
        BorderStyle::OutlineAndShadow => 1,
        BorderStyle::OpaqueBox => 3,
    };
    let alignment = style.alignment.to_ass_alignment();

    format!(
        "Style: {},{},{},{},&H000000FF,&H00000000,&H00000000,{},{},{},{},100,100,0,0,{},{:.0},{:.0},{},{},{},{},1",
        style.name,
        font_name,
        font_size,
        primary_color,
        bold,
        italic,
        underline,
        strikeout,
        border_style,
        style.outline,
        style.shadow,
        alignment,
        style.margins.0,
        style.margins.1,
        style.margins.2
    )
}

/// Formats a dialogue line.
fn format_dialogue(event: &SubtitleEvent) -> String {
    let style_name = event.style_name.as_deref().unwrap_or("Default");
    let speaker = event.speaker.as_deref().unwrap_or("");
    let text = format_styled_text_ass(&event.text);

    format!(
        "Dialogue: {},{},{},{},{},0,0,0,,{}",
        event.layer,
        event.start.to_ass_string(),
        event.end.to_ass_string(),
        style_name,
        speaker,
        text
    )
}

/// Formats styled text segments into ASS override tags.
fn format_styled_text_ass(segments: &[StyledText]) -> String {
    let mut output = String::new();
    let mut prev_style = TextStyle::default();

    for segment in segments {
        let style = &segment.style;

        // Build override tags for style changes
        let mut overrides = String::new();

        if style.bold != prev_style.bold {
            overrides.push_str(if style.bold { "\\b1" } else { "\\b0" });
        }
        if style.italic != prev_style.italic {
            overrides.push_str(if style.italic { "\\i1" } else { "\\i0" });
        }
        if style.underline != prev_style.underline {
            overrides.push_str(if style.underline { "\\u1" } else { "\\u0" });
        }
        if style.strikethrough != prev_style.strikethrough {
            overrides.push_str(if style.strikethrough { "\\s1" } else { "\\s0" });
        }
        if style.color != prev_style.color {
            if let Some(color) = &style.color {
                overrides.push_str(&format!("\\c{}", color.to_ass()));
            } else {
                overrides.push_str("\\r");
            }
        }
        if style.font_name != prev_style.font_name {
            if let Some(name) = &style.font_name {
                overrides.push_str(&format!("\\fn{}", name));
            }
        }
        if style.font_size != prev_style.font_size {
            if let Some(size) = style.font_size {
                overrides.push_str(&format!("\\fs{}", size as i32));
            }
        }

        if !overrides.is_empty() {
            output.push('{');
            output.push_str(&overrides);
            output.push('}');
        }

        // Replace newlines with \N
        let text = segment.text.replace('\n', "\\N");
        output.push_str(&text);

        prev_style = style.clone();
    }

    output
}

/// Writes a subtitle track to an ASS writer.
pub fn write_to<W: Write>(track: &SubtitleTrack, mut writer: W) -> SubtitleResult<()> {
    let content = write(track);
    writer
        .write_all(content.as_bytes())
        .map_err(|e| SubtitleError::IoError(e.to_string()))?;
    Ok(())
}

/// Strips all ASS formatting tags from text.
pub fn strip_formatting(text: &str) -> String {
    let override_regex = Regex::new(r"\{[^}]*\}").unwrap();
    let result = override_regex.replace_all(text, "");
    result.replace("\\N", "\n").replace("\\n", "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_ASS: &str = r#"[Script Info]
Title: Test Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello, world!
Dialogue: 0,0:00:05.00,0:00:08.50,Default,,0,0,0,,This is a {\b1}bold{\b0} test.
Dialogue: 0,0:00:10.00,0:00:15.00,Default,,0,0,0,,Multiple lines\Nof text here.
"#;

    #[test]
    fn test_parse_simple_ass() {
        let track = parse(SAMPLE_ASS).unwrap();

        assert_eq!(track.title, Some("Test Subtitles".to_string()));
        assert_eq!(track.metadata.play_res_x, Some(1920));
        assert_eq!(track.metadata.play_res_y, Some(1080));

        assert_eq!(track.styles.len(), 1);
        assert_eq!(track.styles[0].name, "Default");

        assert_eq!(track.events.len(), 3);
        assert_eq!(track.events[0].start, Timestamp::new(0, 0, 1, 0));
        assert_eq!(track.events[0].end, Timestamp::new(0, 0, 4, 0));
        assert_eq!(track.events[0].plain_text(), "Hello, world!");

        assert!(track.events[2].plain_text().contains('\n'));
    }

    #[test]
    fn test_parse_formatting() {
        let ass = r#"[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\b1}Bold{\b0} and {\i1}italic{\i0}
"#;

        let track = parse(ass).unwrap();
        assert_eq!(track.events.len(), 1);

        let segments = &track.events[0].text;
        assert!(segments.len() >= 2);
    }

    #[test]
    fn test_parse_color() {
        let ass = r#"[Script Info]
ScriptType: v4.00+

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\c&H0000FF&}Red text
"#;

        let track = parse(ass).unwrap();
        assert_eq!(track.events.len(), 1);

        let segments = &track.events[0].text;
        let colored_segment = segments
            .iter()
            .find(|s| s.text.contains("Red text"))
            .unwrap();
        assert!(colored_segment.style.color.is_some());
    }

    #[test]
    fn test_write_ass() {
        let mut track = SubtitleTrack::new();
        track.title = Some("Test".to_string());
        track.add_event(SubtitleEvent::new(
            Timestamp::new(0, 0, 1, 0),
            Timestamp::new(0, 0, 4, 0),
            "Hello, world!",
        ));

        let output = write(&track);
        assert!(output.contains("[Script Info]"));
        assert!(output.contains("Title: Test"));
        assert!(output.contains("[V4+ Styles]"));
        assert!(output.contains("[Events]"));
        assert!(output.contains("Dialogue:"));
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
        assert!(output.contains("{\\b1}Bold text"));
        assert!(output.contains("{\\i1}italic text"));
    }

    #[test]
    fn test_roundtrip() {
        let track = parse(SAMPLE_ASS).unwrap();
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
        let text = "{\\b1}Bold{\\b0} and {\\i1}italic{\\i0}";
        assert_eq!(strip_formatting(text), "Bold and italic");

        let text_with_newline = "Line 1\\NLine 2";
        assert_eq!(strip_formatting(text_with_newline), "Line 1\nLine 2");
    }

    #[test]
    fn test_style_parsing() {
        let track = parse(SAMPLE_ASS).unwrap();
        assert_eq!(track.styles.len(), 1);

        let style = &track.styles[0];
        assert_eq!(style.name, "Default");
        assert_eq!(style.style.font_name, Some("Arial".to_string()));
        assert_eq!(style.style.font_size, Some(20.0));
        assert_eq!(style.alignment, Alignment::BottomCenter);
    }
}
