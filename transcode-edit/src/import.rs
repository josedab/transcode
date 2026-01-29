//! EDL import and FCP XML support.
//!
//! Provides parsers for CMX 3600 EDL, CSV EDL, and Final Cut Pro XML formats,
//! as well as FCP XML export.

#![allow(dead_code)]

use crate::edl::{EditDecisionList, EdlEntry};
use crate::error::{Error, Result};
use crate::ops::{Transition, TransitionType};

/// Result of parsing an EDL or FCP XML file.
#[derive(Debug, Clone)]
pub struct ParsedEdl {
    pub title: Option<String>,
    pub entries: Vec<EdlEntry>,
}

// ---------------------------------------------------------------------------
// Timecode helpers
// ---------------------------------------------------------------------------

/// Convert a `HH:MM:SS:FF` timecode string to seconds at the given frame rate.
pub fn parse_timecode(tc: &str, fps: f64) -> Result<f64> {
    let parts: Vec<&str> = tc.split(':').collect();
    if parts.len() != 4 {
        return Err(Error::EdlParse {
            line: 0,
            message: format!("Invalid timecode format: '{tc}'"),
        });
    }
    let h: u64 = parts[0].parse().map_err(|_| Error::EdlParse {
        line: 0,
        message: format!("Invalid hours in timecode: '{tc}'"),
    })?;
    let m: u64 = parts[1].parse().map_err(|_| Error::EdlParse {
        line: 0,
        message: format!("Invalid minutes in timecode: '{tc}'"),
    })?;
    let s: u64 = parts[2].parse().map_err(|_| Error::EdlParse {
        line: 0,
        message: format!("Invalid seconds in timecode: '{tc}'"),
    })?;
    let f: u64 = parts[3].parse().map_err(|_| Error::EdlParse {
        line: 0,
        message: format!("Invalid frames in timecode: '{tc}'"),
    })?;
    let total_seconds = h as f64 * 3600.0 + m as f64 * 60.0 + s as f64 + f as f64 / fps;
    Ok(total_seconds)
}

/// Convert a frame count to seconds at the given frame rate.
pub fn frames_to_seconds(frames: u64, fps: f64) -> f64 {
    frames as f64 / fps
}

// ---------------------------------------------------------------------------
// CMX 3600 parser
// ---------------------------------------------------------------------------

/// Parse a CMX 3600 EDL string into a [`ParsedEdl`].
///
/// Expects the standard format produced by [`EditDecisionList::to_string`] with
/// `EdlFormat::Cmx3600`.  Comment lines starting with `*` and blank lines are
/// skipped.
pub fn parse_cmx3600(input: &str) -> Result<ParsedEdl> {
    let mut title: Option<String> = None;
    let mut entries: Vec<EdlEntry> = Vec::new();
    let fps = 30.0;

    for (line_idx, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();

        // Skip blanks and comments
        if line.is_empty() || line.starts_with('*') {
            continue;
        }

        // Title line
        if let Some(rest) = line.strip_prefix("TITLE:") {
            title = Some(rest.trim().to_string());
            continue;
        }

        // Event line: 001  SOURCE  V  C  HH:MM:SS:FF HH:MM:SS:FF HH:MM:SS:FF HH:MM:SS:FF
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 8 {
            continue; // not a recognizable event line
        }

        // First token must be a numeric event number
        if tokens[0].parse::<u32>().is_err() {
            continue;
        }

        let source = tokens[1].to_string();
        // tokens[2] is the track type (V, A, etc.) — we skip it
        let transition_code = tokens[3];

        let transition = match transition_code {
            "C" => None,
            "D" => Some(Transition::new(TransitionType::Dissolve, 1.0)),
            "W" => Some(Transition::new(TransitionType::Wipe, 1.0)),
            _ => None,
        };

        let source_in = parse_timecode(tokens[4], fps).map_err(|_| Error::EdlParse {
            line: line_idx + 1,
            message: format!("Invalid source-in timecode: '{}'", tokens[4]),
        })?;
        let source_out = parse_timecode(tokens[5], fps).map_err(|_| Error::EdlParse {
            line: line_idx + 1,
            message: format!("Invalid source-out timecode: '{}'", tokens[5]),
        })?;
        let record_in = parse_timecode(tokens[6], fps).map_err(|_| Error::EdlParse {
            line: line_idx + 1,
            message: format!("Invalid record-in timecode: '{}'", tokens[6]),
        })?;
        let record_out = parse_timecode(tokens[7], fps).map_err(|_| Error::EdlParse {
            line: line_idx + 1,
            message: format!("Invalid record-out timecode: '{}'", tokens[7]),
        })?;

        entries.push(EdlEntry {
            source,
            source_in,
            source_out,
            record_in,
            record_out,
            transition,
        });
    }

    Ok(ParsedEdl { title, entries })
}

// ---------------------------------------------------------------------------
// CSV parser
// ---------------------------------------------------------------------------

/// Parse a CSV EDL string (header + data rows) into a [`ParsedEdl`].
///
/// Expected header: `source,source_in,source_out,record_in,record_out`
pub fn parse_csv(input: &str) -> Result<ParsedEdl> {
    let mut entries: Vec<EdlEntry> = Vec::new();
    let mut lines = input.lines();

    // Skip header
    let header = lines.next().unwrap_or("");
    if !header.contains("source") {
        return Err(Error::EdlParse {
            line: 1,
            message: "Missing CSV header".into(),
        });
    }

    for (line_idx, raw_line) in lines.enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|f| f.trim().trim_matches('"')).collect();
        if fields.len() < 5 {
            return Err(Error::EdlParse {
                line: line_idx + 2,
                message: format!("Expected 5 fields, got {}", fields.len()),
            });
        }

        let parse_f64 = |s: &str, col: &str| -> Result<f64> {
            s.parse::<f64>().map_err(|_| Error::EdlParse {
                line: line_idx + 2,
                message: format!("Invalid {col} value: '{s}'"),
            })
        };

        entries.push(EdlEntry {
            source: fields[0].to_string(),
            source_in: parse_f64(fields[1], "source_in")?,
            source_out: parse_f64(fields[2], "source_out")?,
            record_in: parse_f64(fields[3], "record_in")?,
            record_out: parse_f64(fields[4], "record_out")?,
            transition: None,
        });
    }

    Ok(ParsedEdl {
        title: None,
        entries,
    })
}

// ---------------------------------------------------------------------------
// FCP XML export
// ---------------------------------------------------------------------------

/// Export an [`EditDecisionList`] to Final Cut Pro XMEML-style XML.
pub fn export_fcp_xml(edl: &EditDecisionList, frame_rate: f64) -> String {
    let total_frames = (edl.total_duration() * frame_rate).round() as u64;

    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<!DOCTYPE xmeml>\n");
    xml.push_str("<xmeml version=\"5\">\n");
    xml.push_str("  <sequence>\n");
    xml.push_str(&format!("    <name>{}</name>\n", escape_xml(&edl.title)));
    xml.push_str(&format!("    <duration>{total_frames}</duration>\n"));
    xml.push_str("    <rate>\n");
    xml.push_str(&format!("      <timebase>{frame_rate}</timebase>\n"));
    xml.push_str("    </rate>\n");
    xml.push_str("    <media>\n");
    xml.push_str("      <video>\n");
    xml.push_str("        <track>\n");

    for entry in &edl.entries {
        let start = (entry.record_in * frame_rate).round() as u64;
        let end = (entry.record_out * frame_rate).round() as u64;
        let src_in = (entry.source_in * frame_rate).round() as u64;
        let src_out = (entry.source_out * frame_rate).round() as u64;

        xml.push_str("          <clipitem>\n");
        xml.push_str(&format!(
            "            <name>{}</name>\n",
            escape_xml(&entry.source)
        ));
        xml.push_str(&format!("            <start>{start}</start>\n"));
        xml.push_str(&format!("            <end>{end}</end>\n"));
        xml.push_str(&format!("            <in>{src_in}</in>\n"));
        xml.push_str(&format!("            <out>{src_out}</out>\n"));
        xml.push_str("            <file>\n");
        xml.push_str(&format!(
            "              <pathurl>{}</pathurl>\n",
            escape_xml(&entry.source)
        ));
        xml.push_str("            </file>\n");
        xml.push_str("          </clipitem>\n");
    }

    xml.push_str("        </track>\n");
    xml.push_str("      </video>\n");
    xml.push_str("    </media>\n");
    xml.push_str("  </sequence>\n");
    xml.push_str("</xmeml>\n");

    xml
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ---------------------------------------------------------------------------
// FCP XML import
// ---------------------------------------------------------------------------

/// Parse FCP XML (XMEML-like) into a [`ParsedEdl`].
///
/// This is a lightweight, line-based parser — no external XML crate required.
pub fn parse_fcp_xml(input: &str) -> Result<ParsedEdl> {
    let mut title: Option<String> = None;
    let mut entries: Vec<EdlEntry> = Vec::new();
    let mut frame_rate: f64 = 30.0;

    // Extract frame rate from <rate><timebase>
    if let Some(tb) = extract_tag_value(input, "timebase") {
        if let Ok(fr) = tb.parse::<f64>() {
            frame_rate = fr;
        }
    }

    // Extract sequence name as title
    if let Some(seq_block) = extract_block(input, "sequence") {
        if let Some(name) = extract_direct_child_value(&seq_block, "name") {
            title = Some(name);
        }
    }

    // Extract clipitems
    let mut search_from = 0usize;
    while let Some(block) = extract_block(&input[search_from..], "clipitem") {
        let name = extract_tag_value(&block, "name").unwrap_or_default();
        let start = extract_tag_value(&block, "start")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        let end = extract_tag_value(&block, "end")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        let src_in = extract_tag_value(&block, "in")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        let src_out = extract_tag_value(&block, "out")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        entries.push(EdlEntry {
            source: name,
            source_in: frames_to_seconds(src_in, frame_rate),
            source_out: frames_to_seconds(src_out, frame_rate),
            record_in: frames_to_seconds(start, frame_rate),
            record_out: frames_to_seconds(end, frame_rate),
            transition: None,
        });

        // Advance past this block in the input
        let open_tag = "<clipitem>";
        if let Some(pos) = input[search_from..].find(open_tag) {
            let close_tag = "</clipitem>";
            if let Some(end_pos) = input[search_from + pos..].find(close_tag) {
                search_from = search_from + pos + end_pos + close_tag.len();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    Ok(ParsedEdl { title, entries })
}

/// Extract the text content of the first occurrence of `<tag>…</tag>`.
fn extract_tag_value(input: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = input.find(&open)? + open.len();
    let end = input[start..].find(&close)? + start;
    Some(input[start..end].trim().to_string())
}

/// Extract the full inner content of the first occurrence of `<tag>…</tag>`.
fn extract_block(input: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = input.find(&open)? + open.len();
    let end = input[start..].find(&close)? + start;
    Some(input[start..end].to_string())
}

/// Extract the value of a direct-child `<tag>` inside a block, avoiding
/// nested tags with the same name from deeper elements.
fn extract_direct_child_value(block: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");

    // Find the first occurrence that is NOT inside a nested element
    let start = block.find(&open)? + open.len();
    let end = block[start..].find(&close)? + start;
    let value = block[start..end].trim().to_string();

    // If the value itself contains child tags, it's a nested block — skip it
    if value.contains('<') {
        return None;
    }
    Some(value)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edl::{EditDecisionList, EdlEntry, EdlFormat};
    use crate::ops::TransitionType;

    // ------- timecode helpers -------

    #[test]
    fn test_parse_timecode_basic() {
        let secs = parse_timecode("00:00:01:00", 30.0).unwrap();
        assert!((secs - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_timecode_with_frames() {
        let secs = parse_timecode("00:00:01:15", 30.0).unwrap();
        assert!((secs - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_parse_timecode_hours() {
        let secs = parse_timecode("01:00:00:00", 30.0).unwrap();
        assert!((secs - 3600.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_timecode_invalid() {
        assert!(parse_timecode("invalid", 30.0).is_err());
        assert!(parse_timecode("00:00:00", 30.0).is_err());
    }

    #[test]
    fn test_frames_to_seconds() {
        assert!((frames_to_seconds(30, 30.0) - 1.0).abs() < 0.001);
        assert!((frames_to_seconds(48, 24.0) - 2.0).abs() < 0.001);
        assert!((frames_to_seconds(0, 30.0)).abs() < 0.001);
    }

    // ------- CMX 3600 parser -------

    #[test]
    fn test_parse_cmx3600_basic() {
        let input = "\
TITLE: My Project
001  clip1.mp4  V  C  00:00:00:00 00:00:10:00 00:00:00:00 00:00:10:00
002  clip2.mp4  V  C  00:00:05:00 00:00:20:00 00:00:10:00 00:00:25:00
";
        let parsed = parse_cmx3600(input).unwrap();
        assert_eq!(parsed.title.as_deref(), Some("My Project"));
        assert_eq!(parsed.entries.len(), 2);
        assert_eq!(parsed.entries[0].source, "clip1.mp4");
        assert!((parsed.entries[0].source_out - 10.0).abs() < 0.001);
        assert_eq!(parsed.entries[1].source, "clip2.mp4");
    }

    #[test]
    fn test_parse_cmx3600_dissolve_and_wipe() {
        let input = "\
TITLE: Transitions
001  src1.mp4  V  C  00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00
002  src2.mp4  V  D  00:00:00:00 00:00:10:00 00:00:05:00 00:00:15:00
003  src3.mp4  V  W  00:00:00:00 00:00:08:00 00:00:15:00 00:00:23:00
";
        let parsed = parse_cmx3600(input).unwrap();
        assert_eq!(parsed.entries.len(), 3);
        assert!(parsed.entries[0].transition.is_none());
        assert_eq!(
            parsed.entries[1].transition.as_ref().unwrap().transition_type,
            TransitionType::Dissolve
        );
        assert_eq!(
            parsed.entries[2].transition.as_ref().unwrap().transition_type,
            TransitionType::Wipe
        );
    }

    #[test]
    fn test_parse_cmx3600_skips_comments_and_blanks() {
        let input = "\
TITLE: Test
* This is a comment

001  clip1.mp4  V  C  00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00
* Another comment
";
        let parsed = parse_cmx3600(input).unwrap();
        assert_eq!(parsed.entries.len(), 1);
    }

    #[test]
    fn test_cmx3600_roundtrip() {
        let mut edl = EditDecisionList::with_title("Roundtrip");
        edl.add_entry(EdlEntry {
            source: "clip1.mp4".into(),
            source_in: 0.0,
            source_out: 10.0,
            record_in: 0.0,
            record_out: 10.0,
            transition: None,
        });
        edl.add_entry(EdlEntry {
            source: "clip2.mp4".into(),
            source_in: 5.0,
            source_out: 20.0,
            record_in: 10.0,
            record_out: 25.0,
            transition: None,
        });

        let exported = edl.to_string(EdlFormat::Cmx3600);
        let parsed = parse_cmx3600(&exported).unwrap();

        assert_eq!(parsed.title.as_deref(), Some("Roundtrip"));
        assert_eq!(parsed.entries.len(), 2);
        assert!((parsed.entries[0].source_in - 0.0).abs() < 0.05);
        assert!((parsed.entries[0].source_out - 10.0).abs() < 0.05);
        assert!((parsed.entries[1].source_in - 5.0).abs() < 0.05);
        assert!((parsed.entries[1].record_out - 25.0).abs() < 0.05);
    }

    // ------- CSV parser -------

    #[test]
    fn test_parse_csv_basic() {
        let input = "\
source,source_in,source_out,record_in,record_out
clip1.mp4,0.000,10.000,0.000,10.000
clip2.mp4,5.000,20.000,10.000,25.000
";
        let parsed = parse_csv(input).unwrap();
        assert_eq!(parsed.entries.len(), 2);
        assert_eq!(parsed.entries[0].source, "clip1.mp4");
        assert!((parsed.entries[1].source_out - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_csv_malformed() {
        let input = "no_header_here\nfoo,bar\n";
        assert!(parse_csv(input).is_err());
    }

    // ------- FCP XML export -------

    #[test]
    fn test_export_fcp_xml_structure() {
        let mut edl = EditDecisionList::with_title("FCP Test");
        edl.add_entry(EdlEntry {
            source: "intro.mp4".into(),
            source_in: 0.0,
            source_out: 5.0,
            record_in: 0.0,
            record_out: 5.0,
            transition: None,
        });

        let xml = export_fcp_xml(&edl, 30.0);
        assert!(xml.contains("<xmeml version=\"5\">"));
        assert!(xml.contains("<name>FCP Test</name>"));
        assert!(xml.contains("<timebase>30</timebase>"));
        assert!(xml.contains("<clipitem>"));
        assert!(xml.contains("<name>intro.mp4</name>"));
        assert!(xml.contains("<start>0</start>"));
        assert!(xml.contains("<end>150</end>"));
        assert!(xml.contains("<in>0</in>"));
        assert!(xml.contains("<out>150</out>"));
        assert!(xml.contains("<pathurl>intro.mp4</pathurl>"));
    }

    // ------- FCP XML import -------

    #[test]
    fn test_parse_fcp_xml_basic() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="5">
  <sequence>
    <name>My Sequence</name>
    <duration>450</duration>
    <rate>
      <timebase>30</timebase>
    </rate>
    <media>
      <video>
        <track>
          <clipitem>
            <name>clip1.mp4</name>
            <start>0</start>
            <end>150</end>
            <in>0</in>
            <out>150</out>
            <file>
              <pathurl>clip1.mp4</pathurl>
            </file>
          </clipitem>
          <clipitem>
            <name>clip2.mp4</name>
            <start>150</start>
            <end>450</end>
            <in>90</in>
            <out>390</out>
            <file>
              <pathurl>clip2.mp4</pathurl>
            </file>
          </clipitem>
        </track>
      </video>
    </media>
  </sequence>
</xmeml>"#;

        let parsed = parse_fcp_xml(xml).unwrap();
        assert_eq!(parsed.title.as_deref(), Some("My Sequence"));
        assert_eq!(parsed.entries.len(), 2);
        assert_eq!(parsed.entries[0].source, "clip1.mp4");
        assert!((parsed.entries[0].record_out - 5.0).abs() < 0.001);
        assert_eq!(parsed.entries[1].source, "clip2.mp4");
        assert!((parsed.entries[1].source_in - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_fcp_xml_roundtrip() {
        let mut edl = EditDecisionList::with_title("Roundtrip FCP");
        edl.add_entry(EdlEntry {
            source: "a.mp4".into(),
            source_in: 0.0,
            source_out: 10.0,
            record_in: 0.0,
            record_out: 10.0,
            transition: None,
        });
        edl.add_entry(EdlEntry {
            source: "b.mp4".into(),
            source_in: 2.0,
            source_out: 8.0,
            record_in: 10.0,
            record_out: 16.0,
            transition: None,
        });

        let xml = export_fcp_xml(&edl, 24.0);
        let parsed = parse_fcp_xml(&xml).unwrap();

        assert_eq!(parsed.title.as_deref(), Some("Roundtrip FCP"));
        assert_eq!(parsed.entries.len(), 2);
        assert!((parsed.entries[0].source_out - 10.0).abs() < 0.05);
        assert!((parsed.entries[1].record_in - 10.0).abs() < 0.05);
    }

    #[test]
    fn test_parse_fcp_xml_empty() {
        let xml = r#"<?xml version="1.0"?><xmeml version="5"><sequence><name>Empty</name><rate><timebase>24</timebase></rate><media><video><track></track></video></media></sequence></xmeml>"#;
        let parsed = parse_fcp_xml(xml).unwrap();
        assert_eq!(parsed.title.as_deref(), Some("Empty"));
        assert!(parsed.entries.is_empty());
    }
}
