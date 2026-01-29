//! Edit Decision List (EDL) support.

use serde::{Deserialize, Serialize};

use crate::ops::Transition;

/// EDL format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdlFormat {
    /// CMX 3600 standard EDL.
    Cmx3600,
    /// Simple CSV format.
    Csv,
    /// JSON format.
    Json,
}

/// A single entry in an EDL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdlEntry {
    pub source: String,
    pub source_in: f64,
    pub source_out: f64,
    pub record_in: f64,
    pub record_out: f64,
    pub transition: Option<Transition>,
}

impl EdlEntry {
    pub fn duration(&self) -> f64 {
        self.source_out - self.source_in
    }
}

/// An Edit Decision List.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EditDecisionList {
    pub title: String,
    pub entries: Vec<EdlEntry>,
}

impl EditDecisionList {
    pub fn new() -> Self {
        Self {
            title: "Untitled".into(),
            entries: Vec::new(),
        }
    }

    pub fn with_title(title: &str) -> Self {
        Self {
            title: title.into(),
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: EdlEntry) {
        self.entries.push(entry);
    }

    /// Serialize to the specified format.
    pub fn to_string(&self, format: EdlFormat) -> String {
        match format {
            EdlFormat::Cmx3600 => self.to_cmx3600(),
            EdlFormat::Csv => self.to_csv(),
            EdlFormat::Json => serde_json::to_string_pretty(self).unwrap_or_default(),
        }
    }

    /// Total duration of the EDL.
    pub fn total_duration(&self) -> f64 {
        self.entries.iter().map(|e| e.duration()).sum()
    }

    fn to_cmx3600(&self) -> String {
        let mut out = format!("TITLE: {}\n", self.title);
        for (i, entry) in self.entries.iter().enumerate() {
            let edit_num = i + 1;
            let transition_code = entry
                .transition
                .as_ref()
                .map(|t| match t.transition_type {
                    crate::ops::TransitionType::Cut => "C",
                    crate::ops::TransitionType::Dissolve | crate::ops::TransitionType::CrossFade => "D",
                    crate::ops::TransitionType::Wipe => "W",
                    _ => "C",
                })
                .unwrap_or("C");

            out.push_str(&format!(
                "{:03}  {}  V  {}  {}  {}  {}  {}\n",
                edit_num,
                entry.source,
                transition_code,
                format_timecode(entry.source_in),
                format_timecode(entry.source_out),
                format_timecode(entry.record_in),
                format_timecode(entry.record_out),
            ));
        }
        out
    }

    fn to_csv(&self) -> String {
        let mut out = "source,source_in,source_out,record_in,record_out\n".to_string();
        for entry in &self.entries {
            out.push_str(&format!(
                "{},{:.3},{:.3},{:.3},{:.3}\n",
                entry.source, entry.source_in, entry.source_out, entry.record_in, entry.record_out,
            ));
        }
        out
    }
}

fn format_timecode(seconds: f64) -> String {
    let total_frames = (seconds * 30.0).round() as u64; // assuming 30fps for display
    let h = total_frames / (30 * 60 * 60);
    let m = (total_frames / (30 * 60)) % 60;
    let s = (total_frames / 30) % 60;
    let f = total_frames % 30;
    format!("{:02}:{:02}:{:02}:{:02}", h, m, s, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edl_cmx3600() {
        let mut edl = EditDecisionList::with_title("Test Project");
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

        let text = edl.to_string(EdlFormat::Cmx3600);
        assert!(text.contains("TITLE: Test Project"));
        assert!(text.contains("clip1.mp4"));
        assert!(text.contains("clip2.mp4"));
    }

    #[test]
    fn test_edl_csv() {
        let mut edl = EditDecisionList::new();
        edl.add_entry(EdlEntry {
            source: "test.mp4".into(),
            source_in: 1.0,
            source_out: 5.0,
            record_in: 0.0,
            record_out: 4.0,
            transition: None,
        });
        let csv = edl.to_string(EdlFormat::Csv);
        assert!(csv.contains("source,source_in"));
        assert!(csv.contains("test.mp4"));
    }

    #[test]
    fn test_edl_json() {
        let edl = EditDecisionList::with_title("JSON Test");
        let json = edl.to_string(EdlFormat::Json);
        assert!(json.contains("JSON Test"));
    }

    #[test]
    fn test_total_duration() {
        let mut edl = EditDecisionList::new();
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
            source_in: 0.0,
            source_out: 5.0,
            record_in: 10.0,
            record_out: 15.0,
            transition: None,
        });
        assert!((edl.total_duration() - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_timecode_format() {
        assert_eq!(format_timecode(0.0), "00:00:00:00");
        assert_eq!(format_timecode(1.0), "00:00:01:00");
        assert_eq!(format_timecode(61.0), "00:01:01:00");
    }
}
