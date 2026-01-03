//! Subtitle timing utilities

use crate::Segment;

/// Adjust subtitle timing
pub struct TimingAdjuster;

impl TimingAdjuster {
    /// Shift all segments by offset (positive = later, negative = earlier)
    pub fn shift(segments: &mut [Segment], offset_ms: i64) {
        for segment in segments {
            if offset_ms >= 0 {
                segment.start_ms = segment.start_ms.saturating_add(offset_ms as u64);
                segment.end_ms = segment.end_ms.saturating_add(offset_ms as u64);
            } else {
                segment.start_ms = segment.start_ms.saturating_sub((-offset_ms) as u64);
                segment.end_ms = segment.end_ms.saturating_sub((-offset_ms) as u64);
            }
        }
    }

    /// Scale timing by factor (1.0 = no change)
    pub fn scale(segments: &mut [Segment], factor: f64) {
        for segment in segments {
            segment.start_ms = (segment.start_ms as f64 * factor) as u64;
            segment.end_ms = (segment.end_ms as f64 * factor) as u64;
        }
    }

    /// Split long segments
    pub fn split_long(segments: &mut Vec<Segment>, max_duration_ms: u64, max_chars: usize) {
        let mut i = 0;
        while i < segments.len() {
            let segment = &segments[i];

            if segment.duration_ms() > max_duration_ms || segment.text.len() > max_chars {
                // Split the segment
                let words: Vec<&str> = segment.text.split_whitespace().collect();
                if words.len() > 1 {
                    let mid = words.len() / 2;
                    let first_half: String = words[..mid].join(" ");
                    let second_half: String = words[mid..].join(" ");

                    let mid_time = segment.start_ms + segment.duration_ms() / 2;

                    let first_segment = Segment {
                        start_ms: segment.start_ms,
                        end_ms: mid_time,
                        text: first_half,
                        confidence: segment.confidence,
                        words: Vec::new(),
                        language: segment.language.clone(),
                    };

                    let second_segment = Segment {
                        start_ms: mid_time,
                        end_ms: segment.end_ms,
                        text: second_half,
                        confidence: segment.confidence,
                        words: Vec::new(),
                        language: segment.language.clone(),
                    };

                    segments[i] = first_segment;
                    segments.insert(i + 1, second_segment);
                }
            }
            i += 1;
        }
    }

    /// Merge short segments
    pub fn merge_short(segments: &mut Vec<Segment>, min_duration_ms: u64, max_gap_ms: u64) {
        let mut i = 0;
        while i + 1 < segments.len() {
            let current_short = segments[i].duration_ms() < min_duration_ms;
            let next_short = segments[i + 1].duration_ms() < min_duration_ms;
            let gap = segments[i + 1].start_ms.saturating_sub(segments[i].end_ms);

            if (current_short || next_short) && gap <= max_gap_ms {
                // Merge segments
                let next = segments.remove(i + 1);
                segments[i].end_ms = next.end_ms;
                segments[i].text.push(' ');
                segments[i].text.push_str(&next.text);
                segments[i].confidence = (segments[i].confidence + next.confidence) / 2.0;
            } else {
                i += 1;
            }
        }
    }

    /// Snap to frame boundaries
    pub fn snap_to_frames(segments: &mut [Segment], fps: f64) {
        let frame_ms = (1000.0 / fps) as u64;

        for segment in segments {
            segment.start_ms = (segment.start_ms / frame_ms) * frame_ms;
            segment.end_ms = segment.end_ms.div_ceil(frame_ms) * frame_ms;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(start: u64, end: u64, text: &str) -> Segment {
        Segment {
            start_ms: start,
            end_ms: end,
            text: text.into(),
            confidence: 0.9,
            words: Vec::new(),
            language: None,
        }
    }

    #[test]
    fn test_shift() {
        let mut segments = vec![make_segment(1000, 2000, "Test")];
        TimingAdjuster::shift(&mut segments, 500);
        assert_eq!(segments[0].start_ms, 1500);
        assert_eq!(segments[0].end_ms, 2500);
    }

    #[test]
    fn test_scale() {
        let mut segments = vec![make_segment(1000, 2000, "Test")];
        TimingAdjuster::scale(&mut segments, 2.0);
        assert_eq!(segments[0].start_ms, 2000);
        assert_eq!(segments[0].end_ms, 4000);
    }

    #[test]
    fn test_split_long() {
        let mut segments = vec![make_segment(0, 10000, "This is a long sentence that should be split")];
        TimingAdjuster::split_long(&mut segments, 5000, 20);
        assert!(segments.len() > 1);
    }
}
