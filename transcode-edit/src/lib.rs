//! Frame-accurate video editing API.
//!
//! Provides non-linear editing operations: trim, concatenate, overlay,
//! transitions, and multi-track timeline support with frame-accurate
//! seeking and cut points.
//!
//! # Example
//!
//! ```
//! use transcode_edit::{Timeline, Clip, EditOp, Transition, TransitionType};
//!
//! let mut timeline = Timeline::new(30.0); // 30 fps
//!
//! // Add clips with trim points
//! timeline.add_clip(Clip::new("intro.mp4").trim(0.0, 5.0));
//! timeline.add_clip(Clip::new("main.mp4").trim(10.0, 120.0));
//! timeline.add_clip(Clip::new("outro.mp4").trim(0.0, 3.0));
//!
//! // Add a crossfade between first two clips
//! timeline.add_transition(0, Transition::new(TransitionType::CrossFade, 1.0));
//!
//! // Validate and get total duration
//! assert!(timeline.validate().is_ok());
//! ```

#![allow(dead_code)]

mod error;
mod clip;
mod timeline;
mod edl;
mod ops;
mod import;

pub use error::{Error, Result};
pub use clip::{Clip, ClipRef, MediaType};
pub use timeline::{Timeline, Track, TrackType};
pub use edl::{EditDecisionList, EdlEntry, EdlFormat};
pub use ops::{EditOp, Transition, TransitionType, OverlayPosition};
pub use import::{ParsedEdl, parse_cmx3600, parse_csv, parse_fcp_xml, export_fcp_xml, parse_timecode};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_timeline() {
        let mut timeline = Timeline::new(30.0);
        timeline.add_clip(Clip::new("a.mp4").trim(0.0, 10.0));
        timeline.add_clip(Clip::new("b.mp4").trim(5.0, 15.0));
        assert!(timeline.validate().is_ok());
        assert_eq!(timeline.clip_count(), 2);
        assert!((timeline.total_duration() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_number_calculation() {
        let timeline = Timeline::new(24.0);
        assert_eq!(timeline.time_to_frame(1.0), 24);
        assert_eq!(timeline.time_to_frame(0.5), 12);
        assert_eq!(timeline.frame_to_time(48), 2.0);
    }

    #[test]
    fn test_edl_roundtrip() {
        let mut edl = EditDecisionList::new();
        edl.add_entry(EdlEntry {
            source: "clip1.mp4".into(),
            source_in: 0.0,
            source_out: 10.0,
            record_in: 0.0,
            record_out: 10.0,
            transition: None,
        });
        let text = edl.to_string(EdlFormat::Cmx3600);
        assert!(!text.is_empty());
        assert!(text.contains("clip1.mp4"));
    }
}
