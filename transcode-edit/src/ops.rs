//! Edit operations: transitions, overlays, and effects.

use serde::{Deserialize, Serialize};

/// Supported transition types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionType {
    CrossFade,
    Dissolve,
    Wipe,
    FadeToBlack,
    FadeFromBlack,
    Cut,
}

/// A transition between two clips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub transition_type: TransitionType,
    pub duration_secs: f64,
}

impl Transition {
    pub fn new(transition_type: TransitionType, duration_secs: f64) -> Self {
        Self {
            transition_type,
            duration_secs,
        }
    }
}

/// Position for an overlay.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OverlayPosition {
    pub x: i32,
    pub y: i32,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub opacity: f64,
}

impl Default for OverlayPosition {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: None,
            height: None,
            opacity: 1.0,
        }
    }
}

/// Edit operations that can be applied to clips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditOp {
    /// Trim a clip to a specific time range.
    Trim { start: f64, end: f64 },
    /// Concatenate multiple clips sequentially.
    Concat { sources: Vec<String> },
    /// Overlay one clip on another.
    Overlay {
        base: String,
        overlay: String,
        position: OverlayPosition,
        start_time: f64,
        end_time: f64,
    },
    /// Apply a transition between clips.
    Transition {
        clip_index: usize,
        transition: Transition,
    },
    /// Change playback speed.
    Speed { factor: f64 },
    /// Adjust audio volume.
    Volume { level: f64 },
    /// Reverse the clip.
    Reverse,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_creation() {
        let t = Transition::new(TransitionType::CrossFade, 1.5);
        assert_eq!(t.transition_type, TransitionType::CrossFade);
        assert!((t.duration_secs - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_overlay_defaults() {
        let pos = OverlayPosition::default();
        assert_eq!(pos.x, 0);
        assert_eq!(pos.opacity, 1.0);
    }

    #[test]
    fn test_edit_op_serialization() {
        let op = EditOp::Trim { start: 1.0, end: 5.0 };
        let json = serde_json::to_string(&op).unwrap();
        assert!(json.contains("Trim"));
    }
}
