//! Clip and media source definitions.

use serde::{Deserialize, Serialize};

/// Type of media in a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaType {
    Video,
    Audio,
    Both,
}

/// A reference to a source media file with optional trim points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clip {
    pub source: String,
    pub media_type: MediaType,
    /// In-point in seconds (from source start).
    pub in_point: f64,
    /// Out-point in seconds (from source start). None = end of file.
    pub out_point: Option<f64>,
    /// Playback speed multiplier (1.0 = normal).
    pub speed: f64,
    /// Volume multiplier for audio (1.0 = normal).
    pub volume: f64,
}

impl Clip {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.into(),
            media_type: MediaType::Both,
            in_point: 0.0,
            out_point: None,
            speed: 1.0,
            volume: 1.0,
        }
    }

    /// Set trim points (in seconds).
    pub fn trim(mut self, start: f64, end: f64) -> Self {
        self.in_point = start;
        self.out_point = Some(end);
        self
    }

    /// Set media type (video only, audio only, or both).
    pub fn media(mut self, media_type: MediaType) -> Self {
        self.media_type = media_type;
        self
    }

    /// Set playback speed.
    pub fn speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    /// Set audio volume.
    pub fn volume(mut self, volume: f64) -> Self {
        self.volume = volume;
        self
    }

    /// Effective duration after trim and speed adjustment.
    pub fn duration(&self) -> f64 {
        let raw_duration = match self.out_point {
            Some(out) => out - self.in_point,
            None => 0.0, // Unknown until file is probed
        };
        raw_duration / self.speed
    }
}

/// A reference to a clip within a timeline (with position info).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipRef {
    pub clip: Clip,
    /// Position on the timeline in seconds.
    pub timeline_position: f64,
    /// Index on the track.
    pub track_index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_trim() {
        let clip = Clip::new("test.mp4").trim(5.0, 15.0);
        assert_eq!(clip.in_point, 5.0);
        assert_eq!(clip.out_point, Some(15.0));
        assert!((clip.duration() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_clip_speed() {
        let clip = Clip::new("test.mp4").trim(0.0, 10.0).speed(2.0);
        assert!((clip.duration() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_clip_defaults() {
        let clip = Clip::new("file.mp4");
        assert_eq!(clip.speed, 1.0);
        assert_eq!(clip.volume, 1.0);
        assert_eq!(clip.media_type, MediaType::Both);
    }
}
