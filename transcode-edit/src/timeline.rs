//! Timeline and track management.

use serde::{Deserialize, Serialize};

use crate::clip::Clip;
use crate::error::{Error, Result};
use crate::ops::Transition;

/// Type of content a track holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackType {
    Video,
    Audio,
}

/// A single track on the timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    pub name: String,
    pub track_type: TrackType,
    pub clips: Vec<Clip>,
    pub muted: bool,
    pub locked: bool,
}

impl Track {
    pub fn new(name: &str, track_type: TrackType) -> Self {
        Self {
            name: name.into(),
            track_type,
            clips: Vec::new(),
            muted: false,
            locked: false,
        }
    }

    pub fn duration(&self) -> f64 {
        self.clips.iter().map(|c| c.duration()).sum()
    }
}

/// A multi-track editing timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    pub framerate: f64,
    tracks: Vec<Track>,
    transitions: Vec<(usize, Transition)>,
}

impl Timeline {
    pub fn new(framerate: f64) -> Self {
        let mut tl = Self {
            framerate,
            tracks: Vec::new(),
            transitions: Vec::new(),
        };
        // Default video and audio tracks
        tl.tracks.push(Track::new("V1", TrackType::Video));
        tl.tracks.push(Track::new("A1", TrackType::Audio));
        tl
    }

    /// Add a clip to the default video track.
    pub fn add_clip(&mut self, clip: Clip) {
        if let Some(track) = self.tracks.first_mut() {
            track.clips.push(clip);
        }
    }

    /// Add a clip to a specific track.
    pub fn add_clip_to_track(&mut self, track_index: usize, clip: Clip) -> Result<()> {
        let track = self
            .tracks
            .get_mut(track_index)
            .ok_or(Error::Validation {
                message: format!("Track index {} out of range", track_index),
            })?;
        track.clips.push(clip);
        Ok(())
    }

    /// Add a track.
    pub fn add_track(&mut self, track: Track) {
        self.tracks.push(track);
    }

    /// Add a transition between clip at `index` and the next clip.
    pub fn add_transition(&mut self, clip_index: usize, transition: Transition) {
        self.transitions.push((clip_index, transition));
    }

    /// Validate the timeline for consistency.
    pub fn validate(&self) -> Result<()> {
        if self.framerate <= 0.0 {
            return Err(Error::Validation {
                message: "Frame rate must be positive".into(),
            });
        }

        for track in &self.tracks {
            for clip in &track.clips {
                if let Some(out) = clip.out_point {
                    if clip.in_point >= out {
                        return Err(Error::InvalidTimeRange {
                            start: clip.in_point,
                            end: out,
                        });
                    }
                }
                if clip.speed <= 0.0 {
                    return Err(Error::Validation {
                        message: format!("Clip '{}' has invalid speed: {}", clip.source, clip.speed),
                    });
                }
            }
        }

        // Validate transitions reference valid clip indices
        if let Some(video_track) = self.tracks.first() {
            for (idx, _) in &self.transitions {
                if *idx >= video_track.clips.len().saturating_sub(1) {
                    return Err(Error::InvalidTransition {
                        index: *idx,
                        message: "Transition index exceeds clip count".into(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Total number of clips across all tracks.
    pub fn clip_count(&self) -> usize {
        self.tracks.iter().map(|t| t.clips.len()).sum()
    }

    /// Total duration of the timeline (longest track).
    pub fn total_duration(&self) -> f64 {
        self.tracks
            .iter()
            .map(|t| t.duration())
            .fold(0.0, f64::max)
    }

    /// Convert a time in seconds to a frame number.
    pub fn time_to_frame(&self, time_secs: f64) -> u64 {
        (time_secs * self.framerate).round() as u64
    }

    /// Convert a frame number to time in seconds.
    pub fn frame_to_time(&self, frame: u64) -> f64 {
        frame as f64 / self.framerate
    }

    /// Total frame count.
    pub fn total_frames(&self) -> u64 {
        self.time_to_frame(self.total_duration())
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn transitions(&self) -> &[(usize, Transition)] {
        &self.transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_duration() {
        let mut tl = Timeline::new(30.0);
        tl.add_clip(Clip::new("a.mp4").trim(0.0, 5.0));
        tl.add_clip(Clip::new("b.mp4").trim(0.0, 10.0));
        assert!((tl.total_duration() - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_invalid_trim() {
        let mut tl = Timeline::new(30.0);
        tl.add_clip(Clip::new("bad.mp4").trim(10.0, 5.0)); // invalid: start > end
        assert!(tl.validate().is_err());
    }

    #[test]
    fn test_frame_conversion() {
        let tl = Timeline::new(60.0);
        assert_eq!(tl.time_to_frame(0.5), 30);
        assert!((tl.frame_to_time(120) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_multiple_tracks() {
        let mut tl = Timeline::new(24.0);
        tl.add_track(Track::new("V2", TrackType::Video));
        assert_eq!(tl.tracks().len(), 3); // V1, A1, V2
    }
}
