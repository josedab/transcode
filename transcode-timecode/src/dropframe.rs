//! Drop-frame timecode support for 29.97 and 59.94 fps.
//!
//! Drop-frame timecode compensates for the difference between 30fps and 29.97fps
//! (or 60fps and 59.94fps) by "dropping" frame numbers at specific intervals.
//!
//! The rules are:
//! - Skip frames 0 and 1 (or 0-3 for 59.94) at the start of each minute
//! - Except for minutes 0, 10, 20, 30, 40, 50
//!
//! This results in timecode that accurately represents wall-clock time.

use crate::error::{Result, TimecodeError};
use crate::smpte::{FrameRate, Timecode};
use serde::{Deserialize, Serialize};

/// Drop-frame configuration for a frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DropFrameConfig {
    /// Number of frames dropped per minute (except every 10th minute)
    pub frames_dropped_per_minute: u32,
    /// Nominal frame rate
    pub nominal_fps: u32,
    /// Frames per 10 minutes (accounting for drops)
    pub frames_per_10_minutes: u64,
    /// Frames per minute (accounting for drops, for non-10th minutes)
    pub frames_per_minute: u64,
}

impl DropFrameConfig {
    /// Get the drop-frame configuration for 29.97 fps.
    #[must_use]
    pub const fn for_29_97() -> Self {
        Self {
            frames_dropped_per_minute: 2,
            nominal_fps: 30,
            // 30 * 60 * 10 - 9 * 2 = 18000 - 18 = 17982
            frames_per_10_minutes: 17982,
            // 30 * 60 - 2 = 1798
            frames_per_minute: 1798,
        }
    }

    /// Get the drop-frame configuration for 59.94 fps.
    #[must_use]
    pub const fn for_59_94() -> Self {
        Self {
            frames_dropped_per_minute: 4,
            nominal_fps: 60,
            // 60 * 60 * 10 - 9 * 4 = 36000 - 36 = 35964
            frames_per_10_minutes: 35964,
            // 60 * 60 - 4 = 3596
            frames_per_minute: 3596,
        }
    }

    /// Get the configuration for a frame rate, if it supports drop-frame.
    #[must_use]
    pub fn for_frame_rate(frame_rate: FrameRate) -> Option<Self> {
        match frame_rate {
            FrameRate::Fps29_97 => Some(Self::for_29_97()),
            FrameRate::Fps59_94 => Some(Self::for_59_94()),
            _ => None,
        }
    }
}

/// Convert a frame number to drop-frame timecode.
///
/// # Arguments
/// * `frame_number` - The frame number (0-indexed)
/// * `frame_rate` - The frame rate (must be 29.97 or 59.94)
///
/// # Returns
/// A `Timecode` with drop-frame enabled.
#[must_use]
pub fn frame_to_drop_frame_timecode(frame_number: u64, frame_rate: FrameRate) -> Timecode {
    let config = match DropFrameConfig::for_frame_rate(frame_rate) {
        Some(c) => c,
        None => {
            // Fall back to non-drop-frame for unsupported rates
            return Timecode::from_frame_number(frame_number, frame_rate, false);
        }
    };

    let fps = config.nominal_fps as u64;
    let drop = config.frames_dropped_per_minute as u64;

    // Calculate 10-minute blocks
    let ten_minute_blocks = frame_number / config.frames_per_10_minutes;
    let remaining_after_10min = frame_number % config.frames_per_10_minutes;

    // Calculate minutes within the 10-minute block
    // First minute of each 10-minute block has no drop
    let (extra_minutes, remaining_frames) = if remaining_after_10min < fps * 60 {
        // We're in the first minute of the 10-minute block (no drop)
        (0, remaining_after_10min)
    } else {
        // Account for dropped frames in subsequent minutes
        let frames_after_first_minute = remaining_after_10min - fps * 60;
        let minutes_after_first = frames_after_first_minute / config.frames_per_minute;
        let remaining = frames_after_first_minute % config.frames_per_minute;
        (1 + minutes_after_first, remaining)
    };

    let total_minutes = ten_minute_blocks * 10 + extra_minutes;
    let hours = ((total_minutes / 60) % 24) as u8;
    let minutes = (total_minutes % 60) as u8;

    // Add back the dropped frames for display
    let display_frames = if extra_minutes > 0 && extra_minutes < 10 && minutes % 10 != 0 {
        remaining_frames + drop
    } else {
        remaining_frames
    };

    let seconds = (display_frames / fps) as u8;
    let frames = (display_frames % fps) as u8;

    Timecode {
        hours,
        minutes,
        seconds,
        frames,
        frame_rate,
        drop_frame: true,
    }
}

/// Convert drop-frame timecode to frame number.
///
/// # Arguments
/// * `tc` - The timecode to convert
///
/// # Returns
/// The frame number (0-indexed)
#[must_use]
pub fn drop_frame_timecode_to_frame(tc: &Timecode) -> u64 {
    let config = match DropFrameConfig::for_frame_rate(tc.frame_rate) {
        Some(c) => c,
        None => {
            // Fall back to non-drop-frame calculation
            let fps = tc.frame_rate.nominal_fps() as u64;
            let total_seconds = tc.hours as u64 * 3600 + tc.minutes as u64 * 60 + tc.seconds as u64;
            return total_seconds * fps + tc.frames as u64;
        }
    };

    let fps = config.nominal_fps as u64;
    let drop = config.frames_dropped_per_minute as u64;

    let total_minutes = tc.hours as u64 * 60 + tc.minutes as u64;
    let ten_minute_blocks = total_minutes / 10;
    let remaining_minutes = total_minutes % 10;

    // Calculate frames from complete 10-minute blocks
    let frames_from_10min_blocks = ten_minute_blocks * config.frames_per_10_minutes;

    // Calculate frames from remaining minutes
    let frames_from_remaining_minutes = if remaining_minutes == 0 {
        0
    } else {
        // First minute has full frames, subsequent minutes have drops
        fps * 60 + (remaining_minutes - 1) * config.frames_per_minute
    };

    // Calculate frames from seconds and frame count
    let frames_from_seconds = tc.seconds as u64 * fps + tc.frames as u64;

    // Subtract dropped frames for non-10th minutes
    let dropped_frames = if remaining_minutes > 0 { drop } else { 0 };

    frames_from_10min_blocks + frames_from_remaining_minutes + frames_from_seconds - dropped_frames
}

/// Check if a timecode represents a dropped frame.
///
/// In drop-frame timecode, certain frame numbers are skipped.
#[must_use]
pub fn is_dropped_frame(minutes: u8, seconds: u8, frames: u8, frame_rate: FrameRate) -> bool {
    let config = match DropFrameConfig::for_frame_rate(frame_rate) {
        Some(c) => c,
        None => return false,
    };

    // Frames are dropped at the start of each minute except every 10th minute
    if seconds == 0 && minutes % 10 != 0 {
        frames < config.frames_dropped_per_minute as u8
    } else {
        false
    }
}

/// Validate that a timecode doesn't represent a dropped frame.
pub fn validate_drop_frame_timecode(tc: &Timecode) -> Result<()> {
    if !tc.drop_frame {
        return Ok(());
    }

    if is_dropped_frame(tc.minutes, tc.seconds, tc.frames, tc.frame_rate) {
        return Err(TimecodeError::drop_frame(format!(
            "Frame {:02}:{:02}:{:02};{:02} is a dropped frame",
            tc.hours, tc.minutes, tc.seconds, tc.frames
        )));
    }

    Ok(())
}

/// Calculate the number of frames dropped up to a given timecode.
#[must_use]
pub fn frames_dropped_until(hours: u8, minutes: u8, frame_rate: FrameRate) -> u64 {
    let config = match DropFrameConfig::for_frame_rate(frame_rate) {
        Some(c) => c,
        None => return 0,
    };

    let total_minutes = hours as u64 * 60 + minutes as u64;
    let ten_minute_periods = total_minutes / 10;
    let remaining_minutes = total_minutes % 10;

    // 9 drops per 10-minute period (minutes 1-9, not 0)
    let drops_from_complete_periods =
        ten_minute_periods * 9 * config.frames_dropped_per_minute as u64;

    // Drops from remaining minutes (minutes 1 through remaining_minutes-1 if remaining_minutes > 0)
    let drops_from_remaining = if remaining_minutes > 0 {
        (remaining_minutes) * config.frames_dropped_per_minute as u64
    } else {
        0
    };

    drops_from_complete_periods + drops_from_remaining
}

/// Duration of one frame in seconds for drop-frame rates.
#[must_use]
pub fn frame_duration_seconds(frame_rate: FrameRate) -> f64 {
    match frame_rate {
        FrameRate::Fps29_97 => 1001.0 / 30000.0,
        FrameRate::Fps59_94 => 1001.0 / 60000.0,
        _ => 1.0 / frame_rate.as_f64(),
    }
}

/// Calculate the wall-clock time for a drop-frame timecode.
#[must_use]
pub fn drop_frame_to_wall_time(tc: &Timecode) -> f64 {
    let frame_number = drop_frame_timecode_to_frame(tc);
    frame_number as f64 * frame_duration_seconds(tc.frame_rate)
}

/// Create a drop-frame timecode from wall-clock time.
#[must_use]
pub fn wall_time_to_drop_frame(seconds: f64, frame_rate: FrameRate) -> Timecode {
    let frame_duration = frame_duration_seconds(frame_rate);
    let frame_number = (seconds / frame_duration).round() as u64;
    frame_to_drop_frame_timecode(frame_number, frame_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_drop_frame_config() {
        let config_29_97 = DropFrameConfig::for_29_97();
        assert_eq!(config_29_97.frames_dropped_per_minute, 2);
        assert_eq!(config_29_97.nominal_fps, 30);
        assert_eq!(config_29_97.frames_per_10_minutes, 17982);

        let config_59_94 = DropFrameConfig::for_59_94();
        assert_eq!(config_59_94.frames_dropped_per_minute, 4);
        assert_eq!(config_59_94.nominal_fps, 60);
        assert_eq!(config_59_94.frames_per_10_minutes, 35964);
    }

    #[test]
    fn test_is_dropped_frame() {
        // At minute 1, second 0, frames 0 and 1 are dropped
        assert!(is_dropped_frame(1, 0, 0, FrameRate::Fps29_97));
        assert!(is_dropped_frame(1, 0, 1, FrameRate::Fps29_97));
        assert!(!is_dropped_frame(1, 0, 2, FrameRate::Fps29_97));

        // At minute 10, no frames are dropped
        assert!(!is_dropped_frame(10, 0, 0, FrameRate::Fps29_97));
        assert!(!is_dropped_frame(10, 0, 1, FrameRate::Fps29_97));

        // At minute 0, no frames are dropped
        assert!(!is_dropped_frame(0, 0, 0, FrameRate::Fps29_97));

        // At minute 5, second 1, no frames are dropped
        assert!(!is_dropped_frame(5, 1, 0, FrameRate::Fps29_97));
    }

    #[test]
    fn test_frame_to_drop_frame_timecode_basic() {
        // Frame 0 should be 00:00:00;00
        let tc = frame_to_drop_frame_timecode(0, FrameRate::Fps29_97);
        assert_eq!(tc.hours, 0);
        assert_eq!(tc.minutes, 0);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);

        // Frame 29 should be 00:00:00;29
        let tc = frame_to_drop_frame_timecode(29, FrameRate::Fps29_97);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 29);

        // Frame 30 should be 00:00:01;00
        let tc = frame_to_drop_frame_timecode(30, FrameRate::Fps29_97);
        assert_eq!(tc.seconds, 1);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_drop_frame_at_minute_boundary() {
        // At 1 minute mark (1800 frames for 30fps, but 1798 for drop-frame)
        // Frame 1798 should be 00:01:00;02 (frames 0 and 1 are dropped)
        let tc = frame_to_drop_frame_timecode(1800, FrameRate::Fps29_97);
        assert_eq!(tc.minutes, 1);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 2);
    }

    #[test]
    fn test_drop_frame_roundtrip() {
        // Test that frame -> timecode -> frame roundtrips correctly
        for frame in [0, 29, 30, 1799, 1800, 1801, 17981, 17982, 17983] {
            let tc = frame_to_drop_frame_timecode(frame, FrameRate::Fps29_97);
            let back = drop_frame_timecode_to_frame(&tc);
            assert_eq!(frame, back, "Frame {} roundtrip failed via {:?}", frame, tc);
        }
    }

    #[test]
    fn test_drop_frame_10_minute_boundary() {
        // At 10 minute mark (17982 frames), no drop
        // Frame 17982 should be 00:10:00;00
        let tc = frame_to_drop_frame_timecode(17982, FrameRate::Fps29_97);
        assert_eq!(tc.minutes, 10);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_drop_frame_59_94() {
        // At 1 minute mark for 59.94fps
        // Frame 3596 should be 00:01:00;04 (frames 0-3 are dropped)
        let tc = frame_to_drop_frame_timecode(3600, FrameRate::Fps59_94);
        assert_eq!(tc.minutes, 1);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 4);
    }

    #[test]
    fn test_validate_drop_frame_timecode() {
        // Valid drop-frame timecode
        let tc = Timecode {
            hours: 0,
            minutes: 1,
            seconds: 0,
            frames: 2,
            frame_rate: FrameRate::Fps29_97,
            drop_frame: true,
        };
        assert!(validate_drop_frame_timecode(&tc).is_ok());

        // Invalid drop-frame timecode (dropped frame)
        let tc = Timecode {
            hours: 0,
            minutes: 1,
            seconds: 0,
            frames: 0,
            frame_rate: FrameRate::Fps29_97,
            drop_frame: true,
        };
        assert!(validate_drop_frame_timecode(&tc).is_err());
    }

    #[test]
    fn test_frames_dropped_until() {
        // At minute 0, no drops yet
        assert_eq!(frames_dropped_until(0, 0, FrameRate::Fps29_97), 0);

        // At minute 1, 2 frames dropped
        assert_eq!(frames_dropped_until(0, 1, FrameRate::Fps29_97), 2);

        // At minute 10, 18 frames dropped (9 minutes * 2)
        assert_eq!(frames_dropped_until(0, 10, FrameRate::Fps29_97), 18);

        // At minute 11, 20 frames dropped
        assert_eq!(frames_dropped_until(0, 11, FrameRate::Fps29_97), 20);
    }

    #[test]
    fn test_wall_time_conversion() {
        let tc = Timecode {
            hours: 1,
            minutes: 0,
            seconds: 0,
            frames: 0,
            frame_rate: FrameRate::Fps29_97,
            drop_frame: true,
        };

        let wall_time = drop_frame_to_wall_time(&tc);
        // 1 hour of drop-frame timecode should be very close to 1 hour of wall time
        assert!((wall_time - 3600.0).abs() < 0.1);
    }

    #[test]
    fn test_wall_time_to_drop_frame() {
        // 1 hour of wall time
        let tc = wall_time_to_drop_frame(3600.0, FrameRate::Fps29_97);
        assert_eq!(tc.hours, 1);
        // Should be very close to 00:00:00
        assert!(tc.minutes == 0 || (tc.minutes == 59 && tc.hours == 0));
    }

    #[test]
    fn test_drop_frame_one_hour() {
        // One hour of 29.97fps content
        // 29.97 * 3600 = 107892 frames
        let one_hour_frames = (29.97_f64 * 3600.0).round() as u64;
        let tc = frame_to_drop_frame_timecode(one_hour_frames, FrameRate::Fps29_97);

        // Should display as approximately 01:00:00;00
        assert!(tc.hours == 1 || (tc.hours == 0 && tc.minutes == 59));
    }

    #[test]
    fn test_drop_frame_serialization() {
        let config = DropFrameConfig::for_29_97();
        let json = serde_json::to_string(&config).unwrap();
        let decoded: DropFrameConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, decoded);
    }
}
