//! SMPTE 12M timecode implementation.
//!
//! This module provides SMPTE timecode (HH:MM:SS:FF) support with:
//! - Standard frame rates (24, 25, 30 fps and fractional variants)
//! - Timecode arithmetic (add, subtract, compare)
//! - String parsing and formatting
//! - Frame number conversion

use crate::error::{Result, TimecodeError};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};
use std::str::FromStr;

/// Common frame rates used in video production.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FrameRate {
    /// 24 fps (film)
    Fps24,
    /// 23.976 fps (24000/1001, NTSC film)
    Fps23_976,
    /// 25 fps (PAL)
    Fps25,
    /// 29.97 fps (30000/1001, NTSC)
    Fps29_97,
    /// 30 fps
    Fps30,
    /// 48 fps (HFR film)
    Fps48,
    /// 50 fps (PAL)
    Fps50,
    /// 59.94 fps (60000/1001, NTSC)
    Fps59_94,
    /// 60 fps
    Fps60,
    /// Custom frame rate (numerator, denominator)
    Custom {
        /// Frame rate numerator.
        numerator: u32,
        /// Frame rate denominator.
        denominator: u32,
    },
}

impl FrameRate {
    /// Get the frame rate as a rational number (numerator, denominator).
    #[must_use]
    pub fn as_rational(&self) -> (u32, u32) {
        match self {
            Self::Fps24 => (24, 1),
            Self::Fps23_976 => (24000, 1001),
            Self::Fps25 => (25, 1),
            Self::Fps29_97 => (30000, 1001),
            Self::Fps30 => (30, 1),
            Self::Fps48 => (48, 1),
            Self::Fps50 => (50, 1),
            Self::Fps59_94 => (60000, 1001),
            Self::Fps60 => (60, 1),
            Self::Custom {
                numerator,
                denominator,
            } => (*numerator, *denominator),
        }
    }

    /// Get the nominal frame rate (integer frames per second for timecode display).
    #[must_use]
    pub fn nominal_fps(&self) -> u32 {
        match self {
            Self::Fps24 | Self::Fps23_976 => 24,
            Self::Fps25 => 25,
            Self::Fps29_97 | Self::Fps30 => 30,
            Self::Fps48 => 48,
            Self::Fps50 => 50,
            Self::Fps59_94 | Self::Fps60 => 60,
            Self::Custom {
                numerator,
                denominator,
            } => ((*numerator as f64) / (*denominator as f64)).round() as u32,
        }
    }

    /// Get the frame rate as a floating point value.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        let (num, den) = self.as_rational();
        num as f64 / den as f64
    }

    /// Check if this frame rate requires drop-frame timecode.
    #[must_use]
    pub fn is_drop_frame_rate(&self) -> bool {
        matches!(self, Self::Fps29_97 | Self::Fps59_94)
    }

    /// Create a custom frame rate.
    pub fn custom(numerator: u32, denominator: u32) -> Result<Self> {
        if denominator == 0 {
            return Err(TimecodeError::invalid_frame_rate(numerator, denominator));
        }
        Ok(Self::Custom {
            numerator,
            denominator,
        })
    }

    /// Try to match a rational frame rate to a standard one.
    #[must_use]
    pub fn from_rational(numerator: u32, denominator: u32) -> Self {
        match (numerator, denominator) {
            (24, 1) => Self::Fps24,
            (24000, 1001) => Self::Fps23_976,
            (25, 1) => Self::Fps25,
            (30000, 1001) => Self::Fps29_97,
            (30, 1) => Self::Fps30,
            (48, 1) => Self::Fps48,
            (50, 1) => Self::Fps50,
            (60000, 1001) => Self::Fps59_94,
            (60, 1) => Self::Fps60,
            _ => Self::Custom {
                numerator,
                denominator,
            },
        }
    }
}

impl fmt::Display for FrameRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fps24 => write!(f, "24"),
            Self::Fps23_976 => write!(f, "23.976"),
            Self::Fps25 => write!(f, "25"),
            Self::Fps29_97 => write!(f, "29.97"),
            Self::Fps30 => write!(f, "30"),
            Self::Fps48 => write!(f, "48"),
            Self::Fps50 => write!(f, "50"),
            Self::Fps59_94 => write!(f, "59.94"),
            Self::Fps60 => write!(f, "60"),
            Self::Custom {
                numerator,
                denominator,
            } => {
                write!(f, "{}/{}", numerator, denominator)
            }
        }
    }
}

/// SMPTE timecode representation.
///
/// Represents timecode in HH:MM:SS:FF format as defined by SMPTE 12M.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Timecode {
    /// Hours (0-23)
    pub hours: u8,
    /// Minutes (0-59)
    pub minutes: u8,
    /// Seconds (0-59)
    pub seconds: u8,
    /// Frames (0 to fps-1)
    pub frames: u8,
    /// Frame rate
    pub frame_rate: FrameRate,
    /// Whether this is drop-frame timecode
    pub drop_frame: bool,
}

impl Timecode {
    /// Create a new timecode.
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        frame_rate: FrameRate,
    ) -> Result<Self> {
        let tc = Self {
            hours,
            minutes,
            seconds,
            frames,
            frame_rate,
            drop_frame: false,
        };
        tc.validate()?;
        Ok(tc)
    }

    /// Create a new drop-frame timecode.
    pub fn new_drop_frame(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        frame_rate: FrameRate,
    ) -> Result<Self> {
        if !frame_rate.is_drop_frame_rate() {
            return Err(TimecodeError::unsupported_frame_rate(
                "drop-frame",
                frame_rate.to_string(),
            ));
        }
        let tc = Self {
            hours,
            minutes,
            seconds,
            frames,
            frame_rate,
            drop_frame: true,
        };
        tc.validate()?;
        Ok(tc)
    }

    /// Create timecode from a frame number.
    #[must_use]
    pub fn from_frame_number(frame_number: u64, frame_rate: FrameRate, drop_frame: bool) -> Self {
        let fps = frame_rate.nominal_fps();

        if drop_frame && frame_rate.is_drop_frame_rate() {
            // Use drop-frame calculation
            crate::dropframe::frame_to_drop_frame_timecode(frame_number, frame_rate)
        } else {
            // Non-drop-frame calculation
            let total_seconds = frame_number / fps as u64;
            let frames = (frame_number % fps as u64) as u8;

            let hours = ((total_seconds / 3600) % 24) as u8;
            let minutes = ((total_seconds % 3600) / 60) as u8;
            let seconds = (total_seconds % 60) as u8;

            Self {
                hours,
                minutes,
                seconds,
                frames,
                frame_rate,
                drop_frame: false,
            }
        }
    }

    /// Convert timecode to frame number.
    #[must_use]
    pub fn to_frame_number(&self) -> u64 {
        if self.drop_frame {
            crate::dropframe::drop_frame_timecode_to_frame(self)
        } else {
            let fps = self.frame_rate.nominal_fps() as u64;
            let total_seconds =
                self.hours as u64 * 3600 + self.minutes as u64 * 60 + self.seconds as u64;
            total_seconds * fps + self.frames as u64
        }
    }

    /// Create timecode from seconds.
    #[must_use]
    pub fn from_seconds(seconds: f64, frame_rate: FrameRate, drop_frame: bool) -> Self {
        let fps = frame_rate.as_f64();
        let frame_number = (seconds * fps).round() as u64;
        Self::from_frame_number(frame_number, frame_rate, drop_frame)
    }

    /// Convert timecode to seconds.
    #[must_use]
    pub fn to_seconds(&self) -> f64 {
        let frame_number = self.to_frame_number();
        frame_number as f64 / self.frame_rate.as_f64()
    }

    /// Validate the timecode components.
    pub fn validate(&self) -> Result<()> {
        let max_frames = self.frame_rate.nominal_fps() as u8;

        if self.hours > 23 {
            return Err(TimecodeError::invalid_component(
                "hours",
                self.hours as u32,
                23,
            ));
        }
        if self.minutes > 59 {
            return Err(TimecodeError::invalid_component(
                "minutes",
                self.minutes as u32,
                59,
            ));
        }
        if self.seconds > 59 {
            return Err(TimecodeError::invalid_component(
                "seconds",
                self.seconds as u32,
                59,
            ));
        }
        if self.frames >= max_frames {
            return Err(TimecodeError::invalid_component(
                "frames",
                self.frames as u32,
                (max_frames - 1) as u32,
            ));
        }

        // Validate drop-frame constraints
        if self.drop_frame {
            // In drop-frame, frames 0 and 1 are skipped at the start of each minute
            // except for minutes 0, 10, 20, 30, 40, 50
            let skip_frames = match self.frame_rate {
                FrameRate::Fps29_97 => 2,
                FrameRate::Fps59_94 => 4,
                _ => 0,
            };

            if self.seconds == 0 && !self.minutes.is_multiple_of(10) && self.frames < skip_frames {
                return Err(TimecodeError::drop_frame(format!(
                    "Frame {} is dropped at minute {} (not divisible by 10)",
                    self.frames, self.minutes
                )));
            }
        }

        Ok(())
    }

    /// Convert to a different frame rate.
    #[must_use]
    pub fn convert_to(&self, target_frame_rate: FrameRate, target_drop_frame: bool) -> Self {
        let seconds = self.to_seconds();
        Self::from_seconds(seconds, target_frame_rate, target_drop_frame)
    }

    /// Add frames to the timecode.
    pub fn add_frames(&self, frames: i64) -> Result<Self> {
        let current = self.to_frame_number() as i64;
        let new_frame = current + frames;

        if new_frame < 0 {
            return Err(TimecodeError::Underflow);
        }

        Ok(Self::from_frame_number(
            new_frame as u64,
            self.frame_rate,
            self.drop_frame,
        ))
    }

    /// Subtract timecode (returns frame difference).
    #[must_use]
    pub fn difference(&self, other: &Self) -> i64 {
        self.to_frame_number() as i64 - other.to_frame_number() as i64
    }

    /// Check if timecode is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.hours == 0 && self.minutes == 0 && self.seconds == 0 && self.frames == 0
    }

    /// Get the separator character for display.
    #[must_use]
    pub fn separator(&self) -> char {
        if self.drop_frame {
            ';'
        } else {
            ':'
        }
    }
}

impl Default for Timecode {
    fn default() -> Self {
        Self {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frames: 0,
            frame_rate: FrameRate::Fps24,
            drop_frame: false,
        }
    }
}

impl fmt::Display for Timecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours,
            self.minutes,
            self.seconds,
            self.separator(),
            self.frames
        )
    }
}

impl FromStr for Timecode {
    type Err = TimecodeError;

    fn from_str(s: &str) -> Result<Self> {
        // Parse timecode string: HH:MM:SS:FF or HH:MM:SS;FF
        let s = s.trim();

        // Check for drop-frame separator
        let drop_frame = s.contains(';');

        // Split by separator
        let parts: Vec<&str> = s.split([':', ';']).collect();

        if parts.len() != 4 {
            return Err(TimecodeError::invalid_format(
                "Expected format HH:MM:SS:FF or HH:MM:SS;FF",
            ));
        }

        let hours: u8 = parts[0]
            .parse()
            .map_err(|_| TimecodeError::invalid_format(format!("Invalid hours: {}", parts[0])))?;

        let minutes: u8 = parts[1]
            .parse()
            .map_err(|_| TimecodeError::invalid_format(format!("Invalid minutes: {}", parts[1])))?;

        let seconds: u8 = parts[2]
            .parse()
            .map_err(|_| TimecodeError::invalid_format(format!("Invalid seconds: {}", parts[2])))?;

        let frames: u8 = parts[3]
            .parse()
            .map_err(|_| TimecodeError::invalid_format(format!("Invalid frames: {}", parts[3])))?;

        // Default to 30fps for drop-frame, 24fps otherwise
        let frame_rate = if drop_frame {
            FrameRate::Fps29_97
        } else {
            // Infer from frame count if possible
            if frames >= 30 {
                FrameRate::Fps60
            } else if frames >= 25 {
                FrameRate::Fps30
            } else {
                FrameRate::Fps24
            }
        };

        if drop_frame {
            Self::new_drop_frame(hours, minutes, seconds, frames, frame_rate)
        } else {
            Self::new(hours, minutes, seconds, frames, frame_rate)
        }
    }
}

impl PartialEq for Timecode {
    fn eq(&self, other: &Self) -> bool {
        // Compare by frame number if frame rates match
        if self.frame_rate == other.frame_rate && self.drop_frame == other.drop_frame {
            self.hours == other.hours
                && self.minutes == other.minutes
                && self.seconds == other.seconds
                && self.frames == other.frames
        } else {
            // Compare by time value for different frame rates
            (self.to_seconds() - other.to_seconds()).abs() < 0.0001
        }
    }
}

impl Eq for Timecode {}

impl PartialOrd for Timecode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timecode {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.frame_rate == other.frame_rate && self.drop_frame == other.drop_frame {
            // Direct comparison for same frame rate
            self.to_frame_number().cmp(&other.to_frame_number())
        } else {
            // Compare by time value
            self.to_seconds()
                .partial_cmp(&other.to_seconds())
                .unwrap_or(Ordering::Equal)
        }
    }
}

impl Add for Timecode {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Result<Self> {
        let frames = self.to_frame_number() + other.to_frame_number();
        Ok(Self::from_frame_number(
            frames,
            self.frame_rate,
            self.drop_frame,
        ))
    }
}

impl Sub for Timecode {
    type Output = Result<Self>;

    fn sub(self, other: Self) -> Result<Self> {
        let self_frames = self.to_frame_number();
        let other_frames = other.to_frame_number();

        if other_frames > self_frames {
            return Err(TimecodeError::Underflow);
        }

        Ok(Self::from_frame_number(
            self_frames - other_frames,
            self.frame_rate,
            self.drop_frame,
        ))
    }
}

/// Parse a timecode string with explicit frame rate.
pub fn parse_timecode(s: &str, frame_rate: FrameRate) -> Result<Timecode> {
    let s = s.trim();

    // Check for drop-frame separator
    let drop_frame = s.contains(';');

    // Split by separator
    let parts: Vec<&str> = s.split([':', ';']).collect();

    if parts.len() != 4 {
        return Err(TimecodeError::invalid_format(
            "Expected format HH:MM:SS:FF or HH:MM:SS;FF",
        ));
    }

    let hours: u8 = parts[0]
        .parse()
        .map_err(|_| TimecodeError::invalid_format(format!("Invalid hours: {}", parts[0])))?;

    let minutes: u8 = parts[1]
        .parse()
        .map_err(|_| TimecodeError::invalid_format(format!("Invalid minutes: {}", parts[1])))?;

    let seconds: u8 = parts[2]
        .parse()
        .map_err(|_| TimecodeError::invalid_format(format!("Invalid seconds: {}", parts[2])))?;

    let frames: u8 = parts[3]
        .parse()
        .map_err(|_| TimecodeError::invalid_format(format!("Invalid frames: {}", parts[3])))?;

    if drop_frame {
        if !frame_rate.is_drop_frame_rate() {
            return Err(TimecodeError::unsupported_frame_rate(
                "drop-frame",
                frame_rate.to_string(),
            ));
        }
        Timecode::new_drop_frame(hours, minutes, seconds, frames, frame_rate)
    } else {
        Timecode::new(hours, minutes, seconds, frames, frame_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_timecode_new() {
        let tc = Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap();
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 30);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_timecode_validation() {
        // Valid timecode
        assert!(Timecode::new(23, 59, 59, 23, FrameRate::Fps24).is_ok());

        // Invalid hours
        assert!(Timecode::new(24, 0, 0, 0, FrameRate::Fps24).is_err());

        // Invalid minutes
        assert!(Timecode::new(0, 60, 0, 0, FrameRate::Fps24).is_err());

        // Invalid seconds
        assert!(Timecode::new(0, 0, 60, 0, FrameRate::Fps24).is_err());

        // Invalid frames (24fps)
        assert!(Timecode::new(0, 0, 0, 24, FrameRate::Fps24).is_err());

        // Invalid frames (30fps)
        assert!(Timecode::new(0, 0, 0, 30, FrameRate::Fps30).is_err());
    }

    #[test]
    fn test_timecode_display() {
        let tc = Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap();
        assert_eq!(tc.to_string(), "01:30:45:12");

        let tc_df = Timecode::new_drop_frame(1, 30, 45, 12, FrameRate::Fps29_97).unwrap();
        assert_eq!(tc_df.to_string(), "01:30:45;12");
    }

    #[test]
    fn test_timecode_parse() {
        let tc: Timecode = "01:30:45:12".parse().unwrap();
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 30);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
        assert!(!tc.drop_frame);

        let tc_df: Timecode = "01:30:45;12".parse().unwrap();
        assert!(tc_df.drop_frame);
    }

    #[test]
    fn test_timecode_to_frame_number() {
        // 24fps: 1 hour = 86400 frames
        let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps24).unwrap();
        assert_eq!(tc.to_frame_number(), 86400);

        // 30fps: 1 hour = 108000 frames
        let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps30).unwrap();
        assert_eq!(tc.to_frame_number(), 108000);

        // 24fps: 01:30:45:12 = 130332 frames
        let tc = Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap();
        let expected = 1 * 3600 * 24 + 30 * 60 * 24 + 45 * 24 + 12;
        assert_eq!(tc.to_frame_number(), expected);
    }

    #[test]
    fn test_timecode_from_frame_number() {
        let tc = Timecode::from_frame_number(86400, FrameRate::Fps24, false);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 0);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);

        let tc = Timecode::from_frame_number(130332, FrameRate::Fps24, false);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 30);
        assert_eq!(tc.seconds, 30);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_timecode_arithmetic() {
        let tc1 = Timecode::new(0, 0, 1, 0, FrameRate::Fps24).unwrap();
        let tc2 = Timecode::new(0, 0, 0, 12, FrameRate::Fps24).unwrap();

        let sum = (tc1 + tc2).unwrap();
        assert_eq!(sum.seconds, 1);
        assert_eq!(sum.frames, 12);

        let diff = (tc1 - tc2).unwrap();
        assert_eq!(diff.seconds, 0);
        assert_eq!(diff.frames, 12);
    }

    #[test]
    fn test_timecode_add_frames() {
        let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps24).unwrap();

        let tc2 = tc.add_frames(24).unwrap();
        assert_eq!(tc2.seconds, 1);
        assert_eq!(tc2.frames, 0);

        let tc3 = tc.add_frames(25).unwrap();
        assert_eq!(tc3.seconds, 1);
        assert_eq!(tc3.frames, 1);
    }

    #[test]
    fn test_timecode_comparison() {
        let tc1 = Timecode::new(1, 0, 0, 0, FrameRate::Fps24).unwrap();
        let tc2 = Timecode::new(0, 59, 59, 23, FrameRate::Fps24).unwrap();

        assert!(tc1 > tc2);
    }

    #[test]
    fn test_timecode_seconds_conversion() {
        let tc = Timecode::from_seconds(90.5, FrameRate::Fps24, false);
        assert_eq!(tc.minutes, 1);
        assert_eq!(tc.seconds, 30);
        assert_eq!(tc.frames, 12);

        let seconds = tc.to_seconds();
        assert!((seconds - 90.5).abs() < 0.1);
    }

    #[test]
    fn test_frame_rate_display() {
        assert_eq!(FrameRate::Fps24.to_string(), "24");
        assert_eq!(FrameRate::Fps29_97.to_string(), "29.97");
        assert_eq!(
            FrameRate::custom(48000, 1001).unwrap().to_string(),
            "48000/1001"
        );
    }

    #[test]
    fn test_frame_rate_as_f64() {
        assert!((FrameRate::Fps24.as_f64() - 24.0).abs() < 0.001);
        assert!((FrameRate::Fps29_97.as_f64() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_timecode_serialization() {
        let tc = Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap();
        let json = serde_json::to_string(&tc).unwrap();
        let decoded: Timecode = serde_json::from_str(&json).unwrap();
        assert_eq!(tc, decoded);
    }

    #[test]
    fn test_parse_timecode_with_frame_rate() {
        let tc = parse_timecode("01:30:45:12", FrameRate::Fps25).unwrap();
        assert_eq!(tc.frame_rate, FrameRate::Fps25);

        let tc_df = parse_timecode("01:30:45;12", FrameRate::Fps29_97).unwrap();
        assert!(tc_df.drop_frame);

        // Drop-frame separator with non-drop-frame rate should fail
        assert!(parse_timecode("01:30:45;12", FrameRate::Fps24).is_err());
    }
}
