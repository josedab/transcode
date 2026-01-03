//! Timestamp and time base handling.
//!
//! Provides precise time representation for media synchronization.

use crate::rational::Rational;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Sub};

/// A time base for converting between timestamp units.
///
/// Common time bases:
/// - 1/90000 for MPEG-TS
/// - 1/48000 for 48kHz audio
/// - 1/1000 for milliseconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeBase(pub Rational);

impl TimeBase {
    /// Create a new time base from numerator and denominator.
    pub fn new(num: i64, den: i64) -> Self {
        Self(Rational::new(num, den))
    }

    /// Standard MPEG time base (1/90000).
    pub const MPEG: Self = Self(Rational { num: 1, den: 90000 });

    /// Millisecond time base (1/1000).
    pub const MILLISECONDS: Self = Self(Rational { num: 1, den: 1000 });

    /// Microsecond time base (1/1000000).
    pub const MICROSECONDS: Self = Self(Rational { num: 1, den: 1000000 });

    /// Nanosecond time base (1/1000000000).
    pub const NANOSECONDS: Self = Self(Rational { num: 1, den: 1000000000 });

    /// Second time base (1/1).
    pub const SECONDS: Self = Self(Rational { num: 1, den: 1 });

    /// Convert a timestamp from this time base to another.
    pub fn convert(&self, value: i64, target: TimeBase) -> i64 {
        self.0.rescale(value, target.0)
    }

    /// Convert to seconds as f64.
    pub fn to_seconds(&self, value: i64) -> f64 {
        value as f64 * self.0.to_f64()
    }

    /// Convert from seconds.
    pub fn from_seconds(&self, seconds: f64) -> i64 {
        (seconds / self.0.to_f64()) as i64
    }

    /// Get the time base as a rational.
    pub fn as_rational(&self) -> Rational {
        self.0
    }
}

impl Default for TimeBase {
    fn default() -> Self {
        Self::MPEG
    }
}

impl From<(i32, i32)> for TimeBase {
    fn from((num, den): (i32, i32)) -> Self {
        Self::new(num as i64, den as i64)
    }
}

impl From<Rational> for TimeBase {
    fn from(r: Rational) -> Self {
        Self(r)
    }
}

/// A timestamp with an associated time base.
#[derive(Debug, Clone, Copy)]
pub struct Timestamp {
    /// The raw timestamp value.
    pub value: i64,
    /// The time base for interpreting the value.
    pub time_base: TimeBase,
}

impl Timestamp {
    /// Value representing an undefined timestamp.
    pub const NONE: i64 = i64::MIN;

    /// Create a new timestamp.
    pub fn new(value: i64, time_base: TimeBase) -> Self {
        Self { value, time_base }
    }

    /// Create an undefined timestamp.
    pub fn none() -> Self {
        Self {
            value: Self::NONE,
            time_base: TimeBase::default(),
        }
    }

    /// Check if this timestamp is defined.
    pub fn is_valid(&self) -> bool {
        self.value != Self::NONE
    }

    /// Convert to a different time base.
    pub fn rescale(&self, target: TimeBase) -> Self {
        if !self.is_valid() {
            return Self::none();
        }
        Self {
            value: self.time_base.convert(self.value, target),
            time_base: target,
        }
    }

    /// Convert to seconds.
    pub fn to_seconds(&self) -> Option<f64> {
        if self.is_valid() {
            Some(self.time_base.to_seconds(self.value))
        } else {
            None
        }
    }

    /// Create from seconds.
    pub fn from_seconds(seconds: f64, time_base: TimeBase) -> Self {
        Self {
            value: time_base.from_seconds(seconds),
            time_base,
        }
    }

    /// Create from milliseconds.
    pub fn from_millis(millis: i64) -> Self {
        Self {
            value: millis,
            time_base: TimeBase::MILLISECONDS,
        }
    }

    /// Convert to milliseconds.
    pub fn to_millis(&self) -> Option<i64> {
        if self.is_valid() {
            Some(self.time_base.convert(self.value, TimeBase::MILLISECONDS))
        } else {
            None
        }
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::none()
    }
}

impl PartialEq for Timestamp {
    fn eq(&self, other: &Self) -> bool {
        if !self.is_valid() || !other.is_valid() {
            return !self.is_valid() && !other.is_valid();
        }
        // Compare in higher precision time base
        let tb = if self.time_base.0.den > other.time_base.0.den {
            self.time_base
        } else {
            other.time_base
        };
        self.rescale(tb).value == other.rescale(tb).value
    }
}

impl Eq for Timestamp {}

impl PartialOrd for Timestamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timestamp {
    fn cmp(&self, other: &Self) -> Ordering {
        if !self.is_valid() {
            return if !other.is_valid() {
                Ordering::Equal
            } else {
                Ordering::Less
            };
        }
        if !other.is_valid() {
            return Ordering::Greater;
        }

        // Compare in higher precision time base
        let tb = if self.time_base.0.den > other.time_base.0.den {
            self.time_base
        } else {
            other.time_base
        };
        self.rescale(tb).value.cmp(&other.rescale(tb).value)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(secs) = self.to_seconds() {
            let hours = (secs / 3600.0) as u32;
            let mins = ((secs % 3600.0) / 60.0) as u32;
            let secs = secs % 60.0;
            write!(f, "{:02}:{:02}:{:06.3}", hours, mins, secs)
        } else {
            write!(f, "NONE")
        }
    }
}

/// A duration with an associated time base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration {
    /// The raw duration value.
    pub value: i64,
    /// The time base for interpreting the value.
    pub time_base: TimeBase,
}

impl Duration {
    /// Create a new duration.
    pub fn new(value: i64, time_base: TimeBase) -> Self {
        Self { value, time_base }
    }

    /// Create a zero duration.
    pub fn zero() -> Self {
        Self {
            value: 0,
            time_base: TimeBase::default(),
        }
    }

    /// Check if this duration is zero.
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Convert to a different time base.
    pub fn rescale(&self, target: TimeBase) -> Self {
        Self {
            value: self.time_base.convert(self.value, target),
            time_base: target,
        }
    }

    /// Convert to seconds.
    pub fn to_seconds(&self) -> f64 {
        self.time_base.to_seconds(self.value)
    }

    /// Create from seconds.
    pub fn from_seconds(seconds: f64, time_base: TimeBase) -> Self {
        Self {
            value: time_base.from_seconds(seconds),
            time_base,
        }
    }

    /// Create from milliseconds.
    pub fn from_millis(millis: i64) -> Self {
        Self {
            value: millis,
            time_base: TimeBase::MILLISECONDS,
        }
    }
}

impl Default for Duration {
    fn default() -> Self {
        Self::zero()
    }
}

impl Add for Duration {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let rhs = rhs.rescale(self.time_base);
        Self {
            value: self.value + rhs.value,
            time_base: self.time_base,
        }
    }
}

impl Sub for Duration {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let rhs = rhs.rescale(self.time_base);
        Self {
            value: self.value - rhs.value,
            time_base: self.time_base,
        }
    }
}

impl Add<Duration> for Timestamp {
    type Output = Timestamp;

    fn add(self, rhs: Duration) -> Self::Output {
        if !self.is_valid() {
            return self;
        }
        let rhs = rhs.rescale(self.time_base);
        Timestamp {
            value: self.value + rhs.value,
            time_base: self.time_base,
        }
    }
}

impl Sub<Duration> for Timestamp {
    type Output = Timestamp;

    fn sub(self, rhs: Duration) -> Self::Output {
        if !self.is_valid() {
            return self;
        }
        let rhs = rhs.rescale(self.time_base);
        Timestamp {
            value: self.value - rhs.value,
            time_base: self.time_base,
        }
    }
}

impl Sub for Timestamp {
    type Output = Duration;

    fn sub(self, rhs: Self) -> Self::Output {
        if !self.is_valid() || !rhs.is_valid() {
            return Duration::zero();
        }
        let rhs = rhs.rescale(self.time_base);
        Duration {
            value: self.value - rhs.value,
            time_base: self.time_base,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_base_convert() {
        let tb1 = TimeBase::new(1, 1000); // milliseconds
        let tb2 = TimeBase::new(1, 90000); // MPEG

        // 1000ms = 90000 in MPEG time base
        assert_eq!(tb1.convert(1000, tb2), 90000);
    }

    #[test]
    fn test_timestamp_to_seconds() {
        let ts = Timestamp::new(90000, TimeBase::MPEG);
        let secs = ts.to_seconds().unwrap();
        assert!((secs - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_timestamp_comparison() {
        let ts1 = Timestamp::new(90000, TimeBase::MPEG);
        let ts2 = Timestamp::new(1000, TimeBase::MILLISECONDS);
        assert_eq!(ts1, ts2);
    }

    #[test]
    fn test_duration_add() {
        let d1 = Duration::from_millis(500);
        let d2 = Duration::from_millis(500);
        let d3 = d1 + d2;
        assert_eq!(d3.value, 1000);
    }

    #[test]
    fn test_timestamp_display() {
        let ts = Timestamp::new(3723500, TimeBase::MILLISECONDS);
        assert_eq!(format!("{}", ts), "01:02:03.500");
    }
}
