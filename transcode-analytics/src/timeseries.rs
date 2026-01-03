//! Time series data structures

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Time series data point
#[derive(Debug, Clone)]
pub struct DataPoint<T> {
    /// Timestamp
    pub timestamp: Instant,
    /// Value
    pub value: T,
}

/// Rolling time series buffer
#[derive(Debug, Clone)]
pub struct TimeSeries<T> {
    data: VecDeque<DataPoint<T>>,
    max_duration: Duration,
    max_points: usize,
}

impl<T: Clone> TimeSeries<T> {
    /// Create a new time series
    pub fn new(max_duration: Duration, max_points: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(max_points),
            max_duration,
            max_points,
        }
    }

    /// Add a data point
    pub fn push(&mut self, value: T) {
        let now = Instant::now();

        // Remove old points
        let cutoff = now - self.max_duration;
        while let Some(front) = self.data.front() {
            if front.timestamp < cutoff {
                self.data.pop_front();
            } else {
                break;
            }
        }

        // Remove excess points
        while self.data.len() >= self.max_points {
            self.data.pop_front();
        }

        self.data.push_back(DataPoint {
            timestamp: now,
            value,
        });
    }

    /// Get all data points
    pub fn data(&self) -> impl Iterator<Item = &DataPoint<T>> {
        self.data.iter()
    }

    /// Get latest value
    pub fn latest(&self) -> Option<&T> {
        self.data.back().map(|p| &p.value)
    }

    /// Get number of points
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl TimeSeries<f64> {
    /// Calculate average
    pub fn average(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }

        let sum: f64 = self.data.iter().map(|p| p.value).sum();
        Some(sum / self.data.len() as f64)
    }

    /// Calculate minimum
    pub fn min(&self) -> Option<f64> {
        self.data.iter().map(|p| p.value).fold(None, |acc, v| {
            Some(acc.map_or(v, |a: f64| a.min(v)))
        })
    }

    /// Calculate maximum
    pub fn max(&self) -> Option<f64> {
        self.data.iter().map(|p| p.value).fold(None, |acc, v| {
            Some(acc.map_or(v, |a: f64| a.max(v)))
        })
    }

    /// Calculate standard deviation
    pub fn std_dev(&self) -> Option<f64> {
        let avg = self.average()?;
        if self.data.len() < 2 {
            return Some(0.0);
        }

        let variance: f64 = self.data.iter()
            .map(|p| (p.value - avg).powi(2))
            .sum::<f64>() / (self.data.len() - 1) as f64;

        Some(variance.sqrt())
    }

    /// Get rate of change (per second)
    pub fn rate_of_change(&self) -> Option<f64> {
        if self.data.len() < 2 {
            return None;
        }

        let first = self.data.front()?;
        let last = self.data.back()?;

        let duration = last.timestamp.duration_since(first.timestamp);
        if duration.as_secs_f64() > 0.0 {
            Some((last.value - first.value) / duration.as_secs_f64())
        } else {
            None
        }
    }
}

/// Moving average calculator
pub struct MovingAverage {
    window: VecDeque<f64>,
    sum: f64,
    size: usize,
}

impl MovingAverage {
    /// Create a new moving average with window size
    pub fn new(size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(size),
            sum: 0.0,
            size,
        }
    }

    /// Add a value and get current average
    pub fn add(&mut self, value: f64) -> f64 {
        if self.window.len() >= self.size {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
            }
        }

        self.window.push_back(value);
        self.sum += value;

        self.sum / self.window.len() as f64
    }

    /// Get current average without adding
    pub fn current(&self) -> Option<f64> {
        if self.window.is_empty() {
            None
        } else {
            Some(self.sum / self.window.len() as f64)
        }
    }

    /// Reset the moving average
    pub fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series() {
        let mut ts: TimeSeries<f64> = TimeSeries::new(Duration::from_secs(60), 100);

        ts.push(1.0);
        ts.push(2.0);
        ts.push(3.0);

        assert_eq!(ts.len(), 3);
        assert_eq!(ts.latest(), Some(&3.0));
        assert_eq!(ts.average(), Some(2.0));
    }

    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverage::new(3);

        assert_eq!(ma.add(1.0), 1.0);
        assert_eq!(ma.add(2.0), 1.5);
        assert_eq!(ma.add(3.0), 2.0);
        assert_eq!(ma.add(4.0), 3.0); // (2+3+4)/3
    }
}
