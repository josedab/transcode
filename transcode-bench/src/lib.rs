//! Benchmark utilities for transcode
//!
//! Provides standardized benchmarking infrastructure for codec and filter performance.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Number of iterations
    pub iterations: u64,
    /// Total time
    pub total_time: Duration,
    /// Mean time per iteration
    pub mean_time: Duration,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Throughput (items/second)
    pub throughput: f64,
}

impl BenchmarkResult {
    /// Calculate from samples
    pub fn from_samples(name: &str, samples: &[Duration], items_per_iter: u64) -> Self {
        let n = samples.len() as u64;
        let total: Duration = samples.iter().sum();
        let mean = total / n as u32;

        let min = *samples.iter().min().unwrap_or(&Duration::ZERO);
        let max = *samples.iter().max().unwrap_or(&Duration::ZERO);

        // Calculate standard deviation
        let mean_nanos = mean.as_nanos() as f64;
        let variance: f64 = samples
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        let throughput = items_per_iter as f64 * n as f64 / total.as_secs_f64();

        Self {
            name: name.to_string(),
            iterations: n,
            total_time: total,
            mean_time: mean,
            min_time: min,
            max_time: max,
            std_dev,
            throughput,
        }
    }

    /// Print formatted result
    pub fn print(&self) {
        println!("Benchmark: {}", self.name);
        println!("  Iterations: {}", self.iterations);
        println!("  Mean time:  {:?}", self.mean_time);
        println!("  Min time:   {:?}", self.min_time);
        println!("  Max time:   {:?}", self.max_time);
        println!("  Std dev:    {:?}", self.std_dev);
        println!("  Throughput: {:.2}/s", self.throughput);
    }
}

/// Benchmark runner
pub struct Benchmark {
    name: String,
    warmup_iterations: u64,
    measure_iterations: u64,
    items_per_iteration: u64,
}

impl Benchmark {
    /// Create a new benchmark
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            warmup_iterations: 10,
            measure_iterations: 100,
            items_per_iteration: 1,
        }
    }

    /// Set warmup iterations
    pub fn warmup(mut self, n: u64) -> Self {
        self.warmup_iterations = n;
        self
    }

    /// Set measurement iterations
    pub fn iterations(mut self, n: u64) -> Self {
        self.measure_iterations = n;
        self
    }

    /// Set items processed per iteration (for throughput calculation)
    pub fn items_per_iter(mut self, n: u64) -> Self {
        self.items_per_iteration = n;
        self
    }

    /// Run the benchmark
    pub fn run<F>(&self, mut f: F) -> BenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup
        for _ in 0..self.warmup_iterations {
            f();
        }

        // Measure
        let mut samples = Vec::with_capacity(self.measure_iterations as usize);
        for _ in 0..self.measure_iterations {
            let start = Instant::now();
            f();
            samples.push(start.elapsed());
        }

        BenchmarkResult::from_samples(&self.name, &samples, self.items_per_iteration)
    }
}

/// Video benchmark utilities
pub mod video {
    use transcode_core::frame::{Frame, PixelFormat};
    use transcode_core::timestamp::TimeBase;

    /// Generate test frames
    pub fn generate_test_frames(count: usize, width: u32, height: u32) -> Vec<Frame> {
        (0..count)
            .map(|i| {
                let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);

                // Fill with pattern
                if let Some(y_plane) = frame.plane_mut(0) {
                    for (j, pixel) in y_plane.iter_mut().enumerate() {
                        *pixel = ((i + j) % 256) as u8;
                    }
                }

                frame
            })
            .collect()
    }

    /// Common resolutions for benchmarking
    pub const RESOLUTIONS: &[(u32, u32, &str)] = &[
        (1920, 1080, "1080p"),
        (1280, 720, "720p"),
        (3840, 2160, "4K"),
        (854, 480, "480p"),
    ];
}

/// Audio benchmark utilities
pub mod audio {
    /// Generate test audio samples
    pub fn generate_test_samples(count: usize, sample_rate: u32) -> Vec<f32> {
        let freq = 440.0; // A4
        (0..count)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
            })
            .collect()
    }

    /// Common sample rates for benchmarking
    pub const SAMPLE_RATES: &[u32] = &[44100, 48000, 96000];
}

/// Benchmark suite for running multiple benchmarks
pub struct BenchmarkSuite {
    name: String,
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create new suite
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            results: Vec::new(),
        }
    }

    /// Add a result
    pub fn add(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Print all results
    pub fn print(&self) {
        println!("=== {} ===", self.name);
        println!();
        for result in &self.results {
            result.print();
            println!();
        }
    }

    /// Export to JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.results).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result() {
        let samples = vec![
            Duration::from_micros(100),
            Duration::from_micros(110),
            Duration::from_micros(90),
            Duration::from_micros(105),
        ];

        let result = BenchmarkResult::from_samples("test", &samples, 1);

        assert_eq!(result.iterations, 4);
        assert_eq!(result.min_time, Duration::from_micros(90));
        assert_eq!(result.max_time, Duration::from_micros(110));
    }

    #[test]
    fn test_benchmark_run() {
        let result = Benchmark::new("simple")
            .warmup(5)
            .iterations(10)
            .run(|| {
                std::thread::sleep(Duration::from_micros(10));
            });

        assert_eq!(result.iterations, 10);
        assert!(result.mean_time >= Duration::from_micros(10));
    }
}
