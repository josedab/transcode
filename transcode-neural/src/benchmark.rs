//! Performance benchmarking for neural inference.
//!
//! This module provides:
//! - Model load time measurement
//! - Inference time per frame
//! - Memory usage tracking
//! - FPS calculation
//! - Detailed performance reports

use crate::{NeuralFrame, Result};
use std::time::{Duration, Instant};

/// Performance metrics for a single operation.
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    /// Operation name.
    pub name: String,
    /// Duration of the operation.
    pub duration: Duration,
    /// Start timestamp.
    pub start_time: Instant,
    /// End timestamp.
    pub end_time: Instant,
}

impl OperationMetrics {
    /// Create new operation metrics.
    pub fn new(name: impl Into<String>, start: Instant, end: Instant) -> Self {
        Self {
            name: name.into(),
            duration: end.duration_since(start),
            start_time: start,
            end_time: end,
        }
    }

    /// Get duration in milliseconds.
    pub fn millis(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }

    /// Get duration in microseconds.
    pub fn micros(&self) -> u128 {
        self.duration.as_micros()
    }
}

/// Memory usage snapshot.
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Estimated GPU memory in bytes.
    pub gpu_memory_bytes: u64,
    /// Estimated CPU memory in bytes.
    pub cpu_memory_bytes: u64,
    /// Peak GPU memory in bytes.
    pub peak_gpu_memory: u64,
    /// Peak CPU memory in bytes.
    pub peak_cpu_memory: u64,
}

impl MemoryUsage {
    /// Create a new memory usage snapshot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get GPU memory in megabytes.
    pub fn gpu_memory_mb(&self) -> f64 {
        self.gpu_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get CPU memory in megabytes.
    pub fn cpu_memory_mb(&self) -> f64 {
        self.cpu_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Estimate memory for a tensor.
    pub fn estimate_tensor_memory(shape: &[usize], element_size: usize) -> u64 {
        let elements: usize = shape.iter().product();
        (elements * element_size) as u64
    }

    /// Estimate memory for a NeuralFrame.
    pub fn estimate_frame_memory(frame: &NeuralFrame) -> u64 {
        // f32 = 4 bytes per element
        (frame.data.len() * 4) as u64
    }
}

/// Inference benchmark results.
#[derive(Debug, Clone)]
pub struct InferenceBenchmark {
    /// Model load time.
    pub model_load_time: Option<Duration>,
    /// Warmup inference times.
    pub warmup_times: Vec<Duration>,
    /// Actual inference times.
    pub inference_times: Vec<Duration>,
    /// Preprocessing times.
    pub preprocess_times: Vec<Duration>,
    /// Postprocessing times.
    pub postprocess_times: Vec<Duration>,
    /// Memory usage.
    pub memory: MemoryUsage,
    /// Input dimensions.
    pub input_dimensions: (u32, u32),
    /// Output dimensions.
    pub output_dimensions: (u32, u32),
    /// Scale factor.
    pub scale: u32,
}

impl InferenceBenchmark {
    /// Create a new benchmark result.
    pub fn new(input_width: u32, input_height: u32, scale: u32) -> Self {
        Self {
            model_load_time: None,
            warmup_times: Vec::new(),
            inference_times: Vec::new(),
            preprocess_times: Vec::new(),
            postprocess_times: Vec::new(),
            memory: MemoryUsage::new(),
            input_dimensions: (input_width, input_height),
            output_dimensions: (input_width * scale, input_height * scale),
            scale,
        }
    }

    /// Calculate average inference time.
    pub fn average_inference_time(&self) -> Duration {
        if self.inference_times.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.inference_times.iter().sum();
        total / self.inference_times.len() as u32
    }

    /// Calculate median inference time.
    pub fn median_inference_time(&self) -> Duration {
        if self.inference_times.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = self.inference_times.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    /// Calculate minimum inference time.
    pub fn min_inference_time(&self) -> Duration {
        self.inference_times.iter().min().copied().unwrap_or(Duration::ZERO)
    }

    /// Calculate maximum inference time.
    pub fn max_inference_time(&self) -> Duration {
        self.inference_times.iter().max().copied().unwrap_or(Duration::ZERO)
    }

    /// Calculate standard deviation of inference times.
    pub fn std_dev_inference_time(&self) -> Duration {
        if self.inference_times.len() < 2 {
            return Duration::ZERO;
        }

        let avg = self.average_inference_time().as_secs_f64();
        let variance: f64 = self
            .inference_times
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - avg;
                diff * diff
            })
            .sum::<f64>()
            / (self.inference_times.len() - 1) as f64;

        Duration::from_secs_f64(variance.sqrt())
    }

    /// Calculate frames per second.
    pub fn fps(&self) -> f64 {
        let avg = self.average_inference_time();
        if avg.is_zero() {
            return 0.0;
        }
        1.0 / avg.as_secs_f64()
    }

    /// Calculate total processing time (preprocess + inference + postprocess).
    pub fn total_time(&self) -> Duration {
        let preprocess: Duration = self.preprocess_times.iter().sum();
        let inference: Duration = self.inference_times.iter().sum();
        let postprocess: Duration = self.postprocess_times.iter().sum();
        preprocess + inference + postprocess
    }

    /// Generate a text report.
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Neural Inference Benchmark ===\n\n");

        report.push_str(&format!(
            "Input:  {}x{}\n",
            self.input_dimensions.0, self.input_dimensions.1
        ));
        report.push_str(&format!(
            "Output: {}x{} ({}x scale)\n",
            self.output_dimensions.0, self.output_dimensions.1, self.scale
        ));
        report.push('\n');

        if let Some(load_time) = self.model_load_time {
            report.push_str(&format!(
                "Model load time: {:.2} ms\n\n",
                load_time.as_secs_f64() * 1000.0
            ));
        }

        if !self.warmup_times.is_empty() {
            let warmup_avg: Duration =
                self.warmup_times.iter().sum::<Duration>() / self.warmup_times.len() as u32;
            report.push_str(&format!(
                "Warmup ({} runs): {:.2} ms avg\n\n",
                self.warmup_times.len(),
                warmup_avg.as_secs_f64() * 1000.0
            ));
        }

        report.push_str(&format!("Inference ({} runs):\n", self.inference_times.len()));
        report.push_str(&format!(
            "  Average: {:.2} ms\n",
            self.average_inference_time().as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "  Median:  {:.2} ms\n",
            self.median_inference_time().as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "  Min:     {:.2} ms\n",
            self.min_inference_time().as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "  Max:     {:.2} ms\n",
            self.max_inference_time().as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "  Std Dev: {:.2} ms\n",
            self.std_dev_inference_time().as_secs_f64() * 1000.0
        ));
        report.push_str(&format!("  FPS:     {:.2}\n", self.fps()));
        report.push('\n');

        if !self.preprocess_times.is_empty() {
            let preprocess_avg: Duration =
                self.preprocess_times.iter().sum::<Duration>() / self.preprocess_times.len() as u32;
            report.push_str(&format!(
                "Preprocessing:  {:.2} ms avg\n",
                preprocess_avg.as_secs_f64() * 1000.0
            ));
        }

        if !self.postprocess_times.is_empty() {
            let postprocess_avg: Duration =
                self.postprocess_times.iter().sum::<Duration>() / self.postprocess_times.len() as u32;
            report.push_str(&format!(
                "Postprocessing: {:.2} ms avg\n",
                postprocess_avg.as_secs_f64() * 1000.0
            ));
        }

        report.push('\n');
        report.push_str(&format!(
            "Memory:\n  GPU: {:.2} MB (peak: {:.2} MB)\n  CPU: {:.2} MB (peak: {:.2} MB)\n",
            self.memory.gpu_memory_mb(),
            self.memory.peak_gpu_memory as f64 / (1024.0 * 1024.0),
            self.memory.cpu_memory_mb(),
            self.memory.peak_cpu_memory as f64 / (1024.0 * 1024.0),
        ));

        report
    }
}

/// Benchmark runner for neural inference.
pub struct Benchmarker {
    /// Number of warmup iterations.
    pub warmup_iterations: usize,
    /// Number of benchmark iterations.
    pub benchmark_iterations: usize,
    /// Whether to track memory.
    pub track_memory: bool,
}

impl Default for Benchmarker {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            benchmark_iterations: 10,
            track_memory: true,
        }
    }
}

impl Benchmarker {
    /// Create a new benchmarker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set warmup iterations.
    pub fn with_warmup(mut self, iterations: usize) -> Self {
        self.warmup_iterations = iterations;
        self
    }

    /// Set benchmark iterations.
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.benchmark_iterations = iterations;
        self
    }

    /// Run benchmark on an upscaler.
    pub fn run<F>(&self, frame: &NeuralFrame, scale: u32, mut inference_fn: F) -> Result<InferenceBenchmark>
    where
        F: FnMut(&NeuralFrame) -> Result<NeuralFrame>,
    {
        let mut benchmark = InferenceBenchmark::new(frame.width, frame.height, scale);

        // Estimate input memory
        if self.track_memory {
            let input_mem = MemoryUsage::estimate_frame_memory(frame);
            let output_mem = input_mem * (scale * scale) as u64;
            benchmark.memory.cpu_memory_bytes = input_mem + output_mem;
            benchmark.memory.peak_cpu_memory = benchmark.memory.cpu_memory_bytes;
        }

        // Warmup
        tracing::debug!("Running {} warmup iterations", self.warmup_iterations);
        for _ in 0..self.warmup_iterations {
            let start = Instant::now();
            let _ = inference_fn(frame)?;
            let elapsed = start.elapsed();
            benchmark.warmup_times.push(elapsed);
        }

        // Benchmark
        tracing::debug!("Running {} benchmark iterations", self.benchmark_iterations);
        for i in 0..self.benchmark_iterations {
            let start = Instant::now();
            let _ = inference_fn(frame)?;
            let elapsed = start.elapsed();
            benchmark.inference_times.push(elapsed);

            tracing::trace!("Iteration {}: {:.2} ms", i + 1, elapsed.as_secs_f64() * 1000.0);
        }

        Ok(benchmark)
    }

    /// Run benchmark with separate timing for each stage.
    pub fn run_detailed<P, I, O>(
        &self,
        frame: &NeuralFrame,
        scale: u32,
        mut preprocess_fn: P,
        mut inference_fn: I,
        mut postprocess_fn: O,
    ) -> Result<InferenceBenchmark>
    where
        P: FnMut(&NeuralFrame) -> Result<Vec<f32>>,
        I: FnMut(&[f32]) -> Result<Vec<f32>>,
        O: FnMut(&[f32]) -> Result<NeuralFrame>,
    {
        let mut benchmark = InferenceBenchmark::new(frame.width, frame.height, scale);

        // Warmup
        for _ in 0..self.warmup_iterations {
            let preprocessed = preprocess_fn(frame)?;
            let inferred = inference_fn(&preprocessed)?;
            let _ = postprocess_fn(&inferred)?;
        }

        // Benchmark
        for _ in 0..self.benchmark_iterations {
            // Preprocess
            let pre_start = Instant::now();
            let preprocessed = preprocess_fn(frame)?;
            benchmark.preprocess_times.push(pre_start.elapsed());

            // Inference
            let inf_start = Instant::now();
            let inferred = inference_fn(&preprocessed)?;
            benchmark.inference_times.push(inf_start.elapsed());

            // Postprocess
            let post_start = Instant::now();
            let _ = postprocess_fn(&inferred)?;
            benchmark.postprocess_times.push(post_start.elapsed());
        }

        Ok(benchmark)
    }
}

/// Timer for measuring operations.
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    /// Start a new timer.
    pub fn start(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
        }
    }

    /// Get elapsed time without stopping.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop and get metrics.
    pub fn stop(self) -> OperationMetrics {
        let end = Instant::now();
        OperationMetrics::new(self.name, self.start, end)
    }

    /// Stop and log the result.
    pub fn stop_and_log(self) -> OperationMetrics {
        let metrics = self.stop();
        tracing::debug!("{}: {:.2} ms", metrics.name, metrics.millis());
        metrics
    }
}

/// Scoped timer that logs on drop.
pub struct ScopedTimer {
    timer: Option<Timer>,
    log_on_drop: bool,
}

impl ScopedTimer {
    /// Create a new scoped timer.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            timer: Some(Timer::start(name)),
            log_on_drop: true,
        }
    }

    /// Disable logging on drop.
    pub fn silent(mut self) -> Self {
        self.log_on_drop = false;
        self
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.timer.as_ref().map(|t| t.elapsed()).unwrap_or(Duration::ZERO)
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        if let Some(timer) = self.timer.take() {
            if self.log_on_drop {
                timer.stop_and_log();
            }
        }
    }
}

/// Macro for timing a block of code.
#[macro_export]
macro_rules! time_it {
    ($name:expr, $block:expr) => {{
        let _timer = $crate::benchmark::ScopedTimer::new($name);
        $block
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_metrics() {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        let end = Instant::now();

        let metrics = OperationMetrics::new("test", start, end);

        assert!(metrics.millis() >= 10.0);
        assert!(metrics.micros() >= 10_000);
    }

    #[test]
    fn test_memory_usage() {
        let mem = MemoryUsage::estimate_tensor_memory(&[1, 3, 256, 256], 4);
        assert_eq!(mem, 1 * 3 * 256 * 256 * 4);
    }

    #[test]
    fn test_inference_benchmark() {
        let mut benchmark = InferenceBenchmark::new(100, 100, 4);
        benchmark.inference_times = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(9),
            Duration::from_millis(10),
        ];

        assert_eq!(benchmark.average_inference_time(), Duration::from_millis(10).mul_f64(1.04));
        assert_eq!(benchmark.min_inference_time(), Duration::from_millis(9));
        assert_eq!(benchmark.max_inference_time(), Duration::from_millis(12));
        assert!(benchmark.fps() > 90.0); // ~100 FPS at 10ms
    }

    #[test]
    fn test_benchmark_report() {
        let mut benchmark = InferenceBenchmark::new(1920, 1080, 4);
        benchmark.model_load_time = Some(Duration::from_millis(500));
        benchmark.warmup_times = vec![Duration::from_millis(50)];
        benchmark.inference_times = vec![
            Duration::from_millis(30),
            Duration::from_millis(32),
            Duration::from_millis(31),
        ];

        let report = benchmark.report();

        assert!(report.contains("1920x1080"));
        assert!(report.contains("7680x4320"));
        assert!(report.contains("4x scale"));
        assert!(report.contains("Model load time"));
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start("test_op");
        std::thread::sleep(Duration::from_millis(5));
        let metrics = timer.stop();

        assert!(metrics.millis() >= 5.0);
        assert_eq!(metrics.name, "test_op");
    }

    #[test]
    fn test_benchmarker_run() {
        let benchmarker = Benchmarker::new()
            .with_warmup(1)
            .with_iterations(3);

        let frame = NeuralFrame::new(16, 16);

        let result = benchmarker.run(&frame, 2, |f| {
            std::thread::sleep(Duration::from_millis(1));
            Ok(NeuralFrame::new(f.width * 2, f.height * 2))
        }).unwrap();

        assert_eq!(result.warmup_times.len(), 1);
        assert_eq!(result.inference_times.len(), 3);
        assert!(result.average_inference_time() >= Duration::from_millis(1));
    }

    #[test]
    fn test_frame_memory_estimate() {
        let frame = NeuralFrame::new(1920, 1080);
        let mem = MemoryUsage::estimate_frame_memory(&frame);

        // 1920 * 1080 * 3 channels * 4 bytes = ~24 MB
        assert_eq!(mem, 1920 * 1080 * 3 * 4);
    }

    #[test]
    fn test_std_dev() {
        let mut benchmark = InferenceBenchmark::new(100, 100, 2);
        benchmark.inference_times = vec![
            Duration::from_millis(10),
            Duration::from_millis(10),
            Duration::from_millis(10),
        ];

        // All same values = 0 std dev
        let std = benchmark.std_dev_inference_time();
        assert!(std.as_millis() < 1);
    }

    #[test]
    fn test_scoped_timer_elapsed() {
        let timer = ScopedTimer::new("test").silent();
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(5));
    }
}
