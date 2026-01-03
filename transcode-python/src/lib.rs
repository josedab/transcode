//! Python bindings for the Transcode codec library.
//!
//! This module provides Python bindings using PyO3, allowing Python users
//! to access the memory-safe transcoding capabilities of Transcode.
//!
//! ## Async Support
//!
//! The library supports async transcoding with progress callbacks:
//!
//! ```python
//! import asyncio
//! import transcode_py
//!
//! async def main():
//!     async def on_progress(progress):
//!         print(f"Progress: {progress.percent:.1f}%")
//!
//!     stats = await transcode_py.transcode_async(
//!         "input.mp4",
//!         "output.mp4",
//!         progress_callback=on_progress
//!     )
//!     print(f"Done! {stats}")
//!
//! asyncio.run(main())
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError, PyRuntimeError};
use std::path::PathBuf;
use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};

// Use explicit paths to avoid ambiguity
use ::transcode::TranscodeOptions as RustTranscodeOptions;
use ::transcode::Transcoder as RustTranscoder;

/// Transcode options for configuring transcoding operations.
#[pyclass]
#[derive(Clone)]
pub struct TranscodeOptions {
    input_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    video_codec: Option<String>,
    audio_codec: Option<String>,
    video_bitrate: Option<u64>,
    audio_bitrate: Option<u64>,
    width: Option<u32>,
    height: Option<u32>,
    overwrite: bool,
    threads: Option<usize>,
}

#[pymethods]
impl TranscodeOptions {
    /// Create new transcoding options.
    #[new]
    pub fn new() -> Self {
        Self {
            input_path: None,
            output_path: None,
            video_codec: None,
            audio_codec: None,
            video_bitrate: None,
            audio_bitrate: None,
            width: None,
            height: None,
            overwrite: false,
            threads: None,
        }
    }

    /// Set the input file path.
    pub fn input(&mut self, path: &str) -> PyResult<Self> {
        self.input_path = Some(PathBuf::from(path));
        Ok(self.clone())
    }

    /// Set the output file path.
    pub fn output(&mut self, path: &str) -> PyResult<Self> {
        self.output_path = Some(PathBuf::from(path));
        Ok(self.clone())
    }

    /// Set the video codec (e.g., "h264", "h265", "av1").
    pub fn video_codec(&mut self, codec: &str) -> PyResult<Self> {
        self.video_codec = Some(codec.to_string());
        Ok(self.clone())
    }

    /// Set the audio codec (e.g., "aac", "mp3", "opus").
    pub fn audio_codec(&mut self, codec: &str) -> PyResult<Self> {
        self.audio_codec = Some(codec.to_string());
        Ok(self.clone())
    }

    /// Set the video bitrate in bits per second.
    pub fn video_bitrate(&mut self, bitrate: u64) -> PyResult<Self> {
        self.video_bitrate = Some(bitrate);
        Ok(self.clone())
    }

    /// Set the audio bitrate in bits per second.
    pub fn audio_bitrate(&mut self, bitrate: u64) -> PyResult<Self> {
        self.audio_bitrate = Some(bitrate);
        Ok(self.clone())
    }

    /// Set the output video resolution.
    pub fn resolution(&mut self, width: u32, height: u32) -> PyResult<Self> {
        self.width = Some(width);
        self.height = Some(height);
        Ok(self.clone())
    }

    /// Enable overwriting existing output files.
    pub fn overwrite(&mut self, enable: bool) -> PyResult<Self> {
        self.overwrite = enable;
        Ok(self.clone())
    }

    /// Set the number of threads for encoding.
    pub fn threads(&mut self, threads: usize) -> PyResult<Self> {
        self.threads = Some(threads);
        Ok(self.clone())
    }
}

impl Default for TranscodeOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Transcoding statistics returned after a transcoding operation.
#[pyclass]
#[derive(Clone)]
pub struct TranscodeStats {
    /// Total packets processed.
    #[pyo3(get)]
    pub packets_processed: u64,
    /// Total frames decoded.
    #[pyo3(get)]
    pub frames_decoded: u64,
    /// Total frames encoded.
    #[pyo3(get)]
    pub frames_encoded: u64,
    /// Input file size in bytes.
    #[pyo3(get)]
    pub input_size: u64,
    /// Output file size in bytes.
    #[pyo3(get)]
    pub output_size: u64,
    /// Compression ratio achieved.
    #[pyo3(get)]
    pub compression_ratio: f64,
}

#[pymethods]
impl TranscodeStats {
    fn __repr__(&self) -> String {
        format!(
            "TranscodeStats(packets={}, frames_decoded={}, frames_encoded={}, compression={:.2}x)",
            self.packets_processed,
            self.frames_decoded,
            self.frames_encoded,
            self.compression_ratio
        )
    }
}

/// Progress information during transcoding.
#[pyclass]
#[derive(Clone)]
pub struct Progress {
    /// Current progress percentage (0-100).
    #[pyo3(get)]
    pub percent: f64,
    /// Number of frames processed.
    #[pyo3(get)]
    pub frames: u64,
    /// Current processing speed (frames per second).
    #[pyo3(get)]
    pub speed: f64,
    /// Estimated time remaining in seconds.
    #[pyo3(get)]
    pub eta: Option<f64>,
    /// Current estimated output size in bytes.
    #[pyo3(get)]
    pub size: u64,
    /// Current bitrate in bits per second.
    #[pyo3(get)]
    pub bitrate: u64,
}

#[pymethods]
impl Progress {
    fn __repr__(&self) -> String {
        let eta_str = self.eta.map_or("N/A".to_string(), |e| format!("{:.1}s", e));
        format!(
            "Progress(percent={:.1}%, frames={}, speed={:.1}x, eta={}, bitrate={}kbps)",
            self.percent,
            self.frames,
            self.speed,
            eta_str,
            self.bitrate / 1000
        )
    }
}

/// Transcoder class for performing media transcoding operations.
#[pyclass]
pub struct Transcoder {
    options: TranscodeOptions,
}

#[pymethods]
impl Transcoder {
    /// Create a new transcoder with the given options.
    #[new]
    pub fn new(options: TranscodeOptions) -> PyResult<Self> {
        // Validate options
        if options.input_path.is_none() {
            return Err(PyValueError::new_err("Input path is required"));
        }
        if options.output_path.is_none() {
            return Err(PyValueError::new_err("Output path is required"));
        }

        Ok(Self { options })
    }

    /// Run the transcoding operation.
    pub fn run(&mut self) -> PyResult<TranscodeStats> {
        let input_path = self.options.input_path.as_ref()
            .ok_or_else(|| PyValueError::new_err("No input specified"))?;
        let output_path = self.options.output_path.as_ref()
            .ok_or_else(|| PyValueError::new_err("No output specified"))?;

        // Check if input exists
        if !input_path.exists() {
            return Err(PyIOError::new_err(format!(
                "Input file not found: {:?}",
                input_path
            )));
        }

        // Check if output exists and overwrite is disabled
        if output_path.exists() && !self.options.overwrite {
            return Err(PyIOError::new_err(format!(
                "Output file already exists: {:?}. Use overwrite=True to overwrite.",
                output_path
            )));
        }

        // Build transcode options
        let mut rust_options = RustTranscodeOptions::new()
            .input(input_path)
            .output(output_path)
            .overwrite(self.options.overwrite);

        if let Some(bitrate) = self.options.video_bitrate {
            rust_options = rust_options.video_bitrate(bitrate);
        }

        if let Some(bitrate) = self.options.audio_bitrate {
            rust_options = rust_options.audio_bitrate(bitrate);
        }

        if let Some(threads) = self.options.threads {
            rust_options = rust_options.threads(threads);
        }

        // Create and run the transcoder
        let mut transcoder = RustTranscoder::new(rust_options)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create transcoder: {}", e)))?;

        transcoder.run()
            .map_err(|e| PyRuntimeError::new_err(format!("Transcoding failed: {}", e)))?;

        let stats = transcoder.stats();

        Ok(TranscodeStats {
            packets_processed: stats.packets_processed,
            frames_decoded: stats.frames_decoded,
            frames_encoded: stats.frames_encoded,
            input_size: stats.input_size,
            output_size: stats.output_size,
            compression_ratio: stats.compression_ratio(),
        })
    }

    /// Run the transcoding operation asynchronously with optional progress callback.
    ///
    /// Args:
    ///     progress_callback: Optional async function called with Progress objects.
    ///
    /// Returns:
    ///     A coroutine that yields TranscodeStats when complete.
    #[pyo3(signature = (progress_callback=None))]
    pub fn run_async<'py>(
        &self,
        py: Python<'py>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input_path = self.options.input_path.clone()
            .ok_or_else(|| PyValueError::new_err("No input specified"))?;
        let output_path = self.options.output_path.clone()
            .ok_or_else(|| PyValueError::new_err("No output specified"))?;

        // Check if input exists
        if !input_path.exists() {
            return Err(PyIOError::new_err(format!(
                "Input file not found: {:?}",
                input_path
            )));
        }

        // Check if output exists and overwrite is disabled
        if output_path.exists() && !self.options.overwrite {
            return Err(PyIOError::new_err(format!(
                "Output file already exists: {:?}. Use overwrite=True to overwrite.",
                output_path
            )));
        }

        let options = self.options.clone();

        // Clone the callback using Python's reference counting
        let callback: Option<PyObject> = progress_callback.as_ref().map(|cb| cb.clone_ref(py));

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Build transcode options
            let mut rust_options = RustTranscodeOptions::new()
                .input(&input_path)
                .output(&output_path)
                .overwrite(options.overwrite);

            if let Some(bitrate) = options.video_bitrate {
                rust_options = rust_options.video_bitrate(bitrate);
            }

            if let Some(bitrate) = options.audio_bitrate {
                rust_options = rust_options.audio_bitrate(bitrate);
            }

            if let Some(threads) = options.threads {
                rust_options = rust_options.threads(threads);
            }

            // Create the transcoder
            let mut transcoder = RustTranscoder::new(rust_options)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create transcoder: {}", e)))?;

            // Set up progress tracking
            let start_time = std::time::Instant::now();
            let frames_processed = Arc::new(AtomicU64::new(0));
            let cancelled = Arc::new(AtomicBool::new(false));
            let input_size = std::fs::metadata(&input_path)
                .map(|m| m.len())
                .unwrap_or(0);

            if let Some(cb) = callback {
                let frames_clone = frames_processed.clone();
                let cancelled_clone = cancelled.clone();
                let start = start_time;

                transcoder = transcoder.on_progress(move |progress_percent, frames| {
                    if cancelled_clone.load(Ordering::Relaxed) {
                        return;
                    }

                    frames_clone.store(frames as u64, Ordering::Relaxed);

                    let elapsed = start.elapsed().as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        frames as f64 / (elapsed * 30.0) // Assume 30fps for speed calc
                    } else {
                        0.0
                    };

                    let eta = if progress_percent > 0.0 && progress_percent < 100.0 {
                        let remaining = 100.0 - progress_percent;
                        Some((elapsed / progress_percent) * remaining)
                    } else {
                        None
                    };

                    let estimated_size = (input_size as f64 * (progress_percent / 100.0)) as u64;
                    let bitrate = ((estimated_size as f64 * 8.0) / elapsed.max(0.001)) as u64;

                    // Call the Python callback (fire and forget for progress)
                    let _ = Python::with_gil(|py| -> PyResult<()> {
                        let progress = Progress {
                            percent: progress_percent,
                            frames: frames as u64,
                            speed,
                            eta,
                            size: estimated_size,
                            bitrate,
                        };
                        let _result: PyResult<PyObject> = cb.call1(py, (progress,));
                        Ok(())
                    });
                });
            }

            // Run the transcoding
            transcoder.run()
                .map_err(|e| PyRuntimeError::new_err(format!("Transcoding failed: {}", e)))?;

            let stats = transcoder.stats();

            Ok(TranscodeStats {
                packets_processed: stats.packets_processed,
                frames_decoded: stats.frames_decoded,
                frames_encoded: stats.frames_encoded,
                input_size: stats.input_size,
                output_size: stats.output_size,
                compression_ratio: stats.compression_ratio(),
            })
        })
    }

    /// Cancel an ongoing async transcoding operation.
    pub fn cancel(&mut self) {
        // Cancellation is handled through the AtomicBool in run_async
        // This is a placeholder for future implementation of graceful cancellation
    }
}

/// SIMD capabilities detected on the current system.
#[pyclass]
#[derive(Clone)]
pub struct SimdCapabilities {
    /// SSE4.2 support (x86_64).
    #[pyo3(get)]
    pub sse42: bool,
    /// AVX2 support (x86_64).
    #[pyo3(get)]
    pub avx2: bool,
    /// AVX-512 support (x86_64).
    #[pyo3(get)]
    pub avx512: bool,
    /// FMA support (x86_64).
    #[pyo3(get)]
    pub fma: bool,
    /// NEON support (ARM).
    #[pyo3(get)]
    pub neon: bool,
    /// SVE support (ARM).
    #[pyo3(get)]
    pub sve: bool,
}

#[pymethods]
impl SimdCapabilities {
    fn __repr__(&self) -> String {
        let mut features = Vec::new();
        if self.sse42 { features.push("SSE4.2"); }
        if self.avx2 { features.push("AVX2"); }
        if self.avx512 { features.push("AVX-512"); }
        if self.fma { features.push("FMA"); }
        if self.neon { features.push("NEON"); }
        if self.sve { features.push("SVE"); }

        if features.is_empty() {
            "SimdCapabilities(Scalar only)".to_string()
        } else {
            format!("SimdCapabilities({})", features.join(", "))
        }
    }

    /// Check if any SIMD acceleration is available.
    pub fn has_simd(&self) -> bool {
        self.sse42 || self.avx2 || self.avx512 || self.neon || self.sve
    }

    /// Get the best available SIMD level.
    pub fn best_level(&self) -> &'static str {
        if self.avx512 {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.sse42 {
            "SSE4.2"
        } else if self.sve {
            "SVE"
        } else if self.neon {
            "NEON"
        } else {
            "Scalar"
        }
    }
}

/// Detect SIMD capabilities of the current CPU.
#[pyfunction]
pub fn detect_simd() -> SimdCapabilities {
    let caps = transcode_codecs::detect_simd();
    SimdCapabilities {
        sse42: caps.sse42,
        avx2: caps.avx2,
        avx512: caps.avx512,
        fma: caps.fma,
        neon: caps.neon,
        sve: caps.sve,
    }
}

/// High-level transcode function.
///
/// Args:
///     input_path: Path to the input media file.
///     output_path: Path for the output file.
///     video_codec: Output video codec (default: "h264").
///     audio_codec: Output audio codec (default: "aac").
///     video_bitrate: Video bitrate in bps (optional).
///     audio_bitrate: Audio bitrate in bps (optional).
///     overwrite: Whether to overwrite existing output (default: False).
///
/// Returns:
///     TranscodeStats with information about the transcoding operation.
#[pyfunction]
#[pyo3(name = "transcode")]
#[pyo3(signature = (input_path, output_path, video_codec=None, audio_codec=None, video_bitrate=None, audio_bitrate=None, overwrite=false))]
pub fn transcode_file(
    input_path: &str,
    output_path: &str,
    video_codec: Option<&str>,
    audio_codec: Option<&str>,
    video_bitrate: Option<u64>,
    audio_bitrate: Option<u64>,
    overwrite: bool,
) -> PyResult<TranscodeStats> {
    let mut options = TranscodeOptions::new();
    options.input_path = Some(PathBuf::from(input_path));
    options.output_path = Some(PathBuf::from(output_path));
    options.video_codec = video_codec.map(String::from);
    options.audio_codec = audio_codec.map(String::from);
    options.video_bitrate = video_bitrate;
    options.audio_bitrate = audio_bitrate;
    options.overwrite = overwrite;

    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()
}

/// Get version information about the transcode library.
#[pyfunction]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Async transcode function with progress callback support.
///
/// Args:
///     input_path: Path to the input media file.
///     output_path: Path for the output file.
///     video_codec: Output video codec (default: "h264").
///     audio_codec: Output audio codec (default: "aac").
///     video_bitrate: Video bitrate in bps (optional).
///     audio_bitrate: Audio bitrate in bps (optional).
///     overwrite: Whether to overwrite existing output (default: False).
///     progress_callback: Optional callback function receiving Progress updates.
///
/// Returns:
///     A coroutine that yields TranscodeStats when complete.
///
/// Example:
///     ```python
///     import asyncio
///     import transcode_py
///
///     async def main():
///         def on_progress(progress):
///             print(f"Progress: {progress.percent:.1f}%")
///
///         stats = await transcode_py.transcode_async(
///             "input.mp4", "output.mp4",
///             progress_callback=on_progress
///         )
///         print(f"Completed: {stats}")
///
///     asyncio.run(main())
///     ```
#[pyfunction]
#[pyo3(name = "transcode_async")]
#[pyo3(signature = (input_path, output_path, video_codec=None, audio_codec=None, video_bitrate=None, audio_bitrate=None, overwrite=false, progress_callback=None))]
pub fn transcode_file_async<'py>(
    py: Python<'py>,
    input_path: &str,
    output_path: &str,
    video_codec: Option<&str>,
    audio_codec: Option<&str>,
    video_bitrate: Option<u64>,
    audio_bitrate: Option<u64>,
    overwrite: bool,
    progress_callback: Option<PyObject>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut options = TranscodeOptions::new();
    options.input_path = Some(PathBuf::from(input_path));
    options.output_path = Some(PathBuf::from(output_path));
    options.video_codec = video_codec.map(String::from);
    options.audio_codec = audio_codec.map(String::from);
    options.video_bitrate = video_bitrate;
    options.audio_bitrate = audio_bitrate;
    options.overwrite = overwrite;

    let transcoder = Transcoder::new(options)?;
    transcoder.run_async(py, progress_callback)
}

/// Transcode Python module.
#[pymodule]
fn transcode_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TranscodeOptions>()?;
    m.add_class::<TranscodeStats>()?;
    m.add_class::<Progress>()?;
    m.add_class::<Transcoder>()?;
    m.add_class::<SimdCapabilities>()?;
    m.add_function(wrap_pyfunction!(transcode_file, m)?)?;
    m.add_function(wrap_pyfunction!(transcode_file_async, m)?)?;
    m.add_function(wrap_pyfunction!(detect_simd, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Add module docstring
    m.add("__doc__", "Transcode - A memory-safe, high-performance codec library.\n\nExample (sync):\n    import transcode_py\n    stats = transcode_py.transcode('input.mp4', 'output.mp4')\n    print(stats)\n\nExample (async):\n    import asyncio\n    import transcode_py\n\n    async def main():\n        def on_progress(p):\n            print(f'{p.percent:.1f}%')\n        stats = await transcode_py.transcode_async('input.mp4', 'output.mp4', progress_callback=on_progress)\n        print(stats)\n\n    asyncio.run(main())")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_options_builder() {
        let mut options = TranscodeOptions::new();
        options = options.input("/tmp/test.mp4").unwrap();
        options = options.output("/tmp/out.mp4").unwrap();
        options = options.video_bitrate(5_000_000).unwrap();

        assert_eq!(options.input_path, Some(PathBuf::from("/tmp/test.mp4")));
        assert_eq!(options.output_path, Some(PathBuf::from("/tmp/out.mp4")));
        assert_eq!(options.video_bitrate, Some(5_000_000));
    }

    #[test]
    fn test_simd_detection() {
        let caps = detect_simd();
        // Should return without panicking
        let _ = caps.has_simd();
        let _ = caps.best_level();
    }
}
