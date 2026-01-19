# ADR-0012: Python Bindings via PyO3

## Status

Accepted

## Date

2024-06 (inferred from module structure)

## Context

Python is the dominant language for:

1. **Data science and ML workflows** - video processing pipelines
2. **Scripting and automation** - batch transcoding jobs
3. **Rapid prototyping** - testing codec configurations
4. **Integration** - combining with other Python libraries

Users in these domains expect:

- Pythonic API (not just C-style bindings)
- Async support for non-blocking operations
- Progress callbacks for long-running jobs
- Integration with Python async frameworks (asyncio)
- Type hints for IDE support

The challenge is exposing Rust's performance while providing an ergonomic Python experience.

## Decision

Use **PyO3** for Python bindings with **asyncio integration** via pyo3-async-runtimes, **builder patterns** for configuration, and **progress callbacks** for monitoring.

### 1. PyO3 Module Structure

Define the Python module with PyO3:

```rust
use pyo3::prelude::*;

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
    Ok(())
}
```

### 2. Builder Pattern for Options

Pythonic configuration with method chaining:

```rust
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
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn input(&mut self, path: &str) -> PyResult<Self> {
        self.input_path = Some(PathBuf::from(path));
        Ok(self.clone())
    }

    pub fn output(&mut self, path: &str) -> PyResult<Self> {
        self.output_path = Some(PathBuf::from(path));
        Ok(self.clone())
    }

    pub fn video_codec(&mut self, codec: &str) -> PyResult<Self> {
        self.video_codec = Some(codec.to_string());
        Ok(self.clone())
    }

    pub fn video_bitrate(&mut self, bitrate: u64) -> PyResult<Self> {
        self.video_bitrate = Some(bitrate);
        Ok(self.clone())
    }

    pub fn resolution(&mut self, width: u32, height: u32) -> PyResult<Self> {
        self.width = Some(width);
        self.height = Some(height);
        Ok(self.clone())
    }
}
```

### 3. Progress Tracking

Rich progress information for monitoring:

```rust
#[pyclass]
#[derive(Clone)]
pub struct Progress {
    #[pyo3(get)]
    pub percent: f64,
    #[pyo3(get)]
    pub frames: u64,
    #[pyo3(get)]
    pub speed: f64,
    #[pyo3(get)]
    pub eta: Option<f64>,
    #[pyo3(get)]
    pub size: u64,
    #[pyo3(get)]
    pub bitrate: u64,
}

#[pymethods]
impl Progress {
    fn __repr__(&self) -> String {
        format!(
            "Progress(percent={:.1}%, frames={}, speed={:.1}x, eta={}, bitrate={}kbps)",
            self.percent,
            self.frames,
            self.speed,
            self.eta.map_or("N/A".to_string(), |e| format!("{:.1}s", e)),
            self.bitrate / 1000
        )
    }
}
```

### 4. Async Transcoding with Callbacks

Support asyncio with progress callbacks:

```rust
#[pymethods]
impl Transcoder {
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

        let options = self.options.clone();
        let callback: Option<PyObject> = progress_callback.as_ref().map(|cb| cb.clone_ref(py));

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Build transcode options
            let mut rust_options = RustTranscodeOptions::new()
                .input(&input_path)
                .output(&output_path);

            // Set up progress callback
            if let Some(cb) = callback {
                let start = std::time::Instant::now();

                transcoder = transcoder.on_progress(move |progress_percent, frames| {
                    let elapsed = start.elapsed().as_secs_f64();
                    let speed = frames as f64 / (elapsed * 30.0);

                    Python::with_gil(|py| {
                        let progress = Progress {
                            percent: progress_percent,
                            frames: frames as u64,
                            speed,
                            // ...
                        };
                        let _ = cb.call1(py, (progress,));
                    });
                });
            }

            // Run transcoding
            transcoder.run()?;

            Ok(TranscodeStats { /* ... */ })
        })
    }
}
```

### 5. Synchronous API

Simple blocking API for scripts:

```rust
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
    // ...

    let mut transcoder = Transcoder::new(options)?;
    transcoder.run()
}
```

### 6. Statistics and Results

Rich return types with Python repr:

```rust
#[pyclass]
#[derive(Clone)]
pub struct TranscodeStats {
    #[pyo3(get)]
    pub packets_processed: u64,
    #[pyo3(get)]
    pub frames_decoded: u64,
    #[pyo3(get)]
    pub frames_encoded: u64,
    #[pyo3(get)]
    pub input_size: u64,
    #[pyo3(get)]
    pub output_size: u64,
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
```

### 7. SIMD Detection

Expose hardware capabilities:

```rust
#[pyclass]
#[derive(Clone)]
pub struct SimdCapabilities {
    #[pyo3(get)]
    pub sse42: bool,
    #[pyo3(get)]
    pub avx2: bool,
    #[pyo3(get)]
    pub avx512: bool,
    #[pyo3(get)]
    pub neon: bool,
}

#[pymethods]
impl SimdCapabilities {
    pub fn has_simd(&self) -> bool {
        self.sse42 || self.avx2 || self.avx512 || self.neon
    }

    pub fn best_level(&self) -> &'static str {
        if self.avx512 { "AVX-512" }
        else if self.avx2 { "AVX2" }
        else if self.sse42 { "SSE4.2" }
        else if self.neon { "NEON" }
        else { "Scalar" }
    }
}

#[pyfunction]
pub fn detect_simd() -> SimdCapabilities {
    let caps = transcode_codecs::detect_simd();
    SimdCapabilities {
        sse42: caps.sse42,
        avx2: caps.avx2,
        avx512: caps.avx512,
        neon: caps.neon,
    }
}
```

### Python Usage Examples

```python
# Simple synchronous usage
import transcode_py

stats = transcode_py.transcode(
    "input.mp4",
    "output.mp4",
    video_codec="h264",
    video_bitrate=2_000_000
)
print(stats)

# Builder pattern
options = (transcode_py.TranscodeOptions()
    .input("input.mp4")
    .output("output.mp4")
    .video_codec("h264")
    .video_bitrate(2_000_000)
    .resolution(1920, 1080))

transcoder = transcode_py.Transcoder(options)
stats = transcoder.run()

# Async with progress
import asyncio

async def main():
    def on_progress(progress):
        print(f"Progress: {progress.percent:.1f}%")

    stats = await transcode_py.transcode_async(
        "input.mp4",
        "output.mp4",
        progress_callback=on_progress
    )
    print(f"Done! {stats}")

asyncio.run(main())

# Check hardware capabilities
caps = transcode_py.detect_simd()
print(f"SIMD: {caps.best_level()}")
```

## Consequences

### Positive

1. **Pythonic API**: Feels natural to Python developers

2. **Async support**: Integrates with asyncio for non-blocking operations

3. **Progress tracking**: Real-time feedback for long jobs

4. **Type hints**: IDE autocompletion and type checking

5. **Zero-copy where possible**: Efficient data transfer between Rust and Python

6. **Native performance**: Rust code runs at full speed

### Negative

1. **GIL interaction**: Must release GIL for long operations

2. **Build complexity**: Requires maturin for wheel building

3. **Platform wheels**: Need to build for each OS/Python version

4. **Callback overhead**: Python callbacks from Rust have some overhead

### Mitigations

1. **GIL release**: Use `py.allow_threads()` for CPU-intensive work

2. **maturin**: Streamlined build process for Python packages

3. **CI/CD**: Automated wheel building for major platforms

4. **Batched callbacks**: Limit callback frequency for performance

## Alternatives Considered

### Alternative 1: CFFI

Use CFFI for C-compatible bindings.

Rejected because:
- Manual memory management
- No type safety
- More boilerplate code
- Less Pythonic API

### Alternative 2: ctypes

Use ctypes with extern "C" functions.

Rejected because:
- Very manual and error-prone
- No async support
- Poor error handling
- Difficult to maintain

### Alternative 3: Cython

Write Python extension in Cython.

Rejected because:
- Separate language to maintain
- No direct Rust integration
- Less control over memory

### Alternative 4: SWIG

Use SWIG to generate bindings.

Rejected because:
- Complex configuration
- Generated code hard to customize
- Limited Rust support

## References

- [PyO3 user guide](https://pyo3.rs/)
- [maturin documentation](https://maturin.rs/)
- [pyo3-async-runtimes](https://docs.rs/pyo3-async-runtimes/)
- [Python C API](https://docs.python.org/3/c-api/)
