# transcode-bench

Benchmark suite for the transcode library. Provides standardized benchmarking infrastructure for measuring codec, filter, container, and quality metrics performance.

## Features

- Custom `Benchmark` and `BenchmarkSuite` types for flexible performance testing
- Pre-built utilities for generating test video frames and audio samples
- Common resolutions (480p, 720p, 1080p, 4K) and sample rates (44.1kHz, 48kHz, 96kHz)
- JSON export for benchmark results
- Integration with [Criterion](https://github.com/bheisler/criterion.rs) for statistical benchmarking

## Available Benchmarks

### Codecs (`codecs`)

Frame operations and memory performance:
- **frame_creation** - Frame allocation at various resolutions
- **frame_copy** - Frame cloning performance
- **plane_access** - Y plane data access
- **pixel_iteration** - Pixel traversal throughput

### Filters (`filters`)

Image processing filter performance:
- **filter_brightness** - Brightness adjustment at 720p/1080p
- **filter_contrast** - Contrast adjustment at 720p/1080p
- **filter_threshold** - Binary thresholding at 1080p

### Containers (`containers`)

Packet handling performance:
- **packet_creation** - Packet allocation (1KB, 64KB, 1MB)
- **packet_clone** - Packet cloning (1KB, 64KB)

### Quality Metrics (`quality`)

Quality assessment algorithm performance:
- **psnr** - PSNR calculation at 480p/720p/1080p
- **ssim** - SSIM calculation at 480p/720p/1080p
- **ms_ssim** - Multi-scale SSIM at 480p/720p
- **vmaf** - VMAF approximation at 240p/480p
- **identical_frames** - Edge case performance for identical inputs

## Running Benchmarks

Run all benchmarks:
```bash
cargo bench
```

Run a specific benchmark suite:
```bash
cargo bench --bench codecs
cargo bench --bench filters
cargo bench --bench containers
cargo bench --bench quality
```

Run a specific benchmark group:
```bash
cargo bench -- frame_creation
cargo bench -- psnr
```

Filter by benchmark name pattern:
```bash
cargo bench -- "1080p"
cargo bench -- "ssim"
```

## Using the Library

```rust
use transcode_bench::{Benchmark, BenchmarkSuite, video, audio};

// Run a custom benchmark
let result = Benchmark::new("my_operation")
    .warmup(10)
    .iterations(100)
    .items_per_iter(1920 * 1080)
    .run(|| {
        // operation to benchmark
    });

result.print();

// Generate test data
let frames = video::generate_test_frames(10, 1920, 1080);
let samples = audio::generate_test_samples(48000, 48000);

// Collect results in a suite
let mut suite = BenchmarkSuite::new("My Benchmarks");
suite.add(result);
suite.print();
println!("{}", suite.to_json());
```

## Output

Criterion generates HTML reports in `target/criterion/`. Open `target/criterion/report/index.html` to view interactive benchmark results with statistical analysis.
