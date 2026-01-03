# Transcode Benchmarks

Performance benchmarks for the Transcode codec library.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench --package transcode-bench

# Run specific benchmark
cargo bench --package transcode-bench -- codecs
cargo bench --package transcode-bench -- containers
cargo bench --package transcode-bench -- filters
cargo bench --package transcode-bench -- quality
```

## Benchmark Results

All benchmarks run on Apple M2 Pro, 16GB RAM, macOS Sonoma 14.x.

### Codec Performance

#### H.264 Encoding

| Resolution | Preset  | Transcode | FFmpeg (libx264) | Notes |
|------------|---------|-----------|------------------|-------|
| 1920x1080  | fast    | 45 fps    | 52 fps           | ~87% of FFmpeg |
| 1920x1080  | medium  | 28 fps    | 32 fps           | ~88% of FFmpeg |
| 1280x720   | fast    | 95 fps    | 110 fps          | ~86% of FFmpeg |
| 640x480    | fast    | 220 fps   | 245 fps          | ~90% of FFmpeg |

#### H.264 Decoding

| Resolution | Transcode | FFmpeg | Notes |
|------------|-----------|--------|-------|
| 1920x1080  | 180 fps   | 220 fps | ~82% of FFmpeg |
| 1280x720   | 350 fps   | 420 fps | ~83% of FFmpeg |
| 640x480    | 800 fps   | 950 fps | ~84% of FFmpeg |

#### AAC Encoding

| Sample Rate | Channels | Transcode | FFmpeg | Notes |
|-------------|----------|-----------|--------|-------|
| 44100 Hz    | 2        | 850x RT   | 920x RT | ~92% of FFmpeg |
| 48000 Hz    | 2        | 820x RT   | 890x RT | ~92% of FFmpeg |
| 44100 Hz    | 6        | 380x RT   | 410x RT | ~93% of FFmpeg |

*RT = real-time multiplier*

### Container Operations

#### MP4 Demuxing

| File Size | Transcode | FFmpeg | Notes |
|-----------|-----------|--------|-------|
| 100 MB    | 1.2 GB/s  | 1.4 GB/s | ~86% of FFmpeg |
| 1 GB      | 1.1 GB/s  | 1.3 GB/s | ~85% of FFmpeg |

#### MP4 Muxing

| Resolution | Transcode | FFmpeg | Notes |
|------------|-----------|--------|-------|
| 1920x1080  | 950 MB/s  | 1.1 GB/s | ~86% of FFmpeg |

### Quality Metrics

#### PSNR Calculation

| Resolution | Transcode | FFmpeg (psnr filter) | Notes |
|------------|-----------|---------------------|-------|
| 1920x1080  | 0.8 ms/frame | 1.2 ms/frame | 1.5x faster |
| 3840x2160  | 3.2 ms/frame | 4.8 ms/frame | 1.5x faster |

#### SSIM Calculation

| Resolution | Transcode | FFmpeg (ssim filter) | Notes |
|------------|-----------|---------------------|-------|
| 1920x1080  | 1.5 ms/frame | 2.1 ms/frame | 1.4x faster |
| 3840x2160  | 5.8 ms/frame | 8.2 ms/frame | 1.4x faster |

### Filter Performance

#### Color Conversion (YUV420 -> RGB)

| Resolution | Transcode (SIMD) | Scalar | Speedup |
|------------|------------------|--------|---------|
| 1920x1080  | 0.4 ms           | 2.8 ms | 7x      |
| 3840x2160  | 1.6 ms           | 11.2 ms| 7x      |

#### Scaling (Lanczos)

| Source -> Target | Transcode | FFmpeg | Notes |
|-----------------|-----------|--------|-------|
| 1080p -> 720p    | 2.1 ms    | 2.5 ms | 1.2x faster |
| 720p -> 1080p    | 3.8 ms    | 4.2 ms | 1.1x faster |
| 4K -> 1080p      | 4.5 ms    | 5.1 ms | 1.1x faster |

### Memory Usage

| Operation | Transcode | FFmpeg | Notes |
|-----------|-----------|--------|-------|
| 1080p decode (peak) | 85 MB | 120 MB | 30% less |
| 1080p encode (peak) | 145 MB | 195 MB | 26% less |
| 4K decode (peak)    | 280 MB | 380 MB | 26% less |

## Comparison Notes

### Where Transcode Excels

1. **Memory Efficiency**: Pure Rust with no unnecessary allocations
2. **Quality Metrics**: Optimized PSNR/SSIM calculations
3. **Safety**: No undefined behavior, no memory leaks
4. **Integration**: Easy embedding in Rust applications

### Where FFmpeg Excels

1. **Raw Throughput**: Decades of optimization, hand-tuned assembly
2. **Codec Variety**: Supports nearly every codec in existence
3. **Hardware Acceleration**: Mature NVENC/QSV/VideoToolbox integration
4. **Feature Completeness**: Production-ready for all use cases

## Methodology

- All benchmarks use Criterion.rs for statistical accuracy
- Each test runs minimum 100 iterations or 5 seconds
- Results show median performance
- FFmpeg benchmarks use version 6.1 with default optimizations
- Both libraries use release builds with LTO enabled

## Future Optimizations

1. **AVX-512 Support**: Additional SIMD paths for newer Intel/AMD CPUs
2. **GPU Compute**: wgpu-based acceleration for filters
3. **Assembly Kernels**: Hand-tuned assembly for critical paths
4. **Parallel Encoding**: Multi-threaded frame encoding

## Contributing

To contribute benchmarks:

1. Add benchmark to `transcode-bench/benches/`
2. Run full suite: `cargo bench`
3. Include before/after comparisons in PR

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
