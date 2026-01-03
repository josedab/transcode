//! Comprehensive codec benchmarks.
//!
//! This module benchmarks all codec operations including:
//! - H.264 encoding at various resolutions
//! - SIMD operations (IDCT, SAD, Hadamard, etc.)

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use transcode_codecs::video::h264::{H264Encoder, H264EncoderConfig, H264Profile, H264Level, RateControlMode, EncoderPreset, ThreadingConfig};
use transcode_codecs::traits::VideoEncoder;
use transcode_codecs::{SimdOps, detect_simd};
use transcode_core::{Frame, PixelFormat, TimeBase, Timestamp};

/// Create a test video frame with a gradient pattern.
fn create_test_frame(width: u32, height: u32, frame_num: u32) -> Frame {
    let time_base = TimeBase::new(1, 30);
    let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, time_base);
    frame.pts = Timestamp::new(frame_num as i64, time_base);

    // Fill Y plane with gradient
    if let Some(y_plane) = frame.plane_mut(0) {
        for (i, pixel) in y_plane.iter_mut().enumerate() {
            *pixel = ((i + frame_num as usize) % 256) as u8;
        }
    }

    // Fill U/V planes with neutral chroma
    if let Some(u_plane) = frame.plane_mut(1) {
        for pixel in u_plane.iter_mut() {
            *pixel = 128;
        }
    }

    if let Some(v_plane) = frame.plane_mut(2) {
        for pixel in v_plane.iter_mut() {
            *pixel = 128;
        }
    }

    frame
}

// ============================================================================
// H.264 Benchmarks
// ============================================================================

fn bench_h264_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("h264_encode");
    group.sample_size(20);

    let resolutions = [
        ("480p", 854, 480),
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
    ];

    for (name, width, height) in resolutions {
        let frame = create_test_frame(width, height, 0);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &frame,
            |b, frame| {
                let config = H264EncoderConfig {
                    width,
                    height,
                    frame_rate: (30, 1),
                    profile: H264Profile::High,
                    level: H264Level::from_idc(40),
                    rate_control: RateControlMode::Cbr(5_000_000),
                    preset: EncoderPreset::Fast,
                    gop_size: 30,
                    bframes: 0,
                    ref_frames: 3,
                    cabac: true,
                };
                let mut encoder = H264Encoder::new(config).expect("create encoder");

                b.iter(|| {
                    let _ = encoder.encode(black_box(frame));
                });
            },
        );
    }

    group.finish();
}

fn bench_h264_encode_gop(c: &mut Criterion) {
    let mut group = c.benchmark_group("h264_encode_gop");
    group.sample_size(10);

    let gop_size = 30u32;
    let frames: Vec<Frame> = (0..gop_size)
        .map(|i| create_test_frame(1280, 720, i))
        .collect();

    group.throughput(Throughput::Elements(gop_size as u64));
    group.bench_function("720p_30_frames", |b| {
        let config = H264EncoderConfig {
            width: 1280,
            height: 720,
            frame_rate: (30, 1),
            profile: H264Profile::High,
            level: H264Level::from_idc(31),
            rate_control: RateControlMode::Cbr(5_000_000),
            preset: EncoderPreset::Fast,
            gop_size,
            bframes: 0,
            ref_frames: 3,
            cabac: true,
        };
        let mut encoder = H264Encoder::new(config).expect("create encoder");

        b.iter(|| {
            for frame in &frames {
                let _ = encoder.encode(black_box(frame));
            }
        });
    });

    group.finish();
}

// ============================================================================
// SIMD Benchmarks
// ============================================================================

fn bench_simd_idct4x4(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_idct4x4");

    let simd = SimdOps::new();
    let caps = detect_simd();

    let input: [i16; 16] = [
        100, 50, 25, 10,
        50, 25, 10, 5,
        25, 10, 5, 2,
        10, 5, 2, 1,
    ];
    let mut output = [0i16; 16];

    group.throughput(Throughput::Elements(16));
    group.bench_function(caps.best_level(), |b| {
        b.iter(|| {
            simd.idct4x4(black_box(&input), black_box(&mut output));
        });
    });

    group.finish();
}

fn bench_simd_idct8x8(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_idct8x8");

    let simd = SimdOps::new();
    let caps = detect_simd();

    let input: [i16; 64] = [50; 64];
    let mut output = [0i16; 64];

    group.throughput(Throughput::Elements(64));
    group.bench_function(caps.best_level(), |b| {
        b.iter(|| {
            simd.idct8x8(black_box(&input), black_box(&mut output));
        });
    });

    group.finish();
}

fn bench_simd_sad16x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_sad16x16");

    let simd = SimdOps::new();
    let caps = detect_simd();

    // 16x16 blocks with stride
    let block1 = vec![128u8; 16 * 16];
    let block2 = vec![130u8; 16 * 16];

    group.throughput(Throughput::Bytes(512));
    group.bench_function(caps.best_level(), |b| {
        b.iter(|| {
            simd.sad16x16(black_box(&block1), 16, black_box(&block2), 16)
        });
    });

    group.finish();
}

fn bench_simd_hadamard4x4(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_hadamard4x4");

    let simd = SimdOps::new();
    let caps = detect_simd();

    let input: [i16; 16] = [1; 16];
    let mut output = [0i16; 16];

    group.throughput(Throughput::Elements(16));
    group.bench_function(caps.best_level(), |b| {
        b.iter(|| {
            simd.hadamard4x4(black_box(&input), black_box(&mut output));
        });
    });

    group.finish();
}

fn bench_simd_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_quantize");

    let simd = SimdOps::new();
    let caps = detect_simd();

    let input: [i16; 16] = [100, 200, 300, 400, 50, 60, 70, 80, 10, 20, 30, 40, 5, 6, 7, 8];
    let mut output = [0i16; 16];
    let qp = 22;

    group.throughput(Throughput::Elements(16));
    group.bench_function(caps.best_level(), |b| {
        b.iter(|| {
            simd.quantize(black_box(&input), black_box(&mut output), qp, true);
        });
    });

    group.finish();
}

fn bench_simd_mc_bilinear(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_mc_bilinear");

    let simd = SimdOps::new();
    let caps = detect_simd();

    // 64x64 reference block
    let src = vec![128u8; 64 * 64];
    let mut dst = vec![0u8; 16 * 16];

    group.throughput(Throughput::Bytes(256));
    group.bench_function(caps.best_level(), |b| {
        b.iter(|| {
            simd.mc_bilinear(
                black_box(&mut dst),
                16,
                black_box(&src),
                64,
                16,
                16,
                4,
                4,
            );
        });
    });

    group.finish();
}

// ============================================================================
// Multi-threaded Encoding Benchmarks
// ============================================================================

/// Benchmark single-threaded vs multi-threaded H.264 encoding.
fn bench_h264_threading_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("h264_threading");
    group.sample_size(10);

    let resolutions = [
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
    ];

    for (name, width, height) in resolutions {
        let frames: Vec<Frame> = (0..30)
            .map(|i| create_test_frame(width, height, i))
            .collect();

        // Single-threaded benchmark
        group.throughput(Throughput::Elements(30));
        group.bench_with_input(
            BenchmarkId::new("single_threaded", name),
            &frames,
            |b, frames| {
                let mut threading = ThreadingConfig::default();
                threading.enable_slice_parallel = false;
                threading.enable_frame_parallel = false;

                let config = H264EncoderConfig {
                    width,
                    height,
                    frame_rate: (30, 1),
                    profile: H264Profile::High,
                    level: H264Level::from_idc(40),
                    rate_control: RateControlMode::Cbr(5_000_000),
                    preset: EncoderPreset::Fast,
                    gop_size: 30,
                    bframes: 0,
                    ref_frames: 3,
                    cabac: true,
                    threading,
                };
                let mut encoder = H264Encoder::new(config).expect("create encoder");

                b.iter(|| {
                    for frame in frames {
                        let _ = encoder.encode(black_box(frame));
                    }
                });
            },
        );

        // Multi-threaded benchmark (2 threads)
        group.bench_with_input(
            BenchmarkId::new("multi_threaded_2", name),
            &frames,
            |b, frames| {
                let mut threading = ThreadingConfig::with_threads(2);
                threading.slice_count = 2;

                let config = H264EncoderConfig {
                    width,
                    height,
                    frame_rate: (30, 1),
                    profile: H264Profile::High,
                    level: H264Level::from_idc(40),
                    rate_control: RateControlMode::Cbr(5_000_000),
                    preset: EncoderPreset::Fast,
                    gop_size: 30,
                    bframes: 0,
                    ref_frames: 3,
                    cabac: true,
                    threading,
                };
                let mut encoder = H264Encoder::new(config).expect("create encoder");

                b.iter(|| {
                    for frame in frames {
                        let _ = encoder.encode(black_box(frame));
                    }
                });
            },
        );

        // Multi-threaded benchmark (4 threads)
        group.bench_with_input(
            BenchmarkId::new("multi_threaded_4", name),
            &frames,
            |b, frames| {
                let mut threading = ThreadingConfig::with_threads(4);
                threading.slice_count = 4;

                let config = H264EncoderConfig {
                    width,
                    height,
                    frame_rate: (30, 1),
                    profile: H264Profile::High,
                    level: H264Level::from_idc(40),
                    rate_control: RateControlMode::Cbr(5_000_000),
                    preset: EncoderPreset::Fast,
                    gop_size: 30,
                    bframes: 0,
                    ref_frames: 3,
                    cabac: true,
                    threading,
                };
                let mut encoder = H264Encoder::new(config).expect("create encoder");

                b.iter(|| {
                    for frame in frames {
                        let _ = encoder.encode(black_box(frame));
                    }
                });
            },
        );

        // Multi-threaded benchmark (auto-detect threads)
        group.bench_with_input(
            BenchmarkId::new("multi_threaded_auto", name),
            &frames,
            |b, frames| {
                let threading = ThreadingConfig::default();

                let config = H264EncoderConfig {
                    width,
                    height,
                    frame_rate: (30, 1),
                    profile: H264Profile::High,
                    level: H264Level::from_idc(40),
                    rate_control: RateControlMode::Cbr(5_000_000),
                    preset: EncoderPreset::Fast,
                    gop_size: 30,
                    bframes: 0,
                    ref_frames: 3,
                    cabac: true,
                    threading,
                };
                let mut encoder = H264Encoder::new(config).expect("create encoder");

                b.iter(|| {
                    for frame in frames {
                        let _ = encoder.encode(black_box(frame));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark slice count variations.
fn bench_h264_slice_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("h264_slice_count");
    group.sample_size(10);

    let (width, height) = (1920, 1080);
    let frames: Vec<Frame> = (0..30)
        .map(|i| create_test_frame(width, height, i))
        .collect();

    for slice_count in [1, 2, 4, 8, 16] {
        group.throughput(Throughput::Elements(30));
        group.bench_with_input(
            BenchmarkId::new("slices", slice_count),
            &frames,
            |b, frames| {
                let mut threading = ThreadingConfig::default();
                threading.slice_count = slice_count;

                let config = H264EncoderConfig {
                    width,
                    height,
                    frame_rate: (30, 1),
                    profile: H264Profile::High,
                    level: H264Level::from_idc(40),
                    rate_control: RateControlMode::Cbr(5_000_000),
                    preset: EncoderPreset::Fast,
                    gop_size: 30,
                    bframes: 0,
                    ref_frames: 3,
                    cabac: true,
                    threading,
                };
                let mut encoder = H264Encoder::new(config).expect("create encoder");

                b.iter(|| {
                    for frame in frames {
                        let _ = encoder.encode(black_box(frame));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel motion estimation.
fn bench_h264_motion_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("h264_motion_estimation");
    group.sample_size(10);

    let (width, height) = (1280, 720);
    let frame = create_test_frame(width, height, 0);

    // Single-threaded motion estimation
    group.throughput(Throughput::Elements(1));
    group.bench_function("single_threaded", |b| {
        let mut threading = ThreadingConfig::default();
        threading.enable_slice_parallel = false;
        threading.enable_frame_parallel = true;
        threading.num_threads = 1;

        let config = H264EncoderConfig {
            width,
            height,
            frame_rate: (30, 1),
            profile: H264Profile::High,
            level: H264Level::from_idc(40),
            rate_control: RateControlMode::Cbr(5_000_000),
            preset: EncoderPreset::Fast,
            gop_size: 30,
            bframes: 0,
            ref_frames: 3,
            cabac: true,
            threading,
        };
        let encoder = H264Encoder::new(config).expect("create encoder");

        b.iter(|| {
            let _ = encoder.parallel_motion_estimation(black_box(&frame));
        });
    });

    // Multi-threaded motion estimation
    group.bench_function("multi_threaded", |b| {
        let threading = ThreadingConfig::default();

        let config = H264EncoderConfig {
            width,
            height,
            frame_rate: (30, 1),
            profile: H264Profile::High,
            level: H264Level::from_idc(40),
            rate_control: RateControlMode::Cbr(5_000_000),
            preset: EncoderPreset::Fast,
            gop_size: 30,
            bframes: 0,
            ref_frames: 3,
            cabac: true,
            threading,
        };
        let encoder = H264Encoder::new(config).expect("create encoder");

        b.iter(|| {
            let _ = encoder.parallel_motion_estimation(black_box(&frame));
        });
    });

    group.finish();
}

// ============================================================================
// Combined Benchmarks
// ============================================================================

fn bench_simd_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_all_ops");

    let simd = SimdOps::new();
    let caps = detect_simd();

    // Prepare test data
    let idct_input: [i16; 16] = [100; 16];
    let mut idct_output = [0i16; 16];

    let sad_block1 = vec![128u8; 256];
    let sad_block2 = vec![130u8; 256];

    let quant_input: [i16; 16] = [100; 16];
    let mut quant_output = [0i16; 16];

    println!("\nSIMD Level: {}", caps.best_level());
    println!("  AVX2: {}, NEON: {}", caps.avx2, caps.neon);

    group.bench_function("batch_ops", |b| {
        b.iter(|| {
            simd.idct4x4(black_box(&idct_input), black_box(&mut idct_output));
            let _ = simd.sad16x16(black_box(&sad_block1), 16, black_box(&sad_block2), 16);
            simd.quantize(black_box(&quant_input), black_box(&mut quant_output), 22, true);
        });
    });

    group.finish();
}

// ============================================================================
// Main
// ============================================================================

criterion_group!(
    video_benches,
    bench_h264_encode,
    bench_h264_encode_gop,
);

criterion_group!(
    threading_benches,
    bench_h264_threading_comparison,
    bench_h264_slice_count,
    bench_h264_motion_estimation,
);

criterion_group!(
    simd_benches,
    bench_simd_idct4x4,
    bench_simd_idct8x8,
    bench_simd_sad16x16,
    bench_simd_hadamard4x4,
    bench_simd_quantize,
    bench_simd_mc_bilinear,
    bench_simd_comparison,
);

criterion_main!(video_benches, threading_benches, simd_benches);
