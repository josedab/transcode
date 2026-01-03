//! Quality metrics benchmarks.
//!
//! Benchmarks for PSNR, SSIM, MS-SSIM, and VMAF calculations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use transcode_quality::{Frame, MsSsim, Psnr, PsnrConfig, Ssim, SsimConfig, Vmaf, VmafConfig};

/// Create a test frame with a gradient pattern.
fn create_test_frame(width: u32, height: u32, offset: u8) -> Frame {
    let mut data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let val = ((x * 255 / width) as u8).wrapping_add(offset);
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }

    Frame::new(data, width, height, 3)
}

/// Create a noisy version of a frame.
fn add_noise(frame: &Frame, amount: u8) -> Frame {
    let mut data = frame.data.clone();

    for (i, pixel) in data.iter_mut().enumerate() {
        let noise = ((i * 17 + i / 3) % (amount as usize * 2)) as i16 - amount as i16;
        *pixel = (*pixel as i16 + noise).clamp(0, 255) as u8;
    }

    Frame::new(data, frame.width, frame.height, frame.channels)
}

// ============================================================================
// PSNR Benchmarks
// ============================================================================

fn bench_psnr(c: &mut Criterion) {
    let mut group = c.benchmark_group("psnr");

    let resolutions = [
        ("256x256", 256, 256),
        ("720p", 1280, 720),
        ("1080p", 1920, 1080),
    ];

    let psnr = Psnr::new(PsnrConfig::default());

    for (name, width, height) in resolutions {
        let reference = create_test_frame(width, height, 0);
        let distorted = add_noise(&reference, 10);
        let pixels = (width * height) as u64;

        group.throughput(Throughput::Elements(pixels));
        group.bench_with_input(BenchmarkId::from_parameter(name), &pixels, |b, _| {
            b.iter(|| psnr.calculate(black_box(&reference), black_box(&distorted)));
        });
    }

    group.finish();
}

// ============================================================================
// SSIM Benchmarks
// ============================================================================

fn bench_ssim(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssim");
    group.sample_size(20);

    let resolutions = [
        ("256x256", 256, 256),
        ("720p", 1280, 720),
    ];

    let ssim = Ssim::new(SsimConfig::default());

    for (name, width, height) in resolutions {
        let reference = create_test_frame(width, height, 0);
        let distorted = add_noise(&reference, 10);
        let pixels = (width * height) as u64;

        group.throughput(Throughput::Elements(pixels));
        group.bench_with_input(BenchmarkId::from_parameter(name), &pixels, |b, _| {
            b.iter(|| ssim.calculate(black_box(&reference), black_box(&distorted)));
        });
    }

    group.finish();
}

// ============================================================================
// MS-SSIM Benchmarks
// ============================================================================

fn bench_ms_ssim(c: &mut Criterion) {
    let mut group = c.benchmark_group("ms_ssim");
    group.sample_size(10);

    // MS-SSIM requires larger images for 5 scales
    let resolutions = [("512x512", 512, 512), ("720p", 1280, 720)];

    let ms_ssim = MsSsim::new(5, SsimConfig::default());

    for (name, width, height) in resolutions {
        let reference = create_test_frame(width, height, 0);
        let distorted = add_noise(&reference, 10);
        let pixels = (width * height) as u64;

        group.throughput(Throughput::Elements(pixels));
        group.bench_with_input(BenchmarkId::from_parameter(name), &pixels, |b, _| {
            b.iter(|| ms_ssim.calculate(black_box(&reference), black_box(&distorted)));
        });
    }

    group.finish();
}

// ============================================================================
// VMAF Benchmarks
// ============================================================================

fn bench_vmaf(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmaf_approximation");
    group.sample_size(20);

    let resolutions = [("256x256", 256, 256), ("720p", 1280, 720)];

    let vmaf = Vmaf::new(VmafConfig::default());

    for (name, width, height) in resolutions {
        let reference = create_test_frame(width, height, 0);
        let distorted = add_noise(&reference, 10);
        let pixels = (width * height) as u64;

        group.throughput(Throughput::Elements(pixels));
        group.bench_with_input(BenchmarkId::from_parameter(name), &pixels, |b, _| {
            b.iter(|| vmaf.calculate(black_box(&reference), black_box(&distorted)));
        });
    }

    group.finish();
}

// ============================================================================
// Comparison Benchmarks
// ============================================================================

fn bench_all_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_metrics_720p");
    group.sample_size(10);

    let width = 1280;
    let height = 720;

    let reference = create_test_frame(width, height, 0);
    let distorted = add_noise(&reference, 10);

    let psnr = Psnr::new(PsnrConfig::default());
    let ssim = Ssim::new(SsimConfig::default());
    let vmaf = Vmaf::new(VmafConfig::default());

    group.bench_function("psnr", |b| {
        b.iter(|| psnr.calculate(black_box(&reference), black_box(&distorted)));
    });

    group.bench_function("ssim", |b| {
        b.iter(|| ssim.calculate(black_box(&reference), black_box(&distorted)));
    });

    group.bench_function("vmaf_approx", |b| {
        b.iter(|| vmaf.calculate(black_box(&reference), black_box(&distorted)));
    });

    group.finish();
}

// ============================================================================
// Main
// ============================================================================

criterion_group!(
    quality_benches,
    bench_psnr,
    bench_ssim,
    bench_ms_ssim,
    bench_vmaf,
    bench_all_metrics,
);

criterion_main!(quality_benches);
