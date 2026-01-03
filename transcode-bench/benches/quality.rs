//! Quality metrics benchmarks
//!
//! Benchmarks for PSNR, SSIM, MS-SSIM, and VMAF calculations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use transcode_quality::{psnr, ssim, ms_ssim, vmaf, Frame, Psnr, PsnrConfig, Ssim, SsimConfig};

fn generate_test_frame(width: u32, height: u32, value: u8) -> Frame {
    let data = vec![value; (width * height * 3) as usize];
    Frame::new(data, width, height, 3)
}

fn generate_noisy_frame(width: u32, height: u32, base: u8, noise_level: u8) -> Frame {
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    for i in 0..(width * height * 3) {
        let noise = ((i as u32 * 7) % (noise_level as u32 * 2)) as i16 - noise_level as i16;
        let val = (base as i16 + noise).clamp(0, 255) as u8;
        data.push(val);
    }
    Frame::new(data, width, height, 3)
}

fn bench_psnr(c: &mut Criterion) {
    let mut group = c.benchmark_group("psnr");

    for (width, height, name) in &[
        (1920u32, 1080u32, "1080p"),
        (1280, 720, "720p"),
        (640, 480, "480p"),
    ] {
        let pixels = (*width as u64) * (*height as u64);
        group.throughput(Throughput::Elements(pixels));

        let reference = generate_test_frame(*width, *height, 128);
        let distorted = generate_noisy_frame(*width, *height, 128, 10);

        group.bench_with_input(BenchmarkId::from_parameter(name), &(&reference, &distorted), |b, (r, d)| {
            b.iter(|| {
                let result = psnr(r, d).unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_psnr_calculator(c: &mut Criterion) {
    let mut group = c.benchmark_group("psnr_calculator");

    let reference = generate_test_frame(1920, 1080, 128);
    let distorted = generate_noisy_frame(1920, 1080, 128, 10);
    let calc = Psnr::new(PsnrConfig { bit_depth: 8 });

    group.throughput(Throughput::Elements(1920 * 1080));

    group.bench_function("1080p_reuse_calculator", |b| {
        b.iter(|| {
            let result = calc.calculate(&reference, &distorted).unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_ssim(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssim");

    for (width, height, name) in &[
        (1920u32, 1080u32, "1080p"),
        (1280, 720, "720p"),
        (640, 480, "480p"),
    ] {
        let pixels = (*width as u64) * (*height as u64);
        group.throughput(Throughput::Elements(pixels));

        let reference = generate_test_frame(*width, *height, 128);
        let distorted = generate_noisy_frame(*width, *height, 128, 10);

        group.bench_with_input(BenchmarkId::from_parameter(name), &(&reference, &distorted), |b, (r, d)| {
            b.iter(|| {
                let result = ssim(r, d).unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_ssim_calculator(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssim_calculator");

    let reference = generate_test_frame(1920, 1080, 128);
    let distorted = generate_noisy_frame(1920, 1080, 128, 10);
    let calc = Ssim::new(SsimConfig::default());

    group.throughput(Throughput::Elements(1920 * 1080));

    group.bench_function("1080p_reuse_calculator", |b| {
        b.iter(|| {
            let result = calc.calculate(&reference, &distorted).unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_ms_ssim(c: &mut Criterion) {
    let mut group = c.benchmark_group("ms_ssim");
    group.sample_size(50); // MS-SSIM is slower, use fewer samples

    for (width, height, name) in &[
        (1280u32, 720u32, "720p"),
        (640, 480, "480p"),
    ] {
        let pixels = (*width as u64) * (*height as u64);
        group.throughput(Throughput::Elements(pixels));

        let reference = generate_test_frame(*width, *height, 128);
        let distorted = generate_noisy_frame(*width, *height, 128, 10);

        group.bench_with_input(BenchmarkId::from_parameter(name), &(&reference, &distorted), |b, (r, d)| {
            b.iter(|| {
                let result = ms_ssim(r, d).unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_vmaf(c: &mut Criterion) {
    let mut group = c.benchmark_group("vmaf");
    group.sample_size(30); // VMAF is slow, use fewer samples

    for (width, height, name) in &[
        (640u32, 480u32, "480p"),
        (320, 240, "240p"),
    ] {
        let pixels = (*width as u64) * (*height as u64);
        group.throughput(Throughput::Elements(pixels));

        let reference = generate_test_frame(*width, *height, 128);
        let distorted = generate_noisy_frame(*width, *height, 128, 10);

        group.bench_with_input(BenchmarkId::from_parameter(name), &(&reference, &distorted), |b, (r, d)| {
            b.iter(|| {
                let result = vmaf(r, d).unwrap();
                black_box(result)
            });
        });
    }

    group.finish();
}

fn bench_identical_frames(c: &mut Criterion) {
    let mut group = c.benchmark_group("identical_frames");

    let frame = generate_test_frame(1920, 1080, 128);
    group.throughput(Throughput::Elements(1920 * 1080));

    group.bench_function("psnr_identical", |b| {
        b.iter(|| {
            let result = psnr(&frame, &frame).unwrap();
            black_box(result)
        });
    });

    group.bench_function("ssim_identical", |b| {
        b.iter(|| {
            let result = ssim(&frame, &frame).unwrap();
            black_box(result)
        });
    });

    group.finish();
}

fn bench_frame_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_frame_creation");

    for (width, height, name) in &[
        (1920u32, 1080u32, "1080p"),
        (1280, 720, "720p"),
    ] {
        let pixels = (*width as u64) * (*height as u64);
        group.throughput(Throughput::Elements(pixels));

        group.bench_with_input(BenchmarkId::from_parameter(name), &(*width, *height), |b, &(w, h)| {
            b.iter(|| {
                let frame = generate_test_frame(w, h, 128);
                black_box(frame)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_psnr,
    bench_psnr_calculator,
    bench_ssim,
    bench_ssim_calculator,
    bench_ms_ssim,
    bench_vmaf,
    bench_identical_frames,
    bench_frame_creation,
);

criterion_main!(benches);
