//! Codec benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use transcode_core::frame::{Frame, PixelFormat};
use transcode_core::timestamp::TimeBase;

fn generate_test_frame(width: u32, height: u32) -> Frame {
    let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);

    // Fill with pattern
    if let Some(y_plane) = frame.plane_mut(0) {
        for (i, pixel) in y_plane.iter_mut().enumerate() {
            *pixel = (i % 256) as u8;
        }
    }
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

fn bench_frame_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_creation");

    for (width, height, name) in &[
        (1920u32, 1080u32, "1080p"),
        (1280, 720, "720p"),
        (3840, 2160, "4K"),
    ] {
        let pixels = (*width as u64) * (*height as u64);
        group.throughput(Throughput::Elements(pixels));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(*width, *height),
            |b, &(w, h)| {
                b.iter(|| {
                    let frame = Frame::new(w, h, PixelFormat::Yuv420p, TimeBase::MPEG);
                    black_box(frame)
                });
            },
        );
    }

    group.finish();
}

fn bench_frame_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_copy");

    for (width, height, name) in &[
        (1920u32, 1080u32, "1080p"),
        (1280, 720, "720p"),
    ] {
        let frame = generate_test_frame(*width, *height);
        let pixels = (*width as u64) * (*height as u64);
        group.throughput(Throughput::Elements(pixels));

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &frame,
            |b, frame| {
                b.iter(|| {
                    let cloned = frame.clone();
                    black_box(cloned)
                });
            },
        );
    }

    group.finish();
}

fn bench_plane_access(c: &mut Criterion) {
    let frame = generate_test_frame(1920, 1080);

    c.bench_function("plane_access_y", |b| {
        b.iter(|| {
            let plane = frame.plane(0).unwrap();
            black_box(plane.len())
        });
    });
}

fn bench_pixel_iteration(c: &mut Criterion) {
    let frame = generate_test_frame(1920, 1080);

    c.bench_function("pixel_iteration_1080p", |b| {
        b.iter(|| {
            let plane = frame.plane(0).unwrap();
            let sum: u64 = plane.iter().map(|&p| p as u64).sum();
            black_box(sum)
        });
    });
}

criterion_group!(
    benches,
    bench_frame_creation,
    bench_frame_copy,
    bench_plane_access,
    bench_pixel_iteration,
);

criterion_main!(benches);
