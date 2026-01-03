//! Filter benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use transcode_core::frame::{Frame, PixelFormat};
use transcode_core::timestamp::TimeBase;

fn generate_test_frame(width: u32, height: u32) -> Frame {
    let mut frame = Frame::new(width, height, PixelFormat::Yuv420p, TimeBase::MPEG);

    if let Some(y_plane) = frame.plane_mut(0) {
        for (i, pixel) in y_plane.iter_mut().enumerate() {
            *pixel = ((i * 7) % 256) as u8;
        }
    }

    frame
}

fn apply_brightness(frame: &mut Frame, adjustment: i16) {
    if let Some(y_plane) = frame.plane_mut(0) {
        for pixel in y_plane.iter_mut() {
            let new_val = (*pixel as i16 + adjustment).clamp(0, 255);
            *pixel = new_val as u8;
        }
    }
}

fn apply_contrast(frame: &mut Frame, factor: f32) {
    if let Some(y_plane) = frame.plane_mut(0) {
        for pixel in y_plane.iter_mut() {
            let centered = *pixel as f32 - 128.0;
            let adjusted = centered * factor + 128.0;
            *pixel = adjusted.clamp(0.0, 255.0) as u8;
        }
    }
}

fn apply_threshold(frame: &mut Frame, threshold: u8) {
    if let Some(y_plane) = frame.plane_mut(0) {
        for pixel in y_plane.iter_mut() {
            *pixel = if *pixel > threshold { 255 } else { 0 };
        }
    }
}

fn bench_brightness(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_brightness");

    for (width, height, name) in &[(1920u32, 1080u32, "1080p"), (1280, 720, "720p")] {
        let pixels = (*width as u64) * (*height as u64);
        group.throughput(Throughput::Elements(pixels));

        group.bench_function(*name, |b| {
            b.iter_batched(
                || generate_test_frame(*width, *height),
                |mut frame| {
                    apply_brightness(&mut frame, 20);
                    black_box(frame)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_contrast(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_contrast");

    for (width, height, name) in &[(1920u32, 1080u32, "1080p"), (1280, 720, "720p")] {
        let pixels = (*width as u64) * (*height as u64);
        group.throughput(Throughput::Elements(pixels));

        group.bench_function(*name, |b| {
            b.iter_batched(
                || generate_test_frame(*width, *height),
                |mut frame| {
                    apply_contrast(&mut frame, 1.2);
                    black_box(frame)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_threshold");

    let (width, height) = (1920, 1080);
    let pixels = (width as u64) * (height as u64);
    group.throughput(Throughput::Elements(pixels));

    group.bench_function("1080p", |b| {
        b.iter_batched(
            || generate_test_frame(width, height),
            |mut frame| {
                apply_threshold(&mut frame, 128);
                black_box(frame)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_brightness, bench_contrast, bench_threshold);
criterion_main!(benches);
