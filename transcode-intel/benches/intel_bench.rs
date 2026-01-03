//! Content intelligence benchmarks.
//!
//! Benchmarks for scene detection, content classification, and video analysis.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use transcode_intel::{
    ContentClassifier, DetectionMethod, Frame, SceneConfig, SceneDetector, VideoAnalyzer,
};

/// Create a test frame with a uniform color.
fn create_uniform_frame(width: u32, height: u32, value: u8) -> Frame {
    let data = vec![value; (width * height * 3) as usize];
    Frame::new(data, width, height, 3)
}

/// Create a test frame with a gradient pattern.
fn create_gradient_frame(width: u32, height: u32, offset: u8) -> Frame {
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

/// Create a checkerboard frame.
fn create_checkerboard_frame(width: u32, height: u32, block_size: u32) -> Frame {
    let mut data = vec![0u8; (width * height * 3) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let block_x = x / block_size;
            let block_y = y / block_size;
            let val = if (block_x + block_y) % 2 == 0 { 255 } else { 0 };
            data[idx] = val;
            data[idx + 1] = val;
            data[idx + 2] = val;
        }
    }

    Frame::new(data, width, height, 3)
}

/// Create a test video sequence with scene changes.
fn create_video_sequence(frame_count: usize, width: u32, height: u32) -> Vec<Frame> {
    let mut frames = Vec::with_capacity(frame_count);
    let frames_per_scene = frame_count / 4;

    for i in 0..frame_count {
        let scene = i / frames_per_scene;
        let frame = match scene % 4 {
            0 => create_uniform_frame(width, height, 30),
            1 => create_uniform_frame(width, height, 200),
            2 => create_gradient_frame(width, height, (i % 256) as u8),
            _ => create_checkerboard_frame(width, height, 8),
        };
        frames.push(frame);
    }

    frames
}

// ============================================================================
// Scene Detection Benchmarks
// ============================================================================

fn bench_scene_detection_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("scene_detection_method");

    let width = 256;
    let height = 256;
    let frames = create_video_sequence(100, width, height);

    let methods = [
        ("histogram", DetectionMethod::Histogram),
        ("content_diff", DetectionMethod::ContentDiff),
        ("edge", DetectionMethod::Edge),
        ("combined", DetectionMethod::Combined),
    ];

    for (name, method) in methods {
        group.throughput(Throughput::Elements(frames.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(name), &frames, |b, frames| {
            b.iter(|| {
                let config = SceneConfig::default()
                    .with_method(method)
                    .with_threshold(0.3);
                let mut detector = SceneDetector::new(config);

                for frame in frames {
                    let _ = detector.process_frame(black_box(frame));
                }
            });
        });
    }

    group.finish();
}

fn bench_scene_detection_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("scene_detection_resolution");
    group.sample_size(20);

    let resolutions = [("256x256", 256, 256), ("720p", 1280, 720), ("1080p", 1920, 1080)];

    for (name, width, height) in resolutions {
        let frames = create_video_sequence(30, width, height);
        let pixels = (width * height) as u64;

        group.throughput(Throughput::Elements(pixels));
        group.bench_with_input(BenchmarkId::from_parameter(name), &frames, |b, frames| {
            b.iter(|| {
                let mut detector = SceneDetector::default();
                for frame in frames {
                    let _ = detector.process_frame(black_box(frame));
                }
            });
        });
    }

    group.finish();
}

// ============================================================================
// Content Classification Benchmarks
// ============================================================================

fn bench_content_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_classification");
    group.sample_size(50);

    let resolutions = [("256x256", 256, 256), ("720p", 1280, 720)];

    for (name, width, height) in resolutions {
        let frame = create_gradient_frame(width, height, 0);
        let pixels = (width * height) as u64;

        group.throughput(Throughput::Elements(pixels));
        group.bench_with_input(BenchmarkId::from_parameter(name), &frame, |b, frame| {
            b.iter(|| {
                let mut classifier = ContentClassifier::new();
                classifier.classify(black_box(frame))
            });
        });
    }

    group.finish();
}

fn bench_content_classification_motion(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_classification_motion");
    group.sample_size(20);

    let width = 256;
    let height = 256;

    // Create two frames for motion detection
    let frame1 = create_gradient_frame(width, height, 0);
    let frame2 = create_gradient_frame(width, height, 20);

    group.throughput(Throughput::Elements(2));
    group.bench_function("motion_detection", |b| {
        b.iter(|| {
            let mut classifier = ContentClassifier::new();
            classifier.classify(black_box(&frame1)).unwrap();
            classifier.classify(black_box(&frame2)).unwrap()
        });
    });

    group.finish();
}

// ============================================================================
// Video Analyzer Benchmarks
// ============================================================================

fn bench_video_analyzer(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_analyzer");
    group.sample_size(10);

    let frame_counts = [30, 60, 120];
    let width = 256;
    let height = 256;

    for frame_count in frame_counts {
        let frames = create_video_sequence(frame_count, width, height);

        group.throughput(Throughput::Elements(frame_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_frames", frame_count)),
            &frames,
            |b, frames| {
                b.iter(|| {
                    let mut analyzer = VideoAnalyzer::default();
                    analyzer.analyze_sequence(black_box(frames))
                });
            },
        );
    }

    group.finish();
}

fn bench_video_analyzer_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_analyzer_resolution");
    group.sample_size(10);

    let resolutions = [("256x256", 256, 256), ("720p", 1280, 720)];
    let frame_count = 30;

    for (name, width, height) in resolutions {
        let frames = create_video_sequence(frame_count, width, height);
        let total_pixels = (width * height * frame_count as u32) as u64;

        group.throughput(Throughput::Elements(total_pixels));
        group.bench_with_input(BenchmarkId::from_parameter(name), &frames, |b, frames| {
            b.iter(|| {
                let mut analyzer = VideoAnalyzer::default();
                analyzer.analyze_sequence(black_box(frames))
            });
        });
    }

    group.finish();
}

// ============================================================================
// Combined Benchmarks
// ============================================================================

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_intel_pipeline");
    group.sample_size(10);

    let width = 256;
    let height = 256;
    let frames = create_video_sequence(60, width, height);

    group.throughput(Throughput::Elements(frames.len() as u64));
    group.bench_function("scene_detect_and_classify", |b| {
        b.iter(|| {
            let mut analyzer = VideoAnalyzer::default();
            analyzer.analyze_sequence(black_box(&frames))
        });
    });

    group.finish();
}

// ============================================================================
// Main
// ============================================================================

criterion_group!(
    scene_detection_benches,
    bench_scene_detection_methods,
    bench_scene_detection_resolution,
);

criterion_group!(
    classification_benches,
    bench_content_classification,
    bench_content_classification_motion,
);

criterion_group!(
    analyzer_benches,
    bench_video_analyzer,
    bench_video_analyzer_resolution,
    bench_full_pipeline,
);

criterion_main!(scene_detection_benches, classification_benches, analyzer_benches);
