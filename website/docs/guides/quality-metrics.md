---
sidebar_position: 5
title: Quality Metrics
description: Measure video quality with PSNR, SSIM, and VMAF
---

# Quality Metrics

Transcode provides comprehensive quality assessment tools to measure and compare video quality.

## Overview

The `transcode-quality` crate implements:

- **PSNR** (Peak Signal-to-Noise Ratio) - Simple mathematical comparison
- **SSIM** (Structural Similarity Index) - Perceptual quality metric
- **MS-SSIM** (Multi-Scale SSIM) - SSIM at multiple resolutions
- **VMAF** (Video Multimethod Assessment Fusion) - Netflix's perceptual metric

## Setup

```toml
[dependencies]
transcode = { version = "1.0", features = ["quality"] }
transcode-quality = "1.0"
```

## PSNR

PSNR measures the ratio between maximum signal power and noise power. Higher is better.

### Basic PSNR Calculation

```rust
use transcode_quality::{Psnr, PsnrResult};

let psnr = Psnr::default();
let result = psnr.calculate(&reference_frame, &compressed_frame)?;

println!("PSNR: {:.2} dB", result.average());
println!("  Y:  {:.2} dB", result.y);
println!("  U:  {:.2} dB", result.u);
println!("  V:  {:.2} dB", result.v);
```

### PSNR Interpretation

| PSNR (dB) | Quality |
|-----------|---------|
| > 40 | Excellent (nearly lossless) |
| 35-40 | Very good |
| 30-35 | Good |
| 25-30 | Fair |
| < 25 | Poor |

### Weighted PSNR

Weight channels differently (Y is more important for perceived quality):

```rust
let psnr = Psnr::weighted(0.8, 0.1, 0.1);  // Y=80%, U=10%, V=10%
let result = psnr.calculate(&reference, &compressed)?;
```

## SSIM

SSIM models human perception of image quality, considering luminance, contrast, and structure.

### Basic SSIM Calculation

```rust
use transcode_quality::{Ssim, SsimResult};

let ssim = Ssim::default();
let result = ssim.calculate(&reference_frame, &compressed_frame)?;

println!("SSIM: {:.4}", result.score);  // 0.0 to 1.0
```

### SSIM Interpretation

| SSIM | Quality |
|------|---------|
| > 0.98 | Excellent |
| 0.95-0.98 | Very good |
| 0.90-0.95 | Good |
| 0.80-0.90 | Fair |
| < 0.80 | Poor |

### MS-SSIM (Multi-Scale)

Evaluates quality at multiple scales for better correlation with perception:

```rust
use transcode_quality::{MsSsim, MsSsimConfig};

let config = MsSsimConfig {
    scales: 5,  // Number of scales to evaluate
    ..Default::default()
};

let ms_ssim = MsSsim::new(config);
let result = ms_ssim.calculate(&reference, &compressed)?;

println!("MS-SSIM: {:.4}", result.score);
```

## VMAF

VMAF is Netflix's machine-learning-based metric, trained on subjective quality scores.

### Basic VMAF Calculation

```rust
use transcode_quality::{Vmaf, VmafConfig};

let vmaf = Vmaf::default();
let score = vmaf.calculate(&reference_frame, &compressed_frame)?;

println!("VMAF: {:.2}", score);  // 0 to 100
```

### VMAF Interpretation

| VMAF | Quality |
|------|---------|
| > 93 | Excellent |
| 80-93 | Good |
| 60-80 | Fair |
| 40-60 | Poor |
| < 40 | Bad |

### Phone Model

For mobile viewing (smaller screens are more forgiving):

```rust
let vmaf = Vmaf::with_model(VmafModel::Phone);
let score = vmaf.calculate(&reference, &compressed)?;
```

## Comparing Entire Videos

Assess quality across a full video:

```rust
use transcode_quality::{QualityAssessment, QualityReport};

let qa = QualityAssessment::new();
let report = qa.assess_video("reference.mp4", "compressed.mp4")?;

println!("Overall Quality Report:");
println!("  PSNR: {:.2} dB (min: {:.2}, max: {:.2})",
    report.psnr.average, report.psnr.min, report.psnr.max);
println!("  SSIM: {:.4} (min: {:.4}, max: {:.4})",
    report.ssim.average, report.ssim.min, report.ssim.max);
println!("  VMAF: {:.2} (min: {:.2}, max: {:.2})",
    report.vmaf.average, report.vmaf.min, report.vmaf.max);

// Frame-by-frame scores
for (i, frame_score) in report.frame_scores.iter().enumerate() {
    println!("Frame {}: PSNR={:.2}, SSIM={:.4}, VMAF={:.2}",
        i, frame_score.psnr, frame_score.ssim, frame_score.vmaf);
}
```

## Quality During Transcoding

Monitor quality during the transcoding process:

```rust
use transcode::{Transcoder, TranscodeOptions};
use transcode_quality::{QualityMonitor, QualityCallback};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .video_bitrate(5_000_000);

let mut transcoder = Transcoder::new(options)?;

// Add quality monitoring
let monitor = QualityMonitor::new()
    .metrics(&[Metric::Psnr, Metric::Ssim])
    .on_frame(|frame_num, scores| {
        if scores.ssim < 0.90 {
            println!("Warning: Low quality at frame {}: SSIM={:.4}",
                frame_num, scores.ssim);
        }
    });

transcoder = transcoder.add_monitor(monitor);
transcoder.run()?;
```

## Finding Optimal Bitrate

Use quality metrics to find the right bitrate:

```rust
use transcode_quality::{BitrateAnalyzer, QualityTarget};

let analyzer = BitrateAnalyzer::new("input.mp4");

// Find bitrate for target VMAF
let bitrate = analyzer.find_bitrate_for_quality(
    QualityTarget::Vmaf(93.0),
    "h264"
)?;

println!("Recommended bitrate for VMAF 93: {} kbps", bitrate / 1000);
```

### Quality Ladder Generation

Find optimal bitrates for ABR streaming:

```rust
use transcode_quality::{QualityLadder, Resolution};

let ladder = QualityLadder::analyze("input.mp4", &[
    Resolution::new(1920, 1080),
    Resolution::new(1280, 720),
    Resolution::new(854, 480),
    Resolution::new(640, 360),
])?;

for rung in ladder.rungs() {
    println!("{}p: {} kbps (VMAF: {:.1})",
        rung.height, rung.bitrate / 1000, rung.vmaf);
}
```

## Batch Quality Assessment

Compare multiple encodes:

```rust
use transcode_quality::{BatchAssessment, EncodeVariant};

let variants = vec![
    EncodeVariant::new("crf_18.mp4", "CRF 18"),
    EncodeVariant::new("crf_23.mp4", "CRF 23"),
    EncodeVariant::new("crf_28.mp4", "CRF 28"),
];

let batch = BatchAssessment::new("reference.mp4", variants);
let results = batch.assess()?;

println!("Quality Comparison:");
println!("{:<15} {:>10} {:>10} {:>10} {:>12}",
    "Variant", "PSNR", "SSIM", "VMAF", "File Size");

for result in results {
    println!("{:<15} {:>10.2} {:>10.4} {:>10.2} {:>10.2} MB",
        result.name,
        result.psnr,
        result.ssim,
        result.vmaf,
        result.file_size as f64 / 1_000_000.0);
}
```

## Visualization

Generate quality graphs:

```rust
use transcode_quality::{QualityReport, PlotConfig};

let report = qa.assess_video("reference.mp4", "compressed.mp4")?;

// Generate VMAF timeline plot
report.plot_vmaf("vmaf_timeline.png", PlotConfig::default())?;

// Generate all metrics comparison
report.plot_all("quality_comparison.png", PlotConfig {
    width: 1200,
    height: 600,
    ..Default::default()
})?;
```

## CLI Usage

```bash
# Compare two videos
transcode-quality reference.mp4 compressed.mp4

# Specific metrics only
transcode-quality reference.mp4 compressed.mp4 --metrics psnr,ssim

# Output JSON
transcode-quality reference.mp4 compressed.mp4 --json > report.json

# Frame-by-frame CSV
transcode-quality reference.mp4 compressed.mp4 --csv > frames.csv
```

## Performance Tips

### Subsample for Speed

For quick estimates, sample frames:

```rust
let qa = QualityAssessment::new()
    .sample_interval(10);  // Every 10th frame

let report = qa.assess_video("reference.mp4", "compressed.mp4")?;
```

### GPU Acceleration

Use GPU for faster VMAF calculation:

```rust
let vmaf = Vmaf::new()
    .use_gpu(true);
```

### Parallel Processing

```rust
let qa = QualityAssessment::new()
    .threads(num_cpus::get());
```

## Best Practices

1. **Use VMAF** as the primary metric for perceptual quality
2. **PSNR** is useful for quick comparisons but doesn't correlate well with perception
3. **SSIM** is a good balance between speed and perceptual accuracy
4. **Always compare at the same resolution** - scale reference if needed
5. **Consider viewing conditions** - use phone model for mobile content
6. **Sample frames** for quick iteration, full analysis for final quality checks

## Next Steps

- [Basic Transcoding](/docs/guides/basic-transcoding) - Apply quality targets
- [Streaming Output](/docs/guides/streaming-output) - Quality ladders for ABR
- [Configuration Reference](/docs/reference/configuration) - Quality settings
