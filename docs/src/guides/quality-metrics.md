# Quality Metrics

This guide covers video quality assessment using the `transcode-quality` crate.

## Overview

Transcode provides objective quality metrics to measure compression artifacts:

| Metric | Type | Range | Best For |
|--------|------|-------|----------|
| PSNR | Full-reference | 0-∞ dB | Quick comparison |
| SSIM | Full-reference | 0-1 | Structural quality |
| MS-SSIM | Full-reference | 0-1 | Multi-scale quality |
| VMAF | Full-reference | 0-100 | Perceptual quality |

## PSNR (Peak Signal-to-Noise Ratio)

The simplest quality metric, measuring pixel-level differences:

```rust
use transcode_quality::Psnr;

let psnr = Psnr::default();
let result = psnr.calculate(&reference, &distorted)?;

println!("PSNR: {:.2} dB", result.average());
println!("  Y:  {:.2} dB", result.y);
println!("  Cb: {:.2} dB", result.cb);
println!("  Cr: {:.2} dB", result.cr);
```

### Interpreting PSNR

| PSNR (dB) | Quality |
|-----------|---------|
| > 40 | Excellent (indistinguishable) |
| 35-40 | Good |
| 30-35 | Acceptable |
| 25-30 | Poor |
| < 25 | Bad |

## SSIM (Structural Similarity)

Measures structural similarity, closer to human perception:

```rust
use transcode_quality::Ssim;

let ssim = Ssim::default();
let result = ssim.calculate(&reference, &distorted)?;

println!("SSIM: {:.4}", result.score);
```

### Configuration

```rust
let ssim = Ssim::builder()
    .window_size(11)           // Gaussian window size
    .sigma(1.5)                // Gaussian sigma
    .k1(0.01)                  // Stability constant
    .k2(0.03)                  // Stability constant
    .build()?;
```

### Interpreting SSIM

| SSIM | Quality |
|------|---------|
| > 0.98 | Excellent |
| 0.95-0.98 | Good |
| 0.90-0.95 | Acceptable |
| 0.80-0.90 | Poor |
| < 0.80 | Bad |

## MS-SSIM (Multi-Scale SSIM)

SSIM computed at multiple scales for better correlation with perception:

```rust
use transcode_quality::MsSsim;

let ms_ssim = MsSsim::default();
let result = ms_ssim.calculate(&reference, &distorted)?;

println!("MS-SSIM: {:.4}", result.score);
```

### Scale Weights

Default weights for 5 scales:

| Scale | Weight | Resolution |
|-------|--------|------------|
| 1 | 0.0448 | Original |
| 2 | 0.2856 | 1/2 |
| 3 | 0.3001 | 1/4 |
| 4 | 0.2363 | 1/8 |
| 5 | 0.1333 | 1/16 |

## VMAF (Video Multi-Method Assessment Fusion)

Netflix's perceptual quality metric:

```rust
use transcode_quality::Vmaf;

let vmaf = Vmaf::default();
let result = vmaf.calculate(&reference, &distorted)?;

println!("VMAF: {:.2}", result.score);
```

### VMAF Components

```rust
// Access individual features
println!("VIF: {:.4}", result.vif);      // Visual Information Fidelity
println!("DLM: {:.4}", result.dlm);      // Detail Loss Metric
println!("Motion: {:.4}", result.motion); // Temporal complexity
```

### Interpreting VMAF

| VMAF | Quality |
|------|---------|
| > 93 | Excellent |
| 80-93 | Good |
| 70-80 | Acceptable |
| 60-70 | Poor |
| < 60 | Bad |

## Batch Assessment

Process multiple frames efficiently:

```rust
use transcode_quality::{QualityAssessment, MetricType};

let qa = QualityAssessment::builder()
    .metrics(vec![MetricType::Psnr, MetricType::Ssim, MetricType::Vmaf])
    .parallel(true)
    .build()?;

// Assess frame sequence
let report = qa.assess_sequence(&reference_frames, &distorted_frames)?;

println!("Average PSNR: {:.2} dB", report.psnr.mean());
println!("Average SSIM: {:.4}", report.ssim.mean());
println!("Average VMAF: {:.2}", report.vmaf.mean());
println!("Min VMAF: {:.2}", report.vmaf.min());
```

## Quality Report

Generate a comprehensive report:

```rust
use transcode_quality::QualityReport;

let report = QualityReport::generate(&reference, &distorted)?;

// Print summary
println!("{}", report.summary());

// Export to JSON
let json = report.to_json()?;

// Export to CSV
report.to_csv("quality_report.csv")?;
```

## Per-Frame Analysis

Analyze quality per frame for debugging:

```rust
use transcode_quality::PerFrameAnalysis;

let analysis = PerFrameAnalysis::new(&reference_frames, &distorted_frames)?;

for (i, frame_quality) in analysis.iter().enumerate() {
    if frame_quality.vmaf < 70.0 {
        println!("Frame {} has low quality: VMAF={:.2}", i, frame_quality.vmaf);
    }
}

// Find worst frames
let worst = analysis.worst_frames(5);
for (frame_idx, quality) in worst {
    println!("Frame {}: VMAF={:.2}", frame_idx, quality.vmaf);
}
```

## Choosing a Metric

```
Quick A/B comparison?           → PSNR
Structural quality assessment?  → SSIM
Publication/research?           → MS-SSIM
Production QC?                  → VMAF
```

## Performance

| Metric | 1080p Frame | Notes |
|--------|-------------|-------|
| PSNR | ~2ms | Simple pixel math |
| SSIM | ~15ms | Gaussian filtering |
| MS-SSIM | ~50ms | Multi-scale |
| VMAF | ~100ms | Multiple features |

## Best Practices

1. **Use VMAF for final QC** - Best correlation with human perception
2. **Use PSNR for quick checks** - Fast and easy to interpret
3. **Compare same content** - Metrics vary by content type
4. **Consider temporal** - Single frame metrics miss temporal artifacts
