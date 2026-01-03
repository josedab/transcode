# transcode-quality

Perceptual video quality metrics for the transcode project.

## Overview

This crate provides implementations of common video quality metrics for comparing reference and distorted video frames:

- **PSNR** - Peak Signal-to-Noise Ratio (fast, pixel-based)
- **SSIM** - Structural Similarity Index (perceptual, structure-based)
- **MS-SSIM** - Multi-Scale SSIM (better human perception correlation)
- **VMAF** - Video Multi-Method Assessment Fusion (Netflix's ML-based metric)

## Quality Metrics

| Metric  | Range     | Speed  | Human Perception Correlation |
|---------|-----------|--------|------------------------------|
| PSNR    | 0-inf dB  | Fast   | Low                          |
| SSIM    | -1 to 1   | Medium | Medium                       |
| MS-SSIM | -1 to 1   | Slow   | High                         |
| VMAF    | 0-100     | Slow   | Very High                    |

## Key Types

- `Frame` - Video frame representation (RGB/YUV pixel data)
- `QualityAssessment` - Unified calculator for all metrics
- `QualityReport` - Complete quality report with ratings
- `BatchQualityAssessment` - Aggregate scores across video sequences
- `Psnr`, `Ssim`, `MsSsim`, `Vmaf` - Individual metric calculators

## Usage

### Quick Functions

```rust
use transcode_quality::{psnr, ssim, ms_ssim, vmaf, Frame};

let reference = Frame::new(ref_data, 1920, 1080, 3);
let distorted = Frame::new(dist_data, 1920, 1080, 3);

let psnr_score = psnr(&reference, &distorted)?;
let ssim_score = ssim(&reference, &distorted)?;
let vmaf_score = vmaf(&reference, &distorted)?;
```

### Unified Assessment

```rust
use transcode_quality::{QualityAssessment, QualityConfig, QualityMetrics};

let config = QualityConfig {
    metrics: QualityMetrics::all(),
    bit_depth: 8,
    ..Default::default()
};

let qa = QualityAssessment::new(config);
let report = qa.assess(&reference, &distorted)?;

println!("Overall: {:.1} ({})", report.overall_score(), report.rating());
```

### Fast Assessment (PSNR + SSIM only)

```rust
let qa = QualityAssessment::default();
let report = qa.assess_fast(&reference, &distorted)?;
```

### Batch Processing

```rust
use transcode_quality::BatchQualityAssessment;

let mut batch = BatchQualityAssessment::new();
for (ref_frame, dist_frame) in frames {
    batch.add(qa.assess(&ref_frame, &dist_frame)?);
}

println!("Average VMAF: {:?}", batch.average_vmaf());
println!("{}", batch.summary());
```

## Features

- `vmaf` - Enable full VMAF via libvmaf (requires `vmaf-sys`)

Without the `vmaf` feature, a simplified VMAF approximation using VIF and DLM is used.

## Dependencies

- `ndarray` - Array operations
- `rayon` - Parallel processing
- `vmaf-sys` (optional) - Netflix libvmaf bindings

## Documentation

See the [main transcode documentation](../README.md) for integration with other crates.
