# transcode-deinterlace

Deinterlacing filters for video processing in the transcode library.

## Overview

This crate provides multiple deinterlacing algorithms for converting interlaced video content to progressive format. Interlaced video contains two fields per frame captured at slightly different times, which causes artifacts on modern progressive displays. This crate offers solutions ranging from simple and fast (Bob) to sophisticated motion-adaptive filters (YADIF, BWDIF).

## Deinterlacing Algorithms

| Algorithm | Quality | Speed  | Temporal | Description                          |
|-----------|---------|--------|----------|--------------------------------------|
| Bob       | Low     | Fast   | No       | Simple line doubling                 |
| Linear    | Medium  | Fast   | No       | Linear interpolation                 |
| Bicubic   | Medium  | Medium | No       | Bicubic (Catmull-Rom) interpolation  |
| YADIF     | High    | Medium | Yes      | Motion-adaptive, 3-frame temporal    |
| BWDIF     | Highest | Slow   | Yes      | Motion-adaptive, 5-frame temporal    |

## Key Types

### Deinterlacers
- `BobDeinterlacer` - Simple line-doubling deinterlacer
- `LinearDeinterlacer` - Linear interpolation deinterlacer
- `BicubicDeinterlacer` - Bicubic interpolation deinterlacer
- `YadifDeinterlacer` - Yet Another Deinterlacing Filter (motion-adaptive)
- `BwdifDeinterlacer` - Bob-Weave Deinterlacing Filter (advanced motion-adaptive)

### Detection & Analysis
- `InterlaceDetector` - Detects interlaced vs progressive content
- `FieldOrderDetector` - Determines field order (TFF/BFF)
- `InverseTelecine` - Removes 3:2/2:2 pulldown from telecined content
- `ContentType` - Detection result (Progressive, Interlaced, Telecine, Mixed)

### Configuration
- `FieldOrder` - TopFieldFirst or BottomFieldFirst
- `YadifMode` / `BwdifMode` - Frame or Field output modes
- `TelecinePattern` - Pulldown32, Pulldown22, Pulldown23, Variable

### Errors
- `DeinterlaceError` - Comprehensive error type for all operations

## Usage Examples

### Basic Deinterlacing with YADIF

```rust
use transcode_deinterlace::{YadifDeinterlacer, Deinterlacer};
use transcode_core::Frame;

let mut yadif = YadifDeinterlacer::new();

// Push frames (temporal filters need multiple frames)
let output1 = yadif.process(frame1)?;  // May return empty
let output2 = yadif.process(frame2)?;  // Returns deinterlaced frames

// Flush remaining frames when done
let remaining = yadif.flush()?;
```

### Interlace Detection

```rust
use transcode_deinterlace::{InterlaceDetector, ContentType, FieldOrder};

let mut detector = InterlaceDetector::new();

// Analyze multiple frames
for frame in frames {
    detector.analyze_frame(&frame)?;
}

match detector.get_content_type() {
    ContentType::Progressive => println!("Progressive content"),
    ContentType::Interlaced { field_order } => println!("Interlaced: {:?}", field_order),
    ContentType::Telecine { pattern, .. } => println!("Telecine detected"),
    _ => println!("Unknown or mixed"),
}
```

### Using the Factory Function

```rust
use transcode_deinterlace::{create_deinterlacer, DeinterlaceAlgorithm, Deinterlacer};

let mut deint = create_deinterlacer(DeinterlaceAlgorithm::Yadif);
let output = deint.process(frame)?;
```

## Algorithm Selection Guide

- **Bob**: Real-time preview, speed-critical applications
- **Linear**: Good balance for spatial-only processing
- **YADIF**: Recommended default - good quality/speed tradeoff
- **BWDIF**: Best quality for final output, slower processing

## Feature Flags

- `simd` - Enable SIMD optimizations (SSE2/AVX2/NEON)

## Supported Pixel Formats

- YUV420P, YUV422P, YUV444P (8-bit and 10-bit variants)
- Gray8, Gray16
