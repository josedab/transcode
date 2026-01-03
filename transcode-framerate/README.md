# transcode-framerate

Frame rate conversion library for the Transcode project.

## Overview

This crate provides comprehensive frame rate conversion functionality for video processing, including simple conversions, frame blending, motion-compensated interpolation, and telecine pattern handling.

## Features

- **Drop/Duplicate**: Simple frame dropping or duplication for rate conversion
- **Frame Blending**: Weighted averaging with multiple blend modes (Linear, EaseIn, EaseOut, SmoothStep, Cosine)
- **Motion-Compensated Interpolation**: Smooth frame interpolation using motion estimation
- **Telecine Patterns**: 3:2 pulldown, 2:3 pulldown, 2:2 pulldown, Euro pulldown
- **Inverse Telecine (IVTC)**: Automatic pattern detection and film frame recovery
- **Variable Frame Rate (VFR)**: Convert VFR content to constant frame rate (CFR)

## Key Types

| Type | Description |
|------|-------------|
| `FrameRateConverter` | Main converter with configurable methods |
| `StandardFrameRate` | Common rates (Film24, NtscFilm, Pal25, Ntsc30, Hfr60, etc.) |
| `ConversionMethod` | DropDuplicate, Blend, MotionCompensated, Telecine, InverseTelecine |
| `FrameBlender` | Frame blending with configurable modes and dithering |
| `FrameInterpolator` | Interpolation with Nearest, Linear, Cubic, MotionCompensated methods |
| `MotionEstimator` | Block-based motion estimation (Diamond, Hexagon, ThreeStep, Full search) |
| `TelecineApplier` | Apply telecine patterns (24->30 fps) |
| `InverseTelecine` | Remove telecine patterns (30->24 fps) |
| `VfrHandler` | Handle variable frame rate input |

## Usage

### Basic Frame Rate Conversion

```rust
use transcode_framerate::{FrameRateConverter, StandardFrameRate};

// Create converter for 24->30 fps
let mut converter = FrameRateConverter::standard(
    StandardFrameRate::Film24,
    StandardFrameRate::Ntsc30,
);

// Process frames
for frame in input_frames {
    let output_frames = converter.process(&frame)?;
    for output in output_frames {
        // Handle output frames
    }
}

// Flush remaining frames
let remaining = converter.flush();
```

### Frame Blending

```rust
use transcode_framerate::{FrameBlender, BlendConfig, BlendMode};

let blender = FrameBlender::with_config(BlendConfig {
    mode: BlendMode::SmoothStep,
    dithering: true,
    ..Default::default()
});

// Blend two frames with 50% weight
let blended = blender.blend(&frame1, &frame2, 0.5)?;
```

### Motion-Compensated Interpolation

```rust
use transcode_framerate::{FrameInterpolator, InterpolationConfig, InterpolationMethod};

let interpolator = FrameInterpolator::with_config(InterpolationConfig {
    method: InterpolationMethod::MotionCompensated,
    ..Default::default()
});

// Interpolate frame at 50% position between two frames
let interpolated = interpolator.interpolate(&frame1, &frame2, 0.5)?;
```

### Telecine Handling

```rust
use transcode_framerate::{TelecineApplier, InverseTelecine, TelecinePattern};

// Apply 3:2 pulldown (24->30 fps)
let mut telecine = TelecineApplier::new(TelecinePattern::Pulldown32);
let output_frames = telecine.process(&input_frame)?;

// Remove telecine (30->24 fps)
let mut ivtc = InverseTelecine::with_pattern(TelecinePattern::Pulldown32);
let recovered = ivtc.process(&telecined_frame)?;
```

### Convenience Functions

```rust
use transcode_framerate::{create_converter, blend_frames, interpolate_frames};
use transcode_core::Rational;

// Auto-select best conversion method
let converter = create_converter(Rational::new(24, 1), Rational::new(30, 1));

// Quick frame blending
let blended = blend_frames(&frame1, &frame2, 0.5)?;

// Quick interpolation
let interpolated = interpolate_frames(&frame1, &frame2, 0.5, InterpolationMethod::Linear)?;
```

## License

See the workspace root for license information.
