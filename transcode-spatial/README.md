# transcode-spatial

Spatial audio processing for the transcode library.

## Overview

This crate provides comprehensive spatial audio processing capabilities including channel layouts, ambisonics, binaural rendering, object-based audio, and Dolby Atmos support.

## Features

- **Channel Layouts**: Standard layouts (stereo, 5.1, 7.1, 7.1.4 Atmos)
- **Ambisonics**: First-order (FOA) and higher-order (HOA) ambisonics
- **Binaural Rendering**: HRTF-based headphone rendering with head tracking
- **Object-Based Audio**: 3D audio objects with metadata
- **Dolby Atmos**: ADM parsing, object rendering, bed handling
- **Downmixing**: Configurable 7.1.4 to stereo conversion chains
- **Room Simulation**: Basic room acoustics and reverb

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-spatial = { path = "../transcode-spatial" }
```

### Basic Configuration

```rust
use transcode_spatial::{SpatialConfig, RenderMode, StandardLayout};

let config = SpatialConfig::default()
    .with_output_layout(StandardLayout::Surround51)
    .with_render_mode(RenderMode::Speakers)
    .with_room(0.5);  // RT60 reverb time

let renderer = config.create_renderer()?;
```

### Object-Based Audio

```rust
use transcode_spatial::{ObjectAudioScene, AudioObject, Position3D};

let mut scene = ObjectAudioScene::new(16);

// Create an audio object at position (x=0.5, y=0.8, z=0.0)
let mut object = AudioObject::new(1, Position3D::new(0.5, 0.8, 0.0));
object.set_samples(audio_samples);
scene.add_object(object)?;
```

### Binaural Rendering

```rust
use transcode_spatial::{BinauralRenderer, SpatialConfig};

let config = SpatialConfig::new()
    .with_render_mode(RenderMode::Binaural)
    .with_head_tracking(true);

let renderer = config.create_binaural_renderer();
```

### Ambisonics

```rust
use transcode_spatial::{AmbisonicsEncoder, AmbisonicsDecoder, AmbisonicsFormat};

// Encode to ambisonics
let format = AmbisonicsFormat::ambix(1);  // First-order AmbiX
let encoder = AmbisonicsEncoder::new(format);

// Decode to speaker layout
let decoder = AmbisonicsDecoder::new(format, layout)?;
```

### Downmixing

```rust
use transcode_spatial::{Downmixer, DownmixMatrix, DownmixPreset, LfeMode};

let matrix = DownmixMatrix::with_preset(
    StandardLayout::Surround71,
    StandardLayout::Stereo,
    DownmixPreset::ItuR
);

let mut downmixer = Downmixer::new(matrix);
let stereo = downmixer.process(&surround_71)?;
```

## Presets

```rust
use transcode_spatial::presets;

let config = presets::home_theater_51();   // 5.1 with room
let config = presets::atmos_714();          // Dolby Atmos
let config = presets::headphones();         // Binaural
let config = presets::vr_headphones();      // VR with head tracking
let config = presets::game_audio();         // Gaming 7.1
```

## Channel Layouts

| Layout | Channels | Description |
|--------|----------|-------------|
| Stereo | 2 | L, R |
| Surround51 | 6 | L, R, C, LFE, Ls, Rs |
| Surround71 | 8 | + Lrs, Rrs |
| Atmos714 | 12 | + Ltf, Rtf, Ltr, Rtr |

## Render Modes

| Mode | Output | Use Case |
|------|--------|----------|
| Speakers | N channels | Home theater, studio |
| Binaural | 2 channels | Headphones |
| Ambisonics | (N+1)² channels | VR, ambisonic production |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
