# transcode-pertitle

Per-title encoding optimization for the transcode library. This crate provides content-adaptive encoding with automatic bitrate ladder generation based on video complexity analysis.

## Features

- **Content Complexity Analysis**: Analyzes spatial and temporal complexity using edge detection and frame differencing
- **Automatic Content Classification**: Detects content types (Animation, Film, Sports, Gaming, TalkingHead, Nature, Generic)
- **Adaptive Bitrate Ladder Generation**: Creates optimal rendition sets based on content characteristics
- **VMAF-Targeted Encoding**: Configurable quality targets with VMAF score estimation
- **Encoding Preset Optimization**: Recommends encoder presets based on complexity

## Key Types

| Type | Description |
|------|-------------|
| `PerTitleEncoder` | Main encoder that analyzes video and generates bitrate ladders |
| `PerTitleConfig` | Configuration for encoding (VMAF targets, bitrate limits, resolutions) |
| `BitrateLadder` | Generated ladder containing renditions sorted by bitrate |
| `Rendition` | Single encoding variant with resolution, bitrate, and preset |
| `ContentComplexity` | Analysis results with spatial/temporal scores and content type |
| `ContentAnalyzer` | Advanced content analysis utility |
| `LadderGenerator` | ABR streaming ladder generator |
| `EncodingOptimizer` | Preset recommendation based on complexity |

## Usage

### Basic Per-Title Analysis

```rust
use transcode_pertitle::{PerTitleEncoder, PerTitleConfig, Resolution};
use transcode_core::Frame;

// Create encoder with default config
let config = PerTitleConfig::default();
let encoder = PerTitleEncoder::new(config);

// Analyze frames and generate optimal bitrate ladder
let frames: Vec<Frame> = load_video_frames();
let ladder = encoder.analyze(&frames)?;

// Access generated renditions
for rendition in &ladder.renditions {
    println!(
        "{}x{} @ {} kbps (VMAF: {:.1})",
        rendition.resolution.width,
        rendition.resolution.height,
        rendition.bitrate,
        rendition.estimated_vmaf
    );
}
```

### Custom Configuration

```rust
use transcode_pertitle::{PerTitleConfig, Resolution, PerTitleEncoder};

let config = PerTitleConfig {
    target_vmaf: 95.0,
    min_bitrate: 300,
    max_bitrate: 12000,
    num_renditions: 5,
    resolutions: vec![
        Resolution::new(1920, 1080),
        Resolution::new(1280, 720),
        Resolution::new(854, 480),
        Resolution::new(640, 360),
    ],
    sample_rate: 2.0,
    vmaf_threshold: Some(90.0),
};

let encoder = PerTitleEncoder::new(config);
```

### Using the Encoding Optimizer

```rust
use transcode_pertitle::{EncodingOptimizer, PerTitleConfig};

let optimizer = EncodingOptimizer::new(PerTitleConfig::default());
let preset = optimizer.recommend_preset(&ladder.complexity);
// Returns: Slow for complex content, Medium for moderate, Fast for simple
```

## Default Configuration

- Target VMAF: 93.0
- Bitrate range: 200 - 15,000 kbps
- Renditions: 6 (4K, 1080p, 720p, 480p, 360p, 240p)
- Sample rate: 1.0 fps for analysis

## License

See workspace root for license information.
