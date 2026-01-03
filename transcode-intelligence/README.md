# transcode-intelligence

Content intelligence for video analysis in the transcode ecosystem.

> **Note**: This crate is a placeholder. The content intelligence implementation is currently available in [`transcode-intel`](../transcode-intel/).

## Overview

Content intelligence provides automated video analysis capabilities:

- **Scene Detection** - Identify scene boundaries and transitions
- **Content Classification** - Categorize shots, content types, and motion levels
- **Motion Analysis** - Track motion for adaptive encoding decisions

## Content Intelligence Features

### Scene Detection Methods

| Method | Description | Speed |
|--------|-------------|-------|
| `Histogram` | Histogram difference analysis | Fast |
| `ContentDiff` | Luminance and perceptual hash comparison | Medium |
| `Edge` | Sobel edge detection differences | Medium |
| `Combined` | Weighted combination of all methods | Slower, most accurate |

### Classification Categories

**Shot Types**: ExtremeCloseUp, CloseUp, MediumCloseUp, Medium, MediumLong, Long, ExtremeLong, Overhead

**Content Types**: Dialogue, Action, Landscape, Static, Animation, Text

**Motion Levels**: Static, Slow, Normal, Fast, VeryFast

## Key Types

| Type | Description |
|------|-------------|
| `VideoAnalyzer` | Combined scene detection and classification pipeline |
| `SceneDetector` | Configurable scene boundary detection |
| `ContentClassifier` | Shot type, content, and motion classification |
| `Scene` | Detected scene with frame range, confidence, and type |
| `Classification` | Per-frame classification results |
| `FrameAnalysis` | Combined scene and classification analysis |
| `SequenceAnalysis` | Full video sequence analysis with statistics |

## Usage Examples

### Basic Scene Detection

```rust
use transcode_intel::{SceneDetector, SceneConfig, DetectionMethod, Frame};

let config = SceneConfig::default()
    .with_method(DetectionMethod::Combined)
    .with_threshold(0.3)
    .with_min_scene_length(12);

let mut detector = SceneDetector::new(config);

for frame in frames {
    if let Some(confidence) = detector.process_frame(&frame)? {
        println!("Scene change at confidence: {:.2}", confidence);
    }
}
```

### Content Classification

```rust
use transcode_intel::{ContentClassifier, Frame};

let mut classifier = ContentClassifier::new();
let result = classifier.classify(&frame)?;

println!("Shot: {:?}", result.shot_type);
println!("Content: {:?}", result.content_type);
println!("Motion: {:?} (score: {:.2})", result.motion_level, result.motion_score);
println!("Complexity: {:.2}", result.complexity);

// Adaptive encoding recommendation
let bitrate_factor = result.motion_level.recommended_bitrate_factor();
```

### Full Video Analysis

```rust
use transcode_intel::{VideoAnalyzer, SceneConfig};

let mut analyzer = VideoAnalyzer::new(SceneConfig::default());
let analysis = analyzer.analyze_sequence(&frames)?;

println!("Total scenes: {}", analysis.scene_count());
println!("Average scene length: {:.1} frames", analysis.avg_scene_length());
println!("Dominant content: {:?}", analysis.dominant_content);
println!("Average motion: {:.2}", analysis.avg_motion);
println!("Bitrate factor: {:.2}", analysis.recommended_bitrate_factor());
```

## Use Cases

- **Adaptive bitrate encoding** - Adjust quality based on motion and complexity
- **Scene-based segmentation** - Split video at natural scene boundaries
- **Thumbnail selection** - Choose representative frames from each scene
- **Content-aware processing** - Apply different filters based on content type

## License

See the workspace root for license information.
