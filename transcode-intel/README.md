# transcode-intel

Content intelligence for video analysis. Provides scene detection, content classification, and motion analysis for adaptive video processing.

## Features

### Scene Detection

Detect scene changes using multiple methods:

- **Histogram** - Fast histogram-based detection
- **ContentDiff** - Luminance and perceptual hash comparison (default)
- **Edge** - Sobel edge detection differences
- **Combined** - Weighted combination for best accuracy

Supports adaptive thresholds and configurable minimum scene lengths.

### Content Classification

Classify frames by:

- **Shot Type** - ExtremeCloseUp, CloseUp, Medium, Long, ExtremeLong, Overhead
- **Content Type** - Dialogue, Action, Landscape, Static, Animation, Text
- **Motion Level** - Static, Slow, Normal, Fast, VeryFast

Also extracts visual metrics: complexity, saturation, and brightness.

## Key Types

| Type | Description |
|------|-------------|
| `VideoAnalyzer` | Combined scene detection and classification |
| `SceneDetector` | Detects scene boundaries with configurable methods |
| `ContentClassifier` | Classifies shot types, content, and motion |
| `Scene` | Detected scene with frame range and confidence |
| `Classification` | Frame classification results |
| `Frame` | RGB frame data for analysis |

## Usage

### Scene Detection

```rust
use transcode_intel::{SceneDetector, SceneConfig, DetectionMethod, Frame};

let config = SceneConfig::default()
    .with_method(DetectionMethod::Combined)
    .with_threshold(0.3);

let mut detector = SceneDetector::new(config);

for frame in frames {
    if let Some(confidence) = detector.process_frame(&frame)? {
        println!("Scene change detected (confidence: {:.2})", confidence);
    }
}
```

### Content Classification

```rust
use transcode_intel::{ContentClassifier, Frame};

let mut classifier = ContentClassifier::new();
let classification = classifier.classify(&frame)?;

println!("Shot: {:?}, Content: {:?}, Motion: {:?}",
    classification.shot_type,
    classification.content_type,
    classification.motion_level);

// Use for adaptive encoding
let bitrate_factor = classification.motion_level.recommended_bitrate_factor();
```

### Full Video Analysis

```rust
use transcode_intel::{VideoAnalyzer, SceneConfig};

let mut analyzer = VideoAnalyzer::new(SceneConfig::default());
let analysis = analyzer.analyze_sequence(&frames)?;

println!("Scenes: {}", analysis.scene_count());
println!("Avg scene length: {:.1} frames", analysis.avg_scene_length());
println!("Dominant content: {:?}", analysis.dominant_content);
println!("Recommended bitrate factor: {:.2}", analysis.recommended_bitrate_factor());
```

## Use Cases

- **Adaptive bitrate encoding** - Adjust quality based on motion and complexity
- **Scene-based segmentation** - Split video at natural boundaries
- **Thumbnail selection** - Choose representative frames per scene
- **Content-aware filtering** - Apply different processing based on content type

## License

See the workspace root for license information.
