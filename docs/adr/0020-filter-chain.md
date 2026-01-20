# ADR-0020: Filter Chain Composition

## Status

Accepted

## Date

2024-08

## Context

Video processing pipelines often require transformations between decode and encode:

- **Scaling**: Resize to different resolution
- **Cropping**: Remove letterboxing
- **Frame rate conversion**: 24fps → 30fps
- **Color correction**: Brightness, contrast, saturation
- **Deinterlacing**: Convert interlaced to progressive
- **Overlay**: Add watermarks, logos

These operations must be:
1. **Composable**: Chain multiple filters in sequence
2. **Type-safe**: Can't accidentally mix video and audio filters
3. **Efficient**: Minimize intermediate buffer copies
4. **Extensible**: Easy to add new filter types

## Decision

Implement a **composable filter chain architecture** with typed filters:

### 1. Filter Trait Hierarchy

```rust
/// Base filter trait (marker)
pub trait Filter: Send {
    fn name(&self) -> &str;
    fn is_enabled(&self) -> bool { true }
}

/// Video-specific filter
pub trait VideoFilter: Filter {
    fn process(&mut self, frame: Frame) -> Result<Frame>;
    fn flush(&mut self) -> Result<Vec<Frame>>;

    /// Optional: output multiple frames (interpolation)
    fn process_multi(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        Ok(vec![self.process(frame)?])
    }
}

/// Audio-specific filter
pub trait AudioFilter: Filter {
    fn process(&mut self, samples: SampleBuffer) -> Result<SampleBuffer>;
    fn flush(&mut self) -> Result<Vec<SampleBuffer>>;
}
```

### 2. Filter Chain Container

```rust
pub struct FilterChain<F: Filter> {
    filters: Vec<Box<F>>,
}

impl<F: VideoFilter> FilterChain<F> {
    pub fn new() -> Self;
    pub fn push(&mut self, filter: impl VideoFilter + 'static);

    pub fn process(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        let mut frames = vec![frame];

        for filter in &mut self.filters {
            let mut next_frames = Vec::new();
            for f in frames {
                next_frames.extend(filter.process_multi(f)?);
            }
            frames = next_frames;
        }

        Ok(frames)
    }

    pub fn flush(&mut self) -> Result<Vec<Frame>> {
        // Flush each filter, feeding output to next
    }
}
```

### 3. Concrete Filter Implementations

```rust
/// Scale filter (resize)
pub struct ScaleFilter {
    target_width: u32,
    target_height: u32,
    algorithm: ScaleAlgorithm,
}

impl VideoFilter for ScaleFilter {
    fn process(&mut self, frame: Frame) -> Result<Frame> {
        // Resize frame to target dimensions
    }
}

/// Volume filter (audio gain)
pub struct VolumeFilter {
    gain_db: f32,
    linear_gain: f32,
}

impl AudioFilter for VolumeFilter {
    fn process(&mut self, mut samples: SampleBuffer) -> Result<SampleBuffer> {
        for sample in samples.as_mut_slice() {
            *sample *= self.linear_gain;
        }
        Ok(samples)
    }
}
```

### 4. Null/Pass-through Filters

```rust
/// No-op video filter
pub struct NullVideoFilter;

impl VideoFilter for NullVideoFilter {
    fn process(&mut self, frame: Frame) -> Result<Frame> {
        Ok(frame) // Pass through unchanged
    }
    fn flush(&mut self) -> Result<Vec<Frame>> {
        Ok(vec![])
    }
}
```

### 5. Builder Pattern for Chain Construction

```rust
pub struct FilterChainBuilder {
    video_filters: Vec<Box<dyn VideoFilter>>,
    audio_filters: Vec<Box<dyn AudioFilter>>,
}

impl FilterChainBuilder {
    pub fn new() -> Self;

    pub fn scale(self, width: u32, height: u32) -> Self {
        self.video_filters.push(Box::new(ScaleFilter::new(width, height)));
        self
    }

    pub fn fps(self, target_fps: f64) -> Self {
        self.video_filters.push(Box::new(FpsFilter::new(target_fps)));
        self
    }

    pub fn volume(self, gain_db: f32) -> Self {
        self.audio_filters.push(Box::new(VolumeFilter::new(gain_db)));
        self
    }

    pub fn build(self) -> (FilterChain<dyn VideoFilter>, FilterChain<dyn AudioFilter>);
}
```

## Consequences

### Positive

1. **Type safety**: Compiler prevents mixing video/audio filters

2. **Composability**: Filters chain naturally
   ```rust
   let chain = FilterChainBuilder::new()
       .scale(1920, 1080)
       .fps(30.0)
       .build();
   ```

3. **Extensibility**: New filters implement trait, plug into chain

4. **Testability**: Each filter testable in isolation

5. **Clear data flow**: Frame passes through filters in order

### Negative

1. **Memory overhead**: Each filter step may allocate output

2. **Limited parallelism**: Sequential processing within chain

3. **Frame timing complexity**: Multi-output filters (interpolation) complicate timing

### Mitigations

1. **In-place processing**: Where possible, modify frame in place
   ```rust
   pub trait InPlaceVideoFilter: Filter {
       fn process_in_place(&mut self, frame: &mut Frame) -> Result<()>;
   }
   ```

2. **Buffer reuse**: Filters can reuse internal buffers across calls

3. **Lazy evaluation**: Build filter graph, optimize before execution

## Implementation Details

### Filter Registration

```rust
/// Filter factory for dynamic instantiation
pub type FilterFactory = fn(&FilterParams) -> Result<Box<dyn VideoFilter>>;

pub struct FilterRegistry {
    factories: HashMap<String, FilterFactory>,
}

impl FilterRegistry {
    pub fn register(&mut self, name: &str, factory: FilterFactory);
    pub fn create(&self, name: &str, params: &FilterParams) -> Result<Box<dyn VideoFilter>>;
}

// Usage
registry.register("scale", |params| {
    let width = params.get_u32("w")?;
    let height = params.get_u32("h")?;
    Ok(Box::new(ScaleFilter::new(width, height)))
});
```

### Filter Graph (Advanced)

For complex filtering with splits/merges:

```rust
pub struct FilterGraph {
    nodes: Vec<FilterNode>,
    edges: Vec<(usize, usize)>,  // (from, to)
}

pub struct FilterNode {
    filter: Box<dyn VideoFilter>,
    input_pads: Vec<String>,
    output_pads: Vec<String>,
}

impl FilterGraph {
    pub fn from_string(spec: &str) -> Result<Self>;
    // Parse: "[0:v]scale=1920:1080[scaled];[scaled]fps=30[out]"
}
```

### Available Filters

#### Video Filters

| Filter | Description | Parameters |
|--------|-------------|------------|
| `scale` | Resize frame | `w`, `h`, `algorithm` |
| `crop` | Crop region | `x`, `y`, `w`, `h` |
| `fps` | Frame rate convert | `fps` |
| `pad` | Add borders | `w`, `h`, `x`, `y`, `color` |
| `deinterlace` | Deinterlace | `mode` |
| `rotate` | Rotate frame | `angle` |
| `flip` | Horizontal/vertical flip | `direction` |
| `overlay` | Composite images | `x`, `y`, `alpha` |

#### Audio Filters

| Filter | Description | Parameters |
|--------|-------------|------------|
| `volume` | Adjust gain | `dB` or `ratio` |
| `resample` | Change sample rate | `rate` |
| `channels` | Remap channels | `layout` |
| `normalize` | Loudness normalization | `target_lufs` |
| `compressor` | Dynamic range | `threshold`, `ratio` |

### Frame Timing Preservation

Filters must handle PTS correctly:

```rust
impl ScaleFilter {
    fn process(&mut self, mut frame: Frame) -> Result<Frame> {
        // Preserve timing
        let pts = frame.pts();
        let duration = frame.duration();

        let scaled = self.scale_internal(&frame)?;

        // Restore timing on output
        scaled.set_pts(pts);
        scaled.set_duration(duration);

        Ok(scaled)
    }
}
```

### Multi-Frame Filters

Some filters output multiple frames (frame interpolation, deinterlacing):

```rust
pub struct FrameInterpolator {
    target_fps: f64,
    prev_frame: Option<Frame>,
}

impl VideoFilter for FrameInterpolator {
    fn process_multi(&mut self, frame: Frame) -> Result<Vec<Frame>> {
        let mut output = Vec::new();

        if let Some(prev) = &self.prev_frame {
            // Generate interpolated frames between prev and current
            let interp_frames = self.interpolate(prev, &frame);
            output.extend(interp_frames);
        }

        output.push(frame.clone());
        self.prev_frame = Some(frame);

        Ok(output)
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        // Return any buffered frames
        Ok(self.prev_frame.take().into_iter().collect())
    }
}
```

## Alternatives Considered

### Alternative 1: Monolithic Filter Function

Single function that does all processing:

```rust
fn process_video(frame: Frame, opts: &ProcessOptions) -> Frame {
    let frame = if opts.scale.is_some() { scale(frame, opts) } else { frame };
    let frame = if opts.crop.is_some() { crop(frame, opts) } else { frame };
    // ...
}
```

Rejected because:
- Not extensible
- Hard to test individual operations
- Combinatorial explosion of options

### Alternative 2: Inheritance Hierarchy

Object-oriented filter hierarchy with base class.

Rejected because:
- Rust doesn't have inheritance
- Less flexible than traits
- Harder to compose

### Alternative 3: AST-Based Filter Language

Define filters in a DSL, interpret at runtime.

Rejected because:
- Higher runtime overhead
- Harder to optimize
- Rust's type system already provides composition

## References

- [FFmpeg Filters](https://ffmpeg.org/ffmpeg-filters.html)
- [GStreamer Elements](https://gstreamer.freedesktop.org/documentation/plugin-development/element-types/)
- [Iterator Pattern in Rust](https://doc.rust-lang.org/book/ch13-02-iterators.html)
- [Type State Pattern](https://cliffle.com/blog/rust-typestate/)
