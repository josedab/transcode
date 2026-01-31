---
sidebar_position: 15
title: FFmpeg Filter Compatibility
description: Parse and translate FFmpeg filter graphs to native Transcode filters
---

# FFmpeg Filter Compatibility

The `transcode-ffcompat` crate now includes filter graph parsing and translation, allowing you to reuse FFmpeg `-vf` and `-af` filter expressions with Transcode's native processing.

## Overview

If you're migrating from FFmpeg, you can use familiar filter syntax:

```rust
use transcode_ffcompat::filter::FilterGraph;

let graph = FilterGraph::parse("scale=1920:1080,fps=30,yadif");
let native_filters = graph.translate();

for filter in &native_filters {
    println!("{}: supported={}", filter.name, filter.supported);
}
// Output:
//   scale: supported=true
//   framerate: supported=true
//   deinterlace: supported=true
```

## Quick Start

```toml
[dependencies]
transcode-ffcompat = "1.0"
```

### Parsing Filter Expressions

The parser handles the full FFmpeg filter graph syntax:

```rust
use transcode_ffcompat::filter::FilterGraph;

// Simple filter chain (comma-separated)
let graph = FilterGraph::parse("scale=1280:720,fps=30");
assert_eq!(graph.chains.len(), 1);
assert_eq!(graph.chains[0].filters.len(), 2);

// Multiple chains (semicolon-separated)
let graph = FilterGraph::parse("scale=1280:720;loudnorm=I=-16:TP=-1.5");
assert_eq!(graph.chains.len(), 2);

// Named parameters
let graph = FilterGraph::parse("eq=brightness=0.1:contrast=1.2");
let native = graph.translate();
assert_eq!(native[0].params.get("brightness"), Some(&"0.1".to_string()));
```

### Supported Filters

| FFmpeg Filter | Native Equivalent | Parameters |
|---------------|-------------------|------------|
| `scale` | `scale` | width, height |
| `fps` | `framerate` | fps |
| `crop` | `crop` | width, height, x, y |
| `yadif` / `bwdif` | `deinterlace` | — |
| `transpose` | `rotate` | direction |
| `eq` | `color_adjust` | brightness, contrast, saturation |
| `loudnorm` | `loudness_normalize` | I, TP, LRA |
| `aresample` | `resample` | sample_rate |
| `tonemap` | `hdr_tonemap` | algorithm |
| `pad` | `pad` | width, height |

### Checking Compatibility

Evaluate how much of a filter graph is supported:

```rust
let graph = FilterGraph::parse("scale=1280:720,fps=30,overlay=10:10");
let ratio = graph.support_ratio();
println!("Support ratio: {:.0}%", ratio * 100.0);
// Output: Support ratio: 67%
```

### Using with the FFmpeg Argument Parser

Filter arguments integrate with the existing command-line parser:

```rust
use transcode_ffcompat::{FfmpegArgs, FilterGraph};

let args = FfmpegArgs::parse(&[
    "-i", "input.mp4",
    "-vf", "scale=1920:1080,yadif",
    "-c:v", "libx264",
    "output.mp4",
])?;

let native = args.translate();
println!("Video filters: {}", native.filters.len());
println!("Video codec: {}", native.video_codec.unwrap());
```

## Migration Examples

### FFmpeg → Transcode

**Resize and deinterlace:**
```bash
# FFmpeg
ffmpeg -i input.mp4 -vf "scale=1920:1080,yadif" output.mp4

# Transcode (using FFmpeg compat layer)
let graph = FilterGraph::parse("scale=1920:1080,yadif");
let filters = graph.translate();
// Apply filters to your pipeline
```

**Audio normalization:**
```bash
# FFmpeg
ffmpeg -i input.mp4 -af "loudnorm=I=-16:TP=-1.5:LRA=11" output.mp4

# Transcode
let graph = FilterGraph::parse("loudnorm=I=-16:TP=-1.5:LRA=11");
let filters = graph.translate();
```

## API Reference

| Type | Description |
|------|-------------|
| `FilterGraph` | Parsed filter graph with chains |
| `FilterChain` | Comma-separated filter sequence |
| `ParsedFilter` | Single filter with name and parameters |
| `NativeFilter` | Translated filter with support flag |

## Next Steps

- [FFmpeg Migration](/docs/guides/ffmpeg-migration) — Complete migration guide
- [Filter Chains](/docs/guides/filter-chains) — Native filter pipeline
