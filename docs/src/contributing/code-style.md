# Code Style Guide

This guide describes the coding conventions used in the Transcode project.

## Formatting

We use `rustfmt` with the project's `rustfmt.toml` configuration:

```bash
cargo fmt --all
```

Key settings:
- Maximum line width: 100 characters
- Imports grouped by: std, external, crate
- Use field init shorthand: `Config { path }` not `Config { path: path }`

## Naming Conventions

### Types

```rust
// Structs: PascalCase
pub struct BitReader { ... }

// Enums: PascalCase, variants PascalCase
pub enum PixelFormat {
    Yuv420p,
    Rgb24,
}

// Traits: PascalCase, often adjectives or -able/-er suffixes
pub trait VideoDecoder { ... }
pub trait Encodable { ... }
```

### Functions and Methods

```rust
// Functions: snake_case
pub fn detect_simd() -> SimdCapabilities { ... }

// Methods: snake_case
impl Decoder {
    pub fn decode_frame(&mut self, packet: &Packet) -> Result<Frame> { ... }
}

// Constructors: new(), with_*(), or from_*()
impl Config {
    pub fn new() -> Self { ... }
    pub fn with_threads(mut self, n: usize) -> Self { ... }
    pub fn from_file(path: &Path) -> Result<Self> { ... }
}
```

### Variables

```rust
// Local variables: snake_case
let frame_count = 0;
let mut buffer = Vec::new();

// Constants: SCREAMING_SNAKE_CASE
const MAX_FRAME_SIZE: usize = 1920 * 1080 * 4;

// Statics: SCREAMING_SNAKE_CASE
static GLOBAL_CONFIG: Lazy<Config> = Lazy::new(|| Config::default());
```

## Documentation

### Public Items

All public items must have doc comments:

```rust
/// Decodes a video packet into a raw frame.
///
/// # Arguments
///
/// * `packet` - The compressed packet to decode
///
/// # Returns
///
/// The decoded frame, or an error if decoding failed.
///
/// # Errors
///
/// Returns `CodecError::BitstreamCorruption` if the packet data is invalid.
///
/// # Examples
///
/// ```
/// let mut decoder = H264Decoder::new()?;
/// let frame = decoder.decode(&packet)?;
/// println!("Decoded frame: {}x{}", frame.width, frame.height);
/// ```
pub fn decode(&mut self, packet: &Packet) -> Result<Frame> { ... }
```

### Module Documentation

Each module should have a top-level doc comment:

```rust
//! H.264/AVC decoder implementation.
//!
//! This module provides a pure Rust H.264 decoder with SIMD acceleration.
//!
//! # Features
//!
//! - Baseline, Main, and High profiles
//! - AVX2 acceleration on x86_64
//! - NEON acceleration on aarch64
//!
//! # Example
//!
//! ```
//! use transcode_codecs::video::h264::H264Decoder;
//! ```
```

## Error Handling

### Use the `?` Operator

```rust
// Good
fn process(path: &Path) -> Result<()> {
    let data = std::fs::read(path)?;
    let frame = decode(&data)?;
    Ok(())
}

// Avoid
fn process(path: &Path) -> Result<()> {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => return Err(e.into()),
    };
    // ...
}
```

### Add Context to Errors

```rust
use crate::error::ErrorContext;

fn decode_file(path: &Path) -> Result<Frame> {
    let data = std::fs::read(path)
        .context(format!("Failed to read file: {}", path.display()))?;

    decode(&data)
        .context("Failed to decode frame")?
}
```

### No Panics in Library Code

```rust
// Bad: panics on invalid input
fn get_plane(frame: &Frame, index: usize) -> &[u8] {
    &frame.planes[index]  // Panics if index out of bounds
}

// Good: returns Result
fn get_plane(frame: &Frame, index: usize) -> Result<&[u8]> {
    frame.planes.get(index)
        .ok_or(CodecError::InvalidPlaneIndex { index })
}
```

## Unsafe Code

### Minimize and Isolate

```rust
// Unsafe code should be in dedicated modules
mod simd {
    #[cfg(target_arch = "x86_64")]
    mod x86_64;

    #[cfg(target_arch = "aarch64")]
    mod aarch64;
}
```

### Document Safety Requirements

```rust
/// Performs AVX2-accelerated IDCT.
///
/// # Safety
///
/// * CPU must support AVX2 (check with `detect_simd().avx2`)
/// * `input` must be exactly 16 elements
/// * `output` must be exactly 16 elements and writable
#[target_feature(enable = "avx2")]
pub unsafe fn idct4x4_avx2(input: &[i16; 16], output: &mut [i16; 16]) {
    // ...
}
```

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_valid_input() {
        let input = create_test_packet();
        let result = decode(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_invalid_input() {
        let input = vec![0xFF; 100];  // Invalid data
        let result = decode(&input);
        assert!(matches!(result, Err(CodecError::BitstreamCorruption { .. })));
    }
}
```

### Test Names

Use descriptive names that explain what's being tested:

```rust
#[test]
fn decoder_returns_error_on_truncated_nal_unit() { ... }

#[test]
fn encoder_produces_keyframe_at_gop_boundary() { ... }
```

## Imports

### Organization

```rust
// 1. Standard library
use std::collections::HashMap;
use std::io::{Read, Write};

// 2. External crates
use bytes::Bytes;
use thiserror::Error;

// 3. Crate imports
use crate::error::Result;
use crate::frame::Frame;

// 4. Super/self imports
use super::common::parse_header;
```

### Prefer Explicit Imports

```rust
// Good: explicit
use std::collections::HashMap;

// Avoid: glob imports (except in tests/prelude)
use std::collections::*;
```

## Performance Considerations

### Avoid Unnecessary Allocations

```rust
// Good: reuse buffer
fn process_frames(frames: &[Frame], buffer: &mut Vec<u8>) {
    for frame in frames {
        buffer.clear();
        encode_frame(frame, buffer);
    }
}

// Avoid: allocate each iteration
fn process_frames(frames: &[Frame]) {
    for frame in frames {
        let buffer = Vec::new();  // Allocation per frame
        encode_frame(frame, &mut buffer);
    }
}
```

### Use Iterators

```rust
// Good: iterator chain
let sum: i32 = values.iter().filter(|x| **x > 0).sum();

// Avoid: explicit loop when iterator works
let mut sum = 0;
for x in &values {
    if *x > 0 {
        sum += x;
    }
}
```

## Clippy

All code must pass clippy without warnings:

```bash
cargo clippy --workspace -- -D warnings
```

Module-level allows are acceptable for DSP code with justification:

```rust
#![allow(clippy::needless_range_loop)]  // Common in signal processing
```
