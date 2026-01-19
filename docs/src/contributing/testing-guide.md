# Testing Guide

This guide covers testing practices for the Transcode project.

## Running Tests

### All Tests

```bash
# Run all tests (excluding Python bindings)
cargo test --workspace --exclude transcode-python

# Run with output
cargo test --workspace --exclude transcode-python -- --nocapture

# Run specific test
cargo test test_decode_valid_input
```

### Specific Crate

```bash
# Test a specific crate
cargo test -p transcode-codecs

# Test a specific module
cargo test -p transcode-codecs video::h264
```

### Feature-Gated Tests

```bash
# Run tests with specific features
cargo test -p transcode-codecs --features "h264,aac"

# Run all features
cargo test -p transcode-codecs --all-features
```

## Test Organization

### Unit Tests

Unit tests live in the same file as the code they test:

```rust
// src/video/h264/decoder.rs

pub struct H264Decoder { /* ... */ }

impl H264Decoder {
    pub fn decode(&mut self, packet: &Packet) -> Result<Option<Frame>> {
        // Implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_valid_packet() {
        let mut decoder = H264Decoder::new().unwrap();
        let packet = create_test_packet();
        let result = decoder.decode(&packet);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_empty_packet() {
        let mut decoder = H264Decoder::new().unwrap();
        let packet = Packet::empty();
        let result = decoder.decode(&packet);
        assert!(matches!(result, Err(CodecError::BitstreamCorruption { .. })));
    }
}
```

### Integration Tests

Integration tests live in the `tests/` directory:

```
transcode-codecs/
├── src/
│   └── ...
└── tests/
    ├── h264_decode.rs
    ├── h264_encode.rs
    └── common/
        └── mod.rs
```

```rust
// tests/h264_decode.rs

use transcode_codecs::video::h264::H264Decoder;

#[test]
fn test_full_decode_pipeline() {
    let data = include_bytes!("fixtures/sample.h264");
    let mut decoder = H264Decoder::new().unwrap();

    // Parse and decode all NAL units
    let frames = decode_all(&mut decoder, data).unwrap();

    assert!(!frames.is_empty());
    assert_eq!(frames[0].width(), 1920);
    assert_eq!(frames[0].height(), 1080);
}
```

### Test Fixtures

Store test data in `tests/fixtures/`:

```
tests/
└── fixtures/
    ├── sample.h264      # H.264 bitstream
    ├── sample.aac       # AAC audio
    └── expected/        # Expected output for comparison
        └── frame_0.yuv
```

## Test Patterns

### Testing Error Conditions

```rust
#[test]
fn decoder_returns_error_on_corrupted_header() {
    let mut decoder = H264Decoder::new().unwrap();
    let corrupted = vec![0xFF; 100];  // Invalid data

    let result = decoder.decode(&Packet::new(corrupted));

    assert!(matches!(
        result,
        Err(CodecError::BitstreamCorruption { .. })
    ));
}
```

### Testing SIMD Implementations

```rust
#[test]
fn avx2_idct_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return;  // Skip on unsupported hardware
    }

    let input = [1i16; 64];

    let mut scalar_out = input;
    let mut avx2_out = input;

    idct_8x8_scalar(&mut scalar_out);
    unsafe { idct_8x8_avx2(&mut avx2_out) };

    assert_eq!(scalar_out, avx2_out);
}
```

### Property-Based Testing

Using `proptest`:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn encode_decode_roundtrip(
        width in 16u32..=1920,
        height in 16u32..=1080,
    ) {
        // Ensure dimensions are multiples of 16
        let width = (width / 16) * 16;
        let height = (height / 16) * 16;

        let frame = Frame::new(width, height, PixelFormat::Yuv420p);
        let mut encoder = H264Encoder::new(config_for(width, height)).unwrap();
        let mut decoder = H264Decoder::new().unwrap();

        let packets = encoder.encode(&frame).unwrap();
        let decoded = decode_all(&mut decoder, &packets).unwrap();

        assert_eq!(decoded[0].width(), width);
        assert_eq!(decoded[0].height(), height);
    }
}
```

### Benchmark Tests

```rust
#[cfg(test)]
mod benches {
    use test::Bencher;

    #[bench]
    fn bench_decode_1080p_frame(b: &mut Bencher) {
        let data = include_bytes!("fixtures/1080p.h264");
        let mut decoder = H264Decoder::new().unwrap();

        b.iter(|| {
            decoder.reset();
            decode_all(&mut decoder, data).unwrap()
        });
    }
}
```

## Conformance Tests

### Codec Conformance

```rust
// Test against official test vectors
#[test]
fn h264_conformance_ba1_sony_d() {
    let stream = include_bytes!("conformance/BA1_Sony_D.264");
    let expected = load_yuv("conformance/BA1_Sony_D.yuv");

    let mut decoder = H264Decoder::new().unwrap();
    let frames = decode_all(&mut decoder, stream).unwrap();

    for (i, frame) in frames.iter().enumerate() {
        assert_frames_equal(frame, &expected[i], 0);  // Exact match
    }
}
```

### Quality Metrics

```rust
#[test]
fn encoded_output_meets_quality_threshold() {
    let reference = load_frame("fixtures/reference.yuv");
    let encoded = encode_and_decode(&reference);

    let psnr = calculate_psnr(&reference, &encoded);
    assert!(psnr > 40.0, "PSNR {} below threshold", psnr);

    let ssim = calculate_ssim(&reference, &encoded);
    assert!(ssim > 0.95, "SSIM {} below threshold", ssim);
}
```

## Fuzz Testing

Using `cargo-fuzz`:

```bash
# Install
cargo install cargo-fuzz

# Run fuzzer
cargo fuzz run decode_h264
```

Fuzz target:

```rust
// fuzz/fuzz_targets/decode_h264.rs

#![no_main]
use libfuzzer_sys::fuzz_target;
use transcode_codecs::video::h264::H264Decoder;

fuzz_target!(|data: &[u8]| {
    let mut decoder = H264Decoder::new().unwrap();
    let _ = decoder.decode(&Packet::new(data.to_vec()));
});
```

## Test Utilities

### Common Test Helpers

```rust
// tests/common/mod.rs

pub fn create_test_frame(width: u32, height: u32) -> Frame {
    let mut frame = Frame::new(width, height, PixelFormat::Yuv420p);
    // Fill with gradient pattern for visual debugging
    for y in 0..height {
        for x in 0..width {
            frame.set_pixel(x, y, 0, (x + y) as u8);
        }
    }
    frame
}

pub fn assert_frames_equal(a: &Frame, b: &Frame, tolerance: u8) {
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());

    for plane in 0..3 {
        let a_data = a.plane(plane);
        let b_data = b.plane(plane);
        for (i, (a_val, b_val)) in a_data.iter().zip(b_data).enumerate() {
            let diff = (*a_val as i16 - *b_val as i16).abs() as u8;
            assert!(
                diff <= tolerance,
                "Mismatch at plane {} offset {}: {} vs {} (diff {})",
                plane, i, a_val, b_val, diff
            );
        }
    }
}
```

## CI Integration

Tests run automatically on pull requests:

```yaml
# .github/workflows/ci.yml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Run tests
      run: cargo test --workspace --exclude transcode-python
    - name: Run doctests
      run: cargo test --doc --workspace --exclude transcode-python
```

## Coverage

Generate coverage reports:

```bash
# Using cargo-llvm-cov
cargo install cargo-llvm-cov
cargo llvm-cov --workspace --exclude transcode-python --html

# View report
open target/llvm-cov/html/index.html
```
