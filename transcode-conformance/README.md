# transcode-conformance

H.264 conformance testing infrastructure for the transcode library.

## Overview

This crate provides comprehensive conformance testing against ITU-T H.264 reference test streams. It validates decoder/encoder output against known-good reference values to ensure specification compliance.

## Features

- **Test Infrastructure**: Framework for running conformance tests
- **Stream Management**: Download and cache ITU-T test streams
- **Profile Testing**: Baseline, Main, High profile support
- **Bitstream Validation**: Verify NAL unit structure and syntax
- **Frame Comparison**: MD5 checksum verification
- **Report Generation**: Detailed conformance reports

## Usage

### Running Tests

```bash
# Run all conformance tests (with cached streams)
cargo test -p transcode-conformance

# Run tests including those requiring downloads
cargo test -p transcode-conformance --features download --ignored

# Run specific profile tests
cargo test -p transcode-conformance baseline_
cargo test -p transcode-conformance main_
cargo test -p transcode-conformance high_
```

### Programmatic Usage

```rust
use transcode_conformance::{
    ConformanceConfig, TestStream, H264Profile, H264Level,
    runner::ConformanceRunner,
};

// Create configuration
let config = ConformanceConfig::default()
    .with_download(true)
    .with_cache_dir("/tmp/conformance".into());

// Run tests
let runner = ConformanceRunner::new(config);
let results = runner.run_all_tests()?;

for result in &results {
    println!("{}: {:?}", result.stream_id, result.status);
}
```

### Defining Test Streams

```rust
use transcode_conformance::{TestStream, H264Profile, H264Level};

let stream = TestStream::builder("test_001")
    .name("Basic Intra Test")
    .description("Tests basic I-frame decoding")
    .profile(H264Profile::Baseline)
    .level(H264Level::Level21)
    .resolution(352, 288)
    .expected_frames(10)
    .frame_checksum("d41d8cd98f00b204e9800998ecf8427e")
    .build();
```

## Supported Profiles

| Profile | Status | Description |
|---------|--------|-------------|
| Baseline | Supported | profile_idc = 66 |
| Constrained Baseline | Supported | profile_idc = 66 |
| Main | Supported | profile_idc = 77 |
| Extended | Partial | profile_idc = 88 |
| High | Supported | profile_idc = 100 |
| High 10 | Partial | profile_idc = 110 |
| High 4:2:2 | Partial | profile_idc = 122 |
| High 4:4:4 | Partial | profile_idc = 244 |

## Configuration Options

```rust
use transcode_conformance::ConformanceConfig;

let config = ConformanceConfig::default()
    .with_cache_dir("./test_streams".into())
    .with_local_streams("./local_streams".into())
    .with_download(true);

// Profile-specific configurations
let baseline = ConformanceConfig::baseline_only();
let main = ConformanceConfig::main_only();
let high = ConformanceConfig::high_only();
```

## Test Results

| Status | Description |
|--------|-------------|
| Passed | Test passed all validations |
| Failed | Test failed validation |
| Skipped | Test was skipped |
| Error | Test encountered an error |

## Feature Flags

- `download` - Enable downloading test streams from ITU-T

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
