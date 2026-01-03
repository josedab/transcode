# Fuzz Testing for Transcode

This directory contains fuzz testing targets for the Transcode library using [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

## Prerequisites

Install cargo-fuzz:

```bash
cargo install cargo-fuzz
```

Note: Fuzzing requires a nightly Rust toolchain.

## Available Fuzz Targets

| Target | Description |
|--------|-------------|
| `fuzz_h264_decoder` | Fuzzes the H.264 decoder with random NAL unit data |
| `fuzz_aac_decoder` | Fuzzes the AAC decoder with random ADTS frames |
| `fuzz_mp4_demuxer` | Fuzzes the MP4 demuxer with random container data |
| `fuzz_bitstream` | Fuzzes the bit reader with random operations |

## Running Fuzz Tests

```bash
# Switch to nightly
rustup default nightly

# Run a specific fuzz target
cargo fuzz run fuzz_h264_decoder

# Run with a time limit (in seconds)
cargo fuzz run fuzz_h264_decoder -- -max_total_time=300

# Run with specific number of jobs
cargo fuzz run fuzz_h264_decoder -- -jobs=4

# List all available targets
cargo fuzz list
```

## Corpus Management

Each fuzz target maintains a corpus of interesting inputs:

```bash
# Show corpus location
ls fuzz/corpus/fuzz_h264_decoder/

# Minimize corpus
cargo fuzz cmin fuzz_h264_decoder

# Merge corpus from another run
cargo fuzz merge fuzz_h264_decoder
```

## Coverage

Generate coverage reports for fuzz testing:

```bash
cargo fuzz coverage fuzz_h264_decoder
```

## Reproducing Crashes

When a crash is found, cargo-fuzz saves the input to `fuzz/artifacts/`. To reproduce:

```bash
cargo fuzz run fuzz_h264_decoder fuzz/artifacts/fuzz_h264_decoder/crash-<hash>
```

## Adding New Fuzz Targets

1. Create a new file in `fuzz_targets/`:

```rust
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Your fuzzing code here
});
```

2. Add the target to `Cargo.toml`:

```toml
[[bin]]
name = "fuzz_my_target"
path = "fuzz_targets/fuzz_my_target.rs"
test = false
doc = false
bench = false
```

## Continuous Fuzzing

For CI/CD integration, consider using [OSS-Fuzz](https://github.com/google/oss-fuzz) or running fuzz tests with a time limit in your CI pipeline.
