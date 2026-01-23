---
sidebar_position: 4
title: Contributing
description: Contribute to Transcode development
---

# Contributing to Transcode

Thank you for your interest in contributing to Transcode! This guide will help you get started.

## Getting Started

### Prerequisites

- **Rust** 1.75 or later
- **Git**
- **Clang/LLVM** (for some codecs)

### Clone the Repository

```bash
git clone https://github.com/example/transcode
cd transcode
```

### Build the Project

```bash
# Build all crates (excluding Python bindings)
cargo build --workspace --exclude transcode-python

# Run tests
cargo test --workspace --exclude transcode-python

# Run clippy
cargo clippy --workspace --exclude transcode-python -- -D warnings
```

### Project Structure

```
transcode/
├── transcode-core/         # Core types and utilities
├── transcode-codecs/       # Codec implementations
├── transcode-containers/   # Container format support
├── transcode-pipeline/     # Transcoding pipeline
├── transcode-av1/          # AV1 codec
├── transcode-streaming/    # HLS/DASH streaming
├── transcode-gpu/          # GPU acceleration
├── transcode-ai/           # AI enhancement
├── transcode-quality/      # Quality metrics
├── transcode-distributed/  # Distributed processing
├── transcode-intel/        # Content intelligence
├── transcode-wasm/         # WebAssembly support
├── transcode/              # High-level API
├── transcode-cli/          # CLI interface
└── transcode-python/       # Python bindings
```

## Development Workflow

### Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Make Changes

1. Write your code following the style guidelines
2. Add tests for new functionality
3. Update documentation if needed
4. Run tests and linting

```bash
# Run all tests
cargo test --workspace --exclude transcode-python

# Run specific tests
cargo test -p transcode-codecs test_h264

# Run with logging
RUST_LOG=debug cargo test test_name -- --nocapture
```

### Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add AV1 encoding support
fix: handle empty packets in H.264 decoder
docs: update GPU acceleration guide
perf: optimize YUV to RGB conversion
refactor: simplify pipeline builder API
test: add integration tests for HLS muxer
chore: update dependencies
```

### Submit a Pull Request

1. Push your branch to GitHub
2. Open a pull request against `main`
3. Fill in the PR template
4. Wait for CI checks to pass
5. Address review feedback

## Code Style

### Formatting

Use `rustfmt` with default settings:

```bash
cargo fmt --all
```

### Linting

All code must pass `clippy`:

```bash
cargo clippy --workspace --exclude transcode-python -- -D warnings
```

### Naming Conventions

- Types: `PascalCase` - `VideoDecoder`, `H264Encoder`
- Functions/methods: `snake_case` - `decode_frame`, `get_bitrate`
- Constants: `SCREAMING_SNAKE_CASE` - `MAX_FRAME_SIZE`
- Modules: `snake_case` - `video_decoder`, `h264_encoder`

### Documentation

Document all public items:

```rust
/// Decodes H.264 video frames.
///
/// This decoder supports Baseline, Main, and High profiles up to level 5.1.
///
/// # Example
///
/// ```
/// use transcode_codecs::h264::H264Decoder;
///
/// let mut decoder = H264Decoder::new()?;
/// let frame = decoder.decode(&packet)?;
/// ```
///
/// # Errors
///
/// Returns `CodecError::BitstreamCorruption` if the packet contains invalid data.
pub struct H264Decoder {
    // ...
}
```

### Error Handling

Use the error types from `transcode-core`:

```rust
use transcode_core::error::{CodecError, Result};

fn decode_something(data: &[u8]) -> Result<Frame> {
    if data.is_empty() {
        return Err(CodecError::BitstreamCorruption {
            message: "Empty data".into(),
        }.into());
    }
    // ...
}
```

## Testing

### Unit Tests

Place unit tests in the same file:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_empty_packet() {
        let mut decoder = H264Decoder::new().unwrap();
        let packet = Packet::new(vec![]);
        let result = decoder.decode(&packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_valid_packet() {
        let mut decoder = H264Decoder::new().unwrap();
        let packet = create_test_packet();
        let frame = decoder.decode(&packet).unwrap();
        assert!(frame.is_some());
    }
}
```

### Integration Tests

Place integration tests in `tests/`:

```rust
// tests/h264_roundtrip.rs
use transcode_codecs::h264::{H264Decoder, H264Encoder};

#[test]
fn test_encode_decode_roundtrip() {
    let mut encoder = H264Encoder::new(config).unwrap();
    let mut decoder = H264Decoder::new().unwrap();

    // Create test frame
    let input_frame = create_test_frame();

    // Encode
    let packets = encoder.encode(&input_frame).unwrap();
    assert!(!packets.is_empty());

    // Decode
    let output_frame = decoder.decode(&packets[0]).unwrap().unwrap();

    // Verify dimensions match
    assert_eq!(input_frame.width(), output_frame.width());
    assert_eq!(input_frame.height(), output_frame.height());
}
```

### Test Data

Use the test fixtures in `tests/fixtures/`:

```rust
#[test]
fn test_decode_sample_video() {
    let data = include_bytes!("fixtures/sample.h264");
    let mut decoder = H264Decoder::new().unwrap();
    // ...
}
```

### Benchmarks

Add benchmarks for performance-critical code:

```rust
// benches/h264_encode.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_h264_encode(c: &mut Criterion) {
    let frame = create_1080p_frame();
    let mut encoder = H264Encoder::new(config).unwrap();

    c.bench_function("h264_encode_1080p", |b| {
        b.iter(|| {
            encoder.encode(black_box(&frame)).unwrap()
        })
    });
}

criterion_group!(benches, benchmark_h264_encode);
criterion_main!(benches);
```

Run benchmarks:

```bash
cargo bench -p transcode-codecs
```

## Adding New Features

### Adding a New Codec

1. Create a new module in `transcode-codecs/src/video/` or `audio/`
2. Implement the appropriate trait (`VideoDecoder`, `VideoEncoder`, etc.)
3. Add tests
4. Export from the parent module
5. Update documentation

Example structure:

```
transcode-codecs/src/video/
├── mod.rs
├── h264/
│   ├── mod.rs
│   ├── decoder.rs
│   ├── encoder.rs
│   └── ...
└── your_codec/
    ├── mod.rs
    ├── decoder.rs
    ├── encoder.rs
    └── ...
```

### Adding a New Container

1. Create a new module in `transcode-containers/src/`
2. Implement `Demuxer` and/or `Muxer` traits
3. Add atom/box parsing if applicable
4. Add tests with sample files
5. Update documentation

### Adding a New Feature

1. Discuss in an issue first for large features
2. Create a feature flag if it adds significant dependencies
3. Add documentation
4. Add tests
5. Update the changelog

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally
- [ ] Code is formatted with `rustfmt`
- [ ] Clippy shows no warnings
- [ ] Documentation is updated
- [ ] Changelog is updated (for user-facing changes)

### PR Template

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

Describe how you tested the changes.

## Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog updated
```

### Review Process

1. Maintainers will review within a few days
2. Address feedback and push updates
3. Once approved, maintainers will merge

## Release Process

Releases follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

## Community

### Getting Help

| Channel | Best For |
|---------|----------|
| [GitHub Issues](https://github.com/transcode/transcode/issues) | Bug reports, feature requests |
| [GitHub Discussions](https://github.com/transcode/transcode/discussions) | Questions, ideas, show & tell |
| [Discord](https://discord.gg/transcode) | Real-time chat, quick questions |
| [Twitter/X](https://twitter.com/transcode_rs) | News, announcements |

### Issue Guidelines

**Before opening an issue:**

1. Search existing issues to avoid duplicates
2. Check the [FAQ](/docs/getting-started/faq) for common questions
3. Try the latest version

**Bug reports should include:**

- Transcode version (`transcode --version`)
- Operating system and architecture
- Minimal reproduction steps
- Expected vs actual behavior
- Input file details (codec, resolution, if relevant)

**Feature requests should include:**

- Clear description of the feature
- Use case explanation
- Examples from other tools (if applicable)

### Good First Issues

New contributors should look for issues labeled:

- `good first issue` - Simple, well-defined tasks
- `help wanted` - Larger tasks where help is appreciated
- `documentation` - Docs improvements

### Communication Guidelines

- Be patient - maintainers are volunteers
- Be specific - vague questions get vague answers
- Be respectful - see Code of Conduct below
- Search first - many questions have been answered

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone.

### Our Standards

**Positive behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards other community members

**Unacceptable behaviors:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to the project maintainers at conduct@transcode.dev. All complaints will be reviewed and investigated promptly and fairly.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/) and the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT OR Apache-2.0).

## Recognition

Contributors are recognized in:

- The [CONTRIBUTORS](https://github.com/transcode/transcode/blob/main/CONTRIBUTORS.md) file
- Release notes for significant contributions
- The project website's acknowledgments

### Becoming a Maintainer

Regular contributors may be invited to become maintainers. Maintainers have:

- Write access to the repository
- Ability to merge pull requests
- Responsibility to review others' contributions
- Voice in project direction decisions

## Resources

### Learning Resources

- [The Rust Book](https://doc.rust-lang.org/book/) - Learn Rust
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) - Practical examples
- [Video Codec Specs](https://www.itu.int/rec/T-REC-H.264) - H.264 specification
- [Container Format Specs](https://www.iso.org/standard/68960.html) - ISO base media file format

### Development Tools

- [rust-analyzer](https://rust-analyzer.github.io/) - IDE support
- [cargo-watch](https://crates.io/crates/cargo-watch) - Auto-rebuild on changes
- [cargo-expand](https://crates.io/crates/cargo-expand) - View macro expansions
- [flamegraph](https://crates.io/crates/flamegraph) - Performance profiling

### Testing Tools

- [cargo-nextest](https://nexte.st/) - Faster test runner
- [cargo-llvm-cov](https://crates.io/crates/cargo-llvm-cov) - Code coverage
- [proptest](https://crates.io/crates/proptest) - Property-based testing

---

Thank you for contributing to Transcode! Every contribution, no matter how small, helps make the project better.
