# Contributing to Transcode

Thank you for your interest in contributing to Transcode! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We're all here to build great software together.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Python 3.8+ (for Python bindings development)
- Git

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/transcode.git
   cd transcode
   ```

3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/transcode/transcode.git
   ```

4. Build the project:
   ```bash
   cargo build
   ```

5. Run tests:
   ```bash
   cargo test
   ```

## Development Workflow

### Creating a Branch

Create a feature branch for your work:

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/bug-description
```

### Making Changes

1. Write your code following the project's style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Run the full test suite

### Code Style

We use standard Rust formatting and linting:

```bash
# Format code
cargo fmt

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings
```

### Commit Messages

Use clear, descriptive commit messages:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(h264): add support for B-frame encoding`
- `fix(aac): correct sample rate detection`
- `docs(readme): update installation instructions`

### Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/my-feature
   ```

2. Open a Pull Request on GitHub

3. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing performed
   - Breaking changes (if any)

4. Wait for CI to pass and request review

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p transcode-codecs

# Run a specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture
```

### Writing Tests

- Unit tests go in the same file as the code being tested
- Integration tests go in `tests/` directory
- Use descriptive test names
- Test both success and failure cases

Example:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_produces_valid_output() {
        // Arrange
        let input = create_test_frame();
        let mut encoder = H264Encoder::new(config);

        // Act
        let result = encoder.encode(&input);

        // Assert
        assert!(result.is_ok());
        let packet = result.unwrap();
        assert!(!packet.data().is_empty());
    }

    #[test]
    fn test_encoder_rejects_invalid_input() {
        // Test error handling
    }
}
```

### Benchmarks

Run benchmarks to ensure performance is maintained:

```bash
cargo bench
```

## Architecture Guidelines

### Crate Structure

- `transcode-core` - Core types and traits only
- `transcode-codecs` - Codec implementations
- `transcode-containers` - Container format handling
- `transcode-pipeline` - Orchestration logic
- `transcode-cli` - CLI binary
- `transcode-python` - Python bindings
- `transcode` - Public API facade

### Adding a New Codec

1. Create a new module in `transcode-codecs/src/video/` or `transcode-codecs/src/audio/`
2. Implement the `Decoder` and/or `Encoder` traits from `transcode-core`
3. Add SIMD optimizations if applicable
4. Add comprehensive tests
5. Update documentation
6. Add benchmarks

### SIMD Guidelines

- Always provide a scalar fallback
- Use runtime feature detection
- Test on multiple platforms
- Document performance characteristics

## Documentation

- All public APIs must have documentation comments
- Include examples in doc comments where helpful
- Update README.md for user-facing changes
- Add entries to CHANGELOG.md

## Review Process

1. All PRs require at least one approval
2. CI must pass
3. Code coverage should not decrease
4. No new clippy warnings

## Release Process

Releases are managed by maintainers. Version numbers follow [Semantic Versioning](https://semver.org/).

## Getting Help

- Open an issue for bugs or feature requests
- Use discussions for questions
- Tag maintainers for urgent issues

## License

By contributing, you agree that your contributions will be licensed under the MIT OR Apache-2.0 license.
