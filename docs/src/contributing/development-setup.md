# Development Setup

This guide will help you set up a development environment for contributing to Transcode.

## Prerequisites

- **Rust 1.75+**: Install via [rustup](https://rustup.rs/)
- **Git**: For version control
- **Python 3.8+**: For Python bindings development (optional)

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/transcode/transcode.git
cd transcode
```

### 2. Build the Project

```bash
# Build all crates (excluding Python bindings)
cargo build --workspace --exclude transcode-python

# Or use just (recommended)
just build
```

### 3. Run Tests

```bash
cargo test --workspace --exclude transcode-python

# Or use just
just test
```

### 4. Verify Setup

```bash
# Run the SIMD detection example
cargo run --example simd_detection
```

## Using GitHub Codespaces

The easiest way to start developing is with GitHub Codespaces:

1. Click the "Code" button on the GitHub repository
2. Select "Open with Codespaces"
3. Wait for the environment to build

The devcontainer includes all necessary tools pre-configured.

## Using Docker

```bash
# Start development environment
docker compose up dev

# Run tests in Docker
docker compose run --rm test

# Run linting
docker compose run --rm lint
```

## IDE Setup

### VS Code (Recommended)

Install these extensions:
- [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
- [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml)
- [CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb) (debugging)
- [crates](https://marketplace.visualstudio.com/items?itemName=serayuzgur.crates)

Recommended settings (`.vscode/settings.json`):

```json
{
    "rust-analyzer.check.command": "clippy",
    "rust-analyzer.check.extraArgs": ["--", "-D", "warnings"],
    "editor.formatOnSave": true
}
```

### JetBrains (CLion/RustRover)

1. Open the project directory
2. Install the Rust plugin
3. Configure the toolchain in Settings → Languages & Frameworks → Rust

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

Follow the [Code Style Guide](./code-style.md).

### 3. Run Quality Checks

```bash
# Format code
just fmt

# Run linter
just lint

# Run tests
just test

# All checks
just ci
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat(component): add new feature"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

### 5. Submit Pull Request

Push your branch and open a PR on GitHub.

## Project Structure

```
transcode/
├── transcode-core/         # Core types and utilities
├── transcode-codecs/       # Codec implementations
├── transcode-containers/   # Container formats
├── transcode-pipeline/     # Pipeline orchestration
├── transcode/              # Public API facade
├── transcode-cli/          # CLI tool
├── transcode-python/       # Python bindings
├── fuzz/                   # Fuzz testing targets
├── docs/                   # Documentation (mdBook)
└── docker/                 # Docker configuration
```

## Common Tasks

### Adding a New Codec

See [Adding a Codec](./adding-a-codec.md).

### Running Benchmarks

```bash
just bench
```

### Running Fuzz Tests

```bash
# List available targets
just fuzz-list

# Run a specific target
just fuzz h264_nal

# Run for 60 seconds
just fuzz-timed h264_nal 60
```

### Building Documentation

```bash
# Build API docs
just doc-open

# Build this book
just book-serve
```

## Troubleshooting

### Build Failures

1. Ensure Rust 1.75+ is installed: `rustc --version`
2. Update Rust: `rustup update stable`
3. Clean and rebuild: `just rebuild`

### Test Failures

1. Run specific failing test: `cargo test test_name -- --nocapture`
2. Check for environment-specific issues (CI vs local)

### Clippy Warnings

All clippy warnings must be fixed before merging:

```bash
cargo clippy --workspace --exclude transcode-python -- -D warnings
```

## Getting Help

- Open an issue for bugs
- Use GitHub Discussions for questions
- Tag maintainers for urgent issues
