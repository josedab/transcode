# Transcode Development Commands
# https://github.com/casey/just
#
# Install just: cargo install just
# Run `just` to see all available commands

# Default recipe: show available commands
default:
    @just --list

# ============================================================================
# Building
# ============================================================================

# Build all crates (excluding Python bindings)
build:
    cargo build --workspace --exclude transcode-python

# Build in release mode with optimizations (thin LTO, fast)
build-release:
    cargo build --release --workspace --exclude transcode-python

# Build with maximum optimizations (full LTO, codegen-units=1 — slow but smallest/fastest binary)
build-release-max:
    cargo build --profile release-max --workspace --exclude transcode-python

# Build the CLI tool
build-cli:
    cargo build --release -p transcode-cli

# Build Python bindings
build-python:
    cd transcode-python && maturin build --release

# Build WebAssembly package
build-wasm:
    cd transcode-wasm && wasm-pack build --release

# ============================================================================
# Testing
# ============================================================================

# Run all tests
test:
    cargo test --workspace --exclude transcode-python

# Run tests with output shown
test-verbose:
    cargo test --workspace --exclude transcode-python -- --nocapture

# Run tests for a specific crate
test-crate crate:
    cargo test -p {{crate}}

# Run only unit tests (no integration tests)
test-unit:
    cargo test --workspace --exclude transcode-python --lib

# Run integration tests only
test-integration:
    cargo test --workspace --exclude transcode-python --test '*'

# Run property-based tests
test-proptest:
    cargo test --workspace --exclude transcode-python proptest

# ============================================================================
# Code Quality
# ============================================================================

# Run clippy linter
lint:
    cargo clippy --workspace --exclude transcode-python -- -D warnings

# Run clippy with all features
lint-all:
    cargo clippy --workspace --exclude transcode-python --all-features -- -D warnings

# Format all code
fmt:
    cargo fmt --all

# Check formatting without making changes
fmt-check:
    cargo fmt --all -- --check

# Run all quality checks (format + lint + test)
check: fmt-check lint test

# Full CI check (same as GitHub Actions)
ci: fmt-check lint test doc-check

# ============================================================================
# Security & Auditing
# ============================================================================

# Run security audit
audit:
    cargo audit

# Run dependency license check
deny:
    cargo deny check

# Check for outdated dependencies
outdated:
    cargo outdated -R

# Update dependencies
update:
    cargo update

# ============================================================================
# Documentation
# ============================================================================

# Build documentation
doc:
    cargo doc --workspace --exclude transcode-python --no-deps

# Build and open documentation in browser
doc-open:
    cargo doc --workspace --exclude transcode-python --no-deps --open

# Check documentation for warnings
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude transcode-python --no-deps

# Serve documentation locally (requires Python)
doc-serve: doc
    python3 -m http.server 8000 --directory target/doc

# Build mdBook documentation
book:
    cd docs && mdbook build

# Serve mdBook with live reload
book-serve:
    cd docs && mdbook serve --open

# ============================================================================
# Benchmarks
# ============================================================================

# Run all benchmarks
bench:
    cargo bench --package transcode-bench

# Run benchmarks without running them (compile only)
bench-check:
    cargo bench --package transcode-bench --no-run

# Run codec benchmarks
bench-codecs:
    cargo bench --package transcode-bench -- codecs

# Run quality metric benchmarks
bench-quality:
    cargo bench --package transcode-quality

# ============================================================================
# Fuzzing (requires nightly)
# ============================================================================

# List available fuzz targets
fuzz-list:
    cd fuzz && cargo +nightly fuzz list

# Run a specific fuzz target
fuzz target:
    cd fuzz && cargo +nightly fuzz run {{target}}

# Run fuzz target for a specific duration (in seconds)
fuzz-timed target duration="60":
    cd fuzz && cargo +nightly fuzz run {{target}} -- -max_total_time={{duration}}

# ============================================================================
# Release
# ============================================================================

# Check if ready for release (all checks pass)
release-check: ci audit deny
    @echo "All release checks passed!"

# Dry-run publish to crates.io
publish-dry:
    cargo publish -p transcode-core --dry-run
    cargo publish -p transcode-codecs --dry-run
    cargo publish -p transcode-containers --dry-run
    cargo publish -p transcode --dry-run

# ============================================================================
# Development Helpers
# ============================================================================

# Watch for changes and run checks
watch:
    cargo watch -x check -x test

# Watch for changes and run clippy
watch-lint:
    cargo watch -x clippy

# Clean build artifacts
clean:
    cargo clean

# Clean and rebuild
rebuild: clean build

# Show dependency tree
deps:
    cargo tree

# Show duplicate dependencies
deps-duplicates:
    cargo tree --duplicates

# Build with timing report (opens HTML in browser)
build-timings:
    cargo build --workspace --exclude transcode-python --timings
    @echo "Timing report: target/cargo-timings/"

# Show sccache statistics
cache-stats:
    sccache --show-stats

# Count lines of code
loc:
    @echo "Rust source files:"
    @find . -path ./target -prune -o -path ./fuzz/target -prune -o -name "*.rs" -type f -print | wc -l
    @echo "Total lines of Rust code:"
    @wc -l $$(find . -path ./target -prune -o -path ./fuzz/target -prune -o -name "*.rs" -type f -print) 2>/dev/null | tail -1

# ============================================================================
# Docker
# ============================================================================

# Build Docker image
docker-build:
    docker build -t transcode:latest .

# Run Docker container
docker-run *args:
    docker run -v $(pwd):/data transcode:latest {{args}}

# Start development environment with Docker Compose
docker-dev:
    docker compose up dev

# Run tests in Docker
docker-test:
    docker compose run --rm test

# Start observability stack
docker-observability:
    docker compose --profile observability up

# ============================================================================
# Examples
# ============================================================================

# Run basic transcode example
example-basic:
    cargo run --example basic_transcode

# Run SIMD detection example
example-simd:
    cargo run --example simd_detection

# Run quality metrics example
example-quality:
    cargo run --example quality_metrics

# Run content intelligence example
example-intel:
    cargo run --example content_intelligence

# List all examples
examples:
    @echo "Available examples:"
    @ls transcode/examples/*.rs | xargs -I {} basename {} .rs
