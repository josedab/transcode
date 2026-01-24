# Good First Issues

Welcome to Transcode! This document lists beginner-friendly issues and tasks to help you get started contributing.

## Getting Started

Before picking up an issue:

1. **Set up your environment:**
   ```bash
   git clone https://github.com/anthropics/transcode
   cd transcode
   cargo build --workspace --exclude transcode-python
   cargo test --workspace --exclude transcode-python
   ```

2. **Read the docs:**
   - [CLAUDE.md](./CLAUDE.md) - Project structure and patterns
   - [CONTRIBUTING.md](./CONTRIBUTING.md) - Contribution guidelines

3. **Join the community:**
   - GitHub Discussions for questions
   - Discord for real-time chat

## Difficulty Levels

- **Beginner**: No prior codec knowledge needed, mostly documentation or simple code changes
- **Intermediate**: Requires understanding of video concepts but well-documented
- **Advanced**: Requires codec expertise or complex implementation

---

## Documentation Tasks

### Beginner

#### DOC-001: Add docstrings to public types in `transcode-core`
**Description:** Many public types in `transcode-core/src/` lack documentation comments.
**Skills:** Rust, technical writing
**Files:** `transcode-core/src/*.rs`
**What to do:**
1. Find types without `///` doc comments
2. Add clear, concise documentation
3. Include examples where helpful

#### DOC-002: Improve error messages
**Description:** Some error variants have generic messages. Make them more specific and actionable.
**Skills:** Rust
**Files:** `transcode-core/src/error.rs`
**Example:**
```rust
// Before
Error::InvalidFormat("unsupported format")

// After
Error::InvalidFormat("MOV format detected but only MP4 is supported. Use a container conversion tool first.")
```

#### DOC-003: Add examples to CLI help text
**Description:** The CLI `--help` output could include more practical examples.
**Skills:** Rust, CLI design
**Files:** `transcode-cli/src/main.rs`
**What to do:** Add `#[arg(help = "...")]` with usage examples

#### DOC-004: Document codec parameters
**Description:** Create a reference for codec-specific parameters (profile, level, tuning).
**Skills:** Video encoding knowledge, Markdown
**Files:** `website/docs/reference/codec-parameters.md`

---

## Testing Tasks

### Beginner

#### TEST-001: Add unit tests for bitstream reader
**Description:** `BitReader` in `transcode-core` needs more edge case tests.
**Skills:** Rust testing
**Files:** `transcode-core/src/bitstream/reader.rs`
**Test cases needed:**
- Reading past end of buffer
- Reading 0 bits
- Reading across byte boundaries

#### TEST-002: Add tests for error formatting
**Description:** Ensure all error types format correctly with `Display`.
**Skills:** Rust
**Files:** `transcode-core/src/error.rs`
**What to do:**
1. Add tests that format each error variant
2. Verify error messages are human-readable

#### TEST-003: Test sample media files
**Description:** Add test fixtures for edge cases (very short videos, single frame, etc.).
**Skills:** Video encoding
**Files:** `tests/fixtures/`
**What to do:**
1. Create minimal test files using ffmpeg
2. Add integration tests that process them

### Intermediate

#### TEST-004: Add fuzzing targets
**Description:** Set up cargo-fuzz targets for codec parsers.
**Skills:** Rust, fuzzing
**Files:** `fuzz/`
**Focus areas:**
- NAL unit parsing
- MP4 atom parsing
- ADTS header parsing

---

## Code Tasks

### Beginner

#### CODE-001: Implement `Display` for more types
**Description:** Several types implement `Debug` but not `Display` for human-readable output.
**Skills:** Rust
**Files:** Various
**Types to add `Display` to:**
- `PixelFormat` - e.g., "YUV 4:2:0 planar"
- `SampleFormat` - e.g., "32-bit float, interleaved"
- `Profile` - e.g., "H.264 High Profile"

#### CODE-002: Add `From` implementations for error conversion
**Description:** Some error conversions use `.map_err()` when `From` would be cleaner.
**Skills:** Rust
**Files:** `transcode-core/src/error.rs`
**What to do:**
1. Find patterns like `.map_err(Error::from_io)`
2. Implement `From<std::io::Error>` for `Error`
3. Simplify callers to use `?`

#### CODE-003: Clippy lint cleanup
**Description:** Enable more clippy lints and fix warnings.
**Skills:** Rust
**Files:** Workspace-wide
**Lints to enable:**
- `clippy::pedantic` (selectively)
- `clippy::unwrap_used`
- `clippy::expect_used`

### Intermediate

#### CODE-004: Add progress estimation to CLI
**Description:** Improve progress bar ETA accuracy using frame rate and duration.
**Skills:** Rust
**Files:** `transcode-cli/src/main.rs`, `transcode-cli/src/progress.rs`
**What to do:**
1. Calculate estimated total frames from duration
2. Update ETA based on actual vs expected progress
3. Handle variable frame rate videos

#### CODE-005: Implement `--dry-run` for CLI
**Description:** Add a flag that validates inputs without transcoding.
**Skills:** Rust, CLI design
**Files:** `transcode-cli/src/main.rs`
**What to do:**
1. Add `--dry-run` flag
2. Open and probe input file
3. Validate output codec compatibility
4. Print what would be done

#### CODE-006: Add JSON output for `info` command
**Description:** The `transcode info` command should support `--json` output.
**Skills:** Rust, serde
**Files:** `transcode-cli/src/commands/info.rs`
**What to do:**
1. Add `--json` flag
2. Serialize stream info to JSON
3. Ensure machine-parseable output

### Advanced

#### CODE-007: Implement seeking for MP4 demuxer
**Description:** Add `seek_to_timestamp()` method to `Mp4Demuxer`.
**Skills:** Rust, MP4 format knowledge
**Files:** `transcode-containers/src/mp4/demuxer.rs`
**What to do:**
1. Parse `stss` atom for keyframe positions
2. Implement binary search for timestamp
3. Seek to nearest keyframe before target

#### CODE-008: Add VP9 decoder stub
**Description:** Create the structure for a VP9 decoder (can be minimal implementation).
**Skills:** Rust, VP9 knowledge
**Files:** `transcode-codecs/src/video/vp9/`
**What to do:**
1. Define `Vp9Decoder` struct
2. Parse superframe index
3. Parse frame header (uncompressed portion)

---

## Infrastructure Tasks

### Beginner

#### INFRA-001: Add CI badge to README
**Description:** Add build status, coverage, and docs badges.
**Skills:** Markdown, CI
**Files:** `README.md`

#### INFRA-002: Create issue templates
**Description:** Add GitHub issue templates for bugs and features.
**Skills:** Markdown, GitHub
**Files:** `.github/ISSUE_TEMPLATE/`

### Intermediate

#### INFRA-003: Add benchmarks to CI
**Description:** Run benchmarks on PRs and report significant regressions.
**Skills:** GitHub Actions, Rust
**Files:** `.github/workflows/`
**What to do:**
1. Run `cargo bench` in CI
2. Compare against main branch
3. Comment on PR with results

#### INFRA-004: Set up dependabot
**Description:** Configure automatic dependency updates.
**Skills:** GitHub, YAML
**Files:** `.github/dependabot.yml`

---

## Binding Tasks

### Beginner

#### BIND-001: Add more Python examples
**Description:** Create example scripts showing common use cases.
**Skills:** Python
**Files:** `transcode-python/examples/`
**Examples to add:**
- Batch processing with multiprocessing
- Integration with MoviePy
- Flask/FastAPI endpoint

#### BIND-002: Add Go SDK examples
**Description:** Create example programs for the Go SDK.
**Skills:** Go
**Files:** `bindings/go/examples/`
**Examples to add:**
- Progress monitoring
- Stream selection
- Error handling patterns

### Intermediate

#### BIND-003: Add Swift SDK tests
**Description:** Expand test coverage for Swift bindings.
**Skills:** Swift
**Files:** `bindings/swift/Tests/`
**What to do:**
1. Test error handling paths
2. Test async/await integration
3. Test memory management

---

## How to Claim an Issue

1. Comment on the issue expressing interest
2. Wait for a maintainer to assign it to you
3. Ask questions if anything is unclear
4. Submit a PR when ready

## Tips for Success

- **Start small:** Pick a beginner issue first
- **Ask questions:** Don't struggle alone
- **Read existing code:** Follow established patterns
- **Write tests:** Every change should have tests
- **Document:** Update docs if behavior changes

## Need Help?

- **GitHub Discussions:** For design questions
- **Discord:** For quick help
- **Issue comments:** For clarification on specific issues

Thank you for contributing to Transcode!
