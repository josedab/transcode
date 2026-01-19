# ADR-0002: Workspace Organization

## Status

Accepted

## Date

2024-01

## Context

Transcode needs to support:
- Multiple video codecs (H.264, HEVC, AV1, VP9, etc.)
- Multiple audio codecs (AAC, Opus, FLAC, etc.)
- Multiple container formats (MP4, MKV, WebM, etc.)
- Multiple output targets (library, CLI, Python, WASM)
- Optional features (GPU, AI, distributed processing)

We need to organize the codebase to:
1. Allow users to include only what they need (minimize binary size)
2. Enable parallel development of different components
3. Maintain clear boundaries between modules
4. Support different release cadences for different components

## Decision

Organize the project as a **Cargo workspace** with multiple crates following this hierarchy:

```
transcode/
├── Core Layer
│   ├── transcode-core/       # Fundamental types (no codec-specific code)
│   ├── transcode-codecs/     # Base codec traits and common implementations
│   └── transcode-containers/ # Container format traits and base implementations
│
├── Codec Crates (one per major codec)
│   ├── transcode-av1/
│   ├── transcode-hevc/
│   ├── transcode-vp9/
│   └── ...
│
├── Processing Crates
│   ├── transcode-gpu/
│   ├── transcode-ai/
│   └── transcode-quality/
│
├── Integration Crates
│   ├── transcode/            # Public facade (re-exports)
│   ├── transcode-cli/
│   ├── transcode-python/
│   └── transcode-wasm/
│
└── Internal Crates
    ├── transcode-bench/
    └── transcode-conformance/
```

### Naming Convention

- `transcode-{codec}` for codec implementations
- `transcode-{format}` for container formats
- `transcode-{feature}` for processing features

### Dependency Direction

```
transcode (facade)
    ↓
transcode-pipeline
    ↓
transcode-codecs, transcode-containers
    ↓
transcode-core
```

Lower layers MUST NOT depend on higher layers.

## Consequences

### Positive

1. **Granular dependencies**: Users can depend only on crates they need
   ```toml
   [dependencies]
   transcode-core = "1.0"
   transcode-av1 = "1.0"  # Only AV1, not all codecs
   ```

2. **Parallel compilation**: Independent crates build in parallel

3. **Clear ownership**: Each crate has defined scope and maintainers

4. **Independent versioning**: Crates can have different release schedules (with care)

5. **Smaller binaries**: WASM builds can exclude unused codecs

### Negative

1. **Many crates to manage**: 70+ crates requires tooling for releases

2. **Dependency coordination**: Breaking changes in core affect many crates

3. **Discovery complexity**: Users need to know which crates to use

4. **Circular dependency risk**: Must carefully manage inter-crate dependencies

### Mitigations

1. **Facade crate**: `transcode` re-exports common functionality
   ```rust
   // Users can just use the facade
   use transcode::{Transcoder, TranscodeOptions};
   ```

2. **Workspace inheritance**: Common metadata in root `Cargo.toml`
   ```toml
   [workspace.package]
   version = "1.0.0"
   edition = "2021"
   ```

3. **Release automation**: Scripts to publish crates in dependency order

## Alternatives Considered

### Alternative 1: Monolithic Crate with Features

Single `transcode` crate with feature flags:

```toml
[dependencies]
transcode = { version = "1.0", features = ["av1", "hevc", "gpu"] }
```

Rejected because:
- All code in one compilation unit (slower builds)
- Feature flags can't express complex dependencies well
- Harder to maintain clear boundaries

### Alternative 2: Fewer, Larger Crates

Combine codecs into `transcode-video-codecs` and `transcode-audio-codecs`.

Rejected because:
- Still forces users to compile unused codecs
- Harder to add new codecs without affecting others
- Less clear ownership

## References

- [Cargo Workspaces](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html)
- [ripgrep workspace structure](https://github.com/BurntSushi/ripgrep)
- [image-rs organization](https://github.com/image-rs)
