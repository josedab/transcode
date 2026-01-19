# ADR-0001: Pure Rust Implementation

## Status

Accepted

## Date

2024-01

## Context

When building a codec library, there are two primary approaches:

1. **Wrap existing C libraries** (e.g., FFmpeg, libavcodec)
   - Pros: Mature, battle-tested implementations
   - Cons: FFI complexity, memory safety concerns, system dependencies

2. **Implement codecs from scratch in Rust**
   - Pros: Memory safety, no system dependencies, easier cross-compilation
   - Cons: Significant development effort, potential spec compliance issues

The primary goals of Transcode are:
- Memory safety as a first-class concern
- Easy deployment (single binary, no dependencies)
- Cross-platform support including WebAssembly

## Decision

We will implement codecs entirely in Rust without relying on FFI to C libraries for core functionality.

Exceptions are allowed for:
- **Hardware acceleration** (VAAPI, VideoToolbox, NVENC) which require system APIs
- **AV1 decoding via dav1d** where pure Rust performance is not yet competitive
- **Patent-encumbered codecs** where FFI wrappers are explicitly opt-in

## Consequences

### Positive

1. **Memory safety**: Rust's ownership system eliminates entire classes of vulnerabilities (buffer overflows, use-after-free, data races)

2. **No system dependencies**: Core functionality works on any platform with a Rust compiler

3. **Easy cross-compilation**: Building for different targets is straightforward
   ```bash
   cargo build --target wasm32-unknown-unknown
   cargo build --target aarch64-unknown-linux-gnu
   ```

4. **Single binary deployment**: No need to manage shared libraries

5. **WebAssembly support**: Pure Rust code compiles to WASM without modification

6. **Auditability**: Security audits can examine the full codebase

### Negative

1. **Development effort**: Implementing codecs from scratch requires significant expertise and time

2. **Performance gap**: Initial implementations may be slower than C libraries with decades of optimization

3. **Spec compliance risk**: Subtle spec deviations possible without extensive conformance testing

4. **Ecosystem fragmentation**: Users may need both Transcode and FFmpeg for different use cases

### Neutral

1. **SIMD optimization**: Still requires `unsafe` blocks but contained within well-tested modules

2. **Learning curve**: Contributors need Rust expertise (but this is the target audience)

## Alternatives Considered

### Alternative 1: Pure FFmpeg Bindings

Create safe Rust bindings around FFmpeg (like `ffmpeg-next`).

Rejected because:
- Doesn't achieve memory safety goals
- Complex build process with system dependencies
- Difficult to cross-compile
- No WASM support

### Alternative 2: Hybrid Approach

Use FFmpeg for complex codecs, Rust for simple ones.

Rejected because:
- Inconsistent safety guarantees
- Still requires FFmpeg as a dependency
- Confusing API surface

## References

- [Rust Safety Guarantees](https://doc.rust-lang.org/nomicon/meet-safe-and-unsafe.html)
- [WebAssembly Rust Support](https://rustwasm.github.io/docs/book/)
- [symphonia](https://github.com/pdeljanov/Symphonia) - Similar pure Rust audio library
