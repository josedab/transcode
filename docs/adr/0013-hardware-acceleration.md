# ADR-0013: Hardware Acceleration Abstraction

## Status

Accepted

## Date

2024-06

## Context

Modern video transcoding workflows demand high throughput that software encoding alone cannot achieve cost-effectively. Hardware encoders (GPU and dedicated ASICs) offer 5-20x speedup over software but introduce complexity:

1. **Platform fragmentation**: Each platform has different APIs
   - macOS: VideoToolbox
   - Linux: VA-API (Intel/AMD), NVENC (NVIDIA)
   - Windows: Media Foundation, NVENC, AMF

2. **Feature disparity**: Hardware encoders support different codecs, profiles, and features

3. **Availability uncertainty**: Hardware may not be present or may be in use

4. **Quality tradeoffs**: Hardware encoders often produce larger files at equivalent visual quality

The challenge is providing hardware acceleration benefits without forcing users to write platform-specific code or handle complex fallback logic.

## Decision

Implement a **unified hardware acceleration abstraction layer** in `transcode-hwaccel` with:

### 1. Runtime Detection

```rust
pub fn detect_accelerators() -> Vec<HwAccelInfo> {
    // Probes system for available hardware
    // Returns capabilities, supported codecs, device info
}

pub enum HwAccelType {
    Vaapi,        // Linux (Intel, AMD)
    VideoToolbox, // macOS
    Nvenc,        // NVIDIA (cross-platform)
    Qsv,          // Intel Quick Sync
    Amf,          // AMD
    D3d11va,      // Windows
    Software,     // Fallback
}
```

### 2. Unified Encoder/Decoder Traits

```rust
pub struct HwEncoder {
    accelerator: HwAccelType,
    // Platform-specific handle (opaque)
}

impl HwEncoder {
    pub fn new(accel: HwAccelType, config: HwEncoderConfig) -> Result<Self>;
    pub fn encode(&mut self, frame: &Frame) -> Result<Packet>;
    pub fn flush(&mut self) -> Result<Vec<Packet>>;
}
```

### 3. Graceful Fallback

```rust
pub fn create_encoder(config: EncoderConfig) -> Result<Box<dyn VideoEncoder>> {
    // Try hardware first
    if let Some(hw) = detect_accelerators().first() {
        if let Ok(encoder) = HwEncoder::new(hw.accel_type, config.into()) {
            return Ok(Box::new(encoder));
        }
    }
    // Fall back to software
    Ok(Box::new(SoftwareEncoder::new(config)?))
}
```

### 4. Platform Backends (Conditional Compilation)

```rust
#[cfg(target_os = "macos")]
pub mod videotoolbox;

#[cfg(target_os = "linux")]
pub mod vaapi;

#[cfg(feature = "nvenc")]
pub mod nvenc;
```

## Consequences

### Positive

1. **Write-once deployment**: Same code works on macOS, Linux, Windows with best available acceleration

2. **Automatic fallback**: Applications don't crash when hardware is unavailable

3. **Capability discovery**: Users can query supported codecs/profiles before encoding

4. **Isolation**: Platform-specific unsafe code is contained in backend modules

5. **Performance transparency**: Users can check which accelerator was selected

### Negative

1. **Lowest common denominator**: Some advanced hardware features may not be exposed

2. **Testing complexity**: Requires multiple platforms/GPUs for full test coverage

3. **Driver dependencies**: VAAPI requires libva, NVENC requires CUDA toolkit

4. **Quality variance**: Same settings produce different output across accelerators

### Mitigations

1. **Capability queries**: `HwAccelInfo` reports exact supported features
   ```rust
   if accel.supports_codec(VideoCodec::Hevc)
       && accel.supports_profile(HevcProfile::Main10) {
       // Safe to use
   }
   ```

2. **Quality presets**: Map abstract quality levels to per-accelerator tuning

3. **Optional features**: Hardware backends are feature-gated
   ```toml
   [dependencies]
   transcode-hwaccel = { version = "1.0", features = ["nvenc"] }
   ```

## Implementation Details

### VideoToolbox (macOS)

- Uses `VTCompressionSession` for encoding
- Hardware always available on Apple Silicon
- Excellent H.264/HEVC quality
- ProRes hardware encoding on M1 Pro/Max/Ultra

### VA-API (Linux)

- Requires `libva` runtime and appropriate driver
- Intel: `intel-media-va-driver` or `intel-media-va-driver-non-free`
- AMD: `libva-mesa-driver`
- Supports H.264, HEVC, VP9, AV1 (on newer hardware)

### NVENC (NVIDIA)

- Requires CUDA toolkit and `libnvidia-encode`
- Session limits on consumer GPUs (3 concurrent on GeForce)
- B-frame support varies by generation
- AV1 on RTX 40 series

### QSV (Intel Quick Sync)

- Available on Intel CPUs with integrated graphics
- Linux: via VA-API or standalone
- Windows: via Media Foundation
- Good quality/speed balance for H.264

## Alternatives Considered

### Alternative 1: Direct FFI to Each Platform

Expose raw platform APIs and let users handle abstraction.

Rejected because:
- Defeats the purpose of a unified library
- Users would need platform-specific code paths
- Error handling varies wildly across platforms

### Alternative 2: GStreamer Integration

Use GStreamer's hardware abstraction layer.

Rejected because:
- Heavy dependency (pulls in GLib, etc.)
- Doesn't align with pure-Rust goals
- Complex build requirements

### Alternative 3: Hardware-Only Library

Separate crate that only does hardware, no software fallback.

Rejected because:
- Poor user experience when hardware unavailable
- Forces users to implement fallback logic
- Doesn't match "it just works" goal

## References

- [VideoToolbox Documentation](https://developer.apple.com/documentation/videotoolbox)
- [VA-API Specification](https://github.com/intel/libva)
- [NVENC Programming Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/nvenc-video-encoder-api-prog-guide/)
- [Intel Media SDK](https://github.com/Intel-Media-SDK/MediaSDK)
