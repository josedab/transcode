# ADR-0009: Dual Streaming Protocol Support (HLS/DASH)

## Status

Accepted

## Date

2024-05 (inferred from module structure)

## Context

Adaptive bitrate streaming requires delivering video in segmented formats that allow clients to switch quality levels based on network conditions. Two protocols dominate the industry:

1. **HLS (HTTP Live Streaming)**: Apple's protocol, widely supported on iOS, Safari, and smart TVs
2. **DASH (Dynamic Adaptive Streaming over HTTP)**: MPEG standard, preferred for Android and web

We need to support:

- Multiple quality levels (adaptive bitrate)
- Both HLS and DASH output from the same source
- Low-latency variants (LL-HLS, Low-latency DASH)
- DRM integration (Widevine, FairPlay)
- CMAF (Common Media Application Format) for shared segments

The challenge is generating efficient output for both protocols without duplicating significant work.

## Decision

Implement **dual protocol support** with a **shared CMAF segment layer** and protocol-specific manifest generators.

### 1. Unified Streaming Configuration

Common configuration for both protocols:

```rust
pub struct StreamingConfig {
    pub output_dir: String,
    pub segment_duration: f64,  // Default: 6.0 seconds
    pub qualities: Vec<Quality>,
    pub drm_enabled: bool,
    pub drm_key_id: Option<String>,
}

pub struct Quality {
    pub width: u32,
    pub height: u32,
    pub bitrate: u64,
}

impl Quality {
    pub fn fhd_1080p() -> Self { Quality::new(1920, 1080, 5_000_000) }
    pub fn hd_720p() -> Self { Quality::new(1280, 720, 2_500_000) }
    pub fn sd_480p() -> Self { Quality::new(854, 480, 1_000_000) }
}
```

### 2. HLS Output

Generate Apple HLS playlists and segments:

```rust
pub struct HlsConfig {
    output_dir: String,
    segment_duration: f64,
    qualities: Vec<Quality>,
    playlist_type: PlaylistType,  // Event, VOD, Live
}

pub struct HlsWriter {
    config: HlsConfig,
    master_playlist: MasterPlaylist,
    media_playlists: HashMap<String, MediaPlaylist>,
}

impl HlsWriter {
    pub fn write_master_playlist(&self) -> Result<()> {
        // #EXTM3U
        // #EXT-X-VERSION:7
        // #EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080
        // 1080p/playlist.m3u8
        // #EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720
        // 720p/playlist.m3u8
    }
}
```

### 3. DASH Output

Generate MPEG-DASH manifests (MPD):

```rust
pub struct DashConfig {
    output_dir: String,
    segment_duration: f64,
    qualities: Vec<Quality>,
    profile: DashProfile,  // OnDemand, Live
    mpd_type: MpdType,     // Static, Dynamic
}

pub struct DashWriter {
    config: DashConfig,
    manifest: MpdManifest,
}

impl DashWriter {
    pub fn write_mpd(&self) -> Result<()> {
        // <?xml version="1.0"?>
        // <MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static">
        //   <Period>
        //     <AdaptationSet mimeType="video/mp4">
        //       <Representation bandwidth="5000000" width="1920" height="1080"/>
        //     </AdaptationSet>
        //   </Period>
        // </MPD>
    }
}
```

### 4. CMAF Shared Segments

Use CMAF for segment format compatibility:

```rust
pub struct CmafConfig {
    pub fragment_duration: f64,
    pub output_format: CmafOutputFormat,
    pub encryption: Option<CmafEncryption>,
}

pub enum CmafOutputFormat {
    /// Fragmented MP4 (fMP4)
    FragmentedMp4,
    /// WebM container
    WebM,
}

pub struct CmafWriter {
    config: CmafConfig,
    init_segment: CmafInitSegment,
}

impl CmafWriter {
    pub fn write_init_segment(&self, track: &TrackConfig) -> Result<CmafInitSegment> {
        // Generate initialization segment (ftyp + moov)
    }

    pub fn write_media_segment(&self, samples: &[SampleInfo]) -> Result<CmafMediaSegment> {
        // Generate media segment (styp + moof + mdat)
    }
}
```

### 5. Low-Latency HLS Support

Support LL-HLS for reduced latency:

```rust
pub struct LowLatencyConfig {
    pub part_duration: f64,           // Partial segment duration
    pub preload_hint: bool,           // Enable preload hints
    pub rendition_reports: bool,      // Cross-rendition sync
    pub blocking_playlist: bool,      // Server-side playlist blocking
}

pub struct PartialSegment {
    pub sequence: u64,
    pub part_number: u32,
    pub duration: f64,
    pub independent: bool,  // Starts with keyframe
    pub uri: String,
}

pub struct ServerControl {
    pub can_skip_until: f64,          // Delta update threshold
    pub can_block_reload: bool,       // Blocking playlist reload
    pub part_hold_back: f64,          // Part holdback duration
}
```

### 6. DRM Support

Integrate content protection:

```rust
pub struct HlsDrmConfig {
    pub key_method: HlsKeyMethod,    // AES-128, SAMPLE-AES
    pub key_uri: String,
    pub key_format: HlsKeyFormat,    // Identity, FairPlay
}

pub struct DashDrmConfig {
    pub scheme_id_uri: String,       // Widevine, PlayReady
    pub key_id: String,
    pub pssh: Vec<u8>,               // Protection System Specific Header
}
```

### Architecture Diagram

```
                    ┌──────────────────────────────────────┐
                    │          Source Video                │
                    └────────────────┬─────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────┐
                    │          Transcoder                  │
                    │  (Multiple quality renditions)       │
                    └────────────────┬─────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
    ┌─────────┐               ┌──────────┐               ┌─────────┐
    │ 1080p   │               │   720p   │               │  480p   │
    │ Encode  │               │  Encode  │               │ Encode  │
    └────┬────┘               └────┬─────┘               └────┬────┘
         │                         │                          │
         └────────────────┬────────┴──────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   CMAF Segmenter      │
              │  (fMP4 segments)      │
              └───────────┬───────────┘
                          │
           ┌──────────────┴──────────────┐
           │                             │
           ▼                             ▼
    ┌─────────────┐               ┌─────────────┐
    │ HLS Writer  │               │ DASH Writer │
    │ (m3u8)      │               │ (mpd)       │
    └─────────────┘               └─────────────┘
```

## Consequences

### Positive

1. **Broad compatibility**: Reach all major platforms (iOS, Android, web, smart TVs)

2. **Single encoding pass**: Encode once, package for both protocols

3. **CMAF efficiency**: Shared segment format reduces storage requirements

4. **Low latency option**: LL-HLS enables near-real-time streaming

5. **DRM flexibility**: Support major DRM systems per platform

6. **Adaptive quality**: Clients automatically select appropriate quality

### Negative

1. **Manifest overhead**: Must generate and maintain both HLS and DASH manifests

2. **Complexity**: Two standards with different quirks and requirements

3. **Testing burden**: Must verify playback on multiple clients

4. **Segment alignment**: Ensuring consistent segmentation across qualities

### Mitigations

1. **Unified API**: Single configuration generates both formats

```rust
let config = StreamingConfig::default()
    .with_quality(Quality::fhd_1080p())
    .with_quality(Quality::hd_720p());

let hls = HlsWriter::new(HlsConfig::from(&config))?;
let dash = DashWriter::new(DashConfig::from(&config))?;
```

2. **Segment validation**: Automated checks for segment boundaries

3. **Test player integration**: Built-in validation with reference players

## Alternatives Considered

### Alternative 1: HLS Only

Support only Apple HLS.

Rejected because:
- Poor support on older Android devices
- No native Windows support
- Limits audience reach

### Alternative 2: DASH Only

Support only MPEG-DASH.

Rejected because:
- No native iOS/Safari support (requires MSE polyfill)
- Apple devices are significant market share
- Enterprise customers often require HLS

### Alternative 3: HLS with DASH Fallback

Generate HLS primarily, with DASH as second-class citizen.

Rejected because:
- Both protocols deserve first-class support
- Some deployments need DASH-specific features
- Inconsistent user experience

### Alternative 4: Progressive Download

Serve complete files instead of streaming.

Rejected because:
- No adaptive bitrate
- High initial latency
- Poor user experience on variable networks

## References

- [HLS specification](https://datatracker.ietf.org/doc/html/rfc8216)
- [DASH specification](https://dashif.org/guidelines/)
- [CMAF specification](https://www.iso.org/standard/71975.html)
- [LL-HLS specification](https://developer.apple.com/documentation/http_live_streaming/enabling_low-latency_http_live_streaming_hls)
- [fMP4 segment format](https://www.w3.org/TR/mse-byte-stream-format-isobmff/)
