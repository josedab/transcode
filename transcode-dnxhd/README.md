# transcode-dnxhd

A pure Rust implementation of Avid DNxHD and DNxHR professional video codecs.

## Features

- **Full Profile Support**: All DNxHD and DNxHR profiles
- **DCT-Based Encoding**: Intra-frame only for editing performance
- **8-bit and 10-bit**: Multiple bit depth support
- **Huffman Entropy Coding**: Efficient coefficient encoding
- **Professional Quality**: Broadcast and finishing grade output

## DNxHD Profiles (HD Resolution)

| Profile | Bitrate | Quality |
|---------|---------|---------|
| DNxHD 36/45 | 36-45 Mbps | Offline |
| DNxHD 90/90x | 90 Mbps | Medium |
| DNxHD 120/145 | 120-145 Mbps | Broadcast |
| DNxHD 175/175x | 175 Mbps | High |
| DNxHD 220/220x | 220 Mbps | Highest |

## DNxHR Profiles (Higher Resolutions)

| Profile | Quality | Chroma |
|---------|---------|--------|
| DNxHR LB | Low Bandwidth | 4:2:2 8-bit |
| DNxHR SQ | Standard | 4:2:2 8-bit |
| DNxHR HQ | High | 4:2:2 8-bit |
| DNxHR HQX | High 10-bit | 4:2:2 10-bit |
| DNxHR 444 | Full Chroma | 4:4:4 10-bit |

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-dnxhd = { path = "../transcode-dnxhd" }
```

### Decoding

```rust
use transcode_dnxhd::{DnxDecoder, DnxFrame};

let data = std::fs::read("video.dnxhd")?;
let mut decoder = DnxDecoder::new();
let frame = decoder.decode_frame(&data)?;

println!("Frame: {}x{}", frame.width, frame.height);
println!("Profile: {:?}", frame.profile);
```

### Encoding

```rust
use transcode_dnxhd::{DnxEncoder, EncoderConfig, DnxProfile};

let config = EncoderConfig::new(1920, 1080)
    .with_profile(DnxProfile::DnxHd220);

let mut encoder = DnxEncoder::new(config)?;
let encoded = encoder.encode_frame(&frame)?;
```

### Probe Profile

```rust
use transcode_dnxhd::{probe_dnxhd, get_profile};

if probe_dnxhd(&data) {
    let profile = get_profile(&data)?;
    println!("Detected profile: {:?}", profile);
}
```

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
