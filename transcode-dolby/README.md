# transcode-dolby

Dolby Vision support for the transcode library.

## Overview

This crate provides comprehensive Dolby Vision support for HDR video processing, including RPU parsing, metadata handling, profile conversion, and tone/gamut mapping.

## Features

- **Profile Support**: Profiles 4, 5, 7, 8, and 9 with level definitions
- **RPU Parsing**: Reference Processing Unit NAL parsing and serialization
- **Metadata Handling**: L1-L11 metadata structures and processing
- **Tone/Gamut Mapping**: Polynomial, MMR, and NLQ processing
- **Profile Conversion**: Convert between profiles (e.g., Profile 7 to 8)
- **Stream Extraction**: Extract RPUs and separate layers from HEVC streams

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
transcode-dolby = { path = "../transcode-dolby" }
```

### Basic Configuration

```rust
use transcode_dolby::{DolbyConfig, DolbyVisionProfile};

// Create a configuration for Profile 8 (HDR10 compatible)
let config = DolbyConfig::profile8_hdr10();

// Or create with custom settings
let config = DolbyConfig::new(DolbyVisionProfile::Profile8)
    .with_target_peak(4000.0)
    .with_level(DolbyVisionLevel::Level9);
```

### RPU Extraction

```rust
use transcode_dolby::extractor::extract_all_rpus;

// Extract RPUs from an HEVC stream
let rpus = extract_all_rpus(&hevc_data)?;

// Process metadata
for rpu in &rpus {
    if let Some(l1) = &rpu.metadata.l1 {
        println!("Scene max: {} nits", l1.max_nits());
    }
}
```

### Profile Conversion

```rust
use transcode_dolby::converter::ProfileConverter;
use transcode_dolby::DolbyVisionProfile;

// Convert Profile 7 (dual-layer) to Profile 8 (single-layer)
let converter = ProfileConverter::profile7_to_8();
let converted_rpu = converter.convert_rpu(&source_rpu)?;
```

### Tone Mapping

```rust
use transcode_dolby::mapping::ToneMapper;

// Create a tone mapper for a 1000 nit display
let mapper = ToneMapper::for_1000_nits();
mapper.configure_from_rpu(&rpu)?;

// Map a pixel
let (r, g, b) = mapper.map_pixel(r_in, g_in, b_in);
```

### Stream Analysis

```rust
use transcode_dolby::DolbyVisionInfo;

// Analyze a stream
if let Some(info) = DolbyVisionInfo::from_stream(&data)? {
    println!("Profile: {:?}", info.profile);
    println!("Has EL: {}", info.has_enhancement_layer);
    println!("Max CLL: {:?}", info.max_cll);
    println!("Codec string: {}", info.codec_string());
}
```

## Profiles

| Profile | Description | Layers |
|---------|-------------|--------|
| Profile 4 | HLG HDR | Dual |
| Profile 5 | SDR compatible | Single |
| Profile 7 | MEL + EL | Dual |
| Profile 8 | HDR10 compatible | Single |
| Profile 9 | SDR + enhancement | Dual |

## Metadata Levels

| Level | Description |
|-------|-------------|
| L1 | Content light levels |
| L2 | Trim metadata |
| L3 | Scene-level metadata |
| L5 | Active area |
| L6 | MaxCLL/MaxFALL |
| L8-L11 | Extended metadata |

## Documentation

See the main [transcode documentation](../README.md) for the complete library overview.
