# transcode-hdr

HDR video processing library for the transcode project. Provides HDR10, HDR10+, Dolby Vision support, tone mapping, and color space conversions.

## Features

- **HDR Format Support**: HDR10, HDR10+, Dolby Vision, HLG, and PQ
- **Tone Mapping Algorithms**: Reinhard, Hable/Filmic, ACES, BT.2390 EETF, and Clip
- **Color Space Conversion**: BT.709, BT.2020, Display P3, DCI-P3, sRGB
- **Transfer Functions**: PQ (SMPTE ST 2084), HLG (ARIB STD-B67), Gamma 2.2/2.4, sRGB
- **Metadata Handling**: Parse and generate HDR10 SEI, HDR10+ dynamic metadata, Dolby Vision RPU

## Key Types

| Type | Description |
|------|-------------|
| `HdrPipeline` | Main processing pipeline for HDR conversion |
| `HdrConfig` | Configuration for target format, color space, and tone mapping |
| `HdrMetadata` | HDR metadata including format, color space, and light levels |
| `HdrFormat` | Enum: `Sdr`, `Hdr10`, `Hdr10Plus`, `DolbyVision`, `Hlg`, `Pq` |
| `ColorSpace` | Enum: `Bt709`, `Bt2020`, `DisplayP3`, `DciP3`, `Srgb` |
| `ToneMapAlgorithm` | Enum: `Reinhard`, `Hable`, `Aces`, `Bt2390`, `Clip` |
| `ToneMapper` | Applies tone mapping to pixel data |
| `ColorSpaceConverter` | Converts between color spaces via XYZ |
| `MasterDisplay` | Master display color volume info (primaries, luminance) |

## Usage

### Basic HDR to SDR Conversion

```rust
use transcode_hdr::{HdrPipeline, HdrConfig, HdrMetadata, HdrFormat, ColorSpace, ToneMapAlgorithm};

// Configure pipeline for HDR10 to SDR conversion
let config = HdrConfig {
    target_format: HdrFormat::Sdr,
    target_color_space: ColorSpace::Bt709,
    target_peak_nits: 100.0,
    tone_map_algorithm: ToneMapAlgorithm::Hable,
};

let pipeline = HdrPipeline::new(config);

// Source metadata
let metadata = HdrMetadata {
    format: HdrFormat::Hdr10,
    color_space: ColorSpace::Bt2020,
    max_cll: 1000,
    max_fall: 400,
    ..Default::default()
};

// Process RGB pixel data (linear, f32)
let mut pixels: Vec<f32> = vec![0.5, 0.3, 0.2, /* ... */];
pipeline.process(&mut pixels, &metadata)?;
```

### Tone Mapping

```rust
use transcode_hdr::{ToneMapper, ToneMapAlgorithm};

let mapper = ToneMapper::new(
    ToneMapAlgorithm::Aces,
    1000.0,  // source peak (nits)
    100.0,   // target peak (nits)
);

let mut pixels = vec![0.0, 0.5, 1.0, 10.0];
mapper.apply(&mut pixels);
```

### Color Space Conversion

```rust
use transcode_hdr::{ColorSpaceConverter, ColorSpace};

let converter = ColorSpaceConverter::new(ColorSpace::Bt2020, ColorSpace::Bt709);
let (r, g, b) = converter.convert(0.5, 0.3, 0.2);
```

### Parsing HDR10 Metadata

```rust
use transcode_hdr::{parse_hdr10_sei, generate_hdr10_sei};

// Parse SEI data
let sei_data: &[u8] = &[0x03, 0xE8, 0x01, 0x90]; // MaxCLL=1000, MaxFALL=400
let metadata = parse_hdr10_sei(sei_data)?;

// Generate SEI data
let sei_bytes = generate_hdr10_sei(&metadata);
```

## Dependencies

- `transcode-core` - Core types and utilities
- `thiserror` - Error handling
- `tracing` - Logging
- `palette` - Color math

## License

See the workspace root for license information.
