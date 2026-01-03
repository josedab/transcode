# transcode-hevc

HEVC/H.265 codec implementation for the transcode library.

This crate provides a complete HEVC (High Efficiency Video Coding) encoder and decoder following the ITU-T H.265 specification. HEVC offers approximately 50% better compression efficiency compared to H.264/AVC while maintaining similar visual quality.

## Features

- **NAL Unit Parsing**: VPS, SPS, PPS, IDR, TRAIL, TSA, STSA, RADL, RASL, BLA, CRA, and SEI units
- **Profile Support**: Main, Main10, Main Still Picture, and Range Extensions
- **CABAC Entropy Coding**: Full context-adaptive binary arithmetic coding
- **Transform/Quantization**: DCT-II and DST-VII transforms (4x4 to 32x32)
- **Intra Prediction**: All 35 modes (Planar, DC, Angular 2-34)
- **Inter Prediction**: Quarter-sample precision, bi-directional and weighted prediction
- **In-Loop Filtering**: Deblocking filter and SAO (band offset and edge offset)

## Key Types

### Encoder

- `HevcEncoder` - Main encoder interface
- `HevcEncoderConfig` - Encoder configuration (resolution, framerate, profile, rate control)
- `HevcPreset` - Speed/quality presets (Ultrafast to Placebo)
- `RateControlMode` - CQP, ABR, CBR, or CRF rate control

### Decoder

- `HevcDecoder` - Main decoder interface
- `HevcDecoderConfig` - Decoder configuration (max resolution, output format, threading)

### NAL and Parameter Sets

- `NalUnitType`, `NalUnitHeader` - NAL unit parsing
- `Vps`, `Sps`, `Pps` - Parameter sets
- `SliceSegmentHeader`, `SliceType` - Slice information

### Transform and Entropy

- `HevcTransform`, `HevcQuantizer` - Transform and quantization
- `CabacEncoder`, `CabacDecoder` - Entropy coding

## Usage

### Decoding

```rust
use transcode_hevc::{HevcDecoder, HevcDecoderConfig};

let config = HevcDecoderConfig::default();
let mut decoder = HevcDecoder::new(config)?;

// Decode NAL units from Annex B stream
for nal_data in nal_units {
    if let Some(frame) = decoder.decode_nal(&nal_data)? {
        // Process decoded frame
    }
}

// Flush remaining frames
while let Some(frame) = decoder.flush()? {
    // Process frame
}
```

### Encoding

```rust
use transcode_hevc::{HevcEncoder, HevcEncoderConfig, HevcPreset, RateControlMode};

let config = HevcEncoderConfig::new(1920, 1080)
    .with_preset(HevcPreset::Medium)
    .with_crf(23.0)
    .with_gop_size(60)
    .with_bframes(3);

let mut encoder = HevcEncoder::new(config)?;

// Encode frames
let packets = encoder.encode(&frame)?;

// Flush encoder
let remaining = encoder.flush()?;
```

### Parsing Annex B Streams

```rust
use transcode_hevc::{parse_annexb_stream, is_idr_picture, is_parameter_set};

let nal_units = parse_annexb_stream(&data);
for (header, payload) in nal_units {
    if is_parameter_set(header.nal_unit_type) {
        // Handle VPS/SPS/PPS
    } else if is_idr_picture(header.nal_unit_type) {
        // Handle IDR frame
    }
}
```

## HEVC Architecture

HEVC uses a hierarchical block structure:

- **CTU (Coding Tree Unit)**: Largest block (64x64, 32x32, or 16x16)
- **CU (Coding Unit)**: Variable size from 8x8 to CTU size
- **PU (Prediction Unit)**: Prediction partitioning within a CU
- **TU (Transform Unit)**: Transform partitioning within a CU

## License

See the workspace root for license information.
