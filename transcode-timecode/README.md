# transcode-timecode

SMPTE 12M timecode library for Rust, providing comprehensive timecode support for video production workflows.

## Features

- **SMPTE Timecode**: Standard HH:MM:SS:FF format with validation and arithmetic
- **Drop-Frame Timecode**: Accurate wall-clock synchronization for 29.97/59.94 fps
- **LTC (Linear Timecode)**: Audio encoding/decoding using bi-phase modulation
- **VITC (Vertical Interval Timecode)**: Video line embedding for VBI

## Supported Frame Rates

| Frame Rate | Description | Drop-Frame |
|------------|-------------|------------|
| 23.976 fps | NTSC film (24000/1001) | No |
| 24 fps | Film | No |
| 25 fps | PAL | No |
| 29.97 fps | NTSC (30000/1001) | Yes |
| 30 fps | HD | No |
| 48 fps | HFR film | No |
| 50 fps | PAL HD | No |
| 59.94 fps | NTSC HD (60000/1001) | Yes |
| 60 fps | HD | No |
| Custom | User-defined rational | No |

## Key Types

- `Timecode` - SMPTE timecode with hours, minutes, seconds, frames
- `FrameRate` - Standard and custom frame rates
- `LtcFrame` / `LtcEncoder` / `LtcDecoder` - Linear timecode audio processing
- `VitcData` / `VitcEncoder` / `VitcDecoder` - Vertical interval timecode
- `DropFrameConfig` - Drop-frame calculation parameters

## Usage

### Basic Timecode

```rust
use transcode_timecode::{Timecode, FrameRate, timecode};

// Create timecode
let tc = Timecode::new(1, 30, 45, 12, FrameRate::Fps24).unwrap();
println!("{}", tc); // 01:30:45:12

// Parse from string
let tc: Timecode = "01:30:45:12".parse().unwrap();

// Convert to/from frame number
let frame_num = tc.to_frame_number();
let tc2 = Timecode::from_frame_number(frame_num, FrameRate::Fps24, false);

// Arithmetic
let tc3 = tc.add_frames(100).unwrap();
```

### Drop-Frame Timecode

```rust
use transcode_timecode::{Timecode, FrameRate, timecode_df};

// Create drop-frame timecode (semicolon separator)
let tc = Timecode::new_drop_frame(1, 0, 0, 2, FrameRate::Fps29_97).unwrap();
println!("{}", tc); // 01:00:00;02

// Parse drop-frame from string
let tc: Timecode = "01:00:00;02".parse().unwrap();
assert!(tc.drop_frame);

// Wall-clock time conversion
use transcode_timecode::{drop_frame_to_wall_time, wall_time_to_drop_frame};
let wall_seconds = drop_frame_to_wall_time(&tc);
let tc2 = wall_time_to_drop_frame(3600.0, FrameRate::Fps29_97);
```

### LTC Audio Encoding

```rust
use transcode_timecode::{Timecode, FrameRate, ltc::{LtcEncoder, LtcFrame}};

let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps24).unwrap();
let mut encoder = LtcEncoder::new(48000, FrameRate::Fps24);

let frame = LtcFrame::from_timecode(&tc);
let samples: Vec<f32> = encoder.encode_frame(&frame);
```

### VITC Video Embedding

```rust
use transcode_timecode::{Timecode, FrameRate, vitc::{VitcEncoder, VitcData}};

let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps24).unwrap();
let encoder = VitcEncoder::new(720); // line width

let vitc = VitcData::from_timecode(&tc);
let pixels: Vec<u8> = encoder.encode(&vitc);
```

## SMPTE Standard

This library implements SMPTE 12M-2008 timecode specification.

## License

See workspace LICENSE file.
