---
sidebar_position: 3
title: Hardware Acceleration
description: GPU-accelerated encoding and decoding
---

# Hardware Acceleration

Transcode supports hardware-accelerated video encoding and decoding using GPU hardware encoders.

## Overview

Hardware acceleration offloads video encoding/decoding to dedicated hardware:

| Feature | CPU | Hardware |
|---------|-----|----------|
| Speed | Baseline | 2-10x faster |
| Power | Higher | Lower |
| Quality (same bitrate) | Higher | Slightly lower |
| Parallel tasks | Limited by cores | Multiple streams |

## Supported Hardware

### NVIDIA (NVENC/NVDEC)

| GPU Generation | Encode | Decode | AV1 |
|----------------|:------:|:------:|:---:|
| Turing (RTX 20) | ✅ | ✅ | ❌ |
| Ampere (RTX 30) | ✅ | ✅ | ❌ |
| Ada (RTX 40) | ✅ | ✅ | ✅ |

### Intel (Quick Sync)

| Generation | Encode | Decode | AV1 |
|------------|:------:|:------:|:---:|
| 10th Gen+ | ✅ | ✅ | ❌ |
| 11th Gen+ | ✅ | ✅ | ❌ |
| Arc | ✅ | ✅ | ✅ |

### AMD (VCE/VCN)

| Generation | Encode | Decode |
|------------|:------:|:------:|
| RDNA 2+ | ✅ | ✅ |
| RDNA 3 | ✅ | ✅ |

### Apple (VideoToolbox)

| Platform | Encode | Decode |
|----------|:------:|:------:|
| Apple Silicon | ✅ | ✅ |
| Intel Mac | ✅ | ✅ |

## Setup

### Enable Feature

```toml
[dependencies]
transcode = { version = "1.0", features = ["hwaccel"] }
```

### Check Availability

```rust
use transcode::hwaccel::{HardwareAccel, HwAccelType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let available = HardwareAccel::available();

    for hw in &available {
        println!("{:?}:", hw.hw_type());
        println!("  Device: {}", hw.device_name());
        println!("  Encode: {:?}", hw.supported_encode_codecs());
        println!("  Decode: {:?}", hw.supported_decode_codecs());
    }

    Ok(())
}
```

## NVIDIA NVENC

### Basic Usage

```rust
use transcode::hwaccel::nvenc::{NvencEncoder, NvencConfig};
use transcode_codecs::h264::H264EncoderConfig;

let config = NvencConfig {
    device: 0,  // GPU index
    preset: NvencPreset::P4,  // Quality/speed balance
    tuning: NvencTuning::HighQuality,
    rc_mode: NvencRateControl::Vbr,
    bitrate: 5_000_000,
    max_bitrate: 10_000_000,
    ..Default::default()
};

let mut encoder = NvencEncoder::new_h264(config)?;

for frame in frames {
    let packets = encoder.encode(&frame)?;
    for packet in packets {
        muxer.write_packet(&packet)?;
    }
}
```

### NVENC Presets

| Preset | Quality | Speed | Use Case |
|--------|---------|-------|----------|
| P1 | Lowest | Fastest | Real-time streaming |
| P2 | Low | Faster | Fast encoding |
| P3 | Medium-Low | Fast | Balanced streaming |
| P4 | Medium | Default | General purpose |
| P5 | Medium-High | Slow | High quality streaming |
| P6 | High | Slower | Quality priority |
| P7 | Highest | Slowest | Best quality |

### NVENC Tuning

```rust
let config = NvencConfig {
    tuning: NvencTuning::HighQuality,  // Best visual quality
    // tuning: NvencTuning::LowLatency,   // Streaming
    // tuning: NvencTuning::UltraLowLatency, // Real-time
    // tuning: NvencTuning::Lossless,     // Lossless encoding
    ..Default::default()
};
```

### HEVC with NVENC

```rust
use transcode::hwaccel::nvenc::NvencEncoder;

let config = NvencConfig {
    preset: NvencPreset::P5,
    bitrate: 3_000_000,  // Lower bitrate than H.264
    ..Default::default()
};

let mut encoder = NvencEncoder::new_hevc(config)?;
```

### AV1 with NVENC (Ada+)

```rust
use transcode::hwaccel::nvenc::NvencEncoder;

// Requires RTX 40 series or newer
let config = NvencConfig {
    preset: NvencPreset::P4,
    bitrate: 2_500_000,
    ..Default::default()
};

let mut encoder = NvencEncoder::new_av1(config)?;
```

### NVDEC Decoding

```rust
use transcode::hwaccel::nvdec::{NvdecDecoder, NvdecConfig};

let config = NvdecConfig {
    device: 0,
    output_format: OutputFormat::Nv12,
    ..Default::default()
};

let mut decoder = NvdecDecoder::new_h264(config)?;

for packet in packets {
    if let Some(frame) = decoder.decode(&packet)? {
        // Frame is in GPU memory
        // Can be passed directly to NVENC for transcoding
    }
}
```

## Intel Quick Sync

### Basic Usage

```rust
use transcode::hwaccel::qsv::{QsvEncoder, QsvConfig};

let config = QsvConfig {
    target_usage: QsvTargetUsage::Balanced,
    bitrate: 5_000_000,
    ..Default::default()
};

let mut encoder = QsvEncoder::new_h264(config)?;
```

### QSV Target Usage

| Level | Quality | Speed |
|-------|---------|-------|
| Quality | Highest | Slowest |
| Balanced | Medium | Default |
| Speed | Lowest | Fastest |

```rust
let config = QsvConfig {
    target_usage: QsvTargetUsage::Quality,
    ..Default::default()
};
```

### QSV Decoding

```rust
use transcode::hwaccel::qsv::{QsvDecoder, QsvConfig};

let config = QsvConfig::default();
let mut decoder = QsvDecoder::new_h264(config)?;
```

## Apple VideoToolbox

### Basic Usage

```rust
use transcode::hwaccel::videotoolbox::{VtEncoder, VtConfig};

let config = VtConfig {
    bitrate: 5_000_000,
    realtime: false,
    allow_frame_reordering: true,
    ..Default::default()
};

let mut encoder = VtEncoder::new_h264(config)?;
```

### HEVC with VideoToolbox

```rust
let config = VtConfig {
    bitrate: 3_000_000,
    profile: VtHevcProfile::Main10,  // 10-bit HDR
    ..Default::default()
};

let mut encoder = VtEncoder::new_hevc(config)?;
```

### Hardware vs Software

```rust
let config = VtConfig {
    require_hardware: true,  // Force hardware encoder
    // require_hardware: false,  // Allow software fallback
    ..Default::default()
};
```

## VA-API (Linux)

### Basic Usage

```rust
use transcode::hwaccel::vaapi::{VaapiEncoder, VaapiConfig};

let config = VaapiConfig {
    device: "/dev/dri/renderD128",
    bitrate: 5_000_000,
    ..Default::default()
};

let mut encoder = VaapiEncoder::new_h264(config)?;
```

### List VA-API Devices

```rust
use transcode::hwaccel::vaapi::VaapiDevice;

for device in VaapiDevice::enumerate()? {
    println!("Device: {}", device.path());
    println!("  Vendor: {}", device.vendor());
    println!("  Encode: {:?}", device.encode_profiles());
    println!("  Decode: {:?}", device.decode_profiles());
}
```

## Using Hardware in Transcoding Pipeline

### CLI

```bash
# Auto-detect hardware
transcode -i input.mp4 -o output.mp4 --hwaccel auto

# Force NVENC
transcode -i input.mp4 -o output.mp4 --hwaccel nvenc --video-codec h264

# Intel Quick Sync
transcode -i input.mp4 -o output.mp4 --hwaccel qsv --video-codec h264

# Apple VideoToolbox
transcode -i input.mp4 -o output.mp4 --hwaccel videotoolbox

# Specific GPU device
transcode -i input.mp4 -o output.mp4 --hwaccel nvenc --hwaccel-device 1
```

### API

```rust
use transcode::{Transcoder, TranscodeOptions, HwAccel};

let options = TranscodeOptions::new()
    .input("input.mp4")
    .output("output.mp4")
    .hwaccel(HwAccel::Auto)  // Auto-detect
    // .hwaccel(HwAccel::Nvenc { device: 0 })
    // .hwaccel(HwAccel::Qsv)
    // .hwaccel(HwAccel::VideoToolbox)
    .video_bitrate(5_000_000);

let mut transcoder = Transcoder::new(options)?;
transcoder.run()?;
```

### Full Hardware Pipeline

Zero-copy transcoding (decode + encode on GPU):

```rust
use transcode::hwaccel::{NvdecDecoder, NvencEncoder, GpuFrame};

let decoder_config = NvdecConfig {
    device: 0,
    ..Default::default()
};

let encoder_config = NvencConfig {
    device: 0,
    preset: NvencPreset::P4,
    bitrate: 5_000_000,
    ..Default::default()
};

let mut decoder = NvdecDecoder::new_h264(decoder_config)?;
let mut encoder = NvencEncoder::new_h264(encoder_config)?;

for packet in input_packets {
    if let Some(gpu_frame) = decoder.decode(&packet)? {
        // Frame stays in GPU memory
        let output_packets = encoder.encode_gpu(&gpu_frame)?;
        for pkt in output_packets {
            muxer.write_packet(&pkt)?;
        }
    }
}
```

## Performance Tuning

### Lookahead

Buffer frames for better quality:

```rust
let config = NvencConfig {
    lookahead: 32,  // Frames to buffer (0-32)
    ..Default::default()
};
```

### B-Frames

```rust
let config = NvencConfig {
    b_frames: 2,    // Number of B-frames
    b_ref_mode: BRefMode::Each,  // Use B-frames as references
    ..Default::default()
};
```

### Reference Frames

```rust
let config = NvencConfig {
    ref_frames: 4,  // Reference frames (1-16)
    ..Default::default()
};
```

### Rate Control

```rust
let config = NvencConfig {
    rc_mode: NvencRateControl::Vbr,
    bitrate: 5_000_000,
    max_bitrate: 10_000_000,
    vbv_buffer_size: 5_000_000,
    ..Default::default()
};

// Constant quality
let config = NvencConfig {
    rc_mode: NvencRateControl::ConstQp,
    qp: 23,
    ..Default::default()
};
```

## Multi-GPU Encoding

### Distribute Across GPUs

```rust
use std::thread;
use transcode::hwaccel::nvenc::{NvencEncoder, NvencConfig};

fn encode_on_gpu(gpu_id: usize, input_file: &str, output_file: &str) {
    let config = NvencConfig {
        device: gpu_id,
        ..Default::default()
    };

    let mut encoder = NvencEncoder::new_h264(config).unwrap();
    // ... encode
}

fn main() {
    let files = [
        ("input1.mp4", "output1.mp4"),
        ("input2.mp4", "output2.mp4"),
        ("input3.mp4", "output3.mp4"),
        ("input4.mp4", "output4.mp4"),
    ];

    let num_gpus = 2;

    let handles: Vec<_> = files
        .iter()
        .enumerate()
        .map(|(i, (input, output))| {
            let gpu_id = i % num_gpus;
            let input = input.to_string();
            let output = output.to_string();

            thread::spawn(move || {
                encode_on_gpu(gpu_id, &input, &output);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}
```

## Quality Comparison

### Measuring Quality

```rust
use transcode_quality::{compare, Metric};

// Encode with different methods
encode_cpu(&input, "cpu_output.mp4", bitrate)?;
encode_nvenc(&input, "nvenc_output.mp4", bitrate)?;

// Compare quality
let cpu_report = compare(&input, "cpu_output.mp4", &[Metric::Vmaf])?;
let nvenc_report = compare(&input, "nvenc_output.mp4", &[Metric::Vmaf])?;

println!("CPU VMAF: {:.2}", cpu_report.vmaf.average);
println!("NVENC VMAF: {:.2}", nvenc_report.vmaf.average);
```

### Typical Quality Results

At the same bitrate (5 Mbps H.264):

| Encoder | VMAF | Speed |
|---------|------|-------|
| x264 slow | 92.5 | 1x |
| x264 medium | 91.0 | 3x |
| NVENC P7 | 89.5 | 15x |
| NVENC P4 | 88.0 | 25x |
| QSV Quality | 88.5 | 20x |

## Troubleshooting

### NVENC Errors

```rust
match NvencEncoder::new_h264(config) {
    Ok(encoder) => { /* success */ }
    Err(NvencError::DriverNotFound) => {
        println!("NVIDIA driver not found");
    }
    Err(NvencError::NoCapableDevice) => {
        println!("No NVENC-capable GPU found");
    }
    Err(NvencError::EncoderBusy) => {
        println!("Too many concurrent NVENC sessions");
    }
    Err(e) => println!("Error: {}", e),
}
```

### Session Limits

NVIDIA GPUs have limits on concurrent NVENC sessions:

| GPU Type | Max Sessions |
|----------|-------------|
| GeForce | 3-5 |
| Quadro | Unlimited |
| Tesla | Unlimited |

### Checking GPU Memory

```rust
use transcode::hwaccel::nvenc::NvencDevice;

let device = NvencDevice::new(0)?;
let (total, free) = device.memory_info()?;

println!("GPU Memory: {} / {} MB", free / 1_000_000, total / 1_000_000);
```

## Next Steps

- [SIMD Optimization](/docs/advanced/simd-optimization) - CPU optimization
- [GPU Acceleration](/docs/guides/gpu-acceleration) - GPU processing (non-encoding)
- [Distributed Processing](/docs/guides/distributed-processing) - Multi-machine encoding
