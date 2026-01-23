---
sidebar_position: 1
title: Python
description: Use Transcode from Python with PyO3 bindings
---

# Python Integration

Transcode provides native Python bindings through PyO3, giving you the full power of Rust with Python's ease of use.

## Installation

```bash
pip install transcode
```

### From Source

```bash
git clone https://github.com/example/transcode
cd transcode/transcode-python
pip install maturin
maturin develop --release
```

## Quick Start

```python
import transcode

# Simple transcode
transcode.transcode(
    input="input.mp4",
    output="output.mp4",
    video_bitrate=5_000_000
)
```

## Basic Usage

### Transcode with Options

```python
import transcode
from transcode import VideoCodec, AudioCodec, Preset

result = transcode.transcode(
    input="input.mp4",
    output="output.mp4",
    video_codec=VideoCodec.H264,
    video_bitrate=5_000_000,
    audio_codec=AudioCodec.AAC,
    audio_bitrate=128_000,
    preset=Preset.Medium,
)

print(f"Transcoded in {result.duration:.2f}s")
print(f"Output size: {result.output_size / 1_000_000:.1f} MB")
```

### Progress Callback

```python
def on_progress(progress):
    print(f"\rProgress: {progress.percent:.1f}% "
          f"({progress.frames_done}/{progress.total_frames} frames)",
          end="")

result = transcode.transcode(
    input="input.mp4",
    output="output.mp4",
    on_progress=on_progress
)
print("\nDone!")
```

### Get Media Info

```python
info = transcode.probe("video.mp4")

print(f"Duration: {info.duration:.2f}s")
print(f"Container: {info.format}")

for stream in info.video_streams:
    print(f"Video: {stream.codec} {stream.width}x{stream.height} @ {stream.fps}fps")

for stream in info.audio_streams:
    print(f"Audio: {stream.codec} {stream.sample_rate}Hz {stream.channels}ch")
```

## Transcoder Class

For more control, use the `Transcoder` class:

```python
from transcode import Transcoder, TranscodeOptions

options = TranscodeOptions(
    input="input.mp4",
    output="output.mp4",
    video_bitrate=5_000_000,
    video_codec="h264",
    audio_bitrate=128_000,
    audio_codec="aac",
)

transcoder = Transcoder(options)

# Run with progress
for progress in transcoder.run():
    print(f"Frame {progress.frame}/{progress.total_frames}")

print("Transcoding complete!")
```

### Cancel Transcoding

```python
import threading

transcoder = Transcoder(options)

# Cancel after 10 seconds
def cancel_later():
    import time
    time.sleep(10)
    transcoder.cancel()

threading.Thread(target=cancel_later).start()

try:
    for progress in transcoder.run():
        print(f"Progress: {progress.percent:.1f}%")
except transcode.CancelledError:
    print("Transcoding was cancelled")
```

## Video Filters

### Scaling

```python
from transcode import Transcoder, ScaleFilter

transcoder = Transcoder(options)
transcoder.add_filter(ScaleFilter(1920, 1080))
```

### Crop

```python
from transcode import CropFilter

# Crop to 1280x720 starting at position (100, 50)
transcoder.add_filter(CropFilter(1280, 720, x=100, y=50))
```

### Frame Rate

```python
from transcode import FrameRateFilter

transcoder.add_filter(FrameRateFilter(60))  # Convert to 60fps
```

### Color Adjustment

```python
from transcode import ColorFilter

transcoder.add_filter(ColorFilter(
    brightness=0.1,    # -1.0 to 1.0
    contrast=1.1,      # 0.0 to 2.0
    saturation=1.05,   # 0.0 to 2.0
))
```

### Chain Multiple Filters

```python
transcoder.add_filter(ScaleFilter(1920, 1080))
transcoder.add_filter(ColorFilter(brightness=0.1))
transcoder.add_filter(FrameRateFilter(60))
```

## Audio Processing

### Audio Filters

```python
from transcode import VolumeFilter, ResampleFilter

# Adjust volume
transcoder.add_audio_filter(VolumeFilter(0.8))  # 80% volume

# Resample audio
transcoder.add_audio_filter(ResampleFilter(48000))  # 48kHz
```

### Extract Audio

```python
transcode.transcode(
    input="video.mp4",
    output="audio.mp3",
    video=False,  # Disable video
    audio_codec="mp3",
    audio_bitrate=320_000,
)
```

## Streaming Output

### HLS

```python
from transcode import HlsOutput

transcode.transcode(
    input="input.mp4",
    output=HlsOutput(
        directory="output/hls",
        segment_duration=6,
        variants=[
            {"resolution": (1920, 1080), "bitrate": 5_000_000},
            {"resolution": (1280, 720), "bitrate": 2_500_000},
            {"resolution": (854, 480), "bitrate": 1_000_000},
        ]
    )
)
```

### DASH

```python
from transcode import DashOutput

transcode.transcode(
    input="input.mp4",
    output=DashOutput(
        directory="output/dash",
        segment_duration=4,
    )
)
```

## Quality Metrics

```python
from transcode.quality import compare, Metric

# Compare original and compressed
report = compare(
    reference="original.mp4",
    compressed="compressed.mp4",
    metrics=[Metric.PSNR, Metric.SSIM, Metric.VMAF]
)

print(f"PSNR: {report.psnr.average:.2f} dB")
print(f"SSIM: {report.ssim.average:.4f}")
print(f"VMAF: {report.vmaf.average:.2f}")

# Per-frame scores
for frame in report.frames:
    print(f"Frame {frame.number}: VMAF={frame.vmaf:.2f}")
```

## AI Enhancement

```python
from transcode.ai import Upscaler, Denoiser, UpscaleMethod, DenoiseMethod

# Upscale
upscaler = Upscaler(method=UpscaleMethod.Lanczos, scale=4)
transcoder.add_filter(upscaler)

# Denoise
denoiser = Denoiser(method=DenoiseMethod.Bilateral, strength=0.5)
transcoder.add_filter(denoiser)
```

## GPU Acceleration

```python
from transcode import Transcoder, GpuContext

# Check GPU availability
if GpuContext.is_available():
    gpu = GpuContext()
    print(f"Using GPU: {gpu.device_name}")

    transcoder = Transcoder(options, gpu=gpu)
else:
    print("No GPU available, using CPU")
    transcoder = Transcoder(options)
```

## Batch Processing

```python
import transcode
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def transcode_file(input_path):
    output_path = input_path.with_suffix('.mp4')
    transcode.transcode(
        input=str(input_path),
        output=str(output_path),
        video_bitrate=5_000_000
    )
    return output_path

# Process multiple files in parallel
input_files = list(Path("videos").glob("*.avi"))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(transcode_file, input_files))

print(f"Transcoded {len(results)} files")
```

## Async Support

```python
import asyncio
import transcode.asyncio as transcode_async

async def main():
    result = await transcode_async.transcode(
        input="input.mp4",
        output="output.mp4",
        video_bitrate=5_000_000
    )
    print(f"Done: {result.output_size} bytes")

asyncio.run(main())
```

### Async Progress

```python
async def transcode_with_progress():
    async for progress in transcode_async.transcode_iter(
        input="input.mp4",
        output="output.mp4"
    ):
        print(f"Progress: {progress.percent:.1f}%")

asyncio.run(transcode_with_progress())
```

## Error Handling

```python
from transcode import (
    TranscodeError,
    CodecError,
    FileNotFoundError as TranscodeFileNotFound,
    UnsupportedFormatError,
)

try:
    transcode.transcode(input="input.mp4", output="output.mp4")
except TranscodeFileNotFound as e:
    print(f"File not found: {e.path}")
except UnsupportedFormatError as e:
    print(f"Unsupported format: {e.format}")
except CodecError as e:
    print(f"Codec error: {e}")
except TranscodeError as e:
    print(f"Transcode error: {e}")
```

## Type Hints

Full type hint support for IDE completion:

```python
from transcode import (
    Transcoder,
    TranscodeOptions,
    TranscodeResult,
    Progress,
    MediaInfo,
    VideoStream,
    AudioStream,
)

def process_video(path: str) -> TranscodeResult:
    options: TranscodeOptions = TranscodeOptions(
        input=path,
        output="output.mp4",
    )
    transcoder: Transcoder = Transcoder(options)
    return transcoder.run_sync()
```

## NumPy Integration

Access raw frame data as NumPy arrays:

```python
import numpy as np
from transcode import FrameReader

reader = FrameReader("video.mp4")

for frame in reader:
    # frame.data is a NumPy array (height, width, channels)
    array: np.ndarray = frame.data

    # Process with NumPy/OpenCV/etc
    processed = array * 0.5  # Example: reduce brightness

    # frame properties
    print(f"Frame {frame.pts}: {frame.width}x{frame.height}")
```

### Write Frames

```python
from transcode import FrameWriter
import numpy as np

writer = FrameWriter(
    output="output.mp4",
    width=1920,
    height=1080,
    fps=30,
    codec="h264",
    bitrate=5_000_000,
)

for i in range(300):  # 10 seconds at 30fps
    # Create frame as NumPy array
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[:, :, 0] = i % 256  # Red channel animation

    writer.write(frame)

writer.close()
```

## Configuration

### Environment Variables

```python
import os

# Set thread count
os.environ["TRANSCODE_THREADS"] = "8"

# Set log level
os.environ["TRANSCODE_LOG_LEVEL"] = "debug"

import transcode  # Import after setting env vars
```

### Config File

```python
from transcode import Config

config = Config.from_file("transcode.toml")
transcoder = Transcoder(options, config=config)
```

## Next Steps

- [Node.js Integration](/docs/integrations/nodejs) - JavaScript bindings
- [WebAssembly](/docs/integrations/webassembly) - Browser usage
- [CLI Reference](/docs/reference/cli) - Command-line interface
