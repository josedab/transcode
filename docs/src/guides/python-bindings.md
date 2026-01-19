# Python Bindings

This guide covers using Transcode from Python via the `transcode-py` package.

## Installation

```bash
pip install transcode-py
```

### From Source

```bash
cd transcode-python
pip install maturin
maturin develop
```

## Quick Start

```python
import transcode_py

# Simple transcoding
stats = transcode_py.transcode('input.mp4', 'output.mp4')
print(f"Processed {stats.frames_encoded} frames")
print(f"Compression ratio: {stats.compression_ratio:.2f}x")
```

## TranscodeOptions

Configure transcoding with the builder pattern:

```python
import transcode_py

options = transcode_py.TranscodeOptions()
options = options.input('input.mp4')
options = options.output('output.mp4')
options = options.video_codec('h264')
options = options.video_bitrate(5_000_000)
options = options.audio_codec('aac')
options = options.audio_bitrate(128_000)
options = options.overwrite(True)

transcoder = transcode_py.Transcoder(options)
stats = transcoder.run()
```

### Video Options

```python
options = transcode_py.TranscodeOptions()
options = options.video_codec('h264')      # h264, hevc, av1, vp9
options = options.video_bitrate(5_000_000) # 5 Mbps
options = options.crf(23)                  # Quality (lower = better)
options = options.preset('medium')         # ultrafast to veryslow
options = options.resolution(1920, 1080)   # Output resolution
options = options.framerate(30.0)          # Output framerate
```

### Audio Options

```python
options = options.audio_codec('aac')       # aac, opus, flac
options = options.audio_bitrate(128_000)   # 128 kbps
options = options.audio_channels(2)        # Stereo
options = options.sample_rate(48000)       # 48 kHz
```

## Transcoder

```python
import transcode_py

options = transcode_py.TranscodeOptions()
options = options.input('input.mp4')
options = options.output('output.mp4')

transcoder = transcode_py.Transcoder(options)

# Run transcoding
stats = transcoder.run()

# Or with progress callback
def progress_callback(progress):
    print(f"Progress: {progress * 100:.1f}%")

stats = transcoder.run_with_progress(progress_callback)
```

### Statistics

```python
stats = transcoder.run()

print(f"Frames encoded: {stats.frames_encoded}")
print(f"Duration: {stats.duration_secs:.2f}s")
print(f"Input size: {stats.input_size} bytes")
print(f"Output size: {stats.output_size} bytes")
print(f"Compression ratio: {stats.compression_ratio:.2f}x")
print(f"Encoding speed: {stats.fps:.1f} fps")
```

## SIMD Detection

```python
import transcode_py

caps = transcode_py.detect_simd()

print(f"SSE4.2: {caps.sse42}")
print(f"AVX2: {caps.avx2}")
print(f"AVX-512: {caps.avx512}")
print(f"NEON: {caps.neon}")
print(f"Best level: {caps.best_level()}")
```

## Quality Metrics

```python
import transcode_py

# Calculate PSNR
psnr = transcode_py.calculate_psnr('reference.mp4', 'distorted.mp4')
print(f"PSNR: {psnr:.2f} dB")

# Calculate SSIM
ssim = transcode_py.calculate_ssim('reference.mp4', 'distorted.mp4')
print(f"SSIM: {ssim:.4f}")

# Full quality assessment
report = transcode_py.quality_assessment('reference.mp4', 'distorted.mp4')
print(f"PSNR: {report.psnr:.2f} dB")
print(f"SSIM: {report.ssim:.4f}")
print(f"VMAF: {report.vmaf:.2f}")
```

## Async Support

```python
import asyncio
import transcode_py

async def transcode_async():
    options = transcode_py.TranscodeOptions()
    options = options.input('input.mp4')
    options = options.output('output.mp4')

    transcoder = transcode_py.AsyncTranscoder(options)
    stats = await transcoder.run()
    return stats

# Run async
stats = asyncio.run(transcode_async())
```

## Batch Processing

```python
import transcode_py
from concurrent.futures import ThreadPoolExecutor

def transcode_file(input_path, output_path):
    options = transcode_py.TranscodeOptions()
    options = options.input(input_path)
    options = options.output(output_path)
    options = options.video_bitrate(5_000_000)

    transcoder = transcode_py.Transcoder(options)
    return transcoder.run()

# Process multiple files
files = [
    ('input1.mp4', 'output1.mp4'),
    ('input2.mp4', 'output2.mp4'),
    ('input3.mp4', 'output3.mp4'),
]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(
        lambda x: transcode_file(x[0], x[1]),
        files
    ))
```

## Error Handling

```python
import transcode_py
from transcode_py import TranscodeError

try:
    stats = transcode_py.transcode('input.mp4', 'output.mp4')
except TranscodeError as e:
    print(f"Transcoding failed: {e}")
except FileNotFoundError:
    print("Input file not found")
except PermissionError:
    print("Permission denied")
```

## Frame Access

```python
import transcode_py

# Read frames from video
reader = transcode_py.FrameReader('input.mp4')

for frame in reader:
    # Access frame data as numpy array
    data = frame.to_numpy()
    print(f"Frame {frame.index}: {data.shape}")

    # Access individual planes
    y_plane = frame.plane(0)
    u_plane = frame.plane(1)
    v_plane = frame.plane(2)
```

## NumPy Integration

```python
import numpy as np
import transcode_py

# Create frame from numpy array
data = np.zeros((1080, 1920, 3), dtype=np.uint8)
frame = transcode_py.Frame.from_numpy(data, 'rgb24')

# Convert frame to numpy
reader = transcode_py.FrameReader('input.mp4')
frame = next(iter(reader))
array = frame.to_numpy()
```

## Command-Line Interface

The Python package also provides a CLI:

```bash
# Transcode a video
python -m transcode_py transcode input.mp4 output.mp4 --video-bitrate 5000000

# Show video info
python -m transcode_py info input.mp4

# Calculate quality metrics
python -m transcode_py quality reference.mp4 distorted.mp4
```

## Type Hints

Full type hints are provided:

```python
from transcode_py import (
    Transcoder,
    TranscodeOptions,
    TranscodeStats,
    SimdCapabilities,
)

def process_video(input_path: str, output_path: str) -> TranscodeStats:
    options: TranscodeOptions = TranscodeOptions()
    options = options.input(input_path)
    options = options.output(output_path)

    transcoder: Transcoder = Transcoder(options)
    return transcoder.run()
```

## Performance Tips

1. **Reuse TranscodeOptions** for similar jobs
2. **Use ThreadPoolExecutor** for batch processing
3. **Enable SIMD** - it's automatic but verify with `detect_simd()`
4. **Use CRF** instead of bitrate for quality-based encoding
