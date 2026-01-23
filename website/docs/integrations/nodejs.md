---
sidebar_position: 2
title: Node.js
description: Use Transcode from Node.js with native bindings
---

# Node.js Integration

Transcode provides native Node.js bindings through N-API, delivering Rust performance in JavaScript applications.

## Installation

```bash
npm install @transcode/node
```

Or with yarn:
```bash
yarn add @transcode/node
```

## Quick Start

```javascript
const transcode = require('@transcode/node');

// Simple transcode
await transcode.transcode({
  input: 'input.mp4',
  output: 'output.mp4',
  videoBitrate: 5_000_000
});
```

## ES Modules

```javascript
import { transcode, probe, Transcoder } from '@transcode/node';

const result = await transcode({
  input: 'input.mp4',
  output: 'output.mp4'
});
```

## Basic Usage

### Transcode with Options

```javascript
import { transcode, VideoCodec, AudioCodec } from '@transcode/node';

const result = await transcode({
  input: 'input.mp4',
  output: 'output.mp4',
  videoCodec: VideoCodec.H264,
  videoBitrate: 5_000_000,
  audioCodec: AudioCodec.AAC,
  audioBitrate: 128_000,
});

console.log(`Transcoded in ${result.duration.toFixed(2)}s`);
console.log(`Output size: ${(result.outputSize / 1_000_000).toFixed(1)} MB`);
```

### Progress Events

```javascript
const result = await transcode({
  input: 'input.mp4',
  output: 'output.mp4',
  onProgress: (progress) => {
    process.stdout.write(
      `\rProgress: ${progress.percent.toFixed(1)}% ` +
      `(${progress.framesDone}/${progress.totalFrames} frames)`
    );
  }
});

console.log('\nDone!');
```

### Get Media Info

```javascript
import { probe } from '@transcode/node';

const info = await probe('video.mp4');

console.log(`Duration: ${info.duration.toFixed(2)}s`);
console.log(`Container: ${info.format}`);

for (const stream of info.videoStreams) {
  console.log(`Video: ${stream.codec} ${stream.width}x${stream.height} @ ${stream.fps}fps`);
}

for (const stream of info.audioStreams) {
  console.log(`Audio: ${stream.codec} ${stream.sampleRate}Hz ${stream.channels}ch`);
}
```

## Transcoder Class

For more control, use the `Transcoder` class:

```javascript
import { Transcoder } from '@transcode/node';

const transcoder = new Transcoder({
  input: 'input.mp4',
  output: 'output.mp4',
  videoBitrate: 5_000_000,
  videoCodec: 'h264',
  audioBitrate: 128_000,
  audioCodec: 'aac',
});

// Event-based progress
transcoder.on('progress', (progress) => {
  console.log(`Frame ${progress.frame}/${progress.totalFrames}`);
});

transcoder.on('complete', (result) => {
  console.log('Transcoding complete!', result);
});

transcoder.on('error', (err) => {
  console.error('Transcoding failed:', err);
});

await transcoder.run();
```

### Cancel Transcoding

```javascript
const transcoder = new Transcoder(options);

// Cancel after 10 seconds
setTimeout(() => {
  transcoder.cancel();
}, 10_000);

try {
  await transcoder.run();
} catch (err) {
  if (err.code === 'CANCELLED') {
    console.log('Transcoding was cancelled');
  } else {
    throw err;
  }
}
```

## Video Filters

### Scaling

```javascript
import { Transcoder, ScaleFilter } from '@transcode/node';

const transcoder = new Transcoder(options);
transcoder.addFilter(new ScaleFilter(1920, 1080));
```

### Crop

```javascript
import { CropFilter } from '@transcode/node';

// Crop to 1280x720 starting at position (100, 50)
transcoder.addFilter(new CropFilter(1280, 720, { x: 100, y: 50 }));
```

### Frame Rate

```javascript
import { FrameRateFilter } from '@transcode/node';

transcoder.addFilter(new FrameRateFilter(60));  // Convert to 60fps
```

### Color Adjustment

```javascript
import { ColorFilter } from '@transcode/node';

transcoder.addFilter(new ColorFilter({
  brightness: 0.1,    // -1.0 to 1.0
  contrast: 1.1,      // 0.0 to 2.0
  saturation: 1.05,   // 0.0 to 2.0
}));
```

### Chain Multiple Filters

```javascript
transcoder.addFilter(new ScaleFilter(1920, 1080));
transcoder.addFilter(new ColorFilter({ brightness: 0.1 }));
transcoder.addFilter(new FrameRateFilter(60));
```

## Audio Processing

### Audio Filters

```javascript
import { VolumeFilter, ResampleFilter } from '@transcode/node';

// Adjust volume
transcoder.addAudioFilter(new VolumeFilter(0.8));  // 80% volume

// Resample audio
transcoder.addAudioFilter(new ResampleFilter(48000));  // 48kHz
```

### Extract Audio

```javascript
await transcode({
  input: 'video.mp4',
  output: 'audio.mp3',
  video: false,  // Disable video
  audioCodec: 'mp3',
  audioBitrate: 320_000,
});
```

## Streaming Output

### HLS

```javascript
import { HlsOutput } from '@transcode/node';

await transcode({
  input: 'input.mp4',
  output: new HlsOutput({
    directory: 'output/hls',
    segmentDuration: 6,
    variants: [
      { resolution: [1920, 1080], bitrate: 5_000_000 },
      { resolution: [1280, 720], bitrate: 2_500_000 },
      { resolution: [854, 480], bitrate: 1_000_000 },
    ]
  })
});
```

### DASH

```javascript
import { DashOutput } from '@transcode/node';

await transcode({
  input: 'input.mp4',
  output: new DashOutput({
    directory: 'output/dash',
    segmentDuration: 4,
  })
});
```

## Streams Support

### Read from Stream

```javascript
import { createReadStream } from 'fs';
import { Transcoder } from '@transcode/node';

const inputStream = createReadStream('input.mp4');

const transcoder = new Transcoder({
  input: inputStream,
  output: 'output.mp4',
});

await transcoder.run();
```

### Write to Stream

```javascript
import { createWriteStream } from 'fs';
import { Transcoder } from '@transcode/node';

const outputStream = createWriteStream('output.mp4');

const transcoder = new Transcoder({
  input: 'input.mp4',
  output: outputStream,
});

await transcoder.run();
```

### Transform Stream

```javascript
import { TranscodeStream } from '@transcode/node';
import { createReadStream, createWriteStream } from 'fs';
import { pipeline } from 'stream/promises';

const transcodeStream = new TranscodeStream({
  videoBitrate: 5_000_000,
  videoCodec: 'h264',
});

await pipeline(
  createReadStream('input.mp4'),
  transcodeStream,
  createWriteStream('output.mp4')
);
```

## Quality Metrics

```javascript
import { compare, Metric } from '@transcode/node/quality';

// Compare original and compressed
const report = await compare({
  reference: 'original.mp4',
  compressed: 'compressed.mp4',
  metrics: [Metric.PSNR, Metric.SSIM, Metric.VMAF]
});

console.log(`PSNR: ${report.psnr.average.toFixed(2)} dB`);
console.log(`SSIM: ${report.ssim.average.toFixed(4)}`);
console.log(`VMAF: ${report.vmaf.average.toFixed(2)}`);

// Per-frame scores
for (const frame of report.frames) {
  console.log(`Frame ${frame.number}: VMAF=${frame.vmaf.toFixed(2)}`);
}
```

## GPU Acceleration

```javascript
import { Transcoder, GpuContext } from '@transcode/node';

// Check GPU availability
if (await GpuContext.isAvailable()) {
  const gpu = await GpuContext.create();
  console.log(`Using GPU: ${gpu.deviceName}`);

  const transcoder = new Transcoder({ ...options, gpu });
  await transcoder.run();
} else {
  console.log('No GPU available, using CPU');
  const transcoder = new Transcoder(options);
  await transcoder.run();
}
```

## Express.js Integration

### Upload and Transcode

```javascript
import express from 'express';
import multer from 'multer';
import { transcode } from '@transcode/node';
import path from 'path';

const app = express();
const upload = multer({ dest: 'uploads/' });

app.post('/transcode', upload.single('video'), async (req, res) => {
  try {
    const inputPath = req.file.path;
    const outputPath = path.join('outputs', `${req.file.filename}.mp4`);

    const result = await transcode({
      input: inputPath,
      output: outputPath,
      videoBitrate: 5_000_000,
    });

    res.json({
      success: true,
      outputPath,
      duration: result.duration,
      outputSize: result.outputSize,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(3000);
```

### Progress via Server-Sent Events

```javascript
app.get('/transcode/:id/progress', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const transcoder = activeJobs.get(req.params.id);

  transcoder.on('progress', (progress) => {
    res.write(`data: ${JSON.stringify(progress)}\n\n`);
  });

  transcoder.on('complete', (result) => {
    res.write(`data: ${JSON.stringify({ complete: true, result })}\n\n`);
    res.end();
  });

  req.on('close', () => {
    // Client disconnected
  });
});
```

## Worker Threads

For CPU-intensive work, use worker threads:

```javascript
// worker.js
import { parentPort, workerData } from 'worker_threads';
import { transcode } from '@transcode/node';

const result = await transcode(workerData);
parentPort.postMessage(result);
```

```javascript
// main.js
import { Worker } from 'worker_threads';

function transcodeInWorker(options) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./worker.js', { workerData: options });
    worker.on('message', resolve);
    worker.on('error', reject);
  });
}

const result = await transcodeInWorker({
  input: 'input.mp4',
  output: 'output.mp4',
});
```

## Batch Processing

```javascript
import { transcode } from '@transcode/node';
import { glob } from 'glob';
import pLimit from 'p-limit';
import path from 'path';

const limit = pLimit(4);  // Process 4 files at a time

const inputFiles = await glob('videos/*.avi');

const results = await Promise.all(
  inputFiles.map(inputPath =>
    limit(async () => {
      const outputPath = inputPath.replace(/\.avi$/, '.mp4');
      return transcode({
        input: inputPath,
        output: outputPath,
        videoBitrate: 5_000_000,
      });
    })
  )
);

console.log(`Transcoded ${results.length} files`);
```

## TypeScript

Full TypeScript support with type definitions:

```typescript
import {
  transcode,
  Transcoder,
  TranscodeOptions,
  TranscodeResult,
  Progress,
  MediaInfo,
  VideoStream,
  AudioStream,
} from '@transcode/node';

async function processVideo(path: string): Promise<TranscodeResult> {
  const options: TranscodeOptions = {
    input: path,
    output: 'output.mp4',
  };

  return transcode(options);
}
```

## Error Handling

```javascript
import {
  TranscodeError,
  CodecError,
  FileNotFoundError,
  UnsupportedFormatError,
} from '@transcode/node';

try {
  await transcode({ input: 'input.mp4', output: 'output.mp4' });
} catch (err) {
  if (err instanceof FileNotFoundError) {
    console.log(`File not found: ${err.path}`);
  } else if (err instanceof UnsupportedFormatError) {
    console.log(`Unsupported format: ${err.format}`);
  } else if (err instanceof CodecError) {
    console.log(`Codec error: ${err.message}`);
  } else if (err instanceof TranscodeError) {
    console.log(`Transcode error: ${err.message}`);
  } else {
    throw err;
  }
}
```

## Configuration

### Environment Variables

```javascript
// Set before requiring the module
process.env.TRANSCODE_THREADS = '8';
process.env.TRANSCODE_LOG_LEVEL = 'debug';

const transcode = require('@transcode/node');
```

### Config Object

```javascript
import { configure } from '@transcode/node';

configure({
  threads: 8,
  logLevel: 'debug',
  tempDir: '/tmp/transcode',
});
```

## Next Steps

- [WebAssembly](/docs/integrations/webassembly) - Browser usage
- [Python Integration](/docs/integrations/python) - Python bindings
- [CLI Reference](/docs/reference/cli) - Command-line interface
