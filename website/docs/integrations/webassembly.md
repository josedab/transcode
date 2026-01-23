---
sidebar_position: 3
title: WebAssembly
description: Run Transcode in the browser with WebAssembly
---

# WebAssembly Integration

Transcode compiles to WebAssembly, enabling video transcoding directly in the browser without server-side processing.

## Overview

The `transcode-wasm` module provides:

- **Client-side transcoding** - Process video without uploading
- **Web Worker support** - Non-blocking background processing
- **Memory-efficient** - Streaming chunk processing
- **Full codec support** - H.264, H.265, AV1, VP9, and more

## Installation

### NPM

```bash
npm install @transcode/wasm
```

### CDN

```html
<script type="module">
  import init, { Transcoder } from 'https://unpkg.com/@transcode/wasm@latest';
</script>
```

## Quick Start

```javascript
import init, { Transcoder } from '@transcode/wasm';

// Initialize WASM module
await init();

// Create transcoder
const transcoder = new Transcoder({
  videoCodec: 'h264',
  videoBitrate: 5_000_000,
  audioCodec: 'aac',
  audioBitrate: 128_000,
});

// Get file from input
const file = document.getElementById('fileInput').files[0];

// Transcode
const output = await transcoder.transcode(file);

// Download result
const blob = new Blob([output], { type: 'video/mp4' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = 'output.mp4';
a.click();
```

## Basic Usage

### File Input

```html
<input type="file" id="fileInput" accept="video/*">
<button id="transcodeBtn">Transcode</button>
<progress id="progress" value="0" max="100"></progress>
```

```javascript
import init, { Transcoder } from '@transcode/wasm';

await init();

document.getElementById('transcodeBtn').addEventListener('click', async () => {
  const file = document.getElementById('fileInput').files[0];
  const progress = document.getElementById('progress');

  const transcoder = new Transcoder({
    videoBitrate: 5_000_000,
  });

  transcoder.onProgress = (percent) => {
    progress.value = percent;
  };

  const output = await transcoder.transcode(file);
  console.log('Done!', output.byteLength, 'bytes');
});
```

### Drag and Drop

```javascript
const dropZone = document.getElementById('dropZone');

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');

  const file = e.dataTransfer.files[0];
  if (file.type.startsWith('video/')) {
    await processVideo(file);
  }
});
```

## Web Workers

Process video in a background thread to keep the UI responsive.

### Setup Worker

```javascript
// transcoder.worker.js
import init, { Transcoder } from '@transcode/wasm';

let transcoder = null;

self.onmessage = async (e) => {
  const { type, data } = e.data;

  switch (type) {
    case 'init':
      await init();
      transcoder = new Transcoder(data.options);
      transcoder.onProgress = (percent) => {
        self.postMessage({ type: 'progress', percent });
      };
      self.postMessage({ type: 'ready' });
      break;

    case 'transcode':
      try {
        const output = await transcoder.transcode(data.file);
        self.postMessage({ type: 'complete', output }, [output.buffer]);
      } catch (err) {
        self.postMessage({ type: 'error', message: err.message });
      }
      break;

    case 'cancel':
      transcoder.cancel();
      break;
  }
};
```

### Use Worker

```javascript
// main.js
const worker = new Worker(new URL('./transcoder.worker.js', import.meta.url), {
  type: 'module'
});

function transcodeInWorker(file, options) {
  return new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const { type, ...data } = e.data;

      switch (type) {
        case 'ready':
          worker.postMessage({ type: 'transcode', data: { file } });
          break;
        case 'progress':
          console.log(`Progress: ${data.percent.toFixed(1)}%`);
          break;
        case 'complete':
          resolve(data.output);
          break;
        case 'error':
          reject(new Error(data.message));
          break;
      }
    };

    worker.postMessage({ type: 'init', data: { options } });
  });
}

// Usage
const output = await transcodeInWorker(file, { videoBitrate: 5_000_000 });
```

## Streaming Processing

Process large files chunk by chunk to reduce memory usage.

### Chunk-Based Processing

```javascript
import init, { StreamingTranscoder } from '@transcode/wasm';

await init();

const transcoder = new StreamingTranscoder({
  videoBitrate: 5_000_000,
});

// Process in chunks
const CHUNK_SIZE = 1024 * 1024;  // 1MB chunks
const file = document.getElementById('fileInput').files[0];
const reader = file.stream().getReader();

const outputChunks = [];

while (true) {
  const { done, value } = await reader.read();

  if (done) {
    const final = transcoder.finalize();
    outputChunks.push(final);
    break;
  }

  const output = transcoder.processChunk(value);
  if (output.length > 0) {
    outputChunks.push(output);
  }
}

const result = new Uint8Array(
  outputChunks.reduce((acc, chunk) => acc + chunk.length, 0)
);

let offset = 0;
for (const chunk of outputChunks) {
  result.set(chunk, offset);
  offset += chunk.length;
}
```

## Video Filters

### Scaling

```javascript
const transcoder = new Transcoder({
  width: 1920,
  height: 1080,
  scaleMode: 'lanczos',  // nearest, bilinear, bicubic, lanczos
});
```

### Crop

```javascript
const transcoder = new Transcoder({
  crop: {
    x: 100,
    y: 50,
    width: 1280,
    height: 720,
  }
});
```

### Frame Rate

```javascript
const transcoder = new Transcoder({
  fps: 60,
  frameRateMode: 'interpolate',  // drop, duplicate, interpolate
});
```

## Media Info

Get information about a video file:

```javascript
import init, { probe } from '@transcode/wasm';

await init();

const file = document.getElementById('fileInput').files[0];
const info = await probe(file);

console.log(`Duration: ${info.duration.toFixed(2)}s`);
console.log(`Container: ${info.format}`);

for (const stream of info.videoStreams) {
  console.log(`Video: ${stream.codec} ${stream.width}x${stream.height}`);
}

for (const stream of info.audioStreams) {
  console.log(`Audio: ${stream.codec} ${stream.sampleRate}Hz`);
}
```

## Extract Frames

Extract individual frames as images:

```javascript
import init, { extractFrames } from '@transcode/wasm';

await init();

const frames = await extractFrames(file, {
  format: 'jpeg',
  quality: 80,
  timestamps: [0, 5, 10, 15],  // Seconds
});

for (const frame of frames) {
  const img = document.createElement('img');
  img.src = URL.createObjectURL(new Blob([frame.data], { type: 'image/jpeg' }));
  document.body.appendChild(img);
}
```

### Extract at Interval

```javascript
const frames = await extractFrames(file, {
  format: 'png',
  interval: 1,  // Every 1 second
  maxFrames: 10,
});
```

## Create Thumbnails

Generate video thumbnails:

```javascript
import init, { createThumbnail } from '@transcode/wasm';

await init();

const thumbnail = await createThumbnail(file, {
  width: 320,
  height: 180,
  timestamp: 5,  // At 5 seconds
  format: 'jpeg',
  quality: 80,
});

const img = document.createElement('img');
img.src = URL.createObjectURL(new Blob([thumbnail], { type: 'image/jpeg' }));
```

### Thumbnail Strip

Create a sprite sheet of thumbnails:

```javascript
const strip = await createThumbnailStrip(file, {
  width: 160,
  height: 90,
  columns: 10,
  rows: 10,
  interval: 1,  // 1 second between frames
});

const img = document.createElement('img');
img.src = URL.createObjectURL(new Blob([strip], { type: 'image/png' }));
```

## Audio Extraction

Extract audio from video:

```javascript
import init, { extractAudio } from '@transcode/wasm';

await init();

const audio = await extractAudio(file, {
  codec: 'aac',
  bitrate: 128_000,
  format: 'm4a',
});

const blob = new Blob([audio], { type: 'audio/mp4' });
```

## React Integration

```jsx
import { useState, useCallback } from 'react';
import init, { Transcoder } from '@transcode/wasm';

// Initialize once on module load
const wasmReady = init();

function VideoTranscoder() {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('idle');
  const [outputUrl, setOutputUrl] = useState(null);

  const handleTranscode = useCallback(async (file) => {
    await wasmReady;

    setStatus('transcoding');
    setProgress(0);

    const transcoder = new Transcoder({
      videoBitrate: 5_000_000,
    });

    transcoder.onProgress = setProgress;

    try {
      const output = await transcoder.transcode(file);
      const blob = new Blob([output], { type: 'video/mp4' });
      setOutputUrl(URL.createObjectURL(blob));
      setStatus('complete');
    } catch (err) {
      setStatus('error');
      console.error(err);
    }
  }, []);

  return (
    <div>
      <input
        type="file"
        accept="video/*"
        onChange={(e) => handleTranscode(e.target.files[0])}
      />

      {status === 'transcoding' && (
        <progress value={progress} max={100} />
      )}

      {status === 'complete' && outputUrl && (
        <a href={outputUrl} download="output.mp4">Download</a>
      )}
    </div>
  );
}
```

## Vue Integration

```vue
<template>
  <div>
    <input type="file" accept="video/*" @change="handleFile" />
    <progress v-if="transcoding" :value="progress" max="100" />
    <a v-if="outputUrl" :href="outputUrl" download="output.mp4">Download</a>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import init, { Transcoder } from '@transcode/wasm';

const progress = ref(0);
const transcoding = ref(false);
const outputUrl = ref(null);

onMounted(async () => {
  await init();
});

async function handleFile(event) {
  const file = event.target.files[0];
  transcoding.value = true;
  progress.value = 0;

  const transcoder = new Transcoder({
    videoBitrate: 5_000_000,
  });

  transcoder.onProgress = (p) => { progress.value = p; };

  const output = await transcoder.transcode(file);
  const blob = new Blob([output], { type: 'video/mp4' });
  outputUrl.value = URL.createObjectURL(blob);
  transcoding.value = false;
}
</script>
```

## Performance Tips

### Memory Management

```javascript
// Limit memory usage
const transcoder = new Transcoder({
  maxMemoryMb: 512,  // Limit to 512MB
  chunkSize: 1024 * 1024,  // 1MB chunks
});
```

### Use SharedArrayBuffer

For better performance with workers:

```javascript
// Enable cross-origin isolation in your server headers:
// Cross-Origin-Opener-Policy: same-origin
// Cross-Origin-Embedder-Policy: require-corp

const transcoder = new Transcoder({
  useSharedMemory: true,  // Uses SharedArrayBuffer
});
```

### SIMD Support

Check and use SIMD when available:

```javascript
import init, { simdSupported } from '@transcode/wasm';

await init();

if (simdSupported()) {
  console.log('SIMD supported - using optimized path');
}
```

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome 89+ | Full | SIMD, threads, SharedArrayBuffer |
| Firefox 89+ | Full | SIMD, threads, SharedArrayBuffer |
| Safari 15.2+ | Partial | No SharedArrayBuffer in some contexts |
| Edge 89+ | Full | Same as Chrome |

### Feature Detection

```javascript
const features = {
  wasm: typeof WebAssembly !== 'undefined',
  simd: await simdSupported(),
  threads: typeof SharedArrayBuffer !== 'undefined',
  crossOriginIsolated: window.crossOriginIsolated,
};

console.log('Browser features:', features);
```

## Error Handling

```javascript
import { TranscodeError, MemoryError, CodecError } from '@transcode/wasm';

try {
  const output = await transcoder.transcode(file);
} catch (err) {
  if (err instanceof MemoryError) {
    console.error('Out of memory - try a smaller file or lower settings');
  } else if (err instanceof CodecError) {
    console.error('Unsupported codec:', err.codec);
  } else if (err instanceof TranscodeError) {
    console.error('Transcode error:', err.message);
  } else {
    throw err;
  }
}
```

## Next Steps

- [C API](/docs/integrations/c-api) - Native C bindings
- [Python Integration](/docs/integrations/python) - Server-side processing
- [Node.js Integration](/docs/integrations/nodejs) - Node.js bindings
