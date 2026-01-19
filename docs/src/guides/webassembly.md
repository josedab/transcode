# WebAssembly

This guide covers using Transcode in the browser via WebAssembly.

## Overview

The `transcode-wasm` crate provides WebAssembly bindings for client-side video processing:

- Decode and encode video in the browser
- Process frames without server round-trips
- Use Web Workers for parallel processing
- Zero server costs for video processing

## Installation

### npm

```bash
npm install transcode-wasm
```

### CDN

```html
<script type="module">
import init, { WasmTranscoder } from 'https://unpkg.com/transcode-wasm@latest/transcode_wasm.js';
</script>
```

## Quick Start

```javascript
import init, { WasmTranscoder } from 'transcode-wasm';

async function transcodeVideo() {
    // Initialize WASM module
    await init();

    // Create transcoder
    const transcoder = new WasmTranscoder();

    // Configure
    transcoder.set_input_format('h264');
    transcoder.set_output_format('vp9');
    transcoder.set_output_bitrate(2_000_000);

    // Process video file
    const inputFile = document.querySelector('input[type="file"]').files[0];
    const inputData = new Uint8Array(await inputFile.arrayBuffer());

    const outputData = transcoder.transcode(inputData);

    // Download result
    const blob = new Blob([outputData], { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'output.webm';
    a.click();
}
```

## Configuration

```javascript
const transcoder = new WasmTranscoder();

// Video settings
transcoder.set_input_format('h264');      // h264, hevc, vp8, vp9, av1
transcoder.set_output_format('vp9');      // vp8, vp9, av1
transcoder.set_output_bitrate(2_000_000); // 2 Mbps
transcoder.set_output_resolution(1280, 720);
transcoder.set_crf(30);                   // Quality level

// Audio settings
transcoder.set_audio_codec('opus');       // opus, vorbis
transcoder.set_audio_bitrate(128_000);    // 128 kbps
```

## Frame-by-Frame Processing

```javascript
import init, { WasmDecoder, WasmEncoder } from 'transcode-wasm';

async function processFrames() {
    await init();

    // Create decoder and encoder
    const decoder = new WasmDecoder('h264');
    const encoder = new WasmEncoder('vp9');

    encoder.set_bitrate(2_000_000);
    encoder.set_resolution(1280, 720);

    // Decode frames
    const inputData = new Uint8Array(/* ... */);
    const packets = decoder.decode(inputData);

    // Process each frame
    const outputPackets = [];
    for (const frame of packets) {
        // Apply effects (grayscale, etc.)
        const processed = applyEffects(frame);

        // Encode
        const packet = encoder.encode(processed);
        outputPackets.push(packet);
    }

    // Finalize
    const trailer = encoder.finish();
    outputPackets.push(trailer);

    return mergePackets(outputPackets);
}
```

## Web Workers

For heavy processing, use Web Workers to avoid blocking the main thread:

### worker.js

```javascript
import init, { WasmTranscoder } from 'transcode-wasm';

let transcoder = null;

self.onmessage = async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            await init();
            transcoder = new WasmTranscoder();
            transcoder.set_output_format('vp9');
            self.postMessage({ type: 'ready' });
            break;

        case 'transcode':
            const result = transcoder.transcode(data);
            self.postMessage(
                { type: 'complete', data: result },
                [result.buffer]  // Transfer ownership
            );
            break;

        case 'progress':
            // Report progress back to main thread
            self.postMessage({
                type: 'progress',
                progress: transcoder.progress()
            });
            break;
    }
};
```

### main.js

```javascript
const worker = new Worker('worker.js', { type: 'module' });

worker.onmessage = (e) => {
    const { type, data, progress } = e.data;

    switch (type) {
        case 'ready':
            console.log('Worker ready');
            break;
        case 'progress':
            updateProgressBar(progress);
            break;
        case 'complete':
            downloadResult(data);
            break;
    }
};

// Initialize
worker.postMessage({ type: 'init' });

// Transcode
async function startTranscode(file) {
    const data = new Uint8Array(await file.arrayBuffer());
    worker.postMessage(
        { type: 'transcode', data },
        [data.buffer]  // Transfer ownership
    );
}
```

## Parallel Processing with Worker Pool

```javascript
class TranscodeWorkerPool {
    constructor(numWorkers = navigator.hardwareConcurrency) {
        this.workers = [];
        this.taskQueue = [];
        this.available = [];

        for (let i = 0; i < numWorkers; i++) {
            const worker = new Worker('worker.js', { type: 'module' });
            worker.onmessage = (e) => this.handleMessage(i, e);
            this.workers.push(worker);
            this.available.push(i);
        }
    }

    async transcode(segment) {
        return new Promise((resolve) => {
            this.taskQueue.push({ segment, resolve });
            this.processQueue();
        });
    }

    processQueue() {
        while (this.taskQueue.length > 0 && this.available.length > 0) {
            const workerId = this.available.pop();
            const task = this.taskQueue.shift();
            this.workers[workerId].postMessage({
                type: 'transcode',
                data: task.segment,
                taskId: task
            });
        }
    }

    handleMessage(workerId, e) {
        if (e.data.type === 'complete') {
            e.data.taskId.resolve(e.data.data);
            this.available.push(workerId);
            this.processQueue();
        }
    }
}

// Usage
const pool = new TranscodeWorkerPool(4);
const segments = splitVideo(inputData, 10); // 10-second segments
const results = await Promise.all(segments.map(s => pool.transcode(s)));
const output = mergeSegments(results);
```

## Memory Management

WebAssembly has limited memory. Handle large files in chunks:

```javascript
const CHUNK_SIZE = 10 * 1024 * 1024; // 10 MB chunks

async function transcodeChunked(file) {
    await init();

    const transcoder = new WasmTranscoder();
    transcoder.set_output_format('vp9');

    const outputChunks = [];
    let offset = 0;

    while (offset < file.size) {
        const chunk = file.slice(offset, offset + CHUNK_SIZE);
        const data = new Uint8Array(await chunk.arrayBuffer());

        const result = transcoder.transcode_chunk(data, offset === 0);
        outputChunks.push(result);

        offset += CHUNK_SIZE;

        // Report progress
        console.log(`Progress: ${(offset / file.size * 100).toFixed(1)}%`);
    }

    // Finalize
    const trailer = transcoder.finish();
    outputChunks.push(trailer);

    return new Blob(outputChunks, { type: 'video/webm' });
}
```

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome 57+ | Full | Best performance |
| Firefox 52+ | Full | Good performance |
| Safari 11+ | Full | Requires `wasm-bindgen` |
| Edge 16+ | Full | Chromium-based |

### Feature Detection

```javascript
function checkSupport() {
    const features = {
        wasm: typeof WebAssembly !== 'undefined',
        workers: typeof Worker !== 'undefined',
        sharedMemory: typeof SharedArrayBuffer !== 'undefined',
        simd: WebAssembly.validate(new Uint8Array([
            0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
            3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
        ]))
    };

    return features;
}
```

## Performance Tips

1. **Use Web Workers** - Never block the main thread
2. **Transfer buffers** - Use `postMessage` with transferable objects
3. **Process in chunks** - Don't load entire files into memory
4. **Enable SIMD** - Check browser support for WASM SIMD
5. **Reuse instances** - Don't create new transcoders for each file

## Complete Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>Transcode Demo</title>
</head>
<body>
    <input type="file" id="input" accept="video/*">
    <progress id="progress" value="0" max="100"></progress>
    <button id="download" disabled>Download</button>

    <script type="module">
        import init, { WasmTranscoder } from 'transcode-wasm';

        let result = null;

        document.getElementById('input').onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            await init();

            const transcoder = new WasmTranscoder();
            transcoder.set_output_format('vp9');
            transcoder.set_output_bitrate(2_000_000);

            const progress = document.getElementById('progress');
            const download = document.getElementById('download');

            const data = new Uint8Array(await file.arrayBuffer());
            result = transcoder.transcode(data);

            progress.value = 100;
            download.disabled = false;
        };

        document.getElementById('download').onclick = () => {
            if (!result) return;

            const blob = new Blob([result], { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'output.webm';
            a.click();
            URL.revokeObjectURL(url);
        };
    </script>
</body>
</html>
```
