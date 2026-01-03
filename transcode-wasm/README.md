# transcode-wasm

WebAssembly bindings for the transcode library, enabling client-side video transcoding in the browser without server round-trips.

## Features

- Full video decoding and encoding in the browser
- Streaming API for processing large files
- Web Worker support for background processing
- Zero-copy buffer sharing where possible
- Browser feature detection

## Installation

```bash
wasm-pack build --target web
```

## Usage

### Basic Transcoding

```javascript
import init, { Transcoder, TranscodeOptions } from 'transcode-wasm';

await init();

const transcoder = new Transcoder();
const options = new TranscodeOptions()
    .withVideoCodec('h264')
    .withVideoBitrate(2_000_000)
    .withResolution(1280, 720);

const outputData = await transcoder.transcodeBlob(inputFile);
```

### Streaming Transcoding

```javascript
import { StreamingTranscoder } from 'transcode-wasm';

const transcoder = new StreamingTranscoder();
transcoder.onProgress((progress) => {
    console.log(`Frames: ${progress.framesProcessed}`);
});

// Process chunks as they arrive
for await (const chunk of inputStream) {
    const output = transcoder.processChunk(chunk);
    // Handle output...
}
const finalOutput = transcoder.finish();
```

### Web Worker Usage

```javascript
import { WorkerPool, createWorkerBlobUrl } from 'transcode-wasm';

const workerUrl = createWorkerBlobUrl('./transcode_wasm.js');
const pool = new WorkerPool(navigator.hardwareConcurrency);

const workerIndex = pool.createWorker(workerUrl);
pool.setOnMessage(workerIndex, (event) => {
    // Handle worker messages
});
pool.postMessage(workerIndex, WorkerMessage.start(optionsJson));
```

### Browser Support Check

```javascript
import { check_browser_support } from 'transcode-wasm';

const support = check_browser_support();
if (!support.full_support) {
    console.warn(support.summary());
}
```

## Key Types

| Type | Description |
|------|-------------|
| `Transcoder` | High-level transcoder with progress tracking |
| `StreamingTranscoder` | Chunk-based streaming transcoder |
| `StreamingDecoder` | Streaming video decoder |
| `TranscodeOptions` | Video/audio codec and quality settings |
| `VideoOptions` | Video codec, resolution, bitrate, quality |
| `AudioOptions` | Audio codec, bitrate, sample rate, channels |
| `WorkerPool` | Manages Web Workers for parallel processing |
| `WorkerMessage` | Message protocol for worker communication |
| `BrowserSupport` | Browser feature detection |
| `TranscodeProgress` | Progress info (frames, bytes, speed, ETA) |

## Crate Features

- `default` - Includes `console_error_panic_hook` for better error messages
- `parallel` - Enables `rayon` and `wasm-bindgen-rayon` for parallel processing

## Browser Requirements

For full functionality, ensure these browser features are available:

- **WebAssembly** - Required
- **SharedArrayBuffer** - Required for threading (needs COOP/COEP headers)
- **Web Workers** - Required for background processing
- **Streams API** - Required for streaming transcoding

## License

MIT OR Apache-2.0
