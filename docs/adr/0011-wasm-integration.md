# ADR-0011: WebAssembly Browser Integration

## Status

Accepted

## Date

2024-06 (inferred from module structure)

## Context

Browser-based video processing enables:

1. **Client-side transcoding** without server round-trips
2. **Privacy preservation** - video never leaves user's device
3. **Reduced server costs** for basic transformations
4. **Offline capability** for installed web apps

However, browser environments have unique constraints:

- **Single-threaded by default** (main thread blocking = bad UX)
- **Memory limits** (typically 2-4GB WASM heap)
- **No file system access** (only through Web APIs)
- **Security restrictions** (CORS, COOP/COEP for SharedArrayBuffer)

We need to make transcoding work effectively within these constraints.

## Decision

Use **wasm-bindgen** for WebAssembly bindings with **Web Workers** for parallel processing and **streaming APIs** for memory-efficient large file handling.

### 1. Core WASM Bindings

Use wasm-bindgen for JavaScript interop:

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    utils::set_panic_hook();
    console::log_1(&"Transcode WASM initialized".into());
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
```

### 2. Browser Feature Detection

Check for required APIs at runtime:

```rust
#[wasm_bindgen]
pub struct BrowserSupport {
    pub wasm: bool,
    pub shared_array_buffer: bool,
    pub web_workers: bool,
    pub streams: bool,
    pub full_support: bool,
}

#[wasm_bindgen]
impl BrowserSupport {
    #[wasm_bindgen(constructor)]
    pub fn check() -> Self {
        let wasm = true; // If running, WASM works

        let shared_array_buffer = js_sys::Reflect::get(
            &js_sys::global(),
            &"SharedArrayBuffer".into()
        ).map(|v| !v.is_undefined()).unwrap_or(false);

        let web_workers = js_sys::Reflect::get(
            &js_sys::global(),
            &"Worker".into()
        ).map(|v| !v.is_undefined()).unwrap_or(false);

        let streams = js_sys::Reflect::get(
            &js_sys::global(),
            &"ReadableStream".into()
        ).map(|v| !v.is_undefined()).unwrap_or(false);

        let full_support = wasm && shared_array_buffer && web_workers && streams;

        Self { wasm, shared_array_buffer, web_workers, streams, full_support }
    }
}
```

### 3. Web Worker Pool

Offload processing to background threads:

```rust
#[wasm_bindgen]
pub struct WorkerPool {
    max_workers: usize,
    workers: Vec<web_sys::Worker>,
    pending_tasks: usize,
    completed_tasks: usize,
}

#[wasm_bindgen]
impl WorkerPool {
    #[wasm_bindgen(constructor)]
    pub fn new(max_workers: usize) -> Self {
        Self {
            max_workers: max_workers.max(1),
            workers: Vec::new(),
            pending_tasks: 0,
            completed_tasks: 0,
        }
    }

    pub fn auto() -> Self {
        // Detect hardware concurrency
        let concurrency = js_sys::Reflect::get(&js_sys::global(), &"navigator".into())
            .ok()
            .and_then(|nav| js_sys::Reflect::get(&nav, &"hardwareConcurrency".into()).ok())
            .and_then(|hc| hc.as_f64())
            .map(|n| n as usize)
            .unwrap_or(4);

        Self::new(concurrency)
    }

    #[wasm_bindgen(js_name = createWorker)]
    pub fn create_worker(&mut self, script_url: &str) -> WasmResult<usize> {
        let options = web_sys::WorkerOptions::new();
        options.set_type(web_sys::WorkerType::Module);

        let worker = web_sys::Worker::new_with_options(script_url, &options)?;
        let index = self.workers.len();
        self.workers.push(worker);

        Ok(index)
    }
}
```

### 4. Worker Message Protocol

Structured communication between main thread and workers:

```rust
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub enum WorkerMessageType {
    Init = 0,
    Start = 1,
    AddData = 2,
    Finish = 3,
    Cancel = 4,
    Progress = 5,
    Output = 6,
    Complete = 7,
    Error = 8,
}

#[wasm_bindgen]
pub struct WorkerMessage {
    message_type: WorkerMessageType,
    payload: String,
}

#[wasm_bindgen]
impl WorkerMessage {
    pub fn progress(progress: f64, frames: u64) -> Self {
        Self::new(
            WorkerMessageType::Progress,
            &format!(r#"{{"progress":{},"frames":{}}}"#, progress, frames),
        )
    }

    pub fn error(message: &str, code: ErrorCode) -> Self {
        Self::new(
            WorkerMessageType::Error,
            &format!(r#"{{"message":"{}","code":{}}}"#, message, code as u32),
        )
    }
}
```

### 5. Streaming Transcoder

Process large files without loading entirely into memory:

```rust
#[wasm_bindgen]
pub struct StreamingTranscoder {
    decoder: WasmDecoder,
    encoder: WasmEncoder,
    input_bytes: u64,
    output_bytes: u64,
    frames_processed: u64,
}

#[wasm_bindgen]
impl StreamingTranscoder {
    #[wasm_bindgen(constructor)]
    pub fn new(options: &TranscodeOptions) -> WasmResult<StreamingTranscoder>;

    #[wasm_bindgen(js_name = processChunk)]
    pub fn process_chunk(&mut self, chunk: &[u8]) -> WasmResult<Vec<u8>> {
        self.input_bytes += chunk.len() as u64;

        // Demux and decode incoming data
        let frames = self.decoder.decode(chunk)?;

        // Encode output
        let mut output = Vec::new();
        for frame in frames {
            let encoded = self.encoder.encode(&frame)?;
            output.extend(encoded);
            self.frames_processed += 1;
        }

        self.output_bytes += output.len() as u64;
        Ok(output)
    }

    pub fn finish(&mut self) -> WasmResult<Vec<u8>> {
        self.encoder.flush()
    }
}
```

### 6. Memory Management

Track WASM heap usage:

```rust
#[wasm_bindgen]
pub struct MemoryStats {
    pub total_heap: usize,
    pub used_heap: usize,
    pub free_heap: usize,
}

#[wasm_bindgen]
impl MemoryStats {
    #[wasm_bindgen(constructor)]
    pub fn current() -> Self {
        let memory = wasm_bindgen::memory();
        let buffer = memory.dyn_ref::<js_sys::WebAssembly::Memory>()
            .map(|m| m.buffer());

        let total_heap = buffer
            .and_then(|b| b.dyn_ref::<js_sys::ArrayBuffer>().map(|ab| ab.byte_length() as usize))
            .unwrap_or(0);

        // ...
    }
}
```

### 7. JavaScript API

Clean ergonomic API for JavaScript users:

```javascript
import init, { Transcoder, TranscodeOptions, check_browser_support } from 'transcode-wasm';

async function transcodeVideo(inputFile) {
    await init();

    // Check browser support
    const support = check_browser_support();
    if (!support.full_support) {
        console.warn(support.summary());
    }

    // Configure transcoding
    const options = new TranscodeOptions()
        .withVideoCodec('h264')
        .withVideoBitrate(2_000_000)
        .withResolution(1280, 720);

    // Transcode
    const transcoder = new Transcoder();
    const outputBlob = await transcoder.transcode(inputFile, options);

    return outputBlob;
}
```

## Consequences

### Positive

1. **Client-side processing**: No server required for basic transcoding

2. **Privacy**: Video data never leaves user's device

3. **Parallel processing**: Web Workers utilize multiple CPU cores

4. **Streaming support**: Handle large files without memory exhaustion

5. **Progressive enhancement**: Graceful degradation when features unavailable

6. **Same codebase**: Rust code shared between native and WASM targets

### Negative

1. **Performance gap**: WASM is slower than native code (typically 50-80% of native)

2. **Memory limits**: Browser WASM heap constrained (typically 2-4GB)

3. **No SIMD by default**: Requires explicit opt-in for WASM SIMD

4. **Security headers required**: SharedArrayBuffer needs COOP/COEP headers

5. **Browser compatibility**: Not all browsers support all features

### Mitigations

1. **SIMD support**: Enable WASM SIMD for supported browsers

```rust
#[cfg(target_feature = "simd128")]
fn decode_block_simd(block: &[u8]) -> [i16; 64] {
    // Use wasm_simd128 intrinsics
}
```

2. **Memory-aware processing**: Chunk size based on available memory

3. **Feature detection**: Check support before using advanced APIs

4. **Progressive loading**: Stream chunks rather than loading entire files

5. **Server fallback**: Option to upload to server for unsupported browsers

## Alternatives Considered

### Alternative 1: asm.js

Compile to asm.js for broader compatibility.

Rejected because:
- Significantly slower than WASM
- Larger download size
- Modern browsers all support WASM
- No active development

### Alternative 2: Native Messaging

Use browser extension with native messaging.

Rejected because:
- Requires extension installation
- Platform-specific binaries
- Complex deployment
- Not suitable for web apps

### Alternative 3: ffmpeg.wasm

Use existing ffmpeg.wasm project.

Rejected because:
- Very large download (~30MB)
- Limited customization
- Not memory-safe Rust
- Doesn't align with pure Rust goal

### Alternative 4: Server-Side Only

No client-side processing.

Rejected because:
- Privacy concerns for sensitive video
- Server costs for computation
- Network latency for large files
- No offline capability

## References

- [WebAssembly specification](https://webassembly.github.io/spec/)
- [wasm-bindgen guide](https://rustwasm.github.io/docs/wasm-bindgen/)
- [Web Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)
- [SharedArrayBuffer and COOP/COEP](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer)
- [Streams API](https://developer.mozilla.org/en-US/docs/Web/API/Streams_API)
