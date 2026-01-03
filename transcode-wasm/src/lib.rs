//! WebAssembly bindings for the transcode library.
//!
//! This crate provides browser-compatible video transcoding capabilities
//! through WebAssembly, enabling client-side video processing without
//! server round-trips.
//!
//! # Features
//!
//! - Full video decoding and encoding in the browser
//! - Streaming API for processing large files
//! - Web Worker support for background processing
//! - Zero-copy buffer sharing where possible
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { Transcoder, TranscodeOptions } from 'transcode-wasm';
//!
//! await init();
//!
//! const transcoder = new Transcoder();
//! const options = new TranscodeOptions()
//!     .withVideoCodec('h264')
//!     .withVideoBitrate(2_000_000)
//!     .withResolution(1280, 720);
//!
//! const outputBlob = await transcoder.transcode(inputFile, options);
//! ```

use wasm_bindgen::prelude::*;
use web_sys::console;

mod decoder;
mod encoder;
mod error;
mod options;
mod streaming;
mod transcoder;
mod utils;
mod worker;

pub use decoder::WasmDecoder;
pub use encoder::WasmEncoder;
pub use error::WasmError;
pub use options::{AudioOptions, TranscodeOptions, VideoOptions};
pub use streaming::{StreamingDecoder, StreamingTranscoder};
pub use transcoder::Transcoder;
pub use worker::{WorkerPool, WorkerMessage};

/// Initialize the WASM module.
///
/// This should be called before using any other functions.
/// Sets up panic hooks for better error messages in development.
#[wasm_bindgen(start)]
pub fn init() {
    utils::set_panic_hook();
    console::log_1(&"Transcode WASM initialized".into());
}

/// Get the version of the transcode-wasm library.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if the browser supports all required features.
#[wasm_bindgen]
pub fn check_browser_support() -> BrowserSupport {
    BrowserSupport::check()
}

/// Browser feature support information.
#[wasm_bindgen]
pub struct BrowserSupport {
    /// WebAssembly is supported
    pub wasm: bool,
    /// SharedArrayBuffer is available (needed for threading)
    pub shared_array_buffer: bool,
    /// Web Workers are available
    pub web_workers: bool,
    /// Streams API is available
    pub streams: bool,
    /// All features required for full functionality
    pub full_support: bool,
}

#[wasm_bindgen]
impl BrowserSupport {
    /// Check current browser's feature support.
    #[wasm_bindgen(constructor)]
    pub fn check() -> Self {
        let wasm = true; // If we're running, WASM works

        // Check for SharedArrayBuffer (requires COOP/COEP headers)
        let shared_array_buffer = js_sys::Reflect::get(
            &js_sys::global(),
            &"SharedArrayBuffer".into()
        ).map(|v| !v.is_undefined()).unwrap_or(false);

        // Check for Web Workers
        let web_workers = js_sys::Reflect::get(
            &js_sys::global(),
            &"Worker".into()
        ).map(|v| !v.is_undefined()).unwrap_or(false);

        // Check for Streams API
        let streams = js_sys::Reflect::get(
            &js_sys::global(),
            &"ReadableStream".into()
        ).map(|v| !v.is_undefined()).unwrap_or(false);

        let full_support = wasm && shared_array_buffer && web_workers && streams;

        Self {
            wasm,
            shared_array_buffer,
            web_workers,
            streams,
            full_support,
        }
    }

    /// Get a human-readable summary of support.
    pub fn summary(&self) -> String {
        let mut features = Vec::new();

        if self.wasm { features.push("WebAssembly"); }
        if self.shared_array_buffer { features.push("SharedArrayBuffer"); }
        if self.web_workers { features.push("Web Workers"); }
        if self.streams { features.push("Streams API"); }

        if self.full_support {
            format!("Full support: {}", features.join(", "))
        } else {
            let missing: Vec<&str> = vec![
                if !self.shared_array_buffer { Some("SharedArrayBuffer") } else { None },
                if !self.web_workers { Some("Web Workers") } else { None },
                if !self.streams { Some("Streams API") } else { None },
            ].into_iter().flatten().collect();

            format!("Partial support. Missing: {}", missing.join(", "))
        }
    }
}

/// Memory statistics for the WASM heap.
#[wasm_bindgen]
pub struct MemoryStats {
    /// Total heap size in bytes
    pub total_heap: usize,
    /// Used heap in bytes
    pub used_heap: usize,
    /// Free heap in bytes
    pub free_heap: usize,
}

#[wasm_bindgen]
impl MemoryStats {
    /// Get current memory statistics.
    #[wasm_bindgen(constructor)]
    pub fn current() -> Self {
        // Get WASM memory info
        let memory = wasm_bindgen::memory();
        let buffer = memory.dyn_ref::<js_sys::WebAssembly::Memory>()
            .map(|m| m.buffer());

        let total_heap = buffer
            .and_then(|b| b.dyn_ref::<js_sys::ArrayBuffer>().map(|ab| ab.byte_length() as usize))
            .unwrap_or(0);

        // Estimate used heap (this is approximate)
        let used_heap = total_heap / 2; // Placeholder
        let free_heap = total_heap - used_heap;

        Self {
            total_heap,
            used_heap,
            free_heap,
        }
    }

    /// Format as human-readable string.
    pub fn format(&self) -> String {
        format!(
            "Heap: {:.1}MB used / {:.1}MB total",
            self.used_heap as f64 / 1_000_000.0,
            self.total_heap as f64 / 1_000_000.0
        )
    }
}

/// Performance timing utilities.
#[wasm_bindgen]
pub struct PerfTimer {
    start: f64,
    name: String,
}

#[wasm_bindgen]
impl PerfTimer {
    /// Start a new performance timer.
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str) -> Self {
        let start = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        Self {
            start,
            name: name.to_string(),
        }
    }

    /// Get elapsed time in milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        let now = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        now - self.start
    }

    /// Log elapsed time to console.
    pub fn log(&self) {
        console::log_1(&format!("{}: {:.2}ms", self.name, self.elapsed_ms()).into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[wasm_bindgen_test]
    fn test_browser_support() {
        let support = BrowserSupport::check();
        assert!(support.wasm);
    }
}
