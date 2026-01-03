//! Streaming transcoding support using Web Streams API.

#![allow(dead_code)]

use crate::decoder::WasmDecoder;
use crate::encoder::WasmEncoder;
use crate::error::WasmResult;
use crate::options::TranscodeOptions;
use js_sys::{Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

/// Streaming decoder that processes chunks of data.
#[wasm_bindgen]
pub struct StreamingDecoder {
    /// Internal decoder.
    decoder: WasmDecoder,
    /// Total bytes processed.
    bytes_processed: u64,
    /// Callback for decoded frames.
    on_frame: Option<js_sys::Function>,
}

#[wasm_bindgen]
impl StreamingDecoder {
    /// Create a new streaming decoder.
    #[wasm_bindgen(constructor)]
    pub fn new(codec: &str) -> WasmResult<StreamingDecoder> {
        let decoder = WasmDecoder::new(codec)?;

        Ok(Self {
            decoder,
            bytes_processed: 0,
            on_frame: None,
        })
    }

    /// Set the frame callback.
    #[wasm_bindgen(js_name = onFrame)]
    pub fn on_frame(&mut self, callback: js_sys::Function) {
        self.on_frame = Some(callback);
    }

    /// Process a chunk of data.
    #[wasm_bindgen(js_name = processChunk)]
    pub fn process_chunk(&mut self, chunk: &[u8]) -> WasmResult<u32> {
        self.bytes_processed += chunk.len() as u64;

        let frames_decoded = self.decoder.decode(chunk)?;

        // Call frame callback for each decoded frame
        if let Some(ref callback) = self.on_frame {
            while self.decoder.has_frames() {
                if let Some(frame) = self.decoder.get_frame() {
                    let _ = callback.call1(&JsValue::NULL, &frame.to_object());
                }
            }
        }

        Ok(frames_decoded)
    }

    /// Flush remaining frames.
    pub fn flush(&mut self) -> WasmResult<u32> {
        let remaining = self.decoder.flush()?;

        if let Some(ref callback) = self.on_frame {
            while self.decoder.has_frames() {
                if let Some(frame) = self.decoder.get_frame() {
                    let _ = callback.call1(&JsValue::NULL, &frame.to_object());
                }
            }
        }

        Ok(remaining)
    }

    /// Get bytes processed.
    #[wasm_bindgen(getter, js_name = bytesProcessed)]
    pub fn bytes_processed(&self) -> u64 {
        self.bytes_processed
    }

    /// Get frames decoded.
    #[wasm_bindgen(getter, js_name = framesDecoded)]
    pub fn frames_decoded(&self) -> u64 {
        self.decoder.frames_decoded()
    }

    /// Reset the decoder.
    pub fn reset(&mut self) {
        self.decoder.reset();
        self.bytes_processed = 0;
    }
}

/// Streaming transcoder that processes data in chunks.
#[wasm_bindgen]
pub struct StreamingTranscoder {
    /// Options for transcoding.
    options: TranscodeOptions,
    /// Video decoder.
    decoder: Option<WasmDecoder>,
    /// Video encoder.
    encoder: Option<WasmEncoder>,
    /// Total input bytes.
    input_bytes: u64,
    /// Total output bytes.
    output_bytes: u64,
    /// Frames processed.
    frames_processed: u64,
    /// Callback for output chunks.
    on_output: Option<js_sys::Function>,
    /// Callback for progress updates.
    on_progress: Option<js_sys::Function>,
    /// Chunk size for output.
    output_chunk_size: usize,
    /// Internal output buffer.
    output_buffer: Vec<u8>,
}

#[wasm_bindgen]
impl StreamingTranscoder {
    /// Create a new streaming transcoder.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmResult<StreamingTranscoder> {
        Ok(Self {
            options: TranscodeOptions::new(),
            decoder: None,
            encoder: None,
            input_bytes: 0,
            output_bytes: 0,
            frames_processed: 0,
            on_output: None,
            on_progress: None,
            output_chunk_size: 65536, // 64KB chunks
            output_buffer: Vec::new(),
        })
    }

    /// Create with specific options.
    #[wasm_bindgen(js_name = withOptions)]
    pub fn with_options(options: TranscodeOptions) -> WasmResult<StreamingTranscoder> {
        Ok(Self {
            options,
            decoder: None,
            encoder: None,
            input_bytes: 0,
            output_bytes: 0,
            frames_processed: 0,
            on_output: None,
            on_progress: None,
            output_chunk_size: 65536,
            output_buffer: Vec::new(),
        })
    }

    /// Set the output callback.
    #[wasm_bindgen(js_name = onOutput)]
    pub fn on_output(&mut self, callback: js_sys::Function) {
        self.on_output = Some(callback);
    }

    /// Set the progress callback.
    #[wasm_bindgen(js_name = onProgress)]
    pub fn on_progress(&mut self, callback: js_sys::Function) {
        self.on_progress = Some(callback);
    }

    /// Set output chunk size.
    #[wasm_bindgen(js_name = setChunkSize)]
    pub fn set_chunk_size(&mut self, size: usize) {
        self.output_chunk_size = size.max(1024); // Minimum 1KB
    }

    /// Initialize the transcoder.
    fn ensure_initialized(&mut self) -> WasmResult<()> {
        if self.decoder.is_none() {
            self.decoder = Some(WasmDecoder::new("h264")?);
        }
        if self.encoder.is_none() {
            self.encoder = Some(WasmEncoder::new("h264")?);
        }
        Ok(())
    }

    /// Process an input chunk.
    #[wasm_bindgen(js_name = processChunk)]
    pub fn process_chunk(&mut self, chunk: &[u8]) -> WasmResult<Uint8Array> {
        self.ensure_initialized()?;

        self.input_bytes += chunk.len() as u64;

        // Decode
        if let Some(ref mut decoder) = self.decoder {
            decoder.decode(chunk)?;

            // Encode decoded frames
            while decoder.has_frames() {
                if let Some(frame) = decoder.get_frame() {
                    if let Some(ref mut encoder) = self.encoder {
                        encoder.encode_frame(&frame)?;
                        self.frames_processed += 1;
                    }
                }
            }
        }

        // Collect encoded packets
        self.collect_output();

        // Return available output
        let output = self.flush_output_buffer();

        // Call progress callback
        self.report_progress();

        Ok(output)
    }

    /// Collect output from encoder.
    fn collect_output(&mut self) {
        if let Some(ref mut encoder) = self.encoder {
            while encoder.has_packets() {
                if let Some(packet) = encoder.get_packet() {
                    let data = packet.get_data().to_vec();
                    self.output_buffer.extend_from_slice(&data);
                }
            }
        }
    }

    /// Flush output buffer in chunks.
    fn flush_output_buffer(&mut self) -> Uint8Array {
        if self.output_buffer.len() >= self.output_chunk_size {
            let chunk: Vec<u8> = self.output_buffer.drain(..self.output_chunk_size).collect();
            self.output_bytes += chunk.len() as u64;

            // Call output callback
            if let Some(ref callback) = self.on_output {
                let data = Uint8Array::from(chunk.as_slice());
                let _ = callback.call1(&JsValue::NULL, &data);
            }

            Uint8Array::from(chunk.as_slice())
        } else {
            Uint8Array::new_with_length(0)
        }
    }

    /// Report progress via callback.
    fn report_progress(&self) {
        if let Some(ref callback) = self.on_progress {
            let obj = Object::new();
            let _ = Reflect::set(
                &obj,
                &"inputBytes".into(),
                &JsValue::from_f64(self.input_bytes as f64),
            );
            let _ = Reflect::set(
                &obj,
                &"outputBytes".into(),
                &JsValue::from_f64(self.output_bytes as f64),
            );
            let _ = Reflect::set(
                &obj,
                &"framesProcessed".into(),
                &JsValue::from_f64(self.frames_processed as f64),
            );
            let _ = callback.call1(&JsValue::NULL, &obj);
        }
    }

    /// Finish processing and flush remaining data.
    pub fn finish(&mut self) -> WasmResult<Uint8Array> {
        // Flush decoder
        if let Some(ref mut decoder) = self.decoder {
            decoder.flush()?;

            while decoder.has_frames() {
                if let Some(frame) = decoder.get_frame() {
                    if let Some(ref mut encoder) = self.encoder {
                        encoder.encode_frame(&frame)?;
                        self.frames_processed += 1;
                    }
                }
            }
        }

        // Flush encoder
        if let Some(ref mut encoder) = self.encoder {
            encoder.flush()?;
        }

        // Collect remaining output
        self.collect_output();

        // Return all remaining data
        let output: Vec<u8> = self.output_buffer.drain(..).collect();
        self.output_bytes += output.len() as u64;

        if let Some(ref callback) = self.on_output {
            let data = Uint8Array::from(output.as_slice());
            let _ = callback.call1(&JsValue::NULL, &data);
        }

        self.report_progress();

        Ok(Uint8Array::from(output.as_slice()))
    }

    /// Reset the transcoder.
    pub fn reset(&mut self) {
        if let Some(ref mut decoder) = self.decoder {
            decoder.reset();
        }
        if let Some(ref mut encoder) = self.encoder {
            encoder.reset();
        }
        self.input_bytes = 0;
        self.output_bytes = 0;
        self.frames_processed = 0;
        self.output_buffer.clear();
    }

    /// Get statistics.
    #[wasm_bindgen(getter)]
    pub fn stats(&self) -> Object {
        let obj = Object::new();
        let _ = Reflect::set(
            &obj,
            &"inputBytes".into(),
            &JsValue::from_f64(self.input_bytes as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"outputBytes".into(),
            &JsValue::from_f64(self.output_bytes as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"framesProcessed".into(),
            &JsValue::from_f64(self.frames_processed as f64),
        );

        if self.input_bytes > 0 && self.output_bytes > 0 {
            let ratio = self.input_bytes as f64 / self.output_bytes as f64;
            let _ = Reflect::set(&obj, &"compressionRatio".into(), &ratio.into());
        }

        obj
    }

    /// Get input bytes processed.
    #[wasm_bindgen(getter, js_name = inputBytes)]
    pub fn input_bytes(&self) -> u64 {
        self.input_bytes
    }

    /// Get output bytes produced.
    #[wasm_bindgen(getter, js_name = outputBytes)]
    pub fn output_bytes(&self) -> u64 {
        self.output_bytes
    }

    /// Get frames processed.
    #[wasm_bindgen(getter, js_name = framesProcessed)]
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }
}

impl Default for StreamingTranscoder {
    fn default() -> Self {
        Self::new().expect("Failed to create default StreamingTranscoder")
    }
}
