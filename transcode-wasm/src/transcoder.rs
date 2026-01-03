//! WASM transcoder - high-level transcoding API.

use crate::decoder::WasmDecoder;
use crate::encoder::WasmEncoder;
use crate::error::{WasmError, WasmResult};
use crate::options::TranscodeOptions;
use js_sys::{Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;
use web_sys::Blob;

/// Transcoding progress information.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct TranscodeProgress {
    /// Percentage complete (0-100).
    pub progress: f64,
    /// Frames processed.
    pub frames_processed: u64,
    /// Bytes processed from input.
    pub bytes_processed: u64,
    /// Bytes produced in output.
    pub bytes_produced: u64,
    /// Current processing speed (frames per second).
    pub speed_fps: f64,
    /// Estimated time remaining in milliseconds.
    pub eta_ms: f64,
}

#[wasm_bindgen]
impl TranscodeProgress {
    /// Create a new progress object.
    fn new() -> Self {
        Self {
            progress: 0.0,
            frames_processed: 0,
            bytes_processed: 0,
            bytes_produced: 0,
            speed_fps: 0.0,
            eta_ms: 0.0,
        }
    }

    /// Convert to JavaScript object.
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Object {
        let obj = Object::new();
        let _ = Reflect::set(&obj, &"progress".into(), &self.progress.into());
        let _ = Reflect::set(
            &obj,
            &"framesProcessed".into(),
            &JsValue::from_f64(self.frames_processed as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"bytesProcessed".into(),
            &JsValue::from_f64(self.bytes_processed as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"bytesProduced".into(),
            &JsValue::from_f64(self.bytes_produced as f64),
        );
        let _ = Reflect::set(&obj, &"speedFps".into(), &self.speed_fps.into());
        let _ = Reflect::set(&obj, &"etaMs".into(), &self.eta_ms.into());
        obj
    }
}

/// Transcoding state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscoderState {
    /// Ready to start.
    Ready,
    /// Currently transcoding.
    Transcoding,
    /// Transcoding paused.
    Paused,
    /// Transcoding complete.
    Complete,
    /// Transcoding failed.
    Failed,
    /// Transcoding cancelled.
    Cancelled,
}

/// High-level transcoder for browser use.
#[wasm_bindgen]
pub struct Transcoder {
    /// Transcoding options.
    options: TranscodeOptions,
    /// Current state.
    state: TranscoderState,
    /// Video decoder.
    decoder: Option<WasmDecoder>,
    /// Video encoder.
    encoder: Option<WasmEncoder>,
    /// Progress information.
    progress: TranscodeProgress,
    /// Input data buffer.
    input_buffer: Vec<u8>,
    /// Output data buffer.
    output_buffer: Vec<u8>,
    /// Total input size (if known).
    total_input_size: Option<u64>,
    /// Start time for speed calculation.
    start_time: Option<f64>,
}

#[wasm_bindgen]
impl Transcoder {
    /// Create a new transcoder with default options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            options: TranscodeOptions::new(),
            state: TranscoderState::Ready,
            decoder: None,
            encoder: None,
            progress: TranscodeProgress::new(),
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            total_input_size: None,
            start_time: None,
        }
    }

    /// Create a transcoder with specific options.
    #[wasm_bindgen(js_name = withOptions)]
    pub fn with_options(options: TranscodeOptions) -> Self {
        Self {
            options,
            state: TranscoderState::Ready,
            decoder: None,
            encoder: None,
            progress: TranscodeProgress::new(),
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            total_input_size: None,
            start_time: None,
        }
    }

    /// Set transcoding options.
    #[wasm_bindgen(js_name = setOptions)]
    pub fn set_options(&mut self, options: TranscodeOptions) {
        self.options = options;
    }

    /// Initialize the transcoder for a specific input.
    fn initialize(&mut self) -> WasmResult<()> {
        // Create decoder
        let decoder = WasmDecoder::new("h264")?;
        self.decoder = Some(decoder);

        // Create encoder based on options
        let encoder = WasmEncoder::from_options(
            &crate::options::VideoOptions::new().with_codec("h264"),
        )?;
        self.encoder = Some(encoder);

        self.state = TranscoderState::Ready;
        self.start_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now());

        Ok(())
    }

    /// Transcode a complete file (Blob or ArrayBuffer).
    pub fn transcode(&mut self, data: &[u8]) -> WasmResult<Uint8Array> {
        self.initialize()?;

        self.total_input_size = Some(data.len() as u64);
        self.state = TranscoderState::Transcoding;

        // Process input data
        self.input_buffer.extend_from_slice(data);
        self.progress.bytes_processed = data.len() as u64;

        // Decode and encode
        if let Some(ref mut decoder) = self.decoder {
            decoder.decode(data)?;

            while decoder.has_frames() {
                if let Some(frame) = decoder.get_frame() {
                    if let Some(ref mut encoder) = self.encoder {
                        encoder.encode_frame(&frame)?;
                        self.progress.frames_processed += 1;
                    }
                }
            }

            decoder.flush()?;
        }

        // Flush encoder
        if let Some(ref mut encoder) = self.encoder {
            encoder.flush()?;

            // Collect output
            while encoder.has_packets() {
                if let Some(packet) = encoder.get_packet() {
                    self.output_buffer.extend_from_slice(&packet.get_data().to_vec());
                }
            }
            self.progress.bytes_produced = encoder.bytes_produced();
        }

        self.state = TranscoderState::Complete;
        self.progress.progress = 100.0;

        Ok(Uint8Array::from(self.output_buffer.as_slice()))
    }

    /// Transcode a Blob asynchronously (returns a Promise).
    #[wasm_bindgen(js_name = transcodeBlob)]
    pub fn transcode_blob(&mut self, blob: &Blob) -> Promise {
        let blob_clone = blob.clone();

        future_to_promise(async move {
            // Read blob data
            let array_buffer = wasm_bindgen_futures::JsFuture::from(blob_clone.array_buffer())
                .await
                .map_err(|e| JsValue::from_str(&format!("Failed to read blob: {:?}", e)))?;

            let data = Uint8Array::new(&array_buffer);
            let mut transcoder = Transcoder::new();

            match transcoder.transcode(&data.to_vec()) {
                Ok(result) => Ok(result.into()),
                Err(e) => Err(JsValue::from_str(&e.to_string())),
            }
        })
    }

    /// Add input data for streaming transcoding.
    #[wasm_bindgen(js_name = addInput)]
    pub fn add_input(&mut self, data: &[u8]) -> WasmResult<()> {
        if self.state == TranscoderState::Ready {
            self.initialize()?;
            self.state = TranscoderState::Transcoding;
        }

        if self.state != TranscoderState::Transcoding {
            return Err(WasmError::invalid_input(
                "Transcoder is not in transcoding state",
            ));
        }

        self.input_buffer.extend_from_slice(data);
        self.progress.bytes_processed += data.len() as u64;

        // Decode incoming data
        if let Some(ref mut decoder) = self.decoder {
            decoder.decode(data)?;

            // Encode any decoded frames
            while decoder.has_frames() {
                if let Some(frame) = decoder.get_frame() {
                    if let Some(ref mut encoder) = self.encoder {
                        encoder.encode_frame(&frame)?;
                        self.progress.frames_processed += 1;
                    }
                }
            }
        }

        self.update_progress();
        Ok(())
    }

    /// Get available output data.
    #[wasm_bindgen(js_name = getOutput)]
    pub fn get_output(&mut self) -> Uint8Array {
        let mut output = Vec::new();

        if let Some(ref mut encoder) = self.encoder {
            while encoder.has_packets() {
                if let Some(packet) = encoder.get_packet() {
                    output.extend_from_slice(&packet.get_data().to_vec());
                }
            }
        }

        self.progress.bytes_produced += output.len() as u64;
        Uint8Array::from(output.as_slice())
    }

    /// Signal end of input data.
    #[wasm_bindgen(js_name = finishInput)]
    pub fn finish_input(&mut self) -> WasmResult<()> {
        if let Some(ref mut decoder) = self.decoder {
            decoder.flush()?;

            // Encode remaining frames
            while decoder.has_frames() {
                if let Some(frame) = decoder.get_frame() {
                    if let Some(ref mut encoder) = self.encoder {
                        encoder.encode_frame(&frame)?;
                        self.progress.frames_processed += 1;
                    }
                }
            }
        }

        if let Some(ref mut encoder) = self.encoder {
            encoder.flush()?;
        }

        self.state = TranscoderState::Complete;
        self.progress.progress = 100.0;
        Ok(())
    }

    /// Pause transcoding.
    pub fn pause(&mut self) {
        if self.state == TranscoderState::Transcoding {
            self.state = TranscoderState::Paused;
        }
    }

    /// Resume transcoding.
    pub fn resume(&mut self) {
        if self.state == TranscoderState::Paused {
            self.state = TranscoderState::Transcoding;
        }
    }

    /// Cancel transcoding.
    pub fn cancel(&mut self) {
        self.state = TranscoderState::Cancelled;
        self.decoder = None;
        self.encoder = None;
    }

    /// Reset the transcoder.
    pub fn reset(&mut self) {
        self.state = TranscoderState::Ready;
        self.decoder = None;
        self.encoder = None;
        self.progress = TranscodeProgress::new();
        self.input_buffer.clear();
        self.output_buffer.clear();
        self.total_input_size = None;
        self.start_time = None;
    }

    /// Update progress calculations.
    fn update_progress(&mut self) {
        // Calculate progress percentage
        if let Some(total) = self.total_input_size {
            if total > 0 {
                self.progress.progress =
                    (self.progress.bytes_processed as f64 / total as f64 * 100.0).min(99.9);
            }
        }

        // Calculate speed
        if let Some(start) = self.start_time {
            let now = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(start);

            let elapsed_s = (now - start) / 1000.0;
            if elapsed_s > 0.0 {
                self.progress.speed_fps = self.progress.frames_processed as f64 / elapsed_s;

                // Calculate ETA
                if self.progress.progress > 0.0 && self.progress.progress < 100.0 {
                    let remaining_pct = 100.0 - self.progress.progress;
                    let ms_per_pct = (now - start) / self.progress.progress;
                    self.progress.eta_ms = remaining_pct * ms_per_pct;
                }
            }
        }
    }

    /// Get current progress.
    #[wasm_bindgen(js_name = getProgress)]
    pub fn get_progress(&self) -> TranscodeProgress {
        self.progress.clone()
    }

    /// Check if transcoding is complete.
    #[wasm_bindgen(js_name = isComplete)]
    pub fn is_complete(&self) -> bool {
        self.state == TranscoderState::Complete
    }

    /// Check if transcoding is in progress.
    #[wasm_bindgen(js_name = isTranscoding)]
    pub fn is_transcoding(&self) -> bool {
        self.state == TranscoderState::Transcoding
    }

    /// Check if transcoding was cancelled.
    #[wasm_bindgen(js_name = isCancelled)]
    pub fn is_cancelled(&self) -> bool {
        self.state == TranscoderState::Cancelled
    }

    /// Check if transcoding failed.
    #[wasm_bindgen(js_name = hasFailed)]
    pub fn has_failed(&self) -> bool {
        self.state == TranscoderState::Failed
    }

    /// Get transcoder statistics.
    #[wasm_bindgen(getter)]
    pub fn stats(&self) -> Object {
        let obj = Object::new();
        let _ = Reflect::set(
            &obj,
            &"framesProcessed".into(),
            &JsValue::from_f64(self.progress.frames_processed as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"bytesProcessed".into(),
            &JsValue::from_f64(self.progress.bytes_processed as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"bytesProduced".into(),
            &JsValue::from_f64(self.progress.bytes_produced as f64),
        );
        let _ = Reflect::set(&obj, &"progress".into(), &self.progress.progress.into());
        let _ = Reflect::set(&obj, &"speedFps".into(), &self.progress.speed_fps.into());

        // Compression ratio
        if self.progress.bytes_processed > 0 && self.progress.bytes_produced > 0 {
            let ratio =
                self.progress.bytes_processed as f64 / self.progress.bytes_produced as f64;
            let _ = Reflect::set(&obj, &"compressionRatio".into(), &ratio.into());
        }

        obj
    }
}

impl Default for Transcoder {
    fn default() -> Self {
        Self::new()
    }
}
