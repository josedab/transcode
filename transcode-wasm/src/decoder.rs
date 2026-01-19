//! WASM video decoder wrapper.

#![allow(dead_code)]

use crate::error::{WasmError, WasmResult};
use js_sys::{Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

/// Decoded frame information.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Presentation timestamp in microseconds.
    pts: i64,
    /// Frame duration in microseconds.
    duration: i64,
    /// Whether this is a keyframe.
    keyframe: bool,
    /// Pixel format (e.g., "yuv420p", "rgb24").
    pixel_format: String,
    /// Raw frame data.
    data: Vec<u8>,
}

#[wasm_bindgen]
impl DecodedFrame {
    /// Get frame width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get frame height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get presentation timestamp.
    #[wasm_bindgen(getter)]
    pub fn pts(&self) -> i64 {
        self.pts
    }

    /// Get frame duration.
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> i64 {
        self.duration
    }

    /// Check if this is a keyframe.
    #[wasm_bindgen(getter)]
    pub fn keyframe(&self) -> bool {
        self.keyframe
    }

    /// Get pixel format.
    #[wasm_bindgen(getter, js_name = pixelFormat)]
    pub fn pixel_format(&self) -> String {
        self.pixel_format.clone()
    }

    /// Get frame data as Uint8Array.
    #[wasm_bindgen(js_name = getData)]
    pub fn get_data(&self) -> Uint8Array {
        Uint8Array::from(self.data.as_slice())
    }

    /// Get frame data size.
    #[wasm_bindgen(getter, js_name = dataSize)]
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Convert to JavaScript object.
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Object {
        let obj = Object::new();
        let _ = Reflect::set(&obj, &"width".into(), &self.width.into());
        let _ = Reflect::set(&obj, &"height".into(), &self.height.into());
        let _ = Reflect::set(&obj, &"pts".into(), &JsValue::from_f64(self.pts as f64));
        let _ = Reflect::set(
            &obj,
            &"duration".into(),
            &JsValue::from_f64(self.duration as f64),
        );
        let _ = Reflect::set(&obj, &"keyframe".into(), &self.keyframe.into());
        let _ = Reflect::set(&obj, &"pixelFormat".into(), &self.pixel_format.clone().into());
        obj
    }
}

/// Decoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderState {
    /// Decoder is ready for input.
    Ready,
    /// Decoder needs more data.
    NeedsData,
    /// Decoder has frames available.
    HasFrames,
    /// Decoder has reached end of stream.
    EndOfStream,
    /// Decoder encountered an error.
    Error,
}

/// WASM video decoder.
#[wasm_bindgen]
pub struct WasmDecoder {
    /// Decoder state.
    state: DecoderState,
    /// Codec being used.
    codec: String,
    /// Video width.
    width: u32,
    /// Video height.
    height: u32,
    /// Decoded frames buffer.
    frames: Vec<DecodedFrame>,
    /// Total frames decoded.
    frames_decoded: u64,
    /// Internal buffer for partial NAL units.
    buffer: Vec<u8>,
}

#[wasm_bindgen]
impl WasmDecoder {
    /// Create a new decoder for the specified codec.
    #[wasm_bindgen(constructor)]
    pub fn new(codec: &str) -> WasmResult<WasmDecoder> {
        let supported = matches!(codec.to_lowercase().as_str(), "h264" | "avc" | "h.264");

        if !supported {
            return Err(WasmError::not_supported(&format!(
                "Codec '{}' is not supported. Supported codecs: h264",
                codec
            )));
        }

        Ok(Self {
            state: DecoderState::Ready,
            codec: codec.to_lowercase(),
            width: 0,
            height: 0,
            frames: Vec::new(),
            frames_decoded: 0,
            buffer: Vec::new(),
        })
    }

    /// Configure the decoder with stream parameters.
    pub fn configure(&mut self, width: u32, height: u32) -> WasmResult<()> {
        if width == 0 || height == 0 {
            return Err(WasmError::invalid_config("Width and height must be non-zero"));
        }

        self.width = width;
        self.height = height;
        self.state = DecoderState::Ready;

        Ok(())
    }

    /// Decode a chunk of encoded data.
    pub fn decode(&mut self, data: &[u8]) -> WasmResult<u32> {
        if data.is_empty() {
            return Ok(0);
        }

        // Append to internal buffer
        self.buffer.extend_from_slice(data);

        // Try to decode complete NAL units
        let frames_before = self.frames.len();
        self.process_buffer()?;

        let new_frames = (self.frames.len() - frames_before) as u32;
        self.frames_decoded += new_frames as u64;

        if !self.frames.is_empty() {
            self.state = DecoderState::HasFrames;
        }

        Ok(new_frames)
    }

    /// Process internal buffer to extract frames.
    fn process_buffer(&mut self) -> WasmResult<()> {
        // Simplified NAL unit parsing for demonstration
        // In production, this would use the actual H.264 decoder

        // Look for NAL start codes (0x00 0x00 0x01 or 0x00 0x00 0x00 0x01)
        let mut pos = 0;
        while pos + 4 < self.buffer.len() {
            let is_start_code = (self.buffer[pos] == 0
                && self.buffer[pos + 1] == 0
                && self.buffer[pos + 2] == 1)
                || (self.buffer[pos] == 0
                    && self.buffer[pos + 1] == 0
                    && self.buffer[pos + 2] == 0
                    && self.buffer[pos + 3] == 1);

            if is_start_code {
                // For demo: create a placeholder frame every 4KB of data
                if self.buffer.len() > 4096 && self.frames_decoded % 30 == 0 {
                    let frame = DecodedFrame {
                        width: if self.width > 0 { self.width } else { 1920 },
                        height: if self.height > 0 { self.height } else { 1080 },
                        pts: (self.frames_decoded as i64) * 33333, // ~30fps
                        duration: 33333,
                        keyframe: self.frames_decoded % 30 == 0,
                        pixel_format: "yuv420p".to_string(),
                        data: vec![0; 1920 * 1080 * 3 / 2], // YUV420p placeholder
                    };
                    self.frames.push(frame);
                }
                break;
            }
            pos += 1;
        }

        // Clear processed data (simplified)
        if self.buffer.len() > 65536 {
            self.buffer.drain(..32768);
        }

        Ok(())
    }

    /// Check if decoder has frames available.
    #[wasm_bindgen(js_name = hasFrames)]
    pub fn has_frames(&self) -> bool {
        !self.frames.is_empty()
    }

    /// Get the next decoded frame.
    #[wasm_bindgen(js_name = getFrame)]
    pub fn get_frame(&mut self) -> Option<DecodedFrame> {
        if self.frames.is_empty() {
            self.state = DecoderState::NeedsData;
            None
        } else {
            Some(self.frames.remove(0))
        }
    }

    /// Get all available frames.
    #[wasm_bindgen(js_name = getFrames)]
    pub fn get_frames(&mut self) -> Array {
        let array = Array::new();
        for frame in self.frames.drain(..) {
            array.push(&frame.to_object());
        }
        self.state = DecoderState::NeedsData;
        array
    }

    /// Signal end of stream.
    pub fn flush(&mut self) -> WasmResult<u32> {
        // Process any remaining data
        self.process_buffer()?;

        self.state = DecoderState::EndOfStream;
        Ok(self.frames.len() as u32)
    }

    /// Reset the decoder.
    pub fn reset(&mut self) {
        self.state = DecoderState::Ready;
        self.frames.clear();
        self.buffer.clear();
        self.frames_decoded = 0;
    }

    /// Get decoder statistics.
    #[wasm_bindgen(getter)]
    pub fn stats(&self) -> Object {
        let obj = Object::new();
        let _ = Reflect::set(
            &obj,
            &"framesDecoded".into(),
            &JsValue::from_f64(self.frames_decoded as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"pendingFrames".into(),
            &JsValue::from_f64(self.frames.len() as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"bufferSize".into(),
            &JsValue::from_f64(self.buffer.len() as f64),
        );
        let _ = Reflect::set(&obj, &"codec".into(), &self.codec.clone().into());
        obj
    }

    /// Get the codec name.
    #[wasm_bindgen(getter)]
    pub fn codec(&self) -> String {
        self.codec.clone()
    }

    /// Get total frames decoded.
    #[wasm_bindgen(getter, js_name = framesDecoded)]
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }
}
