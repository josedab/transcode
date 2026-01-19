//! WASM video encoder wrapper.

use crate::decoder::DecodedFrame;
use crate::error::{WasmError, WasmResult};
use crate::options::VideoOptions;
use js_sys::{Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;

/// Encoded packet information.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Presentation timestamp in microseconds.
    pts: i64,
    /// Decode timestamp in microseconds.
    dts: i64,
    /// Packet duration in microseconds.
    duration: i64,
    /// Whether this is a keyframe.
    keyframe: bool,
    /// Encoded data.
    data: Vec<u8>,
}

#[wasm_bindgen]
impl EncodedPacket {
    /// Get presentation timestamp.
    #[wasm_bindgen(getter)]
    pub fn pts(&self) -> i64 {
        self.pts
    }

    /// Get decode timestamp.
    #[wasm_bindgen(getter)]
    pub fn dts(&self) -> i64 {
        self.dts
    }

    /// Get packet duration.
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> i64 {
        self.duration
    }

    /// Check if this is a keyframe.
    #[wasm_bindgen(getter)]
    pub fn keyframe(&self) -> bool {
        self.keyframe
    }

    /// Get packet data as Uint8Array.
    #[wasm_bindgen(js_name = getData)]
    pub fn get_data(&self) -> Uint8Array {
        Uint8Array::from(self.data.as_slice())
    }

    /// Get packet data size.
    #[wasm_bindgen(getter, js_name = dataSize)]
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Convert to JavaScript object.
    #[wasm_bindgen(js_name = toObject)]
    pub fn to_object(&self) -> Object {
        let obj = Object::new();
        let _ = Reflect::set(&obj, &"pts".into(), &JsValue::from_f64(self.pts as f64));
        let _ = Reflect::set(&obj, &"dts".into(), &JsValue::from_f64(self.dts as f64));
        let _ = Reflect::set(
            &obj,
            &"duration".into(),
            &JsValue::from_f64(self.duration as f64),
        );
        let _ = Reflect::set(&obj, &"keyframe".into(), &self.keyframe.into());
        let _ = Reflect::set(
            &obj,
            &"size".into(),
            &JsValue::from_f64(self.data.len() as f64),
        );
        obj
    }
}

/// Encoder configuration.
#[derive(Debug, Clone)]
struct EncoderConfig {
    codec: String,
    width: u32,
    height: u32,
    bitrate: u32,
    framerate: f32,
    quality: u8,
    keyframe_interval: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            codec: "h264".to_string(),
            width: 1920,
            height: 1080,
            bitrate: 2_000_000,
            framerate: 30.0,
            quality: 5,
            keyframe_interval: 30,
        }
    }
}

impl From<&VideoOptions> for EncoderConfig {
    fn from(options: &VideoOptions) -> Self {
        Self {
            codec: options.codec(),
            width: if options.width() > 0 {
                options.width()
            } else {
                1920
            },
            height: if options.height() > 0 {
                options.height()
            } else {
                1080
            },
            bitrate: options.bitrate(),
            framerate: if options.framerate() > 0.0 {
                options.framerate()
            } else {
                30.0
            },
            quality: options.quality(),
            keyframe_interval: options.keyframe_interval(),
        }
    }
}

/// WASM video encoder.
#[wasm_bindgen]
pub struct WasmEncoder {
    /// Encoder configuration.
    config: EncoderConfig,
    /// Encoded packets buffer.
    packets: Vec<EncodedPacket>,
    /// Total frames encoded.
    frames_encoded: u64,
    /// Total bytes produced.
    bytes_produced: u64,
    /// Whether encoder is initialized.
    initialized: bool,
}

#[wasm_bindgen]
impl WasmEncoder {
    /// Create a new encoder with default settings.
    #[wasm_bindgen(constructor)]
    pub fn new(codec: &str) -> WasmResult<WasmEncoder> {
        let supported = matches!(codec.to_lowercase().as_str(), "h264" | "avc" | "h.264");

        if !supported {
            return Err(WasmError::not_supported(&format!(
                "Codec '{}' is not supported for encoding. Supported codecs: h264",
                codec
            )));
        }

        Ok(Self {
            config: EncoderConfig {
                codec: codec.to_lowercase(),
                ..Default::default()
            },
            packets: Vec::new(),
            frames_encoded: 0,
            bytes_produced: 0,
            initialized: false,
        })
    }

    /// Create encoder from video options.
    #[wasm_bindgen(js_name = fromOptions)]
    pub fn from_options(options: &VideoOptions) -> WasmResult<WasmEncoder> {
        let supported = matches!(
            options.codec().to_lowercase().as_str(),
            "h264" | "avc" | "h.264"
        );

        if !supported {
            return Err(WasmError::not_supported(&format!(
                "Codec '{}' is not supported for encoding",
                options.codec()
            )));
        }

        Ok(Self {
            config: EncoderConfig::from(options),
            packets: Vec::new(),
            frames_encoded: 0,
            bytes_produced: 0,
            initialized: false,
        })
    }

    /// Configure the encoder.
    pub fn configure(
        &mut self,
        width: u32,
        height: u32,
        bitrate: u32,
        framerate: f32,
    ) -> WasmResult<()> {
        if width == 0 || height == 0 {
            return Err(WasmError::invalid_config("Width and height must be non-zero"));
        }

        self.config.width = width;
        self.config.height = height;
        self.config.bitrate = bitrate;
        self.config.framerate = if framerate > 0.0 { framerate } else { 30.0 };
        self.initialized = true;

        Ok(())
    }

    /// Set quality preset (0-9).
    #[wasm_bindgen(js_name = setQuality)]
    pub fn set_quality(&mut self, quality: u8) {
        self.config.quality = quality.min(9);
    }

    /// Set keyframe interval.
    #[wasm_bindgen(js_name = setKeyframeInterval)]
    pub fn set_keyframe_interval(&mut self, interval: u32) {
        self.config.keyframe_interval = interval.max(1);
    }

    /// Encode a decoded frame.
    #[wasm_bindgen(js_name = encodeFrame)]
    pub fn encode_frame(&mut self, frame: &DecodedFrame) -> WasmResult<bool> {
        if !self.initialized {
            // Auto-initialize from frame
            self.config.width = frame.width();
            self.config.height = frame.height();
            self.initialized = true;
        }

        // Create encoded packet (placeholder implementation)
        let is_keyframe = self.frames_encoded % self.config.keyframe_interval as u64 == 0;

        // Estimate encoded size based on quality and keyframe status
        let base_size = (self.config.width * self.config.height) as usize;
        let size_factor = if is_keyframe { 0.1 } else { 0.03 };
        let quality_factor = 1.0 + (self.config.quality as f64 * 0.1);
        let estimated_size = (base_size as f64 * size_factor * quality_factor) as usize;

        let packet = EncodedPacket {
            pts: frame.pts(),
            dts: frame.pts(), // Simplified: no B-frames
            duration: frame.duration(),
            keyframe: is_keyframe,
            data: vec![0; estimated_size.max(100)], // Placeholder data
        };

        self.bytes_produced += packet.data.len() as u64;
        self.packets.push(packet);
        self.frames_encoded += 1;

        Ok(true)
    }

    /// Encode raw frame data.
    #[wasm_bindgen(js_name = encodeRaw)]
    pub fn encode_raw(
        &mut self,
        _data: &[u8],
        pts: i64,
        duration: i64,
        keyframe: bool,
    ) -> WasmResult<bool> {
        if !self.initialized {
            return Err(WasmError::invalid_config(
                "Encoder must be configured before encoding raw data",
            ));
        }

        // Create encoded packet (placeholder implementation)
        let is_keyframe =
            keyframe || self.frames_encoded % self.config.keyframe_interval as u64 == 0;

        let base_size = (self.config.width * self.config.height) as usize;
        let size_factor = if is_keyframe { 0.1 } else { 0.03 };
        let estimated_size = (base_size as f64 * size_factor) as usize;

        let packet = EncodedPacket {
            pts,
            dts: pts,
            duration,
            keyframe: is_keyframe,
            data: vec![0; estimated_size.max(100)],
        };

        self.bytes_produced += packet.data.len() as u64;
        self.packets.push(packet);
        self.frames_encoded += 1;

        Ok(true)
    }

    /// Check if encoder has packets available.
    #[wasm_bindgen(js_name = hasPackets)]
    pub fn has_packets(&self) -> bool {
        !self.packets.is_empty()
    }

    /// Get the next encoded packet.
    #[wasm_bindgen(js_name = getPacket)]
    pub fn get_packet(&mut self) -> Option<EncodedPacket> {
        if self.packets.is_empty() {
            None
        } else {
            Some(self.packets.remove(0))
        }
    }

    /// Flush any remaining frames.
    pub fn flush(&mut self) -> WasmResult<u32> {
        // In a real encoder, this would flush delayed frames
        Ok(self.packets.len() as u32)
    }

    /// Reset the encoder.
    pub fn reset(&mut self) {
        self.packets.clear();
        self.frames_encoded = 0;
        self.bytes_produced = 0;
    }

    /// Get encoder statistics.
    #[wasm_bindgen(getter)]
    pub fn stats(&self) -> Object {
        let obj = Object::new();
        let _ = Reflect::set(
            &obj,
            &"framesEncoded".into(),
            &JsValue::from_f64(self.frames_encoded as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"bytesProduced".into(),
            &JsValue::from_f64(self.bytes_produced as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"pendingPackets".into(),
            &JsValue::from_f64(self.packets.len() as f64),
        );
        let _ = Reflect::set(&obj, &"codec".into(), &self.config.codec.clone().into());
        let _ = Reflect::set(
            &obj,
            &"width".into(),
            &JsValue::from_f64(self.config.width as f64),
        );
        let _ = Reflect::set(
            &obj,
            &"height".into(),
            &JsValue::from_f64(self.config.height as f64),
        );

        // Calculate average bitrate
        if self.frames_encoded > 0 {
            let duration_s = self.frames_encoded as f64 / self.config.framerate as f64;
            let avg_bitrate = (self.bytes_produced as f64 * 8.0) / duration_s;
            let _ = Reflect::set(&obj, &"avgBitrate".into(), &JsValue::from_f64(avg_bitrate));
        }

        obj
    }

    /// Get the codec name.
    #[wasm_bindgen(getter)]
    pub fn codec(&self) -> String {
        self.config.codec.clone()
    }

    /// Get total frames encoded.
    #[wasm_bindgen(getter, js_name = framesEncoded)]
    pub fn frames_encoded(&self) -> u64 {
        self.frames_encoded
    }

    /// Get total bytes produced.
    #[wasm_bindgen(getter, js_name = bytesProduced)]
    pub fn bytes_produced(&self) -> u64 {
        self.bytes_produced
    }
}
