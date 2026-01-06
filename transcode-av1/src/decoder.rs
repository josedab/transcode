//! AV1 decoder implementation using dav1d.

use crate::error::Av1Error;
use crate::Result;

#[cfg(feature = "decoder")]
use dav1d::{Decoder as Dav1dDecoder, Settings as Dav1dSettings, PlanarImageComponent};

/// Decoded AV1 frame.
#[derive(Debug, Clone)]
pub struct Av1DecodedFrame {
    /// Y (luma) plane data.
    pub y_plane: Vec<u8>,
    /// U (chroma) plane data.
    pub u_plane: Vec<u8>,
    /// V (chroma) plane data.
    pub v_plane: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Y plane stride.
    pub y_stride: usize,
    /// UV plane stride.
    pub uv_stride: usize,
    /// Bit depth (8 or 10).
    pub bit_depth: u8,
    /// Presentation timestamp.
    pub pts: i64,
    /// Frame duration.
    pub duration: i64,
    /// Whether this is a keyframe.
    pub keyframe: bool,
}

/// Decoder statistics.
#[derive(Debug, Clone, Default)]
pub struct Av1DecoderStats {
    /// Frames decoded.
    pub frames_decoded: u64,
    /// Bytes consumed.
    pub bytes_consumed: u64,
    /// Keyframes decoded.
    pub keyframes: u64,
    /// Decode errors encountered.
    pub decode_errors: u64,
}

/// AV1 decoder configuration.
#[derive(Debug, Clone)]
pub struct Av1DecoderConfig {
    /// Number of frame threads (0 = auto).
    pub frame_threads: u32,
    /// Number of tile threads (0 = auto).
    pub tile_threads: u32,
    /// Apply film grain synthesis.
    pub apply_grain: bool,
    /// Operating point for scalable AV1.
    pub operating_point: u8,
    /// Output all layers for scalable AV1.
    pub all_layers: bool,
}

impl Default for Av1DecoderConfig {
    fn default() -> Self {
        Self {
            frame_threads: 0,
            tile_threads: 0,
            apply_grain: true,
            operating_point: 0,
            all_layers: false,
        }
    }
}

impl Av1DecoderConfig {
    /// Create a new decoder configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of frame threads.
    pub fn with_frame_threads(mut self, threads: u32) -> Self {
        self.frame_threads = threads;
        self
    }

    /// Set the number of tile threads.
    pub fn with_tile_threads(mut self, threads: u32) -> Self {
        self.tile_threads = threads;
        self
    }

    /// Enable or disable film grain synthesis.
    pub fn with_film_grain(mut self, apply: bool) -> Self {
        self.apply_grain = apply;
        self
    }

    /// Set operating point for scalable AV1.
    pub fn with_operating_point(mut self, point: u8) -> Self {
        self.operating_point = point;
        self
    }
}

/// AV1 decoder using dav1d.
pub struct Av1Decoder {
    /// Decoder configuration.
    config: Av1DecoderConfig,
    /// Internal dav1d decoder context.
    #[cfg(feature = "decoder")]
    decoder: Dav1dDecoder,
    /// Statistics.
    stats: Av1DecoderStats,
    /// Whether we've seen a sequence header.
    #[allow(dead_code)]
    initialized: bool,
    /// Last known width.
    width: u32,
    /// Last known height.
    height: u32,
}

impl Av1Decoder {
    /// Create a new AV1 decoder with default configuration.
    pub fn new() -> Result<Self> {
        Self::with_config(Av1DecoderConfig::default())
    }

    /// Create a new AV1 decoder with the given configuration.
    pub fn with_config(config: Av1DecoderConfig) -> Result<Self> {
        #[cfg(feature = "decoder")]
        {
            let mut settings = Dav1dSettings::new();

            if config.frame_threads > 0 {
                settings.set_n_threads(config.frame_threads as usize);
            }

            settings.set_apply_grain(config.apply_grain);
            settings.set_operating_point(config.operating_point as usize);
            settings.set_all_layers(config.all_layers);

            let decoder = Dav1dDecoder::with_settings(&settings)
                .map_err(|e| Av1Error::DecoderError(format!("Failed to create decoder: {}", e)))?;

            Ok(Self {
                config,
                decoder,
                stats: Av1DecoderStats::default(),
                initialized: false,
                width: 0,
                height: 0,
            })
        }

        #[cfg(not(feature = "decoder"))]
        {
            let _ = config;
            Err(Av1Error::DecoderError(
                "AV1 decoder feature not enabled".into(),
            ))
        }
    }

    /// Send data to the decoder.
    ///
    /// This feeds AV1 OBU data to the decoder. Call `get_frame()` to retrieve
    /// decoded frames.
    pub fn send_data(&mut self, data: &[u8], pts: i64, duration: i64) -> Result<()> {
        #[cfg(feature = "decoder")]
        {
            self.stats.bytes_consumed += data.len() as u64;

            self.decoder
                .send_data(data.to_vec(), Some(pts), Some(duration), None)
                .map_err(|e| Av1Error::DecoderError(format!("Failed to send data: {}", e)))?;

            Ok(())
        }

        #[cfg(not(feature = "decoder"))]
        {
            let _ = (data, pts, duration);
            Err(Av1Error::DecoderError("Decoder feature not enabled".into()))
        }
    }

    /// Try to get a decoded frame.
    ///
    /// Returns `Ok(None)` if no frame is available yet (need more data).
    pub fn get_frame(&mut self) -> Result<Option<Av1DecodedFrame>> {
        #[cfg(feature = "decoder")]
        {
            match self.decoder.get_picture() {
                Ok(picture) => {
                    self.initialized = true;
                    self.width = picture.width() as u32;
                    self.height = picture.height() as u32;

                    let y_plane = picture.plane(PlanarImageComponent::Y);
                    let u_plane = picture.plane(PlanarImageComponent::U);
                    let v_plane = picture.plane(PlanarImageComponent::V);

                    let bit_depth = picture.bit_depth() as u8;
                    let keyframe = picture.frame_type() == dav1d::FrameType::KEY;

                    if keyframe {
                        self.stats.keyframes += 1;
                    }
                    self.stats.frames_decoded += 1;

                    // Extract plane data
                    let (y_data, y_stride) = extract_plane_data(&y_plane, bit_depth);
                    let (u_data, uv_stride) = extract_plane_data(&u_plane, bit_depth);
                    let (v_data, _) = extract_plane_data(&v_plane, bit_depth);

                    Ok(Some(Av1DecodedFrame {
                        y_plane: y_data,
                        u_plane: u_data,
                        v_plane: v_data,
                        width: self.width,
                        height: self.height,
                        y_stride,
                        uv_stride,
                        bit_depth,
                        pts: picture.timestamp().unwrap_or(0),
                        duration: picture.duration(),
                        keyframe,
                    }))
                }
                Err(e) if e.is_again() => Ok(None),
                Err(e) => {
                    self.stats.decode_errors += 1;
                    Err(Av1Error::DecoderError(format!("Decode error: {}", e)))
                }
            }
        }

        #[cfg(not(feature = "decoder"))]
        Err(Av1Error::DecoderError("Decoder feature not enabled".into()))
    }

    /// Flush the decoder and get remaining frames.
    pub fn flush(&mut self) -> Result<Vec<Av1DecodedFrame>> {
        #[cfg(feature = "decoder")]
        {
            self.decoder.flush();

            let mut frames = Vec::new();
            while let Some(frame) = self.get_frame()? {
                frames.push(frame);
            }

            Ok(frames)
        }

        #[cfg(not(feature = "decoder"))]
        Err(Av1Error::DecoderError("Decoder feature not enabled".into()))
    }

    /// Get decoder statistics.
    pub fn stats(&self) -> &Av1DecoderStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &Av1DecoderConfig {
        &self.config
    }

    /// Get the video width (after decoding at least one frame).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the video height (after decoding at least one frame).
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) -> Result<()> {
        #[cfg(feature = "decoder")]
        {
            self.decoder.flush();
            self.initialized = false;
            Ok(())
        }

        #[cfg(not(feature = "decoder"))]
        Err(Av1Error::DecoderError("Decoder feature not enabled".into()))
    }
}

/// Extract plane data from a dav1d plane.
#[cfg(feature = "decoder")]
fn extract_plane_data(plane: &dav1d::Plane, bit_depth: u8) -> (Vec<u8>, usize) {
    let stride = plane.stride() as usize;
    let height = plane.height() as usize;

    if bit_depth <= 8 {
        let data: Vec<u8> = plane.as_ref().to_vec();
        (data, stride)
    } else {
        // For 10-bit content, dav1d stores as u16, we convert to bytes
        let u16_data: &[u16] = bytemuck_cast_slice(plane.as_ref());
        let data: Vec<u8> = u16_data.iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        (data, stride * 2)
    }
}

/// Helper to cast u8 slice to u16 slice for 10-bit content.
#[cfg(feature = "decoder")]
fn bytemuck_cast_slice(bytes: &[u8]) -> &[u16] {
    if bytes.len() % 2 != 0 {
        return &[];
    }
    // Safety: We're reinterpreting aligned bytes as u16
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const u16,
            bytes.len() / 2
        )
    }
}

#[cfg(all(test, feature = "decoder"))]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = Av1Decoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_decoder_config() {
        let config = Av1DecoderConfig::new()
            .with_frame_threads(4)
            .with_tile_threads(2)
            .with_film_grain(false);

        assert_eq!(config.frame_threads, 4);
        assert_eq!(config.tile_threads, 2);
        assert!(!config.apply_grain);
    }

    #[test]
    fn test_decoder_with_config() {
        let config = Av1DecoderConfig::new()
            .with_frame_threads(2);

        let decoder = Av1Decoder::with_config(config);
        assert!(decoder.is_ok());
    }
}
