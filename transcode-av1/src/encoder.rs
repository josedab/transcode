//! AV1 encoder implementation using rav1e.

use crate::config::{Av1Config, RateControlMode};
use crate::error::Av1Error;
use crate::Result;

#[cfg(feature = "encoder")]
use rav1e::prelude::*;

/// Encoded AV1 packet.
#[derive(Debug, Clone)]
pub struct Av1Packet {
    /// Encoded data.
    pub data: Vec<u8>,
    /// Presentation timestamp.
    pub pts: i64,
    /// Whether this is a keyframe.
    pub keyframe: bool,
    /// Frame type description.
    pub frame_type: Av1FrameType,
}

/// AV1 frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Av1FrameType {
    /// Keyframe (I-frame).
    Key,
    /// Inter frame (P-frame).
    Inter,
    /// Intra-only frame.
    IntraOnly,
    /// Switch frame.
    Switch,
}

/// Encoder statistics.
#[derive(Debug, Clone, Default)]
pub struct Av1EncoderStats {
    /// Frames encoded.
    pub frames_encoded: u64,
    /// Bytes produced.
    pub bytes_produced: u64,
    /// Keyframes produced.
    pub keyframes: u64,
    /// Average bitrate in bits per second.
    pub avg_bitrate: f64,
    /// Average PSNR (if calculated).
    pub avg_psnr: Option<f64>,
    /// Encoding time in seconds.
    pub encoding_time_s: f64,
}

/// AV1 encoder using rav1e.
pub struct Av1Encoder {
    /// Encoder configuration.
    config: Av1Config,
    /// Internal rav1e context.
    #[cfg(feature = "encoder")]
    context: Option<Context<u8>>,
    /// Internal rav1e context for 10-bit.
    #[cfg(feature = "encoder")]
    context_10bit: Option<Context<u16>>,
    /// Frame count.
    frame_count: u64,
    /// Bytes produced.
    bytes_produced: u64,
    /// Keyframes produced.
    keyframes: u64,
    /// Whether encoder is flushing.
    #[allow(dead_code)]
    flushing: bool,
    /// Whether encoder is finished.
    finished: bool,
}

impl Av1Encoder {
    /// Create a new AV1 encoder with the given configuration.
    pub fn new(config: Av1Config) -> Result<Self> {
        config.validate()?;

        #[cfg(feature = "encoder")]
        {
            let enc_config = Self::build_rav1e_config(&config)?;

            if config.bit_depth == 10 {
                let context: Context<u16> = enc_config.new_context()
                    .map_err(|e| Av1Error::EncoderError(e.to_string()))?;

                Ok(Self {
                    config,
                    context: None,
                    context_10bit: Some(context),
                    frame_count: 0,
                    bytes_produced: 0,
                    keyframes: 0,
                    flushing: false,
                    finished: false,
                })
            } else {
                let context: Context<u8> = enc_config.new_context()
                    .map_err(|e| Av1Error::EncoderError(e.to_string()))?;

                Ok(Self {
                    config,
                    context: Some(context),
                    context_10bit: None,
                    frame_count: 0,
                    bytes_produced: 0,
                    keyframes: 0,
                    flushing: false,
                    finished: false,
                })
            }
        }

        #[cfg(not(feature = "encoder"))]
        {
            let _ = config;
            Err(Av1Error::EncoderError(
                "AV1 encoder feature not enabled".into(),
            ))
        }
    }

    /// Build rav1e configuration from our config.
    #[cfg(feature = "encoder")]
    #[allow(clippy::field_reassign_with_default)]
    fn build_rav1e_config(config: &Av1Config) -> Result<Config> {
        let mut enc = EncoderConfig::default();

        enc.width = config.width as usize;
        enc.height = config.height as usize;
        enc.speed_settings = SpeedSettings::from_preset(config.preset.to_speed());

        enc.time_base = Rational::new(
            config.framerate_den as u64,
            config.framerate_num as u64,
        );

        // Rate control
        match &config.rate_control {
            RateControlMode::ConstantQuality { quantizer } => {
                enc.quantizer = *quantizer as usize;
                enc.min_quantizer = *quantizer;
            }
            RateControlMode::Vbr { bitrate } | RateControlMode::Cbr { bitrate } => {
                enc.bitrate = (*bitrate / 1000) as i32; // rav1e uses kbps
            }
            _ => {}
        }

        // Keyframe settings
        enc.max_key_frame_interval = config.keyframe_interval as u64;
        enc.min_key_frame_interval = config.min_keyframe_interval as u64;

        // Tiles
        enc.tile_cols = 1 << config.tile_cols_log2;
        enc.tile_rows = 1 << config.tile_rows_log2;

        // Low latency
        enc.low_latency = config.low_latency;

        // Threads
        let threads = if config.threads == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        } else {
            config.threads
        };

        Ok(Config::new()
            .with_encoder_config(enc)
            .with_threads(threads))
    }

    /// Encode a frame (8-bit YUV420).
    pub fn encode_frame(
        &mut self,
        y_plane: &[u8],
        u_plane: &[u8],
        v_plane: &[u8],
        pts: i64,
    ) -> Result<Option<Av1Packet>> {
        #[cfg(feature = "encoder")]
        {
            if self.config.bit_depth == 10 {
                return Err(Av1Error::InvalidFrame(
                    "Use encode_frame_10bit for 10-bit content".into(),
                ));
            }

            let context = self.context.as_mut()
                .ok_or_else(|| Av1Error::EncoderError("Encoder not initialized".into()))?;

            // Create frame
            let mut frame = context.new_frame();

            // Copy Y plane
            for (dst_row, src_row) in frame.planes[0]
                .rows_iter_mut()
                .zip(y_plane.chunks(self.config.width as usize))
            {
                dst_row[..src_row.len()].copy_from_slice(src_row);
            }

            // Copy U plane
            let uv_width = (self.config.width / 2) as usize;
            for (dst_row, src_row) in frame.planes[1]
                .rows_iter_mut()
                .zip(u_plane.chunks(uv_width))
            {
                dst_row[..src_row.len()].copy_from_slice(src_row);
            }

            // Copy V plane
            for (dst_row, src_row) in frame.planes[2]
                .rows_iter_mut()
                .zip(v_plane.chunks(uv_width))
            {
                dst_row[..src_row.len()].copy_from_slice(src_row);
            }

            // Send frame to encoder
            context.send_frame(frame)
                .map_err(|e| Av1Error::EncoderError(e.to_string()))?;

            self.frame_count += 1;

            // Try to receive a packet
            self.receive_packet(pts)
        }

        #[cfg(not(feature = "encoder"))]
        {
            let _ = (y_plane, u_plane, v_plane, pts);
            Err(Av1Error::EncoderError("Encoder feature not enabled".into()))
        }
    }

    /// Encode a 10-bit frame (YUV420 10-bit).
    pub fn encode_frame_10bit(
        &mut self,
        y_plane: &[u16],
        u_plane: &[u16],
        v_plane: &[u16],
        pts: i64,
    ) -> Result<Option<Av1Packet>> {
        #[cfg(feature = "encoder")]
        {
            if self.config.bit_depth != 10 {
                return Err(Av1Error::InvalidFrame(
                    "Use encode_frame for 8-bit content".into(),
                ));
            }

            let context = self.context_10bit.as_mut()
                .ok_or_else(|| Av1Error::EncoderError("Encoder not initialized".into()))?;

            // Create frame
            let mut frame = context.new_frame();

            // Copy Y plane
            for (dst_row, src_row) in frame.planes[0]
                .rows_iter_mut()
                .zip(y_plane.chunks(self.config.width as usize))
            {
                dst_row[..src_row.len()].copy_from_slice(src_row);
            }

            // Copy U plane
            let uv_width = (self.config.width / 2) as usize;
            for (dst_row, src_row) in frame.planes[1]
                .rows_iter_mut()
                .zip(u_plane.chunks(uv_width))
            {
                dst_row[..src_row.len()].copy_from_slice(src_row);
            }

            // Copy V plane
            for (dst_row, src_row) in frame.planes[2]
                .rows_iter_mut()
                .zip(v_plane.chunks(uv_width))
            {
                dst_row[..src_row.len()].copy_from_slice(src_row);
            }

            // Send frame to encoder
            context.send_frame(frame)
                .map_err(|e| Av1Error::EncoderError(e.to_string()))?;

            self.frame_count += 1;

            // Try to receive a packet
            self.receive_packet_10bit(pts)
        }

        #[cfg(not(feature = "encoder"))]
        {
            let _ = (y_plane, u_plane, v_plane, pts);
            Err(Av1Error::EncoderError("Encoder feature not enabled".into()))
        }
    }

    /// Receive packet from 8-bit context.
    #[cfg(feature = "encoder")]
    fn receive_packet(&mut self, pts: i64) -> Result<Option<Av1Packet>> {
        let context = self.context.as_mut()
            .ok_or_else(|| Av1Error::EncoderError("Encoder not initialized".into()))?;

        match context.receive_packet() {
            Ok(packet) => {
                let keyframe = packet.frame_type == FrameType::KEY;
                if keyframe {
                    self.keyframes += 1;
                }
                self.bytes_produced += packet.data.len() as u64;

                Ok(Some(Av1Packet {
                    data: packet.data,
                    pts,
                    keyframe,
                    frame_type: match packet.frame_type {
                        FrameType::KEY => Av1FrameType::Key,
                        FrameType::INTER => Av1FrameType::Inter,
                        FrameType::INTRA_ONLY => Av1FrameType::IntraOnly,
                        FrameType::SWITCH => Av1FrameType::Switch,
                    },
                }))
            }
            Err(EncoderStatus::NeedMoreData) => Ok(None),
            Err(EncoderStatus::Encoded) => Ok(None),
            Err(e) => Err(Av1Error::EncoderError(e.to_string())),
        }
    }

    /// Receive packet from 10-bit context.
    #[cfg(feature = "encoder")]
    fn receive_packet_10bit(&mut self, pts: i64) -> Result<Option<Av1Packet>> {
        let context = self.context_10bit.as_mut()
            .ok_or_else(|| Av1Error::EncoderError("Encoder not initialized".into()))?;

        match context.receive_packet() {
            Ok(packet) => {
                let keyframe = packet.frame_type == FrameType::KEY;
                if keyframe {
                    self.keyframes += 1;
                }
                self.bytes_produced += packet.data.len() as u64;

                Ok(Some(Av1Packet {
                    data: packet.data,
                    pts,
                    keyframe,
                    frame_type: match packet.frame_type {
                        FrameType::KEY => Av1FrameType::Key,
                        FrameType::INTER => Av1FrameType::Inter,
                        FrameType::INTRA_ONLY => Av1FrameType::IntraOnly,
                        FrameType::SWITCH => Av1FrameType::Switch,
                    },
                }))
            }
            Err(EncoderStatus::NeedMoreData) => Ok(None),
            Err(EncoderStatus::Encoded) => Ok(None),
            Err(e) => Err(Av1Error::EncoderError(e.to_string())),
        }
    }

    /// Flush the encoder and receive remaining packets.
    pub fn flush(&mut self) -> Result<Vec<Av1Packet>> {
        #[cfg(feature = "encoder")]
        {
            self.flushing = true;
            let mut packets = Vec::new();

            if self.config.bit_depth == 10 {
                if let Some(ref mut context) = self.context_10bit {
                    context.flush();

                    loop {
                        match context.receive_packet() {
                            Ok(packet) => {
                                let keyframe = packet.frame_type == FrameType::KEY;
                                if keyframe {
                                    self.keyframes += 1;
                                }
                                self.bytes_produced += packet.data.len() as u64;

                                packets.push(Av1Packet {
                                    data: packet.data,
                                    pts: packet.input_frameno as i64,
                                    keyframe,
                                    frame_type: match packet.frame_type {
                                        FrameType::KEY => Av1FrameType::Key,
                                        FrameType::INTER => Av1FrameType::Inter,
                                        FrameType::INTRA_ONLY => Av1FrameType::IntraOnly,
                                        FrameType::SWITCH => Av1FrameType::Switch,
                                    },
                                });
                            }
                            Err(EncoderStatus::LimitReached) => break,
                            Err(EncoderStatus::Encoded) => continue,
                            Err(EncoderStatus::NeedMoreData) => continue,
                            Err(e) => return Err(Av1Error::EncoderError(e.to_string())),
                        }
                    }
                }
            } else if let Some(ref mut context) = self.context {
                context.flush();

                loop {
                    match context.receive_packet() {
                        Ok(packet) => {
                            let keyframe = packet.frame_type == FrameType::KEY;
                            if keyframe {
                                self.keyframes += 1;
                            }
                            self.bytes_produced += packet.data.len() as u64;

                            packets.push(Av1Packet {
                                data: packet.data,
                                pts: packet.input_frameno as i64,
                                keyframe,
                                frame_type: match packet.frame_type {
                                    FrameType::KEY => Av1FrameType::Key,
                                    FrameType::INTER => Av1FrameType::Inter,
                                    FrameType::INTRA_ONLY => Av1FrameType::IntraOnly,
                                    FrameType::SWITCH => Av1FrameType::Switch,
                                },
                            });
                        }
                        Err(EncoderStatus::LimitReached) => break,
                        Err(EncoderStatus::Encoded) => continue,
                        Err(EncoderStatus::NeedMoreData) => continue,
                        Err(e) => return Err(Av1Error::EncoderError(e.to_string())),
                    }
                }
            }

            self.finished = true;
            Ok(packets)
        }

        #[cfg(not(feature = "encoder"))]
        Err(Av1Error::EncoderError("Encoder feature not enabled".into()))
    }

    /// Get encoder statistics.
    pub fn stats(&self) -> Av1EncoderStats {
        let framerate = self.config.framerate_num as f64 / self.config.framerate_den as f64;
        let duration = self.frame_count as f64 / framerate;

        Av1EncoderStats {
            frames_encoded: self.frame_count,
            bytes_produced: self.bytes_produced,
            keyframes: self.keyframes,
            avg_bitrate: if duration > 0.0 {
                (self.bytes_produced as f64 * 8.0) / duration
            } else {
                0.0
            },
            avg_psnr: None,
            encoding_time_s: duration,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &Av1Config {
        &self.config
    }

    /// Check if encoder is finished.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get sequence header (for container initialization).
    pub fn get_sequence_header(&self) -> Result<Vec<u8>> {
        #[cfg(feature = "encoder")]
        {
            if self.config.bit_depth == 10 {
                if let Some(ref context) = self.context_10bit {
                    return Ok(context.container_sequence_header());
                }
            } else if let Some(ref context) = self.context {
                return Ok(context.container_sequence_header());
            }
        }

        Err(Av1Error::EncoderError("Encoder not initialized".into()))
    }
}

#[cfg(all(test, feature = "encoder"))]
mod tests {
    use super::*;
    use crate::config::Av1Preset;

    #[test]
    fn test_encoder_creation() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::UltraFast)
            .with_quality(32);

        let encoder = Av1Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encode_frame() {
        let config = Av1Config::new(320, 240)
            .with_preset(Av1Preset::UltraFast)
            .with_quality(32);

        let mut encoder = Av1Encoder::new(config).unwrap();

        // Create a simple test frame
        let y_plane = vec![128u8; 320 * 240];
        let u_plane = vec![128u8; 160 * 120];
        let v_plane = vec![128u8; 160 * 120];

        let result = encoder.encode_frame(&y_plane, &u_plane, &v_plane, 0);
        assert!(result.is_ok());
    }
}
