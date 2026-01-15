//! Common codec traits.
//!
//! This module defines the core traits for video and audio codecs:
//!
//! - [`VideoDecoder`] / [`VideoEncoder`] - Video codec traits
//! - [`AudioDecoder`] / [`AudioEncoder`] - Audio codec traits
//! - [`VideoDecoderExt`] / [`AudioDecoderExt`] - Extension traits for buffer reuse
//! - [`AudioEncoderExt`] - Extension trait for audio encoder buffer reuse
//!
//! # Buffer Reuse
//!
//! The `*Ext` traits provide `decode_into` and `flush_into` methods that write
//! to pre-allocated buffers instead of allocating new vectors. This reduces
//! allocator pressure in hot transcoding loops.
//!
//! ```ignore
//! let mut decoder = H264Decoder::new()?;
//! let mut frames = Vec::new();
//!
//! for packet in packets {
//!     decoder.decode_into(&packet, &mut frames)?;
//!     for frame in frames.drain(..) {
//!         process(frame);
//!     }
//! }
//! ```

use transcode_core::{Frame, Packet, Sample, SampleFormat, Result};

/// Information about a codec.
#[derive(Debug, Clone)]
pub struct CodecInfo {
    /// Codec name.
    pub name: &'static str,
    /// Long name/description.
    pub long_name: &'static str,
    /// Whether this codec supports encoding.
    pub can_encode: bool,
    /// Whether this codec supports decoding.
    pub can_decode: bool,
}

/// Common trait for video decoders.
pub trait VideoDecoder: Send {
    /// Get codec information.
    fn codec_info(&self) -> CodecInfo;

    /// Decode a packet into frames.
    ///
    /// May return zero or more frames depending on the codec.
    fn decode(&mut self, packet: &Packet<'_>) -> Result<Vec<Frame>>;

    /// Flush the decoder, returning any buffered frames.
    fn flush(&mut self) -> Result<Vec<Frame>>;

    /// Reset the decoder state.
    fn reset(&mut self);
}

/// Common trait for video encoders.
pub trait VideoEncoder: Send {
    /// Get codec information.
    fn codec_info(&self) -> CodecInfo;

    /// Encode a frame into packets.
    fn encode(&mut self, frame: &Frame) -> Result<Vec<Packet<'static>>>;

    /// Flush the encoder, returning any buffered packets.
    fn flush(&mut self) -> Result<Vec<Packet<'static>>>;

    /// Reset the encoder state.
    fn reset(&mut self);

    /// Get the codec-specific configuration data (e.g., SPS/PPS for H.264).
    fn extra_data(&self) -> Option<&[u8]>;
}

/// Common trait for audio decoders.
pub trait AudioDecoder: Send {
    /// Get codec information.
    fn codec_info(&self) -> CodecInfo;

    /// Get decoder name.
    fn name(&self) -> &str;

    /// Get sample rate.
    fn sample_rate(&self) -> u32;

    /// Get number of channels.
    fn channels(&self) -> u8;

    /// Get sample format.
    fn sample_format(&self) -> SampleFormat;

    /// Set codec-specific extra data (e.g., AudioSpecificConfig for AAC).
    fn set_extra_data(&mut self, data: &[u8]) -> Result<()>;

    /// Decode a packet into samples.
    fn decode(&mut self, packet: &Packet<'_>) -> Result<Vec<Sample>>;

    /// Flush the decoder.
    fn flush(&mut self) -> Result<Vec<Sample>>;

    /// Reset the decoder state.
    fn reset(&mut self);
}

/// Common trait for audio encoders.
pub trait AudioEncoder: Send {
    /// Get codec information.
    fn codec_info(&self) -> CodecInfo;

    /// Get encoder name.
    fn name(&self) -> &str;

    /// Get sample rate.
    fn sample_rate(&self) -> u32;

    /// Get number of channels.
    fn channels(&self) -> u8;

    /// Get sample format.
    fn sample_format(&self) -> SampleFormat;

    /// Encode samples into packets.
    fn encode(&mut self, sample: &Sample) -> Result<Vec<Packet<'static>>>;

    /// Flush the encoder.
    fn flush(&mut self) -> Result<Vec<Packet<'static>>>;

    /// Reset the encoder state.
    fn reset(&mut self);

    /// Get the codec-specific configuration data.
    fn extra_data(&self) -> Option<&[u8]>;
}

/// Extension trait for video decoders with buffer reuse.
///
/// This trait provides `decode_into` and `flush_into` methods that write to
/// pre-allocated buffers, reducing allocation overhead in hot loops.
///
/// Default implementations are provided that delegate to the base trait methods,
/// but implementors can override them for better performance.
pub trait VideoDecoderExt: VideoDecoder {
    /// Decode a packet into a pre-allocated frame buffer.
    ///
    /// The output buffer is cleared before decoding. This method is more efficient
    /// than [`VideoDecoder::decode`] when decoding multiple packets in a loop.
    ///
    /// # Arguments
    ///
    /// * `packet` - The packet to decode
    /// * `out` - Output buffer for decoded frames (will be cleared)
    fn decode_into(&mut self, packet: &Packet<'_>, out: &mut Vec<Frame>) -> Result<()> {
        out.clear();
        let frames = self.decode(packet)?;
        out.extend(frames);
        Ok(())
    }

    /// Flush decoder into a pre-allocated frame buffer.
    ///
    /// The output buffer is cleared before flushing.
    ///
    /// # Arguments
    ///
    /// * `out` - Output buffer for flushed frames (will be cleared)
    fn flush_into(&mut self, out: &mut Vec<Frame>) -> Result<()> {
        out.clear();
        let frames = self.flush()?;
        out.extend(frames);
        Ok(())
    }
}

// Blanket implementation for all VideoDecoder types
impl<T: VideoDecoder + ?Sized> VideoDecoderExt for T {}

/// Extension trait for audio decoders with buffer reuse.
///
/// This trait provides `decode_into` and `flush_into` methods that write to
/// pre-allocated buffers, reducing allocation overhead in hot loops.
pub trait AudioDecoderExt: AudioDecoder {
    /// Decode a packet into a pre-allocated sample buffer.
    ///
    /// The output buffer is cleared before decoding.
    ///
    /// # Arguments
    ///
    /// * `packet` - The packet to decode
    /// * `out` - Output buffer for decoded samples (will be cleared)
    fn decode_into(&mut self, packet: &Packet<'_>, out: &mut Vec<Sample>) -> Result<()> {
        out.clear();
        let samples = self.decode(packet)?;
        out.extend(samples);
        Ok(())
    }

    /// Flush decoder into a pre-allocated sample buffer.
    ///
    /// The output buffer is cleared before flushing.
    ///
    /// # Arguments
    ///
    /// * `out` - Output buffer for flushed samples (will be cleared)
    fn flush_into(&mut self, out: &mut Vec<Sample>) -> Result<()> {
        out.clear();
        let samples = self.flush()?;
        out.extend(samples);
        Ok(())
    }
}

// Blanket implementation for all AudioDecoder types
impl<T: AudioDecoder + ?Sized> AudioDecoderExt for T {}

/// Extension trait for audio encoders with buffer reuse.
///
/// This trait provides `encode_into` and `flush_into` methods that write to
/// pre-allocated buffers, reducing allocation overhead in hot loops.
pub trait AudioEncoderExt: AudioEncoder {
    /// Encode samples into a pre-allocated packet buffer.
    ///
    /// The output buffer is cleared before encoding.
    ///
    /// # Arguments
    ///
    /// * `sample` - The audio sample to encode
    /// * `out` - Output buffer for encoded packets (will be cleared)
    fn encode_into(&mut self, sample: &Sample, out: &mut Vec<Packet<'static>>) -> Result<()> {
        out.clear();
        let packets = self.encode(sample)?;
        out.extend(packets);
        Ok(())
    }

    /// Flush encoder into a pre-allocated packet buffer.
    ///
    /// The output buffer is cleared before flushing.
    ///
    /// # Arguments
    ///
    /// * `out` - Output buffer for flushed packets (will be cleared)
    fn flush_into(&mut self, out: &mut Vec<Packet<'static>>) -> Result<()> {
        out.clear();
        let packets = self.flush()?;
        out.extend(packets);
        Ok(())
    }
}

// Blanket implementation for all AudioEncoder types
impl<T: AudioEncoder + ?Sized> AudioEncoderExt for T {}

/// Unified decoder enum for boxing different decoder types.
pub enum Decoder {
    Video(Box<dyn VideoDecoder>),
    Audio(Box<dyn AudioDecoder>),
}

/// Unified encoder enum for boxing different encoder types.
pub enum Encoder {
    Video(Box<dyn VideoEncoder>),
    Audio(Box<dyn AudioEncoder>),
}
