//! Common codec traits.

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
