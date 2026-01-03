//! Audio codec implementations.

pub mod aac;
pub mod mp3;

pub use aac::{AacDecoder, AacEncoder};
pub use mp3::{Mp3Decoder, Mp3Encoder, Mp3EncoderConfig};
