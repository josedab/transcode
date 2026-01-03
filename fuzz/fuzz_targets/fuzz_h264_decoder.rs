#![no_main]

use libfuzzer_sys::fuzz_target;
use transcode_codecs::video::h264::H264Decoder;
use transcode_codecs::traits::VideoDecoder;
use transcode_core::packet::Packet;

fuzz_target!(|data: &[u8]| {
    // Create a decoder with default configuration
    let mut decoder = H264Decoder::new_default();

    // Create a packet from the fuzzed data
    let packet = Packet::from_slice(data).with_stream_index(0);

    // Try to decode - we don't care about errors, just panics
    let _ = decoder.decode(&packet);
});
