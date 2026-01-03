#![no_main]

use libfuzzer_sys::fuzz_target;
use transcode_codecs::audio::aac::AacDecoder;

fuzz_target!(|data: &[u8]| {
    // Create a decoder
    let mut decoder = AacDecoder::new();

    // Try to decode a frame - we don't care about errors, just panics
    let _ = decoder.decode_frame(data);
});
