#![no_main]

//! Fuzz target for VP9 bitstream parsing.
//!
//! Tests VP9 frame header and superframe parsing with arbitrary data.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use transcode_vp9::{Vp9Decoder, Vp9DecoderConfig};

#[derive(Arbitrary, Debug)]
struct Vp9Input {
    data: Vec<u8>,
    test_mode: Vp9TestMode,
}

#[derive(Arbitrary, Debug)]
enum Vp9TestMode {
    /// Single frame decode attempt
    SingleFrame,
    /// Superframe decode attempt (multiple frames in one packet)
    Superframe,
    /// Multiple decode calls
    MultiDecode,
}

fuzz_target!(|input: Vp9Input| {
    // Limit input size to prevent OOM
    if input.data.len() > 5 * 1024 * 1024 {
        return;
    }

    // Skip empty input
    if input.data.is_empty() {
        return;
    }

    let config = Vp9DecoderConfig::default();
    let mut decoder = Vp9Decoder::new(config);

    match input.test_mode {
        Vp9TestMode::SingleFrame => {
            // Try to decode as a single frame
            let _ = decoder.decode_frame(&input.data);
        }
        Vp9TestMode::Superframe => {
            // Try to decode as a superframe (VP9 can pack multiple frames)
            let _ = decoder.decode_frame(&input.data);
        }
        Vp9TestMode::MultiDecode => {
            // Split input into chunks and decode each
            let chunk_size = input.data.len().max(1) / 4;
            if chunk_size > 0 {
                for chunk in input.data.chunks(chunk_size) {
                    let _ = decoder.decode_frame(chunk);
                }
            }
        }
    }
});
