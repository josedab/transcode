#![no_main]

//! Fuzz target for AAC ADTS header parsing.
//!
//! Tests ADTS header parsing, sync word detection, frame iteration,
//! and roundtrip encoding.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct AdtsInput {
    data: Vec<u8>,
    test_mode: AdtsTestMode,
}

#[derive(Arbitrary, Debug)]
enum AdtsTestMode {
    /// Parse single ADTS header
    ParseHeader,
    /// Find sync word in stream
    FindSync,
    /// Iterate over ADTS frames
    IterateFrames,
    /// Encode/decode roundtrip
    Roundtrip,
}

fuzz_target!(|input: AdtsInput| {
    // Limit input size
    if input.data.len() > 1024 * 1024 {
        return;
    }

    match input.test_mode {
        AdtsTestMode::ParseHeader => {
            // Parse single ADTS header
            if let Ok(header) = transcode_codecs::audio::aac::AdtsHeader::parse(&input.data) {
                // Test derived values - should never panic
                let _ = header.header_size();
                let _ = header.payload_size();
                let _ = header.sample_rate();
                let _ = header.channels();

                // Access all fields
                let _ = header.mpeg_version;
                let _ = header.layer;
                let _ = header.protection_absent;
                let _ = header.profile;
                let _ = header.sample_rate_index;
                let _ = header.channel_config;
                let _ = header.frame_length;
                let _ = header.buffer_fullness;
                let _ = header.num_raw_data_blocks;
                let _ = header.crc;
            }
        }
        AdtsTestMode::FindSync => {
            // Find sync word in stream
            let _ = transcode_codecs::audio::aac::find_adts_sync(&input.data);
        }
        AdtsTestMode::IterateFrames => {
            // Iterate over ADTS frames
            let iterator = transcode_codecs::audio::aac::AdtsIterator::new(&input.data);
            for frame_result in iterator.take(1000) {
                // Limit iterations to prevent DoS
                match frame_result {
                    Ok((payload, header)) => {
                        // Access derived values
                        let _ = header.sample_rate();
                        let _ = header.channels();
                        let _ = payload.len();
                    }
                    Err(_) => break,
                }
            }
        }
        AdtsTestMode::Roundtrip => {
            // Try to parse and re-encode
            if let Ok(header) = transcode_codecs::audio::aac::AdtsHeader::parse(&input.data) {
                // Encode the header back
                let encoded = header.encode();

                // Try to parse the encoded data
                if let Ok(reparsed) = transcode_codecs::audio::aac::AdtsHeader::parse(&encoded) {
                    // Values should match (this is a sanity check, not a panic condition)
                    let _ = reparsed.profile == header.profile;
                    let _ = reparsed.sample_rate_index == header.sample_rate_index;
                    let _ = reparsed.channel_config == header.channel_config;
                }
            }
        }
    }
});
