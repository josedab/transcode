#![no_main]

//! Fuzz target for MP3 frame header parsing.
//!
//! Tests MP3 sync word detection, frame header parsing, and frame iteration.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

#[derive(Arbitrary, Debug)]
struct Mp3Input {
    data: Vec<u8>,
    test_mode: Mp3TestMode,
}

#[derive(Arbitrary, Debug)]
enum Mp3TestMode {
    /// Parse single frame header
    ParseHeader,
    /// Find sync word in stream
    FindSync,
    /// Iterate over MP3 frames
    IterateFrames,
}

fuzz_target!(|input: Mp3Input| {
    // Limit input size
    if input.data.len() > 10 * 1024 * 1024 {
        return;
    }

    match input.test_mode {
        Mp3TestMode::ParseHeader => {
            // Parse single MP3 frame header
            if let Ok(header) = transcode_codecs::audio::mp3::Mp3FrameHeader::parse(&input.data) {
                // Test derived values - should never panic
                let _ = header.bitrate();
                let _ = header.sample_rate();
                let _ = header.samples_per_frame();
                let _ = header.frame_size();
                let _ = header.side_info_size();
                let _ = header.header_size();

                // Access all fields
                let _ = header.version;
                let _ = header.layer;
                let _ = header.crc_protected;
                let _ = header.bitrate_index;
                let _ = header.sample_rate_index;
                let _ = header.padding;
                let _ = header.channel_mode;
                let _ = header.channel_mode.channels();
                let _ = header.mode_extension;
                let _ = header.copyright;
                let _ = header.original;
                let _ = header.emphasis;
            }
        }
        Mp3TestMode::FindSync => {
            // Find sync word in stream
            let _ = transcode_codecs::audio::mp3::find_mp3_sync(&input.data);
        }
        Mp3TestMode::IterateFrames => {
            // Iterate over MP3 frames
            let iterator = transcode_codecs::audio::mp3::Mp3FrameIterator::new(&input.data);
            for frame_result in iterator.take(1000) {
                // Limit iterations to prevent DoS
                match frame_result {
                    Ok((frame_data, header)) => {
                        // Access derived values
                        let _ = header.bitrate();
                        let _ = header.sample_rate();
                        let _ = frame_data.len();
                    }
                    Err(_) => break,
                }
            }
        }
    }
});
