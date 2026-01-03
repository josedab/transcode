#![no_main]

//! Fuzz target for MP4 container parsing.
//!
//! Tests MP4 demuxer with arbitrary input data.
//! Note: Individual atom parsers are private, so we test via the demuxer.

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use std::io::Cursor;
use transcode_containers::mp4::Mp4Demuxer;
use transcode_containers::Demuxer;

#[derive(Arbitrary, Debug)]
struct Mp4Input {
    data: Vec<u8>,
    test_mode: Mp4TestMode,
}

#[derive(Arbitrary, Debug)]
enum Mp4TestMode {
    /// Create demuxer and check basic properties
    BasicParse,
    /// Try to get track information
    TrackInfo,
    /// Try to read packets
    ReadPackets,
}

fuzz_target!(|input: Mp4Input| {
    // Limit input size
    if input.data.len() > 10 * 1024 * 1024 {
        return;
    }

    let mut demuxer = Mp4Demuxer::new();
    let cursor = Cursor::new(input.data);

    // Try to open - if it fails, that's ok
    if demuxer.open(cursor).is_err() {
        return;
    }

    match input.test_mode {
        Mp4TestMode::BasicParse => {
            // Access basic properties - should never panic
            let _ = demuxer.duration();
            let _ = demuxer.num_streams();
            let _ = demuxer.format_name();
            let _ = demuxer.chapters();
            let _ = demuxer.chapter_track_refs();
        }
        Mp4TestMode::TrackInfo => {
            // Try to access stream info for each stream
            let num_streams = demuxer.num_streams();
            for i in 0..num_streams {
                let _ = demuxer.stream_info(i);
            }

            // Access duration
            let _ = demuxer.duration();
        }
        Mp4TestMode::ReadPackets => {
            // Try to read up to 100 packets
            for _ in 0..100 {
                match demuxer.read_packet() {
                    Ok(Some(_packet)) => {
                        // Successfully read a packet
                    }
                    Ok(None) => {
                        // End of file
                        break;
                    }
                    Err(_) => {
                        // Error, stop reading
                        break;
                    }
                }
            }
        }
    }
});
