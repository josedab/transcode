#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use transcode_containers::mp4::Mp4Demuxer;
use transcode_containers::Demuxer;

fuzz_target!(|data: &[u8]| {
    // Create a demuxer
    let mut demuxer = Mp4Demuxer::new();

    // Create a cursor from the fuzzed data
    let cursor = Cursor::new(data.to_vec());

    // Try to open the demuxer - we don't care about errors, just panics
    if demuxer.open(cursor).is_ok() {
        // Try to get duration
        let _ = demuxer.duration();

        // Try to get number of streams
        let _ = demuxer.num_streams();

        // Try to read a packet
        let _ = demuxer.read_packet();
    }
});
