//! Basic transcoding example.
//!
//! This example demonstrates how to transcode a video file using the
//! high-level Transcoder API.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example basic_transcode -- input.mp4 output.mp4
//! ```

use std::env;
use std::time::Instant;
use transcode::{Transcoder, TranscodeOptions, Result};

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input> <output>", args[0]);
        eprintln!("Example: {} input.mp4 output.mp4", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    println!("Transcoding: {} -> {}", input_path, output_path);

    // Configure transcoding options
    let options = TranscodeOptions::new()
        .input(input_path)
        .output(output_path)
        .video_bitrate(5_000_000)  // 5 Mbps
        .audio_bitrate(128_000)    // 128 kbps
        .overwrite(true);          // Overwrite if exists

    // Create the transcoder
    let mut transcoder = Transcoder::new(options)?;

    // Run transcoding and measure time
    let start = Instant::now();
    transcoder.run()?;
    let elapsed = start.elapsed();

    // Print statistics
    let stats = transcoder.stats();
    println!("\nTranscoding complete!");
    println!("Time elapsed:      {:.2}s", elapsed.as_secs_f64());
    println!("Frames decoded:    {}", stats.frames_decoded);
    println!("Frames encoded:    {}", stats.frames_encoded);

    Ok(())
}
