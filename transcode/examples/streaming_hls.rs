//! HLS streaming output example.
//!
//! This example demonstrates how to generate HLS (HTTP Live Streaming)
//! output with multiple quality levels for adaptive bitrate streaming.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example streaming_hls -- input.mp4 output_dir/
//! ```
//!
//! # Output Structure
//!
//! ```text
//! output_dir/
//! ├── master.m3u8           # Master playlist
//! ├── 1080p/
//! │   ├── stream.m3u8       # Media playlist
//! │   ├── segment_0.ts      # Media segments
//! │   └── ...
//! ├── 720p/
//! │   └── ...
//! └── 480p/
//!     └── ...
//! ```

use std::env;
use std::path::Path;
use transcode_streaming::{HlsConfig, HlsWriter, Quality};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input> <output_dir>", args[0]);
        eprintln!("Example: {} input.mp4 ./hls_output/", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_dir = &args[2];

    println!("HLS Streaming Output Example");
    println!("============================");
    println!("Input:      {}", input_path);
    println!("Output dir: {}", output_dir);
    println!();

    // Verify input exists
    if !Path::new(input_path).exists() {
        eprintln!("Error: Input file not found: {}", input_path);
        std::process::exit(1);
    }

    // Configure HLS output with multiple quality levels
    let config = HlsConfig::new(output_dir)
        // Segment duration (seconds) - 6s is recommended for HLS
        .with_segment_duration(6.0)
        // Add quality variants for adaptive streaming
        .with_quality(Quality::new(1920, 1080, 5_000_000))  // 1080p @ 5Mbps
        .with_quality(Quality::new(1280, 720, 2_500_000))   // 720p @ 2.5Mbps
        .with_quality(Quality::new(854, 480, 1_000_000))    // 480p @ 1Mbps
        .with_quality(Quality::new(640, 360, 500_000));     // 360p @ 500kbps

    println!("HLS Configuration:");
    println!("  Segment duration: {}s", 6.0);
    println!("  Quality levels:   {}", 4);
    println!();

    // Create HLS writer
    let writer = HlsWriter::new(config)?;

    println!("Quality Variants:");
    println!("  1080p - 5.0 Mbps (primary)");
    println!("  720p  - 2.5 Mbps");
    println!("  480p  - 1.0 Mbps");
    println!("  360p  - 0.5 Mbps (fallback)");
    println!();

    // In a real implementation, you would:
    // 1. Decode the input video
    // 2. Encode to each quality level
    // 3. Write segments using the HLS writer
    //
    // For this example, we demonstrate the configuration API.

    println!("HLS writer initialized successfully!");
    println!();
    println!("In production, you would now:");
    println!("  1. Decode input frames with Mp4Demuxer");
    println!("  2. Scale to each quality level");
    println!("  3. Encode with H.264");
    println!("  4. Write segments via writer.write_segment()");
    println!("  5. Call writer.finalize() to write playlists");
    println!();

    // Show what files would be generated
    println!("Expected output files:");
    println!("  {}/master.m3u8", output_dir);
    println!("  {}/1080p/stream.m3u8", output_dir);
    println!("  {}/720p/stream.m3u8", output_dir);
    println!("  {}/480p/stream.m3u8", output_dir);
    println!("  {}/360p/stream.m3u8", output_dir);

    // Clean up (in real usage, call finalize)
    drop(writer);

    Ok(())
}
