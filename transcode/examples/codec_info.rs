//! Codec information example.
//!
//! This example shows how to probe a media file and display
//! information about its streams.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example codec_info -- input.mp4
//! ```

use std::env;
use std::fs::File;
use std::io::BufReader;
use transcode_containers::mp4::Mp4Demuxer;
use transcode_containers::traits::{Demuxer, TrackType};
use transcode_core::Result;

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input>", args[0]);
        eprintln!("Example: {} video.mp4", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    println!("Analyzing: {}", input_path);
    println!();

    // Open the file
    let file = File::open(input_path)
        .map_err(|e| transcode_core::Error::Io(e))?;
    let reader = BufReader::new(file);

    // Create and open demuxer
    let mut demuxer = Mp4Demuxer::new();
    demuxer.open(reader)?;

    // Display container info
    println!("Container Information");
    println!("══════════════════════════════════════════════════════════");
    println!("Format:   {}", demuxer.format_name());
    if let Some(duration_us) = demuxer.duration() {
        println!("Duration: {:.2}s", duration_us as f64 / 1_000_000.0);
    } else {
        println!("Duration: unknown");
    }
    println!();

    // Display stream information
    let num_streams = demuxer.num_streams();
    println!("Streams: {}", num_streams);
    println!("────────────────────────────────────────────────────────────");

    for i in 0..num_streams {
        if let Some(stream) = demuxer.stream_info(i) {
            println!();
            println!("Stream #{}", i + 1);
            println!("  Type:       {:?}", stream.track_type);
            println!("  Codec:      {:?}", stream.codec_id);
            println!("  Time base:  {}/{}", stream.time_base.num, stream.time_base.den);

            if let Some(duration) = stream.duration {
                let duration_secs = duration as f64 * stream.time_base.num as f64
                    / stream.time_base.den as f64;
                println!("  Duration:   {:.2}s", duration_secs);
            }

            // Video-specific info
            if stream.track_type == TrackType::Video {
                if let Some(ref video) = stream.video {
                    println!("  Resolution: {}x{}", video.width, video.height);
                    if let Some(ref fps) = video.frame_rate {
                        println!("  Frame rate: {:.2} fps", fps.num as f64 / fps.den as f64);
                    }
                    println!("  Bit depth:  {}", video.bit_depth);
                }
            }

            // Audio-specific info
            if stream.track_type == TrackType::Audio {
                if let Some(ref audio) = stream.audio {
                    println!("  Sample rate: {} Hz", audio.sample_rate);
                    println!("  Channels:    {}", audio.channels);
                    println!("  Bit depth:   {}", audio.bits_per_sample);
                }
            }
        }
    }

    println!();
    println!("══════════════════════════════════════════════════════════");

    Ok(())
}
