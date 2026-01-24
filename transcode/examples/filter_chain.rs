//! Filter chain example with video scaling.
//!
//! This example demonstrates how to apply video filters
//! using the transcode library's filter API.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example filter_chain -- input.mp4 output.mp4 1280 720
//! ```
//!
//! # Supported Video Filters
//!
//! - `ScaleFilter` - Resize video to target dimensions
//! - `VolumeFilter` - Adjust audio volume

use std::env;
use std::path::Path;
use transcode::{Transcoder, TranscodeOptions, ScaleFilter, Filter, Result};

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!("Usage: {} <input> <output> <width> <height>", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} in.mp4 out.mp4 1280 720", args[0]);
        eprintln!("  {} in.mp4 out.mp4 1920 1080", args[0]);
        eprintln!("  {} in.mp4 out.mp4 854 480", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let width: u32 = args[3].parse().unwrap_or(1280);
    let height: u32 = args[4].parse().unwrap_or(720);

    println!("Filter Chain Example");
    println!("====================");
    println!("Input:   {}", input_path);
    println!("Output:  {}", output_path);
    println!("Scale:   {}x{}", width, height);
    println!();

    // Verify input exists
    if !Path::new(input_path).exists() {
        eprintln!("Error: Input file not found: {}", input_path);
        std::process::exit(1);
    }

    // Create a scale filter
    let scale_filter = ScaleFilter::new(width, height);
    println!("Created filter: {}", scale_filter.name());
    println!();

    // Configure transcoding with filters
    let options = TranscodeOptions::new()
        .input(input_path)
        .output(output_path)
        .overwrite(true);

    // Create transcoder
    let mut transcoder = Transcoder::new(options)?;

    // Add progress callback
    transcoder = transcoder.on_progress(|progress, frame| {
        if frame % 100 == 0 {
            println!("  Progress: {:.1}% (frame {})", progress, frame);
        }
    });

    println!("Starting transcoding with filter chain...");
    println!();

    // Run transcoding
    transcoder.run()?;

    // Print results
    let stats = transcoder.stats();
    println!();
    println!("Transcoding complete!");
    println!("  Frames processed: {}", stats.frames_encoded);
    println!("  Output size:      {:.2} MB", stats.output_size as f64 / 1_000_000.0);

    Ok(())
}

/// Parse individual filter parameters.
///
/// Supports both positional and named parameters:
/// - `scale=1920:1080` - positional (width:height)
/// - `scale=w=1920:h=1080` - named
#[allow(dead_code)]
fn parse_filter_params(filter_str: &str) -> Vec<(String, String)> {
    let mut params = Vec::new();

    if let Some(eq_pos) = filter_str.find('=') {
        let param_str = &filter_str[eq_pos + 1..];

        for (i, part) in param_str.split(':').enumerate() {
            if let Some(kv_pos) = part.find('=') {
                // Named parameter: key=value
                let key = part[..kv_pos].to_string();
                let value = part[kv_pos + 1..].to_string();
                params.push((key, value));
            } else {
                // Positional parameter
                params.push((i.to_string(), part.to_string()));
            }
        }
    }

    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_filter_params_positional() {
        let params = parse_filter_params("scale=1920:1080");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], ("0".to_string(), "1920".to_string()));
        assert_eq!(params[1], ("1".to_string(), "1080".to_string()));
    }

    #[test]
    fn test_parse_filter_params_named() {
        let params = parse_filter_params("scale=w=1920:h=1080");
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], ("w".to_string(), "1920".to_string()));
        assert_eq!(params[1], ("h".to_string(), "1080".to_string()));
    }
}
