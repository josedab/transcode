//! Progress callback example.
//!
//! This example demonstrates how to use progress callbacks for custom
//! progress reporting during transcoding operations.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example progress_callback -- input.mp4 output.mp4
//! ```
//!
//! # Features Demonstrated
//!
//! - Custom progress callback function
//! - Real-time FPS and ETA calculation
//! - Terminal progress bar

use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use transcode::{Transcoder, TranscodeOptions, Result};

/// Progress tracker for calculating real-time statistics.
struct ProgressTracker {
    start_time: Instant,
    last_frame: AtomicU64,
    total_frames: AtomicU64,
}

impl ProgressTracker {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            last_frame: AtomicU64::new(0),
            total_frames: AtomicU64::new(0),
        }
    }

    fn update(&self, progress: f64, frame: u64) {
        self.last_frame.store(frame, Ordering::Relaxed);

        // Estimate total frames from progress
        if progress > 0.0 {
            let estimated_total = (frame as f64 / (progress / 100.0)) as u64;
            self.total_frames.store(estimated_total, Ordering::Relaxed);
        }
    }

    fn fps(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let frames = self.last_frame.load(Ordering::Relaxed);
        if elapsed > 0.0 {
            frames as f64 / elapsed
        } else {
            0.0
        }
    }

    fn eta_seconds(&self, progress: f64) -> Option<f64> {
        if progress > 0.0 && progress < 100.0 {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let remaining = 100.0 - progress;
            Some((elapsed / progress) * remaining)
        } else {
            None
        }
    }

    fn format_eta(&self, progress: f64) -> String {
        match self.eta_seconds(progress) {
            Some(secs) => {
                let total = secs as u64;
                if total >= 3600 {
                    format!("{}h{}m", total / 3600, (total % 3600) / 60)
                } else if total >= 60 {
                    format!("{}m{}s", total / 60, total % 60)
                } else {
                    format!("{}s", total)
                }
            }
            None => "calculating...".to_string(),
        }
    }

    fn print_progress(&self, progress: f64, frame: u64) {
        // Terminal progress bar
        let bar_width = 40;
        let filled = (progress / 100.0 * bar_width as f64) as usize;
        let empty = bar_width - filled;

        print!("\r[{}{}] {:.1}% | {:.1} fps | ETA: {} | Frame {}   ",
            "=".repeat(filled),
            " ".repeat(empty),
            progress,
            self.fps(),
            self.format_eta(progress),
            frame
        );
        io::stdout().flush().ok();
    }

    fn finish(&self) {
        println!();  // New line after progress bar
    }
}

fn main() -> Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let mut input_path = String::new();
    let mut output_path = String::new();

    let mut i = 1;
    while i < args.len() {
        if input_path.is_empty() {
            input_path = args[i].clone();
        } else if output_path.is_empty() {
            output_path = args[i].clone();
        }
        i += 1;
    }

    if input_path.is_empty() || output_path.is_empty() {
        eprintln!("Usage: {} <input> <output>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} input.mp4 output.mp4", args[0]);
        std::process::exit(1);
    }

    // Verify input exists
    if !Path::new(&input_path).exists() {
        eprintln!("Error: Input file not found: {}", input_path);
        std::process::exit(1);
    }

    println!("Progress Callback Example");
    println!("=========================");
    println!("Input:  {}", input_path);
    println!("Output: {}", output_path);
    println!();

    // Create progress tracker
    let tracker = Arc::new(ProgressTracker::new());
    let tracker_clone = Arc::clone(&tracker);

    // Configure transcoding
    let options = TranscodeOptions::new()
        .input(&input_path)
        .output(&output_path)
        .overwrite(true);

    // Create transcoder with progress callback
    let mut transcoder = Transcoder::new(options)?;

    transcoder = transcoder.on_progress(move |progress, frame| {
        tracker_clone.update(progress, frame);

        // Update every 10 frames to reduce overhead
        if frame % 10 == 0 {
            tracker_clone.print_progress(progress, frame);
        }
    });

    println!("Transcoding with progress tracking...");
    println!();

    // Run transcoding
    let start = Instant::now();
    transcoder.run()?;
    let elapsed = start.elapsed();

    // Finish progress display
    tracker.finish();

    // Print final statistics
    let stats = transcoder.stats();

    println!();
    println!("Transcoding complete!");
    println!("  Time elapsed:      {:.2}s", elapsed.as_secs_f64());
    println!("  Frames encoded:    {}", stats.frames_encoded);
    println!("  Average FPS:       {:.1}", stats.frames_encoded as f64 / elapsed.as_secs_f64());
    println!("  Compression ratio: {:.2}x", stats.compression_ratio());

    Ok(())
}
