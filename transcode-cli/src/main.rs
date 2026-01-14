//! Transcode CLI - Command-line interface for media transcoding.

use clap::Parser;
use console::style;
use glob::glob;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use serde::Serialize;
use std::fmt::Write as FmtWrite;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use transcode_compat::FilterChain;

/// Output mode for the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    /// Normal output with progress bar.
    Normal,
    /// JSON output for programmatic parsing.
    Json,
    /// Quiet mode with minimal output.
    Quiet,
    /// Verbose mode with detailed stats.
    Verbose,
}

/// Real-time transcoding statistics for progress reporting.
#[derive(Debug, Clone, Serialize)]
struct ProgressStats {
    /// Current frame being processed.
    current_frame: u64,
    /// Total frames to process (if known).
    total_frames: Option<u64>,
    /// Percentage complete (0.0 - 100.0).
    percentage: f64,
    /// Encoding speed in frames per second.
    fps: f64,
    /// Estimated time remaining in seconds.
    eta_seconds: Option<f64>,
    /// Current bitrate in kbps.
    bitrate_kbps: f64,
    /// Current output file size in bytes.
    output_size_bytes: u64,
    /// Elapsed time in seconds.
    elapsed_seconds: f64,
    /// Duration processed in seconds.
    duration_processed_seconds: f64,
    /// Total duration in seconds (if known).
    total_duration_seconds: Option<f64>,
    /// Real-time speed multiplier.
    speed: f64,
}

/// JSON progress output structure.
#[derive(Debug, Clone, Serialize)]
struct JsonProgressOutput {
    /// Type of message.
    #[serde(rename = "type")]
    msg_type: String,
    /// Progress statistics.
    #[serde(flatten)]
    stats: ProgressStats,
}

/// JSON completion output structure.
#[derive(Debug, Clone, Serialize)]
struct JsonCompleteOutput {
    /// Type of message.
    #[serde(rename = "type")]
    msg_type: String,
    /// Whether transcoding was successful.
    success: bool,
    /// Final statistics.
    stats: FinalStats,
}

/// Final transcoding statistics.
#[derive(Debug, Clone, Serialize)]
struct FinalStats {
    /// Total packets processed.
    packets_processed: u64,
    /// Total frames decoded.
    frames_decoded: u64,
    /// Total frames encoded.
    frames_encoded: u64,
    /// Input file size in bytes.
    input_size_bytes: u64,
    /// Output file size in bytes.
    output_size_bytes: u64,
    /// Compression ratio.
    compression_ratio: f64,
    /// Total elapsed time in seconds.
    elapsed_seconds: f64,
    /// Average encoding speed in fps.
    average_fps: f64,
    /// Average video bitrate in kbps.
    avg_video_bitrate_kbps: f64,
    /// Average audio bitrate in kbps.
    avg_audio_bitrate_kbps: f64,
}

/// Command-line arguments for the transcode tool.
#[derive(Parser, Debug)]
#[command(name = "transcode")]
#[command(version)]
#[command(about = "A memory-safe, high-performance media transcoding tool")]
#[command(long_about = "Transcode is a universal media transcoding tool built in Rust.\n\n\
    It provides memory-safe, high-performance transcoding with SIMD \n\
    optimizations (AVX2, NEON) and automatic runtime detection.\n\n\
    EXAMPLES:\n    \
    transcode -i input.mp4 -o output.mp4\n    \
    transcode -i input.mp4 -o output.mp4 --video-bitrate 5000\n    \
    transcode -i input.mp4 -o output.mp4 --video-codec h264 --audio-codec aac\n    \
    transcode -i input.mp4 -o output.mp4 --json\n    \
    transcode -i input.mp4 -o output.mp4 --verbose")]
struct Args {
    /// Input file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output file path
    #[arg(short, long)]
    output: PathBuf,

    /// Video codec (h264, h265)
    #[arg(long, default_value = "h264")]
    video_codec: String,

    /// Audio codec (aac, mp3)
    #[arg(long, default_value = "aac")]
    audio_codec: String,

    /// Video bitrate in kbps (e.g., 5000 for 5 Mbps)
    #[arg(long)]
    video_bitrate: Option<u32>,

    /// Audio bitrate in kbps (e.g., 128 for 128 kbps)
    #[arg(long)]
    audio_bitrate: Option<u32>,

    /// Number of threads to use (default: auto-detect)
    #[arg(short = 't', long)]
    threads: Option<usize>,

    /// Overwrite output file if it exists
    #[arg(short = 'y', long)]
    overwrite: bool,

    /// Disable progress bar
    #[arg(long)]
    no_progress: bool,

    /// Verbose output (show detailed stats during transcoding)
    #[arg(short, long, conflicts_with = "quiet", conflicts_with = "json")]
    verbose: bool,

    /// Quiet mode (minimal output, only print output path on success)
    #[arg(short, long, conflicts_with = "verbose", conflicts_with = "json")]
    quiet: bool,

    /// JSON output mode for programmatic parsing
    #[arg(long, conflicts_with = "verbose", conflicts_with = "quiet")]
    json: bool,

    /// Progress update interval in milliseconds (for JSON mode)
    #[arg(long, default_value = "500")]
    progress_interval: u64,

    /// Video filter chain (FFmpeg-style syntax, e.g., "scale=1920:1080,fps=30")
    #[arg(short = 'F', long = "vf")]
    video_filter: Option<String>,

    /// Audio filter chain (FFmpeg-style syntax, e.g., "volume=0.5,aresample=48000")
    #[arg(long = "af")]
    audio_filter: Option<String>,

    /// Batch processing: process all files matching glob pattern (e.g., "*.mp4")
    #[arg(long)]
    batch: Option<String>,

    /// Batch processing: output directory for batch mode
    #[arg(long)]
    batch_output_dir: Option<PathBuf>,

    /// Number of parallel jobs for batch processing
    #[arg(long, default_value = "1")]
    jobs: usize,
}

impl Args {
    /// Determine the output mode based on flags.
    fn output_mode(&self) -> OutputMode {
        if self.json {
            OutputMode::Json
        } else if self.quiet {
            OutputMode::Quiet
        } else if self.verbose {
            OutputMode::Verbose
        } else {
            OutputMode::Normal
        }
    }
}

/// Shared state for progress tracking between threads.
#[allow(dead_code)]
struct SharedProgress {
    /// Current frame count.
    current_frame: AtomicU64,
    /// Total frames (if known).
    total_frames: AtomicU64,
    /// Whether total frames is known.
    total_frames_known: AtomicBool,
    /// Duration processed in microseconds (reserved for future use).
    duration_processed_us: AtomicU64,
    /// Total duration in microseconds (reserved for future use).
    total_duration_us: AtomicU64,
    /// Whether total duration is known (reserved for future use).
    total_duration_known: AtomicBool,
    /// Current output size in bytes.
    output_size: AtomicU64,
    /// Whether transcoding is complete (reserved for future use).
    is_complete: AtomicBool,
}

impl Default for SharedProgress {
    fn default() -> Self {
        Self {
            current_frame: AtomicU64::new(0),
            total_frames: AtomicU64::new(0),
            total_frames_known: AtomicBool::new(false),
            duration_processed_us: AtomicU64::new(0),
            total_duration_us: AtomicU64::new(0),
            total_duration_known: AtomicBool::new(false),
            output_size: AtomicU64::new(0),
            is_complete: AtomicBool::new(false),
        }
    }
}

impl SharedProgress {
    /// Update progress from callback.
    fn update(&self, progress: f64, frames: u64) {
        self.current_frame.store(frames, Ordering::Relaxed);

        // Estimate total frames if we have progress
        if progress > 0.0 {
            let estimated_total = (frames as f64 / (progress / 100.0)) as u64;
            self.total_frames.store(estimated_total, Ordering::Relaxed);
            self.total_frames_known.store(true, Ordering::Relaxed);
        }
    }

    /// Get current progress percentage (reserved for future use).
    #[allow(dead_code)]
    fn progress(&self) -> f64 {
        let current = self.current_frame.load(Ordering::Relaxed);
        let total = self.total_frames.load(Ordering::Relaxed);
        if self.total_frames_known.load(Ordering::Relaxed) && total > 0 {
            (current as f64 / total as f64 * 100.0).min(100.0)
        } else {
            0.0
        }
    }
}

/// Progress reporter that handles different output modes.
struct ProgressReporter {
    /// Output mode.
    mode: OutputMode,
    /// Progress bar (for normal/verbose modes).
    progress_bar: Option<ProgressBar>,
    /// Shared progress state.
    shared_progress: Arc<SharedProgress>,
    /// Start time.
    start_time: Instant,
    /// Last update time (for rate limiting JSON output).
    last_update: Instant,
    /// Progress update interval.
    update_interval: Duration,
}

impl ProgressReporter {
    /// Create a new progress reporter.
    fn new(mode: OutputMode, shared_progress: Arc<SharedProgress>, update_interval_ms: u64) -> Self {
        let progress_bar = match mode {
            OutputMode::Normal | OutputMode::Verbose => {
                let pb = ProgressBar::new(100);
                pb.set_style(create_progress_style(mode == OutputMode::Verbose));
                pb.enable_steady_tick(Duration::from_millis(100));
                Some(pb)
            }
            _ => None,
        };

        let now = Instant::now();
        Self {
            mode,
            progress_bar,
            shared_progress,
            start_time: now,
            last_update: now,
            update_interval: Duration::from_millis(update_interval_ms),
        }
    }

    /// Update progress display.
    fn update(&mut self, stats: &transcode::TranscodeStats) {
        let elapsed = self.start_time.elapsed();
        let progress = stats.progress();
        let current_frame = stats.frames_encoded;

        // Calculate FPS
        let fps = if elapsed.as_secs_f64() > 0.0 {
            current_frame as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        // Calculate ETA
        let eta_seconds = if progress > 0.0 && progress < 100.0 {
            let remaining_progress = 100.0 - progress;
            let seconds_per_percent = elapsed.as_secs_f64() / progress;
            Some(seconds_per_percent * remaining_progress)
        } else {
            None
        };

        // Calculate bitrate (estimated from output size)
        let output_size = self.shared_progress.output_size.load(Ordering::Relaxed);
        let duration_secs = stats.duration_processed_us as f64 / 1_000_000.0;
        let bitrate_kbps = if duration_secs > 0.0 {
            (output_size as f64 * 8.0) / (duration_secs * 1000.0)
        } else {
            0.0
        };

        // Calculate speed multiplier
        let speed = if elapsed.as_secs_f64() > 0.0 && duration_secs > 0.0 {
            duration_secs / elapsed.as_secs_f64()
        } else {
            0.0
        };

        let total_frames = if self.shared_progress.total_frames_known.load(Ordering::Relaxed) {
            Some(self.shared_progress.total_frames.load(Ordering::Relaxed))
        } else {
            None
        };

        let total_duration = stats.total_duration_us.map(|us| us as f64 / 1_000_000.0);

        let progress_stats = ProgressStats {
            current_frame,
            total_frames,
            percentage: progress,
            fps,
            eta_seconds,
            bitrate_kbps,
            output_size_bytes: output_size,
            elapsed_seconds: elapsed.as_secs_f64(),
            duration_processed_seconds: duration_secs,
            total_duration_seconds: total_duration,
            speed,
        };

        match self.mode {
            OutputMode::Normal | OutputMode::Verbose => {
                if let Some(pb) = &self.progress_bar {
                    pb.set_position(progress as u64);
                    pb.set_message(format_progress_message(&progress_stats, self.mode == OutputMode::Verbose));
                }
            }
            OutputMode::Json => {
                // Rate limit JSON output
                let now = Instant::now();
                if now.duration_since(self.last_update) >= self.update_interval {
                    self.last_update = now;
                    let output = JsonProgressOutput {
                        msg_type: "progress".to_string(),
                        stats: progress_stats,
                    };
                    if let Ok(json) = serde_json::to_string(&output) {
                        println!("{}", json);
                    }
                }
            }
            OutputMode::Quiet => {
                // No progress output in quiet mode
            }
        }
    }

    /// Finish progress reporting.
    fn finish(&self) {
        if let Some(pb) = &self.progress_bar {
            pb.finish_and_clear();
        }
    }
}

/// Create the progress bar style.
fn create_progress_style(verbose: bool) -> ProgressStyle {
    if verbose {
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}% ({msg})"
        )
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn FmtWrite| {
            write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
        })
        .progress_chars("#>-")
    } else {
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}% | ETA: {eta} | {msg}"
        )
        .unwrap()
        .with_key("eta", |state: &ProgressState, w: &mut dyn FmtWrite| {
            let eta = state.eta();
            if eta.as_secs() > 3600 {
                write!(w, "{}h{}m", eta.as_secs() / 3600, (eta.as_secs() % 3600) / 60).unwrap()
            } else if eta.as_secs() > 60 {
                write!(w, "{}m{}s", eta.as_secs() / 60, eta.as_secs() % 60).unwrap()
            } else {
                write!(w, "{}s", eta.as_secs()).unwrap()
            }
        })
        .progress_chars("#>-")
    }
}

/// Format the progress message for the progress bar.
fn format_progress_message(stats: &ProgressStats, verbose: bool) -> String {
    if verbose {
        format!(
            "Frame: {}/{} | {:.1} fps | {:.1}x speed | {:.0} kbps | {} | ETA: {}",
            stats.current_frame,
            stats.total_frames.map(|t| t.to_string()).unwrap_or_else(|| "?".to_string()),
            stats.fps,
            stats.speed,
            stats.bitrate_kbps,
            format_size(stats.output_size_bytes),
            stats.eta_seconds.map(format_duration).unwrap_or_else(|| "calculating...".to_string())
        )
    } else {
        format!(
            "{:.1} fps | {:.0} kbps | {}",
            stats.fps,
            stats.bitrate_kbps,
            format_size(stats.output_size_bytes)
        )
    }
}

/// Format bytes as human-readable size.
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration in seconds as human-readable string.
fn format_duration(seconds: f64) -> String {
    let total_seconds = seconds as u64;
    if total_seconds >= 3600 {
        format!("{}h{}m{}s", total_seconds / 3600, (total_seconds % 3600) / 60, total_seconds % 60)
    } else if total_seconds >= 60 {
        format!("{}m{}s", total_seconds / 60, total_seconds % 60)
    } else {
        format!("{:.0}s", seconds)
    }
}

fn main() -> anyhow::Result<()> {
    // Parse command-line arguments
    let args = Args::parse();
    let output_mode = args.output_mode();

    // Initialize logging (not in JSON or quiet mode)
    if output_mode != OutputMode::Json && output_mode != OutputMode::Quiet {
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(if args.verbose {
                tracing::Level::DEBUG
            } else {
                tracing::Level::INFO
            })
            .with_target(false)
            .finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    }

    // Print header (only in normal/verbose mode)
    if output_mode == OutputMode::Normal || output_mode == OutputMode::Verbose {
        print_header();
    }

    // Handle batch mode
    if let Some(ref pattern) = args.batch {
        let output_dir = args.batch_output_dir.clone()
            .unwrap_or_else(|| PathBuf::from("./output"));

        if output_mode == OutputMode::Normal || output_mode == OutputMode::Verbose {
            println!();
            println!("{}", style("Batch Processing Mode").cyan().bold());
            println!("  Pattern:      {}", style(pattern).white());
            println!("  Output dir:   {}", style(output_dir.display()).white());
            println!("  Parallel:     {} job(s)", style(args.jobs).white());
            println!();
        }

        match process_batch(&args, pattern, &output_dir) {
            Ok(results) => {
                let success_count = results.iter().filter(|r| r.success).count();
                let fail_count = results.len() - success_count;
                let total_time: Duration = results.iter().map(|r| r.elapsed).sum();

                if output_mode == OutputMode::Json {
                    let output = serde_json::json!({
                        "type": "batch_complete",
                        "total": results.len(),
                        "success": success_count,
                        "failed": fail_count,
                        "elapsed_seconds": total_time.as_secs_f64()
                    });
                    println!("{}", output);
                } else if output_mode != OutputMode::Quiet {
                    println!();
                    println!("{}", style("Batch Complete:").cyan().bold());
                    println!("  Total:    {} files", results.len());
                    println!("  Success:  {}", style(success_count).green());
                    if fail_count > 0 {
                        println!("  Failed:   {}", style(fail_count).red());
                    }
                    println!("  Time:     {:.1}s", total_time.as_secs_f64());
                }

                if fail_count > 0 {
                    std::process::exit(1);
                }
                return Ok(());
            }
            Err(e) => {
                if output_mode == OutputMode::Json {
                    let error = serde_json::json!({
                        "type": "error",
                        "error": "batch_failed",
                        "message": e.to_string()
                    });
                    println!("{}", error);
                } else {
                    eprintln!("{} {}", style("Batch error:").red().bold(), e);
                }
                std::process::exit(1);
            }
        }
    }

    // Validate input file exists
    if !args.input.exists() {
        if output_mode == OutputMode::Json {
            let error = serde_json::json!({
                "type": "error",
                "error": "input_not_found",
                "message": format!("Input file not found: {}", args.input.display())
            });
            println!("{}", error);
        } else if output_mode != OutputMode::Quiet {
            eprintln!(
                "{} Input file not found: {}",
                style("Error:").red().bold(),
                args.input.display()
            );
        }
        std::process::exit(1);
    }

    // Check output file
    if args.output.exists() && !args.overwrite {
        if output_mode == OutputMode::Json {
            let error = serde_json::json!({
                "type": "error",
                "error": "output_exists",
                "message": format!("Output file already exists: {}", args.output.display())
            });
            println!("{}", error);
        } else if output_mode != OutputMode::Quiet {
            eprintln!(
                "{} Output file already exists: {}",
                style("Error:").red().bold(),
                args.output.display()
            );
            eprintln!("       Use -y to overwrite");
        }
        std::process::exit(1);
    }

    // Print configuration (only in normal/verbose mode)
    if output_mode == OutputMode::Normal || output_mode == OutputMode::Verbose {
        println!();
        println!("{}", style("Configuration:").cyan().bold());
        println!("  Input:        {}", style(args.input.display()).white());
        println!("  Output:       {}", style(args.output.display()).white());
        println!("  Video codec:  {}", style(&args.video_codec).white());
        println!("  Audio codec:  {}", style(&args.audio_codec).white());

        if let Some(bitrate) = args.video_bitrate {
            println!("  Video bitrate: {} kbps", style(bitrate).white());
        }
        if let Some(bitrate) = args.audio_bitrate {
            println!("  Audio bitrate: {} kbps", style(bitrate).white());
        }
        if let Some(threads) = args.threads {
            println!("  Threads:      {}", style(threads).white());
        }

        // Parse and display video filters
        if let Some(ref filter_str) = args.video_filter {
            match parse_video_filters(filter_str) {
                Ok(chain) => {
                    display_filter_chain(&chain, "Video", args.verbose);
                }
                Err(e) => {
                    eprintln!("{} {}", style("Filter error:").red(), e);
                    std::process::exit(1);
                }
            }
        }

        // Parse and display audio filters
        if let Some(ref filter_str) = args.audio_filter {
            match parse_audio_filters(filter_str) {
                Ok(chain) => {
                    display_filter_chain(&chain, "Audio", args.verbose);
                }
                Err(e) => {
                    eprintln!("{} {}", style("Filter error:").red(), e);
                    std::process::exit(1);
                }
            }
        }

        // Show SIMD info
        let simd_caps = transcode_codecs::detect_simd();
        println!(
            "  SIMD:         {}",
            style(simd_caps.best_level()).green()
        );
        println!();
    }

    // Print start message for JSON mode
    if output_mode == OutputMode::Json {
        let start = serde_json::json!({
            "type": "start",
            "input": args.input.to_string_lossy(),
            "output": args.output.to_string_lossy(),
            "video_codec": args.video_codec,
            "audio_codec": args.audio_codec
        });
        println!("{}", start);
    }

    // Create transcoder options
    let mut options = transcode::TranscodeOptions::new()
        .input(&args.input)
        .output(&args.output)
        .overwrite(args.overwrite);

    if let Some(bitrate) = args.video_bitrate {
        options = options.video_bitrate(bitrate as u64 * 1000);
    }

    if let Some(bitrate) = args.audio_bitrate {
        options = options.audio_bitrate(bitrate as u64 * 1000);
    }

    if let Some(threads) = args.threads {
        options = options.threads(threads);
    }

    // Create shared progress state
    let shared_progress = Arc::new(SharedProgress::default());
    let shared_progress_clone = Arc::clone(&shared_progress);

    // Create transcoder with progress callback
    let mut transcoder = transcode::Transcoder::new(options)?;
    transcoder = transcoder.on_progress(move |progress, frames| {
        shared_progress_clone.update(progress, frames);
    });

    // Create progress reporter
    let mut reporter = if !args.no_progress {
        Some(ProgressReporter::new(
            output_mode,
            Arc::clone(&shared_progress),
            args.progress_interval,
        ))
    } else {
        None
    };

    // Run transcoding with progress updates
    let start = Instant::now();

    // Initialize transcoder first to get total duration info
    transcoder.initialize()?;

    // For demonstration, we'll poll progress during run
    // In a real implementation, this would be done via callback
    transcoder.run()?;

    let elapsed = start.elapsed();

    // Update progress one final time and finish
    if let Some(ref mut reporter) = reporter {
        reporter.update(transcoder.stats());
        reporter.finish();
    }

    // Get final stats
    let stats = transcoder.stats();

    // Calculate final statistics
    let avg_fps = if elapsed.as_secs_f64() > 0.0 {
        stats.frames_encoded as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    let final_stats = FinalStats {
        packets_processed: stats.packets_processed,
        frames_decoded: stats.frames_decoded,
        frames_encoded: stats.frames_encoded,
        input_size_bytes: stats.input_size,
        output_size_bytes: stats.output_size,
        compression_ratio: stats.compression_ratio(),
        elapsed_seconds: elapsed.as_secs_f64(),
        average_fps: avg_fps,
        avg_video_bitrate_kbps: stats.avg_video_bitrate as f64 / 1000.0,
        avg_audio_bitrate_kbps: stats.avg_audio_bitrate as f64 / 1000.0,
    };

    // Print results based on output mode
    match output_mode {
        OutputMode::Json => {
            let output = JsonCompleteOutput {
                msg_type: "complete".to_string(),
                success: true,
                stats: final_stats,
            };
            if let Ok(json) = serde_json::to_string(&output) {
                println!("{}", json);
            }
        }
        OutputMode::Quiet => {
            // Just print the output path
            println!("{}", args.output.display());
        }
        OutputMode::Normal => {
            println!("{}", style("Transcoding complete!").green().bold());
            println!();
            println!("{}", style("Statistics:").cyan().bold());
            println!("  Time elapsed:       {:.2}s", elapsed.as_secs_f64());
            println!("  Frames encoded:     {}", stats.frames_encoded);
            println!("  Encoding speed:     {:.1} fps", avg_fps);

            // File sizes
            let input_size_mb = stats.input_size as f64 / 1_000_000.0;
            let output_size_mb = stats.output_size as f64 / 1_000_000.0;
            println!("  Input size:         {:.2} MB", input_size_mb);
            println!("  Output size:        {:.2} MB", output_size_mb);
            println!(
                "  Compression ratio:  {}",
                style(format!("{:.2}x", stats.compression_ratio())).yellow()
            );

            println!();
            println!(
                "{} {}",
                style("Output saved to:").white(),
                style(args.output.display()).green().bold()
            );
        }
        OutputMode::Verbose => {
            println!("{}", style("Transcoding complete!").green().bold());
            println!();
            println!("{}", style("Detailed Statistics:").cyan().bold());
            println!("  {:20} {:.2}s", "Time elapsed:", elapsed.as_secs_f64());
            println!("  {:20} {}", "Packets processed:", stats.packets_processed);
            println!("  {:20} {}", "Frames decoded:", stats.frames_decoded);
            println!("  {:20} {}", "Frames encoded:", stats.frames_encoded);
            println!("  {:20} {:.1} fps", "Encoding speed:", avg_fps);
            println!();

            // File sizes
            println!("{}", style("File Information:").cyan().bold());
            println!("  {:20} {} ({} bytes)", "Input size:", format_size(stats.input_size),
                     stats.input_size);
            println!("  {:20} {} ({} bytes)", "Output size:", format_size(stats.output_size),
                     stats.output_size);
            println!(
                "  {:20} {}",
                "Compression ratio:",
                style(format!("{:.2}x", stats.compression_ratio())).yellow()
            );
            println!();

            // Bitrate info
            println!("{}", style("Bitrate Information:").cyan().bold());
            println!("  {:20} {:.0} kbps", "Avg video bitrate:", stats.avg_video_bitrate as f64 / 1000.0);
            println!("  {:20} {:.0} kbps", "Avg audio bitrate:", stats.avg_audio_bitrate as f64 / 1000.0);
            println!();

            // Speed info
            let speed = if elapsed.as_secs_f64() > 0.0 {
                let duration_secs = stats.duration_processed_us as f64 / 1_000_000.0;
                duration_secs / elapsed.as_secs_f64()
            } else {
                0.0
            };
            println!("{}", style("Performance:").cyan().bold());
            println!("  {:20} {:.2}x realtime", "Speed:", speed);

            // Show SIMD info again
            let simd_caps = transcode_codecs::detect_simd();
            println!("  {:20} {}", "SIMD used:", simd_caps.best_level());
            println!();

            println!(
                "{} {}",
                style("Output saved to:").white(),
                style(args.output.display()).green().bold()
            );
        }
    }

    info!("Transcoding completed successfully");
    debug!("Final stats: {:?}", stats);

    Ok(())
}

/// Parse a video filter string and return the filter chain.
fn parse_video_filters(filter_str: &str) -> anyhow::Result<FilterChain> {
    FilterChain::parse(filter_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse video filter: {}", e))
}

/// Parse an audio filter string and return the filter chain.
fn parse_audio_filters(filter_str: &str) -> anyhow::Result<FilterChain> {
    FilterChain::parse(filter_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse audio filter: {}", e))
}

/// Display filter chain information.
fn display_filter_chain(chain: &FilterChain, filter_type: &str, verbose: bool) {
    if verbose {
        println!("  {} filters:", filter_type);
        for (i, filter) in chain.filters.iter().enumerate() {
            // Collect all params - positional are stored as "0", "1", etc.
            let mut positional: Vec<String> = Vec::new();
            let mut named: Vec<String> = Vec::new();

            for (k, v) in &filter.params {
                if k.parse::<usize>().is_ok() {
                    positional.push(v.clone());
                } else {
                    named.push(format!("{}={}", k, v));
                }
            }

            let all_params = if positional.is_empty() && named.is_empty() {
                String::new()
            } else if named.is_empty() {
                positional.join(", ")
            } else if positional.is_empty() {
                named.join(", ")
            } else {
                format!("{}, {}", positional.join(", "), named.join(", "))
            };
            println!("    {}. {} ({})", i + 1, style(&filter.name).yellow(), all_params);
        }
    } else {
        let filter_names: Vec<&str> = chain.filters.iter()
            .map(|f| f.name.as_str())
            .collect();
        println!("  {} filters:  {}", filter_type, style(filter_names.join(" -> ")).white());
    }
}

/// Batch processing result.
#[derive(Debug)]
#[allow(dead_code)]
struct BatchResult {
    input: PathBuf,
    output: PathBuf,
    success: bool,
    elapsed: Duration,
    error: Option<String>,
}

/// Process files in batch mode.
fn process_batch(
    args: &Args,
    pattern: &str,
    output_dir: &PathBuf,
) -> anyhow::Result<Vec<BatchResult>> {
    let mut results = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();

    // Collect files matching pattern
    for entry in glob(pattern)? {
        match entry {
            Ok(path) => {
                if path.is_file() {
                    files.push(path);
                }
            }
            Err(e) => {
                warn!("Error matching pattern: {}", e);
            }
        }
    }

    if files.is_empty() {
        return Err(anyhow::anyhow!("No files found matching pattern: {}", pattern));
    }

    println!("{} Found {} files to process", style("Batch:").cyan().bold(), files.len());

    // Create output directory if needed
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)?;
    }

    // Get output extension from args
    let output_ext = args.output.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("mp4");

    // Process files
    for input in files {
        let file_stem = input.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let output = output_dir.join(format!("{}_transcoded.{}", file_stem, output_ext));

        let start = Instant::now();

        // Create options for this file
        let mut options = transcode::TranscodeOptions::new()
            .input(&input)
            .output(&output)
            .overwrite(args.overwrite);

        if let Some(bitrate) = args.video_bitrate {
            options = options.video_bitrate(bitrate as u64 * 1000);
        }
        if let Some(bitrate) = args.audio_bitrate {
            options = options.audio_bitrate(bitrate as u64 * 1000);
        }
        if let Some(threads) = args.threads {
            options = options.threads(threads);
        }

        // Run transcoding
        let result = (|| -> anyhow::Result<()> {
            let mut transcoder = transcode::Transcoder::new(options)?;
            transcoder.initialize()?;
            transcoder.run()?;
            Ok(())
        })();

        let elapsed = start.elapsed();

        match result {
            Ok(()) => {
                println!(
                    "  {} {} -> {} ({:.1}s)",
                    style("✓").green(),
                    input.display(),
                    output.display(),
                    elapsed.as_secs_f64()
                );
                results.push(BatchResult {
                    input,
                    output,
                    success: true,
                    elapsed,
                    error: None,
                });
            }
            Err(e) => {
                println!(
                    "  {} {} - {}",
                    style("✗").red(),
                    input.display(),
                    e
                );
                results.push(BatchResult {
                    input,
                    output,
                    success: false,
                    elapsed,
                    error: Some(e.to_string()),
                });
            }
        }
    }

    Ok(results)
}

fn print_header() {
    println!();
    println!(
        "{}",
        style("+---------------------------------------------------------+").cyan()
    );
    println!(
        "{}  {}  {}",
        style("|").cyan(),
        style("TRANSCODE").cyan().bold(),
        style("                                           |").cyan()
    );
    println!(
        "{}  {}  {}",
        style("|").cyan(),
        style("Memory-safe, high-performance media transcoding").white(),
        style("|").cyan()
    );
    println!(
        "{}",
        style("+---------------------------------------------------------+").cyan()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== format_size tests =====

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(10240), "10.00 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(5 * 1024 * 1024), "5.00 MB");
        assert_eq!(format_size(100 * 1024 * 1024), "100.00 MB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    // ===== format_duration tests =====

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(0.0), "0s");
        // Rust uses banker's rounding (round half to even): 30.5 -> 30, 31.5 -> 32
        assert_eq!(format_duration(30.5), "30s");
        assert_eq!(format_duration(30.6), "31s");
        assert_eq!(format_duration(59.9), "60s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(60.0), "1m0s");
        assert_eq!(format_duration(90.0), "1m30s");
        assert_eq!(format_duration(3599.0), "59m59s");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3600.0), "1h0m0s");
        assert_eq!(format_duration(7200.0), "2h0m0s");
        assert_eq!(format_duration(3661.0), "1h1m1s");
        assert_eq!(format_duration(7384.0), "2h3m4s");
    }

    // ===== OutputMode tests =====

    #[test]
    fn test_output_mode_default() {
        let args = Args {
            input: PathBuf::from("input.mp4"),
            output: PathBuf::from("output.mp4"),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: None,
            audio_bitrate: None,
            threads: None,
            overwrite: false,
            no_progress: false,
            verbose: false,
            quiet: false,
            json: false,
            progress_interval: 500,
            video_filter: None,
            audio_filter: None,
            batch: None,
            batch_output_dir: None,
            jobs: 1,
        };
        assert_eq!(args.output_mode(), OutputMode::Normal);
    }

    #[test]
    fn test_output_mode_json() {
        let args = Args {
            input: PathBuf::from("input.mp4"),
            output: PathBuf::from("output.mp4"),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: None,
            audio_bitrate: None,
            threads: None,
            overwrite: false,
            no_progress: false,
            verbose: false,
            quiet: false,
            json: true,
            progress_interval: 500,
            video_filter: None,
            audio_filter: None,
            batch: None,
            batch_output_dir: None,
            jobs: 1,
        };
        assert_eq!(args.output_mode(), OutputMode::Json);
    }

    #[test]
    fn test_output_mode_quiet() {
        let args = Args {
            input: PathBuf::from("input.mp4"),
            output: PathBuf::from("output.mp4"),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: None,
            audio_bitrate: None,
            threads: None,
            overwrite: false,
            no_progress: false,
            verbose: false,
            quiet: true,
            json: false,
            progress_interval: 500,
            video_filter: None,
            audio_filter: None,
            batch: None,
            batch_output_dir: None,
            jobs: 1,
        };
        assert_eq!(args.output_mode(), OutputMode::Quiet);
    }

    #[test]
    fn test_output_mode_verbose() {
        let args = Args {
            input: PathBuf::from("input.mp4"),
            output: PathBuf::from("output.mp4"),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: None,
            audio_bitrate: None,
            threads: None,
            overwrite: false,
            no_progress: false,
            verbose: true,
            quiet: false,
            json: false,
            progress_interval: 500,
            video_filter: None,
            audio_filter: None,
            batch: None,
            batch_output_dir: None,
            jobs: 1,
        };
        assert_eq!(args.output_mode(), OutputMode::Verbose);
    }

    // ===== ProgressStats tests =====

    #[test]
    fn test_progress_stats_serialization() {
        let stats = ProgressStats {
            current_frame: 100,
            total_frames: Some(1000),
            percentage: 10.0,
            fps: 30.0,
            eta_seconds: Some(30.0),
            bitrate_kbps: 5000.0,
            output_size_bytes: 1024 * 1024,
            elapsed_seconds: 3.33,
            duration_processed_seconds: 3.33,
            total_duration_seconds: Some(33.3),
            speed: 1.0,
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"current_frame\":100"));
        assert!(json.contains("\"percentage\":10.0"));
        assert!(json.contains("\"fps\":30.0"));
    }

    // ===== FinalStats tests =====

    #[test]
    fn test_final_stats_creation() {
        let stats = FinalStats {
            packets_processed: 1000,
            frames_decoded: 900,
            frames_encoded: 900,
            input_size_bytes: 10 * 1024 * 1024,
            output_size_bytes: 5 * 1024 * 1024,
            compression_ratio: 2.0,
            elapsed_seconds: 30.0,
            average_fps: 30.0,
            avg_video_bitrate_kbps: 4000.0,
            avg_audio_bitrate_kbps: 128.0,
        };
        assert_eq!(stats.compression_ratio, 2.0);
        assert_eq!(stats.frames_encoded, 900);
    }

    // ===== SharedProgress tests =====

    #[test]
    fn test_shared_progress_update() {
        let progress = SharedProgress::default();
        assert_eq!(progress.current_frame.load(Ordering::Relaxed), 0);
        assert!(!progress.total_frames_known.load(Ordering::Relaxed));

        progress.update(50.0, 500);
        assert_eq!(progress.current_frame.load(Ordering::Relaxed), 500);
        assert!(progress.total_frames_known.load(Ordering::Relaxed));
        assert_eq!(progress.total_frames.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn test_shared_progress_progress_calculation() {
        let progress = SharedProgress::default();
        assert_eq!(progress.progress(), 0.0);

        progress.update(25.0, 250);
        // After update with 25% and 250 frames, total should be 1000
        // Progress should be 250/1000 = 25%
        let p = progress.progress();
        assert!((p - 25.0).abs() < 0.1);
    }

    // ===== format_progress_message tests =====

    #[test]
    fn test_format_progress_message_normal() {
        let stats = ProgressStats {
            current_frame: 100,
            total_frames: Some(1000),
            percentage: 10.0,
            fps: 30.0,
            eta_seconds: Some(30.0),
            bitrate_kbps: 5000.0,
            output_size_bytes: 1024 * 1024,
            elapsed_seconds: 3.33,
            duration_processed_seconds: 3.33,
            total_duration_seconds: Some(33.3),
            speed: 1.0,
        };
        let msg = format_progress_message(&stats, false);
        assert!(msg.contains("30.0 fps"));
        assert!(msg.contains("5000 kbps"));
        assert!(msg.contains("1.00 MB"));
    }

    #[test]
    fn test_format_progress_message_verbose() {
        let stats = ProgressStats {
            current_frame: 100,
            total_frames: Some(1000),
            percentage: 10.0,
            fps: 30.0,
            eta_seconds: Some(30.0),
            bitrate_kbps: 5000.0,
            output_size_bytes: 1024 * 1024,
            elapsed_seconds: 3.33,
            duration_processed_seconds: 3.33,
            total_duration_seconds: Some(33.3),
            speed: 1.0,
        };
        let msg = format_progress_message(&stats, true);
        assert!(msg.contains("Frame: 100/1000"));
        assert!(msg.contains("30.0 fps"));
        assert!(msg.contains("1.0x speed"));
        assert!(msg.contains("ETA:"));
    }

    // ===== BatchResult tests =====

    #[test]
    fn test_batch_result_success() {
        let result = BatchResult {
            input: PathBuf::from("input.mp4"),
            output: PathBuf::from("output.mp4"),
            success: true,
            elapsed: Duration::from_secs(10),
            error: None,
        };
        assert!(result.success);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_batch_result_failure() {
        let result = BatchResult {
            input: PathBuf::from("input.mp4"),
            output: PathBuf::from("output.mp4"),
            success: false,
            elapsed: Duration::from_secs(1),
            error: Some("File not found".to_string()),
        };
        assert!(!result.success);
        assert_eq!(result.error.as_deref(), Some("File not found"));
    }
}
