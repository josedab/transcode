//! Basic distributed coordinator example.
//!
//! This example demonstrates how to set up a coordinator and split
//! a video into segments for distributed processing.
//!
//! Run with:
//! ```sh
//! cargo run --example basic_coordinator -p transcode-distributed
//! ```

use std::time::Duration;
use transcode_distributed::{
    CoordinatorConfig, SegmentSplitter,
    WorkerCapabilities, WorkerInfo, DistributedStats,
};

fn main() -> transcode_distributed::Result<()> {
    // Create a segment splitter (10 second segments)
    let splitter = SegmentSplitter::new(10.0);

    // Simulate a 65-second video
    let video_duration = 65.0;
    let segments = splitter.split("/path/to/input.mp4", video_duration);

    println!("Video duration: {} seconds", video_duration);
    println!("Segment count: {}", segments.len());
    println!();

    // Print segment information
    for segment in &segments {
        println!(
            "Segment {}/{}: {:.1}s - {:.1}s (duration: {:.1}s)",
            segment.index + 1,
            segment.total_segments,
            segment.start_time,
            segment.end_time,
            segment.duration()
        );
    }
    println!();

    // Create coordinator configuration
    let config = CoordinatorConfig {
        segment_duration: 10.0,
        max_concurrent_jobs: 10,
        heartbeat_timeout: Duration::from_secs(30),
        retry_delay: Duration::from_secs(5),
        health_check_interval: Duration::from_secs(60),
    };

    println!("Coordinator config:");
    println!("  Segment duration: {}s", config.segment_duration);
    println!("  Max concurrent jobs: {}", config.max_concurrent_jobs);
    println!("  Heartbeat timeout: {:?}", config.heartbeat_timeout);
    println!();

    // Example worker capabilities
    let worker = WorkerInfo::new(
        "worker-1".to_string(),
        "192.168.1.100:8080".to_string(),
        WorkerCapabilities {
            max_concurrent: 2,
            codecs: vec!["h264".into(), "h265".into(), "av1".into()],
            has_gpu: true,
            gpu_model: Some("NVIDIA RTX 4090".into()),
            cpu_cores: 8,
            memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB
        },
    );

    println!("Worker: {}", worker.name);
    println!("  ID: {}", worker.id);
    println!("  Address: {}", worker.address);
    println!("  GPU: {} ({:?})", worker.capabilities.has_gpu, worker.capabilities.gpu_model);
    println!("  CPU cores: {}", worker.capabilities.cpu_cores);
    println!("  Supported codecs: {:?}", worker.capabilities.codecs);
    println!();

    // Example statistics
    let stats = DistributedStats {
        total_jobs: 100,
        completed_jobs: 95,
        failed_jobs: 5,
        total_tasks: 1000,
        completed_tasks: 950,
        failed_tasks: 50,
        active_workers: 4,
        total_processing_time: 28500.0,
    };

    println!("System statistics:");
    println!("  Success rate: {:.1}%", stats.success_rate() * 100.0);
    println!("  Avg processing time: {:.1}s per job", stats.avg_processing_time());
    println!("  Active workers: {}", stats.active_workers);

    Ok(())
}
